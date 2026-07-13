//! MEMW_R (Memory Write/Read -- Register) table.
//!
//! Ultra-slim fast path for register accesses. Registers are always 2 words
//! (DWordWL), always aligned, and `is_register=1`, so this table strips out
//! all memory-specific columns (address decomposition, alignment mask, width
//! flags, per-byte old_timestamps).
//!
//! ## Timestamp ordering: IS_HALF instead of LT
//!
//! The general MEMW table proves `old_timestamp < timestamp` by routing through
//! the LT table, which requires extra LT trace rows and bus interactions.
//! MEMW_R instead checks `IS_HALF[timestamp[0] - old_timestamp[0] - 1]`,
//! which proves the delta is in `[1, 2^16]` in a single lookup. This is safe
//! because registers are accessed very frequently — their timestamp deltas are
//! almost always small — and the routing predicate (`is_register_op`) enforces
//! the delta fits before admitting an op into this table.
//!
//! ## Column layout (10 columns)
//!
//! - `ADDRESS`:          Byte  (register index 0-255: x0-x31, plus x254/x255)
//! - `TIMESTAMP_0`:      Word  (low 32 bits)
//! - `TIMESTAMP_1`:      Word  (high 32 bits)
//! - `VAL_0`:            Word  (low 32 bits of register value)
//! - `VAL_1`:            Word  (high 32 bits of register value)
//! - `OLD_0`:            Word  (low 32 bits of previous value)
//! - `OLD_1`:            Word  (high 32 bits of previous value)
//! - `OLD_TIMESTAMP_LO`: Word  (low 32 bits of old timestamp; upper limb = TIMESTAMP_1)
//! - `MU_READ`:          Bit
//! - `MU_WRITE`:         Bit
//!
//! ## Virtual
//!
//! - `old_timestamp = [OLD_TIMESTAMP_LO, TIMESTAMP_1]` (shares upper limb!)
//! - `mu_sum = MU_READ + MU_WRITE`
//!
//! ## Bus Interactions (7)
//! - 1 IS_HALFWORD[timestamp_0 - old_timestamp_lo - 1]
//! - 4 Memory bus tokens (read-old + write-new, per word)
//! - 2 MEMW output interactions (read + write, from CPU)

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

use super::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
use super::memw::MemwOperation;
use super::types::{BusId, GoldilocksExtension, GoldilocksField, VmTable};
use crate::constraints::templates::emit_is_bit;

// =========================================================================
// Column indices (10 columns)
// =========================================================================

pub mod cols {
    /// Register index (0-255: x0-x31, plus x254/x255). CPU sends base_address = 2*reg_index.
    pub const ADDRESS: usize = 0;

    /// Timestamp low 32 bits
    pub const TIMESTAMP_0: usize = 1;
    /// Timestamp high 32 bits
    pub const TIMESTAMP_1: usize = 2;

    /// Register value low 32 bits
    pub const VAL_0: usize = 3;
    /// Register value high 32 bits
    pub const VAL_1: usize = 4;

    /// Previous value low 32 bits
    pub const OLD_0: usize = 5;
    /// Previous value high 32 bits
    pub const OLD_1: usize = 6;

    /// Old timestamp low 32 bits (upper limb shared with TIMESTAMP_1)
    pub const OLD_TIMESTAMP_LO: usize = 7;

    /// Read multiplicity
    pub const MU_READ: usize = 8;
    /// Write multiplicity
    pub const MU_WRITE: usize = 9;

    pub const NUM_COLUMNS: usize = 10;
}

// =========================================================================
// Trace generation
// =========================================================================

/// Compact, already-decomposed record for one MEMW_R (register fast-path) access.
///
/// This is the "direct-to-column" carrier: it holds exactly the fields the MEMW_R
/// column fill ([`generate_memw_register_trace_from_rows`]) and its IS_HALFWORD
/// bitwise collector ([`collect_bitwise_from_memw_register`]) need, and nothing
/// else. It replaces the full `MemwOperation` (~152 B after the `[u32; 8]`
/// value/old shrink, but still 8-element arrays) for register accesses — the
/// largest table by rows — so the walk never materializes a `MemwOperation` for
/// the register fast path.
///
/// Field domains mirror `MemwOperation`'s:
/// - `address`   = `base_address / 2` (the register index 0..=255; ADDRESS column,
///   and `2*ADDRESS` on the memory/MEMW buses)
/// - `val0/val1` = `value[0]`/`value[1]` (the 32-bit register halves)
/// - `old0/old1` = `old[0]`/`old[1]`
/// - `old_ts_lo` = `old_timestamp[0] & 0xFFFF_FFFF` (the two words share old_timestamp,
///   enforced by `is_register_op`; the upper limb is TIMESTAMP_1 = timestamp>>32)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RegRow {
    /// Register index 0..=255 (`base_address / 2`); u16 keeps the struct at
    /// 32 bytes — it is the largest persisted array of the walk.
    address: u16,
    timestamp: u64,
    val0: u32,
    val1: u32,
    old0: u32,
    old1: u32,
    old_ts_lo: u32,
    is_read: bool,
}

impl RegRow {
    /// Build a `RegRow` from pre-decomposed register-access fields.
    ///
    /// `reg_addr` is `2 * reg_index` as sent by the CPU; `old_ts` is the (shared)
    /// old_timestamp of both register words. This is the ONLY place the MEMW_R
    /// row encoding (halved address, masked `old_ts_lo`) is defined —
    /// [`Self::from_memw`] delegates here, so the walk fast path and the
    /// `MemwOperation` paths cannot drift.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        reg_addr: u64,
        timestamp: u64,
        val0: u32,
        val1: u32,
        old0: u32,
        old1: u32,
        old_ts: u64,
        is_read: bool,
    ) -> Self {
        debug_assert_eq!(
            reg_addr % 2,
            0,
            "register base_address must be even (got {reg_addr})"
        );
        debug_assert!(
            reg_addr / 2 <= u16::MAX as u64,
            "register index exceeds u16 (got base_address {reg_addr})"
        );
        RegRow {
            address: (reg_addr / 2) as u16,
            timestamp,
            val0,
            val1,
            old0,
            old1,
            old_ts_lo: (old_ts & 0xFFFF_FFFF) as u32,
            is_read,
        }
    }

    /// Marshal to the SoA the on-device MEMW_R fill (`memw_register_fill`)
    /// consumes: `(reg_addr = 2*address, timestamp, value, is_read, old_value,
    /// old_ts)`. The old_timestamp upper limb is shared with TIMESTAMP_1
    /// (`timestamp >> 32`), matching the column encoding.
    #[cfg(feature = "cuda")]
    pub(crate) fn fill_soa(&self) -> (u32, u64, u64, u8, u64, u64) {
        let value = (self.val0 as u64) | ((self.val1 as u64) << 32);
        let old_value = (self.old0 as u64) | ((self.old1 as u64) << 32);
        let old_ts = (self.old_ts_lo as u64) | ((self.timestamp >> 32) << 32);
        (
            2 * self.address as u32,
            self.timestamp,
            value,
            self.is_read as u8,
            old_value,
            old_ts,
        )
    }

    /// Build a `RegRow` from a fully-formed register `MemwOperation`. Used on the
    /// precompile / commit / keccak / halt paths, which construct a `MemwOperation`
    /// first and only convert to the compact row once the op is known to route to
    /// MEMW_R.
    ///
    /// Only valid for ops for which `is_register_op` is true (width==2, atomic
    /// old_timestamp).
    #[inline]
    pub(crate) fn from_memw(op: &MemwOperation) -> Self {
        // Both register words must have been last accessed at the same timestamp.
        // MEMW_R stores a single old_timestamp_lo and shares TIMESTAMP_1 as the
        // upper limb, so if the two words differ, the wrong token would be sent
        // to the memory bus. The routing predicate enforces this before dispatch.
        debug_assert_eq!(
            op.old_timestamp[0], op.old_timestamp[1],
            "register words must share old_timestamp ({} != {})",
            op.old_timestamp[0], op.old_timestamp[1]
        );
        Self::new(
            op.base_address,
            op.timestamp,
            op.value[0],
            op.value[1],
            op.old[0],
            op.old[1],
            op.old_timestamp[0],
            op.is_read,
        )
    }
}

/// Generates the MEMW_R trace table from register operations.
///
/// Thin wrapper over [`generate_memw_register_trace_from_rows`] (via
/// [`RegRow::from_memw`]) so there is exactly one MEMW_R column-write sequence.
///
/// Test-only: production code fills MEMW_R directly from [`RegRow`]s, so the walk
/// never routes through this `MemwOperation`-based entry point.
#[cfg(test)]
pub(crate) fn generate_memw_register_trace(
    operations: &[MemwOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let rows: Vec<RegRow> = operations.iter().map(RegRow::from_memw).collect();
    generate_memw_register_trace_from_rows(&rows)
}

/// The MEMW_R column fill from compact [`RegRow`]s. This is the single source of
/// truth for the MEMW_R trace layout; both the walk's direct fast path and the
/// `MemwOperation`-based `generate_memw_register_trace` test wrapper land here.
pub(crate) fn generate_memw_register_trace_from_rows(
    rows: &[RegRow],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = rows.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, r) in rows.iter().enumerate() {
        // ADDRESS = base_address / 2 (already divided in RegRow).
        table.set_u64(row_idx, cols::ADDRESS, r.address as u64);
        // Timestamp split into lo/hi 32-bit words.
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, r.timestamp);
        // Value: registers are DWordWL = 2 words.
        table.set_u64(row_idx, cols::VAL_0, r.val0 as u64);
        table.set_u64(row_idx, cols::VAL_1, r.val1 as u64);
        // Old value.
        table.set_u64(row_idx, cols::OLD_0, r.old0 as u64);
        table.set_u64(row_idx, cols::OLD_1, r.old1 as u64);
        // Old timestamp low (upper limb shared with TIMESTAMP_1).
        table.set_u64(row_idx, cols::OLD_TIMESTAMP_LO, r.old_ts_lo as u64);
        // Multiplicity.
        table.set_bool(row_idx, cols::MU_READ, r.is_read);
        table.set_bool(row_idx, cols::MU_WRITE, !r.is_read);
    }

    trace
}

/// The single IS_HALFWORD lookup a MEMW_R access sends: proves the timestamp delta
/// `ts_lo - old_ts_lo` is in [1, 2^16] by decomposing `ts_lo - old_ts_lo - 1` into
/// two bytes.
///
/// Must stay in lockstep with the IS_HALFWORD send in [`bus_interactions`]: the
/// lookup counted here has to be exactly the lookup each MEMW_R row sends, or the
/// BITWISE bus goes unbalanced.
#[inline]
fn memw_register_is_half_lookup(ts_lo: u32, old_ts_lo: u32) -> BitwiseOperation {
    debug_assert!(
        ts_lo > old_ts_lo,
        "ts_lo must exceed old_ts_lo (enforced by the MEMW_R routing predicate)"
    );
    let diff_minus_1 = (ts_lo - old_ts_lo - 1) as u16;
    BitwiseOperation::halfword(
        BitwiseOperationType::IsHalf,
        (diff_minus_1 & 0xFF) as u8,
        (diff_minus_1 >> 8) as u8,
    )
}

/// IS_HALFWORD bitwise lookups for MEMW_R, bumped straight into the histogram
/// via the shared [`memw_register_is_half_lookup`] helper (the same lookup the
/// MEMW_R trace fill uses), one per row. No intermediate op vector: register
/// rows number in the tens of millions and the histogram is the only consumer.
pub(crate) fn collect_bitwise_from_memw_register(rows: &[RegRow], hist: &mut BitwiseHistogram) {
    for r in rows {
        hist.bump(memw_register_is_half_lookup(
            (r.timestamp & 0xFFFF_FFFF) as u32,
            r.old_ts_lo,
        ));
    }
}

// =========================================================================
// Bus interactions (7 total)
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(7);

    let mu_sum = Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE);

    // -------------------------------------------------------------------------
    // IS_HALFWORD[timestamp_0 - old_timestamp_lo - 1] with mu_sum
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::IsHalfword,
        mu_sum.clone(),
        vec![BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::TIMESTAMP_0,
            },
            LinearTerm::Column {
                coefficient: -1,
                column: cols::OLD_TIMESTAMP_LO,
            },
            LinearTerm::Constant(-1),
        ])],
    ));

    // -------------------------------------------------------------------------
    // Memory bus read-old (sender, for i=0,1)
    // memory[is_register=1, addr_lo=2*ADDRESS+i, addr_hi=0,
    //        OLD_TIMESTAMP_LO, TIMESTAMP_1, OLD[i]]
    // -------------------------------------------------------------------------
    for i in 0..2 {
        let addr_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 2,
                column: cols::ADDRESS,
            },
            LinearTerm::Constant(i as i64),
        ]);

        interactions.push(BusInteraction::sender(
            BusId::Memory,
            mu_sum.clone(),
            vec![
                BusValue::constant(1),
                addr_lo,
                BusValue::constant(0),
                BusValue::Packed {
                    start_column: cols::OLD_TIMESTAMP_LO,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: if i == 0 { cols::OLD_0 } else { cols::OLD_1 },
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // -------------------------------------------------------------------------
    // Memory bus write-new (receiver, for i=0,1)
    // memory[is_register=1, addr_lo=2*ADDRESS+i, addr_hi=0,
    //        TIMESTAMP_0, TIMESTAMP_1, VAL[i]]
    // -------------------------------------------------------------------------
    for i in 0..2 {
        let addr_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 2,
                column: cols::ADDRESS,
            },
            LinearTerm::Constant(i as i64),
        ]);

        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            mu_sum.clone(),
            vec![
                BusValue::constant(1),
                addr_lo,
                BusValue::constant(0),
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: if i == 0 { cols::VAL_0 } else { cols::VAL_1 },
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // -------------------------------------------------------------------------
    // CO24: MEMW read receiver (from CPU M1/M3 sender)
    // -------------------------------------------------------------------------
    let addr_lo_linear = BusValue::linear(vec![LinearTerm::Column {
        coefficient: 2,
        column: cols::ADDRESS,
    }]);

    interactions.push(BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(cols::MU_READ),
        vec![
            // old[0..8]
            BusValue::Packed {
                start_column: cols::OLD_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD_1,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // is_register = 1
            BusValue::constant(1),
            // base_address = [2*ADDRESS, 0]
            addr_lo_linear.clone(),
            BusValue::constant(0),
            // value[0..8]
            BusValue::Packed {
                start_column: cols::VAL_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VAL_1,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // timestamp
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            // write flags: write2=1, write4=0, write8=0 (registers are always 2 words)
            BusValue::constant(1),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    ));

    // -------------------------------------------------------------------------
    // CO25: MEMW write receiver (from CPU M5 sender — register write to rd)
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(cols::MU_WRITE),
        vec![
            // is_register = 1
            BusValue::constant(1),
            // base_address = [2*ADDRESS, 0]
            addr_lo_linear,
            BusValue::constant(0),
            // value[0..8]
            BusValue::Packed {
                start_column: cols::VAL_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VAL_1,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // timestamp
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            // write flags: write2=1, write4=0, write8=0
            BusValue::constant(1),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    ));

    interactions
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// The MEMW_R table's 3 transition constraints as a single [`ConstraintSet`]:
/// - idx 0,1: `IS_BIT` on `μ_read`, `μ_write`;
/// - idx 2:   `IS_BIT<μ_sum>` with `μ_sum = μ_read + μ_write`.
pub struct MemwRegisterConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemwRegisterConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0,1: IS_BIT<μ_read>, IS_BIT<μ_write>
        emit_is_bit(b, 0, cols::MU_READ, None);
        emit_is_bit(b, 1, cols::MU_WRITE, None);

        // idx 2: IS_BIT<μ_sum> = μ_sum * (1 - μ_sum), μ_sum = μ_read + μ_write
        let one = b.one();
        let mu_sum = b.main(0, cols::MU_READ) + b.main(0, cols::MU_WRITE);
        b.emit_base(2, mu_sum.clone() * (one - mu_sum));
    }
}

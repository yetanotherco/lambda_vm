//! MEMW_A (Memory Write/Read — Aligned) table.
//!
//! Fast path for aligned memory/register accesses where all bytes share the
//! same old_timestamp. Most operations (aligned memory + all register accesses)
//! route here instead of the heavier MEMW table.
//!
//! ## The timestamp order: two forms ([`MemwAlignedOrder`])
//!
//! Every access proves its old timestamp below its timestamp. How is a block
//! format constant (`block_whir::BlockFormat::memw_aligned`):
//!
//! - [`MemwAlignedOrder::Lt`]: one `ALU[old_timestamp, timestamp, LT, 1, 0]`
//!   lookup a row, answered by a row of the LT table. The old timestamp is two
//!   columns.
//! - [`MemwAlignedOrder::Limbs`]: the step's two 16-bit limbs are the row's
//!   columns 25 and 26 ([`cols::STEP_0`], [`cols::STEP_1`]), each an `IS_HALF`
//!   lookup weighted `μ_read + μ_write`. The old timestamp is no column: its low
//!   word is `TIMESTAMP_0 − 1 − STEP_0 − 2^16·STEP_1` and its high word the
//!   row's own `TIMESTAMP_1`, in all eight old tokens. That proves the step
//!   `timestamp − old_timestamp` in `1..=2^32` on one high word, which is
//!   exactly what the LT lookup accepts on the timestamps the clock makes (a
//!   high word of 0); an access whose two timestamps differ in their high
//!   words is routed to MEMW. MEMW_R's rule (`memw_register.rs`) at 32 bits.
//!
//! ## Column layout (29 columns)
//!
//! - `is_register`: Bit
//! - `base_address[3]`: DWordWHH
//!   - `base_address[0]`: Half (low 16 bits)
//!   - `base_address[1]`: Half (mid 16 bits)
//!   - `base_address[2]`: Word (high 32 bits)
//! - `value[8]`: BaseField[8]
//! - `timestamp`: DWordWL (2 cols)
//! - `write2/4/8`: Bit (access width flags)
//! - `old[8]`: BaseField[8]
//! - columns 25, 26: `old_timestamp`, DWordWL (`Lt`: single, not 8!), or the
//!   step's limbs `d0`, `d1` (`Limbs`: Half each)
//! - `mu_read`, `mu_write`: multiplicity columns
//!
//! ## Bus Interactions (20 under `Lt`, 21 under `Limbs`)
//! - 1 IS_HALF[base_address[0] + mask] (range check: address span fits in 16 bits)
//! - `Lt`: 1 ALU[old_timestamp, timestamp, opsel(LT), 1, 0] → asserts old_ts < ts
//! - `Limbs`: 2 IS_HALF[d0], IS_HALF[d1] → asserts the step in 1..=2^32
//! - 16 Memory bus tokens
//! - 2 MEMW output interactions (read + write)
//!
//! ## Constraints (8 total, the same in both forms)
//! - IS_BIT<μ_sum> (1)
//! - w2 => μ_sum (1)
//! - IS_BIT<μ_read> (1)
//! - IS_BIT<μ_write> (1)
//! - IS_BIT<write2>, IS_BIT<write4>, IS_BIT<write8> (3)
//! - IS_BIT<w2> (width sum is a bit) (1)
//!
//! ## Assumptions (caller's responsibility, not enforced here)
//! - IS_HALF[base_address[i]] for i ∈ [0, 1]
//! - IS_WORD[base_address[2]]
//! - `Limbs`: every sender on the MEMW bus sends a timestamp whose high word is
//!   a 32-bit value and every timestamp the Memory bus carries is below 2^32 (the
//!   clock: the CPU's high word is the constant 0 and its rows are bounded by
//!   the block verifier), so a step in `1..=2^32` from a produced token is a
//!   later integer.

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

use super::gpack::{TraceForm, WidthHint, generate_main};
use super::memw::MemwOperation;
use super::types::{BusId, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use crate::constraints::templates::emit_is_bit;

/// Maximum number of rows per MEMW_A table chunk.
pub const MAX_ROWS: usize = super::max_rows::MEMW_A;

/// How MEMW_A proves an access's old timestamp below its timestamp (the
/// module docs): a constant of the proof format, which the prover and the
/// verifier of one proof take alike and nothing reads from a proof.
/// The default is the per-table STARK's form, [`Self::Lt`]; the block's
/// format names [`Self::Limbs`] (`block_whir::BlockFormat::memw_aligned`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MemwAlignedOrder {
    /// One `ALU[old_timestamp, timestamp, LT, 1, 0]` lookup a row, answered by
    /// a row of LT; the old timestamp in two columns.
    #[default]
    Lt,
    /// The step's two 16-bit limbs in the row, each an `IS_HALF` lookup; the
    /// old timestamp's low word derived from them and its high word the row's
    /// own `TIMESTAMP_1`. An access whose timestamps differ in their high words
    /// is never a MEMW_A row ([`step_fits_limbs`]).
    Limbs,
}

/// Whether an access from `old_timestamp` to `timestamp` can be a
/// [`MemwAlignedOrder::Limbs`] row: the two share their high word and the old
/// one is the earlier, so the step `timestamp − old_timestamp` is in
/// `1..2^32` and `step − 1` is two 16-bit limbs. Anything else is MEMW's,
/// which keeps the full comparison: refused here, never wrapped.
#[inline]
pub(crate) fn step_fits_limbs(old_timestamp: u64, timestamp: u64) -> bool {
    old_timestamp >> 32 == timestamp >> 32 && old_timestamp < timestamp
}

/// The limbs `(d0, d1)` of `timestamp − old_timestamp − 1` for an access
/// [`step_fits_limbs`] admits: `d0 + 2^16·d1`, each below 2^16.
#[inline]
pub(crate) fn step_limbs(old_timestamp: u64, timestamp: u64) -> (u16, u16) {
    debug_assert!(
        step_fits_limbs(old_timestamp, timestamp),
        "a MEMW_A limb row's step must fit 32 bits on one high word: \
         {old_timestamp:#x} → {timestamp:#x}"
    );
    let d = timestamp.wrapping_sub(old_timestamp).wrapping_sub(1);
    (d as u16, (d >> 16) as u16)
}

// =========================================================================
// Column indices (29 columns)
// =========================================================================

pub mod cols {
    pub const IS_REGISTER: usize = 0;

    /// base_address: DWordWHH (3 columns)
    /// base_address[0] = low half (bits 0-15)
    /// base_address[1] = mid half (bits 16-31)
    /// base_address[2] = high word (bits 32-63)
    pub const BASE_ADDRESS: [usize; 3] = [1, 2, 3];

    pub const VALUE: [usize; 8] = [4, 5, 6, 7, 8, 9, 10, 11];

    pub const TIMESTAMP_0: usize = 12;
    pub const TIMESTAMP_1: usize = 13;

    pub const WRITE2: usize = 14;
    pub const WRITE4: usize = 15;
    pub const WRITE8: usize = 16;

    pub const OLD: [usize; 8] = [17, 18, 19, 20, 21, 22, 23, 24];

    /// Single old_timestamp (shared across all bytes, since they're aligned):
    /// [`MemwAlignedOrder::Lt`](super::MemwAlignedOrder::Lt)'s two columns.
    pub const OLD_TIMESTAMP_0: usize = 25;
    pub const OLD_TIMESTAMP_1: usize = 26;

    /// [`MemwAlignedOrder::Limbs`](super::MemwAlignedOrder::Limbs): the same two
    /// columns hold the step's limbs, `timestamp − old_timestamp − 1 = STEP_0 +
    /// 2^16·STEP_1` on the low words, each a 16-bit value (an `IS_HALF` lookup).
    pub const STEP_0: usize = 25;
    pub const STEP_1: usize = 26;

    pub const MU_READ: usize = 27;
    pub const MU_WRITE: usize = 28;

    pub const NUM_COLUMNS: usize = 29;
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the MEMW_A trace table from aligned operations.
///
/// Reuses `MemwOperation` — the trace generator uses `old_timestamp[0]`
/// (verified equal for all accessed bytes by the routing logic).
/// One MEMW_A row as the walk keeps it: 48 bytes where a [`MemwOperation`]
/// takes 152 (the walk materializes about one every three cycles). An aligned
/// access's bytes share one old timestamp ([`MemwOperation`]'s
/// `old_timestamp[0]`), and its eight value (old) elements fit one `u64`: eight
/// bytes for a memory access, the two 32-bit register halves (the rest zero)
/// for a register access that missed MEMW_R. `limbs`: the row is laid out in
/// [`MemwAlignedOrder::Limbs`] (the run's order, set where the walk routes the
/// access): its columns 25 and 26 are the step's limbs, it sends their two
/// `IS_HALF` lookups and no LT comparison. `#[repr(C)]`: a chunk's rows are
/// the records the card fills its MEMW_A table from ([`card_rows`]).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct AlignedRow {
    base_address: u64,
    timestamp: u64,
    old_timestamp: u64,
    value: u64,
    old: u64,
    width: u8,
    is_read: bool,
    is_register: bool,
    limbs: bool,
}

impl AlignedRow {
    /// The row of an op [`super::trace_builder`] routed to MEMW_A in `order`.
    /// Under [`MemwAlignedOrder::Limbs`] an op whose step does not fit
    /// ([`step_fits_limbs`], the routing's own predicate) is refused with the
    /// reason, never laid out with wrapped limbs: the routing admits none, so
    /// a refusal is a prover bug, and the walk returns it as an error.
    #[inline]
    pub(crate) fn from_memw(op: &MemwOperation, order: MemwAlignedOrder) -> Result<Self, String> {
        let limbs = order == MemwAlignedOrder::Limbs;
        if limbs && !step_fits_limbs(op.old_timestamp[0], op.timestamp) {
            return Err(format!(
                "a MEMW_A row in the limbs' form needs a step in 1..2^32 on one high word, \
                 not {:#x} → {:#x}",
                op.old_timestamp[0], op.timestamp
            ));
        }
        let pack = |elements: &[u32; 8]| -> u64 {
            if op.is_register {
                elements[0] as u64 | (elements[1] as u64) << 32
            } else {
                elements.iter().enumerate().fold(0, |packed, (i, &byte)| {
                    packed | (byte as u64 & 0xFF) << (8 * i)
                })
            }
        };
        let row = Self {
            base_address: op.base_address,
            timestamp: op.timestamp,
            old_timestamp: op.old_timestamp[0],
            value: pack(&op.value),
            old: pack(&op.old),
            width: op.width,
            is_read: op.is_read,
            is_register: op.is_register,
            limbs,
        };
        debug_assert!(
            row.value() == op.value && row.old() == op.old,
            "a MEMW_A op's elements do not fit its row: {op:?}"
        );
        Ok(row)
    }

    #[inline]
    fn unpack(&self, packed: u64) -> [u32; 8] {
        let mut elements = [0u32; 8];
        if self.is_register {
            elements[0] = packed as u32;
            elements[1] = (packed >> 32) as u32;
        } else {
            for (i, element) in elements.iter_mut().enumerate() {
                *element = (packed >> (8 * i)) as u8 as u32;
            }
        }
        elements
    }

    /// [`MemwOperation::value`].
    #[inline]
    pub(crate) fn value(&self) -> [u32; 8] {
        self.unpack(self.value)
    }

    /// [`MemwOperation::old`].
    #[inline]
    pub(crate) fn old(&self) -> [u32; 8] {
        self.unpack(self.old)
    }

    #[inline]
    pub(crate) fn base_address(&self) -> u64 {
        self.base_address
    }

    #[inline]
    pub(crate) fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// The accessed bytes' (register words') shared old timestamp.
    #[inline]
    pub(crate) fn old_timestamp(&self) -> u64 {
        self.old_timestamp
    }

    #[inline]
    pub(crate) fn width(&self) -> u8 {
        self.width
    }

    #[inline]
    pub(crate) fn is_read(&self) -> bool {
        self.is_read
    }

    #[inline]
    pub(crate) fn is_register(&self) -> bool {
        self.is_register
    }

    /// Whether the row is laid out in [`MemwAlignedOrder::Limbs`].
    #[inline]
    pub(crate) fn limbs(&self) -> bool {
        self.limbs
    }

    /// The step's limbs `(d0, d1)` of a [`Self::limbs`] row ([`step_limbs`]).
    #[inline]
    pub(crate) fn step_limbs(&self) -> (u16, u16) {
        step_limbs(self.old_timestamp, self.timestamp)
    }

    /// The comparison `(old timestamp, timestamp)` the row asks LT for: one
    /// under [`MemwAlignedOrder::Lt`], none under [`MemwAlignedOrder::Limbs`]
    /// (the row proves its order itself).
    #[inline]
    pub(crate) fn lt_pair(&self) -> Option<(u64, u64)> {
        (!self.limbs).then_some((self.old_timestamp, self.timestamp))
    }

    /// A row of these fields, the value and old value packed as the row keeps
    /// them (a test's: the walk makes its rows with [`Self::from_memw`]).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        base_address: u64,
        timestamp: u64,
        old_timestamp: u64,
        value: u64,
        old: u64,
        width: u8,
        is_read: bool,
        is_register: bool,
    ) -> Self {
        Self {
            base_address,
            timestamp,
            old_timestamp,
            value,
            old,
            width,
            is_read,
            is_register,
            limbs: false,
        }
    }

    /// A test's row: the same row in `order`.
    #[cfg(test)]
    pub(crate) fn in_order(mut self, order: MemwAlignedOrder) -> Self {
        self.limbs = order == MemwAlignedOrder::Limbs;
        self
    }

    /// A test's fault: the same row, its timestamp `cycles` later.
    #[cfg(test)]
    pub(crate) fn later(mut self, cycles: u64) -> Self {
        self.timestamp += cycles;
        self
    }

    /// [`MemwOperation::write_flags`].
    #[inline]
    pub(crate) fn write_flags(&self) -> (bool, bool, bool) {
        match self.width {
            2 => (true, false, false),
            4 => (false, true, false),
            8 => (false, false, true),
            _ => (false, false, false),
        }
    }
}

const _: () = assert!(std::mem::size_of::<AlignedRow>() == 48);

/// Aligned rows as the card's MEMW_A fill reads them
/// (`multilinear::records::AlignedRow`): the same bytes.
pub(crate) fn card_rows(rows: &[AlignedRow]) -> &[multilinear::records::AlignedRow] {
    use multilinear::records::AlignedRow as Card;
    use std::mem::{align_of, offset_of, size_of};
    const _: () = {
        assert!(size_of::<AlignedRow>() == size_of::<Card>());
        assert!(align_of::<AlignedRow>() == align_of::<Card>());
        assert!(offset_of!(AlignedRow, base_address) == offset_of!(Card, base_address));
        assert!(offset_of!(AlignedRow, timestamp) == offset_of!(Card, timestamp));
        assert!(offset_of!(AlignedRow, old_timestamp) == offset_of!(Card, old_timestamp));
        assert!(offset_of!(AlignedRow, value) == offset_of!(Card, value));
        assert!(offset_of!(AlignedRow, old) == offset_of!(Card, old));
        assert!(offset_of!(AlignedRow, width) == offset_of!(Card, width));
        assert!(offset_of!(AlignedRow, is_read) == offset_of!(Card, is_read));
        assert!(offset_of!(AlignedRow, is_register) == offset_of!(Card, is_register));
        assert!(offset_of!(AlignedRow, limbs) == offset_of!(Card, limbs));
    };
    // SAFETY: both are `#[repr(C)]` with the same fields in the same order
    // (asserted above); the flags are `bool`s, 0 or 1, read as `u8`s, and no
    // reader looks at the padding after them.
    unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const Card, rows.len()) }
}

#[cfg(test)]
pub(crate) fn generate_memw_aligned_trace(
    operations: &[AlignedRow],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    generate_memw_aligned_trace_as(operations, TraceForm::Wide)
}

/// The widths MEMW_A traces needed so far in this process (`tables::gpack`).
static WIDTHS: WidthHint = WidthHint::new();

/// [`generate_memw_aligned_trace`] in `form` (`tables::gpack`).
pub(crate) fn generate_memw_aligned_trace_as(
    operations: &[AlignedRow],
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    generate_memw_aligned_trace_segments_as(&[operations], form)
}

/// [`generate_memw_aligned_trace_as`] over `segments`, the operations one after another: a
/// chunk handed out as the window parts it lies in.
pub(crate) fn generate_memw_aligned_trace_segments_as(
    segments: &[&[AlignedRow]],
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let len: usize = segments.iter().map(|s| s.len()).sum();
    let num_rows = len.next_power_of_two().max(4);
    generate_main!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, |table| {
        for (row_idx, op) in segments.iter().flat_map(|s| s.iter()).enumerate() {
            table.set_bool(row_idx, cols::IS_REGISTER, op.is_register());

            table.set_dword_whh(row_idx, cols::BASE_ADDRESS[0], op.base_address());

            for (column, element) in cols::VALUE.into_iter().zip(op.value()) {
                table.set_u64(row_idx, column, element as u64);
            }

            table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp());

            let (w2, w4, w8) = op.write_flags();
            table.set_bool(row_idx, cols::WRITE2, w2);
            table.set_bool(row_idx, cols::WRITE4, w4);
            table.set_bool(row_idx, cols::WRITE8, w8);

            for (column, element) in cols::OLD.into_iter().zip(op.old()) {
                table.set_u64(row_idx, column, element as u64);
            }

            if op.limbs() {
                // The step's limbs: the old timestamp is not a column.
                let (d0, d1) = op.step_limbs();
                table.set_u64(row_idx, cols::STEP_0, u64::from(d0));
                table.set_u64(row_idx, cols::STEP_1, u64::from(d1));
            } else {
                // Single old_timestamp (verified equal for all bytes when routed here)
                table.set_dword_wl(row_idx, cols::OLD_TIMESTAMP_0, op.old_timestamp());
            }

            table.set_bool(row_idx, cols::MU_READ, op.is_read());
            table.set_bool(row_idx, cols::MU_WRITE, !op.is_read());
        }
    })
}

// =========================================================================
// Bus interactions (20 under `Lt`, 21 under `Limbs`)
// =========================================================================

/// [`bus_interactions_for`] [`MemwAlignedOrder::Lt`]: the per-table STARK's.
pub fn bus_interactions() -> Vec<BusInteraction> {
    bus_interactions_for(MemwAlignedOrder::Lt)
}

/// MEMW_A's interactions with the timestamp order in `order`'s form: under
/// [`MemwAlignedOrder::Lt`] the 20 of the per-table STARK, under
/// [`MemwAlignedOrder::Limbs`] the ALU lookup replaced by the two limbs'
/// `IS_HALF` lookups (interactions 1 and 2) and every old token's timestamp
/// derived from the row's own timestamp and limbs: 21.
pub fn bus_interactions_for(order: MemwAlignedOrder) -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(21);

    let mu_sum = Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE);

    // The old timestamp every old token carries: two columns under `Lt`; under
    // `Limbs` the low word `TIMESTAMP_0 − 1 − STEP_0 − 2^16·STEP_1` and the
    // row's own high word, `TIMESTAMP_1`, so there is no old high word to
    // constrain (with a free one, old = 2^32 and new = 10 would pass on the low
    // words).
    let (old_timestamp_lo, old_timestamp_hi) = match order {
        MemwAlignedOrder::Lt => (
            BusValue::Packed {
                start_column: cols::OLD_TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD_TIMESTAMP_1,
                packing: Packing::Direct,
            },
        ),
        MemwAlignedOrder::Limbs => (
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::TIMESTAMP_0,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::STEP_0,
                },
                LinearTerm::Column {
                    coefficient: -(1 << 16),
                    column: cols::STEP_1,
                },
                LinearTerm::Constant(-1),
            ]),
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
        ),
    };

    // -------------------------------------------------------------------------
    // IS_HALF[base_address[0] + write2 + 3*write4 + 7*write8] with μ_sum
    // Range check: ensures base_address[0] + mask fits in 16 bits, so the
    // byte-address span of the access doesn't overflow the low-half field element.
    // Alignment itself is the caller's (CPU's) responsibility — see Assumptions above.
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::IsHalfword,
        mu_sum.clone(),
        vec![BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::BASE_ADDRESS[0],
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::WRITE2,
            },
            LinearTerm::Column {
                coefficient: 3,
                column: cols::WRITE4,
            },
            LinearTerm::Column {
                coefficient: 7,
                column: cols::WRITE8,
            },
        ])],
    ));

    match order {
        // ---------------------------------------------------------------------
        // ALU[old_timestamp, timestamp, opsel(LT), 1, 0] → asserts old_ts < ts.
        // (Every LT lookup goes through the unified ALU bus with
        // signed=0/invert=0; there is no dedicated `Lt` bus.)
        // ---------------------------------------------------------------------
        MemwAlignedOrder::Lt => interactions.push(BusInteraction::sender(
            BusId::Alu,
            mu_sum.clone(),
            vec![
                BusValue::Packed {
                    start_column: cols::OLD_TIMESTAMP_0,
                    packing: Packing::DWordWL,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::DWordWL,
                },
                BusValue::constant(alu_op::LT as u64),
                BusValue::constant(1),
                BusValue::constant(0),
            ],
        )),
        // ---------------------------------------------------------------------
        // IS_HALF[STEP_0], IS_HALF[STEP_1] with μ_sum → the step
        // `timestamp − old_timestamp` is `1 + STEP_0 + 2^16·STEP_1`, in
        // 1..=2^32, on the high word the two share. No selector: every active
        // row sends both, at the multiplicity of its tokens.
        // ---------------------------------------------------------------------
        MemwAlignedOrder::Limbs => {
            for column in [cols::STEP_0, cols::STEP_1] {
                interactions.push(BusInteraction::sender(
                    BusId::IsHalfword,
                    mu_sum.clone(),
                    vec![BusValue::linear(vec![LinearTerm::Column {
                        coefficient: 1,
                        column,
                    }])],
                ));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Memory bus interactions (16 total)
    // -------------------------------------------------------------------------
    // base_address as DWordWL:
    //   lo32 = base_address[0] + 2^16 * base_address[1]
    //   hi32 = base_address[2]
    // For aligned accesses, address for byte i: lo32 + i (no carry since aligned)

    let base_addr_lo = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::BASE_ADDRESS[0],
        },
        LinearTerm::Column {
            coefficient: 1 << 16,
            column: cols::BASE_ADDRESS[1],
        },
    ]);

    let base_addr_hi = BusValue::Packed {
        start_column: cols::BASE_ADDRESS[2],
        packing: Packing::Direct,
    };

    // CM16: memory[is_register, base_address, old_timestamp, old[0]] with +μ_sum
    interactions.push(BusInteraction::sender(
        BusId::Memory,
        mu_sum.clone(),
        vec![
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            base_addr_lo.clone(),
            base_addr_hi.clone(),
            old_timestamp_lo.clone(),
            old_timestamp_hi.clone(),
            BusValue::Packed {
                start_column: cols::OLD[0],
                packing: Packing::Direct,
            },
        ],
    ));

    // CM17: memory[is_register, base_address, timestamp, value[0]] with -μ_sum
    interactions.push(BusInteraction::receiver(
        BusId::Memory,
        mu_sum,
        vec![
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            base_addr_lo.clone(),
            base_addr_hi.clone(),
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[0],
                packing: Packing::Direct,
            },
        ],
    ));

    // w2 multiplicity: write2 + write4 + write8
    let w2_mult = Multiplicity::Sum3(cols::WRITE2, cols::WRITE4, cols::WRITE8);

    // CM18/19: byte 1 with w2
    // For aligned accesses, adding 1 to lo32 never overflows (alignment guarantees it).
    let addr_1_lo = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::BASE_ADDRESS[0],
        },
        LinearTerm::Column {
            coefficient: 1 << 16,
            column: cols::BASE_ADDRESS[1],
        },
        LinearTerm::Constant(1),
    ]);

    interactions.push(BusInteraction::sender(
        BusId::Memory,
        w2_mult.clone(),
        vec![
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            addr_1_lo.clone(),
            base_addr_hi.clone(),
            old_timestamp_lo.clone(),
            old_timestamp_hi.clone(),
            BusValue::Packed {
                start_column: cols::OLD[1],
                packing: Packing::Direct,
            },
        ],
    ));

    interactions.push(BusInteraction::receiver(
        BusId::Memory,
        w2_mult,
        vec![
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            addr_1_lo,
            base_addr_hi.clone(),
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[1],
                packing: Packing::Direct,
            },
        ],
    ));

    // CM20/21: bytes 2-3 with w4
    for i in 2..=3 {
        let addr_i_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::BASE_ADDRESS[0],
            },
            LinearTerm::Column {
                coefficient: 1 << 16,
                column: cols::BASE_ADDRESS[1],
            },
            LinearTerm::Constant(i as i64),
        ]);

        interactions.push(BusInteraction::sender(
            BusId::Memory,
            Multiplicity::Sum(cols::WRITE4, cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                addr_i_lo.clone(),
                base_addr_hi.clone(),
                old_timestamp_lo.clone(),
                old_timestamp_hi.clone(),
                BusValue::Packed {
                    start_column: cols::OLD[i],
                    packing: Packing::Direct,
                },
            ],
        ));

        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::Sum(cols::WRITE4, cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                addr_i_lo,
                base_addr_hi.clone(),
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::VALUE[i],
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // CM22/23: bytes 4-7 with write8
    for i in 4..=7 {
        let addr_i_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::BASE_ADDRESS[0],
            },
            LinearTerm::Column {
                coefficient: 1 << 16,
                column: cols::BASE_ADDRESS[1],
            },
            LinearTerm::Constant(i as i64),
        ]);

        interactions.push(BusInteraction::sender(
            BusId::Memory,
            Multiplicity::Column(cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                addr_i_lo.clone(),
                base_addr_hi.clone(),
                old_timestamp_lo.clone(),
                old_timestamp_hi.clone(),
                BusValue::Packed {
                    start_column: cols::OLD[i],
                    packing: Packing::Direct,
                },
            ],
        ));

        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::Column(cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                addr_i_lo,
                base_addr_hi.clone(),
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::VALUE[i],
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // -------------------------------------------------------------------------
    // CO24: Read receiver (from CPU)
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(cols::MU_READ),
        vec![
            // old[8]
            BusValue::Packed {
                start_column: cols::OLD[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[2],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[3],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[4],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[5],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[6],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[7],
                packing: Packing::Direct,
            },
            // is_register
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            // base_address as DWordWL: [lo32, hi32]
            base_addr_lo.clone(),
            base_addr_hi.clone(),
            // value[8]
            BusValue::Packed {
                start_column: cols::VALUE[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[2],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[3],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[4],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[5],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[6],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[7],
                packing: Packing::Direct,
            },
            // timestamp
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            // write flags
            BusValue::Packed {
                start_column: cols::WRITE2,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::WRITE4,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::WRITE8,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // CO25: Write receiver (from CPU)
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(cols::MU_WRITE),
        vec![
            // is_register
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            // base_address as DWordWL: [lo32, hi32]
            base_addr_lo,
            base_addr_hi,
            // value[8]
            BusValue::Packed {
                start_column: cols::VALUE[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[2],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[3],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[4],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[5],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[6],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[7],
                packing: Packing::Direct,
            },
            // timestamp
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            // write flags
            BusValue::Packed {
                start_column: cols::WRITE2,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::WRITE4,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::WRITE8,
                packing: Packing::Direct,
            },
        ],
    ));

    interactions
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// `μ_sum = μ_read + μ_write` as a builder expression.
fn mu_sum_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(b: &B) -> B::Expr {
    b.main(0, cols::MU_READ) + b.main(0, cols::MU_WRITE)
}

/// `w2 = write2 + write4 + write8` as a builder expression.
fn w2_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(b: &B) -> B::Expr {
    b.main(0, cols::WRITE2) + b.main(0, cols::WRITE4) + b.main(0, cols::WRITE8)
}

/// The MEMW_A table's 8 transition constraints as a single [`ConstraintSet`]:
/// - idx 0:   `IS_BIT<μ_sum>`;
/// - idx 1:   `w2 ⇒ μ_sum` (`w2·(1 − μ_sum)`);
/// - idx 2,3: `IS_BIT` on `μ_read`, `μ_write`;
/// - idx 4-6: `IS_BIT` on `write2`, `write4`, `write8`;
/// - idx 7:   `IS_BIT<w2>` (width sum is a bit).
#[derive(Clone, Copy)]
pub struct MemwAlignedConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemwAlignedConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0: IS_BIT<μ_sum> = μ_sum * (1 - μ_sum)
        let one = b.one();
        let mu_sum = mu_sum_expr(b);
        b.emit_base(0, mu_sum.clone() * (one - mu_sum));

        // idx 1: w2 ⇒ μ_sum = w2 * (1 - μ_sum)
        let one = b.one();
        let w2 = w2_expr(b);
        let mu_sum = mu_sum_expr(b);
        b.emit_base(1, w2 * (one - mu_sum));

        // idx 2,3: IS_BIT<μ_read>, IS_BIT<μ_write>
        emit_is_bit(b, 2, cols::MU_READ, None);
        emit_is_bit(b, 3, cols::MU_WRITE, None);

        // idx 4-6: IS_BIT on the width flags
        emit_is_bit(b, 4, cols::WRITE2, None);
        emit_is_bit(b, 5, cols::WRITE4, None);
        emit_is_bit(b, 6, cols::WRITE8, None);

        // idx 7: IS_BIT<w2> = w2 * (1 - w2)
        let one = b.one();
        let w2 = w2_expr(b);
        b.emit_base(7, w2.clone() * (one - w2));
    }
}

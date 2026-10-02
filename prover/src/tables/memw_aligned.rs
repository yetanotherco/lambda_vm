//! MEMW_A (Memory Write/Read — Aligned) table.
//!
//! Fast path for aligned memory/register accesses where all bytes share the
//! same old_timestamp. Most operations (aligned memory + all register accesses)
//! route here instead of the heavier MEMW table.
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
//! - `old_timestamp`: DWordWL (2 cols — single, not 8!)
//! - `mu_read`, `mu_write`: multiplicity columns
//!
//! ## Bus Interactions (20)
//! - 1 IS_HALF[base_address[0] + mask] (range check: address span fits in 16 bits)
//! - 1 ALU[old_timestamp, timestamp, opsel(LT), 1, 0] → asserts old_ts < ts
//! - 16 Memory bus tokens
//! - 2 MEMW output interactions (read + write)
//!
//! ## Constraints (8 total)
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

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

use super::memw::MemwOperation;
use super::types::{BusId, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use crate::constraints::templates::emit_is_bit;

/// Maximum number of rows per MEMW_A table chunk.
pub const MAX_ROWS: usize = super::max_rows::MEMW_A;

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

    /// Single old_timestamp (shared across all bytes, since they're aligned)
    pub const OLD_TIMESTAMP_0: usize = 25;
    pub const OLD_TIMESTAMP_1: usize = 26;

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
/// for a register access that missed MEMW_R.
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
}

impl AlignedRow {
    /// The row of an op [`super::trace_builder`] routed to MEMW_A.
    #[inline]
    pub(crate) fn from_memw(op: &MemwOperation) -> Self {
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
        };
        debug_assert!(
            row.value() == op.value && row.old() == op.old,
            "a MEMW_A op's elements do not fit its row: {op:?}"
        );
        row
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

pub(crate) fn generate_memw_aligned_trace(
    operations: &[AlignedRow],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = operations.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in operations.iter().enumerate() {
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

        // Single old_timestamp (verified equal for all bytes when routed here)
        table.set_dword_wl(row_idx, cols::OLD_TIMESTAMP_0, op.old_timestamp());

        table.set_bool(row_idx, cols::MU_READ, op.is_read());
        table.set_bool(row_idx, cols::MU_WRITE, !op.is_read());
    }

    trace
}

// =========================================================================
// Bus interactions (20 total)
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(20);

    let mu_sum = Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE);

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

    // -------------------------------------------------------------------------
    // ALU[old_timestamp, timestamp, opsel(LT), 1, 0] → asserts old_ts < ts.
    // (Every LT lookup goes through the unified ALU bus with
    // signed=0/invert=0; there is no dedicated `Lt` bus.)
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
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
    ));

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
            BusValue::Packed {
                start_column: cols::OLD_TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD_TIMESTAMP_1,
                packing: Packing::Direct,
            },
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
            BusValue::Packed {
                start_column: cols::OLD_TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD_TIMESTAMP_1,
                packing: Packing::Direct,
            },
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
                BusValue::Packed {
                    start_column: cols::OLD_TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::OLD_TIMESTAMP_1,
                    packing: Packing::Direct,
                },
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
                BusValue::Packed {
                    start_column: cols::OLD_TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::OLD_TIMESTAMP_1,
                    packing: Packing::Direct,
                },
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

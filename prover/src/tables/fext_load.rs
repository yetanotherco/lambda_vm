//! FEXT_LOAD accelerator table: load a degree-3 extension element from three
//! registers into field-storage (spec ECALL `-20`).
//!
//! Reads the destination address from x10 and the three coefficients (native
//! u64 form) from x11/x12/x13, range-checks each `< p`, and **writes** them into
//! field-storage (memory domains 3/4/5) at the destination address. The write is
//! a genuine memory write (consume old token, emit new token) — not the draft
//! spec's read-assert (`output = value`, which forces `old == value`).
//!
//! Field-storage rides the low-level `Memory` consistency bus directly (like
//! PAGE/REGISTER/HALT): a full field-element value fits in one token and the
//! domain is a free field element, so no change to the shared MEMW chip. The
//! per-cell init/fini tokens are emitted by the `FEXT_PAGE` bookend table.
//!
//! ## Bus interactions
//! - **Receiver** on `Ecall`: `[ts_lo, ts_hi, FEXT_LOAD_lo32, FEXT_LOAD_hi32]` (mult = μ).
//! - **Sender** on `Memw` ×4: register reads of x10/x11/x12/x13.
//! - **Sender** on `Alu` ×3: `coeff_i < p` range checks.
//! - **Sender/Receiver** on `Memory` ×3 each: per coefficient, consume the old
//!   token `[3+i, addr, old_ts, old_val]` and emit the new token
//!   `[3+i, addr, ts, coeff_lo + 2^32*coeff_hi]`.
//! - **Sender** on `Alu` ×3: `old_ts < ts` temporal ordering.
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use executor::vm::instruction::execution::FEXT_LOAD_SYSCALL_NUMBER;

use crate::constraints::templates::emit_is_bit;

use super::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, SHIFT_32, VmTable, alu_op, zeroed_fe_vec,
};

/// Column indices for the FEXT_LOAD table.
pub mod cols {
    // Timestamp (DWordWL).
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    // Destination field-storage address (DWordWL), from x10.
    pub const ADDR_0: usize = 2;
    pub const ADDR_1: usize = 3;

    // Coefficients in native form (each a DWordWL), from x11/x12/x13.
    pub const C0_0: usize = 4;
    pub const C0_1: usize = 5;
    pub const C1_0: usize = 6;
    pub const C1_1: usize = 7;
    pub const C2_0: usize = 8;
    pub const C2_1: usize = 9;

    /// Multiplicity bit.
    pub const MU: usize = 10;

    // Per-coefficient memory-argument witness: the timestamp (DWordWL) and value
    // of the field-storage cell before this write (0/0 on first touch).
    pub const OLD_TS0_0: usize = 11;
    pub const OLD_TS0_1: usize = 12;
    pub const OLD_VAL0: usize = 13;
    pub const OLD_TS1_0: usize = 14;
    pub const OLD_TS1_1: usize = 15;
    pub const OLD_VAL1: usize = 16;
    pub const OLD_TS2_0: usize = 17;
    pub const OLD_TS2_1: usize = 18;
    pub const OLD_VAL2: usize = 19;

    pub const NUM_COLUMNS: usize = 20;

    /// Low-limb column of coefficient `i` (`i` in 0..3).
    pub const fn coeff(i: usize) -> usize {
        C0_0 + 2 * i
    }
    /// Low-limb column of the old timestamp for coefficient `i`.
    pub const fn old_ts(i: usize) -> usize {
        OLD_TS0_0 + 3 * i
    }
    /// Old value column for coefficient `i`.
    pub const fn old_val(i: usize) -> usize {
        OLD_VAL0 + 3 * i
    }
}

const LOAD_SYSCALL_LO: u64 = FEXT_LOAD_SYSCALL_NUMBER & 0xFFFF_FFFF;
const LOAD_SYSCALL_HI: u64 = FEXT_LOAD_SYSCALL_NUMBER >> 32;

/// Goldilocks prime `p = 2^64 - 2^32 + 1` as a `DWordWL`: low limb `1`, high
/// limb `2^32 - 1`. Carried on the ALU bus as two sub-`p` limbs (avoids the
/// `p ≡ 0` wraparound a single packed field element would suffer).
const P_LO: u64 = 1;
const P_HI: u64 = (1u64 << 32) - 1;

/// One FEXT_LOAD invocation.
#[derive(Debug, Clone)]
pub struct FextLoadOperation {
    pub timestamp: u64,
    pub addr: u64,
    /// The three coefficients in native form (canonical field elements `< p`).
    pub coeffs: [u64; 3],
    /// Timestamp of the previous write to each field-storage cell (0 on first touch).
    pub old_ts: [u64; 3],
    /// Value previously stored in each field-storage cell (0 on first touch).
    pub old_val: [u64; 3],
}

/// Generates the FEXT_LOAD trace (one row per op, padded to next power of two,
/// min 4). Padding rows are all-zero (`μ = 0`).
pub fn generate_fext_load_trace(
    ops: &[FextLoadOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row, op) in ops.iter().enumerate() {
        table.set_dword_wl(row, cols::TIMESTAMP_0, op.timestamp);
        table.set_dword_wl(row, cols::ADDR_0, op.addr);
        for i in 0..3 {
            table.set_dword_wl(row, cols::coeff(i), op.coeffs[i]);
            table.set_dword_wl(row, cols::old_ts(i), op.old_ts[i]);
            table.set_fe(row, cols::old_val(i), FE::from(op.old_val[i]));
        }
        table.set_fe(row, cols::MU, FE::one());
    }

    trace
}

/// A MEMW register-read interaction (24-element CO24 read; `is_register = 1`,
/// `write2 = 1`). Register file is byte-addressed ×2.
fn memw_register_read(lo: usize, hi: usize, reg: u64) -> BusInteraction {
    let col = |c| BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    };
    BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(cols::MU),
        vec![
            col(lo),
            col(hi),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(1), // is_register = 1
            BusValue::constant(2 * reg),
            BusValue::constant(0),
            col(lo),
            col(hi),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            col(cols::TIMESTAMP_0),
            col(cols::TIMESTAMP_1),
            BusValue::constant(1), // write2 = 1
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    )
}

/// `lhs(DWordWL) < rhs(DWordWL)` on the unified ALU bus, asserting the result is
/// 1: `[lhs, rhs, opsel(LT), 1, 0]`.
fn alu_lt(lhs_lo: usize, rhs: [BusValue; 2]) -> BusInteraction {
    let [rhs_lo, rhs_hi] = rhs;
    BusInteraction::sender(
        BusId::Alu,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: lhs_lo,
                packing: Packing::DWordWL,
            },
            rhs_lo,
            rhs_hi,
            BusValue::constant(alu_op::LT as u64),
            BusValue::constant(1),
            BusValue::constant(0),
        ],
    )
}

/// The three `Memory` + `Alu` interactions for writing coefficient `i` to
/// field-storage cell `(domain 3+i, addr)`.
fn field_write(i: usize) -> [BusInteraction; 3] {
    let addr = || {
        [
            BusValue::Packed {
                start_column: cols::ADDR_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::ADDR_1,
                packing: Packing::Direct,
            },
        ]
    };
    let [addr_lo, addr_hi] = addr();
    // consume the old token [3+i, addr, old_ts, old_val]
    let consume = BusInteraction::sender(
        BusId::Memory,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::constant(3 + i as u64),
            addr_lo,
            addr_hi,
            BusValue::Packed {
                start_column: cols::old_ts(i),
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::old_ts(i) + 1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::old_val(i),
                packing: Packing::Direct,
            },
        ],
    );
    let [addr_lo, addr_hi] = addr();
    // emit the new token [3+i, addr, ts, coeff_lo + 2^32*coeff_hi]
    let emit = BusInteraction::receiver(
        BusId::Memory,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::constant(3 + i as u64),
            addr_lo,
            addr_hi,
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::coeff(i),
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: SHIFT_32,
                    column: cols::coeff(i) + 1,
                },
            ]),
        ],
    );
    // old_ts < ts
    let order = alu_lt(
        cols::old_ts(i),
        [
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
        ],
    );
    [consume, emit, order]
}

/// `coeff < p` on the unified ALU bus.
fn coeff_lt_p(i: usize) -> BusInteraction {
    alu_lt(
        cols::coeff(i),
        [BusValue::constant(P_LO), BusValue::constant(P_HI)],
    )
}

/// Bus interactions for FEXT_LOAD: `Ecall` receiver + 4 register reads +
/// 3 `< p` range checks + 3×(consume-old, emit-new, `old_ts < ts`).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = vec![
        BusInteraction::receiver(
            BusId::Ecall,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::constant(LOAD_SYSCALL_LO),
                BusValue::constant(LOAD_SYSCALL_HI),
            ],
        ),
        memw_register_read(cols::ADDR_0, cols::ADDR_1, 10),
        memw_register_read(cols::C0_0, cols::C0_1, 11),
        memw_register_read(cols::C1_0, cols::C1_1, 12),
        memw_register_read(cols::C2_0, cols::C2_1, 13),
    ];
    for i in 0..3 {
        interactions.push(coeff_lt_p(i));
    }
    for i in 0..3 {
        interactions.extend(field_write(i));
    }
    interactions
}

/// FEXT_LOAD constraints: idx 0 is `IS_BIT(μ)`. Coefficient canonicality and
/// memory consistency are enforced by bus interactions, not polynomial constraints.
pub struct FextLoadConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for FextLoadConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU, None);
    }
}

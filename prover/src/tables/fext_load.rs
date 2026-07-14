//! FEXT_LOAD accelerator table: load a degree-3 extension element from three
//! registers into field-storage (spec ECALL `-20`).
//!
//! Reads the destination address from x10 and the three coefficients (native
//! u64 form) from x11/x12/x13, range-checks each coefficient `< p` (canonical
//! Goldilocks element) via the ALU `LT` bus, and writes them into field-storage
//! (memory domains 3/4/5) at the destination address.
//!
//! ## Bus interactions (this phase)
//! - **Receiver** on `Ecall`: `[ts_lo, ts_hi, FEXT_LOAD_lo32, FEXT_LOAD_hi32]` (mult = μ).
//! - **Sender** on `Memw` ×4: register reads of x10/x11/x12/x13.
//! - **Sender** on `Alu` ×3: `coeff_i < p` range checks.
//!
//! The field-storage writes (memory domains 3/4/5, value = `lo + 2^32*hi` per
//! coefficient) and the domain init/finalization are added in the field-storage
//! phase. Unlike the draft spec, this write is a genuine memory *write*, not a
//! read-assert (draft bug: `output = value` forces `old == value`).
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use executor::vm::instruction::execution::FEXT_LOAD_SYSCALL_NUMBER;

use crate::constraints::templates::emit_is_bit;

use super::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op, zeroed_fe_vec,
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

    pub const NUM_COLUMNS: usize = 11;

    /// Low-limb column of coefficient `i` (`i` in 0..3).
    pub const fn coeff(i: usize) -> usize {
        C0_0 + 2 * i
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
        for (i, &c) in op.coeffs.iter().enumerate() {
            table.set_dword_wl(row, cols::coeff(i), c);
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

/// `coeff < p` on the unified ALU bus: `[coeff(DWordWL), p(DWordWL), opsel(LT), 1, 0]`
/// (unsigned, non-inverted, asserting the result is 1).
fn coeff_lt_p(coeff_lo: usize) -> BusInteraction {
    BusInteraction::sender(
        BusId::Alu,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: coeff_lo,
                packing: Packing::DWordWL,
            },
            BusValue::constant(P_LO),
            BusValue::constant(P_HI),
            BusValue::constant(alu_op::LT as u64),
            BusValue::constant(1),
            BusValue::constant(0),
        ],
    )
}

/// Bus interactions for FEXT_LOAD (this phase): `Ecall` receiver + 4 register
/// reads + 3 `< p` range checks. Field-storage writes are added later.
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
        interactions.push(coeff_lt_p(cols::coeff(i)));
    }
    interactions
}

/// FEXT_LOAD constraints: idx 0 is `IS_BIT(μ)`. Coefficient canonicality is
/// enforced by the ALU `LT` bus interactions, not by polynomial constraints.
pub struct FextLoadConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for FextLoadConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU, None);
    }
}

//! FEXT_STORE accelerator table: read a degree-3 extension element from
//! field-storage and write its three coefficients back to registers a1/a2/a3
//! (ECALL `-22`). The read-back companion to FEXT_LOAD (which reads coeffs from
//! a1/a2/a3), so a guest can extract Fp3 results from the accelerator's
//! field-storage into normal registers.
//!
//! ## Bus interactions
//! - **Receiver** on `Ecall`: `[ts_lo, ts_hi, FEXT_STORE_lo32, FEXT_STORE_hi32]` (mult = μ).
//! - **Sender** on `Memw`: register read of x10 (source address).
//! - **Sender/Receiver** on `Memory` ×3 each: read coefficient `d` from cell
//!   `(3+d, src_addr)` (consume old / emit new; value = `lo + 2^32*hi`).
//! - **Sender** on `Alu` ×3: `old_ts < ts` temporal ordering.
//! - **Sender** on `Memw` ×3: register writes of a1/a2/a3 = `[lo, hi]` per coeff.
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use executor::vm::instruction::execution::FEXT_STORE_SYSCALL_NUMBER;

use crate::constraints::templates::emit_is_bit;

use super::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, SHIFT_32, VmTable, alu_op, zeroed_fe_vec,
};

/// Column indices for the FEXT_STORE table.
pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    /// Source field-storage address (DWordWL), from x10.
    pub const SRC_ADDR_0: usize = 2;
    pub const SRC_ADDR_1: usize = 3;

    // Coefficient words written to a1/a2/a3 (each register = [lo, hi] Words).
    pub const C0_LO: usize = 4;
    pub const C0_HI: usize = 5;
    pub const C1_LO: usize = 6;
    pub const C1_HI: usize = 7;
    pub const C2_LO: usize = 8;
    pub const C2_HI: usize = 9;

    // Last-write timestamp (DWordWL) of each read field cell.
    pub const OLD_TS0_0: usize = 10;
    pub const OLD_TS0_1: usize = 11;
    pub const OLD_TS1_0: usize = 12;
    pub const OLD_TS1_1: usize = 13;
    pub const OLD_TS2_0: usize = 14;
    pub const OLD_TS2_1: usize = 15;

    /// Multiplicity bit.
    pub const MU: usize = 16;

    pub const NUM_COLUMNS: usize = 17;

    /// Low-word column of coefficient `d`.
    pub const fn coeff_lo(d: usize) -> usize {
        C0_LO + 2 * d
    }
    /// Low-limb column of the old timestamp for the read of coefficient `d`.
    pub const fn old_ts(d: usize) -> usize {
        OLD_TS0_0 + 2 * d
    }
}

const STORE_SYSCALL_LO: u64 = FEXT_STORE_SYSCALL_NUMBER & 0xFFFF_FFFF;
const STORE_SYSCALL_HI: u64 = FEXT_STORE_SYSCALL_NUMBER >> 32;

/// One FEXT_STORE invocation.
#[derive(Debug, Clone)]
pub struct FextStoreOperation {
    pub timestamp: u64,
    pub src_addr: u64,
    /// The three coefficients read from field-storage.
    pub coeffs: [u64; 3],
    /// Last-write timestamp of each read field cell.
    pub old_ts: [u64; 3],
}

/// Generates the FEXT_STORE trace (one row per op, padded to next power of two,
/// min 4). Padding rows are all-zero (`μ = 0`).
pub fn generate_fext_store_trace(
    ops: &[FextStoreOperation],
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
        table.set_dword_wl(row, cols::SRC_ADDR_0, op.src_addr);
        for d in 0..3 {
            table.set_dword_wl(row, cols::coeff_lo(d), op.coeffs[d]);
            table.set_dword_wl(row, cols::old_ts(d), op.old_ts[d]);
        }
        table.set_fe(row, cols::MU, FE::one());
    }

    trace
}

fn direct(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

/// The coefficient value `lo + 2^32*hi` as a single field element (matches the
/// value LOAD/FMA wrote into field-storage).
fn coeff_value(d: usize) -> BusValue {
    BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::coeff_lo(d),
        },
        LinearTerm::ColumnUnsigned {
            coefficient: SHIFT_32,
            column: cols::coeff_lo(d) + 1,
        },
    ])
}

/// MEMW register **read** (24-element CO24; `old == value`, `is_register = 1`,
/// `write2 = 1`).
fn memw_register_read(lo: usize, hi: usize, reg: u64) -> BusInteraction {
    BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(cols::MU),
        vec![
            direct(lo),
            direct(hi),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(1),
            BusValue::constant(2 * reg),
            BusValue::constant(0),
            direct(lo),
            direct(hi),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            direct(cols::TIMESTAMP_0),
            direct(cols::TIMESTAMP_1),
            BusValue::constant(1),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    )
}

/// MEMW register **write** (16-element write format; `is_register = 1`,
/// `write2 = 1`): writes `[lo, hi]` of coefficient `d` to register `reg`.
fn memw_register_write(reg: u64, lo: usize, hi: usize) -> BusInteraction {
    BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::constant(1), // is_register
            BusValue::constant(2 * reg),
            BusValue::constant(0),
            direct(lo),
            direct(hi),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            direct(cols::TIMESTAMP_0),
            direct(cols::TIMESTAMP_1),
            BusValue::constant(1), // write2
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    )
}

/// `old_ts(DWordWL) < ts` on the ALU bus, asserting the result is 1.
fn alu_lt_ts(old_ts_lo: usize) -> BusInteraction {
    BusInteraction::sender(
        BusId::Alu,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: old_ts_lo,
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
    )
}

/// The three `Memory` interactions for reading coefficient `d` from cell
/// `(3+d, src_addr)`: consume old token, emit new token (read: value unchanged),
/// and `old_ts < ts`.
fn field_read(d: usize) -> [BusInteraction; 3] {
    let domain = 3 + d as u64;
    let consume = BusInteraction::sender(
        BusId::Memory,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::constant(domain),
            direct(cols::SRC_ADDR_0),
            direct(cols::SRC_ADDR_1),
            direct(cols::old_ts(d)),
            direct(cols::old_ts(d) + 1),
            coeff_value(d),
        ],
    );
    let emit = BusInteraction::receiver(
        BusId::Memory,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::constant(domain),
            direct(cols::SRC_ADDR_0),
            direct(cols::SRC_ADDR_1),
            direct(cols::TIMESTAMP_0),
            direct(cols::TIMESTAMP_1),
            coeff_value(d),
        ],
    );
    [consume, emit, alu_lt_ts(cols::old_ts(d))]
}

/// Bus interactions: `Ecall` receiver + register read (x10) + 3×(field read
/// consume/emit + `old_ts<ts`) + 3 register writes (a1/a2/a3).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = vec![
        BusInteraction::receiver(
            BusId::Ecall,
            Multiplicity::Column(cols::MU),
            vec![
                direct(cols::TIMESTAMP_0),
                direct(cols::TIMESTAMP_1),
                BusValue::constant(STORE_SYSCALL_LO),
                BusValue::constant(STORE_SYSCALL_HI),
            ],
        ),
        memw_register_read(cols::SRC_ADDR_0, cols::SRC_ADDR_1, 10),
    ];
    for d in 0..3 {
        interactions.extend(field_read(d));
    }
    for d in 0..3 {
        interactions.push(memw_register_write(
            11 + d as u64,
            cols::coeff_lo(d),
            cols::coeff_lo(d) + 1,
        ));
    }
    interactions
}

/// FEXT_STORE constraints: idx 0 is `IS_BIT(μ)`.
pub struct FextStoreConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for FextStoreConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU, None);
    }
}

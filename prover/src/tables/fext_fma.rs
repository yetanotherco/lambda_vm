//! FEXT_FMA accelerator table: `output = a*b + c` over the native degree-3
//! Goldilocks extension `Fp[x]/(x^3 - 2)` (spec ECALL `-21`).
//!
//! One row per invocation. The extension is `w^3 = 2`, so (matching
//! `Degree3GoldilocksExtensionField::mul`):
//! - `out0 = a0*b0 + 2*(a1*b2 + a2*b1) + c0`
//! - `out1 = a0*b1 + a1*b0 + 2*a2*b2 + c1`
//! - `out2 = a0*b2 + a1*b1 + a2*b0 + c2`
//!
//! These are the spec's general `x^3 = αx^2 + βx + γ` constraints specialized to
//! the VM's native field (`α = β = 0`, `γ = 2`), which makes them degree 2.
//!
//! ## Bus interactions
//! - **Receiver** on `Ecall`: `[ts_lo, ts_hi, FEXT_FMA_lo32, FEXT_FMA_hi32]` (mult = μ).
//! - **Sender** on `Memw` ×4: register reads of x10/x11/x12/x13 (a/b/c/out addrs).
//! - **Memory** reads ×9: coefficient `d` of each of a/b/c from cell `(3+d, addr)`.
//! - **Memory** writes ×3: output coefficient `d` to cell `(3+d, out_addr)`.
//! - **Alu** ×12: `old_ts < ts` temporal ordering per field-storage access.
//!
//! Field-storage rides the low-level `Memory` bus directly (single field-element
//! value, free domain); the per-cell init/fini tokens come from `FEXT_PAGE`. The
//! executor guarantees a/b/c/out addresses are pairwise distinct, so all accesses
//! can share one timestamp.
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use executor::vm::instruction::execution::FEXT_FMA_SYSCALL_NUMBER;

use crate::constraints::templates::emit_is_bit;

use super::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op, zeroed_fe_vec,
};

/// Column indices for the FEXT_FMA table.
pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    // Operand addresses (each DWordWL). Registers: x10=a, x11=b, x12=c, x13=out.
    pub const OUT_ADDR_0: usize = 2;
    pub const OUT_ADDR_1: usize = 3;
    pub const A_ADDR_0: usize = 4;
    pub const A_ADDR_1: usize = 5;
    pub const B_ADDR_0: usize = 6;
    pub const B_ADDR_1: usize = 7;
    pub const C_ADDR_0: usize = 8;
    pub const C_ADDR_1: usize = 9;

    // Operand coefficients (each a single BaseField element).
    pub const A0: usize = 10;
    pub const A1: usize = 11;
    pub const A2: usize = 12;
    pub const B0: usize = 13;
    pub const B1: usize = 14;
    pub const B2: usize = 15;
    pub const C0: usize = 16;
    pub const C1: usize = 17;
    pub const C2: usize = 18;

    // Output coefficients.
    pub const OUT0: usize = 19;
    pub const OUT1: usize = 20;
    pub const OUT2: usize = 21;

    /// Multiplicity bit.
    pub const MU: usize = 22;

    // Old timestamps for the 9 reads, ordered (value a/b/c, coeff 0/1/2), each a
    // DWordWL (2 cols): base 23..41.
    pub const READ_OLD_TS: usize = 23;
    // Old timestamp (DWordWL) + old value for the 3 output writes: base 41..50.
    pub const WRITE_OLD: usize = 41;

    pub const NUM_COLUMNS: usize = 50;

    /// Low-limb column of the old timestamp for read `(v, d)` (v: 0=a,1=b,2=c).
    pub const fn read_old_ts(v: usize, d: usize) -> usize {
        READ_OLD_TS + (v * 3 + d) * 2
    }
    /// Low-limb column of the old timestamp for output write of coefficient `d`.
    pub const fn write_old_ts(d: usize) -> usize {
        WRITE_OLD + d * 3
    }
    /// Old-value column for the output write of coefficient `d`.
    pub const fn write_old_val(d: usize) -> usize {
        WRITE_OLD + d * 3 + 2
    }
    /// Base (low) address column of operand `v` (0=a,1=b,2=c).
    pub const fn operand_addr(v: usize) -> usize {
        A_ADDR_0 + v * 2
    }
    /// Coefficient `d` column of operand `v` (0=a,1=b,2=c).
    pub const fn operand_coeff(v: usize, d: usize) -> usize {
        A0 + v * 3 + d
    }
}

const FMA_SYSCALL_LO: u64 = FEXT_FMA_SYSCALL_NUMBER & 0xFFFF_FFFF;
const FMA_SYSCALL_HI: u64 = FEXT_FMA_SYSCALL_NUMBER >> 32;

/// One FEXT_FMA invocation.
#[derive(Debug, Clone)]
pub struct FextFmaOperation {
    pub timestamp: u64,
    pub out_addr: u64,
    pub a_addr: u64,
    pub b_addr: u64,
    pub c_addr: u64,
    pub a: [u64; 3],
    pub b: [u64; 3],
    pub c: [u64; 3],
    pub output: [u64; 3],
    /// Last-write timestamp of each read cell, `[value a/b/c][coeff d]`.
    pub read_old_ts: [[u64; 3]; 3],
    /// Last-write timestamp of each output cell (coeff d).
    pub write_old_ts: [u64; 3],
    /// Prior value of each output cell (coeff d).
    pub write_old_val: [u64; 3],
}

/// Generates the FEXT_FMA trace. One row per operation, padded to the next power
/// of two (min 4). Padding rows are all-zero (`μ = 0`).
pub fn generate_fext_fma_trace(
    ops: &[FextFmaOperation],
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
        table.set_dword_wl(row, cols::OUT_ADDR_0, op.out_addr);
        table.set_dword_wl(row, cols::A_ADDR_0, op.a_addr);
        table.set_dword_wl(row, cols::B_ADDR_0, op.b_addr);
        table.set_dword_wl(row, cols::C_ADDR_0, op.c_addr);

        for (v, coeffs) in [op.a, op.b, op.c].into_iter().enumerate() {
            for (d, &val) in coeffs.iter().enumerate() {
                table.set_fe(row, cols::operand_coeff(v, d), FE::from(val));
                table.set_dword_wl(row, cols::read_old_ts(v, d), op.read_old_ts[v][d]);
            }
        }
        for d in 0..3 {
            table.set_fe(row, cols::OUT0 + d, FE::from(op.output[d]));
            table.set_dword_wl(row, cols::write_old_ts(d), op.write_old_ts[d]);
            table.set_fe(row, cols::write_old_val(d), FE::from(op.write_old_val[d]));
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

/// A MEMW register-read interaction (24-element CO24 read; `is_register = 1`,
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

/// `old_ts(DWordWL) < ts` on the unified ALU bus, asserting the result is 1.
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

/// The three interactions for one field-storage access at cell `(domain, addr)`.
/// `old_val`/`new_val` are the value columns/exprs of the old and new tokens
/// (equal for a read). `old_ts_lo` is the old-timestamp low column.
fn field_access(
    domain: u64,
    addr_lo: usize,
    addr_hi: usize,
    old_ts_lo: usize,
    old_val: BusValue,
    new_val: BusValue,
) -> [BusInteraction; 3] {
    let consume = BusInteraction::sender(
        BusId::Memory,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::constant(domain),
            direct(addr_lo),
            direct(addr_hi),
            direct(old_ts_lo),
            direct(old_ts_lo + 1),
            old_val,
        ],
    );
    let emit = BusInteraction::receiver(
        BusId::Memory,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::constant(domain),
            direct(addr_lo),
            direct(addr_hi),
            direct(cols::TIMESTAMP_0),
            direct(cols::TIMESTAMP_1),
            new_val,
        ],
    );
    [consume, emit, alu_lt_ts(old_ts_lo)]
}

/// Bus interactions: `Ecall` receiver + 4 register reads + 9 field reads +
/// 3 field writes (3 interactions each).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = vec![
        BusInteraction::receiver(
            BusId::Ecall,
            Multiplicity::Column(cols::MU),
            vec![
                direct(cols::TIMESTAMP_0),
                direct(cols::TIMESTAMP_1),
                BusValue::constant(FMA_SYSCALL_LO),
                BusValue::constant(FMA_SYSCALL_HI),
            ],
        ),
        memw_register_read(cols::A_ADDR_0, cols::A_ADDR_1, 10),
        memw_register_read(cols::B_ADDR_0, cols::B_ADDR_1, 11),
        memw_register_read(cols::C_ADDR_0, cols::C_ADDR_1, 12),
        memw_register_read(cols::OUT_ADDR_0, cols::OUT_ADDR_1, 13),
    ];

    // 9 reads: coefficient d of operand v from cell (3+d, operand_addr(v)).
    // A read leaves the value unchanged, so old_val == new_val == coeff column.
    for v in 0..3 {
        let addr_lo = cols::operand_addr(v);
        for d in 0..3 {
            let val = direct(cols::operand_coeff(v, d));
            interactions.extend(field_access(
                3 + d as u64,
                addr_lo,
                addr_lo + 1,
                cols::read_old_ts(v, d),
                val.clone(),
                val,
            ));
        }
    }

    // 3 writes: output coefficient d to cell (3+d, out_addr).
    for d in 0..3 {
        interactions.extend(field_access(
            3 + d as u64,
            cols::OUT_ADDR_0,
            cols::OUT_ADDR_1,
            cols::write_old_ts(d),
            direct(cols::write_old_val(d)),
            direct(cols::OUT0 + d),
        ));
    }

    interactions
}

/// The FEXT_FMA constraints:
/// - idx 0: `IS_BIT(μ)`;
/// - idx 1-3: the three extension-field FMA coefficient equations (degree 2).
pub struct FextFmaConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for FextFmaConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU, None);

        let two = b.const_base(2);
        let m = |b: &B, col| b.main(0, col);

        let a0 = m(b, cols::A0);
        let a1 = m(b, cols::A1);
        let a2 = m(b, cols::A2);
        let b0 = m(b, cols::B0);
        let b1 = m(b, cols::B1);
        let b2 = m(b, cols::B2);
        let c0 = m(b, cols::C0);
        let c1 = m(b, cols::C1);
        let c2 = m(b, cols::C2);
        let out0 = m(b, cols::OUT0);
        let out1 = m(b, cols::OUT1);
        let out2 = m(b, cols::OUT2);

        // out0 = a0*b0 + 2*(a1*b2 + a2*b1) + c0
        let expr0 = out0
            - (a0.clone() * b0.clone()
                + two.clone() * (a1.clone() * b2.clone() + a2.clone() * b1.clone())
                + c0);
        b.emit_base(1, expr0);

        // out1 = a0*b1 + a1*b0 + 2*a2*b2 + c1
        let expr1 = out1
            - (a0.clone() * b1.clone()
                + a1.clone() * b0.clone()
                + two * (a2.clone() * b2.clone())
                + c1);
        b.emit_base(2, expr1);

        // out2 = a0*b2 + a1*b1 + a2*b0 + c2
        let expr2 = out2 - (a0 * b2 + a1 * b1 + a2 * b0 + c2);
        b.emit_base(3, expr2);
    }
}

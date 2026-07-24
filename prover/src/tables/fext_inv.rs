//! FEXT_INV accelerator table: the witnessed multiplicative inverse
//! `out = x^-1` over the native degree-3 Goldilocks extension `Fp[x]/(x^3 - 2)`
//! (spec ECALL `-24`).
//!
//! The executor supplies the inverse host-side; the chip *constrains* it, so the
//! guest no longer runs the `ext_mul(x, hint) == 1` software check-multiply the
//! `INV_FP3_HINT` path used. It is a **witnessed inverse**: one ext-mul
//! constraint `x · inv == 1` plus a zero flag for soundness at `x = 0`.
//!
//! ## Soundness (witnessed inverse with a zero flag)
//! Let `p = x · inv` (an Fp3 product) and `is_zero ∈ {0,1}`. On a real row (μ=1):
//! - `p0 = μ − is_zero`, `p1 = 0`, `p2 = 0`  ⇒  `x · inv = 1 − is_zero`;
//! - `xd · is_zero = 0` for every coefficient `d`.
//!
//! If `x ≠ 0` some `xd ≠ 0`, so `is_zero = 0` and `x · inv = 1`: `inv` is forced
//! to the unique true inverse — a wrong witness cannot satisfy the constraint.
//! If `x = 0` then `p = 0` forces `is_zero = 1` and `inv` is free (the executor
//! stores 0); the honest guest rejects zero up front, so this branch is never
//! reached legitimately, but the chip stays satisfiable. Writing `μ` where the
//! gadget wants the constant `1` keeps every constraint degree-2 and naturally
//! satisfied on all-zero padding rows (μ = is_zero = 0).
//!
//! ## ABI / bus interactions
//! `x`/`out` are field-storage handles in x10/x11.
//! - **Receiver** on `Ecall`: `[ts_lo, ts_hi, FEXT_INV_lo32, hi32]` (mult μ).
//! - **Sender** on `Memw` ×2: register reads of x10/x11 (x/out addrs).
//! - **Memory** reads ×3: coefficient `d` of `x` from cell `(3+d, x_addr)`.
//! - **Memory** writes ×3: inverse coefficient `d` to cell `(3+d, out_addr)`.
//! - **Alu** ×6: `old_ts < ts` temporal ordering per field-storage access.
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use executor::vm::instruction::execution::FEXT_INV_SYSCALL_NUMBER;

use crate::constraints::templates::emit_is_bit;

use super::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op, zeroed_fe_vec,
};

/// Column indices for the FEXT_INV table.
pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    // Operand addresses (each DWordWL). Registers: x10 = x, x11 = out.
    pub const X_ADDR_0: usize = 2;
    pub const X_ADDR_1: usize = 3;
    pub const OUT_ADDR_0: usize = 4;
    pub const OUT_ADDR_1: usize = 5;

    // Input coefficients (each a single BaseField element).
    pub const X0: usize = 6;
    pub const X1: usize = 7;
    pub const X2: usize = 8;

    // Inverse (output) coefficients.
    pub const INV0: usize = 9;
    pub const INV1: usize = 10;
    pub const INV2: usize = 11;

    /// Zero flag: 1 iff `x == 0` (the non-invertible element).
    pub const IS_ZERO: usize = 12;
    /// Multiplicity bit.
    pub const MU: usize = 13;

    // Old timestamps for the 3 x reads (coeff 0/1/2), each a DWordWL: 14..20.
    pub const READ_OLD_TS: usize = 14;
    // Old timestamp (DWordWL) + old value for the 3 output writes: base 20..29.
    pub const WRITE_OLD: usize = 20;

    pub const NUM_COLUMNS: usize = 29;

    /// Low-limb column of the old timestamp for x read of coefficient `d`.
    pub const fn read_old_ts(d: usize) -> usize {
        READ_OLD_TS + d * 2
    }
    /// Low-limb column of the old timestamp for the output write of coefficient `d`.
    pub const fn write_old_ts(d: usize) -> usize {
        WRITE_OLD + d * 3
    }
    /// Old-value column for the output write of coefficient `d`.
    pub const fn write_old_val(d: usize) -> usize {
        WRITE_OLD + d * 3 + 2
    }
}

const INV_SYSCALL_LO: u64 = FEXT_INV_SYSCALL_NUMBER & 0xFFFF_FFFF;
const INV_SYSCALL_HI: u64 = FEXT_INV_SYSCALL_NUMBER >> 32;

/// One FEXT_INV invocation.
#[derive(Debug, Clone)]
pub struct FextInvOperation {
    pub timestamp: u64,
    pub x_addr: u64,
    pub out_addr: u64,
    pub x: [u64; 3],
    /// The inverse coefficients (executor-supplied; `[0;3]` for `x = 0`).
    pub inv: [u64; 3],
    /// Last-write timestamp of each read `x` cell (coeff d).
    pub read_old_ts: [u64; 3],
    /// Last-write timestamp of each output cell (coeff d).
    pub write_old_ts: [u64; 3],
    /// Prior value of each output cell (coeff d).
    pub write_old_val: [u64; 3],
}

/// Generates the FEXT_INV trace. One row per operation, padded to the next power
/// of two (min 4). Padding rows are all-zero (`μ = is_zero = 0`).
pub fn generate_fext_inv_trace(
    ops: &[FextInvOperation],
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
        table.set_dword_wl(row, cols::X_ADDR_0, op.x_addr);
        table.set_dword_wl(row, cols::OUT_ADDR_0, op.out_addr);

        for (d, &val) in op.x.iter().enumerate() {
            table.set_fe(row, cols::X0 + d, FE::from(val));
            table.set_dword_wl(row, cols::read_old_ts(d), op.read_old_ts[d]);
        }
        for d in 0..3 {
            table.set_fe(row, cols::INV0 + d, FE::from(op.inv[d]));
            table.set_dword_wl(row, cols::write_old_ts(d), op.write_old_ts[d]);
            table.set_fe(row, cols::write_old_val(d), FE::from(op.write_old_val[d]));
        }

        if op.x == [0, 0, 0] {
            table.set_fe(row, cols::IS_ZERO, FE::one());
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
/// `old_val`/`new_val` are the value columns of the old and new tokens (equal for
/// a read). `old_ts_lo` is the old-timestamp low column.
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

/// Bus interactions: `Ecall` receiver + 2 register reads + 3 field reads +
/// 3 field writes (3 interactions each).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = vec![
        BusInteraction::receiver(
            BusId::Ecall,
            Multiplicity::Column(cols::MU),
            vec![
                direct(cols::TIMESTAMP_0),
                direct(cols::TIMESTAMP_1),
                BusValue::constant(INV_SYSCALL_LO),
                BusValue::constant(INV_SYSCALL_HI),
            ],
        ),
        memw_register_read(cols::X_ADDR_0, cols::X_ADDR_1, 10),
        memw_register_read(cols::OUT_ADDR_0, cols::OUT_ADDR_1, 11),
    ];

    // 3 reads: coefficient d of x from cell (3+d, x_addr). A read leaves the
    // value unchanged, so old_val == new_val == the coeff column.
    for d in 0..3 {
        let val = direct(cols::X0 + d);
        interactions.extend(field_access(
            3 + d as u64,
            cols::X_ADDR_0,
            cols::X_ADDR_1,
            cols::read_old_ts(d),
            val.clone(),
            val,
        ));
    }

    // 3 writes: inverse coefficient d to cell (3+d, out_addr).
    for d in 0..3 {
        interactions.extend(field_access(
            3 + d as u64,
            cols::OUT_ADDR_0,
            cols::OUT_ADDR_1,
            cols::write_old_ts(d),
            direct(cols::write_old_val(d)),
            direct(cols::INV0 + d),
        ));
    }

    interactions
}

/// The FEXT_INV constraints:
/// - idx 0: `IS_BIT(μ)`;
/// - idx 1: `IS_BIT(is_zero)`;
/// - idx 2-4: `x · inv = 1 − is_zero` (Fp3, with `μ` in place of the constant);
/// - idx 5-7: `xd · is_zero = 0` (forces `is_zero = 0` whenever `x ≠ 0`).
///
/// All degree 2; see the module soundness note.
pub struct FextInvConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for FextInvConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU, None);
        emit_is_bit(b, 1, cols::IS_ZERO, None);

        let two = b.const_base(2);
        let m = |b: &B, col| b.main(0, col);

        let x0 = m(b, cols::X0);
        let x1 = m(b, cols::X1);
        let x2 = m(b, cols::X2);
        let i0 = m(b, cols::INV0);
        let i1 = m(b, cols::INV1);
        let i2 = m(b, cols::INV2);
        let is_zero = m(b, cols::IS_ZERO);
        let mu = m(b, cols::MU);

        // p = x · inv over Fp[x]/(x^3 - 2) (same schoolbook as FEXT_FMA).
        // p0 = x0*i0 + 2*(x1*i2 + x2*i1)
        let p0 = x0.clone() * i0.clone()
            + two.clone() * (x1.clone() * i2.clone() + x2.clone() * i1.clone());
        // p1 = x0*i1 + x1*i0 + 2*x2*i2
        let p1 = x0.clone() * i1.clone()
            + x1.clone() * i0.clone()
            + two.clone() * (x2.clone() * i2.clone());
        // p2 = x0*i2 + x1*i1 + x2*i0
        let p2 = x0.clone() * i2 + x1.clone() * i1 + x2.clone() * i0;

        // x · inv == 1 − is_zero (μ stands in for the constant 1 so padding holds).
        b.emit_base(2, p0 - (mu - is_zero.clone()));
        b.emit_base(3, p1);
        b.emit_base(4, p2);

        // xd · is_zero == 0 (any nonzero coefficient forces is_zero = 0).
        b.emit_base(5, x0 * is_zero.clone());
        b.emit_base(6, x1 * is_zero.clone());
        b.emit_base(7, x2 * is_zero);
    }
}

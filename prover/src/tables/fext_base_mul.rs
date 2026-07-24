//! FEXT_BASE_MUL accelerator table: the Goldilocks×Fp3 asymmetric product
//! `out = base · ext` over the native degree-3 Goldilocks extension
//! `Fp[x]/(x^3 - 2)` (spec ECALL `-23`).
//!
//! `base` is a single Goldilocks element (the subfield); `ext`/`out` are Fp3.
//! The product is coefficient-wise — exactly three base multiplies, NOT a lifted
//! full extension multiply:
//! - `out0 = base · ext0`
//! - `out1 = base · ext1`
//! - `out2 = base · ext2`
//!
//! This completes the FEXT chip API: `FEXT_FMA` handles ext×ext, this handles the
//! cheaper base×ext product the verifier's FRI butterfly (`𝜐⁻¹·ζ`) and OOD
//! denominator walk (`g·z`) issue in software today.
//!
//! ## ABI / bus interactions
//! `base` rides register x10 by value (not an address); the executor guarantees
//! it is canonical (`< p`). `ext`/`out` are field-storage handles in x11/x12.
//! - **Receiver** on `Ecall`: `[ts_lo, ts_hi, FEXT_BASE_MUL_lo32, hi32]` (mult μ).
//! - **Sender** on `Memw` ×3: register reads of x10 (base value), x11/x12 (addrs).
//! - **Memory** reads ×3: coefficient `d` of `ext` from cell `(3+d, ext_addr)`.
//! - **Memory** writes ×3: output coefficient `d` to cell `(3+d, out_addr)`.
//! - **Alu** ×6: `old_ts < ts` temporal ordering per field-storage access.
//!
//! `base` is read off the register bus as a `DWordWL` (`base_lo + 2^32·base_hi`);
//! the memw counterparty range-checks the two 32-bit limbs, and the executor
//! rejects `base >= p`, so the reconstructed field element is exactly `base`.
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use executor::vm::instruction::execution::FEXT_BASE_MUL_SYSCALL_NUMBER;

use crate::constraints::templates::emit_is_bit;

use super::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op, zeroed_fe_vec,
};

/// Column indices for the FEXT_BASE_MUL table.
pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    // Base Goldilocks element, read from register x10 as a DWordWL (lo/hi 32-bit
    // limbs). Reconstructed to the field element `base_lo + 2^32·base_hi`.
    pub const BASE_0: usize = 2;
    pub const BASE_1: usize = 3;
    // Operand addresses (each DWordWL). Registers: x11 = ext, x12 = out.
    pub const EXT_ADDR_0: usize = 4;
    pub const EXT_ADDR_1: usize = 5;
    pub const OUT_ADDR_0: usize = 6;
    pub const OUT_ADDR_1: usize = 7;

    // ext coefficients (each a single BaseField element).
    pub const E0: usize = 8;
    pub const E1: usize = 9;
    pub const E2: usize = 10;

    // Output coefficients.
    pub const OUT0: usize = 11;
    pub const OUT1: usize = 12;
    pub const OUT2: usize = 13;

    /// Multiplicity bit.
    pub const MU: usize = 14;

    // Old timestamps for the 3 ext reads (coeff 0/1/2), each a DWordWL: 15..21.
    pub const READ_OLD_TS: usize = 15;
    // Old timestamp (DWordWL) + old value for the 3 output writes: base 21..30.
    pub const WRITE_OLD: usize = 21;

    pub const NUM_COLUMNS: usize = 30;

    /// Low-limb column of the old timestamp for ext read of coefficient `d`.
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

const BASE_MUL_SYSCALL_LO: u64 = FEXT_BASE_MUL_SYSCALL_NUMBER & 0xFFFF_FFFF;
const BASE_MUL_SYSCALL_HI: u64 = FEXT_BASE_MUL_SYSCALL_NUMBER >> 32;

/// One FEXT_BASE_MUL invocation.
#[derive(Debug, Clone)]
pub struct FextBaseMulOperation {
    pub timestamp: u64,
    pub base: u64,
    pub ext_addr: u64,
    pub out_addr: u64,
    pub ext: [u64; 3],
    pub output: [u64; 3],
    /// Last-write timestamp of each read `ext` cell (coeff d).
    pub read_old_ts: [u64; 3],
    /// Last-write timestamp of each output cell (coeff d).
    pub write_old_ts: [u64; 3],
    /// Prior value of each output cell (coeff d).
    pub write_old_val: [u64; 3],
}

/// Generates the FEXT_BASE_MUL trace. One row per operation, padded to the next
/// power of two (min 4). Padding rows are all-zero (`μ = 0`).
pub fn generate_fext_base_mul_trace(
    ops: &[FextBaseMulOperation],
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
        table.set_dword_wl(row, cols::BASE_0, op.base);
        table.set_dword_wl(row, cols::EXT_ADDR_0, op.ext_addr);
        table.set_dword_wl(row, cols::OUT_ADDR_0, op.out_addr);

        for (d, &val) in op.ext.iter().enumerate() {
            table.set_fe(row, cols::E0 + d, FE::from(val));
            table.set_dword_wl(row, cols::read_old_ts(d), op.read_old_ts[d]);
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

/// Bus interactions: `Ecall` receiver + 3 register reads + 3 field reads +
/// 3 field writes (3 interactions each).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = vec![
        BusInteraction::receiver(
            BusId::Ecall,
            Multiplicity::Column(cols::MU),
            vec![
                direct(cols::TIMESTAMP_0),
                direct(cols::TIMESTAMP_1),
                BusValue::constant(BASE_MUL_SYSCALL_LO),
                BusValue::constant(BASE_MUL_SYSCALL_HI),
            ],
        ),
        // x10 carries the base VALUE (reconstructed in-constraint), x11/x12 addrs.
        memw_register_read(cols::BASE_0, cols::BASE_1, 10),
        memw_register_read(cols::EXT_ADDR_0, cols::EXT_ADDR_1, 11),
        memw_register_read(cols::OUT_ADDR_0, cols::OUT_ADDR_1, 12),
    ];

    // 3 reads: coefficient d of ext from cell (3+d, ext_addr). A read leaves the
    // value unchanged, so old_val == new_val == the coeff column.
    for d in 0..3 {
        let val = direct(cols::E0 + d);
        interactions.extend(field_access(
            3 + d as u64,
            cols::EXT_ADDR_0,
            cols::EXT_ADDR_1,
            cols::read_old_ts(d),
            val.clone(),
            val,
        ));
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

/// The FEXT_BASE_MUL constraints:
/// - idx 0: `IS_BIT(μ)`;
/// - idx 1-3: `out[d] = base · ext[d]`, with `base = base_lo + 2^32·base_hi`
///   (degree 2).
pub struct FextBaseMulConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for FextBaseMulConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU, None);

        let two_32 = b.const_base(1u64 << 32);
        let m = |b: &B, col| b.main(0, col);

        // Reconstruct the base field element from its two register limbs.
        let base = m(b, cols::BASE_0) + two_32 * m(b, cols::BASE_1);

        for d in 0..3 {
            // out[d] = base * ext[d]
            let expr = m(b, cols::OUT0 + d) - base.clone() * m(b, cols::E0 + d);
            b.emit_base(1 + d, expr);
        }
    }
}

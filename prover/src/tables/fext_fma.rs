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
//! ## Bus interactions (this phase)
//! - **Receiver** on `Ecall`: `[ts_lo, ts_hi, FEXT_FMA_lo32, FEXT_FMA_hi32]` (mult = μ).
//! - **Sender** on `Memw` ×4: register reads of x10/x11/x12/x13 (out/a/b/c addresses).
//!
//! The field-storage reads of `a,b,c` and the write of `output` (memory domains
//! 3/4/5), plus the domain init/finalization, are added in the field-storage
//! phase; until then the coefficient columns are free witness bound only by the
//! arithmetic constraints.
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use executor::vm::instruction::execution::FEXT_FMA_SYSCALL_NUMBER;

use crate::constraints::templates::emit_is_bit;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, zeroed_fe_vec};

/// Column indices for the FEXT_FMA table.
pub mod cols {
    // Timestamp (DWordWL: 2 cols)
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    // Operand addresses (each DWordWL). Registers: x10=out, x11=a, x12=b, x13=c.
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

    /// Multiplicity bit (1 for real rows, 0 for padding).
    pub const MU: usize = 22;

    pub const NUM_COLUMNS: usize = 23;
}

/// FEXT_FMA syscall number split into the two 32-bit limbs the `Ecall` bus carries.
const FMA_SYSCALL_LO: u64 = FEXT_FMA_SYSCALL_NUMBER & 0xFFFF_FFFF;
const FMA_SYSCALL_HI: u64 = FEXT_FMA_SYSCALL_NUMBER >> 32;

/// One FEXT_FMA invocation: `output = a*b + c`, with operand addresses.
#[derive(Debug, Clone)]
pub struct FextFmaOperation {
    pub timestamp: u64,
    pub out_addr: u64,
    pub a_addr: u64,
    pub b_addr: u64,
    pub c_addr: u64,
    /// Coefficients of `a`, `b`, `c` (canonical field elements).
    pub a: [u64; 3],
    pub b: [u64; 3],
    pub c: [u64; 3],
    /// Result coefficients `a*b + c` (canonical).
    pub output: [u64; 3],
}

/// Generates the FEXT_FMA trace. One row per operation, padded to the next power
/// of two (min 4). Padding rows are all-zero (`μ = 0`), which satisfies the
/// arithmetic constraints (`0 = 0`) and `IS_BIT μ`.
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

        for (i, base) in [cols::A0, cols::B0, cols::C0].into_iter().enumerate() {
            let coeffs = [op.a, op.b, op.c][i];
            for (k, &v) in coeffs.iter().enumerate() {
                table.set_fe(row, base + k, FE::from(v));
            }
        }
        for (k, &v) in op.output.iter().enumerate() {
            table.set_fe(row, cols::OUT0 + k, FE::from(v));
        }

        table.set_fe(row, cols::MU, FE::one());
    }

    trace
}

/// A single MEMW register-read interaction (24-element CO24 read: `old == value`,
/// `is_register = 1`, `write2 = 1`). `reg` is the register index; the register
/// file is byte-addressed ×2, so the base address is `2*reg`.
fn memw_register_read(addr_lo: usize, addr_hi: usize, reg: u64) -> BusInteraction {
    let addr = |col| BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    };
    let ts = |col| BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    };
    BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(cols::MU),
        vec![
            // old[0..7] = [addr_lo, addr_hi, 0, 0, 0, 0, 0, 0]
            addr(addr_lo),
            addr(addr_hi),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // is_register = 1
            BusValue::constant(1),
            // base_address = [2*reg, 0]
            BusValue::constant(2 * reg),
            BusValue::constant(0),
            // value[0..7] = same as old (read)
            addr(addr_lo),
            addr(addr_hi),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // timestamp
            ts(cols::TIMESTAMP_0),
            ts(cols::TIMESTAMP_1),
            // w2 = 1, w4 = 0, w8 = 0 (register = 2 words)
            BusValue::constant(1),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    )
}

/// Bus interactions for the FEXT_FMA table (this phase: `Ecall` receiver +
/// 4 register reads). Field-storage `Memory` tokens are added later.
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // Receive the ECALL from the CPU, keyed by the FEXT_FMA syscall number.
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
                BusValue::constant(FMA_SYSCALL_LO),
                BusValue::constant(FMA_SYSCALL_HI),
            ],
        ),
        // Register reads: x10=out, x11=a, x12=b, x13=c.
        memw_register_read(cols::OUT_ADDR_0, cols::OUT_ADDR_1, 10),
        memw_register_read(cols::A_ADDR_0, cols::A_ADDR_1, 11),
        memw_register_read(cols::B_ADDR_0, cols::B_ADDR_1, 12),
        memw_register_read(cols::C_ADDR_0, cols::C_ADDR_1, 13),
    ]
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

//! FP3_MUL table — AIR for the Goldilocks Fp3 multiply precompile.
//!
//! One row per `Fp3Mul` syscall invocation. Each row witnesses:
//!   lhs = [a0, a1, a2] ∈ Fp
//!   rhs = [b0, b1, b2] ∈ Fp
//!   result = [c0, c1, c2] ∈ Fp
//!
//! where multiplication is in the cubic extension field x³ - 2 over Goldilocks:
//!   c0 = a0·b0 + 2·a1·b2 + 2·a2·b1
//!   c1 = a0·b1 + a1·b0  + 2·a2·b2
//!   c2 = a0·b2 + a1·b1  + a2·b0
//!
//! ## ABI (matches executor `SyscallNumbers::Fp3Mul`)
//!
//! - syscall number = `FP3_MUL_SYSCALL_NUMBER` (`u64::MAX - 2`) in a7 (x17)
//! - a0 (x10) = result_ptr (8-byte aligned, [u64; 3] output)
//! - a1 (x11) = lhs_ptr ([u64; 3] input)
//! - a2 (x12) = rhs_ptr ([u64; 3] input)
//!
//! ## Bus wiring (matches the keccak core chip's pattern)
//!
//! The table is a pure receiver on the shared `Ecall` bus (matching the CPU's
//! ECALL sender) and a sender on the shared `Memw` bus for every register read,
//! memory read and memory write performed by the syscall. The matching
//! `MemwOperation`s are generated in `trace_builder::collect_fp3_mul_ops` so the
//! MEMW / MEMW_A / MEMW_R tables receive them and the bus balances.
//!
//! Memory values travel on the `Memw` bus as 8 individual little-endian bytes
//! (each its own bus element), because the per-byte Memory-consistency tokens
//! that MEMW emits are byte-granular and must match the byte-granular PAGE
//! storage. Register values travel as `[lo32, hi32, 0, 0, 0, 0, 0, 0]`
//! (DWordWL packing), matching `pack_register_value` and the REGISTER table.
//!
//! ## Constraints
//!
//! Three degree-2 transition constraints check the multiply formula over the
//! Goldilocks base field.
//!
//! NOTE: as in the original skeleton, these constraints are NOT yet sound over
//! the full Goldilocks field without range-checking the inputs/outputs to
//! `[0, p)` and without binding the field-element columns to the byte columns.
//! That soundness work is deferred; this module provides correct *bus balance*
//! and the multiply constraint skeleton so the precompile integrates end-to-end.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use executor::constants::FP3_MUL_SYSCALL_NUMBER;
use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use smallvec::smallvec;
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
#[cfg(feature = "prove")]
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices
// =========================================================================

pub mod cols {
    /// CPU timestamp (DWordWL low word; the high word is always 0 for the VM).
    pub const TIMESTAMP: usize = 0;

    // ---- pointers, each as DWordWL [lo32, hi32] ----------------------------
    /// result_ptr (a0) low / high 32 bits
    pub const RESULT_PTR_0: usize = 1;
    pub const RESULT_PTR_1: usize = 2;
    /// lhs_ptr (a1) low / high 32 bits
    pub const LHS_PTR_0: usize = 3;
    pub const LHS_PTR_1: usize = 4;
    /// rhs_ptr (a2) low / high 32 bits
    pub const RHS_PTR_0: usize = 5;
    pub const RHS_PTR_1: usize = 6;

    // ---- byte arrays for the 9 doublewords ---------------------------------
    // Each doubleword is 8 little-endian bytes. Reads carry one byte array
    // (old == value); the result writes carry both the new bytes and the prior
    // memory bytes (old).
    /// lhs[i] value bytes (read): LHS_BYTES + i*8 + b
    pub const LHS_BYTES: usize = 7; // 3 * 8 = 24
    /// rhs[i] value bytes (read): RHS_BYTES + i*8 + b
    pub const RHS_BYTES: usize = LHS_BYTES + 24; // 31
    /// result[i] new value bytes (write): RESULT_BYTES + i*8 + b
    pub const RESULT_BYTES: usize = RHS_BYTES + 24; // 55
    /// result[i] old (prior memory) bytes: RESULT_OLD_BYTES + i*8 + b
    pub const RESULT_OLD_BYTES: usize = RESULT_BYTES + 24; // 79

    // ---- field-element columns for the multiply constraint -----------------
    /// lhs limb 0 (a0)
    pub const A0: usize = RESULT_OLD_BYTES + 24; // 103
    /// lhs limb 1 (a1)
    pub const A1: usize = A0 + 1;
    /// lhs limb 2 (a2)
    pub const A2: usize = A0 + 2;
    /// rhs limb 0 (b0)
    pub const B0: usize = A0 + 3;
    /// rhs limb 1 (b1)
    pub const B1: usize = A0 + 4;
    /// rhs limb 2 (b2)
    pub const B2: usize = A0 + 5;
    /// result limb 0 (c0)
    pub const C0: usize = A0 + 6;
    /// result limb 1 (c1)
    pub const C1: usize = A0 + 7;
    /// result limb 2 (c2)
    pub const C2: usize = A0 + 8;

    /// Multiplicity flag (1 on real rows, 0 on padding).
    pub const MU: usize = C2 + 1; // 112

    pub const NUM_COLUMNS: usize = MU + 1; // 113

    // ---- index helpers ------------------------------------------------------

    #[inline]
    pub const fn lhs_byte(i: usize, b: usize) -> usize {
        LHS_BYTES + i * 8 + b
    }
    #[inline]
    pub const fn rhs_byte(i: usize, b: usize) -> usize {
        RHS_BYTES + i * 8 + b
    }
    #[inline]
    pub const fn result_byte(i: usize, b: usize) -> usize {
        RESULT_BYTES + i * 8 + b
    }
    #[inline]
    pub const fn result_old_byte(i: usize, b: usize) -> usize {
        RESULT_OLD_BYTES + i * 8 + b
    }
}

// =========================================================================
// Operation struct (used for trace generation)
// =========================================================================

/// One Fp3Mul syscall invocation. Carries everything the table row needs.
#[derive(Debug, Clone)]
pub struct Fp3MulOperation {
    /// CPU timestamp for this instruction (from the executor Log).
    pub timestamp: u64,
    /// result_ptr (a0).
    pub result_ptr: u64,
    /// lhs_ptr (a1).
    pub lhs_ptr: u64,
    /// rhs_ptr (a2).
    pub rhs_ptr: u64,
    /// lhs field element [a0, a1, a2].
    pub lhs: [u64; 3],
    /// rhs field element [b0, b1, b2].
    pub rhs: [u64; 3],
    /// result field element [c0, c1, c2] (computed by the prover).
    pub result: [u64; 3],
    /// Prior memory contents at result_ptr+{0,8,16} (8 bytes each, little-endian).
    pub result_old: [u64; 3],
}

// =========================================================================
// Trace generation (feature-gated)
// =========================================================================

#[cfg(feature = "prove")]
#[inline]
fn byte_of(val: u64, b: usize) -> u64 {
    (val >> (b * 8)) & 0xFF
}

#[cfg(feature = "prove")]
/// Generate the FP3_MUL trace table from a list of operations.
///
/// Each operation occupies one row. Padding rows are all-zero (MU = 0 gates
/// every bus interaction).
pub fn generate_fp3_mul_trace(
    ops: &[Fp3MulOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n_rows = ops.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); n_rows * cols::NUM_COLUMNS];

    for (row, op) in ops.iter().enumerate() {
        let base = row * cols::NUM_COLUMNS;

        // timestamp (low word; high word is always 0 for VM timestamps)
        data[base + cols::TIMESTAMP] = FE::from(op.timestamp & 0xFFFF_FFFF);

        // pointers as DWordWL
        data[base + cols::RESULT_PTR_0] = FE::from(op.result_ptr & 0xFFFF_FFFF);
        data[base + cols::RESULT_PTR_1] = FE::from(op.result_ptr >> 32);
        data[base + cols::LHS_PTR_0] = FE::from(op.lhs_ptr & 0xFFFF_FFFF);
        data[base + cols::LHS_PTR_1] = FE::from(op.lhs_ptr >> 32);
        data[base + cols::RHS_PTR_0] = FE::from(op.rhs_ptr & 0xFFFF_FFFF);
        data[base + cols::RHS_PTR_1] = FE::from(op.rhs_ptr >> 32);

        // byte arrays for the 9 doublewords
        for i in 0..3 {
            for b in 0..8 {
                data[base + cols::lhs_byte(i, b)] = FE::from(byte_of(op.lhs[i], b));
                data[base + cols::rhs_byte(i, b)] = FE::from(byte_of(op.rhs[i], b));
                data[base + cols::result_byte(i, b)] = FE::from(byte_of(op.result[i], b));
                data[base + cols::result_old_byte(i, b)] = FE::from(byte_of(op.result_old[i], b));
            }
        }

        // field-element columns for the multiply constraint
        data[base + cols::A0] = FE::from(op.lhs[0]);
        data[base + cols::A1] = FE::from(op.lhs[1]);
        data[base + cols::A2] = FE::from(op.lhs[2]);
        data[base + cols::B0] = FE::from(op.rhs[0]);
        data[base + cols::B1] = FE::from(op.rhs[1]);
        data[base + cols::B2] = FE::from(op.rhs[2]);
        data[base + cols::C0] = FE::from(op.result[0]);
        data[base + cols::C1] = FE::from(op.result[1]);
        data[base + cols::C2] = FE::from(op.result[2]);

        // mu = 1 (real row)
        data[base + cols::MU] = FE::one();
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Pack a pointer's DWordWL [lo32, hi32] columns into the two address bus
/// elements the Memw receivers expect.
fn ptr_addr(lo_col: usize, hi_col: usize) -> (BusValue, BusValue) {
    (
        BusValue::Packed {
            start_column: lo_col,
            packing: Packing::Direct,
        },
        BusValue::Packed {
            start_column: hi_col,
            packing: Packing::Direct,
        },
    )
}

/// Append a Memw *read* sender (read-receiver format, 24 values).
///
/// `old`/`value` are the 8 byte/word columns (old == value for pure reads).
/// `addr_lo`/`addr_hi` are the DWordWL address bus elements. `is_register`,
/// `w2`/`w4`/`w8` are constant flags.
#[allow(clippy::too_many_arguments)]
fn push_memw_read(
    interactions: &mut Vec<BusInteraction>,
    old: &[BusValue; 8],
    value: &[BusValue; 8],
    is_register: u64,
    addr_lo: BusValue,
    addr_hi: BusValue,
    w2: u64,
    w4: u64,
    w8: u64,
) {
    let mut values: Vec<BusValue> = Vec::with_capacity(24);
    // old[8]
    for v in old.iter() {
        values.push(v.clone());
    }
    // is_register
    values.push(BusValue::constant(is_register));
    // base_address as DWordWL
    values.push(addr_lo);
    values.push(addr_hi);
    // value[8]
    for v in value.iter() {
        values.push(v.clone());
    }
    // timestamp [lo, hi=0]
    values.push(BusValue::Packed {
        start_column: cols::TIMESTAMP,
        packing: Packing::Direct,
    });
    values.push(BusValue::constant(0));
    // write flags
    values.push(BusValue::constant(w2));
    values.push(BusValue::constant(w4));
    values.push(BusValue::constant(w8));

    interactions.push(BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(cols::MU),
        values,
    ));
}

/// Bus interactions for the FP3_MUL table.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let syscall_lo = FP3_MUL_SYSCALL_NUMBER & 0xFFFF_FFFF;
    let syscall_hi = FP3_MUL_SYSCALL_NUMBER >> 32;
    // 1 ecall + 3 register reads + 6 input reads + 3 output writes = 13
    let mut interactions = Vec::with_capacity(13);

    // 1. ECALL receiver (shared bus). Payload matches the CPU ECALL sender:
    //    [ts_lo, ts_hi=0, syscall_lo32, syscall_hi32].
    interactions.push(BusInteraction::receiver(
        BusId::Ecall,
        Multiplicity::Column(cols::MU),
        smallvec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            BusValue::constant(syscall_lo),
            BusValue::constant(syscall_hi),
        ],
    ));

    // 2. Register reads x10/x11/x12 (result_ptr/lhs_ptr/rhs_ptr).
    //    Register values are packed as [lo32, hi32, 0, 0, 0, 0, 0, 0] (DWordWL),
    //    matching `pack_register_value`. Width = 2 (write2 = 1). old == value.
    let zero = BusValue::constant(0);
    for &(lo_col, hi_col, reg) in &[
        (cols::RESULT_PTR_0, cols::RESULT_PTR_1, 10u64),
        (cols::LHS_PTR_0, cols::LHS_PTR_1, 11u64),
        (cols::RHS_PTR_0, cols::RHS_PTR_1, 12u64),
    ] {
        let lo = BusValue::Packed {
            start_column: lo_col,
            packing: Packing::Direct,
        };
        let hi = BusValue::Packed {
            start_column: hi_col,
            packing: Packing::Direct,
        };
        let reg_val: [BusValue; 8] = [
            lo.clone(),
            hi.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
        ];
        // Register address is the constant 2*reg, as DWordWL [2*reg, 0].
        push_memw_read(
            &mut interactions,
            &reg_val,
            &reg_val,
            1, // is_register
            BusValue::constant(2 * reg),
            BusValue::constant(0),
            1, // w2
            0,
            0,
        );
    }

    // 3. Memory reads for lhs[0..2] and rhs[0..2] (width 8, is_register = 0).
    //    old == value (pure read). Address = ptr + 8*i as DWordWL.
    for (ptr_lo, ptr_hi, byte_fn) in [
        (
            cols::LHS_PTR_0,
            cols::LHS_PTR_1,
            cols::lhs_byte as fn(usize, usize) -> usize,
        ),
        (
            cols::RHS_PTR_0,
            cols::RHS_PTR_1,
            cols::rhs_byte as fn(usize, usize) -> usize,
        ),
    ] {
        for i in 0..3usize {
            let bytes: [BusValue; 8] = core::array::from_fn(|b| BusValue::Packed {
                start_column: byte_fn(i, b),
                packing: Packing::Direct,
            });
            let (addr_lo, addr_hi) = mem_addr(ptr_lo, ptr_hi, i);
            push_memw_read(
                &mut interactions,
                &bytes,
                &bytes,
                0, // memory
                addr_lo,
                addr_hi,
                0,
                0,
                1, // w8
            );
        }
    }

    // 4. Memory writes for result[0..2] (width 8). old = prior memory bytes,
    //    value = computed result bytes. Modelled as keccak does: a single
    //    read-format Memw token with old != value (is_read = true on the op).
    for i in 0..3usize {
        let old_bytes: [BusValue; 8] = core::array::from_fn(|b| BusValue::Packed {
            start_column: cols::result_old_byte(i, b),
            packing: Packing::Direct,
        });
        let new_bytes: [BusValue; 8] = core::array::from_fn(|b| BusValue::Packed {
            start_column: cols::result_byte(i, b),
            packing: Packing::Direct,
        });
        let (addr_lo, addr_hi) = mem_addr(cols::RESULT_PTR_0, cols::RESULT_PTR_1, i);
        push_memw_read(
            &mut interactions,
            &old_bytes,
            &new_bytes,
            0,
            addr_lo,
            addr_hi,
            0,
            0,
            1, // w8
        );
    }

    interactions
}

/// Build the DWordWL address bus elements for `ptr + 8*i`, where `ptr` lives in
/// the (lo, hi) columns. Because every pointer the precompile uses is 8-byte
/// aligned and `8*i <= 16`, the low word never carries into the high word for
/// any realistic address, so we fold the `+8*i` into the low-word linear term.
fn mem_addr(ptr_lo: usize, ptr_hi: usize, i: usize) -> (BusValue, BusValue) {
    if i == 0 {
        return ptr_addr(ptr_lo, ptr_hi);
    }
    let offset = (8 * i) as i64;
    let addr_lo = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: ptr_lo,
        },
        LinearTerm::Constant(offset),
    ]);
    let addr_hi = BusValue::Packed {
        start_column: ptr_hi,
        packing: Packing::Direct,
    };
    (addr_lo, addr_hi)
}

// =========================================================================
// Constraints
// =========================================================================

/// Which of the three Fp3 multiply output constraints this instance checks.
#[derive(Debug, Clone, Copy)]
pub enum Fp3MulConstraintKind {
    /// c0 = a0·b0 + 2·a1·b2 + 2·a2·b1
    C0,
    /// c1 = a0·b1 + a1·b0 + 2·a2·b2
    C1,
    /// c2 = a0·b2 + a1·b1 + a2·b0
    C2,
}

/// A single constraint for the FP3_MUL table.
pub struct Fp3MulConstraint {
    kind: Fp3MulConstraintKind,
    constraint_idx: usize,
}

impl Fp3MulConstraint {
    pub fn new(kind: Fp3MulConstraintKind, constraint_idx: usize) -> Self {
        Self { kind, constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let a0 = step.get_main_evaluation_element(0, cols::A0).clone();
        let a1 = step.get_main_evaluation_element(0, cols::A1).clone();
        let a2 = step.get_main_evaluation_element(0, cols::A2).clone();
        let b0 = step.get_main_evaluation_element(0, cols::B0).clone();
        let b1 = step.get_main_evaluation_element(0, cols::B1).clone();
        let b2 = step.get_main_evaluation_element(0, cols::B2).clone();
        let c0 = step.get_main_evaluation_element(0, cols::C0).clone();
        let c1 = step.get_main_evaluation_element(0, cols::C1).clone();
        let c2 = step.get_main_evaluation_element(0, cols::C2).clone();

        let two = FieldElement::<F>::from(2u64);

        match self.kind {
            // c0 - (a0*b0 + 2*a1*b2 + 2*a2*b1) = 0
            Fp3MulConstraintKind::C0 => c0 - (&a0 * &b0 + &two * &a1 * &b2 + &two * &a2 * &b1),
            // c1 - (a0*b1 + a1*b0 + 2*a2*b2) = 0
            Fp3MulConstraintKind::C1 => c1 - (&a0 * &b1 + &a1 * &b0 + &two * &a2 * &b2),
            // c2 - (a0*b2 + a1*b1 + a2*b0) = 0
            Fp3MulConstraintKind::C2 => c2 - (&a0 * &b2 + &a1 * &b1 + &a2 * &b0),
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for Fp3MulConstraint {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        self.compute(step)
    }
}

/// Create all three constraints for the FP3_MUL table.
///
/// Returns `(constraints, next_constraint_idx)`.
pub fn create_constraints(
    constraint_idx_start: usize,
) -> (
    Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    let mut constraints: Vec<
        Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>,
    > = Vec::with_capacity(3);

    let mut idx = constraint_idx_start;
    constraints.push(Fp3MulConstraint::new(Fp3MulConstraintKind::C0, idx).boxed());
    idx += 1;
    constraints.push(Fp3MulConstraint::new(Fp3MulConstraintKind::C1, idx).boxed());
    idx += 1;
    constraints.push(Fp3MulConstraint::new(Fp3MulConstraintKind::C2, idx).boxed());
    idx += 1;

    (constraints, idx)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_count() {
        assert_eq!(cols::NUM_COLUMNS, 113);
    }

    #[test]
    fn test_constraint_count() {
        let (constraints, next_idx) = create_constraints(0);
        assert_eq!(constraints.len(), 3);
        assert_eq!(next_idx, 3);
    }

    #[test]
    fn test_bus_interaction_count() {
        // 1 ecall receiver + 3 register reads + 6 input reads + 3 output writes
        assert_eq!(bus_interactions().len(), 13);
    }
}

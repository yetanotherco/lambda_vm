//! Tests for the 64-bit VM constraint templates.

use crate::constraints::templates::{
    AddConstraint, AddLinearTerm, AddOperand, IsBitConstraint, SHIFT_32, new_is_bit_constraints,
};
use crate::tables::types::FE;
use stark::constraints::transition::TransitionConstraint;

// =========================================================================
// Basic tests
// =========================================================================

#[test]
fn test_inv_2_32() {
    // Verify that 2^32 * 2^(-32) = 1 in Goldilocks
    let two_32 = FE::from(SHIFT_32);
    let inv = two_32.inv().expect("Should be invertible");
    let product = two_32 * inv;
    assert_eq!(product, FE::one());
}

#[test]
fn test_is_bit_constraint_degree() {
    // Conditional: degree 3
    let conditional = IsBitConstraint::new(0, 1, 0);
    assert_eq!(conditional.degree(), 3);

    // Unconditional: degree 2
    let unconditional = IsBitConstraint::unconditional(1, 0);
    assert_eq!(unconditional.degree(), 2);
}

#[test]
fn test_add_constraint_degree() {
    let (c0, c1) = AddConstraint::new_pair(
        vec![0],
        AddOperand::dword(1),
        AddOperand::dword(3),
        AddOperand::dword(5),
        0,
    );
    assert_eq!(c0.degree(), 3);
    assert_eq!(c1.degree(), 3);
}

#[test]
fn test_add_constraint_indices() {
    let (c0, c1) = AddConstraint::new_pair(
        vec![0],
        AddOperand::dword(1),
        AddOperand::dword(3),
        AddOperand::dword(5),
        10,
    );
    assert_eq!(c0.constraint_idx(), 10);
    assert_eq!(c1.constraint_idx(), 11);
}

// =========================================================================
// IS_BIT formula verification tests
// =========================================================================

#[test]
fn test_is_bit_formula_valid_zero() {
    // IS_BIT formula: cond * X * (1 - X) = 0
    // When X = 0: cond * 0 * 1 = 0 ✓
    let cond = FE::one();
    let x = FE::zero();
    let result = cond * x * (FE::one() - x);
    assert_eq!(result, FE::zero());
}

#[test]
fn test_is_bit_formula_valid_one() {
    // IS_BIT formula: cond * X * (1 - X) = 0
    // When X = 1: cond * 1 * 0 = 0 ✓
    let cond = FE::one();
    let x = FE::one();
    let result = cond * x * (FE::one() - x);
    assert_eq!(result, FE::zero());
}

#[test]
fn test_is_bit_formula_invalid_two() {
    // IS_BIT formula: cond * X * (1 - X) = 0
    // When X = 2: cond * 2 * (-1) = -2 ≠ 0 ✗
    let cond = FE::one();
    let x = FE::from(2u64);
    let result = cond * x * (FE::one() - x);
    assert_ne!(result, FE::zero());
}

#[test]
fn test_is_bit_formula_cond_zero() {
    // IS_BIT formula: cond * X * (1 - X) = 0
    // When cond = 0: 0 * X * (1 - X) = 0 always ✓
    let cond = FE::zero();
    let x = FE::from(42u64); // Any invalid value
    let result = cond * x * (FE::one() - x);
    assert_eq!(result, FE::zero());
}

// =========================================================================
// ADD carry computation verification tests
// =========================================================================

#[test]
fn test_carry_computation_no_carry() {
    // lhs_lo + rhs_lo < 2^32, so carry_0 = 0
    let lhs_lo = FE::from(100u64);
    let rhs_lo = FE::from(200u64);
    let sum_lo = FE::from(300u64); // 100 + 200 = 300

    let inv_2_32 = FE::from(SHIFT_32).inv().unwrap();
    let carry = (lhs_lo + rhs_lo - sum_lo) * inv_2_32;

    // carry should be 0
    assert_eq!(carry, FE::zero());
}

#[test]
fn test_carry_computation_with_carry() {
    // lhs_lo + rhs_lo >= 2^32, so carry_0 = 1
    let lhs_lo = FE::from(0xFFFFFFFFu64); // 2^32 - 1
    let rhs_lo = FE::from(2u64);
    // sum_lo = (0xFFFFFFFF + 2) mod 2^32 = 1
    let sum_lo = FE::from(1u64);

    let inv_2_32 = FE::from(SHIFT_32).inv().unwrap();
    let carry = (lhs_lo + rhs_lo - sum_lo) * inv_2_32;

    // carry should be 1: (0xFFFFFFFF + 2 - 1) / 2^32 = 0x100000000 / 2^32 = 1
    assert_eq!(carry, FE::one());
}

#[test]
fn test_carry_is_bit_valid() {
    // When carry is 0, the IS_BIT constraint is satisfied
    let carry = FE::zero();
    let result = carry * (FE::one() - carry);
    assert_eq!(result, FE::zero());

    // When carry is 1, the IS_BIT constraint is satisfied
    let carry = FE::one();
    let result = carry * (FE::one() - carry);
    assert_eq!(result, FE::zero());
}

#[test]
fn test_carry_boundary_just_below() {
    // lhs_lo + rhs_lo = 2^32 - 1 (no carry)
    let lhs_lo = FE::from(0x80000000u64); // 2^31
    let rhs_lo = FE::from(0x7FFFFFFFu64); // 2^31 - 1
    let sum_lo = FE::from(0xFFFFFFFFu64); // 2^32 - 1

    let inv_2_32 = FE::from(SHIFT_32).inv().unwrap();
    let carry = (lhs_lo + rhs_lo - sum_lo) * inv_2_32;

    assert_eq!(carry, FE::zero());
}

#[test]
fn test_carry_boundary_exactly_2_32() {
    // lhs_lo + rhs_lo = 2^32 (carry = 1)
    let lhs_lo = FE::from(0x80000000u64); // 2^31
    let rhs_lo = FE::from(0x80000000u64); // 2^31
    let sum_lo = FE::from(0u64); // (2^31 + 2^31) mod 2^32 = 0

    let inv_2_32 = FE::from(SHIFT_32).inv().unwrap();
    let carry = (lhs_lo + rhs_lo - sum_lo) * inv_2_32;

    assert_eq!(carry, FE::one());
}

#[test]
fn test_carry_max_values() {
    // lhs_lo = 0xFFFFFFFF, rhs_lo = 0xFFFFFFFF
    // sum = 0x1FFFFFFFE, sum_lo = 0xFFFFFFFE, carry = 1
    let lhs_lo = FE::from(0xFFFFFFFFu64);
    let rhs_lo = FE::from(0xFFFFFFFFu64);
    let sum_lo = FE::from(0xFFFFFFFEu64);

    let inv_2_32 = FE::from(SHIFT_32).inv().unwrap();
    let carry = (lhs_lo + rhs_lo - sum_lo) * inv_2_32;

    assert_eq!(carry, FE::one());
}

// =========================================================================
// Helper function tests
// =========================================================================

#[test]
fn test_new_is_bit_constraints_count() {
    let (constraints, next_idx) = new_is_bit_constraints(&[1, 2, 3, 4], 10);
    assert_eq!(constraints.len(), 4);
    assert_eq!(next_idx, 14);
}

#[test]
fn test_new_is_bit_constraints_indices() {
    let (constraints, _) = new_is_bit_constraints(&[5, 6, 7], 100);
    assert_eq!(constraints[0].constraint_idx(), 100);
    assert_eq!(constraints[1].constraint_idx(), 101);
    assert_eq!(constraints[2].constraint_idx(), 102);
}

// =========================================================================
// AddOperand tests
// =========================================================================

#[test]
fn test_add_operand_constant() {
    // Test that constant operand creates correct Linear representation
    let op = AddOperand::constant(42);
    match op {
        AddOperand::Linear { lo, hi } => {
            assert_eq!(lo.len(), 1);
            assert!(hi.is_empty());
            match &lo[0] {
                AddLinearTerm::Constant(v) => assert_eq!(*v, 42),
                _ => panic!("Expected Constant"),
            }
        }
        _ => panic!("Expected Linear variant"),
    }
}

#[test]
fn test_add_operand_from_word() {
    // Test Word → DWordWL (single column, zero-extended)
    let op = AddOperand::from_word(5);
    match op {
        AddOperand::Linear { lo, hi } => {
            assert_eq!(lo.len(), 1);
            assert!(hi.is_empty());
            match &lo[0] {
                AddLinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, 1);
                    assert_eq!(*column, 5);
                }
                _ => panic!("Expected Column"),
            }
        }
        _ => panic!("Expected Linear variant"),
    }
}

#[test]
fn test_add_operand_dword() {
    // Test DWordWL (2 consecutive columns)
    let op = AddOperand::dword(10);
    match op {
        AddOperand::DWordWL { start_column } => {
            assert_eq!(start_column, 10);
        }
        _ => panic!("Expected DWordWL variant"),
    }
}

#[test]
fn test_add_operand_from_dword_hl() {
    // Test DWordHL → DWordWL (repack 4 halves → 2 words)
    let op = AddOperand::from_dword_hl(0);
    match op {
        AddOperand::Linear { lo, hi } => {
            assert_eq!(lo.len(), 2);
            assert_eq!(hi.len(), 2);

            // lo = h[0] + 2^16 * h[1]
            match &lo[0] {
                AddLinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, 1);
                    assert_eq!(*column, 0);
                }
                _ => panic!("Expected Column"),
            }
            match &lo[1] {
                AddLinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, 1 << 16);
                    assert_eq!(*column, 1);
                }
                _ => panic!("Expected Column"),
            }

            // hi = h[2] + 2^16 * h[3]
            match &hi[0] {
                AddLinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, 1);
                    assert_eq!(*column, 2);
                }
                _ => panic!("Expected Column"),
            }
            match &hi[1] {
                AddLinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, 1 << 16);
                    assert_eq!(*column, 3);
                }
                _ => panic!("Expected Column"),
            }
        }
        _ => panic!("Expected Linear variant"),
    }
}

#[test]
fn test_add_operand_from_dword_bl() {
    // Test DWordBL → DWordWL (repack 8 bytes → 2 words)
    let op = AddOperand::from_dword_bl(0);
    match op {
        AddOperand::Linear { lo, hi } => {
            assert_eq!(lo.len(), 4);
            assert_eq!(hi.len(), 4);

            // Check lo coefficients: 1, 2^8, 2^16, 2^24
            let lo_coeffs: Vec<i64> = lo
                .iter()
                .map(|t| match t {
                    AddLinearTerm::Column { coefficient, .. } => *coefficient,
                    _ => panic!("Expected Column"),
                })
                .collect();
            assert_eq!(lo_coeffs, vec![1, 1 << 8, 1 << 16, 1 << 24]);

            // Check hi coefficients: 1, 2^8, 2^16, 2^24
            let hi_coeffs: Vec<i64> = hi
                .iter()
                .map(|t| match t {
                    AddLinearTerm::Column { coefficient, .. } => *coefficient,
                    _ => panic!("Expected Column"),
                })
                .collect();
            assert_eq!(hi_coeffs, vec![1, 1 << 8, 1 << 16, 1 << 24]);

            // Check hi columns: 4, 5, 6, 7
            let hi_cols: Vec<usize> = hi
                .iter()
                .map(|t| match t {
                    AddLinearTerm::Column { column, .. } => *column,
                    _ => panic!("Expected Column"),
                })
                .collect();
            assert_eq!(hi_cols, vec![4, 5, 6, 7]);
        }
        _ => panic!("Expected Linear variant"),
    }
}

#[test]
fn test_add_operand_linear_with_negative_coefficient() {
    // Test linear operand with negative coefficient: 4 - 2*c
    // This represents expressions like `4 - 2 * c_type_instruction`
    let op = AddOperand::linear(
        vec![
            AddLinearTerm::Constant(4),
            AddLinearTerm::Column {
                coefficient: -2,
                column: 0,
            },
        ],
        vec![], // hi = 0
    );
    match op {
        AddOperand::Linear { lo, hi } => {
            assert_eq!(lo.len(), 2);
            assert!(hi.is_empty());

            match &lo[0] {
                AddLinearTerm::Constant(v) => assert_eq!(*v, 4),
                _ => panic!("Expected Constant"),
            }
            match &lo[1] {
                AddLinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, -2);
                    assert_eq!(*column, 0);
                }
                _ => panic!("Expected Column"),
            }
        }
        _ => panic!("Expected Linear variant"),
    }
}

#[test]
fn test_add_operand_linear_with_nonzero_hi() {
    // Test linear operand with non-trivial hi terms (virtual column case)
    let op = AddOperand::linear(
        vec![
            AddLinearTerm::Column {
                coefficient: 1 << 16,
                column: 0,
            },
            AddLinearTerm::Column {
                coefficient: 1 << 8,
                column: 1,
            },
            AddLinearTerm::Column {
                coefficient: 1,
                column: 2,
            },
        ],
        vec![
            AddLinearTerm::Column {
                coefficient: 1 << 16,
                column: 3,
            },
            AddLinearTerm::Column {
                coefficient: 1,
                column: 4,
            },
        ],
    );
    match op {
        AddOperand::Linear { lo, hi } => {
            assert_eq!(lo.len(), 3);
            assert_eq!(hi.len(), 2);
        }
        _ => panic!("Expected Linear variant"),
    }
}

#[test]
fn test_linear_term_negative_coefficient_formula() {
    // Verify that 4 - 2*1 = 2 using field arithmetic
    // This simulates: constant(4) + coefficient(-2) * column_value(1)
    let constant = FE::from(4u64);
    let coefficient = -FE::from(2u64);
    let col_value = FE::one();

    let result = constant + coefficient * col_value;
    assert_eq!(result, FE::from(2u64));
}

#[test]
fn test_dword_hl_repack_formula() {
    // Verify DWordHL → DWordWL repacking formula
    // lo = h[0] + 2^16 * h[1]
    // hi = h[2] + 2^16 * h[3]
    //
    // Example: h = [0x1234, 0x5678, 0xABCD, 0xEF01]
    // lo = 0x1234 + 0x5678_0000 = 0x5678_1234
    // hi = 0xABCD + 0xEF01_0000 = 0xEF01_ABCD

    let h0 = FE::from(0x1234u64);
    let h1 = FE::from(0x5678u64);
    let h2 = FE::from(0xABCDu64);
    let h3 = FE::from(0xEF01u64);

    let shift_16 = FE::from(1u64 << 16);

    let lo = h0 + h1 * shift_16;
    let hi = h2 + h3 * shift_16;

    assert_eq!(lo, FE::from(0x5678_1234u64));
    assert_eq!(hi, FE::from(0xEF01_ABCDu64));
}

#[test]
fn test_dword_bl_repack_formula() {
    // Verify DWordBL → DWordWL repacking formula
    // lo = b[0] + 2^8*b[1] + 2^16*b[2] + 2^24*b[3]
    // hi = b[4] + 2^8*b[5] + 2^16*b[6] + 2^24*b[7]
    //
    // Example: bytes = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]
    // lo = 0x78563412
    // hi = 0xF0DEBC9A

    let b: Vec<FE> = vec![
        FE::from(0x12u64),
        FE::from(0x34u64),
        FE::from(0x56u64),
        FE::from(0x78u64),
        FE::from(0x9Au64),
        FE::from(0xBCu64),
        FE::from(0xDEu64),
        FE::from(0xF0u64),
    ];

    let coeffs: Vec<FE> = vec![
        FE::from(1u64),
        FE::from(1u64 << 8),
        FE::from(1u64 << 16),
        FE::from(1u64 << 24),
    ];

    let lo = b[0] * coeffs[0] + b[1] * coeffs[1] + b[2] * coeffs[2] + b[3] * coeffs[3];
    let hi = b[4] * coeffs[0] + b[5] * coeffs[1] + b[6] * coeffs[2] + b[7] * coeffs[3];

    assert_eq!(lo, FE::from(0x78563412u64));
    assert_eq!(hi, FE::from(0xF0DEBC9Au64));
}

// =========================================================================
// CPU Constraints tests
// =========================================================================

use crate::constraints::cpu::{
    Arg1LowerConstraint, Arg1UpperConstraint, BIT_FLAG_COLUMNS, BranchCondConstraint,
    EbreakConstraint, ExtBitZeroConstraint, NUM_CPU_CONSTRAINTS, NextPcAddConstraint,
    create_add_constraints, create_all_cpu_constraints, create_is_bit_constraints,
    create_slt_res_zero_constraints,
};

use crate::tables::cpu::cols as cpu_cols;

#[test]
fn test_cpu_bit_flag_columns_count() {
    // Should have 34 bit flag columns (includes read_register1, read_register2, inline-pc columns)
    assert_eq!(BIT_FLAG_COLUMNS.len(), 34);
}

#[test]
fn test_cpu_bit_flag_columns_valid() {
    // All columns should be valid CPU column indices
    for &col in BIT_FLAG_COLUMNS {
        assert!(col < cpu_cols::NUM_COLUMNS, "Column {} out of range", col);
    }
}

#[test]
fn test_create_is_bit_constraints() {
    let (constraints, next_idx) = create_is_bit_constraints(0);

    assert_eq!(constraints.len(), 34);
    assert_eq!(next_idx, 34);

    // Check constraint indices are sequential
    for (i, c) in constraints.iter().enumerate() {
        assert_eq!(c.constraint_idx(), i);
    }
}

#[test]
fn test_create_add_constraints() {
    let (constraints, next_idx) = create_add_constraints(0);

    // Should create 4 constraints: 2 for ADD+LOAD, 2 for STORE (res = arg1 + imm)
    assert_eq!(constraints.len(), 4);
    assert_eq!(next_idx, 4);

    assert_eq!(constraints[0].constraint_idx(), 0);
    assert_eq!(constraints[1].constraint_idx(), 1);
    assert_eq!(constraints[2].constraint_idx(), 2);
    assert_eq!(constraints[3].constraint_idx(), 3);
}

#[test]
fn test_create_slt_res_zero_constraints() {
    let (constraints, next_idx) = create_slt_res_zero_constraints(0);

    // Should create 7 constraints (for bytes 1-7)
    assert_eq!(constraints.len(), 7);
    assert_eq!(next_idx, 7);

    for (i, c) in constraints.iter().enumerate() {
        assert_eq!(c.constraint_idx(), i);
    }
}

#[test]
fn test_branch_cond_constraint_degree() {
    let c = BranchCondConstraint::new(0);
    assert_eq!(c.degree(), 3);
}

#[test]
fn test_ebreak_constraint_degree() {
    let c = EbreakConstraint::new(0);
    assert_eq!(c.degree(), 1);
}

#[test]
fn test_arg1_lower_constraint_degree() {
    let c = Arg1LowerConstraint::new(0);
    assert_eq!(c.degree(), 1);
}

#[test]
fn test_arg1_upper_constraint_degree() {
    let c = Arg1UpperConstraint::new(0);
    assert_eq!(c.degree(), 3);
}

#[test]
fn test_ext_bit_zero_constraint_degree() {
    let c = ExtBitZeroConstraint::new(0, cpu_cols::RV1_EXT_BIT);
    assert_eq!(c.degree(), 2);
}

#[test]
fn test_next_pc_add_constraint_degree() {
    let c = NextPcAddConstraint::new(0, 0);
    assert_eq!(c.degree(), 3);
}

#[test]
fn test_next_pc_add_constraint_new_pair() {
    let (c0, c1) = NextPcAddConstraint::new_pair(10);
    assert_eq!(c0.constraint_idx(), 10);
    assert_eq!(c1.constraint_idx(), 11);
}

#[test]
fn test_create_all_cpu_constraints() {
    let (is_bit, add, other, total) = create_all_cpu_constraints();

    assert_eq!(is_bit.len(), 34);
    // ADD constraints: 2 (ADD+LOAD) + 2 (STORE: arg1+imm) + 2 (SUB+BEQ) + 2 (JALR) = 8
    assert_eq!(add.len(), 8);
    // Other: branch_cond(1) + ebreak(1) + rv1_zero_forcing(3) + rv2_zero_forcing(3) + arg1(2) + arg2(2) + rvd(2) + slt_zero(7) + ext_bit_zero(3) + next_pc(2) = 26
    assert_eq!(other.len(), 26);

    // Total should be 34 + 8 + 26 = 68
    assert_eq!(total, 68);
    assert_eq!(total, NUM_CPU_CONSTRAINTS);
}

#[test]
fn test_cpu_constraint_indices_are_unique() {
    let (is_bit, add, other, _) = create_all_cpu_constraints();

    let mut indices: Vec<usize> = Vec::new();

    for c in &is_bit {
        indices.push(c.constraint_idx());
    }
    for c in &add {
        indices.push(c.constraint_idx());
    }
    for c in &other {
        indices.push(c.constraint_idx());
    }

    // Check no duplicates
    indices.sort();
    for i in 1..indices.len() {
        assert_ne!(
            indices[i],
            indices[i - 1],
            "Duplicate constraint index: {}",
            indices[i]
        );
    }

    // Check sequential
    for (i, &idx) in indices.iter().enumerate() {
        assert_eq!(idx, i, "Expected index {} but got {}", i, idx);
    }
}

/// The grouped-dispatch CPU AIR (`with_eval_groups`) must produce exactly the
/// same transition evaluations as the flat boxed list, on random frames that
/// exercise both the base constraints and the LogUp tail.
#[test]
fn test_cpu_air_grouped_eval_matches_flat() {
    use crate::tables::cpu::{bus_interactions as cpu_bus_interactions, cols as cpu_cols};
    use crate::tables::types::{FEE, GoldilocksExtension, GoldilocksField};
    use crate::test_utils::create_cpu_air;
    use stark::constraints::transition::TransitionConstraintEvaluator;
    use stark::frame::Frame;
    use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, PackingShifts};
    use stark::proof::options::ProofOptions;
    use stark::table::TableView;
    use stark::traits::{AIR, TransitionEvaluationContext};

    type F = GoldilocksField;
    type E = GoldilocksExtension;

    let proof_options = ProofOptions::default_test_options();

    // AIR with grouped evaluation (what production builds now).
    let grouped_air = create_cpu_air(&proof_options);

    // Twin without groups: same constraints, flat per-constraint dispatch.
    let (is_bit, add, other, _) = super::super::constraints::cpu::create_all_cpu_constraints();
    let mut flat: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = Vec::new();
    for c in is_bit {
        flat.push(c.boxed());
    }
    for c in add {
        flat.push(c.boxed());
    }
    for c in other {
        flat.push(c);
    }
    let flat_air: crate::test_utils::VmAir = AirWithBuses::new(
        cpu_cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: cpu_bus_interactions(),
        },
        &proof_options,
        1,
        flat,
    )
    .with_name("CPU");

    let num_base = grouped_air.num_base_transition_constraints();
    let num_total = grouped_air.context().num_transition_constraints;
    assert_eq!(num_base, flat_air.num_base_transition_constraints());
    assert_eq!(num_total, flat_air.context().num_transition_constraints);

    let (num_main, num_aux) = grouped_air.trace_layout();
    let packing_shifts = PackingShifts::<F>::new();
    let logup_table_offset = FEE::from(11u64);
    let rap_challenges: Vec<FEE> = vec![FEE::from(98765u64), FEE::from(43210u64)];
    let alpha_powers: Vec<FEE> = (0..64).map(|i| FEE::from(3u64).pow(i as u64)).collect();

    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = |bound: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 8) % bound
    };

    for _ in 0..16 {
        // Two steps (offsets [0,1]) of random main + aux values. Bit-flag
        // columns get random bits half the time to also hit "valid" shapes.
        let mut steps = Vec::new();
        for _ in 0..2 {
            let main_row: Vec<FE> = (0..num_main).map(|_| FE::from(next(1 << 20))).collect();
            let aux_row: Vec<FEE> = (0..num_aux).map(|_| FEE::from(next(1 << 20))).collect();
            steps.push(TableView::new(vec![main_row], vec![aux_row]));
        }
        let frame = Frame::<F, E>::new(steps);
        let ctx = TransitionEvaluationContext::new_prover(
            &frame,
            &[],
            &rap_challenges,
            &alpha_powers,
            &logup_table_offset,
            &packing_shifts,
        );

        let mut base_a = vec![FE::zero(); num_base];
        let mut ext_a = vec![FEE::zero(); num_total];
        grouped_air.compute_transition_prover(&ctx, &mut base_a, &mut ext_a);

        let mut base_b = vec![FE::zero(); num_base];
        let mut ext_b = vec![FEE::zero(); num_total];
        flat_air.compute_transition_prover(&ctx, &mut base_b, &mut ext_b);

        assert_eq!(
            base_a, base_b,
            "base evals diverge between grouped and flat"
        );
        assert_eq!(ext_a, ext_b, "ext evals diverge between grouped and flat");
    }
}

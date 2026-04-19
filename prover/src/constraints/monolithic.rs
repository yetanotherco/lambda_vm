//! Monolithic constraint evaluators for each table.
//!
//! Instead of N separate `Box<dyn TransitionConstraint>::evaluate_prover()` vtable calls
//! per LDE point, each table provides ONE function that evaluates ALL base-field constraints
//! at once. The compiler can inline and optimize across all constraint expressions.
//!
//! LogUp constraints (extension-field) are still dispatched individually.

// Field element arithmetic uses reference patterns (`&x * &y`) throughout.
// This is idiomatic for Goldilocks field elements (Copy type) and matches the
// existing constraint code style.
#![allow(clippy::op_ref)]

use math::field::element::FieldElement;
use stark::table::TableView;

use crate::tables::types::{GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;
type FE = FieldElement<F>;
type TV = TableView<F, E>;

// =========================================================================
// IS_BIT helper
// =========================================================================

/// `x * (1 - x)` -- the core IS_BIT expression.
#[inline(always)]
fn is_bit(x: &FE) -> FE {
    x * &(FE::one() - x)
}

/// `cond * x * (1 - x)` -- conditional IS_BIT.
#[inline(always)]
fn is_bit_cond(cond: &FE, x: &FE) -> FE {
    cond * &(x * &(FE::one() - x))
}

/// Pack 4 byte columns into a 32-bit word.
#[inline(always)]
fn pack_bytes(step: &TV, c0: usize, c1: usize, c2: usize, c3: usize) -> FE {
    let b0 = step.get_main_evaluation_element(0, c0);
    let b1 = step.get_main_evaluation_element(0, c1);
    let b2 = step.get_main_evaluation_element(0, c2);
    let b3 = step.get_main_evaluation_element(0, c3);
    b0 + b1 * FE::from(1u64 << 8) + b2 * FE::from(1u64 << 16) + b3 * FE::from(1u64 << 24)
}

// =========================================================================
// MEMW_R (3 constraints)
// =========================================================================

/// Monolithic evaluator for MEMW_R (3 base constraints).
pub fn memw_register_eval(step: &TV, evals: &mut [FE]) {
    use crate::tables::memw_register::cols;

    let mu_read = *step.get_main_evaluation_element(0, cols::MU_READ);
    let mu_write = *step.get_main_evaluation_element(0, cols::MU_WRITE);

    // 0: IS_BIT(MU_READ)
    evals[0] = is_bit(&mu_read);
    // 1: IS_BIT(MU_WRITE)
    evals[1] = is_bit(&mu_write);
    // 2: IS_BIT(mu_sum)
    let mu_sum = &mu_read + &mu_write;
    evals[2] = is_bit(&mu_sum);
}

// =========================================================================
// MEMW_A (4 constraints)
// =========================================================================

/// Monolithic evaluator for MEMW_A (4 base constraints).
pub fn memw_aligned_eval(step: &TV, evals: &mut [FE]) {
    use crate::tables::memw_aligned::cols;

    let mu_read = *step.get_main_evaluation_element(0, cols::MU_READ);
    let mu_write = *step.get_main_evaluation_element(0, cols::MU_WRITE);
    let mu_sum = &mu_read + &mu_write;
    let one = FE::one();

    evals[0] = &mu_sum * &(&one - &mu_sum);
    let write2 = *step.get_main_evaluation_element(0, cols::WRITE2);
    let write4 = *step.get_main_evaluation_element(0, cols::WRITE4);
    let write8 = *step.get_main_evaluation_element(0, cols::WRITE8);
    let w2 = &write2 + &write4 + &write8;
    evals[1] = &w2 * &(&one - &mu_sum);
    evals[2] = is_bit(&mu_read);
    evals[3] = is_bit(&mu_write);
}

// =========================================================================
// MEMW (11 constraints)
// =========================================================================

/// Monolithic evaluator for MEMW (11 base constraints).
pub fn memw_eval(step: &TV, evals: &mut [FE]) {
    use crate::tables::memw::cols;

    let mu_read = *step.get_main_evaluation_element(0, cols::MU_READ);
    let mu_write = *step.get_main_evaluation_element(0, cols::MU_WRITE);
    let mu_sum = &mu_read + &mu_write;
    let one = FE::one();

    evals[0] = &mu_sum * &(&one - &mu_sum);
    let write2 = *step.get_main_evaluation_element(0, cols::WRITE2);
    let write4 = *step.get_main_evaluation_element(0, cols::WRITE4);
    let write8 = *step.get_main_evaluation_element(0, cols::WRITE8);
    let w2 = &write2 + &write4 + &write8;
    evals[1] = &w2 * &(&one - &mu_sum);
    evals[2] = is_bit(&mu_read);
    evals[3] = is_bit(&mu_write);
    for (i, &carry_col) in cols::CARRY.iter().enumerate() {
        let c = step.get_main_evaluation_element(0, carry_col);
        evals[4 + i] = is_bit(c);
    }
}

// =========================================================================
// LOAD (8 constraints)
// =========================================================================

/// Monolithic evaluator for LOAD (8 base constraints).
pub fn load_eval(step: &TV, evals: &mut [FE]) {
    use crate::tables::load::cols;

    let one = FE::one();
    let ff = FE::from(255u64);

    let mu = *step.get_main_evaluation_element(0, cols::MU);
    let read2 = *step.get_main_evaluation_element(0, cols::READ2);
    let read4 = *step.get_main_evaluation_element(0, cols::READ4);
    let read8 = *step.get_main_evaluation_element(0, cols::READ8);
    let signed = *step.get_main_evaluation_element(0, cols::SIGNED);
    let sign_bit = *step.get_main_evaluation_element(0, cols::SIGN_BIT);
    let expected = &signed * &sign_bit * &ff;

    let read_sum = &read2 + &read4 + &read8;
    evals[0] = &read_sum * &(&one - &mu);

    let factor_high = &one - &read8;
    for (j, i) in (4..8).enumerate() {
        let res_i = *step.get_main_evaluation_element(0, cols::RES[i]);
        evals[1 + j] = &factor_high * &(&res_i - &expected);
    }

    let factor_mid = &one - &read4 - &read8;
    for (j, i) in (2..4).enumerate() {
        let res_i = *step.get_main_evaluation_element(0, cols::RES[i]);
        evals[5 + j] = &factor_mid * &(&res_i - &expected);
    }

    let factor_low = &one - &read2 - &read4 - &read8;
    let res_1 = *step.get_main_evaluation_element(0, cols::RES[1]);
    evals[7] = &factor_low * &(&res_1 - &expected);
}

// =========================================================================
// BRANCH (4 constraints)
// =========================================================================

/// Monolithic evaluator for BRANCH (4 base constraints).
pub fn branch_eval(step: &TV, evals: &mut [FE]) {
    use crate::constraints::templates::INV_SHIFT_32;
    use crate::tables::branch::cols;

    let one = FE::one();
    let inv_2_32 = FE::from(INV_SHIFT_32);
    let shift_8 = FE::from(1u64 << 8);
    let shift_16 = FE::from(1u64 << 16);

    let jalr = *step.get_main_evaluation_element(0, cols::JALR);

    let unmasked_low_byte = *step.get_main_evaluation_element(0, cols::UNMASKED_LOW_BYTE);
    let next_pc_low_1 = *step.get_main_evaluation_element(0, cols::NEXT_PC_LOW_1);
    let next_pc_high_0 = *step.get_main_evaluation_element(0, cols::NEXT_PC_HIGH_0);
    let next_pc_high_1 = *step.get_main_evaluation_element(0, cols::NEXT_PC_HIGH_1);
    let next_pc_high_2 = *step.get_main_evaluation_element(0, cols::NEXT_PC_HIGH_2);
    let unmasked_0 = &unmasked_low_byte + &next_pc_low_1 * &shift_8 + &next_pc_high_0 * &shift_16;
    let unmasked_1 = &next_pc_high_1 + &next_pc_high_2 * &shift_16;

    let offset_0 = *step.get_main_evaluation_element(0, cols::OFFSET_0);
    let offset_1 = *step.get_main_evaluation_element(0, cols::OFFSET_1);

    let pc_0 = *step.get_main_evaluation_element(0, cols::PC_0);
    let pc_1 = *step.get_main_evaluation_element(0, cols::PC_1);
    let carry_0_pc = (&pc_0 + &offset_0 - &unmasked_0) * &inv_2_32;
    let carry_1_pc = (&pc_1 + &offset_1 + &carry_0_pc - &unmasked_1) * &inv_2_32;

    let reg_0 = *step.get_main_evaluation_element(0, cols::REGISTER_0);
    let reg_1 = *step.get_main_evaluation_element(0, cols::REGISTER_1);
    let carry_0_reg = (&reg_0 + &offset_0 - &unmasked_0) * &inv_2_32;
    let carry_1_reg = (&reg_1 + &offset_1 + &carry_0_reg - &unmasked_1) * &inv_2_32;

    let not_jalr = &one - &jalr;

    evals[0] = &not_jalr * &carry_0_pc * &(&one - &carry_0_pc);
    evals[1] = &not_jalr * &carry_1_pc * &(&one - &carry_1_pc);
    evals[2] = &jalr * &carry_0_reg * &(&one - &carry_0_reg);
    evals[3] = &jalr * &carry_1_reg * &(&one - &carry_1_reg);
}

// =========================================================================
// SHIFT (16 constraints)
// =========================================================================

/// Monolithic evaluator for SHIFT (16 base constraints).
pub fn shift_eval(step: &TV, evals: &mut [FE]) {
    use crate::tables::shift::shift_constraints;

    let (constraints, _) = shift_constraints(0);
    for c in &constraints {
        evals[c.constraint_idx()] = c.compute_monolithic(step);
    }
}

// =========================================================================
// DVRM (19 constraints)
// =========================================================================

/// Monolithic evaluator for DVRM (19 base constraints).
pub fn dvrm_eval(step: &TV, evals: &mut [FE]) {
    use crate::tables::dvrm::dvrm_constraints;

    let (constraints, _) = dvrm_constraints(0);
    for c in &constraints {
        evals[c.constraint_idx()] = c.compute_monolithic(step);
    }
}

// =========================================================================
// COMMIT (8 constraints)
// =========================================================================

/// Monolithic evaluator for COMMIT (8 base constraints).
pub fn commit_eval(step: &TV, evals: &mut [FE]) {
    use crate::tables::commit::cols;

    let one = FE::one();
    let inv_2_32 = FE::from(crate::constraints::templates::INV_SHIFT_32);

    let first = *step.get_main_evaluation_element(0, cols::FIRST);
    let end = *step.get_main_evaluation_element(0, cols::END);
    let mu = *step.get_main_evaluation_element(0, cols::MU);

    evals[0] = is_bit(&first);
    evals[1] = is_bit(&end);
    evals[2] = is_bit(&mu);

    evals[3] = (&first + &end) * &(&one - &mu);

    let addr_lo = *step.get_main_evaluation_element(0, cols::ADDRESS_0);
    let addr_hi = *step.get_main_evaluation_element(0, cols::ADDRESS_1);
    let shift_16 = FE::from(1u64 << 16);
    let incr_lo = step.get_main_evaluation_element(0, cols::ADDRESS_INCR_0)
        + step.get_main_evaluation_element(0, cols::ADDRESS_INCR_1) * &shift_16;
    let incr_hi = step.get_main_evaluation_element(0, cols::ADDRESS_INCR_2)
        + step.get_main_evaluation_element(0, cols::ADDRESS_INCR_3) * &shift_16;

    let carry_0_add = (&addr_lo + &one - &incr_lo) * &inv_2_32;
    let carry_1_add = (&addr_hi + &carry_0_add - &incr_hi) * &inv_2_32;
    evals[4] = is_bit(&carry_0_add);
    evals[5] = is_bit(&carry_1_add);

    let count_lo = *step.get_main_evaluation_element(0, cols::COUNT_0);
    let count_hi = *step.get_main_evaluation_element(0, cols::COUNT_1);
    let decr_lo = step.get_main_evaluation_element(0, cols::COUNT_DECR_0)
        + step.get_main_evaluation_element(0, cols::COUNT_DECR_1) * &shift_16;
    let decr_hi = step.get_main_evaluation_element(0, cols::COUNT_DECR_2)
        + step.get_main_evaluation_element(0, cols::COUNT_DECR_3) * &shift_16;

    let carry_0_sub = (&decr_lo + &one - &count_lo) * &inv_2_32;
    let carry_1_sub = (&decr_hi + &carry_0_sub - &count_hi) * &inv_2_32;
    evals[6] = is_bit(&carry_0_sub);
    evals[7] = is_bit(&carry_1_sub);
}

// =========================================================================
// CPU (66 constraints)
// =========================================================================

/// Monolithic evaluator for CPU (66 base constraints).
pub fn cpu_eval(step: &TV, evals: &mut [FE]) {
    use crate::tables::cpu::cols;

    let one = FE::one();
    let two = FE::from(2u64);
    let four = FE::from(4u64);
    let inv_2_32 = FE::from(crate::constraints::templates::INV_SHIFT_32);
    let shift_16 = FE::from(1u64 << 16);
    let mask_32 = FE::from((1u64 << 32) - 1);

    let read_register1 = *step.get_main_evaluation_element(0, cols::READ_REGISTER1);
    let read_register2 = *step.get_main_evaluation_element(0, cols::READ_REGISTER2);
    let write_register = *step.get_main_evaluation_element(0, cols::WRITE_REGISTER);
    let memory_2bytes = *step.get_main_evaluation_element(0, cols::MEMORY_2BYTES);
    let memory_4bytes = *step.get_main_evaluation_element(0, cols::MEMORY_4BYTES);
    let memory_8bytes = *step.get_main_evaluation_element(0, cols::MEMORY_8BYTES);
    let c_type = *step.get_main_evaluation_element(0, cols::C_TYPE_INSTRUCTION);
    let signed = *step.get_main_evaluation_element(0, cols::SIGNED);
    let mp_selector = *step.get_main_evaluation_element(0, cols::MP_SELECTOR);
    let muldiv_selector = *step.get_main_evaluation_element(0, cols::MULDIV_SELECTOR);
    let word_instr = *step.get_main_evaluation_element(0, cols::WORD_INSTR);
    let add_flag = *step.get_main_evaluation_element(0, cols::ADD);
    let sub_flag = *step.get_main_evaluation_element(0, cols::SUB);
    let slt = *step.get_main_evaluation_element(0, cols::SLT);
    let and_flag = *step.get_main_evaluation_element(0, cols::AND);
    let or_flag = *step.get_main_evaluation_element(0, cols::OR);
    let xor_flag = *step.get_main_evaluation_element(0, cols::XOR);
    let shift_flag = *step.get_main_evaluation_element(0, cols::SHIFT);
    let jalr = *step.get_main_evaluation_element(0, cols::JALR);
    let beq = *step.get_main_evaluation_element(0, cols::BEQ);
    let blt = *step.get_main_evaluation_element(0, cols::BLT);
    let load = *step.get_main_evaluation_element(0, cols::LOAD);
    let store = *step.get_main_evaluation_element(0, cols::STORE);
    let mul_flag = *step.get_main_evaluation_element(0, cols::MUL);
    let divrem = *step.get_main_evaluation_element(0, cols::DIVREM);
    let ecall = *step.get_main_evaluation_element(0, cols::ECALL);
    let ebreak = *step.get_main_evaluation_element(0, cols::EBREAK);
    let rv1_ext_bit = *step.get_main_evaluation_element(0, cols::RV1_EXT_BIT);
    let rv2_ext_bit = *step.get_main_evaluation_element(0, cols::RV2_EXT_BIT);
    let res_ext_bit = *step.get_main_evaluation_element(0, cols::RES_EXT_BIT);
    let is_equal = *step.get_main_evaluation_element(0, cols::IS_EQUAL);
    let branch_cond = *step.get_main_evaluation_element(0, cols::BRANCH_COND);

    // Constraints 0-31: IS_BIT
    let bit_cols = [
        &read_register1,
        &read_register2,
        &write_register,
        &memory_2bytes,
        &memory_4bytes,
        &memory_8bytes,
        &c_type,
        &signed,
        &mp_selector,
        &muldiv_selector,
        &word_instr,
        &add_flag,
        &sub_flag,
        &slt,
        &and_flag,
        &or_flag,
        &xor_flag,
        &shift_flag,
        &jalr,
        &beq,
        &blt,
        &load,
        &store,
        &mul_flag,
        &divrem,
        &ecall,
        &ebreak,
        &rv1_ext_bit,
        &rv2_ext_bit,
        &res_ext_bit,
        &is_equal,
        &branch_cond,
    ];
    for (i, col_val) in bit_cols.iter().enumerate() {
        evals[i] = is_bit(col_val);
    }
    let mut idx = 32;

    let arg1_lo = pack_bytes(step, cols::ARG1_0, cols::ARG1_1, cols::ARG1_2, cols::ARG1_3);
    let arg1_hi = pack_bytes(step, cols::ARG1_4, cols::ARG1_5, cols::ARG1_6, cols::ARG1_7);
    let arg2_lo = pack_bytes(
        step,
        cols::ARG2[0],
        cols::ARG2[1],
        cols::ARG2[2],
        cols::ARG2[3],
    );
    let arg2_hi = pack_bytes(
        step,
        cols::ARG2[4],
        cols::ARG2[5],
        cols::ARG2[6],
        cols::ARG2[7],
    );
    let res_lo = pack_bytes(step, cols::RES_0, cols::RES_1, cols::RES_2, cols::RES_3);
    let res_hi = pack_bytes(step, cols::RES_4, cols::RES_5, cols::RES_6, cols::RES_7);

    // 32-33: ADD carry (ADD + LOAD)
    let add_load_cond = &add_flag + &load;
    let add_carry_0 = (&arg1_lo + &arg2_lo - &res_lo) * &inv_2_32;
    let add_carry_1 = (&arg1_hi + &arg2_hi + &add_carry_0 - &res_hi) * &inv_2_32;
    evals[idx] = is_bit_cond(&add_load_cond, &add_carry_0);
    idx += 1;
    evals[idx] = is_bit_cond(&add_load_cond, &add_carry_1);
    idx += 1;

    // 34-35: STORE ADD carry
    let imm_0 = *step.get_main_evaluation_element(0, cols::IMM_0);
    let imm_1 = *step.get_main_evaluation_element(0, cols::IMM_1);
    let store_carry_0 = (&arg1_lo + &imm_0 - &res_lo) * &inv_2_32;
    let store_carry_1 = (&arg1_hi + &imm_1 + &store_carry_0 - &res_hi) * &inv_2_32;
    evals[idx] = is_bit_cond(&store, &store_carry_0);
    idx += 1;
    evals[idx] = is_bit_cond(&store, &store_carry_1);
    idx += 1;

    // 36-37: SUB carry (SUB + BEQ)
    let sub_beq_cond = &sub_flag + &beq;
    let sub_carry_0 = (&arg2_lo + &res_lo - &arg1_lo) * &inv_2_32;
    let sub_carry_1 = (&arg2_hi + &res_hi + &sub_carry_0 - &arg1_hi) * &inv_2_32;
    evals[idx] = is_bit_cond(&sub_beq_cond, &sub_carry_0);
    idx += 1;
    evals[idx] = is_bit_cond(&sub_beq_cond, &sub_carry_1);
    idx += 1;

    // 38-39: JALR carry
    let pc_0 = *step.get_main_evaluation_element(0, cols::PC_0);
    let pc_1 = *step.get_main_evaluation_element(0, cols::PC_1);
    let instr_size = &four - &two * &c_type;
    let jalr_carry_0 = (&pc_0 + &instr_size - &res_lo) * &inv_2_32;
    let jalr_carry_1 = (&pc_1 + &jalr_carry_0 - &res_hi) * &inv_2_32;
    evals[idx] = is_bit_cond(&jalr, &jalr_carry_0);
    idx += 1;
    evals[idx] = is_bit_cond(&jalr, &jalr_carry_1);
    idx += 1;

    // 40: BranchCond
    let res_0 = *step.get_main_evaluation_element(0, cols::RES_0);
    let res_xor_mp = &res_0 + &mp_selector - &two * &res_0 * &mp_selector;
    let eq_xor_mp = &is_equal + &mp_selector - &two * &is_equal * &mp_selector;
    let expected_bc = &jalr + &blt * &res_xor_mp + &beq * &eq_xor_mp;
    evals[idx] = &branch_cond - &expected_bc;
    idx += 1;

    // 41: EBREAK
    evals[idx] = ebreak;
    idx += 1;

    // 42-44: rv1 zero-forcing
    let rv1_0 = *step.get_main_evaluation_element(0, cols::RV1_0);
    let rv1_1 = *step.get_main_evaluation_element(0, cols::RV1_1);
    let rv1_2 = *step.get_main_evaluation_element(0, cols::RV1_2);
    let not_rr1 = &one - &read_register1;
    evals[idx] = &not_rr1 * &rv1_0;
    idx += 1;
    evals[idx] = &not_rr1 * &rv1_1;
    idx += 1;
    evals[idx] = &not_rr1 * &rv1_2;
    idx += 1;

    // 45-47: rv2 zero-forcing
    let rv2_0 = *step.get_main_evaluation_element(0, cols::RV2_0);
    let rv2_1 = *step.get_main_evaluation_element(0, cols::RV2_1);
    let rv2_2 = *step.get_main_evaluation_element(0, cols::RV2_2);
    let not_rr2 = &one - &read_register2;
    evals[idx] = &not_rr2 * &rv2_0;
    idx += 1;
    evals[idx] = &not_rr2 * &rv2_1;
    idx += 1;
    evals[idx] = &not_rr2 * &rv2_2;
    idx += 1;

    // 48: Arg1 lower
    let rv1_lower = &rv1_0 + &rv1_1 * &shift_16;
    evals[idx] = &arg1_lo - &rv1_lower;
    idx += 1;

    // 49: Arg1 upper
    let expected_arg1_hi = &rv1_2 * &(&one - &word_instr) + &mask_32 * &rv1_ext_bit * &signed;
    evals[idx] = &arg1_hi - &expected_arg1_hi;
    idx += 1;

    // 50: Arg2 lower
    let rv2_lower = &rv2_0 + &rv2_1 * &shift_16;
    let expected_arg2_lo = (&one - &load) * &rv2_lower + (&one - &beq - &blt - &store) * &imm_0;
    evals[idx] = &arg2_lo - &expected_arg2_lo;
    idx += 1;

    // 51: Arg2 upper
    let rv2_term_hi = (&one - &word_instr) * &rv2_2 + &signed * &rv2_ext_bit * &mask_32;
    let expected_arg2_hi = (&one - &load) * &rv2_term_hi + (&one - &beq - &blt - &store) * &imm_1;
    evals[idx] = &arg2_hi - &expected_arg2_hi;
    idx += 1;

    // 52: Rvd lower
    let rvd_0 = *step.get_main_evaluation_element(0, cols::RVD_0);
    let rvd_1 = *step.get_main_evaluation_element(0, cols::RVD_1);
    let not_load = &one - &load;
    evals[idx] = &not_load * &(&rvd_0 - &res_lo);
    idx += 1;

    // 53: Rvd upper
    let expected_rvd_hi = (&one - &word_instr) * &res_hi + &res_ext_bit * &mask_32;
    evals[idx] = &not_load * &(&rvd_1 - &expected_rvd_hi);
    idx += 1;

    // 54-60: SLT res zero
    let slt_blt = &slt + &blt;
    for (j, byte_idx) in (1..8).enumerate() {
        let res_i = *step.get_main_evaluation_element(0, cols::RES[byte_idx]);
        evals[idx + j] = &slt_blt * &res_i;
    }
    idx += 7;

    // 61-63: ExtBitZero
    let not_word = &one - &word_instr;
    evals[idx] = &not_word * &rv1_ext_bit;
    idx += 1;
    evals[idx] = &not_word * &rv2_ext_bit;
    idx += 1;
    evals[idx] = &not_word * &res_ext_bit;
    idx += 1;

    // 64-65: NextPc
    let next_pc_0 = *step.get_main_evaluation_element(0, cols::NEXT_PC_0);
    let next_pc_1 = *step.get_main_evaluation_element(0, cols::NEXT_PC_1);
    let instr_size2 = &four - &two * &c_type;
    let npc_carry_0 = (&pc_0 + &instr_size2 - &next_pc_0) * &inv_2_32;
    let npc_carry_1 = (&pc_1 + &npc_carry_0 - &next_pc_1) * &inv_2_32;
    let not_bc = &one - &branch_cond;
    evals[idx] = &not_bc * &npc_carry_0 * &(&one - &npc_carry_0);
    idx += 1;
    evals[idx] = &not_bc * &npc_carry_1 * &(&one - &npc_carry_1);
    let _ = idx;
}

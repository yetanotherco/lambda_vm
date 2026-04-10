//! Compiled constraint evaluator for the CPU table.
//!
//! Eliminates 77 virtual dispatch calls per LDE row by inlining all constraint
//! computations and using F×E multiplication (3 base muls) instead of the
//! `to_extension()` + E×E multiplication path (6 base muls) for the 66 native
//! constraints that produce base-field results.
//!
//! The 11 LogUp constraints already work in extension field and are computed
//! using the raw-slice helper functions from `stark::lookup`.

use math::field::element::FieldElement;
use stark::lookup::{
    compute_fingerprint_raw, compute_multiplicity_raw, BusInteraction, CompiledEvaluator,
    PackingShifts,
};

use crate::constraints::cpu::BIT_FLAG_COLUMNS;
use crate::constraints::templates::INV_SHIFT_32;
use crate::tables::cpu::cols;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;
type FE = FieldElement<F>;
type FEE = FieldElement<E>;

/// Number of native (non-LogUp) CPU constraints.
const NUM_NATIVE: usize = 66;

/// Compiled CPU constraint evaluator.
///
/// Stores precomputed data needed for LogUp constraint evaluation:
/// the bus interactions from `AirWithBuses::new` construction.
pub struct CpuCompiledEvaluator {
    /// All bus interactions (22 for CPU table), stored for LogUp constraint evaluation.
    interactions: Vec<BusInteraction>,
    /// Number of committed batched pairs (= 10 for 22 interactions).
    num_committed_pairs: usize,
    /// Number of absorbed interactions (= 2 for 22 interactions).
    absorbed_count: usize,
}

impl CpuCompiledEvaluator {
    pub fn new(interactions: Vec<BusInteraction>) -> Self {
        let num_interactions = interactions.len();
        let (num_committed_pairs, absorbed_count) = if num_interactions <= 2 {
            (0, num_interactions)
        } else if num_interactions % 2 == 1 {
            ((num_interactions - 1) / 2, 1)
        } else {
            ((num_interactions - 2) / 2, 2)
        };
        Self {
            interactions,
            num_committed_pairs,
            absorbed_count,
        }
    }
}

// =============================================================================
// Helper: pack 4 byte columns into a 32-bit word
// =============================================================================
#[inline(always)]
fn pack_bytes_raw(main: &[FE], c0: usize, c1: usize, c2: usize, c3: usize) -> FE {
    let shift_8 = FE::from(1u64 << 8);
    let shift_16 = FE::from(1u64 << 16);
    let shift_24 = FE::from(1u64 << 24);
    &main[c0] + &main[c1] * &shift_8 + &main[c2] * &shift_16 + &main[c3] * shift_24
}

// =============================================================================
// Inline ADD constraint: cond * carry * (1 - carry)
// =============================================================================

/// Compute carry_0 for an ADD operand triple (all as raw-slice column reads).
/// carry_0 = (lhs_lo + rhs_lo - sum_lo) * 2^(-32)
#[inline(always)]
fn add_carry0_dword_bl(main: &[FE], lhs_start: usize, rhs_start: usize, sum_start: usize) -> FE {
    let lhs_lo = pack_bytes_raw(
        main,
        lhs_start,
        lhs_start + 1,
        lhs_start + 2,
        lhs_start + 3,
    );
    let rhs_lo = pack_bytes_raw(
        main,
        rhs_start,
        rhs_start + 1,
        rhs_start + 2,
        rhs_start + 3,
    );
    let sum_lo = pack_bytes_raw(
        main,
        sum_start,
        sum_start + 1,
        sum_start + 2,
        sum_start + 3,
    );
    (lhs_lo + rhs_lo - sum_lo) * FE::from(INV_SHIFT_32)
}

/// Compute carry_1 for an ADD operand triple.
/// carry_1 = (lhs_hi + rhs_hi + carry_0 - sum_hi) * 2^(-32)
#[inline(always)]
fn add_carry1_dword_bl(main: &[FE], lhs_start: usize, rhs_start: usize, sum_start: usize) -> FE {
    let lhs_hi = pack_bytes_raw(
        main,
        lhs_start + 4,
        lhs_start + 5,
        lhs_start + 6,
        lhs_start + 7,
    );
    let rhs_hi = pack_bytes_raw(
        main,
        rhs_start + 4,
        rhs_start + 5,
        rhs_start + 6,
        rhs_start + 7,
    );
    let sum_hi = pack_bytes_raw(
        main,
        sum_start + 4,
        sum_start + 5,
        sum_start + 6,
        sum_start + 7,
    );
    let carry_0 = add_carry0_dword_bl(main, lhs_start, rhs_start, sum_start);
    (lhs_hi + rhs_hi + carry_0 - sum_hi) * FE::from(INV_SHIFT_32)
}

/// Compute cond * carry * (1 - carry) for an ADD constraint.
#[inline(always)]
fn add_constraint_dword_bl(
    main: &[FE],
    cond_cols: &[usize],
    lhs_start: usize,
    rhs_start: usize,
    sum_start: usize,
    carry_idx: usize,
) -> FE {
    let one = FE::one();
    let carry = if carry_idx == 0 {
        add_carry0_dword_bl(main, lhs_start, rhs_start, sum_start)
    } else {
        add_carry1_dword_bl(main, lhs_start, rhs_start, sum_start)
    };
    let cond: FE = cond_cols
        .iter()
        .map(|&col| main[col].clone())
        .fold(FE::zero(), |acc, x| acc + x);
    cond * &carry * (one - carry)
}

impl CompiledEvaluator<F, E> for CpuCompiledEvaluator {
    fn evaluate(
        &self,
        main_curr: &[FE],
        main_next: &[FE],
        aux_curr: &[FEE],
        aux_next: &[FEE],
        betas: &[FEE],
        rap_challenges: &[FEE],
        logup_alpha_powers: &[FEE],
        logup_table_offset: &FEE,
    ) -> FEE {
        let mut acc = FEE::zero();
        let one_f = FE::one();

        // =================================================================
        // IS_BIT constraints (indices 0..31): val * (1 - val)
        // All unconditional, degree 2, base-field result
        // =================================================================
        for (idx, &col) in BIT_FLAG_COLUMNS.iter().enumerate() {
            let val = &main_curr[col];
            let constraint = val * &(&one_f - val);
            // F×E multiplication: 3 base muls instead of 6
            acc += &constraint * &betas[idx];
        }

        // =================================================================
        // ADD constraints (indices 32..39)
        // 8 constraints: ADD+LOAD carry0/1, STORE carry0/1, SUB+BEQ carry0/1, JALR carry0/1
        // =================================================================

        // ADD+LOAD carry constraints (idx 32, 33)
        // lhs=arg1, rhs=arg2, sum=res, cond=[ADD, LOAD]
        {
            let c0 = add_constraint_dword_bl(
                main_curr,
                &[cols::ADD, cols::LOAD],
                cols::ARG1_0,
                cols::ARG2_0,
                cols::RES_0,
                0,
            );
            acc += &c0 * &betas[32];

            let c1 = add_constraint_dword_bl(
                main_curr,
                &[cols::ADD, cols::LOAD],
                cols::ARG1_0,
                cols::ARG2_0,
                cols::RES_0,
                1,
            );
            acc += &c1 * &betas[33];
        }

        // STORE carry constraints (idx 34, 35)
        // lhs=arg1, rhs=imm(DWordWL), sum=res, cond=[STORE]
        // STORE uses DWordWL for rhs (imm) rather than DWordBL
        {
            // For STORE: rhs_lo = IMM_0 (single word), rhs_hi = IMM_1
            // carry_0 = (arg1_lo + imm_0 - res_lo) * 2^(-32)
            let arg1_lo = pack_bytes_raw(
                main_curr,
                cols::ARG1_0,
                cols::ARG1_1,
                cols::ARG1_2,
                cols::ARG1_3,
            );
            let arg1_hi = pack_bytes_raw(
                main_curr,
                cols::ARG1_4,
                cols::ARG1_5,
                cols::ARG1_6,
                cols::ARG1_7,
            );
            let res_lo = pack_bytes_raw(
                main_curr,
                cols::RES_0,
                cols::RES_1,
                cols::RES_2,
                cols::RES_3,
            );
            let res_hi = pack_bytes_raw(
                main_curr,
                cols::RES_4,
                cols::RES_5,
                cols::RES_6,
                cols::RES_7,
            );
            let imm_0 = &main_curr[cols::IMM_0];
            let imm_1 = &main_curr[cols::IMM_1];
            let inv_2_32 = FE::from(INV_SHIFT_32);
            let store_carry0 = (&arg1_lo + imm_0 - &res_lo) * &inv_2_32;
            let store_carry1 = (&arg1_hi + imm_1 + &store_carry0 - &res_hi) * &inv_2_32;
            let store = &main_curr[cols::STORE];
            let c0 = store * &store_carry0 * (&one_f - &store_carry0);
            acc += &c0 * &betas[34];
            let c1 = store * &store_carry1 * (&one_f - &store_carry1);
            acc += &c1 * &betas[35];
        }

        // SUB+BEQ carry constraints (idx 36, 37)
        // SUB is verified as: arg2 + res = arg1
        // lhs=arg2, rhs=res, sum=arg1, cond=[SUB, BEQ]
        {
            let c0 = add_constraint_dword_bl(
                main_curr,
                &[cols::SUB, cols::BEQ],
                cols::ARG2_0,
                cols::RES_0,
                cols::ARG1_0,
                0,
            );
            acc += &c0 * &betas[36];

            let c1 = add_constraint_dword_bl(
                main_curr,
                &[cols::SUB, cols::BEQ],
                cols::ARG2_0,
                cols::RES_0,
                cols::ARG1_0,
                1,
            );
            acc += &c1 * &betas[37];
        }

        // JALR carry constraints (idx 38, 39)
        // res = pc + instr_size, where instr_size = 4 - 2*c_type
        // lhs=pc(DWordWL), rhs=instr_size(linear), sum=res(DWordBL)
        {
            let pc_0 = &main_curr[cols::PC_0];
            let pc_1 = &main_curr[cols::PC_1];
            let c_type = &main_curr[cols::C_TYPE_INSTRUCTION];
            let res_lo = pack_bytes_raw(
                main_curr,
                cols::RES_0,
                cols::RES_1,
                cols::RES_2,
                cols::RES_3,
            );
            let res_hi = pack_bytes_raw(
                main_curr,
                cols::RES_4,
                cols::RES_5,
                cols::RES_6,
                cols::RES_7,
            );
            let inv_2_32 = FE::from(INV_SHIFT_32);
            let instr_size = FE::from(4u64) - FE::from(2u64) * c_type;
            let jalr_carry0 = (pc_0 + &instr_size - &res_lo) * &inv_2_32;
            let jalr_carry1 = (pc_1 + &jalr_carry0 - &res_hi) * &inv_2_32;
            let jalr = &main_curr[cols::JALR];
            let c0 = jalr * &jalr_carry0 * (&one_f - &jalr_carry0);
            acc += &c0 * &betas[38];
            let c1 = jalr * &jalr_carry1 * (&one_f - &jalr_carry1);
            acc += &c1 * &betas[39];
        }

        // =================================================================
        // Branch condition constraint (idx 40)
        // branch_cond = JALR + BLT*(res[0] XOR mp) + BEQ*(is_equal XOR mp)
        // =================================================================
        {
            let jalr = &main_curr[cols::JALR];
            let blt = &main_curr[cols::BLT];
            let beq = &main_curr[cols::BEQ];
            let mp = &main_curr[cols::MP_SELECTOR];
            let res_0 = &main_curr[cols::RES_0];
            let is_eq = &main_curr[cols::IS_EQUAL];
            let branch_cond = &main_curr[cols::BRANCH_COND];
            let two = FE::from(2u64);

            // XOR(a,b) = a + b - 2*a*b
            let res_xor_mp = res_0 + mp - &two * res_0 * mp;
            let eq_xor_mp = is_eq + mp - &two * is_eq * mp;
            let expected = jalr + blt * res_xor_mp + beq * eq_xor_mp;
            let c = branch_cond - expected;
            acc += &c * &betas[40];
        }

        // =================================================================
        // EBREAK constraint (idx 41): EBREAK = 0
        // =================================================================
        {
            let c = main_curr[cols::EBREAK].clone();
            acc += &c * &betas[41];
        }

        // =================================================================
        // rv1 zero-forcing CM48 (idx 42, 43, 44)
        // (1 - read_register1) * rv1[i] = 0
        // =================================================================
        {
            let flag = &main_curr[cols::READ_REGISTER1];
            let not_flag = &one_f - flag;
            for (i, &value_col) in [cols::RV1_0, cols::RV1_1, cols::RV1_2].iter().enumerate() {
                let c = &not_flag * &main_curr[value_col];
                acc += &c * &betas[42 + i];
            }
        }

        // =================================================================
        // rv2 zero-forcing CM50 (idx 45, 46, 47)
        // (1 - read_register2) * rv2[i] = 0
        // =================================================================
        {
            let flag = &main_curr[cols::READ_REGISTER2];
            let not_flag = &one_f - flag;
            for (i, &value_col) in [cols::RV2_0, cols::RV2_1, cols::RV2_2].iter().enumerate() {
                let c = &not_flag * &main_curr[value_col];
                acc += &c * &betas[45 + i];
            }
        }

        // =================================================================
        // Arg1 lower constraint (idx 48)
        // arg1[:4] as word = rv1[0] + rv1[1] * 2^16
        // =================================================================
        {
            let arg1_lo = pack_bytes_raw(
                main_curr,
                cols::ARG1_0,
                cols::ARG1_1,
                cols::ARG1_2,
                cols::ARG1_3,
            );
            let shift_16 = FE::from(1u64 << 16);
            let rv1_lower = &main_curr[cols::RV1_0] + &main_curr[cols::RV1_1] * shift_16;
            let c = arg1_lo - rv1_lower;
            acc += &c * &betas[48];
        }

        // =================================================================
        // Arg1 upper constraint (idx 49)
        // arg1[4:] = rv1[2]*(1-word_instr) + mask_32*rv1_ext_bit*signed
        // =================================================================
        {
            let arg1_hi = pack_bytes_raw(
                main_curr,
                cols::ARG1_4,
                cols::ARG1_5,
                cols::ARG1_6,
                cols::ARG1_7,
            );
            let rv1_upper = &main_curr[cols::RV1_2];
            let word_instr = &main_curr[cols::WORD_INSTR];
            let signed = &main_curr[cols::SIGNED];
            let rv1_ext_bit = &main_curr[cols::RV1_EXT_BIT];
            let mask_32 = FE::from((1u64 << 32) - 1);
            let expected =
                rv1_upper * (&one_f - word_instr) + mask_32 * rv1_ext_bit * signed;
            let c = arg1_hi - expected;
            acc += &c * &betas[49];
        }

        // =================================================================
        // Arg2 lower constraint (idx 50)
        // arg2[:4] = (1-LOAD)*rv2[:2] + (1-BEQ-BLT-STORE)*imm[0]
        // =================================================================
        {
            let arg2_lo = pack_bytes_raw(
                main_curr,
                cols::ARG2[0],
                cols::ARG2[1],
                cols::ARG2[2],
                cols::ARG2[3],
            );
            let shift_16 = FE::from(1u64 << 16);
            let rv2_lower =
                &main_curr[cols::RV2_0] + &main_curr[cols::RV2_1] * shift_16;
            let imm_0 = &main_curr[cols::IMM_0];
            let store = &main_curr[cols::STORE];
            let load = &main_curr[cols::LOAD];
            let beq = &main_curr[cols::BEQ];
            let blt = &main_curr[cols::BLT];
            let expected =
                (&one_f - load) * rv2_lower + (&one_f - beq - blt - store) * imm_0;
            let c = arg2_lo - expected;
            acc += &c * &betas[50];
        }

        // =================================================================
        // Arg2 upper constraint (idx 51)
        // arg2[4:] = (1-LOAD)*rv2_term + (1-BEQ-BLT-STORE)*imm[1]
        // where rv2_term = (1-word_instr)*rv2[2] + signed*rv2_ext_bit*(2^32-1)
        // =================================================================
        {
            let arg2_hi = pack_bytes_raw(
                main_curr,
                cols::ARG2[4],
                cols::ARG2[5],
                cols::ARG2[6],
                cols::ARG2[7],
            );
            let rv2_upper = &main_curr[cols::RV2_2];
            let imm_1 = &main_curr[cols::IMM_1];
            let store = &main_curr[cols::STORE];
            let load = &main_curr[cols::LOAD];
            let beq = &main_curr[cols::BEQ];
            let blt = &main_curr[cols::BLT];
            let word_instr = &main_curr[cols::WORD_INSTR];
            let signed = &main_curr[cols::SIGNED];
            let rv2_ext_bit = &main_curr[cols::RV2_EXT_BIT];
            let mask_32 = FE::from((1u64 << 32) - 1);
            let rv2_term =
                (&one_f - word_instr) * rv2_upper + signed * rv2_ext_bit * &mask_32;
            let expected =
                (&one_f - load) * rv2_term + (&one_f - beq - blt - store) * imm_1;
            let c = arg2_hi - expected;
            acc += &c * &betas[51];
        }

        // =================================================================
        // Rvd lower constraint (idx 52)
        // (1-LOAD) * (rvd[0] - res[:4]) = 0
        // =================================================================
        {
            let rvd_0 = &main_curr[cols::RVD_0];
            let res_lo = pack_bytes_raw(
                main_curr,
                cols::RES[0],
                cols::RES[1],
                cols::RES[2],
                cols::RES[3],
            );
            let load = &main_curr[cols::LOAD];
            let c = (&one_f - load) * (rvd_0 - res_lo);
            acc += &c * &betas[52];
        }

        // =================================================================
        // Rvd upper constraint (idx 53)
        // (1-LOAD) * (rvd[1] - ((1-word_instr)*res_hi + res_ext_bit*(2^32-1))) = 0
        // =================================================================
        {
            let rvd_1 = &main_curr[cols::RVD_1];
            let res_hi = pack_bytes_raw(
                main_curr,
                cols::RES[4],
                cols::RES[5],
                cols::RES[6],
                cols::RES[7],
            );
            let load = &main_curr[cols::LOAD];
            let word_instr = &main_curr[cols::WORD_INSTR];
            let res_ext_bit = &main_curr[cols::RES_EXT_BIT];
            let mask_32 = FE::from((1u64 << 32) - 1);
            let expected = (&one_f - word_instr) * res_hi + res_ext_bit * mask_32;
            let c = (&one_f - load) * (rvd_1 - expected);
            acc += &c * &betas[53];
        }

        // =================================================================
        // SLT res zero constraints (idx 54..60)
        // (SLT + BLT) * res[i] = 0 for i in 1..8
        // =================================================================
        {
            let slt = &main_curr[cols::SLT];
            let blt = &main_curr[cols::BLT];
            let slt_blt = slt + blt;
            for i in 0..7 {
                let c = &slt_blt * &main_curr[cols::RES[i + 1]];
                acc += &c * &betas[54 + i];
            }
        }

        // =================================================================
        // Extension bit zero constraints (idx 61, 62, 63)
        // (1 - word_instr) * ext_bit = 0
        // =================================================================
        {
            let word_instr = &main_curr[cols::WORD_INSTR];
            let not_word = &one_f - word_instr;
            let ext_cols = [cols::RV1_EXT_BIT, cols::RV2_EXT_BIT, cols::RES_EXT_BIT];
            for (i, &col) in ext_cols.iter().enumerate() {
                let c = &not_word * &main_curr[col];
                acc += &c * &betas[61 + i];
            }
        }

        // =================================================================
        // Next PC (non-branching) constraints (idx 64, 65)
        // (1 - branch_cond) * carry * (1 - carry) = 0
        // =================================================================
        {
            let branch_cond = &main_curr[cols::BRANCH_COND];
            let not_branch = &one_f - branch_cond;
            let pc_lo = &main_curr[cols::PC_0];
            let pc_hi = &main_curr[cols::PC_1];
            let next_pc_lo = &main_curr[cols::NEXT_PC_0];
            let next_pc_hi = &main_curr[cols::NEXT_PC_1];
            let c_type = &main_curr[cols::C_TYPE_INSTRUCTION];
            let inv_2_32 = FE::from(INV_SHIFT_32);
            let instr_size = FE::from(4u64) - FE::from(2u64) * c_type;
            let carry_0 = (pc_lo + &instr_size - next_pc_lo) * &inv_2_32;
            let carry_1 = (pc_hi + &carry_0 - next_pc_hi) * &inv_2_32;

            let c0 = &not_branch * &carry_0 * (&one_f - &carry_0);
            acc += &c0 * &betas[64];

            let c1 = &not_branch * &carry_1 * (&one_f - &carry_1);
            acc += &c1 * &betas[65];
        }

        // =================================================================
        // LogUp constraints (indices NUM_NATIVE..NUM_NATIVE+num_logup)
        // These are already extension-field operations.
        // =================================================================
        let shifts = PackingShifts::<F>::new();

        if !rap_challenges.is_empty() {
            let z = &rap_challenges[0];

            // Batched term constraints (indices NUM_NATIVE..NUM_NATIVE+num_committed_pairs)
            for pair_idx in 0..self.num_committed_pairs {
                let ia = &self.interactions[pair_idx * 2];
                let ib = &self.interactions[pair_idx * 2 + 1];

                let c = aux_curr[pair_idx].clone();
                let m_a = compute_multiplicity_raw(main_curr, &ia.multiplicity);
                let m_b = compute_multiplicity_raw(main_curr, &ib.multiplicity);
                let fp_a = compute_fingerprint_raw(main_curr, ia, z, logup_alpha_powers, &shifts);
                let fp_b = compute_fingerprint_raw(main_curr, ib, z, logup_alpha_powers, &shifts);

                // c * fp_a * fp_b - sign_a * m_a * fp_b - sign_b * m_b * fp_a = 0
                let term_a = if ia.is_sender {
                    m_a * &fp_b
                } else {
                    -(m_a * &fp_b)
                };
                let term_b = if ib.is_sender {
                    m_b * &fp_a
                } else {
                    -(m_b * &fp_a)
                };
                let constraint = c * &fp_a * &fp_b - term_a - term_b;
                acc += &constraint * &betas[NUM_NATIVE + pair_idx];
            }

            // Accumulated constraint (last index)
            let num_term_columns = self.num_committed_pairs;
            let acc_column_idx = num_term_columns;
            let acc_curr_val = &aux_curr[acc_column_idx];
            let acc_next_val = &aux_next[acc_column_idx];

            // Sum of committed term columns at the next step
            let terms_sum: FEE = (0..num_term_columns)
                .map(|i| aux_next[i].clone())
                .sum();

            // delta = acc_next - acc_curr - terms_sum + L/N
            let delta = acc_next_val - acc_curr_val - terms_sum + logup_table_offset;

            let num_interactions = self.interactions.len();
            let absorbed_start = num_interactions - self.absorbed_count;
            let absorbed = &self.interactions[absorbed_start..];

            let acc_constraint_idx = NUM_NATIVE + self.num_committed_pairs;

            match self.absorbed_count {
                1 => {
                    // (delta) * f - sign * m = 0
                    let m = compute_multiplicity_raw(main_next, &absorbed[0].multiplicity);
                    let f = compute_fingerprint_raw(
                        main_next,
                        &absorbed[0],
                        z,
                        logup_alpha_powers,
                        &shifts,
                    );
                    let sign: FEE = if absorbed[0].is_sender {
                        FEE::one()
                    } else {
                        -FEE::one()
                    };
                    let constraint = delta * &f - m * sign;
                    acc += &constraint * &betas[acc_constraint_idx];
                }
                2 => {
                    // (delta) * f1 * f2 - sign1*m1*f2 - sign2*m2*f1 = 0
                    let m1 = compute_multiplicity_raw(main_next, &absorbed[0].multiplicity);
                    let m2 = compute_multiplicity_raw(main_next, &absorbed[1].multiplicity);
                    let f1 = compute_fingerprint_raw(
                        main_next,
                        &absorbed[0],
                        z,
                        logup_alpha_powers,
                        &shifts,
                    );
                    let f2 = compute_fingerprint_raw(
                        main_next,
                        &absorbed[1],
                        z,
                        logup_alpha_powers,
                        &shifts,
                    );
                    let term1 = if absorbed[0].is_sender {
                        m1 * &f2
                    } else {
                        -(m1 * &f2)
                    };
                    let term2 = if absorbed[1].is_sender {
                        m2 * &f1
                    } else {
                        -(m2 * &f1)
                    };
                    let constraint = delta * &f1 * &f2 - term1 - term2;
                    acc += &constraint * &betas[acc_constraint_idx];
                }
                _ => {}
            }
        }

        acc
    }
}

//! Packed SIMD constraint evaluation functions for table constraints.
//!
//! Each function evaluates all transition constraints for a specific table
//! at WIDTH consecutive LDE points simultaneously using PackedGoldilocks
//! arithmetic. Column data is loaded directly from column-major LDE storage
//! via `load_main_packed`, eliminating the per-point Frame fill overhead.
//!
//! These functions have type `PackedTableConstraintsFn` and are registered
//! via `AirWithBuses::with_packed_constraints`.

use math::field::element::FieldElement;
use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField as GF;
use math::field::fields::fft_friendly::u64_goldilocks_packed::PackedGoldilocks;
use math::field::packed::PackedField;
use stark::lookup::load_main_packed;

use crate::constraints::templates::INV_SHIFT_32;

// Helper: broadcast a u64 constant to all SIMD lanes
#[inline(always)]
fn bcast(val: u64) -> PackedGoldilocks {
    PackedGoldilocks::broadcast(FieldElement::<GF>::from(val))
}

// Helper: load a column at base_row
#[inline(always)]
fn load(
    cols: &[Vec<FieldElement<GF>>],
    col: usize,
    row: usize,
    n: usize,
) -> PackedGoldilocks {
    load_main_packed(&cols[col], row, n)
}

// =============================================================================
// BRANCH table: 2 constraints (Carry0IsBit, Carry1IsBit)
// =============================================================================

pub fn evaluate_branch_packed(
    lde_main_cols: &[Vec<FieldElement<GF>>],
    base_row: usize,
    lde_domain_size: usize,
    results: &mut [PackedGoldilocks],
) {
    use crate::tables::branch::cols;

    let n = lde_domain_size;
    let one = PackedGoldilocks::ones();
    let inv_2_32 = bcast(INV_SHIFT_32);
    let shift_8 = bcast(1 << 8);
    let shift_16 = bcast(1 << 16);

    // Load all needed columns
    let pc_0 = load(lde_main_cols, cols::PC_0, base_row, n);
    let pc_1 = load(lde_main_cols, cols::PC_1, base_row, n);
    let offset_0 = load(lde_main_cols, cols::OFFSET_0, base_row, n);
    let offset_1 = load(lde_main_cols, cols::OFFSET_1, base_row, n);
    let register_0 = load(lde_main_cols, cols::REGISTER_0, base_row, n);
    let register_1 = load(lde_main_cols, cols::REGISTER_1, base_row, n);
    let jalr = load(lde_main_cols, cols::JALR, base_row, n);
    let next_pc_high_0 = load(lde_main_cols, cols::NEXT_PC_HIGH_0, base_row, n);
    let next_pc_high_1 = load(lde_main_cols, cols::NEXT_PC_HIGH_1, base_row, n);
    let next_pc_high_2 = load(lde_main_cols, cols::NEXT_PC_HIGH_2, base_row, n);
    let next_pc_low_1 = load(lde_main_cols, cols::NEXT_PC_LOW_1, base_row, n);
    let unmasked_low_byte = load(lde_main_cols, cols::UNMASKED_LOW_BYTE, base_row, n);

    // Compute base = (1-JALR)*pc + JALR*register
    let one_minus_jalr = one - jalr;
    let base_0 = one_minus_jalr * pc_0 + jalr * register_0;
    let base_1 = one_minus_jalr * pc_1 + jalr * register_1;

    // Compute next_pc_unmasked as DWordWL
    let unmasked_0 = unmasked_low_byte + next_pc_low_1 * shift_8 + next_pc_high_0 * shift_16;
    let unmasked_1 = next_pc_high_1 + next_pc_high_2 * shift_16;

    // carry_0 = (base_0 + offset_0 - unmasked_0) * inv_2_32
    let carry_0 = (base_0 + offset_0 - unmasked_0) * inv_2_32;

    // carry_1 = (base_1 + offset_1 + carry_0 - unmasked_1) * inv_2_32
    let carry_1 = (base_1 + offset_1 + carry_0 - unmasked_1) * inv_2_32;

    // Constraint 0: carry_0 * (1 - carry_0)
    results[0] = carry_0 * (one - carry_0);
    // Constraint 1: carry_1 * (1 - carry_1)
    results[1] = carry_1 * (one - carry_1);
}

// =============================================================================
// LOAD table: 8 constraints
// =============================================================================

pub fn evaluate_load_packed(
    lde_main_cols: &[Vec<FieldElement<GF>>],
    base_row: usize,
    lde_domain_size: usize,
    results: &mut [PackedGoldilocks],
) {
    use crate::tables::load::cols;

    let n = lde_domain_size;
    let one = PackedGoldilocks::ones();
    let ff = bcast(255);

    let mu = load(lde_main_cols, cols::MU, base_row, n);
    let read2 = load(lde_main_cols, cols::READ2, base_row, n);
    let read4 = load(lde_main_cols, cols::READ4, base_row, n);
    let read8 = load(lde_main_cols, cols::READ8, base_row, n);
    let signed = load(lde_main_cols, cols::SIGNED, base_row, n);
    let sign_bit = load(lde_main_cols, cols::SIGN_BIT, base_row, n);

    // Pre-compute shared subexpressions
    let read_sum = read2 + read4 + read8;
    let expected = signed * sign_bit * ff;

    // Constraint 0: ReadImpliesMu: (read2 + read4 + read8) * (1 - μ) = 0
    results[0] = read_sum * (one - mu);

    // Constraints 1-4: ExtensionHigh(4..8): (1 - read8) * (res[i] - expected)
    let one_minus_read8 = one - read8;
    for (k, &i) in [4usize, 5, 6, 7].iter().enumerate() {
        let res_i = load(lde_main_cols, cols::RES[i], base_row, n);
        results[1 + k] = one_minus_read8 * (res_i - expected);
    }

    // Constraints 5-6: ExtensionMid(2..4): (1 - read4 - read8) * (res[i] - expected)
    let one_minus_read4_read8 = one - read4 - read8;
    for (k, &i) in [2usize, 3].iter().enumerate() {
        let res_i = load(lde_main_cols, cols::RES[i], base_row, n);
        results[5 + k] = one_minus_read4_read8 * (res_i - expected);
    }

    // Constraint 7: ExtensionLow: (1 - read2 - read4 - read8) * (res[1] - expected)
    let res_1 = load(lde_main_cols, cols::RES[1], base_row, n);
    results[7] = (one - read_sum) * (res_1 - expected);
}

// =============================================================================
// DVRM table: 18 constraints
// =============================================================================

pub fn evaluate_dvrm_packed(
    lde_main_cols: &[Vec<FieldElement<GF>>],
    base_row: usize,
    lde_domain_size: usize,
    results: &mut [PackedGoldilocks],
) {
    use crate::tables::dvrm::cols;

    let n = lde_domain_size;
    let one = PackedGoldilocks::ones();
    let shift_16 = bcast(1 << 16);
    let inv_2_32 = bcast(INV_SHIFT_32);
    let sign_fill = bcast(0xFFFF);

    // Load all columns we need
    let signed_col = load(lde_main_cols, cols::SIGNED, base_row, n);
    let n_cols: [PackedGoldilocks; 4] = [
        load(lde_main_cols, cols::N_0, base_row, n),
        load(lde_main_cols, cols::N_1, base_row, n),
        load(lde_main_cols, cols::N_2, base_row, n),
        load(lde_main_cols, cols::N_3, base_row, n),
    ];
    let d_cols: [PackedGoldilocks; 4] = [
        load(lde_main_cols, cols::D_0, base_row, n),
        load(lde_main_cols, cols::D_1, base_row, n),
        load(lde_main_cols, cols::D_2, base_row, n),
        load(lde_main_cols, cols::D_3, base_row, n),
    ];
    let q_cols: [PackedGoldilocks; 4] = [
        load(lde_main_cols, cols::Q_0, base_row, n),
        load(lde_main_cols, cols::Q_1, base_row, n),
        load(lde_main_cols, cols::Q_2, base_row, n),
        load(lde_main_cols, cols::Q_3, base_row, n),
    ];
    let r_cols: [PackedGoldilocks; 4] = [
        load(lde_main_cols, cols::R_0, base_row, n),
        load(lde_main_cols, cols::R_1, base_row, n),
        load(lde_main_cols, cols::R_2, base_row, n),
        load(lde_main_cols, cols::R_3, base_row, n),
    ];
    let nsr_cols: [PackedGoldilocks; 4] = [
        load(lde_main_cols, cols::N_SUB_R_0, base_row, n),
        load(lde_main_cols, cols::N_SUB_R_1, base_row, n),
        load(lde_main_cols, cols::N_SUB_R_2, base_row, n),
        load(lde_main_cols, cols::N_SUB_R_3, base_row, n),
    ];
    let sign_n = load(lde_main_cols, cols::SIGN_N, base_row, n);
    let sign_d = load(lde_main_cols, cols::SIGN_D, base_row, n);
    let sign_q = load(lde_main_cols, cols::SIGN_Q, base_row, n);
    let sign_r = load(lde_main_cols, cols::SIGN_R, base_row, n);
    let sign_nsr = load(lde_main_cols, cols::SIGN_N_SUB_R, base_row, n);
    let overflow = load(lde_main_cols, cols::OVERFLOW, base_row, n);
    let div_by_zero = load(lde_main_cols, cols::DIV_BY_ZERO, base_row, n);
    let abs_r_0 = load(lde_main_cols, cols::ABS_R_0, base_row, n);
    let abs_r_1 = load(lde_main_cols, cols::ABS_R_1, base_row, n);
    let abs_d_0 = load(lde_main_cols, cols::ABS_D_0, base_row, n);
    let abs_d_1 = load(lde_main_cols, cols::ABS_D_1, base_row, n);

    // Helper: build extended QuadWL representation
    let build_ext_quad = |halves: &[PackedGoldilocks; 4], sign: PackedGoldilocks| -> [PackedGoldilocks; 4] {
        let ext_word = sign * sign_fill + sign * sign_fill * shift_16;
        [
            halves[0] + halves[1] * shift_16,
            halves[2] + halves[3] * shift_16,
            ext_word,
            ext_word,
        ]
    };

    let mut idx = 0;

    // Constraint 0: SignedIsBit: signed * (1 - signed) = 0
    results[idx] = signed_col * (one - signed_col);
    idx += 1;

    // Constraint 1: RemainderSignMatchesNumerator: (r0+r1+r2+r3) * (sign_r - sign_n) = 0
    let r_sum = r_cols[0] + r_cols[1] + r_cols[2] + r_cols[3];
    results[idx] = r_sum * (sign_r - sign_n);
    idx += 1;

    // Constraints 2-3: AbsRFormula(0), AbsRFormula(1)
    // (1 - sign_r) * (abs_r[i] - r::DWordWL[i]) = 0
    let one_minus_sign_r = one - sign_r;
    let r_wl_0 = r_cols[0] + r_cols[1] * shift_16;
    let r_wl_1 = r_cols[2] + r_cols[3] * shift_16;
    results[idx] = one_minus_sign_r * (abs_r_0 - r_wl_0);
    idx += 1;
    results[idx] = one_minus_sign_r * (abs_r_1 - r_wl_1);
    idx += 1;

    // Constraints 4-5: AbsDFormula(0), AbsDFormula(1)
    let one_minus_sign_d = one - sign_d;
    let d_wl_0 = d_cols[0] + d_cols[1] * shift_16;
    let d_wl_1 = d_cols[2] + d_cols[3] * shift_16;
    results[idx] = one_minus_sign_d * (abs_d_0 - d_wl_0);
    idx += 1;
    results[idx] = one_minus_sign_d * (abs_d_1 - d_wl_1);
    idx += 1;

    // Constraint 6: SignQFormula: signed * (1 - overflow) - sign_q = 0
    results[idx] = signed_col * (one - overflow) - sign_q;
    idx += 1;

    // Constraints 7-10: CarryIsBit(0..4) - virtual carries from n = n_sub_r + r
    let ext_n = build_ext_quad(&n_cols, sign_n);
    let ext_r = build_ext_quad(&r_cols, sign_r);
    let ext_nsr = build_ext_quad(&nsr_cols, sign_nsr);

    let carry_0 = (ext_nsr[0] + ext_r[0] - ext_n[0]) * inv_2_32;
    results[idx] = carry_0 * (one - carry_0);
    idx += 1;

    let carry_1 = (ext_nsr[1] + ext_r[1] + carry_0 - ext_n[1]) * inv_2_32;
    results[idx] = carry_1 * (one - carry_1);
    idx += 1;

    let carry_2 = (ext_nsr[2] + ext_r[2] + carry_1 - ext_n[2]) * inv_2_32;
    results[idx] = carry_2 * (one - carry_2);
    idx += 1;

    let carry_3 = (ext_nsr[3] + ext_r[3] + carry_2 - ext_n[3]) * inv_2_32;
    results[idx] = carry_3 * (one - carry_3);
    idx += 1;

    // Constraint 11: SignNSubRIsBit: sign_n_sub_r * (1 - sign_n_sub_r) = 0
    results[idx] = sign_nsr * (one - sign_nsr);
    idx += 1;

    // Constraint 12: UnsignedSignN: (1 - signed) * sign_n = 0
    let one_minus_signed = one - signed_col;
    results[idx] = one_minus_signed * sign_n;
    idx += 1;

    // Constraint 13: UnsignedSignR: (1 - signed) * sign_r = 0
    results[idx] = one_minus_signed * sign_r;
    idx += 1;

    // Constraint 14: UnsignedSignD: (1 - signed) * sign_d = 0
    results[idx] = one_minus_signed * sign_d;
    idx += 1;

    // Constraints 15-18: DivByZeroQ(0..4): div_by_zero * (q[i] - 65535) = 0
    for i in 0..4 {
        results[idx] = div_by_zero * (q_cols[i] - sign_fill);
        idx += 1;
    }
    debug_assert_eq!(idx, 19);
}

// =============================================================================
// MEMW table: 16 constraints (2 MuSum/W2 + 14 ADD carry constraints)
// =============================================================================

pub fn evaluate_memw_packed(
    lde_main_cols: &[Vec<FieldElement<GF>>],
    base_row: usize,
    lde_domain_size: usize,
    results: &mut [PackedGoldilocks],
) {
    use crate::tables::memw::cols;

    let n = lde_domain_size;
    let one = PackedGoldilocks::ones();
    let inv_2_32 = bcast(INV_SHIFT_32);
    let shift_16 = bcast(1 << 16);

    let mu_read = load(lde_main_cols, cols::MU_READ, base_row, n);
    let mu_write = load(lde_main_cols, cols::MU_WRITE, base_row, n);
    let mu_sum = mu_read + mu_write;

    let write2 = load(lde_main_cols, cols::WRITE2, base_row, n);
    let write4 = load(lde_main_cols, cols::WRITE4, base_row, n);
    let write8 = load(lde_main_cols, cols::WRITE8, base_row, n);
    let w2 = write2 + write4 + write8;

    let base_addr_0 = load(lde_main_cols, cols::BASE_ADDRESS_0, base_row, n);
    let base_addr_1 = load(lde_main_cols, cols::BASE_ADDRESS_1, base_row, n);

    let mut idx = 0;

    // Constraint 0: MuSumIsBit: μ_sum * (1 - μ_sum)
    results[idx] = mu_sum * (one - mu_sum);
    idx += 1;

    // Constraint 1: W2ImpliesMuSum: w2 * (1 - μ_sum)
    results[idx] = w2 * (one - mu_sum);
    idx += 1;

    // Constraints 2-15: ADD constraints for address_add[0..6]
    // Each address_add[i] produces 2 carry constraints (carry_0, carry_1)
    // Verifies: base_address + (i+1) = address_add[i]
    for i in 0..7usize {
        let cols_i = cols::address_add(i);
        // address_add[i] as DWordHL → DWordWL
        let aa_h0 = load(lde_main_cols, cols_i[0], base_row, n);
        let aa_h1 = load(lde_main_cols, cols_i[1], base_row, n);
        let aa_h2 = load(lde_main_cols, cols_i[2], base_row, n);
        let aa_h3 = load(lde_main_cols, cols_i[3], base_row, n);
        let sum_lo = aa_h0 + aa_h1 * shift_16;
        let sum_hi = aa_h2 + aa_h3 * shift_16;

        // lhs = base_address (DWordWL), rhs = constant (i+1)
        let rhs_lo = bcast((i + 1) as u64);
        // rhs_hi = 0, so we omit it

        // Compute condition: sum of cond_cols
        let cond = match i {
            0 => write2 + write4 + write8, // w2
            1 | 2 => write4 + write8,      // w4
            _ => write8,                    // write8 for i=3..6
        };

        // carry_0 = (lhs_lo + rhs_lo - sum_lo) * inv_2_32
        let carry_0 = (base_addr_0 + rhs_lo - sum_lo) * inv_2_32;
        // cond * carry_0 * (1 - carry_0)
        results[idx] = cond * carry_0 * (one - carry_0);
        idx += 1;

        // carry_1 = (lhs_hi + rhs_hi + carry_0 - sum_hi) * inv_2_32
        // rhs_hi = 0
        let carry_1 = (base_addr_1 + carry_0 - sum_hi) * inv_2_32;
        // cond * carry_1 * (1 - carry_1)
        results[idx] = cond * carry_1 * (one - carry_1);
        idx += 1;
    }
    debug_assert_eq!(idx, 16);
}

//! Tests for the ECSM core table — constraint satisfaction on generated traces (x-only and
//! affine rows), the single-source constraint count, and isolated negative checks for the
//! padding closure, the scalar-bit padding guard, the `xG < p` / `yR < p` overflows and the
//! `IS_AFFINE` mode selector.

use crate::tables::ecsm::{EcsmConstraints, EcsmOperation, cols, generate_ecsm_trace};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use ecsm::{N_BYTES, P_BYTES, compute_witness, compute_witness_with_y};
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

// Constraint indices in the single `EcsmConstraints::eval` body (see the index map there).
const IDX_KBITS_ZERO: usize = 257; // KBitsZeroOnPadding
const IDX_X2_CONV0: usize = 258; // ConvCarry(X2, 0)
const IDX_YG_CONV0: usize = 323; // ConvCarry(Yg, 0)
const IDX_Q1_BIT32: usize = 388; // IS_BIT(q1[32])
const IDX_XG_CARRY0: usize = 389; // CarryBit(XgLtP, 0)
const IDX_XG_OVERFLOW: usize = 396; // OverflowRequired(XgLtP)
const IDX_YR_CARRY0: usize = 413; // CarryBit(YrLtP, 0)
const IDX_YR_OVERFLOW: usize = 420; // OverflowRequired(YrLtP)
const IDX_IS_AFFINE_BIT: usize = 421; // IS_BIT(IS_AFFINE)
const IDX_AFFINE_PADDING: usize = 422; // AffineZeroOnPadding

fn gx_le() -> [u8; 32] {
    // secp256k1 Gx, little-endian.
    let mut be = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    be.reverse();
    be
}

fn k_le(v: u64) -> [u8; 32] {
    let mut k = [0u8; 32];
    k[..8].copy_from_slice(&v.to_le_bytes());
    k
}

fn gy_le() -> [u8; 32] {
    // secp256k1 Gy, little-endian.
    let mut be = [
        0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65, 0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08,
        0xA8, 0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19, 0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10,
        0xD4, 0xB8,
    ];
    be.reverse();
    be
}

fn op_for(k: u64) -> EcsmOperation {
    let witness = compute_witness(&k_le(k), &gx_le()).unwrap();
    EcsmOperation {
        timestamp: 444,
        addr_xg: 0x2000,
        addr_k: 0x3000,
        addr_xr: 0x1000,
        is_affine: false,
        witness,
    }
}

/// Affine-variant row: `yG` comes from the caller's buffer instead of the even lift, and
/// `IS_AFFINE = 1`.
fn affine_op_for(k: u64) -> EcsmOperation {
    let witness = compute_witness_with_y(&k_le(k), &gx_le(), &gy_le()).unwrap();
    EcsmOperation {
        timestamp: 448,
        addr_xg: 0x2000,
        addr_k: 0x3000,
        addr_xr: 0x1000,
        is_affine: true,
        witness,
    }
}

/// Evaluate the ECSM [`ConstraintSet`] on a single main-trace row (the compiled
/// prover folder path), returning every base-field constraint value.
fn eval_main_row(main: Vec<FE>) -> Vec<FE> {
    let n = EcsmConstraints.meta().len();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![main],
        vec![vec![]],
    )]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    EcsmConstraints.eval(&mut folder);
    base
}

/// Evaluate the constraint set on trace row `row`.
fn eval_row(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<FE> {
    let main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect();
    eval_main_row(main)
}

/// Every ECSM constraint evaluates to zero on a generated trace (real + padding
/// rows). Exercises the all-zero padding closure (µ-gated `p²` and `b`) end to end.
#[test]
fn constraints_hold_on_generated_trace() {
    let ops: Vec<EcsmOperation> = [1u64, 2, 5, 0xFFFF, 1_000_003]
        .iter()
        .map(|&k| op_for(k))
        .collect();
    let trace = generate_ecsm_trace(&ops);

    for row in 0..trace.num_rows() {
        for (i, v) in eval_row(&trace, row).iter().enumerate() {
            assert_eq!(*v, FE::zero(), "constraint {i} must hold at row {row}");
        }
    }
}

#[test]
fn constraint_set_count() {
    assert_eq!(EcsmConstraints.meta().len(), 423);
}

/// One prover serves both ecall variants, so affine rows (`IS_AFFINE = 1`, `yG` read from
/// the caller's buffer) and x-only rows must satisfy the same constraint set in one trace.
/// Covers `IS_BIT(IS_AFFINE)` and `AffineZeroOnPadding` with the selector actually set.
#[test]
fn constraints_hold_on_mixed_affine_and_xonly_trace() {
    let ops = vec![
        affine_op_for(1),
        op_for(2),
        affine_op_for(0xFFFF),
        op_for(1_000_003),
    ];
    let trace = generate_ecsm_trace(&ops);
    assert_eq!(
        *trace.main_table.get(0, cols::IS_AFFINE),
        FE::one(),
        "row 0 is an affine row"
    );
    assert_eq!(
        *trace.main_table.get(1, cols::IS_AFFINE),
        FE::zero(),
        "row 1 is an x-only row"
    );

    for row in 0..trace.num_rows() {
        for (i, v) in eval_row(&trace, row).iter().enumerate() {
            assert_eq!(*v, FE::zero(), "constraint {i} must hold at row {row}");
        }
    }
}

/// `IS_AFFINE` must be a bit, and zero on padding — otherwise a witness could fire the
/// affine-gated yG-read / yR-write MEMW buses on rows that never ran an affine ecall.
#[test]
fn affine_selector_must_be_a_bit_and_zero_on_padding() {
    let row_with = |mu: u64, is_affine: u64| {
        let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
        main[cols::MU] = FE::from(mu);
        main[cols::IS_AFFINE] = FE::from(is_affine);
        eval_main_row(main)
    };

    assert_ne!(
        row_with(1, 2)[IDX_IS_AFFINE_BIT],
        FE::zero(),
        "IS_BIT must fire for a non-boolean IS_AFFINE"
    );
    assert_ne!(
        row_with(0, 1)[IDX_AFFINE_PADDING],
        FE::zero(),
        "AffineZeroOnPadding must fire for IS_AFFINE = 1 on a padding row"
    );
    for is_affine in [0, 1] {
        let row = row_with(1, is_affine);
        assert_eq!(row[IDX_IS_AFFINE_BIT], FE::zero());
        assert_eq!(row[IDX_AFFINE_PADDING], FE::zero());
    }
}

/// OverflowRequired for YrLtP fires when yR = p, the check that keeps the published `yR`
/// canonical (the byte range checks alone only bound it below 2^256, and the quotient
/// columns would absorb the extra multiple of p). Mirrors the xG case: all CarryBit
/// constraints still hold, but the chain never reaches c_7 = 1.
#[test]
fn yr_ge_p_overflow_required_fires() {
    let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
    main[cols::MU] = FE::one();
    // yR = p, yr_sub_p = 0 (invalid subtraction witness — fine for this isolation test).
    for (i, &b) in P_BYTES.iter().enumerate() {
        main[cols::YR + i] = FE::from(b as u64);
    }
    let row = eval_main_row(main);

    for i in 0..7 {
        assert_eq!(
            row[IDX_YR_CARRY0 + i],
            FE::zero(),
            "carry bit {i}: c_i=0 is a valid bit"
        );
    }
    assert_ne!(
        row[IDX_YR_OVERFLOW],
        FE::zero(),
        "OverflowRequired must fire when yR = p"
    );
}

/// The yG carry recurrence closes on all-zero padding because both the `µ·p²` offset and the
/// curve constant `µ·b` are multiplied by `µ`, so they vanish when `µ = 0`. This checks the
/// closing argument (Yg limb-0 ConvCarry = constraint `IDX_YG_CONV0`) and its two ingredients.
#[test]
fn yg_padding_closes_via_mu_gated_p2_and_b() {
    // yG limb-0 ConvCarry residual on a one-off row with the given `µ` and `q1`.
    let yg_residual = |mu: u64, q1_is_p: bool| {
        let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
        main[cols::MU] = FE::from(mu);
        if q1_is_p {
            for (i, &b) in P_BYTES.iter().enumerate() {
                main[cols::Q1 + i] = FE::from(b as u64);
            }
        }
        eval_main_row(main)[IDX_YG_CONV0]
    };

    // Padding row (µ=0, q1=0): µ gates away both p² and b → closes trivially.
    assert_eq!(
        yg_residual(0, false),
        FE::zero(),
        "padding row (µ=0, q1=0) must close"
    );

    // µ=0 but q1=p: µ·p² is gated away, so q1·p is unmatched → residual = +P_0² = +47² = 2209.
    assert_eq!(
        yg_residual(0, true),
        FE::from(2209u64),
        "µ=0 with q1=p leaves +P_0² = +47² residual"
    );

    // µ=1, q1=0: s_i = µ·P_0² − µ·b = 2209 − 7 = 2202 → residual (256·c − c_prev − s_i) = −2202.
    assert_eq!(
        yg_residual(1, false),
        FE::zero() - FE::from(2202u64),
        "µ=1, q1=0: residual = −(P_0² − b) = −2202"
    );

    // x² has no standalone constant → closes on an all-zero padding row regardless.
    let mut zero = vec![FE::zero(); cols::NUM_COLUMNS];
    zero[cols::MU] = FE::zero();
    assert_eq!(
        eval_main_row(zero)[IDX_X2_CONV0],
        FE::zero(),
        "x² closes on all-zero padding (no standalone constant)"
    );
}

/// A µ=0 padding row with any k_bit set must violate KBitsZeroOnPadding.
/// Guards against a prover injecting phantom `Bit` bus receives on padding rows.
#[test]
fn k_bits_zero_on_padding_rejects_forged_row() {
    // Single forged bit on padding row: sum=1, (1−µ)=1 → 1.
    let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
    main[cols::k_bit(0)] = FE::one();
    assert_eq!(
        eval_main_row(main.clone())[IDX_KBITS_ZERO],
        FE::one(),
        "k_bit[0]=1 on µ=0 must fire (residual=1)"
    );

    // Same bit on an active row (µ=1): constraint holds.
    main[cols::MU] = FE::one();
    assert_eq!(
        eval_main_row(main)[IDX_KBITS_ZERO],
        FE::zero(),
        "k_bit[0]=1 on µ=1 must not fire"
    );

    // Multiple forged bits: sum=3, residual=3.
    let mut main_multi = vec![FE::zero(); cols::NUM_COLUMNS];
    main_multi[cols::k_bit(0)] = FE::one();
    main_multi[cols::k_bit(7)] = FE::one();
    main_multi[cols::k_bit(255)] = FE::one();
    assert_eq!(
        eval_main_row(main_multi)[IDX_KBITS_ZERO],
        FE::from(3u64),
        "3 forged k_bits → residual=3"
    );
}

/// OverflowRequired for XgLtP evaluates non-zero when xG = p (no valid xg_sub_p exists).
/// All CarryBit constraints still hold (c_i=0 is a valid bit), but the carry chain never
/// reaches c_7=1, so OverflowRequired = µ·(1−c_7) = 1 fires.
#[test]
fn xg_ge_p_overflow_required_fires() {
    let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
    main[cols::MU] = FE::one();
    // xG = p, xg_sub_p = 0 (invalid subtraction witness — fine for this isolation test).
    for (i, &b) in P_BYTES.iter().enumerate() {
        main[cols::xg(i)] = FE::from(b as u64);
    }
    let row = eval_main_row(main);

    for i in 0..7 {
        assert_eq!(
            row[IDX_XG_CARRY0 + i],
            FE::zero(),
            "carry bit {i}: c_i=0 is a valid bit"
        );
    }
    assert_ne!(
        row[IDX_XG_OVERFLOW],
        FE::zero(),
        "OverflowRequired must fire when xG = p"
    );
}

fn five_g_x_le() -> [u8; 32] {
    // x-coordinate of 5·G (secp256k1), little-endian.
    // 0x2f8bde4d1a07209355b4a7250a5c5128e88b84bddc619ab7cba8d569b240efe4
    [
        0xe4, 0xef, 0x40, 0xb2, 0x69, 0xd5, 0xa8, 0xcb, 0xb7, 0x9a, 0x61, 0xdc, 0xbd, 0x84, 0x8b,
        0xe8, 0x28, 0x51, 0x5c, 0x0a, 0x25, 0xa7, 0xb4, 0x55, 0x93, 0x20, 0x07, 0x1a, 0x4d, 0xde,
        0x8b, 0x2f,
    ]
}

/// Exercises the q1[32]=1 path by using x(5·G) as the base point, which produces a yG
/// quotient whose high byte (index 32) equals 1. IS_BIT(q1[32]) must still hold.
#[test]
fn q1_bit32_equals_one_path() {
    let witness =
        compute_witness(&k_le(1), &five_g_x_le()).expect("k=1, xG=x(5·G) is a valid ECSM input");
    assert_eq!(
        witness.q1[32], 1,
        "sanity: q1[32] should be 1 for x(5·G) as base point"
    );

    let op = EcsmOperation {
        timestamp: 100,
        addr_xg: 0x2000,
        addr_k: 0x3000,
        addr_xr: 0x1000,
        is_affine: false,
        witness,
    };
    let trace = generate_ecsm_trace(&[op]);

    for row in 0..trace.num_rows() {
        assert_eq!(
            eval_row(&trace, row)[IDX_Q1_BIT32],
            FE::zero(),
            "IS_BIT(q1[32]) must hold (value=1) at row {row}"
        );
    }
}

/// End-to-end constraint check for k = N−1, the maximum valid scalar (len_k = 255).
#[test]
fn constraints_hold_for_k_eq_n_minus_one() {
    let mut k_bytes = N_BYTES;
    k_bytes[0] -= 1; // N is odd, so N_BYTES[0] >= 1; this gives N-1 in little-endian.
    let witness = compute_witness(&k_bytes, &gx_le()).expect("N-1 is a valid scalar");
    assert_eq!(witness.len_k, 255, "N-1 has MSB at bit 255");

    let op = EcsmOperation {
        timestamp: 999,
        addr_xg: 0x2000,
        addr_k: 0x3000,
        addr_xr: 0x1000,
        is_affine: false,
        witness,
    };
    let trace = generate_ecsm_trace(&[op]);

    for row in 0..trace.num_rows() {
        for (i, v) in eval_row(&trace, row).iter().enumerate() {
            assert_eq!(*v, FE::zero(), "constraint {i} must hold at row {row}");
        }
    }
}

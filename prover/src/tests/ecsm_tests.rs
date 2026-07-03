//! Tests for the ECSM core table — constraint satisfaction on generated traces,
//! constraint count, and the yG padding-closure argument.

use crate::constraints::templates::IsBitConstraint;
use crate::tables::ecsm::{
    CarryBit, ColIsZero, ConvCarry, EcsmOperation, KBitsZeroOnPadding, OverflowKind,
    OverflowRequired, Relation, cols, create_constraints, generate_ecsm_trace,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use ecsm::{N_BYTES, P_BYTES, compute_witness};
use stark::constraints::transition::TransitionConstraint;
use stark::table::TableView;
use stark::trace::TraceTable;

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

fn op_for(k: u64) -> EcsmOperation {
    let witness = compute_witness(&k_le(k), &gx_le()).unwrap();
    EcsmOperation {
        timestamp: 444,
        addr_xg: 0x2000,
        addr_k: 0x3000,
        addr_xr: 0x1000,
        witness,
    }
}

fn row_view(
    trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
    row: usize,
) -> TableView<GoldilocksField, GoldilocksExtension> {
    let main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect();
    TableView::new(vec![main], vec![])
}

/// Every ECSM constraint evaluates to zero on a generated trace (real + padding rows).
#[test]
fn constraints_hold_on_generated_trace() {
    let ops: Vec<EcsmOperation> = [1u64, 2, 5, 0xFFFF, 1_000_003]
        .iter()
        .map(|&k| op_for(k))
        .collect();
    let trace = generate_ecsm_trace(&ops);

    for row in 0..trace.num_rows() {
        let view = row_view(&trace, row);
        // Re-evaluate concrete constraints (mirror create_constraints) at this row.
        assert_eq!(
            IsBitConstraint::unconditional(cols::MU, 0).evaluate(&view),
            FE::zero(),
            "is_bit(mu) row {row}"
        );
        for i in 0..256 {
            assert_eq!(
                IsBitConstraint::unconditional(cols::k_bit(i), 0).evaluate(&view),
                FE::zero(),
                "is_bit(k_bit[{i}]) row {row}"
            );
        }
        assert_eq!(
            KBitsZeroOnPadding { constraint_idx: 0 }.evaluate(&view),
            FE::zero(),
            "k_bits_zero_on_padding row {row}"
        );
        for i in 0..64 {
            for relation in [Relation::X2, Relation::Yg] {
                let v = ConvCarry {
                    relation,
                    i,
                    constraint_idx: 0,
                }
                .evaluate(&view);
                assert_eq!(v, FE::zero(), "conv carry i={i} row {row}");
            }
        }
        assert_eq!(
            ColIsZero {
                col: cols::c0(63),
                constraint_idx: 0
            }
            .evaluate(&view),
            FE::zero()
        );
        assert_eq!(
            ColIsZero {
                col: cols::c1(63),
                constraint_idx: 0
            }
            .evaluate(&view),
            FE::zero()
        );
        for kind in [OverflowKind::XgLtP, OverflowKind::KLtN, OverflowKind::XrLtP] {
            for i in 0..7 {
                assert_eq!(
                    CarryBit {
                        kind,
                        i,
                        constraint_idx: 0
                    }
                    .evaluate(&view),
                    FE::zero(),
                    "carry bit kind i={i} row {row}"
                );
            }
            assert_eq!(
                OverflowRequired {
                    kind,
                    constraint_idx: 0
                }
                .evaluate(&view),
                FE::zero(),
                "overflow required row {row}"
            );
        }
    }
}

#[test]
fn create_constraints_count() {
    let (constraints, next) = create_constraints(0);
    assert_eq!(constraints.len(), 413);
    assert_eq!(next, 413);
}

/// The yG carry recurrence is unsatisfiable on a padding row unless two ingredients hold,
/// and this test locks both:
///   (a) `q1` pads to `p`, so the `p² − q1·p` offset cancels;
///   (b) the curve constant `b` is multiplied by `µ`, so it drops when `µ = 0`.
/// Removing either ingredient leaves a nonzero residual on the yG limb-0 relation.
/// The x² relation has no standalone constant, so it closes on all-zero padding and is
/// left fully unconditional.
#[test]
fn yg_padding_closes_via_q1_eq_p_and_mu_gated_b() {
    // yG limb-0 ConvCarry residual on a one-off row with the given `µ` and `q1`.
    let yg_residual = |mu: u64, q1_is_p: bool| {
        let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
        main[cols::MU] = FE::from(mu);
        if q1_is_p {
            for (i, &b) in P_BYTES.iter().enumerate() {
                main[cols::Q1 + i] = FE::from(b as u64);
            }
        }
        let view: TableView<GoldilocksField, GoldilocksExtension> =
            TableView::new(vec![main], vec![]);
        ConvCarry {
            relation: Relation::Yg,
            i: 0,
            constraint_idx: 0,
        }
        .evaluate(&view)
    };

    // The padding row this chip emits (µ = 0, q1 = p): both ingredients present → closes.
    assert_eq!(
        yg_residual(0, true),
        FE::zero(),
        "padding row (µ=0, q1=p) must close"
    );

    // Drop ingredient (a): q1 = 0 instead of p → the p² offset is uncancelled.
    assert_eq!(
        yg_residual(0, false),
        FE::zero() - FE::from(2209u64),
        "without q1=p the residual is −P_0² = −47²"
    );

    // Drop ingredient (b): force the row active (µ = 1) so the curve constant `b`
    // survives even with q1 = p. Residual = b = 7.
    assert_eq!(
        yg_residual(1, true),
        FE::from(7u64),
        "with µ=1 (b ungated) the leftover residual is the curve constant b=7"
    );

    // x² has no standalone constant → closes on an all-zero padding row regardless.
    let mut zero = vec![FE::zero(); cols::NUM_COLUMNS];
    zero[cols::MU] = FE::zero();
    let zview: TableView<GoldilocksField, GoldilocksExtension> = TableView::new(vec![zero], vec![]);
    assert_eq!(
        ConvCarry {
            relation: Relation::X2,
            i: 0,
            constraint_idx: 0,
        }
        .evaluate(&zview),
        FE::zero(),
        "x² closes on all-zero padding (no standalone constant)"
    );
}

/// A µ=0 padding row with any k_bit set must violate KBitsZeroOnPadding.
/// This guards against a prover injecting phantom BIT bus receives on padding rows.
#[test]
fn k_bits_zero_on_padding_rejects_forged_row() {
    let c = KBitsZeroOnPadding { constraint_idx: 0 };

    // Single forged bit on padding row: sum=1, (1−µ)=1 → 1.
    let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
    main[cols::k_bit(0)] = FE::one();
    let view: TableView<GoldilocksField, GoldilocksExtension> = TableView::new(vec![main.clone()], vec![]);
    assert_eq!(c.evaluate(&view), FE::one(), "k_bit[0]=1 on µ=0 must fire (residual=1)");

    // Same bit on an active row (µ=1): constraint holds.
    main[cols::MU] = FE::one();
    let view_active: TableView<GoldilocksField, GoldilocksExtension> = TableView::new(vec![main.clone()], vec![]);
    assert_eq!(c.evaluate(&view_active), FE::zero(), "k_bit[0]=1 on µ=1 must not fire");

    // Multiple forged bits: sum=3, residual=3.
    let mut main_multi = vec![FE::zero(); cols::NUM_COLUMNS];
    main_multi[cols::k_bit(0)] = FE::one();
    main_multi[cols::k_bit(7)] = FE::one();
    main_multi[cols::k_bit(255)] = FE::one();
    let view_multi: TableView<GoldilocksField, GoldilocksExtension> = TableView::new(vec![main_multi], vec![]);
    assert_eq!(c.evaluate(&view_multi), FE::from(3u64), "3 forged k_bits → residual=3");
}

/// OverflowRequired for XgLtP evaluates non-zero when xG = p (no valid xg_sub_p exists).
/// All CarryBit constraints still hold (c_i=0 is a valid bit), but the carry chain never
/// reaches c_7=1, so OverflowRequired = µ·(1−c_7) = 1 fires.
#[test]
fn xg_ge_p_overflow_required_fires() {
    let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
    main[cols::MU] = FE::one();
    // xG = p, xg_sub_p = 0 (zero subtraction witness — invalid for a real prover but valid for
    // this constraint isolation test).
    for (i, &b) in P_BYTES.iter().enumerate() {
        main[cols::xg(i)] = FE::from(b as u64);
    }
    let view: TableView<GoldilocksField, GoldilocksExtension> = TableView::new(vec![main], vec![]);

    for i in 0..7 {
        assert_eq!(
            CarryBit { kind: OverflowKind::XgLtP, i, constraint_idx: 0 }.evaluate(&view),
            FE::zero(),
            "carry bit {i}: c_i=0 is a valid bit"
        );
    }
    assert_ne!(
        OverflowRequired { kind: OverflowKind::XgLtP, constraint_idx: 0 }.evaluate(&view),
        FE::zero(),
        "OverflowRequired must fire when xG = p"
    );
}

fn five_g_x_le() -> [u8; 32] {
    // x-coordinate of 5·G (secp256k1), little-endian.
    // 0x2f8bde4d1a07209355b4a7250a5c5128e88b84bddc619ab7cba8d569b240efe4
    [
        0xe4, 0xef, 0x40, 0xb2, 0x69, 0xd5, 0xa8, 0xcb, 0xb7, 0x9a, 0x61, 0xdc, 0xbd, 0x84,
        0x8b, 0xe8, 0x28, 0x51, 0x5c, 0x0a, 0x25, 0xa7, 0xb4, 0x55, 0x93, 0x20, 0x07, 0x1a,
        0x4d, 0xde, 0x8b, 0x2f,
    ]
}

/// Exercises the q1[32]=1 path by using x(5·G) as the base point, which produces a yG
/// quotient whose high byte (index 32) equals 1. IS_BIT(q1[32]) must still hold.
#[test]
fn q1_bit32_equals_one_path() {
    let witness = compute_witness(&k_le(1), &five_g_x_le())
        .expect("k=1, xG=x(5·G) is a valid ECSM input");
    assert_eq!(witness.q1[32], 1, "sanity: q1[32] should be 1 for x(5·G) as base point");

    let op = EcsmOperation { timestamp: 100, addr_xg: 0x2000, addr_k: 0x3000, addr_xr: 0x1000, witness };
    let trace = generate_ecsm_trace(&[op]);

    for row in 0..trace.num_rows() {
        let view = row_view(&trace, row);
        assert_eq!(
            IsBitConstraint::unconditional(cols::q1(32), 0).evaluate(&view),
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

    let op = EcsmOperation { timestamp: 999, addr_xg: 0x2000, addr_k: 0x3000, addr_xr: 0x1000, witness };
    let trace = generate_ecsm_trace(&[op]);

    for row in 0..trace.num_rows() {
        let view = row_view(&trace, row);
        assert_eq!(IsBitConstraint::unconditional(cols::MU, 0).evaluate(&view), FE::zero(), "is_bit(mu) row {row}");
        for i in 0..256 {
            assert_eq!(
                IsBitConstraint::unconditional(cols::k_bit(i), 0).evaluate(&view),
                FE::zero(),
                "is_bit(k_bit[{i}]) row {row}"
            );
        }
        assert_eq!(KBitsZeroOnPadding { constraint_idx: 0 }.evaluate(&view), FE::zero(), "k_bits_zero_on_padding row {row}");
        for i in 0..64 {
            for relation in [Relation::X2, Relation::Yg] {
                assert_eq!(ConvCarry { relation, i, constraint_idx: 0 }.evaluate(&view), FE::zero(), "conv carry i={i} row {row}");
            }
        }
        assert_eq!(ColIsZero { col: cols::c0(63), constraint_idx: 0 }.evaluate(&view), FE::zero());
        assert_eq!(ColIsZero { col: cols::c1(63), constraint_idx: 0 }.evaluate(&view), FE::zero());
        for kind in [OverflowKind::XgLtP, OverflowKind::KLtN, OverflowKind::XrLtP] {
            for i in 0..7 {
                assert_eq!(CarryBit { kind, i, constraint_idx: 0 }.evaluate(&view), FE::zero(), "carry bit kind i={i} row {row}");
            }
            assert_eq!(OverflowRequired { kind, constraint_idx: 0 }.evaluate(&view), FE::zero(), "overflow required row {row}");
        }
    }
}

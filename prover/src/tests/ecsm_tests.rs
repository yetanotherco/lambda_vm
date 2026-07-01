//! Tests for the ECSM core table — constraint satisfaction on generated traces,
//! constraint count, and the yG padding-closure argument.

use crate::constraints::templates::IsBitConstraint;
use crate::tables::ecsm::{
    CarryBit, ColIsZero, ConvCarry, EcsmOperation, OverflowKind, OverflowRequired, Relation, cols,
    create_constraints, generate_ecsm_trace,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use ecsm::{P_BYTES, compute_witness};
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
    assert_eq!(constraints.len(), 412);
    assert_eq!(next, 412);
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

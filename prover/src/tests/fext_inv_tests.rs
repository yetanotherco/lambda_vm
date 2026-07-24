//! Tests for the FEXT_INV table: constraint satisfaction on generated traces,
//! the constraint/bus counts, the zero-flag branch, and negative checks for a
//! wrong (or forged) inverse.

use crate::tables::fext_inv::{
    FextInvConstraints, FextInvOperation, bus_interactions, cols, generate_fext_inv_trace,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksElement;
use math::field::traits::IsField;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

/// True Fp3 inverse (canonical coefficients), or `[0;3]` for the zero element.
fn inv(x: [u64; 3]) -> [u64; 3] {
    let a = [
        GoldilocksElement::from(x[0]),
        GoldilocksElement::from(x[1]),
        GoldilocksElement::from(x[2]),
    ];
    match Degree3GoldilocksExtensionField::inv(&a) {
        Ok(i) => [
            i[0].canonical_u64(),
            i[1].canonical_u64(),
            i[2].canonical_u64(),
        ],
        Err(_) => [0, 0, 0],
    }
}

fn op(x: [u64; 3]) -> FextInvOperation {
    FextInvOperation {
        timestamp: 100,
        x_addr: 0x10,
        out_addr: 0x40,
        x,
        inv: inv(x),
        read_old_ts: [0; 3],
        write_old_ts: [0; 3],
        write_old_val: [0; 3],
    }
}

/// Evaluate the FEXT_INV constraint set on one main-trace row.
fn eval_main_row(main: Vec<FE>) -> Vec<FE> {
    let n = FextInvConstraints.meta().len();
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
    FextInvConstraints.eval(&mut folder);
    base
}

fn eval_row(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<FE> {
    let main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect();
    eval_main_row(main)
}

#[test]
fn fext_inv_constraint_count_is_eight() {
    // idx 0: IS_BIT(μ); idx 1: IS_BIT(is_zero); idx 2-4: x·inv = 1−is_zero;
    // idx 5-7: xd·is_zero = 0.
    assert_eq!(FextInvConstraints.meta().len(), 8);
}

#[test]
fn fext_inv_max_degree_is_two() {
    assert_eq!(FextInvConstraints.max_degree(), 2);
}

#[test]
fn fext_inv_bus_interaction_count() {
    // 1 Ecall + 2 register reads + 3 field reads + 3 field writes, each field
    // access = consume-old + emit-new + old_ts<ts = 1 + 2 + 9 + 9.
    assert_eq!(bus_interactions().len(), 21);
}

#[test]
fn fext_inv_constraints_hold_on_valid_trace() {
    let ops = vec![
        op([1, 0, 0]),
        op([0, 1, 0]),
        op([0, 0, 1]),
        op([1, 2, 3]),
        op([12345, 67890, 111213]),
        op([0, 0, 0]), // zero element: is_zero = 1, inv = 0
    ];
    let trace = generate_fext_inv_trace(&ops);

    for row in 0..trace.num_rows() {
        for (idx, v) in eval_row(&trace, row).into_iter().enumerate() {
            assert_eq!(v, FE::zero(), "row {row}, constraint {idx} nonzero");
        }
    }
}

#[test]
fn fext_inv_zero_element_sets_flag() {
    let ops = vec![op([0, 0, 0])];
    let trace = generate_fext_inv_trace(&ops);
    assert_eq!(*trace.main_table.get(0, cols::IS_ZERO), FE::one());
}

#[test]
fn fext_inv_detects_wrong_inverse() {
    let ops = vec![op([1, 2, 3])];
    let mut trace = generate_fext_inv_trace(&ops);
    // Corrupt inv0: the product x·inv no longer equals 1, breaking a coeff eq.
    let bad = *trace.main_table.get(0, cols::INV0) + FE::one();
    trace.main_table.set_fe(0, cols::INV0, bad);
    let vals = eval_row(&trace, 0);
    assert!(
        vals[2] != FE::zero() || vals[3] != FE::zero() || vals[4] != FE::zero(),
        "corrupted inverse must break x·inv = 1"
    );
}

#[test]
fn fext_inv_cannot_forge_zero_inverse() {
    // Claim the zero element has an inverse by clearing its is_zero flag: the
    // product is 0 but 1−is_zero = 1, so constraint 2 must reject.
    let ops = vec![op([0, 0, 0])];
    let mut trace = generate_fext_inv_trace(&ops);
    trace.main_table.set_fe(0, cols::IS_ZERO, FE::zero());
    trace.main_table.set_fe(0, cols::INV0, FE::from(1u64));
    let vals = eval_row(&trace, 0);
    assert_ne!(
        vals[2],
        FE::zero(),
        "forged zero-inverse must break constraint 2"
    );
}

#[test]
fn fext_inv_flag_forced_zero_for_nonzero_x() {
    // Setting is_zero = 1 while x ≠ 0 must break xd·is_zero = 0.
    let ops = vec![op([5, 0, 0])];
    let mut trace = generate_fext_inv_trace(&ops);
    trace.main_table.set_fe(0, cols::IS_ZERO, FE::one());
    let vals = eval_row(&trace, 0);
    assert_ne!(
        vals[5],
        FE::zero(),
        "is_zero = 1 with x0 ≠ 0 must break constraint 5"
    );
}

#[test]
fn fext_inv_detects_non_bit_mu() {
    let ops = vec![op([1, 2, 3])];
    let mut trace = generate_fext_inv_trace(&ops);
    trace.main_table.set_fe(0, cols::MU, FE::from(2u64));
    let vals = eval_row(&trace, 0);
    assert_ne!(
        vals[0],
        FE::zero(),
        "μ = 2 must break IS_BIT (constraint 0)"
    );
}

#[test]
fn fext_inv_product_is_one() {
    // Independent check that `inv` really inverts (anchors the trace helper).
    for x in [[1u64, 2, 3], [7, 0, 0], [0, 0, 5], [999, 1000, 1001]] {
        let a = [
            GoldilocksElement::from(x[0]),
            GoldilocksElement::from(x[1]),
            GoldilocksElement::from(x[2]),
        ];
        let i = inv(x);
        let b = [
            GoldilocksElement::from(i[0]),
            GoldilocksElement::from(i[1]),
            GoldilocksElement::from(i[2]),
        ];
        let prod = Degree3GoldilocksExtensionField::mul(&a, &b);
        assert_eq!(
            [
                prod[0].canonical_u64(),
                prod[1].canonical_u64(),
                prod[2].canonical_u64()
            ],
            [1, 0, 0],
            "x·inv must be 1 for x = {x:?}"
        );
    }
}

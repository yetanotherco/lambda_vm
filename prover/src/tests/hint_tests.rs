//! HINT constraint tests.

use crate::tables::hint::{HintConstraints, HintOperation, cols, generate_hint_trace};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

/// Evaluate the HINT constraint set on one main-trace row.
fn eval_main_row(main: Vec<FE>) -> Vec<FE> {
    let n = HintConstraints.meta().len();
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
    HintConstraints.eval(&mut folder);
    base
}

fn op(timestamp: u64, out_addr: u64) -> HintOperation {
    HintOperation {
        timestamp,
        out_addr,
        out_bytes: std::array::from_fn(|i| i as u8),
        hint_id: 0,
        in_addr: 0x3000,
    }
}

#[test]
fn constraint_set_count() {
    assert_eq!(HintConstraints.meta().len(), 1);
}

/// Every constraint holds on a generated trace — real rows (`mu = 1`) and the
/// all-zero padding rows (`mu = 0`) alike.
#[test]
fn constraints_hold_on_generated_trace() {
    let trace = generate_hint_trace(&[op(4, 0x1000), op(8, 0x2000)]);
    for row in 0..trace.num_rows() {
        let main: Vec<FE> = (0..cols::NUM_COLUMNS)
            .map(|c| *trace.main_table.get(row, c))
            .collect();
        for (i, v) in eval_main_row(main).iter().enumerate() {
            assert_eq!(*v, FE::zero(), "constraint {i} must hold at row {row}");
        }
    }
}

/// `IS_BIT(mu)` rejects a row whose multiplicity is not a bit.
///
/// The `Ecall` bus does not establish this on its own: its tuple carries a
/// per-instruction timestamp, so LogUp pins the *sum* of `mu` over the rows sharing a
/// tuple, which a witness can satisfy by spreading `mu` across rows with integer
/// weights summing to 1 (the real exploit uses a `+1`/`-1` pair, not a fractional
/// split; MEMW does not catch it — it only sees the legal `+1`, the `-1` cancelling an
/// honest STORE). This constraint rejects any non-boolean `mu` locally. The test below
/// tampers with a fractional `1/2`, which `IS_BIT` also rejects.
#[test]
fn is_bit_mu_rejects_non_boolean_multiplicity() {
    let trace = generate_hint_trace(&[op(4, 0x1000)]);
    let mut main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(0, c))
        .collect();
    assert_eq!(main[cols::MU], FE::one(), "row 0 must be a real hint row");

    // A halved multiplicity: 1/2 + 1/2 across two rows keeps the Ecall bus balanced.
    let half = (FE::one() / (FE::one() + FE::one())).expect("2 is invertible");
    main[cols::MU] = half;
    assert_ne!(
        eval_main_row(main.clone())[0],
        FE::zero(),
        "IS_BIT(mu) must reject a fractional multiplicity"
    );

    // And any other non-bit value.
    main[cols::MU] = FE::from(2u64);
    assert_ne!(
        eval_main_row(main)[0],
        FE::zero(),
        "IS_BIT(mu) must reject mu = 2"
    );
}

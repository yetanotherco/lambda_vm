use crate::constraints::builder::ConstraintMeta;
use crate::constraints::transition::TransitionConstraintEvaluator;
use crate::constraints::zerofier;
use crate::traits::TransitionEvaluationContext;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsFFTField;
use std::marker::PhantomData;

/// Dummy evaluator that only exposes the trait knobs we need (`period`, `offset`,
/// `end_exemptions`) to exercise the OLD trait-default `end_exemptions_roots`.
///
/// Kept solely for the period ≠ 1 case below; it dies with the trait machinery
/// in the final deletion phase.
struct DummyConstraint<F: IsFFTField + Send + Sync> {
    period: usize,
    offset: usize,
    end_exemptions: usize,
    phantom: PhantomData<F>,
}

impl<F: IsFFTField + Send + Sync> TransitionConstraintEvaluator<F, F> for DummyConstraint<F> {
    fn degree(&self) -> usize {
        1
    }
    fn constraint_idx(&self) -> usize {
        0
    }
    fn period(&self) -> usize {
        self.period
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn end_exemptions(&self) -> usize {
        self.end_exemptions
    }
    fn evaluate_verifier(&self, _: &TransitionEvaluationContext<F, F>, _: &mut [FieldElement<F>]) {}
}

#[test]
fn end_exemptions_roots_default_offset_matches_last_rows() {
    let trace_length = 8usize;
    let g =
        GoldilocksField::get_primitive_root_of_unity(trace_length.trailing_zeros() as u64).unwrap();
    let meta = ConstraintMeta::base(0, 1).with_end_exemptions(2);

    let roots = zerofier::end_exemptions_roots(&meta, &g, trace_length);

    // Constraint applies on rows 0..8; last two rows are 6 and 7.
    assert_eq!(roots, vec![g.pow(7u64), g.pow(6u64)]);
}

// NOTE(single-source constraints): this case exists PURELY to exercise the
// period ≠ 1 zerofier shape, which no production constraint uses. It stays on
// the OLD trait path untouched and dies together with the period machinery in
// the final deletion phase.
#[test]
fn end_exemptions_roots_nonzero_offset_walks_the_offset_domain() {
    let trace_length = 8usize;
    let g =
        GoldilocksField::get_primitive_root_of_unity(trace_length.trailing_zeros() as u64).unwrap();
    let c = DummyConstraint::<GoldilocksField> {
        period: 2,
        offset: 1,
        end_exemptions: 2,
        phantom: PhantomData,
    };

    let roots = c.end_exemptions_roots(&g, trace_length);

    // Constraint applies on rows {1, 3, 5, 7}; last two are 5 and 7.
    assert_eq!(roots, vec![g.pow(7u64), g.pow(5u64)]);
}

#[test]
fn end_exemptions_roots_zero_exemptions_is_empty() {
    let trace_length = 8usize;
    let g =
        GoldilocksField::get_primitive_root_of_unity(trace_length.trailing_zeros() as u64).unwrap();
    let meta = ConstraintMeta::base(0, 1);

    assert!(zerofier::end_exemptions_roots::<GoldilocksField>(&meta, &g, trace_length).is_empty());
}

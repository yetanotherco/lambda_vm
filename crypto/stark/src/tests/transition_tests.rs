use crate::constraints::transition::TransitionConstraintEvaluator;
use crate::traits::TransitionEvaluationContext;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsFFTField;
use std::marker::PhantomData;

/// Dummy evaluator that only exposes the trait knobs we need (`period`, `offset`,
/// `end_exemptions`) to exercise `end_exemptions_roots`.
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
    let c = DummyConstraint::<GoldilocksField> {
        period: 1,
        offset: 0,
        end_exemptions: 2,
        phantom: PhantomData,
    };

    let roots = c.end_exemptions_roots(&g, trace_length);

    // Constraint applies on rows 0..8; last two rows are 6 and 7.
    assert_eq!(roots, vec![g.pow(7u64), g.pow(6u64)]);
}

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
    let c = DummyConstraint::<GoldilocksField> {
        period: 1,
        offset: 0,
        end_exemptions: 0,
        phantom: PhantomData,
    };

    assert!(c.end_exemptions_roots(&g, trace_length).is_empty());
}

/// `DummyConstraint` doesn't override `capture`, so it exercises the default
/// `TransitionConstraintEvaluator::capture` body — which must not panic (see
/// `crypto/stark/src/constraints/transition.rs`) and must mark the resulting
/// `ConstraintProgram` incomplete via `IrBuilder::mark_unsupported`, so
/// `ConstraintEvaluator`/the verifier fall back to the boxed path instead of
/// interpreting a partial program. This is the regression test for the
/// `cargo test -p stark --features constraint-ir` panic fixed alongside this
/// test (every `examples/`/test-only AIR relies on this default).
#[test]
fn default_capture_marks_program_incomplete_without_panicking() {
    use crate::constraint_ir::IrBuilder;

    let c = DummyConstraint::<GoldilocksField> {
        period: 1,
        offset: 0,
        end_exemptions: 0,
        phantom: PhantomData,
    };

    let mut b = IrBuilder::new();
    c.capture(&mut b); // must not panic
    let prog = b.finish(0);

    assert!(
        !prog.complete,
        "a constraint with no Capture impl must mark the program incomplete"
    );
}

// Simple constraint evaluator using the new simplified constraint system
//
// This evaluator wraps the Constraints struct from simple.rs and provides
// a convenient interface for evaluation with owned precomputes.

use super::simple::{Constraints, EvalPrecomputes, EvalResult, evaluate_at_point};
use crate::domain::Domain;
use crate::trace::LDETraceTable;
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Evaluator for simplified constraints.
///
/// This struct owns the precomputes and provides a convenient interface
/// for evaluating constraints. It delegates to the shared `evaluate_at_point`
/// function from simple.rs.
pub struct SimpleConstraintEvaluator<'a, F, E>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    constraints: &'a Constraints<F, E>,
    precomputes: EvalPrecomputes<F, E>,
    frame_offsets: Vec<usize>,
}

impl<'a, F, E> SimpleConstraintEvaluator<'a, F, E>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    pub fn new(
        constraints: &'a Constraints<F, E>,
        domain: &Domain<F>,
        beta: &FieldElement<E>,
        frame_offsets: Vec<usize>,
    ) -> Self {
        let exemption_counts = constraints.exemption_counts();
        let boundary_rows = constraints.boundary_rows();

        let precomputes = EvalPrecomputes::new(
            domain,
            beta,
            constraints.num_constraints(),
            &exemption_counts,
            &boundary_rows,
        );

        Self {
            constraints,
            precomputes,
            frame_offsets,
        }
    }

    /// Evaluate all constraints on the LDE domain (sequential version)
    pub fn evaluate(&self, lde_trace: &LDETraceTable<F, E>, domain: &Domain<F>) -> EvalResult<E> {
        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;
        let max_degree = self.constraints.max_degree();

        let quotient_evals: Vec<_> = (0..lde_size)
            .map(|point_idx| {
                evaluate_at_point(
                    self.constraints,
                    lde_trace,
                    &self.precomputes,
                    &self.frame_offsets,
                    point_idx,
                    max_degree,
                )
            })
            .collect();

        EvalResult {
            quotient_evals,
            failures: Vec::new(),
        }
    }
}

#[cfg(feature = "parallel")]
impl<'a, F, E> SimpleConstraintEvaluator<'a, F, E>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync,
{
    /// Evaluate all constraints on the LDE domain (parallel version)
    pub fn evaluate_parallel(
        &self,
        lde_trace: &LDETraceTable<F, E>,
        domain: &Domain<F>,
    ) -> EvalResult<E> {
        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;
        let max_degree = self.constraints.max_degree();

        let quotient_evals: Vec<_> = (0..lde_size)
            .into_par_iter()
            .map(|point_idx| {
                evaluate_at_point(
                    self.constraints,
                    lde_trace,
                    &self.precomputes,
                    &self.frame_offsets,
                    point_idx,
                    max_degree,
                )
            })
            .collect();

        EvalResult {
            quotient_evals,
            failures: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::simple::{BoundaryConstraint, TransitionConstraint};
    use math::field::fields::fft_friendly::u64_goldilocks::U64GoldilocksPrimeField;

    type F = U64GoldilocksPrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn test_evaluator_creation() {
        // This is a basic smoke test - full integration tests require trace setup
        let constraints: Constraints<F, F> = Constraints {
            degree_1: vec![TransitionConstraint {
                name: "test_constraint",
                evaluate: |_frame| FE::zero(),
                evaluate_ext: |_frame| FE::zero(),
                end_exemptions: 1,
            }],
            degree_2: vec![],
            degree_3: vec![],
            boundary: vec![BoundaryConstraint::new_main("init", 0, 0, FE::one())],
            use_legacy_ordering: false,
            use_legacy_evaluation: false,
        };

        assert_eq!(constraints.num_constraints(), 2);
        assert_eq!(constraints.max_degree(), 2); // boundary constraints count as degree 2
        assert_eq!(constraints.exemption_counts(), vec![1]);
        assert_eq!(constraints.boundary_rows(), vec![0]);
    }
}

// Simple constraint evaluator using the new simplified constraint system
//
// This evaluator works with the Constraints struct from simple.rs
// and provides efficient evaluation using cyclic precomputed values.

use super::simple::{Constraints, EvalPrecomputes, EvalResult};
use crate::domain::Domain;
use crate::frame::Frame;
use crate::trace::LDETraceTable;
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Evaluator for simplified constraints
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
    pub fn evaluate(
        &self,
        lde_trace: &LDETraceTable<F, E>,
        domain: &Domain<F>,
    ) -> EvalResult<E> {
        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;
        let max_degree = self.constraints.max_degree();

        let mut quotient_evals = Vec::with_capacity(lde_size);
        let failures = Vec::new();

        for point_idx in 0..lde_size {
            let eval = self.evaluate_at_point(lde_trace, point_idx, max_degree);
            quotient_evals.push(eval);
        }

        EvalResult {
            quotient_evals,
            failures,
        }
    }

    /// Evaluate constraints at a single point
    fn evaluate_at_point(
        &self,
        lde_trace: &LDETraceTable<F, E>,
        point_idx: usize,
        max_degree: usize,
    ) -> FieldElement<E> {
        let frame = Frame::read_from_lde(lde_trace, point_idx, &self.frame_offsets);

        let inv_z = self.precomputes.vanishing_inv.get(point_idx);
        let x_n = self.precomputes.x_pow_n.get(point_idx);
        let x_2n = self.precomputes.x_pow_2n.get(point_idx);

        let mut acc = FieldElement::<E>::zero();
        let mut beta_idx = 0;

        // Degree 1 constraints
        for c in &self.constraints.degree_1 {
            let c_eval = (c.evaluate)(&frame);
            let exempt = &self.precomputes.exemptions[&c.end_exemptions][point_idx];
            let corrected = match max_degree {
                3 => exempt * x_2n * &c_eval,
                2 => exempt * x_n * &c_eval,
                _ => exempt * &c_eval,
            };
            acc = acc + &self.precomputes.beta_powers[beta_idx] * &corrected;
            beta_idx += 1;
        }

        // Degree 2 constraints
        for c in &self.constraints.degree_2 {
            let c_eval = (c.evaluate)(&frame);
            let exempt = &self.precomputes.exemptions[&c.end_exemptions][point_idx];
            let corrected = match max_degree {
                3 => exempt * x_n * &c_eval,
                _ => exempt * &c_eval,
            };
            acc = acc + &self.precomputes.beta_powers[beta_idx] * &corrected;
            beta_idx += 1;
        }

        // Degree 3 constraints (no degree correction needed)
        for c in &self.constraints.degree_3 {
            let c_eval = (c.evaluate)(&frame);
            let exempt = &self.precomputes.exemptions[&c.end_exemptions][point_idx];
            acc = acc + &self.precomputes.beta_powers[beta_idx] * (exempt * &c_eval);
            beta_idx += 1;
        }

        // Boundary constraints (effectively degree 2)
        for bc in &self.constraints.boundary {
            let trace_val: FieldElement<E> = if bc.is_aux {
                lde_trace.get_aux(point_idx, bc.col).clone()
            } else {
                lde_trace.get_main(point_idx, bc.col).clone().to_extension()
            };
            let c_eval = trace_val - &bc.value;
            let lagrange = &self.precomputes.lagrange[&bc.row][point_idx];
            let corrected = match max_degree {
                3 => lagrange * x_n * &c_eval,
                _ => lagrange * &c_eval,
            };
            acc = acc + &self.precomputes.beta_powers[beta_idx] * &corrected;
            beta_idx += 1;
        }

        // Divide by vanishing polynomial
        inv_z * acc
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
            .map(|point_idx| self.evaluate_at_point(lde_trace, point_idx, max_degree))
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
                end_exemptions: 1,
            }],
            degree_2: vec![],
            degree_3: vec![],
            boundary: vec![BoundaryConstraint::new_main("init", 0, 0, FE::one())],
        };

        assert_eq!(constraints.num_constraints(), 2);
        assert_eq!(constraints.max_degree(), 1);
        assert_eq!(constraints.exemption_counts(), vec![1]);
        assert_eq!(constraints.boundary_rows(), vec![0]);
    }
}

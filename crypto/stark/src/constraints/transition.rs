use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ops::Div;

use crate::domain::Domain;
use crate::prover::evaluate_polynomial_on_lde_domain;
use crate::traits::TransitionEvaluationContext;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::polynomial::Polynomial;

/// TransitionConstraintEvaluator represents the behaviour that a transition constraint
/// over the computation that wants to be proven must comply with.
pub trait TransitionConstraintEvaluator<F, E>: Send + Sync
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    /// The degree of the constraint interpreting it as a multivariate polynomial.
    fn degree(&self) -> usize;

    /// The index of the constraint.
    /// Each transition constraint should have one index in the range [0, N),
    /// where N is the total number of transition constraints.
    fn constraint_idx(&self) -> usize;

    /// The function representing the evaluation of the constraint over elements
    /// of the trace table.
    ///
    /// Elements of the trace table are found in the `frame` input, and depending on the
    /// constraint, elements of `periodic_values` and `rap_challenges` may be used in
    /// the evaluation.
    /// Once computed, the evaluation should be inserted in the `transition_evaluations`
    /// vector, in the index corresponding to the constraint as given by `constraint_idx()`.
    fn evaluate_verifier(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    );

    /// The periodicity the constraint is applied over the trace.
    ///
    /// Default value is 1, meaning that the constraint is applied to every
    /// step of the trace.
    fn period(&self) -> usize {
        1
    }

    /// The offset with respect to the first trace row, where the constraint
    /// is applied.
    /// For example, if the constraint has periodicity 2 and offset 1, this means
    /// the constraint will be applied over trace rows of index 1, 3, 5, etc.
    ///
    /// Default value is 0, meaning that the constraint is applied from the first
    /// element of the trace on.
    fn offset(&self) -> usize {
        0
    }

    /// For a more fine-grained description of where the constraint should apply,
    /// an exemptions period can be defined.
    /// This specifies the periodicity of the row indexes where the constraint should
    /// NOT apply, within the row indexes where the constraint applies, as specified by
    /// `period()` and `offset()`.
    ///
    /// Default value is None.
    fn exemptions_period(&self) -> Option<usize> {
        None
    }

    /// The offset value for periodic exemptions. Check documentation of `period()`,
    /// `offset()` and `exemptions_period` for a better understanding.
    fn periodic_exemptions_offset(&self) -> Option<usize> {
        None
    }

    /// The number of exemptions at the end of the trace.
    ///
    /// This method's output defines what trace elements should not be considered for
    /// the constraint evaluation at the end of the trace. For example, for a fibonacci
    /// computation that has to use the result 2 following steps, this method is defined
    /// to return the value 2.
    ///
    /// Default value is 0, meaning the constraint applies to all rows including the last.
    fn end_exemptions(&self) -> usize {
        0
    }

    /// Prover-optimized evaluation that writes base-field constraints to `base_evals`
    /// and extension-field constraints to `ext_evals`.
    ///
    /// Constraints with `constraint_idx() < base_evals.len()` are "base" constraints
    /// and MUST override this to write `FieldElement<F>` into `base_evals[constraint_idx()]`.
    /// Extension constraints (LogUp etc.) use the default, which asserts the index is
    /// in the extension range and delegates to `evaluate()`.
    fn evaluate_prover(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        base_evals: &mut [FieldElement<F>],
        ext_evals: &mut [FieldElement<E>],
    ) {
        debug_assert!(
            self.constraint_idx() >= base_evals.len(),
            "Base constraint idx {} must override evaluate_prover()",
            self.constraint_idx(),
        );
        self.evaluate_verifier(evaluation_context, ext_evals);
    }

    /// Method for calculating the end exemptions polynomial.
    ///
    /// This polynomial is used to compute zerofiers of the constraint, and the default
    /// implementation should normally not be changed.
    fn end_exemptions_poly(
        &self,
        trace_primitive_root: &FieldElement<F>,
        trace_length: usize,
    ) -> Polynomial<FieldElement<F>> {
        let one_poly = Polynomial::new_monomial(FieldElement::<F>::one(), 0);
        if self.end_exemptions() == 0 {
            return one_poly;
        }
        let period = self.period();
        let decrement = trace_primitive_root.pow(trace_length - period);
        let mut current = decrement.clone();
        // FIXME: CHECK IF WE NEED TO CHANGE THE NEW MONOMIAL'S ARGUMENTS TO trace_root^(offset * trace_length / period) INSTEAD OF ONE!!!!
        (0..self.end_exemptions()).fold(one_poly, |acc, _| {
            let next =
                acc * (Polynomial::new_monomial(FieldElement::<F>::one(), 1) - current.clone());
            current = &current * &decrement;
            next
        })
    }

    /// Compute evaluations of the constraints zerofier over a LDE domain.
    #[allow(unstable_name_collisions)]
    fn zerofier_evaluations_on_extended_domain(&self, domain: &Domain<F>) -> Vec<FieldElement<F>> {
        let blowup_factor = domain.blowup_factor;
        let trace_length = domain.trace_roots_of_unity.len();
        let trace_primitive_root = &domain.trace_primitive_root;
        let coset_offset = &domain.coset_offset;
        let lde_root_order = u64::from((blowup_factor * trace_length).trailing_zeros());
        let lde_root = F::get_primitive_root_of_unity(lde_root_order).unwrap();

        let end_exemptions_poly = self.end_exemptions_poly(trace_primitive_root, trace_length);

        // If there is an exemptions period defined for this constraint, the evaluations are calculated directly
        // by computing P_exemptions(x) / Zerofier(x)
        if let Some(exemptions_period) = self.exemptions_period() {
            // FIXME: Rather than making this assertions here, it would be better to handle these
            // errors or make these checks when the AIR is initialized.

            debug_assert!(exemptions_period.is_multiple_of(self.period()));

            debug_assert!(self.periodic_exemptions_offset().is_some());

            // The elements of the domain have order `trace_length * blowup_factor`, so the zerofier evaluations
            // without the end exemptions, repeat their values after `blowup_factor * exemptions_period` iterations,
            // so we only need to compute those.
            let last_exponent = blowup_factor * exemptions_period;
            let numerator_power = trace_length / exemptions_period;
            let denominator_power = trace_length / self.period();
            let offset_exponent =
                trace_length * self.periodic_exemptions_offset().unwrap() / exemptions_period;
            let numerator_offset = trace_primitive_root.pow(offset_exponent);
            let denominator_offset = trace_primitive_root.pow(self.offset() * denominator_power);
            let numerator_step = lde_root.pow(numerator_power);
            let denominator_step = lde_root.pow(denominator_power);
            let mut numerator_eval = coset_offset.pow(numerator_power);
            let mut denominator_eval = coset_offset.pow(denominator_power);

            let mut numerators = Vec::with_capacity(last_exponent);
            let mut denominators = Vec::with_capacity(last_exponent);
            for _ in 0..last_exponent {
                numerators.push(&numerator_eval - &numerator_offset);
                denominators.push(&denominator_eval - &denominator_offset);
                numerator_eval = &numerator_eval * &numerator_step;
                denominator_eval = &denominator_eval * &denominator_step;
            }

            // Batch inversion: O(3N) muls + 1 inversion instead of N individual inversions
            // (each ~72 muls for Goldilocks Fermat chain). Denominators are guaranteed non-zero
            // because the sets of powers of `offset_times_x` and `trace_primitive_root` are
            // disjoint, provided that the offset is neither an element of the interpolation
            // domain nor part of a subgroup with order less than n.
            FieldElement::inplace_batch_inverse(&mut denominators).unwrap();

            let evaluations: Vec<_> = numerators
                .iter()
                .zip(denominators.iter())
                .map(|(num, denom_inv)| num * denom_inv)
                .collect();

            // FIXME: Instead of computing this evaluations for each constraint, they can be computed
            // once for every constraint with the same end exemptions (combination of end_exemptions()
            // and period).
            let end_exemption_evaluations = evaluate_polynomial_on_lde_domain(
                &end_exemptions_poly,
                blowup_factor,
                domain.interpolation_domain_size,
                coset_offset,
            )
            .unwrap();

            let cycled_evaluations = evaluations
                .iter()
                .cycle()
                .take(end_exemption_evaluations.len());

            core::iter::zip(cycled_evaluations, end_exemption_evaluations)
                .map(|(eval, exemption_eval)| eval * exemption_eval)
                .collect()

        // In this else branch, the zerofiers are computed as the numerator, then inverted
        // using batch inverse and then multiplied by P_exemptions(x). This way we don't do
        // useless divisions.
        } else {
            let last_exponent = blowup_factor * self.period();
            let denominator_power = trace_length / self.period();
            let denominator_offset = trace_primitive_root.pow(self.offset() * denominator_power);
            let denominator_step = lde_root.pow(denominator_power);
            let mut denominator_eval = coset_offset.pow(denominator_power);

            let mut evaluations = Vec::with_capacity(last_exponent);
            for _ in 0..last_exponent {
                evaluations.push(&denominator_eval - &denominator_offset);
                denominator_eval = &denominator_eval * &denominator_step;
            }

            FieldElement::inplace_batch_inverse(&mut evaluations).unwrap();

            // Fast path: when end_exemptions == 0, the end_exemptions_poly is constant 1,
            // so the multiplication is identity. Skip the expensive FFT evaluation.
            if self.end_exemptions() == 0 {
                return evaluations;
            }

            let end_exemption_evaluations = evaluate_polynomial_on_lde_domain(
                &end_exemptions_poly,
                blowup_factor,
                domain.interpolation_domain_size,
                coset_offset,
            )
            .unwrap();

            let cycled_evaluations = evaluations
                .iter()
                .cycle()
                .take(end_exemption_evaluations.len());

            core::iter::zip(cycled_evaluations, end_exemption_evaluations)
                .map(|(eval, exemption_eval)| eval * exemption_eval)
                .collect()
        }
    }

    /// Returns the evaluation of the zerofier corresponding to this constraint in some point
    /// `z`, which could be in a field extension.
    #[allow(unstable_name_collisions)]
    fn evaluate_zerofier(
        &self,
        z: &FieldElement<E>,
        trace_primitive_root: &FieldElement<F>,
        trace_length: usize,
    ) -> FieldElement<E> {
        let end_exemptions_poly = self.end_exemptions_poly(trace_primitive_root, trace_length);

        if let Some(exemptions_period) = self.exemptions_period() {
            debug_assert!(exemptions_period.is_multiple_of(self.period()));

            debug_assert!(self.periodic_exemptions_offset().is_some());

            let periodic_exemptions_offset = self.periodic_exemptions_offset().unwrap();
            let offset_exponent = trace_length * periodic_exemptions_offset / exemptions_period;

            let numerator = -trace_primitive_root.pow(offset_exponent)
                + z.pow(trace_length / exemptions_period);
            let denominator = -trace_primitive_root
                .pow(self.offset() * trace_length / self.period())
                + z.pow(trace_length / self.period());
            // The denominator isn't zero because z is sampled outside the set of primitive roots.
            return unsafe { numerator.div(denominator).unwrap_unchecked() }
                * end_exemptions_poly.evaluate(z);
        }

        (-trace_primitive_root.pow(self.offset() * trace_length / self.period())
            + z.pow(trace_length / self.period()))
        .inv()
        .unwrap()
            * end_exemptions_poly.evaluate(z)
    }
}

// =============================================================================
// User-facing TransitionConstraint trait + adapter
// =============================================================================

use crate::table::TableView;

/// User-facing trait for defining transition constraints.
///
/// Implement `evaluate()` to define the polynomial identity; the verifier and
/// prover evaluation paths are auto-generated via `.boxed()`.
///
/// The `evaluate` method is generic over its field types so the same polynomial
/// works for both the prover (`TableView<F, E>`) and verifier (`TableView<E, E>`).
pub trait TransitionConstraint<F, E>: Send + Sync
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    /// The degree of the constraint as a multivariate polynomial.
    fn degree(&self) -> usize;

    /// Unique index in `[0, N)` where N is the total number of transition constraints.
    fn constraint_idx(&self) -> usize;

    /// Number of exempted rows at the end of the trace.
    fn end_exemptions(&self) -> usize {
        0
    }

    /// Evaluate the constraint polynomial on a trace step.
    ///
    /// Generic over the field so the same polynomial works for both
    /// prover (FF=F, returns FieldElement<F>) and verifier (FF=E, returns FieldElement<E>).
    fn evaluate<FF, EE>(&self, step: &TableView<FF, EE>) -> FieldElement<FF>
    where
        FF: IsSubFieldOf<EE>,
        EE: IsField;

    /// Periodicity (default 1 = every row).
    fn period(&self) -> usize {
        1
    }

    /// Offset for periodic application (default 0).
    fn offset(&self) -> usize {
        0
    }

    /// Exemptions period (default None).
    fn exemptions_period(&self) -> Option<usize> {
        None
    }

    /// Offset for periodic exemptions (default None).
    fn periodic_exemptions_offset(&self) -> Option<usize> {
        None
    }

    /// Wrap into a boxed `TransitionConstraintEvaluator` for the evaluator.
    ///
    /// The adapter auto-generates `evaluate_verifier()` and `evaluate_prover()`
    /// from the generic `evaluate()`.
    fn boxed(self) -> Box<dyn TransitionConstraintEvaluator<F, E>>
    where
        Self: Sized + 'static,
    {
        Box::new(TransitionConstraintAdapter(self))
    }
}

/// Adapter: implements `TransitionConstraintEvaluator` for any `TransitionConstraint`.
///
/// Auto-generates `evaluate_verifier()` (E×E path) and `evaluate_prover()` (F path)
/// from the user's generic `evaluate()`.
pub struct TransitionConstraintAdapter<T>(pub T);

impl<T, F, E> TransitionConstraintEvaluator<F, E> for TransitionConstraintAdapter<T>
where
    T: TransitionConstraint<F, E> + 'static,
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        self.0.degree()
    }
    fn constraint_idx(&self) -> usize {
        self.0.constraint_idx()
    }
    fn end_exemptions(&self) -> usize {
        self.0.end_exemptions()
    }
    fn period(&self) -> usize {
        self.0.period()
    }
    fn offset(&self) -> usize {
        self.0.offset()
    }
    fn exemptions_period(&self) -> Option<usize> {
        self.0.exemptions_period()
    }
    fn periodic_exemptions_offset(&self) -> Option<usize> {
        self.0.periodic_exemptions_offset()
    }

    fn evaluate_verifier(
        &self,
        ctx: &TransitionEvaluationContext<F, E>,
        evals: &mut [FieldElement<E>],
    ) {
        let idx = self.0.constraint_idx();
        match ctx {
            TransitionEvaluationContext::Prover { frame, .. } => {
                evals[idx] = self.0.evaluate(frame.get_evaluation_step(0)).to_extension();
            }
            TransitionEvaluationContext::Verifier { frame, .. } => {
                evals[idx] = self.0.evaluate(frame.get_evaluation_step(0));
            }
        }
    }

    fn evaluate_prover(
        &self,
        ctx: &TransitionEvaluationContext<F, E>,
        base_evals: &mut [FieldElement<F>],
        ext_evals: &mut [FieldElement<E>],
    ) {
        let idx = self.0.constraint_idx();
        if idx < base_evals.len() {
            // Base-field fast path: write FieldElement<F> directly
            if let TransitionEvaluationContext::Prover { frame, .. } = ctx {
                base_evals[idx] = self.0.evaluate(frame.get_evaluation_step(0));
            } else {
                unreachable!("evaluate_prover called with non-Prover context");
            }
        } else {
            // Fallback: AIR did not opt into base-field splitting,
            // delegate to the verifier path which writes E evals.
            self.evaluate_verifier(ctx, ext_evals);
        }
    }
}

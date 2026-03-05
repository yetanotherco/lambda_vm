//! 64-bit VM prover constraint templates.
//!
//! This module provides constraint templates for the 64-bit VM prover
//! using the Goldilocks field.

pub mod cpu;
pub mod templates;

/// Generate an `evaluate_prover` override that writes `self.compute(step)`
/// directly into `base_evaluations` as `FieldElement<F>`, enabling F×E
/// accumulation in the composition polynomial.
///
/// All VM base-field constraints share this identical body; the macro
/// eliminates duplication and ensures the bounds check is always present.
#[macro_export]
macro_rules! impl_base_field_evaluate_prover {
    () => {
        fn evaluate_prover(
            &self,
            ctx: &stark::traits::TransitionEvaluationContext<
                $crate::tables::types::GoldilocksField,
                $crate::tables::types::GoldilocksExtension,
            >,
            base_evaluations: &mut [math::field::element::FieldElement<
                $crate::tables::types::GoldilocksField,
            >],
            _ext_evaluations: &mut [math::field::element::FieldElement<
                $crate::tables::types::GoldilocksExtension,
            >],
        ) {
            assert!(
                self.constraint_idx < base_evaluations.len(),
                "constraint_idx {} out of bounds for base_evaluations (len {})",
                self.constraint_idx,
                base_evaluations.len(),
            );
            if let stark::traits::TransitionEvaluationContext::Prover { frame, .. } = ctx {
                base_evaluations[self.constraint_idx] = self.compute(frame.get_evaluation_step(0));
            } else {
                unreachable!("evaluate_prover called with non-Prover context");
            }
        }
    };
}

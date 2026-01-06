use std::marker::PhantomData;

use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

use crate::{
    constraints::transition::TransitionConstraint, table::TableView,
    traits::TransitionEvaluationContext,
};

// struct LookupConstraint<F: IsFFTField + IsSubFieldOf<E> + Send + Sync, E: IsField + Send + Sync, PI>
// {
//     column_a: usize,
//     column_b: usize,
//     phantom: PhantomData<(F, E, PI)>,
// }

// impl<F: IsFFTField + IsSubFieldOf<E> + Send + Sync, E: IsField + Send + Sync, PI>
//     LookupConstraint<F, E, PI>
// {
//     expand_trace -> generar columna auxiliar en ambas tablas (simil build_auxiliary_constraint)
//     add_constraints -> genera las transition constraints
//     get_boundary_constraint (public value) -> boundary constraint que genera los public values
//     check_public_values -> verifier checkea ambos public values
//     fn build_auxiliary_trace();
// }

// trait AirWithLookup {
//     add_transition_constraints()
//     add_boundary()
// }

// impl AIR for AirWithLookup {
//     build_auxiliary_trace(),
//     add_transition_constraints()

// }

// struct MyAIR {
//     transition_constrainsts
// }

// al multi-verify pasarle que indices de public values y de que tablas checkear
//
// Potential solutions
//
// * Trait AirWithLookup that implements AIR -> yields conflicting impl with other AIR implementations
// * Struct LookupAIRWrapper<A: AIR> that implements AIR and adds the lookup constraints to already existing air -> we cannot add transition constraints to the underlying air
// nor can we add them on `transition_constraints()` method as it returns a reference
// * Don't fully automatize it but add helper functions to aid the process: aka functions to define transition & boundary constraints common to all lookup airs
//
//
//
//
//

// Impl (failing) of wrapper solution
// use std::marker::PhantomData;

// use math::field::{
//     element::FieldElement,
//     traits::{IsFFTField, IsField, IsSubFieldOf},
// };

// use crate::{
//     constraints::transition::TransitionConstraint,
//     context::AirContext,
//     traits::{AIR, TransitionEvaluationContext},
// };

// /// Wrapper for AIRS using lookups
// /// PR
// pub struct LookupAIRWrapper<A>
// where
//     A: AIR,
// {
//     air: A,
// }

// impl<A> AIR for LookupAIRWrapper<A>
// where
//     A: AIR,
// {
//     type Field = A::Field;

//     type FieldExtension = A::FieldExtension;

//     type PublicInputs = A::PublicInputs;

//     fn step_size(&self) -> usize {
//         self.air.step_size()
//     }

//     fn new(
//         trace_length: usize,
//         pub_inputs: &Self::PublicInputs,
//         proof_options: &crate::proof::options::ProofOptions,
//     ) -> Self
//     where
//         Self: Sized,
//     {
//         Self {
//             air: A::new(trace_length, pub_inputs, proof_options),
//         }
//     }

//     fn trace_layout(&self) -> (usize, usize) {
//         self.air.trace_layout() //TODO: check if we need to modify
//     }

//     fn composition_poly_degree_bound(&self) -> usize {
//         self.air.composition_poly_degree_bound() //TODO: check if we need to modify
//     }

//     fn boundary_constraints(
//         &self,
//         rap_challenges: &[FieldElement<Self::FieldExtension>],
//     ) -> super::boundary::BoundaryConstraints<Self::FieldExtension> {
//         let mut constraints = self.air.boundary_constraints(rap_challenges);
//         // TODO: add boundary constraints here
//         constraints
//     }

//     fn context(&self) -> &AirContext {
//         self.air.context()
//     }

//     fn trace_length(&self) -> usize {
//         self.air.trace_length()
//     }

//     fn pub_inputs(&self) -> &Self::PublicInputs {
//         self.air.pub_inputs()
//     }

//     fn transition_constraints(
//         &self,
//     ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> {
//         let mut tc: Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> = vec![];
//         // TODO: add constraints
//         tc.push(Box::new(PermutationConstraint::new(PermutationColumns {
//             a: 0,
//             v: 0,
//             a_s: 0,
//             v_s: 0,
//             m: 0,
//         })));
//         tc.extend(self.air.transition_constraints().into_iter().cloned());
//         &tc
//     }
// }

// Transition constraint that ensures that the sorted columns are a permutation of the original ones.
/// We are using the LogUp construction described in:
/// <https://0xpolygonmiden.github.io/miden-vm/design/lookups/logup.html>.
/// See also our post of LogUp argument in blog.lambdaclass.com.
#[derive(Clone)]
pub struct PermutationConstraint<
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
> {
    phantom: PhantomData<(F, E)>,
    columns: PermutationColumns, // TODO: If we use fewer columns this could also be const generic
}
impl<F, E> PermutationConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new(columns: PermutationColumns) -> Self {
        Self {
            columns,
            phantom: PhantomData::<(F, E)>,
        }
    }
}

#[derive(Clone)]
pub struct PermutationColumns {
    a: usize,
    v: usize,
    a_s: usize,
    v_s: usize,
    m: usize,
}

impl<F, E> TransitionConstraint<F, E> for PermutationConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        3
    }

    fn constraint_idx(&self) -> usize {
        2
    }

    fn end_exemptions(&self) -> usize {
        1
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        // In both evaluation contexts, Prover and Verfier will evaluate the transition polynomial in the same way.
        // The only difference is that the Prover's Frame has base field and field extension elements,
        // while the Verfier's Frame has only field extension elements.
        let res = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                rap_challenges,
                ..
            } => compute_permutation_constraint(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                rap_challenges,
                self.columns.clone(),
            ),
            TransitionEvaluationContext::Verifier {
                frame,
                rap_challenges,
                ..
            } => compute_permutation_constraint(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                rap_challenges,
                self.columns.clone(),
            ),
        };

        // The eval always exists, except if the constraint idx were incorrectly defined.
        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
            *eval = res;
        }
    }
}

fn compute_permutation_constraint<F, E>(
    first_step: &TableView<'_, F, E>,
    second_step: &TableView<'_, F, E>,
    rap_challenges: &[FieldElement<E>],
    columns: PermutationColumns,
) -> FieldElement<E>
where
    F: IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    // Auxiliary frame elements
    let s0 = first_step.get_aux_evaluation_element(0, 0);
    let s1 = second_step.get_aux_evaluation_element(0, 0);

    // Challenges
    let z = &rap_challenges[0];
    let alpha = &rap_challenges[1];

    // Main frame elements
    let a1 = second_step.get_main_evaluation_element(0, columns.a);
    let v1 = second_step.get_main_evaluation_element(0, columns.v);
    let a_sorted_1 = second_step.get_main_evaluation_element(0, columns.a_s);
    let v_sorted_1 = second_step.get_main_evaluation_element(0, columns.v_s);
    let m = second_step.get_main_evaluation_element(0, columns.m);

    let unsorted_term = -(a1 + v1 * alpha) + z;
    let sorted_term = -(a_sorted_1 + v_sorted_1 * alpha) + z;

    // We are using the following LogUp equation:
    // s1 = s0 + m / sorted_term - 1/unsorted_term.
    // Since constraints must be expressed without division, we multiply each term by sorted_term * unsorted_term:
    s0 * &unsorted_term * &sorted_term + m * &unsorted_term
        - &sorted_term
        - s1 * unsorted_term * sorted_term
}

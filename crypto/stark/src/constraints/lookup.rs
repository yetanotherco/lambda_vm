use std::marker::PhantomData;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

use crate::{
    constraints::{boundary::BoundaryConstraints, transition::TransitionConstraint},
    context::AirContext,
    table::TableView,
    trace::TraceTable,
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
// Por que AIR es un trait y no un struct? Todos los implementors tienen los mismos fields (context, trace_lengh, inputs, constraints)
// Capaz podriamos separar los fields del air de su behaviour quedando algo mas sencillo:
//
//
pub struct Air<
    L: AirLogic<F, E>,
    PI,
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
> {
    pub context: AirContext,
    pub trace_length: usize,
    pub pub_inputs: PI,
    pub step_size: usize,
    pub transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    pub logic: PhantomData<L>,
}

pub trait AirLogic<F: IsFFTField + IsSubFieldOf<E> + Send + Sync, E: IsField + Send + Sync> {
    fn build_auxiliary_trace(_trace: &mut TraceTable<F, E>, _challenges: &[FieldElement<E>]) {}

    fn build_rap_challenges(_transcript: &mut dyn IsStarkTranscript<E, F>) -> Vec<FieldElement<E>> {
        vec![]
    }
    fn boundary_constraints(_rap_challenges: &[FieldElement<E>]) -> BoundaryConstraints<E> {
        BoundaryConstraints::from_constraints(vec![])
    }
}
//
// Y en lugar de tener diferentes structs que implementan AIR tendriamos funciones que crean AIRS con distintas constraints y logicas
// Los metodos default de AIR que no usamos pasarian a ser metodos de struct Air
// Ya no es necesario tener un new comun a todos los AIR porque nuestros prove y verify reciben un air
// Esto nos da mas flexibilidad y nos permite hacer el wrapper de lookups ya que podemos modificar y agregarle constraints al air
//

impl<L, PI, F, E> crate::traits::AIR for Air<L, PI, F, E>
where
    L: AirLogic<F, E> + Send + Sync,
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    PI: Send + Sync,
{
    type Field = F;

    type FieldExtension = E;

    type PublicInputs = PI;

    fn step_size(&self) -> usize {
        self.step_size
    }

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &crate::proof::options::ProofOptions,
    ) -> Self
    where
        Self: Sized,
    {
        unreachable!("THIS SHOULD NO LONGER BE USED")
    }

    fn trace_layout(&self) -> (usize, usize) {
        todo!() // Add to struct or infer
    }

    fn composition_poly_degree_bound(&self) -> usize {
        todo!() // Add to struct or infer
    }

    fn boundary_constraints(
        &self,
        rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> BoundaryConstraints<Self::FieldExtension> {
        L::boundary_constraints(rap_challenges)
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> {
        &self.transition_constraints
    }

    fn build_auxiliary_trace(
        &self,
        main_trace: &mut TraceTable<Self::Field, Self::FieldExtension>,
        rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) {
        L::build_auxiliary_trace(main_trace, rap_challenges);
    }

    fn build_rap_challenges(
        &self,
        transcript: &mut dyn IsStarkTranscript<Self::FieldExtension, Self::Field>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        L::build_rap_challenges(transcript)
    }
}

impl<L, PI, F, E> Air<L, PI, F, E>
where
    L: AirLogic<F, E> + Send + Sync,
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    pub fn into_lookup(
        mut self,
        columns: PermutationColumns,
    ) -> Air<LookUpAirLogicWrapper<L, F, E>, PI, F, E> {
        self.transition_constraints
            .push(Box::new(PermutationConstraint::<F, E>::new(columns)));
        Air {
            context: self.context,
            trace_length: self.trace_length,
            pub_inputs: self.pub_inputs,
            step_size: self.step_size,
            transition_constraints: self.transition_constraints,
            logic: PhantomData::<LookUpAirLogicWrapper<L, F, E>>,
        }
    }
}

pub struct LookUpAirLogicWrapper<L, F, E>
where
    L: AirLogic<F, E> + Send + Sync,
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    phanton: PhantomData<(L, F, E)>,
}

impl<L, F, E> AirLogic<F, E> for LookUpAirLogicWrapper<L, F, E>
where
    L: AirLogic<F, E> + Send + Sync,
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    fn build_auxiliary_trace(trace: &mut TraceTable<F, E>, challenges: &[FieldElement<E>]) {
        L::build_auxiliary_trace(trace, challenges);
        // TODO: Add common lookup auxiliary trace logic
        // I NEED TO KNOW:
        // for each aux column:
        // - What will its index be (maybe)
        // - Which columns make up the flags
        // - Which columns make up the values
    }

    fn build_rap_challenges(transcript: &mut dyn IsStarkTranscript<E, F>) -> Vec<FieldElement<E>> {
        L::build_rap_challenges(transcript)
        // Do we need more rap challneges for the added boundary constraints??\
        // We will only use rap challenges for building auxiliary trace, not for anything else
        // We must use the same rap challenges for all tables
        // This method shall be removed and rap challenges shall be sampled only once for all airs in prove methdo AFTER COMITTING!!!
    }
    fn boundary_constraints(rap_challenges: &[FieldElement<E>]) -> BoundaryConstraints<E> {
        let mut boundary_constraints = L::boundary_constraints(rap_challenges);
        // TODO: Add boundary constraints
        // Are these constraints dependant on the columns we use? aka do they differ between each lookup table?
        boundary_constraints
    }
}

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
    columns: PermutationColumns,
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

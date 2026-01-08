use std::marker::PhantomData;

use crate::{
    constraints::{
        lookup::{
            AirWithLookup, AuxiliaryTraceBuildData, LookUpPublicInputs,
            NullBoundaryConstraintBuilder, TableInteraction,
        },
        transition::TransitionConstraint,
    },
    context::AirContext,
    proof::options::ProofOptions,
};
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};
type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;

pub fn new_mul_air_with_lookup(
    trace_length: usize,
    pub_inputs: LookUpPublicInputs<F>,
    proof_options: &ProofOptions,
) -> AirWithLookup<F, E, NullBoundaryConstraintBuilder> {
    // TODO: define cpu-specific constraints here
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let context = AirContext {
        proof_options: proof_options.clone(),
        trace_columns: 5,
        transition_offsets: vec![0, 1],
        num_transition_constraints: transition_constraints.len(),
    };

    AirWithLookup {
        context,
        trace_length,
        pub_inputs,
        step_size: 1,
        trace_layout: (5, 3),
        transition_constraints,
        auxiliary_trace_build_data: AuxiliaryTraceBuildData {
            interactions: vec![
                // Interaction with ADD table
                TableInteraction {
                    // ADD flag column
                    flag_columns: vec![0],
                    // values a , b, c
                    value_columns: vec![2, 3, 4],
                },
                // Interaction with MUL table
                TableInteraction {
                    // MUL flag column
                    flag_columns: vec![1],
                    // values a , b, c
                    value_columns: vec![2, 3, 4],
                },
            ],
        },
        boundary_constraint_builder: PhantomData::<NullBoundaryConstraintBuilder>,
    }
}

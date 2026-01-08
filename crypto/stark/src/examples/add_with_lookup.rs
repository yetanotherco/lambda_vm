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

pub fn new_add_air_with_lookup(
    trace_length: usize,
    pub_inputs: LookUpPublicInputs<F>,
    proof_options: &ProofOptions,
) -> AirWithLookup<F, E, NullBoundaryConstraintBuilder> {
    // TODO: define add-specific constraints here
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let context = AirContext {
        proof_options: proof_options.clone(),
        trace_columns: 6,
        transition_offsets: vec![0, 1],
        num_transition_constraints: transition_constraints.len(),
    };

    let step_size = 1;
    let trace_layout = (4, 2);

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with CPU table
            TableInteraction {
                // multiplicity column
                flag_columns: vec![3],
                // values a , b, c
                value_columns: vec![0, 1, 2],
            },
        ],
    };

    AirWithLookup::create(
        auxiliary_trace_build_data,
        pub_inputs,
        context,
        trace_length,
        step_size,
        trace_layout,
        transition_constraints,
    )
}

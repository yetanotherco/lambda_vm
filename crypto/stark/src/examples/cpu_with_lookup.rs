use crate::{
    constraints::{
        lookup::{
            AirWithLookup, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder, TableInteraction,
        },
        transition::TransitionConstraint,
    },
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
};
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};
type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;

pub fn new_cpu_air_with_lookup(
    trace: &TraceTable<F, E>,
    proof_options: &ProofOptions,
) -> AirWithLookup<F, E, NullBoundaryConstraintBuilder> {
    // TODO: define cpu-specific constraints here
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let context = AirContext {
        proof_options: proof_options.clone(),
        trace_columns: 8,
        transition_offsets: vec![0, 1],
        num_transition_constraints: transition_constraints.len(),
    };

    let step_size = 1;
    let trace_layout = (5, 3);
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
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
    };

    AirWithLookup::create(
        trace,
        auxiliary_trace_build_data,
        context,
        step_size,
        trace_layout,
        transition_constraints,
    )
}

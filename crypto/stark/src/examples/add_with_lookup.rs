use crate::{
    constraints::{
        lookup::{
            AirWithLookup, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder, TableInteraction,
        },
        transition::TransitionConstraint,
    },
    proof::options::ProofOptions,
    trace::TraceTable,
};
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};
type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;

pub fn new_add_air_with_lookup(
    trace: &TraceTable<F, E>,
    proof_options: &ProofOptions,
) -> AirWithLookup<F, E, NullBoundaryConstraintBuilder> {
    // TODO: define add-specific constraints here
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

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
        trace,
        auxiliary_trace_build_data,
        proof_options,
        step_size,
        trace_layout,
        transition_constraints,
    )
}

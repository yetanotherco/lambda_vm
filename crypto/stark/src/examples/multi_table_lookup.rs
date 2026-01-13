use crate::{
    constraints::transition::TransitionConstraint,
    lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder, TableInteraction,
    },
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
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let step_size = 1;
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with ADD table (CPU sends to ADD bus)
            TableInteraction {
                // ADD multiplicity column
                multiplicity_column: 0,
                // values a, b, c
                value_columns: vec![2, 3, 4],
                is_sender: true,
            },
            // Interaction with MUL table (CPU sends to MUL bus)
            TableInteraction {
                // MUL multiplicity column
                multiplicity_column: 1,
                // values a, b, c
                value_columns: vec![2, 3, 4],
                is_sender: true,
            },
        ],
    };

    AirWithBuses::create(
        trace,
        auxiliary_trace_build_data,
        proof_options,
        step_size,
        transition_constraints,
        (),
    )
}

pub fn new_mul_air_with_lookup(
    trace: &TraceTable<F, E>,
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let step_size = 1;

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with CPU table (MUL table receives from MUL bus)
            TableInteraction {
                // multiplicity column
                multiplicity_column: 3,
                // values a, b, c
                value_columns: vec![0, 1, 2],
                is_sender: false,
            },
        ],
    };

    AirWithBuses::create(
        trace,
        auxiliary_trace_build_data,
        proof_options,
        step_size,
        transition_constraints,
        (),
    )
}

pub fn new_add_air_with_lookup(
    trace: &TraceTable<F, E>,
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    // TODO: define add-specific constraints here
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let step_size = 1;

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with CPU table (ADD table receives from ADD bus)
            TableInteraction {
                // multiplicity column
                multiplicity_column: 3,
                // values a, b, c
                value_columns: vec![0, 1, 2],
                is_sender: false,
            },
        ],
    };

    AirWithBuses::create(
        trace,
        auxiliary_trace_build_data,
        proof_options,
        step_size,
        transition_constraints,
        (),
    )
}

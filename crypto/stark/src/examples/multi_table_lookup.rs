use crate::{
    constraints::transition::TransitionConstraint,
    lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder, TableInteraction,
    },
    proof::options::ProofOptions,
};
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};
type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;

pub fn new_cpu_air_with_lookup(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with ADD table (CPU sends to ADD bus)
            TableInteraction {
                multiplicity_column: Some(0),
                value_columns: vec![2, 3, 4],
                is_sender: true,
            },
            // Interaction with MUL table (CPU sends to MUL bus)
            TableInteraction {
                multiplicity_column: Some(1),
                value_columns: vec![2, 3, 4],
                is_sender: true,
            },
        ],
    };

    AirWithBuses::new(
        5, // CPU has 5 main columns: add_flag, mul_flag, a, b, c
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

pub fn new_mul_air_with_lookup(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with CPU table (MUL table receives from MUL bus)
            TableInteraction {
                multiplicity_column: Some(3),
                value_columns: vec![0, 1, 2],
                is_sender: false,
            },
        ],
    };

    AirWithBuses::new(
        4, // MUL has 4 main columns: a, b, c, multiplicity
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

pub fn new_add_air_with_lookup(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with CPU table (ADD table receives from ADD bus)
            TableInteraction {
                multiplicity_column: Some(3),
                value_columns: vec![0, 1, 2],
                is_sender: false,
            },
        ],
    };

    AirWithBuses::new(
        4, // ADD has 4 main columns: a, b, c, multiplicity
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

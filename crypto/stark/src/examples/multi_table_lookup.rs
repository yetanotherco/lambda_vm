use crate::{
    constraints::transition::TransitionConstraintEvaluator,
    lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
        NullBoundaryConstraintBuilder, Packing,
    },
    proof::options::ProofOptions,
};
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};
type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;

/// Bus IDs for the multi-table lookup example
#[repr(u64)]
pub enum BusId {
    Add,
    Mul,
}

impl From<BusId> for u64 {
    fn from(id: BusId) -> u64 {
        id as u64
    }
}

pub fn new_cpu_air_with_lookup(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with ADD table (CPU sends to ADD bus)
            BusInteraction::sender(
                BusId::Add,
                Multiplicity::Column(0),
                Packing::Direct.columns(&[2, 3, 4]),
            ),
            // Interaction with MUL table (CPU sends to MUL bus)
            BusInteraction::sender(
                BusId::Mul,
                Multiplicity::Column(1),
                Packing::Direct.columns(&[2, 3, 4]),
            ),
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
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with CPU table (MUL table receives from MUL bus)
            BusInteraction::receiver(
                BusId::Mul,
                Multiplicity::Column(3),
                Packing::Direct.columns(&[0, 1, 2]),
            ),
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
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            // Interaction with CPU table (ADD table receives from ADD bus)
            BusInteraction::receiver(
                BusId::Add,
                Multiplicity::Column(3),
                Packing::Direct.columns(&[0, 1, 2]),
            ),
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

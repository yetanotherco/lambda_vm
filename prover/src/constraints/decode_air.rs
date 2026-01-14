use math::field::{
    element::FieldElement,
    fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField, quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
    },
};
use stark::{
    constraints::{boundary::BoundaryConstraints, transition::TransitionConstraint},
    context::AirContext,
    proof::options::ProofOptions,
    traits::AIR,
};

use crate::tables::decode::DecodeTableRow;

pub struct DecodeTableAIR {
    context: AirContext,
    constraints:
        Vec<Box<dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>>>,
}

impl AIR for DecodeTableAIR {
    type Field = Babybear31PrimeField;
    type FieldExtension = Degree4BabyBearU32ExtensionField;
    type PublicInputs = ();

    fn new(proof_options: &ProofOptions) -> Self {
        let constraints = Vec::new();
        let num_transition_constraints = 0;

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: DecodeTableRow::NUM_COLUMNS,
            transition_offsets: vec![0],
            num_transition_constraints,
        };

        Self {
            context,
            constraints,
        }
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> {
        &self.constraints
    }

    fn boundary_constraints(
        &self,
        _pub_inputs: &Self::PublicInputs,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
        _bus_interactions: Option<&[stark::lookup::BusPublicInputs<Self::FieldExtension>]>,
        _trace_length: usize,
    ) -> BoundaryConstraints<Self::FieldExtension> {
        BoundaryConstraints::from_constraints(vec![])
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length * 2
    }

    fn trace_layout(&self) -> (usize, usize) {
        (DecodeTableRow::NUM_COLUMNS, 0)
    }

    fn step_size(&self) -> usize {
        1
    }
}

use crate::constraints::constraints_templates::{
    Arg2ValidityColumnIndexes, new_add_constraint, new_add_four_constraint,
    new_arg2_validity_constraint, new_bit_constraints, new_sub_constraint,
};
use crate::tables::cpu::CpuTableRow;

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
    trace::TraceTable,
    traits::AIR,
};

type FE = FieldElement<Babybear31PrimeField>;

type CPUTraceTable = TraceTable<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>;

/// AIR for the `CPU` trace.
///
/// Enforces two types of constraints:
/// - **Bit constraints**: selected columns must be binary.
/// - **32-bit add constraint** (limb-decomposed): enforces `lhs + rhs = res` when the ADD instruction is executed
pub struct CPUTableAIR {
    context: AirContext,
    constraints:
        Vec<Box<dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>>>,
}

impl AIR for CPUTableAIR {
    type Field = Babybear31PrimeField;
    type FieldExtension = Degree4BabyBearU32ExtensionField;
    type PublicInputs = ();

    fn new(_pub_inputs: &Self::PublicInputs, proof_options: &ProofOptions) -> Self {
        let constraint_index = 0;
        // Bit constraints:
        // Enforce that these columns are binary. They include:
        // - decode flags like `write_register`, `signed`, `mp_selector`, `muldiv_selector`
        // - the one-hot instruction flags (ADD, SUB, ..., EBREAK)
        let bit_columns_index_to_constraint = [
            CpuTableRow::WRITE_REGISTER,
            CpuTableRow::MEMORY_2BYTES,
            CpuTableRow::MEMORY_4BYTES,
            CpuTableRow::SIGNED,
            CpuTableRow::MP_SELECTOR,
            CpuTableRow::MULDIV_SELECTOR,
            CpuTableRow::ADD,
            CpuTableRow::SUB,
            CpuTableRow::SLT,
            CpuTableRow::AND,
            CpuTableRow::OR,
            CpuTableRow::XOR,
            CpuTableRow::SL,
            CpuTableRow::SR,
            CpuTableRow::JALR,
            CpuTableRow::BEQ,
            CpuTableRow::BLT,
            CpuTableRow::LOAD,
            CpuTableRow::STORE,
            CpuTableRow::MUL,
            CpuTableRow::DIVREM,
            CpuTableRow::ECALL,
            CpuTableRow::EBREAK,
        ];
        let (bit_constraints, constraint_index) =
            new_bit_constraints(&bit_columns_index_to_constraint, constraint_index);

        // Add constraint
        // Enforces that lhs (Word4L) + rhs (Word4L) = res (Word4L), with carry bits constrained.
        // It is enforced only on rows where the selected instruction flags are active.
        let (add_constraints, constraint_index) = new_add_constraint(
            vec![CpuTableRow::ADD, CpuTableRow::LOAD, CpuTableRow::STORE], // flags_idx,
            CpuTableRow::RV1_0,                                            // lhs_start_idx,
            CpuTableRow::ARG2_0,                                           // rhs_start_idx,
            CpuTableRow::RES_0,                                            // res_start_idx,
            constraint_index,                                              // constraint_idx_start,
        );

        let (sub_constraints, constraint_index) = new_sub_constraint(
            vec![CpuTableRow::SUB, CpuTableRow::BEQ], // flags_idx,
            CpuTableRow::RV1_0,                       // lhs_start_idx,
            CpuTableRow::ARG2_0,                      // rhs_start_idx,
            CpuTableRow::RES_0,                       // res_start_idx,
            constraint_index,                         // constraint_idx_start,
        );

        // Enforces RES = PC + 4 in a JALR instruction
        let (next_pc_value_constraint, constraint_index) = new_add_four_constraint(
            CpuTableRow::JALR,  // flags_idx,
            CpuTableRow::PC_0,  // rhs_start_idx,
            CpuTableRow::RES_0, // res_start_idx,
            constraint_index,   // constraint_idx_start,
        );

        let (arg2_validity_constraints, _) = new_arg2_validity_constraint(
            CpuTableRow::ARG2_0,
            CpuTableRow::RV2_0,
            CpuTableRow::IMM_0,
            Arg2ValidityColumnIndexes {
                load_index: CpuTableRow::LOAD,
                store_index: CpuTableRow::STORE,
                beq_index: CpuTableRow::BEQ,
                blt_index: CpuTableRow::BLT,
            },
            constraint_index,
        );

        let mut constraints = bit_constraints;
        constraints.extend(add_constraints);
        constraints.extend(sub_constraints);
        constraints.extend(next_pc_value_constraint);
        constraints.extend(arg2_validity_constraints);

        let num_transition_constraints = constraints.len();

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: CpuTableRow::NUM_COLUMNS,
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
        (CpuTableRow::NUM_COLUMNS, 0)
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &()
    }

    fn step_size(&self) -> usize {
        1
    }
}

// Assumption: the number of rows is a power of two
pub fn build_cpu_trace(columns: Vec<Vec<FE>>) -> CPUTraceTable {
    TraceTable::from_columns_main(columns, 1)
}

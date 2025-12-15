use crate::air::constraints_templates::{
    new_add_constraint, new_bit_constraints, new_sub_constraint,
};

use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField, quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
    },
};
use stark_platinum_prover::{
    constraints::{boundary::BoundaryConstraints, transition::TransitionConstraint},
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::AIR,
};

// CPU Columns indeces:
// const TIMESTAMP: usize = 0;
// const PC: usize = 2;
// const RS: usize = 4;
// const RD: usize = 6;
const WRITE_REGISTER: usize = 7;
const MEMORY_2BYTES: usize = 8;
const MEMORY_4BYTES: usize = 9;
// const IMM: usize = 10;
const SIGNED: usize = 12;
const MP_SELECTOR: usize = 13;
const MULDIV_SELECTOR: usize = 14;
const ADD: usize = 15;
const SUB: usize = 16;
const SLT: usize = 17;
const AND: usize = 18;
const OR: usize = 19;
const XOR: usize = 20;
const SL: usize = 21;
const SR: usize = 22;
const JALR: usize = 23;
const BEQ: usize = 24;
const BLT: usize = 25;
const LOAD: usize = 26;
const STORE: usize = 27;
const MUL: usize = 28;
const DIVREM: usize = 29;
const ECALL: usize = 30;
const EBREAK: usize = 31;
// const NEXT_PC: usize = 32;
const RV_ONE: usize = 34;
// const RV_TWO: usize = 38;
// const RVD: usize = 42;
const ARG_TWO: usize = 44;
const RES: usize = 48;
// const IS_EQUAL: usize = 52;
// const BRANCH_COND: usize = 53;

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
    trace_length: usize,
}

impl AIR for CPUTableAIR {
    type Field = Babybear31PrimeField;
    type FieldExtension = Degree4BabyBearU32ExtensionField;
    type PublicInputs = ();

    const STEP_SIZE: usize = 1;

    fn new(
        trace_length: usize,
        _pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        // Bit constraints:
        // Enforce that these columns are binary. They include:
        // - decode flags like `write_register`, `signed`, `mp_selector`, `muldiv_selector`
        // - the one-hot instruction flags (ADD, SUB, ..., EBREAK)
        let bit_columns_index_to_constraint = [
            WRITE_REGISTER,
            MEMORY_2BYTES,
            MEMORY_4BYTES,
            SIGNED,
            MP_SELECTOR,
            MULDIV_SELECTOR,
            ADD,
            SUB,
            SLT,
            AND,
            OR,
            XOR,
            SL,
            SR,
            JALR,
            BEQ,
            BLT,
            LOAD,
            STORE,
            MUL,
            DIVREM,
            ECALL,
            EBREAK,
        ];
        let bit_constraints = new_bit_constraints(&bit_columns_index_to_constraint, 0);

        let mut next_index = bit_columns_index_to_constraint.len();
        // Add constraint
        // Enforces that lhs (Word4L) + rhs (Word4L) = res (Word4L), with carry bits constrained.
        // It is enforced only on rows where the selected instruction flags are active.
        let add_constraints = new_add_constraint(
            vec![ADD, LOAD, STORE], // flags_idx,
            RV_ONE,                 // lhs_start_idx,
            ARG_TWO,                // rhs_start_idx,
            RES,                    // res_start_idx,
            next_index,             // constraint_idx_start,
        );
        next_index += 2;

        let sub_constraints = new_sub_constraint(
            vec![SUB, BEQ], // flags_idx,
            RV_ONE,         // lhs_start_idx,
            ARG_TWO,        // rhs_start_idx,
            RES,            // res_start_idx,
            next_index,     // constraint_idx_start,
        );

        let mut constraints = bit_constraints;
        constraints.extend(add_constraints);
        constraints.extend(sub_constraints);

        let num_transition_constraints = constraints.len();

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 54,
            transition_offsets: vec![0],
            num_transition_constraints,
        };

        Self {
            context,
            trace_length,
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
    ) -> BoundaryConstraints<Self::FieldExtension> {
        BoundaryConstraints::from_constraints(vec![])
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length * 2
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn trace_layout(&self) -> (usize, usize) {
        (54, 0)
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &()
    }
}

// Assumption: the number of rows is a power of two
pub fn build_cpu_trace(columns: Vec<Vec<FE>>) -> CPUTraceTable {
    TraceTable::from_columns_main(columns, 1)
}

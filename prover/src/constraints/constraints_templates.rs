use math::field::{
    element::FieldElement,
    fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField, quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
    },
    traits::{IsField, IsSubFieldOf},
};
use stark::{
    constraints::transition::TransitionConstraint, table::TableView,
    traits::TransitionEvaluationContext,
};

use crate::constraints::utils::{
    compute_element_from_two_limbs_starting_at,
    compute_element_from_two_limbs_starting_at_extension,
};

pub const INV_65536: u64 = 2013235201;

/// Enforces that a specific trace column contains only binary values (0 or 1).
/// For a trace value `x` in the specified column, the constraint enforces:
/// x * (x - 1) = 0
pub struct BitConstraint {
    column_idx: usize,
    constraint_idx: usize,
}

impl BitConstraint {
    /// Creates a new binary constraint for the specified column.
    /// * `column_idx` - The trace column index that must contain only 0 or 1
    /// * `constraint_idx` - Unique constraint identifier used by the STARK prover
    pub fn new(column_idx: usize, constraint_idx: usize) -> Self {
        Self {
            column_idx,
            constraint_idx,
        }
    }

    fn compute_bit_constraint<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let flag = step.get_main_evaluation_element(0, self.column_idx);
        let one = FieldElement::<F>::one();

        flag * (flag - one)
    }
}

impl TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>
    for BitConstraint
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn exemptions_period(&self) -> Option<usize> {
        None
    }

    fn periodic_exemptions_offset(&self) -> Option<usize> {
        None
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    /// Evaluates the bit constraint: `flag * (flag - 1) = 0`
    ///
    /// This method is called during both by the Prover and Verifier.
    /// Prover to work with base field elements while the verifier
    /// operates in a larger extension field.
    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<
            Babybear31PrimeField,
            Degree4BabyBearU32ExtensionField,
        >,
        transition_evaluations: &mut [FieldElement<Degree4BabyBearU32ExtensionField>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                let bit_constraint = self.compute_bit_constraint(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx()] = bit_constraint.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                let bit_constraint = self.compute_bit_constraint(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx()] = bit_constraint;
            }
        }
    }
}

/// Helper function to create multiple bit constraints for different columns.
///
/// # Arguments
/// * `column_idx` - Slice of column indices to constrain
/// * `constraint_idx_start` - Starting index for constraint numbering (sequential from here)
///
/// # Returns
/// A vector of boxed `BitConstraint` trait objects, one for each specified column.
pub fn new_bit_constraints(
    column_idx: &[usize],
    constraint_idx_start: usize,
) -> Vec<Box<dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>>> {
    column_idx
        .iter()
        .enumerate()
        .map(|(i, &column_idx)| {
            Box::new(BitConstraint::new(column_idx, constraint_idx_start + i))
                as Box<
                    dyn TransitionConstraint<
                        Babybear31PrimeField,
                        Degree4BabyBearU32ExtensionField,
                    >,
                >
        })
        .collect()
}

/// Identifies which carry bit (from a two-word addition) to constrain.
///
/// A 32-bit addition is split into two 16-bit word additions:
/// - **Carry 0**: The carry out of the low word (bits 0-15)
/// - **Carry 1**: The carry out of the high word (bits 16-31), which includes carry_0 as input
#[derive(Clone)]
pub enum CarryIndex {
    Zero,
    One,
}

/// Enforces correct carry bit values in multi-limb addition operations.
///
/// As the lhs and res inputs are 4-limb words, we cast them into two 16-bit words:
/// CAST(a, word2L) -> a[i] + 256 * a[i + 1]
/// Then we compute the carries to be constrained.
///
/// Carry 0:
/// lhs_0 = lhs[0] + 256 * lhs[1]
/// rhs_0 = rhs[1]
/// res_0 = res[0] + 256 * res[1]
///
/// carry_0 = (lhs_0 + rhs_0 - res_0) / 65536
/// constraint: carry_0 * (carry_0 - 1) = 0
///
/// Carry 1:
/// lhs_1 = lhs[2] + 256 * lhs[3]
/// rhs_1 = rhs[1]
/// res_1 = res[2] + 256 * res[3]
///
/// carry_1 = (lhs_1 + rhs_1 - res_1 + carry_0) / 65536
/// constraint: flag * carry_1 * (carry_1 - 1) = 0
///
/// The `flag` factor allows selective activation: the constraint is only enforced when one
/// flag column is active. (No more than 1 flag can be active at the same time)
///
/// Constraint Degree 3 (cubic), due to the multiplication of three terms: `flag * carry * (carry - 1)`.
#[derive(Clone)]
pub struct CarryBitConstraint {
    carry_idx: CarryIndex,
    flags_idx: Vec<usize>,
    lhs_start_idx: usize,
    rhs_start_idx: usize,
    res_start_idx: usize,
    constraint_idx: usize,
}

impl CarryBitConstraint {
    /// Creates a new carry bit constraint.
    ///
    /// # Arguments
    /// * `carry_idx` - Which carry to constrain (Zero or One)
    /// * `flags_idx` - Columns containing activation flags
    /// * `lhs_start_idx` - Starting column index for left operand's 4 limbs
    /// * `rhs_start_idx` - Starting column index for right operand's 2 limbs
    /// * `res_start_idx` - Starting column index for result's 4 limbs
    /// * `constraint_idx` - Unique constraint identifier
    fn new(
        carry_idx: CarryIndex,
        flags_idx: Vec<usize>,
        lhs_start_idx: usize,
        rhs_start_idx: usize,
        res_start_idx: usize,
        constraint_idx: usize,
    ) -> Self {
        Self {
            carry_idx,
            flags_idx,
            lhs_start_idx,
            rhs_start_idx,
            res_start_idx,
            constraint_idx,
        }
    }

    fn compute_carry_bit_constraint<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        // Sum all activation flags
        let flag = self
            .flags_idx
            .iter()
            .fold(FieldElement::<F>::zero(), |acc, &idx| {
                acc + step.get_main_evaluation_element(0, idx)
            });

        // Compute the low word using the first 2 operand limbs.
        let lhs_0 = compute_element_from_two_limbs_starting_at(step, self.lhs_start_idx);
        let rhs_0 = step.get_main_evaluation_element(0, self.rhs_start_idx);
        let res_0 = compute_element_from_two_limbs_starting_at(step, self.res_start_idx);

        let one = FieldElement::<F>::one();
        let inverse = FieldElement::<F>::from(INV_65536);
        let carry_0 = (lhs_0 + rhs_0 - res_0) * inverse.clone();

        match self.carry_idx {
            CarryIndex::Zero => flag * carry_0.clone() * (carry_0 - one),
            CarryIndex::One => {
                // Compute the high word using the first 2 operand limbs.
                let lhs_1 =
                    compute_element_from_two_limbs_starting_at(step, self.lhs_start_idx + 2);
                let rhs_1 = step.get_main_evaluation_element(0, self.rhs_start_idx + 1);
                let res_1 =
                    compute_element_from_two_limbs_starting_at(step, self.res_start_idx + 2);
                let carry_1 = (lhs_1 + rhs_1 - res_1 + carry_0) * inverse;
                flag * carry_1.clone() * (carry_1 - one)
            }
        }
    }
}

impl TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>
    for CarryBitConstraint
{
    fn degree(&self) -> usize {
        3
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    /// Evaluates the carry bit constraint: `flag * carry * (carry - 1) = 0`
    ///
    /// This ensures that when the instruction flag is active (flag = 1), the computed
    /// carry bit must be binary (0 or 1). When the flag is inactive (flag = 0),
    /// the constraint is trivially satisfied regardless of carry value.
    ///
    /// This method is called during both by the Prover and Verifier.
    /// Prover to work with base field elements while the verifier
    /// operates in a larger extension field.
    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<
            Babybear31PrimeField,
            Degree4BabyBearU32ExtensionField,
        >,
        transition_evaluations: &mut [FieldElement<Degree4BabyBearU32ExtensionField>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                let bit_constraint =
                    self.compute_carry_bit_constraint(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx()] = bit_constraint.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                let bit_constraint =
                    self.compute_carry_bit_constraint(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx()] = bit_constraint
            }
        }
    }
}

/// Creates a pair of carry bit constraints for a complete 32-bit addition operation.
///
/// A full 32-bit addition with 16-bit limb decomposition requires validating two carry bits:
/// - Carry from the low word (bits 0-15)
/// - Carry from the high word (bits 16-31)
///
/// This helper function creates both constraints with sequential constraint indices.
///
/// The operands use the following limb representation:
/// - `lhs` is represented as 4 limbs of 8 bits each
/// - `rhs` is represented as 4 limbs of 8 bits each
/// - `res` is represented as 4 limbs of 8 bits each
///
/// ## Arguments
/// * `flags_idx` - Column indices for instruction selector flags
/// * `lhs_start_idx` - Starting column for left operand (requires 4 consecutive columns)
/// * `rhs_start_idx` - Starting column for right operand (requires 2 consecutive columns)
/// * `res_start_idx` - Starting column for result (requires 4 consecutive columns)
/// * `constraint_idx_start` - Starting constraint index (will use idx and idx+1)
///
/// ## Returns
/// A vector of two boxed constraints: [carry_0_constraint, carry_1_constraint]
pub fn new_add_constraint(
    flags_idx: Vec<usize>,
    lhs_start_idx: usize,
    rhs_start_idx: usize,
    res_start_idx: usize,
    constraint_idx_start: usize,
) -> Vec<Box<dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>>> {
    vec![
        Box::new(CarryBitConstraint::new(
            CarryIndex::Zero,
            flags_idx.clone(),
            lhs_start_idx,
            rhs_start_idx,
            res_start_idx,
            constraint_idx_start,
        )),
        Box::new(CarryBitConstraint::new(
            CarryIndex::One,
            flags_idx,
            lhs_start_idx,
            rhs_start_idx,
            res_start_idx,
            constraint_idx_start + 1,
        )),
    ]
}

/// Creates a pair of carry bit constraints for a complete 32-bit substraction operation.
///
/// This uses the same constraints than addition
/// check that: lhs - rhs  = res is equivalent to: res + rhs = lhs
///
/// This helper function creates both constraints with sequential constraint indices.
///
/// The operands use the following limb representation:
/// - `lhs` is represented as 4 limbs of 8 bits each
/// - `rhs` is represented as 4 limbs of 8 bits each
/// - `res` is represented as 4 limbs of 8 bits each
///
/// ## Arguments
/// * `flags_idx` - Column indices for instruction selector flags
/// * `lhs_start_idx` - Starting column for left operand (requires 4 consecutive columns)
/// * `rhs_start_idx` - Starting column for right operand (requires 2 consecutive columns)
/// * `res_start_idx` - Starting column for result (requires 4 consecutive columns)
/// * `constraint_idx_start` - Starting constraint index (will use idx and idx+1)
///
/// ## Returns
/// A vector of two boxed constraints: [carry_0_constraint, carry_1_constraint]
pub fn new_sub_constraint(
    flags_idx: Vec<usize>,
    lhs_start_idx: usize,
    rhs_start_idx: usize,
    res_start_idx: usize,
    constraint_idx_start: usize,
) -> Vec<Box<dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>>> {
    vec![
        Box::new(CarryBitConstraint::new(
            CarryIndex::Zero,
            flags_idx.clone(),
            res_start_idx,
            rhs_start_idx,
            lhs_start_idx,
            constraint_idx_start,
        )),
        Box::new(CarryBitConstraint::new(
            CarryIndex::One,
            flags_idx,
            res_start_idx,
            rhs_start_idx,
            lhs_start_idx,
            constraint_idx_start + 1,
        )),
    ]
}

#[derive(Clone)]
pub struct Arg2ValidityColumnIndexes {
    pub load_index: usize,
    pub store_index: usize,
    pub beq_index: usize,
    pub blt_index: usize,
}

/// Enforces the validity of arg2.
///
/// The constraint enforces:
/// (1 - load - store) * rv2 + (1 - beq - blt) * imm - arg2 = 0
#[derive(Clone)]
pub struct Arg2ValidityConstraint {
    arg2_start_index: usize,
    rv2_start_index: usize,
    imm_start_index: usize,
    column_indexes: Arg2ValidityColumnIndexes,
    constraint_idx: usize,
}

impl Arg2ValidityConstraint {
    /// Creates a new arg2 validity constraint.
    ///
    /// # Arguments
    /// * `arg2_start_index` - Starting column index for arg2's 2 limbs
    /// * `rv2_start_index` - Starting column index for rv2's 4 limbs
    /// * `imm_start_index` - Starting column index for imm 1 limb
    /// * `column_indexes` - Column indexes for LOAD, STORE, BEQ, BLT
    /// * `constraint_idx` - Unique constraint identifier
    fn new(
        arg2_start_index: usize,
        rv2_start_index: usize,
        imm_start_index: usize,
        column_indexes: Arg2ValidityColumnIndexes,
        constraint_idx: usize,
    ) -> Self {
        Self {
            arg2_start_index,
            rv2_start_index,
            imm_start_index,
            column_indexes,
            constraint_idx,
        }
    }

    fn compute_arg2_validity_constraint<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let arg2 = step.get_main_evaluation_element(0, self.arg2_start_index);

        let rv2 = compute_element_from_two_limbs_starting_at(step, self.rv2_start_index);

        let imm = step.get_main_evaluation_element(0, self.imm_start_index);

        let one = FieldElement::<F>::one();

        let store = step.get_main_evaluation_element(0, self.column_indexes.store_index);
        let load = step.get_main_evaluation_element(0, self.column_indexes.load_index);
        let beq = step.get_main_evaluation_element(0, self.column_indexes.beq_index);
        let blt = step.get_main_evaluation_element(0, self.column_indexes.blt_index);

        (one.clone() - (store + load)) * rv2 + (one - (beq + blt)) * imm - arg2
    }
}

impl TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>
    for Arg2ValidityConstraint
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    /// Evaluates the arg2 validity constraint: `(1 - load - store) * rv2 + (1 - beq - blt) * imm - arg2 = 0`
    ///
    /// This method is called during both by the Prover and Verifier.
    /// Prover to work with base field elements while the verifier
    /// operates in a larger extension field.
    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<
            Babybear31PrimeField,
            Degree4BabyBearU32ExtensionField,
        >,
        transition_evaluations: &mut [FieldElement<Degree4BabyBearU32ExtensionField>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                let constraint =
                    self.compute_arg2_validity_constraint(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx()] = constraint.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                let constraint =
                    self.compute_arg2_validity_constraint(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx()] = constraint;
            }
        }
    }
}

/// Creates a arg2 validity constraint.
///
/// ## Arguments
/// * `arg2_start_index` - Starting column for arg2's 2 limbs
/// * `rv2_start_index` - Starting column for rv2's 4 limbs
/// * `imm_start_index` - Starting column for imm 1 limb
/// * `column_indexes` - Column indexes for LOAD, STORE, BEQ, BLT
/// * `constraint_idx_start` - Starting constraint index (will use idx and idx+1)
///
/// ## Returns
/// A boxed argv2 validity constraint.
pub fn new_arg2_validity_constraint(
    arg2_start_index: usize,
    rv2_start_index: usize,
    imm_start_index: usize,
    column_indexes: Arg2ValidityColumnIndexes,
    constraint_idx: usize,
) -> Vec<Box<dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>>> {
    vec![
        Box::new(Arg2ValidityConstraint::new(
            arg2_start_index,
            rv2_start_index,
            imm_start_index,
            column_indexes.clone(),
            constraint_idx,
        ))
            as Box<
                dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>,
            >,
        Box::new(Arg2ValidityConstraint::new(
            arg2_start_index + 1,
            rv2_start_index + 2,
            imm_start_index + 1,
            column_indexes,
            constraint_idx + 1,
        ))
            as Box<
                dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>,
            >,
    ]
}

/// Enforces correct carry bit values for adding 4 to a 32-bit table value.
/// - lhs is a 2-limb word
/// - rhs is the constant 4, casted to a 2-limb word.
/// - res is a 4-limb word, casted to a 2-limb word
///
/// Carry 0:
/// lhs_0 = lhs[0]
/// rhs_0 = 4
/// res_0 = res[0] + 256 * res[1]
///
/// carry_0 = (lhs_0 + rhs_0 - res_0) / 65536
/// constraint: carry_0 * (carry_0 - 1) = 0
///
/// Carry 1:
/// lhs_1 = lhs[1]
/// rhs_1 = 0
/// res_1 = res[2] + 256 * res[3]
///
/// carry_1 = (lhs_1 - res_1 + carry_0) / 65536
/// constraint: flag * carry_1 * (carry_1 - 1) = 0
///
/// The `flag` factor allows selective activation: the constraint is only enforced when one
/// flag column is active. (No more than 1 flag can be active at the same time)
///
/// Constraint Degree 3 (cubic), due to the multiplication of three terms: `flag * carry * (carry - 1)`.
#[derive(Clone)]
pub struct AddFourCarryBitConstraint {
    carry_idx: CarryIndex,
    flag_idx: usize,
    lhs_start_idx: usize,
    res_start_idx: usize,
    constraint_idx: usize,
}

impl AddFourCarryBitConstraint {
    /// Creates a new carry bit constraint.
    ///
    /// # Arguments
    /// * `carry_idx` - Which carry to constrain (Zero or One)
    /// * `flags_idx` - Columns containing activation flags
    /// * `lhs_start_idx` - Starting column index for left operand's 2 limbs
    /// * `res_start_idx` - Starting column index for result's 4 limbs
    /// * `constraint_idx` - Unique constraint identifier
    fn new(
        carry_idx: CarryIndex,
        flag_idx: usize,
        lhs_start_idx: usize,
        res_start_idx: usize,
        constraint_idx: usize,
    ) -> Self {
        Self {
            carry_idx,
            flag_idx,
            lhs_start_idx,
            res_start_idx,
            constraint_idx,
        }
    }
}

impl TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>
    for AddFourCarryBitConstraint
{
    fn degree(&self) -> usize {
        3
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    /// Evaluates the carry bit constraint: `flag * carry * (carry - 1) = 0`
    ///
    /// This ensures that when the instruction flag is active (flag = 1), the computed
    /// carry bit must be binary (0 or 1). When the flag is inactive (flag = 0),
    /// the constraint is trivially satisfied regardless of carry value.
    ///
    /// This method is called during both by the Prover and Verifier.
    /// Prover to work with base field elements while the verifier
    /// operates in a larger extension field.
    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<
            Babybear31PrimeField,
            Degree4BabyBearU32ExtensionField,
        >,
        transition_evaluations: &mut [FieldElement<Degree4BabyBearU32ExtensionField>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                let step = frame.get_evaluation_step(0);

                let flag = step.get_main_evaluation_element(0, self.flag_idx);

                let lhs_0 = step.get_main_evaluation_element(0, self.lhs_start_idx);
                let rhs_0 = FieldElement::<Babybear31PrimeField>::from(4);
                let res_0 = compute_element_from_two_limbs_starting_at(step, self.res_start_idx);

                let one = FieldElement::<Babybear31PrimeField>::one();
                let inverse = FieldElement::<Babybear31PrimeField>::from(INV_65536);
                let carry_0 = (lhs_0 + rhs_0 - res_0) * inverse;

                let bit_constraint: FieldElement<Babybear31PrimeField> = match self.carry_idx {
                    CarryIndex::Zero => flag * carry_0 * (carry_0 - one),
                    CarryIndex::One => {
                        // Compute the high word using the first 2 operand limbs.
                        let lhs_1 = step.get_main_evaluation_element(0, self.lhs_start_idx + 1);
                        let rhs_1 = FieldElement::<Babybear31PrimeField>::zero();
                        let res_1 = compute_element_from_two_limbs_starting_at(
                            step,
                            self.res_start_idx + 2,
                        );

                        let carry_1 = (lhs_1 + rhs_1 - res_1 + carry_0) * inverse;
                        flag * carry_1 * (carry_1 - one)
                    }
                };
                transition_evaluations[self.constraint_idx()] = bit_constraint.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                let step = frame.get_evaluation_step(0);

                let flag = step.get_main_evaluation_element(0, self.flag_idx);

                let lhs_0 = step.get_main_evaluation_element(0, self.lhs_start_idx);
                let rhs_0 = FieldElement::<Degree4BabyBearU32ExtensionField>::from(4);
                let res_0 =
                    compute_element_from_two_limbs_starting_at_extension(step, self.res_start_idx);

                let one = FieldElement::<Degree4BabyBearU32ExtensionField>::one();
                let inverse = FieldElement::<Degree4BabyBearU32ExtensionField>::from(INV_65536);
                let carry_0 = (lhs_0 + rhs_0 - res_0) * inverse;

                let bit_constraint = match self.carry_idx {
                    CarryIndex::Zero => flag * carry_0 * (carry_0 - one),
                    CarryIndex::One => {
                        let lhs_1 = step.get_main_evaluation_element(0, self.lhs_start_idx + 1);
                        let rhs_1 = FieldElement::<Degree4BabyBearU32ExtensionField>::zero();
                        let res_1 = compute_element_from_two_limbs_starting_at_extension(
                            step,
                            self.res_start_idx + 2,
                        );
                        let carry_1 = (lhs_1 + rhs_1 - res_1 + carry_0) * inverse;
                        flag * carry_1 * (carry_1 - one)
                    }
                };
                transition_evaluations[self.constraint_idx()] = bit_constraint
            }
        }
    }
}

/// Creates the carry bit constraints required for adding 4 to a 32-bit table value.
///
/// The operands use the following limb representation:
/// - `lhs` is represented as 2 limbs of 16 bits each
/// - `res` is represented as 4 limbs of 8 bits each
///
/// Requires validating two carry bits:
/// - Carry from the low word (bits 0-15)
/// - Carry from the high word (bits 16-31)
///
/// This helper function creates both constraints with sequential constraint indices.
///
/// ## Arguments
/// * `flag_idx` - Column index for instruction selector flag
/// * `lhs_start_idx` - Starting column for left operand (requires 2 consecutive columns)
/// * `res_start_idx` - Starting column for result (requires 4 consecutive columns)
/// * `constraint_idx_start` - Starting constraint index (will use idx and idx+1)
///
/// ## Returns
/// A vector of two boxed constraints: [carry_0_constraint, carry_1_constraint]
pub fn new_add_four_constraint(
    flag_idx: usize,
    lhs_start_idx: usize,
    res_start_idx: usize,
    constraint_idx_start: usize,
) -> Vec<Box<dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>>> {
    vec![
        Box::new(AddFourCarryBitConstraint::new(
            CarryIndex::Zero,
            flag_idx,
            lhs_start_idx,
            res_start_idx,
            constraint_idx_start,
        )),
        Box::new(AddFourCarryBitConstraint::new(
            CarryIndex::One,
            flag_idx,
            lhs_start_idx,
            res_start_idx,
            constraint_idx_start + 1,
        )),
    ]
}

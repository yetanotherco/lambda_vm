use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField, quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
    },
};
use stark_platinum_prover::{
    constraints::transition::TransitionConstraint, traits::TransitionEvaluationContext,
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
                let step = frame.get_evaluation_step(0);
                let flag = step.get_main_evaluation_element(0, self.column_idx);
                let one = FieldElement::<Babybear31PrimeField>::one();
                let bit_constraint = flag * (flag - one);
                transition_evaluations[self.constraint_idx()] = bit_constraint.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                let step = frame.get_evaluation_step(0);
                let flag = step.get_main_evaluation_element(0, self.column_idx);
                let one = FieldElement::<Degree4BabyBearU32ExtensionField>::one();
                let bit_constraint = flag * (flag - one);
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
}


  impl CarryBitConstraint {
      fn compute_bit_constraint<F, E>(&self, step: &stark_platinum_prover::table::TableView<'_, F, E>) -> FieldElement<F>
      where
          F: lambdaworks_math::field::traits::IsSubFieldOf<E>,
          E: lambdaworks_math::field::traits::IsField,
      {
          let two_fifty_six = FieldElement::<F>::from(256);

          let flag = self.flags_idx.iter().fold(
              FieldElement::<F>::zero(),
              |acc, &idx| acc + step.get_main_evaluation_element(0, idx),
          );

          let lhs_0 = step.get_main_evaluation_element(0, self.lhs_start_idx)
              + &two_fifty_six * step.get_main_evaluation_element(0, self.lhs_start_idx + 1);
          let rhs_0 = step.get_main_evaluation_element(0, self.rhs_start_idx);
          let res_0 = step.get_main_evaluation_element(0, self.res_start_idx)
              + &two_fifty_six * step.get_main_evaluation_element(0, self.res_start_idx + 1);

          let one = FieldElement::<F>::one();
          let inverse = FieldElement::<F>::from(INV_65536);
          let carry_0 = (lhs_0 + rhs_0 - res_0) * &inverse;

          match self.carry_idx {
              CarryIndex::Zero => flag * &carry_0 * (&carry_0 - one),
              CarryIndex::One => {
                  let lhs_1 = step.get_main_evaluation_element(0, self.lhs_start_idx + 2)
                      + &two_fifty_six * step.get_main_evaluation_element(0, self.lhs_start_idx + 3);
                  let rhs_1 = step.get_main_evaluation_element(0, self.rhs_start_idx + 1);
                  let res_1 = step.get_main_evaluation_element(0, self.res_start_idx + 2)
                      + two_fifty_six * step.get_main_evaluation_element(0, self.res_start_idx + 3);
                  let carry_1 = (lhs_1 + rhs_1 - res_1 + carry_0) * &inverse;
                  flag * &carry_1 * (&carry_1 - one)
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
              TransitionEvaluationContext::Prover { frame, .. } => {
                  let bit_constraint = self.compute_bit_constraint(frame.get_evaluation_step(0));
                  transition_evaluations[self.constraint_idx()] = bit_constraint.to_extension();
              }
              TransitionEvaluationContext::Verifier { frame, .. } => {
                  let bit_constraint = self.compute_bit_constraint(frame.get_evaluation_step(0));
                  transition_evaluations[self.constraint_idx()] = bit_constraint;
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

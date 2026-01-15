//! Type-aware LogUp implementation.
//!
//! This module extends the basic LogUp with type-aware fingerprinting.
//! It provides `TypedAirWithBuses` which is a drop-in replacement for
//! `AirWithBuses` but with support for typed value packing.
//!
//! # Key Differences from Basic LogUp
//!
//! - Uses `TypedTableInteraction` instead of `TableInteraction`
//! - Combines limbs with powers of 2 before applying α powers
//! - Fingerprint computation respects type boundaries
//!
//! # Example Usage
//!
//! ```ignore
//! use stark::lookup_typed::{TypedAirWithBuses, TypedAuxiliaryTraceBuildData};
//! use stark::lookup_types::{TypedTableInteraction, TypedValue};
//!
//! // Define an ADD interaction: [DWordWL, DWordWL] → DWordWL
//! let add_interaction = TypedTableInteraction::sender(
//!     Some(ADD_FLAG_COL),
//!     vec![
//!         TypedValue::dword_wl(LHS_COL),
//!         TypedValue::dword_wl(RHS_COL),
//!         TypedValue::dword_wl(SUM_COL),
//!     ],
//! );
//!
//! let build_data = TypedAuxiliaryTraceBuildData {
//!     interactions: vec![add_interaction],
//! };
//!
//! let air = TypedAirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
//!     num_main_columns,
//!     build_data,
//!     &proof_options,
//!     1,
//!     vec![],
//! );
//! ```

use std::marker::PhantomData;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        transition::TransitionConstraint,
    },
    context::AirContext,
    lookup::{BoundaryConstraintBuilder, BusPublicInputs, LOGUP_CHALLENGE_ALPHA, LOGUP_CHALLENGE_Z},
    lookup_types::TypedTableInteraction,
    proof::options::ProofOptions,
    table::TableView,
    trace::TraceTable,
    traits::TransitionEvaluationContext,
};

// =============================================================================
// Typed Auxiliary Trace Build Data
// =============================================================================

/// Build data for typed LogUp auxiliary trace.
///
/// Contains a list of typed interactions that define how the table
/// participates in bus communications.
#[derive(Clone)]
pub struct TypedAuxiliaryTraceBuildData {
    pub interactions: Vec<TypedTableInteraction>,
}

// =============================================================================
// Typed Air With Buses
// =============================================================================

/// AIR with type-aware LogUp bus support.
///
/// This is similar to `AirWithBuses` but uses `TypedTableInteraction`
/// for type-aware fingerprint computation.
///
/// # Type Parameters
///
/// - `F`: Base field (e.g., BabyBear)
/// - `E`: Extension field
/// - `B`: Boundary constraint builder
/// - `PI`: Public inputs type
pub struct TypedAirWithBuses<
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    B: BoundaryConstraintBuilder<F, E, PI>,
    PI,
> {
    context: AirContext,
    step_size: usize,
    trace_layout: (usize, usize),
    transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    auxiliary_trace_build_data: TypedAuxiliaryTraceBuildData,
    boundary_constraint_builder: PhantomData<(B, PI)>,
}

impl<
        F: IsFFTField + IsSubFieldOf<E> + Send + Sync + 'static,
        E: IsField + Send + Sync + 'static,
        B: BoundaryConstraintBuilder<F, E, PI>,
        PI,
    > TypedAirWithBuses<F, E, B, PI>
{
    /// Creates a TypedAirWithBuses with type-aware LogUp constraints.
    ///
    /// # Arguments
    ///
    /// * `num_main_columns` - Number of main trace columns
    /// * `auxiliary_trace_build_data` - Typed interaction definitions
    /// * `proof_options` - STARK proof options
    /// * `step_size` - Step size for transition constraints
    /// * `transition_constraints` - Additional user-defined constraints
    ///
    /// # Auxiliary Column Layout
    ///
    /// - Columns 0..N-1: Term columns (one per interaction)
    /// - Column N: Accumulated column (running sum)
    ///
    /// Total aux columns = N + 1 where N is the number of interactions.
    pub fn new(
        num_main_columns: usize,
        auxiliary_trace_build_data: TypedAuxiliaryTraceBuildData,
        proof_options: &ProofOptions,
        step_size: usize,
        mut transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    ) -> Self {
        let num_interactions = auxiliary_trace_build_data.interactions.len();

        // Add a typed term constraint for each interaction
        for (i, interaction) in auxiliary_trace_build_data.interactions.iter().enumerate() {
            let constraint =
                TypedLookupTermConstraint::new(interaction.clone(), i, transition_constraints.len());
            transition_constraints.push(Box::new(constraint));
        }

        // Add the accumulated constraint
        if num_interactions > 0 {
            let accumulated_constraint =
                TypedLookupAccumulatedConstraint::new(transition_constraints.len(), num_interactions);
            transition_constraints.push(Box::new(accumulated_constraint));
        }

        // Layout: N term columns + 1 accumulated column
        let num_aux_columns = if num_interactions > 0 {
            num_interactions + 1
        } else {
            0
        };
        let trace_layout = (num_main_columns, num_aux_columns);

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: trace_layout.0 + trace_layout.1,
            transition_offsets: vec![0, 1],
            num_transition_constraints: transition_constraints.len(),
        };

        Self {
            context,
            step_size,
            trace_layout,
            transition_constraints,
            auxiliary_trace_build_data,
            boundary_constraint_builder: PhantomData,
        }
    }
}

impl<F, E, B, PI> crate::traits::AIR for TypedAirWithBuses<F, E, B, PI>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    B: BoundaryConstraintBuilder<F, E, PI>,
    PI: Send + Sync,
{
    type Field = F;
    type FieldExtension = E;
    type PublicInputs = PI;

    fn step_size(&self) -> usize {
        self.step_size
    }

    fn new(_proof_options: &ProofOptions) -> Self
    where
        Self: Sized,
    {
        unreachable!("TypedAirWithBuses should only be created via TypedAirWithBuses::new()")
    }

    fn trace_layout(&self) -> (usize, usize) {
        self.trace_layout
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length * 2
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn transition_constraints(&self) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> {
        &self.transition_constraints
    }

    fn build_auxiliary_trace(
        &self,
        trace: &mut TraceTable<F, E>,
        challenges: &[FieldElement<E>],
    ) -> Option<BusPublicInputs<E>> {
        let (_, num_aux_columns) = self.trace_layout();
        if num_aux_columns > 0 && trace.num_aux_columns == 0 {
            trace.allocate_aux_table(num_aux_columns);
        }

        let num_interactions = self.auxiliary_trace_build_data.interactions.len();
        if num_interactions == 0 {
            return None;
        }

        // Build term columns with type-aware fingerprinting
        for (i, interaction) in self.auxiliary_trace_build_data.interactions.iter().enumerate() {
            build_typed_logup_term_column(i, interaction, trace, challenges);
        }

        // Build accumulated column
        let acc_col_idx = num_interactions;
        build_accumulated_column(acc_col_idx, num_interactions, trace);

        // Return BusPublicInputs
        let last_row = trace.num_rows() - 1;
        Some(BusPublicInputs {
            initial_value: trace.get_aux(0, acc_col_idx).clone(),
            final_accumulated: trace.get_aux(last_row, acc_col_idx).clone(),
        })
    }

    fn build_rap_challenges(
        &self,
        transcript: &mut dyn IsStarkTranscript<E, F>,
    ) -> Vec<FieldElement<E>> {
        vec![
            transcript.sample_field_element(), // z
            transcript.sample_field_element(), // alpha
        ]
    }

    fn boundary_constraints(
        &self,
        pub_inputs: &Self::PublicInputs,
        rap_challenges: &[FieldElement<E>],
        bus_public_inputs: Option<&BusPublicInputs<E>>,
        trace_length: usize,
    ) -> BoundaryConstraints<E> {
        let mut boundary_constraints = vec![];

        if let Some(acc_interaction) = bus_public_inputs {
            let acc_col_idx = self.auxiliary_trace_build_data.interactions.len();

            boundary_constraints.push(BoundaryConstraint::new_aux(
                acc_col_idx,
                0,
                acc_interaction.initial_value.clone(),
            ));
            boundary_constraints.push(BoundaryConstraint::new_aux(
                acc_col_idx,
                trace_length - 1,
                acc_interaction.final_accumulated.clone(),
            ));
        }

        boundary_constraints.extend(B::boundary_constraints(pub_inputs, rap_challenges));
        BoundaryConstraints::from_constraints(boundary_constraints)
    }
}

// =============================================================================
// Typed Term Column Building
// =============================================================================

/// Builds a term column for a typed table interaction.
///
/// Each row contains: `term[i] = sign * multiplicity[i] / fingerprint[i]`
///
/// The fingerprint is computed type-aware:
/// 1. For each TypedValue, combine columns with powers of 2
/// 2. Combine resulting bus elements with powers of α
/// 3. fingerprint = z - combined
fn build_typed_logup_term_column<F, E>(
    aux_column_idx: usize,
    interaction: &TypedTableInteraction,
    trace: &mut TraceTable<F, E>,
    challenges: &[FieldElement<E>],
) where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    let main_cols = trace.columns_main();
    let trace_len = trace.num_rows();

    // Handle optional multiplicity
    let multiplicity_owned: Vec<FieldElement<F>>;
    let multiplicity: &[FieldElement<F>] = match interaction.multiplicity_column {
        Some(col) => &main_cols[col],
        None => {
            multiplicity_owned = vec![FieldElement::one(); trace_len];
            &multiplicity_owned
        }
    };

    let z = &challenges[LOGUP_CHALLENGE_Z];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];

    // Sign: +1 for senders, -1 for receivers
    let sign = if interaction.is_sender {
        FieldElement::<E>::one()
    } else {
        -FieldElement::<E>::one()
    };

    for row in 0..trace_len {
        // Compute type-aware fingerprint
        let fingerprint = interaction.compute_fingerprint(
            |col_idx| main_cols[col_idx][row].clone().to_extension(),
            z,
            alpha,
        );

        // term = sign * multiplicity / fingerprint
        let term = &multiplicity[row]
            * &sign
            * fingerprint
                .inv()
                .expect("fingerprint is zero - probability negligible");

        trace.set_aux(row, aux_column_idx, term);
    }
}

/// Builds the accumulated column that sums all term columns across rows.
fn build_accumulated_column<F, E>(
    acc_column_idx: usize,
    num_term_columns: usize,
    trace: &mut TraceTable<F, E>,
) where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    let trace_len = trace.num_rows();
    let mut accumulated = FieldElement::<E>::zero();

    for row in 0..trace_len {
        let mut row_sum = FieldElement::<E>::zero();
        for term_col in 0..num_term_columns {
            row_sum = row_sum + trace.get_aux(row, term_col);
        }
        accumulated += row_sum;
        trace.set_aux(row, acc_column_idx, accumulated.clone());
    }
}

// =============================================================================
// Typed Lookup Term Constraint
// =============================================================================

/// Constraint for typed term columns.
///
/// Verifies: `term[i] * fingerprint[i] - sign * multiplicity[i] = 0`
///
/// where fingerprint is computed type-aware.
struct TypedLookupTermConstraint {
    interaction: TypedTableInteraction,
    term_column_idx: usize,
    constraint_idx: usize,
}

impl TypedLookupTermConstraint {
    pub fn new(
        interaction: TypedTableInteraction,
        term_column_idx: usize,
        constraint_idx: usize,
    ) -> Self {
        Self {
            interaction,
            term_column_idx,
            constraint_idx,
        }
    }
}

impl<F, E> TransitionConstraint<F, E> for TypedLookupTermConstraint
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        // term * fingerprint where fingerprint is linear in main trace
        // But with typed combining, fingerprint can have higher degree
        // due to multiplications with shift constants
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        fn evaluate_typed_term<'a, A: IsSubFieldOf<B>, B: IsField>(
            step: &TableView<'a, A, B>,
            term_column_idx: usize,
            interaction: &TypedTableInteraction,
            rap_challenges: &[FieldElement<B>],
        ) -> FieldElement<B> {
            let term = step.get_aux_evaluation_element(0, term_column_idx);

            let z = &rap_challenges[LOGUP_CHALLENGE_Z];
            let alpha = &rap_challenges[LOGUP_CHALLENGE_ALPHA];

            // Multiplicity
            let multiplicity: FieldElement<B> = match interaction.multiplicity_column {
                Some(col) => step.get_main_evaluation_element(0, col).clone().to_extension(),
                None => FieldElement::<B>::one(),
            };

            // Type-aware fingerprint
            let fingerprint = interaction.compute_fingerprint(
                |col_idx| step.get_main_evaluation_element(0, col_idx).clone().to_extension(),
                z,
                alpha,
            );

            // Sign
            let sign = if interaction.is_sender {
                FieldElement::<B>::one()
            } else {
                -FieldElement::<B>::one()
            };

            // Constraint: term * fingerprint - sign * multiplicity = 0
            term * &fingerprint - multiplicity * sign
        }

        let res = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                rap_challenges,
                ..
            } => evaluate_typed_term(
                frame.get_evaluation_step(0),
                self.term_column_idx,
                &self.interaction,
                rap_challenges,
            ),
            TransitionEvaluationContext::Verifier {
                frame,
                rap_challenges,
                ..
            } => evaluate_typed_term(
                frame.get_evaluation_step(0),
                self.term_column_idx,
                &self.interaction,
                rap_challenges,
            ),
        };

        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx) {
            *eval = res;
        }
    }
}

// =============================================================================
// Typed Lookup Accumulated Constraint
// =============================================================================

/// Constraint for the accumulated column.
///
/// Verifies: `acc[i+1] = acc[i] + sum_k(term_k[i+1])`
struct TypedLookupAccumulatedConstraint {
    constraint_idx: usize,
    num_term_columns: usize,
    acc_column_idx: usize,
}

impl TypedLookupAccumulatedConstraint {
    pub fn new(constraint_idx: usize, num_term_columns: usize) -> Self {
        Self {
            constraint_idx,
            num_term_columns,
            acc_column_idx: num_term_columns,
        }
    }
}

impl<F, E> TransitionConstraint<F, E> for TypedLookupAccumulatedConstraint
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        1
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        fn evaluate_accumulated<'a, A: IsSubFieldOf<B>, B: IsField>(
            first_step: &TableView<'a, A, B>,
            second_step: &TableView<'a, A, B>,
            acc_column_idx: usize,
            num_term_columns: usize,
        ) -> FieldElement<B> {
            let acc_curr = first_step.get_aux_evaluation_element(0, acc_column_idx);
            let acc_next = second_step.get_aux_evaluation_element(0, acc_column_idx);

            let terms_sum: FieldElement<B> = (0..num_term_columns)
                .map(|i| second_step.get_aux_evaluation_element(0, i).clone())
                .sum();

            acc_next - acc_curr - terms_sum
        }

        let res = match evaluation_context {
            TransitionEvaluationContext::Prover { frame, .. } => evaluate_accumulated(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                self.acc_column_idx,
                self.num_term_columns,
            ),
            TransitionEvaluationContext::Verifier { frame, .. } => evaluate_accumulated(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                self.acc_column_idx,
                self.num_term_columns,
            ),
        };

        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx) {
            *eval = res;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lookup::NullBoundaryConstraintBuilder;
    use crate::lookup_types::TypedValue;
    use crate::trace::TraceTable;
    use crate::traits::AIR;
    use math::field::fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField,
        quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
    };

    type F = Babybear31PrimeField;
    type E = Degree4BabyBearU32ExtensionField;
    type FE = FieldElement<F>;

    /// Test that typed interactions work with a simple sender/receiver pair.
    #[test]
    fn test_typed_air_with_buses_simple() {
        // Create a simple trace with 4 rows
        // Columns: [mult, a, b, c] where c = a + b (conceptually)
        let num_rows = 4;
        let num_cols = 4;

        let mut columns: Vec<Vec<FE>> = vec![vec![FE::zero(); num_rows]; num_cols];

        // Fill with test data
        // Row 0: mult=1, a=10, b=20, c=30
        // Row 1: mult=1, a=5, b=15, c=20
        // Row 2: mult=0, a=0, b=0, c=0 (padding)
        // Row 3: mult=0, a=0, b=0, c=0 (padding)
        columns[0][0] = FE::from(1u64); // mult
        columns[1][0] = FE::from(10u64); // a
        columns[2][0] = FE::from(20u64); // b
        columns[3][0] = FE::from(30u64); // c

        columns[0][1] = FE::from(1u64);
        columns[1][1] = FE::from(5u64);
        columns[2][1] = FE::from(15u64);
        columns[3][1] = FE::from(20u64);

        // Create trace
        let mut trace: TraceTable<F, E> = TraceTable::from_columns_main(columns, 1);

        // Define a typed interaction using Single type (no combining)
        let interaction = TypedTableInteraction::sender(
            Some(0), // multiplicity column
            vec![
                TypedValue::single(1), // a
                TypedValue::single(2), // b
                TypedValue::single(3), // c
            ],
        );

        let build_data = TypedAuxiliaryTraceBuildData {
            interactions: vec![interaction],
        };

        // Create AIR
        let proof_options = ProofOptions::default_test_options();
        let air = TypedAirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
            num_cols,
            build_data,
            &proof_options,
            1,
            vec![],
        );

        // Verify layout
        assert_eq!(air.trace_layout(), (4, 2)); // 4 main, 2 aux (1 term + 1 acc)

        // Build auxiliary trace with dummy challenges
        let z = FieldElement::<E>::from(12345u64);
        let alpha = FieldElement::<E>::from(67890u64);
        let challenges = vec![z, alpha];

        let bus_public_inputs = air.build_auxiliary_trace(&mut trace, &challenges);

        // Should have bus public inputs
        assert!(bus_public_inputs.is_some());
        let bpi = bus_public_inputs.unwrap();

        // Final accumulated should be non-zero (we have 2 rows with mult=1)
        assert_ne!(bpi.final_accumulated, FieldElement::<E>::zero());
    }

    /// Test with DWordWL type (2 columns combined)
    #[test]
    fn test_typed_interaction_dword_wl() {
        // Columns: [mult, lhs_lo, lhs_hi, rhs_lo, rhs_hi, sum_lo, sum_hi]
        let num_rows = 4;
        let num_cols = 7;

        let mut columns: Vec<Vec<FE>> = vec![vec![FE::zero(); num_rows]; num_cols];

        // Row 0: 100 + 200 = 300 (as 64-bit split into two 32-bit words)
        columns[0][0] = FE::from(1u64); // mult
        columns[1][0] = FE::from(100u64); // lhs_lo
        columns[2][0] = FE::from(0u64); // lhs_hi
        columns[3][0] = FE::from(200u64); // rhs_lo
        columns[4][0] = FE::from(0u64); // rhs_hi
        columns[5][0] = FE::from(300u64); // sum_lo
        columns[6][0] = FE::from(0u64); // sum_hi

        let mut trace: TraceTable<F, E> = TraceTable::from_columns_main(columns, 1);

        // Define interaction with DWordWL types
        let interaction = TypedTableInteraction::sender(
            Some(0),
            vec![
                TypedValue::dword_wl(1), // lhs: columns 1,2
                TypedValue::dword_wl(3), // rhs: columns 3,4
                TypedValue::dword_wl(5), // sum: columns 5,6
            ],
        );

        let build_data = TypedAuxiliaryTraceBuildData {
            interactions: vec![interaction],
        };

        let proof_options = ProofOptions::default_test_options();
        let air = TypedAirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
            num_cols,
            build_data,
            &proof_options,
            1,
            vec![],
        );

        let z = FieldElement::<E>::from(99999u64);
        let alpha = FieldElement::<E>::from(11111u64);
        let challenges = vec![z, alpha];

        let bus_public_inputs = air.build_auxiliary_trace(&mut trace, &challenges);
        assert!(bus_public_inputs.is_some());
    }
}

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
    proof::options::ProofOptions,
    table::TableView,
    trace::TraceTable,
    traits::TransitionEvaluationContext,
};

// =============================================================================
// LogUp Challenge Indices
// =============================================================================
// The LogUp protocol requires two random challenges sampled via Fiat-Shamir:
//
// - `z`: The evaluation point for the fingerprint. Each row's values are compressed
//   into a single field element as: fingerprint = 1 / (z - linear_combination)
//
// - `alpha`: The base for the linear combination of column values within a row.
//   For values [v0, v1, ..., vn], the linear combination is: v0 + v1*α + v2*α² + ...
//
// These challenges MUST be shared across all AIRs in a multi-table proof for the
// LogUp bus to balance correctly (sum of all fingerprints equals zero).

/// Index of the `z` challenge in the LogUp challenges vector.
/// Used as the evaluation point in fingerprint computation.
pub const LOGUP_CHALLENGE_Z: usize = 0;

/// Index of the `alpha` (α) challenge in the LogUp challenges vector.
/// Used as the base for linear combination of row values.
pub const LOGUP_CHALLENGE_ALPHA: usize = 1;

/// Number of challenges required by the LogUp protocol.
pub const LOGUP_NUM_CHALLENGES: usize = 2;

/// Struct representing an AIR with Lookup. Contains own implementation of boundary constraints and auxiliary trace building
pub struct AirWithBuses<
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    B: BoundaryConstraintBuilder<F, E, PI>,
    PI,
> {
    context: AirContext,
    step_size: usize,
    trace_layout: (usize, usize),
    transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    auxiliary_trace_build_data: AuxiliaryTraceBuildData,
    boundary_constraint_builder: PhantomData<(B, PI)>,
}

impl<
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    B: BoundaryConstraintBuilder<F, E, PI>,
    PI,
> AirWithBuses<F, E, B, PI>
{
    /// Creates an AirWithBuses with LogUp-specific transition constraints.
    /// If no boundary constraints are needed, use `NullBoundaryConstraintBuilder` as B and () as PI.
    pub fn new(
        num_main_columns: usize,
        auxiliary_trace_build_data: AuxiliaryTraceBuildData,
        proof_options: &ProofOptions,
        step_size: usize,
        mut transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    ) -> Self {
        // Add a transition constraint for each auxiliary column representing a table interaction
        for (i, interaction) in auxiliary_trace_build_data.interactions.iter().enumerate() {
            let constraint = LookupTransitionConstraint::new(
                interaction.clone(),
                i,
                transition_constraints.len(),
            );
            transition_constraints.push(Box::new(constraint));
        }
        // Add a transition constraint for the grand sum auxiliary constraint (sum of all previous aux columns) if we have more than one interaction
        if auxiliary_trace_build_data.interactions.len() > 1 {
            let grand_sum_constraint = LookupGrandSumTransitionConstraint::new(
                transition_constraints.len(),
                auxiliary_trace_build_data.interactions.len(),
            );
            transition_constraints.push(Box::new(grand_sum_constraint));
        }

        // Create Layout
        let num_aux_columns = auxiliary_trace_build_data.interactions.len()
            + (auxiliary_trace_build_data.interactions.len() > 1) as usize;
        let trace_layout = (num_main_columns, num_aux_columns);

        // Create context
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

impl<F, E, B, PI> crate::traits::AIR for AirWithBuses<F, E, B, PI>
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

    fn new(_proof_options: &crate::proof::options::ProofOptions) -> Self
    where
        Self: Sized,
    {
        // AirWithBuses should be created using `AirWithBuses::new` method
        unreachable!("AirWithBuses should only be created via AirWithBuses::new()")
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

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> {
        &self.transition_constraints
    }

    fn build_auxiliary_trace(
        &self,
        trace: &mut TraceTable<F, E>,
        challenges: &[FieldElement<E>],
    ) -> Vec<BusPublicInputs<E>> {
        // Allocate aux table if not already present
        let (_, num_aux_columns) = self.trace_layout();
        if num_aux_columns > 0 && trace.num_aux_columns == 0 {
            trace.allocate_aux_table(num_aux_columns);
        }

        let last_row = trace.num_rows() - 1;
        let mut bus_interactions = Vec::new();

        // Build aux column for each interaction
        for (i, interaction) in self
            .auxiliary_trace_build_data
            .interactions
            .iter()
            .enumerate()
        {
            build_auxiliary_trace_column(i, interaction, trace, challenges);
            // Collect both initial (row 0) and final (last row) values
            bus_interactions.push(BusPublicInputs {
                initial_value: trace.get_aux(0, i).clone(),
                final_accumulated: trace.get_aux(last_row, i).clone(),
                is_sender: interaction.is_sender,
            });
        }

        // If there are multiple interactions, build the grand sum column
        if self.auxiliary_trace_build_data.interactions.len() > 1 {
            let grand_sum_col_idx = self.auxiliary_trace_build_data.interactions.len();
            for row in 0..trace.num_rows() {
                let mut grand_sum = FieldElement::<E>::zero();
                for i in 0..self.auxiliary_trace_build_data.interactions.len() {
                    grand_sum = grand_sum + trace.get_aux(row, i);
                }
                trace.set_aux(row, grand_sum_col_idx, grand_sum);
            }
        }

        bus_interactions
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
        bus_interactions: Option<&[BusPublicInputs<E>]>,
        trace_length: usize,
    ) -> BoundaryConstraints<E> {
        let mut boundary_constraints = vec![];

        // Boundary constraints for aux columns (from bus interactions in proof)
        if let Some(interactions) = bus_interactions {
            for (i, interaction) in interactions.iter().enumerate() {
                // Constraint for row 0: aux column must start with initial_value
                boundary_constraints.push(BoundaryConstraint::new_aux(
                    i,
                    0,
                    interaction.initial_value.clone(),
                ));
                // Constraint for last row: aux column must end with final_accumulated
                boundary_constraints.push(BoundaryConstraint::new_aux(
                    i,
                    trace_length - 1,
                    interaction.final_accumulated.clone(),
                ));
            }
        }

        // User-defined boundary constraints
        boundary_constraints.extend(B::boundary_constraints(pub_inputs, rap_challenges));

        BoundaryConstraints::from_constraints(boundary_constraints)
    }
}

/// Struct representing how each lookup air should build its auxiliary trace
/// Contains a list of all lookup interactions
pub struct AuxiliaryTraceBuildData {
    pub interactions: Vec<TableInteraction>,
}

/// Struct representing a lookup interaction for a given table.
/// Contains the multiplicity and value columns involved in said interaction.
#[derive(Clone)]
pub struct TableInteraction {
    /// Column index containing the multiplicity for this interaction.
    /// Can be a binary flag (0 or 1) or a general multiplicity (0, 1, 2, ...).
    /// Determines how many times each row contributes to the bus.
    /// If None, a constant multiplicity of 1 is used for all rows.
    pub multiplicity_column: Option<usize>,
    pub value_columns: Vec<usize>,
    /// Whether this side of the interaction is a sender (true) or receiver (false).
    /// Senders contribute positive values to the bus sum, receivers contribute negative.
    /// For bus balance: Σ sender_values - Σ receiver_values = 0
    pub is_sender: bool,
}

/// Public inputs for a single bus interaction.
/// Contains the initial and final aux column values needed for boundary constraints
/// and bus balance verification.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct BusPublicInputs<E>
where
    E: IsField,
{
    /// Aux column value at row 0 (initial fingerprint)
    pub initial_value: FieldElement<E>,
    /// Aux column value at last row (accumulated sum)
    pub final_accumulated: FieldElement<E>,
    /// Whether this interaction is a sender (true) or receiver (false).
    /// Senders contribute positive values to the bus sum, receivers contribute negative.
    /// For bus balance: Σ sender_values - Σ receiver_values = 0
    pub is_sender: bool,
}

/// Trait representing boundary constraint building behaviour.
///  Should be defined when creating an `AirWithBuses` if the AIR requires its own boundary constraints aside from the lookup ones
pub trait BoundaryConstraintBuilder<
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    PI,
>: Send + Sync
{
    fn boundary_constraints(
        _pub_inputs: &PI,
        _rap_challenges: &[FieldElement<E>],
    ) -> Vec<BoundaryConstraint<E>> {
        vec![]
    }
}

/// NoOp implementor of `BoundaryConstraintBuilder` for `AirWithBuses`s than don't use other boundary constraints
pub struct NullBoundaryConstraintBuilder {}
impl<F, E, PI> BoundaryConstraintBuilder<F, E, PI> for NullBoundaryConstraintBuilder
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
}

/// Builds an auxiliary trace column from the given table interaction
fn build_auxiliary_trace_column<F, E>(
    aux_column_idx: usize,
    table_interaction: &TableInteraction,
    trace: &mut TraceTable<F, E>,
    challenges: &[FieldElement<E>],
) where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    // Main table
    let main_segment_cols = trace.columns_main();
    let values = table_interaction
        .value_columns
        .iter()
        .map(|i| &main_segment_cols[*i])
        .collect::<Vec<_>>();

    let trace_len = trace.num_rows();

    // Handle optional multiplicity column - use constant 1 if None
    let multiplicity_owned: Vec<FieldElement<F>>;
    let multiplicity: &[FieldElement<F>] = match table_interaction.multiplicity_column {
        Some(col) => &main_segment_cols[col],
        None => {
            multiplicity_owned = vec![FieldElement::one(); trace_len];
            &multiplicity_owned
        }
    };

    // LogUp challenges (must be shared across all tables for bus to balance)
    let z = &challenges[LOGUP_CHALLENGE_Z];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];
    // Coefficients for each value column
    let coeffs: Vec<FieldElement<E>> = (0..values.len()).map(|i| alpha.pow(i)).collect();

    let mut aux_col: Vec<FieldElement<E>> = Vec::new();

    // fingerprint = z - (v[0] * alpha^0 + v[1] * alpha^1 +...+ value[n] * alpha^n)
    // Where v are the values for each row and n the number of value columns
    // We calculate the first fingerprint separately using the values from the first row
    // Fingerprint can only be zero if z equals the linear combination of values,
    // which happens with negligible probability since z is randomly sampled over the extension field
    let fingerprint_inv: FieldElement<E> = (-(values
        .iter()
        .zip(coeffs.iter())
        .map(|(v, coeff)| &v[0] * coeff)
        .sum::<FieldElement<E>>())
        + z)
        .inv()
        .expect("fingerprint is zero - probability of sampling zero is negligible");
    // Fill first aux column row
    aux_col.push(&multiplicity[0] * fingerprint_inv);

    for i in 0..trace_len - 1 {
        // fingerprint = z - (v[0] * alpha^0 + v[1] * alpha^1 +...+ value[n] * alpha^n)
        // Where v are the values for each row and n the number of value columns
        let fingerprint_inv: FieldElement<E> = (-(values
            .iter()
            .zip(coeffs.iter())
            .map(|(v, coeff)| &v[i + 1] * coeff)
            .sum::<FieldElement<E>>())
            + z)
            .inv()
            .expect("fingerprint is zero - probability of sampling zero is negligible");
        // Fill the auxiliary column row
        aux_col.push(&aux_col[i] + &multiplicity[i + 1] * fingerprint_inv);
    }

    for (i, aux_elem) in aux_col.iter().enumerate().take(trace.num_rows()) {
        trace.set_aux(i, aux_column_idx, aux_elem.clone())
    }
}

// Constraint for each auxiliary column representing a table interaction
// Checks the calculation of the next auxiliary column value based on the next row's multiplicity and values
struct LookupTransitionConstraint {
    // Indicates columns with multiplicity and values used to build the auxiliary column
    interaction: TableInteraction,
    // Index of the auxiliary column
    interaction_number: usize,
    // Index of the constraint
    constraint_idx: usize,
}

impl LookupTransitionConstraint {
    pub fn new(
        interaction: TableInteraction,
        interaction_number: usize,
        constraint_idx: usize,
    ) -> Self {
        Self {
            interaction,
            interaction_number,
            constraint_idx,
        }
    }
}

impl<F, E> TransitionConstraint<F, E> for LookupTransitionConstraint
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
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
        fn evaluate_lookup_constraint<'a, A: IsSubFieldOf<B>, B: IsField>(
            first_step: &TableView<'a, A, B>,
            second_step: &TableView<'a, A, B>,
            aux_column_idx: usize,
            interaction: &TableInteraction,
            rap_challenges: &&[FieldElement<B>],
        ) -> FieldElement<B> {
            // Auxiliary frame elements
            let s0 = first_step.get_aux_evaluation_element(0, aux_column_idx);
            let s1 = second_step.get_aux_evaluation_element(0, aux_column_idx);

            let z = &rap_challenges[LOGUP_CHALLENGE_Z];
            let alpha = &rap_challenges[LOGUP_CHALLENGE_ALPHA];

            // Main frame elements - handle optional multiplicity
            let multiplicity = match interaction.multiplicity_column {
                Some(col) => second_step.get_main_evaluation_element(0, col).clone(),
                None => FieldElement::<A>::one(),
            };
            let values = interaction
                .value_columns
                .iter()
                .map(|c| second_step.get_main_evaluation_element(0, *c))
                .collect::<Vec<_>>();

            // Coefficients for each value column
            let coeffs: Vec<FieldElement<B>> = (0..values.len()).map(|i| alpha.pow(i)).collect();

            // fingerprint = z - (v[0] * alpha^0 + v[1] * alpha^1 +...+ value[n] * alpha^n)
            // Where v are the values for each row and n the number of value columns
            let fingerprint: FieldElement<B> = (-values
                .iter()
                .zip(coeffs.iter())
                .map(|(v, coeff)| *v * coeff)
                .sum::<FieldElement<B>>())
                + z;

            // We are using the following LogUp equation:
            // s1 = s0 + multiplicity / fingerprint
            // 0 = s0 * fingerprint + multiplicity - s1 * fingerprint
            // Since constraints must be expressed without division, we rearrange:
            multiplicity + s0 * &fingerprint - s1 * fingerprint
        }

        let res = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                rap_challenges,
                ..
            } => evaluate_lookup_constraint(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                self.interaction_number,
                &self.interaction,
                rap_challenges,
            ),
            TransitionEvaluationContext::Verifier {
                frame,
                rap_challenges,
                ..
            } => evaluate_lookup_constraint(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                self.interaction_number,
                &self.interaction,
                rap_challenges,
            ),
        };
        // The eval always exists, except if the constraint idx were incorrectly defined.
        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx) {
            *eval = res;
        }
    }
}

/// Constraint for the last auxiliary column
/// Checks that the grand sum column is the sum of all previous auxiliary columns
struct LookupGrandSumTransitionConstraint {
    // Index of the constraint
    constraint_idx: usize,
    // Amount of interactions -> we could infer this from the amount of aux columns
    interaction_amount: usize,
}

impl LookupGrandSumTransitionConstraint {
    pub fn new(constraint_idx: usize, interaction_amount: usize) -> Self {
        Self {
            constraint_idx,
            interaction_amount,
        }
    }
}

impl<F, E> TransitionConstraint<F, E> for LookupGrandSumTransitionConstraint
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
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
        fn evaluate_grand_sum_constraint<'a, A: IsSubFieldOf<B>, B: IsField>(
            step: &TableView<'a, A, B>,
            aux_column_idx: usize,
        ) -> FieldElement<B> {
            // Auxiliary frame elements
            let grand_sum = step.get_aux_evaluation_element(0, aux_column_idx);

            let interaction_values_sum: FieldElement<B> = (0..aux_column_idx)
                .map(|i| step.get_aux_evaluation_element(0, i).clone())
                .sum();

            // Check that the grand sum is equal to the sum of all other auxiliary columns in the same row
            // Aka that we correctly built the grand sum auxiliary column
            grand_sum - interaction_values_sum
        }
        let res = match evaluation_context {
            TransitionEvaluationContext::Prover { frame, .. } => {
                evaluate_grand_sum_constraint(frame.get_evaluation_step(0), self.interaction_amount)
            }

            TransitionEvaluationContext::Verifier { frame, .. } => {
                evaluate_grand_sum_constraint(frame.get_evaluation_step(0), self.interaction_amount)
            }
        };
        // The eval always exists, except if the constraint idx were incorrectly defined.
        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx) {
            *eval = res;
        }
    }
}

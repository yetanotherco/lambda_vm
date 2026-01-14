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
    ///
    /// Auxiliary column layout:
    /// - Columns 0..N-1: Term columns (one per interaction), each containing ±m[i]/fp[i]
    /// - Column N: Accumulated column, containing the running sum of all terms
    ///
    /// Total aux columns = N + 1 where N is the number of interactions.
    pub fn new(
        num_main_columns: usize,
        auxiliary_trace_build_data: AuxiliaryTraceBuildData,
        proof_options: &ProofOptions,
        step_size: usize,
        mut transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    ) -> Self {
        let num_interactions = auxiliary_trace_build_data.interactions.len();

        // Add a term constraint for each interaction
        // Each term constraint checks: term[i] * fingerprint[i] = sign * multiplicity[i]
        for (i, interaction) in auxiliary_trace_build_data.interactions.iter().enumerate() {
            let constraint =
                LookupTermConstraint::new(interaction.clone(), i, transition_constraints.len());
            transition_constraints.push(Box::new(constraint));
        }

        // Add the accumulated constraint (always, even for 1 interaction)
        // This checks: acc[i+1] = acc[i] + sum of all terms at row i+1
        if num_interactions > 0 {
            let accumulated_constraint =
                LookupAccumulatedConstraint::new(transition_constraints.len(), num_interactions);
            transition_constraints.push(Box::new(accumulated_constraint));
        }

        // Create Layout: N term columns + 1 accumulated column
        let num_aux_columns = if num_interactions > 0 {
            num_interactions + 1
        } else {
            0
        };
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

        let num_interactions = self.auxiliary_trace_build_data.interactions.len();

        if num_interactions == 0 {
            return vec![];
        }

        // Build term columns (one per interaction)
        // Each term column contains: sign * m[i] / fp[i]
        for (i, interaction) in self
            .auxiliary_trace_build_data
            .interactions
            .iter()
            .enumerate()
        {
            build_logup_term_column(i, interaction, trace, challenges);
        }

        // Build accumulated column (sums all term columns across rows)
        let acc_col_idx = num_interactions;
        build_accumulated_column(acc_col_idx, num_interactions, trace);

        // Return single BusPublicInputs for the accumulated column
        let last_row = trace.num_rows() - 1;
        vec![BusPublicInputs {
            initial_value: trace.get_aux(0, acc_col_idx).clone(),
            final_accumulated: trace.get_aux(last_row, acc_col_idx).clone(),
        }]
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

        // Boundary constraints for the accumulated column only
        // (term columns are fully determined by main trace and don't need boundary constraints)
        if let Some(interactions) = bus_interactions
            && let Some(acc_interaction) = interactions.first()
        {
            // The accumulated column is at index = num_interactions
            let acc_col_idx = self.auxiliary_trace_build_data.interactions.len();

            // Constraint for row 0: accumulated column must start with initial_value
            boundary_constraints.push(BoundaryConstraint::new_aux(
                acc_col_idx,
                0,
                acc_interaction.initial_value.clone(),
            ));
            // Constraint for last row: accumulated column must end with final_accumulated
            boundary_constraints.push(BoundaryConstraint::new_aux(
                acc_col_idx,
                trace_length - 1,
                acc_interaction.final_accumulated.clone(),
            ));
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

/// Public inputs for a table's accumulated LogUp column.
/// Contains the initial and final values needed for boundary constraints
/// and bus balance verification.
///
/// Each table has exactly one BusPublicInputs, representing its accumulated column.
/// The sign (sender vs receiver) is already baked into the accumulated values,
/// so the bus balance check is simply: Σ final_accumulated across all tables = 0
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct BusPublicInputs<E>
where
    E: IsField,
{
    /// Accumulated column value at row 0
    pub initial_value: FieldElement<E>,
    /// Accumulated column value at last row (total sum of all terms)
    pub final_accumulated: FieldElement<E>,
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

/// Builds a term column for a table interaction.
/// Each row contains: sign * multiplicity[i] / fingerprint[i]
/// where sign = +1 for senders, -1 for receivers.
/// This is NOT accumulated - just the individual term for each row.
fn build_logup_term_column<F, E>(
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

    // Sign: +1 for senders, -1 for receivers
    // This bakes the sign into the term so the accumulated column can just sum everything
    let sign = if table_interaction.is_sender {
        FieldElement::<E>::one()
    } else {
        -FieldElement::<E>::one()
    };

    for row in 0..trace_len {
        // fingerprint = z - (v[0] * alpha^0 + v[1] * alpha^1 +...+ value[n] * alpha^n)
        // Fingerprint can only be zero if z equals the linear combination of values,
        // which happens with negligible probability since z is randomly sampled over the extension field
        let fingerprint: FieldElement<E> = -(values
            .iter()
            .zip(coeffs.iter())
            .map(|(v, coeff)| &v[row] * coeff)
            .sum::<FieldElement<E>>())
            + z;

        // term = sign * multiplicity / fingerprint
        // Convert multiplicity from base field F to extension field E
        let mult_ext: FieldElement<E> = multiplicity[row].clone().to_extension();
        let term = &sign
            * mult_ext
            * fingerprint
                .inv()
                .expect("fingerprint is zero - probability of sampling zero is negligible");
        trace.set_aux(row, aux_column_idx, term);
    }
}

/// Builds the accumulated column that sums all term columns across rows.
/// acc[0] = sum of all term columns at row 0
/// acc[i] = acc[i-1] + sum of all term columns at row i
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
        // Sum all term columns for this row
        let mut row_sum = FieldElement::<E>::zero();
        for term_col in 0..num_term_columns {
            row_sum = row_sum + trace.get_aux(row, term_col);
        }

        // Add to running accumulated value
        accumulated += row_sum;
        trace.set_aux(row, acc_column_idx, accumulated.clone());
    }
}

/// Constraint for each term column.
/// Checks that: aux_k[i] * fingerprint[i] = sign * multiplicity[i]
/// where sign = +1 for senders, -1 for receivers.
/// This is NOT a running sum - just verifying each term is correctly computed.
struct LookupTermConstraint {
    // Indicates columns with multiplicity and values used to compute the term
    interaction: TableInteraction,
    // Index of the term column (aux column)
    term_column_idx: usize,
    // Index of the constraint
    constraint_idx: usize,
}

impl LookupTermConstraint {
    pub fn new(
        interaction: TableInteraction,
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

impl<F, E> TransitionConstraint<F, E> for LookupTermConstraint
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        2 // aux * fingerprint (fingerprint is linear in main trace values)
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0 // Check all rows including the last
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        fn evaluate_term_constraint<'a, A: IsSubFieldOf<B>, B: IsField>(
            step: &TableView<'a, A, B>,
            term_column_idx: usize,
            interaction: &TableInteraction,
            rap_challenges: &&[FieldElement<B>],
        ) -> FieldElement<B> {
            // Term column value
            let term = step.get_aux_evaluation_element(0, term_column_idx);

            let z = &rap_challenges[LOGUP_CHALLENGE_Z];
            let alpha = &rap_challenges[LOGUP_CHALLENGE_ALPHA];

            // Main frame elements - handle optional multiplicity
            let multiplicity: FieldElement<A> = match interaction.multiplicity_column {
                Some(col) => step.get_main_evaluation_element(0, col).clone(),
                None => FieldElement::<A>::one(),
            };
            let values = interaction
                .value_columns
                .iter()
                .map(|c| step.get_main_evaluation_element(0, *c))
                .collect::<Vec<_>>();

            // Coefficients for each value column
            let coeffs: Vec<FieldElement<B>> = (0..values.len()).map(|i| alpha.pow(i)).collect();

            // fingerprint = z - (v[0] * alpha^0 + v[1] * alpha^1 +...+ value[n] * alpha^n)
            let fingerprint: FieldElement<B> = (-values
                .iter()
                .zip(coeffs.iter())
                .map(|(v, coeff)| *v * coeff)
                .sum::<FieldElement<B>>())
                + z;

            // Sign: +1 for senders, -1 for receivers
            let sign = if interaction.is_sender {
                FieldElement::<B>::one()
            } else {
                -FieldElement::<B>::one()
            };

            // Constraint: term * fingerprint = sign * multiplicity
            // Rearranged: term * fingerprint - sign * multiplicity = 0
            // Convert multiplicity from base field A to extension field B
            let mult_ext: FieldElement<B> = multiplicity.to_extension();
            term * &fingerprint - sign * mult_ext
        }

        let res = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                rap_challenges,
                ..
            } => evaluate_term_constraint(
                frame.get_evaluation_step(0),
                self.term_column_idx,
                &self.interaction,
                rap_challenges,
            ),
            TransitionEvaluationContext::Verifier {
                frame,
                rap_challenges,
                ..
            } => evaluate_term_constraint(
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

/// Constraint for the accumulated column.
/// Checks that: acc[i+1] = acc[i] + sum of all term columns at row i+1
/// This is the running sum that accumulates all terms across all interactions.
struct LookupAccumulatedConstraint {
    // Index of the constraint
    constraint_idx: usize,
    // Number of term columns (one per interaction)
    num_term_columns: usize,
    // Index of the accumulated column (= num_term_columns)
    acc_column_idx: usize,
}

impl LookupAccumulatedConstraint {
    pub fn new(constraint_idx: usize, num_term_columns: usize) -> Self {
        Self {
            constraint_idx,
            num_term_columns,
            acc_column_idx: num_term_columns,
        }
    }
}

impl<F, E> TransitionConstraint<F, E> for LookupAccumulatedConstraint
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        1 // Just additions, no multiplications with main trace
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        1 // Last row doesn't have a "next row"
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        fn evaluate_accumulated_constraint<'a, A: IsSubFieldOf<B>, B: IsField>(
            first_step: &TableView<'a, A, B>,
            second_step: &TableView<'a, A, B>,
            acc_column_idx: usize,
            num_term_columns: usize,
        ) -> FieldElement<B> {
            // Accumulated column values
            let acc_curr = first_step.get_aux_evaluation_element(0, acc_column_idx);
            let acc_next = second_step.get_aux_evaluation_element(0, acc_column_idx);

            // Sum of all term columns at the next step
            let terms_sum: FieldElement<B> = (0..num_term_columns)
                .map(|i| second_step.get_aux_evaluation_element(0, i).clone())
                .sum();

            // Constraint: acc[i+1] = acc[i] + sum of terms at row i+1
            // Rearranged: acc[i+1] - acc[i] - terms_sum = 0
            acc_next - acc_curr - terms_sum
        }

        let res = match evaluation_context {
            TransitionEvaluationContext::Prover { frame, .. } => evaluate_accumulated_constraint(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                self.acc_column_idx,
                self.num_term_columns,
            ),
            TransitionEvaluationContext::Verifier { frame, .. } => evaluate_accumulated_constraint(
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

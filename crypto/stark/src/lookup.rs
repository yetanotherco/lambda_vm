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
// Shift Constants for Type Combining
// =============================================================================

/// 2^8 - shift for combining bytes
pub const SHIFT_8: u64 = 256;
/// 2^16 - shift for combining halves
pub const SHIFT_16: u64 = 65536;
/// 2^32 - shift for combining words
pub const SHIFT_32: u64 = 4294967296;

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

// =============================================================================
// Bus Types
// =============================================================================

/// Defines how multiple columns (limbs) are combined into bus elements.
///
/// Values are combined in two stages:
/// 1. **Casting** (powers of 2): Combine limbs within a type (e.g., 4 bytes → 1 word)
/// 2. **Bus fingerprint** (powers of α): Combine all typed values into one fingerprint
///
/// Each variant specifies:
/// - How many columns it consumes
/// - How many bus elements it produces
/// - The shift factors (powers of 2) used for combining
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packing {
    /// Direct: single field element, no combining.
    /// Columns: 1, Bus elements: 1
    /// Used for: Bit, Byte, Half, Word, B4, B20, etc.
    Direct,

    /// Two 16-bit halves → one 32-bit word.
    /// Columns: 2, Bus elements: 1
    /// Combination: h₀ + 2¹⁶·h₁
    Word2L,

    /// Four 8-bit bytes → one 32-bit word.
    /// Columns: 4, Bus elements: 1
    /// Combination: b₀ + 2⁸·b₁ + 2¹⁶·b₂ + 2²⁴·b₃
    Word4L,

    /// Two 32-bit words → two bus elements (no combining within, just grouping).
    /// Columns: 2, Bus elements: 2
    /// Each word stays as-is: [w₀, w₁]
    /// Note: For bus, these become w₀ + α·w₁
    DWordWL,

    /// Four 16-bit halves → two bus elements (pairs combined).
    /// Columns: 4, Bus elements: 2
    /// Combination: [h₀ + 2¹⁶·h₁, h₂ + 2¹⁶·h₃]
    DWordHL,

    /// Eight 8-bit bytes → two bus elements (quads combined).
    /// Columns: 8, Bus elements: 2
    /// Combination: [b₀ + 2⁸·b₁ + 2¹⁶·b₂ + 2²⁴·b₃, b₄ + 2⁸·b₅ + 2¹⁶·b₆ + 2²⁴·b₇]
    DWordBL,

    /// Word + Half + Half → two bus elements.
    /// Columns: 3, Bus elements: 2
    /// Layout: [Word (LSB), Half, Half (MSB)]
    /// Combination: [w₀, h₀ + 2¹⁶·h₁]
    DWordHHW,

    /// Half + Half + Word → two bus elements.
    /// Columns: 3, Bus elements: 2
    /// Layout: [Half (LSB), Half, Word (MSB)]
    /// Combination: [h₀ + 2¹⁶·h₁, w₀]
    DWordWHH,
}

impl Packing {
    /// Returns the number of trace columns this type consumes.
    pub fn num_columns(&self) -> usize {
        match self {
            Packing::Direct => 1,
            Packing::Word2L => 2,
            Packing::Word4L => 4,
            Packing::DWordWL => 2,
            Packing::DWordHL => 4,
            Packing::DWordBL => 8,
            Packing::DWordHHW => 3,
            Packing::DWordWHH => 3,
        }
    }

    /// Returns the number of bus elements this type produces after combining.
    pub fn num_bus_elements(&self) -> usize {
        match self {
            Packing::Direct => 1,
            Packing::Word2L => 1,
            Packing::Word4L => 1,
            Packing::DWordWL => 2,
            Packing::DWordHL => 2,
            Packing::DWordBL => 2,
            Packing::DWordHHW => 2,
            Packing::DWordWHH => 2,
        }
    }

    /// Creates TypedValues at the given start columns.
    ///
    /// Each element in `start_columns` becomes a separate TypedValue using this packing.
    ///
    /// # Example
    /// ```ignore
    /// Packing::Direct.columns(&[0, 1, 2])  // 3 direct values at columns 0, 1, 2
    /// Packing::DWordWL.columns(&[0, 2])    // 2 DWordWL values: cols 0-1 and cols 2-3
    /// Packing::DWordHHW.columns(&[0])      // 1 DWordHHW value at cols 0, 1, 2
    /// ```
    pub fn columns(self, start_columns: &[usize]) -> Vec<TypedValue> {
        start_columns
            .iter()
            .map(|&col| TypedValue::new(col, self))
            .collect()
    }

    /// Combines column values into bus elements using powers of 2.
    ///
    /// # Arguments
    /// * `columns` - Slice of field elements from the trace columns
    ///
    /// # Returns
    /// Vector of combined bus elements
    ///
    /// # Panics
    /// If `columns.len() != self.num_columns()`
    pub fn combine<E: IsField>(&self, columns: &[FieldElement<E>]) -> Vec<FieldElement<E>> {
        assert_eq!(
            columns.len(),
            self.num_columns(),
            "Packing {:?} expects {} columns, got {}",
            self,
            self.num_columns(),
            columns.len()
        );

        match self {
            Packing::Direct => {
                vec![columns[0].clone()]
            }

            Packing::Word2L => {
                // h₀ + 2¹⁶·h₁
                let shift_16 = FieldElement::<E>::from(SHIFT_16);
                vec![&columns[0] + &columns[1] * &shift_16]
            }

            Packing::Word4L => {
                // b₀ + 2⁸·b₁ + 2¹⁶·b₂ + 2²⁴·b₃
                let shift_8 = FieldElement::<E>::from(SHIFT_8);
                let shift_16 = FieldElement::<E>::from(SHIFT_16);
                let shift_24 = &shift_8 * &shift_16;
                vec![
                    &columns[0]
                        + &columns[1] * &shift_8
                        + &columns[2] * &shift_16
                        + &columns[3] * &shift_24,
                ]
            }

            Packing::DWordWL => {
                // Two words, no combining - each stays as bus element
                // [w₀, w₁]
                vec![columns[0].clone(), columns[1].clone()]
            }

            Packing::DWordHL => {
                // [h₀ + 2¹⁶·h₁, h₂ + 2¹⁶·h₃]
                let shift_16 = FieldElement::<E>::from(SHIFT_16);
                vec![
                    &columns[0] + &columns[1] * &shift_16,
                    &columns[2] + &columns[3] * &shift_16,
                ]
            }

            Packing::DWordBL => {
                // [b₀ + 2⁸·b₁ + 2¹⁶·b₂ + 2²⁴·b₃, b₄ + 2⁸·b₅ + 2¹⁶·b₆ + 2²⁴·b₇]
                let shift_8 = FieldElement::<E>::from(SHIFT_8);
                let shift_16 = FieldElement::<E>::from(SHIFT_16);
                let shift_24 = &shift_8 * &shift_16;
                vec![
                    &columns[0]
                        + &columns[1] * &shift_8
                        + &columns[2] * &shift_16
                        + &columns[3] * &shift_24,
                    &columns[4]
                        + &columns[5] * &shift_8
                        + &columns[6] * &shift_16
                        + &columns[7] * &shift_24,
                ]
            }

            Packing::DWordHHW => {
                // [Word (LSB), Half, Half (MSB)]
                // → [w₀, h₀ + 2¹⁶·h₁]
                let shift_16 = FieldElement::<E>::from(SHIFT_16);
                vec![
                    columns[0].clone(),
                    &columns[1] + &columns[2] * &shift_16,
                ]
            }

            Packing::DWordWHH => {
                // [Half (LSB), Half, Word (MSB)]
                // → [h₀ + 2¹⁶·h₁, w₀]
                let shift_16 = FieldElement::<E>::from(SHIFT_16);
                vec![
                    &columns[0] + &columns[1] * &shift_16,
                    columns[2].clone(),
                ]
            }
        }
    }
}

// =============================================================================
// Typed Value
// =============================================================================

/// A typed value for bus interactions.
///
/// Specifies which trace columns hold the limbs and how to combine them.
#[derive(Debug, Clone)]
pub struct TypedValue {
    /// Starting column index in the trace
    pub start_column: usize,
    /// How to interpret and combine the columns
    pub bus_type: Packing,
}

impl TypedValue {
    /// Creates a new typed value.
    ///
    /// Prefer using `Packing::at()` instead:
    /// ```ignore
    /// Packing::Direct.columns(0)
    /// Packing::DWordWL.columns(1)
    /// ```
    pub fn new(start_column: usize, bus_type: Packing) -> Self {
        Self {
            start_column,
            bus_type,
        }
    }

    /// Returns the number of columns this value spans.
    pub fn num_columns(&self) -> usize {
        self.bus_type.num_columns()
    }

    /// Returns the number of bus elements this value produces.
    pub fn num_bus_elements(&self) -> usize {
        self.bus_type.num_bus_elements()
    }

    /// Returns the column indices this value uses.
    pub fn column_indices(&self) -> Vec<usize> {
        (self.start_column..self.start_column + self.num_columns()).collect()
    }

    /// Extracts column values from a row and combines them.
    ///
    /// # Arguments
    /// * `get_column` - Function to get column value by index
    ///
    /// # Returns
    /// Vector of combined bus elements
    pub fn combine_from<E: IsField, F: Fn(usize) -> FieldElement<E>>(
        &self,
        get_column: F,
    ) -> Vec<FieldElement<E>> {
        let columns: Vec<_> = self.column_indices().iter().map(|&i| get_column(i)).collect();
        self.bus_type.combine(&columns)
    }
}

// =============================================================================
// AirWithBuses
// =============================================================================

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
        // Each term constraint verifies: term[i] = sign * multiplicity[i] / fingerprint[i]
        // Rearranged as: term[i] * fingerprint[i] - sign * multiplicity[i] = 0
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
    ) -> Option<BusPublicInputs<E>> {
        // Allocate aux table if not already present
        let (_, num_aux_columns) = self.trace_layout();
        if num_aux_columns > 0 && trace.num_aux_columns == 0 {
            trace.allocate_aux_table(num_aux_columns);
        }

        let num_interactions = self.auxiliary_trace_build_data.interactions.len();

        if num_interactions == 0 {
            return None;
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

        // Boundary constraints for the accumulated column only
        // (term columns are fully determined by main trace and don't need boundary constraints)
        if let Some(acc_interaction) = bus_public_inputs {
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
    pub interactions: Vec<BusInteraction>,
}

/// Struct representing a lookup interaction for a given table.
/// Contains the multiplicity and typed values involved in said interaction.
///
/// Values are combined in two stages:
/// 1. **Casting** (powers of 2): Combine limbs within each TypedValue
/// 2. **Bus fingerprint** (powers of α): Combine all bus elements into one fingerprint
#[derive(Clone)]
pub struct BusInteraction {
    /// Column index containing the multiplicity for this interaction.
    /// Can be a binary flag (0 or 1) or a general multiplicity (0, 1, 2, ...).
    /// Determines how many times each row contributes to the bus.
    /// If None, a constant multiplicity of 1 is used for all rows.
    pub multiplicity_column: Option<usize>,
    /// Typed values that make up this interaction
    pub values: Vec<TypedValue>,
    /// Whether this side of the interaction is a sender (true) or receiver (false).
    /// Senders contribute positive values to the bus sum, receivers contribute negative.
    /// For bus balance: Σ sender_values - Σ receiver_values = 0
    pub is_sender: bool,
}

impl BusInteraction {
    /// Creates a new table interaction.
    pub fn new(
        multiplicity_column: Option<usize>,
        values: Vec<TypedValue>,
        is_sender: bool,
    ) -> Self {
        Self {
            multiplicity_column,
            values,
            is_sender,
        }
    }

    /// Creates a sender interaction.
    pub fn sender(multiplicity_column: Option<usize>, values: Vec<TypedValue>) -> Self {
        Self::new(multiplicity_column, values, true)
    }

    /// Creates a receiver interaction.
    pub fn receiver(multiplicity_column: Option<usize>, values: Vec<TypedValue>) -> Self {
        Self::new(multiplicity_column, values, false)
    }

    /// Returns total number of bus elements (for α power computation).
    pub fn num_bus_elements(&self) -> usize {
        self.values.iter().map(|v| v.num_bus_elements()).sum()
    }
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
///
/// Each row contains the LogUp quotient: `term[i] = sign * multiplicity[i] / fingerprint[i]`
///
/// where:
/// - `fingerprint[i] = z - (v0 + v1*α + v2*α² + ...)` (bus elements after type combining)
/// - `sign = +1` for senders, `-1` for receivers
/// - `multiplicity` = number of times this row contributes to the bus
///
/// This is NOT accumulated - just the individual contribution for each row.
fn build_logup_term_column<F, E>(
    aux_column_idx: usize,
    table_interaction: &BusInteraction,
    trace: &mut TraceTable<F, E>,
    challenges: &[FieldElement<E>],
) where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    let main_segment_cols = trace.columns_main();
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

    // Precompute powers of alpha for all bus elements
    let num_bus_elements = table_interaction.num_bus_elements();
    let alpha_powers: Vec<FieldElement<E>> =
        (0..num_bus_elements).map(|i| alpha.pow(i)).collect();

    // Sign: +1 for senders, -1 for receivers
    // This bakes the sign into the term so the accumulated column can just sum everything
    let sign = if table_interaction.is_sender {
        FieldElement::<E>::one()
    } else {
        -FieldElement::<E>::one()
    };

    for row in 0..trace_len {
        // Stage 1: Combine each typed value's columns using powers of 2
        // Stage 2: Flatten all bus elements and combine with powers of α
        let bus_elements: Vec<FieldElement<E>> = table_interaction
            .values
            .iter()
            .flat_map(|tv| {
                let columns: Vec<FieldElement<E>> = tv
                    .column_indices()
                    .iter()
                    .map(|&col| main_segment_cols[col][row].clone().to_extension())
                    .collect();
                tv.bus_type.combine(&columns)
            })
            .collect();

        // fingerprint = z - (v[0] * alpha^0 + v[1] * alpha^1 + ... + v[n] * alpha^n)
        let linear_combination: FieldElement<E> = bus_elements
            .iter()
            .zip(alpha_powers.iter())
            .map(|(v, coeff)| v * coeff)
            .sum();

        let fingerprint = z - &linear_combination;

        // term = sign * multiplicity / fingerprint
        let term = &multiplicity[row]
            * &sign
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
///
/// Verifies: `term[i] = sign * multiplicity[i] / fingerprint[i]`
///
/// Rearranged to avoid division: `term[i] * fingerprint[i] - sign * multiplicity[i] = 0`
///
/// where:
/// - `fingerprint[i] = z - (v0 + v1*α + v2*α² + ...)` (bus elements after type combining)
/// - `sign = +1` for senders, `-1` for receivers
struct LookupTermConstraint {
    // Indicates columns with multiplicity and values used to compute the term
    interaction: BusInteraction,
    // Index of the term column (aux column)
    term_column_idx: usize,
    // Index of the constraint
    constraint_idx: usize,
}

impl LookupTermConstraint {
    pub fn new(
        interaction: BusInteraction,
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
            interaction: &BusInteraction,
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

            // Stage 1: Combine each typed value's columns using powers of 2
            // Stage 2: Flatten all bus elements
            let bus_elements: Vec<FieldElement<B>> = interaction
                .values
                .iter()
                .flat_map(|tv| {
                    let columns: Vec<FieldElement<B>> = tv
                        .column_indices()
                        .iter()
                        .map(|&col| step.get_main_evaluation_element(0, col).clone().to_extension())
                        .collect();
                    tv.bus_type.combine(&columns)
                })
                .collect();

            // Coefficients for each bus element
            let coeffs: Vec<FieldElement<B>> =
                (0..bus_elements.len()).map(|i| alpha.pow(i)).collect();

            // fingerprint = z - (v[0] * alpha^0 + v[1] * alpha^1 + ... + v[n] * alpha^n)
            let fingerprint: FieldElement<B> = (-bus_elements
                .iter()
                .zip(coeffs.iter())
                .map(|(v, coeff)| v * coeff)
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
            term * &fingerprint - multiplicity * sign
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
///
/// Verifies: `acc[i+1] = acc[i] + sum_k(term_k[i+1])`
///
/// Rearranged: `acc[i+1] - acc[i] - sum_k(term_k[i+1]) = 0`
///
/// where `term_k[i] = sign * multiplicity[i] / fingerprint[i]` for the k-th interaction.
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::TraceTable;
    use crate::traits::AIR;
    use math::field::fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField,
        quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
    };

    type F = Babybear31PrimeField;
    type E = Degree4BabyBearU32ExtensionField;
    type FE = FieldElement<F>;

    #[test]
    fn test_word4l_combine() {
        // 4 bytes: [0x12, 0x34, 0x56, 0x78]
        // Expected: 0x78563412 (little-endian)
        let bytes = vec![
            FE::from(0x12u64),
            FE::from(0x34u64),
            FE::from(0x56u64),
            FE::from(0x78u64),
        ];
        let combined = Packing::Word4L.combine(&bytes);
        assert_eq!(combined.len(), 1);
        // 0x12 + 0x34*256 + 0x56*65536 + 0x78*16777216 = 2018915346
        assert_eq!(combined[0], FE::from(0x78563412u64));
    }

    #[test]
    fn test_dword_hl_combine() {
        // 4 halves: [0x1234, 0x5678, 0x9ABC, 0xDEF0]
        // Expected: [0x56781234, 0xDEF09ABC]
        let halves = vec![
            FE::from(0x1234u64),
            FE::from(0x5678u64),
            FE::from(0x9ABCu64),
            FE::from(0xDEF0u64),
        ];
        let combined = Packing::DWordHL.combine(&halves);
        assert_eq!(combined.len(), 2);
        assert_eq!(combined[0], FE::from(0x56781234u64));
        assert_eq!(combined[1], FE::from(0xDEF09ABCu64));
    }

    #[test]
    fn test_dword_hhw_combine() {
        // [Word, Half, Half] where Word is LSB
        // columns: [0xAABBCCDD, 0x1234, 0x5678]
        // Expected: [0xAABBCCDD, 0x56781234]
        let cols = vec![
            FE::from(0xAABBCCDDu64),
            FE::from(0x1234u64),
            FE::from(0x5678u64),
        ];
        let combined = Packing::DWordHHW.combine(&cols);
        assert_eq!(combined.len(), 2);
        assert_eq!(combined[0], FE::from(0xAABBCCDDu64));
        assert_eq!(combined[1], FE::from(0x56781234u64));
    }

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

        // Define a typed interaction using Direct (no combining)
        let interaction = BusInteraction::sender(
            Some(0), // multiplicity column
            Packing::Direct.columns(&[1, 2, 3]), // a, b, c
        );

        let build_data = AuxiliaryTraceBuildData {
            interactions: vec![interaction],
        };

        // Create AIR
        let proof_options = ProofOptions::default_test_options();
        let air = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
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
        // lhs: columns 1,2 | rhs: columns 3,4 | sum: columns 5,6
        let interaction = BusInteraction::sender(Some(0), Packing::DWordWL.columns(&[1, 3, 5]));

        let build_data = AuxiliaryTraceBuildData {
            interactions: vec![interaction],
        };

        let proof_options = ProofOptions::default_test_options();
        let air = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
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

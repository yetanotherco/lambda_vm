#[cfg(feature = "debug-checks")]
use std::collections::HashMap;
use std::marker::PhantomData;

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
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
};
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

// =============================================================================
// Shift Constants for Type Combining
// =============================================================================

/// 2^8 - shift for combining bytes
pub const SHIFT_8: u64 = 256;
/// 2^16 - shift for combining halves
pub const SHIFT_16: u64 = 65536;
/// 2^32 - shift for combining words
pub const SHIFT_32: u64 = 4294967296;

/// Computes powers of alpha incrementally: [1, α, α², α³, ...]
///
/// This is more efficient than calling `alpha.pow(i)` for each i,
/// as it only requires one multiplication per element instead of
/// a full exponentiation.
pub(crate) fn compute_alpha_powers<E: IsField>(
    alpha: &FieldElement<E>,
    count: usize,
) -> Vec<FieldElement<E>> {
    let mut powers = Vec::with_capacity(count);
    let mut current = FieldElement::<E>::one();
    for _ in 0..count {
        powers.push(current.clone());
        current = &current * alpha;
    }
    powers
}

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
/// ## Primitive vs Compound Packings
///
/// **Primitive** packings define unique combining formulas:
/// - `Direct`, `Word2L`, `Word4L`
///
/// **Compound** packings are built from primitives (for convenience):
/// - `DWordHL` = 2× Word2L
/// - `DWordBL` = 2× Word4L
/// - `DWordHHW` = Direct + Word2L
/// - `DWordWHH` = Word2L + Direct
/// - `QuadHL` = 4× Word2L
///
/// Compound packings delegate to primitives internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packing {
    // =========================================================================
    // Primitive packings - define unique combining formulas
    // =========================================================================
    /// Single field element, no combining.
    /// Columns: 1, Bus elements: 1
    /// Used for: Bit, Byte, Half, Word, B4, B20, etc.
    Direct,

    /// Two 16-bit halves → one 32-bit word.
    /// Columns: 2, Bus elements: 1
    /// Formula: h₀ + 2¹⁶·h₁
    Word2L,

    /// Four 8-bit bytes → one 32-bit word.
    /// Columns: 4, Bus elements: 1
    /// Formula: b₀ + 2⁸·b₁ + 2¹⁶·b₂ + 2²⁴·b₃
    Word4L,

    // =========================================================================
    // Compound packings - built from primitives above
    // Sorted by: output count, then input count
    // =========================================================================
    /// 2 words → 2 bus elements. **Compound: 2× Direct.**
    /// Columns: 2, Bus elements: 2
    /// No combining, just groups two words together.
    DWordWL,

    /// [Word, Half, Half] → 2 elements. **Compound: Direct + Word2L.**
    /// Columns: 3, Bus elements: 2
    /// Layout: Word is LSB.
    DWordHHW,

    /// [Half, Half, Word] → 2 elements. **Compound: Word2L + Direct.**
    /// Columns: 3, Bus elements: 2
    /// Layout: Word is MSB.
    DWordWHH,

    /// 4 halves → 2 words. **Compound: 2× Word2L.**
    /// Columns: 4, Bus elements: 2
    DWordHL,

    /// 8 bytes → 2 words. **Compound: 2× Word4L.**
    /// Columns: 8, Bus elements: 2
    DWordBL,

    /// 8 halves → 4 words. **Compound: 4× Word2L.**
    /// Columns: 8, Bus elements: 4
    QuadHL,

    /// 4 words → 4 bus elements. **Compound: 4× Direct.**
    /// Columns: 4, Bus elements: 4
    /// Used for MUL table result (128-bit as 4 words).
    QuadWL,
}

impl Packing {
    /// Returns the number of trace columns this type consumes.
    pub fn num_columns(&self) -> usize {
        match self {
            // Primitives
            Packing::Direct => 1,
            Packing::Word2L => 2,
            Packing::Word4L => 4,
            // Compounds (sorted by output count, then input count)
            Packing::DWordWL => 2,  // 2× Direct
            Packing::DWordHHW => 3, // Direct + Word2L
            Packing::DWordWHH => 3, // Word2L + Direct
            Packing::DWordHL => 4,  // 2× Word2L
            Packing::DWordBL => 8,  // 2× Word4L
            Packing::QuadHL => 8,   // 4× Word2L
            Packing::QuadWL => 4,   // 4× Direct
        }
    }

    /// Returns the number of bus elements this type produces after combining.
    pub fn num_bus_elements(&self) -> usize {
        match self {
            // Primitives
            Packing::Direct => 1,
            Packing::Word2L => 1,
            Packing::Word4L => 1,
            // Compounds (sorted by output count, then input count)
            Packing::DWordWL => 2,  // 2× Direct
            Packing::DWordHHW => 2, // Direct + Word2L
            Packing::DWordWHH => 2, // Word2L + Direct
            Packing::DWordHL => 2,  // 2× Word2L
            Packing::DWordBL => 2,  // 2× Word4L
            Packing::QuadHL => 4,   // 4× Word2L
            Packing::QuadWL => 4,   // 4× Direct
        }
    }

    /// Creates BusValues at the given start columns.
    ///
    /// Each element in `start_columns` becomes a separate `BusValue::Packed` using this packing.
    ///
    /// Examples:
    /// - `Packing::Direct.columns(&[0, 1, 2])` - 3 direct values at columns 0, 1, 2
    /// - `Packing::DWordHL.columns(&[0, 4])` - 2 DWordHL values: cols 0-3 and cols 4-7
    /// - `Packing::DWordHHW.columns(&[0])` - 1 DWordHHW value at cols 0, 1, 2
    pub fn columns(self, start_columns: &[usize]) -> Vec<BusValue> {
        start_columns
            .iter()
            .map(|&col| BusValue::Packed {
                start_column: col,
                packing: self,
            })
            .collect()
    }

    /// Accumulates the fingerprint contribution of this packing into `acc`.
    ///
    /// Computes: `acc += Σ combined_element_i * alpha_powers[alpha_offset + i]`
    /// where `combined_element_i` are the bus elements produced by this packing.
    ///
    /// Returns the number of alpha powers consumed (= num_bus_elements()).
    ///
    /// This avoids allocating intermediate Vecs for the combined elements.
    /// `main_cols` are column-major: `main_cols[col][row]`.
    pub fn accumulate_fingerprint<F, E>(
        &self,
        main_cols: &[Vec<FieldElement<F>>],
        row: usize,
        start_col: usize,
        alpha_powers: &[FieldElement<E>],
        alpha_offset: usize,
        acc: &mut FieldElement<E>,
    ) -> usize
    where
        F: IsField + IsSubFieldOf<E>,
        E: IsField,
    {
        match self {
            Packing::Direct => {
                *acc += &main_cols[start_col][row] * &alpha_powers[alpha_offset];
                1
            }
            Packing::Word2L => {
                let shift_16 = FieldElement::<F>::from(SHIFT_16);
                let combined =
                    &main_cols[start_col][row] + &main_cols[start_col + 1][row] * &shift_16;
                *acc += &combined * &alpha_powers[alpha_offset];
                1
            }
            Packing::Word4L => {
                let shift_8 = FieldElement::<F>::from(SHIFT_8);
                let shift_16 = FieldElement::<F>::from(SHIFT_16);
                let shift_24 = &shift_8 * &shift_16;
                let combined = &main_cols[start_col][row]
                    + &main_cols[start_col + 1][row] * &shift_8
                    + &main_cols[start_col + 2][row] * &shift_16
                    + &main_cols[start_col + 3][row] * &shift_24;
                *acc += &combined * &alpha_powers[alpha_offset];
                1
            }
            // Compound packings: decompose into primitives.
            // No recursion through generic closures — all reference main_cols directly.
            Packing::DWordWL => {
                // 2× Direct
                *acc += &main_cols[start_col][row] * &alpha_powers[alpha_offset];
                *acc += &main_cols[start_col + 1][row] * &alpha_powers[alpha_offset + 1];
                2
            }
            Packing::DWordHHW => {
                // Direct + Word2L
                *acc += &main_cols[start_col][row] * &alpha_powers[alpha_offset];
                let shift_16 = FieldElement::<F>::from(SHIFT_16);
                let w = &main_cols[start_col + 1][row] + &main_cols[start_col + 2][row] * &shift_16;
                *acc += &w * &alpha_powers[alpha_offset + 1];
                2
            }
            Packing::DWordWHH => {
                // Word2L + Direct
                let shift_16 = FieldElement::<F>::from(SHIFT_16);
                let w = &main_cols[start_col][row] + &main_cols[start_col + 1][row] * &shift_16;
                *acc += &w * &alpha_powers[alpha_offset];
                *acc += &main_cols[start_col + 2][row] * &alpha_powers[alpha_offset + 1];
                2
            }
            Packing::DWordHL => {
                // 2× Word2L
                let shift_16 = FieldElement::<F>::from(SHIFT_16);
                let w0 = &main_cols[start_col][row] + &main_cols[start_col + 1][row] * &shift_16;
                *acc += &w0 * &alpha_powers[alpha_offset];
                let w1 =
                    &main_cols[start_col + 2][row] + &main_cols[start_col + 3][row] * &shift_16;
                *acc += &w1 * &alpha_powers[alpha_offset + 1];
                2
            }
            Packing::DWordBL => {
                // 2× Word4L
                let shift_8 = FieldElement::<F>::from(SHIFT_8);
                let shift_16 = FieldElement::<F>::from(SHIFT_16);
                let shift_24 = &shift_8 * &shift_16;
                let w0 = &main_cols[start_col][row]
                    + &main_cols[start_col + 1][row] * &shift_8
                    + &main_cols[start_col + 2][row] * &shift_16
                    + &main_cols[start_col + 3][row] * &shift_24;
                *acc += &w0 * &alpha_powers[alpha_offset];
                let w1 = &main_cols[start_col + 4][row]
                    + &main_cols[start_col + 5][row] * &shift_8
                    + &main_cols[start_col + 6][row] * &shift_16
                    + &main_cols[start_col + 7][row] * &shift_24;
                *acc += &w1 * &alpha_powers[alpha_offset + 1];
                2
            }
            Packing::QuadHL => {
                // 4× Word2L
                let shift_16 = FieldElement::<F>::from(SHIFT_16);
                for i in 0..4 {
                    let c = start_col + i * 2;
                    let w = &main_cols[c][row] + &main_cols[c + 1][row] * &shift_16;
                    *acc += &w * &alpha_powers[alpha_offset + i];
                }
                4
            }
            Packing::QuadWL => {
                // 4× Direct
                for i in 0..4 {
                    *acc += &main_cols[start_col + i][row] * &alpha_powers[alpha_offset + i];
                }
                4
            }
        }
    }

    /// Combines column values into bus elements using powers of 2.
    ///
    /// Primitive packings define the combining formulas.
    /// Compound packings delegate to primitives.
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
            // =================================================================
            // Primitives - define the actual combining formulas
            // =================================================================
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

            // =================================================================
            // Compounds - delegate to primitives
            // (sorted by output count, then input count)
            // =================================================================
            Packing::DWordWL => {
                // 2× Direct
                let mut result = Packing::Direct.combine(&columns[0..1]);
                result.extend(Packing::Direct.combine(&columns[1..2]));
                result
            }

            Packing::DWordHHW => {
                // Direct + Word2L
                let mut result = Packing::Direct.combine(&columns[0..1]);
                result.extend(Packing::Word2L.combine(&columns[1..3]));
                result
            }

            Packing::DWordWHH => {
                // Word2L + Direct
                let mut result = Packing::Word2L.combine(&columns[0..2]);
                result.extend(Packing::Direct.combine(&columns[2..3]));
                result
            }

            Packing::DWordHL => {
                // 2× Word2L
                let mut result = Packing::Word2L.combine(&columns[0..2]);
                result.extend(Packing::Word2L.combine(&columns[2..4]));
                result
            }

            Packing::DWordBL => {
                // 2× Word4L
                let mut result = Packing::Word4L.combine(&columns[0..4]);
                result.extend(Packing::Word4L.combine(&columns[4..8]));
                result
            }

            Packing::QuadHL => {
                // 4× Word2L
                let mut result = Packing::Word2L.combine(&columns[0..2]);
                result.extend(Packing::Word2L.combine(&columns[2..4]));
                result.extend(Packing::Word2L.combine(&columns[4..6]));
                result.extend(Packing::Word2L.combine(&columns[6..8]));
                result
            }

            Packing::QuadWL => {
                // 4× Direct
                let mut result = Packing::Direct.combine(&columns[0..1]);
                result.extend(Packing::Direct.combine(&columns[1..2]));
                result.extend(Packing::Direct.combine(&columns[2..3]));
                result.extend(Packing::Direct.combine(&columns[3..4]));
                result
            }
        }
    }
}

// =============================================================================
// Linear Term and Bus Value
// =============================================================================

/// A term in a linear combination.
///
/// Used to build custom linear combinations of column values and constants.
/// Supports both positive and negative coefficients (i64) for use in
/// Multiplicity::Linear (e.g., μ - read2 - read4 - read8).
#[derive(Debug, Clone)]
pub enum LinearTerm {
    /// coefficient * column_value (coefficient can be negative)
    Column {
        /// The multiplier for the column value (signed to support subtraction)
        coefficient: i64,
        /// The column index to read from
        column: usize,
    },
    /// coefficient * column_value (unsigned, for large field elements like inverses)
    ///
    /// Use this when the coefficient is a large field element (e.g., 2^-32 mod p)
    /// that doesn't fit in i64.
    ColumnUnsigned {
        /// The multiplier as an unsigned value (for large field elements)
        coefficient: u64,
        /// The column index to read from
        column: usize,
    },
    /// A constant value to add (signed to support subtraction)
    Constant(i64),
}

/// A value that contributes to the bus fingerprint.
///
/// Each `BusValue` produces exactly **1 bus element** for the fingerprint.
/// The fingerprint is computed as: `z - (v₀ + α·v₁ + α²·v₂ + ...)`
/// where each `vᵢ` is a bus element from a `BusValue`.
#[derive(Debug, Clone)]
pub enum BusValue {
    /// Columns combined with predefined packing (powers of 2).
    ///
    /// Uses the `Packing` enum's formula to combine consecutive columns.
    /// Example: `Word2L` at column 0 reads columns 0,1 and computes `c₀ + 2¹⁶·c₁`
    Packed {
        /// Starting column index
        start_column: usize,
        /// How to combine the columns
        packing: Packing,
    },

    /// Custom linear combination of columns and/or constants.
    ///
    /// Computes: `a₀·col[i₀] + a₁·col[i₁] + ... + c`
    /// where `aᵢ` are coefficients, `col[iᵢ]` are column values, and `c` is a constant.
    Linear(Vec<LinearTerm>),
}

impl BusValue {
    /// Creates a constant value (no columns).
    ///
    /// Example: `BusValue::constant(0x42)` for a table ID or opcode.
    pub fn constant(value: u64) -> Self {
        BusValue::Linear(vec![LinearTerm::Constant(value as i64)])
    }

    /// Creates a single column value with coefficient 1.
    ///
    /// Example: `BusValue::column(2)` reads column 2 directly.
    pub fn column(col: usize) -> Self {
        BusValue::Linear(vec![LinearTerm::Column {
            coefficient: 1,
            column: col,
        }])
    }

    /// Creates a linear combination from terms.
    ///
    /// Example: `BusValue::linear(vec![...])` with `LinearTerm::Column` and `LinearTerm::Constant`
    /// terms computes something like `3·col[0] + 7·col[1] + 42`.
    pub fn linear(terms: Vec<LinearTerm>) -> Self {
        BusValue::Linear(terms)
    }

    /// Returns the number of bus elements this value produces (always 1).
    pub fn num_bus_elements(&self) -> usize {
        match self {
            BusValue::Packed { packing, .. } => packing.num_bus_elements(),
            BusValue::Linear(_) => 1,
        }
    }

    /// Returns the column indices this value reads from.
    pub fn column_indices(&self) -> Vec<usize> {
        match self {
            BusValue::Packed {
                start_column,
                packing,
            } => (*start_column..*start_column + packing.num_columns()).collect(),
            BusValue::Linear(terms) => terms
                .iter()
                .filter_map(|term| match term {
                    LinearTerm::Column { column, .. } => Some(*column),
                    LinearTerm::ColumnUnsigned { column, .. } => Some(*column),
                    LinearTerm::Constant(_) => None,
                })
                .collect(),
        }
    }

    /// Accumulates the fingerprint contribution of this bus value into `acc`.
    ///
    /// Computes: `acc += Σ element_i * alpha_powers[alpha_offset + i]`
    /// Returns the number of alpha powers consumed.
    pub fn accumulate_fingerprint<F, E>(
        &self,
        main_cols: &[Vec<FieldElement<F>>],
        row: usize,
        alpha_powers: &[FieldElement<E>],
        alpha_offset: usize,
        acc: &mut FieldElement<E>,
    ) -> usize
    where
        F: IsField + IsSubFieldOf<E>,
        E: IsField,
    {
        match self {
            BusValue::Packed {
                start_column,
                packing,
            } => packing.accumulate_fingerprint(
                main_cols,
                row,
                *start_column,
                alpha_powers,
                alpha_offset,
                acc,
            ),
            BusValue::Linear(terms) => {
                let mut result = FieldElement::<F>::zero();
                for term in terms {
                    match term {
                        LinearTerm::Column {
                            coefficient,
                            column,
                        } => {
                            let coeff = FieldElement::<F>::from(*coefficient);
                            result += &main_cols[*column][row] * coeff;
                        }
                        LinearTerm::ColumnUnsigned {
                            coefficient,
                            column,
                        } => {
                            let coeff = FieldElement::<F>::from(*coefficient);
                            result += &main_cols[*column][row] * coeff;
                        }
                        LinearTerm::Constant(value) => {
                            result += FieldElement::<F>::from(*value);
                        }
                    }
                }
                *acc += &result * &alpha_powers[alpha_offset];
                1
            }
        }
    }

    /// Computes the bus element value from column values.
    ///
    /// # Arguments
    /// * `get_column` - Function to get column value by index
    ///
    /// # Returns
    /// Vector of combined bus elements (length = num_bus_elements())
    pub fn combine_from<E: IsField, F: Fn(usize) -> FieldElement<E>>(
        &self,
        get_column: F,
    ) -> Vec<FieldElement<E>> {
        match self {
            BusValue::Packed {
                start_column,
                packing,
            } => {
                let columns: Vec<_> = (*start_column..*start_column + packing.num_columns())
                    .map(&get_column)
                    .collect();
                packing.combine(&columns)
            }
            BusValue::Linear(terms) => {
                let mut result = FieldElement::<E>::zero();
                for term in terms {
                    match term {
                        LinearTerm::Column {
                            coefficient,
                            column,
                        } => {
                            let coeff = FieldElement::<E>::from(*coefficient);
                            result += get_column(*column) * coeff;
                        }
                        LinearTerm::ColumnUnsigned {
                            coefficient,
                            column,
                        } => {
                            // Unsigned coefficient (for large field elements)
                            let coeff = FieldElement::<E>::from(*coefficient);
                            result += get_column(*column) * coeff;
                        }
                        LinearTerm::Constant(value) => {
                            result += FieldElement::<E>::from(*value);
                        }
                    }
                }
                vec![result]
            }
        }
    }
}

// =============================================================================
// AirWithBuses
// =============================================================================

/// Struct representing an AIR with Lookup. Contains own implementation of boundary constraints and auxiliary trace building
pub struct AirWithBuses<
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
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
    /// Commitment to precomputed columns (if this is a preprocessed table)
    preprocessed_commitment: Option<crate::config::Commitment>,
    /// Number of precomputed columns (columns 0..n are precomputed, rest are multiplicities)
    num_precomputed_cols: Option<usize>,
    /// Optional name for debug output (per-table bus sum tracking)
    name: Option<String>,
}

impl<
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync + 'static,
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
            preprocessed_commitment: None,
            num_precomputed_cols: None,
            name: None,
        }
    }

    /// Marks this AIR as a preprocessed table with a hardcoded commitment.
    ///
    /// Preprocessed tables have columns that are fully deterministic and known
    /// to both prover and verifier (e.g., bitwise lookup tables). The verifier
    /// uses the hardcoded commitment instead of trusting the prover.
    ///
    /// # Arguments
    /// * `commitment` - The Merkle root commitment to the precomputed columns
    /// * `num_precomputed_cols` - Number of precomputed columns (0..n are precomputed,
    ///   remaining columns are multiplicities that vary per proof)
    ///
    /// # Example
    /// ```ignore
    /// let air = AirWithBuses::new(num_cols, aux_data, opts, 1, constraints)
    ///     .with_preprocessed(bitwise::preprocessed_commitment(), bitwise::NUM_PRECOMPUTED_COLS);
    /// ```
    pub fn with_preprocessed(
        mut self,
        commitment: crate::config::Commitment,
        num_precomputed_cols: usize,
    ) -> Self {
        self.preprocessed_commitment = Some(commitment);
        self.num_precomputed_cols = Some(num_precomputed_cols);
        self
    }

    /// Set a debug name for this AIR (for per-table bus sum tracking).
    ///
    /// When set, debug output will show bus sums prefixed with this name,
    /// making it easy to identify which table is contributing to bus imbalances.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
}

impl<F, E, B, PI> crate::traits::AIR for AirWithBuses<F, E, B, PI>
where
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
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

    fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("unknown")
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

    fn has_trace_interaction(&self) -> bool {
        !self.auxiliary_trace_build_data.interactions.is_empty()
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

        // Clone main columns once (shared across all interactions)
        let main_segment_cols = trace.columns_main();
        let trace_len = trace.num_rows();
        let table_name = self.name.as_deref().unwrap_or("UNKNOWN");

        // Compute term columns in parallel — even with 1-3 interactions per table,
        // each column computation is heavy enough to benefit from parallelism on
        // high-core-count machines. Internal parallelism within each column is also used.
        #[cfg(feature = "parallel")]
        let interactions_iter = self
            .auxiliary_trace_build_data
            .interactions
            .as_slice()
            .into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let interactions_iter = self.auxiliary_trace_build_data.interactions.iter();

        let term_columns: Vec<Vec<FieldElement<E>>> = interactions_iter
            .map(|interaction| {
                compute_logup_term_column(
                    interaction,
                    &main_segment_cols,
                    trace_len,
                    challenges,
                    table_name,
                )
            })
            .collect();

        // Write term columns to trace
        for (col_idx, col_data) in term_columns.iter().enumerate() {
            for (row, value) in col_data.iter().enumerate() {
                trace.set_aux(row, col_idx, value.clone());
            }
        }

        #[cfg(feature = "debug-checks")]
        let (per_bus_sums, per_bus_sender_sums, per_bus_receiver_sums) =
            compute_debug_bus_sums(&self.auxiliary_trace_build_data.interactions, trace);

        // Build accumulated column from pre-computed term columns
        let acc_col_idx = num_interactions;
        build_accumulated_column_from_terms(acc_col_idx, &term_columns, trace);

        // Collect term column values at row 0 (public inputs for row-0 boundary constraints)
        let initial_terms: Vec<FieldElement<E>> = (0..num_interactions)
            .map(|i| trace.get_aux(0, i).clone())
            .collect();

        // Return BusPublicInputs with initial terms and accumulated column endpoints
        let last_row = trace.num_rows() - 1;
        Some(BusPublicInputs {
            initial_terms,
            final_accumulated: trace.get_aux(last_row, acc_col_idx).clone(),
            #[cfg(feature = "debug-checks")]
            per_bus_sums,
            #[cfg(feature = "debug-checks")]
            per_bus_sender_sums,
            #[cfg(feature = "debug-checks")]
            per_bus_receiver_sums,
            #[cfg(feature = "debug-checks")]
            table_name: self.name.clone().unwrap_or_else(|| "UNKNOWN".to_string()),
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

        if let Some(bus_inputs) = bus_public_inputs {
            let acc_col_idx = self.auxiliary_trace_build_data.interactions.len();

            // One boundary constraint per term column at row 0: term_i(0) = initial_terms[i].
            // This makes each term's initial value a verifier-enforced public input.
            // The verifier rejects proofs where initial_terms.len() != num_interactions
            // before reaching constraint evaluation, so the length of initial_terms is guaranteed to be correct.
            for (i, expected) in bus_inputs.initial_terms.iter().enumerate() {
                boundary_constraints.push(BoundaryConstraint::new_aux(i, 0, expected.clone()));
            }

            // Boundary constraint for the accumulated column at row 0: acc(0) = Σ initial_terms.
            let initial_acc: FieldElement<E> = bus_inputs.initial_terms.iter().cloned().sum();
            boundary_constraints.push(BoundaryConstraint::new_aux(acc_col_idx, 0, initial_acc));

            // Boundary constraint for the accumulated column at last row.
            boundary_constraints.push(BoundaryConstraint::new_aux(
                acc_col_idx,
                trace_length - 1,
                bus_inputs.final_accumulated.clone(),
            ));
        }

        // User-defined boundary constraints
        boundary_constraints.extend(B::boundary_constraints(pub_inputs, rap_challenges));

        BoundaryConstraints::from_constraints(boundary_constraints)
    }

    fn is_preprocessed(&self) -> bool {
        self.preprocessed_commitment.is_some()
    }

    fn num_precomputed_columns(&self) -> usize {
        self.num_precomputed_cols.unwrap_or(0)
    }

    fn precomputed_commitment(&self) -> crate::config::Commitment {
        self.preprocessed_commitment.unwrap_or([0u8; 32])
    }
}

/// Struct representing how each lookup air should build its auxiliary trace
/// Contains a list of all lookup interactions
pub struct AuxiliaryTraceBuildData {
    pub interactions: Vec<BusInteraction>,
}

// =============================================================================
// Multiplicity
// =============================================================================

/// Specifies how to compute the multiplicity for a bus interaction.
///
/// The multiplicity determines how many times each row contributes to the bus.
/// Different use cases require different ways to compute this value.
#[derive(Clone, Debug)]
pub enum Multiplicity {
    /// Constant multiplicity of 1 for all rows.
    /// Use when every row participates exactly once.
    One,

    /// Read multiplicity from a single column (index).
    Column(usize),

    /// Sum of two columns: `col_a + col_b`.
    /// Useful when multiple flags indicate participation.
    Sum(usize, usize),

    /// Negation of a bit column: `1 - col_value`.
    /// The column must contain only 0 or 1.
    /// Useful for "all rows except those marked by this flag".
    Negated(usize),

    /// Arbitrary linear combination of columns and constants.
    /// Supports signed coefficients for subtraction.
    /// Example: `μ - read2 - read4 - read8` can be expressed as:
    /// ```ignore
    /// Multiplicity::Linear(vec![
    ///     LinearTerm::Column { coefficient: 1, column: cols::MU },
    ///     LinearTerm::Column { coefficient: -1, column: cols::READ2 },
    ///     LinearTerm::Column { coefficient: -1, column: cols::READ4 },
    ///     LinearTerm::Column { coefficient: -1, column: cols::READ8 },
    /// ])
    /// ```
    Linear(Vec<LinearTerm>),
}

/// Struct representing a lookup interaction for a given table.
/// Contains the multiplicity and bus values involved in said interaction.
///
/// Values are combined in two stages:
/// 1. **Casting** (powers of 2 or custom linear combination): Combine limbs within each BusValue
/// 2. **Bus fingerprint** (powers of α): Combine all bus elements into one fingerprint
///
/// The `bus_id` distinguishes different buses. Senders and receivers must use
/// the same `bus_id` for their fingerprints to match. Define bus IDs as an enum:
/// ```ignore
/// #[repr(u64)]
/// enum BusId { Add, Mul, Sub }  // auto-increments: 0, 1, 2
///
/// BusInteraction::sender(BusId::Add, Multiplicity::Column(0), Packing::Direct.columns(&[1, 2, 3]))
/// ```
#[derive(Clone)]
pub struct BusInteraction {
    /// Bus identifier. Senders and receivers on the same bus must use the same ID.
    /// Different buses can have different IDs to prevent cross-bus consumption.
    pub bus_id: u64,
    /// How to compute the multiplicity for this interaction.
    /// Determines how many times each row contributes to the bus.
    pub multiplicity: Multiplicity,
    /// Bus values that make up this interaction.
    /// Each BusValue produces one or more bus elements for the fingerprint.
    pub values: Vec<BusValue>,
    /// Whether this side of the interaction is a sender (true) or receiver (false).
    /// Senders contribute positive values to the bus sum, receivers contribute negative.
    /// For bus balance: Σ sender_values - Σ receiver_values = 0
    pub is_sender: bool,
}

impl BusInteraction {
    /// Creates a new table interaction.
    ///
    /// # Arguments
    /// * `bus_id` - Unique identifier for the bus. Can be a raw `u64` or an enum with `Into<u64>`
    /// * `multiplicity` - How to compute the multiplicity for this interaction
    /// * `values` - Typed values that make up this interaction
    /// * `is_sender` - true for sender, false for receiver
    pub fn new(
        bus_id: impl Into<u64>,
        multiplicity: Multiplicity,
        values: Vec<BusValue>,
        is_sender: bool,
    ) -> Self {
        Self {
            bus_id: bus_id.into(),
            multiplicity,
            values,
            is_sender,
        }
    }

    /// Creates a sender interaction.
    ///
    /// # Arguments
    /// * `bus_id` - Unique identifier for the bus
    /// * `multiplicity` - How to compute the multiplicity for this interaction
    /// * `values` - Typed values to send
    pub fn sender(
        bus_id: impl Into<u64>,
        multiplicity: Multiplicity,
        values: Vec<BusValue>,
    ) -> Self {
        Self::new(bus_id, multiplicity, values, true)
    }

    /// Creates a receiver interaction.
    ///
    /// # Arguments
    /// * `bus_id` - Must match the sender's bus_id
    /// * `multiplicity` - How to compute the multiplicity for this interaction
    /// * `values` - Typed values to receive
    pub fn receiver(
        bus_id: impl Into<u64>,
        multiplicity: Multiplicity,
        values: Vec<BusValue>,
    ) -> Self {
        Self::new(bus_id, multiplicity, values, false)
    }

    /// Returns total number of bus elements (for α power computation).
    /// Includes the bus_id as the first element.
    pub fn num_bus_elements(&self) -> usize {
        1 + self
            .values
            .iter()
            .map(|v| v.num_bus_elements())
            .sum::<usize>()
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
    /// Term column values at row 0 (one per interaction).
    /// Used for boundary constraints that enforce term_i(0) = initial_terms[i].
    pub initial_terms: Vec<FieldElement<E>>,
    /// Accumulated column value at last row (total sum of all terms)
    pub final_accumulated: FieldElement<E>,
    /// Per-bus sums for this table (bus_id → sum) - for debug aggregation
    #[cfg(feature = "debug-checks")]
    pub per_bus_sums: HashMap<u64, FieldElement<E>>,
    /// Per-bus sender sums (bus_id → sum) - positive contributions
    #[cfg(feature = "debug-checks")]
    pub per_bus_sender_sums: HashMap<u64, FieldElement<E>>,
    /// Per-bus receiver sums (bus_id → sum) - absolute value (before negation)
    #[cfg(feature = "debug-checks")]
    pub per_bus_receiver_sums: HashMap<u64, FieldElement<E>>,
    /// Table name for debug output
    #[cfg(feature = "debug-checks")]
    pub table_name: String,
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

// =============================================================================
// AVX2 Fingerprint Acceleration (x86_64 only)
// =============================================================================

/// AVX2-accelerated fingerprint computation for Goldilocks cubic extension.
///
/// Processes 4 rows simultaneously using 256-bit SIMD. Each F×E multiply
/// (base field × cubic extension) becomes 3 parallel 4-wide Goldilocks multiplies.
///
/// # Arguments
/// * `bus_element_cols` - Pre-computed bus element columns (column-major, raw u64).
///   Each inner slice has `trace_len` elements.
/// * `alpha_powers_raw` - Alpha powers as raw [u64; 3] (cubic extension components).
/// * `z_raw` - The z challenge as raw [u64; 3].
/// * `trace_len` - Number of rows.
///
/// # Returns
/// Fingerprints as Vec of [u64; 3] (cubic extension, SoA→AoS converted).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_fingerprints_avx2(
    bus_element_cols: &[&[u64]],
    alpha_powers_raw: &[[u64; 3]],
    z_raw: [u64; 3],
    trace_len: usize,
) -> Vec<[u64; 3]> {
    use core::arch::x86_64::*;
    use math::field::fields::fft_friendly::goldilocks_avx2::{add4, mul4, sub4};

    let num_elements = bus_element_cols.len();
    assert_eq!(alpha_powers_raw.len(), num_elements);

    let chunks = trace_len / 4;
    let mut result = Vec::with_capacity(trace_len);

    // SAFETY: caller verified AVX2 is available via is_x86_feature_detected
    unsafe {
        // Broadcast z components
        let z0 = _mm256_set1_epi64x(z_raw[0] as i64);
        let z1 = _mm256_set1_epi64x(z_raw[1] as i64);
        let z2 = _mm256_set1_epi64x(z_raw[2] as i64);

        // Process 4 rows at a time
        for chunk in 0..chunks {
            let base_row = chunk * 4;

            // Accumulators for the linear combination (3 cubic components, 4 rows each)
            let mut acc0 = _mm256_setzero_si256();
            let mut acc1 = _mm256_setzero_si256();
            let mut acc2 = _mm256_setzero_si256();

            for k in 0..num_elements {
                // Load 4 consecutive base-field values from column k
                let vals = _mm256_loadu_si256(
                    bus_element_cols[k].as_ptr().add(base_row) as *const __m256i
                );

                // Broadcast the 3 cubic components of alpha_powers[k]
                let ap0 = _mm256_set1_epi64x(alpha_powers_raw[k][0] as i64);
                let ap1 = _mm256_set1_epi64x(alpha_powers_raw[k][1] as i64);
                let ap2 = _mm256_set1_epi64x(alpha_powers_raw[k][2] as i64);

                // F×E multiply: val * [ap0, ap1, ap2] = [val*ap0, val*ap1, val*ap2]
                // Accumulate into acc
                acc0 = add4(acc0, mul4(vals, ap0));
                acc1 = add4(acc1, mul4(vals, ap1));
                acc2 = add4(acc2, mul4(vals, ap2));
            }

            // fingerprint = z - acc
            let fp0 = sub4(z0, acc0);
            let fp1 = sub4(z1, acc1);
            let fp2 = sub4(z2, acc2);

            // Store results (SoA → AoS)
            let mut out0 = [0u64; 4];
            let mut out1 = [0u64; 4];
            let mut out2 = [0u64; 4];
            _mm256_storeu_si256(out0.as_mut_ptr() as *mut __m256i, fp0);
            _mm256_storeu_si256(out1.as_mut_ptr() as *mut __m256i, fp1);
            _mm256_storeu_si256(out2.as_mut_ptr() as *mut __m256i, fp2);

            for i in 0..4 {
                result.push([out0[i], out1[i], out2[i]]);
            }
        }
    }

    // Scalar tail
    for row in (chunks * 4)..trace_len {
        let mut acc = [0u64; 3];
        for k in 0..num_elements {
            let val = bus_element_cols[k][row];
            for (c, acc_c) in acc.iter_mut().enumerate() {
                let product = (val as u128) * (alpha_powers_raw[k][c] as u128);
                *acc_c = goldilocks_add(*acc_c, goldilocks_reduce128(product));
            }
        }
        result.push([
            goldilocks_sub(z_raw[0], acc[0]),
            goldilocks_sub(z_raw[1], acc[1]),
            goldilocks_sub(z_raw[2], acc[2]),
        ]);
    }

    result
}

/// Scalar Goldilocks add (used in tail processing).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn goldilocks_add(a: u64, b: u64) -> u64 {
    const EPSILON: u64 = 0xFFFF_FFFF;
    let (sum, over) = a.overflowing_add(b);
    let (sum, over2) = sum.overflowing_add((over as u64) * EPSILON);
    if over2 {
        sum.wrapping_add(EPSILON)
    } else {
        sum
    }
}

/// Scalar Goldilocks sub (used in tail processing).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn goldilocks_sub(a: u64, b: u64) -> u64 {
    const EPSILON: u64 = 0xFFFF_FFFF;
    let (diff, under) = a.overflowing_sub(b);
    let (diff, under2) = diff.overflowing_sub((under as u64) * EPSILON);
    if under2 {
        diff.wrapping_sub(EPSILON)
    } else {
        diff
    }
}

/// Scalar Goldilocks reduce128 (used in tail processing).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn goldilocks_reduce128(x: u128) -> u64 {
    const EPSILON: u64 = 0xFFFF_FFFF;
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;
    let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };
    let t1 = (x_hi_lo << 32).wrapping_sub(x_hi_lo);
    let (result, carry) = t0.overflowing_add(t1);
    if carry {
        result.wrapping_add(EPSILON)
    } else {
        result
    }
}

/// Pre-compute bus element columns as raw u64 values (column-major).
///
/// Flattens each `BusInteraction`'s bus elements into a Vec of u64 columns, handling
/// packing/combining in the base field. The bus_id contributes the first element.
#[cfg(target_arch = "x86_64")]
fn precompute_bus_element_columns(
    table_interaction: &BusInteraction,
    main_segment_cols: &[Vec<u64>],
    trace_len: usize,
) -> Vec<Vec<u64>> {
    let num_elements = table_interaction.num_bus_elements();
    let mut columns: Vec<Vec<u64>> = Vec::with_capacity(num_elements);

    // First element: bus_id (constant column)
    columns.push(vec![table_interaction.bus_id; trace_len]);

    // Remaining elements from bus values
    for bv in &table_interaction.values {
        match bv {
            BusValue::Packed {
                start_column,
                packing,
            } => {
                precompute_packing_columns(
                    *packing,
                    *start_column,
                    main_segment_cols,
                    trace_len,
                    &mut columns,
                );
            }
            BusValue::Linear(terms) => {
                let mut col = Vec::with_capacity(trace_len);
                for row in 0..trace_len {
                    let mut val = 0u64;
                    for term in terms {
                        match *term {
                            LinearTerm::Column {
                                coefficient,
                                column,
                            } => {
                                let coeff = if coefficient >= 0 {
                                    coefficient as u64
                                } else {
                                    math::field::fields::fft_friendly::u64_goldilocks::GOLDILOCKS_PRIME
                                        .wrapping_sub((-coefficient) as u64)
                                };
                                let product =
                                    (main_segment_cols[column][row] as u128) * (coeff as u128);
                                val = goldilocks_add(val, goldilocks_reduce128(product));
                            }
                            LinearTerm::ColumnUnsigned {
                                coefficient,
                                column,
                            } => {
                                let product = (main_segment_cols[column][row] as u128)
                                    * (coefficient as u128);
                                val = goldilocks_add(val, goldilocks_reduce128(product));
                            }
                            LinearTerm::Constant(value) => {
                                let c = if value >= 0 {
                                    value as u64
                                } else {
                                    math::field::fields::fft_friendly::u64_goldilocks::GOLDILOCKS_PRIME
                                        .wrapping_sub((-value) as u64)
                                };
                                val = goldilocks_add(val, c);
                            }
                        }
                    }
                    col.push(val);
                }
                columns.push(col);
            }
        }
    }

    assert_eq!(columns.len(), num_elements);
    columns
}

/// Pre-compute columns for a specific packing type.
#[cfg(target_arch = "x86_64")]
fn precompute_packing_columns(
    packing: Packing,
    start_col: usize,
    main_cols: &[Vec<u64>],
    trace_len: usize,
    out: &mut Vec<Vec<u64>>,
) {
    match packing {
        Packing::Direct => {
            out.push(main_cols[start_col].clone());
        }
        Packing::Word2L => {
            let mut col = Vec::with_capacity(trace_len);
            for row in 0..trace_len {
                let combined = goldilocks_add(
                    main_cols[start_col][row],
                    goldilocks_reduce128(main_cols[start_col + 1][row] as u128 * SHIFT_16 as u128),
                );
                col.push(combined);
            }
            out.push(col);
        }
        Packing::Word4L => {
            let mut col = Vec::with_capacity(trace_len);
            for row in 0..trace_len {
                let mut combined = main_cols[start_col][row];
                combined = goldilocks_add(
                    combined,
                    goldilocks_reduce128(main_cols[start_col + 1][row] as u128 * SHIFT_8 as u128),
                );
                combined = goldilocks_add(
                    combined,
                    goldilocks_reduce128(main_cols[start_col + 2][row] as u128 * SHIFT_16 as u128),
                );
                let shift_24 = SHIFT_8 as u128 * SHIFT_16 as u128;
                combined = goldilocks_add(
                    combined,
                    goldilocks_reduce128(main_cols[start_col + 3][row] as u128 * shift_24),
                );
                col.push(combined);
            }
            out.push(col);
        }
        // Compound packings: decompose into primitives
        Packing::DWordWL => {
            precompute_packing_columns(Packing::Direct, start_col, main_cols, trace_len, out);
            precompute_packing_columns(Packing::Direct, start_col + 1, main_cols, trace_len, out);
        }
        Packing::DWordHHW => {
            precompute_packing_columns(Packing::Direct, start_col, main_cols, trace_len, out);
            precompute_packing_columns(Packing::Word2L, start_col + 1, main_cols, trace_len, out);
        }
        Packing::DWordWHH => {
            precompute_packing_columns(Packing::Word2L, start_col, main_cols, trace_len, out);
            precompute_packing_columns(Packing::Direct, start_col + 2, main_cols, trace_len, out);
        }
        Packing::DWordHL => {
            precompute_packing_columns(Packing::Word2L, start_col, main_cols, trace_len, out);
            precompute_packing_columns(Packing::Word2L, start_col + 2, main_cols, trace_len, out);
        }
        Packing::DWordBL => {
            precompute_packing_columns(Packing::Word4L, start_col, main_cols, trace_len, out);
            precompute_packing_columns(Packing::Word4L, start_col + 4, main_cols, trace_len, out);
        }
        Packing::QuadHL => {
            for i in 0..4 {
                precompute_packing_columns(
                    Packing::Word2L,
                    start_col + i * 2,
                    main_cols,
                    trace_len,
                    out,
                );
            }
        }
        Packing::QuadWL => {
            for i in 0..4 {
                precompute_packing_columns(
                    Packing::Direct,
                    start_col + i,
                    main_cols,
                    trace_len,
                    out,
                );
            }
        }
    }
}

/// Computes a term column for a table interaction without writing to the trace.
///
/// Each row contains the LogUp quotient: `term[i] = sign * multiplicity[i] / fingerprint[i]`
///
/// This is a pure function that takes shared main columns and returns the computed column,
/// enabling parallel computation across interactions within a table.
#[allow(clippy::needless_range_loop)]
fn compute_logup_term_column<F, E>(
    table_interaction: &BusInteraction,
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
    #[cfg_attr(not(feature = "debug-checks"), allow(unused))] _table_name: &str,
) -> Vec<FieldElement<E>>
where
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
    E: IsField + Send + Sync,
{
    // Handle multiplicity column(s)
    let multiplicities_owned: Vec<FieldElement<F>>;
    let multiplicities: &[FieldElement<F>] = match table_interaction.multiplicity {
        Multiplicity::One => {
            multiplicities_owned = vec![FieldElement::one(); trace_len];
            &multiplicities_owned
        }
        Multiplicity::Column(col) => &main_segment_cols[col],
        Multiplicity::Sum(col_a, col_b) => {
            multiplicities_owned = main_segment_cols[col_a]
                .iter()
                .zip(main_segment_cols[col_b].iter())
                .map(|(a, b)| a + b)
                .collect();
            &multiplicities_owned
        }
        Multiplicity::Negated(col) => {
            multiplicities_owned = main_segment_cols[col]
                .iter()
                .map(|elem| FieldElement::<F>::one() - elem)
                .collect();
            &multiplicities_owned
        }
        Multiplicity::Linear(ref terms) => {
            multiplicities_owned = (0..trace_len)
                .map(|row| {
                    let mut result = FieldElement::<F>::zero();
                    for term in terms {
                        match *term {
                            LinearTerm::Column {
                                coefficient,
                                column,
                            } => {
                                let coeff = FieldElement::<F>::from(coefficient);
                                result += &main_segment_cols[column][row] * coeff;
                            }
                            LinearTerm::ColumnUnsigned {
                                coefficient,
                                column,
                            } => {
                                let coeff = FieldElement::<F>::from(coefficient);
                                result += &main_segment_cols[column][row] * coeff;
                            }
                            LinearTerm::Constant(value) => {
                                result += FieldElement::<F>::from(value);
                            }
                        }
                    }
                    result
                })
                .collect();
            &multiplicities_owned
        }
    };

    // LogUp challenges (must be shared across all tables for bus to balance)
    let z = &challenges[LOGUP_CHALLENGE_Z];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];

    // Precompute powers of alpha for all bus elements (using incremental multiplication)
    let num_bus_elements = table_interaction.num_bus_elements();
    let alpha_powers = compute_alpha_powers(alpha, num_bus_elements);

    // Batch inversion: collect all fingerprints, invert once, then multiply back.
    // Compute fingerprint = z - (bus_id*α^0 + v0*α^1 + v1*α^2 + ...) using
    // base-field × extension-field multiplication (F×E→E) to avoid to_extension().
    //
    // AVX2 fast path: when on x86_64 with AVX2 support and using Goldilocks + cubic
    // extension, process 4 rows simultaneously with SIMD.
    #[cfg(target_arch = "x86_64")]
    let avx2_fingerprints = 'avx2: {
        use math::field::fields::fft_friendly::extensions_goldilocks::Degree3GoldilocksExtensionField;
        use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
        type FpE = FieldElement<GoldilocksField>;
        type Fp3E = FieldElement<Degree3GoldilocksExtensionField>;

        // Guard: verify concrete types via size_of. In this codebase F is always
        // GoldilocksField (BaseType = u64, 8 bytes) and E is always
        // Degree3GoldilocksExtensionField (BaseType = [FpE; 3], 24 bytes).
        // The AVX2 kernels use Goldilocks-specific reduction constants, so applying
        // them to any other field would produce incorrect results immediately.
        if std::mem::size_of::<FieldElement<F>>() != std::mem::size_of::<FpE>()
            || std::mem::size_of::<FieldElement<E>>() != std::mem::size_of::<Fp3E>()
            || !is_x86_feature_detected!("avx2")
            || trace_len < 4
        {
            break 'avx2 None;
        }

        // SAFETY: FieldElement<F> is #[repr(transparent)] over u64 (verified by the
        // size check above), so &[Vec<FieldElement<F>>] and &[Vec<u64>] have identical
        // layout. This is a zero-cost reinterpret cast — no allocation or copy.
        let main_cols_raw: &[Vec<u64>] =
            unsafe { &*(main_segment_cols as *const [Vec<FieldElement<F>>] as *const [Vec<u64>]) };

        // Pre-compute bus element columns as raw u64
        let bus_element_cols =
            precompute_bus_element_columns(table_interaction, &main_cols_raw, trace_len);
        let col_refs: Vec<&[u64]> = bus_element_cols.iter().map(|c| c.as_slice()).collect();

        // Extract alpha_powers as raw [u64; 3]
        let alpha_powers_raw: Vec<[u64; 3]> = alpha_powers
            .iter()
            .map(|ap| unsafe { std::ptr::read(ap as *const FieldElement<E> as *const [u64; 3]) })
            .collect();

        // Extract z as raw [u64; 3]
        let z_raw: [u64; 3] =
            unsafe { std::ptr::read(z as *const FieldElement<E> as *const [u64; 3]) };

        // Run AVX2 kernel
        let raw_fingerprints =
            unsafe { compute_fingerprints_avx2(&col_refs, &alpha_powers_raw, z_raw, trace_len) };

        // Convert raw [u64; 3] back to FieldElement<E>
        let fingerprints: Vec<FieldElement<E>> = raw_fingerprints
            .into_iter()
            .map(|raw| unsafe { std::ptr::read(&raw as *const [u64; 3] as *const FieldElement<E>) })
            .collect();

        Some(fingerprints)
    };

    #[cfg(not(target_arch = "x86_64"))]
    let avx2_fingerprints: Option<Vec<FieldElement<E>>> = None;

    let mut fingerprints = if let Some(fp) = avx2_fingerprints {
        use std::sync::Once;
        static PRINT: Once = Once::new();
        PRINT.call_once(|| println!("[BENCH] LogUp fingerprints: AVX2"));
        fp
    } else {
        use std::sync::Once;
        static PRINT: Once = Once::new();
        PRINT.call_once(|| println!("[BENCH] LogUp fingerprints: scalar"));
        // Scalar fallback: accumulate the linear combination directly
        // into the fingerprint without collecting bus elements into intermediate Vecs.
        let bus_id_f = FieldElement::<F>::from(table_interaction.bus_id);
        let mut fingerprints: Vec<FieldElement<E>> = Vec::with_capacity(trace_len);
        for row in 0..trace_len {
            let mut linear_combination = &bus_id_f * &alpha_powers[0];
            let mut alpha_offset = 1;
            for bv in &table_interaction.values {
                let consumed = bv.accumulate_fingerprint(
                    main_segment_cols,
                    row,
                    &alpha_powers,
                    alpha_offset,
                    &mut linear_combination,
                );
                alpha_offset += consumed;
            }
            fingerprints.push(z - &linear_combination);
        }
        fingerprints
    };

    #[cfg(feature = "debug-checks")]
    {
        let bus_id_f = FieldElement::<F>::from(table_interaction.bus_id);
        for row in 0..trace_len {
            let mut base_elements: Vec<FieldElement<F>> = vec![bus_id_f.clone()];
            base_elements.extend(
                table_interaction
                    .values
                    .iter()
                    .flat_map(|bv| bv.combine_from(|col| main_segment_cols[col][row].clone())),
            );
            crate::bus_debug::log_interaction(
                _table_name,
                row,
                table_interaction.bus_id,
                table_interaction.is_sender,
                &multiplicities[row].canonical(),
                &base_elements,
                &fingerprints[row],
            );
        }
    }

    FieldElement::inplace_batch_inverse(&mut fingerprints)
        .expect("fingerprint is zero - probability of sampling zero is negligible");

    // Compute terms: term[i] = sign * multiplicity[i] * fingerprint_inv[i]
    // Restructured to avoid E×E multiply: the original `m * sign * fp_inv` performs
    // F×E (3 base muls) then E×E (6 base muls). Instead, do F×E + conditional negate
    // (3 base muls + ~free), or skip arithmetic entirely when multiplicity is 1.
    let is_sender = table_interaction.is_sender;
    match (&table_interaction.multiplicity, is_sender) {
        (Multiplicity::One, true) => {
            // term = fp_inv, no arithmetic needed
            fingerprints
        }
        (Multiplicity::One, false) => {
            for fp in fingerprints.iter_mut() {
                *fp = -&*fp;
            }
            fingerprints
        }
        (_, true) => {
            for (fp, m) in fingerprints.iter_mut().zip(multiplicities.iter()) {
                *fp = m * &*fp;
            }
            fingerprints
        }
        (_, false) => {
            for (fp, m) in fingerprints.iter_mut().zip(multiplicities.iter()) {
                *fp = -(m * &*fp);
            }
            fingerprints
        }
    }
}

/// Builds the accumulated column from pre-computed term columns.
///
/// acc[0] = sum of all term columns at row 0
/// acc[i] = acc[i-1] + sum of all term columns at row i
///
/// Takes term columns directly to avoid row-major trace access.
fn build_accumulated_column_from_terms<F, E>(
    acc_column_idx: usize,
    term_columns: &[Vec<FieldElement<E>>],
    trace: &mut TraceTable<F, E>,
) where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    if term_columns.is_empty() {
        return;
    }
    let trace_len = term_columns[0].len();
    let mut accumulated = FieldElement::<E>::zero();

    for row in 0..trace_len {
        let mut row_sum = FieldElement::<E>::zero();
        for col in term_columns {
            row_sum = row_sum + &col[row];
        }
        accumulated += row_sum;
        trace.set_aux(row, acc_column_idx, accumulated.clone());
    }
}

/// Sum aux term columns by bus_id to produce per-bus totals for the debug report.
#[cfg(feature = "debug-checks")]
#[allow(clippy::type_complexity)]
fn compute_debug_bus_sums<F, E>(
    interactions: &[BusInteraction],
    trace: &TraceTable<F, E>,
) -> (
    HashMap<u64, FieldElement<E>>,
    HashMap<u64, FieldElement<E>>,
    HashMap<u64, FieldElement<E>>,
)
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    let mut bus_sums: HashMap<u64, FieldElement<E>> = HashMap::new();
    let mut sender_sums: HashMap<u64, FieldElement<E>> = HashMap::new();
    let mut receiver_sums: HashMap<u64, FieldElement<E>> = HashMap::new();

    for (i, interaction) in interactions.iter().enumerate() {
        let mut col_sum = FieldElement::<E>::zero();
        for row in 0..trace.num_rows() {
            col_sum = col_sum + trace.get_aux(row, i);
        }
        *bus_sums
            .entry(interaction.bus_id)
            .or_insert(FieldElement::zero()) += col_sum.clone();

        if interaction.is_sender {
            *sender_sums
                .entry(interaction.bus_id)
                .or_insert(FieldElement::zero()) += col_sum;
        } else {
            let entry = receiver_sums
                .entry(interaction.bus_id)
                .or_insert(FieldElement::zero());
            *entry = entry.clone() - col_sum;
        }
    }
    (bus_sums, sender_sums, receiver_sums)
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
    pub fn new(interaction: BusInteraction, term_column_idx: usize, constraint_idx: usize) -> Self {
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
        fn evaluate_term_constraint<A: IsSubFieldOf<B>, B: IsField>(
            step: &TableView<A, B>,
            term_column_idx: usize,
            interaction: &BusInteraction,
            rap_challenges: &&[FieldElement<B>],
            alpha_powers: &[FieldElement<B>],
        ) -> FieldElement<B> {
            // Term column value
            let term = step.get_aux_evaluation_element(0, term_column_idx);

            let z = &rap_challenges[LOGUP_CHALLENGE_Z];

            // Compute multiplicity based on the Multiplicity variant
            let multiplicity: FieldElement<A> = match &interaction.multiplicity {
                Multiplicity::One => FieldElement::<A>::one(),
                Multiplicity::Column(col) => step.get_main_evaluation_element(0, *col).clone(),
                Multiplicity::Sum(col_a, col_b) => {
                    step.get_main_evaluation_element(0, *col_a)
                        + step.get_main_evaluation_element(0, *col_b)
                }
                Multiplicity::Negated(col) => {
                    FieldElement::<A>::one() - step.get_main_evaluation_element(0, *col)
                }
                Multiplicity::Linear(terms) => {
                    let mut result = FieldElement::<A>::zero();
                    for term in terms {
                        match term {
                            LinearTerm::Column {
                                coefficient,
                                column,
                            } => {
                                let coeff = FieldElement::<A>::from(*coefficient);
                                result += step.get_main_evaluation_element(0, *column) * coeff;
                            }
                            LinearTerm::ColumnUnsigned {
                                coefficient,
                                column,
                            } => {
                                let coeff = FieldElement::<A>::from(*coefficient);
                                result += step.get_main_evaluation_element(0, *column) * coeff;
                            }
                            LinearTerm::Constant(value) => {
                                result += FieldElement::<A>::from(*value);
                            }
                        }
                    }
                    result
                }
            };

            // Compute fingerprint using pre-computed alpha powers (indexed access).
            // fingerprint = z - (bus_id*α^0 + v[0]*α^1 + v[1]*α^2 + ...)
            // alpha_powers[0] = 1, alpha_powers[1] = α, alpha_powers[2] = α², ...
            let bus_id_f = FieldElement::<A>::from(interaction.bus_id);
            let mut linear_combination = &bus_id_f * &alpha_powers[0];
            #[allow(unused_assignments)]
            let mut alpha_idx: usize = 1;
            for bv in &interaction.values {
                match bv {
                    BusValue::Packed {
                        start_column,
                        packing,
                    } => {
                        match packing {
                            Packing::Direct => {
                                linear_combination += step
                                    .get_main_evaluation_element(0, *start_column)
                                    * &alpha_powers[alpha_idx];
                                alpha_idx += 1;
                            }
                            Packing::Word2L => {
                                let shift_16 = FieldElement::<A>::from(SHIFT_16);
                                let combined = step.get_main_evaluation_element(0, *start_column)
                                    + step.get_main_evaluation_element(0, *start_column + 1)
                                        * &shift_16;
                                linear_combination += &combined * &alpha_powers[alpha_idx];
                                alpha_idx += 1;
                            }
                            Packing::Word4L => {
                                let shift_8 = FieldElement::<A>::from(SHIFT_8);
                                let shift_16 = FieldElement::<A>::from(SHIFT_16);
                                let shift_24 = &shift_8 * &shift_16;
                                let combined = step.get_main_evaluation_element(0, *start_column)
                                    + step.get_main_evaluation_element(0, *start_column + 1)
                                        * &shift_8
                                    + step.get_main_evaluation_element(0, *start_column + 2)
                                        * &shift_16
                                    + step.get_main_evaluation_element(0, *start_column + 3)
                                        * &shift_24;
                                linear_combination += &combined * &alpha_powers[alpha_idx];
                                alpha_idx += 1;
                            }
                            Packing::DWordWL => {
                                // 2× Direct
                                linear_combination += step
                                    .get_main_evaluation_element(0, *start_column)
                                    * &alpha_powers[alpha_idx];
                                linear_combination += step
                                    .get_main_evaluation_element(0, *start_column + 1)
                                    * &alpha_powers[alpha_idx + 1];
                                alpha_idx += 2;
                            }
                            Packing::DWordHL => {
                                // 2× Word2L
                                let shift_16 = FieldElement::<A>::from(SHIFT_16);
                                let w0 = step.get_main_evaluation_element(0, *start_column)
                                    + step.get_main_evaluation_element(0, *start_column + 1)
                                        * &shift_16;
                                linear_combination += &w0 * &alpha_powers[alpha_idx];
                                let w1 = step.get_main_evaluation_element(0, *start_column + 2)
                                    + step.get_main_evaluation_element(0, *start_column + 3)
                                        * &shift_16;
                                linear_combination += &w1 * &alpha_powers[alpha_idx + 1];
                                alpha_idx += 2;
                            }
                            Packing::DWordBL => {
                                // 2× Word4L
                                let shift_8 = FieldElement::<A>::from(SHIFT_8);
                                let shift_16 = FieldElement::<A>::from(SHIFT_16);
                                let shift_24 = &shift_8 * &shift_16;
                                let w0 = step.get_main_evaluation_element(0, *start_column)
                                    + step.get_main_evaluation_element(0, *start_column + 1)
                                        * &shift_8
                                    + step.get_main_evaluation_element(0, *start_column + 2)
                                        * &shift_16
                                    + step.get_main_evaluation_element(0, *start_column + 3)
                                        * &shift_24;
                                linear_combination += &w0 * &alpha_powers[alpha_idx];
                                let w1 = step.get_main_evaluation_element(0, *start_column + 4)
                                    + step.get_main_evaluation_element(0, *start_column + 5)
                                        * &shift_8
                                    + step.get_main_evaluation_element(0, *start_column + 6)
                                        * &shift_16
                                    + step.get_main_evaluation_element(0, *start_column + 7)
                                        * &shift_24;
                                linear_combination += &w1 * &alpha_powers[alpha_idx + 1];
                                alpha_idx += 2;
                            }
                            Packing::DWordHHW => {
                                // Direct + Word2L
                                linear_combination += step
                                    .get_main_evaluation_element(0, *start_column)
                                    * &alpha_powers[alpha_idx];
                                let shift_16 = FieldElement::<A>::from(SHIFT_16);
                                let w = step.get_main_evaluation_element(0, *start_column + 1)
                                    + step.get_main_evaluation_element(0, *start_column + 2)
                                        * &shift_16;
                                linear_combination += &w * &alpha_powers[alpha_idx + 1];
                                alpha_idx += 2;
                            }
                            Packing::DWordWHH => {
                                // Word2L + Direct
                                let shift_16 = FieldElement::<A>::from(SHIFT_16);
                                let w = step.get_main_evaluation_element(0, *start_column)
                                    + step.get_main_evaluation_element(0, *start_column + 1)
                                        * &shift_16;
                                linear_combination += &w * &alpha_powers[alpha_idx];
                                linear_combination += step
                                    .get_main_evaluation_element(0, *start_column + 2)
                                    * &alpha_powers[alpha_idx + 1];
                                alpha_idx += 2;
                            }
                            Packing::QuadHL => {
                                // 4× Word2L
                                let shift_16 = FieldElement::<A>::from(SHIFT_16);
                                for i in 0..4 {
                                    let c = *start_column + i * 2;
                                    let w = step.get_main_evaluation_element(0, c)
                                        + step.get_main_evaluation_element(0, c + 1) * &shift_16;
                                    linear_combination += &w * &alpha_powers[alpha_idx + i];
                                }
                                alpha_idx += 4;
                            }
                            Packing::QuadWL => {
                                // 4× Direct
                                for i in 0..4 {
                                    linear_combination += step
                                        .get_main_evaluation_element(0, *start_column + i)
                                        * &alpha_powers[alpha_idx + i];
                                }
                                alpha_idx += 4;
                            }
                        }
                    }
                    BusValue::Linear(terms) => {
                        let mut result = FieldElement::<A>::zero();
                        for term in terms {
                            match term {
                                LinearTerm::Column {
                                    coefficient,
                                    column,
                                } => {
                                    let coeff = FieldElement::<A>::from(*coefficient);
                                    result += step.get_main_evaluation_element(0, *column) * coeff;
                                }
                                LinearTerm::ColumnUnsigned {
                                    coefficient,
                                    column,
                                } => {
                                    let coeff = FieldElement::<A>::from(*coefficient);
                                    result += step.get_main_evaluation_element(0, *column) * coeff;
                                }
                                LinearTerm::Constant(value) => {
                                    result += FieldElement::<A>::from(*value);
                                }
                            }
                        }
                        linear_combination += &result * &alpha_powers[alpha_idx];
                        alpha_idx += 1;
                    }
                }
            }
            let fingerprint = z - &linear_combination;

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
                logup_alpha_powers,
                ..
            } => evaluate_term_constraint(
                frame.get_evaluation_step(0),
                self.term_column_idx,
                &self.interaction,
                rap_challenges,
                logup_alpha_powers,
            ),
            TransitionEvaluationContext::Verifier {
                frame,
                rap_challenges,
                logup_alpha_powers,
                ..
            } => evaluate_term_constraint(
                frame.get_evaluation_step(0),
                self.term_column_idx,
                &self.interaction,
                rap_challenges,
                logup_alpha_powers,
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
        fn evaluate_accumulated_constraint<A: IsSubFieldOf<B>, B: IsField>(
            first_step: &TableView<A, B>,
            second_step: &TableView<A, B>,
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

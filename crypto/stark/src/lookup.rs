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

/// Split N interactions into committed batched pairs and absorbed remainder.
///
/// Returns `(num_committed_pairs, absorbed_count)` where:
/// - Committed pairs get dedicated auxiliary term columns (2 interactions per column)
/// - Absorbed interactions (1 or 2) are folded into the accumulated constraint
fn split_interactions(num_interactions: usize) -> (usize, usize) {
    if num_interactions <= 2 {
        (0, num_interactions)
    } else if num_interactions % 2 == 1 {
        ((num_interactions - 1) / 2, 1)
    } else {
        ((num_interactions - 2) / 2, 2)
    }
}

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
    /// Maximum number of bus elements across all interactions.
    /// Used to compute the correct number of alpha powers.
    max_bus_elements: usize,
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
    /// Auxiliary column layout (with interaction batching + absorption):
    /// - Columns 0..num_committed_pairs-1: Committed term columns (batched pairs)
    /// - Last column: Accumulated column (running sum + 1-2 absorbed interactions)
    ///
    /// The last 1-2 interactions are "absorbed" into the accumulated constraint
    /// by clearing denominators, eliminating one committed term column per table.
    ///
    /// Total aux columns = ⌈N/2⌉ where N is the number of interactions.
    pub fn new(
        num_main_columns: usize,
        auxiliary_trace_build_data: AuxiliaryTraceBuildData,
        proof_options: &ProofOptions,
        step_size: usize,
        mut transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    ) -> Self {
        let num_interactions = auxiliary_trace_build_data.interactions.len();

        // Split interactions: committed pairs get term columns, last 1-2 are absorbed
        let (num_committed_pairs, absorbed_count) = split_interactions(num_interactions);
        let absorbed =
            auxiliary_trace_build_data.interactions[num_interactions - absorbed_count..].to_vec();

        // Create batched term constraints for committed pairs only
        for pair_idx in 0..num_committed_pairs {
            let constraint = LookupBatchedTermConstraint::new(
                auxiliary_trace_build_data.interactions[pair_idx * 2].clone(),
                auxiliary_trace_build_data.interactions[pair_idx * 2 + 1].clone(),
                pair_idx,
                transition_constraints.len(),
            );
            transition_constraints.push(Box::new(constraint));
        }

        let num_term_columns = num_committed_pairs;

        // Add the accumulated constraint with absorbed interactions
        if num_interactions > 0 {
            let accumulated_constraint = LookupAccumulatedConstraint::new(
                transition_constraints.len(),
                num_term_columns,
                absorbed,
            );
            transition_constraints.push(Box::new(accumulated_constraint));
        }

        // Layout: num_committed_pairs term columns + 1 accumulated = ⌈N/2⌉
        let num_aux_columns = if num_interactions > 0 {
            num_term_columns + 1
        } else {
            0
        };
        let trace_layout = (num_main_columns, num_aux_columns);

        // Compute max bus elements across all interactions for alpha power count
        let max_bus_elements = auxiliary_trace_build_data
            .interactions
            .iter()
            .map(|i| i.num_bus_elements())
            .max()
            .unwrap_or(0);

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
            max_bus_elements,
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

    fn max_bus_elements(&self) -> usize {
        self.max_bus_elements
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        let max_degree = self
            .transition_constraints
            .iter()
            .map(|c| c.degree())
            .max()
            .unwrap_or(1);
        trace_length * max_degree
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

        // Split interactions: committed pairs get term columns, last 1-2 are absorbed (virtual)
        let (num_committed_pairs, absorbed_count) = split_interactions(num_interactions);

        // Compute committed term columns in parallel (batched pairs only)
        #[cfg(feature = "parallel")]
        let committed_columns: Vec<Vec<FieldElement<E>>> = (0..num_committed_pairs)
            .into_par_iter()
            .map(|i| {
                compute_logup_batched_term_column(
                    &self.auxiliary_trace_build_data.interactions[i * 2],
                    &self.auxiliary_trace_build_data.interactions[i * 2 + 1],
                    &main_segment_cols,
                    trace_len,
                    challenges,
                    table_name,
                )
            })
            .collect();
        #[cfg(not(feature = "parallel"))]
        let committed_columns: Vec<Vec<FieldElement<E>>> = (0..num_committed_pairs)
            .map(|i| {
                compute_logup_batched_term_column(
                    &self.auxiliary_trace_build_data.interactions[i * 2],
                    &self.auxiliary_trace_build_data.interactions[i * 2 + 1],
                    &main_segment_cols,
                    trace_len,
                    challenges,
                    table_name,
                )
            })
            .collect();

        // Compute virtual column for absorbed interactions (NOT written to trace)
        let virtual_column = if absorbed_count == 2 {
            compute_logup_batched_term_column(
                &self.auxiliary_trace_build_data.interactions[num_interactions - 2],
                &self.auxiliary_trace_build_data.interactions[num_interactions - 1],
                &main_segment_cols,
                trace_len,
                challenges,
                table_name,
            )
        } else {
            compute_logup_term_column(
                &self.auxiliary_trace_build_data.interactions[num_interactions - 1],
                &main_segment_cols,
                trace_len,
                challenges,
                table_name,
            )
        };

        // Write only committed columns to trace
        for (col_idx, col_data) in committed_columns.iter().enumerate() {
            for (row, value) in col_data.iter().enumerate() {
                trace.set_aux(row, col_idx, value.clone());
            }
        }

        #[cfg(feature = "debug-checks")]
        let (per_bus_sums, per_bus_sender_sums, per_bus_receiver_sums) =
            compute_debug_bus_sums_batched(
                &self.auxiliary_trace_build_data.interactions,
                &main_segment_cols,
                trace_len,
                challenges,
                table_name,
            );

        // Build accumulated from all columns (committed + virtual)
        let mut all_columns = committed_columns;
        all_columns.push(virtual_column);
        let acc_col_idx = num_committed_pairs; // accumulated column in trace follows committed columns
        let table_contribution =
            build_accumulated_column_from_terms(acc_col_idx, &all_columns, trace);

        Some(BusPublicInputs {
            table_contribution,
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
        _bus_public_inputs: Option<&BusPublicInputs<E>>,
        trace_length: usize,
    ) -> BoundaryConstraints<E> {
        let mut boundary_constraints = B::boundary_constraints(pub_inputs, rap_challenges);

        // Pin acc[N-1] = 0 to remove the constant-shift degree of freedom
        // in the circular transition constraint.
        if !self.auxiliary_trace_build_data.interactions.is_empty() {
            let acc_col_idx = self.trace_layout.1 - 1; // last aux column = accumulated
            boundary_constraints.push(BoundaryConstraint::new_aux(
                acc_col_idx,
                trace_length - 1,
                FieldElement::zero(),
            ));
        }

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

/// Public inputs for a table's LogUp accumulated column.
///
/// Each table has exactly one BusPublicInputs, representing the total
/// contribution of all its LogUp terms (L = Σ all terms across all rows).
/// The sign (sender vs receiver) is already baked into the values,
/// so the bus balance check is: Σ table_contribution across all tables = 0.
///
/// For the circular constraint, `table_contribution / N` is the per-row offset
/// that makes the accumulated column wrap to zero at row N-1.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct BusPublicInputs<E>
where
    E: IsField,
{
    /// Total sum of all LogUp terms across all rows (L).
    /// Used for bus balance check and to derive the per-row offset L/N.
    pub table_contribution: FieldElement<E>,
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

    // Sign: +1 for senders, -1 for receivers
    let sign = if table_interaction.is_sender {
        FieldElement::<E>::one()
    } else {
        -FieldElement::<E>::one()
    };

    // Batch inversion: collect all fingerprints, invert once, then multiply back.
    // Compute fingerprint = z - (bus_id*α^0 + v0*α^1 + v1*α^2 + ...) using
    // base-field × extension-field multiplication (F×E→E) to avoid to_extension().
    //
    // Zero-allocation inner loop: accumulate the linear combination directly
    // into the fingerprint without collecting bus elements into intermediate Vecs.
    let bus_id_f = FieldElement::<F>::from(table_interaction.bus_id);
    let mut fingerprints: Vec<FieldElement<E>> = Vec::with_capacity(trace_len);
    for row in 0..trace_len {
        // Accumulate fingerprint directly: bus_id * α^0 + Σ element_i * α^(1+i)
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

        #[cfg(feature = "debug-checks")]
        {
            // Reconstruct base_elements for debug logging
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
                fingerprints.last().unwrap(),
            );
        }
    }

    FieldElement::inplace_batch_inverse(&mut fingerprints)
        .expect("fingerprint is zero - probability of sampling zero is negligible");

    // Compute terms: term[i] = sign * multiplicity[i] * fingerprint_inv[i]
    multiplicities
        .iter()
        .zip(fingerprints.iter())
        .map(|(multiplicity, fingerprint_inv)| multiplicity * &sign * fingerprint_inv)
        .collect()
}

/// Computes a batched term column for two interactions sharing one aux column.
///
/// Each row contains: `term[i] = sign_a * m_a[i] / fp_a[i] + sign_b * m_b[i] / fp_b[i]`
///
/// Uses a single batch inversion for both fingerprint vectors (2*N elements).
#[allow(clippy::needless_range_loop)]
fn compute_logup_batched_term_column<F, E>(
    interaction_a: &BusInteraction,
    interaction_b: &BusInteraction,
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
    #[cfg_attr(not(feature = "debug-checks"), allow(unused))] _table_name: &str,
) -> Vec<FieldElement<E>>
where
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
    E: IsField + Send + Sync,
{
    let z = &challenges[LOGUP_CHALLENGE_Z];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];

    let max_bus_elements = interaction_a
        .num_bus_elements()
        .max(interaction_b.num_bus_elements());
    let alpha_powers = compute_alpha_powers(alpha, max_bus_elements);

    let sign_a = if interaction_a.is_sender {
        FieldElement::<E>::one()
    } else {
        -FieldElement::<E>::one()
    };
    let sign_b = if interaction_b.is_sender {
        FieldElement::<E>::one()
    } else {
        -FieldElement::<E>::one()
    };

    // Helper to compute multiplicities for an interaction
    let compute_multiplicities = |interaction: &BusInteraction| -> Vec<FieldElement<F>> {
        match &interaction.multiplicity {
            Multiplicity::One => vec![FieldElement::one(); trace_len],
            Multiplicity::Column(col) => main_segment_cols[*col].clone(),
            Multiplicity::Sum(col_a, col_b) => main_segment_cols[*col_a]
                .iter()
                .zip(main_segment_cols[*col_b].iter())
                .map(|(a, b)| a + b)
                .collect(),
            Multiplicity::Negated(col) => main_segment_cols[*col]
                .iter()
                .map(|elem| FieldElement::<F>::one() - elem)
                .collect(),
            Multiplicity::Linear(terms) => (0..trace_len)
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
                .collect(),
        }
    };

    let multiplicities_a = compute_multiplicities(interaction_a);
    let multiplicities_b = compute_multiplicities(interaction_b);

    // Compute fingerprints for both interactions using accumulate_fingerprint
    // (zero-allocation inner loop: F×E multiplication instead of to_extension())
    let bus_id_a = FieldElement::<F>::from(interaction_a.bus_id);
    let bus_id_b = FieldElement::<F>::from(interaction_b.bus_id);

    // Concatenate both fingerprint vectors for a single batch inversion
    let mut all_fingerprints: Vec<FieldElement<E>> = Vec::with_capacity(2 * trace_len);

    for row in 0..trace_len {
        let mut lc_a = &bus_id_a * &alpha_powers[0];
        let mut alpha_offset = 1;
        for bv in &interaction_a.values {
            let consumed = bv.accumulate_fingerprint(
                main_segment_cols,
                row,
                &alpha_powers,
                alpha_offset,
                &mut lc_a,
            );
            alpha_offset += consumed;
        }
        all_fingerprints.push(z - &lc_a);
    }
    for row in 0..trace_len {
        let mut lc_b = &bus_id_b * &alpha_powers[0];
        let mut alpha_offset = 1;
        for bv in &interaction_b.values {
            let consumed = bv.accumulate_fingerprint(
                main_segment_cols,
                row,
                &alpha_powers,
                alpha_offset,
                &mut lc_b,
            );
            alpha_offset += consumed;
        }
        all_fingerprints.push(z - &lc_b);
    }

    // Single batch inversion for all 2*N fingerprints
    FieldElement::inplace_batch_inverse(&mut all_fingerprints)
        .expect("fingerprint is zero - probability of sampling zero is negligible");

    // Compute batched terms: term[i] = sign_a * m_a[i] * fp_a_inv[i] + sign_b * m_b[i] * fp_b_inv[i]
    (0..trace_len)
        .map(|row| {
            let fp_a_inv = &all_fingerprints[row];
            let fp_b_inv = &all_fingerprints[trace_len + row];
            &multiplicities_a[row] * &sign_a * fp_a_inv
                + &multiplicities_b[row] * &sign_b * fp_b_inv
        })
        .collect()
}

/// Builds the circular accumulated column from pre-computed term columns.
///
/// For the circular constraint: acc[(i+1) mod N] - acc[i] - terms[(i+1) mod N] + L/N = 0
/// We build: acc[0] = terms[0] - L/N, acc[i] = acc[i-1] + terms[i] - L/N
/// Result: acc[N-1] = L - N*(L/N) = 0
///
/// Returns L (table_contribution = sum of all terms across all rows).
fn build_accumulated_column_from_terms<F, E>(
    acc_column_idx: usize,
    term_columns: &[Vec<FieldElement<E>>],
    trace: &mut TraceTable<F, E>,
) -> FieldElement<E>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    if term_columns.is_empty() {
        return FieldElement::zero();
    }
    let trace_len = term_columns[0].len();

    // Compute L = sum of all terms across all rows
    let mut table_contribution = FieldElement::<E>::zero();
    for row in 0..trace_len {
        for col in term_columns {
            table_contribution = &table_contribution + &col[row];
        }
    }

    // offset_per_row = L / N
    let n = FieldElement::<E>::from(trace_len as u64);
    let offset_per_row = &table_contribution * n.inv().unwrap();

    // Build circular accumulated column
    let mut accumulated = FieldElement::<E>::zero();
    for row in 0..trace_len {
        let mut row_sum = FieldElement::<E>::zero();
        for col in term_columns {
            row_sum = row_sum + &col[row];
        }
        accumulated = &accumulated + &row_sum - &offset_per_row;
        trace.set_aux(row, acc_column_idx, accumulated.clone());
    }

    table_contribution
}

/// Sum per-interaction contributions by bus_id for debug reporting.
///
/// With batched term columns, we can't read individual interaction sums from
/// the trace anymore (each column holds the sum of two interactions). Instead,
/// we compute each interaction's sum from its raw term column.
#[cfg(feature = "debug-checks")]
#[allow(clippy::type_complexity)]
fn compute_debug_bus_sums_batched<F, E>(
    interactions: &[BusInteraction],
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
    table_name: &str,
) -> (
    HashMap<u64, FieldElement<E>>,
    HashMap<u64, FieldElement<E>>,
    HashMap<u64, FieldElement<E>>,
)
where
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
    E: IsField + Send + Sync,
{
    let mut bus_sums: HashMap<u64, FieldElement<E>> = HashMap::new();
    let mut sender_sums: HashMap<u64, FieldElement<E>> = HashMap::new();
    let mut receiver_sums: HashMap<u64, FieldElement<E>> = HashMap::new();

    // Compute each interaction's individual term column for summing
    for interaction in interactions.iter() {
        let individual_terms = compute_logup_term_column(
            interaction,
            main_segment_cols,
            trace_len,
            challenges,
            table_name,
        );
        let col_sum: FieldElement<E> = individual_terms
            .iter()
            .fold(FieldElement::zero(), |acc, x| acc + x);

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

/// Computes multiplicity for an interaction from a `TableView`.
fn compute_multiplicity_from_step<A: IsSubFieldOf<B>, B: IsField>(
    step: &TableView<A, B>,
    multiplicity: &Multiplicity,
) -> FieldElement<A> {
    match multiplicity {
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
    }
}

/// Computes the fingerprint for an interaction from a `TableView`.
///
/// Returns `z - (bus_id*α^0 + v[0]*α^1 + v[1]*α^2 + ...)`
fn compute_fingerprint_from_step<A: IsSubFieldOf<B>, B: IsField>(
    step: &TableView<A, B>,
    interaction: &BusInteraction,
    z: &FieldElement<B>,
    alpha_powers: &[FieldElement<B>],
) -> FieldElement<B> {
    let mut bus_elements: Vec<FieldElement<B>> = vec![FieldElement::from(interaction.bus_id)];
    bus_elements.extend(interaction.values.iter().flat_map(|bv| {
        let combined: Vec<FieldElement<A>> =
            bv.combine_from(|col| step.get_main_evaluation_element(0, col).clone());
        combined.into_iter().map(|v| v.to_extension())
    }));

    let linear_combination: FieldElement<B> = bus_elements
        .iter()
        .zip(alpha_powers.iter())
        .map(|(v, coeff)| v * coeff)
        .sum();

    z - &linear_combination
}

/// Constraint for a batched pair of interactions sharing one aux column.
///
/// Verifies: `c = m_a/fp_a + m_b/fp_b` where signs are baked into m_a, m_b.
///
/// Clearing denominators: `c * fp_a * fp_b - sign_a * m_a * fp_b - sign_b * m_b * fp_a = 0`
///
/// Degree 3: c (aux) × fp_a (linear in main) × fp_b (linear in main).
struct LookupBatchedTermConstraint {
    interaction_a: BusInteraction,
    interaction_b: BusInteraction,
    term_column_idx: usize,
    constraint_idx: usize,
}

impl LookupBatchedTermConstraint {
    pub fn new(
        interaction_a: BusInteraction,
        interaction_b: BusInteraction,
        term_column_idx: usize,
        constraint_idx: usize,
    ) -> Self {
        Self {
            interaction_a,
            interaction_b,
            term_column_idx,
            constraint_idx,
        }
    }
}

impl<F, E> TransitionConstraint<F, E> for LookupBatchedTermConstraint
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        3 // c * fp_a * fp_b
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
        fn evaluate_batched_term_constraint<A: IsSubFieldOf<B>, B: IsField>(
            step: &TableView<A, B>,
            term_column_idx: usize,
            interaction_a: &BusInteraction,
            interaction_b: &BusInteraction,
            rap_challenges: &&[FieldElement<B>],
            alpha_powers: &[FieldElement<B>],
        ) -> FieldElement<B> {
            let c = step.get_aux_evaluation_element(0, term_column_idx);
            let z = &rap_challenges[LOGUP_CHALLENGE_Z];

            let m_a = compute_multiplicity_from_step(step, &interaction_a.multiplicity);
            let m_b = compute_multiplicity_from_step(step, &interaction_b.multiplicity);

            let fp_a = compute_fingerprint_from_step(step, interaction_a, z, alpha_powers);
            let fp_b = compute_fingerprint_from_step(step, interaction_b, z, alpha_powers);

            let sign_a: FieldElement<B> = if interaction_a.is_sender {
                FieldElement::one()
            } else {
                -FieldElement::one()
            };
            let sign_b: FieldElement<B> = if interaction_b.is_sender {
                FieldElement::one()
            } else {
                -FieldElement::one()
            };

            // c * fp_a * fp_b - sign_a * m_a * fp_b - sign_b * m_b * fp_a = 0
            c * &fp_a * &fp_b - m_a * sign_a * &fp_b - m_b * sign_b * &fp_a
        }

        let res = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                rap_challenges,
                logup_alpha_powers,
                ..
            } => evaluate_batched_term_constraint(
                frame.get_evaluation_step(0),
                self.term_column_idx,
                &self.interaction_a,
                &self.interaction_b,
                rap_challenges,
                logup_alpha_powers,
            ),
            TransitionEvaluationContext::Verifier {
                frame,
                rap_challenges,
                logup_alpha_powers,
                ..
            } => evaluate_batched_term_constraint(
                frame.get_evaluation_step(0),
                self.term_column_idx,
                &self.interaction_a,
                &self.interaction_b,
                rap_challenges,
                logup_alpha_powers,
            ),
        };

        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx) {
            *eval = res;
        }
    }
}

/// Constraint for the accumulated column with absorbed interactions.
///
/// The accumulated column tracks the running sum of all committed term columns
/// plus 1-2 "absorbed" interactions whose terms are verified inline (not committed).
///
/// For 1 absorbed interaction:
///   `(acc_next - acc_curr - Σ terms + L/N) · f - sign · m = 0` (degree 2)
///
/// For 2 absorbed interactions:
///   `(acc_next - acc_curr - Σ terms + L/N) · f₁·f₂ - sign₁·m₁·f₂ - sign₂·m₂·f₁ = 0` (degree 3)
struct LookupAccumulatedConstraint {
    constraint_idx: usize,
    /// Number of committed term columns (excludes absorbed interactions)
    num_term_columns: usize,
    /// Index of the accumulated column (= num_term_columns)
    acc_column_idx: usize,
    /// 1 or 2 interactions absorbed into this constraint (not committed as columns)
    absorbed: Vec<BusInteraction>,
}

impl LookupAccumulatedConstraint {
    pub fn new(
        constraint_idx: usize,
        num_term_columns: usize,
        absorbed: Vec<BusInteraction>,
    ) -> Self {
        Self {
            constraint_idx,
            num_term_columns,
            acc_column_idx: num_term_columns,
            absorbed,
        }
    }
}

impl<F, E> TransitionConstraint<F, E> for LookupAccumulatedConstraint
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        1 + self.absorbed.len() // 2 for 1 absorbed, 3 for 2 absorbed
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0 // Circular constraint applies to all rows including last→first wrap
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        #[allow(clippy::too_many_arguments)]
        fn evaluate_accumulated_constraint<A: IsSubFieldOf<B>, B: IsField>(
            first_step: &TableView<A, B>,
            second_step: &TableView<A, B>,
            acc_column_idx: usize,
            num_term_columns: usize,
            logup_table_offset: &FieldElement<B>,
            absorbed: &[BusInteraction],
            rap_challenges: &&[FieldElement<B>],
            alpha_powers: &[FieldElement<B>],
        ) -> FieldElement<B> {
            // Accumulated column values
            let acc_curr = first_step.get_aux_evaluation_element(0, acc_column_idx);
            let acc_next = second_step.get_aux_evaluation_element(0, acc_column_idx);

            // Sum of all committed term columns at the next step
            let terms_sum: FieldElement<B> = (0..num_term_columns)
                .map(|i| second_step.get_aux_evaluation_element(0, i).clone())
                .sum();

            // delta = acc_next - acc_curr - terms_sum + L/N
            let delta = acc_next - acc_curr - terms_sum + logup_table_offset;

            let z = &rap_challenges[LOGUP_CHALLENGE_Z];

            // Clear denominators of absorbed interactions
            debug_assert!(matches!(absorbed.len(), 1 | 2));
            match absorbed.len() {
                1 => {
                    // (delta) · f - sign · m = 0
                    let m = compute_multiplicity_from_step(second_step, &absorbed[0].multiplicity);
                    let f =
                        compute_fingerprint_from_step(second_step, &absorbed[0], z, alpha_powers);
                    let sign: FieldElement<B> = if absorbed[0].is_sender {
                        FieldElement::one()
                    } else {
                        -FieldElement::one()
                    };
                    delta * &f - m * sign
                }
                2 => {
                    // (delta) · f₁ · f₂ - sign₁·m₁·f₂ - sign₂·m₂·f₁ = 0
                    let m1 = compute_multiplicity_from_step(second_step, &absorbed[0].multiplicity);
                    let m2 = compute_multiplicity_from_step(second_step, &absorbed[1].multiplicity);
                    let f1 =
                        compute_fingerprint_from_step(second_step, &absorbed[0], z, alpha_powers);
                    let f2 =
                        compute_fingerprint_from_step(second_step, &absorbed[1], z, alpha_powers);
                    let sign1: FieldElement<B> = if absorbed[0].is_sender {
                        FieldElement::one()
                    } else {
                        -FieldElement::one()
                    };
                    let sign2: FieldElement<B> = if absorbed[1].is_sender {
                        FieldElement::one()
                    } else {
                        -FieldElement::one()
                    };
                    delta * &f1 * &f2 - m1 * sign1 * &f2 - m2 * sign2 * &f1
                }
                _ => unreachable!("absorbed must contain 1 or 2 interactions"),
            }
        }

        let res = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                logup_table_offset,
                rap_challenges,
                logup_alpha_powers,
                ..
            } => evaluate_accumulated_constraint(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                self.acc_column_idx,
                self.num_term_columns,
                logup_table_offset,
                &self.absorbed,
                rap_challenges,
                logup_alpha_powers,
            ),
            TransitionEvaluationContext::Verifier {
                frame,
                logup_table_offset,
                rap_challenges,
                logup_alpha_powers,
                ..
            } => evaluate_accumulated_constraint(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                self.acc_column_idx,
                self.num_term_columns,
                logup_table_offset,
                &self.absorbed,
                rap_challenges,
                logup_alpha_powers,
            ),
        };

        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx) {
            *eval = res;
        }
    }
}

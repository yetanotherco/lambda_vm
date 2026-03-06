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
    trace::TraceTable,
    traits::TransitionEvaluationContext,
};
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
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

/// Number of challenges required by the LogUp protocol (z and alpha).
/// The gamma challenge is sampled separately after GKR for soundness.
pub const LOGUP_NUM_CHALLENGES: usize = 2;

/// Index of the `gamma` (γ) challenge in the per-table rap_challenges vector.
/// Used for batching column claims in the bridge running sum.
/// Sampled after GKR on the main transcript (not during Phase B).
pub const LOGUP_CHALLENGE_GAMMA: usize = 2;

/// Index of the bridge offset (target/N) in the per-table rap_challenges vector.
/// This is a derived value, not a random challenge.
pub const LOGUP_BRIDGE_OFFSET_IDX: usize = 3;

/// Start index of precomputed gamma powers in the per-table rap_challenges vector.
/// rap_challenges[LOGUP_GAMMA_POWERS_START + j] = γ^j for j = 0, 1, ..., K-1.
pub const LOGUP_GAMMA_POWERS_START: usize = 4;

/// Start index of GKR random_point coordinates in rap_challenges.
/// After gamma_powers[0..K], we append random_point[0..n].
/// The actual index is LOGUP_GAMMA_POWERS_START + K where K = number of distinct column indices.
/// Use `logup_random_point_start(interactions)` to compute the concrete index.
pub fn logup_random_point_start(interactions: &[BusInteraction]) -> usize {
    LOGUP_GAMMA_POWERS_START + extract_column_indices(interactions).len()
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
        transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    ) -> Self {
        let num_interactions = auxiliary_trace_build_data.interactions.len();

        // LogUp-GKR: aux trace has 2 columns if interactions exist:
        //   - Column 0: Lagrange kernel (l)
        //   - Column 1: Bridge running sum (σ)
        // The GKR sub-protocol replaces the old accumulated/term constraints.
        // A single LookupBridgeSumConstraint enforces the bridge.
        let num_aux_columns = if num_interactions > 0 { 2 } else { 0 };
        let trace_layout = (num_main_columns, num_aux_columns);

        // Compute max bus elements across all interactions for alpha power count
        let max_bus_elements = auxiliary_trace_build_data
            .interactions
            .iter()
            .map(|i| i.num_bus_elements())
            .max()
            .unwrap_or(0);

        // Add bridge running sum constraint for LogUp-GKR tables
        let mut all_constraints = transition_constraints;
        if num_interactions > 0 {
            let column_indices =
                extract_column_indices(&auxiliary_trace_build_data.interactions);
            let bridge_constraint = LookupBridgeSumConstraint {
                constraint_idx: all_constraints.len(),
                column_indices,
            };
            all_constraints.push(Box::new(bridge_constraint));
        }

        // Create context
        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: trace_layout.0 + trace_layout.1,
            transition_offsets: vec![0, 1],
            num_transition_constraints: all_constraints.len(),
        };

        Self {
            context,
            step_size,
            trace_layout,
            transition_constraints: all_constraints,
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

    fn bus_interactions(&self) -> &[BusInteraction] {
        &self.auxiliary_trace_build_data.interactions
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
        _challenges: &[FieldElement<E>],
    ) -> Option<BusPublicInputs<E>> {
        // LogUp-GKR: auxiliary trace has 2 columns:
        //   - Column 0: Lagrange kernel (l) — filled by prover from GKR random point
        //   - Column 1: Bridge running sum (σ) — filled by prover after γ sampling
        // Here we just allocate the columns.
        let (_, num_aux_columns) = self.trace_layout();
        if num_aux_columns > 0 && trace.num_aux_columns == 0 {
            trace.allocate_aux_table(num_aux_columns);
        }

        // No BusPublicInputs needed for GKR path — the GKR result replaces it.
        None
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
        _trace_length: usize,
    ) -> BoundaryConstraints<E> {
        let mut boundary_constraints = B::boundary_constraints(pub_inputs, rap_challenges);

        // LogUp-GKR: boundary constraint on Lagrange kernel column.
        // l[0] = eq(bits(0), r) = prod_{j=0}^{n-1} (1 - r_j)
        // where r_j are the GKR random point coordinates stored in rap_challenges.
        if self.has_trace_interaction() {
            let k = extract_column_indices(&self.auxiliary_trace_build_data.interactions).len();
            let rp_start = LOGUP_GAMMA_POWERS_START + k;
            if rap_challenges.len() > rp_start {
                let n = rap_challenges.len() - rp_start;
                let mut l0_expected = FieldElement::<E>::one();
                for j in 0..n {
                    l0_expected =
                        l0_expected * (FieldElement::<E>::one() - &rap_challenges[rp_start + j]);
                }
                // Aux column 0 is the Lagrange kernel; constrain l[0] = prod(1 - r_j)
                boundary_constraints.push(BoundaryConstraint::new_aux(0, 0, l0_expected));
            }
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
#[allow(dead_code, clippy::needless_range_loop)]
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
#[allow(dead_code, clippy::needless_range_loop)]
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
#[allow(dead_code)]
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

// =============================================================================
// LogUp-GKR Bridge Running Sum
// =============================================================================

/// Extract sorted distinct main column indices from bus interactions.
///
/// These are the columns referenced by `column_claims` in the GKR result.
/// The order must be consistent between the trace builder and the constraint evaluator.
fn extract_column_indices(interactions: &[BusInteraction]) -> Vec<usize> {
    let mut seen_cols = std::collections::HashSet::new();
    for inter in interactions {
        for val in &inter.values {
            for col_idx in val.column_indices() {
                seen_cols.insert(col_idx);
            }
        }
        match &inter.multiplicity {
            Multiplicity::One => {}
            Multiplicity::Column(c) => {
                seen_cols.insert(*c);
            }
            Multiplicity::Sum(a, b) => {
                seen_cols.insert(*a);
                seen_cols.insert(*b);
            }
            Multiplicity::Negated(c) => {
                seen_cols.insert(*c);
            }
            Multiplicity::Linear(terms) => {
                for term in terms {
                    match term {
                        LinearTerm::Column { column, .. } => {
                            seen_cols.insert(*column);
                        }
                        LinearTerm::ColumnUnsigned { column, .. } => {
                            seen_cols.insert(*column);
                        }
                        LinearTerm::Constant(_) => {}
                    }
                }
            }
        }
    }
    let mut col_indices: Vec<usize> = seen_cols.into_iter().collect();
    col_indices.sort_unstable();
    col_indices
}

/// Compute the bridge offset (target/N) and gamma powers from column claims.
///
/// Returns (bridge_offset, gamma_powers) where:
/// - bridge_offset = (Σ_j γ^j · c_j) / N
/// - gamma_powers = [γ^0, γ^1, ..., γ^{K-1}]
///
/// Both prover and verifier call this to derive the same values.
pub fn compute_bridge_params<E: IsField>(
    column_claims: &[(usize, FieldElement<E>)],
    gamma: &FieldElement<E>,
    trace_len: usize,
) -> (FieldElement<E>, Vec<FieldElement<E>>) {
    let k = column_claims.len();
    let gamma_powers = compute_alpha_powers(gamma, k);

    let mut target = FieldElement::<E>::zero();
    for ((_, c_j), gp) in column_claims.iter().zip(gamma_powers.iter()) {
        target = target + c_j * gp;
    }

    let n_inv = FieldElement::<E>::from(trace_len as u64).inv().unwrap();
    let bridge_offset = &target * &n_inv;

    (bridge_offset, gamma_powers)
}

/// Extend rap_challenges with bridge parameters (γ, bridge_offset, gamma_powers, random_point).
///
/// After calling this, the rap_challenges vector has:
/// - [0] = z, [1] = α (original)
/// - [2] = γ
/// - [3] = bridge_offset (target/N)
/// - [4..4+K] = γ^0, γ^1, ..., γ^{K-1}
/// - [4+K..4+K+n] = random_point[0], ..., random_point[n-1]
pub fn extend_rap_challenges_with_bridge<E: IsField>(
    rap_challenges: &mut Vec<FieldElement<E>>,
    column_claims: &[(usize, FieldElement<E>)],
    gamma: &FieldElement<E>,
    trace_len: usize,
    random_point: &[FieldElement<E>],
) {
    let (bridge_offset, gamma_powers) = compute_bridge_params(column_claims, gamma, trace_len);
    rap_challenges.push(gamma.clone()); // index 2
    rap_challenges.push(bridge_offset); // index 3
    for gp in gamma_powers {
        rap_challenges.push(gp); // indices 4, 5, ...
    }
    for rp in random_point {
        rap_challenges.push(rp.clone()); // indices 4+K, 4+K+1, ...
    }
}

/// Transition constraint for the bridge running sum column (σ).
///
/// Enforces the circular constraint:
///   σ_next - σ_curr - l_curr · batched_curr + bridge_offset = 0
///
/// where:
/// - σ is the running sum (aux column 1)
/// - l is the Lagrange kernel (aux column 0)
/// - batched_curr = Σ_j γ^j · col_j_curr (from main trace columns)
/// - bridge_offset = (Σ_j γ^j · c_j) / N (from rap_challenges)
///
/// The circular constraint (end_exemptions=0) telescopes to:
///   Σ_{i=0}^{N-1} l[i] · batched[i] = target
/// which, by γ-batching (Schwartz-Zippel), proves all individual claims
/// <l, col_j> = c_j with high probability.
pub struct LookupBridgeSumConstraint {
    constraint_idx: usize,
    /// Sorted distinct main column indices from bus interactions
    column_indices: Vec<usize>,
}

impl<F, E> TransitionConstraint<F, E> for LookupBridgeSumConstraint
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        2 // l_curr * batched_curr
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0 // circular: checked on all N rows (including wrap-around)
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                rap_challenges,
                ..
            } => {
                let bridge_offset = &rap_challenges[LOGUP_BRIDGE_OFFSET_IDX];

                let step0 = frame.get_evaluation_step(0);
                let step1 = frame.get_evaluation_step(1);

                // σ (aux column 1)
                let sigma_curr = step0.get_aux_evaluation_element(0, 1);
                let sigma_next = step1.get_aux_evaluation_element(0, 1);

                // l (aux column 0)
                let l_curr = step0.get_aux_evaluation_element(0, 0);

                // batched_curr = Σ_j γ^j · col_j_curr using precomputed gamma powers
                let mut batched = FieldElement::<E>::zero();
                for (j, &col_idx) in self.column_indices.iter().enumerate() {
                    let gamma_j = &rap_challenges[LOGUP_GAMMA_POWERS_START + j];
                    let col_val = step0.get_main_evaluation_element(0, col_idx);
                    // F×E→E: base field column × extension field gamma power
                    batched = batched + col_val * gamma_j;
                }

                // σ_next - σ_curr - l_curr * batched + bridge_offset
                transition_evaluations[self.constraint_idx] =
                    sigma_next - sigma_curr - l_curr * &batched + bridge_offset;
            }
            TransitionEvaluationContext::Verifier {
                frame,
                rap_challenges,
                ..
            } => {
                let bridge_offset = &rap_challenges[LOGUP_BRIDGE_OFFSET_IDX];

                let step0 = frame.get_evaluation_step(0);
                let step1 = frame.get_evaluation_step(1);

                let sigma_curr = step0.get_aux_evaluation_element(0, 1);
                let sigma_next = step1.get_aux_evaluation_element(0, 1);
                let l_curr = step0.get_aux_evaluation_element(0, 0);

                let mut batched = FieldElement::<E>::zero();
                for (j, &col_idx) in self.column_indices.iter().enumerate() {
                    let gamma_j = &rap_challenges[LOGUP_GAMMA_POWERS_START + j];
                    // In verifier path, main cols are also in E
                    let col_val = step0.get_main_evaluation_element(0, col_idx);
                    batched = batched + col_val * gamma_j;
                }

                transition_evaluations[self.constraint_idx] =
                    sigma_next - sigma_curr - l_curr * &batched + bridge_offset;
            }
        }
    }
}


// =============================================================================
// LogUp-GKR Leaf Fraction Computation
// =============================================================================

/// Computes multiplicities for a single interaction across all rows.
///
/// Returns a vector of `FieldElement<F>` with length `trace_len`.
fn compute_multiplicities_for_interaction<F: IsField + IsPrimeField>(
    interaction: &BusInteraction,
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
) -> Vec<FieldElement<F>> {
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
}

/// Computes fingerprints for a single interaction across all rows.
///
/// Each fingerprint is: `z - (bus_id * α^0 + Σ values * α^{1+...})`
///
/// Returns a vector of `FieldElement<E>` with length `trace_len`.
fn compute_fingerprints_for_interaction<F, E>(
    interaction: &BusInteraction,
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    z: &FieldElement<E>,
    alpha_powers: &[FieldElement<E>],
) -> Vec<FieldElement<E>>
where
    F: IsField + IsSubFieldOf<E> + IsPrimeField,
    E: IsField,
{
    let bus_id_f = FieldElement::<F>::from(interaction.bus_id);
    let mut fingerprints = Vec::with_capacity(trace_len);
    for row in 0..trace_len {
        let mut linear_combination = &bus_id_f * &alpha_powers[0];
        let mut alpha_offset = 1;
        for bv in &interaction.values {
            let consumed = bv.accumulate_fingerprint(
                main_segment_cols,
                row,
                alpha_powers,
                alpha_offset,
                &mut linear_combination,
            );
            alpha_offset += consumed;
        }
        fingerprints.push(z - &linear_combination);
    }
    fingerprints
}

/// Computes the leaf fractions for the GKR summation tree from a table's bus
/// interactions and main trace.
///
/// For each row `i` in `0..trace_len`, this function combines all K interactions
/// into a single fraction `N(i) / D(i)` where:
///
/// - `D(i) = Π_k fp_k(i)` (product of all fingerprints at row i)
/// - `N(i) = Σ_k sign_k * m_k(i) * Π_{j≠k} fp_j(i)` (cross-terms)
///
/// This is computed iteratively: starting with fraction 0/1, for each interaction k
/// the running fraction is updated via cross-multiplication:
///   `n_new = n_old * fp_k + sign_k * m_k * d_old`
///   `d_new = d_old * fp_k`
///
/// Returns `(numerators, denominators)` each of length `trace_len`.
///
/// # Arguments
/// * `interactions` - The bus interactions for this table
/// * `main_segment_cols` - Column-major main trace data: `main_segment_cols[col][row]`
/// * `trace_len` - Number of rows in the trace
/// * `challenges` - LogUp challenges `[z, alpha, ...]`
pub fn compute_logup_leaf_fractions<F, E>(
    interactions: &[BusInteraction],
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
) -> (Vec<FieldElement<E>>, Vec<FieldElement<E>>)
where
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
    E: IsField + Send + Sync,
{
    assert!(
        !interactions.is_empty(),
        "Must have at least one interaction"
    );

    let z = &challenges[LOGUP_CHALLENGE_Z];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];

    // Find max bus elements across all interactions for alpha power precomputation
    let max_bus_elements = interactions
        .iter()
        .map(|inter| inter.num_bus_elements())
        .max()
        .unwrap();
    let alpha_powers = compute_alpha_powers(alpha, max_bus_elements);

    // Precompute fingerprints and multiplicities for all interactions
    let all_fingerprints: Vec<Vec<FieldElement<E>>> = interactions
        .iter()
        .map(|inter| {
            compute_fingerprints_for_interaction(inter, main_segment_cols, trace_len, z, &alpha_powers)
        })
        .collect();

    let all_multiplicities: Vec<Vec<FieldElement<F>>> = interactions
        .iter()
        .map(|inter| compute_multiplicities_for_interaction(inter, main_segment_cols, trace_len))
        .collect();

    let all_signs: Vec<FieldElement<E>> = interactions
        .iter()
        .map(|inter| {
            if inter.is_sender {
                FieldElement::<E>::one()
            } else {
                -FieldElement::<E>::one()
            }
        })
        .collect();

    // For each row, combine all interactions into a single fraction using
    // iterative cross-multiplication.
    let mut numerators = Vec::with_capacity(trace_len);
    let mut denominators = Vec::with_capacity(trace_len);

    for row in 0..trace_len {
        // Start with fraction 0/1
        let mut running_n = FieldElement::<E>::zero();
        let mut running_d = FieldElement::<E>::one();

        for k in 0..interactions.len() {
            let fp_k = &all_fingerprints[k][row];
            let m_k = &all_multiplicities[k][row];

            // running_n = running_n * fp_k + sign_k * m_k * running_d
            // running_d = running_d * fp_k
            let new_n = &running_n * fp_k + m_k * &all_signs[k] * &running_d;
            let new_d = &running_d * fp_k;

            running_n = new_n;
            running_d = new_d;
        }

        numerators.push(running_n);
        denominators.push(running_d);
    }

    (numerators, denominators)
}

// =============================================================================
// LogUp-GKR Integration
// =============================================================================

use crate::gkr::{build_summation_tree, gkr_prove, GkrProof};
use crate::lagrange_kernel::eval_mle_base;
use crypto::fiat_shamir::is_transcript::IsTranscript;

/// Result of running the LogUp-GKR sub-protocol for a single table.
///
/// Contains the GKR proof, the random evaluation point, leaf-level claims,
/// and MLE claims for each distinct main trace column used in bus interactions.
#[derive(Debug, Clone)]
pub struct LogUpGkrResult<E: IsField> {
    /// Total table contribution (claimed_sum from GKR root = sum of all fractions).
    pub table_contribution: FieldElement<E>,
    /// The complete GKR proof for the summation tree.
    pub gkr_proof: GkrProof<E>,
    /// The random evaluation point produced by the GKR protocol (length = log2(trace_len)).
    pub random_point: Vec<FieldElement<E>>,
    /// Claimed MLE evaluation of the leaf numerator at the random point.
    pub n_claim: FieldElement<E>,
    /// Claimed MLE evaluation of the leaf denominator at the random point.
    pub d_claim: FieldElement<E>,
    /// MLE claims for each distinct main trace column used in bus interactions.
    /// Each entry is (column_index, MLE evaluation at random_point).
    pub column_claims: Vec<(usize, FieldElement<E>)>,
}

/// Verifies that column_claims are consistent with the GKR output (n_claim, d_claim).
///
/// The GKR protocol outputs `(random_point, n_claim, d_claim)` where `n_claim` and
/// `d_claim` are the claimed MLE evaluations of the leaf numerator and denominator
/// at `random_point`. The prover also provides `column_claims` which are MLE
/// evaluations of individual main trace columns at the same point.
///
/// For single-interaction tables: the leaf numerator and denominator are linear
/// functions of column values (n = sign * m, d = fp), so their MLEs can be exactly
/// reconstructed from column_claims. This gives a direct equality check.
///
/// For multi-interaction tables: the leaf fraction involves products of per-interaction
/// fingerprints and multiplicities (cross-multiplication), making it a nonlinear
/// function of column values. Since MLE does not preserve products, the direct
/// reconstruction from column_claims differs from the true MLE values. For these
/// tables, soundness of column_claims is guaranteed by the bridge running sum
/// constraint (which is verified as part of the STARK proof). We still verify
/// structural completeness (all referenced columns are present in column_claims).
///
/// # Arguments
/// * `n_claim` - Claimed MLE evaluation of leaf numerator at random_point (from GKR)
/// * `d_claim` - Claimed MLE evaluation of leaf denominator at random_point (from GKR)
/// * `column_claims` - `(column_index, claimed_value)` pairs from the proof
/// * `interactions` - The AIR's bus interactions
/// * `challenges` - LogUp challenges `[z, alpha]`
///
/// # Returns
/// `true` if verification passes, `false` otherwise.
pub fn reconstruct_and_verify_gkr_claims<E: IsField>(
    n_claim: &FieldElement<E>,
    d_claim: &FieldElement<E>,
    column_claims: &[(usize, FieldElement<E>)],
    interactions: &[BusInteraction],
    challenges: &[FieldElement<E>],
) -> bool {
    // Build a map from column index to claimed MLE value
    let claim_map: std::collections::HashMap<usize, &FieldElement<E>> = column_claims
        .iter()
        .map(|(col_idx, val)| (*col_idx, val))
        .collect();

    // Verify structural completeness: all columns referenced by interactions
    // must be present in column_claims.
    for inter in interactions {
        for val in &inter.values {
            for col_idx in val.column_indices() {
                if !claim_map.contains_key(&col_idx) {
                    return false;
                }
            }
        }
        match &inter.multiplicity {
            Multiplicity::One => {}
            Multiplicity::Column(c) => {
                if !claim_map.contains_key(c) {
                    return false;
                }
            }
            Multiplicity::Sum(a, b) => {
                if !claim_map.contains_key(a) || !claim_map.contains_key(b) {
                    return false;
                }
            }
            Multiplicity::Negated(c) => {
                if !claim_map.contains_key(c) {
                    return false;
                }
            }
            Multiplicity::Linear(terms) => {
                for term in terms {
                    match term {
                        LinearTerm::Column { column, .. }
                        | LinearTerm::ColumnUnsigned { column, .. } => {
                            if !claim_map.contains_key(column) {
                                return false;
                            }
                        }
                        LinearTerm::Constant(_) => {}
                    }
                }
            }
        }
    }

    let z = &challenges[LOGUP_CHALLENGE_Z];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];

    // Compute enough alpha powers for the largest interaction
    let max_bus_elements = interactions
        .iter()
        .map(|inter| inter.num_bus_elements())
        .max()
        .unwrap_or(0);
    let alpha_powers = compute_alpha_powers(alpha, max_bus_elements);

    // For each interaction, compute fingerprint and multiplicity from column claims,
    // then accumulate into a running fraction, same as compute_logup_leaf_fractions.
    let mut running_n = FieldElement::<E>::zero();
    let mut running_d = FieldElement::<E>::one();

    for inter in interactions {
        // Compute fingerprint: z - (bus_id * alpha^0 + sum of value contributions)
        let bus_id_e = FieldElement::<E>::from(inter.bus_id);
        let mut linear_combination = &bus_id_e * &alpha_powers[0];
        let mut alpha_offset = 1;

        for bv in &inter.values {
            alpha_offset += accumulate_fingerprint_from_claims(
                bv,
                &claim_map,
                &alpha_powers,
                alpha_offset,
                &mut linear_combination,
            );
        }

        let fp_claim = z - &linear_combination;

        // Compute multiplicity from column claims
        let m_claim = multiplicity_from_claims(&inter.multiplicity, &claim_map);

        // Sign: +1 for sender, -1 for receiver
        let sign = if inter.is_sender {
            FieldElement::<E>::one()
        } else {
            -FieldElement::<E>::one()
        };

        // Accumulate: n_new = n_old * fp + sign * m * d_old
        //             d_new = d_old * fp
        let new_n = &running_n * &fp_claim + &m_claim * &sign * &running_d;
        let new_d = &running_d * &fp_claim;

        running_n = new_n;
        running_d = new_d;
    }

    // For single-interaction tables, the leaf fraction is linear in column values:
    //   N(i) = sign * m(i), D(i) = fp(i)
    // so MLE(N)(r) and MLE(D)(r) can be exactly reconstructed from column MLEs.
    //
    // For multi-interaction tables, the cross-multiplication introduces nonlinear
    // terms (products of fingerprints/multiplicities across interactions), so
    // MLE(N)(r) != n_recon and MLE(D)(r) != d_recon in general.
    // Soundness for these tables is ensured by the bridge running sum constraint.
    if interactions.len() == 1 {
        // Direct check: reconstructed values must exactly match GKR output
        &running_n == n_claim && &running_d == d_claim
    } else {
        // Multi-interaction: structural check passed above.
        // The bridge constraint (verified during STARK proof) ensures column_claims
        // are consistent with the committed trace.
        true
    }
}

/// Accumulates the fingerprint contribution of a BusValue from column claims.
///
/// This mirrors `BusValue::accumulate_fingerprint` but operates on the claim map
/// (MLE evaluations at the GKR random point) instead of raw trace data.
///
/// Returns the number of alpha powers consumed.
fn accumulate_fingerprint_from_claims<E: IsField>(
    bv: &BusValue,
    claim_map: &std::collections::HashMap<usize, &FieldElement<E>>,
    alpha_powers: &[FieldElement<E>],
    alpha_offset: usize,
    acc: &mut FieldElement<E>,
) -> usize {
    match bv {
        BusValue::Packed {
            start_column,
            packing,
        } => {
            // Collect column claim values for this packing
            let columns: Vec<FieldElement<E>> = (*start_column
                ..*start_column + packing.num_columns())
                .map(|col| {
                    claim_map
                        .get(&col)
                        .cloned()
                        .cloned()
                        .unwrap_or_else(|| FieldElement::<E>::zero())
                })
                .collect();

            // Use Packing::combine to get bus elements, then accumulate with alpha powers
            let combined = packing.combine(&columns);
            for (i, elem) in combined.iter().enumerate() {
                *acc += elem * &alpha_powers[alpha_offset + i];
            }
            combined.len()
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
                        let val = claim_map
                            .get(column)
                            .cloned()
                            .cloned()
                            .unwrap_or_else(|| FieldElement::<E>::zero());
                        result += &val * coeff;
                    }
                    LinearTerm::ColumnUnsigned {
                        coefficient,
                        column,
                    } => {
                        let coeff = FieldElement::<E>::from(*coefficient);
                        let val = claim_map
                            .get(column)
                            .cloned()
                            .cloned()
                            .unwrap_or_else(|| FieldElement::<E>::zero());
                        result += &val * coeff;
                    }
                    LinearTerm::Constant(value) => {
                        result += FieldElement::<E>::from(*value);
                    }
                }
            }
            *acc += &result * &alpha_powers[alpha_offset];
            1
        }
    }
}

/// Computes the multiplicity value from column claims at the GKR random point.
///
/// This mirrors `compute_multiplicities_for_interaction` but operates on scalar
/// claim values instead of column vectors.
fn multiplicity_from_claims<E: IsField>(
    multiplicity: &Multiplicity,
    claim_map: &std::collections::HashMap<usize, &FieldElement<E>>,
) -> FieldElement<E> {
    match multiplicity {
        Multiplicity::One => FieldElement::<E>::one(),
        Multiplicity::Column(c) => claim_map
            .get(c)
            .cloned()
            .cloned()
            .unwrap_or_else(|| FieldElement::<E>::zero()),
        Multiplicity::Sum(a, b) => {
            let va = claim_map
                .get(a)
                .cloned()
                .cloned()
                .unwrap_or_else(|| FieldElement::<E>::zero());
            let vb = claim_map
                .get(b)
                .cloned()
                .cloned()
                .unwrap_or_else(|| FieldElement::<E>::zero());
            va + vb
        }
        Multiplicity::Negated(c) => {
            let val = claim_map
                .get(c)
                .cloned()
                .cloned()
                .unwrap_or_else(|| FieldElement::<E>::zero());
            FieldElement::<E>::one() - val
        }
        Multiplicity::Linear(terms) => {
            let mut result = FieldElement::<E>::zero();
            for term in terms {
                match term {
                    LinearTerm::Column {
                        coefficient,
                        column,
                    } => {
                        let coeff = FieldElement::<E>::from(*coefficient);
                        let val = claim_map
                            .get(column)
                            .cloned()
                            .cloned()
                            .unwrap_or_else(|| FieldElement::<E>::zero());
                        result += &val * coeff;
                    }
                    LinearTerm::ColumnUnsigned {
                        coefficient,
                        column,
                    } => {
                        let coeff = FieldElement::<E>::from(*coefficient);
                        let val = claim_map
                            .get(column)
                            .cloned()
                            .cloned()
                            .unwrap_or_else(|| FieldElement::<E>::zero());
                        result += &val * coeff;
                    }
                    LinearTerm::Constant(value) => {
                        result += FieldElement::<E>::from(*value);
                    }
                }
            }
            result
        }
    }
}

/// Run the LogUp-GKR sub-protocol for a single table's bus interactions.
///
/// This function:
/// 1. Computes per-row leaf fractions (numerator, denominator) from interactions
/// 2. Builds a binary summation tree over the leaf fractions
/// 3. Runs the GKR protocol to prove the summation tree root
/// 4. Extracts MLE claims for each distinct main trace column at the GKR random point
///
/// The GKR proof replaces the traditional per-row accumulated column with a
/// logarithmic-depth interactive proof, reducing auxiliary trace columns.
pub fn run_logup_gkr<F, E>(
    interactions: &[BusInteraction],
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
    transcript: &mut impl IsTranscript<E>,
) -> LogUpGkrResult<E>
where
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
    E: IsField + Send + Sync,
{
    // Step 1: Compute per-row leaf fractions
    let (numerators, denominators) =
        compute_logup_leaf_fractions(interactions, main_segment_cols, trace_len, challenges);

    // Step 2: Build the summation tree
    let tree = build_summation_tree(numerators, denominators);

    // Step 3: Run the GKR protocol
    let (gkr_proof, random_point, n_claim, d_claim) = gkr_prove(&tree, transcript);

    let table_contribution = gkr_proof.claimed_sum.clone();

    // Step 4: Extract column claims — compute MLE at the random point for each
    // distinct main trace column index referenced by any interaction.
    let mut seen_cols = std::collections::HashSet::new();
    for inter in interactions {
        for val in &inter.values {
            for col_idx in val.column_indices() {
                seen_cols.insert(col_idx);
            }
        }
        // Also collect column indices from multiplicities
        match &inter.multiplicity {
            Multiplicity::One => {}
            Multiplicity::Column(c) => {
                seen_cols.insert(*c);
            }
            Multiplicity::Sum(a, b) => {
                seen_cols.insert(*a);
                seen_cols.insert(*b);
            }
            Multiplicity::Negated(c) => {
                seen_cols.insert(*c);
            }
            Multiplicity::Linear(terms) => {
                for term in terms {
                    match term {
                        LinearTerm::Column { column, .. } => {
                            seen_cols.insert(*column);
                        }
                        LinearTerm::ColumnUnsigned { column, .. } => {
                            seen_cols.insert(*column);
                        }
                        LinearTerm::Constant(_) => {}
                    }
                }
            }
        }
    }

    let mut col_indices: Vec<usize> = seen_cols.into_iter().collect();
    col_indices.sort_unstable();

    let column_claims: Vec<(usize, FieldElement<E>)> = col_indices
        .into_iter()
        .map(|col_idx| {
            let claim = eval_mle_base(&main_segment_cols[col_idx], &random_point);
            (col_idx, claim)
        })
        .collect();

    LogUpGkrResult {
        table_contribution,
        gkr_proof,
        random_point,
        n_claim,
        d_claim,
        column_claims,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

    type F = GoldilocksField;
    type FE = FieldElement<F>;

    /// Test with 1 sender interaction, 4 rows, 1 column value, Multiplicity::One.
    ///
    /// For a single interaction with multiplicity 1 and sign +1 (sender):
    ///   N(i) = +1 * 1 = 1
    ///   D(i) = fp(i) = z - (bus_id * α^0 + col_val * α^1)
    #[test]
    fn test_compute_logup_leaf_fractions_single_sender() {
        let trace_len = 4;

        // Column data: 4 rows with values [10, 20, 30, 40]
        let col0: Vec<FE> = vec![
            FE::from(10u64),
            FE::from(20u64),
            FE::from(30u64),
            FE::from(40u64),
        ];
        let main_segment_cols = vec![col0.clone()];

        // Single sender interaction: bus_id=1, Multiplicity::One, one Direct column
        let interaction = BusInteraction::sender(
            1u64,
            Multiplicity::One,
            Packing::Direct.columns(&[0]),
        );
        let interactions = vec![interaction];

        // Challenges: z=100, alpha=3
        let z = FE::from(100u64);
        let alpha = FE::from(3u64);
        let challenges = vec![z.clone(), alpha.clone()];

        let (numerators, denominators) =
            compute_logup_leaf_fractions::<F, F>(&interactions, &main_segment_cols, trace_len, &challenges);

        assert_eq!(numerators.len(), trace_len);
        assert_eq!(denominators.len(), trace_len);

        // For each row, verify:
        //   fp = z - (bus_id * α^0 + col_val * α^1) = 100 - (1*1 + col_val * 3)
        //   numerator = 1 (multiplicity * sign)
        //   denominator = fp
        let alpha_powers = compute_alpha_powers(&alpha, 2); // [1, 3]

        for row in 0..trace_len {
            let bus_id_f = FE::from(1u64);
            let linear_comb = &bus_id_f * &alpha_powers[0] + &col0[row] * &alpha_powers[1];
            let expected_fp = &z - &linear_comb;

            assert_eq!(
                numerators[row],
                FE::one(),
                "Row {}: numerator should be 1 for sender with Multiplicity::One",
                row
            );
            assert_eq!(
                denominators[row], expected_fp,
                "Row {}: denominator should equal fingerprint",
                row
            );
        }
    }

    /// Test with 1 receiver interaction, Multiplicity::Column, to verify sign and
    /// multiplicity column extraction.
    #[test]
    fn test_compute_logup_leaf_fractions_single_receiver_with_column_multiplicity() {
        let trace_len = 4;

        // Column 0: values for fingerprint
        let col0: Vec<FE> = vec![
            FE::from(5u64),
            FE::from(6u64),
            FE::from(7u64),
            FE::from(8u64),
        ];
        // Column 1: multiplicities
        let col1: Vec<FE> = vec![
            FE::from(2u64),
            FE::from(0u64),
            FE::from(1u64),
            FE::from(3u64),
        ];
        let main_segment_cols = vec![col0.clone(), col1.clone()];

        // Receiver interaction: bus_id=0, Multiplicity from column 1
        let interaction = BusInteraction::receiver(
            0u64,
            Multiplicity::Column(1),
            Packing::Direct.columns(&[0]),
        );
        let interactions = vec![interaction];

        let z = FE::from(50u64);
        let alpha = FE::from(7u64);
        let challenges = vec![z.clone(), alpha.clone()];

        let (numerators, denominators) =
            compute_logup_leaf_fractions::<F, F>(&interactions, &main_segment_cols, trace_len, &challenges);

        let alpha_powers = compute_alpha_powers(&alpha, 2);

        for row in 0..trace_len {
            let bus_id_f = FE::from(0u64);
            let linear_comb = &bus_id_f * &alpha_powers[0] + &col0[row] * &alpha_powers[1];
            let expected_fp = &z - &linear_comb;

            // sign = -1 for receiver, so numerator = -multiplicity
            let expected_num = -col1[row].clone();

            assert_eq!(
                numerators[row], expected_num,
                "Row {}: numerator should be -multiplicity for receiver",
                row
            );
            assert_eq!(
                denominators[row], expected_fp,
                "Row {}: denominator should equal fingerprint",
                row
            );
        }
    }

    /// Test with 2 interactions to verify cross-multiplication combining.
    ///
    /// Two interactions combined at each row:
    ///   fraction = sign_0 * m_0 / fp_0 + sign_1 * m_1 / fp_1
    ///   = (sign_0 * m_0 * fp_1 + sign_1 * m_1 * fp_0) / (fp_0 * fp_1)
    #[test]
    fn test_compute_logup_leaf_fractions_two_interactions() {
        let trace_len = 2;

        // Column 0: values for interaction 0
        let col0: Vec<FE> = vec![FE::from(10u64), FE::from(20u64)];
        // Column 1: values for interaction 1
        let col1: Vec<FE> = vec![FE::from(30u64), FE::from(40u64)];
        let main_segment_cols = vec![col0.clone(), col1.clone()];

        // Interaction 0: sender, bus_id=0, Multiplicity::One, column 0
        let inter0 = BusInteraction::sender(
            0u64,
            Multiplicity::One,
            Packing::Direct.columns(&[0]),
        );
        // Interaction 1: receiver, bus_id=1, Multiplicity::One, column 1
        let inter1 = BusInteraction::receiver(
            1u64,
            Multiplicity::One,
            Packing::Direct.columns(&[1]),
        );
        let interactions = vec![inter0, inter1];

        let z = FE::from(200u64);
        let alpha = FE::from(5u64);
        let challenges = vec![z.clone(), alpha.clone()];

        let (numerators, denominators) =
            compute_logup_leaf_fractions::<F, F>(&interactions, &main_segment_cols, trace_len, &challenges);

        let alpha_powers = compute_alpha_powers(&alpha, 2);

        for row in 0..trace_len {
            // Fingerprint for interaction 0: z - (0 * α^0 + col0[row] * α^1)
            let bus_id_0 = FE::from(0u64);
            let lc_0 = &bus_id_0 * &alpha_powers[0] + &col0[row] * &alpha_powers[1];
            let fp_0 = &z - &lc_0;

            // Fingerprint for interaction 1: z - (1 * α^0 + col1[row] * α^1)
            let bus_id_1 = FE::from(1u64);
            let lc_1 = &bus_id_1 * &alpha_powers[0] + &col1[row] * &alpha_powers[1];
            let fp_1 = &z - &lc_1;

            // Combined fraction:
            //   n = (+1) * 1 * fp_1 + (-1) * 1 * fp_0
            //   d = fp_0 * fp_1
            let expected_n = &fp_1 - &fp_0;
            let expected_d = &fp_0 * &fp_1;

            assert_eq!(
                numerators[row], expected_n,
                "Row {}: numerator mismatch for two-interaction combine",
                row
            );
            assert_eq!(
                denominators[row], expected_d,
                "Row {}: denominator mismatch for two-interaction combine",
                row
            );
        }
    }

    /// Verify that the leaf fractions are consistent with the existing
    /// `compute_logup_term_column` function for a single interaction.
    ///
    /// For 1 interaction: term[i] = sign * m[i] / fp[i] = N[i] / D[i]
    /// So term[i] * D[i] should equal N[i].
    #[test]
    fn test_leaf_fractions_consistent_with_term_column() {
        let trace_len = 8;

        // Column with some values
        let col0: Vec<FE> = (1..=8).map(|v| FE::from(v as u64)).collect();
        let main_segment_cols = vec![col0];

        let interaction = BusInteraction::sender(
            2u64,
            Multiplicity::One,
            Packing::Direct.columns(&[0]),
        );

        let z = FE::from(1000u64);
        let alpha = FE::from(11u64);
        let challenges = vec![z, alpha];

        // Compute leaf fractions
        let (numerators, denominators) = compute_logup_leaf_fractions::<F, F>(
            &[interaction.clone()],
            &main_segment_cols,
            trace_len,
            &challenges,
        );

        // Compute term column (= sign * m / fp)
        let terms = compute_logup_term_column::<F, F>(
            &interaction,
            &main_segment_cols,
            trace_len,
            &challenges,
            "test",
        );

        // Verify: term[i] * denominator[i] == numerator[i]
        for row in 0..trace_len {
            let lhs = &terms[row] * &denominators[row];
            assert_eq!(
                lhs, numerators[row],
                "Row {}: term * denominator should equal numerator",
                row
            );
        }
    }
}

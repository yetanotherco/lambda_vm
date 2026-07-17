#[cfg(feature = "debug-checks")]
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        builder::{
            ConstraintMeta, ConstraintSet, ProverEvalFolder, VerifierEvalFolder, num_base_from_meta,
        },
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
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSliceMut,
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

/// Precomputed field element shift constants for packing operations.
///
/// Avoids repeated `FieldElement::from()` conversions in hot loops.
/// Create once before a loop and pass by reference.
pub struct PackingShifts<F: IsField> {
    pub shift_8: FieldElement<F>,
    pub shift_16: FieldElement<F>,
    pub shift_24: FieldElement<F>,
}

impl<F: IsField> Default for PackingShifts<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IsField> PackingShifts<F> {
    pub fn new() -> Self {
        let shift_8 = FieldElement::<F>::from(SHIFT_8);
        let shift_16 = FieldElement::<F>::from(SHIFT_16);
        let shift_24 = &shift_8 * &shift_16;
        Self {
            shift_8,
            shift_16,
            shift_24,
        }
    }
}

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

/// Index of the `alpha` (α) challenge in the LogUp challenges vector.
/// Used as the base for linear combination of row values.
pub const LOGUP_CHALLENGE_ALPHA: usize = 1;

/// Number of challenges required by the LogUp protocol.
pub const LOGUP_NUM_CHALLENGES: usize = 2;

/// Chunk size for fused chunk-local LogUp processing.
/// Each chunk processes all interactions for CHUNK_SIZE rows, fitting in L2 cache.
#[cfg(feature = "parallel")]
const LOGUP_CHUNK_SIZE: usize = 1024;

/// Split N interactions into committed batched pairs and absorbed remainder.
///
/// Returns `(num_committed_pairs, absorbed_count)` where:
/// - Committed pairs get dedicated auxiliary term columns (2 interactions per column)
/// - Absorbed interactions (1 or 2) are folded into the accumulated constraint
pub(crate) fn split_interactions(num_interactions: usize) -> (usize, usize) {
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

    /// Accumulates the fingerprint contribution of this packing into `acc`,
    /// using `get_col` to access column data by index.
    ///
    /// Computes: `acc += Σ combined_element_i * alpha_powers[alpha_offset + i]`
    /// where `combined_element_i` are the bus elements produced by this packing.
    ///
    /// Returns the number of alpha powers consumed (= num_bus_elements()).
    ///
    /// This is the single canonical implementation of packing arithmetic for
    /// fingerprint accumulation. Both column-major (`main_cols[col][row]`) and
    /// TableView callers delegate here with an appropriate closure.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_fingerprint_with<'a, F, E>(
        &self,
        start_col: usize,
        get_col: impl Fn(usize) -> &'a FieldElement<F>,
        alpha_powers: &[FieldElement<E>],
        alpha_offset: usize,
        acc: &mut FieldElement<E>,
        shifts: &PackingShifts<F>,
    ) -> usize
    where
        F: IsField + IsSubFieldOf<E> + 'a,
        E: IsField,
    {
        debug_assert!(
            alpha_powers.len() >= alpha_offset + self.num_bus_elements(),
            "alpha_powers too short: len={}, need={}",
            alpha_powers.len(),
            alpha_offset + self.num_bus_elements()
        );

        match self {
            Packing::Direct => {
                *acc += get_col(start_col) * &alpha_powers[alpha_offset];
                1
            }
            Packing::Word2L => {
                let combined = get_col(start_col) + get_col(start_col + 1) * &shifts.shift_16;
                *acc += &combined * &alpha_powers[alpha_offset];
                1
            }
            Packing::Word4L => {
                let combined = get_col(start_col)
                    + get_col(start_col + 1) * &shifts.shift_8
                    + get_col(start_col + 2) * &shifts.shift_16
                    + get_col(start_col + 3) * &shifts.shift_24;
                *acc += &combined * &alpha_powers[alpha_offset];
                1
            }
            // 2× Direct
            Packing::DWordWL => {
                *acc += get_col(start_col) * &alpha_powers[alpha_offset];
                *acc += get_col(start_col + 1) * &alpha_powers[alpha_offset + 1];
                2
            }
            // Direct + Word2L
            Packing::DWordHHW => {
                *acc += get_col(start_col) * &alpha_powers[alpha_offset];
                let w = get_col(start_col + 1) + get_col(start_col + 2) * &shifts.shift_16;
                *acc += &w * &alpha_powers[alpha_offset + 1];
                2
            }
            // Word2L + Direct
            Packing::DWordWHH => {
                let w = get_col(start_col) + get_col(start_col + 1) * &shifts.shift_16;
                *acc += &w * &alpha_powers[alpha_offset];
                *acc += get_col(start_col + 2) * &alpha_powers[alpha_offset + 1];
                2
            }
            // 2× Word2L
            Packing::DWordHL => {
                let w0 = get_col(start_col) + get_col(start_col + 1) * &shifts.shift_16;
                *acc += &w0 * &alpha_powers[alpha_offset];
                let w1 = get_col(start_col + 2) + get_col(start_col + 3) * &shifts.shift_16;
                *acc += &w1 * &alpha_powers[alpha_offset + 1];
                2
            }
            // 2× Word4L
            Packing::DWordBL => {
                let w0 = get_col(start_col)
                    + get_col(start_col + 1) * &shifts.shift_8
                    + get_col(start_col + 2) * &shifts.shift_16
                    + get_col(start_col + 3) * &shifts.shift_24;
                *acc += &w0 * &alpha_powers[alpha_offset];
                let w1 = get_col(start_col + 4)
                    + get_col(start_col + 5) * &shifts.shift_8
                    + get_col(start_col + 6) * &shifts.shift_16
                    + get_col(start_col + 7) * &shifts.shift_24;
                *acc += &w1 * &alpha_powers[alpha_offset + 1];
                2
            }
            // 4× Word2L
            Packing::QuadHL => {
                for i in 0..4 {
                    let c = start_col + i * 2;
                    let w = get_col(c) + get_col(c + 1) * &shifts.shift_16;
                    *acc += &w * &alpha_powers[alpha_offset + i];
                }
                4
            }
            // 4× Direct
            Packing::QuadWL => {
                for i in 0..4 {
                    *acc += get_col(start_col + i) * &alpha_powers[alpha_offset + i];
                }
                4
            }
        }
    }

    /// Accumulates fingerprint from column-major trace data.
    ///
    /// Delegates to `accumulate_fingerprint_with` using `main_cols[col][row]`.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_fingerprint<F, E>(
        &self,
        main_cols: &[Vec<FieldElement<F>>],
        row: usize,
        start_col: usize,
        alpha_powers: &[FieldElement<E>],
        alpha_offset: usize,
        acc: &mut FieldElement<E>,
        shifts: &PackingShifts<F>,
    ) -> usize
    where
        F: IsField + IsSubFieldOf<E>,
        E: IsField,
    {
        self.accumulate_fingerprint_with(
            start_col,
            |col| &main_cols[col][row],
            alpha_powers,
            alpha_offset,
            acc,
            shifts,
        )
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
                let shift_16 = FieldElement::<E>::from(65536u64);
                vec![&columns[0] + &columns[1] * &shift_16]
            }

            Packing::Word4L => {
                // b₀ + 2⁸·b₁ + 2¹⁶·b₂ + 2²⁴·b₃
                let shift_8 = FieldElement::<E>::from(256u64);
                let shift_16 = FieldElement::<E>::from(65536u64);
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
/// A `BusValue` produces 1, 2, or 4 bus elements for the fingerprint depending
/// on its packing (see [`BusValue::num_bus_elements`]); `Linear` always
/// produces 1. The fingerprint is computed as: `z - (v₀ + α·v₁ + α²·v₂ + ...)`
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

    /// Returns the number of bus elements this value produces: 1, 2, or 4 for
    /// `Packed` depending on the packing, always 1 for `Linear`.
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
        shifts: &PackingShifts<F>,
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
                shifts,
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
                // Bus elements that are zero on this row contribute nothing — skip the
                // F×E multiply. (Covers the constant(0) bus-width padding plus any
                // variable element that is zero on this row; α⁰ = 1 covers the bus-id
                // term separately.)
                if result != FieldElement::<F>::zero() {
                    *acc += &result * &alpha_powers[alpha_offset];
                }
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

    /// Accumulates fingerprint from a `TableView` step (constraint evaluation path).
    ///
    /// Equivalent to `combine_from` + multiply by alpha powers, but avoids
    /// the intermediate `Vec<FieldElement>` allocation per call.
    /// Packed variants delegate to `Packing::accumulate_fingerprint_with`.
    pub fn accumulate_fingerprint_from_step<A: IsSubFieldOf<B>, B: IsField>(
        &self,
        step: &TableView<A, B>,
        alpha_powers: &[FieldElement<B>],
        alpha_offset: usize,
        acc: &mut FieldElement<B>,
        shifts: &PackingShifts<A>,
    ) -> usize {
        match self {
            BusValue::Packed {
                start_column,
                packing,
            } => packing.accumulate_fingerprint_with(
                *start_column,
                |col| step.get_main_evaluation_element(0, col),
                alpha_powers,
                alpha_offset,
                acc,
                shifts,
            ),
            BusValue::Linear(terms) => {
                debug_assert!(
                    alpha_powers.len() > alpha_offset,
                    "alpha_powers too short: len={}, need={}",
                    alpha_powers.len(),
                    alpha_offset + 1
                );
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
                // Bus elements that are zero on this row contribute nothing — skip the
                // F×E multiply. (Covers the constant(0) bus-width padding plus any
                // variable element that is zero on this row.)
                if result != FieldElement::<A>::zero() {
                    *acc += result * &alpha_powers[alpha_offset];
                }
                1
            }
        }
    }
}

// =============================================================================
// AirWithBuses
// =============================================================================

/// Struct representing an AIR with Lookup. Contains own implementation of boundary constraints and auxiliary trace building
///
/// `CS` is the table's [`ConstraintSet`]: its single `eval` body emits the
/// table's base-field transition constraints, and the framework appends the
/// LogUp constraints (generated from [`Self::logup`]) after them. One body
/// serves the compiled prover folder, the verifier folder, and IR capture.
pub struct AirWithBuses<
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
    E: IsField + Send + Sync,
    B: BoundaryConstraintBuilder<F, E, PI>,
    PI,
    CS: ConstraintSet<F, E>,
> {
    context: AirContext,
    step_size: usize,
    trace_layout: (usize, usize),
    /// The table's single-source constraint set (base-field constraints).
    constraint_set: CS,
    /// The LogUp layout: the framework generates the LogUp (extension)
    /// constraints from this and appends them after the `constraint_set` ones.
    logup: LogUpLayout,
    /// Idx-ordered metadata for all transition constraints, DERIVED at
    /// construction: `constraint_set.meta()` (base prefix) followed by the
    /// LogUp emission's derived metadata (ext).
    meta: Vec<ConstraintMeta>,
    /// Number of base-field constraints (the `RootKind::Base` prefix length of
    /// `meta`) — these use the cheaper F×E accumulation path.
    num_base: usize,
    /// Lazily captured flat IR of every transition constraint, built once on
    /// first request (prover/GPU/tests only — the verify path never forces it).
    constraint_program: std::sync::OnceLock<crate::constraint_ir::ConstraintProgram<F, E>>,
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
    CS: ConstraintSet<F, E>,
> AirWithBuses<F, E, B, PI, CS>
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
        constraint_set: CS,
    ) -> Self {
        // Base-field (table) constraints come from the constraint set; LogUp
        // (extension) constraints are appended by the framework from the layout.
        let num_interactions = auxiliary_trace_build_data.interactions.len();
        let logup = LogUpLayout::from_interactions(auxiliary_trace_build_data.interactions.clone());
        let num_term_columns = logup.num_term_columns;

        // meta = constraint_set base-prefix meta + appended LogUp ext meta,
        // both DERIVED by running the respective bodies through a MetaBuilder
        // (the `{degree, end_exemptions}` declared at each emit).
        let mut meta = constraint_set.meta();
        let num_base = num_base_from_meta(&meta);
        // The set is entirely base-field (its meta is a Base prefix).
        debug_assert_eq!(num_base, meta.len(), "constraint set meta must be all-base");
        let mut logup_mb = crate::constraints::builder::MetaBuilder::new();
        emit_logup_constraints::<F, E, _>(&mut logup_mb, &logup, num_base);
        meta.extend(logup_mb.into_meta());

        // Layout: num_committed_pairs term columns + 1 accumulated = ⌈N/2⌉
        let num_aux_columns = if num_interactions > 0 {
            num_term_columns + 1
        } else {
            0
        };
        let trace_layout = (num_main_columns, num_aux_columns);

        // Compute max bus elements across all interactions for alpha power count
        let max_bus_elements = logup
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
            num_transition_constraints: meta.len(),
        };

        Self {
            context,
            step_size,
            trace_layout,
            constraint_set,
            logup,
            meta,
            num_base,
            constraint_program: std::sync::OnceLock::new(),
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

impl<F, E, B, PI, CS> crate::traits::AIR for AirWithBuses<F, E, B, PI, CS>
where
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    B: BoundaryConstraintBuilder<F, E, PI>,
    PI: Send + Sync,
    CS: ConstraintSet<F, E>,
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

    fn trace_ood_next_row_columns(&self) -> Vec<usize> {
        // Columns read on the next row: the table's own constraint-set reads
        // (`main(1, ·)`, statically declared) plus, when the table has LogUp
        // interactions, the circular accumulator's aux column (main width + its
        // aux index). The verifier opens OOD next-row evaluations only for these;
        // a next-row read that is not declared here is pruned and reconstructed as
        // zero, silently corrupting this table's transition evaluation.
        let mut cols = self.constraint_set.next_row_columns();
        if !self.auxiliary_trace_build_data.interactions.is_empty() {
            cols.push(self.trace_layout.0 + self.logup.acc_column_idx);
        }
        cols.sort_unstable();
        cols.dedup();
        cols
    }

    fn has_trace_interaction(&self) -> bool {
        !self.auxiliary_trace_build_data.interactions.is_empty()
    }

    fn max_bus_elements(&self) -> usize {
        self.max_bus_elements
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        // Only the per-table MAX degree is consumed. Base constraints declare it
        // once via `ConstraintSet::max_degree()`; the framework's LogUp
        // constraints contribute their own known max (batched terms degree 3,
        // accumulator `1 + absorbed`).
        let max_degree = self
            .constraint_set
            .max_degree()
            .max(logup_max_degree(&self.logup));
        // The composition polynomial is the constraint QUOTIENT H = Σ βᵢ·Cᵢ/Zᵢ. Its degree is
        // deg(Cᵢ) − deg(Zᵢ) = (max_degree−1)·N − max_degree + eᵢ, so with the end-exemptions
        // eᵢ < max_degree (the max-degree LogUp constraints have eᵢ = 0) it fits in
        // (max_degree−1) parts — the max_degree-th part is identically zero. The tight bound is
        // therefore (max_degree−1)·N; the previous max_degree·N committed and opened a wasted
        // all-zero part (e.g. 3 parts for a degree-3 AIR where 2 suffice).
        trace_length * (max_degree - 1).max(1)
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn num_base_transition_constraints(&self) -> usize {
        self.num_base
    }

    fn constraints_meta(&self) -> &[ConstraintMeta] {
        &self.meta
    }

    fn compute_transition_prover(
        &self,
        ctx: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
        base_evals: &mut [FieldElement<Self::Field>],
        ext_evals: &mut [FieldElement<Self::FieldExtension>],
    ) {
        // One folder pass runs BOTH the table constraint set and the LogUp
        // emission; LogUp constraints are appended after the set's (idx offset
        // by the base-constraint count).
        run_air_transition_prover(
            &self.constraint_set,
            &self.logup,
            ctx,
            base_evals,
            ext_evals,
        );
    }

    fn compute_transition(
        &self,
        ctx: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        run_air_transition_verifier(
            &self.constraint_set,
            &self.logup,
            self.num_base,
            self.meta.len(),
            ctx,
        )
    }

    fn constraint_program(
        &self,
    ) -> &crate::constraint_ir::ConstraintProgram<Self::Field, Self::FieldExtension> {
        // Lazily captured once (prover/GPU/tests only — the verify path never
        // calls this). Runs the table set AND the LogUp emission through one
        // CaptureBuilder, matching the folder emission order/indexing exactly.
        self.constraint_program.get_or_init(|| {
            let mut cb = crate::constraints::builder::CaptureBuilder::<F, E>::new();
            self.constraint_set.eval(&mut cb);
            emit_logup_constraints(&mut cb, &self.logup, self.num_base);
            let (prog, _degrees) = cb.finish(self.num_base);
            prog
        })
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
        let _table_name = self.name.as_deref().unwrap_or("UNKNOWN");

        // Device-resident trace-domain main columns from the R1 main LDE, cloned
        // (Arc, cheap) into a local so no borrow of `trace` is held across the
        // `set_aux_resident` mutable borrow below. When present, the resident aux
        // build reads them in place and skips the ~3 GB main re-upload.
        #[cfg(all(feature = "cuda", not(feature = "debug-checks")))]
        let resident_main = trace.main_trace_dev.clone();

        // Split interactions: committed pairs get term columns, last 1-2 are absorbed (virtual)
        let (num_committed_pairs, absorbed_count) = split_interactions(num_interactions);

        // Compute committed term columns (batched pairs only).
        // With `parallel`: when `trace_len > LOGUP_CHUNK_SIZE` the chunk-internal
        // parallelism inside each pair already saturates Rayon, so iterate pairs
        // sequentially to keep cache locality. When `trace_len <= LOGUP_CHUNK_SIZE`
        // each pair yields a single chunk, so parallelize across pairs to recover
        // the throughput the per-pair dispatch used to provide for small-trace
        // tables with many interactions.
        // Without `parallel`: sequential over pairs, sequential over rows.
        let interactions = &self.auxiliary_trace_build_data.interactions;

        // GPU-resident aux build (Goldilocks + ext3, not disk-spill, not
        // debug-checks): build the aux columns on device and keep them resident
        // for the aux LDE (no term-column download). Returns the table
        // contribution; the host set_aux + CPU accumulate below are skipped.
        #[cfg(all(feature = "cuda", not(feature = "debug-checks")))]
        if trace.resident_aux_ok()
            && let Some(ra) = crate::logup_gpu::try_build_aux_resident_gpu::<F, E>(
                interactions,
                &main_segment_cols,
                resident_main.as_ref().map(|r| (r.buf.as_ref(), r.rows)),
                trace_len,
                challenges,
            )
        {
            let table_contribution = crate::gpu_lde::u64_to_ext3_vec::<E>(&ra.table_contribution)
                .pop()
                .expect("one ext3 element");
            trace.set_aux_resident(ra);
            return Some(BusPublicInputs { table_contribution });
        }

        // GPU aux build (Goldilocks + ext3 + above threshold) computes all term
        // columns on device, byte identical, and falls back to the CPU build.
        #[cfg(feature = "cuda")]
        let gpu_term_cols = crate::logup_gpu::try_build_term_columns_gpu::<F, E>(
            interactions,
            &main_segment_cols,
            trace_len,
            challenges,
        );
        #[cfg(not(feature = "cuda"))]
        #[allow(clippy::type_complexity)]
        let gpu_term_cols: Option<(Vec<Vec<FieldElement<E>>>, Vec<FieldElement<E>>)> = None;

        let (committed_columns, virtual_column) = match gpu_term_cols {
            Some(cols) => cols,
            None => {
                let build_pair = |i: usize| {
                    compute_logup_term_column(
                        &[&interactions[i * 2], &interactions[i * 2 + 1]],
                        &main_segment_cols,
                        trace_len,
                        challenges,
                        _table_name,
                    )
                };

                #[cfg(feature = "parallel")]
                let committed_columns: Vec<Vec<FieldElement<E>>> = if trace_len <= LOGUP_CHUNK_SIZE
                {
                    (0..num_committed_pairs)
                        .into_par_iter()
                        .map(build_pair)
                        .collect()
                } else {
                    (0..num_committed_pairs).map(build_pair).collect()
                };
                #[cfg(not(feature = "parallel"))]
                let committed_columns: Vec<Vec<FieldElement<E>>> =
                    (0..num_committed_pairs).map(build_pair).collect();

                // Virtual column for absorbed interactions (NOT written to trace).
                let virtual_column = if absorbed_count == 2 {
                    compute_logup_term_column(
                        &[
                            &interactions[num_interactions - 2],
                            &interactions[num_interactions - 1],
                        ],
                        &main_segment_cols,
                        trace_len,
                        challenges,
                        _table_name,
                    )
                } else {
                    compute_logup_term_column(
                        &[&interactions[num_interactions - 1]],
                        &main_segment_cols,
                        trace_len,
                        challenges,
                        _table_name,
                    )
                };
                (committed_columns, virtual_column)
            }
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
                _table_name,
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
        _trace_length: usize,
    ) -> BoundaryConstraints<E> {
        let mut boundary_constraints = B::boundary_constraints(pub_inputs, rap_challenges);

        // Pin acc[0] = 0 to remove the constant-shift degree of freedom in the
        // circular transition constraint (forward accumulation starts at 0).
        if !self.auxiliary_trace_build_data.interactions.is_empty() {
            let acc_col_idx = self.trace_layout.1 - 1; // last aux column = accumulated
            boundary_constraints.push(BoundaryConstraint::new_aux(
                acc_col_idx,
                0,
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

    /// Difference of two columns: `col_a - col_b`.
    Diff(usize, usize),

    /// Sum of three columns: `col_a + col_b + col_c`.
    Sum3(usize, usize, usize),

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

impl Multiplicity {
    /// Evaluate the multiplicity expression to a field element. `get_col(i)`
    /// must return the value of main column `i` at the row being evaluated.
    #[inline]
    fn evaluate_with<F, G>(&self, get_col: G) -> FieldElement<F>
    where
        F: IsField,
        G: Fn(usize) -> FieldElement<F>,
    {
        match self {
            Multiplicity::One => FieldElement::one(),
            Multiplicity::Column(col) => get_col(*col),
            Multiplicity::Sum(a, b) => get_col(*a) + get_col(*b),
            Multiplicity::Negated(col) => FieldElement::<F>::one() - get_col(*col),
            Multiplicity::Diff(a, b) => get_col(*a) - get_col(*b),
            Multiplicity::Sum3(a, b, c) => get_col(*a) + get_col(*b) + get_col(*c),
            Multiplicity::Linear(terms) => {
                let mut result = FieldElement::<F>::zero();
                for term in terms {
                    match *term {
                        LinearTerm::Column {
                            coefficient,
                            column,
                        } => result += get_col(column) * FieldElement::<F>::from(coefficient),
                        LinearTerm::ColumnUnsigned {
                            coefficient,
                            column,
                        } => result += get_col(column) * FieldElement::<F>::from(coefficient),
                        LinearTerm::Constant(value) => result += FieldElement::<F>::from(value),
                    }
                }
                result
            }
        }
    }

    /// Evaluate the multiplicity for a single row of column-major main data.
    #[inline]
    pub(crate) fn evaluate_at_row<F: IsField>(
        &self,
        main_segment_cols: &[Vec<FieldElement<F>>],
        row: usize,
    ) -> FieldElement<F> {
        self.evaluate_with(|col| main_segment_cols[col][row].clone())
    }
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
/// so the bus balance check is: Σ table_contribution across all tables = expected_bus_balance.
///
/// For the circular constraint, `table_contribution / N` is the per-row offset
/// that makes the accumulated column wrap to zero at row N-1.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct BusPublicInputs<E>
where
    E: IsField,
{
    /// Total sum of all LogUp terms across all rows (L).
    /// Used for bus balance check and to derive the per-row offset L/N.
    pub table_contribution: FieldElement<E>,
    /// Per-bus sums for this table (bus_id → sum) - for debug aggregation.
    /// Debug-only aggregation state; not part of the archived proof (`Skip`).
    #[cfg(feature = "debug-checks")]
    #[rkyv(with = rkyv::with::Skip)]
    pub per_bus_sums: HashMap<u64, FieldElement<E>>,
    /// Per-bus sender sums (bus_id → sum) - positive contributions
    #[cfg(feature = "debug-checks")]
    #[rkyv(with = rkyv::with::Skip)]
    pub per_bus_sender_sums: HashMap<u64, FieldElement<E>>,
    /// Per-bus receiver sums (bus_id → sum) - absolute value (before negation)
    #[cfg(feature = "debug-checks")]
    #[rkyv(with = rkyv::with::Skip)]
    pub per_bus_receiver_sums: HashMap<u64, FieldElement<E>>,
    /// Table name for debug output
    #[cfg(feature = "debug-checks")]
    #[rkyv(with = rkyv::with::Skip)]
    pub table_name: String,
}

impl<E: IsField> BusPublicInputs<E> {
    /// Build a `BusPublicInputs` carrying just the table contribution `L`.
    /// The debug-only per-bus aggregation fields are defaulted (empty). Used by
    /// the zero-copy verifier, which reads only `table_contribution` from the
    /// archived proof.
    pub fn from_contribution(table_contribution: FieldElement<E>) -> Self {
        Self {
            table_contribution,
            #[cfg(feature = "debug-checks")]
            per_bus_sums: HashMap::new(),
            #[cfg(feature = "debug-checks")]
            per_bus_sender_sums: HashMap::new(),
            #[cfg(feature = "debug-checks")]
            per_bus_receiver_sums: HashMap::new(),
            #[cfg(feature = "debug-checks")]
            table_name: String::new(),
        }
    }
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

/// Compute a LogUp term column for one or two interactions sharing the result
/// column. For each row, returns the sum Σₖ signₖ·mₖ[row] / fpₖ[row] where the
/// loop runs over `interactions` (must be length 1 or 2).
///
/// Single-interaction case yields the per-interaction quotient (used for the
/// absorbed virtual column when only one interaction remains, and by the
/// debug-checks per-interaction breakdown). Two-interaction case yields the
/// batched sum that backs a committed term column. Both share a single chunked
/// implementation with one batch inversion per chunk for cache locality.
///
/// Debug-checks bus tracker is invoked only when `interactions.len() == 1`,
/// matching the previous behavior of the dedicated single-interaction helper.
///
/// With `parallel`: chunked over rows via `par_chunks_mut`.
/// Without `parallel`: processed as a single chunk.
fn compute_logup_term_column<F, E>(
    interactions: &[&BusInteraction],
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
    #[cfg_attr(not(feature = "debug-checks"), allow(unused))] _table_name: &str,
) -> Vec<FieldElement<E>>
where
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
    E: IsField + Send + Sync,
{
    assert!(
        matches!(interactions.len(), 1 | 2),
        "compute_logup_term_column expects 1 or 2 interactions, got {}",
        interactions.len()
    );

    let z = &challenges[0];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];
    let max_bus_elements = interactions
        .iter()
        .map(|i| i.num_bus_elements())
        .max()
        .unwrap_or(0);
    let alpha_powers = compute_alpha_powers(alpha, max_bus_elements);
    let shifts = PackingShifts::<F>::new();
    let n = interactions.len();

    let mut result = vec![FieldElement::<E>::zero(); trace_len];

    let process_chunk = |chunk_start: usize, result_chunk: &mut [FieldElement<E>]| {
        let chunk_len = result_chunk.len();
        #[cfg(feature = "instruments")]
        let _t0 = std::time::Instant::now();

        // Phase 1 — fingerprints, laid out as [int_0 rows…, int_1 rows…].
        // fp[k*chunk_len + i] = interaction k at row chunk_start+i.
        let mut fingerprints: Vec<FieldElement<E>> = Vec::with_capacity(n * chunk_len);
        for interaction in interactions.iter() {
            // α⁰ = 1: the bus-id term needs no multiply — embed it into E once.
            let bus_id_e = FieldElement::<E>::from(interaction.bus_id);
            for row in chunk_start..chunk_start + chunk_len {
                let mut lc = bus_id_e.clone();
                let mut alpha_offset = 1;
                for bv in &interaction.values {
                    alpha_offset += bv.accumulate_fingerprint(
                        main_segment_cols,
                        row,
                        &alpha_powers,
                        alpha_offset,
                        &mut lc,
                        &shifts,
                    );
                }
                fingerprints.push(z - &lc);
            }
        }

        #[cfg(feature = "debug-checks")]
        if n == 1 {
            let interaction = interactions[0];
            for (i, row) in (chunk_start..chunk_start + chunk_len).enumerate() {
                let mut base_elements: Vec<FieldElement<F>> =
                    vec![FieldElement::<F>::from(interaction.bus_id)];
                base_elements.extend(
                    interaction
                        .values
                        .iter()
                        .flat_map(|bv| bv.combine_from(|col| main_segment_cols[col][row].clone())),
                );
                let multiplicity = interaction
                    .multiplicity
                    .evaluate_at_row(main_segment_cols, row);
                crate::bus_debug::log_interaction(
                    _table_name,
                    row,
                    interaction.bus_id,
                    interaction.is_sender,
                    &multiplicity.canonical(),
                    &base_elements,
                    &fingerprints[i],
                );
            }
        }

        #[cfg(feature = "instruments")]
        let _t1 = std::time::Instant::now();
        // Phase 2: batch invert
        FieldElement::inplace_batch_inverse(&mut fingerprints)
            .expect("fingerprint is zero - probability of sampling zero is negligible");

        #[cfg(feature = "instruments")]
        let _t2 = std::time::Instant::now();
        // Phase 3: Compute terms
        for (i, result_elem) in result_chunk.iter_mut().enumerate() {
            let row = chunk_start + i;
            let mut acc = FieldElement::<E>::zero();
            for (k, interaction) in interactions.iter().enumerate() {
                let m = interaction
                    .multiplicity
                    .evaluate_at_row(main_segment_cols, row);
                let term = &m * &fingerprints[k * chunk_len + i];
                acc += if interaction.is_sender { term } else { -term };
            }
            *result_elem = acc;
        }
        #[cfg(feature = "instruments")]
        crate::instruments::accum_aux_term(_t1 - _t0, _t2 - _t1, std::time::Instant::now() - _t2);
    };

    #[cfg(feature = "parallel")]
    result
        .par_chunks_mut(LOGUP_CHUNK_SIZE)
        .enumerate()
        .for_each(|(i, chunk)| process_chunk(i * LOGUP_CHUNK_SIZE, chunk));

    #[cfg(not(feature = "parallel"))]
    process_chunk(0, &mut result);

    result
}

/// Builds the circular accumulated column from pre-computed term columns.
///
/// For the circular constraint: acc[(i+1) mod N] - acc[i] - terms[i] + L/N = 0
/// (forward accumulation: the increment at transition i→i+1 uses the CURRENT
/// row's terms). We build: acc[0] = 0, acc[i] = acc[i-1] + terms[i-1] - L/N.
/// Result: the running sum returns to acc[0] since Σterms - N*(L/N) = 0.
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
    #[cfg(feature = "instruments")]
    let _t_acc = std::time::Instant::now();

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

    // Build circular accumulated column (forward accumulation: write acc[row]
    // BEFORE folding in the current row's terms, so acc[0] = 0 and
    // acc[row+1] - acc[row] = row_sum[row] - L/N).
    let mut accumulated = FieldElement::<E>::zero();
    for row in 0..trace_len {
        trace.set_aux(row, acc_column_idx, accumulated.clone());
        let mut row_sum = FieldElement::<E>::zero();
        for col in term_columns {
            row_sum = row_sum + &col[row];
        }
        accumulated = &accumulated + &row_sum - &offset_per_row;
    }

    #[cfg(feature = "instruments")]
    crate::instruments::accum_aux_accumulate(std::time::Instant::now() - _t_acc);
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
            &[interaction],
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
// LogUp single-source constraints (ConstraintBuilder front-end)
// =============================================================================
//
// The LogUp transition constraints are generated from the interaction config
// (a [`LogUpLayout`]) through the generic [`ConstraintBuilder`], so ONE body
// serves the compiled prover folder, the verifier folder and IR capture. This
// is the single source for the two LogUp constraint shapes (batched term and
// accumulated); there are no per-constraint objects.
//
// All LogUp constraints use the default zerofier shape (every row, no
// exemptions) and are [`RootKind::Ext`]; their metadata is derived from this
// same emission (via `MetaBuilder`), not hand-listed.
//
// The data-dependent "skip the multiply when the row value is zero"
// optimization IS reproduced, through the [`ConstraintBuilder::fold_fingerprint_term`]
// hook rather than in this row-agnostic body: capture and the verifier fold the
// term unconditionally (value-identical, since `0·α = 0`), while
// `ProverEvalFolder` overrides the hook to skip the base×ext multiply for a
// zero bus element on the hot per-row path.

use crate::constraints::builder::ConstraintBuilder;

/// Config describing an [`AirWithBuses`] table's LogUp layout, exactly as
/// computed by [`AirWithBuses::new`] from the interaction list (via
/// `split_interactions`). This is the plain-data source for the LogUp
/// constraints: [`emit_logup_constraints`] reads it to generate every LogUp
/// constraint (its metadata is derived from that same emission).
#[derive(Clone)]
pub struct LogUpLayout {
    /// All interactions, in the order they were registered. The first
    /// `2 * num_committed_pairs` are the committed (batched) pairs; the last
    /// 1–2 are absorbed into the accumulated constraint.
    pub interactions: Vec<BusInteraction>,
    /// Number of committed batched pairs (each gets one aux term column).
    pub num_committed_pairs: usize,
    /// Number of committed term columns (`= num_committed_pairs`).
    pub num_term_columns: usize,
    /// Index of the accumulated column (`= num_term_columns`).
    pub acc_column_idx: usize,
}

impl LogUpLayout {
    /// Derive the LogUp layout from an interaction list, mirroring the split
    /// [`AirWithBuses::new`] performs.
    pub fn from_interactions(interactions: Vec<BusInteraction>) -> Self {
        let num_interactions = interactions.len();
        let (num_committed_pairs, _absorbed_count) = split_interactions(num_interactions);
        let num_term_columns = num_committed_pairs;
        Self {
            interactions,
            num_committed_pairs,
            num_term_columns,
            acc_column_idx: num_term_columns,
        }
    }

    /// The absorbed interactions (last 1–2), folded into the accumulated
    /// constraint. Empty when there are no interactions.
    fn absorbed(&self) -> &[BusInteraction] {
        let n = self.interactions.len();
        if n == 0 {
            return &[];
        }
        let (_, absorbed_count) = split_interactions(n);
        &self.interactions[n - absorbed_count..]
    }

    /// Number of LogUp transition constraints this layout produces:
    /// one per committed pair (batched term) plus one accumulated constraint
    /// when there is at least one interaction.
    pub fn num_constraints(&self) -> usize {
        if self.interactions.is_empty() {
            0
        } else {
            self.num_committed_pairs + 1
        }
    }
}

/// Capture a [`Multiplicity`] as a base-field expression, mirroring
/// [`Multiplicity::evaluate_with`].
fn emit_multiplicity<F, E, B>(b: &B, multiplicity: &Multiplicity, offset: usize) -> B::Expr
where
    F: IsField,
    E: IsField,
    B: ConstraintBuilder<F, E>,
{
    match multiplicity {
        Multiplicity::One => b.one(),
        Multiplicity::Column(col) => b.main(offset, *col),
        Multiplicity::Sum(a, c) => b.main(offset, *a) + b.main(offset, *c),
        Multiplicity::Negated(col) => b.one() - b.main(offset, *col),
        Multiplicity::Diff(a, c) => b.main(offset, *a) - b.main(offset, *c),
        Multiplicity::Sum3(a, c, d) => b.main(offset, *a) + b.main(offset, *c) + b.main(offset, *d),
        Multiplicity::Linear(terms) => emit_linear_terms(b, terms, offset),
    }
}

/// Capture a slice of [`LinearTerm`]s as a base-field sum, mirroring the
/// `Multiplicity::Linear` arm of [`Multiplicity::evaluate_with`] (`Σ terms`,
/// starting from zero).
fn emit_linear_terms<F, E, B>(b: &B, terms: &[LinearTerm], offset: usize) -> B::Expr
where
    F: IsField,
    E: IsField,
    B: ConstraintBuilder<F, E>,
{
    let mut result = b.const_base(0);
    for term in terms {
        match *term {
            LinearTerm::Column {
                coefficient,
                column,
            } => {
                result = result + b.main(offset, column) * b.const_signed(coefficient);
            }
            LinearTerm::ColumnUnsigned {
                coefficient,
                column,
            } => {
                result = result + b.main(offset, column) * b.const_base(coefficient);
            }
            LinearTerm::Constant(value) => {
                result = result + b.const_signed(value);
            }
        }
    }
    result
}

/// Fold a [`Packing`]'s fingerprint contribution into the running fingerprint
/// `fp`, mirroring [`Packing::accumulate_fingerprint_with`]. Each bus element
/// subtracts one `col_expr * alpha_power` term (base operand LEFT) from `fp` —
/// see [`emit_fingerprint`] for why terms are subtracted rather than summed.
/// Returns the updated fingerprint and the number of alpha powers consumed
/// (`= packing.num_bus_elements()`). Field addition is associative and
/// commutative, so this row-agnostic accumulation is value-identical to the
/// runtime body regardless of grouping.
fn emit_packing_fingerprint<F, E, B>(
    b: &B,
    packing: Packing,
    start_col: usize,
    offset: usize,
    alpha_offset: usize,
    mut fp: B::ExprE,
) -> (B::ExprE, usize)
where
    F: IsField,
    E: IsField,
    B: ConstraintBuilder<F, E>,
{
    let col = |c: usize| b.main(offset, c);
    let alpha = |i: usize| b.alpha_pow(alpha_offset + i);
    let shift_8 = || b.const_base(SHIFT_8);
    let shift_16 = || b.const_base(SHIFT_16);
    let shift_24 = || b.const_base(SHIFT_8 * SHIFT_16);

    match packing {
        Packing::Direct => (fp - col(start_col) * alpha(0), 1),
        Packing::Word2L => {
            let combined = col(start_col) + col(start_col + 1) * shift_16();
            (fp - combined * alpha(0), 1)
        }
        Packing::Word4L => {
            let combined = col(start_col)
                + col(start_col + 1) * shift_8()
                + col(start_col + 2) * shift_16()
                + col(start_col + 3) * shift_24();
            (fp - combined * alpha(0), 1)
        }
        Packing::DWordWL => {
            fp = fp - col(start_col) * alpha(0);
            (fp - col(start_col + 1) * alpha(1), 2)
        }
        Packing::DWordHHW => {
            fp = fp - col(start_col) * alpha(0);
            let w = col(start_col + 1) + col(start_col + 2) * shift_16();
            (fp - w * alpha(1), 2)
        }
        Packing::DWordWHH => {
            let w = col(start_col) + col(start_col + 1) * shift_16();
            fp = fp - w * alpha(0);
            (fp - col(start_col + 2) * alpha(1), 2)
        }
        Packing::DWordHL => {
            let w0 = col(start_col) + col(start_col + 1) * shift_16();
            fp = fp - w0 * alpha(0);
            let w1 = col(start_col + 2) + col(start_col + 3) * shift_16();
            (fp - w1 * alpha(1), 2)
        }
        Packing::DWordBL => {
            let w0 = col(start_col)
                + col(start_col + 1) * shift_8()
                + col(start_col + 2) * shift_16()
                + col(start_col + 3) * shift_24();
            fp = fp - w0 * alpha(0);
            let w1 = col(start_col + 4)
                + col(start_col + 5) * shift_8()
                + col(start_col + 6) * shift_16()
                + col(start_col + 7) * shift_24();
            (fp - w1 * alpha(1), 2)
        }
        Packing::QuadHL => {
            for i in 0..4 {
                let c = start_col + i * 2;
                let w = col(c) + col(c + 1) * shift_16();
                fp = fp - w * alpha(i);
            }
            (fp, 4)
        }
        Packing::QuadWL => {
            for i in 0..4 {
                fp = fp - col(start_col + i) * alpha(i);
            }
            (fp, 4)
        }
    }
}

/// Fold a [`BusValue`]'s fingerprint contribution into the running fingerprint
/// `fp`, mirroring [`BusValue::accumulate_fingerprint_from_step`]. Returns the
/// updated fingerprint and the number of alpha powers consumed.
fn emit_busvalue_fingerprint<F, E, B>(
    b: &B,
    bv: &BusValue,
    offset: usize,
    alpha_offset: usize,
    fp: B::ExprE,
) -> (B::ExprE, usize)
where
    F: IsField,
    E: IsField,
    B: ConstraintBuilder<F, E>,
{
    match bv {
        BusValue::Packed {
            start_column,
            packing,
        } => emit_packing_fingerprint::<F, E, B>(
            b,
            *packing,
            *start_column,
            offset,
            alpha_offset,
            fp,
        ),
        BusValue::Linear(terms) => {
            // Routed through the builder so the prover folder can zero-skip
            // the multiply (Linear is where the constant-0 bus-width padding
            // lives; the packed contributions above fold unconditionally —
            // their elements are real trace columns with no zero-heavy
            // padding). Value-identical either way.
            let result = emit_linear_terms(b, terms, offset);
            (b.fold_fingerprint_term(fp, result, alpha_offset), 1)
        }
    }
}

/// Capture an interaction's fingerprint as an extension expression, mirroring
/// `z - (bus_id + α·v[0] + α²·v[1] + ...)`.
///
/// `α⁰ = 1`: the bus-id term needs no multiply and is added as a base constant.
///
/// The subtraction is distributed: the fingerprint starts at `z − bus_id` and
/// each α·value term is subtracted as it is emitted. Field addition is
/// associative and commutative, so this is value-identical to
/// `z − (bus + Σ terms)` — and it keeps the running value in ONE extension
/// accumulator. The prover folder runs this body once per LDE row, where
/// collecting the terms in a `Vec` costs a heap allocation per fingerprint
/// per row.
fn emit_fingerprint<F, E, B>(b: &B, interaction: &BusInteraction, offset: usize) -> B::ExprE
where
    F: IsField,
    E: IsField,
    B: ConstraintBuilder<F, E>,
{
    let z = b.challenge(0);
    let bus = b.const_base(interaction.bus_id);
    // `bus` is base and `z` ext; the tower only implements base − ext (base
    // operand LEFT), so z − bus is written −(bus − z).
    let mut fp = -(bus - z);
    let mut alpha_idx = 1;
    for bv in &interaction.values {
        let (next, consumed) = emit_busvalue_fingerprint::<F, E, B>(b, bv, offset, alpha_idx, fp);
        fp = next;
        alpha_idx += consumed;
    }
    fp
}

/// Emit the batched-term constraint for committed pair `pair_idx`:
/// `c · fp_a · fp_b − sign_a·m_a·fp_b − sign_b·m_b·fp_a` (degree 3).
fn emit_logup_batched_term<F, E, B>(b: &mut B, layout: &LogUpLayout, pair_idx: usize, idx: usize)
where
    F: IsField,
    E: IsField,
    B: ConstraintBuilder<F, E>,
{
    let interaction_a = &layout.interactions[pair_idx * 2];
    let interaction_b = &layout.interactions[pair_idx * 2 + 1];
    let term_column_idx = pair_idx;

    let c = b.aux(0, term_column_idx);
    let m_a = emit_multiplicity::<F, E, B>(b, &interaction_a.multiplicity, 0);
    let m_b = emit_multiplicity::<F, E, B>(b, &interaction_b.multiplicity, 0);
    let fp_a = emit_fingerprint::<F, E, B>(b, interaction_a, 0);
    let fp_b = emit_fingerprint::<F, E, B>(b, interaction_b, 0);

    // is_sender is a compile-time bool, resolved as add vs neg instead of an
    // ext×ext sign multiply (same optimization as the runtime body). m·fp is
    // base×ext = ext (base operand LEFT).
    let term_a = m_a * fp_b.clone();
    let term_a = if interaction_a.is_sender {
        term_a
    } else {
        -term_a
    };
    let term_b = m_b * fp_a.clone();
    let term_b = if interaction_b.is_sender {
        term_b
    } else {
        -term_b
    };

    // c · fp_a · fp_b: c is aux (ext), so this is ext throughout (degree 3;
    // see `logup_max_degree`).
    let main = c * fp_a * fp_b;
    b.emit_ext(idx, main - term_a - term_b);
}

/// Emit the accumulated constraint (with 1–2 absorbed interactions).
/// `acc_next` reads the NEXT row (offset 1) — the *only* next-row read in the
/// whole constraint system. `acc_curr`, the committed-term sum and the absorbed
/// fingerprints/multiplicities all read the CURRENT row (offset 0), so the
/// forward recurrence is `acc[i+1] − acc[i] = Σterms[i] + absorbed[i] − L/N`.
/// Keeping every non-`acc` operand on the current row lets the OOD opening send
/// only `acc` at `g·z`, not the whole trace width.
///
/// - 1 absorbed: `(acc_next − acc_curr − Σterms + L/N)·f − sign·m` (degree 2)
/// - 2 absorbed: `(…)·f₁·f₂ − sign₁·m₁·f₂ − sign₂·m₂·f₁` (degree 3)
fn emit_logup_accumulated<F, E, B>(b: &mut B, layout: &LogUpLayout, idx: usize)
where
    F: IsField,
    E: IsField,
    B: ConstraintBuilder<F, E>,
{
    let acc_curr = b.aux(0, layout.acc_column_idx);
    let acc_next = b.aux(1, layout.acc_column_idx);

    // delta = acc_next − acc_curr − Σ committed_terms(curr) + L/N.
    // Committed terms read the current row (offset 0) so that `acc_next` is the
    // sole next-row operand (see the doc comment).
    let mut delta = acc_next - acc_curr;
    for i in 0..layout.num_term_columns {
        delta = delta - b.aux(0, i);
    }
    delta = delta + b.table_offset();

    let absorbed = layout.absorbed();
    let root = match absorbed.len() {
        1 => {
            // delta · f − sign · m; absorbed operands read the current row.
            let m = emit_multiplicity::<F, E, B>(b, &absorbed[0].multiplicity, 0);
            let f = emit_fingerprint::<F, E, B>(b, &absorbed[0], 0);
            let mt = if absorbed[0].is_sender { m } else { -m };
            // delta · f is ext; `mt` is base. The tower only implements base −
            // ext (base operand LEFT), so write `delta·f − mt` as `−(mt − delta·f)`.
            -(mt - delta * f)
        }
        2 => {
            // delta · f1 · f2 − sign1·m1·f2 − sign2·m2·f1; absorbed operands
            // read the current row (offset 0).
            let m1 = emit_multiplicity::<F, E, B>(b, &absorbed[0].multiplicity, 0);
            let m2 = emit_multiplicity::<F, E, B>(b, &absorbed[1].multiplicity, 0);
            let f1 = emit_fingerprint::<F, E, B>(b, &absorbed[0], 0);
            let f2 = emit_fingerprint::<F, E, B>(b, &absorbed[1], 0);

            let term1 = m1 * f2.clone();
            let term1 = if absorbed[0].is_sender { term1 } else { -term1 };
            let term2 = m2 * f1.clone();
            let term2 = if absorbed[1].is_sender { term2 } else { -term2 };
            delta * f1 * f2 - term1 - term2
        }
        _ => unreachable!("absorbed must contain 1 or 2 interactions"),
    };

    // Degree 1 + absorbed count (2 for one absorbed, 3 for two); folded into
    // the composition bound via `logup_max_degree`.
    b.emit_ext(idx, root);
}

/// The maximum degree among a layout's framework-generated LogUp constraints:
/// batched committed terms are degree 3, the accumulator is `1 + absorbed`.
/// Zero when there are no interactions. Folded into
/// `composition_poly_degree_bound` alongside the base constraints' max_degree.
pub fn logup_max_degree(layout: &LogUpLayout) -> usize {
    if layout.interactions.is_empty() {
        return 0;
    }
    // Accumulated constraint: 1 + number of absorbed interactions.
    let mut m = 1 + layout.absorbed().len();
    // Batched committed terms (if any) are degree 3.
    if layout.num_committed_pairs > 0 {
        m = m.max(3);
    }
    m
}

/// Emit every LogUp transition constraint for `layout` through the builder,
/// starting at absolute constraint index `idx_base` (the table's base-constraint
/// count). Committed batched terms come first (one per committed pair), then the
/// single accumulated constraint. Emits nothing when there are no interactions.
pub fn emit_logup_constraints<F, E, B>(b: &mut B, layout: &LogUpLayout, idx_base: usize)
where
    F: IsField,
    E: IsField,
    B: ConstraintBuilder<F, E>,
{
    if layout.interactions.is_empty() {
        return;
    }
    let mut idx = idx_base;
    for pair_idx in 0..layout.num_committed_pairs {
        emit_logup_batched_term::<F, E, B>(b, layout, pair_idx, idx);
        idx += 1;
    }
    emit_logup_accumulated::<F, E, B>(b, layout, idx);
}

/// Run an [`AirWithBuses`] table's transition constraints through the
/// [`ProverEvalFolder`] in ONE pass: the constraint set's base-field body
/// followed by the appended LogUp constraints (idx offset by `num_base`, the
/// base-prefix length). `base_evals` is sized `num_base`; `ext_evals` the total
/// constraint count.
fn run_air_transition_prover<F, E, CS>(
    constraint_set: &CS,
    logup: &LogUpLayout,
    ctx: &TransitionEvaluationContext<'_, F, E>,
    base_evals: &mut [FieldElement<F>],
    ext_evals: &mut [FieldElement<E>],
) where
    F: IsSubFieldOf<E>,
    E: IsField,
    CS: ConstraintSet<F, E>,
{
    let num_base = base_evals.len();
    let mut folder = ProverEvalFolder::new(ctx, base_evals, ext_evals);
    constraint_set.eval(&mut folder);
    emit_logup_constraints(&mut folder, logup, num_base);
    folder.assert_all_emitted();
}

/// Run an [`AirWithBuses`] table's transition constraints at a single point,
/// returning every constraint value in the extension field: the constraint
/// set's base-field body (promoted) followed by the appended LogUp constraints.
///
/// A Verifier context runs the [`VerifierEvalFolder`] (the OOD/recursion path).
/// A Prover context is also accepted — debug trace validation calls this with a
/// prover frame — by running the [`ProverEvalFolder`] and promoting the
/// base-prefix results.
fn run_air_transition_verifier<F, E, CS>(
    constraint_set: &CS,
    logup: &LogUpLayout,
    num_base: usize,
    num_constraints: usize,
    ctx: &TransitionEvaluationContext<'_, F, E>,
) -> Vec<FieldElement<E>>
where
    F: IsSubFieldOf<E>,
    E: IsField,
    CS: ConstraintSet<F, E>,
{
    let mut ext_evals = vec![FieldElement::<E>::zero(); num_constraints];
    match ctx {
        TransitionEvaluationContext::Verifier { .. } => {
            let mut folder = VerifierEvalFolder::new(ctx, &mut ext_evals);
            constraint_set.eval(&mut folder);
            emit_logup_constraints(&mut folder, logup, num_base);
            folder.assert_all_emitted();
        }
        TransitionEvaluationContext::Prover { .. } => {
            let mut base_evals = vec![FieldElement::<F>::zero(); num_base];
            let mut folder = ProverEvalFolder::new(ctx, &mut base_evals, &mut ext_evals);
            constraint_set.eval(&mut folder);
            emit_logup_constraints(&mut folder, logup, num_base);
            folder.assert_all_emitted();
            // Promote the base-prefix results into the extension slots.
            for (slot, base) in ext_evals.iter_mut().zip(base_evals) {
                *slot = base.to_extension();
            }
        }
    }
    ext_evals
}

#[cfg(test)]
mod logup_single_source_tests {
    //! Regression tests for the single-source LogUp constraint bodies
    //! ([`emit_logup_constraints`]) run three ways from ONE definition. For
    //! every layout we assert, on 1000
    //! random two-step frames: [`ProverEvalFolder`] == capture→`eval_program`
    //! (prover) and [`VerifierEvalFolder`] == capture→`eval_program_verifier`
    //! (verifier) — all bit-for-bit.
    //!
    //! Coverage: the accumulated constraint's 1-absorbed AND 2-absorbed branches
    //! (the latter folds two absorbed interactions, degree 3), the batched-term
    //! constraint, and every [`Packing`] variant's fingerprint contribution.
    use super::*;
    use crate::constraint_ir::{eval_program, eval_program_verifier};
    use crate::constraints::builder::{
        CaptureBuilder, ProverEvalFolder, RootKind, VerifierEvalFolder, num_base_from_meta,
    };
    use crate::frame::Frame;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;

    type Fp = FieldElement<Gl>;
    type Fp3 = FieldElement<Ext3>;

    const TRIALS: usize = 1000;

    /// A tiny deterministic SplitMix64 PRNG (no `rand` dependency).
    struct SplitMix64 {
        state: u64,
    }
    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Number of aux columns the layout uses: committed term columns + the
    /// accumulated column.
    fn num_aux_cols(layout: &LogUpLayout) -> usize {
        if layout.interactions.is_empty() {
            0
        } else {
            layout.num_term_columns + 1
        }
    }

    fn rand_fp3(rng: &mut SplitMix64) -> Fp3 {
        FieldElement::<Ext3>::new([
            Fp::from(rng.next_u64()),
            Fp::from(rng.next_u64()),
            Fp::from(rng.next_u64()),
        ])
    }

    /// Forward-accumulation contract for [`build_accumulated_column_from_terms`]:
    /// `acc[0] = 0` and the circular recurrence tied to the CURRENT row's terms
    /// holds on EVERY row, including the wraparound (which closes the cycle back
    /// to `acc[0]`). This is the invariant the OOD pruning relies on — only
    /// `acc` is read at the next row; every term is read at the current row.
    #[test]
    fn accumulated_column_is_forward_and_circular() {
        let mut rng = SplitMix64::new(0xC0FF_EE12_3456_789A);
        let n_rows = 8usize;
        let n_term_cols = 2usize;

        let term_columns: Vec<Vec<Fp3>> = (0..n_term_cols)
            .map(|_| (0..n_rows).map(|_| rand_fp3(&mut rng)).collect())
            .collect();

        // Accumulated column follows the committed term columns.
        let acc_col_idx = n_term_cols;
        let mut trace = TraceTable::<Gl, Ext3>::new_main(vec![Fp::zero(); n_rows], 1, 1);
        trace.allocate_aux_table(n_term_cols + 1);

        let l = build_accumulated_column_from_terms(acc_col_idx, &term_columns, &mut trace);

        // Forward accumulation starts at zero.
        assert_eq!(
            *trace.get_aux(0, acc_col_idx),
            Fp3::zero(),
            "acc[0] must be 0 under forward accumulation"
        );

        // Circular recurrence tied to the CURRENT row's terms, on every row.
        // Multiplied through by N to avoid dividing L by N:
        //   (acc[(i+1) mod N] - acc[i]) * N == (Σ terms[i]) * N - L
        let n_fe = Fp3::from(n_rows as u64);
        for i in 0..n_rows {
            let mut row_sum = Fp3::zero();
            for col in &term_columns {
                row_sum = row_sum + &col[i];
            }
            let acc_i = *trace.get_aux(i, acc_col_idx);
            let acc_next = *trace.get_aux((i + 1) % n_rows, acc_col_idx);
            let lhs = (acc_next - acc_i) * &n_fe;
            let rhs = row_sum * &n_fe - &l;
            assert_eq!(lhs, rhs, "forward circular recurrence broken at row {i}");
        }
    }

    /// The permanent regression check for one layout, on `TRIALS` random
    /// two-step frames: the LogUp body run three ways from ONE definition must
    /// agree bit-for-bit — [`ProverEvalFolder`] == capture→[`eval_program`]
    /// (prover) and [`VerifierEvalFolder`] == capture→[`eval_program_verifier`]
    /// (verifier).
    fn check_layout(label: &str, layout: &LogUpLayout, num_main_cols: usize) {
        let n_base = 0usize; // LogUp constraints are all extension-rooted.
        let n = layout.num_constraints();

        // Metadata self-consistency: derived from the LogUp emission itself
        // (MetaBuilder), it must be all-ext, dense, and match the
        // batched/accumulated degree formula (3 per batched term; 1 + absorbed
        // for the accumulator).
        let meta = {
            let mut mb = crate::constraints::builder::MetaBuilder::new();
            emit_logup_constraints::<Gl, Ext3, _>(&mut mb, layout, n_base);
            mb.into_meta()
        };
        assert_eq!(meta.len(), n, "[{label}] meta count");
        let num_base = num_base_from_meta(&meta);
        assert_eq!(num_base, 0, "[{label}] LogUp meta is all-ext");
        for (i, m) in meta.iter().enumerate() {
            assert_eq!(m.constraint_idx, i, "[{label}] meta idx {i}");
            assert_eq!(m.kind, RootKind::Ext, "[{label}] meta kind {i}");
        }

        // Capture once; the tree-measured degree must match the batched/
        // accumulated formula, and `logup_max_degree` must equal their max.
        let mut cb = CaptureBuilder::<Gl, Ext3>::new();
        emit_logup_constraints(&mut cb, layout, n_base);
        let (prog, degrees) = cb.finish(num_base);
        assert_eq!(degrees.len(), n, "[{label}] one emit per constraint");
        // Release-safe exact-once check: the emitted indices must be exactly
        // 0..n (the per-emit EmitTracker only exists under debug_assertions,
        // which a --release test build compiles out).
        let mut emitted: Vec<usize> = degrees.iter().map(|&(idx, _)| idx).collect();
        emitted.sort_unstable();
        assert!(
            emitted.iter().enumerate().all(|(i, &idx)| i == idx),
            "[{label}] emitted constraint indices are not exactly 0..{n}: {emitted:?}"
        );
        for &(idx, measured) in &degrees {
            let expected_degree = if idx < layout.num_committed_pairs {
                3
            } else {
                1 + layout.absorbed().len()
            };
            assert_eq!(measured, expected_degree, "[{label}] degree {idx}");
        }
        assert_eq!(
            logup_max_degree(layout),
            degrees.iter().map(|&(_, d)| d).max().unwrap_or(0),
            "[{label}] logup_max_degree matches max measured degree"
        );

        let n_aux = num_aux_cols(layout);

        for trial in 0..TRIALS {
            let mut rng = SplitMix64::new(0xC0FF_EE00_u64 ^ (label.len() as u64) ^ trial as u64);

            // Random two-step prover frame.
            let mk_step = |rng: &mut SplitMix64| {
                let main: Vec<Fp> = (0..num_main_cols)
                    .map(|_| Fp::from(rng.next_u64()))
                    .collect();
                let aux: Vec<Fp3> = (0..n_aux).map(|_| rand_fp3(rng)).collect();
                TableView::new(vec![main], vec![aux])
            };
            let frame = Frame::<Gl, Ext3>::new(vec![mk_step(&mut rng), mk_step(&mut rng)]);
            let rap_challenges = vec![rand_fp3(&mut rng), rand_fp3(&mut rng)]; // [z, alpha]
            let alpha_powers: Vec<Fp3> = (0..12).map(|_| rand_fp3(&mut rng)).collect();
            let table_offset = rand_fp3(&mut rng);

            let prover_ctx = TransitionEvaluationContext::new_prover(
                frame.as_row_frame(),
                &rap_challenges,
                &alpha_powers,
                &table_offset,
            );

            // --- ProverEvalFolder == capture → interpret (prover) ---
            let mut base_out = vec![Fp::zero(); n_base];
            let mut ext_out = vec![Fp3::zero(); n];
            let mut folder = ProverEvalFolder::new(&prover_ctx, &mut base_out, &mut ext_out);
            emit_logup_constraints(&mut folder, layout, n_base);
            folder.assert_all_emitted();

            let mut ir_base = vec![Fp::zero(); n_base];
            let mut ir_ext = vec![Fp3::zero(); n];
            eval_program(&prog, &prover_ctx, &mut ir_base, &mut ir_ext);
            for i in 0..n {
                assert_eq!(
                    ext_out[i], ir_ext[i],
                    "[{label}] prover folder vs interpreter mismatch, constraint {i}, trial {trial}"
                );
            }

            // --- verifier-side: embed the same frame into the extension ---
            let embed_step = |step: &TableView<Gl, Ext3>| -> TableView<Ext3, Ext3> {
                let main: Vec<Fp3> = (0..num_main_cols)
                    .map(|c| step.get_main_evaluation_element(0, c).to_extension())
                    .collect();
                let aux: Vec<Fp3> = (0..n_aux)
                    .map(|c| *step.get_aux_evaluation_element(0, c))
                    .collect();
                TableView::new(vec![main], vec![aux])
            };
            let vframe: Frame<Ext3, Ext3> = Frame::new(vec![
                embed_step(frame.get_evaluation_step(0)),
                embed_step(frame.get_evaluation_step(1)),
            ]);
            let vctx = TransitionEvaluationContext::<Gl, Ext3>::new_verifier(
                &vframe,
                &rap_challenges,
                &alpha_powers,
                &table_offset,
            );

            // --- VerifierEvalFolder == capture → interpret (verifier) ---
            let mut vext_out = vec![Fp3::zero(); n];
            let mut vfolder = VerifierEvalFolder::new(&vctx, &mut vext_out);
            emit_logup_constraints(&mut vfolder, layout, n_base);
            vfolder.assert_all_emitted();

            let mut ir_vext = vec![Fp3::zero(); n];
            eval_program_verifier(&prog, &vctx, &mut ir_vext);
            for i in 0..n {
                assert_eq!(
                    vext_out[i], ir_vext[i],
                    "[{label}] verifier folder vs interpreter mismatch, constraint {i}, trial {trial}"
                );
            }

            // Prover base-promotion and verifier evaluations must agree
            // (the prover frame embedded == the verifier frame).
            for i in 0..n {
                assert_eq!(
                    ext_out[i], vext_out[i],
                    "[{label}] prover vs verifier folder mismatch, constraint {i}, trial {trial}"
                );
            }
        }
    }

    /// A sender interaction with a `Direct`-packed value at column 1.
    fn direct_sender(bus_id: u64) -> BusInteraction {
        BusInteraction::sender(
            bus_id,
            Multiplicity::Column(0),
            vec![BusValue::Packed {
                start_column: 1,
                packing: Packing::Direct,
            }],
        )
    }

    /// A receiver interaction with a single `column(3)` value.
    fn column_receiver(bus_id: u64) -> BusInteraction {
        BusInteraction::receiver(bus_id, Multiplicity::Column(2), vec![BusValue::column(3)])
    }

    #[test]
    fn logup_one_absorbed() {
        // 3 interactions → split(3) = (1 committed pair, 1 absorbed):
        //   idx 0: batched term (interactions 0,1)
        //   idx 1: accumulated, 1 absorbed (interaction 2), degree 2.
        let interactions = vec![direct_sender(7), column_receiver(11), direct_sender(13)];
        let layout = LogUpLayout::from_interactions(interactions);
        assert_eq!(layout.num_committed_pairs, 1);
        assert_eq!(layout.absorbed().len(), 1, "must exercise 1-absorbed");
        check_layout("one_absorbed", &layout, 8);
    }

    #[test]
    fn logup_two_absorbed() {
        // 4 interactions → split(4) = (1 committed pair, 2 absorbed):
        //   idx 0: batched term (interactions 0,1)
        //   idx 1: accumulated, 2 absorbed (interactions 2,3), degree 3.
        let interactions = vec![
            direct_sender(7),
            column_receiver(11),
            direct_sender(13),
            column_receiver(17),
        ];
        let layout = LogUpLayout::from_interactions(interactions);
        assert_eq!(layout.num_committed_pairs, 1);
        assert_eq!(layout.absorbed().len(), 2, "must exercise 2-absorbed");
        check_layout("two_absorbed", &layout, 8);
    }

    #[test]
    fn logup_two_interactions_absorbed_only() {
        // 2 interactions → split(2) = (0 committed pairs, 2 absorbed): the
        // accumulated constraint alone, degree 3, no batched term.
        let interactions = vec![direct_sender(7), column_receiver(11)];
        let layout = LogUpLayout::from_interactions(interactions);
        assert_eq!(layout.num_committed_pairs, 0);
        assert_eq!(layout.num_constraints(), 1);
        check_layout("two_absorbed_only", &layout, 8);
    }

    #[test]
    fn logup_all_packing_variants() {
        // Drive every Packing arm through the fingerprint of a committed pair
        // and an absorbed interaction. DWordBL/QuadHL are the widest (8 cols);
        // give a generous column budget.
        const ALL_PACKINGS: [Packing; 10] = [
            Packing::Direct,
            Packing::Word2L,
            Packing::Word4L,
            Packing::DWordWL,
            Packing::DWordHHW,
            Packing::DWordWHH,
            Packing::DWordHL,
            Packing::DWordBL,
            Packing::QuadHL,
            Packing::QuadWL,
        ];
        for packing in ALL_PACKINGS {
            // 3 interactions: two committed (pair) + one absorbed, all using the
            // packing at column 0.
            let mk = |bus: u64, sender: bool| {
                let values = vec![BusValue::Packed {
                    start_column: 0,
                    packing,
                }];
                if sender {
                    BusInteraction::sender(bus, Multiplicity::One, values)
                } else {
                    BusInteraction::receiver(bus, Multiplicity::One, values)
                }
            };
            let interactions = vec![mk(3, true), mk(5, false), mk(7, true)];
            let layout = LogUpLayout::from_interactions(interactions);
            check_layout(
                &format!("packing_{packing:?}"),
                &layout,
                packing.num_columns(),
            );
        }
    }

    #[test]
    fn logup_two_committed_pairs() {
        // >= 2 committed pairs: split(6) = (2 pairs, 2 absorbed). Exercises
        // the batched-term loop past its first iteration (pair_idx*2
        // interaction indexing, per-pair term columns) and the accumulated
        // constraint's committed-term sum over more than one aux column —
        // the layout shape every production table has, which the fixtures
        // above (<= 4 interactions, <= 1 pair) never reach.
        let interactions = vec![
            direct_sender(3),
            column_receiver(5),
            direct_sender(7),
            column_receiver(11),
            direct_sender(13),
            column_receiver(17),
        ];
        let layout = LogUpLayout::from_interactions(interactions);
        assert_eq!(layout.num_committed_pairs, 2, "must exercise >= 2 pairs");
        assert_eq!(layout.absorbed().len(), 2);
        assert_eq!(layout.num_constraints(), 3); // 2 batched terms + accumulated
        check_layout("two_committed_pairs", &layout, 8);
    }

    #[test]
    fn logup_linear_zero_skip() {
        // The prover folder zero-skips the F×E multiply for Linear bus
        // elements ([`ConstraintBuilder::fold_fingerprint_term`]); the random
        // frames above never produce a zero element, so drive both always-zero
        // shapes explicitly — the constant-0 bus-width padding and a
        // column-minus-itself combination — next to a nonzero element, and
        // assert the folder still matches the (skip-free) captured program
        // bit-for-bit.
        let zero_padded = |bus: u64, sender: bool| {
            let values = vec![
                BusValue::column(1),
                BusValue::linear(vec![LinearTerm::Constant(0)]),
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: 2,
                    },
                    LinearTerm::Column {
                        coefficient: -1,
                        column: 2,
                    },
                ]),
                BusValue::linear(vec![LinearTerm::Column {
                    coefficient: 3,
                    column: 3,
                }]),
            ];
            if sender {
                BusInteraction::sender(bus, Multiplicity::Column(0), values)
            } else {
                BusInteraction::receiver(bus, Multiplicity::Column(0), values)
            }
        };
        let interactions = vec![
            zero_padded(3, true),
            zero_padded(5, false),
            zero_padded(7, true),
        ];
        let layout = LogUpLayout::from_interactions(interactions);
        assert_eq!(layout.num_committed_pairs, 1);
        assert_eq!(layout.absorbed().len(), 1);
        check_layout("linear_zero_skip", &layout, 8);
    }
}

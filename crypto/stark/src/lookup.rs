#[cfg(feature = "debug-checks")]
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        transition::TransitionConstraintEvaluator,
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
pub struct AirWithBuses<
    F: IsFFTField + IsSubFieldOf<E> + IsPrimeField + Send + Sync,
    E: IsField + Send + Sync,
    B: BoundaryConstraintBuilder<F, E, PI>,
    PI,
> {
    context: AirContext,
    step_size: usize,
    trace_layout: (usize, usize),
    transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>>,
    /// Number of domain (base-field) constraints. These come before LogUp constraints
    /// in the transition_constraints vec and use the cheaper F×E accumulation path.
    num_base_constraints: usize,
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
        mut transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>>,
    ) -> Self {
        // Domain constraints are passed in first; LogUp constraints are appended below.
        // The domain constraints use the F×E accumulation path (3 muls vs 9).
        let num_base_constraints = transition_constraints.len();

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
            num_base_constraints,
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
        self.num_base_constraints
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>> {
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
        let _table_name = self.name.as_deref().unwrap_or("UNKNOWN");

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
        let committed_columns: Vec<Vec<FieldElement<E>>> = if trace_len <= LOGUP_CHUNK_SIZE {
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
    fn evaluate_at_row<F: IsField>(
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

        // Phase 2: batch invert
        FieldElement::inplace_batch_inverse(&mut fingerprints)
            .expect("fingerprint is zero - probability of sampling zero is negligible");

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

/// Computes multiplicity for an interaction from a `TableView`.
fn compute_multiplicity_from_step<A: IsSubFieldOf<B>, B: IsField>(
    step: &TableView<A, B>,
    multiplicity: &Multiplicity,
) -> FieldElement<A> {
    multiplicity.evaluate_with(|col| step.get_main_evaluation_element(0, col).clone())
}

/// Computes the fingerprint for an interaction from a `TableView`.
///
/// Returns `z - (bus_id + α·v[0] + α²·v[1] + ...)`
fn compute_fingerprint_from_step<A: IsSubFieldOf<B>, B: IsField>(
    step: &TableView<A, B>,
    interaction: &BusInteraction,
    z: &FieldElement<B>,
    alpha_powers: &[FieldElement<B>],
    shifts: &PackingShifts<A>,
) -> FieldElement<B> {
    // α⁰ = 1: the bus-id term needs no multiply — embed it into B directly.
    let mut linear_combination = FieldElement::<B>::from(interaction.bus_id);
    let mut alpha_idx = 1;
    for bv in &interaction.values {
        alpha_idx += bv.accumulate_fingerprint_from_step(
            step,
            alpha_powers,
            alpha_idx,
            &mut linear_combination,
            shifts,
        );
    }
    z - &linear_combination
}

// =============================================================================
// LogUp single-source constraints (ConstraintBuilder front-end)
// =============================================================================
//
// The LogUp transition constraints are generated from the interaction config
// (a [`LogUpLayout`]) through the generic [`ConstraintBuilder`], so ONE body
// serves the compiled prover folder, the verifier folder and IR capture. These
// are the single-source twins of the boxed `LookupBatchedTermConstraint` /
// `LookupAccumulatedConstraint` structs (which stay for now as the differential
// oracle; the engine switch deletes them).
//
// All LogUp constraints use the default zerofier shape (every row, no
// exemptions) — the structs override none of period/offset/exemptions — so
// [`logup_meta`] emits plain [`RootKind::Ext`] entries.
//
// Honesty note (matches the runtime body): `BusValue::Linear`'s data-dependent
// "skip the multiply when the row value is zero" optimization is NOT reproduced
// here — the constraint body is row-agnostic and always emits the multiply.
// This is value-preserving (adding `0·α` is a no-op) and only costs a few extra
// base×ext muls per row.

use crate::constraints::builder::{ConstraintBuilder, ConstraintMeta};

/// Config describing an [`AirWithBuses`] table's LogUp layout, exactly as
/// computed by [`AirWithBuses::new`] from the interaction list (via
/// [`split_interactions`]). This is the plain-data replacement for the
/// per-constraint `LookupBatchedTermConstraint` / `LookupAccumulatedConstraint`
/// objects: [`emit_logup_constraints`] reads it to generate every LogUp
/// constraint, and [`logup_meta`] reads it for the metadata.
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
/// [`Multiplicity::evaluate_with`] (via [`compute_multiplicity_from_step`]).
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

/// Capture a [`Packing`]'s fingerprint contribution as a sum of extension
/// terms, mirroring [`Packing::accumulate_fingerprint_with`]. Terms are pushed
/// to `acc` (`col_expr * alpha_power`, base operand LEFT); returns the number
/// of alpha powers consumed (`= packing.num_bus_elements()`). Field addition is
/// associative and commutative, so this row-agnostic accumulation is
/// value-identical to the runtime body regardless of grouping.
fn emit_packing_fingerprint<F, E, B>(
    b: &B,
    packing: Packing,
    start_col: usize,
    offset: usize,
    alpha_offset: usize,
    acc: &mut Vec<B::ExprE>,
) -> usize
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
        Packing::Direct => {
            acc.push(col(start_col) * alpha(0));
            1
        }
        Packing::Word2L => {
            let combined = col(start_col) + col(start_col + 1) * shift_16();
            acc.push(combined * alpha(0));
            1
        }
        Packing::Word4L => {
            let combined = col(start_col)
                + col(start_col + 1) * shift_8()
                + col(start_col + 2) * shift_16()
                + col(start_col + 3) * shift_24();
            acc.push(combined * alpha(0));
            1
        }
        Packing::DWordWL => {
            acc.push(col(start_col) * alpha(0));
            acc.push(col(start_col + 1) * alpha(1));
            2
        }
        Packing::DWordHHW => {
            acc.push(col(start_col) * alpha(0));
            let w = col(start_col + 1) + col(start_col + 2) * shift_16();
            acc.push(w * alpha(1));
            2
        }
        Packing::DWordWHH => {
            let w = col(start_col) + col(start_col + 1) * shift_16();
            acc.push(w * alpha(0));
            acc.push(col(start_col + 2) * alpha(1));
            2
        }
        Packing::DWordHL => {
            let w0 = col(start_col) + col(start_col + 1) * shift_16();
            acc.push(w0 * alpha(0));
            let w1 = col(start_col + 2) + col(start_col + 3) * shift_16();
            acc.push(w1 * alpha(1));
            2
        }
        Packing::DWordBL => {
            let w0 = col(start_col)
                + col(start_col + 1) * shift_8()
                + col(start_col + 2) * shift_16()
                + col(start_col + 3) * shift_24();
            acc.push(w0 * alpha(0));
            let w1 = col(start_col + 4)
                + col(start_col + 5) * shift_8()
                + col(start_col + 6) * shift_16()
                + col(start_col + 7) * shift_24();
            acc.push(w1 * alpha(1));
            2
        }
        Packing::QuadHL => {
            for i in 0..4 {
                let c = start_col + i * 2;
                let w = col(c) + col(c + 1) * shift_16();
                acc.push(w * alpha(i));
            }
            4
        }
        Packing::QuadWL => {
            for i in 0..4 {
                acc.push(col(start_col + i) * alpha(i));
            }
            4
        }
    }
}

/// Capture a [`BusValue`]'s fingerprint contribution into `acc`, mirroring
/// [`BusValue::accumulate_fingerprint_from_step`]. Returns the number of alpha
/// powers consumed.
fn emit_busvalue_fingerprint<F, E, B>(
    b: &B,
    bv: &BusValue,
    offset: usize,
    alpha_offset: usize,
    acc: &mut Vec<B::ExprE>,
) -> usize
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
            acc,
        ),
        BusValue::Linear(terms) => {
            // Value-preserving: always emit the multiply (see the module note).
            let result = emit_linear_terms(b, terms, offset);
            acc.push(result * b.alpha_pow(alpha_offset));
            1
        }
    }
}

/// Capture an interaction's fingerprint as an extension expression, mirroring
/// [`compute_fingerprint_from_step`]: `z - (bus_id + α·v[0] + α²·v[1] + ...)`.
///
/// `α⁰ = 1`: the bus-id term needs no multiply and is added as a base constant.
fn emit_fingerprint<F, E, B>(b: &B, interaction: &BusInteraction, offset: usize) -> B::ExprE
where
    F: IsField,
    E: IsField,
    B: ConstraintBuilder<F, E>,
{
    let z = b.challenge(0);
    let bus = b.const_base(interaction.bus_id);
    // Collect the α·value terms, then fold. Field addition is associative and
    // commutative, so the grouping does not change the value.
    let mut terms: Vec<B::ExprE> = Vec::new();
    let mut alpha_idx = 1;
    for bv in &interaction.values {
        alpha_idx += emit_busvalue_fingerprint::<F, E, B>(b, bv, offset, alpha_idx, &mut terms);
    }
    // lc = bus_id + Σ terms  (base + ext = ext, base operand LEFT).
    let mut iter = terms.into_iter();
    let lc = match iter.next() {
        Some(first) => {
            let mut lc = bus + first;
            for t in iter {
                lc = lc + t;
            }
            lc
        }
        // No values: fingerprint is z - bus_id. `bus` is base, `z` is ext, and
        // the tower only implements base − ext (base operand LEFT), so write
        // z − bus as −(bus − z).
        None => return -(bus - z),
    };
    z - lc
}

/// Emit the batched-term constraint for committed pair `pair_idx`, mirroring
/// `LookupBatchedTermConstraint::capture`:
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

    // c · fp_a · fp_b: c is aux (ext), so this is ext throughout.
    let main = c * fp_a * fp_b;
    b.emit_ext(idx, main - term_a - term_b);
}

/// Emit the accumulated constraint (with 1–2 absorbed interactions), mirroring
/// `LookupAccumulatedConstraint::capture`. `acc_curr` reads row 0; `acc_next`,
/// the committed-term sum and the absorbed fingerprints/multiplicities all read
/// the NEXT row (offset 1).
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

    // delta = acc_next − acc_curr − Σ committed_terms(next) + L/N
    let mut delta = acc_next - acc_curr;
    for i in 0..layout.num_term_columns {
        delta = delta - b.aux(1, i);
    }
    delta = delta + b.table_offset();

    let absorbed = layout.absorbed();
    let root = match absorbed.len() {
        1 => {
            // delta · f − sign · m
            let m = emit_multiplicity::<F, E, B>(b, &absorbed[0].multiplicity, 1);
            let f = emit_fingerprint::<F, E, B>(b, &absorbed[0], 1);
            let mt = if absorbed[0].is_sender { m } else { -m };
            // delta · f is ext; `mt` is base. The tower only implements base −
            // ext (base operand LEFT), so write `delta·f − mt` as `−(mt − delta·f)`.
            -(mt - delta * f)
        }
        2 => {
            // delta · f1 · f2 − sign1·m1·f2 − sign2·m2·f1
            let m1 = emit_multiplicity::<F, E, B>(b, &absorbed[0].multiplicity, 1);
            let m2 = emit_multiplicity::<F, E, B>(b, &absorbed[1].multiplicity, 1);
            let f1 = emit_fingerprint::<F, E, B>(b, &absorbed[0], 1);
            let f2 = emit_fingerprint::<F, E, B>(b, &absorbed[1], 1);

            let term1 = m1 * f2.clone();
            let term1 = if absorbed[0].is_sender { term1 } else { -term1 };
            let term2 = m2 * f1.clone();
            let term2 = if absorbed[1].is_sender { term2 } else { -term2 };
            delta * f1 * f2 - term1 - term2
        }
        _ => unreachable!("absorbed must contain 1 or 2 interactions"),
    };

    b.emit_ext(idx, root);
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

/// The idx-ordered [`ConstraintMeta`] for `layout`'s LogUp constraints, starting
/// at `idx_start`. Reproduces the boxed structs' answers exactly: batched terms
/// are degree 3; the accumulated constraint is degree `1 + absorbed.len()`
/// (2 for one absorbed, 3 for two). All are [`RootKind::Ext`] with the default
/// zerofier shape (period 1, offset 0, no exemptions) — the structs override
/// none of those.
pub fn logup_meta(layout: &LogUpLayout, idx_start: usize) -> Vec<ConstraintMeta> {
    let mut meta = Vec::with_capacity(layout.num_constraints());
    if layout.interactions.is_empty() {
        return meta;
    }
    let mut idx = idx_start;
    for _ in 0..layout.num_committed_pairs {
        meta.push(ConstraintMeta::ext(idx, 3)); // c · fp_a · fp_b
        idx += 1;
    }
    let absorbed_len = layout.absorbed().len();
    meta.push(ConstraintMeta::ext(idx, 1 + absorbed_len));
    meta
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

impl<F, E> TransitionConstraintEvaluator<F, E> for LookupBatchedTermConstraint
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

    fn evaluate_verifier(
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
            shifts: &PackingShifts<A>,
        ) -> FieldElement<B> {
            let c = step.get_aux_evaluation_element(0, term_column_idx);
            let z = &rap_challenges[0];

            let m_a = compute_multiplicity_from_step(step, &interaction_a.multiplicity);
            let m_b = compute_multiplicity_from_step(step, &interaction_b.multiplicity);

            let fp_a = compute_fingerprint_from_step(step, interaction_a, z, alpha_powers, shifts);
            let fp_b = compute_fingerprint_from_step(step, interaction_b, z, alpha_powers, shifts);

            // c * fp_a * fp_b - sign_a * m_a * fp_b - sign_b * m_b * fp_a = 0
            // Use conditional negation instead of E×E sign multiplication
            let term_a = m_a * &fp_b;
            let term_a = if interaction_a.is_sender {
                term_a
            } else {
                -term_a
            };
            let term_b = m_b * &fp_a;
            let term_b = if interaction_b.is_sender {
                term_b
            } else {
                -term_b
            };
            c * &fp_a * &fp_b - term_a - term_b
        }

        let res = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                rap_challenges,
                logup_alpha_powers,
                packing_shifts,
                ..
            } => evaluate_batched_term_constraint(
                frame.get_evaluation_step(0),
                self.term_column_idx,
                &self.interaction_a,
                &self.interaction_b,
                rap_challenges,
                logup_alpha_powers,
                packing_shifts,
            ),
            TransitionEvaluationContext::Verifier {
                frame,
                rap_challenges,
                logup_alpha_powers,
                packing_shifts,
                ..
            } => evaluate_batched_term_constraint(
                frame.get_evaluation_step(0),
                self.term_column_idx,
                &self.interaction_a,
                &self.interaction_b,
                rap_challenges,
                logup_alpha_powers,
                packing_shifts,
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

impl<F, E> TransitionConstraintEvaluator<F, E> for LookupAccumulatedConstraint
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

    fn evaluate_verifier(
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
            shifts: &PackingShifts<A>,
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

            let z = &rap_challenges[0];

            // Clear denominators of absorbed interactions
            debug_assert!(matches!(absorbed.len(), 1 | 2));
            // Use conditional negation instead of E×E sign multiplication where possible
            match absorbed.len() {
                1 => {
                    // (delta) · f - sign · m = 0
                    // sign multiply also promotes m from base field A to extension B
                    let m = compute_multiplicity_from_step(second_step, &absorbed[0].multiplicity);
                    let f = compute_fingerprint_from_step(
                        second_step,
                        &absorbed[0],
                        z,
                        alpha_powers,
                        shifts,
                    );
                    let sign: FieldElement<B> = if absorbed[0].is_sender {
                        FieldElement::one()
                    } else {
                        -FieldElement::one()
                    };
                    delta * &f - m * sign
                }
                2 => {
                    // (delta) · f₁ · f₂ - sign₁·m₁·f₂ - sign₂·m₂·f₁ = 0
                    // m_i * f_j naturally promotes A→B, then conditionally negate
                    let m1 = compute_multiplicity_from_step(second_step, &absorbed[0].multiplicity);
                    let m2 = compute_multiplicity_from_step(second_step, &absorbed[1].multiplicity);
                    let f1 = compute_fingerprint_from_step(
                        second_step,
                        &absorbed[0],
                        z,
                        alpha_powers,
                        shifts,
                    );
                    let f2 = compute_fingerprint_from_step(
                        second_step,
                        &absorbed[1],
                        z,
                        alpha_powers,
                        shifts,
                    );
                    let term1 = m1 * &f2;
                    let term1 = if absorbed[0].is_sender { term1 } else { -term1 };
                    let term2 = m2 * &f1;
                    let term2 = if absorbed[1].is_sender { term2 } else { -term2 };
                    delta * &f1 * &f2 - term1 - term2
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
                packing_shifts,
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
                packing_shifts,
            ),
            TransitionEvaluationContext::Verifier {
                frame,
                logup_table_offset,
                rap_challenges,
                logup_alpha_powers,
                packing_shifts,
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
                packing_shifts,
            ),
        };

        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx) {
            *eval = res;
        }
    }
}

#[cfg(test)]
mod logup_single_source_tests {
    //! Differential tests for the single-source LogUp constraint bodies
    //! ([`emit_logup_constraints`]) against the OLD boxed constraint structs
    //! (`LookupBatchedTermConstraint` / `LookupAccumulatedConstraint`) that stay
    //! in-branch as the transcription oracle until the final deletion phase.
    //!
    //! For every layout we compare, on 1000 random two-step frames (off-trace
    //! points where a weakened or slipped transcription diverges with
    //! overwhelming probability):
    //!   1. [`ProverEvalFolder`] output == old `evaluate_prover` (ext slots);
    //!   2. [`VerifierEvalFolder`] output == old `evaluate_verifier`;
    //!   3. capture → flatten → interpret == old `evaluate_verifier`.
    //!
    //! Coverage: the accumulated constraint's 1-absorbed AND 2-absorbed branches
    //! (the latter reads `aux(1, ·)` next-row cells), the batched-term
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

    /// Build the OLD boxed constraints for a layout, index-for-index with
    /// [`emit_logup_constraints`]: committed batched terms first (idx `idx_base`
    /// onward), then the accumulated constraint.
    fn old_boxed(
        layout: &LogUpLayout,
        idx_base: usize,
    ) -> Vec<Box<dyn TransitionConstraintEvaluator<Gl, Ext3>>> {
        let mut out: Vec<Box<dyn TransitionConstraintEvaluator<Gl, Ext3>>> = Vec::new();
        let mut idx = idx_base;
        for pair_idx in 0..layout.num_committed_pairs {
            out.push(Box::new(LookupBatchedTermConstraint::new(
                layout.interactions[pair_idx * 2].clone(),
                layout.interactions[pair_idx * 2 + 1].clone(),
                pair_idx,
                idx,
            )));
            idx += 1;
        }
        if !layout.interactions.is_empty() {
            out.push(Box::new(LookupAccumulatedConstraint::new(
                idx,
                layout.num_term_columns,
                layout.absorbed().to_vec(),
            )));
        }
        out
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

    /// The full three-way differential check for one layout, on `TRIALS` random
    /// two-step frames.
    fn check_layout(label: &str, layout: &LogUpLayout, num_main_cols: usize) {
        let n_base = 0usize; // LogUp constraints are all extension-rooted.
        let old = old_boxed(layout, n_base);
        let n = old.len();
        assert_eq!(n, layout.num_constraints(), "[{label}] constraint count");

        // Meta parity vs the old boxed objects.
        let meta = logup_meta(layout, n_base);
        assert_eq!(meta.len(), n, "[{label}] meta count");
        let num_base = num_base_from_meta(&meta);
        assert_eq!(num_base, 0, "[{label}] LogUp meta is all-ext");
        for (i, m) in meta.iter().enumerate() {
            let c = old.iter().find(|c| c.constraint_idx() == i).expect("dense");
            assert_eq!(m.constraint_idx, i, "[{label}] meta idx {i}");
            assert_eq!(m.kind, RootKind::Ext, "[{label}] meta kind {i}");
            assert_eq!(m.degree, c.degree(), "[{label}] degree {i}");
            assert_eq!(m.period, c.period(), "[{label}] period {i}");
            assert_eq!(m.offset, c.offset(), "[{label}] offset {i}");
            assert_eq!(
                m.end_exemptions,
                c.end_exemptions(),
                "[{label}] end_exempt {i}"
            );
        }

        // Capture once; tree-measured degree <= declared.
        let mut cb = CaptureBuilder::<Gl, Ext3>::new();
        emit_logup_constraints(&mut cb, layout, n_base);
        let (prog, degrees) = cb.finish(num_base);
        assert!(prog.complete, "[{label}] capture must be complete");
        assert_eq!(degrees.len(), n, "[{label}] one emit per constraint");
        for &(idx, measured) in &degrees {
            assert!(
                measured <= meta[idx].degree,
                "[{label}] constraint {idx}: tree degree {measured} exceeds declared {}",
                meta[idx].degree
            );
        }

        let n_aux = num_aux_cols(layout);
        let shifts = PackingShifts::<Gl>::new();
        let vshifts = PackingShifts::<Ext3>::new();
        let no_periodic: Vec<Fp> = vec![];
        let no_periodic_e: Vec<Fp3> = vec![];

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
                &frame,
                &no_periodic,
                &rap_challenges,
                &alpha_powers,
                &table_offset,
                &shifts,
            );

            // --- old prover-side reference: evaluate_prover into ext slots ---
            let mut old_base = vec![Fp::zero(); n_base];
            let mut old_ext = vec![Fp3::zero(); n];
            for c in old.iter() {
                c.evaluate_prover(&prover_ctx, &mut old_base, &mut old_ext);
            }

            // --- 1. ProverEvalFolder == old evaluate_prover ---
            let mut base_out = vec![Fp::zero(); n_base];
            let mut ext_out = vec![Fp3::zero(); n];
            let mut folder = ProverEvalFolder::new(&prover_ctx, &mut base_out, &mut ext_out);
            emit_logup_constraints(&mut folder, layout, n_base);
            folder.assert_all_emitted();
            for i in 0..n {
                assert_eq!(
                    ext_out[i], old_ext[i],
                    "[{label}] prover folder mismatch, constraint {i}, trial {trial}"
                );
            }

            // --- 3. capture → interpret == old evaluate_prover ---
            let mut ir_base = vec![Fp::zero(); n_base];
            let mut ir_ext = vec![Fp3::zero(); n];
            eval_program(&prog, &prover_ctx, &mut ir_base, &mut ir_ext);
            for i in 0..n {
                assert_eq!(
                    ir_ext[i], old_ext[i],
                    "[{label}] interpreter mismatch, constraint {i}, trial {trial}"
                );
            }

            // --- verifier-side: embed the same frame into the extension ---
            let embed_step = |step: &TableView<Gl, Ext3>| -> TableView<Ext3, Ext3> {
                let main: Vec<Fp3> = (0..num_main_cols)
                    .map(|c| {
                        step.get_main_evaluation_element(0, c)
                            .clone()
                            .to_extension()
                    })
                    .collect();
                let aux: Vec<Fp3> = (0..n_aux)
                    .map(|c| step.get_aux_evaluation_element(0, c).clone())
                    .collect();
                TableView::new(vec![main], vec![aux])
            };
            let vframe: Frame<Ext3, Ext3> = Frame::new(vec![
                embed_step(frame.get_evaluation_step(0)),
                embed_step(frame.get_evaluation_step(1)),
            ]);
            let vctx = TransitionEvaluationContext::<Gl, Ext3>::new_verifier(
                &vframe,
                &no_periodic_e,
                &rap_challenges,
                &alpha_powers,
                &table_offset,
                &vshifts,
            );
            let mut old_vext = vec![Fp3::zero(); n];
            for c in old.iter() {
                c.evaluate_verifier(&vctx, &mut old_vext);
            }

            // --- 2. VerifierEvalFolder == old evaluate_verifier ---
            let mut vext_out = vec![Fp3::zero(); n];
            let mut vfolder = VerifierEvalFolder::new(&vctx, &mut vext_out);
            emit_logup_constraints(&mut vfolder, layout, n_base);
            vfolder.assert_all_emitted();
            for i in 0..n {
                assert_eq!(
                    vext_out[i], old_vext[i],
                    "[{label}] verifier folder mismatch, constraint {i}, trial {trial}"
                );
            }

            // capture → interpret (verifier) == old evaluate_verifier ---
            let mut ir_vext = vec![Fp3::zero(); n];
            eval_program_verifier(&prog, &vctx, &mut ir_vext);
            for i in 0..n {
                assert_eq!(
                    ir_vext[i], old_vext[i],
                    "[{label}] verifier interpreter mismatch, constraint {i}, trial {trial}"
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
    fn logup_one_absorbed_matches_old() {
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
    fn logup_two_absorbed_matches_old() {
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
    fn logup_matches_old_for_all_packing_variants() {
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
}

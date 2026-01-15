//! Type-aware bus packing for LogUp lookup arguments.
//!
//! This module defines types and combining rules for the bus fingerprint computation.
//! Values are combined in two stages:
//! 1. **Casting** (powers of 2): Combine limbs within a type (e.g., 4 bytes → 1 word)
//! 2. **Bus fingerprint** (powers of α): Combine all typed values into one fingerprint

use math::field::{element::FieldElement, traits::IsField};

// =============================================================================
// Shift Constants
// =============================================================================

/// 2^8 - shift for combining bytes
pub const SHIFT_8: u64 = 256;
/// 2^16 - shift for combining halves
pub const SHIFT_16: u64 = 65536;
/// 2^32 - shift for combining words
pub const SHIFT_32: u64 = 4294967296;
/// 2^48 - shift for high half in DWordHHW/DWordWHH
pub const SHIFT_48: u64 = 281474976710656;

// =============================================================================
// Bus Types
// =============================================================================

/// Defines how multiple columns (limbs) are combined into bus elements.
///
/// Each variant specifies:
/// - How many columns it consumes
/// - How many bus elements it produces
/// - The shift factors (powers of 2) used for combining
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusType {
    /// Single field element, no combining.
    /// Columns: 1, Bus elements: 1
    /// Used for: Bit, Byte, Half, Word, B4, B20, etc.
    Single,

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

impl BusType {
    /// Returns the number of trace columns this type consumes.
    pub fn num_columns(&self) -> usize {
        match self {
            BusType::Single => 1,
            BusType::Word2L => 2,
            BusType::Word4L => 4,
            BusType::DWordWL => 2,
            BusType::DWordHL => 4,
            BusType::DWordBL => 8,
            BusType::DWordHHW => 3,
            BusType::DWordWHH => 3,
        }
    }

    /// Returns the number of bus elements this type produces after combining.
    pub fn num_bus_elements(&self) -> usize {
        match self {
            BusType::Single => 1,
            BusType::Word2L => 1,
            BusType::Word4L => 1,
            BusType::DWordWL => 2,
            BusType::DWordHL => 2,
            BusType::DWordBL => 2,
            BusType::DWordHHW => 2,
            BusType::DWordWHH => 2,
        }
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
            "BusType {:?} expects {} columns, got {}",
            self,
            self.num_columns(),
            columns.len()
        );

        match self {
            BusType::Single => {
                vec![columns[0].clone()]
            }

            BusType::Word2L => {
                // h₀ + 2¹⁶·h₁
                let shift_16 = FieldElement::<E>::from(SHIFT_16);
                vec![&columns[0] + &columns[1] * &shift_16]
            }

            BusType::Word4L => {
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

            BusType::DWordWL => {
                // Two words, no combining - each stays as bus element
                // [w₀, w₁]
                vec![columns[0].clone(), columns[1].clone()]
            }

            BusType::DWordHL => {
                // [h₀ + 2¹⁶·h₁, h₂ + 2¹⁶·h₃]
                let shift_16 = FieldElement::<E>::from(SHIFT_16);
                vec![
                    &columns[0] + &columns[1] * &shift_16,
                    &columns[2] + &columns[3] * &shift_16,
                ]
            }

            BusType::DWordBL => {
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

            BusType::DWordHHW => {
                // [Word (LSB), Half, Half (MSB)]
                // → [w₀, h₀ + 2¹⁶·h₁]
                let shift_16 = FieldElement::<E>::from(SHIFT_16);
                vec![
                    columns[0].clone(),
                    &columns[1] + &columns[2] * &shift_16,
                ]
            }

            BusType::DWordWHH => {
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
    pub bus_type: BusType,
}

impl TypedValue {
    /// Creates a new typed value.
    ///
    /// # Arguments
    /// * `start_column` - First column index in the trace
    /// * `bus_type` - Type defining how many columns and how to combine
    pub fn new(start_column: usize, bus_type: BusType) -> Self {
        Self {
            start_column,
            bus_type,
        }
    }

    /// Creates a single-element value (no combining).
    pub fn single(column: usize) -> Self {
        Self::new(column, BusType::Single)
    }

    /// Creates a DWordWL value (2 words).
    pub fn dword_wl(start_column: usize) -> Self {
        Self::new(start_column, BusType::DWordWL)
    }

    /// Creates a DWordHL value (4 halves).
    pub fn dword_hl(start_column: usize) -> Self {
        Self::new(start_column, BusType::DWordHL)
    }

    /// Creates a DWordBL value (8 bytes).
    pub fn dword_bl(start_column: usize) -> Self {
        Self::new(start_column, BusType::DWordBL)
    }

    /// Creates a DWordHHW value (word + 2 halves).
    pub fn dword_hhw(start_column: usize) -> Self {
        Self::new(start_column, BusType::DWordHHW)
    }

    /// Creates a DWordWHH value (2 halves + word).
    pub fn dword_whh(start_column: usize) -> Self {
        Self::new(start_column, BusType::DWordWHH)
    }

    /// Creates a Word2L value (2 halves → 1 word).
    pub fn word_2l(start_column: usize) -> Self {
        Self::new(start_column, BusType::Word2L)
    }

    /// Creates a Word4L value (4 bytes → 1 word).
    pub fn word_4l(start_column: usize) -> Self {
        Self::new(start_column, BusType::Word4L)
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
// Typed Table Interaction
// =============================================================================

/// A type-aware table interaction for LogUp.
///
/// Unlike the basic `TableInteraction` which uses raw column indices,
/// this version understands types and combines limbs appropriately.
#[derive(Debug, Clone)]
pub struct TypedTableInteraction {
    /// Column index containing the multiplicity (or None for constant 1)
    pub multiplicity_column: Option<usize>,
    /// Typed values that make up this interaction
    pub values: Vec<TypedValue>,
    /// Whether this is a sender (true) or receiver (false)
    pub is_sender: bool,
}

impl TypedTableInteraction {
    /// Creates a new typed interaction.
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

    /// Computes the fingerprint for a row.
    ///
    /// # Process
    /// 1. For each TypedValue, extract columns and combine with powers of 2
    /// 2. Flatten all bus elements into a single list
    /// 3. Combine with powers of α: v₀ + α·v₁ + α²·v₂ + ...
    /// 4. Compute fingerprint: z - combined
    ///
    /// # Arguments
    /// * `get_column` - Function to get column value by index
    /// * `z` - Random challenge for fingerprint
    /// * `alpha` - Random challenge for linear combination
    pub fn compute_fingerprint<E: IsField, F: Fn(usize) -> FieldElement<E>>(
        &self,
        get_column: F,
        z: &FieldElement<E>,
        alpha: &FieldElement<E>,
    ) -> FieldElement<E> {
        // Step 1 & 2: Combine each typed value and flatten
        let bus_elements: Vec<FieldElement<E>> = self
            .values
            .iter()
            .flat_map(|tv| tv.combine_from(&get_column))
            .collect();

        // Step 3: Combine with powers of α
        let linear_combination: FieldElement<E> = bus_elements
            .iter()
            .enumerate()
            .map(|(i, elem)| elem * &alpha.pow(i))
            .sum();

        // Step 4: fingerprint = z - linear_combination
        z - &linear_combination
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::fields::fft_friendly::babybear_u32::Babybear31PrimeField;

    type FE = FieldElement<Babybear31PrimeField>;

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
        let combined = BusType::Word4L.combine(&bytes);
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
        let combined = BusType::DWordHL.combine(&halves);
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
        let combined = BusType::DWordHHW.combine(&cols);
        assert_eq!(combined.len(), 2);
        assert_eq!(combined[0], FE::from(0xAABBCCDDu64));
        assert_eq!(combined[1], FE::from(0x56781234u64));
    }
}

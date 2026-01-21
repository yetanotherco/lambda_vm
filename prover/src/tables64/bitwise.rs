//! BITWISE precomputed lookup table.
//!
//! This table provides 11 different lookup types used by other tables:
//!
//! ## Range Checks
//! - `IS_BYTE[X]` - X is a valid byte [0, 256)
//! - `IS_HALF[X]` - X is a valid halfword [0, 2^16)
//! - `IS_B20[X]` - X is a valid 20-bit value [0, 2^20)
//!
//! ## Bitwise Operations
//! - `AND_BYTE[X, Y]` -> X & Y
//! - `OR_BYTE[X, Y]` -> X | Y
//! - `XOR_BYTE[X, Y]` -> X ^ Y
//! - `MSB8[X]` -> most significant bit of byte
//! - `MSB16[X]` -> most significant bit of halfword
//! - `ZERO[X]` -> whether X is zero
//!
//! ## Shift Helpers
//! - `HWSL[X, Z]` -> (X << Z) mod 2^16
//! - `HWSLC[X, Z]` -> X >> (16 - Z)
//!
//! ## Table Structure
//!
//! The table has 2^20 rows (256 * 256 * 16) with precomputed values.
//! Each row is indexed by (X: Byte, Y: Byte, Z: B4).
//!
//! All lookups are provided as receivers with negative multiplicity,
//! meaning other tables send to this table.

use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices for BITWISE table
// =========================================================================

/// Input columns (precomputed)
pub mod cols {
    /// X: Byte input (0-255)
    pub const X: usize = 0;
    /// Y: Byte input (0-255)
    pub const Y: usize = 1;
    /// Z: 4-bit input (0-15) for shift amount
    pub const Z: usize = 2;

    /// AND result: X & Y
    pub const AND: usize = 3;
    /// OR result: X | Y
    pub const OR: usize = 4;
    /// XOR result: X ^ Y
    pub const XOR: usize = 5;
    /// MSB of byte X: (X >> 7) & 1
    pub const MSB8: usize = 6;
    /// MSB of halfword (X + 256*Y): ((X + 256*Y) >> 15) & 1
    pub const MSB16: usize = 7;
    /// Zero check: (X == 0 && Y == 0) ? 1 : 0
    pub const ZERO: usize = 8;
    /// Shift left result: ((X + 256*Y) << Z) & 0xFFFF
    pub const SLL: usize = 9;
    /// Shift left carry: (X + 256*Y) >> (16 - Z)
    pub const SLLC: usize = 10;

    // Multiplicity columns for each lookup type
    /// Multiplicity for AND_BYTE lookups
    pub const MU_AND: usize = 11;
    /// Multiplicity for OR_BYTE lookups
    pub const MU_OR: usize = 12;
    /// Multiplicity for XOR_BYTE lookups
    pub const MU_XOR: usize = 13;
    /// Multiplicity for MSB8 lookups
    pub const MU_MSB8: usize = 14;
    /// Multiplicity for MSB16 lookups
    pub const MU_MSB16: usize = 15;
    /// Multiplicity for ZERO lookups
    pub const MU_ZERO: usize = 16;
    /// Multiplicity for IS_BYTE lookups
    pub const MU_IS_BYTE: usize = 17;
    /// Multiplicity for IS_HALF lookups
    pub const MU_IS_HALF: usize = 18;
    /// Multiplicity for IS_B20 lookups
    pub const MU_IS_B20: usize = 19;
    /// Multiplicity for HWSL lookups
    pub const MU_HWSL: usize = 20;
    /// Multiplicity for HWSLC lookups
    pub const MU_HWSLC: usize = 21;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 22;
}

/// Number of rows in the BITWISE table: 256 * 256 * 16 = 2^20
pub const NUM_ROWS: usize = 256 * 256 * 16;

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the precomputed BITWISE trace table.
///
/// This creates a table with 2^20 rows, one for each combination of:
/// - X: 0..256 (byte)
/// - Y: 0..256 (byte)
/// - Z: 0..16 (4-bit shift amount)
///
/// All output columns are precomputed. Multiplicity columns are initialized
/// to zero and will be updated when other tables send lookups.
pub fn generate_bitwise_trace() -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let mut data = vec![FE::zero(); NUM_ROWS * cols::NUM_COLUMNS];

    for x in 0u32..256 {
        for y in 0u32..256 {
            for z in 0u32..16 {
                let row_idx = (x as usize) + (y as usize) * 256 + (z as usize) * 256 * 256;
                let base = row_idx * cols::NUM_COLUMNS;

                // Input columns
                data[base + cols::X] = FE::from(x as u64);
                data[base + cols::Y] = FE::from(y as u64);
                data[base + cols::Z] = FE::from(z as u64);

                // Bitwise operation results
                data[base + cols::AND] = FE::from((x & y) as u64);
                data[base + cols::OR] = FE::from((x | y) as u64);
                data[base + cols::XOR] = FE::from((x ^ y) as u64);

                // MSB extractions
                let msb8 = (x >> 7) & 1;
                let halfword = x + y * 256;
                let msb16 = (halfword >> 15) & 1;
                data[base + cols::MSB8] = FE::from(msb8 as u64);
                data[base + cols::MSB16] = FE::from(msb16 as u64);

                // Zero check (both X and Y must be zero)
                let is_zero = if x == 0 && y == 0 { 1u64 } else { 0u64 };
                data[base + cols::ZERO] = FE::from(is_zero);

                // Shift operations on halfword
                let sll = if z == 0 {
                    halfword
                } else {
                    (halfword << z) & 0xFFFF
                };
                let sllc = if z == 0 { 0 } else { halfword >> (16 - z) };
                data[base + cols::SLL] = FE::from(sll as u64);
                data[base + cols::SLLC] = FE::from(sllc as u64);

                // Multiplicity columns start at zero
                // They will be updated by update_multiplicities()
            }
        }
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

/// Computes the row index for a given (X, Y, Z) tuple.
#[inline]
pub fn row_index(x: u8, y: u8, z: u8) -> usize {
    debug_assert!(z < 16, "Z must be in range [0, 16)");
    (x as usize) + (y as usize) * 256 + (z as usize) * 256 * 256
}

/// Updates multiplicity columns based on lookups from other tables.
///
/// This function should be called after all other tables have recorded their lookups.
///
/// # Arguments
/// * `trace` - The BITWISE trace table to update
/// * `lookups` - Vector of (lookup_type, x, y, z) tuples
pub fn update_multiplicities(
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    lookups: &[(BitwiseLookup, u8, u8, u8)],
) {
    for (lookup_type, x, y, z) in lookups {
        let row = row_index(*x, *y, *z);
        let mu_col = match lookup_type {
            BitwiseLookup::AndByte => cols::MU_AND,
            BitwiseLookup::OrByte => cols::MU_OR,
            BitwiseLookup::XorByte => cols::MU_XOR,
            BitwiseLookup::Msb8 => cols::MU_MSB8,
            BitwiseLookup::Msb16 => cols::MU_MSB16,
            BitwiseLookup::Zero => cols::MU_ZERO,
            BitwiseLookup::IsByte => cols::MU_IS_BYTE,
            BitwiseLookup::IsHalf => cols::MU_IS_HALF,
            BitwiseLookup::IsB20 => cols::MU_IS_B20,
            BitwiseLookup::Hwsl => cols::MU_HWSL,
            BitwiseLookup::Hwslc => cols::MU_HWSLC,
        };

        // Increment multiplicity
        let current = trace.main_table.get_row(row)[mu_col];
        trace.set_main(row, mu_col, current + FE::one());
    }
}

/// Types of lookups the BITWISE table provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitwiseLookup {
    AndByte,
    OrByte,
    XorByte,
    Msb8,
    Msb16,
    Zero,
    IsByte,
    IsHalf,
    IsB20,
    Hwsl,
    Hwslc,
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the BITWISE table.
///
/// The BITWISE table is a **receiver** for all lookups (negative multiplicity
/// in the spec corresponds to receiving lookups from other tables).
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // AND_BYTE[X, Y] -> AND
        BusInteraction::receiver(
            BusId::AndByte,
            Multiplicity::Column(cols::MU_AND),
            vec![
                BusValue::Packed {
                    start_column: cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::AND,
                    packing: Packing::Direct,
                },
            ],
        ),
        // OR_BYTE[X, Y] -> OR
        BusInteraction::receiver(
            BusId::OrByte,
            Multiplicity::Column(cols::MU_OR),
            vec![
                BusValue::Packed {
                    start_column: cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::OR,
                    packing: Packing::Direct,
                },
            ],
        ),
        // XOR_BYTE[X, Y] -> XOR
        BusInteraction::receiver(
            BusId::XorByte,
            Multiplicity::Column(cols::MU_XOR),
            vec![
                BusValue::Packed {
                    start_column: cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::XOR,
                    packing: Packing::Direct,
                },
            ],
        ),
        // MSB8[X] -> MSB8
        BusInteraction::receiver(
            BusId::Msb8,
            Multiplicity::Column(cols::MU_MSB8),
            vec![
                BusValue::Packed {
                    start_column: cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::MSB8,
                    packing: Packing::Direct,
                },
            ],
        ),
        // MSB16[X + 256*Y] -> MSB16
        // Input is packed as Word2L (X + 2^8 * Y would need custom, but spec says X + 256*Y)
        // Since X and Y are bytes, we use a linear combination
        BusInteraction::receiver(
            BusId::Msb16,
            Multiplicity::Column(cols::MU_MSB16),
            vec![
                // X + 256*Y as linear combination
                BusValue::linear(vec![
                    stark::lookup::LinearTerm::Column {
                        coefficient: 1,
                        column: cols::X,
                    },
                    stark::lookup::LinearTerm::Column {
                        coefficient: 256,
                        column: cols::Y,
                    },
                ]),
                BusValue::Packed {
                    start_column: cols::MSB16,
                    packing: Packing::Direct,
                },
            ],
        ),
        // ZERO[X + 256*Y] -> ZERO
        BusInteraction::receiver(
            BusId::Zero,
            Multiplicity::Column(cols::MU_ZERO),
            vec![
                BusValue::linear(vec![
                    stark::lookup::LinearTerm::Column {
                        coefficient: 1,
                        column: cols::X,
                    },
                    stark::lookup::LinearTerm::Column {
                        coefficient: 256,
                        column: cols::Y,
                    },
                ]),
                BusValue::Packed {
                    start_column: cols::ZERO,
                    packing: Packing::Direct,
                },
            ],
        ),
        // IS_BYTE[X] - range check, no output
        BusInteraction::receiver(
            BusId::IsByte,
            Multiplicity::Column(cols::MU_IS_BYTE),
            vec![BusValue::Packed {
                start_column: cols::X,
                packing: Packing::Direct,
            }],
        ),
        // IS_HALF[X + 256*Y] - range check for halfword
        BusInteraction::receiver(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU_IS_HALF),
            vec![BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::X,
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 256,
                    column: cols::Y,
                },
            ])],
        ),
        // IS_B20[X + 256*Y + 65536*Z] - range check for 20-bit
        BusInteraction::receiver(
            BusId::IsB20,
            Multiplicity::Column(cols::MU_IS_B20),
            vec![BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::X,
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 256,
                    column: cols::Y,
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 65536,
                    column: cols::Z,
                },
            ])],
        ),
        // HWSL[X + 256*Y, Z] -> SLL
        BusInteraction::receiver(
            BusId::Hwsl,
            Multiplicity::Column(cols::MU_HWSL),
            vec![
                BusValue::linear(vec![
                    stark::lookup::LinearTerm::Column {
                        coefficient: 1,
                        column: cols::X,
                    },
                    stark::lookup::LinearTerm::Column {
                        coefficient: 256,
                        column: cols::Y,
                    },
                ]),
                BusValue::Packed {
                    start_column: cols::Z,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::SLL,
                    packing: Packing::Direct,
                },
            ],
        ),
        // HWSLC[X + 256*Y, Z] -> SLLC
        BusInteraction::receiver(
            BusId::Hwslc,
            Multiplicity::Column(cols::MU_HWSLC),
            vec![
                BusValue::linear(vec![
                    stark::lookup::LinearTerm::Column {
                        coefficient: 1,
                        column: cols::X,
                    },
                    stark::lookup::LinearTerm::Column {
                        coefficient: 256,
                        column: cols::Y,
                    },
                ]),
                BusValue::Packed {
                    start_column: cols::Z,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::SLLC,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
}

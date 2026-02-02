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

use lazy_static::lazy_static;
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{TraceTable, columns2rows};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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

/// Number of precomputed (non-multiplicity) columns
pub const NUM_PRECOMPUTED_COLS: usize = 11;

// =========================================================================
// Compile-time row generation
// =========================================================================

/// Generate bitwise table row values at compile time.
///
/// This is a `const fn` that can be evaluated at compile time, allowing
/// the verifier to compute table values without runtime overhead.
///
/// Index encoding: `index = x + y * 256 + z * 65536`
/// where x, y ∈ [0, 255] and z ∈ [0, 15]
///
/// Returns the 11 precomputed columns: [X, Y, Z, AND, OR, XOR, MSB8, MSB16, ZERO, SLL, SLLC]
#[inline]
pub const fn generate_bitwise_row(index: usize) -> [u64; NUM_PRECOMPUTED_COLS] {
    let x = (index & 0xFF) as u32;
    let y = ((index >> 8) & 0xFF) as u32;
    let z = ((index >> 16) & 0xF) as u32;

    // Bitwise operations on bytes
    let and_val = x & y;
    let or_val = x | y;
    let xor_val = x ^ y;

    // MSB extractions
    let msb8 = (x >> 7) & 1;
    let halfword = x + y * 256;
    let msb16 = (halfword >> 15) & 1;

    // Zero check (both X and Y must be zero)
    let is_zero = if x == 0 && y == 0 { 1 } else { 0 };

    // Shift operations on halfword
    let sll = if z == 0 {
        halfword
    } else {
        (halfword << z) & 0xFFFF
    };
    let sllc = if z == 0 { 0 } else { halfword >> (16 - z) };

    [
        x as u64,       // X
        y as u64,       // Y
        z as u64,       // Z
        and_val as u64, // AND
        or_val as u64,  // OR
        xor_val as u64, // XOR
        msb8 as u64,    // MSB8
        msb16 as u64,   // MSB16
        is_zero as u64, // ZERO
        sll as u64,     // SLL
        sllc as u64,    // SLLC
    ]
}

/// Whether this table is preprocessed (commitment is hardcoded).
///
/// Preprocessed tables have their commitment known at compile time,
/// so it's not included in proofs - both prover and verifier use the
/// hardcoded value in the Fiat-Shamir transcript.
pub const fn is_preprocessed() -> bool {
    true
}

// =========================================================================
// Preprocessed commitment (computed once, cached)
// =========================================================================

lazy_static! {
    /// Commitment to the LDE of the precomputed bitwise table columns.
    ///
    /// This is a Merkle root over 2^22 rows (2^20 * blowup_factor=4) of the
    /// LDE-evaluated 11 precomputed columns (X, Y, Z, AND, OR, XOR, MSB8,
    /// MSB16, ZERO, SLL, SLLC).
    ///
    /// The commitment is over LDE values (not raw values) because FRI queries
    /// can target any index in the extended domain [0, N*blowup). The verifier
    /// checks that proofs open against this exact commitment.
    ///
    /// Computed once on first access and cached. Both prover and verifier
    /// use this same value in the Fiat-Shamir transcript.
    pub static ref BITWISE_TABLE_COMMITMENT: Commitment = compute_preprocessed_commitment();
}

/// Standard blowup factor for LDE domain (matches proof options).
const LDE_BLOWUP_FACTOR: usize = 4;

/// Standard coset offset for LDE domain (matches proof options).
const LDE_COSET_OFFSET: u64 = 3;

/// Computes the Merkle commitment over the precomputed bitwise table columns.
///
/// This builds a Merkle tree over the LDE (Low Degree Extension) of the precomputed
/// columns, matching exactly how the prover commits to traces. The tree has
/// NUM_ROWS * LDE_BLOWUP_FACTOR = 2^22 leaves, enabling FRI queries at any index
/// in the extended domain.
///
/// Critical for security: the commitment must be over LDE values (not raw values)
/// because FRI queries can target any index in [0, N*blowup). A raw-value commitment
/// would only have N leaves, unable to verify queries at indices >= N.
fn compute_preprocessed_commitment() -> Commitment {
    // Step 1: Generate precomputed columns in parallel
    // Each column is generated independently by iterating over all row indices
    #[cfg(feature = "parallel")]
    let columns: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
        .into_par_iter()
        .map(|col_idx| {
            (0..NUM_ROWS)
                .map(|idx| {
                    let row = generate_bitwise_row(idx);
                    FE::from(row[col_idx])
                })
                .collect()
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let columns: Vec<Vec<FE>> = {
        let mut cols: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
            .map(|_| Vec::with_capacity(NUM_ROWS))
            .collect();
        for idx in 0..NUM_ROWS {
            let row = generate_bitwise_row(idx);
            for (col_idx, &value) in row.iter().enumerate() {
                cols[col_idx].push(FE::from(value));
            }
        }
        cols
    };

    // Step 2: Interpolate each column to a polynomial (parallel)
    #[cfg(feature = "parallel")]
    let polys: Vec<Polynomial<FE>> = columns
        .par_iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for bitwise column")
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for bitwise column")
        })
        .collect();

    // Step 3: Evaluate polynomials on LDE domain (parallel)
    let coset_offset = FE::from(LDE_COSET_OFFSET);

    #[cfg(feature = "parallel")]
    let mut lde_columns: Vec<Vec<FE>> = polys
        .par_iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, LDE_BLOWUP_FACTOR, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for bitwise polynomial")
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let mut lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, LDE_BLOWUP_FACTOR, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for bitwise polynomial")
        })
        .collect();

    // Step 4: Bit-reverse permute (parallel)
    #[cfg(feature = "parallel")]
    lde_columns.par_iter_mut().for_each(|col| {
        in_place_bit_reverse_permute(col);
    });

    #[cfg(not(feature = "parallel"))]
    for col in lde_columns.iter_mut() {
        in_place_bit_reverse_permute(col);
    }

    // Step 5: Convert columns to rows for Merkle tree
    let lde_rows = columns2rows(lde_columns);

    // Step 6: Build Merkle tree over LDE (N * blowup leaves)
    let tree = BatchedMerkleTree::<GoldilocksField>::build(&lde_rows)
        .expect("Failed to build Merkle tree for bitwise LDE");

    tree.root
}

/// Returns the preprocessed commitment for the bitwise table.
///
/// This is a convenience function that dereferences the lazy_static.
#[inline]
pub fn preprocessed_commitment() -> Commitment {
    *BITWISE_TABLE_COMMITMENT
}

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
/// * `ops` - Vector of BitwiseOperation requests
pub fn update_multiplicities(
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    ops: &[BitwiseOperation],
) {
    for op in ops {
        let row = row_index(op.x, op.y, op.z);
        let mu_col = match op.lookup_type {
            BitwiseOperationType::AndByte => cols::MU_AND,
            BitwiseOperationType::OrByte => cols::MU_OR,
            BitwiseOperationType::XorByte => cols::MU_XOR,
            BitwiseOperationType::Msb8 => cols::MU_MSB8,
            BitwiseOperationType::Msb16 => cols::MU_MSB16,
            BitwiseOperationType::Zero => cols::MU_ZERO,
            BitwiseOperationType::IsByte => cols::MU_IS_BYTE,
            BitwiseOperationType::IsHalf => cols::MU_IS_HALF,
            BitwiseOperationType::IsB20 => cols::MU_IS_B20,
            BitwiseOperationType::Hwsl => cols::MU_HWSL,
            BitwiseOperationType::Hwslc => cols::MU_HWSLC,
        };

        // Increment multiplicity
        let current = trace.main_table.get_row(row)[mu_col];
        trace.set_main(row, mu_col, current + FE::one());
    }
}

/// Removes rows where all multiplicity columns are zero.
/// Returns a smaller table containing only rows with actual lookups.
///
/// # WARNING: UNSOUND FOR PRODUCTION
///
/// This function is for tests only. The reduced table is NOT a valid
/// preprocessed table because:
/// 1. Row indices no longer match the (x, y, z) encoding
/// 2. The verifier cannot verify against a preprocessed commitment
/// 3. A malicious prover could claim incorrect bitwise results
///
/// This is acceptable for tests because we're testing:
/// - Bus interaction balancing (sends = receives)
/// - Constraint satisfaction
/// - LogUp protocol correctness
#[cfg(test)]
pub(crate) fn trim_zero_rows(
    trace: TraceTable<GoldilocksField, GoldilocksExtension>,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    use super::types::FE;

    let num_rows = trace.main_table.height;

    // Find rows with any non-zero multiplicity
    let kept_rows: Vec<usize> = (0..num_rows)
        .filter(|&row| {
            let row_data = trace.main_table.get_row(row);
            // Check all multiplicity columns (indices 11-21)
            (cols::MU_AND..=cols::MU_HWSLC).any(|col| row_data[col] != FE::zero())
        })
        .collect();

    if kept_rows.is_empty() {
        // No lookups - return minimal table with 16 rows of zeros
        let data = vec![FE::zero(); 16 * cols::NUM_COLUMNS];
        return TraceTable::new_main(data, cols::NUM_COLUMNS, 1);
    }

    // Determine new table size (next power of 2, minimum 16)
    let new_size = kept_rows.len().next_power_of_two().max(16);

    // Allocate new trace data
    let mut new_data = vec![FE::zero(); new_size * cols::NUM_COLUMNS];

    // Copy kept rows to new table
    for (new_row, &old_row) in kept_rows.iter().enumerate() {
        let old_row_data = trace.main_table.get_row(old_row);
        let base = new_row * cols::NUM_COLUMNS;
        for (col, &val) in old_row_data.iter().enumerate() {
            new_data[base + col] = val;
        }
    }

    TraceTable::new_main(new_data, cols::NUM_COLUMNS, 1)
}

/// Types of lookups the BITWISE table provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitwiseOperationType {
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

/// A lookup request to the BITWISE precomputed table.
///
/// The BITWISE table has 2^20 rows indexed by `(x, y, z)`.
/// Each row contains precomputed results for various operations.
///
/// # Fields (matching spec column names)
/// - `lookup_type`: Which operation result to look up
/// - `x`: Byte input (0-255)
/// - `y`: Byte input (0-255)
/// - `z`: 4-bit value (0-15), shift amount for HWSL/HWSLC
///
/// # How inputs map to operations
/// - AND/OR/XOR: `x OP y`
/// - MSB8: MSB of `x`
/// - MSB16: MSB of halfword `x + y * 256`
/// - IS_BYTE/IS_HALF: Range check on `x + y * 256`
/// - HWSL/HWSLC: Shift `x + y * 256` by `z` bits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitwiseOperation {
    pub lookup_type: BitwiseOperationType,
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

impl BitwiseOperation {
    /// Create a new bitwise operation.
    pub fn new(lookup_type: BitwiseOperationType, x: u8, y: u8, z: u8) -> Self {
        debug_assert!(z < 16, "z must be in range [0, 16)");
        Self {
            lookup_type,
            x,
            y,
            z,
        }
    }

    /// Create an operation for byte ops (AND, OR, XOR) where z is unused.
    pub fn byte_op(lookup_type: BitwiseOperationType, x: u8, y: u8) -> Self {
        Self::new(lookup_type, x, y, 0)
    }

    /// Create an operation for single-byte ops (MSB8, IS_BYTE).
    pub fn single_byte(lookup_type: BitwiseOperationType, x: u8) -> Self {
        Self::new(lookup_type, x, 0, 0)
    }

    /// Create an operation for halfword ops (MSB16, IS_HALF, ZERO).
    pub fn halfword(lookup_type: BitwiseOperationType, x: u8, y: u8) -> Self {
        Self::new(lookup_type, x, y, 0)
    }

    /// Create an operation for shift ops (HWSL, HWSLC).
    pub fn shift_op(lookup_type: BitwiseOperationType, x: u8, y: u8, z: u8) -> Self {
        Self::new(lookup_type, x, y, z)
    }
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

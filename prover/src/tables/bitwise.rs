//! BITWISE precomputed lookup table.
//!
//! This table provides byte/range lookup types used by other tables:
//!
//! ## Range Checks
//! - `ARE_BYTES[X, Y]` - X and Y are valid bytes [0, 256). Spec template
//!   `IS_BYTE<X>` is implemented by sending `ARE_BYTES[X, 0]`.
//! - `IS_HALF[X]` - X is a valid halfword [0, 2^16)
//! - `IS_B20[X]` - X is a valid 20-bit value [0, 2^20)
//!
//! ## Bitwise Operations
//! - `BYTE_ALU[opsel, X, Y] -> out` for byte AND/OR/XOR
//! - `MSB8[X]` -> most significant bit of byte
//! - `MSB16[X]` -> most significant bit of halfword
//! - `ZERO[X]` -> whether X is zero
//!
//! ## Shift Helpers
//! - `HWSL[X, Z]` -> [(X << Z) mod 2^16, X >> (16 - Z)]
//!
//! ## Table Structure
//!
//! The table has 2^20 rows (256 * 256 * 16) with precomputed values.
//! Each row is indexed by (X: Byte, Y: Byte, Z: B4).
//!
//! All lookups are provided as receivers with negative multiplicity,
//! meaning other tables send to this table.

use math::fft::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{TraceTable, columns2rows};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};

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
    /// Multiplicity for MSB8 lookups
    pub const MU_MSB8: usize = 11;
    /// Multiplicity for MSB16 lookups
    pub const MU_MSB16: usize = 12;
    /// Multiplicity for ZERO lookups
    pub const MU_ZERO: usize = 13;
    /// Multiplicity for ARE_BYTES lookups. Each lookup checks X and Y; pass Y=0
    /// for a single-byte range check (spec template `IS_BYTE<X>`).
    pub const MU_ARE_BYTES: usize = 14;
    /// Multiplicity for IS_HALF lookups
    pub const MU_IS_HALF: usize = 15;
    /// Multiplicity for IS_B20 lookups
    pub const MU_IS_B20: usize = 16;
    /// Multiplicity for HWSL lookups
    pub const MU_HWSL: usize = 17;
    /// Multiplicity for `BYTE_ALU[opsel=AND]` lookups
    pub const MU_BYTE_ALU_AND: usize = 18;
    /// Multiplicity for `BYTE_ALU[opsel=OR]` lookups
    pub const MU_BYTE_ALU_OR: usize = 19;
    /// Multiplicity for `BYTE_ALU[opsel=XOR]` lookups
    pub const MU_BYTE_ALU_XOR: usize = 20;
    /// Total number of columns
    pub const NUM_COLUMNS: usize = 21;
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

    // Zero check (X + 256*Y + 65536*Z must be zero)
    let is_zero = if x == 0 && y == 0 && z == 0 { 1 } else { 0 };

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

/// Whether this table is preprocessed (commitment is static).
///
/// Preprocessed tables have their commitment known at compile time,
/// so it's not included in proofs - both prover and verifier use the
/// static value in the Fiat-Shamir transcript.
pub const fn is_preprocessed() -> bool {
    true
}

// =========================================================================
// Preprocessed commitment
// =========================================================================

/// Returns the static BITWISE preprocessed commitment for `blowup_factor`,
/// or `None` if no value is shipped for it. Values were generated by the
/// `compute_static_commitments` binary at the project's standard
/// `coset_offset = 3` (the value every in-tree `ProofOptions` constructor
/// pins) and pinned by `bitwise_static_matches_recompute_*` tests so any
/// drift in the AIR or FFT pipeline is caught at test time. The verifier
/// reads these from its compiled binary — no input data is trusted.
///
/// # Regenerating
///
/// Only regenerate these match arms after a *deliberate, reviewed* change
/// to the BITWISE table layout, the AIR's preprocessed column count, or
/// the FFT / LDE / Merkle pipeline. Run:
///
/// ```text
/// cargo run --bin compute_static_commitments --release
/// ```
///
/// and paste the printed match arms over the ones below.
///
/// **If a drift test failed, do not regenerate first.** The drift tests
/// exist to force a human to ask "why did this change?" before the new
/// bytes get blessed. Re-pasting on a drift failure silently launders an
/// unintended table change into the verifier's compiled-in trust anchor.
fn static_commitment(blowup_factor: u8) -> Option<Commitment> {
    match blowup_factor {
        2 => Some([
            0xfb, 0x46, 0xff, 0x1c, 0xed, 0x4c, 0x97, 0xfb, 0xb2, 0x17, 0x55, 0x24, 0x08, 0x04,
            0x15, 0xee, 0xbe, 0xa6, 0xee, 0x86, 0x69, 0xaf, 0x3a, 0x4f, 0x9e, 0x2a, 0x44, 0x81,
            0xf9, 0xb0, 0xf3, 0xff,
        ]),
        4 => Some([
            0xb5, 0xc4, 0xc0, 0x80, 0x03, 0x5b, 0xb6, 0x12, 0x78, 0x8c, 0x4d, 0xd4, 0x9e, 0x3d,
            0xc4, 0xe2, 0xef, 0x95, 0xf0, 0xbf, 0xe8, 0x1d, 0x98, 0xec, 0x7f, 0x58, 0x3a, 0x47,
            0x18, 0x03, 0x7e, 0xa5,
        ]),
        8 => Some([
            0x8a, 0x18, 0x70, 0x51, 0x34, 0x1a, 0x65, 0xaa, 0x79, 0x17, 0x07, 0x9a, 0xf3, 0x0b,
            0xcb, 0xd0, 0x7c, 0xe3, 0x2a, 0xce, 0x89, 0x9a, 0xfd, 0xc8, 0x0d, 0x6b, 0x48, 0x43,
            0x83, 0x5d, 0x18, 0xb8,
        ]),
        _ => None,
    }
}

/// Computes the Merkle commitment over the precomputed bitwise table columns.
///
/// This builds a Merkle tree over the LDE (Low Degree Extension) of the precomputed
/// columns, matching exactly how the prover commits to traces. The tree has
/// NUM_ROWS * blowup_factor leaves, enabling FRI queries at any index
/// in the extended domain.
///
/// Critical for security: the commitment must be over LDE values (not raw values)
/// because FRI queries can target any index in [0, N*blowup). A raw-value commitment
/// would only have N leaves, unable to verify queries at indices >= N.
///
/// Exposed for the `compute_static_commitments` binary and the
/// drift-detection tests in `static_commitments_tests`. Production callers
/// should go through [`preprocessed_commitment`] so the static const-table
/// shortcut is used when applicable.
#[doc(hidden)]
pub fn compute_preprocessed_commitment(options: &ProofOptions) -> Commitment {
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
    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);

    #[cfg(feature = "parallel")]
    let mut lde_columns: Vec<Vec<FE>> = polys
        .par_iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for bitwise polynomial")
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let mut lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, NUM_ROWS, &coset_offset)
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
/// Looks up `blowup_factor` via [`static_commitment`] when `coset_offset == 3`
/// (the value every in-tree `ProofOptions` constructor pins, and the offset
/// the static bytes were generated for); on miss — either a non-3 coset or a
/// `blowup_factor` outside `STATIC_BLOWUP_FACTORS` — recomputes from scratch.
pub fn preprocessed_commitment(options: &ProofOptions) -> Commitment {
    if options.coset_offset == 3
        && let Some(commitment) = static_commitment(options.blowup_factor)
    {
        return commitment;
    }
    log::warn!(
        "bitwise preprocessed commitment not static for (blowup={}, coset={}); \
         falling back to recompute. Add a match arm to `static_commitment` by running \
         `cargo run --bin compute_static_commitments --release`.",
        options.blowup_factor,
        options.coset_offset,
    );
    compute_preprocessed_commitment(options)
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
    let mut trace = TraceTable::new_main(
        vec![FE::zero(); NUM_ROWS * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for x in 0u32..256 {
        for y in 0u32..256 {
            for z in 0u32..16 {
                let row_idx = (x as usize) + (y as usize) * 256 + (z as usize) * 256 * 256;

                // Input columns
                table.set_byte(row_idx, cols::X, x as u8);
                table.set_byte(row_idx, cols::Y, y as u8);
                table.set_byte(row_idx, cols::Z, z as u8);

                // Bitwise operation results
                table.set_byte(row_idx, cols::AND, (x & y) as u8);
                table.set_byte(row_idx, cols::OR, (x | y) as u8);
                table.set_byte(row_idx, cols::XOR, (x ^ y) as u8);

                // MSB extractions
                let msb8 = (x >> 7) & 1;
                let halfword = x + y * 256;
                let msb16 = (halfword >> 15) & 1;
                table.set_bool(row_idx, cols::MSB8, msb8 == 1);
                table.set_bool(row_idx, cols::MSB16, msb16 == 1);

                // Zero check (X + 256*Y + 65536*Z must be zero)
                table.set_bool(row_idx, cols::ZERO, x == 0 && y == 0 && z == 0);

                // Shift operations on halfword
                let sll = if z == 0 {
                    halfword
                } else {
                    (halfword << z) & 0xFFFF
                };
                let sllc = if z == 0 { 0 } else { halfword >> (16 - z) };
                table.set_half(row_idx, cols::SLL, sll as u16);
                table.set_half(row_idx, cols::SLLC, sllc as u16);

                // Multiplicity columns start at zero
                // They will be updated by update_multiplicities()
            }
        }
    }

    trace
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
            BitwiseOperationType::Msb8 => cols::MU_MSB8,
            BitwiseOperationType::Msb16 => cols::MU_MSB16,
            BitwiseOperationType::Zero => cols::MU_ZERO,
            BitwiseOperationType::AreBytes => cols::MU_ARE_BYTES,
            BitwiseOperationType::IsHalf => cols::MU_IS_HALF,
            BitwiseOperationType::IsB20 => cols::MU_IS_B20,
            BitwiseOperationType::Hwsl => cols::MU_HWSL,
            BitwiseOperationType::ByteAluAnd => cols::MU_BYTE_ALU_AND,
            BitwiseOperationType::ByteAluOr => cols::MU_BYTE_ALU_OR,
            BitwiseOperationType::ByteAluXor => cols::MU_BYTE_ALU_XOR,
        };

        // Increment multiplicity
        let current = trace.main_table.get_row(row)[mu_col];
        trace.main_table.set_fe(row, mu_col, current + FE::one());
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
            // Check all multiplicity columns, including rows used only by a
            // BYTE_ALU lookup.
            (cols::MU_MSB8..=cols::MU_BYTE_ALU_XOR).any(|col| row_data[col] != FE::zero())
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
    Msb8,
    Msb16,
    Zero,
    AreBytes,
    IsHalf,
    IsB20,
    Hwsl,
    ByteAluAnd,
    ByteAluOr,
    ByteAluXor,
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
/// - `z`: 4-bit value (0-15), shift amount for HWSL
///
/// # How inputs map to operations
/// - AND/OR/XOR: `x OP y`
/// - MSB8: MSB of `x`
/// - MSB16: MSB of halfword `x + y * 256`
/// - ARE_BYTES: Range check both `x` and `y`; use `y = 0` for a single byte
/// - IS_HALF: Range check on `x + y * 256`
/// - HWSL: Shift `x + y * 256` by `z` bits, returning [SLL, SLLC] as a pair
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

    /// Create an operation for single-byte ops (MSB8, ARE_BYTES with y=0).
    pub fn single_byte(lookup_type: BitwiseOperationType, x: u8) -> Self {
        Self::new(lookup_type, x, 0, 0)
    }

    /// Create an operation for halfword ops (MSB16, IS_HALF).
    pub fn halfword(lookup_type: BitwiseOperationType, x: u8, y: u8) -> Self {
        Self::new(lookup_type, x, y, 0)
    }

    /// Create a ZERO lookup for a value up to 20 bits.
    /// Value is decomposed as: x + 256*y + 65536*z.
    pub fn zero(value: u32) -> Self {
        assert!(value < (1 << 20), "ZERO value must fit in 20 bits");
        let x = (value & 0xFF) as u8;
        let y = ((value >> 8) & 0xFF) as u8;
        let z = ((value >> 16) & 0xF) as u8;
        Self::new(BitwiseOperationType::Zero, x, y, z)
    }

    /// Create an operation for HWSL shift lookups.
    pub fn shift_op(lookup_type: BitwiseOperationType, x: u8, y: u8, z: u8) -> Self {
        Self::new(lookup_type, x, y, z)
    }

    /// Create an IS_B20 operation for 20-bit range checks.
    /// Value is packed as: x + 256*y + 65536*z (where z is 4 bits).
    pub fn b20(x: u8, y: u8, z: u8) -> Self {
        debug_assert!(z < 16, "z must be 4-bit value for IS_B20");
        Self::new(BitwiseOperationType::IsB20, x, y, z)
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
        // ZERO[X + 256*Y + 65536*Z] -> ZERO
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
                    stark::lookup::LinearTerm::Column {
                        coefficient: 65536,
                        column: cols::Z,
                    },
                ]),
                BusValue::Packed {
                    start_column: cols::ZERO,
                    packing: Packing::Direct,
                },
            ],
        ),
        // ARE_BYTES[X, Y] - range check two byte values, no output.
        // Single-byte checks (spec template `IS_BYTE<X>`) send Y=0.
        BusInteraction::receiver(
            BusId::AreBytes,
            Multiplicity::Column(cols::MU_ARE_BYTES),
            vec![
                BusValue::Packed {
                    start_column: cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::Y,
                    packing: Packing::Direct,
                },
            ],
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
        // HWSL[X + 256*Y, Z] -> [SLL, SLLC]
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
                BusValue::Packed {
                    start_column: cols::SLLC,
                    packing: Packing::Direct,
                },
            ],
        ),
        // BYTE_ALU[opsel, X, Y] -> out.
        // Unifies AND/OR/XOR into one bus keyed by the `alu_op` descriptor.
        // Implemented as one receiver per opsel, reusing the precomputed
        // AND/OR/XOR result columns (the "single 2^20 column" in bitwise.typ is
        // an optimization note, not a requirement).
        BusInteraction::receiver(
            BusId::ByteAlu,
            Multiplicity::Column(cols::MU_BYTE_ALU_AND),
            vec![
                BusValue::constant(alu_op::AND as u64),
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
        BusInteraction::receiver(
            BusId::ByteAlu,
            Multiplicity::Column(cols::MU_BYTE_ALU_OR),
            vec![
                BusValue::constant(alu_op::OR as u64),
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
        BusInteraction::receiver(
            BusId::ByteAlu,
            Multiplicity::Column(cols::MU_BYTE_ALU_XOR),
            vec![
                BusValue::constant(alu_op::XOR as u64),
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
    ]
}

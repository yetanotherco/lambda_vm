//! BYTE_OPS precomputed lookup table for byte-pair operations.
//!
//! Holds every `(X, Y) ∈ [0, 256)²` and the precomputed result of the
//! byte-pair lookups that BITWISE used to multiplex into its 2²⁰ rows. The
//! 16× factor was driven only by 20-bit ops (HWSL/IS_B20/ZERO); pulling these
//! out into a dedicated 2¹⁶ table cuts ~12M cells.
//!
//! ## Operations served (Step 2 will wire the receivers)
//! - `AND_BYTE[X, Y]` -> X & Y
//! - `OR_BYTE[X, Y]` -> X | Y
//! - `XOR_BYTE[X, Y]` -> X ^ Y
//! - `MSB8[X]` -> most significant bit of byte (Y = 0)
//! - `MSB16[X + 256*Y]` -> most significant bit of halfword
//! - `IS_BYTE[X, Y]` -> range check on a byte pair
//! - `IS_HALF[X + 256*Y]` -> range check on a halfword
//!
//! ## Table Structure
//!
//! 2¹⁶ = 65,536 rows indexed by `(X: Byte, Y: Byte)`. All lookups are received
//! with negative multiplicity (other tables send to this one).

use std::sync::OnceLock;

use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::BusInteraction;
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{TraceTable, columns2rows};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::bitwise::{BitwiseOperation, BitwiseOperationType};
use super::types::{FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices for BYTE_OPS table
// =========================================================================

pub mod cols {
    /// X: Byte input (0-255)
    pub const X: usize = 0;
    /// Y: Byte input (0-255)
    pub const Y: usize = 1;
    /// AND result: X & Y
    pub const AND: usize = 2;
    /// OR result: X | Y
    pub const OR: usize = 3;
    /// XOR result: X ^ Y
    pub const XOR: usize = 4;
    /// MSB of byte X: (X >> 7) & 1
    pub const MSB8: usize = 5;
    /// MSB of halfword X + 256*Y: ((X + 256*Y) >> 15) & 1
    pub const MSB16: usize = 6;

    /// Multiplicity for AND_BYTE lookups
    pub const MU_AND: usize = 7;
    /// Multiplicity for OR_BYTE lookups
    pub const MU_OR: usize = 8;
    /// Multiplicity for XOR_BYTE lookups
    pub const MU_XOR: usize = 9;
    /// Multiplicity for MSB8 lookups
    pub const MU_MSB8: usize = 10;
    /// Multiplicity for MSB16 lookups
    pub const MU_MSB16: usize = 11;
    /// Multiplicity for IS_BYTE lookups
    pub const MU_IS_BYTE: usize = 12;
    /// Multiplicity for IS_HALF lookups
    pub const MU_IS_HALF: usize = 13;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 14;
}

/// 2¹⁶ rows = 65,536.
pub const NUM_ROWS: usize = 256 * 256;

/// Number of precomputed (non-multiplicity) columns.
pub const NUM_PRECOMPUTED_COLS: usize = 7;

// =========================================================================
// Compile-time row generation
// =========================================================================

/// Generate one row of the byte_ops table.
///
/// Index encoding: `index = x + y * 256` with `x, y ∈ [0, 255]`.
#[inline]
pub const fn generate_byte_ops_row(index: usize) -> [u64; NUM_PRECOMPUTED_COLS] {
    let x = (index & 0xFF) as u32;
    let y = ((index >> 8) & 0xFF) as u32;

    let and_val = x & y;
    let or_val = x | y;
    let xor_val = x ^ y;

    let msb8 = (x >> 7) & 1;
    let halfword = x + y * 256;
    let msb16 = (halfword >> 15) & 1;

    [
        x as u64,
        y as u64,
        and_val as u64,
        or_val as u64,
        xor_val as u64,
        msb8 as u64,
        msb16 as u64,
    ]
}

/// Whether this table is preprocessed (commitment is hardcoded).
pub const fn is_preprocessed() -> bool {
    true
}

// =========================================================================
// Preprocessed commitment (computed once, cached)
// =========================================================================

static BYTE_OPS_COMMITMENT: OnceLock<Commitment> = OnceLock::new();

/// Computes the Merkle commitment over the precomputed byte_ops columns.
///
/// Mirrors [`bitwise::compute_preprocessed_commitment`] — see that for the
/// rationale (LDE-rooted commitment is required so FRI queries at any
/// blow-up index can be opened against this precomputed table).
fn compute_preprocessed_commitment(options: &ProofOptions) -> Commitment {
    #[cfg(feature = "parallel")]
    let columns: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
        .into_par_iter()
        .map(|col_idx| {
            (0..NUM_ROWS)
                .map(|idx| {
                    let row = generate_byte_ops_row(idx);
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
            let row = generate_byte_ops_row(idx);
            for (col_idx, &value) in row.iter().enumerate() {
                cols[col_idx].push(FE::from(value));
            }
        }
        cols
    };

    #[cfg(feature = "parallel")]
    let polys: Vec<Polynomial<FE>> = columns
        .par_iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for byte_ops column")
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for byte_ops column")
        })
        .collect();

    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);

    #[cfg(feature = "parallel")]
    let mut lde_columns: Vec<Vec<FE>> = polys
        .par_iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for byte_ops polynomial")
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let mut lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for byte_ops polynomial")
        })
        .collect();

    #[cfg(feature = "parallel")]
    lde_columns.par_iter_mut().for_each(|col| {
        in_place_bit_reverse_permute(col);
    });

    #[cfg(not(feature = "parallel"))]
    for col in lde_columns.iter_mut() {
        in_place_bit_reverse_permute(col);
    }

    let lde_rows = columns2rows(lde_columns);

    let tree = BatchedMerkleTree::<GoldilocksField>::build(&lde_rows)
        .expect("Failed to build Merkle tree for byte_ops LDE");

    tree.root
}

#[inline]
pub fn preprocessed_commitment(options: &ProofOptions) -> Commitment {
    *BYTE_OPS_COMMITMENT.get_or_init(|| compute_preprocessed_commitment(options))
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generate the precomputed BYTE_OPS trace table (multiplicities zeroed).
pub fn generate_byte_ops_trace() -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let mut data = vec![FE::zero(); NUM_ROWS * cols::NUM_COLUMNS];

    for x in 0u32..256 {
        for y in 0u32..256 {
            let row_idx = (x as usize) + (y as usize) * 256;
            let base = row_idx * cols::NUM_COLUMNS;

            data[base + cols::X] = FE::from(x as u64);
            data[base + cols::Y] = FE::from(y as u64);
            data[base + cols::AND] = FE::from((x & y) as u64);
            data[base + cols::OR] = FE::from((x | y) as u64);
            data[base + cols::XOR] = FE::from((x ^ y) as u64);

            let msb8 = (x >> 7) & 1;
            let halfword = x + y * 256;
            let msb16 = (halfword >> 15) & 1;
            data[base + cols::MSB8] = FE::from(msb8 as u64);
            data[base + cols::MSB16] = FE::from(msb16 as u64);

            // Multiplicity columns initialized to zero by the vec! above.
        }
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

#[inline]
pub fn row_index(x: u8, y: u8) -> usize {
    (x as usize) + (y as usize) * 256
}

/// Apply lookups to multiplicity columns.
///
/// Step 1 leaves this as a no-op — BITWISE still receives every byte-pair
/// bus, so byte_ops's multiplicity columns stay zeroed. Step 2 will route
/// AndByte/OrByte/XorByte/Msb8/Msb16/IsByte/IsHalf events here using the
/// same `BitwiseOperation` stream the BITWISE generator already produces.
pub fn update_multiplicities(
    _trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    _ops: &[BitwiseOperation],
) {
    // No-op until Step 2.
    let _ = BitwiseOperationType::AndByte; // keep import live
}

// =========================================================================
// Bus interactions (empty in Step 1; populated in Step 2)
// =========================================================================

/// Receivers for byte-pair lookups. Step 1 returns an empty list — BITWISE
/// keeps all receivers; Step 2 moves them here.
pub fn bus_interactions() -> Vec<BusInteraction> {
    Vec::new()
}

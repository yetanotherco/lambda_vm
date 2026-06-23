//! KECCAK_RC: Precomputed round constant lookup table for Keccak-f[1600].
//!
//! 24 rows (one per round), padded to 32. Each row maps a round index to its
//! 8-byte round constant. The round chip looks up `(round) → rc[8]` via the
//! `KeccakRc` bus.
//!
//! Follows the BITWISE preprocessed-table pattern: precomputed columns are
//! committed once and cached via `OnceLock`.

use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "prove")]
use std::sync::OnceLock;

use math::fft::bit_reversing::in_place_bit_reverse_permute;
use math::field::element::FieldElement;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{TraceTable, columns2rows};

use executor::constants::KECCAK_RC;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices
// =========================================================================

pub mod cols {
    /// Round index (0..23)
    pub const ROUND: usize = 0;
    /// RC bytes [0..7] — 8 bytes of the round constant (little-endian)
    pub const RC: usize = 1;
    pub const RC_END: usize = RC + 8; // = 9
    /// Multiplicity (how many times this row is looked up)
    pub const MU: usize = 9;

    pub const NUM_COLUMNS: usize = 10;
}

/// Number of precomputed columns (everything except MU).
pub const NUM_PRECOMPUTED_COLS: usize = 9;

/// Number of real rows (one per keccak round).
pub const NUM_REAL_ROWS: usize = 24;

/// Number of rows in the trace (padded to next power of 2).
pub const NUM_ROWS: usize = 32;

/// Whether this table is preprocessed.
pub const fn is_preprocessed() -> bool {
    true
}

/// Generate one precomputed row: [round, rc_byte0, ..., rc_byte7].
pub const fn generate_row(round: usize) -> [u64; NUM_PRECOMPUTED_COLS] {
    let rc_val = if round < 24 { KECCAK_RC[round] } else { 0 };
    [
        round as u64,
        rc_val & 0xFF,
        (rc_val >> 8) & 0xFF,
        (rc_val >> 16) & 0xFF,
        (rc_val >> 24) & 0xFF,
        (rc_val >> 32) & 0xFF,
        (rc_val >> 40) & 0xFF,
        (rc_val >> 48) & 0xFF,
        (rc_val >> 56) & 0xFF,
    ]
}

// =========================================================================
// Preprocessed commitment
// =========================================================================

#[cfg(feature = "prove")]
static KECCAK_RC_COMMITMENT: OnceLock<Commitment> = OnceLock::new();

fn compute_preprocessed_commitment(options: &ProofOptions) -> Commitment {
    // Generate precomputed columns
    let mut columns: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
        .map(|_| Vec::with_capacity(NUM_ROWS))
        .collect();
    for idx in 0..NUM_ROWS {
        let row = generate_row(idx);
        for (col_idx, &value) in row.iter().enumerate() {
            columns[col_idx].push(FE::from(value));
        }
    }

    // Interpolate each column to a polynomial
    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for keccak_rc column")
        })
        .collect();

    // Evaluate on LDE domain
    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);
    let mut lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for keccak_rc polynomial")
        })
        .collect();

    // Bit-reverse permute
    for col in lde_columns.iter_mut() {
        in_place_bit_reverse_permute(col);
    }

    // Build Merkle tree
    let lde_rows = columns2rows(lde_columns);
    let tree = BatchedMerkleTree::<GoldilocksField>::build(&lde_rows)
        .expect("Failed to build Merkle tree for keccak_rc LDE");

    tree.root
}

#[inline]
pub fn preprocessed_commitment(options: &ProofOptions) -> Commitment {
    #[cfg(feature = "prove")]
    {
        *KECCAK_RC_COMMITMENT.get_or_init(|| compute_preprocessed_commitment(options))
    }
    #[cfg(not(feature = "prove"))]
    {
        compute_preprocessed_commitment(options)
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generate the KECCAK_RC trace table.
///
/// All precomputed columns are filled; MU is initialized to zero and must be
/// updated via `update_multiplicities` after all round-chip lookups are known.
pub fn generate_keccak_rc_trace() -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let mut data = vec![FE::zero(); NUM_ROWS * cols::NUM_COLUMNS];

    for idx in 0..NUM_ROWS {
        let base = idx * cols::NUM_COLUMNS;
        let row = generate_row(idx);
        for (col_idx, &value) in row.iter().enumerate() {
            data[base + col_idx] = FE::from(value);
        }
        // MU = 0 (will be updated later)
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

/// Increment MU for each round lookup.
///
/// Called after the round chip's trace is generated. Each keccak permutation
/// call produces 24 round lookups (one per round), so each round row's MU
/// equals the number of keccak operations.
pub fn update_multiplicities(
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    num_keccak_ops: usize,
) {
    let mu = FieldElement::from(num_keccak_ops as u64);
    for round in 0..NUM_REAL_ROWS {
        let base = round * cols::NUM_COLUMNS;
        trace.main_table.data[base + cols::MU] = mu;
    }
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Single receiver on the KeccakRc bus.
///
/// Format: [round(Direct), rc[0](Direct), ..., rc[7](Direct)]
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut values = vec![BusValue::Packed {
        start_column: cols::ROUND,
        packing: Packing::Direct,
    }];
    for i in 0..8 {
        values.push(BusValue::Packed {
            start_column: cols::RC + i,
            packing: Packing::Direct,
        });
    }

    vec![BusInteraction::receiver(
        BusId::KeccakRc,
        Multiplicity::Column(cols::MU),
        values,
    )]
}

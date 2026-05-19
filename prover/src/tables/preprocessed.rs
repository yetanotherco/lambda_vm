//! Shared commitment pipeline for preprocessed tables.
//!
//! The DECODE, BITWISE, KECCAK_RC, PAGE, and REGISTER tables all commit their
//! precomputed columns through the same six steps:
//!
//! 1. Interpolate each column on the trace domain (FFT).
//! 2. Evaluate every polynomial on the LDE coset.
//! 3. Bit-reverse permute each LDE column.
//! 4. Transpose columns -> rows.
//! 5. Build a batched Merkle tree over the rows.
//! 6. Return the tree root.
//!
//! This module factors the pipeline out so each table only has to build its
//! own columns.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::columns2rows;

use super::types::{FE, GoldilocksField};

/// Run the full preprocessed-commitment pipeline on `columns`.
///
/// All columns must have the same length (the trace domain size, typically a
/// power of two). `table_label` is included in panic messages on failure of
/// the FFT / LDE / Merkle steps; these are construction-time failures on the
/// table's own data and indicate a bug in the code, never adversarial input.
pub fn commit_preprocessed_columns(
    columns: Vec<Vec<FE>>,
    options: &ProofOptions,
    table_label: &'static str,
) -> Commitment {
    let num_rows = columns[0].len();
    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);

    let interpolate = |col: &Vec<FE>| {
        Polynomial::interpolate_fft::<GoldilocksField>(col)
            .unwrap_or_else(|_| panic!("FFT interpolation failed for {table_label} column"))
    };
    let to_lde = |poly: &Polynomial<FE>| {
        evaluate_polynomial_on_lde_domain(poly, blowup_factor, num_rows, &coset_offset)
            .unwrap_or_else(|_| panic!("LDE evaluation failed for {table_label} polynomial"))
    };

    #[cfg(feature = "parallel")]
    let polys: Vec<Polynomial<FE>> = columns.par_iter().map(interpolate).collect();
    #[cfg(not(feature = "parallel"))]
    let polys: Vec<Polynomial<FE>> = columns.iter().map(interpolate).collect();

    #[cfg(feature = "parallel")]
    let mut lde_columns: Vec<Vec<FE>> = polys.par_iter().map(to_lde).collect();
    #[cfg(not(feature = "parallel"))]
    let mut lde_columns: Vec<Vec<FE>> = polys.iter().map(to_lde).collect();

    #[cfg(feature = "parallel")]
    lde_columns
        .par_iter_mut()
        .for_each(|col| in_place_bit_reverse_permute(col));
    #[cfg(not(feature = "parallel"))]
    for col in lde_columns.iter_mut() {
        in_place_bit_reverse_permute(col);
    }

    let lde_rows = columns2rows(lde_columns);
    let tree = BatchedMerkleTree::<GoldilocksField>::build(&lde_rows)
        .unwrap_or_else(|| panic!("Failed to build Merkle tree for {table_label} LDE"));
    tree.root
}

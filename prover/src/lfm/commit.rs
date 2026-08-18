//! Instruction-column-group commitment: interpolate → LDE → row-pair Merkle.
//!
//! The same pipeline the static preprocessed tables use (see
//! `tables/bitwise.rs::compute_preprocessed_commitment`), generalized over an
//! arbitrary column matrix so every LFM chip's group — and the registry
//! builder — shares one implementation. Host-side only; runs at program-build
//! and registry-regeneration time (seconds, not a ceremony — there is no
//! keygen in this framework).

use math::polynomial::Polynomial;
use stark::commitment::{ROWS_PER_LEAF, commit_bit_reversed};
use stark::config::Commitment;
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;

use crate::tables::types::{FE, GoldilocksField};

use super::compiler::ColumnGroup;

/// Commits a column matrix (each inner `Vec` one column, power-of-two height).
pub fn commit_columns(columns: &[Vec<FE>], options: &ProofOptions) -> Commitment {
    let num_rows = columns.first().map_or(0, Vec::len);
    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for LFM column group")
        })
        .collect();
    let coset_offset = FE::from(options.coset_offset);
    let lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(
                poly,
                options.blowup_factor as usize,
                num_rows,
                &coset_offset,
            )
            .expect("LDE evaluation failed for LFM column group")
        })
        .collect();
    let (_, root) = commit_bit_reversed(&lde_columns, ROWS_PER_LEAF)
        .expect("Merkle build failed for LFM column group");
    root
}

/// A [`ColumnGroup`]'s data, column-major (the commit pipeline's input shape).
pub fn group_columns(group: &ColumnGroup) -> Vec<Vec<FE>> {
    (0..group.width)
        .map(|c| (0..group.padded_rows).map(|r| *group.at(r, c)).collect())
        .collect()
}

/// Commits one instruction column group.
pub fn commit_group(group: &ColumnGroup, options: &ProofOptions) -> Commitment {
    commit_columns(&group_columns(group), options)
}

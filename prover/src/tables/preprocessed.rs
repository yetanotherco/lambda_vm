//! Shared commitment pipeline for preprocessed tables.
//!
//! DECODE, BITWISE, KECCAK_RC, PAGE, and REGISTER all commit their precomputed
//! columns the same way the prover commits any trace column: a single fused
//! coset-LDE pass per column (shared twiddles), then a bit-reversed batched
//! Merkle tree. This module is a thin adapter over `stark::prover::commit_lde_columns`
//! so the preprocessed-commitment path stays byte-identical to the prover's.

use stark::config::Commitment;
use stark::proof::options::ProofOptions;
use stark::prover::commit_lde_columns;

use super::types::FE;

/// Commit the precomputed `columns` of a preprocessed table and return the
/// Merkle root.
///
/// All columns must share the same power-of-two length and be non-empty.
/// `table_label` names the table in the invariant-failure message: an empty
/// column set is a table-definition bug, never adversarial input.
pub fn commit_preprocessed_columns(
    columns: Vec<Vec<FE>>,
    options: &ProofOptions,
    table_label: &'static str,
) -> Commitment {
    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);

    // `commit_lde_columns` only returns `None` for empty input. A preprocessed
    // table always has a fixed, non-empty column set, so empty `columns` here is
    // a table-definition bug — assert it explicitly (naming the table) rather
    // than letting it surface as an opaque commit failure downstream.
    assert!(
        !columns.is_empty() && !columns[0].is_empty(),
        "{table_label}: preprocessed table has no columns to commit (table-definition bug)",
    );

    // `F` is inferred as GoldilocksField from `columns` / `coset_offset` — the
    // prover crate is monomorphic over Goldilocks; the genericity lives in `stark`.
    commit_lde_columns(&columns, blowup_factor, &coset_offset)
        .expect("commit_lde_columns is infallible for the non-empty columns asserted above")
}

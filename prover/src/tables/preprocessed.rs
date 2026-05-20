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
/// All columns must share the same power-of-two length. `table_label` is only
/// used for the panic message if the commit fails — a commit failure here is a
/// code bug on the table's own data, never adversarial input.
pub fn commit_preprocessed_columns(
    columns: Vec<Vec<FE>>,
    options: &ProofOptions,
    table_label: &'static str,
) -> Commitment {
    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);

    // `F` is inferred as GoldilocksField from `columns` / `coset_offset` — the
    // prover crate is monomorphic over Goldilocks; the genericity lives in `stark`.
    commit_lde_columns(&columns, blowup_factor, &coset_offset)
        .unwrap_or_else(|| panic!("failed to commit preprocessed columns for {table_label}"))
}

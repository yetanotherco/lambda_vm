//! Instruction-column-group commitment: interpolate → LDE → row-pair Merkle.
//!
//! The same pipeline the static preprocessed tables use (see
//! `tables/bitwise.rs::compute_preprocessed_commitment`), generalized over an
//! arbitrary column matrix so every LFM chip's group — and the registry
//! builder — shares one implementation. Host-side only; runs at program-build
//! and registry-regeneration time (seconds, not a ceremony — there is no
//! keygen in this framework).

use math::polynomial::Polynomial;
use stark::commitment::{ROWS_PER_LEAF, commit_bit_reversed_with};
use stark::config::Commitment;
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;

use crate::tables::types::{FE, GoldilocksField};

use super::compiler::ColumnGroup;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Whether the instruction-group commit pass runs its column loops in parallel.
///
/// ⛔ THE A/B CONTROL FOR THE PARALLEL BUILD, and it is a separate knob from
/// [`groups_in_flight`](super::registry::groups_in_flight) on purpose.
/// `LFM_ARTIFACT_GROUPS_IN_FLIGHT=1` bounds RESIDENCY and leaves the columns
/// parallel; `LFM_ARTIFACT_PARALLEL=0` restores the pre-change walk exactly —
/// serial columns, serial groups.
///
/// ⚠ `RAYON_NUM_THREADS=1` is NOT this control. It also serializes work that was
/// already parallel before this change (the Merkle leaf hashing in
/// `stark::commitment`, and the fixed tables' own commitments), so an A/B taken
/// that way attributes their speedup to this commit.
pub fn parallel_build() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(
        // An EMPTY value reads as unset: `FOO= cmd` is the shell's way of
        // clearing a variable, and panicking on it would fail a run for a
        // spelling of "default".
        || match std::env::var("LFM_ARTIFACT_PARALLEL").ok().as_deref() {
            Some("0") => false,
            None | Some("") | Some("1") => true,
            Some(other) => panic!("LFM_ARTIFACT_PARALLEL must be `0` or `1`, got `{other}`"),
        },
    )
}

/// The coset LDE of a column matrix, column-major and in NATURAL order.
///
/// Split out of [`commit_columns`] so a caller that needs the evaluations for
/// something else can expand once and commit from the same copy, rather than
/// building a second, independently expanded one.
///
/// # One pass per column, and it is both faster and smaller
///
/// This used to interpolate EVERY column into a `Vec<Polynomial<FE>>` and then
/// walk that vector expanding each — so the whole coefficient set, one full copy
/// of the group, was alive at once beside the input and the growing LDE. Each
/// column's interpolate → expand is independent of every other's, so the two
/// loops are now one: a column's coefficients are born and consumed inside the
/// same closure and only the workers' in-flight polynomials are ever live.
///
/// That is what makes the pass parallelizable, and the parallelism is where lane
/// P's 22.0 s per node goes. ★ The output is UNCHANGED — same operations on the
/// same column in the same order, and `collect` on an indexed parallel iterator
/// preserves position — which the registry drift tests pin against blessed
/// roots.
pub fn lde_columns(columns: &[Vec<FE>], options: &ProofOptions) -> Vec<Vec<FE>> {
    let num_rows = columns.first().map_or(0, Vec::len);
    let coset_offset = FE::from(options.coset_offset);
    let expand = |col: &Vec<FE>| {
        let poly = Polynomial::interpolate_fft::<GoldilocksField>(col)
            .expect("FFT interpolation failed for LFM column group");
        evaluate_polynomial_on_lde_domain(
            &poly,
            options.blowup_factor as usize,
            num_rows,
            &coset_offset,
        )
        .expect("LDE evaluation failed for LFM column group")
    };
    #[cfg(feature = "parallel")]
    if parallel_build() {
        return columns.par_iter().map(expand).collect();
    }
    columns.iter().map(expand).collect()
}

/// Commits an already-expanded LDE column matrix.
pub fn commit_lde_columns(lde_columns: &[Vec<FE>]) -> Commitment {
    // ★ Under the block path's PIN, not `stark`'s default aliases. These commit
    // the production tables whose roots `lfm_program_id` names, so the hash that
    // BUILDS them and the hash the program identity CLAIMS have to be the same
    // one — `registry.rs` records that as the condition under which this read
    // moves, and the pin is what moved it.
    let (_, root) = commit_bit_reversed_with::<
        GoldilocksField,
        <crate::hash_pin::BlockStarkHash as stark::config::StarkHash>::Batched<GoldilocksField>,
    >(lde_columns, ROWS_PER_LEAF)
    .expect("Merkle build failed for LFM column group");
    root
}

/// Commits a column matrix (each inner `Vec` one column, power-of-two height).
pub fn commit_columns(columns: &[Vec<FE>], options: &ProofOptions) -> Commitment {
    commit_lde_columns(&lde_columns(columns, options))
}

/// A [`ColumnGroup`]'s data, column-major (the commit pipeline's input shape).
///
/// A strided gather: the group is row-major, so column `c` is read with stride
/// `width`. Parallel over columns because each output column is written by
/// exactly one closure from a shared `&group` — and because on a production node
/// this moves hundreds of megabytes before any FFT starts.
pub fn group_columns(group: &ColumnGroup) -> Vec<Vec<FE>> {
    let column = |c: usize| (0..group.padded_rows).map(|r| *group.at(r, c)).collect();
    #[cfg(feature = "parallel")]
    if parallel_build() {
        return (0..group.width).into_par_iter().map(column).collect();
    }
    (0..group.width).map(column).collect()
}

/// Commits one instruction column group.
pub fn commit_group(group: &ColumnGroup, options: &ProofOptions) -> Commitment {
    commit_columns(&group_columns(group), options)
}

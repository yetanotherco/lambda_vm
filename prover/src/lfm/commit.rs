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
use stark::fri::mmcs::{BorrowedMatrix, StreamingMmcsBuilder};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;

use crate::tables::types::{FE, GoldilocksField};

use super::compiler::ColumnGroup;

/// The coset LDE of a column matrix, column-major and in NATURAL order.
///
/// Split out of [`commit_columns`] because two things now consume it: the
/// per-slot row-pair commitment below, and the batched preprocessed round
/// ([`prep_round_root`]), which reads exactly this shape through
/// `BorrowedMatrix::ColMajorNatural`. Computing it once and handing it to both
/// is what keeps the batched root a commitment to *the same* evaluations the
/// per-slot root commits to, rather than to a second, independently built copy
/// of them.
pub fn lde_columns(columns: &[Vec<FE>], options: &ProofOptions) -> Vec<Vec<FE>> {
    let num_rows = columns.first().map_or(0, Vec::len);
    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for LFM column group")
        })
        .collect();
    let coset_offset = FE::from(options.coset_offset);
    polys
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
        .collect()
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
pub fn group_columns(group: &ColumnGroup) -> Vec<Vec<FE>> {
    (0..group.width)
        .map(|c| (0..group.padded_rows).map(|r| *group.at(r, c)).collect())
        .collect()
}

/// Commits one instruction column group.
pub fn commit_group(group: &ColumnGroup, options: &ProofOptions) -> Commitment {
    commit_columns(&group_columns(group), options)
}

/// The batched preprocessed round's root: ONE mixed-height MMCS over several
/// slots' LDE matrices, in slot order.
///
/// # What this is for
///
/// Under the batched commitment path a query opens ONE authentication path
/// covering every preprocessed matrix, instead of one path per slot. This is
/// the root such a verifier compares against
/// ([`stark::fri::mmcs::MixedMmcs::verify_batch`]), and the registry pins it
/// alongside the per-slot roots it does not replace.
///
/// # Streaming, deliberately
///
/// Absorbing through [`StreamingMmcsBuilder`] rather than `MixedMmcs::commit`
/// is what lets the caller expand one slot's LDE, commit it, absorb it and drop
/// it. `commit` reads every matrix of a height group at once, which for the
/// registry builder would mean holding all twelve groups' LDEs simultaneously —
/// a memory regression in a function the king gate and a dozen tests call.
///
/// # Determinism
///
/// The tree is a pure function of the matrices AND their order, so the caller
/// must absorb in the same slot order a verifier will present openings in. The
/// heights are LDE heights (`log2(rows * blowup)`), not trace heights — the
/// registry's own `log_heights` are trace heights, and the two differ by
/// `log2(blowup)`.
pub struct PrepRoundBuilder {
    builder: StreamingMmcsBuilder<GoldilocksField, crate::hash_pin::BlockStarkHash>,
}

impl PrepRoundBuilder {
    /// Declare the round's shape: `(log_height, width)` per participating slot,
    /// in absorption order. `log_height` is the LDE height.
    pub fn new(dims: &[(usize, usize)]) -> Self {
        Self {
            builder: StreamingMmcsBuilder::new(dims),
        }
    }

    /// Absorb one slot's LDE matrix. The caller may drop it as soon as this
    /// returns.
    ///
    /// # Panics
    ///
    /// On an empty matrix, or a column length that is not a power of two.
    /// Deriving the height as `len.trailing_zeros()` is only the height when the
    /// length is a power of two — for anything else it silently reports a
    /// SMALLER height (a length of 12 reads as 4), and the round would then
    /// commit a tree over a shape nobody declared. This runs at program-build
    /// and registry-regeneration time, never on a verify path, so an unusable
    /// input is a caller bug and asserting is correct here (unlike on the
    /// verifier, where the house rule is to reject rather than panic).
    pub fn absorb(&mut self, lde_columns: &[Vec<FE>]) {
        let len = lde_columns
            .first()
            .map(Vec::len)
            .expect("a participating slot has at least one column");
        assert!(
            len.is_power_of_two(),
            "an LDE column length must be a power of two, got {len}"
        );
        let log_height = len.trailing_zeros() as usize;
        let source = vec![BorrowedMatrix::ColMajorNatural {
            cols: lde_columns,
            log_height,
        }];
        self.builder.absorb(&source, 0);
    }

    /// The round's root.
    pub fn finish(self) -> Commitment {
        self.builder.finish().root()
    }
}

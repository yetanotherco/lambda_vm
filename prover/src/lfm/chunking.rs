//! `KECCAK_RND` chunking — how the hosted keccak family scales past one table.
//!
//! `KECCAK_RND` costs 24 rows per permutation at 1480 columns, so a single
//! instance saturates a 2^19-row table at ~21.8k permutations while a real
//! proof wrap needs ~460k. The RV64 VM solves the same problem for its own
//! tables by splitting them into chunk-AIRs; LFM does the same, with one
//! simplification: **the chunk count is static program shape**, fixed by
//! [`KeccakChunking`] at compile time, pinned in the registry and bound into
//! the program digest — never derived at prove time and never read off the
//! proof.
//!
//! # Why splitting the rows is free
//!
//! `KECCAK_RND` has no row-to-row transition constraints at all (its
//! [`ConstraintSet`](crate::tables::keccak_rnd::KeccakRndConstraints) is 20
//! per-row `IS_BIT` checks). The 24-round chain is carried entirely by the
//! `Keccak` bus: row *r* receives `(tag, r, state)` and sends `(tag, r+1,
//! out)`, so consecutive rounds are linked by token *matching*, not by row
//! adjacency. LogUp balances the multiset over every AIR in the proof, so it
//! cannot tell which instance a row lived in. That is what makes chunking need
//! zero pairing logic — the same property the VM's chunked tables rely on.
//!
//! # What is *not* chunked
//!
//! `KECCAK_RC` and `BITWISE` stay single shared instances. Both are receivers
//! whose multiplicity columns count lookups from the whole proof:
//! `keccak_rc::update_multiplicities` writes the total permutation count into
//! every round row, and `bitwise::BitwiseHistogram` accumulates every operation
//! before the trace is filled. Per-chunk copies would each have to carry the
//! full histogram and would then over-receive. Their sizes are fixed anyway
//! (32 and 2^20 rows), so they never needed splitting.

/// Trace rows one permutation occupies in `KECCAK_RND` — one per round.
pub const KECCAK_RND_ROWS_PER_PERMUTATION: usize = 24;

/// Rows per `KECCAK_RND` chunk in the default policy.
///
/// Retuning knob: this trades sub-proof count against per-chunk prover memory,
/// exactly like `max_rows` does for the VM's split tables. 2^19 rows is 21,845
/// permutations per chunk.
pub const KECCAK_RND_MAX_CHUNK_ROWS: usize = 1 << 19;

/// How a program's permutations are distributed over `KECCAK_RND` instances.
///
/// Carried on [`LfmProgram`](super::compiler::LfmProgram), so trace generation
/// and artifact building read the same policy and cannot disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeccakChunking {
    permutations_per_chunk: usize,
}

impl KeccakChunking {
    /// The policy that fills chunks to at most `max_rows` trace rows.
    ///
    /// Panics at compile time (it is `const`) if `max_rows` cannot hold a
    /// single permutation.
    pub const fn from_max_rows(max_rows: usize) -> Self {
        let permutations_per_chunk = max_rows / KECCAK_RND_ROWS_PER_PERMUTATION;
        assert!(
            permutations_per_chunk > 0,
            "a KECCAK_RND chunk must hold at least one permutation (24 rows)"
        );
        Self {
            permutations_per_chunk,
        }
    }

    /// The policy that puts at most `permutations_per_chunk` permutations in
    /// each chunk. The small-limit constructor tests use to force several
    /// chunks out of a tiny program.
    pub const fn from_permutations(permutations_per_chunk: usize) -> Self {
        assert!(
            permutations_per_chunk > 0,
            "a KECCAK_RND chunk must hold at least one permutation"
        );
        Self {
            permutations_per_chunk,
        }
    }

    pub const fn permutations_per_chunk(self) -> usize {
        self.permutations_per_chunk
    }

    /// Number of `KECCAK_RND` instances a program with `num_permutations`
    /// permutations gets — never zero, so the chip is present (and its
    /// constraints verified) even for a program containing no keccak at all.
    pub fn chunk_count(self, num_permutations: usize) -> usize {
        num_permutations
            .div_ceil(self.permutations_per_chunk)
            .max(1)
    }

    /// Splits per-permutation records into exactly [`Self::chunk_count`]
    /// slices. The single rule both trace generation and the artifact/AIR
    /// shape derive from; `split_agrees_with_chunk_count` pins the agreement.
    pub fn split<T>(self, permutations: &[T]) -> Vec<&[T]> {
        if permutations.is_empty() {
            vec![&permutations[..0]]
        } else {
            permutations.chunks(self.permutations_per_chunk).collect()
        }
    }
}

impl Default for KeccakChunking {
    fn default() -> Self {
        Self::from_max_rows(KECCAK_RND_MAX_CHUNK_ROWS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_is_the_documented_geometry() {
        let c = KeccakChunking::default();
        assert_eq!(c.permutations_per_chunk(), 21845);
        assert!(
            c.permutations_per_chunk() * KECCAK_RND_ROWS_PER_PERMUTATION
                <= KECCAK_RND_MAX_CHUNK_ROWS
        );
        // One chunk up to the limit, two past it.
        assert_eq!(c.chunk_count(21845), 1);
        assert_eq!(c.chunk_count(21846), 2);
        // The ~460k-permutation wrap the design targets.
        assert_eq!(c.chunk_count(460_000), 22);
    }

    #[test]
    fn empty_programs_still_get_one_chunk() {
        for per in [1usize, 2, 7, 21845] {
            let c = KeccakChunking::from_permutations(per);
            assert_eq!(c.chunk_count(0), 1);
            assert_eq!(c.split::<u8>(&[]).len(), 1);
            assert!(c.split::<u8>(&[])[0].is_empty());
        }
    }

    /// `split` and `chunk_count` are the same rule seen twice; if they ever
    /// disagree the prover builds a different number of traces than the
    /// verifier builds AIRs.
    #[test]
    fn split_agrees_with_chunk_count() {
        for per in [1usize, 2, 3, 5, 24] {
            let c = KeccakChunking::from_permutations(per);
            for n in 0..40usize {
                let ops: Vec<usize> = (0..n).collect();
                let split = c.split(&ops);
                assert_eq!(
                    split.len(),
                    c.chunk_count(n),
                    "per={per} n={n}: split and chunk_count disagree"
                );
                assert_eq!(
                    split.iter().map(|s| s.len()).sum::<usize>(),
                    n,
                    "per={per} n={n}: split lost or duplicated permutations"
                );
                assert!(
                    split.iter().all(|s| s.len() <= per),
                    "per={per} n={n}: a chunk exceeded the limit"
                );
            }
        }
    }
}

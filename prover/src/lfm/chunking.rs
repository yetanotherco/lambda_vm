//! Row chunking — how the machine's splittable tables scale past one instance.
//!
//! Two chips are chunked, by the same mechanism and for the same reason:
//! [`KeccakChunking`] splits `KECCAK_RND`, and [`Blake3Chunking`] splits
//! `LFM_BLAKE3`. Everything the next paragraphs say about the first holds for
//! the second; the differences are collected under [`Blake3Chunking`].
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

/// The environment knob that turns `LFM_BLAKE3` chunking on, read at program
/// EMISSION time by the driver that emits the program.
///
/// Unset means one table — today's machine, byte for byte. Set to `k` means
/// chunks of at most `2^k` rows, i.e. `2^k` compressions.
pub const BLAKE3_MAX_CHUNK_ROWS_LOG2_ENV: &str = "LFM_BLAKE3_MAX_CHUNK_ROWS_LOG2";

/// Trace rows one compression occupies in `LFM_BLAKE3` — exactly one.
///
/// Named rather than inlined so the row arithmetic below reads the same as
/// [`KECCAK_RND_ROWS_PER_PERMUTATION`]'s, and so a chip that ever compressed
/// over several rows would be a one-line change here instead of a hunt.
pub const BLAKE3_ROWS_PER_COMPRESSION: usize = 1;

/// How a program's compressions are distributed over `LFM_BLAKE3` instances.
///
/// Carried on [`LfmProgram`](super::compiler::LfmProgram) beside
/// [`KeccakChunking`], read by trace generation and artifact building alike, and
/// bound into the program digest — the same discipline, for the same reason.
///
/// # Why this chip needs it
///
/// `LFM_BLAKE3` is one row per compression at 3,056 value columns, so its
/// matrix is WIDE rather than tall: the aggregation program's ~1.39M
/// compressions land in a 2^21 x 3,056 table whose blowup-2 LDE is a single
/// ~102 GB allocation. Splitting the rows turns that one transient into `n`
/// independent ones without changing a byte of what is proved.
///
/// # Why splitting the rows is free
///
/// Same property [`KeccakChunking`] rests on, checked on this chip: every
/// constraint `Blake3LfmConstraints` emits reads `main(0, ..)` — there is no
/// row-to-row coupling at all — and every bus interaction is a within-row token
/// gated by `MU` or by a per-word multiplicity column. A compression's inputs
/// and outputs travel on the `LfmMem` bus by address matching, and the addresses
/// are PREPROCESSED program data, so which instance a row lives in is invisible
/// to the balance. Unlike `LFM_KECCAK`, the chip carries no row-ordinal tag, so
/// there is not even a positional value to preserve.
///
/// # ★ Where this differs from `KECCAK_RND`, and it matters
///
/// `KECCAK_RND` has NO preprocessed columns, so its chunk count moves no root:
/// every instance is the identical AIR. `LFM_BLAKE3` carries an instruction
/// column group (addresses, multiplicities, `MU` — 20 columns), so **each chunk
/// is its own committed matrix with its own Merkle root and its own height**.
/// Chunk 0's root is slot 11's entry in the roots array — which is why a
/// single-chunk program is bit-identical to an unchunked one — and the roots of
/// chunks 1.. ride [`LfmArtifacts`](super::registry::LfmArtifacts) and are
/// folded into `program_id`. A chunked program is therefore a different program
/// identity by name, not merely a different layout.
///
/// # What is *not* chunked
///
/// `BITWISE`. The chip is a `ByteAlu`/`AreBytes` sender ~1,248 times per row,
/// and `BITWISE` is the shared receiver whose multiplicity columns count the
/// lookups of the WHOLE proof — `bitwise_ops_for` is handed the complete record
/// list regardless of how the rows were split. Per-chunk copies would each carry
/// the full histogram and over-receive. This is the same exclusion
/// [`KeccakChunking`] records for `KECCAK_RC` and `BITWISE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blake3Chunking {
    compressions_per_chunk: usize,
}

impl Blake3Chunking {
    /// One table, whatever the compression count — **the default**, and the
    /// machine as it stood before chunking existed.
    ///
    /// `usize::MAX` rather than an `Option` so there is ONE code path: every
    /// derivation below is the same arithmetic whether chunking is on or off,
    /// which is what makes "knob unset" mean "one chunk" rather than "skip the
    /// chunking code".
    pub const fn unbounded() -> Self {
        Self {
            compressions_per_chunk: usize::MAX,
        }
    }

    /// The policy that fills chunks to at most `max_rows` trace rows.
    pub const fn from_max_rows(max_rows: usize) -> Self {
        let compressions_per_chunk = max_rows / BLAKE3_ROWS_PER_COMPRESSION;
        assert!(
            compressions_per_chunk > 0,
            "an LFM_BLAKE3 chunk must hold at least one compression"
        );
        Self {
            compressions_per_chunk,
        }
    }

    /// The policy that puts at most `compressions_per_chunk` compressions in
    /// each chunk. The small-limit constructor tests use to force several chunks
    /// out of a tiny program.
    pub const fn from_compressions(compressions_per_chunk: usize) -> Self {
        assert!(
            compressions_per_chunk > 0,
            "an LFM_BLAKE3 chunk must hold at least one compression"
        );
        Self {
            compressions_per_chunk,
        }
    }

    /// The policy [`BLAKE3_MAX_CHUNK_ROWS_LOG2_ENV`] names, or `None` when it is
    /// unset.
    ///
    /// `None` rather than [`Self::unbounded`] so a caller can tell "the operator
    /// chose one table" from "the operator said nothing" and print accordingly;
    /// both produce the same shape.
    ///
    /// # Panics
    ///
    /// On a value that is not a `u32`, or one at or above `usize::BITS`. This is
    /// read once, at program-emission time, by a driver an operator launched —
    /// a typo there must stop the run, not silently prove a different shape than
    /// the one asked for.
    pub fn from_env() -> Option<Self> {
        Self::from_env_value(
            std::env::var(BLAKE3_MAX_CHUNK_ROWS_LOG2_ENV)
                .ok()
                .as_deref(),
        )
    }

    /// [`Self::from_env`] with the variable's value supplied.
    ///
    /// Split out so the parse is testable without mutating process-global state:
    /// `std::env::set_var` races every other thread of a parallel test binary,
    /// and a knob whose parsing is untested is a knob that silently proves the
    /// wrong shape.
    pub fn from_env_value(raw: Option<&str>) -> Option<Self> {
        let raw = raw?;
        let log2: u32 = raw.parse().unwrap_or_else(|_| {
            panic!("{BLAKE3_MAX_CHUNK_ROWS_LOG2_ENV} must be a base-2 row exponent, got {raw:?}")
        });
        assert!(
            log2 < usize::BITS,
            "{BLAKE3_MAX_CHUNK_ROWS_LOG2_ENV}={log2} is not a representable row count"
        );
        Some(Self::from_max_rows(1usize << log2))
    }

    pub const fn compressions_per_chunk(self) -> usize {
        self.compressions_per_chunk
    }

    /// Number of `LFM_BLAKE3` instances a program with `num_compressions`
    /// compressions gets — never zero, so the chip is present (and its
    /// constraints verified) even for a program containing no BLAKE3 at all.
    /// The chip MASK, not this, is what drops an unused family; see
    /// [`ChipSet::blake3_chunks`](super::airs::ChipSet::blake3_chunks).
    pub fn chunk_count(self, num_compressions: usize) -> usize {
        num_compressions
            .div_ceil(self.compressions_per_chunk)
            .max(1)
    }

    /// The half-open row range chunk `chunk` covers, clamped to
    /// `num_compressions`. The single rule the group split, the record split and
    /// the census heights all read, so they cannot disagree about a boundary.
    pub fn chunk_range(self, num_compressions: usize, chunk: usize) -> core::ops::Range<usize> {
        let start = self
            .compressions_per_chunk
            .saturating_mul(chunk)
            .min(num_compressions);
        let end = start
            .saturating_add(self.compressions_per_chunk)
            .min(num_compressions);
        start..end
    }

    /// Splits per-compression records into exactly [`Self::chunk_count`]
    /// slices — the analogue of [`KeccakChunking::split`].
    pub fn split<T>(self, compressions: &[T]) -> Vec<&[T]> {
        (0..self.chunk_count(compressions.len()))
            .map(|c| &compressions[self.chunk_range(compressions.len(), c)])
            .collect()
    }
}

impl Default for Blake3Chunking {
    fn default() -> Self {
        Self::unbounded()
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

    /// The default is ONE table at any scale — the property that makes the
    /// knob-unset machine the machine that was there before.
    #[test]
    fn the_blake3_default_is_a_single_table() {
        let c = Blake3Chunking::default();
        assert_eq!(c, Blake3Chunking::unbounded());
        for n in [0usize, 1, 1_000, 1_390_000, usize::MAX - 1] {
            assert_eq!(c.chunk_count(n), 1, "n={n} must stay one table");
        }
    }

    /// `2^k` rows is `2^k` compressions: the chip is one row per compression, so
    /// the row knob and the compression count are the same number.
    #[test]
    fn the_blake3_row_cap_is_a_compression_cap() {
        for log2 in [3usize, 10, 18] {
            let c = Blake3Chunking::from_max_rows(1 << log2);
            assert_eq!(c.compressions_per_chunk(), 1 << log2);
            assert_eq!(c.chunk_count(1 << log2), 1);
            assert_eq!(c.chunk_count((1 << log2) + 1), 2);
        }
        // The aggregation program's shape at the 2^18 target.
        assert_eq!(
            Blake3Chunking::from_max_rows(1 << 18).chunk_count(1_390_000),
            6
        );
    }

    /// `split`, `chunk_count` and `chunk_range` are one rule seen three times;
    /// if they ever disagree the prover builds a different number of traces than
    /// the verifier builds AIRs, or a chunk's rows and its records come from
    /// different boundaries.
    #[test]
    fn blake3_split_agrees_with_chunk_count_and_range() {
        for per in [1usize, 2, 3, 5, 8] {
            let c = Blake3Chunking::from_compressions(per);
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
                    "per={per} n={n}: split lost or duplicated compressions"
                );
                assert!(
                    split.iter().all(|s| s.len() <= per),
                    "per={per} n={n}: a chunk exceeded the limit"
                );
                for (i, s) in split.iter().enumerate() {
                    let r = c.chunk_range(n, i);
                    assert_eq!(
                        &ops[r], *s,
                        "per={per} n={n} chunk {i}: chunk_range and split disagree"
                    );
                }
            }
        }
    }

    /// The knob's own parse: unset is one table, and a value is a row exponent.
    #[test]
    fn the_blake3_env_knob_parses_a_row_exponent() {
        assert_eq!(Blake3Chunking::from_env_value(None), None);
        for log2 in [0usize, 3, 18, 21] {
            assert_eq!(
                Blake3Chunking::from_env_value(Some(&log2.to_string())),
                Some(Blake3Chunking::from_max_rows(1 << log2)),
                "{log2} must name 2^{log2} rows per chunk"
            );
        }
        // The aggregation program's target: 1.39M compressions at 2^18.
        assert_eq!(
            Blake3Chunking::from_env_value(Some("18"))
                .expect("set")
                .chunk_count(1_390_000),
            6
        );
    }

    /// A typo stops the run rather than silently proving a different shape.
    #[test]
    #[should_panic(expected = "must be a base-2 row exponent")]
    fn a_malformed_blake3_knob_panics() {
        let _ = Blake3Chunking::from_env_value(Some("2^18"));
    }

    /// An empty program still gets one chunk, so the chip stays in the set.
    #[test]
    fn empty_blake3_programs_still_get_one_chunk() {
        for per in [1usize, 2, 7] {
            let c = Blake3Chunking::from_compressions(per);
            assert_eq!(c.chunk_count(0), 1);
            assert_eq!(c.split::<u8>(&[]).len(), 1);
            assert!(c.split::<u8>(&[])[0].is_empty());
            assert_eq!(c.chunk_range(0, 0), 0..0);
        }
    }
}

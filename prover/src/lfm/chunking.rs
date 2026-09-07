//! Row chunking — how the machine's splittable tables scale past one instance.
//!
//! Three chips are chunked, by the same mechanism and for the same reason:
//! [`KeccakChunking`] splits `KECCAK_RND`, [`Blake3Chunking`] splits
//! `LFM_BLAKE3`, and [`BaluChunking`] splits `LFM_BALU` — the first two because
//! their matrices are WIDE, the third because its matrix is TALL. Everything
//! the next paragraphs say about the first holds for the others; the
//! differences are collected under [`Blake3Chunking`] and [`BaluChunking`].
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

/// Chunks `total` one-row records need at `per` records per chunk — never
/// zero, so a chip stays present for an empty program. The one rule
/// [`Blake3Chunking`] and [`BaluChunking`] share.
fn row_chunk_count(per: usize, total: usize) -> usize {
    total.div_ceil(per).max(1)
}

/// The half-open record range chunk `chunk` covers at `per` records per
/// chunk, clamped to `total`.
fn row_chunk_range(per: usize, total: usize, chunk: usize) -> core::ops::Range<usize> {
    let start = per.saturating_mul(chunk).min(total);
    let end = start.saturating_add(per).min(total);
    start..end
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
        row_chunk_count(self.compressions_per_chunk, num_compressions)
    }

    /// The half-open row range chunk `chunk` covers, clamped to
    /// `num_compressions`. The single rule the group split, the record split and
    /// the census heights all read, so they cannot disagree about a boundary.
    pub fn chunk_range(self, num_compressions: usize, chunk: usize) -> core::ops::Range<usize> {
        row_chunk_range(self.compressions_per_chunk, num_compressions, chunk)
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

/// The environment knob that turns `LFM_BALU` chunking on, read at program
/// EMISSION time like [`BLAKE3_MAX_CHUNK_ROWS_LOG2_ENV`].
///
/// Unset means one table — today's machine, byte for byte. Set to `k` means
/// chunks of at most `2^k` rows, i.e. `2^k` ALU operations.
pub const BALU_MAX_CHUNK_ROWS_LOG2_ENV: &str = "LFM_BALU_MAX_CHUNK_ROWS_LOG2";

/// Trace rows one ALU operation occupies in `LFM_BALU` — exactly one
/// (`emit_column_groups` opens one row per `Balu` instruction).
pub const BALU_ROWS_PER_OP: usize = 1;

/// The chunk height the sizing under [`BaluChunking`] arrives at: `2^22` rows.
pub const BALU_TARGET_CHUNK_ROWS_LOG2: u32 = 22;

/// How a program's ALU operations are distributed over `LFM_BALU` instances —
/// the row-chunking arm for the machine's TALL-NARROW chips.
///
/// # Why this chip needs it
///
/// `LFM_BALU` is one row per operation at 4 value + 10 preprocessed columns:
/// narrow, and on the aggregator very tall — the program pads to `2^27` rows
/// at 110 queries and `2^28` at 219, a census contribution that is trivial in
/// cells and enormous in rows. At blowup 2 the ONE-table device set of the R1
/// commit alone is
///
/// | rows | LDE `2n·14·8` | snapshot `n·14·8` | tree `(2n−1)·32` | R1 set |
/// |------|---------------|-------------------|------------------|--------|
/// | 2^27 | 28.0 GiB      | 14.0 GiB          | 8.0 GiB          | 50 GiB |
/// | 2^22 | 0.875 GiB     | 0.44 GiB          | 0.25 GiB         | 1.6 GiB|
///
/// and rounds 2–4 add the aux LDE (2 ext3 columns, `2n·2·24`) and its tree,
/// the two composition parts (`2·2n·24`) and their tree, and the DEEP
/// codeword (`2n·24`): a whole-prove set of ~96 GiB at `2^27` against a
/// 32 GiB card, ~3 GiB per chunk at `2^22` (`the_balu_chunk_sizing_is_the_doc`
/// pins the arithmetic). `2^22` is the height at which eight chunks prove
/// concurrently inside a 25.6 GiB admission budget, which is why it is the
/// target: `2^24` chunks (~12 GiB each) would hold the concurrency at two, and
/// `2^20` chunks would quadruple the per-chunk FRI and query overhead the
/// verifier pays for nothing. That overhead — one FRI commit and one set of
/// openings PER CHUNK — is the counter-pressure against smaller chunks, and
/// the reason the default stays one table until the aggregator is emitted
/// with the knob set.
///
/// # Why row chunking, not column streaming
///
/// Streaming the commit column group by column group lowers only the commit's
/// own peak; rounds 2–4 read every column of every row from the RESIDENT LDE,
/// so the `2n · cols · 8` buffer has to be on the card for the whole table
/// however the leaves were hashed. For a tall-narrow chip that buffer IS the
/// problem (28 GiB at `2^27`), and nothing short of splitting the rows shrinks
/// it. Column streaming is the shape for SHORT-WIDE chips (`LFM_HASH` at
/// `2^21 × 449`), and since the fused commit transposes its one LDE buffer in
/// place it buys little even there: the LDE stays resident for rounds 2–4
/// either way.
///
/// # Why splitting the rows is free
///
/// The property [`KeccakChunking`] and [`Blake3Chunking`] rest on, checked on
/// this chip: every constraint `BaluConstraints` emits reads `main(0, ..)` —
/// no row-to-row coupling — and every bus interaction is a within-row `LfmMem`
/// token gated by the row's own selector or multiplicity column. Operands and
/// results travel by address matching, and the addresses are PREPROCESSED
/// program data, so which instance a row lives in is invisible to the balance.
///
/// # What it costs
///
/// Like `LFM_BLAKE3` and unlike `KECCAK_RND`, this chip carries a preprocessed
/// instruction group, so **each chunk is its own committed matrix with its
/// own root and its own height**: a chunked program is a different program
/// identity by name, and the chunk roots ride the artifacts and fold into
/// `program_id` exactly as the BLAKE3 chunk roots do. Wiring — the program
/// field, the per-chunk group, the AIR instances, the artifact roots and the
/// slot map — follows the BLAKE3 template one for one and is not in this
/// module.
///
/// `LFM_LANES` (4 value + 12 preprocessed, `2^24` rows in the 110-query wrap)
/// is the next chip of this shape; its whole-prove set at `2^24` is ~14 GiB,
/// one doubling from needing the same arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaluChunking {
    ops_per_chunk: usize,
}

impl BaluChunking {
    /// One table, whatever the operation count — **the default**, and the
    /// machine as it stands before the aggregator is emitted chunked.
    /// `usize::MAX` rather than an `Option` for [`Blake3Chunking::unbounded`]'s
    /// reason: one code path.
    pub const fn unbounded() -> Self {
        Self {
            ops_per_chunk: usize::MAX,
        }
    }

    /// The policy that fills chunks to at most `max_rows` trace rows.
    pub const fn from_max_rows(max_rows: usize) -> Self {
        let ops_per_chunk = max_rows / BALU_ROWS_PER_OP;
        assert!(
            ops_per_chunk > 0,
            "an LFM_BALU chunk must hold at least one operation"
        );
        Self { ops_per_chunk }
    }

    /// The policy that puts at most `ops_per_chunk` operations in each chunk.
    /// The small-limit constructor tests use to force several chunks out of a
    /// tiny program.
    pub const fn from_ops(ops_per_chunk: usize) -> Self {
        assert!(
            ops_per_chunk > 0,
            "an LFM_BALU chunk must hold at least one operation"
        );
        Self { ops_per_chunk }
    }

    /// The sizing's target: chunks of `2^22` rows
    /// ([`BALU_TARGET_CHUNK_ROWS_LOG2`]).
    pub const fn target() -> Self {
        Self::from_max_rows(1usize << BALU_TARGET_CHUNK_ROWS_LOG2)
    }

    /// The policy [`BALU_MAX_CHUNK_ROWS_LOG2_ENV`] names, or `None` when it is
    /// unset — on [`Blake3Chunking::from_env`]'s terms, including the panic on
    /// a value that is not a row exponent.
    pub fn from_env() -> Option<Self> {
        Self::from_env_value(std::env::var(BALU_MAX_CHUNK_ROWS_LOG2_ENV).ok().as_deref())
    }

    /// [`Self::from_env`] with the variable's value supplied, so the parse is
    /// testable without mutating process-global state.
    pub fn from_env_value(raw: Option<&str>) -> Option<Self> {
        let raw = raw?;
        let log2: u32 = raw.parse().unwrap_or_else(|_| {
            panic!("{BALU_MAX_CHUNK_ROWS_LOG2_ENV} must be a base-2 row exponent, got {raw:?}")
        });
        assert!(
            log2 < usize::BITS,
            "{BALU_MAX_CHUNK_ROWS_LOG2_ENV}={log2} is not a representable row count"
        );
        Some(Self::from_max_rows(1usize << log2))
    }

    pub const fn ops_per_chunk(self) -> usize {
        self.ops_per_chunk
    }

    /// Number of `LFM_BALU` instances a program with `num_ops` operations gets
    /// — never zero, so the chip is present (and its constraints verified)
    /// even for a program containing no ALU operation at all.
    pub fn chunk_count(self, num_ops: usize) -> usize {
        row_chunk_count(self.ops_per_chunk, num_ops)
    }

    /// The half-open row range chunk `chunk` covers, clamped to `num_ops` —
    /// the single rule a group split, a record split and a census height
    /// would all read.
    pub fn chunk_range(self, num_ops: usize, chunk: usize) -> core::ops::Range<usize> {
        row_chunk_range(self.ops_per_chunk, num_ops, chunk)
    }

    /// Splits per-operation records into exactly [`Self::chunk_count`] slices.
    pub fn split<T>(self, ops: &[T]) -> Vec<&[T]> {
        (0..self.chunk_count(ops.len()))
            .map(|c| &ops[self.chunk_range(ops.len(), c)])
            .collect()
    }
}

impl Default for BaluChunking {
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

    /// The default is ONE `LFM_BALU` table at any scale — the machine as it
    /// stands.
    #[test]
    fn the_balu_default_is_a_single_table() {
        let c = BaluChunking::default();
        assert_eq!(c, BaluChunking::unbounded());
        for n in [0usize, 1, 1 << 27, 1 << 28, usize::MAX - 1] {
            assert_eq!(c.chunk_count(n), 1, "n={n} must stay one table");
        }
        assert_eq!(BaluChunking::from_env_value(None), None);
    }

    /// The target policy splits the aggregator's `2^27` (110 q) and `2^28`
    /// (219 q) rows into 32 and 64 chunks of `2^22`; the knob names the same
    /// policy.
    #[test]
    fn the_balu_target_sizes_the_aggregator() {
        let c = BaluChunking::target();
        assert_eq!(c.ops_per_chunk(), 1 << 22);
        assert_eq!(c.chunk_count(1 << 27), 32);
        assert_eq!(c.chunk_count(1 << 28), 64);
        assert_eq!(c.chunk_count((1 << 27) + 1), 33);
        assert_eq!(BaluChunking::from_env_value(Some("22")), Some(c));
        for log2 in [0usize, 3, 18, 22] {
            assert_eq!(
                BaluChunking::from_env_value(Some(&log2.to_string())),
                Some(BaluChunking::from_max_rows(1 << log2)),
                "{log2} must name 2^{log2} rows per chunk"
            );
        }
    }

    /// The device-set arithmetic the `BaluChunking` doc tabulates: one table at
    /// `2^27` does not fit a 32 GiB card; a `2^22` chunk's whole-prove set is
    /// ~3 GiB, so eight prove concurrently inside the 25.6 GiB budget. Columns:
    /// 14 base (4 value + 10 preprocessed), 2 ext3 aux, 2 ext3 composition
    /// parts, one ext3 DEEP codeword, blowup 2.
    #[test]
    fn the_balu_chunk_sizing_is_the_doc() {
        const GIB: u64 = 1 << 30;
        let whole_prove_set = |n: u64| -> u64 {
            let lde = 2 * n;
            let tree = (lde - 1) * 32;
            let main_lde = lde * 14 * 8;
            let snapshot = n * 14 * 8;
            let aux_lde = lde * 2 * 24;
            let parts = lde * 2 * 24;
            let deep = lde * 24;
            main_lde + snapshot + tree + aux_lde + tree + parts + tree + deep
        };
        let one_table = whole_prove_set(1 << 27);
        assert!(one_table > 95 * GIB && one_table < 97 * GIB, "{one_table}");
        let r1_only = (1u64 << 28) * 14 * 8 + (1u64 << 27) * 14 * 8 + ((1u64 << 28) - 1) * 32;
        assert!(r1_only > 49 * GIB && r1_only < 51 * GIB, "{r1_only}");
        let chunk = whole_prove_set(1 << 22);
        assert!(chunk < 3 * GIB + GIB / 16, "{chunk}");
        assert!(
            8 * chunk <= 32 * GIB / 5 * 4,
            "eight chunks must fit the budget"
        );
        let big_chunk = whole_prove_set(1 << 24);
        assert!(2 * big_chunk <= 32 * GIB / 5 * 4 && 3 * big_chunk > 32 * GIB / 5 * 4);
    }

    /// `split`, `chunk_count` and `chunk_range` are one rule seen three times
    /// for this chip too.
    #[test]
    fn balu_split_agrees_with_chunk_count_and_range() {
        for per in [1usize, 2, 3, 5, 8] {
            let c = BaluChunking::from_ops(per);
            for n in 0..40usize {
                let ops: Vec<usize> = (0..n).collect();
                let split = c.split(&ops);
                assert_eq!(split.len(), c.chunk_count(n), "per={per} n={n}");
                assert_eq!(split.iter().map(|s| s.len()).sum::<usize>(), n);
                assert!(split.iter().all(|s| s.len() <= per));
                for (i, s) in split.iter().enumerate() {
                    assert_eq!(&ops[c.chunk_range(n, i)], *s, "per={per} n={n} chunk {i}");
                }
            }
            assert_eq!(c.chunk_count(0), 1);
            assert_eq!(c.chunk_range(0, 0), 0..0);
        }
    }

    /// A typo stops the run rather than silently proving a different shape.
    #[test]
    #[should_panic(expected = "must be a base-2 row exponent")]
    fn a_malformed_balu_knob_panics() {
        let _ = BaluChunking::from_env_value(Some("2^22"));
    }
}

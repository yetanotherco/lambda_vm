//! The LFM straight-line compiler.
//!
//! Pass 1 backfills static multiplicities from the builder's read counters,
//! guarded by the two invariant panics (tripwires — the release-mode
//! admission validator is the gate, the registry is the record):
//!   - **panic #1**: an address assigned twice (write-once violated in the
//!     builder itself);
//!   - **panic #2**: the read-count map is not drained after backfill (a
//!     read of an address no instruction writes).
//!
//! Pass 2 emits the per-chip **instruction column groups** — the preprocessed
//! matrices whose Merkle roots become the program's identity. Layouts live in
//! [`super::layout`]; group commitment (interpolate → LDE → Merkle) is wired
//! at registry-build time (Milestone B) through the same pipeline the static
//! tables use.

use crate::tables::types::FE;

use super::builder::{ArenaSchema, LfmProgramSource};
use super::chunking::{Blake3Chunking, HashChunking, KeccakChunking};
use super::instr::{Addr, BaseOp, ExtOp, HashMode, Instr, KeccakMode};
use super::layout::{self, padded_rows};

/// One chip's instruction column group: a row-major matrix, zero-padded to a
/// power-of-two height (min 4).
#[derive(Debug, Clone)]
pub struct ColumnGroup {
    pub width: usize,
    pub real_rows: usize,
    pub padded_rows: usize,
    /// Row-major, `padded_rows × width`.
    pub data: Vec<FE>,
}

/// Accumulates one chip's rows directly into the flat row-major buffer the
/// finished [`ColumnGroup`] holds.
///
/// The emitter used to collect `Vec<Vec<FE>>` and copy it row by row. That kept
/// two full materializations of every group alive at once and paid a heap
/// allocation per instruction — and the per-row `Vec`s over-allocate badly,
/// because they are grown by `extend`/`push` rather than sized: a 10-wide BALU
/// row lands at capacity 18. Appending into one buffer removes the second
/// materialization, the headers, the rounding waste and the ~271M malloc/free
/// pairs.
///
/// Rows come out bit-identical: the same values are written at the same
/// row-major offsets, and the tail is zero-padded to the same height.
struct ColumnGroupBuilder {
    width: usize,
    /// The ordinal of this builder's first row in the whole group: 0, or a
    /// later slice's start when a program's rows are laid out in slices
    /// ([`compile_parallel`]).
    first: usize,
    real_rows: usize,
    data: Vec<FE>,
}

impl ColumnGroupBuilder {
    fn new(width: usize) -> Self {
        Self::starting_at(width, 0)
    }

    /// A builder whose first row is the group's row `first`.
    fn starting_at(width: usize, first: usize) -> Self {
        ColumnGroupBuilder {
            width,
            first,
            real_rows: 0,
            data: Vec::new(),
        }
    }

    /// The ordinal the next row will take in the whole group. `LFM_KECCAK`
    /// and the lean head bind it into the row as a structural tag, so it has
    /// to be read before [`Self::open_row`].
    fn next_row(&self) -> usize {
        self.first + self.real_rows
    }

    /// Append a zero-filled row, returning its base offset for [`Self::set`].
    fn open_row(&mut self) -> usize {
        let base = self.data.len();
        self.data.resize(base + self.width, FE::zero());
        self.real_rows += 1;
        base
    }

    fn set(&mut self, base: usize, col: usize, v: FE) {
        debug_assert!(
            col < self.width,
            "column {col} outside width {}",
            self.width
        );
        self.data[base + col] = v;
    }

    fn finish(mut self) -> ColumnGroup {
        let padded = padded_rows(self.real_rows);
        self.data.resize(padded * self.width, FE::zero());
        ColumnGroup {
            width: self.width,
            real_rows: self.real_rows,
            padded_rows: padded,
            data: self.data,
        }
    }
}

impl ColumnGroup {
    pub fn at(&self, row: usize, col: usize) -> &FE {
        &self.data[row * self.width + col]
    }

    pub fn set(&mut self, row: usize, col: usize, v: FE) {
        self.data[row * self.width + col] = v;
    }
}

/// The program-dependent instruction column groups, in the frozen chip order.
/// (`LFM_RANGE`'s group is program-independent and materialized at commitment
/// time; the three hosted keccak-family tables carry no LFM group at all.)
#[derive(Debug, Clone)]
pub struct LfmColumnGroups {
    pub const_: ColumnGroup,
    pub balu: ColumnGroup,
    pub xalu: ColumnGroup,
    pub select: ColumnGroup,
    pub bitdec: ColumnGroup,
    pub hash: ColumnGroup,
    pub keccak: ColumnGroup,
    pub blake3: ColumnGroup,
    pub lanes: ColumnGroup,
    pub hint: ColumnGroup,
    pub public: ColumnGroup,
    /// `B3L_HEAD`'s instruction group: one row an [`Instr::B3l`]
    /// ([`layout::b3l`]). Empty in every program that has no such row, and
    /// outside the fixed STARK chip set: only the block prover proves it.
    pub b3l: ColumnGroup,
    /// `LEAN_MUX`'s instruction group: one row an [`Instr::Mux`]
    /// ([`layout::mux`]), on the same terms as [`Self::b3l`].
    pub mux: ColumnGroup,
    /// `LEAN_CAP`'s instruction group: one row a node of an
    /// [`Instr::CapTable`] ([`layout::cap`]), tables in program order, on the
    /// same terms as [`Self::b3l`].
    pub cap: ColumnGroup,
    /// `LEAN_CAP_READ`'s instruction group: one row an [`Instr::CapRead`]
    /// ([`layout::cap_read`]), on the same terms as [`Self::b3l`].
    pub cap_read: ColumnGroup,
}

/// A compiled LFM program: multiplicity-backfilled instructions plus the
/// emitted instruction column groups.
#[derive(Debug)]
pub struct LfmProgram {
    pub instrs: Vec<Instr>,
    pub num_addrs: u64,
    pub arena_schema: ArenaSchema,
    pub public_len: u32,
    pub groups: LfmColumnGroups,
    /// How this program's permutations are spread over `KECCAK_RND`
    /// instances. Program shape, not a runtime knob: it is fixed here, bound
    /// into the program digest and pinned in the registry.
    pub chunking: KeccakChunking,
    /// How this program's compressions are spread over `LFM_BLAKE3` instances.
    /// Program shape on the same terms as [`Self::chunking`] — and with one
    /// consequence that policy does not have: `LFM_BLAKE3` carries a
    /// preprocessed instruction group, so each chunk commits its own root.
    /// Defaults to a single table, which is the machine as it stood before
    /// chunking existed.
    pub blake3_chunking: Blake3Chunking,
    /// How this program's hash rows are spread over `LFM_HASH` instances — one
    /// table unless the hash split divides it ([`HashChunking`]). Program shape on
    /// [`Self::blake3_chunking`]'s terms: each chunk commits its own
    /// instruction group, and the split is bound into `program_id`.
    pub hash_chunking: HashChunking,
    /// Whether this program's `LFM_HASH` rows are width-16 (`Instr::Hash16`):
    /// it then proves under `HasherKind::Poseidon1W16`
    /// ([`LfmProgram::hasher`]). Program shape, read off the instructions at
    /// compile time; the admission validator refuses a program that mixes the
    /// two widths. Private: only [`compile`] sets it, from the instructions
    /// (the socket review's A4); read it through [`LfmProgram::hash16`].
    hash16: bool,
}

impl LfmProgram {
    /// Whether this program's `LFM_HASH` rows are width-16 (`Instr::Hash16`).
    pub fn hash16(&self) -> bool {
        self.hash16
    }

    /// The `LFM_HASH` permutation this program proves under: the width-16
    /// socket's for a program with `Hash16` rows, else `default` (the block
    /// hasher every other program uses).
    pub fn hasher(&self, default: super::hash::HasherKind) -> super::hash::HasherKind {
        if self.hash16 {
            super::hash::HasherKind::Poseidon1W16
        } else {
            default
        }
    }

    /// Replaces the `KECCAK_RND` chunking policy.
    ///
    /// Chunking affects only how the round-chip rows are distributed over AIR
    /// instances — never what is compiled — so it is safe to set after
    /// compilation. Tests use it to force several chunks out of a program with
    /// a handful of permutations; retuning uses it to size chunks per preset.
    pub fn with_keccak_chunking(mut self, chunking: KeccakChunking) -> Self {
        self.chunking = chunking;
        self
    }

    /// Replaces the `LFM_BLAKE3` chunking policy.
    ///
    /// Safe after compilation for [`Self::with_keccak_chunking`]'s reason —
    /// chunking redistributes rows over AIR instances and never changes what is
    /// compiled — but note the extra consequence: this chip's chunks each commit
    /// their own instruction group, so a program re-chunked after
    /// `build_artifacts` has been called needs FRESH artifacts, and its
    /// `program_id` differs from the unchunked one's.
    pub fn with_blake3_chunking(mut self, chunking: Blake3Chunking) -> Self {
        self.blake3_chunking = chunking;
        self
    }

    /// `LFM_BLAKE3` instances this program's policy asks for — at least one,
    /// even with no compressions at all. The chip MASK is what drops an unused
    /// family; see [`ChipSet::blake3_chunks`](super::airs::ChipSet::blake3_chunks).
    pub fn blake3_chunk_count(&self) -> usize {
        self.blake3_chunking
            .chunk_count(self.groups.blake3.real_rows)
    }

    /// Replaces the `LFM_HASH` chunking policy — on
    /// [`Self::with_blake3_chunking`]'s terms, fresh artifacts included.
    pub fn with_hash_chunking(mut self, chunking: HashChunking) -> Self {
        self.hash_chunking = chunking;
        self
    }

    /// `LFM_HASH` instances this program's policy asks for — one or two.
    pub fn hash_chunk_count(&self) -> usize {
        self.hash_chunking.chunk_count(self.groups.hash.real_rows)
    }

    /// Hash rows in each `LFM_HASH` chunk, in chunk order.
    pub fn hash_chunk_real_rows(&self) -> Vec<usize> {
        let total = self.groups.hash.real_rows;
        (0..self.hash_chunk_count())
            .map(|c| self.hash_chunking.chunk_range(total, c).len())
            .collect()
    }

    /// One `LFM_HASH` chunk's instruction column group, re-padded to its own
    /// power-of-two height — [`Self::blake3_chunk_group`] for the hash chip.
    ///
    /// # Panics
    ///
    /// On a chunk index at or past [`Self::hash_chunk_count`].
    pub fn hash_chunk_group(&self, chunk: usize) -> ColumnGroup {
        let count = self.hash_chunk_count();
        assert!(
            chunk < count,
            "LFM_HASH chunk {chunk} requested of a {count}-chunk program"
        );
        let group = &self.groups.hash;
        if count == 1 {
            return group.clone();
        }
        let rows = self.hash_chunking.chunk_range(group.real_rows, chunk);
        let real_rows = rows.len();
        let padded_rows = padded_rows(real_rows);
        let mut data = group.data[rows.start * group.width..rows.end * group.width].to_vec();
        data.resize(padded_rows * group.width, FE::zero());
        ColumnGroup {
            width: group.width,
            real_rows,
            padded_rows,
            data,
        }
    }

    /// Compressions in each `LFM_BLAKE3` chunk, in chunk order.
    pub fn blake3_chunk_real_rows(&self) -> Vec<usize> {
        let total = self.groups.blake3.real_rows;
        (0..self.blake3_chunk_count())
            .map(|c| self.blake3_chunking.chunk_range(total, c).len())
            .collect()
    }

    /// One `LFM_BLAKE3` chunk's instruction column group: the rows the policy
    /// assigns it, re-padded to its own power-of-two height.
    ///
    /// Built one chunk at a time rather than returned as a `Vec` because the
    /// aggregation program's group is ~1.39M rows x 20 columns; the artifact
    /// builder and the trace builder each commit or copy a chunk and drop it, so
    /// peak residency stays one chunk instead of a second whole group.
    ///
    /// # Panics
    ///
    /// On a chunk index at or past [`Self::blake3_chunk_count`] — a caller bug,
    /// and this runs at program-build time, never on a verify path.
    pub fn blake3_chunk_group(&self, chunk: usize) -> ColumnGroup {
        let count = self.blake3_chunk_count();
        assert!(
            chunk < count,
            "LFM_BLAKE3 chunk {chunk} requested of a {count}-chunk program"
        );
        let group = &self.groups.blake3;
        let rows = self.blake3_chunking.chunk_range(group.real_rows, chunk);
        let real_rows = rows.len();
        let padded_rows = padded_rows(real_rows);
        let mut data = group.data[rows.start * group.width..rows.end * group.width].to_vec();
        data.resize(padded_rows * group.width, FE::zero());
        ColumnGroup {
            width: group.width,
            real_rows,
            padded_rows,
            data,
        }
    }
}

/// Emission backends. Backend 1 (column groups) is the machine; backend 2 is
/// the future circuit specialization, a stub by design so the option stays an
/// edit instead of a rewrite.
pub trait LfmBackend {
    type Artifacts;
    fn emit(&self, program: &LfmProgram) -> Self::Artifacts;
}

pub struct ColumnGroupBackend;
impl LfmBackend for ColumnGroupBackend {
    type Artifacts = LfmColumnGroups;
    fn emit(&self, program: &LfmProgram) -> LfmColumnGroups {
        program.groups.clone()
    }
}

/// The circuit backend does not exist yet; it panics so nothing can silently
/// depend on it.
pub struct CircuitBackend;
impl LfmBackend for CircuitBackend {
    type Artifacts = ();
    fn emit(&self, _program: &LfmProgram) -> () {
        unimplemented!(
            "LFM circuit backend is a v1+ specialization; only the column-group backend exists"
        )
    }
}

/// Pass 1 for one instruction: each written address's multiplicity, as
/// `take` gives it.
fn backfill(instr: &mut Instr, take_mult: &mut impl FnMut(Addr) -> u64) {
    match instr {
        Instr::Const { out, mult, .. }
        | Instr::BaseAlu { out, mult, .. }
        | Instr::ExtAlu { out, mult, .. }
        | Instr::Hint { out, mult, .. }
        | Instr::HintOn { out, mult, .. }
        | Instr::Pack { out, mult, .. } => {
            *mult = take_mult(*out);
        }
        Instr::Unpack { outs, mults, .. } => {
            for i in 0..4 {
                mults[i] = take_mult(outs[i]);
            }
        }
        Instr::KeccakF(k) => {
            for i in 0..layout::keccak::NUM_WORDS {
                k.mults[i] = take_mult(k.outs[i]);
            }
            if let Some(rev) = &mut k.rev {
                for i in 0..layout::keccak::DIGEST_WORDS {
                    rev.mults[i] = take_mult(rev.outs[i]);
                }
            }
        }
        Instr::Blake3(k) => {
            for i in 0..layout::blake3::OUT_WORDS {
                k.mults[i] = take_mult(k.outs[i]);
            }
            if let Some(rev) = &mut k.rev {
                for i in 0..layout::blake3::DIGEST_WORDS {
                    rev.mults[i] = take_mult(rev.outs[i]);
                }
            }
        }
        Instr::Select {
            out_l,
            out_r,
            mult_l,
            mult_r,
            ..
        } => {
            *mult_l = take_mult(*out_l);
            *mult_r = take_mult(*out_r);
        }
        Instr::Mux { out, mult, .. } | Instr::CapRead { out, mult, .. } => {
            *mult = take_mult(*out);
        }
        Instr::CapTable(t) => {
            t.handle_mult = take_mult(t.handle);
        }
        Instr::BitDec { bits, halves, .. } => {
            for (addr, mult) in bits.iter_mut() {
                *mult = take_mult(*addr);
            }
            if let Some(hs) = halves {
                for (addr, mult) in hs.iter_mut() {
                    *mult = take_mult(*addr);
                }
            }
        }
        Instr::Hash {
            mode, outs, mults, ..
        } => {
            let num_outs = mode.num_output_cells();
            for i in 0..num_outs {
                mults[i] = take_mult(outs[i]);
            }
        }
        Instr::Hash16(h) => {
            for i in 0..4 {
                h.mults[i] = take_mult(h.outs[i]);
            }
        }
        Instr::B3l(k) => {
            for i in 0..layout::b3l::DIGEST_WORDS {
                k.mults[i] = take_mult(k.outs[i]);
            }
            if let Some(rev) = &mut k.rev {
                for i in 0..layout::b3l::DIGEST_WORDS {
                    rev.mults[i] = take_mult(rev.outs[i]);
                }
            }
        }
        Instr::Public { .. } => {}
    }
}

pub fn compile(source: LfmProgramSource) -> LfmProgram {
    let LfmProgramSource {
        mut instrs,
        num_addrs,
        mut read_counts,
        arena_schema,
        public_len,
    } = source;

    // Pass 1: occupancy + multiplicity backfill.
    let mut written = vec![false; num_addrs as usize];
    let take = |addr: Addr, written: &mut Vec<bool>, counts: &mut [u64]| -> u64 {
        let slot = written
            .get_mut(addr.0 as usize)
            .unwrap_or_else(|| panic!("LFM compiler invariant: address {} out of range", addr.0));
        if *slot {
            panic!("LFM compiler invariant: address {} written twice", addr.0);
        }
        *slot = true;
        // Taking (not reading) is what drains the counter, so the emptiness
        // check below still means "every read had a writer".
        core::mem::take(&mut counts[addr.0 as usize])
    };
    {
        let mut take_one = |addr: Addr| take(addr, &mut written, &mut read_counts);
        for instr in &mut instrs {
            backfill(instr, &mut take_one);
        }
    }
    assert!(
        read_counts.iter().all(|&c| c == 0),
        "LFM compiler invariant: read-count map not drained after backfill — reads of never-written addresses: {:?}",
        read_counts
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(a, _)| Addr(a as u64))
            .collect::<Vec<_>>()
    );

    // Both are dead from here on and together outweigh the groups being built.
    // Dropping them explicitly keeps the emitter's peak off the sum of the two
    // materializations — the scope would otherwise hold them to the end.
    drop(read_counts);
    drop(written);

    let groups = emit_column_groups(&instrs, public_len);
    let hash16 = instrs.iter().any(|i| matches!(i, Instr::Hash16(_)));

    LfmProgram {
        hash16,
        instrs,
        num_addrs,
        arena_schema,
        public_len,
        chunking: KeccakChunking::default(),
        blake3_chunking: Blake3Chunking::default(),
        // The hash split: read once at compile time, so every program this
        // process compiles carries the same policy.
        hash_chunking: HashChunking::for_process(groups.hash.real_rows),
        groups,
    }
}

/// One chip's slices joined into its group, padded: each cell copied once, in
/// parallel, into the group's one buffer.
#[cfg(feature = "parallel")]
fn join_slices(slices: Vec<ColumnGroupBuilder>) -> ColumnGroup {
    use rayon::prelude::*;
    let width = slices[0].width;
    let real_rows: usize = slices.iter().map(|b| b.real_rows).sum();
    let padded = padded_rows(real_rows);
    let ends: Vec<usize> = slices
        .iter()
        .scan(0usize, |at, b| {
            *at += b.data.len();
            Some(*at)
        })
        .collect();
    let mut data = Vec::new();
    (0..padded * width)
        .into_par_iter()
        .with_min_len(1 << 14)
        .map(|i| {
            let k = ends.partition_point(|&end| end <= i);
            match slices.get(k) {
                Some(b) => b.data[i + b.data.len() - ends[k]],
                None => FE::zero(),
            }
        })
        .collect_into_vec(&mut data);
    ColumnGroup {
        width,
        real_rows,
        padded_rows: padded,
        data,
    }
}

/// [`compile`] with both passes cut into slices run in parallel: the same
/// program, instruction for instruction and row for row, for every source
/// the admission validator accepts. Its invariant checks are left to the
/// validator: a source with an address written twice, or read and never
/// written, compiles to a program [`super::validator::validate`] refuses,
/// where [`compile`] panics. For a prover that validates the program anyway
/// and keeps no proof of a program that fails (the levels' driver,
/// `levels::PROGRAM_FAST_ENV`, `levels::VALIDATE_LATE_ENV`); a verifier
/// compiles with [`compile`].
#[cfg(feature = "parallel")]
pub fn compile_parallel(source: LfmProgramSource) -> LfmProgram {
    compile_parallel_sliced(source, 1 << 20)
}

/// [`compile_parallel`] at `slice` instructions a slice.
#[cfg(feature = "parallel")]
pub(crate) fn compile_parallel_sliced(source: LfmProgramSource, slice: usize) -> LfmProgram {
    use rayon::prelude::*;
    let LfmProgramSource {
        mut instrs,
        num_addrs,
        read_counts,
        arena_schema,
        public_len,
    } = source;
    let slice = slice.max(1);
    // Pass 1: each written address's multiplicity is its read count.
    instrs.par_chunks_mut(slice).for_each(|chunk| {
        let mut count = |addr: Addr| read_counts.get(addr.0 as usize).copied().unwrap_or(0);
        for instr in chunk {
            backfill(instr, &mut count);
        }
    });
    drop(read_counts);
    // Pass 2: each slice's rows from where its ordinal-bound groups start.
    let counts: Vec<(usize, usize)> = instrs
        .par_chunks(slice)
        .map(|c| {
            c.iter().fold((0, 0), |(k, b), i| match i {
                Instr::KeccakF(_) => (k + 1, b),
                Instr::B3l(_) => (k, b + 1),
                _ => (k, b),
            })
        })
        .collect();
    let mut at = (0usize, 0usize);
    let starts: Vec<(usize, usize)> = counts
        .iter()
        .map(|&(k, b)| {
            let start = at;
            at = (at.0 + k, at.1 + b);
            start
        })
        .collect();
    let mut parts: Vec<[ColumnGroupBuilder; 15]> = instrs
        .par_chunks(slice)
        .zip(starts.par_iter())
        .map(|(c, &(k, b))| emit_rows(c, k, b).into_array())
        .collect();
    if parts.is_empty() {
        parts.push(emit_rows(&[], 0, 0).into_array());
    }
    // Chip by chip, the slices' rows in order, padded: each cell copied once,
    // in parallel, into the group's one buffer.
    let mut by_chip: Vec<Vec<ColumnGroupBuilder>> = (0..15).map(|_| Vec::new()).collect();
    for part in parts {
        for (chip, builder) in part.into_iter().enumerate() {
            by_chip[chip].push(builder);
        }
    }
    let joined: Vec<ColumnGroup> = by_chip.into_iter().map(join_slices).collect();
    let groups = GroupBuilders::groups_from(joined);
    let hash16 = instrs.par_iter().any(|i| matches!(i, Instr::Hash16(_)));
    LfmProgram {
        hash16,
        instrs,
        num_addrs,
        arena_schema,
        public_len,
        chunking: KeccakChunking::default(),
        blake3_chunking: Blake3Chunking::default(),
        hash_chunking: HashChunking::for_process(groups.hash.real_rows),
        groups,
    }
}

fn fe(v: u64) -> FE {
    FE::from(v)
}

/// Every chip's rows of one slice of a program, unfinished: what pass 2 lays
/// out, slice by slice when [`compile_parallel`] cuts the program.
struct GroupBuilders {
    const_: ColumnGroupBuilder,
    balu: ColumnGroupBuilder,
    xalu: ColumnGroupBuilder,
    select: ColumnGroupBuilder,
    bitdec: ColumnGroupBuilder,
    hash: ColumnGroupBuilder,
    keccak: ColumnGroupBuilder,
    blake3: ColumnGroupBuilder,
    lanes: ColumnGroupBuilder,
    hint: ColumnGroupBuilder,
    public: ColumnGroupBuilder,
    b3l: ColumnGroupBuilder,
    mux: ColumnGroupBuilder,
    cap: ColumnGroupBuilder,
    cap_read: ColumnGroupBuilder,
}

impl GroupBuilders {
    /// The chips' builders in [`LfmColumnGroups`]' order.
    #[cfg(feature = "parallel")]
    fn into_array(self) -> [ColumnGroupBuilder; 15] {
        [
            self.const_,
            self.balu,
            self.xalu,
            self.select,
            self.bitdec,
            self.hash,
            self.keccak,
            self.blake3,
            self.lanes,
            self.hint,
            self.public,
            self.b3l,
            self.mux,
            self.cap,
            self.cap_read,
        ]
    }

    /// The finished groups in [`Self::into_array`]'s order.
    #[cfg(feature = "parallel")]
    fn groups_from(groups: Vec<ColumnGroup>) -> LfmColumnGroups {
        let mut groups = groups.into_iter();
        let mut next = || groups.next().expect("fifteen chips");
        LfmColumnGroups {
            const_: next(),
            balu: next(),
            xalu: next(),
            select: next(),
            bitdec: next(),
            hash: next(),
            keccak: next(),
            blake3: next(),
            lanes: next(),
            hint: next(),
            public: next(),
            b3l: next(),
            mux: next(),
            cap: next(),
            cap_read: next(),
        }
    }

    fn finish(self) -> LfmColumnGroups {
        LfmColumnGroups {
            const_: self.const_.finish(),
            balu: self.balu.finish(),
            xalu: self.xalu.finish(),
            select: self.select.finish(),
            bitdec: self.bitdec.finish(),
            hash: self.hash.finish(),
            keccak: self.keccak.finish(),
            blake3: self.blake3.finish(),
            lanes: self.lanes.finish(),
            hint: self.hint.finish(),
            public: self.public.finish(),
            b3l: self.b3l.finish(),
            mux: self.mux.finish(),
            cap: self.cap.finish(),
            cap_read: self.cap_read.finish(),
        }
    }
}

/// Pass 2: partition instructions per chip (program order preserved) and lay
/// out each chip's instruction fields per [`super::layout`].
fn emit_column_groups(instrs: &[Instr], _public_len: u32) -> LfmColumnGroups {
    emit_rows(instrs, 0, 0).finish()
}

/// Pass 2 over a slice of a program whose `LFM_KECCAK` rows start at the
/// group's row `keccak_first` and whose lean head rows at `b3l_first` (the
/// two groups whose rows bind their ordinal).
fn emit_rows(instrs: &[Instr], keccak_first: usize, b3l_first: usize) -> GroupBuilders {
    let mut const_ = ColumnGroupBuilder::new(layout::const_::PREP_WIDTH);
    let mut balu = ColumnGroupBuilder::new(layout::balu::PREP_WIDTH);
    let mut xalu = ColumnGroupBuilder::new(layout::xalu::PREP_WIDTH);
    let mut select = ColumnGroupBuilder::new(layout::select::PREP_WIDTH);
    let mut bitdec = ColumnGroupBuilder::new(layout::bitdec::PREP_WIDTH);
    let mut hash = ColumnGroupBuilder::new(layout::hash::PREP_WIDTH);
    let mut keccak = ColumnGroupBuilder::starting_at(layout::keccak::PREP_WIDTH, keccak_first);
    let mut blake3 = ColumnGroupBuilder::new(layout::blake3::PREP_WIDTH);
    let mut lanes = ColumnGroupBuilder::new(layout::lanes::PREP_WIDTH);
    let mut hint = ColumnGroupBuilder::new(layout::hint::PREP_WIDTH);
    let mut public = ColumnGroupBuilder::new(layout::public::PREP_WIDTH);
    let mut b3l = ColumnGroupBuilder::starting_at(layout::b3l::PREP_WIDTH, b3l_first);
    let mut mux = ColumnGroupBuilder::new(layout::mux::PREP_WIDTH);
    let mut cap = ColumnGroupBuilder::new(layout::cap::PREP_WIDTH);
    let mut cap_read = ColumnGroupBuilder::new(layout::cap_read::PREP_WIDTH);

    for instr in instrs {
        match instr {
            Instr::Const { out, value, mult } => {
                use layout::const_ as c;
                let r = const_.open_row();
                const_.set(r, c::ADDR, fe(out.0));
                for (i, v) in value.iter().enumerate() {
                    const_.set(r, c::V0 + i, *v);
                }
                const_.set(r, c::MULT, fe(*mult));
            }
            Instr::BaseAlu {
                op,
                out,
                a,
                b,
                c,
                mult,
            } => {
                use layout::balu as l;
                let r = balu.open_row();
                balu.set(r, l::A_ADDR, fe(a.0));
                balu.set(r, l::B_ADDR, fe(b.0));
                balu.set(r, l::C_ADDR, fe(c.0));
                balu.set(r, l::OUT_ADDR, fe(out.0));
                let sel = match op {
                    BaseOp::Add => l::SEL_ADD,
                    BaseOp::Sub => l::SEL_SUB,
                    BaseOp::Mul => l::SEL_MUL,
                    BaseOp::Div => l::SEL_DIV,
                    BaseOp::MulAdd => l::SEL_MULADD,
                };
                balu.set(r, sel, FE::one());
                balu.set(r, l::MULT, fe(*mult));
            }
            Instr::ExtAlu {
                op,
                out,
                a,
                b,
                c,
                mult,
            } => {
                use layout::xalu as l;
                let r = xalu.open_row();
                xalu.set(r, l::A_ADDR, fe(a.0));
                xalu.set(r, l::B_ADDR, fe(b.0));
                xalu.set(r, l::C_ADDR, fe(c.0));
                xalu.set(r, l::OUT_ADDR, fe(out.0));
                let sel = match op {
                    ExtOp::Add => l::SEL_ADD,
                    ExtOp::Sub => l::SEL_SUB,
                    ExtOp::Mul => l::SEL_MUL,
                    ExtOp::Div => l::SEL_DIV,
                    ExtOp::MulAdd => l::SEL_MULADD,
                    ExtOp::MulBase => l::SEL_MULBASE,
                };
                xalu.set(r, sel, FE::one());
                xalu.set(r, l::MULT, fe(*mult));
            }
            Instr::Select {
                bit,
                out_l,
                out_r,
                in_l,
                in_r,
                mult_l,
                mult_r,
            } => {
                use layout::select as l;
                let r = select.open_row();
                select.set(r, l::BIT_ADDR, fe(bit.0));
                select.set(r, l::INL_ADDR, fe(in_l.0));
                select.set(r, l::INR_ADDR, fe(in_r.0));
                select.set(r, l::OUTL_ADDR, fe(out_l.0));
                select.set(r, l::OUTR_ADDR, fe(out_r.0));
                select.set(r, l::MULT_L, fe(*mult_l));
                select.set(r, l::MULT_R, fe(*mult_r));
                select.set(r, l::IS_REAL, FE::one());
            }
            Instr::BitDec {
                input,
                bits,
                halves,
            } => {
                use layout::bitdec as l;
                let r = bitdec.open_row();
                bitdec.set(r, l::IN_ADDR, fe(input.0));
                bitdec.set(r, l::IS_REAL, FE::one());
                for (i, (addr, mult)) in bits.iter().enumerate() {
                    bitdec.set(r, l::bit_addr(i), fe(addr.0));
                    bitdec.set(r, l::bit_mult(i), fe(*mult));
                }
                if let Some([h0, h1]) = halves {
                    bitdec.set(r, l::HALF0_ADDR, fe(h0.0.0));
                    bitdec.set(r, l::HALF0_MULT, fe(h0.1));
                    bitdec.set(r, l::HALF1_ADDR, fe(h1.0.0));
                    bitdec.set(r, l::HALF1_MULT, fe(h1.1));
                }
            }
            Instr::Hash {
                mode,
                ins,
                outs,
                mults,
            } => {
                // One-hot over the three modes. The AIR pins only the SUM to a
                // bit; exactly-one-of is this emitter's job, re-checked by the
                // admission validator.
                use layout::hash as l;
                let r = hash.open_row();
                hash.set(r, l::IN_ADDR0, fe(ins[0].0));
                hash.set(r, l::IN_ADDR1, fe(ins[1].0));
                hash.set(r, l::IN_ADDR2, fe(ins[2].0));
                hash.set(r, l::OUT_ADDR0, fe(outs[0].0));
                hash.set(r, l::OUT_ADDR1, fe(outs[1].0));
                hash.set(r, l::OUT_ADDR2, fe(outs[2].0));
                let mode_col = match mode {
                    HashMode::Compress => l::MODE_C,
                    HashMode::Transcript => l::MODE_T,
                    HashMode::Leaf => l::MODE_L,
                    HashMode::Permute => l::MODE_P,
                };
                hash.set(r, mode_col, FE::one());
                hash.set(r, l::MULT0, fe(mults[0]));
                hash.set(r, l::MULT1, fe(mults[1]));
                hash.set(r, l::MULT2, fe(mults[2]));
            }
            Instr::Hash16(h) => {
                // The width-16 socket's instruction group: the same 13 columns
                // as the twelve-felt socket's, in its own order
                // (`p1w16_socket::cols`). A program carries one or the other,
                // so the group is never mixed (the admission validator).
                use super::p1w16_socket::cols as l;
                let r = hash.open_row();
                for k in 0..4 {
                    hash.set(r, l::IN_ADDR0 + k, fe(h.ins[k].0));
                    hash.set(r, l::OUT_ADDR0 + k, fe(h.outs[k].0));
                    hash.set(r, l::MULT0 + k, fe(h.mults[k]));
                }
                hash.set(r, l::IS_REAL, FE::one());
            }
            Instr::KeccakF(op) => {
                use layout::keccak as k;
                // The tag is the row ordinal, so uniqueness is structural and
                // the prover has no say (it is preprocessed data). See
                // `layout::keccak::tag_for_row`.
                let tag = k::tag_for_row(keccak.next_row());
                let r = keccak.open_row();
                keccak.set(r, k::TAG_LO, fe(tag & 0xFFFF_FFFF));
                keccak.set(r, k::TAG_HI, fe(tag >> 32));
                for j in 0..k::NUM_WORDS {
                    keccak.set(r, k::in_addr(j), fe(op.ins[j].0));
                    keccak.set(r, k::out_addr(j), fe(op.outs[j].0));
                    keccak.set(r, k::mult(j), fe(op.mults[j]));
                }
                if let Some(rev) = &op.rev {
                    for w in 0..k::DIGEST_WORDS {
                        keccak.set(r, k::rev_addr(w), fe(rev.outs[w].0));
                        keccak.set(r, k::rev_mult(w), fe(rev.mults[w]));
                    }
                }
                match op.mode {
                    KeccakMode::Permute => keccak.set(r, k::MODE_PERM, FE::one()),
                    KeccakMode::Absorb => {
                        keccak.set(r, k::MODE_ABSORB, FE::one());
                        for j in 0..k::BLOCK_WORDS {
                            keccak.set(r, k::block_addr(j), fe(op.block[j].0));
                        }
                    }
                }
            }
            Instr::Blake3(op) => {
                use layout::blake3 as l;
                let r = blake3.open_row();
                for j in 0..l::IN_WORDS {
                    blake3.set(r, l::in_addr(j), fe(op.ins[j].0));
                }
                for j in 0..l::OUT_WORDS {
                    blake3.set(r, l::out_addr(j), fe(op.outs[j].0));
                    blake3.set(r, l::mult(j), fe(op.mults[j]));
                }
                if let Some(rev) = &op.rev {
                    for w in 0..l::DIGEST_WORDS {
                        blake3.set(r, l::rev_addr(w), fe(rev.outs[w].0));
                        blake3.set(r, l::rev_mult(w), fe(rev.mults[w]));
                    }
                }
                blake3.set(r, l::MU, FE::one());
            }
            Instr::Mux {
                bit,
                out,
                in_l,
                in_r,
                mult,
            } => {
                use layout::mux as l;
                let r = mux.open_row();
                mux.set(r, l::BIT_ADDR, fe(bit.0));
                mux.set(r, l::INL_ADDR, fe(in_l.0));
                mux.set(r, l::INR_ADDR, fe(in_r.0));
                mux.set(r, l::OUT_ADDR, fe(out.0));
                mux.set(r, l::MULT, fe(*mult));
                mux.set(r, l::IS_REAL, FE::one());
            }
            Instr::CapTable(t) => {
                use layout::cap as l;
                for (i, node) in t.nodes.iter().enumerate() {
                    let r = cap.open_row();
                    cap.set(r, l::TAG, fe(u64::from(t.tag)));
                    cap.set(r, l::INDEX, fe(i as u64));
                    cap.set(r, l::NODE_ADDR, fe(node.0));
                    if i == 0 {
                        cap.set(r, l::H_ADDR, fe(t.handle.0));
                        cap.set(r, l::H_MULT, fe(t.handle_mult));
                    }
                    cap.set(r, l::IS_REAL, FE::one());
                }
            }
            Instr::CapRead {
                handle,
                index,
                out,
                mult,
            } => {
                use layout::cap_read as l;
                let r = cap_read.open_row();
                cap_read.set(r, l::HANDLE_ADDR, fe(handle.0));
                cap_read.set(r, l::INDEX_ADDR, fe(index.0));
                cap_read.set(r, l::OUT_ADDR, fe(out.0));
                cap_read.set(r, l::MULT, fe(*mult));
                cap_read.set(r, l::IS_REAL, FE::one());
            }
            Instr::B3l(op) => {
                use super::instr::{B3lBlock, B3lCv};
                use layout::b3l as l;
                // The tag is the row ordinal + 1, so uniqueness is structural
                // and the prover has no say (it is preprocessed data).
                let tag = l::tag_for_row(b3l.next_row());
                let r = b3l.open_row();
                b3l.set(r, l::MU, FE::one());
                b3l.set(r, l::TAG, fe(tag));
                match &op.cv {
                    B3lCv::Iv => b3l.set(r, l::IVF, FE::one()),
                    B3lCv::Words(w) => {
                        for (j, a) in w.iter().enumerate() {
                            b3l.set(r, l::haddr(j), fe(a.0));
                        }
                    }
                }
                match &op.block {
                    B3lBlock::Lanes(w) => {
                        b3l.set(r, l::MLANES, FE::one());
                        for (j, a) in w.iter().enumerate() {
                            b3l.set(r, l::maddr(j), fe(a.0));
                        }
                    }
                    B3lBlock::Felts(w) => {
                        b3l.set(r, l::MFELTS, FE::one());
                        for (j, a) in w.iter().enumerate() {
                            b3l.set(r, l::maddr(j), fe(a.0));
                        }
                    }
                }
                b3l.set(r, l::BLEN, fe(u64::from(op.block_len)));
                b3l.set(r, l::FLAGS, fe(u64::from(op.flags)));
                for j in 0..l::DIGEST_WORDS {
                    b3l.set(r, l::oaddr(j), fe(op.outs[j].0));
                    b3l.set(r, l::ocnt(j), fe(op.mults[j]));
                }
                if let Some(rev) = &op.rev {
                    for j in 0..l::DIGEST_WORDS {
                        b3l.set(r, l::raddr(j), fe(rev.outs[j].0));
                        b3l.set(r, l::rcnt(j), fe(rev.mults[j]));
                    }
                }
            }
            // A `HintOn` is a hint row: which arena word it read is the
            // executor's, not the row's.
            Instr::Hint { out, mult, .. } | Instr::HintOn { out, mult, .. } => {
                use layout::hint as l;
                let r = hint.open_row();
                hint.set(r, l::OUT_ADDR, fe(out.0));
                hint.set(r, l::MULT, fe(*mult));
            }
            Instr::Pack {
                lanes: ls,
                out,
                mult,
            } => {
                use layout::lanes as l;
                let r = lanes.open_row();
                lanes.set(r, l::WORD_ADDR, fe(out.0));
                for (i, lane) in ls.iter().enumerate() {
                    lanes.set(r, l::LANE_ADDR0 + i, fe(lane.0));
                }
                lanes.set(r, l::MODE_PACK, FE::one());
                lanes.set(r, l::WORD_MULT, fe(*mult));
            }
            Instr::Unpack { input, outs, mults } => {
                use layout::lanes as l;
                let r = lanes.open_row();
                lanes.set(r, l::WORD_ADDR, fe(input.0));
                for i in 0..4 {
                    lanes.set(r, l::LANE_ADDR0 + i, fe(outs[i].0));
                    lanes.set(r, l::LANE_MULT0 + i, fe(mults[i]));
                }
                lanes.set(r, l::MODE_UNPACK, FE::one());
            }
            Instr::Public { addr, index } => {
                use layout::public as l;
                let r = public.open_row();
                public.set(r, l::IN_ADDR, fe(addr.0));
                public.set(r, l::INDEX, fe(*index as u64));
                public.set(r, l::IS_REAL, FE::one());
            }
        }
    }

    GroupBuilders {
        const_,
        balu,
        xalu,
        select,
        bitdec,
        hash,
        keccak,
        blake3,
        lanes,
        hint,
        public,
        b3l,
        mux,
        cap,
        cap_read,
    }
}

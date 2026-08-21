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
use super::chunking::KeccakChunking;
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
    real_rows: usize,
    data: Vec<FE>,
}

impl ColumnGroupBuilder {
    fn new(width: usize) -> Self {
        ColumnGroupBuilder {
            width,
            real_rows: 0,
            data: Vec::new(),
        }
    }

    /// The ordinal the next row will take. `LFM_KECCAK` binds it into the row
    /// as a structural tag, so it has to be read before [`Self::open_row`].
    fn next_row(&self) -> usize {
        self.real_rows
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
}

impl LfmProgram {
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
    for instr in &mut instrs {
        match instr {
            Instr::Const { out, mult, .. }
            | Instr::BaseAlu { out, mult, .. }
            | Instr::ExtAlu { out, mult, .. }
            | Instr::Hint { out, mult, .. }
            | Instr::Pack { out, mult, .. } => {
                *mult = take(*out, &mut written, &mut read_counts);
            }
            Instr::Unpack { outs, mults, .. } => {
                for i in 0..4 {
                    mults[i] = take(outs[i], &mut written, &mut read_counts);
                }
            }
            Instr::KeccakF(k) => {
                for i in 0..layout::keccak::NUM_WORDS {
                    k.mults[i] = take(k.outs[i], &mut written, &mut read_counts);
                }
                if let Some(rev) = &mut k.rev {
                    for i in 0..layout::keccak::DIGEST_WORDS {
                        rev.mults[i] = take(rev.outs[i], &mut written, &mut read_counts);
                    }
                }
            }
            Instr::Blake3(k) => {
                for i in 0..layout::blake3::OUT_WORDS {
                    k.mults[i] = take(k.outs[i], &mut written, &mut read_counts);
                }
                if let Some(rev) = &mut k.rev {
                    for i in 0..layout::blake3::DIGEST_WORDS {
                        rev.mults[i] = take(rev.outs[i], &mut written, &mut read_counts);
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
                *mult_l = take(*out_l, &mut written, &mut read_counts);
                *mult_r = take(*out_r, &mut written, &mut read_counts);
            }
            Instr::BitDec { bits, halves, .. } => {
                for (addr, mult) in bits.iter_mut() {
                    *mult = take(*addr, &mut written, &mut read_counts);
                }
                if let Some(hs) = halves {
                    for (addr, mult) in hs.iter_mut() {
                        *mult = take(*addr, &mut written, &mut read_counts);
                    }
                }
            }
            Instr::Hash {
                mode, outs, mults, ..
            } => {
                let num_outs = mode.num_output_cells();
                for i in 0..num_outs {
                    mults[i] = take(outs[i], &mut written, &mut read_counts);
                }
            }
            Instr::Public { .. } => {}
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

    LfmProgram {
        instrs,
        num_addrs,
        arena_schema,
        public_len,
        groups,
        chunking: KeccakChunking::default(),
    }
}

fn fe(v: u64) -> FE {
    FE::from(v)
}

/// Pass 2: partition instructions per chip (program order preserved) and lay
/// out each chip's instruction fields per [`super::layout`].
fn emit_column_groups(instrs: &[Instr], _public_len: u32) -> LfmColumnGroups {
    let mut const_ = ColumnGroupBuilder::new(layout::const_::PREP_WIDTH);
    let mut balu = ColumnGroupBuilder::new(layout::balu::PREP_WIDTH);
    let mut xalu = ColumnGroupBuilder::new(layout::xalu::PREP_WIDTH);
    let mut select = ColumnGroupBuilder::new(layout::select::PREP_WIDTH);
    let mut bitdec = ColumnGroupBuilder::new(layout::bitdec::PREP_WIDTH);
    let mut hash = ColumnGroupBuilder::new(layout::hash::PREP_WIDTH);
    let mut keccak = ColumnGroupBuilder::new(layout::keccak::PREP_WIDTH);
    let mut blake3 = ColumnGroupBuilder::new(layout::blake3::PREP_WIDTH);
    let mut lanes = ColumnGroupBuilder::new(layout::lanes::PREP_WIDTH);
    let mut hint = ColumnGroupBuilder::new(layout::hint::PREP_WIDTH);
    let mut public = ColumnGroupBuilder::new(layout::public::PREP_WIDTH);

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
            Instr::Hint { out, mult, .. } => {
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

    LfmColumnGroups {
        const_: const_.finish(),
        balu: balu.finish(),
        xalu: xalu.finish(),
        select: select.finish(),
        bitdec: bitdec.finish(),
        hash: hash.finish(),
        keccak: keccak.finish(),
        blake3: blake3.finish(),
        lanes: lanes.finish(),
        hint: hint.finish(),
        public: public.finish(),
    }
}

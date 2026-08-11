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

use std::collections::HashMap;

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

impl ColumnGroup {
    fn from_rows(width: usize, rows: Vec<Vec<FE>>) -> Self {
        let real_rows = rows.len();
        let padded = padded_rows(real_rows);
        let mut data = vec![FE::zero(); padded * width];
        for (r, row) in rows.into_iter().enumerate() {
            debug_assert_eq!(row.len(), width);
            data[r * width..(r + 1) * width].clone_from_slice(&row);
        }
        ColumnGroup {
            width,
            real_rows,
            padded_rows: padded,
            data,
        }
    }

    pub fn at(&self, row: usize, col: usize) -> &FE {
        &self.data[row * self.width + col]
    }

    pub fn set(&mut self, row: usize, col: usize, v: FE) {
        self.data[row * self.width + col] = v;
    }
}

/// The eight program-dependent instruction column groups, in the frozen chip
/// order. (`LFM_RANGE`'s group is program-independent and materialized at
/// commitment time.)
#[derive(Debug, Clone)]
pub struct LfmColumnGroups {
    pub const_: ColumnGroup,
    pub balu: ColumnGroup,
    pub xalu: ColumnGroup,
    pub select: ColumnGroup,
    pub bitdec: ColumnGroup,
    pub hash: ColumnGroup,
    pub keccak: ColumnGroup,
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
    let take = |addr: Addr, written: &mut Vec<bool>, counts: &mut HashMap<Addr, u64>| -> u64 {
        let slot = written
            .get_mut(addr.0 as usize)
            .unwrap_or_else(|| panic!("LFM compiler invariant: address {} out of range", addr.0));
        if *slot {
            panic!("LFM compiler invariant: address {} written twice", addr.0);
        }
        *slot = true;
        counts.remove(&addr).unwrap_or(0)
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
            Instr::BitDec { bits, .. } => {
                for (addr, mult) in bits.iter_mut() {
                    *mult = take(*addr, &mut written, &mut read_counts);
                }
            }
            Instr::Hash {
                mode, outs, mults, ..
            } => {
                let num_outs = if mode.is_two_to_one() { 1 } else { 3 };
                for i in 0..num_outs {
                    mults[i] = take(outs[i], &mut written, &mut read_counts);
                }
            }
            Instr::Public { .. } => {}
        }
    }
    assert!(
        read_counts.is_empty(),
        "LFM compiler invariant: read-count map not drained after backfill — reads of never-written addresses: {:?}",
        read_counts.keys().collect::<Vec<_>>()
    );

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
    let mut const_rows = Vec::new();
    let mut balu_rows = Vec::new();
    let mut xalu_rows = Vec::new();
    let mut select_rows = Vec::new();
    let mut bitdec_rows = Vec::new();
    let mut hash_rows = Vec::new();
    let mut keccak_rows: Vec<Vec<FE>> = Vec::new();
    let mut lanes_rows = Vec::new();
    let mut hint_rows = Vec::new();
    let mut public_rows = Vec::new();

    for instr in instrs {
        match instr {
            Instr::Const { out, value, mult } => {
                let mut row = vec![fe(out.0)];
                row.extend(value.iter().cloned());
                row.push(fe(*mult));
                const_rows.push(row);
            }
            Instr::BaseAlu {
                op,
                out,
                a,
                b,
                c,
                mult,
            } => {
                let mut row = vec![fe(a.0), fe(b.0), fe(c.0), fe(out.0)];
                let mut sels = [FE::zero(), FE::zero(), FE::zero(), FE::zero(), FE::zero()];
                let idx = match op {
                    BaseOp::Add => 0,
                    BaseOp::Sub => 1,
                    BaseOp::Mul => 2,
                    BaseOp::Div => 3,
                    BaseOp::MulAdd => 4,
                };
                sels[idx] = FE::one();
                row.extend(sels);
                row.push(fe(*mult));
                balu_rows.push(row);
            }
            Instr::ExtAlu {
                op,
                out,
                a,
                b,
                c,
                mult,
            } => {
                let mut row = vec![fe(a.0), fe(b.0), fe(c.0), fe(out.0)];
                let mut sels = vec![FE::zero(); layout::xalu::NUM_SELECTORS];
                let idx = match op {
                    ExtOp::Add => 0,
                    ExtOp::Sub => 1,
                    ExtOp::Mul => 2,
                    ExtOp::Div => 3,
                    ExtOp::MulAdd => 4,
                    ExtOp::MulBase => 5,
                };
                sels[idx] = FE::one();
                row.extend(sels);
                row.push(fe(*mult));
                xalu_rows.push(row);
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
                select_rows.push(vec![
                    fe(bit.0),
                    fe(in_l.0),
                    fe(in_r.0),
                    fe(out_l.0),
                    fe(out_r.0),
                    fe(*mult_l),
                    fe(*mult_r),
                    FE::one(),
                ]);
            }
            Instr::BitDec { input, bits } => {
                let mut row = vec![FE::zero(); layout::bitdec::PREP_WIDTH];
                row[layout::bitdec::IN_ADDR] = fe(input.0);
                row[layout::bitdec::IS_REAL] = FE::one();
                for (i, (addr, mult)) in bits.iter().enumerate() {
                    row[layout::bitdec::bit_addr(i)] = fe(addr.0);
                    row[layout::bitdec::bit_mult(i)] = fe(*mult);
                }
                bitdec_rows.push(row);
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
                let mut row = vec![FE::zero(); layout::hash::PREP_WIDTH];
                row[layout::hash::IN_ADDR0] = fe(ins[0].0);
                row[layout::hash::IN_ADDR1] = fe(ins[1].0);
                row[layout::hash::IN_ADDR2] = fe(ins[2].0);
                row[layout::hash::OUT_ADDR0] = fe(outs[0].0);
                row[layout::hash::OUT_ADDR1] = fe(outs[1].0);
                row[layout::hash::OUT_ADDR2] = fe(outs[2].0);
                row[match mode {
                    HashMode::Compress => layout::hash::MODE_C,
                    HashMode::Transcript => layout::hash::MODE_T,
                    HashMode::Permute => layout::hash::MODE_P,
                }] = FE::one();
                row[layout::hash::MULT0] = fe(mults[0]);
                row[layout::hash::MULT1] = fe(mults[1]);
                row[layout::hash::MULT2] = fe(mults[2]);
                hash_rows.push(row);
            }
            Instr::KeccakF(op) => {
                use layout::keccak as k;
                let mut row = vec![FE::zero(); k::PREP_WIDTH];
                // The tag is the row ordinal, so uniqueness is structural and
                // the prover has no say (it is preprocessed data). See
                // `layout::keccak::tag_for_row`.
                let tag = k::tag_for_row(keccak_rows.len());
                row[k::TAG_LO] = fe(tag & 0xFFFF_FFFF);
                row[k::TAG_HI] = fe(tag >> 32);
                for j in 0..k::NUM_WORDS {
                    row[k::in_addr(j)] = fe(op.ins[j].0);
                    row[k::out_addr(j)] = fe(op.outs[j].0);
                    row[k::mult(j)] = fe(op.mults[j]);
                }
                if let Some(rev) = &op.rev {
                    for w in 0..k::DIGEST_WORDS {
                        row[k::rev_addr(w)] = fe(rev.outs[w].0);
                        row[k::rev_mult(w)] = fe(rev.mults[w]);
                    }
                }
                match op.mode {
                    KeccakMode::Permute => row[k::MODE_PERM] = FE::one(),
                    KeccakMode::Absorb => {
                        row[k::MODE_ABSORB] = FE::one();
                        for j in 0..k::BLOCK_WORDS {
                            row[k::block_addr(j)] = fe(op.block[j].0);
                        }
                    }
                }
                keccak_rows.push(row);
            }
            Instr::Hint { out, mult, .. } => {
                hint_rows.push(vec![fe(out.0), fe(*mult)]);
            }
            Instr::Pack { lanes, out, mult } => {
                let mut row = vec![FE::zero(); layout::lanes::PREP_WIDTH];
                row[layout::lanes::WORD_ADDR] = fe(out.0);
                for (i, lane) in lanes.iter().enumerate() {
                    row[layout::lanes::LANE_ADDR0 + i] = fe(lane.0);
                }
                row[layout::lanes::MODE_PACK] = FE::one();
                row[layout::lanes::WORD_MULT] = fe(*mult);
                lanes_rows.push(row);
            }
            Instr::Unpack { input, outs, mults } => {
                let mut row = vec![FE::zero(); layout::lanes::PREP_WIDTH];
                row[layout::lanes::WORD_ADDR] = fe(input.0);
                for i in 0..4 {
                    row[layout::lanes::LANE_ADDR0 + i] = fe(outs[i].0);
                    row[layout::lanes::LANE_MULT0 + i] = fe(mults[i]);
                }
                row[layout::lanes::MODE_UNPACK] = FE::one();
                lanes_rows.push(row);
            }
            Instr::Public { addr, index } => {
                public_rows.push(vec![fe(addr.0), fe(*index as u64), FE::one()]);
            }
        }
    }

    LfmColumnGroups {
        const_: ColumnGroup::from_rows(layout::const_::PREP_WIDTH, const_rows),
        balu: ColumnGroup::from_rows(layout::balu::PREP_WIDTH, balu_rows),
        xalu: ColumnGroup::from_rows(layout::xalu::PREP_WIDTH, xalu_rows),
        select: ColumnGroup::from_rows(layout::select::PREP_WIDTH, select_rows),
        bitdec: ColumnGroup::from_rows(layout::bitdec::PREP_WIDTH, bitdec_rows),
        hash: ColumnGroup::from_rows(layout::hash::PREP_WIDTH, hash_rows),
        keccak: ColumnGroup::from_rows(layout::keccak::PREP_WIDTH, keccak_rows),
        lanes: ColumnGroup::from_rows(layout::lanes::PREP_WIDTH, lanes_rows),
        hint: ColumnGroup::from_rows(layout::hint::PREP_WIDTH, hint_rows),
        public: ColumnGroup::from_rows(layout::public::PREP_WIDTH, public_rows),
    }
}

//! The LFM executor / witness generator.
//!
//! One `for` over the straight-line program, against write-once memory and
//! the host-supplied arenas. Produces per-chip **value-only** records —
//! addresses, selectors and multiplicities come from the program (they are
//! preprocessed data), so records carry values only, and the executor ignores
//! `mult` entirely: execution semantics never depend on it.
//!
//! Defense in depth the reference machine omits: double-writes and
//! read-before-write are checked at runtime here, independently of both the
//! compiler's tripwire panics and the admission validator.

use math::field::traits::IsPrimeField;

use crate::tables::types::{FE, FEE, GoldilocksField};

use super::blake3_chip::Blake3Values;
use super::compiler::LfmProgram;
use super::hash::{HASH_STATE_FELTS, LfmHasher};
use super::instr::{Addr, BaseOp, ExtOp, HashMode, Instr, KeccakMode};
use super::word::{LfmWord, base_word, ext_word};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LfmExecError {
    DoubleWrite(u64),
    ReadBeforeWrite(u64),
    /// `x / 0` with `x ≠ 0` — this is also how a failed assertion surfaces.
    DivByZero {
        addr: u64,
    },
    NonBooleanBit(u64),
    /// A base-typed read found nonzero lanes 1–3 (the bus token would not
    /// match any base write, so the AIR-side program would be unprovable).
    NotBaseWord(u64),
    /// An ext-typed read found a nonzero lane 3.
    NotExtWord(u64),
    /// A `KeccakF` or `Blake3` input word lane held a value at or above `2^32`,
    /// so it is not a `u32` half of a keccak lane (respectively a BLAKE3 input
    /// word). Both chips recompose each `u32` from four BITWISE-constrained byte
    /// columns, so no such value exists on the AIR side — the program would be
    /// unprovable.
    NotU32Half {
        addr: u64,
        lane: usize,
    },
    /// A `KeccakF` input word's unused top lane (the state is 50 halves in 52
    /// slots) was nonzero; the bus pins those slots to zero as tuple constants.
    KeccakSpareLaneNonZero {
        addr: u64,
        lane: usize,
    },
    ArenaCountMismatch {
        expected: usize,
        found: usize,
    },
    ArenaLenMismatch {
        arena: u32,
        expected: u32,
        found: usize,
    },
    ArenaOutOfBounds {
        arena: u32,
        index: u32,
    },
    /// An `Instr::Hash` outside the selected hasher's domain, with the reason
    /// the hasher gave (`LfmHasher::admits`). BLAKE3 raises both of its: a
    /// `Permute` row, for which it has no socket, and a `Compress` input lane
    /// at or above `2^32`, which its chip cannot decompose into bytes. In both
    /// cases the program is unprovable under that hasher, so failing here — with
    /// a reason — beats failing later inside the prover.
    HasherRejected(&'static str),
    Internal(&'static str),
}

// ---- per-chip value records (values only; the program carries the rest) ----

#[derive(Debug, Clone)]
pub struct BaluRow {
    pub a: FE,
    pub b: FE,
    pub c: FE,
    pub out: FE,
}

#[derive(Debug, Clone)]
pub struct XaluRow {
    pub a: [FE; 3],
    pub b: [FE; 3],
    pub c: [FE; 3],
    pub out: [FE; 3],
}

#[derive(Debug, Clone)]
pub struct SelectRow {
    pub bit: FE,
    pub in_l: LfmWord,
    pub in_r: LfmWord,
    pub out_l: LfmWord,
    pub out_r: LfmWord,
}

#[derive(Debug, Clone)]
pub struct BitDecRow {
    /// All 64 bit values, low-to-high (constrained witness columns).
    pub bits: [FE; 64],
    /// The canonicity gadget's witnesses: `z` = "top 32 bits all ones",
    /// `ginv` = inverse of `(2^32 − 1) − top` when that is nonzero.
    pub z: FE,
    pub ginv: FE,
}

#[derive(Debug, Clone)]
pub struct HashRow {
    /// The 12 input columns: full state for `Permute`; `[a ‖ b ‖ 0⁴]` for the
    /// two-to-one modes (lanes 8–11 are unconstrained on those rows — the AIR
    /// injects the IV there via the mode selector).
    pub ins: [FE; HASH_STATE_FELTS],
    /// The full permuted state.
    pub outs: [FE; HASH_STATE_FELTS],
}

/// One `LFM_KECCAK` row. The 400 byte columns are derived from these two
/// states; the tag is the row ordinal (`layout::keccak::tag_for_row`), so it is
/// not recorded here — it is program data, not witness.
#[derive(Debug, Clone)]
pub struct KeccakRow {
    pub mode: KeccakMode,
    /// The state as received from memory.
    pub state: [u64; 25],
    /// The 136-byte rate block as received (all zero on `Permute` rows, where
    /// the block columns are dead — nothing reads them).
    pub block: [u8; 136],
    /// What actually enters the permutation: `state` with `block` XORed into
    /// its rate region on absorb rows, `state` unchanged on permute rows.
    pub perm_in: [u64; 25],
    pub output: [u64; 25],
}

#[derive(Debug, Default)]
pub struct LfmRecords {
    pub num_consts: usize,
    pub balu: Vec<BaluRow>,
    pub xalu: Vec<XaluRow>,
    pub select: Vec<SelectRow>,
    pub bitdec: Vec<BitDecRow>,
    pub hash: Vec<HashRow>,
    pub keccak: Vec<KeccakRow>,
    /// One `LFM_BLAKE3` row. Values only, like every other record: the chip's
    /// addresses and multiplicities are preprocessed program data.
    pub blake3: Vec<Blake3Values>,
    /// One word per Pack/Unpack row (the shared value columns).
    pub lanes: Vec<LfmWord>,
    pub hint: Vec<LfmWord>,
    pub public: Vec<LfmWord>,
}

#[derive(Debug)]
pub struct LfmExecution {
    pub records: LfmRecords,
    /// The public output, in emission order: `(index, word)`.
    pub public_words: Vec<(u32, LfmWord)>,
    /// Final memory, exposed for tests and debugging.
    pub memory: Vec<Option<LfmWord>>,
}

struct Machine<'a> {
    memory: Vec<Option<LfmWord>>,
    arenas: &'a [Vec<LfmWord>],
}

impl Machine<'_> {
    fn write(&mut self, addr: Addr, w: LfmWord) -> Result<(), LfmExecError> {
        let slot = self
            .memory
            .get_mut(addr.0 as usize)
            .ok_or(LfmExecError::Internal("address out of range"))?;
        if slot.is_some() {
            return Err(LfmExecError::DoubleWrite(addr.0));
        }
        *slot = Some(w);
        Ok(())
    }

    fn read_word(&self, addr: Addr) -> Result<LfmWord, LfmExecError> {
        self.memory
            .get(addr.0 as usize)
            .cloned()
            .flatten()
            .ok_or(LfmExecError::ReadBeforeWrite(addr.0))
    }

    fn read_base(&self, addr: Addr) -> Result<FE, LfmExecError> {
        let w = self.read_word(addr)?;
        super::word::word_as_base(&w).ok_or(LfmExecError::NotBaseWord(addr.0))
    }

    fn read_ext(&self, addr: Addr) -> Result<FEE, LfmExecError> {
        let w = self.read_word(addr)?;
        super::word::word_as_ext(&w).ok_or(LfmExecError::NotExtWord(addr.0))
    }
}

pub fn execute(
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
    hasher: &impl LfmHasher,
) -> Result<LfmExecution, LfmExecError> {
    let schema = &program.arena_schema.lens;
    if arenas.len() != schema.len() {
        return Err(LfmExecError::ArenaCountMismatch {
            expected: schema.len(),
            found: arenas.len(),
        });
    }
    for (i, (arena, &len)) in arenas.iter().zip(schema).enumerate() {
        if arena.len() != len as usize {
            return Err(LfmExecError::ArenaLenMismatch {
                arena: i as u32,
                expected: len,
                found: arena.len(),
            });
        }
    }

    let mut m = Machine {
        memory: vec![None; program.num_addrs as usize],
        arenas,
    };
    let mut records = LfmRecords::default();
    let mut public_words = Vec::new();

    for instr in &program.instrs {
        match instr {
            Instr::Const { out, value, .. } => {
                m.write(*out, *value)?;
                records.num_consts += 1;
            }
            Instr::BaseAlu {
                op, out, a, b, c, ..
            } => {
                let av = m.read_base(*a)?;
                let bv = m.read_base(*b)?;
                let cv = if *op == BaseOp::MulAdd {
                    m.read_base(*c)?
                } else {
                    FE::zero()
                };
                let ov = match op {
                    BaseOp::Add => &av + &bv,
                    BaseOp::Sub => &av - &bv,
                    BaseOp::Mul => &av * &bv,
                    BaseOp::Div => {
                        if bv == FE::zero() {
                            if av == FE::zero() {
                                FE::one() // the 0/0 = 1 convention
                            } else {
                                return Err(LfmExecError::DivByZero { addr: a.0 });
                            }
                        } else {
                            &av * &bv.inv().map_err(|_| LfmExecError::Internal("base inv"))?
                        }
                    }
                    BaseOp::MulAdd => &av * &bv + &cv,
                };
                m.write(*out, base_word(ov))?;
                records.balu.push(BaluRow {
                    a: av,
                    b: bv,
                    c: cv,
                    out: ov,
                });
            }
            Instr::ExtAlu {
                op, out, a, b, c, ..
            } => {
                let ae = m.read_ext(*a)?;
                let (be, bv_base) = if *op == ExtOp::MulBase {
                    let bb = m.read_base(*b)?;
                    (FEE::zero(), Some(bb))
                } else {
                    (m.read_ext(*b)?, None)
                };
                let ce = if *op == ExtOp::MulAdd {
                    m.read_ext(*c)?
                } else {
                    FEE::zero()
                };
                let oe = match op {
                    ExtOp::Add => &ae + &be,
                    ExtOp::Sub => &ae - &be,
                    ExtOp::Mul => &ae * &be,
                    ExtOp::Div => {
                        if be == FEE::zero() {
                            if ae == FEE::zero() {
                                FEE::one() // 0/0 = (1, 0, 0)
                            } else {
                                return Err(LfmExecError::DivByZero { addr: a.0 });
                            }
                        } else {
                            &ae * &be.inv().map_err(|_| LfmExecError::Internal("ext inv"))?
                        }
                    }
                    ExtOp::MulAdd => &ae * &be + &ce,
                    ExtOp::MulBase => {
                        let bb = bv_base.ok_or(LfmExecError::Internal("mulbase"))?;
                        let [a0, a1, a2] = *ae.value();
                        FEE::new([&a0 * &bb, &a1 * &bb, &a2 * &bb])
                    }
                };
                m.write(*out, ext_word(&oe))?;
                let lanes = |e: &FEE| -> [FE; 3] { *e.value() };
                records.xalu.push(XaluRow {
                    a: lanes(&ae),
                    b: bv_base.map_or_else(|| lanes(&be), |bb| [bb, FE::zero(), FE::zero()]),
                    c: lanes(&ce),
                    out: lanes(&oe),
                });
            }
            Instr::Select {
                bit,
                out_l,
                out_r,
                in_l,
                in_r,
                ..
            } => {
                let bv = m.read_base(*bit).map_err(|e| match e {
                    LfmExecError::NotBaseWord(a) => LfmExecError::NonBooleanBit(a),
                    other => other,
                })?;
                let l = m.read_word(*in_l)?;
                let r = m.read_word(*in_r)?;
                let (ol, or) = if bv == FE::zero() {
                    (l, r)
                } else if bv == FE::one() {
                    (r, l)
                } else {
                    return Err(LfmExecError::NonBooleanBit(bit.0));
                };
                m.write(*out_l, ol)?;
                m.write(*out_r, or)?;
                records.select.push(SelectRow {
                    bit: bv,
                    in_l: l,
                    in_r: r,
                    out_l: ol,
                    out_r: or,
                });
            }
            Instr::BitDec {
                input,
                bits,
                halves,
            } => {
                let v = m.read_base(*input)?;
                let canon = GoldilocksField::canonical(v.value());
                let bit_vals: [FE; 64] = core::array::from_fn(|i| FE::from((canon >> i) & 1));
                let top = (canon >> 32) as u32;
                let g = 0xFFFF_FFFFu64 - top as u64;
                let (z, ginv) = if g == 0 {
                    (FE::one(), FE::zero())
                } else {
                    (
                        FE::zero(),
                        FE::from(g)
                            .inv()
                            .map_err(|_| LfmExecError::Internal("bitdec ginv"))?,
                    )
                };
                for (i, (addr, _)) in bits.iter().enumerate() {
                    m.write(*addr, base_word(bit_vals[i]))?;
                }
                if let Some([h0, h1]) = halves {
                    // Half 0 is the HIGH word: it leads in big-endian order.
                    let hi = (canon >> 32) as u32;
                    let lo = (canon & 0xFFFF_FFFF) as u32;
                    m.write(h0.0, base_word(FE::from(hi.swap_bytes() as u64)))?;
                    m.write(h1.0, base_word(FE::from(lo.swap_bytes() as u64)))?;
                }
                records.bitdec.push(BitDecRow {
                    bits: bit_vals,
                    z,
                    ginv,
                });
            }
            Instr::Hash {
                mode, ins, outs, ..
            } => {
                let mut state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|_| FE::zero());
                let mut in_cols: [FE; HASH_STATE_FELTS] = core::array::from_fn(|_| FE::zero());
                if mode.num_input_cells() == 2 {
                    // Two cells, whatever they MEAN: two digests under Compress
                    // and Transcript, a chaining accumulator and four field
                    // elements under Leaf. What each cell is read AS belongs to
                    // the hasher and to the chip's lane split; what the executor
                    // owes is the memory reads the `LfmMem` receives claim, and
                    // those are the same two under all three.
                    let a = m.read_word(ins[0])?;
                    let b = m.read_word(ins[1])?;
                    state[0..4].clone_from_slice(&a);
                    state[4..8].clone_from_slice(&b);
                    state[8..12].clone_from_slice(&hasher.compress_iv());
                    in_cols[0..4].clone_from_slice(&a);
                    in_cols[4..8].clone_from_slice(&b);
                    // lanes 8–11 of the IN columns stay zero on two-cell rows
                } else {
                    for (cell, chunk) in ins.iter().zip(state.chunks_exact_mut(4)) {
                        chunk.clone_from_slice(&m.read_word(*cell)?);
                    }
                    in_cols = state;
                }
                // A hasher whose socket does not cover this row says so here,
                // with a reason, rather than producing a witness no AIR accepts.
                hasher
                    .admits(*mode, &state)
                    .map_err(LfmExecError::HasherRejected)?;
                let out_state = match mode {
                    // Through `compress_out`/`transcript_out`, NOT `permute`: a
                    // hasher that overrides the two-to-one construction —
                    // BLAKE3 does, its IV entering through `h` rather than the
                    // capacity lanes, and its transcript domain differing from
                    // its Merkle one — must have both overrides reach the `OUT`
                    // columns.
                    HashMode::Compress | HashMode::Transcript => {
                        let a: LfmWord = core::array::from_fn(|i| state[i]);
                        let b: LfmWord = core::array::from_fn(|i| state[4 + i]);
                        if *mode == HashMode::Compress {
                            hasher.compress_out(&a, &b)
                        } else {
                            hasher.transcript_out(&a, &b)
                        }
                    }
                    HashMode::Leaf => {
                        let acc: LfmWord = core::array::from_fn(|i| state[i]);
                        let f: LfmWord = core::array::from_fn(|i| state[4 + i]);
                        hasher.leaf_out(&acc, &f)
                    }
                    HashMode::Permute => hasher.permute(state),
                };
                if mode.num_output_cells() == 1 {
                    let digest: LfmWord = core::array::from_fn(|i| out_state[i]);
                    m.write(outs[0], digest)?;
                } else {
                    for (cell, chunk) in outs.iter().zip(out_state.chunks_exact(4)) {
                        let w: LfmWord = core::array::from_fn(|i| chunk[i]);
                        m.write(*cell, w)?;
                    }
                }
                records.hash.push(HashRow {
                    ins: in_cols,
                    outs: out_state,
                });
            }
            Instr::KeccakF(op) => {
                use super::layout::keccak as k;
                // 13 words × 4 lanes → 50 u32 halves (+ 2 must-be-zero slots).
                let mut halves = [0u32; k::NUM_HALVES];
                for (j, cell) in op.ins.iter().enumerate() {
                    let w = m.read_word(*cell)?;
                    for (l, lane) in w.iter().enumerate() {
                        let h = 4 * j + l;
                        let v = GoldilocksField::canonical(lane.value());
                        if h >= k::NUM_HALVES {
                            if v != 0 {
                                return Err(LfmExecError::KeccakSpareLaneNonZero {
                                    addr: cell.0,
                                    lane: l,
                                });
                            }
                        } else if v >= 1u64 << 32 {
                            return Err(LfmExecError::NotU32Half {
                                addr: cell.0,
                                lane: l,
                            });
                        } else {
                            halves[h] = v as u32;
                        }
                    }
                }
                let state = super::keccak_adapter::halves_to_state(&halves);

                // Absorb: XOR the rate block into the state's first 136 bytes.
                // Block byte k is byte k % 8 of lane k / 8, which is exactly
                // state byte offset k — rate bytes are lane-major and
                // little-endian within a lane, same as the byte columns.
                let mut block = [0u8; k::RATE_BYTES];
                let mut perm_in = state;
                if op.mode == KeccakMode::Absorb {
                    let mut bh = [0u32; k::BLOCK_HALVES];
                    for (j, cell) in op.block.iter().enumerate() {
                        let w = m.read_word(*cell)?;
                        for (l, lane) in w.iter().enumerate() {
                            let h = 4 * j + l;
                            let v = GoldilocksField::canonical(lane.value());
                            if h >= k::BLOCK_HALVES {
                                if v != 0 {
                                    return Err(LfmExecError::KeccakSpareLaneNonZero {
                                        addr: cell.0,
                                        lane: l,
                                    });
                                }
                            } else if v >= 1u64 << 32 {
                                return Err(LfmExecError::NotU32Half {
                                    addr: cell.0,
                                    lane: l,
                                });
                            } else {
                                bh[h] = v as u32;
                            }
                        }
                    }
                    for (h, half) in bh.iter().enumerate() {
                        block[4 * h..4 * h + 4].copy_from_slice(&half.to_le_bytes());
                    }
                    for lane in 0..k::RATE_LANES {
                        let mut chunk = [0u8; 8];
                        chunk.copy_from_slice(&block[lane * 8..lane * 8 + 8]);
                        perm_in[lane] ^= u64::from_le_bytes(chunk);
                    }
                }

                let output = super::keccak_adapter::permute(perm_in);
                for (cell, w) in op
                    .outs
                    .iter()
                    .zip(super::keccak_adapter::state_to_words(&output))
                {
                    m.write(*cell, w)?;
                }
                if let Some(rev) = &op.rev {
                    let words = super::keccak_adapter::reversed_digest_words(&output);
                    for (cell, w) in rev.outs.iter().zip(words) {
                        m.write(*cell, w)?;
                    }
                }
                records.keccak.push(KeccakRow {
                    mode: op.mode,
                    state,
                    block,
                    perm_in,
                    output,
                });
            }
            Instr::Blake3(op) => {
                use super::layout::blake3 as l;
                // 7 words × 4 lanes → the 28 input `u32` words, with no spare
                // slots: 28 divides by 4 exactly, so unlike `KeccakF` there is
                // no must-be-zero tail lane to police.
                let mut words = [0u32; l::IN_U32];
                for (j, cell) in op.ins.iter().enumerate() {
                    let w = m.read_word(*cell)?;
                    for (lane, value) in w.iter().enumerate() {
                        let v = GoldilocksField::canonical(value.value());
                        if v >= 1u64 << 32 {
                            return Err(LfmExecError::NotU32Half { addr: cell.0, lane });
                        }
                        words[4 * j + lane] = v as u32;
                    }
                }
                let values = Blake3Values {
                    h: core::array::from_fn(|i| words[i]),
                    m: core::array::from_fn(|i| words[8 + i]),
                    t: u64::from(words[24]) | (u64::from(words[25]) << 32),
                    block_len: words[26],
                    flags: words[27],
                };

                let out = values.output_words();
                for (j, cell) in op.outs.iter().enumerate() {
                    let word: LfmWord =
                        core::array::from_fn(|lane| FE::from(u64::from(out[4 * j + lane])));
                    m.write(*cell, word)?;
                }
                if let Some(rev) = &op.rev {
                    // The 32-byte digest is `out[0..8]` little-endian; reversing
                    // it is reading those bytes back-to-front, which is what the
                    // chip's flipped-coefficient send computes on the AIR side.
                    let mut digest = [0u8; 32];
                    for i in 0..8 {
                        digest[4 * i..4 * i + 4].copy_from_slice(&out[i].to_le_bytes());
                    }
                    digest.reverse();
                    for (w, cell) in rev.outs.iter().enumerate() {
                        let word: LfmWord = core::array::from_fn(|lane| {
                            let h = 4 * w + lane;
                            let mut b = [0u8; 4];
                            b.copy_from_slice(&digest[4 * h..4 * h + 4]);
                            FE::from(u64::from(u32::from_le_bytes(b)))
                        });
                        m.write(*cell, word)?;
                    }
                }
                records.blake3.push(values);
            }
            Instr::Hint {
                arena, index, out, ..
            } => {
                let words =
                    m.arenas
                        .get(*arena as usize)
                        .ok_or(LfmExecError::ArenaOutOfBounds {
                            arena: *arena,
                            index: *index,
                        })?;
                let w = *words
                    .get(*index as usize)
                    .ok_or(LfmExecError::ArenaOutOfBounds {
                        arena: *arena,
                        index: *index,
                    })?;
                m.write(*out, w)?;
                records.hint.push(w);
            }
            Instr::Pack { lanes, out, .. } => {
                let mut word = [FE::zero(), FE::zero(), FE::zero(), FE::zero()];
                for (i, lane) in lanes.iter().enumerate() {
                    word[i] = m.read_base(*lane)?;
                }
                m.write(*out, word)?;
                records.lanes.push(word);
            }
            Instr::Unpack { input, outs, .. } => {
                let word = m.read_word(*input)?;
                for (i, out) in outs.iter().enumerate() {
                    m.write(*out, base_word(word[i]))?;
                }
                records.lanes.push(word);
            }
            Instr::Public { addr, index } => {
                let w = m.read_word(*addr)?;
                records.public.push(w);
                public_words.push((*index, w));
            }
        }
    }

    Ok(LfmExecution {
        records,
        public_words,
        memory: m.memory,
    })
}

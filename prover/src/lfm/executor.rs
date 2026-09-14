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

use std::time::Instant;

use math::field::traits::IsPrimeField;

use crate::tables::types::{FE, FEE, GoldilocksField};

use super::blake3_chip::Blake3Values;
use super::compiler::{LfmColumnGroups, LfmProgram};
use super::exec_schedule::{self, LevelSchedule};
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

impl LfmRecords {
    /// Records sized from the census the compiler already holds.
    ///
    /// Pass 2 opens exactly one column-group row per instruction, and this
    /// executor pushes exactly one record per instruction of the same chip — so
    /// `ColumnGroup::real_rows` *is* each vector's final length, not an upper
    /// bound. The trace fill already depends on that identity: it fills rows
    /// `0..group.real_rows` by indexing the record vector
    /// ([`super::trace`]), so a vector short of its group's row count is
    /// already a panic there.
    ///
    /// Growing these by `push` instead re-allocated and copied every one of them
    /// ~log2(rows) times per proof — a few hundred MB of memcpy and ~20 large
    /// `mremap`s under the process allocator, on every concurrent worker at once.
    fn with_capacity(groups: &LfmColumnGroups) -> Self {
        LfmRecords {
            num_consts: 0,
            balu: Vec::with_capacity(groups.balu.real_rows),
            xalu: Vec::with_capacity(groups.xalu.real_rows),
            select: Vec::with_capacity(groups.select.real_rows),
            bitdec: Vec::with_capacity(groups.bitdec.real_rows),
            hash: Vec::with_capacity(groups.hash.real_rows),
            keccak: Vec::with_capacity(groups.keccak.real_rows),
            blake3: Vec::with_capacity(groups.blake3.real_rows),
            lanes: Vec::with_capacity(groups.lanes.real_rows),
            hint: Vec::with_capacity(groups.hint.real_rows),
            public: Vec::with_capacity(groups.public.real_rows),
        }
    }
}

impl LfmRecords {
    /// The same vectors, but with every row already present so an arm can write
    /// its slot instead of appending.
    ///
    /// ⛔ **The allocation is not new.** [`LfmRecords::with_capacity`] already
    /// reserved exactly these counts — the census is exact, not an upper bound —
    /// so what this costs over it is the zero-fill, one pass of stores over
    /// memory the push path was going to write anyway.
    fn with_slots(groups: &LfmColumnGroups) -> Self {
        let w = |_| FE::zero();
        let word = || -> LfmWord { core::array::from_fn(w) };
        LfmRecords {
            num_consts: 0,
            balu: vec![
                BaluRow {
                    a: FE::zero(),
                    b: FE::zero(),
                    c: FE::zero(),
                    out: FE::zero()
                };
                groups.balu.real_rows
            ],
            xalu: vec![
                XaluRow {
                    a: core::array::from_fn(w),
                    b: core::array::from_fn(w),
                    c: core::array::from_fn(w),
                    out: core::array::from_fn(w),
                };
                groups.xalu.real_rows
            ],
            select: vec![
                SelectRow {
                    bit: FE::zero(),
                    in_l: word(),
                    in_r: word(),
                    out_l: word(),
                    out_r: word(),
                };
                groups.select.real_rows
            ],
            bitdec: vec![
                BitDecRow {
                    bits: core::array::from_fn(w),
                    z: FE::zero(),
                    ginv: FE::zero(),
                };
                groups.bitdec.real_rows
            ],
            hash: vec![
                HashRow {
                    ins: core::array::from_fn(w),
                    outs: core::array::from_fn(w),
                };
                groups.hash.real_rows
            ],
            keccak: vec![
                KeccakRow {
                    mode: KeccakMode::Permute,
                    state: [0; 25],
                    block: [0; 136],
                    perm_in: [0; 25],
                    output: [0; 25],
                };
                groups.keccak.real_rows
            ],
            blake3: vec![
                Blake3Values {
                    h: [0; 8],
                    m: [0; 16],
                    t: 0,
                    block_len: 0,
                    flags: 0,
                };
                groups.blake3.real_rows
            ],
            lanes: vec![word(); groups.lanes.real_rows],
            hint: vec![word(); groups.hint.real_rows],
            public: vec![word(); groups.public.real_rows],
        }
    }
}

/// Where an executed instruction puts its record.
///
/// ⛔ **The level schedule does not run instructions in program order, but the
/// records must end up in it** — the trace fill indexes a chip's vector by row
/// and the row is the instruction's ordinal within its chip. So the serial loop
/// appends, as it always has, and the level loop writes the slot the schedule
/// assigned. One set of arms serves both; the only thing that differs is where
/// the row lands.
#[derive(Debug, Clone, Copy)]
enum Sink {
    /// Program order is the walk order: append.
    Push,
    /// Level order is the walk order: write row `n` of this chip.
    Slot(u32),
}

impl Sink {
    #[inline]
    fn put<T>(self, v: &mut Vec<T>, value: T) {
        match self {
            Sink::Push => v.push(value),
            Sink::Slot(n) => v[n as usize] = value,
        }
    }
}

/// The executor's write-once memory: one value array plus one occupancy bit per
/// address.
///
/// This used to be a single `Vec<Option<LfmWord>>`. `LfmWord` is `[F; 4]` = 32
/// bytes and `F` has no niche, so `Option<LfmWord>` is **40** bytes: the
/// occupancy flag cost 8 bytes per address *and* pushed the stride off the
/// cache line, so half of all cells spanned two lines. Splitting the flag out
/// restores the 32-byte stride (a cell never spans two lines) and puts the whole
/// occupancy map in `num_addrs / 8` bytes, which stays cache-resident where the
/// value array cannot.
///
/// The write-once rules are unchanged and still checked here, independently of
/// the compiler's tripwire panics and of the admission validator: a second write
/// to an address is [`LfmExecError::DoubleWrite`], and a read of an address no
/// instruction has written yet is [`LfmExecError::ReadBeforeWrite`]. ⚠ The bit
/// is the *only* thing that separates "unwritten" from "written zero" now — the
/// value array is zero-filled, so dropping the check would silently hand out
/// zeros instead of failing. The `executor_rejects_read_before_write` test is
/// what makes that unreachable.
#[derive(Debug)]
pub struct WriteOnceMemory {
    words: Vec<LfmWord>,
    /// Bit `i % 64` of word `i / 64` is "address `i` has been written".
    written: Vec<u64>,
}

impl WriteOnceMemory {
    fn new(num_addrs: usize) -> Self {
        WriteOnceMemory {
            words: vec![[FE::zero(); 4]; num_addrs],
            written: vec![0u64; num_addrs.div_ceil(64)],
        }
    }

    /// The final value at `addr`, or `None` if no instruction wrote it (which
    /// includes an address outside the program's range).
    pub fn get(&self, addr: Addr) -> Option<LfmWord> {
        self.read(addr).ok()
    }

    #[inline]
    fn read(&self, addr: Addr) -> Result<LfmWord, LfmExecError> {
        let i = addr.0 as usize;
        let Some(&w) = self.words.get(i) else {
            return Err(LfmExecError::ReadBeforeWrite(addr.0));
        };
        // `written` is sized `words.len().div_ceil(64)`, so the index is in
        // range here; `get` rather than `[]` keeps a would-be panic an error.
        let bits = self.written.get(i >> 6).copied().unwrap_or(0);
        if (bits >> (i & 63)) & 1 == 0 {
            return Err(LfmExecError::ReadBeforeWrite(addr.0));
        }
        Ok(w)
    }

    #[inline]
    fn write(&mut self, addr: Addr, w: LfmWord) -> Result<(), LfmExecError> {
        let i = addr.0 as usize;
        let slot = self
            .words
            .get_mut(i)
            .ok_or(LfmExecError::Internal("address out of range"))?;
        let bits = self
            .written
            .get_mut(i >> 6)
            .ok_or(LfmExecError::Internal("address out of range"))?;
        let mask = 1u64 << (i & 63);
        if *bits & mask != 0 {
            return Err(LfmExecError::DoubleWrite(addr.0));
        }
        *bits |= mask;
        *slot = w;
        Ok(())
    }
}

#[derive(Debug)]
pub struct LfmExecution {
    pub records: LfmRecords,
    /// The public output, in emission order: `(index, word)`.
    pub public_words: Vec<(u32, LfmWord)>,
    /// Final memory, exposed for tests and debugging.
    pub memory: WriteOnceMemory,
    /// Where the walk's time went. Diagnostic only: no proving decision reads
    /// it, and it is deliberately NOT compared by the identity gate, because a
    /// clock is the one thing two schedules are expected to disagree about.
    pub split: ExecSplit,
}

struct Machine<'a> {
    memory: WriteOnceMemory,
    arenas: &'a [Vec<LfmWord>],
}

impl Machine<'_> {
    #[inline]
    fn write(&mut self, addr: Addr, w: LfmWord) -> Result<(), LfmExecError> {
        self.memory.write(addr, w)
    }

    #[inline]
    fn read_word(&self, addr: Addr) -> Result<LfmWord, LfmExecError> {
        self.memory.read(addr)
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

/// The READ-ONLY half of a hash instruction: gather its input cells, check the
/// hasher admits the row, run the permutation, and hand back the record.
///
/// ⛔ **This half takes `&WriteOnceMemory` and that is the whole safety argument
/// of the parallel schedule.** A level's hashes are mutually independent
/// ([`super::exec_schedule`]), so they may run at once — but only because
/// nothing in this function can write. The borrow checker, not a comment and not
/// a runtime check, is what stops a worker from publishing a cell another worker
/// in the same level might read: there is no `&mut` to do it with. A hash
/// launched too early therefore reads a cell nobody has written and gets
/// [`LfmExecError::ReadBeforeWrite`] deterministically, on every run, instead of
/// a race whose outcome depends on the thread schedule.
///
/// It is also the one transcription of the hash semantics. The serial loop and
/// the level loop both call it, so the two schedules cannot drift apart in what
/// they compute — only in when they compute it.
fn hash_compute(
    memory: &WriteOnceMemory,
    hasher: &impl LfmHasher,
    mode: HashMode,
    ins: &[Addr; 3],
) -> Result<HashRow, LfmExecError> {
    let mut state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|_| FE::zero());
    let mut in_cols: [FE; HASH_STATE_FELTS] = core::array::from_fn(|_| FE::zero());
    if mode.num_input_cells() == 2 {
        // Two cells, whatever they MEAN: two digests under Compress and
        // Transcript, a chaining accumulator and four field elements under Leaf.
        // What each cell is read AS belongs to the hasher and to the chip's lane
        // split; what the executor owes is the memory reads the `LfmMem`
        // receives claim, and those are the same two under all three.
        let a = memory.read(ins[0])?;
        let b = memory.read(ins[1])?;
        state[0..4].clone_from_slice(&a);
        state[4..8].clone_from_slice(&b);
        // The capacity is the MODE's, not always the compress one: a hasher that
        // domain-separates through the capacity (RPO) makes a transcript step
        // and a parent different functions here, and the chip's `S8` copy
        // constraint agrees because both read `LfmHasher::mode_iv`.
        state[8..12].clone_from_slice(&hasher.mode_iv(mode));
        in_cols[0..4].clone_from_slice(&a);
        in_cols[4..8].clone_from_slice(&b);
        // lanes 8–11 of the IN columns stay zero on two-cell rows
    } else {
        for (cell, chunk) in ins.iter().zip(state.chunks_exact_mut(4)) {
            chunk.clone_from_slice(&memory.read(*cell)?);
        }
        in_cols = state;
    }
    // A hasher whose socket does not cover this row says so here, with a reason,
    // rather than producing a witness no AIR accepts.
    hasher
        .admits(mode, &state)
        .map_err(LfmExecError::HasherRejected)?;
    let out_state = match mode {
        // Through `compress_out`/`transcript_out`, NOT `permute`: a hasher that
        // overrides the two-to-one construction — BLAKE3 does, its IV entering
        // through `h` rather than the capacity lanes, and its transcript domain
        // differing from its Merkle one — must have both overrides reach the
        // `OUT` columns.
        HashMode::Compress | HashMode::Transcript => {
            let a: LfmWord = core::array::from_fn(|i| state[i]);
            let b: LfmWord = core::array::from_fn(|i| state[4 + i]);
            if mode == HashMode::Compress {
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
    Ok(HashRow {
        ins: in_cols,
        outs: out_state,
    })
}

/// The WRITING half: publish a computed row's output cells.
///
/// Always on the thread that owns the memory, under both schedules, so
/// [`LfmExecError::DoubleWrite`] stays exactly the check it has always been and
/// the occupancy bitset is never touched from two threads at once. That is why
/// the parallel path needs no atomics: making `written` an `AtomicU64` would put
/// a read-modify-write on every one of a wrap's ~6–9 M writes, on the serial
/// path too, to protect a word this design never shares.
fn hash_apply(
    memory: &mut WriteOnceMemory,
    mode: HashMode,
    outs: &[Addr; 3],
    row: &HashRow,
) -> Result<(), LfmExecError> {
    if mode.num_output_cells() == 1 {
        let digest: LfmWord = core::array::from_fn(|i| row.outs[i]);
        memory.write(outs[0], digest)?;
    } else {
        for (cell, chunk) in outs.iter().zip(row.outs.chunks_exact(4)) {
            let w: LfmWord = core::array::from_fn(|i| chunk[i]);
            memory.write(*cell, w)?;
        }
    }
    Ok(())
}

/// One instruction, against the machine — the body the serial loop always had,
/// lifted out so the level schedule runs the SAME arms rather than a second copy
/// of them.
///
/// The level schedule calls this only for non-hash instructions; its hash rows
/// go through [`hash_compute`] and [`hash_apply`], which this function's own
/// `Hash` arm is itself built from. So both schedules execute one transcription
/// of every arm.
fn step(
    m: &mut Machine<'_>,
    records: &mut LfmRecords,
    public_words: &mut Vec<(u32, LfmWord)>,
    instr: &Instr,
    hasher: &impl LfmHasher,
    sink: Sink,
) -> Result<(), LfmExecError> {
    {
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
                sink.put(
                    &mut records.balu,
                    BaluRow {
                        a: av,
                        b: bv,
                        c: cv,
                        out: ov,
                    },
                );
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
                sink.put(
                    &mut records.xalu,
                    XaluRow {
                        a: lanes(&ae),
                        b: bv_base.map_or_else(|| lanes(&be), |bb| [bb, FE::zero(), FE::zero()]),
                        c: lanes(&ce),
                        out: lanes(&oe),
                    },
                );
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
                sink.put(
                    &mut records.select,
                    SelectRow {
                        bit: bv,
                        in_l: l,
                        in_r: r,
                        out_l: ol,
                        out_r: or,
                    },
                );
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
                sink.put(
                    &mut records.bitdec,
                    BitDecRow {
                        bits: bit_vals,
                        z,
                        ginv,
                    },
                );
            }
            Instr::Hash {
                mode, ins, outs, ..
            } => {
                let row = hash_compute(&m.memory, hasher, *mode, ins)?;
                hash_apply(&mut m.memory, *mode, outs, &row)?;
                sink.put(&mut records.hash, row);
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
                sink.put(
                    &mut records.keccak,
                    KeccakRow {
                        mode: op.mode,
                        state,
                        block,
                        perm_in,
                        output,
                    },
                );
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
                sink.put(&mut records.blake3, values);
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
                sink.put(&mut records.hint, w);
            }
            Instr::Pack { lanes, out, .. } => {
                let mut word = [FE::zero(), FE::zero(), FE::zero(), FE::zero()];
                for (i, lane) in lanes.iter().enumerate() {
                    word[i] = m.read_base(*lane)?;
                }
                m.write(*out, word)?;
                sink.put(&mut records.lanes, word);
            }
            Instr::Unpack { input, outs, .. } => {
                let word = m.read_word(*input)?;
                for (i, out) in outs.iter().enumerate() {
                    m.write(*out, base_word(word[i]))?;
                }
                sink.put(&mut records.lanes, word);
            }
            Instr::Public { addr, index } => {
                let w = m.read_word(*addr)?;
                sink.put(&mut records.public, w);
                sink.put(public_words, (*index, w));
            }
        }
    }
    Ok(())
}

/// Which order [`execute_scheduled`] evaluates a program in.
///
/// ⛔ **A schedule may not change one word of the witness.** The program is a
/// DAG and every instruction is a pure function of the cells it reads, so the
/// result is schedule-independent by construction; `exec_identity_tests` is
/// what shows the construction was implemented, and the tree-scale `IDENTITY:`
/// diff is what shows it at production shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    /// The plain `for instr in &program.instrs` loop. The reference, kept in the
    /// file rather than in history: every A/B, and the byte gate itself, is this
    /// against the other arm in ONE binary.
    Serial,
    /// Level-scheduled: the mutually independent hash instructions of each
    /// depth level run on rayon's global pool, everything else stays serial.
    LevelParallel {
        /// Levels narrower than this run on the calling thread instead of being
        /// handed to rayon.
        ///
        /// ⓘ A parameter rather than a constant because it decides whether a
        /// test can see the parallel path at all: the laptop cases are 15 wide
        /// at their widest, so anything at or above 16 sends every level of
        /// every case down the serial branch and the gate goes quiet. The gate
        /// runs at 2; production runs at [`PRODUCTION_COALESCE_BELOW`].
        coalesce_below: usize,
    },
}

/// Where a level stops being worth a fork/join, on the 09-12 wrap histogram:
/// 1,767 of 2,237 levels are ≤ 16 wide and hold 7,303 hashes (1.1%). Serially
/// those cost 7,303 × 2,356 ns ≈ 17 ms; on rayon they would cost ≈ 4 ms of
/// permutation plus 1,767 fork/joins, which at 2–5 µs is 3.5–8.8 ms. So the two
/// are within ~10 ms of each other on a ~450 ms budget and the serial branch
/// wins on the tie-breaker that matters: it removes 79% of this executor's
/// interactions with a pool that up to four sibling proofs are sharing.
pub const PRODUCTION_COALESCE_BELOW: usize = 16;

/// Where `execute`'s time went, so a lever aimed at one phase can be read.
///
/// ⚠ The phases are recorded, not derived: a number obtained by subtracting two
/// others assumes there is nothing else between them, which is how a lever gets
/// credited with a stage it never touched.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExecSplit {
    /// Levels in the schedule; `0` under [`Schedule::Serial`].
    pub levels: usize,
    /// Levels actually handed to rayon — **`0` here with `levels` large means
    /// every level was coalesced and nothing ran in parallel**, which is the
    /// reading that would otherwise be invisible.
    pub parallel_levels: usize,
    /// Hash rows inside those levels.
    pub parallel_hashes: usize,
    /// Building the schedule: one forward pass plus a counting sort.
    pub depth_pass: f64,
    /// Computing the permutations, whether on the pool or coalesced.
    pub hash_phase: f64,
    /// Publishing each level's output cells and record rows, on this thread.
    pub apply: f64,
    /// Everything that is not a hash instruction.
    pub residue: f64,
}

/// The witness, under whichever schedule the caller names.
///
/// `Schedule::Serial` is the reference loop, unchanged. The level schedule
/// evaluates, for each depth level in order, the level's hashes (on rayon when
/// the level is wide enough) and then the level's non-hash instructions in
/// program order — which is sound for the reason spelled out in
/// [`super::exec_schedule`] and safe for the reason spelled out on
/// [`hash_compute`].
pub fn execute_scheduled(
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
    hasher: &(impl LfmHasher + Sync),
    schedule: Schedule,
) -> Result<LfmExecution, LfmExecError> {
    execute_inner(program, arenas, hasher, schedule, 1)
}

/// The production entry point. Signature unchanged; the schedule comes from the
/// environment so every existing caller keeps working and the box arms select an
/// arm per process.
pub fn execute(
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
    hasher: &(impl LfmHasher + Sync),
) -> Result<LfmExecution, LfmExecError> {
    execute_inner(program, arenas, hasher, default_schedule(), 1)
}

/// `LFM_EXEC_PARALLEL=0` selects the serial reference; unset or `1` selects the
/// level schedule.
///
/// ⚠ Cached, so one process sees one value — which is why it is NOT how the
/// identity gate reaches the two arms (a test flipping this would gate whichever
/// arm ran first). The gate calls [`execute_scheduled`]; this exists so a box arm
/// can pick a schedule without a rebuild.
pub fn default_schedule() -> Schedule {
    static ON: std::sync::OnceLock<Schedule> = std::sync::OnceLock::new();
    *ON.get_or_init(
        // An EMPTY value reads as unset: `FOO= cmd` is the shell's way of
        // clearing a variable, and panicking on it would fail a run for a
        // spelling of "default".
        || match std::env::var("LFM_EXEC_PARALLEL").ok().as_deref() {
            Some("0") => Schedule::Serial,
            None | Some("") | Some("1") => Schedule::LevelParallel {
                coalesce_below: PRODUCTION_COALESCE_BELOW,
            },
            Some(other) => panic!("LFM_EXEC_PARALLEL must be `0` or `1`, got `{other}`"),
        },
    )
}

/// ⛔ Test-only: run the level schedule over a DELIBERATELY MERGED set of
/// levels, so a hash and one of its own inputs share a level.
///
/// The executor must then fail with [`LfmExecError::ReadBeforeWrite`], at the
/// same address, on every run — see [`super::exec_schedule::build_merged`]. A
/// gate that still passed with the level boundary removed would not be a gate.
#[cfg(test)]
pub fn execute_with_merged_levels(
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
    hasher: &(impl LfmHasher + Sync),
    coalesce_below: usize,
    merge: u32,
) -> Result<LfmExecution, LfmExecError> {
    execute_inner(
        program,
        arenas,
        hasher,
        Schedule::LevelParallel { coalesce_below },
        merge,
    )
}

fn execute_inner(
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
    hasher: &(impl LfmHasher + Sync),
    schedule: Schedule,
    merge: u32,
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

    // ⚠ Built BEFORE the machine, and its two big scratch arrays are freed
    // inside, so the pass's own peak (~66–80 MiB on a wrap) never stacks on top
    // of the memory array's 188–275 MiB. What survives into the run is the index
    // vectors alone.
    let t = Instant::now();
    let levels = match schedule {
        Schedule::Serial => None,
        Schedule::LevelParallel { .. } => Some(exec_schedule::build_with_merge(program, merge)),
    };
    let mut split = ExecSplit {
        depth_pass: t.elapsed().as_secs_f64(),
        levels: levels.as_ref().map_or(0, |l| l.levels()),
        ..ExecSplit::default()
    };

    let mut m = Machine {
        memory: WriteOnceMemory::new(program.num_addrs as usize),
        arenas,
    };
    // The serial walk IS program order, so it appends into empty vectors exactly
    // as it always did; the level walk is not, so it writes slots that already
    // exist. Same bytes reserved either way.
    let mut records = match &levels {
        None => LfmRecords::with_capacity(&program.groups),
        Some(_) => LfmRecords::with_slots(&program.groups),
    };
    let mut public_words = match &levels {
        None => Vec::with_capacity(program.groups.public.real_rows),
        Some(_) => vec![(0u32, [FE::zero(); 4]); program.groups.public.real_rows],
    };

    match &levels {
        None => {
            let t = Instant::now();
            for instr in &program.instrs {
                step(
                    &mut m,
                    &mut records,
                    &mut public_words,
                    instr,
                    hasher,
                    Sink::Push,
                )?;
            }
            split.residue = t.elapsed().as_secs_f64();
        }
        Some(levels) => {
            let coalesce_below = match schedule {
                Schedule::LevelParallel { coalesce_below } => coalesce_below,
                Schedule::Serial => unreachable!("a schedule was built, so it is not Serial"),
            };
            run_levels(
                program,
                levels,
                hasher,
                coalesce_below,
                &mut m,
                &mut records,
                &mut public_words,
                &mut split,
            )?;
        }
    }

    // The sizing above is exact, not an estimate: assert it on the success path
    // so a chip whose emitter and executor arm drift apart says so here, where
    // the two are written, instead of as an index panic inside the parallel
    // fill. Chip order: const, balu, xalu, select, bitdec, hash, keccak,
    // blake3, lanes, hint, public. (A `?` above returns early with short
    // vectors by design — that program did not finish.)
    debug_assert_eq!(
        [
            records.num_consts,
            records.balu.len(),
            records.xalu.len(),
            records.select.len(),
            records.bitdec.len(),
            records.hash.len(),
            records.keccak.len(),
            records.blake3.len(),
            records.lanes.len(),
            records.hint.len(),
            records.public.len(),
        ],
        [
            program.groups.const_.real_rows,
            program.groups.balu.real_rows,
            program.groups.xalu.real_rows,
            program.groups.select.real_rows,
            program.groups.bitdec.real_rows,
            program.groups.hash.real_rows,
            program.groups.keccak.real_rows,
            program.groups.blake3.real_rows,
            program.groups.lanes.real_rows,
            program.groups.hint.real_rows,
            program.groups.public.real_rows,
        ],
        "record counts must equal the emitted column-group row counts"
    );

    Ok(LfmExecution {
        records,
        public_words,
        memory: m.memory,
        split,
    })
}

/// The level schedule: for each depth level, its hashes, then its non-hash work.
///
/// # The protocol, and why it needs no lock, no atomic and no `unsafe`
///
/// A level's hashes are mutually independent, so they may be computed in any
/// order. They are computed and NOT applied: the workers see `&WriteOnceMemory`
/// and hand back rows, and this thread then publishes every output cell and
/// every record slot before the next level starts. Three things fall out of
/// that, and each of them is a bug that did not have to be prevented:
///
/// * **No data race is expressible.** No `&mut` to the memory exists while the
///   level runs, so the borrow checker rejects a worker that tried to write.
/// * **A wrong schedule is a deterministic error, not a race.** A hash launched
///   before its input is applied reads an unwritten cell and gets
///   [`LfmExecError::ReadBeforeWrite`] — the same error, at the same address, on
///   every run. That is what the merged-level mutation gate rides on.
/// * **`DoubleWrite` and the occupancy bitset are untouched.** Both still go
///   through the one `WriteOnceMemory::write` on one thread.
///
/// # Order
///
/// Rows are applied in ascending row number, which inside a level is program
/// order, so `records.hash` ends up exactly the vector the serial loop would
/// have pushed. The row numbers come from the schedule, so a row is written once
/// and read once and the vector is never searched.
#[allow(clippy::too_many_arguments)]
fn run_levels(
    program: &LfmProgram,
    levels: &LevelSchedule,
    hasher: &(impl LfmHasher + Sync),
    coalesce_below: usize,
    m: &mut Machine<'_>,
    records: &mut LfmRecords,
    public_words: &mut Vec<(u32, LfmWord)>,
    split: &mut ExecSplit,
) -> Result<(), LfmExecError> {
    // One buffer, reused by every level, so a 2,237-level program allocates once
    // rather than 2,237 times. It grows to the widest level and stays there:
    // ~3.4 MiB on a wrap, ~4.8 MiB on a node.
    let mut computed: Vec<Result<HashRow, LfmExecError>> = Vec::new();

    for d in 0..levels.levels() {
        let rows = levels.hashes_at(d);

        let t = Instant::now();
        if rows.len() < coalesce_below {
            // Narrow: a fork/join for a handful of permutations costs more than
            // it saves, and on a box up to four sibling proofs are sharing this
            // pool. 79% of a wrap's levels come through here holding 1.1% of its
            // hashes.
            computed.clear();
            for &row in rows {
                let instr = &program.instrs[levels.instr_of_row(row)];
                computed.push(hash_one(&m.memory, hasher, instr));
            }
        } else {
            compute_level(&m.memory, hasher, program, levels, rows, &mut computed);
            split.parallel_levels += 1;
            split.parallel_hashes += rows.len();
        }
        split.hash_phase += t.elapsed().as_secs_f64();

        // ⓘ The FIRST failure by row number, not whichever worker finished
        // first: the same program must report the same address on every run, or
        // the mutation gate below it is flaky and the campaign's error messages
        // stop being comparable between arms.
        let t = Instant::now();
        for (&row, out) in rows.iter().zip(&computed) {
            let row_value = match out {
                Ok(r) => r,
                Err(e) => return Err(e.clone()),
            };
            let instr = &program.instrs[levels.instr_of_row(row)];
            let Instr::Hash { mode, outs, .. } = instr else {
                return Err(LfmExecError::Internal(
                    "the schedule listed a non-hash instruction as a hash row",
                ));
            };
            hash_apply(&mut m.memory, *mode, outs, row_value)?;
            records.hash[row as usize] = row_value.clone();
        }
        split.apply += t.elapsed().as_secs_f64();

        // The level's non-hash work. It may read this level's hashes and each
        // other, so it stays serial and in program order — which is a
        // topological order, so that is enough.
        let t = Instant::now();
        for &i in levels.others_at(d) {
            let i = i as usize;
            step(
                m,
                records,
                public_words,
                &program.instrs[i],
                hasher,
                Sink::Slot(levels.record_row(i)),
            )?;
        }
        split.residue += t.elapsed().as_secs_f64();
    }
    Ok(())
}

/// One hash instruction's read-only half, dispatched from a program index.
fn hash_one(
    memory: &WriteOnceMemory,
    hasher: &impl LfmHasher,
    instr: &Instr,
) -> Result<HashRow, LfmExecError> {
    match instr {
        Instr::Hash { mode, ins, .. } => hash_compute(memory, hasher, *mode, ins),
        _ => Err(LfmExecError::Internal(
            "the schedule listed a non-hash instruction as a hash row",
        )),
    }
}

/// A level's permutations, on rayon's GLOBAL pool.
///
/// ⚠ The global pool on purpose, and never a private one. With four sibling
/// proofs live at level 0 a private pool per executor would be 120 threads on 30
/// cores; the global pool shares itself between them instead, which is the
/// oversubscription finding turned into a default rather than a knob.
#[cfg(feature = "parallel")]
fn compute_level(
    memory: &WriteOnceMemory,
    hasher: &(impl LfmHasher + Sync),
    program: &LfmProgram,
    levels: &LevelSchedule,
    rows: &[u32],
    out: &mut Vec<Result<HashRow, LfmExecError>>,
) {
    use rayon::prelude::*;

    rows.par_iter()
        .map(|&row| hash_one(memory, hasher, &program.instrs[levels.instr_of_row(row)]))
        .collect_into_vec(out);
}

/// [`compute_level`] without the feature: the same bodies in order. This is the
/// schedule, not the reference — a `--no-default-features` build still has to
/// produce the same witness, and it does, because the level order alone is what
/// makes the result right.
#[cfg(not(feature = "parallel"))]
fn compute_level(
    memory: &WriteOnceMemory,
    hasher: &(impl LfmHasher + Sync),
    program: &LfmProgram,
    levels: &LevelSchedule,
    rows: &[u32],
    out: &mut Vec<Result<HashRow, LfmExecError>>,
) {
    out.clear();
    out.extend(
        rows.iter()
            .map(|&row| hash_one(memory, hasher, &program.instrs[levels.instr_of_row(row)])),
    );
}

/// ★ Which instruction WROTE `addr`, with a window of its neighbours — the map
/// from a [`LfmExecError::DivByZero`] address back to the assert that failed.
///
/// ⚠ **This is more generally useful than it looks.** `assert_eq` lowers to
/// `diff = a − b; _ = diff / ZERO` (`builder.rs`), and the executor reports the
/// NUMERATOR's address. So a `DivByZero` is never an inversion gone wrong — it
/// is always a failing equality assert, and the address always names the `diff`
/// cell. Given the address, the `Sub` that produced it and the two operands
/// feeding that `Sub` identify the assert exactly, without bisecting the
/// emitter.
///
/// Returns a human-readable report; for diagnostics, not for proving.
pub fn locate_addr(program: &LfmProgram, addr: u64) -> String {
    use core::fmt::Write as _;

    let writes = |i: &Instr| -> Vec<u64> {
        match i {
            Instr::Const { out, .. } => vec![out.0],
            Instr::BaseAlu { out, .. } | Instr::ExtAlu { out, .. } => vec![out.0],
            Instr::Select { out_l, out_r, .. } => vec![out_l.0, out_r.0],
            Instr::BitDec { bits, halves, .. } => bits
                .iter()
                .map(|(a, _)| a.0)
                .chain(halves.iter().flatten().map(|(a, _)| a.0))
                .collect(),
            Instr::Hash { outs, .. } => outs.iter().map(|a| a.0).collect(),
            Instr::Unpack { outs, .. } => outs.iter().map(|a| a.0).collect(),
            Instr::Pack { out, .. } => vec![out.0],
            _ => Vec::new(),
        }
    };

    let mut out = String::new();
    let Some(idx) = program
        .instrs
        .iter()
        .position(|i| writes(i).contains(&addr))
    else {
        let _ = writeln!(
            out,
            "addr {addr}: no instruction writes it (an arena word?)"
        );
        return out;
    };
    let _ = writeln!(
        out,
        "addr {addr} written by instruction {idx} of {}:",
        program.instrs.len()
    );
    let lo = idx.saturating_sub(4);
    let hi = (idx + 3).min(program.instrs.len());
    for (k, i) in program.instrs[lo..hi].iter().enumerate() {
        let marker = if lo + k == idx { "→" } else { " " };
        let _ = writeln!(out, "  {marker} [{}] {i:?}", lo + k);
    }
    out
}

//! The LFM eDSL: typed SSA handles over write-once cells.
//!
//! The builder is an ordinary Rust API that *emits instructions*; host-side
//! `for` loops unroll and nothing loop-shaped reaches the machine. Every
//! emitted destination gets the next dense address (SSA — uniqueness and
//! acyclicity by construction), every operand use bumps that cell's read
//! count, and the compiler later backfills the counts as the static
//! multiplicities the write-once memory argument needs.

use std::collections::HashMap;

use math::field::traits::IsPrimeField;

use crate::tables::types::{FE, FEE, GoldilocksField};

use super::instr::{Addr, ArenaId, BaseOp, ExtOp, HashMode, Instr, KeccakMode};
use super::layout;
use super::word::{LfmWord, base_word, ext_word};

/// A cell holding a base field value `(v, 0, 0, 0)`.
#[derive(Debug, Clone, Copy)]
pub struct Felt(pub(crate) Addr);
/// A cell holding an Fp3 value `(a0, a1, a2, 0)`.
#[derive(Debug, Clone, Copy)]
pub struct Ext(pub(crate) Addr);
/// A cell holding a digest (all four lanes).
#[derive(Debug, Clone, Copy)]
pub struct DigestVal(pub(crate) Addr);
/// An untyped word cell.
#[derive(Debug, Clone, Copy)]
pub struct Cell(pub(crate) Addr);
/// A cell holding a boolean `(b, 0, 0, 0)`, `b ∈ {0, 1}`.
#[derive(Debug, Clone, Copy)]
pub struct Bit(pub(crate) Addr);

macro_rules! handle_addr {
    ($($t:ty),*) => {$(
        impl $t {
            /// The underlying cell address.
            pub fn addr(&self) -> Addr { self.0 }
            /// Erase the type: any handle is a word cell.
            pub fn as_cell(&self) -> Cell { Cell(self.0) }
        }
    )*};
}
handle_addr!(Felt, Ext, DigestVal, Cell, Bit);

impl Bit {
    /// A bit is a valid base felt.
    pub fn as_felt(&self) -> Felt {
        Felt(self.0)
    }
}

impl Cell {
    /// Reinterpret as a digest cell (e.g. hint words feeding `compress`).
    pub fn as_digest(&self) -> DigestVal {
        DigestVal(self.0)
    }

    /// Reinterpret as an ext value. Sound by construction: every ext-typed
    /// bus receive carries a constant zero in lane 3, so a word whose lane 3
    /// is nonzero makes the program unprovable (and the executor errors).
    pub fn as_ext(&self) -> Ext {
        Ext(self.0)
    }
}

impl Felt {
    /// A base cell `(v, 0, 0, 0)` is a valid ext cell `(v, 0, 0)`.
    pub fn as_ext(&self) -> Ext {
        Ext(self.0)
    }
}

/// Declared arena lengths (in words), fixed at build time. The executor
/// checks supplied arenas against this schema; the admission validator checks
/// every `Hint` lands inside it.
#[derive(Debug, Clone, Default)]
pub struct ArenaSchema {
    pub lens: Vec<u32>,
}

/// Everything the compiler needs: the emitted instructions plus the builder's
/// bookkeeping. Fields are public so tests can hand-build malformed sources
/// to exercise the compiler's invariant panics.
#[derive(Debug)]
pub struct LfmProgramSource {
    pub instrs: Vec<Instr>,
    pub num_addrs: u64,
    /// Reads per address, indexed BY address.
    ///
    /// Dense rather than a map because [`LfmBuilder::alloc`] hands out
    /// addresses sequentially from zero, so the key space is exactly
    /// `0..num_addrs` with no holes — a `HashMap` was paying ~16 bytes plus
    /// control per entry, and its bucket array is the emitter's second-largest
    /// allocation at production query counts.
    pub read_counts: Vec<u64>,
    pub arena_schema: ArenaSchema,
    pub public_len: u32,
}

#[derive(Default)]
pub struct LfmBuilder {
    instrs: Vec<Instr>,
    next_addr: u64,
    const_pool: HashMap<[u64; 4], Addr>,
    /// Parallel to the address space; see [`LfmProgramSource::read_counts`].
    read_counts: Vec<u64>,
    arena_schema: ArenaSchema,
    public_len: u32,
}

impl LfmBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    fn alloc(&mut self) -> Addr {
        let addr = Addr(self.next_addr);
        self.next_addr += 1;
        // Keeps `read_counts` exactly as long as the address space, which is
        // what lets `read` index instead of hash.
        self.read_counts.push(0);
        addr
    }

    fn read(&mut self, addr: Addr) {
        self.read_counts[addr.0 as usize] += 1;
    }

    // ---- constants (interned; one LFM_CONST row per distinct word) ----

    fn word_const(&mut self, value: LfmWord) -> Addr {
        let key: [u64; 4] = core::array::from_fn(|i| GoldilocksField::canonical(value[i].value()));
        if let Some(&addr) = self.const_pool.get(&key) {
            return addr;
        }
        let out = self.alloc();
        self.instrs.push(Instr::Const {
            out,
            value,
            mult: 0,
        });
        self.const_pool.insert(key, out);
        out
    }

    pub fn felt_const(&mut self, v: FE) -> Felt {
        Felt(self.word_const(base_word(v)))
    }

    pub fn ext_const(&mut self, v: &FEE) -> Ext {
        Ext(self.word_const(ext_word(v)))
    }

    pub fn digest_const(&mut self, v: LfmWord) -> DigestVal {
        DigestVal(self.word_const(v))
    }

    pub fn bit_const(&mut self, b: bool) -> Bit {
        Bit(self.word_const(base_word(if b { FE::one() } else { FE::zero() })))
    }

    // ---- base ALU ----

    fn balu(&mut self, op: BaseOp, a: Felt, b: Felt, c: Option<Felt>) -> Felt {
        self.read(a.0);
        self.read(b.0);
        if let Some(c) = c {
            self.read(c.0);
        }
        let out = self.alloc();
        self.instrs.push(Instr::BaseAlu {
            op,
            out,
            a: a.0,
            b: b.0,
            c: c.map_or(Addr(0), |c| c.0),
            mult: 0,
        });
        Felt(out)
    }

    pub fn add(&mut self, a: Felt, b: Felt) -> Felt {
        self.balu(BaseOp::Add, a, b, None)
    }
    pub fn sub(&mut self, a: Felt, b: Felt) -> Felt {
        self.balu(BaseOp::Sub, a, b, None)
    }
    pub fn mul(&mut self, a: Felt, b: Felt) -> Felt {
        self.balu(BaseOp::Mul, a, b, None)
    }
    /// `a / b` under the machine convention `0/0 = 1`, `x/0 = error`.
    pub fn div(&mut self, a: Felt, b: Felt) -> Felt {
        self.balu(BaseOp::Div, a, b, None)
    }
    /// `a·b + c` — the Horner step, first-class.
    pub fn mul_add(&mut self, a: Felt, b: Felt, c: Felt) -> Felt {
        self.balu(BaseOp::MulAdd, a, b, Some(c))
    }

    // ---- Fp3 ALU (lanes 0–2, w³ = 2) ----

    fn xalu(&mut self, op: ExtOp, a: Addr, b: Addr, c: Option<Addr>) -> Ext {
        self.read(a);
        self.read(b);
        if let Some(c) = c {
            self.read(c);
        }
        let out = self.alloc();
        self.instrs.push(Instr::ExtAlu {
            op,
            out,
            a,
            b,
            c: c.unwrap_or(Addr(0)),
            mult: 0,
        });
        Ext(out)
    }

    pub fn eadd(&mut self, a: Ext, b: Ext) -> Ext {
        self.xalu(ExtOp::Add, a.0, b.0, None)
    }
    pub fn esub(&mut self, a: Ext, b: Ext) -> Ext {
        self.xalu(ExtOp::Sub, a.0, b.0, None)
    }
    pub fn emul(&mut self, a: Ext, b: Ext) -> Ext {
        self.xalu(ExtOp::Mul, a.0, b.0, None)
    }
    /// `a / b` under `0/0 = (1, 0, 0)`, `x/0 = error`.
    pub fn ediv(&mut self, a: Ext, b: Ext) -> Ext {
        self.xalu(ExtOp::Div, a.0, b.0, None)
    }
    pub fn emul_add(&mut self, a: Ext, b: Ext, c: Ext) -> Ext {
        self.xalu(ExtOp::MulAdd, a.0, b.0, Some(c.0))
    }
    /// Extension × base — 3 base multiplies instead of 9.
    pub fn emul_base(&mut self, a: Ext, b: Felt) -> Ext {
        self.xalu(ExtOp::MulBase, a.0, b.0, None)
    }

    // ---- assertions (lowered, no chip) ----

    /// `assert_eq` lowers to `diff = a − b; _ = diff / ZERO`: provable (and
    /// executable) iff `diff = 0` under the `0/0 = 1` convention.
    pub fn assert_eq(&mut self, a: Felt, b: Felt) {
        let diff = self.sub(a, b);
        let zero = self.felt_const(FE::zero());
        let _ = self.div(diff, zero);
    }

    pub fn assert_eq_ext(&mut self, a: Ext, b: Ext) {
        let diff = self.esub(a, b);
        let zero = self.ext_const(&FEE::zero());
        let _ = self.ediv(diff, zero);
    }

    // ---- select / bitdec ----

    /// Conditional swap: `bit = 0 ⇒ (l, r)`; `bit = 1 ⇒ (r, l)`.
    pub fn select(&mut self, bit: Bit, l: Cell, r: Cell) -> (Cell, Cell) {
        self.read(bit.0);
        self.read(l.0);
        self.read(r.0);
        let out_l = self.alloc();
        let out_r = self.alloc();
        self.instrs.push(Instr::Select {
            bit: bit.0,
            out_l,
            out_r,
            in_l: l.0,
            in_r: r.0,
            mult_l: 0,
            mult_r: 0,
        });
        (Cell(out_l), Cell(out_r))
    }

    /// Canonical 64-bit decomposition; returns the low `nbits` bits as cells
    /// (low-to-high). Only these become memory cells; all 64 bits exist as
    /// constrained witness columns either way.
    pub fn bit_dec(&mut self, x: Felt, nbits: usize) -> Vec<Bit> {
        assert!(nbits <= 64, "bit_dec: at most 64 bits");
        self.read(x.0);
        let bits: Vec<(Addr, u64)> = (0..nbits).map(|_| (self.alloc(), 0)).collect();
        let handles = bits.iter().map(|(a, _)| Bit(*a)).collect();
        self.instrs.push(Instr::BitDec { input: x.0, bits });
        handles
    }

    // ---- hash ----

    /// Two digest cells → one digest cell.
    pub fn compress(&mut self, a: DigestVal, b: DigestVal) -> DigestVal {
        self.two_to_one(HashMode::Compress, a, b)
    }

    /// A Merkle LEAF over one cell read as four FIELD ELEMENTS.
    ///
    /// The only mode whose input is not a digest: each felt is split into a
    /// checked `lo`/`hi` `u32` pair inside the chip, so arbitrary Goldilocks
    /// data can be hashed by a socket whose lanes must be `u32`. The `"LFML"`
    /// domain keeps a leaf un-replayable as a parent whatever the tree's shape.
    pub fn leaf(&mut self, felts: Cell) -> DigestVal {
        self.read(felts.0);
        let out = self.alloc();
        self.instrs.push(Instr::Hash {
            mode: HashMode::Leaf,
            ins: [felts.0, Addr(0), Addr(0)],
            outs: [out, Addr(0), Addr(0)],
            mults: [0, 0, 0],
        });
        DigestVal(out)
    }

    /// One step of the Fiat–Shamir transcript chain: two cells → one cell, in
    /// the TRANSCRIPT hash domain.
    ///
    /// The same socket and the same columns as [`LfmBuilder::compress`]; the
    /// row's preprocessed mode selects the domain tag, so a transcript step and
    /// a Merkle parent over the same two cells are different digests. Callers
    /// go through [`super::edsl::SpongeVar`] rather than here — the chain's
    /// operand sequence is what its security argument rests on, and a raw step
    /// is an easy way to break it.
    pub fn transcript_step(&mut self, a: DigestVal, b: DigestVal) -> DigestVal {
        self.two_to_one(HashMode::Transcript, a, b)
    }

    fn two_to_one(&mut self, mode: HashMode, a: DigestVal, b: DigestVal) -> DigestVal {
        debug_assert!(mode.is_two_to_one());
        self.read(a.0);
        self.read(b.0);
        let out = self.alloc();
        self.instrs.push(Instr::Hash {
            mode,
            ins: [a.0, b.0, Addr(0)],
            outs: [out, Addr(0), Addr(0)],
            mults: [0, 0, 0],
        });
        DigestVal(out)
    }

    /// Full three-cell state permutation.
    pub fn permute(&mut self, state: [Cell; 3]) -> [Cell; 3] {
        for c in &state {
            self.read(c.0);
        }
        let outs = [self.alloc(), self.alloc(), self.alloc()];
        self.instrs.push(Instr::Hash {
            mode: HashMode::Permute,
            ins: [state[0].0, state[1].0, state[2].0],
            outs,
            mults: [0, 0, 0],
        });
        outs.map(Cell)
    }

    // ---- lane conversion (LFM_LANES) ----

    /// Split a word into its four lanes as base cells — the only route from
    /// a hash-state/digest cell into the ALU.
    pub fn unpack(&mut self, c: Cell) -> [Felt; 4] {
        self.read(c.0);
        let outs = [self.alloc(), self.alloc(), self.alloc(), self.alloc()];
        self.instrs.push(Instr::Unpack {
            input: c.0,
            outs,
            mults: [0; 4],
        });
        outs.map(Felt)
    }

    /// Assemble a word from four base cells.
    pub fn pack_word(&mut self, lanes: [Felt; 4]) -> Cell {
        for l in &lanes {
            self.read(l.0);
        }
        let out = self.alloc();
        self.instrs.push(Instr::Pack {
            lanes: lanes.map(|f| f.0),
            out,
            mult: 0,
        });
        Cell(out)
    }

    /// Assemble an ext cell `(a0, a1, a2, 0)` from three base cells (lane 3
    /// is the shared zero constant).
    pub fn pack_ext(&mut self, a0: Felt, a1: Felt, a2: Felt) -> Ext {
        let zero = self.felt_const(FE::zero());
        Ext(self.pack_word([a0, a1, a2, zero]).0)
    }

    // ---- keccak-f[1600] (LFM_KECCAK) ----

    /// One `keccak-f[1600]` permutation over 13 state words.
    ///
    /// The 25 `u64` lanes travel as 50 `u32` halves packed four to a word:
    /// word `j` carries halves `4j..4j+3`, half `h` is the low (`h` even) or
    /// high (`h` odd) 32 bits of lane `h / 2`. The last word's top two lanes
    /// are unused and must be zero — the bus pins them as tuple constants, and
    /// the executor errors on a nonzero one. Every lane of every input word
    /// must be a canonical value below `2^32`.
    pub fn keccak_f(&mut self, state: [Cell; layout::keccak::NUM_WORDS]) -> [Cell; 13] {
        self.emit_keccak(KeccakMode::Permute, state, [Cell(Addr(0)); 9], false)
            .0
    }

    /// One sponge absorb step: XOR a 136-byte rate block (9 words of `u32`
    /// halves, the top two half slots unused and zero) into the state's rate
    /// region, then permute.
    pub fn keccak_absorb(
        &mut self,
        state: [Cell; layout::keccak::NUM_WORDS],
        block: [Cell; layout::keccak::BLOCK_WORDS],
    ) -> [Cell; 13] {
        self.emit_keccak(KeccakMode::Absorb, state, block, false).0
    }

    /// Absorb, and additionally materialize the byte-REVERSED digest of the
    /// resulting state as two words — the production transcript's `sample()`,
    /// which both returns those bytes and re-absorbs them as the next segment's
    /// prefix. Free on the bus (see `layout::keccak::REV_ADDR0`).
    pub fn keccak_absorb_rev(
        &mut self,
        state: [Cell; layout::keccak::NUM_WORDS],
        block: [Cell; layout::keccak::BLOCK_WORDS],
    ) -> ([Cell; 13], [Cell; 2]) {
        let (outs, rev) = self.emit_keccak(KeccakMode::Absorb, state, block, true);
        (outs, rev.expect("requested"))
    }

    fn emit_keccak(
        &mut self,
        mode: KeccakMode,
        state: [Cell; layout::keccak::NUM_WORDS],
        block: [Cell; layout::keccak::BLOCK_WORDS],
        want_rev: bool,
    ) -> ([Cell; 13], Option<[Cell; 2]>) {
        for c in &state {
            self.read(c.0);
        }
        if mode == KeccakMode::Absorb {
            for c in &block {
                self.read(c.0);
            }
        }
        let outs: [Addr; 13] = core::array::from_fn(|_| self.alloc());
        let rev_outs: Option<[Addr; 2]> = want_rev.then(|| core::array::from_fn(|_| self.alloc()));
        self.instrs
            .push(Instr::KeccakF(Box::new(super::instr::KeccakOperands {
                mode,
                ins: state.map(|c| c.0),
                block: block.map(|c| c.0),
                outs,
                mults: [0; 13],
                rev: rev_outs.map(|outs| super::instr::KeccakReversedDigest {
                    outs,
                    mults: [0; 2],
                }),
            })));
        (outs.map(Cell), rev_outs.map(|r| r.map(Cell)))
    }

    // ---- hints / public ----

    pub fn declare_arena(&mut self, len: u32) -> ArenaId {
        self.arena_schema.lens.push(len);
        (self.arena_schema.lens.len() - 1) as ArenaId
    }

    /// One arena word → one memory cell. Arena values are unconstrained by
    /// the reading chip; the arena rule (transitively hash-authenticate
    /// everything hinted; never derive challenges from arenas) is what makes
    /// this sound.
    pub fn hint_word(&mut self, arena: ArenaId, index: u32) -> Cell {
        let out = self.alloc();
        self.instrs.push(Instr::Hint {
            arena,
            index,
            out,
            mult: 0,
        });
        Cell(out)
    }

    pub fn hint_felt(&mut self, arena: ArenaId, index: u32) -> Felt {
        Felt(self.hint_word(arena, index).0)
    }

    /// Expose a cell on the public-output bus (auto-incrementing index).
    pub fn public(&mut self, c: Cell) {
        self.read(c.0);
        let index = self.public_len;
        self.public_len += 1;
        self.instrs.push(Instr::Public { addr: c.0, index });
    }

    pub fn finish(self) -> LfmProgramSource {
        LfmProgramSource {
            instrs: self.instrs,
            num_addrs: self.next_addr,
            read_counts: self.read_counts,
            arena_schema: self.arena_schema,
            public_len: self.public_len,
        }
    }
}

//! `LFM_BLAKE3` — the BLAKE3 compression chip, hosted on the LFM bus.
//!
//! Ported from PR #903's `prover/src/tables/blake3.rs` (`yetanotherco/lambda_vm`,
//! head `89aeeb8c2b0389e9d21a861c9e3a10a7b1b5704e`), which is the syscall
//! variant: it takes its inputs and returns its outputs through the VM's memory,
//! so its I/O side is an `Ecall` receiver, an x10 register read and 22 `Memw`
//! dword ops over a 176-byte state region. **The mixing core is unchanged.**
//! What this module replaces is the I/O side, with `LfmMem` word tokens in the
//! discipline `chips::keccak` (`LFM_KECCAK`) established: addresses and
//! multiplicities are preprocessed program data, and a machine word carries
//! four `u32` lanes.
//!
//! # What the swap costs and buys, send for send
//!
//! | | #903 (syscall) | here (LFM) |
//! |---|---|---|
//! | `Ecall` receiver | 1 | — |
//! | `Memw` x10 register read | 1 | — |
//! | `Memw` per state dword | 22 | — |
//! | `LfmMem` word tokens | — | 7 reads + 4 writes + 2 reversed-digest = 13 |
//! | `ByteAlu[XOR]` mixing + feed-forward | 832 | 832 |
//! | `AreBytes` shift halfwords | 384 | 384 |
//! | `AreBytes` message bytes | 32 | 32 |
//! | `AreBytes` OLD_OUT bytes | 32 | — |
//! | `AreBytes` addr bytes + alignment `AND` | 5 | — |
//! | `IsHalfword` pointer halfwords | 88 | — |
//! | **total interactions** | **1,397** | **1,261** |
//! | value columns | 3,219 | 3,056 |
//!
//! (The two reversed-digest sends arrived with registration; the standalone
//! chip measured 1,259. They cost no value column — see
//! `layout::blake3::REV_ADDR0`.)
//!
//! The dropped columns are `TIMESTAMP` (2), `ADDR` (8), `PTR` (88) and
//! `OLD_OUT` (64) — 162 — and `MU` moves into the preprocessed prefix, which
//! the census excludes, for 163 in total.
//!
//! # Why dropping those range checks is sound, not just cheaper
//!
//! Each dropped lookup guarded something that no longer exists:
//!
//! - **`OLD_OUT`'s 32 `AreBytes`.** #903 needs them because the previous memory
//!   content of the out region appears only in the `Memw` write ops' `old`
//!   field — never XOR-consumed, so its packed linear combinations could alias.
//!   An `LfmMem` write carries no `old` field; there are no such columns here.
//! - **The address bytes, the alignment `AND` and the 88 pointer `IsHalfword`s.**
//!   #903's state address is prover witness read out of x10 and must be
//!   range-checked and shown 8-aligned before 22 pointers are derived from it.
//!   Here every address is a *preprocessed* column supplied by the program and
//!   vouched by the admission validator, exactly as for every other LFM chip —
//!   a prover cannot choose it at all.
//!
//! What is NOT dropped is the byte-range coverage of the data columns, and it
//! carries over intact:
//!
//! - all 64 `m` bytes keep their explicit `AreBytes` (they are never XORed);
//! - `h`'s 32 bytes are XOR operands of the feed-forward (`out[i+8] = v[i+8] ^ h[i]`);
//! - `t_lo`, `t_hi`, `block_len`, `flags` are `v[12..16]`, each the `vd` operand
//!   of a round-0 `G`, hence an operand of that `G`'s first XOR;
//! - all 64 `OUT` bytes are *results* of feed-forward XOR lookups.
//!
//! So every byte column reaching an `LfmMem` token is range-checked before the
//! token recomposes it, and a `u32` lane — four values below 2^8 with
//! coefficients 1, 2^8, 2^16, 2^24 — cannot reach 2^32. This is the same
//! transitive argument `chips::keccak` records for its 400 state bytes.
//!
//! # The single-dataflow rule, inherited
//!
//! The compression dataflow is written ONCE, in [`run_flow`], and interpreted
//! twice: [`WireFlow`] (columns — drives constraints and senders) and
//! [`ValueFlow`] (u32 witness — drives the trace and the BITWISE multiplicities).
//! The two cannot diverge on wiring, only on interpretation, which the probe's
//! bus-balance gate checks. That property is #903's and is worth preserving on
//! sight: it is why the sender list and the witness cannot drift apart.
//!
//! # Status
//!
//! This chip is **registered**: it is slot 11 of the fixed AIR set
//! (`airs::LFM_CHIP_NAMES`), reached from the eDSL through
//! [`super::builder::LfmBuilder::blake3_compress`]. `blake3_probe` still proves
//! it standalone against a synthetic `LfmMem` mirror, which is what prices the
//! chip on its own; `blake3_chip_tests` covers it as a member of the machine.
//!
//! ⚠ Round count follows [`super::blake3::BLAKE3_ROUNDS`]: 7 (standard BLAKE3)
//! by default, 6 under the `blake3-6round` feature. The 6-round instantiation
//! rests on the unratified security assumption **A6R**; the 7-round one carries
//! no assumption. Every column count in the table above is the 6-round one and
//! is quoted for continuity with #903 — `blake3_probe` pins both.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use crate::constraints::templates::{INV_SHIFT_32, emit_is_bit};
use crate::tables::bitwise::{BitwiseOperation, BitwiseOperationType};
use crate::tables::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, alu_op, zeroed_fe_vec,
};

use super::blake3::{BLAKE3_IV, BLAKE3_MSG_PERMUTATION, BLAKE3_ROUNDS, blake3_compress_rounds};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// G-instances per compression: 8 per round, at the compiled round count.
pub const NUM_G: usize = BLAKE3_ROUNDS * 8;

pub use super::layout::blake3::{IN_U32, IN_WORDS, OUT_U32, OUT_WORDS};

/// The (a, b, c, d) state indices of the 8 G-calls of one round:
/// 4 column mixes then 4 diagonal mixes (BLAKE3 spec §2.1).
const G_INDICES: [(usize, usize, usize, usize); 8] = [
    (0, 4, 8, 12),
    (1, 5, 9, 13),
    (2, 6, 10, 14),
    (3, 7, 11, 15),
    (0, 5, 10, 15),
    (1, 6, 11, 12),
    (2, 7, 8, 13),
    (3, 4, 9, 14),
];

/// Shift amounts of the two non-free rotations, as `rotl` inner shifts:
/// rotr12 = rotl20 = rotl16∘rotl4 (r=4); rotr7 = rotl25 = rotl16∘rotl9 (r=9).
pub(crate) const ROT_SHIFT_R: [u32; 2] = [4, 9];

// =========================================================================
// Column layout
// =========================================================================

/// The chip's columns: a preprocessed instruction group, then value columns.
///
/// The prefix is [`crate::lfm::layout::blake3`], re-exported rather than
/// restated — it is the chip's *instruction* column group and belongs beside
/// every other chip's, which is what the compiler and the admission validator
/// read. What stays here is the mixing core's value layout, which nothing
/// outside this chip and `blake3_socket`'s shared G-block offsets shares.
pub mod cols {
    pub use crate::lfm::layout::blake3::*;

    use super::NUM_G;

    // --- value columns ---
    //
    // ⚠ The trailing literals are ROUND-DEPENDENT: everything from `OUT` on is a
    // function of `NUM_G = 8 · BLAKE3_ROUNDS`, so each is given at both counts
    // rather than at whichever one the reader's build happens to compile.
    // `the_registry_blessing_is_round_count_invariant` pins both, and pins that
    // the PREPROCESSED prefix — which is what the registry commits — moves with
    // neither.
    /// Input bytes: `h[32] | m[64] | t_lo[4] | t_hi[4] | block_len[4] | flags[4]`.
    pub const IN: usize = PREP_WIDTH; // 20
    /// `NUM_G` G-blocks × 60 cells (56 bytes + 4 carry bits).
    pub const G: usize = IN + 4 * IN_U32; // 132
    pub const G_SIZE: usize = 60;
    /// Feed-forward output bytes `out[0..16]` (64 bytes).
    pub const OUT: usize = G + NUM_G * G_SIZE; // 3012 at 6 rounds, 3492 at 7

    pub const NUM_COLUMNS: usize = OUT + 4 * OUT_U32; // 3076 at 6 rounds, 3556 at 7

    /// Input word `i` (0..28: `h[0..8]`, `m[8..24]`, `t_lo=24`, `t_hi=25`,
    /// `block_len=26`, `flags=27`), byte `b`.
    #[inline]
    pub const fn in_word(i: usize, b: usize) -> usize {
        IN + i * 4 + b
    }

    /// Feed-forward output word `i` (0..16), byte `b`.
    #[inline]
    pub const fn out_word(i: usize, b: usize) -> usize {
        OUT + i * 4 + b
    }

    /// Base column of G-block `g`.
    #[inline]
    pub const fn g_base(g: usize) -> usize {
        G + g * G_SIZE
    }

    // Offsets inside one G block (56 byte cells + 4 carry bits = 60):
    /// add3 #1 output word (4 bytes).
    pub const G_A1: usize = 0;
    /// add3 #1 carry bits c1, c2.
    pub const G_A1_C: usize = 4;
    /// X1 = vd ^ A1 (4 bytes).
    pub const G_X1: usize = 6;
    /// add2 #1 output word (4 bytes).
    pub const G_C1: usize = 10;
    /// X2 = vb ^ C1 (4 bytes).
    pub const G_X2: usize = 14;
    /// rotr12 block: SLL_lo(2) SLLC_lo(2) SLL_hi(2) SLLC_hi(2) Y(4).
    pub const G_R1: usize = 18;
    /// add3 #2 output word (4 bytes).
    pub const G_A2: usize = 30;
    /// add3 #2 carry bits.
    pub const G_A2_C: usize = 34;
    /// X3 = vd ^ A2 (4 bytes).
    pub const G_X3: usize = 36;
    /// add2 #2 output word (4 bytes).
    pub const G_C2: usize = 40;
    /// X4 = B1 ^ C2 (4 bytes).
    pub const G_X4: usize = 44;
    /// rotr7 block: same layout as `G_R1`.
    pub const G_R2: usize = 48;
}

/// Value columns the census counts: everything past the preprocessed prefix.
pub const MAIN_COLUMNS: usize = cols::NUM_COLUMNS - cols::PREP_WIDTH;

// =========================================================================
// The single dataflow, interpreted twice (verbatim from #903)
// =========================================================================

/// The framing degrees of freedom [`run_flow`] itself decides, as opposed to
/// the ones an interpretation decides.
///
/// These two live here, and not in each `Blake3Flow` impl, for one reason: they
/// change *which calls happen*, so an impl that got them wrong would silently
/// desynchronise the wire interpretation from the value interpretation and the
/// bus sends would stop matching the multiplicities. Deciding them in the single
/// dataflow is what keeps the single-dataflow rule true when there is more than
/// one framing (`blake3_chip`'s syscall shape and `blake3_socket`'s 2-to-1
/// compress).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlowConfig {
    /// Rounds of 8 G-calls. 6 for this chip; [`super::blake3_socket`] sweeps.
    pub rounds: usize,
    /// How many of the eight `out[i] = v[i] ^ v[i+8]` words to produce — the
    /// truncation window. 8 here; 4 for the socket, whose digest is one cell.
    pub out_window: usize,
    /// Whether to produce `out[i+8] = v[i+8] ^ h[i]` as well. The socket does
    /// not: those words are not part of a truncated 128-bit digest, and never
    /// building them is where most of its saving over this chip comes from.
    pub full_output: bool,
}

impl FlowConfig {
    /// The syscall-shaped chip's framing: the full 16-word output.
    pub(crate) const fn full(rounds: usize) -> Self {
        Self {
            rounds,
            out_window: 8,
            full_output: true,
        }
    }
}

/// The BLAKE3 compression dataflow, abstracted over its word representation.
pub(crate) trait Blake3Flow {
    type Word: Copy;

    /// `h[i]` input word.
    fn input_h(&mut self, i: usize) -> Self::Word;
    /// `v[12..16]` init words: t_lo, t_hi, block_len, flags.
    fn input_v12(&mut self, j: usize) -> Self::Word;
    /// `IV[i]` constant (`v[8..12]`).
    fn iv_const(&mut self, i: usize) -> Self::Word;

    /// 3-operand add `s = a + b + m[m_idx] mod 2^32` (half 0/1 = which add3 of G g).
    fn add3(
        &mut self,
        g: usize,
        half: usize,
        a: Self::Word,
        b: Self::Word,
        m_idx: usize,
    ) -> Self::Word;
    /// 2-operand add `s = a + b mod 2^32`.
    fn add2(&mut self, g: usize, half: usize, a: Self::Word, b: Self::Word) -> Self::Word;
    /// XOR (slot 0..4 = X1..X4 of G g). Operand order is part of the wire format.
    fn xor(&mut self, g: usize, slot: usize, a: Self::Word, b: Self::Word) -> Self::Word;
    /// rotr16: free byte relabel `[b2,b3,b0,b1]`.
    fn rotr16(&mut self, w: Self::Word) -> Self::Word;
    /// rotr8: free byte relabel `[b1,b2,b3,b0]`.
    fn rotr8(&mut self, w: Self::Word) -> Self::Word;
    /// rotr12 (half=0) / rotr7 (half=1) via the inline shift identity.
    fn rot_shift(&mut self, g: usize, half: usize, w: Self::Word) -> Self::Word;
    /// Feed-forward, low half: `out[i] = v[i] ^ v[i+8]`.
    fn feed_forward_low(&mut self, i: usize, vi: Self::Word, vi8: Self::Word);
    /// Feed-forward, high half: `out[i+8] = v[i+8] ^ h[i]`. Called only under
    /// [`FlowConfig::full_output`].
    fn feed_forward_high(&mut self, i: usize, vi8: Self::Word, hi: Self::Word);
}

/// Drive the compression through `f`. The message schedule is tracked as
/// indices into the ORIGINAL m (permute^r composition), so both interpretations
/// reference original message words — never copies.
pub(crate) fn run_flow<T: Blake3Flow>(f: &mut T, cfg: FlowConfig) {
    let h: [T::Word; 8] = core::array::from_fn(|i| f.input_h(i));
    let mut v: [T::Word; 16] = core::array::from_fn(|i| {
        if i < 8 {
            h[i]
        } else if i < 12 {
            f.iv_const(i - 8)
        } else {
            f.input_v12(i - 12)
        }
    });

    // sched[i] = index into the original m of the word consumed at position i
    // this round. permute: m'[i] = m[P[i]] ⇒ sched'[i] = sched[P[i]].
    let mut sched: [usize; 16] = core::array::from_fn(|i| i);

    for r in 0..cfg.rounds {
        for (j, &(ia, ib, ic, id)) in G_INDICES.iter().enumerate() {
            let g = r * 8 + j;
            let (va, vb, vc, vd) = (v[ia], v[ib], v[ic], v[id]);
            let mx = sched[2 * j];
            let my = sched[2 * j + 1];

            let a1 = f.add3(g, 0, va, vb, mx);
            let x1 = f.xor(g, 0, vd, a1);
            let vd1 = f.rotr16(x1);
            let c1 = f.add2(g, 0, vc, vd1);
            let x2 = f.xor(g, 1, vb, c1);
            let b1 = f.rot_shift(g, 0, x2); // rotr12
            let a2 = f.add3(g, 1, a1, b1, my);
            let x3 = f.xor(g, 2, vd1, a2);
            let vd2 = f.rotr8(x3);
            let c2 = f.add2(g, 1, c1, vd2);
            let x4 = f.xor(g, 3, b1, c2);
            let b2 = f.rot_shift(g, 1, x4); // rotr7

            v[ia] = a2;
            v[ib] = b2;
            v[ic] = c2;
            v[id] = vd2;
        }
        if r < cfg.rounds - 1 {
            let prev = sched;
            for (i, &p) in BLAKE3_MSG_PERMUTATION.iter().enumerate() {
                sched[i] = prev[p];
            }
        }
    }

    for i in 0..cfg.out_window {
        f.feed_forward_low(i, v[i], v[i + 8]);
        if cfg.full_output {
            f.feed_forward_high(i, v[i + 8], h[i]);
        }
    }
}

// =========================================================================
// Wire interpretation (columns)
// =========================================================================

/// A 32-bit word as wiring: four byte columns (LSB first), a constant, or a
/// constant selected by preprocessed mode columns.
/// Constants only ever appear as the IV `v[c]` operands of round-0 add2s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WordRef {
    Cols([usize; 4]),
    Const(u32),
    /// `Σ_k col_k · tag_k` — a constant selected by PREPROCESSED mode columns.
    ///
    /// The socket's domain tag, for when one tag is no longer enough. It is a
    /// linear form over columns the prover cannot choose, so it is as unchosen
    /// as the plain `Const` it replaces, and it costs the same: zero witness
    /// columns, zero range checks, degree 1 where the constant was degree 0.
    /// `add3`'s body is degree 1 in its operands either way, so the arm's max
    /// degree does not move.
    ///
    /// It is a WHOLE-WORD form only — see [`WordRef::byte`].
    ModeSelected(&'static [(usize, u32)]),
}

impl WordRef {
    /// This word's byte `b`, for a byte-granular consumer (`ByteAlu[XOR]`).
    ///
    /// # Panics
    ///
    /// On [`WordRef::ModeSelected`]. A mode-selected word has no byte
    /// decomposition without witnessing one, and the whole reason the tag lives
    /// in a message word is that message words reach `add3` and nothing else. Panicking
    /// says so out loud rather than letting a future byte consumer quietly
    /// acquire four columns nobody committed.
    pub(crate) fn byte(self, b: usize) -> ByteRef {
        match self {
            WordRef::Cols(c) => ByteRef::Col(c[b]),
            WordRef::Const(w) => ByteRef::Const(((w >> (8 * b)) & 0xFF) as u8),
            WordRef::ModeSelected(_) => unreachable!(
                "a mode-selected word is a whole-word value: it reaches add3 and \
                 nothing byte-granular, so it has no byte columns to name"
            ),
        }
    }

    /// This word rotated right by `bytes` bytes — free wiring, no columns and
    /// no constraint: `Cols` permutes its byte columns and `Const` rotates its
    /// value. It is BLAKE3's `rotr16`/`rotr8`, whose shifts are byte-aligned.
    ///
    /// # Panics
    ///
    /// On [`WordRef::ModeSelected`], for the reason [`WordRef::byte`] gives:
    /// the rotation is a byte permutation, and a mode-selected word has no
    /// bytes to permute.
    pub(crate) fn rotr_bytes(self, bytes: usize) -> WordRef {
        match self {
            WordRef::Cols(c) => WordRef::Cols(core::array::from_fn(|j| c[(j + bytes) % 4])),
            WordRef::Const(v) => WordRef::Const(v.rotate_right(8 * bytes as u32)),
            WordRef::ModeSelected(_) => unreachable!(
                "a mode-selected word is a whole-word value: it has no byte \
                 columns to rotate"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ByteRef {
    Col(usize),
    Const(u8),
}

/// One recorded 3-op add: operands (a, b, m), output columns, carries.
///
/// `m` is a [`WordRef`] rather than four columns because the socket framing
/// makes every message word above the input lanes a compile-time constant — the
/// domain tag and the zero padding of a 52-byte message. Constant message words
/// cost no columns and no
/// range checks, which is the whole reason the tag is free there.
pub(crate) struct Add3Wire {
    pub a: WordRef,
    pub b: WordRef,
    pub m: WordRef,
    pub s: [usize; 4],
    pub c1: usize,
    pub c2: usize,
}

/// One recorded 2-op add: operands, output columns (carry is an expression).
pub(crate) struct Add2Wire {
    pub a: WordRef,
    pub b: WordRef,
    pub s: [usize; 4],
}

/// One recorded XOR: per-byte operands and output columns.
pub(crate) struct XorWire {
    pub a: WordRef,
    pub b: WordRef,
    pub out: [usize; 4],
}

/// One recorded shift rotation: input word, the 8 shift-halfword byte columns
/// (SLL_lo, SLLC_lo, SLL_hi, SLLC_hi — 2 bytes each), output columns, r.
pub(crate) struct RotWire {
    pub input: WordRef,
    pub sll_lo: [usize; 2],
    pub sllc_lo: [usize; 2],
    pub sll_hi: [usize; 2],
    pub sllc_hi: [usize; 2],
    pub y: [usize; 4],
    pub r: u32,
}

/// The full wiring of one compression row, recorded in canonical order.
pub(crate) struct WireFlow {
    pub add3s: Vec<Add3Wire>,
    pub add2s: Vec<Add2Wire>,
    pub xors: Vec<XorWire>,
    pub rots: Vec<RotWire>,
}

impl WireFlow {
    pub(crate) fn build() -> Self {
        let mut w = WireFlow {
            add3s: Vec::with_capacity(NUM_G * 2),
            add2s: Vec::with_capacity(NUM_G * 2),
            xors: Vec::with_capacity(NUM_G * 4 + 16),
            rots: Vec::with_capacity(NUM_G * 2),
        };
        run_flow(&mut w, FlowConfig::full(BLAKE3_ROUNDS));
        w
    }
}

#[inline]
pub(crate) fn word_cols(start: usize) -> [usize; 4] {
    [start, start + 1, start + 2, start + 3]
}

impl Blake3Flow for WireFlow {
    type Word = WordRef;

    fn input_h(&mut self, i: usize) -> WordRef {
        WordRef::Cols(word_cols(cols::in_word(i, 0)))
    }
    fn input_v12(&mut self, j: usize) -> WordRef {
        WordRef::Cols(word_cols(cols::in_word(24 + j, 0)))
    }
    fn iv_const(&mut self, i: usize) -> WordRef {
        WordRef::Const(BLAKE3_IV[i])
    }

    fn add3(&mut self, g: usize, half: usize, a: WordRef, b: WordRef, m_idx: usize) -> WordRef {
        let base = cols::g_base(g) + if half == 0 { cols::G_A1 } else { cols::G_A2 };
        let cbase = cols::g_base(g)
            + if half == 0 {
                cols::G_A1_C
            } else {
                cols::G_A2_C
            };
        let s = word_cols(base);
        self.add3s.push(Add3Wire {
            a,
            b,
            m: WordRef::Cols(word_cols(cols::in_word(8 + m_idx, 0))),
            s,
            c1: cbase,
            c2: cbase + 1,
        });
        WordRef::Cols(s)
    }

    fn add2(&mut self, g: usize, half: usize, a: WordRef, b: WordRef) -> WordRef {
        let base = cols::g_base(g) + if half == 0 { cols::G_C1 } else { cols::G_C2 };
        let s = word_cols(base);
        self.add2s.push(Add2Wire { a, b, s });
        WordRef::Cols(s)
    }

    fn xor(&mut self, g: usize, slot: usize, a: WordRef, b: WordRef) -> WordRef {
        let off = match slot {
            0 => cols::G_X1,
            1 => cols::G_X2,
            2 => cols::G_X3,
            _ => cols::G_X4,
        };
        let out = word_cols(cols::g_base(g) + off);
        self.xors.push(XorWire { a, b, out });
        WordRef::Cols(out)
    }

    fn rotr16(&mut self, w: WordRef) -> WordRef {
        w.rotr_bytes(2)
    }
    fn rotr8(&mut self, w: WordRef) -> WordRef {
        w.rotr_bytes(1)
    }

    fn rot_shift(&mut self, g: usize, half: usize, w: WordRef) -> WordRef {
        let base = cols::g_base(g) + if half == 0 { cols::G_R1 } else { cols::G_R2 };
        let y = word_cols(base + 8);
        self.rots.push(RotWire {
            input: w,
            sll_lo: [base, base + 1],
            sllc_lo: [base + 2, base + 3],
            sll_hi: [base + 4, base + 5],
            sllc_hi: [base + 6, base + 7],
            y,
            r: ROT_SHIFT_R[half],
        });
        WordRef::Cols(y)
    }

    fn feed_forward_low(&mut self, i: usize, vi: WordRef, vi8: WordRef) {
        self.xors.push(XorWire {
            a: vi,
            b: vi8,
            out: word_cols(cols::out_word(i, 0)),
        });
    }

    fn feed_forward_high(&mut self, i: usize, vi8: WordRef, hi: WordRef) {
        self.xors.push(XorWire {
            a: vi8,
            b: hi,
            out: word_cols(cols::out_word(i + 8, 0)),
        });
    }
}

// =========================================================================
// Value interpretation (u32 witness)
// =========================================================================

/// Everything the trace filler and the BITWISE collector need for one
/// compression, recorded cell-exactly in the same canonical order as
/// [`WireFlow`].
pub struct ValueFlow {
    /// (s, c1, c2) per add3, canonical order.
    pub add3s: Vec<(u32, u8, u8)>,
    /// s per add2 (the carry is an expression, not a cell).
    pub add2s: Vec<u32>,
    /// (a, b, out) per XOR word, canonical order (Gs then feed-forward).
    pub xors: Vec<(u32, u32, u32)>,
    /// (sll_lo, sllc_lo, sll_hi, sllc_hi, y) per shift rotation.
    pub rots: Vec<(u16, u16, u16, u16, u32)>,
    /// The output words. Entries outside the framing's truncation window are
    /// never computed and stay zero — reading one is a caller bug.
    pub out: [u32; 16],

    h: [u32; 8],
    m: [u32; 16],
    v12: [u32; 4],
}

impl ValueFlow {
    /// The syscall-shaped chip's full 16-word compression.
    pub fn compute(h: &[u32; 8], m: &[u32; 16], t: u64, block_len: u32, flags: u32) -> Self {
        Self::compute_with(h, m, t, block_len, flags, FlowConfig::full(BLAKE3_ROUNDS))
    }

    /// [`ValueFlow::compute`] under an explicit framing.
    pub(crate) fn compute_with(
        h: &[u32; 8],
        m: &[u32; 16],
        t: u64,
        block_len: u32,
        flags: u32,
        cfg: FlowConfig,
    ) -> Self {
        let g = cfg.rounds * 8;
        let mut f = ValueFlow {
            add3s: Vec::with_capacity(g * 2),
            add2s: Vec::with_capacity(g * 2),
            xors: Vec::with_capacity(g * 4 + 16),
            rots: Vec::with_capacity(g * 2),
            out: [0; 16],
            h: *h,
            m: *m,
            v12: [t as u32, (t >> 32) as u32, block_len, flags],
        };
        run_flow(&mut f, cfg);
        f
    }
}

impl Blake3Flow for ValueFlow {
    type Word = u32;

    fn input_h(&mut self, i: usize) -> u32 {
        self.h[i]
    }
    fn input_v12(&mut self, j: usize) -> u32 {
        self.v12[j]
    }
    fn iv_const(&mut self, i: usize) -> u32 {
        BLAKE3_IV[i]
    }

    fn add3(&mut self, _g: usize, _half: usize, a: u32, b: u32, m_idx: usize) -> u32 {
        let m = self.m[m_idx];
        let wide = a as u64 + b as u64 + m as u64;
        let s = wide as u32;
        let carry = (wide >> 32) as u8; // 0, 1 or 2
        // Two summed carry bits: c1 + c2 = carry.
        let (c1, c2) = match carry {
            0 => (0, 0),
            1 => (1, 0),
            _ => (1, 1),
        };
        self.add3s.push((s, c1, c2));
        s
    }

    fn add2(&mut self, _g: usize, _half: usize, a: u32, b: u32) -> u32 {
        let s = a.wrapping_add(b);
        self.add2s.push(s);
        s
    }

    fn xor(&mut self, _g: usize, _slot: usize, a: u32, b: u32) -> u32 {
        let out = a ^ b;
        self.xors.push((a, b, out));
        out
    }

    fn rotr16(&mut self, w: u32) -> u32 {
        w.rotate_right(16)
    }
    fn rotr8(&mut self, w: u32) -> u32 {
        w.rotate_right(8)
    }

    fn rot_shift(&mut self, _g: usize, half: usize, w: u32) -> u32 {
        let r = ROT_SHIFT_R[half];
        let xlo = w & 0xFFFF;
        let xhi = w >> 16;
        // xlo·2^r = SLLC_lo·2^16 + SLL_lo (and same for hi): Euclidean split.
        let sll_lo = ((xlo << r) & 0xFFFF) as u16;
        let sllc_lo = ((xlo << r) >> 16) as u16;
        let sll_hi = ((xhi << r) & 0xFFFF) as u16;
        let sllc_hi = ((xhi << r) >> 16) as u16;
        // Recombine + halfword swap: Ylo = SLL_hi + SLLC_lo, Yhi = SLL_lo + SLLC_hi.
        let ylo = sll_hi as u32 + sllc_lo as u32;
        let yhi = sll_lo as u32 + sllc_hi as u32;
        let y = ylo | (yhi << 16);
        debug_assert_eq!(y, w.rotate_right(if r == 4 { 12 } else { 7 }));
        self.rots.push((sll_lo, sllc_lo, sll_hi, sllc_hi, y));
        y
    }

    fn feed_forward_low(&mut self, i: usize, vi: u32, vi8: u32) {
        let lo = vi ^ vi8;
        self.xors.push((vi, vi8, lo));
        self.out[i] = lo;
    }

    fn feed_forward_high(&mut self, i: usize, vi8: u32, hi: u32) {
        let w = vi8 ^ hi;
        self.xors.push((vi8, hi, w));
        self.out[i + 8] = w;
    }
}

// =========================================================================
// Operation struct + trace generation
// =========================================================================

/// The VALUE half of one compression — everything the mixing core, the trace's
/// value columns and the BITWISE multiplicities depend on, and nothing else.
///
/// This is what the machine's executor records, because the other half —
/// addresses and multiplicities — is *program* data living in the preprocessed
/// instruction group, exactly as for every other LFM chip. Keeping the split in
/// the type is what stops a witness from describing a compression at addresses
/// the committed program does not name.
#[derive(Debug, Clone, Copy)]
pub struct Blake3Values {
    pub h: [u32; 8],
    pub m: [u32; 16],
    pub t: u64,
    pub block_len: u32,
    pub flags: u32,
}

impl Blake3Values {
    /// The 28 input `u32` words in machine order: `h | m | t_lo | t_hi | len | flags`.
    pub fn input_words(&self) -> [u32; IN_U32] {
        let mut w = [0u32; IN_U32];
        w[0..8].copy_from_slice(&self.h);
        w[8..24].copy_from_slice(&self.m);
        w[24] = self.t as u32;
        w[25] = (self.t >> 32) as u32;
        w[26] = self.block_len;
        w[27] = self.flags;
        w
    }

    /// The compression output.
    pub fn output_words(&self) -> [u32; OUT_U32] {
        blake3_compress_rounds(
            &self.h,
            &self.m,
            self.t,
            self.block_len,
            self.flags,
            BLAKE3_ROUNDS,
        )
    }
}

/// One compression, as the standalone probe issues it: [`Blake3Values`] plus
/// the addresses and multiplicities a program would have supplied.
///
/// The machine does not use this — it reads the prefix out of the committed
/// column group. It exists so `blake3_probe` can drive the chip against a
/// synthetic `LfmMem` mirror with no program behind it.
#[derive(Debug, Clone)]
pub struct Blake3Operation {
    pub in_addr: [u64; IN_WORDS],
    pub out_addr: [u64; OUT_WORDS],
    pub read_counts: [u64; OUT_WORDS],
    pub values: Blake3Values,
}

/// Write a 32-bit word as 4 byte cells at `row[col..col+4]`.
#[inline]
fn set_word_bytes(row: &mut [FE], col: usize, w: u32) {
    for b in 0..4 {
        row[col + b] = FE::from(u64::from((w >> (8 * b)) & 0xFF));
    }
}

/// ★ The chip's value columns for one compression, written into a row slice.
///
/// The single definition, shared by the standalone probe's trace and by the
/// machine's ([`super::trace::build_traces_with_hasher`]). Registration made two
/// callers out of one, and a second transcription of this — 3,056 columns laid
/// out in a fixed order — is exactly the drift the single-dataflow rule above
/// exists to prevent: the constraints and the senders are generated from
/// [`WireFlow`], so a filler that disagreed with them would fail only as an
/// unbalanced bus at prove time, with no signature saying why.
///
/// Writes nothing before `cols::IN`: the preprocessed prefix is the caller's,
/// and in the machine it is copied verbatim from the committed group.
pub fn fill_blake3_witness(row: &mut [FE], v: &Blake3Values) {
    for (i, &w) in v.input_words().iter().enumerate() {
        set_word_bytes(row, cols::in_word(i, 0), w);
    }

    // The mixing core, cell-exactly in canonical order.
    let flow = ValueFlow::compute(&v.h, &v.m, v.t, v.block_len, v.flags);
    let mut a3 = flow.add3s.iter();
    let mut a2 = flow.add2s.iter();
    let mut xo = flow.xors.iter();
    let mut ro = flow.rots.iter();
    for g in 0..NUM_G {
        let base = cols::g_base(g);
        for half in 0..2 {
            let (s_off, c_off, x_off, c2_off, x2_off, r_off) = if half == 0 {
                (
                    cols::G_A1,
                    cols::G_A1_C,
                    cols::G_X1,
                    cols::G_C1,
                    cols::G_X2,
                    cols::G_R1,
                )
            } else {
                (
                    cols::G_A2,
                    cols::G_A2_C,
                    cols::G_X3,
                    cols::G_C2,
                    cols::G_X4,
                    cols::G_R2,
                )
            };
            let &(s, c1, c2) = a3.next().expect("add3 count");
            set_word_bytes(row, base + s_off, s);
            row[base + c_off] = FE::from(u64::from(c1));
            row[base + c_off + 1] = FE::from(u64::from(c2));

            let &(_, _, x) = xo.next().expect("xor count");
            set_word_bytes(row, base + x_off, x);

            let &c = a2.next().expect("add2 count");
            set_word_bytes(row, base + c2_off, c);

            let &(_, _, x2) = xo.next().expect("xor count");
            set_word_bytes(row, base + x2_off, x2);

            let &(sll_lo, sllc_lo, sll_hi, sllc_hi, y) = ro.next().expect("rot count");
            row[base + r_off] = FE::from(u64::from(sll_lo & 0xFF));
            row[base + r_off + 1] = FE::from(u64::from(sll_lo >> 8));
            row[base + r_off + 2] = FE::from(u64::from(sllc_lo & 0xFF));
            row[base + r_off + 3] = FE::from(u64::from(sllc_lo >> 8));
            row[base + r_off + 4] = FE::from(u64::from(sll_hi & 0xFF));
            row[base + r_off + 5] = FE::from(u64::from(sll_hi >> 8));
            row[base + r_off + 6] = FE::from(u64::from(sllc_hi & 0xFF));
            row[base + r_off + 7] = FE::from(u64::from(sllc_hi >> 8));
            set_word_bytes(row, base + r_off + 8, y);
        }
    }
    for i in 0..OUT_U32 {
        set_word_bytes(row, cols::out_word(i, 0), flow.out[i]);
    }
}

/// One row per compression; padding rows are ALL ZERO.
///
/// #903 needs a nonzero pad (`ptr[k] = 8k`) because its pointer columns carry an
/// ungated `addr + 8k` identity. Nothing here is ungated except `IS_BIT(MU)`,
/// which a zero row satisfies, so the pad is genuinely empty — and
/// `padding_rows_are_all_zero` in `blake3_probe` pins that rather than assuming
/// it.
///
/// The probe's trace only: the machine builds the same rows through
/// `trace::chip_trace`, which copies the preprocessed prefix out of the
/// committed group instead of re-deriving it from an op.
pub fn generate_blake3_trace(ops: &[Blake3Operation]) -> TraceTable<F, E> {
    let num_rows = ops.len().next_power_of_two().max(4);
    let mut data = zeroed_fe_vec(num_rows * cols::NUM_COLUMNS);

    for (row, op) in ops.iter().enumerate() {
        let r = &mut data[row * cols::NUM_COLUMNS..(row + 1) * cols::NUM_COLUMNS];
        for j in 0..IN_WORDS {
            r[cols::in_addr(j)] = FE::from(op.in_addr[j]);
        }
        for j in 0..OUT_WORDS {
            r[cols::out_addr(j)] = FE::from(op.out_addr[j]);
            r[cols::mult(j)] = FE::from(op.read_counts[j]);
        }
        r[cols::MU] = FE::one();
        fill_blake3_witness(r, &op.values);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

fn direct(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

/// `u32` word `index` of the byte family at `bytes_start`, recomposed from its
/// four byte columns as `Σ byte_k · 256^k`.
///
/// The same trick `chips::keccak::half_value` uses: the machine-side `u32`
/// never gets its own column, so there is nothing extra to keep consistent, and
/// the bytes are already range-checked by the lookups that consume them.
fn lane_value(bytes_start: usize, index: usize) -> BusValue {
    BusValue::Linear(
        (0..4)
            .map(|k| LinearTerm::ColumnUnsigned {
                coefficient: 1u64 << (8 * k),
                column: bytes_start + index * 4 + k,
            })
            .collect(),
    )
}

/// An `LfmMem` token `(addr, v0..v3)` for machine word `word` of a byte family.
fn word_token(addr_col: usize, bytes_start: usize, word: usize) -> Vec<BusValue> {
    let mut v = vec![direct(addr_col)];
    v.extend((0..4).map(|l| lane_value(bytes_start, 4 * word + l)));
    v
}

/// Lane `l` of the byte-REVERSED digest: reversed byte `j` is digest byte
/// `31 − j`, so this lane's bytes are `OUT[31 − 4l − k]` for `k = 0..3` with the
/// usual little-endian coefficients.
///
/// The digest is `out[0..8]` little-endian, so digest byte `j` is column
/// `cols::OUT + j` — the same identity `chips::keccak::reversed_half_value`
/// relies on, over a different chip's OUT block. Both the byte order WITHIN a
/// lane and the order OF the lanes come out reversed, which is exactly what
/// reversing all 32 bytes means.
fn reversed_lane_value(lane: usize) -> BusValue {
    BusValue::Linear(
        (0..4)
            .map(|k| LinearTerm::ColumnUnsigned {
                coefficient: 1u64 << (8 * k),
                column: cols::OUT + 31 - 4 * lane - k,
            })
            .collect(),
    )
}

/// An `LfmMem` token for word `w` of the reversed digest.
fn reversed_digest_token(addr_col: usize, w: usize) -> Vec<BusValue> {
    let mut v = vec![direct(addr_col)];
    v.extend((0..4).map(|l| reversed_lane_value(4 * w + l)));
    v
}

/// Order groups: the `LfmMem` reads and writes, then the mixing core's ByteAlu
/// XORs (canonical `WireFlow` order), the shift `AreBytes`, and the message
/// `AreBytes`.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let wires = WireFlow::build();
    let mut interactions = Vec::with_capacity(1_261);

    let byte_bus_value = |b: ByteRef| -> BusValue {
        match b {
            ByteRef::Col(c) => direct(c),
            ByteRef::Const(v) => BusValue::constant(v as u64),
        }
    };

    // 1. Reads: the 7 input machine words.
    for j in 0..IN_WORDS {
        interactions.push(BusInteraction::receiver(
            BusId::LfmMem,
            Multiplicity::Column(cols::MU),
            word_token(cols::in_addr(j), cols::IN, j),
        ));
    }
    // 2. Writes: the 4 output machine words, each with its own read count.
    for j in 0..OUT_WORDS {
        interactions.push(BusInteraction::sender(
            BusId::LfmMem,
            Multiplicity::Column(cols::mult(j)),
            word_token(cols::out_addr(j), cols::OUT, j),
        ));
    }
    // 2b. The reversed digest: the 32 digest bytes read back-to-front, as two
    // more words. Free — a different `Linear` over the OUT columns already
    // present. See `layout::blake3::REV_ADDR0` for why the transcript needs it.
    for w in 0..cols::DIGEST_WORDS {
        interactions.push(BusInteraction::sender(
            BusId::LfmMem,
            Multiplicity::Column(cols::rev_mult(w)),
            reversed_digest_token(cols::rev_addr(w), w),
        ));
    }

    // 3. Mixing core + feed-forward: ByteAlu[XOR] per byte, canonical order.
    for xw in &wires.xors {
        for b in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::ByteAlu,
                Multiplicity::Column(cols::MU),
                vec![
                    BusValue::constant(alu_op::XOR as u64),
                    byte_bus_value(xw.a.byte(b)),
                    byte_bus_value(xw.b.byte(b)),
                    direct(xw.out[b]),
                ],
            ));
        }
    }

    // 4. Shift-halfword AreBytes: 4 pairs per rotation.
    for rw in &wires.rots {
        for pair in [rw.sll_lo, rw.sllc_lo, rw.sll_hi, rw.sllc_hi] {
            interactions.push(BusInteraction::sender(
                BusId::AreBytes,
                Multiplicity::Column(cols::MU),
                vec![direct(pair[0]), direct(pair[1])],
            ));
        }
    }

    // 5. Message AreBytes: m is never XORed, so its 64 bytes get no transitive
    // range check (#903 DESIGN §4.7/§7.5). 32 pairs.
    for i in 0..16 {
        for p in 0..2 {
            interactions.push(BusInteraction::sender(
                BusId::AreBytes,
                Multiplicity::Column(cols::MU),
                vec![
                    direct(cols::in_word(8 + i, 2 * p)),
                    direct(cols::in_word(8 + i, 2 * p + 1)),
                ],
            ));
        }
    }

    interactions
}

/// The BITWISE lookups `bus_interactions` sends, mirrored send for send.
///
/// Forked from #903's `collect_bitwise_from_blake3` with the address-shaped
/// lookups (the alignment `AND`, 4 addr `AreBytes`, 88 pointer `IsHalf`) and
/// the 32 `OLD_OUT` `AreBytes` dropped — the columns they guarded do not exist
/// here. Enumeration order is the senders' own, via the shared `ValueFlow`.
pub fn bitwise_ops_for(ops: &[Blake3Values]) -> Vec<BitwiseOperation> {
    let mut out = Vec::with_capacity(ops.len() * 1_248);

    for op in ops {
        let flow = ValueFlow::compute(&op.h, &op.m, op.t, op.block_len, op.flags);
        for &(a, b, _out) in &flow.xors {
            for byte in 0..4 {
                out.push(BitwiseOperation::byte_op(
                    BitwiseOperationType::ByteAluXor,
                    ((a >> (8 * byte)) & 0xFF) as u8,
                    ((b >> (8 * byte)) & 0xFF) as u8,
                ));
            }
        }
        for &(sll_lo, sllc_lo, sll_hi, sllc_hi, _y) in &flow.rots {
            for hw in [sll_lo, sllc_lo, sll_hi, sllc_hi] {
                out.push(BitwiseOperation::byte_op(
                    BitwiseOperationType::AreBytes,
                    (hw & 0xFF) as u8,
                    (hw >> 8) as u8,
                ));
            }
        }
        for &word in &op.m {
            for p in 0..2 {
                out.push(BitwiseOperation::byte_op(
                    BitwiseOperationType::AreBytes,
                    ((word >> (16 * p)) & 0xFF) as u8,
                    ((word >> (16 * p + 8)) & 0xFF) as u8,
                ));
            }
        }
    }

    out
}

// =========================================================================
// Constraints
// =========================================================================

/// Word expression from a [`WordRef`]: `b0 + 256·b1 + 2^16·b2 + 2^24·b3`.
pub(crate) fn word_expr<B: ConstraintBuilder<F, E>>(b: &B, w: &WordRef) -> B::Expr {
    match w {
        WordRef::Cols(c) => {
            b.main(0, c[0])
                + b.main(0, c[1]) * b.const_base(256)
                + b.main(0, c[2]) * b.const_base(65536)
                + b.main(0, c[3]) * b.const_base(16777216)
        }
        WordRef::Const(v) => b.const_base(*v as u64),
        WordRef::ModeSelected(terms) => {
            let mut iter = terms.iter();
            let term = |b: &B, (col, tag): (usize, u32)| b.main(0, col) * b.const_base(tag as u64);
            let &first = iter.next().expect("a mode-selected word selects something");
            iter.fold(term(b, first), |acc, &t| acc + term(b, t))
        }
    }
}

/// Halfword expression from 2 byte columns: `b0 + 256·b1`.
pub(crate) fn half_expr<B: ConstraintBuilder<F, E>>(b: &B, c: &[usize; 2]) -> B::Expr {
    b.main(0, c[0]) + b.main(0, c[1]) * b.const_base(256)
}

/// The hosted chip's 769 transition constraints:
/// - idx 0..288:    96 add3 groups (sum identity + 2 carry booleanities);
/// - idx 288..384:  96 add2 expression-carry booleanities;
/// - idx 384..768:  96 rotations (2 shift identities + 2 recombine each);
/// - idx 768:       `IS_BIT(MU)`, ungated.
///
/// #903's first 45 constraints — the 22 `ptr[k] = addr + 8k` carry pairs and
/// the top-dword no-overflow check — have no counterpart: addresses here are
/// preprocessed, so there is nothing to derive and nothing a prover chooses.
///
/// All μ-gated, max degree 3 (the booleanities; identities are degree 2).
#[derive(Clone, Copy)]
pub struct Blake3LfmConstraints;

impl ConstraintSet<F, E> for Blake3LfmConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let wires = WireFlow::build();
        let mu = |b: &B| b.main(0, cols::MU);
        let mut idx = 0usize;

        let two_32 = b.const_base(1u64 << 32);
        let inv_2_32 = b.const_base(INV_SHIFT_32);

        // add3: μ·(a + b + m − s − 2^32·(c1+c2)) = 0; μ·ci·(1−ci) = 0.
        for aw in &wires.add3s {
            let a = word_expr(b, &aw.a);
            let bb = word_expr(b, &aw.b);
            let m_w = word_expr(b, &aw.m);
            let s = word_expr(b, &WordRef::Cols(aw.s));
            let c1 = b.main(0, aw.c1);
            let c2 = b.main(0, aw.c2);
            let sum_id = a + bb + m_w - s - (c1.clone() + c2.clone()) * two_32.clone();
            let m = mu(b);
            b.emit_base(idx, m * sum_id);
            idx += 1;
            let one = b.one();
            let m = mu(b);
            b.emit_base(idx, m * c1.clone() * (one - c1));
            idx += 1;
            let one = b.one();
            let m = mu(b);
            b.emit_base(idx, m * c2.clone() * (one - c2));
            idx += 1;
        }

        // add2: carry = (a + b − s)·2^−32; μ·carry·(1−carry) = 0.
        for aw in &wires.add2s {
            let a = word_expr(b, &aw.a);
            let bb = word_expr(b, &aw.b);
            let s = word_expr(b, &WordRef::Cols(aw.s));
            let carry = (a + bb - s) * inv_2_32.clone();
            let one = b.one();
            let m = mu(b);
            b.emit_base(idx, m * carry.clone() * (one - carry));
            idx += 1;
        }

        // Rotations: 2 shift identities + 2 recombine identities each.
        for rw in &wires.rots {
            let (xlo, xhi) = match &rw.input {
                WordRef::Cols(c) => (half_expr(b, &[c[0], c[1]]), half_expr(b, &[c[2], c[3]])),
                WordRef::Const(_) | WordRef::ModeSelected(_) => {
                    unreachable!("shift inputs are always committed XOR outputs")
                }
            };
            let sll_lo = half_expr(b, &rw.sll_lo);
            let sllc_lo = half_expr(b, &rw.sllc_lo);
            let sll_hi = half_expr(b, &rw.sll_hi);
            let sllc_hi = half_expr(b, &rw.sllc_hi);
            let ylo = half_expr(b, &[rw.y[0], rw.y[1]]);
            let yhi = half_expr(b, &[rw.y[2], rw.y[3]]);
            let two_r = b.const_base(1u64 << rw.r);
            let two_16 = b.const_base(65536);

            // μ·(xlo·2^r − SLLC_lo·2^16 − SLL_lo) = 0 (and hi).
            let m = mu(b);
            b.emit_base(
                idx,
                m * (xlo * two_r.clone() - sllc_lo.clone() * two_16.clone() - sll_lo.clone()),
            );
            idx += 1;
            let m = mu(b);
            b.emit_base(
                idx,
                m * (xhi * two_r - sllc_hi.clone() * two_16 - sll_hi.clone()),
            );
            idx += 1;
            // μ·(Ylo − SLL_hi − SLLC_lo) = 0; μ·(Yhi − SLL_lo − SLLC_hi) = 0.
            let m = mu(b);
            b.emit_base(idx, m * (ylo - sll_hi - sllc_lo));
            idx += 1;
            let m = mu(b);
            b.emit_base(idx, m * (yhi - sll_lo - sllc_hi));
            idx += 1;
        }

        // Ungated booleanity of the is-real flag. Preprocessed, so the
        // registrar already vouches for it; kept because `chips::keccak` keeps
        // its mode-sum booleanity for the same belt-over-suspenders reason.
        emit_is_bit(b, idx, cols::MU, None);
    }
}

/// Constraints the chip emits — the number the degree/count tests pin.
pub const NUM_CONSTRAINTS: usize = 3 * (NUM_G * 2) + (NUM_G * 2) + 4 * (NUM_G * 2) + 1;

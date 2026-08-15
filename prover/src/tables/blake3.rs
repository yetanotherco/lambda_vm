//! BLAKE3 6-round compression accelerator chip (syscall variant).
//!
//! One row per compression call, fully unrolled (Layout B of
//! `thoughts/blake3/blake3-chip/DESIGN.md`): all 6 rounds × 8 G-functions are
//! laid out in SSA form across the row, so the message schedule is a
//! compile-time permutation of the 16 committed message words and there is no
//! state/message handoff between rows.
//!
//! I/O follows the KECCAK core idiom (`keccak.rs`): an `Ecall` receiver binds
//! (timestamp, syscall#), a `Memw` register read binds the x10 state pointer,
//! and per-dword `Memw` ops read the 112 input bytes / write the 64 output
//! bytes. The 176-byte state region layout is documented on
//! [`executor::vm::instruction::execution::BLAKE3_SYSCALL_NUMBER`].
//!
//! ## The single-dataflow rule
//!
//! The compression dataflow is written ONCE, in [`run_flow`], and interpreted
//! twice: [`WireFlow`] (columns — drives the constraints and bus senders) and
//! [`ValueFlow`] (u32 witness — drives trace filling and the BITWISE
//! multiplicity collection in `trace_builder.rs`). The two cannot diverge on
//! wiring, only on interpretation, which the e2e bus-balance gate checks.
//!
//! ## Soundness ledger (DESIGN.md §7, adapted to the syscall variant)
//!
//! 1. Every eval constraint is μ-gated; padding rows are all-zero (except the
//!    keccak-style PTR pad) with μ=0.
//! 2. 3-op adds use TWO summed committed carry bits + the explicit sum
//!    identity (a ternary carry would be degree 4 after gating).
//! 3. 2-op adds use the `emit_add_pair`-style expression carry (no committed
//!    cell) with μ-gated booleanity; the output's bytes are range-checked by
//!    the downstream XOR lookup that consumes them.
//! 4. Every add/shift output feeds a downstream `ByteAlu` XOR — that lookup is
//!    its only byte range check. The last-round outputs are consumed by the
//!    feed-forward XORs, closing the chain.
//! 5. The message words `m` are never XORed, so their 64 bytes get explicit
//!    `AreBytes` sends. Same for the 64 `OLD_OUT` bytes (the previous memory
//!    content of the out region, which appear only on the Memw bus) and the 8
//!    address bytes (aliasing — see keccak.rs's addr comment).
//! 6. rotr16/rotr8 are free byte relabels `[b2,b3,b0,b1]` / `[b1,b2,b3,b0]`.
//! 7. rotr12/rotr7 are inline μ-gated shift identities with `AreBytes` on all
//!    four shift halfwords (`SLL_lo/SLLC_lo/SLL_hi/SLLC_hi`); soundness needs
//!    the tight bound on the `SLL` pair (2^16 invertible mod p — the audited
//!    Euclidean-division argument).
//! 8. The message schedule is `permute^r` wired from the ORIGINAL M columns.
//! 9. All identities stay < 2^35 ≪ p (non-overflow side conditions), given
//!    byte-range operands and boolean carries.
//! 10. (Internal-bus binding — N/A here: the syscall variant has no `Blake3`
//!     bus; a row's inputs and outputs are tied by being the same row.)
//!
//! ⚠ This chip implements the **6-round internal variant** — NOT standard
//! 7-round BLAKE3. Its collision resistance is a named assumption
//! (DESIGN.md "If this is picked up again").

use executor::vm::instruction::execution::{
    BLAKE3_IV, BLAKE3_MSG_PERMUTATION, BLAKE3_ROUNDS, BLAKE3_SYSCALL_NUMBER,
};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use crate::constraints::templates::{AddOperand, INV_SHIFT_32};

/// G-instances per compression: 8 per round × 6 rounds.
pub const NUM_G: usize = BLAKE3_ROUNDS * 8;

/// Dwords in the state region: 14 input (h|m|t|len_flags) + 8 output.
pub const STATE_DWORDS: usize = 22;
/// Input dwords (read-only).
pub const IN_DWORDS: usize = 14;

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
const ROT_SHIFT_R: [u32; 2] = [4, 9];

// =========================================================================
// Column indices
// =========================================================================

pub mod cols {
    use super::NUM_G;

    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    /// State address as 8 bytes (DWordBL).
    pub const ADDR: usize = 2;

    /// Per-dword pointers [22][4] halfwords (DWordHL), ptr[k] = addr + 8k.
    pub const PTR: usize = ADDR + 8; // 10

    /// Input bytes: h[32] | m[64] | t_lo[4] | t_hi[4] | block_len[4] | flags[4].
    pub const IN: usize = PTR + super::STATE_DWORDS * 4; // 98

    /// 48 G-blocks × 60 cells (56 bytes + 4 carry bits) — see `g` accessors.
    pub const G: usize = IN + 112; // 210
    pub const G_SIZE: usize = 60;

    /// Feed-forward output bytes out[0..16] (64 bytes).
    pub const OUT: usize = G + NUM_G * G_SIZE; // 3090

    /// Previous memory content of the out region (64 bytes). Appears only in
    /// the Memw write ops' `old` field; range-checked by AreBytes.
    pub const OLD_OUT: usize = OUT + 64; // 3154

    /// Multiplicity / gate flag.
    pub const MU: usize = OLD_OUT + 64; // 3218

    pub const NUM_COLUMNS: usize = MU + 1; // 3219

    // -------------------------------------------------------------------------
    // Index helpers
    // -------------------------------------------------------------------------

    #[inline]
    pub const fn addr(byte: usize) -> usize {
        ADDR + byte
    }

    /// ptr[k][hw] — halfword hw of the pointer to dword k.
    #[inline]
    pub const fn ptr(k: usize, hw: usize) -> usize {
        PTR + k * 4 + hw
    }

    /// Input word i (0..28: h[0..8], m[8..24], t_lo=24, t_hi=25, len=26, flags=27),
    /// byte b.
    #[inline]
    pub const fn in_word(i: usize, b: usize) -> usize {
        IN + i * 4 + b
    }

    /// Base column of G-block g.
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
    /// rotr7 block: same layout as G_R1.
    pub const G_R2: usize = 48;

    /// Feed-forward output word i (0..16), byte b.
    #[inline]
    pub const fn out_word(i: usize, b: usize) -> usize {
        OUT + i * 4 + b
    }

    /// Previous-content byte b (0..64) of the out region.
    #[inline]
    pub const fn old_out(b: usize) -> usize {
        OLD_OUT + b
    }
}

// =========================================================================
// The single dataflow, interpreted twice
// =========================================================================

/// The BLAKE3 compression dataflow, abstracted over its word representation.
///
/// [`run_flow`] is the only place the G-function wiring, message schedule and
/// feed-forward exist; implementors interpret the primitive ops either as
/// column wiring ([`WireFlow`]) or as u32 witness computation ([`ValueFlow`]).
pub(crate) trait Blake3Flow {
    type Word: Copy;

    /// h[i] input word.
    fn input_h(&mut self, i: usize) -> Self::Word;
    /// v[12..16] init words: t_lo, t_hi, block_len, flags.
    fn input_v12(&mut self, j: usize) -> Self::Word;
    /// IV[i] constant (v[8..12]).
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
    /// rotr16: free byte relabel [b2,b3,b0,b1].
    fn rotr16(&mut self, w: Self::Word) -> Self::Word;
    /// rotr8: free byte relabel [b1,b2,b3,b0].
    fn rotr8(&mut self, w: Self::Word) -> Self::Word;
    /// rotr12 (half=0) / rotr7 (half=1) via the inline shift identity.
    fn rot_shift(&mut self, g: usize, half: usize, w: Self::Word) -> Self::Word;
    /// Feed-forward XOR pair: out[i] = v[i] ^ v[i+8], out[i+8] = v[i+8] ^ h[i].
    fn feed_forward(&mut self, i: usize, vi: Self::Word, vi8: Self::Word, hi: Self::Word);
}

/// Drive the full 6-round compression through `f`. The message schedule is
/// tracked as indices into the ORIGINAL m (permute^r composition), so both
/// interpretations reference original message words — never copies.
pub(crate) fn run_flow<F: Blake3Flow>(f: &mut F) {
    let h: [F::Word; 8] = core::array::from_fn(|i| f.input_h(i));
    let mut v: [F::Word; 16] = core::array::from_fn(|i| {
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

    for r in 0..BLAKE3_ROUNDS {
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
        if r < BLAKE3_ROUNDS - 1 {
            let prev = sched;
            for (i, &p) in BLAKE3_MSG_PERMUTATION.iter().enumerate() {
                sched[i] = prev[p];
            }
        }
    }

    for i in 0..8 {
        f.feed_forward(i, v[i], v[i + 8], h[i]);
    }
}

// =========================================================================
// Wire interpretation (columns)
// =========================================================================

/// A 32-bit word as wiring: four byte columns (LSB first) or a constant.
/// Constants only ever appear as the IV `v[c]` operands of round-0 add2s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WordRef {
    Cols([usize; 4]),
    Const(u32),
}

impl WordRef {
    fn byte(self, b: usize) -> ByteRef {
        match self {
            WordRef::Cols(c) => ByteRef::Col(c[b]),
            WordRef::Const(w) => ByteRef::Const(((w >> (8 * b)) & 0xFF) as u8),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ByteRef {
    Col(usize),
    Const(u8),
}

/// One recorded 3-op add: operands (a, b, m columns), output columns, carries.
pub(crate) struct Add3Wire {
    pub a: WordRef,
    pub b: WordRef,
    pub m: [usize; 4],
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

/// The full wiring of one compression row: everything the constraints and the
/// bus senders need, recorded in canonical order by [`run_flow`].
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
        run_flow(&mut w);
        w
    }
}

#[inline]
fn word_cols(start: usize) -> [usize; 4] {
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
            m: word_cols(cols::in_word(8 + m_idx, 0)),
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
        match w {
            WordRef::Cols([b0, b1, b2, b3]) => WordRef::Cols([b2, b3, b0, b1]),
            WordRef::Const(v) => WordRef::Const(v.rotate_right(16)),
        }
    }
    fn rotr8(&mut self, w: WordRef) -> WordRef {
        match w {
            WordRef::Cols([b0, b1, b2, b3]) => WordRef::Cols([b1, b2, b3, b0]),
            WordRef::Const(v) => WordRef::Const(v.rotate_right(8)),
        }
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

    fn feed_forward(&mut self, i: usize, vi: WordRef, vi8: WordRef, hi: WordRef) {
        let out_lo = word_cols(cols::out_word(i, 0));
        let out_hi = word_cols(cols::out_word(i + 8, 0));
        self.xors.push(XorWire {
            a: vi,
            b: vi8,
            out: out_lo,
        });
        self.xors.push(XorWire {
            a: vi8,
            b: hi,
            out: out_hi,
        });
    }
}

// =========================================================================
// Value interpretation (u32 witness)
// =========================================================================

/// Everything the trace filler and the BITWISE collector need for one
/// compression, recorded cell-exactly in the same canonical order as
/// [`WireFlow`]. `xor_ops` carries (a, b) operand VALUES per XOR word — the
/// per-byte lookups are `(a_byte, b_byte)` in the same operand order the
/// senders use.
pub(crate) struct ValueFlow {
    /// (s, c1, c2) per add3, canonical order.
    pub add3s: Vec<(u32, u8, u8)>,
    /// s per add2 (the carry is an expression, not a cell).
    pub add2s: Vec<u32>,
    /// (a, b, out) per XOR word, canonical order (Gs then feed-forward).
    pub xors: Vec<(u32, u32, u32)>,
    /// (sll_lo, sllc_lo, sll_hi, sllc_hi, y) per shift rotation.
    pub rots: Vec<(u16, u16, u16, u16, u32)>,
    /// The 16-word output.
    pub out: [u32; 16],

    h: [u32; 8],
    m: [u32; 16],
    v12: [u32; 4],
}

impl ValueFlow {
    pub(crate) fn compute(h: &[u32; 8], m: &[u32; 16], t: u64, block_len: u32, flags: u32) -> Self {
        let mut f = ValueFlow {
            add3s: Vec::with_capacity(NUM_G * 2),
            add2s: Vec::with_capacity(NUM_G * 2),
            xors: Vec::with_capacity(NUM_G * 4 + 16),
            rots: Vec::with_capacity(NUM_G * 2),
            out: [0; 16],
            h: *h,
            m: *m,
            v12: [t as u32, (t >> 32) as u32, block_len, flags],
        };
        run_flow(&mut f);
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

    fn feed_forward(&mut self, i: usize, vi: u32, vi8: u32, hi: u32) {
        let lo = vi ^ vi8;
        let hi_w = vi8 ^ hi;
        self.xors.push((vi, vi8, lo));
        self.xors.push((vi8, hi, hi_w));
        self.out[i] = lo;
        self.out[i + 8] = hi_w;
    }
}

// =========================================================================
// Operation struct + trace generation
// =========================================================================

#[derive(Debug, Clone)]
pub struct Blake3Operation {
    pub timestamp: u64,
    pub state_addr: u64,
    pub h: [u32; 8],
    pub m: [u32; 16],
    pub t: u64,
    pub block_len: u32,
    pub flags: u32,
    /// Previous memory content of the 64-byte out region (for the Memw `old`).
    pub old_out: [u8; 64],
    /// The 16-word compression output (recomputed by the trace builder).
    pub out: [u32; 16],
}

/// Write a 32-bit word as 4 byte cells at `col..col+4`.
#[inline]
fn set_word_bytes<T: VmTable>(table: &mut T, row: usize, col: usize, w: u32) {
    for b in 0..4 {
        table.set_u64(row, col + b, ((w >> (8 * b)) & 0xFF) as u64);
    }
}

pub fn generate_blake3_trace(
    ops: &[Blake3Operation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row, op) in ops.iter().enumerate() {
        table.set_dword_wl(row, cols::TIMESTAMP_0, op.timestamp);
        table.set_dword_bl(row, cols::addr(0), op.state_addr);

        // Pointers ptr[k] = addr + 8k.
        for k in 0..STATE_DWORDS {
            let ptr = op
                .state_addr
                .checked_add(k as u64 * 8)
                .expect("blake3 state address range must be validated by the executor");
            table.set_dword_hl(row, cols::ptr(k, 0), ptr);
        }

        // Input words: h | m | t_lo t_hi len flags.
        for i in 0..8 {
            set_word_bytes(table, row, cols::in_word(i, 0), op.h[i]);
        }
        for i in 0..16 {
            set_word_bytes(table, row, cols::in_word(8 + i, 0), op.m[i]);
        }
        set_word_bytes(table, row, cols::in_word(24, 0), op.t as u32);
        set_word_bytes(table, row, cols::in_word(25, 0), (op.t >> 32) as u32);
        set_word_bytes(table, row, cols::in_word(26, 0), op.block_len);
        set_word_bytes(table, row, cols::in_word(27, 0), op.flags);

        // The mixing core, cell-exactly in canonical order.
        let flow = ValueFlow::compute(&op.h, &op.m, op.t, op.block_len, op.flags);
        debug_assert_eq!(
            flow.out, op.out,
            "trace-builder output must match the executor"
        );

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
                set_word_bytes(table, row, base + s_off, s);
                table.set_u64(row, base + c_off, c1 as u64);
                table.set_u64(row, base + c_off + 1, c2 as u64);

                let &(_, _, x) = xo.next().expect("xor count");
                set_word_bytes(table, row, base + x_off, x);

                let &c = a2.next().expect("add2 count");
                set_word_bytes(table, row, base + c2_off, c);

                let &(_, _, x2) = xo.next().expect("xor count");
                set_word_bytes(table, row, base + x2_off, x2);

                let &(sll_lo, sllc_lo, sll_hi, sllc_hi, y) = ro.next().expect("rot count");
                table.set_u64(row, base + r_off, (sll_lo & 0xFF) as u64);
                table.set_u64(row, base + r_off + 1, (sll_lo >> 8) as u64);
                table.set_u64(row, base + r_off + 2, (sllc_lo & 0xFF) as u64);
                table.set_u64(row, base + r_off + 3, (sllc_lo >> 8) as u64);
                table.set_u64(row, base + r_off + 4, (sll_hi & 0xFF) as u64);
                table.set_u64(row, base + r_off + 5, (sll_hi >> 8) as u64);
                table.set_u64(row, base + r_off + 6, (sllc_hi & 0xFF) as u64);
                table.set_u64(row, base + r_off + 7, (sllc_hi >> 8) as u64);
                set_word_bytes(table, row, base + r_off + 8, y);
            }
        }
        // Feed-forward outputs (the last 16 entries of flow.xors).
        for i in 0..16 {
            set_word_bytes(table, row, cols::out_word(i, 0), flow.out[i]);
        }
        // Previous content of the out region.
        for b in 0..64 {
            table.set_u64(row, cols::old_out(b), op.old_out[b] as u64);
        }

        table.set_fe(row, cols::MU, FE::one());
    }

    // Padding rows: ptr[k][0] = 8k (all fit in the low halfword), matching the
    // keccak pad idiom. μ = 0 gates every constraint and interaction.
    for row in n..num_rows {
        for k in 0..STATE_DWORDS {
            table.set_u64(row, cols::ptr(k, 0), (k as u64) * 8);
        }
    }

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Order groups: I/O (Ecall + reg-read + 22 Memw), then the mixing core's
/// ByteAlu XORs (canonical WireFlow order), then the shift AreBytes, then the
/// message/old-out/addr AreBytes, the alignment AND and the pointer IS_HALFs.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let syscall_lo = BLAKE3_SYSCALL_NUMBER & 0xFFFF_FFFF;
    let syscall_hi = BLAKE3_SYSCALL_NUMBER >> 32;
    let wires = WireFlow::build();
    let mut interactions = Vec::with_capacity(1400);

    let byte_bus_value = |b: ByteRef| -> BusValue {
        match b {
            ByteRef::Col(c) => BusValue::Packed {
                start_column: c,
                packing: Packing::Direct,
            },
            ByteRef::Const(v) => BusValue::constant(v as u64),
        }
    };

    // 1. ECALL receiver: [ts_lo, ts_hi, syscall_lo32, syscall_hi32].
    interactions.push(BusInteraction::receiver(
        BusId::Ecall,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::constant(syscall_lo),
            BusValue::constant(syscall_hi),
        ],
    ));

    // 2. MEMW read of register x10 binding the state address (keccak idiom):
    // [old(8), is_register=1, base=20, value(8), ts(2), w2=1, w4=0, w8=0].
    {
        let addr_word = |lo_byte: usize| -> BusValue {
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::addr(lo_byte),
                },
                LinearTerm::Column {
                    coefficient: 256,
                    column: cols::addr(lo_byte + 1),
                },
                LinearTerm::Column {
                    coefficient: 65536,
                    column: cols::addr(lo_byte + 2),
                },
                LinearTerm::Column {
                    coefficient: 16777216,
                    column: cols::addr(lo_byte + 3),
                },
            ])
        };
        let mut values = Vec::with_capacity(24);
        values.push(addr_word(0));
        values.push(addr_word(4));
        for _ in 2..8 {
            values.push(BusValue::constant(0));
        }
        values.push(BusValue::constant(1)); // is_register
        values.push(BusValue::constant(20)); // x10 → address 2*10
        values.push(BusValue::constant(0));
        values.push(addr_word(0));
        values.push(addr_word(4));
        for _ in 2..8 {
            values.push(BusValue::constant(0));
        }
        values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_0,
            packing: Packing::Direct,
        });
        values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_1,
            packing: Packing::Direct,
        });
        values.push(BusValue::constant(1)); // w2 (register)
        values.push(BusValue::constant(0));
        values.push(BusValue::constant(0));
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::MU),
            values,
        ));
    }

    // 3. MEMW per state dword: [old(8), is_register=0, addr(2), value(8), ts(2),
    // w2=0, w4=0, w8=1]. Input dwords are pure reads (old = value = input
    // bytes); output dwords write OUT over OLD_OUT.
    for k in 0..STATE_DWORDS {
        let addr_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::ptr(k, 0),
            },
            LinearTerm::Column {
                coefficient: 65536,
                column: cols::ptr(k, 1),
            },
        ]);
        let addr_hi = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::ptr(k, 2),
            },
            LinearTerm::Column {
                coefficient: 65536,
                column: cols::ptr(k, 3),
            },
        ]);

        // (old bytes, value bytes) column bases for this dword.
        let (old_base, val_base): (Vec<usize>, Vec<usize>) = if k < IN_DWORDS {
            let cols8: Vec<usize> = (0..8).map(|b| cols::in_word(2 * k, 0) + b).collect();
            (cols8.clone(), cols8)
        } else {
            let o = k - IN_DWORDS;
            (
                (0..8).map(|b| cols::old_out(o * 8 + b)).collect(),
                (0..8).map(|b| cols::out_word(2 * o, 0) + b).collect(),
            )
        };

        let mut values = Vec::with_capacity(24);
        for &c in &old_base {
            values.push(BusValue::Packed {
                start_column: c,
                packing: Packing::Direct,
            });
        }
        values.push(BusValue::constant(0)); // is_register
        values.push(addr_lo);
        values.push(addr_hi);
        for &c in &val_base {
            values.push(BusValue::Packed {
                start_column: c,
                packing: Packing::Direct,
            });
        }
        values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_0,
            packing: Packing::Direct,
        });
        values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_1,
            packing: Packing::Direct,
        });
        values.push(BusValue::constant(0));
        values.push(BusValue::constant(0));
        values.push(BusValue::constant(1)); // w8
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::MU),
            values,
        ));
    }

    // 4. Mixing core + feed-forward: ByteAlu[XOR] per byte, canonical order.
    for xw in &wires.xors {
        for b in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::ByteAlu,
                Multiplicity::Column(cols::MU),
                vec![
                    BusValue::constant(alu_op::XOR as u64),
                    byte_bus_value(xw.a.byte(b)),
                    byte_bus_value(xw.b.byte(b)),
                    BusValue::Packed {
                        start_column: xw.out[b],
                        packing: Packing::Direct,
                    },
                ],
            ));
        }
    }

    // 5. Shift-halfword AreBytes: 4 pairs per rotation
    // (SLL_lo, SLLC_lo, SLL_hi, SLLC_hi bytes).
    for rw in &wires.rots {
        for pair in [rw.sll_lo, rw.sllc_lo, rw.sll_hi, rw.sllc_hi] {
            interactions.push(BusInteraction::sender(
                BusId::AreBytes,
                Multiplicity::Column(cols::MU),
                vec![
                    BusValue::Packed {
                        start_column: pair[0],
                        packing: Packing::Direct,
                    },
                    BusValue::Packed {
                        start_column: pair[1],
                        packing: Packing::Direct,
                    },
                ],
            ));
        }
    }

    // 6. Message AreBytes (m is never XORed — DESIGN §4.7/§7.5): 32 pairs.
    for i in 0..16 {
        for p in 0..2 {
            interactions.push(BusInteraction::sender(
                BusId::AreBytes,
                Multiplicity::Column(cols::MU),
                vec![
                    BusValue::Packed {
                        start_column: cols::in_word(8 + i, 2 * p),
                        packing: Packing::Direct,
                    },
                    BusValue::Packed {
                        start_column: cols::in_word(8 + i, 2 * p + 1),
                        packing: Packing::Direct,
                    },
                ],
            ));
        }
    }

    // 7. OLD_OUT AreBytes: those bytes only ride the Memw bus; without a byte
    // range check their packed linear combinations alias (same argument as the
    // addr bytes in keccak.rs).
    for p in 0..32 {
        interactions.push(BusInteraction::sender(
            BusId::AreBytes,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::old_out(2 * p),
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::old_out(2 * p + 1),
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // 8. Address byte range checks (4 pairs) + alignment addr[0] & 7 = 0.
    for i in 0..4 {
        interactions.push(BusInteraction::sender(
            BusId::AreBytes,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::addr(2 * i),
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::addr(2 * i + 1),
                    packing: Packing::Direct,
                },
            ],
        ));
    }
    interactions.push(BusInteraction::sender(
        BusId::ByteAlu,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::constant(alu_op::AND as u64),
            BusValue::Packed {
                start_column: cols::addr(0),
                packing: Packing::Direct,
            },
            BusValue::constant(7),
            BusValue::constant(0),
        ],
    ));

    // 9. IS_HALF range checks on the 22 pointers' halfwords.
    for k in 0..STATE_DWORDS {
        for hw in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::IsHalfword,
                Multiplicity::Column(cols::MU),
                vec![BusValue::Packed {
                    start_column: cols::ptr(k, hw),
                    packing: Packing::Direct,
                }],
            ));
        }
    }

    interactions
}

// =========================================================================
// Single-source constraint set
// =========================================================================

/// The BLAKE3 table's transition constraints (814 total):
/// - idx 0..44:    22 pointer `ADD` carry pairs (`ptr[k] = addr + 8k`, μ-gated);
/// - idx 44:       μ·carry_1 = 0 — top-dword no-overflow (`addr + 168 = ptr[21]`);
/// - idx 45..333:  all 96 add3 groups (sum identity + 2 carry booleanities);
/// - idx 333..429: all 96 add2 expression-carry booleanities;
/// - idx 429..813: all 96 rotations (2 shift identities + 2 recombine each).
///   NOTE the grouping is by op type across the whole row, NOT per G — G #g's
///   16 constraints are scattered across the three bands.
/// - idx 813:      `IS_BIT(MU)` — μ·(1−μ) = 0, ungated. The bus argument pins
///   μ to {0,1} indirectly (the Ecall receive anchors μ>0 rows to a CPU ecall
///   whose ECALL flag is IS_BIT; MEMW's width flags are boolean), but that is
///   an inter-table argument — this makes it local, matching ecsm/commit.
///
/// All μ-gated, max degree 3 (the booleanities; identities are degree 2).
#[derive(Clone, Copy)]
pub struct Blake3Constraints;

/// Word expression from a [`WordRef`]: b0 + 256·b1 + 2^16·b2 + 2^24·b3.
fn word_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    w: &WordRef,
) -> B::Expr {
    match w {
        WordRef::Cols(c) => {
            b.main(0, c[0])
                + b.main(0, c[1]) * b.const_base(256)
                + b.main(0, c[2]) * b.const_base(65536)
                + b.main(0, c[3]) * b.const_base(16777216)
        }
        WordRef::Const(v) => b.const_base(*v as u64),
    }
}

/// Halfword expression from 2 byte columns: b0 + 256·b1.
fn half_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    c: &[usize; 2],
) -> B::Expr {
    b.main(0, c[0]) + b.main(0, c[1]) * b.const_base(256)
}

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for Blake3Constraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        use crate::constraints::templates::emit_add_pair;

        let wires = WireFlow::build();
        let mu = |b: &B| b.main(0, cols::MU);

        // idx 0..44: ptr[k] = addr + 8k (μ-gated carry pairs).
        for k in 0..STATE_DWORDS {
            emit_add_pair(
                b,
                k * 2,
                &[cols::MU],
                &AddOperand::from_dword_bl(cols::ADDR),
                &AddOperand::constant((k * 8) as i64),
                &AddOperand::from_dword_hl(cols::ptr(k, 0)),
            );
        }

        // idx 44: top-dword no-overflow — μ·carry_1 of addr + 168 = ptr[21].
        let mut idx = STATE_DWORDS * 2;
        {
            let c256 = b.const_base(256);
            let c65536 = b.const_base(65536);
            let c16777216 = b.const_base(16777216);
            let addr_lo = b.main(0, cols::addr(0))
                + b.main(0, cols::addr(1)) * c256.clone()
                + b.main(0, cols::addr(2)) * c65536.clone()
                + b.main(0, cols::addr(3)) * c16777216.clone();
            let addr_hi = b.main(0, cols::addr(4))
                + b.main(0, cols::addr(5)) * c256
                + b.main(0, cols::addr(6)) * c65536.clone()
                + b.main(0, cols::addr(7)) * c16777216;
            let last = STATE_DWORDS - 1;
            let ptr_lo =
                b.main(0, cols::ptr(last, 0)) + b.main(0, cols::ptr(last, 1)) * c65536.clone();
            let ptr_hi = b.main(0, cols::ptr(last, 2)) + b.main(0, cols::ptr(last, 3)) * c65536;

            let inv_2_32 = b.const_base(INV_SHIFT_32);
            let off = b.const_base((8 * last) as u64);
            let carry_0 = (addr_lo + off - ptr_lo) * inv_2_32.clone();
            let carry_1 = (addr_hi + carry_0 - ptr_hi) * inv_2_32;
            let m = mu(b);
            b.emit_base(idx, m * carry_1);
            idx += 1;
        }

        // Mixing core. Same canonical order as the wire builder records.
        let two_32 = b.const_base(1u64 << 32);
        let inv_2_32 = b.const_base(INV_SHIFT_32);

        // add3: μ·(a + b + m − s − 2^32·(c1+c2)) = 0; μ·ci·(1−ci) = 0.
        for aw in &wires.add3s {
            let a = word_expr(b, &aw.a);
            let bb = word_expr(b, &aw.b);
            let m_w = word_expr(b, &WordRef::Cols(aw.m));
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
                WordRef::Const(_) => unreachable!("shift inputs are always committed XOR outputs"),
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

        // idx 813: IS_BIT(MU) — ungated booleanity, degree 2. See the struct
        // doc for why this is emitted even though the bus argument already
        // pins μ indirectly.
        crate::constraints::templates::emit_is_bit(b, idx, cols::MU, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value interpretation must reproduce the executor's compression —
    /// same function the canonical oracle vectors validate.
    #[test]
    fn value_flow_matches_executor() {
        use executor::vm::instruction::execution::blake3_compress_6round;
        let h: [u32; 8] = core::array::from_fn(|i| 0x9E3779B9u32.wrapping_mul(i as u32 + 1));
        let m: [u32; 16] = core::array::from_fn(|i| 0x85EBCA6Bu32.wrapping_mul(i as u32 + 7));
        let t = 0x0123_4567_89AB_CDEFu64;
        let (bl, fl) = (64u32, 11u32);
        let flow = ValueFlow::compute(&h, &m, t, bl, fl);
        assert_eq!(flow.out, blake3_compress_6round(&h, &m, t, bl, fl));
    }

    /// Canonical op counts: 96 add3s, 96 add2s, 96 rotations, 192+16 XORs.
    #[test]
    fn wire_flow_counts() {
        let w = WireFlow::build();
        assert_eq!(w.add3s.len(), NUM_G * 2);
        assert_eq!(w.add2s.len(), NUM_G * 2);
        assert_eq!(w.rots.len(), NUM_G * 2);
        assert_eq!(w.xors.len(), NUM_G * 4 + 16);
        // Every output column lands exactly once, and inside the row.
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let mut claim = |c: usize| {
            assert!(c < cols::NUM_COLUMNS, "column {c} out of range");
            assert!(seen.insert(c), "column {c} written twice");
        };
        for aw in &w.add3s {
            for c in aw.s {
                claim(c);
            }
            claim(aw.c1);
            claim(aw.c2);
        }
        for aw in &w.add2s {
            for c in aw.s {
                claim(c);
            }
        }
        for xw in &w.xors {
            for c in xw.out {
                claim(c);
            }
        }
        for rw in &w.rots {
            for c in rw
                .sll_lo
                .iter()
                .chain(&rw.sllc_lo)
                .chain(&rw.sll_hi)
                .chain(&rw.sllc_hi)
                .chain(&rw.y)
            {
                claim(*c);
            }
        }
        // 48 G-blocks × 60 cells + 64 out bytes, all distinct.
        assert_eq!(seen.len(), NUM_G * cols::G_SIZE + 64);
    }
}

/// ★ The executor's compression and `crypto`'s shared primitive are the same
/// function.
///
/// #903 landed a second host transcription of the BLAKE3 compression:
/// `executor::vm::instruction::execution::blake3_compress_6round`, with its own
/// `blake3_g`, its own `BLAKE3_IV` and its own `BLAKE3_ROUNDS = 6`. `crypto`'s
/// `blake3_compress_rounds` is the one P-a Stage 1 hoisted precisely so there
/// would be a single definition — the same treatment the CUDA reference got.
///
/// Two independently written encodings of one function is what PA-PLAN §1.4
/// forbids ("do not prove that two … coincide; make them one function"), and
/// merging #903 reintroduced it. Until they are unified, this is the gate: the
/// executor is what the guest's syscall actually runs, so a divergence here is
/// a guest that hashes differently from the host prover — R5's invisible
/// failure, which surfaces only as in-guest proof rejection.
///
/// Checked over the message schedule's structural edge cases plus a pseudo-random
/// sweep, at every `(t, block_len, flags)` shape the chain framing produces.
#[cfg(test)]
mod executor_primitive_parity {
    use crypto::hash::blake3::{BLAKE3_SIX_ROUNDS, blake3_compress_rounds};
    use executor::vm::instruction::execution::blake3_compress_6round;

    #[test]
    fn the_executor_compression_is_the_shared_primitive() {
        // A cheap deterministic stream; no rand dependency in this crate.
        let mut z = 0x243f_6a88_85a3_08d3u64;
        let mut next = move || {
            z ^= z << 13;
            z ^= z >> 7;
            z ^= z << 17;
            z as u32
        };

        // The flag/counter shapes `Blake3Chain` actually emits: first block
        // (CHUNK_START), interior, and last (CHUNK_END|ROOT with the true byte
        // count as block_len). t is 0 throughout for the chain, but the syscall
        // takes a full 64-bit counter, so both halves are exercised.
        let shapes: [(u64, u32, u32); 5] = [
            (0, 64, 1),                // CHUNK_START
            (0, 64, 0),                // interior
            (0, 7, 2 | 8),             // CHUNK_END | ROOT, partial final block
            (u64::MAX, 64, 0),         // both counter halves set
            (1 << 32, 0, 0xffff_ffff), // high half only; degenerate len/flags
        ];

        for (t, block_len, flags) in shapes {
            for _ in 0..64 {
                let h: [u32; 8] = core::array::from_fn(|_| next());
                let m: [u32; 16] = core::array::from_fn(|_| next());

                assert_eq!(
                    blake3_compress_6round(&h, &m, t, block_len, flags),
                    blake3_compress_rounds(&h, &m, t, block_len, flags, BLAKE3_SIX_ROUNDS),
                    "executor and crypto disagree at t={t}, block_len={block_len}, flags={flags}"
                );
            }
        }
    }

    /// CONTROL: the two would NOT agree at a different round count, so the test
    /// above is comparing round counts as well as wiring.
    #[test]
    fn the_parity_is_round_count_sensitive() {
        let h = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let m: [u32; 16] = core::array::from_fn(|i| (i as u32) * 7 + 1);
        assert_ne!(
            blake3_compress_6round(&h, &m, 0, 64, 1),
            blake3_compress_rounds(&h, &m, 0, 64, 1, BLAKE3_SIX_ROUNDS + 1),
            "a 7-round reference must not match the 6-round executor"
        );
    }
}

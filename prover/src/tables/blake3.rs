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
//! ## Two modes, one table
//!
//! The table serves two syscalls, selected per row by a pair of mutually
//! exclusive multiplicity columns with `MU = MU_S + MU_A`:
//!
//! * **`MU_S` — single compression** ([`BLAKE3_SYSCALL_NUMBER`]). One ecall,
//!   one row, the 176-byte state region described above. Unchanged.
//! * **`MU_A` — chained absorb**
//!   ([`executor::vm::instruction::execution::BLAKE3_ABSORB_SYSCALL_NUMBER`]).
//!   One ecall folds `num_blocks` consecutive 64-byte blocks into a chaining
//!   value, and occupies `num_blocks + 1` rows: one compression per block, then
//!   an END row that does no work and writes `cv_out`. The rows are linked by
//!   the self-referencing [`BusId::Blake3Absorb`] bus, COMMIT's idiom
//!   (`commit.rs`) with the countdown carried in a single field element.
//!
//! Every row of one absorb group carries that ecall's timestamp, so the group
//! is identified on the bus by `(timestamp, …)`. `END` is never free: it is the
//! output of a `Zero[REMAINING]` lookup, so a prover cannot end a group early
//! (the lookup rejects) or late (`REMAINING = 0 ⇒ END = 1 ⇒ no send`, and a
//! send with no receiver unbalances the bus).
//!
//! The absorb mode's soundness ledger, beyond the compression core's:
//!
//! 11. `MU_S · MU_A = 0` and `MU − MU_S − MU_A = 0`, both `IS_BIT` — a row is in
//!     exactly one mode, and neither mode's interactions fire in the other.
//! 12. `(FIRST + END)·(1 − MU_A) = 0` locks both boundary flags to absorb rows,
//!     so a `μ = 0` padding row cannot mint a group boundary; `FIRST · END = 0`
//!     forbids the zero-block group the executor also rejects.
//! 13. The chain bus tuple leads with `TIMESTAMP_0, TIMESTAMP_1` and carries the
//!     control address and message base as well as the chaining value, so a row
//!     cannot receive one group's state and send another's (DESIGN.md §1.1), nor
//!     read a block of its own choosing.
//! 14. The END row's mixing core is gated OFF (`MU − END`), so its `h` bytes are
//!     range-checked by an explicit `AreBytes` instead — without it the `cv_out`
//!     write could place non-canonical "bytes" in memory.
//! 15. The block cap is enforced in-circuit (`IsB20[REM_DECR · 2^10]` on the
//!     FIRST row), not inherited from the executor, so the chip accepts exactly
//!     the `1..=1024` the VM semantics accept.
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
    BLAKE3_ABSORB_SYSCALL_NUMBER, BLAKE3_IV, BLAKE3_MSG_PERMUTATION, BLAKE3_ROUNDS,
    BLAKE3_SYSCALL_NUMBER,
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
/// Dwords of the state region that carry the chaining value `h`. In absorb mode
/// the same columns and the same `PTR[0..4]` addresses hold `cv_in`.
pub const CV_DWORDS: usize = 4;
/// Dwords of one 64-byte message block.
pub const MSG_DWORDS: usize = 8;
/// Dwords of the absorb control region: 4 for `cv_in`, 4 for `cv_out`. Only
/// `PTR[0..CTRL_DWORDS]` is meaningful on an absorb row.
pub const CTRL_DWORDS: usize = 8;
/// Dword offset of `cv_out` inside the absorb control region — pinned equal to
/// the executor's [`BLAKE3_ABSORB_CV_OUT_DWORD`] by `the_chip_agrees_on_the_absorb_abi`.
pub const CV_OUT_DWORD: usize = 4;
/// Largest `num_blocks` the chip's own range check admits, restated in-circuit
/// rather than inherited from the executor: see `absorb_cap_in_circuit`.
pub const ABSORB_MAX_BLOCKS: u64 = 1 << 10;
/// Multiplier that turns the `IsB20` 20-bit lookup into the exact `≤ 2^10` cap:
/// `REM_DECR · 2^10 < 2^20 ⟺ REM_DECR < 2^10`.
pub const ABSORB_CAP_SCALE: u64 = (1 << 20) / ABSORB_MAX_BLOCKS;

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

    /// Multiplicity / gate flag. `MU = MU_S + MU_A`.
    pub const MU: usize = OLD_OUT + 64; // 3218

    // -------------------------------------------------------------------------
    // Absorb mode (see the module docs' "Two modes" section)
    // -------------------------------------------------------------------------

    /// Single-compression row (the `BLAKE3_SYSCALL_NUMBER` mode).
    pub const MU_S: usize = MU + 1; // 3219
    /// Absorb row — a compression row of a group, or its END row.
    pub const MU_A: usize = MU_S + 1; // 3220
    /// Absorb row that compresses: `MU_C = MU_A − END`. A column rather than an
    /// expression because `emit_add_pair` gates on a SUM of columns.
    pub const MU_C: usize = MU_A + 1; // 3221
    /// First row of an absorb group (the one that receives the `Ecall`).
    pub const FIRST: usize = MU_C + 1; // 3222
    /// Last row of an absorb group: does no compression, writes `cv_out`.
    pub const END: usize = FIRST + 1; // 3223
    /// Blocks left to fold INCLUDING this row's: `num_blocks` on FIRST, 0 on END.
    pub const REMAINING: usize = END + 1; // 3224
    /// `REMAINING − 1`, the countdown the chain bus carries forward.
    pub const REM_DECR: usize = REMAINING + 1; // 3225
    /// Message address of this row's block, as 4 halfwords (DWordHL).
    pub const M_BASE: usize = REM_DECR + 1; // 3226
    /// `M_BASE + 64` — the next row's block address (DWordHL).
    pub const M_BASE_INCR: usize = M_BASE + 4; // 3230
    /// Per-dword message pointers `[8][4]` halfwords, `msg_ptr[j] = M_BASE + 8j`.
    pub const MSG_PTR: usize = M_BASE_INCR + 4; // 3234

    pub const NUM_COLUMNS: usize = MSG_PTR + super::MSG_DWORDS * 4; // 3266

    /// `msg_ptr[j][hw]` — halfword `hw` of the pointer to message dword `j`.
    #[inline]
    pub const fn msg_ptr(j: usize, hw: usize) -> usize {
        MSG_PTR + j * 4 + hw
    }

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

/// One chained-absorb ecall: `num_blocks` compressions plus the END row that
/// writes `cv_out`, all at one timestamp.
#[derive(Debug, Clone)]
pub struct Blake3AbsorbOperation {
    pub timestamp: u64,
    /// x10 — the 64-byte control region (`cv_in` in dwords 0..4, `cv_out` 4..8).
    pub ctrl_addr: u64,
    /// x11 — the message, read in place.
    pub msg_addr: u64,
    /// x13 — the flag word of block 0; every later block carries 0.
    pub first_flags: u32,
    /// The incoming chaining value, read from `ctrl_addr`.
    pub cv_in: [u32; 8],
    /// The message, one entry per block. `num_blocks` is its length.
    pub blocks: Vec<[u32; 16]>,
    /// Previous memory content of the 32-byte `cv_out` region (the Memw `old`).
    pub old_cv_out: [u8; 32],
}

/// One row of an absorb group, in trace order: `num_blocks` compressing rows
/// then the END row.
///
/// ★ This expansion is the single origin of the group's per-row witness. The
/// trace filler and `trace_builder::collect_bitwise_from_blake3` both iterate
/// it, so the BITWISE multiplicities cannot drift from the rows that send them.
#[derive(Debug, Clone)]
pub(crate) struct AbsorbRow {
    /// Chaining value entering this row (`cv_in` on FIRST, the previous row's
    /// output after that; on the END row, the group's result).
    pub h: [u32; 8],
    /// This row's message block; all-zero on the END row.
    pub m: [u32; 16],
    /// `first_flags` on the FIRST row, 0 elsewhere.
    pub flags: u32,
    /// Blocks left including this row's; 0 on the END row.
    pub remaining: u32,
    /// Address of this row's block. On the END row this is one past the last
    /// block, which may WRAP when a message ends at the top of memory — the
    /// chain carries it and nothing reads it, so no no-overflow constraint
    /// applies to it.
    pub m_base: u64,
    pub first: bool,
    pub end: bool,
}

/// Expand one absorb ecall into its `num_blocks + 1` rows.
pub(crate) fn expand_absorb(op: &Blake3AbsorbOperation) -> Vec<AbsorbRow> {
    let n = op.blocks.len();
    let mut rows = Vec::with_capacity(n + 1);
    let mut cv = op.cv_in;
    for (i, block) in op.blocks.iter().enumerate() {
        let flags = if i == 0 { op.first_flags } else { 0 };
        rows.push(AbsorbRow {
            h: cv,
            m: *block,
            flags,
            remaining: (n - i) as u32,
            m_base: op.msg_addr.wrapping_add((i as u64) * 64),
            first: i == 0,
            end: false,
        });
        // The chaining value the chip carries forward is the compression's first
        // eight output words — the same `out[0..8]` the internal bus sends.
        let out = ValueFlow::compute(&cv, block, 0, 64, flags).out;
        cv = out[0..8].try_into().expect("out[0..8] is 8 words");
    }
    rows.push(AbsorbRow {
        h: cv,
        m: [0; 16],
        flags: 0,
        remaining: 0,
        m_base: op.msg_addr.wrapping_add((n as u64) * 64),
        first: false,
        end: true,
    });
    rows
}

/// Write a 32-bit word as 4 byte cells at `col..col+4`.
#[inline]
fn set_word_bytes<T: VmTable>(table: &mut T, row: usize, col: usize, w: u32) {
    for b in 0..4 {
        table.set_u64(row, col + b, ((w >> (8 * b)) & 0xFF) as u64);
    }
}

/// Fill one row's mixing core and feed-forward output from a computed flow,
/// cell-exactly in [`WireFlow`]'s canonical order. Shared by both modes: an
/// absorb compression row is the same 814-constraint core as a single one.
fn fill_mixing_core<T: VmTable>(table: &mut T, row: usize, flow: &ValueFlow) {
    {
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
    }
}

/// Fill the compression inputs `h | m | t_lo t_hi block_len flags` of one row.
fn fill_compression_inputs<T: VmTable>(
    table: &mut T,
    row: usize,
    h: &[u32; 8],
    m: &[u32; 16],
    t: u64,
    block_len: u32,
    flags: u32,
) {
    for (i, &w) in h.iter().enumerate() {
        set_word_bytes(table, row, cols::in_word(i, 0), w);
    }
    for (i, &w) in m.iter().enumerate() {
        set_word_bytes(table, row, cols::in_word(8 + i, 0), w);
    }
    set_word_bytes(table, row, cols::in_word(24, 0), t as u32);
    set_word_bytes(table, row, cols::in_word(25, 0), (t >> 32) as u32);
    set_word_bytes(table, row, cols::in_word(26, 0), block_len);
    set_word_bytes(table, row, cols::in_word(27, 0), flags);
}

/// Generate the BLAKE3 trace: the single-compression rows first, then each
/// absorb ecall's group of `num_blocks + 1` rows.
///
/// Groups are laid down contiguously, but nothing depends on that — every
/// cross-row link goes through the [`BusId::Blake3Absorb`] bus, which is keyed
/// on the ecall's timestamp and is therefore order-free.
pub fn generate_blake3_trace(
    ops: &[Blake3Operation],
    absorb_ops: &[Blake3AbsorbOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let absorb_rows: Vec<(&Blake3AbsorbOperation, AbsorbRow)> = absorb_ops
        .iter()
        .flat_map(|op| expand_absorb(op).into_iter().map(move |r| (op, r)))
        .collect();
    let n = ops.len() + absorb_rows.len();
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

        // Pointers ptr[k] = addr + 8k. All 22: single mode reads the whole
        // 176-byte state region, whose top the executor's range check binds.
        for k in 0..STATE_DWORDS {
            let ptr = op
                .state_addr
                .checked_add(k as u64 * 8)
                .expect("blake3 state address range must be validated by the executor");
            table.set_dword_hl(row, cols::ptr(k, 0), ptr);
        }

        fill_compression_inputs(table, row, &op.h, &op.m, op.t, op.block_len, op.flags);

        // The mixing core, cell-exactly in canonical order.
        let flow = ValueFlow::compute(&op.h, &op.m, op.t, op.block_len, op.flags);
        debug_assert_eq!(
            flow.out, op.out,
            "trace-builder output must match the executor"
        );
        fill_mixing_core(table, row, &flow);

        // Previous content of the out region.
        for b in 0..64 {
            table.set_u64(row, cols::old_out(b), op.old_out[b] as u64);
        }

        table.set_fe(row, cols::MU, FE::one());
        table.set_fe(row, cols::MU_S, FE::one());
    }

    for (i, (op, r)) in absorb_rows.iter().enumerate() {
        let row = ops.len() + i;
        table.set_dword_wl(row, cols::TIMESTAMP_0, op.timestamp);
        table.set_dword_bl(row, cols::addr(0), op.ctrl_addr);

        // Only the control region's 8 pointers. ptr[8..22] stay 0 and their
        // constraints and range checks are gated on MU_S — an absorb's x10 is
        // bounded by the executor at ctrl + 63, NOT at ctrl + 168, so computing
        // the upper pointers here would panic on an address the ABI accepts.
        for k in 0..CTRL_DWORDS {
            let ptr = op
                .ctrl_addr
                .checked_add(k as u64 * 8)
                .expect("absorb control region range must be validated by the executor");
            table.set_dword_hl(row, cols::ptr(k, 0), ptr);
        }

        // t = 0 and block_len = 64 on every absorbed block: the interior
        // schedule, constrained on the row rather than assumed.
        fill_compression_inputs(table, row, &r.h, &r.m, 0, 64, r.flags);

        table.set_fe(row, cols::MU, FE::one());
        table.set_fe(row, cols::MU_A, FE::one());
        table.set_bool(row, cols::FIRST, r.first);
        table.set_bool(row, cols::END, r.end);
        table.set_u64(row, cols::REMAINING, r.remaining as u64);
        table.set_dword_hl(row, cols::M_BASE, r.m_base);

        if r.end {
            // The END row does no compression; the chain receive lands the
            // group's result in `h`, and the cv_out write reads those columns.
            for (b, &old) in op.old_cv_out.iter().enumerate() {
                table.set_u64(row, cols::old_out(b), old as u64);
            }
        } else {
            table.set_fe(row, cols::MU_C, FE::one());
            table.set_u64(row, cols::REM_DECR, (r.remaining - 1) as u64);
            table.set_dword_hl(row, cols::M_BASE_INCR, r.m_base.wrapping_add(64));
            for j in 0..MSG_DWORDS {
                let ptr = r
                    .m_base
                    .checked_add(j as u64 * 8)
                    .expect("absorb message range must be validated by the executor");
                table.set_dword_hl(row, cols::msg_ptr(j, 0), ptr);
            }
            let flow = ValueFlow::compute(&r.h, &r.m, 0, 64, r.flags);
            fill_mixing_core(table, row, &flow);
        }
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
/// message/old-out/addr AreBytes, the alignment AND and the pointer IS_HALFs,
/// then the absorb mode's own block (registers, message, `cv_out`, the chain
/// bus, the countdown lookups and the absorb-only range checks).
///
/// ## Per-mode multiplicities
///
/// | interaction | multiplicity |
/// |---|---|
/// | `Ecall` receive, single syscall | `MU_S` |
/// | `Ecall` receive, absorb syscall | `FIRST` |
/// | x10 read (state addr / control addr) | `MU_S + FIRST` |
/// | state dwords 0..4 (`h` / `cv_in`) | `MU_S + FIRST` |
/// | state dwords 4..22 (`m`, `t|len|flags`, `out`) | `MU_S` |
/// | x11/x12/x13 reads | `FIRST` |
/// | message block read (8 dwords) | `MU_C` |
/// | `cv_out` write (4 dwords) | `END` |
/// | chain send / receive | `MU_C` / `MU_A − FIRST` |
/// | `Zero[REMAINING] → END` | `MU_A` |
/// | mixing core (ByteAlu, AreBytes) | `MU − END` |
/// | `h` byte range check | `END` (the one row the mixing core does not cover) |
///
/// ★ The two `Ecall` receives cannot be one interaction with multiplicity
/// `MU_S + FIRST`: the syscall number is a CONSTANT in the tuple and the two
/// modes have different ones. Merging them would make every single-compression
/// row claim the absorb syscall (or vice versa).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let syscall_lo = BLAKE3_SYSCALL_NUMBER & 0xFFFF_FFFF;
    let syscall_hi = BLAKE3_SYSCALL_NUMBER >> 32;
    let absorb_syscall_lo = BLAKE3_ABSORB_SYSCALL_NUMBER & 0xFFFF_FFFF;
    let absorb_syscall_hi = BLAKE3_ABSORB_SYSCALL_NUMBER >> 32;
    let wires = WireFlow::build();
    let mut interactions = Vec::with_capacity(1500);

    // Reusable multiplicities. `MU − END` covers both modes' compression rows:
    // single rows have END = 0, so the single mode's gating is unchanged.
    let mu_s_or_first = Multiplicity::Sum(cols::MU_S, cols::FIRST);
    let compressing = Multiplicity::Diff(cols::MU, cols::END);

    let byte_bus_value = |b: ByteRef| -> BusValue {
        match b {
            ByteRef::Col(c) => BusValue::Packed {
                start_column: c,
                packing: Packing::Direct,
            },
            ByteRef::Const(v) => BusValue::constant(v as u64),
        }
    };
    let direct = |c: usize| -> BusValue {
        BusValue::Packed {
            start_column: c,
            packing: Packing::Direct,
        }
    };
    // A 32-bit word from 4 byte columns, LSB first.
    let word_of_bytes = |start: usize| -> BusValue {
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: start,
            },
            LinearTerm::Column {
                coefficient: 256,
                column: start + 1,
            },
            LinearTerm::Column {
                coefficient: 65536,
                column: start + 2,
            },
            LinearTerm::Column {
                coefficient: 16777216,
                column: start + 3,
            },
        ])
    };
    // One 32-bit half of a DWordHL pair (2 halfword columns), LSB first.
    let dword_hl_half = |start: usize| -> BusValue {
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: start,
            },
            LinearTerm::Column {
                coefficient: 65536,
                column: start + 1,
            },
        ])
    };
    let addr_word = |lo_byte: usize| -> BusValue { word_of_bytes(cols::addr(lo_byte)) };

    // 1. ECALL receiver: [ts_lo, ts_hi, syscall_lo32, syscall_hi32].
    interactions.push(BusInteraction::receiver(
        BusId::Ecall,
        Multiplicity::Column(cols::MU_S),
        vec![
            direct(cols::TIMESTAMP_0),
            direct(cols::TIMESTAMP_1),
            BusValue::constant(syscall_lo),
            BusValue::constant(syscall_hi),
        ],
    ));

    // 2. MEMW read of register x10 binding the state address (keccak idiom):
    // [old(8), is_register=1, base=20, value(8), ts(2), w2=1, w4=0, w8=0].
    // In absorb mode x10 is the control-region address and only the group's
    // FIRST row reads it; the rest of the group receives it over the chain bus.
    {
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
            mu_s_or_first.clone(),
            values,
        ));
    }

    // An 8-byte memory access: [old(8), is_register=0, addr(2), value(8), ts(2),
    // w2=0, w4=0, w8=1], addressed by a DWordHL pointer. A pure read passes the
    // same columns as `old` and `value`; a write passes the previous content as
    // `old`. Shared by the state region, the absorb message and `cv_out`.
    let mem_dword =
        |ptr_col: usize, old_cols: &[usize; 8], val_cols: &[usize; 8]| -> Vec<BusValue> {
            let mut values = Vec::with_capacity(24);
            for &c in old_cols {
                values.push(direct(c));
            }
            values.push(BusValue::constant(0)); // is_register
            values.push(dword_hl_half(ptr_col));
            values.push(dword_hl_half(ptr_col + 2));
            for &c in val_cols {
                values.push(direct(c));
            }
            values.push(direct(cols::TIMESTAMP_0));
            values.push(direct(cols::TIMESTAMP_1));
            values.push(BusValue::constant(0));
            values.push(BusValue::constant(0));
            values.push(BusValue::constant(1)); // w8
            values
        };
    let eight_from = |base: usize| -> [usize; 8] { core::array::from_fn(|b| base + b) };

    // 3. MEMW per state dword. Input dwords are pure reads (old = value = input
    // bytes); output dwords write OUT over OLD_OUT.
    //
    // Dwords 0..4 are the chaining value. In absorb mode the SAME interaction
    // reads `cv_in` into the same `h` columns from the same `PTR[0..4]`
    // addresses — the control region's layout was chosen so it could. Only the
    // group's FIRST row reads it; later rows take `h` off the chain bus.
    for k in 0..STATE_DWORDS {
        // (old bytes, value bytes) column bases for this dword.
        let (old_base, val_base): ([usize; 8], [usize; 8]) = if k < IN_DWORDS {
            let c = eight_from(cols::in_word(2 * k, 0));
            (c, c)
        } else {
            let o = k - IN_DWORDS;
            (
                eight_from(cols::old_out(o * 8)),
                eight_from(cols::out_word(2 * o, 0)),
            )
        };
        let mult = if k < CV_DWORDS {
            mu_s_or_first.clone()
        } else {
            Multiplicity::Column(cols::MU_S)
        };
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            mult,
            mem_dword(cols::ptr(k, 0), &old_base, &val_base),
        ));
    }

    // 4. Mixing core + feed-forward: ByteAlu[XOR] per byte, canonical order.
    // Gated on `MU − END`: the END row of an absorb group does no compression
    // and has no witness for these. Single rows have END = 0, so `MU − END = MU`
    // and the single mode's gating is bit-identical to before.
    for xw in &wires.xors {
        for b in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::ByteAlu,
                compressing.clone(),
                vec![
                    BusValue::constant(alu_op::XOR as u64),
                    byte_bus_value(xw.a.byte(b)),
                    byte_bus_value(xw.b.byte(b)),
                    direct(xw.out[b]),
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
                compressing.clone(),
                vec![direct(pair[0]), direct(pair[1])],
            ));
        }
    }

    // 6. Message AreBytes (m is never XORed — DESIGN §4.7/§7.5): 32 pairs.
    // Absorb rows read `m` from memory just as single rows do, and MEMW does not
    // range-check what it carries, so the same check covers both modes.
    for i in 0..16 {
        for p in 0..2 {
            interactions.push(BusInteraction::sender(
                BusId::AreBytes,
                compressing.clone(),
                vec![
                    direct(cols::in_word(8 + i, 2 * p)),
                    direct(cols::in_word(8 + i, 2 * p + 1)),
                ],
            ));
        }
    }

    // 6b. ★ `h` AreBytes on the END row ONLY — the one row whose mixing core is
    // gated off, and therefore the one row where `h`'s bytes are not already
    // range-checked as XOR operands (soundness ledger 14). Without it a prover
    // could satisfy the chain receive with non-canonical bytes summing to the
    // right word and have the `cv_out` write place a "byte" ≥ 256 in memory,
    // which a later read would faithfully return.
    for p in 0..16 {
        interactions.push(BusInteraction::sender(
            BusId::AreBytes,
            Multiplicity::Column(cols::END),
            vec![direct(cols::IN + 2 * p), direct(cols::IN + 2 * p + 1)],
        ));
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

    // 9. IS_HALF range checks on the 22 pointers' halfwords. The control
    // region's 8 are live in both modes; the 14 above it only in single mode,
    // where the executor's `state_addr + 175` check bounds them (an absorb's x10
    // is only bounded at `ctrl + 63`).
    for k in 0..STATE_DWORDS {
        let mult = if k < CTRL_DWORDS {
            Multiplicity::Column(cols::MU)
        } else {
            Multiplicity::Column(cols::MU_S)
        };
        for hw in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::IsHalfword,
                mult.clone(),
                vec![direct(cols::ptr(k, hw))],
            ));
        }
    }

    // =====================================================================
    // 10. Absorb mode
    // =====================================================================

    // 10a. ECALL receiver for the absorb syscall, on the group's FIRST row.
    interactions.push(BusInteraction::receiver(
        BusId::Ecall,
        Multiplicity::Column(cols::FIRST),
        vec![
            direct(cols::TIMESTAMP_0),
            direct(cols::TIMESTAMP_1),
            BusValue::constant(absorb_syscall_lo),
            BusValue::constant(absorb_syscall_hi),
        ],
    ));

    // 10b. Register reads x11/x12/x13 on the FIRST row (x10 is shared with the
    // single mode above). Register accesses put [lo32, hi32] in the first two
    // value slots and zero in the rest.
    //
    // The `hi32 = 0` constants are load-bearing, not padding:
    //   * x12 pins `num_blocks` to the single field element `REMAINING`, so a
    //     count above 2^32 cannot be smuggled in through the high half;
    //   * x13 pins `first_flags < 2^32`, which is what makes it fit the chip's
    //     one-word flags column — the same bound the executor rejects above.
    {
        let reg_read = |base_addr: u64, lo: BusValue, hi: BusValue| -> Vec<BusValue> {
            let mut values = Vec::with_capacity(24);
            values.push(lo.clone());
            values.push(hi.clone());
            for _ in 2..8 {
                values.push(BusValue::constant(0));
            }
            values.push(BusValue::constant(1)); // is_register
            values.push(BusValue::constant(base_addr));
            values.push(BusValue::constant(0));
            values.push(lo);
            values.push(hi);
            for _ in 2..8 {
                values.push(BusValue::constant(0));
            }
            values.push(direct(cols::TIMESTAMP_0));
            values.push(direct(cols::TIMESTAMP_1));
            values.push(BusValue::constant(1)); // w2 (register)
            values.push(BusValue::constant(0));
            values.push(BusValue::constant(0));
            values
        };
        // x11 → message base.
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            reg_read(
                22,
                dword_hl_half(cols::M_BASE),
                dword_hl_half(cols::M_BASE + 2),
            ),
        ));
        // x12 → num_blocks, which IS the group's initial countdown.
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            reg_read(24, direct(cols::REMAINING), BusValue::constant(0)),
        ));
        // x13 → first_flags, landing in the flags column the compression reads.
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            reg_read(
                26,
                word_of_bytes(cols::in_word(27, 0)),
                BusValue::constant(0),
            ),
        ));
    }

    // 10c. The message block: 8 dword reads at MSG_PTR[j], into the same `m`
    // columns the single mode reads out of its state region. Gated on `MU_C` —
    // the END row reads no block.
    for j in 0..MSG_DWORDS {
        let c = eight_from(cols::in_word(8 + 2 * j, 0));
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::MU_C),
            mem_dword(cols::msg_ptr(j, 0), &c, &c),
        ));
    }

    // 10d. `cv_out`: 4 dword writes at PTR[4..8] on the END row, whose value is
    // the `h` the chain receive delivered. Disjoint from `cv_in` by the ABI, so
    // the two never touch one address at one timestamp.
    for k in 0..CV_DWORDS {
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::END),
            mem_dword(
                cols::ptr(CV_OUT_DWORD + k, 0),
                &eight_from(cols::old_out(k * 8)),
                &eight_from(cols::in_word(2 * k, 0)),
            ),
        ));
    }

    // 10e. ★ The chain. Both endpoints come from ONE builder, so the send and
    // the receive cannot drift apart — the failure DESIGN.md §1.1 describes is
    // exactly a drift between them. The tuple leads with the timestamp (which
    // identifies the group) and carries the control address and message base as
    // well as the chaining value: without `ADDR` the END row would need a second
    // x10 read at the group's one timestamp, and without `M_BASE` a prover could
    // point any row at a block of its choosing and still balance MEMW, because
    // reading some other address is a legitimate read.
    let chain_values = |remaining_col: usize, m_base_col: usize, cv_base: usize| -> Vec<BusValue> {
        let mut values = Vec::with_capacity(15);
        values.push(direct(cols::TIMESTAMP_0));
        values.push(direct(cols::TIMESTAMP_1));
        values.push(direct(remaining_col));
        values.push(dword_hl_half(m_base_col));
        values.push(dword_hl_half(m_base_col + 2));
        values.push(addr_word(0));
        values.push(addr_word(4));
        for i in 0..8 {
            values.push(word_of_bytes(cv_base + i * 4));
        }
        values
    };
    interactions.push(BusInteraction::sender(
        BusId::Blake3Absorb,
        Multiplicity::Column(cols::MU_C),
        chain_values(cols::REM_DECR, cols::M_BASE_INCR, cols::OUT),
    ));
    interactions.push(BusInteraction::receiver(
        BusId::Blake3Absorb,
        Multiplicity::Diff(cols::MU_A, cols::FIRST),
        chain_values(cols::REMAINING, cols::M_BASE, cols::IN),
    ));

    // 10f. `END` is DERIVED, never free: ZERO[REMAINING] → END. The lookup also
    // bounds `REMAINING < 2^20` (the BITWISE ZERO table's domain), which is what
    // stops a countdown from wrapping the field back to zero, and with it the
    // "chain of rows with no FIRST and no END" the bus would otherwise balance.
    interactions.push(BusInteraction::sender(
        BusId::Zero,
        Multiplicity::Column(cols::MU_A),
        vec![direct(cols::REMAINING), direct(cols::END)],
    ));

    // 10g. ★ The block cap, in-circuit. `IsB20` admits [0, 2^20), so
    // `REM_DECR · 2^10 ∈ IsB20 ⟺ REM_DECR < 2^10` — no wraparound, since the
    // ZERO lookup already put `REMAINING` under 2^20 and the product is < 2^30.
    // With `FIRST · END = 0` forcing `REMAINING ≠ 0`, the group's block count is
    // exactly the `1..=1024` the executor accepts, rather than inherited from it.
    interactions.push(BusInteraction::sender(
        BusId::IsB20,
        Multiplicity::Column(cols::FIRST),
        vec![BusValue::linear(vec![LinearTerm::Column {
            coefficient: ABSORB_CAP_SCALE as i64,
            column: cols::REM_DECR,
        }])],
    ));

    // 10h. IS_HALF on the absorb-only pointer arithmetic: the message base (live
    // on every absorb row, including END, where it rides the chain), its
    // increment and the 8 message dword pointers.
    for hw in 0..4 {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU_A),
            vec![direct(cols::M_BASE + hw)],
        ));
    }
    for hw in 0..4 {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU_C),
            vec![direct(cols::M_BASE_INCR + hw)],
        ));
    }
    for j in 0..MSG_DWORDS {
        for hw in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::IsHalfword,
                Multiplicity::Column(cols::MU_C),
                vec![direct(cols::msg_ptr(j, hw))],
            ));
        }
    }

    interactions
}

// =========================================================================
// Single-source constraint set
// =========================================================================

/// The BLAKE3 table's transition constraints (848 total):
/// - idx 0..44:    22 pointer `ADD` carry pairs (`ptr[k] = addr + 8k`); the
///   control region's 8 are μ-gated, the 14 above it MU_S-gated;
/// - idx 44, 45:   top-pointer no-overflow, once per mode — `MU_S·carry_1` of
///   `addr + 168 = ptr[21]` and `MU_A·carry_1` of `addr + 56 = ptr[7]`;
/// - idx 46..334:  all 96 add3 groups (sum identity + 2 carry booleanities);
/// - idx 334..430: all 96 add2 expression-carry booleanities;
/// - idx 430..814: all 96 rotations (2 shift identities + 2 recombine each).
///   NOTE the grouping is by op type across the whole row, NOT per G — G #g's
///   16 constraints are scattered across the three bands.
/// - idx 814:      `IS_BIT(MU)` — μ·(1−μ) = 0, ungated. The bus argument pins
///   μ to {0,1} indirectly (the Ecall receive anchors μ>0 rows to a CPU ecall
///   whose ECALL flag is IS_BIT; MEMW's width flags are boolean), but that is
///   an inter-table argument — this makes it local, matching ecsm/commit.
///
/// Absorb mode (idx 815..848):
/// - idx 815..819: `IS_BIT` on MU_S, MU_A, FIRST, END;
/// - idx 819..824: the mode/boundary algebra — `μ = MU_S + MU_A`,
///   `MU_S·MU_A = 0`, `MU_C = MU_A − END`, the boundary lock
///   `(FIRST + END)·(1 − MU_A) = 0`, and `FIRST·END = 0`;
/// - idx 824:      the countdown `MU_C·(REM_DECR + 1 − REMAINING) = 0`;
/// - idx 825..843: `M_BASE + 64 = M_BASE_INCR` and the 8 `msg_ptr[j]` ADD pairs;
/// - idx 843:      `MU_C·carry_1` of `m_base + 56 = msg_ptr[7]`;
/// - idx 844..848: the interior schedule — `t = 0`, `block_len = 64`, and flags
///   zero on every compressing row but the FIRST.
///
/// Every constraint is gated by the mode it belongs to, and the mixing core by
/// `μ − END`. Max degree 3 (the booleanities; identities are degree 2).
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

        // idx 0..44: ptr[k] = addr + 8k. The control region's 8 pointers are
        // gated on μ (both modes address through them); the 14 above it on MU_S,
        // because an absorb's x10 is bounded by the executor at ctrl + 63 and
        // deriving ptr[21] = ctrl + 168 from it would demand a no-overflow the
        // ABI never promised.
        for k in 0..STATE_DWORDS {
            let gate = if k < CTRL_DWORDS {
                cols::MU
            } else {
                cols::MU_S
            };
            emit_add_pair(
                b,
                k * 2,
                &[gate],
                &AddOperand::from_dword_bl(cols::ADDR),
                &AddOperand::constant((k * 8) as i64),
                &AddOperand::from_dword_hl(cols::ptr(k, 0)),
            );
        }

        // idx 44, 45: top-pointer no-overflow, once per mode — MU_S over the
        // 176-byte state region (`addr + 168 = ptr[21]`) and MU_A over the
        // 64-byte control region (`addr + 56 = ptr[7]`). Each mode forbids the
        // wrap only across the range it actually addresses, which is exactly the
        // range the executor's own overflow check binds.
        let mut idx = STATE_DWORDS * 2;
        for (gate, last) in [
            (cols::MU_S, STATE_DWORDS - 1),
            (cols::MU_A, CTRL_DWORDS - 1),
        ] {
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
            let ptr_lo =
                b.main(0, cols::ptr(last, 0)) + b.main(0, cols::ptr(last, 1)) * c65536.clone();
            let ptr_hi = b.main(0, cols::ptr(last, 2)) + b.main(0, cols::ptr(last, 3)) * c65536;

            let inv_2_32 = b.const_base(INV_SHIFT_32);
            let off = b.const_base((8 * last) as u64);
            let carry_0 = (addr_lo + off - ptr_lo) * inv_2_32.clone();
            let carry_1 = (addr_hi + carry_0 - ptr_hi) * inv_2_32;
            let g = b.main(0, gate);
            b.emit_base(idx, g * carry_1);
            idx += 1;
        }

        // Mixing core. Same canonical order as the wire builder records.
        //
        // ★ Gated on `μ − END`, not `μ`. The END row of an absorb group holds no
        // compression witness — it exists to drain the chain and write `cv_out`
        // — so it must not have to satisfy the 814 identities below. A single
        // row has END = 0, so `μ − END = μ` and this mode is untouched.
        let mu = |b: &B| b.main(0, cols::MU) - b.main(0, cols::END);
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

        // IS_BIT(MU) — ungated booleanity, degree 2. See the struct doc for why
        // this is emitted even though the bus argument already pins μ
        // indirectly.
        crate::constraints::templates::emit_is_bit(b, idx, cols::MU, None);
        idx += 1;

        // =================================================================
        // Absorb mode
        // =================================================================

        // The mode and boundary flags are bits, ungated for the same reason
        // IS_BIT(MU) is: the alternative is an inter-table argument.
        for col in [cols::MU_S, cols::MU_A, cols::FIRST, cols::END] {
            crate::constraints::templates::emit_is_bit(b, idx, col, None);
            idx += 1;
        }

        // μ = MU_S + MU_A: the two modes partition the real rows, so every
        // interaction gated on μ fires exactly once per real row whichever
        // mode it is in.
        {
            let root = b.main(0, cols::MU) - b.main(0, cols::MU_S) - b.main(0, cols::MU_A);
            b.emit_base(idx, root);
            idx += 1;
        }

        // ★ The modes are exclusive. Without this a row could set both and
        // receive BOTH `Ecall` tuples — one row answering two syscalls.
        {
            let root = b.main(0, cols::MU_S) * b.main(0, cols::MU_A);
            b.emit_base(idx, root);
            idx += 1;
        }

        // MU_C = MU_A − END: "an absorb row that compresses", as a column so the
        // audited `emit_add_pair` template can gate on it (its condition is a
        // SUM of columns, not an arbitrary expression).
        {
            let root = b.main(0, cols::MU_C) + b.main(0, cols::END) - b.main(0, cols::MU_A);
            b.emit_base(idx, root);
            idx += 1;
        }

        // ★ (FIRST + END)·(1 − MU_A) = 0 — COMMIT's boundary lock (commit.rs
        // idx 3), retargeted from μ to MU_A. A padding row (μ = 0) or a
        // single-compression row cannot mint a group boundary: FIRST would let
        // it receive a second `Ecall`, END would let it write a `cv_out`.
        {
            let one = b.one();
            let root =
                (b.main(0, cols::FIRST) + b.main(0, cols::END)) * (one - b.main(0, cols::MU_A));
            b.emit_base(idx, root);
            idx += 1;
        }

        // FIRST · END = 0 — no zero-block group. The executor rejects
        // `num_blocks = 0`; a group that was its own END row would copy `cv_in`
        // to `cv_out` and prove an ecall the VM semantics never accept.
        {
            let root = b.main(0, cols::FIRST) * b.main(0, cols::END);
            b.emit_base(idx, root);
            idx += 1;
        }

        // The countdown: REM_DECR + 1 = REMAINING on every compressing row.
        // Combined with the chain (which carries REM_DECR forward as the next
        // row's REMAINING) and ZERO[REMAINING] → END, the group has exactly
        // `num_blocks` compressing rows and one END row.
        {
            let one = b.one();
            let root = b.main(0, cols::MU_C)
                * (b.main(0, cols::REM_DECR) + one - b.main(0, cols::REMAINING));
            b.emit_base(idx, root);
            idx += 1;
        }

        // Message pointer arithmetic, MU_C-gated: the next block's base and the
        // 8 dword pointers into this one.
        emit_add_pair(
            b,
            idx,
            &[cols::MU_C],
            &AddOperand::from_dword_hl(cols::M_BASE),
            &AddOperand::constant(64),
            &AddOperand::from_dword_hl(cols::M_BASE_INCR),
        );
        idx += 2;
        for j in 0..MSG_DWORDS {
            emit_add_pair(
                b,
                idx,
                &[cols::MU_C],
                &AddOperand::from_dword_hl(cols::M_BASE),
                &AddOperand::constant((j * 8) as i64),
                &AddOperand::from_dword_hl(cols::msg_ptr(j, 0)),
            );
            idx += 2;
        }

        // No-overflow on the block's own 64 bytes (`m_base + 56 = msg_ptr[7]`),
        // which the executor's `msg_addr + 64·num_blocks − 1` check bounds.
        //
        // ★ Deliberately NOT applied to M_BASE_INCR: a message ending exactly at
        // the top of memory makes the last row's `m_base + 64` equal 2^64, and
        // that value only rides the chain to the END row, which never reads it.
        // Forbidding the wrap there would reject an absorb the ABI accepts.
        {
            let c65536 = b.const_base(65536);
            let last = MSG_DWORDS - 1;
            let base_lo = b.main(0, cols::M_BASE) + b.main(0, cols::M_BASE + 1) * c65536.clone();
            let base_hi =
                b.main(0, cols::M_BASE + 2) + b.main(0, cols::M_BASE + 3) * c65536.clone();
            let ptr_lo = b.main(0, cols::msg_ptr(last, 0))
                + b.main(0, cols::msg_ptr(last, 1)) * c65536.clone();
            let ptr_hi =
                b.main(0, cols::msg_ptr(last, 2)) + b.main(0, cols::msg_ptr(last, 3)) * c65536;
            let inv_2_32 = b.const_base(INV_SHIFT_32);
            let off = b.const_base((8 * last) as u64);
            let carry_0 = (base_lo + off - ptr_lo) * inv_2_32.clone();
            let carry_1 = (base_hi + carry_0 - ptr_hi) * inv_2_32;
            let c = b.main(0, cols::MU_C);
            b.emit_base(idx, c * carry_1);
            idx += 1;
        }

        // ★ The interior schedule, constrained rather than assumed: every
        // absorbed block runs at `t = 0` with `block_len = 64`, and only the
        // FIRST block carries flags. These four columns are the compression
        // inputs an absorb row does NOT read from memory, so nothing else pins
        // them — a prover free to choose them could absorb the caller's blocks
        // under a framing that produces a different digest and still balance
        // every bus.
        //
        // Their bytes are range-checked as XOR operands by the mixing core,
        // which is live on exactly the rows these gate, so a zero word
        // expression means four zero bytes.
        for (word, want) in [(24usize, 0u64), (25, 0), (26, 64)] {
            let c = b.main(0, cols::MU_C);
            let expr = word_expr(b, &WordRef::Cols(word_cols(cols::in_word(word, 0))));
            let target = b.const_base(want);
            let root = c * (expr - target);
            b.emit_base(idx, root);
            idx += 1;
        }
        {
            // Flags are zero on every compressing row but the first; the FIRST
            // row's value is bound by the x13 register read.
            let gate = b.main(0, cols::MU_C) - b.main(0, cols::FIRST);
            let flags = word_expr(b, &WordRef::Cols(word_cols(cols::in_word(27, 0))));
            let root = gate * flags;
            b.emit_base(idx, root);
        }
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

/// ★ The state the guest hands the accelerator is the layout the executor
/// reads back.
///
/// `Blake3Chain`'s riscv64 arm marshals a compression into
/// `crypto::hash::blake3::chain::pack_syscall_state`'s 22 dwords and reads the
/// result out of `unpack_syscall_out`. That marshaling is the last unchecked
/// link between the host prover's hash and the guest's: `executor_primitive_parity`
/// above gates the *compression*, and `crypto` gates the chain framing, but a
/// transposed dword or a swapped counter half would leave both of those green
/// and still make the guest hash differently — R5's invisible failure, visible
/// only as in-guest proof rejection.
///
/// It needs no guest to check. The executor's syscall handler is ordinary host
/// code driven by a real `EcallEbreak`, so this lays the packed dwords into a VM
/// `Memory` exactly as the guest's `&mut [u64; 22]` presents them, runs the
/// instruction, and unpacks the result — closing the loop through the same two
/// functions the guest calls.
#[cfg(test)]
mod executor_syscall_packing {
    use crypto::hash::blake3::chain::{SYSCALL_OUT_DWORD, pack_syscall_state, unpack_syscall_out};
    use crypto::hash::blake3::{BLAKE3_SIX_ROUNDS, blake3_compress_rounds};
    use executor::vm::instruction::decoding::Instruction;
    use executor::vm::instruction::execution::BLAKE3_SYSCALL_NUMBER;
    use executor::vm::memory::Memory;
    use executor::vm::registers::Registers;

    /// Pre-filled into the output dwords, so a packing that pointed the
    /// accelerator at the wrong part of the region cannot pass on stale data.
    const SENTINEL: u64 = 0xDEAD_BEEF_DEAD_BEEF;

    /// One compression the way a guest performs it: pack, ecall, unpack.
    fn through_the_accelerator(
        h: &[u32; 8],
        m: &[u32; 16],
        t: u64,
        block_len: u32,
        flags: u32,
    ) -> [u32; 16] {
        let addr = 0x1000u64;
        let mut memory = Memory::default();
        let mut registers = Registers::default();

        let mut state = pack_syscall_state(h, m, t, block_len, flags);
        for dword in &mut state[SYSCALL_OUT_DWORD..] {
            *dword = SENTINEL;
        }
        for (k, dword) in state.iter().enumerate() {
            memory
                .store_doubleword(addr + (k as u64) * 8, *dword)
                .unwrap();
        }

        let mut pc = 0;
        registers.write(17, BLAKE3_SYSCALL_NUMBER).unwrap();
        registers.write(10, addr).unwrap();
        Instruction::EcallEbreak
            .run(&mut pc, &mut registers, &mut memory)
            .unwrap();

        for (k, dword) in state.iter_mut().enumerate() {
            *dword = memory.load_doubleword(addr + (k as u64) * 8).unwrap();
        }
        for (k, dword) in state.iter().enumerate().skip(SYSCALL_OUT_DWORD) {
            assert_ne!(*dword, SENTINEL, "output dword {k} was never written");
        }
        unpack_syscall_out(&state)
    }

    #[test]
    fn the_packed_state_is_what_the_executor_reads() {
        // A cheap deterministic stream; no rand dependency in this crate.
        let mut z = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            z ^= z << 13;
            z ^= z >> 7;
            z ^= z << 17;
            z as u32
        };

        // The flag shapes `Blake3Chain` emits — first block, interior, and last
        // with a partial `block_len` — plus two counters setting exactly one
        // half. The chain always sends `t = 0`, so only these pin the split
        // order the packing chose.
        let shapes: [(u64, u32, u32); 5] = [
            (0, 64, 1),                        // CHUNK_START
            (0, 64, 0),                        // interior
            (0, 7, 2 | 8),                     // CHUNK_END | ROOT, partial block
            (0x0000_0000_FFFF_FFFF, 64, 0),    // low counter half only
            (0xFFFF_FFFF_0000_0000, 0, !0u32), // high half; degenerate len/flags
        ];

        for (t, block_len, flags) in shapes {
            for _ in 0..16 {
                let h: [u32; 8] = core::array::from_fn(|_| next());
                let m: [u32; 16] = core::array::from_fn(|_| next());
                assert_eq!(
                    through_the_accelerator(&h, &m, t, block_len, flags),
                    blake3_compress_rounds(&h, &m, t, block_len, flags, BLAKE3_SIX_ROUNDS),
                    "packed state mismatch at t={t}, block_len={block_len}, flags={flags}"
                );
            }
        }
    }

    /// CONTROL: the check above discriminates the *layout*, not merely the
    /// compression. The counter's two halves share one dword and are the
    /// likeliest thing to transpose — and the chain, which only ever sends
    /// `t = 0`, would never notice. They must be distinguishable, or agreement
    /// above would survive the swap.
    #[test]
    fn the_packing_check_is_layout_sensitive() {
        let h: [u32; 8] = core::array::from_fn(|i| (i as u32).wrapping_mul(2_654_435_761));
        let m: [u32; 16] =
            core::array::from_fn(|i| (i as u32).wrapping_mul(40_503).wrapping_add(7));
        let t = 0x0000_0001_0000_0000u64;

        assert_ne!(
            through_the_accelerator(&h, &m, t, 64, 1),
            blake3_compress_rounds(&h, &m, t.rotate_left(32), 64, 1, BLAKE3_SIX_ROUNDS),
            "the counter halves must be distinguishable"
        );
    }
}

/// ★ The absorb ecall folds blocks the way `Blake3Chain` folds them.
///
/// The absorb syscall exists so a long message costs one ecall instead of one
/// per 64 bytes, and paying for that means the executor applies a flag and
/// `block_len` schedule of its own: `t = 0` and `block_len = 64` on every block,
/// the caller's `first_flags` on block 0 and nothing after it
/// ([`executor::vm::instruction::execution::blake3_absorb_chain_6round`]).
/// `executor` cannot call `crypto`, so that schedule is a SECOND statement of
/// the framing `Blake3Chain` owns — precisely the thing PA-PLAN §1.4 forbids
/// leaving unchecked, and precisely the thing that fails invisibly: a guest that
/// absorbs under the wrong flags produces a perfectly valid proof of a digest
/// nobody else computes.
///
/// This is the gate. It drives real `EcallEbreak` instructions through the
/// executor exactly as the guest's `bulk_absorb` arm does, finishes the hash
/// with `Blake3Chain` itself, and compares against `blake3_chain_rounds` — over
/// every length from the empty message past two blocks, plus the boundaries
/// that discriminate the schedule. Nothing here restates the framing: the run's
/// first flag word comes from [`block_flags`], and resuming after the run goes
/// through `Blake3Chain::resume_with_rounds`, the same call the guest makes.
#[cfg(test)]
mod executor_absorb_parity {
    use crypto::hash::blake3::chain::{
        ABSORB_CV_OUT_DWORD, BLOCK_LEN, Blake3Chain, blake3_chain_rounds, block_flags,
        bulk_absorb_blocks, kat_message_byte, pack_absorb_ctrl, unpack_absorb_cv,
    };
    use crypto::hash::blake3::{BLAKE3_IV, BLAKE3_SIX_ROUNDS};
    use executor::vm::instruction::decoding::Instruction;
    use executor::vm::instruction::execution::BLAKE3_ABSORB_SYSCALL_NUMBER;
    use executor::vm::memory::Memory;
    use executor::vm::registers::Registers;

    /// Pre-filled into `cv_out`, so an absorb that wrote nowhere near the
    /// control region cannot pass on stale bytes.
    const SENTINEL: u64 = 0xDEAD_BEEF_DEAD_BEEF;
    /// Disjoint by construction — the ecall rejects overlapping regions, and
    /// 0x2000 is past the control region's 64 bytes at 0x1000.
    const CTRL_ADDR: u64 = 0x1000;
    const MSG_ADDR: u64 = 0x2000;

    fn message(len: usize) -> Vec<u8> {
        (0..len).map(kat_message_byte).collect()
    }

    /// One absorb the way a guest performs it: pack the control region, lay the
    /// blocks out in memory, ecall, read the chaining value back.
    fn absorb_through_the_accelerator(
        cv_in: &[u32; 8],
        blocks: &[u8],
        first_flags: u32,
    ) -> [u32; 8] {
        assert!(blocks.len().is_multiple_of(BLOCK_LEN) && !blocks.is_empty());
        let mut memory = Memory::default();
        let mut registers = Registers::default();

        let mut ctrl = pack_absorb_ctrl(cv_in);
        for dword in &mut ctrl[ABSORB_CV_OUT_DWORD..] {
            *dword = SENTINEL;
        }
        for (k, dword) in ctrl.iter().enumerate() {
            memory
                .store_doubleword(CTRL_ADDR + (k as u64) * 8, *dword)
                .unwrap();
        }
        for (k, chunk) in blocks.chunks_exact(8).enumerate() {
            let dword = u64::from_le_bytes(chunk.try_into().unwrap());
            memory
                .store_doubleword(MSG_ADDR + (k as u64) * 8, dword)
                .unwrap();
        }

        let mut pc = 0;
        registers.write(17, BLAKE3_ABSORB_SYSCALL_NUMBER).unwrap();
        registers.write(10, CTRL_ADDR).unwrap();
        registers.write(11, MSG_ADDR).unwrap();
        registers
            .write(12, (blocks.len() / BLOCK_LEN) as u64)
            .unwrap();
        registers.write(13, first_flags as u64).unwrap();
        Instruction::EcallEbreak
            .run(&mut pc, &mut registers, &mut memory)
            .unwrap();

        for (k, dword) in ctrl.iter_mut().enumerate() {
            *dword = memory.load_doubleword(CTRL_ADDR + (k as u64) * 8).unwrap();
        }
        for (k, dword) in ctrl.iter().enumerate().skip(ABSORB_CV_OUT_DWORD) {
            assert_ne!(*dword, SENTINEL, "cv_out dword {k} was never written");
        }
        unpack_absorb_cv(&ctrl)
    }

    /// `Blake3Chain::update`'s bulk path, driven against the real ecall: absorb
    /// whole blocks while the schedule will take them, then let the hasher
    /// finish the tail — which is where the final block's `CHUNK_END | ROOT`
    /// and true byte count stay.
    fn chain_through_the_accelerator(msg: &[u8]) -> [u8; 32] {
        let mut cv = BLAKE3_IV;
        let mut started = false;
        let mut rest = msg;
        loop {
            // VM addresses here are 8-aligned by construction.
            let blocks = bulk_absorb_blocks(0, rest.len(), true);
            if blocks == 0 {
                break;
            }
            // What `Blake3Chain::flags(false)` evaluates to at this point: the
            // closed form at the next block's index, read rather than restated.
            let index = usize::from(started);
            let first_flags = block_flags(index, index + 2);
            let taken = blocks * BLOCK_LEN;
            cv = absorb_through_the_accelerator(&cv, &rest[..taken], first_flags);
            started = true;
            rest = &rest[taken..];
        }
        let mut chain = if started {
            Blake3Chain::resume_with_rounds(cv, BLAKE3_SIX_ROUNDS)
        } else {
            Blake3Chain::with_rounds(BLAKE3_SIX_ROUNDS)
        };
        chain.update(rest);
        chain.finalize_digest()
    }

    /// ★ Exhaustive over every length through two blocks and past, plus the
    /// lengths PA-PLAN §1.7.4 singles out. Anything the executor gets wrong
    /// about the interior schedule moves a digest here.
    #[test]
    fn the_absorb_ecall_is_the_chain() {
        let lengths = (0..=300usize).chain([511, 512, 513, 1023, 1024, 1025, 1088, 4096]);
        for len in lengths {
            let msg = message(len);
            assert_eq!(
                chain_through_the_accelerator(&msg),
                blake3_chain_rounds(&msg, BLAKE3_SIX_ROUNDS),
                "the absorb path must be the chain, at length {len}"
            );
        }
    }

    /// CONTROL: the test above must actually be taking the ecall. If
    /// `bulk_absorb_blocks` returned 0 everywhere, every assertion would still
    /// pass — `chain_through_the_accelerator` would degrade to plain
    /// `Blake3Chain` and gate nothing at all.
    #[test]
    fn the_parity_test_really_reaches_the_accelerator() {
        assert_eq!(bulk_absorb_blocks(0, 65, true), 1);
        assert_eq!(bulk_absorb_blocks(0, 1024, true), 15);
        // And an absorb changes the chaining value away from the IV, so the
        // ecall is doing work rather than copying `cv_in` through.
        let msg = message(128);
        let cv = absorb_through_the_accelerator(&BLAKE3_IV, &msg[..64], block_flags(0, 2));
        assert_ne!(cv, BLAKE3_IV, "the absorb must fold the block in");
    }

    /// ★ A message past the block cap takes SEVERAL absorb ecalls, and the
    /// second one is where the flag schedule can go wrong.
    ///
    /// Every other case here fits in one ecall, so `first_flags` is always
    /// `CHUNK_START` and the "0 on every later block" half of the schedule is
    /// exercised only *within* a run. Past `ABSORB_MAX_BLOCKS` the guest calls
    /// again with `started` already true, and the second run's first block must
    /// carry NO flags — a run that re-sent `CHUNK_START`, or an executor that
    /// applied it because it is block 0 of *its* run, would produce a valid
    /// proof of the wrong digest and nothing else here would notice.
    ///
    /// `(ABSORB_MAX_BLOCKS + 2) * 64` forces exactly two ecalls. Not the
    /// *smallest* message that does — that is `ABSORB_MAX_BLOCKS * 64 + 1`,
    /// one byte past a full cap's worth — but it leaves the second run a clean
    /// whole block, which is the shape worth pinning.
    #[test]
    fn a_message_past_the_block_cap_takes_several_ecalls() {
        use crypto::hash::blake3::chain::ABSORB_MAX_BLOCKS;

        let len = (ABSORB_MAX_BLOCKS + 2) * BLOCK_LEN;
        // Two full runs plus the final block that `finalize_digest` keeps.
        assert_eq!(bulk_absorb_blocks(0, len, true), ABSORB_MAX_BLOCKS);
        let after_first = len - ABSORB_MAX_BLOCKS * BLOCK_LEN;
        assert_eq!(bulk_absorb_blocks(0, after_first, true), 1);

        let msg = message(len);
        assert_eq!(
            chain_through_the_accelerator(&msg),
            blake3_chain_rounds(&msg, BLAKE3_SIX_ROUNDS),
            "a multi-ecall absorb must still be the chain"
        );
    }

    /// ★ The two crates' copies of the absorb ABI's shape are the same numbers.
    ///
    /// `crypto` states the control-region layout and the block cap for the guest
    /// (`ABSORB_*`), and `executor` states them again for the handler
    /// (`BLAKE3_ABSORB_*`), because neither crate can see the other's. Nothing
    /// above catches drift: the parity test drives well-formed absorbs, so a
    /// guest whose cap exceeded the executor's would fault only on the long
    /// messages no test reaches. `prover` sees both crates, so this is where
    /// they can be pinned.
    ///
    /// The guest's own control array is already pinned to the syscall wrapper's
    /// by the type system (`[u64; N]` on both sides) when the guest compiles;
    /// this covers the half no type can see.
    #[test]
    fn the_two_crates_agree_on_the_absorb_abi() {
        use crypto::hash::blake3::chain::{ABSORB_CTRL_DWORDS, ABSORB_MAX_BLOCKS};
        use executor::vm::instruction::execution::{
            BLAKE3_ABSORB_CTRL_DWORDS, BLAKE3_ABSORB_CV_OUT_DWORD, BLAKE3_ABSORB_MAX_BLOCKS,
            BLAKE3_BLOCK_BYTES,
        };

        assert_eq!(ABSORB_CTRL_DWORDS as u64, BLAKE3_ABSORB_CTRL_DWORDS);
        assert_eq!(ABSORB_CV_OUT_DWORD as u64, BLAKE3_ABSORB_CV_OUT_DWORD);
        assert_eq!(ABSORB_MAX_BLOCKS as u64, BLAKE3_ABSORB_MAX_BLOCKS);
        assert_eq!(BLOCK_LEN as u64, BLAKE3_BLOCK_BYTES);
        // `cv_out` starts past `cv_in`'s four dwords and the region holds both —
        // the disjointness the single-timestamp memory argument depends on.
        assert_eq!(BLAKE3_ABSORB_CV_OUT_DWORD, 4);
        assert_eq!(BLAKE3_ABSORB_CTRL_DWORDS, 8);
    }

    /// CONTROL: the parity is sensitive to the run's FIRST flag word — the one
    /// value the guest marshals rather than the executor deriving it. Absorbing
    /// under interior flags where `CHUNK_START` belongs must not agree, or the
    /// gate would survive dropping the flag entirely.
    #[test]
    fn the_absorb_parity_is_first_flag_sensitive() {
        let msg = message(128);
        let honest = absorb_through_the_accelerator(&BLAKE3_IV, &msg[..64], block_flags(0, 2));
        let tampered = absorb_through_the_accelerator(&BLAKE3_IV, &msg[..64], 0);
        assert_ne!(honest, tampered, "CHUNK_START must change the absorb");
    }

    /// CONTROL: the executor applies `first_flags` to block 0 ONLY. A run of
    /// two blocks must differ from one where the flag were applied to both, or
    /// "and nothing after it" would be untested.
    #[test]
    fn the_absorb_applies_the_first_flag_to_one_block_only() {
        let msg = message(128);
        let two_at_once =
            absorb_through_the_accelerator(&BLAKE3_IV, &msg[..128], block_flags(0, 2));
        // The same two blocks as two separate runs, each carrying the flag.
        let first = absorb_through_the_accelerator(&BLAKE3_IV, &msg[..64], block_flags(0, 2));
        let flagged_twice =
            absorb_through_the_accelerator(&first, &msg[64..128], block_flags(0, 2));
        assert_ne!(
            two_at_once, flagged_twice,
            "the flag must land on block 0 alone"
        );
        // ...and chaining two single-block runs with the interior flag on the
        // second IS the two-block run: that is what "chained" means.
        let flagged_once = absorb_through_the_accelerator(&first, &msg[64..128], 0);
        assert_eq!(two_at_once, flagged_once, "the run must chain its blocks");
    }
}

/// ★ The absorb mode's soundness suite: every claim the chip makes, with an
/// honest-path control beside each attack.
///
/// Two levels, because the mode's guarantees live at two levels:
///
/// * **Row-local constraints** — checked with the cheap `eval_main_row` idiom
///   (`prover/src/tests/hint_tests.rs`): generate a trace, assert every
///   constraint is zero on every row, tamper one cell, assert a constraint
///   fires. This covers the schedule, the boundary algebra and the pointer
///   arithmetic.
/// * **Bus tuples** — the chain, the `Zero` derivation of END and the `Ecall`
///   anchoring are cross-row properties no constraint can see. They are checked
///   by evaluating the table's own `bus_interactions` over the trace and
///   asserting the multiset of `Blake3Absorb` tuples cancels, which is exactly
///   what LogUp asks of them.
#[cfg(test)]
mod absorb_tests {
    use super::*;
    use math::field::element::FieldElement;
    use stark::constraints::builder::ProverEvalFolder;
    use stark::frame::Frame;
    use stark::lookup::{BusInteraction, Multiplicity};
    use stark::table::TableView;
    use stark::traits::TransitionEvaluationContext;
    use std::collections::BTreeMap;

    // ---------------------------------------------------------------------
    // Fixtures and helpers
    // ---------------------------------------------------------------------

    const CTRL: u64 = 0x1000;
    const MSG: u64 = 0x8000;

    /// A deterministic absorb of `n` blocks. No `rand` dependency in this crate.
    fn absorb_op(timestamp: u64, n: usize, first_flags: u32) -> Blake3AbsorbOperation {
        let mut z = 0x243f_6a88_85a3_08d3u64 ^ (n as u64);
        let mut next = move || {
            z ^= z << 13;
            z ^= z >> 7;
            z ^= z << 17;
            z as u32
        };
        Blake3AbsorbOperation {
            timestamp,
            ctrl_addr: CTRL,
            msg_addr: MSG,
            first_flags,
            cv_in: core::array::from_fn(|_| next()),
            blocks: (0..n).map(|_| core::array::from_fn(|_| next())).collect(),
            old_cv_out: core::array::from_fn(|i| (i as u8).wrapping_mul(7)),
        }
    }

    fn single_op(timestamp: u64) -> Blake3Operation {
        let h: [u32; 8] = core::array::from_fn(|i| 0x9E3779B9u32.wrapping_mul(i as u32 + 1));
        let m: [u32; 16] = core::array::from_fn(|i| 0x85EBCA6Bu32.wrapping_mul(i as u32 + 7));
        let (t, block_len, flags) = (0x0123_4567u64, 64u32, 11u32);
        Blake3Operation {
            timestamp,
            state_addr: 0x2000,
            h,
            m,
            t,
            block_len,
            flags,
            old_out: [0; 64],
            out: executor::vm::instruction::execution::blake3_compress_6round(
                &h, &m, t, block_len, flags,
            ),
        }
    }

    fn trace_of(
        singles: &[Blake3Operation],
        absorbs: &[Blake3AbsorbOperation],
    ) -> TraceTable<GoldilocksField, GoldilocksExtension> {
        generate_blake3_trace(singles, absorbs)
    }

    fn row_of(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<FE> {
        (0..cols::NUM_COLUMNS)
            .map(|c| *trace.main_table.get(row, c))
            .collect()
    }

    /// Evaluate the BLAKE3 constraint set on one main-trace row.
    fn eval_main_row(main: Vec<FE>) -> Vec<FE> {
        let n = Blake3Constraints.meta().len();
        let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
            vec![main],
            vec![vec![]],
        )]);
        let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
        let offset_e = FieldElement::<GoldilocksExtension>::zero();
        let ctx =
            TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
        let mut base = vec![FE::zero(); n];
        let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
        let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
        Blake3Constraints.eval(&mut folder);
        base
    }

    /// Assert every constraint holds on every row of `rows` (all of them when
    /// `rows` is `None` — the constraint set rebuilds the wire flow per call, so
    /// large traces are sampled instead).
    fn assert_constraints_hold(
        trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
        rows: Option<&[usize]>,
    ) {
        let all: Vec<usize> = (0..trace.num_rows()).collect();
        for &row in rows.unwrap_or(&all) {
            for (i, v) in eval_main_row(row_of(trace, row)).iter().enumerate() {
                assert_eq!(*v, FE::zero(), "constraint {i} must hold at row {row}");
            }
        }
    }

    /// Evaluate a multiplicity expression on one row.
    fn multiplicity_at(m: &Multiplicity, row: &[FE]) -> FE {
        match m {
            Multiplicity::One => FE::one(),
            Multiplicity::Column(c) => row[*c],
            Multiplicity::Sum(a, b) => row[*a] + row[*b],
            Multiplicity::Negated(c) => FE::one() - row[*c],
            Multiplicity::Diff(a, b) => row[*a] - row[*b],
            Multiplicity::Sum3(a, b, c) => row[*a] + row[*b] + row[*c],
            Multiplicity::Linear(_) => unreachable!("no Linear multiplicity in this chip"),
        }
    }

    /// The net multiset of tuples on `bus`, as `key → (senders − receivers)`.
    /// An honest trace leaves this empty: that IS the LogUp condition.
    fn bus_net(
        trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
        bus: BusId,
    ) -> BTreeMap<String, i64> {
        let interactions: Vec<BusInteraction> = bus_interactions()
            .into_iter()
            .filter(|i| i.bus_id == u64::from(bus))
            .collect();
        let mut net: BTreeMap<String, i64> = BTreeMap::new();
        for row_idx in 0..trace.num_rows() {
            let row = row_of(trace, row_idx);
            for it in &interactions {
                let mult = multiplicity_at(&it.multiplicity, &row);
                if mult == FE::zero() {
                    continue;
                }
                assert_eq!(mult, FE::one(), "chip multiplicities are 0 or 1");
                let values: Vec<FE> = it
                    .values
                    .iter()
                    .flat_map(|v| v.combine_from::<GoldilocksField, _>(|c| row[c]))
                    .collect();
                let key = format!("{values:?}");
                *net.entry(key).or_insert(0) += if it.is_sender { 1 } else { -1 };
            }
        }
        net.retain(|_, v| *v != 0);
        net
    }

    /// Number of rows whose `Ecall` receive fires, split by syscall.
    fn ecall_receives(trace: &TraceTable<GoldilocksField, GoldilocksExtension>) -> (usize, usize) {
        let mut single = 0;
        let mut absorb = 0;
        for row_idx in 0..trace.num_rows() {
            let row = row_of(trace, row_idx);
            if row[cols::MU_S] == FE::one() {
                single += 1;
            }
            if row[cols::FIRST] == FE::one() {
                absorb += 1;
            }
        }
        (single, absorb)
    }

    // ---------------------------------------------------------------------
    // Honest-path controls
    // ---------------------------------------------------------------------

    /// ★ CONTROL for everything below: a trace holding both modes at once — a
    /// single compression, a one-block absorb and a three-block absorb —
    /// satisfies every constraint on every row, padding included.
    #[test]
    fn both_modes_share_the_table() {
        let trace = trace_of(&[single_op(4)], &[absorb_op(8, 1, 1), absorb_op(12, 3, 1)]);
        assert_constraints_hold(&trace, None);
        assert!(
            bus_net(&trace, BusId::Blake3Absorb).is_empty(),
            "the honest chain must balance"
        );
        assert_eq!(ecall_receives(&trace), (1, 2), "one Ecall per ecall");
    }

    /// The group is `num_blocks + 1` rows: N compressions and one END row that
    /// does no work. This is CHANGE 1 of the design, and the row budget the
    /// 64 KiB cap is chosen for depends on it.
    #[test]
    fn a_group_is_n_plus_one_rows() {
        for n in [1usize, 2, 5] {
            let rows = expand_absorb(&absorb_op(4, n, 1));
            assert_eq!(rows.len(), n + 1);
            assert!(rows[0].first && !rows[0].end);
            assert!(rows[n].end && !rows[n].first);
            assert_eq!(rows[0].remaining, n as u32);
            assert_eq!(rows[n].remaining, 0);
            for (i, r) in rows.iter().enumerate() {
                assert_eq!(r.flags == 0, i > 0, "flags land on block 0 alone");
                assert_eq!(r.m_base, MSG + 64 * i as u64);
            }
        }
    }

    /// ★ The chip folds blocks the way the executor's ecall does. Without this
    /// the chip could be internally consistent and still prove a different
    /// digest from the one the VM semantics define.
    #[test]
    fn the_expansion_is_the_executors_absorb() {
        use executor::vm::instruction::execution::blake3_absorb_chain_6round;
        for n in [1usize, 2, 7] {
            for flags in [0u32, 1, 1 | 8] {
                let op = absorb_op(4, n, flags);
                let flat: Vec<[u32; 16]> = op.blocks.clone();
                let rows = expand_absorb(&op);
                assert_eq!(
                    rows.last().unwrap().h,
                    blake3_absorb_chain_6round(&op.cv_in, &flat, flags),
                    "the chip's chain must be the executor's, n={n} flags={flags}"
                );
            }
        }
    }

    /// ★ The single-compression mode is untouched. Every absorb column is zero
    /// on a single row, so no absorb interaction and no absorb constraint can
    /// fire in that mode — the property a reviewer should check first.
    #[test]
    fn the_single_mode_is_untouched() {
        let trace = trace_of(&[single_op(4), single_op(8)], &[]);
        assert_constraints_hold(&trace, None);
        assert!(bus_net(&trace, BusId::Blake3Absorb).is_empty());
        for row_idx in 0..2 {
            let row = row_of(&trace, row_idx);
            assert_eq!(row[cols::MU], FE::one());
            assert_eq!(row[cols::MU_S], FE::one());
            for (c, v) in row.iter().enumerate().skip(cols::MU_A) {
                assert_eq!(*v, FE::zero(), "absorb column {c} must be 0 in single mode");
            }
        }
    }

    /// Both ends of the legal block range prove: the degenerate one-block group
    /// and a group at the 1 024-block cap.
    #[test]
    fn the_cap_boundary_and_the_degenerate_group_both_hold() {
        let one = trace_of(&[], &[absorb_op(4, 1, 1)]);
        assert_constraints_hold(&one, None);
        assert!(bus_net(&one, BusId::Blake3Absorb).is_empty());

        let n = ABSORB_MAX_BLOCKS as usize;
        let full = trace_of(&[], &[absorb_op(4, n, 1)]);
        // The row budget the 64 KiB cap exists for: 1 024 compressions + END,
        // padded to 2 048. This is the largest group the ABI admits.
        assert_eq!(n + 1, 1025, "rows per ecall at the cap");
        assert_eq!(full.num_rows(), 2048);
        assert_eq!(
            expand_absorb(&absorb_op(4, n, 1)).len(),
            1025,
            "the group really is at the cap"
        );
        // The constraint set rebuilds the wire flow per call, so sample the
        // boundaries rather than all 2 048 rows.
        assert_constraints_hold(&full, Some(&[0, 1, n - 1, n, n + 1, full.num_rows() - 1]));
        assert!(
            bus_net(&full, BusId::Blake3Absorb).is_empty(),
            "a group at the cap must still chain"
        );
    }

    // ---------------------------------------------------------------------
    // Falsification: the chain
    // ---------------------------------------------------------------------

    /// ★ A tampered chaining value breaks the chain. The bus is what ties a
    /// row's output to the next row's input; nothing row-local can see it.
    #[test]
    fn a_tampered_chained_cv_unbalances_the_chain() {
        let mut trace = trace_of(&[], &[absorb_op(4, 3, 1)]);
        assert!(bus_net(&trace, BusId::Blake3Absorb).is_empty(), "control");
        // Row 1's incoming chaining value, one byte off.
        let c = cols::in_word(0, 0);
        let old = *trace.main_table.get(1, c);
        trace.main_table.set_fe(1, c, old + FE::one());
        assert_eq!(
            bus_net(&trace, BusId::Blake3Absorb).len(),
            2,
            "the send it no longer matches and the receive nobody sent"
        );
    }

    /// ★ Running past the end: drop the END row and the last compression's send
    /// has no receiver. This is why END must exist at all.
    #[test]
    fn a_group_without_its_end_row_leaves_a_dangling_send() {
        let mut trace = trace_of(&[], &[absorb_op(4, 2, 1)]);
        assert!(bus_net(&trace, BusId::Blake3Absorb).is_empty(), "control");
        // Blank the END row (row 2) exactly as a padding row.
        for c in 0..cols::NUM_COLUMNS {
            trace.main_table.set_fe(2, c, FE::zero());
        }
        assert_eq!(
            bus_net(&trace, BusId::Blake3Absorb).len(),
            1,
            "the last compression's send must dangle"
        );
    }

    /// ★ END is derived, not chosen: `ZERO[REMAINING] → END`. Claiming END early
    /// leaves the chip sending a ZERO tuple the BITWISE table does not contain,
    /// so the range-check bus cannot balance.
    #[test]
    fn an_early_end_sends_a_zero_tuple_that_does_not_exist() {
        // Every ZERO tuple the chip sends, as (input, claimed output). BITWISE
        // answers `input == 0`, so a tuple disagreeing with that is a lookup no
        // row of the precomputed table can satisfy.
        let zero_tuples = |trace: &TraceTable<GoldilocksField, GoldilocksExtension>| {
            let sends: Vec<BusInteraction> = bus_interactions()
                .into_iter()
                .filter(|i| i.bus_id == u64::from(BusId::Zero))
                .collect();
            let mut out = Vec::new();
            for row_idx in 0..trace.num_rows() {
                let row = row_of(trace, row_idx);
                for it in &sends {
                    if multiplicity_at(&it.multiplicity, &row) == FE::zero() {
                        continue;
                    }
                    let v: Vec<FE> = it
                        .values
                        .iter()
                        .flat_map(|x| x.combine_from::<GoldilocksField, _>(|c| row[c]))
                        .collect();
                    out.push((v[0], v[1]));
                }
            }
            out
        };
        let answerable =
            |(input, output): &(FE, FE)| (*input == FE::zero()) == (*output == FE::one());

        // CONTROL: an honest group sends one ZERO tuple per absorb row, and the
        // precomputed table answers every one of them.
        let trace = trace_of(&[], &[absorb_op(4, 3, 1)]);
        let honest = zero_tuples(&trace);
        assert_eq!(honest.len(), 4, "one ZERO send per row of a 3-block group");
        assert!(
            honest.iter().all(answerable),
            "BITWISE answers every honest lookup"
        );

        // The attack: END = 1 while REMAINING = 3. The chip then sends ZERO[3] → 1,
        // and the BITWISE row for 3 holds 0 — the range-check bus cannot balance.
        let mut tampered = trace_of(&[], &[absorb_op(4, 3, 1)]);
        tampered.main_table.set_fe(0, cols::END, FE::one());
        let attacked = zero_tuples(&tampered);
        assert_eq!(
            attacked.iter().filter(|t| !answerable(t)).count(),
            1,
            "claiming END early must leave a ZERO lookup nothing can answer"
        );
    }

    /// ★ A padding row cannot mint a group boundary — COMMIT's boundary lock,
    /// retargeted to MU_A. Without it a `μ = 0` row could claim FIRST and
    /// receive a second `Ecall`, or claim END and write a `cv_out`.
    #[test]
    fn a_padding_row_cannot_mint_first_or_end() {
        let trace = trace_of(&[], &[absorb_op(4, 1, 1)]);
        let pad = trace.num_rows() - 1;
        assert_eq!(row_of(&trace, pad)[cols::MU], FE::zero(), "row is padding");

        for col in [cols::FIRST, cols::END] {
            let mut main = row_of(&trace, pad);
            main[col] = FE::one();
            assert!(
                eval_main_row(main).iter().any(|v| *v != FE::zero()),
                "the boundary lock must reject a padding row claiming {col}"
            );
        }
    }

    /// A single-compression row cannot claim a boundary either: the lock is on
    /// MU_A, not μ, which is CHANGE 3 of the design. Copying COMMIT's `μ − FIRST`
    /// receive multiplicity across would have made every single row receive from
    /// the absorb bus.
    #[test]
    fn a_single_row_cannot_mint_a_boundary() {
        let trace = trace_of(&[single_op(4)], &[]);
        for col in [cols::FIRST, cols::END] {
            let mut main = row_of(&trace, 0);
            main[col] = FE::one();
            assert!(
                eval_main_row(main).iter().any(|v| *v != FE::zero()),
                "a single-compression row must not claim {col}"
            );
        }
    }

    /// The two modes are exclusive, so one row cannot answer two syscalls.
    #[test]
    fn a_row_cannot_be_in_both_modes() {
        let trace = trace_of(&[single_op(4)], &[]);
        let mut main = row_of(&trace, 0);
        main[cols::MU_A] = FE::one();
        assert!(
            eval_main_row(main).iter().any(|v| *v != FE::zero()),
            "MU_S · MU_A = 0 must reject a row in both modes"
        );
    }

    /// A zero-block group is rejected: `FIRST · END = 0`. The executor rejects
    /// `num_blocks = 0`, and a group that were its own END row would copy
    /// `cv_in` straight to `cv_out`.
    #[test]
    fn a_zero_block_group_is_rejected() {
        let trace = trace_of(&[], &[absorb_op(4, 1, 1)]);
        let mut main = row_of(&trace, 0);
        assert_eq!(main[cols::FIRST], FE::one());
        main[cols::END] = FE::one();
        assert!(
            eval_main_row(main).iter().any(|v| *v != FE::zero()),
            "FIRST · END = 0 must reject a zero-block group"
        );
    }

    /// Two FIRST rows at one timestamp means two `Ecall` receives against the
    /// CPU's single send.
    #[test]
    fn two_first_rows_receive_one_ecall_twice() {
        let mut trace = trace_of(&[], &[absorb_op(4, 2, 1)]);
        assert_eq!(
            ecall_receives(&trace).1,
            1,
            "control: one group, one receive"
        );
        trace.main_table.set_fe(1, cols::FIRST, FE::one());
        assert_eq!(
            ecall_receives(&trace).1,
            2,
            "a second FIRST at the same timestamp doubles the receive"
        );
    }

    // ---------------------------------------------------------------------
    // Falsification: the schedule
    // ---------------------------------------------------------------------

    /// ★ `first_flags` lands on block 0 and nowhere else. Flags on a later block
    /// would produce a valid proof of a digest nobody else computes.
    #[test]
    fn flags_on_a_later_block_are_rejected() {
        let trace = trace_of(&[], &[absorb_op(4, 3, 1)]);
        let mut main = row_of(&trace, 1);
        assert_eq!(
            main[cols::FIRST],
            FE::zero(),
            "row 1 is not the first block"
        );
        main[cols::in_word(27, 0)] = FE::one();
        assert!(
            eval_main_row(main).iter().any(|v| *v != FE::zero()),
            "flags must be zero on every compressing row but the first"
        );
    }

    /// ★ The FIRST row's flags are not free either — they are bound by the x13
    /// register read, so tampering them changes a tuple that must match the
    /// register file. (Row-local constraints deliberately do NOT pin this: the
    /// guest chooses its own flag word.)
    #[test]
    fn the_first_rows_flags_ride_the_x13_register_read() {
        let trace = trace_of(&[], &[absorb_op(4, 2, 5)]);
        let row = row_of(&trace, 0);
        // The register reads gated on FIRST, evaluated on the group's first row.
        // Slot 9 of a MEMW tuple is the register's word address (2·regno).
        let reads: Vec<Vec<FE>> = bus_interactions()
            .iter()
            .filter(|i| {
                i.bus_id == u64::from(BusId::Memw)
                    && matches!(i.multiplicity, Multiplicity::Column(c) if c == cols::FIRST)
            })
            .map(|i| {
                i.values
                    .iter()
                    .flat_map(|v| v.combine_from::<GoldilocksField, _>(|c| row[c]))
                    .collect()
            })
            .collect();
        assert_eq!(reads.len(), 3, "x11, x12 and x13 (x10 is shared with MU_S)");
        let x13 = reads
            .iter()
            .find(|v| v[9] == FE::from(26u64))
            .expect("x13 -> word address 26");
        assert_eq!(x13[0], FE::from(5u64), "x13 carries first_flags");
        assert_eq!(x13[1], FE::zero(), "and pins its high half to zero");
        // x12 carries the block count, which IS the countdown's initial value:
        // nothing else pins the group's length to the ecall's argument.
        let x12 = reads
            .iter()
            .find(|v| v[9] == FE::from(24u64))
            .expect("x12 -> word address 24");
        assert_eq!(x12[0], FE::from(2u64), "x12 is REMAINING on the FIRST row");
        assert_eq!(x12[1], FE::zero(), "no block count above 2^32");
    }

    /// The interior schedule is constrained, not assumed: `t = 0` and
    /// `block_len = 64` on every absorbed block.
    #[test]
    fn the_interior_schedule_is_pinned() {
        let trace = trace_of(&[], &[absorb_op(4, 2, 1)]);
        for (col, delta) in [
            (cols::in_word(24, 0), FE::one()), // t_lo
            (cols::in_word(25, 0), FE::one()), // t_hi
            (cols::in_word(26, 0), FE::one()), // block_len 64 -> 65
        ] {
            let mut main = row_of(&trace, 0);
            main[col] += delta;
            assert!(
                eval_main_row(main).iter().any(|v| *v != FE::zero()),
                "column {col} must be pinned on an absorb row"
            );
        }
    }

    /// ★ The message base advances by exactly 64 per block. A prover free to
    /// choose it would read a block of its own choosing and still balance MEMW,
    /// because reading some other address is a legitimate read.
    #[test]
    fn the_message_base_must_advance_by_64() {
        let trace = trace_of(&[], &[absorb_op(4, 2, 1)]);
        let mut main = row_of(&trace, 0);
        // 32 instead of 64: the ADD template's carry stops being a bit.
        main[cols::M_BASE_INCR] = main[cols::M_BASE_INCR] - FE::from(32u64);
        assert!(
            eval_main_row(main).iter().any(|v| *v != FE::zero()),
            "M_BASE + 64 = M_BASE_INCR must reject a short step"
        );

        // ...and the per-dword pointers into the block are pinned the same way.
        let mut main = row_of(&trace, 0);
        main[cols::msg_ptr(3, 0)] = main[cols::msg_ptr(3, 0)] + FE::from(8u64);
        assert!(
            eval_main_row(main).iter().any(|v| *v != FE::zero()),
            "msg_ptr[j] = M_BASE + 8j must reject a shifted pointer"
        );
    }

    /// The countdown cannot be re-based mid-group.
    #[test]
    fn the_countdown_must_decrement_by_one() {
        let trace = trace_of(&[], &[absorb_op(4, 3, 1)]);
        let mut main = row_of(&trace, 0);
        main[cols::REM_DECR] += FE::one();
        assert!(
            eval_main_row(main).iter().any(|v| *v != FE::zero()),
            "REM_DECR + 1 = REMAINING must reject a skipped block"
        );
    }

    // ---------------------------------------------------------------------
    // Pins
    // ---------------------------------------------------------------------

    /// ★ The chip, `crypto` and `executor` agree on the absorb ABI's shape.
    /// The chip is a third statement of the control-region layout and the cap;
    /// drift would fault only on inputs no test reaches.
    #[test]
    fn the_chip_agrees_on_the_absorb_abi() {
        use executor::vm::instruction::execution::{
            BLAKE3_ABSORB_CTRL_DWORDS, BLAKE3_ABSORB_CV_OUT_DWORD, BLAKE3_ABSORB_MAX_BLOCKS,
            BLAKE3_BLOCK_BYTES,
        };
        assert_eq!(CTRL_DWORDS as u64, BLAKE3_ABSORB_CTRL_DWORDS);
        assert_eq!(CV_OUT_DWORD as u64, BLAKE3_ABSORB_CV_OUT_DWORD);
        assert_eq!(ABSORB_MAX_BLOCKS, BLAKE3_ABSORB_MAX_BLOCKS);
        assert_eq!(MSG_DWORDS as u64 * 8, BLAKE3_BLOCK_BYTES);
        // `cv_in` occupies the dwords below `cv_out`, which is what lets the
        // FIRST row's read share the single mode's `h` interaction.
        assert_eq!(CV_DWORDS, CV_OUT_DWORD);
    }

    /// ★ The in-circuit cap admits exactly `1..=1024`. `IsB20` holds [0, 2^20),
    /// so the scaled countdown is in the table iff the block count is legal —
    /// and the product cannot wrap, being under 2^30.
    #[test]
    fn the_block_cap_is_enforced_in_circuit() {
        const B20: u64 = 1 << 20;
        for n in [1u64, 2, 1023, ABSORB_MAX_BLOCKS] {
            assert!(
                (n - 1) * ABSORB_CAP_SCALE < B20,
                "a legal count of {n} blocks must pass IsB20"
            );
        }
        for n in [ABSORB_MAX_BLOCKS + 1, ABSORB_MAX_BLOCKS + 2, 4096] {
            assert!(
                (n - 1) * ABSORB_CAP_SCALE >= B20,
                "an over-cap count of {n} blocks must fail IsB20"
            );
            // ...and it fails by being out of range, not by wrapping the field.
            assert!((n - 1) * ABSORB_CAP_SCALE < 1u64 << 30);
        }
    }

    /// The table's shape, pinned so a change has to be deliberate. The absorb
    /// mode costs 47 columns and 76 interactions on top of the single mode's
    /// 3 219 / 1 397, and every one of those interactions costs a LogUp
    /// denominator on single-compression rows too.
    #[test]
    fn the_tables_shape_is_pinned() {
        assert_eq!(cols::NUM_COLUMNS, 3266);
        assert_eq!(bus_interactions().len(), 1473);
        assert_eq!(Blake3Constraints.meta().len(), 848);
        assert_eq!(Blake3Constraints.max_degree(), 3);
    }

    /// Every interaction's columns are inside the row.
    #[test]
    fn interactions_stay_inside_the_row() {
        for (i, it) in bus_interactions().iter().enumerate() {
            for v in &it.values {
                for c in v.column_indices() {
                    assert!(c < cols::NUM_COLUMNS, "interaction {i} reads column {c}");
                }
            }
        }
    }
}

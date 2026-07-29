//! eDSL libraries: transcript, Merkle and FRI expressed as ordinary Rust
//! that *emits instructions*. Host-side `for` loops unroll — nothing
//! loop-shaped reaches the machine; shapes (path depths, query counts,
//! domain parameters) are compile-time constants of the emitted program.
//!
//! The duplex sponge here is the machine side of the test transcript and is
//! mirrored bit-exactly by `fixture::HostSponge`. Like `TestPermutation`
//! itself it is NOT a production construction — the real transcript lands
//! with the ecosystem hash decision; this one exists so the protocol loop
//! can be built and measured now.

use crate::tables::types::FE;

use super::builder::{Bit, Cell, DigestVal, Ext, Felt, LfmBuilder};

/// Overwrite-rate duplex sponge over `LFM_HASH`: state = 3 cells (rate 2,
/// capacity 1).
pub struct SpongeVar {
    state: [Cell; 3],
}

impl SpongeVar {
    pub fn new(b: &mut LfmBuilder) -> Self {
        let z = b.felt_const(FE::zero()).as_cell();
        SpongeVar { state: [z, z, z] }
    }

    /// Absorb two cells: overwrite the rate, keep the capacity, permute.
    pub fn absorb2(&mut self, b: &mut LfmBuilder, c0: Cell, c1: Cell) {
        self.state = b.permute([c0, c1, self.state[2]]);
    }

    pub fn absorb(&mut self, b: &mut LfmBuilder, c: Cell) {
        let z = b.felt_const(FE::zero()).as_cell();
        self.absorb2(b, c, z);
    }

    /// Squeeze one cell (the current rate cell), then permute.
    pub fn squeeze_cell(&mut self, b: &mut LfmBuilder) -> Cell {
        let out = self.state[0];
        self.state = b.permute(self.state);
        out
    }

    /// Squeeze an ext challenge: lanes 0–2 of a squeezed cell.
    pub fn squeeze_ext(&mut self, b: &mut LfmBuilder) -> Ext {
        let c = self.squeeze_cell(b);
        let [l0, l1, l2, _] = b.unpack(c);
        b.pack_ext(l0, l1, l2)
    }

    /// Squeeze `nbits` index bits: the canonical bit decomposition of lane 0
    /// of a squeezed cell (masking to a power-of-two bound, so no rejection
    /// loop — the convention the RV64 verifier's query sampling already uses).
    pub fn squeeze_bits(&mut self, b: &mut LfmBuilder, nbits: usize) -> Vec<Bit> {
        let c = self.squeeze_cell(b);
        let [l0, _, _, _] = b.unpack(c);
        b.bit_dec(l0, nbits)
    }
}

/// Walk one Merkle authentication path. `bits` are the leaf-index bits
/// low-to-high (level 0 first): bit = 0 ⇒ the current node is the LEFT
/// child. Sibling digests come as (arena-hinted) cells; every hinted value
/// ends up inside a `compress`, which is what authenticates it.
pub fn merkle_walk(
    b: &mut LfmBuilder,
    leaf: DigestVal,
    bits: &[Bit],
    siblings: &[Cell],
) -> DigestVal {
    assert_eq!(bits.len(), siblings.len(), "one sibling per level");
    let mut current = leaf;
    for (bit, sibling) in bits.iter().zip(siblings) {
        let (left, right) = b.select(*bit, current.as_cell(), *sibling);
        current = b.compress(left.as_digest(), right.as_digest());
    }
    current
}

/// Assert two words are equal, lane by lane (2 unpacks + 4 lowered asserts).
pub fn assert_word_eq(b: &mut LfmBuilder, x: Cell, y: Cell) {
    let yl = b.unpack(y);
    assert_word_eq_lanes(b, x, &yl);
}

/// Assert a word equals four already-unpacked lanes (hoist the reference
/// word's unpack out of a loop — e.g. one root compared per query).
pub fn assert_word_eq_lanes(b: &mut LfmBuilder, x: Cell, y_lanes: &[Felt; 4]) {
    let xl = b.unpack(x);
    for i in 0..4 {
        b.assert_eq(xl[i], y_lanes[i]);
    }
}

/// `scale · Π factors[i]^{bits[i]}` — one Select + one Mul per bit. Used to
/// derive domain points (and their inverses) from query-index bits; the
/// factors are program constants, so nothing here touches an arena.
pub fn pow_bits(b: &mut LfmBuilder, bits: &[Bit], factors: &[FE], scale: FE) -> Felt {
    assert_eq!(bits.len(), factors.len());
    let mut acc = b.felt_const(scale);
    for (bit, factor) in bits.iter().zip(factors) {
        let one = b.felt_const(FE::one());
        let f = b.felt_const(*factor);
        let (chosen, _) = b.select(*bit, one.as_cell(), f.as_cell());
        acc = b.mul(acc, Felt(chosen.0));
    }
    acc
}

/// `Σ_i coeffs[i]·α^i` over ext, coeffs given low-to-high (base cells are
/// valid ext operands). One `MulAdd` per coefficient — the Horner shape.
pub fn horner_ext(b: &mut LfmBuilder, alpha: Ext, coeffs_low_to_high: &[Ext]) -> Ext {
    let mut iter = coeffs_low_to_high.iter().rev();
    let mut acc = *iter.next().expect("at least one coefficient");
    for c in iter {
        acc = b.emul_add(acc, alpha, *c);
    }
    acc
}

/// One unnormalized FRI fold — our production convention exactly:
/// `(lo + hi) + inv_x·ζ·(lo − hi)` (the missing ½ is absorbed into the
/// terminal polynomial).
pub fn fri_fold(b: &mut LfmBuilder, lo: Ext, hi: Ext, zeta: Ext, inv_x: Felt) -> Ext {
    let sum = b.eadd(lo, hi);
    let diff = b.esub(lo, hi);
    let zd = b.emul(zeta, diff);
    let scaled = b.emul_base(zd, inv_x);
    b.eadd(sum, scaled)
}

// ============================== keccak256 ==============================

/// `keccak256` over a byte stream supplied as `u32`-half felts (four bytes
/// each, little-endian — see [`super::keccak_host::pack_stream`]). Returns the
/// 32-byte digest as two machine words of halves.
///
/// Shapes are compile-time, as everywhere in this machine: `len_bytes` fixes
/// the block count and the padding positions, so `pad10*1` is emitted as
/// interned program CONSTANTS rather than computed. A different length is a
/// different program with a different digest — which is the straight-line
/// discipline working as intended, not a limitation to route around.
///
/// The digest is the state's first 32 bytes = halves 0..7 = words 0 and 1 of
/// the state's word representation, which is exactly `PlatformKeccak256`'s
/// output byte order (byte `j` = byte `j % 4` of half `j / 4`).
pub fn keccak256(b: &mut LfmBuilder, stream: &[Felt], len_bytes: usize) -> [Cell; 2] {
    let (state, _) = keccak256_absorb_all(b, stream, len_bytes, false);
    [state[0], state[1]]
}

/// `keccak256`, additionally returning the byte-REVERSED digest — the value the
/// production `DefaultTranscript::sample()` both returns as the challenge and
/// re-absorbs as the next segment's prefix.
///
/// `sample()` is byte-for-byte identical before and after #841, so this is
/// independent of which transcript revision the caller targets.
pub fn keccak256_rev(b: &mut LfmBuilder, stream: &[Felt], len_bytes: usize) -> [Cell; 2] {
    let (_, rev) = keccak256_absorb_all(b, stream, len_bytes, true);
    rev.expect("requested")
}

/// `keccak256` returning BOTH digests — plain and byte-reversed — off the one
/// keccak row that produces them.
///
/// The transcript replay needs both at once and they are not interchangeable:
/// the reversed digest is what `sample()` returns and re-absorbs, while
/// candidates are read off the PLAIN digest (the reversal and the big-endian
/// candidate read cancel — see [`super::keccak_host::candidate_from_state`]).
pub fn keccak256_with_rev(
    b: &mut LfmBuilder,
    stream: &[Felt],
    len_bytes: usize,
) -> ([Cell; 2], [Cell; 2]) {
    let (state, rev) = keccak256_absorb_all(b, stream, len_bytes, true);
    ([state[0], state[1]], rev.expect("requested"))
}

/// `Σ_i 2^i · bits[i]`, bits low-to-high — the value a bit decomposition stands
/// for. Horner from the top: one `MulAdd` per bit after the first.
pub fn bits_to_felt(b: &mut LfmBuilder, bits: &[Bit]) -> Felt {
    let two = b.felt_const(FE::from(2u64));
    let mut iter = bits.iter().rev();
    let mut acc = iter.next().expect("at least one bit").as_felt();
    for bit in iter {
        acc = b.mul_add(acc, two, bit.as_felt());
    }
    acc
}

fn keccak256_absorb_all(
    b: &mut LfmBuilder,
    stream: &[Felt],
    len_bytes: usize,
    want_rev: bool,
) -> ([Cell; 13], Option<[Cell; 2]>) {
    use super::keccak_host::{num_blocks, num_stream_halves, pad_half};
    use super::layout::keccak::{BLOCK_HALVES, BLOCK_WORDS, NUM_WORDS};

    assert_eq!(
        stream.len(),
        num_stream_halves(len_bytes),
        "stream must hold exactly ceil(len_bytes / 4) halves"
    );

    let zero = b.felt_const(FE::zero());
    let mut state: [Cell; NUM_WORDS] = [zero.as_cell(); NUM_WORDS];
    let mut rev: Option<[Cell; 2]> = None;

    for block in 0..num_blocks(len_bytes) {
        // Half `h` of this block is half `block * BLOCK_HALVES + h` of the
        // padded message; both the rate (136 bytes) and a half (4 bytes) divide
        // evenly, so the two indexings line up with no straddling across blocks.
        let halves: Vec<Felt> = (0..BLOCK_HALVES)
            .map(|h| {
                let g = block * BLOCK_HALVES + h;
                let pad = pad_half(len_bytes, g);
                match (g < num_stream_halves(len_bytes), pad) {
                    // Entirely inside the message.
                    (true, 0) => stream[g],
                    // Straddles the end: the stream half's high bytes are zero
                    // by the packing convention, so adding merges the padding in
                    // without carrying.
                    (true, p) => {
                        let c = b.felt_const(FE::from(p));
                        b.add(stream[g], c)
                    }
                    // Entirely padding (possibly all-zero).
                    (false, p) => b.felt_const(FE::from(p)),
                }
            })
            .collect();

        // 34 halves into 9 words; the last word's top two slots are the unused
        // half slots the chip pins to zero.
        let block_words: [Cell; BLOCK_WORDS] = core::array::from_fn(|w| {
            let lane = |l: usize| {
                let h = 4 * w + l;
                if h < BLOCK_HALVES { halves[h] } else { zero }
            };
            b.pack_word([lane(0), lane(1), lane(2), lane(3)])
        });

        let last = block + 1 == num_blocks(len_bytes);
        if last && want_rev {
            let (next, rev_words) = b.keccak_absorb_rev(state, block_words);
            state = next;
            rev = Some(rev_words);
        } else {
            state = b.keccak_absorb(state, block_words);
        }
    }

    (state, rev)
}

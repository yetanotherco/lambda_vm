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

// ===================== production keccak Merkle =====================

/// A 32-byte keccak digest as it lives in the machine: two words of four `u32`
/// halves each, half `h` carrying digest bytes `4h..4h+4`.
pub type KeccakDigest = [Cell; 2];

/// Halves in a 32-byte digest.
pub const DIGEST_HALVES: usize = 8;

/// The eight halves of a keccak digest, ready to be streamed into another
/// `keccak256`.
pub fn keccak_digest_halves(b: &mut LfmBuilder, d: KeccakDigest) -> [Felt; DIGEST_HALVES] {
    let lo = b.unpack(d[0]);
    let hi = b.unpack(d[1]);
    core::array::from_fn(|h| if h < 4 { lo[h] } else { hi[h - 4] })
}

/// The Merkle LEAF hash of a row pair, in the production commitment layout.
///
/// `values` is `evaluations ‖ evaluations_sym` — the two bit-reversed rows the
/// leaf covers, each written column by column. Every element is a base field
/// element rendered as its canonical `u64` in BIG-endian bytes
/// (`FieldElement<GoldilocksField>::stream_bytes`), so each costs one
/// [`super::transcript_replay::felt_be_halves`]: one `LFM_BITDEC` row and 64
/// `LFM_BALU` rows. The hash itself is `keccak256` over `8 · values.len()`
/// bytes.
///
/// ## The byteswapping is NOT what this costs — measured, against expectation
///
/// A `c`-column table gives `2c` elements, so `2c` decompositions and `128c`
/// ALU rows against only `⌈(16c + 1) / 136⌉` permutations. On row counts the
/// byteswapping looks overwhelming, which is what the R1f handoff predicted.
/// That reading is wrong: rows of different chips are not comparable units. An
/// `LFM_BALU` row carries 4 non-preprocessed columns, while one permutation
/// expands into 24 `KECCAK_RND` rounds of 1480 columns — so in main-trace cells
/// a permutation costs 113 byteswaps, and the hash term dominates at every
/// table width. `machine_tests::keccak_merkle_opening_cost` measures it and
/// asserts the inequality holds.
///
/// The swap is real work regardless, and it is not avoidable by pre-swapping in
/// the arena: the same opened values are consumed as FIELD ELEMENTS by the FRI
/// algebra and as BYTES by this hash, so something has to connect the two
/// representations, and only the machine can do it in a way the proof binds.
pub fn keccak_leaf_hash(b: &mut LfmBuilder, values: &[Felt]) -> KeccakDigest {
    use super::keccak_host::BYTES_PER_HALF;
    use super::transcript_replay::felt_be_halves;

    assert!(!values.is_empty(), "a leaf covers at least one column");
    let mut stream = Vec::with_capacity(2 * values.len());
    for v in values {
        stream.extend(felt_be_halves(b, *v));
    }
    let len_bytes = BYTES_PER_HALF * stream.len();
    keccak256(b, &stream, len_bytes)
}

/// Walk one Merkle authentication path under the PRODUCTION hash.
///
/// This is the keccak counterpart of [`merkle_walk`], and the two are not
/// interchangeable: `merkle_walk` compresses with `LFM_HASH`/`TestPermutation`,
/// the deliberately non-cryptographic Milestone-C placeholder, so it can only
/// ever authenticate the Milestone-C fixture tree. Production trees are keccak
/// throughout, and this is the walk that authenticates them.
///
/// `bits` are the leaf index low-to-high, level 0 first; `bit = 0` means the
/// current node is the LEFT child, matching `verify_merkle_path_from_leaf_hash`
/// (`index % 2 == 0 ⇒ hash(current, sibling)`).
///
/// ## The parent step
///
/// `hash_new_parent(l, r) = keccak(l ‖ r)` — 64 bytes, no domain separation and
/// no ordering flag, so the ordering is carried entirely by the index bit. 64
/// bytes sits inside one 136-byte rate block, so a level is exactly ONE
/// permutation. Per level the machine pays two `Select`s (a digest is two words
/// and both must swap on the same bit), four `Unpack`s and that permutation.
pub fn keccak_merkle_walk(
    b: &mut LfmBuilder,
    leaf: KeccakDigest,
    bits: &[Bit],
    siblings: &[KeccakDigest],
) -> KeccakDigest {
    assert_eq!(bits.len(), siblings.len(), "one sibling per level");
    let mut current = leaf;
    for (bit, sibling) in bits.iter().zip(siblings) {
        // Both halves of the digest must swap on the SAME bit.
        let (l0, r0) = b.select(*bit, current[0], sibling[0]);
        let (l1, r1) = b.select(*bit, current[1], sibling[1]);
        let left = keccak_digest_halves(b, [l0, l1]);
        let right = keccak_digest_halves(b, [r0, r1]);
        let mut stream = Vec::with_capacity(2 * DIGEST_HALVES);
        stream.extend(left);
        stream.extend(right);
        current = keccak256(b, &stream, 2 * COMMITMENT_BYTES);
    }
    current
}

/// Bytes in a commitment / Merkle node.
pub const COMMITMENT_BYTES: usize = 32;

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

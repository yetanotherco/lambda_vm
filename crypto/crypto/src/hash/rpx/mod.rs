//! Rescue-Prime eXtended (RPX256 / XHash12) over Goldilocks at width 12.
//!
//! The algebraic hash the WHIR recursion arm commits, transcripts and grinds
//! with. Ported from `prover::lfm::{rpo, rpx, algebraic_commit}` on the
//! per-table branch, where it is the production-candidate tenant of the
//! `LFM_HASH` socket; the permutation, the leaf construction, the parent and
//! every constant are byte-for-byte the same, because the CUDA kernel and its
//! known-answer tables are pinned to exactly those.
//!
//! # Why an algebraic hash at all
//!
//! Only for a proof that is going to be VERIFIED INSIDE a proof. In software a
//! keccak-f is far cheaper than this. In a field-native verifier the ratio
//! inverts by two orders of magnitude: a keccak-f[1600] costs ~73,700 trace
//! cells against RPX's 325, which is the difference between a WHIR wrap that is
//! twice today's and one that is a third of it. Nothing about this hash is an
//! improvement on keccak for a host prover, and the seam that selects it says
//! so.
//!
//! # What it is, and what it shares with RPO
//!
//! RPX is a **round-function swap on RPO's geometry**, not a redesign
//! ([eprint 2023/1045](https://eprint.iacr.org/2023/1045)): the same state
//! width 12, the same rate 8 / capacity 4, the same four-felt digest, the same
//! MDS and literally the same `ARK1`/`ARK2` tables. What changes is the
//! seven-round schedule:
//!
//! | round | kind | content |
//! |---|---|---|
//! | 0, 2, 4 | **FB** | MDS → +ARK1 → `x^7` → MDS → +ARK2 → `x^{1/7}` — RPO's round exactly |
//! | 1, 3, 5 | **E** | +ARK1 → `x^7` in the degree-3 EXTENSION, on four lane-triples. **No MDS.** |
//! | 6 | **M** | MDS → +ARK1. A linear finish, no S-box. |
//!
//! The E round has no linear layer: its only mixing is the extension
//! multiplication inside each triple, and diffusion across triples is the FB
//! rounds' job. That is the design, not an omission (✓ miden's
//! `Rpx256::apply_ext_round_ref`).
//!
//! # ⚠ PROVENANCE — WEAKER THAN RPO'S, AND THAT MUST BE SAID
//!
//! **miden publishes no RPX known-answer table** — ✓ VERIFIED, its `rpx/tests.rs`
//! carries only structural tests (consistency, determinism, padding, no-panic),
//! no oracle. So RPX cannot be anchored end to end the way RPO is, and this
//! module does not pretend otherwise. What it anchors instead:
//!
//! 1. **The shared half is externally anchored through RPO.** Seven `fb_round`s
//!    compose to RPO256, and [`tests`] replays that composition over
//!    miden-crypto's nineteen `hash_elements` vectors — numbers nothing in this
//!    repository produced. They pin `ARK1`/`ARK2`, the MDS row and its
//!    orientation, both S-box chains and the lane convention at once.
//! 2. **The new half is pinned to INDEPENDENT algorithms.** The cubic
//!    extension's product against naive polynomial multiplication reduced mod
//!    `φ³ − φ − 1`; `power7` against generic square-and-multiply in that
//!    extension; the inverse S-box chain against `pow(INV_ALPHA)`. Different
//!    algorithms for the same functions, not a second transcription.
//! 3. **The schedule** is the one miden's `Rpx256::apply_permutation` runs.
//!
//! ⚖ Net: strong on arithmetic, weaker on end-to-end identity than RPO. A
//! deployment decision should treat "no published KAT" as a real cost. RPX is
//! also a 2023 design and carries a young-design discount BLAKE3 and keccak do
//! not.
//!
//! # ⚠ NOT XHash8
//!
//! XHash8 is the faster sibling and is deliberately not built here. Its extra
//! speed comes from a PARTIAL S-box layer (8 lanes of 12), and a partial layer
//! is one of the structural footholds the 2026 Poseidon collapse used —
//! eprint 2026/1692's S-box-skipping gadget restricts into the affine
//! complement of the un-S-boxed lanes, independent of round constants and MDS
//! choice. XHash8's S-boxes are not Poseidon's and eprint 2024/605 analyses
//! XHASH8/12 directly, so this is a flag rather than a verdict — but it is not
//! a thing to adopt quietly for the speed.
//!
//! # Lane convention, domains, and the rules that must not drift
//!
//! Lanes follow **miden's**: rate `0..8`, capacity `8..12`, digest `0..4`.
//! Capacity lane 0 carries the sponge's padding flag `len mod 8`; capacity lane
//! 1 carries a DOMAIN tag, which is miden's `merge_in_domain` mechanism. The
//! security argument is the RPX spec's Appendix C: setting a capacity element
//! to a domain tag degrades only pre-image resistance, by at most the log2 of
//! the domain space, and pre-image is not the sponge's binding term until it
//! falls under 2^128.
//!
//! ⚠ **The domain VALUES are pinned by the device kernel and its KAT tables**
//! (`rpx.cu`'s `DOMAIN_COMPRESS = 0`, `DOMAIN_LEAF = 0x4C4D464C`). The `LFM`
//! spelling of [`DOMAIN_LEAF`] is a historical name — it is `"LFML"` read as a
//! little-endian `u32` — and renaming the constant is free while **changing its
//! value forks the hash** from the kernel, from the KAT header and from every
//! root the per-table branch produced.

pub mod constants;
#[cfg(test)]
mod tests;

use alloc::vec::Vec;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};
use math::traits::AsBytes;

use constants::{ARK1, ARK2, MDS_CIRC_ROW, NUM_ROUNDS};

/// A Goldilocks field element — the only field this hash is defined over.
pub type Fp = FieldElement<GoldilocksField>;

/// Lanes in the permutation's state.
pub const STATE_FELTS: usize = 12;
/// Lanes a block of input overwrites — the sponge's rate.
pub const RATE_FELTS: usize = 8;
/// Felts in a digest, hence a 32-byte commitment.
pub const DIGEST_FELTS: usize = 4;
/// Bytes one Goldilocks felt serialises to.
pub const BYTES_PER_FELT: usize = 8;

/// Capacity lane carrying the sponge padding flag — reserved, never a domain.
pub const CAPACITY_PAD_LANE: usize = 0;
/// Capacity lane carrying the DOMAIN identifier — miden's `merge_in_domain` slot.
pub const CAPACITY_DOMAIN_LANE: usize = 1;

/// The Merkle-parent domain: ZERO, deliberately.
///
/// A parent is then bit-identical to `Rpo256::merge`/`Rpx256::merge`, so a
/// parent this code produces is checkable against miden's shipped
/// implementation without knowing anything about this codebase.
pub const DOMAIN_COMPRESS: u64 = 0;

/// The Merkle LEAF domain — `"LFML"` as a little-endian `u32`, i.e.
/// `0x4C4D464C`. See the module header: the name is historical, the VALUE is
/// pinned by the device kernel and the KAT tables.
pub const DOMAIN_LEAF: u64 = u32::from_le_bytes(*b"LFML") as u64;

/// Lanes per extension element: the RPX E round reads the state as FOUR triples.
pub const EXT_DEGREE: usize = 3;
/// Extension elements per E round.
pub const EXT_ELEMENTS: usize = STATE_FELTS / EXT_DEGREE;

/// A four-felt digest.
pub type Digest = [Fp; DIGEST_FELTS];

/// The capacity cell for a domain: `[0, domain, 0, 0]`.
///
/// One rule, stated once, so nothing can disagree about which lane the tag
/// lives in.
pub const fn domain_iv(domain: u64) -> [u64; DIGEST_FELTS] {
    let mut iv = [0u64; DIGEST_FELTS];
    iv[CAPACITY_DOMAIN_LANE] = domain;
    iv
}

/// Is round `r` an **FB** round — MDS, forward S-box, MDS, inverse S-box?
pub const fn is_fb_round(r: usize) -> bool {
    r.is_multiple_of(2) && r + 1 < NUM_ROUNDS
}

/// Is round `r` an **E** round — constants then `x^7` in the cubic extension,
/// with NO linear layer?
pub const fn is_ext_round(r: usize) -> bool {
    !r.is_multiple_of(2)
}

/// Is round `r` the **M** round — MDS then constants, and nothing else?
pub const fn is_final_round(r: usize) -> bool {
    r + 1 == NUM_ROUNDS
}

/// Arithmetic in `GF(p³) = GF(p)[φ] / (φ³ − φ − 1)`.
///
/// ⚠ **Not the VM's own extension**, which is built on `w³ = 2`. Mixing them
/// would be a wrong hash that still type-checks, so this carries its own
/// arithmetic explicitly and never reaches for the VM's.
pub mod cubic_ext {
    use super::{EXT_DEGREE, Fp};

    /// An extension element `a0 + a1·φ + a2·φ²`.
    pub type Ext = [Fp; EXT_DEGREE];

    /// The product, reduced by `φ³ = φ + 1` and `φ⁴ = φ² + φ`.
    ///
    /// The closed form rather than miden's Karatsuba arrangement, so the three
    /// coefficients read as the definition.
    /// `tests::the_extension_product_matches_naive_polynomial_arithmetic` pins
    /// it against an independent algorithm.
    pub fn mul(a: &Ext, b: &Ext) -> Ext {
        [
            &(&a[0] * &b[0]) + &(&(&a[1] * &b[2]) + &(&a[2] * &b[1])),
            &(&(&a[0] * &b[1]) + &(&a[1] * &b[0]))
                + &(&(&(&a[1] * &b[2]) + &(&a[2] * &b[1])) + &(&a[2] * &b[2])),
            &(&(&a[0] * &b[2]) + &(&a[1] * &b[1])) + &(&(&a[2] * &b[0]) + &(&a[2] * &b[2])),
        ]
    }

    /// The square. One function, so a squaring and a product cannot disagree.
    pub fn square(a: &Ext) -> Ext {
        mul(a, a)
    }

    /// `a^7` by the chain `a² → a³ → a⁶ → a⁷`.
    pub fn power7(a: &Ext) -> Ext {
        let a2 = square(a);
        let a3 = mul(&a2, a);
        let a6 = square(&a3);
        mul(&a6, a)
    }
}

/// `x^7`, in exactly the association the AIR's degree-3 lowering uses
/// (`x²`, `x³ = x²·x`, `x^7 = (x³)²·x`).
pub fn sbox(x: &Fp) -> Fp {
    let x2 = x * x;
    let x3 = &x2 * x;
    let x6 = &x3 * &x3;
    &x6 * x
}

/// `x^{1/7}` over the WHOLE STATE, by miden-crypto's documented addition chain
/// (72 multiplications for a ~2^63 exponent, against ~93 for naive
/// square-and-multiply).
///
/// ★ **Whole-state rather than per-element, and that is a measurement.** The
/// chain is 72 multiplications each depending on the last, so a single lane is
/// LATENCY-bound and the multiplier pipeline sits idle between them. The twelve
/// lanes are independent, so running them in lockstep interleaves twelve chains
/// and fills it. This layer is the dominant cost of the permutation.
pub fn inv_sbox_layer(state: &mut [Fp; STATE_FELTS]) {
    /// `base^(2^m) · tail`, lane-wise — the chain's one building block.
    fn exp_acc(base: &[Fp; STATE_FELTS], tail: &[Fp; STATE_FELTS], m: usize) -> [Fp; STATE_FELTS] {
        let mut acc = *base;
        for _ in 0..m {
            for a in acc.iter_mut() {
                *a = a.square();
            }
        }
        core::array::from_fn(|i| &acc[i] * &tail[i])
    }

    let t1: [Fp; STATE_FELTS] = core::array::from_fn(|i| state[i].square());
    let t2: [Fp; STATE_FELTS] = core::array::from_fn(|i| t1[i].square());
    let t3 = exp_acc(&t2, &t2, 3);
    let t4 = exp_acc(&t3, &t3, 6);
    let t5 = exp_acc(&t4, &t4, 12);
    let t6 = exp_acc(&t5, &t3, 6);
    let t7 = exp_acc(&t6, &t6, 31);
    for (i, s) in state.iter_mut().enumerate() {
        let a = (&t7[i].square() * &t6[i]).square().square();
        let b = &(&t1[i] * &t2[i]) * &*s;
        *s = &a * &b;
    }
}

/// [`inv_sbox_layer`] for a single element — the same chain, not a second
/// transcription of it.
pub fn inv_sbox(x: &Fp) -> Fp {
    let mut state = [*x; STATE_FELTS];
    inv_sbox_layer(&mut state);
    state[0]
}

/// The circulant MDS product, `out_i = Σ_j MDS_CIRC_ROW[(j − i) mod 12]·s_j`.
///
/// ★ **One `u128` accumulation and one reduction per lane, not twelve field
/// multiplications.** The constants are all ≤ 26, so every term `c·s_j` fits in
/// 70 bits and the twelve-term row sum fits in 73 — comfortably inside a
/// `u128`. The row is accumulated with no reduction and reduced once at the end
/// using `2^64 ≡ EPSILON (mod p)`: `hi·2^64 + lo ≡ lo + hi·EPSILON`, and with
/// `hi < 2^9` the correction `hi·EPSILON < 2^41` needs no reduction of its own.
/// `tests::the_mds_row_sum_cannot_overflow_a_u128` asserts the bound.
pub fn mds(state: &[Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    /// `2^32 − 1`, and `2^64 ≡ EPSILON (mod p)` for the Goldilocks prime.
    /// Written here rather than imported because the field crate keeps its own
    /// copy private; `tests::the_epsilon_identity_holds` re-derives it.
    const EPSILON: u64 = 0xFFFF_FFFF;

    let raw: [u64; STATE_FELTS] = core::array::from_fn(|j| *state[j].value());
    core::array::from_fn(|i| {
        let mut acc: u128 = 0;
        for (j, s) in raw.iter().enumerate() {
            let c = MDS_CIRC_ROW[(j + STATE_FELTS - i) % STATE_FELTS];
            acc += (*s as u128) * (c as u128);
        }
        let lo = acc as u64;
        let hi = (acc >> 64) as u64;
        // hi < 2^9, so hi·EPSILON < 2^41 and neither `from` reduces twice.
        Fp::from(lo) + Fp::from(hi * EPSILON)
    })
}

/// ★ The RPX permutation: `FB E FB E FB E M`.
pub fn permute(state: [Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    let mut s = state;
    // Over ARK1 rather than over `0..NUM_ROUNDS`: the round index is still what
    // `fb_round` takes, but the E and M rounds read ARK1 and only ARK1, and
    // iterating it says so.
    for (r, ark1) in ARK1.iter().enumerate() {
        if is_fb_round(r) {
            s = fb_round(s, r);
        } else if is_ext_round(r) {
            for (lane, v) in s.iter_mut().enumerate() {
                *v += Fp::from(ark1[lane]);
            }
            let mut next = [Fp::zero(); STATE_FELTS];
            for e in 0..EXT_ELEMENTS {
                let base = e * EXT_DEGREE;
                let x: cubic_ext::Ext = core::array::from_fn(|k| s[base + k]);
                let p = cubic_ext::power7(&x);
                next[base..base + EXT_DEGREE].copy_from_slice(&p);
            }
            s = next;
        } else {
            debug_assert!(is_final_round(r));
            s = mds(&s);
            for (lane, v) in s.iter_mut().enumerate() {
                *v += Fp::from(ark1[lane]);
            }
        }
    }
    s
}

/// One **FB** round: `MDS → +ARK1 → x^7 → MDS → +ARK2 → x^{1/7}`.
///
/// ★ Exported because it is RPO's round EXACTLY, and seven of them composed are
/// RPO256 — which is how the nineteen external miden vectors reach RPX's
/// constants. [`tests::seven_fb_rounds_are_rpo256`] is that bridge.
pub fn fb_round(state: [Fp; STATE_FELTS], r: usize) -> [Fp; STATE_FELTS] {
    let mut s = mds(&state);
    for (lane, v) in s.iter_mut().enumerate() {
        *v += Fp::from(ARK1[r][lane]);
    }
    for v in s.iter_mut() {
        *v = sbox(v);
    }
    s = mds(&s);
    for (lane, v) in s.iter_mut().enumerate() {
        *v += Fp::from(ARK2[r][lane]);
    }
    inv_sbox_layer(&mut s);
    s
}

// =========================================================================
// The sponge: leaves, parents, and the felt/byte conventions
// =========================================================================

/// ★ **THE LEAF CAPACITY RULE, stated once.**
///
/// Lane 0 is the padding flag `len mod 8` — zero when the length divides the
/// rate, which is why no trailing block is spent on an exact multiple — and
/// lane 1 the LEAF domain.
pub fn leaf_capacity(num_felts: usize) -> Digest {
    let iv = domain_iv(DOMAIN_LEAF);
    let mut cap: Digest = core::array::from_fn(|k| Fp::from(iv[k]));
    cap[CAPACITY_PAD_LANE] = Fp::from((num_felts % RATE_FELTS) as u64);
    cap
}

/// ★ The rate-8 OVERWRITE duplex over a felt stream — the leaf construction.
///
/// Each block OVERWRITES the eight rate lanes (RPO spec §2.6), so absorption
/// costs no field arithmetic outside the permutation; the tail block is
/// zero-padded. It absorbs eight fresh felts per permutation where a
/// four-felt chain absorbs four.
pub fn sponge_leaf(felts: &[Fp]) -> Digest {
    let mut state = [Fp::zero(); STATE_FELTS];
    let cap = leaf_capacity(felts.len());
    state[RATE_FELTS..].copy_from_slice(&cap);

    if felts.is_empty() {
        return [state[0], state[1], state[2], state[3]];
    }
    for block in felts.chunks(RATE_FELTS) {
        for (lane, slot) in state.iter_mut().take(RATE_FELTS).enumerate() {
            *slot = block.get(lane).copied().unwrap_or_else(Fp::zero);
        }
        state = permute(state);
    }
    [state[0], state[1], state[2], state[3]]
}

/// `sponge_leaf(&felts_from_bytes(bytes))`, without materialising the felts.
///
/// ⚠ Equivalent to the two-step form BY TEST
/// (`tests::sponge_leaf_bytes_matches_the_felt_form`), not by construction: the
/// trailing partial group is zero-extended on the LOW side here, which is what
/// [`felts_from_bytes`] does and is easy to get backwards.
pub fn sponge_leaf_bytes(bytes: &[u8]) -> Digest {
    let num_felts = bytes.len().div_ceil(BYTES_PER_FELT);
    let mut state = [Fp::zero(); STATE_FELTS];
    let cap = leaf_capacity(num_felts);
    state[RATE_FELTS..].copy_from_slice(&cap);

    if bytes.is_empty() {
        return [state[0], state[1], state[2], state[3]];
    }
    // One rate block is eight felts, i.e. 64 bytes.
    for block in bytes.chunks(RATE_FELTS * BYTES_PER_FELT) {
        for (lane, slot) in state.iter_mut().take(RATE_FELTS).enumerate() {
            let start = lane * BYTES_PER_FELT;
            *slot = if start >= block.len() {
                Fp::zero()
            } else {
                let end = (start + BYTES_PER_FELT).min(block.len());
                let mut b = [0u8; BYTES_PER_FELT];
                b[..end - start].copy_from_slice(&block[start..end]);
                Fp::from(u64::from_be_bytes(b))
            };
        }
        state = permute(state);
    }
    [state[0], state[1], state[2], state[3]]
}

/// ★ A Merkle parent: ONE permutation of `[left ‖ right ‖ capacity]` with the
/// compress domain, which is zero — so a parent is literally `Rpx256::merge`
/// and externally checkable against miden.
pub fn compress(left: &Digest, right: &Digest) -> Digest {
    let mut state = [Fp::zero(); STATE_FELTS];
    state[..DIGEST_FELTS].copy_from_slice(left);
    state[DIGEST_FELTS..RATE_FELTS].copy_from_slice(right);
    let iv = domain_iv(DOMAIN_COMPRESS);
    for (k, slot) in state[RATE_FELTS..].iter_mut().enumerate() {
        *slot = Fp::from(iv[k]);
    }
    let out = permute(state);
    [out[0], out[1], out[2], out[3]]
}

/// Four felts as 32 canonical BIG-endian bytes.
pub fn digest_to_commitment(d: &Digest) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, f) in d.iter().enumerate() {
        let v = GoldilocksField::canonical(f.value());
        out[i * BYTES_PER_FELT..(i + 1) * BYTES_PER_FELT].copy_from_slice(&v.to_be_bytes());
    }
    out
}

/// 32 bytes back to four felts.
pub fn commitment_to_digest(c: &[u8; 32]) -> Digest {
    core::array::from_fn(|i| {
        let mut b = [0u8; BYTES_PER_FELT];
        b.copy_from_slice(&c[i * BYTES_PER_FELT..(i + 1) * BYTES_PER_FELT]);
        Fp::from(u64::from_be_bytes(b))
    })
}

/// Every 8-byte big-endian group of `bytes` as a felt.
///
/// The inverse of the serialisation `ByteConversion::write_bytes_be` performs,
/// which is how leaves reach a Merkle backend. A trailing partial group is
/// zero-extended on the LOW side, matching how a short write would land.
pub fn felts_from_bytes(bytes: &[u8]) -> Vec<Fp> {
    bytes
        .chunks(BYTES_PER_FELT)
        .map(|c| {
            let mut b = [0u8; BYTES_PER_FELT];
            b[..c.len()].copy_from_slice(c);
            Fp::from(u64::from_be_bytes(b))
        })
        .collect()
}

/// Decompose a field element — base or extension — into its base felts, by the
/// same serialisation the STARK uses.
///
/// ★ Through `AsBytes::stream_bytes` rather than `ByteConversion::write_bytes_be`,
/// and the two are the SAME bytes. The reason for the weaker trait is not
/// style: a Merkle backend generic over `F` has `FieldElement<F>: AsBytes` and
/// nothing more, so a decomposition that required `ByteConversion` could not be
/// used there at all.
pub fn element_felts<F>(e: &FieldElement<F>, out: &mut Vec<Fp>)
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    let mut buf = [0u8; 64];
    let mut len = 0usize;
    e.stream_bytes(&mut |bytes| {
        debug_assert!(
            len + bytes.len() <= buf.len(),
            "a field element must fit the scratch"
        );
        buf[len..len + bytes.len()].copy_from_slice(bytes);
        len += bytes.len();
    });
    out.extend(felts_from_bytes(&buf[..len]));
}

// =========================================================================
// The `digest::Digest` adapter — what a transcript and the grind consume
// =========================================================================

/// RPX256 as a `digest::Digest`, for the two places that take one: the
/// Fiat-Shamir sponge and the proof-of-work grind.
///
/// # The construction is the LEAF one, deliberately
///
/// Grinding hashes a byte string — `state ‖ nonce`, 40 bytes, five felts —
/// which is DATA, exactly what a leaf is. It therefore reuses [`sponge_leaf`]
/// and the LEAF domain rather than inventing a fourth. The reuse is not
/// exploitable: the grinding check tests leading zeros of a hash whose preimage
/// is transcript-bound, so colliding it with some leaf digest buys an adversary
/// nothing.
///
/// # Why it buffers
///
/// [`sponge_leaf`]'s padding flag is `len mod 8`, needed in the capacity before
/// the FIRST permutation, so an incremental sponge cannot start until the total
/// length is known. Inventing a length-free padding rule instead would be a
/// cryptographic decision this port does not get to make.
#[derive(Default, Clone)]
pub struct Rpx256Digest {
    buf: Vec<u8>,
}

impl Rpx256Digest {
    /// The digest of everything absorbed so far.
    pub fn finalize_digest(&self) -> [u8; 32] {
        digest_to_commitment(&sponge_leaf_bytes(&self.buf))
    }
}

impl digest::HashMarker for Rpx256Digest {}

impl digest::OutputSizeUser for Rpx256Digest {
    type OutputSize = digest::typenum::U32;
}

impl digest::Update for Rpx256Digest {
    fn update(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }
}

impl digest::FixedOutput for Rpx256Digest {
    fn finalize_into(self, out: &mut digest::Output<Self>) {
        out.copy_from_slice(&self.finalize_digest());
    }
}

impl digest::Reset for Rpx256Digest {
    fn reset(&mut self) {
        self.buf.clear();
    }
}

impl digest::FixedOutputReset for Rpx256Digest {
    fn finalize_into_reset(&mut self, out: &mut digest::Output<Self>) {
        out.copy_from_slice(&self.finalize_digest());
        self.buf.clear();
    }
}

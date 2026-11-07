//! ZisK's Poseidon1 Fiat–Shamir configuration: the transcript sponge, the
//! grinding digest and the [`TranscriptHash`] that names them.
//!
//! The sponge is [`crate::hash::poseidon1_stark::Transcript`] (checked against
//! ZisK's own code); this module fixes the encoding at the transcript's seams:
//!
//! | seam | encoding |
//! |---|---|
//! | `append_bytes(b)` | `len(b)`, then `b` as 8-byte big-endian felts, the last zero-padded |
//! | `append_field_element(x)` | `x`'s three coefficients |
//! | `state()` | lanes 0..4 of the sponge after a flush, big-endian — the grinding seed |
//! | `sample_field_element()` | three squeezed felts |
//! | `sample_u64(n)` | one squeezed felt, canonical, masked to `n − 1` (one felt per draw) |
//! | grinding | one width-8 permutation of the input's 8-byte felts, zero-padded; digest = lanes 0..4 |
//!
//! Grinding keeps the two-level construction of [`crate::grinding`]: the
//! per-nonce hash is `[inner0..3, nonce, 0, 0, 0]` through the width-8
//! permutation, ZisK's cost (one width-8 permutation per nonce) over our seed
//! layout (ZisK's is `[c0, c1, c2, nonce, 0, 0, 0, 0]`).
//!
//! ⚠ The sponge has no framing between absorbs (`[a]` then `[b]` is `[a, b]`):
//! two transcripts agree only if they make the same calls. Every caller absorbs
//! a schedule fixed by its statement, so prover and verifier stay in step.

use core::num::NonZeroUsize;

use alloc::vec::Vec;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsPrimeField;

use crate::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};
use crate::fiat_shamir::transcript_hash::{HasTranscriptHash, TranscriptHash};
use crate::grinding::{DeviceGrindKey, GrindDigest};
use crate::hash::poseidon1_stark::{self as p1, linear_hash};
use crate::hash::poseidon1_w8;
use crate::hash::rpx::{BYTES_PER_FELT, Fp, digest_to_commitment, felts_from_bytes};

type Ext = FieldElement<Degree3GoldilocksExtensionField>;

/// The grinding digest: one width-8 permutation of the absorbed bytes read
/// as 8-byte big-endian felts, zero-padded to eight lanes; the digest is
/// lanes 0..4. Both grinding hashes fit: the inner input is 41 bytes (6
/// felts), the per-nonce input 40 (5 felts). A longer input, which grinding
/// never produces, takes the width-16 leaf hash so the digest stays total.
#[derive(Clone, Default)]
pub struct P1GrindDigest {
    buf: Vec<u8>,
}

impl P1GrindDigest {
    /// The digest of everything absorbed so far.
    pub fn finalize_digest(&self) -> [u8; 32] {
        let felts = felts_from_bytes(&self.buf);
        if felts.len() > poseidon1_w8::STATE_FELTS {
            return digest_to_commitment(&linear_hash(&felts));
        }
        let mut s = [Fp::zero(); poseidon1_w8::STATE_FELTS];
        s[..felts.len()].copy_from_slice(&felts);
        let out = poseidon1_w8::permute(s);
        digest_to_commitment(&[out[0], out[1], out[2], out[3]])
    }
}

impl GrindDigest for P1GrindDigest {
    /// `math_cuda::grinding::generate_nonce_p1_gpu` (`p1s_grind_w8`): one
    /// width-8 permutation of `[inner0..3, nonce, 0, 0, 0]` per nonce, the
    /// per-nonce half of [`Self::finalize_digest`] over 40 bytes.
    const DEVICE_GRIND: Option<DeviceGrindKey> = Some(DeviceGrindKey::Poseidon1);
}

impl digest::HashMarker for P1GrindDigest {}

impl digest::OutputSizeUser for P1GrindDigest {
    type OutputSize = digest::typenum::U32;
}

impl digest::Update for P1GrindDigest {
    fn update(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }
}

impl digest::FixedOutput for P1GrindDigest {
    fn finalize_into(self, out: &mut digest::Output<Self>) {
        out.copy_from_slice(&self.finalize_digest());
    }
}

impl digest::Reset for P1GrindDigest {
    fn reset(&mut self) {
        self.buf.clear();
    }
}

impl digest::FixedOutputReset for P1GrindDigest {
    fn finalize_into_reset(&mut self, out: &mut digest::Output<Self>) {
        out.copy_from_slice(&self.finalize_digest());
        self.buf.clear();
    }
}

/// The Fiat–Shamir configuration: what grinding computes over. The challenge
/// stream is [`P1Transcript`], which callers build and hand to the prover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct P1TranscriptHash;

impl TranscriptHash for P1TranscriptHash {
    type Digest = P1GrindDigest;
    const REVERSES_SQUEEZE: bool = false;
    /// Structural: a squeeze is canonical felts.
    const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize> = NonZeroUsize::new(1);
    const NAME: &'static str = "poseidon1-w16";
}

/// ZisK's transcript sponge under the encodings of the module table.
#[derive(Clone)]
pub struct P1Transcript {
    sponge: p1::Transcript,
}

impl Default for P1Transcript {
    fn default() -> Self {
        Self::new()
    }
}

impl P1Transcript {
    /// An empty transcript.
    pub fn new() -> Self {
        Self {
            sponge: p1::Transcript::new(),
        }
    }

    /// A fresh transcript with `seed` absorbed as its first `append_bytes`.
    pub fn with_seed(seed: &[u8]) -> Self {
        let mut t = Self::new();
        <Self as IsTranscript<Degree3GoldilocksExtensionField>>::append_bytes(&mut t, seed);
        t
    }

    /// The felts `append_bytes(bytes)` absorbs: the length, then the bytes in
    /// 8-byte big-endian groups, the last zero-padded.
    pub fn append_bytes_felts(bytes: &[u8]) -> Vec<Fp> {
        let mut felts = Vec::with_capacity(1 + bytes.len().div_ceil(BYTES_PER_FELT));
        felts.push(Fp::from(bytes.len() as u64));
        felts.extend(felts_from_bytes(bytes));
        felts
    }
}

impl HasTranscriptHash for P1Transcript {
    type Hash = P1TranscriptHash;
}

impl IsTranscript<Degree3GoldilocksExtensionField> for P1Transcript {
    fn append_field_element(&mut self, element: &Ext) {
        self.sponge.put(element.value());
    }

    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.sponge.put(&Self::append_bytes_felts(new_bytes));
    }

    /// Lanes 0..4 after a flush, canonical big-endian. Flushes a copy, so the
    /// stream itself is unchanged: prover and verifier read it at the same
    /// points and agree.
    fn state(&self) -> [u8; 32] {
        let s = self.sponge.clone().state();
        digest_to_commitment(&[s[0], s[1], s[2], s[3]])
    }

    fn sample_field_element(&mut self) -> Ext {
        Ext::new(self.sponge.field())
    }

    /// Constant consumption: one felt per draw, masked to a power-of-two bound.
    /// Any other bound (zero included) is refused in every build: the mask
    /// would bias it, and no caller may draw that way (REV-P1W-JUDGE V1).
    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        assert!(
            upper_bound.is_power_of_two(),
            "sample_u64 is masked, so a non-power-of-two bound ({upper_bound}) would be biased"
        );
        GoldilocksField::canonical(self.sponge.squeeze1().value()) & (upper_bound.wrapping_sub(1))
    }
}

impl IsStarkTranscript<Degree3GoldilocksExtensionField, GoldilocksField> for P1Transcript {}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(i: u64) -> Fp {
        Fp::from(i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    fn fee(i: u64) -> Ext {
        Ext::new([fe(3 * i), fe(3 * i + 1), fe(3 * i + 2)])
    }

    fn state(t: &P1Transcript) -> [u8; 32] {
        <P1Transcript as IsTranscript<Degree3GoldilocksExtensionField>>::state(t)
    }

    fn sample_u64(t: &mut P1Transcript, bound: u64) -> u64 {
        <P1Transcript as IsTranscript<Degree3GoldilocksExtensionField>>::sample_u64(t, bound)
    }

    #[test]
    fn the_grinding_hash_is_one_width_8_permutation_per_nonce() {
        let seed = [7u8; 32];
        let factor = 8u8;
        let inner = crate::grinding::inner_hash_felts::<P1GrindDigest>(&seed, factor);
        let nonce = crate::grinding::generate_nonce_smallest::<P1GrindDigest>(&seed, factor)
            .expect("a nonce at 8 bits");
        assert!(crate::grinding::is_valid_nonce::<P1GrindDigest>(
            &seed, nonce, factor
        ));
        // The per-nonce hash, by hand: [inner0..3, nonce, 0, 0, 0] through W8.
        let mut s = [Fp::zero(); 8];
        for (lane, v) in s.iter_mut().zip(inner) {
            *lane = Fp::from(v);
        }
        s[4] = Fp::from(nonce);
        let lane0 = GoldilocksField::canonical(poseidon1_w8::permute(s)[0].value());
        assert!(lane0 < 1u64 << (64 - factor), "lane 0 meets the factor");
        if nonce > 0 {
            assert!(!crate::grinding::is_valid_nonce::<P1GrindDigest>(
                &seed,
                nonce - 1,
                factor
            ));
        }
    }

    #[test]
    fn the_transcript_binds_lengths_bytes_and_field_elements() {
        let first = |f: &dyn Fn(&mut P1Transcript)| {
            let mut t = P1Transcript::new();
            f(&mut t);
            <P1Transcript as IsTranscript<Degree3GoldilocksExtensionField>>::sample_field_element(
                &mut t,
            )
        };
        let root = [9u8; 32];
        let a = first(&|t| t.append_bytes(&root));
        // A 32-byte root whose tail is zero, against the 8-byte integer it starts with.
        let mut short_root = [0u8; 32];
        short_root[..8].copy_from_slice(&root[..8]);
        assert_ne!(
            first(&|t| t.append_bytes(&short_root)),
            first(&|t| t.append_bytes(&root[..8]))
        );
        for i in 0..root.len() {
            let mut bad = root;
            bad[i] ^= 1;
            assert_ne!(first(&|t| t.append_bytes(&bad)), a, "byte {i}");
        }
        let x = fee(5);
        let b = first(&|t| t.append_field_element(&x));
        for k in 0..3 {
            let mut v = *x.value();
            v[k] += Fp::one();
            assert_ne!(
                first(&|t| t.append_field_element(&Ext::new(v))),
                b,
                "coefficient {k}"
            );
        }
    }

    #[test]
    fn reading_the_state_does_not_move_the_stream() {
        let mut a = P1Transcript::with_seed(b"seed");
        let mut b = P1Transcript::with_seed(b"seed");
        a.append_bytes(&[1, 2, 3]);
        b.append_bytes(&[1, 2, 3]);
        let s = state(&a);
        assert_eq!(s, state(&b));
        for _ in 0..20 {
            assert_eq!(sample_u64(&mut a, 1 << 20), sample_u64(&mut b, 1 << 20));
        }
        assert_ne!(state(&a), s, "squeezing moves the state");
    }

    #[test]
    fn append_bytes_absorbs_the_length_then_the_padded_felts() {
        // The encoding, checked against the raw sponge.
        let bytes: Vec<u8> = (1..=13u8).collect();
        let mut t = P1Transcript::new();
        t.append_bytes(&bytes);
        let mut raw = p1::Transcript::new();
        raw.put(&[Fp::from(13u64)]);
        raw.put(&felts_from_bytes(&bytes));
        assert_eq!(t.sponge.squeeze1(), raw.squeeze1());
        assert_eq!(P1Transcript::append_bytes_felts(&bytes).len(), 1 + 2);
    }

    #[test]
    fn a_squeezed_index_stays_below_its_bound() {
        let mut t = P1Transcript::with_seed(b"bound");
        for log in [0u32, 1, 5, 20, 63] {
            let bound = 1u64 << log;
            for _ in 0..8 {
                assert!(sample_u64(&mut t, bound) < bound);
            }
        }
    }

    /// The width-8 grind credits its bits (REV-P1-B): `is_valid_nonce` under
    /// the P1 digest is exactly "lane 0 of W8(inner ‖ nonce ‖ 0³), canonical,
    /// below 2^(64 − g)", the inner digest binds the seed and the factor, and
    /// the pass rate over nonces is 2^−g.
    #[test]
    fn the_width8_grind_credits_its_bits() {
        use crate::grinding::is_valid_nonce;
        use crate::hash::rpx::commitment_to_digest;
        use digest::Digest;

        let seed: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0x5a);
        let g = 6u8;
        // The definition, recomputed from the permutation.
        let inner: [u8; 32] = {
            let mut d = P1GrindDigest::new();
            Digest::update(&mut d, [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xed]);
            Digest::update(&mut d, seed);
            Digest::update(&mut d, [g]);
            d.finalize().into()
        };
        let lane0 = |nonce: u64| {
            let i = commitment_to_digest(&inner);
            let mut s = [Fp::zero(); poseidon1_w8::STATE_FELTS];
            s[..4].copy_from_slice(&i);
            s[4] = Fp::from(nonce);
            GoldilocksField::canonical(poseidon1_w8::permute(s)[0].value())
        };
        let n = 1u64 << 14;
        let mut passes = 0u64;
        for nonce in 0..n {
            let ok = is_valid_nonce::<P1GrindDigest>(&seed, nonce, g);
            assert_eq!(ok, lane0(nonce) < 1u64 << (64 - g), "nonce {nonce}");
            passes += u64::from(ok);
        }
        // Binomial(2^14, 2^-6): mean 256, sd ≈ 15.9; ±6 sd.
        assert!((160..=352).contains(&passes), "{passes} passes of {n}");
        // The inner digest binds the seed: one flipped seed bit moves the
        // passing set.
        let mut other = seed;
        other[31] ^= 1;
        assert!(
            (0..1024u64).any(|k| is_valid_nonce::<P1GrindDigest>(&other, k, g)
                != is_valid_nonce::<P1GrindDigest>(&seed, k, g)),
            "the seed moves the passing set"
        );
    }

    #[test]
    fn the_configuration_names_its_grind_digest() {
        assert_eq!(
            <P1TranscriptHash as TranscriptHash>::NAME,
            "poseidon1-w16",
            "the name P1 artifacts carry"
        );
        assert!(matches!(
            <P1GrindDigest as GrindDigest>::DEVICE_GRIND,
            Some(DeviceGrindKey::Poseidon1)
        ));
    }
}

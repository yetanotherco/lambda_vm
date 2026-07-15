use crate::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};

use crate::hash::platform_keccak::PlatformKeccak256 as Keccak256;
use core::marker::PhantomData;
use digest::Digest;
use math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField, IsSubFieldOf},
    },
    traits::ByteConversion,
};
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

pub struct DefaultTranscript<F: HasDefaultTranscript> {
    hasher: Keccak256,
    /// Byte reservoir feeding [`IsTranscript::sample_bits`]: bytes squeezed
    /// from the sponge that have not been consumed yet. A single 32-byte
    /// squeeze serves many small index samples — FRI query indices need only
    /// `log2(domain)` bits each — instead of one Keccak permutation per index.
    /// Cleared on every absorb so sampled bits stay bound to the full prior
    /// transcript (mirrors Plonky3's `HashChallenger`, which clears its output
    /// buffer on `observe`).
    sample_buffer: Vec<u8>,
    /// Number of bytes at the front of `sample_buffer` already consumed.
    sample_cursor: usize,
    /// Count of Keccak squeezes performed (`sample()` calls). A cheap profiling
    /// counter for the verifier's Keccak-permutation budget — the FRI query
    /// phase dominates it, which is exactly what `sample_bits` amortizes.
    keccak_squeezes: usize,
    phantom: PhantomData<F>,
}

impl<F: HasDefaultTranscript> Clone for DefaultTranscript<F> {
    fn clone(&self) -> Self {
        Self {
            hasher: self.hasher.clone(),
            sample_buffer: self.sample_buffer.clone(),
            sample_cursor: self.sample_cursor,
            keccak_squeezes: self.keccak_squeezes,
            phantom: PhantomData,
        }
    }
}

impl<F> DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
{
    pub fn new(data: &[u8]) -> Self {
        let mut res = Self {
            hasher: Keccak256::new(),
            sample_buffer: Vec::new(),
            sample_cursor: 0,
            keccak_squeezes: 0,
            phantom: PhantomData,
        };
        res.append_bytes(data);
        res
    }

    pub fn sample(&mut self) -> [u8; 32] {
        let mut result_hash: [u8; 32] = self.hasher.finalize_reset().into();
        result_hash.reverse();
        self.hasher.update(result_hash);
        self.keccak_squeezes += 1;
        result_hash
    }

    /// Number of Keccak squeezes (`sample()` calls) performed so far. The FRI
    /// query phase dominates the verifier's Keccak cost; `sample_bits`
    /// amortizes one squeeze across many query indices, so this counter drops
    /// by roughly `256 / bits` on that phase.
    pub fn keccak_squeezes(&self) -> usize {
        self.keccak_squeezes
    }

    /// Pull `n` fresh reservoir bytes (`n <= 8`) as a big-endian integer,
    /// refilling from the sponge in 32-byte squeezes when drained. Big-endian
    /// assembly matches the existing `sample_u64` byte convention.
    fn next_reservoir_bytes(&mut self, n: usize) -> u64 {
        let mut acc: u64 = 0;
        for _ in 0..n {
            if self.sample_cursor == self.sample_buffer.len() {
                let block = self.sample();
                self.sample_buffer.clear();
                self.sample_buffer.extend_from_slice(&block);
                self.sample_cursor = 0;
            }
            acc = (acc << 8) | self.sample_buffer[self.sample_cursor] as u64;
            self.sample_cursor += 1;
        }
        acc
    }
}

impl<F> Default for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
{
    fn default() -> Self {
        Self::new(&[])
    }
}

impl<F> IsTranscript<F> for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
{
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.hasher.update(new_bytes);
        // Absorbing new data invalidates any leftover squeezed bits: a later
        // `sample_bits` must reflect everything absorbed so far. `append_field_element`
        // routes through here, so field-element absorbs clear the reservoir too.
        self.sample_buffer.clear();
        self.sample_cursor = 0;
    }

    fn append_field_element(&mut self, element: &FieldElement<F>) {
        self.append_bytes(&element.to_bytes_be());
    }

    fn state(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

    fn sample_field_element(&mut self) -> FieldElement<F> {
        let mut rng = <ChaCha20Rng as SeedableRng>::from_seed(self.sample());
        F::get_random_field_element_from_rng(&mut rng)
    }

    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        assert!(upper_bound > 0, "upper_bound must be greater than 0");
        let threshold = upper_bound.wrapping_neg() % upper_bound;
        loop {
            let candidate = u64::from_be_bytes(self.sample()[..8].try_into().unwrap());
            if candidate >= threshold {
                return candidate % upper_bound;
            }
        }
    }

    fn sample_bits(&mut self, bits: usize) -> u64 {
        assert!(
            (1..64).contains(&bits),
            "sample_bits: bits must be in 1..=63"
        );
        // Power-of-two range: masking is exactly uniform, so no rejection is
        // needed. Draw the fewest whole bytes covering `bits` from the shared
        // reservoir and keep the low `bits`.
        let raw = self.next_reservoir_bytes(bits.div_ceil(8));
        raw & ((1u64 << bits) - 1)
    }
}

impl<F, S> IsStarkTranscript<F, S> for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    S: IsField + IsSubFieldOf<F>,
{
    // nothing to implement: sample_z_ood uses the default body
}

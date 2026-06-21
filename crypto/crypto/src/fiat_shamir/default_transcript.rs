use crate::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};

use core::marker::PhantomData;
use math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField, IsSubFieldOf},
    },
    traits::ByteConversion,
};
use sha3::{Digest, Keccak256};

pub struct DefaultTranscript<F: HasDefaultTranscript> {
    hasher: Keccak256,
    phantom: PhantomData<F>,
}

impl<F: HasDefaultTranscript> Clone for DefaultTranscript<F> {
    fn clone(&self) -> Self {
        Self {
            hasher: self.hasher.clone(),
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
            phantom: PhantomData,
        };
        res.append_bytes(data);
        res
    }

    pub fn sample(&mut self) -> [u8; 32] {
        let mut result_hash: [u8; 32] = self.hasher.finalize_reset().into();
        result_hash.reverse();
        self.hasher.update(result_hash);
        result_hash
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
    }

    fn append_field_element(&mut self, element: &FieldElement<F>) {
        // `to_bytes_be` returns a fixed-size array (no allocation); feed its
        // bytes straight to the hasher. This is a hot path in verification.
        self.append_bytes(element.to_bytes_be().as_ref());
    }

    fn state(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

    fn sample_field_element(&mut self) -> FieldElement<F> {
        // Squeeze field-element entropy directly from the transcript's Keccak
        // sponge instead of seeding a per-call ChaCha20 PRG. Each `self.sample()`
        // returns a fresh 32-byte squeeze block; the field type pulls the limbs it
        // needs from one block (rejection-resampling only on the ~1-in-4-billion
        // out-of-range draw). This reuses the Keccak permutation precompile already
        // backing the transcript and drops the `rand_chacha` dependency.
        F::sample_field_element_from_squeeze(|| self.sample())
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
}

impl<F, S> IsStarkTranscript<F, S> for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    S: IsField + IsSubFieldOf<F>,
{
    // nothing to implement: sample_z_ood uses the default body
}

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
}

impl<F, S> IsStarkTranscript<F, S> for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    S: IsField + IsSubFieldOf<F>,
{
    // nothing to implement: sample_z_ood uses the default body
}

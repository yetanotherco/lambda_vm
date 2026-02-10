use super::errors::InsecureOptionError;
use math::field::traits::IsPrimeField;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::wasm_bindgen;

/// The options for the proof
///
/// - `blowup_factor`: the blowup factor for the trace
/// - `fri_number_of_queries`: the number of queries for the FRI layer
/// - `coset_offset`: the offset for the coset
/// - `grinding_factor`: the number of leading zeros that we want for the Hash(hash || nonce)
#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProofOptions {
    pub blowup_factor: u8,
    pub fri_number_of_queries: usize,
    pub coset_offset: u64,
    pub grinding_factor: u8,
}

impl ProofOptions {
    // TODO: Make it work for extended fields
    const EXTENSION_DEGREE: usize = 1;
    // Estimated maximum domain size. 2^40 = 1 TB
    const NUM_BITS_MAX_DOMAIN_SIZE: usize = 40;

    /// Checks security of proof options given 128 bits of security
    pub fn new_with_checked_security<F: IsPrimeField>(
        blowup_factor: u8,
        fri_number_of_queries: usize,
        coset_offset: u64,
        grinding_factor: u8,
        security_target: u8,
    ) -> Result<Self, InsecureOptionError> {
        Self::check_field_security::<F>(security_target)?;

        let num_bits_blowup_factor = blowup_factor.trailing_zeros() as usize;

        if security_target as usize
            >= grinding_factor as usize + num_bits_blowup_factor * fri_number_of_queries - 1
        {
            return Err(InsecureOptionError::LowSecurityBits);
        }

        Ok(ProofOptions {
            blowup_factor,
            fri_number_of_queries,
            coset_offset,
            grinding_factor,
        })
    }

    /// Checks provable security of proof options given 128 bits of security
    /// This is an approximation. It's stricter than the formula in the paper.
    /// See https://eprint.iacr.org/2021/582.pdf
    pub fn new_with_checked_provable_security<F: IsPrimeField>(
        blowup_factor: u8,
        fri_number_of_queries: usize,
        coset_offset: u64,
        grinding_factor: u8,
        security_target: u8,
    ) -> Result<Self, InsecureOptionError> {
        Self::check_field_security::<F>(security_target)?;

        let num_bits_blowup_factor = blowup_factor.leading_zeros() as usize;

        if (security_target as usize)
            < grinding_factor as usize + num_bits_blowup_factor * fri_number_of_queries / 2
        {
            return Err(InsecureOptionError::LowSecurityBits);
        }

        Ok(ProofOptions {
            blowup_factor,
            fri_number_of_queries,
            coset_offset,
            grinding_factor,
        })
    }

    fn check_field_security<F: IsPrimeField>(
        security_target: u8,
    ) -> Result<(), InsecureOptionError> {
        if F::field_bit_size() * Self::EXTENSION_DEGREE
            <= security_target as usize + Self::NUM_BITS_MAX_DOMAIN_SIZE
        {
            return Err(InsecureOptionError::FieldSize);
        }

        Ok(())
    }

    /// Insecure proof options for fast tests only. Never use in production.
    pub fn default_test_options() -> Self {
        Self {
            blowup_factor: 4,
            fri_number_of_queries: 3,
            coset_offset: 3,
            grinding_factor: 1,
        }
    }
}

/// Production-secure defaults (100-bit provable security).
impl Default for ProofOptions {
    fn default() -> Self {
        Self {
            blowup_factor: 4,
            fri_number_of_queries: 104,
            coset_offset: 3,
            grinding_factor: 20,
        }
    }
}

#[cfg(test)]
mod tests {
    use math::field::fields::{
        fft_friendly::stark_252_prime_field::Stark252PrimeField, u64_prime_field::F17,
    };

    use crate::proof::errors::InsecureOptionError;

    use super::ProofOptions;

    const BLOWUP_FACTOR: u8 = 4;
    const COSET_OFFSET: u64 = 1;
    const GRINDING_FACTOR: u8 = 20;

    // FRI queries needed per security level (conjecturable).
    // See section 5.10.1 of https://eprint.iacr.org/2021/582.pdf
    const FRI_QUERIES_80_BITS: usize = 31;
    const FRI_QUERIES_100_BITS: usize = 41;
    const FRI_QUERIES_128_BITS: usize = 55;

    #[test]
    fn u64_prime_field_is_not_large_enough_to_be_secure() {
        let u64_options = ProofOptions::new_with_checked_security::<F17>(
            BLOWUP_FACTOR,
            FRI_QUERIES_128_BITS,
            COSET_OFFSET,
            GRINDING_FACTOR,
            128,
        );
        assert!(matches!(u64_options, Err(InsecureOptionError::FieldSize)));
    }

    #[test]
    fn conjecturable_128_bits_are_secure() {
        let secure_options = ProofOptions::new_with_checked_security::<Stark252PrimeField>(
            BLOWUP_FACTOR,
            FRI_QUERIES_128_BITS,
            COSET_OFFSET,
            GRINDING_FACTOR,
            128,
        );
        assert!(secure_options.is_ok());
    }

    #[test]
    fn conjecturable_128_bits_with_one_fri_query_less_are_insecure() {
        let insecure_options = ProofOptions::new_with_checked_security::<Stark252PrimeField>(
            BLOWUP_FACTOR,
            FRI_QUERIES_128_BITS - 1,
            COSET_OFFSET,
            GRINDING_FACTOR,
            128,
        );
        assert!(matches!(
            insecure_options,
            Err(InsecureOptionError::LowSecurityBits)
        ));
    }

    #[test]
    fn conjecturable_100_bits_are_secure() {
        let secure_options = ProofOptions::new_with_checked_security::<Stark252PrimeField>(
            BLOWUP_FACTOR,
            FRI_QUERIES_100_BITS,
            COSET_OFFSET,
            GRINDING_FACTOR,
            100,
        );
        assert!(secure_options.is_ok());
    }

    #[test]
    fn conjecturable_80_bits_are_secure() {
        let secure_options = ProofOptions::new_with_checked_security::<Stark252PrimeField>(
            BLOWUP_FACTOR,
            FRI_QUERIES_80_BITS,
            COSET_OFFSET,
            GRINDING_FACTOR,
            80,
        );
        assert!(secure_options.is_ok());
    }
}

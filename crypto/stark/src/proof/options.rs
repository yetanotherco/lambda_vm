use core::fmt;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::wasm_bindgen;

/// Error returned when proof options are invalid.
#[derive(Debug, Clone)]
pub enum ProofOptionsError {
    /// blowup_factor must be a power of 2 >= 2
    InvalidBlowup(u8),
    /// security_bits must exceed grinding_factor
    SecurityTooLow {
        security_bits: u8,
        grinding_factor: u8,
    },
}

impl fmt::Display for ProofOptionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBlowup(b) => {
                write!(f, "blowup_factor must be a power of 2 >= 2, got {b}")
            }
            Self::SecurityTooLow {
                security_bits,
                grinding_factor,
            } => write!(
                f,
                "security_bits ({security_bits}) must exceed grinding_factor ({grinding_factor})"
            ),
        }
    }
}

/// The options for the proof
///
/// - `blowup_factor`: the blowup factor for the trace
/// - `fri_number_of_queries`: the number of queries for the FRI layer
/// - `coset_offset`: the offset for the coset
/// - `grinding_factor`: the number of leading zeros that we want for the Hash(hash || nonce)
#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
pub struct ProofOptions {
    pub blowup_factor: u8,
    pub fri_number_of_queries: usize,
    pub coset_offset: u64,
    pub grinding_factor: u8,
}

impl ProofOptions {
    /// Default proof options used for testing purposes.
    /// These options should never be used in production.
    pub fn default_test_options() -> Self {
        Self {
            blowup_factor: 2,
            fri_number_of_queries: 3,
            coset_offset: 3,
            grinding_factor: 1,
        }
    }
}

/// Proof options builder for Goldilocks **cubic** extension field (degree 3).
///
/// Goldilocks base field: 64 bits (p = 2^64 - 2^32 + 1)
/// Cubic extension: degree 3 (w^3 = 2), giving 192-bit effective field size.
///
/// Computes FRI query count using the Johnson Bound Regime (JBR):
///   proximity = 1 - sqrt(1/blowup) - 1/300
///   bits_per_query = -log2(1 - proximity)
///   queries = ceil((security_bits - grinding) / bits_per_query)
///
/// The 192-bit effective field comfortably supports up to 152-bit security
/// (192 - 40 bits max domain), so the FRI query count is always the
/// security bottleneck — field size is not.
pub struct GoldilocksCubicProofOptions;

impl GoldilocksCubicProofOptions {
    const DEFAULT_GRINDING: u8 = 20;

    /// Create proof options targeting 128-bit security with default grinding (20 bits).
    ///
    /// `blowup_factor` must be a power of 2 >= 2 (e.g., 2, 4, 8, 16, 32, 64).
    pub fn with_blowup(blowup_factor: u8) -> Result<ProofOptions, ProofOptionsError> {
        Self::with_params(blowup_factor, 128, Self::DEFAULT_GRINDING)
    }

    /// Create proof options with custom security target and grinding factor.
    pub fn with_params(
        blowup_factor: u8,
        security_bits: u8,
        grinding_factor: u8,
    ) -> Result<ProofOptions, ProofOptionsError> {
        if !blowup_factor.is_power_of_two() || blowup_factor < 2 {
            return Err(ProofOptionsError::InvalidBlowup(blowup_factor));
        }
        if security_bits <= grinding_factor {
            return Err(ProofOptionsError::SecurityTooLow {
                security_bits,
                grinding_factor,
            });
        }

        #[cfg(feature = "std")]
        let (sqrt, log2, ceil) = (
            f64::sqrt as fn(f64) -> f64,
            f64::log2 as fn(f64) -> f64,
            f64::ceil as fn(f64) -> f64,
        );
        #[cfg(not(feature = "std"))]
        let (sqrt, log2, ceil) = (
            libm::sqrt as fn(f64) -> f64,
            libm::log2 as fn(f64) -> f64,
            libm::ceil as fn(f64) -> f64,
        );

        let rate = 1.0 / blowup_factor as f64;
        let proximity = 1.0 - sqrt(rate) - 1.0 / 300.0;
        let bits_per_query = -log2(1.0 - proximity);
        let fri_number_of_queries =
            ceil((security_bits as f64 - grinding_factor as f64) / bits_per_query) as usize;

        Ok(ProofOptions {
            blowup_factor,
            fri_number_of_queries,
            coset_offset: 3,
            grinding_factor,
        })
    }
}

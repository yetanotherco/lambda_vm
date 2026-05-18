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
    /// fri_folding_factor must be a power of 2 >= 2
    InvalidFoldingFactor(usize),
    /// fri_last_layer_degree_bound must be 0, or (bound + 1) must be a power of 2
    InvalidDegreeBound(usize),
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
            Self::InvalidFoldingFactor(ff) => write!(
                f,
                "fri_folding_factor must be a power of 2 >= 2, got {ff}"
            ),
            Self::InvalidDegreeBound(b) => write!(
                f,
                "fri_last_layer_degree_bound must be 0 or (bound + 1) must be a power of 2, \
                 got {b}"
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
/// - `fri_last_layer_degree_bound`: stop FRI folding when the polynomial degree
///   reaches this bound, send remaining coefficients directly. `0` (default)
///   means fold all the way down to a constant (current behavior). For early
///   stopping, set to e.g. 7 (last poly of degree ≤ 7, sent as 8 coefficients).
///   When non-zero, `(bound + 1)` must be a power of two.
/// - `fri_folding_factor`: how many binary folds per committed FRI layer.
///   `2` (default) keeps current behavior (one fold per layer). Higher
///   factors (4, 8, ...) group multiple folds into one Merkle commit,
///   reducing the number of layers (and Merkle trees + their roots in the
///   proof). Must be a power of two ≥ 2.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProofOptions {
    pub blowup_factor: u8,
    pub fri_number_of_queries: usize,
    pub coset_offset: u64,
    pub grinding_factor: u8,
    /// FRI early-stopping bound. `0` means fold to constant (legacy behavior).
    /// Default: `0` so existing call sites keep their current FRI shape.
    #[serde(default)]
    pub fri_last_layer_degree_bound: usize,
    /// FRI folding factor (binary folds per committed layer). `0` means
    /// "use 2" (the legacy default); on (de)serialization of older proofs
    /// without this field, `serde(default)` gives `0` which is then
    /// normalized to 2 via [`Self::effective_folding_factor`]. Setting to a
    /// power-of-two ≥ 2 enables higher folding.
    #[serde(default)]
    pub fri_folding_factor: usize,
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
            fri_last_layer_degree_bound: 0,
            fri_folding_factor: 2,
        }
    }

    /// Normalize `fri_folding_factor`: the default-constructed `0` (from old
    /// proofs without this field, or callers that haven't set it) is treated
    /// as `2`, the legacy folding factor.
    pub fn effective_folding_factor(&self) -> usize {
        match self.fri_folding_factor {
            0 => 2,
            ff => ff,
        }
    }

    /// Validate that the FRI fields hold the invariants the prover/verifier
    /// rely on. Other fields (`blowup_factor`, security) are validated by the
    /// `GoldilocksCubicProofOptions` builder; this method just covers the
    /// extra invariants introduced by FRI early-stopping + folding.
    pub fn validate(&self) -> Result<(), ProofOptionsError> {
        let ff = self.effective_folding_factor();
        if ff < 2 || !ff.is_power_of_two() {
            return Err(ProofOptionsError::InvalidFoldingFactor(ff));
        }
        let bound = self.fri_last_layer_degree_bound;
        if bound != 0 && !(bound + 1).is_power_of_two() {
            return Err(ProofOptionsError::InvalidDegreeBound(bound));
        }
        Ok(())
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

        let rate = 1.0 / blowup_factor as f64;
        let proximity = 1.0 - rate.sqrt() - 1.0 / 300.0;
        let bits_per_query = -(1.0 - proximity).log2();
        let fri_number_of_queries =
            ((security_bits as f64 - grinding_factor as f64) / bits_per_query).ceil() as usize;

        Ok(ProofOptions {
            blowup_factor,
            fri_number_of_queries,
            coset_offset: 3,
            grinding_factor,
            // Production defaults targeting smaller proofs without changing
            // soundness: fold by 4 (one Merkle commit per 2 binary folds saved),
            // stop early when degree ≤ 7 (send 8 coefficients directly).
            // The verifier still does the same total amount of folding work;
            // only the number of FRI Merkle trees + serialized layers shrinks.
            fri_last_layer_degree_bound: 7,
            fri_folding_factor: 4,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_legacy_defaults() {
        // Existing `default_test_options()` uses folding_factor=2,
        // last_layer_degree_bound=0 (legacy). Must validate.
        ProofOptions::default_test_options().validate().unwrap();
    }

    #[test]
    fn validate_treats_zero_folding_factor_as_two() {
        // Proofs deserialized from older versions (no fri_folding_factor field)
        // come in with `0`. `effective_folding_factor` normalizes to 2 and
        // validate must accept that.
        let opts = ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 3,
            coset_offset: 3,
            grinding_factor: 1,
            fri_last_layer_degree_bound: 0,
            fri_folding_factor: 0,
        };
        assert_eq!(opts.effective_folding_factor(), 2);
        opts.validate().unwrap();
    }

    #[test]
    fn validate_rejects_non_power_of_two_folding_factor() {
        let mut opts = ProofOptions::default_test_options();
        opts.fri_folding_factor = 3;
        assert!(matches!(
            opts.validate(),
            Err(ProofOptionsError::InvalidFoldingFactor(3))
        ));
    }

    #[test]
    fn validate_rejects_folding_factor_below_two() {
        let mut opts = ProofOptions::default_test_options();
        opts.fri_folding_factor = 1;
        assert!(matches!(
            opts.validate(),
            Err(ProofOptionsError::InvalidFoldingFactor(1))
        ));
    }

    #[test]
    fn validate_accepts_valid_degree_bounds() {
        for bound in [0usize, 1, 3, 7, 15, 31, 63] {
            let mut opts = ProofOptions::default_test_options();
            opts.fri_last_layer_degree_bound = bound;
            opts.validate()
                .unwrap_or_else(|e| panic!("expected bound={bound} to validate, got {e}"));
        }
    }

    #[test]
    fn validate_rejects_degree_bound_not_pow2_minus_one() {
        let mut opts = ProofOptions::default_test_options();
        opts.fri_last_layer_degree_bound = 5; // (5+1)=6 is not a power of two
        assert!(matches!(
            opts.validate(),
            Err(ProofOptionsError::InvalidDegreeBound(5))
        ));
    }

    #[test]
    fn goldilocks_cubic_builder_picks_production_defaults() {
        let opts = GoldilocksCubicProofOptions::with_blowup(2).unwrap();
        assert_eq!(opts.fri_last_layer_degree_bound, 7);
        assert_eq!(opts.fri_folding_factor, 4);
        opts.validate().unwrap();
    }
}

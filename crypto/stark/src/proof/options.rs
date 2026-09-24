use core::fmt;
use core::str::FromStr;

pub use crypto::merkle_tree::cap::CapPolicy;

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
/// - `fri_final_poly_log_degree`: log2 degree bound at which FRI terminates folding
/// - `format`: the proof FORMAT ([`ProofFormat`], the ZF campaign's levers).
///   Its default is today's format, byte for byte.
///
/// # The format is not serialized
///
/// `format` is skipped by serde and rkyv (and restored to its default on
/// deserialize), so a serialized `ProofOptions` has exactly the bytes it had
/// before the field existed. Nothing repo-wide was found to serialize a
/// `ProofOptions` into pinned bytes (the one by-value holder, `AirContext`,
/// derives neither), and skipping it makes that true by construction rather
/// than by search. The format is a verifier-side constant: it comes from the
/// code that builds the options, never from bytes a prover supplied — a
/// proof never carries it.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct ProofOptions {
    pub blowup_factor: u8,
    pub fri_number_of_queries: usize,
    pub coset_offset: u64,
    pub grinding_factor: u8,
    /// Log2 of the FRI final-polynomial degree bound. FRI stops folding when the
    /// polynomial has degree < 2^fri_final_poly_log_degree; the prover sends those
    /// 2^k coefficients instead of folding to a constant.
    pub fri_final_poly_log_degree: u8,
    /// The proof format. [`ProofFormat::DEFAULT`] = today. Not serialized.
    #[serde(skip)]
    #[rkyv(with = rkyv::with::Skip)]
    #[cfg_attr(feature = "wasm", wasm_bindgen(skip))]
    pub format: ProofFormat,
}

/// The proof-format levers of a univariate STARK proof. Grouped so a literal
/// `ProofOptions` names the format in one line (`format: ProofFormat::DEFAULT`)
/// and a lever added later touches this struct only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ProofFormat {
    /// Merkle cap policy for every tree of the proof (S1). `Off` = today.
    pub merkle_cap: CapPolicy,
    /// FRI fold schedule of the committed layers (S3). `Pair` = today.
    pub fri_mode: FriMode,
    /// One-row trace openings with a committed FRI input (S2). `Off` = today.
    pub one_row: OneRowMode,
}

impl ProofFormat {
    /// Today's format: every lever off.
    pub const DEFAULT: Self = Self {
        merkle_cap: CapPolicy::Off,
        fri_mode: FriMode::Pair,
        one_row: OneRowMode::Off,
    };

    /// True when this is today's format (`Fixed(0)` counts as `Off`).
    pub fn is_default(&self) -> bool {
        self.merkle_cap.is_off()
            && self.fri_mode == FriMode::Pair
            && self.one_row == OneRowMode::Off
    }
}

/// How the committed FRI layers fold (S3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FriMode {
    /// One binary fold per committed layer, pair leaves. Today's format.
    #[default]
    Pair,
    /// Folds of `2^d` per committed layer, `d` chosen by the verifier-side DP.
    Dp,
}

impl FriMode {
    /// The knob spelling (`LAMBDA_VM_ZF_FRI`).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Pair => "pair",
            Self::Dp => "dp",
        }
    }
}

impl fmt::Display for FriMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for FriMode {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "pair" => Ok(Self::Pair),
            "dp" => Ok(Self::Dp),
            _ => Err(()),
        }
    }
}

/// Whether the trace trees commit one LDE row per leaf (S2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OneRowMode {
    /// Row-pair leaves, the DEEP pair rebuilt from trace openings. Today's format.
    #[default]
    Off,
    /// One-row leaves and a committed FRI-input tree for every table.
    On,
    /// Per table, whichever the cost model prefers from the AIR's widths.
    Auto,
}

impl OneRowMode {
    /// The knob spelling (`LAMBDA_VM_ZF_ONE_ROW`).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Off => "0",
            Self::On => "1",
            Self::Auto => "auto",
        }
    }
}

impl fmt::Display for OneRowMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for OneRowMode {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "0" => Ok(Self::Off),
            "1" => Ok(Self::On),
            "auto" => Ok(Self::Auto),
            _ => Err(()),
        }
    }
}

/// Which format levers THIS build implements. A lever that is only parsed —
/// its field exists so the option structs and the `ZF FORMAT` banner stay
/// stable while the campaign lands it — must not be selectable, or a run
/// could print a non-default format and prove the default one. Each lane
/// flips its own flag in the commit that makes the lever real.
///
/// The Merkle cap is real on the host and device STARK provers and the host
/// verifier (design/CAP.md C3 + C4). ⚠ NOT yet in the LFM in-guest verifier
/// (C5): a recursion run that wraps a capped proof fails closed there, so
/// `LAMBDA_VM_ZF_CAP` is for STARK-level tests and measurements until C5
/// lands.
pub const MERKLE_CAP_IMPLEMENTED: bool = true;

/// See [`MERKLE_CAP_IMPLEMENTED`].
pub const FRI_MODE_IMPLEMENTED: bool = false;

/// See [`MERKLE_CAP_IMPLEMENTED`].
pub const ONE_ROW_IMPLEMENTED: bool = false;

impl ProofOptions {
    /// True when every format field is at its default: the proof this
    /// produces is today's format, byte for byte.
    pub fn has_default_format(&self) -> bool {
        self.format.is_default()
    }

    /// Default proof options used for testing purposes.
    /// These options should never be used in production.
    pub fn default_test_options() -> Self {
        Self {
            blowup_factor: 2,
            fri_number_of_queries: 3,
            coset_offset: 3,
            grinding_factor: 1,
            fri_final_poly_log_degree: DEFAULT_FRI_FINAL_POLY_LOG_DEGREE,
            format: ProofFormat::DEFAULT,
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

// Shared by both ProofOptions::default_test_options and GoldilocksCubicProofOptions::with_params.
const DEFAULT_FRI_FINAL_POLY_LOG_DEGREE: u8 = 7;

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
            fri_final_poly_log_degree: DEFAULT_FRI_FINAL_POLY_LOG_DEGREE,
            format: ProofFormat::DEFAULT,
        })
    }
}

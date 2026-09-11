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
/// - `fri_final_poly_log_degree`: log2 degree bound at which FRI terminates folding
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
            fri_final_poly_log_degree: DEFAULT_FRI_FINAL_POLY_LOG_DEGREE,
        }
    }
}

/// Proof options builder for Goldilocks **cubic** extension field (degree 3).
///
/// Goldilocks base field: 64 bits (p = 2^64 - 2^32 + 1)
/// Cubic extension: degree 3 (w^3 = 2), giving 192-bit effective field size.
///
/// Computes the FRI query count in the Johnson Bound Regime (JBR):
///   proximity = 1 - sqrt(1/blowup) - eta
///   bits_per_query = -log2(1 - proximity)
///   queries = ceil((security_bits - grinding) / bits_per_query)
///
/// # ⚠ What `security_bits` is, and what it is not
///
/// Delivered soundness is the WORSE of two independent error terms: the FRI
/// query error sized here, and the commit-phase / proximity-gaps error
/// (`eps_C`). **This module computes only the first.** `security_bits` is
/// therefore a QUERY BUDGET, not the security the system delivers — nothing
/// here models `eps_C`, and grinding attaches to the query round alone, so it
/// cannot buy `eps_C` back either.
///
/// At the production posture the query term is NOT the bottleneck: `eps_C` is,
/// by a wide margin, and neither the query count nor the grinding factor moves
/// it. Only [`johnson_gap`] does. Sizing queries past the point where `eps_C`
/// binds buys nothing and costs proving time.
pub struct GoldilocksCubicProofOptions;

/// The Johnson gap `eta` — a FREE ANALYSIS PARAMETER, not a property of the
/// system, and the only knob in this file that moves delivered soundness.
///
/// It sets how close the decoding radius sits to the Johnson bound, and it
/// TRADES THE TWO ERROR TERMS AGAINST EACH OTHER: a smaller `eta` pushes the
/// radius toward the bound, which maximises `bits_per_query` (cheap queries)
/// and simultaneously MINIMISES `eps_C`. Since soundness is the max of the two,
/// tuning `eta` down past the crossover buys query-term bits that cannot be
/// spent while wrecking the term that actually binds. Equivalently `eta` sets
/// the proximity parameter `m = sqrt(rate) / (2 * eta)`, so this is an `m`
/// re-optimisation — any external result that also re-optimises `m` OVERLAPS
/// with this and must not be multiplied in.
///
/// ⚠ **The values below are calibrated for one posture and are not universal.**
/// The optimum depends on the tallest evaluation domain `|D_0|`, the DEEP batch
/// size `L` and the number of FRI instances the union runs over — none of which
/// `with_params` receives. They are derived for the block posture: blowup 4 at
/// 110 queries and grinding 20, `|D_0| = 2^24` (LOCAL_TO_GLOBAL saturates at
/// 2^22 rows), ~40 per-table FRI instances per epoch and ~32 proof instances
/// per block. Applying them to a materially different shape is unanalysed.
///
/// Blowups without a calibrated value keep the historical `1/300`. That is not
/// an endorsement of it — it is the value whose `eps_C` cost has been measured
/// and is known to be poor, kept because a substitute has not been derived for
/// those shapes. `blowup 8` is in that set deliberately: its DELIVERED bits are
/// not computed, and inheriting a neighbour's constant would state a soundness
/// claim nobody has checked.
fn johnson_gap(blowup_factor: u8) -> f64 {
    match blowup_factor {
        2 => 1.0 / 39.6,
        4 => 1.0 / 32.0,
        _ => 1.0 / 300.0,
    }
}

/// The query budget [`GoldilocksCubicProofOptions::with_blowup`] targets, per
/// blowup.
///
/// It moves WITH [`johnson_gap`] and for one reason: `eta` feeds both `eps_C`
/// and `bits_per_query`, so re-tuning it alone would make each query cheaper
/// and buy MORE of them (110 -> 119 at blowup 4). The pairs below are chosen to
/// leave every production query count exactly where it was — the re-tune is
/// free, and a query count that moves means the pair is wrong.
fn query_budget_bits(blowup_factor: u8) -> u8 {
    match blowup_factor {
        2 => 118,
        4 => 120,
        _ => 128,
    }
}

// Shared by both ProofOptions::default_test_options and GoldilocksCubicProofOptions::with_params.
const DEFAULT_FRI_FINAL_POLY_LOG_DEGREE: u8 = 7;

impl GoldilocksCubicProofOptions {
    const DEFAULT_GRINDING: u8 = 20;

    /// Create proof options at this blowup's query budget, with default
    /// grinding (20 bits).
    ///
    /// The budget is per-blowup ([`query_budget_bits`]) rather than a flat 128
    /// because 128 was never delivered: it sized the query term while `eps_C`
    /// bound far below it. See [`johnson_gap`] for what actually moves
    /// soundness and for the posture these constants are calibrated at.
    ///
    /// `blowup_factor` must be a power of 2 >= 2 (e.g., 2, 4, 8, 16, 32, 64).
    pub fn with_blowup(blowup_factor: u8) -> Result<ProofOptions, ProofOptionsError> {
        Self::with_params(
            blowup_factor,
            query_budget_bits(blowup_factor),
            Self::DEFAULT_GRINDING,
        )
    }

    /// Create proof options with a custom query budget and grinding factor.
    ///
    /// ⚠ `security_bits` sizes the QUERY term only — see the note on
    /// [`GoldilocksCubicProofOptions`]. A caller passing its own budget still
    /// gets this blowup's [`johnson_gap`], which is calibrated for the block
    /// posture; at a materially different shape the resulting `eps_C` is
    /// unanalysed, and the delivered soundness is not the number passed here.
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
        let proximity = 1.0 - rate.sqrt() - johnson_gap(blowup_factor);
        let bits_per_query = -(1.0 - proximity).log2();
        let fri_number_of_queries =
            ((security_bits as f64 - grinding_factor as f64) / bits_per_query).ceil() as usize;

        Ok(ProofOptions {
            blowup_factor,
            fri_number_of_queries,
            coset_offset: 3,
            grinding_factor,
            fri_final_poly_log_degree: DEFAULT_FRI_FINAL_POLY_LOG_DEGREE,
        })
    }
}

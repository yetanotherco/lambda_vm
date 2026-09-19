//! Multilinear machinery for a sumcheck-based proof system: extensions over the
//! Boolean hypercube, batched sumcheck, zerocheck, LogUp-GKR, stacking and
//! WHIR.
//!
//! Not wired into the prover. The codeword domain is a two-adic subgroup of the
//! base field; codeword values live in the extension.

pub mod batch;
pub mod claim_reduce;
pub mod constraint_argument;
pub mod eq;
pub mod gkr;
pub mod gpu;
pub mod logup;
pub mod mle;
pub mod poly;
pub mod program;
pub mod query_count;
pub mod selector;
pub mod stacked_eval;
pub mod stacking;
pub mod sumcheck;
pub mod uneven;
pub mod uni_skip;
pub mod virtual_poly;
pub mod whir;
pub mod whir_chain;
pub mod whir_commit;
pub mod whir_eval;
pub mod whir_hash;
pub mod whir_round;
pub mod whir_split;
pub mod zerocheck;

use math::field::{element::FieldElement, traits::IsField};
use thiserror::Error;

/// Below this a pass stays on the thread that asked for it: handing a slice to
/// the pool costs tens of microseconds whatever is in it, and a sumcheck's last
/// rounds are over a few hundred values.
#[cfg(feature = "parallel")]
pub(crate) const SERIAL_BELOW: usize = 1 << 12;

/// The cube below which a sumcheck's rounds belong here, for a rule the host
/// evaluates directly: a device round costs the same whatever the cube — one
/// thread walks the whole rule at each index — and a host round is linear in it.
pub(crate) const HOST_CUBE_DIRECT: usize = 1 << 9;

/// The same for a rule the host walks through the program interpreter, which
/// costs about ten times a multiplication written out per step while the device
/// round costs the same either way.
pub(crate) const HOST_CUBE_COMPILED: usize = 1 << 5;

/// `[1, gamma, gamma^2, ..]` — the weights a batching challenge expands into.
///
/// `pub` for the recursion emitter, which reproduces these weights inside the
/// field machine: a leg with no host function on the other side of the equals
/// sign can only be gated against a copy of itself, which is no evidence about
/// either. Every caller inside this crate reaches it through the batching
/// routines below.
pub fn challenge_powers<F: IsField>(gamma: &FieldElement<F>, count: usize) -> Vec<FieldElement<F>> {
    let mut acc = FieldElement::<F>::one();
    (0..count)
        .map(|_| {
            let current = acc.clone();
            acc = &acc * gamma;
            current
        })
        .collect()
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("expected a power-of-two number of evaluations, got {0}")]
    NotPowerOfTwo(usize),
    #[error("expected {expected} variables, got {got}")]
    VariableCountMismatch { expected: usize, got: usize },
    #[error("polynomial has no variables left to fold")]
    NoVariablesLeft,
    #[error("term references polynomial index {index}, but only {len} are registered")]
    UnknownPolynomial { index: usize, len: usize },
    #[error("virtual polynomial has no terms")]
    EmptyPolynomial,
    #[error("round {round}: expected a degree-{expected} polynomial, got {got} evaluations")]
    RoundDegreeMismatch {
        round: usize,
        expected: usize,
        got: usize,
    },
    #[error("proof has {got} rounds, expected {expected}")]
    RoundCountMismatch { expected: usize, got: usize },
    #[error("final evaluation does not match the oracle: expected {expected}, got {got}")]
    FinalEvaluationMismatch { expected: String, got: String },
    #[error("{exemptions} exempted steps exceed the {size}-step trace")]
    TooManyExemptions { exemptions: usize, size: usize },
    #[error("layer {layer}: the sumcheck residual does not match the fraction-fold relation")]
    LayerRelationMismatch { layer: usize },
    #[error("a column on {column_vars} variables does not fit a {n_stack}-variable stack")]
    ColumnTallerThanStack { column_vars: usize, n_stack: usize },
    #[error("no subgroup of order 2^{l_skip} exists (field two-adicity is {two_adicity})")]
    SkipDomainUnavailable { l_skip: usize, two_adicity: usize },
    #[error("{coefficients} coefficients do not fit a domain of {domain} points")]
    CodewordTooShort { coefficients: usize, domain: usize },
    #[error("query index {index} is outside the {bound} committed leaves")]
    QueryOutOfRange { index: usize, bound: usize },
    #[error("expected {expected} query openings, got {got}")]
    QueryCountMismatch { expected: usize, got: usize },
    #[error("query {query}: the Merkle opening does not match the commitment")]
    OpeningRejected { query: usize },
    #[error("query {query}: the folded block does not match the committed successor")]
    FoldInconsistent { query: usize },
    #[error("the folded codeword and the sumcheck disagree on the evaluation")]
    EvaluationMismatch,
    #[error("eq(z, alpha) vanished, leaving the evaluation unconstrained")]
    DegenerateEvaluationPoint,
    #[error("the out-of-domain point landed inside the evaluation domain")]
    OodPointInDomain,
    #[error("no proof-of-work nonce found for {bits} bits")]
    GrindingFailed { bits: u8 },
    #[error("a proof-of-work nonce does not carry the required {bits} bits")]
    GrindingRejected { bits: u8 },
    #[error("the buses do not balance across the proof")]
    BusImbalance,
    #[error("the statements rebuilt from the factor values are not what the batch demands")]
    BatchMismatch,
    #[error("a shifted read is not the shift of the column it claims to shift")]
    ShiftedReadMismatch,
    #[error("column {column}: its claimed value does not match the commitment")]
    ColumnOpeningRejected { column: usize },
    /// A device path that had already drawn a challenge cannot be retried on
    /// the host: the transcript has moved.
    #[error("the device failed mid-{stage}, after the transcript had moved")]
    DeviceFailed { stage: &'static str },
}

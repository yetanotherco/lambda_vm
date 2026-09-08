//! Multilinear machinery for a sumcheck-based proof system: extensions over the
//! Boolean hypercube, sumcheck, zerocheck, LogUp-GKR, stacking and WHIR.
//!
//! Not wired into the prover. The codeword domain is a two-adic subgroup of the
//! base field, so extension-valued columns need the field tower generalized.

pub mod constraint_argument;
pub mod eq;
pub mod gkr;
pub mod mle;
pub mod poly;
pub mod selector;
pub mod stacking;
pub mod sumcheck;
pub mod uni_skip;
pub mod virtual_poly;
pub mod whir;
pub mod whir_commit;
pub mod whir_eval;
pub mod whir_round;
pub mod zerocheck;

use thiserror::Error;

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
    #[error("round {round}: claimed sum {claimed} does not match g(0) + g(1) = {got}")]
    RoundSumMismatch {
        round: usize,
        claimed: String,
        got: String,
    },
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
    #[error("column {column}: its claimed value does not match the commitment")]
    ColumnOpeningRejected { column: usize },
    #[error("the constraint rebuilt from the column values is not what the zerocheck demands")]
    ConstraintMismatch,
}

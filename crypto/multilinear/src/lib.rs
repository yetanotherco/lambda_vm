//! Multilinear machinery for a sumcheck-based proof system.
//!
//! Our STARK proves an AIR constraint `C` by *dividing*: the trace columns are
//! univariate polynomials over a multiplicative subgroup, and `C` vanishing on
//! every row is shown via the quotient `C / Z`. This crate is the other
//! formulation: a trace of `2^n` rows is a function on the Boolean hypercube
//! `{0,1}^n`, and `C` vanishing on every row is shown by [`zerocheck`], which
//! reduces to a [`sumcheck`] — no division, no zerofier, no blown-up domain.
//!
//! Layout of the pieces:
//!
//! - [`mle`]: a multilinear extension, stored as its `2^n` hypercube evaluations.
//! - [`eq`]: the equality polynomial `eq(r, x)`, the kernel every zerocheck needs.
//! - [`poly`]: what sumcheck needs from a polynomial — factors plus a combine
//!   rule. Keeps an AIR's constraint DAG out of expanded form.
//! - [`virtual_poly`]: a sum of products of MLEs — one implementation of that.
//! - [`selector`]: which steps a constraint applies to, for the transition
//!   constraints that must skip the wrap-around step.
//! - [`gkr`]: LogUp as a tree of fractions, replacing the committed
//!   running-sum columns.
//! - [`stacking`]: packing tables of different heights into shared cubes, so
//!   the commitment count stops tracking the table count.
//! - [`sumcheck`]: the interactive proof that `Σ_x f(x)` equals a claimed value.
//! - [`zerocheck`]: `f` vanishes on the whole hypercube, via `sumcheck`.
//!
//! Everything is generic over the field. In this VM the intended instantiation
//! is Goldilocks for trace values and its degree-3 extension for challenges.
//!
//! - [`uni_skip`]: the prism `D × {0,1}^n`, for running the first rounds over a
//!   subgroup instead of the cube.
//! - [`whir`]: encoding a multilinear as a Reed–Solomon codeword and folding it
//!   in step with the sumcheck.
//! - [`whir_commit`]: committing that codeword and opening the blocks a query
//!   asks for.
//! - [`whir_round`]: one round assembled — sample queries, open, fold locally,
//!   and reject a successor that is not the fold.
//! - [`whir_eval`]: the evaluation argument end to end, which is what settles
//!   the residual claims the other arguments hand back.
//! - [`constraint_argument`]: zerocheck plus `whir_eval`, so a **committed**
//!   trace can be shown to satisfy a constraint.
//!
//! ## Not yet here
//!
//! - The base-field optimization: evaluations start in the base field and only
//!   become extension elements after the first fold. The API does not preclude
//!   it, but every polynomial currently lives in one field.
//! - Wiring the univariate skip into `sumcheck`. The prism geometry it needs is
//!   in [`uni_skip`]; the rounds still run over the cube.

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

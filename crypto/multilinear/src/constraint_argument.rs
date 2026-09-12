//! The argument for one table: statements over a **committed** trace, all in a
//! single sumcheck, settled against one opening per column.
//!
//! A statement is `Σ_x rule(x) = claimed`. A constraint is one — with `eq(r, ·)`
//! as its weight, the sum being zero is the zerocheck. A bus's LogUp-GKR
//! input-layer claim is another. [`prove_statements`] takes them together, so
//! the factors are folded once rather than once per argument.
//!
//! Only the trace's *columns* are committed, and only what cannot be recomputed
//! travels in the proof:
//!
//! - a read of the next step is a **view** of a column — materialized for the
//!   sumcheck to fold, never committed, and its claimed value bound back to the
//!   column through [`claim_reduce`];
//! - a **public** factor, like a selector or a weight table, is committed by
//!   nobody and claimed by nobody: the verifier evaluates it in closed form.
//!
//! And the columns that are committed are **stacked**: every one of them sits
//! in a subcube of a shared polynomial, so the whole trace is a handful of
//! commitments and a handful of openings — not one per column — however many
//! statements, shifted reads and public tables are in play.
//!
//! The columns live in the **base field**, like the trace they are, so the
//! stacked polynomial and its codeword do too. The sumcheck's factors have to
//! share a field, so the views are lifted for it; the codeword is not, and that
//! is the allocation that matters.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::AsBytes,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    Error,
    batch::{self, Rule},
    claim_reduce::{self, FactorSource, ReduceProof},
    eq::{eq_eval, eq_mle},
    mle::Mle,
    stacked_eval::{self, StackedCommitment, StackedProof},
    stacking::StackedLayout,
    sumcheck::SumcheckProof,
    whir_chain::ChainConfig,
    whir_commit::Commitment,
};

/// The stack width that fits every column in a single polynomial.
///
/// `num_vars + log_blowup + log2(columns)` must stay inside the base field's
/// two-adicity, or the evaluation domain does not exist — [`Domain::new`] says
/// so rather than producing something wrong.
///
/// [`Domain::new`]: crate::whir::Domain::new
pub fn one_stack(num_vars: usize, num_columns: usize) -> usize {
    num_vars + num_columns.next_power_of_two().trailing_zeros() as usize
}

/// Where a sumcheck factor's table comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactorKind {
    /// A committed column, read at a frame-step offset.
    Committed(FactorSource),
    /// A table both sides can compute, so it is neither committed nor claimed.
    Public,
}

impl FactorKind {
    pub const fn direct(column: usize) -> Self {
        Self::Committed(FactorSource::direct(column))
    }

    pub const fn shifted(column: usize, offset: usize) -> Self {
        Self::Committed(FactorSource::shifted(column, offset))
    }

    /// The column this factor reads, or `None` if it is public.
    pub fn source(&self) -> Option<FactorSource> {
        match self {
            Self::Committed(s) => Some(*s),
            Self::Public => None,
        }
    }
}

/// The committed factors' sources, in factor order — what [`claim_reduce`]
/// binds. Public factors drop out.
fn sources_of(kinds: &[FactorKind]) -> Vec<FactorSource> {
    kinds.iter().filter_map(FactorKind::source).collect()
}

/// Weaves the committed factors' values back together with the public ones, in
/// factor order — the order the rules index.
///
/// `public` covers the trace's public factors first; anything left over is a
/// weight table a statement appended, so it lands at the end.
fn weave<E: IsField>(
    kinds: &[FactorKind],
    committed: &[FieldElement<E>],
    public: &[FieldElement<E>],
) -> Result<Vec<FieldElement<E>>, Error> {
    let want_committed = kinds.iter().filter(|k| k.source().is_some()).count();
    if committed.len() != want_committed {
        return Err(Error::QueryCountMismatch {
            expected: want_committed,
            got: committed.len(),
        });
    }
    let want_public = kinds.len() - want_committed;
    if public.len() < want_public {
        return Err(Error::QueryCountMismatch {
            expected: want_public,
            got: public.len(),
        });
    }

    let (public, weights) = public.split_at(want_public);
    let mut committed = committed.iter();
    let mut public = public.iter();
    let mut values: Vec<FieldElement<E>> = kinds
        .iter()
        .map(|kind| match kind {
            FactorKind::Committed(_) => committed.next(),
            FactorKind::Public => public.next(),
        })
        .map(|v| v.expect("counts were checked").clone())
        .collect();
    values.extend(weights.iter().cloned());
    Ok(values)
}

/// A committed trace, ready to be argued about.
///
/// Holds the columns that were committed and, for every trace-level factor,
/// where its table comes from. Weight tables are not in here: they depend on
/// challenges drawn after the commitments, so the statements bring their own.
pub struct CommittedTrace<F: IsFFTField + IsPrimeField + IsSubFieldOf<E>, E: IsField>
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    data: TraceData<F, E>,
    stacked: StackedCommitment<F>,
}

/// A table's factors: its columns, the public tables, and what each factor
/// reads.
///
/// Says nothing about where the columns are committed, which is the point:
/// several tables can share one commitment, and then no single table owns it.
#[derive(Debug)]
pub struct TraceData<F: IsField, E: IsField> {
    columns: Vec<Mle<F>>,
    /// The public factors' tables, in the order they appear in `kinds`. Held
    /// because they are few — selectors and the like — while the shifted views
    /// are rebuilt on demand rather than kept for the whole proof.
    public: Vec<Mle<E>>,
    kinds: Vec<FactorKind>,
    /// The factors on a device, put there by whoever needed them first. The
    /// input layer reads them and the sumcheck folds them, and they are the
    /// biggest thing a table's argument holds — uploading them twice would
    /// cost more than either use.
    device: std::sync::Mutex<Option<std::sync::Arc<crate::gpu::DeviceFactors>>>,
}

impl<F: IsField + 'static, E: IsField + 'static> TraceData<F, E> {
    /// Checks the shapes agree and that `kinds` asks for exactly the public
    /// tables given.
    pub fn new(
        columns: Vec<Mle<F>>,
        kinds: Vec<FactorKind>,
        public: Vec<Mle<E>>,
    ) -> Result<Self, Error> {
        let num_vars = columns.first().map(Mle::num_vars).unwrap_or(0);
        for got in columns
            .iter()
            .map(Mle::num_vars)
            .chain(public.iter().map(Mle::num_vars))
        {
            if got != num_vars {
                return Err(Error::VariableCountMismatch {
                    expected: num_vars,
                    got,
                });
            }
        }
        if columns.is_empty() {
            return Err(Error::EmptyPolynomial);
        }
        let wanted = kinds.iter().filter(|k| k.source().is_none()).count();
        if public.len() != wanted {
            return Err(Error::QueryCountMismatch {
                expected: wanted,
                got: public.len(),
            });
        }
        Ok(Self {
            columns,
            public,
            device: std::sync::Mutex::new(None),
            kinds,
        })
    }

    pub fn columns(&self) -> &[Mle<F>] {
        &self.columns
    }

    pub fn kinds(&self) -> &[FactorKind] {
        &self.kinds
    }

    pub fn num_vars(&self) -> usize {
        self.columns.first().map(Mle::num_vars).unwrap_or(0)
    }

    /// The factors the sumcheck runs over: each committed column shifted by
    /// its offset, and the public tables as they are.
    ///
    /// Materialized on each call rather than stored. The sumcheck has to own
    /// and fold them anyway, so a second copy kept for the whole proof would be
    /// one more resident copy of the trace and nothing else.
    /// The factors on a device, if they are there.
    ///
    /// Whoever needs them first calls [`reside`](Self::reside); the handle is
    /// shared from then on. Folding them — which the sumcheck does — spends
    /// them, and nothing reads them after that.
    pub fn device_factors(&self) -> Option<std::sync::Arc<crate::gpu::DeviceFactors>> {
        self.device.lock().ok()?.clone()
    }

    /// Puts the factors on a device and keeps the handle, or leaves it empty
    /// when the device declines.
    ///
    /// They are built there, out of the columns: a factor is a column read at
    /// a frame-step offset and lifted into the extension, and both of those are
    /// cheaper on the side that is going to fold them. This path never goes
    /// through [`factors`](Self::factors), so on a device the host copy is
    /// never made.
    /// Lets go of the factors on the device.
    ///
    /// The sumcheck folds them where they lie, which spends them, and nothing
    /// reads them afterwards — but the table outlives its own argument, so
    /// without this every table's factors stay on the device until the whole
    /// proof is done. On a real trace that is gigabytes held for nothing.
    pub fn release_device(&self) {
        if let Ok(mut slot) = self.device.lock() {
            *slot = None;
        }
    }

    pub fn reside_from_columns(&self) -> Option<std::sync::Arc<crate::gpu::DeviceFactors>> {
        let mut slot = self.device.lock().ok()?;
        if slot.is_none() {
            *slot =
                crate::gpu::upload_factors_from_columns(&self.columns, &self.kinds, &self.public)
                    .map(std::sync::Arc::new);
        }
        slot.clone()
    }

    pub fn factors(&self) -> Result<Vec<Mle<E>>, Error>
    where
        F: IsSubFieldOf<E>,
        FieldElement<E>: Send + Sync,
    {
        // Which public table each public factor takes, resolved up front so the
        // factors can be built out of order.
        let mut public_at = Vec::with_capacity(self.kinds.len());
        let mut seen = 0usize;
        for kind in &self.kinds {
            public_at.push(seen);
            if kind.source().is_none() {
                seen += 1;
            }
        }

        // One lift of the whole trace into the extension, which is the biggest
        // allocation the argument makes after the codeword.
        let build = |(kind, at): (&FactorKind, &usize)| -> Result<Mle<E>, Error> {
            match kind {
                // The sumcheck's factors share a field, so a base view is
                // lifted for it. The codeword is what stays base.
                //
                // The shift rides the lift rather than preceding it: a view
                // materialized first is a second copy of the column, and the
                // rotation is two runs of consecutive cells, not a modulus per
                // cell.
                FactorKind::Committed(s) => {
                    let column = self.columns.get(s.column).ok_or(Error::UnknownPolynomial {
                        index: s.column,
                        len: self.columns.len(),
                    })?;
                    let shift = s.offset % column.len();
                    let (wrapped, rest) = column.evals().split_at(shift);
                    Mle::new(
                        rest.iter()
                            .chain(wrapped)
                            .map(|v| v.clone().to_extension::<E>())
                            .collect(),
                    )
                }
                FactorKind::Public => self.public.get(*at).cloned().ok_or(Error::EmptyPolynomial),
            }
        };
        #[cfg(feature = "parallel")]
        return self
            .kinds
            .par_iter()
            .zip(public_at.par_iter())
            .map(build)
            .collect();
        #[cfg(not(feature = "parallel"))]
        return self.kinds.iter().zip(public_at.iter()).map(build).collect();
    }
}

impl<
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
> CommittedTrace<F, E>
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    /// Commits every column, each read unshifted as one factor.
    pub fn commit(columns: Vec<Mle<F>>, config: &ChainConfig) -> Result<Self, Error> {
        let kinds = (0..columns.len()).map(FactorKind::direct).collect();
        let n_stack = one_stack(
            columns.first().map(|c| c.num_vars()).unwrap_or(0),
            columns.len(),
        );
        Self::commit_views(columns, kinds, Vec::new(), n_stack, config)
    }

    /// Stacks and commits every column, and assembles the factors `kinds`
    /// describes.
    ///
    /// `public` supplies the public factors' tables, in the order they appear
    /// in `kinds`. Everything must agree on height. A shifted factor is
    /// materialized here for the sumcheck to fold, but never committed.
    ///
    /// `n_stack` is the width of a stacked polynomial; [`one_stack`] is the
    /// value that fits the whole trace in one.
    pub fn commit_views(
        columns: Vec<Mle<F>>,
        kinds: Vec<FactorKind>,
        public: Vec<Mle<E>>,
        n_stack: usize,
        config: &ChainConfig,
    ) -> Result<Self, Error> {
        let num_vars = columns.first().map(|c| c.num_vars()).unwrap_or(0);
        for got in columns
            .iter()
            .map(Mle::num_vars)
            .chain(public.iter().map(Mle::num_vars))
        {
            if got != num_vars {
                return Err(Error::VariableCountMismatch {
                    expected: num_vars,
                    got,
                });
            }
        }
        if columns.is_empty() {
            return Err(Error::EmptyPolynomial);
        }
        let layout = StackedLayout::build(&vec![num_vars; columns.len()], n_stack)?;
        Self::commit_stacked(columns, kinds, public, layout, config)
    }

    /// The same, against a layout the caller already holds.
    ///
    /// A verifier derives the layout from the column heights alone, so a prover
    /// that built one to describe its statement passes that very one here
    /// instead of building a second that could drift from it.
    pub fn commit_stacked(
        columns: Vec<Mle<F>>,
        kinds: Vec<FactorKind>,
        public: Vec<Mle<E>>,
        layout: StackedLayout,
        config: &ChainConfig,
    ) -> Result<Self, Error> {
        let stacked =
            StackedCommitment::<F>::commit(layout, &crate::stacking::borrow(&columns), config)?;
        Ok(Self {
            data: TraceData::new(columns, kinds, public)?,
            stacked,
        })
    }

    /// The factors, without the commitment.
    pub fn data(&self) -> &TraceData<F, E> {
        &self.data
    }

    /// One root per stacked polynomial, not per column.
    pub fn roots(&self) -> Vec<Commitment> {
        self.stacked.roots()
    }

    pub fn domain(&self) -> &crate::whir::Domain<F> {
        self.stacked.domain()
    }

    pub fn layout(&self) -> &StackedLayout {
        self.stacked.layout()
    }

    pub fn num_vars(&self) -> usize {
        self.data.columns.first().map(|c| c.num_vars()).unwrap_or(0)
    }

    pub fn kinds(&self) -> &[FactorKind] {
        &self.data.kinds
    }

    /// The factors the sumcheck runs over: each committed column shifted by its
    /// offset, and the public tables as they are.
    ///
    /// Materialized on each call rather than stored. The sumcheck has to own
    /// and fold them anyway, so a second copy kept for the whole proof would be
    /// one more resident copy of the trace and nothing else.
    pub fn factors(&self) -> Result<Vec<Mle<E>>, Error> {
        self.data.factors()
    }

    pub fn columns(&self) -> &[Mle<F>] {
        &self.data.columns
    }
}

/// The public part of the statement: what was committed, over what domain, and
/// where each trace-level factor comes from.
#[derive(Clone, Copy, Debug)]
pub struct TraceClaim<'a, F: IsFFTField + IsPrimeField> {
    /// One per stacked polynomial.
    pub roots: &'a [Commitment],
    pub kinds: &'a [FactorKind],
    /// Where each column sits in the stack. Public, derived from the heights.
    pub layout: &'a StackedLayout,
    pub domain: &'a crate::whir::Domain<F>,
    pub num_vars: usize,
}

/// A proof that the committed columns satisfy the statements.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct ConstraintProof<F: IsField, E: IsField> {
    pub core: ConstraintCore<E>,
    /// The columns' values at the reduced point, against the stack.
    pub columns: StackedProof<F, E>,
}

/// A statement bundle's argument **up to** the columns' claims.
///
/// Split out because the opening need not belong to one bundle: when several
/// tables share a commitment, each produces a core of its own and the stack is
/// opened once for all of them.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct ConstraintCore<E: IsField> {
    /// The one sumcheck every statement shares.
    pub sumcheck: SumcheckProof<E>,
    /// Each **committed** factor's value at the sumcheck point. The public
    /// factors are absent: the verifier computes those.
    pub factor_values: Vec<FieldElement<E>>,
    /// Binds every committed factor value to the column it reads.
    pub reduce: ReduceProof<E>,
}

/// Proves every statement in one sumcheck and settles it against the
/// commitments.
///
/// The factor list is the trace's factors followed by `weights`, and that is
/// what `rules` index. `claims[i]` is what `Σ_x rules[i](x)` must come to.
///
/// The caller must have absorbed the commitment roots and drawn whatever
/// challenges its statements need, identically on both sides.
pub fn prove_statements<F, E, T>(
    trace: &CommittedTrace<F, E>,
    weights: Vec<Mle<E>>,
    rules: Vec<Rule<'_, E>>,
    claims: &[FieldElement<E>],
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<ConstraintProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
{
    let (core, reduced_point) =
        prove_core::<F, E, T>(&trace.data, weights, rules, claims, transcript)?;

    // Every column's value at one shared point, so the whole trace is settled
    // against the stack in one go.
    let columns = stacked_eval::prove::<F, E, T>(
        &trace.stacked,
        &stacked_eval::Claimed::Shared(&reduced_point),
        &core.reduce.column_values,
        config,
        transcript,
    )?;

    Ok(ConstraintProof { core, columns })
}

/// The argument up to the columns' claims, leaving the opening to the caller.
///
/// Returns the point the columns are claimed at, which is what a caller
/// settling several tables against one commitment collects before opening it
/// once.
pub fn prove_core<F, E, T>(
    trace: &TraceData<F, E>,
    weights: Vec<Mle<E>>,
    rules: Vec<Rule<'_, E>>,
    claims: &[FieldElement<E>],
    transcript: &mut T,
) -> Result<(ConstraintCore<E>, Vec<FieldElement<E>>), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
{
    // The trace's factors are not built here when a device holds them: that
    // build is the whole trace in the extension, and nothing on that path
    // reads it. The closure is what makes them if the device turns the rounds
    // down.
    let resident = trace.device_factors();
    let (sumcheck, point, bound) = batch::prove_resident(
        weights,
        resident,
        || trace.factors(),
        rules,
        claims,
        transcript,
    )?;

    // The sumcheck leaves a claim about the factors at its point. Settle it in
    // two steps: reduce every committed factor's value there to a claim about
    // the column it reads, then open each column once. The public factors need
    // neither step.
    let factor_values = if bound.len() >= trace.kinds.len() {
        // The rounds folded every factor to exactly this, so reading it back is
        // the whole of it. Slot order is `kinds` order, and the weight tables
        // the batch added sit past the end.
        trace
            .kinds
            .iter()
            .zip(&bound)
            .filter(|(kind, _)| kind.source().is_some())
            .map(|(_, value)| value.clone())
            .collect()
    } else {
        // The host ran the rounds and let each factor go as it folded, so the
        // values have to be built again: a shifted view is rebuilt, read and
        // dropped; an unshifted one is read where it lies. Either way this
        // costs one column rather than a second copy of the whole factor list.
        trace
            .kinds
            .iter()
            .filter_map(FactorKind::source)
            .map(|source| claim_reduce::evaluate_source(&trace.columns, &source, &point))
            .collect::<Result<Vec<_>, _>>()?
    };

    let (reduce, reduced_point) = claim_reduce::prove::<F, E, T>(
        &trace.columns,
        &sources_of(&trace.kinds),
        &factor_values,
        &point,
        transcript,
    )?;

    Ok((
        ConstraintCore {
            sumcheck,
            factor_values,
            reduce,
        },
        reduced_point,
    ))
}

/// Verifies the statements against the commitments.
///
/// `rules` and `claims` must be the ones the prover used; they are the
/// statement, not part of the proof. `public_values` is handed the sumcheck
/// point and returns the public factors' values there: the trace's first, in
/// `kinds` order, then one per weight table.
///
/// Returns each **column's** value at the reduced point. A caller that knows
/// what some column has to be — a preprocessed table, say — checks it there:
/// the commitment says the prover is consistent with what it committed, not
/// that what it committed is right.
#[must_use = "the column values are the only place a known column can be checked"]
pub fn verify_statements<F, E, T, P>(
    proof: &ConstraintProof<F, E>,
    claim_shape: TraceClaim<'_, F>,
    rules: &[Rule<'_, E>],
    claims: &[FieldElement<E>],
    public_values: P,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<claim_reduce::ReducedClaim<E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    P: FnOnce(&[FieldElement<E>]) -> Result<Vec<FieldElement<E>>, Error>,
{
    let reduced = verify_core(
        &proof.core,
        claim_shape.kinds,
        claim_shape.layout.placements().len(),
        claim_shape.num_vars,
        rules,
        claims,
        public_values,
        transcript,
    )?;

    stacked_eval::verify::<F, E, T>(
        &proof.columns,
        claim_shape.layout,
        claim_shape.roots,
        &stacked_eval::Claimed::Shared(&reduced.point),
        &reduced.column_values,
        claim_shape.domain,
        config,
        transcript,
    )?;

    Ok(reduced)
}

/// The verifying half of [`prove_core`]: everything up to the columns' claims,
/// leaving the opening to the caller.
///
/// `num_columns` is the table's own column count, which is what
/// [`claim_reduce`] reduces to — not the width of whatever stack those columns
/// end up sharing.
#[allow(clippy::too_many_arguments)]
#[must_use = "the column values are the only place a known column can be checked"]
pub fn verify_core<E, T, P>(
    core: &ConstraintCore<E>,
    kinds: &[FactorKind],
    num_columns: usize,
    num_vars: usize,
    rules: &[Rule<'_, E>],
    claims: &[FieldElement<E>],
    public_values: P,
    transcript: &mut T,
) -> Result<claim_reduce::ReducedClaim<E>, Error>
where
    E: IsField + 'static,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    P: FnOnce(&[FieldElement<E>]) -> Result<Vec<FieldElement<E>>, Error>,
{
    // The rules rebuild the statements from the factor values: the committed
    // ones out of the proof, the public ones recomputed here.
    let point = batch::verify(
        &core.sumcheck,
        rules,
        claims,
        |at: &[FieldElement<E>]| weave(kinds, &core.factor_values, &public_values(at)?),
        num_vars,
        transcript,
    )?;

    claim_reduce::verify(
        &core.reduce,
        &sources_of(kinds),
        &core.factor_values,
        &point,
        num_columns,
        transcript,
    )
}

/// The single-constraint case: `Σ_x eq(r,x)·C(x) = 0`.
///
/// Absorbs the roots, draws `r`, and adds `eq(r, ·)` as one more public factor
/// — so `combine` sees exactly the trace's factors, and the weight costs one
/// degree.
pub fn prove<F, E, T, C>(
    trace: &CommittedTrace<F, E>,
    combine: C,
    degree: usize,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<ConstraintProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    C: Fn(&[FieldElement<E>]) -> FieldElement<E> + Sync,
{
    for root in trace.roots() {
        transcript.append_bytes(&root);
    }
    let r: Vec<FieldElement<E>> = (0..trace.num_vars())
        .map(|_| transcript.sample_field_element())
        .collect();

    let weight = trace.kinds().len();
    let rule = Rule::new(degree + 1, move |v: &[FieldElement<E>]| {
        &v[weight] * combine(&v[..weight])
    });
    prove_statements(
        trace,
        vec![eq_mle(&r)?],
        vec![rule],
        &[FieldElement::zero()],
        config,
        transcript,
    )
}

/// Verifies the single-constraint case. See [`prove`].
pub fn verify<F, E, T, C, P>(
    proof: &ConstraintProof<F, E>,
    claim_shape: TraceClaim<'_, F>,
    combine: C,
    public_values: P,
    degree: usize,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    C: Fn(&[FieldElement<E>]) -> FieldElement<E> + Sync,
    P: FnOnce(&[FieldElement<E>]) -> Result<Vec<FieldElement<E>>, Error>,
{
    for root in claim_shape.roots {
        transcript.append_bytes(root);
    }
    let r: Vec<FieldElement<E>> = (0..claim_shape.num_vars)
        .map(|_| transcript.sample_field_element())
        .collect();

    let weight = claim_shape.kinds.len();
    let rule = Rule::new(degree + 1, move |v: &[FieldElement<E>]| {
        &v[weight] * combine(&v[..weight])
    });
    verify_statements(
        proof,
        claim_shape,
        &[rule],
        &[FieldElement::zero()],
        |at: &[FieldElement<E>]| {
            let mut values = public_values(at)?;
            values.push(eq_eval(&r, at)?);
            Ok(values)
        },
        config,
        transcript,
    )?;
    Ok(())
}

/// No public factors, so nothing for the verifier to recompute.
pub fn no_public_factors<E: IsField>(
    _point: &[FieldElement<E>],
) -> Result<Vec<FieldElement<E>>, Error> {
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::{
        gkr::{self, FractionLayer, FractionTree},
        selector::Selector,
        whir_chain::GrindBits,
    };

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"constraint-argument-test")
    }

    fn config() -> ChainConfig {
        ChainConfig {
            log_blowup: 2,
            log_folding: 2,
            num_queries: 3,
            grind: GrindBits::default(),
        }
    }

    /// The constraint: `a·b − c = 0`. Degree 2, three columns — the shape a
    /// real AIR relation takes.
    fn constraint(v: &[FE]) -> FE {
        v[0] * v[1] - v[2]
    }

    fn mle(vals: &[u64]) -> Mle<F> {
        Mle::new(vals.iter().map(|x| FE::from(*x)).collect()).unwrap()
    }

    /// Columns that satisfy `a·b = c` on every row.
    fn satisfying(num_vars: usize) -> Vec<Mle<F>> {
        let size = 1usize << num_vars;
        let a: Vec<u64> = (0..size as u64).map(|i| i * 3 + 1).collect();
        let b: Vec<u64> = (0..size as u64).map(|i| i * 5 + 2).collect();
        let c: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        vec![mle(&a), mle(&b), mle(&c)]
    }

    fn run(columns: Vec<Mle<F>>) -> Result<(), Error> {
        let num_vars = columns[0].num_vars();
        let trace = CommittedTrace::<F, F>::commit(columns, &config()).unwrap();
        let proof = prove(&trace, constraint, 2, &config(), &mut transcript())?;

        verify(
            &proof,
            TraceClaim {
                roots: &trace.roots(),
                kinds: trace.kinds(),
                layout: trace.layout(),
                domain: trace.domain(),
                num_vars,
            },
            constraint,
            no_public_factors,
            2,
            &config(),
            &mut transcript(),
        )
    }

    #[test]
    fn a_satisfying_trace_verifies() {
        for num_vars in 2..=4usize {
            run(satisfying(num_vars)).unwrap_or_else(|e| panic!("num_vars={num_vars}: {e:?}"));
        }
    }

    /// The whole point: a trace that breaks the constraint in one row must be
    /// rejected, end to end, against its own commitments.
    #[test]
    fn a_trace_that_breaks_the_constraint_is_rejected() {
        let mut columns = satisfying(3);
        let mut c = columns[2].evals().to_vec();
        c[5] += FE::one();
        columns[2] = Mle::new(c).unwrap();

        assert!(run(columns).is_err());
    }

    #[test]
    fn a_forged_factor_value_is_rejected() {
        let columns = satisfying(3);
        let num_vars = 3;
        let trace = CommittedTrace::<F, F>::commit(columns, &config()).unwrap();
        let mut proof = prove(&trace, constraint, 2, &config(), &mut transcript()).unwrap();

        // Claim a different value for one factor, leaving everything else.
        proof.core.factor_values[0] += FE::one();

        let err = verify(
            &proof,
            TraceClaim {
                roots: &trace.roots(),
                kinds: trace.kinds(),
                layout: trace.layout(),
                domain: trace.domain(),
                num_vars,
            },
            constraint,
            no_public_factors,
            2,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert_eq!(err, Error::BatchMismatch);
    }

    #[test]
    fn verifying_a_different_constraint_is_rejected() {
        let columns = satisfying(3);
        let trace = CommittedTrace::<F, F>::commit(columns, &config()).unwrap();
        let proof = prove(&trace, constraint, 2, &config(), &mut transcript()).unwrap();

        // `a·b + c` instead of `a·b − c`: same degree, same columns.
        let err = verify(
            &proof,
            TraceClaim {
                roots: &trace.roots(),
                kinds: trace.kinds(),
                layout: trace.layout(),
                domain: trace.domain(),
                num_vars: 3,
            },
            |v: &[FE]| v[0] * v[1] + v[2],
            no_public_factors,
            2,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert_eq!(err, Error::BatchMismatch);
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        let columns = satisfying(3);
        let trace = CommittedTrace::<F, F>::commit(columns, &config()).unwrap();
        let proof = prove(&trace, constraint, 2, &config(), &mut transcript()).unwrap();

        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        assert!(
            verify(
                &proof,
                TraceClaim {
                    roots: &trace.roots(),
                    kinds: trace.kinds(),
                    layout: trace.layout(),
                    domain: trace.domain(),
                    num_vars: 3,
                },
                constraint,
                no_public_factors,
                2,
                &config(),
                &mut other,
            )
            .is_err()
        );
    }

    #[test]
    fn a_proof_missing_a_factor_is_rejected() {
        let columns = satisfying(3);
        let trace = CommittedTrace::<F, F>::commit(columns, &config()).unwrap();
        let mut proof = prove(&trace, constraint, 2, &config(), &mut transcript()).unwrap();
        proof.core.factor_values.pop();

        let err = verify(
            &proof,
            TraceClaim {
                roots: &trace.roots(),
                kinds: trace.kinds(),
                layout: trace.layout(),
                domain: trace.domain(),
                num_vars: 3,
            },
            constraint,
            no_public_factors,
            2,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::QueryCountMismatch { expected: 3, .. }));
    }

    #[test]
    fn columns_of_differing_heights_are_rejected() {
        let err = CommittedTrace::<F, F>::commit(vec![mle(&[1, 2]), mle(&[1, 2, 3, 4])], &config())
            .err()
            .unwrap();
        assert!(matches!(err, Error::VariableCountMismatch { .. }));
    }

    /// The field tower in use: a base-field evaluation domain with
    /// extension-valued columns, which is the shape a real trace has.
    #[test]
    fn the_argument_runs_over_a_field_tower() {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
        type ExtE = FieldElement<Ext>;

        let num_vars = 3;
        let size = 1usize << num_vars;
        // The trace is base-field; only the challenges and claims are not.
        let a: Vec<FE> = (0..size as u64).map(|i| FE::from(i * 3 + 1)).collect();
        let b: Vec<FE> = (0..size as u64).map(|i| FE::from(i * 5 + 2)).collect();
        let c: Vec<FE> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        let columns = vec![
            Mle::new(a).unwrap(),
            Mle::new(b).unwrap(),
            Mle::new(c).unwrap(),
        ];

        let cfg = ChainConfig {
            log_blowup: 2,
            log_folding: 2,
            num_queries: 3,
            grind: GrindBits::default(),
        };
        // Domain in Goldilocks, values in its degree-3 extension.
        let trace = CommittedTrace::<F, Ext>::commit(columns, &cfg).unwrap();
        let roots = trace.roots();
        let constraint = |v: &[ExtE]| v[0] * v[1] - v[2];

        let mut prover_t = DefaultTranscript::<Ext>::new(b"tower");
        let proof = prove::<F, Ext, _, _>(&trace, constraint, 2, &cfg, &mut prover_t).unwrap();

        let mut verifier_t = DefaultTranscript::<Ext>::new(b"tower");
        verify::<F, Ext, _, _, _>(
            &proof,
            TraceClaim {
                roots: &roots,
                kinds: trace.kinds(),
                layout: trace.layout(),
                domain: trace.domain(),
                num_vars,
            },
            constraint,
            no_public_factors,
            2,
            &cfg,
            &mut verifier_t,
        )
        .unwrap();
    }

    #[test]
    fn a_broken_row_over_the_tower_is_rejected() {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
        type ExtE = FieldElement<Ext>;

        let num_vars = 3;
        let size = 1usize << num_vars;
        let a: Vec<FE> = (0..size as u64).map(|i| FE::from(i * 3 + 1)).collect();
        let b: Vec<FE> = (0..size as u64).map(|i| FE::from(i * 5 + 2)).collect();
        let mut c: Vec<FE> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        c[4] += FE::one();
        let columns = vec![
            Mle::new(a).unwrap(),
            Mle::new(b).unwrap(),
            Mle::new(c).unwrap(),
        ];

        let cfg = ChainConfig {
            log_blowup: 2,
            log_folding: 2,
            num_queries: 3,
            grind: GrindBits::default(),
        };
        let trace = CommittedTrace::<F, Ext>::commit(columns, &cfg).unwrap();
        let roots = trace.roots();
        let constraint = |v: &[ExtE]| v[0] * v[1] - v[2];

        let mut prover_t = DefaultTranscript::<Ext>::new(b"tower");
        let proof = prove::<F, Ext, _, _>(&trace, constraint, 2, &cfg, &mut prover_t).unwrap();

        let mut verifier_t = DefaultTranscript::<Ext>::new(b"tower");
        assert!(
            verify::<F, Ext, _, _, _>(
                &proof,
                TraceClaim {
                    roots: &roots,
                    kinds: trace.kinds(),
                    layout: trace.layout(),
                    domain: trace.domain(),
                    num_vars,
                },
                constraint,
                no_public_factors,
                2,
                &cfg,
                &mut verifier_t,
            )
            .is_err()
        );
    }

    // ---------------------------------------------------------------
    // Shifted reads: a constraint that reads the next step, with no
    // commitment of its own for the shifted view.
    // ---------------------------------------------------------------

    /// `next(a) − a − b = 0`, with the next-step read as factor 2.
    fn transition(v: &[FE]) -> FE {
        v[2] - v[0] - v[1]
    }

    /// Two columns where `b` is `a`'s cyclic forward difference, so the
    /// transition holds on every row including the wrap.
    fn cyclic_columns(num_vars: usize) -> Vec<Mle<F>> {
        let size = 1usize << num_vars;
        let a: Vec<FE> = (0..size as u64).map(|i| FE::from(i * 7 + 3)).collect();
        let b: Vec<FE> = (0..size).map(|i| a[(i + 1) % size] - a[i]).collect();
        vec![Mle::new(a).unwrap(), Mle::new(b).unwrap()]
    }

    fn transition_kinds() -> Vec<FactorKind> {
        vec![
            FactorKind::direct(0),
            FactorKind::direct(1),
            FactorKind::shifted(0, 1),
        ]
    }

    fn run_transition(columns: Vec<Mle<F>>) -> Result<(), Error> {
        let num_vars = columns[0].num_vars();
        let n_stack = one_stack(num_vars, 2);
        let trace = CommittedTrace::<F, F>::commit_views(
            columns,
            transition_kinds(),
            Vec::new(),
            n_stack,
            &config(),
        )
        .unwrap();
        let proof = prove(&trace, transition, 1, &config(), &mut transcript())?;

        // Three factors, two columns, and one commitment holding both: the
        // shifted view rides on the column it shifts, and the columns ride in
        // one stack.
        assert_eq!(trace.kinds().len(), 3);
        assert_eq!(trace.roots().len(), 1);
        assert_eq!(proof.columns.polys.len(), 1);

        verify(
            &proof,
            TraceClaim {
                roots: &trace.roots(),
                kinds: trace.kinds(),
                layout: trace.layout(),
                domain: trace.domain(),
                num_vars,
            },
            transition,
            no_public_factors,
            1,
            &config(),
            &mut transcript(),
        )
    }

    #[test]
    fn a_constraint_reading_the_next_step_verifies() {
        for num_vars in 2..=4usize {
            run_transition(cyclic_columns(num_vars))
                .unwrap_or_else(|e| panic!("num_vars={num_vars}: {e:?}"));
        }
    }

    #[test]
    fn a_broken_transition_is_rejected() {
        let mut columns = cyclic_columns(3);
        let mut b = columns[1].evals().to_vec();
        b[5] += FE::one();
        columns[1] = Mle::new(b).unwrap();

        assert!(run_transition(columns).is_err());
    }

    /// The shifted factor's value is bound to the column through the reduction,
    /// so tampering with what the columns are worth at the reduced point is
    /// caught before any commitment is opened.
    #[test]
    fn a_forged_reduced_column_value_is_rejected() {
        let columns = cyclic_columns(3);
        let trace = CommittedTrace::<F, F>::commit_views(
            columns,
            transition_kinds(),
            Vec::new(),
            one_stack(3, 2),
            &config(),
        )
        .unwrap();
        let mut proof = prove(&trace, transition, 1, &config(), &mut transcript()).unwrap();
        proof.core.reduce.column_values[0] += FE::one();

        let err = verify(
            &proof,
            TraceClaim {
                roots: &trace.roots(),
                kinds: trace.kinds(),
                layout: trace.layout(),
                domain: trace.domain(),
                num_vars: 3,
            },
            transition,
            no_public_factors,
            1,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert_eq!(err, Error::ShiftedReadMismatch);
    }

    /// A verifier told the factor reads the column unshifted, when the prover
    /// shifted it. The sources are the statement, so the two must agree.
    #[test]
    fn disagreeing_about_a_factor_s_offset_is_rejected() {
        let columns = cyclic_columns(3);
        let trace = CommittedTrace::<F, F>::commit_views(
            columns,
            transition_kinds(),
            Vec::new(),
            one_stack(3, 2),
            &config(),
        )
        .unwrap();
        let proof = prove(&trace, transition, 1, &config(), &mut transcript()).unwrap();

        let mut kinds = transition_kinds();
        kinds[2] = FactorKind::direct(0);

        assert!(
            verify(
                &proof,
                TraceClaim {
                    roots: &trace.roots(),
                    kinds: &kinds,
                    layout: trace.layout(),
                    domain: trace.domain(),
                    num_vars: 3,
                },
                transition,
                no_public_factors,
                1,
                &config(),
                &mut transcript(),
            )
            .is_err()
        );
    }

    // ---------------------------------------------------------------
    // Public factors: a selector the verifier recomputes instead of
    // taking anyone's word for.
    // ---------------------------------------------------------------

    /// `s(x)·(a·b − c) = 0`, with the selector as factor 3.
    fn selected(v: &[FE]) -> FE {
        v[3] * (v[0] * v[1] - v[2])
    }

    fn selector() -> Selector {
        Selector::except_last(1)
    }

    /// Columns satisfying `a·b = c` everywhere **but the last row**, which is
    /// exactly what the selector exempts.
    fn satisfying_except_the_last(num_vars: usize) -> Vec<Mle<F>> {
        let mut columns = satisfying(num_vars);
        let mut c = columns[2].evals().to_vec();
        let last = c.len() - 1;
        c[last] += FE::one();
        columns[2] = Mle::new(c).unwrap();
        columns
    }

    fn selected_kinds() -> Vec<FactorKind> {
        vec![
            FactorKind::direct(0),
            FactorKind::direct(1),
            FactorKind::direct(2),
            FactorKind::Public,
        ]
    }

    fn run_selected(columns: Vec<Mle<F>>) -> Result<(), Error> {
        let num_vars = columns[0].num_vars();
        let trace = CommittedTrace::<F, F>::commit_views(
            columns,
            selected_kinds(),
            vec![selector().table(num_vars)?],
            one_stack(num_vars, 3),
            &config(),
        )?;
        let proof = prove(&trace, selected, 3, &config(), &mut transcript())?;

        // Four factors, one commitment for the three committed columns, and
        // only those three factors' values in the proof.
        assert_eq!(trace.kinds().len(), 4);
        assert_eq!(trace.roots().len(), 1);
        assert_eq!(proof.columns.polys.len(), 1);
        assert_eq!(proof.core.factor_values.len(), 3);

        verify(
            &proof,
            TraceClaim {
                roots: &trace.roots(),
                kinds: trace.kinds(),
                layout: trace.layout(),
                domain: trace.domain(),
                num_vars,
            },
            selected,
            |point: &[FE]| Ok(vec![selector().evaluate(point)?]),
            3,
            &config(),
            &mut transcript(),
        )
    }

    #[test]
    fn a_public_factor_needs_no_commitment() {
        for num_vars in 2..=4usize {
            run_selected(satisfying_except_the_last(num_vars))
                .unwrap_or_else(|e| panic!("num_vars={num_vars}: {e:?}"));
        }
    }

    #[test]
    fn the_selector_does_not_hide_a_violation_it_does_not_exempt() {
        let mut columns = satisfying_except_the_last(3);
        let mut c = columns[2].evals().to_vec();
        c[4] += FE::one();
        columns[2] = Mle::new(c).unwrap();

        assert!(run_selected(columns).is_err());
    }

    /// The verifier computes the public factor itself, so it decides what the
    /// table is: a prover proving against a different one is rejected.
    #[test]
    fn a_prover_using_another_public_table_is_rejected() {
        let columns = satisfying_except_the_last(3);
        // The prover folds a selector that exempts nothing, so its constraint
        // is violated on the last row while the verifier's masks it.
        let trace = CommittedTrace::<F, F>::commit_views(
            columns,
            selected_kinds(),
            vec![Selector::ALL.table(3).unwrap()],
            one_stack(3, 3),
            &config(),
        )
        .unwrap();
        let proof = prove(&trace, selected, 3, &config(), &mut transcript()).unwrap();

        assert!(
            verify(
                &proof,
                TraceClaim {
                    roots: &trace.roots(),
                    kinds: trace.kinds(),
                    layout: trace.layout(),
                    domain: trace.domain(),
                    num_vars: 3,
                },
                selected,
                |point: &[FE]| Ok(vec![selector().evaluate(point)?]),
                3,
                &config(),
                &mut transcript(),
            )
            .is_err()
        );
    }

    #[test]
    fn a_public_value_the_prover_did_not_fold_is_rejected() {
        let columns = satisfying_except_the_last(3);
        let trace = CommittedTrace::<F, F>::commit_views(
            columns,
            selected_kinds(),
            vec![selector().table(3).unwrap()],
            one_stack(3, 3),
            &config(),
        )
        .unwrap();
        let proof = prove(&trace, selected, 3, &config(), &mut transcript()).unwrap();

        let err = verify(
            &proof,
            TraceClaim {
                roots: &trace.roots(),
                kinds: trace.kinds(),
                layout: trace.layout(),
                domain: trace.domain(),
                num_vars: 3,
            },
            selected,
            |point: &[FE]| Ok(vec![selector().evaluate(point)? + FE::one()]),
            3,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert_eq!(err, Error::BatchMismatch);
    }

    #[test]
    fn a_public_table_of_the_wrong_height_is_rejected() {
        let err = CommittedTrace::<F, F>::commit_views(
            satisfying(3),
            selected_kinds(),
            vec![selector().table(2).unwrap()],
            one_stack(3, 3),
            &config(),
        )
        .err()
        .unwrap();
        assert!(matches!(err, Error::VariableCountMismatch { .. }));
    }

    #[test]
    fn a_public_table_per_public_factor_is_required() {
        let err = CommittedTrace::<F, F>::commit_views(
            satisfying(3),
            selected_kinds(),
            Vec::new(),
            one_stack(3, 3),
            &config(),
        )
        .err()
        .unwrap();
        assert_eq!(
            err,
            Error::QueryCountMismatch {
                expected: 1,
                got: 0
            }
        );
    }

    // ---------------------------------------------------------------
    // The whole composition: a transition constraint and a bus, over one
    // committed trace, in one sumcheck.
    // ---------------------------------------------------------------

    /// Factor layout of the fused argument.
    const V: usize = 0;
    const MULT: usize = 1;
    const ACC: usize = 2;
    const ACC_NEXT: usize = 3;
    const SEL: usize = 4;
    const EQ_R: usize = 5;
    const EQ_Z: usize = 6;

    fn fused_kinds() -> Vec<FactorKind> {
        vec![
            FactorKind::direct(0),
            FactorKind::direct(1),
            FactorKind::direct(2),
            FactorKind::shifted(2, 1),
            FactorKind::Public,
        ]
    }

    /// The three statements, all indexing the one factor list:
    ///
    /// - `Σ_x eq(r,x)·s(x)·(next(acc) − acc − v) = 0`, the transition;
    /// - `Σ_x eq(z,x)·mult = p(z)`, the bus's numerator claim;
    /// - `Σ_x eq(z,x)·(alpha − v) = q(z)`, its denominator claim.
    ///
    /// The denominator is never a column: it is read off the committed `v`.
    fn fused_rules(alpha: FE) -> Vec<Rule<'static, F>> {
        vec![
            Rule::new(3, |f: &[FE]| {
                f[EQ_R] * f[SEL] * (f[ACC_NEXT] - f[ACC] - f[V])
            }),
            Rule::new(2, |f: &[FE]| f[EQ_Z] * f[MULT]),
            Rule::new(2, move |f: &[FE]| f[EQ_Z] * (alpha - f[V])),
        ]
    }

    /// A trace with both shapes at once: a bus whose sends and receives cancel,
    /// and a running sum whose recurrence holds off the wrap.
    fn fused_columns(num_vars: usize) -> Vec<Mle<F>> {
        let size = 1usize << num_vars;
        let half = size / 2;
        let v: Vec<FE> = (0..size).map(|i| FE::from((i % half) as u64 + 1)).collect();
        let mult: Vec<FE> = (0..size)
            .map(|i| if i < half { FE::one() } else { -FE::one() })
            .collect();
        let mut acc = vec![FE::zero()];
        for i in 1..size {
            acc.push(acc[i - 1] + v[i - 1]);
        }
        vec![
            Mle::new(v).unwrap(),
            Mle::new(mult).unwrap(),
            Mle::new(acc).unwrap(),
        ]
    }

    /// Proves and verifies the fused argument, returning the sumcheck's round
    /// count so a caller can check it really was one pass.
    fn argue_table(columns: Vec<Mle<F>>, num_vars: usize) -> Result<usize, Error> {
        let selector = Selector::except_last(1);
        let trace = CommittedTrace::<F, F>::commit_views(
            columns,
            fused_kinds(),
            vec![selector.table(num_vars)?],
            one_stack(num_vars, 3),
            &config(),
        )?;
        let roots = trace.roots();

        // ---- prover ----
        let mut prover = transcript();
        for root in &roots {
            prover.append_bytes(root);
        }
        // The fingerprint challenge comes after the commitments.
        let alpha: FE = prover.sample_field_element();

        let v = trace.columns()[0].clone();
        let mult = trace.columns()[1].clone();
        let denominator = Mle::new(v.evals().iter().map(|x| alpha - x).collect())?;
        let tree = FractionTree::build(FractionLayer::new(mult, denominator)?)?;
        let (out_p, out_q) = tree.output();
        let gkr_out = gkr::prove(&tree, &mut prover)?;

        let r: Vec<FE> = (0..num_vars)
            .map(|_| prover.sample_field_element())
            .collect();
        let z = gkr_out.claim.point.clone();
        let claims = [FE::zero(), gkr_out.claim.p, gkr_out.claim.q];

        let proof = prove_statements(
            &trace,
            vec![eq_mle(&r)?, eq_mle(&z)?],
            fused_rules(alpha),
            &claims,
            &config(),
            &mut prover,
        )?;

        // ---- verifier ----
        let mut verifier = transcript();
        for root in &roots {
            verifier.append_bytes(root);
        }
        let alpha: FE = verifier.sample_field_element();
        // The bus balances exactly when the output numerator is zero; the
        // denominator is the only part of the output that has to be sent.
        let gkr_claim = gkr::verify(&gkr_out.proof, (FE::zero(), out_q), &mut verifier)?;
        let vr: Vec<FE> = (0..num_vars)
            .map(|_| verifier.sample_field_element())
            .collect();

        verify_statements(
            &proof,
            TraceClaim {
                roots: &roots,
                kinds: trace.kinds(),
                layout: trace.layout(),
                domain: trace.domain(),
                num_vars,
            },
            &fused_rules(alpha),
            &claims,
            // Trace publics first — the selector — then one per weight table.
            |at: &[FE]| {
                Ok(vec![
                    selector.evaluate(at)?,
                    eq_eval(&vr, at)?,
                    eq_eval(&gkr_claim.point, at)?,
                ])
            },
            &config(),
            &mut verifier,
        )?;

        assert_eq!(out_p, FE::zero(), "the bus was expected to balance");
        // Five factors over three columns, all in one commitment and settled by
        // one opening.
        assert_eq!(trace.kinds().len(), 5);
        assert_eq!(roots.len(), 1);
        assert_eq!(proof.columns.polys.len(), 1);
        assert_eq!(proof.core.factor_values.len(), 4);
        Ok(proof.core.sumcheck.rounds.len())
    }

    /// The composition this crate is being built for: a constraint that reads
    /// the next step and a bus that has to balance, over three committed
    /// columns, settled in **one** sumcheck with **one** opening per column.
    #[test]
    fn a_constraint_and_a_bus_argue_in_one_sumcheck() {
        for num_vars in 2..=4usize {
            let rounds = argue_table(fused_columns(num_vars), num_vars)
                .unwrap_or_else(|e| panic!("num_vars={num_vars}: {e:?}"));
            assert_eq!(rounds, num_vars, "one pass over the cube, not one each");
        }
    }

    #[test]
    fn a_broken_transition_breaks_the_fused_argument() {
        let mut columns = fused_columns(3);
        let mut acc = columns[2].evals().to_vec();
        acc[4] += FE::one();
        columns[2] = Mle::new(acc).unwrap();

        assert!(argue_table(columns, 3).is_err());
    }

    #[test]
    fn an_unbalanced_bus_breaks_the_fused_argument() {
        // The verifier requires a zero output numerator, so a bus that does not
        // cancel cannot get past the GKR.
        let mut columns = fused_columns(3);
        let mut mult = columns[1].evals().to_vec();
        mult[2] = FE::from(3);
        columns[1] = Mle::new(mult).unwrap();

        assert!(argue_table(columns, 3).is_err());
    }

    /// The transition and the bus both read `v` and it is folded once, and the
    /// three columns are one commitment and one opening.
    #[test]
    fn columns_shared_by_two_statements_are_opened_once() {
        argue_table(fused_columns(3), 3).unwrap();
    }
}

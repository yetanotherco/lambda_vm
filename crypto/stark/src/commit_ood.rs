//! Commitment-time out-of-domain answers for a multi-table proof.
//!
//! In a multi-table proof every table's main trace is committed, then the
//! LogUp challenges `(z, α)` shared by all tables are drawn, then each table
//! runs its own DEEP/FRI on its own fork. A committed main trace is close to
//! several codewords at once (a list, in the Johnson regime), and without
//! anything absorbed before `(z, α)` but roots, a prover could pick each
//! table's codeword after seeing `(z, α)`: the one bus equation over all
//! tables would then range over the product of the lists.
//!
//! The bound entry points (`multi_prove_bound`, `multi_verify_views_bound`)
//! close that: after the last root, one point `z0` is drawn outside every
//! table's domains ([`draw_point`]); every table answers `t_c(z0)` for each
//! main column `c` that is not precomputed ([`answered_columns`]), every
//! answer is absorbed in the main transcript ([`absorb_answers`]), and only
//! then are `(z, α)` drawn. Two distinct codewords of one table differ in
//! some answered column, so they disagree at `z0` except with probability
//! `< n/|E|`: from `(z, α)` on, each table has one named codeword.
//!
//! Precomputed columns answer nothing: their root is the verifier's
//! constant, so the committed word is the codeword itself. Aux columns and
//! composition parts answer nothing: they are committed on the table's own
//! fork, after `(z, α)`, where every later challenge is per table.
//!
//! Each table's DEEP polynomial carries the answers inside each answered
//! column's existing quotient at `z`: with `c_c` the column's current-row
//! coefficient, its term becomes
//!
//! ```text
//!   c_c·[(t_c(X) − t_c(z))/(X − z) − (t_c(X) − v_c)/(X − z0)]
//!     = c_c·(z − z0)·(t_c(X) − I_c(X))/((X − z)(X − z0)),
//! ```
//!
//! `I_c` the line through `(z, t_c(z))` and `(z0, v_c)`: one two-point
//! quotient per column, under the coefficient it already had. The batch keeps
//! its number of terms and its powers of `γ`; on the grid the answers' pole
//! takes `−c_c` ([`answer_coefficient`]), and a verifier folds the answered
//! columns' run of the current row once for both points.

use core::ops::Range;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};

use crate::traits::AIR;

/// One table's answers as its own rounds 2 to 4 take them.
#[derive(Clone, Copy, Debug)]
pub struct CommitOodTable<'a, E: IsField> {
    /// The shared point `z0`.
    pub point: &'a FieldElement<E>,
    /// The table's answers, one per [`answered_columns`] entry, in order.
    pub answers: &'a [FieldElement<E>],
}

/// The main columns of `air` that answer at `z0`: every main column after
/// the precomputed ones. A verifier constant, read from the AIR alone.
pub fn answered_columns<F, E, PI>(
    air: &dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>,
) -> Range<usize>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    let (main, _) = air.trace_layout();
    let precomputed = if air.is_preprocessed() {
        air.num_precomputed_columns()
    } else {
        0
    };
    precomputed.min(main)..main
}

/// The domain `z0` must avoid: the largest trace domain and the largest LDE
/// coset over the tables, `(trace_length, lde_length, coset_offset)`. Every
/// table's trace domain is a subgroup of the largest and every LDE coset (one
/// offset for all) a subset of the largest, so a point outside these two is
/// outside every table's. `None` when the tables do not share a coset offset
/// or there are none.
pub fn point_domain(
    tables: impl IntoIterator<Item = (usize, usize, u64)>,
) -> Option<(usize, usize, u64)> {
    let mut out: Option<(usize, usize, u64)> = None;
    for (trace_length, blowup, offset) in tables {
        let lde = trace_length.checked_mul(blowup)?;
        out = Some(match out {
            None => (trace_length, lde, offset),
            Some((t, l, o)) if o == offset => (t.max(trace_length), l.max(lde), o),
            Some(_) => return None,
        });
    }
    out
}

/// Draw `z0` from the main transcript, outside `domain` ([`point_domain`]).
pub fn draw_point<F, E>(
    transcript: &mut impl IsStarkTranscript<E, F>,
    domain: (usize, usize, u64),
) -> FieldElement<E>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let (trace_length, lde_length, offset) = domain;
    transcript.sample_z_ood_with_domain_params(
        trace_length,
        lde_length,
        &FieldElement::<F>::from(offset),
    )
}

/// Absorb every table's answers, table order then column order: the step
/// between [`draw_point`] and the shared LogUp draws.
pub fn absorb_answers<F, E>(
    transcript: &mut impl IsStarkTranscript<E, F>,
    answers: &[Vec<FieldElement<E>>],
) where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    for table in answers {
        for answer in table {
            transcript.append_field_element(answer);
        }
    }
}

/// An answered column's coefficient at the answers' pole `z0`: minus its
/// coefficient at `z` (the current row's), so the two quotients form the
/// column's one two-point quotient.
pub fn answer_coefficient<E: IsField>(current_row: &FieldElement<E>) -> FieldElement<E> {
    FieldElement::<E>::zero() - current_row
}

/// Whether `z0` is a point a table's DEEP polynomial already has a quotient
/// at: `z·gʳ` for each of the `rows` rows of its OOD grid (`g` the trace
/// domain's generator), or `z^parts`, where its composition parts open. At
/// such a point a false answer's quotient could cancel against that one
/// (at `z0 = z` the column's two-point quotient degenerates), so a verifier
/// refuses the proof instead.
pub fn meets_deep_point<F, E>(
    z0: &FieldElement<E>,
    z: &FieldElement<E>,
    g: &FieldElement<F>,
    rows: usize,
    parts: usize,
) -> bool
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    let mut row_point = z.clone();
    for _ in 0..rows {
        if &row_point == z0 {
            return true;
        }
        row_point = g * &row_point;
    }
    &z.pow(parts) == z0
}

/// `t_c(point)` for each column `c` in `columns` of a row-major trace
/// (`width` columns) over the trace domain `⟨ω⟩` of `rows` rows, by the
/// barycentric formula `t(z) = (zⁿ − 1)/n · Σ_k t_k·ωᵏ/(z − ωᵏ)`.
///
/// `None` when `point` lies in the domain (a zero denominator).
pub fn evaluate_trace_columns<F, E>(
    data: &[FieldElement<F>],
    width: usize,
    columns: Range<usize>,
    point: &FieldElement<E>,
) -> Option<Vec<FieldElement<E>>>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync,
{
    if columns.is_empty() {
        return Some(Vec::new());
    }
    if width == 0 || !data.len().is_multiple_of(width) {
        return None;
    }
    let rows = data.len() / width;
    if !rows.is_power_of_two() {
        return None;
    }
    let omega = F::get_primitive_root_of_unity(rows.trailing_zeros() as u64).ok()?;
    let mut powers = Vec::with_capacity(rows);
    let mut w = FieldElement::<F>::one();
    for _ in 0..rows {
        powers.push(w.clone());
        w = &w * &omega;
    }
    let mut weights: Vec<FieldElement<E>> = powers
        .iter()
        .map(|w| point - w.clone().to_extension::<E>())
        .collect();
    FieldElement::inplace_batch_inverse(&mut weights).ok()?;
    for (weight, w) in weights.iter_mut().zip(&powers) {
        *weight = w * &*weight;
    }
    let n_inv = FieldElement::<F>::from(rows as u64).inv().ok()?;
    let factor = (point.pow(rows) - FieldElement::<E>::one()) * n_inv.to_extension::<E>();
    Some(crate::par::par_map_collect(columns, |c| {
        let mut sum = FieldElement::<E>::zero();
        for (k, weight) in weights.iter().enumerate() {
            sum += &data[k * width + c] * weight;
        }
        &sum * &factor
    }))
}

/// Faults for the negative tests: what the prover sends and what both sides
/// fold, switched per thread around a prove and a verify. Never reachable
/// outside a test build.
#[cfg(any(test, feature = "test-utils"))]
pub mod faults {
    use std::cell::Cell;

    /// One fault set.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Faults {
        /// The prover adds one to answer `index` of table `table` before it is
        /// absorbed, and carries the false value into that table's DEEP.
        pub false_answer: Option<(usize, usize)>,
        /// Prover and verifier leave the `z0` quotients out of DEEP: the
        /// answers are still sent and absorbed, but nothing checks them.
        pub without_deep_term: bool,
        /// The verifier's check that `z0` avoids every DEEP point sees `z`
        /// in place of `z0`, as if the two draws had met.
        pub z0_meets_z: bool,
    }

    thread_local! {
        static FAULTS: Cell<Faults> = const {
            Cell::new(Faults {
                false_answer: None,
                without_deep_term: false,
                z0_meets_z: false,
            })
        };
    }

    /// The faults set on this thread.
    pub fn current() -> Faults {
        FAULTS.with(|f| f.get())
    }

    /// Run `body` with `faults` set on this thread, restoring the previous
    /// set afterwards.
    pub fn with<R>(faults: Faults, body: impl FnOnce() -> R) -> R {
        let previous = FAULTS.with(|f| f.replace(faults));
        let out = body();
        FAULTS.with(|f| f.set(previous));
        out
    }
}

/// The `z0` the verifier's DEEP-point check sees: `z0` itself, outside a
/// test fault.
pub(crate) fn z0_for_point_check<'a, E: IsField>(
    z0: &'a FieldElement<E>,
    z: &'a FieldElement<E>,
) -> &'a FieldElement<E> {
    #[cfg(any(test, feature = "test-utils"))]
    if faults::current().z0_meets_z {
        return z;
    }
    let _ = z;
    z0
}

/// Whether DEEP folds the `z0` quotients: always, outside a test fault.
pub(crate) fn deep_term_on() -> bool {
    #[cfg(any(test, feature = "test-utils"))]
    {
        !faults::current().without_deep_term
    }
    #[cfg(not(any(test, feature = "test-utils")))]
    {
        true
    }
}

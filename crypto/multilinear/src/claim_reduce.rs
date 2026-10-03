//! Binding a shifted read to the column it shifts.
//!
//! A constraint that reads the next step becomes a sumcheck factor that is the
//! cyclic shift of a committed column. The zerocheck does not care where a
//! factor came from, so a prover free to pick both the column and its "shift"
//! could satisfy a constraint with unrelated tables. The shift kernel closes
//! that: `f_shift_k(alpha) = Σ_y shift_k(alpha, y)·f(y)`, which is a claim about
//! the column itself.
//!
//! Every factor's claim is batched into one degree-2 sumcheck, so the whole
//! trace costs a single pass and leaves **one evaluation claim per committed
//! column, all at the same point** — one commitment opening each.
//!
//! The guarantee is conditional, and the caller supplies the other half: *if*
//! the column values at the reduced point are the committed columns' true
//! values, then the factor values were the true shifted values at `alpha`.
//! Pinning those is the commitment scheme's job.
//!
//! The columns are base-field, like the trace; the challenges and the claims
//! are not, so the batching reads them through mixed products.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    Error, challenge_powers,
    eq::{shift_eval, shift_mle},
    mle::Mle,
    poly::{Composed, SumcheckPolynomial},
    program::{Builder, Program},
    sumcheck::{self, SumcheckProof},
};

/// Which committed column a sumcheck factor reads, and how many steps ahead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FactorSource {
    pub column: usize,
    /// Frame-step offset; on the cube it is a cyclic shift by that many rows.
    pub offset: usize,
}

impl FactorSource {
    /// The column itself.
    pub const fn direct(column: usize) -> Self {
        Self { column, offset: 0 }
    }

    pub const fn shifted(column: usize, offset: usize) -> Self {
        Self { column, offset }
    }
}

/// The batched reduction.
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
pub struct ReduceProof<E: IsField> {
    pub sumcheck: SumcheckProof<E>,
    /// Every committed column's value at the reduced point.
    pub column_values: Vec<FieldElement<E>>,
}

/// One evaluation claim per committed column, all at the same point.
#[derive(Clone, Debug)]
pub struct ReducedClaim<E: IsField> {
    pub point: Vec<FieldElement<E>>,
    pub column_values: Vec<FieldElement<E>>,
}

/// Distinct offsets in ascending order: the grouping both sides must agree on.
fn offsets(sources: &[FactorSource]) -> Vec<usize> {
    let mut all: Vec<usize> = sources.iter().map(|s| s.offset).collect();
    all.sort_unstable();
    all.dedup();
    all
}

/// `Σ_o K_o(y)·B_o(y)`, the factors laid out in kernel/column pairs.
fn pair_products<E: IsField>(values: &[FieldElement<E>]) -> FieldElement<E> {
    values
        .chunks(2)
        .fold(FieldElement::zero(), |acc, pair| acc + &pair[0] * &pair[1])
}

/// The same sum as straight-line code, over `pairs` kernel/column pairs.
fn pair_products_program<E: IsField>(pairs: usize) -> Result<Program<E>, Error> {
    let mut b = Builder::<E>::new();
    let terms: Vec<u32> = (0..pairs)
        .map(|pair| {
            let kernel = b.var(2 * pair);
            let column = b.var(2 * pair + 1);
            b.mul(kernel, column)
        })
        .collect();
    let root = b.sum(&terms);
    b.finish(root)
}

fn check_shape<E: IsField>(
    sources: &[FactorSource],
    factor_values: &[FieldElement<E>],
    num_columns: usize,
) -> Result<(), Error> {
    if sources.is_empty() {
        return Err(Error::EmptyPolynomial);
    }
    if sources.len() != factor_values.len() {
        return Err(Error::VariableCountMismatch {
            expected: sources.len(),
            got: factor_values.len(),
        });
    }
    if let Some(bad) = sources.iter().find(|s| s.column >= num_columns) {
        return Err(Error::UnknownPolynomial {
            index: bad.column,
            len: num_columns,
        });
    }
    Ok(())
}

/// Accumulator cells a worker takes at a time. Small enough that a short table
/// still spreads, large enough that the columns streaming through one slab pay
/// for the handoff.
const ACCUMULATOR_CHUNK: usize = 1 << 12;

/// `Σ_{i reads at `offset`} gamma^i · column_i`, the one polynomial that offset's
/// kernel multiplies.
pub(crate) fn batched_column<F, E>(
    columns: &[Mle<F>],
    sources: &[FactorSource],
    weights: &[FieldElement<E>],
    offset: usize,
    num_vars: usize,
) -> Result<Mle<E>, Error>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    // Which columns read at this offset, so the walk below carries its whole
    // list and the accumulator is written once rather than once per column.
    let members: Vec<(usize, usize)> = sources
        .iter()
        .enumerate()
        .filter(|(_, source)| source.offset == offset)
        .map(|(i, source)| (i, source.column))
        .collect();
    let mut acc = vec![FieldElement::<E>::zero(); 1usize << num_vars];
    // A slab of the accumulator, with every column streamed through it: a
    // table brings dozens of columns, and this way its cells are touched once.
    let fill = |(index, slab): (usize, &mut [FieldElement<E>])| {
        let at = index * ACCUMULATOR_CHUNK;
        for (weight, column) in &members {
            let values = &columns[*column].evals()[at..at + slab.len()];
            for (slot, value) in slab.iter_mut().zip(values) {
                // The base element on the left: the only direction the tower
                // gives.
                *slot += value * &weights[*weight];
            }
        }
    };
    #[cfg(feature = "parallel")]
    acc.par_chunks_mut(ACCUMULATOR_CHUNK)
        .enumerate()
        .for_each(fill);
    #[cfg(not(feature = "parallel"))]
    acc.chunks_mut(ACCUMULATOR_CHUNK).enumerate().for_each(fill);
    Mle::new(acc)
}

/// Every column at `point`, one at a time: what runs when the card does not
/// take the table whole.
///
/// Under `LAMBDA_VM_ARGUE_DEVICE_COLUMNS` the walk is spread over the pool when
/// each column is a host loop — the values are the same either way, every
/// column's fold being its own. Columns tall enough that `evaluate_in` sends
/// each to the device stay in order: they only get here when the card turned
/// the table down, and a pool of them would each ask it for a slab it has not
/// promised.
fn evaluate_each<F, E>(
    columns: &[Mle<F>],
    point: &[FieldElement<E>],
) -> Result<Vec<FieldElement<E>>, Error>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    let on_host = !crate::gpu::evaluates_on_device(columns.first().map_or(0, Mle::len));
    if on_host {
        crate::gpu::note_host_evaluations(columns.len());
    }
    let one = |column: &Mle<F>| column.evaluate_in(point);
    #[cfg(feature = "parallel")]
    if on_host && crate::gpu::argue_device_columns() {
        return columns.par_iter().map(one).collect();
    }
    columns.iter().map(one).collect()
}

/// `LAMBDA_VM_ARGUE_XCHECK`: every value the card handed back, recomputed here
/// and compared. A mismatch fails the proof and names the column, where a
/// verifier would only say that the proof does not verify.
fn check_against_the_host<F, E>(
    columns: &[Mle<F>],
    point: &[FieldElement<E>],
    device: &[FieldElement<E>],
) -> Result<(), Error>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    let one = |column: &Mle<F>| column.evaluate_in_on_host(point);
    #[cfg(feature = "parallel")]
    let host: Vec<FieldElement<E>> = columns.par_iter().map(one).collect::<Result<_, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let host: Vec<FieldElement<E>> = columns.iter().map(one).collect::<Result<_, _>>()?;
    let differs = if host.len() != device.len() {
        Some(host.len().min(device.len()))
    } else {
        (0..host.len()).find(|&k| host[k] != device[k])
    };
    if let Some(k) = differs {
        eprintln!(
            "[argue] XCHECK: the card's value of column {k} of {} (2^{} rows) is not the \
             host's: card {:?}, host {:?}",
            columns.len(),
            point.len(),
            device.get(k),
            host.get(k),
        );
        return Err(Error::DeviceFailed {
            stage: "column evaluation",
        });
    }
    crate::gpu::note_xchecked(columns.len());
    Ok(())
}

/// Reduces every factor's claimed value at `alpha` to one claim per column.
///
/// Returns the proof and the reduced point. `factor_values[i]` must be the
/// value of the factor `sources[i]` describes — the column shifted by its
/// offset, evaluated at `alpha`.
pub fn prove<F, E, T>(
    columns: &[Mle<F>],
    sources: &[FactorSource],
    factor_values: &[FieldElement<E>],
    alpha: &[FieldElement<E>],
    resident: Option<(&crate::gpu::ResidentColumns, usize)>,
    transcript: &mut T,
) -> Result<(ReduceProof<E>, Vec<FieldElement<E>>), Error>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
    T: IsTranscript<E>,
{
    let t = crate::whir_split::tick();
    check_shape(sources, factor_values, columns.len())?;
    for column in columns {
        if column.num_vars() != alpha.len() {
            return Err(Error::VariableCountMismatch {
                expected: alpha.len(),
                got: column.num_vars(),
            });
        }
    }

    // The claims are what is being reduced, so the batching challenge must come
    // after them.
    for value in factor_values {
        transcript.append_field_element(value);
    }
    let weights = challenge_powers(&transcript.sample_field_element(), sources.len());

    // Under `LAMBDA_VM_ARGUE_DEVICE_TABLES` the tables are built where they are
    // folded: the card takes the point, the columns each offset reads and their
    // weights, and hands back the tables folded to the crossover.
    let on_card = if crate::gpu::argue_device_tables() {
        let offsets = offsets(sources);
        crate::gpu::prove_reduce_resident(
            columns,
            sources,
            &weights,
            &offsets,
            alpha,
            resident,
            &pair_products_program::<E>(offsets.len())?,
            |evaluations| {
                for e in evaluations {
                    transcript.append_field_element(e);
                }
                transcript.sample_field_element()
            },
        )
    } else {
        None
    };
    let (sumcheck, point) = match on_card {
        Some(outcome) => {
            let (mut rounds, mut point, folded) = outcome?;
            // The rounds past the crossover run here, over the tables the card
            // folded — as they do after an upload, and through the same loop:
            // a cube at the crossover is one no device takes.
            let pairs = folded.len() / 2;
            let mut rest = Composed::new(folded, pair_products::<E>, 2)?
                .with_program(pair_products_program::<E>(pairs)?);
            let left = rest.num_vars();
            let (tail, tail_point) = sumcheck::prove_rounds(&mut rest, left, transcript)?;
            rounds.extend(tail);
            point.extend(tail_point);
            (SumcheckProof { rounds }, point)
        }
        None => {
            let mut polys = Vec::with_capacity(2 * offsets(sources).len());
            for offset in offsets(sources) {
                polys.push(shift_mle(alpha, offset)?);
                polys.push(batched_column::<F, E>(
                    columns,
                    sources,
                    &weights,
                    offset,
                    alpha.len(),
                )?);
            }

            let pairs = polys.len() / 2;
            sumcheck::prove(
                Composed::new(polys, pair_products::<E>, 2)?
                    .with_program(pair_products_program::<E>(pairs)?),
                transcript,
            )?
        }
    };

    crate::whir_split::add_tick(&crate::whir_split::REST_REDUCE, t);

    // All at the same point, so they fold together: one upload and one launch
    // per level for the table instead of per column.
    let t = crate::whir_split::tick();
    let column_values = match crate::gpu::evaluate_many_base(columns, &point, resident) {
        Some(values) => {
            if crate::gpu::argue_xcheck() {
                check_against_the_host(columns, &point, &values)?;
            }
            values
        }
        None => evaluate_each(columns, &point)?,
    };
    for value in &column_values {
        transcript.append_field_element(value);
    }
    crate::whir_split::add_tick(&crate::whir_split::REST_COLUMNS, t);

    Ok((
        ReduceProof {
            sumcheck,
            column_values,
        },
        point,
    ))
}

/// Checks the reduction and returns the claims the commitment scheme must
/// settle.
pub fn verify<E, T>(
    proof: &ReduceProof<E>,
    sources: &[FactorSource],
    factor_values: &[FieldElement<E>],
    alpha: &[FieldElement<E>],
    num_columns: usize,
    transcript: &mut T,
) -> Result<ReducedClaim<E>, Error>
where
    E: IsField + 'static,
    T: IsTranscript<E>,
{
    check_shape(sources, factor_values, num_columns)?;
    if proof.column_values.len() != num_columns {
        return Err(Error::QueryCountMismatch {
            expected: num_columns,
            got: proof.column_values.len(),
        });
    }

    for value in factor_values {
        transcript.append_field_element(value);
    }
    let weights = challenge_powers(&transcript.sample_field_element(), sources.len());

    let claimed = weights
        .iter()
        .zip(factor_values)
        .fold(FieldElement::<E>::zero(), |acc, (w, v)| acc + w * v);
    let claim = sumcheck::verify(&proof.sumcheck, claimed, alpha.len(), 2, transcript)?;

    // The kernels are closed forms, so the residual is entirely about the
    // columns — and the columns are what the commitments answer for.
    let mut rebuilt = FieldElement::<E>::zero();
    for offset in offsets(sources) {
        let batched =
            sources
                .iter()
                .enumerate()
                .fold(FieldElement::<E>::zero(), |acc, (i, source)| {
                    if source.offset == offset {
                        acc + &weights[i] * &proof.column_values[source.column]
                    } else {
                        acc
                    }
                });
        rebuilt += shift_eval(alpha, &claim.point, offset)? * batched;
    }
    if rebuilt != claim.expected_evaluation {
        return Err(Error::ShiftedReadMismatch);
    }

    for value in &proof.column_values {
        transcript.append_field_element(value);
    }

    Ok(ReducedClaim {
        point: claim.point,
        column_values: proof.column_values.clone(),
    })
}

/// The factor `source` describes, materialized: the column shifted cyclically
/// by its offset.
/// A factor's value at `point`, without building the view when there is none
/// to build.
///
/// An unshifted source *is* its column, and a column is the biggest thing the
/// argument holds: copying one to read a single value out of it is the whole
/// trace copied again, once per factor.
pub fn evaluate_source<F, E>(
    columns: &[Mle<F>],
    source: &FactorSource,
    point: &[FieldElement<E>],
) -> Result<FieldElement<E>, Error>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    let column = columns.get(source.column).ok_or(Error::UnknownPolynomial {
        index: source.column,
        len: columns.len(),
    })?;
    if source.offset.is_multiple_of(column.len()) {
        return column.evaluate_in(point);
    }
    materialize(columns, source)?.evaluate_in(point)
}

pub fn materialize<E: IsField + 'static>(
    columns: &[Mle<E>],
    source: &FactorSource,
) -> Result<Mle<E>, Error> {
    let column = columns.get(source.column).ok_or(Error::UnknownPolynomial {
        index: source.column,
        len: columns.len(),
    })?;
    let size = column.len();
    let shift = source.offset % size;
    if shift == 0 {
        return Ok(column.clone());
    }
    Mle::new(
        (0..size)
            .map(|i| column.evals()[(i + shift) % size].clone())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"claim-reduce-test")
    }

    fn column(num_vars: usize, seed: u64) -> Mle<F> {
        let vals: Vec<FE> = (0..(1u64 << num_vars))
            .map(|i| FE::from(i.wrapping_mul(seed).wrapping_add(seed * 7 + 1)))
            .collect();
        Mle::new(vals).unwrap()
    }

    fn point(num_vars: usize) -> Vec<FE> {
        (0..num_vars).map(|i| FE::from(101 + i as u64)).collect()
    }

    /// What an honest prover claims: each factor is its column shifted by its
    /// offset, evaluated at `alpha`.
    fn honest_values(columns: &[Mle<F>], sources: &[FactorSource], alpha: &[FE]) -> Vec<FE> {
        sources
            .iter()
            .map(|s| materialize(columns, s).unwrap().evaluate(alpha).unwrap())
            .collect()
    }

    /// Proves honestly, then verifies whatever `factor_values` the caller wants
    /// to present.
    fn run(
        columns: &[Mle<F>],
        sources: &[FactorSource],
        prover_values: &[FE],
        verifier_values: &[FE],
        alpha: &[FE],
    ) -> Result<ReducedClaim<F>, Error> {
        let (proof, _) = prove(
            columns,
            sources,
            prover_values,
            alpha,
            None,
            &mut transcript(),
        )?;
        verify(
            &proof,
            sources,
            verifier_values,
            alpha,
            columns.len(),
            &mut transcript(),
        )
    }

    #[test]
    fn honest_claims_reduce_to_the_columns() {
        for num_vars in 1..=4usize {
            let columns = vec![column(num_vars, 3), column(num_vars, 5)];
            let sources = [
                FactorSource::direct(0),
                FactorSource::shifted(0, 1),
                FactorSource::direct(1),
            ];
            let alpha = point(num_vars);
            let values = honest_values(&columns, &sources, &alpha);

            let claim = run(&columns, &sources, &values, &values, &alpha)
                .unwrap_or_else(|e| panic!("num_vars={num_vars}: {e:?}"));

            // The claims handed on must be the columns' real values there.
            for (c, value) in columns.iter().zip(&claim.column_values) {
                assert_eq!(&c.evaluate(&claim.point).unwrap(), value);
            }
        }
    }

    /// The reason this module exists: a "next step" factor that is not the shift
    /// of the column it names cannot be passed off as one.
    #[test]
    fn an_unrelated_table_cannot_pass_as_a_shift() {
        let num_vars = 3;
        let columns = vec![column(num_vars, 3)];
        let sources = [FactorSource::direct(0), FactorSource::shifted(0, 1)];
        let alpha = point(num_vars);

        let mut lying = honest_values(&columns, &sources, &alpha);
        lying[1] = column(num_vars, 11).evaluate(&alpha).unwrap();
        assert_ne!(lying[1], honest_values(&columns, &sources, &alpha)[1]);

        assert!(run(&columns, &sources, &lying, &lying, &alpha).is_err());
    }

    /// Same lie, but from a prover that proves the relation it wants: the
    /// sumcheck is then internally consistent and the rebuild is what rejects.
    #[test]
    fn a_forged_column_value_is_rejected() {
        let num_vars = 3;
        let columns = vec![column(num_vars, 3), column(num_vars, 5)];
        let sources = [FactorSource::direct(0), FactorSource::shifted(1, 1)];
        let alpha = point(num_vars);
        let values = honest_values(&columns, &sources, &alpha);

        let (mut proof, _) =
            prove(&columns, &sources, &values, &alpha, None, &mut transcript()).unwrap();
        proof.column_values[1] += FE::one();

        let err = verify(
            &proof,
            &sources,
            &values,
            &alpha,
            columns.len(),
            &mut transcript(),
        )
        .unwrap_err();
        assert_eq!(err, Error::ShiftedReadMismatch);
    }

    #[test]
    fn swapping_two_column_values_is_rejected() {
        let num_vars = 3;
        let columns = vec![column(num_vars, 3), column(num_vars, 5)];
        let sources = [FactorSource::direct(0), FactorSource::direct(1)];
        let alpha = point(num_vars);
        let values = honest_values(&columns, &sources, &alpha);

        let (mut proof, _) =
            prove(&columns, &sources, &values, &alpha, None, &mut transcript()).unwrap();
        proof.column_values.swap(0, 1);

        assert_eq!(
            verify(
                &proof,
                &sources,
                &values,
                &alpha,
                columns.len(),
                &mut transcript()
            )
            .unwrap_err(),
            Error::ShiftedReadMismatch
        );
    }

    #[test]
    fn a_forged_factor_value_is_rejected() {
        let num_vars = 3;
        let columns = vec![column(num_vars, 7)];
        let sources = [FactorSource::direct(0), FactorSource::shifted(0, 1)];
        let alpha = point(num_vars);
        let honest = honest_values(&columns, &sources, &alpha);
        let mut forged = honest.clone();
        forged[0] += FE::one();

        assert!(run(&columns, &sources, &honest, &forged, &alpha).is_err());
    }

    #[test]
    fn offsets_beyond_one_reduce_too() {
        // The example AIRs in this repo read two steps ahead, so the kernel is
        // not specialized to the rotation.
        let num_vars = 3;
        let columns = vec![column(num_vars, 3)];
        let sources = [
            FactorSource::direct(0),
            FactorSource::shifted(0, 1),
            FactorSource::shifted(0, 2),
        ];
        let alpha = point(num_vars);
        let values = honest_values(&columns, &sources, &alpha);

        run(&columns, &sources, &values, &values, &alpha).unwrap();
    }

    #[test]
    fn one_sumcheck_covers_every_factor() {
        // The cost of the reduction is one pass over the cube, not one per
        // factor.
        let num_vars = 4;
        let columns = vec![column(num_vars, 3), column(num_vars, 5)];
        let sources: Vec<FactorSource> = (0..2)
            .flat_map(|c| {
                [
                    FactorSource::direct(c),
                    FactorSource::shifted(c, 1),
                    FactorSource::shifted(c, 2),
                ]
            })
            .collect();
        let alpha = point(num_vars);
        let values = honest_values(&columns, &sources, &alpha);

        let (proof, _) =
            prove(&columns, &sources, &values, &alpha, None, &mut transcript()).unwrap();
        assert_eq!(proof.sumcheck.rounds.len(), num_vars);
        assert_eq!(proof.column_values.len(), 2);
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        let num_vars = 3;
        let columns = vec![column(num_vars, 3)];
        let sources = [FactorSource::direct(0), FactorSource::shifted(0, 1)];
        let alpha = point(num_vars);
        let values = honest_values(&columns, &sources, &alpha);

        let (proof, _) =
            prove(&columns, &sources, &values, &alpha, None, &mut transcript()).unwrap();
        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        assert!(verify(&proof, &sources, &values, &alpha, columns.len(), &mut other).is_err());
    }

    #[test]
    fn a_source_naming_a_column_that_is_not_there_is_rejected() {
        let columns = vec![column(3, 3)];
        let sources = [FactorSource::direct(1)];
        let alpha = point(3);
        assert_eq!(
            prove(
                &columns,
                &sources,
                &[FE::zero()],
                &alpha,
                None,
                &mut transcript()
            )
            .unwrap_err(),
            Error::UnknownPolynomial { index: 1, len: 1 }
        );
    }

    #[test]
    fn a_claim_per_factor_is_required() {
        let columns = vec![column(3, 3)];
        let sources = [FactorSource::direct(0), FactorSource::shifted(0, 1)];
        let alpha = point(3);
        assert_eq!(
            prove(
                &columns,
                &sources,
                &[FE::zero()],
                &alpha,
                None,
                &mut transcript()
            )
            .unwrap_err(),
            Error::VariableCountMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    /// `LAMBDA_VM_ARGUE_DEVICE_COLUMNS` spreads the host walk over the pool, and
    /// the proof must not notice: the same rounds, the same column values, the
    /// same transcript after them.
    ///
    /// The knob is process-wide, so this is the only test here that sets it.
    #[test]
    fn the_device_columns_knob_moves_no_value_of_the_host_walk() {
        let num_vars = 6;
        let columns: Vec<Mle<F>> = (0..9).map(|k| column(num_vars, 3 + 2 * k)).collect();
        let sources: Vec<FactorSource> = (0..9)
            .flat_map(|c| [FactorSource::direct(c), FactorSource::shifted(c, 1 + c % 3)])
            .collect();
        let alpha = point(num_vars);
        let values = honest_values(&columns, &sources, &alpha);

        let run = |on: bool| {
            crate::gpu::force_argue_device_columns(Some(on));
            let walked = crate::gpu::host_evaluate_calls();
            let mut t = transcript();
            let (proof, reduced) = prove(&columns, &sources, &values, &alpha, None, &mut t)
                .unwrap_or_else(|e| panic!("knob {on}: {e:?}"));
            (
                proof,
                reduced,
                t,
                crate::gpu::host_evaluate_calls() - walked,
            )
        };
        let (off, off_point, mut off_t, off_walked) = run(false);
        let (on, on_point, mut on_t, on_walked) = run(true);
        crate::gpu::force_argue_device_columns(None);

        assert_eq!(off.sumcheck, on.sumcheck, "the rounds moved");
        assert_eq!(off.column_values, on.column_values, "a column value moved");
        assert_eq!(off_point, on_point, "the reduced point moved");
        assert_eq!(off_t.state(), on_t.state(), "the transcripts parted");
        assert_eq!(
            off_t.sample_field_element(),
            on_t.sample_field_element(),
            "the next challenge moved"
        );
        // No card takes a base-field reduce, so both walked every column here.
        assert!(
            off_walked >= 9 && on_walked >= 9,
            "the host walk went uncounted: {off_walked} and {on_walked} of 9"
        );
        verify(
            &on,
            &sources,
            &values,
            &alpha,
            columns.len(),
            &mut transcript(),
        )
        .expect("the proof with the knob on verifies");
    }

    #[test]
    fn materializing_offset_zero_is_the_column_itself() {
        let columns = vec![column(3, 3)];
        assert_eq!(
            materialize(&columns, &FactorSource::direct(0)).unwrap(),
            columns[0]
        );
    }

    #[test]
    fn materializing_a_shift_moves_every_row_up() {
        let columns = vec![column(3, 3)];
        let shifted = materialize(&columns, &FactorSource::shifted(0, 1)).unwrap();
        let size = columns[0].len();
        for i in 0..size {
            assert_eq!(shifted.evals()[i], columns[0].evals()[(i + 1) % size]);
        }
    }
}

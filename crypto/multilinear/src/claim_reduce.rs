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

use crate::{
    Error, challenge_powers,
    eq::{shift_eval, shift_mle},
    mle::Mle,
    poly::Composed,
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

/// `Σ_{i reads at `offset`} gamma^i · column_i`, the one polynomial that offset's
/// kernel multiplies.
fn batched_column<F, E>(
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
    let mut acc = vec![FieldElement::<E>::zero(); 1usize << num_vars];
    for (i, source) in sources.iter().enumerate() {
        if source.offset != offset {
            continue;
        }
        for (slot, value) in acc.iter_mut().zip(columns[source.column].evals()) {
            // The base element on the left: the only direction the tower gives.
            *slot += value * &weights[i];
        }
    }
    Mle::new(acc)
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
    transcript: &mut T,
) -> Result<(ReduceProof<E>, Vec<FieldElement<E>>), Error>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
    T: IsTranscript<E>,
{
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
    let (sumcheck, point) = sumcheck::prove(
        Composed::new(polys, pair_products::<E>, 2)?
            .with_program(pair_products_program::<E>(pairs)?),
        transcript,
    )?;

    let column_values = columns
        .iter()
        .map(|c| c.evaluate_in(&point))
        .collect::<Result<Vec<_>, _>>()?;
    for value in &column_values {
        transcript.append_field_element(value);
    }

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
        let (proof, _) = prove(columns, sources, prover_values, alpha, &mut transcript())?;
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

        let (mut proof, _) = prove(&columns, &sources, &values, &alpha, &mut transcript()).unwrap();
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

        let (mut proof, _) = prove(&columns, &sources, &values, &alpha, &mut transcript()).unwrap();
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

        let (proof, _) = prove(&columns, &sources, &values, &alpha, &mut transcript()).unwrap();
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

        let (proof, _) = prove(&columns, &sources, &values, &alpha, &mut transcript()).unwrap();
        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        assert!(verify(&proof, &sources, &values, &alpha, columns.len(), &mut other).is_err());
    }

    #[test]
    fn a_source_naming_a_column_that_is_not_there_is_rejected() {
        let columns = vec![column(3, 3)];
        let sources = [FactorSource::direct(1)];
        let alpha = point(3);
        assert_eq!(
            prove(&columns, &sources, &[FE::zero()], &alpha, &mut transcript()).unwrap_err(),
            Error::UnknownPolynomial { index: 1, len: 1 }
        );
    }

    #[test]
    fn a_claim_per_factor_is_required() {
        let columns = vec![column(3, 3)];
        let sources = [FactorSource::direct(0), FactorSource::shifted(0, 1)];
        let alpha = point(3);
        assert_eq!(
            prove(&columns, &sources, &[FE::zero()], &alpha, &mut transcript()).unwrap_err(),
            Error::VariableCountMismatch {
                expected: 2,
                got: 1
            }
        );
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

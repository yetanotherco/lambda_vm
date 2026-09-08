//! The whole thing: proving a **committed** trace satisfies a constraint.
//!
//! This is where the pieces stop being separately-correct machinery and become
//! an argument. [`zerocheck`](crate::zerocheck) reduces "`C` vanishes on every
//! row" to "`C` takes this value at this point", and hands that back unsettled.
//! [`whir_eval`](crate::whir_eval) settles exactly that kind of claim about a
//! committed polynomial. Composing them closes the loop:
//!
//! 1. commit each trace column;
//! 2. zerocheck the constraint, leaving a claim at a random point `p`;
//! 3. prove each column's value at `p` against its commitment;
//! 4. rebuild `C(p)` from those values and check it against the claim.
//!
//! Step 4 is what makes step 3 necessary and step 2 meaningful. Without the
//! commitments a prover answers step 2 with whatever number closes the proof;
//! without step 4 the column values are unconstrained.
//!
//! # Structure without data
//!
//! The verifier needs to recompute `C(p)` but must not have the trace. The
//! constraint therefore travels as a closure over *values* — see
//! [`Composed`](crate::poly::Composed) — which both sides hold, while only the
//! prover holds the columns.
//!
//! # Scope
//!
//! Columns are committed and opened one at a time. A real system batches both,
//! and folds `k` variables per WHIR round rather than all at once; see
//! [`whir_eval`](crate::whir_eval). Nothing here changes what is proven, only
//! how much it costs.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsPrimeField},
    },
    traits::AsBytes,
};

use crate::{
    Error,
    mle::Mle,
    poly::Composed,
    whir_commit::Commitment,
    whir_eval::{self, EvalConfig, EvalProof},
    zerocheck::{self, ZeroCheckProof},
};

/// A committed trace, ready to be argued about.
pub struct CommittedTrace<F: IsFFTField + IsPrimeField>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    columns: Vec<Mle<F>>,
    commitments: Vec<crate::whir_commit::CodewordCommitment<F>>,
    domain: crate::whir::Domain<F>,
}

impl<F: IsFFTField + IsPrimeField> CommittedTrace<F>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    /// Commits every column. All must agree on height.
    pub fn commit(columns: Vec<Mle<F>>, config: &EvalConfig) -> Result<Self, Error> {
        let num_vars = columns.first().map(|c| c.num_vars()).unwrap_or(0);
        let mut commitments = Vec::with_capacity(columns.len());
        let mut domain = None;
        for column in &columns {
            if column.num_vars() != num_vars {
                return Err(Error::VariableCountMismatch {
                    expected: num_vars,
                    got: column.num_vars(),
                });
            }
            let (commitment, d) = whir_eval::commit(column, config)?;
            commitments.push(commitment);
            domain = Some(d);
        }
        Ok(Self {
            columns,
            commitments,
            domain: domain.ok_or(Error::EmptyPolynomial)?,
        })
    }

    pub fn roots(&self) -> Vec<Commitment> {
        self.commitments.iter().map(|c| c.root()).collect()
    }

    pub fn domain(&self) -> &crate::whir::Domain<F> {
        &self.domain
    }

    pub fn num_vars(&self) -> usize {
        self.columns.first().map(|c| c.num_vars()).unwrap_or(0)
    }
}

/// The public part of the statement: what was committed and over what domain.
#[derive(Clone, Copy, Debug)]
pub struct TraceClaim<'a, F: IsFFTField + IsPrimeField> {
    pub roots: &'a [Commitment],
    pub domain: &'a crate::whir::Domain<F>,
    pub num_vars: usize,
}

/// A proof that the committed columns satisfy the constraint.
#[derive(Clone, Debug)]
pub struct ConstraintProof<F: IsFFTField + IsPrimeField> {
    pub zerocheck: ZeroCheckProof<F>,
    /// Each column's value at the zerocheck point.
    pub column_values: Vec<FieldElement<F>>,
    /// One evaluation proof per column, in the same order.
    pub column_proofs: Vec<EvalProof<F>>,
}

/// Proves that `combine` applied to the committed columns vanishes on every row.
///
/// `combine` and `degree` describe the constraint and must match what the
/// verifier is given.
pub fn prove<F, T, C>(
    trace: &CommittedTrace<F>,
    combine: C,
    degree: usize,
    config: &EvalConfig,
    transcript: &mut T,
) -> Result<ConstraintProof<F>, Error>
where
    F: IsFFTField + IsPrimeField,
    FieldElement<F>: AsBytes + Sync + Send,
    T: IsTranscript<F>,
    C: Fn(&[FieldElement<F>]) -> FieldElement<F>,
{
    for root in trace.roots() {
        transcript.append_bytes(&root);
    }

    let constraint = Composed::new(trace.columns.clone(), &combine, degree)?;
    let out = zerocheck::prove(constraint, transcript)?;

    // The claim the zerocheck leaves is about C at its sumcheck point; settle it
    // by opening every column there.
    let point = out.point;
    let mut column_values = Vec::with_capacity(trace.columns.len());
    let mut column_proofs = Vec::with_capacity(trace.columns.len());
    for (column, commitment) in trace.columns.iter().zip(&trace.commitments) {
        let value = column.evaluate(&point)?;
        let proof = whir_eval::prove(
            column,
            &point,
            commitment,
            &trace.domain,
            config,
            transcript,
        )?;
        column_values.push(value);
        column_proofs.push(proof);
    }

    Ok(ConstraintProof {
        zerocheck: out.proof,
        column_values,
        column_proofs,
    })
}

/// Verifies the constraint against the commitments.
///
/// `combine` and `degree` must be the ones the prover used; they are the
/// statement, not part of the proof.
pub fn verify<F, T, C>(
    proof: &ConstraintProof<F>,
    claim_shape: TraceClaim<'_, F>,
    combine: C,
    degree: usize,
    config: &EvalConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    T: IsTranscript<F>,
    C: Fn(&[FieldElement<F>]) -> FieldElement<F>,
{
    let roots = claim_shape.roots;
    if proof.column_values.len() != roots.len() || proof.column_proofs.len() != roots.len() {
        return Err(Error::QueryCountMismatch {
            expected: roots.len(),
            got: proof.column_values.len().min(proof.column_proofs.len()),
        });
    }
    for root in roots {
        transcript.append_bytes(root);
    }

    let claim = zerocheck::verify(&proof.zerocheck, claim_shape.num_vars, degree, transcript)?;
    let required = claim
        .constraint_evaluation()
        .ok_or(Error::DegenerateEvaluationPoint)?;

    // Each claimed column value must really be that column's, at the same point.
    for (i, ((value, eval_proof), root)) in proof
        .column_values
        .iter()
        .zip(&proof.column_proofs)
        .zip(roots)
        .enumerate()
    {
        whir_eval::verify(
            eval_proof,
            root,
            &claim.point,
            value.clone(),
            claim_shape.domain,
            config,
            transcript,
        )
        .map_err(|_| Error::ColumnOpeningRejected { column: i })?;
    }

    // And the constraint rebuilt from them must be what the zerocheck demanded.
    if combine(&proof.column_values) != required {
        return Err(Error::ConstraintMismatch);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"constraint-argument-test")
    }

    fn config() -> EvalConfig {
        EvalConfig {
            log_blowup: 2,
            num_queries: 3,
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
        let trace = CommittedTrace::commit(columns, &config()).unwrap();
        let proof = prove(&trace, constraint, 2, &config(), &mut transcript())?;

        verify(
            &proof,
            TraceClaim {
                roots: &trace.roots(),
                domain: trace.domain(),
                num_vars,
            },
            constraint,
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
    fn a_forged_column_value_is_rejected() {
        let columns = satisfying(3);
        let num_vars = 3;
        let trace = CommittedTrace::commit(columns, &config()).unwrap();
        let mut proof = prove(&trace, constraint, 2, &config(), &mut transcript()).unwrap();

        // Claim a different value for one column, leaving everything else.
        proof.column_values[0] += FE::one();

        let err = verify(
            &proof,
            TraceClaim {
                roots: &trace.roots(),
                domain: trace.domain(),
                num_vars,
            },
            constraint,
            2,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::ColumnOpeningRejected { column: 0 }));
    }

    #[test]
    fn verifying_a_different_constraint_is_rejected() {
        let columns = satisfying(3);
        let trace = CommittedTrace::commit(columns, &config()).unwrap();
        let proof = prove(&trace, constraint, 2, &config(), &mut transcript()).unwrap();

        // `a·b + c` instead of `a·b − c`: same degree, same columns.
        let err = verify(
            &proof,
            TraceClaim {
                roots: &trace.roots(),
                domain: trace.domain(),
                num_vars: 3,
            },
            |v: &[FE]| v[0] * v[1] + v[2],
            2,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::ConstraintMismatch | Error::RoundSumMismatch { .. }
        ));
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        let columns = satisfying(3);
        let trace = CommittedTrace::commit(columns, &config()).unwrap();
        let proof = prove(&trace, constraint, 2, &config(), &mut transcript()).unwrap();

        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        assert!(
            verify(
                &proof,
                TraceClaim {
                    roots: &trace.roots(),
                    domain: trace.domain(),
                    num_vars: 3,
                },
                constraint,
                2,
                &config(),
                &mut other,
            )
            .is_err()
        );
    }

    #[test]
    fn a_proof_missing_a_column_is_rejected() {
        let columns = satisfying(3);
        let trace = CommittedTrace::commit(columns, &config()).unwrap();
        let mut proof = prove(&trace, constraint, 2, &config(), &mut transcript()).unwrap();
        proof.column_values.pop();

        let err = verify(
            &proof,
            TraceClaim {
                roots: &trace.roots(),
                domain: trace.domain(),
                num_vars: 3,
            },
            constraint,
            2,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::QueryCountMismatch { expected: 3, .. }));
    }

    #[test]
    fn columns_of_differing_heights_are_rejected() {
        let err = CommittedTrace::commit(vec![mle(&[1, 2]), mle(&[1, 2, 3, 4])], &config())
            .err()
            .unwrap();
        assert!(matches!(err, Error::VariableCountMismatch { .. }));
    }
}

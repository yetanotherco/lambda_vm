//! One WHIR round: sample queries, open the current codeword's blocks, fold them
//! locally, and check they match the committed successor.
//!
//! The successor is committed before the queries are drawn, so it cannot be
//! chosen to match. Consistency holds only where the queries land.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField},
    },
    traits::AsBytes,
};

use crate::{
    Error,
    whir::Domain,
    whir_commit::{CodewordCommitment, Commitment, CosetOpening, fold_coset, leaf_and_slot},
};

/// How hard a round is to cheat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundConfig {
    /// Positions checked. Each one an inconsistent prover must get lucky on.
    pub num_queries: usize,
    /// `k`: the codeword folds by `2^k` and blocks hold `2^k` values.
    pub log_folding: usize,
}

/// What the verifier already knows about the two codewords: their roots, and
/// how the successor was blocked.
#[derive(Clone, Copy, Debug)]
pub struct RoundCommitments<'a> {
    pub current_root: &'a Commitment,
    pub next_root: &'a Commitment,
    /// Leaves in the successor's tree, needed to locate a position in it.
    pub next_num_leaves: usize,
}

/// The openings one round sends.
#[derive(Clone, Debug)]
pub struct RoundProof<F: IsField> {
    /// Per query: the block of the current codeword that folds onto the query.
    pub current: Vec<CosetOpening<F>>,
    /// Per query: the successor block holding the folded value.
    pub next: Vec<CosetOpening<F>>,
}

/// Draws the query positions. Both sides run this on the same transcript.
///
/// Positions index the *folded* codeword, which is also the block index of the
/// current one.
fn sample_queries<F, T>(transcript: &mut T, num_queries: usize, bound: usize) -> Vec<usize>
where
    F: IsField,
    T: IsTranscript<F>,
{
    (0..num_queries)
        .map(|_| transcript.sample_u64(bound as u64) as usize)
        .collect()
}

/// Produces the openings for one round.
///
/// `current` and `next` must already be committed, and `next` must be the fold
/// of `current` by `alphas` — [`verify`] is what checks that claim.
pub fn prove<F, T>(
    current: &CodewordCommitment<F>,
    next: &CodewordCommitment<F>,
    config: &RoundConfig,
    transcript: &mut T,
) -> Result<RoundProof<F>, Error>
where
    F: IsField,
    FieldElement<F>: AsBytes + Sync + Send,
    T: IsTranscript<F>,
{
    let queries = sample_queries(transcript, config.num_queries, current.num_leaves());

    let mut current_openings = Vec::with_capacity(queries.len());
    let mut next_openings = Vec::with_capacity(queries.len());
    for &q in &queries {
        current_openings.push(current.open(q)?);
        let (leaf, _) = leaf_and_slot(q, next.num_leaves());
        next_openings.push(next.open(leaf)?);
    }

    Ok(RoundProof {
        current: current_openings,
        next: next_openings,
    })
}

/// Checks a round against the two commitments.
///
/// Re-derives the queries from the transcript, so the prover could not have
/// chosen them.
pub fn verify<F, T>(
    proof: &RoundProof<F>,
    commitments: RoundCommitments<'_>,
    domain: &Domain<F>,
    alphas: &[FieldElement<F>],
    config: &RoundConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    T: IsTranscript<F>,
{
    if alphas.len() != config.log_folding {
        return Err(Error::VariableCountMismatch {
            expected: config.log_folding,
            got: alphas.len(),
        });
    }
    if proof.current.len() != config.num_queries || proof.next.len() != config.num_queries {
        return Err(Error::QueryCountMismatch {
            expected: config.num_queries,
            got: proof.current.len().min(proof.next.len()),
        });
    }

    let num_leaves = domain.size() >> config.log_folding;
    let queries = sample_queries(transcript, config.num_queries, num_leaves);

    for (i, (&q, (cur, nxt))) in queries
        .iter()
        .zip(proof.current.iter().zip(&proof.next))
        .enumerate()
    {
        if !crate::whir_commit::verify_opening::<F>(commitments.current_root, q, cur) {
            return Err(Error::OpeningRejected { query: i });
        }
        let (leaf, slot) = leaf_and_slot(q, commitments.next_num_leaves);
        if !crate::whir_commit::verify_opening::<F>(commitments.next_root, leaf, nxt) {
            return Err(Error::OpeningRejected { query: i });
        }

        let folded = fold_coset(&cur.values, domain, q, alphas)?;
        let claimed = nxt.values.get(slot).ok_or(Error::QueryOutOfRange {
            index: slot,
            bound: nxt.values.len(),
        })?;
        if folded != *claimed {
            return Err(Error::FoldInconsistent { query: i });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::{
        mle::Mle,
        whir::{encode, fold_codeword_k, monomial_coefficients},
    };

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"whir-round-test")
    }

    struct Fixture {
        current: CodewordCommitment<F>,
        next: CodewordCommitment<F>,
        domain: Domain<F>,
        alphas: Vec<FE>,
        config: RoundConfig,
    }

    /// An honest round: `next` really is `current` folded by `alphas`.
    fn fixture(num_vars: usize, log_blowup: usize, k: usize) -> Fixture {
        let vals: Vec<FE> = (0..(1u64 << num_vars))
            .map(|i| FE::from((i.wrapping_mul(6364136223846793005).wrapping_add(7)) >> 13))
            .collect();
        let f = Mle::new(vals).unwrap();
        let domain = Domain::<F>::new(num_vars + log_blowup).unwrap();
        let cw = encode(&monomial_coefficients(&f), &domain).unwrap();

        let alphas: Vec<FE> = (0..k).map(|i| FE::from(17 + i as u64)).collect();
        let (folded, _) = fold_codeword_k(&cw, &domain, &alphas).unwrap();

        Fixture {
            current: CodewordCommitment::new(&cw, k).unwrap(),
            next: CodewordCommitment::new(&folded, k.min(1)).unwrap(),
            domain,
            alphas,
            config: RoundConfig {
                num_queries: 4,
                log_folding: k,
            },
        }
    }

    fn run(fx: &Fixture, proof: &RoundProof<F>) -> Result<(), Error> {
        verify(
            proof,
            RoundCommitments {
                current_root: &fx.current.root(),
                next_root: &fx.next.root(),
                next_num_leaves: fx.next.num_leaves(),
            },
            &fx.domain,
            &fx.alphas,
            &fx.config,
            &mut transcript(),
        )
    }

    #[test]
    fn an_honest_round_verifies() {
        for k in 1..=3usize {
            let mut fx = fixture(4, 2, k);
            fx.config.num_queries = 4;
            let proof = prove(&fx.current, &fx.next, &fx.config, &mut transcript()).unwrap();
            run(&fx, &proof).unwrap_or_else(|e| panic!("k={k}: {e:?}"));
        }
    }

    #[test]
    fn the_prover_and_verifier_draw_the_same_queries() {
        let fx = fixture(4, 2, 2);
        let a = sample_queries::<F, _>(&mut transcript(), 8, fx.current.num_leaves());
        let b = sample_queries::<F, _>(&mut transcript(), 8, fx.current.num_leaves());
        assert_eq!(a, b);
        assert!(a.iter().all(|q| *q < fx.current.num_leaves()));
    }

    #[test]
    fn a_tampered_current_opening_is_rejected() {
        let fx = fixture(4, 2, 2);
        let mut proof = prove(&fx.current, &fx.next, &fx.config, &mut transcript()).unwrap();
        proof.current[0].values[0] += FE::one();

        assert!(matches!(
            run(&fx, &proof).unwrap_err(),
            Error::OpeningRejected { query: 0 }
        ));
    }

    #[test]
    fn a_tampered_successor_opening_is_rejected() {
        let fx = fixture(4, 2, 2);
        let mut proof = prove(&fx.current, &fx.next, &fx.config, &mut transcript()).unwrap();
        proof.next[1].values[0] += FE::one();

        assert!(matches!(
            run(&fx, &proof).unwrap_err(),
            Error::OpeningRejected { query: 1 }
        ));
    }

    /// The check the whole round exists for: a successor that is *validly
    /// committed* but is not the fold of the current codeword.
    #[test]
    fn a_successor_that_is_not_the_fold_is_rejected() {
        let mut fx = fixture(4, 2, 2);

        // Commit a different codeword, honestly. Every Merkle opening will
        // verify; only the fold relation breaks.
        let vals: Vec<FE> = (0..(1u64 << 2)).map(|i| FE::from(i + 100)).collect();
        let other = Mle::new(vals).unwrap();
        let other_domain = Domain::<F>::new(2 + 2).unwrap();
        let other_cw = encode(&monomial_coefficients(&other), &other_domain).unwrap();
        fx.next = CodewordCommitment::new(&other_cw, 1).unwrap();

        let proof = prove(&fx.current, &fx.next, &fx.config, &mut transcript()).unwrap();
        assert!(matches!(
            run(&fx, &proof).unwrap_err(),
            Error::FoldInconsistent { .. }
        ));
    }

    #[test]
    fn the_wrong_folding_randomness_is_rejected() {
        let mut fx = fixture(4, 2, 2);
        let proof = prove(&fx.current, &fx.next, &fx.config, &mut transcript()).unwrap();

        fx.alphas[0] += FE::one();
        assert!(matches!(
            run(&fx, &proof).unwrap_err(),
            Error::FoldInconsistent { .. }
        ));
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        // Queries are redrawn, so the openings no longer line up with them.
        let fx = fixture(4, 2, 2);
        let proof = prove(&fx.current, &fx.next, &fx.config, &mut transcript()).unwrap();
        run(&fx, &proof).unwrap();

        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        let result = verify(
            &proof,
            RoundCommitments {
                current_root: &fx.current.root(),
                next_root: &fx.next.root(),
                next_num_leaves: fx.next.num_leaves(),
            },
            &fx.domain,
            &fx.alphas,
            &fx.config,
            &mut other,
        );
        assert!(result.is_err());
    }

    #[test]
    fn a_proof_with_too_few_openings_is_rejected() {
        let fx = fixture(4, 2, 2);
        let mut proof = prove(&fx.current, &fx.next, &fx.config, &mut transcript()).unwrap();
        proof.current.pop();

        assert!(matches!(
            run(&fx, &proof).unwrap_err(),
            Error::QueryCountMismatch {
                expected: 4,
                got: 3
            }
        ));
    }

    #[test]
    fn randomness_of_the_wrong_arity_is_rejected() {
        let mut fx = fixture(4, 2, 2);
        let proof = prove(&fx.current, &fx.next, &fx.config, &mut transcript()).unwrap();
        fx.alphas.pop();

        assert!(matches!(
            run(&fx, &proof).unwrap_err(),
            Error::VariableCountMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn more_queries_check_more_positions() {
        // The knob is real: it changes how many openings travel.
        let mut fx = fixture(4, 2, 1);
        fx.config.num_queries = 7;
        let proof = prove(&fx.current, &fx.next, &fx.config, &mut transcript()).unwrap();
        assert_eq!(proof.current.len(), 7);
        run(&fx, &proof).unwrap();
    }
}

//! Several WHIR chains opened together: one Merkle tree per round for all of
//! them, one set of challenges, one set of query positions.
//!
//! `K` polynomials of the same size, each with its **own** weight, prove
//!
//! ```text
//! Σ_j Σ_x w_j(x)·f_j(x) = y
//! ```
//!
//! by running their chains in lockstep. Every round's sumcheck is over the sum
//! of the `K` products, so it is one round polynomial; every fold uses the same
//! challenges; the `K` successors are committed in **one** tree; one
//! out-of-domain point is drawn and each chain answers it, batched with its own
//! power of the challenge so no chain's answer can hide behind another's; and
//! the queries are drawn once, after every commitment, so each chain is checked
//! at positions its own commitments could not anticipate.
//!
//! # The shared tree
//!
//! Leaf `q` of a round's tree is every chain's block `q` — the strided coset
//! `{q + s·N/2^k}` of its codeword — end to end, chain `j`'s at
//! `j·2^k..(j+1)·2^k` ([`WideCommitment`]). A query pays one authentication
//! path for all of them, and the leaf is hashed by the same backend a single
//! chain's is, so [`verify_opening`] checks it unchanged. Any number of chains:
//! nothing is padded.

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
    eq::{eq_eval, eq_evals},
    mle::Mle,
    poly::Composed,
    sumcheck::{self, RoundProof as SumcheckRoundProof},
    whir::{Domain, encode, fold_codeword_k, lift_coefficients},
    whir_chain::{
        ChainConfig, RoundNonces, RoundOpenings, check_grind, grind, ood_point,
        require_out_of_domain,
    },
    whir_commit::{
        Backend, Commitment, CosetOpening, Tree, coset_of, fold_coset, leaf_and_slot,
        verify_opening,
    },
    whir_round::{RoundConfig, RoundProof},
};
use crypto::merkle_tree::traits::IsMerkleTreeBackend;

/// Several codewords of one length, committed leaf by leaf together: leaf `q`
/// is every codeword's block `q`, end to end.
pub struct WideCommitment<C: IsField>
where
    FieldElement<C>: AsBytes + Sync + Send,
{
    tree: Tree<C>,
    codewords: Vec<Vec<FieldElement<C>>>,
    log_folding: usize,
    log_domain_size: usize,
}

impl<C: IsField + 'static> WideCommitment<C>
where
    FieldElement<C>: AsBytes + Sync + Send,
{
    pub fn new(codewords: Vec<Vec<FieldElement<C>>>, log_folding: usize) -> Result<Self, Error> {
        let len = codewords.first().ok_or(Error::EmptyPolynomial)?.len();
        if !len.is_power_of_two() || codewords.iter().any(|c| c.len() != len) {
            return Err(Error::NotPowerOfTwo(len));
        }
        let log_domain_size = len.trailing_zeros() as usize;
        if log_folding > log_domain_size {
            return Err(Error::ColumnTallerThanStack {
                column_vars: log_folding,
                n_stack: log_domain_size,
            });
        }
        let num_leaves = len >> log_folding;
        let block = 1usize << log_folding;
        let width = block * codewords.len();
        let hash_leaf = |buffer: &mut Vec<FieldElement<C>>, q: usize| {
            buffer.clear();
            for codeword in &codewords {
                buffer.extend((0..block).map(|t| codeword[q + t * num_leaves].clone()));
            }
            Backend::<C>::hash_data(buffer)
        };
        #[cfg(feature = "parallel")]
        let hashed: Vec<_> = (0..num_leaves)
            .into_par_iter()
            .map_init(|| Vec::with_capacity(width), hash_leaf)
            .collect();
        #[cfg(not(feature = "parallel"))]
        let hashed: Vec<_> = {
            let mut buffer = Vec::with_capacity(width);
            (0..num_leaves).map(|q| hash_leaf(&mut buffer, q)).collect()
        };
        let tree = Tree::<C>::build_from_hashed_leaves(hashed).ok_or(Error::EmptyPolynomial)?;
        Ok(Self {
            tree,
            codewords,
            log_folding,
            log_domain_size,
        })
    }

    pub fn root(&self) -> Commitment {
        self.tree.root
    }

    pub fn num_leaves(&self) -> usize {
        1usize << (self.log_domain_size - self.log_folding)
    }

    pub fn codewords(&self) -> &[Vec<FieldElement<C>>] {
        &self.codewords
    }

    pub fn open_many(&self, indices: &[usize]) -> Result<Vec<CosetOpening<C>>, Error> {
        indices
            .iter()
            .map(|&q| {
                let proof = self.tree.get_proof_by_pos(q).ok_or(Error::QueryOutOfRange {
                    index: q,
                    bound: self.num_leaves(),
                })?;
                let positions = coset_of(q, self.log_domain_size, self.log_folding);
                let values = self
                    .codewords
                    .iter()
                    .flat_map(|c| positions.iter().map(move |&p| c[p].clone()))
                    .collect();
                Ok(CosetOpening { values, proof })
            })
            .collect()
    }
}

/// One round of the batch.
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
pub struct BatchRound<F: IsField, E: IsField> {
    pub sumcheck: Vec<SumcheckRoundProof<E>>,
    pub next_root: Option<Commitment>,
    /// Every chain's value at the round's out-of-domain point.
    pub ood_values: Vec<FieldElement<E>>,
    pub nonces: RoundNonces,
    pub openings: RoundOpenings<F, E>,
}

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
pub struct BatchProof<F: IsField, E: IsField> {
    pub rounds: Vec<BatchRound<F, E>>,
    /// Every chain's fully folded value.
    pub final_values: Vec<FieldElement<E>>,
}

impl<F: IsField, E: IsField> BatchProof<F, E> {
    pub fn opened_elements(&self) -> usize {
        self.rounds
            .iter()
            .map(|r| match &r.openings {
                RoundOpenings::Base(p) => {
                    p.current.iter().map(|c| c.values.len()).sum::<usize>()
                        + p.next.iter().map(|c| c.values.len()).sum::<usize>()
                }
                RoundOpenings::Extension(p) => {
                    p.current.iter().map(|c| c.values.len()).sum::<usize>()
                        + p.next.iter().map(|c| c.values.len()).sum::<usize>()
                }
            })
            .sum()
    }
}

/// Commits `polys` — all of the same size — as one shared tree.
pub fn commit<F>(
    polys: &[Mle<F>],
    config: &ChainConfig,
) -> Result<(WideCommitment<F>, Domain<F>), Error>
where
    F: IsFFTField + IsPrimeField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let num_vars = polys.first().ok_or(Error::EmptyPolynomial)?.num_vars();
    if polys.iter().any(|p| p.num_vars() != num_vars) {
        return Err(Error::VariableCountMismatch {
            expected: num_vars,
            got: polys.iter().map(Mle::num_vars).find(|&n| n != num_vars).unwrap_or(0),
        });
    }
    let first = config.schedule(num_vars).first().copied().unwrap_or(0);
    let domain = Domain::<F>::new(num_vars + config.log_blowup)?;
    let encode_one = |p: &Mle<F>| encode::<F, F>(&lift_coefficients(p), &domain);
    #[cfg(feature = "parallel")]
    let codewords = polys.par_iter().map(encode_one).collect::<Result<Vec<_>, Error>>()?;
    #[cfg(not(feature = "parallel"))]
    let codewords = polys.iter().map(encode_one).collect::<Result<Vec<_>, Error>>()?;
    Ok((WideCommitment::new(codewords, first)?, domain))
}

/// Proves `Σ_j Σ_x w_j(x)·f_j(x)` equals what the weights and polynomials say
/// it does, against the shared tree [`commit`] built.
pub fn prove<F, E, T>(
    polys: &[Mle<F>],
    weights: Vec<Mle<E>>,
    commitment: &WideCommitment<F>,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<BatchProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
{
    let chains = polys.len();
    if weights.len() != chains || chains == 0 {
        return Err(Error::QueryCountMismatch {
            expected: chains,
            got: weights.len(),
        });
    }
    let num_vars = polys[0].num_vars();
    let schedule = config.schedule(num_vars);

    // [w_0, f_0, w_1, f_1, ..]: the sumcheck's factors, in the extension.
    let mut factors: Vec<Mle<E>> = Vec::with_capacity(2 * chains);
    for (weight, poly) in weights.into_iter().zip(polys) {
        factors.push(weight);
        factors.push(Mle::new(
            poly.evals().iter().map(|v| v.clone().to_extension::<E>()).collect(),
        )?);
    }
    let sum_of_products = |v: &[FieldElement<E>]| {
        v.chunks(2)
            .fold(FieldElement::<E>::zero(), |acc, pair| acc + &pair[0] * &pair[1])
    };

    let mut current_base: Option<&WideCommitment<F>> = Some(commitment);
    let mut current_ext: Option<WideCommitment<E>> = None;
    let mut current_domain = domain.clone();

    let mut rounds = Vec::with_capacity(schedule.len());
    let mut final_values = Vec::new();

    for (r, &k) in schedule.iter().enumerate() {
        let mut nonces = RoundNonces {
            folding: grind(transcript, config.grind.folding)?,
            ..RoundNonces::default()
        };
        let mut poly = Composed::new(factors, sum_of_products, 2)?;
        let (sumcheck_rounds, alphas) = sumcheck::prove_rounds(&mut poly, k, transcript)?;
        factors = poly.into_polys();

        // Every chain folds by the same challenges.
        let fold = |j: usize| -> Result<(Vec<FieldElement<E>>, Domain<F>), Error> {
            match (current_base, &current_ext) {
                (Some(held), _) => {
                    fold_codeword_k::<F, F, E>(&held.codewords()[j], &current_domain, &alphas)
                }
                (None, Some(held)) => {
                    fold_codeword_k::<F, E, E>(&held.codewords()[j], &current_domain, &alphas)
                }
                (None, None) => Err(Error::EmptyPolynomial),
            }
        };
        #[cfg(feature = "parallel")]
        let folded: Vec<_> = (0..chains).into_par_iter().map(fold).collect::<Result<_, _>>()?;
        #[cfg(not(feature = "parallel"))]
        let folded: Vec<_> = (0..chains).map(fold).collect::<Result<_, _>>()?;
        let folded_domain = folded[0].1.clone();
        let folded: Vec<Vec<FieldElement<E>>> = folded.into_iter().map(|(c, _)| c).collect();

        let round_config = RoundConfig {
            num_queries: config.num_queries,
            log_folding: k,
        };
        let (next, ood_values) = match schedule.get(r + 1) {
            Some(&next_k) => {
                let next = WideCommitment::new(folded, next_k)?;
                transcript.append_bytes(&next.root());

                let z0: FieldElement<E> = transcript.sample_field_element();
                require_out_of_domain::<F, E>(&z0, &folded_domain)?;
                let point = ood_point(&z0, factors[1].num_vars());
                let ood_values: Vec<FieldElement<E>> = (0..chains)
                    .map(|j| factors[2 * j + 1].evaluate(&point))
                    .collect::<Result<_, _>>()?;
                for y0 in &ood_values {
                    transcript.append_field_element(y0);
                }
                nonces.ood = grind(transcript, config.grind.ood)?;
                let gamma: FieldElement<E> = transcript.sample_field_element();
                let eq = eq_evals(&point);
                let mut scale = gamma.clone();
                for j in 0..chains {
                    let weight = &factors[2 * j];
                    let updated: Vec<FieldElement<E>> = weight
                        .evals()
                        .iter()
                        .zip(&eq)
                        .map(|(w, e)| w + &scale * e)
                        .collect();
                    factors[2 * j] = Mle::new(updated)?;
                    scale = &scale * &gamma;
                }
                (Some(next), ood_values)
            }
            None => {
                final_values = folded
                    .iter()
                    .map(|c| c.first().cloned().ok_or(Error::EmptyPolynomial))
                    .collect::<Result<_, _>>()?;
                for value in &final_values {
                    transcript.append_field_element(value);
                }
                (None, Vec::new())
            }
        };
        let next_root = next.as_ref().map(|c| c.root());

        nonces.query = grind(transcript, config.grind.query)?;
        let openings = match (current_base, &current_ext) {
            (Some(held), _) => {
                RoundOpenings::Base(open_round(held, next.as_ref(), &round_config, transcript)?)
            }
            (None, Some(held)) => {
                RoundOpenings::Extension(open_round(held, next.as_ref(), &round_config, transcript)?)
            }
            (None, None) => return Err(Error::EmptyPolynomial),
        };

        rounds.push(BatchRound {
            sumcheck: sumcheck_rounds,
            next_root,
            ood_values,
            nonces,
            openings,
        });
        if let Some(next) = next {
            current_base = None;
            current_ext = Some(next);
        }
        current_domain = folded_domain;
    }
    Ok(BatchProof {
        rounds,
        final_values,
    })
}

/// One round's openings: the queries are drawn as [`crate::whir_round`] draws
/// them, and each opens the current tree's leaf and, but for the last round,
/// the successor's leaf holding the folded value.
fn open_round<C, N, T>(
    current: &WideCommitment<C>,
    next: Option<&WideCommitment<N>>,
    config: &RoundConfig,
    transcript: &mut T,
) -> Result<RoundProof<C, N>, Error>
where
    C: IsField + 'static,
    N: IsField + 'static,
    FieldElement<C>: AsBytes + Sync + Send,
    FieldElement<N>: AsBytes + Sync + Send,
    T: IsTranscript<N>,
{
    let queries: Vec<usize> = (0..config.num_queries)
        .map(|_| transcript.sample_u64(current.num_leaves() as u64) as usize)
        .collect();
    let next_openings = match next {
        Some(next) => {
            let leaves: Vec<usize> = queries
                .iter()
                .map(|&q| leaf_and_slot(q, next.num_leaves()).0)
                .collect();
            next.open_many(&leaves)?
        }
        None => Vec::new(),
    };
    Ok(RoundProof {
        current: current.open_many(&queries)?,
        next: next_openings,
    })
}

/// Checks one round's shared openings: every chain's block folds onto its value
/// in the successor, at positions drawn after every commitment.
#[allow(clippy::too_many_arguments)]
fn verify_openings<F, C, N, T>(
    proof: &RoundProof<C, N>,
    chains: usize,
    current_root: &Commitment,
    next: Option<(&Commitment, usize, usize)>,
    domain: &Domain<F>,
    alphas: &[FieldElement<N>],
    config: &RoundConfig,
    final_values: &[FieldElement<N>],
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<C> + IsSubFieldOf<N>,
    C: IsField + IsSubFieldOf<N> + 'static,
    N: IsField + 'static,
    FieldElement<C>: AsBytes + Sync + Send,
    FieldElement<N>: AsBytes + Sync + Send,
    T: IsTranscript<N>,
{
    let expected_next = if next.is_some() { config.num_queries } else { 0 };
    if proof.current.len() != config.num_queries || proof.next.len() != expected_next {
        return Err(Error::QueryCountMismatch {
            expected: config.num_queries,
            got: proof.current.len(),
        });
    }
    let block = 1usize << config.log_folding;
    let num_leaves = domain.size() >> config.log_folding;
    for i in 0..config.num_queries {
        let q = transcript.sample_u64(num_leaves as u64) as usize;
        let current = &proof.current[i];
        if current.values.len() != block * chains || !verify_opening::<C>(current_root, q, current) {
            return Err(Error::OpeningRejected { query: i });
        }
        let successor = match next {
            Some((root, next_num_leaves, next_block)) => {
                let (leaf, slot) = leaf_and_slot(q, next_num_leaves);
                let opening = &proof.next[i];
                if opening.values.len() != next_block * chains
                    || !verify_opening::<N>(root, leaf, opening)
                {
                    return Err(Error::OpeningRejected { query: i });
                }
                Some((opening, slot, next_block))
            }
            None => None,
        };
        for j in 0..chains {
            let folded = fold_coset::<F, C, N>(&current.values[j * block..(j + 1) * block], domain, q, alphas)?;
            let claimed = match successor {
                Some((opening, slot, next_block)) => &opening.values[j * next_block + slot],
                None => &final_values[j],
            };
            if folded != *claimed {
                return Err(Error::FoldInconsistent { query: i });
            }
        }
    }
    Ok(())
}

/// Verifies `Σ_j Σ_x w_j(x)·f_j(x) = claim`. `weight_at(j, α)` is chain `j`'s
/// weight in closed form at the concatenation of every round's challenges.
#[allow(clippy::too_many_arguments)]
pub fn verify<F, E, T, W>(
    proof: &BatchProof<F, E>,
    root: &Commitment,
    chains: usize,
    weight_at: W,
    claim: FieldElement<E>,
    num_vars: usize,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    W: Fn(usize, &[FieldElement<E>]) -> Result<FieldElement<E>, Error>,
{
    let schedule = config.schedule(num_vars);
    if proof.rounds.len() != schedule.len() || proof.final_values.len() != chains {
        return Err(Error::RoundCountMismatch {
            expected: schedule.len(),
            got: proof.rounds.len(),
        });
    }
    let mut claim = claim;
    let mut alphas: Vec<FieldElement<E>> = Vec::with_capacity(num_vars);
    let mut current_root = *root;
    let mut current_domain = domain.clone();
    let mut ood: Vec<(FieldElement<E>, Vec<FieldElement<E>>, usize)> = Vec::new();

    for (r, (round, &k)) in proof.rounds.iter().zip(&schedule).enumerate() {
        if round.sumcheck.len() != k || (r == 0) != matches!(round.openings, RoundOpenings::Base(_)) {
            return Err(Error::RoundCountMismatch {
                expected: k,
                got: round.sumcheck.len(),
            });
        }
        check_grind(transcript, config.grind.folding, round.nonces.folding)?;
        let group = sumcheck::verify_rounds(&round.sumcheck, claim, 2, transcript)?;
        claim = group.expected_evaluation;

        let mut next_domain = current_domain.clone();
        for _ in 0..k {
            next_domain = next_domain.squared()?;
        }
        let bound = alphas.len() + k;
        let round_config = RoundConfig {
            num_queries: config.num_queries,
            log_folding: k,
        };
        let next = match (&round.next_root, schedule.get(r + 1)) {
            (Some(next_root), Some(&next_k)) => {
                if round.ood_values.len() != chains {
                    return Err(Error::QueryCountMismatch {
                        expected: chains,
                        got: round.ood_values.len(),
                    });
                }
                transcript.append_bytes(next_root);
                let z0: FieldElement<E> = transcript.sample_field_element();
                require_out_of_domain::<F, E>(&z0, &next_domain)?;
                let point = ood_point(&z0, num_vars - bound);
                for y0 in &round.ood_values {
                    transcript.append_field_element(y0);
                }
                check_grind(transcript, config.grind.ood, round.nonces.ood)?;
                let gamma: FieldElement<E> = transcript.sample_field_element();
                let mut scale = gamma.clone();
                for y0 in &round.ood_values {
                    claim += &scale * y0;
                    scale = &scale * &gamma;
                }
                ood.push((gamma, point, bound));
                Some((next_root, next_domain.size() >> next_k, 1usize << next_k))
            }
            (None, None) if round.ood_values.is_empty() => {
                for value in &proof.final_values {
                    transcript.append_field_element(value);
                }
                None
            }
            _ => {
                return Err(Error::RoundCountMismatch {
                    expected: schedule.len(),
                    got: r,
                });
            }
        };
        check_grind(transcript, config.grind.query, round.nonces.query)?;
        match &round.openings {
            RoundOpenings::Base(openings) => verify_openings::<F, F, E, T>(
                openings,
                chains,
                &current_root,
                next,
                &current_domain,
                &group.point,
                &round_config,
                &proof.final_values,
                transcript,
            )?,
            RoundOpenings::Extension(openings) => verify_openings::<F, E, E, T>(
                openings,
                chains,
                &current_root,
                next,
                &current_domain,
                &group.point,
                &round_config,
                &proof.final_values,
                transcript,
            )?,
        }
        if let Some((next_root, _, _)) = next {
            current_root = *next_root;
        }
        alphas.extend(group.point);
        current_domain = next_domain;
    }

    // The sumcheck's residual is Σ_j w_j(α)·f_j(α), and each chain's fully
    // folded value is its f_j(α).
    let mut expected = FieldElement::<E>::zero();
    for (j, value) in proof.final_values.iter().enumerate() {
        let mut weight = weight_at(j, &alphas)?;
        for (gamma, point, bound) in &ood {
            let mut scale = gamma.clone();
            for _ in 0..j {
                scale = &scale * gamma;
            }
            weight += scale * eq_eval(point, &alphas[*bound..])?;
        }
        expected += weight * value;
    }
    if expected != claim {
        return Err(Error::EvaluationMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eq::eq_mle;
    use crate::whir_chain::GrindBits;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn config() -> ChainConfig {
        ChainConfig {
            log_blowup: 2,
            log_folding: 2,
            num_queries: 5,
            grind: GrindBits::default(),
        }
    }

    fn poly(num_vars: usize, seed: u64) -> Mle<F> {
        Mle::new(
            (0..(1u64 << num_vars))
                .map(|i| FE::from(i.wrapping_add(seed).wrapping_mul(6364136223846793005) >> 11))
                .collect(),
        )
        .unwrap()
    }

    fn point(n: usize, seed: u64) -> Vec<FE> {
        (0..n).map(|i| FE::from(seed * 1009 + i as u64 * 31 + 7)).collect()
    }

    struct Case {
        polys: Vec<Mle<F>>,
        points: Vec<Vec<FE>>,
        claim: FE,
        num_vars: usize,
    }

    /// `K` polynomials, each weighted by `eq` at its own point.
    fn case(chains: usize, num_vars: usize) -> Case {
        let polys: Vec<Mle<F>> = (0..chains).map(|j| poly(num_vars, 3 + j as u64 * 17)).collect();
        let points: Vec<Vec<FE>> = (0..chains).map(|j| point(num_vars, 50 + j as u64)).collect();
        let claim = polys
            .iter()
            .zip(&points)
            .fold(FE::zero(), |acc, (p, z)| acc + p.evaluate(z).unwrap());
        Case {
            polys,
            points,
            claim,
            num_vars,
        }
    }

    fn run(c: &Case, tamper: impl FnOnce(&mut BatchProof<F, F>, &mut FE)) -> Result<(), Error> {
        let (commitment, domain) = commit(&c.polys, &config())?;
        let weights: Vec<Mle<F>> = c.points.iter().map(|z| eq_mle(z).unwrap()).collect();
        let mut t = DefaultTranscript::<F>::new(b"batch");
        let mut proof = prove::<F, F, _>(&c.polys, weights, &commitment, &domain, &config(), &mut t)?;
        let mut claim = c.claim.clone();
        tamper(&mut proof, &mut claim);
        let mut t = DefaultTranscript::<F>::new(b"batch");
        verify::<F, F, _, _>(
            &proof,
            &commitment.root(),
            c.polys.len(),
            |j, alphas| eq_eval(&c.points[j], alphas),
            claim,
            c.num_vars,
            &domain,
            &config(),
            &mut t,
        )
    }

    #[test]
    fn a_batch_verifies() {
        for chains in [1, 2, 3, 4, 5] {
            run(&case(chains, 6), |_, _| {}).unwrap_or_else(|e| panic!("K={chains}: {e:?}"));
        }
    }

    #[test]
    fn a_batch_of_odd_sizes_verifies() {
        run(&case(3, 5), |_, _| {}).unwrap();
        run(&case(7, 7), |_, _| {}).unwrap();
    }

    #[test]
    fn a_wrong_claim_is_rejected() {
        assert!(run(&case(3, 6), |_, claim| *claim += FE::one()).is_err());
    }

    #[test]
    fn a_wrong_final_value_is_rejected() {
        assert!(run(&case(3, 6), |proof, _| proof.final_values[1] += FE::one()).is_err());
    }

    #[test]
    fn swapped_ood_answers_are_rejected() {
        // Moving value between two chains keeps a plain sum; the per-chain
        // powers of the challenge do not let it through.
        assert!(run(&case(3, 6), |proof, _| {
            let values = &mut proof.rounds[0].ood_values;
            values[0] += FE::one();
            values[1] = &values[1] - FE::one();
        })
        .is_err());
    }

    #[test]
    fn a_tampered_shared_opening_is_rejected() {
        assert!(run(&case(4, 6), |proof, _| {
            if let RoundOpenings::Base(openings) = &mut proof.rounds[0].openings {
                openings.current[0].values[5] += FE::one();
            }
        })
        .is_err());
    }

    #[test]
    fn a_batch_pays_one_path_per_query() {
        // Four chains share a tree: the first round opens one leaf per query,
        // four blocks wide.
        let c = case(4, 6);
        let (commitment, domain) = commit(&c.polys, &config()).unwrap();
        let weights: Vec<Mle<F>> = c.points.iter().map(|z| eq_mle(z).unwrap()).collect();
        let mut t = DefaultTranscript::<F>::new(b"batch");
        let proof = prove::<F, F, _>(&c.polys, weights, &commitment, &domain, &config(), &mut t).unwrap();
        let RoundOpenings::Base(first) = &proof.rounds[0].openings else {
            panic!("the first round is base-field");
        };
        assert_eq!(first.current.len(), config().num_queries);
        assert_eq!(first.current[0].values.len(), 4 << config().log_folding);
    }
}

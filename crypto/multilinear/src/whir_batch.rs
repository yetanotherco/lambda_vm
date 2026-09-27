//! Several stacked polynomials of one group opened TOGETHER (S1): one Merkle
//! tree per round for all of them, one sumcheck, one set of challenges, one
//! set of grinds and one set of query positions.
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
//! power `γ^{j+1}` of the challenge so no chain's answer can hide behind
//! another's; and the queries are drawn once, after every commitment of the
//! round, so each chain is checked at positions its own commitments could not
//! anticipate. Everything is [`whir_chain`](crate::whir_chain)'s round, run on
//! `K` codewords at once — the order of the transcript is that chain's exactly.
//!
//! # The shared tree
//!
//! Leaf `q` of a round's tree is every chain's block `q` — the strided coset
//! `{q + s·N/2^k}` of its codeword — end to end, chain `j`'s at
//! `j·2^k..(j+1)·2^k` ([`WideCommitment`]). A query pays one authentication
//! path for all of them, and the leaf is hashed by the same backend a single
//! chain's is. Nothing is padded.
//!
//! ★ **A batch of one IS a chain**, byte for byte: the same transcript, a
//! one-block leaf that hashes the bytes a chain's does, and `γ^1 = γ`. That is
//! what lets a group of one — every bookend, DECODE's prepared stack — open the
//! same way under either format.
//!
//! # How a batch travels in a [`StackedProof`](crate::stacked_eval::StackedProof)
//!
//! As `K` [`ChainProof`]s, so the proof types — and today's bytes — do not
//! change. The first, the LEAD, carries everything the batch shares: the
//! sumcheck rounds, the successor roots, the nonces and the (wide) openings,
//! plus its own out-of-domain answers and final value. Every other, a FOLLOWER,
//! carries only its own out-of-domain answers and final value; every other
//! field is empty, and [`verify`] refuses a follower that says anything else
//! rather than ignoring it.
//!
//! # Soundness (WBATCH.md §1.2)
//!
//! The `K` codewords are one word over `F^K` of the interleaved code: its list
//! size is a single code's, mutual correlated agreement carries over with the
//! error scaled by `1/(1 − (K−1)/|F|)`, the out-of-domain step is a single
//! code's, and a query checks every chain at its position. The one term that
//! sees `K` is the `γ` combination of the `K` answers with the running claim —
//! a degree-`K` polynomial in `γ` per list element, `ℓ·K/|F|`.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::merkle_tree::{
    cap::{CappedRoot, embed_cap},
    proof::Proof,
    traits::IsMerkleTreeBackend,
};
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
    eq::{eq_eval, eq_mle},
    mle::Mle,
    poly::Composed,
    stacked_eval::{WeightShare, weight_table},
    sumcheck,
    whir::{Domain, encode, lift_coefficients},
    whir_chain::{
        ChainConfig, ChainProof, ChainRound, RoundNonces, RoundOpenings, Stacked, batch_weight,
        check_grind, first_value, fold_held, grind, ood_point, require_out_of_domain,
    },
    whir_commit::{
        Backend, Codeword, Commitment, CosetOpening, Tree, coset_of, fold_coset, leaf_and_slot,
        verify_opening_capped,
    },
    whir_hash::WhirHash,
    whir_round::{RoundCaps, RoundConfig, RoundProof, TreeCheck},
};

/// Codewords of one length, committed leaf by leaf together: leaf `q` is every
/// codeword's block `q`, end to end.
pub struct WideCommitment<F: IsField + 'static, H: WhirHash>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    tree: Tree<F, H>,
    codewords: Vec<Codeword<F>>,
    log_folding: usize,
    log_domain_size: usize,
}

impl<F: IsField + 'static, H: WhirHash> std::fmt::Debug for WideCommitment<F, H>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WideCommitment")
            .field("root", &self.root())
            .field("width", &self.width())
            .field("leaves", &self.num_leaves())
            .field("log_folding", &self.log_folding)
            .finish()
    }
}

impl<F: IsField + 'static, H: WhirHash> WideCommitment<F, H>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    /// Groups every codeword into fold blocks and commits leaf `q` as the
    /// blocks `q` of all of them, in order.
    ///
    /// At one codeword this is [`CodewordCommitment::from_codeword_on_host`]'s
    /// tree exactly — the same leaves, so the same root.
    ///
    /// [`CodewordCommitment::from_codeword_on_host`]: crate::whir_commit::CodewordCommitment::from_codeword_on_host
    pub fn from_codewords(codewords: Vec<Codeword<F>>, log_folding: usize) -> Result<Self, Error> {
        let len = codewords.first().ok_or(Error::EmptyPolynomial)?.len();
        if !len.is_power_of_two() {
            return Err(Error::NotPowerOfTwo(len));
        }
        if codewords.iter().any(|c| c.len() != len) {
            return Err(Error::BatchShape {
                what: "codeword lengths",
            });
        }
        let log_domain_size = len.trailing_zeros() as usize;
        if log_folding > log_domain_size {
            return Err(Error::ColumnTallerThanStack {
                column_vars: log_folding,
                n_stack: log_domain_size,
            });
        }
        let values: Vec<&[FieldElement<F>]> = codewords
            .iter()
            .map(Codeword::host)
            .collect::<Option<_>>()
            .ok_or(Error::DeviceFailed {
                stage: "wide commitment",
            })?;
        let num_leaves = len >> log_folding;
        let block = 1usize << log_folding;
        let width = block * values.len();
        // One reused buffer per worker: a leaf is every codeword's strided
        // block, so it has to be gathered somewhere before it is hashed.
        let hash_leaf = |buffer: &mut Vec<FieldElement<F>>, q: usize| {
            buffer.clear();
            for codeword in &values {
                buffer.extend((0..block).map(|t| codeword[q + t * num_leaves].clone()));
            }
            Backend::<F, H>::hash_data(buffer)
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
        let tree = Tree::<F, H>::build_from_hashed_leaves(hashed).ok_or(Error::EmptyPolynomial)?;
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

    /// How many codewords share the tree.
    pub fn width(&self) -> usize {
        self.codewords.len()
    }

    pub fn num_leaves(&self) -> usize {
        1usize << (self.log_domain_size - self.log_folding)
    }

    /// Siblings on a full authentication path: `log2(num_leaves)`.
    pub fn depth(&self) -> usize {
        self.log_domain_size - self.log_folding
    }

    pub fn log_folding(&self) -> usize {
        self.log_folding
    }

    pub fn codewords(&self) -> &[Codeword<F>] {
        &self.codewords
    }

    /// Opens every leaf a round asks for, under a Merkle cap of height
    /// `cap_height` — the owner-path encoding of
    /// [`CodewordCommitment::open_many_capped`], applied to the shared tree.
    ///
    /// [`CodewordCommitment::open_many_capped`]: crate::whir_commit::CodewordCommitment::open_many_capped
    pub fn open_many_capped(
        &self,
        indices: &[usize],
        cap_height: usize,
        owner: bool,
    ) -> Result<Vec<CosetOpening<F>>, Error> {
        let num_leaves = self.num_leaves();
        // One call, one tree walk: the same count a chain's `open_many` keeps,
        // so the rebuild identity (`2R − 1` per chain) holds per batch.
        crate::whir_split::bump(&crate::whir_split::REBUILD_CALLS);
        let __wq_tree = crate::whir_split::mark();
        let proofs = self.paths_capped(indices, cap_height, owner)?;
        crate::whir_split::add(&crate::whir_split::TREE_REBUILD, __wq_tree);

        let __wq_gather = crate::whir_split::mark();
        let mut blocks: Vec<Vec<FieldElement<F>>> =
            vec![Vec::with_capacity(self.width() << self.log_folding); indices.len()];
        for codeword in &self.codewords {
            let values = codeword.host().ok_or(Error::DeviceFailed {
                stage: "wide opening",
            })?;
            for (block, index) in blocks.iter_mut().zip(indices) {
                if *index >= num_leaves {
                    return Err(Error::QueryOutOfRange {
                        index: *index,
                        bound: num_leaves,
                    });
                }
                block.extend(
                    coset_of(*index, self.log_domain_size, self.log_folding)
                        .into_iter()
                        .map(|p| values[p].clone()),
                );
            }
        }
        crate::whir_split::add(&crate::whir_split::COSET_GATHER, __wq_gather);

        let __wq_assemble = crate::whir_split::mark();
        let openings = blocks
            .into_iter()
            .zip(proofs)
            .map(|(values, proof)| CosetOpening { values, proof })
            .collect();
        crate::whir_split::add(&crate::whir_split::OPEN_ASSEMBLE, __wq_assemble);
        Ok(openings)
    }

    /// One authentication path per index, cut to the cap and, for the owner,
    /// with the cap appended to the first.
    fn paths_capped(
        &self,
        indices: &[usize],
        cap_height: usize,
        owner: bool,
    ) -> Result<Vec<Proof<Commitment>>, Error> {
        let num_leaves = self.num_leaves();
        let mut proofs: Vec<Proof<Commitment>> = indices
            .iter()
            .map(|index| {
                self.tree
                    .get_proof_by_pos(*index)
                    .ok_or(Error::QueryOutOfRange {
                        index: *index,
                        bound: num_leaves,
                    })
            })
            .collect::<Result<_, _>>()?;
        if cap_height == 0 {
            return Ok(proofs);
        }
        let depth = self.depth();
        if cap_height > depth {
            return Err(Error::CapEmbedFailed {
                reason: "cap taller than the tree",
            });
        }
        let embed_failed = |_: crypto::merkle_tree::cap::CapError| Error::CapEmbedFailed {
            reason: "path or cap of the wrong length",
        };
        if owner {
            let cap = self.tree.cap(cap_height).ok_or(Error::CapEmbedFailed {
                reason: "the host tree has no cap at this height",
            })?;
            let mut refs: Vec<&mut Vec<Commitment>> =
                proofs.iter_mut().map(|p| &mut p.merkle_path).collect();
            embed_cap(&mut refs, depth, &cap).map_err(embed_failed)?;
        } else {
            for proof in &mut proofs {
                proof
                    .truncate_to_cap(depth, cap_height)
                    .map_err(embed_failed)?;
            }
        }
        Ok(proofs)
    }
}

/// Commits a batch's polynomials — all of `n_stack` variables — as one wide
/// tree, blocked for the first round's fold.
///
/// The codewords stay in the base field, as a chain's do.
pub fn commit<F, H>(
    sources: &[Stacked<'_, F>],
    config: &ChainConfig,
) -> Result<(WideCommitment<F, H>, Domain<F>), Error>
where
    F: IsFFTField + IsPrimeField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let num_vars = sources.first().ok_or(Error::EmptyPolynomial)?.num_vars();
    if let Some(other) = sources.iter().find(|s| s.num_vars() != num_vars) {
        return Err(Error::VariableCountMismatch {
            expected: num_vars,
            got: other.num_vars(),
        });
    }
    let first = config.schedule(num_vars).first().copied().unwrap_or(0);
    let domain = Domain::<F>::new(num_vars + config.log_blowup)?;
    let encode_one = |source: &Stacked<'_, F>| -> Result<Codeword<F>, Error> {
        Ok(Codeword::Host(encode::<F, F>(
            &lift_coefficients(&source.assemble()?),
            &domain,
        )?))
    };
    #[cfg(feature = "parallel")]
    let codewords = sources
        .par_iter()
        .map(encode_one)
        .collect::<Result<Vec<_>, Error>>()?;
    #[cfg(not(feature = "parallel"))]
    let codewords = sources
        .iter()
        .map(encode_one)
        .collect::<Result<Vec<_>, Error>>()?;
    Ok((WideCommitment::from_codewords(codewords, first)?, domain))
}

/// What a round opens, and where it is: only the first round's is base-field.
enum Held<'a, F: IsField + 'static, E: IsField + 'static, H: WhirHash>
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    Base(&'a WideCommitment<F, H>),
    Extension(WideCommitment<E, H>),
}

/// `Σ_j w_j·f_j` over `[w_0, f_0, w_1, f_1, ..]`: the shape every round's
/// sumcheck runs over. At one pair it is a chain's `w·f`.
fn sum_of_products<E: IsField>(values: &[FieldElement<E>]) -> FieldElement<E> {
    values
        .chunks_exact(2)
        .fold(FieldElement::<E>::zero(), |acc, pair| {
            acc + &pair[0] * &pair[1]
        })
}

/// The powers `γ^1 ..= γ^K` chain `j`'s out-of-domain answer and `eq` enter
/// the claim and the weight at — chain `j` at `γ^{j+1}`. DISTINCT powers are
/// what keep one chain's answer from paying for another's; both sides take
/// them from here.
pub(crate) fn answer_scales<E: IsField>(
    gamma: &FieldElement<E>,
    chains: usize,
) -> Vec<FieldElement<E>> {
    let mut scale = gamma.clone();
    (0..chains)
        .map(|_| {
            let current = scale.clone();
            scale = &scale * gamma;
            current
        })
        .collect()
}

/// The openings a follower carries in round `r`: none, in the variant the
/// round's position fixes (`Base` in the first round only).
fn empty_openings<F: IsField, E: IsField>(r: usize) -> RoundOpenings<F, E> {
    if r == 0 {
        RoundOpenings::Base(RoundProof {
            current: Vec::new(),
            next: Vec::new(),
        })
    } else {
        RoundOpenings::Extension(RoundProof {
            current: Vec::new(),
            next: Vec::new(),
        })
    }
}

/// Proves `Σ_j Σ_x w_j(x)·f_j(x) = Σ_j claim_j` for a batch: `sources[j]` is
/// polynomial `j` and `shares[j]` its weight, against the wide tree [`commit`]
/// built. Returns one [`ChainProof`] per polynomial, the lead first — see the
/// module header for what each carries.
#[allow(clippy::too_many_arguments)]
pub fn prove<F, E, T, H>(
    sources: &[Stacked<'_, F>],
    shares: &[Vec<WeightShare<'_, E>>],
    n_stack: usize,
    commitment: &WideCommitment<F, H>,
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<Vec<ChainProof<F, E>>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    let chains = sources.len();
    if chains == 0 || shares.len() != chains || commitment.width() != chains {
        return Err(Error::BatchShape {
            what: "chain count",
        });
    }
    if let Some(other) = sources.iter().find(|s| s.num_vars() != n_stack) {
        return Err(Error::VariableCountMismatch {
            expected: n_stack,
            got: other.num_vars(),
        });
    }
    // A batch is one chain to the counters: `R` rounds and `2R − 1` openings.
    crate::whir_split::bump(&crate::whir_split::CHAIN_COUNT);
    let schedule = config.schedule(n_stack);
    let caps = config.tree_caps(n_stack);

    // [w_0, f_0, w_1, f_1, ..]: the sumcheck's factors, in the extension. The
    // messages are lifted here; the codewords are not, which is where the size
    // is.
    let mut factors: Vec<Mle<E>> = Vec::with_capacity(2 * chains);
    for (source, share) in sources.iter().zip(shares) {
        factors.push(weight_table(share, n_stack)?);
        let message = source.assemble()?;
        factors.push(Mle::new(
            message
                .evals()
                .iter()
                .map(|v| v.clone().to_extension::<E>())
                .collect(),
        )?);
    }

    let mut current = Held::<F, E, H>::Base(commitment);
    let mut current_domain = domain.clone();
    let mut shared: Vec<ChainRound<F, E>> = Vec::with_capacity(schedule.len());
    // Round by round, every chain's out-of-domain answer; `None` on the last.
    let mut answers: Vec<Option<Vec<FieldElement<E>>>> = Vec::with_capacity(schedule.len());
    let mut finals = vec![FieldElement::<E>::zero(); chains];

    for (r, &k) in schedule.iter().enumerate() {
        let __wc_round = crate::whir_split::mark();
        crate::whir_split::bump(&crate::whir_split::ROUND_COUNT);
        let __wc_g = crate::whir_split::mark();
        let mut nonces = RoundNonces {
            folding: grind::<E, T, H>(transcript, config.grind.folding)?,
            ..RoundNonces::default()
        };
        crate::whir_split::add(&crate::whir_split::GRIND, __wc_g);

        let __wc_sc = crate::whir_split::mark();
        let mut poly = Composed::new(core::mem::take(&mut factors), sum_of_products::<E>, 2)?;
        let (sumcheck_rounds, alphas) = sumcheck::prove_rounds(&mut poly, k, transcript)?;
        factors = poly.into_polys();
        crate::whir_split::add(&crate::whir_split::SUMCHECK, __wc_sc);

        // Every chain folds by the same challenges, each where it is.
        let __wc_fold = crate::whir_split::mark();
        let mut folded = Vec::with_capacity(chains);
        let mut folded_domain = current_domain.clone();
        for j in 0..chains {
            let (codeword, d) = match &current {
                Held::Base(held) => {
                    fold_held::<F, F, E>(&held.codewords()[j], &current_domain, &alphas)?
                }
                Held::Extension(held) => {
                    fold_held::<F, E, E>(&held.codewords()[j], &current_domain, &alphas)?
                }
            };
            folded.push(codeword);
            folded_domain = d;
        }
        crate::whir_split::add(&crate::whir_split::FOLD, __wc_fold);

        // The successors are committed before the queries are drawn, in ONE
        // tree, so none of them can be chosen to match the positions.
        let __wc_cf = crate::whir_split::mark();
        let next = match schedule.get(r + 1) {
            Some(&next_k) => {
                let next = WideCommitment::<E, H>::from_codewords(folded, next_k)?;
                transcript.append_bytes(&next.root());
                Some(next)
            }
            None => {
                for (j, codeword) in folded.iter().enumerate() {
                    finals[j] = first_value::<E>(codeword)?;
                }
                for value in &finals {
                    transcript.append_field_element(value);
                }
                None
            }
        };
        crate::whir_split::add(&crate::whir_split::COMMIT_FOLDED, __wc_cf);
        let next_root = next.as_ref().map(|c| c.root());

        // One out-of-domain point, every chain's answer, then ONE challenge
        // whose powers keep the answers apart. Two windows around the grind, as
        // a chain's, so the slots partition the round.
        if next.is_some() {
            let __wc_ood_a = crate::whir_split::mark();
            let z0: FieldElement<E> = transcript.sample_field_element();
            require_out_of_domain::<F, E>(&z0, &folded_domain)?;
            let point = ood_point(&z0, factors[1].num_vars());
            let ys: Vec<FieldElement<E>> = (0..chains)
                .map(|j| factors[2 * j + 1].evaluate(&point))
                .collect::<Result<_, _>>()?;
            for y0 in &ys {
                transcript.append_field_element(y0);
            }
            crate::whir_split::add(&crate::whir_split::OOD, __wc_ood_a);

            let __wc_g2 = crate::whir_split::mark();
            nonces.ood = grind::<E, T, H>(transcript, config.grind.ood)?;
            crate::whir_split::add(&crate::whir_split::GRIND, __wc_g2);

            let __wc_ood_b = crate::whir_split::mark();
            let gamma: FieldElement<E> = transcript.sample_field_element();
            let eq = eq_mle(&point)?;
            for (j, scale) in answer_scales(&gamma, chains).iter().enumerate() {
                factors[2 * j] = batch_weight(&factors[2 * j], &eq, scale)?;
            }
            crate::whir_split::add(&crate::whir_split::OOD, __wc_ood_b);
            answers.push(Some(ys));
        } else {
            answers.push(None);
        }

        let __wc_g3 = crate::whir_split::mark();
        nonces.query = grind::<E, T, H>(transcript, config.grind.query)?;
        crate::whir_split::add(&crate::whir_split::GRIND, __wc_g3);
        let round_config = RoundConfig {
            num_queries: config.num_queries,
            log_folding: k,
        };
        let round_caps = RoundCaps {
            current: caps[r],
            current_owner: r == 0,
            next: caps.get(r + 1).copied().unwrap_or(0),
        };
        let __wc_q = crate::whir_split::mark();
        let openings = match &current {
            Held::Base(held) => RoundOpenings::Base(open_round(
                *held,
                next.as_ref(),
                &round_config,
                round_caps,
                transcript,
            )?),
            Held::Extension(held) => RoundOpenings::Extension(open_round(
                held,
                next.as_ref(),
                &round_config,
                round_caps,
                transcript,
            )?),
        };
        crate::whir_split::add(&crate::whir_split::QUERIES, __wc_q);

        shared.push(ChainRound {
            sumcheck: sumcheck_rounds,
            next_root,
            ood_value: None,
            nonces,
            openings,
        });
        crate::whir_split::add(&crate::whir_split::ROUND, __wc_round);
        if let Some(next) = next {
            current = Held::Extension(next);
        }
        current_domain = folded_domain;
    }

    // The lead takes the shared rounds; every chain takes its own answers.
    let mut proofs: Vec<ChainProof<F, E>> = finals
        .into_iter()
        .map(|final_value| ChainProof {
            rounds: Vec::with_capacity(schedule.len()),
            final_value,
        })
        .collect();
    for (r, (round, ys)) in shared.into_iter().zip(answers).enumerate() {
        let answer = |j: usize| ys.as_ref().map(|ys| ys[j].clone());
        for (j, follower) in proofs.iter_mut().enumerate().skip(1) {
            follower.rounds.push(ChainRound {
                sumcheck: Vec::new(),
                next_root: None,
                ood_value: answer(j),
                nonces: RoundNonces::default(),
                openings: empty_openings(r),
            });
        }
        proofs[0].rounds.push(ChainRound {
            ood_value: answer(0),
            ..round
        });
    }
    Ok(proofs)
}

/// One round's openings: the queries are drawn as a chain's are, and each
/// opens the current tree's leaf and, but for the last round, the successor's
/// leaf holding the folded value.
pub(crate) fn open_round<C, N, T, H>(
    current: &WideCommitment<C, H>,
    next: Option<&WideCommitment<N, H>>,
    config: &RoundConfig,
    caps: RoundCaps,
    transcript: &mut T,
) -> Result<RoundProof<C, N>, Error>
where
    C: IsField + 'static,
    N: IsField + 'static,
    FieldElement<C>: AsBytes + Sync + Send,
    FieldElement<N>: AsBytes + Sync + Send,
    T: IsTranscript<N>,
    H: WhirHash,
{
    let __wq_sample = crate::whir_split::mark();
    let queries: Vec<usize> = (0..config.num_queries)
        .map(|_| transcript.sample_u64(current.num_leaves() as u64) as usize)
        .collect();
    let leaves: Option<Vec<usize>> = next.map(|next| {
        queries
            .iter()
            .map(|q| leaf_and_slot(*q, next.num_leaves()).0)
            .collect()
    });
    crate::whir_split::add(&crate::whir_split::QUERY_SAMPLE, __wq_sample);
    Ok(RoundProof {
        current: current.open_many_capped(&queries, caps.current, caps.current_owner)?,
        next: match (next, leaves) {
            (Some(next), Some(leaves)) => next.open_many_capped(&leaves, caps.next, true)?,
            _ => Vec::new(),
        },
    })
}

/// A follower's round must carry nothing but its own answer.
fn check_follower<F: IsField, E: IsField>(
    proof: &ChainProof<F, E>,
    rounds: usize,
) -> Result<(), Error> {
    if proof.rounds.len() != rounds {
        return Err(Error::BatchShape {
            what: "follower round count",
        });
    }
    for (r, round) in proof.rounds.iter().enumerate() {
        let empty = match &round.openings {
            RoundOpenings::Base(p) => r == 0 && p.current.is_empty() && p.next.is_empty(),
            RoundOpenings::Extension(p) => r > 0 && p.current.is_empty() && p.next.is_empty(),
        };
        if !empty
            || !round.sumcheck.is_empty()
            || round.next_root.is_some()
            || round.nonces != RoundNonces::default()
        {
            return Err(Error::BatchShape {
                what: "follower (a field only the lead carries)",
            });
        }
    }
    Ok(())
}

/// Verifies `Σ_j Σ_x w_j(x)·f_j(x) = claim` for a batch against its root.
///
/// `proofs` is the batch's `K` chain proofs, the lead first; `weight_at(j, α)`
/// is chain `j`'s weight in closed form at the concatenation of every round's
/// challenges. `K` is `proofs.len()`, and the caller has already checked it
/// against the split its format and layout fix.
#[allow(clippy::too_many_arguments)]
pub fn verify<'a, F, E, T, W, H>(
    proofs: &'a [ChainProof<F, E>],
    root: &'a Commitment,
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
    H: WhirHash,
{
    let (lead, followers) = proofs.split_first().ok_or(Error::BatchShape {
        what: "chain count",
    })?;
    let chains = proofs.len();
    let schedule = config.schedule(num_vars);
    if lead.rounds.len() != schedule.len() {
        return Err(Error::RoundCountMismatch {
            expected: schedule.len(),
            got: lead.rounds.len(),
        });
    }
    for follower in followers {
        check_follower(follower, schedule.len())?;
    }

    let mut claim = claim;
    let mut alphas: Vec<FieldElement<E>> = Vec::with_capacity(num_vars);
    let caps = config.tree_caps(num_vars);
    let mut current: TreeCheck<'a> = TreeCheck::Owner {
        root,
        cap_height: caps[0],
    };
    let mut current_domain = domain.clone();
    let mut ood: Vec<OodTerm<E>> = Vec::new();

    for (r, (round, &k)) in lead.rounds.iter().zip(&schedule).enumerate() {
        if round.sumcheck.len() != k {
            return Err(Error::RoundCountMismatch {
                expected: k,
                got: round.sumcheck.len(),
            });
        }
        if (r == 0) != matches!(round.openings, RoundOpenings::Base(_)) {
            return Err(Error::RoundCountMismatch {
                expected: 0,
                got: r,
            });
        }
        check_grind::<E, T, H>(transcript, config.grind.folding, round.nonces.folding)?;
        let group = sumcheck::verify_rounds(&round.sumcheck, claim, 2, transcript)?;
        claim = group.expected_evaluation;

        let mut next_domain = current_domain.clone();
        for _ in 0..k {
            next_domain = next_domain.squared()?;
        }
        let bound = alphas.len() + k;
        // Every chain's answer this round, in order; all present or all absent.
        let answers: Vec<Option<&FieldElement<E>>> = proofs
            .iter()
            .map(|p| p.rounds[r].ood_value.as_ref())
            .collect();

        match (&round.next_root, schedule.get(r + 1)) {
            (Some(next_root), Some(&next_k)) if answers.iter().all(Option::is_some) => {
                transcript.append_bytes(next_root);

                let z0: FieldElement<E> = transcript.sample_field_element();
                require_out_of_domain::<F, E>(&z0, &next_domain)?;
                let point = ood_point(&z0, num_vars - bound);
                for y0 in answers.iter().flatten() {
                    transcript.append_field_element(y0);
                }

                check_grind::<E, T, H>(transcript, config.grind.ood, round.nonces.ood)?;
                let gamma: FieldElement<E> = transcript.sample_field_element();
                let scales = answer_scales(&gamma, chains);
                for (y0, scale) in answers.iter().flatten().zip(&scales) {
                    claim += scale * *y0;
                }
                ood.push(OodTerm {
                    scales,
                    point,
                    bound,
                });

                check_grind::<E, T, H>(transcript, config.grind.query, round.nonces.query)?;
                let next = NextTree {
                    root: next_root,
                    num_leaves: next_domain.size() >> next_k,
                    log_folding: next_k,
                    cap_height: caps[r + 1],
                };
                let next_check = match &round.openings {
                    RoundOpenings::Base(openings) => verify_round::<F, F, E, T, H>(
                        openings,
                        chains,
                        current,
                        next,
                        &current_domain,
                        &group.point,
                        config.num_queries,
                        transcript,
                    )?,
                    RoundOpenings::Extension(openings) => verify_round::<F, E, E, T, H>(
                        openings,
                        chains,
                        current,
                        next,
                        &current_domain,
                        &group.point,
                        config.num_queries,
                        transcript,
                    )?,
                };
                current = TreeCheck::Checked(next_check);
            }
            (None, None) if answers.iter().all(Option::is_none) => {
                for proof in proofs {
                    transcript.append_field_element(&proof.final_value);
                }
                check_grind::<E, T, H>(transcript, config.grind.query, round.nonces.query)?;
                let finals: Vec<&FieldElement<E>> = proofs.iter().map(|p| &p.final_value).collect();
                match &round.openings {
                    RoundOpenings::Base(openings) => verify_final::<F, F, E, T, H>(
                        openings,
                        current,
                        &current_domain,
                        &group.point,
                        config.num_queries,
                        &finals,
                        transcript,
                    )?,
                    RoundOpenings::Extension(openings) => verify_final::<F, E, E, T, H>(
                        openings,
                        current,
                        &current_domain,
                        &group.point,
                        config.num_queries,
                        &finals,
                        transcript,
                    )?,
                }
            }
            // A successor or an answer where the schedule ends, or one missing
            // where it does not — for the lead or for any follower.
            _ => {
                return Err(Error::RoundCountMismatch {
                    expected: schedule.len(),
                    got: r,
                });
            }
        }

        alphas.extend(group.point);
        current_domain = next_domain;
    }

    // The wire: the sumcheck's residual is `Σ_j w_j(α)·f_j(α)`, and each chain's
    // fully folded value is its `f_j(α)`. Chain `j`'s weight is its own plus
    // `γ_r^{j+1}·eq` of every round's out-of-domain point.
    let mut total = FieldElement::<E>::zero();
    let mut degenerate = true;
    for (j, proof) in proofs.iter().enumerate() {
        let mut weight = weight_at(j, &alphas)?;
        for term in &ood {
            weight += &term.scales[j] * eq_eval(&term.point, &alphas[term.bound..])?;
        }
        if weight != FieldElement::<E>::zero() {
            degenerate = false;
        }
        total += weight * &proof.final_value;
    }
    // A chain's `claim·w⁻¹` refuses a zero weight; at one chain this is that
    // refusal, and at more it refuses only when no final value is weighed at all.
    if degenerate {
        return Err(Error::DegenerateEvaluationPoint);
    }
    if total != claim {
        return Err(Error::EvaluationMismatch);
    }
    Ok(())
}

/// A round's out-of-domain point as it enters the weights: chain `j` gains
/// `scales[j]·eq(point, α[bound..])`, the challenges after the `bound` ones
/// already fixed when the point was drawn.
struct OodTerm<E: IsField> {
    scales: Vec<FieldElement<E>>,
    point: Vec<FieldElement<E>>,
    bound: usize,
}

/// What the verifier knows about a round's successor tree before it opens it.
#[derive(Clone, Copy)]
pub(crate) struct NextTree<'a> {
    pub(crate) root: &'a Commitment,
    pub(crate) num_leaves: usize,
    pub(crate) log_folding: usize,
    pub(crate) cap_height: usize,
}

/// A round's wide openings against its two trees: every chain's block folds
/// onto its value in the successor, at positions drawn after every commitment.
///
/// Mirrors [`crate::whir_round::verify`] step for step — the count guard, both
/// trees' checks from their first openings, the query draw, then per query the
/// current opening, the successor opening and the folds — so a batch of one
/// refuses exactly what a chain refuses. What it adds is the WIDTH: a leaf is
/// `chains` blocks, checked before anything is sliced out of it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_round<'a, F, C, N, T, H>(
    proof: &'a RoundProof<C, N>,
    chains: usize,
    current: TreeCheck<'a>,
    next: NextTree<'a>,
    domain: &Domain<F>,
    alphas: &[FieldElement<N>],
    num_queries: usize,
    transcript: &mut T,
) -> Result<CappedRoot<'a, Commitment>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<C> + IsSubFieldOf<N>,
    C: IsField + IsSubFieldOf<N> + 'static,
    N: IsField + 'static,
    FieldElement<C>: AsBytes + Sync + Send,
    FieldElement<N>: AsBytes + Sync + Send,
    T: IsTranscript<N>,
    H: WhirHash,
{
    let k = alphas.len();
    if proof.current.len() != num_queries || proof.next.len() != num_queries {
        return Err(Error::QueryCountMismatch {
            expected: num_queries,
            got: proof.current.len().min(proof.next.len()),
        });
    }
    let num_leaves = domain.size() >> k;
    let current_depth = num_leaves.trailing_zeros() as usize;
    if !next.num_leaves.is_power_of_two() {
        return Err(Error::NotPowerOfTwo(next.num_leaves));
    }
    let next_depth = next.num_leaves.trailing_zeros() as usize;

    let (current_check, current_first, next_check, next_first) =
        match (proof.current.first(), proof.next.first()) {
            (Some(cur), Some(nxt)) => {
                let (current_check, current_first) = current.open::<C, H>(current_depth, cur)?;
                let (next_check, next_first) = TreeCheck::Owner {
                    root: next.root,
                    cap_height: next.cap_height,
                }
                .open::<N, H>(next_depth, nxt)?;
                (current_check, current_first, next_check, next_first)
            }
            _ => {
                if next.cap_height != 0 {
                    return Err(Error::CapRejected);
                }
                return Ok(CappedRoot::uncapped(next.root, next_depth));
            }
        };

    let queries: Vec<usize> = (0..num_queries)
        .map(|_| transcript.sample_u64(num_leaves as u64) as usize)
        .collect();
    let block = 1usize << k;
    let next_block = 1usize << next.log_folding;

    for (i, (&q, (cur, nxt))) in queries
        .iter()
        .zip(proof.current.iter().zip(&proof.next))
        .enumerate()
    {
        let (cur_siblings, nxt_siblings) = if i == 0 {
            (current_first, next_first)
        } else {
            (
                cur.proof.merkle_path.as_slice(),
                nxt.proof.merkle_path.as_slice(),
            )
        };
        if cur.values.len() != chains * block
            || !verify_opening_capped::<C, H>(&current_check, q, cur, cur_siblings)
        {
            return Err(Error::OpeningRejected { query: i });
        }
        let (leaf, slot) = leaf_and_slot(q, next.num_leaves);
        if nxt.values.len() != chains * next_block
            || !verify_opening_capped::<N, H>(&next_check, leaf, nxt, nxt_siblings)
        {
            return Err(Error::OpeningRejected { query: i });
        }
        if slot >= next_block {
            return Err(Error::QueryOutOfRange {
                index: slot,
                bound: next_block,
            });
        }
        for (current_block, next_values) in cur
            .values
            .chunks_exact(block)
            .zip(nxt.values.chunks_exact(next_block))
        {
            if fold_coset::<F, C, N>(current_block, domain, q, alphas)? != next_values[slot] {
                return Err(Error::FoldInconsistent { query: i });
            }
        }
    }

    Ok(next_check)
}

/// The last round: every chain's queried block must fold to its own final
/// value. Mirrors the chain's `verify_final`, with the width checked first.
pub(crate) fn verify_final<'a, F, C, N, T, H>(
    proof: &'a RoundProof<C, N>,
    current: TreeCheck<'a>,
    domain: &Domain<F>,
    alphas: &[FieldElement<N>],
    num_queries: usize,
    finals: &[&FieldElement<N>],
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<C> + IsSubFieldOf<N>,
    C: IsField + IsSubFieldOf<N> + 'static,
    N: IsField + 'static,
    FieldElement<C>: AsBytes + Sync + Send,
    T: IsTranscript<N>,
    H: WhirHash,
{
    if proof.current.len() != num_queries || !proof.next.is_empty() {
        return Err(Error::QueryCountMismatch {
            expected: num_queries,
            got: proof.current.len(),
        });
    }
    let block = 1usize << alphas.len();
    let num_leaves = domain.size() >> alphas.len();
    let depth = num_leaves.trailing_zeros() as usize;
    let Some(first) = proof.current.first() else {
        return Ok(());
    };
    let (check, first_siblings) = current.open::<C, H>(depth, first)?;

    for (i, opening) in proof.current.iter().enumerate() {
        let q = transcript.sample_u64(num_leaves as u64) as usize;
        let siblings = if i == 0 {
            first_siblings
        } else {
            opening.proof.merkle_path.as_slice()
        };
        if opening.values.len() != finals.len() * block
            || !verify_opening_capped::<C, H>(&check, q, opening, siblings)
        {
            return Err(Error::OpeningRejected { query: i });
        }
        for (values, final_value) in opening.values.chunks_exact(block).zip(finals) {
            if fold_coset::<F, C, N>(values, domain, q, alphas)? != **final_value {
                return Err(Error::FoldInconsistent { query: i });
            }
        }
    }
    Ok(())
}

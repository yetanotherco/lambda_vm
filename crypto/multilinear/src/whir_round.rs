//! One WHIR round: sample queries, open the current codeword's blocks, fold them
//! locally, and check they match the committed successor.
//!
//! The successor is committed before the queries are drawn, so it cannot be
//! chosen to match. Consistency holds only where the queries land.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::merkle_tree::cap::CappedRoot;
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::AsBytes,
};

use crate::{
    Error,
    whir::Domain,
    whir_commit::{
        CodewordCommitment, Commitment, CosetOpening, fold_coset, leaf_and_slot,
        verify_opening_capped,
    },
    whir_hash::WhirHash,
};

type Backend<F, H> = <H as WhirHash>::Backend<F>;

/// How hard a round is to cheat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundConfig {
    /// Positions checked. Each one an inconsistent prover must get lucky on.
    pub num_queries: usize,
    /// `k`: the codeword folds by `2^k` and blocks hold `2^k` values.
    pub log_folding: usize,
}

/// How a tree's openings are authenticated in a round.
///
/// A tree is authenticated ONCE: tree 0 by the
/// cap its first opening in round 0 carries, and tree `t ≥ 1` by the cap its
/// first opening as round `t − 1`'s SUCCESSOR carries. Round `t` then opens
/// tree `t` as its current tree against that stored check, and never re-reads
/// a cap from its own first opening.
#[derive(Clone, Copy, Debug)]
pub enum TreeCheck<'a> {
    /// Authenticated in an earlier round.
    Checked(CappedRoot<'a, Commitment>),
    /// Owned by this round: its first opening carries its cap of height
    /// `cap_height` (none at 0).
    Owner {
        root: &'a Commitment,
        cap_height: usize,
    },
}

impl<'a> TreeCheck<'a> {
    /// The tree's check and the siblings of `first`, the tree's first opening
    /// in this round.
    ///
    /// An [`Owner`](Self::Owner) tree's cap is split off `first`'s path and
    /// authenticated against the root here — once, before any opening of the
    /// tree is checked against it. At `cap_height = 0` there is no cap: the
    /// whole path is siblings, and the per-query check enforces its length.
    /// A [`Checked`](Self::Checked) tree must have the depth this round
    /// derives, and `first` carries no cap: its whole path is siblings.
    pub(crate) fn open<C, H>(
        self,
        depth: usize,
        first: &'a CosetOpening<C>,
    ) -> Result<(CappedRoot<'a, Commitment>, &'a [Commitment]), Error>
    where
        C: IsField + 'static,
        FieldElement<C>: AsBytes + Sync + Send,
        H: WhirHash,
    {
        let path = first.proof.merkle_path.as_slice();
        match self {
            TreeCheck::Checked(check) => {
                if check.depth() != depth {
                    return Err(Error::CapRejected);
                }
                Ok((check, path))
            }
            TreeCheck::Owner {
                root,
                cap_height: 0,
            } => Ok((CappedRoot::uncapped(root, depth), path)),
            TreeCheck::Owner { root, cap_height } => {
                CappedRoot::from_owner::<Backend<C, H>>(root, path, depth, cap_height)
                    .ok_or(Error::CapRejected)
            }
        }
    }
}

/// What the verifier already knows about the two codewords: how the current
/// tree is authenticated, the successor's root, how the successor was blocked,
/// and its cap height.
#[derive(Clone, Copy, Debug)]
pub struct RoundCommitments<'a> {
    pub current: TreeCheck<'a>,
    pub next_root: &'a Commitment,
    /// Leaves in the successor's tree, needed to locate a position in it.
    pub next_num_leaves: usize,
    /// The successor tree's cap height. Its cap rides this round's first
    /// successor opening, which is that tree's first opening in proof order.
    pub next_cap_height: usize,
}

/// The cap heights a round opens its two trees under (prover side).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RoundCaps {
    /// The current tree's cap height.
    pub current: usize,
    /// True when this round's first current opening is the current tree's
    /// first opening in proof order (round 0), and so carries its cap.
    pub current_owner: bool,
    /// The successor tree's cap height. The successor is always owned by the
    /// round that commits it.
    pub next: usize,
}

/// The openings one round sends.
///
/// The two value fields differ in the first round of a chain: the committed
/// codeword is base-field, because a trace is, while its successor has been
/// folded with an extension challenge. Later rounds have `C = N`.
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
pub struct RoundProof<C: IsField, N: IsField> {
    /// Per query: the block of the current codeword that folds onto the query.
    pub current: Vec<CosetOpening<C>>,
    /// Per query: the successor block holding the folded value.
    pub next: Vec<CosetOpening<N>>,
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
///
/// `caps` is the Merkle cap each tree is opened under ([`RoundCaps`]); the
/// default is no cap on either.
pub fn prove<C, N, T, H>(
    current: &CodewordCommitment<C, H>,
    next: &CodewordCommitment<N, H>,
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
    // The transcript squeezes that choose the indices, and the map onto the
    // successor's leaves. Host work, and the only part of a round's openings
    // that is not `open_many`.
    let __wq_sample = crate::whir_split::mark();
    let queries = sample_queries(transcript, config.num_queries, current.num_leaves());

    let leaves: Vec<usize> = queries
        .iter()
        .map(|q| leaf_and_slot(*q, next.num_leaves()).0)
        .collect();
    crate::whir_split::add(&crate::whir_split::QUERY_SAMPLE, __wq_sample);

    Ok(RoundProof {
        current: current.open_many_capped(&queries, caps.current, caps.current_owner)?,
        next: next.open_many_capped(&leaves, caps.next, true)?,
    })
}

/// Checks a round against the two commitments.
///
/// Re-derives the queries from the transcript, so the prover could not have
/// chosen them. Returns the successor tree's authenticated check, which the
/// next round opens its current tree against.
///
/// ⚠ ORDER. The opening counts are checked before any opening is indexed or
/// any cap is read, so a proof with too few openings is refused and never
/// panics.
pub fn verify<'a, F, C, N, T, H>(
    proof: &'a RoundProof<C, N>,
    commitments: RoundCommitments<'a>,
    domain: &Domain<F>,
    alphas: &[FieldElement<N>],
    config: &RoundConfig,
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
    let current_depth = num_leaves.trailing_zeros() as usize;
    if !commitments.next_num_leaves.is_power_of_two() {
        return Err(Error::NotPowerOfTwo(commitments.next_num_leaves));
    }
    let next_depth = commitments.next_num_leaves.trailing_zeros() as usize;

    // Both trees' checks, each built once from the tree's first opening —
    // after the count guard above, so index 0 exists (M2). With no openings
    // (`num_queries == 0`) no cap exists either: `CapPolicy::height` is 0 for
    // an unopened tree, and the successor's check is its bare root.
    let (current_check, current_first, next_check, next_first) =
        match (proof.current.first(), proof.next.first()) {
            (Some(cur), Some(nxt)) => {
                let (current_check, current_first) =
                    commitments.current.open::<C, H>(current_depth, cur)?;
                let (next_check, next_first) = TreeCheck::Owner {
                    root: commitments.next_root,
                    cap_height: commitments.next_cap_height,
                }
                .open::<N, H>(next_depth, nxt)?;
                (current_check, current_first, next_check, next_first)
            }
            _ => {
                if commitments.next_cap_height != 0 {
                    return Err(Error::CapRejected);
                }
                return Ok(CappedRoot::uncapped(commitments.next_root, next_depth));
            }
        };

    let queries = sample_queries(transcript, config.num_queries, num_leaves);

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
        if !verify_opening_capped::<C, H>(&current_check, q, cur, cur_siblings) {
            return Err(Error::OpeningRejected { query: i });
        }
        let (leaf, slot) = leaf_and_slot(q, commitments.next_num_leaves);
        if !verify_opening_capped::<N, H>(&next_check, leaf, nxt, nxt_siblings) {
            return Err(Error::OpeningRejected { query: i });
        }

        let folded = fold_coset::<F, C, N>(&cur.values, domain, q, alphas)?;
        let claimed = nxt.values.get(slot).ok_or(Error::QueryOutOfRange {
            index: slot,
            bound: nxt.values.len(),
        })?;
        if folded != *claimed {
            return Err(Error::FoldInconsistent { query: i });
        }
    }

    Ok(next_check)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    use crypto::merkle_tree::{
        cap::verify_merkle_path_to_cap_from_leaf_hash, traits::IsMerkleTreeBackend,
    };

    use crate::{
        mle::Mle,
        whir::{encode, fold_codeword_k, monomial_coefficients},
        whir_hash::KeccakWhir,
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

    fn commitments<'a>(
        fx: &Fixture,
        current_root: &'a Commitment,
        next_root: &'a Commitment,
    ) -> RoundCommitments<'a> {
        RoundCommitments {
            current: TreeCheck::Owner {
                root: current_root,
                cap_height: 0,
            },
            next_root,
            next_num_leaves: fx.next.num_leaves(),
            next_cap_height: 0,
        }
    }

    fn run(fx: &Fixture, proof: &RoundProof<F, F>) -> Result<(), Error> {
        let (current_root, next_root) = (fx.current.root(), fx.next.root());
        verify::<F, F, F, _, KeccakWhir>(
            proof,
            commitments(fx, &current_root, &next_root),
            &fx.domain,
            &fx.alphas,
            &fx.config,
            &mut transcript(),
        )
        .map(|_| ())
    }

    #[test]
    fn an_honest_round_verifies() {
        for k in 1..=3usize {
            let mut fx = fixture(4, 2, k);
            fx.config.num_queries = 4;
            let proof = prove(
                &fx.current,
                &fx.next,
                &fx.config,
                RoundCaps::default(),
                &mut transcript(),
            )
            .unwrap();
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
        let mut proof = prove(
            &fx.current,
            &fx.next,
            &fx.config,
            RoundCaps::default(),
            &mut transcript(),
        )
        .unwrap();
        proof.current[0].values[0] += FE::one();

        assert!(matches!(
            run(&fx, &proof).unwrap_err(),
            Error::OpeningRejected { query: 0 }
        ));
    }

    #[test]
    fn a_tampered_successor_opening_is_rejected() {
        let fx = fixture(4, 2, 2);
        let mut proof = prove(
            &fx.current,
            &fx.next,
            &fx.config,
            RoundCaps::default(),
            &mut transcript(),
        )
        .unwrap();
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

        let proof = prove(
            &fx.current,
            &fx.next,
            &fx.config,
            RoundCaps::default(),
            &mut transcript(),
        )
        .unwrap();
        assert!(matches!(
            run(&fx, &proof).unwrap_err(),
            Error::FoldInconsistent { .. }
        ));
    }

    #[test]
    fn the_wrong_folding_randomness_is_rejected() {
        let mut fx = fixture(4, 2, 2);
        let proof = prove(
            &fx.current,
            &fx.next,
            &fx.config,
            RoundCaps::default(),
            &mut transcript(),
        )
        .unwrap();

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
        let proof = prove(
            &fx.current,
            &fx.next,
            &fx.config,
            RoundCaps::default(),
            &mut transcript(),
        )
        .unwrap();
        run(&fx, &proof).unwrap();

        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        let (current_root, next_root) = (fx.current.root(), fx.next.root());
        let result = verify::<F, F, F, _, KeccakWhir>(
            &proof,
            commitments(&fx, &current_root, &next_root),
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
        let mut proof = prove(
            &fx.current,
            &fx.next,
            &fx.config,
            RoundCaps::default(),
            &mut transcript(),
        )
        .unwrap();
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
        let proof = prove(
            &fx.current,
            &fx.next,
            &fx.config,
            RoundCaps::default(),
            &mut transcript(),
        )
        .unwrap();
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
        let proof = prove(
            &fx.current,
            &fx.next,
            &fx.config,
            RoundCaps::default(),
            &mut transcript(),
        )
        .unwrap();
        assert_eq!(proof.current.len(), 7);
        run(&fx, &proof).unwrap();
    }

    // ---------------------------------------------------------------- caps

    type B = <KeccakWhir as WhirHash>::Backend<F>;

    /// A round opened and checked under caps, both trees.
    fn run_capped(
        fx: &Fixture,
        proof: &RoundProof<F, F>,
        caps: RoundCaps,
        label: &[u8],
    ) -> Result<(), Error> {
        let (current_root, next_root) = (fx.current.root(), fx.next.root());
        verify::<F, F, F, _, KeccakWhir>(
            proof,
            RoundCommitments {
                current: TreeCheck::Owner {
                    root: &current_root,
                    cap_height: caps.current,
                },
                next_root: &next_root,
                next_num_leaves: fx.next.num_leaves(),
                next_cap_height: caps.next,
            },
            &fx.domain,
            &fx.alphas,
            &fx.config,
            &mut DefaultTranscript::<F>::new(label),
        )
        .map(|_| ())
    }

    #[test]
    fn a_capped_round_verifies_and_carries_its_caps_on_the_first_openings() {
        // current: 32 leaves (depth 5); successor: 16 leaves (depth 4).
        let mut fx = fixture(4, 2, 1);
        fx.config.num_queries = 5;
        for (c_cur, c_next) in [(0, 0), (1, 0), (0, 2), (3, 2), (5, 4)] {
            let caps = RoundCaps {
                current: c_cur,
                current_owner: true,
                next: c_next,
            };
            let proof = prove(&fx.current, &fx.next, &fx.config, caps, &mut transcript()).unwrap();
            for (i, (cur, nxt)) in proof.current.iter().zip(&proof.next).enumerate() {
                let own = |c: usize| if i == 0 && c > 0 { 1usize << c } else { 0 };
                assert_eq!(cur.proof.merkle_path.len(), 5 - c_cur + own(c_cur));
                assert_eq!(nxt.proof.merkle_path.len(), 4 - c_next + own(c_next));
            }
            run_capped(&fx, &proof, caps, b"whir-round-test")
                .unwrap_or_else(|e| panic!("caps ({c_cur}, {c_next}): {e:?}"));
        }
    }

    /// A cap node no query reaches, flipped. Every
    /// per-query check still accepts against the forged cap — shown below —
    /// so ONLY the cap-to-root check can refuse it. The error names it.
    #[test]
    fn a_cap_node_no_query_reaches_is_refused_by_the_cap_check_alone() {
        let mut fx = fixture(4, 2, 1);
        fx.config.num_queries = 2;
        let (d_cur, c_cur, d_next, c_next) = (5usize, 3usize, 4usize, 2usize);
        let caps = RoundCaps {
            current: c_cur,
            current_owner: true,
            next: c_next,
        };
        let proof = prove(&fx.current, &fx.next, &fx.config, caps, &mut transcript()).unwrap();
        run_capped(&fx, &proof, caps, b"whir-round-test").unwrap();
        let queries = sample_queries::<F, _>(&mut transcript(), 2, fx.current.num_leaves());

        // The current tree.
        let reached: Vec<usize> = queries.iter().map(|q| q >> (d_cur - c_cur)).collect();
        let j = (0..1 << c_cur).find(|j| !reached.contains(j)).unwrap();
        let mut forged = proof.clone();
        forged.current[0].proof.merkle_path[d_cur - c_cur + j][0] ^= 1;
        let cap = forged.current[0].proof.merkle_path[d_cur - c_cur..].to_vec();
        for (q, opening) in queries.iter().zip(&forged.current) {
            let siblings = &opening.proof.merkle_path[..d_cur - c_cur];
            assert!(
                verify_merkle_path_to_cap_from_leaf_hash::<B>(
                    siblings,
                    &cap,
                    d_cur,
                    *q,
                    B::hash_data(&opening.values)
                ),
                "every query must still fold onto the forged cap, or this is not the fixture"
            );
        }
        assert!(matches!(
            run_capped(&fx, &forged, caps, b"whir-round-test"),
            Err(Error::CapRejected)
        ));

        // The successor tree: leaf `q mod 16`.
        let leaves: Vec<usize> = queries
            .iter()
            .map(|q| leaf_and_slot(*q, fx.next.num_leaves()).0)
            .collect();
        let reached: Vec<usize> = leaves.iter().map(|l| l >> (d_next - c_next)).collect();
        let j = (0..1 << c_next).find(|j| !reached.contains(j)).unwrap();
        let mut forged = proof.clone();
        forged.next[0].proof.merkle_path[d_next - c_next + j][0] ^= 1;
        let cap = forged.next[0].proof.merkle_path[d_next - c_next..].to_vec();
        for (leaf, opening) in leaves.iter().zip(&forged.next) {
            let siblings = &opening.proof.merkle_path[..d_next - c_next];
            assert!(verify_merkle_path_to_cap_from_leaf_hash::<B>(
                siblings,
                &cap,
                d_next,
                *leaf,
                B::hash_data(&opening.values)
            ));
        }
        assert!(matches!(
            run_capped(&fx, &forged, caps, b"whir-round-test"),
            Err(Error::CapRejected)
        ));
    }

    /// The WHIR analogue of the STARK's exact path-length check: a leaf forged from an
    /// INTERNAL node. Under keccak a 64-byte block (eight base values at
    /// `k = 3`) is a valid parent input, so values whose bytes are the level-1
    /// node's two children hash to that node, and a path one sibling short
    /// then folds to the root — the raw fold accepts it (asserted). At index 0
    /// or all-ones the shifted index bits agree with the true ones, so the
    /// fixture needs a statement whose first query lands there.
    ///
    /// Only the exact-length check can refuse it: without it the Merkle check
    /// passes and the round fails later at the FOLD (`FoldInconsistent`), so
    /// the `OpeningRejected { query: 0 }` this asserts is the length check's.
    #[test]
    fn a_leaf_forged_from_an_internal_node_is_refused_by_the_path_length_alone() {
        const P: u64 = 0xFFFF_FFFF_0000_0001;
        let fx = fixture(4, 2, 3);
        let depth = fx.current.depth();
        let leaves = fx.current.num_leaves();
        assert_eq!((depth, leaves), (3, 8));
        let root = fx.current.root();

        let (label, proof, forged) = (0u64..256)
            .find_map(|i| {
                let label = format!("m1a-{i}");
                let mut proof = prove(
                    &fx.current,
                    &fx.next,
                    &fx.config,
                    RoundCaps::default(),
                    &mut DefaultTranscript::<F>::new(label.as_bytes()),
                )
                .ok()?;
                let q0 = sample_queries::<F, _>(
                    &mut DefaultTranscript::<F>::new(label.as_bytes()),
                    1,
                    leaves,
                )[0];
                if q0 != 0 && q0 != leaves - 1 {
                    return None;
                }
                let honest = &proof.current[0];
                let leaf = B::hash_data(&honest.values);
                let s0 = honest.proof.merkle_path[0];
                let (l, r) = if q0 % 2 == 0 { (leaf, s0) } else { (s0, leaf) };
                let values: Vec<FE> = l
                    .iter()
                    .chain(r.iter())
                    .copied()
                    .collect::<Vec<u8>>()
                    .chunks_exact(8)
                    .map(|c| u64::from_be_bytes(c.try_into().unwrap()))
                    .map(|v| (v < P).then(|| FE::from(v)))
                    .collect::<Option<_>>()?;
                assert_eq!(
                    B::hash_data(&values),
                    B::hash_new_parent(&l, &r),
                    "the forged block must hash to the level-1 node"
                );
                let forged = CosetOpening {
                    values,
                    proof: crypto::merkle_tree::proof::Proof {
                        merkle_path: honest.proof.merkle_path[1..].to_vec(),
                    },
                };
                assert!(
                    crypto::merkle_tree::proof::verify_merkle_path_from_leaf_hash::<B>(
                        &forged.proof.merkle_path,
                        &root,
                        q0,
                        B::hash_data(&forged.values)
                    ),
                    "the raw fold must accept the short path, or the fixture tests nothing"
                );
                assert!(!crate::whir_commit::verify_opening::<F, KeccakWhir>(
                    &root, depth, q0, &forged
                ));
                proof.current[0] = forged.clone();
                Some((label, proof, forged))
            })
            .expect("a statement whose first query is 0 or all-ones");
        assert_eq!(forged.proof.merkle_path.len(), depth - 1);
        assert!(matches!(
            run_capped(&fx, &proof, RoundCaps::default(), label.as_bytes()),
            Err(Error::OpeningRejected { query: 0 })
        ));
    }

    /// M2: a proof with no openings at all is refused by the count guard, not
    /// by a panic on the owner's index.
    #[test]
    fn a_capped_round_with_no_openings_is_refused_without_panicking() {
        let mut fx = fixture(4, 2, 1);
        fx.config.num_queries = 3;
        let caps = RoundCaps {
            current: 2,
            current_owner: true,
            next: 2,
        };
        let mut proof = prove(&fx.current, &fx.next, &fx.config, caps, &mut transcript()).unwrap();
        proof.current.clear();
        proof.next.clear();
        assert!(matches!(
            run_capped(&fx, &proof, caps, b"whir-round-test"),
            Err(Error::QueryCountMismatch { .. })
        ));
    }
}

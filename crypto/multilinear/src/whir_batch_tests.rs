//! S1 — batched openings, on the host: a batch of one is a chain byte for
//! byte, a batch of up to eleven (the largest group on block 25368371) verifies
//! under the production schedule, caps and grinds, and every value a batch
//! shares or keeps apart is refused when it is forged.

use crypto::fiat_shamir::{default_transcript::DefaultTranscript, is_transcript::IsTranscript};
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as F,
};

use crate::{
    Error,
    eq::eq_eval,
    mle::Mle,
    stacked_eval::{self, Claimed, StackedCommitment, WeightShare},
    stacking::StackedLayout,
    whir::{Domain, encode, fold_codeword_k, lift_coefficients},
    whir_batch::{self, NextTree, WideCommitment},
    whir_chain::{
        self, BATCH_WORD_BIT, BatchCap, CapPolicy, ChainConfig, ChainFormat, ChainProof, FirstFold,
        GrindBits, RoundNonces, RoundOpenings, Stacked, WhirBatch, WhirFolds,
    },
    whir_commit::{Codeword, Commitment},
    whir_hash::{KeccakWhir, RpxWhir, WhirHash},
    whir_round::{RoundCaps, RoundConfig, TreeCheck},
};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;
type Tr<H> = DefaultTranscript<Ext, <H as WhirHash>::Transcript>;

/// The production posture at a small height: first6 then uniform4, blowup 4,
/// `cap` on every tree, `grind` bits in all three places.
fn config(num_queries: usize, cap: CapPolicy, grind: u8) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries,
        grind: GrindBits::uniform(grind),
        format: ChainFormat {
            cap,
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            batch: WhirBatch::Off,
        },
    }
}

fn batched(config: ChainConfig, cap: usize) -> ChainConfig {
    ChainConfig {
        format: ChainFormat {
            batch: WhirBatch::Cap(BatchCap::new(cap).unwrap()),
            ..config.format
        },
        ..config
    }
}

fn poly(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| {
                FE::from(
                    i.wrapping_add(seed)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(seed.wrapping_mul(6364136223846793005))
                        >> 11,
                )
            })
            .collect(),
    )
    .unwrap()
}

/// `K` polynomials of `n` variables, each weighted by `eq` at its own point.
struct Batch {
    polys: Vec<Mle<F>>,
    points: Vec<Vec<EE>>,
    claim: EE,
    n: usize,
}

fn batch(chains: usize, n: usize, seed: u64) -> Batch {
    let polys: Vec<Mle<F>> = (0..chains).map(|j| poly(n, seed + 17 * j as u64)).collect();
    let points: Vec<Vec<EE>> = (0..chains)
        .map(|j| {
            (0..n)
                .map(|i| EE::from(seed * 1009 + 31 * j as u64 + 7 * i as u64 + 3))
                .collect()
        })
        .collect();
    let claim = polys
        .iter()
        .zip(&points)
        .fold(EE::zero(), |acc, (p, z)| acc + p.evaluate_in(z).unwrap());
    Batch {
        polys,
        points,
        claim,
        n,
    }
}

impl Batch {
    fn sources(&self) -> Vec<Stacked<'_, F>> {
        self.polys
            .iter()
            .map(|p| Stacked {
                parts: vec![(p, 0)],
                resident: None,
                num_vars: self.n,
            })
            .collect()
    }

    fn shares(&self) -> Vec<Vec<WeightShare<'_, Ext>>> {
        self.points
            .iter()
            .map(|z| {
                vec![WeightShare {
                    offset: 0,
                    point: z,
                    scale: EE::one(),
                }]
            })
            .collect()
    }

    fn prove<H: WhirHash>(
        &self,
        cfg: &ChainConfig,
    ) -> (Vec<ChainProof<F, Ext>>, Commitment, Domain<F>) {
        let (commitment, domain) =
            whir_batch::commit::<F, H>(&self.sources(), cfg).expect("commit");
        let proofs = whir_batch::prove::<F, Ext, _, H>(
            &self.sources(),
            &self.shares(),
            self.n,
            &commitment,
            &domain,
            cfg,
            &mut Tr::<H>::new(b"whir-batch"),
        )
        .expect("prove");
        (proofs, commitment.root(), domain)
    }

    fn verify<H: WhirHash>(
        &self,
        proofs: &[ChainProof<F, Ext>],
        root: &Commitment,
        domain: &Domain<F>,
        cfg: &ChainConfig,
        claim: EE,
    ) -> Result<(), Error> {
        whir_batch::verify::<F, Ext, _, _, H>(
            proofs,
            root,
            |j: usize, at: &[EE]| eq_eval(&self.points[j], at),
            claim,
            self.n,
            domain,
            cfg,
            &mut Tr::<H>::new(b"whir-batch"),
        )
    }
}

/// Proves, applies `tamper`, verifies — the shape of every refusal below.
fn run<H: WhirHash>(
    b: &Batch,
    cfg: &ChainConfig,
    tamper: impl FnOnce(&mut Vec<ChainProof<F, Ext>>, &mut EE),
) -> Result<(), Error> {
    let (mut proofs, root, domain) = b.prove::<H>(cfg);
    let mut claim = b.claim;
    tamper(&mut proofs, &mut claim);
    b.verify::<H>(&proofs, &root, &domain, cfg, claim)
}

/// The first round's current blocks, which are base-field.
fn base_current(proof: &mut ChainProof<F, Ext>) -> &mut Vec<crate::whir_commit::CosetOpening<F>> {
    match &mut proof.rounds[0].openings {
        RoundOpenings::Base(p) => &mut p.current,
        RoundOpenings::Extension(_) => panic!("round 0 is base-field"),
    }
}

/// Round `r`'s successor blocks (any round with a successor).
fn successor(
    proof: &mut ChainProof<F, Ext>,
    r: usize,
) -> &mut Vec<crate::whir_commit::CosetOpening<Ext>> {
    match &mut proof.rounds[r].openings {
        RoundOpenings::Base(p) => &mut p.next,
        RoundOpenings::Extension(p) => &mut p.next,
    }
}

// ───────────────────────────── the split and the word ─────────────────────────

#[test]
fn a_group_splits_into_balanced_batches_in_order() {
    let cap = |c| WhirBatch::Cap(BatchCap::new(c).unwrap());
    assert_eq!(cap(4).split(11), vec![0..4, 4..8, 8..11]);
    assert_eq!(cap(4).split(9), vec![0..3, 3..6, 6..9]);
    assert_eq!(cap(4).split(8), vec![0..4, 4..8]);
    assert_eq!(cap(4).split(7), vec![0..4, 4..7]);
    assert_eq!(cap(4).split(1), vec![0..1]);
    assert_eq!(cap(4).split(0), Vec::<core::ops::Range<usize>>::new());
    assert_eq!(cap(16).split(11), vec![0..11]);
    assert_eq!(cap(1).split(3), vec![0..1, 1..2, 2..3]);
    assert_eq!(WhirBatch::Off.split(3), vec![0..1, 1..2, 2..3]);
    // Every split covers the group exactly once, and no batch passes the cap.
    for c in 1..=12 {
        for k in 0..=40 {
            let ranges = cap(c).split(k);
            assert_eq!(ranges.iter().map(|r| r.len()).sum::<usize>(), k);
            assert!(ranges.iter().all(|r| !r.is_empty() && r.len() <= c));
            assert!(ranges.windows(2).all(|w| w[0].end == w[1].start));
            let (lo, hi) = (
                ranges.iter().map(|r| r.len()).min().unwrap_or(0),
                ranges.iter().map(|r| r.len()).max().unwrap_or(0),
            );
            assert!(hi - lo <= 1, "cap {c}, K = {k}: {ranges:?}");
        }
    }
}

#[test]
fn a_cap_outside_the_accounted_range_is_unconstructible() {
    assert!(BatchCap::new(0).is_none());
    assert!(BatchCap::new(whir_chain::MAX_BATCH + 1).is_none());
    assert_eq!(BatchCap::new(whir_chain::MAX_BATCH).unwrap().get(), 32);
}

#[test]
fn the_commitment_count_is_one_per_polynomial_or_one_per_batch() {
    let off = config(5, CapPolicy::Off, 0);
    assert_eq!(off.commitments_for(11), 11);
    assert_eq!(batched(off, 4).commitments_for(11), 3);
    assert_eq!(batched(off, 4).commitments_for(1), 1);
    assert_eq!(batched(off, 16).commitments_for(11), 1);
}

/// ★ The statement binds the mode: per-chain words are today's, and every
/// batched word is distinct from every per-chain word and from every other cap.
#[test]
fn the_statement_word_binds_the_opening_mode_and_cap() {
    let uniform = ChainConfig {
        format: ChainFormat::DEFAULT,
        ..config(5, CapPolicy::Off, 0)
    };
    let first6 = config(5, CapPolicy::Off, 0);
    assert_eq!(uniform.fold_word(), 4);
    assert_eq!(first6.fold_word(), 0x8041_0000_0000_0006);
    let mut words = vec![uniform.fold_word(), first6.fold_word()];
    for base in [uniform, first6] {
        for cap in 1..=whir_chain::MAX_BATCH {
            let word = batched(base, cap).fold_word();
            assert_ne!(word & BATCH_WORD_BIT, 0);
            assert_eq!((word >> 32) & 0xff, cap as u64);
            words.push(word);
        }
    }
    let mut unique = words.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), words.len(), "two formats share a word");
}

// ─────────────────────────────── batch of one ─────────────────────────────────

/// ★ A batch of one is a chain byte for byte: the same root, the same proof
/// and the same transcript state after, under both hashes.
fn one_is_a_chain<H: WhirHash>(cfg: &ChainConfig) {
    let b = batch(1, 12, 5);
    let source = &b.sources()[0];
    let (chain_commitment, domain) = whir_chain::commit_stacked::<F, H>(source, cfg, true).unwrap();
    let mut chain_transcript = Tr::<H>::new(b"whir-batch");
    let chain = whir_chain::prove_shared::<F, Ext, _, H>(
        source,
        &b.shares()[0],
        b.n,
        &chain_commitment,
        &domain,
        cfg,
        &mut chain_transcript,
    )
    .unwrap();

    let (wide, wide_domain) = whir_batch::commit::<F, H>(&b.sources(), cfg).unwrap();
    assert_eq!(wide.root(), chain_commitment.root(), "the roots differ");
    assert_eq!(wide_domain.size(), domain.size());
    let mut batch_transcript = Tr::<H>::new(b"whir-batch");
    let proofs = whir_batch::prove::<F, Ext, _, H>(
        &b.sources(),
        &b.shares(),
        b.n,
        &wide,
        &domain,
        cfg,
        &mut batch_transcript,
    )
    .unwrap();
    assert_eq!(proofs.len(), 1);
    assert_eq!(
        rkyv::to_bytes::<rkyv::rancor::Error>(&proofs[0])
            .unwrap()
            .to_vec(),
        rkyv::to_bytes::<rkyv::rancor::Error>(&chain)
            .unwrap()
            .to_vec(),
        "a batch of one is not the chain's proof"
    );
    let (a, c): (EE, EE) = (
        batch_transcript.sample_field_element(),
        chain_transcript.sample_field_element(),
    );
    assert_eq!(a, c, "the transcripts end in different states");

    // Both verifiers accept it.
    b.verify::<H>(&proofs, &wide.root(), &domain, cfg, b.claim)
        .unwrap();
    whir_chain::verify_weighted::<F, Ext, _, _, H>(
        &chain,
        &chain_commitment.root(),
        |at: &[EE]| eq_eval(&b.points[0], at),
        b.claim,
        b.n,
        &domain,
        cfg,
        &mut Tr::<H>::new(b"whir-batch"),
    )
    .unwrap();
}

#[test]
fn a_batch_of_one_is_the_chain_byte_for_byte_keccak() {
    one_is_a_chain::<KeccakWhir>(&config(9, CapPolicy::Off, 0));
    one_is_a_chain::<KeccakWhir>(&config(9, CapPolicy::Auto, 2));
}

#[test]
fn a_batch_of_one_is_the_chain_byte_for_byte_rpx() {
    one_is_a_chain::<RpxWhir>(&config(9, CapPolicy::Off, 0));
    one_is_a_chain::<RpxWhir>(&config(9, CapPolicy::Auto, 2));
}

// ────────────────────────────── batches verify ────────────────────────────────

#[test]
fn a_batch_verifies_at_every_size_up_to_the_largest_group() {
    let cfg = config(7, CapPolicy::Auto, 0);
    for chains in 1..=11 {
        run::<KeccakWhir>(&batch(chains, 12, 40 + chains as u64), &cfg, |_, _| {})
            .unwrap_or_else(|e| panic!("K = {chains}: {e:?}"));
    }
}

/// The largest group on the block, at the production posture (RPX, first6,
/// caps on every tree, grinds in all three places).
#[test]
fn eleven_chains_verify_at_the_production_posture() {
    let cfg = config(20, CapPolicy::Auto, 3);
    let b = batch(11, 12, 91);
    let (proofs, root, domain) = b.prove::<RpxWhir>(&cfg);
    b.verify::<RpxWhir>(&proofs, &root, &domain, &cfg, b.claim)
        .unwrap();
    // One path per query for the whole batch: the lead's first current leaf is
    // eleven 64-wide blocks, and a follower carries no opening at all.
    let lead_first = match &proofs[0].rounds[0].openings {
        RoundOpenings::Base(p) => p.current[0].values.len(),
        RoundOpenings::Extension(_) => unreachable!(),
    };
    assert_eq!(lead_first, 11 * 64);
    for follower in &proofs[1..] {
        assert!(follower.rounds.iter().all(|r| r.sumcheck.is_empty()
            && r.next_root.is_none()
            && r.nonces == RoundNonces::default()));
    }
}

// ─────────────────────────────── refusals ─────────────────────────────────────

fn eleven() -> (Batch, ChainConfig) {
    (batch(11, 12, 7), config(20, CapPolicy::Auto, 0))
}

#[test]
fn a_forged_claim_is_rejected() {
    let (b, cfg) = eleven();
    assert!(run::<KeccakWhir>(&b, &cfg, |_, claim| *claim += EE::one()).is_err());
}

#[test]
fn a_forged_out_of_domain_answer_of_a_follower_is_rejected() {
    let (b, cfg) = eleven();
    for r in 0..2 {
        assert!(
            run::<KeccakWhir>(&b, &cfg, |proofs, _| {
                let y = proofs[6].rounds[r].ood_value.as_mut().unwrap();
                *y += EE::one();
            })
            .is_err(),
            "round {r}"
        );
    }
}

/// Moving value from one chain's answer to another's keeps a plain sum; the
/// per-chain powers of `γ` do not let it through.
#[test]
fn out_of_domain_answers_shifted_between_chains_are_rejected() {
    let (b, cfg) = eleven();
    let outcome = run::<KeccakWhir>(&b, &cfg, |proofs, _| {
        *proofs[2].rounds[0].ood_value.as_mut().unwrap() += EE::one();
        let y = proofs[5].rounds[0].ood_value.as_mut().unwrap();
        *y = *y - EE::one();
    });
    assert!(outcome.is_err(), "{outcome:?}");
}

#[test]
fn a_forged_final_value_of_a_follower_is_rejected() {
    let (b, cfg) = eleven();
    assert!(run::<KeccakWhir>(&b, &cfg, |proofs, _| proofs[9].final_value += EE::one()).is_err());
}

#[test]
fn swapped_final_values_are_rejected() {
    let (b, cfg) = eleven();
    assert!(
        run::<KeccakWhir>(&b, &cfg, |proofs, _| {
            let (a, c) = (proofs[3].final_value, proofs[8].final_value);
            proofs[3].final_value = c;
            proofs[8].final_value = a;
        })
        .is_err()
    );
}

/// A value of a FOLLOWER's block inside the lead's wide leaf: the leaf's hash
/// covers every chain's block.
#[test]
fn a_tampered_block_of_a_follower_in_a_wide_base_leaf_is_rejected() {
    let (b, cfg) = eleven();
    let outcome = run::<KeccakWhir>(&b, &cfg, |proofs, _| {
        base_current(&mut proofs[0])[3].values[7 * 64 + 5] += FE::one();
    });
    assert!(
        matches!(outcome, Err(Error::OpeningRejected { query: 3 })),
        "{outcome:?}"
    );
}

#[test]
fn a_tampered_block_of_a_follower_in_a_wide_successor_leaf_is_rejected() {
    let (b, cfg) = eleven();
    let outcome = run::<KeccakWhir>(&b, &cfg, |proofs, _| {
        // Round 0's successor is tree 1, blocked at the next fold (4): 16
        // extension values a chain.
        successor(&mut proofs[0], 0)[2].values[10 * 16 + 1] += EE::one();
    });
    assert!(
        matches!(outcome, Err(Error::OpeningRejected { query: 2 })),
        "{outcome:?}"
    );
}

/// ⛔ A leaf one block short would be sliced out of range if the width were
/// not checked first: it must be refused, not panic.
#[test]
fn a_wide_leaf_missing_a_block_is_refused_without_panicking() {
    let (b, cfg) = eleven();
    let outcome = run::<KeccakWhir>(&b, &cfg, |proofs, _| {
        let values = &mut base_current(&mut proofs[0])[0].values;
        values.truncate(values.len() - 64);
    });
    assert!(
        matches!(outcome, Err(Error::OpeningRejected { query: 0 })),
        "{outcome:?}"
    );
}

/// A follower may carry its own answers and final value, and nothing else.
#[test]
fn a_follower_carrying_a_field_only_the_lead_may_carry_is_rejected() {
    let (b, cfg) = eleven();
    type Tamper = fn(&mut Vec<ChainProof<F, Ext>>);
    let cases: [Tamper; 4] = [
        |p| p[4].rounds[1].nonces.query = 1,
        |p| p[4].rounds[0].next_root = p[0].rounds[0].next_root,
        |p| p[4].rounds[2].sumcheck = p[0].rounds[2].sumcheck.clone(),
        |p| {
            let opening = base_current(&mut p[0])[0].clone();
            base_current(&mut p[4]).push(opening);
        },
    ];
    for (i, case) in cases.into_iter().enumerate() {
        let outcome = run::<KeccakWhir>(&b, &cfg, |proofs, _| case(proofs));
        assert!(
            matches!(outcome, Err(Error::BatchShape { .. })),
            "case {i}: {outcome:?}"
        );
    }
}

/// Eleven chains' root with ten chains' proofs: every leaf is one block too
/// wide for the count the verifier holds.
#[test]
fn a_batch_missing_a_chain_is_rejected() {
    let (b, cfg) = eleven();
    let outcome = run::<KeccakWhir>(&b, &cfg, |proofs, _| {
        proofs.pop();
    });
    assert!(outcome.is_err());
}

#[test]
fn a_forged_nonce_of_the_batch_is_rejected() {
    let b = batch(5, 12, 13);
    let cfg = config(9, CapPolicy::Auto, 4);
    for place in 0..3 {
        let outcome = run::<KeccakWhir>(&b, &cfg, |proofs, _| {
            let nonces = &mut proofs[0].rounds[0].nonces;
            match place {
                0 => nonces.folding += 1,
                1 => nonces.ood += 1,
                _ => nonces.query += 1,
            }
        });
        assert!(outcome.is_err(), "place {place}");
    }
}

#[test]
fn a_batch_replayed_under_another_transcript_is_rejected() {
    let (b, cfg) = eleven();
    let (proofs, root, domain) = b.prove::<KeccakWhir>(&cfg);
    let outcome = whir_batch::verify::<F, Ext, _, _, KeccakWhir>(
        &proofs,
        &root,
        |j: usize, at: &[EE]| eq_eval(&b.points[j], at),
        b.claim,
        b.n,
        &domain,
        &cfg,
        &mut Tr::<KeccakWhir>::new(b"another-statement"),
    );
    assert!(outcome.is_err());
}

// ──────────────────────────── the round, alone ────────────────────────────────

/// ★ THE CHECK A BATCH EXISTS FOR, per chain: a successor that is validly
/// committed but is NOT the fold of chain 2 of three. Every Merkle opening
/// verifies — only chain 2's fold relation breaks, so only the per-chain fold
/// check can refuse it.
#[test]
fn a_successor_that_is_not_the_fold_for_one_follower_is_rejected() {
    let (n, k, next_k) = (6usize, 2usize, 2usize);
    let domain = Domain::<F>::new(n + 2).unwrap();
    let codewords: Vec<Vec<FE>> = (0..3)
        .map(|j| encode::<F, F>(&lift_coefficients(&poly(n, 70 + j)), &domain).unwrap())
        .collect();
    let alphas: Vec<EE> = (0..k).map(|i| EE::from(17 + i as u64)).collect();
    let mut folded: Vec<Vec<EE>> = codewords
        .iter()
        .map(|c| fold_codeword_k::<F, F, Ext>(c, &domain, &alphas).unwrap().0)
        .collect();
    // Chain 2's successor: an honest codeword of ANOTHER polynomial.
    let folded_domain = Domain::<F>::new(n + 2 - k).unwrap();
    folded[2] = encode::<F, Ext>(
        &lift_coefficients(
            &Mle::new(
                poly(n - k, 99)
                    .evals()
                    .iter()
                    .map(|v| (*v).to_extension::<Ext>())
                    .collect(),
            )
            .unwrap(),
        ),
        &folded_domain,
    )
    .unwrap();

    let current = WideCommitment::<F, KeccakWhir>::from_codewords(
        codewords.into_iter().map(Codeword::Host).collect(),
        k,
    )
    .unwrap();
    let next = WideCommitment::<Ext, KeccakWhir>::from_codewords(
        folded.into_iter().map(Codeword::Host).collect(),
        next_k,
    )
    .unwrap();
    let round = RoundConfig {
        num_queries: 6,
        log_folding: k,
    };
    let proof = whir_batch::open_round(
        &current,
        Some(&next),
        &round,
        RoundCaps::default(),
        &mut Tr::<KeccakWhir>::new(b"round"),
    )
    .unwrap();
    let (current_root, next_root) = (current.root(), next.root());
    let outcome = whir_batch::verify_round::<F, F, Ext, _, KeccakWhir>(
        &proof,
        3,
        TreeCheck::Owner {
            root: &current_root,
            cap_height: 0,
        },
        NextTree {
            root: &next_root,
            num_leaves: next.num_leaves(),
            log_folding: next_k,
            cap_height: 0,
        },
        &domain,
        &alphas,
        round.num_queries,
        &mut Tr::<KeccakWhir>::new(b"round"),
    );
    assert!(
        matches!(outcome, Err(Error::FoldInconsistent { .. })),
        "{outcome:?}"
    );
}

/// A round whose current tree holds `current` chains and whose successor
/// holds the honest folds of `next` of them, with its openings — the fixture
/// the width refusals below read with a verifier told another count.
struct Narrow {
    domain: Domain<F>,
    alphas: Vec<EE>,
    current: WideCommitment<F, KeccakWhir>,
    next: WideCommitment<Ext, KeccakWhir>,
    proof: crate::whir_round::RoundProof<F, Ext>,
}

fn narrow(k: usize, next_k: Option<usize>, current: usize, next: usize) -> Narrow {
    let n = 6usize;
    let domain = Domain::<F>::new(n + 2).unwrap();
    let codewords: Vec<Vec<FE>> = (0..current.max(next) as u64)
        .map(|j| encode::<F, F>(&lift_coefficients(&poly(n, 50 + j)), &domain).unwrap())
        .collect();
    let alphas: Vec<EE> = (0..k).map(|i| EE::from(23 + i as u64)).collect();
    let folded: Vec<Codeword<Ext>> = codewords[..next]
        .iter()
        .map(|c| Codeword::Host(fold_codeword_k::<F, F, Ext>(c, &domain, &alphas).unwrap().0))
        .collect();
    let current = WideCommitment::<F, KeccakWhir>::from_codewords(
        codewords[..current]
            .iter()
            .cloned()
            .map(Codeword::Host)
            .collect(),
        k,
    )
    .unwrap();
    let next =
        WideCommitment::<Ext, KeccakWhir>::from_codewords(folded, next_k.unwrap_or(0)).unwrap();
    let proof = whir_batch::open_round(
        &current,
        next_k.map(|_| &next),
        &RoundConfig {
            num_queries: 6,
            log_folding: k,
        },
        RoundCaps::default(),
        &mut Tr::<KeccakWhir>::new(b"narrow"),
    )
    .unwrap();
    Narrow {
        domain,
        alphas,
        current,
        next,
        proof,
    }
}

/// ⛔ A tree whose leaves hold FEWER chains than the batch claims. Every Merkle
/// opening is honest for that narrower tree, and the fold of every chain it
/// does hold is right — so without the width check the chains it leaves out are
/// never checked at all. Each of the round's two trees is narrowed on its own,
/// so each width check is the ONLY one that can refuse its case.
#[test]
fn a_round_whose_tree_is_narrower_than_the_batch_is_refused() {
    // (current tree's chains, successor tree's chains, chains claimed)
    for (current, next, chains, honest) in [(2, 2, 2, true), (2, 3, 3, false), (3, 2, 3, false)] {
        let outcome = round_width(current, next, chains);
        if honest {
            outcome.expect("the fixture is an honest round");
        } else {
            assert!(
                matches!(outcome, Err(Error::OpeningRejected { query: 0 })),
                "trees of {current} and {next} chains read as {chains}: {outcome:?}"
            );
        }
    }
}

fn round_width(current: usize, next: usize, chains: usize) -> Result<(), Error> {
    let x = narrow(2, Some(2), current, next);
    let (current_root, next_root) = (x.current.root(), x.next.root());
    let check = |chains: usize| {
        whir_batch::verify_round::<F, F, Ext, _, KeccakWhir>(
            &x.proof,
            chains,
            TreeCheck::Owner {
                root: &current_root,
                cap_height: 0,
            },
            NextTree {
                root: &next_root,
                num_leaves: x.next.num_leaves(),
                log_folding: 2,
                cap_height: 0,
            },
            &x.domain,
            &x.alphas,
            6,
            &mut Tr::<KeccakWhir>::new(b"narrow"),
        )
        .map(|_| ())
    };
    check(chains)
}

/// The same for the last round, where a left-out chain's final value would
/// otherwise be bound by nothing.
#[test]
fn a_final_round_narrower_than_the_batch_is_refused() {
    let x = narrow(6, None, 2, 2);
    let finals: Vec<EE> = x
        .next
        .codewords()
        .iter()
        .map(|c| c.host().unwrap()[0])
        .collect();
    let root = x.current.root();
    let check = |finals: &[&EE]| {
        whir_batch::verify_final::<F, F, Ext, _, KeccakWhir>(
            &x.proof,
            TreeCheck::Owner {
                root: &root,
                cap_height: 0,
            },
            &x.domain,
            &x.alphas,
            6,
            finals,
            &mut Tr::<KeccakWhir>::new(b"narrow"),
        )
    };
    check(&[&finals[0], &finals[1]]).expect("the fixture is an honest last round");
    let extra = EE::from(12345u64);
    let outcome = check(&[&finals[0], &finals[1], &extra]);
    assert!(
        matches!(outcome, Err(Error::OpeningRejected { query: 0 })),
        "{outcome:?}"
    );
}

/// ★ Each chain's answer enters at its OWN power of `γ`: moving value from one
/// answer to another moves the combination. With one power for every chain it
/// would not, and a prover could pay one chain's forged answer out of another's.
#[test]
fn the_answers_enter_the_claim_at_distinct_powers() {
    let gamma = EE::new([FE::from(0x1234_5678u64), FE::from(99u64), FE::from(7u64)]);
    let scales = whir_batch::answer_scales(&gamma, 11);
    assert_eq!(scales.len(), 11);
    assert_eq!(scales[0], gamma, "a batch of one is a chain: gamma itself");
    for i in 0..scales.len() {
        for j in i + 1..scales.len() {
            assert_ne!(scales[i], scales[j], "chains {i} and {j} share a power");
        }
    }
    let combine = |ys: &[EE]| {
        ys.iter()
            .zip(&scales)
            .fold(EE::zero(), |acc, (y, s)| acc + y * s)
    };
    let answers: Vec<EE> = (0..11).map(|j| EE::from(1000 + j as u64)).collect();
    let mut shifted = answers.clone();
    shifted[2] += EE::one();
    shifted[5] = shifted[5] - EE::one();
    assert_ne!(combine(&answers), combine(&shifted));
}

// ───────────────────────────── the stacked opening ────────────────────────────

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    poly(num_vars, seed)
}

/// Forty-four columns of 2^6 rows in a 2^8 stack: eleven stacked polynomials,
/// the largest group on the block.
struct Group {
    layout: StackedLayout,
    columns: Vec<Mle<F>>,
    at: Vec<EE>,
    values: Vec<EE>,
}

fn group() -> Group {
    let (count, num_vars, n_stack) = (44usize, 6usize, 8usize);
    let columns: Vec<Mle<F>> = (0..count)
        .map(|i| column(num_vars, 3 + 13 * i as u64))
        .collect();
    let layout = StackedLayout::build(&vec![num_vars; count], n_stack).unwrap();
    assert_eq!(layout.num_polys(), 11);
    let at: Vec<EE> = (0..num_vars).map(|i| EE::from(101 + i as u64)).collect();
    let values = columns
        .iter()
        .map(|c| c.evaluate_in(&at).unwrap())
        .collect();
    Group {
        layout,
        columns,
        at,
        values,
    }
}

fn stacked_config(batch: WhirBatch) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 6,
        grind: GrindBits::default(),
        format: ChainFormat {
            cap: CapPolicy::Auto,
            batch,
            ..ChainFormat::DEFAULT
        },
    }
}

fn stacked_prove(
    g: &Group,
    cfg: &ChainConfig,
    values: &[EE],
) -> (
    stacked_eval::StackedProof<F, Ext>,
    Vec<Commitment>,
    Domain<F>,
) {
    let columns = crate::stacking::borrow(&g.columns);
    let stacked =
        StackedCommitment::<F, KeccakWhir>::commit(g.layout.clone(), &columns, None, cfg).unwrap();
    let proof = stacked_eval::prove::<F, Ext, _, KeccakWhir>(
        &stacked,
        &columns,
        None,
        &Claimed::Shared(&g.at),
        values,
        cfg,
        &mut Tr::<KeccakWhir>::new(b"stack"),
    )
    .unwrap();
    (proof, stacked.roots(), stacked.domain().clone())
}

fn stacked_verify(
    g: &Group,
    proof: &stacked_eval::StackedProof<F, Ext>,
    roots: &[Commitment],
    domain: &Domain<F>,
    cfg: &ChainConfig,
    values: &[EE],
) -> Result<(), Error> {
    stacked_eval::verify::<F, Ext, _, KeccakWhir>(
        proof,
        &g.layout,
        roots,
        &Claimed::Shared(&g.at),
        values,
        domain,
        cfg,
        &mut Tr::<KeccakWhir>::new(b"stack"),
    )
}

/// ★ A group of eleven under a cap of four: three batches, three roots, and
/// the columns settle; a forged column value is refused.
#[test]
fn a_group_opened_in_batches_pays_one_root_per_batch_and_settles_every_column() {
    let g = group();
    let cfg = stacked_config(WhirBatch::Cap(BatchCap::new(4).unwrap()));
    let (proof, roots, domain) = stacked_prove(&g, &cfg, &g.values);
    assert_eq!(roots.len(), 3);
    assert_eq!(proof.polys.len(), 11);
    stacked_verify(&g, &proof, &roots, &domain, &cfg, &g.values).unwrap();

    let mut forged = g.values.clone();
    forged[29] += EE::one();
    let (proof, roots, domain) = stacked_prove(&g, &cfg, &forged);
    assert!(stacked_verify(&g, &proof, &roots, &domain, &cfg, &forged).is_err());
}

/// A cap of one is every polynomial in a batch of its own — and a batch of one
/// is a chain, so the proof and the roots are the per-chain format's exactly.
#[test]
fn a_cap_of_one_opens_exactly_what_the_per_chain_format_opens() {
    let g = group();
    let (off, off_roots, _) = stacked_prove(&g, &stacked_config(WhirBatch::Off), &g.values);
    let (one, one_roots, _) = stacked_prove(
        &g,
        &stacked_config(WhirBatch::Cap(BatchCap::new(1).unwrap())),
        &g.values,
    );
    assert_eq!(off_roots, one_roots);
    assert_eq!(
        rkyv::to_bytes::<rkyv::rancor::Error>(&off)
            .unwrap()
            .to_vec(),
        rkyv::to_bytes::<rkyv::rancor::Error>(&one)
            .unwrap()
            .to_vec()
    );
}

/// The mode is the verifier's: a batched proof is refused under the per-chain
/// format and under another cap, and a per-chain proof under a batched one.
#[test]
fn a_proof_is_refused_under_another_opening_mode() {
    let g = group();
    let four = stacked_config(WhirBatch::Cap(BatchCap::new(4).unwrap()));
    let six = stacked_config(WhirBatch::Cap(BatchCap::new(6).unwrap()));
    let off = stacked_config(WhirBatch::Off);
    let (proof, roots, domain) = stacked_prove(&g, &four, &g.values);
    assert!(stacked_verify(&g, &proof, &roots, &domain, &off, &g.values).is_err());
    assert!(stacked_verify(&g, &proof, &roots, &domain, &six, &g.values).is_err());
    let (proof, roots, domain) = stacked_prove(&g, &off, &g.values);
    assert!(stacked_verify(&g, &proof, &roots, &domain, &four, &g.values).is_err());
}

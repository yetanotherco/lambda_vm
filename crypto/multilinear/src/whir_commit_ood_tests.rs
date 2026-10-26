//! A stack's commit-time out-of-domain claims ([`crate::commit_ood`]) through
//! the stacked opening: the answers are the stack's values at the point, an
//! opening carrying them verifies under both chain splits and both hashes the
//! block runs, and a false, moved, missing or extra answer is refused, as is a
//! verifier that leaves them out.

use crypto::fiat_shamir::{default_transcript::DefaultTranscript, is_transcript::IsTranscript};
use math::{
    field::{
        element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
        goldilocks::GoldilocksField as F,
    },
    traits::AsBytes,
};

use crate::{
    Error,
    mle::Mle,
    stacked_eval::{
        Claimed, CommitOod, OodClaims, OodCounts, StackedCommitment, StackedProof, ood_claims,
        prove_mapped_ood, verify_ood,
    },
    stacking::{StackedLayout, borrow},
    whir_chain::{ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, WhirFolds},
    whir_commit::Commitment,
    whir_hash::{KeccakWhir, P1Whir, WhirHash},
};

type EE = FieldElement<Ext>;

/// Mixed heights with gaps between them, seven polynomials of ten variables:
/// under `Indexed` a run of four and a run of three.
const HEIGHTS: [usize; 14] = [8, 9, 7, 9, 6, 9, 9, 8, 5, 9, 9, 4, 9, 7];
const N_STACK: usize = 10;

fn config(group: GroupFold) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
        format: ChainFormat {
            folds: WhirFolds::First(FirstFold::new(3).unwrap()),
            group,
            ..ChainFormat::DEFAULT
        },
    }
}

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| {
                FieldElement::<F>::from(
                    i.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11,
                )
            })
            .collect(),
    )
    .unwrap()
}

/// A stack committed, its claims answered at `z0`, and its opening proved
/// with them.
struct Opened {
    layout: StackedLayout,
    roots: Vec<Commitment>,
    domain: crate::whir::Domain<F>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
    claims: OodClaims<Ext>,
    counts: OodCounts,
    proof: StackedProof<F, Ext>,
}

fn open<H, T>(group: GroupFold, z0: &EE, transcript: fn() -> T) -> Opened
where
    H: WhirHash,
    T: IsTranscript<Ext>,
    EE: AsBytes,
{
    let cfg = config(group);
    let columns: Vec<Mle<F>> = HEIGHTS
        .iter()
        .enumerate()
        .map(|(i, &h)| column(h, 5 + 31 * i as u64))
        .collect();
    let points: Vec<Vec<EE>> = HEIGHTS
        .iter()
        .enumerate()
        .map(|(i, &h)| {
            (0..h)
                .map(|j| EE::from(7 + 11 * i as u64 + 3 * j as u64))
                .collect()
        })
        .collect();
    let values: Vec<EE> = columns
        .iter()
        .zip(&points)
        .map(|(c, p)| c.evaluate_in(p).unwrap())
        .collect();
    let layout = StackedLayout::build(&HEIGHTS, N_STACK).unwrap();
    assert_eq!(layout.num_polys(), 7, "the stack this test means");
    let stacked =
        StackedCommitment::<F, H>::commit(layout.clone(), &borrow(&columns), None, &cfg).unwrap();
    let (claims, counts) = ood_claims::<F, Ext, _>(&layout, &borrow(&columns), None, z0).unwrap();
    let proof = prove_mapped_ood::<F, Ext, T, H, _>(
        &stacked,
        &borrow(&columns),
        None,
        &Claimed::PerColumn(&points),
        &values,
        Some(&claims),
        &cfg,
        &mut transcript(),
    )
    .unwrap();
    Opened {
        layout,
        roots: stacked.roots(),
        domain: stacked.domain().clone(),
        points,
        values,
        claims,
        counts,
        proof,
    }
}

fn check<H, T>(
    o: &Opened,
    group: GroupFold,
    ood: Option<CommitOod<'_, Ext>>,
    transcript: fn() -> T,
) -> Result<(), Error>
where
    H: WhirHash,
    T: IsTranscript<Ext>,
{
    verify_ood::<F, Ext, T, H>(
        &o.proof,
        &o.layout,
        &o.roots,
        &Claimed::PerColumn(&o.points),
        &o.values,
        ood,
        &o.domain,
        &config(group),
        &mut transcript(),
    )
}

fn keccak() -> DefaultTranscript<Ext> {
    DefaultTranscript::new(b"commit-ood")
}

fn p1() -> <P1Whir as WhirHash>::Sponge {
    P1Whir::sponge()
}

/// Every refusal the claims owe, and the honest opening's acceptance, under
/// one hash and one split.
fn every_way<H, T>(group: GroupFold, transcript: fn() -> T)
where
    H: WhirHash,
    T: IsTranscript<Ext>,
{
    let z0 = EE::from(0x0005_eed0_f00d_u64);
    let o = open::<H, T>(group, &z0, transcript);
    let answers = &o.claims.answers;
    assert_eq!(answers.len(), o.layout.num_polys());
    assert_eq!(
        o.counts,
        OodCounts {
            on_card: 0,
            on_host: HEIGHTS.len()
        }
    );
    let with =
        |answers: &[EE]| check::<H, T>(&o, group, Some(CommitOod { z0: &z0, answers }), transcript);
    with(answers).unwrap_or_else(|e| panic!("{group:?}: the honest answers: {e:?}"));

    for at in 0..answers.len() {
        let mut wrong = answers.clone();
        wrong[at] += EE::one();
        assert!(with(&wrong).is_err(), "{group:?}: answer {at} false");
    }
    // Two answers of one run (both splits put polynomials 0..4 in one).
    let mut swapped = answers.clone();
    swapped.swap(1, 2);
    assert_ne!(answers[1], answers[2]);
    assert!(with(&swapped).is_err(), "{group:?}: two answers swapped");
    let n = answers.len();
    assert_eq!(
        with(&answers[..n - 1]),
        Err(Error::CommitOodCount {
            expected: n,
            got: n - 1
        }),
        "{group:?}: an answer missing"
    );
    let mut extra = answers.clone();
    extra.push(EE::one());
    assert_eq!(
        with(&extra),
        Err(Error::CommitOodCount {
            expected: n,
            got: n + 1
        }),
        "{group:?}: an answer too many"
    );
    let other = &z0 + EE::one();
    assert!(
        check::<H, T>(
            &o,
            group,
            Some(CommitOod {
                z0: &other,
                answers,
            }),
            transcript,
        )
        .is_err(),
        "{group:?}: the answers read at another point"
    );
    assert!(
        check::<H, T>(&o, group, None, transcript).is_err(),
        "{group:?}: a verifier that leaves the answers out"
    );
}

#[test]
fn an_opening_carries_its_answers_and_refuses_them_altered_per_polynomial() {
    every_way::<KeccakWhir, _>(GroupFold::PerPoly, keccak);
    every_way::<P1Whir, _>(GroupFold::PerPoly, p1);
}

/// ★ The production split: a run's polynomials share one chain, each answer
/// at its own power of `γ`, so one answer cannot pay for another's.
#[test]
fn an_opening_carries_its_answers_and_refuses_them_altered_per_run() {
    every_way::<KeccakWhir, _>(GroupFold::Indexed, keccak);
    every_way::<P1Whir, _>(GroupFold::Indexed, p1);
}

/// The answers are the stacked polynomials' values at the point, and a run's
/// answers cannot cancel: answers moved between two polynomials of a run by
/// equal and opposite amounts — their sum kept — are refused.
#[test]
fn a_runs_answers_cannot_trade_against_each_other() {
    let z0 = EE::from(0x1234_5678_u64);
    let o = open::<P1Whir, _>(GroupFold::Indexed, &z0, p1);
    let columns: Vec<Mle<F>> = HEIGHTS
        .iter()
        .enumerate()
        .map(|(i, &h)| column(h, 5 + 31 * i as u64))
        .collect();
    let stacked = o.layout.stack(&borrow(&columns)).unwrap();
    let point = crate::whir_chain::ood_point(&z0, N_STACK);
    for (poly, answer) in o.claims.answers.iter().enumerate() {
        assert_eq!(
            *answer,
            stacked[poly].evaluate_in(&point).unwrap(),
            "poly {poly}"
        );
    }
    let mut traded = o.claims.answers.clone();
    let delta = EE::from(99u64);
    traded[0] = &traded[0] + &delta;
    traded[1] = &traded[1] - &delta;
    assert!(
        check::<P1Whir, _>(
            &o,
            GroupFold::Indexed,
            Some(CommitOod {
                z0: &z0,
                answers: &traded,
            }),
            p1,
        )
        .is_err(),
        "two answers of one run traded, their sum kept"
    );
}

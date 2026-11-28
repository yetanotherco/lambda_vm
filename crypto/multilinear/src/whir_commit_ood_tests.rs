//! A stack's commit-time out-of-domain claims ([`crate::commit_ood`]) through
//! the stacked opening: the answers are the stack's values at the point, an
//! opening carrying them verifies under every chain form the block runs — a
//! chain a polynomial, a chain a run with a round-0 tree a polynomial or one
//! tree a run, and the BLAKE3 base's production form (one tree a run, coset
//! leaves, the WHIR round form) — and a false, moved, missing or extra answer
//! is refused, as is a false answer the prover's own chain carries and a
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
    whir_chain::{
        ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, Round0Trees, RoundForm,
        WhirFolds,
    },
    whir_commit::{Commitment, LeafOrder},
    whir_hash::{Blake3Whir, KeccakWhir, P1Whir, WhirHash},
};

type EE = FieldElement<Ext>;

/// Mixed heights with gaps between them, seven polynomials of ten variables:
/// under `Indexed` a run of four and a run of three.
const HEIGHTS: [usize; 14] = [8, 9, 7, 9, 6, 9, 9, 8, 5, 9, 9, 4, 9, 7];
const N_STACK: usize = 10;

/// How the stack's polynomials are opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// One chain a polynomial.
    PerPoly,
    /// One chain a run, a round-0 tree a polynomial.
    PerRun,
    /// One chain a run, one round-0 tree a run (joint leaves, coset order):
    /// how production commits every group of two or more polynomials.
    OneTree,
    /// [`Self::OneTree`] under the WHIR round form, its counts derived for
    /// the target: the BLAKE3 base's chains.
    Production,
}

/// The forms a run of two or more is opened under.
const RUN_FORMS: [Form; 3] = [Form::PerRun, Form::OneTree, Form::Production];

fn config(form: Form) -> ChainConfig {
    let base = ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
        format: ChainFormat {
            folds: WhirFolds::First(FirstFold::new(3).unwrap()),
            group: match form {
                Form::PerPoly => GroupFold::PerPoly,
                _ => GroupFold::Indexed,
            },
            ..ChainFormat::DEFAULT
        },
    };
    let one_tree = ChainConfig {
        format: ChainFormat {
            round0: Round0Trees::One,
            leaves: LeafOrder::Coset,
            ..base.format
        },
        ..base
    };
    match form {
        Form::PerPoly | Form::PerRun => base,
        Form::OneTree => one_tree,
        Form::Production => {
            let mut c = one_tree
                .with_whir_rounds(2, 32, N_STACK)
                .expect("a tested shrink");
            let RoundForm::Whir(w) = c.format.rounds else {
                unreachable!("the round form")
            };
            c.num_queries = crate::query_count::num_queries(
                c.log_blowup,
                w.union_rounds(),
                w.security_bits(),
                c.grind.query,
            );
            crate::whir_chain::check_round_form(&c).expect("the target sizes round 0");
            c
        }
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

fn columns() -> Vec<Mle<F>> {
    HEIGHTS
        .iter()
        .enumerate()
        .map(|(i, &h)| column(h, 5 + 31 * i as u64))
        .collect()
}

/// The stack's opening proved from the true polynomials with the claims
/// `claims` stated and carried: the honest proof when they are the true ones.
fn prove_with<H, T>(
    form: Form,
    o: &Opened,
    values: &[EE],
    claims: &OodClaims<Ext>,
    transcript: fn() -> T,
) -> Result<StackedProof<F, Ext>, Error>
where
    H: WhirHash,
    T: IsTranscript<Ext>,
{
    let cfg = config(form);
    let columns = columns();
    let stacked =
        StackedCommitment::<F, H>::commit(o.layout.clone(), &borrow(&columns), None, &cfg)?;
    prove_mapped_ood::<F, Ext, T, H, _>(
        &stacked,
        &borrow(&columns),
        None,
        &Claimed::PerColumn(&o.points),
        values,
        Some(claims),
        &cfg,
        &mut transcript(),
    )
}

fn open<H, T>(form: Form, z0: &EE, transcript: fn() -> T) -> Opened
where
    H: WhirHash,
    T: IsTranscript<Ext>,
    EE: AsBytes,
{
    let cfg = config(form);
    let columns = columns();
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
    form: Form,
    ood: Option<CommitOod<'_, Ext>>,
    transcript: fn() -> T,
) -> Result<(), Error>
where
    H: WhirHash,
    T: IsTranscript<Ext>,
{
    check_proof::<H, T>(&o.proof, o, &o.values, form, ood, transcript)
}

/// [`check`] of `proof` with the column values `values`.
fn check_proof<H, T>(
    proof: &StackedProof<F, Ext>,
    o: &Opened,
    values: &[EE],
    form: Form,
    ood: Option<CommitOod<'_, Ext>>,
    transcript: fn() -> T,
) -> Result<(), Error>
where
    H: WhirHash,
    T: IsTranscript<Ext>,
{
    verify_ood::<F, Ext, T, H>(
        proof,
        &o.layout,
        &o.roots,
        &Claimed::PerColumn(&o.points),
        values,
        ood,
        &o.domain,
        &config(form),
        &mut transcript(),
    )
}

fn keccak() -> DefaultTranscript<Ext> {
    DefaultTranscript::new(b"commit-ood")
}

fn p1() -> <P1Whir as WhirHash>::Sponge {
    P1Whir::sponge()
}

fn b3() -> <Blake3Whir as WhirHash>::Sponge {
    let mut t = Blake3Whir::sponge();
    t.append_bytes(b"commit-ood");
    t
}

/// Every refusal the claims owe, and the honest opening's acceptance, under
/// one hash and one form.
fn every_way<H, T>(form: Form, transcript: fn() -> T)
where
    H: WhirHash,
    T: IsTranscript<Ext>,
{
    let z0 = EE::from(0x0005_eed0_f00d_u64);
    let o = open::<H, T>(form, &z0, transcript);
    let answers = &o.claims.answers;
    assert_eq!(answers.len(), o.layout.num_polys());
    assert_eq!(
        o.counts,
        OodCounts {
            on_card: 0,
            on_host: HEIGHTS.len()
        }
    );
    if form != Form::PerPoly {
        let runs = o.proof.polys.len();
        assert!(runs < answers.len(), "{form:?}: a run of two or more");
    }
    let with =
        |answers: &[EE]| check::<H, T>(&o, form, Some(CommitOod { z0: &z0, answers }), transcript);
    with(answers).unwrap_or_else(|e| panic!("{form:?}: the honest answers: {e:?}"));

    for at in 0..answers.len() {
        let mut wrong = answers.clone();
        wrong[at] += EE::one();
        assert!(with(&wrong).is_err(), "{form:?}: answer {at} false");
    }
    // ★ A false answer the prover states and its chain carries — absorbed
    // and proved as given, so every challenge is the one the verifier draws —
    // is refused by that chain: the binding itself, not the transcript.
    for at in 0..answers.len() {
        let mut claims = o.claims.clone();
        claims.answers[at] += EE::one();
        let proof = prove_with::<H, T>(form, &o, &o.values, &claims, transcript)
            .unwrap_or_else(|e| panic!("{form:?}: the prover proves a false answer {at}: {e:?}"));
        let refused = check_proof::<H, T>(
            &proof,
            &o,
            &o.values,
            form,
            Some(CommitOod {
                z0: &z0,
                answers: &claims.answers,
            }),
            transcript,
        );
        assert!(
            refused.is_err(),
            "{form:?}: a false answer {at} carried by its chain was accepted"
        );
    }
    // Two answers of one run (every run form puts polynomials 0..4 in one).
    let mut swapped = answers.clone();
    swapped.swap(1, 2);
    assert_ne!(answers[1], answers[2]);
    assert!(with(&swapped).is_err(), "{form:?}: two answers swapped");
    let n = answers.len();
    assert_eq!(
        with(&answers[..n - 1]),
        Err(Error::CommitOodCount {
            expected: n,
            got: n - 1
        }),
        "{form:?}: an answer missing"
    );
    let mut extra = answers.clone();
    extra.push(EE::one());
    assert_eq!(
        with(&extra),
        Err(Error::CommitOodCount {
            expected: n,
            got: n + 1
        }),
        "{form:?}: an answer too many"
    );
    let other = &z0 + EE::one();
    assert!(
        check::<H, T>(
            &o,
            form,
            Some(CommitOod {
                z0: &other,
                answers,
            }),
            transcript,
        )
        .is_err(),
        "{form:?}: the answers read at another point"
    );
    assert!(
        check::<H, T>(&o, form, None, transcript).is_err(),
        "{form:?}: a verifier that leaves the answers out"
    );
}

#[test]
fn an_opening_carries_its_answers_and_refuses_them_altered_per_polynomial() {
    every_way::<KeccakWhir, _>(Form::PerPoly, keccak);
    every_way::<P1Whir, _>(Form::PerPoly, p1);
}

/// A run's polynomials share one chain, each answer at its own power of `γ`,
/// so one answer cannot pay for another's.
#[test]
fn an_opening_carries_its_answers_and_refuses_them_altered_per_run() {
    every_way::<KeccakWhir, _>(Form::PerRun, keccak);
    every_way::<P1Whir, _>(Form::PerRun, p1);
}

/// ★ The production split: one round-0 tree a run (joint leaves), under both
/// hashes that have a joint leaf.
#[test]
fn an_opening_carries_its_answers_and_refuses_them_altered_under_one_tree_a_run() {
    every_way::<P1Whir, _>(Form::OneTree, p1);
    every_way::<Blake3Whir, _>(Form::OneTree, b3);
}

/// ★ The BLAKE3 base's chains: one tree a run under the WHIR round form.
#[test]
fn an_opening_carries_its_answers_and_refuses_them_altered_under_the_blake3_production_form() {
    every_way::<Blake3Whir, _>(Form::Production, b3);
}

/// The answers are the stacked polynomials' values at the point, and a run's
/// answers cannot cancel: answers moved between two polynomials of a run by
/// equal and opposite amounts — their sum kept — are refused, under every
/// run form.
#[test]
fn a_runs_answers_cannot_trade_against_each_other() {
    fn under<H: WhirHash, T: IsTranscript<Ext>>(form: Form, transcript: fn() -> T) {
        let z0 = EE::from(0x1234_5678_u64);
        let o = open::<H, T>(form, &z0, transcript);
        let columns = columns();
        let stacked = o.layout.stack(&borrow(&columns)).unwrap();
        let point = crate::whir_chain::ood_point(&z0, N_STACK);
        for (poly, answer) in o.claims.answers.iter().enumerate() {
            assert_eq!(
                *answer,
                stacked[poly].evaluate_in(&point).unwrap(),
                "{form:?} poly {poly}"
            );
        }
        let mut traded = o.claims.answers.clone();
        let delta = EE::from(99u64);
        traded[0] = &traded[0] + &delta;
        traded[1] = &traded[1] - &delta;
        assert!(
            check::<H, T>(
                &o,
                form,
                Some(CommitOod {
                    z0: &z0,
                    answers: &traded,
                }),
                transcript,
            )
            .is_err(),
            "{form:?}: two answers of one run traded, their sum kept"
        );
    }
    under::<P1Whir, _>(Form::PerRun, p1);
    under::<P1Whir, _>(Form::OneTree, p1);
    under::<Blake3Whir, _>(Form::OneTree, b3);
    under::<Blake3Whir, _>(Form::Production, b3);
}

/// No answer is carried at power one. Column 0's claim rides at
/// `ρ^0 = 1`; were polynomial 0's answer carried at `γ^0 = 1` too, a prover
/// could move `Δ` from the answer onto column 0's value and keep every chain's
/// claim, proving the opening from the true polynomials. With `γ^(i+1)` the
/// moved pair changes the claim and the opening is refused — under every form,
/// the BLAKE3 base's included.
#[test]
fn an_answer_cannot_trade_against_column_zero() {
    fn under<H: WhirHash, T: IsTranscript<Ext>>(form: Form, transcript: fn() -> T) {
        let z0 = EE::from(0x0bad_cafe_u64);
        let o = open::<H, T>(form, &z0, transcript);
        assert_eq!(
            o.layout.placements()[0].poly,
            0,
            "column 0 lies in polynomial 0"
        );
        let delta = EE::from(1234u64);
        let mut values = o.values.clone();
        values[0] = &values[0] + &delta;
        let mut claims = o.claims.clone();
        claims.answers[0] = &claims.answers[0] - &delta;
        // The prover's chains run over the true polynomials; only the claims
        // it states are moved.
        let accepted =
            prove_with::<H, T>(form, &o, &values, &claims, transcript).is_ok_and(|proof| {
                check_proof::<H, T>(
                    &proof,
                    &o,
                    &values,
                    form,
                    Some(CommitOod {
                        z0: &z0,
                        answers: &claims.answers,
                    }),
                    transcript,
                )
                .is_ok()
            });
        assert!(
            !accepted,
            "{form:?}: answer 0 traded against column 0's value"
        );
    }
    under::<P1Whir, _>(Form::PerPoly, p1);
    for form in RUN_FORMS {
        if form != Form::Production {
            under::<P1Whir, _>(form, p1);
        }
    }
    under::<Blake3Whir, _>(Form::OneTree, b3);
    under::<Blake3Whir, _>(Form::Production, b3);
}

//! ★ The WHIR round form ([`RoundForm::Whir`]) on the host: a round that
//! commits a successor no longer opens it; each query's fold value becomes a
//! claim on the successor at `pow(y_j)`, weighted by `γ^{j+1}` for a `γ` drawn
//! after the openings, and the successor lives on a domain that shrinks by
//! `min(shrink, k)` bits a round, so the rate rises and later rounds query
//! fewer positions.
//!
//! What is pinned here, by the round form's soundness obligations:
//! 1. Each fold value is a claim on the successor at the folded domain's
//!    point: a false evaluation proved over another polynomial than the
//!    committed one is refused, and accepted only when the shift claims are
//!    dropped on both sides (the mutant); each convention of the claim (the
//!    point's domain, its squarings, `γ`'s powers) is load-bearing (a verifier
//!    that reads it otherwise refuses an honest proof).
//! 2. The successor is committed and its out-of-domain point answered before
//!    the round's positions are drawn, and `γ` after them: the transcript's
//!    order, pinned op by op on both sides; a verifier that draws the positions
//!    first refuses.
//! 3. Domains and counts are verifier constants: a proof under another shrink,
//!    another form or with one query too few is refused; one that takes the
//!    count from the proof accepts the short proof (the mutant).
//! 4. The final weight sums every out-of-domain and shift claim: leaving
//!    either out refuses an honest proof; a KAT pins a proof's bytes.
//! 5. Per-phase proven bits against a reference calculator's printed values,
//!    every round of the round form at or over 128 bits per proof; a rule
//!    that overstates a later round's rate drops a round below it (the
//!    mutant).
//!
//! Today's form ([`RoundForm::Fri`], the default) moves no byte: its fold
//! word, counts and caps are pinned here, and the existing pins and KATs run
//! unchanged.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::merkle_tree::cap::CapPolicy;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as F,
};

use crate::{
    Error,
    eq::{add_eq_sum, eq_evals},
    mle::Mle,
    stacked_eval::{Claimed, Refusal, StackedCommitment, prove as prove_stacked, verify_traced},
    stacking::{StackedLayout, borrow},
    whir::Domain,
    whir_chain::{
        ChainConfig, ChainFormat, ChainProof, FOLD_WORD_TAG, FOLD_WORD_WHIR, FinalPoly, FirstFold,
        GrindBits, GroupFold, LeafOrder, Round0Trees, RoundForm, RoundOpenings, WhirFolds,
        WhirRounds, check_round_form, commit, prove, round_form_faults::Fault, verify,
    },
    whir_commit::{CodewordCommitment, CosetOpening},
    whir_hash::{Blake3Whir, KeccakWhir, WhirHash},
};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;
type H = KeccakWhir;

fn with<R>(fault: Fault, f: impl FnOnce() -> R) -> R {
    crate::whir_chain::round_form_faults::with(fault, f)
}

fn mle(num_vars: usize, seed: u64) -> Mle<F> {
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

fn point(num_vars: usize, seed: u64) -> Vec<EE> {
    (0..num_vars as u64)
        .map(|i| {
            EE::new([
                FE::from(101 + 7 * i + seed),
                FE::from(3 + 11 * i + seed),
                FE::from(17 + 5 * i),
            ])
        })
        .collect()
}

fn folds(first: Option<usize>) -> WhirFolds {
    first.map_or(WhirFolds::Uniform, |k| {
        WhirFolds::First(FirstFold::new(k).expect("a tested first fold"))
    })
}

/// Today's config for a target of `bits` (no grind), FRI-style.
fn fri(
    log_blowup: usize,
    first: Option<usize>,
    log_folding: usize,
    n: usize,
    bits: u8,
) -> ChainConfig {
    ChainConfig::with_security_folds(
        log_blowup,
        log_folding,
        folds(first),
        n,
        bits,
        GrindBits::default(),
    )
}

/// The same config under the round form at `shrink`.
fn whir(
    log_blowup: usize,
    first: Option<usize>,
    log_folding: usize,
    shrink: usize,
    n: usize,
    bits: u8,
) -> ChainConfig {
    fri(log_blowup, first, log_folding, n, bits)
        .with_whir_rounds(shrink, bits, n)
        .expect("a tested shrink")
}

fn sponge() -> <H as WhirHash>::Sponge {
    let mut t = H::sponge();
    t.append_bytes(b"round-form");
    t
}

/// One chain: a polynomial, its claim, its commitment and a proof.
struct Single {
    z: Vec<EE>,
    y: EE,
    commitment: CodewordCommitment<F, H>,
    domain: Domain<F>,
    proof: ChainProof<F, Ext>,
}

fn single(cfg: &ChainConfig, n: usize, seed: u64) -> Single {
    let f = mle(n, seed);
    let z = point(n, seed);
    let y = f.evaluate_in(&z).unwrap();
    let (commitment, domain) = commit::<F, H>(&f, cfg, true).unwrap();
    let proof = prove::<F, Ext, _, H>(&f, &z, &commitment, &domain, cfg, &mut sponge()).unwrap();
    Single {
        z,
        y,
        commitment,
        domain,
        proof,
    }
}

fn check(s: &Single, proof: &ChainProof<F, Ext>, cfg: &ChainConfig) -> Result<(), Error> {
    verify::<F, Ext, _, H>(
        proof,
        &s.commitment.root(),
        &s.z,
        s.y,
        &s.domain,
        cfg,
        &mut sponge(),
    )
}

/// The current-tree blocks a round opened, and the successor blocks it
/// carries.
fn opened(openings: &RoundOpenings<F, Ext>) -> (usize, usize) {
    match openings {
        RoundOpenings::Base(p) => (p.current.len(), p.next.len()),
        RoundOpenings::Extension(p) => (p.current.len(), p.next.len()),
        RoundOpenings::BaseGrouped(p) => (p.current.len(), p.next.len()),
        RoundOpenings::BaseJoint(p) => (p.current.len(), p.next.len()),
    }
}

// ───────────────────────────── the format and its counts ─────────────────────

/// ★ Today's form is the default and moves nothing: the default format says
/// [`RoundForm::Fri`]; under it every round queries `num_queries`, the caps
/// are today's (`Q` for tree 0, `2Q` after), the domains fold by `k`, and the
/// fold word carries no round-form bit, at every fold schedule and format
/// field the word already encodes.
#[test]
fn the_fri_form_is_the_default_and_moves_no_count_cap_or_word() {
    assert_eq!(ChainFormat::DEFAULT.rounds, RoundForm::Fri);
    assert_eq!(RoundForm::default(), RoundForm::Fri);
    assert!(ChainFormat::DEFAULT.is_default());
    for first in [None, Some(4), Some(6)] {
        for n in [6usize, 13, 25, 27] {
            let cfg = fri(1, first, 4, n, 128);
            let schedule = cfg.schedule(n);
            assert_eq!(cfg.round_queries(n), vec![cfg.num_queries; schedule.len()]);
            assert_eq!(cfg.round_log_blowups(n), vec![1; schedule.len()]);
            let mut d = n + 1;
            let domains: Vec<usize> = schedule
                .iter()
                .map(|&k| {
                    let here = d;
                    d -= k;
                    here
                })
                .collect();
            assert_eq!(cfg.round_domain_logs(n), domains);
            assert_eq!(cfg.fold_word() & FOLD_WORD_WHIR, 0);
            for cap in [CapPolicy::Off, CapPolicy::Auto, CapPolicy::Fixed(3)] {
                let capped = ChainConfig {
                    format: ChainFormat { cap, ..cfg.format },
                    ..cfg
                };
                let mut depth = n + 1;
                let today: Vec<usize> = schedule
                    .iter()
                    .enumerate()
                    .map(|(t, &k)| {
                        depth -= k;
                        let openings = if t == 0 {
                            cfg.num_queries
                        } else {
                            2 * cfg.num_queries
                        };
                        crate::whir_commit::tree_cap_height(cap, openings, depth, 2)
                    })
                    .collect();
                assert_eq!(capped.tree_caps(n), today, "{cap:?} at {n}");
            }
        }
    }
    // The words the earlier format fields pinned are untouched.
    assert_eq!(fri(1, None, 4, 25, 128).fold_word(), 4);
    assert_eq!(
        fri(1, Some(6), 4, 25, 128).fold_word(),
        0x8041_0000_0000_0006
    );
}

/// The round form's fold word: tagged, its shrink, target and union in their
/// bits, distinct across shrinks, targets and unions, and from every
/// FRI-style word.
#[test]
fn the_round_form_moves_the_fold_word_by_shrink_and_target() {
    let base = fri(1, Some(6), 4, 27, 128);
    let mut words = std::collections::BTreeSet::new();
    words.insert(base.fold_word());
    for shrink in 1..=6usize {
        let cfg = base.with_whir_rounds(shrink, 128, 27).unwrap();
        let w = cfg.fold_word();
        assert_eq!(
            w & (FOLD_WORD_TAG | FOLD_WORD_WHIR),
            FOLD_WORD_TAG | FOLD_WORD_WHIR
        );
        assert_eq!((w >> 40) & 0x7, shrink as u64);
        assert_eq!((w >> 24) & 0xff, 128);
        assert_eq!(
            (w >> 16) & 0xff,
            7,
            "first6 then four a round: 7 rounds at 27"
        );
        assert_eq!(
            w & !(FOLD_WORD_WHIR | (0x7 << 40) | (0xffff << 16)),
            base.fold_word()
        );
        assert!(words.insert(w), "shrink {shrink}");
    }
    let other_target = ChainConfig {
        format: ChainFormat {
            rounds: RoundForm::Whir(WhirRounds::new(3, 100, 7).unwrap()),
            ..base.format
        },
        ..base
    };
    assert!(words.insert(other_target.fold_word()));
    let other_union = ChainConfig {
        format: ChainFormat {
            rounds: RoundForm::Whir(WhirRounds::new(3, 128, 8).unwrap()),
            ..base.format
        },
        ..base
    };
    assert!(words.insert(other_union.fold_word()));
    assert!(WhirRounds::new(0, 128, 7).is_none());
    assert!(WhirRounds::new(7, 128, 7).is_none());
    assert!(WhirRounds::new(3, 128, 0).is_none());
}

/// ★ THE PER-ROUND COUNTS, against a reference calculator's printed counts
/// (`Q` by queried round, the rule of `query_count` at each round's rate in
/// floating point): blowup 2 at first fold 4
/// and 6, blowup 4 at first fold 4 and 6, shrink 3 and 2, query grind 18 and
/// 24, a `2^27` stack, 128 bits. Round 0's count is today's; each later
/// round's is the rule at its own rate, which rises by `k − shrink` a round.
#[test]
fn the_round_queries_are_the_calculators() {
    let grind = |g: u8| GrindBits::query_only(g);
    let cfg = |lb: usize, k0: usize, shrink: usize, g: u8| {
        ChainConfig::with_security_folds(lb, 4, folds(Some(k0)), 27, 128, grind(g))
            .with_whir_rounds(shrink, 128, 27)
            .unwrap()
    };
    // (log_blowup, k0, shrink, grind, the calculator's counts by queried round)
    let cases: [(usize, usize, usize, u8, &[usize]); 6] = [
        (1, 6, 3, 18, &[229, 57, 46, 39]),
        (2, 6, 3, 18, &[114, 46, 39, 33]),
        (2, 4, 3, 18, &[114, 76, 57, 46, 39]),
        (2, 4, 2, 18, &[114, 57, 39, 29, 24]),
        (1, 4, 3, 18, &[229, 114, 76, 57, 46]),
        (2, 4, 3, 24, &[108, 72, 54, 44, 37]),
    ];
    for (lb, k0, shrink, g, want) in cases {
        let c = cfg(lb, k0, shrink, g);
        check_round_form(&c).unwrap();
        let got = c.round_queries(27);
        assert_eq!(
            &got[..want.len()],
            want,
            "blowup 2^{lb}, k0 {k0}, shrink {shrink}, g {g}"
        );
    }
}

/// The rate under the round form: each round's domain and inverse rate, by
/// schedule. A round folding fewer variables than the shrink shrinks its
/// domain by its fold (the rate never falls).
#[test]
fn the_round_domains_and_rates_follow_the_shrink() {
    let c = whir(1, Some(4), 4, 3, 27, 128);
    assert_eq!(c.schedule(27), vec![4, 4, 4, 4, 4, 4, 3]);
    assert_eq!(c.round_domain_logs(27), vec![28, 25, 22, 19, 16, 13, 10]);
    assert_eq!(c.round_log_blowups(27), vec![1, 2, 3, 4, 5, 6, 7]);
    // A shrink past a round's fold: the domain loses that fold.
    let c = whir(1, None, 2, 3, 7, 32);
    assert_eq!(c.schedule(7), vec![2, 2, 2, 1]);
    assert_eq!(c.round_domain_logs(7), vec![8, 6, 4, 2]);
    assert_eq!(c.round_log_blowups(7), vec![1, 1, 1, 1]);
    let c = whir(2, Some(6), 4, 1, 14, 32);
    assert_eq!(c.round_domain_logs(14), vec![16, 15, 14]);
    assert_eq!(c.round_log_blowups(14), vec![2, 7, 10]);
}

/// A round-form config whose target does not give its own round-0 count is
/// refused before anything is proved or read; the constructor's never is.
#[test]
fn a_target_that_does_not_size_round_zero_is_refused() {
    let c = whir(1, Some(4), 4, 3, 13, 40);
    check_round_form(&c).unwrap();
    let off = ChainConfig {
        num_queries: c.num_queries + 1,
        ..c
    };
    assert_eq!(
        check_round_form(&off),
        Err(Error::RoundFormTarget {
            expected: c.num_queries,
            got: c.num_queries + 1
        })
    );
    let f = mle(13, 3);
    let (commitment, domain) = commit::<F, H>(&f, &off, true).unwrap();
    let z = point(13, 3);
    assert!(matches!(
        prove::<F, Ext, _, H>(&f, &z, &commitment, &domain, &off, &mut sponge()),
        Err(Error::RoundFormTarget { .. })
    ));
    let s = single(&c, 13, 3);
    assert!(matches!(
        check(&s, &s.proof, &off),
        Err(Error::RoundFormTarget { .. })
    ));
}

/// The host's weight update for many claims at once is the sum of their
/// scaled `eq` tables, at every width and with repeated points.
#[test]
fn the_shift_claims_weight_is_the_sum_of_their_eq_tables() {
    for n in [1usize, 2, 5, 8] {
        let points: Vec<Vec<EE>> = (0..9u64).map(|j| point(n, 3 * j)).collect();
        let mut points = points;
        points.push(points[2].clone());
        let scales: Vec<EE> = (0..points.len() as u64)
            .map(|j| EE::from(1000 + 17 * j))
            .collect();
        let start: Vec<EE> = (0..1u64 << n).map(|i| EE::from(5 * i + 1)).collect();
        let mut want = start.clone();
        for (p, scale) in points.iter().zip(&scales) {
            for (w, e) in want.iter_mut().zip(eq_evals(p)) {
                *w += scale * e;
            }
        }
        let mut got = start;
        add_eq_sum(&mut got, &points, &scales).unwrap();
        assert_eq!(got, want, "{n} variables");
    }
}

// ───────────────────────────── round trips ───────────────────────────────────

/// ★ A chain under the round form proves and verifies at shrinks 1, 2 and 3,
/// first folds 4 and 6 and the uniform schedule, at blowups 2 and 4; no round
/// that commits a successor carries a successor block, round `r` carries
/// exactly `Q_r` openings, and the proof opens fewer elements than today's.
#[test]
fn the_round_form_round_trips_by_shrink_and_first_fold() {
    let n = 14;
    for lb in [1usize, 2] {
        for first in [Some(4), Some(6), None] {
            for shrink in 1..=3usize {
                let cfg = whir(lb, first, 4, shrink, n, 48);
                let s = single(&cfg, n, 5 + shrink as u64);
                check(&s, &s.proof, &cfg)
                    .unwrap_or_else(|e| panic!("lb {lb}, {first:?}, shrink {shrink}: {e}"));
                let queries = cfg.round_queries(n);
                let rounds = cfg.rounds(n);
                for (r, round) in s.proof.rounds.iter().enumerate() {
                    let (current, next) = opened(&round.openings);
                    assert_eq!(current, queries[r], "round {r}'s count");
                    assert_eq!(next, 0, "round {r} opens no successor");
                    assert_eq!(round.next_root.is_some(), r + 1 < rounds);
                }
                assert!(queries.windows(2).all(|w| w[1] <= w[0]), "{queries:?}");
                let today = single(&fri(lb, first, 4, n, 48), n, 5 + shrink as u64);
                assert!(s.proof.opened_elements() < today.proof.opened_elements());
            }
        }
    }
}

/// A chain under the round form with a final polynomial: the final round
/// queries its own count against the sent polynomial, and the tail is
/// sumchecks alone.
#[test]
fn a_final_polynomial_round_trips_under_the_round_form() {
    let n = 15;
    for shrink in [2usize, 3] {
        let mut cfg = whir(1, Some(4), 4, shrink, n, 40);
        cfg.format.final_poly = FinalPoly::vars(4).unwrap();
        let (final_r, rem) = cfg.final_round(n).expect("a final round");
        let s = single(&cfg, n, 9);
        check(&s, &s.proof, &cfg).unwrap();
        let queries = cfg.round_queries(n);
        let (current, next) = opened(&s.proof.rounds[final_r].openings);
        assert_eq!(current, queries[final_r]);
        assert_eq!(next, 1, "the polynomial rides the successor slot");
        assert!(rem <= 4);
        for round in &s.proof.rounds[final_r + 1..] {
            assert_eq!(opened(&round.openings), (0, 0));
        }
    }
}

/// Small domains where a round queries more positions than its tree has
/// leaves, so positions repeat: each repeat is its own claim with its own
/// power of `γ`, on both sides, and the proof verifies.
#[test]
fn a_repeated_position_is_its_own_claim_on_both_sides() {
    let n = 7;
    let cfg = whir(1, None, 2, 1, n, 64);
    let queries = cfg.round_queries(n);
    let domains = cfg.round_domain_logs(n);
    let schedule = cfg.schedule(n);
    // Round 1's tree has fewer leaves than its queries: repeats are certain.
    assert!(
        queries[1] > 1 << (domains[1] - schedule[1]),
        "{queries:?} {domains:?}"
    );
    let s = single(&cfg, n, 21);
    check(&s, &s.proof, &cfg).unwrap();
    let blocks = match &s.proof.rounds[1].openings {
        RoundOpenings::Extension(p) => &p.current,
        _ => panic!("round 1 is an extension round"),
    };
    let distinct: std::collections::BTreeSet<String> =
        blocks.iter().map(|b| format!("{:?}", b.values)).collect();
    assert!(distinct.len() < blocks.len(), "some block is opened twice");
}

/// The round form over a stack: grouped runs (one tree a polynomial and one
/// joint tree a run), both leaf orders under 7-round BLAKE3, a final
/// polynomial; each verifies, and a run's round 0 carries no successor block.
#[test]
fn grouped_and_joint_runs_round_trip_under_the_round_form() {
    type B = Blake3Whir;
    let n_stack = 12;
    let heights = [12usize, 12, 11, 12, 10, 12, 12];
    for trees in [Round0Trees::Each, Round0Trees::One] {
        for leaves in [LeafOrder::Natural, LeafOrder::Coset] {
            for final_poly in [FinalPoly::Off, FinalPoly::vars(3).unwrap()] {
                let mut cfg = fri(1, Some(4), 4, n_stack, 40);
                cfg.format.group = GroupFold::Indexed;
                cfg.format.leaves = leaves;
                cfg.format.round0 = trees;
                cfg.format.final_poly = final_poly;
                cfg.format.cap = CapPolicy::Fixed(2);
                let cfg = cfg.with_whir_rounds(3, 40, n_stack).unwrap();
                let label = format!("{trees:?} {leaves:?} {final_poly:?}");
                let r = stacked_round_trip::<B>(&heights, n_stack, &cfg, |_| {});
                r.unwrap_or_else(|e| panic!("{label}: {e:?}"));
            }
        }
    }
}

/// Commits `heights` as one stack under `cfg`, proves every column at its own
/// point, lets `tamper` change the proof, and verifies.
fn stacked_round_trip<B: WhirHash>(
    heights: &[usize],
    n_stack: usize,
    cfg: &ChainConfig,
    tamper: impl FnOnce(&mut crate::stacked_eval::StackedProof<F, Ext>),
) -> Result<(), Refusal> {
    let columns: Vec<Mle<F>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| mle(h, 5 + 31 * i as u64))
        .collect();
    let points: Vec<Vec<EE>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| point(h, 13 * i as u64))
        .collect();
    let values: Vec<EE> = columns
        .iter()
        .zip(&points)
        .map(|(c, p)| c.evaluate_in(p).unwrap())
        .collect();
    let layout = StackedLayout::build(heights, n_stack).unwrap();
    let committed =
        StackedCommitment::<F, B>::commit(layout.clone(), &borrow(&columns), None, cfg).unwrap();
    let roots = committed.roots();
    let domain = committed.domain().clone();
    let label = |mut t: B::Sponge| {
        t.append_bytes(b"round-form-stack");
        t
    };
    let mut proof = prove_stacked::<F, Ext, _, B, _>(
        &committed,
        &borrow(&columns),
        None,
        &Claimed::PerColumn(&points),
        &values,
        cfg,
        &mut label(B::sponge()),
    )
    .unwrap();
    for chain in &proof.polys {
        for (r, round) in chain.rounds.iter().enumerate() {
            if round.next_root.is_some() {
                assert_eq!(opened(&round.openings).1, 0, "round {r} opens no successor");
            }
        }
    }
    tamper(&mut proof);
    verify_traced::<F, Ext, _, B>(
        &proof,
        &layout,
        &roots,
        &Claimed::PerColumn(&points),
        &values,
        &domain,
        cfg,
        &mut label(B::sponge()),
    )
}

// ───────────────────────────── obligation 1: the shift claims ─────────────────

/// ★★ A FALSE EVALUATION, refused: the prover commits `f` and proves the
/// claim of `f' = f + δ·e_a` (another value at one cube point, so another
/// value at `z`) — its sumcheck and every successor are `f'`'s, honest for
/// `f'`; only round 0's openings are `f`'s. The fold values of those openings
/// are `f`'s, so the shift claims of round 0 are false for the successor, and
/// the verifier refuses.
///
/// ★ THE MUTANT: with the shift claims dropped on both sides (the round form
/// without its one new check) the same forgery is ACCEPTED — the out-of-domain
/// point and the later rounds bind the successor to itself, nothing binds it
/// to the committed polynomial's fold. The honest proof verifies under the
/// mutant too, so the mutant is a complete protocol that is not sound.
#[test]
fn a_false_evaluation_is_refused_and_accepted_only_without_the_shift_claims() {
    let n = 13;
    for (first, shrink) in [(Some(4), 3usize), (Some(6), 2), (None, 1)] {
        let cfg = whir(1, first, 4, shrink, n, 40);
        let f = mle(n, 77);
        let z = point(n, 4);
        let (commitment, domain) = commit::<F, H>(&f, &cfg, true).unwrap();
        // f' = f + δ at cube point a: f'(z) = f(z) + δ·eq(z, a).
        let a = 5usize;
        let delta = FE::from(12345u64);
        let mut forged = f.evals().to_vec();
        forged[a] += delta;
        let forged = Mle::new(forged).unwrap();
        let y_forged = forged.evaluate_in(&z).unwrap();
        assert_ne!(
            y_forged,
            f.evaluate_in(&z).unwrap(),
            "a false claim about f"
        );

        let forge = || {
            prove::<F, Ext, _, H>(&forged, &z, &commitment, &domain, &cfg, &mut sponge()).unwrap()
        };
        let check = |proof: &ChainProof<F, Ext>| {
            verify::<F, Ext, _, H>(
                proof,
                &commitment.root(),
                &z,
                y_forged,
                &domain,
                &cfg,
                &mut sponge(),
            )
        };
        let label = format!("{first:?} shrink {shrink}");
        assert!(check(&forge()).is_err(), "{label}: the forgery is refused");
        with(Fault::DropShiftClaims, || {
            assert_eq!(
                check(&forge()),
                Ok(()),
                "{label}: without the shift claims the forgery is accepted"
            );
            let honest = single(&cfg, n, 77);
            check_single_ok(&honest, &cfg);
        });
    }
}

/// ★★ A LATER-ROUND FORGERY, refused at every round: a cheating prover
/// commits, at round `r`, a successor tree encoding another polynomial than
/// the message its sumcheck carries on with (the message plus one at a cube
/// point), and answers that round's out-of-domain point from the message, as
/// an honest prover would. Only the next round's openings can tell: their
/// folds are the forged polynomial's, and the round form checks them as
/// claims on the message. Refused at every round, round 0's successor and
/// the later ones alike; the last tree (opened by the final round against
/// the final value) is refused too.
///
/// ★ THE MUTANT: with the shift claims dropped on both sides, every forgery
/// of a tree that a successor-committing round opens is ACCEPTED — nothing
/// else binds a later tree to the message.
#[test]
fn a_successor_that_is_not_the_message_is_refused_at_every_round() {
    use crate::whir_chain::round_form_faults::with_forged_successor;
    let n = 18;
    let cfg = whir(1, Some(4), 4, 3, n, 40);
    let rounds = cfg.rounds(n);
    assert!(
        rounds >= 5,
        "three successors past round 0's: {rounds} rounds"
    );
    let honest = single(&cfg, n, 33);
    check_single_ok(&honest, &cfg);
    for r in 0..rounds - 1 {
        let forged = with_forged_successor(r, || single(&cfg, n, 33));
        assert_ne!(
            forged.proof.rounds[r].next_root, honest.proof.rounds[r].next_root,
            "round {r}: another tree"
        );
        assert!(
            check(&forged, &forged.proof, &cfg).is_err(),
            "round {r}: refused"
        );
        // A tree the final round opens is checked against the final value,
        // claims or no claims; the earlier ones only through the claims.
        if r + 2 < rounds {
            with(Fault::DropShiftClaims, || {
                let forged = with_forged_successor(r, || single(&cfg, n, 33));
                assert_eq!(
                    check(&forged, &forged.proof, &cfg),
                    Ok(()),
                    "round {r}: without the shift claims the forgery is accepted"
                );
            });
        }
    }
}

fn check_single_ok(s: &Single, cfg: &ChainConfig) {
    assert_eq!(check(s, &s.proof, cfg), Ok(()));
}

/// Each convention the shift claims hold to is load-bearing: a verifier that
/// takes a query's point in the round's own domain instead of the folded one,
/// a point one squaring short, or every claim weighted by `γ` itself, refuses
/// an honest proof — per polynomial and over a grouped run.
#[test]
fn every_convention_of_a_shift_claim_is_load_bearing() {
    let n = 13;
    let cfg = whir(1, Some(4), 4, 3, n, 40);
    let s = single(&cfg, n, 31);
    check_single_ok(&s, &cfg);
    let mut grouped = fri(1, Some(4), 4, 12, 40);
    grouped.format.group = GroupFold::Indexed;
    let grouped = grouped.with_whir_rounds(3, 40, 12).unwrap();
    stacked_round_trip::<Blake3Whir>(&[12, 12, 11], 12, &grouped, |_| {}).unwrap();
    for fault in [
        Fault::ShiftPointUnfolded,
        Fault::ShiftPointOneSquaringShort,
        Fault::ReuseGammaPower,
    ] {
        with(fault, || {
            assert!(
                check(&s, &s.proof, &cfg).is_err(),
                "{fault:?}: a single chain"
            );
            assert!(
                stacked_round_trip::<Blake3Whir>(&[12, 12, 11], 12, &grouped, |_| {}).is_err(),
                "{fault:?}: a grouped run"
            );
        });
    }
}

/// A later round's opened block, changed in one value, is refused by its
/// tree; so is a round-0 block. The claims are only as good as the openings
/// they come from.
#[test]
fn a_tampered_opening_is_refused_at_every_queried_round() {
    let n = 14;
    let cfg = whir(1, Some(4), 4, 3, n, 40);
    let s = single(&cfg, n, 8);
    for r in 0..s.proof.rounds.len() {
        let mut proof = s.proof.clone();
        let touched = match &mut proof.rounds[r].openings {
            RoundOpenings::Base(p) => {
                p.current[0].values[1] += FE::one();
                true
            }
            RoundOpenings::Extension(p) if !p.current.is_empty() => {
                p.current[0].values[0] += EE::one();
                true
            }
            _ => false,
        };
        assert!(touched, "every round of this chain is queried");
        assert!(
            matches!(
                check(&s, &proof, &cfg),
                Err(Error::OpeningRejected { query: 0 })
            ),
            "round {r}"
        );
    }
}

// ───────────────────────────── obligation 2: the order ───────────────────────

/// A transcript that records each operation's kind, around a real one.
struct Recording<T> {
    inner: T,
    ops: Vec<Op>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Fe,
    Bytes(usize),
    SampleFe,
    SampleU64(u64),
}

impl<T: IsTranscript<Ext>> IsTranscript<Ext> for Recording<T> {
    fn append_field_element(&mut self, element: &EE) {
        self.ops.push(Op::Fe);
        self.inner.append_field_element(element);
    }
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.ops.push(Op::Bytes(new_bytes.len()));
        self.inner.append_bytes(new_bytes);
    }
    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
    fn sample_field_element(&mut self) -> EE {
        self.ops.push(Op::SampleFe);
        self.inner.sample_field_element()
    }
    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        self.ops.push(Op::SampleU64(upper_bound));
        self.inner.sample_u64(upper_bound)
    }
}

/// ★ THE ORDER OF A ROUND, pinned op by op on both sides: after round 0's
/// sumcheck, the successor's root, the out-of-domain point, its answer, the
/// out-of-domain batching challenge, round 0's `Q_0` positions over its
/// blocks, and only then the shift claims' `γ` — then round 1's first
/// sumcheck message. The prover's and the verifier's logs are one log.
/// A verifier that draws the positions before it absorbs the root refuses
/// the honest proof.
#[test]
fn the_order_of_a_round_form_round_is_pinned() {
    let n = 12;
    let cfg = whir(1, Some(4), 4, 3, n, 40);
    let f = mle(n, 2);
    let z = point(n, 2);
    let y = f.evaluate_in(&z).unwrap();
    let (commitment, domain) = commit::<F, H>(&f, &cfg, true).unwrap();
    let mut t = Recording {
        inner: sponge(),
        ops: Vec::new(),
    };
    let proof = prove::<F, Ext, _, H>(&f, &z, &commitment, &domain, &cfg, &mut t).unwrap();
    let prover = t.ops;
    let mut t = Recording {
        inner: sponge(),
        ops: Vec::new(),
    };
    verify::<F, Ext, _, H>(&proof, &commitment.root(), &z, y, &domain, &cfg, &mut t).unwrap();
    assert_eq!(prover, t.ops, "the prover's and the verifier's transcripts");

    let q0 = cfg.round_queries(n)[0];
    let leaves0 = 1u64 << (n + 1 - 4);
    let root = prover
        .iter()
        .position(|op| *op == Op::Bytes(32))
        .expect("the successor's root");
    // Round 0's sumcheck: four rounds of two values and a challenge.
    assert_eq!(prover[..root], [Op::Fe, Op::Fe, Op::SampleFe].repeat(4));
    let mut want = vec![Op::Bytes(32), Op::SampleFe, Op::Fe, Op::SampleFe];
    want.extend(std::iter::repeat_n(Op::SampleU64(leaves0), q0));
    want.push(Op::SampleFe);
    want.push(Op::Fe);
    assert_eq!(prover[root..root + want.len()], want[..]);

    let s = Single {
        z,
        y,
        commitment,
        domain,
        proof,
    };
    with(Fault::PositionsBeforeSuccessor, || {
        assert!(check(&s, &s.proof, &cfg).is_err());
    });
}

// ───────────────────────────── obligation 3: constants ───────────────────────

/// ★ A proof under another shrink, another form, or another target is refused
/// — each side reads its own config, never the proof's.
#[test]
fn a_proof_under_another_round_form_is_refused() {
    let n = 13;
    let configs = [
        fri(1, Some(4), 4, n, 40),
        whir(1, Some(4), 4, 1, n, 40),
        whir(1, Some(4), 4, 2, n, 40),
        whir(1, Some(4), 4, 3, n, 40),
    ];
    let proofs: Vec<Single> = configs.iter().map(|c| single(c, n, 12)).collect();
    for (i, s) in proofs.iter().enumerate() {
        for (j, c) in configs.iter().enumerate() {
            let r = check(s, &s.proof, c);
            if i == j {
                assert_eq!(r, Ok(()), "{i} under its own config");
            } else {
                assert!(r.is_err(), "proof {i} under config {j}");
            }
        }
    }
    // The same shrink with a union of one more round: the later counts move.
    let c = &configs[3];
    let w = match c.format.rounds {
        RoundForm::Whir(w) => w,
        RoundForm::Fri => unreachable!(),
    };
    let wider = ChainConfig {
        format: ChainFormat {
            rounds: RoundForm::Whir(WhirRounds::new(3, 40, w.union_rounds() + 3).unwrap()),
            ..c.format
        },
        ..*c
    };
    assert!(check(&proofs[3], &proofs[3].proof, &wider).is_err());
}

/// ★ ONE QUERY TOO FEW: a proof whose rounds after round 0 query `Q_r − 1`
/// positions (a cheating prover) is refused by count, and a round carrying a
/// successor block is refused by name. THE MUTANT: a verifier that reads a
/// round's count from the proof accepts the short proof.
#[test]
fn a_round_short_of_its_count_is_refused() {
    let n = 14;
    let cfg = whir(1, Some(4), 4, 3, n, 40);
    let short = with(Fault::ProverQueriesShort, || single(&cfg, n, 44));
    let queries = cfg.round_queries(n);
    assert_eq!(opened(&short.proof.rounds[1].openings).0, queries[1] - 1);
    assert_eq!(
        check(&short, &short.proof, &cfg),
        Err(Error::QueryCountMismatch {
            expected: queries[1],
            got: queries[1] - 1
        })
    );
    with(Fault::QueriesFromProof, || {
        assert_eq!(
            check(&short, &short.proof, &cfg),
            Ok(()),
            "the mutant accepts"
        );
    });

    // A successor block where the round form has none.
    let s = single(&cfg, n, 44);
    let mut proof = s.proof.clone();
    match &mut proof.rounds[1].openings {
        RoundOpenings::Extension(p) => {
            let extra = p.current[0].clone();
            p.next.push(extra);
        }
        _ => panic!("round 1 is an extension round"),
    }
    assert_eq!(
        check(&s, &proof, &cfg),
        Err(Error::SuccessorOpened { round: 1 })
    );
    // An opening dropped from an honest proof.
    let mut proof = s.proof.clone();
    match &mut proof.rounds[2].openings {
        RoundOpenings::Extension(p) => {
            p.current.pop();
        }
        _ => panic!("round 2 is an extension round"),
    }
    assert!(matches!(
        check(&s, &proof, &cfg),
        Err(Error::QueryCountMismatch { .. })
    ));
}

/// A grouped run's round 0 under the round form carrying a successor block
/// is refused by the grouped shape check.
#[test]
fn a_grouped_round_zero_with_a_successor_block_is_refused() {
    let mut cfg = fri(1, Some(4), 4, 12, 40);
    cfg.format.group = GroupFold::Indexed;
    let cfg = cfg.with_whir_rounds(3, 40, 12).unwrap();
    let r = stacked_round_trip::<Blake3Whir>(&[12, 12, 11], 12, &cfg, |proof| {
        match &mut proof.polys[0].rounds[0].openings {
            RoundOpenings::BaseGrouped(p) => {
                let extra = CosetOpening {
                    values: vec![EE::zero(); 16],
                    proof: crypto::merkle_tree::proof::Proof {
                        merkle_path: Vec::new(),
                    },
                };
                p.next.push(extra);
            }
            other => panic!("round 0 of a run of three is grouped: {:?}", opened(other)),
        }
    });
    assert!(
        matches!(
            r,
            Err(Refusal::Chain {
                cause: Error::GroupedShape(crate::whir_chain::GroupedShape::SuccessorBlocks { .. }),
                ..
            })
        ),
        "{r:?}"
    );
}

// ───────────────────────────── obligation 4: the final weight ────────────────

/// The final weight sums every out-of-domain claim and every shift claim:
/// leaving either out refuses an honest proof.
#[test]
fn the_final_weight_takes_every_claim() {
    let n = 13;
    let cfg = whir(1, Some(4), 4, 3, n, 40);
    let s = single(&cfg, n, 61);
    check_single_ok(&s, &cfg);
    for fault in [Fault::ShiftClaimsNotInWeight, Fault::OodNotInWeight] {
        with(fault, || {
            assert_eq!(
                check(&s, &s.proof, &cfg),
                Err(Error::EvaluationMismatch),
                "{fault:?}"
            );
        });
    }
}

/// ★ KAT: a round-form proof's bytes, pinned (shrink 2, first fold 4, a
/// `2^12` polynomial, keccak trees, cap 2). Any change to the order, the
/// points, the powers or the counts moves it.
#[test]
fn a_round_form_proof_is_pinned() {
    let n = 12;
    let mut cfg = whir(1, Some(4), 4, 2, n, 40);
    cfg.format.cap = CapPolicy::Fixed(2);
    let s = single(&cfg, n, 1);
    check_single_ok(&s, &cfg);
    let mut h = crypto::hash::blake3::Blake3Chain::with_rounds(
        crypto::hash::blake3::BLAKE3_STANDARD_ROUNDS,
    );
    h.update(&s.commitment.root());
    h.update(
        rkyv::to_bytes::<rkyv::rancor::Error>(&s.proof)
            .unwrap()
            .as_slice(),
    );
    let digest: String = h
        .finalize_digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("ROUND FORM KAT {digest}");
    assert_eq!(
        digest,
        "15a6f6b57f2125b070ecf3ae55438c676732ad0812b8fa50468477eab6b574ca"
    );
}

// ───────────────────────────── obligation 5: proven bits ─────────────────────

/// Per-phase proven bits, ported from a reference calculator's formulas (its
/// reading of BCHKS25 Thm 1.5 / 4.6, `m = max(⌈√ρ/(2η)⌉, 3)`, `η = 1/300`,
/// the Johnson list bound, and ACFY24's round-by-round terms): the fold terms
/// (mutual correlated agreement at each code a round folds through), the
/// out-of-domain terms, each round's shift term and the final round's, for a
/// grouped chain of `m = 2` index variables over a `2^27` stack. The
/// schedule, shrinks and counts are the CONFIG's, so the counts the code
/// derives are what is priced.
struct Phases {
    fold: f64,
    ood: f64,
    /// `shift[i - 1]`: round `i`'s, for `i` in `1..=last`.
    shift: Vec<f64>,
    final_round: f64,
}

fn phase_bits(cfg: &ChainConfig, n: usize, m: usize, g: f64) -> Phases {
    const ETA: f64 = 1.0 / 300.0;
    let p = 18446744069414584321f64;
    let log_e = 3.0 * p.log2();
    let lsize = |rho: f64| (1.0 / (2.0 * ETA * rho.sqrt())).floor();
    let fold_bits = |rho: f64, log_n: f64| {
        let sr = rho.sqrt();
        let gamma = 1.0 - sr - ETA;
        let mm = (sr / (2.0 * ETA)).ceil().max(3.0) + 0.5;
        let a = (2.0 * mm.powi(5) + 3.0 * mm * gamma * rho) / (3.0 * rho.powf(1.5))
            * 2f64.powf(log_n)
            + mm / sr;
        log_e - (a + 3.0 * lsize(rho)).log2()
    };
    let shift_bits = |rho: f64, q: usize| q as f64 * -(rho.sqrt() + ETA).log2() + g;
    let ood_bits = |rho: f64, n_i: f64| {
        let l = lsize(rho);
        log_e - (l * l * 2f64.powf(n_i) / 2.0).log2()
    };

    let schedule = cfg.schedule(n);
    let queries = cfg.round_queries(n);
    let mut ks = schedule.clone();
    ks[0] += m;
    let mut dims = vec![n + m];
    for &k in &ks {
        dims.push(dims[dims.len() - 1] - k);
    }
    let mut d = vec![n + m + cfg.log_blowup];
    for (r, &k) in schedule.iter().enumerate().take(schedule.len() - 1) {
        let s = cfg.shrink_of(k) + if r == 0 { m } else { 0 };
        d.push(d[d.len() - 1] - s);
    }
    let lbr: Vec<usize> = (0..ks.len()).map(|r| d[r] - dims[r]).collect();
    let last = cfg.final_round(n).map_or(ks.len() - 1, |(r, _)| r);
    let rho = |r: usize| 2f64.powi(-(lbr[r] as i32));
    let mut fold = f64::INFINITY;
    for i in 0..=last {
        for s in 1..=ks[i] {
            fold = fold.min(fold_bits(rho(i), (d[i] - s) as f64));
        }
    }
    let ood = (1..=last)
        .map(|i| ood_bits(rho(i), dims[i] as f64))
        .fold(f64::INFINITY, f64::min);
    let shift = (1..=last)
        .map(|i| shift_bits(rho(i - 1), queries[i - 1]))
        .collect();
    Phases {
        fold,
        ood,
        shift,
        final_round: shift_bits(rho(last), queries[last]),
    }
}

/// The production shape the calculator reads: blowup 2 or 4, `first4` or
/// `first6` then four a round, a `2^27` stack, the final polynomial at 9
/// variables, 18 query-grind bits, 128 bits.
fn production(lb: usize, k0: usize, shrink: Option<usize>, g: u8) -> ChainConfig {
    let mut cfg =
        ChainConfig::with_security_folds(lb, 4, folds(Some(k0)), 27, 128, GrindBits::query_only(g));
    cfg.format.final_poly = FinalPoly::vars(9).unwrap();
    match shrink {
        Some(s) => cfg.with_whir_rounds(s, 128, 27).unwrap(),
        None => cfg,
    }
}

fn close(got: f64, want: f64, places: i32) -> bool {
    (got - want).abs() <= 0.5 * 10f64.powi(-places) + 1e-9
}

/// ★★ PROVEN BITS PER PHASE, pinned against the reference calculator's
/// printed values, and EVERY round of the round form — each shift term, the
/// final round's, the fold and the out-of-domain terms — at or over 128 bits
/// per proof (bits are counted per proof: 128.34 on this reading, as the
/// Poseidon1 pipeline reports).
#[test]
fn every_phase_of_the_round_form_keeps_the_per_proof_target() {
    // (label, config, fold, ood, min shift, final): the calculator's printed
    // values.
    let cases: [(&str, ChainConfig, f64, f64, f64, f64); 6] = [
        (
            "today",
            production(1, 6, None, 18),
            128.344,
            156.5,
            130.946,
            130.946,
        ),
        (
            "C",
            production(1, 6, Some(3), 18),
            128.344,
            153.5,
            130.911,
            133.519,
        ),
        (
            "G",
            production(2, 6, Some(3), 18),
            128.393,
            152.5,
            130.907,
            131.738,
        ),
        (
            "J",
            production(2, 4, Some(3), 18),
            128.393,
            152.5,
            130.907,
            133.519,
        ),
        (
            "L",
            production(2, 4, Some(2), 18),
            128.393,
            151.5,
            130.907,
            134.491,
        ),
        (
            "M",
            production(1, 4, Some(3), 18),
            128.344,
            153.5,
            130.907,
            131.760,
        ),
    ];
    for (label, cfg, fold, ood, shift, final_round) in cases {
        let b = phase_bits(&cfg, 27, 2, 18.0);
        let min_shift = b.shift.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(close(b.fold, fold, 3), "{label}: fold {} vs {fold}", b.fold);
        assert!(close(b.ood, ood, 1), "{label}: ood {} vs {ood}", b.ood);
        assert!(
            close(min_shift, shift, 3),
            "{label}: shift {min_shift} vs {shift}"
        );
        assert!(
            close(b.final_round, final_round, 3),
            "{label}: final {} vs {final_round}",
            b.final_round
        );
        assert!(
            b.fold >= 128.0 && b.ood >= 128.0 && b.final_round >= 128.0,
            "{label}"
        );
        for (i, s) in b.shift.iter().enumerate() {
            assert!(*s >= 128.0, "{label}: round {}'s shift term {s}", i + 1);
        }
    }
}

/// THE MUTANT: a count rule that overstates a later round's rate (its domain
/// taken as unshrunk) leaves a round under the per-proof target, which the
/// bits check above catches.
#[test]
fn a_rate_that_ignores_the_shrink_drops_a_round_below_the_target() {
    let cfg = production(1, 4, Some(3), 18);
    let b = with(Fault::RateIgnoresShrink, || phase_bits(&cfg, 27, 2, 18.0));
    let worst = b.shift.iter().copied().fold(b.final_round, f64::min);
    assert!(worst < 128.0, "{worst}");
}

// ───────────────────────────── the round form's grinds and OOD domain ────────

/// ★ A ROUND-FORM ROUND CHECKS ITS NONCES' WORK. Under nonzero grinds (5
/// out-of-domain bits, 6 query bits) an honest proof verifies; a prover that
/// skips the work at the round form's rounds — its out-of-domain nonces alone,
/// or its query nonces alone — is refused by that grind's bits; THE MUTANT, a
/// verifier that absorbs those nonces unchecked, accepts either cheat. A nonce
/// changed in an honest proof is refused too.
#[test]
fn a_round_form_round_checks_its_nonces_work() {
    let n = 13;
    let grind = GrindBits {
        folding: 0,
        ood: 5,
        query: 6,
    };
    let cfg = ChainConfig::with_security_folds(1, 4, folds(Some(4)), n, 40, grind)
        .with_whir_rounds(3, 40, n)
        .unwrap();
    check_round_form(&cfg).unwrap();
    let s = single(&cfg, n, 17);
    check_single_ok(&s, &cfg);
    let rounds = s.proof.rounds.len();
    assert!(rounds >= 3, "two rounds that commit a successor at least");
    for (fault, bits) in [
        (Fault::SpoilRoundFormOodNonces, 5u8),
        (Fault::SpoilRoundFormQueryNonces, 6),
    ] {
        let cheat = with(fault, || single(&cfg, n, 17));
        assert_eq!(
            check(&cheat, &cheat.proof, &cfg),
            Err(Error::GrindingRejected { bits }),
            "{fault:?}"
        );
        with(Fault::RoundFormNonceUnchecked, || {
            assert_eq!(
                check(&cheat, &cheat.proof, &cfg),
                Ok(()),
                "{fault:?}: the mutant accepts"
            );
        });
    }
    // An honest proof with one round-form nonce moved: refused at that round
    // (a moved nonce fails the work but for a chance of 2^-bits; one of the
    // first sixteen candidates past it fails).
    for r in 0..rounds - 1 {
        for slot in ["ood", "query"] {
            let refused = (1u64..=16).any(|d| {
                let mut proof = s.proof.clone();
                match slot {
                    "ood" => proof.rounds[r].nonces.ood ^= d,
                    _ => proof.rounds[r].nonces.query ^= d,
                }
                matches!(check(&s, &proof, &cfg), Err(Error::GrindingRejected { .. }))
            });
            assert!(refused, "round {r} {slot}");
        }
    }
}

/// ★ A chain that runs more rounds than its config's union bound is refused
/// on both sides: a config sized for chains of 8 variables proves no chain
/// of 13, and reads no proof of one (its counts would be charged too few
/// rounds).
#[test]
fn a_chain_past_the_union_bound_is_refused() {
    let small = whir(1, Some(4), 4, 3, 8, 40);
    check_round_form(&small).unwrap();
    let n = 13;
    let w = match small.format.rounds {
        RoundForm::Whir(w) => w,
        RoundForm::Fri => unreachable!(),
    };
    assert!(small.rounds(n) > w.union_rounds());
    let f = mle(n, 3);
    let z = point(n, 3);
    let (commitment, domain) = commit::<F, H>(&f, &small, true).unwrap();
    assert!(matches!(
        prove::<F, Ext, _, H>(&f, &z, &commitment, &domain, &small, &mut sponge()),
        Err(Error::RoundFormUnion { .. })
    ));
    let sized = whir(1, Some(4), 4, 3, n, 40);
    let s = single(&sized, n, 3);
    check_single_ok(&s, &sized);
    assert!(matches!(
        check(&s, &s.proof, &small),
        Err(Error::RoundFormUnion { .. })
    ));
}

/// A transcript that hands out `forced` at its `at`-th field-element draw and
/// the inner transcript's draws otherwise (the inner one still draws).
struct Forced<T> {
    inner: T,
    draws: usize,
    at: usize,
    forced: EE,
}

impl<T: IsTranscript<Ext>> IsTranscript<Ext> for Forced<T> {
    fn append_field_element(&mut self, element: &EE) {
        self.inner.append_field_element(element);
    }
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.inner.append_bytes(new_bytes);
    }
    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
    fn sample_field_element(&mut self) -> EE {
        let drawn = self.inner.sample_field_element();
        self.draws += 1;
        if self.draws == self.at {
            self.forced
        } else {
            drawn
        }
    }
    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        self.inner.sample_u64(upper_bound)
    }
}

/// ★ The out-of-domain point is checked against the SUCCESSOR's domain, the
/// larger one under the round form (round `r`'s domain squared `shrink`
/// times, not `k` times): a verifier whose draw lands on a point of the
/// successor's domain that the folded domain does not hold refuses it as in
/// the domain. THE MUTANT: checked against the folded domain, the same point
/// passes that check (and the proof fails later, for another cause).
#[test]
fn the_out_of_domain_point_is_kept_off_the_successors_domain() {
    let n = 12;
    let cfg = whir(1, Some(4), 4, 3, n, 40);
    let s = single(&cfg, n, 2);
    check_single_ok(&s, &cfg);
    // Round 0's domain is 2^(n+1); its successor's 2^(n+1−3), its folded
    // domain 2^(n+1−4). The successor's generator lies off the folded domain.
    let successor = Domain::<F>::new(n + 1 - 3).unwrap();
    let z = successor.generator().to_extension::<Ext>();
    let mut folded_power = z;
    for _ in 0..(n + 1 - 4) {
        folded_power = folded_power.square();
    }
    assert_ne!(folded_power, EE::one(), "not in the folded domain");
    // Round 0's out-of-domain point is the fifth draw: four sumcheck
    // challenges first (`the_order_of_a_round_form_round_is_pinned`).
    let rigged = |s: &Single| {
        verify::<F, Ext, _, H>(
            &s.proof,
            &s.commitment.root(),
            &s.z,
            s.y,
            &s.domain,
            &cfg,
            &mut Forced {
                inner: sponge(),
                draws: 0,
                at: 5,
                forced: z,
            },
        )
    };
    assert_eq!(rigged(&s), Err(Error::OodPointInDomain));
    with(Fault::OodAgainstFoldedDomain, || {
        let r = rigged(&s);
        assert!(
            r.is_err() && r != Err(Error::OodPointInDomain),
            "the mutant passes the point: {r:?}"
        );
    });
}

/// ★ The card builds the claims' half tables from the points' base values only
/// when every point is `pow(y)` with `y` in the base field
/// (`gpu::pow_point_ys`): the round form's shift points are, and give back
/// their `y`s; a point with one coordinate moved, a point whose `y` is off
/// the base field (its powers still in order), or one such point among
/// honest ones, is refused — those claims keep today's kernel.
#[test]
fn only_pow_points_of_base_values_take_the_card_halves() {
    use crate::gpu::pow_point_ys;
    use crate::whir_chain::ood_point;
    let n = 13;
    let folded = Domain::<F>::new(15).unwrap();
    let g = folded.generator();
    let positions = [0usize, 1, 7, 7, 4095, 32767];
    let ys: Vec<FE> = positions.iter().map(|&q| g.pow(q as u64)).collect();
    let points: Vec<Vec<EE>> = ys
        .iter()
        .map(|y| ood_point(&y.to_extension::<Ext>(), n))
        .collect();
    let want: Vec<u64> = ys.iter().map(|y| y.canonical()).collect();
    assert_eq!(pow_point_ys(&points), Some(want));
    assert_eq!(pow_point_ys::<Ext>(&[]), Some(Vec::new()));
    // One coordinate moved.
    let mut moved = points.clone();
    moved[2][5] += EE::one();
    assert_eq!(pow_point_ys(&moved), None);
    let mut last = points.clone();
    last[0][n - 1] += EE::one();
    assert_eq!(pow_point_ys(&last), None);
    // A `y` off the base field, its powers in order.
    let off = EE::new([FE::from(3u64), FE::from(1u64), FE::zero()]);
    let mut ext = points.clone();
    ext.push(ood_point(&off, n));
    assert_eq!(pow_point_ys(&ext), None);
    // Not the extension field at all.
    let base_points: Vec<Vec<FE>> = ys.iter().map(|y| ood_point(y, n)).collect();
    assert_eq!(pow_point_ys(&base_points), None);
}

// ───────────────────────────── caps by Q ─────────────────────────────────────

/// ★ CAPS BY Q, the counts: `cap_paid_by` is `⌈log2 Q⌉` binary levels
/// (`⌈log4 Q⌉` at arity 4), and at the production shape (blowup 2, first fold
/// 4, shrink 3, a `2^27` stack, 18 query-grind bits, cap 8) each tree's cap
/// is the design model's compression optimum, 8 · 7 · 7 · 6 · 6 over the
/// queried rounds — against 8 at every tree without it. Round 0's is today's
/// 8. The fold word differs by exactly [`FOLD_WORD_WHIR_CAPS_BY_Q`].
#[test]
fn caps_by_q_hold_each_tree_to_what_its_openings_pay_for() {
    use crate::whir_chain::{FOLD_WORD_WHIR_CAPS_BY_Q, cap_paid_by};
    for (q, want) in [
        (0usize, 0usize),
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 2),
        (5, 3),
        (229, 8),
    ] {
        assert_eq!(cap_paid_by(q, 2), want, "Q {q}");
    }
    assert_eq!(cap_paid_by(256, 2), 8);
    assert_eq!(cap_paid_by(257, 2), 9);
    assert_eq!(cap_paid_by(229, 4), 4);
    assert_eq!(cap_paid_by(16, 4), 2);
    assert_eq!(cap_paid_by(17, 4), 3);

    let mut plain = production(1, 4, Some(3), 18);
    plain.format.cap = CapPolicy::Fixed(8);
    let by_q = plain.with_caps_by_queries().expect("a round-form config");
    check_round_form(&by_q).unwrap();
    assert_eq!(
        by_q.fold_word(),
        plain.fold_word() | FOLD_WORD_WHIR_CAPS_BY_Q
    );
    assert_ne!(by_q.fold_word(), plain.fold_word());
    let (final_r, _) = by_q.final_round(27).expect("a final round");
    let queried = final_r + 1;
    assert_eq!(&by_q.round_queries(27)[..queried], &[229, 114, 76, 57, 46]);
    assert_eq!(&by_q.tree_caps(27)[..queried], &[8, 7, 7, 6, 6]);
    assert_eq!(&plain.tree_caps(27)[..queried], &[8, 8, 8, 8, 8]);
    // The policy stays the ceiling: a lower fixed cap is never raised.
    let mut low = by_q;
    low.format.cap = CapPolicy::Fixed(6);
    assert_eq!(&low.tree_caps(27)[..queried], &[6, 6, 6, 6, 6]);
    // The FRI-style form has no caps by Q.
    assert!(fri(1, Some(4), 4, 27, 128).with_caps_by_queries().is_none());
}

/// ★ A proof under caps by Q verifies, carries each tree's cap at its own
/// height, and is refused under the same round form without caps by Q (and
/// the other way round). THE MUTANT: a verifier that takes the policy's caps
/// ([`Fault::CapsIgnoreQueries`]) refuses the honest proof — the rule is load-
/// bearing on the verifier's side.
#[test]
fn a_proof_under_caps_by_q_is_refused_without_them() {
    let n = 14;
    let mut plain = whir(1, Some(4), 4, 3, n, 48);
    plain.format.cap = CapPolicy::Fixed(8);
    let by_q = plain.with_caps_by_queries().unwrap();
    let caps = by_q.tree_caps(n);
    let plain_caps = plain.tree_caps(n);
    assert_ne!(caps, plain_caps, "the shape moves at least one cap");
    let s_q = single(&by_q, n, 71);
    let s_plain = single(&plain, n, 71);
    check_single_ok(&s_q, &by_q);
    check_single_ok(&s_plain, &plain);
    assert!(
        check(&s_q, &s_q.proof, &plain).is_err(),
        "caps by Q under plain"
    );
    assert!(
        check(&s_plain, &s_plain.proof, &by_q).is_err(),
        "plain under caps by Q"
    );
    with(Fault::CapsIgnoreQueries, || {
        assert!(
            check(&s_q, &s_q.proof, &by_q).is_err(),
            "the mutant refuses"
        );
    });
    // A grouped run of BLAKE3 trees in the coset order, one joint tree a run.
    let mut grouped = fri(1, Some(4), 4, 12, 40);
    grouped.format.group = GroupFold::Indexed;
    grouped.format.leaves = LeafOrder::Coset;
    grouped.format.round0 = Round0Trees::One;
    grouped.format.cap = CapPolicy::Fixed(8);
    let grouped = grouped
        .with_whir_rounds(3, 40, 12)
        .unwrap()
        .with_caps_by_queries()
        .unwrap();
    stacked_round_trip::<Blake3Whir>(&[12, 12, 11, 12], 12, &grouped, |_| {}).unwrap();
}

// ───────────────────────────── the card ──────────────────────────────────────

/// Runs `f` with every WHIR commit, re-encode, fold and opening declined by
/// the card: the host arm of a card parity test, in the same binary.
#[cfg(feature = "cuda")]
fn on_the_host<R>(f: impl FnOnce() -> R) -> R {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            crate::gpu::force_host_whir_commit(false);
            crate::gpu::force_shared_open_declined(false);
        }
    }
    crate::gpu::force_host_whir_commit(true);
    crate::gpu::force_shared_open_declined(true);
    let _restore = Restore;
    f()
}

/// A value's rkyv bytes.
#[cfg(feature = "cuda")]
macro_rules! bytes {
    ($value:expr) => {
        rkyv::to_bytes::<rkyv::rancor::Error>($value)
            .unwrap()
            .to_vec()
    };
}

/// Commits, proves and verifies a stack under `cfg`, returning the proof's
/// bytes.
#[cfg(feature = "cuda")]
fn stacked_bytes<B: WhirHash>(heights: &[usize], n_stack: usize, cfg: &ChainConfig) -> Vec<u8> {
    let mut out = Vec::new();
    stacked_round_trip::<B>(heights, n_stack, cfg, |proof| out = bytes!(&*proof))
        .expect("the proof verifies");
    out
}

/// ★★ A ROUND-FORM PROOF ON THE CARD IS THE HOST'S, BYTE FOR BYTE: a chain
/// over a `2^16` polynomial (keccak trees, first fold 4, shrink 3) and a
/// grouped stack of `2^16` polynomials (7-round BLAKE3, the coset leaf order,
/// one joint round-0 tree a run, a final polynomial, shrink 2 and 3) proved
/// with the commits, the openings, the successors' encodes and the shift
/// claims' weights on the card, and again with the card declining all of it.
/// The card arm encoded every round-form successor there (none on the host)
/// and added the claims in batches; both proofs verify. Box tier: one
/// `#[test]`, since the arms switch process-wide overrides.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a card; box tier"]
fn a_round_form_proof_on_the_card_is_the_hosts_byte_for_byte() {
    let n = 16;
    let cfg = whir(1, Some(4), 4, 3, n, 48);
    let host = on_the_host(|| single(&cfg, n, 3));
    check_single_ok(&host, &cfg);
    crate::gpu::reset_call_counters();
    let card = single(&cfg, n, 3);
    assert!(
        card.commitment.codeword().on_device(),
        "the card took the commit"
    );
    let successors = cfg.rounds(n) as u64 - 1;
    assert_eq!(
        crate::gpu::round_form_host_encodes(),
        0,
        "no successor on the host"
    );
    assert_eq!(
        crate::gpu::encoded_successors(),
        successors,
        "every successor on the card"
    );
    assert_eq!(
        crate::gpu::shift_claim_batches(),
        successors,
        "one batch a successor"
    );
    check_single_ok(&card, &cfg);
    assert_eq!(bytes!(&card.proof), bytes!(&host.proof), "a single chain");
    assert_eq!(card.commitment.root(), host.commitment.root());

    let heights = [16usize, 16, 15, 16, 14, 16, 16];
    for shrink in [2usize, 3] {
        let mut stack = fri(1, Some(4), 4, 16, 48);
        stack.format.group = GroupFold::Indexed;
        stack.format.leaves = LeafOrder::Coset;
        stack.format.round0 = Round0Trees::One;
        stack.format.final_poly = FinalPoly::vars(5).unwrap();
        stack.format.cap = CapPolicy::Fixed(3);
        let stack = stack.with_whir_rounds(shrink, 48, 16).unwrap();
        let host = on_the_host(|| stacked_bytes::<Blake3Whir>(&heights, 16, &stack));
        crate::gpu::reset_call_counters();
        let card = stacked_bytes::<Blake3Whir>(&heights, 16, &stack);
        assert_eq!(crate::gpu::round_form_host_encodes(), 0, "shrink {shrink}");
        assert!(crate::gpu::shift_claim_batches() > 0, "shrink {shrink}");
        assert_eq!(card, host, "a grouped stack at shrink {shrink}");
    }
}

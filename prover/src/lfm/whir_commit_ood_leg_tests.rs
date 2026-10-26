//! The block leaf's group opening with its commit-time answers
//! (`stacked_eval::verify_ood`), as a machine leg under Poseidon1, gated
//! against the host's `stacked_eval::prove_mapped_ood` proofs: the point
//! drawn in-guest from the group's root cells is the host's
//! (`multilinear_block::commit_ood_point`), an honest opening executes, a
//! false or traded answer has no execution, and each half of the batching —
//! the answers on the claims, their `eq` on the weights — is load-bearing.
//!
//! Under each of the leaf's query checks: V1 on a small chain, and V1 and
//! lean on the production's chain shape (first fold 6, the automatic 4-ary
//! cap, coset leaves, enough queries that the lean check's tail rounds run).

use crypto::fiat_shamir::p1_transcript::P1Transcript;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{Claimed, OodClaims, StackedCommitment, StackedProof};
use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, LeafOrder, WhirFolds,
};
use multilinear::whir_commit::Commitment;
use multilinear::whir_hash::P1Whir;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::commitment_to_digest;
use super::builder::{Cell, Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapHash;
use super::executor::execute;
use super::hash::HasherKind;
use super::validator::validate;
use super::whir_block::emit_commit_ood_point;
use super::whir_chain::{ChainShape, LeafQuery, RoundStorage, try_push_round_words};
use super::whir_stacked::{
    OodWires, StackedChainWires, emit_stacked_verify_chains, ood_faults, stacked_verify_cost_ood,
};
use super::whir_transcript::{SpongeEntry, WhirTranscript};
use super::word::{LfmWord, ext_word, word_as_ext};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// The group's index in the block: a constant of its point's prefix.
const GROUP: usize = 2;

/// Mixed heights with gaps, seven polynomials of ten variables: under
/// `Indexed` a run of four and a run of three.
const HEIGHTS: [usize; 14] = [8, 9, 7, 9, 6, 9, 9, 8, 5, 9, 9, 4, 9, 7];
const N_STACK: usize = 10;

/// A chain shape and the leaf's query check over it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Case {
    /// A small chain: first fold 3, folds of 2, three queries; V1.
    Small,
    /// The production's chain shape at this stack — first fold 6, folds of 4,
    /// blowup 2, the automatic 4-ary cap, coset leaves — at 96 queries, under
    /// the leaf's query check `LeafQuery`.
    Production(LeafQuery),
}

const CASES: [Case; 3] = [
    Case::Small,
    Case::Production(LeafQuery::V1),
    Case::Production(LeafQuery::Lean),
];

fn config(case: Case, group: GroupFold) -> ChainConfig {
    match case {
        Case::Small => ChainConfig {
            log_blowup: 2,
            log_folding: 2,
            num_queries: 3,
            grind: GrindBits::query_only(4),
            format: ChainFormat {
                folds: WhirFolds::First(FirstFold::new(3).expect("a tested first fold")),
                group,
                ..ChainFormat::DEFAULT
            },
        },
        Case::Production(_) => ChainConfig {
            log_blowup: 1,
            log_folding: 4,
            num_queries: 96,
            grind: GrindBits::query_only(4),
            format: ChainFormat {
                folds: WhirFolds::First(FirstFold::new(6).expect("a tested first fold")),
                arity4_cap: CapPolicy::Auto,
                leaves: LeafOrder::Coset,
                group,
                ..ChainFormat::DEFAULT
            },
        },
    }
}

fn query(case: Case) -> LeafQuery {
    match case {
        Case::Small => LeafQuery::V1,
        Case::Production(query) => query,
    }
}

fn fee(v: u64) -> FEE {
    FEE::new([
        FE::from(v.wrapping_mul(6364136223846793005) >> 11),
        FE::from(v ^ 0xA5A5),
        FE::from(v.wrapping_add(7)),
    ])
}

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..1u64 << num_vars)
            .map(|i| {
                FE::from(
                    i.wrapping_mul(0x2545_F491_4F6C_DD1D)
                        .wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15))
                        >> 13,
                )
            })
            .collect(),
    )
    .expect("a cube")
}

/// A group committed and opened on the host with its answers, every column
/// at a point of its own.
struct Fixture {
    case: Case,
    cfg: ChainConfig,
    layout: StackedLayout,
    points: Vec<Vec<FEE>>,
    values: Vec<FEE>,
    roots: Vec<Commitment>,
    claims: OodClaims<E>,
    proof: StackedProof<F, E>,
    domain: Domain<F>,
    shape: ChainShape,
}

fn fixture(case: Case, group: GroupFold) -> Fixture {
    let cfg = config(case, group);
    let layout = StackedLayout::build(&HEIGHTS, N_STACK).expect("the layout packs");
    assert_eq!(layout.num_polys(), 7, "the stack this test means");
    let columns: Vec<Mle<F>> = HEIGHTS
        .iter()
        .enumerate()
        .map(|(c, &h)| column(h, 17 + 13 * c as u64))
        .collect();
    let borrowed = multilinear::stacking::borrow(&columns);
    let points: Vec<Vec<FEE>> = HEIGHTS
        .iter()
        .enumerate()
        .map(|(c, &h)| (0..h).map(|i| fee(0x5eed + (c * 37 + i) as u64)).collect())
        .collect();
    let values: Vec<FEE> = columns
        .iter()
        .zip(&points)
        .map(|(c, p)| c.evaluate_in::<E>(p).expect("a column takes its own point"))
        .collect();
    let stacked = StackedCommitment::<F, P1Whir>::commit(layout.clone(), &borrowed, None, &cfg)
        .expect("the group commits");
    let roots = stacked.roots();
    let z0 =
        stark::multilinear_block::commit_ood_point::<E, P1Transcript>(GROUP, &layout, &roots, &cfg)
            .expect("the group's point");
    let (claims, _) =
        multilinear::stacked_eval::ood_claims::<F, E, _>(&layout, &borrowed, None, &z0)
            .expect("the answers");
    let proof = multilinear::stacked_eval::prove_mapped_ood::<F, E, _, P1Whir, _>(
        &stacked,
        &borrowed,
        None,
        &Claimed::PerColumn(&points),
        &values,
        Some(&claims),
        &cfg,
        &mut P1Transcript::new(),
    )
    .expect("the group proves");
    Fixture {
        case,
        cfg,
        shape: ChainShape {
            query: query(case),
            ..ChainShape::new_at(&cfg, layout.n_stack(), 4)
        },
        layout,
        points,
        values,
        roots,
        claims,
        proof,
        domain: stacked.domain().clone(),
    }
}

/// The host's verdict on `answers`, read at the group's point.
fn host_accepts(f: &Fixture, answers: &[FEE]) -> bool {
    multilinear::stacked_eval::verify_ood::<F, E, _, P1Whir>(
        &f.proof,
        &f.layout,
        &f.roots,
        &Claimed::PerColumn(&f.points),
        &f.values,
        Some(multilinear::stacked_eval::CommitOod {
            z0: &f.claims.z0,
            answers,
        }),
        &f.domain,
        &f.cfg,
        &mut P1Transcript::new(),
    )
    .is_ok()
}

/// The leg as a program: the group's roots and answers hinted, then each
/// column's point, the values and the chains; the group's point drawn from
/// the root cells ([`emit_commit_ood_point`]); the opening with the answers
/// carried; the point's first coordinate `z0` and the batching challenge
/// published.
fn program(f: &Fixture) -> LfmProgram {
    emit_leg(f, Leg::Derived)
}

/// What a leg emits around the opening.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Leg {
    /// The point drawn from the root cells, the answers carried.
    Derived,
    /// The point's coordinates hinted, the answers carried: the opening's own
    /// rows alone.
    Hinted,
    /// No answers: today's opening.
    Without,
}

fn emit_leg(f: &Fixture, leg: Leg) -> LfmProgram {
    let polys = f.layout.num_polys();
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Poseidon1);
    let arena = b.declare_arena(0);
    let mut at = 0u32;
    let next = |b: &mut LfmBuilder, at: &mut u32| {
        let cell = b.hint_word(arena, *at);
        *at += 1;
        cell
    };
    let roots: Vec<Cell> = (0..polys).map(|_| next(&mut b, &mut at)).collect();
    let answers: Vec<Ext> = (0..polys).map(|_| next(&mut b, &mut at).as_ext()).collect();
    let points: Vec<Vec<Ext>> = f
        .points
        .iter()
        .map(|p| p.iter().map(|_| next(&mut b, &mut at).as_ext()).collect())
        .collect();
    let values: Vec<Ext> = f
        .values
        .iter()
        .map(|_| next(&mut b, &mut at).as_ext())
        .collect();
    let runs = f.shape.chains(polys);
    let mut finals = Vec::new();
    let mut storage = Vec::new();
    for run in &runs {
        finals.push(next(&mut b, &mut at).as_ext());
        let shape = f.shape.run(run.len());
        storage.push(RoundStorage::hint(&mut b, arena, at, &shape));
        at += RoundStorage::words(&shape);
    }
    let point: Vec<Ext> = match leg {
        Leg::Derived => emit_commit_ood_point(
            &mut b,
            &stark::multilinear_block::commit_ood_prefix(GROUP, &f.layout, &f.cfg),
            &roots,
            f.layout.n_stack(),
            f.layout.n_stack() + f.cfg.log_blowup,
        ),
        Leg::Hinted | Leg::Without => (0..f.layout.n_stack())
            .map(|_| next(&mut b, &mut at).as_ext())
            .collect(),
    };
    let openings: Vec<_> = storage.iter().map(RoundStorage::openings).collect();
    let rounds: Vec<Vec<_>> = storage
        .iter()
        .zip(&openings)
        .map(|(chain, (current, next))| chain.wires(current, next))
        .collect();
    let chains: Vec<StackedChainWires<'_>> = runs
        .iter()
        .enumerate()
        .map(|(i, run)| StackedChainWires {
            rounds: &rounds[i],
            roots: &roots[run.clone()],
            final_value: finals[i],
        })
        .collect();
    let claimed_at: Vec<&[Ext]> = points.iter().map(Vec::as_slice).collect();
    let mut transcript = WhirTranscript::for_builder(&mut b);
    let gamma = emit_stacked_verify_chains(
        &mut b,
        &mut transcript,
        &f.layout,
        &chains,
        &claimed_at,
        &values,
        (leg != Leg::Without).then_some(OodWires {
            point: &point,
            answers: &answers,
        }),
        &f.shape,
        &f.domain,
    );
    b.public(point[0].as_cell());
    b.public(gamma.as_cell());
    b.set_arena_len(arena, at);
    let program = compile(b.finish());
    validate(&program).expect("the leg must be admissible");
    program
}

/// The arena from the fixture with `answers`, in [`program`]'s order.
fn arena(f: &Fixture, answers: &[FEE]) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = f.roots.iter().map(commitment_to_digest).collect();
    words.extend(answers.iter().map(ext_word));
    words.extend(f.points.iter().flatten().map(ext_word));
    words.extend(f.values.iter().map(ext_word));
    for (chain, run) in f
        .proof
        .polys
        .iter()
        .zip(f.shape.chains(f.layout.num_polys()))
    {
        words.push(ext_word(&chain.final_value));
        try_push_round_words(&mut words, &f.shape.run(run.len()), chain)
            .expect("the proof covers every leaf the lean check reads");
    }
    words
}

/// Every case and split, with the lean case's tail rounds asserted present:
/// the fixtures the tests below walk.
fn fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    for case in CASES {
        for group in GROUPS {
            let f = fixture(case, group);
            if case == Case::Production(LeafQuery::Lean) {
                let runs = f.shape.chains(f.layout.num_polys());
                assert!(
                    runs.iter().any(|run| {
                        let shape = f.shape.run(run.len());
                        (0..shape.rounds()).any(|r| shape.tail_round(r))
                    }),
                    "{case:?} {group:?}: the lean check's tail rounds run"
                );
            }
            out.push(f);
        }
    }
    out
}

fn runs(program: &LfmProgram, words: Vec<LfmWord>) -> Result<Vec<FEE>, String> {
    let exec =
        execute(program, &[words], &HasherKind::Poseidon1W16).map_err(|e| format!("{e:?}"))?;
    exec.public_words
        .iter()
        .map(|(_, w)| word_as_ext(w).ok_or_else(|| "a published extension".to_string()))
        .collect()
}

const GROUPS: [GroupFold; 2] = [GroupFold::PerPoly, GroupFold::Indexed];

/// ★ The leg executes the host's honest opening, and the point it draws from
/// the root cells is the host's.
#[test]
fn an_opening_with_its_answers_executes_at_the_hosts_point() {
    for f in fixtures() {
        let group = (f.case, f.cfg.format.group);
        assert!(
            host_accepts(&f, &f.claims.answers),
            "{group:?}: the control"
        );
        let published = runs(&program(&f), arena(&f, &f.claims.answers))
            .unwrap_or_else(|e| panic!("{group:?}: the machine refused the host's opening: {e}"));
        assert_eq!(published[0], f.claims.z0, "{group:?}: the group's point");
    }
}

/// ★ A false answer — any one of them — and two answers of one run traded,
/// their sum kept, have no execution, as the host refuses them.
#[test]
fn a_false_or_traded_answer_has_no_execution() {
    for f in fixtures() {
        let group = (f.case, f.cfg.format.group);
        let program = program(&f);
        let answers = &f.claims.answers;
        for at in 0..answers.len() {
            let mut wrong = answers.clone();
            wrong[at] += FEE::one();
            assert!(!host_accepts(&f, &wrong), "{group:?}: answer {at}, host");
            assert!(
                runs(&program, arena(&f, &wrong)).is_err(),
                "{group:?}: answer {at} false executed"
            );
        }
        let mut traded = answers.clone();
        let delta = fee(99);
        traded[0] += delta;
        traded[1] = traded[1] - delta;
        assert!(!host_accepts(&f, &traded), "{group:?}: traded, host");
        assert!(
            runs(&program, arena(&f, &traded)).is_err(),
            "{group:?}: two answers of one run traded"
        );
    }
}

/// ★ Mutations: with the answers left out of the chains' claims, or their
/// `eq` out of the chains' weights, the honest opening has no execution —
/// each half of the batching is what the chain checks.
#[test]
fn each_half_of_the_batching_is_load_bearing() {
    for f in fixtures() {
        let group = (f.case, f.cfg.format.group);
        for fault in [
            ood_faults::Fault::ClaimWithoutAnswers,
            ood_faults::Fault::WeightWithoutAnswers,
        ] {
            let program = ood_faults::with(fault, || program(&f));
            assert!(
                runs(&program, arena(&f, &f.claims.answers)).is_err(),
                "{group:?} {fault:?}: the honest opening executed"
            );
        }
    }
}

/// Poseidon1 permutations of a program: its socket's `Hash16` rows.
fn hash16_rows(program: &LfmProgram) -> usize {
    program
        .instrs
        .iter()
        .filter(|i| matches!(i, super::instr::Instr::Hash16(_)))
        .count()
}

/// ★ The plan prices what the leaf emits: the opening's permutations with
/// and without its answers are the cost model's
/// (`stacked_verify_cost_ood`), and the point's own sponge is
/// `commit_ood_point_perms`'s, at either split and either query check.
#[test]
fn the_answers_cost_what_the_plan_prices() {
    for f in fixtures() {
        let group = (f.case, f.cfg.format.group);
        let group_of: Vec<usize> = (0..HEIGHTS.len()).collect();
        let entry = SpongeEntry::fresh_for(WrapHash::Poseidon1);
        for (leg, ood) in [(Leg::Hinted, true), (Leg::Without, false)] {
            let cost = stacked_verify_cost_ood(&f.layout, &group_of, &f.shape, entry, ood);
            assert_eq!(
                hash16_rows(&emit_leg(&f, leg)),
                cost.perms(),
                "{group:?} answers {ood}: permutations"
            );
        }
        assert_eq!(
            hash16_rows(&emit_leg(&f, Leg::Derived)) - hash16_rows(&emit_leg(&f, Leg::Hinted)),
            super::whir_block::commit_ood_point_perms(WrapHash::Poseidon1, f.layout.num_polys()),
            "{group:?}: the point's sponge"
        );
    }
}

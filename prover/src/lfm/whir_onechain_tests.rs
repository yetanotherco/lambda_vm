//! D-ONECHAIN B2: the WHIR leaf's leg for one chain per run of a group's
//! stacked polynomials (`GroupFold::Indexed`), gated against
//! `stacked_eval::verify` on real `stacked_eval::prove` proofs, under RPX and
//! under Poseidon1's 4-ary capped trees. The gates are REV-B2-JUDGE §4's B2-1
//! to B2-6, each named where it is tested; B2-7, the median census by
//! emission, is `whir_p1_census_tests::whir_onechain_census`.
//!
//! As in `whir_stacked_tests`, executing an honest proof IS the stream
//! comparison: every value the machine draws feeds a refusal.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::fiat_shamir::p1_transcript::P1Transcript;
use crypto::fiat_shamir::transcript_hash::RpxTranscriptHash;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{Claimed, Refusal, StackedCommitment, StackedProof};
use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, LeafOrder, RoundOpenings,
    WhirFolds,
};
use multilinear::whir_commit::Commitment;
use multilinear::whir_hash::{P1Whir, RpxWhir, WhirHash};
use multilinear::whir_round::GroupedRoundProof;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::commitment_to_digest;
use super::builder::{Cell, Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapHash;
use super::executor::{LfmExecError, execute, execute_serial};
use super::hash::HasherKind;
use super::instr::Instr;
use super::validator::validate;
use super::whir_chain::onechain_steps::{self, Step};
use super::whir_chain::{
    ChainShape, LeafQuery, RoundStorage, chain_opening_perms, onechain_faults, push_round_words,
};
use super::whir_chain_tests::{Recording, const_rows, hint_rows, perm_rows};
use super::whir_stacked::{StackedChainWires, emit_stacked_verify_chains, stacked_verify_cost};
use super::whir_transcript::{SpongeEntry, WhirTranscript};
use super::word::{LfmWord, ext_word, word_as_ext};

type F = GoldilocksField;
type E = GoldilocksExtension;
type RpxHost = DefaultTranscript<E, RpxTranscriptHash>;

/// The base a case runs under: RPX's binary trees and byte transcript, or
/// Poseidon1's 4-ary trees and field sponge (the block's production base).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Base {
    Rpx,
    P1,
}

/// One case: columns' heights in a stack of `n_stack`, the fold schedule,
/// `Q`, a uniform grind, the base's cap (binary under RPX, 4-ary under
/// Poseidon1) and the round-0 trees' leaf order. Every column is claimed at a
/// point of its own.
struct Case {
    name: &'static str,
    base: Base,
    heights: Vec<usize>,
    n_stack: usize,
    first: Option<usize>,
    log_folding: usize,
    log_blowup: usize,
    num_queries: usize,
    grind: u8,
    cap: CapPolicy,
    leaves: LeafOrder,
}

fn rpx(name: &'static str, heights: &[usize], n_stack: usize, grind: u8) -> Case {
    Case {
        name,
        base: Base::Rpx,
        heights: heights.to_vec(),
        n_stack,
        first: None,
        log_folding: 2,
        log_blowup: 2,
        num_queries: 3,
        grind,
        cap: CapPolicy::Off,
        leaves: LeafOrder::Natural,
    }
}

/// The cases: a run of three (slot 3 absent), of two (`m = 1`) and of four (no
/// slot absent); a split group (five → [4, 1]); mixed heights whose short
/// columns' selectors run past `k0`; a first fold of six over eight variables
/// (round 0 carries 6 + 2 messages); a grouped chain of one round (`n ≤ k0`,
/// R = 1, REV-B2-JUDGE J2); no grind; a binary cap; and under Poseidon1 a run
/// of three at the C2 cap and a split group uncapped.
fn cases() -> Vec<Case> {
    vec![
        rpx("three polynomials, one chain", &[6; 3], 6, 8),
        rpx("two polynomials, one index variable", &[6; 2], 6, 8),
        rpx("four polynomials, no slot absent", &[6; 4], 6, 8),
        rpx("five polynomials split [4, 1]", &[6; 5], 6, 8),
        rpx(
            "mixed heights, selectors past k0",
            &[6, 3, 6, 2, 4, 6],
            6,
            8,
        ),
        Case {
            first: Some(6),
            log_folding: 4,
            ..rpx("first fold 6 over eight variables", &[8; 3], 8, 8)
        },
        // Blowup 2^6: a one-round chain's tree has 2^log_blowup leaves, and a
        // forgery from another position needs positions that differ.
        Case {
            first: Some(6),
            log_folding: 4,
            log_blowup: 6,
            ..rpx("one round (n ≤ k0)", &[5; 3], 5, 8)
        },
        rpx("three polynomials, no grind", &[6; 3], 6, 0),
        Case {
            cap: CapPolicy::Fixed(1),
            ..rpx("three polynomials under a binary cap", &[6; 3], 6, 8)
        },
        Case {
            base: Base::P1,
            first: Some(6),
            log_folding: 4,
            cap: CapPolicy::Fixed(2),
            ..rpx("P1, three polynomials at C2", &[10; 3], 10, 6)
        },
        Case {
            base: Base::P1,
            first: Some(6),
            log_folding: 4,
            ..rpx("P1, five polynomials split [4, 1]", &[8; 5], 8, 0)
        },
    ]
}

fn config(case: &Case) -> ChainConfig {
    let mut format = ChainFormat {
        folds: case.first.map_or(WhirFolds::Uniform, |k| {
            WhirFolds::First(FirstFold::new(k).expect("a tested first fold"))
        }),
        group: GroupFold::Indexed,
        leaves: case.leaves,
        ..ChainFormat::DEFAULT
    };
    match case.base {
        Base::Rpx => format.cap = case.cap,
        Base::P1 => format.arity4_cap = case.cap,
    }
    ChainConfig {
        log_blowup: case.log_blowup,
        log_folding: case.log_folding,
        num_queries: case.num_queries,
        grind: GrindBits::uniform(case.grind),
        format,
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

/// Everything one case's run needs on both sides.
struct Fixture {
    name: &'static str,
    base: Base,
    cfg: ChainConfig,
    layout: StackedLayout,
    /// One claimed point per column.
    points: Vec<Vec<FEE>>,
    values: Vec<FEE>,
    roots: Vec<Commitment>,
    proof: StackedProof<F, E>,
    domain: Domain<F>,
    /// The per-polynomial shape the stack is cut from ([`ChainShape::run`]).
    shape: ChainShape,
    /// RPX only: the batching challenge and the hash events the HOST verifier's
    /// own transcript performed (`Recording`).
    gamma: Option<FEE>,
    host_hashes: Option<Vec<super::whir_transcript::SpongeHash>>,
}

/// Commits and proves one case on the host under its base, opened per
/// `group`.
fn build(case: &Case, group: GroupFold) -> Fixture {
    build_over(case, group, None)
}

/// [`build`], column `slot` replaced by another polynomial (`seed`) when
/// `other` is `Some((slot, seed))`: the same points, another stack.
fn build_over(case: &Case, group: GroupFold, other: Option<(usize, u64)>) -> Fixture {
    let mut cfg = config(case);
    cfg.format.group = group;
    let layout = StackedLayout::build(&case.heights, case.n_stack).expect("the layout packs");
    let columns: Vec<Mle<F>> = case
        .heights
        .iter()
        .enumerate()
        .map(|(c, &h)| match other {
            Some((slot, seed)) if slot == c => column(h, seed),
            _ => column(h, 17 + 13 * c as u64),
        })
        .collect();
    let borrowed = multilinear::stacking::borrow(&columns);
    let points: Vec<Vec<FEE>> = case
        .heights
        .iter()
        .enumerate()
        .map(|(c, &h)| (0..h).map(|i| fee(0x5eed + (c * 37 + i) as u64)).collect())
        .collect();
    let values: Vec<FEE> = columns
        .iter()
        .zip(&points)
        .map(|(c, p)| c.evaluate_in::<E>(p).expect("a column takes its own point"))
        .collect();
    fn run<H: WhirHash, T: IsTranscript<E>>(
        layout: &StackedLayout,
        borrowed: &[&Mle<F>],
        points: &[Vec<FEE>],
        values: &[FEE],
        cfg: &ChainConfig,
        transcript: fn() -> T,
    ) -> (Vec<Commitment>, StackedProof<F, E>, Domain<F>) {
        let stacked = StackedCommitment::<F, H>::commit(layout.clone(), borrowed, None, cfg)
            .expect("the group commits");
        let proof = multilinear::stacked_eval::prove::<F, E, T, H, _>(
            &stacked,
            borrowed,
            None,
            &Claimed::PerColumn(points),
            values,
            cfg,
            &mut transcript(),
        )
        .expect("the group proves");
        (stacked.roots(), proof, stacked.domain().clone())
    }
    let (roots, proof, domain) = match case.base {
        Base::Rpx => run::<RpxWhir, _>(&layout, &borrowed, &points, &values, &cfg, || {
            RpxHost::new(&[])
        }),
        Base::P1 => run::<P1Whir, _>(
            &layout,
            &borrowed,
            &points,
            &values,
            &cfg,
            P1Transcript::new,
        ),
    };
    let arity = match case.base {
        Base::Rpx => 2,
        Base::P1 => 4,
    };
    Fixture {
        name: case.name,
        base: case.base,
        cfg,
        shape: ChainShape::new_at(&cfg, layout.n_stack(), arity),
        layout,
        points,
        values,
        roots,
        proof,
        domain,
        gamma: None,
        host_hashes: None,
    }
}

/// [`build`] under `Indexed`, with a grouped chain, accepted by the host —
/// under RPX through `Recording`, keeping the host's batching challenge and
/// hash events.
fn fixture(case: &Case) -> Fixture {
    let mut f = build(case, GroupFold::Indexed);
    assert!(
        f.shape
            .chains(f.layout.num_polys())
            .iter()
            .any(|r| r.len() >= 2),
        "{}: a grouped chain",
        f.name
    );
    match f.base {
        Base::Rpx => {
            let mut recorded = Recording::new();
            host_verify(&f, &f.proof, &mut recorded).expect("the control proof must verify");
            f.gamma = Some(recorded.sampled[0]);
            f.host_hashes = Some(recorded.duplex.borrow().hashes.clone());
        }
        Base::P1 => assert!(host_accepts(&f, &f.proof), "{}: the control", f.name),
    }
    f
}

fn host_verify<T: IsTranscript<E>>(
    f: &Fixture,
    proof: &StackedProof<F, E>,
    transcript: &mut T,
) -> Result<(), multilinear::Error> {
    macro_rules! verify_under {
        ($h:ty) => {
            multilinear::stacked_eval::verify::<F, E, T, $h>(
                proof,
                &f.layout,
                &f.roots,
                &Claimed::PerColumn(&f.points),
                &f.values,
                &f.domain,
                &f.cfg,
                transcript,
            )
        };
    }
    match f.base {
        Base::Rpx => verify_under!(RpxWhir),
        Base::P1 => verify_under!(P1Whir),
    }
}

fn host_accepts(f: &Fixture, proof: &StackedProof<F, E>) -> bool {
    match f.base {
        Base::Rpx => host_verify(f, proof, &mut RpxHost::new(&[])).is_ok(),
        Base::P1 => host_verify(f, proof, &mut P1Transcript::new()).is_ok(),
    }
}

fn socket(base: Base) -> HasherKind {
    match base {
        Base::Rpx => crate::hash_pin::BLOCK_HASHER,
        Base::P1 => HasherKind::Poseidon1W16,
    }
}

/// Where every wire lives: each column's point, the values, then per chain
/// its run's roots, its final value and its rounds at the run's shape. Built
/// once for the program and the arena filler alike.
struct Arena {
    values_at: u32,
    chain_at: Vec<u32>,
    total: u32,
}

impl Arena {
    fn new(f: &Fixture) -> Self {
        let mut at: u32 = f.points.iter().map(|p| p.len() as u32).sum();
        let values_at = at;
        at += f.values.len() as u32;
        let mut chain_at = Vec::new();
        for run in f.shape.chains(f.layout.num_polys()) {
            chain_at.push(at);
            at += run.len() as u32 + 1 + RoundStorage::words(&f.shape.run(run.len()));
        }
        Self {
            values_at,
            chain_at,
            total: at,
        }
    }
}

/// The leg as a program: the hints in [`Arena`]'s order, then
/// [`emit_stacked_verify_chains`] — the block leaf's group opening — and the
/// batching challenge published.
fn program(f: &Fixture) -> LfmProgram {
    traced_program(f).0
}

/// [`program`] with the marks of its checks ([`onechain_steps`]), so a refusal
/// can be named by the check that refused.
fn traced_program(f: &Fixture) -> (LfmProgram, Vec<(Step, usize)>) {
    onechain_steps::record(|| emit_program(f))
}

fn emit_program(f: &Fixture) -> LfmProgram {
    let at = Arena::new(f);
    let hash = match f.base {
        Base::Rpx => WrapHash::production(),
        Base::P1 => WrapHash::Poseidon1,
    };
    let mut b = LfmBuilder::new().with_wrap_hash(hash);
    let arena = b.declare_arena(at.total);
    let mut transcript = WhirTranscript::for_builder(&mut b);
    let mut next = 0u32;
    let points: Vec<Vec<Ext>> = f
        .points
        .iter()
        .map(|p| {
            p.iter()
                .map(|_| {
                    let w = b.hint_word(arena, next).as_ext();
                    next += 1;
                    w
                })
                .collect()
        })
        .collect();
    let values: Vec<Ext> = (0..f.values.len() as u32)
        .map(|i| b.hint_word(arena, at.values_at + i).as_ext())
        .collect();
    let runs = f.shape.chains(f.layout.num_polys());
    let mut roots: Vec<Vec<Cell>> = Vec::new();
    let mut finals: Vec<Ext> = Vec::new();
    let mut storage: Vec<RoundStorage> = Vec::new();
    for (run, &base) in runs.iter().zip(&at.chain_at) {
        let n = run.len() as u32;
        roots.push((0..n).map(|i| b.hint_word(arena, base + i)).collect());
        finals.push(b.hint_word(arena, base + n).as_ext());
        storage.push(RoundStorage::hint(
            &mut b,
            arena,
            base + n + 1,
            &f.shape.run(run.len()),
        ));
    }
    let openings: Vec<_> = storage.iter().map(RoundStorage::openings).collect();
    let rounds: Vec<Vec<_>> = storage
        .iter()
        .zip(&openings)
        .map(|(chain, (current, next))| chain.wires(current, next))
        .collect();
    let chains: Vec<StackedChainWires<'_>> = (0..runs.len())
        .map(|i| StackedChainWires {
            rounds: &rounds[i],
            roots: &roots[i],
            final_value: finals[i],
        })
        .collect();
    let claimed_at: Vec<&[Ext]> = points.iter().map(Vec::as_slice).collect();
    let gamma = emit_stacked_verify_chains(
        &mut b,
        &mut transcript,
        &f.layout,
        &chains,
        &claimed_at,
        &values,
        None,
        &f.shape,
        &f.domain,
    );
    b.public(gamma.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the grouped leg must be admissible");
    program
}

/// The arena from a proof, in [`program`]'s order.
fn arena(f: &Fixture, proof: &StackedProof<F, E>) -> Vec<LfmWord> {
    arena_under(f, &f.roots, proof)
}

/// [`arena`] laid out by `f`'s shape over `roots` and `proof`, which may be
/// another fixture's: a proof written into the arena of another config.
fn arena_under(f: &Fixture, roots: &[Commitment], proof: &StackedProof<F, E>) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = f.points.iter().flatten().map(ext_word).collect();
    words.extend(f.values.iter().map(ext_word));
    let runs = f.shape.chains(f.layout.num_polys());
    for (chain, run) in proof.polys.iter().zip(&runs) {
        words.extend(roots[run.clone()].iter().map(commitment_to_digest));
        words.push(ext_word(&chain.final_value));
        push_round_words(&mut words, &f.shape.run(run.len()), chain);
    }
    words
}

fn machine_runs(f: &Fixture, program: &LfmProgram, words: Vec<LfmWord>) -> Result<FEE, String> {
    let exec = execute(program, &[words], &socket(f.base)).map_err(|e| format!("{e:?}"))?;
    word_as_ext(&exec.public_words[0].1).ok_or_else(|| "no published challenge".to_string())
}

/// How the machine refused an arena: at an emitted check (a failed assert,
/// named by its chain and step), or with another executor error.
#[derive(Debug)]
enum Refused {
    At(Option<usize>, Step),
    Other(LfmExecError),
}

/// The machine's refusal of `words`, or `None` when it executes. On the serial
/// schedule, so the refusal reported is the FIRST failing check in program
/// order — the host's order; the level schedule may meet a later one first (it
/// refuses all the same: [`machine_runs`] runs it).
fn machine_refusal(
    f: &Fixture,
    (program, marks): &(LfmProgram, Vec<(Step, usize)>),
    words: Vec<LfmWord>,
) -> Option<Refused> {
    assert_eq!(
        execute(program, std::slice::from_ref(&words), &socket(f.base)).is_ok(),
        execute_serial(program, std::slice::from_ref(&words), &socket(f.base)).is_ok(),
        "{}: both schedules give one verdict",
        f.name
    );
    let error = execute_serial(program, &[words], &socket(f.base)).err()?;
    Some(match onechain_steps::failed_at(program, marks, &error) {
        Some((chain, step)) => Refused::At(chain, step),
        None => Refused::Other(error),
    })
}

/// The host's refusal of `proof` with its cause, or `None` when it accepts.
fn host_refusal(f: &Fixture, proof: &StackedProof<F, E>) -> Option<Refusal> {
    macro_rules! traced_under {
        ($h:ty, $t:expr) => {
            multilinear::stacked_eval::verify_traced::<F, E, _, $h>(
                proof,
                &f.layout,
                &f.roots,
                &Claimed::PerColumn(&f.points),
                &f.values,
                &f.domain,
                &f.cfg,
                &mut $t,
            )
        };
    }
    match f.base {
        Base::Rpx => traced_under!(RpxWhir, RpxHost::new(&[])),
        Base::P1 => traced_under!(P1Whir, P1Transcript::new()),
    }
    .err()
}

/// Whether the machine's step is the check the host's cause names: a grouped
/// round-0 tree's opening, another opening, a fold, a grind, or the final
/// check — in the same chain, at the same query and tree where the host says.
fn same_check(host: &Refusal, machine: &Refused) -> bool {
    use multilinear::Error as H;
    let (Refusal::Chain { chain, cause, .. }, Refused::At(Some(at_chain), step)) = (host, machine)
    else {
        return false;
    };
    chain == at_chain
        && match (cause, *step) {
            (
                H::GroupedOpeningRejected { query, tree },
                Step::Opening {
                    round,
                    query: q,
                    tree: p,
                },
            ) => round == 0 && *query == q && *tree == p,
            (
                H::OpeningRejected { query },
                Step::Opening {
                    round, query: q, ..
                },
            ) => round > 0 && *query == q,
            (H::OpeningRejected { query }, Step::Successor { query: q, .. }) => *query == q,
            (H::FoldInconsistent { query }, Step::Fold { query: q, .. }) => *query == q,
            (H::GrindingRejected { .. }, Step::Sumcheck { .. } | Step::Ood { .. }) => true,
            (H::EvaluationMismatch, Step::Final) => true,
            _ => false,
        }
}

fn grouped(proof: &mut StackedProof<F, E>, chain: usize) -> &mut GroupedRoundProof<F, E> {
    match &mut proof.polys[chain].rounds[0].openings {
        RoundOpenings::BaseGrouped(g) => g,
        _ => panic!("chain {chain} is not grouped"),
    }
}

/// The arena index of value `t` of tree `p`'s block at query `q`, in chain 0
/// (a grouped one): past its roots and final value, round 0's messages and
/// nonces and its trees' caps, then query-major blocks and paths.
fn round0_value_at(f: &Fixture, q: usize, p: usize, t: usize) -> usize {
    let at = Arena::new(f);
    let runs = f.shape.chains(f.layout.num_polys());
    let shape = f.shape.run(runs[0].len());
    let nonces = shape.carries(0).iter().filter(|&&c| c).count();
    let caps = shape.trees * shape.cap_words(shape.current_depth(0), shape.caps[0]);
    let block = 1usize << shape.schedule[0];
    at.chain_at[0] as usize
        + runs[0].len()
        + 1
        + 2 * shape.messages(0)
        + nonces
        + caps
        + (q * shape.trees + p) * (block + shape.current_path(0))
        + t
}

// ─────────────── B2-1, B2-6: the leg executes what the host accepts ───────────────

/// ★ REV-B2-JUDGE B2-6 (host acceptance implies guest acceptance) at every
/// case, and B2-1's split: the machine executes each proof the host accepts,
/// one chain per run of the verifier's own split, and draws the batching
/// challenge the HOST drew.
#[test]
fn grouped_openings_execute_on_proofs_the_host_accepts() {
    for case in cases() {
        let f = fixture(&case);
        let program = program(&f);
        let gamma = machine_runs(&f, &program, arena(&f, &f.proof))
            .unwrap_or_else(|e| panic!("{}: the machine refused an accepted proof: {e}", f.name));
        if let Some(host) = f.gamma {
            assert_eq!(
                gamma, host,
                "{}: the batching challenge is the host's",
                f.name
            );
        }
        println!(
            "onechain [{}]: runs {:?}, {} instructions",
            f.name,
            f.shape.chains(f.layout.num_polys()),
            program.instrs.len()
        );
    }
}

/// `case` at blowup `2^log_blowup`, everything else the same.
fn at_blowup(case: &Case, log_blowup: usize) -> Case {
    Case {
        name: case.name,
        base: case.base,
        heights: case.heights.clone(),
        n_stack: case.n_stack,
        first: case.first,
        log_folding: case.log_folding,
        log_blowup,
        num_queries: case.num_queries,
        grind: case.grind,
        cap: case.cap,
        leaves: case.leaves,
    }
}

/// ★ The base code's rate (`LAMBDA_VM_ZF_WHIR_BLOWUP`) under `Indexed`: every
/// case at blowup 2 — runs of two, three and four, a split group, mixed
/// heights, both bases, capped and not — executes on the machine built for it
/// over a proof the host accepts. The host and the machine built for blowup 4
/// refuse the same proof over the same roots: every tree is a level shallower,
/// so the other rate's domain and arena (verifier constants) do not fit it.
#[test]
fn grouped_openings_at_blowup_2_execute_and_are_refused_at_blowup_4() {
    for case in cases().into_iter().filter(|c| c.log_blowup == 2) {
        let half = fixture(&at_blowup(&case, 1));
        assert_eq!(
            half.domain.log_size(),
            half.layout.n_stack() + 1,
            "{}: the stack's domain at blowup 2",
            half.name
        );
        machine_runs(&half, &program(&half), arena(&half, &half.proof)).unwrap_or_else(|e| {
            panic!(
                "{}: the machine at blowup 2 refused an accepted proof: {e}",
                half.name
            )
        });

        let four = build(&case, GroupFold::Indexed);
        macro_rules! refused_by_four {
            ($h:ty, $t:expr) => {
                multilinear::stacked_eval::verify::<F, E, _, $h>(
                    &half.proof,
                    &half.layout,
                    &half.roots,
                    &Claimed::PerColumn(&half.points),
                    &half.values,
                    &four.domain,
                    &four.cfg,
                    &mut $t,
                )
                .is_err()
            };
        }
        let host = match half.base {
            Base::Rpx => refused_by_four!(RpxWhir, RpxHost::new(&[])),
            Base::P1 => refused_by_four!(P1Whir, P1Transcript::new()),
        };
        assert!(host, "{}: the host at blowup 4 must refuse it", half.name);
        assert!(
            machine_runs(
                &four,
                &program(&four),
                arena_under(&four, &half.roots, &half.proof)
            )
            .is_err(),
            "{}: the machine at blowup 4 must refuse it",
            half.name
        );
    }
}

/// `case` with its round-0 trees in `leaves`' order, everything else the same.
fn in_order(case: &Case, leaves: LeafOrder) -> Case {
    Case {
        leaves,
        ..at_blowup(case, case.log_blowup)
    }
}

/// Whether the host verifier of `f`'s config accepts `proof` over `roots`,
/// which may be another fixture's.
fn host_accepts_under(f: &Fixture, roots: &[Commitment], proof: &StackedProof<F, E>) -> bool {
    macro_rules! accepts {
        ($h:ty, $t:expr) => {
            multilinear::stacked_eval::verify::<F, E, _, $h>(
                proof,
                &f.layout,
                roots,
                &Claimed::PerColumn(&f.points),
                &f.values,
                &f.domain,
                &f.cfg,
                &mut $t,
            )
            .is_ok()
        };
    }
    match f.base {
        Base::Rpx => accepts!(RpxWhir, RpxHost::new(&[])),
        Base::P1 => accepts!(P1Whir, P1Transcript::new()),
    }
}

/// ★ The composed format at laptop width — one chain a run, the coset round-0
/// order and blowup 2 (`LAMBDA_VM_ZF_WHIR_LEAVES=coset`,
/// `LAMBDA_VM_ZF_WHIR_BLOWUP=2`): every Poseidon1 case executes on the machine
/// built for it over a proof the host accepts, and the host and the machine of
/// each format one setting away — the coset order at blowup 4, the natural
/// order at blowup 2 — refuse it over its roots.
#[test]
fn grouped_openings_under_the_composed_format_execute_and_are_refused_one_setting_away() {
    for case in cases().into_iter().filter(|c| c.base == Base::P1) {
        let composed = fixture(&in_order(&at_blowup(&case, 1), LeafOrder::Coset));
        assert_eq!(
            composed.domain.log_size(),
            composed.layout.n_stack() + 1,
            "{}: blowup 2",
            composed.name
        );
        assert_eq!(composed.shape.leaves, LeafOrder::Coset);
        machine_runs(
            &composed,
            &program(&composed),
            arena(&composed, &composed.proof),
        )
        .unwrap_or_else(|e| {
            panic!(
                "{}: the machine under the composed format refused an accepted proof: {e}",
                composed.name
            )
        });
        let coset_four = fixture(&in_order(&case, LeafOrder::Coset));
        let natural_two = fixture(&at_blowup(&case, 1));
        for (label, verifier) in [
            ("the coset order at blowup 4", &coset_four),
            ("the natural order at blowup 2", &natural_two),
        ] {
            assert!(
                !host_accepts_under(verifier, &composed.roots, &composed.proof),
                "{}: {label} (host)",
                case.name
            );
            assert!(
                machine_runs(
                    verifier,
                    &program(verifier),
                    arena_under(verifier, &composed.roots, &composed.proof)
                )
                .is_err(),
                "{}: {label} (machine)",
                case.name
            );
        }
    }
}

/// ★ The coset round-0 leaf order (`LAMBDA_VM_ZF_WHIR_LEAVES`, Poseidon1 only)
/// under `Indexed`: every Poseidon1 case — a run of three at the C2 cap, a
/// split group — executes on the machine built for it over a proof the host
/// accepts, so EVERY round-0 tree of a run is walked at the position the order
/// puts the block at (one tree walked at the index would refuse the honest
/// proof). Across the two orders the host and the machine each refuse the
/// other order's proof over its roots, both ways.
#[test]
fn grouped_openings_under_the_coset_order_execute_and_are_refused_in_the_natural_one() {
    for case in cases().into_iter().filter(|c| c.base == Base::P1) {
        let coset = fixture(&in_order(&case, LeafOrder::Coset));
        let natural = fixture(&case);
        assert_eq!(coset.shape.leaves, LeafOrder::Coset);
        assert_ne!(
            coset.roots, natural.roots,
            "{}: the order moves the roots",
            coset.name
        );
        machine_runs(&coset, &program(&coset), arena(&coset, &coset.proof)).unwrap_or_else(|e| {
            panic!(
                "{}: the machine under the coset order refused an accepted proof: {e}",
                coset.name
            )
        });
        for (label, verifier, prover) in [
            ("natural refuses coset", &natural, &coset),
            ("coset refuses natural", &coset, &natural),
        ] {
            assert!(
                !host_accepts_under(verifier, &prover.roots, &prover.proof),
                "{}: {label} (host)",
                case.name
            );
            assert!(
                machine_runs(
                    verifier,
                    &program(verifier),
                    arena_under(verifier, &prover.roots, &prover.proof)
                )
                .is_err(),
                "{}: {label} (machine)",
                case.name
            );
        }
    }
}

/// ★ REV-B2-JUDGE B2-1: the grouped legs' THREADED schedule is the host's own
/// (`the_schedule_is_the_host_transcripts` at `P' = 2, 3, 4` and a split
/// group): `stacked_verify_cost` walks the runs of the split, each at its
/// shape, and the host's `stacked_eval::verify` performs exactly its hashes.
#[test]
fn the_grouped_schedule_is_the_host_transcripts() {
    for case in cases().into_iter().filter(|c| c.base == Base::Rpx) {
        let f = fixture(&case);
        let group_of: Vec<usize> = (0..f.values.len()).collect();
        let cost = stacked_verify_cost(&f.layout, &group_of, &f.shape, SpongeEntry::fresh());
        let mine = cost.schedule().hashes();
        let host = f.host_hashes.as_ref().expect("an RPX fixture records");
        assert_eq!(
            cost.chains(),
            f.shape.chains(f.layout.num_polys()).len(),
            "{}: one chain a run",
            f.name
        );
        assert_eq!(mine.len(), host.len(), "{}: hash count", f.name);
        for (i, (derived, observed)) in mine.iter().zip(host).enumerate() {
            assert_eq!(derived, observed, "{}: hash {i}", f.name);
        }
    }
}

/// ★ REV-B2-JUDGE B2-1: the closed forms are the emitted counts at every case —
/// rows (const-free, `chain_rows`' convention) and permutations under RPX, the
/// socket's `Hash16` rows under Poseidon1 — so the W3 plan prices exactly the
/// leg the leaf emits.
#[test]
fn grouped_openings_emit_their_closed_form() {
    for case in cases() {
        let f = fixture(&case);
        let program = program(&f);
        let at = Arena::new(&f);
        let group_of: Vec<usize> = (0..f.values.len()).collect();
        let entry = match f.base {
            Base::Rpx => SpongeEntry::fresh(),
            Base::P1 => SpongeEntry::fresh_for(WrapHash::Poseidon1),
        };
        let cost = stacked_verify_cost(&f.layout, &group_of, &f.shape, entry);
        assert_eq!(
            hint_rows(&program),
            at.total as usize,
            "{}: every arena word hinted once",
            f.name
        );
        let perms = match f.base {
            Base::Rpx => perm_rows(&program),
            Base::P1 => program
                .instrs
                .iter()
                .filter(|i| matches!(i, Instr::Hash16(_)))
                .count(),
        };
        assert_eq!(perms, cost.perms(), "{}: permutations", f.name);
        if f.base == Base::Rpx {
            // One `Hint` a word and the published challenge are plumbing.
            let measured = program.instrs.len() - const_rows(&program) - (at.total as usize + 1);
            assert_eq!(measured, cost.operations(), "{}: rows", f.name);
        }
    }
}

/// ★ A run of ONE polynomial under `Indexed` emits today's program, row for
/// row, and reads today's arena: the singleton chain of a split is the
/// per-polynomial chain exactly. The program at a real grind width; the
/// arena unground, because a ground proof's nonces are the grinder's choice
/// (`whir_identity_tests`).
#[test]
fn one_polynomial_under_indexed_emits_todays_program() {
    let case = rpx("one polynomial", &[6], 6, 8);
    let (indexed, per_poly) = (
        build(&case, GroupFold::Indexed),
        build(&case, GroupFold::PerPoly),
    );
    assert!(host_accepts(&indexed, &indexed.proof));
    assert_eq!(
        format!("{:?}", program(&indexed).instrs),
        format!("{:?}", program(&per_poly).instrs)
    );
    let case = rpx("one polynomial, unground", &[6], 6, 0);
    let (indexed, per_poly) = (
        build(&case, GroupFold::Indexed),
        build(&case, GroupFold::PerPoly),
    );
    assert_eq!(
        arena(&indexed, &indexed.proof),
        arena(&per_poly, &per_poly.proof)
    );
}

// ───────────────────── B2-2: the base-lift pin, per slot ─────────────────────

/// ★ REV-B2-JUDGE B2-2 (REV-B2-B RF-2): in EVERY slot `p < P'`, a hinted
/// round-0 value with a nonzero upper lane has no execution, refused as
/// `NotBaseWord` at tree `p`'s leaf `Pack` — the slice the combine lifts is the
/// slice that leaf hashed (`whir_open_tests::
/// the_leaf_pins_a_hinted_base_value_to_its_low_lane`, for one tree).
#[test]
fn every_slots_round_zero_value_is_pinned_to_its_low_lane() {
    let pinned =
        |c: &Case| (c.n_stack == 6 && c.heights.len() <= 4) || c.cap == CapPolicy::Fixed(2);
    for case in cases().into_iter().filter(|c| pinned(c) && c.grind > 0) {
        let f = fixture(&case);
        let program = program(&f);
        let honest = arena(&f, &f.proof);
        assert!(machine_runs(&f, &program, honest.clone()).is_ok());
        let trees = f.shape.chains(f.layout.num_polys())[0].len();
        for p in 0..trees {
            let mut smuggled = honest.clone();
            let at = round0_value_at(&f, 1, p, 2);
            assert_eq!(smuggled[at][1], FE::zero(), "{}: a base word", f.name);
            smuggled[at][1] = FE::one();
            let refusal = execute(&program, &[smuggled], &socket(f.base))
                .expect_err("a lifted value with a nonzero lane 1 must have no execution");
            assert!(
                matches!(refusal, LfmExecError::NotBaseWord(_)),
                "{} slot {p}: the refusal is the base-word check at the leaf's Pack, got \
                 {refusal:?}",
                f.name
            );
        }
    }
}

/// The arena word and lane of value `t` of tree `p`'s block at query `q` in
/// chain 0, under [`LeafQuery::Lean`]: [`round0_value_at`]'s layout with each
/// block four values a word.
fn round0_lane_at(f: &Fixture, q: usize, p: usize, t: usize) -> (usize, usize) {
    let at = Arena::new(f);
    let runs = f.shape.chains(f.layout.num_polys());
    let shape = f.shape.run(runs[0].len());
    assert!(shape.round0_packed(), "a lean P1 shape");
    let nonces = shape.carries(0).iter().filter(|&&c| c).count();
    let caps = shape.trees * shape.cap_words(shape.current_depth(0), shape.caps[0]);
    let words = shape.round0_block_words(1usize << shape.schedule[0]);
    let word = at.chain_at[0] as usize
        + runs[0].len()
        + 1
        + 2 * shape.messages(0)
        + nonces
        + caps
        + (q * shape.trees + p) * (words + shape.current_path(0))
        + t / 4;
    (word, t % 4)
}

/// ★ Under [`LeafQuery::Lean`] a grouped round 0's blocks arrive four values a
/// word: every Poseidon1 case executes on the proof the host accepts, with
/// fewer hinted words and the same socket rows as under V1, and a value one
/// off in any lane of any slot's tree has no execution.
#[test]
fn grouped_round_zero_words_under_the_lean_query_execute_and_refuse_a_lane_off() {
    for case in cases().into_iter().filter(|c| c.base == Base::P1) {
        let mut f = fixture(&case);
        let v1 = program(&f);
        f.shape.query = LeafQuery::Lean;
        let lean = program(&f);
        let honest = arena(&f, &f.proof);
        machine_runs(&f, &lean, honest.clone())
            .unwrap_or_else(|e| panic!("{}: the machine refused an accepted proof: {e}", f.name));
        assert!(
            hint_rows(&lean) < hint_rows(&v1),
            "{}: fewer hinted words",
            f.name
        );
        let block = 1usize << f.shape.schedule[0];
        let trees = f.shape.chains(f.layout.num_polys())[0].len();
        for p in 0..trees {
            for t in [0, 1, 2, 3, block - 1] {
                let (word, lane) = round0_lane_at(&f, 1, p, t);
                let mut forged = honest.clone();
                forged[word][lane] += FE::one();
                assert!(
                    execute(&lean, &[forged], &socket(f.base)).is_err(),
                    "{} slot {p}: value {t} one off must have no execution",
                    f.name
                );
            }
        }
    }
}

// ────────────── B2-3, B2-4, B2-6: every forgery has no execution ──────────────

/// A forgery: what it is, the forged proof, the host's cause when it is fixed
/// (`None` when the forgery moves the transcript), the machine's step likewise.
type Forgery = (
    String,
    StackedProof<F, E>,
    Option<multilinear::Error>,
    Option<Step>,
);

/// ★ REV-B2-JUDGE B2-3, B2-4 and B2-6: every forgery the host refuses has no
/// execution, and the machine refuses it at the SAME check the host does
/// ([`same_check`], against the honest twin, which executes):
/// - a value in slot `p` at query 1 — tree `p`'s opening there;
/// - blocks swapped between trees 0 and 1 — tree 0's opening (each tree
///   against its own root, by position);
/// - the last tree's block from another query's position — that tree's
///   opening (one draw feeds every tree);
/// - a round-0 sibling of tree 1 at query 2 — tree 1's opening there;
/// - the successor's block at query 0 — the successor's opening;
/// - an index round's message and the final value, which move the
///   transcript: wherever the host's refusal lands (a grind, an opening at a
///   moved position, a fold), the machine's is the same check;
/// - B1-2's shapes (a block short, one in the absent slot, `k0` round-0
///   messages): the host's named shape refusal, and an arena of another length,
///   which the program — reading a fixed number of wires — refuses as such.
#[test]
fn every_grouped_forgery_the_host_refuses_has_no_execution() {
    use multilinear::Error as H;
    use multilinear::whir_chain::GroupedShape as S;
    let three = |c: &Case| c.heights.len() == 3 && c.cap == CapPolicy::Off;
    for case in cases()
        .into_iter()
        .filter(|c| three(c) || c.base == Base::P1)
    {
        // Eight queries: a block from another position needs one whose
        // position differs from query 1's.
        let f = fixture(&Case {
            num_queries: 8,
            ..case
        });
        let traced = traced_program(&f);
        assert!(machine_refusal(&f, &traced, arena(&f, &f.proof)).is_none());
        let k0 = f.shape.schedule[0];
        let trees = f.shape.chains(f.layout.num_polys())[0].len();
        let opening = |query, tree| Some(H::GroupedOpeningRejected { query, tree });
        // (what, the forgery, the host's cause when it is fixed, the machine's step)
        let mut forgeries: Vec<Forgery> = Vec::new();
        for p in 0..trees {
            let mut proof = f.proof.clone();
            grouped(&mut proof, 0).current[1][p].values[2] += FE::one();
            forgeries.push((
                format!("a value in slot {p}"),
                proof,
                opening(1, p),
                Some(onechain_steps::opening(0, 1, p)),
            ));
        }
        let mut swapped = f.proof.clone();
        grouped(&mut swapped, 0).current[1].swap(0, 1);
        forgeries.push((
            "blocks swapped between trees 0 and 1".into(),
            swapped,
            opening(1, 0),
            Some(onechain_steps::opening(0, 1, 0)),
        ));
        let g = grouped(&mut f.proof.clone(), 0).clone();
        let last = trees - 1;
        // From a query past the owner (query 0, whose paths carry the caps), so
        // the arena keeps its length and only the position differs.
        let other = (2..g.current.len())
            .find(|&j| g.current[j][last].values != g.current[1][last].values)
            .unwrap_or_else(|| panic!("{}: no query at another position", f.name));
        let mut moved = f.proof.clone();
        grouped(&mut moved, 0).current[1][last] = g.current[other][last].clone();
        forgeries.push((
            "the last tree's block from another position".into(),
            moved,
            opening(1, last),
            Some(onechain_steps::opening(0, 1, last)),
        ));
        let mut sibling = f.proof.clone();
        grouped(&mut sibling, 0).current[2][1]
            .proof
            .merkle_path
            .first_mut()
            .unwrap_or_else(|| panic!("{}: a walked level to tamper", f.name))[0] ^= 1;
        forgeries.push((
            "a round-0 sibling of tree 1".into(),
            sibling,
            opening(2, 1),
            Some(onechain_steps::opening(0, 2, 1)),
        ));
        if f.shape.rounds() > 1 {
            let mut next = f.proof.clone();
            grouped(&mut next, 0).next[0].values[0] += FEE::one();
            forgeries.push((
                "the successor's block".into(),
                next,
                Some(H::OpeningRejected { query: 0 }),
                Some(Step::Successor { round: 0, query: 0 }),
            ));
        }
        let mut index = f.proof.clone();
        index.polys[0].rounds[0].sumcheck[k0].evaluations[0] += FEE::one();
        forgeries.push(("an index round's message".into(), index, None, None));
        let mut final_value = f.proof.clone();
        final_value.polys[0].final_value += FEE::one();
        forgeries.push(("the final value".into(), final_value, None, None));

        for (what, forged, cause, step) in forgeries {
            let host = host_refusal(&f, &forged)
                .unwrap_or_else(|| panic!("{} [{what}]: the host must refuse", f.name));
            let machine = machine_refusal(&f, &traced, arena(&f, &forged))
                .unwrap_or_else(|| panic!("{} [{what}]: the machine executed it", f.name));
            println!(
                "ONECHAIN FORGERY [{}] {what}: host {host:?} · machine {machine:?}",
                f.name
            );
            if let Some(cause) = cause {
                assert_eq!(
                    host,
                    Refusal::Chain {
                        chain: 0,
                        poly: 0,
                        cause
                    },
                    "{} [{what}]: the host's check",
                    f.name
                );
            }
            if let Some(step) = step {
                assert!(
                    matches!(machine, Refused::At(Some(0), s) if s == step),
                    "{} [{what}]: the machine refused at {machine:?}, not {step:?}",
                    f.name
                );
            }
            assert!(
                same_check(&host, &machine),
                "{} [{what}]: the host refused at {host:?}, the machine at {machine:?}",
                f.name
            );
        }

        // B1-2's shapes: the host names them; the arena has another length.
        let mut short = f.proof.clone();
        grouped(&mut short, 0).current[1].pop();
        let mut absent = f.proof.clone();
        let extra = grouped(&mut absent, 0).current[1][0].clone();
        grouped(&mut absent, 0).current[1].push(extra);
        let mut messages = f.proof.clone();
        messages.polys[0].rounds[0].sumcheck.truncate(k0);
        let m = f.shape.run(trees).index_vars();
        for (what, forged, shape) in [
            (
                "a block short at a query",
                short,
                S::BlocksPerQuery {
                    query: 1,
                    expected: trees,
                    got: trees - 1,
                },
            ),
            (
                "a block in the absent slot",
                absent,
                S::BlocksPerQuery {
                    query: 1,
                    expected: trees,
                    got: trees + 1,
                },
            ),
            (
                "k0 round-0 messages",
                messages,
                S::Round0Messages {
                    expected: k0 + m,
                    got: k0,
                },
            ),
        ] {
            assert_eq!(
                host_refusal(&f, &forged),
                Some(Refusal::Chain {
                    chain: 0,
                    poly: 0,
                    cause: H::GroupedShape(shape)
                }),
                "{} [{what}]: the host's named shape",
                f.name
            );
            let machine = machine_refusal(&f, &traced, arena(&f, &forged));
            assert!(
                matches!(
                    machine,
                    Some(Refused::Other(LfmExecError::ArenaLenMismatch { .. }))
                ),
                "{} [{what}]: refused as another arena length, got {machine:?}",
                f.name
            );
        }
    }
}

// ─────────────────── B2-5: each reading of the index ───────────────────

/// ★ REV-B2-JUDGE B2-5, the host's B1-3 negatives in-guest: each place the
/// leg reads the index is load-bearing, and each refuses at its own check.
/// With `ρ` swapped in the combine only, the honest arena is refused at round
/// 0's first fold; with `ρ` swapped in the weight only, the index factor
/// dropped, or the absent slot given weight 1, at the final check; with none of
/// them it executes. (`eq(ρ, p)` is computed from the sampled `ρ` wires: there
/// is no hint wire to set it by.)
#[test]
fn each_in_guest_reading_of_the_index_is_load_bearing() {
    use onechain_faults::Fault;
    // Runs of three and four: at two, one index variable, swapping `ρ` is the
    // identity.
    let wide = |c: &Case| {
        (c.base == Base::Rpx && c.n_stack == 6 && [3, 4].contains(&c.heights.len()) && c.grind > 0)
            || c.cap == CapPolicy::Fixed(2)
    };
    for case in cases().into_iter().filter(wide) {
        let f = fixture(&case);
        let words = arena(&f, &f.proof);
        let honest = onechain_faults::with(Fault::None, || traced_program(&f));
        assert!(machine_refusal(&f, &honest, words.clone()).is_none());
        let trees = f.shape.chains(f.layout.num_polys())[0].len();
        for (fault, step) in [
            (Fault::SwapRhoInCombine, Step::Fold { round: 0, query: 0 }),
            (Fault::SwapRhoInWeight, Step::Final),
            (Fault::DropIndexFactor, Step::Final),
            (Fault::AbsentSlotWeightOne, Step::Final),
        ] {
            if fault == Fault::AbsentSlotWeightOne && trees.is_power_of_two() {
                continue;
            }
            let faulty = onechain_faults::with(fault, || traced_program(&f));
            let refused = machine_refusal(&f, &faulty, words.clone());
            assert!(
                matches!(refused, Some(Refused::At(Some(0), s)) if s == step),
                "{} {fault:?}: refused at {refused:?}, not {step:?}",
                f.name
            );
        }
    }
}

/// REV-B2-JUDGE B1-1 in guest: a run whose virtual domain `H_{n+m+lb}` would
/// pass the field's two-adicity has no program — the shape refuses it at emit
/// time, as the host refuses the config before reading.
#[test]
#[should_panic(expected = "has no virtual domain")]
fn a_run_past_the_two_adicity_has_no_program() {
    let mut cfg = config(&rpx("wide", &[6; 3], 6, 0));
    cfg.log_blowup = 32 - 6 - 2 + 1;
    let shape = ChainShape::new_at(&cfg, 6, 2);
    let _ = shape.run(3);
}

// ─────────────────── the production shape's cost (B2-7) ───────────────────

/// ★ The production grouped chain's in-guest openings, from the closed form
/// the W3 plan prices with: Poseidon1, C2, Q 114, schedule [6, 4, 4, 4, 4, 4,
/// 1] at a 2^27 stack — D-ONECHAIN §4.1's B2-sep column (`size_b2.py`), to the
/// permutation: 11,191 for one polynomial (today's chain), 13,018 / 14,845 /
/// 16,672 for runs of two, three and four, against today's 22,382 / 33,573 /
/// 44,764 for the same polynomials chained apart.
#[test]
fn the_production_grouped_chain_costs_what_the_design_sized() {
    let mut cfg = ChainConfig::with_security_folds(
        2,
        4,
        WhirFolds::First(FirstFold::new(6).expect("first6")),
        27,
        128,
        GrindBits::query_only(18),
    );
    cfg.format.arity4_cap = CapPolicy::Fixed(2);
    cfg.format.group = GroupFold::Indexed;
    let shape = ChainShape::new_at(&cfg, 27, 4);
    assert_eq!(shape.schedule, [6, 4, 4, 4, 4, 4, 1]);
    assert_eq!(shape.num_queries, 114);
    let one = chain_opening_perms(&shape.run(1));
    assert_eq!(one, 11_191, "today's chain");
    for (trees, want) in [(2usize, 13_018usize), (3, 14_845), (4, 16_672)] {
        let grouped = chain_opening_perms(&shape.run(trees));
        println!(
            "P1 C2 Q114 n27: a run of {trees} opens for {grouped} permutations (apart {})",
            trees * one
        );
        assert_eq!(grouped, want, "a run of {trees}");
    }
    // ★ The B2 gate's per-group figure (D-ONECHAIN §5, "14,845 + sponge ±
    // 3 %"): a three-polynomial group's whole opening leg is the chain's
    // openings and its threaded sponge, and the sponge is inside the 3 %. The
    // design's ≈ 720 a chain [I] (`size_b2.py`, read off W3 PLAN costs that
    // also hold the tables' legs) is not the sponge: one grouped chain's is
    // 99 here, three per-polynomial chains' 289.
    let layout = StackedLayout::build(&[27; 3], 27).expect("three 2^27 polynomials");
    let entry = SpongeEntry::fresh_for(WrapHash::Poseidon1);
    let opening = stacked_verify_cost(&layout, &[0, 1, 2], &shape, entry).perms();
    let mut per_poly = shape.clone();
    per_poly.group = GroupFold::PerPoly;
    let today = stacked_verify_cost(&layout, &[0, 1, 2], &per_poly, entry).perms();
    println!(
        "P1 C2 Q114 n27, a three-polynomial group's opening leg: {opening} permutations \
         Indexed (sponge {}), {today} PerPoly (sponge {}), {:+.1} %",
        opening - 14_845,
        today - 3 * 11_191,
        100.0 * (opening as f64 / today as f64 - 1.0)
    );
    assert_eq!((opening, today), (14_845 + 99, 33_573 + 289));
    assert!(
        (opening - 14_845) * 100 <= 3 * 14_845,
        "the sponge within the gate's 3 %"
    );
}

// ─────────────────── from the code review (REV-OC-B) ───────────────────

/// The code review's M-a, in guest: each round-0 tree's cap is authenticated
/// against ITS root, by position. Under the binary cap and Poseidon1's C2 cap,
/// the arena words of trees 0's and 1's caps traded have no execution, and
/// the failing assert lies in the chain's cap authentication — after its
/// `Chain` mark and before round 0's first check; the honest arena executes.
#[test]
fn round_zero_caps_are_bound_per_tree_in_guest() {
    for case in cases().into_iter().filter(|c| c.cap != CapPolicy::Off) {
        let f = fixture(&case);
        let (program, marks) = traced_program(&f);
        let honest = arena(&f, &f.proof);
        assert!(machine_runs(&f, &program, honest.clone()).is_ok());
        let at = Arena::new(&f);
        let runs = f.shape.chains(f.layout.num_polys());
        let shape = f.shape.run(runs[0].len());
        let words = shape.cap_words(shape.current_depth(0), shape.caps[0]);
        assert!(words > 0, "{}: a capped round-0 tree", f.name);
        let nonces = shape.carries(0).iter().filter(|&&c| c).count();
        let caps = at.chain_at[0] as usize + runs[0].len() + 1 + 2 * shape.messages(0) + nonces;
        let mut traded = honest.clone();
        for w in 0..words {
            traded.swap(caps + w, caps + words + w);
        }
        assert_ne!(traded, honest, "{}: two trees, two caps", f.name);
        let error = execute_serial(&program, &[traded], &socket(f.base))
            .expect_err("traded caps must have no execution");
        let LfmExecError::DivByZero { addr } = error else {
            panic!("{}: a failed assert, got {error:?}", f.name);
        };
        let index = program
            .instrs
            .iter()
            .position(|i| match i {
                Instr::BaseAlu { out, .. } | Instr::ExtAlu { out, .. } => out.0 == addr,
                _ => false,
            })
            .expect("the assert's difference");
        let chain0 = marks
            .iter()
            .find(|(s, _)| *s == Step::Chain(0))
            .map(|&(_, at)| at)
            .expect("chain 0's mark");
        let round0 = marks
            .iter()
            .find(|(s, _)| *s == Step::Sumcheck { round: 0 })
            .map(|&(_, at)| at)
            .expect("round 0's mark");
        println!(
            "ONECHAIN CAPS [{}]: {words} cap words a tree; refused at instruction {index} in \
             [{chain0}, {round0})",
            f.name
        );
        assert!(
            chain0 <= index && index < round0,
            "{}: refused at {index}, outside the cap authentication [{chain0}, {round0})",
            f.name
        );
    }
}

/// An instrument, asserting nothing: a digest of each case's emitted
/// instruction stream and its arena length, at `Fault::None` with the step
/// marks recording. Run once as is and once with the emitters' `cfg(test)`
/// statements compiled out, as a release build compiles them: equal digests
/// show the test build emits what a release build does (the code review found
/// 24 of 24 equal).
#[test]
#[ignore = "instrument: prints digests to compare across two builds"]
fn emitted_program_digests() {
    use std::hash::{Hash, Hasher};
    let mut all = cases();
    all.push(rpx("one polynomial", &[6], 6, 8));
    for case in all {
        let mut f = build(&case, GroupFold::Indexed);
        for group in [GroupFold::Indexed, GroupFold::PerPoly] {
            if group == GroupFold::PerPoly {
                f = build(&case, GroupFold::PerPoly);
            }
            let program = onechain_faults::with(onechain_faults::Fault::None, || program(&f));
            let mut h = std::collections::hash_map::DefaultHasher::new();
            format!("{:?}", program.instrs).hash(&mut h);
            println!(
                "ONECHAIN DIGEST [{} {group:?}] {} instructions {:016x} arena {}",
                f.name,
                program.instrs.len(),
                h.finish(),
                Arena::new(&f).total
            );
        }
    }
}

/// The leg with its roots as PROGRAM CONSTANTS, as the block leaf's prepared
/// branch wires them (`whir_block`: derived roots as `digest_const`, each chain
/// its run's slice): tree `p` is checked against the constant of root
/// `order[p]`. Its arena holds no root word.
fn const_root_program(f: &Fixture, order: &[usize]) -> (LfmProgram, Vec<(Step, usize)>) {
    onechain_steps::record(|| {
        let runs = f.shape.chains(f.layout.num_polys());
        let values_at: u32 = f.points.iter().map(|p| p.len() as u32).sum();
        let mut total = values_at + f.values.len() as u32;
        let chain_at: Vec<u32> = runs
            .iter()
            .map(|run| {
                let at = total;
                total += 1 + RoundStorage::words(&f.shape.run(run.len()));
                at
            })
            .collect();
        let hash = match f.base {
            Base::Rpx => WrapHash::production(),
            Base::P1 => WrapHash::Poseidon1,
        };
        let mut b = LfmBuilder::new().with_wrap_hash(hash);
        let arena = b.declare_arena(total);
        let mut transcript = WhirTranscript::for_builder(&mut b);
        let mut next = 0u32;
        let points: Vec<Vec<Ext>> = f
            .points
            .iter()
            .map(|p| {
                p.iter()
                    .map(|_| {
                        let w = b.hint_word(arena, next).as_ext();
                        next += 1;
                        w
                    })
                    .collect()
            })
            .collect();
        let values: Vec<Ext> = (0..f.values.len() as u32)
            .map(|i| b.hint_word(arena, values_at + i).as_ext())
            .collect();
        let mut finals: Vec<Ext> = Vec::new();
        let mut storage: Vec<RoundStorage> = Vec::new();
        for (run, &base) in runs.iter().zip(&chain_at) {
            finals.push(b.hint_word(arena, base).as_ext());
            storage.push(RoundStorage::hint(
                &mut b,
                arena,
                base + 1,
                &f.shape.run(run.len()),
            ));
        }
        let roots: Vec<Cell> = order
            .iter()
            .map(|&p| b.digest_const(commitment_to_digest(&f.roots[p])).as_cell())
            .collect();
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
        let gamma = emit_stacked_verify_chains(
            &mut b,
            &mut transcript,
            &f.layout,
            &chains,
            &claimed_at,
            &values,
            None,
            &f.shape,
            &f.domain,
        );
        b.public(gamma.as_cell());
        let program = compile(b.finish());
        validate(&program).expect("the constant-root leg must be admissible");
        program
    })
}

/// The arena [`const_root_program`] reads: the points, the values, then per
/// chain its final value and its rounds.
fn const_root_arena(f: &Fixture, proof: &StackedProof<F, E>) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = f.points.iter().flatten().map(ext_word).collect();
    words.extend(f.values.iter().map(ext_word));
    let runs = f.shape.chains(f.layout.num_polys());
    for (chain, run) in proof.polys.iter().zip(&runs) {
        words.push(ext_word(&chain.final_value));
        push_round_words(&mut words, &f.shape.run(run.len()), chain);
    }
    words
}

/// The code review's M-g at the stacked level: grouped chains opened
/// against roots that are PROGRAM CONSTANTS, the wiring of the block leaf's
/// prepared branch. The honest arena executes; with the constants of trees 0
/// and 1 exchanged it has no execution (tree 0's opening at query 0, or its cap
/// where the tree is capped); a value of tree 1's block at query 1 is refused
/// at tree 1's opening there. Under RPX (a run of three, a split [4, 1]) and
/// Poseidon1 (C2, and a split).
#[test]
fn grouped_chains_open_against_constant_roots() {
    let picked = |c: &Case| {
        c.base == Base::P1
            || (c.n_stack == 6
                && [3, 5].contains(&c.heights.len())
                && c.grind > 0
                && c.cap == CapPolicy::Off)
    };
    for case in cases().into_iter().filter(picked) {
        let f = fixture(&Case {
            num_queries: 8,
            ..case
        });
        let order: Vec<usize> = (0..f.roots.len()).collect();
        let traced = const_root_program(&f, &order);
        let words = const_root_arena(&f, &f.proof);
        assert!(
            machine_refusal(&f, &traced, words.clone()).is_none(),
            "{}: the honest arena executes against constant roots",
            f.name
        );

        let mut exchanged = order.clone();
        exchanged.swap(0, 1);
        let wrong = const_root_program(&f, &exchanged);
        let refused = machine_refusal(&f, &wrong, words.clone());
        let capped = f.shape.caps[0] > 0;
        assert!(
            match &refused {
                Some(Refused::At(Some(0), s)) => !capped && *s == onechain_steps::opening(0, 0, 0),
                // A capped tree's cap is authenticated before the chain's first
                // marked check.
                Some(Refused::Other(LfmExecError::DivByZero { .. })) => capped,
                _ => false,
            },
            "{}: constants exchanged, refused at {refused:?}",
            f.name
        );

        let mut forged = f.proof.clone();
        grouped(&mut forged, 0).current[1][1].values[2] += FE::one();
        let refused = machine_refusal(&f, &traced, const_root_arena(&f, &forged));
        assert!(
            matches!(&refused, Some(Refused::At(Some(0), s)) if *s == onechain_steps::opening(0, 1, 1)),
            "{}: a value of tree 1's block, refused at {refused:?}",
            f.name
        );
        println!(
            "ONECHAIN CONST ROOTS [{}]: runs {:?}, capped {capped}",
            f.name,
            f.shape.chains(f.layout.num_polys())
        );
    }
}

// ─────────── each binding of a grouped round 0, on its own ───────────

/// The chain whose PROLOGUE holds the assert a `DivByZero` failed at: after
/// that chain's `Chain` mark and before its round 0's `Sumcheck` mark — its
/// roots unpacked, its claim batched and its round-0 trees' caps
/// authenticated, of which only the caps assert. `None` for any other refusal.
fn refused_in_prologue(
    (program, marks): &(LfmProgram, Vec<(Step, usize)>),
    error: &LfmExecError,
) -> Option<usize> {
    let LfmExecError::DivByZero { addr } = error else {
        return None;
    };
    let index = program.instrs.iter().position(|i| match i {
        Instr::BaseAlu { out, .. } | Instr::ExtAlu { out, .. } => out.0 == *addr,
        _ => false,
    })?;
    match marks.iter().rev().find(|(_, at)| *at <= index)? {
        (Step::Chain(c), _) => Some(*c),
        _ => None,
    }
}

/// EACH round-0 tree of a grouped chain is bound, in guest, to the root at its
/// own position, its cap included — tree 0 as much as a later one. Trading
/// trees 0's and 1's caps ([`round_zero_caps_are_bound_per_tree_in_guest`])
/// breaks two bindings at once, so it cannot show either tree bound alone;
/// this forgery moves one slot and leaves every other tree as committed.
///
/// For every slot `p` in turn, slot `p` commits another polynomial; the proof
/// is that stack's honest one (its own trees, caps and paths,
/// self-consistent), claimed against the stated roots, which differ from its
/// own at slot `p` alone. The claim is false, and every check but tree `p`'s
/// binding to its root passes. A capped tree refuses at its cap (host
/// `CapRejected`; the machine in the chain's prologue, where the caps are
/// authenticated), an uncapped one at its first opening
/// (`GroupedOpeningRejected { 0, p }`; the machine at tree `p`'s opening at
/// query 0).
#[test]
fn a_tree_committing_another_polynomial_has_no_execution() {
    let cases = [
        Case {
            cap: CapPolicy::Fixed(1),
            ..rpx("three polynomials under a binary cap", &[6; 3], 6, 8)
        },
        rpx("three polynomials, uncapped", &[6; 3], 6, 8),
        Case {
            base: Base::P1,
            first: Some(6),
            log_folding: 4,
            cap: CapPolicy::Fixed(2),
            ..rpx("P1, three polynomials at C2", &[10; 3], 10, 6)
        },
    ];
    for case in cases {
        let capped = case.cap != CapPolicy::Off;
        let honest = build(&case, GroupFold::Indexed);
        for slot in 0..3 {
            let mut forged =
                build_over(&case, GroupFold::Indexed, Some((slot, 0xBAD + slot as u64)));
            assert_eq!(forged.points, honest.points);
            assert_ne!(forged.values, honest.values);
            for p in 0..3 {
                assert_eq!(
                    forged.roots[p] == honest.roots[p],
                    p != slot,
                    "{}: root {p}",
                    case.name
                );
            }
            let traced = traced_program(&forged);
            assert!(
                host_accepts(&forged, &forged.proof),
                "{}: the control",
                case.name
            );
            assert!(
                machine_refusal(&forged, &traced, arena(&forged, &forged.proof)).is_none(),
                "{}: the control executes",
                case.name
            );
            let proof = forged.proof.clone();
            forged.roots = honest.roots.clone();
            let cause = if capped {
                multilinear::Error::CapRejected
            } else {
                multilinear::Error::GroupedOpeningRejected {
                    query: 0,
                    tree: slot,
                }
            };
            assert_eq!(
                host_refusal(&forged, &proof),
                Some(Refusal::Chain {
                    chain: 0,
                    poly: 0,
                    cause
                }),
                "{} slot {slot}: the host's check",
                case.name
            );
            let words = arena(&forged, &proof);
            if capped {
                let Err(error) = execute_serial(
                    &traced.0,
                    std::slice::from_ref(&words),
                    &socket(forged.base),
                ) else {
                    panic!(
                        "{} slot {slot}: a tree's cap off its root was executed",
                        case.name
                    );
                };
                assert!(execute(&traced.0, &[words], &socket(forged.base)).is_err());
                assert_eq!(
                    refused_in_prologue(&traced, &error),
                    Some(0),
                    "{} slot {slot}: refused at the chain's caps, got {error:?}",
                    case.name
                );
            } else {
                let refused = machine_refusal(&forged, &traced, words);
                assert!(
                    matches!(
                        refused,
                        Some(Refused::At(Some(0), Step::Opening { round: 0, query: 0, tree })) if tree == slot
                    ),
                    "{} slot {slot}: refused at {refused:?}",
                    case.name
                );
            }
        }
    }
}

/// The grouped leg's instruction stream, emitted over hinted wires alone — it
/// is a function of the layout and the config, never of a proof — as
/// `(instructions, FNV-1a of their Debug text)`. ★ The SAME body sits in
/// `prover/tests/onechain_release_stream.rs`, built against the library
/// WITHOUT `cfg(test)` (no faults, no step marks), and both builds are held to
/// the same pins: the test build emits exactly the release build's program.
fn grouped_leg_stream(p1: bool, heights: &[usize], n_stack: usize) -> (usize, u64) {
    use multilinear::whir_chain::{
        CapPolicy, ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, WhirFolds,
    };
    let mut format = ChainFormat {
        folds: WhirFolds::First(FirstFold::new(6).expect("first6")),
        group: GroupFold::Indexed,
        ..ChainFormat::DEFAULT
    };
    if p1 {
        format.arity4_cap = CapPolicy::Fixed(2);
    } else {
        format.cap = CapPolicy::Fixed(1);
    }
    let cfg = ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 3,
        grind: GrindBits::uniform(8),
        format,
    };
    let layout = multilinear::stacking::StackedLayout::build(heights, n_stack).expect("packs");
    let shape = ChainShape::new_at(&cfg, n_stack, if p1 { 4 } else { 2 });
    let domain = multilinear::whir::Domain::<GoldilocksField>::new(n_stack + cfg.log_blowup)
        .expect("domain");
    let hash = if p1 {
        WrapHash::Poseidon1
    } else {
        WrapHash::production()
    };
    let runs = shape.chains(layout.num_polys());
    let mut total: u32 = heights.iter().sum::<usize>() as u32 + heights.len() as u32;
    for run in &runs {
        total += run.len() as u32 + 1 + RoundStorage::words(&shape.run(run.len()));
    }
    let mut b = LfmBuilder::new().with_wrap_hash(hash);
    let arena = b.declare_arena(total);
    let mut transcript = WhirTranscript::for_builder(&mut b);
    let mut at = 0u32;
    let next = |b: &mut LfmBuilder, at: &mut u32| {
        let cell = b.hint_word(arena, *at);
        *at += 1;
        cell
    };
    let points: Vec<Vec<Ext>> = heights
        .iter()
        .map(|&h| (0..h).map(|_| next(&mut b, &mut at).as_ext()).collect())
        .collect();
    let values: Vec<Ext> = heights
        .iter()
        .map(|_| next(&mut b, &mut at).as_ext())
        .collect();
    let mut roots: Vec<Vec<Cell>> = Vec::new();
    let mut finals: Vec<Ext> = Vec::new();
    let mut storage: Vec<RoundStorage> = Vec::new();
    for run in &runs {
        roots.push(run.clone().map(|_| next(&mut b, &mut at)).collect());
        finals.push(next(&mut b, &mut at).as_ext());
        let run_shape = shape.run(run.len());
        let base = at;
        storage.push(RoundStorage::hint(&mut b, arena, base, &run_shape));
        at += RoundStorage::words(&run_shape);
    }
    let openings: Vec<_> = storage.iter().map(RoundStorage::openings).collect();
    let rounds: Vec<Vec<_>> = storage
        .iter()
        .zip(&openings)
        .map(|(chain, (current, next))| chain.wires(current, next))
        .collect();
    let chains: Vec<StackedChainWires<'_>> = (0..runs.len())
        .map(|i| StackedChainWires {
            rounds: &rounds[i],
            roots: &roots[i],
            final_value: finals[i],
        })
        .collect();
    let claimed_at: Vec<&[Ext]> = points.iter().map(Vec::as_slice).collect();
    let gamma = emit_stacked_verify_chains(
        &mut b,
        &mut transcript,
        &layout,
        &chains,
        &claimed_at,
        &values,
        None,
        &shape,
        &domain,
    );
    b.public(gamma.as_cell());
    let program = compile(b.finish());
    let text = format!("{:?}", program.instrs);
    let digest = text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, byte| {
        (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    (program.instrs.len(), digest)
}

/// One pinned case: the base (`true` for Poseidon1), the columns' heights, the
/// stack's height, and the stream's `(instructions, FNV-1a)`.
type StreamPin = (bool, &'static [usize], usize, (usize, u64));

/// The cases both builds are held to, each with its stream's pin: RPX and
/// Poseidon1, five polynomials split [4, 1] and six split [4, 2] whose short
/// columns' selectors run past `k0`. A pin moves only with the emitted
/// program; move it in both files together.
const GROUPED_LEG_STREAMS: [StreamPin; 4] = [
    (false, &[8; 5], 8, (5241, 0x8d95_4e39_3cca_f387)),
    (false, &[8, 3, 8, 2, 4, 8], 8, (5229, 0xac5d_a244_80ae_4271)),
    (true, &[10; 5], 10, (28895, 0x52ec_ae97_ae80_6593)),
    (
        true,
        &[10, 3, 10, 2, 4, 10],
        10,
        (28868, 0x02e6_e3fd_5c51_0e98),
    ),
];

/// Under `Fault::None`, and with the step marks recorded or not, the test
/// build emits one program — and it is the pinned one, which
/// `prover/tests/onechain_release_stream.rs` holds a build without `cfg(test)`
/// to as well: the test build's program is the release build's.
#[test]
fn the_grouped_leg_stream_of_a_test_build() {
    for (p1, heights, n, pin) in GROUPED_LEG_STREAMS {
        let plain = grouped_leg_stream(p1, heights, n);
        let (marked, marks) = onechain_steps::record(|| {
            onechain_faults::with(onechain_faults::Fault::None, || {
                grouped_leg_stream(p1, heights, n)
            })
        });
        assert_eq!(plain, marked, "the marks and Fault::None emit nothing");
        assert!(!marks.is_empty(), "the marks were recorded");
        println!(
            "ONECHAIN STREAM p1={p1} heights={heights:?}: {} instructions, digest {:016x}",
            plain.0, plain.1
        );
        assert_eq!(plain, pin, "p1={p1} heights={heights:?}: the pinned stream");
    }
}

/// In guest, a grouped round 0 that is the chain's last round (R = 1) compares
/// each query's combined fold with the final value, as its own check ahead of
/// the final weight check. Unground, so a moved final
/// value moves only the query positions; the forgery is the first such value
/// whose first position does not move (two leaves a tree), so that query's
/// openings stay valid and the host refuses at its fold — and the machine at
/// the same fold.
#[test]
fn a_one_round_grouped_chain_folds_to_its_final_value_in_guest() {
    let case = Case {
        first: Some(6),
        log_folding: 4,
        log_blowup: 1,
        ..rpx("one round, unground", &[5; 3], 5, 0)
    };
    let f = fixture(&case);
    assert_eq!(f.shape.rounds(), 1, "R = 1");
    let traced = traced_program(&f);
    assert!(machine_refusal(&f, &traced, arena(&f, &f.proof)).is_none());
    let fold = || Refusal::Chain {
        chain: 0,
        poly: 0,
        cause: multilinear::Error::FoldInconsistent { query: 0 },
    };
    let forged = (1u64..512)
        .map(|delta| {
            let mut proof = f.proof.clone();
            proof.polys[0].final_value += FEE::from(delta);
            proof
        })
        .find(|proof| host_refusal(&f, proof) == Some(fold()))
        .expect("a final value whose first query position does not move, refused at the fold");
    let refused = machine_refusal(&f, &traced, arena(&f, &forged));
    assert!(
        matches!(
            refused,
            Some(Refused::At(Some(0), Step::Fold { round: 0, query: 0 }))
        ),
        "the machine refused at {refused:?}, not round 0's first fold"
    );
}

// ─────────── the grouped round 0's successor, bound to its root in guest ───────────

/// A grouped round 0's SUCCESSOR tree is bound, through its cap, to the root
/// the round absorbed — in guest as on the host (`whir_round::verify_grouped`).
/// Each node of the successor's cap moved in turn (in the proof, so the host
/// reads the same forgery) is refused by the host at the cap, and has no
/// execution: the failing assert lies in round 0's out-of-domain step, where
/// the successor's cap is authenticated — never at an opening, which a cap
/// node no query reaches would not meet.
#[test]
fn a_grouped_successor_cap_off_its_root_has_no_execution() {
    let cases = [
        // A cap of eight nodes, unground: the successor is opened at most six
        // times (three as round 0's successor, three as round 1's current
        // tree), so some node is met by no opening whatever the draw, and the
        // draw is the same every build. Were the cap not authenticated, the
        // arenas that move those nodes would execute.
        Case {
            cap: CapPolicy::Fixed(3),
            ..rpx(
                "three polynomials under a binary cap of eight, unground",
                &[6; 3],
                6,
                0,
            )
        },
        Case {
            cap: CapPolicy::Fixed(1),
            ..rpx("three polynomials under a binary cap", &[6; 3], 6, 8)
        },
        Case {
            base: Base::P1,
            first: Some(6),
            log_folding: 4,
            cap: CapPolicy::Fixed(2),
            ..rpx("P1, three polynomials at C2", &[10; 3], 10, 6)
        },
    ];
    for case in cases {
        let f = build(&case, GroupFold::Indexed);
        let traced = traced_program(&f);
        let run = f.shape.run(f.shape.chains(f.layout.num_polys())[0].len());
        assert!(run.trees >= 2, "{}: a grouped chain", case.name);
        let depth = run.next_depth(0).expect("round 0 has a successor");
        let cap_words = run.cap_words(depth, run.caps[1]);
        assert!(
            cap_words >= 2,
            "{}: a capped successor, {cap_words} cap words",
            case.name
        );
        assert!(
            machine_refusal(&f, &traced, arena(&f, &f.proof)).is_none(),
            "{}: the control executes",
            case.name
        );
        // Every node first, then the verdict: a node no query reaches is the
        // one only the cap's own check refuses.
        let mut elsewhere = Vec::new();
        for node in 0..cap_words {
            let mut forged = f.proof.clone();
            let owner = &mut grouped(&mut forged, 0).next[0].proof.merkle_path;
            let at = owner.len() - cap_words + node;
            owner[at][0] ^= 1;
            let host = host_refusal(&f, &forged);
            let refused = machine_refusal(&f, &traced, arena(&f, &forged));
            println!(
                "ONECHAIN SUCCESSOR CAP [{}] node {node} of {cap_words}: host {host:?}, machine \
                 {refused:?}",
                case.name
            );
            let at_the_cap =
                host == Some(Refusal::Chain {
                    chain: 0,
                    poly: 0,
                    cause: multilinear::Error::CapRejected,
                }) && matches!(refused, Some(Refused::At(Some(0), Step::Ood { round: 0 })));
            if !at_the_cap {
                elsewhere.push((node, host, refused));
            }
        }
        assert!(
            elsewhere.is_empty(),
            "{}: a successor cap node off its root is refused at the cap by the host and where \
             the cap is authenticated by the machine; not so: {elsewhere:?}",
            case.name
        );
    }
}

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
use multilinear::stacked_eval::{Claimed, StackedCommitment, StackedProof};
use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, RoundOpenings, WhirFolds,
};
use multilinear::whir_commit::Commitment;
use multilinear::whir_hash::{P1Whir, RpxWhir, WhirHash};
use multilinear::whir_round::GroupedRoundProof;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::commitment_to_digest;
use super::builder::{Cell, Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapHash;
use super::executor::{LfmExecError, execute};
use super::hash::HasherKind;
use super::instr::Instr;
use super::validator::validate;
use super::whir_chain::{
    ChainShape, RoundStorage, chain_opening_perms, onechain_faults, push_round_words,
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
/// `Q`, a uniform grind, and the base's cap (binary under RPX, 4-ary under
/// Poseidon1). Every column is claimed at a point of its own.
struct Case {
    name: &'static str,
    base: Base,
    heights: Vec<usize>,
    n_stack: usize,
    first: Option<usize>,
    log_folding: usize,
    num_queries: usize,
    grind: u8,
    cap: CapPolicy,
}

fn rpx(name: &'static str, heights: &[usize], n_stack: usize, grind: u8) -> Case {
    Case {
        name,
        base: Base::Rpx,
        heights: heights.to_vec(),
        n_stack,
        first: None,
        log_folding: 2,
        num_queries: 3,
        grind,
        cap: CapPolicy::Off,
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
        Case {
            first: Some(6),
            log_folding: 4,
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
            ..rpx("P1, three polynomials at C2", &[8; 3], 8, 6)
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
        ..ChainFormat::DEFAULT
    };
    match case.base {
        Base::Rpx => format.cap = case.cap,
        Base::P1 => format.arity4_cap = case.cap,
    }
    ChainConfig {
        log_blowup: 2,
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
    let mut cfg = config(case);
    cfg.format.group = group;
    let layout = StackedLayout::build(&case.heights, case.n_stack).expect("the layout packs");
    let columns: Vec<Mle<F>> = case
        .heights
        .iter()
        .enumerate()
        .map(|(c, &h)| column(h, 17 + 13 * c as u64))
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
    let mut words: Vec<LfmWord> = f.points.iter().flatten().map(ext_word).collect();
    words.extend(f.values.iter().map(ext_word));
    let runs = f.shape.chains(f.layout.num_polys());
    for (chain, run) in proof.polys.iter().zip(&runs) {
        words.extend(f.roots[run.clone()].iter().map(commitment_to_digest));
        words.push(ext_word(&chain.final_value));
        push_round_words(&mut words, &f.shape.run(run.len()), chain);
    }
    words
}

fn machine_runs(f: &Fixture, program: &LfmProgram, words: Vec<LfmWord>) -> Result<FEE, String> {
    let exec = execute(program, &[words], &socket(f.base)).map_err(|e| format!("{e:?}"))?;
    word_as_ext(&exec.public_words[0].1).ok_or_else(|| "no published challenge".to_string())
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

// ────────────── B2-3, B2-4, B2-6: every forgery has no execution ──────────────

/// ★ REV-B2-JUDGE B2-3, B2-4 and B2-6: every forgery the host refuses has no
/// execution — a value in each slot; blocks swapped between trees (each tree
/// against its own root, by position); one tree's block from another
/// position (one draw feeds every tree); a round-0 sibling; an index message;
/// the successor's block; the final value — and every shape B1-2 names (a
/// block short, one in the absent slot, `k0` round-0 messages) reads another
/// arena length, which the program, reading a fixed number of wires, refuses.
#[test]
fn every_grouped_forgery_the_host_refuses_has_no_execution() {
    let three = |c: &Case| c.heights.len() == 3 && c.grind > 0 && c.cap == CapPolicy::Off;
    for case in cases()
        .into_iter()
        .filter(|c| three(c) || c.base == Base::P1)
    {
        let f = fixture(&case);
        let program = program(&f);
        assert!(machine_runs(&f, &program, arena(&f, &f.proof)).is_ok());
        let k0 = f.shape.schedule[0];
        let trees = f.shape.chains(f.layout.num_polys())[0].len();
        let mut forgeries: Vec<(String, StackedProof<F, E>)> = Vec::new();
        for p in 0..trees {
            let mut proof = f.proof.clone();
            grouped(&mut proof, 0).current[1][p].values[2] += FE::one();
            forgeries.push((format!("a value in slot {p}"), proof));
        }
        let mut swapped = f.proof.clone();
        grouped(&mut swapped, 0).current[1].swap(0, 1);
        forgeries.push(("blocks swapped between trees 0 and 1".into(), swapped));
        let g = {
            let mut proof = f.proof.clone();
            grouped(&mut proof, 0).clone()
        };
        if let Some(other) = (0..g.current.len())
            .find(|&j| g.current[j][trees - 1].values != g.current[1][trees - 1].values)
        {
            let mut moved = f.proof.clone();
            grouped(&mut moved, 0).current[1][trees - 1] = g.current[other][trees - 1].clone();
            forgeries.push(("the last tree's block from another position".into(), moved));
        }
        let mut sibling = f.proof.clone();
        if let Some(node) = grouped(&mut sibling, 0).current[2][1]
            .proof
            .merkle_path
            .first_mut()
        {
            node[0] ^= 1;
            forgeries.push(("a round-0 sibling of tree 1".into(), sibling));
        }
        let mut index = f.proof.clone();
        index.polys[0].rounds[0].sumcheck[k0].evaluations[0] += FEE::one();
        forgeries.push(("an index round's message".into(), index));
        if f.shape.rounds() > 1 {
            let mut next = f.proof.clone();
            grouped(&mut next, 0).next[0].values[0] += FEE::one();
            forgeries.push(("the successor's block".into(), next));
        }
        let mut last = f.proof.clone();
        last.polys[0].final_value += FEE::one();
        forgeries.push(("the final value".into(), last));
        // B1-2's shapes: another arena length.
        let mut short = f.proof.clone();
        grouped(&mut short, 0).current[1].pop();
        forgeries.push(("a block short at a query".into(), short));
        let mut absent = f.proof.clone();
        let extra = grouped(&mut absent, 0).current[1][0].clone();
        grouped(&mut absent, 0).current[1].push(extra);
        forgeries.push(("a block in the absent slot".into(), absent));
        let mut messages = f.proof.clone();
        messages.polys[0].rounds[0].sumcheck.truncate(k0);
        forgeries.push((format!("{k0} round-0 messages"), messages));

        for (what, forged) in forgeries {
            assert!(
                !host_accepts(&f, &forged),
                "{} [{what}]: the host must refuse, or the machine's refusal is a refusal of \
                 something valid",
                f.name
            );
            assert!(
                machine_runs(&f, &program, arena(&f, &forged)).is_err(),
                "{} [{what}]: the machine executed a forgery the host refuses",
                f.name
            );
        }
    }
}

// ─────────────────── B2-5: each reading of the index ───────────────────

/// ★ REV-B2-JUDGE B2-5, the host's B1-3 negatives in-guest: each place the
/// leg reads the index is load-bearing. With `ρ` swapped in the combine only,
/// `ρ` swapped in the weight only, the index factor dropped, or the absent
/// slot given weight 1, the honest arena has no execution; with none of them
/// it executes. (`eq(ρ, p)` is computed from the sampled `ρ` wires: there is
/// no hint wire to set it by.)
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
        let honest = onechain_faults::with(Fault::None, || program(&f));
        assert!(machine_runs(&f, &honest, words.clone()).is_ok());
        let trees = f.shape.chains(f.layout.num_polys())[0].len();
        for fault in [
            Fault::SwapRhoInCombine,
            Fault::SwapRhoInWeight,
            Fault::DropIndexFactor,
            Fault::AbsentSlotWeightOne,
        ] {
            if fault == Fault::AbsentSlotWeightOne && trees.is_power_of_two() {
                continue;
            }
            let faulty = onechain_faults::with(fault, || program(&f));
            assert!(
                machine_runs(&f, &faulty, words.clone()).is_err(),
                "{} {fault:?}: the faulty leg executed the honest arena",
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

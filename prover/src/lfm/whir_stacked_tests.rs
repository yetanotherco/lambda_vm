//! Gates for the stacked-evaluation weight, and for the wrapper around it.
//!
//! `weight_at` is private to the multilinear crate, so the gate is against the
//! public pair that DEFINES it — `weight_table`, the table the prover actually
//! commits, evaluated at the same point. That is a stronger comparison than the
//! host's own closed form would be: it catches an error the two closed forms
//! could share.
//!
//! The wrapper's gates are further down and are against `stacked_eval::verify`
//! itself, on a real `stacked_eval::prove` proof.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::transcript_hash::RpxTranscriptHash;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{
    Claimed, StackedCommitment, StackedProof, WeightShare, weight_table,
};
use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;
use multilinear::whir_chain::{ChainConfig, GrindBits};
use multilinear::whir_hash::RpxWhir;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::{commitment_to_digest, leaf_capacity};
use super::builder::{Cell, Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_chain::{ChainRoundWires, ChainShape, chain_hash_schedule};
use super::whir_chain_tests::{
    Recording, RoundStorage, chain_program, const_rows, hint_rows, perm_rows, push_round_words,
};
use super::whir_stacked::{
    ColumnClaim, StackedPolyWires, emit_stacked_verify, emit_weight_at, stacked_verify_cost,
    weight_at_consts, weight_at_rows,
};
use super::whir_transcript::{SpongeEntry, SpongeHash, WhirTranscript};
use super::word::{LfmWord, ext_word, word_as_ext};

type F = GoldilocksField;
type E = GoldilocksExtension;
type HostTranscript = DefaultTranscript<E, RpxTranscriptHash>;

fn fee(v: u64) -> FEE {
    FEE::new([
        FE::from(v.wrapping_mul(6364136223846793005) >> 11),
        FE::from(v ^ 0xA5A5),
        FE::from(v.wrapping_add(7)),
    ])
}

/// One shape under test: a layout, which claimed point each column shares, and
/// the number of variables each of those points carries.
struct Shape {
    name: &'static str,
    layout: StackedLayout,
    group_of: Vec<usize>,
    group_vars: Vec<usize>,
}

fn shape(name: &'static str, heights: &[usize], n_stack: usize, group_of: Vec<usize>) -> Shape {
    let layout = StackedLayout::build(heights, n_stack).expect("the layout packs");
    let groups = group_of.iter().copied().max().map(|g| g + 1).unwrap_or(0);
    let mut group_vars = vec![usize::MAX; groups];
    for (column, &group) in group_of.iter().enumerate() {
        let vars = layout.placements()[column].num_vars;
        if group_vars[group] == usize::MAX {
            group_vars[group] = vars;
        }
        assert_eq!(
            group_vars[group], vars,
            "{name}: columns sharing a point must share a height"
        );
    }
    Shape {
        name,
        layout,
        group_of,
        group_vars,
    }
}

/// The shapes: one polynomial with every column the same height and two
/// distinct points; columns of MIXED heights, so prefix lengths differ inside
/// one polynomial; and a stack that spills into more than one polynomial.
fn shapes() -> Vec<Shape> {
    vec![
        shape(
            "eight equal columns, two points",
            &[3; 8],
            6,
            (0..8).map(|c| c % 2).collect(),
        ),
        shape(
            "mixed heights, one point each",
            &[4, 3, 3, 2, 2, 5],
            6,
            vec![0, 1, 1, 2, 2, 3],
        ),
        shape(
            "twelve columns spilling into two polynomials",
            &[3; 12],
            6,
            (0..12).map(|c| c % 3).collect(),
        ),
        shape("one column filling the stack", &[5, 5], 5, vec![0, 1]),
    ]
}

fn arena_words(shape: &Shape) -> usize {
    shape.layout.n_stack() + shape.group_vars.iter().sum::<usize>() + shape.group_of.len()
}

/// Hints `at`, then each group's point, then every column's weight — the order
/// [`weight_arena`] fills.
fn weight_program(shape: &Shape, poly: usize) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(arena_words(shape) as u32);
    let mut idx = 0u32;
    let next = |b: &mut LfmBuilder, idx: &mut u32| -> Ext {
        let cell = b.hint_word(arena, *idx).as_ext();
        *idx += 1;
        cell
    };

    let mut at = Vec::with_capacity(shape.layout.n_stack());
    for _ in 0..shape.layout.n_stack() {
        at.push(next(&mut b, &mut idx));
    }
    let mut points: Vec<Vec<Ext>> = Vec::with_capacity(shape.group_vars.len());
    for &vars in &shape.group_vars {
        let mut point = Vec::with_capacity(vars);
        for _ in 0..vars {
            point.push(next(&mut b, &mut idx));
        }
        points.push(point);
    }
    let mut weights = Vec::with_capacity(shape.group_of.len());
    for _ in 0..shape.group_of.len() {
        weights.push(next(&mut b, &mut idx));
    }

    let claims: Vec<ColumnClaim<'_>> = shape
        .group_of
        .iter()
        .enumerate()
        .map(|(column, &group)| ColumnClaim {
            point: &points[group],
            weight: weights[column],
        })
        .collect();

    let weight = emit_weight_at(&mut b, &shape.layout, poly, &claims, &at);
    b.public(weight.as_cell());

    let program = compile(b.finish());
    validate(&program).expect("the weight leg must be admissible");
    program
}

fn weight_arena(at: &[FEE], points: &[Vec<FEE>], weights: &[FEE]) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = at.iter().map(ext_word).collect();
    for point in points {
        words.extend(point.iter().map(ext_word));
    }
    words.extend(weights.iter().map(ext_word));
    words
}

/// ★ F1 for the weight, on every shape and every polynomial of it.
#[test]
fn the_weight_leg_emits_its_closed_form() {
    for shape in shapes() {
        for poly in 0..shape.layout.num_polys() {
            let program = weight_program(&shape, poly);
            let measured = program.instrs.len() - arena_words(&shape) - 1;
            let predicted =
                weight_at_rows(&shape.layout, poly, &shape.group_of) + weight_at_consts();
            println!(
                "weight [{}] poly {poly}: {measured:>4} rows emitted, {predicted:>4} predicted",
                shape.name
            );
            assert_eq!(
                measured, predicted,
                "{}: polynomial {poly} must emit its closed form",
                shape.name
            );
        }
    }
}

/// ★ The leg computes what the committed weight table evaluates to.
#[test]
fn the_weight_leg_computes_what_the_host_computes() {
    for shape in shapes() {
        let n_stack = shape.layout.n_stack();
        for seed in [0x31u64, 0x77] {
            let at: Vec<FEE> = (0..n_stack).map(|i| fee(seed + i as u64)).collect();
            let points: Vec<Vec<FEE>> = shape
                .group_vars
                .iter()
                .enumerate()
                .map(|(group, &vars)| {
                    (0..vars)
                        .map(|i| fee(seed * 13 + (group * 100 + i) as u64))
                        .collect()
                })
                .collect();
            let weights: Vec<FEE> = (0..shape.group_of.len())
                .map(|c| fee(seed * 29 + 1 + c as u64))
                .collect();

            for poly in 0..shape.layout.num_polys() {
                let shares: Vec<WeightShare<'_, _>> = shape
                    .layout
                    .placements()
                    .iter()
                    .enumerate()
                    .filter(|(_, place)| place.poly == poly)
                    .map(|(column, place)| WeightShare {
                        offset: place.offset,
                        point: &points[shape.group_of[column]],
                        scale: weights[column],
                    })
                    .collect();
                let table: Mle<_> =
                    weight_table(&shares, n_stack).expect("the weight table builds");
                let want = table
                    .evaluate(&at)
                    .expect("the table takes the stack's point");

                let program = weight_program(&shape, poly);
                let arena = weight_arena(&at, &points, &weights);
                let exec = execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER)
                    .expect("the weight leg executes");
                let got = word_as_ext(&exec.public_words[0].1).expect("a published weight");
                assert_eq!(
                    got, want,
                    "{}: polynomial {poly} at seed {seed:#x} disagrees with the committed \
                     weight table",
                    shape.name
                );
            }
        }
    }
}

/// ★ `multilinear::challenge_powers` is REACHABLE from this crate, and means
/// what the batching weights need it to mean.
///
/// Without this, item 0's visibility change is a check that cannot fail: a
/// `pub` item with no consumer draws no warning and no test, so "the emitter
/// can gate against it" would be a claim nothing stands behind until the
/// stacked wiring lands. This is the cheapest thing that fails if the
/// visibility is reverted — it would not compile — and it pins the semantics
/// the γ-powers leg will be gated against: `[1, γ, γ², …]`, the FIRST weight
/// one and not γ, which is the end that is easy to get wrong.
#[test]
fn challenge_powers_is_reachable_and_starts_at_one() {
    let gamma = fee(0x51);
    let powers = multilinear::challenge_powers::<
        math::field::extensions_goldilocks::Degree3GoldilocksExtensionField,
    >(&gamma, 4);
    let want = [FEE::one(), gamma, gamma * gamma, gamma * gamma * gamma];
    assert_eq!(powers.len(), want.len(), "one weight per source");
    for (i, (got, expected)) in powers.iter().zip(&want).enumerate() {
        assert_eq!(got, expected, "gamma^{i}");
    }
}

// ───────────────────────── `stacked_eval::verify`, whole ─────────────────────
//
// The weight above is one closure inside it. What follows gates the wrapper:
// the column values absorbed, the batching draw, and one chain per stacked
// polynomial with the transcript THREADED through them.

/// One group under test: its columns' heights, the stack they pack into, which
/// columns settle at a shared point, and the chain posture.
struct Group {
    name: &'static str,
    heights: Vec<usize>,
    n_stack: usize,
    /// `group_of[column]` names the claimed point the column shares. Columns in
    /// one group must share a height, because a point is on the column's own
    /// cube.
    group_of: Vec<usize>,
    num_queries: usize,
    grind: u8,
}

/// The groups: one polynomial with a single shared point (the epoch's common
/// case — one table's sumcheck leaves all of its columns at one point); twelve
/// columns SPILLING into two polynomials, which is the only shape that pays the
/// threading twice; mixed heights, so prefix lengths differ inside one
/// polynomial and three distinct points are live at once; and one column
/// FILLING the stack, whose prefix indicator is empty.
///
/// ⚠ Grind 0 is not a smaller grind 8, for the chain suite's own reason: with no
/// query grind the query phase is entered with a candidate still in hand, which
/// is a branch of the schedule the production posture never takes.
fn groups() -> Vec<Group> {
    vec![
        Group {
            name: "six equal columns, one polynomial, one shared point",
            heights: vec![3; 6],
            n_stack: 6,
            group_of: vec![0; 6],
            num_queries: 3,
            grind: 8,
        },
        Group {
            name: "twelve columns spilling into two polynomials, three points",
            heights: vec![3; 12],
            n_stack: 6,
            group_of: (0..12).map(|c| c % 3).collect(),
            num_queries: 3,
            grind: 8,
        },
        Group {
            name: "mixed heights, three points, no grind",
            heights: vec![4, 3, 3, 2, 2],
            n_stack: 6,
            group_of: vec![0, 1, 1, 2, 2],
            num_queries: 3,
            grind: 0,
        },
        Group {
            name: "one column filling the stack",
            heights: vec![5],
            n_stack: 5,
            group_of: vec![0],
            num_queries: 3,
            grind: 8,
        },
    ]
}

fn group_config(group: &Group) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: group.num_queries,
        grind: GrindBits::uniform(group.grind),
    }
}

fn pseudo_column(num_vars: usize, seed: u64) -> Mle<F> {
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

/// Everything one group's run needs on both sides.
struct Fixture {
    name: &'static str,
    layout: StackedLayout,
    group_of: Vec<usize>,
    /// One claimed point per GROUP; columns of a group share its wires, which
    /// is what `emit_weight_at` groups on.
    points: Vec<Vec<FEE>>,
    values: Vec<FEE>,
    root_words: Vec<LfmWord>,
    proof: StackedProof<F, E>,
    domain: Domain<F>,
    shape: ChainShape,
    /// The batching challenge the HOST verifier drew, read off the recorder
    /// rather than recomputed.
    gamma: FEE,
    /// The hashes the host's own transcript performed, reconstructed from the
    /// calls a real `stacked_eval::verify` made.
    host_hashes: Vec<SpongeHash>,
}

/// Commits, proves and verifies one group on the host, keeping the verifier's
/// draws and its hash events.
///
/// ⚠ Under `RpxTranscriptHash` and `RpxWhir`, not the default keccak pair — the
/// machine replays that hash and no other, so a fixture on the default
/// transcript would be a fixture of a different protocol.
fn fixture(group: &Group) -> Fixture {
    let config = group_config(group);
    let layout = StackedLayout::build(&group.heights, group.n_stack).expect("the layout packs");
    let columns: Vec<Mle<F>> = group
        .heights
        .iter()
        .enumerate()
        .map(|(c, &height)| pseudo_column(height, 17 + 13 * c as u64))
        .collect();
    let borrowed = multilinear::stacking::borrow(&columns);

    let count = group.group_of.iter().copied().max().map_or(0, |g| g + 1);
    let mut points: Vec<Vec<FEE>> = Vec::with_capacity(count);
    for g in 0..count {
        let first = group
            .group_of
            .iter()
            .position(|&x| x == g)
            .expect("every group has a column");
        let vars = group.heights[first];
        for (column, &of) in group.group_of.iter().enumerate() {
            assert!(
                of != g || group.heights[column] == vars,
                "{}: columns sharing a point share a height",
                group.name
            );
        }
        points.push(
            (0..vars)
                .map(|i| fee(0x5eed + (g * 37 + i) as u64))
                .collect(),
        );
    }
    let per_column: Vec<Vec<FEE>> = group.group_of.iter().map(|&g| points[g].clone()).collect();
    let values: Vec<FEE> = columns
        .iter()
        .enumerate()
        .map(|(c, column)| {
            column
                .evaluate_in::<E>(&per_column[c])
                .expect("a column takes its own point")
        })
        .collect();

    let stacked = StackedCommitment::<F, RpxWhir>::commit(layout.clone(), &borrowed, None, &config)
        .expect("the group commits");
    let roots = stacked.roots();
    let mut proving = HostTranscript::new(&[]);
    let proof = multilinear::stacked_eval::prove::<F, E, _, RpxWhir>(
        &stacked,
        &borrowed,
        None,
        &Claimed::PerColumn(&per_column),
        &values,
        &config,
        &mut proving,
    )
    .expect("the group proves");

    let mut recorded = Recording::new();
    multilinear::stacked_eval::verify::<F, E, _, RpxWhir>(
        &proof,
        stacked.layout(),
        &roots,
        &Claimed::PerColumn(&per_column),
        &values,
        stacked.domain(),
        &config,
        &mut recorded,
    )
    .expect("the control proof must verify");

    // ★ The host's draw stream has the shape the emitter's order assumes: the
    // batching challenge FIRST, then each chain's own. Derived from the
    // structure, not read off the recorder.
    let shape = ChainShape::new(&config, layout.n_stack());
    let polys = layout.num_polys();
    assert_eq!(
        recorded.sampled.len(),
        1 + polys * (shape.num_vars + 2 * (shape.rounds() - 1)),
        "{}: extension draws are gamma, then one a sumcheck round plus z0 and gamma on each \
         round with a successor, in each of the {polys} chains",
        group.name
    );

    let host_hashes = recorded.duplex.borrow().hashes.clone();
    Fixture {
        name: group.name,
        layout,
        group_of: group.group_of.clone(),
        points,
        values,
        root_words: roots.iter().map(commitment_to_digest).collect(),
        proof,
        domain: stacked.domain().clone(),
        shape,
        gamma: recorded.sampled[0],
        host_hashes,
    }
}

/// Where every wire of the wrapper lives. Built once and used by both the
/// program and the arena filler, so the two cannot drift.
struct Arena {
    group_at: Vec<u32>,
    values_at: u32,
    poly_at: Vec<u32>,
    total: u32,
}

impl Arena {
    fn new(fixture: &Fixture) -> Self {
        let mut at = 0u32;
        let mut group_at = Vec::with_capacity(fixture.points.len());
        for point in &fixture.points {
            group_at.push(at);
            at += point.len() as u32;
        }
        let values_at = at;
        at += fixture.values.len() as u32;
        let mut poly_at = Vec::with_capacity(fixture.layout.num_polys());
        for _ in 0..fixture.layout.num_polys() {
            poly_at.push(at);
            // The root, the final value, then the chain's rounds.
            at += 2 + RoundStorage::words(&fixture.shape);
        }
        Self {
            group_at,
            values_at,
            poly_at,
            total: at,
        }
    }
}

/// Rows the wrapper PROGRAM carries that are not the leg's own cost: one `Hint`
/// per arena word, and the published batching challenge.
///
/// The root `Unpack`s and the weight closure are NOT subtracted here — unlike
/// the chain's plumbing, they belong to this leg and its form charges them.
fn plumbing(arena: &Arena) -> usize {
    arena.total as usize + 1
}

fn stacked_program(fixture: &Fixture) -> LfmProgram {
    let at = Arena::new(fixture);
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(at.total);
    let mut transcript = WhirTranscript::new();

    let points: Vec<Vec<Ext>> = fixture
        .points
        .iter()
        .enumerate()
        .map(|(g, point)| {
            (0..point.len() as u32)
                .map(|i| b.hint_word(arena, at.group_at[g] + i).as_ext())
                .collect()
        })
        .collect();
    let values: Vec<Ext> = (0..fixture.values.len() as u32)
        .map(|i| b.hint_word(arena, at.values_at + i).as_ext())
        .collect();

    let mut roots: Vec<Cell> = Vec::new();
    let mut finals: Vec<Ext> = Vec::new();
    let mut storage: Vec<RoundStorage> = Vec::new();
    for poly in 0..fixture.layout.num_polys() {
        let base = at.poly_at[poly];
        roots.push(b.hint_word(arena, base));
        finals.push(b.hint_word(arena, base + 1).as_ext());
        storage.push(RoundStorage::hint(&mut b, arena, base + 2, &fixture.shape));
    }

    let openings: Vec<_> = storage.iter().map(RoundStorage::openings).collect();
    let wires: Vec<Vec<ChainRoundWires<'_>>> = storage
        .iter()
        .zip(&openings)
        .map(|(chain, (current, next))| chain.wires(current, next))
        .collect();
    let polys: Vec<StackedPolyWires<'_>> = (0..fixture.layout.num_polys())
        .map(|poly| StackedPolyWires {
            rounds: &wires[poly],
            root: roots[poly],
            final_value: finals[poly],
        })
        .collect();
    let claimed_at: Vec<&[Ext]> = fixture
        .group_of
        .iter()
        .map(|&g| points[g].as_slice())
        .collect();

    let gamma = emit_stacked_verify(
        &mut b,
        &mut transcript,
        &fixture.layout,
        &polys,
        &claimed_at,
        &values,
        &fixture.shape,
        &fixture.domain,
    );
    b.public(gamma.as_cell());

    let program = compile(b.finish());
    validate(&program).expect("the stacked leg must be admissible");
    program
}

/// The arena in the order [`stacked_program`] hints it.
fn stacked_arena(fixture: &Fixture) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = Vec::new();
    for point in &fixture.points {
        words.extend(point.iter().map(ext_word));
    }
    words.extend(fixture.values.iter().map(ext_word));
    for (poly, chain) in fixture.proof.polys.iter().enumerate() {
        words.push(fixture.root_words[poly]);
        words.push(ext_word(&chain.final_value));
        push_round_words(&mut words, &fixture.shape, chain);
    }
    words
}

fn const_words(program: &LfmProgram) -> Vec<LfmWord> {
    program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            super::instr::Instr::Const { value, .. } => Some(*value),
            _ => None,
        })
        .collect()
}

/// ★ GATE ONE: the wrapper executes on a proof `stacked_eval::verify` accepts,
/// and draws the batching challenge the HOST drew.
///
/// Every challenge the machine derives feeds a refusal — the chain's final
/// check, its out-of-domain checks, its openings — so executing IS the stream
/// comparison. The published gamma turns the one thing that would otherwise
/// show up only as an absence of execution, the position of the draw relative
/// to the column absorbs, into a named assertion.
#[test]
fn the_stacked_verify_executes_on_a_proof_the_host_accepts() {
    for group in groups() {
        let fixture = fixture(&group);
        let program = stacked_program(&fixture);
        let arena = stacked_arena(&fixture);
        let exec =
            execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER).unwrap_or_else(|e| {
                panic!(
                    "{}: the machine must execute the host's proof: {e:?}",
                    fixture.name
                )
            });
        let gamma = word_as_ext(&exec.public_words[0].1).expect("a published challenge");
        assert_eq!(
            gamma, fixture.gamma,
            "{}: the batching challenge must be the one the host verifier sampled",
            fixture.name
        );
    }
}

/// ★ GATE TWO: the THREADED schedule is the host's own.
///
/// `HostDuplex` reconstructs the host's hash events from the calls a real
/// `stacked_eval::verify` makes, and is itself checked against a shadow
/// transcript squeezed at exactly those points. This compares the wrapper's
/// schedule against it event for event and felt count for felt count — which is
/// what makes the threading a measurement rather than a claim: a chain entered
/// fresh reads a different buffer at its very first hash.
#[test]
fn the_threaded_schedule_is_the_host_transcripts() {
    for group in groups() {
        let fixture = fixture(&group);
        let cost = stacked_verify_cost(
            &fixture.layout,
            &fixture.group_of,
            &fixture.shape,
            SpongeEntry::fresh(),
        );
        let mine = cost.schedule().hashes();
        let squeezes = mine
            .iter()
            .filter(|h| matches!(h, SpongeHash::Squeeze(_)))
            .count();
        println!(
            "stacked [{}]: {} chains, {squeezes} squeezes, {} state reads, {} schedule rows, \
             {} schedule permutations",
            fixture.name,
            cost.chains(),
            mine.len() - squeezes,
            cost.schedule().rows(),
            cost.schedule().perms(),
        );

        assert_eq!(
            mine.len(),
            fixture.host_hashes.len(),
            "{}: the host performs {} transcript hashes and the form says {}",
            fixture.name,
            fixture.host_hashes.len(),
            mine.len()
        );
        for (i, (derived, observed)) in mine.iter().zip(&fixture.host_hashes).enumerate() {
            assert_eq!(
                derived,
                observed,
                "{}: hash {i} of {} — the form says {derived:?} and the host's own transcript \
                 did {observed:?}",
                fixture.name,
                fixture.host_hashes.len()
            );
        }
    }
}

/// ★ GATE THREE: the rows and the permutations against the program emitted.
///
/// Pinned apart for the chain leg's reason: a form that moved work between the
/// sponge and the arithmetic at constant total would fail one of them rather
/// than neither.
#[test]
fn the_stacked_verify_emits_its_closed_form() {
    for group in groups() {
        let fixture = fixture(&group);
        let program = stacked_program(&fixture);
        let at = Arena::new(&fixture);
        let cost = stacked_verify_cost(
            &fixture.layout,
            &fixture.group_of,
            &fixture.shape,
            SpongeEntry::fresh(),
        );

        let hints = hint_rows(&program);
        assert_eq!(
            hints, at.total as usize,
            "{}: every arena word is hinted exactly once — the subtraction below is only honest \
             while this holds",
            fixture.name
        );
        let consts = const_rows(&program);
        let measured = program.instrs.len() - consts - plumbing(&at);
        let perms = perm_rows(&program);

        println!(
            "stacked [{}]: {measured} rows emitted ({consts} constants, {hints} hints), \
             {} predicted; {perms} permutations, {} predicted; {} instructions",
            fixture.name,
            cost.operations(),
            cost.perms(),
            program.instrs.len(),
        );

        assert_eq!(
            measured,
            cost.operations(),
            "{}: rows (const-free, the convention `chain_rows` is in)",
            fixture.name
        );
        assert_eq!(perms, cost.perms(), "{}: permutations", fixture.name);
    }
}

/// ★ GATE THREE's other half: the program interns EXACTLY the constants the
/// chain's arithmetic does, plus the two kinds this leg names.
///
/// Instance 63's question, asked at the wrapper: a count that comes up short is
/// a question and not a fudge, so this asserts the SET rather than the number.
/// Both sides are derived independently — the left from a one-chain program at
/// the same shape, with its own sponge's leaf capacities removed because a chain
/// entered FRESH hashes different lengths than one entered mid-transcript; the
/// right from `StackedCost::own_constants`.
///
/// ★ What it establishes for a census: the pool does NOT grow with the number of
/// chains. Every chain of a group has the same shape and the same domain, so
/// their constants collide in the one pool a program has — which is why a sum
/// over a group's chains adds a chain's constants ONCE and not once each.
#[test]
fn the_wrapper_interns_the_chains_constants_once_and_names_its_own() {
    for group in groups() {
        let fixture = fixture(&group);
        let program = stacked_program(&fixture);
        let cost = stacked_verify_cost(
            &fixture.layout,
            &fixture.group_of,
            &fixture.shape,
            SpongeEntry::fresh(),
        );

        let standalone = chain_program(&fixture.shape);
        let chain_leaves: Vec<LfmWord> = chain_hash_schedule(&fixture.shape, SpongeEntry::fresh())
            .iter()
            .map(|hash| leaf_capacity(hash.felts()))
            .collect();
        let arithmetic: Vec<LfmWord> = const_words(&standalone)
            .into_iter()
            .filter(|word| !chain_leaves.contains(word))
            .collect();

        let mut predicted = arithmetic.clone();
        for word in cost.own_constants() {
            if !predicted.contains(&word) {
                predicted.push(word);
            }
        }

        let interned = const_words(&program);
        println!(
            "stacked [{}]: {} constants interned, {} predicted ({} the chain's arithmetic + {} \
             named by this leg), over {} chains",
            fixture.name,
            interned.len(),
            predicted.len(),
            arithmetic.len(),
            cost.own_constants().len(),
            cost.chains(),
        );

        for word in &interned {
            assert!(
                predicted.contains(word),
                "{}: the program interns {word:?}, which the form does not name — ask which \
                 constant it is before adding a number",
                fixture.name
            );
        }
        for word in &predicted {
            assert!(
                interned.contains(word),
                "{}: the form names {word:?} and the program does not intern it",
                fixture.name
            );
        }
        assert_eq!(
            interned.len(),
            predicted.len(),
            "{}: the pools are the same size",
            fixture.name
        );
    }
}

/// ★ The PRODUCTION cross-check, from the form alone: the DECODE group's
/// transcript counters as the box measured them.
///
/// Opening DECODE's out-of-band commitment is a WHIR chain per epoch inside the
/// stacked wrapper (`n_stack` 23, five columns, one polynomial), and the box's
/// `hash-metrics` counters read +202 squeezes and +17 state reads per epoch over
/// the run without it — the second production point recorded in
/// `V1-handoff-2026-09-18-V1d.md`. Those are `stacked_eval::verify`'s totals and
/// not the chain's: a chain-only form is one squeeze short at that boundary,
/// which is the boundary and not a discrepancy.
///
/// Needs no proof, no commitment and no guest ELF — the schedule is a function
/// of the round structure, the column count and the polynomial count. ⚠ The
/// column HEIGHTS below are not DECODE's; they are any heights that pack five
/// columns into one polynomial of the right stack, and no term of the schedule
/// reads them.
#[test]
fn the_decode_groups_threaded_schedule_reproduces_the_box() {
    // The posture the whole VM proof runs at, read from `chain_config`
    // (`multilinear_prove.rs:93`): blowup 2, fold 4, 128 bits, 20-bit grinds,
    // with the query count sized by the TALLEST stack in the proof, 25.
    let config = ChainConfig::with_security(2, 4, 25, 128, GrindBits::uniform(20));
    let layout = StackedLayout::build(&[20; 5], 23).expect("five columns pack into one stack");
    assert_eq!(
        layout.num_polys(),
        1,
        "the DECODE group is one stacked polynomial"
    );
    let shape = ChainShape::new(&config, 23);
    assert_eq!(shape.rounds(), 6, "n_stack 23 at fold 4 is six rounds");
    assert_eq!(
        shape.num_queries, 112,
        "the shipped posture's query count for a seven-round tallest stack"
    );

    let cost = stacked_verify_cost(&layout, &[0; 5], &shape, SpongeEntry::fresh());
    let hashes = cost.schedule().hashes();
    let squeezes = hashes
        .iter()
        .filter(|hash| matches!(hash, SpongeHash::Squeeze(_)))
        .count();
    let states = hashes.len() - squeezes;
    println!(
        "DECODE group: {squeezes} squeezes, {states} state reads, {} schedule rows, \
         {} permutations whole, chain rows {}",
        cost.schedule().rows(),
        cost.perms(),
        cost.operations(),
    );

    assert_eq!(
        squeezes, 202,
        "the chain's 201 plus the wrapper's own batching draw"
    );
    assert_eq!(states, 17, "3R - 1 grinds, each reading the sponge once");
}

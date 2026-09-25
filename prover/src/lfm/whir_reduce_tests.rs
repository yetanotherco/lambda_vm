//! Gates for the claim-reduce leg.
//!
//! The proof under test is a real one: `claim_reduce::prove` over real columns,
//! under the RPX transcript the machine replays, with the factor values taken
//! from the host's own `evaluate_source` so the claim being reduced is the one
//! the columns actually hold.
//!
//! # Why executing the honest proof pins the transcript SEQUENCE too
//!
//! The leg draws γ and one challenge per sumcheck round, and every one of them
//! feeds a refusal: γ weights the batch the sumcheck is run on, and a round's
//! challenge is where the next round's claim is taken. A leg that absorbed in
//! the wrong order, or drew a challenge one absorb early, would draw different
//! values from the same proof and the residual would not close — so the
//! machine would refuse a proof the host accepts, which is what the first gate
//! asserts cannot happen.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::transcript_hash::RpxTranscriptHash;
use multilinear::claim_reduce::{self, FactorSource, ReduceProof, ReducedClaim};
use multilinear::mle::Mle;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_reduce::{REDUCE_DEGREE, ReduceWires, claim_reduce_rows, emit_claim_reduce_verify};
use super::whir_transcript::{
    COORDINATES_PER_EXT, SpongeEntry, SpongeSchedule, WhirTranscript, absorb_unpack_rows,
};
use super::word::{LfmWord, ext_word, word_as_ext};

type F = GoldilocksField;
type E = GoldilocksExtension;
type HostTranscript = DefaultTranscript<E, RpxTranscriptHash>;

fn pseudo(seed: u64, count: usize) -> Vec<FE> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            FE::from(state >> 2)
        })
        .collect()
}

fn ext_point(seed: u64, n: usize) -> Vec<FEE> {
    pseudo(seed, 3 * n)
        .chunks(3)
        .map(|c| FEE::new([c[0], c[1], c[2]]))
        .collect()
}

/// The factor layouts the gate runs at.
///
/// A VM table reads its columns at the row and at the row after, so offsets
/// `{0, 1}` is the production case; `{0, 2, 3}` is here because a form that
/// only ever saw one kernel per offset could not tell "per offset" from "per
/// factor", and because an offset whose first factor is not factor zero is the
/// case where the `γ^0 = 1` saving does NOT apply.
fn layouts() -> Vec<(usize, usize, Vec<FactorSource>)> {
    vec![
        (3, 2, vec![FactorSource::direct(0), FactorSource::direct(1)]),
        (
            4,
            3,
            vec![
                FactorSource::direct(0),
                FactorSource::direct(1),
                FactorSource::shifted(1, 1),
                FactorSource::direct(2),
                FactorSource::shifted(2, 1),
            ],
        ),
        (
            5,
            4,
            vec![
                FactorSource::direct(0),
                FactorSource::shifted(0, 2),
                FactorSource::direct(1),
                FactorSource::shifted(2, 3),
                FactorSource::direct(3),
                FactorSource::shifted(3, 2),
            ],
        ),
    ]
}

struct Fixture {
    proof: ReduceProof<E>,
    factor_values: Vec<FEE>,
    alpha: Vec<FEE>,
    claim: ReducedClaim<E>,
}

/// Proves and verifies one reduction on the host, keeping what both sides saw.
fn fixture(num_vars: usize, num_columns: usize, sources: &[FactorSource]) -> Fixture {
    let columns: Vec<Mle<F>> = (0..num_columns)
        .map(|c| {
            Mle::new(pseudo(0x51ced + c as u64, 1 << num_vars)).expect("a power-of-two column")
        })
        .collect();
    let alpha = ext_point(0x5EED_u64, num_vars);
    // The claims being reduced must be the ones the columns hold, or the
    // reduction is a proof about nothing.
    let factor_values: Vec<FEE> = sources
        .iter()
        .map(|source| {
            claim_reduce::evaluate_source::<F, E>(&columns, source, &alpha)
                .expect("the host evaluates its own factor")
        })
        .collect();

    let mut proving = HostTranscript::new(&[]);
    let (proof, _) = claim_reduce::prove::<F, E, _>(
        &columns,
        sources,
        &factor_values,
        &alpha,
        None,
        &mut proving,
    )
    .expect("the reduction proves");

    let mut verifying = HostTranscript::new(&[]);
    let claim = claim_reduce::verify::<E, _>(
        &proof,
        sources,
        &factor_values,
        &alpha,
        num_columns,
        &mut verifying,
    )
    .expect("the control proof must verify");

    Fixture {
        proof,
        factor_values,
        alpha,
        claim,
    }
}

/// Where each wire lives in the arena, used by the program and the filler so
/// the two cannot drift.
struct Layout {
    factors: usize,
    num_vars: usize,
    num_columns: usize,
}

impl Layout {
    fn sumcheck_at(&self, round: usize, which: usize) -> u32 {
        (self.factors + REDUCE_DEGREE * round + which) as u32
    }
    fn column_at(&self, column: usize) -> u32 {
        (self.factors + REDUCE_DEGREE * self.num_vars + column) as u32
    }
    fn alpha_at(&self, variable: usize) -> u32 {
        (self.factors + REDUCE_DEGREE * self.num_vars + self.num_columns + variable) as u32
    }
    fn total(&self) -> u32 {
        self.alpha_at(self.num_vars)
    }
}

fn reduce_program(num_vars: usize, num_columns: usize, sources: &[FactorSource]) -> LfmProgram {
    let layout = Layout {
        factors: sources.len(),
        num_vars,
        num_columns,
    };
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(layout.total());
    let mut transcript = WhirTranscript::new();

    let factor_values: Vec<_> = (0..sources.len())
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let sumcheck: Vec<Vec<_>> = (0..num_vars)
        .map(|round| {
            (0..REDUCE_DEGREE)
                .map(|which| {
                    b.hint_word(arena, layout.sumcheck_at(round, which))
                        .as_ext()
                })
                .collect()
        })
        .collect();
    let column_values: Vec<_> = (0..num_columns)
        .map(|c| b.hint_word(arena, layout.column_at(c)).as_ext())
        .collect();
    let alpha: Vec<_> = (0..num_vars)
        .map(|v| b.hint_word(arena, layout.alpha_at(v)).as_ext())
        .collect();

    let wires = ReduceWires {
        sumcheck: &sumcheck,
        column_values: &column_values,
    };
    let reduced = emit_claim_reduce_verify(
        &mut b,
        &mut transcript,
        &wires,
        sources,
        &factor_values,
        &alpha,
        num_columns,
    );
    for value in &reduced.point {
        b.public(value.as_cell());
    }

    let program = compile(b.finish());
    validate(&program).expect("the reduce leg must be admissible");
    program
}

fn reduce_arena(f: &Fixture, proof: &ReduceProof<E>, num_columns: usize) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = f.factor_values.iter().map(ext_word).collect();
    for round in &proof.sumcheck.rounds {
        assert_eq!(round.evaluations.len(), REDUCE_DEGREE);
        words.extend(round.evaluations.iter().map(ext_word));
    }
    assert_eq!(proof.column_values.len(), num_columns);
    words.extend(proof.column_values.iter().map(ext_word));
    words.extend(f.alpha.iter().map(ext_word));
    words
}

/// The sponge's own hashing for this leg, derived from its absorbs and draws.
///
/// Every absorb here is one extension element — the factor values, the
/// sumcheck's evaluations, the column values — and every draw is one extension
/// challenge. The leg reads no `state()`, because it spends no grind.
fn reduce_schedule(num_vars: usize, num_columns: usize, factors: usize) -> SpongeSchedule {
    let mut sponge = SpongeSchedule::new(SpongeEntry::fresh());
    for _ in 0..factors {
        sponge.absorb(COORDINATES_PER_EXT);
    }
    sponge.draw_ext();
    for _ in 0..num_vars {
        for _ in 0..REDUCE_DEGREE {
            sponge.absorb(COORDINATES_PER_EXT);
        }
        sponge.draw_ext();
    }
    for _ in 0..num_columns {
        sponge.absorb(COORDINATES_PER_EXT);
    }
    sponge
}

fn const_rows(program: &LfmProgram) -> usize {
    program
        .instrs
        .iter()
        .filter(|i| matches!(i, super::instr::Instr::Const { .. }))
        .count()
}

/// ★ The leg executes on a proof the host accepts, and lands on the host's
/// point.
///
/// The point is what the stacked opening settles every column at, so a leg that
/// executed but reduced to a different point would be a verifier that proves
/// the wrong claim. It is published and compared rather than left to the
/// execution.
#[test]
fn the_reduce_leg_executes_and_lands_where_the_host_does() {
    for (num_vars, num_columns, sources) in layouts() {
        let f = fixture(num_vars, num_columns, &sources);
        let program = reduce_program(num_vars, num_columns, &sources);
        let arena = reduce_arena(&f, &f.proof, num_columns);
        let exec =
            execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER).unwrap_or_else(|e| {
                panic!(
                    "{num_vars} vars, {} factors: the machine refused an accepted proof: {e:?}",
                    sources.len()
                )
            });
        let point: Vec<FEE> = exec
            .public_words
            .iter()
            .map(|(_, w)| word_as_ext(w).expect("a published extension value"))
            .collect();
        assert_eq!(
            point, f.claim.point,
            "{num_vars} vars: the reduced point must be the host's"
        );
        println!(
            "reduce {num_vars} vars, {} factors, {num_columns} columns: {} instructions",
            sources.len(),
            program.instrs.len()
        );
    }
}

/// ★ The tamper arm, in three halves at three sites.
///
/// Each half earns its place: the untouched proof must execute, or a refusal
/// below is a refusal of everything; the HOST must reject the same forgery, or
/// the machine is refusing something valid and the finding is a completeness
/// bug rather than a soundness success; and only then does the machine's
/// refusal say anything.
#[test]
fn a_tampered_reduction_cannot_execute() {
    let (num_vars, num_columns, sources) = layouts()[1].clone();
    let f = fixture(num_vars, num_columns, &sources);
    let program = reduce_program(num_vars, num_columns, &sources);

    assert!(
        execute(
            &program,
            &[reduce_arena(&f, &f.proof, num_columns)],
            &crate::hash_pin::BLOCK_HASHER
        )
        .is_ok(),
        "the ground proof must execute"
    );

    let mut refused = 0;
    for site in ["column", "evaluation", "factor"] {
        let mut forged = f.proof.clone();
        let mut factor_values = f.factor_values.clone();
        match site {
            "column" => forged.column_values[1] += FEE::one(),
            "evaluation" => forged.sumcheck.rounds[0].evaluations[1] += FEE::one(),
            _ => factor_values[2] += FEE::one(),
        }

        let mut verifying = HostTranscript::new(&[]);
        assert!(
            claim_reduce::verify::<E, _>(
                &forged,
                &sources,
                &factor_values,
                &f.alpha,
                num_columns,
                &mut verifying,
            )
            .is_err(),
            "the host must reject the forgery at {site}, or the machine's refusal refuses \
             something the host accepts"
        );

        let mut arena = reduce_arena(&f, &forged, num_columns);
        for (i, value) in factor_values.iter().enumerate() {
            arena[i] = ext_word(value);
        }
        assert!(
            execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER).is_err(),
            "the machine must refuse the forgery at {site}"
        );
        refused += 1;
    }
    println!("reduce tamper arm: {refused} sites, all refused");
}

/// ★ F1 for the leg: its rows against the closed form, with the sponge's own
/// hashing derived separately from the absorbs and draws.
///
/// The two halves are separated for the reason the chain's are: the schedule is
/// what the sponge's BUFFER holds at each hash, which the leg's arithmetic
/// cannot see.
#[test]
fn the_reduce_leg_emits_its_closed_form() {
    for (num_vars, num_columns, sources) in layouts() {
        let program = reduce_program(num_vars, num_columns, &sources);
        let layout = Layout {
            factors: sources.len(),
            num_vars,
            num_columns,
        };
        let schedule = reduce_schedule(num_vars, num_columns, sources.len());
        // The plumbing: one `Hint` per arena word, and one `Public` per
        // published coordinate of the reduced point.
        let plumbing = layout.total() as usize + num_vars;
        let consts = const_rows(&program);
        let measured = program.instrs.len() - consts - plumbing;
        let predicted = claim_reduce_rows(&sources, num_columns, num_vars) + schedule.rows();
        println!(
            "reduce {num_vars} vars, {} factors, {num_columns} columns: {measured} rows \
             ({} leg + {} schedule predicted {predicted}), {consts} constants, \
             {} permutations",
            sources.len(),
            claim_reduce_rows(&sources, num_columns, num_vars),
            schedule.rows(),
            schedule.perms(),
        );
        assert_eq!(
            measured, predicted,
            "{num_vars} vars, {num_columns} columns"
        );
    }
}

/// The absorb count the schedule is built from is the one the leg emits: one
/// `Unpack` per absorbed extension element, and the leg absorbs the factor
/// values, the sumcheck's evaluations and the column values and nothing else.
///
/// Stated as its own pin because the schedule above would otherwise be a
/// second, unchecked description of the same sequence.
#[test]
fn the_reduce_leg_absorbs_exactly_what_it_reads() {
    let (num_vars, num_columns, sources) = layouts()[2].clone();
    let program = reduce_program(num_vars, num_columns, &sources);
    let unpacks = program
        .instrs
        .iter()
        .filter(|i| matches!(i, super::instr::Instr::Unpack { .. }))
        .count();
    let schedule = reduce_schedule(num_vars, num_columns, sources.len());
    let absorbed = sources.len() + REDUCE_DEGREE * num_vars + num_columns;
    // One `Unpack` per absorbed element, plus one per squeeze reading its
    // digest — the two are the only `Unpack`s the leg emits.
    assert_eq!(
        unpacks,
        absorbed * absorb_unpack_rows() + schedule.hashes().len(),
        "every `Unpack` is either an absorbed element or a squeeze's digest"
    );
}

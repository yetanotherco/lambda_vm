//! Gates for the batched argue's leg (D-BATCH B-4a), against REAL batched
//! argues from the host reference (`stark::multilinear_table::batched`).
//!
//! Three tables of different heights — two with a shifted read (so a claim
//! reduction), one without — argued in one bin and in several, the leg run
//! over the proof values, and its verdict compared with the host's
//! `verify_argue`. The closed form is gated as the per-table leg's is: the
//! program's marginal rows against `batched_argue_cost`.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::transcript_hash::RpxTranscriptHash;
use multilinear::whir_chain::ArgueFormat;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet, RowDomain};
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::multilinear_air::Uniforms;
use stark::multilinear_logup::{InteractionShape, interaction_shapes};
use stark::multilinear_table::batched::{
    BatchedArgue, ProverFaults, VerifierChecks, Where, prove_argue, verify_argue,
};
use stark::multilinear_table::{ArguePlan, ArgueShape, CommittedTable, TableLayout, argue_plan};
use stark::traits::AIR;

use crate::tables::types::{FE, FEE};

use super::builder::{Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_batch::{
    BatchedArgueWires, LadderStepWires, batched_argue_cost, emit_batched_argue,
};
use super::whir_bus::alpha_powers_read;
use super::whir_reduce::ReduceWires;
use super::whir_table::TableShape;
use super::whir_transcript::{SpongeEntry, WhirTranscript};
use super::word::{LfmWord, ext_word, word_as_ext};

type F = crate::tables::types::GoldilocksField;
type E = crate::tables::types::GoldilocksExtension;
type HostTranscript = DefaultTranscript<E, RpxTranscriptHash>;

/// One `Public` instruction per published value.
const PUBLISH_ROW: usize = 1;

fn pseudo(seed: u64, count: usize) -> Vec<FEE> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        FE::from(state >> 2)
    };
    (0..count)
        .map(|_| FEE::new([next(), next(), next()]))
        .collect()
}

/// `c = a·b`, `d = a + b` on every row and — when `shifted` — `next(a) = a + 1`
/// off the last row: a public selector and a shifted read.
struct Rules {
    shifted: bool,
}

const COLUMNS: usize = 4;

impl ConstraintSet<F, E> for Rules {
    fn max_degree(&self) -> usize {
        2
    }

    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let a = b.main(0, 0);
        let y = b.main(0, 1);
        let c = b.main(0, 2);
        b.emit_base(0, a * y - c);
        let a = b.main(0, 0);
        let y = b.main(0, 1);
        let d = b.main(0, 3);
        b.emit_base(1, a + y - d);
        if self.shifted {
            let one = b.one();
            let next = b.main(1, 0);
            let here = b.main(0, 0);
            b.emit_base_rows(2, RowDomain::except_last(1), next - here - one);
        }
    }
}

fn buses() -> Vec<BusInteraction> {
    vec![
        BusInteraction::sender(
            0u64,
            Multiplicity::Column(1),
            Packing::Direct.columns(&[0, 2]),
        ),
        BusInteraction::receiver(
            0u64,
            Multiplicity::Column(1),
            Packing::Direct.columns(&[0, 3]),
        ),
    ]
}

fn columns(num_vars: usize) -> Vec<Vec<FE>> {
    let rows = 1usize << num_vars;
    let b = 7u64;
    vec![
        (0..rows).map(|i| FE::from(i as u64)).collect(),
        (0..rows).map(|_| FE::from(b)).collect(),
        (0..rows).map(|i| FE::from(i as u64 * b)).collect(),
        (0..rows).map(|i| FE::from(i as u64 + b)).collect(),
    ]
}

type Air = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), Rules>;

fn air(shifted: bool) -> Air {
    AirWithBuses::new(
        COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: buses(),
        },
        &stark::proof::options::GoldilocksCubicProofOptions::with_params(4, 128, 20)
            .expect("valid options"),
        1,
        Rules { shifted },
    )
}

fn challenges() -> (FEE, FEE, FEE) {
    let drawn = pseudo(0xBA7C4, 3);
    (drawn[0], drawn[1], drawn[2])
}

/// The fixture: `(air, num_vars)` per table.
struct Fixture {
    airs: Vec<(Air, usize)>,
}

impl Fixture {
    /// A shifted table of 2^3 rows, an unshifted one of 2^4, a shifted one of
    /// 2^5: every table shorter than another but the last.
    fn new() -> Self {
        Self {
            airs: vec![(air(true), 3), (air(false), 4), (air(true), 5)],
        }
    }

    fn layouts(&self) -> Vec<TableLayout<'_, F, E>> {
        self.airs
            .iter()
            .map(|(air, n)| {
                TableLayout::<F, E>::new(
                    air.constraint_program(),
                    air.constraints_meta(),
                    air.bus_interactions(),
                    COLUMNS,
                    *n,
                    Uniforms::default(),
                )
                .expect("the table lays out")
            })
            .collect()
    }

    fn buses(&self, layouts: &[TableLayout<'_, F, E>]) -> Vec<Vec<InteractionShape<E>>> {
        self.airs
            .iter()
            .zip(layouts)
            .map(|((air, _), layout)| {
                let slots = layout.slot_of().to_vec();
                interaction_shapes(air.bus_interactions(), COLUMNS, |column| {
                    slots
                        .get(column)
                        .copied()
                        .ok_or(multilinear::Error::UnknownPolynomial {
                            index: column,
                            len: slots.len(),
                        })
                })
                .expect("the bus probes")
            })
            .collect()
    }

    fn argue(&self, cap: u8) -> BatchedArgue<E> {
        let tables: Vec<CommittedTable<'_, F, E>> = self
            .airs
            .iter()
            .map(|(air, n)| {
                let cols = columns(*n);
                CommittedTable::<F, E>::new(
                    air.constraint_program(),
                    air.constraints_meta(),
                    air.bus_interactions(),
                    COLUMNS,
                    *n,
                    Uniforms::default(),
                    move |col| cols[col as usize].clone(),
                )
                .expect("the table lays out")
            })
            .collect();
        let (z, alpha, beta) = challenges();
        let mut prover = HostTranscript::new(&[]);
        prove_argue(
            &tables,
            &z,
            &alpha,
            &beta,
            cap,
            &mut prover,
            ProverFaults::default(),
            Where::Host,
        )
        .expect("the tables argue")
        .0
    }
}

fn plan_of(layouts: &[TableLayout<'_, F, E>], cap: u8) -> ArguePlan {
    let shapes: Vec<ArgueShape> = layouts
        .iter()
        .map(|l| ArgueShape::of(&l.statement()))
        .collect();
    argue_plan(&shapes, cap)
}

fn alpha_ladder(buses: &[Vec<InteractionShape<E>>], alpha: FEE) -> Vec<FEE> {
    let count = buses
        .iter()
        .map(|b| alpha_powers_read(b))
        .max()
        .unwrap_or(0);
    let mut powers = Vec::with_capacity(count);
    let mut power = FEE::one();
    for _ in 0..count {
        powers.push(power);
        power *= alpha;
    }
    powers
}

/// Every value the program hints, in the order [`batched_program`] takes them.
fn flatten(argue: &BatchedArgue<E>, z: FEE, alpha_powers: &[FEE], beta: FEE) -> Vec<FEE> {
    let mut values = vec![z];
    values.extend_from_slice(alpha_powers);
    values.push(beta);
    for (p, q) in &argue.bus_outputs {
        values.extend([*p, *q]);
    }
    for ladder in &argue.gkr {
        for step in &ladder.layers {
            for round in &step.sumcheck.rounds {
                values.extend(round.evaluations.iter().copied());
            }
            for h in &step.halves {
                values.extend([h.p_lo, h.p_hi, h.q_lo, h.q_hi]);
            }
        }
    }
    for round in &argue.constraint.rounds {
        values.extend(round.evaluations.iter().copied());
    }
    for fv in &argue.factor_values {
        values.extend(fv.iter().copied());
    }
    for reduce in argue.reduces.iter().flatten() {
        for round in &reduce.sumcheck.rounds {
            values.extend(round.evaluations.iter().copied());
        }
        values.extend(reduce.column_values.iter().copied());
    }
    values
}

/// A reduction's rounds and column values, as wires.
type ReduceParts = (Vec<Vec<Ext>>, Vec<Ext>);

/// The assembled program over the hinted proof, or — with `leg` false — the
/// same arena with no leg, so the difference is the leg.
#[allow(clippy::too_many_arguments)]
fn batched_program(
    argue: &BatchedArgue<E>,
    shapes: &[TableShape<'_>],
    plan: &ArguePlan,
    alpha_count: usize,
    total: usize,
    leg: bool,
) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(total as u32);
    let mut index = 0u32;
    let mut take = |b: &mut LfmBuilder, count: usize| -> Vec<Ext> {
        (0..count)
            .map(|_| {
                let wire = b.hint_word(arena, index).as_ext();
                index += 1;
                wire
            })
            .collect()
    };
    let z = take(&mut b, 1)[0];
    let alpha_powers = take(&mut b, alpha_count);
    let beta = take(&mut b, 1)[0];
    let outputs: Vec<(Ext, Ext)> = argue
        .bus_outputs
        .iter()
        .map(|_| {
            let pq = take(&mut b, 2);
            (pq[0], pq[1])
        })
        .collect();
    let gkr: Vec<Vec<LadderStepWires>> = argue
        .gkr
        .iter()
        .map(|ladder| {
            ladder
                .layers
                .iter()
                .map(|step| LadderStepWires {
                    sumcheck: step
                        .sumcheck
                        .rounds
                        .iter()
                        .map(|round| take(&mut b, round.evaluations.len()))
                        .collect(),
                    halves: step
                        .halves
                        .iter()
                        .map(|_| {
                            let h = take(&mut b, 4);
                            [h[0], h[1], h[2], h[3]]
                        })
                        .collect(),
                })
                .collect()
        })
        .collect();
    let constraint: Vec<Vec<Ext>> = argue
        .constraint
        .rounds
        .iter()
        .map(|round| take(&mut b, round.evaluations.len()))
        .collect();
    let factor_values: Vec<Vec<Ext>> = argue
        .factor_values
        .iter()
        .map(|fv| take(&mut b, fv.len()))
        .collect();
    let reduce_parts: Vec<Option<ReduceParts>> = argue
        .reduces
        .iter()
        .map(|reduce| {
            reduce.as_ref().map(|reduce| {
                let rounds: Vec<Vec<Ext>> = reduce
                    .sumcheck
                    .rounds
                    .iter()
                    .map(|round| take(&mut b, round.evaluations.len()))
                    .collect();
                let columns = take(&mut b, reduce.column_values.len());
                (rounds, columns)
            })
        })
        .collect();

    if !leg {
        b.public(z.as_cell());
        b.public(beta.as_cell());
        return compile(b.finish());
    }
    let reduces: Vec<Option<ReduceWires<'_>>> = reduce_parts
        .iter()
        .map(|part| {
            part.as_ref().map(|(rounds, columns)| ReduceWires {
                sumcheck: rounds,
                column_values: columns,
            })
        })
        .collect();
    let mut transcript = WhirTranscript::new();
    let verdict = emit_batched_argue(
        &mut b,
        &mut transcript,
        &BatchedArgueWires {
            bus_outputs: &outputs,
            gkr: &gkr,
            constraint: &constraint,
            factor_values: &factor_values,
            reduces: &reduces,
        },
        shapes,
        plan,
        z,
        &alpha_powers,
        beta,
    );
    for (p, q) in &verdict.bus_outputs {
        b.public(p.as_cell());
        b.public(q.as_cell());
    }
    for (point, columns) in &verdict.tables {
        for wire in point.iter().chain(columns) {
            b.public(wire.as_cell());
        }
    }
    compile(b.finish())
}

fn words(values: &[FEE]) -> Vec<Vec<LfmWord>> {
    vec![values.iter().map(ext_word).collect()]
}

/// Everything one run needs, built from the fixture at `cap`.
fn with_run<R>(
    cap: u8,
    tamper: impl FnOnce(&mut BatchedArgue<E>),
    check: impl FnOnce(RunView<'_>) -> R,
) -> R {
    let fixture = Fixture::new();
    let layouts = fixture.layouts();
    let buses = fixture.buses(&layouts);
    let plan = plan_of(&layouts, cap);
    let shapes: Vec<TableShape<'_>> = layouts
        .iter()
        .zip(&buses)
        .map(|(layout, bus)| TableShape {
            ir: layout.shape(),
            bus,
            kinds: layout.kinds(),
            num_columns: layout.num_columns(),
            num_vars: layout.num_vars(),
        })
        .collect();
    let mut argue = fixture.argue(cap);
    tamper(&mut argue);
    let (z, alpha, beta) = challenges();
    let alpha_powers = alpha_ladder(&buses, alpha);
    let statements: Vec<_> = layouts.iter().map(|l| l.statement()).collect();
    let host = {
        let mut verifier = HostTranscript::new(&[]);
        verify_argue(
            &argue,
            &statements,
            cap,
            &z,
            &alpha,
            &beta,
            &mut verifier,
            VerifierChecks::ALL,
        )
        .map_err(|e| format!("{e:?}"))
    };
    check(RunView {
        argue: &argue,
        shapes: &shapes,
        plan: &plan,
        z,
        beta,
        alpha_powers: &alpha_powers,
        host,
    })
}

struct RunView<'a> {
    argue: &'a BatchedArgue<E>,
    shapes: &'a [TableShape<'a>],
    plan: &'a ArguePlan,
    z: FEE,
    beta: FEE,
    alpha_powers: &'a [FEE],
    host: Result<Vec<multilinear::claim_reduce::ReducedClaim<E>>, String>,
}

impl RunView<'_> {
    fn values(&self) -> Vec<FEE> {
        flatten(self.argue, self.z, self.alpha_powers, self.beta)
    }

    fn program(&self, leg: bool) -> LfmProgram {
        batched_program(
            self.argue,
            self.shapes,
            self.plan,
            self.alpha_powers.len(),
            self.values().len(),
            leg,
        )
    }

    fn execute(&self) -> Result<Vec<FEE>, String> {
        let program = self.program(true);
        validate(&program).map_err(|e| format!("invalid: {e:?}"))?;
        execute(
            &program,
            &words(&self.values()),
            &crate::hash_pin::BLOCK_HASHER,
        )
        .map(|exec| {
            exec.public_words
                .iter()
                .map(|(_, word)| word_as_ext(word).expect("a published extension value"))
                .collect()
        })
        .map_err(|e| format!("{e:?}"))
    }
}

/// ★ B-4a G1 — the leg computes what the host's `verify_argue` computes, in
/// one bin and with every table alone: the bus outputs, and every table's
/// column point and values.
#[test]
fn the_batched_leg_computes_what_the_host_computes() {
    for cap in [27u8, 0] {
        with_run(
            cap,
            |_| {},
            |run| {
                let host = run.host.clone().expect("the host verifies its own argue");
                let got = run
                    .execute()
                    .unwrap_or_else(|e| panic!("cap {cap}: the leg must execute: {e}"));
                let mut want: Vec<FEE> = Vec::new();
                for (p, q) in &run.argue.bus_outputs {
                    want.extend([*p, *q]);
                }
                for claim in &host {
                    want.extend(claim.point.iter().copied());
                    want.extend(claim.column_values.iter().copied());
                }
                assert_eq!(got, want, "cap {cap}: the leg's verdict is the host's");
                println!(
                    "batched leg cap={cap}: bins {:?}, n_max {}, D_max {}, reductions {}",
                    run.plan.bins,
                    run.plan.num_vars,
                    run.plan.degree,
                    run.argue.reduces.iter().flatten().count()
                );
            },
        );
    }
}

/// ★ B-4a G2 — the closed form is the emitted census: the program's marginal
/// rows against `batched_argue_cost`, and the form names every constant.
#[test]
fn the_batched_leg_emits_its_closed_form() {
    for cap in [27u8, 0] {
        with_run(
            cap,
            |_| {},
            |run| {
                let with = run.program(true);
                let without = run.program(false);
                let published = 2 * run.argue.bus_outputs.len()
                    + run
                        .host
                        .as_ref()
                        .expect("host")
                        .iter()
                        .map(|c| c.point.len() + c.column_values.len())
                        .sum::<usize>();
                let measured =
                    with.instrs.len() - without.instrs.len() - (published - 2) * PUBLISH_ROW;
                let cost = batched_argue_cost(run.shapes, run.plan, SpongeEntry::fresh());
                let predicted = cost.leg.constant_values();
                let mut unnamed: Vec<LfmWord> = Vec::new();
                for instr in &with.instrs {
                    if let super::instr::Instr::Const { value, .. } = instr
                        && !predicted.contains(value)
                        && !unnamed.contains(value)
                    {
                        unnamed.push(*value);
                    }
                }
                assert!(
                    unnamed.is_empty(),
                    "cap {cap}: the form does not name {unnamed:?}"
                );
                println!(
                    "batched leg cap={cap}: {measured} rows emitted, {} predicted ({} ops + {} \
                     constants + {} sponge rows, {} permutations)",
                    cost.rows(),
                    cost.leg.operations(),
                    cost.leg.constants(),
                    cost.schedule.rows(),
                    cost.perms()
                );
                assert_eq!(measured, cost.rows(), "cap {cap}");
            },
        );
    }
}

fn bump(value: &mut FEE) {
    *value += FEE::one();
}

/// ★ B-4a G3 — every tamper the host refuses, the leg refuses: the untouched
/// argue executes, the host rejects each forgery, and the machine cannot
/// execute it.
#[test]
fn the_batched_leg_refuses_what_the_host_rejects() {
    type Site = (&'static str, fn(&mut BatchedArgue<E>));
    let sites: Vec<Site> = vec![
        ("a bus output", |a| bump(&mut a.bus_outputs[1].0)),
        ("a mid-step half", |a| {
            bump(&mut a.gkr[0].layers[2].halves[1].q_lo)
        }),
        ("a ladder round", |a| {
            bump(&mut a.gkr[0].layers[3].sumcheck.rounds[1].evaluations[0])
        }),
        ("a constraint round", |a| {
            bump(&mut a.constraint.rounds[1].evaluations[1])
        }),
        ("a factor value", |a| bump(&mut a.factor_values[1][0])),
        ("a reduced column value", |a| {
            bump(&mut a.reduces[2].as_mut().expect("shifted").column_values[0])
        }),
    ];
    with_run(
        27,
        |_| {},
        |run| {
            run.host
                .clone()
                .expect("the honest argue verifies on the host");
            run.execute().expect("the honest argue executes");
        },
    );
    for (name, tamper) in sites {
        with_run(27, tamper, |run| {
            assert!(run.host.is_err(), "{name}: the HOST accepted it");
            assert!(
                run.execute().is_err(),
                "{name}: the machine executed a forgery the host rejects"
            );
        });
    }
}

/// The format constant is what plans: `ArgueFormat::BATCHED`'s cap puts these
/// three tables in one bin.
#[test]
fn the_production_cap_bins_the_fixture_once() {
    let ArgueFormat::Batched { bin_log_cells } = ArgueFormat::BATCHED else {
        unreachable!()
    };
    let fixture = Fixture::new();
    let layouts = fixture.layouts();
    assert_eq!(plan_of(&layouts, bin_log_cells).bins, vec![vec![0, 1, 2]]);
}

/// A printing comparison, not a gate: the batched leg's rows and permutations
/// against the per-table legs' (`table_verify_cost`, chained through one
/// sponge) over the same three tables — the in-guest side of D-BATCH §5.2.
#[test]
fn the_batched_leg_against_the_per_table_legs() {
    use super::whir_table::table_verify_cost;
    for cap in [27u8, 0] {
        with_run(
            cap,
            |_| {},
            |run| {
                let batched = batched_argue_cost(run.shapes, run.plan, SpongeEntry::fresh());
                let mut entry = SpongeEntry::fresh();
                let (mut rows, mut perms) = (0usize, 0usize);
                for shape in run.shapes {
                    let cost = table_verify_cost(shape, entry);
                    rows += cost.rows();
                    perms += cost.perms();
                    entry = cost.entry();
                }
                println!(
                    "batched leg cap={cap}: {} rows / {} perms; per-table legs {rows} rows / {perms} perms",
                    batched.rows(),
                    batched.perms()
                );
            },
        );
    }
}

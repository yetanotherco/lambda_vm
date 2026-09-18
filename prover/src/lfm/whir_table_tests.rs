//! Gates for the per-table verify's legs.
//!
//! The selector is gated against `Selector::evaluate` itself, at EVERY
//! exemption count rather than at a sample of them: the cost and the value both
//! turn on the bit pattern of `cutoff`, so a gate that picked a few counts
//! would be picking a few bit patterns.

use multilinear::selector::Selector;

use crate::tables::types::{FE, FEE};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_bus::Cost;
use super::whir_table::{emit_selector, selector_cost};
use super::word::{LfmWord, ext_word, word_as_ext};

/// One `Public` instruction per published value (`builder.rs:637-642`), which
/// the marginal below has to take out to be about the leg.
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

fn words(point: &[FEE]) -> Vec<Vec<LfmWord>> {
    vec![point.iter().map(ext_word).collect()]
}

/// Every exemption count a table of `2^num_vars` rows can carry, the trivial
/// and the all-exempt included.
fn every_selector(num_vars: usize) -> Vec<Selector> {
    (0..=(1usize << num_vars))
        .map(Selector::except_last)
        .collect()
}

/// The first `count` selectors emitted over one hinted point, each published.
fn selectors_program(num_vars: usize, count: usize) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(num_vars.max(1) as u32);
    let point: Vec<_> = (0..num_vars)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    // A program with no variables still needs something in its arena to hint,
    // and something to publish.
    let anchor = if num_vars == 0 {
        b.hint_word(arena, 0).as_ext()
    } else {
        point[0]
    };
    b.public(anchor.as_cell());
    for selector in every_selector(num_vars).into_iter().take(count) {
        let value = emit_selector(&mut b, selector, &point);
        b.public(value.as_cell());
    }
    compile(b.finish())
}

/// ★ The selector leg computes what the host computes, at every cutoff.
#[test]
fn the_selector_is_the_hosts_indicator() {
    for num_vars in 1..=5usize {
        let selectors = every_selector(num_vars);
        let program = selectors_program(num_vars, selectors.len());
        validate(&program).expect("the selector leg must be admissible");
        for seed in [0x5E1EC7u64, 0xB17, 0xC0FFEE] {
            let point = pseudo(seed, num_vars);
            let exec = execute(&program, &words(&point), &crate::hash_pin::BLOCK_HASHER)
                .unwrap_or_else(|e| panic!("num_vars {num_vars}: must execute: {e:?}"));
            let got: Vec<FEE> = exec.public_words[1..]
                .iter()
                .map(|(_, word)| word_as_ext(word).expect("a published extension value"))
                .collect();
            assert_eq!(got.len(), selectors.len());
            for (selector, value) in selectors.iter().zip(&got) {
                let want = selector
                    .evaluate::<crate::tables::types::GoldilocksExtension>(&point)
                    .expect("the host evaluates it");
                assert_eq!(
                    *value, want,
                    "num_vars {num_vars}, end_exemptions {} at seed {seed:#x}",
                    selector.end_exemptions
                );
            }
        }
    }
}

/// The two cases the general form does not reach by arithmetic: a selector that
/// applies everywhere is the literal one, and one that applies nowhere is the
/// literal zero.
///
/// Named separately because they are the branches that RETURN before the loop,
/// and a gate that only swept the middle would leave both unexercised at the
/// value level.
#[test]
fn the_trivial_and_the_all_exempt_selectors_are_literals() {
    let num_vars = 3;
    let point = pseudo(0xAB1E, num_vars);
    for (selector, want) in [
        (Selector::ALL, FEE::one()),
        (Selector::except_last(1usize << num_vars), FEE::zero()),
    ] {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let arena = b.declare_arena(num_vars as u32);
        let wires: Vec<_> = (0..num_vars)
            .map(|i| b.hint_word(arena, i as u32).as_ext())
            .collect();
        let value = emit_selector(&mut b, selector, &wires);
        b.public(value.as_cell());
        let program = compile(b.finish());
        let exec = execute(&program, &words(&point), &crate::hash_pin::BLOCK_HASHER)
            .expect("the literal cases must execute");
        let got = word_as_ext(&exec.public_words[0].1).expect("a published extension value");
        assert_eq!(got, want, "end_exemptions {}", selector.end_exemptions);
        assert_eq!(
            got,
            selector
                .evaluate::<crate::tables::types::GoldilocksExtension>(&point)
                .expect("the host evaluates it"),
            "and it must be the host's answer, not just the literal I expected"
        );
    }
}

/// ★ F1, incrementally: each selector's own marginal rows against its form.
///
/// Incremental rather than total, because the `1` is shared across the whole
/// set — a total would let one selector's over-count cancel another's, and the
/// first selector is the only one that pays for the constant.
#[test]
fn each_selector_emits_its_closed_form() {
    for num_vars in 1..=4usize {
        let selectors = every_selector(num_vars);
        let mut previous = selectors_program(num_vars, 0).instrs.len();
        let mut pool = Cost::default();
        for (index, selector) in selectors.iter().enumerate() {
            let before = pool.rows();
            selector_cost(*selector, num_vars, &mut pool);
            let predicted = pool.rows() - before;

            let now = selectors_program(num_vars, index + 1).instrs.len();
            let measured = now - previous - PUBLISH_ROW;
            previous = now;
            assert_eq!(
                measured, predicted,
                "num_vars {num_vars}, end_exemptions {}",
                selector.end_exemptions
            );
        }
        println!(
            "selectors num_vars={num_vars}: {} selectors, {} rows pooled ({} ops + {} constants)",
            selectors.len(),
            pool.rows(),
            pool.operations(),
            pool.constants()
        );
    }
}

/// The pooled total is what the incremental steps add up to, and the whole set
/// costs it once.
///
/// This is the statement the incremental test cannot make: that the ACCUMULATOR
/// shares its constants rather than charging each selector its own `1`.
#[test]
fn the_selectors_share_one_interned_constant() {
    let num_vars = 4;
    let selectors = every_selector(num_vars);
    let mut pool = Cost::default();
    for selector in &selectors {
        selector_cost(*selector, num_vars, &mut pool);
    }
    let mut separately = 0usize;
    for selector in &selectors {
        let mut alone = Cost::default();
        selector_cost(*selector, num_vars, &mut alone);
        separately += alone.rows();
    }
    // Every selector interns the `1`; the all-exempt one also interns a `0`.
    // Pooled, those are two rows for the whole set.
    let shared = separately - pool.rows();
    assert_eq!(
        shared,
        selectors.len() - 1,
        "the `1` is paid once and not {} times",
        selectors.len()
    );

    let measured = selectors_program(num_vars, selectors.len()).instrs.len()
        - selectors_program(num_vars, 0).instrs.len()
        - selectors.len() * PUBLISH_ROW;
    assert_eq!(measured, pool.rows(), "the pooled form is the emitted set");
}

// ---------------------------------------------------------------------------
// The assembly, against a REAL per-table proof.
// ---------------------------------------------------------------------------

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::transcript_hash::RpxTranscriptHash;
use multilinear::claim_reduce::FactorSource;
use multilinear::constraint_argument::FactorKind;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet, RowDomain};
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::multilinear_air::{IrShape, Uniforms};
use stark::multilinear_logup::{InteractionShape, interaction_shapes};
use stark::multilinear_table::{CommittedTable, TableProof, weight_slots};
use stark::traits::AIR;

use super::whir_bus::alpha_powers_read;
use super::whir_gkr::GkrLayerWires;
use super::whir_reduce::ReduceWires;
use super::whir_table::{
    TableCost, TableProofWires, TableShape, emit_table_verify, table_verify_cost,
};
use super::whir_transcript::{SpongeEntry, WhirTranscript};

type F = crate::tables::types::GoldilocksField;
type E = crate::tables::types::GoldilocksExtension;
type HostTranscript = DefaultTranscript<E, RpxTranscriptHash>;

/// The table under gate: three constraint roots, one of them carrying a
/// selector, over a trace that SATISFIES them.
///
/// ⚠ A test constraint set, said so. What makes it worth gating against is that
/// it goes through the same `AirWithBuses` + `TableLayout` pipeline a VM table
/// does, so its `IrShape`, its factor kinds and its public selector are built
/// by the production compiler — and unlike 2c's shape this one is SATISFIABLE,
/// which is what lets a real `multilinear_table::prove` run over it.
///
/// - `c = a·b` and `d = a + b` apply on every row;
/// - `next(a) = a + 1` cannot apply on the last, so it carries
///   `except_last(1)` — and that selector is what puts a PUBLIC FACTOR in the
///   woven factor vector, which is how the assembly's gate reaches the selector
///   leg as well as its own.
struct ThreeRootsOneSelector;

const COLUMNS: usize = 4;

impl ConstraintSet<F, E> for ThreeRootsOneSelector {
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

        let one = b.one();
        let next = b.main(1, 0);
        let here = b.main(0, 0);
        b.emit_base_rows(2, RowDomain::except_last(1), next - here - one);
    }
}

/// Two interactions over the same four columns: a real bus, and one the layout
/// will not refuse.
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

/// The trace the constraints hold on: `a` counts, `b` is fixed, `c = a·b`,
/// `d = a + b`.
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

fn air() -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), ThreeRootsOneSelector> {
    AirWithBuses::new(
        COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: buses(),
        },
        &table_options(),
        1,
        ThreeRootsOneSelector,
    )
}

fn table_options() -> stark::proof::options::ProofOptions {
    stark::proof::options::GoldilocksCubicProofOptions::with_params(4, 128, 20)
        .expect("valid options")
}

/// The LogUp challenges and the zerocheck's batching challenge, as the epoch
/// would have drawn them.
fn table_challenges() -> (FEE, FEE, FEE) {
    let drawn = pseudo(0x7AB1E, 3);
    (drawn[0], drawn[1], drawn[2])
}

/// A real per-table proof. `multilinear_table::prove` reads no commitment root
/// (`multilinear_table.rs:630-700`), so this needs no WHIR commitment and no
/// guest ELF.
fn real_proof(
    air: &AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), ThreeRootsOneSelector>,
    num_vars: usize,
) -> (TableProof<E>, Vec<FE>, Vec<Vec<FE>>) {
    let cols = columns(num_vars);
    let lifted = cols.clone();
    let table = CommittedTable::<F, E>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        COLUMNS,
        num_vars,
        Uniforms::default(),
        move |col| lifted[col as usize].clone(),
    )
    .expect("the table lays out");

    let (z, alpha, beta) = table_challenges();
    let mut prover = HostTranscript::new(&[]);
    let (proof, _point) = stark::multilinear_table::prove(&table, &z, &alpha, &beta, &mut prover)
        .expect("the table proves");
    (proof, Vec::new(), cols)
}

/// The bus, with the challenges factored out, and the layout it came from.
fn shape_of(
    air: &AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), ThreeRootsOneSelector>,
    num_vars: usize,
) -> (
    stark::multilinear_table::TableLayout<'_, F, E>,
    Vec<InteractionShape<E>>,
) {
    let layout = stark::multilinear_table::TableLayout::<F, E>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        COLUMNS,
        num_vars,
        Uniforms::default(),
    )
    .expect("the table lays out");
    let slots = layout.slot_of().to_vec();
    let bus = interaction_shapes(air.bus_interactions(), COLUMNS, |column| {
        slots
            .get(column)
            .copied()
            .ok_or(multilinear::Error::UnknownPolynomial {
                index: column,
                len: slots.len(),
            })
    })
    .expect("the bus probes");
    (layout, bus)
}

/// The alpha ladder the emitter reads.
fn alpha_ladder(bus: &[InteractionShape<E>], alpha: FEE) -> Vec<FEE> {
    let mut powers = Vec::with_capacity(alpha_powers_read(bus));
    let mut power = FEE::one();
    for _ in 0..alpha_powers_read(bus) {
        powers.push(power);
        power *= alpha;
    }
    powers
}

/// Every value the program hints, in arena order.
fn flatten(proof: &TableProof<E>, z: FEE, alpha_powers: &[FEE], beta: FEE) -> Vec<FEE> {
    let mut values = vec![z];
    values.extend_from_slice(alpha_powers);
    values.push(beta);
    values.push(proof.bus_output.0);
    values.push(proof.bus_output.1);
    for layer in &proof.gkr.layers {
        for round in &layer.sumcheck.rounds {
            values.extend(round.evaluations.iter().copied());
        }
        values.extend([layer.p_lo, layer.p_hi, layer.q_lo, layer.q_hi]);
    }
    for round in &proof.constraint.sumcheck.rounds {
        values.extend(round.evaluations.iter().copied());
    }
    values.extend(proof.constraint.factor_values.iter().copied());
    for round in &proof.constraint.reduce.sumcheck.rounds {
        values.extend(round.evaluations.iter().copied());
    }
    values.extend(proof.constraint.reduce.column_values.iter().copied());
    values
}

/// The assembled program, or — with `leg` false — the same hinted arena with no
/// leg at all, so the difference is the leg.
fn table_program(
    proof: &TableProof<E>,
    shape: &TableShape<'_>,
    alpha_count: usize,
    total: usize,
    leg: bool,
) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(total as u32);
    let mut index = 0u32;
    let mut take = |b: &mut LfmBuilder, count: usize| -> Vec<super::builder::Ext> {
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
    let output = take(&mut b, 2);
    let mut gkr = Vec::with_capacity(proof.gkr.layers.len());
    for layer in &proof.gkr.layers {
        let sumcheck: Vec<Vec<_>> = layer
            .sumcheck
            .rounds
            .iter()
            .map(|round| take(&mut b, round.evaluations.len()))
            .collect();
        let halves = take(&mut b, 4);
        gkr.push(GkrLayerWires {
            sumcheck,
            p_lo: halves[0],
            p_hi: halves[1],
            q_lo: halves[2],
            q_hi: halves[3],
        });
    }
    let sumcheck: Vec<Vec<_>> = proof
        .constraint
        .sumcheck
        .rounds
        .iter()
        .map(|round| take(&mut b, round.evaluations.len()))
        .collect();
    let factor_values = take(&mut b, proof.constraint.factor_values.len());
    let reduce_sumcheck: Vec<Vec<_>> = proof
        .constraint
        .reduce
        .sumcheck
        .rounds
        .iter()
        .map(|round| take(&mut b, round.evaluations.len()))
        .collect();
    let column_values = take(&mut b, proof.constraint.reduce.column_values.len());

    if !leg {
        b.public(z.as_cell());
        b.public(beta.as_cell());
        return compile(b.finish());
    }

    let mut transcript = WhirTranscript::new();
    let verdict = emit_table_verify(
        &mut b,
        &mut transcript,
        &TableProofWires {
            bus_output: (output[0], output[1]),
            gkr: &gkr,
            sumcheck: &sumcheck,
            factor_values: &factor_values,
            reduce: ReduceWires {
                sumcheck: &reduce_sumcheck,
                column_values: &column_values,
            },
        },
        shape,
        z,
        &alpha_powers,
        beta,
    );
    b.public(verdict.bus_output.0.as_cell());
    b.public(verdict.bus_output.1.as_cell());
    for wire in &verdict.point {
        b.public(wire.as_cell());
    }
    for wire in &verdict.column_values {
        b.public(wire.as_cell());
    }
    compile(b.finish())
}

/// ★ G1 — the assembled per-table verify computes what the host computes.
///
/// The machine derives every challenge itself, so executing an honest proof IS
/// the challenge-stream comparison: a wrong challenge anywhere leaves one of
/// three refusals the machine cannot satisfy. The verdict compared against the
/// host's is the value gate on top of that.
#[test]
fn the_table_verify_computes_what_the_host_computes() {
    for num_vars in [3usize, 4] {
        let built = air();
        let (proof, _, _) = real_proof(&built, num_vars);
        let (layout, bus) = shape_of(&built, num_vars);
        let (z, alpha, beta) = table_challenges();

        let mut verifier = HostTranscript::new(&[]);
        let (bus_output, reduced) = stark::multilinear_table::verify(
            &proof,
            layout.statement(),
            &z,
            &alpha,
            &beta,
            &mut verifier,
        )
        .expect("the host must verify its own proof — the fixture is the precondition");

        let shape = TableShape {
            ir: layout.shape(),
            bus: &bus,
            kinds: layout.kinds(),
            num_columns: layout.num_columns(),
            num_vars,
        };
        let alpha_powers = alpha_ladder(&bus, alpha);
        let values = flatten(&proof, z, &alpha_powers, beta);
        let program = table_program(&proof, &shape, alpha_powers.len(), values.len(), true);
        validate(&program).expect("the table leg must be admissible");

        let exec = execute(&program, &words(&values), &crate::hash_pin::BLOCK_HASHER)
            .unwrap_or_else(|e| panic!("num_vars {num_vars}: the table leg must execute: {e:?}"));
        let got: Vec<FEE> = exec
            .public_words
            .iter()
            .map(|(_, word)| word_as_ext(word).expect("a published extension value"))
            .collect();

        assert_eq!(got[0], bus_output.0, "num_vars {num_vars}: bus output p");
        assert_eq!(got[1], bus_output.1, "num_vars {num_vars}: bus output q");
        assert_eq!(
            &got[2..2 + num_vars],
            reduced.point.as_slice(),
            "num_vars {num_vars}: the reduced point"
        );
        assert_eq!(
            &got[2 + num_vars..],
            reduced.column_values.as_slice(),
            "num_vars {num_vars}: the column values"
        );
        println!(
            "table verify num_vars={num_vars}: {} GKR layers, degree {}, {} factors, \
             {} columns, {} public selectors",
            proof.gkr.layers.len(),
            shape.sumcheck_degree(),
            layout.kinds().len(),
            layout.num_columns(),
            layout.shape().public_selectors().len()
        );
    }
}

/// ★ G2 — F1 for the whole table: its marginal rows and its permutations.
#[test]
fn the_table_verify_emits_its_closed_form() {
    for num_vars in [3usize, 4] {
        let built = air();
        let (proof, _, _) = real_proof(&built, num_vars);
        let (layout, bus) = shape_of(&built, num_vars);
        let (z, alpha, beta) = table_challenges();
        let shape = TableShape {
            ir: layout.shape(),
            bus: &bus,
            kinds: layout.kinds(),
            num_columns: layout.num_columns(),
            num_vars,
        };
        let alpha_powers = alpha_ladder(&bus, alpha);
        let values = flatten(&proof, z, &alpha_powers, beta);
        let with = table_program(&proof, &shape, alpha_powers.len(), values.len(), true);
        let without = table_program(&proof, &shape, alpha_powers.len(), values.len(), false);
        // The leg publishes the whole verdict; the bare program publishes two.
        let published = 2 + num_vars + layout.num_columns();
        let measured = with.instrs.len() - without.instrs.len() - (published - 2) * PUBLISH_ROW;

        let cost: TableCost = table_verify_cost(&shape, SpongeEntry::fresh());
        let consts = |p: &LfmProgram| {
            p.instrs
                .iter()
                .filter(|i| matches!(i, super::instr::Instr::Const { .. }))
                .count()
        };
        let measured_consts = consts(&with) - consts(&without);
        {
            // Diagnostic: which VALUES the program interns that the form does
            // not name. A residual here is a leg whose own constants are not in
            // the pool, and naming it is the point.
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
                "the form must NAME every constant the program interns, and it does not name {:?}",
                unnamed
            );
        }
        println!(
            "table num_vars={num_vars}: {measured} rows emitted ({} of them LFM_CONST), \
             {} predicted ({} leg ops + {} constants + {} sponge rows, {} permutations)",
            measured_consts,
            cost.rows(),
            cost.leg.operations(),
            cost.leg.constants(),
            cost.schedule.rows(),
            cost.perms()
        );
        assert_eq!(measured, cost.rows(), "num_vars {num_vars}");
    }
}

/// ★ G3 — the tamper arm, in THREE halves (instance 49).
///
/// Every site runs all three: the untouched proof EXECUTES, the HOST rejects
/// the forgery, and the machine REFUSES it. The middle half is not decoration —
/// without it a machine that refused something the host accepts would score as
/// a soundness success when it is a completeness bug.
///
/// The three sites are chosen to land on three different refusals: a GKR half
/// breaks the layer relation, a main-sumcheck evaluation breaks the batch's
/// residual, and a column value breaks `claim_reduce`'s shifted read. Each is
/// reached only if the machine drew the same challenges the host did, which is
/// the other thing this arm witnesses.
#[test]
fn the_tamper_arm_refuses_what_the_host_rejects() {
    let num_vars = 3usize;
    let built = air();
    let (honest, _, _) = real_proof(&built, num_vars);
    let (layout, bus) = shape_of(&built, num_vars);
    let (z, alpha, beta) = table_challenges();
    let shape = TableShape {
        ir: layout.shape(),
        bus: &bus,
        kinds: layout.kinds(),
        num_columns: layout.num_columns(),
        num_vars,
    };
    let alpha_powers = alpha_ladder(&bus, alpha);

    let run = |proof: &TableProof<E>| -> Result<(), String> {
        let values = flatten(proof, z, &alpha_powers, beta);
        let program = table_program(proof, &shape, alpha_powers.len(), values.len(), true);
        execute(&program, &words(&values), &crate::hash_pin::BLOCK_HASHER)
            .map(|_| ())
            .map_err(|e| format!("{e:?}"))
    };
    let host = |proof: &TableProof<E>| -> Result<(), String> {
        let mut verifier = HostTranscript::new(&[]);
        stark::multilinear_table::verify(
            proof,
            layout.statement(),
            &z,
            &alpha,
            &beta,
            &mut verifier,
        )
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
    };

    // HALF ONE: the untouched proof executes, on both sides.
    host(&honest).expect("the honest proof must verify on the host");
    run(&honest).expect("the honest proof must EXECUTE on the machine");

    let bump = |value: &mut FEE| *value += FEE::one();
    let sites: Vec<(&str, Box<dyn Fn(&mut TableProof<E>)>)> = vec![
        (
            "a GKR layer's q_lo",
            Box::new(|p: &mut TableProof<E>| bump(&mut p.gkr.layers[0].q_lo)),
        ),
        (
            "the main sumcheck's first evaluation",
            Box::new(|p: &mut TableProof<E>| {
                bump(&mut p.constraint.sumcheck.rounds[0].evaluations[0])
            }),
        ),
        (
            "a claimed column value",
            Box::new(|p: &mut TableProof<E>| bump(&mut p.constraint.reduce.column_values[0])),
        ),
    ];

    for (name, tamper) in sites {
        let mut forged = honest.clone();
        tamper(&mut forged);
        // HALF TWO: the host must reject it, or it is not a forgery and the
        // machine's refusal would say nothing.
        let rejected = host(&forged);
        assert!(
            rejected.is_err(),
            "{name}: the HOST accepted the tampered proof, so this site is not a forgery"
        );
        // HALF THREE: the machine refuses it.
        let refused = run(&forged);
        assert!(
            refused.is_err(),
            "{name}: the machine EXECUTED a proof the host rejected with {:?}",
            rejected.unwrap_err()
        );
        println!(
            "tamper {name}: host {} / machine {}",
            rejected.unwrap_err(),
            refused.unwrap_err()
        );
    }
}

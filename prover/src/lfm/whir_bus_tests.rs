//! Gates for the bus statements.
//!
//! Two bus sets, for the reason 2c needed two shapes:
//!
//! - the PRODUCTION one is `l2g_memory_air`'s own interactions through the real
//!   `TableLayout::new`, so the slot map, the multiplicities and the packings
//!   are the ones a real table hands `interactions`;
//! - the STRESS one carries what that bus has not: a `QuadHL` value worth FOUR
//!   bus elements with `2^16` coefficients, a signed multi-term multiplicity
//!   with a constant, an element that is a bare constant, and an interaction
//!   count that is not a power of two — which is the only way the padding slots
//!   exist at all. Said here rather than left to read as coverage.
//!
//! Unlike 2c's, the stress set needs no AIR: `interactions` and
//! `claim_statements` take `&[BusInteraction]`, which is exactly what the
//! production path hands them (`multilinear_table.rs:737-743`), so a hand-built
//! vector is the production input type and not a stand-in for one.

use multilinear::Error as MlError;
use multilinear::eq::eq_evals;
use multilinear::logup::{self, Interaction, input_layer_vars};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::multilinear_air::Uniforms;
use stark::multilinear_logup::{InteractionShape, interaction_shapes, interactions};
use stark::multilinear_table::TableLayout;
use stark::traits::AIR;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_bus::{
    Cost, alpha_powers_read, claim_statements_cost, emit_claim_statements, emit_eq_evals,
    emit_interaction, eq_evals_cost, interaction_cost,
};
use super::word::{ext_word, word_as_ext};

type F = GoldilocksField;
type E = GoldilocksExtension;
type Shape = InteractionShape<E>;

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

fn options() -> stark::proof::options::ProofOptions {
    stark::proof::options::GoldilocksCubicProofOptions::with_params(4, 128, 20)
        .expect("valid options")
}

/// A bus set with everything both sides of the gate need to read it.
struct BusCase {
    name: &'static str,
    buses: Vec<BusInteraction>,
    /// The main width `interactions` filters its probe candidates against.
    width: usize,
    /// Main column -> the factor that reads it.
    slots: Vec<usize>,
    num_row_vars: usize,
}

impl BusCase {
    fn slot_of(&self) -> impl FnMut(usize) -> Result<usize, MlError> + '_ {
        |column| {
            self.slots
                .get(column)
                .copied()
                .ok_or(MlError::UnknownPolynomial {
                    index: column,
                    len: self.slots.len(),
                })
        }
    }

    fn shapes(&self) -> Vec<Shape> {
        interaction_shapes(&self.buses, self.width, self.slot_of()).expect("the bus probes")
    }

    fn fused(&self, z: &FEE, alpha: &FEE) -> Vec<Interaction<E>> {
        interactions(&self.buses, self.width, z, alpha, self.slot_of()).expect("the bus probes")
    }

    /// One factor value per slot the bus can read, plus the row weight the
    /// statements multiply by, which sits in the LAST slot here.
    fn num_values(&self) -> usize {
        self.slots.iter().copied().max().unwrap_or(0) + 2
    }

    fn weight(&self) -> usize {
        self.num_values() - 1
    }
}

/// The production bus: `l2g_memory_air`'s interactions, laid out by the same
/// `TableLayout::new` both sides of a real proof call.
fn production_case() -> BusCase {
    let opts = options();
    let air =
        crate::continuation::l2g_memory_air(&opts, crate::tables::local_to_global::epoch_label(1));
    let layout = TableLayout::<F, E>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        crate::tables::local_to_global::cols::NUM_COLUMNS,
        4,
        Uniforms::default(),
    )
    .expect("the table lays out");
    BusCase {
        name: "l2g_memory (production bus, production slot map)",
        buses: air.bus_interactions().to_vec(),
        width: crate::tables::local_to_global::cols::NUM_COLUMNS,
        slots: layout.slot_of().to_vec(),
        num_row_vars: 4,
    }
}

/// The terms the production bus does not have.
fn stress_case() -> BusCase {
    let buses = vec![
        // Four bus elements out of eight columns, with the packing's own 2^16
        // coefficients — and a multiplicity that is a bare one, so its
        // numerator is an affine with NO terms at all.
        BusInteraction::sender(7u64, Multiplicity::One, Packing::QuadHL.columns(&[0])),
        // A receiver, so the numerator carries the sign; a multiplicity with
        // signed coefficients and a constant; an element that is a scaled
        // column plus a constant; and an element read with coefficient one,
        // which is the term that opens a fold for free.
        BusInteraction::receiver(
            7u64,
            Multiplicity::Linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: 8,
                },
                LinearTerm::Column {
                    coefficient: -3,
                    column: 9,
                },
                LinearTerm::Constant(5),
            ]),
            vec![
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 2,
                        column: 10,
                    },
                    LinearTerm::Constant(11),
                ]),
                BusValue::column(11),
            ],
        ),
        // A second bus id, and an element that is a constant and nothing else.
        BusInteraction::sender(9u64, Multiplicity::Column(12), vec![BusValue::constant(42)]),
    ];
    BusCase {
        name: "stress (QuadHL, signed multiplicity, 3 interactions)",
        buses,
        width: 13,
        slots: (0..13).collect(),
        num_row_vars: 3,
    }
}

fn cases() -> Vec<BusCase> {
    vec![production_case(), stress_case()]
}

/// The arena a bus program reads: `z`, the alpha ladder, the claim point, then
/// the factor values.
struct Inputs {
    z: FEE,
    alpha_powers: Vec<FEE>,
    claim_point: Vec<FEE>,
    values: Vec<FEE>,
}

impl Inputs {
    fn words(&self) -> Vec<Vec<super::word::LfmWord>> {
        vec![
            std::iter::once(&self.z)
                .chain(&self.alpha_powers)
                .chain(&self.claim_point)
                .chain(&self.values)
                .map(ext_word)
                .collect(),
        ]
    }

    fn len(&self) -> usize {
        1 + self.alpha_powers.len() + self.claim_point.len() + self.values.len()
    }
}

fn inputs(case: &BusCase, shapes: &[Shape], z: FEE, alpha: FEE, seed: u64) -> Inputs {
    let count = alpha_powers_read(shapes);
    let mut alpha_powers = Vec::with_capacity(count);
    let mut power = FEE::one();
    for _ in 0..count {
        alpha_powers.push(power);
        power = power * alpha;
    }
    Inputs {
        z,
        alpha_powers,
        claim_point: pseudo(seed, input_layer_vars(shapes.len(), case.num_row_vars)),
        values: pseudo(seed ^ 0x5A5A, case.num_values()),
        }
}

/// The same hinted arena for every program under gate, so "with the leg" and
/// "without it" differ by the leg alone.
fn hint_inputs(b: &mut LfmBuilder, shape: &Inputs) -> (super::builder::Ext, Vec<super::builder::Ext>, Vec<super::builder::Ext>, Vec<super::builder::Ext>) {
    let arena = b.declare_arena(shape.len() as u32);
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
    let z = take(b, 1)[0];
    let alpha_powers = take(b, shape.alpha_powers.len());
    let claim_point = take(b, shape.claim_point.len());
    let values = take(b, shape.values.len());
    (z, alpha_powers, claim_point, values)
}

/// The leg's whole program: both rule values published.
fn statements_program(case: &BusCase, shapes: &[Shape], shape: &Inputs) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let (z, alpha_powers, claim_point, values) = hint_inputs(&mut b, shape);
    let out = emit_claim_statements(
        &mut b,
        shapes,
        &claim_point,
        case.num_row_vars,
        z,
        &alpha_powers,
        &values,
        case.weight(),
    );
    b.public(out.numerator.as_cell());
    b.public(out.denominator.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the bus leg must be admissible");
    program
}

/// The same program without the leg: the marginal is the leg.
fn empty_program(shape: &Inputs) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let (z, _alpha, _point, values) = hint_inputs(&mut b, shape);
    b.public(z.as_cell());
    b.public(values[0].as_cell());
    compile(b.finish())
}

fn run(program: &LfmProgram, shape: &Inputs, name: &str) -> Vec<FEE> {
    let exec = execute(program, &shape.words(), &crate::hash_pin::BLOCK_HASHER)
        .unwrap_or_else(|e| panic!("{name}: the bus leg must execute: {e:?}"));
    exec.public_words
        .iter()
        .map(|(_, word)| word_as_ext(word).expect("a published extension value"))
        .collect()
}

/// Three challenge sets, because an identity that only holds at one `alpha` is
/// not one and a gate at a fixed `z` passes an emitter that baked it in.
fn challenges() -> Vec<(FEE, FEE)> {
    vec![
        (FEE::from(7u64), FEE::from(11u64)),
        (
            FEE::new([FE::from(3u64), FE::from(5u64), FE::from(9u64)]),
            FEE::new([FE::from(2u64), FE::from(0u64), FE::from(4u64)]),
        ),
        (pseudo(0xD00Du64, 1)[0], pseudo(0xFEEDu64, 1)[0]),
    ]
}

/// ★ G1 — the weights are the host's `eq_evals`, entry for entry.
///
/// Every interaction's weight and every padding slot's comes out of this one
/// table, so a table that is right only on its live prefix would still settle
/// the wrong padding.
#[test]
fn the_weights_are_the_hosts_eq_table() {
    for n in 0..=5usize {
        let point = pseudo(0x1234 + n as u64, n);
        let shape = Inputs {
            z: FEE::zero(),
            alpha_powers: Vec::new(),
            claim_point: point.clone(),
            values: Vec::new(),
        };

        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let (_z, _alpha, wires, _values) = hint_inputs(&mut b, &shape);
        let table = emit_eq_evals(&mut b, &wires);
        for entry in &table {
            b.public(entry.as_cell());
        }
        let program = compile(b.finish());
        validate(&program).expect("the weight table must be admissible");

        let got = run(&program, &shape, "eq_evals");
        let want = eq_evals(&point);
        assert_eq!(got.len(), 1usize << n, "the table's length at n={n}");
        assert_eq!(got, want, "the weight table at n={n}");
    }
}

/// ★ G1b — F1 for the weights: the closed form `1 + n + 2·(2^n − 1)`.
#[test]
fn the_weight_table_emits_its_closed_form() {
    for n in 0..=5usize {
        let point = pseudo(0x1234 + n as u64, n);
        let shape = Inputs {
            z: FEE::zero(),
            alpha_powers: Vec::new(),
            claim_point: point,
            values: Vec::new(),
        };
        let with = {
            let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
            let (z, _alpha, wires, _values) = hint_inputs(&mut b, &shape);
            let table = emit_eq_evals(&mut b, &wires);
            b.public(table[0].as_cell());
            b.public(z.as_cell());
            compile(b.finish())
        };
        let without = {
            let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
            let (z, _alpha, wires, _values) = hint_inputs(&mut b, &shape);
            let first = wires.first().copied().unwrap_or(z);
            b.public(first.as_cell());
            b.public(z.as_cell());
            compile(b.finish())
        };
        let measured = with.instrs.len() - without.instrs.len();

        let mut cost = Cost::default();
        eq_evals_cost(n, &mut cost);
        let closed = 1 + n + 2 * ((1usize << n) - 1);
        assert_eq!(
            cost.rows(),
            closed,
            "the accumulator and the closed form must agree at n={n}"
        );
        println!("eq_evals n={n}: {measured} rows emitted, {closed} predicted");
        assert_eq!(measured, closed, "at n={n}");
    }
}

/// ★ G2 — each interaction's two values are what `interactions` builds.
///
/// Per interaction and not per sum: a leg that got the batch right by
/// compensating two errors passes a total and fails this.
#[test]
fn every_interaction_is_what_the_host_probes() {
    for case in cases() {
        let shapes = case.shapes();
        for (index, (z, alpha)) in challenges().into_iter().enumerate() {
            let shape = inputs(&case, &shapes, z, alpha, 0xB0_5A11 + index as u64);
            let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
            let (z_wire, alpha_wires, _point, values) = hint_inputs(&mut b, &shape);
            for interaction in &shapes {
                let (num, den) = emit_interaction(&mut b, interaction, z_wire, &alpha_wires, &values);
                b.public(num.as_cell());
                b.public(den.as_cell());
            }
            let program = compile(b.finish());
            validate(&program).expect("the interaction leg must be admissible");

            let got = run(&program, &shape, case.name);
            let fused = case.fused(&z, &alpha);
            assert_eq!(fused.len(), shapes.len(), "{}", case.name);
            for (i, interaction) in fused.iter().enumerate() {
                assert_eq!(
                    got[2 * i],
                    interaction.numerator.evaluate(&shape.values),
                    "{} interaction {i} numerator at challenge set {index}",
                    case.name
                );
                assert_eq!(
                    got[2 * i + 1],
                    interaction.denominator.evaluate(&shape.values),
                    "{} interaction {i} denominator at challenge set {index}",
                    case.name
                );
            }
        }
    }
}

/// ★ G3 — the two rule VALUES are what `claim_statements` applies.
///
/// The host returns programs; `constraint_argument::verify_core` runs them
/// through `Rule::apply`, which is the quantity this leg has to land on.
#[test]
fn the_bus_statements_compute_what_the_host_computes() {
    for case in cases() {
        let shapes = case.shapes();
        for (index, (z, alpha)) in challenges().into_iter().enumerate() {
            let shape = inputs(&case, &shapes, z, alpha, 0xC0_FFEE + index as u64);
            let program = statements_program(&case, &shapes, &shape);
            let got = run(&program, &shape, case.name);

            let fused = case.fused(&z, &alpha);
            let statements = logup::claim_statements(
                &fused,
                &shape.claim_point,
                case.num_row_vars,
                case.weight(),
            )
            .expect("the host builds its rules");
            assert_eq!(
                got[0],
                statements.numerator.apply(&shape.values),
                "{} numerator at challenge set {index}",
                case.name
            );
            assert_eq!(
                got[1],
                statements.denominator.apply(&shape.values),
                "{} denominator at challenge set {index}",
                case.name
            );
            assert_eq!(
                statements.row_point.len(),
                case.num_row_vars,
                "the row half of the claim point"
            );
        }
        println!(
            "bus {}: {} interactions, {} row vars",
            case.name,
            case.shapes().len(),
            case.num_row_vars
        );
    }
}

/// ★ G4 — F1 for the whole leg.
#[test]
fn the_bus_statements_emit_their_closed_form() {
    for case in cases() {
        let shapes = case.shapes();
        let shape = inputs(&case, &shapes, FEE::from(13u64), FEE::from(17u64), 0xF00D);
        let measured =
            statements_program(&case, &shapes, &shape).instrs.len() - empty_program(&shape).instrs.len();
        let cost = claim_statements_cost(&shapes, case.num_row_vars);
        let n = input_layer_vars(shapes.len(), case.num_row_vars) - case.num_row_vars;
        println!(
            "bus {}: {measured} rows emitted, {} predicted ({} ops + {} constants; \
             I={} 2^n={} alpha powers read={})",
            case.name,
            cost.rows(),
            cost.operations(),
            cost.constants(),
            shapes.len(),
            1usize << n,
            alpha_powers_read(&shapes),
        );
        assert_eq!(measured, cost.rows(), "{}", case.name);
    }
}

/// The per-interaction form sums to the whole leg's, less the parts only the
/// batch has.
///
/// Not a restatement: `interaction_cost` is what item 2e's per-table form will
/// call, and this is the statement that calling it that way reaches the same
/// number the whole-leg gate above pinned.
#[test]
fn the_per_interaction_cost_is_the_legs_own() {
    for case in cases() {
        let shapes = case.shapes();
        let whole = claim_statements_cost(&shapes, case.num_row_vars);
        let mut parts = Cost::default();
        let n = input_layer_vars(shapes.len(), case.num_row_vars) - case.num_row_vars;
        eq_evals_cost(n, &mut parts);
        for shape in &shapes {
            interaction_cost(shape, &mut parts);
        }
        let padding = (1usize << n) - shapes.len();
        let batch = 2 * shapes.len() + 2 + if padding > 0 { padding } else { 0 };
        assert_eq!(
            whole.rows(),
            parts.rows() + batch,
            "{}: the batch's own rows are {batch}",
            case.name
        );
    }
}

/// `Affine::constant_term` is the `k` of the expression and not some other
/// field: it is what the host's own evaluator returns at the origin.
#[test]
fn the_affine_getter_is_the_hosts_constant() {
    for case in cases() {
        let shapes = case.shapes();
        let zeros = vec![FEE::zero(); case.num_values()];
        for (i, shape) in shapes.iter().enumerate() {
            assert_eq!(
                shape.numerator.constant_term(),
                &shape.numerator.evaluate(&zeros),
                "{} interaction {i}'s numerator",
                case.name
            );
            for (p, element) in shape.elements.iter().enumerate() {
                assert_eq!(
                    element.constant_term(),
                    &element.evaluate(&zeros),
                    "{} interaction {i}'s element {p}",
                    case.name
                );
            }
        }
    }
}

/// The stress bus exists to carry terms the production one does not; this is
/// the assertion that it actually does, so a later edit cannot quietly turn it
/// into a second copy of the first.
#[test]
fn the_stress_bus_carries_what_the_production_one_does_not() {
    let stress = stress_case();
    let shapes = stress.shapes();
    assert_eq!(shapes.len(), 3, "not a power of two, so padding slots exist");
    assert_eq!(shapes[0].elements.len(), 4, "QuadHL is four bus elements");
    assert!(
        shapes[0].numerator.terms().is_empty(),
        "a multiplicity of one is an affine with no terms"
    );
    assert!(
        shapes[1]
            .numerator
            .terms()
            .iter()
            .any(|(_, c)| *c != FEE::one() && *c != -FEE::one()),
        "the signed multiplicity must carry a coefficient that is not a unit"
    );
    assert_ne!(
        shapes[1].numerator.constant_term(),
        &FEE::zero(),
        "the multiplicity's constant must survive the probe"
    );
    assert!(
        shapes[2].elements[0].terms().is_empty(),
        "a constant bus value is an element with no terms"
    );
    assert!(
        shapes[0]
            .elements
            .iter()
            .any(|e| e.terms().iter().any(|(_, c)| *c != FEE::one())),
        "the packing's 2^16 coefficients must reach the affine"
    );
}

/// The whole leg, published once so its shape is on the record beside the
/// numbers: what the bus statements cost per interaction on a real bus.
#[test]
fn the_bus_leg_prints_its_shape() {
    for case in cases() {
        let shapes = case.shapes();
        let elements: usize = shapes.iter().map(|s| s.elements.len()).sum();
        let terms: usize = shapes
            .iter()
            .map(|s| {
                s.numerator.terms().len()
                    + s.elements
                        .iter()
                        .map(|e| e.terms().len())
                        .sum::<usize>()
            })
            .sum();
        let cost = claim_statements_cost(&shapes, case.num_row_vars);
        println!(
            "bus shape {}: I={} elements={elements} affine terms={terms} rows={} \
             (ops {} + constants {})",
            case.name,
            shapes.len(),
            cost.rows(),
            cost.operations(),
            cost.constants(),
        );
    }
}

/// ★ The row form recounted from the shapes, by a walk that is not
/// `affine_cost`'s.
///
/// The operations an interaction costs are: one per affine TERM, less the one a
/// leading coefficient of one saves; one per affine that carries a nonzero
/// constant beside at least one term; one per bus element folding its alpha
/// power in; and one for `z − fingerprint`. Counted here off the shapes
/// directly, so a drift in the emitter's accumulator has a second opinion —
/// and the saving the leg actually takes is named rather than implied.
#[test]
fn the_row_form_recounts_from_the_shapes() {
    for case in cases() {
        let shapes = case.shapes();
        let affines = |shape: &Shape| -> Vec<multilinear::logup::Affine<E>> {
            std::iter::once(shape.numerator.clone())
                .chain(shape.elements.iter().cloned())
                .collect()
        };

        let mut leading_units = 0usize;
        let mut recount = 0usize;
        for shape in &shapes {
            for affine in affines(shape) {
                recount += affine.terms().len();
                if affine.terms().first().is_some_and(|(_, c)| *c == FEE::one()) {
                    leading_units += 1;
                    recount -= 1;
                }
                if !affine.terms().is_empty() && *affine.constant_term() != FEE::zero() {
                    recount += 1;
                }
            }
            recount += shape.elements.len() + 1;
        }

        let mut charged = Cost::default();
        for shape in &shapes {
            interaction_cost(shape, &mut charged);
        }
        println!(
            "bus {}: {leading_units} affines open on a unit coefficient and cost no row; \
             {} operations charged, {recount} recounted",
            case.name,
            charged.operations()
        );
        assert_eq!(charged.operations(), recount, "{}", case.name);
        assert!(
            leading_units > 0,
            "{}: some affine must open on a unit coefficient, or the saving is untested",
            case.name
        );
    }
}

/// The alpha ladder's length is what the elements READ, and the host's is two
/// longer.
#[test]
fn the_alpha_ladder_is_as_long_as_the_widest_interaction() {
    for case in cases() {
        let shapes = case.shapes();
        let widest = shapes.iter().map(|s| s.elements.len()).max().unwrap();
        assert_eq!(alpha_powers_read(&shapes), widest + 1);
        // `num_bus_elements` COUNTS THE BUS ID, and `interactions` sizes its
        // ladder at `width + 1` — so the host holds one power past the last
        // index `fingerprint_at` ever reads.
        let host = case
            .buses
            .iter()
            .map(BusInteraction::num_bus_elements)
            .max()
            .unwrap()
            + 1;
        assert_eq!(host, widest + 2, "{}", case.name);
        assert_eq!(
            host - alpha_powers_read(&shapes),
            1,
            "{}: the ladder this leg emits is exactly one shorter",
            case.name
        );
    }
}

/// A gate that can fail on the thing it names: the bus values must not be
/// independent of the factor values.
///
/// Both rules multiply by the row weight and by the interaction weights, so a
/// leg that dropped the factors entirely would still produce something that
/// moves with the challenges. This moves a FACTOR and demands the answer
/// follows.
#[test]
fn the_statements_read_the_factor_values() {
    for case in cases() {
        let shapes = case.shapes();
        let (z, alpha) = challenges()[0];
        let base = inputs(&case, &shapes, z, alpha, 0xAB_CDEF);
        let program = statements_program(&case, &shapes, &base);
        let first = run(&program, &base, case.name);

        let mut moved = inputs(&case, &shapes, z, alpha, 0xAB_CDEF);
        // A slot some element actually reads — moving one nothing reads would
        // leave the answer alone for a correct leg too.
        let slot = shapes
            .iter()
            .flat_map(|s| &s.elements)
            .flat_map(|e| e.terms())
            .map(|(slot, _)| *slot)
            .next()
            .expect("the bus reads at least one factor");
        moved.values[slot] = moved.values[slot] + FEE::one();
        let second = run(&program, &moved, case.name);
        assert_ne!(
            first[1], second[1],
            "{}: moving factor {slot} must move the denominator",
            case.name
        );
    }
}

/// `BusValues` is two distinct wires, not one published twice.
#[test]
fn the_two_rules_are_different_values() {
    let case = stress_case();
    let shapes = case.shapes();
    let (z, alpha) = challenges()[0];
    let shape = inputs(&case, &shapes, z, alpha, 0x2468);
    let out = run(&statements_program(&case, &shapes, &shape), &shape, case.name);
    assert_ne!(out[0], out[1], "the numerator and denominator must differ");
}

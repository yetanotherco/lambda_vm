//! Gates for the zerocheck rule's value.
//!
//! The shape under test is a REAL one: `l2g_memory_air`'s, laid out by the same
//! `TableLayout::new` both sides of the argument call (`multilinear_prove.rs`'s
//! `layout_of`), so the DAG, its roots, its selectors and its factor kinds are
//! the production ones. Its `num_vars` is small because the SHAPE is what the
//! emitter reproduces and the shape does not vary with the row count — epoch-0
//! heights belong to the census and to the box, not to this gate.
//!
//! ⚠ MEASURED, and the reason a second shape is here: the l2g memory AIR
//! compiles to ONE root with NO selector, so on it alone the batching fold and
//! the selector multiply are terms the form states and the gate never reaches.
//! [`ThreeRootsOneSelector`] carries both, through the same compiler.

use stark::multilinear_air::{IrShape, Uniforms, beta_powers};
use stark::multilinear_table::TableLayout;
use stark::traits::AIR;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_air::{combine_rows, emit_combine};
use super::word::{ext_word, word_as_ext};

type F = GoldilocksField;
type E = GoldilocksExtension;
type Shape = IrShape<F, E>;

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

/// The production shape: the local-to-global memory AIR, laid out exactly as
/// both sides of a real proof lay it out.
///
/// Chosen because it is a real AIR that this laptop can build — most of the
/// VM's tables need the guest ELF, and the two global AIRs beside this one
/// carry `EmptyConstraints` and so have no roots to batch at all.
fn production_shape(num_vars: usize) -> Shape {
    let opts = options();
    let air =
        crate::continuation::l2g_memory_air(&opts, crate::tables::local_to_global::epoch_label(1));
    let layout = TableLayout::<F, E>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        crate::tables::local_to_global::cols::NUM_COLUMNS,
        num_vars,
        Uniforms::default(),
    )
    .expect("the table lays out");
    layout.shape().clone()
}

/// Three roots, one of them carrying a selector — the terms the production AIR
/// above does not have.
///
/// ⚠ A TEST constraint set, and said so: what makes it worth gating against is
/// that it goes through the SAME pipeline — `AirWithBuses::new` and
/// `TableLayout::new` — so its `IrShape` is built by the real compiler, with
/// real roots, a real selector and real factor slots. The production AIR says
/// the pipeline is the production one; this says the form is right about the
/// roots that AIR happens not to have. Neither alone would do.
///
/// `except_last(1)` is what gives a root a selector: a constraint that reads one
/// row ahead cannot apply on the last row (`builder.rs:177-195`).
struct ThreeRootsOneSelector;

impl stark::constraints::builder::ConstraintSet<F, E> for ThreeRootsOneSelector {
    fn max_degree(&self) -> usize {
        2
    }

    fn eval<B: stark::constraints::builder::ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let x = b.main(0, 0);
        let y = b.main(0, 1);
        let z = b.main(0, 2);
        b.emit_base(0, x * y - z);

        let x = b.main(0, 0);
        let y = b.main(0, 1);
        let z = b.main(0, 2);
        b.emit_base(1, x + y - z);

        let next = b.main(1, 0);
        let here = b.main(0, 0);
        b.emit_base_rows(
            2,
            stark::constraints::builder::RowDomain::except_last(1),
            next - here,
        );
    }
}

fn stress_shape(num_vars: usize) -> Shape {
    let opts = options();
    let air = stark::lookup::AirWithBuses::<
        F,
        E,
        stark::lookup::NullBoundaryConstraintBuilder,
        (),
        ThreeRootsOneSelector,
    >::new(
        crate::tables::local_to_global::cols::NUM_COLUMNS,
        stark::lookup::AuxiliaryTraceBuildData {
            // ⚠ The layout needs a bus — `TableLayout::new` refuses a table with
            // no interactions (`EmptyPolynomial`, measured). These are the
            // production AIR's own, so the column indices they read exist.
            interactions: crate::tables::local_to_global::memory_bus_interactions(),
        },
        &opts,
        1,
        ThreeRootsOneSelector,
    );
    let layout = TableLayout::<F, E>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        crate::tables::local_to_global::cols::NUM_COLUMNS,
        num_vars,
        Uniforms::default(),
    )
    .expect("the table lays out");
    layout.shape().clone()
}

/// The shapes under gate, with a name for the failure message.
///
/// ⚠ Both are the SAME AIR at two heights, because it is the only real AIR this
/// laptop can build with constraints on it: every VM table needs the guest ELF,
/// and the two global AIRs beside it carry `EmptyConstraints` and have no roots
/// to batch. Whatever roots and selectors this one does not have are terms the
/// form states and this gate does not exercise — said here rather than left to
/// read as full coverage.
fn shapes() -> Vec<(&'static str, Shape)> {
    vec![
        ("l2g_memory num_vars=4", production_shape(4)),
        ("l2g_memory num_vars=7", production_shape(7)),
        ("three roots, one selector, num_vars=4", stress_shape(4)),
        ("three roots, one selector, num_vars=6", stress_shape(6)),
    ]
}

/// Values for every factor the shape can read, plus the two weight slots the
/// woven factor vector carries (`multilinear_table.rs:616-618`).
fn factor_values(shape: &Shape, seed: u64) -> Vec<FEE> {
    let slots = shape
        .steps_as_ops()
        .iter()
        .filter_map(|op| match op {
            multilinear::program::Op::Var(i) => Some(*i as usize),
            _ => None,
        })
        .chain(shape.selector_of_root().iter().filter_map(|s| *s))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    pseudo(seed, slots.max(1) + 2)
}

/// `combine` over hinted betas and hinted factor values, the answer published.
fn combine_program(shape: &Shape, num_values: usize) -> LfmProgram {
    let roots = shape.num_roots();
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena((roots + num_values) as u32);
    let betas: Vec<_> = (0..roots)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let values: Vec<_> = (0..num_values)
        .map(|i| b.hint_word(arena, (roots + i) as u32).as_ext())
        .collect();
    let combined = emit_combine(&mut b, shape, &betas, &values);
    b.public(combined.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the combine leg must be admissible");
    program
}

/// The leg's marginal cost: the same program without it.
fn marginal_rows(shape: &Shape, num_values: usize) -> usize {
    let roots = shape.num_roots();
    let with = combine_program(shape, num_values);
    let without = {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let arena = b.declare_arena((roots + num_values) as u32);
        let first = b.hint_word(arena, 0).as_ext();
        for i in 1..roots + num_values {
            let _ = b.hint_word(arena, i as u32);
        }
        b.public(first.as_cell());
        compile(b.finish())
    };
    with.instrs.len() - without.instrs.len()
}

/// ★ The leg computes what the host computes, at RANDOM betas.
///
/// The betas are the point: the host takes them as an argument, and the whole
/// reason this leg exists rather than `emit_program` is that they are drawn
/// after the commitment and cannot be interned. A gate that fixed them at one
/// value would pass for an emitter that had baked them in.
#[test]
fn the_combine_leg_computes_what_the_host_computes() {
    for (name, shape) in shapes() {
        let values = factor_values(&shape, 0xC0FFEE);
        let program = combine_program(&shape, values.len());
        for seed in [0x11u64, 0x22, 0x33] {
            let betas_free = pseudo(seed, shape.num_roots());
            // Both a free vector and a real power ladder: `beta_powers` is what
            // the caller actually supplies, and its first element is ONE, which
            // is the case an emitter that dropped the first multiply would pass.
            let ladder = beta_powers(&pseudo(seed ^ 0xFF, 1)[0], shape.num_roots());
            for betas in [betas_free, ladder] {
                let arenas = vec![
                    betas
                        .iter()
                        .chain(values.iter())
                        .map(ext_word)
                        .collect::<Vec<_>>(),
                ];
                let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
                    .unwrap_or_else(|e| panic!("{name}: the combine leg must execute: {e:?}"));
                let got =
                    word_as_ext(&exec.public_words[0].1).expect("a published extension value");
                let want = shape.combine(&betas, &values);
                assert_eq!(got, want, "{name} at seed {seed:#x}");
            }
        }
        println!(
            "combine {name}: {} roots, {} with selectors, {} factors supplied",
            shape.num_roots(),
            shape
                .selector_of_root()
                .iter()
                .filter(|s| s.is_some())
                .count(),
            values.len()
        );
    }
}

/// ★ F1 for the leg: its rows against the closed form.
#[test]
fn the_combine_leg_emits_its_closed_form() {
    for (name, shape) in shapes() {
        let values = factor_values(&shape, 0xC0FFEE);
        let measured = marginal_rows(&shape, values.len());
        let predicted = combine_rows(&shape);
        println!(
            "combine {name}: {measured} rows emitted, {predicted} predicted \
             ({} DAG steps, {} roots)",
            shape.steps_as_ops().len(),
            shape.num_roots()
        );
        assert_eq!(measured, predicted, "{name}");
    }
}

/// The batching is a function of the BETAS and not a sum: scaling one beta
/// scales exactly that root's contribution.
///
/// A leg that summed the roots and multiplied once, or that reused one beta for
/// every root, agrees with the host wherever the betas happen to be equal. This
/// is the linearity that says each root got its own.
#[test]
fn each_root_carries_its_own_beta() {
    let shape = stress_shape(4);
    let roots = shape.num_roots();
    assert!(roots > 1, "this shape exists to batch more than one root");
    let values = factor_values(&shape, 0xBEEF);

    let base = vec![FEE::one(); roots];
    let combined = shape.combine(&base, &values);
    for i in 0..roots {
        let mut bumped = base.clone();
        bumped[i] = FEE::one() + FEE::one();
        // Doubling one beta adds exactly that root's own term.
        let delta = &shape.combine(&bumped, &values) - &combined;
        let mut only = vec![FEE::zero(); roots];
        only[i] = FEE::one();
        assert_eq!(
            delta,
            shape.combine(&only, &values),
            "root {i}'s contribution must be what its own beta scales"
        );
    }
}

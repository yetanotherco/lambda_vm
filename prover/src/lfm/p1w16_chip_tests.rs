//! The width-16 measurement chip: its two widths, its degree bound, what it
//! accepts and what it rejects.
//!
//! The permutation is pinned in `crypto::hash::poseidon1_w16` (vectors agreed
//! by four implementations); here the chip's rows are pinned to that host
//! reference, and every value column is shown load-bearing.

use math::field::element::FieldElement;
use stark::constraints::builder::{
    CaptureBuilder, ConstraintSet, ProverEvalFolder, RootKind, num_base_from_meta,
};
use stark::frame::Frame;
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use crypto::hash::poseidon1_w16 as p1;

use super::p1w16_chip::*;

type Gl = GoldilocksField;
type Gl3 = GoldilocksExtension;

const LAYOUTS: [Layout; 2] = [Layout::Rule, Layout::Compact];
const MODES: [Mode; 4] = [Mode::Node, Mode::Permute, Mode::Transcript, Mode::Leaf];

/// Pinned as literals, from the closed forms in the module header, so a layout
/// bug cannot agree with itself.
/// Rule: 36 + 7·48 + 32 + 22·18. Compact: 36 + 8·32 + 22·2.
const RULE_VALUE_COLUMNS: usize = 800;
const COMPACT_VALUE_COLUMNS: usize = 336;
/// Plus four LogUp aux columns at three cells each.
const RULE_CELLS: usize = 812;
const COMPACT_CELLS: usize = 348;
/// 5 + 8·32 + 22·2 + 30·16, and 5 + 8·32 + 22·2 + 16.
const RULE_CONSTRAINTS: usize = 785;
const COMPACT_CONSTRAINTS: usize = 321;

fn evaluate(layout: Layout, row: &[FE]) -> Vec<FE> {
    let set = P1W16Constraints { layout };
    let n = ConstraintSet::<Gl, Gl3>::meta(&set).len();
    let no_ch: Vec<FieldElement<Gl3>> = vec![];
    let offset = FieldElement::<Gl3>::zero();
    let frame = Frame::<Gl, Gl3>::new(vec![TableView::new(vec![row.to_vec()], vec![vec![]])]);
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_ch, &no_ch, &offset);
    let mut base_out = vec![FE::zero(); n];
    let mut ext_out = vec![FieldElement::<Gl3>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base_out, &mut ext_out);
    set.eval(&mut folder);
    folder.assert_all_emitted();
    base_out
}

fn violations(layout: Layout, row: &[FE]) -> Vec<usize> {
    evaluate(layout, row)
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != FE::zero())
        .map(|(i, _)| i)
        .collect()
}

fn sample_input() -> [FE; p1::STATE_FELTS] {
    core::array::from_fn(|i| FE::from(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1)))
}

#[test]
fn the_widths_are_the_closed_forms() {
    assert_eq!(Layout::Rule.value_columns(), RULE_VALUE_COLUMNS);
    assert_eq!(Layout::Compact.value_columns(), COMPACT_VALUE_COLUMNS);
    assert_eq!(RULE_VALUE_COLUMNS, 36 + 7 * 48 + 32 + 22 * 18);
    assert_eq!(COMPACT_VALUE_COLUMNS, 36 + 8 * 32 + 22 * 2);
    assert_eq!(Layout::Rule.cells_per_permutation(), RULE_CELLS);
    assert_eq!(Layout::Compact.cells_per_permutation(), COMPACT_CELLS);
    assert_eq!(Layout::Rule.num_constraints(), RULE_CONSTRAINTS);
    assert_eq!(Layout::Compact.num_constraints(), COMPACT_CONSTRAINTS);
}

/// Every value column is handed out exactly once (the Rule layout's last-round
/// output IS `OUT`, asserted as an alias).
#[test]
fn every_layout_assigns_every_column_exactly_once() {
    for layout in LAYOUTS {
        let mut seen = vec![0usize; layout.num_columns()];
        let prefix = [(IN0, p1::STATE_FELTS), (S12, 4), (OUT0, p1::STATE_FELTS)];
        for (start, len) in prefix {
            for n in &mut seen[start..start + len] {
                *n += 1;
            }
        }
        for r in 0..p1::NUM_ROUNDS {
            for lane in 0..sboxed_lanes(r) {
                seen[layout.w1(r, lane)] += 1;
                seen[layout.w2(r, lane)] += 1;
            }
            if layout == Layout::Rule {
                if r + 1 < p1::NUM_ROUNDS {
                    for j in 0..p1::STATE_FELTS {
                        seen[layout.out(r, j)] += 1;
                    }
                } else {
                    assert_eq!(layout.out(r, 0), OUT0);
                }
            }
        }
        for (c, &n) in seen.iter().enumerate() {
            let want = usize::from(c >= PREP_WIDTH);
            assert_eq!(n, want, "{layout:?}: column {c} claimed {n} times");
        }
    }
}

#[test]
fn every_constraint_is_degree_three_or_less_and_some_reach_three() {
    for layout in LAYOUTS {
        let set = P1W16Constraints { layout };
        let meta = ConstraintSet::<Gl, Gl3>::meta(&set);
        assert_eq!(meta.len(), layout.num_constraints(), "{layout:?}");
        for (i, m) in meta.iter().enumerate() {
            assert_eq!(m.constraint_idx, i, "{layout:?}: meta dense and ordered");
            assert_eq!(m.kind, RootKind::Base);
        }
        let mut cb = CaptureBuilder::<Gl, Gl3>::new();
        set.eval(&mut cb);
        let (_prog, degrees) = cb.finish(num_base_from_meta(&meta));
        assert_eq!(degrees.len(), meta.len());
        assert!(degrees.iter().all(|&(_, d)| d <= 3), "{layout:?}");
        assert_eq!(degrees.iter().map(|&(_, d)| d).max(), Some(3), "{layout:?}");
    }
}

/// An honest row satisfies every constraint in every mode, and its `OUT` is
/// the host reference's permutation of the state the mode builds.
#[test]
fn honest_rows_satisfy_every_constraint_and_match_the_host_reference() {
    for layout in LAYOUTS {
        for mode in MODES {
            let input = sample_input();
            let row = fill_row(layout, mode, input);
            assert_eq!(
                violations(layout, &row),
                Vec::<usize>::new(),
                "{layout:?} {mode:?}"
            );
            let mut state = input;
            if matches!(mode, Mode::Transcript | Mode::Leaf) {
                let domain = if mode == Mode::Leaf {
                    p1::DOMAIN_LEAF
                } else {
                    DOMAIN_TRANSCRIPT
                };
                for (k, v) in [0, domain, 0, 0].iter().enumerate() {
                    state[p1::RATE_FELTS + k] = FE::from(*v);
                }
            }
            assert_eq!(
                row[OUT0..OUT0 + p1::STATE_FELTS].to_vec(),
                p1::permute(state).to_vec(),
                "{layout:?} {mode:?}"
            );
        }
    }
}

/// The chip reproduces the first known-answer vector end to end.
#[test]
fn a_permutation_row_reproduces_the_first_known_answer() {
    const KAT0_OUT_LANE0: u64 = 9_350_316_517_402_464_675;
    for layout in LAYOUTS {
        let input: [FE; p1::STATE_FELTS] = core::array::from_fn(|i| FE::from(i as u64));
        let row = fill_row(layout, Mode::Permute, input);
        assert_eq!(row[OUT0], FE::from(KAT0_OUT_LANE0), "{layout:?}");
    }
}

#[test]
fn an_all_zero_padding_row_satisfies_every_constraint() {
    for layout in LAYOUTS {
        let row = vec![FE::zero(); layout.num_columns()];
        assert_eq!(violations(layout, &row), Vec::<usize>::new(), "{layout:?}");
    }
}

/// Mutation: perturbing ANY value column of an honest permutation row breaks
/// some constraint, so no column is free (a layout that forgot to constrain
/// one would pass every test above).
#[test]
fn every_value_column_is_load_bearing() {
    for layout in LAYOUTS {
        let row = fill_row(layout, Mode::Permute, sample_input());
        for c in PREP_WIDTH..layout.num_columns() {
            let mut bad = row.clone();
            bad[c] += FE::one();
            assert!(
                !violations(layout, &bad).is_empty(),
                "{layout:?}: column {c} is unconstrained"
            );
        }
    }
}

/// A row claiming the wrong mode is rejected: a permutation's capacity under
/// the leaf selector violates the capacity copy.
#[test]
fn a_mode_swap_is_rejected() {
    for layout in LAYOUTS {
        let mut row = fill_row(layout, Mode::Permute, sample_input());
        row[MODE_P] = FE::zero();
        row[MODE_L] = FE::one();
        assert!(
            violations(layout, &row).iter().any(|&i| i < 4),
            "{layout:?}"
        );
    }
}

/// The evaluation cost the census does not see: hash-consed constraint-program
/// nodes (and multiplications) per row, for both layouts against the RPX and
/// W12 Poseidon chips. Informational (`--nocapture`); asserts only that the
/// programs are non-empty.
#[test]
fn constraint_program_sizes() {
    use super::chips::hash::HashConstraints;
    use stark::constraint_ir::ir::Op;

    fn size<S: ConstraintSet<Gl, Gl3>>(set: &S) -> (usize, usize) {
        let meta = set.meta();
        let mut cb = CaptureBuilder::<Gl, Gl3>::new();
        set.eval(&mut cb);
        let (prog, _) = cb.finish(num_base_from_meta(&meta));
        let muls = prog
            .nodes
            .iter()
            .filter(|o| matches!(o, Op::Mul(..)))
            .count();
        (prog.len(), muls)
    }
    for (name, (nodes, muls), cols) in [
        (
            "p1w16 rule",
            size(&P1W16Constraints {
                layout: Layout::Rule,
            }),
            Layout::Rule.value_columns(),
        ),
        (
            "p1w16 compact",
            size(&P1W16Constraints {
                layout: Layout::Compact,
            }),
            Layout::Compact.value_columns(),
        ),
        ("rpx", size(&HashConstraints::RPX), 0),
        ("poseidon w12", size(&HashConstraints::POSEIDON), 0),
    ] {
        assert!(nodes > 0);
        println!("PROGRAM {name}: nodes {nodes} muls {muls} value-cols {cols}");
    }
}

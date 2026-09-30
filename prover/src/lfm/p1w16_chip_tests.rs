//! The width-16 measurement chip: its two widths, its degree bound, what it
//! accepts and what it rejects.
//!
//! The permutation is pinned in `crypto::hash::poseidon1_w16` (vectors agreed
//! by four implementations); here the chip's rows are pinned to that host
//! reference, and every value column is shown load-bearing — under both MDS
//! options (Plonky3's circulant and the Grain Cauchy alternative).

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
const MDSS: [Mds; 2] = [Mds::Circulant, Mds::Cauchy];
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

fn evaluate(layout: Layout, mds: Mds, row: &[FE]) -> Vec<FE> {
    let set = P1W16Constraints { layout, mds };
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

fn violations(layout: Layout, mds: Mds, row: &[FE]) -> Vec<usize> {
    evaluate(layout, mds, row)
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
    for (layout, mds) in LAYOUTS.into_iter().flat_map(|l| MDSS.map(|m| (l, m))) {
        let set = P1W16Constraints { layout, mds };
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
    for (layout, mds) in LAYOUTS.into_iter().flat_map(|l| MDSS.map(|m| (l, m))) {
        for mode in MODES {
            let input = sample_input();
            let row = fill_row(layout, mds, mode, input);
            assert_eq!(
                violations(layout, mds, &row),
                Vec::<usize>::new(),
                "{layout:?} {mds:?} {mode:?}"
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
            let want = match mds {
                Mds::Circulant => p1::permute(state),
                Mds::Cauchy => p1::permute_cauchy(state),
            };
            assert_eq!(
                row[OUT0..OUT0 + p1::STATE_FELTS].to_vec(),
                want.to_vec(),
                "{layout:?} {mds:?} {mode:?}"
            );
        }
    }
}

/// The chip reproduces the first known-answer vector end to end.
#[test]
fn a_permutation_row_reproduces_the_first_known_answer() {
    const KAT0_OUT_LANE0: u64 = 9_350_316_517_402_464_675;
    const CAUCHY_KAT0_OUT_LANE0: u64 = 7_675_469_683_446_723_213;
    for layout in LAYOUTS {
        let input: [FE; p1::STATE_FELTS] = core::array::from_fn(|i| FE::from(i as u64));
        let row = fill_row(layout, Mds::Circulant, Mode::Permute, input);
        assert_eq!(row[OUT0], FE::from(KAT0_OUT_LANE0), "{layout:?}");
        let row = fill_row(layout, Mds::Cauchy, Mode::Permute, input);
        assert_eq!(
            row[OUT0],
            FE::from(CAUCHY_KAT0_OUT_LANE0),
            "{layout:?} cauchy"
        );
    }
}

/// A row filled under one MDS does not satisfy the other's constraints — the
/// constants are load-bearing, so the two instances are not interchangeable.
#[test]
fn a_row_under_one_mds_fails_the_other() {
    for layout in LAYOUTS {
        let row = fill_row(layout, Mds::Circulant, Mode::Permute, sample_input());
        assert!(
            !violations(layout, Mds::Cauchy, &row).is_empty(),
            "{layout:?}"
        );
        let row = fill_row(layout, Mds::Cauchy, Mode::Permute, sample_input());
        assert!(
            !violations(layout, Mds::Circulant, &row).is_empty(),
            "{layout:?}"
        );
    }
}

#[test]
fn an_all_zero_padding_row_satisfies_every_constraint() {
    for (layout, mds) in LAYOUTS.into_iter().flat_map(|l| MDSS.map(|m| (l, m))) {
        let row = vec![FE::zero(); layout.num_columns()];
        assert_eq!(
            violations(layout, mds, &row),
            Vec::<usize>::new(),
            "{layout:?} {mds:?}"
        );
    }
}

/// Mutation: perturbing ANY value column of an honest permutation row breaks
/// some constraint, so no column is free (a layout that forgot to constrain
/// one would pass every test above).
#[test]
fn every_value_column_is_load_bearing() {
    for (layout, mds) in LAYOUTS.into_iter().flat_map(|l| MDSS.map(|m| (l, m))) {
        let row = fill_row(layout, mds, Mode::Permute, sample_input());
        for c in PREP_WIDTH..layout.num_columns() {
            let mut bad = row.clone();
            bad[c] += FE::one();
            assert!(
                !violations(layout, mds, &bad).is_empty(),
                "{layout:?} {mds:?}: column {c} is unconstrained"
            );
        }
    }
}

/// A row claiming the wrong mode is rejected: a permutation's capacity under
/// the leaf selector violates the capacity copy.
#[test]
fn a_mode_swap_is_rejected() {
    for (layout, mds) in LAYOUTS.into_iter().flat_map(|l| MDSS.map(|m| (l, m))) {
        let mut row = fill_row(layout, mds, Mode::Permute, sample_input());
        row[MODE_P] = FE::zero();
        row[MODE_L] = FE::one();
        assert!(
            violations(layout, mds, &row).iter().any(|&i| i < 4),
            "{layout:?} {mds:?}"
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
    let p1 = |layout, mds| size(&P1W16Constraints { layout, mds });
    for (name, (nodes, muls), cols) in [
        (
            "p1w16 rule",
            p1(Layout::Rule, Mds::Circulant),
            Layout::Rule.value_columns(),
        ),
        (
            "p1w16 compact",
            p1(Layout::Compact, Mds::Circulant),
            Layout::Compact.value_columns(),
        ),
        (
            "p1w16-cauchy rule",
            p1(Layout::Rule, Mds::Cauchy),
            Layout::Rule.value_columns(),
        ),
        (
            "p1w16-cauchy compact",
            p1(Layout::Compact, Mds::Cauchy),
            Layout::Compact.value_columns(),
        ),
        ("rpx", size(&HashConstraints::RPX), 0),
        ("poseidon w12", size(&HashConstraints::POSEIDON), 0),
    ] {
        assert!(nodes > 0);
        println!("PROGRAM {name}: nodes {nodes} muls {muls} value-cols {cols}");
    }
}

/// ★ D-HASH stage 1: does the compact chip's evaluation work bind on the card?
///
/// The recursion side prices a chip by its CELLS (the card law, 4.59 ns per
/// census cell, fitted on today's chips). The compact width-16 chip has 1.07×
/// RPX's cells per permutation but 1.93× its constraint-program nodes, and
/// evaluation work is paid per node, not per cell. This times the card's fused
/// composition evaluation (`try_eval_composition_gpu`, the production path at
/// this base) of each chip's program over `2^20` LDE rows of random columns,
/// and prints each chip's nanoseconds per permutation at blowup 2 (two LDE rows
/// per trace row, one permutation per row) with the part the cell law does NOT
/// already charge: `E = 2·(t_chip − t_RPX · cells_chip / cells_RPX)`, i.e. what
/// the chip costs beyond RPX's evaluation density, and the effective width
/// `cells + E / 4.59`. Only the chips' own constraints are timed (their LogUp
/// columns are 3 or 4 aux on both sides and are left out).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: times the chips' constraint evaluation on the GPU; run on the box with --ignored"]
fn chip_constraint_eval_cost_on_the_card() {
    use std::sync::Arc;
    use std::time::Instant;

    use math_cuda::lde::{GpuLdeBase, GpuLdeExt3};
    use stark::constraint_ir::gpu_interp::{CompositionInputs, try_eval_composition_gpu};

    use super::chips::hash::{HashConstraints, num_columns};
    use super::hash::HasherKind;

    const LOG_ROWS: u32 = 20;
    const BLOWUP: usize = 2;
    const REPS: usize = 5;
    const NS_PER_CELL: f64 = 4.59;
    let n = 1usize << LOG_ROWS;

    // Nanoseconds per LDE row for one constraint set over `width` columns.
    fn time_set<S: ConstraintSet<Gl, Gl3>>(set: &S, width: usize, n: usize) -> (f64, usize, usize) {
        let meta = set.meta();
        let mut cb = CaptureBuilder::<Gl, Gl3>::new();
        set.eval(&mut cb);
        let (prog, _) = cb.finish(num_base_from_meta(&meta));
        let main = GpuLdeBase {
            ready: None,
            buf: Arc::new(math_cuda::p1w16::random_device_matrix(width * n).expect("card")),
            m: width,
            lde_size: n,
            tree: None,
            trace_dev: None,
            trace_rows: 0,
        };
        let aux = GpuLdeExt3 {
            ready: None,
            buf: Arc::new(math_cuda::p1w16::random_device_matrix(3).expect("card")),
            m: 0,
            lde_size: n,
            tree: None,
        };
        let beta: Vec<FieldElement<Gl3>> = (0..prog.roots.len())
            .map(|i| FieldElement::<Gl3>::from(0x9E37_79B9 + 7 * i as u64))
            .collect();
        let z_inv: Vec<FE> = (0..BLOWUP).map(|i| FE::from(3 + i as u64)).collect();
        let inputs = CompositionInputs::<Gl, Gl3> {
            beta_trans: &beta,
            z_inv: &z_inv,
            b_col: &[],
            b_is_aux: &[],
            b_value: &[],
            b_beta: &[],
            b_z_inv: &[],
        };
        let zero = FieldElement::<Gl3>::zero();
        let run = || {
            try_eval_composition_gpu(
                &prog,
                &main,
                &aux,
                &[],
                &[],
                &zero,
                BLOWUP,
                n,
                &inputs,
                false,
            )
            .expect("the card evaluates the program")
        };
        run(); // warm-up: the lowering cache, the module, clocks
        let mut t: Vec<f64> = (0..REPS)
            .map(|_| {
                let start = Instant::now();
                run();
                start.elapsed().as_nanos() as f64
            })
            .collect();
        t.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        (t[REPS / 2] / n as f64, prog.len(), prog.roots.len())
    }

    let rpx_cols = num_columns(HasherKind::Rpx);
    let (t_rpx, nodes_rpx, roots_rpx) = time_set(&HashConstraints::RPX, rpx_cols, n);
    let rpx_cells = 325.0;
    println!(
        "EVAL rpx            cols {rpx_cols} nodes {nodes_rpx} roots {roots_rpx} · {t_rpx:.3} ns/LDE-row · \
         {:.1} ns/perm (b{BLOWUP})",
        BLOWUP as f64 * t_rpx
    );
    let w12_cols = num_columns(HasherKind::Poseidon);
    let (t_w12, nodes_w12, _) = time_set(&HashConstraints::POSEIDON, w12_cols, n);
    println!(
        "EVAL poseidon-w12   cols {w12_cols} nodes {nodes_w12} · {t_w12:.3} ns/LDE-row · {:.1} ns/perm",
        BLOWUP as f64 * t_w12
    );
    for (name, layout, mds) in [
        ("p1w16-compact", Layout::Compact, Mds::Circulant),
        ("p1w16-rule", Layout::Rule, Mds::Circulant),
        ("p1w16c-compact", Layout::Compact, Mds::Cauchy),
        ("p1w16c-rule", Layout::Rule, Mds::Cauchy),
    ] {
        let cells = layout.cells_per_permutation() as f64;
        let (t, nodes, roots) =
            time_set(&P1W16Constraints { layout, mds }, layout.num_columns(), n);
        let e = BLOWUP as f64 * (t - t_rpx * cells / rpx_cells);
        println!(
            "EVAL {name:<15} cols {} nodes {nodes} roots {roots} · {t:.3} ns/LDE-row · {:.1} ns/perm · \
             beyond the cell law {e:+.1} ns/perm = {:+.1} % of {:.0} ns · effective width {:.0} cells",
            layout.num_columns(),
            BLOWUP as f64 * t,
            100.0 * e / (NS_PER_CELL * cells),
            NS_PER_CELL * cells,
            cells + e / NS_PER_CELL
        );
    }
}

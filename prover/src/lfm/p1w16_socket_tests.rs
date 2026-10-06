//! The width-16 socket: its widths, what it accepts and rejects, its bus
//! tokens, and (on the card) what one permutation costs an LFM prove against
//! RPX's `LFM_HASH`.

use math::field::element::FieldElement;
use stark::constraints::builder::{
    CaptureBuilder, ConstraintSet, ProverEvalFolder, RootKind, num_base_from_meta,
};
use stark::frame::Frame;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, BusValue, LinearTerm, Multiplicity};
use stark::proof::options::ProofOptions;
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use crypto::hash::poseidon1_w16 as p1;
use math::field::traits::IsPrimeField;

use super::airs::LfmAir;
use super::p1w16_socket::*;

type Gl = GoldilocksField;
type Gl3 = GoldilocksExtension;

const FORMS: [OutForm; 2] = [OutForm::Columns, OutForm::OnBus];

/// Pinned as literals from the closed forms, so a layout bug cannot agree with
/// itself: 16 `IN` + 8·32 + 22·2 S-box witnesses (+ 16 `OUT`).
const ONBUS_VALUE_COLUMNS: usize = 316;
const COLUMNS_VALUE_COLUMNS: usize = 332;
/// Plus four LogUp aux columns at three cells each.
const ONBUS_CELLS: usize = 328;
const COLUMNS_CELLS: usize = 344;
/// 1 + 300 (+ 16).
const ONBUS_CONSTRAINTS: usize = 301;
const COLUMNS_CONSTRAINTS: usize = 317;

fn evaluate(form: OutForm, row: &[FE]) -> Vec<FE> {
    let set = P1W16SocketConstraints { form };
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

fn violations(form: OutForm, row: &[FE]) -> Vec<usize> {
    evaluate(form, row)
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != FE::zero())
        .map(|(i, _)| i)
        .collect()
}

fn sample_input() -> [FE; p1::STATE_FELTS] {
    core::array::from_fn(|i| FE::from(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1)))
}

/// A real permutation row: the preprocessed prefix says `IS_REAL`, the value
/// columns are the honest witness.
fn real_row(form: OutForm, input: [FE; p1::STATE_FELTS]) -> (Vec<FE>, [FE; p1::STATE_FELTS]) {
    let mut row = vec![FE::zero(); form.num_columns()];
    row[cols::IS_REAL] = FE::one();
    let out = fill_row(form, input, &mut row);
    (row, out)
}

/// A bus value read off a row, as the LogUp fingerprint reads it.
fn bus_value(v: &BusValue, row: &[FE]) -> FE {
    match v {
        BusValue::Packed { start_column, .. } => row[*start_column],
        BusValue::Linear(terms) => terms.iter().fold(FE::zero(), |acc, t| {
            acc + match t {
                LinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    let k = if *coefficient >= 0 {
                        FE::from(*coefficient as u64)
                    } else {
                        -FE::from(coefficient.unsigned_abs())
                    };
                    k * row[*column]
                }
                LinearTerm::ColumnUnsigned {
                    coefficient,
                    column,
                } => FE::from(*coefficient) * row[*column],
                LinearTerm::Constant(c) => {
                    if *c >= 0 {
                        FE::from(*c as u64)
                    } else {
                        -FE::from(c.unsigned_abs())
                    }
                }
            }
        }),
    }
}

#[test]
fn the_widths_are_the_closed_forms() {
    assert_eq!(cols::PREP_WIDTH, 13);
    assert_eq!(
        cols::PREP_WIDTH,
        crate::lfm::layout::hash::PREP_WIDTH,
        "the same instruction group width as the twelve-felt socket"
    );
    assert_eq!(OutForm::OnBus.value_columns(), ONBUS_VALUE_COLUMNS);
    assert_eq!(OutForm::Columns.value_columns(), COLUMNS_VALUE_COLUMNS);
    assert_eq!(ONBUS_VALUE_COLUMNS, 16 + 8 * 32 + 22 * 2);
    assert_eq!(COLUMNS_VALUE_COLUMNS, ONBUS_VALUE_COLUMNS + 16);
    assert_eq!(OutForm::OnBus.cells_per_permutation(), ONBUS_CELLS);
    assert_eq!(OutForm::Columns.cells_per_permutation(), COLUMNS_CELLS);
    assert_eq!(OutForm::OnBus.num_constraints(), ONBUS_CONSTRAINTS);
    assert_eq!(OutForm::Columns.num_constraints(), COLUMNS_CONSTRAINTS);
    for form in FORMS {
        assert_eq!(bus_interactions(form).len(), 8);
    }
}

/// Every value column is handed out exactly once.
#[test]
fn every_value_column_is_assigned_exactly_once() {
    for form in FORMS {
        let mut seen = vec![0usize; form.num_columns()];
        for n in &mut seen[cols::IN0..cols::IN0 + p1::STATE_FELTS] {
            *n += 1;
        }
        if form == OutForm::Columns {
            for n in &mut seen[form.out0()..form.out0() + p1::STATE_FELTS] {
                *n += 1;
            }
        }
        for r in 0..p1::NUM_ROUNDS {
            for lane in 0..sboxed_lanes(r) {
                seen[form.x3(r, lane)] += 1;
                seen[form.x7(r, lane)] += 1;
            }
        }
        for (c, &n) in seen.iter().enumerate() {
            let want = usize::from(c >= cols::PREP_WIDTH);
            assert_eq!(n, want, "{form:?}: column {c} claimed {n} times");
        }
    }
}

#[test]
fn every_constraint_is_degree_three_or_less_and_some_reach_three() {
    for form in FORMS {
        let set = P1W16SocketConstraints { form };
        let meta = ConstraintSet::<Gl, Gl3>::meta(&set);
        assert_eq!(meta.len(), form.num_constraints(), "{form:?}");
        for (i, m) in meta.iter().enumerate() {
            assert_eq!(m.constraint_idx, i, "{form:?}: meta dense and ordered");
            assert_eq!(m.kind, RootKind::Base);
        }
        let mut cb = CaptureBuilder::<Gl, Gl3>::new();
        set.eval(&mut cb);
        let (_prog, degrees) = cb.finish(num_base_from_meta(&meta));
        assert!(degrees.iter().all(|&(_, d)| d <= 3), "{form:?}");
        assert_eq!(degrees.iter().map(|&(_, d)| d).max(), Some(3), "{form:?}");
    }
}

/// An honest row satisfies every constraint, and the filler's output is the
/// host reference's permutation (and the first known answer).
#[test]
fn honest_rows_satisfy_every_constraint_and_match_the_host_reference() {
    const KAT0_OUT_LANE0: u64 = 9_350_316_517_402_464_675;
    for form in FORMS {
        for input in [
            sample_input(),
            core::array::from_fn(|i| FE::from(i as u64)),
            [FE::zero(); p1::STATE_FELTS],
            [-FE::one(); p1::STATE_FELTS],
        ] {
            let (row, out) = real_row(form, input);
            assert_eq!(violations(form, &row), Vec::<usize>::new(), "{form:?}");
            assert_eq!(out, p1::permute(input), "{form:?}");
        }
        let (_, out) = real_row(form, core::array::from_fn(|i| FE::from(i as u64)));
        assert_eq!(out[0], FE::from(KAT0_OUT_LANE0), "{form:?}");
    }
}

#[test]
fn an_all_zero_padding_row_satisfies_every_constraint() {
    for form in FORMS {
        let row = vec![FE::zero(); form.num_columns()];
        assert_eq!(violations(form, &row), Vec::<usize>::new(), "{form:?}");
    }
}

/// A padding row cannot carry a permutation: with `IS_REAL = 0` the round
/// constants drop out, so an honest witness is rejected.
#[test]
fn a_real_witness_under_is_real_zero_is_rejected() {
    for form in FORMS {
        let (mut row, _) = real_row(form, sample_input());
        row[cols::IS_REAL] = FE::zero();
        assert!(!violations(form, &row).is_empty(), "{form:?}");
    }
}

/// `IS_REAL` is a bit.
#[test]
fn a_non_boolean_is_real_is_rejected() {
    for form in FORMS {
        let mut row = vec![FE::zero(); form.num_columns()];
        row[cols::IS_REAL] = FE::from(2u64);
        assert!(violations(form, &row).contains(&0), "{form:?}");
    }
}

/// Mutation: perturbing ANY value column of an honest row breaks some
/// constraint, so no column is free.
#[test]
fn every_value_column_is_load_bearing() {
    for form in FORMS {
        let (row, _) = real_row(form, sample_input());
        for c in cols::PREP_WIDTH..form.num_columns() {
            let mut bad = row.clone();
            bad[c] += FE::one();
            assert!(
                !violations(form, &bad).is_empty(),
                "{form:?}: column {c} is unconstrained"
            );
        }
    }
}

/// The tokens: four reads of the input cells at `IS_REAL`, four writes of the
/// output cells at their own multiplicities, and each written word IS the
/// permutation's output cell — under `OnBus` that is the MDS combination the
/// sender carries, under `Columns` the `OUT` columns.
#[test]
fn the_tokens_carry_the_input_and_output_cells() {
    use stark::lookup::BusInteraction;
    for form in FORMS {
        let input = sample_input();
        let (mut row, out) = real_row(form, input);
        for k in 0..CELLS {
            row[cols::IN_ADDR0 + k] = FE::from(100 + k as u64);
            row[cols::OUT_ADDR0 + k] = FE::from(200 + k as u64);
        }
        let ix: Vec<BusInteraction> = bus_interactions(form);
        for (k, i) in ix.iter().enumerate() {
            let cell = k % CELLS;
            let token: Vec<FE> = i.values.iter().map(|v| bus_value(v, &row)).collect();
            assert_eq!(token.len(), 1 + CELL_FELTS, "{form:?} interaction {k}");
            if k < CELLS {
                assert!(!i.is_sender, "{form:?}: interaction {k} reads");
                assert!(matches!(i.multiplicity, Multiplicity::Column(c) if c == cols::IS_REAL));
                assert_eq!(token[0], FE::from(100 + cell as u64));
                assert_eq!(&token[1..], &input[4 * cell..4 * cell + 4], "{form:?}");
            } else {
                assert!(i.is_sender, "{form:?}: interaction {k} writes");
                assert!(
                    matches!(i.multiplicity, Multiplicity::Column(c) if c == cols::MULT0 + cell)
                );
                assert_eq!(token[0], FE::from(200 + cell as u64));
                assert_eq!(&token[1..], &out[4 * cell..4 * cell + 4], "{form:?}");
            }
        }
    }
}

/// A row whose last round is perturbed writes a different output token: the
/// `OnBus` senders read the last round's `x⁷` columns, which the S-box
/// constraints pin, so a forged output must break a constraint.
#[test]
fn a_forged_output_cell_breaks_a_constraint() {
    let form = OutForm::OnBus;
    let (row, _) = real_row(form, sample_input());
    let last = p1::NUM_ROUNDS - 1;
    for lane in 0..p1::STATE_FELTS {
        let mut bad = row.clone();
        bad[form.x7(last, lane)] += FE::one();
        assert!(!violations(form, &bad).is_empty(), "x7 lane {lane}");
    }
}

/// The RPX `LFM_HASH` AIR exactly as the machine builds it.
pub(super) fn rpx_air(
    options: &ProofOptions,
    root: stark::config::Commitment,
) -> LfmAir<super::chips::hash::HashConstraints> {
    use super::chips::hash;
    use super::hash::HasherKind;
    AirWithBuses::new(
        hash::num_columns(HasherKind::Rpx),
        AuxiliaryTraceBuildData {
            interactions: hash::bus_interactions(HasherKind::Rpx),
        },
        options,
        1,
        hash::HashConstraints::RPX,
    )
    .with_name("LFM_HASH")
    .with_preprocessed(root, crate::lfm::layout::hash::PREP_WIDTH)
}

/// The evaluation cost the cell census does not see: each AIR's full
/// composition program (its constraints plus the LogUp terms) as the card
/// lowers it, and whether the compiled-kernel generator would take it.
/// Informational (`--nocapture`).
#[test]
fn composition_program_sizes() {
    use stark::constraint_ir::{DeviceProgram, codegen};
    use stark::traits::AIR;
    let opts = super::proof::aggregation_wrap_options();
    let report = |name: &str, prog: &stark::constraint_ir::ConstraintProgram<Gl, Gl3>| {
        let dev = DeviceProgram::lower(prog);
        println!(
            "PROGRAM {name:<14} nodes {} lowered {} roots {} row-cost {} compiles {}",
            prog.len(),
            dev.nodes.len(),
            dev.roots.len(),
            codegen::estimated_row_cost(&dev),
            codegen::worth_compiling(&dev),
        );
        dev.nodes.len()
    };
    let rpx = report("rpx", rpx_air(&opts, [0u8; 32]).constraint_program());
    assert!(rpx > 0);
    for form in FORMS {
        let n = report(
            &format!("p1w16 {form:?}"),
            air(form, &opts, [0u8; 32]).constraint_program(),
        );
        assert!(n > 0);
    }
}

/// Canonical `u64` of a felt (test printing).
#[allow(dead_code)]
fn canon(x: &FE) -> u64 {
    Gl::canonical(x.value())
}

/// ★ P3 stage 1: one permutation's cost in an LFM prove, the width-16 socket
/// against RPX's `LFM_HASH`, on the card.
///
/// Each arm proves ONE table of `2^k` honest permutation rows through the
/// block prover's `multi_prove` under the LFM's options
/// (`aggregation_wrap_options`, blowup 4), one warm-up then `P3_BENCH_REPS`
/// timed proves on a fresh copy of the trace each (the copy is off the clock).
/// The preprocessed tree is cached after the warm-up, as a program's is after
/// its first prove; both chips have 13 preprocessed columns.
///
/// Arms (`P3_BENCH_ARMS`, comma-separated): `rpx` (production, its compiled
/// composition kernel), `p1bus` / `p1cols` (the socket with its outputs on the
/// bus / in columns, under whatever the card picks for it: a compiled kernel
/// if one exists, else the bounded-slot interpreter), and each of those with
/// `-si` (the bounded-slot interpreter forced). Sizes: `P3_BENCH_LOGS`.
///
/// Prints one `P3BENCH` line per arm and size, and with `instruments` the
/// prover's round split for the last rep.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: proves 2^18..2^20-row hash tables; run on the box with --ignored"]
fn socket_prove_cost_on_the_card() {
    use std::time::Instant;

    use rayon::prelude::*;
    use stark::constraint_ir::gpu_interp::{
        GPU_COMPOSITION_COMPILED_CALLS, GPU_COMPOSITION_SI_CALLS, SiMode, SiTuning,
        override_compiled_constraints, override_interp_si,
    };
    use stark::leaf_layout::table_leaf_layout;
    use stark::prover::IsStarkProver;
    use stark::residency_mode::ResidencyMode;
    use stark::trace::TraceTable;
    use stark::traits::AIR;

    use super::chips::hash::cols as rcols;
    use super::executor::HashRow;
    use super::hash::{HasherKind, LfmHasher};
    use super::instr::HashMode;

    let env_list = |name: &str, default: &str| -> Vec<String> {
        std::env::var(name)
            .unwrap_or_else(|_| default.to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    let logs: Vec<u32> = env_list("P3_BENCH_LOGS", "18,19,20")
        .iter()
        .map(|s| s.parse().expect("P3_BENCH_LOGS: integers"))
        .collect();
    let reps: usize = std::env::var("P3_BENCH_REPS")
        .ok()
        .map(|s| s.parse().expect("P3_BENCH_REPS: an integer"))
        .unwrap_or(5);
    let arms = env_list(
        "P3_BENCH_ARMS",
        "rpx,p1bus,p1cols,rpx-si,p1bus-si,p1cols-si",
    );
    let opts = super::proof::aggregation_wrap_options();

    // Deterministic pseudo-random felts (splitmix64), per row.
    let felt = |row: usize, lane: usize| {
        let mut z = (row as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(lane as u64 + 1);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        FE::from(z ^ (z >> 31))
    };

    // A real row's addresses: distinct per row and cell.
    let addr = |row: usize, k: usize| FE::from((row * 8 + k) as u64);

    let rpx_trace = |n: usize| -> Vec<FE> {
        let w = super::chips::hash::num_columns(HasherKind::Rpx);
        let mut data = vec![FE::zero(); n * w];
        data.par_chunks_mut(w).enumerate().for_each(|(r, out)| {
            let ins: [FE; 12] = core::array::from_fn(|i| felt(r, i));
            let outs = HasherKind::Rpx.permute(ins);
            for k in 0..3 {
                out[rcols::IN_ADDR0 + k] = addr(r, k);
                out[rcols::OUT_ADDR0 + k] = addr(r, 4 + k);
            }
            out[rcols::MODE_P] = FE::one();
            super::trace::fill_hash_row(
                HasherKind::Rpx,
                &HashRow { ins, outs },
                HashMode::Permute,
                out,
            );
        });
        data
    };
    let socket_trace = |form: OutForm, n: usize| -> Vec<FE> {
        let w = form.num_columns();
        let mut data = vec![FE::zero(); n * w];
        data.par_chunks_mut(w).enumerate().for_each(|(r, out)| {
            for k in 0..CELLS {
                out[cols::IN_ADDR0 + k] = addr(r, k);
                out[cols::OUT_ADDR0 + k] = addr(r, 4 + k);
            }
            out[cols::IS_REAL] = FE::one();
            fill_row(form, core::array::from_fn(|i| felt(r, i)), out);
        });
        data
    };

    // The preprocessed root of `data`'s first `prep` columns under the leaf
    // layout the prover resolves for `air` at `n` rows.
    let prep_root =
        |data: &[FE],
         w: usize,
         prep: usize,
         n: usize,
         air: &dyn AIR<Field = Gl, FieldExtension = Gl3, PublicInputs = ()>| {
            let columns: Vec<Vec<FE>> = (0..prep)
                .map(|c| (0..n).map(|r| data[r * w + c]).collect())
                .collect();
            let layout = table_leaf_layout(air, n);
            super::commit::commit_columns_with(&columns, &opts, layout)
        };

    let set_eval = |forced_si: bool| {
        if forced_si {
            override_compiled_constraints(Some(false));
            override_interp_si(Some((SiMode::All, SiTuning::default())));
        } else {
            override_compiled_constraints(None);
            override_interp_si(None);
        }
    };

    let prove = |air: &dyn AIR<Field = Gl, FieldExtension = Gl3, PublicInputs = ()>,
                 data: &[FE],
                 w: usize|
     -> f64 {
        let mut trace = TraceTable::<Gl, Gl3>::new_main(data.to_vec(), w, 1);
        let mut transcript = crate::hash_pin::block_transcript(&[]);
        let t = Instant::now();
        crate::hash_pin::BlockProver::<Gl, Gl3, ()>::multi_prove(
            vec![(air, &mut trace, &())],
            &mut transcript,
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
            ResidencyMode::Retain,
        )
        .expect("the bench table proves");
        t.elapsed().as_secs_f64()
    };

    println!(
        "P3BENCH-HEAD options blowup {} reps {reps} logs {logs:?} arms {arms:?} \
         rpx-cols {} p1bus-cols {} p1cols-cols {}",
        opts.blowup_factor,
        super::chips::hash::num_columns(HasherKind::Rpx),
        OutForm::OnBus.num_columns(),
        OutForm::Columns.num_columns(),
    );
    type DynAir = Box<dyn AIR<Field = Gl, FieldExtension = Gl3, PublicInputs = ()>>;
    for &log in &logs {
        let n = 1usize << log;
        // One trace and AIR per chip, shared by its `-si` arm.
        let mut chips: Vec<(&str, usize, usize, Vec<FE>, DynAir)> = Vec::new();
        for base in ["rpx", "p1bus", "p1cols"] {
            if !arms.iter().any(|a| a.trim_end_matches("-si") == base) {
                continue;
            }
            chips.push(match base {
                "rpx" => {
                    let w = super::chips::hash::num_columns(HasherKind::Rpx);
                    let data = rpx_trace(n);
                    let probe = rpx_air(&opts, [0u8; 32]);
                    let root = prep_root(&data, w, rcols::PREP_WIDTH, n, &probe);
                    (base, 325, w, data, Box::new(rpx_air(&opts, root)) as DynAir)
                }
                _ => {
                    let form = if base == "p1bus" {
                        OutForm::OnBus
                    } else {
                        OutForm::Columns
                    };
                    let w = form.num_columns();
                    let data = socket_trace(form, n);
                    let probe = air(form, &opts, [0u8; 32]);
                    let root = prep_root(&data, w, cols::PREP_WIDTH, n, &probe);
                    let a = Box::new(air(form, &opts, root)) as DynAir;
                    (base, form.cells_per_permutation(), w, data, a)
                }
            });
        }
        for arm in &arms {
            let base = arm.trim_end_matches("-si");
            assert!(
                chips.iter().any(|c| c.0 == base),
                "P3_BENCH_ARMS: unknown arm {arm:?}"
            );
        }
        // One prove of `arm`: (seconds, the composition engine it ran).
        let run = |arm: &str| -> (f64, &'static str) {
            let base = arm.trim_end_matches("-si");
            let (_, _, w, data, a) = chips.iter().find(|c| c.0 == base).expect("a chip");
            set_eval(arm.ends_with("-si"));
            let (c0, s0) = (
                GPU_COMPOSITION_COMPILED_CALLS.load(std::sync::atomic::Ordering::Relaxed),
                GPU_COMPOSITION_SI_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            );
            let secs = prove(a.as_ref(), data, *w);
            let (c1, s1) = (
                GPU_COMPOSITION_COMPILED_CALLS.load(std::sync::atomic::Ordering::Relaxed),
                GPU_COMPOSITION_SI_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            );
            let engine = match (c1 - c0, s1 - s0) {
                (c, 0) if c > 0 => "compiled",
                (0, s) if s > 0 => "si",
                (0, 0) => "interp",
                _ => "mixed",
            };
            #[cfg(feature = "instruments")]
            if let Some(tm) = stark::instruments::take() {
                for (name, rows, d, sub) in &tm.table_timings {
                    println!(
                        "P3BENCH-SPLIT arm {arm} log {log} table {name} rows {rows} \
                         main_commits_s {:.4} rounds_2_4_s {:.4} table_s {:.4} \
                         constraints_s {:.4} comp_decompose_s {:.4} comp_commit_s {:.4} \
                         ood_s {:.4} deep_s {:.4} fri_s {:.4} queries_s {:.4}",
                        tm.main_commits.as_secs_f64(),
                        tm.rounds_2_4.as_secs_f64(),
                        d.as_secs_f64(),
                        sub.constraints.as_secs_f64(),
                        sub.comp_decompose.as_secs_f64(),
                        sub.comp_commit.as_secs_f64(),
                        sub.ood.as_secs_f64(),
                        sub.deep_comp.as_secs_f64() + sub.deep_extend.as_secs_f64(),
                        sub.fri_commit.as_secs_f64(),
                        sub.queries.as_secs_f64(),
                    );
                }
            }
            (secs, engine)
        };
        // A warm-up round (discarded), then the arms round-robin, the order
        // reversed every other round so no arm always runs on a warmer card.
        for arm in &arms {
            let (warm, engine) = run(arm);
            println!("P3BENCH-WARM arm {arm} log {log} warm_s {warm:.4} engine {engine}");
        }
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        let mut engines: Vec<&str> = vec![""; arms.len()];
        for rep in 0..reps {
            let order: Vec<usize> = if rep % 2 == 0 {
                (0..arms.len()).collect()
            } else {
                (0..arms.len()).rev().collect()
            };
            for i in order {
                let (secs, engine) = run(&arms[i]);
                times[i].push(secs);
                engines[i] = engine;
            }
        }
        for (i, arm) in arms.iter().enumerate() {
            let base = arm.trim_end_matches("-si");
            let (_, cells, w, _, _) = chips.iter().find(|c| c.0 == base).expect("a chip");
            let mut t = times[i].clone();
            t.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let median = t[t.len() / 2];
            println!(
                "P3BENCH arm {arm} log {log} rows {n} cols {w} cells {cells} engine {} \
                 median_s {median:.4} min_s {:.4} ns_per_perm {:.1} ns_per_cell {:.3} reps {:?}",
                engines[i],
                t[0],
                median * 1e9 / n as f64,
                median * 1e9 / (n as f64 * *cells as f64),
                times[i],
            );
        }
    }
    set_eval(false);
}

//! The width-16 socket: its widths, what it accepts and rejects, its bus
//! tokens, and (on the card) what one permutation costs an LFM prove against
//! RPX's `LFM_HASH`.

use math::field::element::FieldElement;
use stark::constraints::builder::{
    CaptureBuilder, ConstraintSet, ProverEvalFolder, RootKind, num_base_from_meta,
};
use stark::frame::Frame;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, BusValue, Multiplicity};
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
/// 1 + 4 + 300 (+ 16).
const ONBUS_CONSTRAINTS: usize = 305;
const COLUMNS_CONSTRAINTS: usize = 321;

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

/// A bus value read off a row by the production evaluator,
/// `BusValue::combine_from` (the LogUp fingerprint's, with its own i64
/// coefficient conversion): one felt per socket value.
fn bus_value(v: &BusValue, row: &[FE]) -> FE {
    let got = v.combine_from(|c| row[c]);
    assert_eq!(got.len(), 1, "a socket value is one felt");
    got[0]
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

/// A padding row cannot write: an output multiplicity under `IS_REAL = 0`
/// is rejected (its `IN` is unread, so the write would be prover-chosen).
#[test]
fn an_output_multiplicity_on_a_padding_row_is_rejected() {
    for form in FORMS {
        for k in 0..CELLS {
            let mut row = vec![FE::zero(); form.num_columns()];
            row[cols::MULT0 + k] = FE::one();
            assert_eq!(violations(form, &row), vec![1 + k], "{form:?} cell {k}");
            // The same multiplicity on a real row is fine.
            let (mut real, _) = real_row(form, sample_input());
            real[cols::MULT0 + k] = FE::from(3u64);
            assert_eq!(
                violations(form, &real),
                Vec::<usize>::new(),
                "{form:?} cell {k}"
            );
        }
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
/// lowers it. Informational (`--nocapture`).
#[test]
fn composition_program_sizes() {
    use stark::constraint_ir::DeviceProgram;
    use stark::traits::AIR;
    let opts = super::proof::aggregation_wrap_options();
    let report = |name: &str, prog: &stark::constraint_ir::ConstraintProgram<Gl, Gl3>| {
        let dev = DeviceProgram::lower(prog);
        println!(
            "PROGRAM {name:<14} nodes {} lowered {} roots {}",
            prog.len(),
            dev.nodes.len(),
            dev.roots.len(),
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

//! Regression tests for the single-source LogUp constraint bodies
//! ([`emit_logup_constraints`]) run three ways from ONE definition. For
//! every layout we assert, on 1000
//! random two-step frames: [`ProverEvalFolder`] == capture→`eval_program`
//! (prover) and [`VerifierEvalFolder`] == capture→`eval_program_verifier`
//! (verifier) — all bit-for-bit.
//!
//! Coverage: the accumulated constraint's 1-absorbed AND 2-absorbed branches
//! (the latter folds two absorbed interactions, degree 3), the batched-term
//! constraint, and every [`Packing`] variant's fingerprint contribution.
use crate::constraint_ir::{eval_program, eval_program_verifier};
use crate::constraints::builder::{
    CaptureBuilder, ProverEvalFolder, RootKind, VerifierEvalFolder, num_base_from_meta,
};
use crate::frame::Frame;
use crate::lookup::*;
use crate::table::TableView;
use crate::trace::TraceTable;
use crate::traits::TransitionEvaluationContext;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;

type Fp = FieldElement<Gl>;
type Fp3 = FieldElement<Ext3>;

const TRIALS: usize = 1000;

/// A tiny deterministic SplitMix64 PRNG (no `rand` dependency).
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Number of aux columns the layout uses: committed term columns + the
/// accumulated column.
fn num_aux_cols(layout: &LogUpLayout) -> usize {
    if layout.interactions.is_empty() {
        0
    } else {
        layout.num_term_columns + 1
    }
}

fn rand_fp3(rng: &mut SplitMix64) -> Fp3 {
    FieldElement::<Ext3>::new([
        Fp::from(rng.next_u64()),
        Fp::from(rng.next_u64()),
        Fp::from(rng.next_u64()),
    ])
}

/// Forward-accumulation contract for [`build_accumulated_column_from_terms`]:
/// `acc[0] = 0` and the circular recurrence tied to the CURRENT row's terms
/// holds on EVERY row, including the wraparound (which closes the cycle back
/// to `acc[0]`). This is the invariant the OOD pruning relies on — only
/// `acc` is read at the next row; every term is read at the current row.
#[test]
fn accumulated_column_is_forward_and_circular() {
    let mut rng = SplitMix64::new(0xC0FF_EE12_3456_789A);
    let n_rows = 8usize;
    let n_term_cols = 2usize;

    let term_columns: Vec<Vec<Fp3>> = (0..n_term_cols)
        .map(|_| (0..n_rows).map(|_| rand_fp3(&mut rng)).collect())
        .collect();

    // Accumulated column follows the committed term columns.
    let acc_col_idx = n_term_cols;
    let mut trace = TraceTable::<Gl, Ext3>::new_main(vec![Fp::zero(); n_rows], 1, 1);
    trace.allocate_aux_table(n_term_cols + 1);

    let l = build_accumulated_column_from_terms(acc_col_idx, &term_columns, &mut trace);

    // Forward accumulation starts at zero.
    assert_eq!(
        *trace.get_aux(0, acc_col_idx),
        Fp3::zero(),
        "acc[0] must be 0 under forward accumulation"
    );

    // Circular recurrence tied to the CURRENT row's terms, on every row.
    // Multiplied through by N to avoid dividing L by N:
    //   (acc[(i+1) mod N] - acc[i]) * N == (Σ terms[i]) * N - L
    let n_fe = Fp3::from(n_rows as u64);
    for i in 0..n_rows {
        let mut row_sum = Fp3::zero();
        for col in &term_columns {
            row_sum = row_sum + &col[i];
        }
        let acc_i = *trace.get_aux(i, acc_col_idx);
        let acc_next = *trace.get_aux((i + 1) % n_rows, acc_col_idx);
        let lhs = (acc_next - acc_i) * &n_fe;
        let rhs = row_sum * &n_fe - &l;
        assert_eq!(lhs, rhs, "forward circular recurrence broken at row {i}");
    }
}

/// The permanent regression check for one layout, on `TRIALS` random
/// two-step frames: the LogUp body run three ways from ONE definition must
/// agree bit-for-bit — [`ProverEvalFolder`] == capture→[`eval_program`]
/// (prover) and [`VerifierEvalFolder`] == capture→[`eval_program_verifier`]
/// (verifier).
fn check_layout(label: &str, layout: &LogUpLayout, num_main_cols: usize) {
    let n_base = 0usize; // LogUp constraints are all extension-rooted.
    let n = layout.num_constraints();

    // Metadata self-consistency: derived from the LogUp emission itself
    // (MetaBuilder), it must be all-ext, dense, and match the
    // batched/accumulated degree formula (3 per batched term; 1 + absorbed
    // for the accumulator).
    let meta = {
        let mut mb = crate::constraints::builder::MetaBuilder::new();
        emit_logup_constraints::<Gl, Ext3, _>(&mut mb, layout, n_base);
        mb.into_meta()
    };
    assert_eq!(meta.len(), n, "[{label}] meta count");
    let num_base = num_base_from_meta(&meta);
    assert_eq!(num_base, 0, "[{label}] LogUp meta is all-ext");
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i, "[{label}] meta idx {i}");
        assert_eq!(m.kind, RootKind::Ext, "[{label}] meta kind {i}");
    }

    // Capture once; the tree-measured degree must match the batched/
    // accumulated formula, and `logup_max_degree` must equal their max.
    let mut cb = CaptureBuilder::<Gl, Ext3>::new();
    emit_logup_constraints(&mut cb, layout, n_base);
    let (prog, degrees) = cb.finish(num_base);
    assert_eq!(degrees.len(), n, "[{label}] one emit per constraint");
    // Release-safe exact-once check: the emitted indices must be exactly
    // 0..n (the per-emit EmitTracker only exists under debug_assertions,
    // which a --release test build compiles out).
    let mut emitted: Vec<usize> = degrees.iter().map(|&(idx, _)| idx).collect();
    emitted.sort_unstable();
    assert!(
        emitted.iter().enumerate().all(|(i, &idx)| i == idx),
        "[{label}] emitted constraint indices are not exactly 0..{n}: {emitted:?}"
    );
    for &(idx, measured) in &degrees {
        let expected_degree = if idx < layout.num_committed_pairs {
            3
        } else {
            1 + layout.absorbed().len()
        };
        assert_eq!(measured, expected_degree, "[{label}] degree {idx}");
    }
    assert_eq!(
        logup_max_degree(layout),
        degrees.iter().map(|&(_, d)| d).max().unwrap_or(0),
        "[{label}] logup_max_degree matches max measured degree"
    );

    let n_aux = num_aux_cols(layout);

    for trial in 0..TRIALS {
        let mut rng = SplitMix64::new(0xC0FF_EE00_u64 ^ (label.len() as u64) ^ trial as u64);

        // Random two-step prover frame.
        let mk_step = |rng: &mut SplitMix64| {
            let main: Vec<Fp> = (0..num_main_cols)
                .map(|_| Fp::from(rng.next_u64()))
                .collect();
            let aux: Vec<Fp3> = (0..n_aux).map(|_| rand_fp3(rng)).collect();
            TableView::new(vec![main], vec![aux])
        };
        let frame = Frame::<Gl, Ext3>::new(vec![mk_step(&mut rng), mk_step(&mut rng)]);
        let rap_challenges = vec![rand_fp3(&mut rng), rand_fp3(&mut rng)]; // [z, alpha]
        let alpha_powers: Vec<Fp3> = (0..12).map(|_| rand_fp3(&mut rng)).collect();
        let table_offset = rand_fp3(&mut rng);

        let prover_ctx = TransitionEvaluationContext::new_prover(
            frame.as_row_frame(),
            &rap_challenges,
            &alpha_powers,
            &table_offset,
        );

        // --- ProverEvalFolder == capture → interpret (prover) ---
        let mut base_out = vec![Fp::zero(); n_base];
        let mut ext_out = vec![Fp3::zero(); n];
        let mut folder = ProverEvalFolder::new(&prover_ctx, &mut base_out, &mut ext_out);
        emit_logup_constraints(&mut folder, layout, n_base);
        folder.assert_all_emitted();

        let mut ir_base = vec![Fp::zero(); n_base];
        let mut ir_ext = vec![Fp3::zero(); n];
        eval_program(&prog, &prover_ctx, &mut ir_base, &mut ir_ext);
        for i in 0..n {
            assert_eq!(
                ext_out[i], ir_ext[i],
                "[{label}] prover folder vs interpreter mismatch, constraint {i}, trial {trial}"
            );
        }

        // --- verifier-side: embed the same frame into the extension ---
        let embed_step = |step: &TableView<Gl, Ext3>| -> TableView<Ext3, Ext3> {
            let main: Vec<Fp3> = (0..num_main_cols)
                .map(|c| step.get_main_evaluation_element(0, c).to_extension())
                .collect();
            let aux: Vec<Fp3> = (0..n_aux)
                .map(|c| *step.get_aux_evaluation_element(0, c))
                .collect();
            TableView::new(vec![main], vec![aux])
        };
        let vframe: Frame<Ext3, Ext3> = Frame::new(vec![
            embed_step(frame.get_evaluation_step(0)),
            embed_step(frame.get_evaluation_step(1)),
        ]);
        let vctx = TransitionEvaluationContext::<Gl, Ext3>::new_verifier(
            &vframe,
            &rap_challenges,
            &alpha_powers,
            &table_offset,
        );

        // --- VerifierEvalFolder == capture → interpret (verifier) ---
        let mut vext_out = vec![Fp3::zero(); n];
        let mut vfolder = VerifierEvalFolder::new(&vctx, &mut vext_out);
        emit_logup_constraints(&mut vfolder, layout, n_base);
        vfolder.assert_all_emitted();

        let mut ir_vext = vec![Fp3::zero(); n];
        eval_program_verifier(&prog, &vctx, &mut ir_vext);
        for i in 0..n {
            assert_eq!(
                vext_out[i], ir_vext[i],
                "[{label}] verifier folder vs interpreter mismatch, constraint {i}, trial {trial}"
            );
        }

        // Prover base-promotion and verifier evaluations must agree
        // (the prover frame embedded == the verifier frame).
        for i in 0..n {
            assert_eq!(
                ext_out[i], vext_out[i],
                "[{label}] prover vs verifier folder mismatch, constraint {i}, trial {trial}"
            );
        }
    }
}

/// A sender interaction with a `Direct`-packed value at column 1.
fn direct_sender(bus_id: u64) -> BusInteraction {
    BusInteraction::sender(
        bus_id,
        Multiplicity::Column(0),
        vec![BusValue::Packed {
            start_column: 1,
            packing: Packing::Direct,
        }],
    )
}

/// A receiver interaction with a single `column(3)` value.
fn column_receiver(bus_id: u64) -> BusInteraction {
    BusInteraction::receiver(bus_id, Multiplicity::Column(2), vec![BusValue::column(3)])
}

#[test]
fn logup_one_absorbed() {
    // 3 interactions → split(3) = (1 committed pair, 1 absorbed):
    //   idx 0: batched term (interactions 0,1)
    //   idx 1: accumulated, 1 absorbed (interaction 2), degree 2.
    let interactions = vec![direct_sender(7), column_receiver(11), direct_sender(13)];
    let layout = LogUpLayout::from_interactions(interactions);
    assert_eq!(layout.num_committed_pairs, 1);
    assert_eq!(layout.absorbed().len(), 1, "must exercise 1-absorbed");
    check_layout("one_absorbed", &layout, 8);
}

#[test]
fn logup_two_absorbed() {
    // 4 interactions → split(4) = (1 committed pair, 2 absorbed):
    //   idx 0: batched term (interactions 0,1)
    //   idx 1: accumulated, 2 absorbed (interactions 2,3), degree 3.
    let interactions = vec![
        direct_sender(7),
        column_receiver(11),
        direct_sender(13),
        column_receiver(17),
    ];
    let layout = LogUpLayout::from_interactions(interactions);
    assert_eq!(layout.num_committed_pairs, 1);
    assert_eq!(layout.absorbed().len(), 2, "must exercise 2-absorbed");
    check_layout("two_absorbed", &layout, 8);
}

#[test]
fn logup_two_interactions_absorbed_only() {
    // 2 interactions → split(2) = (0 committed pairs, 2 absorbed): the
    // accumulated constraint alone, degree 3, no batched term.
    let interactions = vec![direct_sender(7), column_receiver(11)];
    let layout = LogUpLayout::from_interactions(interactions);
    assert_eq!(layout.num_committed_pairs, 0);
    assert_eq!(layout.num_constraints(), 1);
    check_layout("two_absorbed_only", &layout, 8);
}

#[test]
fn logup_all_packing_variants() {
    // Drive every Packing arm through the fingerprint of a committed pair
    // and an absorbed interaction. DWordBL/QuadHL are the widest (8 cols);
    // give a generous column budget.
    const ALL_PACKINGS: [Packing; 10] = [
        Packing::Direct,
        Packing::Word2L,
        Packing::Word4L,
        Packing::DWordWL,
        Packing::DWordHHW,
        Packing::DWordWHH,
        Packing::DWordHL,
        Packing::DWordBL,
        Packing::QuadHL,
        Packing::QuadWL,
    ];
    for packing in ALL_PACKINGS {
        // 3 interactions: two committed (pair) + one absorbed, all using the
        // packing at column 0.
        let mk = |bus: u64, sender: bool| {
            let values = vec![BusValue::Packed {
                start_column: 0,
                packing,
            }];
            if sender {
                BusInteraction::sender(bus, Multiplicity::One, values)
            } else {
                BusInteraction::receiver(bus, Multiplicity::One, values)
            }
        };
        let interactions = vec![mk(3, true), mk(5, false), mk(7, true)];
        let layout = LogUpLayout::from_interactions(interactions);
        check_layout(
            &format!("packing_{packing:?}"),
            &layout,
            packing.num_columns(),
        );
    }
}

#[test]
fn logup_two_committed_pairs() {
    // >= 2 committed pairs: split(6) = (2 pairs, 2 absorbed). Exercises
    // the batched-term loop past its first iteration (pair_idx*2
    // interaction indexing, per-pair term columns) and the accumulated
    // constraint's committed-term sum over more than one aux column —
    // the layout shape every production table has, which the fixtures
    // above (<= 4 interactions, <= 1 pair) never reach.
    let interactions = vec![
        direct_sender(3),
        column_receiver(5),
        direct_sender(7),
        column_receiver(11),
        direct_sender(13),
        column_receiver(17),
    ];
    let layout = LogUpLayout::from_interactions(interactions);
    assert_eq!(layout.num_committed_pairs, 2, "must exercise >= 2 pairs");
    assert_eq!(layout.absorbed().len(), 2);
    assert_eq!(layout.num_constraints(), 3); // 2 batched terms + accumulated
    check_layout("two_committed_pairs", &layout, 8);
}

#[test]
fn logup_linear_zero_skip() {
    // The prover folder zero-skips the F×E multiply for Linear bus
    // elements ([`ConstraintBuilder::fold_fingerprint_term`]); the random
    // frames above never produce a zero element, so drive both always-zero
    // shapes explicitly — the constant-0 bus-width padding and a
    // column-minus-itself combination — next to a nonzero element, and
    // assert the folder still matches the (skip-free) captured program
    // bit-for-bit.
    let zero_padded = |bus: u64, sender: bool| {
        let values = vec![
            BusValue::column(1),
            BusValue::linear(vec![LinearTerm::Constant(0)]),
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: 2,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: 2,
                },
            ]),
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: 3,
                column: 3,
            }]),
        ];
        if sender {
            BusInteraction::sender(bus, Multiplicity::Column(0), values)
        } else {
            BusInteraction::receiver(bus, Multiplicity::Column(0), values)
        }
    };
    let interactions = vec![
        zero_padded(3, true),
        zero_padded(5, false),
        zero_padded(7, true),
    ];
    let layout = LogUpLayout::from_interactions(interactions);
    assert_eq!(layout.num_committed_pairs, 1);
    assert_eq!(layout.absorbed().len(), 1);
    check_layout("linear_zero_skip", &layout, 8);
}

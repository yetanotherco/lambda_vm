//! Tests for the ECSM core table — constraint satisfaction on generated traces
//! and the single-source constraint count.

use crate::tables::ecsm::{EcsmConstraints, EcsmOperation, cols, generate_ecsm_trace};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use ecsm::compute_witness;
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::lookup::PackingShifts;
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

fn gx_le() -> [u8; 32] {
    // secp256k1 Gx, little-endian.
    let mut be = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    be.reverse();
    be
}

fn k_le(v: u64) -> [u8; 32] {
    let mut k = [0u8; 32];
    k[..8].copy_from_slice(&v.to_le_bytes());
    k
}

fn op_for(k: u64) -> EcsmOperation {
    let witness = compute_witness(&k_le(k), &gx_le()).unwrap();
    EcsmOperation {
        timestamp: 444,
        addr_xg: 0x2000,
        addr_k: 0x3000,
        addr_xr: 0x1000,
        witness,
    }
}

/// Evaluate the ECSM [`ConstraintSet`] on one trace row (the compiled prover
/// folder path), returning every base-field constraint value.
fn eval_row(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<FE> {
    let main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect();
    let n = EcsmConstraints.meta().len();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![main],
        vec![vec![]],
    )]);
    let shifts = PackingShifts::<GoldilocksField>::new();
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(&frame, &[], &no_e, &no_e, &offset_e, &shifts);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    EcsmConstraints.eval(&mut folder);
    base
}

/// Every ECSM constraint evaluates to zero on a generated trace (real + padding
/// rows). This exercises the padding closure (`q1 = p`, µ-gated `b`) end to end.
#[test]
fn constraints_hold_on_generated_trace() {
    let ops: Vec<EcsmOperation> = [1u64, 2, 5, 0xFFFF, 1_000_003]
        .iter()
        .map(|&k| op_for(k))
        .collect();
    let trace = generate_ecsm_trace(&ops);

    for row in 0..trace.num_rows() {
        for (i, v) in eval_row(&trace, row).iter().enumerate() {
            assert_eq!(*v, FE::zero(), "constraint {i} must hold at row {row}");
        }
    }
}

#[test]
fn constraint_set_count() {
    assert_eq!(EcsmConstraints.meta().len(), 148);
}

//! Tests for the ECDAS double/add table — the `R_BYTES` offset constant,
//! constraint satisfaction on generated traces across many scalars, and the
//! single-source constraint count.

use crate::tables::ecdas::{EcdasConstraints, EcdasOperation, R_BYTES, cols, generate_ecdas_trace};
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

fn ops_for_bytes(k_le: &[u8; 32]) -> Vec<EcdasOperation> {
    let w = compute_witness(k_le, &gx_le()).unwrap();
    w.steps
        .into_iter()
        .map(|step| EcdasOperation {
            timestamp: 444,
            step,
        })
        .collect()
}

fn ops_for(k: u64) -> Vec<EcdasOperation> {
    ops_for_bytes(&k_le(k))
}

/// Evaluate the ECDAS [`ConstraintSet`] on one trace row (the compiled prover
/// folder path), returning every base-field constraint value.
fn eval_row(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<FE> {
    let main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect();
    let n = EcdasConstraints.meta().len();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![main],
        vec![vec![]],
    )]);
    let shifts = PackingShifts::<GoldilocksField>::new();
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx = TransitionEvaluationContext::new_prover(
        frame.as_row_frame(),
        &no_e,
        &no_e,
        &offset_e,
        &shifts,
    );
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    EcdasConstraints.eval(&mut folder);
    base
}

fn assert_trace_holds(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, label: &str) {
    for row in 0..trace.num_rows() {
        for (i, v) in eval_row(trace, row).iter().enumerate() {
            assert_eq!(
                *v,
                FE::zero(),
                "{label}: constraint {i} must hold at row {row}"
            );
        }
    }
}

#[test]
fn r_bytes_is_three_p() {
    // 3·p as 33 little-endian bytes, cross-checked against the ecsm field modulus.
    let p = ecsm::p();
    let three_p = &p * 3u32;
    let mut bytes = three_p.to_bytes_le();
    bytes.resize(33, 0);
    assert_eq!(&bytes[..], &R_BYTES[..]);
}

/// Every ECDAS constraint evaluates to zero on a generated trace across many
/// scalars (exercising both double and add steps), including padding rows.
#[test]
fn constraints_hold_on_generated_trace() {
    for k in [2u64, 3, 5, 7, 0xFF, 0xABCD, 1_000_003] {
        let ops = ops_for(k);
        assert!(!ops.is_empty(), "k={k} should have steps");
        let trace = generate_ecdas_trace(&ops);
        assert_trace_holds(&trace, &format!("k={k}"));
    }
}

/// Worst-case carries: N-1 (largest valid scalar) runs the full 256-bit ladder.
#[test]
fn constraints_hold_for_near_order_scalar() {
    let mut k = ecsm::N_BYTES;
    k[0] -= 1;
    let ops = ops_for_bytes(&k);
    assert!(!ops.is_empty());
    let trace = generate_ecdas_trace(&ops);
    assert_trace_holds(&trace, "N-1");
}

#[test]
fn constraint_set_count() {
    assert_eq!(EcdasConstraints.meta().len(), 200);
}

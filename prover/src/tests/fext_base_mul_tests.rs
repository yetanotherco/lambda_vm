//! Tests for the FEXT_BASE_MUL table: constraint satisfaction on generated
//! traces, the constraint/bus counts, base-limb reconstruction, and negative
//! checks for a wrong output.

use crate::tables::fext_base_mul::{
    FextBaseMulConstraints, FextBaseMulOperation, bus_interactions, cols,
    generate_fext_base_mul_trace,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

/// `out[d] = base · ext[d]` over Goldilocks, canonical coefficients.
fn base_mul(base: u64, ext: [u64; 3]) -> [u64; 3] {
    let b = GoldilocksElement::from(base);
    [
        (b * GoldilocksElement::from(ext[0])).canonical_u64(),
        (b * GoldilocksElement::from(ext[1])).canonical_u64(),
        (b * GoldilocksElement::from(ext[2])).canonical_u64(),
    ]
}

fn op(base: u64, ext: [u64; 3]) -> FextBaseMulOperation {
    FextBaseMulOperation {
        timestamp: 100,
        base,
        ext_addr: 0x10,
        out_addr: 0x40,
        ext,
        output: base_mul(base, ext),
        read_old_ts: [0; 3],
        write_old_ts: [0; 3],
        write_old_val: [0; 3],
    }
}

/// Evaluate the FEXT_BASE_MUL constraint set on one main-trace row.
fn eval_main_row(main: Vec<FE>) -> Vec<FE> {
    let n = FextBaseMulConstraints.meta().len();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![main],
        vec![vec![]],
    )]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    FextBaseMulConstraints.eval(&mut folder);
    base
}

fn eval_row(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<FE> {
    let main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect();
    eval_main_row(main)
}

#[test]
fn fext_base_mul_constraint_count_is_four() {
    // idx 0: IS_BIT(μ); idx 1-3: the three coefficient equations.
    assert_eq!(FextBaseMulConstraints.meta().len(), 4);
}

#[test]
fn fext_base_mul_max_degree_is_two() {
    assert_eq!(FextBaseMulConstraints.max_degree(), 2);
}

#[test]
fn fext_base_mul_bus_interaction_count() {
    // 1 Ecall + 3 register reads + 3 field reads + 3 field writes, each field
    // access = consume-old + emit-new + old_ts<ts = 1 + 3 + 9 + 9.
    assert_eq!(bus_interactions().len(), 22);
}

#[test]
fn fext_base_mul_constraints_hold_on_valid_trace() {
    let ops = vec![
        op(1, [1, 2, 3]),
        op(0, [9, 9, 9]), // base 0 -> all-zero output
        op(7, [0, 0, 0]),
        op(12345, [222324, 252627, 282930]),
        // High 32 bits set: exercises base_lo + 2^32·base_hi reconstruction.
        op((1u64 << 32) + 5, [1, 2, 3]),
        op(0xFFFF_FFFF_0000_0000, [4, 5, 6]),
    ];
    let trace = generate_fext_base_mul_trace(&ops);

    for row in 0..trace.num_rows() {
        for (idx, v) in eval_row(&trace, row).into_iter().enumerate() {
            assert_eq!(v, FE::zero(), "row {row}, constraint {idx} nonzero");
        }
    }
}

#[test]
fn fext_base_mul_detects_wrong_output() {
    let ops = vec![op(3, [4, 5, 6])];
    let mut trace = generate_fext_base_mul_trace(&ops);
    // Corrupt out1 (constraint index 2).
    trace.main_table.set_fe(0, cols::OUT1, FE::from(999_999u64));
    let vals = eval_row(&trace, 0);
    assert_ne!(
        vals[2],
        FE::zero(),
        "corrupted out1 must break constraint 2"
    );
}

#[test]
fn fext_base_mul_detects_wrong_base() {
    // A wrong base high limb must break the reconstruction and thus every
    // coefficient equation with a nonzero ext.
    let ops = vec![op(5, [1, 1, 1])];
    let mut trace = generate_fext_base_mul_trace(&ops);
    trace.main_table.set_fe(0, cols::BASE_1, FE::from(1u64)); // adds 2^32 to base
    let vals = eval_row(&trace, 0);
    assert_ne!(
        vals[1],
        FE::zero(),
        "corrupted base must break constraint 1"
    );
}

#[test]
fn fext_base_mul_detects_non_bit_mu() {
    let ops = vec![op(3, [4, 5, 6])];
    let mut trace = generate_fext_base_mul_trace(&ops);
    trace.main_table.set_fe(0, cols::MU, FE::from(2u64));
    let vals = eval_row(&trace, 0);
    assert_ne!(
        vals[0],
        FE::zero(),
        "μ = 2 must break IS_BIT (constraint 0)"
    );
}

#[test]
fn fext_base_mul_trace_shape() {
    let ops = vec![op(3, [4, 5, 6])];
    let trace = generate_fext_base_mul_trace(&ops);
    assert_eq!(trace.num_rows(), 4);
    assert_eq!(*trace.main_table.get(0, cols::MU), FE::one());
    for row in 1..4 {
        assert_eq!(*trace.main_table.get(row, cols::MU), FE::zero());
    }
}

#[test]
fn fext_base_mul_literal_vectors() {
    // base · ext scales each coefficient independently (no cross terms, unlike a
    // full extension multiply).
    assert_eq!(base_mul(2, [1, 3, 5]), [2, 6, 10]);
    assert_eq!(base_mul(0, [7, 8, 9]), [0, 0, 0]);
    assert_eq!(base_mul(1, [11, 22, 33]), [11, 22, 33]);
}

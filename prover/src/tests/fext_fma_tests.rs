//! Tests for the FEXT_FMA table: constraint satisfaction on generated traces,
//! the constraint count, and negative checks for a wrong output.

use crate::tables::fext_fma::{
    FextFmaConstraints, FextFmaOperation, bus_interactions, cols, generate_fext_fma_trace,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

/// `a*b + c` over the native Fp3, canonical coefficients.
fn fma(a: [u64; 3], b: [u64; 3], c: [u64; 3]) -> [u64; 3] {
    let to_fp3 = |x: [u64; 3]| {
        Fp3::from_raw([
            GoldilocksElement::from(x[0]),
            GoldilocksElement::from(x[1]),
            GoldilocksElement::from(x[2]),
        ])
    };
    let r = to_fp3(a) * to_fp3(b) + to_fp3(c);
    let v = r.value();
    [
        v[0].canonical_u64(),
        v[1].canonical_u64(),
        v[2].canonical_u64(),
    ]
}

fn op(a: [u64; 3], b: [u64; 3], c: [u64; 3]) -> FextFmaOperation {
    FextFmaOperation {
        timestamp: 100,
        out_addr: 0x40,
        a_addr: 0x10,
        b_addr: 0x20,
        c_addr: 0x30,
        a,
        b,
        c,
        output: fma(a, b, c),
    }
}

/// Evaluate the FEXT_FMA constraint set on one main-trace row.
fn eval_main_row(main: Vec<FE>) -> Vec<FE> {
    let n = FextFmaConstraints.meta().len();
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
    FextFmaConstraints.eval(&mut folder);
    base
}

fn eval_row(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<FE> {
    let main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect();
    eval_main_row(main)
}

#[test]
fn fext_fma_constraint_count_is_four() {
    // idx 0: IS_BIT(μ); idx 1-3: the three coefficient equations.
    assert_eq!(FextFmaConstraints.meta().len(), 4);
}

#[test]
fn fext_fma_max_degree_is_two() {
    assert_eq!(FextFmaConstraints.max_degree(), 2);
}

#[test]
fn fext_fma_bus_interaction_count() {
    // 1 Ecall receiver + 4 register reads.
    assert_eq!(bus_interactions().len(), 5);
}

#[test]
fn fext_fma_constraints_hold_on_valid_trace() {
    let ops = vec![
        op([1, 0, 0], [1, 0, 0], [0, 0, 0]),
        op([0, 1, 0], [0, 1, 0], [0, 0, 0]), // w*w = w^2
        op([0, 0, 1], [0, 0, 1], [0, 0, 0]), // w^2*w^2 = 2w
        op([1, 2, 3], [4, 5, 6], [7, 8, 9]),
        op([12345, 67890, 111213], [222324, 252627, 282930], [1, 2, 3]),
    ];
    let trace = generate_fext_fma_trace(&ops);

    // Real rows and padding rows all satisfy every constraint.
    for row in 0..trace.num_rows() {
        for (idx, v) in eval_row(&trace, row).into_iter().enumerate() {
            assert_eq!(v, FE::zero(), "row {row}, constraint {idx} nonzero");
        }
    }
}

#[test]
fn fext_fma_detects_wrong_output() {
    let ops = vec![op([1, 2, 3], [4, 5, 6], [7, 8, 9])];
    let mut trace = generate_fext_fma_trace(&ops);
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
fn fext_fma_detects_non_bit_mu() {
    let ops = vec![op([1, 2, 3], [4, 5, 6], [7, 8, 9])];
    let mut trace = generate_fext_fma_trace(&ops);
    trace.main_table.set_fe(0, cols::MU, FE::from(2u64));
    let vals = eval_row(&trace, 0);
    assert_ne!(
        vals[0],
        FE::zero(),
        "μ = 2 must break IS_BIT (constraint 0)"
    );
}

#[test]
fn fext_fma_trace_shape() {
    let ops = vec![op([1, 2, 3], [4, 5, 6], [7, 8, 9])];
    let trace = generate_fext_fma_trace(&ops);
    // 1 op → padded to min 4 rows.
    assert_eq!(trace.num_rows(), 4);
    // Real row has μ = 1, padding rows μ = 0.
    assert_eq!(*trace.main_table.get(0, cols::MU), FE::one());
    for row in 1..4 {
        assert_eq!(*trace.main_table.get(row, cols::MU), FE::zero());
    }
}

//! Algebraic and trace mutation tests for the SHA compression chips.
use crate::tables::types::{FE, GoldilocksExtension as E, GoldilocksField as F};
use crate::tables::{sha256, sha256_round, sha256_schedule};
use stark::{
    constraints::builder::{ConstraintSet, ProverEvalFolder},
    frame::Frame,
    table::TableView,
    trace::TraceTable,
    traits::TransitionEvaluationContext,
};
fn holds<C: ConstraintSet<F, E>>(c: C, t: &TraceTable<F, E>) -> bool {
    for row in 0..t.num_rows() {
        let main = (0..t.main_table.width)
            .map(|i| *t.main_table.get(row, i))
            .collect();
        let frame = Frame::<F, E>::new(vec![TableView::new(vec![main], vec![vec![]])]);
        let empty = vec![];
        let zero = math::field::element::FieldElement::<E>::zero();
        let ctx =
            TransitionEvaluationContext::new_prover(frame.as_row_frame(), &empty, &empty, &zero);
        let mut base = vec![FE::zero(); c.meta().len()];
        let mut ext = vec![zero; c.meta().len()];
        c.eval(&mut ProverEvalFolder::new(&ctx, &mut base, &mut ext));
        if base.iter().any(|x| *x != FE::zero()) {
            eprintln!(
                "row {row} failures {:?}",
                base.iter()
                    .enumerate()
                    .filter(|(_, v)| **v != FE::zero())
                    .collect::<Vec<_>>()
            );
            return false;
        }
    }
    true
}
fn ops() -> Vec<sha256::Operation> {
    (0..3)
        .map(|i| sha256::Operation {
            timestamp: 400 + 4 * i,
            state_addr: 0x1003,
            message_addr: 0x2007,
            state: std::array::from_fn(|j| (j * 17 + i as usize) as u8),
            message: std::array::from_fn(|j| (j * 23 + i as usize) as u8),
        })
        .collect()
}
#[test]
fn sha256_constraints_and_mutations() {
    let ops = ops();
    let mut core = sha256::generate(&ops);
    assert!(holds(sha256::Constraints, &core));
    core.main_table.set(0, sha256::OUT, FE::from(256));
    assert!(!holds(sha256::Constraints, &core));
    let mut rounds = sha256_round::generate(&ops);
    assert!(holds(sha256_round::Constraints, &rounds));
    rounds.main_table.set(0, sha256_round::OUT_A, FE::from(2));
    assert!(!holds(sha256_round::Constraints, &rounds));
    let mut schedule = sha256_schedule::generate(&ops);
    assert!(holds(sha256_schedule::Constraints, &schedule));
    // A bit of w[i-15] holding 2: σ0 is an expression over these, so the bit
    // check is what stands between a forged rotation and a valid trace.
    schedule.main_table.set(0, sha256_schedule::B15, FE::from(2));
    assert!(!holds(sha256_schedule::Constraints, &schedule));
    let mut round_bits = sha256_round::generate(&ops);
    round_bits
        .main_table
        .set(0, sha256_round::A, FE::from(2));
    assert!(!holds(sha256_round::Constraints, &round_bits));
}

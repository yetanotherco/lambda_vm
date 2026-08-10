//! HINT constraint tests.

use crate::tables::hint::{
    HINT_ADDR_LIMB_BOUND, HintConstraints, HintOperation, bus_interactions, cols,
    generate_hint_trace,
};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::lookup::{BusValue, LinearTerm};
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

/// Evaluate the HINT constraint set on one main-trace row.
fn eval_main_row(main: Vec<FE>) -> Vec<FE> {
    let n = HintConstraints.meta().len();
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
    HintConstraints.eval(&mut folder);
    base
}

fn op(timestamp: u64, out_addr: u64) -> HintOperation {
    HintOperation {
        timestamp,
        out_addr,
        out_bytes: std::array::from_fn(|i| i as u8),
        hint_id: 0,
        in_addr: 0x3000,
    }
}

#[test]
fn constraint_set_count() {
    assert_eq!(HintConstraints.meta().len(), 1);
}

/// Every constraint holds on a generated trace — real rows (`mu = 1`) and the
/// all-zero padding rows (`mu = 0`) alike.
#[test]
fn constraints_hold_on_generated_trace() {
    let trace = generate_hint_trace(&[op(4, 0x1000), op(8, 0x2000)]);
    for row in 0..trace.num_rows() {
        let main: Vec<FE> = (0..cols::NUM_COLUMNS)
            .map(|c| *trace.main_table.get(row, c))
            .collect();
        for (i, v) in eval_main_row(main).iter().enumerate() {
            assert_eq!(*v, FE::zero(), "constraint {i} must hold at row {row}");
        }
    }
}

/// `IS_BIT(mu)` rejects a row whose multiplicity is not a bit.
///
/// The `Ecall` bus does not establish this on its own: its tuple carries a
/// per-instruction timestamp, so LogUp pins the *sum* of `mu` over the rows sharing a
/// tuple, which a witness can satisfy by spreading `mu` across rows with integer
/// weights summing to 1 (the real exploit uses a `+1`/`-1` pair, not a fractional
/// split; MEMW does not catch it — it only sees the legal `+1`, the `-1` cancelling an
/// honest STORE). This constraint rejects any non-boolean `mu` locally. The test below
/// tampers with a fractional `1/2`, which `IS_BIT` also rejects.
#[test]
fn is_bit_mu_rejects_non_boolean_multiplicity() {
    let trace = generate_hint_trace(&[op(4, 0x1000)]);
    let mut main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(0, c))
        .collect();
    assert_eq!(main[cols::MU], FE::one(), "row 0 must be a real hint row");

    // A halved multiplicity: 1/2 + 1/2 across two rows keeps the Ecall bus balanced.
    let half = (FE::one() / (FE::one() + FE::one())).expect("2 is invertible");
    main[cols::MU] = half;
    assert_ne!(
        eval_main_row(main.clone())[0],
        FE::zero(),
        "IS_BIT(mu) must reject a fractional multiplicity"
    );

    // And any other non-bit value.
    main[cols::MU] = FE::from(2u64);
    assert_ne!(
        eval_main_row(main)[0],
        FE::zero(),
        "IS_BIT(mu) must reject mu = 2"
    );
}

/// The lhs column of an ALU `LT` sender, and the constant it is compared against.
fn alu_lt_senders() -> Vec<(usize, u64)> {
    let id: u64 = BusId::Alu.into();
    bus_interactions()
        .iter()
        .filter(|i| i.is_sender && i.bus_id == id)
        .map(|i| {
            let lhs = match &i.values[0] {
                BusValue::Packed { start_column, .. } => *start_column,
                BusValue::Linear(_) => panic!("LT lhs must be a column, not a constant"),
            };
            let bound = match &i.values[2] {
                BusValue::Linear(terms) => match terms.as_slice() {
                    [LinearTerm::Constant(c)] => *c as u64,
                    _ => panic!("LT rhs must be a single constant"),
                },
                BusValue::Packed { .. } => panic!("LT rhs must be a constant"),
            };
            (lhs, bound)
        })
        .collect()
}

/// Both address low limbs are range-checked, not just `in_addr`.
///
/// `out_addr` is on the memory bus, which is why it originally had no LT sender — but the
/// bus bounds it only to `2^32 - 25` (the largest write base is `out_addr_lo + 24`, and
/// MEMW's carry columns resolve the bytes past it), while the executor rejects anything
/// above `2^32 - 32`. Without this sender the AIR accepted the seven-value window in
/// [`addr_limb_bound_rejects_every_operand_the_executor_rejects`].
#[test]
fn alu_lt_senders_range_check_selector_and_both_address_limbs() {
    let senders = alu_lt_senders();
    assert_eq!(senders.len(), 3, "selector + in_addr + out_addr");

    for col in [cols::ADDR_IN_0, cols::ADDR_OUT_0] {
        let bound = senders
            .iter()
            .find_map(|(lhs, bound)| (*lhs == col).then_some(*bound))
            .unwrap_or_else(|| panic!("column {col} must have an ALU LT range-check"));
        assert_eq!(
            bound, HINT_ADDR_LIMB_BOUND,
            "column {col} must be checked against the executor's bound"
        );
    }
}

/// The bound accepts exactly the operands `addr_limb_ok(addr, 31)` accepts.
///
/// The seven values in `2^32-31 ..= 2^32-25` are the regression: the executor rejects
/// them with `HintAddressOverflow`, and before the `out_addr` sender existed the AIR
/// accepted them for the output address — a provable hint call the VM halts on.
#[test]
fn addr_limb_bound_rejects_every_operand_the_executor_rejects() {
    // `addr_limb_ok(addr, 31)`: the 32-byte range must fit under 2^32.
    let executor_accepts = |limb: u64| limb + 31 < (1 << 32);
    // The AIR accepts iff the LT range-check passes.
    let air_accepts = |limb: u64| limb < HINT_ADDR_LIMB_BOUND;

    for limb in (1u64 << 32) - 40..1u64 << 32 {
        assert_eq!(
            air_accepts(limb),
            executor_accepts(limb),
            "AIR and executor disagree on out_addr low limb {limb:#x}"
        );
    }

    // The window that used to verify while the executor halted on it.
    for limb in (1u64 << 32) - 31..=(1u64 << 32) - 25 {
        assert!(!air_accepts(limb), "{limb:#x} must be rejected");
    }
    // And the largest operand that must still run.
    assert!(air_accepts((1 << 32) - 32));
}

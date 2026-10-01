//! Tests for the CPU32 table — column layout, sign-extension aux math, and the
//! sign-extension / register-zero constraints.

use crate::tables::cpu32::{
    Cpu32Constraints, Cpu32Operation, bus_interactions, cols, generate_cpu32_trace,
};
use crate::tables::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, alu_op, build_alu_flags,
};
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

/// Evaluate the CPU32 [`ConstraintSet`] on one main row, returning every
/// base-field constraint value (the compiled prover folder path).
fn eval_cpu32(row: &[FE]) -> Vec<FE> {
    let n = Cpu32Constraints.meta().len();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![row.to_vec()],
        vec![vec![]],
    )]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    Cpu32Constraints.eval(&mut folder);
    base
}

#[test]
fn test_aux_signed_input_extension() {
    // Signed op (signed bit set in alu_flags) with a negative low word.
    let op = Cpu32Operation {
        rv1: 0x8000_0000, // bit 31 set → negative as i32
        alu_flags: build_alu_flags(alu_op::SHIFTW, true, true, false), // signed = true
        ..Default::default()
    };
    let aux = op.compute_aux();
    assert!(aux.signed);
    assert!(aux.rv1_sign);
    // arg1 sign-extended: high word all ones.
    assert_eq!(aux.arg1, 0xFFFF_FFFF_8000_0000);
}

#[test]
fn test_aux_unsigned_input_zero_extension() {
    // Unsigned op (signed bit clear) with the same low word → zero-extended.
    let op = Cpu32Operation {
        rv1: 0x8000_0000,
        alu_flags: build_alu_flags(alu_op::SHIFTW, false, false, false), // signed = false
        ..Default::default()
    };
    let aux = op.compute_aux();
    assert!(!aux.signed);
    assert_eq!(aux.arg1, 0x0000_0000_8000_0000);
}

#[test]
fn test_aux_arg2_from_immediate() {
    // Immediate path: rv2 = 0, imm fully sign-extended.
    let op = Cpu32Operation {
        rv2: 0,
        read_register2: false,
        imm: 0xFFFF_FFFF_FFFF_FF00,
        alu_flags: build_alu_flags(alu_op::SHIFTW, true, false, false),
        ..Default::default()
    };
    let aux = op.compute_aux();
    assert_eq!(aux.arg2, 0xFFFF_FFFF_FFFF_FF00);
}

#[test]
fn test_aux_arg2_from_register() {
    // Register path: imm = 0, rv2 negative, signed → sign-extended rv2.
    let op = Cpu32Operation {
        rv2: 0x8000_0001,
        read_register2: true,
        imm: 0,
        alu_flags: build_alu_flags(alu_op::SHIFTW, true, true, false), // signed
        ..Default::default()
    };
    let aux = op.compute_aux();
    assert!(aux.rv2_sign);
    assert_eq!(aux.arg2, 0xFFFF_FFFF_8000_0001);
}

#[test]
fn test_aux_rvd_always_sign_extended() {
    // rvd is always sign-extended from the low 32 bits of res, regardless of `signed`.
    let op = Cpu32Operation {
        res: 0x0000_0000_8000_0000, // low word negative
        alu_flags: build_alu_flags(alu_op::SHIFTW, false, false, false), // unsigned op
        ..Default::default()
    };
    let aux = op.compute_aux();
    assert!(aux.res_sign);
    assert_eq!(aux.rvd, 0xFFFF_FFFF_8000_0000);

    // Positive low word → zero high word.
    let op2 = Cpu32Operation {
        res: 0x0000_0000_0000_0001,
        ..Default::default()
    };
    assert_eq!(op2.compute_aux().rvd, 0x0000_0000_0000_0001);
}

#[test]
fn test_trace_layout() {
    let op = Cpu32Operation {
        timestamp: 0x1234,
        pc: 0xABCD,
        rs1: 3,
        read_register1: true,
        rv1: 0x1122_3344_5566_7788,
        rs2: 5,
        read_register2: true,
        rv2: 0x9900,
        rd: 7,
        write_register: true,
        res: 0x42,
        alu: true,
        alu_flags: build_alu_flags(alu_op::SHIFTW, true, true, false),
        half_instruction_length: 2,
        ..Default::default()
    };
    let trace = generate_cpu32_trace(&[op]);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);
    assert_eq!(trace.main_table.height, 4); // padded to min 4

    let row = trace.main_table.get_row(0);
    assert_eq!(row[cols::PC_0], FE::from(0xABCDu64));
    assert_eq!(row[cols::RS1], FE::from(3u64));
    // rv1 as DWordWHH: half0, half1, word.
    assert_eq!(row[cols::RV1_0], FE::from(0x7788u64));
    assert_eq!(row[cols::RV1_1], FE::from(0x5566u64));
    assert_eq!(row[cols::RV1_2], FE::from(0x1122_3344u64));
    assert_eq!(row[cols::RD], FE::from(7u64));
    assert_eq!(row[cols::HALF_INSTRUCTION_LENGTH], FE::from(2u64));
    assert_eq!(row[cols::SIGNED], FE::from(1u64));
    assert_eq!(row[cols::MU], FE::from(1u64));
}

#[test]
fn test_ext_and_regzero_constraints_hold_on_valid_row() {
    // A signed word op via the immediate path (read_register2 = 0, rv2 = 0).
    let op = Cpu32Operation {
        rv1: 0x8000_0001, // negative low word
        read_register1: true,
        rv2: 0,
        read_register2: false,
        imm: 0xFFFF_FFFF_FFFF_FFF0,
        res: 0x0000_0000_1234_5678,
        rd: 5,
        write_register: true,
        alu: true,
        alu_flags: build_alu_flags(alu_op::SHIFTW, true, true, false), // signed
        half_instruction_length: 2,
        ..Default::default()
    };
    let trace = generate_cpu32_trace(&[op]);
    let row = trace.main_table.get_row(0).to_vec();

    // Every CPU32 constraint (sign-extension arithmetic + register-zero checks)
    // holds on the valid row.
    for (i, v) in eval_cpu32(&row).iter().enumerate() {
        assert_eq!(*v, FE::zero(), "constraint {i} must hold on a valid row");
    }
}

#[test]
fn test_constraints_catch_corruption() {
    let op = Cpu32Operation {
        rv1: 0x8000_0001,
        read_register1: true,
        res: 0x0000_0000_8000_0000,
        write_register: true,
        alu: true,
        alu_flags: build_alu_flags(alu_op::SHIFTW, true, true, false),
        half_instruction_length: 2,
        ..Default::default()
    };
    let trace = generate_cpu32_trace(&[op]);

    // Corrupt arg1[1] (the sign-extended high word) → some constraint must fire.
    let mut row = trace.main_table.get_row(0).to_vec();
    row[cols::ARG1_1] += FE::one();
    assert!(
        eval_cpu32(&row).iter().any(|v| *v != FE::zero()),
        "a corrupted arg1[1] must break some constraint"
    );

    // A non-zero unread register value (read_register2 = 0, rv2 ≠ 0) must fire
    // the register-zero check.
    let op2 = Cpu32Operation {
        rv2: 0x1234,           // non-zero
        read_register2: false, // but flagged unread
        ..Default::default()
    };
    let trace2 = generate_cpu32_trace(&[op2]);
    let row2 = trace2.main_table.get_row(0).to_vec();
    assert!(
        eval_cpu32(&row2).iter().any(|v| *v != FE::zero()),
        "rv2≠0 while unread must break some constraint"
    );
}

#[test]
fn test_bus_interactions_shape() {
    let interactions = bus_interactions();
    assert_eq!(interactions.len(), 23);

    let count = |bus: BusId, sender: bool| {
        interactions
            .iter()
            .filter(|i| i.bus_id == u64::from(bus) && i.is_sender == sender)
            .count()
    };

    assert_eq!(count(BusId::Decode, true), 1);
    assert_eq!(count(BusId::AreBytes, true), 5);
    assert_eq!(count(BusId::IsHalfword, true), 8);
    assert_eq!(count(BusId::Memw, true), 3); // rv1 read, rv2 read, rvd write
    assert_eq!(count(BusId::Alu, true), 1);
    assert_eq!(count(BusId::ByteAlu, true), 1);
    assert_eq!(count(BusId::Msb16, true), 3);

    // CPU32 is a receiver (the main CPU sends the delegation).
    let cpu32: Vec<_> = interactions
        .iter()
        .filter(|i| i.bus_id == u64::from(BusId::Cpu32))
        .collect();
    assert_eq!(cpu32.len(), 1);
    assert!(!cpu32[0].is_sender, "CPU32 receives from the main CPU");
    assert_eq!(cpu32[0].values.len(), 3); // [timestamp, pc, instruction_length]
}

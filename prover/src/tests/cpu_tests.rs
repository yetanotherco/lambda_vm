//! Tests for the CPU table.

use crate::tables64::cpu::{CpuOperation, bus_interactions, cols, generate_cpu_trace};
use crate::tables64::types::FE;

#[test]
fn test_cpu_operation_default() {
    let op = CpuOperation::new();
    assert_eq!(op.timestamp, 0);
    assert_eq!(op.pc, 0);
    assert!(!op.op_add);
    assert!(!op.branch_cond);
}

#[test]
fn test_cpu_operation_compute_arg1_no_extension() {
    let mut op = CpuOperation::new();
    op.rv1 = 0x1234_5678_9ABC_DEF0;
    op.word_instr = false;

    assert_eq!(op.compute_arg1(), 0x1234_5678_9ABC_DEF0);
}

#[test]
fn test_cpu_operation_compute_arg1_word_zero_extend() {
    let mut op = CpuOperation::new();
    op.rv1 = 0x1234_5678_9ABC_DEF0;
    op.word_instr = true;
    op.signed = false;

    // Should zero-extend from lower 32 bits
    assert_eq!(op.compute_arg1(), 0x9ABC_DEF0);
}

#[test]
fn test_cpu_operation_compute_arg1_word_sign_extend_positive() {
    let mut op = CpuOperation::new();
    op.rv1 = 0x1234_5678_1ABC_DEF0; // Positive 32-bit value
    op.word_instr = true;
    op.signed = true;

    // Bit 31 is 0, so sign extension keeps it positive
    assert_eq!(op.compute_arg1(), 0x1ABC_DEF0);
}

#[test]
fn test_cpu_operation_compute_arg1_word_sign_extend_negative() {
    let mut op = CpuOperation::new();
    op.rv1 = 0x1234_5678_8000_0001; // Negative when viewed as 32-bit signed
    op.word_instr = true;
    op.signed = true;

    // Bit 31 is 1, so sign extension fills upper 32 bits with 1s
    assert_eq!(op.compute_arg1(), 0xFFFF_FFFF_8000_0001);
}

#[test]
fn test_cpu_operation_compute_arg2_store() {
    let mut op = CpuOperation::new();
    op.rv2 = 0xDEAD_BEEF;
    op.imm = 0x1234;
    op.op_store = true;

    // STORE uses rv2
    assert_eq!(op.compute_arg2(), 0xDEAD_BEEF);
}

#[test]
fn test_cpu_operation_compute_arg2_load() {
    let mut op = CpuOperation::new();
    op.rv2 = 0xDEAD_BEEF;
    op.imm = 0x1234;
    op.op_load = true;

    // LOAD uses rv2
    assert_eq!(op.compute_arg2(), 0xDEAD_BEEF);
}

#[test]
fn test_cpu_operation_compute_arg2_beq() {
    let mut op = CpuOperation::new();
    op.rv2 = 0xCAFE_BABE;
    op.imm = 0x5678;
    op.op_beq = true;

    // BEQ uses rv2
    assert_eq!(op.compute_arg2(), 0xCAFE_BABE);
}

#[test]
fn test_cpu_operation_compute_arg2_add_with_imm() {
    let mut op = CpuOperation::new();
    op.rv2 = 0;
    op.rs2 = 0; // rs2 = 0 means use immediate
    op.imm = 0x1234_5678;
    op.op_add = true;

    // ADD with rs2=0 uses imm
    assert_eq!(op.compute_arg2(), 0x1234_5678);
}

#[test]
fn test_cpu_operation_compute_arg2_add_with_rs2() {
    let mut op = CpuOperation::new();
    op.rv2 = 0xABCD_EF00;
    op.rs2 = 5; // Non-zero rs2
    op.imm = 0x1234_5678;
    op.op_add = true;

    // ADD with rs2 != 0 uses rv2
    assert_eq!(op.compute_arg2(), 0xABCD_EF00);
}

#[test]
fn test_sign_bit_32_positive() {
    assert!(!CpuOperation::sign_bit_32(0x7FFF_FFFF));
    assert!(!CpuOperation::sign_bit_32(0x0000_0000));
    assert!(!CpuOperation::sign_bit_32(0x1234_5678));
}

#[test]
fn test_sign_bit_32_negative() {
    assert!(CpuOperation::sign_bit_32(0x8000_0000));
    assert!(CpuOperation::sign_bit_32(0xFFFF_FFFF));
    assert!(CpuOperation::sign_bit_32(0x8000_0001));
}

#[test]
fn test_trace_generation_basic() {
    let ops = vec![CpuOperation {
        timestamp: 0,
        pc: 0x1000,
        rs1: 1,
        rs2: 2,
        rd: 3,
        write_register: true,
        op_add: true,
        rv1: 10,
        rv2: 20,
        res: 30,
        next_pc: 0x1004,
        rvd: 30,
        ..Default::default()
    }];

    let trace = generate_cpu_trace(&ops);

    // Should be padded to power of 2 (min 4 for FRI)
    assert_eq!(trace.main_table.height, 4);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);

    // Check first row values
    let row0 = trace.main_table.get_row(0);
    assert_eq!(row0[cols::TIMESTAMP], FE::from(0u64));
    assert_eq!(row0[cols::PC_0], FE::from(0x1000u64));
    assert_eq!(row0[cols::PC_1], FE::zero());
    assert_eq!(row0[cols::RS1], FE::from(1u64));
    assert_eq!(row0[cols::RS2], FE::from(2u64));
    assert_eq!(row0[cols::RD], FE::from(3u64));
    assert_eq!(row0[cols::WRITE_REGISTER], FE::one());
    assert_eq!(row0[cols::ADD], FE::one());
    assert_eq!(row0[cols::SUB], FE::zero());
}

#[test]
fn test_trace_generation_64bit_pc() {
    let ops = vec![CpuOperation {
        pc: 0x8000_0000_1234_5678,
        next_pc: 0x8000_0000_1234_567C,
        op_add: true,
        ..Default::default()
    }];

    let trace = generate_cpu_trace(&ops);
    let row0 = trace.main_table.get_row(0);

    // Check 64-bit PC is split correctly
    assert_eq!(row0[cols::PC_0], FE::from(0x1234_5678u64));
    assert_eq!(row0[cols::PC_1], FE::from(0x8000_0000u64));
    assert_eq!(row0[cols::NEXT_PC_0], FE::from(0x1234_567Cu64));
    assert_eq!(row0[cols::NEXT_PC_1], FE::from(0x8000_0000u64));
}

#[test]
fn test_trace_generation_rv1_dwordwhh() {
    let ops = vec![CpuOperation {
        rv1: 0xFFFF_EEEE_DDDD_CCCCu64,
        op_add: true,
        ..Default::default()
    }];

    let trace = generate_cpu_trace(&ops);
    let row0 = trace.main_table.get_row(0);

    // rv1 stored as DWordWHH: [Word, Half, Half]
    assert_eq!(row0[cols::RV1_0], FE::from(0xDDDD_CCCCu64)); // bits 0-31
    assert_eq!(row0[cols::RV1_1], FE::from(0xEEEEu64)); // bits 32-47
    assert_eq!(row0[cols::RV1_2], FE::from(0xFFFFu64)); // bits 48-63
}

#[test]
fn test_trace_generation_arg1_dwordbl() {
    let ops = vec![CpuOperation {
        rv1: 0x0807_0605_0403_0201u64,
        word_instr: false,
        op_add: true,
        ..Default::default()
    }];

    let trace = generate_cpu_trace(&ops);
    let row0 = trace.main_table.get_row(0);

    // arg1 stored as DWordBL: 8 bytes
    assert_eq!(row0[cols::ARG1_0], FE::from(0x01u64));
    assert_eq!(row0[cols::ARG1_1], FE::from(0x02u64));
    assert_eq!(row0[cols::ARG1_2], FE::from(0x03u64));
    assert_eq!(row0[cols::ARG1_3], FE::from(0x04u64));
    assert_eq!(row0[cols::ARG1_4], FE::from(0x05u64));
    assert_eq!(row0[cols::ARG1_5], FE::from(0x06u64));
    assert_eq!(row0[cols::ARG1_6], FE::from(0x07u64));
    assert_eq!(row0[cols::ARG1_7], FE::from(0x08u64));
}

#[test]
fn test_trace_generation_res_dwordbl() {
    let ops = vec![CpuOperation {
        res: 0xFEDC_BA98_7654_3210u64,
        op_add: true,
        ..Default::default()
    }];

    let trace = generate_cpu_trace(&ops);
    let row0 = trace.main_table.get_row(0);

    // res stored as DWordBL: 8 bytes
    assert_eq!(row0[cols::RES_0], FE::from(0x10u64));
    assert_eq!(row0[cols::RES_1], FE::from(0x32u64));
    assert_eq!(row0[cols::RES_2], FE::from(0x54u64));
    assert_eq!(row0[cols::RES_3], FE::from(0x76u64));
    assert_eq!(row0[cols::RES_4], FE::from(0x98u64));
    assert_eq!(row0[cols::RES_5], FE::from(0xBAu64));
    assert_eq!(row0[cols::RES_6], FE::from(0xDCu64));
    assert_eq!(row0[cols::RES_7], FE::from(0xFEu64));
}

#[test]
fn test_trace_generation_sign_bits() {
    let ops = vec![CpuOperation {
        rv1: 0x0000_0000_8000_0000u64, // bit 31 set
        res: 0x0000_0000_8000_0000u64, // bit 31 set
        word_instr: true,
        op_add: true,
        ..Default::default()
    }];

    let trace = generate_cpu_trace(&ops);
    let row0 = trace.main_table.get_row(0);

    assert_eq!(row0[cols::RV1_SIGN_BIT], FE::one());
    assert_eq!(row0[cols::RES_SIGN_BIT], FE::one());
}

#[test]
fn test_trace_generation_padding() {
    // 3 operations should pad to 4 rows
    let ops = vec![
        CpuOperation {
            pc: 0x1000,
            op_add: true,
            ..Default::default()
        },
        CpuOperation {
            pc: 0x1004,
            op_add: true,
            ..Default::default()
        },
        CpuOperation {
            pc: 0x1008,
            op_add: true,
            ..Default::default()
        },
    ];

    let trace = generate_cpu_trace(&ops);
    assert_eq!(trace.main_table.height, 4);

    // Check padding row is zeros
    let row3 = trace.main_table.get_row(3);
    assert_eq!(row3[cols::PC_0], FE::zero());
    assert_eq!(row3[cols::ADD], FE::zero());
}

#[test]
fn test_bus_interactions_count() {
    let interactions = bus_interactions();

    // Expected interactions:
    // - 8 AND_BYTE
    // - 8 OR_BYTE
    // - 8 XOR_BYTE
    // Total: 8 + 8 + 8 = 24
    // Note: LT interaction is TODO (needs DWordHHW packing)
    // Note: IS_BYTE, MSB8, ZERO, BRANCH are TODO for later
    assert_eq!(interactions.len(), 24);
}

#[test]
fn test_column_count() {
    assert_eq!(cols::NUM_COLUMNS, 72);
}

#[test]
fn test_column_arrays() {
    // Verify ARG1, ARG2, RES arrays are correct
    assert_eq!(cols::ARG1.len(), 8);
    assert_eq!(cols::ARG2.len(), 8);
    assert_eq!(cols::RES.len(), 8);

    // Check they're consecutive
    for i in 0..7 {
        assert_eq!(cols::ARG1[i + 1], cols::ARG1[i] + 1);
        assert_eq!(cols::ARG2[i + 1], cols::ARG2[i] + 1);
        assert_eq!(cols::RES[i + 1], cols::RES[i] + 1);
    }
}

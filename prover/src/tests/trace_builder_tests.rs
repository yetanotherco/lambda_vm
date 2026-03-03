//! Tests for the trace builder module.

use crate::tables::bitwise;
use crate::tables::cpu::cols;
use crate::tables::lt;
use crate::tables::memw;
use crate::tables::trace_builder::Traces;
use crate::tables::types::FE;
use executor::vm::instruction::decoding::{ArithOp, Comparison, Instruction};
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;

fn make_log(pc: u64, rs1_val: u64, rs2_val: u64, dst_val: u64, taken: bool, offset: i32) -> Log {
    Log {
        current_pc: pc,
        next_pc: if taken {
            (pc as i64 + offset as i64) as u64
        } else {
            pc + 4
        },
        src1_val: rs1_val,
        src2_val: rs2_val,
        dst_val,
        ecall_info: None,
    }
}

fn make_add_log(pc: u64, rs1_val: u64, rs2_val: u64, dst_val: u64) -> Log {
    make_log(pc, rs1_val, rs2_val, dst_val, false, 0)
}

fn make_slt_log(pc: u64, rs1_val: u64, rs2_val: u64, result: u64) -> Log {
    make_log(pc, rs1_val, rs2_val, result, false, 0)
}

fn make_blt_log(pc: u64, rs1_val: u64, rs2_val: u64, taken: bool) -> Log {
    make_log(pc, rs1_val, rs2_val, 0, taken, 8)
}

fn make_and_log(pc: u64, rs1_val: u64, rs2_val: u64, result: u64) -> Log {
    make_log(pc, rs1_val, rs2_val, result, false, 0)
}

/// Build instructions map for test logs
fn make_instructions(logs: &[Log], instrs: &[Instruction]) -> U64HashMap<Instruction> {
    let mut map = U64HashMap::default();
    for (log, instr) in logs.iter().zip(instrs.iter()) {
        map.insert(log.current_pc, *instr);
    }
    map
}

/// Append an ecall (halt) log+instruction to test data so Traces::from_logs succeeds.
fn append_ecall(logs: &mut Vec<Log>, instrs: &mut Vec<Instruction>) {
    let last_pc = logs.last().map(|l| l.current_pc + 4).unwrap_or(0x1000);
    logs.push(Log {
        current_pc: last_pc,
        next_pc: 0, // executor sets next_pc=0 for halt; prover overrides to pc+4
        src1_val: 0,
        src2_val: 0,
        dst_val: 0,
        ecall_info: Some(executor::vm::logs::EcallInfo::Halt),
    });
    instrs.push(Instruction::EcallEbreak);
}

#[test]
fn test_empty_logs() {
    let result = Traces::from_logs(&[], U64HashMap::default(), &Default::default());
    assert!(result.is_err(), "Empty logs should return an error");
}

#[test]
fn test_single_log() {
    // Single ecall log should work (padding handles power-of-2)
    let mut logs = vec![];
    let mut instrs = vec![];
    append_ecall(&mut logs, &mut instrs);
    let instructions = make_instructions(&logs, &instrs);
    let _traces = Traces::from_logs(&logs, instructions, &Default::default()).unwrap();
}

#[test]
fn test_power_of_two_logs() {
    let mut logs: Vec<Log> = (0..3)
        .map(|i| make_add_log(0x1000 + i * 4, i, i, i * 2))
        .collect();
    let mut instrs: Vec<Instruction> = (0..3)
        .map(|_| Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        })
        .collect();
    append_ecall(&mut logs, &mut instrs);
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions, &Default::default()).unwrap();
    assert_eq!(traces.cpus[0].main_table.height, 4);
}

#[test]
fn test_padding_to_power_of_two() {
    // 5 ops (not power of 2) should be padded to 8
    let mut logs: Vec<Log> = (0..4)
        .map(|i| make_add_log(0x1000 + i * 4, i, i, i * 2))
        .collect();
    let mut instrs: Vec<Instruction> = (0..4)
        .map(|_| Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        })
        .collect();
    append_ecall(&mut logs, &mut instrs);
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions, &Default::default()).unwrap();
    // 5 ops padded to 8
    assert_eq!(traces.cpus[0].main_table.height, 8);
}

#[test]
fn test_lt_operations_collected() {
    let mut logs = vec![
        make_slt_log(0x1000, 5, 10, 1),
        make_slt_log(0x1004, 10, 5, 0),
        make_add_log(0x1008, 1, 2, 3),
        make_blt_log(0x100c, 3, 7, true),
    ];
    let mut instrs = vec![
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::SetLessThan,
        },
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::SetLessThan,
        },
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
        Instruction::Branch {
            src1: 2,
            src2: 3,
            cond: Comparison::LessThan,
            offset: 8,
        },
    ];
    append_ecall(&mut logs, &mut instrs);
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions, &Default::default()).unwrap();

    // LT trace should have rows (2 SLT + 1 BLT = 3 ops, deduplicated)
    assert!(traces.lts[0].main_table.height >= 2);
}

#[test]
fn test_lt_deduplication() {
    let mut logs = vec![
        make_slt_log(0x1000, 5, 10, 1),
        make_slt_log(0x1004, 5, 10, 1), // duplicate
        make_slt_log(0x1008, 5, 10, 1), // duplicate
        make_add_log(0x100c, 0, 0, 0),  // padding to 4
    ];
    let mut instrs = vec![
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::SetLessThan,
        },
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::SetLessThan,
        },
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::SetLessThan,
        },
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
    ];
    append_ecall(&mut logs, &mut instrs);
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions, &Default::default()).unwrap();

    // The 3 identical SLT operations (5 < 10, signed) should be deduplicated.
    // With MEMW timestamp ordering LT ops also added, the table is larger,
    // but we can verify the SLT deduplication by finding the row with lhs=5, rhs=10.
    let mut found_slt = false;
    for row_idx in 0..traces.lts[0].main_table.height {
        let row = traces.lts[0].main_table.get_row(row_idx);
        // Check for our SLT: lhs=5, rhs=10, signed=1
        // lhs is stored as DWordHHW: [half0, half1, word2]
        // For value 5: half0=5, half1=0, word2=0
        if row[lt::cols::LHS_0] == FE::from(5u64)
            && row[lt::cols::LHS_1] == FE::from(0u64)
            && row[lt::cols::LHS_2] == FE::from(0u64)
            && row[lt::cols::RHS_0] == FE::from(10u64)
            && row[lt::cols::SIGNED] == FE::from(1u64)
        {
            // Found our SLT row - verify multiplicity is 3
            assert_eq!(row[lt::cols::MU], FE::from(3u64));
            found_slt = true;
            break;
        }
    }
    assert!(
        found_slt,
        "SLT operation (5 < 10, signed) not found in LT table"
    );
}

#[test]
fn test_bitwise_lookups_collected() {
    let mut logs = vec![
        make_and_log(0x1000, 0x12, 0x34, 0x10),
        make_add_log(0x1004, 0, 0, 0),
        make_add_log(0x1008, 0, 0, 0),
        make_add_log(0x100c, 0, 0, 0),
    ];
    let mut instrs = vec![
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::And,
        },
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
    ];
    append_ecall(&mut logs, &mut instrs);
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions, &Default::default()).unwrap();

    // Check AND multiplicity was updated for (0x12, 0x34, 0)
    let row_idx = bitwise::row_index(0x12, 0x34, 0);
    let row = traces.bitwise.main_table.get_row(row_idx);
    assert_eq!(row[bitwise::cols::MU_AND], FE::one());
}

#[test]
fn test_cpu_timestamps() {
    let mut logs = vec![
        make_add_log(0x1000, 1, 2, 3),
        make_add_log(0x1004, 4, 5, 6),
        make_add_log(0x1008, 7, 8, 9),
        make_add_log(0x100c, 10, 11, 12),
    ];
    let mut instrs: Vec<Instruction> = (0..4)
        .map(|_| Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        })
        .collect();
    append_ecall(&mut logs, &mut instrs);
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions, &Default::default()).unwrap();

    // Check timestamps are 4, 8, 12, 16 (starting from 4 to ensure old_timestamp < timestamp)
    for i in 0..4 {
        let row = traces.cpus[0].main_table.get_row(i);
        assert_eq!(row[cols::TIMESTAMP], FE::from((i * 4 + 4) as u64));
    }
}

#[test]
fn test_mixed_instructions() {
    let mut logs = vec![
        make_add_log(0x1000, 10, 20, 30),
        make_slt_log(0x1004, 5, 10, 1),
        make_and_log(0x1008, 0xFF, 0xF0, 0xF0),
        make_blt_log(0x100c, 1, 2, true),
    ];
    let mut instrs = vec![
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::SetLessThan,
        },
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::And,
        },
        Instruction::Branch {
            src1: 2,
            src2: 3,
            cond: Comparison::LessThan,
            offset: 8,
        },
    ];
    append_ecall(&mut logs, &mut instrs);
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions, &Default::default()).unwrap();

    // 5 ops (4 + ecall) padded to 8
    assert_eq!(traces.cpus[0].main_table.height, 8);
    assert_eq!(traces.bitwise.main_table.height, bitwise::NUM_ROWS);
    // 1 SLT + 1 BLT = 2 LT ops
    assert!(traces.lts[0].main_table.height >= 2);
}

// =============================================================================
// Phase 2 Tests: CPU ops → MEMW, LOAD, LT, Bitwise
// =============================================================================

#[test]
fn test_memw_generated_from_register_ops() {
    // Test that MEMW operations are generated for register reads/writes
    // ADD x1, x2, x3 reads x2 (M1), x3 (M3), writes x1 (M5)
    let mut logs = vec![
        make_add_log(0x1000, 100, 200, 300), // x2=100, x3=200, x1=300
        make_add_log(0x1004, 0, 0, 0),
        make_add_log(0x1008, 0, 0, 0),
        make_add_log(0x100c, 0, 0, 0),
    ];
    let mut instrs = vec![
        Instruction::Arith {
            dst: 1,  // x1
            src1: 2, // x2
            src2: 3, // x3
            op: ArithOp::Add,
        },
        Instruction::Arith {
            dst: 0,
            src1: 0,
            src2: 0,
            op: ArithOp::Add,
        },
        Instruction::Arith {
            dst: 0,
            src1: 0,
            src2: 0,
            op: ArithOp::Add,
        },
        Instruction::Arith {
            dst: 0,
            src1: 0,
            src2: 0,
            op: ArithOp::Add,
        },
    ];
    append_ecall(&mut logs, &mut instrs);
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions, &Default::default()).unwrap();

    // MEMW table should have register operations
    // First instruction generates: M1 (read x2), M3 (read x3), M5 (write x1)
    assert!(
        traces.memws[0].main_table.height >= 3,
        "MEMW should have at least 3 rows for register ops"
    );

    // Find the register write to x1 (address = 2 * 1 = 2, is_register = 1)
    let mut found_write = false;
    for row_idx in 0..traces.memws[0].main_table.height {
        let row = traces.memws[0].main_table.get_row(row_idx);
        // Check for register write: is_register=1, address=2 (x1), mu_write=1
        if row[memw::cols::IS_REGISTER] == FE::one()
            && row[memw::cols::BASE_ADDRESS_0] == FE::from(2u64)
            && row[memw::cols::MU_WRITE] == FE::one()
        {
            // Check value is 300 (lo32=300, hi32=0)
            assert_eq!(row[memw::cols::VALUE[0]], FE::from(300u64));
            found_write = true;
            break;
        }
    }
    assert!(found_write, "Register write to x1 not found in MEMW table");
}

// =============================================================================
// Phase 3 Tests: MEMW → LT (timestamp ordering)
// =============================================================================

#[test]
fn test_memw_generates_lt_for_timestamp_ordering() {
    // Test Phase 3: MEMW operations generate LT ops for old_timestamp < timestamp
    // Each MEMW op generates at least one LT op (C7: old_timestamp[0] < timestamp)
    let mut logs = vec![
        make_add_log(0x1000, 100, 200, 300),
        make_add_log(0x1004, 0, 0, 0),
        make_add_log(0x1008, 0, 0, 0),
        make_add_log(0x100c, 0, 0, 0),
    ];
    let mut instrs = vec![
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
        Instruction::Arith {
            dst: 0,
            src1: 0,
            src2: 0,
            op: ArithOp::Add,
        },
        Instruction::Arith {
            dst: 0,
            src1: 0,
            src2: 0,
            op: ArithOp::Add,
        },
        Instruction::Arith {
            dst: 0,
            src1: 0,
            src2: 0,
            op: ArithOp::Add,
        },
    ];
    append_ecall(&mut logs, &mut instrs);
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions, &Default::default()).unwrap();

    // LT table should have ops from MEMW timestamp ordering
    // First instruction: 3 register ops (M1, M3, M5) → at least 3 LT ops for C7
    // Each LT op checks old_timestamp < timestamp
    // For first access, old_timestamp=0, timestamp=4, so LT(0, 4) should exist

    // Find LT op with lhs=0, rhs=4 (first register read's timestamp check)
    let mut found_timestamp_lt = false;
    for row_idx in 0..traces.lts[0].main_table.height {
        let row = traces.lts[0].main_table.get_row(row_idx);
        // Check for LT(0, 4): lhs=0, rhs=4, signed=0
        if row[lt::cols::LHS_0] == FE::zero()
            && row[lt::cols::LHS_1] == FE::zero()
            && row[lt::cols::LHS_2] == FE::zero()
            && row[lt::cols::RHS_0] == FE::from(4u64)
            && row[lt::cols::RHS_1] == FE::zero()
            && row[lt::cols::RHS_2] == FE::zero()
            && row[lt::cols::SIGNED] == FE::zero()
        {
            found_timestamp_lt = true;
            break;
        }
    }
    assert!(
        found_timestamp_lt,
        "LT op for timestamp ordering (0 < 4) not found"
    );
}

// =============================================================================
// Phase 4 Tests: LT, MEMW → Bitwise lookups
// =============================================================================

#[test]
fn test_lt_generates_bitwise_lookups() {
    // Test Phase 4: LT operations generate MSB16 and IS_HALF lookups
    // Each LT op generates:
    // - 2 MSB16 lookups (for lhs[2] and rhs[2])
    // - 6 IS_HALF lookups (4 for lhs_sub_rhs, 2 for lhs[1] and rhs[1])
    let mut logs = vec![
        make_slt_log(0x1000, 0x1234, 0x5678, 1), // SLT generates LT op
        make_add_log(0x1004, 0, 0, 0),
        make_add_log(0x1008, 0, 0, 0),
        make_add_log(0x100c, 0, 0, 0),
    ];
    let mut instrs = vec![
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::SetLessThan,
        },
        Instruction::Arith {
            dst: 0,
            src1: 0,
            src2: 0,
            op: ArithOp::Add,
        },
        Instruction::Arith {
            dst: 0,
            src1: 0,
            src2: 0,
            op: ArithOp::Add,
        },
        Instruction::Arith {
            dst: 0,
            src1: 0,
            src2: 0,
            op: ArithOp::Add,
        },
    ];
    append_ecall(&mut logs, &mut instrs);
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions, &Default::default()).unwrap();

    // For SLT(0x1234, 0x5678):
    // lhs_sub_rhs = 0x1234 - 0x5678 = 0xFFFF_FFFF_FFFF_BBBC (wrapping)
    // IS_HALF lookup for lhs_sub_rhs[0] = 0xBBBC
    let lhs_sub_rhs = 0x1234u64.wrapping_sub(0x5678);
    let sub_0 = (lhs_sub_rhs & 0xFFFF) as u16; // 0xBBBC

    // Check IS_HALF multiplicity for lhs_sub_rhs[0]
    let row_idx = bitwise::row_index((sub_0 & 0xFF) as u8, (sub_0 >> 8) as u8, 0);
    let row = traces.bitwise.main_table.get_row(row_idx);
    assert_ne!(
        row[bitwise::cols::MU_IS_HALF],
        FE::zero(),
        "IS_HALF lookup for lhs_sub_rhs[0] should have non-zero multiplicity"
    );
}

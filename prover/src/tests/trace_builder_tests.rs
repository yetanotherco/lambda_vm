//! Tests for the trace builder module.

use crate::tables::bitwise;
use crate::tables::cpu::cols;
use crate::tables::lt;
use crate::tables::memw_register;
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
        next_pc: 0,   // executor sets next_pc=0 for halt; prover overrides to pc+4
        src1_val: 93, // a7 = 93 (sys_exit); ECALL has read_register1=true, rs1=17
        src2_val: 0,
        dst_val: 0,
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
            // Found our SLT row - verify multiplicity is 3. Every LT lookup
            // (including SLT) goes through the unified ALU bus and
            // is counted in the single `MU` column.
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

    // AND/OR/XOR now go through the BYTEWISE chip on the unified BYTE_ALU bus,
    // so the AND byte (0x12, 0x34) increments MU_BYTE_ALU_AND.
    let row_idx = bitwise::row_index(0x12, 0x34, 0);
    let row = traces.bitwise.main_table.get_row(row_idx);
    assert_eq!(row[bitwise::cols::MU_BYTE_ALU_AND], FE::one());
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

    // Check timestamps are 4, 8, 12, 16 (starting from 4 so inline PC prev_ts = 1 on first row,
    // matching REGISTER init at timestamp 1 per spec/memory.typ).
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

    // Register ops should route to MEMW_R (memw_registers), not MEMW_A.
    // First instruction generates: M1 (read x2), M3 (read x3), M5 (write x1).
    assert!(
        !traces.memw_registers.is_empty(),
        "MEMW_R should have at least one chunk for register ops"
    );
    assert!(
        traces.memw_registers[0].main_table.height >= 3,
        "MEMW_R should have at least 3 rows for register ops (reads x2, x3 + write x1)"
    );

    // Find the register write to x1 in MEMW_R.
    // MEMW_R columns: ADDRESS = register_index (x1 → index 1),
    //                 MU_WRITE = 1 for writes, VAL_0 = value low 32 bits.
    let mut found_write = false;
    for row_idx in 0..traces.memw_registers[0].main_table.height {
        let row = traces.memw_registers[0].main_table.get_row(row_idx);
        // ADDRESS = 1 (x1), MU_WRITE = 1, VAL_0 = 300
        if row[memw_register::cols::ADDRESS] == FE::from(1u64)
            && row[memw_register::cols::MU_WRITE] == FE::one()
        {
            assert_eq!(
                row[memw_register::cols::VAL_0],
                FE::from(300u64),
                "Write value for x1 should be 300"
            );
            found_write = true;
            break;
        }
    }
    assert!(
        found_write,
        "Register write to x1 (ADDRESS=1, MU_WRITE=1, VAL_0=300) not found in MEMW_R"
    );
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

    // Register ops route to MEMW_R (IS_HALFWORD, not LT).
    assert!(
        !traces.memw_registers.is_empty(),
        "Register ops should route to MEMW_R"
    );

    // Register ops use IS_HALF for timestamp ordering instead of LT.
    // Verify the bitwise table has at least one IS_HALF entry with non-zero
    // multiplicity, proving that MEMW_R's IS_HALF lookups were emitted.
    let has_is_half_entry = (0..traces.bitwise.main_table.height)
        .any(|i| traces.bitwise.main_table.get_row(i)[bitwise::cols::MU_IS_HALF] != FE::zero());
    assert!(
        has_is_half_entry,
        "MEMW_R register ops should produce IS_HALF bitwise entries"
    );

    // The LT table should still have ops from non-register MEMW accesses
    // (e.g. PC next-pc write is a non-register memory op that needs LT).
    let total_lt_rows: usize = traces.lts.iter().map(|t| t.main_table.height).sum();
    assert!(
        total_lt_rows > 0,
        "LT table should have ops from non-register MEMW timestamp ordering"
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

mod keccak_tests {
    use crate::tables::bitwise::BitwiseOperationType;
    use crate::tables::keccak::cols as core_cols;
    use crate::tables::keccak::{self, KeccakOperation};
    use crate::tables::keccak_rc;
    use crate::tables::keccak_rnd::cols as rnd_cols;
    use crate::tables::keccak_rnd::{self, KeccakRoundOperation};
    use crate::tables::trace_builder::*;
    use crate::tables::types::FE;
    use executor::vm::instruction::execution::keccak_f1600;

    fn make_keccak_ops() -> (KeccakOperation, KeccakRoundOperation) {
        let input = [0u64; 25];
        let mut output = input;
        keccak_f1600(&mut output);
        let kop = KeccakOperation {
            timestamp: 42,
            state_addr: 0x1000,
            input,
            output,
        };
        let rop = KeccakRoundOperation {
            timestamp: 42,
            input,
            output,
        };
        (kop, rop)
    }

    #[test]
    fn test_keccak_bitwise_ops_count() {
        let (kop, _) = make_keccak_ops();
        let ops = collect_bitwise_from_keccak(&[kop]);

        let xor = ops
            .iter()
            .filter(|o| o.lookup_type == BitwiseOperationType::ByteAluXor)
            .count();
        let and = ops
            .iter()
            .filter(|o| o.lookup_type == BitwiseOperationType::ByteAluAnd)
            .count();
        let are_bytes = ops
            .iter()
            .filter(|o| o.lookup_type == BitwiseOperationType::AreBytes)
            .count();
        let hwsl = ops
            .iter()
            .filter(|o| o.lookup_type == BitwiseOperationType::Hwsl)
            .count();
        let is_half = ops
            .iter()
            .filter(|o| o.lookup_type == BitwiseOperationType::IsHalf)
            .count();

        assert_eq!(xor, 24 * 608, "ByteAluXor count");
        assert_eq!(and, 24 * 200 + 1, "ByteAluAnd count");
        // Cxz_right Byte→Bit (spec d75944ee): drops 40 ARE_BYTES per round.
        // Spec emits one IS_BYTE template per byte; ops pair adjacent bytes
        // into ARE_BYTES (20 cxz_left + 200 rho per round, 4 addr per call).
        assert_eq!(are_bytes, 24 * 220 + 4, "AreBytes count");
        // θ/ρ halfword shifts are enforced by inline μ-gated identities on the
        // keccak_rnd chip, so no HWSL lookups are emitted (was 24 * 120).
        assert_eq!(hwsl, 0, "Hwsl count");
        assert_eq!(is_half, 100, "IsHalf count");
        assert_eq!(ops.len(), 105 + 24 * 1028, "Total bitwise ops");
    }

    #[test]
    fn test_keccak_round_trace_matches_f1600() {
        let (_, rop) = make_keccak_ops();
        let rnd_trace = keccak_rnd::generate_keccak_rnd_trace(&[rop]);

        let mut ref_state = [0u64; 25];
        for round in 0..24 {
            let rc = executor::vm::instruction::execution::KECCAK_RC[round];
            let mut c = [0u64; 5];
            for x in 0..5 {
                c[x] = ref_state[x]
                    ^ ref_state[x + 5]
                    ^ ref_state[x + 10]
                    ^ ref_state[x + 15]
                    ^ ref_state[x + 20];
            }
            let mut d = [0u64; 5];
            for x in 0..5 {
                d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            }
            for i in 0..25 {
                ref_state[i] ^= d[i % 5];
            }
            let mut b = [0u64; 25];
            for x in 0..5 {
                for y in 0..5 {
                    b[y + 5 * ((2 * x + 3 * y) % 5)] = ref_state[x + 5 * y]
                        .rotate_left(executor::vm::instruction::execution::KECCAK_RHO[x][y]);
                }
            }
            for x in 0..5 {
                for y in 0..5 {
                    ref_state[x + 5 * y] =
                        b[x + 5 * y] ^ (!b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
                }
            }
            ref_state[0] ^= rc;

            for (lane, &lane_val) in ref_state.iter().enumerate() {
                let x = lane % 5;
                let y = lane / 5;
                for byte_idx in 0..8 {
                    let expected = FE::from((lane_val >> (byte_idx * 8)) & 0xFF);
                    let col = if x == 0 && y == 0 {
                        rnd_cols::iota(byte_idx)
                    } else {
                        rnd_cols::chi(x, y, byte_idx)
                    };
                    let trace_val = rnd_trace.get_main(round, col);
                    assert_eq!(
                        &expected, trace_val,
                        "Round {round} lane ({x},{y}) byte {byte_idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_keccak_core_round_state_consistency() {
        let (kop, rop) = make_keccak_ops();
        let core_trace = keccak::generate_keccak_trace(&[kop]);
        let rnd_trace = keccak_rnd::generate_keccak_rnd_trace(&[rop]);

        // Round 0 start == core input_state
        for x in 0..5 {
            for y in 0..5 {
                for b in 0..8 {
                    let core_val = core_trace.get_main(0, core_cols::input_state(x, y, b));
                    let rnd_val = rnd_trace.get_main(0, rnd_cols::start(x, y, b));
                    assert_eq!(core_val, rnd_val, "Round 0 start mismatch at ({x},{y},{b})");
                }
            }
        }

        // Round 23 out == core output_state
        for x in 0..5 {
            for y in 0..5 {
                for b in 0..8 {
                    let core_val = core_trace.get_main(0, core_cols::output_state(x, y, b));
                    let rnd_val = if x == 0 && y == 0 {
                        rnd_trace.get_main(23, rnd_cols::iota(b))
                    } else {
                        rnd_trace.get_main(23, rnd_cols::chi(x, y, b))
                    };
                    assert_eq!(core_val, rnd_val, "Round 23 out mismatch at ({x},{y},{b})");
                }
            }
        }
    }

    #[test]
    fn test_keccak_bus_interaction_counts() {
        assert_eq!(
            keccak::bus_interactions().len(),
            134,
            "KECCAK core: 1 ECALL + 1 MEMW read_addr + 25 MEMW lanes + 100 IS_HALF + 1 BYTE_ALU alignment + 4 ARE_BYTES addr pairs + 1 Keccak send + 1 Keccak recv"
        );
        assert_eq!(
            keccak_rnd::bus_interactions().len(),
            1031,
            "KECCAK_RND: 3 IO + 420 theta + 200 rho + 400 chi + 8 iota \
             (θ/ρ HWSL sends replaced by inline μ-gated shift identities: −20 θ, −100 ρ; \
             Cxz_right Byte→Bit drops 40 ARE_BYTES per spec d75944ee; \
             ARE_BYTES sends are paired per spec ARE_BYTES interaction signature)"
        );
        assert_eq!(
            keccak_rc::bus_interactions().len(),
            1,
            "KECCAK_RC: 1 receiver"
        );
    }

    #[test]
    fn test_keccak_column_counts() {
        assert_eq!(core_cols::NUM_COLUMNS, 511, "KECCAK core columns");
        assert_eq!(
            rnd_cols::NUM_COLUMNS,
            1480,
            "KECCAK_RND columns (rnc/rbc inlined; pi virtual; Cxz_right Bit-typed)"
        );
        assert_eq!(keccak_rc::cols::NUM_COLUMNS, 10, "KECCAK_RC columns");
    }

    #[test]
    fn test_keccak_constraint_counts() {
        use stark::constraints::builder::ConstraintSet;
        assert_eq!(
            keccak::KeccakConstraints.meta().len(),
            51,
            "KECCAK core: 25 ADD pairs + no-overflow"
        );
        assert_eq!(
            keccak_rnd::KeccakRndConstraints.meta().len(),
            140,
            "KECCAK_RND: 20 IS_BIT(μ; Cxz_right_bit) + 20 θ + 100 ρ inline shift identities"
        );
    }
}

mod routing_tests {
    use crate::tables::memw::MemwOperation;
    use crate::tables::trace_builder::*;

    fn make_register_op(timestamp: u64, old_timestamp: u64) -> MemwOperation {
        MemwOperation::new(true, 2, [1, 0, 0, 0, 0, 0, 0, 0], timestamp, 2, false)
            .with_old([0; 8], [old_timestamp, old_timestamp, 0, 0, 0, 0, 0, 0])
    }

    #[test]
    fn test_is_register_op_delta_at_boundary_routes_in() {
        // delta = 0x10000 = 2^16: spec allows this (IS_HALF[0xFFFF] is valid)
        let op = make_register_op(0x10000, 0);
        assert!(is_register_op(&op), "delta = 2^16 should route to MEMW_R");
    }

    #[test]
    fn test_is_register_op_delta_above_boundary_falls_back() {
        // delta = 0x10001: one above the IS_HALF range, must fall back to MEMW_A
        let op = make_register_op(0x10001, 0);
        assert!(
            !is_register_op(&op),
            "delta = 2^16 + 1 should fall back to MEMW_A"
        );
    }

    #[test]
    fn test_is_register_op_delta_one_routes_in() {
        // delta = 1: minimum allowed value
        let op = make_register_op(1, 0);
        assert!(is_register_op(&op), "delta = 1 should route to MEMW_R");
    }

    #[test]
    fn test_is_register_op_delta_zero_falls_back() {
        // delta = 0: ts[0] not strictly greater than old_ts[0]
        let op = make_register_op(5, 5);
        assert!(!is_register_op(&op), "delta = 0 should not route to MEMW_R");
    }

    #[test]
    fn test_is_register_op_upper_limb_mismatch_falls_back() {
        // ts_hi != old_ts_hi: shared upper limb assumption violated
        let op = make_register_op(0x1_0000_0001, 0x0_0000_0000);
        assert!(
            !is_register_op(&op),
            "different upper limbs should fall back to MEMW_A"
        );
    }
}

/// `from_image_and_logs` is a faithful generalization of `from_elf_and_logs`:
/// fed the ELF-derived image, it must produce identical traces.
#[test]
fn test_from_image_and_logs_matches_from_elf_and_logs() {
    use crate::tables::MaxRowsConfig;
    use crate::tables::trace_builder::build_initial_image;
    use crate::test_utils::asm_elf_bytes;
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let elf_bytes = asm_elf_bytes("basic_program");
    let program = Elf::load(&elf_bytes).unwrap();
    let logs = Executor::new(&program, vec![]).unwrap().run().unwrap().logs;
    let max_rows = MaxRowsConfig::default();

    let from_elf = Traces::from_elf_and_logs(
        &program,
        &logs,
        &max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .unwrap();

    let image = build_initial_image(&program, &[]);
    let register_init =
        crate::tables::register::register_init_from_entry_point(program.entry_point);
    let from_image = Traces::from_image_and_logs(
        &program,
        &image,
        &register_init,
        &logs,
        &max_rows,
        &[],
        true,
        false,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .unwrap();

    assert_eq!(
        from_elf.total_field_elements(),
        from_image.total_field_elements()
    );
    assert_eq!(
        format!("{:?}", from_elf.table_counts()),
        format!("{:?}", from_image.table_counts())
    );
}

/// A memory snapshot at an epoch boundary converts into a non-empty initial
/// image (the input `from_image_and_logs` consumes for the next epoch).
#[test]
fn test_epoch_end_memory_converts_to_image() {
    use crate::test_utils::asm_elf_bytes;
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use std::collections::HashMap;

    let elf_bytes = asm_elf_bytes("basic_program");
    let program = Elf::load(&elf_bytes).unwrap();

    let total = Executor::new(&program, vec![])
        .unwrap()
        .run()
        .unwrap()
        .logs
        .len();
    let epoch_size = (total / 3).max(1);
    let epochs = Executor::new(&program, vec![])
        .unwrap()
        .run_epochs(epoch_size)
        .unwrap();
    assert!(epochs.len() >= 2);

    let image: HashMap<u64, u8> = epochs[0].end_memory.iter_bytes().collect();
    assert!(!image.is_empty());
}

/// Every epoch builds traces: intermediate epochs (`is_final = false`) skip HALT
/// and start from the previous epoch's memory; the last epoch terminates.
#[test]
fn test_build_traces_for_all_epochs() {
    use crate::tables::MaxRowsConfig;
    use crate::tables::trace_builder::build_initial_image;
    use crate::test_utils::asm_elf_bytes;
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use std::collections::HashMap;

    let elf_bytes = asm_elf_bytes("basic_program");
    let program = Elf::load(&elf_bytes).unwrap();

    let total = Executor::new(&program, vec![])
        .unwrap()
        .run()
        .unwrap()
        .logs
        .len();
    let epoch_size = (total / 3).max(1);
    let epochs = Executor::new(&program, vec![])
        .unwrap()
        .run_epochs(epoch_size)
        .unwrap();
    assert!(epochs.len() >= 2);

    let max_rows = MaxRowsConfig::default();
    let last = epochs.len() - 1;

    for (i, epoch) in epochs.iter().enumerate() {
        // Epoch 0 starts from the program-start image; later epochs from the
        // previous epoch's ending memory + register snapshot.
        let (image, register_init): (HashMap<u64, u8>, Vec<u32>) = if i == 0 {
            (
                build_initial_image(&program, &[]),
                crate::tables::register::register_init_from_entry_point(program.entry_point),
            )
        } else {
            (
                epochs[i - 1].end_memory.iter_bytes().collect(),
                crate::tables::register::register_init_from_snapshot(
                    &epochs[i - 1].end_registers,
                    epochs[i - 1].end_pc,
                ),
            )
        };

        let traces = Traces::from_image_and_logs(
            &program,
            &image,
            &register_init,
            &epoch.logs,
            &max_rows,
            &[],
            i == last,
            false,
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        )
        .unwrap_or_else(|e| panic!("epoch {i} (is_final={}) failed to build: {e:?}", i == last));

        assert!(
            traces.table_counts().cpu > 0,
            "epoch {i} produced an empty CPU trace"
        );
    }
}

/// A non-final epoch carrying the program-terminating instruction is rejected
/// (rather than silently producing an unverifiable proof).
#[test]
fn test_terminating_epoch_rejected_when_not_final() {
    use crate::tables::MaxRowsConfig;
    use crate::tables::register::register_init_from_snapshot;
    use crate::test_utils::asm_elf_bytes;
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use std::collections::HashMap;

    let elf_bytes = asm_elf_bytes("basic_program");
    let program = Elf::load(&elf_bytes).unwrap();

    let total = Executor::new(&program, vec![])
        .unwrap()
        .run()
        .unwrap()
        .logs
        .len();
    let epoch_size = (total / 3).max(1);
    let epochs = Executor::new(&program, vec![])
        .unwrap()
        .run_epochs(epoch_size)
        .unwrap();
    assert!(epochs.len() >= 2);

    // The last epoch holds the terminating instruction; building it as a
    // non-final epoch (is_final = false) must error.
    let last = epochs.len() - 1;
    let image: HashMap<u64, u8> = epochs[last - 1].end_memory.iter_bytes().collect();
    let register_init =
        register_init_from_snapshot(&epochs[last - 1].end_registers, epochs[last - 1].end_pc);

    let result = Traces::from_image_and_logs(
        &program,
        &image,
        &register_init,
        &epochs[last].logs,
        &MaxRowsConfig::default(),
        &[],
        false,
        false,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    );

    assert!(
        matches!(result, Err(crate::Error::HaltInNonFinalEpoch)),
        "expected HaltInNonFinalEpoch error for a non-final terminating epoch"
    );
}

/// End to end: extract real per-epoch touched cells from execution, feed them
/// through the local-to-global boundary logic, and render each epoch's trace.
#[test]
fn test_local_to_global_traces_from_real_execution() {
    use crate::tables::local_to_global::{epoch_boundaries, generate_local_to_global_trace};
    use crate::tables::trace_builder::{build_initial_image, epoch_touched_cells};
    use crate::test_utils::asm_elf_bytes;
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use std::collections::HashMap;

    // A program that exercises memory (loads/stores), so some cells are touched.
    let elf_bytes = asm_elf_bytes("all_loadstore_32");
    let program = Elf::load(&elf_bytes).unwrap();

    let total = Executor::new(&program, vec![])
        .unwrap()
        .run()
        .unwrap()
        .logs
        .len();
    let epoch_size = (total / 3).max(1);
    let epochs = Executor::new(&program, vec![])
        .unwrap()
        .run_epochs(epoch_size)
        .unwrap();
    assert!(epochs.len() >= 2);

    let elf_image = build_initial_image(&program, &[]);
    let total_memory = elf_image.len();

    // Per-epoch touched cells from real execution (epoch 0 from the ELF image,
    // later epochs from the previous epoch's ending memory).
    let mut per_epoch_touches: Vec<Vec<(u64, u64, u64)>> = Vec::new();
    for (i, epoch) in epochs.iter().enumerate() {
        let image: HashMap<u64, u8> = if i == 0 {
            elf_image.clone()
        } else {
            epochs[i - 1].end_memory.iter_bytes().collect()
        };
        let register_init = if i == 0 {
            crate::tables::register::register_init_from_entry_point(program.entry_point)
        } else {
            crate::tables::register::register_init_from_snapshot(
                &epochs[i - 1].end_registers,
                epochs[i - 1].end_pc,
            )
        };
        per_epoch_touches
            .push(epoch_touched_cells(&program, &image, &register_init, &epoch.logs).unwrap());
    }

    // The program touches memory somewhere, and every per-epoch touched set is
    // sparse (far smaller than the whole memory image).
    let total_touched: usize = per_epoch_touches.iter().map(Vec::len).sum();
    assert!(total_touched > 0);
    for touched in &per_epoch_touches {
        assert!(touched.len() < total_memory);
    }

    // Boundary claims + rendered L2G trace per epoch.
    let initial_memory: HashMap<u64, u64> =
        elf_image.iter().map(|(&a, &v)| (a, v as u64)).collect();
    let boundaries = epoch_boundaries(&initial_memory, &per_epoch_touches);

    for (i, boundary_set) in boundaries.iter().enumerate() {
        let trace = generate_local_to_global_trace(boundary_set);
        let expected_rows = per_epoch_touches[i].len().next_power_of_two().max(1);
        assert_eq!(trace.num_rows(), expected_rows);
    }
}

/// Two builds of the same logs must produce byte-identical traces.
///
/// The dedup'd tables (LT, MUL, DVRM, BRANCH, EQ, BYTEWISE) collect their rows
/// out of a `HashMap`, whose iteration order std randomizes per instance — so
/// the rows were identical in content but arbitrary in order. Harmless while a
/// trace is built once, fatal for rebuilding a retired one: the rebuild has to
/// hash to the root the first build committed.
///
/// Fails if any of the `sort_unstable_by` calls after those dedups is removed.
#[test]
fn trace_build_is_deterministic_across_builds() {
    type TT = stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >;

    // Several DISTINCT ops per table, so each dedup'd `unique_ops` holds more
    // than one element and its order can actually vary.
    let mut logs = vec![
        make_slt_log(0x1000, 5, 10, 1),
        make_slt_log(0x1004, 200, 7, 0),
        make_slt_log(0x1008, 42, 42, 0),
        make_slt_log(0x100c, 1, 999, 1),
        make_blt_log(0x1010, 3, 4, true),
        make_blt_log(0x1014, 50, 9, false),
        make_blt_log(0x1018, 77, 77, false),
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
            op: ArithOp::SetLessThan,
        },
        Instruction::Branch {
            src1: 2,
            src2: 3,
            cond: Comparison::LessThan,
            offset: 8,
        },
        Instruction::Branch {
            src1: 2,
            src2: 3,
            cond: Comparison::LessThan,
            offset: 8,
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
    let max_rows = Default::default();

    let a = Traces::from_logs(&logs, instructions.clone(), &max_rows).unwrap();
    let b = Traces::from_logs(&logs, instructions, &max_rows).unwrap();

    fn flat(t: &TT) -> Vec<u64> {
        let (data, _cols) = t.main_data_row_major();
        data.iter().map(|fe| *fe.value()).collect()
    }
    fn eq_chunks(x: &[TT], y: &[TT], name: &str) {
        assert_eq!(
            x.len(),
            y.len(),
            "{name}: chunk count differs across builds"
        );
        for (i, (s, m)) in x.iter().zip(y.iter()).enumerate() {
            assert_eq!(
                flat(s),
                flat(m),
                "{name} chunk {i}: trace data differs across builds (non-deterministic order)"
            );
        }
    }
    eq_chunks(&a.lts, &b.lts, "LT");
    eq_chunks(&a.muls, &b.muls, "MUL");
    eq_chunks(&a.dvrms, &b.dvrms, "DVRM");
    eq_chunks(&a.branches, &b.branches, "BRANCH");
    eq_chunks(&a.eqs, &b.eqs, "EQ");
    eq_chunks(&a.bytewises, &b.bytewises, "BYTEWISE");
    eq_chunks(&a.cpus, &b.cpus, "CPU");
    eq_chunks(&a.memws, &b.memws, "MEMW");
    eq_chunks(&a.shifts, &b.shifts, "SHIFT");
    eq_chunks(&a.loads, &b.loads, "LOAD");
}

/// `build_chunk(kind, i)` must equal `build_table(kind)[i]`, byte for byte.
///
/// This is the equality the streaming prover rests on: Round 1 commits the
/// table built one way, and the fused chain rebuilds the chunk it needs the
/// other way. If they ever diverge, the rebuilt trace hashes to a root the
/// verifier will not accept.
#[test]
fn build_chunk_matches_the_full_table_build() {
    use crate::tables::trace_builder::{CollectedOps, TableKind};

    // More ops than the chunk limit below, so several chunks exist and the
    // last one is short.
    let lt_ops: Vec<_> = (0..10u64)
        .map(|i| crate::tables::lt::LtOperation::new(i, i * 7 + 1, false))
        .collect();
    let routed = CollectedOps {
        lt_ops,
        ..Default::default()
    };

    let max_rows = crate::tables::MaxRowsConfig {
        lt: 4,
        ..Default::default()
    };

    let whole = routed
        .build_table(
            TableKind::Lt,
            &max_rows,
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        )
        .expect("full build");
    assert_eq!(
        whole.len(),
        routed.num_chunks(TableKind::Lt, &max_rows),
        "num_chunks disagrees with what build_table produced"
    );

    for (i, expected) in whole.iter().enumerate() {
        let one = routed.build_chunk(TableKind::Lt, i, &max_rows);
        let (a, _) = expected.main_data_row_major();
        let (b, _) = one.main_data_row_major();
        assert_eq!(
            a.iter().map(|fe| *fe.value()).collect::<Vec<_>>(),
            b.iter().map(|fe| *fe.value()).collect::<Vec<_>>(),
            "chunk {i}: on-demand build differs from the full build"
        );
    }
}

/// `chunk_shape` must agree with the chunk it declines to build, for every kind.
///
/// It reads the shape off op counts and a constant width instead of generating
/// a trace, which is only valid while every generator pads to
/// `count.next_power_of_two().max(4)`. This is the test that fails if one of
/// them ever stops.
#[test]
fn chunk_shape_matches_the_built_chunk() {
    use crate::tables::trace_builder::{CollectedOps, TableKind};

    // Ops with deliberate repeats, so the deduplicating kinds and the plain ones
    // disagree on count and the distinction is actually exercised.
    // 8 ops, 3 distinct: the deduplicating rule pads to 4 rows where counting
    // them raw would pad to 8. Without that gap the test would pass even if
    // `chunk_shape` ignored deduplication entirely.
    let lt_ops: Vec<_> = (0..8u64)
        .map(|i| crate::tables::lt::LtOperation::new(i % 3, i % 3 + 1, false))
        .collect();
    // SHIFT is a plain kind (no dedup): 20 ops over a limit of 8 give chunks of
    // 8/8/4, i.e. row counts of 8/8/4. Sizes above 4 matter — an empty op list
    // pads to 4, so a fixture whose chunks all land on 4 compares the padding
    // floor against itself and would pass even if the row rule were wrong.
    let shift_ops: Vec<_> = (0..20u64)
        .map(|i| crate::tables::shift::ShiftOperation::new(i, i % 5, false, false, false))
        .collect();
    // MUL deduplicates like LT, and its rows are keyed on the op with the lo/hi
    // flag folded into the multiplicity: 12 entries, 3 distinct.
    let mul_ops: Vec<_> = (0..6u64)
        .flat_map(|i| {
            let op = crate::tables::mul::MulOperation::new(i % 3, false, i % 3 + 1, false);
            [(op.clone(), false), (op, true)]
        })
        .collect();
    let routed = CollectedOps {
        lt_ops,
        shift_ops,
        mul_ops,
        ..Default::default()
    };

    let max_rows = crate::tables::MaxRowsConfig {
        lt: 16,
        shift: 8,
        mul: 16,
        ..Default::default()
    };

    assert_eq!(
        routed.chunk_shape(TableKind::Lt, 0, &max_rows).0,
        4,
        "the LT fixture must exercise deduplication (8 ops, 3 distinct)"
    );
    assert_eq!(
        routed.chunk_shape(TableKind::Mul, 0, &max_rows).0,
        4,
        "the MUL fixture must exercise deduplication (12 entries, 3 distinct)"
    );
    assert_eq!(
        routed.num_chunks(TableKind::Shift, &max_rows),
        3,
        "the SHIFT fixture must split into several chunks, or the plain path is \
         only ever checked on one"
    );
    assert_eq!(
        routed.chunk_shape(TableKind::Shift, 0, &max_rows).0,
        8,
        "the SHIFT fixture must produce chunks wider than the 4-row padding floor"
    );

    let mut populated = 0usize;
    for kind in [
        TableKind::Cpu,
        TableKind::Memw,
        TableKind::MemwAligned,
        TableKind::MemwRegister,
        TableKind::Load,
        TableKind::Lt,
        TableKind::Shift,
        TableKind::Mul,
        TableKind::Dvrm,
        TableKind::Branch,
        TableKind::Eq,
        TableKind::Bytewise,
        TableKind::Store,
        TableKind::Cpu32,
    ] {
        for chunk in 0..routed.num_chunks(kind, &max_rows) {
            let built = routed.build_chunk(kind, chunk, &max_rows);
            assert_eq!(
                routed.chunk_shape(kind, chunk, &max_rows),
                (built.num_rows(), built.num_main_columns),
                "{kind:?} chunk {chunk}: declared shape differs from the built one"
            );
            if routed.buffered(kind) > 0 {
                populated += 1;
            }
        }
    }
    // A kind with no ops is left out of the layout entirely, so the loop above
    // only visits chunks that exist and every comparison it makes is over real
    // rows. CPU and MEMW_R are the exceptions — they keep their padded chunk —
    // but a fixture that populated neither would leave nothing to compare.
    assert!(
        populated >= 3,
        "the fixture must give several kinds real ops, or the row half of the \
         comparison is vacuous (populated chunks: {populated})"
    );
}

/// Collecting an execution chunk by chunk must produce exactly what collecting
/// it all at once produces.
///
/// This is what lets the prover walk an execution instead of starting from a
/// materialized log of the whole thing. It holds because phases 1-3 are
/// segment-local once `MemoryState` and `RegisterState` are carried: the LT ops
/// a memory access implies come from the timestamps that access already
/// carries, not from an ordering over the whole run.
///
/// Compared through the built traces rather than the op lists, since that is
/// what the commitment is taken over.
#[test]
fn collect_streaming_matches_collect_epoch() {
    use crate::tables::register::register_init_from_entry_point;
    use crate::tables::trace_builder::{DecodeArtifacts, Traces as T, build_initial_image};
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let elf_bytes = crate::test_utils::asm_elf_bytes("fib_iterative_160k");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    let image = build_initial_image(&elf, &[]);
    let register_init = register_init_from_entry_point(elf.entry_point);
    let max_rows = crate::tables::MaxRowsConfig::default();

    let logs = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("run")
        .logs;
    let at_once =
        T::collect_epoch(&artifacts, &image, &register_init, &logs, true).expect("collect at once");
    let streamed =
        T::collect_epoch_streaming(&artifacts, &elf, vec![], &image, &register_init, true)
            .expect("collect streaming");

    let build = |collected| {
        T::build_from_collected(
            &artifacts,
            collected,
            Some(&image),
            &register_init,
            &max_rows,
            &[],
            true,
            false,
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        )
        .expect("build")
    };
    let a = build(at_once);
    let b = build(streamed);

    let flat = |t: &stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >| {
        let (data, _) = t.main_data_row_major();
        data.iter().map(|fe| *fe.value()).collect::<Vec<_>>()
    };
    let same = |x: &[_], y: &[_], name: &str| {
        assert_eq!(x.len(), y.len(), "{name}: chunk count differs");
        for (i, (p, q)) in x.iter().zip(y.iter()).enumerate() {
            assert_eq!(flat(p), flat(q), "{name} chunk {i} differs");
        }
    };
    same(&a.cpus, &b.cpus, "CPU");
    same(&a.memws, &b.memws, "MEMW");
    same(&a.lts, &b.lts, "LT");
    same(&a.loads, &b.loads, "LOAD");
    same(&a.shifts, &b.shifts, "SHIFT");
    same(&a.branches, &b.branches, "BRANCH");
    same(&a.memw_registers, &b.memw_registers, "MEMW_R");
    assert_eq!(flat(&a.bitwise), flat(&b.bitwise), "BITWISE differs");
    assert_eq!(flat(&a.register), flat(&b.register), "REGISTER differs");
}

/// The Commit-phase walk must hand out exactly the chunks the all-at-once build
/// produces, in the same order.
///
/// It is the same equality `build_chunk` rests on, one level up: a table closed
/// mid-execution and a table built from the finished op list have to be the
/// same table, or committing early means committing something else.
fn assert_walk_matches(fixture: &str, max_rows: crate::tables::MaxRowsConfig, min_split: usize) {
    use crate::tables::register::register_init_from_entry_point;
    use crate::tables::trace_builder::{
        DecodeArtifacts, TableKind, Traces as T, build_initial_image,
    };
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let elf_bytes = crate::test_utils::asm_elf_bytes(fixture);
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    let image = build_initial_image(&elf, &[]);
    let register_init = register_init_from_entry_point(elf.entry_point);
    // Small enough that the walk closes several chunks before the run ends,
    // which is the case that matters — a single tail chunk would prove nothing.
    let logs = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("run")
        .logs;
    let collected =
        T::collect_epoch(&artifacts, &image, &register_init, &logs, true).expect("collect");
    let expected = T::build_from_collected(
        &artifacts,
        collected,
        Some(&image),
        &register_init,
        &max_rows,
        &[],
        true,
        false,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("build");

    let mut seen: Vec<(TableKind, usize, Vec<u64>)> = Vec::new();
    T::walk_and_emit_chunks(
        &artifacts,
        &elf,
        vec![],
        &image,
        &register_init,
        &max_rows,
        |kind, chunk, table| {
            let (data, _) = table.main_data_row_major();
            seen.push((kind, chunk, data.iter().map(|fe| *fe.value()).collect()));
        },
    )
    .expect("walk");

    let flat = |t: &stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >| {
        let (data, _) = t.main_data_row_major();
        data.iter().map(|fe| *fe.value()).collect::<Vec<u64>>()
    };
    let check = |kind: TableKind,
                 want: &[stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >]| {
        let got: Vec<_> = seen.iter().filter(|(k, _, _)| *k == kind).collect();
        // The walk emits only the chunks that filled up during the run; the
        // partial tail waits for the end-of-run finalization, which still
        // appends to these lists. So the emitted chunks are a prefix.
        assert_eq!(
            got.len(),
            want.len().saturating_sub(1),
            "{kind:?}: the walk should emit every chunk but the tail"
        );
        for (i, (_, chunk, data)) in got.iter().enumerate() {
            assert_eq!(*chunk, i, "{kind:?}: chunks arrived out of order");
            assert_eq!(*data, flat(&want[i]), "{kind:?} chunk {i} differs");
        }
    };
    // Only the tables this fixture actually splits are worth asserting on: a
    // table with one chunk compares an empty prefix and proves nothing.
    let chunked: Vec<&str> = [
        ("CPU", expected.cpus.len()),
        ("MEMW", expected.memws.len()),
        ("MEMW_A", expected.memw_aligneds.len()),
        ("MEMW_R", expected.memw_registers.len()),
        ("LOAD", expected.loads.len()),
        ("BRANCH", expected.branches.len()),
        ("EQ", expected.eqs.len()),
        ("BYTEWISE", expected.bytewises.len()),
        ("STORE", expected.stores.len()),
    ]
    .into_iter()
    .filter(|(_, n)| *n > 1)
    .map(|(name, _)| name)
    .collect();
    assert!(
        chunked.len() >= min_split,
        "{fixture} must split at least {min_split} tables mid-walk, split: {chunked:?}"
    );

    check(TableKind::Cpu, &expected.cpus);
    check(TableKind::Memw, &expected.memws);
    check(TableKind::MemwAligned, &expected.memw_aligneds);
    check(TableKind::MemwRegister, &expected.memw_registers);
    check(TableKind::Load, &expected.loads);
    check(TableKind::Cpu32, &expected.cpu32s);
    check(TableKind::Branch, &expected.branches);
    check(TableKind::Eq, &expected.eqs);
    check(TableKind::Bytewise, &expected.bytewises);
    check(TableKind::Store, &expected.stores);
}

#[test]
fn commit_walk_emits_the_same_chunks() {
    // Limits small enough that several tables close chunks mid-walk.
    assert_walk_matches(
        "fib_iterative_160k",
        crate::tables::MaxRowsConfig {
            cpu: 1 << 15,
            memw: 1 << 10,
            load: 1 << 10,
            branch: 1 << 12,
            eq: 1 << 12,
            bytewise: 1 << 12,
            store: 1 << 12,
            ..Default::default()
        },
        3,
    );
}

/// The same, on a program that uses the word instructions.
///
/// `cpu32_chip_op` appends to SHIFT, MUL and DVRM for every `*W` op, so those
/// tables are not final when a segment ends. A fibonacci fixture has no word
/// instructions and would let a table that is closed too early pass unnoticed;
/// this one would not.
#[test]
fn commit_walk_emits_the_same_chunks_with_word_instructions() {
    assert_walk_matches("basic_arith_32", crate::tables::MaxRowsConfig::small(), 1);
}

/// A chunk committed during the walk must carry the root the normal prover
/// gives that same chunk.
///
/// This is what makes Approach 1's Commit phase legitimate rather than merely
/// convenient: the phase closes a table before the tables after it exist and
/// puts its commitment in the transcript then and there. If that commitment
/// differed from the one the all-at-once prover would produce, everything
/// downstream — challenges, openings, the verifier — would be reading a
/// different table.
#[test]
fn chunks_committed_during_the_walk_carry_the_normal_roots() {
    use crate::tables::register::register_init_from_entry_point;
    use crate::tables::trace_builder::{
        DecodeArtifacts, TableKind, Traces as T, build_initial_image,
    };
    use executor::elf::Elf;
    use stark::prover::IsStarkProver;

    let elf_bytes = crate::test_utils::asm_elf_bytes("fib_iterative_160k");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    let image = build_initial_image(&elf, &[]);
    let register_init = register_init_from_entry_point(elf.entry_point);
    let max_rows = crate::tables::MaxRowsConfig {
        cpu: 1 << 15,
        memw: 1 << 15,
        ..Default::default()
    };
    let proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .expect("blowup 2 is valid");

    let traces = T::from_elf_and_logs(
        &elf,
        &executor::vm::execution::Executor::new(&elf, vec![])
            .expect("executor")
            .run()
            .expect("run")
            .logs,
        &max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("traces");
    let counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        false,
        &traces.page_configs,
        &counts,
        None,
        true,
        None,
        None,
        None,
    );

    // The AIR each emitted chunk belongs to, by kind and position.
    let air_for = |kind: TableKind, chunk: usize| match kind {
        TableKind::Cpu => airs.cpus.get(chunk).map(|a| a.as_ref()),
        TableKind::Memw => airs.memws.get(chunk).map(|a| a.as_ref()),
        TableKind::MemwAligned => airs.memw_aligneds.get(chunk).map(|a| a.as_ref()),
        TableKind::MemwRegister => airs.memw_registers.get(chunk).map(|a| a.as_ref()),
        TableKind::Load => airs.loads.get(chunk).map(|a| a.as_ref()),
        TableKind::Cpu32 => airs.cpu32s.get(chunk).map(|a| a.as_ref()),
        TableKind::Branch => airs.branches.get(chunk).map(|a| a.as_ref()),
        TableKind::Eq => airs.eqs.get(chunk).map(|a| a.as_ref()),
        TableKind::Bytewise => airs.bytewises.get(chunk).map(|a| a.as_ref()),
        TableKind::Store => airs.stores.get(chunk).map(|a| a.as_ref()),
        _ => None,
    };

    type P = stark::prover::Prover<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
        (),
    >;
    let commit_root = |air: &dyn stark::traits::AIR<
        Field = crate::tables::types::GoldilocksField,
        FieldExtension = crate::tables::types::GoldilocksExtension,
        PublicInputs = (),
    >,
                       t: &stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >| { <P as IsStarkProver<_, _, _>>::commit_table_root(air, t) };

    let mut checked = 0usize;
    T::walk_and_emit_chunks(
        &artifacts,
        &elf,
        vec![],
        &image,
        &register_init,
        &max_rows,
        |kind, chunk, table| {
            let Some(air) = air_for(kind, chunk) else {
                return;
            };
            let walked = commit_root(air, &table).expect("the walk's chunk commits");
            let resident = match kind {
                TableKind::Cpu => &traces.cpus[chunk],
                TableKind::Memw => &traces.memws[chunk],
                TableKind::MemwAligned => &traces.memw_aligneds[chunk],
                TableKind::MemwRegister => &traces.memw_registers[chunk],
                TableKind::Load => &traces.loads[chunk],
                TableKind::Cpu32 => &traces.cpu32s[chunk],
                TableKind::Branch => &traces.branches[chunk],
                TableKind::Eq => &traces.eqs[chunk],
                TableKind::Bytewise => &traces.bytewises[chunk],
                TableKind::Store => &traces.stores[chunk],
                _ => unreachable!(),
            };
            let expected = commit_root(air, resident).expect("the resident chunk commits");
            assert_eq!(
                walked, expected,
                "{kind:?} chunk {chunk}: committed during the walk under a different root"
            );
            checked += 1;
        },
    )
    .expect("walk");

    assert!(
        checked > 0,
        "the fixture must close at least one chunk mid-walk"
    );
}

/// No table CPU32 feeds may be closed early by the Commit-phase walk.
///
/// `cpu32_chip_op` appends to SHIFT, MUL and DVRM once per word instruction, so
/// those are not final when a segment ends — closing one early would cut its
/// chunks somewhere the finished run does not.
///
/// Stated as an invariant rather than left to a fixture: catching it by data
/// needs a program with word instructions AND enough of the affected ops to
/// split a chunk, and a fixture that stops meeting that quietly stops testing
/// it. SHIFT was in fact closed early until this was noticed.
#[test]
fn cpu32_appends_are_excluded_from_early_closing() {
    use crate::tables::trace_builder::{CHUNKED_KINDS, CPU32_APPENDS_TO};

    for kind in CPU32_APPENDS_TO {
        assert!(
            !CHUNKED_KINDS.contains(&kind),
            "{kind:?} takes ops from cpu32_chip_op after a segment ends, so the walk \
             must not close it early"
        );
    }
}

/// The Commit phase must commit every chunk it closes under the root the
/// ordinary prover gives that chunk, and hand back the rest of the run.
///
/// This is the pass end to end: it walks the execution, commits and drops each
/// table as it fills, and returns what it could not close. Two things have to
/// hold for that to be a prover and not just a producer — the commitments have
/// to be the right ones, and nothing may fall between the chunks it closed and
/// the tail it kept.
#[test]
fn the_commit_phase_commits_what_it_closes_and_keeps_the_rest() {
    use crate::tables::trace_builder::{TableKind, Traces as T};
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let elf_bytes = crate::test_utils::asm_elf_bytes("fib_iterative_160k");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let max_rows = crate::tables::MaxRowsConfig {
        cpu: 1 << 15,
        memw: 1 << 10,
        load: 1 << 10,
        branch: 1 << 12,
        ..Default::default()
    };
    let proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .expect("blowup 2 is valid");

    let phase =
        crate::commit_phase::run(&elf, &[], &max_rows, &proof_options).expect("commit phase");
    assert!(
        !phase.closed.is_empty(),
        "the fixture must close at least one chunk mid-walk"
    );

    // Every commitment must be the one the resident chunk carries.
    let logs = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("run")
        .logs;
    let resident = T::from_elf_and_logs(
        &elf,
        &logs,
        &max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("traces");
    let counts = resident.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        false,
        &resident.page_configs,
        &counts,
        None,
        true,
        None,
        None,
        None,
    );
    type P = stark::prover::Prover<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
        (),
    >;
    use stark::prover::IsStarkProver;
    for (kind, chunk, root) in &phase.closed {
        let (air, table) = match kind {
            TableKind::Cpu => (airs.cpus[*chunk].as_ref(), &resident.cpus[*chunk]),
            TableKind::Memw => (airs.memws[*chunk].as_ref(), &resident.memws[*chunk]),
            TableKind::MemwAligned => (
                airs.memw_aligneds[*chunk].as_ref(),
                &resident.memw_aligneds[*chunk],
            ),
            TableKind::MemwRegister => (
                airs.memw_registers[*chunk].as_ref(),
                &resident.memw_registers[*chunk],
            ),
            TableKind::Load => (airs.loads[*chunk].as_ref(), &resident.loads[*chunk]),
            TableKind::Cpu32 => (airs.cpu32s[*chunk].as_ref(), &resident.cpu32s[*chunk]),
            TableKind::Branch => (airs.branches[*chunk].as_ref(), &resident.branches[*chunk]),
            TableKind::Eq => (airs.eqs[*chunk].as_ref(), &resident.eqs[*chunk]),
            TableKind::Bytewise => (airs.bytewises[*chunk].as_ref(), &resident.bytewises[*chunk]),
            TableKind::Store => (airs.stores[*chunk].as_ref(), &resident.stores[*chunk]),
            other => unreachable!("{other:?} is not closed mid-walk"),
        };
        let expected =
            <P as IsStarkProver<_, _, _>>::commit_table_root(air, table).expect("resident commits");
        assert_eq!(
            *root, expected,
            "{kind:?} chunk {chunk}: the Commit phase used a different root"
        );
    }

    // And nothing falls between what it closed and what it kept: the chunks it
    // closed plus the tail it kept are the chunks the resident build produced.
    // Exact for CPU, which is one op per executed cycle and takes nothing from
    // the end-of-run finalization; a bound elsewhere, since that finalization
    // appends after the last cycle and can spill the tail into another chunk.
    assert_eq!(
        phase.walked.leftover.cycles(),
        logs.len(),
        "the walk executed a different number of cycles than the straight run"
    );
    let closed_of = |kind: TableKind| phase.closed.iter().filter(|(k, _, _)| *k == kind).count();
    assert_eq!(
        closed_of(TableKind::Cpu) + 1,
        resident.cpus.len(),
        "CPU: the closed chunks plus the tail are not the run's chunks"
    );
    for (kind, produced) in [
        (TableKind::Memw, resident.memws.len()),
        (TableKind::MemwAligned, resident.memw_aligneds.len()),
        (TableKind::MemwRegister, resident.memw_registers.len()),
        (TableKind::Load, resident.loads.len()),
        (TableKind::Cpu32, resident.cpu32s.len()),
        (TableKind::Branch, resident.branches.len()),
        (TableKind::Eq, resident.eqs.len()),
        (TableKind::Bytewise, resident.bytewises.len()),
        (TableKind::Store, resident.stores.len()),
    ] {
        assert_eq!(
            phase.walked.leftover.emitted(kind),
            closed_of(kind),
            "{kind:?}: the leftover disagrees with what was committed"
        );
        assert!(
            closed_of(kind) < produced,
            "{kind:?}: closed {} of the run's {produced} chunks, leaving no tail",
            closed_of(kind)
        );
    }
}

/// Commit plus Challenge must cover every chunked table, each under the root
/// the ordinary prover gives it.
///
/// Together the two passes are supposed to account for the chunked side of the
/// proof with nothing missing and nothing committed twice: the chunks closed
/// mid-walk, the tails padded at the end, and the tables the walk could not
/// close at all. Checked against a real proof's roots, in position, so a table
/// committed under the wrong root or in the wrong slot fails here.
#[test]
fn the_two_phases_cover_every_chunked_table() {
    use crate::tables::trace_builder::{TableKind, Traces as T};
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use std::collections::HashMap;

    let elf_bytes = crate::test_utils::asm_elf_bytes("fib_iterative_160k");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let max_rows = crate::tables::MaxRowsConfig {
        cpu: 1 << 15,
        memw: 1 << 10,
        load: 1 << 10,
        branch: 1 << 12,
        ..Default::default()
    };
    let proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .expect("blowup 2 is valid");

    let phase = crate::commit_phase::run(&elf, &[], &max_rows, &proof_options).expect("commit");
    let closed = phase.closed.clone();
    let (rest, _) =
        crate::commit_phase::commit_remaining(phase.walked, &[], &max_rows, &proof_options)
            .expect("challenge");

    let mut got: HashMap<(TableKind, usize), _> = HashMap::new();
    for (kind, chunk, root) in closed.into_iter().chain(rest) {
        assert!(
            got.insert((kind, chunk), root).is_none(),
            "{kind:?} chunk {chunk} was committed twice"
        );
    }

    let logs = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("run")
        .logs;
    let resident = T::from_elf_and_logs(
        &elf,
        &logs,
        &max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("traces");
    let counts = resident.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        false,
        &resident.page_configs,
        &counts,
        None,
        true,
        None,
        None,
        None,
    );
    type P = stark::prover::Prover<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
        (),
    >;
    use stark::prover::IsStarkProver;

    let groups: [(TableKind, &Vec<_>, &Vec<_>); 4] = [
        (TableKind::Cpu, &airs.cpus, &resident.cpus),
        (TableKind::Branch, &airs.branches, &resident.branches),
        (TableKind::Lt, &airs.lts, &resident.lts),
        (TableKind::Memw, &airs.memws, &resident.memws),
    ];
    let mut checked = 0usize;
    for (kind, kind_airs, kind_traces) in groups {
        assert_eq!(
            kind_airs.len(),
            kind_traces.len(),
            "{kind:?}: AIR and trace counts disagree"
        );
        for (chunk, (air, table)) in kind_airs.iter().zip(kind_traces.iter()).enumerate() {
            let expected = <P as IsStarkProver<_, _, _>>::commit_table_root(air.as_ref(), table)
                .expect("resident commits");
            let actual = got
                .get(&(kind, chunk))
                .unwrap_or_else(|| panic!("{kind:?} chunk {chunk} was never committed"));
            assert_eq!(
                *actual, expected,
                "{kind:?} chunk {chunk}: committed under a different root"
            );
            checked += 1;
        }
    }
    assert!(checked > 4, "the fixture must cover several chunks");
}

/// A retired chunk must leave its BITWISE lookups behind.
///
/// BITWISE counts lookups from tables the Commit phase closes and drops, so the
/// contribution has to be taken while the chunk still exists. If it is not, the
/// table comes out short and the bus stops balancing — a failure that surfaces
/// at verification, far from the chunk that caused it.
///
/// The limits here are small for the kinds that owe BITWISE, so most of their
/// lookups belong to chunks that were closed and dropped rather than to the
/// tail. Removing the fold that runs before a chunk is drained fails this.
#[test]
fn retiring_a_chunk_keeps_its_bitwise_lookups() {
    use crate::tables::register::register_init_from_entry_point;
    use crate::tables::trace_builder::{
        DecodeArtifacts, TableKind, Traces as T, WalkLeftover, build_initial_image,
    };
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let elf_bytes = crate::test_utils::asm_elf_bytes("fib_iterative_160k");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    let image = build_initial_image(&elf, &[]);
    let register_init = register_init_from_entry_point(elf.entry_point);
    let max_rows = crate::tables::MaxRowsConfig {
        memw_aligned: 1 << 10,
        memw_register: 1 << 10,
        branch: 1 << 10,
        eq: 1 << 10,
        bytewise: 1 << 10,
        store: 1 << 10,
        ..Default::default()
    };

    let mut retired = 0usize;
    let mut leftover = T::walk_and_emit_chunks(
        &artifacts,
        &elf,
        vec![],
        &image,
        &register_init,
        &max_rows,
        |kind, _, _| {
            if matches!(
                kind,
                TableKind::MemwAligned
                    | TableKind::MemwRegister
                    | TableKind::Branch
                    | TableKind::Eq
                    | TableKind::Bytewise
                    | TableKind::Store
            ) {
                retired += 1;
            }
        },
    )
    .expect("walk");
    assert!(
        retired > 1,
        "the fixture must retire chunks that owe BITWISE, or this proves nothing"
    );
    leftover.finalize(&max_rows);

    let mut hist = leftover.bitwise_histogram();
    leftover.build_pages(&image, &[], &mut hist);
    let built = WalkLeftover::build_bitwise_from(&hist);

    let logs = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("run")
        .logs;
    let resident = T::from_elf_and_logs(
        &elf,
        &logs,
        &max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("traces");

    let flat = |t: &stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >| {
        let (data, _) = t.main_data_row_major();
        data.iter().map(|fe| *fe.value()).collect::<Vec<u64>>()
    };
    assert_eq!(
        flat(&built),
        flat(&resident.bitwise),
        "the lookups of the retired chunks are missing from BITWISE"
    );
}

/// The tables built from accumulated op lists must match the ordinary build.
///
/// COMMIT, KECCAK and its round tables, and the accelerator tables are written
/// once at the end from everything the run produced. The Commit phase drops the
/// chunked tables as it goes but has to keep feeding these, so the risk is the
/// opposite one: an op list quietly not accumulated comes out as an empty table
/// that still looks well-formed.
#[test]
fn the_accumulated_tables_match_the_ordinary_build() {
    use crate::tables::register::register_init_from_entry_point;
    use crate::tables::trace_builder::{DecodeArtifacts, Traces as T, build_initial_image};
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    // Uses keccak and the commit ecall, so the tables under test are not empty.
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_keccak");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    let image = build_initial_image(&elf, &[]);
    let register_init = register_init_from_entry_point(elf.entry_point);
    let max_rows = crate::tables::MaxRowsConfig::default();

    let mut leftover = T::walk_and_emit_chunks(
        &artifacts,
        &elf,
        vec![],
        &image,
        &register_init,
        &max_rows,
        |_, _, _| {},
    )
    .expect("walk");
    leftover.finalize(&max_rows);
    let built = leftover.build_accumulated();

    let logs = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("run")
        .logs;
    let resident = T::from_elf_and_logs(
        &elf,
        &logs,
        &max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("traces");

    let flat = |t: &stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >| {
        let (data, _) = t.main_data_row_major();
        data.iter().map(|fe| *fe.value()).collect::<Vec<u64>>()
    };
    for (name, a, b) in [
        ("COMMIT", &built.commit, &resident.commit),
        ("KECCAK", &built.keccak, &resident.keccak),
        ("KECCAK_RND", &built.keccak_rnd, &resident.keccak_rnd),
        ("KECCAK_RC", &built.keccak_rc, &resident.keccak_rc),
        ("ECSM", &built.ecsm, &resident.ecsm),
        ("ECDAS", &built.ecdas, &resident.ecdas),
        ("HINT", &built.hint, &resident.hint),
    ] {
        assert_eq!(flat(a), flat(b), "{name} differs from the ordinary build");
    }
    assert!(
        flat(&built.keccak).iter().any(|v| *v != 0),
        "the fixture must exercise KECCAK, or this proves nothing"
    );
}

/// DECODE's multiplicities must survive the chunks being dropped.
///
/// Every executed cycle looks DECODE up at its pc, and every padding row looks
/// it up at the padding pc. The Commit phase drops the CPU ops that carry those
/// pcs, so the lookups have to be counted while they still exist — by pc, since
/// listing them costs one entry per cycle, which is the thing being avoided.
#[test]
fn decode_multiplicities_survive_retiring_the_cpu_chunks() {
    use crate::tables::register::register_init_from_entry_point;
    use crate::tables::trace_builder::{DecodeArtifacts, Traces as T, build_initial_image};
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let elf_bytes = crate::test_utils::asm_elf_bytes("fib_iterative_160k");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    let image = build_initial_image(&elf, &[]);
    let register_init = register_init_from_entry_point(elf.entry_point);
    // Several CPU chunks, so most of the lookups belong to chunks that were
    // closed and dropped rather than to the tail. Deliberately NOT a power of
    // two: a full chunk then pads, and the padding lookups are a term that a
    // power-of-two limit would leave at zero and therefore untested.
    let max_rows = crate::tables::MaxRowsConfig {
        cpu: 10_000,
        ..Default::default()
    };

    let mut closed_cpu = 0usize;
    let mut leftover = T::walk_and_emit_chunks(
        &artifacts,
        &elf,
        vec![],
        &image,
        &register_init,
        &max_rows,
        |kind, _, _| {
            if kind == crate::tables::trace_builder::TableKind::Cpu {
                closed_cpu += 1;
            }
        },
    )
    .expect("walk");
    // DECODE is built in the end-of-run phase, after finalization freezes the
    // padding count; building it before would read a tail that is still growing.
    leftover.finalize(&max_rows);
    assert!(
        closed_cpu > 1,
        "the fixture must drop several CPU chunks, or the counting is untested"
    );
    let built = leftover.build_decode(
        artifacts.decode_trace.clone(),
        &artifacts.decode_pc_to_row,
        &max_rows,
    );

    let logs = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("run")
        .logs;
    let resident = T::from_elf_and_logs(
        &elf,
        &logs,
        &max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("traces");

    let flat = |t: &stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >| {
        let (data, _) = t.main_data_row_major();
        data.iter().map(|fe| *fe.value()).collect::<Vec<u64>>()
    };
    assert_eq!(
        flat(&built),
        flat(&resident.decode),
        "DECODE's multiplicities differ from the ordinary build"
    );
}

/// The end-of-run phase must produce every non-chunked table the ordinary build
/// produces, identically.
///
/// These are the tables that cannot be closed while the run continues: HALT
/// comes from the terminating ECALL, REGISTER's final PC token has to match the
/// last padding write, PAGE reads the memory image at the last cycle, and
/// BITWISE owes lookups that include PAGE's. Each depends on state the walk had
/// to carry rather than on an op list it could keep.
#[test]
fn the_end_of_run_tables_match_the_ordinary_build() {
    use crate::tables::trace_builder::Traces as T;
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let elf_bytes = crate::test_utils::asm_elf_bytes("fib_iterative_160k");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    // Not a power of two, so the CPU chunks pad and REGISTER's final PC token
    // depends on a padding count the walk had to accumulate.
    let max_rows = crate::tables::MaxRowsConfig {
        cpu: 10_000,
        ..Default::default()
    };
    let proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .expect("blowup 2 is valid");

    let phase = crate::commit_phase::run(&elf, &[], &max_rows, &proof_options).expect("commit");
    let (_, rest) =
        crate::commit_phase::commit_remaining(phase.walked, &[], &max_rows, &proof_options)
            .expect("challenge");

    let logs = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("run")
        .logs;
    let resident = T::from_elf_and_logs(
        &elf,
        &logs,
        &max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("traces");

    let flat = |t: &stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >| {
        let (data, _) = t.main_data_row_major();
        data.iter().map(|fe| *fe.value()).collect::<Vec<u64>>()
    };
    assert_eq!(flat(&rest.halt), flat(&resident.halt), "HALT differs");
    {
        let a = flat(&rest.register);
        let b = flat(&resident.register);
        let first = a.iter().zip(b.iter()).position(|(x, y)| x != y);
        eprintln!(
            "REGISTER: len {} vs {}, first diff at {:?} -> {:?} vs {:?}",
            a.len(),
            b.len(),
            first,
            first.map(|i| a[i]),
            first.map(|i| b[i])
        );
    }
    assert_eq!(
        flat(&rest.register),
        flat(&resident.register),
        "REGISTER differs"
    );
    assert_eq!(flat(&rest.decode), flat(&resident.decode), "DECODE differs");
    assert_eq!(
        flat(&rest.bitwise),
        flat(&resident.bitwise),
        "BITWISE differs"
    );
    assert_eq!(
        rest.pages.len(),
        resident.pages.len(),
        "a different number of PAGE tables"
    );
    assert!(
        !rest.pages.is_empty(),
        "the fixture must produce PAGE tables"
    );
    for (i, (a, b)) in rest.pages.iter().zip(resident.pages.iter()).enumerate() {
        assert_eq!(flat(a), flat(b), "PAGE {i} differs");
    }
}

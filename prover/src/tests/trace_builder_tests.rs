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
fn test_lt_no_dedup_one_row_per_op() {
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

    // Per the spec, `μ` is a Bit: one row per op (no deduplication). The 3
    // identical SLT operations (5 < 10, signed) each get their own row with μ = 1.
    let mut slt_rows = 0;
    for row_idx in 0..traces.lts[0].main_table.height {
        let row = traces.lts[0].main_table.get_row(row_idx);
        // lhs is stored as DWordHHW: [half0, half1, word2]; value 5 => [5, 0, 0].
        if row[lt::cols::LHS_0] == FE::from(5u64)
            && row[lt::cols::LHS_1] == FE::from(0u64)
            && row[lt::cols::LHS_2] == FE::from(0u64)
            && row[lt::cols::RHS_0] == FE::from(10u64)
            && row[lt::cols::SIGNED] == FE::from(1u64)
        {
            assert_eq!(
                row[lt::cols::MU],
                FE::from(1u64),
                "μ is the Bit 1 for each real LT row"
            );
            slt_rows += 1;
        }
    }
    assert_eq!(
        slt_rows, 3,
        "expected one LT row per SLT op (spec: μ is a Bit, no dedup)"
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
        assert_eq!(hwsl, 24 * 120, "Hwsl count");
        assert_eq!(is_half, 100, "IsHalf count");
        assert_eq!(ops.len(), 105 + 24 * 1148, "Total bitwise ops");
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
            1151,
            "KECCAK_RND: 3 IO + 440 theta + 300 rho + 400 chi + 8 iota \
             (Cxz_right Byte→Bit drops 40 ARE_BYTES per spec d75944ee; \
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
        let (core_constraints, _) = keccak::create_constraints(0);
        assert_eq!(
            core_constraints.len(),
            51,
            "KECCAK core: 25 ADD pairs + no-overflow"
        );

        let (rnd_constraints, _) = keccak_rnd::create_constraints(0);
        assert_eq!(
            rnd_constraints.len(),
            20,
            "KECCAK_RND: 20 IS_BIT(μ; Cxz_right_bit) per spec d75944ee"
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

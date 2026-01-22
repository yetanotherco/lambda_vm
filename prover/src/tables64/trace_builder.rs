//! Trace generation from execution logs.
//!
//! This module provides a single entry point for generating all trace tables
//! from execution logs in a single pass.
//!
//! ## Usage
//!
//! ```ignore
//! use prover::tables64::trace_builder::Traces;
//!
//! let traces = Traces::from_logs(&logs);
//! // Use traces.cpu, traces.bitwise, traces.lt
//! ```

use executor::vm::logs::Log;
use stark::trace::TraceTable;

use super::bitwise;
use super::cpu::{self, CpuOperation};
use super::lt::{self, LtOperation};
use super::types::{GoldilocksExtension, GoldilocksField};

/// All generated trace tables.
pub struct Traces {
    /// CPU execution trace (one row per instruction)
    pub cpu: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// BITWISE precomputed lookup table (2^20 rows)
    pub bitwise: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// LT comparison trace (deduplicated operations)
    pub lt: TraceTable<GoldilocksField, GoldilocksExtension>,
}

impl Traces {
    /// Generates all traces from execution logs in a single pass.
    pub fn from_logs(logs: &[Log]) -> Self {
        // Pre-allocate collectors
        let mut cpu_ops = Vec::with_capacity(logs.len());
        let mut bitwise_lookups = Vec::with_capacity(logs.len() * 4);
        let mut lt_ops = Vec::with_capacity(logs.len() / 10 + 1);

        // Single pass over logs
        for (i, log) in logs.iter().enumerate() {
            let timestamp = (i as u64) * 4;
            let op = CpuOperation::from_log(log, timestamp);

            // Collect bitwise lookups from this operation
            bitwise_lookups.extend(op.collect_bitwise_lookups());

            // Collect LT operations for SLT and BLT instructions
            if op.op_slt || op.op_blt {
                let arg1 = op.compute_arg1();
                let arg2 = op.compute_arg2();
                lt_ops.push(LtOperation::new(arg1, arg2, op.signed));
            }

            cpu_ops.push(op);
        }

        // Generate CPU trace (with padding)
        let cpu = Self::generate_cpu_trace_padded(cpu_ops);

        // Generate BITWISE trace and update multiplicities
        let mut bitwise = bitwise::generate_bitwise_trace();
        bitwise::update_multiplicities(&mut bitwise, &bitwise_lookups);

        // Generate LT trace (includes deduplication and padding)
        let lt = if lt_ops.is_empty() {
            lt::generate_lt_trace(&[LtOperation::new(0, 0, false)])
        } else {
            lt::generate_lt_trace(&lt_ops)
        };

        Traces { cpu, bitwise, lt }
    }

    /// Generates CPU trace with padding to power of 2.
    fn generate_cpu_trace_padded(
        mut ops: Vec<CpuOperation>,
    ) -> TraceTable<GoldilocksField, GoldilocksExtension> {
        if ops.is_empty() {
            ops = Self::create_padding_ops(4);
        } else {
            let n = ops.len();
            let target = n.next_power_of_two().max(4);
            if n < target {
                ops.extend(Self::create_padding_ops(target - n));
            }
        }
        cpu::generate_cpu_trace(&ops)
    }

    /// Creates padding CPU operations (ADD x0, x0, x0 - no-op).
    fn create_padding_ops(count: usize) -> Vec<CpuOperation> {
        (0..count)
            .map(|i| {
                let mut op = CpuOperation::default();
                op.timestamp = (i as u64) * 4;
                op.op_add = true;
                op
            })
            .collect()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables64::types::FE;
    use executor::vm::instruction::decoding::{ArithOp, Comparison, Instruction};

    fn make_add_log(pc: u64, rs1_val: u64, rs2_val: u64, dst_val: u64) -> Log {
        Log {
            instruction: Instruction::Arith {
                dst: 1,
                src1: 2,
                src2: 3,
                op: ArithOp::Add,
            },
            current_pc: pc,
            next_pc: pc + 4,
            src1_val: rs1_val,
            src2_val: rs2_val,
            dst_val,
        }
    }

    fn make_slt_log(pc: u64, rs1_val: u64, rs2_val: u64, result: u64) -> Log {
        Log {
            instruction: Instruction::Arith {
                dst: 1,
                src1: 2,
                src2: 3,
                op: ArithOp::SetLessThan,
            },
            current_pc: pc,
            next_pc: pc + 4,
            src1_val: rs1_val,
            src2_val: rs2_val,
            dst_val: result,
        }
    }

    fn make_blt_log(pc: u64, rs1_val: u64, rs2_val: u64, taken: bool) -> Log {
        Log {
            instruction: Instruction::Branch {
                src1: 2,
                src2: 3,
                cond: Comparison::LessThan,
                offset: 8,
            },
            current_pc: pc,
            next_pc: if taken { pc + 8 } else { pc + 4 },
            src1_val: rs1_val,
            src2_val: rs2_val,
            dst_val: 0,
        }
    }

    fn make_and_log(pc: u64, rs1_val: u64, rs2_val: u64, result: u64) -> Log {
        Log {
            instruction: Instruction::Arith {
                dst: 1,
                src1: 2,
                src2: 3,
                op: ArithOp::And,
            },
            current_pc: pc,
            next_pc: pc + 4,
            src1_val: rs1_val,
            src2_val: rs2_val,
            dst_val: result,
        }
    }

    #[test]
    fn test_empty_logs() {
        let traces = Traces::from_logs(&[]);

        assert!(traces.cpu.main_table.height >= 4);
        assert_eq!(traces.bitwise.main_table.height, bitwise::NUM_ROWS);
        assert!(traces.lt.main_table.height >= 2);
    }

    #[test]
    fn test_single_log() {
        let logs = vec![make_add_log(0x1000, 10, 20, 30)];
        let traces = Traces::from_logs(&logs);

        assert_eq!(traces.cpu.main_table.height, 4); // padded
    }

    #[test]
    fn test_power_of_two_logs() {
        let logs: Vec<Log> = (0..4)
            .map(|i| make_add_log(0x1000 + i * 4, i, i, i * 2))
            .collect();

        let traces = Traces::from_logs(&logs);
        assert_eq!(traces.cpu.main_table.height, 4);
    }

    #[test]
    fn test_padding_to_power_of_two() {
        let logs: Vec<Log> = (0..5)
            .map(|i| make_add_log(0x1000 + i * 4, i, i, i * 2))
            .collect();

        let traces = Traces::from_logs(&logs);
        assert_eq!(traces.cpu.main_table.height, 8); // 5 -> 8
    }

    #[test]
    fn test_lt_operations_collected() {
        let logs = vec![
            make_slt_log(0x1000, 5, 10, 1),
            make_slt_log(0x1004, 10, 5, 0),
            make_add_log(0x1008, 1, 2, 3),
            make_blt_log(0x100c, 3, 7, true),
        ];

        let traces = Traces::from_logs(&logs);

        // LT trace should have rows (2 SLT + 1 BLT = 3 ops, deduplicated)
        assert!(traces.lt.main_table.height >= 2);
    }

    #[test]
    fn test_lt_deduplication() {
        let logs = vec![
            make_slt_log(0x1000, 5, 10, 1),
            make_slt_log(0x1004, 5, 10, 1), // duplicate
            make_slt_log(0x1008, 5, 10, 1), // duplicate
            make_add_log(0x100c, 0, 0, 0),  // padding to 4
        ];

        let traces = Traces::from_logs(&logs);

        // Should have 1 unique LT op with multiplicity 3
        assert_eq!(traces.lt.main_table.height, 2); // 1 op padded to 2
        let row = traces.lt.main_table.get_row(0);
        assert_eq!(row[lt::cols::MU], FE::from(3u64));
    }

    #[test]
    fn test_bitwise_lookups_collected() {
        let logs = vec![
            make_and_log(0x1000, 0x12, 0x34, 0x10),
            make_add_log(0x1004, 0, 0, 0),
            make_add_log(0x1008, 0, 0, 0),
            make_add_log(0x100c, 0, 0, 0),
        ];

        let traces = Traces::from_logs(&logs);

        // Check AND multiplicity was updated for (0x12, 0x34, 0)
        let row_idx = bitwise::row_index(0x12, 0x34, 0);
        let row = traces.bitwise.main_table.get_row(row_idx);
        assert_eq!(row[bitwise::cols::MU_AND], FE::one());
    }

    #[test]
    fn test_cpu_timestamps() {
        let logs = vec![
            make_add_log(0x1000, 1, 2, 3),
            make_add_log(0x1004, 4, 5, 6),
            make_add_log(0x1008, 7, 8, 9),
            make_add_log(0x100c, 10, 11, 12),
        ];

        let traces = Traces::from_logs(&logs);

        // Check timestamps are 0, 4, 8, 12
        for i in 0..4 {
            let row = traces.cpu.main_table.get_row(i);
            assert_eq!(row[cpu::cols::TIMESTAMP], FE::from((i * 4) as u64));
        }
    }

    #[test]
    fn test_mixed_instructions() {
        let logs = vec![
            make_add_log(0x1000, 10, 20, 30),
            make_slt_log(0x1004, 5, 10, 1),
            make_and_log(0x1008, 0xFF, 0xF0, 0xF0),
            make_blt_log(0x100c, 1, 2, true),
        ];

        let traces = Traces::from_logs(&logs);

        assert_eq!(traces.cpu.main_table.height, 4);
        assert_eq!(traces.bitwise.main_table.height, bitwise::NUM_ROWS);
        // 1 SLT + 1 BLT = 2 LT ops
        assert!(traces.lt.main_table.height >= 2);
    }
}

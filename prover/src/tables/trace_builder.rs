//! Trace generation from execution logs.
//!
//! This module provides a single entry point for generating all trace tables
//! from execution logs in a single pass.
//!
//! ## Usage
//!
//! ```ignore
//! use prover::tables::trace_builder::Traces;
//!
//! let traces = Traces::from_logs(&logs);
//! // Use traces.cpu, traces.bitwise, traces.lt
//! ```

use executor::vm::instruction;
use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
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
    pub fn from_logs(logs: &[Log], instructions: U64HashMap<Instruction>) -> Self {
        // Pre-allocate collectors
        let mut cpu_ops = Vec::with_capacity(logs.len());
        let mut bitwise_lookups = Vec::with_capacity(logs.len() * 4);
        let mut lt_ops = Vec::with_capacity(logs.len() / 10 + 1);

        // Single pass over logs
        for (i, log) in logs.iter().enumerate() {
            let timestamp = (i as u64) * 4;
            let instruction = instructions
                .get(&log.current_pc)
                .cloned().expect("");
            let op = CpuOperation::from_log(log, timestamp, instruction);

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

        // Generate CPU trace (handles padding internally)
        let cpu = cpu::generate_cpu_trace(&cpu_ops);

        // Generate BITWISE trace and update multiplicities
        let mut bitwise = bitwise::generate_bitwise_trace();
        bitwise::update_multiplicities(&mut bitwise, &bitwise_lookups);

        // Generate LT trace (handles deduplication and padding internally)
        let lt = lt::generate_lt_trace(&lt_ops);

        Traces { cpu, bitwise, lt }
    }
}

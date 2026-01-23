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
//! let traces = Traces::from_logs(&logs)?;
//! // Use traces.cpu, traces.bitwise, traces.lt
//! ```

use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
use stark::trace::TraceTable;

use super::bitwise;
use super::cpu::{self, CpuOperation};
use super::decode;
use super::lt::{self, LtOperation};
use super::types::{GoldilocksExtension, GoldilocksField};
use crate::ProverError;

/// All generated trace tables.
pub struct Traces {
    /// CPU execution trace (one row per instruction)
    pub cpu: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// BITWISE precomputed lookup table (2^20 rows)
    pub bitwise: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// LT comparison trace (deduplicated operations)
    pub lt: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// DECODE instruction decode trace (deduplicated by PC)
    pub decode: TraceTable<GoldilocksField, GoldilocksExtension>,
}

impl Traces {
    /// Generates all traces from execution logs.
    pub fn from_logs(
        logs: &[Log],
        instructions: U64HashMap<Instruction>,
    ) -> Result<Self, ProverError> {
        // Generate DECODE trace first from instructions (MU=0 initially)
        let (mut decode, pc_to_row) = decode::generate_decode_trace(&instructions);

        // Generate BITWISE precomputed table (MU=0 initially)
        let mut bitwise = bitwise::generate_bitwise_trace();

        // Pre-allocate collectors for log processing
        let mut cpu_ops = Vec::with_capacity(logs.len());
        let mut bitwise_lookups = Vec::with_capacity(logs.len() * 4);
        let mut lt_ops = Vec::with_capacity(logs.len() / 10 + 1);
        let mut decode_lookups = Vec::with_capacity(logs.len());

        // Process logs to build CPU trace and collect lookups
        for (i, log) in logs.iter().enumerate() {
            let timestamp = (i as u64) * 4;
            let instruction = instructions
                .get(&log.current_pc)
                .copied()
                .ok_or(ProverError::MissingInstruction(log.current_pc))?;
            let op = CpuOperation::from_log(log, timestamp, instruction);

            // Collect bitwise lookups from this operation
            bitwise_lookups.extend(op.collect_bitwise_lookups());

            // Collect LT operations for SLT and BLT instructions
            if op.op_slt || op.op_blt {
                let arg1 = op.compute_arg1();
                let arg2 = op.compute_arg2();
                lt_ops.push(LtOperation::new(arg1, arg2, op.signed));
            }

            // Collect PC for DECODE lookups
            decode_lookups.push(log.current_pc);

            cpu_ops.push(op);
        }

        // Generate CPU trace
        let cpu = cpu::generate_cpu_trace(&cpu_ops);

        // Update BITWISE multiplicities
        bitwise::update_multiplicities(&mut bitwise, &bitwise_lookups);

        // Generate LT trace (handles deduplication and padding internally)
        let lt = lt::generate_lt_trace(&lt_ops);

        // Update DECODE multiplicities
        decode::update_multiplicities(&mut decode, &pc_to_row, &decode_lookups);

        Ok(Traces { cpu, bitwise, lt, decode })
    }
}

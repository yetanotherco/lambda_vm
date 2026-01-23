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

use std::collections::HashMap;

use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
use stark::trace::TraceTable;

use super::bitwise;
use super::cpu::{self, CpuOperation};
use super::decode::{self, DecodeEntry};
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
    ///
    /// Uses streaming processing to minimize memory usage - CPU rows are written
    /// directly during log iteration without collecting all CpuOperations first.
    pub fn from_logs(
        logs: &[Log],
        instructions: U64HashMap<Instruction>,
    ) -> Result<Self, ProverError> {
        // Build DecodeEntry map from instructions (single source of truth for decode logic)
        let decode_entries: HashMap<u64, DecodeEntry> = instructions
            .iter()
            .map(|(&pc, &instr)| (pc, DecodeEntry::from_instruction(pc, instr)))
            .collect();

        // Generate DECODE trace from instructions (MU=0 initially)
        let (mut decode, pc_to_row) = decode::generate_decode_trace(&instructions);

        // Generate BITWISE precomputed table (MU=0 initially)
        let mut bitwise = bitwise::generate_bitwise_trace();

        // Pre-allocate CPU trace data for streaming writes
        let mut cpu_data = cpu::create_cpu_trace(logs.len());

        // Pre-allocate collectors for lookups
        let mut bitwise_lookups = Vec::new();
        let mut lt_ops = Vec::new();
        let mut decode_lookups = Vec::with_capacity(logs.len());

        // Process logs: stream CPU rows directly, collect only lookups
        for (i, log) in logs.iter().enumerate() {
            let timestamp = (i as u64) * 4;

            // Look up pre-computed DecodeEntry (reuses decode logic from DECODE table)
            let decode_entry = decode_entries
                .get(&log.current_pc)
                .ok_or(ProverError::MissingInstruction(log.current_pc))?;

            // Create CpuOperation from DecodeEntry + runtime values
            let op = CpuOperation::from_decode_entry(decode_entry, log, timestamp);

            // Write CPU row directly (streaming - op not stored)
            cpu::write_cpu_row(&mut cpu_data, i, &op);

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

            // op is dropped here - no accumulation in memory
        }

        // Finalize CPU trace
        let cpu = cpu::finalize_cpu_trace(cpu_data);

        // Update BITWISE multiplicities
        bitwise::update_multiplicities(&mut bitwise, &bitwise_lookups);

        // Generate LT trace (handles deduplication and padding internally)
        let lt = lt::generate_lt_trace(&lt_ops);

        // Update DECODE multiplicities
        decode::update_multiplicities(&mut decode, &pc_to_row, &decode_lookups);

        Ok(Traces {
            cpu,
            bitwise,
            lt,
            decode,
        })
    }
}

//! Flamegraph generation for guest RISC-V programs.
//!
//! Tracks function calls during execution and outputs folded stack format
//! compatible with flamegraph.pl and inferno.

use std::collections::HashMap;
use std::io::{self, Write};

use rustc_demangle::demangle as rustc_demangle;

use crate::elf::SymbolTable;
use crate::profile::classify;
use crate::vm::execution::InstructionCache;
use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::KECCAK_SYSCALL_NUMBER;
use crate::vm::logs::Log;

/// Errors that can occur during flamegraph generation.
#[derive(Debug)]
pub enum FlamegraphError {
    /// Instruction not found for a given program counter.
    InstructionNotFound,
}

/// How each instruction contributes to a frame's weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightMode {
    /// Each instruction adds 1 — frame width is dynamic instruction count.
    InstructionCount,
    /// Each instruction adds its approximate trace-row weight
    /// ([`InstrClass::trace_row_weight`]) — frame width tracks proving cost.
    /// This is a coarse, documented estimate, not the exact committed row count
    /// (see `lambda_vm_prover::table_report` for exact per-table figures).
    TraceCost,
}

/// Generates flamegraph data by tracking function calls during execution.
pub struct FlamegraphGenerator {
    /// Symbol table for address-to-name resolution
    symbols: SymbolTable,
    /// Current call stack (function entry addresses)
    call_stack: Vec<u64>,
    /// Accumulated weight per stack state: "main;foo;bar" -> weight
    stack_counts: HashMap<String, u64>,
    /// Whether frames are weighted by instruction count or estimated trace cost.
    weight_mode: WeightMode,
}

impl FlamegraphGenerator {
    /// Create a new flamegraph generator with the given symbol table. Frames
    /// are weighted by dynamic instruction count.
    pub fn new(symbols: SymbolTable, entry_point: u64) -> Self {
        Self::with_weight_mode(symbols, entry_point, WeightMode::InstructionCount)
    }

    /// Create a flamegraph generator with an explicit weighting mode.
    pub fn with_weight_mode(
        symbols: SymbolTable,
        entry_point: u64,
        weight_mode: WeightMode,
    ) -> Self {
        Self {
            symbols,
            call_stack: vec![entry_point], // Start with entry point on stack
            stack_counts: HashMap::new(),
            weight_mode,
        }
    }

    /// Process a batch of execution logs, updating call stack and instruction counts.
    pub fn process_logs(
        &mut self,
        logs: &[Log],
        instructions: &InstructionCache,
    ) -> Result<(), FlamegraphError> {
        for log in logs {
            let instruction = instructions
                .get(log.current_pc)
                .copied()
                .ok_or(FlamegraphError::InstructionNotFound)?;

            // Count this instruction under the current stack. ECALLs (syscalls)
            // are not Rust function calls and have no return semantics, so we
            // attribute them to a synthetic leaf frame `ecall:<name>` appended
            // under the current caller rather than pushing onto the call stack.
            // This makes precompile syscalls (keccak, ecsm, commit) — which
            // dominate verifier runs — visible instead of being folded into
            // their caller.
            let stack_key = match syscall_name(log, instruction) {
                Some(name) => format!("{};{}", self.format_stack(), name),
                None => self.format_stack(),
            };
            let weight = match self.weight_mode {
                WeightMode::InstructionCount => 1,
                WeightMode::TraceCost => classify(instruction, log).trace_row_weight(),
            };
            *self.stack_counts.entry(stack_key).or_insert(0) += weight;

            // Update call stack based on instruction type
            self.update_stack(log, instruction);
        }
        Ok(())
    }

    /// Format the current call stack as a semicolon-separated string.
    fn format_stack(&self) -> String {
        if self.call_stack.is_empty() {
            return "<root>".to_string();
        }

        self.call_stack
            .iter()
            .map(|&addr| self.resolve_address(addr))
            .collect::<Vec<_>>()
            .join(";")
    }

    /// Resolve an address to a function name, or hex address if unknown.
    fn resolve_address(&self, address: u64) -> String {
        self.symbols
            .lookup(address)
            .map(|sym| demangle(&sym.name))
            .unwrap_or_else(|| format!("0x{:x}", address))
    }

    /// Update the call stack based on the instruction type.
    fn update_stack(&mut self, log: &Log, instruction: Instruction) {
        match instruction {
            // Function CALL: JAL with dst=ra (register 1)
            // Saves return address to ra and jumps to offset
            Instruction::JumpAndLink { dst: 1, .. } => {
                self.call_stack.push(log.next_pc);
            }

            // Function CALL: JALR with dst=ra (register 1)
            // Indirect call through register
            Instruction::JumpAndLinkRegister { dst: 1, .. } => {
                self.call_stack.push(log.next_pc);
            }

            // Function RETURN: JALR with base=ra (register 1), dst=zero (register 0)
            // This is the standard "ret" instruction (jalr x0, ra, 0)
            // Only pop if we have more than the root frame to prevent stack underflow
            Instruction::JumpAndLinkRegister { base, dst, .. } if base == 1 && dst == 0 => {
                if self.call_stack.len() > 1 {
                    self.call_stack.pop();
                }
            }

            // Tail call: JAL/JALR with dst=zero (doesn't save return address)
            // Pop current function and push the new one
            // Only pop if we have more than the root frame to prevent stack underflow
            Instruction::JumpAndLink { dst: 0, .. } => {
                if self.call_stack.len() > 1 {
                    self.call_stack.pop();
                }
                self.call_stack.push(log.next_pc);
            }
            Instruction::JumpAndLinkRegister { dst: 0, base, .. } if base != 1 => {
                // Tail call through register (not a return)
                if self.call_stack.len() > 1 {
                    self.call_stack.pop();
                }
                self.call_stack.push(log.next_pc);
            }

            _ => {}
        }
    }

    /// Write the folded stack output to a writer.
    ///
    /// Output format: `stack;frame;names count`
    /// Example: `main;quicksort;partition 12345`
    pub fn write_folded<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Sort by stack path for deterministic output
        let mut stacks: Vec<_> = self.stack_counts.iter().collect();
        stacks.sort_by_key(|(k, _)| k.as_str());

        for (stack, count) in stacks {
            if !stack.is_empty() {
                writeln!(writer, "{} {}", stack, count)?;
            }
        }

        Ok(())
    }

    /// Total accumulated weight across all frames. In
    /// [`WeightMode::InstructionCount`] this is the dynamic instruction count; in
    /// [`WeightMode::TraceCost`] it is the summed estimated trace-row weight.
    pub fn total_instructions(&self) -> u64 {
        self.stack_counts.values().sum()
    }

    /// The weighting mode this generator was built with.
    pub fn weight_mode(&self) -> WeightMode {
        self.weight_mode
    }
}

/// If `instruction` is an ECALL, return the synthetic flamegraph frame name for
/// its syscall, e.g. `ecall:keccak_permute`. The syscall number is taken from
/// `log.src1_val` (the guest's x17, as recorded by the executor for ECALLs).
/// Returns `None` for every non-ECALL instruction.
fn syscall_name(log: &Log, instruction: Instruction) -> Option<&'static str> {
    if !matches!(instruction, Instruction::EcallEbreak) {
        return None;
    }
    // This branch's executor has no ECSM syscall; an ECSM ecall (if any) falls
    // through to "ecall:unknown".
    Some(match log.src1_val {
        v if v == KECCAK_SYSCALL_NUMBER => "ecall:keccak_permute",
        64 => "ecall:commit",
        93 => "ecall:halt",
        1 => "ecall:print",
        2 => "ecall:panic",
        _ => "ecall:unknown",
    })
}

/// Demangle a Rust symbol name using the official rustc-demangle crate.
///
/// Uses the alternate format (`{:#}`) to omit the hash suffix for cleaner output.
pub(crate) fn demangle(name: &str) -> String {
    // Use rustc-demangle with alternate format to omit hash
    format!("{:#}", rustc_demangle(name))
}

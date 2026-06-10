//! Flamegraph generation for guest RISC-V programs.
//!
//! Tracks function calls during execution and outputs folded stack format
//! compatible with flamegraph.pl and inferno.

use std::collections::HashMap;
use std::io::{self, Write};

use rustc_demangle::demangle as rustc_demangle;

use crate::elf::SymbolTable;
use crate::vm::execution::InstructionCache;
use crate::vm::instruction::decoding::Instruction;
use crate::vm::logs::Log;

/// Errors that can occur during flamegraph generation.
#[derive(Debug)]
pub enum FlamegraphError {
    /// Instruction not found for a given program counter.
    InstructionNotFound,
}

/// Generates flamegraph data by tracking function calls during execution.
pub struct FlamegraphGenerator {
    /// Symbol table for address-to-name resolution
    symbols: SymbolTable,
    /// Current call stack (function entry addresses)
    call_stack: Vec<u64>,
    /// Instruction counts per stack state: "main;foo;bar" -> count
    stack_counts: HashMap<String, u64>,
}

impl FlamegraphGenerator {
    /// Create a new flamegraph generator with the given symbol table.
    pub fn new(symbols: SymbolTable, entry_point: u64) -> Self {
        Self {
            symbols,
            call_stack: vec![entry_point], // Start with entry point on stack
            stack_counts: HashMap::new(),
        }
    }

    /// Process a batch of execution logs, updating call stack and instruction counts.
    pub fn process_logs(
        &mut self,
        logs: &[Log],
        instructions: &InstructionCache,
    ) -> Result<(), FlamegraphError> {
        for log in logs {
            // Count this instruction under the current stack
            let stack_key = self.format_stack();
            *self.stack_counts.entry(stack_key).or_insert(0) += 1;

            // Update call stack based on instruction type
            let instruction = instructions
                .get(log.current_pc)
                .copied()
                .ok_or(FlamegraphError::InstructionNotFound)?;
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

    /// Get the total number of instructions processed.
    pub fn total_instructions(&self) -> u64 {
        self.stack_counts.values().sum()
    }
}

/// Demangle a Rust symbol name using the official rustc-demangle crate.
///
/// Uses the alternate format (`{:#}`) to omit the hash suffix for cleaner output.
pub(crate) fn demangle(name: &str) -> String {
    // Use rustc-demangle with alternate format to omit hash
    format!("{:#}", rustc_demangle(name))
}

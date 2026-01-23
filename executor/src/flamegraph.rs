//! Flamegraph generation for guest RISC-V programs.
//!
//! Tracks function calls during execution and outputs folded stack format
//! compatible with flamegraph.pl and inferno.

use std::collections::HashMap;
use std::io::{self, Write};

use crate::elf::SymbolTable;
use crate::vm::instruction::decoding::Instruction;
use crate::vm::logs::Log;
use crate::vm::memory::U64HashMap;

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
    /// Entry point address (for root frame)
    entry_point: u64,
}

impl FlamegraphGenerator {
    /// Create a new flamegraph generator with the given symbol table.
    pub fn new(symbols: SymbolTable, entry_point: u64) -> Self {
        Self {
            symbols,
            call_stack: vec![entry_point], // Start with entry point on stack
            stack_counts: HashMap::new(),
            entry_point,
        }
    }

    /// Process a batch of execution logs, updating call stack and instruction counts.
    pub fn process_logs(&mut self, logs: &[Log], instructions: U64HashMap<Instruction>) -> Result<(), FlamegraphError> {
        for log in logs {
            // Count this instruction under the current stack
            let stack_key = self.format_stack();
            *self.stack_counts.entry(stack_key).or_insert(0) += 1;

            // Update call stack based on instruction type
            let instruction = instructions
                .get(&log.current_pc)
                .copied().unwrap_or_else(|| FlamegraphError::InstructionNotFound)?;
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
            Instruction::JumpAndLink { dst, .. } if dst == 1 => {
                self.call_stack.push(log.next_pc);
            }

            // Function CALL: JALR with dst=ra (register 1)
            // Indirect call through register
            Instruction::JumpAndLinkRegister { dst, .. } if dst == 1 => {
                self.call_stack.push(log.next_pc);
            }

            // Function RETURN: JALR with base=ra (register 1), dst=zero (register 0)
            // This is the standard "ret" instruction (jalr x0, ra, 0)
            Instruction::JumpAndLinkRegister { base, dst, .. } if base == 1 && dst == 0 => {
                self.call_stack.pop();
            }

            // Tail call: JAL/JALR with dst=zero (doesn't save return address)
            // Pop current function and push the new one
            Instruction::JumpAndLink { dst, .. } if dst == 0 => {
                self.call_stack.pop();
                self.call_stack.push(log.next_pc);
            }
            Instruction::JumpAndLinkRegister { dst, base, .. } if dst == 0 && base != 1 => {
                // Tail call through register (not a return)
                self.call_stack.pop();
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

    /// Get the entry point address.
    pub fn entry_point(&self) -> u64 {
        self.entry_point
    }
}

/// Basic Rust symbol demangling.
///
/// Handles common patterns but not all edge cases.
/// For full support, use the `rustc-demangle` crate.
fn demangle(name: &str) -> String {
    // Handle Rust v0 mangling: _R...
    if name.starts_with("_R") {
        // Try to extract a readable name from v0 mangling
        // This is a simplified version - full demangling is complex
        if let Some(demangled) = try_demangle_v0(name) {
            return clean_demangled(&demangled);
        }
    }

    // Handle legacy Rust mangling: _ZN...
    if name.starts_with("_ZN") {
        if let Some(demangled) = try_demangle_legacy(name) {
            return clean_demangled(&demangled);
        }
    }

    // Return original if can't demangle
    name.to_string()
}

/// Clean up demangled names by removing common noise patterns.
fn clean_demangled(name: &str) -> String {
    // Split by :: and filter out segments that look like internal/generated names
    let parts: Vec<&str> = name
        .split("::")
        .filter(|part| {
            // Keep the part if it doesn't look like a hash or internal name
            // Hashes are typically short alphanumeric strings
            if part.len() <= 10 && part.chars().all(|c| c.is_alphanumeric()) {
                // Check if it looks like a crate hash (mixed case, starts with uppercase or contains digits)
                let has_upper = part.chars().any(|c| c.is_uppercase());
                let has_digit = part.chars().any(|c| c.is_ascii_digit());
                let starts_upper = part
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false);

                // Likely a hash if it has mixed case with digits or starts with uppercase with digits
                if (has_upper && has_digit) || (starts_upper && has_digit) {
                    return false;
                }
            }
            // Filter out closure markers like {closure#0}
            if part.starts_with('{') {
                return false;
            }
            // Filter out anonymous type markers
            if part.starts_with('_') && part.len() <= 3 {
                return false;
            }
            true
        })
        .collect();

    if parts.is_empty() {
        name.to_string()
    } else {
        parts.join("::")
    }
}

/// Try to demangle Rust v0 symbol names (_R prefix).
fn try_demangle_v0(name: &str) -> Option<String> {
    // v0 format: _R followed by path encoding
    // Very simplified: just extract identifiers
    let mut result = Vec::new();
    let mut chars = name[2..].chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            'N' => {
                // Namespace - skip the namespace identifier
                if let Some(ns) = chars.next() {
                    // v/V = value, t/T = type, etc.
                    if !ns.is_ascii_digit() {
                        continue;
                    }
                }
            }
            'C' | 'M' => {
                // Crate root or inherent impl - parse length-prefixed identifier
                if let Some(ident) = parse_length_prefixed(&mut chars) {
                    result.push(ident);
                }
            }
            c if c.is_ascii_digit() => {
                // Length-prefixed identifier
                let mut len_str = c.to_string();
                while let Some(&next) = chars.peek() {
                    if next.is_ascii_digit() {
                        len_str.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                if let Ok(len) = len_str.parse::<usize>() {
                    let ident: String = chars.by_ref().take(len).collect();
                    if !ident.is_empty() {
                        result.push(ident);
                    }
                }
            }
            _ => {}
        }
    }

    if result.is_empty() {
        None
    } else {
        Some(result.join("::"))
    }
}

/// Try to demangle Rust legacy symbol names (_ZN prefix).
fn try_demangle_legacy(name: &str) -> Option<String> {
    // Legacy format: _ZN followed by length-prefixed segments ending with E
    let mut result = Vec::new();
    let content = name.strip_prefix("_ZN")?;
    let content = content.strip_suffix('E').unwrap_or(content);

    let mut chars = content.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            if let Some(ident) = parse_length_prefixed(&mut chars) {
                // Skip hash suffixes (17h followed by hex)
                if ident.starts_with("h") && ident.len() == 17 {
                    continue;
                }
                result.push(ident);
            }
        } else {
            chars.next();
        }
    }

    if result.is_empty() {
        None
    } else {
        Some(result.join("::"))
    }
}

/// Parse a length-prefixed identifier from the character stream.
fn parse_length_prefixed(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<String> {
    let mut len_str = String::new();

    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            len_str.push(chars.next().unwrap());
        } else {
            break;
        }
    }

    let len = len_str.parse::<usize>().ok()?;
    let ident: String = chars.take(len).collect();

    if ident.len() == len {
        Some(ident)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demangle_simple() {
        assert_eq!(demangle("main"), "main");
        assert_eq!(demangle("_start"), "_start");
    }

    #[test]
    fn test_demangle_legacy() {
        // _ZN4core3ptr4readE -> core::ptr::read
        let demangled = demangle("_ZN4core3ptr4readE");
        assert!(demangled.contains("core") || demangled.contains("read"));
    }
}

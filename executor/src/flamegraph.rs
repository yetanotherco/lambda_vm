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
    pub fn process_logs(
        &mut self,
        logs: &[Log],
        instructions: &U64HashMap<Instruction>,
    ) -> Result<(), FlamegraphError> {
        for log in logs {
            // Count this instruction under the current stack
            let stack_key = self.format_stack();
            *self.stack_counts.entry(stack_key).or_insert(0) += 1;

            // Update call stack based on instruction type
            let instruction = instructions
                .get(&log.current_pc)
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

    // Handle simple length-prefixed names like _9quicksort15quicksort_range
    if name.starts_with('_')
        && name.len() > 1
        && name
            .chars()
            .nth(1)
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
    {
        if let Some(demangled) = try_demangle_length_prefixed(&name[1..]) {
            return demangled;
        }
    }

    // Return original if can't demangle
    name.to_string()
}

/// Try to demangle simple length-prefixed names like "9quicksort15quicksort_range"
fn try_demangle_length_prefixed(name: &str) -> Option<String> {
    let mut result = Vec::new();
    let mut chars = name.chars().peekable();

    while chars.peek().is_some() {
        // Parse length
        let mut len_str = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_ascii_digit() {
                len_str.push(chars.next().unwrap());
            } else {
                break;
            }
        }

        if len_str.is_empty() {
            break;
        }

        let len = len_str.parse::<usize>().ok()?;
        if len == 0 {
            break;
        }

        // Extract identifier
        let ident: String = chars.by_ref().take(len).collect();
        if ident.len() != len {
            break;
        }

        result.push(ident);
    }

    if result.is_empty() {
        None
    } else {
        Some(result.join("::"))
    }
}

/// Clean up demangled names by removing common noise patterns.
fn clean_demangled(name: &str) -> String {
    // Split by :: and process each segment
    let parts: Vec<String> = name
        .split("::")
        .filter_map(|part| {
            // Try to parse length-prefixed identifiers like "_9quicksort" or "9quicksort"
            let cleaned = clean_length_prefixed(part);

            // Filter out hashes and internal names
            if should_filter_part(&cleaned) {
                return None;
            }

            Some(cleaned)
        })
        .collect();

    if parts.is_empty() {
        name.to_string()
    } else {
        // Post-process to remove leading "ue::" artifacts
        let result = parts.join("::");
        result
            .strip_prefix("ue::")
            .map(|s| s.to_string())
            .unwrap_or(result)
    }
}

/// Clean up a single part that may contain length-prefixed identifiers
fn clean_length_prefixed(part: &str) -> String {
    let s = part.strip_prefix('_').unwrap_or(part);

    // Check if it starts with a digit (length-prefixed)
    if s.chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        if let Some(parsed) = try_demangle_length_prefixed(s) {
            return parsed;
        }
    }

    // Check for hash prefix pattern like "guj_14syscall_commit" -> strip "guj_" and parse
    // Pattern: [a-zA-Z]{2,4}_[0-9]+... (short hash followed by underscore and length)
    if let Some(parsed) = try_strip_hash_prefix(s) {
        return parsed;
    }

    part.to_string()
}

/// Try to strip hash prefix like "guj_14syscall_commit" -> "syscall_commit"
fn try_strip_hash_prefix(s: &str) -> Option<String> {
    // Look for pattern: short_hash + underscore + length-prefixed identifier
    // e.g., "guj_14syscall_commit" or "H_17compiler_builtins"
    if let Some(underscore_pos) = s.find('_') {
        let prefix = &s[..underscore_pos];
        let rest = &s[underscore_pos + 1..];

        // Prefix should be short (1-6 chars) and alphanumeric (likely a hash)
        if prefix.len() <= 6
            && prefix.chars().all(|c| c.is_alphanumeric())
            && rest
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
        {
            // Try to parse the rest as length-prefixed
            if let Some(parsed) = try_demangle_length_prefixed(rest) {
                return Some(parsed);
            }
            // If full parsing fails, try to extract just the first identifier
            if let Some(single) = extract_single_length_prefixed(rest) {
                return Some(single);
            }
        }
    }
    None
}

/// Extract a single length-prefixed identifier (doesn't require consuming all input)
fn extract_single_length_prefixed(s: &str) -> Option<String> {
    let mut chars = s.chars().peekable();

    // Parse length
    let mut len_str = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            len_str.push(chars.next().unwrap());
        } else {
            break;
        }
    }

    let len = len_str.parse::<usize>().ok()?;
    if len == 0 {
        return None;
    }

    // Extract identifier (may contain underscores, so can't just use take())
    let remaining: String = chars.collect();
    if remaining.len() >= len {
        Some(remaining[..len].to_string())
    } else {
        None
    }
}

/// Check if a part should be filtered out (hashes, internal names)
fn should_filter_part(part: &str) -> bool {
    // Filter out empty parts
    if part.is_empty() {
        return true;
    }

    // Filter out single-character parts (usually artifacts)
    if part.len() == 1 {
        return true;
    }

    // Filter out common 2-char artifacts from v0 demangling
    if part.len() == 2 && part.chars().all(|c| c.is_lowercase()) {
        return true;
    }

    // Filter out closure markers like {closure#0}
    if part.starts_with('{') {
        return true;
    }

    // Filter out short underscore-prefixed parts (like ___rust, _1, _2)
    if part.starts_with('_') && part.len() <= 8 {
        return true;
    }

    // Filter out parts that look like partial hash_length patterns (e.g., "H_17compi")
    if part.contains("_") {
        if let Some(pos) = part.find('_') {
            let after = &part[pos + 1..];
            // If it's hash_lengthpartial (digits followed by short text), filter it
            if after
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
            {
                let digit_end = after
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(after.len());
                if let Ok(expected_len) = after[..digit_end].parse::<usize>() {
                    let actual_len = after.len() - digit_end;
                    // If actual length doesn't match expected, it's a partial/broken name
                    if actual_len < expected_len {
                        return true;
                    }
                }
            }
        }
    }

    // Filter out hash-like parts (short alphanumeric with mixed case and digits)
    if part.len() <= 10 && part.chars().all(|c| c.is_alphanumeric()) {
        let has_upper = part.chars().any(|c| c.is_uppercase());
        let has_digit = part.chars().any(|c| c.is_ascii_digit());
        let starts_upper = part
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false);

        if (has_upper && has_digit) || (starts_upper && has_digit) {
            return true;
        }
    }

    false
}

/// Try to demangle Rust v0 symbol names (_R prefix).
fn try_demangle_v0(name: &str) -> Option<String> {
    // v0 format: _R followed by path encoding
    // Look for _[0-9]+[identifier] patterns (crate/module names with disambiguators)
    // and bare [0-9]+[identifier] patterns (simple identifiers)
    let mut result = Vec::new();
    let mut i = 0;
    let bytes = name.as_bytes();

    while i < bytes.len() {
        // Look for underscore followed by digit (crate with disambiguator pattern)
        // e.g., _17compiler_builtins or _9quicksort
        if bytes[i] == b'_' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            i += 1; // skip the underscore

            // Parse the length
            let mut len_str = String::new();
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                len_str.push(bytes[i] as char);
                i += 1;
            }

            if let Ok(len) = len_str.parse::<usize>() {
                if len > 0 && len < 200 && i + len <= bytes.len() {
                    let ident = String::from_utf8_lossy(&bytes[i..i + len]).to_string();
                    if !ident.is_empty() {
                        result.push(ident);
                    }
                    i += len;
                    continue;
                }
            }
        }
        // Look for bare digit sequences that are length prefixes
        // Must follow a non-alphanumeric character (like N, C, etc.)
        else if bytes[i].is_ascii_digit() {
            // Check if previous char is a v0 marker (not alphanumeric)
            let prev_ok = i == 0 || !bytes[i - 1].is_ascii_digit();
            if prev_ok {
                let start = i;
                let mut len_str = String::new();
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    len_str.push(bytes[i] as char);
                    i += 1;
                }

                if let Ok(len) = len_str.parse::<usize>() {
                    if len > 0 && len < 200 && i + len <= bytes.len() {
                        let ident = String::from_utf8_lossy(&bytes[i..i + len]).to_string();
                        // Accept if it starts with a letter (valid identifier)
                        if !ident.is_empty()
                            && ident
                                .chars()
                                .next()
                                .map(|c| c.is_alphabetic())
                                .unwrap_or(false)
                        {
                            result.push(ident);
                            i += len;
                            continue;
                        }
                    }
                }
                // If parsing failed, just skip one digit
                i = start + 1;
            } else {
                i += 1;
            }
        } else {
            i += 1;
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

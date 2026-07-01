//! Flamegraph generation for guest RISC-V programs.
//!
//! Tracks function calls during execution and outputs folded stack format
//! compatible with flamegraph.pl and inferno.

use std::collections::HashMap;
use std::io::{self, Write};

use rustc_demangle::demangle as rustc_demangle;

use crate::elf::{Elf, SymbolTable};
use crate::vm::execution::{Executor, ExecutorError, InstructionCache};
use crate::vm::instruction::decoding::Instruction;
use crate::vm::logs::Log;

/// Errors that can occur during flamegraph generation.
#[derive(Debug, thiserror::Error)]
pub enum FlamegraphError {
    /// Instruction not found for a given program counter.
    #[error("instruction not found for a given program counter")]
    InstructionNotFound,
}

/// Errors from the shared execute+flamegraph drive loop.
#[derive(Debug, thiserror::Error)]
pub enum FlamegraphDriveError {
    #[error(transparent)]
    Executor(#[from] ExecutorError),
    #[error(transparent)]
    Flamegraph(#[from] FlamegraphError),
}

/// Generates flamegraph data by tracking function calls during execution.
pub struct FlamegraphGenerator {
    /// Symbol table for address-to-name resolution
    symbols: SymbolTable,
    /// Current call stack (function entry addresses)
    call_stack: Vec<u64>,
    /// Instruction counts per stack state: "main;foo;bar" -> count
    stack_counts: HashMap<String, u64>,
    /// Key stacks by raw hex address instead of resolving through the ELF
    /// symtab (pairs with scripts/enrich_flamegraph.py). Fixed at
    /// construction, since the stack key is formatted on every log.
    raw: bool,
}

impl FlamegraphGenerator {
    /// Create a new flamegraph generator with the given symbol table.
    pub fn new(symbols: SymbolTable, entry_point: u64) -> Self {
        Self::with_mode(symbols, entry_point, false)
    }

    /// Like `new`, but keys folded stacks by raw hex address instead of
    /// resolving through the symtab.
    pub fn new_raw(symbols: SymbolTable, entry_point: u64) -> Self {
        Self::with_mode(symbols, entry_point, true)
    }

    fn with_mode(symbols: SymbolTable, entry_point: u64, raw: bool) -> Self {
        Self {
            symbols,
            call_stack: vec![entry_point], // Start with entry point on stack
            stack_counts: HashMap::new(),
            raw,
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
            .map(|&addr| self.format_frame(addr))
            .collect::<Vec<_>>()
            .join(";")
    }

    /// Format one frame: a raw hex address, or the resolved (demangled)
    /// function name, depending on `self.raw`.
    fn format_frame(&self, address: u64) -> String {
        if self.raw {
            format!("0x{:x}", address)
        } else {
            self.resolve_address(address)
        }
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

            // JAL/JALR with dst=zero doesn't save a return address. This
            // covers both true tail calls AND ordinary intra-function jumps
            // (loop back-edges, if/else, jump tables, self-tail-recursion) —
            // only a jump that actually crosses a function boundary is a
            // tail call; same-function jumps must not mutate the stack.
            Instruction::JumpAndLink { dst: 0, .. } => self.maybe_tail_call(log),
            Instruction::JumpAndLinkRegister { dst: 0, base, .. } if base != 1 => {
                self.maybe_tail_call(log)
            }

            _ => {}
        }
    }

    /// A `dst=0` jump: pop+push only if `next_pc` lands in a different
    /// function than `current_pc` (a true tail call). Same function (or
    /// either address unresolved) is treated as an ordinary jump — no stack
    /// mutation. Symbols with `size == 0` (stripped/ASM) accept any address
    /// at or past their start, so a `dst=0` jump landing exactly on such a
    /// boundary can misattribute — not fixed here, see flamegraph_plan.md
    /// bug #1.
    fn maybe_tail_call(&mut self, log: &Log) {
        let same_function = match (
            self.symbols.lookup(log.current_pc),
            self.symbols.lookup(log.next_pc),
        ) {
            (Some(a), Some(b)) => a.address == b.address,
            _ => false,
        };
        if same_function {
            return;
        }
        if self.call_stack.len() > 1 {
            self.call_stack.pop();
        }
        self.call_stack.push(log.next_pc);
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

/// Drive `executor` to completion (or until `cycle_budget` is hit), feeding
/// every log to `generator` and calling `on_chunk(total_cycles_so_far,
/// generator)` after each processed chunk so callers can implement periodic
/// partial persistence (e.g. checkpoint `write_folded` to disk every N
/// cycles) without reimplementing the drive loop. Returns the total number
/// of cycles processed.
///
/// `cycle_budget` of `None` runs to completion; `Some(n)` stops once at
/// least `n` cycles have been processed (the last chunk may overshoot
/// slightly, since chunks aren't split mid-way).
pub fn drive_with_flamegraph(
    executor: &mut Executor,
    generator: &mut FlamegraphGenerator,
    cycle_budget: Option<u64>,
    mut on_chunk: impl FnMut(u64, &FlamegraphGenerator),
) -> Result<u64, FlamegraphDriveError> {
    // The program's code never changes during execution, so cloning this
    // once up front (not per chunk) means `process_logs` never needs to
    // borrow `executor` again inside the loop — avoiding a conflict with the
    // `&mut self` borrow `resume()`'s returned slice is tied to, without
    // paying to copy every log chunk just to end that borrow early.
    let instructions = executor.instructions.clone();

    let mut total_cycles: u64 = 0;
    while let Some(logs) = executor.resume()? {
        total_cycles += logs.len() as u64;
        generator.process_logs(logs, &instructions)?;
        on_chunk(total_cycles, generator);

        if cycle_budget.is_some_and(|budget| total_cycles >= budget) {
            break;
        }
    }
    Ok(total_cycles)
}

/// Options for [`run_with_flamegraph`].
#[derive(Debug, Clone, Copy, Default)]
pub struct FlamegraphRunOptions {
    /// Stop once at least this many cycles have been processed. `None` runs
    /// to completion.
    pub cycle_budget: Option<u64>,
    /// Key folded stacks by raw hex address instead of resolving through
    /// the symtab.
    pub raw: bool,
}

/// Reusable execute+flamegraph path: build the `SymbolTable`, construct the
/// `Executor`, and drive it via [`drive_with_flamegraph`]. This is what the
/// CLI's `execute --flamegraph` path and any test/caller should use instead
/// of hand-rolling the same `SymbolTable`/`Executor`/drive-loop wiring.
///
/// `on_chunk` is forwarded to `drive_with_flamegraph` for periodic partial
/// persistence; pass `|_, _| {}` if not needed.
pub fn run_with_flamegraph(
    elf_bytes: &[u8],
    program: &Elf,
    private_inputs: Vec<u8>,
    options: FlamegraphRunOptions,
    on_chunk: impl FnMut(u64, &FlamegraphGenerator),
) -> Result<(FlamegraphGenerator, u64), FlamegraphDriveError> {
    let symbols = SymbolTable::parse(elf_bytes);
    let mut generator = if options.raw {
        FlamegraphGenerator::new_raw(symbols, program.entry_point)
    } else {
        FlamegraphGenerator::new(symbols, program.entry_point)
    };
    let mut executor = Executor::new(program, private_inputs)?;
    let total_cycles = drive_with_flamegraph(
        &mut executor,
        &mut generator,
        options.cycle_budget,
        on_chunk,
    )?;
    Ok((generator, total_cycles))
}

/// Demangle a Rust symbol name using the official rustc-demangle crate.
///
/// Uses the alternate format (`{:#}`) to omit the hash suffix for cleaner output.
pub(crate) fn demangle(name: &str) -> String {
    // Use rustc-demangle with alternate format to omit hash
    format!("{:#}", rustc_demangle(name))
}

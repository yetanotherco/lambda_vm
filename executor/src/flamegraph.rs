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
use crate::vm::memory::U64HashMap;

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

/// One node of the call-graph trie. `addr` is the function-entry address of
/// the frame this node represents; `count` is the number of instructions
/// attributed directly to this exact call-stack state.
struct TrieNode {
    parent: u32,
    addr: u64,
    count: u64,
    // u64-keyed by function-entry address; the crate's identity-ish u64 hasher
    // avoids SipHash on every `push` lookup/insert (a hot-path operation).
    children: U64HashMap<u32>,
}

/// Root node index. Its own `parent` field is a self-loop sentinel and is
/// never followed — `pop` refuses to move past it.
const ROOT: u32 = 0;

/// Generates flamegraph data by tracking function calls during execution.
///
/// Instruction counts are stored in a call-graph trie keyed by address, not a
/// demangled string per stack — pushing/popping/counting are all O(1)
/// pointer/hashmap operations independent of call-stack depth. Symbol
/// resolution and demangling happen once per unique address, only when
/// `write_folded` walks the trie.
pub struct FlamegraphGenerator {
    /// Symbol table for address-to-name resolution.
    symbols: SymbolTable,
    /// Arena of trie nodes; index 0 is the root (the entry-point frame).
    nodes: Vec<TrieNode>,
    /// Index into `nodes` of the current call-stack leaf.
    current: u32,
    /// Sum of `count` across all nodes, tracked incrementally.
    total_counted: u64,
    /// `[start, end)` address range of the function most recently resolved in
    /// `maybe_tail_call`. A `dst=0` jump whose endpoints both fall inside it is
    /// an intra-function jump — the overwhelmingly common case — and short-
    /// circuits without the two `SymbolTable` binary searches.
    cached_fn_range: Option<(u64, u64)>,
}

impl FlamegraphGenerator {
    /// Create a new flamegraph generator with the given symbol table.
    pub fn new(symbols: SymbolTable, entry_point: u64) -> Self {
        Self {
            symbols,
            nodes: vec![TrieNode {
                parent: ROOT,
                addr: entry_point,
                count: 0,
                children: U64HashMap::default(),
            }],
            current: ROOT,
            total_counted: 0,
            cached_fn_range: None,
        }
    }

    /// Process a batch of execution logs, updating the call stack and
    /// instruction counts.
    pub fn process_logs(
        &mut self,
        logs: &[Log],
        instructions: &InstructionCache,
    ) -> Result<(), FlamegraphError> {
        for log in logs {
            self.nodes[self.current as usize].count += 1;
            self.total_counted += 1;

            let instruction = instructions
                .get(log.current_pc)
                .copied()
                .ok_or(FlamegraphError::InstructionNotFound)?;
            self.update_stack(log, instruction);
        }
        Ok(())
    }

    /// Resolve an address to a function name, or hex address if unknown.
    fn resolve_address(&self, address: u64) -> String {
        self.symbols
            .lookup(address)
            .map(|sym| demangle(&sym.name))
            .unwrap_or_else(|| format!("0x{:x}", address))
    }

    /// Descend to (or create) the child of the current node keyed by `addr`.
    fn push(&mut self, addr: u64) {
        let current = self.current as usize;
        if let Some(&child) = self.nodes[current].children.get(&addr) {
            self.current = child;
            return;
        }
        let new_idx = self.nodes.len() as u32;
        self.nodes.push(TrieNode {
            parent: self.current,
            addr,
            count: 0,
            children: U64HashMap::default(),
        });
        self.nodes[current].children.insert(addr, new_idx);
        self.current = new_idx;
    }

    /// Move to the parent node. Refuses to pop past the root.
    fn pop(&mut self) {
        if self.current != ROOT {
            self.current = self.nodes[self.current as usize].parent;
        }
    }

    /// Update the call stack based on the instruction type.
    fn update_stack(&mut self, log: &Log, instruction: Instruction) {
        match instruction {
            // Function CALL: JAL with dst=ra (register 1)
            // Saves return address to ra and jumps to offset
            Instruction::JumpAndLink { dst: 1, .. } => self.push(log.next_pc),

            // Function CALL: JALR with dst=ra (register 1)
            // Indirect call through register
            Instruction::JumpAndLinkRegister { dst: 1, .. } => self.push(log.next_pc),

            // Function RETURN: JALR with base=ra (register 1), dst=zero (register 0)
            // This is the standard "ret" instruction (jalr x0, ra, 0)
            Instruction::JumpAndLinkRegister { base, dst, .. } if base == 1 && dst == 0 => {
                self.pop();
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
    /// boundary can misattribute the jump as a tail call into that symbol
    /// instead of an ordinary intra-function jump — not fixed here.
    fn maybe_tail_call(&mut self, log: &Log) {
        // Fast path: both endpoints inside the last-resolved function's range
        // ⇒ an intra-function jump. `lookup_range` guarantees the range holds
        // exactly the addresses that `lookup` resolves to that function, so
        // this is equivalent to two same-function lookups — without running
        // them. Covers loop back-edges, switch arms, self-tail-recursion, etc.
        if let Some((start, end)) = self.cached_fn_range
            && (start..end).contains(&log.current_pc)
            && (start..end).contains(&log.next_pc)
        {
            return;
        }

        let from = self.symbols.lookup_range(log.current_pc);
        if let Some((f, end)) = from {
            self.cached_fn_range = Some((f.address, end));
        }

        // Only a resolved cross-function jump is a real tail call; if either
        // endpoint is unresolved, treat it as an ordinary jump (no mutation),
        // matching the doc comment and this PR's stance against spurious
        // pop+push in unsymbolized code.
        if let (Some((f, _)), Some(t)) = (from, self.symbols.lookup(log.next_pc))
            && f.address != t.address
        {
            self.pop();
            self.push(log.next_pc);
        }
    }

    /// Write the folded stack output to a writer.
    ///
    /// Output format: `stack;frame;names count`
    /// Example: `main;quicksort;partition 12345`
    ///
    /// Symbol resolution/demangling happens here, once per unique address
    /// (memoized), rather than per instruction.
    pub fn write_folded<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let mut name_cache: HashMap<u64, String> = HashMap::new();
        let entries = self.fold(|addr| {
            name_cache
                .entry(addr)
                .or_insert_with(|| self.resolve_address(addr))
                .clone()
        });

        for (stack, count) in entries {
            writeln!(writer, "{} {}", stack, count)?;
        }

        Ok(())
    }

    /// Write folded stack output keyed by raw hex addresses instead of
    /// resolved names (pairs with scripts/enrich_flamegraph.py).
    pub fn write_folded_raw<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let entries = self.fold(|addr| format!("0x{addr:x}"));

        for (stack, count) in entries {
            writeln!(writer, "{stack} {count}")?;
        }
        Ok(())
    }

    /// Fill `path` with `node_idx`'s root-to-node address chain by walking
    /// `parent` pointers — avoids one host stack frame per trie level, since
    /// trie depth mirrors guest call-stack depth and a deeply recursive guest
    /// would otherwise risk overflowing the host stack here.
    fn path_to(&self, node_idx: u32, path: &mut Vec<u64>) {
        path.clear();
        let mut cur = node_idx;
        loop {
            path.push(self.nodes[cur as usize].addr);
            if cur == ROOT {
                break;
            }
            cur = self.nodes[cur as usize].parent;
        }
        path.reverse();
    }

    /// Walk every counted trie node, render its root-to-node address chain
    /// through `render_addr` (memoized name resolution for `write_folded`,
    /// raw hex for `write_folded_raw`), and fold same-rendered-path nodes
    /// (e.g. two different call-site addresses inside the same function)
    /// into summed counts. Returns entries sorted by stack path for
    /// deterministic output.
    fn fold(&self, mut render_addr: impl FnMut(u64) -> String) -> Vec<(String, u64)> {
        let mut path = Vec::new();
        let mut counts: HashMap<String, u64> = HashMap::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            if node.count == 0 {
                continue;
            }
            self.path_to(idx as u32, &mut path);
            let stack = path
                .iter()
                .map(|&addr| render_addr(addr))
                .collect::<Vec<_>>()
                .join(";");
            *counts.entry(stack).or_insert(0) += node.count;
        }

        let mut entries: Vec<_> = counts.into_iter().collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));
        entries
    }

    /// Get the total number of instructions counted so far.
    pub fn total_instructions(&self) -> u64 {
        self.total_counted
    }
}

/// Drive `executor` to completion (or until `cycle_budget` is hit), feeding
/// every log to `generator` and calling `on_chunk(total_cycles_so_far,
/// generator)` after each processed chunk so callers can implement periodic
/// partial persistence (e.g. checkpoint `write_folded` to disk every N
/// cycles) without reimplementing the drive loop. Returns the total number
/// of cycles processed.
///
/// `cycle_budget` of `None` runs to completion; `Some(n)` stops at exactly
/// `n` cycles: the final chunk's cycle limit is capped to the cycles still
/// owed, so the loop neither overshoots nor runs (and discards) a whole extra
/// chunk past the budget.
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
    loop {
        let Some(logs) = executor.resume_budgeted(total_cycles, cycle_budget)? else {
            break;
        };
        total_cycles += logs.len() as u64;
        generator.process_logs(logs, &instructions)?;
        on_chunk(total_cycles, generator);

        if cycle_budget.is_some_and(|budget| total_cycles >= budget) {
            break;
        }
    }
    Ok(total_cycles)
}

/// Reusable execute+flamegraph path: build the `SymbolTable`, construct the
/// `Executor`, and drive it via [`drive_with_flamegraph`]. This is what the
/// CLI's `execute --flamegraph` path and any test/caller should use instead
/// of hand-rolling the same `SymbolTable`/`Executor`/drive-loop wiring.
///
/// `cycle_budget` is forwarded to [`drive_with_flamegraph`]; `on_chunk` is
/// forwarded for periodic partial persistence (pass `|_, _| {}` if not
/// needed).
///
/// The generator is always returned, even on error: a fault partway through
/// a long, uncheckpointed run would otherwise silently discard everything
/// accumulated so far, since this function is the one that owns it.
pub fn run_with_flamegraph(
    elf_bytes: &[u8],
    program: &Elf,
    private_inputs: Vec<u8>,
    cycle_budget: Option<u64>,
    on_chunk: impl FnMut(u64, &FlamegraphGenerator),
) -> (FlamegraphGenerator, Result<u64, FlamegraphDriveError>) {
    let symbols = SymbolTable::parse(elf_bytes);
    let mut generator = FlamegraphGenerator::new(symbols, program.entry_point);
    let mut executor = match Executor::new(program, private_inputs) {
        Ok(executor) => executor,
        Err(e) => return (generator, Err(e.into())),
    };
    let result = drive_with_flamegraph(&mut executor, &mut generator, cycle_budget, on_chunk);
    (generator, result)
}

/// Demangle a Rust symbol name using the official rustc-demangle crate.
///
/// Uses the alternate format (`{:#}`) to omit the hash suffix for cleaner output.
pub fn demangle(name: &str) -> String {
    // Use rustc-demangle with alternate format to omit hash
    format!("{:#}", rustc_demangle(name))
}

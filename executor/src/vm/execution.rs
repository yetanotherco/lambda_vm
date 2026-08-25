use std::{cmp::Ordering, fmt::Debug};

use crate::{
    elf::Elf,
    vm::{
        instruction::{
            decoding::{Instruction, InstructionError},
            execution::ExecutionError,
        },
        logs::Log,
        memory::{Memory, MemoryError, U64HashMap},
        registers::Registers,
    },
};

pub struct ReturnValues {
    pub memory_values: Vec<u8>,
    pub register_values: (i64, i64),
}

/// Result of program execution including logs and predecoded instructions
pub struct ExecutionResult {
    pub return_values: ReturnValues,
    pub logs: Vec<Log>,
    /// Predecoded instructions map (pc -> instruction)
    /// Use this to look up instructions by their PC from the logs
    pub instructions: U64HashMap<Instruction>,
    /// `(hint_id, input)` pairs the guest appended to the hint request log, in
    /// request order. Every one of them was answered during this run; this is a
    /// measurement view, not work left over.
    pub hint_requests: Vec<(u64, [u8; 32])>,
    /// The hint arena this run actually used: the `hints` passed to
    /// [`Executor::new`] plus every slot the executor seeded on demand. The
    /// prover MUST pass this to the trace builder — it is what the private-input
    /// region's bytes encode, and therefore what the initial image has to hold.
    pub hints: Vec<[u8; 32]>,
}

/// Size of each log chunk - balances memory usage vs callback overhead
pub(crate) const CHUNK_SIZE: usize = 100_000;

/// Result of executing one continuation epoch: the logs produced during the
/// epoch and the VM state at the epoch boundary. The boundary state is the
/// starting state of the next epoch.
#[derive(Debug)]
pub struct EpochExecution {
    pub logs: Vec<Log>,
    pub end_pc: u64,
    pub end_registers: Registers,
    pub end_memory: Memory,
}

/// Executor state for chunked execution
pub struct Executor {
    memory: Memory,
    registers: Registers,
    pc: u64,
    pub instructions: InstructionCache,
    logs: Vec<Log>,
}

impl Executor {
    pub fn new(
        program: &Elf,
        private_inputs: Vec<u8>,
        hints: &[[u8; 32]],
    ) -> Result<Self, ExecutorError> {
        let mut memory = Memory::default();
        memory.store_private_inputs(private_inputs, hints)?;
        let instructions = InstructionCache::new(&program.data)?;
        load_program(&program.data, &mut memory)?;

        Ok(Self {
            memory,
            registers: Registers::default(),
            pc: program.entry_point,
            instructions,
            logs: Vec::with_capacity(CHUNK_SIZE),
        })
    }

    /// Resume execution and return next logs. Returns None when program is finished.
    pub fn resume(&mut self) -> Result<Option<&[Log]>, ExecutorError> {
        self.resume_with_limit(CHUNK_SIZE)
    }

    /// Resume execution for the next chunk, capping it so `total_cycles`
    /// never overshoots `cycle_budget`: a full `CHUNK_SIZE` normally, or just
    /// the cycles still owed for the final chunk. `cycle_budget` of `None`
    /// always runs a full chunk. Centralizes the cap math so the flamegraph
    /// and plain execute drive loops can't drift apart on it.
    pub fn resume_budgeted(
        &mut self,
        total_cycles: u64,
        cycle_budget: Option<u64>,
    ) -> Result<Option<&[Log]>, ExecutorError> {
        let limit = cycle_budget
            .map(|budget| ((budget - total_cycles) as usize).min(CHUNK_SIZE))
            .unwrap_or(CHUNK_SIZE);
        self.resume_with_limit(limit)
    }

    /// Current program counter (0 once the program has halted).
    pub fn pc(&self) -> u64 {
        self.pc
    }

    /// Current register state.
    pub fn registers(&self) -> &Registers {
        &self.registers
    }

    /// Current memory state.
    pub fn memory(&self) -> &Memory {
        &self.memory
    }

    /// Run without answering any hint request, so the guest recomputes every
    /// hint in software. Measurement and test hook — see
    /// [`Memory::silence_hints`].
    pub fn silence_hints(&mut self) {
        self.memory.silence_hints();
    }

    /// Drain the initial-memory bytes decided while answering hint requests
    /// since the last call — see [`Memory::take_seeded_bytes`]. A driver that
    /// froze an initial image before running (the continuation prover) must
    /// fold these in after each chunk, before replaying that chunk's memory.
    pub fn take_seeded_bytes(&mut self) -> Vec<(u64, u8)> {
        self.memory.take_seeded_bytes()
    }

    /// The hint arena this run has used so far — the `hints` it started with
    /// plus every slot answered on demand.
    pub fn hint_arena(&self) -> &[[u8; 32]] {
        self.memory.hint_arena()
    }

    /// Resume execution, running at most `limit` cycles, and return the logs
    /// produced. Returns None when the program is finished.
    pub fn resume_with_limit(&mut self, limit: usize) -> Result<Option<&[Log]>, ExecutorError> {
        if self.pc == 0 {
            return Ok(None);
        }

        self.logs.clear();

        while self.pc != 0 && self.logs.len() < limit {
            if !self.pc.is_multiple_of(4) {
                return Err(ExecutorError::InstructionAddressMisaligned(self.pc));
            }
            let instruction = match self.instructions.get(self.pc) {
                Some(&instr) => instr,
                None => {
                    let next_instruction = self.memory.load_word(self.pc)?;
                    Instruction::parse(next_instruction)?
                }
            };
            let log = instruction.run(&mut self.pc, &mut self.registers, &mut self.memory)?;
            self.logs.push(log);
        }

        if self.logs.is_empty() {
            Ok(None)
        } else {
            Ok(Some(&self.logs))
        }
    }

    fn get_return_values(&self) -> Result<ReturnValues, ExecutorError> {
        let memory_return_value = self.memory.read_return_value()?;
        let registers_return_values = self.registers.read_return_values();

        Ok(ReturnValues {
            memory_values: memory_return_value,
            register_values: (
                registers_return_values.0 as i64,
                registers_return_values.1 as i64,
            ),
        })
    }

    /// Get return values after execution is complete (call after resume() returns None)
    pub fn finish(self) -> Result<ReturnValues, ExecutorError> {
        self.get_return_values()
    }

    /// Run to completion and return all logs (consumes executor)
    pub fn run(mut self) -> Result<ExecutionResult, ExecutorError> {
        let mut logs = Vec::with_capacity(CHUNK_SIZE);

        while let Some(chunk) = self.resume()? {
            logs.extend_from_slice(chunk);
        }

        Ok(ExecutionResult {
            return_values: self.get_return_values()?,
            logs,
            instructions: self.instructions.into_instruction_map(),
            hint_requests: self.memory.hint_requests()?,
            hints: self.memory.hint_arena().to_vec(),
        })
    }

    /// Run to completion, splitting execution into epochs of at most `epoch_size`
    /// cycles. Each epoch captures its logs and the VM state at the epoch
    /// boundary, which is the starting state of the next epoch. Consumes the
    /// executor.
    ///
    /// Test/bench helper — the production continuation prover streams epochs via
    /// `resume_with_limit` directly.
    pub fn run_epochs(mut self, epoch_size: usize) -> Result<Vec<EpochExecution>, ExecutorError> {
        assert!(epoch_size > 0, "epoch_size must be greater than zero");

        let mut epochs = Vec::new();
        while let Some(logs) = self.resume_with_limit(epoch_size)? {
            let logs = logs.to_vec();
            epochs.push(EpochExecution {
                logs,
                end_pc: self.pc,
                end_registers: self.registers.clone(),
                end_memory: self.memory.clone(),
            });
        }
        Ok(epochs)
    }
}

/// Run the program and return the hint arena it used, for callers that need the
/// arena BEFORE they start proving — today only the continuation prover, which
/// freezes its initial image and provenance before streaming epochs and so
/// cannot take the arena the run itself produces.
///
/// Every request is answered inline (see `Memory::answer_hint_request`), so this
/// run takes the same cheap in-guest-verify path the proved run will: it is not
/// the software-fallback path. The logs are drained rather than collected —
/// nothing here needs them, and a real block would otherwise materialize tens of
/// millions of `Log`s just to read an arena.
///
/// The monolithic prove path does NOT use this: it takes `ExecutionResult::hints`
/// from the single run it already performs.
pub fn collect_hints(
    program: &Elf,
    private_inputs: Vec<u8>,
) -> Result<Vec<[u8; 32]>, ExecutorError> {
    let mut executor = Executor::new(program, private_inputs, &[])?;
    while executor.resume()?.is_some() {}
    Ok(executor.memory.hint_arena().to_vec())
}

fn load_program(segments: &[crate::elf::Segment], memory: &mut Memory) -> Result<(), MemoryError> {
    for segment in segments {
        for (i, inst) in segment.values.iter().enumerate() {
            let addr = segment.base_addr + (i as u64 * 4);
            memory.store_word(addr, *inst)?;
        }
    }
    Ok(())
}

#[derive(Clone)]
pub struct InstructionSegment {
    base_addr: u64,
    instructions: Vec<Instruction>,
}

impl InstructionSegment {
    fn end_addr(&self) -> u64 {
        self.base_addr + (self.instructions.len() as u64 * 4)
    }
}

#[derive(Clone)]
pub struct InstructionCache {
    segments: Vec<InstructionSegment>,
}

impl InstructionCache {
    /// Creates an InstructionCache from a hashmap of address -> instruction.
    /// Used for testing where we don't have real ELF segments.
    pub fn from_map(map: &U64HashMap<Instruction>) -> Self {
        if map.is_empty() {
            return Self {
                segments: Vec::new(),
            };
        }

        let mut entries: Vec<_> = map.iter().collect();
        entries.sort_by_key(|(addr, _)| *addr);

        let mut segments = Vec::new();
        let mut current_base = *entries[0].0;
        let mut current_instructions = vec![*entries[0].1];

        for (addr, instruction) in entries.into_iter().skip(1) {
            let expected_addr = current_base + (current_instructions.len() as u64 * 4);
            if *addr == expected_addr {
                current_instructions.push(*instruction);
            } else {
                segments.push(InstructionSegment {
                    base_addr: current_base,
                    instructions: current_instructions,
                });
                current_base = *addr;
                current_instructions = vec![*instruction];
            }
        }

        segments.push(InstructionSegment {
            base_addr: current_base,
            instructions: current_instructions,
        });

        Self { segments }
    }

    pub fn new(segments: &[crate::elf::Segment]) -> Result<Self, InstructionError> {
        let mut result = Vec::new();
        for seg in segments.iter().filter(|s| s.is_executable) {
            let instructions = seg
                .values
                .iter()
                .map(|v| Instruction::parse(*v))
                .collect::<Result<Vec<_>, _>>()?;
            result.push(InstructionSegment {
                base_addr: seg.base_addr,
                instructions,
            });
        }
        Ok(Self { segments: result })
    }

    pub fn get(&self, pc: u64) -> Option<&Instruction> {
        // Fast path: most programs have a single executable segment
        let segment = if self.segments.len() == 1 {
            let seg = &self.segments[0];
            if pc < seg.base_addr || pc >= seg.end_addr() {
                return None;
            }
            seg
        } else {
            // Use binary search to find the segment containing pc
            let idx = self
                .segments
                .binary_search_by(|seg| {
                    if pc < seg.base_addr {
                        Ordering::Greater
                    } else if pc >= seg.end_addr() {
                        Ordering::Less
                    } else {
                        Ordering::Equal
                    }
                })
                .ok()?;
            &self.segments[idx]
        };

        let byte_offset = pc - segment.base_addr;
        if !byte_offset.is_multiple_of(4) {
            return None;
        }
        segment.instructions.get((byte_offset / 4) as usize)
    }

    pub fn instruction_count(&self) -> usize {
        self.segments.iter().map(|s| s.instructions.len()).sum()
    }

    pub fn into_instruction_map(self) -> U64HashMap<Instruction> {
        let mut map = U64HashMap::default();
        for segment in self.segments {
            for (i, instruction) in segment.instructions.into_iter().enumerate() {
                let addr = segment.base_addr + (i as u64 * 4);
                map.insert(addr, instruction);
            }
        }
        map
    }
}

/// Decode a `stark::profile_markers::step_marker` hit at `pc`: the marker
/// convention is `addi x0, x0, N` (an `ArithImm` with `dst == 0`, `src == 0`,
/// `op == Add`, `N != 0`), which real code never emits spontaneously since
/// writes to `x0` are always discarded and the canonical NOP is `addi x0, x0,
/// 0`. Returns the marker's `N` if `pc` decodes to one.
pub fn decode_step_marker(instructions: &InstructionCache, pc: u64) -> Option<u32> {
    match instructions.get(pc)? {
        Instruction::ArithImm {
            dst: 0,
            src: 0,
            op: crate::vm::instruction::decoding::ArithOp::Add,
            imm,
        } if *imm != 0 => Some(*imm as u32),
        _ => None,
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ExecutorError {
    #[error("Failed to decode instruction: {0}")]
    Instruction(#[from] InstructionError),
    #[error("Failed to execute instruction: {0}")]
    ExecutionError(#[from] ExecutionError),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
    #[error("Instruction address misaligned: {0:#018x}")]
    InstructionAddressMisaligned(u64),
}

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
}

/// Size of each log chunk - balances memory usage vs callback overhead
const CHUNK_SIZE: usize = 100_000;

/// Executor state for chunked execution
pub struct Executor {
    memory: Memory,
    registers: Registers,
    pc: u64,
    pub instructions: InstructionCache,
    logs: Vec<Log>,
}

impl Executor {
    pub fn new(program: &Elf, private_inputs: Vec<u8>) -> Result<Self, ExecutorError> {
        let mut memory = Memory::default();
        memory.store_private_inputs(private_inputs)?;
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
        if self.pc == 0 {
            return Ok(None);
        }

        self.logs.clear();

        while self.pc != 0 && self.logs.len() < CHUNK_SIZE {
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
        let mut logs = Vec::new();

        // `resume()` fills `self.logs` (a reused chunk buffer) and returns a
        // borrow of it. Drop that borrow immediately (`.is_some()`), then *move*
        // the chunk out with `append` instead of cloning it via
        // `extend_from_slice`: this avoids holding a second copy of every chunk
        // and the per-log clone, lowering peak log memory during proving.
        while self.resume()?.is_some() {
            logs.append(&mut self.logs);
        }

        Ok(ExecutionResult {
            return_values: self.get_return_values()?,
            logs,
            instructions: self.instructions.into_instruction_map(),
        })
    }
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

pub struct InstructionSegment {
    base_addr: u64,
    instructions: Vec<Instruction>,
}

impl InstructionSegment {
    fn end_addr(&self) -> u64 {
        self.base_addr + (self.instructions.len() as u64 * 4)
    }
}

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

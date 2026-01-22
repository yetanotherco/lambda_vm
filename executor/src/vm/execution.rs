use std::{cmp::Ordering, fmt::Debug};

use crate::vm::{
    instruction::{
        decoding::{Instruction, InstructionError},
        execution::ExecutionError,
    },
    logs::Log,
    memory::{Memory, MemoryError},
    registers::Registers,
};

pub struct ReturnValues {
    pub memory_values: Vec<u8>,
    pub register_values: (i64, i64),
}

pub fn run_program(
    segments: &[crate::elf::Segment],
    entrypoint: u64,
    private_inputs: Vec<u8>,
) -> Result<(ReturnValues, Vec<Log>), ExecutorError> {
    let mut memory = Memory::default();
    memory.store_private_inputs(private_inputs)?;
    // Pre-decode all instructions from executable segments
    let instruction_cache = InstructionCache::new(segments)?;
    load_program(segments, &mut memory)?;
    run_from_entrypoint(
        &mut memory,
        entrypoint,
        &instruction_cache,
        instruction_cache.instruction_count(),
    )
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

fn run_from_entrypoint(
    memory: &mut Memory,
    entrypoint: u64,
    instruction_cache: &InstructionCache,
    instruction_count: usize,
) -> Result<(ReturnValues, Vec<Log>), ExecutorError> {
    let mut pc = entrypoint;
    let mut registers = Registers::default();
    // Pre-Allocate logs with an estimated capacity
    let mut logs = Vec::with_capacity(instruction_count * 1000);
    while pc != 0 {
        // Use pre-decoded instruction if available, otherwise fall back to parsing
        let instruction = match instruction_cache.get(pc) {
            Some(&instr) => instr,
            None => {
                let next_instruction = memory.load_word(pc)?;
                Instruction::parse(next_instruction)?
            }
        };
        let log = instruction.run(&mut pc, &mut registers, memory)?;
        logs.push(log);
    }
    println!("Final Register Values:\n {}", &registers);
    let memory_return_value = memory.read_return_value()?;
    let registers_return_values = registers.read_return_values();
    println!("Registers Return Values: {registers_return_values:?}");
    Ok((
        ReturnValues {
            memory_values: memory_return_value,
            register_values: (
                registers_return_values.0 as i64,
                registers_return_values.1 as i64,
            ),
        },
        logs,
    ))
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
}

#[derive(thiserror::Error, Debug)]
pub enum ExecutorError {
    #[error("Failed to decode instruction: {0}")]
    Instruction(#[from] InstructionError),
    #[error("Failed to execute instruction: {0}")]
    ExecutionError(#[from] ExecutionError),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
}

use std::{collections::HashMap, fmt::Debug};

use crate::vm::{
    instruction::{
        decoding::{Instruction, InstructionError},
        execution::ExecutionError,
    },
    logs::Log,
    memory::{Memory, MemoryError, U64HashMap},
    registers::Registers,
};

const MAX_INITIAL_LOG_CAPACITY: usize = 10000;
const LOG_PRE_ALLOCATION_FACTOR: usize = 10;

pub struct ReturnValues {
    pub memory_values: Vec<u8>,
    pub register_values: (i64, i64),
}

pub fn run_program(
    instruction_map: HashMap<u64, u32>,
    entrypoint: u64,
    private_inputs: Vec<u8>,
) -> Result<(ReturnValues, Vec<Log>), ExecutorError> {
    let mut memory = Memory::default();
    memory.store_private_inputs(private_inputs)?;
    // Pre-decode all instructions
    let decoded_instructions = predecode_instructions(&instruction_map);
    let instruction_count = instruction_map.len();
    load_program(instruction_map, &mut memory)?;
    run_from_entrypoint(
        &mut memory,
        entrypoint,
        &decoded_instructions,
        instruction_count,
    )
}

fn predecode_instructions(instruction_map: &HashMap<u64, u32>) -> U64HashMap<Instruction> {
    let mut decoded = U64HashMap::default();
    for (&addr, &raw) in instruction_map {
        // Skip addresses that don't contain valid instructions (data sections)
        if let Ok(instr) = Instruction::parse(raw) {
            decoded.insert(addr, instr);
        }
    }
    decoded
}

fn load_program(
    instruction_map: HashMap<u64, u32>,
    memory: &mut Memory,
) -> Result<(), MemoryError> {
    for (addr, instruction) in instruction_map {
        memory.store_word(addr, instruction)?;
    }
    Ok(())
}

fn run_from_entrypoint(
    memory: &mut Memory,
    entrypoint: u64,
    decoded_instructions: &U64HashMap<Instruction>,
    instruction_count: usize,
) -> Result<(ReturnValues, Vec<Log>), ExecutorError> {
    let mut pc = entrypoint;
    let mut registers = Registers::default();
    // Pre-Allocate logs with an estimated capacity
    let estimated_execution_count = instruction_count
        .saturating_mul(LOG_PRE_ALLOCATION_FACTOR)
        .max(MAX_INITIAL_LOG_CAPACITY);
    let mut logs = Vec::with_capacity(estimated_execution_count);
    while pc != 0 {
        // Use pre-decoded instruction if available, otherwise fall back to parsing
        let instruction = match decoded_instructions.get(&pc) {
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

#[derive(thiserror::Error, Debug)]
pub enum ExecutorError {
    #[error("Failed to decode instruction: {0}")]
    Instruction(#[from] InstructionError),
    #[error("Failed to execute instruction: {0}")]
    ExecutionError(#[from] ExecutionError),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
}

use std::{collections::BTreeMap, fmt::Debug};

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
    pub register_values: (i32, i32),
}

pub fn run_program(
    instruction_map: BTreeMap<u32, u32>,
    entrypoint: u32,
    verbose: bool,
) -> Result<(ReturnValues, Vec<Log>), ExecutorError> {
    let mut memory = Memory::default();
    load_program(instruction_map, &mut memory)?;
    run_from_entrypoint(&mut memory, entrypoint, verbose)
}

fn load_program(
    instruction_map: BTreeMap<u32, u32>,
    memory: &mut Memory,
) -> Result<(), MemoryError> {
    for (addr, instruction) in instruction_map {
        memory.store_word(addr, instruction)?;
    }
    Ok(())
}

fn run_from_entrypoint(
    memory: &mut Memory,
    entrypoint: u32,
    verbose: bool,
) -> Result<(ReturnValues, Vec<Log>), ExecutorError> {
    let mut pc = entrypoint;
    let mut registers = Registers::default();
    let mut logs = Vec::new();
    while pc != 0 {
        let next_instruction = memory.load_word(pc)?;
        let instruction = Instruction::parse(next_instruction)?;
        if verbose {
            //println!("registers: {}", &registers);
            //println!("Executing instruction at 0x{:08x}: {:?}", pc, instruction);
        }
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
                registers_return_values.0 as i32,
                registers_return_values.1 as i32,
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

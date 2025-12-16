use std::{
    collections::BTreeMap,
    fmt::{Debug, Display},
};

use crate::vm::{
    instruction::{
        decoding::{Instruction, InstructionError},
        execution::ExecutionError,
    },
    logs::Log,
    registers::Registers,
};

pub fn run_program(
    instruction_map: BTreeMap<u32, u32>,
    entrypoint: u32,
) -> Result<((i32, i32), Vec<Log>), ExecutorError> {
    let mut memory = Memory::default();
    load_program(instruction_map, &mut memory);
    run_from_entrypoint(&mut memory, entrypoint)
}

fn load_program(instruction_map: BTreeMap<u32, u32>, memory: &mut Memory) {
    for (addr, instruction) in instruction_map {
        memory.0.insert(addr, instruction);
    }
}

fn run_from_entrypoint(
    memory: &mut Memory,
    entrypoint: u32,
) -> Result<((i32, i32), Vec<Log>), ExecutorError> {
    let mut pc = entrypoint;
    let mut registers = Registers::default();
    registers.0[2] = 0xFFFFFFFCu32; // 4GB (Multiple of 4)
    let mut logs = Vec::new();
    while pc != 0 {
        let next_instruction = memory.0[&pc];
        let instruction = Instruction::parse(next_instruction)?;
        let log = instruction.run(&mut pc, &mut registers, memory)?;
        logs.push(log);
    }
    println!("Final Register Values:\n {}", &registers);
    let return_values = (registers.0[10] as i32, registers.0[11] as i32);
    println!("Return Values: {return_values:?}");
    Ok((return_values, logs))
}

// Toy Memory, TODO: Make expandable memory
#[derive(Default, Debug)]
pub struct Memory(pub BTreeMap<u32, u32>);

#[derive(thiserror::Error, Debug)]
pub enum ExecutorError {
    #[error("Failed to decode instruction: {0}")]
    Instruction(#[from] InstructionError),
    #[error("Failed to execute instruction: {0}")]
    ExecutionError(#[from] ExecutionError),
}

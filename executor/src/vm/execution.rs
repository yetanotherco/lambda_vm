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
    pub instructions: U64HashMap<Instruction>,
    logs: Vec<Log>,
}

impl Executor {
    pub fn new(
        instruction_map: HashMap<u64, u32>,
        entrypoint: u64,
        private_inputs: Vec<u8>,
    ) -> Result<Self, ExecutorError> {
        let mut memory = Memory::default();
        memory.store_private_inputs(private_inputs)?;
        let instructions = predecode_instructions(&instruction_map);
        load_program(instruction_map, &mut memory)?;

        Ok(Self {
            memory,
            registers: Registers::default(),
            pc: entrypoint,
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
            let instruction = match self.instructions.get(&self.pc) {
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

    /// Run to completion and return all logs (consumes executor)
    pub fn run(mut self) -> Result<ExecutionResult, ExecutorError> {
        let mut logs = Vec::with_capacity(CHUNK_SIZE);

        while let Some(chunk) = self.resume()? {
            logs.extend_from_slice(chunk);
        }

        println!("Final Register Values:\n {}", &self.registers);
        let memory_return_value = self.memory.read_return_value()?;
        let registers_return_values = self.registers.read_return_values();
        println!("Registers Return Values: {registers_return_values:?}");

        Ok(ExecutionResult {
            return_values: ReturnValues {
                memory_values: memory_return_value,
                register_values: (
                    registers_return_values.0 as i64,
                    registers_return_values.1 as i64,
                ),
            },
            logs,
            instructions: self.instructions,
        })
    }
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

#[derive(thiserror::Error, Debug)]
pub enum ExecutorError {
    #[error("Failed to decode instruction: {0}")]
    Instruction(#[from] InstructionError),
    #[error("Failed to execute instruction: {0}")]
    ExecutionError(#[from] ExecutionError),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
}

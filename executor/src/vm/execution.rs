use std::{
    collections::BTreeMap,
    fmt::{Debug, Display},
};

use crate::vm::{instruction::decoding::Instruction, logs::Log};

pub fn run_program(instruction_map: BTreeMap<u32, u32>, entrypoint: u32) -> ((i32, i32), Vec<Log>) {
    let mut memory = Memory::default();
    load_program(instruction_map, &mut memory);
    run_from_entrypoint(&mut memory, entrypoint)
}

fn load_program(instruction_map: BTreeMap<u32, u32>, memory: &mut Memory) {
    for (addr, instruction) in instruction_map {
        memory.0.insert(addr, instruction);
    }
}

fn run_from_entrypoint(memory: &mut Memory, entrypoint: u32) -> ((i32, i32), Vec<Log>) {
    let mut pc = entrypoint;
    let mut registers = Registers::default();
    registers.0[2] = 0xFFFFFFFFu32; // 4GB
    let mut logs = Vec::new();
    while pc != 0 {
        let next_instruction = memory.0[&pc];
        let instruction = Instruction::parse(next_instruction);
        let log = instruction.run(&mut pc, &mut registers, memory);
        logs.push(log);
    }
    println!("Final Register Values:\n {}", &registers);
    let return_values = (registers.0[10] as i32, registers.0[11] as i32);
    println!("Return Values: {return_values:?}");
    (return_values, logs)
}

// Toy Memory, TODO: Make expandable memory
#[derive(Default, Debug)]
pub struct Memory(pub BTreeMap<u32, u32>);

#[derive(Default, Debug)]
pub struct Registers(pub [u32; 32]);
// Registers:
// 0x zero
// a0-ax function arguments: 0x10 -etc
// 0x1 return address (ra)
//
impl Display for Registers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Registers:")?;
        writeln!(f, "Zero(zero): {}", self.0[0])?;
        writeln!(f, "ReturnAddress(ra): {}", self.0[1])?;
        writeln!(f, "StackPointer(sp): {}", self.0[2])?;
        // Not used for now
        // writeln!(f, "GlobalPointer(gp): {}", self.0[2])?;
        // writeln!(f, "ThreadPointer(tp): {}", self.0[3])?;
        let function_arguments = self.0[10..17]
            .iter()
            .enumerate()
            .map(|(i, val)| match i {
                i @ 0..=1 => format!("a{i} (return value {i}) : {val} "),
                i => format!("a{i}: {val} "),
            })
            .collect::<Vec<_>>()
            .concat();
        writeln!(f, "FunctionArguments: {function_arguments}")?;
        // TODO: Add other registers as we use them
        Ok(())
    }
}

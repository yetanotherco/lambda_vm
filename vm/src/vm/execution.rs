use std::{
    collections::BTreeMap,
    fmt::{Debug, Display},
};

use crate::vm::instructions::{ArithOp, Comparison, Instruction, LoadStoreWidth};

pub fn run_program(instruction_map: BTreeMap<u32, u32>, entrypoint: u32) -> (i32, i32) {
    let mut memory = Memory::default();
    load_program(instruction_map, &mut memory);
    run_from_entrypoint(&mut memory, entrypoint)
}

fn load_program(instruction_map: BTreeMap<u32, u32>, memory: &mut Memory) {
    for (addr, instruction) in instruction_map {
        memory.0.insert(addr, instruction);
    }
}

fn run_from_entrypoint(memory: &mut Memory, entrypoint: u32) -> (i32, i32) {
    let mut pc = entrypoint;
    let mut registers = Registers::default();
    registers.0[2] = 0xFFFFFFFFu32; // 4GB
    while pc != 0 {
        let next_instruction = memory.0[&pc];
        let instruction = Instruction::parse(next_instruction);
        run_instruction(&instruction, &mut registers, &mut pc, memory);
    }
    println!("Final Register Values:\n {}", &registers);
    let return_values = (registers.0[10] as i32, registers.0[11] as i32);
    println!("Return Values: {return_values:?}");
    return_values
}

// Toy Memory, TODO: Make expandable memory
#[derive(Default, Debug)]
struct Memory(BTreeMap<u32, u32>);

#[derive(Default, Debug)]
struct Registers([u32; 32]);
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

fn run_instruction(
    inst: &Instruction,
    registers: &mut Registers,
    pc: &mut u32,
    memory: &mut Memory,
) {
    println!("registers: {:?}", &registers);
    println!("Executing instruction at 0x{:08x}: {:?}", pc.clone(), inst);
    *pc += 4;
    match inst {
        Instruction::ArithImm { dst, src, imm, op } => {
            let (a, b) = (registers.0[*src as usize] as i32, *imm);
            let res = match op {
                ArithOp::Add => a.wrapping_add(b),
                ArithOp::Sub => panic!("SubImm not supported"),
                ArithOp::Xor => a ^ b,
                ArithOp::Or => a | b,
                ArithOp::And => a & b,
                ArithOp::ShiftLeftLogical => a << b,
                ArithOp::ShiftRightLogical => ((a as u32) >> (b as u32)) as i32,
                ArithOp::ShiftRightArith => a >> b,
                ArithOp::SetLessThan => (a < b) as i32,
                ArithOp::SetLessThanU => ((a as u32) < (b as u32)) as i32,
            };
            registers.0[*dst as usize] = res as u32;
        }
        Instruction::JumpAndLinkRegister { dst, base, offset } => {
            let new_pc = ((registers.0[*base as usize] as i32 + offset) & !1) as u32;
            if *dst != 0 {
                registers.0[*dst as usize] = *pc;
            }
            *pc = new_pc;
        }
        Instruction::JumpAndLink { dst, offset } => {
            if *dst != 0 {
                registers.0[*dst as usize] = *pc;
            }
            *pc -= 4;
            *pc = (*pc as i32 + offset) as u32;
        }
        Instruction::Store {
            src,
            offset,
            base,
            width,
        } => {
            let value = registers.0[*src as usize];
            let value = match width {
                LoadStoreWidth::Byte => todo!(),
                LoadStoreWidth::Half => todo!(),
                LoadStoreWidth::Word => value,
            };
            memory
                .0
                .insert(registers.0[*base as usize] + *offset, value);
        }
        Instruction::Load {
            dst,
            offset,
            base,
            width,
        } => {
            let value = memory.0[&((registers.0[*base as usize] as i32 + *offset) as u32)];
            let value = match width {
                LoadStoreWidth::Byte => todo!(),
                LoadStoreWidth::Half => todo!(),
                LoadStoreWidth::Word => value,
            };
            registers.0[*dst as usize] = value;
        }
        Instruction::Branch {
            src1,
            src2,
            cond,
            offset,
        } => {
            let (a, b) = (registers.0[*src1 as usize], registers.0[*src2 as usize]);
            let cmp_result = match cond {
                Comparison::Equal => a == b,
                Comparison::NotEqual => a != b,
                Comparison::LessThan => (a as i32) < (b as i32),
                Comparison::GreaterOrEqual => (a as i32) >= (b as i32),
                Comparison::LessThanUnsigned => a < b,
                Comparison::GreaterOrEqualUnsigned => a >= b,
            };
            if cmp_result {
                *pc -= 4;
                let new_pc = (*pc as i32 + offset) as u32;
                *pc = new_pc;
            }
        }
        Instruction::LoadUpperImm { dst, imm } => registers.0[*dst as usize] = *imm,
        Instruction::AddUpperImmToPc { dst, imm } => registers.0[*dst as usize] = *pc - 4 + *imm,
        Instruction::Arith {
            dst,
            src1,
            src2,
            op,
        } => {
            let (a, b) = (
                registers.0[*src1 as usize] as i32,
                registers.0[*src2 as usize] as i32,
            );
            let res = match op {
                ArithOp::Add => a.wrapping_add(b),
                ArithOp::Sub => a - b,
                ArithOp::Xor => a ^ b,
                ArithOp::Or => a | b,
                ArithOp::And => a & b,
                ArithOp::ShiftLeftLogical => a << b,
                ArithOp::ShiftRightLogical => ((a as u32) >> (b as u32)) as i32,
                ArithOp::ShiftRightArith => a >> b,
                ArithOp::SetLessThan => (a < b) as i32,
                ArithOp::SetLessThanU => ((a as u32) < (b as u32)) as i32,
            };
            registers.0[*dst as usize] = res as u32;
        }
    }
}

use std::ops::Rem;

use crate::vm::{
    execution::{Memory, Registers},
    instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
    logs::Log,
};

const REGULAR_PC_UPDATE: u32 = 4;

impl Instruction {
    /// Runs the given instruction and returns its execution log
    pub fn run(self, pc: &mut u32, registers: &mut Registers, memory: &mut Memory) -> Log {
        println!("registers: {:?}", &registers);
        println!("Executing instruction at 0x{:08x}: {:?}", *pc, self);
        let (new_pc, updated_register, new_register_value) = self.execute(*pc, registers, memory);
        *pc = new_pc;
        if updated_register != 0 {
            registers.0[updated_register as usize] = new_register_value;
        }
        Log {
            instruction: self,
            updated_register_value: new_register_value,
            updated_pc: *pc,
        }
    }

    /// Executes the given instruction returning the new value of pc, the register to be updated and the new value of said register
    fn execute(&self, pc: u32, registers: &Registers, memory: &mut Memory) -> (u32, u32, u32) {
        match self {
            Instruction::ArithImm { dst, src, imm, op } => {
                let op1 = registers.0[*src as usize] as i32;
                if matches!(op, ArithOp::Sub) {
                    panic!("SubImm not supported");
                }
                let res = op.apply(op1, *imm) as u32;
                (pc + REGULAR_PC_UPDATE, *dst, res)
            }
            Instruction::JumpAndLinkRegister { dst, base, offset } => {
                let new_pc = ((registers.0[*base as usize] as i32 + offset) & !1) as u32;
                (new_pc, *dst, pc + REGULAR_PC_UPDATE)
            }
            Instruction::JumpAndLink { dst, offset } => {
                ((pc as i32 + offset) as u32, *dst, pc + REGULAR_PC_UPDATE)
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
                (pc + REGULAR_PC_UPDATE, 0, 0)
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
                (pc + REGULAR_PC_UPDATE, *dst, value)
            }
            Instruction::Branch {
                src1,
                src2,
                cond,
                offset,
            } => {
                let (a, b) = (registers.0[*src1 as usize], registers.0[*src2 as usize]);
                let new_pc = if cond.apply(a, b) {
                    (pc as i32 + *offset) as u32
                } else {
                    pc + REGULAR_PC_UPDATE
                };
                (new_pc, 0, 0)
            }
            Instruction::LoadUpperImm { dst, imm } => (pc + REGULAR_PC_UPDATE, *dst, *imm),
            Instruction::AddUpperImmToPc { dst, imm } => (pc + REGULAR_PC_UPDATE, *dst, pc + *imm),
            Instruction::Arith {
                dst,
                src1,
                src2,
                op,
            } => {
                let a = registers.0[*src1 as usize] as i32;
                let b = registers.0[*src2 as usize] as i32;
                let res = op.apply(a, b) as u32;
                (pc + REGULAR_PC_UPDATE, *dst, res)
            }
        }
    }
}

impl ArithOp {
    fn apply(&self, a: i32, b: i32) -> i32 {
        match self {
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
            ArithOp::Mul => (a as i64 * b as i64) as i32,
            ArithOp::MulHigh => ((a as i64 * b as i64) >> 32) as i32,
            ArithOp::MulHighSignedUnsigned => ((a as i64 * (b as u32) as i64) >> 32) as i32, //?
            ArithOp::MulHighUnsigned => ((a as u64 * b as u64) >> 32) as i32,
            ArithOp::Div => a / b,
            ArithOp::DivUnsigned => (a as u32 / b as u32) as i32,
            ArithOp::Remainder => a.rem(b),
            ArithOp::RemainderUnsigned => (a as u32).rem(b as u32) as i32,
        }
    }
}

impl Comparison {
    fn apply(&self, a: u32, b: u32) -> bool {
        match self {
            Comparison::Equal => a == b,
            Comparison::NotEqual => a != b,
            Comparison::LessThan => (a as i32) < (b as i32),
            Comparison::GreaterOrEqual => (a as i32) >= (b as i32),
            Comparison::LessThanUnsigned => a < b,
            Comparison::GreaterOrEqualUnsigned => a >= b,
        }
    }
}

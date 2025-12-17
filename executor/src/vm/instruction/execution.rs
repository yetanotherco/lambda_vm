use crate::vm::{
    execution::Registers,
    instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
    logs::Log,
    memory::Memory,
};

const REGULAR_PC_UPDATE: u32 = 4;

impl Instruction {
    /// Runs the given instruction and returns its execution log
    pub fn run(
        self,
        pc: &mut u32,
        registers: &mut Registers,
        memory: &mut Memory,
    ) -> Result<Log, ExecutionError> {
        println!("registers: {:?}", &registers);
        println!("Executing instruction at 0x{:08x}: {:?}", *pc, self);
        let log = self.execute(*pc, registers, memory)?;
        // Cleanup zero register in case it was written to
        // TODO: The `Register` struct should handle this, this is a quick and dirty solution
        registers.0[0] = 0;
        *pc = log.next_pc;
        Ok(log)
    }

    /// Executes the given instruction returning the new value of pc, the register to be updated and the new value of said register
    fn execute(
        self,
        pc: u32,
        registers: &mut Registers,
        memory: &mut Memory,
    ) -> Result<Log, ExecutionError> {
        Ok(match self {
            Instruction::ArithImm { dst, src, imm, op } => {
                let op1 = registers.0[src as usize] as i32;
                if matches!(op, ArithOp::Sub) {
                    return Err(ExecutionError::SubImmNotSupported);
                }
                let res = op.apply(op1, imm) as u32;
                registers.0[dst as usize] = res;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc + REGULAR_PC_UPDATE,
                    src1_val: op1 as u32,
                    src2_val: 0,
                    dst_val: res,
                }
            }
            Instruction::JumpAndLinkRegister { dst, base, offset } => {
                let base_value = registers.0[base as usize];
                let new_pc = ((base_value as i32 + offset) & !1) as u32;
                registers.0[dst as usize] = pc + REGULAR_PC_UPDATE;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: new_pc,
                    src1_val: base_value,
                    src2_val: 0,
                    dst_val: pc + REGULAR_PC_UPDATE,
                }
            }
            Instruction::JumpAndLink { dst, offset } => {
                registers.0[dst as usize] = pc + REGULAR_PC_UPDATE;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: (pc as i32 + offset) as u32,
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: pc + REGULAR_PC_UPDATE,
                }
            }
            Instruction::Store {
                src,
                offset,
                base,
                width,
            } => {
                let read_value = registers.0[src as usize];
                let base = registers.0[base as usize];
                let addr = base + offset;
                match width {
                    LoadStoreWidth::Byte => {
                        let value = read_value & 0xFF;
                        memory.store_byte(addr, value as u8);
                    }
                    LoadStoreWidth::Half => {
                        let value = read_value & 0xFFFF;
                        memory.store_half(addr, value as u16)?;
                    }
                    LoadStoreWidth::Word => {
                        memory.store_word(addr, read_value)?;
                    }
                    LoadStoreWidth::ByteUnsigned => {
                        return Err(ExecutionError::StoreBytesUnsignedNotSupported);
                    }
                    LoadStoreWidth::HalfUnsigned => {
                        return Err(ExecutionError::StoreHalfUnsignedNotSupported);
                    }
                };
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc + REGULAR_PC_UPDATE,
                    src1_val: base,
                    src2_val: read_value,
                    dst_val: 0,
                }
            }
            Instruction::Load {
                dst,
                offset,
                base,
                width,
            } => {
                let base = registers.0[base as usize];
                let addr = (base as i32 + offset) as u32;
                let value = match width {
                    LoadStoreWidth::Byte => memory.load_byte(addr) as u32,
                    LoadStoreWidth::Half => memory.load_half(addr)? as u32,
                    LoadStoreWidth::Word => memory.load_word(addr)?,
                    LoadStoreWidth::ByteUnsigned => memory.load_byte(addr) as u32,
                    LoadStoreWidth::HalfUnsigned => memory.load_half(addr)? as u32,
                };
                registers.0[dst as usize] = value;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc + REGULAR_PC_UPDATE,
                    src1_val: base,
                    src2_val: 0,
                    dst_val: value,
                }
            }
            Instruction::Branch {
                src1,
                src2,
                cond,
                offset,
            } => {
                let (a, b) = (registers.0[src1 as usize], registers.0[src2 as usize]);
                let new_pc = if cond.apply(a, b) {
                    (pc as i32 + offset) as u32
                } else {
                    pc + REGULAR_PC_UPDATE
                };
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: new_pc,
                    src1_val: a,
                    src2_val: b,
                    dst_val: 0,
                }
            }
            Instruction::LoadUpperImm { dst, imm } => {
                registers.0[dst as usize] = imm;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc + REGULAR_PC_UPDATE,
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: imm,
                }
            }
            Instruction::AddUpperImmToPc { dst, imm } => {
                registers.0[dst as usize] = pc.wrapping_add(imm);
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc + REGULAR_PC_UPDATE,
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: pc.wrapping_add(imm),
                }
            }
            Instruction::Arith {
                dst,
                src1,
                src2,
                op,
            } => {
                let a = registers.0[src1 as usize];
                let b = registers.0[src2 as usize];
                let res = op.apply(a as i32, b as i32) as u32;
                registers.0[dst as usize] = res;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc + REGULAR_PC_UPDATE,
                    src1_val: a,
                    src2_val: b,
                    dst_val: res,
                }
            }
        })
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
            ArithOp::MulHigh => (((a as i64) * (b as i64)) >> 32) as i32,
            ArithOp::MulHighSignedUnsigned => ((a as i64 * (b as u32) as i64) >> 32) as i32,
            ArithOp::MulHighUnsigned => (((a as u32) as u64 * (b as u32) as u64) >> 32) as i32,
            ArithOp::Div => {
                if b == 0 {
                    u32::MAX as i32
                } else {
                    a.wrapping_div(b)
                }
            }
            ArithOp::DivUnsigned => {
                if b == 0 {
                    u32::MAX as i32
                } else {
                    (a as u32).wrapping_div(b as u32) as i32
                }
            }
            ArithOp::Remainder => {
                if b == 0 {
                    a
                } else {
                    a.wrapping_rem(b)
                }
            }
            ArithOp::RemainderUnsigned => {
                if b == 0 {
                    a
                } else {
                    (a as u32).wrapping_rem(b as u32) as i32
                }
            }
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

#[derive(thiserror::Error, Debug)]
pub enum ExecutionError {
    #[error("Sub immediate instruction is not supported")]
    SubImmNotSupported,
    #[error("Store bytes unsigned instruction is not supported")]
    StoreBytesUnsignedNotSupported,
    #[error("Store half unsigned instruction is not supported")]
    StoreHalfUnsignedNotSupported,
    #[error("Memory error: {0}")]
    MemoryError(#[from] crate::vm::memory::MemoryError),
}

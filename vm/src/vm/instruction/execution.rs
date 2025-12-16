use crate::vm::{
    execution::Memory,
    instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
    logs::Log,
    registers::Registers,
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
                let op1 = registers.read(src) as i32;
                if matches!(op, ArithOp::Sub) {
                    return Err(ExecutionError::SubImmNotSupported);
                }
                let res = op.apply(op1, imm) as u32;
                registers.write(dst, res);
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
                let base_value = registers.read(base);
                let new_pc = ((base_value as i32 + offset) & !1) as u32;
                registers.write(dst, pc + REGULAR_PC_UPDATE);
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
                registers.write(dst, pc + REGULAR_PC_UPDATE);
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
                let read_value = registers.read(src);
                let base = registers.read(base);
                let addr = base + offset;
                match width {
                    LoadStoreWidth::Byte => {
                        let value = read_value & 0xFF;
                        let aligned_addr = addr - (addr % 4);
                        let aligned_value = value << ((addr % 4) * 8);
                        let previous_value =
                            memory.0.get(&aligned_addr).cloned().unwrap_or_default();
                        let new_value =
                            (previous_value & !(0xFF << ((addr % 4) * 8))) | aligned_value;
                        memory.0.insert(aligned_addr, new_value);
                    }
                    LoadStoreWidth::Half => todo!(),
                    LoadStoreWidth::Word => {
                        if !addr.is_multiple_of(4) {
                            unimplemented!(
                                "Store at unaligned memory by word at address 0x{:08x}",
                                addr
                            );
                        }
                        memory.0.insert(addr, read_value);
                    }
                    LoadStoreWidth::ByteUnsigned => {
                        return Err(ExecutionError::StoreBytesUnsignedNotSupported);
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
                let base = registers.read(base);
                let addr = (base as i32 + offset) as u32;
                let value = match width {
                    LoadStoreWidth::Byte => todo!(),
                    LoadStoreWidth::Half => todo!(),
                    LoadStoreWidth::Word => {
                        if !addr.is_multiple_of(4) {
                            unimplemented!("Load at unaligned memory at address 0x{:08x}", addr);
                        }
                        memory.0.get(&addr).cloned().unwrap_or_default()
                    }
                    LoadStoreWidth::ByteUnsigned => {
                        let aligned_addr = addr - (addr % 4);
                        let value = memory.0[&aligned_addr];
                        value & (0xFF << ((addr % 4) * 8))
                    }
                };
                registers.write(dst, value);
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
                let (a, b) = (registers.read(src1), registers.read(src2));
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
                registers.write(dst, imm);
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
                registers.write(dst, pc + imm);
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc + REGULAR_PC_UPDATE,
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: pc + imm,
                }
            }
            Instruction::Arith {
                dst,
                src1,
                src2,
                op,
            } => {
                let a = registers.read(src1);
                let b = registers.read(src2);
                let res = op.apply(a as i32, b as i32) as u32;
                registers.write(dst, res);
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
}

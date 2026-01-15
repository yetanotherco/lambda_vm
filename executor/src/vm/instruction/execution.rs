use crate::vm::{
    instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
    logs::Log,
    memory::Memory,
    registers::Registers,
};

const REGULAR_PC_UPDATE: u32 = 4;

enum SyscallNumbers {
    Print = 1,
    Panic = 2,
    Commit = 3,
    GetPrivateInputs = 4,
    Halt = 5,
}

impl TryFrom<u32> for SyscallNumbers {
    type Error = ();
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(SyscallNumbers::Print),
            2 => Ok(SyscallNumbers::Panic),
            3 => Ok(SyscallNumbers::Commit),
            4 => Ok(SyscallNumbers::GetPrivateInputs),
            5 => Ok(SyscallNumbers::Halt),
            _ => Err(()),
        }
    }
}

impl Instruction {
    /// Runs the given instruction and returns its execution log
    pub fn run(
        self,
        pc: &mut u32,
        registers: &mut Registers,
        memory: &mut Memory,
    ) -> Result<Log, ExecutionError> {
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
                let op1 = registers.read(src)? as i32;
                if matches!(op, ArithOp::Sub) {
                    return Err(ExecutionError::SubImmNotSupported);
                }
                let res = op.apply(op1, imm) as u32;
                registers.write(dst, res)?;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: op1 as u32,
                    src2_val: 0,
                    dst_val: res,
                }
            }
            Instruction::JumpAndLinkRegister { dst, base, offset } => {
                let base_value = registers.read(base)?;
                let new_pc = (((base_value as i32).wrapping_add(offset)) & !1) as u32;
                registers.write(dst, pc.wrapping_add(REGULAR_PC_UPDATE))?;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: new_pc,
                    src1_val: base_value,
                    src2_val: 0,
                    dst_val: pc.wrapping_add(REGULAR_PC_UPDATE),
                }
            }
            Instruction::JumpAndLink { dst, offset } => {
                registers.write(dst, pc.wrapping_add(REGULAR_PC_UPDATE))?;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: (pc as i32).wrapping_add(offset) as u32,
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: pc.wrapping_add(REGULAR_PC_UPDATE),
                }
            }
            Instruction::Store {
                src,
                offset,
                base,
                width,
            } => {
                let read_value = registers.read(src)?;
                let base = registers.read(base)?;
                let addr = (base as i32).wrapping_add(offset) as u32;
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
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
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
                let base = registers.read(base)?;
                let addr = (base as i32).wrapping_add(offset) as u32;
                let value = match width {
                    LoadStoreWidth::Byte => memory.load_byte(addr) as i8 as i32 as u32,
                    LoadStoreWidth::Half => memory.load_half(addr)? as i16 as i32 as u32,
                    LoadStoreWidth::Word => memory.load_word(addr)?,
                    LoadStoreWidth::ByteUnsigned => memory.load_byte(addr) as u32,
                    LoadStoreWidth::HalfUnsigned => memory.load_half(addr)? as u32,
                };
                registers.write(dst, value)?;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
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
                let (a, b) = (registers.read(src1)?, registers.read(src2)?);
                let new_pc = if cond.apply(a, b) {
                    (pc as i32).wrapping_add(offset) as u32
                } else {
                    pc.wrapping_add(REGULAR_PC_UPDATE)
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
                registers.write(dst, imm)?;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: imm,
                }
            }
            Instruction::AddUpperImmToPc { dst, imm } => {
                registers.write(dst, pc.wrapping_add(imm))?;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
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
                let a = registers.read(src1)?;
                let b = registers.read(src2)?;
                let res = op.apply(a as i32, b as i32) as u32;
                registers.write(dst, res)?;
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: a,
                    src2_val: b,
                    dst_val: res,
                }
            }
            Instruction::CSR {
                csr: _,
                src: _,
                dst: _,
                op: _,
            } => {
                // Todo: CSR are currently no-ops
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: 0,
                }
            }
            Instruction::EcallEbreak => {
                let syscall_number = registers.read(17)?; // a7
                let syscall_number = SyscallNumbers::try_from(syscall_number)
                    .map_err(|_| ExecutionError::UnknownSyscall(syscall_number))?;
                match syscall_number {
                    SyscallNumbers::Print => {
                        // print
                        // For now this is just a mechanism to print
                        // It is not the correct implementation of ecall/ebreak
                        let pointer = registers.read(10)?;
                        let len = registers.read(11)?;
                        let mut bytes = vec![];
                        for i in 0..len {
                            bytes.push(memory.load_byte(pointer + i));
                        }
                        let value =
                            str::from_utf8(&bytes).map_err(|_| ExecutionError::IncorrectMessage)?;
                        println!("PRINT VM: {}", value);
                    }
                    SyscallNumbers::Panic => {
                        // panic
                        let pointer = registers.read(10)?;
                        let len = registers.read(11)?;
                        let mut bytes = vec![];
                        for i in 0..len {
                            bytes.push(memory.load_byte(pointer + i));
                        }
                        let value =
                            str::from_utf8(&bytes).map_err(|_| ExecutionError::IncorrectMessage)?;
                        return Err(ExecutionError::Panic(value.to_owned()));
                    }
                    SyscallNumbers::Commit => {
                        // commit
                        let pointer = registers.read(10)?;
                        let len = registers.read(11)?;
                        memory.commit_public_output(pointer, len)?;
                    }
                    SyscallNumbers::GetPrivateInputs => {
                        // get private inputs
                        let pointer = registers.read(10)?;
                        let private_inputs = memory.load_private_inputs()?;
                        for (i, byte) in private_inputs.iter().enumerate() {
                            memory.store_byte(pointer + i as u32, *byte);
                        }
                    }
                    SyscallNumbers::Halt => {
                        // halt
                        return Ok(Log {
                            instruction: self,
                            current_pc: pc,
                            next_pc: 0, // We halt by setting pc to 0
                            src1_val: 0,
                            src2_val: 0,
                            dst_val: 0,
                        });
                    }
                }
                Log {
                    instruction: self,
                    current_pc: pc,
                    next_pc: pc + REGULAR_PC_UPDATE,
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: 0,
                }
            }
        })
    }
}

impl ArithOp {
    fn apply(&self, a: i32, b: i32) -> i32 {
        match self {
            ArithOp::Add => a.wrapping_add(b),
            ArithOp::Sub => a.wrapping_sub(b),
            ArithOp::Xor => a ^ b,
            ArithOp::Or => a | b,
            ArithOp::And => a & b,
            ArithOp::ShiftLeftLogical => a.wrapping_shl(b as u32),
            ArithOp::ShiftRightLogical => ((a as u32).wrapping_shr(b as u32)) as i32,
            ArithOp::ShiftRightArith => a.wrapping_shr(b as u32),
            ArithOp::SetLessThan => (a < b) as i32,
            ArithOp::SetLessThanU => ((a as u32) < (b as u32)) as i32,
            ArithOp::Mul => ((a as i64).wrapping_mul(b as i64)) as i32,
            ArithOp::MulHigh => (((a as i64).wrapping_mul(b as i64)) >> 32) as i32,
            ArithOp::MulHighSignedUnsigned => {
                (((a as i64).wrapping_mul(b as u32 as i64)) >> 32) as i32
            }
            ArithOp::MulHighUnsigned => {
                ((((a as u32) as u64).wrapping_mul(b as u32 as u64)) >> 32) as i32
            }
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
    #[error("Register error: {0}")]
    RegisterError(#[from] crate::vm::registers::RegisterError),
    #[error("Unknown syscall number: {0}")]
    UnknownSyscall(u32),
    #[error("Panic called with message: {0}")]
    Panic(String),
    #[error("Incorrect message encoding")]
    IncorrectMessage,
}

use crate::vm::{
    instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
    logs::Log,
    memory::Memory,
    registers::Registers,
};

const REGULAR_PC_UPDATE: u64 = 4;

enum SyscallNumbers {
    Print = 1,
    Panic = 2,
    Commit = 3,
    GetPrivateInputs = 4,
    Halt = 5,
}

impl TryFrom<u64> for SyscallNumbers {
    type Error = ();
    fn try_from(value: u64) -> Result<Self, Self::Error> {
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
        pc: &mut u64,
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
        pc: u64,
        registers: &mut Registers,
        memory: &mut Memory,
    ) -> Result<Log, ExecutionError> {
        Ok(match self {
            Instruction::ArithImm { dst, src, imm, op } => {
                let op1 = registers.read(src)? as i64;
                if matches!(op, ArithOp::Sub) {
                    return Err(ExecutionError::SubImmNotSupported);
                }
                let res = op.apply(op1, imm as i64) as u64;
                registers.write(dst, res)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: op1 as u64,
                    src2_val: 0,
                    dst_val: res,
                }
            }
            Instruction::ArithImmW { dst, src, imm, op } => {
                // W-suffix: operate on lower 32 bits, sign-extend result to 64 bits
                let op1 = registers.read(src)? as i32;
                if matches!(op, ArithOp::Sub) {
                    return Err(ExecutionError::SubImmNotSupported);
                }
                let res32 = op.apply_word(op1, imm)?;
                let res = res32 as i64 as u64; // Sign-extend to 64 bits
                registers.write(dst, res)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: op1 as u64,
                    src2_val: 0,
                    dst_val: res,
                }
            }
            Instruction::JumpAndLinkRegister { dst, base, offset } => {
                let base_value = registers.read(base)?;
                let new_pc = (((base_value as i64).wrapping_add(offset as i64)) & !1) as u64;
                registers.write(dst, pc.wrapping_add(REGULAR_PC_UPDATE))?;
                Log {
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
                    current_pc: pc,
                    next_pc: (pc as i64).wrapping_add(offset as i64) as u64,
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
                let addr = (base as i64).wrapping_add(offset as i64) as u64;
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
                        memory.store_word(addr, read_value as u32)?;
                    }
                    LoadStoreWidth::DoubleWord => {
                        memory.store_doubleword(addr, read_value)?;
                    }
                    LoadStoreWidth::ByteUnsigned => {
                        return Err(ExecutionError::StoreBytesUnsignedNotSupported);
                    }
                    LoadStoreWidth::HalfUnsigned => {
                        return Err(ExecutionError::StoreHalfUnsignedNotSupported);
                    }
                    LoadStoreWidth::WordUnsigned => {
                        return Err(ExecutionError::StoreWordUnsignedNotSupported);
                    }
                };
                Log {
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
                let addr = (base as i64).wrapping_add(offset as i64) as u64;
                let value = match width {
                    // RV64: LB sign-extends to 64 bits
                    LoadStoreWidth::Byte => (memory.load_byte(addr) as i8) as i64 as u64,
                    // RV64: LH sign-extends to 64 bits
                    LoadStoreWidth::Half => (memory.load_half(addr)? as i16) as i64 as u64,
                    // RV64: LW sign-extends to 64 bits
                    LoadStoreWidth::Word => (memory.load_word(addr)? as i32) as i64 as u64,
                    // RV64: LD loads 64 bits
                    LoadStoreWidth::DoubleWord => memory.load_doubleword(addr)?,
                    // RV64: LBU zero-extends to 64 bits
                    LoadStoreWidth::ByteUnsigned => memory.load_byte(addr) as u64,
                    // RV64: LHU zero-extends to 64 bits
                    LoadStoreWidth::HalfUnsigned => memory.load_half(addr)? as u64,
                    // RV64: LWU zero-extends to 64 bits
                    LoadStoreWidth::WordUnsigned => memory.load_word(addr)? as u64,
                };
                registers.write(dst, value)?;
                Log {
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
                    (pc as i64).wrapping_add(offset as i64) as u64
                } else {
                    pc.wrapping_add(REGULAR_PC_UPDATE)
                };
                Log {
                    current_pc: pc,
                    next_pc: new_pc,
                    src1_val: a,
                    src2_val: b,
                    dst_val: 0,
                }
            }
            Instruction::LoadUpperImm { dst, imm } => {
                // RV64: LUI sign-extends the 32-bit result to 64 bits
                let value = (imm as i32) as i64 as u64;
                registers.write(dst, value)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: value,
                }
            }
            Instruction::AddUpperImmToPc { dst, imm } => {
                // RV64: AUIPC adds sign-extended imm to PC
                let value = pc.wrapping_add((imm as i32) as i64 as u64);
                registers.write(dst, value)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: value,
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
                let res = op.apply(a as i64, b as i64) as u64;
                registers.write(dst, res)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: a,
                    src2_val: b,
                    dst_val: res,
                }
            }
            Instruction::ArithW {
                dst,
                src1,
                src2,
                op,
            } => {
                // W-suffix: operate on lower 32 bits, sign-extend result to 64 bits
                let a = registers.read(src1)? as i32;
                let b = registers.read(src2)? as i32;
                let res32 = op.apply_word(a, b)?;
                let res = res32 as i64 as u64; // Sign-extend to 64 bits
                registers.write(dst, res)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: a as u64,
                    src2_val: b as u64,
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
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: 0,
                }
            }
            Instruction::EcallEbreak => {
                let syscall_raw = registers.read(17)?; // a7
                let syscall_number = SyscallNumbers::try_from(syscall_raw)
                    .map_err(|_| ExecutionError::UnknownSyscall(syscall_raw))?;
                let mut src2_val = 0u64;
                let mut dst_val = 0u64;
                match syscall_number {
                    SyscallNumbers::Print => {
                        // print
                        // For now this is just a mechanism to print
                        // It is not the correct implementation of ecall/ebreak
                        let pointer = registers.read(10)?;
                        let len = registers.read(11)?;
                        let bytes = memory.load_bytes(pointer, len);
                        let value =
                            str::from_utf8(&bytes).map_err(|_| ExecutionError::IncorrectMessage)?;
                        println!("PRINT VM: {}", value);
                    }
                    SyscallNumbers::Panic => {
                        // panic
                        let pointer = registers.read(10)?;
                        let len = registers.read(11)?;
                        let bytes = memory.load_bytes(pointer, len);
                        let value =
                            str::from_utf8(&bytes).map_err(|_| ExecutionError::IncorrectMessage)?;
                        return Err(ExecutionError::Panic(value.to_owned()));
                    }
                    SyscallNumbers::Commit => {
                        // commit: write(fd, buf_addr, count) per POSIX convention
                        // x10 = fd (must be 1 for stdout, validated by COMMIT chip)
                        // x11 = buf_addr (buffer address in memory)
                        // x12 = count (number of bytes to write)
                        let _fd = registers.read(10)?;
                        let buf_addr = registers.read(11)?;
                        let count = registers.read(12)?;
                        memory.commit_public_output(buf_addr, count)?;
                        src2_val = buf_addr;
                        dst_val = count;
                    }
                    SyscallNumbers::GetPrivateInputs => {
                        // get private inputs
                        let pointer = registers.read(10)?;
                        let private_inputs = memory.load_private_inputs()?;
                        for (i, byte) in private_inputs.iter().enumerate() {
                            memory.store_byte(pointer + i as u64, *byte);
                        }
                    }
                    SyscallNumbers::Halt => {
                        // halt
                        return Ok(Log {
                            current_pc: pc,
                            next_pc: 0, // We halt by setting pc to 0
                            src1_val: SyscallNumbers::Halt as u64,
                            src2_val: 0,
                            dst_val: 0,
                        });
                    }
                }
                Log {
                    current_pc: pc,
                    next_pc: pc + REGULAR_PC_UPDATE,
                    src1_val: syscall_raw,
                    src2_val,
                    dst_val,
                }
            }
            Instruction::Fence => {
                // FENCE is a memory barrier - in single-threaded, in-order execution it's a no-op
                Log {
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
    /// 64-bit arithmetic operations (RV64I base)
    fn apply(&self, a: i64, b: i64) -> i64 {
        match self {
            ArithOp::Add => a.wrapping_add(b),
            ArithOp::Sub => a.wrapping_sub(b),
            ArithOp::Xor => a ^ b,
            ArithOp::Or => a | b,
            ArithOp::And => a & b,
            // RV64: shift amount is 6 bits (0-63)
            ArithOp::ShiftLeftLogical => a.wrapping_shl((b as u32) & 0x3F),
            ArithOp::ShiftRightLogical => ((a as u64).wrapping_shr((b as u32) & 0x3F)) as i64,
            ArithOp::ShiftRightArith => a.wrapping_shr((b as u32) & 0x3F),
            ArithOp::SetLessThan => (a < b) as i64,
            ArithOp::SetLessThanU => ((a as u64) < (b as u64)) as i64,
            // RV64: 64×64 multiplication
            ArithOp::Mul => a.wrapping_mul(b),
            ArithOp::MulHigh => (((a as i128).wrapping_mul(b as i128)) >> 64) as i64,
            ArithOp::MulHighSignedUnsigned => {
                (((a as i128).wrapping_mul(b as u64 as i128)) >> 64) as i64
            }
            ArithOp::MulHighUnsigned => {
                (((a as u64 as u128).wrapping_mul(b as u64 as u128)) >> 64) as i64
            }
            // RV64: 64÷64 division
            ArithOp::Div => {
                if b == 0 {
                    -1i64
                } else {
                    a.wrapping_div(b)
                }
            }
            ArithOp::DivUnsigned => {
                if b == 0 {
                    u64::MAX as i64
                } else {
                    (a as u64).wrapping_div(b as u64) as i64
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
                    (a as u64).wrapping_rem(b as u64) as i64
                }
            }
        }
    }

    /// 32-bit arithmetic operations with sign extension (RV64 W-suffix)
    fn apply_word(&self, a: i32, b: i32) -> Result<i32, ExecutionError> {
        Ok(match self {
            ArithOp::Add => a.wrapping_add(b),
            ArithOp::Sub => a.wrapping_sub(b),
            // W-suffix shifts use 5-bit shift amount
            ArithOp::ShiftLeftLogical => a.wrapping_shl((b as u32) & 0x1F),
            ArithOp::ShiftRightLogical => ((a as u32).wrapping_shr((b as u32) & 0x1F)) as i32,
            ArithOp::ShiftRightArith => a.wrapping_shr((b as u32) & 0x1F),
            // MULW: 32×32→32 (low bits), sign-extend
            ArithOp::Mul => a.wrapping_mul(b),
            // DIVW: 32÷32
            ArithOp::Div => {
                if b == 0 {
                    -1i32
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
            // These operations are not valid for W-suffix instructions
            _ => return Err(ExecutionError::InvalidWSuffixOperation(*self)),
        })
    }
}

impl Comparison {
    fn apply(&self, a: u64, b: u64) -> bool {
        match self {
            Comparison::Equal => a == b,
            Comparison::NotEqual => a != b,
            Comparison::LessThan => (a as i64) < (b as i64),
            Comparison::GreaterOrEqual => (a as i64) >= (b as i64),
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
    #[error("Store word unsigned instruction is not supported")]
    StoreWordUnsignedNotSupported,
    #[error("Memory error: {0}")]
    MemoryError(#[from] crate::vm::memory::MemoryError),
    #[error("Register error: {0}")]
    RegisterError(#[from] crate::vm::registers::RegisterError),
    #[error("Unknown syscall number: {0}")]
    UnknownSyscall(u64),
    #[error("Panic called with message: {0}")]
    Panic(String),
    #[error("Incorrect message encoding")]
    IncorrectMessage,
    #[error("Invalid W-suffix operation: {0:?}")]
    InvalidWSuffixOperation(ArithOp),
}

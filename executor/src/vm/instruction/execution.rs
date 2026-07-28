use crate::vm::{
    instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
    logs::Log,
    memory::{Memory, MemoryError},
    registers::Registers,
};

const REGULAR_PC_UPDATE: u64 = 4;

pub enum SyscallNumbers {
    // Placeholder discriminant. The actual syscall value is KECCAK_SYSCALL_NUMBER.
    KeccakPermute = 0,
    Print = 1,
    Panic = 2,
    Commit = 64,
    Halt = 93,
    // Placeholder discriminant. The actual syscall value is ECSM_SYSCALL_NUMBER.
    Ecsm = 94,
    // Placeholder discriminant. The actual syscall value is HINT_SYSCALL_NUMBER.
    // BENCH ONLY: non-constraining hint (host computes modular inverse/sqrt, guest verifies).
    Hint = 95,
}

/// Syscall number for KeccakPermute (u64::MAX - 1 = 0xFFFF_FFFF_FFFF_FFFE).
///
/// Cannot be an enum discriminant because it exceeds isize::MAX.
pub const KECCAK_SYSCALL_NUMBER: u64 = u64::MAX - 1;
const KECCAK_STATE_BYTES: u64 = 25 * 8;

/// Syscall number for the ECSM (elliptic-curve scalar multiply) accelerator.
///
/// The spec uses ECALL number `-11`; interpreted as an unsigned 64-bit value that is
/// `u64::MAX - 10 = 0xFFFF_FFFF_FFFF_FFF5`, which the ECSM core table puts on the `Ecall`
/// bus as `[lo32, hi32] = [2^32 - 11, 2^32 - 1]`.
pub const ECSM_SYSCALL_NUMBER: u64 = u64::MAX - 10;

/// Syscall number for the non-constraining `Hint` ecall (BENCH ONLY).
///
/// The host computes a modular inverse or square root and writes it back to the
/// guest, which must verify it (e.g. `x·inv == 1`). This adds no in-circuit
/// correctness constraint of its own — it exists to measure the cost of the
/// hint-then-verify pattern versus computing the operation in the guest.
pub const HINT_SYSCALL_NUMBER: u64 = u64::MAX - 20;

/// Hint operation selector passed in `a0`.
pub const HINT_FIELD_INV: u64 = 0; // secp256k1 base-field inverse (mod p)
pub const HINT_SCALAR_INV: u64 = 1; // secp256k1 scalar-field inverse (mod n)
pub const HINT_FIELD_SQRT: u64 = 2; // secp256k1 base-field square root

/// `2^32`. ECSM memory operands must not overflow their lower 32-bit address limb when the
/// largest per-access offset is added: the 32-byte operands reach offset +31 (last byte).
const LOW_LIMB: u64 = 1 << 32;

impl TryFrom<u64> for SyscallNumbers {
    type Error = ();
    fn try_from(value: u64) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(SyscallNumbers::Print),
            2 => Ok(SyscallNumbers::Panic),
            64 => Ok(SyscallNumbers::Commit),
            93 => Ok(SyscallNumbers::Halt),
            v if v == KECCAK_SYSCALL_NUMBER => Ok(SyscallNumbers::KeccakPermute),
            v if v == ECSM_SYSCALL_NUMBER => Ok(SyscallNumbers::Ecsm),
            v if v == HINT_SYSCALL_NUMBER => Ok(SyscallNumbers::Hint),
            _ => Err(()),
        }
    }
}

/// A syscall that drives a specialized in-circuit accelerator chip.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Accelerator {
    Keccak,
    Ecsm,
}

impl SyscallNumbers {
    /// The accelerator this syscall drives, if any. Exhaustive `match self`:
    /// adding a `SyscallNumbers` variant is a compile error here, so a new
    /// accelerator can't be silently missed by counters that consume this.
    pub fn accelerator(self) -> Option<Accelerator> {
        match self {
            SyscallNumbers::KeccakPermute => Some(Accelerator::Keccak),
            SyscallNumbers::Ecsm => Some(Accelerator::Ecsm),
            SyscallNumbers::Print
            | SyscallNumbers::Panic
            | SyscallNumbers::Commit
            | SyscallNumbers::Halt
            | SyscallNumbers::Hint => None,
        }
    }
}

/// Reads a 256-bit little-endian value as four doublewords at `addr + 8i`.
fn load_u256_le(memory: &Memory, addr: u64) -> Result<[u8; 32], MemoryError> {
    let mut out = [0u8; 32];
    for i in 0..4 {
        let dw = memory.load_doubleword(addr + (i as u64) * 8)?;
        out[i * 8..i * 8 + 8].copy_from_slice(&dw.to_le_bytes());
    }
    Ok(out)
}

/// Writes a 256-bit little-endian value as four doublewords at `addr + 8i`.
fn store_u256_le(memory: &mut Memory, addr: u64, bytes: &[u8; 32]) -> Result<(), MemoryError> {
    for i in 0..4 {
        let mut dw = [0u8; 8];
        dw.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
        memory.store_doubleword(addr + (i as u64) * 8, u64::from_le_bytes(dw))?;
    }
    Ok(())
}

/// BENCH ONLY. Compute a non-constraining hint (modular inverse / sqrt) with the
/// same k256 arithmetic the guest verifies against. Input/output are 32-byte
/// big-endian, k256's own serialization — unlike the ECSM ABI, which is
/// little-endian because its chip consumes little-endian limbs. The HINT table
/// only copies these bytes into memory writes, so the order is free to match the
/// consumers. On any failure (non-canonical input, no inverse/sqrt) returns zeros;
/// the guest's verify then fails loudly.
///
/// `pub` so the prover's `collect_hint_ops` can reproduce the exact output value
/// the executor wrote to guest memory (the value is not carried in the CPU log).
pub fn compute_hint(hint_id: u64, in_be: &[u8; 32]) -> [u8; 32] {
    use k256::elliptic_curve::PrimeField;
    let mut fb = k256::FieldBytes::default();
    fb.copy_from_slice(in_be);

    match hint_id {
        HINT_FIELD_INV => {
            let x: Option<k256::FieldElement> = Option::from(k256::FieldElement::from_bytes(&fb));
            match x.and_then(|x| Option::<k256::FieldElement>::from(x.invert())) {
                Some(inv) => inv.to_bytes().into(),
                None => [0u8; 32],
            }
        }
        HINT_SCALAR_INV => {
            let x: Option<k256::Scalar> = Option::from(k256::Scalar::from_repr(fb));
            match x.and_then(|x| Option::<k256::Scalar>::from(x.invert())) {
                Some(inv) => inv.to_bytes().into(),
                None => [0u8; 32],
            }
        }
        HINT_FIELD_SQRT => {
            let x: Option<k256::FieldElement> = Option::from(k256::FieldElement::from_bytes(&fb));
            match x.and_then(|x| Option::<k256::FieldElement>::from(x.sqrt())) {
                Some(r) => r.to_bytes().into(),
                None => [0u8; 32],
            }
        }
        _ => [0u8; 32],
    }
}

/// Checks that a 32-byte operand does not overflow its lower 32-bit address limb:
/// `(addr mod 2^32) + max_offset < 2^32`. Tables that send an address to the memory
/// bus as a `[lo32, hi32]` pair with the per-access offset added to `lo32` alone
/// cannot represent a carry into `hi32`, so an operand straddling the limb boundary
/// makes the trace unprovable. Used by the ECSM and Hint ecalls.
fn addr_limb_ok(addr: u64, max_offset: u64) -> bool {
    (addr % LOW_LIMB) + max_offset < LOW_LIMB
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
                // W-suffix: operate on lower 32 bits, sign-extend result to 64 bits.
                // Log must store the RAW register value in src1_val (full 64 bits)
                // for the prover's MEMW register chain. The truncation to i32 is only
                // for the ALU computation.
                let raw_src = registers.read(src)?;
                let op1 = raw_src as i32;
                if matches!(op, ArithOp::Sub) {
                    return Err(ExecutionError::SubImmNotSupported);
                }
                let res32 = op.apply_word(op1, imm)?;
                let res = res32 as i64 as u64; // Sign-extend to 64 bits
                registers.write(dst, res)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: raw_src,
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
                // W-suffix: operate on lower 32 bits, sign-extend result to 64 bits.
                // Log must store RAW register values (full 64 bits) for the prover's
                // MEMW register chain. Truncation to i32 is only for ALU computation.
                let raw_src1 = registers.read(src1)?;
                let raw_src2 = registers.read(src2)?;
                let a = raw_src1 as i32;
                let b = raw_src2 as i32;
                let res32 = op.apply_word(a, b)?;
                let res = res32 as i64 as u64; // Sign-extend to 64 bits
                registers.write(dst, res)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: raw_src1,
                    src2_val: raw_src2,
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
                let syscall_number_raw = registers.read(17)?; // a7
                let syscall_number = SyscallNumbers::try_from(syscall_number_raw)
                    .map_err(|_| ExecutionError::UnknownSyscall(syscall_number_raw))?;
                let mut src2_val = 0u64;
                let mut dst_val = 0u64;
                match syscall_number {
                    SyscallNumbers::Print => {
                        // print
                        // For now this is just a mechanism to print
                        // It is not the correct implementation of ecall/ebreak
                        let pointer = registers.read(10)?;
                        let len = registers.read(11)?;
                        let bytes = memory.load_bytes(pointer, len)?;
                        let value =
                            str::from_utf8(&bytes).map_err(|_| ExecutionError::IncorrectMessage)?;
                        println!("PRINT VM: {}", value);
                    }
                    SyscallNumbers::Panic => {
                        // panic
                        let pointer = registers.read(10)?;
                        let len = registers.read(11)?;
                        let bytes = memory.load_bytes(pointer, len)?;
                        let value =
                            str::from_utf8(&bytes).map_err(|_| ExecutionError::IncorrectMessage)?;
                        return Err(ExecutionError::Panic(value.to_owned()));
                    }
                    SyscallNumbers::Commit => {
                        // commit: write(fd, buf_addr, count) per POSIX convention
                        // x10 = fd (must be 1 for stdout)
                        // x11 = buf_addr (buffer address in memory)
                        // x12 = count (number of bytes to write)
                        let fd = registers.read(10)?;
                        if fd != 1 {
                            return Err(ExecutionError::InvalidCommitFd(fd));
                        }
                        let buf_addr = registers.read(11)?;
                        let count = registers.read(12)?;
                        memory.commit_public_output(buf_addr, count)?;
                        src2_val = buf_addr;
                        dst_val = count;
                    }
                    SyscallNumbers::KeccakPermute => {
                        // keccak-f[1600] permutation on 200 bytes (25 × u64) at address in x10
                        let state_addr = registers.read(10)?;
                        if !state_addr.is_multiple_of(8) {
                            return Err(ExecutionError::UnalignedKeccakStateAddress(state_addr));
                        }
                        state_addr
                            .checked_add(KECCAK_STATE_BYTES - 1)
                            .ok_or(ExecutionError::KeccakStateAddressOverflow(state_addr))?;

                        let mut state = [0u64; 25];
                        for (i, lane) in state.iter_mut().enumerate() {
                            let lane_addr = state_addr
                                .checked_add((i as u64) * 8)
                                .ok_or(ExecutionError::KeccakStateAddressOverflow(state_addr))?;
                            *lane = memory.load_doubleword(lane_addr)?;
                        }
                        keccak_f1600(&mut state);
                        for (i, &lane) in state.iter().enumerate() {
                            let lane_addr = state_addr
                                .checked_add((i as u64) * 8)
                                .ok_or(ExecutionError::KeccakStateAddressOverflow(state_addr))?;
                            memory.store_doubleword(lane_addr, lane)?;
                        }
                        src2_val = state_addr;
                    }
                    SyscallNumbers::Ecsm => {
                        // ECSM(-11): k×G on secp256k1.
                        // x10 = addr to write xR, x11 = addr of xG, x12 = addr of k.
                        // xG, k, xR are 32-byte little-endian values; xG and xR must be
                        // canonical field elements and k must be in [1, N).
                        let addr_xr = registers.read(10)?;
                        let addr_xg = registers.read(11)?;
                        let addr_k = registers.read(12)?;
                        // AFFINE: both the input (xG‖yG) and the output (xR‖yR) are
                        // contiguous 64-byte buffers, so each spans offset 63; k is 32B.
                        if !addr_limb_ok(addr_xg, 63)
                            || !addr_limb_ok(addr_xr, 63)
                            || !addr_limb_ok(addr_k, 31)
                        {
                            return Err(ExecutionError::EcsmAddressOverflow);
                        }
                        // AFFINE: the input is a contiguous 64-byte point (xG at +0, yG at
                        // +32) and k a 32-byte scalar. They must occupy disjoint regions —
                        // the trace builder reads xG/yG at T and k at T+1 as unaligned
                        // doubleword MEMW accesses; overlap touches the same address at both
                        // timestamps and the MEMW consistency argument can't prove the chain.
                        // The guard is about trace provability, not correctness. xR (output)
                        // may alias either: its accesses are at a later timestamp.
                        if addr_xg.abs_diff(addr_k) < 64 {
                            return Err(ExecutionError::EcsmOperandOverlap);
                        }
                        let xg = load_u256_le(memory, addr_xg)?;
                        let yg = load_u256_le(memory, addr_xg.wrapping_add(32))?;
                        let k = load_u256_le(memory, addr_k)?;
                        // AFFINE: caller passes the full point (xG, yG); return both
                        // coordinates of k·(xG, yG) so the guest skips the x-only
                        // (k+1)·P y-reconstruction. xR at addr_xr, yR at +32.
                        let (xr, yr) = ecsm::scalar_mul_xy_with_y(&k, &xg, &yg)?;
                        store_u256_le(memory, addr_xr, &xr)?;
                        store_u256_le(memory, addr_xr.wrapping_add(32), &yr)?;
                        // Carry addr_xG/addr_k in the CPU log; addr_xR is recovered from x10
                        // by the ECSM register-read path in the trace builder.
                        src2_val = addr_xg;
                        dst_val = addr_k;
                    }
                    SyscallNumbers::Hint => {
                        // BENCH ONLY. Non-constraining hint: host computes a modular
                        // inverse/sqrt and writes it to the guest, which verifies it.
                        // a0 = hint_id, a1 = input addr (32-byte BE), a2 = output addr.
                        // The `_le` helpers only move bytes in address order, which is
                        // what a raw big-endian buffer needs.
                        let hint_id = registers.read(10)?;
                        let in_addr = registers.read(11)?;
                        let out_addr = registers.read(12)?;
                        // The HINT table sends the output writes as `[out_addr_lo + 8i,
                        // out_addr_hi]`, so an `out_addr` whose 32-byte range crosses the
                        // limb boundary would unbalance the memory bus. `in_addr` is not on
                        // the bus (the input read is not modeled) but is bounded too, so the
                        // ecall's operand contract is uniform and `load_u256_le` cannot
                        // overflow its address arithmetic.
                        if !addr_limb_ok(in_addr, 31) || !addr_limb_ok(out_addr, 31) {
                            return Err(ExecutionError::HintAddressOverflow);
                        }
                        let input = load_u256_le(memory, in_addr)?;
                        let output = compute_hint(hint_id, &input);
                        store_u256_le(memory, out_addr, &output)?;
                        src2_val = in_addr;
                        dst_val = out_addr;
                    }
                    SyscallNumbers::Halt => {
                        // halt
                        return Ok(Log {
                            current_pc: pc,
                            next_pc: 0,                   // We halt by setting pc to 0
                            src1_val: syscall_number_raw, // actual a7 value for rv1
                            src2_val: 0,
                            dst_val: 0,
                        });
                    }
                }
                Log {
                    current_pc: pc,
                    next_pc: pc + REGULAR_PC_UPDATE,
                    src1_val: syscall_number_raw,
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
    #[error("Invalid commit fd: expected 1 (stdout), got {0}")]
    InvalidCommitFd(u64),
    #[error("Unaligned Keccak state address: {0:#018x}")]
    UnalignedKeccakStateAddress(u64),
    #[error("Keccak state address range overflows: {0:#018x}")]
    KeccakStateAddressOverflow(u64),
    #[error("ECSM address range overflows the lower 32-bit limb")]
    EcsmAddressOverflow,
    #[error("ECSM xG and k operand ranges overlap")]
    EcsmOperandOverlap,
    #[error("Hint address range overflows the lower 32-bit limb")]
    HintAddressOverflow,
    #[error("ECSM scalar multiplication error: {0}")]
    Ecsm(#[from] ecsm::EcsmError),
}

// =============================================================================
// Keccak-f[1600] permutation
// =============================================================================

/// Round constants for Keccak-f[1600] (24 rounds).
pub const KECCAK_RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Rotation offsets R[x][y] for the rho step of Keccak-f[1600].
pub const KECCAK_RHO: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// Apply the Keccak-f[1600] permutation (24 rounds) to a 25-word state.
///
/// The state is indexed as `state[x + 5*y]` where `x, y ∈ {0..4}`.
pub fn keccak_f1600(state: &mut [u64; 25]) {
    for &rc in &KECCAK_RC {
        // θ (theta)
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        let mut d = [0u64; 5];
        for x in 0..5 {
            d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
        }
        for x in 0..5 {
            for y in 0..5 {
                state[x + 5 * y] ^= d[x];
            }
        }

        // ρ (rho) and π (pi)
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] = state[x + 5 * y].rotate_left(KECCAK_RHO[x][y]);
            }
        }

        // χ (chi)
        for x in 0..5 {
            for y in 0..5 {
                state[x + 5 * y] =
                    b[x + 5 * y] ^ (!b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
            }
        }

        // ι (iota)
        state[0] ^= rc;
    }
}

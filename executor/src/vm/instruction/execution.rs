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
    // Placeholder discriminant. The actual syscall value is BLAKE3_SYSCALL_NUMBER.
    Blake3Compress = 3,
    Commit = 64,
    Halt = 93,
    // Placeholder discriminant. The actual syscall value is ECSM_SYSCALL_NUMBER.
    Ecsm = 94,
    // Placeholder discriminant. The actual syscall value is HINT_SYSCALL_NUMBER.
    // Non-constraining hint (host computes modular inverse/sqrt, guest verifies).
    Hint = 95,
}

/// Syscall number for KeccakPermute (u64::MAX - 1 = 0xFFFF_FFFF_FFFF_FFFE).
///
/// Cannot be an enum discriminant because it exceeds isize::MAX.
pub const KECCAK_SYSCALL_NUMBER: u64 = u64::MAX - 1;
const KECCAK_STATE_BYTES: u64 = 25 * 8;

/// Syscall number for the BLAKE3 6-round compression accelerator
/// (u64::MAX - 2 = 0xFFFF_FFFF_FFFF_FFFD).
///
/// This is the **6-round internal variant** of the BLAKE3 compression function
/// (`thoughts/blake3/blake3-chip/DESIGN.md`), intended for in-house Merkle /
/// Fiat–Shamir use — it is NOT standard 7-round BLAKE3 and its security rests
/// on the named 6-round assumption recorded in the design.
///
/// ABI: `x10` = 8-byte-aligned pointer to a 176-byte state region laid out as
/// consecutive little-endian dwords at `addr + 8k`:
///
/// | dword k | contents                                   |
/// |---------|--------------------------------------------|
/// | 0..=3   | `h[0..8]` chaining value (2 u32 words/dword) |
/// | 4..=11  | `m[0..16]` message block                     |
/// | 12      | `t` counter (`t_lo = low u32 → v[12]`, `t_hi = high u32 → v[13]`) |
/// | 13      | `block_len` (low u32) \| `flags` (high u32)  |
/// | 14..=21 | `out[0..16]` — written by the syscall        |
pub const BLAKE3_SYSCALL_NUMBER: u64 = u64::MAX - 2;
/// Bytes of the BLAKE3 state region: 112 input + 64 output.
const BLAKE3_STATE_BYTES: u64 = 22 * 8;
/// Dword offset of `out[0..16]` inside the BLAKE3 state region.
const BLAKE3_OUT_DWORDS: u64 = 14;

/// Syscall number for the ECSM (elliptic-curve scalar multiply) accelerator.
///
/// The spec uses ECALL number `-11`; interpreted as an unsigned 64-bit value that is
/// `u64::MAX - 10 = 0xFFFF_FFFF_FFFF_FFF5`, which the ECSM core table puts on the `Ecall`
/// bus as `[lo32, hi32] = [2^32 - 11, 2^32 - 1]`.
pub const ECSM_SYSCALL_NUMBER: u64 = u64::MAX - 10;

/// Syscall number for the non-constraining `Hint` ecall.
///
/// The host computes a modular inverse or square root and writes it back to the
/// guest, which MUST verify it (e.g. `x·inv == 1`) and recompute in software on a
/// verification failure. The ecall adds no in-circuit correctness constraint of its
/// own — it lets the guest replace an expensive computation with a cheap check,
/// without letting the (prover-chosen) hinted value change the guest's result.
pub const HINT_SYSCALL_NUMBER: u64 = u64::MAX - 30;

/// Hint operation selector passed in `a0`.
pub const HINT_FIELD_INV: u64 = 0; // secp256k1 base-field inverse (mod p)
pub const HINT_SCALAR_INV: u64 = 1; // secp256k1 scalar-field inverse (mod n)
pub const HINT_FIELD_SQRT: u64 = 2; // secp256k1 base-field square root

/// One past the largest valid hint selector. The prover's HINT table range-checks
/// `a0 < HINT_SELECTOR_BOUND` on the ALU bus to accept exactly the set
/// [`is_valid_hint_selector`] accepts, so both live here rather than being restated
/// independently in the AIR.
pub const HINT_SELECTOR_BOUND: u64 = 3;

/// Whether `hint_id` names a hint [`compute_hint`] can produce. The ecall rejects
/// anything else up front with [`ExecutionError::HintUnknownSelector`].
pub const fn is_valid_hint_selector(hint_id: u64) -> bool {
    matches!(hint_id, HINT_FIELD_INV | HINT_SCALAR_INV | HINT_FIELD_SQRT)
}

// The AIR's range-check and the executor's accepted set must denote the same set: every
// selector below the bound is valid, and the bound itself is not. Appending a selector
// without moving the bound (or vice versa) fails to compile here, instead of making the
// HINT table assert `LT(selector, bound) = 1` against an LT row the builder emits as 0 —
// an unbalanced ALU bus with no algebraic pointer to the cause.
const _: () = {
    let mut id = 0;
    while id < HINT_SELECTOR_BOUND {
        assert!(is_valid_hint_selector(id));
        id += 1;
    }
    assert!(!is_valid_hint_selector(HINT_SELECTOR_BOUND));
};

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
            v if v == BLAKE3_SYSCALL_NUMBER => Ok(SyscallNumbers::Blake3Compress),
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
    Blake3,
    Ecsm,
}

impl SyscallNumbers {
    /// The accelerator this syscall drives, if any. Exhaustive `match self`:
    /// adding a `SyscallNumbers` variant is a compile error here, so a new
    /// accelerator can't be silently missed by counters that consume this.
    pub fn accelerator(self) -> Option<Accelerator> {
        match self {
            SyscallNumbers::KeccakPermute => Some(Accelerator::Keccak),
            SyscallNumbers::Blake3Compress => Some(Accelerator::Blake3),
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

/// Compute a non-constraining hint (modular inverse / sqrt) with the same k256
/// arithmetic the guest verifies against. Input/output are 32-byte big-endian,
/// k256's own serialization — unlike the ECSM ABI, which is little-endian because
/// its chip consumes little-endian limbs. The HINT table only copies these bytes
/// into memory writes, so the order is free to match the consumers.
///
/// On a numeric failure (non-canonical input, no inverse/sqrt) returns zeros. This
/// is NOT a loud failure and must not be treated as one: the guest's in-circuit
/// verify rejects the value and recomputes it in software (see the `ethrex-crypto`
/// crate), so a zero/garbage hint only costs the guest extra work — it can never
/// change the guest's result. An *unknown* `hint_id` never reaches here: the ecall
/// dispatch rejects it up front with [`ExecutionError::HintUnknownSelector`], so the
/// `_` arm below is defensive only.
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
                    SyscallNumbers::Blake3Compress => {
                        // BLAKE3 6-round compression on a 176-byte region at the
                        // address in x10 (layout: see BLAKE3_SYSCALL_NUMBER docs).
                        let state_addr = registers.read(10)?;
                        if !state_addr.is_multiple_of(8) {
                            return Err(ExecutionError::UnalignedBlake3StateAddress(state_addr));
                        }
                        state_addr
                            .checked_add(BLAKE3_STATE_BYTES - 1)
                            .ok_or(ExecutionError::Blake3StateAddressOverflow(state_addr))?;

                        // Input: 14 dwords = h[8] | m[16] | t | (block_len, flags),
                        // each dword two little-endian u32 words.
                        let mut words = [0u32; 28];
                        for k in 0..14 {
                            let dw = memory.load_doubleword(state_addr + (k as u64) * 8)?;
                            words[2 * k] = dw as u32;
                            words[2 * k + 1] = (dw >> 32) as u32;
                        }
                        let h: [u32; 8] = words[0..8].try_into().unwrap();
                        let m: [u32; 16] = words[8..24].try_into().unwrap();
                        let t = (words[24] as u64) | ((words[25] as u64) << 32);
                        let block_len = words[26];
                        let flags = words[27];

                        let out = blake3_compress_6round(&h, &m, t, block_len, flags);
                        for k in 0..8 {
                            let dw = (out[2 * k] as u64) | ((out[2 * k + 1] as u64) << 32);
                            memory.store_doubleword(
                                state_addr + (BLAKE3_OUT_DWORDS + k as u64) * 8,
                                dw,
                            )?;
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
                        if !addr_limb_ok(addr_xg, 31)
                            || !addr_limb_ok(addr_xr, 31)
                            || !addr_limb_ok(addr_k, 31)
                        {
                            return Err(ExecutionError::EcsmAddressOverflow);
                        }
                        // xG and k must occupy disjoint 32-byte regions. The trace builder
                        // reads each operand as unaligned doubleword MEMW accesses (xG at T,
                        // k at T+1); if the regions overlap, the same address is touched at
                        // both timestamps and the MEMW consistency argument can't prove the
                        // access chain. The loaded values would still be well-defined — this
                        // guard is about trace provability, not correctness of the multiply.
                        // xR may alias either: its accesses are at a later timestamp.
                        if addr_xg.abs_diff(addr_k) < 32 {
                            return Err(ExecutionError::EcsmOperandOverlap);
                        }
                        let xg = load_u256_le(memory, addr_xg)?;
                        let k = load_u256_le(memory, addr_k)?;
                        let xr = ecsm::scalar_mul_x(&k, &xg)?;
                        store_u256_le(memory, addr_xr, &xr)?;
                        // Carry addr_xG/addr_k in the CPU log; addr_xR is recovered from x10
                        // by the ECSM register-read path in the trace builder.
                        src2_val = addr_xg;
                        dst_val = addr_k;
                    }
                    SyscallNumbers::Hint => {
                        // Non-constraining hint: host computes a modular inverse/sqrt
                        // and writes it to the guest, which verifies it (and falls back
                        // to software on failure). a0 = hint_id, a1 = input addr
                        // (32-byte BE), a2 = output addr. The `_le` helpers only move
                        // bytes in address order, which is what a raw big-endian buffer
                        // needs.
                        let hint_id = registers.read(10)?;
                        let in_addr = registers.read(11)?;
                        let out_addr = registers.read(12)?;
                        // Reject an unrecognized selector up front: an unknown `hint_id`
                        // would otherwise silently produce a zero output (see
                        // `compute_hint`), indistinguishable from a legitimate numeric
                        // failure. Fail loudly instead so a guest bug surfaces here.
                        if !is_valid_hint_selector(hint_id) {
                            return Err(ExecutionError::HintUnknownSelector(hint_id));
                        }
                        // Both operands are bounded so their 32-byte ranges cannot cross the
                        // 2^32 limb boundary, and the HINT table range-checks both low limbs
                        // against the same bound (`HINT_ADDR_LIMB_BOUND`) so the AIR accepts
                        // exactly what this rejects. The memory bus does not do that job on
                        // its own: it bounds `out_addr` only to 2^32 - 25, because the write
                        // bases are `out_addr_lo + 8i` and MEMW's carry columns resolve the
                        // bytes past the largest base. `in_addr` is not on the bus at all
                        // (the input read is not modeled). Bounding both also keeps
                        // `load_u256_le`/`store_u256_le` from overflowing their address
                        // arithmetic.
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
    #[error("Unaligned BLAKE3 state address: {0:#018x}")]
    UnalignedBlake3StateAddress(u64),
    #[error("BLAKE3 state address range overflows: {0:#018x}")]
    Blake3StateAddressOverflow(u64),
    #[error("ECSM address range overflows the lower 32-bit limb")]
    EcsmAddressOverflow,
    #[error("ECSM xG and k operand ranges overlap")]
    EcsmOperandOverlap,
    #[error("Hint address range overflows the lower 32-bit limb")]
    HintAddressOverflow,
    #[error("Unknown hint selector: {0}")]
    HintUnknownSelector(u64),
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

// =============================================================================
// BLAKE3 6-round compression (internal variant)
// =============================================================================
//
// A Rust port of the validated oracle `thoughts/blake3/blake3-oracle/blake3_ref.py`
// with `rounds = 6` fixed. This is the **6-round internal variant** — NOT
// standard BLAKE3 (7 rounds); its security rests on the named 6-round
// assumption recorded in `thoughts/blake3/blake3-chip/DESIGN.md`. Differentially
// tested against the oracle's canonical 6-round vectors (pinned in
// `thoughts/blake3/blake3-oracle/canonical_6round_vectors.json`, themselves
// validated against the official `blake3` crate).

/// The BLAKE3 IV (identical to SHA-256's initial state). `IV[0..4]` seeds
/// `v[8..12]` of the compression working state.
pub const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// The BLAKE3 message-schedule permutation, applied between rounds
/// (`m'[i] = m[MSG_PERMUTATION[i]]`).
pub const BLAKE3_MSG_PERMUTATION: [usize; 16] =
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Rounds of the internal variant. 6, per the design; standard BLAKE3 is 7.
pub const BLAKE3_ROUNDS: usize = 6;

/// The BLAKE3 quarter-round G (spec §2.1).
#[inline]
fn blake3_g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(mx);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(12);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(my);
    v[d] = (v[d] ^ v[a]).rotate_right(8);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(7);
}

/// The BLAKE3 compression function `f` at 6 rounds (spec §2.2, oracle §2.4).
///
/// State init: `v[0..8] = h`, `v[8..12] = IV[0..4]`, `v[12] = t as u32`,
/// `v[13] = (t >> 32) as u32`, `v[14] = block_len`, `v[15] = flags`. Six
/// rounds of 8 G-calls (4 columns then 4 diagonals), permuting the message
/// schedule between rounds (`r < rounds - 1`, i.e. 5 permutes — the trailing
/// permute is never consumed). Feed-forward: `out[i] = v[i] ^ v[i+8]`,
/// `out[i+8] = v[i+8] ^ h[i]`. The truncated chaining value is `out[0..8]`.
pub fn blake3_compress_6round(
    h: &[u32; 8],
    m: &[u32; 16],
    t: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    let mut v: [u32; 16] = [
        h[0],
        h[1],
        h[2],
        h[3],
        h[4],
        h[5],
        h[6],
        h[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        t as u32,
        (t >> 32) as u32,
        block_len,
        flags,
    ];

    let mut m = *m;
    for r in 0..BLAKE3_ROUNDS {
        // Mix the columns.
        blake3_g(&mut v, 0, 4, 8, 12, m[0], m[1]);
        blake3_g(&mut v, 1, 5, 9, 13, m[2], m[3]);
        blake3_g(&mut v, 2, 6, 10, 14, m[4], m[5]);
        blake3_g(&mut v, 3, 7, 11, 15, m[6], m[7]);
        // Mix the diagonals.
        blake3_g(&mut v, 0, 5, 10, 15, m[8], m[9]);
        blake3_g(&mut v, 1, 6, 11, 12, m[10], m[11]);
        blake3_g(&mut v, 2, 7, 8, 13, m[12], m[13]);
        blake3_g(&mut v, 3, 4, 9, 14, m[14], m[15]);
        // Permute between rounds; the permute after the last round is never
        // consumed (oracle: `r < rounds - 1`).
        if r < BLAKE3_ROUNDS - 1 {
            let prev = m;
            for (i, &p) in BLAKE3_MSG_PERMUTATION.iter().enumerate() {
                m[i] = prev[p];
            }
        }
    }

    let mut out = [0u32; 16];
    for i in 0..8 {
        out[i] = v[i] ^ v[i + 8];
        out[i + 8] = v[i + 8] ^ h[i];
    }
    out
}

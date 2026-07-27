use num_bigint::BigUint;

use ecsm::witness::{Lincomb2Error, Lincomb2Witness};

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
    // Placeholder discriminant. The actual syscall value is
    // ECSM_LINCOMB2_SYSCALL_NUMBER.
    EcsmLincomb2 = 95,
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

/// Syscall number for the ECSM `lincomb2` accelerator: `Q = u1·P1 + u2·P2` on secp256k1
/// in one call (the ecrecover shape `s·R − z·G`).
///
/// ECALL number `-12`, i.e. `u64::MAX - 11 = 0xFFFF_FFFF_FFFF_FFF4` — the slot directly
/// below ECSM's `-11`, keeping the EC accelerators contiguous. `-2` (Keccak) and `-11`
/// (ECSM) are the only other negative syscall numbers this VM defines, so `-12` is free.
pub const ECSM_LINCOMB2_SYSCALL_NUMBER: u64 = u64::MAX - 11;

/// `2^32`. ECSM memory operands must not overflow their lower 32-bit address limb when the
/// largest per-access offset is added: the 32-byte operands reach offset +31 (last byte),
/// the 64-byte `lincomb2` operands offset +63.
const LOW_LIMB: u64 = 1 << 32;

/// Byte length of one `ecsm_lincomb2` memory operand: two 32-byte little-endian values
/// (`xP‖yP`, or `u1‖u2`) laid out back to back.
const LINCOMB2_OPERAND_BYTES: u64 = 64;

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
            v if v == ECSM_LINCOMB2_SYSCALL_NUMBER => Ok(SyscallNumbers::EcsmLincomb2),
            _ => Err(()),
        }
    }
}

/// A syscall that drives a specialized in-circuit accelerator chip.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Accelerator {
    Keccak,
    Ecsm,
    EcsmLincomb2,
}

impl SyscallNumbers {
    /// The accelerator this syscall drives, if any. Exhaustive `match self`:
    /// adding a `SyscallNumbers` variant is a compile error here, so a new
    /// accelerator can't be silently missed by counters that consume this.
    pub fn accelerator(self) -> Option<Accelerator> {
        match self {
            SyscallNumbers::KeccakPermute => Some(Accelerator::Keccak),
            SyscallNumbers::Ecsm => Some(Accelerator::Ecsm),
            SyscallNumbers::EcsmLincomb2 => Some(Accelerator::EcsmLincomb2),
            SyscallNumbers::Print
            | SyscallNumbers::Panic
            | SyscallNumbers::Commit
            | SyscallNumbers::Halt => None,
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

/// Reads a 512-bit little-endian `ecsm_lincomb2` operand as eight doublewords at `addr + 8i`.
fn load_u512_le(memory: &Memory, addr: u64) -> Result<[u8; 64], MemoryError> {
    let mut out = [0u8; 64];
    for (i, chunk) in out.chunks_mut(8).enumerate() {
        let dw = memory.load_doubleword(addr + (i as u64) * 8)?;
        chunk.copy_from_slice(&dw.to_le_bytes());
    }
    Ok(out)
}

/// Checks the ECSM address-alignment assumption: `(addr mod 2^32) + max_offset < 2^32`.
fn ecsm_addr_ok(addr: u64, max_offset: u64) -> bool {
    (addr % LOW_LIMB) + max_offset < LOW_LIMB
}

// =============================================================================
// ECSM lincomb2 status contract
// =============================================================================

// The word the `ecsm_lincomb2` syscall leaves in `a0`. `0` means "the result at the
// `a0` buffer is `Q = u1·P1 + u2·P2`"; every other value means "no sound witness
// exists for these inputs, nothing was written", and the guest falls back to its
// pure-Rust `ProjectivePoint::lincomb`.
//
// A non-zero status is ALWAYS sound: the fallback is proven guest execution, so a
// lying status can only waste cycles, never forge a result. That is why degenerate
// *values* return a status instead of trapping the way `ecsm_mul` does — the scalars
// and points come from transaction data, and a trap would let one crafted transaction
// abort the proof of an entire block. (Operand *addresses* are a different matter:
// they are chosen by the guest program, not by the transaction, so a bad address is a
// guest bug and stays a hard `ExecutionError` — see `lincomb2_addrs_ok`.)
//
// The codes are distinct per `Lincomb2Error` variant purely so debugging and bench
// runs can tell the cases apart; the guest only ever tests `!= 0`.

/// `Q` was computed and written to the `a0` buffer.
pub const LINCOMB2_STATUS_OK: u64 = 0;
/// `u1` or `u2` is zero.
pub const LINCOMB2_STATUS_SCALAR_IS_ZERO: u64 = 1;
/// `u1` or `u2` is `>= N`.
pub const LINCOMB2_STATUS_SCALAR_OUT_OF_RANGE: u64 = 2;
/// `P1` or `P2` is not on the curve.
pub const LINCOMB2_STATUS_POINT_NOT_ON_CURVE: u64 = 3;
/// `P1` or `P2` has a coordinate `>= p`.
pub const LINCOMB2_STATUS_POINT_NOT_CANONICAL: u64 = 4;
/// `P1 = ±P2`, so the `P1 + P2` precompute is not a chord.
pub const LINCOMB2_STATUS_SUM_DEGENERATE: u64 = 5;
/// `Q` is the point at infinity (or an accumulator collided with its addend).
pub const LINCOMB2_STATUS_RESULT_INFINITY: u64 = 6;
/// `P1` is not the secp256k1 generator `G`. See [`GENERATOR_LE`].
pub const LINCOMB2_STATUS_P1_NOT_GENERATOR: u64 = 7;

/// The secp256k1 generator `G` as the syscall's 64-byte operand: `xG ‖ yG`, little-endian.
///
/// Pinned here rather than imported because the executor has no curve library of its own;
/// `generator_le_is_the_secp256k1_generator` re-derives it from `k256` so a typo cannot
/// survive. Same idiom as `ecsm::witness`'s pinned `T₀`.
pub const GENERATOR_LE: [u8; 64] = [
    0x98, 0x17, 0xF8, 0x16, 0x5B, 0x81, 0xF2, 0x59, 0xD9, 0x28, 0xCE, 0x2D, 0xDB, 0xFC, 0x9B, 0x02,
    0x07, 0x0B, 0x87, 0xCE, 0x95, 0x62, 0xA0, 0x55, 0xAC, 0xBB, 0xDC, 0xF9, 0x7E, 0x66, 0xBE, 0x79,
    0xB8, 0xD4, 0x10, 0xFB, 0x8F, 0xD0, 0x47, 0x9C, 0x19, 0x54, 0x85, 0xA6, 0x48, 0xB4, 0x17, 0xFD,
    0xA8, 0x08, 0x11, 0x0E, 0xFC, 0xFB, 0xA4, 0x5D, 0x65, 0xC4, 0xA3, 0x26, 0x77, 0xDA, 0x3A, 0x48,
];

/// Maps a witness-generation failure to its `a0` status word.
///
/// Exhaustive on purpose: a new `Lincomb2Error` variant must be a compile error here
/// rather than silently collapsing onto an existing code.
fn lincomb2_status(error: &Lincomb2Error) -> u64 {
    match error {
        Lincomb2Error::ScalarIsZero => LINCOMB2_STATUS_SCALAR_IS_ZERO,
        Lincomb2Error::ScalarOutOfRange => LINCOMB2_STATUS_SCALAR_OUT_OF_RANGE,
        Lincomb2Error::PointNotOnCurve => LINCOMB2_STATUS_POINT_NOT_ON_CURVE,
        Lincomb2Error::PointNotCanonical => LINCOMB2_STATUS_POINT_NOT_CANONICAL,
        Lincomb2Error::SumDegenerate => LINCOMB2_STATUS_SUM_DEGENERATE,
        Lincomb2Error::ResultInfinity => LINCOMB2_STATUS_RESULT_INFINITY,
    }
}

/// The outcome of one `ecsm_lincomb2` invocation: the status word written back to `a0`
/// and, on success, the chip witness for the call.
///
/// This is the "row log" the proving side consumes. The executor deliberately does NOT
/// retain these: one `Lincomb2Witness` holds ~450 double/add steps of 1,872 bytes each
/// (three `[i64; 64]` carry arrays apiece), i.e. ~820 KiB per call — a block's worth would
/// be gigabytes. Instead the trace builder re-reads the operand bytes at the ecall's
/// timestamp and calls [`lincomb2_outcome`] — the very function the executor arm calls
/// below — so the two sides cannot disagree about the status or about which rows exist.
/// This mirrors ECSM, where the executor computes only `xR` and `collect_ecsm_ops`
/// rebuilds the full witness at trace-build time.
pub struct Lincomb2Outcome {
    /// The word written to `a0`. [`LINCOMB2_STATUS_OK`] iff `witness.is_some()`.
    pub status: u64,
    /// The chip witness, present exactly when the status is OK. Boxed because it is
    /// large enough that moving it around by value is measurable.
    pub witness: Option<Box<Lincomb2Witness>>,
}

/// Evaluates one `lincomb2` call from its three 64-byte operands, each holding two
/// 32-byte little-endian values: `p1 = xP1‖yP1`, `p2 = xP2‖yP2`, `u = u1‖u2`.
///
/// `status == LINCOMB2_STATUS_OK` holds exactly when the proving side can back the
/// result, which takes two things:
///
///  1. **A witness exists.** The status comes from `lincomb2_witness` itself, never from a
///     cheaper pre-check. A "just compute Q" shortcut could succeed where witness
///     generation fails (an interior accumulator collision, say), and the executor would
///     then promise the guest a result the prover cannot produce.
///  2. **`P1` is the generator.** `Lincomb2Witness` carries `mem_p2` but no `mem_p1` and
///     no `P1` canonicalization witness, so ECSM′ binds `a1`'s bytes to `G` by
///     construction (constant-valued MEMW reads) instead of proving membership. Without
///     this check a caller passing `P1 ≠ G` would get `status == 0`, the trace builder
///     would emit a row asserting bytes that are not there, the constraint would fail,
///     and the whole **block would become unprovable** — a completeness hole, not a
///     forgery, but one that a non-ecrecover caller could open. Returning
///     [`LINCOMB2_STATUS_P1_NOT_GENERATOR`] instead degrades that caller to the software
///     fallback and keeps executor and chip in agreement by construction.
///
/// The `P1` operand is still read in full, so the ABI stays general: **if a `mem_p1`
/// membership witness is added later, deleting the `GENERATOR_LE` comparison below is the
/// only change needed here.** Today's only caller is the guest's ecrecover, which always
/// passes `ProjectivePoint::GENERATOR`.
pub fn lincomb2_outcome(p1: &[u8; 64], p2: &[u8; 64], u: &[u8; 64]) -> Lincomb2Outcome {
    fn halves(operand: &[u8; 64]) -> ([u8; 32], [u8; 32]) {
        let mut lo = [0u8; 32];
        let mut hi = [0u8; 32];
        lo.copy_from_slice(&operand[..32]);
        hi.copy_from_slice(&operand[32..]);
        (lo, hi)
    }
    fn point(x: &[u8; 32], y: &[u8; 32]) -> ecsm::AffinePoint {
        ecsm::AffinePoint {
            x: BigUint::from_bytes_le(x),
            y: BigUint::from_bytes_le(y),
        }
    }

    // Checked before witness generation: it is the cheapest of the conditions and the one
    // the chip's row shape depends on, so there is nothing to compute if it fails.
    if *p1 != GENERATOR_LE {
        return Lincomb2Outcome {
            status: LINCOMB2_STATUS_P1_NOT_GENERATOR,
            witness: None,
        };
    }

    let (x_p1, y_p1) = halves(p1);
    let (x_p2, y_p2) = halves(p2);
    let (u1, u2) = halves(u);

    match ecsm::witness::lincomb2_witness(&u1, &u2, &point(&x_p1, &y_p1), &point(&x_p2, &y_p2)) {
        Ok(witness) => Lincomb2Outcome {
            status: LINCOMB2_STATUS_OK,
            witness: Some(Box::new(witness)),
        },
        Err(error) => Lincomb2Outcome {
            status: lincomb2_status(&error),
            witness: None,
        },
    }
}

/// Validates the four `ecsm_lincomb2` operand addresses (`a0` result, then the `a1`/`a2`/`a3`
/// inputs, in that order).
///
/// Each 64-byte region must be 8-byte aligned, stay inside its lower 32-bit address limb,
/// and be pairwise disjoint from the other three:
///
/// * **Alignment** — the proving side reads and writes each region as eight *aligned*
///   doubleword MEMW accesses, the same requirement the Keccak state address carries.
/// * **Limb bound** — an operand whose last byte (offset +63) crosses `2^32` would split
///   across the MEMW address limbs. This also makes `addr + 63` overflow-free in `u64`:
///   any `addr > u64::MAX - 63` has `addr % 2^32 > 2^32 - 64` and is rejected here.
/// * **Disjointness** — the chip proves one MEMW access chain per operand within a single
///   ecall; overlapping regions would touch the same address from two of those chains,
///   which the MEMW consistency argument cannot order. This is `ecsm_mul`'s
///   `EcsmOperandOverlap` rule widened to four operands. Note the result region may NOT
///   alias an input, unlike `ecsm_mul`'s `xR`: it is written after all three reads, so an
///   alias would leave the stored input bytes disagreeing with what the chip consumed.
///
/// These are guest-program bugs, not attacker-controlled input, so they are hard errors
/// rather than a status word — see the status-contract note above.
fn lincomb2_addrs_ok(addrs: [u64; 4]) -> Result<(), ExecutionError> {
    for &addr in &addrs {
        if !addr.is_multiple_of(8) {
            return Err(ExecutionError::Lincomb2UnalignedAddress(addr));
        }
        if !ecsm_addr_ok(addr, LINCOMB2_OPERAND_BYTES - 1) {
            return Err(ExecutionError::Lincomb2AddressOverflow(addr));
        }
    }
    for (i, &a) in addrs.iter().enumerate() {
        for &b in &addrs[i + 1..] {
            if a.abs_diff(b) < LINCOMB2_OPERAND_BYTES {
                return Err(ExecutionError::Lincomb2OperandOverlap(a, b));
            }
        }
    }
    Ok(())
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
                        if !ecsm_addr_ok(addr_xg, 31)
                            || !ecsm_addr_ok(addr_xr, 31)
                            || !ecsm_addr_ok(addr_k, 31)
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
                    SyscallNumbers::EcsmLincomb2 => {
                        // ECSM lincomb2(-12): Q = u1·P1 + u2·P2 on secp256k1.
                        // x10 = addr to write Q, x11 = addr of P1, x12 = addr of P2,
                        // x13 = addr of the scalars. Every operand is 64 bytes: two
                        // 32-byte little-endian values back to back (xQ‖yQ, xP‖yP,
                        // u1‖u2). On return x10 holds the status word: 0 means Q was
                        // written, non-zero means Q was NOT written and the guest must
                        // use its software fallback.
                        //
                        // THE MEMORY ACCESS PATTERN IS THE SAME ON BOTH PATHS. All three
                        // operands are read, and x10 is written, whether or not a witness
                        // exists — only the 64-byte Q store is conditional. The proving
                        // side gives every lincomb2 ecall one row that receives the Ecall
                        // bus and performs the same fixed set of MEMW accesses; an early
                        // return on the degenerate path would desynchronise those
                        // timestamps and leave the ecall with no receiver, unbalancing the
                        // bus. Do not "optimize" the reads away: `lincomb2_outcome` takes
                        // all three operands by value precisely so the status cannot be
                        // decided before they have been read.
                        let addr_q = registers.read(10)?;
                        let addr_p1 = registers.read(11)?;
                        let addr_p2 = registers.read(12)?;
                        let addr_u = registers.read(13)?;
                        // Address faults are the one hard error here, and they abort
                        // before any read or write, so they never leave a partial trace.
                        lincomb2_addrs_ok([addr_q, addr_p1, addr_p2, addr_u])?;

                        let p1 = load_u512_le(memory, addr_p1)?;
                        let p2 = load_u512_le(memory, addr_p2)?;
                        let u = load_u512_le(memory, addr_u)?;

                        let outcome = lincomb2_outcome(&p1, &p2, &u);
                        // Q is written only on success: there is no witness to prove on
                        // the error path, so there must also be no bytes for the guest to
                        // mistake for a result.
                        if let Some(witness) = &outcome.witness {
                            store_u256_le(memory, addr_q, &witness.x_q)?;
                            store_u256_le(memory, addr_q + 32, &witness.y_q)?;
                        }
                        // Unconditional: the status write is what makes the error path
                        // expressible as a row at all.
                        registers.write(10, outcome.status)?;
                        // The three input addresses survive in x11/x12/x13 and are
                        // recoverable from the register state exactly as ECSM recovers
                        // its own, but x10 is clobbered by the status, so the CPU log
                        // carries both: `src2_val` = the result address, `dst_val` =
                        // x10's post-execution value (the status). Both are redundant
                        // with a trace-builder replay; they are carried because the
                        // fields are otherwise unused and the status saves the collector
                        // a witness recomputation on the error path.
                        src2_val = addr_q;
                        dst_val = outcome.status;
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
    #[error("ECSM scalar multiplication error: {0}")]
    Ecsm(#[from] ecsm::EcsmError),
    #[error("ECSM lincomb2 operand address is not 8-byte aligned: {0:#018x}")]
    Lincomb2UnalignedAddress(u64),
    #[error("ECSM lincomb2 operand range overflows the lower 32-bit limb: {0:#018x}")]
    Lincomb2AddressOverflow(u64),
    #[error("ECSM lincomb2 operand ranges overlap: {0:#018x} and {1:#018x}")]
    Lincomb2OperandOverlap(u64, u64),
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

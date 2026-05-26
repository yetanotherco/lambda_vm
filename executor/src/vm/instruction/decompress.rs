//! RV64C (compressed instruction) decompression.
//!
//! The "C" extension encodes common instructions in 16 bits. Every compressed
//! instruction is, by construction, an alias for exactly one 32-bit base
//! instruction. We therefore expand each 16-bit encoding to its equivalent base
//! [`Instruction`] at decode time: the rest of the pipeline (execution,
//! constraints, prover) only ever sees base instructions, and the 2-byte width is
//! tracked separately (see [`super::decoding::DecodedInstruction`]) to drive `pc`
//! advance and the prover's `c_type` flag.
//!
//! Scope: the integer RV64C subset. Floating-point compressed encodings
//! (`C.FLD/C.FSD/C.FLDSP/C.FSDSP`) and the RV32-only `C.JAL` are rejected, since
//! Lambda VM is `rv64im` (no F/D). Reserved/illegal encodings (including the
//! all-zero halfword) return an error, matching how 32-bit decode rejects unknown
//! encodings.
//!
//! Immediate bit-scrambling follows the "RVC instruction set listings" tables in
//! the RISC-V unprivileged ISA spec; each scramble is annotated with the
//! `instr[..]` source bits → immediate `[..]` destination bits.

use super::decoding::{ArithOp, Comparison, Instruction, InstructionError, LoadStoreWidth};

/// Length in bytes of the instruction starting with `first_half`.
///
/// Per the RISC-V encoding, any halfword whose low two bits are not `0b11` is a
/// 2-byte compressed instruction; everything else is a 4-byte base instruction.
pub fn instr_len(first_half: u16) -> u8 {
    if first_half & 0b11 == 0b11 { 4 } else { 2 }
}

/// Sign-extend the low `bits` of `value` to a 32-bit signed integer.
fn sign_extend(value: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((value << shift) as i32) >> shift
}

/// Expand a 16-bit RV64C instruction into its equivalent base [`Instruction`].
///
/// Returns [`InstructionError::IllegalCompressed`] for the all-zero/illegal
/// encodings and excluded floating-point encodings, and
/// [`InstructionError::ReservedCompressed`] for encodings the spec marks reserved.
pub fn decompress(half: u16) -> Result<Instruction, InstructionError> {
    let h = half as u32;
    let funct3 = (h >> 13) & 0b111;
    match h & 0b11 {
        0b00 => decompress_q0(h, funct3, half),
        0b01 => decompress_q1(h, funct3, half),
        0b10 => decompress_q2(h, funct3, half),
        // 0b11 is not a compressed instruction.
        _ => Err(InstructionError::IllegalCompressed(half)),
    }
}

/// Quadrant 0 (`op = 0b00`): stack-pointer-relative `ADDI4SPN` and register-pair
/// loads/stores.
fn decompress_q0(h: u32, funct3: u32, half: u16) -> Result<Instruction, InstructionError> {
    let rs1p = ((h >> 7) & 0x7) + 8; // inst[9:7] -> x8..x15
    let rdp_rs2p = ((h >> 2) & 0x7) + 8; // inst[4:2] -> x8..x15
    match funct3 {
        // C.ADDI4SPN: addi rd', x2, nzuimm
        0b000 => {
            let nzuimm = ((h >> 7) & 0x30)   // inst[12:11] -> imm[5:4]
                | ((h >> 1) & 0x3C0)         // inst[10:7]  -> imm[9:6]
                | ((h >> 4) & 0x4)           // inst[6]     -> imm[2]
                | ((h >> 2) & 0x8); // inst[5]     -> imm[3]
            // nzuimm == 0 is reserved; this also catches the all-zero illegal word.
            if nzuimm == 0 {
                return Err(InstructionError::IllegalCompressed(half));
            }
            Ok(Instruction::ArithImm {
                dst: rdp_rs2p,
                src: 2,
                imm: nzuimm as i32,
                op: ArithOp::Add,
            })
        }
        // C.LW: lw rd', uimm(rs1')
        0b010 => Ok(Instruction::Load {
            dst: rdp_rs2p,
            offset: lw_uimm(h) as i32,
            base: rs1p,
            width: LoadStoreWidth::Word,
        }),
        // C.LD: ld rd', uimm(rs1')
        0b011 => Ok(Instruction::Load {
            dst: rdp_rs2p,
            offset: ld_uimm(h) as i32,
            base: rs1p,
            width: LoadStoreWidth::DoubleWord,
        }),
        // C.SW: sw rs2', uimm(rs1')
        0b110 => Ok(Instruction::Store {
            src: rdp_rs2p,
            offset: lw_uimm(h) as i32,
            base: rs1p,
            width: LoadStoreWidth::Word,
        }),
        // C.SD: sd rs2', uimm(rs1')
        0b111 => Ok(Instruction::Store {
            src: rdp_rs2p,
            offset: ld_uimm(h) as i32,
            base: rs1p,
            width: LoadStoreWidth::DoubleWord,
        }),
        // 0b100 is reserved; 0b001/0b101 are C.FLD/C.FSD (float, excluded).
        0b100 => Err(InstructionError::ReservedCompressed(half)),
        _ => Err(InstructionError::IllegalCompressed(half)),
    }
}

/// Quadrant 1 (`op = 0b01`): immediate ALU ops, jumps and branches.
fn decompress_q1(h: u32, funct3: u32, half: u16) -> Result<Instruction, InstructionError> {
    match funct3 {
        // C.NOP / C.ADDI: addi rd, rd, nzimm  (rd == 0 is C.NOP / a HINT)
        0b000 => {
            let rd = (h >> 7) & 0x1F;
            Ok(Instruction::ArithImm {
                dst: rd,
                src: rd,
                imm: imm6(h),
                op: ArithOp::Add,
            })
        }
        // C.ADDIW: addiw rd, rd, imm  (rd == 0 reserved). RV32 would be C.JAL (excluded).
        0b001 => {
            let rd = (h >> 7) & 0x1F;
            if rd == 0 {
                return Err(InstructionError::ReservedCompressed(half));
            }
            Ok(Instruction::ArithImmW {
                dst: rd,
                src: rd,
                imm: imm6(h),
                op: ArithOp::Add,
            })
        }
        // C.LI: addi rd, x0, imm  (rd == 0 HINT)
        0b010 => {
            let rd = (h >> 7) & 0x1F;
            Ok(Instruction::ArithImm {
                dst: rd,
                src: 0,
                imm: imm6(h),
                op: ArithOp::Add,
            })
        }
        // C.LUI / C.ADDI16SP (rd == 2)
        0b011 => {
            let rd = (h >> 7) & 0x1F;
            if rd == 2 {
                // C.ADDI16SP: addi x2, x2, nzimm
                let field = ((h >> 3) & 0x200)  // inst[12]   -> imm[9]
                    | ((h >> 2) & 0x10)         // inst[6]    -> imm[4]
                    | ((h << 1) & 0x40)         // inst[5]    -> imm[6]
                    | ((h << 4) & 0x180)        // inst[4:3]  -> imm[8:7]
                    | ((h << 3) & 0x20); // inst[2]    -> imm[5]
                if field == 0 {
                    return Err(InstructionError::ReservedCompressed(half));
                }
                Ok(Instruction::ArithImm {
                    dst: 2,
                    src: 2,
                    imm: sign_extend(field, 10),
                    op: ArithOp::Add,
                })
            } else {
                // C.LUI: lui rd, nzimm  (rd == 0 HINT)
                let nzimm6 = ((h >> 7) & 0x20) | ((h >> 2) & 0x1F);
                if nzimm6 == 0 {
                    return Err(InstructionError::ReservedCompressed(half));
                }
                // LUI places imm[17:12] sign-extended; the base LUI imm field holds
                // the value already shifted into bits [31:12].
                Ok(Instruction::LoadUpperImm {
                    dst: rd,
                    imm: (sign_extend(nzimm6, 6) << 12) as u32,
                })
            }
        }
        // MISC-ALU
        0b100 => decompress_q1_alu(h, half),
        // C.J: jal x0, offset
        0b101 => Ok(Instruction::JumpAndLink {
            dst: 0,
            offset: cj_offset(h),
        }),
        // C.BEQZ: beq rs1', x0, offset
        0b110 => Ok(Instruction::Branch {
            src1: ((h >> 7) & 0x7) + 8,
            src2: 0,
            cond: Comparison::Equal,
            offset: cb_offset(h),
        }),
        // C.BNEZ: bne rs1', x0, offset
        0b111 => Ok(Instruction::Branch {
            src1: ((h >> 7) & 0x7) + 8,
            src2: 0,
            cond: Comparison::NotEqual,
            offset: cb_offset(h),
        }),
        _ => Err(InstructionError::IllegalCompressed(half)),
    }
}

/// Quadrant 1, `funct3 = 0b100`: register/immediate ALU on the `x8..x15` subset.
fn decompress_q1_alu(h: u32, half: u16) -> Result<Instruction, InstructionError> {
    let rdp = ((h >> 7) & 0x7) + 8; // rs1'/rd'
    match (h >> 10) & 0x3 {
        // C.SRLI: srli rd', rd', shamt
        0b00 => Ok(Instruction::ArithImm {
            dst: rdp,
            src: rdp,
            imm: shamt6(h) as i32,
            op: ArithOp::ShiftRightLogical,
        }),
        // C.SRAI: srai rd', rd', shamt
        0b01 => Ok(Instruction::ArithImm {
            dst: rdp,
            src: rdp,
            imm: shamt6(h) as i32,
            op: ArithOp::ShiftRightArith,
        }),
        // C.ANDI: andi rd', rd', imm
        0b10 => Ok(Instruction::ArithImm {
            dst: rdp,
            src: rdp,
            imm: imm6(h),
            op: ArithOp::And,
        }),
        // Register-register ops, discriminated by inst[12] and inst[6:5].
        0b11 => {
            let rs2p = ((h >> 2) & 0x7) + 8;
            let op = match ((h >> 12) & 1, (h >> 5) & 0x3) {
                (0, 0b00) => return arith(rdp, rs2p, ArithOp::Sub), // C.SUB
                (0, 0b01) => return arith(rdp, rs2p, ArithOp::Xor), // C.XOR
                (0, 0b10) => return arith(rdp, rs2p, ArithOp::Or),  // C.OR
                (0, 0b11) => return arith(rdp, rs2p, ArithOp::And), // C.AND
                (1, 0b00) => ArithOp::Sub,                          // C.SUBW
                (1, 0b01) => ArithOp::Add,                          // C.ADDW
                // (1, 0b10) and (1, 0b11) are reserved.
                _ => return Err(InstructionError::ReservedCompressed(half)),
            };
            Ok(Instruction::ArithW {
                dst: rdp,
                src1: rdp,
                src2: rs2p,
                op,
            })
        }
        _ => Err(InstructionError::IllegalCompressed(half)),
    }
}

/// Quadrant 2 (`op = 0b10`): `SLLI`, stack-pointer loads/stores and the
/// `JR/MV/EBREAK/JALR/ADD` family.
fn decompress_q2(h: u32, funct3: u32, half: u16) -> Result<Instruction, InstructionError> {
    match funct3 {
        // C.SLLI: slli rd, rd, shamt  (rd == 0 HINT)
        0b000 => {
            let rd = (h >> 7) & 0x1F;
            Ok(Instruction::ArithImm {
                dst: rd,
                src: rd,
                imm: shamt6(h) as i32,
                op: ArithOp::ShiftLeftLogical,
            })
        }
        // C.LWSP: lw rd, uimm(x2)  (rd == 0 reserved)
        0b010 => {
            let rd = (h >> 7) & 0x1F;
            if rd == 0 {
                return Err(InstructionError::ReservedCompressed(half));
            }
            let uimm = ((h >> 7) & 0x20)     // inst[12]   -> imm[5]
                | ((h >> 2) & 0x1C)          // inst[6:4]  -> imm[4:2]
                | ((h << 4) & 0xC0); // inst[3:2]  -> imm[7:6]
            Ok(Instruction::Load {
                dst: rd,
                offset: uimm as i32,
                base: 2,
                width: LoadStoreWidth::Word,
            })
        }
        // C.LDSP: ld rd, uimm(x2)  (rd == 0 reserved)
        0b011 => {
            let rd = (h >> 7) & 0x1F;
            if rd == 0 {
                return Err(InstructionError::ReservedCompressed(half));
            }
            let uimm = ((h >> 7) & 0x20)     // inst[12]   -> imm[5]
                | ((h >> 2) & 0x18)          // inst[6:5]  -> imm[4:3]
                | ((h << 4) & 0x1C0); // inst[4:2]  -> imm[8:6]
            Ok(Instruction::Load {
                dst: rd,
                offset: uimm as i32,
                base: 2,
                width: LoadStoreWidth::DoubleWord,
            })
        }
        // C.JR / C.MV / C.EBREAK / C.JALR / C.ADD
        0b100 => decompress_q2_cr(h, half),
        // C.SWSP: sw rs2, uimm(x2)
        0b110 => {
            let uimm = ((h >> 7) & 0x3C)     // inst[12:9] -> imm[5:2]
                | ((h >> 1) & 0xC0); // inst[8:7]  -> imm[7:6]
            Ok(Instruction::Store {
                src: (h >> 2) & 0x1F,
                offset: uimm as i32,
                base: 2,
                width: LoadStoreWidth::Word,
            })
        }
        // C.SDSP: sd rs2, uimm(x2)
        0b111 => {
            let uimm = ((h >> 7) & 0x38)     // inst[12:10] -> imm[5:3]
                | ((h >> 1) & 0x1C0); // inst[9:7]   -> imm[8:6]
            Ok(Instruction::Store {
                src: (h >> 2) & 0x1F,
                offset: uimm as i32,
                base: 2,
                width: LoadStoreWidth::DoubleWord,
            })
        }
        // 0b001/0b101 are C.FLDSP/C.FSDSP (float, excluded).
        _ => Err(InstructionError::IllegalCompressed(half)),
    }
}

/// Quadrant 2, `funct3 = 0b100`: the `CR`-format `JR/MV/EBREAK/JALR/ADD` group.
fn decompress_q2_cr(h: u32, half: u16) -> Result<Instruction, InstructionError> {
    let rd_rs1 = (h >> 7) & 0x1F;
    let rs2 = (h >> 2) & 0x1F;
    match ((h >> 12) & 1, rs2) {
        // C.JR: jalr x0, 0(rs1)  (rs1 == 0 reserved)
        (0, 0) => {
            if rd_rs1 == 0 {
                return Err(InstructionError::ReservedCompressed(half));
            }
            Ok(Instruction::JumpAndLinkRegister {
                base: rd_rs1,
                dst: 0,
                offset: 0,
            })
        }
        // C.MV: add rd, x0, rs2  (rd == 0 HINT)
        (0, _) => Ok(Instruction::Arith {
            dst: rd_rs1,
            src1: 0,
            src2: rs2,
            op: ArithOp::Add,
        }),
        // C.EBREAK (rs1 == 0) or C.JALR: jalr x1, 0(rs1)
        (1, 0) => {
            if rd_rs1 == 0 {
                Ok(Instruction::EcallEbreak)
            } else {
                Ok(Instruction::JumpAndLinkRegister {
                    base: rd_rs1,
                    dst: 1,
                    offset: 0,
                })
            }
        }
        // C.ADD: add rd, rd, rs2  (rd == 0 HINT)
        (1, _) => Ok(Instruction::Arith {
            dst: rd_rs1,
            src1: rd_rs1,
            src2: rs2,
            op: ArithOp::Add,
        }),
        _ => Err(InstructionError::IllegalCompressed(half)),
    }
}

/// Build a register-register `Arith` instruction (`dst = rs1' = rdp`).
fn arith(rdp: u32, rs2p: u32, op: ArithOp) -> Result<Instruction, InstructionError> {
    Ok(Instruction::Arith {
        dst: rdp,
        src1: rdp,
        src2: rs2p,
        op,
    })
}

/// 6-bit signed immediate (`CI`): inst[12] -> imm[5], inst[6:2] -> imm[4:0].
fn imm6(h: u32) -> i32 {
    sign_extend(((h >> 7) & 0x20) | ((h >> 2) & 0x1F), 6)
}

/// 6-bit shift amount (RV64: full 6 bits, inst[12] -> shamt[5], inst[6:2] -> shamt[4:0]).
fn shamt6(h: u32) -> u32 {
    ((h >> 7) & 0x20) | ((h >> 2) & 0x1F)
}

/// `C.LW`/`C.SW` unsigned offset: inst[12:10] -> imm[5:3], inst[6] -> imm[2], inst[5] -> imm[6].
fn lw_uimm(h: u32) -> u32 {
    ((h >> 7) & 0x38) | ((h >> 4) & 0x4) | ((h << 1) & 0x40)
}

/// `C.LD`/`C.SD` unsigned offset: inst[12:10] -> imm[5:3], inst[6:5] -> imm[7:6].
fn ld_uimm(h: u32) -> u32 {
    ((h >> 7) & 0x38) | ((h << 1) & 0xC0)
}

/// `C.J` jump offset (`CJ`): imm[11|4|9:8|10|6|7|3:1|5] = inst[12|11|10:9|8|7|6|5:3|2].
fn cj_offset(h: u32) -> i32 {
    let imm = ((h >> 1) & 0x800)   // inst[12]   -> imm[11]
        | ((h >> 7) & 0x10)        // inst[11]   -> imm[4]
        | ((h >> 1) & 0x300)       // inst[10:9] -> imm[9:8]
        | ((h << 2) & 0x400)       // inst[8]    -> imm[10]
        | ((h >> 1) & 0x40)        // inst[7]    -> imm[6]
        | ((h << 1) & 0x80)        // inst[6]    -> imm[7]
        | ((h >> 2) & 0xE)         // inst[5:3]  -> imm[3:1]
        | ((h << 3) & 0x20); // inst[2]    -> imm[5]
    sign_extend(imm, 12)
}

/// `C.BEQZ`/`C.BNEZ` branch offset (`CB`): imm[8|4:3|7:6|2:1|5] = inst[12|11:10|6:5|4:3|2].
fn cb_offset(h: u32) -> i32 {
    let imm = ((h >> 4) & 0x100)   // inst[12]    -> imm[8]
        | ((h >> 7) & 0x18)        // inst[11:10] -> imm[4:3]
        | ((h << 1) & 0xC0)        // inst[6:5]   -> imm[7:6]
        | ((h >> 2) & 0x6)         // inst[4:3]   -> imm[2:1]
        | ((h << 3) & 0x20); // inst[2]     -> imm[5]
    sign_extend(imm, 9)
}

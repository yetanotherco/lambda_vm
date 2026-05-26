// Opcodes
const ARITH_OPCODE: u32 = 0b0110011;
const ARITH_IMM_OPCODE: u32 = 0b0010011;
const LOAD_OPCODE: u32 = 0b0000011;
const STORE_OPCODE: u32 = 0b0100011;
const BRANCH_OPCODE: u32 = 0b1100011;
const JUMP_AND_LINK_REGISTER_OPCCODE: u32 = 0b1100111;
const JUMP_AND_LINK_OPCODE: u32 = 0b1101111;
const LOAD_UPPER_IMM_OPCODE: u32 = 0b0110111;
const ADD_UPPER_IMM_TO_PC: u32 = 0b0010111;
const SYSTEM_OPCODE: u32 = 0b1110011;
const FENCE_OPCODE: u32 = 0b0001111; // 0x0F - FENCE, FENCE.I
// RV64 specific opcodes
const ARITH_IMM_32_OPCODE: u32 = 0b0011011; // 0x1B - ADDIW, SLLIW, SRLIW, SRAIW
const ARITH_32_OPCODE: u32 = 0b0111011; // 0x3B - ADDW, SUBW, SLLW, SRLW, SRAW, MULW, etc.

#[derive(Debug)]
pub enum Opcode {
    Arith,
    ArithImm,
    Load,
    Store,
    Branch,
    JumpAndLinkRegister,
    JumpAndLink,
    LoadUpperImm,
    AddUpperImmToPc,
    System,
    // RV64 specific
    ArithImm32, // OP-IMM-32: W-suffix immediate instructions
    Arith32,    // OP-32: W-suffix register instructions
    Fence,
}

impl TryFrom<u32> for Opcode {
    type Error = InstructionError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            ARITH_OPCODE => Opcode::Arith,
            ARITH_IMM_OPCODE => Opcode::ArithImm,
            LOAD_OPCODE => Opcode::Load,
            STORE_OPCODE => Opcode::Store,
            BRANCH_OPCODE => Opcode::Branch,
            JUMP_AND_LINK_REGISTER_OPCCODE => Opcode::JumpAndLinkRegister,
            JUMP_AND_LINK_OPCODE => Opcode::JumpAndLink,
            LOAD_UPPER_IMM_OPCODE => Opcode::LoadUpperImm,
            ADD_UPPER_IMM_TO_PC => Opcode::AddUpperImmToPc,
            SYSTEM_OPCODE => Opcode::System,
            FENCE_OPCODE => Opcode::Fence,
            ARITH_IMM_32_OPCODE => Opcode::ArithImm32,
            ARITH_32_OPCODE => Opcode::Arith32,
            _ => return Err(InstructionError::UnknownOpcode(value)),
        })
    }
}

enum InstructionFormat {
    R,
    I,
    S,
    B,
    U,
    J,
}

impl Opcode {
    fn instruction_format(&self) -> InstructionFormat {
        match self {
            Opcode::Arith | Opcode::Arith32 => InstructionFormat::R,
            Opcode::ArithImm
            | Opcode::ArithImm32
            | Opcode::Load
            | Opcode::JumpAndLinkRegister
            | Opcode::System
            | Opcode::Fence => InstructionFormat::I,
            Opcode::Store => InstructionFormat::S,
            Opcode::Branch => InstructionFormat::B,
            Opcode::JumpAndLink => InstructionFormat::J,
            Opcode::LoadUpperImm | Opcode::AddUpperImmToPc => InstructionFormat::U,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ArithOp {
    Add,
    Sub,
    Xor,
    Or,
    And,
    ShiftLeftLogical,
    ShiftRightLogical,
    ShiftRightArith,
    SetLessThan,
    SetLessThanU,
    Mul,
    MulHigh,
    MulHighSignedUnsigned,
    MulHighUnsigned,
    Div,
    DivUnsigned,
    Remainder,
    RemainderUnsigned,
}

const LOAD_STORE_BYTE_WIDTH: u32 = 0x0;
const LOAD_STORE_HALF_WIDTH: u32 = 0x1;
const LOAD_STORE_WORD_WIDTH: u32 = 0x2;
const LOAD_STORE_DOUBLEWORD_WIDTH: u32 = 0x3; // RV64: LD/SD
const LOAD_BYTE_UNSIGNED_FUNC: u32 = 0x4;
const LOAD_HALF_UNSIGNED_FUNC: u32 = 0x5;
const LOAD_WORD_UNSIGNED_FUNC: u32 = 0x6; // RV64: LWU

#[derive(Debug, Clone, Copy)]
pub enum LoadStoreWidth {
    Byte,
    Half,
    Word,
    DoubleWord, // RV64: LD/SD
    ByteUnsigned,
    HalfUnsigned,
    WordUnsigned, // RV64: LWU
}

impl LoadStoreWidth {
    fn from_func3(func3: u32) -> Result<LoadStoreWidth, InstructionError> {
        Ok(match func3 {
            LOAD_STORE_BYTE_WIDTH => LoadStoreWidth::Byte,
            LOAD_STORE_HALF_WIDTH => LoadStoreWidth::Half,
            LOAD_STORE_WORD_WIDTH => LoadStoreWidth::Word,
            LOAD_STORE_DOUBLEWORD_WIDTH => LoadStoreWidth::DoubleWord,
            LOAD_BYTE_UNSIGNED_FUNC => LoadStoreWidth::ByteUnsigned,
            LOAD_HALF_UNSIGNED_FUNC => LoadStoreWidth::HalfUnsigned,
            LOAD_WORD_UNSIGNED_FUNC => LoadStoreWidth::WordUnsigned,
            width => return Err(InstructionError::InvalidLoadStoreWidth(width)),
        })
    }

    fn from_func3_store(func3: u32) -> Result<LoadStoreWidth, InstructionError> {
        Ok(match func3 {
            LOAD_STORE_BYTE_WIDTH => LoadStoreWidth::Byte,
            LOAD_STORE_HALF_WIDTH => LoadStoreWidth::Half,
            LOAD_STORE_WORD_WIDTH => LoadStoreWidth::Word,
            LOAD_STORE_DOUBLEWORD_WIDTH => LoadStoreWidth::DoubleWord,
            width => return Err(InstructionError::InvalidLoadStoreWidth(width)),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Comparison {
    Equal,
    NotEqual,
    LessThan,
    GreaterOrEqual,
    LessThanUnsigned,
    GreaterOrEqualUnsigned,
}

#[derive(Debug, Clone, Copy)]
pub enum CsrOp {
    CSRRW,
    CSRRS,
    CSRRC,
    CSRRWI,
    CSRRSI,
    CSRRCI,
}

#[derive(Debug, Clone, Copy)]
pub enum Instruction {
    // 64-bit arithmetic (RV64I base)
    Arith {
        dst: u32,
        src1: u32,
        src2: u32,
        op: ArithOp,
    },
    ArithImm {
        dst: u32,
        src: u32,
        imm: i32,
        op: ArithOp,
    },
    // 32-bit arithmetic with sign extension (RV64 W-suffix)
    ArithW {
        dst: u32,
        src1: u32,
        src2: u32,
        op: ArithOp,
    },
    ArithImmW {
        dst: u32,
        src: u32,
        imm: i32,
        op: ArithOp,
    },
    JumpAndLink {
        dst: u32,
        offset: i32,
    },
    JumpAndLinkRegister {
        base: u32,
        dst: u32,
        offset: i32,
    },
    Store {
        src: u32,
        offset: i32,
        base: u32,
        width: LoadStoreWidth,
    },
    Load {
        dst: u32,
        offset: i32,
        base: u32,
        width: LoadStoreWidth,
    },
    Branch {
        src1: u32,
        src2: u32,
        cond: Comparison,
        offset: i32,
    },
    LoadUpperImm {
        dst: u32,
        imm: u32,
    },
    AddUpperImmToPc {
        dst: u32,
        imm: u32,
    },
    CSR {
        csr: u32,
        src: u32,
        dst: u32,
        op: CsrOp,
    },
    EcallEbreak,
    Fence,
}

const OPCODE_MASK: u32 = 0x0000007f;
const FUNC7_MASK: u32 = 0xfe000000;
const FUNC3_MASK: u32 = 0x00007000;
const RS1_MASK: u32 = 0x000f8000;
const RS2_MASK: u32 = 0x01f00000;
const RD_MASK: u32 = 0x00000f80;
const SIGN_MASK: u32 = 0x80000000;
const I_TYPE_IMM_MASK: u32 = 0xfff;
const U_TYPE_IMM_MASK: u32 = 0xfffff000;

impl Instruction {
    pub fn parse(instruction: u32) -> Result<Instruction, InstructionError> {
        let opcode = parse_opcode(instruction)?;
        match opcode.instruction_format() {
            InstructionFormat::R => parse_r_instruction(instruction, opcode),
            InstructionFormat::I => parse_i_instruction(instruction, opcode),
            InstructionFormat::S => parse_s_instruction(instruction, opcode),
            InstructionFormat::B => parse_b_instruction(instruction, opcode),
            InstructionFormat::J => parse_j_instruction(instruction, opcode),
            InstructionFormat::U => parse_u_instruction(instruction, opcode),
        }
    }
}

/// A decoded instruction together with the number of bytes it occupies in the
/// instruction stream (`2` for an RV64C compressed instruction, `4` otherwise).
///
/// Compressed instructions are expanded to their equivalent base `Instruction`
/// at decode time, so the rest of the pipeline (execution, constraints) only ever
/// sees base instructions; `len` is what distinguishes them and drives `pc` advance
/// and the prover's `c_type` flag.
#[derive(Debug, Clone, Copy)]
pub struct DecodedInstruction {
    pub instr: Instruction,
    pub len: u8,
}

/// Decode a single instruction from a little-endian instruction word.
///
/// Only the low 16 bits are inspected to determine the length: if they do not end
/// in `0b11` the instruction is a 2-byte RV64C compressed instruction and is
/// expanded via [`decompress`](super::decompress::decompress); otherwise the full
/// 32 bits are parsed as a base instruction. Callers that only have 16 bits
/// available (e.g. at a segment boundary) must guarantee the high 16 bits are valid
/// whenever the low half indicates a 4-byte instruction.
pub fn decode_instruction(word: u32) -> Result<DecodedInstruction, InstructionError> {
    let first_half = word as u16;
    if super::decompress::instr_len(first_half) == 2 {
        Ok(DecodedInstruction {
            instr: super::decompress::decompress(first_half)?,
            len: 2,
        })
    } else {
        Ok(DecodedInstruction {
            instr: Instruction::parse(word)?,
            len: 4,
        })
    }
}

/// Decode an executable segment, given as the little-endian 4-byte memory words it
/// was loaded from, into the sequence of `(byte_offset, instruction)` it contains.
///
/// The words are reinterpreted as a halfword stream and walked by the actual
/// instruction width, so a segment may mix 2-byte (compressed) and 4-byte
/// instructions, and a 4-byte instruction may start at a 2-byte (non-4) offset. A
/// final dangling halfword (a 4-byte instruction whose second half lies past the
/// segment) is treated as a non-instruction tail and dropped.
///
/// This is the single decode entry point shared by the executor's instruction
/// cache and the prover/verifier's DECODE generation, so the two cannot disagree
/// on instruction boundaries or `c_type`.
pub fn decode_segment_words(
    words: &[u32],
) -> Result<Vec<(u64, DecodedInstruction)>, InstructionError> {
    let halfwords: Vec<u16> = words
        .iter()
        .flat_map(|w| [*w as u16, (*w >> 16) as u16])
        .collect();

    let mut out = Vec::new();
    let mut i = 0usize;
    while i < halfwords.len() {
        let lo = halfwords[i];
        // A zero halfword is `c.unimp` / alignment padding (the ELF loader also
        // zero-fills the high half of a final word when the segment's byte length
        // is not a multiple of 4). It is never a real instruction start, so skip it
        // instead of failing the whole segment decode. If such a slot is ever the
        // target of a jump, the on-demand fetch path will surface the error.
        if lo == 0 {
            i += 1;
            continue;
        }
        let byte_offset = (i as u64) * 2;
        if super::decompress::instr_len(lo) == 2 {
            out.push((
                byte_offset,
                DecodedInstruction {
                    instr: super::decompress::decompress(lo)?,
                    len: 2,
                },
            ));
            i += 1;
        } else {
            // 4-byte instruction: needs the following halfword.
            let Some(&hi) = halfwords.get(i + 1) else {
                break; // dangling trailing halfword, not an instruction
            };
            let word = ((hi as u32) << 16) | (lo as u32);
            out.push((
                byte_offset,
                DecodedInstruction {
                    instr: Instruction::parse(word)?,
                    len: 4,
                },
            ));
            i += 2;
        }
    }
    Ok(out)
}

fn parse_opcode(instruction: u32) -> Result<Opcode, InstructionError> {
    let opcode = instruction & OPCODE_MASK;
    Opcode::try_from(opcode)
}

// Function Identifiers (func7 & func3)
const ADD_FUNC_IDENTIFIERS: (u32, u32) = (0x0, 0x00);
const SUB_FUNC_IDENTIFIERS: (u32, u32) = (0x0, 0x20);
const XOR_FUNC_IDENTIFIERS: (u32, u32) = (0x4, 0x00);
const OR_FUNC_IDENTIFIERS: (u32, u32) = (0x6, 0x00);
const AND_FUNC_IDENTIFIERS: (u32, u32) = (0x7, 0x00);
const SHL_FUNC_IDENTIFIERS: (u32, u32) = (0x1, 0x00);
const SRL_FUNC_IDENTIFIERS: (u32, u32) = (0x5, 0x00);
const SRA_FUNC_IDENTIFIERS: (u32, u32) = (0x5, 0x20);
const SLT_FUNC_IDENTIFIERS: (u32, u32) = (0x2, 0x00);
const SLTU_FUNC_IDENTIFIERS: (u32, u32) = (0x3, 0x00);
const MUL_FUNC_IDENTIFIERS: (u32, u32) = (0x0, 0x01);
const MUL_H_FUNC_IDENTIFIERS: (u32, u32) = (0x1, 0x01);
const MUL_H_S_U_FUNC_IDENTIFIERS: (u32, u32) = (0x2, 0x01);
const MUL_H_U_FUNC_IDENTIFIERS: (u32, u32) = (0x3, 0x01);
const DIV_FUNC_IDENTIFIERS: (u32, u32) = (0x4, 0x01);
const DIV_U_FUNC_IDENTIFIERS: (u32, u32) = (0x5, 0x01);
const REM_FUNC_IDENTIFIERS: (u32, u32) = (0x6, 0x01);
const REM_U_FUNC_IDENTIFIERS: (u32, u32) = (0x7, 0x01);

// R-Type Instruction Format
// |func7 | rs2  | rs1  |funct3|  rd |opcode|
// |31..25|24..20|19..15|14..12|11..7| 6..0 |
fn parse_r_instruction(instruction: u32, opcode: Opcode) -> Result<Instruction, InstructionError> {
    let func7 = (instruction & FUNC7_MASK) >> 25;
    let func3 = (instruction & FUNC3_MASK) >> 12;
    let rs2 = (instruction & RS2_MASK) >> 20;
    let rs1 = (instruction & RS1_MASK) >> 15;
    let rd = (instruction & RD_MASK) >> 7;

    let operation = match (func3, func7) {
        ADD_FUNC_IDENTIFIERS => ArithOp::Add,
        SUB_FUNC_IDENTIFIERS => ArithOp::Sub,
        XOR_FUNC_IDENTIFIERS => ArithOp::Xor,
        OR_FUNC_IDENTIFIERS => ArithOp::Or,
        AND_FUNC_IDENTIFIERS => ArithOp::And,
        SHL_FUNC_IDENTIFIERS => ArithOp::ShiftLeftLogical,
        SRL_FUNC_IDENTIFIERS => ArithOp::ShiftRightLogical,
        SRA_FUNC_IDENTIFIERS => ArithOp::ShiftRightArith,
        SLT_FUNC_IDENTIFIERS => ArithOp::SetLessThan,
        SLTU_FUNC_IDENTIFIERS => ArithOp::SetLessThanU,
        MUL_FUNC_IDENTIFIERS => ArithOp::Mul,
        MUL_H_FUNC_IDENTIFIERS => ArithOp::MulHigh,
        MUL_H_S_U_FUNC_IDENTIFIERS => ArithOp::MulHighSignedUnsigned,
        MUL_H_U_FUNC_IDENTIFIERS => ArithOp::MulHighUnsigned,
        DIV_FUNC_IDENTIFIERS => ArithOp::Div,
        DIV_U_FUNC_IDENTIFIERS => ArithOp::DivUnsigned,
        REM_FUNC_IDENTIFIERS => ArithOp::Remainder,
        REM_U_FUNC_IDENTIFIERS => ArithOp::RemainderUnsigned,
        _ => return Err(InstructionError::UnknownOpcodeFuncIdentifier(opcode, func3)),
    };

    Ok(match opcode {
        Opcode::Arith => Instruction::Arith {
            dst: rd,
            src1: rs1,
            src2: rs2,
            op: operation,
        },
        Opcode::Arith32 => {
            // W-suffix instructions only support a subset of operations
            match operation {
                ArithOp::Add
                | ArithOp::Sub
                | ArithOp::ShiftLeftLogical
                | ArithOp::ShiftRightLogical
                | ArithOp::ShiftRightArith
                | ArithOp::Mul
                | ArithOp::Div
                | ArithOp::DivUnsigned
                | ArithOp::Remainder
                | ArithOp::RemainderUnsigned => Instruction::ArithW {
                    dst: rd,
                    src1: rs1,
                    src2: rs2,
                    op: operation,
                },
                _ => return Err(InstructionError::InvalidW32Instruction),
            }
        }
        _ => return Err(InstructionError::InvalidInstruction),
    })
}

// Function Identifiers (func3)
const ADD_FUNC_IDENTIFIER: u32 = 0x0;
const XOR_FUNC_IDENTIFIER: u32 = 0x4;
const OR_FUNC_IDENTIFIER: u32 = 0x6;
const AND_FUNC_IDENTIFIER: u32 = 0x7;
const SHL_FUNC_IDENTIFIER: u32 = 0x1;
const SR_FUNC_IDENTIFIER: u32 = 0x5;
const SLT_FUNC_IDENTIFIER: u32 = 0x2;
const SLTU_FUNC_IDENTIFIER: u32 = 0x3;
const CSRRW_FUNC_IDENTIFIER: u32 = 0x1;
const CSRRS_FUNC_IDENTIFIER: u32 = 0x2;
const CSRRC_FUNC_IDENTIFIER: u32 = 0x3;
const CSRRWI_FUNC_IDENTIFIER: u32 = 0x5;
const CSRRSI_FUNC_IDENTIFIER: u32 = 0x6;
const CSRRCI_FUNC_IDENTIFIER: u32 = 0x7;
const ECALL_EBREAK_FUNC_IDENTIFIER: u32 = 0x0;

// I-Type Instruction Format
// | imm  | rs1  |funct3|  rd |opcode|
// |31..20|19..15|14..12|11..7| 6..0 |
fn parse_i_instruction(instruction: u32, opcode: Opcode) -> Result<Instruction, InstructionError> {
    let func3 = (instruction & FUNC3_MASK) >> 12;
    let rs1 = (instruction & RS1_MASK) >> 15;
    let csr = (instruction >> 20) & I_TYPE_IMM_MASK;
    let imm = csr as i32;
    let mut imm: i32 = if (instruction & SIGN_MASK) != 0 {
        imm.wrapping_sub(1 << 12)
    } else {
        imm
    };

    let rd = (instruction & RD_MASK) >> 7;
    Ok(match opcode {
        Opcode::ArithImm => {
            let operation = match func3 {
                ADD_FUNC_IDENTIFIER => ArithOp::Add,
                XOR_FUNC_IDENTIFIER => ArithOp::Xor,
                OR_FUNC_IDENTIFIER => ArithOp::Or,
                AND_FUNC_IDENTIFIER => ArithOp::And,
                SHL_FUNC_IDENTIFIER => {
                    // RV64: shift amount is 6 bits (imm[5:0])
                    let func_id = imm >> 6;
                    if func_id != 0 {
                        return Err(InstructionError::UnknownSLVariant(func_id));
                    }
                    imm &= 0x3F; // 6-bit shift amount for RV64
                    ArithOp::ShiftLeftLogical
                }
                SR_FUNC_IDENTIFIER => {
                    // RV64: shift amount is 6 bits, func7 is in bits [11:6]
                    let func_id = imm >> 6;
                    imm &= 0x3F; // 6-bit shift amount for RV64
                    match func_id {
                        0x00 => ArithOp::ShiftRightLogical,
                        0x10 => ArithOp::ShiftRightArith, // 0x20 >> 1 = 0x10 when looking at bits [11:6]
                        _ => return Err(InstructionError::UnknownSRVariant(func_id)),
                    }
                }
                SLT_FUNC_IDENTIFIER => ArithOp::SetLessThan,
                SLTU_FUNC_IDENTIFIER => ArithOp::SetLessThanU,
                _ => return Err(InstructionError::UnknownOpcodeFuncIdentifier(opcode, func3)),
            };
            Instruction::ArithImm {
                dst: rd,
                src: rs1,
                imm,
                op: operation,
            }
        }
        Opcode::ArithImm32 => {
            // W-suffix immediate instructions (ADDIW, SLLIW, SRLIW, SRAIW)
            let operation = match func3 {
                ADD_FUNC_IDENTIFIER => ArithOp::Add,
                SHL_FUNC_IDENTIFIER => {
                    // SLLIW: shift amount is 5 bits for 32-bit operation
                    let func_id = imm >> 5;
                    if func_id != 0 {
                        return Err(InstructionError::UnknownSLVariant(func_id));
                    }
                    imm &= 0x1F; // 5-bit shift amount for W instructions
                    ArithOp::ShiftLeftLogical
                }
                SR_FUNC_IDENTIFIER => {
                    // SRLIW/SRAIW: shift amount is 5 bits
                    let func_id = imm >> 5;
                    imm &= 0x1F; // 5-bit shift amount for W instructions
                    match func_id {
                        0x00 => ArithOp::ShiftRightLogical,
                        0x20 => ArithOp::ShiftRightArith,
                        _ => return Err(InstructionError::UnknownSRVariant(func_id)),
                    }
                }
                _ => return Err(InstructionError::InvalidW32Instruction),
            };
            Instruction::ArithImmW {
                dst: rd,
                src: rs1,
                imm,
                op: operation,
            }
        }
        Opcode::JumpAndLinkRegister => {
            if func3 != 0x00 {
                return Err(InstructionError::InvalidJALR);
            };
            Instruction::JumpAndLinkRegister {
                base: rs1,
                dst: rd,
                offset: imm,
            }
        }
        Opcode::Load => Instruction::Load {
            dst: rd,
            offset: imm,
            base: rs1,
            width: LoadStoreWidth::from_func3(func3)?,
        },
        Opcode::System => {
            match func3 {
                ECALL_EBREAK_FUNC_IDENTIFIER => Instruction::EcallEbreak,
                CSRRCI_FUNC_IDENTIFIER => Instruction::CSR {
                    csr,
                    src: rs1,
                    dst: rd,
                    op: CsrOp::CSRRCI,
                },
                CSRRS_FUNC_IDENTIFIER => Instruction::CSR {
                    csr,
                    src: rs1,
                    dst: rd,
                    op: CsrOp::CSRRS,
                },
                CSRRW_FUNC_IDENTIFIER => Instruction::CSR {
                    csr,
                    src: rs1,
                    dst: rd,
                    op: CsrOp::CSRRW,
                },
                CSRRC_FUNC_IDENTIFIER | CSRRWI_FUNC_IDENTIFIER | CSRRSI_FUNC_IDENTIFIER => {
                    // For now, we do not support these CSR instructions
                    return Err(InstructionError::InvalidSystemInstruction(func3));
                }
                _ => return Err(InstructionError::UnknownOpcodeFuncIdentifier(opcode, func3)),
            }
        }
        Opcode::Fence => Instruction::Fence,
        _ => return Err(InstructionError::InvalidInstruction),
    })
}

// S-Type Instruction Format
// imm[11:5] rs2 rs1 funct3 imm[4:0] opcode
// |imm[11:5]| rs2  | rs1  |funct3|imm[4:0]|opcode|
// | 31..25  |24..20|19..15|14..12| 11..7  | 6..0 |
fn parse_s_instruction(instruction: u32, opcode: Opcode) -> Result<Instruction, InstructionError> {
    let func7 = ((instruction & FUNC7_MASK) >> 25) << 5;
    let func3 = (instruction & FUNC3_MASK) >> 12;
    let rs2 = (instruction & RS2_MASK) >> 20;
    let rs1 = (instruction & RS1_MASK) >> 15;
    let rd = (instruction & RD_MASK) >> 7;
    let imm = (func7 | rd) as i32;
    let imm: i32 = if (instruction & SIGN_MASK) != 0 {
        imm.wrapping_sub(1 << 12)
    } else {
        imm
    };

    Ok(match opcode {
        Opcode::Store => Instruction::Store {
            src: rs2,
            offset: imm,
            base: rs1,
            width: LoadStoreWidth::from_func3_store(func3)?,
        },
        _ => return Err(InstructionError::InvalidInstruction),
    })
}

// Function Identifiers (func3)
const BRANCH_EQ_IDENTIFIER: u32 = 0x0;
const BRANCH_NEQ_IDENTIFIER: u32 = 0x1;
const BRANCH_LT_IDENTIFIER: u32 = 0x4;
const BRANCH_GE_IDENTIFIER: u32 = 0x5;
const BRANCH_LTU_IDENTIFIER: u32 = 0x6;
const BRANCH_GTU_IDENTIFIER: u32 = 0x7;

// B-Type Instruction Format
// |imm[12|10:5]| rs2  | rs1  |funct3|imm[4:1|11]|opcode|
// |    31..25  |24..20|19..15|14..12|  11..7    | 6..0 |
fn parse_b_instruction(instruction: u32, opcode: Opcode) -> Result<Instruction, InstructionError> {
    let func3 = (instruction & FUNC3_MASK) >> 12;
    let rs2 = (instruction & RS2_MASK) >> 20;
    let rs1 = (instruction & RS1_MASK) >> 15;
    let imm = (((instruction >> 20) & 0x7e0)
        | ((instruction >> 7) & 0x1e)
        | ((instruction & 0x80) << 4)) as i32;
    let imm: i32 = if (instruction & SIGN_MASK) != 0 {
        imm.wrapping_add(0xFFFFF000u32 as i32)
    } else {
        imm
    };
    Ok(match opcode {
        Opcode::Branch => {
            let comparison = match func3 {
                BRANCH_EQ_IDENTIFIER => Comparison::Equal,
                BRANCH_NEQ_IDENTIFIER => Comparison::NotEqual,
                BRANCH_LT_IDENTIFIER => Comparison::LessThan,
                BRANCH_GE_IDENTIFIER => Comparison::GreaterOrEqual,
                BRANCH_LTU_IDENTIFIER => Comparison::LessThanUnsigned,
                BRANCH_GTU_IDENTIFIER => Comparison::GreaterOrEqualUnsigned,
                _ => return Err(InstructionError::UnknownOpcodeFuncIdentifier(opcode, func3)),
            };
            Instruction::Branch {
                src1: rs1,
                src2: rs2,
                cond: comparison,
                offset: imm,
            }
        }
        _ => return Err(InstructionError::InvalidInstruction),
    })
}

// J-Type Instruction Format
// |imm[20|10:1|11|19:12] | rd  |opcode|
// |         31..12       |11..7| 6..0 |
fn parse_j_instruction(instruction: u32, opcode: Opcode) -> Result<Instruction, InstructionError> {
    let imm =
        instruction & 0xff000 | ((instruction & 0x100000) >> 9) | ((instruction >> 20) & 0x7fe);
    let imm: i32 = if (instruction & SIGN_MASK) != 0 {
        (imm as i32).wrapping_sub(1 << 20)
    } else {
        imm as i32
    };
    let rd = (instruction & RD_MASK) >> 7;
    Ok(match opcode {
        Opcode::JumpAndLink => Instruction::JumpAndLink {
            dst: rd,
            offset: imm,
        },
        _ => return Err(InstructionError::InvalidInstruction),
    })
}

// U-Type Instruction Format
// |imm[31:12] | rd  |opcode|
// | 31..12    |11..7| 6..0 |
fn parse_u_instruction(instruction: u32, opcode: Opcode) -> Result<Instruction, InstructionError> {
    let imm = instruction & U_TYPE_IMM_MASK;
    let rd = (instruction & RD_MASK) >> 7;
    Ok(match opcode {
        Opcode::LoadUpperImm => Instruction::LoadUpperImm { dst: rd, imm },
        Opcode::AddUpperImmToPc => Instruction::AddUpperImmToPc { dst: rd, imm },
        _ => return Err(InstructionError::InvalidInstruction),
    })
}

#[derive(thiserror::Error, Debug)]
pub enum InstructionError {
    #[error("Unknown Opcode {0:0x}")]
    UnknownOpcode(u32),
    #[error("Unknown func3 component {1:0x} for Instruction with Opcode {0:?}")]
    UnknownOpcodeFuncIdentifier(Opcode, u32),
    #[error("Invalid instruction encoding")]
    InvalidInstruction,
    #[error("Invalid width for Load/Store instruction: {0:0x}")]
    InvalidLoadStoreWidth(u32),
    #[error("Unknown ShiftRight variant: {0:0x}")]
    UnknownSRVariant(i32),
    #[error("Unknown ShiftLeftvariant: {0:0x}")]
    UnknownSLVariant(i32),
    #[error("Invalid JALR Instruction: func3 component is not 0x0")]
    InvalidJALR,
    #[error("Invalid system instruction encoding with func3: {0:0x}")]
    InvalidSystemInstruction(u32),
    #[error("Invalid W32 instruction: operation not supported")]
    InvalidW32Instruction,
    #[error("Illegal compressed instruction encoding: {0:#06x}")]
    IllegalCompressed(u16),
    #[error("Reserved compressed instruction encoding: {0:#06x}")]
    ReservedCompressed(u16),
}

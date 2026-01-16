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
            &Opcode::Arith => InstructionFormat::R,
            &Opcode::ArithImm | &Opcode::Load | &Opcode::JumpAndLinkRegister | &Opcode::System => {
                InstructionFormat::I
            }
            &Opcode::Store => InstructionFormat::S,
            &Opcode::Branch => InstructionFormat::B,
            &Opcode::JumpAndLink => InstructionFormat::J,
            &Opcode::LoadUpperImm | &Opcode::AddUpperImmToPc => InstructionFormat::U,
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
const LOAD_BYTE_UNSIGNED_FUNC: u32 = 0x4;
const LOAD_HALF_UNSIGNED_FUNC: u32 = 0x5;

#[derive(Debug, Clone, Copy)]
pub enum LoadStoreWidth {
    Byte,
    Half,
    Word,
    ByteUnsigned,
    HalfUnsigned,
}

impl LoadStoreWidth {
    fn from_func3(func3: u32) -> Result<LoadStoreWidth, InstructionError> {
        Ok(match func3 {
            LOAD_STORE_BYTE_WIDTH => LoadStoreWidth::Byte,
            LOAD_STORE_HALF_WIDTH => LoadStoreWidth::Half,
            LOAD_STORE_WORD_WIDTH => LoadStoreWidth::Word,
            LOAD_BYTE_UNSIGNED_FUNC => LoadStoreWidth::ByteUnsigned,
            LOAD_HALF_UNSIGNED_FUNC => LoadStoreWidth::HalfUnsigned,
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

#[derive(Debug, Clone)]
pub enum Instruction {
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
    Ok(match opcode {
        Opcode::Arith => {
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
            Instruction::Arith {
                dst: rd,
                src1: rs1,
                src2: rs2,
                op: operation,
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
                    let func_id = imm >> 5;
                    if func_id != 0 {
                        return Err(InstructionError::UnknownSLVariant(func_id));
                    }
                    imm &= 0x1F;
                    ArithOp::ShiftLeftLogical
                }
                SR_FUNC_IDENTIFIER => {
                    let func_id = imm >> 5;
                    imm &= 0x1F;
                    match func_id {
                        0x00 => ArithOp::ShiftRightLogical,
                        0x20 => ArithOp::ShiftRightArith,
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
            width: LoadStoreWidth::from_func3(func3)?,
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
}

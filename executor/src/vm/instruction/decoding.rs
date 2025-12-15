// Opcodes
const ARITH_OPCODE: u32 = 0b0110011;
const ARITH_IMM_OPCODE: u32 = 0b0010011;
const LOAD_OPCODE: u32 = 0b0000011;
const STORE_OPCODE: u32 = 0b0100011;
const BRANCH_OPCODE: u32 = 0b1100011;
const JUMP_AND_LINK_REGISTER_OPCCODE: u32 = 0b1100111;
const JUMP_AND_LINK_OPCCODE: u32 = 0b1101111;
const LOAD_UPPER_IMM_OPCODE: u32 = 0b0110111;
const ADD_UPPER_IMM_TO_PC: u32 = 0b0010111;

enum Opcode {
    Arith,
    ArithImm,
    Load,
    Store,
    Branch,
    JumpAndLinkRegister,
    JumpAndLink,
    LoadUpperImm,
    AddUpperImmToPc,
}

impl TryFrom<u32> for Opcode {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            ARITH_OPCODE => Opcode::Arith,
            ARITH_IMM_OPCODE => Opcode::ArithImm,
            LOAD_OPCODE => Opcode::Load,
            STORE_OPCODE => Opcode::Store,
            BRANCH_OPCODE => Opcode::Branch,
            JUMP_AND_LINK_REGISTER_OPCCODE => Opcode::JumpAndLinkRegister,
            JUMP_AND_LINK_OPCCODE => Opcode::JumpAndLink,
            LOAD_UPPER_IMM_OPCODE => Opcode::LoadUpperImm,
            ADD_UPPER_IMM_TO_PC => Opcode::AddUpperImmToPc,
            _ => panic!("Unknown Opcode: {value}"),
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
            &Opcode::ArithImm | &Opcode::Load | &Opcode::JumpAndLinkRegister => {
                InstructionFormat::I
            }
            &Opcode::Store => InstructionFormat::S,
            &Opcode::Branch => InstructionFormat::B,
            &Opcode::JumpAndLink => InstructionFormat::J,
            &Opcode::LoadUpperImm | &Opcode::AddUpperImmToPc => InstructionFormat::U,
        }
    }
}

#[derive(Debug)]
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
}

#[derive(Debug)]
pub enum LoadStoreWidth {
    Byte,
    Half,
    Word,
}

impl LoadStoreWidth {
    fn from_func3(func3: u32) -> LoadStoreWidth {
        const LOAD_STORE_BYTE_WIDTH: u32 = 0x0;
        const LOAD_STORE_HALF_WIDTH: u32 = 0x1;
        const LOAD_STORE_WORD_WIDTH: u32 = 0x2;
        match func3 {
            LOAD_STORE_BYTE_WIDTH => LoadStoreWidth::Byte,
            LOAD_STORE_HALF_WIDTH => LoadStoreWidth::Half,
            LOAD_STORE_WORD_WIDTH => LoadStoreWidth::Word,
            _ => panic!("Invalid Width"),
        }
    }
}

#[derive(Debug)]
pub enum Comparison {
    Equal,
    NotEqual,
    LessThan,
    GreaterOrEqual,
    LessThanUnsigned,
    GreaterOrEqualUnsigned,
}

#[derive(Debug)]
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
        offset: u32,
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
}

const OPCODE_MASK: u32 = 0x0000007f;
const FUNC7_MASK: u32 = 0xfe000000;
const FUNC3_MASK: u32 = 0x00007000;
const RS1_MASK: u32 = 0x000f8000;
const RS2_MASK: u32 = 0x01f00000;
const RD_MASK: u32 = 0x00000f80;
const SIGN_MASK: u32 = 0x80000000;
const I_TYPE_IMM_MASK: u32 = 0x7ff;
const U_TYPE_IMM_MASK: u32 = 0xfffff000;

impl Instruction {
    pub fn parse(instruction: u32) -> Instruction {
        let opcode = parse_opcode(instruction);
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

fn parse_opcode(instruction: u32) -> Opcode {
    let opcode = instruction & OPCODE_MASK;
    Opcode::try_from(opcode).unwrap()
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

// R-Type Instruction Format
// |func7 | rs2  | rs1  |funct3|  rd |opcode|
// |31..25|24..20|19..15|14..12|11..7| 6..0 |
fn parse_r_instruction(instruction: u32, opcode: Opcode) -> Instruction {
    let func7 = (instruction & FUNC7_MASK) >> 25;
    let func3 = (instruction & FUNC3_MASK) >> 12;
    let rs2 = (instruction & RS2_MASK) >> 20;
    let rs1 = (instruction & RS1_MASK) >> 15;
    let rd = (instruction & RD_MASK) >> 7;
    match opcode {
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
                _ => panic!("Unknown  arith opcode identifier"),
            };
            Instruction::Arith {
                dst: rd,
                src1: rs1,
                src2: rs2,
                op: operation,
            }
        }
        _ => panic!("Invalid Instruction Encoding"),
    }
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

// I-Type Instruction Format
// | imm  | rs1  |funct3|  rd |opcode|
// |31..20|19..15|14..12|11..7| 6..0 |
fn parse_i_instruction(instruction: u32, opcode: Opcode) -> Instruction {
    let func3 = (instruction & FUNC3_MASK) >> 12;
    let rs1 = (instruction & RS1_MASK) >> 15;
    let imm = ((instruction >> 20) & I_TYPE_IMM_MASK) as i32;
    let mut imm: i32 = if (instruction & SIGN_MASK) != 0 {
        imm - (1 << 11)
    } else {
        imm
    };

    let rd = (instruction & RD_MASK) >> 7;
    match opcode {
        Opcode::ArithImm => {
            let operation = match func3 {
                ADD_FUNC_IDENTIFIER => ArithOp::Add,
                XOR_FUNC_IDENTIFIER => ArithOp::Xor,
                OR_FUNC_IDENTIFIER => ArithOp::Or,
                AND_FUNC_IDENTIFIER => ArithOp::And,
                SHL_FUNC_IDENTIFIER => {
                    assert!(imm >> 5 == 0);
                    imm &= 0x1F;
                    ArithOp::ShiftLeftLogical
                }
                SR_FUNC_IDENTIFIER => {
                    let func_id = imm >> 5;
                    imm &= 0x1F;
                    match func_id {
                        0x00 => ArithOp::ShiftRightLogical,
                        0x20 => ArithOp::ShiftRightArith,
                        _ => unimplemented!(),
                    }
                }
                SLT_FUNC_IDENTIFIER => ArithOp::SetLessThan,
                SLTU_FUNC_IDENTIFIER => ArithOp::SetLessThanU,
                _ => panic!("Unknown  arith opcode identifier"),
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
                panic!("Invalid JALR Instruction")
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
            width: LoadStoreWidth::from_func3(func3),
        },
        _ => panic!("Invalid Instruction Encoding"),
    }
}

// S-Type Instruction Format
// imm[11:5] rs2 rs1 funct3 imm[4:0] opcode
// |imm[11:5]| rs2  | rs1  |funct3|imm[4:0]|opcode|
// | 31..25  |24..20|19..15|14..12| 11..7  | 6..0 |
fn parse_s_instruction(instruction: u32, opcode: Opcode) -> Instruction {
    let func7 = (instruction & FUNC7_MASK) >> 25;
    let func3 = (instruction & FUNC3_MASK) >> 12;
    let rs2 = (instruction & RS2_MASK) >> 20;
    let rs1 = (instruction & RS1_MASK) >> 15;
    let rd = (instruction & RD_MASK) >> 7;
    let imm = func7 | rd;
    match opcode {
        Opcode::Store => Instruction::Store {
            src: rs2,
            offset: imm,
            base: rs1,
            width: LoadStoreWidth::from_func3(func3),
        },
        _ => panic!("Invalid Instruction Encoding"),
    }
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
fn parse_b_instruction(instruction: u32, opcode: Opcode) -> Instruction {
    let func3 = (instruction & FUNC3_MASK) >> 12;
    let rs2 = (instruction & RS2_MASK) >> 20;
    let rs1 = (instruction & RS1_MASK) >> 15;
    let imm = (((instruction >> 20) & 0x7e0)
        | ((instruction >> 7) & 0x1e)
        | ((instruction & 0x80) << 4)) as i32;
    let imm: i32 = if (instruction & SIGN_MASK) != 0 {
        imm + 0xFFFFF000u32 as i32
    } else {
        imm
    };
    match opcode {
        Opcode::Branch => {
            let comparison = match func3 {
                BRANCH_EQ_IDENTIFIER => Comparison::Equal,
                BRANCH_NEQ_IDENTIFIER => Comparison::NotEqual,
                BRANCH_LT_IDENTIFIER => Comparison::LessThan,
                BRANCH_GE_IDENTIFIER => Comparison::GreaterOrEqual,
                BRANCH_LTU_IDENTIFIER => Comparison::LessThanUnsigned,
                BRANCH_GTU_IDENTIFIER => Comparison::GreaterOrEqualUnsigned,
                _ => unimplemented!(),
            };
            Instruction::Branch {
                src1: rs1,
                src2: rs2,
                cond: comparison,
                offset: imm,
            }
        }
        _ => panic!("Unknown Opcode"),
    }
}

// J-Type Instruction Format
// |imm[20|10:1|11|19:12] | rd  |opcode|
// |         31..12       |11..7| 6..0 |
fn parse_j_instruction(instruction: u32, opcode: Opcode) -> Instruction {
    let imm =
        instruction & 0xff000 | ((instruction & 0x100000) >> 9) | ((instruction >> 20) & 0x7fe);
    let imm: i32 = if (instruction & SIGN_MASK) != 0 {
        imm as i32 - (1 << 20)
    } else {
        imm as i32
    };
    let rd = (instruction & RD_MASK) >> 7;
    match opcode {
        Opcode::JumpAndLink => Instruction::JumpAndLink {
            dst: rd,
            offset: imm,
        },
        _ => unimplemented!(),
    }
}

// U-Type Instruction Format
// |imm[31:12] | rd  |opcode|
// | 31..12    |11..7| 6..0 |
fn parse_u_instruction(instruction: u32, opcode: Opcode) -> Instruction {
    let imm = instruction & U_TYPE_IMM_MASK;
    let rd = (instruction & RD_MASK) >> 7;
    match opcode {
        Opcode::LoadUpperImm => Instruction::LoadUpperImm { dst: rd, imm },
        Opcode::AddUpperImmToPc => Instruction::AddUpperImmToPc { dst: rd, imm },
        _ => unimplemented!(),
    }
}

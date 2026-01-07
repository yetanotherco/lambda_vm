use crate::utils::{i32_to_2_limbs, i32_to_4_limbs, u32_to_2_limbs, u32_to_4_limbs};
use executor::vm::{
    instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
    logs::Log,
};
use math::field::{
    element::FieldElement, fields::fft_friendly::babybear_u32::Babybear31PrimeField,
};

type FE = FieldElement<Babybear31PrimeField>;

pub const NUM_COLUMNS: usize = 15;

pub mod instruction {
    pub const ADD: u32 = 1;
    pub const SUB: u32 = 1 << 1;
    pub const SLT: u32 = 1 << 2;
    pub const AND: u32 = 1 << 3;
    pub const OR: u32 = 1 << 4;
    pub const XOR: u32 = 1 << 5;
    pub const SL: u32 = 1 << 6;
    pub const SR: u32 = 1 << 7;
    pub const JALR: u32 = 1 << 8;
    pub const BEQ: u32 = 1 << 9;
    pub const BLT: u32 = 1 << 10;
    pub const LOAD: u32 = 1 << 11;
    pub const STORE: u32 = 1 << 12;
    pub const MUL: u32 = 1 << 13;
    pub const DIVREM: u32 = 1 << 14;
    pub const ECALL: u32 = 1 << 15;
    pub const EBREAK: u32 = 1 << 16;
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct DecodeKey {
    pub pc: [FE; 2],
    pub rs1: FE,
    pub rs2: FE,
    pub rd: FE,
    pub write_register: FE,
    pub memory_2bytes: FE,
    pub memory_4bytes: FE,
    pub imm: [FE; 2],
    pub signed: FE,
    pub mp_selector: FE,
    pub muldiv_selector: FE,
    pub instruction: FE,
}

#[derive(Default)]
pub struct DecodeTableRow {
    pub pc: [FE; 2],
    pub rs1: FE,
    pub rs2: FE,
    pub rd: FE,
    pub write_register: FE,
    pub memory_2bytes: FE,
    pub memory_4bytes: FE,
    pub imm: [FE; 2],
    pub signed: FE,
    pub mp_selector: FE,
    pub muldiv_selector: FE,
    pub instruction: FE,
    pub multiplicity: FE,
}

impl DecodeTableRow {
    pub fn from_log(log: &Log) -> Self {
        let mut row = Self {
            pc: u32_to_2_limbs(log.current_pc),
            ..Default::default()
        };

        match log.instruction {
            Instruction::Arith {
                dst,
                src1,
                src2,
                op,
            } => {
                row.rd = FE::from(&dst);
                row.rs1 = FE::from(&src1);
                row.rs2 = FE::from(&src2);
                if dst != 0 {
                    row.write_register = FE::one();
                }
                match op {
                    ArithOp::Add => row.instruction = FE::from(&instruction::ADD),
                    ArithOp::Sub => row.instruction = FE::from(&instruction::SUB),
                    ArithOp::Xor => row.instruction = FE::from(&instruction::XOR),
                    ArithOp::Or => row.instruction = FE::from(&instruction::OR),
                    ArithOp::And => row.instruction = FE::from(&instruction::AND),
                    ArithOp::ShiftLeftLogical => row.instruction = FE::from(&instruction::SL),
                    ArithOp::ShiftRightLogical => row.instruction = FE::from(&instruction::SR),
                    ArithOp::ShiftRightArith => {
                        row.instruction = FE::from(&instruction::SR);
                        row.signed = FE::one();
                    }
                    ArithOp::SetLessThan => {
                        row.instruction = FE::from(&instruction::SLT);
                        row.signed = FE::one();
                    }
                    ArithOp::SetLessThanU => row.instruction = FE::from(&instruction::SLT),
                    ArithOp::Mul => {
                        row.instruction = FE::from(&instruction::MUL);
                        row.mp_selector = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::MulHigh => {
                        row.instruction = FE::from(&instruction::MUL);
                        row.muldiv_selector = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::MulHighSignedUnsigned => {
                        row.instruction = FE::from(&instruction::MUL);
                        row.muldiv_selector = FE::one();
                        row.mp_selector = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::MulHighUnsigned => {
                        row.instruction = FE::from(&instruction::MUL);
                        row.muldiv_selector = FE::one();
                    }
                    ArithOp::Div => {
                        row.instruction = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::DivUnsigned => {
                        row.instruction = FE::from(&instruction::DIVREM);
                    }
                    ArithOp::Remainder => {
                        row.instruction = FE::from(&instruction::DIVREM);
                        row.muldiv_selector = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::RemainderUnsigned => {
                        row.instruction = FE::from(&instruction::DIVREM);
                        row.muldiv_selector = FE::one();
                    }
                }
            }

            Instruction::ArithImm { dst, src, imm, op } => {
                row.rd = FE::from(&dst);
                row.rs1 = FE::from(&src);
                row.rs2 = FE::zero();
                row.imm = i32_to_2_limbs(imm);
                if dst != 0 {
                    row.write_register = FE::one();
                }
                match op {
                    ArithOp::Add => row.instruction = FE::from(&instruction::ADD),
                    ArithOp::Sub => row.instruction = FE::from(&instruction::SUB),
                    ArithOp::Xor => row.instruction = FE::from(&instruction::XOR),
                    ArithOp::Or => row.instruction = FE::from(&instruction::OR),
                    ArithOp::And => row.instruction = FE::from(&instruction::AND),
                    ArithOp::ShiftLeftLogical => row.instruction = FE::from(&instruction::SL),
                    ArithOp::ShiftRightLogical => row.instruction = FE::from(&instruction::SR),
                    ArithOp::ShiftRightArith => {
                        row.instruction = FE::from(&instruction::SR);
                        row.signed = FE::one();
                    }
                    ArithOp::SetLessThan => {
                        row.instruction = FE::from(&instruction::SLT);
                        row.signed = FE::one();
                    }
                    ArithOp::SetLessThanU => row.instruction = FE::from(&instruction::SLT),
                    _ => todo!(),
                }
            }

            Instruction::JumpAndLink { dst, offset } => {
                row.instruction = FE::from(&instruction::JALR);
                row.rd = FE::from(&dst);
                row.imm = i32_to_2_limbs(offset);
                if dst != 0 {
                    row.write_register = FE::one();
                }
            }

            Instruction::JumpAndLinkRegister { base, dst, offset } => {
                row.instruction = FE::from(&instruction::JALR);
                row.rd = FE::from(&dst);
                row.rs1 = FE::from(&base);
                row.imm = i32_to_2_limbs(offset);
                if dst != 0 {
                    row.write_register = FE::one();
                }
            }

            Instruction::Store {
                src,
                offset,
                base,
                width,
            } => {
                row.instruction = FE::from(&instruction::STORE);
                row.rs1 = FE::from(&base);
                row.rs2 = FE::from(&src);
                // Fix this afer changing STORE instruction.
                row.imm = i32_to_2_limbs(offset as i32);

                match width {
                    LoadStoreWidth::Half => row.memory_2bytes = FE::one(),
                    LoadStoreWidth::Word => {
                        row.memory_2bytes = FE::one();
                        row.memory_4bytes = FE::one();
                    }
                    _ => (),
                }
            }

            Instruction::Load {
                dst,
                offset,
                base,
                width,
            } => {
                row.instruction = FE::from(&instruction::LOAD);
                row.rd = FE::from(&dst);
                row.rs1 = FE::from(&base);
                row.imm = i32_to_2_limbs(offset);

                if dst != 0 {
                    row.write_register = FE::one();
                }

                match width {
                    LoadStoreWidth::Half => row.memory_2bytes = FE::one(),
                    LoadStoreWidth::Word => {
                        row.memory_2bytes = FE::one();
                        row.memory_4bytes = FE::one();
                    }
                    _ => (),
                }
            }

            Instruction::Branch {
                src1,
                src2,
                cond,
                offset,
            } => {
                row.rs1 = FE::from(&src1);
                row.rs2 = FE::from(&src2);
                row.imm = u32_to_2_limbs(offset as u32);
                match cond {
                    Comparison::Equal => row.instruction = FE::from(&instruction::BEQ),
                    Comparison::NotEqual => {
                        row.instruction = FE::from(&instruction::BEQ);
                        row.mp_selector = FE::one()
                    }
                    Comparison::LessThan => {
                        row.instruction = FE::from(&instruction::BLT);
                        row.signed = FE::one();
                    }
                    Comparison::LessThanUnsigned => row.instruction = FE::from(&instruction::BLT),
                    Comparison::GreaterOrEqual => {
                        row.instruction = FE::from(&instruction::BLT);
                        row.signed = FE::one();
                        row.mp_selector = FE::one()
                    }
                    Comparison::GreaterOrEqualUnsigned => {
                        row.instruction = FE::from(&instruction::BLT);
                        row.mp_selector = FE::one()
                    }
                }
            }

            Instruction::LoadUpperImm { dst, imm } => {
                row.instruction = FE::from(&instruction::ADD);
                row.rd = FE::from(&dst);
                row.rs1 = FE::zero();
                row.rs2 = FE::zero();
                row.imm = u32_to_2_limbs(imm);
                if dst != 0 {
                    row.write_register = FE::one();
                }
            }

            Instruction::AddUpperImmToPc { dst, imm } => {
                row.instruction = FE::from(&instruction::ADD);
                row.rd = FE::from(&dst);
                row.imm = u32_to_2_limbs(imm);
                if dst != 0 {
                    row.write_register = FE::one();
                }
            }

            _ => {}
        }
        row
    }

    pub fn set_multiplicity(&mut self, multiplicity: usize) {
        self.multiplicity = FE::from(multiplicity as u64);
    }

    pub fn to_vec(self) -> Vec<FE> {
        let mut row = Vec::with_capacity(NUM_COLUMNS);

        // pc[2]
        row.extend_from_slice(&self.pc);
        row.push(self.rs1);
        row.push(self.rs2);
        row.push(self.rd);
        row.push(self.write_register);
        row.push(self.memory_2bytes);
        row.push(self.memory_4bytes);
        // imm[2]
        row.extend_from_slice(&self.imm);
        row.push(self.signed);
        row.push(self.mp_selector);
        row.push(self.muldiv_selector);
        row.push(self.multiplicity);

        row
    }

    pub fn to_key(&self) -> DecodeKey {
        DecodeKey {
            pc: self.pc,
            rs1: self.rs1,
            rs2: self.rs2,
            rd: self.rd,
            write_register: self.write_register,
            memory_2bytes: self.memory_2bytes,
            memory_4bytes: self.memory_4bytes,
            imm: self.imm,
            signed: self.signed,
            mp_selector: self.mp_selector,
            muldiv_selector: self.muldiv_selector,
            instruction: self.instruction,
        }
    }
}

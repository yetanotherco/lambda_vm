use crate::utils::{i32_to_4_limbs, u32_to_2_limbs, u32_to_4_limbs};
use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::babybear_u32::Babybear31PrimeField,
};
use vm::vm::{
    instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
    logs::Log,
};

use stark_platinum_prover::trace::TraceTable;

type FE = FieldElement<Babybear31PrimeField>;

pub struct CpuTable {
    pub rows: Vec<CpuTableRow>,
}

#[derive(Default)]
pub struct CpuTableRow {
    pub timestamp: [FE; 2],
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

    pub add: FE,
    pub sub: FE,
    pub slt: FE,
    pub and: FE,
    pub or: FE,
    pub xor: FE,
    pub sl: FE,
    pub sr: FE,
    pub jalr: FE,
    pub beq: FE,
    pub blt: FE,
    pub load: FE,
    pub store: FE,
    pub mul: FE,
    pub divrem: FE,
    pub ecall: FE,
    pub ebreak: FE,

    pub next_pc: [FE; 2],
    pub rv1: [FE; 4],
    pub rv2: [FE; 4],
    pub rvd: [FE; 2],
    pub arg2: [FE; 4],
    pub res: [FE; 4],

    pub is_equal: FE,
    pub branch_cond: FE,
}

impl CpuTable {
    pub fn from_logs(logs: Vec<Log>) -> Self {
        let rows = logs
            .into_iter()
            .enumerate()
            .map(|(i, log)| CpuTableRow::from_log(log, (i * 4) as u32))
            .collect();
        CpuTable { rows }
    }
}

impl CpuTableRow {
    pub fn from_log(log: Log, timestamp: u32) -> Self {
        let mut row = Self {
            timestamp: u32_to_2_limbs(timestamp),
            pc: u32_to_2_limbs(log.current_pc),
            next_pc: u32_to_2_limbs(log.next_pc),
            rv1: u32_to_4_limbs(log.src1_val),
            rv2: u32_to_4_limbs(log.src2_val),
            rvd: u32_to_2_limbs(log.dst_val),
            res: u32_to_4_limbs(log.dst_val),
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
                row.arg2 = row.rv2;
                if dst != 0 {
                    row.write_register = FE::one();
                }
                match op {
                    ArithOp::Add => row.add = FE::one(),
                    ArithOp::Sub => row.sub = FE::one(),
                    ArithOp::Xor => row.xor = FE::one(),
                    ArithOp::Or => row.or = FE::one(),
                    ArithOp::And => row.and = FE::one(),
                    ArithOp::ShiftLeftLogical => row.sl = FE::one(),
                    ArithOp::ShiftRightLogical => row.sr = FE::one(),
                    ArithOp::ShiftRightArith => {
                        row.sr = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::SetLessThan => {
                        row.slt = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::SetLessThanU => row.slt = FE::one(),
                }
            }

            Instruction::ArithImm { dst, src, imm, op } => {
                row.rd = FE::from(&dst);
                row.rs1 = FE::from(&src);
                row.rs2 = FE::zero();
                row.arg2 = i32_to_4_limbs(imm);
                if dst != 0 {
                    row.write_register = FE::one();
                }
                match op {
                    ArithOp::Add => row.add = FE::one(),
                    ArithOp::Sub => row.sub = FE::one(),
                    ArithOp::Xor => row.xor = FE::one(),
                    ArithOp::Or => row.or = FE::one(),
                    ArithOp::And => row.and = FE::one(),
                    ArithOp::ShiftLeftLogical => row.sl = FE::one(),
                    ArithOp::ShiftRightLogical => row.sr = FE::one(),
                    ArithOp::ShiftRightArith => {
                        row.sr = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::SetLessThan => {
                        row.slt = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::SetLessThanU => row.slt = FE::one(),
                }
            }

            Instruction::JumpAndLink { dst, offset } => {
                row.jalr = FE::one();
                row.rd = FE::from(&dst);
                // REVISAR!! Notion dice: PC index = 255.
                row.rs1 = FE::from(&255u32);
                row.imm = u32_to_2_limbs(offset as u32);
                if dst != 0 {
                    row.write_register = FE::one();
                }
            }

            Instruction::JumpAndLinkRegister { base, dst, offset } => {
                row.jalr = FE::one();
                row.rd = FE::from(&dst);
                row.rs1 = FE::from(&base);
                row.imm = u32_to_2_limbs(offset as u32);
                row.arg2 = row.rv1;
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
                row.store = FE::one();
                row.rs1 = FE::from(&base);
                row.rs2 = FE::from(&src);
                row.imm = u32_to_2_limbs(offset as u32);

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
                row.load = FE::one();
                row.rd = FE::from(&dst);
                row.rs1 = FE::from(&base);
                row.imm = u32_to_2_limbs(offset as u32);

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
                row.arg2 = row.rv2;
                match cond {
                    Comparison::Equal => row.beq = FE::one(),
                    Comparison::NotEqual => {
                        row.beq = FE::one();
                        row.mp_selector = FE::one()
                    }
                    Comparison::LessThan => {
                        row.blt = FE::one();
                        row.signed = FE::one();
                    }
                    Comparison::LessThanUnsigned => row.blt = FE::one(),
                    Comparison::GreaterOrEqual => {
                        row.blt = FE::one();
                        row.signed = FE::one();
                        row.mp_selector = FE::one()
                    }
                    Comparison::GreaterOrEqualUnsigned => {
                        row.blt = FE::one();
                        row.mp_selector = FE::one()
                    }
                }
            }

            Instruction::LoadUpperImm { dst, imm } => {
                row.add = FE::one();
                row.rd = FE::from(&dst);
                row.rs1 = FE::zero();
                row.rs2 = FE::zero();
                row.imm = u32_to_2_limbs(imm << 12);
                row.arg2 = u32_to_4_limbs(imm << 12);
                if dst != 0 {
                    row.write_register = FE::one();
                }
            }

            Instruction::AddUpperImmToPc { dst, imm } => {
                row.add = FE::one();
                row.rd = FE::from(&dst);
                row.rs1 = FE::from(&255u32);
                row.rs2 = FE::zero();
                row.imm = u32_to_2_limbs(imm << 12);
                row.arg2 = u32_to_4_limbs(imm << 12);
                row.rv1 = u32_to_4_limbs(log.current_pc);
                if dst != 0 {
                    row.write_register = FE::one();
                }
            }
        }
        row
    }
}

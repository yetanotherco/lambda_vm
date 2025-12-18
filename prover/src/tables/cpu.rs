use crate::utils::{i32_to_4_limbs, u32_to_2_limbs, u32_to_4_limbs};
use vm::vm::{
    instruction::decoding::{ArithOp, Instruction},
    logs::Log,
};

pub struct CpuTable {
    pub rows: Vec<CpuTableRow>,
}

#[derive(Default)]
pub struct CpuTableRow {
    pub timestamp: [u32; 2],
    pub pc: [u32; 2],
    pub rs1: u32,
    pub rs2: u32,
    pub rd: u32,
    pub write_register: u32,
    pub memory_2bytes: u32,
    pub memory_4bytes: u32,
    pub imm: [u32; 2],
    pub signed: u32,
    pub mp_selector: u32,
    pub muldiv_selector: u32,

    pub add: u32,
    pub sub: u32,
    pub slt: u32,
    pub and: u32,
    pub or: u32,
    pub xor: u32,
    pub sl: u32,
    pub sr: u32,
    pub jalr: u32,
    pub beq: u32,
    pub blt: u32,
    pub load: u32,
    pub store: u32,
    pub mul: u32,
    pub divrem: u32,
    pub ecall: u32,
    pub ebreak: u32,

    pub next_pc: [u32; 2],
    pub rv1: [u32; 4],
    pub rv2: [u32; 4],
    pub rvd: [u32; 2],
    pub arg2: [u32; 4],
    pub res: [u32; 4],

    pub is_equal: u32,
    pub branch_cond: u32,
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
                row.rd = dst;
                row.rs1 = src1;
                row.rs2 = src2;
                row.arg2 = row.rv2;
                if dst != 0 {
                    row.write_register = 1u32;
                }
                match op {
                    ArithOp::Add => row.add = 1u32,
                    ArithOp::Sub => row.sub = 1u32,
                    ArithOp::Xor => row.xor = 1u32,
                    ArithOp::Or => row.or = 1u32,
                    ArithOp::And => row.and = 1u32,
                    ArithOp::ShiftLeftLogical => row.sl = 1u32,
                    ArithOp::ShiftRightLogical => row.sr = 1u32,
                    ArithOp::ShiftRightArith => {
                        row.sr = 1u32;
                        row.signed = 1u32;
                    }
                    ArithOp::SetLessThan => {
                        row.slt = 1u32;
                        row.signed = 1u32;
                    }
                    ArithOp::SetLessThanU => row.slt = 1u32,
                }
            }

            Instruction::ArithImm { dst, src, imm, op } => {
                row.rd = dst;
                row.rs1 = src;
                row.arg2 = i32_to_4_limbs(imm);
                if dst != 0 {
                    row.write_register = 1u32;
                }
                match op {
                    ArithOp::Add => row.add = 1u32,
                    ArithOp::Sub => row.sub = 1u32,
                    ArithOp::Xor => row.xor = 1u32,
                    ArithOp::Or => row.or = 1u32,
                    ArithOp::And => row.and = 1u32,
                    ArithOp::ShiftLeftLogical => row.sl = 1u32,
                    ArithOp::ShiftRightLogical => row.sr = 1u32,
                    ArithOp::ShiftRightArith => {
                        row.sr = 1u32;
                        row.signed = 1u32;
                    }
                    ArithOp::SetLessThan => {
                        row.slt = 1u32;
                        row.signed = 1u32;
                    }
                    ArithOp::SetLessThanU => row.slt = 1u32,
                }
            }

            Instruction::JumpAndLink { dst, offset } => {
                todo!()
            }

            Instruction::JumpAndLinkRegister { base, dst, offset } => todo!(),

            Instruction::Store {
                src,
                offset,
                base,
                width,
            } => todo!(),

            Instruction::Load {
                dst,
                offset,
                base,
                width,
            } => todo!(),

            Instruction::Branch {
                src1,
                src2,
                cond,
                offset,
            } => todo!(),

            Instruction::LoadUpperImm { dst, imm } => todo!(),

            Instruction::AddUpperImmToPc { dst, imm } => todo!(),
        }
        row
    }
}

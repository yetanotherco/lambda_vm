use crate::utils::{i32_to_2_limbs, i32_to_4_limbs, u32_to_2_limbs, u32_to_4_limbs};
use executor::vm::{
    instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
    logs::Log,
};
use math::field::{
    element::FieldElement,
    fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField, quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
    },
};
use stark::trace::TraceTable;

type FE = FieldElement<Babybear31PrimeField>;

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
    pub arg2: [FE; 2],
    pub res: [FE; 4],

    pub is_equal: FE,
    pub branch_cond: FE,
}

impl CpuTableRow {
    pub const NUM_COLUMNS: usize = 52;

    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;
    pub const PC_0: usize = 2;
    pub const PC_1: usize = 3;
    pub const RS1: usize = 4;
    pub const RS2: usize = 5;
    pub const RD: usize = 6;
    pub const WRITE_REGISTER: usize = 7;
    pub const MEMORY_2BYTES: usize = 8;
    pub const MEMORY_4BYTES: usize = 9;
    pub const IMM_0: usize = 10;
    pub const IMM_1: usize = 11;
    pub const SIGNED: usize = 12;
    pub const MP_SELECTOR: usize = 13;
    pub const MULDIV_SELECTOR: usize = 14;
    pub const ADD: usize = 15;
    pub const SUB: usize = 16;
    pub const SLT: usize = 17;
    pub const AND: usize = 18;
    pub const OR: usize = 19;
    pub const XOR: usize = 20;
    pub const SL: usize = 21;
    pub const SR: usize = 22;
    pub const JALR: usize = 23;
    pub const BEQ: usize = 24;
    pub const BLT: usize = 25;
    pub const LOAD: usize = 26;
    pub const STORE: usize = 27;
    pub const MUL: usize = 28;
    pub const DIVREM: usize = 29;
    pub const ECALL: usize = 30;
    pub const EBREAK: usize = 31;
    pub const NEXT_PC_0: usize = 32;
    pub const NEXT_PC_1: usize = 33;
    pub const RV1_0: usize = 34;
    pub const RV1_1: usize = 35;
    pub const RV1_2: usize = 36;
    pub const RV1_3: usize = 37;
    pub const RV2_0: usize = 38;
    pub const RV2_1: usize = 39;
    pub const RV2_2: usize = 40;
    pub const RV2_3: usize = 41;
    pub const RVD_0: usize = 42;
    pub const RVD_1: usize = 43;
    pub const ARG2_0: usize = 44;
    pub const ARG2_1: usize = 45;
    pub const RES_0: usize = 46;
    pub const RES_1: usize = 47;
    pub const RES_2: usize = 48;
    pub const RES_3: usize = 49;
    pub const IS_EQUAL: usize = 50;
    pub const BRANCH_COND: usize = 51;

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
                row.arg2 = u32_to_2_limbs(log.src2_val);
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
                    ArithOp::Mul => {
                        row.mul = FE::one();
                        row.mp_selector = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::MulHigh => {
                        row.mul = FE::one();
                        row.muldiv_selector = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::MulHighSignedUnsigned => {
                        row.mul = FE::one();
                        row.muldiv_selector = FE::one();
                        row.mp_selector = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::MulHighUnsigned => {
                        row.mul = FE::one();
                        row.muldiv_selector = FE::one();
                    }
                    ArithOp::Div => {
                        row.divrem = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::DivUnsigned => {
                        row.divrem = FE::one();
                    }
                    ArithOp::Remainder => {
                        row.divrem = FE::one();
                        row.muldiv_selector = FE::one();
                        row.signed = FE::one();
                    }
                    ArithOp::RemainderUnsigned => {
                        row.divrem = FE::one();
                        row.muldiv_selector = FE::one();
                    }
                }
            }

            Instruction::ArithImm { dst, src, imm, op } => {
                row.rd = FE::from(&dst);
                row.rs1 = FE::from(&src);
                row.rs2 = FE::zero();
                row.imm = i32_to_2_limbs(imm);
                row.arg2 = i32_to_2_limbs(imm);
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
                    _ => todo!(),
                }
            }

            Instruction::JumpAndLink { dst, offset } => {
                row.jalr = FE::one();
                row.rd = FE::from(&dst);
                row.imm = i32_to_2_limbs(offset);
                row.arg2 = i32_to_2_limbs(offset);
                if dst != 0 {
                    row.write_register = FE::one();
                }
            }

            Instruction::JumpAndLinkRegister { base, dst, offset } => {
                row.jalr = FE::one();
                row.rd = FE::from(&dst);
                row.rs1 = FE::from(&base);
                row.imm = i32_to_2_limbs(offset);
                row.arg2 = i32_to_2_limbs(offset);
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
                // Fix this afer changing STORE instruction.
                row.imm = i32_to_2_limbs(offset as i32);
                row.arg2 = i32_to_2_limbs(offset as i32);
                row.res = u32_to_4_limbs(log.src1_val + offset);

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
                row.imm = i32_to_2_limbs(offset);
                row.arg2 = i32_to_2_limbs(offset);
                row.res = i32_to_4_limbs(log.src1_val as i32 + offset);

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
                row.arg2 = u32_to_2_limbs(log.src2_val);
                row.res = u32_to_4_limbs(log.src1_val.wrapping_sub(log.src2_val));
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
                row.imm = u32_to_2_limbs(imm);
                row.arg2 = u32_to_2_limbs(imm);
                if dst != 0 {
                    row.write_register = FE::one();
                }
            }

            Instruction::AddUpperImmToPc { dst, imm } => {
                row.add = FE::one();
                row.rd = FE::from(&dst);
                row.imm = u32_to_2_limbs(imm);
                row.arg2 = u32_to_2_limbs(imm);
                row.rv1 = u32_to_4_limbs(log.current_pc);
                if dst != 0 {
                    row.write_register = FE::one();
                }
            }

            _ => {}
        }
        row
    }

    pub fn to_vec(self) -> Vec<FE> {
        let mut row = Vec::with_capacity(54);

        // timestamp[2]
        row.extend_from_slice(&self.timestamp);
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

        row.push(self.add);
        row.push(self.sub);
        row.push(self.slt);
        row.push(self.and);
        row.push(self.or);
        row.push(self.xor);
        row.push(self.sl);
        row.push(self.sr);
        row.push(self.jalr);
        row.push(self.beq);
        row.push(self.blt);
        row.push(self.load);
        row.push(self.store);
        row.push(self.mul);
        row.push(self.divrem);
        row.push(self.ecall);
        row.push(self.ebreak);

        // next_pc[2]
        row.extend_from_slice(&self.next_pc);
        // rv1[4]
        row.extend_from_slice(&self.rv1);
        // rv2[4]
        row.extend_from_slice(&self.rv2);
        // rvd[2]
        row.extend_from_slice(&self.rvd);
        // arg2[2]
        row.extend_from_slice(&self.arg2);
        // res[4]
        row.extend_from_slice(&self.res);

        row.push(self.is_equal);
        row.push(self.branch_cond);

        debug_assert_eq!(row.len(), 52, "CpuTableRow length mismatch");
        row
    }
}

pub fn cpu_trace_from_logs(
    logs: Vec<Log>,
) -> TraceTable<Babybear31PrimeField, Degree4BabyBearU32ExtensionField> {
    const NUM_COLUMNS: usize = 52;
    const MIN_ROWS: usize = 4;

    let num_logs = logs.len();
    let target_rows = num_logs.max(MIN_ROWS).next_power_of_two();

    let mut main_data: Vec<FE> = logs
        .into_iter()
        .enumerate()
        .flat_map(|(i, log)| {
            let timestamp = (i * 4) as u32;
            CpuTableRow::from_log(log, timestamp).to_vec()
        })
        .collect();

    main_data.resize(target_rows * NUM_COLUMNS, FE::zero());

    TraceTable::new_main(main_data, NUM_COLUMNS, 1)
}

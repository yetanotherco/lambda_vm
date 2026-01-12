use executor::vm::instruction::decoding::{ArithOp, Instruction, LoadStoreWidth};
use executor::vm::logs::Log;

pub fn get_add_logs() -> Vec<Log> {
    vec![
        Log {
            instruction: Instruction::ArithImm {
                dst: 12,
                src: 0,
                imm: 10,
                op: ArithOp::Add,
            },
            current_pc: 65652,
            next_pc: 65656,
            src1_val: 0,
            src2_val: 0,
            dst_val: 10,
        },
        Log {
            instruction: Instruction::ArithImm {
                dst: 13,
                src: 0,
                imm: 20,
                op: ArithOp::Add,
            },
            current_pc: 65656,
            next_pc: 65660,
            src1_val: 0,
            src2_val: 0,
            dst_val: 20,
        },
        Log {
            instruction: Instruction::Arith {
                dst: 10,
                src1: 12,
                src2: 13,
                op: ArithOp::Add,
            },
            current_pc: 65660,
            next_pc: 65664,
            src1_val: 10,
            src2_val: 20,
            dst_val: 30,
        },
        Log {
            instruction: Instruction::JumpAndLinkRegister {
                base: 1,
                dst: 0,
                offset: 0,
            },
            current_pc: 65664,
            next_pc: 0,
            src1_val: 0,
            src2_val: 0,
            dst_val: 65668,
        },
    ]
}

pub fn get_rust_logs() -> Vec<Log> {
    vec![
        Log {
            instruction: Instruction::ArithImm {
                dst: 2,
                src: 2,
                imm: -16,
                op: ArithOp::Add,
            },
            current_pc: 70136,
            next_pc: 70140,
            src1_val: 4294967292,
            src2_val: 0,
            dst_val: 4294967276,
        },
        Log {
            instruction: Instruction::Store {
                src: 1,
                offset: 12,
                base: 2,
                width: LoadStoreWidth::Word,
            },
            current_pc: 70140,
            next_pc: 70144,
            src1_val: 4294967276,
            src2_val: 0,
            dst_val: 0,
        },
        Log {
            instruction: Instruction::LoadUpperImm {
                dst: 10,
                imm: 3735928832,
            },
            current_pc: 70144,
            next_pc: 70148,
            src1_val: 0,
            src2_val: 0,
            dst_val: 3735928832,
        },
        Log {
            instruction: Instruction::ArithImm {
                dst: 10,
                src: 10,
                imm: -273,
                op: ArithOp::Add,
            },
            current_pc: 70148,
            next_pc: 70152,
            src1_val: 3735928832,
            src2_val: 0,
            dst_val: 3735928559,
        },
        Log {
            instruction: Instruction::AddUpperImmToPc { dst: 1, imm: 0 },
            current_pc: 70152,
            next_pc: 70156,
            src1_val: 0,
            src2_val: 0,
            dst_val: 70152,
        },
        Log {
            instruction: Instruction::JumpAndLinkRegister {
                base: 1,
                dst: 1,
                offset: -308,
            },
            current_pc: 70156,
            next_pc: 69844,
            src1_val: 70152,
            src2_val: 0,
            dst_val: 70160,
        },
        Log {
            instruction: Instruction::ArithImm {
                dst: 2,
                src: 2,
                imm: -16,
                op: ArithOp::Add,
            },
            current_pc: 69844,
            next_pc: 69848,
            src1_val: 4294967276,
            src2_val: 0,
            dst_val: 4294967260,
        },
        Log {
            instruction: Instruction::Store {
                src: 1,
                offset: 12,
                base: 2,
                width: LoadStoreWidth::Word,
            },
            current_pc: 69848,
            next_pc: 69852,
            src1_val: 4294967260,
            src2_val: 70160,
            dst_val: 0,
        },
        Log {
            instruction: Instruction::AddUpperImmToPc { dst: 1, imm: 0 },
            current_pc: 69852,
            next_pc: 69856,
            src1_val: 0,
            src2_val: 0,
            dst_val: 69852,
        },
        Log {
            instruction: Instruction::JumpAndLinkRegister {
                base: 1,
                dst: 1,
                offset: 72,
            },
            current_pc: 69856,
            next_pc: 69924,
            src1_val: 69852,
            src2_val: 0,
            dst_val: 69860,
        },
        Log {
            instruction: Instruction::ArithImm {
                dst: 2,
                src: 2,
                imm: -16,
                op: ArithOp::Add,
            },
            current_pc: 69924,
            next_pc: 69928,
            src1_val: 4294967260,
            src2_val: 0,
            dst_val: 4294967244,
        },
        Log {
            instruction: Instruction::Store {
                src: 1,
                offset: 12,
                base: 2,
                width: LoadStoreWidth::Word,
            },
            current_pc: 69928,
            next_pc: 69932,
            src1_val: 4294967244,
            src2_val: 69860,
            dst_val: 0,
        },
        Log {
            instruction: Instruction::ArithImm {
                dst: 11,
                src: 10,
                imm: 8,
                op: ArithOp::ShiftRightLogical,
            },
            current_pc: 69932,
            next_pc: 69936,
            src1_val: 3735928559,
            src2_val: 0,
            dst_val: 14593470,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::cpu_air::CPUTableAIR;
    use crate::tables::cpu::cpu_trace_from_logs;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::fields::fft_friendly::babybear_u32::Babybear31PrimeField;
    use math::field::fields::fft_friendly::quartic_babybear_u32::Degree4BabyBearU32ExtensionField;
    use stark::traits::AIR;
    use stark::{
        proof::options::ProofOptions,
        prover::{IsStarkProver, Prover},
        verifier::{IsStarkVerifier, Verifier},
    };

    #[test]
    fn test_cpu_table_from_logs() {
        let logs = get_rust_logs();

        let mut trace = cpu_trace_from_logs(logs);

        let proof_options = ProofOptions::default_test_options();

        let air = CPUTableAIR::new(trace.num_rows(), &(), &proof_options);

        let proof = Prover::<Babybear31PrimeField, Degree4BabyBearU32ExtensionField, _>::prove(
            &air,
            &mut trace,
            &mut DefaultTranscript::<Degree4BabyBearU32ExtensionField>::new(&[]),
        )
        .unwrap();

        assert!(Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<Degree4BabyBearU32ExtensionField>::new(&[]),
        ));
    }
}

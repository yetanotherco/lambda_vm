use executor::{
    elf::Elf,
    vm::{execution::run_program, logs::Log},
};

pub fn run_rust_program(elf_path: &str, expected_output: i32) -> Vec<Log> {
    let elf_data = std::fs::read(elf_path).unwrap();
    let program = Elf::load(&elf_data).unwrap();

    let (results, logs) =
        run_program(program.image, program.entry_point).expect("Failed to run program");

    assert!(results.0 == expected_output);
    logs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::cpu_air::CPUTableAIR;
    use crate::tables::cpu::cpu_trace_from_logs;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::fft_friendly::quartic_babybear_u32::Degree4BabyBearU32ExtensionField;
    use stark_platinum_prover::{
        proof::options::ProofOptions,
        prover::{IsStarkProver, Prover},
        verifier::{IsStarkVerifier, Verifier},
    };
    #[test]
    fn test_basic_rust_program() {
        let logs = run_rust_program("../executor/program_artifacts/rust/basic_rust.elf", 0);

        let mut trace = cpu_trace_from_logs(logs);

        let proof_options = ProofOptions::default_test_options();

        let proof = Prover::<CPUTableAIR>::prove(
            &mut trace,
            &(),
            &proof_options,
            DefaultTranscript::<Degree4BabyBearU32ExtensionField>::new(&[]),
        )
        .unwrap();

        assert!(Verifier::<CPUTableAIR>::verify(
            &proof,
            &(),
            &proof_options,
            DefaultTranscript::<Degree4BabyBearU32ExtensionField>::new(&[]),
        ));
    }
}

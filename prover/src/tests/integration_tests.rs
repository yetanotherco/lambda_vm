use crate::{
    constraints::{cpu_air::CPUTableAIR, decode_air::DecodeTableAIR},
    tables::trace::Trace,
};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::{elf::Elf, vm::execution::run_program};
use math::field::fields::fft_friendly::{
    babybear_u32::Babybear31PrimeField, quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
};
use stark::{
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    traits::AIR,
    verifier::{IsStarkVerifier, Verifier},
};

pub fn run_program_and_prover(elf_path: &str) {
    let elf_data = std::fs::read(elf_path).unwrap();
    let program = Elf::load(&elf_data).unwrap();

    let (_results, logs) =
        run_program(program.image, program.entry_point, false).expect("Failed to run program");

    let mut trace = Trace::generate_trace_from_logs(logs);

    let proof_options = ProofOptions::default_test_options();

    let cpu_air = CPUTableAIR::new(trace.cpu_trace_table.num_rows(), &(), &proof_options);
    let decode_air = DecodeTableAIR::new(trace.decode_trace_table.num_rows(), &(), &proof_options);

    println!("CPU table: {:?}", trace.cpu_trace_table.num_rows());
    println!("Decode table: {:?}", trace.decode_trace_table.num_rows());

    let airs: Vec<(
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearU32ExtensionField,
            PublicInputs = (),
        >,
        &mut _,
    )> = vec![
        (&cpu_air, &mut trace.cpu_trace_table),
        (&decode_air, &mut trace.decode_trace_table),
    ];

    let proofs = Prover::multi_prove(
        airs,
        &mut DefaultTranscript::<Degree4BabyBearU32ExtensionField>::new(&[]),
    )
    .unwrap();

    let airs_and_proofs: Vec<(
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearU32ExtensionField,
            PublicInputs = (),
        >,
        &_,
    )> = vec![(&cpu_air, &proofs[0]), (&decode_air, &proofs[1])];

    assert!(Verifier::multi_verify(
        &airs_and_proofs,
        &mut DefaultTranscript::<Degree4BabyBearU32ExtensionField>::new(&[]),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_rust() {
        run_program_and_prover("../executor/program_artifacts/rust/basic_rust.elf");
    }
    #[test]
    fn test_add() {
        run_program_and_prover("../executor/program_artifacts/rust/add.elf");
    }

    #[test]
    fn test_if() {
        run_program_and_prover("../executor/program_artifacts/rust/if.elf");
    }

    #[test]
    fn test_fibonacci() {
        run_program_and_prover("../executor/program_artifacts/rust/fibonacci.elf");
    }

    #[test]
    fn test_fibonacci_iterative() {
        run_program_and_prover("../executor/program_artifacts/rust/fibonacci_iterative.elf");
    }

    #[test]
    fn test_byte() {
        run_program_and_prover("../executor/program_artifacts/rust/byte.elf");
    }

    #[test]
    fn test_byte_signed() {
        run_program_and_prover("../executor/program_artifacts/rust/byte_signed.elf");
    }

    #[test]
    fn test_half() {
        run_program_and_prover("../executor/program_artifacts/rust/half.elf");
    }

    #[test]
    fn test_half_signed() {
        run_program_and_prover("../executor/program_artifacts/rust/half_signed.elf");
    }

    #[test]
    fn test_rlp() {
        run_program_and_prover("../executor/program_artifacts/rust/rlp.elf");
    }
}

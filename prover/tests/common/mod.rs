use lambda_vm_prover::security_posture::{TEST_CHAIN_SECURITY_BITS, TEST_GRIND_BITS};
use lambda_vm_prover::{Error, GoldilocksCubicProofOptions, MaxRowsConfig, VmProof};
use stark::proof::options::ProofOptions;

pub fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_params(2, TEST_CHAIN_SECURITY_BITS, TEST_GRIND_BITS)
        .expect("test proof options")
}

pub fn prove(elf: &[u8], input: &[u8]) -> Result<VmProof, Error> {
    lambda_vm_prover::prove_with_options_and_inputs(
        elf,
        input,
        &options(),
        &MaxRowsConfig::default(),
    )
}

pub fn verify(proof: &VmProof, elf: &[u8]) -> Result<bool, Error> {
    lambda_vm_prover::verify_with_options(proof, elf, &options(), None, None)
}

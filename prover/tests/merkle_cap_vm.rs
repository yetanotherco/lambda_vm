//! A real VM proof under the Merkle cap policy the PROCESS FORMAT names
//! (`LAMBDA_VM_ZF_CAP`, design/CAP.md §4): every production table — the
//! preprocessed ones (precomputed + main trees), the LogUp aux trees, the
//! composition trees and every committed FRI layer — capped, proved and
//! verified through the public `prove_with_options_and_inputs` /
//! `verify_with_options` entry points.
//!
//! Knob-on only, hence `#[ignore]`: at the default format it would prove
//! nothing new, so it refuses to run unless `LAMBDA_VM_ZF_CAP` selects a cap.
//!
//! ```text
//! LAMBDA_VM_ZF_CAP=auto cargo test --release -p lambda-vm-prover --test merkle_cap_vm -- --ignored --nocapture
//! LAMBDA_VM_ZF_CAP=auto cargo test --release -p lambda-vm-prover --features cuda --test merkle_cap_vm -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Under `cuda` the fixture is big enough that its tables commit on the device,
//! and the caps must come off the resident trees (`gpu_cap_read_calls`).

use lambda_vm_prover::test_utils::asm_elf_bytes;
use lambda_vm_prover::zf_format::ZfFormat;
use lambda_vm_prover::{
    GoldilocksCubicProofOptions, MaxRowsConfig, prove_with_options_and_inputs, verify_with_options,
};

/// CPU: a fixture that touches every instruction class (many tables). Device:
/// the fixture the cuda integration tests use, whose tables cross the GPU LDE
/// threshold.
#[cfg(not(feature = "cuda"))]
const FIXTURE: &str = "all_instructions_64";
#[cfg(feature = "cuda")]
const FIXTURE: &str = "fib_iterative_1M";

#[test]
#[ignore = "knob-on: run with LAMBDA_VM_ZF_CAP=auto (or a height) and -- --ignored"]
fn a_vm_proof_round_trips_under_the_process_cap_policy() {
    let format = ZfFormat::from_env().expect("a valid ZF format");
    assert!(
        !format.cap.is_off(),
        "LAMBDA_VM_ZF_CAP is unset or off: this test only means something with a cap"
    );
    println!("{}", format.banner());
    let base = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup 2");
    let capped = format.options(base.clone());
    let mut default = base;
    default.format = Default::default();
    assert!(default.has_default_format());

    let elf = asm_elf_bytes(FIXTURE);
    #[cfg(feature = "cuda")]
    let before = stark::gpu_lde::gpu_cap_read_calls();
    let proof = prove_with_options_and_inputs(&elf, &[], &capped, &MaxRowsConfig::default())
        .expect("prove under the cap policy");
    #[cfg(feature = "cuda")]
    {
        let reads = stark::gpu_lde::gpu_cap_read_calls() - before;
        println!("CAPVM device cap reads: {reads}");
        assert!(
            reads > 0,
            "no cap came off the device: {FIXTURE} proved on host trees"
        );
    }

    assert!(
        verify_with_options(&proof, &elf, &capped, None, None).expect("verify"),
        "a capped VM proof must verify under its own policy"
    );
    // The cap height is a verifier constant: the default verifier must refuse
    // the capped proof (full-length paths expected), without panicking.
    assert!(
        !matches!(
            verify_with_options(&proof, &elf, &default, None, None),
            Ok(true)
        ),
        "a capped proof accepted by the default-format verifier"
    );
}

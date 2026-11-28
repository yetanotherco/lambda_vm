//! A real VM proof under the Merkle cap policy the PROCESS FORMAT names
//! (`LAMBDA_VM_ZF_CAP`): every production table — the
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
use stark::merkle_caps::StarkCaps;

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
    // Non-vacuity: the policy engaged. Every table's main-tree paths have the
    // lengths the verifier's `StarkCaps` gives its shape, and at least one
    // table is really capped — so the default verifier below meets paths that
    // differ from the ones it expects, and its refusal means something.
    let mut capped_tables = 0usize;
    for (t, p) in proof.proof.proofs.iter().enumerate() {
        let lde_log = (p.trace_length * usize::from(capped.blowup_factor)).trailing_zeros();
        let caps = StarkCaps::new(
            format.cap,
            capped.fri_number_of_queries,
            lde_log as usize,
            p.fri_layers_merkle_roots.len(),
        );
        let path = |q: usize| {
            p.deep_poly_openings[q]
                .main_trace_polys
                .proof
                .merkle_path
                .len()
        };
        assert_eq!(
            path(1),
            caps.trace_depth - caps.trace,
            "table {t}: a non-owner main path is not cut to the cap"
        );
        if caps.trace > 0 {
            assert_eq!(
                path(0),
                caps.trace_depth - caps.trace + (1 << caps.trace),
                "table {t}: the owner path does not carry the cap"
            );
            capped_tables += 1;
        }
    }
    println!(
        "CAPVM capped tables: {capped_tables} of {}",
        proof.proof.proofs.len()
    );
    assert!(
        capped_tables > 0,
        "no table was capped: the policy did not engage and the cross-format check below is vacuous"
    );
    // The cap height is a verifier constant: the default verifier must refuse
    // the capped proof (full-length paths expected), without panicking. This
    // held only once the AIR prototype cache keyed the proof format: before,
    // the prove above cached capped AIRs and `&default` got them back.
    assert!(
        !matches!(
            verify_with_options(&proof, &elf, &default, None, None),
            Ok(true)
        ),
        "a capped proof accepted by the default-format verifier"
    );
}

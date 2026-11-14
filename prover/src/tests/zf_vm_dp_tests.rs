//! S3 end to end on the production VM path: a real multi-table VM proof
//! (every table the program touches, the preprocessed ones included, under the
//! RPX block pin, host CPU FRI) proved and verified at `fri = dp`.
//!
//! Proves a full VM trace (the 2^20-row BITWISE table among them), so it runs
//! in the box lib suite, not on the laptop — like its default-format sibling
//! `skip_empty_tables_tests::dropping_a_used_table_through_the_real_verifier_is_rejected`.

use stark::proof::options::{FriMode, ProofFormat, ProofOptions};

#[test]
fn a_vm_proof_round_trips_at_fri_dp() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_mul_8");
    let default = ProofOptions::default_test_options();
    let dp = ProofOptions {
        format: ProofFormat {
            fri_mode: FriMode::Dp,
            ..ProofFormat::DEFAULT
        },
        ..default.clone()
    };
    let vm_proof = crate::prove_with_options(&elf_bytes, &dp, &Default::default())
        .expect("the fixture must prove at fri = dp");
    assert!(
        crate::verify_with_options(&vm_proof, &elf_bytes, &dp, None, None)
            .expect("honest verify must not error"),
        "an honest dp VM proof must verify"
    );
    // Non-vacuity: some table folds a committed layer by more than 2, i.e.
    // carries more opened values per query than committed layers.
    assert!(
        vm_proof.proof.proofs.iter().any(|p| {
            let layers = p.fri_layers_merkle_roots.len();
            layers > 0 && p.query_list[0].layers_evaluations_sym.len() > 2 * layers
        }),
        "no table used a group of more than two values"
    );
    // The format is a verifier constant: the default verifier rejects it.
    assert!(
        !crate::verify_with_options(&vm_proof, &elf_bytes, &default, None, None).unwrap_or(false),
        "a dp proof must not verify under the default format"
    );
    // A tampered FRI group value is rejected.
    let mut bad = vm_proof.clone();
    let table = bad
        .proof
        .proofs
        .iter()
        .position(|p| !p.fri_layers_merkle_roots.is_empty())
        .expect("a table with committed layers");
    bad.proof.proofs[table].query_list[0].layers_evaluations_sym[0] +=
        math::field::element::FieldElement::<
            math::field::extensions_goldilocks::Degree3GoldilocksExtensionField,
        >::one();
    assert!(
        !crate::verify_with_options(&bad, &elf_bytes, &dp, None, None).unwrap_or(false),
        "a tampered group value must be rejected"
    );
}

/// The same VM proof on the device path (a cuda build, the default device
/// thresholds): every table whose LDE the device admits commits its FRI
/// layers with the device group loop, and the proof still verifies. The device
/// FRI counter must move — under `dp` every device FRI commit is a group
/// commit, so a host-only run fails here. Run with `--test-threads=1`: the
/// counter is process-wide.
#[cfg(feature = "cuda")]
#[test]
fn a_vm_proof_round_trips_at_fri_dp_on_the_device() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_mul_8");
    let dp = ProofOptions {
        format: ProofFormat {
            fri_mode: FriMode::Dp,
            ..ProofFormat::DEFAULT
        },
        ..ProofOptions::default_test_options()
    };
    let before = stark::gpu_lde::gpu_fri_calls();
    let vm_proof = crate::prove_with_options(&elf_bytes, &dp, &Default::default())
        .expect("the fixture must prove at fri = dp on the device path");
    let device_commits = stark::gpu_lde::gpu_fri_calls() - before;
    println!("FRIDEV VM dp proof: {device_commits} device FRI commits");
    assert!(
        device_commits > 0,
        "no table took the device FRI commit at fri = dp"
    );
    assert!(
        crate::verify_with_options(&vm_proof, &elf_bytes, &dp, None, None)
            .expect("honest verify must not error"),
        "a device-proved dp VM proof must verify"
    );
    assert!(
        vm_proof.proof.proofs.iter().any(|p| {
            let layers = p.fri_layers_merkle_roots.len();
            layers > 0 && p.query_list[0].layers_evaluations_sym.len() > 2 * layers
        }),
        "no table used a group of more than two values"
    );
}

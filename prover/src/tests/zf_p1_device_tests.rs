//! The Poseidon1 STARK configuration (`P1StarkHash`: 4-ary trees, caps at
//! 4-ary height `P1_STARK_CAP`) on the device: the P1 twins of
//! `zf_rpx_device_tests`, and the P1 goldens proved on the card.
//!
//! Every test needs a GPU (`#[ignore]`d); the FRI parity cases also a lowered
//! `LAMBDA_VM_GPU_LDE_THRESHOLD`, and the goldens one at or below 4096 so
//! every case commits on the card:
//!
//! ```text
//! LAMBDA_VM_GPU_LDE_THRESHOLD=2 cargo test --release -p lambda-vm-prover --features cuda \
//!     --lib tests::zf_p1_device_tests::parity_ -- --ignored
//! cargo test --release -p lambda-vm-prover --features cuda \
//!     --lib tests::zf_p1_device_tests::trees_one_row_p1 -- --ignored --exact
//! LAMBDA_VM_GPU_LDE_THRESHOLD=1024 cargo test --release -p lambda-vm-prover --features cuda \
//!     --lib tests::zf_p1_device_tests::proved_p1_goldens_equal_the_host_bytes \
//!     -- --ignored --exact --test-threads=1
//! ```

use stark::fri::device_parity::{
    Case, legacy_cases, production_cases, resident_cases, run_cases, sweep_cases,
};

use crate::lfm::p1_commit::P1StarkHash;

fn check(cases: &[Case], resident: bool, seed: u64) {
    if let Err(failures) = run_cases::<P1StarkHash>("p1", cases, resident, seed) {
        panic!("p1: {failures:#?}");
    }
}

/// Every distinct DP schedule (the stark crate's pinned 29 shapes): 4-ary
/// group trees, their device paths against the host's.
#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_every_dp_shape_p1() {
    let cases = sweep_cases();
    assert_eq!(cases.len(), 29);
    check(&cases, false, 0x5031_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_production_sizes_p1() {
    check(&production_cases(), false, 0x5031_1000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_resident_layers_p1() {
    check(&resident_cases(), true, 0x5031_2000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_legacy_encoding_p1() {
    check(&legacy_cases(), false, 0x5031_3000);
}

/// Main, preprocessed-split, aux and composition trees committed on the card
/// at both leaf layouts, and their device-gathered paths, against the host.
#[test]
#[ignore = "requires a GPU; run with --features cuda -- --ignored"]
fn trees_one_row_p1() {
    if let Err(failures) = stark::s2_device_parity::run_tree_parity::<P1StarkHash>("p1") {
        panic!("p1: {failures:#?}");
    }
}

/// The P1 goldens (`p1_stark_tests::P1_GOLDENS`, the host's bytes) proved on
/// the card: the 4-ary trace, aux, composition and FRI trees committed there,
/// every opening gathered off the resident trees and every cap read off them,
/// and the proof bytes equal to the host's. Each case must take the device
/// FRI commit, a device trace tree and a device composition tree, and gather
/// its openings and caps on the device; a host fallback fails the test. Run
/// alone: the counters are process-wide.
#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD<=4096; run alone with --features cuda -- --ignored --exact --test-threads=1"]
fn proved_p1_goldens_equal_the_host_bytes() {
    use super::p1_stark_tests::{P1_GOLDENS, p1_golden_cases, p1_goldens};
    use stark::gpu_lde::{
        gpu_cap_read_calls, gpu_comp_poly_tree_calls, gpu_fri_calls, gpu_merkle_tree_calls,
        gpu_opening_gather_calls,
    };
    let counts = || {
        [
            gpu_fri_calls(),
            gpu_merkle_tree_calls(),
            gpu_comp_poly_tree_calls(),
            gpu_opening_gather_calls(),
            gpu_cap_read_calls(),
        ]
    };
    let cases = p1_golden_cases().len();
    let before = counts();
    let got = p1_goldens();
    let after = counts();
    let moved: Vec<u64> = after.iter().zip(before).map(|(a, b)| a - b).collect();
    println!(
        "P1DEV goldens: {} cases; device FRI commits {}, trace trees {}, composition trees {}, \
         opening gathers {}, cap reads {}",
        got.len(),
        moved[0],
        moved[1],
        moved[2],
        moved[3],
        moved[4]
    );
    for (what, &n) in [
        "FRI commits",
        "trace trees",
        "composition trees",
        "opening gathers",
        "cap reads",
    ]
    .iter()
    .zip(&moved)
    {
        assert!(
            n >= cases as u64,
            "every P1 golden case must take the device {what} ({n} for {cases} cases; lower \
             LAMBDA_VM_GPU_LDE_THRESHOLD)"
        );
    }
    assert_eq!(got.len(), P1_GOLDENS.len(), "one pin per case");
    for ((name, digest), (pin_name, pin)) in got.iter().zip(P1_GOLDENS) {
        assert_eq!(name, pin_name);
        assert_eq!(
            digest, pin,
            "{name}: the card's P1 proof differs from the host's bytes"
        );
    }
}

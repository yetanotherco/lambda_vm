//! S3 on the device under the production RPX pin (lane I-FRI-D, D1): the RPX
//! twins of the stark crate's `tests::zf_fri_device_tests` (which cover Keccak
//! and Blake3; the stark crate cannot name `RpxStarkHash`).
//!
//! Every `#[ignore]`d test needs a GPU and a lowered
//! `LAMBDA_VM_GPU_LDE_THRESHOLD`, and fails when the device path does not run:
//!
//! ```text
//! LAMBDA_VM_GPU_LDE_THRESHOLD=2 cargo test --release -p lambda-vm-prover --features cuda \
//!     --lib tests::zf_rpx_device_tests::parity_ -- --ignored
//! LAMBDA_VM_GPU_LDE_THRESHOLD=1024 cargo test --release -p lambda-vm-prover --features cuda \
//!     --lib tests::zf_rpx_device_tests::proved_rpx_vectors_equal_the_cpu_bytes \
//!     -- --ignored --exact --test-threads=1
//! ```
//!
//! S2 on the device (lane I-S2-D, D2): `trees_one_row_rpx` (default threshold),
//! `fri_one_row_*` (threshold 2), `proved_rpx_one_row_vectors_equal_the_cpu_bytes`
//! (threshold 1024, alone), the RPX twins of `stark`'s `tests::zf_s2_device_tests`.

use stark::fri::device_parity::{
    Case, legacy_cases, production_cases, resident_cases, run_cases, sweep_cases,
};

use crate::lfm::algebraic_commit::RpxStarkHash;

fn check(cases: &[Case], resident: bool, seed: u64) {
    if let Err(failures) = run_cases::<RpxStarkHash>("rpx", cases, resident, seed) {
        panic!("rpx: {failures:#?}");
    }
}

/// Every distinct DP schedule for B ≤ 23 plus the extra shapes (the list is
/// pinned by the stark crate's `dp_shapes_are_pinned`: 29 shapes).
#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_every_dp_shape_rpx() {
    let cases = sweep_cases();
    assert_eq!(cases.len(), 29);
    check(&cases, false, 0x5a46_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_production_sizes_rpx() {
    check(&production_cases(), false, 0x5a47_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_resident_layers_rpx() {
    check(&resident_cases(), true, 0x5a48_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_legacy_encoding_rpx() {
    check(&legacy_cases(), false, 0x5a49_0000);
}

/// The RPX (d) vector proofs (`pair`, `dp`, `dp_3_1_3`, `cap_pair`, `cap_dp`;
/// LDE 4096) proved on
/// the device path are byte-identical to the checked-in CPU-proved files. The
/// device FRI counter must move once per proof. Run alone: the counter is
/// process-wide.
#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD<=4096; run alone with --features cuda -- --ignored --exact --test-threads=1"]
fn proved_rpx_vectors_equal_the_cpu_bytes() {
    use stark::fri::vectors::{check_or_write, proof_vectors};
    let before = stark::gpu_lde::gpu_fri_calls();
    let comp_before = stark::gpu_lde::gpu_composition_calls();
    let files = proof_vectors::<RpxStarkHash>("rpx");
    let device_commits = stark::gpu_lde::gpu_fri_calls() - before;
    let compositions = stark::gpu_lde::gpu_composition_calls() - comp_before;
    println!(
        "FRIDEV rpx vector proofs: {} files, {device_commits} device FRI commits, {compositions} device compositions",
        files.len()
    );
    // Five (d) formats (pair, dp, dp_3_1_3 at Q = 3; cap_pair, cap_dp at
    // Q = 20), two files and one FRI commit per proof.
    assert_eq!(files.len(), 5 * 2);
    assert_eq!(
        device_commits, 5,
        "every vector proof must take the device FRI commit (lower LAMBDA_VM_GPU_LDE_THRESHOLD)"
    );
    // Every proof composes on the device (the AIR's constraint program,
    // I-FIX-D2); a host composition would not be counted here.
    assert_eq!(
        compositions, 5,
        "every RPX vector proof must compose on the device ({compositions} device compositions)"
    );
    let bad = check_or_write(&files, false);
    assert!(
        bad.is_empty(),
        "device-proved RPX vectors differ from the checked-in CPU bytes: {bad:?}"
    );
}

// ---------------------------------------------------------------------------
// S2 on the device (FRI.md §7.6, lane I-S2-D, D2) under the RPX pin.
// ---------------------------------------------------------------------------

/// One-row main / preprocessed split / aux (host and resident) / composition
/// trees and their device openings against the host (the RPX twin of
/// `stark`'s `trees_one_row_*`).
#[test]
#[ignore = "requires a GPU; run with --features cuda -- --ignored"]
fn trees_one_row_rpx() {
    if let Err(failures) = stark::s2_device_parity::run_tree_parity::<RpxStarkHash>("rpx") {
        panic!("rpx: {failures:#?}");
    }
}

/// The one-row FRI commit (input tree from the codeword, then the group chain)
/// and query phases against the host CPU loop.
#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn fri_one_row_rpx() {
    check(
        &stark::fri::device_parity::one_row_cases(),
        false,
        0x5234_0000,
    );
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn fri_one_row_resident_rpx() {
    check(
        &stark::fri::device_parity::one_row_resident_cases(),
        true,
        0x5235_0000,
    );
}

/// The RPX (e) vector proofs (`one_row_pair`, `one_row_3_2_1_2`; LDE 4096)
/// proved on the device path are byte-identical to the checked-in CPU-proved
/// files. Each proof must take the one-row device FRI commit and build at
/// least its main, aux and composition trees one-row on the device. Run alone:
/// the counters are process-wide.
#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD<=4096; run alone with --features cuda -- --ignored --exact --test-threads=1"]
fn proved_rpx_one_row_vectors_equal_the_cpu_bytes() {
    use stark::fri::vectors::{check_or_write, one_row_proof_vectors};
    let fri_before = stark::gpu_lde::gpu_one_row_fri_calls();
    let trees_before = stark::gpu_lde::gpu_one_row_trees();
    let comp_before = stark::gpu_lde::gpu_composition_calls();
    let files = one_row_proof_vectors::<RpxStarkHash>("rpx");
    let fri_commits = stark::gpu_lde::gpu_one_row_fri_calls() - fri_before;
    let trees = stark::gpu_lde::gpu_one_row_trees() - trees_before;
    let compositions = stark::gpu_lde::gpu_composition_calls() - comp_before;
    println!(
        "S2DEV rpx vector proofs: {} files, {fri_commits} one-row device FRI commits, {trees} one-row device trees, {compositions} device compositions",
        files.len()
    );
    assert_eq!(files.len(), 2 * 2);
    assert_eq!(
        fri_commits, 2,
        "every one-row vector proof must take the device FRI commit (lower LAMBDA_VM_GPU_LDE_THRESHOLD)"
    );
    assert!(
        trees >= 3 * 2,
        "every one-row vector proof must build its main, aux and composition trees on the device \
         ({trees} one-row device trees for 2 proofs)"
    );
    // Every proof composes on the device (the AIR's constraint program,
    // I-FIX-D2); a host composition would not be counted here.
    assert_eq!(
        compositions, 2,
        "every one-row RPX vector proof must compose on the device ({compositions} device compositions)"
    );
    let bad = check_or_write(&files, false);
    assert!(
        bad.is_empty(),
        "device-proved one-row RPX vectors differ from the checked-in CPU bytes: {bad:?}"
    );
}

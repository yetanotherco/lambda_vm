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

/// The RPX (d) vector proofs (`pair`, `dp`, `dp_3_1_3`, LDE 4096) proved on
/// the device path are byte-identical to the checked-in CPU-proved files. The
/// device FRI counter must move once per proof. Run alone: the counter is
/// process-wide.
#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD<=4096; run alone with --features cuda -- --ignored --exact --test-threads=1"]
fn proved_rpx_vectors_equal_the_cpu_bytes() {
    use stark::fri::vectors::{check_or_write, proof_vectors};
    let before = stark::gpu_lde::gpu_fri_calls();
    let files = proof_vectors::<RpxStarkHash>("rpx");
    let device_commits = stark::gpu_lde::gpu_fri_calls() - before;
    println!(
        "FRIDEV rpx vector proofs: {} files, {device_commits} device FRI commits",
        files.len()
    );
    assert_eq!(files.len(), 3 * 2);
    assert_eq!(
        device_commits, 3,
        "every vector proof must take the device FRI commit (lower LAMBDA_VM_GPU_LDE_THRESHOLD)"
    );
    let bad = check_or_write(&files, false);
    assert!(
        bad.is_empty(),
        "device-proved RPX vectors differ from the checked-in CPU bytes: {bad:?}"
    );
}

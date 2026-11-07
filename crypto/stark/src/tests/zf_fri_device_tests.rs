//! S3 on the device: the device FRI commit and
//! query phases against the host CPU loop, under Keccak and Blake3 (the RPX
//! twins live in the prover crate's `tests::zf_rpx_device_tests`).
//!
//! Every `#[ignore]`d test here needs a GPU and a lowered
//! `LAMBDA_VM_GPU_LDE_THRESHOLD`; each one fails loudly when the device path
//! does not run (a declined commit is an `Err`, a vector proof must move the
//! device FRI counter), so none can pass by falling back to the host:
//!
//! ```text
//! LAMBDA_VM_GPU_LDE_THRESHOLD=2 cargo test -p stark --release --features cuda \
//!     --lib tests::zf_fri_device_tests::parity_ -- --ignored
//! LAMBDA_VM_GPU_LDE_THRESHOLD=1024 cargo test -p stark --release --features cuda \
//!     --lib tests::zf_fri_device_tests::proved_vectors_equal_the_cpu_bytes \
//!     -- --ignored --exact --test-threads=1
//! ```
//!
//! `dp_shapes_are_pinned` needs no GPU (it only computes the shape list).

use crate::config::{Blake3StarkHash, KeccakStarkHash, StarkHash};
use crate::fri::device_parity::{
    Case, dp_shapes, legacy_cases, production_cases, resident_cases, run_cases, sweep_cases,
};

/// The shapes the parity sweep covers: every distinct DP schedule for
/// `B ≤ 23` (T ∈ {4, 9, 10}, Q ∈ {3, 110}, cap off/auto) plus the extras.
/// Pinned so the box run's pre-registered count means something; a DP change
/// that moves this list is a format change and re-pins it deliberately.
#[test]
fn dp_shapes_are_pinned() {
    let shapes = dp_shapes();
    let expected: &[&[u8]] = PINNED_SHAPES;
    assert_eq!(
        shapes,
        expected.iter().map(|s| s.to_vec()).collect::<Vec<_>>(),
        "the DP's schedule set moved"
    );
}

const PINNED_SHAPES: &[&[u8]] = &[
    // The DP's own (21, in first-appearance order).
    &[1],
    &[2],
    &[3],
    &[2, 2],
    &[3, 2],
    &[3, 3],
    &[3, 2, 2],
    &[3, 3, 2],
    &[3, 3, 3],
    &[3, 3, 2, 2],
    &[3, 3, 3, 2],
    &[3, 3, 3, 3],
    &[3, 3, 3, 2, 2],
    &[3, 3, 3, 3, 2],
    &[3, 3, 3, 3, 3],
    &[3, 3, 3, 3, 2, 2],
    &[3, 3, 3, 3, 3, 2],
    &[3, 3, 3, 3, 3, 3],
    &[4, 3, 3],
    &[4, 3, 3, 3],
    &[4, 3],
    // EXTRA_SHAPES (8).
    &[4],
    &[6],
    &[1, 6],
    &[6, 1],
    &[3, 1, 3],
    &[1, 3],
    &[2, 5, 1],
    &[1, 1, 1],
];

fn check<H: StarkHash>(name: &str, cases: &[Case], resident: bool, seed: u64) {
    if let Err(failures) = run_cases::<H>(name, cases, resident, seed) {
        panic!("{name}: {failures:#?}");
    }
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_every_dp_shape_keccak() {
    check::<KeccakStarkHash>("keccak", &sweep_cases(), false, 0x5a46_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_every_dp_shape_blake3() {
    check::<Blake3StarkHash>("blake3", &sweep_cases(), false, 0x5a46_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_production_sizes_keccak() {
    check::<KeccakStarkHash>("keccak", &production_cases(), false, 0x5a47_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_production_sizes_blake3() {
    check::<Blake3StarkHash>("blake3", &production_cases(), false, 0x5a47_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_resident_layers_keccak() {
    check::<KeccakStarkHash>("keccak", &resident_cases(), true, 0x5a48_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_resident_layers_blake3() {
    check::<Blake3StarkHash>("blake3", &resident_cases(), true, 0x5a48_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_legacy_encoding_keccak() {
    check::<KeccakStarkHash>("keccak", &legacy_cases(), false, 0x5a49_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn parity_legacy_encoding_blake3() {
    check::<Blake3StarkHash>("blake3", &legacy_cases(), false, 0x5a49_0000);
}

/// The (d) vector proofs (the README's (d): `pair`, `dp`, `dp_3_1_3`, and the
/// Merkle-capped `cap_pair`, `cap_dp` at Q = 20) proved on
/// the device path — LDE 4096, so `LAMBDA_VM_GPU_LDE_THRESHOLD` must be at
/// most 4096 — are byte-identical to the checked-in CPU-proved files (rkyv
/// bytes and the verifier-derived JSON), under Keccak and Blake3. The device
/// FRI counter must move once per proof, so a host fallback fails the test.
/// Run alone (`--exact --test-threads=1`): the counter is process-wide.
#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD<=4096; run alone with --features cuda -- --ignored --exact --test-threads=1"]
fn proved_vectors_equal_the_cpu_bytes() {
    use crate::fri::vectors::{check_or_write, proof_vectors};
    let before = crate::gpu_lde::gpu_fri_calls();
    let comp_before = crate::gpu_lde::gpu_composition_calls();
    let mut files = proof_vectors::<KeccakStarkHash>("keccak");
    files.extend(proof_vectors::<Blake3StarkHash>("blake3"));
    let device_commits = crate::gpu_lde::gpu_fri_calls() - before;
    let compositions = crate::gpu_lde::gpu_composition_calls() - comp_before;
    println!(
        "FRIDEV vector proofs: {} files, {device_commits} device FRI commits, {compositions} device compositions",
        files.len()
    );
    // Five (d) formats (pair, dp, dp_3_1_3 at Q = 3; cap_pair, cap_dp at
    // Q = 20) x two hashes, two files and one FRI commit per proof.
    assert_eq!(files.len(), 2 * 5 * 2);
    assert_eq!(
        device_commits, 10,
        "every vector proof must take the device FRI commit (lower LAMBDA_VM_GPU_LDE_THRESHOLD)"
    );
    // Every proof composes on the device (the AIR's constraint program); a
    // host composition would not be counted here.
    assert_eq!(
        compositions, 10,
        "every vector proof must compose on the device ({compositions} device compositions)"
    );
    let bad = check_or_write(&files, false);
    assert!(
        bad.is_empty(),
        "device-proved vectors differ from the checked-in CPU bytes: {bad:?}"
    );
}

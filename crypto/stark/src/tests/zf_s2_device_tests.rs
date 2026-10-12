//! S2 on the device: one-row trees and
//! openings, and the committed input tree from the DEEP codeword, against the
//! host CPU paths, under Keccak and Blake3 (the RPX twins live in the prover
//! crate's `tests::zf_rpx_device_tests`).
//!
//! Every test here needs a GPU and fails loudly when the device path does not
//! run (a declined commit is an `Err`, a vector proof must move the one-row
//! device counters), so none can pass by falling back to the host:
//!
//! ```text
//! cargo test -p stark --release --features cuda --lib \
//!     tests::zf_s2_device_tests::trees_ -- --ignored
//! LAMBDA_VM_GPU_LDE_THRESHOLD=2 cargo test -p stark --release --features cuda --lib \
//!     tests::zf_s2_device_tests::fri_ -- --ignored
//! LAMBDA_VM_GPU_LDE_THRESHOLD=1024 cargo test -p stark --release --features cuda --lib \
//!     tests::zf_s2_device_tests::proved_one_row_vectors_equal_the_cpu_bytes \
//!     -- --ignored --exact --test-threads=1
//! ```

use crate::config::{Blake3StarkHash, KeccakStarkHash, StarkHash};
use crate::fri::device_parity::{Case, one_row_cases, one_row_resident_cases, run_cases};
use crate::s2_device_parity::run_tree_parity;

fn trees<H: StarkHash>(name: &str) {
    if let Err(failures) = run_tree_parity::<H>(name) {
        panic!("{name}: {failures:#?}");
    }
}

fn fri<H: StarkHash>(name: &str, cases: &[Case], resident: bool, seed: u64) {
    if let Err(failures) = run_cases::<H>(name, cases, resident, seed) {
        panic!("{name}: {failures:#?}");
    }
}

/// The one-row case list is pinned by count, so the box run's pre-registered
/// `FRIDEV … cases equal` lines mean something (29 DP shapes + 6 pair-mode + 5
/// production; 5 resident).
#[test]
fn one_row_case_lists_are_pinned() {
    assert_eq!(one_row_cases().len(), 29 + 6 + 5);
    assert_eq!(one_row_resident_cases().len(), 5);
    for (lde_log, o) in one_row_cases().iter().chain(&one_row_resident_cases()) {
        assert_eq!(
            o.format.one_row,
            crate::proof::options::OneRowMode::On,
            "LDE 2^{lde_log}: a one-row case without one-row openings"
        );
    }
}

#[test]
#[ignore = "requires a GPU; run with --features cuda -- --ignored"]
fn trees_one_row_keccak() {
    trees::<KeccakStarkHash>("keccak");
}

#[test]
#[ignore = "requires a GPU; run with --features cuda -- --ignored"]
fn trees_one_row_blake3() {
    trees::<Blake3StarkHash>("blake3");
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn fri_one_row_keccak() {
    fri::<KeccakStarkHash>("keccak", &one_row_cases(), false, 0x5234_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn fri_one_row_blake3() {
    fri::<Blake3StarkHash>("blake3", &one_row_cases(), false, 0x5234_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn fri_one_row_resident_keccak() {
    fri::<KeccakStarkHash>("keccak", &one_row_resident_cases(), true, 0x5235_0000);
}

#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD=2; run with --features cuda -- --ignored"]
fn fri_one_row_resident_blake3() {
    fri::<Blake3StarkHash>("blake3", &one_row_resident_cases(), true, 0x5235_0000);
}

/// The (e) vector proofs (the README's (e): `one_row_pair` and
/// `one_row_3_2_1_2`, LDE 4096, Q = 3, grinding 0) proved on the device path
/// are byte-identical to the checked-in CPU-proved files (rkyv bytes and the
/// verifier-derived JSON), under Keccak and Blake3. Each proof must take the
/// one-row device FRI commit once (the input tree off the DEEP codeword) and
/// build its main, aux and composition trees one-row on the device (at least
/// three one-row device trees per proof), so a host fallback fails the test.
/// Run alone (`--exact --test-threads=1`): the counters are process-wide.
#[test]
#[ignore = "requires a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD<=4096; run alone with --features cuda -- --ignored --exact --test-threads=1"]
fn proved_one_row_vectors_equal_the_cpu_bytes() {
    use crate::fri::vectors::{check_or_write, one_row_proof_vectors};
    let fri_before = crate::gpu_lde::gpu_one_row_fri_calls();
    let trees_before = crate::gpu_lde::gpu_one_row_trees();
    let comp_before = crate::gpu_lde::gpu_composition_calls();
    let mut files = one_row_proof_vectors::<KeccakStarkHash>("keccak");
    files.extend(one_row_proof_vectors::<Blake3StarkHash>("blake3"));
    let fri_commits = crate::gpu_lde::gpu_one_row_fri_calls() - fri_before;
    let trees = crate::gpu_lde::gpu_one_row_trees() - trees_before;
    let compositions = crate::gpu_lde::gpu_composition_calls() - comp_before;
    println!(
        "S2DEV vector proofs: {} files, {fri_commits} one-row device FRI commits, {trees} one-row device trees, {compositions} device compositions",
        files.len()
    );
    // Two (e) formats x two hashes, two files per proof.
    assert_eq!(files.len(), 2 * 2 * 2);
    assert_eq!(
        fri_commits, 4,
        "every one-row vector proof must take the device FRI commit (lower LAMBDA_VM_GPU_LDE_THRESHOLD)"
    );
    assert!(
        trees >= 3 * 4,
        "every one-row vector proof must build its main, aux and composition trees on the device \
         ({trees} one-row device trees for 4 proofs)"
    );
    // Every proof composes on the device (the AIR's constraint program); a
    // host composition would not be counted here.
    assert_eq!(
        compositions, 4,
        "every one-row vector proof must compose on the device ({compositions} device compositions)"
    );
    let bad = check_or_write(&files, false);
    assert!(
        bad.is_empty(),
        "device-proved one-row vectors differ from the checked-in CPU bytes: {bad:?}"
    );
}

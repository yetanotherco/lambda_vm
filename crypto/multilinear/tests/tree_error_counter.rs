//! ★ A device Merkle tree over a HOST codeword that FAILS is counted and logged,
//! and the host builds the same tree.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test tree_error_counter -- --nocapture
//! ```
//!
//! Needs a GPU. `commit_tree_ext3` is where a chain whose codeword lives on the
//! host (after a commit fell back) hashes its extension-field folds. It used to
//! turn a device error into a silent `None`, which no counter saw: it is
//! neither a device commit nor a commit fallback. The failure here is a real
//! one. That path allocates without the VRAM ledger, so the card itself is
//! filled first, and the tree's first allocation returns
//! `CUDA_ERROR_OUT_OF_MEMORY`. The run should print one
//! `[whir] device commit (ext3 tree) ... failed` line.
//!
//! Its own integration binary: the counters are process-wide, and a full card
//! would fail anything running beside it.
#![cfg(feature = "cuda")]

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;
use multilinear::gpu::{commit_calls, commit_errors, reset_call_counters, tree_commit_errors};
use multilinear::whir_commit::CodewordCommitment;
use multilinear::whir_hash::RpxWhir;

/// An extension-field codeword with every limb in use.
fn codeword(log_len: u32) -> Vec<FieldElement<Ext3>> {
    (0..1u64 << log_len)
        .map(|i| {
            let limb = |k: u64| FieldElement::<Gl>::from(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k);
            FieldElement::<Ext3>::new([limb(1), limb(2), limb(3)])
        })
        .collect()
}

/// `(device commits, tree commit errors, commit errors)`.
fn counts() -> (u64, u64, u64) {
    (commit_calls(), tree_commit_errors(), commit_errors())
}

#[test]
fn a_failed_device_tree_is_counted_as_an_error_and_built_on_the_host() {
    let be = math_cuda::device::backend().expect("this test needs a GPU");
    const LOG_FOLDING: usize = 4;
    // 2^18 values (6 MiB): above the device threshold, so the device is asked.
    let values = codeword(18);
    reset_call_counters();

    // The whole card held, down to the last MiB the pool will hand out.
    let stream = be.next_stream();
    let mut held = Vec::new();
    let mut chunk = 1usize << 30;
    while chunk >= 1 << 20 {
        match stream.alloc_zeros::<u8>(chunk) {
            Ok(buffer) => held.push(buffer),
            Err(_) => chunk /= 2,
        }
    }
    let held_mib: usize = held.iter().map(|buffer| buffer.len() >> 20).sum();
    println!("tree_error_counter: {held_mib} MiB of device memory held");
    let on_host = CodewordCommitment::<Ext3, RpxWhir>::new(&values, LOG_FOLDING).expect("commit");
    assert_eq!(counts(), (0, 1, 0), "the tree's error, and nothing else");
    drop(held);
    stream.synchronize().expect("the frees");
    // Back to the driver, so the next allocation does not depend on the pool
    // handing another stream what this one freed.
    be.trim_mempool_to(0);

    // The same tree with the card free is built on the device, to the same root.
    let on_device = CodewordCommitment::<Ext3, RpxWhir>::new(&values, LOG_FOLDING).expect("commit");
    assert_eq!(counts(), (1, 1, 0), "a device tree is not an error");
    assert_eq!(on_host.root(), on_device.root());

    // Below the threshold the device is not asked: nothing is counted.
    let _ =
        CodewordCommitment::<Ext3, RpxWhir>::new(&values[..1 << 10], LOG_FOLDING).expect("commit");
    assert_eq!(counts(), (1, 1, 0), "a decline is not an error");
}

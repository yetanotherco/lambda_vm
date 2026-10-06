//! ★ The Poseidon1 grind's dispatch gate: `generate_nonce_maybe_gpu` over
//! `P1GrindDigest` reaches the width-8 device kernel (`p1s_grind_w8`), and
//! only that one. The device search's own parity is
//! `crypto/math-cuda/tests/p1_device_parity.rs`; this file needs the prover's
//! `cuda` feature because the counters are `crypto`'s, behind `crypto/cuda`.
//!
//! Every arm returns a valid nonce whatever happens (the host search is
//! correct), so validity alone would pass with the device never touched. The
//! counters count only device searches whose nonce passed the host check, so
//! the deltas are the claim. One test, because the counters are process-global.
//! Needs a GPU:
//!
//!   cargo test -p lambda-vm-prover --release --features cuda --test p1_grind_dispatch
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::p1_transcript::P1GrindDigest;
use crypto::grinding::{DeviceGrindKey, GrindDigest};

#[test]
fn the_p1_grind_dispatch_reaches_its_own_kernel() {
    assert_eq!(
        <P1GrindDigest as GrindDigest>::DEVICE_GRIND,
        Some(DeviceGrindKey::Poseidon1),
        "the P1 digest must declare the P1 device arm"
    );
    let seed = [9u8; 32];
    let factor = 18u8;
    let (p1, rpx, keccak) = (
        crypto::grinding::gpu_grind_calls_p1(),
        crypto::grinding::gpu_grind_calls_rpx(),
        crypto::grinding::gpu_grind_calls(),
    );
    let nonce = crypto::grinding::generate_nonce_maybe_gpu::<P1GrindDigest>(&seed, factor)
        .expect("a nonce");
    assert!(
        crypto::grinding::is_valid_nonce::<P1GrindDigest>(&seed, nonce, factor),
        "the dispatched nonce {nonce} fails is_valid_nonce"
    );
    assert_eq!(
        crypto::grinding::generate_nonce_smallest::<P1GrindDigest>(&seed, factor),
        Some(nonce),
        "the dispatched nonce is the smallest valid one"
    );
    assert_eq!(
        crypto::grinding::gpu_grind_calls_p1(),
        p1 + 1,
        "the P1 arm must have run on the device (LAMBDA_VM_NO_GPU_GRIND makes this fail, which \
         is the point)"
    );
    assert_eq!(crypto::grinding::gpu_grind_calls_rpx(), rpx, "no RPX grind");
    assert_eq!(
        crypto::grinding::gpu_grind_calls(),
        keccak,
        "no keccak grind"
    );
}

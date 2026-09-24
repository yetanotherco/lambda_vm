//! The device commit of a stacked polynomial against the host pipeline it
//! mirrors: same codeword, same root, and openings that verify against the
//! device tree.
//!
//! Runs on the merge-queue GPU box via `make test-math-cuda` — `commit_codeword`
//! needs a real device, like the other tests here.
//!
//! The reference is `multilinear`'s own pipeline rather than a copy of it: a
//! second implementation of the Möbius transform in this file could drift from
//! the one the prover runs with every test still green.
//!
//! ⚠ **Both hashes, and the device path asserted TAKEN.** The commit falls back
//! to the host silently when the device declines (`gpu::commit_tree_ext3`
//! returns `None` below a size threshold, on a missing card, or under
//! `LAMBDA_VM_NO_GPU_WHIR_COMMIT`). A parity test that let that happen would be
//! comparing the host against itself and passing — the exact shape recon B
//! found in `whir_fold.rs`, where two host paths were checked against each
//! other. So `commit_codeword_to_host` is called directly, which has no host
//! fallback at all: it either runs the kernels or returns an error this test
//! turns into a failure.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::mle::Mle;
use multilinear::whir::{self, Domain};
use multilinear::whir_commit::{CodewordCommitment, verify_opening};
use multilinear::whir_hash::{DeviceHashKey, KeccakWhir, RpxWhir, WhirHash};

type FE = FieldElement<F>;

/// The dispatch key `math-cuda` takes, from the configuration that names it.
///
/// ⚠ A MIRROR of `DeviceHashKey::into_math_cuda`, not a call to it. That method
/// is `#[cfg(feature = "cuda")]` on `multilinear`, and `multilinear` is a plain
/// dev-dependency here — enabling its cuda feature from this crate would be a
/// dependency cycle through the crate under test. The production bridge is
/// asserted total and bijective at compile time inside `multilinear`
/// (`whir_hash.rs`); what this mirror can still get wrong is being written
/// backwards, which `the_device_dispatch_really_selects_the_kernel_family`
/// would catch — it requires the two keys to produce DIFFERENT trees, and a
/// swapped mirror produces the same two trees in the other order.
fn device_key<H: WhirHash>() -> math_cuda::DeviceHash {
    match H::DEVICE {
        DeviceHashKey::Keccak256 => math_cuda::DeviceHash::Keccak256,
        DeviceHashKey::Rpx256 => math_cuda::DeviceHash::Rpx256,
    }
}

/// A polynomial with no structure a kernel could accidentally satisfy.
fn poly(num_vars: usize, seed: u64) -> Mle<F> {
    let evals: Vec<FE> = (0..(1u64 << num_vars))
        .map(|i| FE::from(i.wrapping_mul(6364136223846793005).wrapping_add(seed) >> 11))
        .collect();
    Mle::new(evals).expect("power of two")
}

fn parity<H: WhirHash>(num_vars: usize, log_blowup: usize, log_folding: usize) {
    let f = poly(num_vars, 1 + num_vars as u64);
    let raw: Vec<u64> = f.evals().iter().map(|v| *v.value()).collect();
    // No host fallback on this entry point: it hashes on the device or errors.
    let (device_codeword, nodes) =
        math_cuda::whir::commit_codeword_to_host(&raw, log_blowup, log_folding, device_key::<H>())
            .unwrap_or_else(|e| panic!("device commit under {} (needs a GPU): {e:?}", H::NAME));

    let domain = Domain::<F>::new(num_vars + log_blowup).expect("domain");
    let host_codeword =
        whir::encode::<F, F>(&whir::lift_coefficients(&f), &domain).expect("encode");
    let host = CodewordCommitment::<_, H>::new(&host_codeword, log_folding).expect("host commit");

    assert_eq!(device_codeword.len(), host_codeword.len());
    for (i, (device, host)) in device_codeword.iter().zip(&host_codeword).enumerate() {
        assert_eq!(
            device,
            host.value(),
            "codeword position {i} differs at 2^{num_vars}, blowup 2^{log_blowup}"
        );
    }

    let nodes: Vec<[u8; 32]> = nodes
        .chunks_exact(32)
        .map(|node| node.try_into().expect("32 bytes"))
        .collect();
    let codeword: Vec<FE> = device_codeword.into_iter().map(FE::from_raw).collect();
    let device = CodewordCommitment::<_, H>::from_precomputed(codeword, nodes, log_folding)
        .expect("device commitment");
    assert_eq!(device.root(), host.root(), "roots differ under {}", H::NAME);
    assert_eq!(device.num_leaves(), host.num_leaves());

    // The root alone would pass on a tree whose inner nodes are garbage below
    // it, so an opening is checked against it too.
    for index in [0, 1, device.num_leaves() / 3, device.num_leaves() - 1] {
        let opening = device.open(index).expect("open");
        assert!(
            verify_opening::<_, H>(&device.root(), device.depth(), index, &opening),
            "device opening at {index} does not verify under {}",
            H::NAME
        );
        assert_eq!(
            opening.values,
            host.open(index).expect("open").values,
            "opened block {index} differs"
        );
    }
}

/// The shapes: both sides of the fused-8-level NTT threshold, a fold width that
/// is not the whole blowup, and the Möbius windows — below the contiguous
/// kernel, exactly one window, one window plus a tiled level, and several full
/// tiles with a partial one on top.
fn every_shape<H: WhirHash>() {
    parity::<H>(14, 2, 4);
    parity::<H>(16, 2, 4);
    parity::<H>(11, 1, 1);
    parity::<H>(12, 3, 5);

    parity::<H>(5, 2, 3);
    parity::<H>(8, 2, 4);
    parity::<H>(9, 1, 2);
    parity::<H>(13, 2, 5);
    parity::<H>(17, 1, 4);
}

#[test]
fn device_commit_matches_the_host_pipeline() {
    every_shape::<KeccakWhir>();
}

/// ★ The same, under the algebraic hash — the kernels this branch adds.
#[test]
fn device_commit_matches_the_host_pipeline_under_rpx() {
    every_shape::<RpxWhir>();
}

/// ★★ The two hashes really do build DIFFERENT trees on the device.
///
/// Without this, both tests above would pass on a dispatch that ignored its key
/// and always launched keccak's kernels: the RPX host reference would be
/// compared against a keccak device tree and fail — unless the host side had
/// been mis-wired the same way, which is exactly the failure a single-hash
/// parity test cannot see. Comparing the two device roots directly closes it.
#[test]
fn the_device_dispatch_really_selects_the_kernel_family() {
    let f = poly(12, 7);
    let raw: Vec<u64> = f.evals().iter().map(|v| *v.value()).collect();

    let (_, keccak) =
        math_cuda::whir::commit_codeword_to_host(&raw, 2, 4, device_key::<KeccakWhir>())
            .expect("device commit (needs a GPU)");
    let (_, rpx) = math_cuda::whir::commit_codeword_to_host(&raw, 2, 4, device_key::<RpxWhir>())
        .expect("device commit (needs a GPU)");

    assert_eq!(keccak.len(), rpx.len(), "the node layout must not change");
    assert_ne!(
        keccak, rpx,
        "the two kernel families produced identical trees, so the key is not being read"
    );
}

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

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::mle::Mle;
use multilinear::whir::{self, Domain};
use multilinear::whir_commit::{CodewordCommitment, verify_opening};

type FE = FieldElement<F>;

/// A polynomial with no structure a kernel could accidentally satisfy.
fn poly(num_vars: usize, seed: u64) -> Mle<F> {
    let evals: Vec<FE> = (0..(1u64 << num_vars))
        .map(|i| FE::from(i.wrapping_mul(6364136223846793005).wrapping_add(seed) >> 11))
        .collect();
    Mle::new(evals).expect("power of two")
}

fn parity(num_vars: usize, log_blowup: usize, log_folding: usize) {
    let f = poly(num_vars, 1 + num_vars as u64);
    let raw: Vec<u64> = f.evals().iter().map(|v| *v.value()).collect();
    let (device_codeword, nodes) =
        math_cuda::whir::commit_codeword_to_host(&raw, log_blowup, log_folding)
            .expect("device commit (needs a GPU)");

    let domain = Domain::<F>::new(num_vars + log_blowup).expect("domain");
    let host_codeword =
        whir::encode::<F, F>(&whir::lift_coefficients(&f), &domain).expect("encode");
    let host = CodewordCommitment::new(&host_codeword, log_folding).expect("host commit");

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
    let device = CodewordCommitment::from_precomputed(codeword, nodes, log_folding)
        .expect("device commitment");
    assert_eq!(device.root(), host.root(), "roots differ");
    assert_eq!(device.num_leaves(), host.num_leaves());

    // The root alone would pass on a tree whose inner nodes are garbage below
    // it, so an opening is checked against it too.
    for index in [0, 1, device.num_leaves() / 3, device.num_leaves() - 1] {
        let opening = device.open(index).expect("open");
        assert!(
            verify_opening(&device.root(), index, &opening),
            "device opening at {index} does not verify"
        );
        assert_eq!(
            opening.values,
            host.open(index).expect("open").values,
            "opened block {index} differs"
        );
    }
}

#[test]
fn device_commit_matches_the_host_pipeline() {
    // Both sides of the fused-8-level NTT threshold, and a fold width that is
    // not the whole blowup.
    parity(14, 2, 4);
    parity(16, 2, 4);
    parity(11, 1, 1);
    parity(12, 3, 5);
}

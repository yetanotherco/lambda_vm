//! Direct parity test for `coset_lde_batch_base_into` (lde.rs), the
//! caller-allocated-buffer variant of `coset_lde_batch_base`. The two should
//! produce bit-identical canonical output for the same inputs; the only
//! difference between them is who owns the output Vec.
//!
//! This is otherwise covered indirectly through `coset_lde_batch_base_into_with_leaf_hash`
//! and similar, but the base `_into` variant ships as public API with no
//! direct test in the original PR.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;

fn coset_weights(n: usize, g: u64) -> Vec<u64> {
    let inv_n = *Fp::from(n as u64).inv().unwrap().value();
    let mut w = Vec::with_capacity(n);
    let mut cur = inv_n;
    for _ in 0..n {
        w.push(cur);
        cur = GoldilocksField::mul(&cur, &g);
    }
    w
}

fn canon(xs: &[u64]) -> Vec<u64> {
    xs.iter().map(GoldilocksField::canonical).collect()
}

fn run_pair(log_n: u64, blowup: usize, m: usize, seed: u64) {
    let n = 1usize << log_n;
    let lde_size = n * blowup;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let columns: Vec<Vec<u64>> = (0..m)
        .map(|_| (0..n).map(|_| rng.r#gen::<u64>()).collect())
        .collect();

    let weights = coset_weights(n, 7);

    let slices: Vec<&[u64]> = columns.iter().map(|c| c.as_slice()).collect();

    // Reference: the existing batch-allocates-Vec API. Random-input tests
    // already cross-validate this against the CPU single-column LDE in
    // `lde_batch.rs`, so any divergence from it pinpoints `_into`.
    let ref_out = math_cuda::lde::coset_lde_batch_base(&slices, blowup, &weights).unwrap();
    assert_eq!(ref_out.len(), m);

    // Caller-allocated buffers.
    let mut owned: Vec<Vec<u64>> = (0..m).map(|_| vec![0u64; lde_size]).collect();
    {
        let mut outs: Vec<&mut [u64]> = owned.iter_mut().map(|v| v.as_mut_slice()).collect();
        math_cuda::lde::coset_lde_batch_base_into(&slices, blowup, &weights, &mut outs)
            .expect("into variant");
    }

    for c in 0..m {
        assert_eq!(
            canon(&owned[c]),
            canon(&ref_out[c]),
            "_into vs _batch_base diverge at column {c}, log_n={log_n}, blowup={blowup}, m={m}"
        );
    }
}

#[test]
fn into_matches_batch_base_small() {
    run_pair(8, 4, 4, 1);
    run_pair(10, 4, 1, 2);
    run_pair(10, 4, 16, 3);
}

#[test]
fn into_matches_batch_base_medium() {
    run_pair(14, 4, 8, 4);
    run_pair(15, 4, 32, 5);
}

#[test]
fn into_matches_batch_base_uneven_blowup() {
    // Non-default blowup factors (still power of two) — confirms the
    // _into variant respects blowup_factor identically.
    run_pair(8, 2, 4, 6);
    run_pair(8, 8, 4, 7);
}

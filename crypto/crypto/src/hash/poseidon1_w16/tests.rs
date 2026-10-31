//! The width-16 instance's known answers, and the checks that make them bind.
//!
//! The vectors in [`super::kat`] come from `scripts/poseidon1/p1_params.py`,
//! a Python reference written from the Poseidon paper, and the permutation
//! vectors are reproduced by two implementations we did not write: Plonky3's
//! generator's reference permutation and Plonky3's Rust `Poseidon1` (sparse
//! partial rounds) fed these constants (lane note `I-HASH.md` §1).

use super::kat::{
    CAUCHY_LEAF_VECTORS, CAUCHY_NODE4_VECTOR, CAUCHY_PERMUTATION_VECTORS, LEAF_VECTORS,
    NODE4_VECTOR, PERMUTATION_VECTORS,
};
use super::*;

fn fp16(v: &[u64; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    core::array::from_fn(|i| Fp::from(v[i]))
}

fn canon(v: &[Fp]) -> alloc::vec::Vec<u64> {
    v.iter().map(|f| f.canonical()).collect()
}

#[test]
fn the_permutation_matches_the_reference_vectors() {
    for (input, expected) in PERMUTATION_VECTORS.iter() {
        assert_eq!(canon(&permute(fp16(input))), expected.to_vec());
    }
}

#[test]
fn a_one_bit_change_in_one_round_constant_breaks_every_vector() {
    // Mutation: the vectors must pin the LAST constant of the LAST round (the
    // one a truncated or off-by-one table would lose), not only the early ones.
    let mut rc = ROUND_CONSTANTS;
    rc[NUM_ROUNDS - 1][STATE_FELTS - 1] ^= 1;
    for (input, expected) in PERMUTATION_VECTORS.iter() {
        assert_ne!(
            canon(&permute_with(&rc, mds, fp16(input))),
            expected.to_vec()
        );
    }
}

#[test]
fn the_leaf_sponge_matches_the_reference_vectors() {
    for (n, expected) in LEAF_VECTORS.iter() {
        let felts: alloc::vec::Vec<Fp> = (0..*n as u64)
            // i < 64, so i·c + n < p and the reference's `mod p` is a no-op.
            .map(|i| Fp::from(i * 0x0123_4567_89ab_cdef + *n as u64))
            .collect();
        assert_eq!(
            canon(&sponge_leaf(&felts)),
            expected.to_vec(),
            "leaf of {n} felts"
        );
    }
}

#[test]
fn the_4ary_node_matches_the_reference_vector() {
    let out = permute(fp16(&PERMUTATION_VECTORS[0].0));
    let children: [Digest; ARITY] =
        core::array::from_fn(|c| core::array::from_fn(|l| out[c * DIGEST_FELTS + l]));
    assert_eq!(canon(&compress4(&children)), NODE4_VECTOR.to_vec());
}

/// The Cauchy alternative's known answers — from the Python reference, whose
/// permutation vectors Plonky3's generator (and its Cauchy matrix) reproduce.
#[test]
fn the_cauchy_alternative_matches_its_reference_vectors() {
    for (input, expected) in CAUCHY_PERMUTATION_VECTORS.iter() {
        assert_eq!(canon(&permute_cauchy(fp16(input))), expected.to_vec());
    }
    for (n, expected) in CAUCHY_LEAF_VECTORS.iter() {
        let felts: alloc::vec::Vec<Fp> = (0..*n as u64)
            .map(|i| Fp::from(i * 0x0123_4567_89ab_cdef + *n as u64))
            .collect();
        assert_eq!(
            canon(&sponge_leaf_cauchy(&felts)),
            expected.to_vec(),
            "leaf of {n} felts"
        );
    }
    let out = permute_cauchy(fp16(&CAUCHY_PERMUTATION_VECTORS[0].0));
    let children: [Digest; ARITY] =
        core::array::from_fn(|c| core::array::from_fn(|l| out[c * DIGEST_FELTS + l]));
    assert_eq!(
        canon(&compress4_cauchy(&children)),
        CAUCHY_NODE4_VECTOR.to_vec()
    );
    // The two instances differ everywhere they should.
    assert_ne!(CAUCHY_PERMUTATION_VECTORS[0].1, PERMUTATION_VECTORS[0].1);
}

#[test]
fn the_node_depends_on_child_order() {
    let out = permute(fp16(&PERMUTATION_VECTORS[0].0));
    let mut children: [Digest; ARITY] =
        core::array::from_fn(|c| core::array::from_fn(|l| out[c * DIGEST_FELTS + l]));
    children.swap(1, 2);
    assert_ne!(canon(&compress4(&children)), NODE4_VECTOR.to_vec());
}

#[test]
fn the_mds_matches_a_dense_product_on_extreme_inputs() {
    let s = [Fp::from(GOLDILOCKS_P - 1); STATE_FELTS];
    let fast = mds(&s);
    for (i, f) in fast.iter().enumerate() {
        let mut acc = Fp::zero();
        for (j, v) in s.iter().enumerate() {
            acc += Fp::from(MDS_CIRC_ROW[(j + STATE_FELTS - i) % STATE_FELTS]) * v;
        }
        assert_eq!(*f, acc);
    }
}

#[test]
fn the_round_counts_meet_the_papers_bounds_with_the_margin() {
    // Minimum before the margin: R_F = 6 (statistical, M = 128 <= (64 - 3)·17),
    // R_F + R_P >= 1 + ceil(64·log_7 2) + ceil(log_7 16) = 1 + 23 + 2 (interpolation).
    let log7_2 = core::f64::consts::LN_2 / 7f64.ln();
    let interp = 1 + (64.0 * log7_2).ceil() as usize + (16f64.ln() / 7f64.ln()).ceil() as usize;
    assert_eq!(interp, 26);
    let (rf_min, rp_min) = (6usize, interp - 6);
    assert_eq!(2 * HALF_FULL_ROUNDS, rf_min + 2);
    assert_eq!(PARTIAL_ROUNDS, (rp_min as f64 * 1.075).ceil() as usize);
}

const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// Host cost per permutation against RPX's, same process and inputs
/// (informational, `--ignored --nocapture`; a laptop number is not a box number).
#[test]
#[ignore = "timing; run with --ignored --nocapture"]
fn host_permutation_cost_against_rpx() {
    use std::time::Instant;
    const N: usize = 200_000;
    let mut s16 = fp16(&PERMUTATION_VECTORS[3].0);
    let t = Instant::now();
    for _ in 0..N {
        s16 = permute(s16);
    }
    let p1 = t.elapsed().as_nanos() as f64 / N as f64;
    let mut sc = fp16(&PERMUTATION_VECTORS[3].0);
    let t = Instant::now();
    for _ in 0..N {
        sc = permute_cauchy(sc);
    }
    let cauchy = t.elapsed().as_nanos() as f64 / N as f64;
    let mut s12: [crate::hash::rpx::Fp; 12] = core::array::from_fn(|i| s16[i]);
    let t = Instant::now();
    for _ in 0..N {
        s12 = crate::hash::rpx::permute(s12);
    }
    let rpx = t.elapsed().as_nanos() as f64 / N as f64;
    println!(
        "HOSTPERM p1w16 {p1:.0} ns/perm · p1w16-cauchy {cauchy:.0} · rpx {rpx:.0} ns/perm · \
         ratios {:.2} / {:.2} (sinks {} {} {})",
        p1 / rpx,
        cauchy / rpx,
        s16[0].canonical(),
        sc[0].canonical(),
        s12[0].canonical()
    );
}

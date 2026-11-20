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

// =============================================================================
// The sparse partial rounds against the textbook form
// =============================================================================

/// Random states with edge values planted: every seventh lane is 0, 1, p − 1
/// or a non-canonical word.
#[cfg(feature = "std")]
fn random_states(n: usize, seed: u64) -> alloc::vec::Vec<[Fp; STATE_FELTS]> {
    const P: u64 = 0xFFFF_FFFF_0000_0001;
    const EDGES: [u64; 6] = [0, 1, P - 1, P, P + 1, u64::MAX];
    let mut z = seed;
    let mut next = || {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };
    (0..n)
        .map(|k| {
            core::array::from_fn(|i| {
                if (k * STATE_FELTS + i).is_multiple_of(7) {
                    Fp::from(EDGES[(k + i) % EDGES.len()])
                } else {
                    Fp::from(next())
                }
            })
        })
        .collect()
}

/// ★ The sparse form is the textbook permutation: equal on 100,000 random
/// states (edge values planted) and on the known answers' inputs, and
/// `permute` (the production entry) is the sparse form under `std`.
#[cfg(feature = "std")]
#[test]
fn the_sparse_form_is_the_textbook_permutation() {
    let t = fast::tables();
    for (k, s) in random_states(100_000, 7).into_iter().enumerate() {
        assert_eq!(
            canon(&fast::permute(t, s)),
            canon(&permute_textbook(s)),
            "state {k}"
        );
    }
    for (input, expected) in PERMUTATION_VECTORS.iter() {
        assert_eq!(canon(&fast::permute(t, fp16(input))), expected.to_vec());
        assert_eq!(canon(&permute_textbook(fp16(input))), expected.to_vec());
    }
    let s = fp16(&PERMUTATION_VECTORS[1].0);
    assert_eq!(canon(&permute(s)), canon(&fast::permute(t, s)));
}

/// Each derived table is load-bearing: one moved entry of the lane-0
/// constants, of a sparse layer's `w` or `v`, of the tail matrix or of the
/// carried constants makes the sparse form differ from the textbook one.
#[cfg(feature = "std")]
#[test]
fn every_derived_table_of_the_sparse_form_is_load_bearing() {
    let states = random_states(64, 11);
    type Mutant = (&'static str, fn(&mut fast::Tables));
    let mutants: [Mutant; 6] = [
        ("lane-0 constant", |t| t.lane0[7] ^= 1),
        ("sparse m00", |t| t.sparse[0].m00 ^= 1),
        ("sparse w", |t| t.sparse[12].w[3] ^= 1),
        ("sparse v", |t| t.sparse[21].v[14] ^= 1),
        ("tail matrix", |t| t.tail[5][9] ^= 1),
        ("carried constants", |t| t.full[4][13] += Fp::one()),
    ];
    for (what, mutate) in mutants {
        let mut t = fast::tables().clone();
        mutate(&mut t);
        let differs = states
            .iter()
            .any(|s| canon(&fast::permute(&t, *s)) != canon(&permute_textbook(*s)));
        assert!(differs, "a moved {what} is not caught");
    }
}

/// ⛔ A MEASUREMENT (ignored): microseconds a permutation, textbook against
/// sparse, one thread. Prints `P1W16 PERMUTE` lines.
#[cfg(feature = "std")]
#[test]
#[ignore = "a measurement"]
fn probe_the_sparse_forms_speed() {
    let states = random_states(200_000, 3);
    let t = fast::tables();
    for _ in 0..2 {
        let start = std::time::Instant::now();
        let mut acc = Fp::zero();
        for s in &states {
            acc += permute_textbook(*s)[0];
        }
        let textbook = start.elapsed().as_secs_f64() * 1e6 / states.len() as f64;
        let start = std::time::Instant::now();
        let mut acc2 = Fp::zero();
        for s in &states {
            acc2 += fast::permute(t, *s)[0];
        }
        let sparse = start.elapsed().as_secs_f64() * 1e6 / states.len() as f64;
        assert_eq!(acc, acc2);
        std::println!(
            "P1W16 PERMUTE: textbook {textbook:.3} µs · sparse {sparse:.3} µs · {:.2}×",
            textbook / sparse
        );
    }
}

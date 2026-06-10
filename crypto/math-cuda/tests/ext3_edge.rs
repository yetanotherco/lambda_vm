//! Adversarial edge-case parity tests for `ext3::mul` on the GPU.
//!
//! The CUDA `dot3` in kernels/ext3.cuh manually tracks overflow when
//! summing three u128 products in split u64 hi/lo registers. The CPU
//! reference (`crypto/math/src/field/goldilocks.rs::dot_product_3` via
//! `Degree3GoldilocksExtensionField::mul`) uses native u128 and so reaches
//! the same answer via a totally different code path. These tests pick
//! inputs that maximally stress the overflow-count tracking, the
//! non-canonical input handling, and the identity/zero cases that random
//! tests are unlikely to cover.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

const P: u64 = 0xFFFF_FFFF_0000_0001; // Goldilocks prime
const EPSILON: u64 = 0xFFFF_FFFF; // 2^32 - 1

fn canon(x: u64) -> u64 {
    GoldilocksField::canonical(&x)
}

fn canon3(t: [u64; 3]) -> [u64; 3] {
    [canon(t[0]), canon(t[1]), canon(t[2])]
}

/// Run `pairs` of (a, b) through the GPU `ext3_mul_u64` and compare
/// canonical output to the CPU `Degree3GoldilocksExtensionField::mul`.
/// `label_fn(i)` produces a per-case label printed on failure.
fn assert_ext3_mul_pairs(pairs: &[([u64; 3], [u64; 3])], label_fn: impl Fn(usize) -> String) {
    let mut a_raw = Vec::with_capacity(pairs.len() * 3);
    let mut b_raw = Vec::with_capacity(pairs.len() * 3);
    for (a, b) in pairs {
        a_raw.extend_from_slice(a);
        b_raw.extend_from_slice(b);
    }
    let gpu = math_cuda::ext3_mul_u64(&a_raw, &b_raw).expect("GPU ext3 mul launch");
    assert_eq!(gpu.len(), 3 * pairs.len());
    for (i, (a, b)) in pairs.iter().enumerate() {
        // CPU reference. Build via from_raw so non-canonical inputs (like
        // u64::MAX or p, p+1) are passed in untouched — matching what the
        // GPU sees.
        let ae = [Fp::from_raw(a[0]), Fp::from_raw(a[1]), Fp::from_raw(a[2])];
        let be = [Fp::from_raw(b[0]), Fp::from_raw(b[1]), Fp::from_raw(b[2])];
        let cpu = Degree3GoldilocksExtensionField::mul(&ae, &be);
        let cpu_fp3 = Fp3::new(cpu);
        let g = canon3([gpu[3 * i], gpu[3 * i + 1], gpu[3 * i + 2]]);
        let c = canon3([
            *cpu_fp3.value()[0].value(),
            *cpu_fp3.value()[1].value(),
            *cpu_fp3.value()[2].value(),
        ]);
        assert_eq!(
            g,
            c,
            "ext3 mul mismatch [{}]: a={:?} b={:?} gpu={:?} cpu={:?}",
            label_fn(i),
            a,
            b,
            g,
            c,
        );
    }
}

#[test]
fn ext3_mul_max_canonical_inputs() {
    // (p-1, p-1, p-1) * (p-1, p-1, p-1) — every base limb is (p-1), every
    // dot3 product a_i*b_i = (p-1)^2 ~ 2^128, so summing three of them
    // forces the overflow path twice on each component.
    let m1 = [P - 1, P - 1, P - 1];
    let pairs = vec![(m1, m1)];
    assert_ext3_mul_pairs(&pairs, |_| "(p-1)^3 squared".into());
}

#[test]
fn ext3_mul_zero_cases() {
    // (0,0,0) * (p-1,p-1,p-1) must be zero; covers the "all-zero a" path
    // where every dot3 product is zero and no overflow occurs.
    let z = [0u64, 0, 0];
    let m = [P - 1, P - 1, P - 1];
    let pairs = vec![(z, m), (m, z), (z, z)];
    assert_ext3_mul_pairs(&pairs, |i| format!("zero case {i}"));
}

#[test]
fn ext3_mul_identity() {
    // (1, 0, 0) * (a, b, c) == (a, b, c). One-component is multiplicative
    // identity in Fp[w]/(w^3 - 2). Use varied b to also exercise small
    // non-zero dot3 products.
    let id = [1u64, 0, 0];
    let cases: Vec<([u64; 3], [u64; 3])> = vec![
        (id, [0, 0, 0]),
        (id, [1, 0, 0]),
        (id, [0, 1, 0]),
        (id, [0, 0, 1]),
        (id, [P - 1, 1, 2]),
        (id, [123, 456, 789]),
        (id, [P - 1, P - 1, P - 1]),
        // Reverse order: (a, b, c) * (1, 0, 0).
        ([0, 0, 0], id),
        ([1, 0, 0], id),
        ([0, 1, 0], id),
        ([0, 0, 1], id),
        ([P - 1, 1, 2], id),
        ([123, 456, 789], id),
    ];
    assert_ext3_mul_pairs(&cases, |i| format!("identity case {i}"));
}

#[test]
fn ext3_mul_non_canonical_zero_p() {
    // (p, p, p) is a non-canonical representation of (0, 0, 0). The CPU
    // canonicalises before the dot3 in some code paths, the GPU does not.
    // Either way the product must canonicalise to zero.
    let p = [P, P, P];
    let some = [123u64, 456, 789];
    let pairs = vec![(p, p), (p, some), (some, p)];
    assert_ext3_mul_pairs(&pairs, |i| format!("non-canonical-p case {i}"));
}

#[test]
fn ext3_mul_u64_max_all_overflow_paths() {
    // (u64::MAX, u64::MAX, u64::MAX) for both operands. u64::MAX = p + (2^32 - 2),
    // i.e., a non-canonical representation of (2^32 - 2) mod p. Every dot3
    // product is ~2^128 - small, so summing three of them is the hardest
    // possible exercise of `over1`, `over2`, and the EPSILON^2 correction
    // path in `dot3`.
    let m = [u64::MAX, u64::MAX, u64::MAX];
    assert_ext3_mul_pairs(&[(m, m)], |_| "u64::MAX^3 squared".into());
}

#[test]
fn ext3_mul_base_edge_pairs_embedded() {
    // Embed every base-field edge value as the `a` component of an ext3
    // element (so b = c = 0) and run all NxN pairs through GPU mul. This
    // reduces to base-field multiplication on the a-component but
    // exercises all the dot3 zero/non-zero combinations.
    let edges: Vec<u64> = vec![0, 1, P - 1, P, P + 1, u64::MAX, EPSILON];
    let mut pairs = Vec::with_capacity(edges.len() * edges.len());
    for x in &edges {
        for y in &edges {
            pairs.push(([*x, 0, 0], [*y, 0, 0]));
        }
    }
    assert_ext3_mul_pairs(&pairs, |i| {
        let xi = i / edges.len();
        let yi = i % edges.len();
        format!("base-edge a={:#x} b={:#x}", edges[xi], edges[yi])
    });
}

#[test]
fn ext3_mul_base_edges_in_b_and_c_slots() {
    // Same edge values, but placed in the b and c slots so the cross terms
    // (which involve the `b1_2 = 2*y.b`, `b2_2 = 2*y.c` doubling) are also
    // exercised at edge inputs. Non-canonical doubling of P-1 etc. is a
    // path that random tests rarely hit.
    let edges: Vec<u64> = vec![0, 1, P - 1, P, P + 1, u64::MAX, EPSILON];
    let mut pairs = Vec::with_capacity(edges.len() * edges.len());
    for x in &edges {
        for y in &edges {
            // Put edge values in b/c slots, a non-trivial.
            pairs.push(([1, *x, *y], [1, *x, *y]));
        }
    }
    assert_ext3_mul_pairs(&pairs, |i| {
        let xi = i / edges.len();
        let yi = i % edges.len();
        format!("ext3-bc-edge b={:#x} c={:#x}", edges[xi], edges[yi])
    });
}

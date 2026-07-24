//! Host validation for `lincomb2_witness` (phase A).
//!
//! Three independent cross-checks per case:
//!  1. `Q` matches an `Fp`-reference lincomb (`tests::reference`, a different
//!     implementation from the witness's own BigInt group law) AND k256.
//!  2. every emitted joint-chain row re-satisfies its double/add relation and
//!     slope (on top of the `limb_carries` asserts that already fire at build
//!     time if any quotient/carry is wrong).
//!  3. the NUMS blind cancels: the pre-correction accumulator equals
//!     `Q + 2^len·T₀`, and the correction row lands on `Q`.

use num_bigint::BigUint;

use crate::curve::AffinePoint;
use crate::tests::reference::{point_add, point_double, step_lambda};
use crate::witness::{JointSel, Lincomb2Error, lincomb2_witness, t0};
use crate::{n, p};

use k256::elliptic_curve::ff::PrimeField as _;
use k256::elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
use k256::{AffinePoint as K256Affine, EncodedPoint, ProjectivePoint, Scalar};

// ---- helpers ----

fn be32(v: &BigUint) -> [u8; 32] {
    let b = v.to_bytes_be();
    let mut out = [0u8; 32];
    out[32 - b.len()..].copy_from_slice(&b);
    out
}

fn to_k256(a: &AffinePoint) -> ProjectivePoint {
    let ep = EncodedPoint::from_affine_coordinates(&be32(&a.x).into(), &be32(&a.y).into(), false);
    ProjectivePoint::from(K256Affine::from_encoded_point(&ep).unwrap())
}

fn from_k256(p: &ProjectivePoint) -> AffinePoint {
    let a = p.to_affine();
    let ep = a.to_encoded_point(false);
    AffinePoint {
        x: BigUint::from_bytes_be(ep.x().unwrap()),
        y: BigUint::from_bytes_be(ep.y().unwrap()),
    }
}

fn generator() -> AffinePoint {
    from_k256(&ProjectivePoint::GENERATOR)
}

/// Independent point-scalar-mul via the `Fp` reference (MSB-first).
fn ref_scalar_mul(k: &BigUint, pt: &AffinePoint) -> AffinePoint {
    let bits = k.bits();
    let mut acc = pt.clone();
    for i in (0..bits - 1).rev() {
        acc = point_double(&acc);
        if k.bit(i) {
            acc = point_add(&acc, pt);
        }
    }
    acc
}

/// Independent lincomb2 via the `Fp` reference.
fn ref_lincomb2(u1: &BigUint, p1: &AffinePoint, u2: &BigUint, p2: &AffinePoint) -> AffinePoint {
    let a = ref_scalar_mul(u1, p1);
    let b = ref_scalar_mul(u2, p2);
    if a.x == b.x {
        assert_eq!(a.y, b.y, "reference lincomb2 hit infinity");
        point_double(&a)
    } else {
        point_add(&a, &b)
    }
}

fn k256_lincomb2(u1: &BigUint, p1: &AffinePoint, u2: &BigUint, p2: &AffinePoint) -> AffinePoint {
    let s1 = Scalar::from_repr(be32(u1).into()).unwrap();
    let s2 = Scalar::from_repr(be32(u2).into()).unwrap();
    from_k256(&(to_k256(p1) * s1 + to_k256(p2) * s2))
}

fn pt(x: &[u8; 32], y: &[u8; 32]) -> AffinePoint {
    AffinePoint {
        x: BigUint::from_bytes_le(x),
        y: BigUint::from_bytes_le(y),
    }
}

/// Deterministic pseudo-random scalar in `[1, N)` from a seed.
fn scalar(seed: u64) -> BigUint {
    // splitmix64-ish expansion to 256 bits, then reduce into [1, N).
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut bytes = [0u8; 32];
    for chunk in bytes.chunks_mut(8) {
        s ^= s >> 30;
        s = s.wrapping_mul(0xBF58476D1CE4E5B9);
        s ^= s >> 27;
        s = s.wrapping_mul(0x94D049BB133111EB);
        s ^= s >> 31;
        chunk.copy_from_slice(&s.to_le_bytes());
    }
    BigUint::from_bytes_le(&bytes) % (n() - 1u32) + 1u32
}

fn le32(v: &BigUint) -> [u8; 32] {
    crate::to_le_32(v)
}

/// The full validation battery for one input tuple. Returns the row count.
fn validate(u1: &BigUint, p1: &AffinePoint, u2: &BigUint, p2: &AffinePoint) -> usize {
    let w = lincomb2_witness(&le32(u1), &le32(u2), p1, p2).expect("witness");

    // 1. Q correctness — two independent references.
    let q = pt(&w.x_q, &w.y_q);
    let q_ref = ref_lincomb2(u1, p1, u2, p2);
    assert_eq!(q, q_ref, "Q != Fp-reference lincomb2");
    assert_eq!(q, k256_lincomb2(u1, p1, u2, p2), "Q != k256 lincomb2");

    // canonicalization witnesses must correspond to values < p / < N.
    assert!(q.x < p() && q.y < p(), "Q not canonical");
    assert!(pt(&w.x_p2, &w.y_p2).y < p(), "yP2 not canonical");

    // 2. every row re-satisfies its relation and slope.
    let mut n_dbl = 0usize;
    let mut n_add = 0usize;
    for js in &w.steps {
        let s = &js.step;
        let a = pt(&s.x_a, &s.y_a);
        let r = pt(&s.x_r, &s.y_r);
        let lambda = BigUint::from_bytes_le(&s.lambda);
        match js.sel {
            JointSel::Double => {
                assert_eq!(s.op, 0);
                assert_eq!(point_double(&a), r, "double row wrong result");
                assert_eq!(step_lambda(&a, &a, 0), lambda, "double row wrong slope");
                n_dbl += 1;
            }
            _ => {
                assert_eq!(s.op, 1);
                let addend = pt(&s.x_g, &s.y_g);
                assert_eq!(point_add(&a, &addend), r, "add row wrong result");
                assert_eq!(step_lambda(&a, &addend, 1), lambda, "add row wrong slope");
                if matches!(js.sel, JointSel::AddP1 | JointSel::AddP2 | JointSel::AddP12) {
                    n_add += 1;
                }
            }
        }
    }

    // 3. blind cancels: 2^len·T₀ recorded, and pre-correction acc = Q + 2^len·T₀.
    let tpow = pt(&w.x_t0_pow, &w.y_t0_pow);
    // recompute 2^len·T₀ independently from T₀.
    let mut t = t0();
    for _ in 0..w.len {
        t = point_double(&t);
    }
    assert_eq!(t, tpow, "recorded 2^len·T0 wrong");
    // the correction row's incoming accumulator must equal Q + 2^len·T₀.
    let corr = w
        .steps
        .iter()
        .find(|j| matches!(j.sel, JointSel::Correction))
        .unwrap();
    let acc_before = pt(&corr.step.x_a, &corr.step.y_a);
    assert_eq!(acc_before, point_add(&q, &t), "acc_before != Q + 2^len·T0");

    // rows = P12-precompute + doublings + adds + correction.
    assert_eq!(n_dbl, w.len as usize, "doubling count != len");
    2 + n_dbl + n_add
}

// ---- tests ----

#[test]
fn t0_is_on_curve_and_pinned() {
    let t = t0();
    assert!(t.x < p() && t.y < p());
    let lhs = (&t.y * &t.y) % p();
    let rhs = (&t.x * &t.x % p() * &t.x + 7u32) % p();
    assert_eq!(lhs, rhs, "T0 not on curve");
    assert!(&t.y % 2u32 == BigUint::from(0u32), "T0.y not even");
}

#[test]
fn lincomb2_random_matches_references() {
    let g = generator();
    let mut rows_sum = 0usize;
    let mut rows_max = 0usize;
    const CASES: usize = 512;
    for i in 0..CASES {
        let u1 = scalar(2 * i as u64 + 1);
        let u2 = scalar(2 * i as u64 + 2);
        let p1 = if i % 3 == 0 {
            g.clone()
        } else {
            ref_scalar_mul(&scalar(1000 + i as u64), &g)
        };
        let p2 = ref_scalar_mul(&scalar(5000 + i as u64), &g);
        let rows = validate(&u1, &p1, &u2, &p2);
        rows_sum += rows;
        rows_max = rows_max.max(rows);
    }
    println!(
        "lincomb2 rows/ecrecover: mean {:.1}, max {} over {} cases",
        rows_sum as f64 / CASES as f64,
        rows_max,
        CASES
    );
}

#[test]
fn lincomb2_edge_scalars() {
    let g = generator();
    let r = ref_scalar_mul(&scalar(42), &g);
    let edges = [
        (BigUint::from(1u32), BigUint::from(1u32)),
        (BigUint::from(1u32), BigUint::from(2u32)),
        (BigUint::from(3u32), BigUint::from(5u32)),
        (
            BigUint::from(1u32) << 255u32,
            (BigUint::from(1u32) << 255u32) - 1u32,
        ),
        (n() - 1u32, n() - 1u32),
        (n() - 1u32, BigUint::from(1u32)),
        (
            (BigUint::from(1u32) << 200u32) - 1u32,
            BigUint::from(1u32) << 199u32,
        ),
    ];
    for (u1, u2) in edges {
        // P1 = G, P2 = a random point (R-shaped); skip if Q hits infinity.
        match lincomb2_witness(&crate::to_le_32(&u1), &crate::to_le_32(&u2), &g, &r) {
            Ok(_) => {
                let _ = validate(&u1, &g, &u2, &r);
            }
            Err(Lincomb2Error::ResultInfinity) => { /* legitimate infinity */ }
            Err(e) => panic!("unexpected error {e:?} for u1={u1}, u2={u2}"),
        }
    }
}

#[test]
fn lincomb2_rejects_degenerate_sum() {
    let g = generator();
    let neg_g = AffinePoint {
        x: g.x.clone(),
        y: (p() - &g.y) % p(),
    };
    // P1 = G, P2 = -G ⇒ P1 = -P2 ⇒ SumDegenerate.
    let err = lincomb2_witness(
        &crate::to_le_32(&scalar(1)),
        &crate::to_le_32(&scalar(2)),
        &g,
        &neg_g,
    );
    assert_eq!(err.unwrap_err(), Lincomb2Error::SumDegenerate);
    // P1 = P2 = G ⇒ equal x ⇒ SumDegenerate.
    let err2 = lincomb2_witness(
        &crate::to_le_32(&scalar(1)),
        &crate::to_le_32(&scalar(2)),
        &g,
        &g,
    );
    assert_eq!(err2.unwrap_err(), Lincomb2Error::SumDegenerate);
}

#[test]
fn lincomb2_rejects_bad_scalars() {
    let g = generator();
    let r = ref_scalar_mul(&scalar(7), &g);
    let zero = [0u8; 32];
    assert_eq!(
        lincomb2_witness(&zero, &crate::to_le_32(&scalar(1)), &g, &r).unwrap_err(),
        Lincomb2Error::ScalarIsZero
    );
    assert_eq!(
        lincomb2_witness(&crate::to_le_32(&n()), &crate::to_le_32(&scalar(1)), &g, &r).unwrap_err(),
        Lincomb2Error::ScalarOutOfRange
    );
}

#[test]
fn t0_derivation_matches() {
    use sha2::{Digest, Sha256};
    let tag = b"lambdavm/ecsm/lincomb2/T0/v1";
    let mut counter: u32 = 0;
    let (x, y) = loop {
        let mut h = Sha256::new();
        h.update(tag);
        h.update(counter.to_be_bytes());
        let digest = h.finalize();
        let x = BigUint::from_bytes_be(&digest);
        if x < p() {
            // even y with y² = x³ + 7, if x is on the curve.
            let rhs = (&x * &x % p() * &x + 7u32) % p();
            // sqrt via p ≡ 3 mod 4
            let exp = (p() + 1u32) / 4u32;
            let root = rhs.modpow(&exp, &p());
            if (&root * &root) % p() == rhs {
                let y = if &root % 2u32 == BigUint::from(0u32) {
                    root.clone()
                } else {
                    p() - &root
                };
                break (x, y);
            }
        }
        counter += 1;
    };
    assert_eq!(counter, 1, "T0 derivation counter changed");
    let t = t0();
    assert_eq!(t.x, x, "T0.x derivation mismatch");
    assert_eq!(t.y, y, "T0.y derivation mismatch");
}

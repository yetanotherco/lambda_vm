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
//!  4. the joint schedule is self-consistent: `nb` really is the "an add
//!     follows" bit, and the round recurrence `round − 1 + nb` walks the chain
//!     from `len − 1` down to the drain sentinel `−1` (`check_nb_schedule`).

use num_bigint::BigUint;

use crate::curve::AffinePoint;
use crate::tests::reference::{point_add, point_double, step_lambda};
use crate::witness::{JointSel, Lincomb2Error, Lincomb2Witness, lincomb2_witness, t0};
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

/// The joint-schedule invariants the ECDAS′ chip will constrain, checked at the
/// source.
///
/// `nb` is the only thing pinning a double row's successor round — the doubling
/// and its optional add share a round, so `round − 1 + nb` is not derivable from
/// the other columns. A prover who could stall `round` would insert or drop
/// doublings while every per-row relation still held, so a failure here is a
/// soundness hole in whatever chip consumes this witness, not a cosmetic bug.
fn check_nb_schedule(w: &Lincomb2Witness, u1: &BigUint, u2: &BigUint) {
    let steps = &w.steps;
    assert!(
        steps.len() >= 3,
        "chain must be precompute + at least one round + correction"
    );

    // Row-local facts that hold everywhere, boundary rows included. These are
    // exactly the two constraints the chip needs to define `nb`:
    //     OP · NB = 0                                  (degree 2)
    //     (1 − OP) · (NB − D1 − D2 + D1·D2) = 0        (degree 3)
    // Note the second is op-gated: an ADD row carries its round's digits (it
    // needs them to select the addend) but `nb = 0`, because the row after an
    // add is always the next round's double.
    for (i, js) in steps.iter().enumerate() {
        assert!(js.d1 <= 1 && js.d2 <= 1, "row {i}: digits are not bits");
        assert!(js.nb <= 1, "row {i}: nb is not a bit");
        assert!(js.step.op <= 1, "row {i}: op is not a bit");
        // `nb` is mirrored into the EcdasStep slot so a chip may read either.
        assert_eq!(js.nb, js.step.next_op, "row {i}: nb != step.next_op");
        // an add is never followed by an add (today's ECDAS `OP · NEXT_OP = 0`)
        assert_eq!(js.step.op * js.nb, 0, "row {i}: op*nb != 0");
        // on a double, `nb` is the OR of the digits, in the degree-2 form
        if js.step.op == 0 {
            assert_eq!(
                js.nb,
                js.d1 + js.d2 - js.d1 * js.d2,
                "row {i}: double's nb != d1 + d2 - d1*d2"
            );
        }
    }

    // The two rows that sit off the accumulator line carry no digit.
    let first = &steps[0];
    assert_eq!(
        first.sel,
        JointSel::Precompute,
        "row 0 must be the precompute"
    );
    assert_eq!(
        (first.d1, first.d2, first.nb),
        (0, 0, 0),
        "precompute row must be digit-free"
    );
    let last = steps.last().expect("non-empty");
    assert_eq!(
        last.sel,
        JointSel::Correction,
        "last row must be the correction"
    );
    assert_eq!(
        (last.d1, last.d2, last.nb),
        (0, 0, 0),
        "correction row must be digit-free"
    );

    // Walk the main chain: round starts at len-1 and each row steps it by
    // `-1 + nb`, draining at the sentinel -1.
    let main = &steps[1..steps.len() - 1];
    let mut expected_round = w.len as i32 - 1;
    for (i, js) in main.iter().enumerate() {
        assert_eq!(
            js.step.round as i32, expected_round,
            "main row {i}: round drifted from the recurrence"
        );
        let round = expected_round as u64;

        // Both row kinds carry this round's true digits (the double needs them
        // for `nb` and its per-stream Bit sends, the add to pick the addend).
        assert_eq!(
            js.d1,
            u1.bit(round) as u8,
            "row {i} (round {round}): d1 != u1 bit"
        );
        assert_eq!(
            js.d2,
            u2.bit(round) as u8,
            "row {i} (round {round}): d2 != u2 bit"
        );

        match js.sel {
            JointSel::Double => assert_eq!(js.step.op, 0),
            JointSel::AddP1 | JointSel::AddP2 | JointSel::AddP12 => {
                assert_eq!(js.step.op, 1);
                // the addend is a function of the digits
                let want = match (js.d1, js.d2) {
                    (1, 0) => JointSel::AddP1,
                    (0, 1) => JointSel::AddP2,
                    (1, 1) => JointSel::AddP12,
                    _ => panic!("add row at round {round} with a zero joint digit"),
                };
                assert_eq!(js.sel, want, "row {i} (round {round}): sel != f(d1, d2)");
                assert_eq!(js.nb, 0, "an add row must not claim a pending add");
            }
            other => panic!("main row {i} carries boundary selector {other:?}"),
        }

        // `nb == 1` exactly when the next emitted row is this round's add.
        match main.get(i + 1) {
            Some(next) if js.nb == 1 => {
                assert_eq!(
                    next.step.op, 1,
                    "row {i}: nb=1 but the next row is not an add"
                );
                assert_eq!(
                    next.step.round, js.step.round,
                    "row {i}: the add must share the double's round"
                );
            }
            Some(next) => {
                assert_eq!(next.step.op, 0, "row {i}: nb=0 but the next row is an add");
                assert_eq!(
                    next.step.round as i32,
                    js.step.round as i32 - 1,
                    "row {i}: nb=0 must step the round down by one"
                );
            }
            None => assert_eq!(js.nb, 0, "the last main row cannot have a pending add"),
        }

        expected_round = expected_round - 1 + js.nb as i32;
    }
    assert_eq!(
        expected_round, -1,
        "main chain must drain at the sentinel round -1"
    );
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

    // 4. the joint schedule is self-consistent.
    check_nb_schedule(&w, u1, u2);

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

/// The whole emitted schedule for a hand-checkable digit pattern, spelled out
/// row by row. `check_nb_schedule` proves the recurrence is *self*-consistent
/// over the random corpus; this pins what the recurrence actually produces, so
/// a change of convention (say, moving the digits onto the add row only, or
/// decrementing the round on the double) fails here loudly instead of silently
/// re-balancing.
///
/// `u1 = 0b1010`, `u2 = 0b0011`, so `len = 4` and the four rounds carry joint
/// digits `(1,0) (0,0) (1,1) (0,1)` — every addend selector plus one round with
/// no add at all.
#[test]
fn nb_schedule_matches_hand_worked_example() {
    let g = generator();
    let r = ref_scalar_mul(&scalar(42), &g);
    let (u1, u2) = (BigUint::from(0b1010u32), BigUint::from(0b0011u32));

    let w = lincomb2_witness(&le32(&u1), &le32(&u2), &g, &r).expect("witness");
    assert_eq!(w.len, 4, "len = max(bits(u1), bits(u2))");
    check_nb_schedule(&w, &u1, &u2);

    // (sel, round, nb, d1, d2)
    let expected = [
        (JointSel::Precompute, 0u8, 0u8, 0u8, 0u8),
        (JointSel::Double, 3, 1, 1, 0),
        (JointSel::AddP1, 3, 0, 1, 0),
        (JointSel::Double, 2, 0, 0, 0),
        (JointSel::Double, 1, 1, 1, 1),
        (JointSel::AddP12, 1, 0, 1, 1),
        (JointSel::Double, 0, 1, 0, 1),
        (JointSel::AddP2, 0, 0, 0, 1),
        (JointSel::Correction, 0, 0, 0, 0),
    ];
    let got: Vec<_> = w
        .steps
        .iter()
        .map(|js| (js.sel, js.step.round, js.nb, js.d1, js.d2))
        .collect();
    assert_eq!(got, expected.to_vec(), "joint schedule changed");
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

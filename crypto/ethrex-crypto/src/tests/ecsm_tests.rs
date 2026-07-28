//! Tests for the x-only ECSM linear-combination reconstruction
//! (`lincomb2_with_oracle`) against the software `ProjectivePoint::lincomb`,
//! plus the degenerate-configuration fallback guards.

use crate::*;

/// Software stand-in for the affine ECSM precompile: form the curve point `(x, y)` from
/// the caller's actual coordinates and return `(xR, yR)` of `k·(x, y)`. No parity
/// convention — the real ecall receives the full input point too.
fn soft_oracle(x: &FieldElement, y: &FieldElement, k: &Scalar) -> Option<(FieldElement, FieldElement)> {
    let p = point_from_xy(&x.normalize(), &y.normalize())?;
    let prod = (p * k).to_affine();
    let (xr, yr) = affine_xy(&prod)?;
    Some((xr.normalize(), yr.normalize()))
}

fn g_times(n: u64) -> ProjectivePoint {
    ProjectivePoint::GENERATOR * Scalar::from(n)
}

#[test]
fn matches_software_lincomb_on_fixed_inputs() {
    let cases = [
        (g_times(3), 123_456_789u64, g_times(7), 987_654_321u64),
        (g_times(11), 2u64.pow(20) + 5, g_times(2), 42u64),
        (ProjectivePoint::GENERATOR, 7u64, g_times(5), 9u64),
    ];
    for (p1, k1, p2, k2) in cases {
        let (k1, k2) = (Scalar::from(k1), Scalar::from(k2));
        let expected = ProjectivePoint::lincomb(&p1, &k1, &p2, &k2);
        let got = lincomb2_with_oracle(&p1.to_affine(), &k1, &p2.to_affine(), &k2, soft_oracle)
            .expect("non-degenerate inputs must reconstruct");
        assert_eq!(got, expected.to_affine());
    }
}

#[test]
fn matches_software_lincomb_on_recovery_shape() {
    // u1·G + u2·R, generator first, like ECDSA recovery.
    let g = ProjectivePoint::GENERATOR;
    let r = g_times(0x1234);
    let u1 = Scalar::from(0xdead_beefu64);
    let u2 = Scalar::from(0x0bad_f00du64);
    let expected = ProjectivePoint::lincomb(&g, &u1, &r, &u2);
    let got = lincomb2_with_oracle(&g.to_affine(), &u1, &r.to_affine(), &u2, soft_oracle)
        .expect("non-degenerate inputs must reconstruct");
    assert_eq!(got, expected.to_affine());
}

#[test]
fn edge_scalars_fall_back() {
    // AFFINE PoC: only k=0 falls back now. The old x-only path also rejected k=1
    // and k=n−1 (the (k+1)·P query wrapped); the affine oracle makes no such query,
    // so those scalars reconstruct normally.
    let p1 = g_times(3);
    let p2 = g_times(5);
    let ok = Scalar::from(12345u64);
    for bad in [Scalar::ZERO] {
        assert!(lincomb2_with_oracle(&p1.to_affine(), &bad, &p2.to_affine(), &ok, soft_oracle).is_none());
        assert!(lincomb2_with_oracle(&p1.to_affine(), &ok, &p2.to_affine(), &bad, soft_oracle).is_none());
    }
    // k=1 and k=n−1 now reconstruct correctly.
    for good in [Scalar::ONE, -Scalar::ONE] {
        let expected = ProjectivePoint::lincomb(&p1, &good, &p2, &ok);
        let got = lincomb2_with_oracle(&p1.to_affine(), &good, &p2.to_affine(), &ok, soft_oracle)
            .expect("k=1 / k=n−1 are valid for the affine oracle");
        assert_eq!(got, expected.to_affine());
    }
}

#[test]
fn identity_points_fall_back() {
    let p = g_times(3);
    let k = Scalar::from(7u64);
    let id = ProjectivePoint::IDENTITY;
    assert!(lincomb2_with_oracle(&id.to_affine(), &k, &p.to_affine(), &k, soft_oracle).is_none());
    assert!(lincomb2_with_oracle(&p.to_affine(), &k, &id.to_affine(), &k, soft_oracle).is_none());
}

#[test]
fn cancelling_and_doubling_terms_fall_back() {
    let p = g_times(3);
    let k = Scalar::from(7u64);
    // A = B (doubling chord) and A = −B (Q = O): both share x(A) = x(B).
    assert!(lincomb2_with_oracle(&p.to_affine(), &k, &p.to_affine(), &k, soft_oracle).is_none());
    assert!(lincomb2_with_oracle(&p.to_affine(), &k, &(-p).to_affine(), &k, soft_oracle).is_none());
}

#[test]
fn k_half_n_minus_1_reconstructs_correctly() {
    // k = (n-1)/2 was a special case for the old x-only path; with the affine
    // oracle it is an ordinary scalar and must reconstruct correctly.
    let two_inv = Scalar::from(2u64)
        .invert_vartime()
        .expect("2 is invertible mod n");
    let k_half = -Scalar::ONE * two_inv; // (n-1)/2

    let p1 = g_times(5);
    let p2 = g_times(11);
    let k2 = Scalar::from(99999u64);

    let expected = ProjectivePoint::lincomb(&p1, &k_half, &p2, &k2);
    let got = lincomb2_with_oracle(&p1.to_affine(), &k_half, &p2.to_affine(), &k2, soft_oracle)
        .expect("k=(n-1)/2 is not near-edge and must reconstruct correctly");
    assert_eq!(got, expected.to_affine());
}

#[test]
fn cross_point_cancellation_falls_back() {
    // Construct k1, k2, P1 ≠ ±P2 such that k1·P1 = -(k2·P2), so
    // k1·P1 + k2·P2 = O. The shared x-coordinate makes dxq = 0 → None.
    // P1 = 3G, P2 = 7G: k1·3G = -k2·7G → k1 = -k2·7·3^{-1} mod n.
    let p1 = g_times(3);
    let p2 = g_times(7);
    let k2 = Scalar::from(12345u64);
    let three_inv = Scalar::from(3u64)
        .invert_vartime()
        .expect("3 is invertible mod n");
    let k1 = -(k2 * Scalar::from(7u64) * three_inv);
    assert!(
        lincomb2_with_oracle(&p1.to_affine(), &k1, &p2.to_affine(), &k2, soft_oracle).is_none(),
        "cross-point cancellation (P1 ≠ ±P2, result = O) must fall back"
    );
}

#[test]
fn odd_y_base_point_reconstructs_correctly() {
    // Validates the affine oracle's odd-y sign flip: when P1 has odd y the caller
    // must negate the oracle's even-y result, matching ProjectivePoint::lincomb.
    let (p1, _k_gen) = (2u64..200)
        .find_map(|n| {
            let p = g_times(n);
            let (_, y) = affine_xy(&p.to_affine())?;
            if y.normalize().to_bytes()[31] & 1 == 1 {
                Some((p, n))
            } else {
                None
            }
        })
        .expect("at least one of the first 200 multiples of G has odd y");

    let p2 = g_times(13);
    let k1 = Scalar::from(54321u64);
    let k2 = Scalar::from(11111u64);
    let expected = ProjectivePoint::lincomb(&p1, &k1, &p2, &k2);
    let got = lincomb2_with_oracle(&p1.to_affine(), &k1, &p2.to_affine(), &k2, soft_oracle)
        .expect("odd-y base point is non-degenerate and must reconstruct correctly");
    assert_eq!(got, expected.to_affine());
}

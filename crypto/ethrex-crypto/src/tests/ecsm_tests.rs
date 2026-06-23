//! Tests for the x-only ECSM linear-combination reconstruction
//! (`lincomb2_with_oracle`) against the software `ProjectivePoint::lincomb`,
//! plus the degenerate-configuration fallback guards.

use crate::*;

/// secp256k1 curve constant `b = 7`.
fn curve_b() -> FieldElement {
    let mut bytes = [0u8; 32];
    bytes[31] = 7;
    FieldElement::from_bytes(&bytes.into()).unwrap()
}

/// Software stand-in for the ECSM precompile: lift `x` to a curve point and
/// return `x(k·P)` (parity-invariant, like the real ecall).
fn soft_oracle(x: &FieldElement, k: &Scalar) -> Option<FieldElement> {
    let xn = x.normalize();
    let y2 = (xn.square() * xn + curve_b()).normalize();
    let y = Option::<FieldElement>::from(y2.sqrt())?;
    let p = point_from_xy(&xn, &y.normalize())?;
    let prod = (p * k).to_affine();
    Some(affine_xy(&prod)?.0)
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
        let got = lincomb2_with_oracle(&p1, &k1, &p2, &k2, soft_oracle)
            .expect("non-degenerate inputs must reconstruct");
        assert_eq!(got.to_affine(), expected.to_affine());
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
    let got = lincomb2_with_oracle(&g, &u1, &r, &u2, soft_oracle)
        .expect("non-degenerate inputs must reconstruct");
    assert_eq!(got.to_affine(), expected.to_affine());
}

#[test]
fn edge_scalars_fall_back() {
    let p1 = g_times(3);
    let p2 = g_times(5);
    let ok = Scalar::from(12345u64);
    for bad in [Scalar::ZERO, Scalar::ONE, -Scalar::ONE] {
        assert!(lincomb2_with_oracle(&p1, &bad, &p2, &ok, soft_oracle).is_none());
        assert!(lincomb2_with_oracle(&p1, &ok, &p2, &bad, soft_oracle).is_none());
    }
}

#[test]
fn identity_points_fall_back() {
    let p = g_times(3);
    let k = Scalar::from(7u64);
    let id = ProjectivePoint::IDENTITY;
    assert!(lincomb2_with_oracle(&id, &k, &p, &k, soft_oracle).is_none());
    assert!(lincomb2_with_oracle(&p, &k, &id, &k, soft_oracle).is_none());
}

#[test]
fn cancelling_and_doubling_terms_fall_back() {
    let p = g_times(3);
    let k = Scalar::from(7u64);
    // A = B (doubling chord) and A = −B (Q = O): both share x(A) = x(B).
    assert!(lincomb2_with_oracle(&p, &k, &p, &k, soft_oracle).is_none());
    assert!(lincomb2_with_oracle(&p, &k, &(-p), &k, soft_oracle).is_none());
}

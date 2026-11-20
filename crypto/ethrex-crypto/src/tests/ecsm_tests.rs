//! Tests for the ECSM linear-combination reconstruction
//! (`lincomb2_with_oracle`) against the software `ProjectivePoint::lincomb`,
//! plus the root fix-up and the degenerate-configuration fallback guards.

use crate::*;

/// secp256k1 curve constant `b = 7`.
fn curve_b() -> FieldElement {
    let mut bytes = [0u8; 32];
    bytes[31] = 7;
    FieldElement::from_bytes(&bytes.into()).unwrap()
}

fn is_odd(y: &FieldElement) -> bool {
    y.normalize().to_bytes()[31] & 1 == 1
}

/// Software stand-in for the ECSM precompile, parameterised by which root of `x` the chip
/// witnesses. The real chip is free to pick either (the AIR binds only `yG² ≡ xG³ + b`), so
/// both settings are legal traces and the caller must handle them identically.
/// Returns `(x(k·P̂), y(k·P̂), ŷ)` with `P̂ = (x, ŷ)`.
fn soft_oracle_with_root(
    x: &FieldElement,
    k: &Scalar,
    want_odd_root: bool,
) -> Option<(FieldElement, FieldElement, FieldElement)> {
    let xn = x.normalize();
    let y2 = (xn.square() * xn + curve_b()).normalize();
    let y = Option::<FieldElement>::from(y2.sqrt())?.normalize();
    let yg = if is_odd(&y) == want_odd_root {
        y
    } else {
        (-y).normalize()
    };
    let p = point_from_xy(&xn, &yg)?;
    let prod = (p * k).to_affine();
    let (xr, yr) = affine_xy(&prod)?;
    Some((xr, yr, yg))
}

/// The canonical (even-root) lift, matching what `ecsm::recover_y_canonical` produces.
fn soft_oracle(x: &FieldElement, k: &Scalar) -> Option<(FieldElement, FieldElement, FieldElement)> {
    soft_oracle_with_root(x, k, false)
}

/// The other legal choice: the odd root. `k·P̂ = −(k·P)` here, so the caller's sign fix-up
/// is what keeps the answer right.
fn soft_oracle_odd(
    x: &FieldElement,
    k: &Scalar,
) -> Option<(FieldElement, FieldElement, FieldElement)> {
    soft_oracle_with_root(x, k, true)
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

/// The property the echoed root buys: whichever root the chip picks, the reconstruction is
/// the same point. With the odd root every `k·P̂` comes back negated, and only the caller's
/// fix-up puts it right — so an unfixed implementation fails this and passes the one above.
#[test]
fn either_witnessed_root_gives_the_same_result() {
    let cases = [
        (g_times(3), 123_456_789u64, g_times(7), 987_654_321u64),
        (
            ProjectivePoint::GENERATOR,
            0xdead_beefu64,
            g_times(0x1234),
            0x0bad_f00du64,
        ),
        (g_times(11), 2u64.pow(20) + 5, g_times(2), 42u64),
    ];
    for (p1, k1, p2, k2) in cases {
        let (k1, k2) = (Scalar::from(k1), Scalar::from(k2));
        let expected = ProjectivePoint::lincomb(&p1, &k1, &p2, &k2).to_affine();
        let even = lincomb2_with_oracle(&p1.to_affine(), &k1, &p2.to_affine(), &k2, soft_oracle)
            .expect("even-root oracle must reconstruct");
        let odd = lincomb2_with_oracle(&p1.to_affine(), &k1, &p2.to_affine(), &k2, soft_oracle_odd)
            .expect("odd-root oracle must reconstruct");
        assert_eq!(even, expected);
        assert_eq!(
            odd, expected,
            "the root fix-up must absorb the chip's choice"
        );
    }
}

/// `ŷ` that is neither `y` nor `−y` means the oracle did not multiply the caller's point.
/// The caller must decline rather than use the result.
#[test]
fn foreign_root_falls_back() {
    let bogus = |x: &FieldElement, k: &Scalar| {
        let (xr, yr, _) = soft_oracle(x, k)?;
        // A valid field element, but not a root of this x.
        let mut bytes = [0u8; 32];
        bytes[31] = 9;
        Some((xr, yr, FieldElement::from_bytes(&bytes.into()).unwrap()))
    };
    let p1 = g_times(3);
    let p2 = g_times(7);
    let k = Scalar::from(12345u64);
    assert!(
        lincomb2_with_oracle(&p1.to_affine(), &k, &p2.to_affine(), &k, bogus).is_none(),
        "a yG that is neither root must be rejected"
    );
}

/// `k = 1` and `k = N−1` were degenerate for the x-only predecessor (which needed a second
/// `(k+1)·P` query); with `y` returned they are ordinary scalars.
#[test]
fn former_edge_scalars_now_reconstruct() {
    let p1 = g_times(3);
    let p2 = g_times(7);
    let ok = Scalar::from(12345u64);
    for k in [Scalar::ONE, -Scalar::ONE] {
        for (a, ka, b, kb) in [(p1, k, p2, ok), (p1, ok, p2, k)] {
            let expected = ProjectivePoint::lincomb(&a, &ka, &b, &kb);
            let got = lincomb2_with_oracle(&a.to_affine(), &ka, &b.to_affine(), &kb, soft_oracle)
                .expect("k = 1 / N−1 are ordinary scalars now");
            assert_eq!(got, expected.to_affine());
        }
    }
}

#[test]
fn zero_scalars_fall_back() {
    let p1 = g_times(3);
    let p2 = g_times(5);
    let ok = Scalar::from(12345u64);
    assert!(lincomb2_with_oracle(
        &p1.to_affine(),
        &Scalar::ZERO,
        &p2.to_affine(),
        &ok,
        soft_oracle
    )
    .is_none());
    assert!(lincomb2_with_oracle(
        &p1.to_affine(),
        &ok,
        &p2.to_affine(),
        &Scalar::ZERO,
        soft_oracle
    )
    .is_none());
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
    // k = (n-1)/2 satisfies k·P = -(k+1)·P for any P. It broke nothing before and it
    // breaks nothing now; kept as a regression pin on a scalar with structure.
    let two_inv = Scalar::from(2u64)
        .invert_vartime()
        .expect("2 is invertible mod n");
    let k_half = -Scalar::ONE * two_inv; // (n-1)/2

    let p1 = g_times(5);
    let p2 = g_times(11);
    let k2 = Scalar::from(99999u64);

    let expected = ProjectivePoint::lincomb(&p1, &k_half, &p2, &k2);
    let got = lincomb2_with_oracle(&p1.to_affine(), &k_half, &p2.to_affine(), &k2, soft_oracle)
        .expect("k=(n-1)/2 must reconstruct correctly");
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
    // A base point with odd y exercises the fix-up from the other side: the caller's own y
    // is the odd root, so the canonical-lift oracle is the one that comes back negated.
    let p1 = (2u64..200)
        .find_map(|n| {
            let p = g_times(n);
            let (_, y) = affine_xy(&p.to_affine())?;
            is_odd(&y).then_some(p)
        })
        .expect("at least one of the first 200 multiples of G has odd y");

    let p2 = g_times(13);
    let k1 = Scalar::from(54321u64);
    let k2 = Scalar::from(11111u64);
    let expected = ProjectivePoint::lincomb(&p1, &k1, &p2, &k2).to_affine();
    for oracle in [
        soft_oracle as fn(&FieldElement, &Scalar) -> _,
        soft_oracle_odd as fn(&FieldElement, &Scalar) -> _,
    ] {
        let got = lincomb2_with_oracle(&p1.to_affine(), &k1, &p2.to_affine(), &k2, oracle)
            .expect("odd-y base point is non-degenerate and must reconstruct correctly");
        assert_eq!(got, expected);
    }
}

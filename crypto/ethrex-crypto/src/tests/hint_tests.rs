//! Host tests for the untrusted-hint verify-then-fallback paths (`scalar_inv`,
//! `field_inv`, `decompress_r`).
//!
//! The guest asks the (untrusted, prover-chosen) `hint` ecall for a modular
//! inverse / square root, then verifies it in-circuit. These tests inject the
//! oracle directly — an *honest* oracle (matching the executor's `compute_hint`)
//! and a *lying* one — and assert the software fallback makes the result identical
//! either way. That is the property C1 turns on: a bad hint can only make the guest
//! do more work, never change its accept/reject outcome. On the guest this code is
//! `cfg(target_arch = "riscv64")`; the `test` gate on `*_with_oracle` is what lets
//! CI compile and exercise it on the host.

use crate::*;

/// A `[u8; 32]` big-endian field element from a small integer.
fn fe_from_u64(k: u64) -> FieldElement {
    let mut be = [0u8; 32];
    be[24..32].copy_from_slice(&k.to_be_bytes());
    Option::<FieldElement>::from(FieldElement::from_bytes(&be.into())).expect("k < p")
}

/// Honest scalar-inverse oracle (BE in/out, mod n) — mirrors the executor's
/// `compute_hint(HINT_SCALAR_INV, ..)`: the inverse if it exists, else zeros.
fn honest_scalar_inv(x_be: &[u8; 32]) -> [u8; 32] {
    let x = Option::<Scalar>::from(Scalar::from_repr((*x_be).into())).expect("canonical input");
    match Option::<Scalar>::from(x.invert()) {
        Some(inv) => inv.to_bytes().into(),
        None => [0u8; 32],
    }
}

/// Honest base-field sqrt oracle (BE in/out, mod p) — mirrors
/// `compute_hint(HINT_FIELD_SQRT, ..)`: a root if one exists, else zeros.
fn honest_field_sqrt(rhs_be: &[u8; 32]) -> [u8; 32] {
    let rhs = Option::<FieldElement>::from(FieldElement::from_bytes(&(*rhs_be).into()))
        .expect("canonical");
    match Option::<FieldElement>::from(rhs.sqrt()) {
        Some(y) => y.to_bytes().into(),
        None => [0u8; 32],
    }
}

fn sec1(p: &AffinePoint) -> Vec<u8> {
    p.to_encoded_point(false).as_bytes().to_vec()
}

#[test]
fn scalar_inv_honest_hint_matches_software() {
    for k in [1u64, 2, 3, 7, 1000, 12345, u64::MAX] {
        let x = Scalar::from(k);
        let sw = x.invert_vartime().expect("k != 0 is invertible");
        let got = scalar_inv_with_oracle(&x, honest_scalar_inv).expect("inverse exists");
        assert_eq!(
            got, sw,
            "honest hint must equal the software inverse (k={k})"
        );
    }
}

#[test]
fn scalar_inv_lying_hint_falls_back_to_software() {
    // The prover-chosen hint returns garbage; the result must be unchanged. `x⁻¹`
    // exists (the caller guarantees `r != 0`), so the software fallback is
    // authoritative — a lie cannot turn a recoverable signature into a failure.
    for lie in [[0u8; 32], [0xFFu8; 32]] {
        for k in [1u64, 2, 12345, u64::MAX] {
            let x = Scalar::from(k);
            let sw = x.invert_vartime().unwrap();
            let got = scalar_inv_with_oracle(&x, |_| lie).expect("fallback recomputes");
            assert_eq!(
                got, sw,
                "lying hint must fall back to the software inverse (k={k})"
            );
        }
    }
}

#[test]
fn decompress_r_honest_hint_matches_software() {
    // x-coordinates of real points are guaranteed residues.
    for k in [1u64, 2, 5, 12345] {
        let p = (ProjectivePoint::GENERATOR * Scalar::from(k)).to_affine();
        let (x, y) = affine_xy(&p).unwrap();
        let rb = x.to_bytes();
        let y_is_odd = (y.normalize().to_bytes()[31] & 1) == 1;
        let got = decompress_r_with_oracle(&rb, y_is_odd, honest_field_sqrt)
            .expect("valid residue decompresses");
        assert_eq!(
            sec1(&got),
            sec1(&p),
            "honest hint must recover the point (k={k})"
        );
    }
}

#[test]
fn decompress_r_lying_hint_falls_back_to_software() {
    // A residue x with a garbage sqrt hint must still decompress to the true point.
    for lie in [[0u8; 32], [0xFFu8; 32]] {
        for k in [1u64, 5, 12345] {
            let p = (ProjectivePoint::GENERATOR * Scalar::from(k)).to_affine();
            let (x, y) = affine_xy(&p).unwrap();
            let rb = x.to_bytes();
            let y_is_odd = (y.normalize().to_bytes()[31] & 1) == 1;
            let got = decompress_r_with_oracle(&rb, y_is_odd, |_| lie)
                .expect("software fallback decompresses a residue");
            assert_eq!(
                sec1(&got),
                sec1(&p),
                "lying hint must fall back to software (k={k})"
            );
        }
    }
}

#[test]
fn decompress_r_non_residue_is_none_regardless_of_hint() {
    // Find a small x whose x³+7 has no square root: R is genuinely undecompressable
    // and must be `None`. A lying hint must NOT be able to force a `Some`, and the
    // honest path must NOT spuriously fail — both stem from the same software
    // fallback being the sole authority on rejection.
    let mut seven = [0u8; 32];
    seven[31] = 7;
    let seven = Option::<FieldElement>::from(FieldElement::from_bytes(&seven.into())).unwrap();

    let x = (1u64..10_000)
        .map(fe_from_u64)
        .find(|x| {
            let rhs = (x.square() * *x + seven).normalize();
            Option::<FieldElement>::from(rhs.sqrt()).is_none()
        })
        .expect("some small x has a non-residue x³+7");
    let rb = x.to_bytes();

    assert!(
        decompress_r_with_oracle(&rb, false, honest_field_sqrt).is_none(),
        "a genuine non-residue must decompress to None (honest hint)"
    );
    for lie in [[0u8; 32], [0xFFu8; 32]] {
        assert!(
            decompress_r_with_oracle(&rb, false, |_| lie).is_none(),
            "a lying hint must not force a non-residue to decompress"
        );
    }
}

/// Honest base-field inverse oracle (BE in/out, mod p) — mirrors the executor's
/// `compute_hint(HINT_FIELD_INV, ..)`: the inverse if it exists, else zeros.
fn honest_field_inv(x_be: &[u8; 32]) -> [u8; 32] {
    let x = Option::<FieldElement>::from(FieldElement::from_bytes(&(*x_be).into()))
        .expect("canonical input");
    match Option::<FieldElement>::from(x.invert()) {
        Some(inv) => inv.to_bytes().into(),
        None => [0u8; 32],
    }
}

#[test]
fn field_inv_honest_hint_matches_software() {
    for k in [1u64, 2, 3, 7, 1000, 12345] {
        let x = fe_from_u64(k);
        let sw = Option::<FieldElement>::from(x.invert()).expect("k != 0 is invertible");
        let got = field_inv_with_oracle(&x, honest_field_inv).expect("inverse exists");
        assert_eq!(
            got.normalize().to_bytes(),
            sw.normalize().to_bytes(),
            "honest hint must equal the software inverse (k={k})"
        );
    }
}

#[test]
fn field_inv_lying_hint_falls_back_to_software() {
    // A prover-chosen garbage inverse must not change the result: `x⁻¹` exists for
    // every input the callers pass (guarded non-zero denominators), so the software
    // fallback is authoritative — a lie can only cost work, never steer the outcome.
    for lie in [[0u8; 32], [0xFFu8; 32]] {
        for k in [1u64, 2, 12345] {
            let x = fe_from_u64(k);
            let sw = Option::<FieldElement>::from(x.invert()).unwrap();
            let got = field_inv_with_oracle(&x, |_| lie).expect("fallback recomputes");
            assert_eq!(
                got.normalize().to_bytes(),
                sw.normalize().to_bytes(),
                "lying hint must fall back to the software inverse (k={k})"
            );
        }
    }
}

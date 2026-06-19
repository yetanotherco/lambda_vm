//! Unit tests for the crate's public entry points (relocated from `lib.rs`).

use num_bigint::BigUint;

use crate::{B, EcsmError, n, p, recover_y_canonical, scalar_mul_x, to_le_32};

/// Parses a big-endian hex string into a `BigUint`.
fn be_hex(s: &str) -> BigUint {
    BigUint::parse_bytes(s.as_bytes(), 16).unwrap()
}

// secp256k1 generator G.
const GX_HEX: &str = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
const GY_HEX: &str = "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";

fn gx() -> BigUint {
    be_hex(GX_HEX)
}

#[test]
fn constants_match_known_secp256k1_values() {
    assert_eq!(
        p(),
        be_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F")
    );
    assert_eq!(
        n(),
        be_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141")
    );
    // p ≡ 3 mod 4 (a known secp256k1 property).
    assert_eq!(&p() % 4u32, BigUint::from(3u8));
}

#[test]
fn generator_is_on_curve_and_y_is_canonical() {
    // Gy ends in 0xB8 (even), so the canonical (even) root is Gy itself.
    let y = recover_y_canonical(&gx()).expect("G is on the curve");
    assert_eq!(y, be_hex(GY_HEX));
    assert!(!y.bit(0), "canonical root must be even");
}

#[test]
fn recover_y_handles_residues_and_non_residues() {
    // Roughly half of all x are non-residues; scan a small range and check both
    // branches deterministically: every recovered y is even and on the curve, and at
    // least one x has no valid y (the `None` path).
    let mut saw_none = false;
    let mut saw_some = false;
    for x in 1u32..40 {
        let xb = BigUint::from(x);
        match recover_y_canonical(&xb) {
            Some(y) => {
                saw_some = true;
                assert!(!y.bit(0), "recovered y must be even");
                // y^2 == x^3 + b mod p
                let lhs = (&y * &y) % p();
                let rhs = (&xb * &xb % p() * &xb + BigUint::from(B)) % p();
                assert_eq!(lhs, rhs);
            }
            None => saw_none = true,
        }
    }
    assert!(
        saw_some && saw_none,
        "expected both residues and non-residues in range"
    );
}

#[test]
fn scalar_mul_one_is_identity() {
    let k = to_le_32(&BigUint::from(1u8));
    let xg = to_le_32(&gx());
    assert_eq!(scalar_mul_x(&k, &xg).unwrap(), xg);
}

#[test]
fn scalar_mul_two_matches_known_2g() {
    // x(2G) for secp256k1.
    let expected = be_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5");
    let k = to_le_32(&BigUint::from(2u8));
    let xg = to_le_32(&gx());
    assert_eq!(scalar_mul_x(&k, &xg).unwrap(), to_le_32(&expected));
}

#[test]
fn scalar_mul_three_matches_known_3g() {
    let expected = be_hex("F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9");
    let k = to_le_32(&BigUint::from(3u8));
    let xg = to_le_32(&gx());
    assert_eq!(scalar_mul_x(&k, &xg).unwrap(), to_le_32(&expected));
}

#[test]
fn scalar_mul_n_minus_one_shares_x_with_g() {
    // (N-1)·G = -G, which has the same x-coordinate as G.
    let k = to_le_32(&(n() - BigUint::from(1u8)));
    let xg = to_le_32(&gx());
    assert_eq!(scalar_mul_x(&k, &xg).unwrap(), xg);
}

#[test]
fn rejects_zero_and_out_of_range_scalars() {
    let xg = to_le_32(&gx());
    assert_eq!(
        scalar_mul_x(&to_le_32(&BigUint::from(0u8)), &xg),
        Err(EcsmError::ScalarIsZero)
    );
    assert_eq!(
        scalar_mul_x(&to_le_32(&n()), &xg),
        Err(EcsmError::ScalarOutOfRange)
    );
}

#[test]
fn rejects_non_canonical_xg() {
    // xG = p and xG = p + 1 (the alias of x = 1) must be rejected, not
    // silently reduced: with k = 1 the input bytes would be echoed back as
    // xR, which the prover's xR < p range check cannot prove.
    let k = to_le_32(&BigUint::from(1u8));
    for delta in [0u8, 1] {
        assert_eq!(
            scalar_mul_x(&k, &to_le_32(&(p() + BigUint::from(delta)))),
            Err(EcsmError::CoordinateOutOfRange),
            "xG = p + {delta} must be rejected"
        );
    }
    // p − 1 is below the bound, so it must NOT hit the canonicity check
    // (it is not on the curve, which is a different error).
    assert_eq!(
        scalar_mul_x(&k, &to_le_32(&(p() - BigUint::from(1u8)))),
        Err(EcsmError::NotOnCurve)
    );
}

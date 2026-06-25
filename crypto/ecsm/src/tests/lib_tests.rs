//! Unit tests for the crate's public entry points (relocated from `lib.rs`).

use crypto_bigint::{Encoding, NonZero, U256};

use crate::{B, EcsmError, n, p, recover_y_canonical, scalar_mul_x};

// secp256k1 generator G.
const GX_HEX: &str = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
const GY_HEX: &str = "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";

fn gx() -> U256 {
    U256::from_be_hex(GX_HEX)
}

#[test]
fn constants_match_known_secp256k1_values() {
    assert_eq!(
        p(),
        U256::from_be_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F")
    );
    assert_eq!(
        n(),
        U256::from_be_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141")
    );
    // p ≡ 3 mod 4 (a known secp256k1 property).
    let four = NonZero::new(U256::from(4u32)).expect("4 != 0");
    assert_eq!(p().div_rem(&four).1, U256::from(3u32));
}

#[test]
fn generator_is_on_curve_and_y_is_canonical() {
    // Gy ends in 0xB8 (even), so the canonical (even) root is Gy itself.
    let y = recover_y_canonical(&gx()).expect("G is on the curve");
    assert_eq!(y, U256::from_be_hex(GY_HEX));
    assert!(!y.bit_vartime(0), "canonical root must be even");
}

#[test]
fn recover_y_handles_residues_and_non_residues() {
    // Roughly half of all x are non-residues; scan a small range and check both
    // branches deterministically: every recovered y is even and on the curve, and at
    // least one x has no valid y (the `None` path).
    let mut saw_none = false;
    let mut saw_some = false;
    for x in 1u32..40 {
        let xb = U256::from(x);
        match recover_y_canonical(&xb) {
            Some(y) => {
                saw_some = true;
                assert!(!y.bit_vartime(0), "recovered y must be even");
                // y^2 == x^3 + b mod p  (using U512 for the products)
                use crypto_bigint::U512;
                let (yy_lo, yy_hi) = y.mul_wide(&y);
                let yy = yy_hi.concat(&yy_lo);
                let mut p_le64 = [0u8; 64];
                p_le64[..32].copy_from_slice(&p().to_le_bytes());
                let p512 = NonZero::new(U512::from_le_slice(&p_le64)).expect("p != 0");
                let lhs = yy.div_rem(&p512).1;
                let (xx_lo, xx_hi) = xb.mul_wide(&xb);
                let xx = xx_hi.concat(&xx_lo);
                let x2 = xx.div_rem(&p512).1;
                let (x3_lo, x3_hi) = xb.mul_wide(&U256::from_le_slice(&x2.to_le_bytes()[..32]));
                let x3 = x3_hi.concat(&x3_lo);
                let rhs = x3.wrapping_add(&U512::from(B)).div_rem(&p512).1;
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
    let k = U256::ONE.to_le_bytes();
    let xg = gx().to_le_bytes();
    assert_eq!(scalar_mul_x(&k, &xg).expect("1·G is valid"), xg);
}

#[test]
fn scalar_mul_two_matches_known_2g() {
    let expected =
        U256::from_be_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5");
    let k = U256::from(2u32).to_le_bytes();
    let xg = gx().to_le_bytes();
    assert_eq!(
        scalar_mul_x(&k, &xg).expect("2·G is valid"),
        expected.to_le_bytes()
    );
}

#[test]
fn scalar_mul_three_matches_known_3g() {
    let expected =
        U256::from_be_hex("F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9");
    let k = U256::from(3u32).to_le_bytes();
    let xg = gx().to_le_bytes();
    assert_eq!(
        scalar_mul_x(&k, &xg).expect("3·G is valid"),
        expected.to_le_bytes()
    );
}

#[test]
fn scalar_mul_n_minus_one_shares_x_with_g() {
    // (N-1)·G = -G, which has the same x-coordinate as G.
    let k = n().wrapping_sub(&U256::ONE).to_le_bytes();
    let xg = gx().to_le_bytes();
    assert_eq!(scalar_mul_x(&k, &xg).expect("(N-1)·G is valid"), xg);
}

#[test]
fn rejects_zero_and_out_of_range_scalars() {
    let xg = gx().to_le_bytes();
    assert_eq!(
        scalar_mul_x(&U256::ZERO.to_le_bytes(), &xg),
        Err(EcsmError::ScalarIsZero)
    );
    assert_eq!(
        scalar_mul_x(&n().to_le_bytes(), &xg),
        Err(EcsmError::ScalarOutOfRange)
    );
}

#[test]
fn rejects_non_canonical_xg() {
    // xG = p and xG = p + 1 (the alias of x = 1) must be rejected, not
    // silently reduced: with k = 1 the input bytes would be echoed back as
    // xR, which the prover's xR < p range check cannot prove.
    let k = U256::ONE.to_le_bytes();
    for delta in [0u32, 1] {
        assert_eq!(
            scalar_mul_x(&k, &p().wrapping_add(&U256::from(delta)).to_le_bytes()),
            Err(EcsmError::CoordinateOutOfRange),
            "xG = p + {delta} must be rejected"
        );
    }
    // p − 1 is below the bound, so it must NOT hit the canonicity check
    // (it is not on the curve, which is a different error).
    assert_eq!(
        scalar_mul_x(&k, &p().wrapping_sub(&U256::ONE).to_le_bytes()),
        Err(EcsmError::NotOnCurve)
    );
}

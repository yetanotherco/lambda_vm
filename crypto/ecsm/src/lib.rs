//! Reference secp256k1 scalar multiplication and ECSM-accelerator witness generation.
//!
//! This crate is shared by the executor (which needs `k·G`'s x-coordinate to write back
//! to guest memory) and the prover (which replays the full double-and-add sequence to
//! fill the ECSM / ECDAS / EC_SCALAR trace witnesses). Keeping a single implementation
//! guarantees the two never diverge — in particular they pick the same `yG` square root.
//!
//! Curve point operations (decompression, doubling, addition) are delegated to the audited
//! RustCrypto `k256` crate; the limb arithmetic for witness generation uses `num-bigint`
//! (already a workspace dependency). All of this runs once per `ECALL`, so it is not
//! performance critical.
//!
//! Curve: secp256k1, `y^2 = x^3 + 7 mod p`, `p = 2^256 - 2^32 - 977`, order `N`.

pub mod curve;
pub mod field;
pub mod witness;

use num_bigint::BigUint;

pub use curve::{AffinePoint, recover_y_canonical, replay_double_and_add};
pub use witness::{EcdasStep, EcsmWitness, compute_witness};

/// secp256k1 curve coefficient `b`.
pub const B: u64 = 7;

/// Prime field modulus `p = 2^256 - 2^32 - 977`, little-endian bytes.
pub const P_BYTES: [u8; 32] = [
    0x2F, 0xFC, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

/// Curve group order `N`, little-endian bytes.
pub const N_BYTES: [u8; 32] = [
    0x41, 0x41, 0x36, 0xD0, 0x8C, 0x5E, 0xD2, 0xBF, 0x3B, 0xA0, 0x48, 0xAF, 0xE6, 0xDC, 0xAE, 0xBA,
    0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

/// The prime field modulus `p` as a `BigUint`.
pub fn p() -> BigUint {
    BigUint::from_bytes_le(&P_BYTES)
}

/// The curve order `N` as a `BigUint`.
pub fn n() -> BigUint {
    BigUint::from_bytes_le(&N_BYTES)
}

/// Errors that prevent a sound ECSM witness from existing for the given inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EcsmError {
    /// `k == 0`: `0·G` is the point at infinity, which the accelerator cannot represent.
    ScalarIsZero,
    /// `k >= N`: outside the valid scalar range `[1, N)`.
    ScalarOutOfRange,
    /// `x^3 + b` is not a quadratic residue, so `xG` is not a valid x-coordinate.
    NotOnCurve,
    /// `xG >= p`: not a canonical field element. Reducing it silently would
    /// diverge from the prover, whose `xR < p` range check makes a non-canonical
    /// input unprovable (with `k = 1` the input is echoed back as `xR`).
    CoordinateOutOfRange,
}

impl core::fmt::Display for EcsmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EcsmError::ScalarIsZero => write!(f, "ECSM scalar k must be non-zero"),
            EcsmError::ScalarOutOfRange => write!(f, "ECSM scalar k must be < N"),
            EcsmError::NotOnCurve => write!(f, "ECSM xG is not a valid curve x-coordinate"),
            EcsmError::CoordinateOutOfRange => write!(f, "ECSM xG must be < p"),
        }
    }
}

impl std::error::Error for EcsmError {}

/// Converts a `BigUint` to 32 little-endian bytes (zero-padded / truncated to 32).
pub fn to_le_32(v: &BigUint) -> [u8; 32] {
    let mut bytes = v.to_bytes_le();
    bytes.resize(32, 0);
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes[..32]);
    out
}

/// Validates the scalar and recovers the generator point from `(xG, k)`.
///
/// Shared front-end for both entry points: checks `0 < k < N`, rebuilds `xG`, and recovers
/// the canonical `yG`.
pub(crate) fn prepare(
    k_le: &[u8; 32],
    xg_le: &[u8; 32],
) -> Result<(BigUint, AffinePoint), EcsmError> {
    let k = BigUint::from_bytes_le(k_le);
    if k == BigUint::from(0u8) {
        return Err(EcsmError::ScalarIsZero);
    }
    if k >= n() {
        return Err(EcsmError::ScalarOutOfRange);
    }
    let xg = BigUint::from_bytes_le(xg_le);
    if xg >= p() {
        return Err(EcsmError::CoordinateOutOfRange);
    }
    let yg = recover_y_canonical(&xg).ok_or(EcsmError::NotOnCurve)?;
    Ok((k, AffinePoint { x: xg, y: yg }))
}

/// Computes the x-coordinate of `k·G` over secp256k1, given `k` and `xG` as little-endian
/// 32-byte values. This is the executor's entry point — it writes the returned bytes back
/// to guest memory at `addr_xR`.
pub fn scalar_mul_x(k_le: &[u8; 32], xg_le: &[u8; 32]) -> Result<[u8; 32], EcsmError> {
    let (k, g) = prepare(k_le, xg_le)?;
    let (_steps, result) = replay_double_and_add(&k, &g);
    Ok(to_le_32(&result.x))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

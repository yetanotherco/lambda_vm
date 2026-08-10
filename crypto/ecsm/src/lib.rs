//! Reference secp256k1 scalar multiplication and ECSM-accelerator witness generation.
//!
//! This crate is shared by the executor (which needs `k·G`'s coordinates to write back
//! to guest memory) and the prover (which replays the full double-and-add sequence to
//! fill the ECSM / ECDAS trace witnesses). Both entry points compute the same
//! `k·G` over the audited `k256` curve arithmetic — the executor via `k256`'s scalar
//! multiplication, the prover via a projective double-and-add replay — so the coordinates
//! they write/prove agree.
//!
//! There are two families of entry point, and they differ in exactly one respect — which
//! `yG` they use:
//! - **x-only** (`scalar_mul_x` / [`compute_witness`], via `prepare`): `yG` is recovered as
//!   the canonical *even* lift of `xG`, so the result is independent of the root — only
//!   `xR` is returned, and `k·P` and `k·(-P)` share an x.
//! - **affine** (`scalar_mul_xy_with_y` / [`compute_witness_with_y`], via `prepare_with_y`):
//!   `yG` is the caller's own value, validated on-curve but *not* canonicalized. The result
//!   IS root-dependent — `yR` is the y of the caller's chosen lift — which is the whole
//!   point, since the affine ecall returns `yR` to the guest. In the prover the witnessed
//!   `yG` is pinned to the caller's input buffer by a memory read, so the root is not a
//!   free choice.
//!
//! Curve point operations are delegated to the RustCrypto `k256` crate; witness generation
//! replays the schedule in `k256` projective coordinates and batch-inverts the slope
//! denominators, while `num-bigint` carries the coordinate/limb representation the trace
//! needs. All of this runs once per `ECALL`, so it is not performance critical.
//!
//! Curve: secp256k1, `y^2 = x^3 + 7 mod p`, `p = 2^256 - 2^32 - 977`, order `N`.

pub mod curve;
pub mod witness;

#[cfg(test)]
mod tests;

use num_bigint::BigUint;

pub use curve::{AffinePoint, recover_y_canonical, replay_double_and_add};
pub use witness::{EcdasStep, EcsmWitness, compute_witness, compute_witness_with_y};

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

/// Shift offset `r = 3p`, little-endian bytes.
pub const R_BYTES: [u8; 33] = [
    0x8D, 0xF4, 0xFF, 0xFF, 0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x02,
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
    /// The input point is not on the curve: on the x-only path `x³ + b` is not a quadratic
    /// residue, so `xG` is not a valid x-coordinate; on the affine path the caller's own
    /// `yG` fails `yG² ≡ xG³ + b`.
    NotOnCurve,
    /// A coordinate is `>= p`, so it is not a canonical field element — `xG` on either path,
    /// `yG` on the affine one. Reducing it silently would diverge from the prover, whose
    /// `xR < p` / `yR < p` range checks make a non-canonical input unprovable (with `k = 1`
    /// the x-only input is echoed back as `xR`).
    CoordinateOutOfRange,
}

impl core::fmt::Display for EcsmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EcsmError::ScalarIsZero => write!(f, "ECSM scalar k must be non-zero"),
            EcsmError::ScalarOutOfRange => write!(f, "ECSM scalar k must be < N"),
            EcsmError::NotOnCurve => write!(f, "ECSM input point is not on the curve"),
            EcsmError::CoordinateOutOfRange => write!(f, "ECSM coordinates must be < p"),
        }
    }
}

impl std::error::Error for EcsmError {}

/// Converts a `BigUint` to 32 little-endian bytes (zero-padded / truncated to 32).
pub fn to_le_32(v: &BigUint) -> [u8; 32] {
    debug_assert!(v.bits() <= 256, "to_le_32: value exceeds 256 bits");
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

/// Like [`prepare`] but takes an explicit `yG` (the caller's full input point) instead of
/// lifting `xG` to the canonical even root. Validates `0 < k < N`, `xG < p`, `yG < p`, and
/// that `(xG, yG)` is on the curve (`yG² ≡ xG³ + b mod p`). Used by the affine path so the
/// returned `yR` matches the caller's actual point (no parity convention / guest-side sign
/// flip). `yG`'s value is pinned in the prover by a memory read of the caller's input.
pub(crate) fn prepare_with_y(
    k_le: &[u8; 32],
    xg_le: &[u8; 32],
    yg_le: &[u8; 32],
) -> Result<(BigUint, AffinePoint), EcsmError> {
    let k = BigUint::from_bytes_le(k_le);
    if k == BigUint::from(0u8) {
        return Err(EcsmError::ScalarIsZero);
    }
    if k >= n() {
        return Err(EcsmError::ScalarOutOfRange);
    }
    let p = p();
    let xg = BigUint::from_bytes_le(xg_le);
    let yg = BigUint::from_bytes_le(yg_le);
    if xg >= p || yg >= p {
        return Err(EcsmError::CoordinateOutOfRange);
    }
    // On-curve: yG² ≡ xG³ + b (mod p).
    let lhs = (&yg * &yg) % &p;
    let rhs = (&xg * &xg % &p * &xg + BigUint::from(B)) % &p;
    if lhs != rhs {
        return Err(EcsmError::NotOnCurve);
    }
    Ok((k, AffinePoint { x: xg, y: yg }))
}

/// Affine entry point with an explicit input `yG`: both coordinates of `k·(xG, yG)` as
/// little-endian 32-byte values. The executor writes `xR` then `yR` back (64-byte output).
pub fn scalar_mul_xy_with_y(
    k_le: &[u8; 32],
    xg_le: &[u8; 32],
    yg_le: &[u8; 32],
) -> Result<([u8; 32], [u8; 32]), EcsmError> {
    let (k, g) = prepare_with_y(k_le, xg_le, yg_le)?;
    let r = curve::scalar_mul_affine(&k, &g);
    Ok((to_le_32(&r.x), to_le_32(&r.y)))
}

/// Computes the x-coordinate of `k·G` over secp256k1, given `k` and `xG` as little-endian
/// 32-byte values. This is the executor's entry point — it writes the returned bytes back
/// to guest memory at `addr_xR`.
pub fn scalar_mul_x(k_le: &[u8; 32], xg_le: &[u8; 32]) -> Result<[u8; 32], EcsmError> {
    let (k, g) = prepare(k_le, xg_le)?;
    Ok(to_le_32(&curve::scalar_mul_affine_x(&k, &g)))
}

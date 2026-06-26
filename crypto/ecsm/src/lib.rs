//! Reference secp256k1 scalar multiplication and ECSM-accelerator witness generation.
//!
//! This crate is shared by the executor (which needs `k·G`'s x-coordinate to write back
//! to guest memory) and the prover (which replays the full double-and-add sequence to
//! fill the ECSM / ECDAS / EC_SCALAR trace witnesses). Both entry points compute the same
//! `k·G` over the audited `k256` curve arithmetic — the executor via `k256`'s scalar
//! multiplication, the prover via a projective double-and-add replay — so the x-coordinate
//! they write/prove agrees. It is also independent of the `yG` root: both recover the same
//! canonical `yG` in `prepare`, and `k·P` and `k·(-P)` share an x.
//!
//! Curve point operations are delegated to the RustCrypto `k256` crate; witness generation
//! replays the schedule in `k256` projective coordinates and batch-inverts the slope
//! denominators, while `crypto-bigint` carries the coordinate/limb representation the trace
//! needs. All of this runs once per `ECALL`, so it is not performance critical.
//!
//! Curve: secp256k1, `y^2 = x^3 + 7 mod p`, `p = 2^256 - 2^32 - 977`, order `N`.

pub mod curve;
pub mod witness;

#[cfg(test)]
mod tests;

use crypto_bigint::U256;

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

/// Shift offset `r = 3p`, little-endian bytes.
pub const R_BYTES: [u8; 33] = [
    0x8D, 0xF4, 0xFF, 0xFF, 0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x02,
];

/// The prime field modulus `p` as a `U256`.
pub const P: U256 =
    U256::from_be_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

/// The curve group order `N` as a `U256`.
pub const N: U256 =
    U256::from_be_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

/// The prime field modulus `p` as a `U256`.
pub const fn p() -> U256 {
    P
}

/// The curve order `N` as a `U256`.
pub const fn n() -> U256 {
    N
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

/// Validates the scalar and recovers the generator point from `(xG, k)`.
///
/// Shared front-end for both entry points: checks `0 < k < N`, rebuilds `xG`, and recovers
/// the canonical `yG`.
pub(crate) fn prepare(
    k_le: &[u8; 32],
    xg_le: &[u8; 32],
) -> Result<(U256, AffinePoint), EcsmError> {
    let k = U256::from_le_slice(k_le);
    if k == U256::ZERO {
        return Err(EcsmError::ScalarIsZero);
    }
    if k >= n() {
        return Err(EcsmError::ScalarOutOfRange);
    }
    let xg = U256::from_le_slice(xg_le);
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
    Ok(curve::scalar_mul_affine_x(&k, &g).to_le_bytes().into())
}

//! secp256k1 curve arithmetic and the chip-faithful double-and-add replay.
//!
//! The curve point operations (decompression, doubling, addition) are delegated to the
//! RustCrypto `k256` crate; this module adapts them to the `BigUint` representation used by
//! the witness builder and drives the double-and-add schedule. The curve is
//! `y^2 = x^3 + 7 mod p` (short Weierstrass with `a = 0`). The point
//! at infinity never appears: the ECSM/ECDAS design guarantees it cannot occur for
//! `k in [1, N)` (see `ecsm.typ` "Point at infinity" / ECDAS soundness argument), so every
//! affine point below is well defined.

use k256::elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
use k256::{AffinePoint as K256Affine, EncodedPoint, ProjectivePoint};
use num_bigint::BigUint;

/// An affine curve point. Never the point at infinity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffinePoint {
    pub x: BigUint,
    pub y: BigUint,
}

/// `v` as 32 big-endian bytes (k256's `FieldBytes` layout). `v` must be `< 2^256`.
fn to_be32(v: &BigUint) -> [u8; 32] {
    let bytes = v.to_bytes_be();
    let mut out = [0u8; 32];
    out[32 - bytes.len()..].copy_from_slice(&bytes);
    out
}

/// Maps an on-curve affine point into k256's representation.
fn to_k256(p: &AffinePoint) -> K256Affine {
    let mut enc = [0u8; 65];
    enc[0] = 0x04; // uncompressed SEC1: 0x04 || x || y
    enc[1..33].copy_from_slice(&to_be32(&p.x));
    enc[33..65].copy_from_slice(&to_be32(&p.y));
    let ep = EncodedPoint::from_bytes(enc).expect("valid uncompressed SEC1 encoding");
    Option::from(K256Affine::from_encoded_point(&ep)).expect("point is on the curve")
}

/// Maps a k256 affine point back into `BigUint` coordinates. The identity never occurs in
/// the accelerator's schedule, so its (absent) coordinates are treated as unreachable.
fn from_k256(p: &K256Affine) -> AffinePoint {
    let ep = p.to_encoded_point(false);
    AffinePoint {
        x: BigUint::from_bytes_be(ep.x().expect("not the identity").as_slice()),
        y: BigUint::from_bytes_be(ep.y().expect("not the identity").as_slice()),
    }
}

/// Recovers the canonical (even) `y` for a given `x` such that `y^2 = x^3 + b mod p`.
///
/// Both `y` and `p - y` are valid; we pick the even one so the executor and prover agree
/// deterministically. The chip never constrains the parity (it only writes back `xR`, and
/// `k·P` and `k·(-P)` share an x-coordinate), so any consistent choice is sound.
///
/// Returns `None` when `x` is not a valid curve x-coordinate (`x^3 + b` is not a quadratic
/// residue, or `x` is not a canonical field element).
pub fn recover_y_canonical(x: &BigUint) -> Option<BigUint> {
    // SEC1 compressed encoding: the `0x02` prefix selects the even-`y` root.
    let mut enc = [0u8; 33];
    enc[0] = 0x02;
    enc[1..33].copy_from_slice(&to_be32(x));
    let ep = EncodedPoint::from_bytes(enc).ok()?;
    let affine: K256Affine = Option::from(K256Affine::from_encoded_point(&ep))?;
    Some(from_k256(&affine).y)
}

/// `2·a` on the curve.
pub fn point_double(a: &AffinePoint) -> AffinePoint {
    let p = ProjectivePoint::from(to_k256(a));
    from_k256(&(p + p).to_affine())
}

/// `a + g` on the curve. Requires `a.x != g.x` (always true in the chip's add steps).
pub fn point_add(a: &AffinePoint, g: &AffinePoint) -> AffinePoint {
    let r = ProjectivePoint::from(to_k256(a)) + ProjectivePoint::from(to_k256(g));
    from_k256(&r.to_affine())
}

/// One step of the double-and-add replay, at point level.
///
/// Mirrors a single ECDAS row: receive accumulator `a` (and base `g`), perform `op`
/// (0 = double, 1 = add), and decide `next_op` (whether the next row is an add).
#[derive(Clone, Debug)]
pub struct StepPts {
    pub a: AffinePoint,
    pub g: AffinePoint,
    pub round: u8,
    pub op: u8,
    pub next_op: u8,
    pub r: AffinePoint,
}

/// Bit length minus one = position of the most significant set bit (`len_k`).
/// Requires `k >= 1`.
pub fn msb_position(k: &BigUint) -> u32 {
    debug_assert!(k > &BigUint::from(0u8));
    (k.bits() as u32) - 1
}

/// Replays the ECDAS double-and-add sequence for `k·g`, returning every step and the
/// final point. This is the single source of truth for both the executor (which needs
/// only `final.x`) and the prover (which needs the full step list to build witnesses).
///
/// The schedule matches the spec exactly: start with `A = g`, `round = len_k - 1`,
/// `op = double`; a double at `round` sets `next_op` to the scalar bit at `round`
/// (1 ⇒ the next row adds at the same round); an add forces `next_op = 0` and advances
/// the round. The MSB itself is represented by the initial `A = g` (consumed by ECSM via
/// the `BIT[len_k]` interaction), so it is never processed as an add here.
pub fn replay_double_and_add(k: &BigUint, g: &AffinePoint) -> (Vec<StepPts>, AffinePoint) {
    let m = msb_position(k) as i64; // len_k
    let mut a = g.clone();
    let mut round: i64 = m - 1;
    let mut op: u8 = 0; // double
    let mut steps = Vec::new();

    while round >= 0 {
        let (r, next_op) = if op == 0 {
            let r = point_double(&a);
            let bit = if k.bit(round as u64) { 1u8 } else { 0u8 };
            (r, bit)
        } else {
            let r = point_add(&a, g);
            (r, 0u8)
        };
        steps.push(StepPts {
            a: a.clone(),
            g: g.clone(),
            round: round as u8,
            op,
            next_op,
            r: r.clone(),
        });
        let round_sent = round - (1 - next_op as i64);
        a = r;
        if round_sent < 0 {
            break;
        }
        round = round_sent;
        op = next_op;
    }

    (steps, a)
}

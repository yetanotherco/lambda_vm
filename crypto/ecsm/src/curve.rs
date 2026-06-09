//! secp256k1 curve arithmetic in affine coordinates and the chip-faithful
//! double-and-add replay.
//!
//! The curve is `y^2 = x^3 + 7 mod p` (short Weierstrass with `a = 0`). The point at
//! infinity never appears: the ECSM/ECDAS design guarantees it cannot occur for
//! `k in [1, N)` (see `ecsm.typ` "Point at infinity" / ECDAS soundness argument), so the
//! affine formulas below are always well defined.

use num_bigint::BigUint;

use crate::B;
use crate::field::Fp;
use crate::p;

/// An affine curve point. Never the point at infinity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffinePoint {
    pub x: BigUint,
    pub y: BigUint,
}

/// Recovers the canonical (even) `y` for a given `x` such that `y^2 = x^3 + b mod p`.
///
/// Both `y` and `p - y` are valid; we pick the even one so the executor and prover agree
/// deterministically. The chip never constrains the parity (it only writes back `xR`, and
/// `k·P` and `k·(-P)` share an x-coordinate), so any consistent choice is sound.
///
/// Returns `None` when `x^3 + b` is not a quadratic residue (i.e. `x` is not a valid
/// x-coordinate on the curve).
pub fn recover_y_canonical(x: &BigUint) -> Option<BigUint> {
    let x = Fp::new(x.clone());
    let rhs = x.mul(&x).mul(&x).add(&Fp::from_u64(B)); // x^3 + b
    let y = rhs.sqrt()?;
    let y = if y.0.bit(0) {
        // odd → take the even root p - y
        Fp::new(p() - &y.0)
    } else {
        y
    };
    Some(y.0)
}

/// `2·a` on the curve. Requires `a.y != 0` (always true on secp256k1).
pub fn point_double(a: &AffinePoint) -> AffinePoint {
    let x = Fp::new(a.x.clone());
    let y = Fp::new(a.y.clone());
    // λ = 3x² / 2y
    let three_x2 = x.mul(&x).mul(&Fp::from_u64(3));
    let two_y = y.add(&y);
    let lambda = three_x2.mul(&two_y.inv());
    // xr = λ² - 2x
    let xr = lambda.mul(&lambda).sub(&x).sub(&x);
    // yr = λ(x - xr) - y
    let yr = lambda.mul(&x.sub(&xr)).sub(&y);
    AffinePoint { x: xr.0, y: yr.0 }
}

/// `a + g` on the curve. Requires `a.x != g.x` (always true in the chip's add steps).
pub fn point_add(a: &AffinePoint, g: &AffinePoint) -> AffinePoint {
    let xa = Fp::new(a.x.clone());
    let ya = Fp::new(a.y.clone());
    let xg = Fp::new(g.x.clone());
    let yg = Fp::new(g.y.clone());
    // λ = (yg - ya) / (xg - xa)
    let lambda = yg.sub(&ya).mul(&xg.sub(&xa).inv());
    // xr = λ² - xa - xg
    let xr = lambda.mul(&lambda).sub(&xa).sub(&xg);
    // yr = λ(xa - xr) - ya
    let yr = lambda.mul(&xa.sub(&xr)).sub(&ya);
    AffinePoint { x: xr.0, y: yr.0 }
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

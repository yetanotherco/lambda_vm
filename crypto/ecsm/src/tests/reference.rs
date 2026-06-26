//! Spec-faithful reference double-and-add over secp256k1 in affine `U256`
//! arithmetic. Test-only: it cross-checks the production k256-backed
//! [`replay_double_and_add`](crate::curve::replay_double_and_add) fast path,
//! which the parity test pins to this reference.

use crypto_bigint::U256;

use crate::curve::{AffinePoint, StepPts, msb_position};
use crate::tests::reference_field::Fp;

/// `2·a` on the curve. Requires `a.y != 0` (always true on secp256k1).
pub fn point_double(a: &AffinePoint) -> AffinePoint {
    let x = Fp::new(a.x);
    let y = Fp::new(a.y);
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
    let xa = Fp::new(a.x);
    let ya = Fp::new(a.y);
    let xg = Fp::new(g.x);
    let yg = Fp::new(g.y);
    // λ = (yg - ya) / (xg - xa)
    let lambda = yg.sub(&ya).mul(&xg.sub(&xa).inv());
    // xr = λ² - xa - xg
    let xr = lambda.mul(&lambda).sub(&xa).sub(&xg);
    // yr = λ(xa - xr) - ya
    let yr = lambda.mul(&xa.sub(&xr)).sub(&ya);
    AffinePoint { x: xr.0, y: yr.0 }
}

/// Reference slope `lambda` for one step, computed in `U256` `F_p`.
/// Used by the reference replay.
pub fn step_lambda(a: &AffinePoint, g: &AffinePoint, op: u8) -> U256 {
    let xa = Fp::new(a.x);
    let ya = Fp::new(a.y);
    if op == 1 {
        let xg = Fp::new(g.x);
        let yg = Fp::new(g.y);
        yg.sub(&ya).mul(&xg.sub(&xa).inv()).0
    } else {
        let three_x2 = xa.mul(&xa).mul(&Fp::from_u64(3));
        let two_y = ya.add(&ya);
        three_x2.mul(&two_y.inv()).0
    }
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
pub fn replay_double_and_add_reference(
    k: &U256,
    g: &AffinePoint,
) -> (Vec<StepPts>, AffinePoint) {
    let m = msb_position(k) as i64; // len_k
    let mut a = g.clone();
    let mut round: i64 = m - 1;
    let mut op: u8 = 0; // double
    let mut steps = Vec::new();

    while round >= 0 {
        let (r, next_op) = if op == 0 {
            let r = point_double(&a);
            let bit = if k.bit_vartime(round as u32) { 1u8 } else { 0u8 };
            (r, bit)
        } else {
            let r = point_add(&a, g);
            (r, 0u8)
        };
        steps.push(StepPts {
            lambda: step_lambda(&a, g, op),
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

//! secp256k1 curve arithmetic in affine coordinates and the chip-faithful
//! double-and-add replay.
//!
//! The curve is `y^2 = x^3 + 7 mod p` (short Weierstrass with `a = 0`). The point at
//! infinity never appears: the ECSM/ECDAS design guarantees it cannot occur for
//! `k in [1, N)` (see `ecsm.typ` "Point at infinity" / ECDAS soundness argument), so the
//! affine formulas below are always well defined.

use crypto_bigint::U256;
use crypto_bigint::modular::ConstMontyForm;

// Compile-time Montgomery parameters for secp256k1 p.
crypto_bigint::const_monty_params!(
    Secp256k1Field,
    U256,
    "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"
);

type Fp = ConstMontyForm<Secp256k1Field, 4>;

/// An affine curve point. Never the point at infinity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffinePoint {
    pub x: U256,
    pub y: U256,
}

fn fe_from_u256(v: &U256) -> Fp {
    ConstMontyForm::new(v)
}

fn u256_from_fe(f: &Fp) -> U256 {
    f.retrieve()
}

fn fp_invert(f: Fp) -> Option<Fp> {
    // safegcd inversion; `None` for a zero input (which has no inverse).
    Option::from(f.invert())
}

/// Recovers the canonical (even) `y` for a given `x` such that `y^2 = x^3 + b mod p`.
///
/// Both `y` and `p - y` are valid; we pick the even one so the executor and prover agree
/// deterministically. The chip never constrains the parity (it only writes back `xR`, and
/// `k·P` and `k·(-P)` share an x-coordinate), so any consistent choice is sound.
///
/// Returns `None` when `x` is not a valid curve x-coordinate (`x^3 + b` is not a quadratic
/// residue, or `x` is not a canonical field element).
pub fn recover_y_canonical(x: &U256) -> Option<U256> {
    use k256::elliptic_curve::sec1::{FromSec1Point, Sec1Point};
    let x_bytes: [u8; 32] = x.to_be_bytes().into();
    let mut enc = [0u8; 33];
    enc[0] = 0x02;
    enc[1..33].copy_from_slice(&x_bytes);
    let ep = Sec1Point::<k256::Secp256k1>::from_bytes(enc).ok()?;
    let affine: K256Affine = Option::from(K256Affine::from_sec1_point(&ep))?;
    Some(from_k256_affine(&affine).y)
}

/// One step of the double-and-add replay, at point level.
///
/// Mirrors a single ECDAS row: receive accumulator `a` (and base `g`), perform `op`
/// (0 = double, 1 = add), and decide `next_op` (whether the next row is an add).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepPts {
    pub a: AffinePoint,
    pub g: AffinePoint,
    pub round: u8,
    pub op: u8,
    pub next_op: u8,
    pub r: AffinePoint,
    /// Slope of this step: add => (yG-yA)/(xG-xA), double => 3xA^2/(2yA).
    /// Precomputed here (batched) so the witness builder never inverts per step.
    pub lambda: U256,
}

/// Bit length minus one = position of the most significant set bit (`len_k`).
/// Requires `k >= 1`.
pub fn msb_position(k: &U256) -> u32 {
    debug_assert!(*k != U256::ZERO);
    (k.bits_vartime() as u32) - 1
}

// =========================================================================
// k256-backed fast path: projective double-and-add replay + batch inversion.
//
// The witness generator is untrusted (the ECDAS chip re-proves every step), so
// any audited arithmetic is sound here. We replay the schedule in k256
// projective coordinates (no per-op inversion), `batch_normalize` all points to
// affine in one shot, and batch-invert the slope denominators — replacing the
// ~2*len_k Fermat inversions of the reference with two batched inversions.
// =========================================================================

use k256::elliptic_curve::group::Curve as _;
use k256::elliptic_curve::sec1::{FromSec1Point, Sec1Point, ToSec1Point};
use k256::{AffinePoint as K256Affine, ProjectivePoint, Scalar};
use k256::elliptic_curve::PrimeField as _;

fn to_k256_affine(a: &AffinePoint) -> K256Affine {
    let x_bytes: [u8; 32] = a.x.to_be_bytes().into();
    let y_bytes: [u8; 32] = a.y.to_be_bytes().into();
    let ep = Sec1Point::<k256::Secp256k1>::from_affine_coordinates(
        <&k256::elliptic_curve::FieldBytes<k256::Secp256k1>>::from(&x_bytes),
        <&k256::elliptic_curve::FieldBytes<k256::Secp256k1>>::from(&y_bytes),
        false,
    );
    Option::from(K256Affine::from_sec1_point(&ep)).expect("ECSM: point must be on the curve")
}

fn from_k256_affine(p: &K256Affine) -> AffinePoint {
    let ep = p.to_sec1_point(false);
    AffinePoint {
        x: U256::from_be_slice(ep.x().expect("ECSM: affine point has x")),
        y: U256::from_be_slice(ep.y().expect("ECSM: affine point has y")),
    }
}

/// Montgomery's batch inversion over `Fp`: one real inversion total.
fn batch_invert(xs: &[Fp]) -> Vec<Fp> {
    let n = xs.len();
    let mut prefix = Vec::with_capacity(n);
    let mut acc = Fp::ONE;
    for x in xs {
        prefix.push(acc);
        acc = acc * *x;
    }
    let mut inv = fp_invert(acc).expect("ECSM: batch denominator is nonzero");
    let mut out = vec![Fp::ONE; n];
    for i in (0..n).rev() {
        out[i] = prefix[i] * inv;
        inv = inv * xs[i];
    }
    out
}

/// The double-and-add schedule for `k`: one `(round, op, next_op)` per ECDAS row.
/// Pure bit logic (data-independent of point values), identical control flow to
/// the reference replay.
fn schedule(k: &U256) -> Vec<(u8, u8, u8)> {
    let m = msb_position(k) as i64;
    let mut sched = Vec::new();
    let mut round: i64 = m - 1;
    let mut op: u8 = 0;
    while round >= 0 {
        let next_op = if op == 0 {
            if k.bit_vartime(round as u32) { 1u8 } else { 0u8 }
        } else {
            0u8
        };
        sched.push((round as u8, op, next_op));
        let round_sent = round - (1 - next_op as i64);
        if round_sent < 0 {
            break;
        }
        round = round_sent;
        op = next_op;
    }
    sched
}

/// Executor fast path: the x-coordinate of `k·g`, via k256's optimized scalar
/// multiplication. Needs no step list or slopes, so it skips all witness work.
/// `k` must be in `[1, N)` (guaranteed by `prepare`).
pub fn scalar_mul_affine_x(k: &U256, g: &AffinePoint) -> U256 {
    let k_bytes: [u8; 32] = k.to_be_bytes().into();
    let scalar = Option::<Scalar>::from(Scalar::from_repr(k_bytes.into()))
        .expect("ECSM: scalar k must be < N");
    let g_proj = ProjectivePoint::from(to_k256_affine(g));
    let r = (g_proj * scalar).to_affine();
    from_k256_affine(&r).x
}

/// Replays the ECDAS double-and-add for `k·g` using k256 projective arithmetic and
/// batched inversion. Produces the identical `StepPts` sequence as the BigUint
/// reference replay (validated by the parity test in `tests::curve_tests`), but with
/// two batched inversions instead of one per double/add step.
pub fn replay_double_and_add(k: &U256, g: &AffinePoint) -> (Vec<StepPts>, AffinePoint) {
    let sched = schedule(k);
    if sched.is_empty() {
        return (Vec::new(), g.clone()); // k == 1: result is g, no steps
    }
    let n = sched.len();

    // 1. projective replay (no inversions): record a and r at every step.
    let g_proj = ProjectivePoint::from(to_k256_affine(g));
    let mut a_proj = g_proj;
    let mut points = Vec::with_capacity(2 * n); // [a_0..a_{n-1}, r_0..r_{n-1}]
    let mut r_projs = Vec::with_capacity(n);
    for &(_, op, _) in &sched {
        let r_proj = if op == 0 {
            a_proj.double()
        } else {
            a_proj + g_proj
        };
        points.push(a_proj);
        r_projs.push(r_proj);
        a_proj = r_proj;
    }
    points.extend_from_slice(&r_projs);

    // 2. one batch_normalize for every a and r.
    let mut affine = vec![K256Affine::IDENTITY; points.len()];
    ProjectivePoint::batch_normalize(&points, &mut affine);
    let a_aff: Vec<AffinePoint> = affine[..n].iter().map(from_k256_affine).collect();
    let r_aff: Vec<AffinePoint> = affine[n..].iter().map(from_k256_affine).collect();

    // 3. batch-invert all slope denominators (add: xG-xA, double: 2yA).
    let gx_fe = fe_from_u256(&g.x);
    let gy_fe = fe_from_u256(&g.y);
    let denoms: Vec<Fp> = (0..n)
        .map(|i| {
            if sched[i].1 == 1 {
                gx_fe - fe_from_u256(&a_aff[i].x)
            } else {
                let ya = fe_from_u256(&a_aff[i].y);
                ya + ya
            }
        })
        .collect();
    let inv_denoms = batch_invert(&denoms);

    // 4. slopes and StepPts.
    let steps: Vec<StepPts> = (0..n)
        .map(|i| {
            let num = if sched[i].1 == 1 {
                gy_fe - fe_from_u256(&a_aff[i].y)
            } else {
                let x2 = {
                    let xa = fe_from_u256(&a_aff[i].x);
                    xa * xa
                };
                x2 + x2 + x2 // 3 xA^2
            };
            StepPts {
                a: a_aff[i].clone(),
                g: g.clone(),
                round: sched[i].0,
                op: sched[i].1,
                next_op: sched[i].2,
                r: r_aff[i].clone(),
                lambda: u256_from_fe(&(num * inv_denoms[i])),
            }
        })
        .collect();

    let result = r_aff[n - 1].clone();
    (steps, result)
}

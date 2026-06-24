//! secp256k1 curve arithmetic in affine coordinates and the chip-faithful
//! double-and-add replay.
//!
//! The curve is `y^2 = x^3 + 7 mod p` (short Weierstrass with `a = 0`). The point at
//! infinity never appears: the ECSM/ECDAS design guarantees it cannot occur for
//! `k in [1, N)` (see `ecsm.typ` "Point at infinity" / ECDAS soundness argument), so the
//! affine formulas below are always well defined.

use num_bigint::BigUint;

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
/// Returns `None` when `x` is not a valid curve x-coordinate (`x^3 + b` is not a quadratic
/// residue, or `x` is not a canonical field element).
pub fn recover_y_canonical(x: &BigUint) -> Option<BigUint> {
    // SEC1 compressed encoding: the `0x02` prefix selects the even-`y` root, delegated to k256.
    let mut enc = [0u8; 33];
    enc[0] = 0x02;
    enc[1..33].copy_from_slice(&be32(x));
    let ep = EncodedPoint::from_bytes(enc).ok()?;
    let affine: K256Affine = Option::from(K256Affine::from_encoded_point(&ep))?;
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
    pub lambda: BigUint,
}

/// Bit length minus one = position of the most significant set bit (`len_k`).
/// Requires `k >= 1`.
pub fn msb_position(k: &BigUint) -> u32 {
    debug_assert!(k > &BigUint::from(0u8));
    (k.bits() as u32) - 1
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

use k256::elliptic_curve::ff::PrimeField as _;
use k256::elliptic_curve::group::Curve as _;
use k256::elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
use k256::{AffinePoint as K256Affine, EncodedPoint, FieldElement, ProjectivePoint, Scalar};

/// 32 big-endian bytes of a value known to fit in 256 bits (left zero-padded).
fn be32(v: &BigUint) -> [u8; 32] {
    let b = v.to_bytes_be();
    debug_assert!(b.len() <= 32, "value exceeds 256 bits");
    let mut out = [0u8; 32];
    out[32 - b.len()..].copy_from_slice(&b);
    out
}

fn fe_from_biguint(v: &BigUint) -> FieldElement {
    Option::from(FieldElement::from_bytes(&be32(v).into()))
        .expect("ECSM: field element must be < p")
}

fn biguint_from_fe(f: &FieldElement) -> BigUint {
    BigUint::from_bytes_be(&f.to_bytes())
}

fn to_k256_affine(a: &AffinePoint) -> K256Affine {
    let ep = EncodedPoint::from_affine_coordinates(&be32(&a.x).into(), &be32(&a.y).into(), false);
    Option::from(K256Affine::from_encoded_point(&ep)).expect("ECSM: point must be on the curve")
}

fn from_k256_affine(p: &K256Affine) -> AffinePoint {
    let ep = p.to_encoded_point(false);
    AffinePoint {
        x: BigUint::from_bytes_be(ep.x().expect("ECSM: affine point has x")),
        y: BigUint::from_bytes_be(ep.y().expect("ECSM: affine point has y")),
    }
}

/// Montgomery's batch inversion over `FieldElement`: one real inversion total.
fn batch_invert(xs: &[FieldElement]) -> Vec<FieldElement> {
    let n = xs.len();
    let mut prefix = Vec::with_capacity(n);
    let mut acc = FieldElement::ONE;
    for x in xs {
        prefix.push(acc);
        acc *= *x;
    }
    let mut inv =
        Option::<FieldElement>::from(acc.invert()).expect("ECSM: batch denominator is nonzero");
    let mut out = vec![FieldElement::ONE; n];
    for i in (0..n).rev() {
        out[i] = prefix[i] * inv;
        inv *= xs[i];
    }
    out
}

/// The double-and-add schedule for `k`: one `(round, op, next_op)` per ECDAS row.
/// Pure bit logic (data-independent of point values), identical control flow to
/// the reference replay.
fn schedule(k: &BigUint) -> Vec<(u8, u8, u8)> {
    let m = msb_position(k) as i64;
    let mut sched = Vec::new();
    let mut round: i64 = m - 1;
    let mut op: u8 = 0;
    while round >= 0 {
        let next_op = if op == 0 {
            if k.bit(round as u64) { 1u8 } else { 0u8 }
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
pub fn scalar_mul_affine_x(k: &BigUint, g: &AffinePoint) -> BigUint {
    let scalar = Option::<Scalar>::from(Scalar::from_repr(be32(k).into()))
        .expect("ECSM: scalar k must be < N");
    let g_proj = ProjectivePoint::from(to_k256_affine(g));
    let r = (g_proj * scalar).to_affine();
    from_k256_affine(&r).x
}

/// Replays the ECDAS double-and-add for `k·g` using k256 projective arithmetic and
/// batched inversion. Produces the identical `StepPts` sequence as the BigUint
/// reference replay (validated by the parity test in `tests::curve_tests`), but with
/// two batched inversions instead of one per double/add step.
pub fn replay_double_and_add(k: &BigUint, g: &AffinePoint) -> (Vec<StepPts>, AffinePoint) {
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
    let gx_fe = fe_from_biguint(&g.x);
    let gy_fe = fe_from_biguint(&g.y);
    let denoms: Vec<FieldElement> = (0..n)
        .map(|i| {
            if sched[i].1 == 1 {
                gx_fe - fe_from_biguint(&a_aff[i].x)
            } else {
                let ya = fe_from_biguint(&a_aff[i].y);
                ya + ya
            }
        })
        .collect();
    let inv_denoms = batch_invert(&denoms);

    // 4. slopes and StepPts.
    let steps: Vec<StepPts> = (0..n)
        .map(|i| {
            let num = if sched[i].1 == 1 {
                gy_fe - fe_from_biguint(&a_aff[i].y)
            } else {
                let x2 = {
                    let xa = fe_from_biguint(&a_aff[i].x);
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
                lambda: biguint_from_fe(&(num * inv_denoms[i])),
            }
        })
        .collect();

    let result = r_aff[n - 1].clone();
    (steps, result)
}

//! ECDSA public-key recovery in one call: the reference the executor runs and the
//! witness the ECREC chip proves.
//!
//! The call takes a signature `(r, s)`, a message hash `z` (all 32-byte big-endian,
//! as Ethereum carries them) and the parity of `R`'s `y`, and returns the public key
//! `Q = r⁻¹·(s·R − z·G) = u1·G + u2·R` with `u1 = −z·r⁻¹`, `u2 = s·r⁻¹ (mod N)`, or
//! *bails*: it reports that it computed nothing and the guest recovers the key in
//! software. A bail is always sound (the guest's software path decides), so the chip
//! only has to prove the key when it reports one. The honest executor bails exactly
//! when the chip cannot represent the answer:
//!
//! - `r` or `s` is not in `[1, N)`;
//! - `r` is not the x-coordinate of a curve point (`r³ + 7` has no square root);
//! - `u1 = 0` (`z ≡ 0 mod N`), which the first chain cannot take as a scalar;
//! - `x(u1·G) = x(u2·R)`, where the final chord addition is undefined (this covers
//!   `Q` at infinity).
//!
//! The chip proves the key as two scalar multiplications through the ECSM machinery's
//! `Ecdas` chains (`A = u1·G` keyed at the call's timestamp `T`, `B = u2·R` at
//! `T + 1`) and one ECDAS addition row `Q = A + B` at `T + 2`, plus the scalar
//! arithmetic mod `N` that turns `(r, s, z)` into `(u1, u2)` and the decompression
//! of `R`.

use k256::elliptic_curve::PrimeField;
use k256::elliptic_curve::ops::Reduce;
use k256::elliptic_curve::point::DecompressPoint;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use num_bigint::{BigInt, BigUint};

use crate::curve::{AffinePoint, StepPts, msb_position};
use crate::witness::{
    build_step, carries_x2, carries_yg, conv, ext64, limb_carries, shifted_quotient, to_le_33,
    witness_for_point,
};
use crate::{B, EcdasStep, P_BYTES, R_BYTES, n, p, to_le_32};

/// Status the call writes when it computed the key.
pub const ECREC_STATUS_KEY: u8 = 0;
/// Status the call writes when it bailed (the key bytes are zero).
pub const ECREC_STATUS_BAIL: u8 = 1;

/// The values one non-bailing recovery uses.
struct Recovered {
    /// `r⁻¹ mod N`
    w: BigUint,
    /// `u1 = −z·w mod N`, in `[1, N)`
    u1: BigUint,
    /// `u2 = s·w mod N`, in `[1, N)`
    u2: BigUint,
    /// `R = (r, y)` with the requested parity
    r_point: AffinePoint,
    /// `u1·G`
    a: AffinePoint,
    /// `u2·R`
    b: AffinePoint,
    /// `A + B`
    q: AffinePoint,
}

fn affine(p: &k256::AffinePoint) -> Option<AffinePoint> {
    let ep = p.to_encoded_point(false);
    Some(AffinePoint {
        x: BigUint::from_bytes_be(ep.x()?),
        y: BigUint::from_bytes_be(ep.y()?),
    })
}

fn scalar_big(s: &k256::Scalar) -> BigUint {
    BigUint::from_bytes_be(&s.to_bytes())
}

/// The recovery, or `None` where the call bails (see the module documentation).
fn recover(sig_be: &[u8; 64], msg_be: &[u8; 32], parity: u8) -> Option<Recovered> {
    let mut r_be = [0u8; 32];
    let mut s_be = [0u8; 32];
    r_be.copy_from_slice(&sig_be[..32]);
    s_be.copy_from_slice(&sig_be[32..]);
    let r = Option::<k256::Scalar>::from(k256::Scalar::from_repr(r_be.into()))?;
    let s = Option::<k256::Scalar>::from(k256::Scalar::from_repr(s_be.into()))?;
    if bool::from(r.is_zero()) || bool::from(s.is_zero()) {
        return None;
    }
    // r < N < p, so r is a canonical x; the parity selects the root.
    let r_point = Option::<k256::AffinePoint>::from(k256::AffinePoint::decompress(
        &r_be.into(),
        (parity & 1).into(),
    ))?;
    let z = <k256::Scalar as Reduce<k256::U256>>::reduce_bytes(&(*msg_be).into());
    let w = Option::<k256::Scalar>::from(r.invert())?;
    let u1 = -(z * w);
    let u2 = s * w;
    if bool::from(u1.is_zero()) {
        return None;
    }
    let a = (k256::ProjectivePoint::GENERATOR * u1).to_affine();
    let b = (k256::ProjectivePoint::from(r_point) * u2).to_affine();
    let (a, b) = (affine(&a)?, affine(&b)?);
    if a.x == b.x {
        return None;
    }
    let q = (k256::ProjectivePoint::from(r_point) * u2 + k256::ProjectivePoint::GENERATOR * u1)
        .to_affine();
    Some(Recovered {
        w: scalar_big(&w),
        u1: scalar_big(&u1),
        u2: scalar_big(&u2),
        r_point: affine(&r_point)?,
        a,
        b,
        q: affine(&q)?,
    })
}

/// 32 big-endian bytes of a value below 2^256.
fn be32(v: &BigUint) -> [u8; 32] {
    let mut out = to_le_32(v);
    out.reverse();
    out
}

/// The executor's entry point: the key `(Q.x, Q.y)` as 32-byte big-endian
/// coordinates, or `None` where the call bails. `parity` is the requested parity of
/// `R`'s y (the recovery id's low bit).
pub fn ecrecover(sig_be: &[u8; 64], msg_be: &[u8; 32], parity: u8) -> Option<([u8; 32], [u8; 32])> {
    let rec = recover(sig_be, msg_be, parity)?;
    Some((be32(&rec.q.x), be32(&rec.q.y)))
}

/// Full ECREC-chip witness for one call (one ECREC row) plus its ECDAS rows.
/// Every 256-bit quantity is 32 little-endian bytes. On a bail (`ok == false`) only
/// the inputs are set; every other field is zero and every list empty.
#[derive(Debug, Clone)]
pub struct EcrecWitness {
    /// `r`, `s` and `z` as read from memory, little-endian limbs.
    pub r: [u8; 32],
    pub s: [u8; 32],
    pub z: [u8; 32],
    /// The requested parity of `R`'s y.
    pub parity: u8,
    /// Whether the call computed the key (status 0) or bailed (status 1).
    pub ok: bool,
    /// `w = r⁻¹ mod N`; `r·w − 1 − qw·N = 0`.
    pub w: [u8; 32],
    pub qw: [u8; 32],
    pub cw: [i64; 64],
    /// `u1 = −z·w mod N`; `z·w + u1 − qz·N = 0`.
    pub u1: [u8; 32],
    pub qz: [u8; 32],
    pub cz: [i64; 64],
    /// `u2 = s·w mod N`; `s·w − u2 − qu·N = 0`.
    pub u2: [u8; 32],
    pub qu: [u8; 32],
    pub cu: [i64; 64],
    /// Positions of the most significant set bits of `u1` and `u2`.
    pub len1: u8,
    pub len2: u8,
    /// `(v − N) mod 2^256` for the four `< N` checks.
    pub r_sub_n: [u8; 32],
    pub s_sub_n: [u8; 32],
    pub u1_sub_n: [u8; 32],
    pub u2_sub_n: [u8; 32],
    /// `R`'s y and the curve relations on `(r, y_R)`, as in ECSM.
    pub y_r: [u8; 32],
    pub x2: [u8; 32],
    pub q0: [u8; 32],
    pub c0: [i64; 64],
    pub q1: [u8; 33],
    pub c1: [i64; 64],
    pub y_r_sub_p: [u8; 32],
    /// `y_R[0] = 2·h + parity`.
    pub h: u8,
    /// The two chains' results `A = u1·G`, `B = u2·R`.
    pub x_a: [u8; 32],
    pub y_a: [u8; 32],
    pub x_b: [u8; 32],
    pub y_b: [u8; 32],
    pub x_a_sub_p: [u8; 32],
    pub x_b_sub_p: [u8; 32],
    /// The key `Q = A + B`.
    pub x_q: [u8; 32],
    pub y_q: [u8; 32],
    pub x_q_sub_p: [u8; 32],
    pub y_q_sub_p: [u8; 32],
    /// ECDAS rows: the first chain (timestamp `T`), the second (`T + 1`), the final
    /// addition (`T + 2`).
    pub chain1: Vec<EcdasStep>,
    pub chain2: Vec<EcdasStep>,
    pub add: Option<EcdasStep>,
}

impl EcrecWitness {
    fn bail(r: [u8; 32], s: [u8; 32], z: [u8; 32], parity: u8) -> Self {
        Self {
            r,
            s,
            z,
            parity,
            ok: false,
            w: [0; 32],
            qw: [0; 32],
            cw: [0; 64],
            u1: [0; 32],
            qz: [0; 32],
            cz: [0; 64],
            u2: [0; 32],
            qu: [0; 32],
            cu: [0; 64],
            len1: 0,
            len2: 0,
            r_sub_n: [0; 32],
            s_sub_n: [0; 32],
            u1_sub_n: [0; 32],
            u2_sub_n: [0; 32],
            y_r: [0; 32],
            x2: [0; 32],
            q0: [0; 32],
            c0: [0; 64],
            q1: [0; 33],
            c1: [0; 64],
            y_r_sub_p: [0; 32],
            h: 0,
            x_a: [0; 32],
            y_a: [0; 32],
            x_b: [0; 32],
            y_b: [0; 32],
            x_a_sub_p: [0; 32],
            x_b_sub_p: [0; 32],
            x_q: [0; 32],
            y_q: [0; 32],
            x_q_sub_p: [0; 32],
            y_q_sub_p: [0; 32],
            chain1: Vec::new(),
            chain2: Vec::new(),
            add: None,
        }
    }

    /// A witness still to be computed: the inputs as read from memory
    /// (little-endian limbs) and every other field zero, shaped as a bail. The
    /// prover holds one where it computes the call's witness later; no table
    /// may be built from it.
    pub fn unresolved(r: [u8; 32], s: [u8; 32], z: [u8; 32], parity: u8) -> Self {
        Self::bail(r, s, z, parity & 1)
    }

    /// The status byte the call writes.
    pub fn status(&self) -> u8 {
        if self.ok {
            ECREC_STATUS_KEY
        } else {
            ECREC_STATUS_BAIL
        }
    }
}

/// The generator's coordinates, little-endian.
pub fn generator_le() -> ([u8; 32], [u8; 32]) {
    let g = affine(&k256::AffinePoint::GENERATOR).expect("G is not the identity");
    (to_le_32(&g.x), to_le_32(&g.y))
}

/// The ECDAS rows of the double-and-add chain of `k·P` for a full point `P = (x, y)`
/// (little-endian), and its result; `None` unless `0 < k < N` and `P` is a canonical
/// curve point.
pub fn ecdas_chain(
    k_le: &[u8; 32],
    x_le: &[u8; 32],
    y_le: &[u8; 32],
) -> Option<(Vec<EcdasStep>, [u8; 32], [u8; 32])> {
    let k = BigUint::from_bytes_le(k_le);
    if k == BigUint::from(0u8) || k >= n() {
        return None;
    }
    let pt = AffinePoint {
        x: BigUint::from_bytes_le(x_le),
        y: BigUint::from_bytes_le(y_le),
    };
    let pp = p();
    if pt.x >= pp || pt.y >= pp || (&pt.y * &pt.y) % &pp != (&pt.x * &pt.x * &pt.x + B) % &pp {
        return None;
    }
    let w = witness_for_point(&k, &pt, k_le);
    Some((w.steps, w.x_r, w.y_r))
}

/// The ECDAS addition row `A + B` (round 0, `op` 1, `next_op` 0) for two curve
/// points given little-endian, with its result in `x_r`, `y_r`; `None` when
/// `xA = xB` (the chord is undefined).
pub fn ecdas_add_step(
    xa: &[u8; 32],
    ya: &[u8; 32],
    xb: &[u8; 32],
    yb: &[u8; 32],
) -> Option<EcdasStep> {
    let pp = p();
    let big = |b: &[u8; 32]| BigUint::from_bytes_le(b);
    let (a, b) = (
        AffinePoint {
            x: big(xa),
            y: big(ya),
        },
        AffinePoint {
            x: big(xb),
            y: big(yb),
        },
    );
    if a.x == b.x {
        return None;
    }
    let dy = (&pp + &b.y - &a.y) % &pp;
    let dx = (&pp + &b.x - &a.x) % &pp;
    let lambda = dy * dx.modpow(&(&pp - BigUint::from(2u8)), &pp) % &pp;
    let x = (&lambda * &lambda + &pp * 2u8 - &a.x - &b.x) % &pp;
    let y = (&lambda * ((&pp + &a.x - &x) % &pp) + &pp - &a.y) % &pp;
    let q = AffinePoint { x, y };
    Some(build_step(
        &StepPts {
            a,
            g: b,
            round: 0,
            op: 1,
            next_op: 0,
            r: q,
            lambda,
        },
        &BigInt::from(pp.clone()),
        &BigInt::from(BigUint::from_bytes_le(&R_BYTES)),
        &ext64(&R_BYTES),
        &ext64(&P_BYTES),
    ))
}

/// Little-endian limbs of a big-endian 32-byte value.
fn le_of_be(be: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, b) in be.iter().rev().enumerate() {
        out[i] = *b;
    }
    out
}

/// `(v − m) mod 2^256` as 32 little-endian bytes.
fn sub_mod_2_256(v: &BigUint, m: &BigUint) -> [u8; 32] {
    let two_256 = BigUint::from(1u8) << 256u32;
    to_le_32(&((two_256 + v) - m))
}

/// Witness of `a·b + sign·c − q·m = 0` over 32-byte limbs (`sign` = ±1): the quotient
/// `q` (asserted to be a non-negative integer that fits 32 bytes) and the 64 carries.
fn product_relation(
    relation: &str,
    a: &BigUint,
    b: &BigUint,
    c: &BigUint,
    sign: i8,
    m: &BigUint,
) -> ([u8; 32], [i64; 64]) {
    let num = if sign > 0 { a * b + c } else { a * b - c };
    assert!(
        (&num % m) == BigUint::from(0u8),
        "ECREC witness {relation}: not divisible by the modulus"
    );
    let q = &num / m;
    assert!(
        q.bits() <= 256,
        "ECREC witness {relation}: quotient exceeds 32 bytes"
    );
    let (ae, be, ce, qe, me) = (
        ext64(&to_le_32(a)),
        ext64(&to_le_32(b)),
        ext64(&to_le_32(c)),
        ext64(&to_le_32(&q)),
        ext64(&to_le_32(m)),
    );
    let mut terms = [0i128; 64];
    for (i, t) in terms.iter_mut().enumerate() {
        *t = conv(&ae, &be, i) + i128::from(sign) * ce[i] - conv(&qe, &me, i);
    }
    (to_le_32(&q), limb_carries(relation, &terms))
}

/// Computes the ECREC witness for one call: the inputs exactly as read from memory
/// (`r ‖ s` and `z` big-endian), and `parity` (0 or 1).
pub fn compute_ecrec_witness(sig_be: &[u8; 64], msg_be: &[u8; 32], parity: u8) -> EcrecWitness {
    let r_le = le_of_be(&sig_be[..32]);
    let s_le = le_of_be(&sig_be[32..]);
    let z_le = le_of_be(msg_be);
    let Some(rec) = recover(sig_be, msg_be, parity) else {
        return EcrecWitness::bail(r_le, s_le, z_le, parity);
    };
    let (nn, pp_big) = (n(), p());
    let r = BigUint::from_bytes_le(&r_le);
    let s = BigUint::from_bytes_le(&s_le);
    let z = BigUint::from_bytes_le(&z_le);
    let one = BigUint::from(1u8);

    // --- scalars mod N ---
    let (qw, cw) = product_relation("w", &r, &rec.w, &one, -1, &nn);
    let (qz, cz) = product_relation("u1", &z, &rec.w, &rec.u1, 1, &nn);
    let (qu, cu) = product_relation("u2", &s, &rec.w, &rec.u2, -1, &nn);

    // --- R on the curve, as ECSM proves (xG, yG): x2 = r² mod p, yR² + p² − r·x2 − b − q1·p = 0 ---
    let rp = &rec.r_point;
    let x_sq = &rp.x * &rp.x;
    let x2_big = &x_sq % &pp_big;
    let q0_big = (&x_sq - &x2_big) / &pp_big;
    let p_ext = ext64(&P_BYTES);
    let (x_b, x2_b, q0_b, yr_b) = (
        to_le_32(&rp.x),
        to_le_32(&x2_big),
        to_le_32(&q0_big),
        to_le_32(&rp.y),
    );
    let c0 = carries_x2(&ext64(&x_b), &ext64(&x2_b), &ext64(&q0_b), &p_ext);
    let p_signed = BigInt::from(pp_big.clone());
    let num_yg = BigInt::from(&rp.y * &rp.y) - BigInt::from(&rp.x * &x2_big) - BigInt::from(B);
    let q1_big = shifted_quotient("ecrec yR", &num_yg, &p_signed, &p_signed);
    let q1_b = to_le_33("ecrec yR", &q1_big);
    let b_ext = {
        let mut a = [0u8; 32];
        a[0] = B as u8;
        ext64(&a)
    };
    let c1 = carries_yg(
        &ext64(&yr_b),
        &p_ext,
        &ext64(&x2_b),
        &ext64(&x_b),
        &ext64(&q1_b),
        &b_ext,
    );
    assert_eq!(yr_b[0] & 1, parity & 1, "R's y has the requested parity");

    // --- the two chains ---
    let g = affine(&k256::AffinePoint::GENERATOR).expect("G is not the identity");
    let u1_le = to_le_32(&rec.u1);
    let u2_le = to_le_32(&rec.u2);
    let chain1 = witness_for_point(&rec.u1, &g, &u1_le);
    let chain2 = witness_for_point(&rec.u2, rp, &u2_le);
    assert_eq!(
        (chain1.x_r, chain1.y_r),
        (to_le_32(&rec.a.x), to_le_32(&rec.a.y)),
        "the first chain ends at u1·G"
    );
    assert_eq!(
        (chain2.x_r, chain2.y_r),
        (to_le_32(&rec.b.x), to_le_32(&rec.b.y)),
        "the second chain ends at u2·R"
    );

    // --- the final addition, one ECDAS add row ---
    let (a, b, q) = (&rec.a, &rec.b, &rec.q);
    let lambda = {
        let dy = (&pp_big + &b.y - &a.y) % &pp_big;
        let dx = (&pp_big + &b.x - &a.x) % &pp_big;
        let inv = dx.modpow(&(&pp_big - BigUint::from(2u8)), &pp_big);
        dy * inv % &pp_big
    };
    let add = build_step(
        &StepPts {
            a: a.clone(),
            g: b.clone(),
            round: 0,
            op: 1,
            next_op: 0,
            r: q.clone(),
            lambda,
        },
        &p_signed,
        &BigInt::from(BigUint::from_bytes_le(&R_BYTES)),
        &ext64(&R_BYTES),
        &p_ext,
    );

    EcrecWitness {
        r: r_le,
        s: s_le,
        z: z_le,
        parity: parity & 1,
        ok: true,
        w: to_le_32(&rec.w),
        qw,
        cw,
        u1: u1_le,
        qz,
        cz,
        u2: u2_le,
        qu,
        cu,
        len1: msb_position(&rec.u1) as u8,
        len2: msb_position(&rec.u2) as u8,
        r_sub_n: sub_mod_2_256(&r, &nn),
        s_sub_n: sub_mod_2_256(&s, &nn),
        u1_sub_n: sub_mod_2_256(&rec.u1, &nn),
        u2_sub_n: sub_mod_2_256(&rec.u2, &nn),
        y_r: yr_b,
        x2: x2_b,
        q0: q0_b,
        c0,
        q1: q1_b,
        c1,
        y_r_sub_p: sub_mod_2_256(&rp.y, &pp_big),
        h: yr_b[0] >> 1,
        x_a: to_le_32(&a.x),
        y_a: to_le_32(&a.y),
        x_b: to_le_32(&b.x),
        y_b: to_le_32(&b.y),
        x_a_sub_p: sub_mod_2_256(&a.x, &pp_big),
        x_b_sub_p: sub_mod_2_256(&b.x, &pp_big),
        x_q: to_le_32(&q.x),
        y_q: to_le_32(&q.y),
        x_q_sub_p: sub_mod_2_256(&q.x, &pp_big),
        y_q_sub_p: sub_mod_2_256(&q.y, &pp_big),
        chain1: chain1.steps,
        chain2: chain2.steps,
        add: Some(add),
    }
}

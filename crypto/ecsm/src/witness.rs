//! ECSM / ECDAS witness generation.
//!
//! For one `ECALL`, the prover must fill the byte-limb witnesses that the ECSM and ECDAS
//! chips constrain: the `yG` reconstruction, the scalar range data, and — per double/add
//! step — the slope `λ`, three quotients, and three carry arrays. This module computes all
//! of them by literally reproducing the spec's limb-convolution recurrences, so the values
//! it emits satisfy the AIR constraints by construction.
//!
//! ## Limb-convolution carries
//!
//! Each "`x ≡ y mod p`" relation is expressed in the spec as a 512-bit integer identity
//! `LHS − RHS = 0`, written limb-by-limb (8-bit limbs) with a chain of carries:
//! `2^8·c_i = c_{i-1} + S_i`, `c_{-1} = 0`, closing with `c_63 = 0` (see `ecsm.typ`
//! "Discussing the carries"). `S_i` is the coefficient of `2^{8i}` in `LHS − RHS`
//! (a sum of byte products — the convolution — plus single-limb terms). Carries can be
//! negative; the chip range-checks `c_i + offset` as a halfword. We reproduce the exact
//! integer recurrence here; the prover converts the resulting integers to field elements.

use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, Zero};

use crate::curve::{StepPts, replay_double_and_add};
use crate::{B, EcsmError, P_BYTES, R_BYTES, n, p, prepare, to_le_32};

/// Full ECSM-chip witness for one scalar multiplication (one ECSM row).
#[derive(Debug, Clone)]
pub struct EcsmWitness {
    pub x_g: [u8; 32],
    pub y_g: [u8; 32],
    pub k: [u8; 32],
    /// `x2 = xG^2 mod p`
    pub x2: [u8; 32],
    /// quotient for the `x2` relation
    pub q0: [u8; 32],
    /// carries for the `x2` relation
    pub c0: [i64; 64],
    /// quotient for the `yG` relation (33 bytes; byte 32 is a single bit)
    pub q1: [u8; 33],
    /// carries for the `yG` relation
    pub c1: [i64; 64],
    /// `(xG - p) mod 2^256`
    pub x_g_sub_p: [u8; 32],
    /// `(k - N) mod 2^256`
    pub k_sub_n: [u8; 32],
    /// `(xR - p) mod 2^256`
    pub x_r_sub_p: [u8; 32],
    /// position of the most significant set bit of `k`
    pub len_k: u8,
    pub x_r: [u8; 32],
    pub y_r: [u8; 32],
    /// the double/add steps (one ECDAS row each; empty when `k == 1`)
    pub steps: Vec<EcdasStep>,
}

/// Full ECDAS-chip witness for one double/add step (one ECDAS row).
#[derive(Debug, Clone)]
pub struct EcdasStep {
    pub x_a: [u8; 32],
    pub y_a: [u8; 32],
    pub x_g: [u8; 32],
    pub y_g: [u8; 32],
    pub round: u8,
    /// 0 = double, 1 = add
    pub op: u8,
    /// op-flag of the next step (1 ⇒ next row adds at this round)
    pub next_op: u8,
    pub lambda: [u8; 32],
    pub x_r: [u8; 32],
    pub y_r: [u8; 32],
    /// quotient for the `λ` relation (33 bytes)
    pub q0: [u8; 33],
    /// quotient for the `xR` relation (33 bytes)
    pub q1: [u8; 33],
    /// quotient for the `yR` relation (33 bytes)
    pub q2: [u8; 33],
    pub c0: [i64; 64],
    pub c1: [i64; 64],
    pub c2: [i64; 64],
}

// =========================================================================
// Limb helpers
// =========================================================================

/// Zero-extends a little-endian byte slice (≤ 64 bytes) to 64 `i128` limbs.
fn ext64(bytes: &[u8]) -> [i128; 64] {
    let mut a = [0i128; 64];
    for (i, &b) in bytes.iter().enumerate() {
        a[i] = b as i128;
    }
    a
}

/// Convolution `Σ_{j=0}^{i} a[j]·b[i-j]`.
fn conv(a: &[i128; 64], b: &[i128; 64], i: usize) -> i128 {
    let mut s = 0i128;
    for j in 0..=i {
        s += a[j] * b[i - j];
    }
    s
}

/// Computes the 64 carries from per-limb terms via `2^8·c_i = c_{i-1} + terms_i`,
/// `c_{-1} = 0`, asserting exact divisibility at every limb and the closing `c_63 = 0`.
///
/// These asserts catch any transcription error in the `terms` builders: for valid inputs
/// the relation `LHS − RHS = 0` holds exactly, so every partial sum is divisible by 256.
fn limb_carries(relation: &str, terms: &[i128; 64]) -> [i64; 64] {
    let mut c = [0i64; 64];
    let mut carry: i128 = 0;
    for i in 0..64 {
        let s = carry + terms[i];
        assert!(
            (s & 0xFF) == 0,
            "ECSM witness {relation}: limb {i} not divisible by 256"
        );
        // `s` is a multiple of 256 (asserted), so the arithmetic shift equals the
        // truncating division `s / 256` even when `s` is negative.
        carry = s >> 8;
        c[i] = carry as i64;
    }
    assert!(
        c[63] == 0,
        "ECSM witness {relation}: closing carry c_63 must be 0"
    );
    c
}

// =========================================================================
// Per-relation carry builders (mirror the spec TOML polys exactly)
// =========================================================================

/// ECSM `x2` relation: `xG^2 − x2 − q0·p = 0`.
fn carries_x2(xg: &[i128; 64], x2: &[i128; 64], q0: &[i128; 64], pp: &[i128; 64]) -> [i64; 64] {
    let mut terms = [0i128; 64];
    for i in 0..64 {
        terms[i] = conv(xg, xg, i) - x2[i] - conv(q0, pp, i);
    }
    limb_carries("x2", &terms)
}

/// ECSM `yG` relation: `yG^2 + p^2 − xG·x2 − b − q1·p = 0`.
fn carries_yg(
    yg: &[i128; 64],
    pp: &[i128; 64],
    x2: &[i128; 64],
    xg: &[i128; 64],
    q1: &[i128; 64],
    b: &[i128; 64],
) -> [i64; 64] {
    let mut terms = [0i128; 64];
    for i in 0..64 {
        terms[i] = conv(yg, yg, i) + conv(pp, pp, i) - conv(x2, xg, i) - conv(q1, pp, i) - b[i];
    }
    limb_carries("yG", &terms)
}

/// ECDAS `λ` relation:
/// `op·(λ(xG−xA) − yG + yA) + (1−op)(2λyA − 3xA²) + (r − q0)p = 0`.
#[allow(clippy::too_many_arguments)]
fn carries_lambda(
    op: u8,
    lam: &[i128; 64],
    xg: &[i128; 64],
    xa: &[i128; 64],
    ya: &[i128; 64],
    yg: &[i128; 64],
    r: &[i128; 64],
    pp: &[i128; 64],
    q0: &[i128; 64],
) -> [i64; 64] {
    let mut terms = [0i128; 64];
    for i in 0..64 {
        let branch = if op == 1 {
            // op · (Σ_j λ_j (xG_{i-j} − xA_{i-j}) + (yA_i − yG_i))
            let mut s = ya[i] - yg[i];
            for j in 0..=i {
                s += lam[j] * (xg[i - j] - xa[i - j]);
            }
            s
        } else {
            // (1−op) · Σ_j (2 λ_j yA_{i-j} − 3 xA_j xA_{i-j})
            let mut s = 0i128;
            for j in 0..=i {
                s += 2 * lam[j] * ya[i - j] - 3 * xa[j] * xa[i - j];
            }
            s
        };
        terms[i] = branch + conv(r, pp, i) - conv(q0, pp, i);
    }
    limb_carries("lambda", &terms)
}

/// ECDAS `xR` relation:
/// `λ² − xA − xG − xR − (1−op)(xA − xG) + (r − q1)p = 0`.
#[allow(clippy::too_many_arguments)]
fn carries_xr(
    op: u8,
    lam: &[i128; 64],
    xa: &[i128; 64],
    xg: &[i128; 64],
    xr: &[i128; 64],
    r: &[i128; 64],
    pp: &[i128; 64],
    q1: &[i128; 64],
) -> [i64; 64] {
    let mut terms = [0i128; 64];
    for i in 0..64 {
        let op_term = if op == 0 { xa[i] - xg[i] } else { 0 };
        terms[i] =
            conv(lam, lam, i) - xa[i] - xg[i] - xr[i] - op_term + conv(r, pp, i) - conv(q1, pp, i);
    }
    limb_carries("xR", &terms)
}

/// ECDAS `yR` relation: `λ(xA − xR) − yA − yR + (r − q2)p = 0`.
#[allow(clippy::too_many_arguments)]
fn carries_yr(
    lam: &[i128; 64],
    xa: &[i128; 64],
    xr: &[i128; 64],
    ya: &[i128; 64],
    yr: &[i128; 64],
    r: &[i128; 64],
    pp: &[i128; 64],
    q2: &[i128; 64],
) -> [i64; 64] {
    let mut terms = [0i128; 64];
    for i in 0..64 {
        let mut conv_lam = 0i128;
        for j in 0..=i {
            conv_lam += lam[j] * (xa[i - j] - xr[i - j]);
        }
        terms[i] = conv_lam - ya[i] - yr[i] + conv(r, pp, i) - conv(q2, pp, i);
    }
    limb_carries("yR", &terms)
}

// =========================================================================
// BigInt helpers
// =========================================================================

/// Little-endian 33 bytes of a non-negative value that fits in 264 bits.
fn to_le_33(relation: &str, v: &BigUint) -> [u8; 33] {
    let mut bytes = v.to_bytes_le();
    assert!(
        bytes.len() <= 33,
        "ECSM witness {relation}: quotient exceeds 33 bytes"
    );
    bytes.resize(33, 0);
    let mut out = [0u8; 33];
    out.copy_from_slice(&bytes[..33]);
    out
}

/// `r + numerator / p`, where `numerator` must be divisible by `p`. Asserts divisibility
/// and that the result is non-negative (guaranteed by the spec quotient ranges).
fn shifted_quotient(relation: &str, numerator: &BigInt, p_big: &BigInt, r_big: &BigInt) -> BigUint {
    assert!(
        (numerator % p_big).is_zero(),
        "ECSM witness {relation}: numerator not divisible by p"
    );
    let q = r_big + numerator / p_big;
    assert!(
        !q.is_negative(),
        "ECSM witness {relation}: quotient unexpectedly negative"
    );
    q.to_biguint().expect("non-negative")
}

// =========================================================================
// Witness construction
// =========================================================================

/// Computes the full ECSM/ECDAS witness for `k·G` over secp256k1, given `k` and `xG` as
/// little-endian 32-byte values. This is the prover's entry point.
pub fn compute_witness(k_le: &[u8; 32], xg_le: &[u8; 32]) -> Result<EcsmWitness, EcsmError> {
    let (k, g) = prepare(k_le, xg_le)?;

    let p_big = BigInt::from(p());
    let r_big = BigInt::from(BigUint::from_bytes_le(&R_BYTES)); // r = 3p

    // Common zero-extended constants.
    let pp = ext64(&P_BYTES);
    let r_ext = ext64(&R_BYTES);
    let b_bytes = {
        let mut a = [0u8; 32];
        a[0] = B as u8;
        a
    };
    let b_ext = ext64(&b_bytes);

    // --- ECSM: x2 = xG^2 mod p, quotient q0 ---
    let xg_sq = &g.x * &g.x;
    let x2_big = &xg_sq % p();
    let q0_big = (&xg_sq - &x2_big) / p(); // exact
    let xg_b = to_le_32(&g.x);
    let yg_b = to_le_32(&g.y);
    let x2_b = to_le_32(&x2_big);
    let q0_b = to_le_32(&q0_big);
    let c0 = carries_x2(&ext64(&xg_b), &ext64(&x2_b), &ext64(&q0_b), &pp);

    // --- ECSM: yG relation, quotient q1 = (yG^2 − xG·x2 − b)/p + p ---
    let num_yg = BigInt::from(&g.y * &g.y) - BigInt::from(&g.x * &x2_big) - BigInt::from(B);
    let q1_big = shifted_quotient("yG", &num_yg, &p_big, &p_big);
    let q1_b = to_le_33("yG", &q1_big);
    let c1 = carries_yg(
        &ext64(&yg_b),
        &pp,
        &ext64(&x2_b),
        &ext64(&xg_b),
        &ext64(&q1_b),
        &b_ext,
    );

    // --- scalar range data ---
    let len_k = crate::curve::msb_position(&k) as u8;
    let two_256 = BigUint::from(1u8) << 256u32;
    let x_g_sub_p = to_le_32(&((&two_256 + &g.x) - p())); // xG < p
    let k_sub_n = to_le_32(&((&two_256 + &k) - n())); // k < N

    // --- double/add replay ---
    let (steps_pts, result) = replay_double_and_add(&k, &g);
    let x_r = to_le_32(&result.x);
    let y_r = to_le_32(&result.y);
    let x_r_sub_p = to_le_32(&((&two_256 + &result.x) - p()));

    let steps = steps_pts
        .iter()
        .map(|s| build_step(s, &p_big, &r_big, &r_ext, &pp))
        .collect();

    Ok(EcsmWitness {
        x_g: xg_b,
        y_g: yg_b,
        k: *k_le,
        x2: x2_b,
        q0: q0_b,
        c0,
        q1: q1_b,
        c1,
        x_g_sub_p,
        k_sub_n,
        x_r_sub_p,
        len_k,
        x_r,
        y_r,
        steps,
    })
}

/// Builds one ECDAS step witness (λ, quotients, carries) from a point-level step.
fn build_step(
    s: &StepPts,
    p_big: &BigInt,
    r_big: &BigInt,
    r_ext: &[i128; 64],
    pp: &[i128; 64],
) -> EcdasStep {
    // λ is precomputed (batched) during the double-and-add replay.
    let lam_b = to_le_32(&s.lambda);
    let xa_b = to_le_32(&s.a.x);
    let ya_b = to_le_32(&s.a.y);
    let xg_b = to_le_32(&s.g.x);
    let yg_b = to_le_32(&s.g.y);
    let xr_b = to_le_32(&s.r.x);
    let yr_b = to_le_32(&s.r.y);

    let (lam_ext, xa_ext, ya_ext, xg_ext, yg_ext, xr_ext, yr_ext) = (
        ext64(&lam_b),
        ext64(&xa_b),
        ext64(&ya_b),
        ext64(&xg_b),
        ext64(&yg_b),
        ext64(&xr_b),
        ext64(&yr_b),
    );

    let lam_i = BigInt::from(s.lambda.clone());
    let xa_i = BigInt::from(s.a.x.clone());
    let ya_i = BigInt::from(s.a.y.clone());
    let xg_i = BigInt::from(s.g.x.clone());
    let yg_i = BigInt::from(s.g.y.clone());
    let xr_i = BigInt::from(s.r.x.clone());
    let yr_i = BigInt::from(s.r.y.clone());

    // q0: λ relation numerator.
    let num0 = if s.op == 1 {
        (&xg_i - &xa_i) * &lam_i - &yg_i + &ya_i
    } else {
        2 * &lam_i * &ya_i - 3 * &xa_i * &xa_i
    };
    let q0_big = shifted_quotient("lambda", &num0, p_big, r_big);
    let q0_b = to_le_33("lambda", &q0_big);

    // q1: xR relation numerator  λ² − xA − xG − xR + (1−op)(xG − xA).
    let mut num1 = &lam_i * &lam_i - &xa_i - &xg_i - &xr_i;
    if s.op == 0 {
        num1 += &xg_i - &xa_i;
    }
    let q1_big = shifted_quotient("xR", &num1, p_big, r_big);
    let q1_b = to_le_33("xR", &q1_big);

    // q2: yR relation numerator  λ(xA − xR) − yA − yR.
    let num2 = &lam_i * (&xa_i - &xr_i) - &ya_i - &yr_i;
    let q2_big = shifted_quotient("yR", &num2, p_big, r_big);
    let q2_b = to_le_33("yR", &q2_big);

    let c0 = carries_lambda(
        s.op,
        &lam_ext,
        &xg_ext,
        &xa_ext,
        &ya_ext,
        &yg_ext,
        r_ext,
        pp,
        &ext64(&q0_b),
    );
    let c1 = carries_xr(
        s.op,
        &lam_ext,
        &xa_ext,
        &xg_ext,
        &xr_ext,
        r_ext,
        pp,
        &ext64(&q1_b),
    );
    let c2 = carries_yr(
        &lam_ext,
        &xa_ext,
        &xr_ext,
        &ya_ext,
        &yr_ext,
        r_ext,
        pp,
        &ext64(&q2_b),
    );

    EcdasStep {
        x_a: xa_b,
        y_a: ya_b,
        x_g: xg_b,
        y_g: yg_b,
        round: s.round,
        op: s.op,
        next_op: s.next_op,
        lambda: lam_b,
        x_r: xr_b,
        y_r: yr_b,
        q0: q0_b,
        q1: q1_b,
        q2: q2_b,
        c0,
        c1,
        c2,
    }
}

// =========================================================================
// lincomb2: Q = u1·P1 + u2·P2 via NUMS-blinded joint Shamir/Straus.
//
// PHASE A (this block): witness generation + host validation only. The
// prover chip, executor syscall, and guest switch come in later phases; this
// is the layout lock the chip consumes. The per-row math REUSES the audited,
// gate-verified `build_step` / `carries_*` machinery above — a joint-chain row
// is exactly one double-or-add step, so only the schedule and the choice of
// addend differ from the single-scalar chain.
// =========================================================================

use crate::curve::AffinePoint;

/// NUMS blinding point `T₀` — hash-to-curve of the tag
/// `"lambdavm/ecsm/lincomb2/T0/v1"` by SHA-256 try-and-increment: candidate
/// `x = SHA-256(tag || counter_be32)` (big-endian), first `counter` giving a
/// canonical curve `x` with the even `y`. Result: counter = 1, y even. Derived
/// and reproduced independently in `thoughts/ec-recover-opt/lincomb2/T0.md`,
/// the Python `oracle/lincomb2_ref.py::t0_ref`, and the `t0_derivation_matches`
/// dev test (which re-runs the SHA-256 search). Pinned here as a constant so
/// the crate needs no runtime hash dependency.
pub const T0_X_LE: [u8; 32] = [
    0x64, 0x78, 0xAE, 0xB1, 0x0C, 0x07, 0x49, 0x1B, 0xDB, 0x93, 0x88, 0xA9, 0x9A, 0xA7, 0xFB, 0x5E,
    0x66, 0x0A, 0x33, 0xDB, 0x5E, 0xE8, 0x7D, 0x29, 0x6B, 0xA8, 0x91, 0x0F, 0xA9, 0x9A, 0x31, 0xAF,
];
pub const T0_Y_LE: [u8; 32] = [
    0xA0, 0x77, 0x62, 0x60, 0x4A, 0x25, 0x3C, 0x3F, 0x19, 0x82, 0x7D, 0x21, 0xFA, 0x24, 0xE6, 0xA2,
    0x8C, 0x5B, 0xB0, 0xF3, 0xBC, 0xB0, 0x1D, 0x07, 0x32, 0x07, 0x3C, 0x14, 0x38, 0xA0, 0x81, 0x14,
];

/// The pinned NUMS blinding point `T₀` as an affine point.
pub fn t0() -> AffinePoint {
    AffinePoint {
        x: BigUint::from_bytes_le(&T0_X_LE),
        y: BigUint::from_bytes_le(&T0_Y_LE),
    }
}

/// Reasons a sound lincomb2 witness cannot be built. The executor maps these to
/// a non-zero status word; the guest then takes its pure-Rust software fallback
/// (`ProjectivePoint::lincomb`), which is sound because guest code is proven
/// execution — a lying status only wastes cycles, it can never forge a result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lincomb2Error {
    /// `u1` or `u2` is zero.
    ScalarIsZero,
    /// `u1` or `u2` is `>= N`.
    ScalarOutOfRange,
    /// `P1` or `P2` is not on the curve.
    PointNotOnCurve,
    /// `P1` or `P2` has a non-canonical coordinate (`>= p`).
    PointNotCanonical,
    /// `P1 = ±P2`: the `P1 + P2` precompute is a double or infinity (the
    /// addend table would be degenerate).
    SumDegenerate,
    /// `Q = u1·P1 + u2·P2` is the point at infinity, or an intermediate
    /// accumulator collided with its addend (blinding makes the latter a
    /// discrete-log event; either way there is no affine witness).
    ResultInfinity,
}

/// Curve-membership sub-witness proving `y² ≡ x³ + b (mod p)` for a point
/// `(x, y)` — the same two byte-limb convolutions ECSM proves for its
/// generator, reused here for the variable point `P2`.
#[derive(Debug, Clone)]
pub struct MembershipWitness {
    /// `x2 = x² mod p`
    pub x2: [u8; 32],
    /// quotient for the `x2` relation
    pub q0: [u8; 32],
    /// carries for the `x2` relation
    pub c0: [i64; 64],
    /// quotient for the `y²` relation (33 bytes)
    pub q1: [u8; 33],
    /// carries for the `y²` relation
    pub c1: [i64; 64],
}

/// The role of one joint-chain row, and which addend (if any) it consumes.
/// Distinguishes the rows the chip treats differently even though all share the
/// double/add convolution core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointSel {
    /// A doubling (`op = 0`); no addend (the addend columns are zero and cancel
    /// out of all three relations).
    Double,
    /// An add of `P1` (joint digit `10`).
    AddP1,
    /// An add of `P2` (joint digit `01`).
    AddP2,
    /// An add of `P12 = P1 + P2` (joint digit `11`).
    AddP12,
    /// The `P1 + P2` precompute row (produces `P12`).
    Precompute,
    /// The final blinding correction: add `−2^len·T₀` to strip the blind.
    Correction,
}

/// One row of the joint chain: the double/add step witness (identical shape to
/// [`EcdasStep`], reusing its λ/quotient/carry math) plus the joint-chain
/// bookkeeping the chip needs (which addend, and the two scalar digit bits).
#[derive(Debug, Clone)]
pub struct JointStep {
    pub step: EcdasStep,
    pub sel: JointSel,
    /// `u1`'s bit at this round (0 on non-schedule rows).
    pub d1: u8,
    /// `u2`'s bit at this round (0 on non-schedule rows).
    pub d2: u8,
}

/// Full lincomb2-chip witness for one `Q = u1·P1 + u2·P2` evaluation.
///
/// This struct IS the layout lock: every field is a column block the ECSM′/joint
/// chip must carry. LE byte order throughout, matching the ECSM ABI.
#[derive(Debug, Clone)]
pub struct Lincomb2Witness {
    pub x_p1: [u8; 32],
    pub y_p1: [u8; 32],
    pub x_p2: [u8; 32],
    pub y_p2: [u8; 32],
    /// `P2` curve-membership witness (`P1` is the fixed generator for ecrecover,
    /// on-curve by definition — the chip may hardcode it).
    pub mem_p2: MembershipWitness,
    /// `yP2 < p` canonicalization (NEW load-bearing: `yP2 + p` would flip the
    /// effective parity of `P2` as bytes and hence the sign of `Q`).
    pub y_p2_sub_p: [u8; 32],
    pub x_p12: [u8; 32],
    pub y_p12: [u8; 32],
    pub u1: [u8; 32],
    pub u2: [u8; 32],
    /// `u1 < N` / `u2 < N` overflow witnesses (`2^256 + u − N`).
    pub u1_sub_n: [u8; 32],
    pub u2_sub_n: [u8; 32],
    /// Number of doublings = `max(msb(u1), msb(u2)) + 1`; also the exponent of
    /// the blind `2^len·T₀` the correction row strips.
    pub len: u16,
    pub x_q: [u8; 32],
    pub y_q: [u8; 32],
    /// `xQ < p` / `yQ < p` canonicalization (NEW load-bearing: a `+p`-shifted
    /// output coordinate hashes to a different address in the guest's keccak).
    pub x_q_sub_p: [u8; 32],
    pub y_q_sub_p: [u8; 32],
    /// `T₀` and `2^len·T₀`, the blinding constants (from the preprocessed
    /// T₀ table, indexed by `len`); recorded so the chip can bind them.
    pub x_t0: [u8; 32],
    pub y_t0: [u8; 32],
    pub x_t0_pow: [u8; 32],
    pub y_t0_pow: [u8; 32],
    /// The chain rows: `[Precompute, <len doublings interleaved with adds>, Correction]`.
    pub steps: Vec<JointStep>,
}

// ---- minimal affine group law over BigInt mod p (witness-side, not perf-critical) ----

/// `v mod p`, always in `[0, p)`.
fn red(v: &BigInt, p_big: &BigInt) -> BigInt {
    let m = v % p_big;
    if m.is_negative() { m + p_big } else { m }
}

/// Modular inverse via Fermat (`p` prime).
fn finv_i(a: &BigInt, p_big: &BigInt) -> BigInt {
    red(a, p_big).modpow(&(p_big - BigInt::from(2)), p_big)
}

/// Point double, returning `(2·P, λ)` with `λ = 3x²/(2y)`.
fn ec_double(a: &(BigInt, BigInt), p_big: &BigInt) -> (BigInt, BigInt, BigInt) {
    let (x, y) = a;
    let two = BigInt::from(2u32);
    let three = BigInt::from(3u32);
    let lam = red(&(&three * x * x * finv_i(&(&two * y), p_big)), p_big);
    let x3 = red(&(&lam * &lam - &two * x), p_big);
    let y3 = red(&(&lam * (x - &x3) - y), p_big);
    (x3, y3, lam)
}

/// Point add of distinct-x points, returning `(P + G, λ)` with `λ = (yG−y)/(xG−x)`.
fn ec_add(a: &(BigInt, BigInt), g: &(BigInt, BigInt), p_big: &BigInt) -> (BigInt, BigInt, BigInt) {
    let (xa, ya) = a;
    let (xg, yg) = g;
    let lam = red(&((yg - ya) * finv_i(&(xg - xa), p_big)), p_big);
    let x3 = red(&(&lam * &lam - xa - xg), p_big);
    let y3 = red(&(&lam * (xa - &x3) - ya), p_big);
    (x3, y3, lam)
}

fn to_pt(a: &AffinePoint) -> (BigInt, BigInt) {
    (BigInt::from(a.x.clone()), BigInt::from(a.y.clone()))
}

fn to_affine(p: &(BigInt, BigInt)) -> AffinePoint {
    AffinePoint {
        x: p.0.to_biguint().expect("reduced"),
        y: p.1.to_biguint().expect("reduced"),
    }
}

/// `2^256 + v − modulus` as 32 LE bytes — the strict-inequality (`v < modulus`)
/// overflow witness, mirroring `x_g_sub_p` in `compute_witness`. Requires
/// `v < modulus` so the result is `< 2^256`.
fn sub_witness(v: &BigUint, modulus: &BigUint) -> [u8; 32] {
    let two_256 = BigUint::from(1u8) << 256u32;
    to_le_32(&((&two_256 + v) - modulus))
}

/// Curve-membership sub-witness for `(x, y)`: reuses the exact `x2` and `yG`
/// convolutions of `compute_witness`.
fn membership_witness(x: &BigUint, y: &BigUint, pp: &[i128; 64]) -> MembershipWitness {
    let p_big = BigInt::from(p());
    let x_b = to_le_32(x);
    let y_b = to_le_32(y);

    let x_sq = x * x;
    let x2_big = &x_sq % p();
    let q0_big = (&x_sq - &x2_big) / p(); // exact
    let x2_b = to_le_32(&x2_big);
    let q0_b = to_le_32(&q0_big);
    let c0 = carries_x2(&ext64(&x_b), &ext64(&x2_b), &ext64(&q0_b), pp);

    let num_y = BigInt::from(y * y) - BigInt::from(x * &x2_big) - BigInt::from(B);
    let q1_big = shifted_quotient("membership-y", &num_y, &p_big, &p_big);
    let q1_b = to_le_33("membership-y", &q1_big);
    let b_ext = ext64(&{
        let mut a = [0u8; 32];
        a[0] = B as u8;
        a
    });
    let c1 = carries_yg(
        &ext64(&y_b),
        pp,
        &ext64(&x2_b),
        &ext64(&x_b),
        &ext64(&q1_b),
        &b_ext,
    );

    MembershipWitness {
        x2: x2_b,
        q0: q0_b,
        c0,
        q1: q1_b,
        c1,
    }
}

/// Validates a point is on-curve and canonical (both coordinates `< p`).
fn check_point(pt: &AffinePoint) -> Result<(), Lincomb2Error> {
    if pt.x >= p() || pt.y >= p() {
        return Err(Lincomb2Error::PointNotCanonical);
    }
    let lhs = (&pt.y * &pt.y) % p();
    let rhs = (&pt.x * &pt.x % p() * &pt.x + B) % p();
    if lhs != rhs {
        return Err(Lincomb2Error::PointNotOnCurve);
    }
    Ok(())
}

/// Builds one joint-chain row from a raw (a, addend, op, result, λ) tuple by
/// delegating to the shared `build_step`. `round`/`next_op` are bookkeeping the
/// convolution relations do not consume; on a `Double` row the addend is `(0,0)`
/// and cancels out of all three relations.
#[allow(clippy::too_many_arguments)]
fn joint_row(
    a: &AffinePoint,
    addend: &AffinePoint,
    op: u8,
    round: u8,
    next_op: u8,
    r: &AffinePoint,
    lambda: BigUint,
    sel: JointSel,
    d1: u8,
    d2: u8,
    p_big: &BigInt,
    r_big: &BigInt,
    r_ext: &[i128; 64],
    pp: &[i128; 64],
) -> JointStep {
    let s = StepPts {
        a: a.clone(),
        g: addend.clone(),
        round,
        op,
        next_op,
        r: r.clone(),
        lambda,
    };
    JointStep {
        step: build_step(&s, p_big, r_big, r_ext, pp),
        sel,
        d1,
        d2,
    }
}

/// Computes the full lincomb2 witness for `Q = u1·P1 + u2·P2` over secp256k1.
/// `u1_le`, `u2_le` are little-endian 32-byte scalars in `[1, N)`; `p1`, `p2`
/// are canonical on-curve affine points. For ecrecover `P1 = G`, `P2 = R`.
pub fn lincomb2_witness(
    u1_le: &[u8; 32],
    u2_le: &[u8; 32],
    p1: &AffinePoint,
    p2: &AffinePoint,
) -> Result<Lincomb2Witness, Lincomb2Error> {
    let u1 = BigUint::from_bytes_le(u1_le);
    let u2 = BigUint::from_bytes_le(u2_le);
    for u in [&u1, &u2] {
        if u.is_zero() {
            return Err(Lincomb2Error::ScalarIsZero);
        }
        if *u >= n() {
            return Err(Lincomb2Error::ScalarOutOfRange);
        }
    }
    check_point(p1)?;
    check_point(p2)?;

    let p_big = BigInt::from(p());
    let r_big = BigInt::from(BigUint::from_bytes_le(&R_BYTES)); // r = 3p
    let pp = ext64(&P_BYTES);
    let r_ext = ext64(&R_BYTES);
    let zero_pt = AffinePoint {
        x: BigUint::zero(),
        y: BigUint::zero(),
    };

    // --- P12 = P1 + P2 (must be a genuine chord: P1 ≠ ±P2) ---
    if p1.x == p2.x {
        return Err(Lincomb2Error::SumDegenerate);
    }
    let p1_i = to_pt(p1);
    let p2_i = to_pt(p2);
    let (x12, y12, lam12) = ec_add(&p1_i, &p2_i, &p_big);
    let p12 = to_affine(&(x12.clone(), y12.clone()));

    // --- schedule length and the blind exponent ---
    let len = core::cmp::max(u1.bits(), u2.bits()) as usize; // = max_msb + 1
    let t0_pt = t0();
    let t0_i = to_pt(&t0_pt);

    // --- NUMS-blinded joint Shamir/Straus replay ---
    let mut steps: Vec<JointStep> = Vec::with_capacity(len + 194);

    // Row 0: the P12 precompute (a chord add of P1 + P2).
    steps.push(joint_row(
        p1,
        p2,
        1,
        0,
        0,
        &p12,
        lam12.to_biguint().expect("reduced"),
        JointSel::Precompute,
        0,
        0,
        &p_big,
        &r_big,
        &r_ext,
        &pp,
    ));

    let mut acc = t0_i.clone();
    for round in (0..len).rev() {
        // double
        let a_dbl = acc.clone();
        let (dx, dy, dlam) = ec_double(&a_dbl, &p_big);
        acc = (dx.clone(), dy.clone());
        steps.push(joint_row(
            &to_affine(&a_dbl),
            &zero_pt,
            0,
            round as u8,
            0,
            &to_affine(&acc),
            dlam.to_biguint().expect("reduced"),
            JointSel::Double,
            0,
            0,
            &p_big,
            &r_big,
            &r_ext,
            &pp,
        ));

        // conditional add of the selected addend
        let d1 = u1.bit(round as u64) as u8;
        let d2 = u2.bit(round as u64) as u8;
        let (addend_i, addend_pt, sel) = match (d1, d2) {
            (0, 0) => continue,
            (1, 0) => (p1_i.clone(), p1.clone(), JointSel::AddP1),
            (0, 1) => (p2_i.clone(), p2.clone(), JointSel::AddP2),
            _ => ((x12.clone(), y12.clone()), p12.clone(), JointSel::AddP12),
        };
        // acc = ±addend would be a discrete-log collision (blinding); no witness.
        if acc.0 == addend_i.0 {
            return Err(Lincomb2Error::ResultInfinity);
        }
        let a_add = acc.clone();
        let (ax, ay, alam) = ec_add(&a_add, &addend_i, &p_big);
        acc = (ax, ay);
        steps.push(joint_row(
            &to_affine(&a_add),
            &addend_pt,
            1,
            round as u8,
            0,
            &to_affine(&acc),
            alam.to_biguint().expect("reduced"),
            sel,
            d1,
            d2,
            &p_big,
            &r_big,
            &r_ext,
            &pp,
        ));
    }

    // --- blind = 2^len·T₀ ---
    let mut tpow = t0_i.clone();
    for _ in 0..len {
        let (x, y, _) = ec_double(&tpow, &p_big);
        tpow = (x, y);
    }
    // acc == Q + 2^len·T₀; correction row adds −2^len·T₀ to recover Q.
    let neg_tpow = (tpow.0.clone(), red(&(-&tpow.1), &p_big));
    if acc.0 == neg_tpow.0 {
        // acc = ±(−tpow): Q = ∞ (or Q = −2^{len+1}T₀, a dlog event) — degenerate.
        return Err(Lincomb2Error::ResultInfinity);
    }
    let a_corr = acc.clone();
    let (qx, qy, clam) = ec_add(&a_corr, &neg_tpow, &p_big);
    let q = (qx.clone(), qy.clone());
    steps.push(joint_row(
        &to_affine(&a_corr),
        &to_affine(&neg_tpow),
        1,
        0,
        0,
        &to_affine(&q),
        clam.to_biguint().expect("reduced"),
        JointSel::Correction,
        0,
        0,
        &p_big,
        &r_big,
        &r_ext,
        &pp,
    ));

    let q_aff = to_affine(&q);
    let tpow_aff = to_affine(&tpow);

    Ok(Lincomb2Witness {
        x_p1: to_le_32(&p1.x),
        y_p1: to_le_32(&p1.y),
        x_p2: to_le_32(&p2.x),
        y_p2: to_le_32(&p2.y),
        mem_p2: membership_witness(&p2.x, &p2.y, &pp),
        y_p2_sub_p: sub_witness(&p2.y, &p()),
        x_p12: to_le_32(&p12.x),
        y_p12: to_le_32(&p12.y),
        u1: *u1_le,
        u2: *u2_le,
        u1_sub_n: sub_witness(&u1, &n()),
        u2_sub_n: sub_witness(&u2, &n()),
        len: len as u16,
        x_q: to_le_32(&q_aff.x),
        y_q: to_le_32(&q_aff.y),
        x_q_sub_p: sub_witness(&q_aff.x, &p()),
        y_q_sub_p: sub_witness(&q_aff.y, &p()),
        x_t0: T0_X_LE,
        y_t0: T0_Y_LE,
        x_t0_pow: to_le_32(&tpow_aff.x),
        y_t0_pow: to_le_32(&tpow_aff.y),
        steps,
    })
}

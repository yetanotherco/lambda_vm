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
use num_integer::Integer;
use num_traits::{Signed, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
    /// `(yR - p) mod 2^256`, the addend that forces `yR < p`.
    pub y_r_sub_p: [u8; 32],
    /// `(yG - p) mod 2^256`, the addend that forces `yG < p`.
    ///
    /// Both are needed because `yR` and `yG` are published to guest memory. The byte range
    /// checks alone bound them by `2^256`, and the quotient columns absorb a multiple of `p`,
    /// so a witness could publish `y + p` for any `y < 2^256 - p` (~2^32) — and such points
    /// are constructible, since `3 | p-1` makes cubing 3-to-1, so a third of small `y` have a
    /// curve `x`. `y + p` carries the opposite parity, which is exactly what the caller reads
    /// to resolve the root.
    pub y_g_sub_p: [u8; 32],
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
    let (q, rem) = numerator.div_rem(p_big);
    assert!(
        rem.is_zero(),
        "ECSM witness {relation}: numerator not divisible by p"
    );
    let q = r_big + q;
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
    let y_r_sub_p = to_le_32(&((&two_256 + &result.y) - p()));
    let y_g_sub_p = to_le_32(&((&two_256 + &g.y) - p()));

    // Steps are independent witnesses (each builds its own λ/quotient/carry data
    // from one StepPts), so they parallelize freely when rayon is available.
    #[cfg(feature = "parallel")]
    let steps = steps_pts
        .par_iter()
        .map(|s| build_step(s, &p_big, &r_big, &r_ext, &pp))
        .collect();
    #[cfg(not(feature = "parallel"))]
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
        y_r_sub_p,
        y_g_sub_p,
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

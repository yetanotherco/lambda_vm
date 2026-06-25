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

use crypto_bigint::{Encoding, NonZero, U256, U512, U1024};

/// Widens a U512 to U1024 by zero-extending.
#[inline(always)]
fn w1024(v: U512) -> U1024 {
    let mut b = [0u8; 128];
    b[..64].copy_from_slice(&v.to_le_bytes());
    U1024::from_le_slice(&b)
}

/// `p²` as a `U1024` constant (`p = 2^256 - 2^32 - 977`).
pub(crate) const P_SQ: U1024 = U1024::from_be_hex(
    "0000000000000000000000000000000000000000000000000000000000000000\
     0000000000000000000000000000000000000000000000000000000000000000\
     fffffffffffffffffffffffffffffffffffffffffffffffffffffffdfffff85e\
     000000000000000000000000000000000000000000000001000007a2000e90a1",
);

/// `3p²` as a `U1024` constant.
pub(crate) const THREE_P_SQ: U1024 = U1024::from_be_hex(
    "0000000000000000000000000000000000000000000000000000000000000000\
     0000000000000000000000000000000000000000000000000000000000000002\
     fffffffffffffffffffffffffffffffffffffffffffffffffffffff9ffffe91a\
     000000000000000000000000000000000000000000000003000016e6002bb1e3",
);

use crate::curve::{StepPts, replay_double_and_add};
use crate::{B, EcsmError, P_BYTES, R_BYTES, n, p, prepare};

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
// U512 helpers
// =========================================================================

/// Extracts the low 33 little-endian bytes of a U512 as a `[u8; 33]`.
/// Asserts the value fits (bytes 33–63 are zero).
fn to_le_33(relation: &str, v: &U512) -> [u8; 33] {
    let bytes = v.to_le_bytes(); // [u8; 64]
    assert!(
        bytes[33..].iter().all(|&b| b == 0),
        "ECSM witness {relation}: quotient exceeds 33 bytes"
    );
    let mut out = [0u8; 33];
    out.copy_from_slice(&bytes[..33]);
    out
}

/// Computes `r_offset + (pos - neg) / p` where `pos - neg` is divisible by `p`.
///
/// Both `pos` and `neg` are U512 (products of 256-bit values, each < p²).
/// Since `pos ≡ neg mod p`, dividing each by p gives quotients whose difference
/// equals `(pos - neg)/p`. Handles the case where `pos < neg` (signed result) by
/// using `r_offset` (= p or 3p) to keep the final result positive.
/// Computes `(offset + pos - neg) / p` as U1024 arithmetic, returns quotient as U512.
/// `offset` is `p²` or `3p²` (chosen so the expression is always positive).
fn shifted_quotient(
    relation: &str,
    pos: U1024,
    neg: U1024,
    p1024: &NonZero<U1024>,
    offset: U1024,
) -> U512 {
    let total = offset.wrapping_add(&pos).wrapping_sub(&neg);
    let (q, r) = total.div_rem(p1024);
    assert!(r == U1024::ZERO, "ECSM witness {relation}: numerator not divisible by p");
    let q_bytes = q.to_le_bytes();
    assert!(q_bytes[64..].iter().all(|&b| b == 0), "ECSM witness {relation}: quotient exceeds U512");
    U512::from_le_slice(&q_bytes[..64])
}

// =========================================================================
// Witness construction
// =========================================================================

/// Computes the full ECSM/ECDAS witness for `k·G` over secp256k1, given `k` and `xG` as
/// little-endian 32-byte values. This is the prover's entry point.
pub fn compute_witness(k_le: &[u8; 32], xg_le: &[u8; 32]) -> Result<EcsmWitness, EcsmError> {
    let (k, g) = prepare(k_le, xg_le)?;

    // NonZero<U512> of p for the x2 div_rem; NonZero<U1024> of p for shifted_quotient.
    let mut p_le64 = [0u8; 64];
    p_le64[..32].copy_from_slice(&P_BYTES);
    let p512 = NonZero::new(U512::from_le_slice(&p_le64)).expect("p != 0");
    let mut p_le128 = [0u8; 128];
    p_le128[..32].copy_from_slice(&P_BYTES);
    let p1024 = NonZero::new(U1024::from_le_slice(&p_le128)).expect("p != 0");

    // Common zero-extended constants for carry builders.
    let pp = ext64(&P_BYTES);
    let r_ext = ext64(&R_BYTES);
    let b_bytes = {
        let mut a = [0u8; 32];
        a[0] = B as u8;
        a
    };
    let b_ext = ext64(&b_bytes);

    // --- ECSM: x2 = xG^2 mod p, quotient q0 ---
    // xg_sq = xG * xG as U512 (widening multiply).
    let (xg_sq_lo, xg_sq_hi) = g.x.mul_wide(&g.x);
    let xg_sq = xg_sq_hi.concat(&xg_sq_lo);
    let (q0_512, x2_512) = xg_sq.div_rem(&p512);
    let x2 = U256::from_le_slice(&x2_512.to_le_bytes()[..32]);
    let q0 = U256::from_le_slice(&q0_512.to_le_bytes()[..32]);
    let xg_b = g.x.to_le_bytes();
    let yg_b = g.y.to_le_bytes();
    let x2_b = x2.to_le_bytes();
    let q0_b = q0.to_le_bytes();
    let c0 = carries_x2(&ext64(&xg_b), &ext64(&x2_b), &ext64(&q0_b), &pp);

    // --- ECSM: yG relation, quotient q1 = (yG^2 − xG·x2 − b)/p + p ---
    // pos = yG^2, neg = xG·x2 + b. r_offset = p.
    let (yg_sq_lo, yg_sq_hi) = g.y.mul_wide(&g.y);
    let yg_sq = yg_sq_hi.concat(&yg_sq_lo);
    let (xg_x2_lo, xg_x2_hi) = g.x.mul_wide(&x2);
    let xg_x2 = xg_x2_hi.concat(&xg_x2_lo);
    let neg_yg = xg_x2.wrapping_add(&U512::from(B));
    let q1_512 = shifted_quotient("yG", w1024(yg_sq), w1024(neg_yg), &p1024, P_SQ);
    let q1_b = to_le_33("yG", &q1_512);
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
    // k_sub_n = (k - N) mod 2^256; since k < N this wraps: 2^256 + k - N.
    let k_sub_n = k.wrapping_sub(&n()).to_le_bytes();

    // --- double/add replay ---
    let (steps_pts, result) = replay_double_and_add(&k, &g);
    let x_r = result.x.to_le_bytes();
    let y_r = result.y.to_le_bytes();
    // x_r_sub_p = (xR - p) mod 2^256; since xR < p this wraps: 2^256 + xR - p.
    let x_r_sub_p = result.x.wrapping_sub(&p()).to_le_bytes();

    let steps = steps_pts
        .iter()
        .map(|s| build_step(s, &p1024, THREE_P_SQ, &r_ext, &pp))
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
    p1024: &NonZero<U1024>,
    r_offset: U1024,
    r_ext: &[i128; 64],
    pp: &[i128; 64],
) -> EcdasStep {
    // λ is precomputed (batched) during the double-and-add replay.
    let lam_b = s.lambda.to_le_bytes();
    let xa_b = s.a.x.to_le_bytes();
    let ya_b = s.a.y.to_le_bytes();
    let xg_b = s.g.x.to_le_bytes();
    let yg_b = s.g.y.to_le_bytes();
    let xr_b = s.r.x.to_le_bytes();
    let yr_b = s.r.y.to_le_bytes();

    let (lam_ext, xa_ext, ya_ext, xg_ext, yg_ext, xr_ext, yr_ext) = (
        ext64(&lam_b),
        ext64(&xa_b),
        ext64(&ya_b),
        ext64(&xg_b),
        ext64(&yg_b),
        ext64(&xr_b),
        ext64(&yr_b),
    );

    // Multiply two U256 values, result as U512.
    let mul512 = |a: &U256, b: &U256| -> U512 {
        let (lo, hi) = a.mul_wide(b);
        hi.concat(&lo)
    };

    // Widen a U256 to U512 (zero-extend into high half).
    let wide = |v: &U256| -> U512 {
        let (lo, hi) = (v, &U256::ZERO);
        hi.concat(lo)
    };

    // q0: λ relation.
    //   add: num0 = λ*xG + yA  -  λ*xA - yG     (= (xG-xA)*λ - yG + yA)
    //   dbl: num0 = 2*λ*yA     -  3*xA²
    // Products fit in U512; sums (2x or 3x) are done in U1024 to avoid overflow.
    let (pos0, neg0) = if s.op == 1 {
        let lam_xg = w1024(mul512(&s.lambda, &s.g.x));
        let lam_xa = w1024(mul512(&s.lambda, &s.a.x));
        (lam_xg.wrapping_add(&w1024(wide(&s.a.y))), lam_xa.wrapping_add(&w1024(wide(&s.g.y))))
    } else {
        let lam_ya = w1024(mul512(&s.lambda, &s.a.y));
        let xa_sq  = w1024(mul512(&s.a.x, &s.a.x));
        (lam_ya.wrapping_add(&lam_ya), xa_sq.wrapping_add(&xa_sq).wrapping_add(&xa_sq))
    };
    let q0_512 = shifted_quotient("lambda", pos0, neg0, p1024, r_offset);
    let q0_b = to_le_33("lambda", &q0_512);

    // q1: xR relation.
    //   add: num1 = λ²  -  xA + xG + xR
    //   dbl: num1 = λ²  -  2*xA + xR
    let lam_sq = mul512(&s.lambda, &s.lambda);
    let (pos1, neg1) = if s.op == 1 {
        (lam_sq, wide(&s.a.x).wrapping_add(&wide(&s.g.x)).wrapping_add(&wide(&s.r.x)))
    } else {
        (lam_sq, wide(&s.a.x).wrapping_add(&wide(&s.a.x)).wrapping_add(&wide(&s.r.x)))
    };
    let q1_512 = shifted_quotient("xR", w1024(pos1), w1024(neg1), p1024, r_offset);
    let q1_b = to_le_33("xR", &q1_512);

    // q2: yR relation — num2 = λ*(xA - xR) - yA - yR
    //   pos = λ*xA,  neg = λ*xR + yA + yR
    let lam_xa = w1024(mul512(&s.lambda, &s.a.x));
    let lam_xr = w1024(mul512(&s.lambda, &s.r.x));
    let neg2 = lam_xr.wrapping_add(&w1024(wide(&s.a.y))).wrapping_add(&w1024(wide(&s.r.y)));
    let q2_512 = shifted_quotient("yR", lam_xa, neg2, p1024, r_offset);
    let q2_b = to_le_33("yR", &q2_512);

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

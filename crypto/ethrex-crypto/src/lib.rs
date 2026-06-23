//! LambdaVM crypto provider for ethrex's EVM.
//!
//! Implements ethrex's `Crypto` trait with LambdaVM precompile acceleration and
//! is injected into the ethrex guest (`Arc::new(LambdaVmEcsmCrypto)` →
//! `execution_program`). Living in the lambda_vm repo (not in ethrex) means
//! accelerator changes don't require an ethrex PR.
//!
//! Accelerated today:
//! - `keccak256`: a sponge over the `keccak_permute` precompile (riscv64; on
//!   host it falls back to software keccak for tests).
//! - `secp256k1_ecrecover`: the ECDSA recovery's 2-term linear combination is
//!   evaluated through the ECSM `ecsm_mul` precompile (riscv64), reconstructing
//!   the full point from x-only queries; on host / degenerate inputs it falls
//!   back to the pure-Rust `ProjectivePoint::lincomb`.
//!
//! Every other `Crypto` method inherits the trait default (vetted pure-Rust
//! crates: `ark-bn254`, `bls12_381`, `p256`, `sha2`, `ripemd`, …).

use ethrex_crypto::keccak::keccak_hash;
use ethrex_crypto::{Crypto, CryptoError};
use k256::elliptic_curve::group::prime::PrimeCurveAffine;
use k256::elliptic_curve::ops::{Invert, LinearCombination, Reduce};
use k256::elliptic_curve::point::DecompressPoint;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::elliptic_curve::PrimeField;
use k256::{AffinePoint, FieldBytes, ProjectivePoint, Scalar, U256};

// Used only by the x-only point reconstruction (riscv accelerated path + the
// host unit tests); unused on a non-test host build.
#[cfg(any(target_arch = "riscv64", test))]
use k256::elliptic_curve::sec1::FromEncodedPoint;
#[cfg(any(target_arch = "riscv64", test))]
use k256::{EncodedPoint, FieldElement};

/// LambdaVM crypto provider — inject via `Arc::new(LambdaVmEcsmCrypto)`.
#[derive(Debug)]
pub struct LambdaVmEcsmCrypto;

impl Crypto for LambdaVmEcsmCrypto {
    fn secp256k1_ecrecover(
        &self,
        sig: &[u8; 64],
        recid: u8,
        msg: &[u8; 32],
    ) -> Result<[u8; 32], CryptoError> {
        ecsm_ecrecover(sig, recid, msg)
    }

    fn keccak256(&self, input: &[u8]) -> [u8; 32] {
        // riscv64 guest: sponge over the keccak_permute precompile.
        #[cfg(target_arch = "riscv64")]
        return keccak256_via_lambdavm(input);
        // host (tests / non-guest): software keccak — the precompile syscall
        // isn't available off-target.
        #[cfg(not(target_arch = "riscv64"))]
        return keccak_hash(input);
    }
}

// ── ECDSA secp256k1 recovery via the ECSM precompile ────────────────────────

/// Recover the keccak hash of the uncompressed public key from a 64-byte
/// signature, recovery id, and 32-byte message hash. Used by the ECRECOVER
/// precompile (0x01).
///
/// Mirrors the pure-Rust recovery in the `Crypto` trait default
/// (`pk = r⁻¹·(s·R − z·G)`), but evaluates the 2-term linear combination
/// `lincomb(G, u1, R, u2)` through the ECSM accelerator via [`ecsm_lincomb2`],
/// falling back to the software `ProjectivePoint::lincomb` whenever the
/// accelerated path declines (degenerate scalars/points, or non-riscv builds).
/// We compute the recovery directly rather than calling k256's
/// `recover_from_prehash`, which internally runs a *second* lincomb to
/// re-verify the key — doubling the ECSM ecalls for no gain here.
fn ecsm_ecrecover(sig: &[u8; 64], recid: u8, msg: &[u8; 32]) -> Result<[u8; 32], CryptoError> {
    let r_bytes = <&FieldBytes>::from(&sig[..32]);
    let s_bytes = <&FieldBytes>::from(&sig[32..]);

    // Parse r and s as scalars, rejecting values >= the curve order.
    let r: Option<Scalar> = Scalar::from_repr(*r_bytes).into();
    let s: Option<Scalar> = Scalar::from_repr(*s_bytes).into();
    let (Some(r), Some(s)) = (r, s) else {
        return Err(CryptoError::InvalidSignature);
    };
    if r.is_zero().into() || s.is_zero().into() {
        return Err(CryptoError::InvalidSignature);
    }

    // Decompress R from r and the recovery-id parity bit.
    // recid >= 2 (R.x = r + n) has ~2^-128 probability and never occurs for the
    // precompile; we don't handle it (decompression simply fails), matching the
    // trait default.
    let y_is_odd = (recid & 1) != 0;
    let r_point: Option<AffinePoint> =
        AffinePoint::decompress(r_bytes, u8::from(y_is_odd).into()).into();
    let Some(r_point) = r_point else {
        return Err(CryptoError::RecoveryFailed);
    };
    let r_proj = ProjectivePoint::from(r_point);

    let z = <Scalar as Reduce<U256>>::reduce_bytes(&FieldBytes::from(*msg));
    let r_inv: Option<Scalar> = r.invert_vartime().into();
    let Some(r_inv) = r_inv else {
        return Err(CryptoError::RecoveryFailed);
    };
    let u1 = -(r_inv * z);
    let u2 = r_inv * s;

    // pk = u1·G + u2·R, accelerated via ECSM with a software fallback.
    let g = ProjectivePoint::GENERATOR;
    let pk = ecsm_lincomb2(&g, &u1, &r_proj, &u2)
        .unwrap_or_else(|| ProjectivePoint::lincomb(&g, &u1, &r_proj, &u2));

    let pk_affine = pk.to_affine();
    if bool::from(pk_affine.is_identity()) {
        return Err(CryptoError::RecoveryFailed);
    }

    // SEC1 uncompressed: 0x04 || X(32) || Y(32). The address is keccak(X || Y).
    let uncompressed = pk_affine.to_encoded_point(false);
    Ok(keccak_hash(&uncompressed.as_bytes()[1..65]))
}

/// ECSM-accelerated 2-term linear combination `k1·P1 + k2·P2`.
///
/// On riscv64 this reconstructs the full affine result from four x-only ECSM
/// queries (see [`lincomb2_with_oracle`]); on other targets, and whenever a
/// degenerate-configuration guard trips, it returns `None` so the caller uses
/// the pure-Rust `ProjectivePoint::lincomb`.
#[cfg(target_arch = "riscv64")]
fn ecsm_lincomb2(
    p1: &ProjectivePoint,
    k1: &Scalar,
    p2: &ProjectivePoint,
    k2: &Scalar,
) -> Option<ProjectivePoint> {
    lincomb2_with_oracle(p1, k1, p2, k2, ecsm_oracle)
}

#[cfg(not(target_arch = "riscv64"))]
fn ecsm_lincomb2(
    _p1: &ProjectivePoint,
    _k1: &Scalar,
    _p2: &ProjectivePoint,
    _k2: &Scalar,
) -> Option<ProjectivePoint> {
    None
}

/// x-only scalar-mul oracle backed by the ECSM precompile: computes `x(k·P)`
/// for the curve point P whose x-coordinate is passed in. `x` must be the
/// x-coordinate of a curve point and `k` in `(0, N)` (N = curve order) —
/// guaranteed by the guards in [`lincomb2_with_oracle`]. Values cross the ABI
/// as 32-byte little-endian; `x_le` and `k_le` are distinct stack arrays so
/// the executor's `|addr_x_le − addr_k_le| ≥ 32` assumption holds by
/// construction.
#[cfg(target_arch = "riscv64")]
fn ecsm_oracle(x: &FieldElement, k: &Scalar) -> Option<FieldElement> {
    let x_be = x.to_bytes();
    let k_be = k.to_bytes();
    let mut x_le = [0u8; 32];
    let mut k_le = [0u8; 32];
    for i in 0..32 {
        x_le[i] = x_be[31 - i];
        k_le[i] = k_be[31 - i];
    }
    let mut xr_le = [0u8; 32];
    lambda_vm_syscalls::syscalls::ecsm_mul(&mut xr_le, &x_le, &k_le);
    xr_le.reverse();
    Option::from(FieldElement::from_bytes(&xr_le.into()))
}

/// Computes `k1·P1 + k2·P2` from four x-only oracle queries, or `None` if any
/// degenerate-configuration guard trips.
///
/// The lambda-vm ECSM precompile returns only `x(k·P)`. For `A = k1·P1` with
/// `P1 = (xp, yp)` fully known, query `xa = x(k1·P1)` and `xc = x((k1+1)·P1)`.
/// The chord-addition law gives `λ² = xc + xa + xp =: t` and `ya = yp + λ·dx`
/// with `dx = xa − xp`; substituting into `ya² = xa³ + b` makes λ *linear*:
/// `λ = (xa³ − xp³ − t·dx²) / (2·yp·dx)`. The wrong sign `−ya` would force
/// `x((k1−1)·P1) = xc`, i.e. `k1 ≡ 0` or `2·k1 ≡ 0 (mod n)`, excluded by the
/// scalar guards. x-only queries are parity-invariant (`x(k·P) = x(k·(−P))`),
/// so the precompile's canonical-y lift never matters. Same for `B = k2·P2`,
/// then `Q = A + B` is one affine addition. All three inversions are batched.
///
/// Generic over the oracle so unit tests can substitute a software stand-in.
#[cfg(any(target_arch = "riscv64", test))]
fn lincomb2_with_oracle<O>(
    p1: &ProjectivePoint,
    k1: &Scalar,
    p2: &ProjectivePoint,
    k2: &Scalar,
    oracle: O,
) -> Option<ProjectivePoint>
where
    O: Fn(&FieldElement, &Scalar) -> Option<FieldElement>,
{
    let a1 = p1.to_affine();
    let a2 = p2.to_affine();
    if bool::from(a1.is_identity()) || bool::from(a2.is_identity()) {
        return None;
    }
    if scalar_near_edge(k1) || scalar_near_edge(k2) {
        return None;
    }

    let (x1, y1) = affine_xy(&a1)?;
    let (x2, y2) = affine_xy(&a2)?;

    let xa = oracle(&x1, k1)?;
    let xc1 = oracle(&x1, &(*k1 + Scalar::ONE))?;
    let xb = oracle(&x2, k2)?;
    let xc2 = oracle(&x2, &(*k2 + Scalar::ONE))?;

    let dx1 = (xa - x1).normalize();
    let dx2 = (xb - x2).normalize();
    let dxq = (xb - xa).normalize();
    if bool::from(dx1.is_zero()) || bool::from(dx2.is_zero()) || bool::from(dxq.is_zero()) {
        return None;
    }

    // One shared inversion for the two λ denominators and the final chord.
    let den1 = y1.double() * dx1;
    let den2 = y2.double() * dx2;
    let inv = Option::<FieldElement>::from((den1 * den2 * dxq).invert())?;
    let inv_den1 = inv * den2 * dxq;
    let inv_den2 = inv * den1 * dxq;
    let inv_dxq = inv * den1 * den2;

    let ya = solve_y(&x1, &y1, &xa, &xc1, &dx1, &inv_den1)?;
    let yb = solve_y(&x2, &y2, &xb, &xc2, &dx2, &inv_den2)?;

    // Q = A + B, with A ≠ ±B ensured by dxq ≠ 0.
    let lq = (yb - ya) * inv_dxq;
    let xq = (lq.square() - xa - xb).normalize();
    let yq = (lq * (xa - xq) - ya).normalize();

    // `point_from_xy` checks the result is on the curve as a cheap backstop:
    // it rejects gross off-curve garbage and falls back to software, but
    // correctness rests on the algebra above — an on-curve-but-wrong point
    // would still pass this check.
    point_from_xy(&xq, &yq)
}

/// Recovers `y(k·P)` for `P = (xp, yp)` from `xa = x(k·P)` and
/// `xc = x((k+1)·P)`, given `dx = xa − xp` and `inv_den = (2·yp·dx)⁻¹`.
/// `None` if the λ² consistency check fails (degenerate configuration).
#[cfg(any(target_arch = "riscv64", test))]
fn solve_y(
    xp: &FieldElement,
    yp: &FieldElement,
    xa: &FieldElement,
    xc: &FieldElement,
    dx: &FieldElement,
    inv_den: &FieldElement,
) -> Option<FieldElement> {
    let t = *xc + xa + xp;
    let xa3 = xa.square() * xa;
    let xp3 = xp.square() * xp;
    let lambda = (xa3 - xp3 - t * dx.square()) * inv_den;
    if lambda.square().normalize() != t.normalize() {
        return None;
    }
    Some((*yp + lambda * dx).normalize())
}

/// `k ∈ {0, 1, n−1}`: cases where `k` or `k+1` is an invalid ecall scalar or
/// the chord algebra degenerates (`A = ±P`).
#[cfg(any(target_arch = "riscv64", test))]
fn scalar_near_edge(k: &Scalar) -> bool {
    use k256::elliptic_curve::subtle::ConstantTimeEq;
    bool::from(k.is_zero())
        || bool::from(k.ct_eq(&Scalar::ONE))
        || bool::from(k.ct_eq(&(-Scalar::ONE)))
}

/// Affine `(x, y)` of a non-identity point as field elements, via its SEC1
/// uncompressed encoding (k256 keeps `AffinePoint`'s coordinate fields private).
#[cfg(any(target_arch = "riscv64", test))]
fn affine_xy(p: &AffinePoint) -> Option<(FieldElement, FieldElement)> {
    let ep = p.to_encoded_point(false);
    let x = Option::<FieldElement>::from(FieldElement::from_bytes(ep.x()?))?;
    let y = Option::<FieldElement>::from(FieldElement::from_bytes(ep.y()?))?;
    Some((x, y))
}

/// Builds a curve point from affine coordinates, returning `None` if the point
/// is not on the curve (`AffinePoint::from_encoded_point` validates this).
#[cfg(any(target_arch = "riscv64", test))]
fn point_from_xy(x: &FieldElement, y: &FieldElement) -> Option<ProjectivePoint> {
    let ep = EncodedPoint::from_affine_coordinates(&x.to_bytes(), &y.to_bytes(), false);
    let affine = Option::<AffinePoint>::from(AffinePoint::from_encoded_point(&ep))?;
    Some(ProjectivePoint::from(affine))
}

// ── Keccak-256 over the keccak_permute precompile (riscv64 guest) ───────────

/// Keccak-256 sponge with an injected permutation function.
///
/// Keccak-f[1600], rate 1088 bits (136 bytes), capacity 512 bits.
/// Padding: `0x01 ... 0x80` (multi-rate, last bit set). The state is a
/// 25-element u64 array; bytes are absorbed into the state via little-endian
/// XOR (matching the standard Keccak byte-to-lane mapping).
///
/// Gated to `riscv64 | test` so the generic function is available to the host
/// unit tests without being dead code in the non-test host build.
#[cfg(any(target_arch = "riscv64", test))]
fn keccak256_with_permute<F: FnMut(&mut [u64; 25])>(input: &[u8], mut permute: F) -> [u8; 32] {
    const RATE: usize = 136;

    let mut state = [0u64; 25];
    let mut offset = 0;

    while input.len().saturating_sub(offset) >= RATE {
        absorb_block(&mut state, &input[offset..offset + RATE]);
        permute(&mut state);
        offset = offset.saturating_add(RATE);
    }

    // Final block with multi-rate padding.
    let mut last = [0u8; RATE];
    let remaining = input.len().saturating_sub(offset);
    if let Some(tail) = last.get_mut(..remaining) {
        if let Some(src) = input.get(offset..) {
            tail.copy_from_slice(src);
        }
    }
    if let Some(b) = last.get_mut(remaining) {
        *b ^= 0x01;
    }
    if let Some(b) = last.get_mut(RATE - 1) {
        *b ^= 0x80;
    }
    absorb_block(&mut state, &last);
    permute(&mut state);

    // Squeeze the first 32 bytes (four lanes) as little-endian.
    let mut output = [0u8; 32];
    for (i, lane) in state.iter().take(4).enumerate() {
        let bytes = lane.to_le_bytes();
        let start = i.saturating_mul(8);
        if let Some(dst) = output.get_mut(start..start.saturating_add(8)) {
            dst.copy_from_slice(&bytes);
        }
    }
    output
}

/// Keccak-256 via LambdaVM's `keccak_permute` syscall (riscv64 guest only).
#[cfg(target_arch = "riscv64")]
fn keccak256_via_lambdavm(input: &[u8]) -> [u8; 32] {
    keccak256_with_permute(input, |s| lambda_vm_syscalls::syscalls::keccak_permute(s))
}

/// XOR one rate-sized block of bytes into the state lanes (little-endian).
#[cfg(any(target_arch = "riscv64", test))]
fn absorb_block(state: &mut [u64; 25], block: &[u8]) {
    for (lane, chunk) in state.iter_mut().zip(block.chunks_exact(8)) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        *lane ^= u64::from_le_bytes(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// secp256k1 curve constant `b = 7`.
    fn curve_b() -> FieldElement {
        let mut bytes = [0u8; 32];
        bytes[31] = 7;
        FieldElement::from_bytes(&bytes.into()).unwrap()
    }

    /// Software stand-in for the ECSM precompile: lift `x` to a curve point and
    /// return `x(k·P)` (parity-invariant, like the real ecall).
    fn soft_oracle(x: &FieldElement, k: &Scalar) -> Option<FieldElement> {
        let xn = x.normalize();
        let y2 = (xn.square() * xn + curve_b()).normalize();
        let y = Option::<FieldElement>::from(y2.sqrt())?;
        let p = point_from_xy(&xn, &y.normalize())?;
        let prod = (p * k).to_affine();
        Some(affine_xy(&prod)?.0)
    }

    fn g_times(n: u64) -> ProjectivePoint {
        ProjectivePoint::GENERATOR * Scalar::from(n)
    }

    #[test]
    fn matches_software_lincomb_on_fixed_inputs() {
        let cases = [
            (g_times(3), 123_456_789u64, g_times(7), 987_654_321u64),
            (g_times(11), 2u64.pow(20) + 5, g_times(2), 42u64),
            (ProjectivePoint::GENERATOR, 7u64, g_times(5), 9u64),
        ];
        for (p1, k1, p2, k2) in cases {
            let (k1, k2) = (Scalar::from(k1), Scalar::from(k2));
            let expected = ProjectivePoint::lincomb(&p1, &k1, &p2, &k2);
            let got = lincomb2_with_oracle(&p1, &k1, &p2, &k2, soft_oracle)
                .expect("non-degenerate inputs must reconstruct");
            assert_eq!(got.to_affine(), expected.to_affine());
        }
    }

    #[test]
    fn matches_software_lincomb_on_recovery_shape() {
        // u1·G + u2·R, generator first, like ECDSA recovery.
        let g = ProjectivePoint::GENERATOR;
        let r = g_times(0x1234);
        let u1 = Scalar::from(0xdead_beefu64);
        let u2 = Scalar::from(0x0bad_f00du64);
        let expected = ProjectivePoint::lincomb(&g, &u1, &r, &u2);
        let got = lincomb2_with_oracle(&g, &u1, &r, &u2, soft_oracle)
            .expect("non-degenerate inputs must reconstruct");
        assert_eq!(got.to_affine(), expected.to_affine());
    }

    #[test]
    fn edge_scalars_fall_back() {
        let p1 = g_times(3);
        let p2 = g_times(5);
        let ok = Scalar::from(12345u64);
        for bad in [Scalar::ZERO, Scalar::ONE, -Scalar::ONE] {
            assert!(lincomb2_with_oracle(&p1, &bad, &p2, &ok, soft_oracle).is_none());
            assert!(lincomb2_with_oracle(&p1, &ok, &p2, &bad, soft_oracle).is_none());
        }
    }

    #[test]
    fn identity_points_fall_back() {
        let p = g_times(3);
        let k = Scalar::from(7u64);
        let id = ProjectivePoint::IDENTITY;
        assert!(lincomb2_with_oracle(&id, &k, &p, &k, soft_oracle).is_none());
        assert!(lincomb2_with_oracle(&p, &k, &id, &k, soft_oracle).is_none());
    }

    #[test]
    fn cancelling_and_doubling_terms_fall_back() {
        let p = g_times(3);
        let k = Scalar::from(7u64);
        // A = B (doubling chord) and A = −B (Q = O): both share x(A) = x(B).
        assert!(lincomb2_with_oracle(&p, &k, &p, &k, soft_oracle).is_none());
        assert!(lincomb2_with_oracle(&p, &k, &(-p), &k, soft_oracle).is_none());
    }

    // ── Issue 1: known-answer tests for the full ecsm_ecrecover path ─────────

    /// Build a valid ECDSA/secp256k1 signature from (d, kk, msg) using only the
    /// k256 primitives already imported and return `(sig, recid, expected_addr)`.
    ///
    /// `expected_addr` = keccak(X‖Y) of the uncompressed public key, exactly as
    /// `ecsm_ecrecover` computes it.
    fn make_ecdsa_fixture(
        d: Scalar,
        kk: Scalar,
        msg: [u8; 32],
    ) -> ([u8; 64], u8, [u8; 32]) {
        assert!(!bool::from(d.is_zero()), "private key must be nonzero");
        assert!(!bool::from(kk.is_zero()), "nonce must be nonzero");

        // Public key Q = d·G.
        let q = (ProjectivePoint::GENERATOR * d).to_affine();
        let q_uncompressed = q.to_encoded_point(false);
        let expected = keccak_hash(&q_uncompressed.as_bytes()[1..65]);

        // R = kk·G; r = reduce(Rx); assert r ≠ 0.
        let r_point = (ProjectivePoint::GENERATOR * kk).to_affine();
        let (rx, ry) = affine_xy(&r_point).expect("R is not identity");
        let r = <Scalar as Reduce<U256>>::reduce_bytes(&rx.to_bytes());
        assert!(!bool::from(r.is_zero()), "r must be nonzero");

        // recid parity: low bit of Ry (big-endian, byte 31).
        let recid = ry.normalize().to_bytes()[31] & 1;

        // z = reduce(msg).
        let z = <Scalar as Reduce<U256>>::reduce_bytes(&FieldBytes::from(msg));

        // s = kk⁻¹ · (z + r·d).
        let s = kk.invert_vartime().expect("kk is nonzero") * (z + r * d);
        assert!(!bool::from(s.is_zero()), "s must be nonzero");

        // sig = r (BE, 32 bytes) ‖ s (BE, 32 bytes).
        let mut sig = [0u8; 64];
        sig[..32].copy_from_slice(&r.to_bytes());
        sig[32..].copy_from_slice(&s.to_bytes());

        (sig, recid, expected)
    }

    #[test]
    fn ecrecover_known_answer_three_tuples() {
        // Three distinct (d, kk, msg) tuples — deterministic, no RNG.
        let tuples: &[(u64, u64, [u8; 32])] = &[
            (
                0x0000_0000_0000_0001u64,
                0x0000_0000_dead_beefu64,
                {
                    let mut m = [0u8; 32];
                    m[31] = 0x42;
                    m
                },
            ),
            (
                0x00c0_ffee_dead_beef_u64,
                0x0123_4567_89ab_cdef_u64,
                {
                    let mut m = [0u8; 32];
                    m[0] = 0xff;
                    m[31] = 0x01;
                    m
                },
            ),
            (
                0x0bad_f00d_1337_cafe,
                0xfeed_face_0000_0001,
                {
                    let mut m = [0u8; 32];
                    for (i, b) in m.iter_mut().enumerate() {
                        *b = i as u8;
                    }
                    m
                },
            ),
        ];

        for &(d_u64, kk_u64, msg) in tuples {
            let d = Scalar::from(d_u64);
            let kk = Scalar::from(kk_u64);
            let (sig, recid, expected) = make_ecdsa_fixture(d, kk, msg);
            match ecsm_ecrecover(&sig, recid, &msg) {
                Ok(got) => assert_eq!(
                    got, expected,
                    "ecrecover returned wrong address for d={d_u64:#x} kk={kk_u64:#x}"
                ),
                Err(e) => panic!(
                    "ecrecover failed for d={d_u64:#x} kk={kk_u64:#x}: {e:?}"
                ),
            }
        }
    }

    #[test]
    fn ecrecover_rejects_zero_s() {
        // sig = valid r ‖ 0x00..00 (s = 0) must return InvalidSignature.
        let mut sig = [0u8; 64];
        // r = 1 (nonzero, but s = 0 in the second half).
        sig[31] = 0x01;
        let msg = [0u8; 32];
        assert!(
            matches!(ecsm_ecrecover(&sig, 0, &msg), Err(CryptoError::InvalidSignature)),
            "expected InvalidSignature for zero s"
        );
    }

    #[test]
    fn ecrecover_rejects_zero_r() {
        // sig = 0x00..00 ‖ valid s must return InvalidSignature.
        let mut sig = [0u8; 64];
        sig[63] = 0x01; // s = 1, r = 0
        let msg = [0u8; 32];
        assert!(
            matches!(ecsm_ecrecover(&sig, 0, &msg), Err(CryptoError::InvalidSignature)),
            "expected InvalidSignature for zero r"
        );
    }

    // ── Issue 2: host-side Keccak sponge tests via injected permutation ───────

    /// Cross-check our sponge body against the trusted `keccak` crate's f1600.
    fn check_keccak(input: &[u8]) {
        let got = keccak256_with_permute(input, keccak::f1600);
        let want = keccak_hash(input);
        assert_eq!(
            got, want,
            "keccak256 mismatch for {}-byte input",
            input.len()
        );
    }

    #[test]
    fn keccak_sponge_matches_trusted_permutation() {
        // Empty input.
        check_keccak(&[]);
        // One byte.
        check_keccak(&[0xab]);
        // 135 bytes — RATE-1: padding lands on byte 135 (0x01) and byte 135 is
        // also the last byte (0x80), so both bits land on the same byte: 0x81.
        check_keccak(&[0x5a; 135]);
        // Exactly RATE (136): fills one full block, final block is all-padding.
        check_keccak(&[0x3c; 136]);
        // RATE+1: one full block + one-byte remainder.
        check_keccak(&[0x7e; 137]);
        // Multi-block: ~1.5 × RATE (200 bytes), deterministic pattern.
        let long: Vec<u8> = (0u8..200).collect();
        check_keccak(&long);
    }
}

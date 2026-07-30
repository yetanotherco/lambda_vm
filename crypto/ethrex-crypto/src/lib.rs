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
use k256::elliptic_curve::ops::{LinearCombination, Reduce};
// `Invert` (software `x.invert()`) is only used by the host fallback; on the
// riscv64 guest all inversions go through the `hint` ecall.
#[cfg(not(target_arch = "riscv64"))]
use k256::elliptic_curve::ops::Invert;
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
        let pk_bytes = ecsm_ecrecover(sig, recid, msg)?;
        Ok(self.keccak256(&pk_bytes))
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

/// Obtain a 32-byte big-endian hint for `x_be` via the executor `hint` ecall
/// (the host computes the modular inverse / sqrt; the value is provable via the
/// prover's HINT table). The result is UNVERIFIED — every caller MUST check it
/// in-guest (`x·inv == 1`, `y² == x³+7`), since the ecall adds no correctness
/// constraint. BENCH scaffolding.
#[cfg(target_arch = "riscv64")]
fn get_hint(hint_id: usize, x_be: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    lambda_vm_syscalls::syscalls::hint(hint_id, &mut out, x_be);
    out
}

/// Scalar-field inverse `x⁻¹ mod n`. On riscv64 (guest) the inverse comes from the
/// `hint` ecall and we verify `x·inv == 1`; off-target (host tests) it computes the
/// inverse in software. BENCH scaffolding.
fn scalar_inv(x: &Scalar) -> Option<Scalar> {
    #[cfg(target_arch = "riscv64")]
    {
        use k256::elliptic_curve::subtle::ConstantTimeEq;
        let x_be: [u8; 32] = x.to_bytes().into();
        let inv_be = get_hint(lambda_vm_syscalls::syscalls::HINT_SCALAR_INV, &x_be);
        let inv: Scalar = Option::from(Scalar::from_repr(inv_be.into()))?;
        // Verify the untrusted hint: x·inv must equal 1 (mod n).
        if bool::from((*x * inv).ct_eq(&Scalar::ONE)) {
            Some(inv)
        } else {
            None
        }
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        x.invert_vartime().into()
    }
}

/// Decompress R from its x-coordinate + parity. On riscv64 the `y = sqrt(x³+7)`
/// is an `hint`-ecall value verified in-guest (`y² == x³+7`), with parity
/// selection; off-target it uses k256's software `decompress`. BENCH scaffolding.
fn decompress_r(r_bytes: &FieldBytes, y_is_odd: bool) -> Option<AffinePoint> {
    #[cfg(target_arch = "riscv64")]
    {
        let x: FieldElement = Option::from(FieldElement::from_bytes(r_bytes))?;
        // secp256k1: y² = x³ + 7.
        let mut seven_bytes = [0u8; 32];
        seven_bytes[31] = 7;
        let seven: FieldElement = Option::from(FieldElement::from_bytes(&seven_bytes.into()))?;
        let x3: FieldElement = x.square() * x;
        let rhs: FieldElement = x3 + seven;
        // Hinted sqrt (BE in/out), then verify y² == rhs canonically.
        let rhs_be: [u8; 32] = rhs.to_bytes().into();
        let y_be = get_hint(lambda_vm_syscalls::syscalls::HINT_FIELD_SQRT, &rhs_be);
        let mut y: FieldElement = Option::from(FieldElement::from_bytes(&y_be.into()))?;
        let y2: FieldElement = y.square();
        // Verify the untrusted root: y² must equal x³+7. Negate `y2`, not `rhs`:
        // `Neg` is `negate(1)` and only accepts magnitude 1, which `square()` always
        // returns, whereas `rhs` is a sum and carries magnitude 2 — negating it would
        // silently compute the wrong value in release, where the debug assert is gone.
        // (`ct_eq` is unusable here for the same reason as in `field_inv`.)
        if !bool::from((rhs + y2.negate(1)).normalizes_to_zero()) {
            return None;
        }
        // Select the root whose canonical LSB matches the requested parity.
        let y_odd = (y.to_bytes()[31] & 1) == 1;
        if y_odd != y_is_odd {
            y = -y;
        }
        // Build the affine point; `from_encoded_point` re-checks it's on-curve.
        let ep = EncodedPoint::from_affine_coordinates(&x.to_bytes(), &y.to_bytes(), false);
        Option::from(AffinePoint::from_encoded_point(&ep))
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        use k256::elliptic_curve::point::DecompressPoint;
        AffinePoint::decompress(r_bytes, u8::from(y_is_odd).into()).into()
    }
}

/// Recover the uncompressed public key bytes (X‖Y, 64 bytes) from a 64-byte
/// signature, recovery id, and 32-byte message hash. Used by the ECRECOVER
/// precompile (0x01).
///
/// Returns the raw 64-byte key; the caller is responsible for hashing it.
/// Keeping keccak out of this function lets `secp256k1_ecrecover` route the
/// hash through `self.keccak256`, which uses the keccak_permute precompile on
/// riscv64 instead of always falling back to software.
///
/// Mirrors the pure-Rust recovery in the `Crypto` trait default
/// (`pk = r⁻¹·(s·R − z·G)`), but evaluates the 2-term linear combination
/// `lincomb(G, u1, R, u2)` through the ECSM accelerator via [`ecsm_lincomb2`],
/// falling back to the software `ProjectivePoint::lincomb` whenever the
/// accelerated path declines (degenerate scalars/points, or non-riscv builds).
/// We compute the recovery directly rather than calling k256's
/// `recover_from_prehash`, which internally runs a *second* lincomb to
/// re-verify the key — doubling the ECSM ecalls for no gain here.
fn ecsm_ecrecover(sig: &[u8; 64], recid: u8, msg: &[u8; 32]) -> Result<[u8; 64], CryptoError> {
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
    let r_point: Option<AffinePoint> = decompress_r(r_bytes, y_is_odd);
    let Some(r_point) = r_point else {
        return Err(CryptoError::RecoveryFailed);
    };
    let r_proj = ProjectivePoint::from(r_point);

    let z = <Scalar as Reduce<U256>>::reduce_bytes(&FieldBytes::from(*msg));
    let r_inv: Option<Scalar> = scalar_inv(&r);
    let Some(r_inv) = r_inv else {
        return Err(CryptoError::RecoveryFailed);
    };
    let u1 = -(r_inv * z);
    let u2 = r_inv * s;

    // pk = u1·G + u2·R, accelerated via ECSM with a software fallback.
    // The ECSM path takes affine inputs and returns the affine result directly:
    // its inputs (G, R) and output are Z=1 points, so passing affines avoids the
    // wasteful projective→affine inversions (`to_affine` of a Z=1 point still runs
    // a full constant-time field inversion in k256). The rare software fallback
    // still converts via `to_affine`.
    let g = ProjectivePoint::GENERATOR;
    let pk_affine = ecsm_lincomb2(&AffinePoint::GENERATOR, &u1, &r_point, &u2)
        .unwrap_or_else(|| ProjectivePoint::lincomb(&g, &u1, &r_proj, &u2).to_affine());
    if bool::from(pk_affine.is_identity()) {
        return Err(CryptoError::RecoveryFailed);
    }

    // SEC1 uncompressed: 0x04 || X(32) || Y(32). Return X‖Y for the caller to hash.
    let uncompressed = pk_affine.to_encoded_point(false);
    let mut pk_bytes = [0u8; 64];
    pk_bytes.copy_from_slice(&uncompressed.as_bytes()[1..65]);
    Ok(pk_bytes)
}

/// ECSM-accelerated 2-term linear combination `k1·P1 + k2·P2`.
///
/// AFFINE PoC: on riscv64 this uses TWO affine ECSM queries (the precompile now
/// returns `(x, y)`, see [`lincomb2_with_oracle`]) instead of four x-only queries
/// plus chord-law y-reconstruction; on other targets, and whenever a guard trips,
/// it returns `None` so the caller uses the pure-Rust `ProjectivePoint::lincomb`.
#[cfg(target_arch = "riscv64")]
fn ecsm_lincomb2(
    a1: &AffinePoint,
    k1: &Scalar,
    a2: &AffinePoint,
    k2: &Scalar,
) -> Option<AffinePoint> {
    lincomb2_with_oracle(a1, k1, a2, k2, ecsm_oracle)
}

#[cfg(not(target_arch = "riscv64"))]
fn ecsm_lincomb2(
    _a1: &AffinePoint,
    _k1: &Scalar,
    _a2: &AffinePoint,
    _k2: &Scalar,
) -> Option<AffinePoint> {
    None
}

/// AFFINE oracle backed by the ECSM precompile: computes the full point `k·(x, y)` for the
/// caller's actual input point `(x, y)`. Returns `(xR, yR)` as normalized field elements —
/// no parity convention or sign flip, because the precompile receives the real `y` and the
/// prover pins it by a memory read. `(x, y)` must be a curve point and `k` in `(0, N)`.
/// Values cross the ABI as 32-byte little-endian; `input` is a 64-byte `[xG‖yG]` buffer,
/// `out` a 64-byte `[xR‖yR]` buffer, `k_le` a distinct 32-byte array (executor's
/// `|addr_input − addr_k| ≥ 64` disjointness assumption).
#[cfg(target_arch = "riscv64")]
fn ecsm_oracle(x: &FieldElement, y: &FieldElement, k: &Scalar) -> Option<(FieldElement, FieldElement)> {
    let x_be = x.to_bytes();
    let y_be = y.to_bytes();
    let k_be = k.to_bytes();
    let mut input = [0u8; 64];
    let mut k_le = [0u8; 32];
    for i in 0..32 {
        input[i] = x_be[31 - i];
        input[32 + i] = y_be[31 - i];
        k_le[i] = k_be[31 - i];
    }
    let mut out = [0u8; 64];
    lambda_vm_syscalls::syscalls::ecsm_mul_affine(&mut out, &input, &k_le);
    let mut xr_be = [0u8; 32];
    let mut yr_be = [0u8; 32];
    for i in 0..32 {
        xr_be[i] = out[31 - i];
        yr_be[i] = out[32 + 31 - i];
    }
    let xr = Option::<FieldElement>::from(FieldElement::from_bytes(&xr_be.into()))?;
    let yr = Option::<FieldElement>::from(FieldElement::from_bytes(&yr_be.into()))?;
    Some((xr.normalize(), yr.normalize()))
}

/// Base-field inverse `x⁻¹ mod p`. On riscv64 the host supplies it via the
/// `hint` ecall and we verify `x·inv == 1`; off-target it inverts in software.
#[cfg(any(target_arch = "riscv64", test))]
fn field_inv(x: &FieldElement) -> Option<FieldElement> {
    #[cfg(target_arch = "riscv64")]
    {
        let x_be: [u8; 32] = x.to_bytes().into();
        let inv_be = get_hint(lambda_vm_syscalls::syscalls::HINT_FIELD_INV, &x_be);
        let inv: FieldElement = Option::from(FieldElement::from_bytes(&inv_be.into()))?;
        // Verify the untrusted hint: x·inv must equal 1 (mod p). Compare by asking
        // whether the difference normalizes to zero — a value-level test that skips
        // the two full normalizations a `to_bytes()` compare pays. `ct_eq` is NOT a
        // substitute: k256's FieldElement compares raw limbs *and* the magnitude and
        // `normalized` tags, so a `mul` result (magnitude 1, unnormalized) never
        // compares equal to the normalized `ONE` constant whatever its value.
        // `Neg` is `negate(1)`, valid here because `mul` yields magnitude 1.
        if bool::from((*x * inv - FieldElement::ONE).normalizes_to_zero()) {
            Some(inv)
        } else {
            None
        }
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        Option::from(x.invert())
    }
}

/// Computes `k1·P1 + k2·P2` from two affine oracle queries, or `None` if a
/// degenerate configuration trips a guard.
///
/// The affine ECSM ecall returns the full point, so `A = k1·P1` and `B = k2·P2`
/// each cost one query and `Q = A + B` is a single chord addition — one field
/// inversion, for `1/(xb − xa)`. `dx = 0` covers both degenerate cases at once
/// (two curve points share an x only when they are equal or negatives), so the
/// caller falls back to the software `lincomb` there.
///
/// The x-only predecessor needed a second query `x((k+1)·P)` per point to solve
/// for `y` through the chord-addition law, which is what made `k1 = 1` and
/// `k1 = N−1` degenerate; with `y` supplied by the chip those scalars are
/// ordinary. secp256k1 has cofactor 1 and prime `N`, so `k·P ≠ O` for every
/// `k ∈ (0, N)` and no further scalar guard is needed.
///
/// Generic over the oracle so unit tests can substitute a software stand-in.
#[cfg(any(target_arch = "riscv64", test))]
fn lincomb2_with_oracle<O>(
    a1: &AffinePoint,
    k1: &Scalar,
    a2: &AffinePoint,
    k2: &Scalar,
    oracle: O,
) -> Option<AffinePoint>
where
    O: Fn(&FieldElement, &FieldElement, &Scalar) -> Option<(FieldElement, FieldElement)>,
{
    // Inputs are affine already (the ecrecover path lifts them from known Z=1
    // points), so no projective→affine inversion is needed here.
    if bool::from(a1.is_identity()) || bool::from(a2.is_identity()) {
        return None;
    }
    if bool::from(k1.is_zero()) || bool::from(k2.is_zero()) {
        return None;
    }

    let (x1, y1) = affine_xy(a1)?;
    let (x2, y2) = affine_xy(a2)?;

    // The oracle receives the full point (x, y) and returns k·(x, y) directly — no parity
    // convention or sign flip, since the precompile gets the real y (pinned in the prover
    // by a memory read).
    let (xa, ya) = oracle(&x1, &y1, k1)?;
    let (xb, yb) = oracle(&x2, &y2, k2)?;

    // Q = A + B via one chord addition (A ≠ ±B ⇒ dxq ≠ 0). One field inversion.
    let dxq = (xb - xa).normalize();
    if bool::from(dxq.is_zero()) {
        return None;
    }
    let inv_dxq = field_inv(&dxq)?;
    let lq = ((yb - ya) * inv_dxq).normalize();
    let xq = (lq.square() - xa - xb).normalize();
    let yq = (lq * (xa - xq) - ya).normalize();

    // `point_from_xy` checks the result is on the curve as a cheap backstop:
    // it rejects gross off-curve garbage and falls back to software, but
    // correctness rests on the algebra above — an on-curve-but-wrong point
    // would still pass this check.
    point_from_xy(&xq, &yq)
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

/// Builds an affine curve point from coordinates, returning `None` if the point
/// is not on the curve (`AffinePoint::from_encoded_point` validates this).
#[cfg(any(target_arch = "riscv64", test))]
fn point_from_xy(x: &FieldElement, y: &FieldElement) -> Option<AffinePoint> {
    let ep = EncodedPoint::from_affine_coordinates(&x.to_bytes(), &y.to_bytes(), false);
    Option::<AffinePoint>::from(AffinePoint::from_encoded_point(&ep))
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

    while input.len() - offset >= RATE {
        absorb_block(&mut state, &input[offset..offset + RATE]);
        permute(&mut state);
        offset += RATE;
    }

    // Final block with multi-rate padding.
    let mut last = [0u8; RATE];
    let remaining = input.len() - offset;
    last[..remaining].copy_from_slice(&input[offset..]);
    last[remaining] ^= 0x01;
    last[RATE - 1] ^= 0x80;
    absorb_block(&mut state, &last);
    permute(&mut state);

    // Squeeze the first 32 bytes (four lanes) as little-endian.
    let mut output = [0u8; 32];
    for (i, lane) in state.iter().take(4).enumerate() {
        output[i * 8..i * 8 + 8].copy_from_slice(&lane.to_le_bytes());
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
mod tests;

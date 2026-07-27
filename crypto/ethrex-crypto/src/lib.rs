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
//! - `secp256k1_ecrecover`: the ECDSA recovery's 2-term linear combination
//!   `u1·G + u2·R` is evaluated by a single `ecsm_lincomb2` ecall (riscv64),
//!   which returns the full affine point; on host, and whenever the accelerator
//!   reports a non-zero status, it falls back to the pure-Rust
//!   `ProjectivePoint::lincomb`.
//!
//! The `(r, v)` → `R` decompression stays in the guest and is **not** delegated:
//! the guest is the parity authority (proven CPU execution), and the
//! accelerator's obligations are curve membership and canonicalization only.
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

// Used only by the `ecsm_lincomb2` ABI marshalling (riscv accelerated path + the
// host unit tests that cover it); unused on a non-test host build.
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
/// `lincomb(G, u1, R, u2)` through the lincomb2 accelerator via
/// [`ecsm_lincomb2`], falling back to the software `ProjectivePoint::lincomb`
/// whenever the accelerated path declines (a non-zero accelerator status, or a
/// non-riscv build).
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

    // SEC1 uncompressed: 0x04 || X(32) || Y(32). Return X‖Y for the caller to hash.
    let uncompressed = pk_affine.to_encoded_point(false);
    let mut pk_bytes = [0u8; 64];
    pk_bytes.copy_from_slice(&uncompressed.as_bytes()[1..65]);
    Ok(pk_bytes)
}

/// ECSM-accelerated 2-term linear combination `k1·P1 + k2·P2`.
///
/// On riscv64 this is **one** `ecsm_lincomb2` ecall: the accelerator evaluates
/// the joint Shamir/Straus chain and writes the full affine `Q`. On other
/// targets it returns `None` so the caller uses the pure-Rust
/// `ProjectivePoint::lincomb`.
///
/// # The status contract is the only guard
///
/// The guest performs no degeneracy checks of its own beyond marshalling: every
/// case that the accelerator cannot prove — `k = 0`, `k >= N`, an off-curve or
/// non-canonical `P2`, `P1 != G`, `P1 = ±P2`, `Q = ∞`, and the chain's own
/// interior collisions — comes back as a non-zero status with `q` untouched, and
/// a non-zero status means `None` means the software fallback. Falling back is
/// always sound: the fallback is proven guest execution, so a status the
/// accelerator gets *wrong* costs cycles and nothing else.
///
/// The one guest-side guard that remains is structural rather than a policy
/// choice: an identity point has no affine `(x, y)` to marshal, so
/// [`lincomb2_operands`] returns `None` before any ecall.
///
/// `P1` must be the generator — the chip has no membership witness for an
/// arbitrary first point. We do not check that here: the accelerator reports
/// status `7` for any other `P1`, which is exactly the same fallback path, and
/// the only caller passes `ProjectivePoint::GENERATOR`.
#[cfg(target_arch = "riscv64")]
fn ecsm_lincomb2(
    p1: &ProjectivePoint,
    k1: &Scalar,
    p2: &ProjectivePoint,
    k2: &Scalar,
) -> Option<ProjectivePoint> {
    let (p1_op, p2_op, u_op) = lincomb2_operands(p1, k1, p2, k2)?;
    // Four distinct live locals: pairwise disjoint by construction, which the
    // executor requires (an overlap is a hard `ExecutionError`, not a status).
    let mut q = Operand([0u8; 64]);
    let status = lambda_vm_syscalls::syscalls::ecsm_lincomb2(&mut q.0, &p1_op.0, &p2_op.0, &u_op.0);
    if status != lambda_vm_syscalls::syscalls::ECSM_LINCOMB2_OK {
        return None;
    }
    point_from_le_q(&q.0)
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

/// One 64-byte `ecsm_lincomb2` operand: two 32-byte little-endian values.
///
/// The `repr(align(8))` is load-bearing, not cosmetic. The executor reads and
/// writes each operand as eight *aligned* doubleword MEMW accesses and rejects a
/// misaligned address as a hard `ExecutionError` — it is a guest bug, not an
/// attacker input, so it aborts the run rather than returning a status. A bare
/// `[u8; 64]` is only 1-byte aligned and would abort on a bad stack layout.
#[cfg(any(target_arch = "riscv64", test))]
#[repr(align(8))]
struct Operand([u8; 64]);

/// Packs two big-endian 32-byte values into one operand, each reversed into the
/// little-endian order the ABI uses.
#[cfg(any(target_arch = "riscv64", test))]
fn operand(first_be: &FieldBytes, second_be: &FieldBytes) -> Operand {
    let mut out = Operand([0u8; 64]);
    for i in 0..32 {
        out.0[i] = first_be[31 - i];
        out.0[32 + i] = second_be[31 - i];
    }
    out
}

/// The three input operands `(xP1‖yP1, xP2‖yP2, u1‖u2)` for one
/// `ecsm_lincomb2` ecall, or `None` if either point is the identity — which has
/// no affine `(x, y)` encoding to marshal, and is the one degenerate case the
/// status word cannot report because the call cannot be formed at all.
///
/// Split out from [`ecsm_lincomb2`] so the ABI is reachable from host tests: the
/// ecall itself only exists on riscv64, but the byte-order conversion — the part
/// most likely to carry a bug — is ordinary code.
#[cfg(any(target_arch = "riscv64", test))]
fn lincomb2_operands(
    p1: &ProjectivePoint,
    k1: &Scalar,
    p2: &ProjectivePoint,
    k2: &Scalar,
) -> Option<(Operand, Operand, Operand)> {
    let (x1, y1) = affine_xy(&p1.to_affine())?;
    let (x2, y2) = affine_xy(&p2.to_affine())?;
    Some((
        operand(&x1.to_bytes(), &y1.to_bytes()),
        operand(&x2.to_bytes(), &y2.to_bytes()),
        operand(&k1.to_bytes(), &k2.to_bytes()),
    ))
}

/// Rebuilds `Q` from the syscall's 64-byte result operand (`xQ‖yQ`, each 32-byte
/// little-endian).
///
/// `point_from_xy` re-checks the point is on the curve. That is a cheap backstop
/// against a marshalling bug on our side, not a soundness check on the
/// accelerator: the chip's own constraints are what bind `Q`, and an
/// on-curve-but-wrong point would pass this.
#[cfg(any(target_arch = "riscv64", test))]
fn point_from_le_q(q: &[u8; 64]) -> Option<ProjectivePoint> {
    let mut x_be = [0u8; 32];
    let mut y_be = [0u8; 32];
    for i in 0..32 {
        x_be[i] = q[31 - i];
        y_be[i] = q[63 - i];
    }
    let x = Option::<FieldElement>::from(FieldElement::from_bytes(&x_be.into()))?;
    let y = Option::<FieldElement>::from(FieldElement::from_bytes(&y_be.into()))?;
    point_from_xy(&x, &y)
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

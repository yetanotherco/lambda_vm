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
//!   evaluated through the ECSM `ecsm_mul` precompile (riscv64), which returns
//!   each `k·P` in full together with the base-point root it used — so the two
//!   products cost one query each and are combined with a single chord addition.
//!   On host / degenerate inputs it falls back to `ProjectivePoint::lincomb`.
//!
//! Every other `Crypto` method inherits the trait default (vetted pure-Rust
//! crates: `ark-bn254`, `bls12_381`, `p256`, `sha2`, `ripemd`, …).

use ethrex_crypto::keccak::keccak_hash;
use ethrex_crypto::{Crypto, CryptoError};
use k256::elliptic_curve::group::prime::PrimeCurveAffine;
use k256::elliptic_curve::ops::{LinearCombination, Reduce};
// `Invert` provides the software `x.invert()/invert_vartime()`. It is used by the
// host path AND, on the riscv64 guest, by the mandatory software fallback that
// runs whenever a hinted inverse fails to verify (a lying host). It is therefore
// needed in every build, not only off-target.
use k256::elliptic_curve::ops::Invert;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::elliptic_curve::PrimeField;
use k256::{AffinePoint, FieldBytes, ProjectivePoint, Scalar, U256};

// Used only by the point reconstruction (riscv accelerated path + the
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
/// prover's HINT table). The result is UNTRUSTED — the ecall adds no correctness
/// constraint, so every caller MUST verify it in-guest (`x·inv == 1`, `y² == x³+7`)
/// AND recompute in software on any verification failure. The hint is only ever
/// allowed to save work, never to change the answer: because the prover chooses the
/// bytes, an unverified-or-rejected-outright hint would let it steer a caller's
/// accept/reject outcome (e.g. force a valid signature to look invalid). See
/// [`scalar_inv`] / [`decompress_r`] for the fallback that closes that hole.
#[cfg(target_arch = "riscv64")]
fn get_hint(hint_id: usize, x_be: &[u8; 32]) -> [u8; 32] {
    let mut out = Align8([0u8; 32]);
    lambda_vm_syscalls::syscalls::hint(hint_id, &mut out.0, x_be);
    out.0
}

/// 8-byte-aligned wrapper for an ecall operand buffer, so the table's 8-byte accesses land
/// on the aligned memory path (MEMW_A, 29 columns + 1 range check) instead of the general
/// one (49 + 8). A bare `[u8; N]` on the stack is only 1-aligned, which forces every access
/// onto the unaligned path and inflates the trace.
#[cfg(target_arch = "riscv64")]
#[repr(C, align(8))]
struct Align8<const N: usize>([u8; N]);

/// Scalar-field inverse `x⁻¹ mod n`.
///
/// On riscv64 the inverse is first requested from the untrusted `hint` ecall and
/// verified in-guest (`x·inv == 1`); **on any verification failure it is recomputed
/// in software.** `x⁻¹` exists for every `x` this is called with — the only caller,
/// `ecsm_ecrecover`, guarantees `r ≠ 0` before calling — so a failed verify can only
/// mean the host lied, and the software value is authoritative. This is what keeps
/// the result independent of the prover-chosen hint: a bad hint makes the guest do
/// more work, it can never change the answer, so it cannot turn a valid signature
/// into a recovery failure. Off-target (host) it inverts in software directly.
fn scalar_inv(x: &Scalar) -> Option<Scalar> {
    #[cfg(target_arch = "riscv64")]
    {
        scalar_inv_with_oracle(x, |x_be| {
            get_hint(lambda_vm_syscalls::syscalls::HINT_SCALAR_INV, x_be)
        })
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        x.invert_vartime().into()
    }
}

/// Core of [`scalar_inv`], generic over the hint source so host tests can inject an
/// honest or a lying oracle and assert the software fallback keeps the result
/// correct either way. See [`scalar_inv`] for the verify-then-fallback rationale.
#[cfg(any(target_arch = "riscv64", test))]
fn scalar_inv_with_oracle<O>(x: &Scalar, hint: O) -> Option<Scalar>
where
    O: FnOnce(&[u8; 32]) -> [u8; 32],
{
    use k256::elliptic_curve::subtle::ConstantTimeEq;
    let x_be: [u8; 32] = x.to_bytes().into();
    let inv_be = hint(&x_be);
    // Fast path: a canonical hint that verifies (x·inv == 1 mod n) is used as-is.
    if let Some(inv) = Option::<Scalar>::from(Scalar::from_repr(inv_be.into())) {
        if bool::from((*x * inv).ct_eq(&Scalar::ONE)) {
            return Some(inv);
        }
    }
    // Hint absent / malformed / wrong: recompute authoritatively. `x⁻¹` exists for
    // every input the callers pass (`r ≠ 0`), so this is `Some` on the honest path.
    x.invert_vartime().into()
}

/// Decompress R from its x-coordinate + parity.
///
/// On riscv64 the square root `y = sqrt(x³+7)` is first requested from the untrusted
/// `hint` ecall and verified in-guest (`y² == x³+7`), with parity selection; **on any
/// verification failure the point is recomputed with the software
/// `AffinePoint::decompress`.** Unlike the inverse, a failure here is *not*
/// necessarily a lying host: a genuine non-residue (an invalid signature) has no
/// root and must legitimately yield `None`. So the fallback is the authoritative
/// software decompress, which returns `Some` for a residue and `None` for a
/// non-residue regardless of the prover-chosen hint — the hint can only save work,
/// never steer the accept/reject outcome. Off-target it uses the software
/// decompress directly.
fn decompress_r(r_bytes: &FieldBytes, y_is_odd: bool) -> Option<AffinePoint> {
    #[cfg(target_arch = "riscv64")]
    {
        decompress_r_with_oracle(r_bytes, y_is_odd, |rhs_be| {
            get_hint(lambda_vm_syscalls::syscalls::HINT_FIELD_SQRT, rhs_be)
        })
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        use k256::elliptic_curve::point::DecompressPoint;
        AffinePoint::decompress(r_bytes, u8::from(y_is_odd).into()).into()
    }
}

/// Core of [`decompress_r`], generic over the hint source for host tests: try the
/// hinted sqrt, then fall back to the authoritative software decompress on any
/// failure. See [`decompress_r`] for the rationale.
#[cfg(any(target_arch = "riscv64", test))]
fn decompress_r_with_oracle<O>(r_bytes: &FieldBytes, y_is_odd: bool, hint: O) -> Option<AffinePoint>
where
    O: FnOnce(&[u8; 32]) -> [u8; 32],
{
    if let Some(p) = decompress_r_hinted(r_bytes, y_is_odd, hint) {
        return Some(p);
    }
    // Hinted root absent / malformed / wrong, OR a genuine non-residue: the software
    // decompress is authoritative — `Some` for a residue, `None` for a non-residue.
    use k256::elliptic_curve::point::DecompressPoint;
    AffinePoint::decompress(r_bytes, u8::from(y_is_odd).into()).into()
}

/// The hint-accelerated decompress attempt: returns the point only if the hinted
/// root verifies (`y² == x³+7`); `None` on any failure, so the caller falls back to
/// the software decompress. Never the last word — a `None` here is not a decision
/// that R is invalid, only that the fast path did not produce a verified root.
#[cfg(any(target_arch = "riscv64", test))]
fn decompress_r_hinted<O>(r_bytes: &FieldBytes, y_is_odd: bool, hint: O) -> Option<AffinePoint>
where
    O: FnOnce(&[u8; 32]) -> [u8; 32],
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
    let y_be = hint(&rhs_be);
    let mut y: FieldElement = Option::from(FieldElement::from_bytes(&y_be.into()))?;
    let y2: FieldElement = y.square();
    // Verify the untrusted root: y² must equal x³+7. Negate `y2`, not `rhs`:
    // `Neg` is `negate(1)`, whose debug assert requires magnitude <= 1. `square()`
    // always returns magnitude 1, whereas `rhs` is a sum carrying magnitude 2, so
    // negating it would trip that assert and panic in debug builds. (The value would
    // still come out right — `negate(m)` computes `2*(m+1)*P_limb - self`, which for a
    // magnitude-2 operand stays non-negative — so this is a build-configuration
    // hazard, not a wrong answer.)
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
/// On riscv64 this uses two ECSM queries (the precompile returns the full `k·P` plus
/// the root it used, see [`lincomb2_with_oracle`]) instead of four x-only queries plus
/// chord-law y-reconstruction; on other targets, and whenever a guard trips (degenerate
/// input or an unusable oracle result), it returns `None` so the caller uses the
/// pure-Rust `ProjectivePoint::lincomb`.
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

/// Scalar-mul oracle backed by the ECSM precompile: for the curve point `P` whose
/// x-coordinate is passed in, returns `(x(k·P̂), y(k·P̂), ŷ)`, where `P̂ = (x, ŷ)` is the root
/// of `x` the chip actually witnessed. The chip is free to pick either root — the AIR binds
/// only `ŷ² ≡ x³ + b` — so the caller resolves the sign from `ŷ` (see
/// [`lincomb2_with_oracle`]). `x` must be the x-coordinate of a curve point and `k` in
/// `(0, N)` (N = curve order), guaranteed by the guards there.
///
/// Values cross the ABI as 32-byte little-endian; `x_le` and `k_le` are distinct stack
/// arrays so the executor's `|addr_x_le − addr_k_le| ≥ 32` assumption holds by construction.
///
/// `None` on any coordinate that is not a canonical field element. That parse is load-bearing
/// for `ŷ`, not just hygiene: `p` is odd, so `y` and `p − y` differ in parity, but a value
/// `y + p` (a second 256-bit representative of `y`, possible when `y < 2^256 − p ≈ 2^32`)
/// would carry the *opposite* parity. Rejecting `≥ p` here is what pins `ŷ` to exactly one
/// of the two true roots, and it costs nothing — it is the field-element parse.
#[cfg(target_arch = "riscv64")]
fn ecsm_oracle(x: &FieldElement, k: &Scalar) -> Option<(FieldElement, FieldElement, FieldElement)> {
    let x_be = x.to_bytes();
    let k_be = k.to_bytes();
    let mut x_le = Align8([0u8; 32]);
    let mut k_le = Align8([0u8; 32]);
    for i in 0..32 {
        x_le.0[i] = x_be[31 - i];
        k_le.0[i] = k_be[31 - i];
    }
    let mut out = Align8([0u8; 96]);
    lambda_vm_syscalls::syscalls::ecsm_mul(&mut out.0, &x_le.0, &k_le.0);
    let load = |chunk: usize| -> Option<FieldElement> {
        let mut be = [0u8; 32];
        for i in 0..32 {
            be[i] = out.0[chunk * 32 + 31 - i];
        }
        Option::from(FieldElement::from_bytes(&be.into()))
    };
    Some((load(0)?, load(1)?, load(2)?))
}

/// Base-field inverse `x⁻¹ mod p`.
///
/// On riscv64 the inverse is first requested from the untrusted `hint` ecall and
/// verified in-guest (`x·inv == 1`); **on any verification failure it is recomputed
/// in software.** A bad hint can only cost the guest extra work, never change the
/// answer — it cannot steer a caller's accept/reject outcome. Off-target it inverts
/// in software directly. Returns `None` only for a genuinely non-invertible input
/// (`x = 0`), which the callers' degeneracy guards already exclude.
#[cfg(any(target_arch = "riscv64", test))]
fn field_inv(x: &FieldElement) -> Option<FieldElement> {
    #[cfg(target_arch = "riscv64")]
    {
        field_inv_with_oracle(x, |x_be| {
            get_hint(lambda_vm_syscalls::syscalls::HINT_FIELD_INV, x_be)
        })
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        Option::from(x.invert())
    }
}

/// Core of [`field_inv`], generic over the hint source so host tests can inject an
/// honest or a lying oracle and assert the software fallback keeps the result
/// correct either way. See [`scalar_inv`] for the verify-then-fallback rationale.
#[cfg(any(target_arch = "riscv64", test))]
fn field_inv_with_oracle<O>(x: &FieldElement, hint: O) -> Option<FieldElement>
where
    O: FnOnce(&[u8; 32]) -> [u8; 32],
{
    let x_be: [u8; 32] = x.to_bytes().into();
    let inv_be = hint(&x_be);
    // Fast path: a canonical hint that verifies (x·inv == 1 mod p) is used as-is.
    // Verify by asking whether the difference normalizes to zero — a value-level test
    // that skips the two full normalizations a `to_bytes()` compare pays. `ct_eq` is
    // NOT a substitute: k256's FieldElement compares raw limbs *and* the magnitude and
    // `normalized` tags, so a `mul` result (magnitude 1, unnormalized) never compares
    // equal to the normalized `ONE` constant whatever its value.
    // `Neg` is `negate(1)`, valid here because `mul` yields magnitude 1.
    if let Some(inv) = Option::<FieldElement>::from(FieldElement::from_bytes(&inv_be.into())) {
        if bool::from((*x * inv - FieldElement::ONE).normalizes_to_zero()) {
            return Some(inv);
        }
    }
    // Hint absent / malformed / wrong: recompute authoritatively. `None` only for a
    // genuine `x = 0`, excluded by the callers' guards.
    Option::from(x.invert())
}

/// Computes `k1·P1 + k2·P2` from two oracle queries, or `None` if a degenerate
/// configuration or an unusable oracle result trips a guard.
///
/// The ECSM ecall returns the full point `k·P̂` together with the root `ŷ` it used, and the
/// chip may pick either root of `x(P)` — the AIR binds only `ŷ² ≡ x³ + b`. So `k·P̂ = ±(k·P)`:
/// comparing `ŷ` against the caller's own `y` says which, and one conditional negation
/// recovers `k·P`. `Q = A + B` is then a single chord addition with a single field inversion.
///
/// The x-only predecessor needed a second query `x((k+1)·P)` per point plus the chord-law
/// y-reconstruction, which is what made `k1 = 1` and `k1 = N−1` degenerate; with `y` in hand
/// those scalars are ordinary. secp256k1 has cofactor 1 and prime `N`, so `k·P ≠ O` for every
/// `k ∈ (0, N)` and no further scalar guard is needed. `dx = 0` still covers both remaining
/// degenerate cases at once (two curve points share an x only when they are equal or
/// negatives), and the caller falls back to the software `lincomb` there.
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
    O: Fn(&FieldElement, &Scalar) -> Option<(FieldElement, FieldElement, FieldElement)>,
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

    let (xa, ya) = oracle_point(&x1, &y1, k1, &oracle)?;
    let (xb, yb) = oracle_point(&x2, &y2, k2, &oracle)?;

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

/// One oracle query plus the root fix-up: `k·(xp, yp)` in affine coordinates.
///
/// The oracle multiplied `(xp, ŷ)` for whichever root `ŷ` the chip witnessed, so the result
/// is `k·(xp, yp)` when `ŷ = yp` and `−k·(xp, yp)` when `ŷ = −yp`. Since `ŷ` is canonical
/// (the oracle's field-element parse rejected `≥ p`) and satisfies `ŷ² ≡ xp³ + b`, those are
/// the only two cases; anything else means the oracle did not multiply *this* point, so we
/// return `None` and the caller falls back to software.
///
/// Compared by value rather than `ct_eq`: k256 compares raw limbs *and* the magnitude and
/// `normalized` tags, so a subtraction result never compares equal to a normalized constant
/// whatever its value. Both operands are `from_bytes` outputs (magnitude 1), which keeps
/// `Sub`'s internal `negate(1)` within its contract; the negated `yr` is re-normalized so
/// the caller's later subtraction stays within it too.
#[cfg(any(target_arch = "riscv64", test))]
fn oracle_point<O>(
    xp: &FieldElement,
    yp: &FieldElement,
    k: &Scalar,
    oracle: &O,
) -> Option<(FieldElement, FieldElement)>
where
    O: Fn(&FieldElement, &Scalar) -> Option<(FieldElement, FieldElement, FieldElement)>,
{
    let (xr, yr, yg) = oracle(xp, k)?;
    if bool::from((*yp - yg).normalizes_to_zero()) {
        return Some((xr, yr));
    }
    if bool::from((*yp + yg).normalizes_to_zero()) {
        return Some((xr, (-yr).normalize()));
    }
    None
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

//! Unit tests for `compute_commit_bus_offset`.
//!
//! Pins the three behaviours the verify-path helper must preserve:
//! empty input short-circuit, success-path equivalence with a naive
//! per-element-inverse reference, and the zero-fingerprint failure path.
//!
//! The COMMIT output rides the Memory bus in the `commit` domain
//! (`domain = 2`), token `[2, index, 0, 0, 0, value]`, so the fingerprint is
//! `z - (BusId::Memory + 2·α + index·α² + value·α⁶)` (bus_id at α⁰, then the six
//! token values at α¹..α⁶; the addr_hi/ts_lo/ts_hi slots are 0 and drop out).

use math::field::element::FieldElement;

use crate::compute_commit_bus_offset;
use crate::tables::types::{BusId, GoldilocksExtension};

type E = GoldilocksExtension;

/// Reference implementation: one `inv()` per fingerprint, then sum.
/// Mirrors the original loop bit-for-bit modulo addition order, so any
/// future refactor of the batched routine must remain equivalent to this.
fn naive_offset(
    public_output: &[u8],
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
) -> Option<FieldElement<E>> {
    let bus_id = FieldElement::<E>::from(BusId::Memory as u64);
    let alpha_sq = alpha * alpha;
    let alpha_pow6 = &alpha_sq * &alpha_sq * &alpha_sq;
    let domain = &bus_id + (FieldElement::<E>::from(2u64) * alpha);
    let mut total = FieldElement::<E>::zero();
    for (i, &value) in public_output.iter().enumerate() {
        let lc = &domain
            + (FieldElement::<E>::from(i as u64) * &alpha_sq)
            + (FieldElement::<E>::from(value as u64) * &alpha_pow6);
        let fingerprint = z - lc;
        total += fingerprint.inv().ok()?;
    }
    Some(total)
}

#[test]
fn test_empty_public_output_returns_zero() {
    let z = FieldElement::<E>::from(7u64);
    let alpha = FieldElement::<E>::from(11u64);
    assert_eq!(
        compute_commit_bus_offset(&[], &z, &alpha),
        Some(FieldElement::<E>::zero()),
    );
}

#[test]
fn test_non_empty_matches_naive_per_element_inverse() {
    let z = FieldElement::<E>::from(987_654_321u64);
    let alpha = FieldElement::<E>::from(31_415_926u64);
    let public_output: [u8; 5] = [0x01, 0x02, 0xff, 0x10, 0x80];

    let batched = compute_commit_bus_offset(&public_output, &z, &alpha);
    let naive = naive_offset(&public_output, &z, &alpha);

    assert_eq!(batched, naive);
    assert!(batched.is_some(), "no fingerprint should collide here");
}

#[test]
fn test_longer_input_matches_naive() {
    let z = FieldElement::<E>::from(0xdead_beefu64);
    let alpha = FieldElement::<E>::from(0xcafe_babeu64);
    let public_output: Vec<u8> = (0..=255u16).map(|x| x as u8).collect();

    let batched = compute_commit_bus_offset(&public_output, &z, &alpha);
    let naive = naive_offset(&public_output, &z, &alpha);

    assert_eq!(batched, naive);
    assert!(batched.is_some());
}

#[test]
fn test_index_is_bound_into_the_offset() {
    // The commit index rides in the token, so the same bytes at shifted positions
    // fingerprint differently. Reversing a multi-byte output must change the offset
    // (each byte's α² index term changes), guarding that the index is bound in.
    let z = FieldElement::<E>::from(0x1234_5678u64);
    let alpha = FieldElement::<E>::from(0x9abc_def0u64);
    let forward: [u8; 3] = [0xCC, 0xDD, 0xEE];
    let reversed: [u8; 3] = [0xEE, 0xDD, 0xCC];

    let a = compute_commit_bus_offset(&forward, &z, &alpha);
    let b = compute_commit_bus_offset(&reversed, &z, &alpha);
    assert!(a.is_some() && b.is_some());
    assert_ne!(a, b, "index must be bound into each byte's fingerprint");
}

#[test]
fn test_zero_fingerprint_returns_none() {
    // Craft fingerprint_0 = 0: index = 0, value = 0, then
    //   fingerprint_0 = z - (BusId::Memory + 2·α + 0·α² + 0·α⁶) = z - (BusId::Memory + 2α).
    // Setting z to that forces the collision regardless of the higher alpha powers.
    let alpha = FieldElement::<E>::from(42u64);
    let z = FieldElement::<E>::from(BusId::Memory as u64) + (FieldElement::<E>::from(2u64) * alpha);
    let public_output: [u8; 1] = [0];

    assert_eq!(
        compute_commit_bus_offset(&public_output, &z, &alpha),
        None,
        "zero fingerprint must propagate as None",
    );
}

#[test]
fn test_zero_fingerprint_in_middle_returns_none() {
    // Same idea at i = 2, so some valid fingerprints precede the zero one.
    let alpha = FieldElement::<E>::from(5u64);
    let alpha_sq = alpha * alpha;
    let alpha_pow6 = alpha_sq * alpha_sq * alpha_sq;
    let bus_id = FieldElement::<E>::from(BusId::Memory as u64);
    // value = 3 at index 2 → z = Memory + 2α + 2·α² + 3·α⁶ forces fingerprint_2 = 0.
    let z = bus_id
        + (FieldElement::<E>::from(2u64) * alpha)
        + (FieldElement::<E>::from(2u64) * alpha_sq)
        + (FieldElement::<E>::from(3u64) * alpha_pow6);
    let public_output: [u8; 4] = [1, 2, 3, 4];

    assert_eq!(compute_commit_bus_offset(&public_output, &z, &alpha), None);
}

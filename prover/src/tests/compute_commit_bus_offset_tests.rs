//! Unit tests for `compute_commit_bus_offset`.
//!
//! Pins the three behaviours the verify-path helper must preserve:
//! empty input short-circuit, success-path equivalence with a naive
//! per-element-inverse reference, and the zero-fingerprint failure path.

use math::field::element::FieldElement;

use executor::vm::instruction::execution::memmove_row_width;

use crate::compute_commit_bus_offset;
use crate::tables::types::{BusId, GoldilocksExtension};

type E = GoldilocksExtension;

/// Reference implementation: one `inv()` per fingerprint, then sum.
/// Mirrors the original loop bit-for-bit modulo addition order, so any
/// future refactor of the batched routine must remain equivalent to this.
fn naive_offset(
    public_output: &[u8],
    start_index: u64,
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
) -> Option<FieldElement<E>> {
    let bus_id = FieldElement::<E>::from(BusId::Commit as u64);
    let alpha_sq = alpha * alpha;
    let mut total = FieldElement::<E>::zero();
    for (i, &value) in public_output.iter().enumerate() {
        let lc = bus_id
            + (FieldElement::<E>::from(start_index + i as u64) * alpha)
            + (FieldElement::<E>::from(value as u64) * alpha_sq);
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
        compute_commit_bus_offset(&[], 0, &z, &alpha),
        Some(FieldElement::<E>::zero()),
    );
}

#[test]
fn test_non_empty_matches_naive_per_element_inverse() {
    let z = FieldElement::<E>::from(987_654_321u64);
    let alpha = FieldElement::<E>::from(31_415_926u64);
    let public_output: [u8; 5] = [0x01, 0x02, 0xff, 0x10, 0x80];

    let batched = compute_commit_bus_offset(&public_output, 0, &z, &alpha);
    let naive = naive_offset(&public_output, 0, &z, &alpha);

    assert_eq!(batched, naive);
    assert!(batched.is_some(), "no fingerprint should collide here");
}

#[test]
fn test_longer_input_matches_naive() {
    let z = FieldElement::<E>::from(0xdead_beefu64);
    let alpha = FieldElement::<E>::from(0xcafe_babeu64);
    let public_output: Vec<u8> = (0..=255u16).map(|x| x as u8).collect();

    let batched = compute_commit_bus_offset(&public_output, 0, &z, &alpha);
    let naive = naive_offset(&public_output, 0, &z, &alpha);

    assert_eq!(batched, naive);
    assert!(batched.is_some());
}

#[test]
fn test_nonzero_start_index_matches_naive() {
    // A continuation epoch whose commits continue a prior epoch: the offset must
    // index from the carried x254, not 0.
    let z = FieldElement::<E>::from(0x1234_5678u64);
    let alpha = FieldElement::<E>::from(0x9abc_def0u64);
    let public_output: [u8; 3] = [0xCC, 0xDD, 0xEE];
    let start_index = 7u64;

    let batched = compute_commit_bus_offset(&public_output, start_index, &z, &alpha);
    let naive = naive_offset(&public_output, start_index, &z, &alpha);

    assert_eq!(batched, naive);
    assert!(batched.is_some());

    // A different start index yields a different offset (the index is bound in).
    let shifted = compute_commit_bus_offset(&public_output, start_index + 1, &z, &alpha);
    assert_ne!(batched, shifted);
}

#[test]
fn test_zero_fingerprint_returns_none() {
    // Craft fingerprint_0 = 0: start_index = 0, value = 0, then
    //   fingerprint_0 = z - (BusId::Commit + 0·α + 0·α²) = z - BusId::Commit.
    // Setting z = BusId::Commit forces the collision regardless of alpha.
    let z = FieldElement::<E>::from(BusId::Commit as u64);
    let alpha = FieldElement::<E>::from(42u64);
    let public_output: [u8; 1] = [0];

    assert_eq!(
        compute_commit_bus_offset(&public_output, 0, &z, &alpha),
        None,
        "zero fingerprint must propagate as None",
    );
}

#[test]
fn test_zero_fingerprint_in_middle_returns_none() {
    // Same idea at i = 2, so some valid fingerprints precede the zero one.
    let alpha = FieldElement::<E>::from(5u64);
    let alpha_sq = alpha * alpha;
    let bus_id = FieldElement::<E>::from(BusId::Commit as u64);
    // value = 3 at index 2 → z = BusId + 2α + 3α² forces fingerprint_2 = 0.
    let z = bus_id
        + (FieldElement::<E>::from(2u64) * alpha)
        + (FieldElement::<E>::from(3u64) * alpha_sq);
    let public_output: [u8; 4] = [1, 2, 3, 4];

    assert_eq!(
        compute_commit_bus_offset(&public_output, 0, &z, &alpha),
        None,
    );
}

/// The COMMIT tuples the MEMMOVE chip actually sends, walking the production row
/// schedule one commit ECALL at a time.
///
/// This is the prover side, not a second copy of the verifier: the widths come from
/// `memmove_row_width`, the same function the trace builder and the sizing pass use,
/// and the tuple shape is the chip's — one `(global index, byte)` pair per byte.
/// `commits` is the per-ECALL split of the public output, which is exactly the thing
/// the verifier never learns.
fn prover_offset(
    commits: &[&[u8]],
    start_index: u64,
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
) -> Option<FieldElement<E>> {
    let bus_id = FieldElement::<E>::from(BusId::Commit as u64);
    let alpha_sq = alpha * alpha;
    let mut total = FieldElement::<E>::zero();
    let mut index = start_index;

    for bytes in commits {
        let base = index;
        let mut offset = 0u64;
        let mut remaining = bytes.len() as u64;
        while remaining != 0 {
            let width = u64::from(memmove_row_width(0, base, offset, remaining, true));
            for lane in 0..width {
                let byte = bytes[(offset + lane) as usize];
                let lc = bus_id
                    + (FieldElement::<E>::from(base + offset + lane) * alpha)
                    + (FieldElement::<E>::from(byte as u64) * alpha_sq);
                total += (z - lc).inv().ok()?;
            }
            offset += width;
            remaining -= width;
        }
        index += bytes.len() as u64;
    }

    Some(total)
}

/// The verifier rebuilds the COMMIT bus from the concatenated `public_output` and
/// never learns where one commit ECALL ended and the next began. So the prover's
/// tuples must not depend on that split.
///
/// This is the regression test for the eight-lane tuple: with one tuple per row,
/// `[&[..4], &[..4]]` sent eight one-byte tuples while the verifier, chunking the
/// eight bytes it sees, expected a single eight-byte one — an honest proof rejected.
#[test]
fn test_prover_tuples_are_independent_of_the_ecall_split() {
    let z = FieldElement::<E>::from(9_876_543_211u64);
    let alpha = FieldElement::<E>::from(1_357u64);

    let splits: &[&[&[u8]]] = &[
        // One ECALL, sub-eight, exact eight, and a wide body with a tail.
        &[&[1, 2, 3, 4]],
        &[&[1, 2, 3, 4, 5, 6, 7, 8]],
        &[&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        // Several ECALLs. Only the last may be a multiple of eight without the
        // schedule and the verifier's chunking drifting apart.
        &[&[1, 2, 3, 4], &[5, 6, 7, 8]],
        &[&[1, 2, 3], &[4, 5, 6, 7, 8, 9, 10, 11, 12]],
        &[&[1, 2, 3, 4, 5, 6, 7, 8], &[9], &[10, 11, 12, 13, 14]],
        &[&[1], &[2], &[3], &[4], &[5], &[6], &[7], &[8], &[9]],
    ];

    for (case, commits) in splits.iter().enumerate() {
        for &start_index in &[0u64, 1, 7, 8, 4_294_967_290] {
            let concatenated: Vec<u8> = commits.concat();
            let prover = prover_offset(commits, start_index, &z, &alpha)
                .expect("no fingerprint collision on the prover side");
            let verifier = compute_commit_bus_offset(&concatenated, start_index, &z, &alpha)
                .expect("no fingerprint collision on the verifier side");
            assert_eq!(
                prover, verifier,
                "case {case} at start_index {start_index}: the COMMIT bus does not \
                 balance, so an honest proof would be rejected"
            );
        }
    }
}

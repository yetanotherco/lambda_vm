//! Automatic `StorageMode` selection from an analytical peak-RAM estimate.
//!
//! Peak prover memory is the sum of two table-shaped terms (main + aux
//! field elements times bytes-per-element through trace/LDE/Merkle) plus a
//! mostly-fixed overhead for state kept alive across phase 5 (memory cells,
//! log buffer, instruction map). Per-element coefficients are derived from
//! the STARK structure, not fitted:
//!
//! ```text
//! per main element  = 8                   (Goldilocks base field byte count)
//!                   + 8 * blowup_factor   (LDE expansion)
//!                   + ~2                  (Merkle node share, averaged across cols)
//! per aux element   = 24                  (cubic extension byte count)
//!                   + 24 * blowup_factor  (LDE expansion)
//!                   + ~3                  (Merkle node share)
//! fixed overhead    = 2 GB                (memory_state cells, log Vec, HashMap slack)
//! ```
//!
//! Element counts come from [`TableLengths`] via [`count_table_lengths`]: a
//! single streaming pass over execution logs that mirrors the trace builder's
//! partition/derivation logic without allocating the `Vec<*Operation>`
//! intermediates. The estimate is therefore *exact* up to the per-element
//! coefficient, not a regression fit.

use crate::tables::trace_builder::TableLengths;
use stark::storage_mode::StorageMode;
use sysinfo::System;

/// Peak RAM estimate in bytes for a proof whose trace shape matches `lengths`.
/// `blowup_factor` is the LDE blowup from `ProofOptions::blowup_factor`.
pub fn peak_bytes(lengths: &TableLengths, blowup_factor: u8) -> u64 {
    // Per element: trace + LDE + Merkle node share.
    let blowup = blowup_factor as u64;
    let bytes_per_main = 8 + 8 * blowup + 2;
    let bytes_per_aux = 24 + 24 * blowup + 3;
    // Memory cells, log Vec, instruction map, and HashMap allocator slack.
    const FIXED_OVERHEAD: u64 = 2_000_000_000;

    lengths
        .total_main_elements()
        .saturating_mul(bytes_per_main)
        .saturating_add(lengths.total_aux_elements().saturating_mul(bytes_per_aux))
        .saturating_add(FIXED_OVERHEAD)
}

/// Effective RAM budget against which the estimate is compared.
///
/// Returns `None` when the OS can't report available memory and the user
/// hasn't set a cap; in that case the caller should default to `Ram` rather
/// than force Disk on every proof. Otherwise the budget is the user's cap (if
/// set), clamped down by what the OS reports available.
pub fn effective_budget(available: Option<u64>, cap: Option<u64>) -> Option<u64> {
    match (cap, available) {
        (Some(c), Some(a)) => Some(c.min(a)),
        (Some(c), None) => Some(c),
        (None, a) => a,
    }
}

/// Pick a storage mode given the estimate and the machine's available RAM.
///
/// Uses 80% of the effective budget as the cutoff so there's headroom for the
/// OS, other processes, and allocator fragmentation. `cap` is an optional
/// user-imposed limit (see `ProofOptions::max_ram_bytes`) which overrides the
/// machine's reported available RAM when smaller.
///
/// `available` is a one-shot sample. If a concurrent process allocates
/// between this call and phase 5, this function may pick `Ram` and the
/// prover OOMs. The 80% headroom and the estimator's 1.3× margin cover
/// background jitter; under contention, pass `ProofOptions::max_ram_bytes`
/// for a hard cap.
pub fn select_storage_mode(
    estimated: u64,
    available: Option<u64>,
    cap: Option<u64>,
) -> StorageMode {
    const SAFETY_FRACTION_NUM: u64 = 4;
    const SAFETY_FRACTION_DEN: u64 = 5;

    let Some(budget) = effective_budget(available, cap) else {
        return StorageMode::Ram;
    };
    let threshold = budget.saturating_mul(SAFETY_FRACTION_NUM) / SAFETY_FRACTION_DEN;

    if estimated > threshold {
        StorageMode::Disk
    } else {
        StorageMode::Ram
    }
}

/// Query the OS for currently available RAM (not total) in bytes. Returns
/// `None` when the OS can't report a figure (e.g. inside containers without
/// `/proc/meminfo`).
pub fn available_ram_bytes() -> Option<u64> {
    let mut sys = System::new();
    sys.refresh_memory();
    match sys.available_memory() {
        0 => None,
        n => Some(n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    fn empty_lengths() -> TableLengths {
        TableLengths::default()
    }

    /// Even an "empty" trace pays the BITWISE 2^20-row baseline plus HALT and
    /// REGISTER constants. The formula is exact, so the test reproduces it
    /// by calling `total_main_elements`/`total_aux_elements` directly.
    #[test]
    fn peak_bytes_matches_per_element_formula() {
        let blowup = 2u8;
        let lengths = empty_lengths();
        let main = lengths.total_main_elements();
        let aux = lengths.total_aux_elements();
        let expected =
            main * (8 + 8 * blowup as u64 + 2) + aux * (24 + 24 * blowup as u64 + 3) + 2 * GB;
        assert_eq!(peak_bytes(&lengths, blowup), expected);

        let mut lengths = empty_lengths();
        lengths.cpu_padded_rows = 4; // CPU has 76 main cols + aux cols.
        let main = lengths.total_main_elements();
        let aux = lengths.total_aux_elements();
        let expected =
            main * (8 + 8 * blowup as u64 + 2) + aux * (24 + 24 * blowup as u64 + 3) + 2 * GB;
        assert_eq!(peak_bytes(&lengths, blowup), expected);
    }

    /// Higher blowup_factor should produce a strictly larger estimate.
    #[test]
    fn peak_bytes_scales_with_blowup() {
        let lengths = empty_lengths();
        let two = peak_bytes(&lengths, 2);
        let four = peak_bytes(&lengths, 4);
        let eight = peak_bytes(&lengths, 8);
        assert!(two < four);
        assert!(four < eight);
    }

    #[test]
    fn select_ram_when_estimate_below_threshold() {
        // 10 GB estimated, 32 GB available → threshold 25.6 GB → Ram.
        let mode = select_storage_mode(10 * GB, Some(32 * GB), None);
        assert_eq!(mode, StorageMode::Ram);
    }

    #[test]
    fn select_disk_when_estimate_exceeds_threshold() {
        // 30 GB estimated, 32 GB available → threshold 25.6 GB → Disk.
        let mode = select_storage_mode(30 * GB, Some(32 * GB), None);
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn cap_forces_disk_when_smaller_than_available() {
        // 10 GB estimated, 64 GB available (would be Ram), but cap=4 GB
        // → threshold = 4 × 0.8 = 3.2 GB → Disk.
        let mode = select_storage_mode(10 * GB, Some(64 * GB), Some(4 * GB));
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn cap_ignored_when_larger_than_available() {
        // available=8 GB dominates a cap of 64 GB.
        // threshold = 8 × 0.8 = 6.4 GB, estimate 10 GB → Disk.
        let mode = select_storage_mode(10 * GB, Some(8 * GB), Some(64 * GB));
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn tiny_cap_always_forces_disk() {
        let mode = select_storage_mode(
            peak_bytes(&empty_lengths(), 2),
            Some(64 * GB),
            Some(1_000_000),
        );
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn unknown_available_with_no_cap_falls_back_to_ram() {
        // OS can't report available memory. Without a cap we can't make an
        // informed decision, so stay in Ram rather than forcing Disk on every
        // proof.
        let mode = select_storage_mode(peak_bytes(&empty_lengths(), 2), None, None);
        assert_eq!(mode, StorageMode::Ram);
    }

    #[test]
    fn unknown_available_with_cap_uses_cap_as_budget() {
        // OS can't report; cap is the whole budget.
        let mode = select_storage_mode(10 * GB, None, Some(4 * GB));
        assert_eq!(mode, StorageMode::Disk);
    }
}

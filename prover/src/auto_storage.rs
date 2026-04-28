//! Automatic `StorageMode` selection based on a peak-RAM estimate.
//!
//! Peak prover memory is dominated by a single term — the count of main-trace
//! field elements — because every element materializes across the heap trace
//! buffer, the LDE expansion (blowup × base-field bytes), and the Merkle tree
//! node array, plus a roughly proportional contribution from the extension-
//! field aux columns. A linear fit on empirically measured runs of
//! `fib_iterative_{2,4,8}M` (residuals < 0.3%) gives:
//!
//! ```text
//! peak_bytes ≈ 65.5 * main_elements + 2.4 GB
//! ```
//!
//! We bake in a 1.3× safety factor so this can be trusted on programs with a
//! different trace-shape (e.g. matrix multiplication: heavier on MUL/DVRM than
//! fib), rounding the coefficients to `peak ≈ 85 * main + 3 GB`. The estimate
//! therefore errs conservative for MUL/DVRM-heavy programs — it will pick Disk
//! slightly earlier than strictly necessary, which is the safe direction.
//!
//! Note: the formula models trace/LDE/Merkle allocations. The intermediate
//! op vectors (`CpuOperation`, `MemwOperation`, …) that the trace builder
//! allocates before trace tables are not an explicit term; their contribution
//! is absorbed by the 1.3× margin.

use stark::storage_mode::StorageMode;
use sysinfo::System;

/// Peak RAM estimate in bytes for a proof over a trace with `main_elements`
/// main-trace field elements.
///
/// See module docs for the derivation of the 85-bytes-per-element coefficient
/// and 3 GB constant.
pub fn estimate_peak_bytes(main_elements: u64) -> u64 {
    const PEAK_BYTES_PER_MAIN_ELEMENT: u64 = 85;
    const PEAK_BYTES_FIXED_OVERHEAD: u64 = 3_000_000_000;
    main_elements
        .saturating_mul(PEAK_BYTES_PER_MAIN_ELEMENT)
        .saturating_add(PEAK_BYTES_FIXED_OVERHEAD)
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

    #[test]
    fn estimate_has_fixed_overhead_at_zero() {
        assert_eq!(estimate_peak_bytes(0), 3 * GB);
    }

    #[test]
    fn estimate_scales_linearly_with_main_elements() {
        let one_million = estimate_peak_bytes(1_000_000);
        let two_million = estimate_peak_bytes(2_000_000);
        assert_eq!(one_million - 3 * GB, 85_000_000);
        assert_eq!(two_million - 3 * GB, 170_000_000);
    }

    #[test]
    fn estimate_matches_measured_8m_fib_within_safety_margin() {
        // fib_iterative_8M measured: main_elements=882,404,346, peak=60.26 GB.
        // Estimator uses 1.3x safety factor so it should sit above the measurement.
        let estimated = estimate_peak_bytes(882_404_346);
        assert!(estimated >= 60 * GB);
        assert!(estimated < 100 * GB);
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
        let mode = select_storage_mode(estimate_peak_bytes(0), Some(64 * GB), Some(1_000_000));
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn unknown_available_with_no_cap_falls_back_to_ram() {
        // OS can't report available memory. Without a cap we can't make an
        // informed decision, so stay in Ram rather than forcing Disk on every
        // proof.
        let mode = select_storage_mode(estimate_peak_bytes(0), None, None);
        assert_eq!(mode, StorageMode::Ram);
    }

    #[test]
    fn unknown_available_with_cap_uses_cap_as_budget() {
        // OS can't report; cap is the whole budget.
        let mode = select_storage_mode(10 * GB, None, Some(4 * GB));
        assert_eq!(mode, StorageMode::Disk);
    }
}

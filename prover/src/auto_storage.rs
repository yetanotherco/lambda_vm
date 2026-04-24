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
//! fib), rounding the coefficients to `peak ≈ 85 * main + 3 GB`.

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
    main_elements.saturating_mul(PEAK_BYTES_PER_MAIN_ELEMENT) + PEAK_BYTES_FIXED_OVERHEAD
}

/// Pick a storage mode given the estimate and the machine's available RAM.
///
/// Uses 80% of available RAM as the cutoff so there's headroom for the OS,
/// other processes, and allocator fragmentation. `cap` is an optional user-
/// imposed limit (see `ProofOptions::max_ram_bytes`) which overrides the
/// machine's reported available RAM when smaller.
pub fn select_storage_mode(estimated: u64, available: u64, cap: Option<u64>) -> StorageMode {
    const SAFETY_FRACTION_NUM: u64 = 4;
    const SAFETY_FRACTION_DEN: u64 = 5;

    let budget = match cap {
        Some(c) => c.min(available),
        None => available,
    };
    let threshold = budget.saturating_mul(SAFETY_FRACTION_NUM) / SAFETY_FRACTION_DEN;

    if estimated > threshold {
        StorageMode::Disk
    } else {
        StorageMode::Ram
    }
}

/// Query the OS for currently available RAM (not total) in bytes.
pub fn available_ram_bytes() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory()
}

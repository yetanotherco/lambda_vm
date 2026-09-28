//! Where the DEEP and out-of-domain (OOD) denominators are inverted.
//!
//! By default every row inverts its own few denominators in registers: the R3
//! inverse buffers come from one row-wise kernel
//! ([`crate::inverse::invert_denoms_rowwise_dev`]), the fully resident R4 DEEP
//! kernel inverts its row's `1 + K` denominators itself
//! ([`crate::deep::deep_composition_ext3_fused_keep`]), and the single-point
//! OOD sums run row-chunked like the multi-point ones ([`crate::barycentric`]).
//!
//! `LAMBDA_VM_DEEP_INV_LEGACY=1` sends all three back to the legacy path — the
//! six-kernel global scan, the buffered DEEP kernel, one block per OOD column —
//! an A/B and rollback switch, not a tuning knob: both compute the same field
//! elements, so the same roots and proofs. Read once per process.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// The legacy switch. Only `1` selects the legacy path; the banner
/// [`rowwise_enabled`] prints says which one a process took.
pub const LEGACY_ENV: &str = "LAMBDA_VM_DEEP_INV_LEGACY";

/// Whether this process inverts the DEEP and OOD denominators row-wise.
pub fn rowwise_enabled() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| {
        let legacy = std::env::var(LEGACY_ENV).is_ok_and(|v| v == "1");
        // One line per process, so every box log states which path it ran.
        eprintln!(
            "[gpu] DEEP/OOD denominators: {}",
            if legacy {
                "the legacy scan, buffered DEEP, one block per OOD column \
                 (LAMBDA_VM_DEEP_INV_LEGACY=1)"
            } else {
                "inverted row-wise, in the DEEP kernel itself, OOD sums row-chunked"
            }
        );
        !legacy
    })
}

static ROWWISE_INVERSIONS: AtomicU64 = AtomicU64::new(0);
static CHUNKED_OOD_SUMS: AtomicU64 = AtomicU64::new(0);

/// Denominator inversions the row-wise kernel has done, process-wide: the
/// counter a test reads to know the row-wise path, and not the scan, did the
/// work.
pub fn rowwise_inversions() -> u64 {
    ROWWISE_INVERSIONS.load(Ordering::Relaxed)
}

/// Single-point OOD sums the row-chunked kernel has done, process-wide.
pub fn chunked_ood_sums() -> u64 {
    CHUNKED_OOD_SUMS.load(Ordering::Relaxed)
}

pub(crate) fn count_rowwise_inversion() {
    ROWWISE_INVERSIONS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn count_chunked_ood_sum() {
    CHUNKED_OOD_SUMS.fetch_add(1, Ordering::Relaxed);
}

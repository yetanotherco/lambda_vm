//! Where the DEEP and out-of-domain (OOD) denominators are inverted.
//!
//! By default every row inverts its own few denominators in registers: the R3
//! inverse buffers come from one row-wise kernel
//! ([`crate::inverse::invert_denoms_rowwise_dev`]), the fully resident R4 DEEP
//! kernel inverts its row's `1 + K` denominators itself
//! ([`crate::deep::deep_composition_ext3_fused_keep`]), and the single-point
//! OOD sums run row-chunked like the multi-point ones ([`crate::barycentric`]).
//!
//! The legacy paths — the six-kernel global scan, the buffered DEEP kernel,
//! one block per OOD column — stay for the shapes the row-wise kernels do not
//! take and as the parity tests' oracle (`tests/deep_inv_parity.rs`): both
//! compute the same field elements, so the same roots and proofs.

use std::sync::atomic::{AtomicU64, Ordering};

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

//! Sticky fault-injection hooks for the GPU error-path tests.
//!
//! Unlike the one-shot hooks in `fri` and `inverse` (which disarm after
//! firing, so a drain-and-retry absorbs the injected error before it can
//! surface), a sticky hook keeps failing once its armed call count is
//! reached, until explicitly disarmed. The device-decline recovery tests
//! need that: a stage falls through to its host path only when every device
//! arm of that stage declines in the same prove.

use std::sync::atomic::{AtomicI64, Ordering};

use crate::Result;

/// R3 barycentric entries (`barycentric_{base,ext3}_on_device{,_with_dev_inv_denoms}`).
pub static FAULT_BARYCENTRIC_STICKY: AtomicI64 = AtomicI64::new(-1);
/// R4 DEEP composition entries (`deep_composition_ext3*`).
pub static FAULT_DEEP_STICKY: AtomicI64 = AtomicI64::new(-1);
/// R2 comp-poly tree entries (`build_comp_poly_tree_from_{evals_ext3_keep,slabs_dev}`).
pub static FAULT_COMP_TREE_STICKY: AtomicI64 = AtomicI64::new(-1);

/// Countdown check shared by the sticky hooks: negative = disarmed (the
/// production state); N > 0 counts down across calls and the Nth call — and
/// every call after it — returns Err (the counter parks at 0); 0 therefore
/// doubles as the "fired" marker. Disarm by storing -1.
pub fn check_sticky(counter: &AtomicI64) -> Result<()> {
    let v = counter.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    if v > 0 {
        counter.fetch_sub(1, Ordering::Relaxed);
    }
    if v <= 1 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

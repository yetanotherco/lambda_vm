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
///
/// The transition is a single `fetch_update`, so concurrent table dispatches
/// (the prover runs a rayon task per table) cannot race the load against the
/// decrement: each caller walks the counter one step (the closure returns
/// `None` at `<= 0`, so it parks at 0 and never underflows), which keeps both
/// the sticky guarantee and the `== 0` fired check sound. The fire decision
/// reads `fetch_update`'s own result — `Ok(prev)` for the call that
/// decremented, `Err(cur)` for a no-op — so no second load is needed.
pub fn check_sticky(counter: &AtomicI64) -> Result<()> {
    // One atomic transition, so concurrent dispatches saturate at 0 rather
    // than underflowing: a decrement only happens from a positive value.
    let fired = counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| match v {
            n if n < 0 => None, // disarmed: never fires
            0 => None,          // already parked: stay fired (sticky)
            _ => Some(v - 1),   // count down toward the parked 0
        })
        // Ok(prev): this call decremented — the 1 → 0 step fires.
        // Err(cur): no-op — fires only if already parked at 0.
        .map_or_else(|cur| cur == 0, |prev| prev <= 1);
    if fired {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

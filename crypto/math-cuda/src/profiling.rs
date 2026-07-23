//! Profiling hooks for external GPU profilers (Nsight Systems).
//!
//! Two independent mechanisms:
//!
//! * [`crate::nvtx_range!`] — opens a thread-scoped NVTX range that closes at
//!   the end of the enclosing scope, so nsys can attribute kernels/memcpys/syncs
//!   to named pipeline phases. Compiles to nothing without the `nvtx` feature;
//!   with the feature on but no profiler attached each call is a cheap no-op.
//! * [`profiler_range_start`] / [`profiler_range_stop`] — bracket the capture
//!   window for `nsys profile -c cudaProfilerApi`, so a profile of a full prove
//!   can exclude CPU-side trace generation. Always compiled (not `nvtx`-gated):
//!   without an attached profiler the driver treats them as no-ops.

/// Ends the current thread's innermost NVTX range on drop. Created by
/// [`crate::nvtx_range!`]; not meant to be used directly.
#[cfg(feature = "nvtx")]
pub struct NvtxPopGuard;

#[cfg(feature = "nvtx")]
impl Drop for NvtxPopGuard {
    fn drop(&mut self) {
        nvtx::range_pop!();
    }
}

/// Opens an NVTX range that closes at the end of the enclosing scope.
///
/// Usage (statement position): `crate::nvtx_range!("h2d:{} cols", n);`
#[cfg(feature = "nvtx")]
#[macro_export]
macro_rules! nvtx_range {
    ($($tt:tt)*) => {
        ::nvtx::range_push!($($tt)*);
        let _nvtx_guard = $crate::profiling::NvtxPopGuard;
    };
}

/// No-op without the `nvtx` feature — arguments are not even evaluated.
#[cfg(not(feature = "nvtx"))]
#[macro_export]
macro_rules! nvtx_range {
    ($($tt:tt)*) => {};
}

/// Opens the external-profiler capture window (`nsys profile -c cudaProfilerApi`).
///
/// Initializes the CUDA backend first (the profiler API needs a current
/// context); silently does nothing when no GPU is available or no profiler is
/// attached, so callers may invoke it unconditionally.
pub fn profiler_range_start() {
    if crate::device::backend().is_ok() {
        let _ = cudarc::driver::profiler_start();
    }
}

/// Closes the external-profiler capture window. See [`profiler_range_start`].
pub fn profiler_range_stop() {
    if crate::device::backend().is_ok() {
        let _ = cudarc::driver::profiler_stop();
    }
}

//! In-process GPU phase timing via CUDA events (feature `instruments`).
//!
//! Companion to the NVTX ranges in [`crate::profiling`]: NVTX needs nsys
//! attached; this module produces the same per-phase device timing from inside
//! the process, so every run (CI, benches) can report GPU time without a
//! profiler.
//!
//! Mechanism: [`crate::gpu_span!`] records a pair of `CudaEvent`s on the op's
//! stream — one at scope entry, one at scope exit — and pushes them to a global
//! collector **without synchronizing**. At the end of the prove,
//! [`drain_timeline`] resolves every pair against a shared epoch event
//! (`cuEventElapsedTime` uses the GPU's global clock, so spans from different
//! streams land on one timeline) and returns host-clock-aligned spans.
//!
//! Runtime gate: collection only happens when `LAMBDA_VM_GPU_TIMELINE=1` (or
//! `LAMBDA_VM_GPU_TIMELINE_JSON` is set), so an `instruments` build pays
//! nothing unless asked to measure. Compile-time gate: without the
//! `instruments` feature everything here is a no-op and [`crate::gpu_span!`]
//! expands to nothing.

#[cfg(not(feature = "instruments"))]
use cudarc::driver::CudaStream;

/// Host wall time spent inside a `stream.synchronize()`, by label.
#[cfg(feature = "instruments")]
#[derive(Debug, Clone)]
pub struct SyncWait {
    pub label: &'static str,
    pub wall_ns: u64,
}

/// One resolved device span: GPU-clock offsets from the epoch, in ms.
#[cfg(feature = "instruments")]
#[derive(Debug, Clone)]
pub struct DevSpan {
    pub label: String,
    /// Raw `CUstream` handle value — groups spans by stream; renumber for display.
    pub stream: u64,
    pub start_ms: f64,
    pub end_ms: f64,
}

/// Everything collected during a prove, resolved and host-aligned.
#[cfg(feature = "instruments")]
#[derive(Debug, Clone, Default)]
pub struct GpuTimeline {
    /// Wall-clock epoch (ns since UNIX_EPOCH) corresponding to device offset 0.
    pub host_epoch_ns: u128,
    pub spans: Vec<DevSpan>,
    pub syncs: Vec<SyncWait>,
}

#[cfg(feature = "instruments")]
mod imp {
    use super::{DevSpan, GpuTimeline, SyncWait};
    use cudarc::driver::sys::CUevent_flags;
    use cudarc::driver::{CudaEvent, CudaStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct PendingSpan {
        label: String,
        stream: u64,
        start: CudaEvent,
        end: CudaEvent,
    }

    struct State {
        epoch: CudaEvent,
        host_epoch_ns: u128,
        spans: Vec<PendingSpan>,
        syncs: Vec<SyncWait>,
    }

    // Two-level gate: `ENABLED` is the cheap env check; `STATE` is created by
    // the first span (needs a stream to record the epoch event on).
    static ENABLED: OnceLock<bool> = OnceLock::new();
    static STATE: Mutex<Option<State>> = Mutex::new(None);
    // Set on any event-API failure so we stop trying (e.g. exotic devices);
    // never panics the prover.
    static POISONED: AtomicBool = AtomicBool::new(false);

    pub fn enabled() -> bool {
        *ENABLED.get_or_init(|| {
            std::env::var_os("LAMBDA_VM_GPU_TIMELINE").is_some_and(|v| v != "0")
                || std::env::var_os("LAMBDA_VM_GPU_TIMELINE_JSON").is_some()
        }) && !POISONED.load(Ordering::Relaxed)
    }

    /// RAII guard: records the end event on drop and files the span.
    pub struct GpuSpanGuard {
        label: Option<String>,
        stream: Arc<CudaStream>,
        start: Option<CudaEvent>,
    }

    impl Drop for GpuSpanGuard {
        fn drop(&mut self) {
            let (Some(label), Some(start)) = (self.label.take(), self.start.take()) else {
                return;
            };
            let Ok(end) = self
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            else {
                POISONED.store(true, Ordering::Relaxed);
                return;
            };
            let mut st = STATE.lock().unwrap();
            if let Some(st) = st.as_mut() {
                st.spans.push(PendingSpan {
                    label,
                    stream: self.stream.cu_stream() as u64,
                    start,
                    end,
                });
            }
        }
    }

    /// Open a device span on `stream`. The label closure only runs when
    /// collection is enabled. Returns `None` (no-op) when disabled.
    pub fn gpu_span(
        stream: &Arc<CudaStream>,
        label: impl FnOnce() -> String,
    ) -> Option<GpuSpanGuard> {
        if !enabled() {
            return None;
        }
        ensure_epoch(stream)?;
        let start = match stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT)) {
            Ok(e) => e,
            Err(_) => {
                POISONED.store(true, Ordering::Relaxed);
                return None;
            }
        };
        Some(GpuSpanGuard {
            label: Some(label()),
            stream: Arc::clone(stream),
            start: Some(start),
        })
    }

    /// Record the shared epoch event once. Uses the first span's stream: the
    /// host timestamp is taken after `epoch.synchronize()`, i.e. at the moment
    /// the event actually completed on device, so GPU offsets and wall clock
    /// stay aligned even if that stream had queued work.
    fn ensure_epoch(stream: &Arc<CudaStream>) -> Option<()> {
        let mut st = STATE.lock().unwrap();
        if st.is_some() {
            return Some(());
        }
        let epoch = stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .ok()?;
        epoch.synchronize().ok()?;
        let host_epoch_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        *st = Some(State {
            epoch,
            host_epoch_ns,
            spans: Vec::with_capacity(4096),
            syncs: Vec::with_capacity(1024),
        });
        Some(())
    }

    /// `stream.synchronize()` that also records how long the host blocked.
    pub fn timed_sync(
        stream: &CudaStream,
        label: &'static str,
    ) -> Result<(), cudarc::driver::DriverError> {
        if !enabled() {
            return stream.synchronize();
        }
        let t0 = std::time::Instant::now();
        let r = stream.synchronize();
        let wall_ns = t0.elapsed().as_nanos() as u64;
        let mut st = STATE.lock().unwrap();
        if let Some(st) = st.as_mut() {
            st.syncs.push(SyncWait { label, wall_ns });
        }
        r
    }

    /// Resolve all pending spans against the epoch and reset the collector.
    /// Call after the final sync of the prove; `elapsed_ms` synchronizes each
    /// event internally, so this is safe (if slow) even mid-flight.
    pub fn drain_timeline() -> Option<GpuTimeline> {
        let state = STATE.lock().unwrap().take()?;
        let mut spans = Vec::with_capacity(state.spans.len());
        for p in state.spans {
            let (Ok(s), Ok(e)) = (
                state.epoch.elapsed_ms(&p.start),
                state.epoch.elapsed_ms(&p.end),
            ) else {
                continue;
            };
            spans.push(DevSpan {
                label: p.label,
                stream: p.stream,
                start_ms: s as f64,
                end_ms: e as f64,
            });
        }
        spans.sort_by(|a, b| a.start_ms.total_cmp(&b.start_ms));
        Some(GpuTimeline {
            host_epoch_ns: state.host_epoch_ns,
            spans,
            syncs: state.syncs,
        })
    }
}

#[cfg(feature = "instruments")]
pub use imp::{GpuSpanGuard, drain_timeline, enabled, gpu_span, timed_sync};

/// Without `instruments`: plain synchronize, no recording.
#[cfg(not(feature = "instruments"))]
pub fn timed_sync(
    stream: &CudaStream,
    _label: &'static str,
) -> Result<(), cudarc::driver::DriverError> {
    stream.synchronize()
}

/// Opens a device span on `$stream` (an `&Arc<CudaStream>`) that closes at the
/// end of the enclosing scope. Statement position, like [`crate::nvtx_range!`]:
/// `crate::gpu_span!(&stream, "gpu:lde_row_major");`
/// No-op without the `instruments` feature; with it, the label is only
/// formatted when `LAMBDA_VM_GPU_TIMELINE` is set.
#[cfg(feature = "instruments")]
#[macro_export]
macro_rules! gpu_span {
    ($stream:expr, $($tt:tt)*) => {
        let _gpu_span_guard = $crate::timing::gpu_span($stream, || format!($($tt)*));
    };
}

/// No-op without the `instruments` feature — arguments are not evaluated.
#[cfg(not(feature = "instruments"))]
#[macro_export]
macro_rules! gpu_span {
    ($stream:expr, $($tt:tt)*) => {};
}

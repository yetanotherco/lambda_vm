//! Minimal NVTX bindings so Nsight Systems timelines show named host-side
//! ranges (mirrored instruments spans — prover phases and per-epoch marks)
//! instead of a wall of anonymous CUDA API calls.
//!
//! Loading mirrors the crate's cudarc `dynamic-loading` philosophy: no
//! build-time or link-time dependency on the CUDA toolkit layout. At first use
//! we dlopen `libnvToolsExt.so` (NVTX v2, shipped with every CUDA toolkit and
//! honored by nsys/ncu); when it is absent every call is a cheap no-op, so a
//! `--features nvtx` binary runs unchanged on machines without the library.
//! Override the library path with `LAMBDA_VM_NVTX_LIB` if it lives somewhere
//! unusual.
//!
//! With the `nvtx` cargo feature *disabled* this module compiles to empty
//! inline stubs — the label closures passed to [`Range::fmt`] are never
//! evaluated and the whole thing vanishes.
//!
//! Semantics note: a [`Range`] measures the *host-side* extent of a call. For
//! the `_keep`/`_dev` entry points that enqueue async GPU work without
//! syncing, kernels execute after the range closes — that is fine: nsys
//! correlates each kernel to the range that launched it.

pub use imp::*;

#[cfg(feature = "nvtx")]
mod imp {
    use std::ffi::{CString, c_char, c_int};
    use std::marker::PhantomData;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    struct Api {
        // Field order is drop order; the fn pointers are only valid while the
        // library is loaded, and both live for the whole process anyway
        // (static OnceLock).
        range_push: unsafe extern "C" fn(*const c_char) -> c_int,
        range_pop: unsafe extern "C" fn() -> c_int,
        mark: unsafe extern "C" fn(*const c_char),
        _lib: libloading::Library,
    }
    // SAFETY: the NVTX v2 API is thread-safe (push/pop stacks are per-thread)
    // and the Library handle is only kept alive, never re-entered.
    unsafe impl Send for Api {}
    unsafe impl Sync for Api {}

    fn nvtx_lib_candidates() -> Vec<PathBuf> {
        let mut c = Vec::new();
        if let Some(p) = std::env::var_os("LAMBDA_VM_NVTX_LIB") {
            c.push(PathBuf::from(p));
        }
        c.push(PathBuf::from("libnvToolsExt.so.1"));
        c.push(PathBuf::from("libnvToolsExt.so"));
        let cuda_home = std::env::var_os("CUDA_HOME")
            .or_else(|| std::env::var_os("CUDA_PATH"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
        c.push(cuda_home.join("lib64").join("libnvToolsExt.so.1"));
        c.push(cuda_home.join("lib64").join("libnvToolsExt.so"));
        c
    }

    fn api() -> Option<&'static Api> {
        static API: OnceLock<Option<Api>> = OnceLock::new();
        API.get_or_init(|| {
            for path in nvtx_lib_candidates() {
                // SAFETY: loading a shared library runs its initializers;
                // libnvToolsExt is NVIDIA's stub dispatcher with no side
                // effects beyond tool injection.
                let Ok(lib) = (unsafe { libloading::Library::new(&path) }) else {
                    continue;
                };
                // SAFETY: signatures match the NVTX v2 C API.
                let syms = unsafe {
                    (
                        lib.get::<unsafe extern "C" fn(*const c_char) -> c_int>(
                            b"nvtxRangePushA\0",
                        )
                        .map(|s| *s),
                        lib.get::<unsafe extern "C" fn() -> c_int>(b"nvtxRangePop\0")
                            .map(|s| *s),
                        lib.get::<unsafe extern "C" fn(*const c_char)>(b"nvtxMarkA\0")
                            .map(|s| *s),
                    )
                };
                if let (Ok(range_push), Ok(range_pop), Ok(mark)) = syms {
                    return Some(Api {
                        range_push,
                        range_pop,
                        mark,
                        _lib: lib,
                    });
                }
            }
            None
        })
        .as_ref()
    }

    /// True when libnvToolsExt was found; use to skip label formatting work.
    #[inline]
    pub fn is_active() -> bool {
        api().is_some()
    }

    fn push_str(api: &Api, name: &str) {
        // NVTX takes a NUL-terminated C string; a label containing NUL is a
        // bug we don't care to surface here — fall back to a fixed name.
        let c = CString::new(name).unwrap_or_else(|_| CString::new("invalid-label").unwrap());
        // SAFETY: `c` is a valid NUL-terminated string for the duration of the call.
        unsafe { (api.range_push)(c.as_ptr()) };
    }

    /// Push a range on this thread's NVTX stack. Prefer [`Range`]; this raw
    /// form exists for RAII guards that live in other crates (instruments).
    #[inline]
    pub fn range_push(name: &str) {
        if let Some(api) = api() {
            push_str(api, name);
        }
    }

    /// Pop this thread's innermost NVTX range. Must pair with [`range_push`].
    #[inline]
    pub fn range_pop() {
        if let Some(api) = api() {
            // SAFETY: no arguments; unbalanced pops are handled by NVTX (no-op).
            unsafe { (api.range_pop)() };
        }
    }

    /// Instantaneous marker on the timeline.
    #[inline]
    pub fn mark(name: &str) {
        if let Some(api) = api() {
            let c = CString::new(name).unwrap_or_else(|_| CString::new("invalid-label").unwrap());
            // SAFETY: `c` is a valid NUL-terminated string for the duration of the call.
            unsafe { (api.mark)(c.as_ptr()) };
        }
    }

    /// RAII NVTX range: pushed on construction, popped on drop. `!Send` on
    /// purpose — NVTX push/pop stacks are per-thread, so a guard must drop on
    /// the thread that created it.
    pub struct Range {
        pushed: bool,
        _not_send: PhantomData<*const ()>,
    }

    impl Range {
        #[inline]
        pub fn new(name: &str) -> Range {
            let pushed = api().map(|a| push_str(a, name)).is_some();
            Range {
                pushed,
                _not_send: PhantomData,
            }
        }

        /// Like [`Range::new`] but the label is only formatted when a
        /// profiler-visible NVTX library is actually loaded.
        #[inline]
        pub fn fmt<F: FnOnce() -> String>(label: F) -> Range {
            if is_active() {
                Range::new(&label())
            } else {
                Range {
                    pushed: false,
                    _not_send: PhantomData,
                }
            }
        }
    }

    impl Drop for Range {
        fn drop(&mut self) {
            if self.pushed {
                range_pop();
            }
        }
    }

    // --- CUDA profiler capture-range control -------------------------------
    //
    // cuProfilerStart/Stop gate `nsys profile --capture-range=cudaProfilerApi`,
    // letting a session capture one phase/epoch of a long prove instead of the
    // whole run. Loaded from libcuda (already resident via cudarc) — separate
    // dlopen so this module stays independent of cudarc's bound symbol set.

    struct ProfilerApi {
        start: unsafe extern "C" fn() -> c_int,
        stop: unsafe extern "C" fn() -> c_int,
        _lib: libloading::Library,
    }
    unsafe impl Send for ProfilerApi {}
    unsafe impl Sync for ProfilerApi {}

    fn profiler_api() -> Option<&'static ProfilerApi> {
        static API: OnceLock<Option<ProfilerApi>> = OnceLock::new();
        API.get_or_init(|| {
            for name in ["libcuda.so.1", "libcuda.so"] {
                // SAFETY: libcuda is loaded by cudarc already; this bumps a refcount.
                let Ok(lib) = (unsafe { libloading::Library::new(name) }) else {
                    continue;
                };
                // SAFETY: signatures match the CUDA driver profiler API.
                let syms = unsafe {
                    (
                        lib.get::<unsafe extern "C" fn() -> c_int>(b"cuProfilerStart\0")
                            .map(|s| *s),
                        lib.get::<unsafe extern "C" fn() -> c_int>(b"cuProfilerStop\0")
                            .map(|s| *s),
                    )
                };
                if let (Ok(start), Ok(stop)) = syms {
                    return Some(ProfilerApi {
                        start,
                        stop,
                        _lib: lib,
                    });
                }
            }
            None
        })
        .as_ref()
    }

    /// Begin a profiler capture range (`nsys --capture-range=cudaProfilerApi`).
    /// No-op without libcuda or outside a profiler session.
    #[inline]
    pub fn profiler_start() {
        if let Some(api) = profiler_api() {
            // SAFETY: no arguments; valid to call any time after libcuda loads.
            unsafe { (api.start)() };
        }
    }

    /// End a profiler capture range. Must pair with [`profiler_start`].
    #[inline]
    pub fn profiler_stop() {
        if let Some(api) = profiler_api() {
            // SAFETY: no arguments; valid to call any time after libcuda loads.
            unsafe { (api.stop)() };
        }
    }
}

#[cfg(not(feature = "nvtx"))]
mod imp {
    /// No-op stub; see the `nvtx`-feature implementation above.
    pub struct Range;

    impl Range {
        #[inline(always)]
        pub fn new(_name: &str) -> Range {
            Range
        }
        #[inline(always)]
        pub fn fmt<F: FnOnce() -> String>(_label: F) -> Range {
            Range
        }
    }

    #[inline(always)]
    pub fn is_active() -> bool {
        false
    }
    #[inline(always)]
    pub fn range_push(_name: &str) {}
    #[inline(always)]
    pub fn range_pop() {}
    #[inline(always)]
    pub fn mark(_name: &str) {}
    #[inline(always)]
    pub fn profiler_start() {}
    #[inline(always)]
    pub fn profiler_stop() {}
}

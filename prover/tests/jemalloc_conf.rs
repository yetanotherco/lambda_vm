//! jemalloc's purge policy is compiled into the binary — checked by reading it
//! back out of the allocator serving this process.
//!
//! `bin/cli/src/main.rs` and `calibration.rs` each export
//! `_rjem_malloc_conf = "dirty_decay_ms:-1,muzzy_decay_ms:-1"` beside their
//! `#[global_allocator]`, so the never-purge policy travels in the binary
//! rather than in a launcher's environment. Nothing about that export is
//! checked by the compiler: a misspelled symbol, a wrong value type, or a
//! jemalloc built without the `_rjem_` prefix each leave a binary that
//! compiles, links, runs — and quietly purges.
//!
//! This is its own test binary because the check needs a jemalloc process of
//! its own: the prover's lib tests run under the platform allocator, where a
//! `mallctl` read would say nothing, and `calibration.rs` is behind
//! `disk-spill` and pays for a full proof. What it pins is the export pattern —
//! symbol, type, initializer, edition spelling — in the copy below, which is
//! byte-identical to the two production sites but not mechanically tied to
//! them: delete either of those and this still passes. It is a self-test of the
//! pattern, not a regression guard on the two sites that ship it. That the
//! shipped `cli` binary carries the symbol is a link-time property, read with
//! `nm` rather than asserted here.

use tikv_jemalloc_ctl::raw;

#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// The same string the two production sites export, character for character.
const NEVER_PURGE: &[u8] = b"dirty_decay_ms:-1,muzzy_decay_ms:-1\0";

#[allow(non_upper_case_globals)]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static malloc_conf: Option<&'static core::ffi::c_char> =
    Some(unsafe { &*(NEVER_PURGE.as_ptr() as *const core::ffi::c_char) });

/// jemalloc's default `opt.dirty_decay_ms`, quoted in the failure message so a
/// red test says which value it found and where that value comes from.
const DEFAULT_DIRTY_DECAY_MS: isize = 10_000;

#[test]
fn jemalloc_never_purge_is_compiled_in() {
    // `_RJEM_MALLOC_CONF` sets these same options from the environment, and
    // benchmark runs do set it; with it set, reading `-1` back would say nothing
    // about the compiled-in export, so refuse to run rather than pass for the
    // wrong reason. Plain `MALLOC_CONF` is inert in this prefixed build — guarded
    // anyway, so that a future unprefixed build does not silently pass here.
    // Not covered: `/etc/_rjem_malloc.conf`, the one remaining source that could
    // set `opt.*` from outside this binary.
    for var in ["_RJEM_MALLOC_CONF", "MALLOC_CONF"] {
        assert!(
            std::env::var_os(var).is_none(),
            "{var} is set in this process's environment. jemalloc reads \
             `_RJEM_MALLOC_CONF` (this build is prefixed), which sets `opt.*` on \
             its own, so this test could not tell the compiled-in export from the \
             environment; unset it and re-run."
        );
    }

    // `opt.dirty_decay_ms` and `opt.muzzy_decay_ms` are jemalloc `ssize_t`s.
    // `raw::read` asserts the mallctl's width equals `size_of::<T>()`, so a
    // wrong Rust width fails here rather than reading a truncated value.
    let dirty: isize =
        unsafe { raw::read(b"opt.dirty_decay_ms\0") }.expect("opt.dirty_decay_ms is readable");
    let muzzy: isize =
        unsafe { raw::read(b"opt.muzzy_decay_ms\0") }.expect("opt.muzzy_decay_ms is readable");

    assert_eq!(
        dirty, -1,
        "opt.dirty_decay_ms is {dirty}, not -1 (jemalloc's default is \
         {DEFAULT_DIRTY_DECAY_MS}): the `_rjem_malloc_conf` export beside this \
         file's `#[global_allocator]` is missing, misspelled, or was not read, \
         and a binary built this way returns dirty pages to the OS on a timer"
    );
    assert_eq!(
        muzzy, -1,
        "opt.muzzy_decay_ms is {muzzy}, not -1 (jemalloc's default is 0): the \
         `_rjem_malloc_conf` export beside this file's `#[global_allocator]` is \
         missing, misspelled, or was not read, and a binary built this way \
         unmaps muzzy pages immediately"
    );
}

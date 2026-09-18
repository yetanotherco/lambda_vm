//! The shipped allocator's purge policy is compiled into this binary — checked
//! by reading it back out of the allocator that is serving this process.
//!
//! `bin/cli/src/main.rs` and `lib.rs`'s test harness each export
//! `_rjem_malloc_conf = "dirty_decay_ms:-1,muzzy_decay_ms:-1"` beside their
//! `#[global_allocator]`. Nothing about that export is checked by the compiler:
//! a misspelled symbol, a wrong value type, or a jemalloc built without the
//! `_rjem_` prefix each leave a binary that compiles, links, runs — and quietly
//! purges. The only evidence that the string was read is jemalloc's own view of
//! its options.

use tikv_jemalloc_ctl::raw;

/// jemalloc's default `opt.dirty_decay_ms`, quoted in the failure message so a
/// red test says which value it found and where that value comes from.
const DEFAULT_DIRTY_DECAY_MS: isize = 10_000;

#[test]
fn jemalloc_never_purge_is_compiled_in() {
    // The same options can be set from the environment (`MALLOC_CONF` /
    // `_RJEM_MALLOC_CONF`) — which is how the box ran the arms that measured
    // this setting. With either set, reading `-1` back would say nothing about
    // the compiled-in export, so refuse to run rather than pass for the wrong
    // reason.
    for var in ["MALLOC_CONF", "_RJEM_MALLOC_CONF"] {
        assert!(
            std::env::var_os(var).is_none(),
            "{var} is set in this process's environment. It sets `opt.*` on its \
             own, so this test could not tell the compiled-in export from the \
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
         {DEFAULT_DIRTY_DECAY_MS}): the `_rjem_malloc_conf` export beside the \
         `#[global_allocator]` in `prover/src/lib.rs` is missing, misspelled, or \
         was not read, and this binary returns dirty pages to the OS on a timer"
    );
    assert_eq!(
        muzzy, -1,
        "opt.muzzy_decay_ms is {muzzy}, not -1 (jemalloc's default is 0): the \
         `_rjem_malloc_conf` export beside the `#[global_allocator]` in \
         `prover/src/lib.rs` is missing, misspelled, or was not read, and this \
         binary unmaps muzzy pages immediately"
    );
}

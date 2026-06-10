//! Thread-local hit/miss counter for the `multi_prove` domain dedup cache.
//!
//! Used only by tests that exercise the dedup behavior (See
//! `tests::prover_tests::test_multi_prove_dedups_shared_domain_params`).
//! Recorded from inside `multi_prove` under `#[cfg(test)]` and read back
//! by the test after proving completes.

use std::cell::Cell;

thread_local! {
    static COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

pub(crate) fn reset() {
    COUNTS.with(|c| c.set((0, 0)));
}

pub(crate) fn get() -> (usize, usize) {
    COUNTS.with(Cell::get)
}

pub(crate) fn record(was_hit: bool) {
    COUNTS.with(|c| {
        let (hits, misses) = c.get();
        c.set(if was_hit {
            (hits + 1, misses)
        } else {
            (hits, misses + 1)
        });
    });
}

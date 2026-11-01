//! ★ The base-only declaration: the one thing that lets a block's base proof
//! be made under another hash than the recursion's (D-B3HYB S3, the BLAKE3
//! base experiment).
//!
//! The block path proves one hash end to end (`hash_pin`, Mauro 10-10): no base
//! of one hash under another's recursion. A base proved with NO recursion over
//! it is outside that rule, and that is the only case this module admits: a
//! process [`declare`]s, once, that it proves block bases and nothing above
//! them — `lambda-vm prove-block-base` does, before anything is proved — and
//! from then on
//!
//! - the block's base prover and verifier admit the 7-round BLAKE3 base
//!   (`block_whir::checked_base_only`, `with_base_hash!`), and
//! - every recursion entry point refuses to run at all, under any base
//!   ([`refuse_recursion`]).
//!
//! The declaration is a one-way latch: nothing clears it, so a process that
//! could have made a BLAKE3 base can never reach a recursion. Nothing reads the
//! environment here — the CLI's subcommand is the declaration — and nothing
//! about the default moves: a process that never declares proves exactly
//! today's Poseidon1 block, base and recursion alike.

use std::sync::atomic::{AtomicBool, Ordering};

static DECLARED: AtomicBool = AtomicBool::new(false);

/// Declare this process base-only. Idempotent; never undone.
pub fn declare() {
    DECLARED.store(true, Ordering::SeqCst);
}

/// Whether this process declared it proves block bases only.
pub fn declared() -> bool {
    DECLARED.load(Ordering::SeqCst)
}

#[cfg(test)]
thread_local! {
    /// Test only: this thread holds the base-only permit (a test process runs
    /// the recursion's tests beside the base's, so the latch is per thread
    /// there). The base's admission reads it on the calling thread, before
    /// any worker is spawned.
    static PERMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test only: run `f` with this thread's base-only permit held.
#[cfg(test)]
pub(crate) fn with_permit<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            PERMIT.with(|p| p.set(self.0));
        }
    }
    let _restore = Restore(PERMIT.with(|p| p.replace(true)));
    f()
}

/// Whether a base other than the recursion's may be admitted here: the
/// process declared base-only, or (tests) this thread holds the permit.
pub fn admitted() -> bool {
    #[cfg(test)]
    if PERMIT.with(std::cell::Cell::get) {
        return true;
    }
    declared()
}

/// The recursion's refusal in a base-only process: `Err` naming the
/// declaration once [`declare`] ran, `Ok` otherwise. Every recursion entry
/// point calls it before it proves or verifies anything.
pub fn refuse_recursion() -> Result<(), String> {
    if declared() {
        return Err(
            "this process declared it proves block bases only (prove-block-base): no recursion \
             runs in it, over any base"
                .to_string(),
        );
    }
    Ok(())
}

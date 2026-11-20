//! ★ The base-only declaration: the one thing that lets a block's base proof
//! be made under another hash than the recursion's (the opt-in BLAKE3
//! base, proved with no recursion over it).
//!
//! The block path proves one hash end to end (`hash_pin`): no base
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
//!
//! # A BLAKE3 base under a BLAKE3 recursion
//!
//! The other case the rule admits is a base whose recursion is the same hash:
//! the BLAKE3 levels (`lfm::levels` under `WlfmFamily::B3`) verify a 7-round
//! BLAKE3 base in-guest and prove every level under BLAKE3, one hash end to
//! end. Their driver proves its base inside [`with_blake3_levels`], which
//! admits the BLAKE3 base on the calling thread for that prove alone. The
//! Poseidon1 recursion admits its base through `block_whir::checked_base`,
//! which refuses a BLAKE3 base whatever is held here, so no base of one hash is
//! verified under another's recursion; and a process that declared base-only
//! refuses the levels too ([`refuse_recursion`]).

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

thread_local! {
    /// This thread is proving the base of a block whose recursion is BLAKE3
    /// too ([`with_blake3_levels`]).
    static BLAKE3_LEVELS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f`, the base prove of a block whose levels are BLAKE3 (one hash end to
/// end), with the 7-round BLAKE3 base admitted on this thread. The admission
/// is read on the calling thread, at the base's prove entry, before any worker
/// is spawned; it ends with `f`, panic or not. Only the base's own prove and
/// verify sites read it (`block_whir::checked_base_only`): the Poseidon1
/// recursion's admission (`block_whir::checked_base`) refuses a BLAKE3 base
/// inside it as outside.
pub(crate) fn with_blake3_levels<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            BLAKE3_LEVELS.with(|p| p.set(self.0));
        }
    }
    let _restore = Restore(BLAKE3_LEVELS.with(|p| p.replace(true)));
    f()
}

/// Whether a base other than the recursion's may be admitted here: the
/// process declared base-only, this thread is proving a BLAKE3 levels block's
/// base ([`with_blake3_levels`]), or (tests) this thread holds the permit.
pub fn admitted() -> bool {
    #[cfg(test)]
    if PERMIT.with(std::cell::Cell::get) {
        return true;
    }
    if BLAKE3_LEVELS.with(std::cell::Cell::get) {
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

#[cfg(test)]
mod tests {
    use crate::block_whir::{checked_base, checked_base_only};
    use stark::proof::options::BaseFormat;

    /// The BLAKE3 levels' admission holds for its scope on its thread alone:
    /// the base's own admission takes the BLAKE3 base inside it and refuses it
    /// before, after, and on another thread; the recursion's admission refuses
    /// it inside it too; and a panic inside it still ends it.
    #[test]
    fn the_blake3_levels_admission_is_scoped_to_its_prove() {
        // On a fresh thread: no declaration, no permit, nothing held.
        std::thread::spawn(|| {
            let b3 = BaseFormat::B3_WHIR;
            assert!(!super::declared());
            assert!(checked_base_only(&b3).is_err(), "refused before");
            super::with_blake3_levels(|| {
                assert!(checked_base_only(&b3).is_ok(), "admitted inside");
                assert!(
                    checked_base(&b3).is_err(),
                    "the recursion's admission refuses it inside"
                );
                let other = std::thread::spawn(move || checked_base_only(&b3).is_err());
                assert!(other.join().expect("joins"), "another thread is refused");
                super::with_blake3_levels(|| assert!(checked_base_only(&b3).is_ok()));
                assert!(
                    checked_base_only(&b3).is_ok(),
                    "a nested scope restores it held"
                );
            });
            assert!(checked_base_only(&b3).is_err(), "refused after");
            let _ = std::panic::catch_unwind(|| {
                super::with_blake3_levels(|| panic!("inside the scope"));
            });
            assert!(
                checked_base_only(&b3).is_err(),
                "refused after a panic inside"
            );
            assert!(
                checked_base_only(&crate::hash_pin::BLOCK_BASE).is_ok(),
                "the block family's base is admitted throughout"
            );
        })
        .join()
        .expect("the checks hold");
    }
}

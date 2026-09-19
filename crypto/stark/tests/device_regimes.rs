//! One card, one admission regime per process: the refusal, the wait, the
//! exclusion itself, and the two same-thread cases that would otherwise be
//! hangs rather than errors.
//!
//! The rule is asymmetric on purpose. A live byte-budget gate REFUSES a serial
//! window, because that gate lives for a whole `multi_prove`; an open window
//! makes a gate claim WAIT, because a window lives for one round-1 call. Two
//! tests here are that asymmetry, one per direction.
//!
//! These run on a card-free build. Nothing here builds a proof or touches a
//! device; the rule they check is pure process state, which is exactly why it
//! can be checked without one.
//!
//! WHY THIS IS AN INTEGRATION TEST AND NOT A UNIT TEST. The registry is
//! process-wide, and the stark crate's unit-test binary also runs several
//! `multi_prove` tests, each of which builds the byte-budget gate. A test in
//! that binary holding a serial window would make a concurrent `multi_prove`
//! refuse and fail an unrelated test. Cargo gives each integration file its
//! own process, so nothing here can collide with those — and the lock below
//! keeps these five from colliding with each other.
//!
//! ⛔ SO DO NOT MOVE THEM INTO THE UNIT TESTS FOR TIDINESS. The named tests are
//! `prove_verify_roundtrip_tests` and `air_tests`, both of which reach
//! `test_utils::multi_prove_ram` and `.unwrap()` the result; a window held
//! anywhere in that binary turns one of them red at random. The isolation is
//! the point of the file, not an accident of where it was written.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use stark::device_window::{ByteBudgetRegime, SerialWindow};

/// Every test here manipulates the one process-wide registry, so they take
/// turns. Recovered rather than unwrapped so one failing test reports its own
/// assertion instead of poisoning the rest into a second, misleading panic.
static SERIALISE: Mutex<()> = Mutex::new(());

fn exclusive() -> MutexGuard<'static, ()> {
    SERIALISE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn a_serial_window_refuses_while_a_byte_budget_regime_is_claimed() {
    let _serialised = exclusive();

    // The state it must ACCEPT.
    let window = SerialWindow::enter().expect("nothing is claimed, so the window opens");
    drop(window);

    // The state it must REFUSE, manufactured rather than waited for.
    let regime = ByteBudgetRegime::claim().expect("no window is open, so the regime is claimable");
    let refused = SerialWindow::enter()
        .expect_err("a claimed byte-budget regime must refuse the serial window");
    assert!(
        refused
            .message()
            .contains("byte-budget admission gate is live"),
        "the refusal must name the regime that was already live, got: {refused}"
    );

    // And it accepts again once that regime is released, so the refusal is a
    // rule about the state and not a permanent failure.
    drop(regime);
    SerialWindow::enter().expect("the regime was released, so the window opens again");
}

/// The asymmetric half: the gate WAITS for a window another thread holds,
/// rather than refusing it. A window lives for one round-1 call, so the wait
/// is bounded; the gate lives for a whole `multi_prove`, which is why the
/// other direction still refuses.
#[test]
fn a_byte_budget_regime_waits_for_an_open_serial_window() {
    let _serialised = exclusive();

    // The state it takes immediately.
    let regime = ByteBudgetRegime::claim().expect("nothing is open, so the regime is claimable");
    drop(regime);

    // The state it must WAIT through, read the way the exclusion test reads
    // its own: the holder clears the flag BEFORE releasing, so a claim that
    // waited sees false and a claim that walked straight in sees true.
    let inside = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let holder_inside = Arc::clone(&inside);
    let holder = std::thread::spawn(move || {
        let window = SerialWindow::enter().expect("the window opens");
        holder_inside.store(true, Ordering::SeqCst);
        tx.send(()).expect("the test thread is still listening");
        std::thread::sleep(Duration::from_millis(50));
        holder_inside.store(false, Ordering::SeqCst);
        drop(window);
    });

    rx.recv().expect("the holder thread took the window");
    let regime = ByteBudgetRegime::claim().expect("the claim waits for the window, never refuses");
    assert!(
        !inside.load(Ordering::SeqCst),
        "the byte-budget gate was claimed while a serial window was still open: it did not wait"
    );

    drop(regime);
    holder.join().expect("the holder thread finished");
}

/// The one refusal that survives on the gate side, and the reason it has to:
/// this claim would wait for a window only its own caller can release.
///
/// ⚠ Its failure mode is a HANG, not a red assertion — the same shape as the
/// window's own re-entry test.
#[test]
fn a_thread_holding_the_window_is_refused_a_gate_rather_than_deadlocked() {
    let _serialised = exclusive();

    let window = SerialWindow::enter().expect("the window opens");
    let refused = ByteBudgetRegime::claim()
        .expect_err("a claim from the thread holding the window must be refused, not blocked");
    assert!(
        refused.message().contains("only this thread can release"),
        "the refusal must say whose window it is waiting on, got: {refused}"
    );

    drop(window);
    ByteBudgetRegime::claim().expect("the window was released, so the regime is claimable again");
}

#[test]
fn two_serial_windows_cannot_overlap() {
    let _serialised = exclusive();

    let inside = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let holder_inside = Arc::clone(&inside);
    let holder = std::thread::spawn(move || {
        let window = SerialWindow::enter().expect("the first window opens");
        holder_inside.store(true, Ordering::SeqCst);
        tx.send(()).expect("the test thread is still listening");
        std::thread::sleep(Duration::from_millis(50));
        // Cleared BEFORE the release. A slow machine can only make the second
        // thread wait longer; it cannot make the assertion below pass by
        // accident, because the flag is already false by the time the window
        // is free. The only way to observe `true` is to have been let in
        // while the first window was still held.
        holder_inside.store(false, Ordering::SeqCst);
        drop(window);
    });

    rx.recv().expect("the holder thread took the window");
    let second = SerialWindow::enter().expect("the window is free once the holder released it");
    assert!(
        !inside.load(Ordering::SeqCst),
        "a second window opened while the first was still held: the device window is not exclusive"
    );

    drop(second);
    holder.join().expect("the holder thread finished");
}

/// ⚠ This one's failure mode is a HANG, not a red assertion: without the
/// re-entry check the second `enter` blocks on a window this very thread
/// holds, forever. A timeout here means the check is gone, not that the test
/// is flaky.
#[test]
fn a_thread_that_already_holds_the_window_is_refused_not_deadlocked() {
    let _serialised = exclusive();

    let window = SerialWindow::enter().expect("the window opens");
    let refused =
        SerialWindow::enter().expect_err("re-entering on one thread must be refused, not blocked");
    assert!(
        refused
            .message()
            .contains("entering it twice would deadlock"),
        "the refusal must say why re-entry is refused, got: {refused}"
    );
    drop(window);
}

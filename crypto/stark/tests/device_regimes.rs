//! One card, one admission regime per process: both refusals, the exclusion
//! itself, and the re-entry that would otherwise be a hang.
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
//! keeps these four from colliding with each other.

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

#[test]
fn a_byte_budget_regime_refuses_while_a_serial_window_is_open() {
    let _serialised = exclusive();

    // The state it must ACCEPT.
    let regime = ByteBudgetRegime::claim().expect("nothing is open, so the regime is claimable");
    drop(regime);

    // The state it must REFUSE.
    let window = SerialWindow::enter().expect("nothing is claimed, so the window opens");
    let refused = ByteBudgetRegime::claim()
        .expect_err("an open serial window must refuse the byte-budget regime");
    assert!(
        refused.message().contains("serial device window is open"),
        "the refusal must name the window that was already open, got: {refused}"
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

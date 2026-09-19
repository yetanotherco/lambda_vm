//! One card, one admission regime per process.
//!
//! Two different things in this crate decide when a table may occupy the GPU,
//! and they cannot both be right at once:
//!
//! - the monolithic prover's byte budget, `VramGate` in [`crate::prover`],
//!   which admits several tables at a time while their estimated device
//!   working sets fit under a budget;
//! - the prove-and-retire passes' SERIAL window, [`SerialWindow`] here, which
//!   admits exactly one table at a time and estimates nothing.
//!
//! Neither can see the other's accounting, so a process that ran both would
//! spend the same card twice — the two-gates-one-card condition that caused
//! the production base-prove aborts. This module makes that unreachable rather
//! than merely unlikely: a live byte-budget regime refuses a serial window, a
//! live serial window refuses a byte-budget regime, and each is released by
//! its guard's `Drop`.
//!
//! Nothing here touches CUDA or reads a device, so the rule and its tests run
//! on a card-free build exactly as they run on the prover.
//!
//! The byte-budget regime is COUNTED, not capped at one. Two concurrent
//! `multi_prove` calls remain possible, because that is what the continuation
//! driver does today and the block records stand on it; serialising those is a
//! different change with its own evidence, and this module is not the place to
//! make it silently.
//!
//! [`SerialWindow::enter`] blocks on a condition variable while another window
//! is open, so — like `VramGate::acquire` — only OS driver threads may call
//! it. A rayon worker that blocks here starves the pool the admitted table is
//! using.

use std::sync::{Condvar, Mutex, MutexGuard};
use std::thread::ThreadId;

/// Why a device admission regime could not be entered.
///
/// An error rather than a panic: a process that asks for both regimes has a
/// wiring defect, and the prover reports those to its caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRegimeError(&'static str);

impl DeviceRegimeError {
    /// The refusal, in the words the caller sees.
    pub fn message(self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for DeviceRegimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for DeviceRegimeError {}

const SERIAL_REFUSED: &str = "the monolithic byte-budget admission gate is live in this process; \
     the prove-and-retire serial device window cannot open beside it \
     (one card, one admission regime per process)";

const BUDGET_REFUSED: &str = "a prove-and-retire serial device window is open in this process; \
     the monolithic byte-budget admission gate cannot be created beside it \
     (one card, one admission regime per process)";

const REENTRY_REFUSED: &str = "this thread already holds the prove-and-retire serial device \
     window; entering it twice would deadlock";

/// The whole process-wide state: how many byte-budget regimes are live, and
/// which thread holds the serial window.
struct Regimes {
    byte_budget: usize,
    serial_holder: Option<ThreadId>,
}

impl Regimes {
    const fn new() -> Self {
        Self {
            byte_budget: 0,
            serial_holder: None,
        }
    }
}

static REGIMES: Mutex<Regimes> = Mutex::new(Regimes::new());
static RELEASED: Condvar = Condvar::new();

/// The registry lock, recovered rather than unwrapped.
///
/// The critical sections below run no caller code, so a panic while holding
/// this lock would have to come from the standard library itself; recovering
/// keeps a poisoned mutex from turning into a panic in the prover.
fn regimes() -> MutexGuard<'static, Regimes> {
    REGIMES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The prove-and-retire passes' device window: one table on the card at a
/// time, for the lifetime of this guard.
///
/// Only constructible through [`SerialWindow::enter`], and released by its
/// `Drop`, so a table cannot be on the card outside a window and a window
/// cannot outlive the value that represents it.
#[derive(Debug)]
pub struct SerialWindow(());

impl SerialWindow {
    /// Take the serial device window.
    ///
    /// Blocks while another thread holds it — that wait IS the mutual
    /// exclusion. Refuses, rather than waiting, when the monolithic
    /// byte-budget gate is live in this process, and refuses a thread that
    /// already holds the window instead of letting it block on itself.
    pub fn enter() -> Result<Self, DeviceRegimeError> {
        let me = std::thread::current().id();
        let mut state = regimes();
        loop {
            if state.byte_budget > 0 {
                return Err(DeviceRegimeError(SERIAL_REFUSED));
            }
            if state.serial_holder == Some(me) {
                return Err(DeviceRegimeError(REENTRY_REFUSED));
            }
            if state.serial_holder.is_none() {
                state.serial_holder = Some(me);
                return Ok(Self(()));
            }
            state = RELEASED
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}

impl Drop for SerialWindow {
    fn drop(&mut self) {
        let mut state = regimes();
        state.serial_holder = None;
        drop(state);
        RELEASED.notify_all();
    }
}

/// The monolithic prover's byte-budget admission regime, held for as long as
/// its `VramGate` lives.
///
/// `VramGate` carries one of these as a private field, which is what makes the
/// rule unconstructible-by-design: there is no way to build a byte-budget gate
/// without claiming the regime, and no way to drop one without releasing it.
#[derive(Debug)]
pub struct ByteBudgetRegime(());

impl ByteBudgetRegime {
    /// Claim the byte-budget regime for this process.
    ///
    /// Refuses while a [`SerialWindow`] is open. Several byte-budget regimes
    /// may be live at once; see this module's note on why that count is not
    /// capped here.
    pub fn claim() -> Result<Self, DeviceRegimeError> {
        let mut state = regimes();
        if state.serial_holder.is_some() {
            return Err(DeviceRegimeError(BUDGET_REFUSED));
        }
        state.byte_budget += 1;
        Ok(Self(()))
    }
}

impl Drop for ByteBudgetRegime {
    fn drop(&mut self) {
        let mut state = regimes();
        state.byte_budget = state.byte_budget.saturating_sub(1);
        drop(state);
        RELEASED.notify_all();
    }
}

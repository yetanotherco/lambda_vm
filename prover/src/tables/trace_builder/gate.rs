//! The byte gate between the streamed finish and the packer: the bytes of the
//! rest's tables generated and not yet placed, held under a budget.
//!
//! The finish takes a table's bytes before it generates the table
//! ([`ByteGate::acquire`]); the packer gives them back as it places the table
//! ([`ByteGate::release`]), in the order the tables come. The lowest table not
//! yet released is always admitted, whatever its bytes: the packer is waiting
//! for exactly that table, so a budget smaller than one table cannot deadlock.
//! A stop ([`ByteGate::stop`]: a generator's error, or a packer that stopped)
//! reaches every waiter.

use std::collections::BTreeMap;
use std::sync::{Condvar, Mutex};

use crate::Error;

/// See the module docs. Tables are numbered from 0 in the order they are
/// placed.
pub(crate) struct ByteGate {
    budget: usize,
    state: Mutex<GateState>,
    changed: Condvar,
}

#[derive(Default)]
struct GateState {
    /// Each admitted table not yet released: its bytes.
    held: BTreeMap<usize, usize>,
    /// Their sum.
    bytes: usize,
    /// Tables `0..released` are released.
    released: usize,
    /// The most bytes held at once.
    most: usize,
    /// Why the gate stopped, once it has.
    stopped: Option<String>,
}

impl GateState {
    fn admits(&self, budget: usize, table: usize, bytes: usize) -> bool {
        table == self.released || self.bytes.saturating_add(bytes) <= budget
    }

    fn admit(&mut self, table: usize, bytes: usize) {
        self.bytes += bytes;
        if let Some(old) = self.held.insert(table, bytes) {
            self.bytes -= old;
        }
        self.most = self.most.max(self.bytes);
    }

    fn check(&self) -> Result<(), Error> {
        match &self.stopped {
            Some(why) => Err(Error::Prover(format!(
                "the finish's byte gate stopped: {why}"
            ))),
            None => Ok(()),
        }
    }
}

impl ByteGate {
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            budget,
            state: Mutex::new(GateState::default()),
            changed: Condvar::new(),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, GateState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Waits until `table`'s `bytes` fit under the budget, or `table` is the
    /// lowest not yet released, and holds them. An error once the gate stops.
    pub(crate) fn acquire(&self, table: usize, bytes: usize) -> Result<(), Error> {
        let mut state = self.state();
        loop {
            state.check()?;
            if state.admits(self.budget, table, bytes) {
                state.admit(table, bytes);
                return Ok(());
            }
            state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// [`Self::acquire`] without waiting: whether `table` was admitted.
    pub(crate) fn try_acquire(&self, table: usize, bytes: usize) -> Result<bool, Error> {
        let mut state = self.state();
        state.check()?;
        let admitted = state.admits(self.budget, table, bytes);
        if admitted {
            state.admit(table, bytes);
        }
        Ok(admitted)
    }

    /// `table` holds `bytes` now: its size once generated, in place of the
    /// estimate it was admitted on.
    pub(crate) fn resize(&self, table: usize, bytes: usize) {
        let mut state = self.state();
        if state.held.contains_key(&table) {
            state.admit(table, bytes);
        }
        drop(state);
        self.changed.notify_all();
    }

    /// `table` is placed: its bytes go back. Tables are released in order.
    pub(crate) fn release(&self, table: usize) {
        let mut state = self.state();
        if let Some(bytes) = state.held.remove(&table) {
            state.bytes -= bytes;
        }
        state.released = state.released.max(table + 1);
        drop(state);
        self.changed.notify_all();
    }

    /// Stops the gate: every waiter, now and later, gets an error naming `why`.
    pub(crate) fn stop(&self, why: &str) {
        let mut state = self.state();
        if state.stopped.is_none() {
            state.stopped = Some(why.to_string());
        }
        drop(state);
        self.changed.notify_all();
    }

    /// The most bytes held at once.
    pub(crate) fn most(&self) -> usize {
        self.state().most
    }

    pub(crate) fn budget(&self) -> usize {
        self.budget
    }
}

/// Stops the gate when dropped, however its holder stops (a packer that
/// returned, or unwound).
pub(crate) struct StopGate<'g>(pub(crate) &'g ByteGate, pub(crate) &'static str);

impl Drop for StopGate<'_> {
    fn drop(&mut self) {
        self.0.stop(self.1);
    }
}

#[cfg(test)]
mod tests {
    use super::ByteGate;
    use std::sync::Arc;
    use std::time::Duration;

    /// Runs `f` on a thread of its own; `None` when it has not returned
    /// within `secs` (the thread is left behind).
    fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(Duration::from_secs(secs)).ok()
    }

    /// A table over the budget is admitted when it is the lowest not yet
    /// released, and the next waits until it is placed (within a timeout: a
    /// gate that never admits fails the test instead of hanging the suite).
    #[test]
    fn the_lowest_pending_table_is_admitted_over_the_budget() {
        let done = within(20, || {
            let gate = Arc::new(ByteGate::new(10));
            gate.acquire(0, 100).expect("the lowest pending table");
            assert!(!gate.try_acquire(1, 1).expect("not stopped"));
            let waiter = {
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || gate.acquire(1, 100))
            };
            std::thread::sleep(Duration::from_millis(50));
            assert!(!waiter.is_finished(), "table 1 waits for table 0");
            gate.release(0);
            waiter
                .join()
                .expect("joined")
                .expect("admitted once table 0 is placed");
            gate.most()
        });
        assert_eq!(done, Some(100), "the lowest pending table must be admitted");
    }

    /// Tables under the budget go in together; the held bytes follow the
    /// resized tables.
    #[test]
    fn tables_under_the_budget_are_held_together_and_resized() {
        let gate = ByteGate::new(100);
        gate.acquire(0, 40).expect("0");
        assert!(gate.try_acquire(1, 40).expect("1"));
        assert!(!gate.try_acquire(2, 40).expect("2 waits"));
        gate.resize(1, 10);
        assert!(gate.try_acquire(2, 40).expect("2 fits once 1 is resized"));
        assert_eq!(gate.most(), 90);
    }

    /// A stop reaches a waiter, and every later call.
    #[test]
    fn a_stop_reaches_every_waiter() {
        let gate = Arc::new(ByteGate::new(10));
        gate.acquire(0, 10).expect("0");
        let waiters: Vec<_> = (1..4)
            .map(|k| {
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || gate.acquire(k, 10))
            })
            .collect();
        std::thread::sleep(Duration::from_millis(50));
        gate.stop("a generator failed");
        for waiter in waiters {
            let err = waiter.join().expect("joined").expect_err("stopped");
            assert!(format!("{err:?}").contains("a generator failed"), "{err:?}");
        }
        assert!(gate.try_acquire(9, 0).is_err());
    }

    /// One table's budget, every table larger than it: they all pass, one at
    /// a time, as each is placed.
    #[test]
    fn a_one_byte_budget_completes() {
        let done = within(20, || {
            let gate = Arc::new(ByteGate::new(1));
            let placer = {
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || {
                    for k in 0..50 {
                        while !gate.state().held.contains_key(&k) {
                            std::thread::yield_now();
                        }
                        gate.release(k);
                    }
                })
            };
            for k in 0..50 {
                gate.acquire(k, 1000).expect("admitted");
            }
            placer.join().expect("placer");
            gate.most()
        });
        assert_eq!(done, Some(1000), "a one-byte budget must not deadlock");
    }
}

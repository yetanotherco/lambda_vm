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
    /// The bytes the sliding finish may build at once ([`Self::with_build`]).
    build: usize,
    /// The most bytes it built at once.
    building_most: std::sync::atomic::AtomicUsize,
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
    /// The seconds acquirers waited for room, summed over them, and how many
    /// acquisitions waited.
    waited: f64,
    waits: usize,
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
            build: budget,
            building_most: std::sync::atomic::AtomicUsize::new(0),
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
        let mut since: Option<std::time::Instant> = None;
        loop {
            state.check()?;
            if state.admits(self.budget, table, bytes) {
                state.admit(table, bytes);
                if let Some(since) = since {
                    state.waited += since.elapsed().as_secs_f64();
                    state.waits += 1;
                }
                return Ok(());
            }
            since.get_or_insert_with(std::time::Instant::now);
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

    /// The seconds acquirers waited for room, summed over them, and how many
    /// acquisitions waited (a measurement: the finish's time not spent
    /// building).
    pub(crate) fn waited(&self) -> (f64, usize) {
        let state = self.state();
        (state.waited, state.waits)
    }

    pub(crate) fn budget(&self) -> usize {
        self.budget
    }

    /// The gate with a cap on the estimated bytes the sliding finish builds at
    /// once, apart from the bytes it holds (built and not yet placed, or
    /// building): `None` leaves the budget alone bounding them, as before,
    /// which the budget already implies.
    pub(crate) fn with_build(mut self, build: Option<usize>) -> Self {
        if let Some(build) = build {
            self.build = build;
        }
        self
    }

    /// The cap on the bytes building at once ([`Self::with_build`]).
    pub(crate) fn build(&self) -> usize {
        self.build
    }

    /// Notes `bytes` building at once.
    pub(crate) fn note_building(&self, bytes: usize) {
        self.building_most
            .fetch_max(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// The most estimated bytes the sliding finish built at once.
    pub(crate) fn building_most(&self) -> usize {
        self.building_most
            .load(std::sync::atomic::Ordering::Relaxed)
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

    /// The build cap is the budget unless one is given, and the most bytes
    /// built at once only grows.
    #[test]
    fn the_build_cap_is_the_budget_unless_given() {
        assert_eq!(ByteGate::new(10).build(), 10);
        assert_eq!(ByteGate::new(10).with_build(None).build(), 10);
        let gate = ByteGate::new(10).with_build(Some(4));
        assert_eq!((gate.budget(), gate.build()), (10, 4));
        gate.note_building(3);
        gate.note_building(2);
        assert_eq!(gate.building_most(), 3);
    }

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
    /// gate that never admits fails the test instead of hanging the suite);
    /// the gate counts that one wait and its time, not the acquisition that
    /// did not wait.
    #[test]
    fn the_lowest_pending_table_is_admitted_over_the_budget() {
        let done = within(20, || {
            let gate = Arc::new(ByteGate::new(10));
            gate.acquire(0, 100).expect("the lowest pending table");
            assert_eq!(gate.waited(), (0.0, 0), "no wait yet");
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
            (gate.most(), gate.waited())
        });
        let (most, (waited, waits)) = done.expect("the lowest pending table must be admitted");
        assert_eq!(most, 100);
        assert_eq!(waits, 1, "one acquisition waited");
        assert!(waited >= 0.04, "it waited {waited} s for table 0's release");
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

//! Packed main traces dropped after their Round-1 commit and built again for
//! their fused task (D-REGEN §2, R2).
//!
//! A block can rebuild some of its instances' main traces from the run itself
//! (the streamed chunks: a replay of the windows they span). Such an instance
//! need not stay on the host, nor go to a spill file, between its Round-1
//! commit and its fused task: its packed trace ([`NarrowMain`]) is dropped
//! ([`crate::trace::TraceTable::drop_main_for_regen`]) and a caller's
//! regenerator deposits it again in phase B.
//!
//! - A [`RegenWindow`] holds the dropped instances' [`RegenSlot`]s and bounds
//!   the bytes deposited and not yet taken (`ahead`): a deposit past it waits,
//!   so the regenerator never runs more than a window ahead of the fused tasks.
//! - A deposit is checked against the shape and the digest taken when the
//!   trace was dropped ([`NarrowMain::digest`]); another trace fails its slot,
//!   and the prove refuses that table before any of its device work. The
//!   digest is not cryptographic: the threat is a bug, not an adversary, and
//!   the kept-top check and the verifier stay behind it.
//! - Nothing waits forever. The window is created with its first
//!   [`RegenProducer`]; once every producer is dropped (the regenerator ended,
//!   panicked, or never started), every slot not deposited fails. Closing the
//!   window ([`RegenWindow::close`], the prove's end or a panicked task) wakes
//!   every waiter and every depositor.
//!
//! Nothing is dropped unless a caller opens a window and drops traces into it.

use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

use crate::narrow::NarrowMain;

const GIB: f64 = (1u64 << 30) as f64;

/// A poisoned lock here is a panic already being reported elsewhere; the
/// state behind it stays consistent (every transition is one assignment).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Why a dropped trace did not come back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegenError {
    /// The deposited trace is not the one dropped: another shape or digest.
    Mismatch,
    /// The regenerator could not build it, or stopped before it did.
    Failed(String),
    /// The window was closed before it came back (the prove ended or stopped).
    Closed(String),
}

impl std::fmt::Display for RegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegenError::Mismatch => write!(f, "the regenerated trace is not the one dropped"),
            RegenError::Failed(why) => write!(f, "not regenerated: {why}"),
            RegenError::Closed(why) => write!(f, "the regeneration window closed: {why}"),
        }
    }
}

/// A dropped packed trace's shape and digest: what its deposit must be.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Want {
    rows: usize,
    widths: Vec<u8>,
    len: usize,
    digest: [u64; 2],
}

impl Want {
    fn of(narrow: &NarrowMain) -> Self {
        Self {
            rows: narrow.rows(),
            widths: narrow.widths().to_vec(),
            len: narrow.data().len(),
            digest: narrow.digest(),
        }
    }
}

enum SlotState {
    Waiting,
    Ready(NarrowMain),
    Failed(RegenError),
    Taken,
}

#[derive(Default)]
struct Shared {
    slots: Vec<SlotState>,
    /// Bytes deposited and not taken, and the most at once.
    parked: u64,
    high_water: u64,
    /// Producers alive; once the last one is dropped, every waiting slot fails.
    producers: usize,
    /// Why the window was closed, once it is.
    closed: Option<String>,
    deposits: u64,
    mismatches: u64,
    taken: u64,
    /// Seconds depositors waited for room, and takers for their trace.
    deposit_wait_ns: u64,
    take_wait_ns: u64,
}

impl Shared {
    /// The state a taker of slot `id` sees now: its trace, its failure, or
    /// nothing yet (`None`).
    fn settled(&self, id: usize) -> Option<Result<(), RegenError>> {
        match &self.slots[id] {
            SlotState::Ready(_) => Some(Ok(())),
            SlotState::Failed(e) => Some(Err(e.clone())),
            SlotState::Taken => Some(Err(RegenError::Failed(
                "its trace was already taken".to_string(),
            ))),
            SlotState::Waiting => {
                if let Some(why) = &self.closed {
                    Some(Err(RegenError::Closed(why.clone())))
                } else if self.producers == 0 {
                    Some(Err(RegenError::Failed(
                        "the regenerator stopped before it".to_string(),
                    )))
                } else {
                    None
                }
            }
        }
    }
}

/// The dropped traces of one prove and the bytes deposited ahead of their
/// fused tasks (see the module docs).
pub struct RegenWindow {
    shared: Mutex<Shared>,
    changed: Condvar,
    ahead: u64,
}

impl RegenWindow {
    /// A window that holds at most `ahead` bytes deposited and not taken (one
    /// trace larger than that is deposited alone), and its first producer:
    /// the caller hands it to the regenerator, so a regenerator that never
    /// starts, ends early or panics fails every slot it did not deposit.
    pub fn new(ahead: u64) -> (Arc<Self>, RegenProducer) {
        let window = Arc::new(Self {
            shared: Mutex::new(Shared {
                producers: 1,
                ..Shared::default()
            }),
            changed: Condvar::new(),
            ahead,
        });
        let producer = RegenProducer {
            window: Arc::clone(&window),
        };
        (window, producer)
    }

    /// Another producer, alive as long as it is held.
    pub fn producer(self: &Arc<Self>) -> RegenProducer {
        lock(&self.shared).producers += 1;
        RegenProducer {
            window: Arc::clone(self),
        }
    }

    /// A slot for `narrow`, about to be dropped: `rank` orders it among the
    /// window's slots (the regenerator deposits in rank order, and phase B
    /// takes them in it).
    pub fn slot(self: &Arc<Self>, narrow: &NarrowMain, rank: u64) -> RegenSlot {
        let mut shared = lock(&self.shared);
        let id = shared.slots.len();
        shared.slots.push(SlotState::Waiting);
        RegenSlot {
            window: Arc::clone(self),
            id,
            rank,
            want: Want::of(narrow),
        }
    }

    /// Close: every slot not deposited fails with `why`, every waiter and
    /// every depositor returns. The first reason stays.
    pub fn close(&self, why: &str) {
        let mut shared = lock(&self.shared);
        if shared.closed.is_none() {
            shared.closed = Some(why.to_string());
        }
        drop(shared);
        self.changed.notify_all();
    }

    /// Whether it is closed.
    pub fn is_closed(&self) -> bool {
        lock(&self.shared).closed.is_some()
    }

    /// One line for the prove's log.
    pub fn report(&self) -> String {
        let shared = lock(&self.shared);
        let waiting = shared
            .slots
            .iter()
            .filter(|s| matches!(s, SlotState::Waiting))
            .count();
        format!(
            "{} slots · {} deposited · {} taken · {} mismatches · {waiting} never deposited · window \
             {:.2} GiB, high-water {:.2} GiB · depositors waited {:.2} s · takers waited {:.2} s",
            shared.slots.len(),
            shared.deposits,
            shared.taken,
            shared.mismatches,
            self.ahead as f64 / GIB,
            shared.high_water as f64 / GIB,
            shared.deposit_wait_ns as f64 / 1e9,
            shared.take_wait_ns as f64 / 1e9,
        )
    }
}

/// A regenerator's hold on a window: while one is alive, a slot not
/// deposited waits for it; when the last is dropped, those slots fail.
pub struct RegenProducer {
    window: Arc<RegenWindow>,
}

impl Drop for RegenProducer {
    fn drop(&mut self) {
        let mut shared = lock(&self.window.shared);
        shared.producers = shared.producers.saturating_sub(1);
        drop(shared);
        self.window.changed.notify_all();
    }
}

/// A dropped packed trace's place in its window: the trace holds one, the
/// regenerator a clone. Clones are the same slot.
#[derive(Clone)]
pub struct RegenSlot {
    window: Arc<RegenWindow>,
    id: usize,
    rank: u64,
    want: Want,
}

impl std::fmt::Debug for RegenSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegenSlot")
            .field("id", &self.id)
            .field("rank", &self.rank)
            .field("rows", &self.want.rows)
            .field("cols", &self.want.widths.len())
            .field("bytes", &self.want.len)
            .finish_non_exhaustive()
    }
}

/// Two slots are equal when they are one slot, or when the traces they stand
/// for have the same shape, length and digest.
impl PartialEq for RegenSlot {
    fn eq(&self, other: &Self) -> bool {
        (Arc::ptr_eq(&self.window, &other.window) && self.id == other.id) || self.want == other.want
    }
}

impl Eq for RegenSlot {}

impl RegenSlot {
    /// Its place in the order the regenerator deposits and phase B takes.
    pub fn rank(&self) -> u64 {
        self.rank
    }

    /// Rows of the dropped trace.
    pub fn rows(&self) -> usize {
        self.want.rows
    }

    /// Packed bytes of the dropped trace.
    pub fn len(&self) -> usize {
        self.want.len
    }

    /// Whether the dropped trace had no bytes.
    pub fn is_empty(&self) -> bool {
        self.want.len == 0
    }

    /// The digest taken when the trace was dropped.
    pub fn digest(&self) -> [u64; 2] {
        self.want.digest
    }

    /// Its window.
    pub fn window(&self) -> &Arc<RegenWindow> {
        &self.window
    }

    /// Put the regenerated trace in the slot: checked against the dropped
    /// one's shape and digest, then held until its fused task takes it,
    /// waiting first while the window holds `ahead` bytes. A trace that is not
    /// the dropped one fails the slot (its table is refused) and returns
    /// [`RegenError::Mismatch`]; a closed window returns
    /// [`RegenError::Closed`]; a slot already settled refuses a second trace.
    pub fn deposit(&self, narrow: NarrowMain) -> Result<(), RegenError> {
        let matches = Want::of(&narrow) == self.want;
        let window = &self.window;
        let t = Instant::now();
        let mut shared = lock(&window.shared);
        if !matches!(shared.slots[self.id], SlotState::Waiting) {
            return Err(RegenError::Failed("deposited twice".to_string()));
        }
        if !matches {
            shared.slots[self.id] = SlotState::Failed(RegenError::Mismatch);
            shared.mismatches += 1;
            drop(shared);
            window.changed.notify_all();
            return Err(RegenError::Mismatch);
        }
        let len = self.want.len as u64;
        while shared.closed.is_none()
            && shared.parked > 0
            && shared.parked + len > window.ahead
            && matches!(shared.slots[self.id], SlotState::Waiting)
        {
            shared = window
                .changed
                .wait(shared)
                .unwrap_or_else(|e| e.into_inner());
        }
        shared.deposit_wait_ns += t.elapsed().as_nanos() as u64;
        if let Some(why) = &shared.closed {
            return Err(RegenError::Closed(why.clone()));
        }
        if !matches!(shared.slots[self.id], SlotState::Waiting) {
            return Err(RegenError::Failed("deposited twice".to_string()));
        }
        shared.slots[self.id] = SlotState::Ready(narrow);
        shared.parked += len;
        shared.high_water = shared.high_water.max(shared.parked);
        shared.deposits += 1;
        drop(shared);
        window.changed.notify_all();
        Ok(())
    }

    /// The regenerator gives this slot up: its table is refused with `why`.
    pub fn fail(&self, why: &str) {
        let mut shared = lock(&self.window.shared);
        if matches!(shared.slots[self.id], SlotState::Waiting) {
            shared.slots[self.id] = SlotState::Failed(RegenError::Failed(why.to_string()));
        }
        drop(shared);
        self.window.changed.notify_all();
    }

    /// Whether [`Self::wait`] would return at once: the trace is in, or it
    /// will never be.
    pub fn is_ready(&self) -> bool {
        lock(&self.window.shared).settled(self.id).is_some()
    }

    /// Block until the trace is in, or it will never be.
    pub fn wait(&self) {
        let t = Instant::now();
        let mut shared = lock(&self.window.shared);
        while shared.settled(self.id).is_none() {
            shared = self
                .window
                .changed
                .wait(shared)
                .unwrap_or_else(|e| e.into_inner());
        }
        shared.take_wait_ns += t.elapsed().as_nanos() as u64;
    }

    /// A copy of the regenerated trace (blocking until it is in or will never
    /// be), for a reader beside its fused task: the bytes stay in the slot.
    pub fn load(&self) -> Result<NarrowMain, RegenError> {
        self.wait();
        let shared = lock(&self.window.shared);
        match &shared.slots[self.id] {
            SlotState::Ready(narrow) => Ok(narrow.clone()),
            _ => match shared.settled(self.id) {
                Some(Err(e)) => Err(e),
                _ => Err(RegenError::Failed("not deposited".to_string())),
            },
        }
    }

    /// The regenerated trace, for its one reader (blocking until it is in or
    /// will never be): its bytes leave the window, which makes room for the
    /// next deposit.
    pub(crate) fn take(&self) -> Result<NarrowMain, RegenError> {
        self.wait();
        let mut shared = lock(&self.window.shared);
        let settled = shared.settled(self.id);
        match settled {
            Some(Ok(())) => {
                let state = std::mem::replace(&mut shared.slots[self.id], SlotState::Taken);
                let SlotState::Ready(narrow) = state else {
                    return Err(RegenError::Failed(
                        "its trace was already taken".to_string(),
                    ));
                };
                shared.parked = shared.parked.saturating_sub(self.want.len as u64);
                shared.taken += 1;
                drop(shared);
                self.window.changed.notify_all();
                Ok(narrow)
            }
            Some(Err(e)) => {
                // A slot that will never be deposited settles as failed, so
                // a second look says the same.
                if matches!(shared.slots[self.id], SlotState::Waiting) {
                    shared.slots[self.id] = SlotState::Failed(e.clone());
                }
                Err(e)
            }
            None => Err(RegenError::Failed("not settled after its wait".to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::TraceTable;
    use math::field::element::FieldElement;
    use math::field::{
        extensions_goldilocks::Degree3GoldilocksExtensionField as E,
        goldilocks::GoldilocksField as F,
    };
    use std::sync::mpsc;
    use std::time::Duration;

    /// A packed trace of `rows` × 3 whose words depend on `seed`.
    fn narrow(rows: usize, seed: u64) -> NarrowMain {
        let words: Vec<u64> = (0..rows as u64 * 3)
            .map(|i| (i ^ seed).wrapping_mul(0x9e37_79b9) % 70_000)
            .collect();
        NarrowMain::pack(&words, 3)
    }

    /// The thread's result within `secs`, or a failure naming the hang.
    fn within<T: Send + 'static>(
        secs: u64,
        what: &str,
        f: impl FnOnce() -> T + Send + 'static,
    ) -> T {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(Duration::from_secs(secs))
            .unwrap_or_else(|_| panic!("{what}: hung"))
    }

    /// The trace deposited is the trace taken, bit for bit, and its bytes
    /// leave the window.
    #[test]
    fn a_deposited_trace_is_taken_bit_for_bit() {
        let (window, _producer) = RegenWindow::new(1 << 30);
        let a = narrow(100, 1);
        let slot = window.slot(&a, 0);
        assert!(!slot.is_ready());
        slot.deposit(a.clone()).unwrap();
        assert!(slot.is_ready());
        assert_eq!(
            slot.load().unwrap(),
            a,
            "a copy leaves the bytes in the slot"
        );
        assert_eq!(slot.take().unwrap(), a);
        assert!(slot.take().is_err(), "one reader");
        let report = window.report();
        assert!(
            report.contains("1 deposited · 1 taken · 0 mismatches"),
            "{report}"
        );
    }

    /// ★ A deposit that is not the dropped trace — another word, another row
    /// count — fails its slot: the deposit and the taker both see
    /// `Mismatch`, and a second deposit is refused.
    #[test]
    fn another_trace_fails_its_slot() {
        let (window, _producer) = RegenWindow::new(1 << 30);
        let a = narrow(100, 1);
        for other in [narrow(100, 2), narrow(99, 1)] {
            let slot = window.slot(&a, 0);
            assert_eq!(slot.deposit(other), Err(RegenError::Mismatch));
            assert!(slot.is_ready(), "a failed slot is settled");
            assert_eq!(slot.take(), Err(RegenError::Mismatch));
            assert!(matches!(
                slot.deposit(a.clone()),
                Err(RegenError::Failed(_))
            ));
        }
        let mut bent = a.clone();
        bent.flip_first_bit();
        let slot = window.slot(&a, 1);
        assert_eq!(slot.deposit(bent), Err(RegenError::Mismatch));
    }

    /// ★ A regenerator that stops — its producer dropped, as when its thread
    /// ends or panics — wakes every taker waiting on a slot it never
    /// deposited, which fails; what it deposited is still taken.
    #[test]
    fn a_dead_regenerator_fails_what_it_did_not_deposit() {
        let (window, producer) = RegenWindow::new(1 << 30);
        let a = narrow(64, 1);
        let b = narrow(64, 2);
        let slot_a = window.slot(&a, 0);
        let slot_b = window.slot(&b, 1);
        let waiter = {
            let slot_b = slot_b.clone();
            std::thread::spawn(move || slot_b.take())
        };
        std::thread::sleep(Duration::from_millis(50));
        let regenerator = std::thread::spawn(move || {
            let _producer = producer;
            slot_a.deposit(a).unwrap();
            panic!("the regenerator dies before its second trace");
        });
        assert!(regenerator.join().is_err());
        let got = within(10, "the taker of a slot never deposited", move || {
            waiter.join().unwrap()
        });
        assert!(matches!(got, Err(RegenError::Failed(_))), "{got:?}");
        assert!(window.report().contains("1 deposited"));
    }

    /// ★ A window holds at most `ahead` bytes deposited and not taken: the
    /// next deposit waits for a take; one trace larger than the window goes
    /// alone.
    #[test]
    fn a_deposit_waits_for_room_in_the_window() {
        let a = narrow(64, 1);
        let b = narrow(64, 2);
        let (window, _producer) = RegenWindow::new(a.data().len() as u64);
        let slot_a = window.slot(&a, 0);
        let slot_b = window.slot(&b, 1);
        slot_a.deposit(a.clone()).unwrap();
        let (tx, rx) = mpsc::channel();
        let depositor = {
            let slot_b = slot_b.clone();
            std::thread::spawn(move || {
                let out = slot_b.deposit(b);
                tx.send(()).unwrap();
                out
            })
        };
        assert!(
            rx.recv_timeout(Duration::from_millis(200)).is_err(),
            "deposited past the window"
        );
        assert_eq!(slot_a.take().unwrap(), a);
        rx.recv_timeout(Duration::from_secs(5))
            .expect("deposited once there is room");
        depositor.join().unwrap().unwrap();
        assert!(slot_b.take().is_ok());
        // A trace larger than the window, into an empty window, goes alone.
        let (small, _p) = RegenWindow::new(1);
        let big = narrow(1000, 3);
        let slot = small.slot(&big, 0);
        slot.deposit(big).unwrap();
    }

    /// ★ Closing the window wakes a depositor waiting for room and a taker
    /// waiting for its trace: both return `Closed`, and nothing hangs.
    #[test]
    fn closing_the_window_wakes_every_waiter() {
        let a = narrow(64, 1);
        let b = narrow(64, 2);
        let c = narrow(64, 3);
        let (window, _producer) = RegenWindow::new(a.data().len() as u64);
        let slot_a = window.slot(&a, 0);
        let slot_b = window.slot(&b, 1);
        let slot_c = window.slot(&c, 2);
        slot_a.deposit(a).unwrap();
        let depositor = std::thread::spawn(move || slot_b.deposit(b));
        let taker = std::thread::spawn(move || slot_c.take());
        std::thread::sleep(Duration::from_millis(100));
        window.close("the prove ended");
        let (dep, take) = within(10, "a closed window's waiters", move || {
            (depositor.join().unwrap(), taker.join().unwrap())
        });
        assert!(matches!(dep, Err(RegenError::Closed(_))), "{dep:?}");
        assert!(matches!(take, Err(RegenError::Closed(_))), "{take:?}");
        assert!(window.is_closed());
    }

    /// A slot given up fails with the regenerator's reason.
    #[test]
    fn a_slot_given_up_fails() {
        let (window, _producer) = RegenWindow::new(1 << 30);
        let a = narrow(10, 1);
        let slot = window.slot(&a, 0);
        slot.fail("the walk refused window 3");
        assert_eq!(
            slot.take(),
            Err(RegenError::Failed("the walk refused window 3".to_string()))
        );
    }

    fn packed_trace(rows: usize) -> (TraceTable<F, E>, Vec<Vec<FieldElement<F>>>) {
        let words: Vec<FieldElement<F>> = (0..rows as u64 * 4)
            .map(|i| FieldElement::from((i * 7919) % 300))
            .collect();
        let mut trace = TraceTable::<F, E>::new_main(words, 4, 1);
        let columns = trace.columns_main();
        assert!(trace.pack_main_narrow());
        (trace, columns)
    }

    /// ★ A packed trace drops: it keeps its shape, no other state can take
    /// it (spill, a device-packed copy, a second drop), a reader beside its
    /// fused task reads the regenerated words without taking them, and its
    /// fused task's take gives back the dropped words bit for bit.
    #[test]
    fn a_dropped_trace_comes_back_bit_for_bit() {
        let (mut trace, columns) = packed_trace(256);
        let packed = trace.narrow_main().unwrap().clone();
        let (window, _producer) = RegenWindow::new(1 << 30);
        let slot = trace
            .drop_main_for_regen(&window, 7)
            .expect("a packed trace drops");
        assert_eq!(slot.rank(), 7);
        assert_eq!(slot.digest(), packed.digest());
        assert!(trace.is_main_regenerable() && trace.narrow_main().is_none());
        assert_eq!((trace.num_rows(), trace.num_main_columns), (256, 4));
        assert!(
            trace.drop_main_for_regen(&window, 8).is_none(),
            "dropped twice"
        );
        assert!(
            !trace.install_main_narrow(packed.clone()),
            "a device copy over a dropped trace"
        );
        slot.deposit(packed.clone()).unwrap();
        assert_eq!(
            trace.columns_main(),
            columns,
            "a reader beside the fused task"
        );
        let copy = trace.widened_copy();
        assert_eq!(copy.columns_main(), columns);
        assert!(
            trace.is_main_regenerable(),
            "the copy did not take the slot"
        );
        trace.unregen_main().unwrap();
        assert!(!trace.is_main_regenerable());
        assert_eq!(trace.narrow_main().unwrap(), &packed);
        assert_eq!(trace.columns_main(), columns);
    }

    /// Only an unshared packed trace drops: a 64-bit one, or one whose packed
    /// copy another clone shares, stays as it was.
    #[test]
    fn only_an_unshared_packed_trace_drops() {
        let (window, _producer) = RegenWindow::new(1 << 30);
        let words: Vec<FieldElement<F>> = (0..64u64).map(FieldElement::from).collect();
        let mut wide = TraceTable::<F, E>::new_main(words, 2, 1);
        assert!(wide.drop_main_for_regen(&window, 0).is_none());
        let (mut trace, _) = packed_trace(32);
        let shared = trace.clone();
        assert!(trace.drop_main_for_regen(&window, 0).is_none());
        assert!(trace.narrow_main().is_some());
        drop(shared);
        assert!(trace.drop_main_for_regen(&window, 0).is_some());
    }

    /// A dropped trace whose regenerator failed it refuses its take, and the
    /// trace is left with no words.
    #[test]
    fn a_failed_dropped_trace_refuses_its_take() {
        let (mut trace, _) = packed_trace(32);
        let (window, producer) = RegenWindow::new(1 << 30);
        let slot = trace.drop_main_for_regen(&window, 0).unwrap();
        drop(producer);
        assert!(slot.is_ready());
        assert!(matches!(trace.unregen_main(), Err(RegenError::Failed(_))));
        assert!(trace.narrow_main().is_none() && !trace.is_main_regenerable());
    }
}

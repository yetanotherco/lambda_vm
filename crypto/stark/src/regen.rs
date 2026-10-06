//! A block's packed tables dropped after their group's commit and built again
//! for their group in phase B (D-REGEN §2, D-WHIR-NODISK §2.1).
//!
//! A block can rebuild some of its committed tables from the run itself (the
//! streamed chunks: a replay of the windows they span). Such a table need not
//! stay on the host, nor go to a spill file, between its group's commit and its
//! group's turn in phase B: its packed columns
//! ([`multilinear::narrow::NarrowColumns`]) are dropped and a caller's
//! regenerator deposits them again in phase B.
//!
//! - A [`RegenWindow`] holds the dropped tables' [`RegenSlot`]s and paces the
//!   regenerator by rank: a deposit of rank `r` goes once the slots from the
//!   frontier (the lowest rank not yet taken or failed) up to `r` hold at most
//!   `ahead` bytes, deposited or not; the frontier itself always goes. So the
//!   regenerator runs at most a window ahead of the groups that take them, in
//!   rank order, and its generators may finish out of order inside it: the
//!   window is reserved from the frontier, so later ranks can never fill it
//!   while the rank a taker waits on cannot deposit (R-REGEN R1).
//! - A deposit is checked against the shape and the digest taken when the
//!   table was dropped ([`digest_of`], the spill store's own digest of the same
//!   parts); another table fails its slot, and the prove refuses that table
//!   before any of its device work. The digest is not cryptographic: the
//!   threat is a bug, not an adversary, and the kept-top check and the verifier
//!   stay behind it.
//! - Nothing waits forever. The window is created with its first
//!   [`RegenProducer`]; once every producer is dropped (the regenerator ended,
//!   panicked, or never started), every slot not deposited fails. Closing the
//!   window ([`RegenWindow::close`], the prove's end) wakes every waiter and
//!   every depositor. A take never blocks: its taker waited for the slot first.
//!
//! #1013's `stark::regen` (814a4b656), over the block's committed form instead
//! of a trace's `NarrowMain`: the window, its pacing and its failure semantics
//! are the same. Nothing is dropped unless a caller opens a window and drops
//! tables into it.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

use multilinear::narrow::NarrowColumns;

const GIB: f64 = (1u64 << 30) as f64;

/// The 128-bit digest of packed columns: the spill store's digest of the same
/// rows, widths and bytes ([`crate::narrow::NarrowMain::digest`]). A fast,
/// non-cryptographic check that columns built again are these.
pub fn digest_of(packed: &NarrowColumns) -> [u64; 2] {
    crate::narrow::digest_parts(packed.rows(), packed.widths(), packed.data())
}

/// A poisoned lock here is a panic already being reported elsewhere; the
/// state behind it stays consistent (every transition is one assignment).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Why a dropped table did not come back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegenError {
    /// The deposited columns are not the ones dropped: another shape or digest.
    Mismatch,
    /// The regenerator could not build them, or stopped before it did.
    Failed(String),
    /// The window was closed before they came back (the prove ended or stopped).
    Closed(String),
}

impl std::fmt::Display for RegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegenError::Mismatch => write!(f, "the regenerated columns are not the ones dropped"),
            RegenError::Failed(why) => write!(f, "not regenerated: {why}"),
            RegenError::Closed(why) => write!(f, "the regeneration window closed: {why}"),
        }
    }
}

/// Dropped packed columns' shape and digest: what their deposit must be.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Want {
    rows: usize,
    widths: Vec<u8>,
    len: usize,
    digest: [u64; 2],
}

impl Want {
    fn of(packed: &NarrowColumns) -> Self {
        Self {
            rows: packed.rows(),
            widths: packed.widths().to_vec(),
            len: packed.data().len(),
            digest: digest_of(packed),
        }
    }

    /// The shape alone: rows, column widths and bytes. Always checked, so a
    /// deposit can never be widened into a buffer of another size.
    fn same_shape(&self, packed: &NarrowColumns) -> bool {
        self.rows == packed.rows()
            && self.widths == packed.widths()
            && self.len == packed.data().len()
    }
}

enum SlotState {
    Waiting,
    Ready(NarrowColumns),
    Failed(RegenError),
    Taken,
}

struct Slot {
    rank: u64,
    state: SlotState,
    /// Set once the state leaves `Waiting`: the lock-free half of
    /// [`RegenSlot::is_ready`].
    settled: Arc<AtomicBool>,
}

#[derive(Default)]
struct Shared {
    slots: Vec<Slot>,
    /// The slots not yet taken or failed, by (rank, id), with their bytes:
    /// the first is the frontier, and the window is reserved from it.
    live: BTreeMap<(u64, usize), u64>,
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
    /// Seconds depositors waited for room, and takers for their columns.
    deposit_wait_ns: u64,
    take_wait_ns: u64,
}

impl Shared {
    /// What a reader of slot `id` sees now: its columns, its failure, or
    /// nothing yet (`None`).
    fn settled(&self, id: usize) -> Option<Result<(), RegenError>> {
        match &self.slots[id].state {
            SlotState::Ready(_) => Some(Ok(())),
            SlotState::Failed(e) => Some(Err(e.clone())),
            SlotState::Taken => Some(Err(RegenError::Failed(
                "its columns were already taken".to_string(),
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

    /// Slot `id` leaves `Waiting` for `state` (a failed slot is consumed:
    /// it leaves the window's reservation).
    fn settle(&mut self, id: usize, state: SlotState) {
        let slot = &mut self.slots[id];
        if matches!(state, SlotState::Failed(_)) {
            self.live.remove(&(slot.rank, id));
        }
        slot.state = state;
        slot.settled.store(true, Ordering::Release);
    }

    /// Whether slot `key` may deposit now: it is the frontier, or the slots
    /// from the frontier up to it reserve at most `ahead` bytes.
    fn admits(&self, key: (u64, usize), ahead: u64) -> bool {
        if self.live.keys().next() == Some(&key) {
            return true;
        }
        let mut reserved = 0u64;
        for (&k, &len) in &self.live {
            if k > key {
                break;
            }
            reserved = reserved.saturating_add(len);
            if reserved > ahead {
                return false;
            }
        }
        true
    }
}

/// The dropped tables of one prove and the bytes deposited ahead of the
/// groups that take them (see the module docs).
pub struct RegenWindow {
    shared: Mutex<Shared>,
    changed: Condvar,
    ahead: u64,
    /// The lock-free halves of a waiting slot's readiness: the window closed,
    /// and every producer gone.
    closed: AtomicBool,
    gone: AtomicBool,
    /// Test only: deposits checked on their shape alone, so the checks behind
    /// the digest can be tested.
    #[cfg(any(test, feature = "test-utils"))]
    verify: AtomicBool,
}

impl RegenWindow {
    /// A window that reserves at most `ahead` bytes from its frontier (the
    /// frontier, however large, always goes), and its first producer: the
    /// caller hands it to the regenerator, so a regenerator that never
    /// starts, ends early or panics fails every slot it did not deposit.
    pub fn new(ahead: u64) -> (Arc<Self>, RegenProducer) {
        let window = Arc::new(Self {
            shared: Mutex::new(Shared {
                producers: 1,
                ..Shared::default()
            }),
            changed: Condvar::new(),
            ahead,
            closed: AtomicBool::new(false),
            gone: AtomicBool::new(false),
            #[cfg(any(test, feature = "test-utils"))]
            verify: AtomicBool::new(true),
        });
        let producer = RegenProducer {
            window: Arc::clone(&window),
        };
        (window, producer)
    }

    /// Another producer, alive as long as it is held. None once every
    /// producer is gone (the slots not deposited have failed already).
    pub fn producer(self: &Arc<Self>) -> Option<RegenProducer> {
        let mut shared = lock(&self.shared);
        if shared.producers == 0 {
            return None;
        }
        shared.producers += 1;
        Some(RegenProducer {
            window: Arc::clone(self),
        })
    }

    /// A slot for `packed`, about to be dropped: `rank` orders it among the
    /// window's slots (the regenerator deposits in rank order, and phase B
    /// takes them in it).
    pub fn slot(self: &Arc<Self>, packed: &NarrowColumns, rank: u64) -> RegenSlot {
        // The digest first: concurrent drops must not queue on the window's
        // lock behind it (R-REGEN N1).
        let want = Want::of(packed);
        let settled = Arc::new(AtomicBool::new(false));
        let mut shared = lock(&self.shared);
        let id = shared.slots.len();
        shared.slots.push(Slot {
            rank,
            state: SlotState::Waiting,
            settled: Arc::clone(&settled),
        });
        shared.live.insert((rank, id), packed.data().len() as u64);
        RegenSlot {
            window: Arc::clone(self),
            id,
            rank,
            want,
            settled,
        }
    }

    /// Close: every slot not deposited fails with `why`, every waiter and
    /// every depositor returns. The first reason stays.
    pub fn close(&self, why: &str) {
        let mut shared = lock(&self.shared);
        if shared.closed.is_none() {
            shared.closed = Some(why.to_string());
        }
        self.closed.store(true, Ordering::Release);
        drop(shared);
        self.changed.notify_all();
    }

    /// Whether it is closed.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Test only: check deposits on their shape alone (`false`), so wrong
    /// columns of the right shape reach the checks behind the digest.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn set_verify(&self, on: bool) {
        self.verify.store(on, Ordering::SeqCst);
    }

    fn verifies(&self) -> bool {
        #[cfg(any(test, feature = "test-utils"))]
        return self.verify.load(Ordering::SeqCst);
        #[cfg(not(any(test, feature = "test-utils")))]
        true
    }

    /// One line for the prove's log.
    pub fn report(&self) -> String {
        let shared = lock(&self.shared);
        format!(
            "{} slots · {} deposited · {} taken · {} mismatches · {} never deposited · window \
             {:.2} GiB, high-water {:.2} GiB · depositors waited {:.2} s · takers waited {:.2} s",
            shared.slots.len(),
            shared.deposits,
            shared.taken,
            shared.mismatches,
            shared
                .slots
                .iter()
                .filter(|s| matches!(s.state, SlotState::Waiting))
                .count(),
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
        if shared.producers == 0 {
            self.window.gone.store(true, Ordering::Release);
        }
        drop(shared);
        self.window.changed.notify_all();
    }
}

/// Dropped packed columns' place in their window: the dropped table holds
/// one, the regenerator a clone. Clones are the same slot.
#[derive(Clone)]
pub struct RegenSlot {
    window: Arc<RegenWindow>,
    id: usize,
    rank: u64,
    want: Want,
    settled: Arc<AtomicBool>,
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

/// Two slots are equal when they are one slot, or when the columns they stand
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

    /// Its key in the window's order: (rank, then the order slots were made
    /// in), the order the window reserves in, so equal ranks cannot cross.
    pub fn order_key(&self) -> (u64, usize) {
        (self.rank, self.id)
    }

    /// Rows of the dropped columns.
    pub fn rows(&self) -> usize {
        self.want.rows
    }

    /// Columns dropped.
    pub fn cols(&self) -> usize {
        self.want.widths.len()
    }

    /// Packed bytes dropped.
    pub fn len(&self) -> usize {
        self.want.len
    }

    /// Whether the dropped columns had no bytes.
    pub fn is_empty(&self) -> bool {
        self.want.len == 0
    }

    /// The digest taken when the columns were dropped.
    pub fn digest(&self) -> [u64; 2] {
        self.want.digest
    }

    /// Its window.
    pub fn window(&self) -> &Arc<RegenWindow> {
        &self.window
    }

    /// Put the regenerated columns in the slot: checked against the dropped
    /// ones' shape and digest, then held until their group takes them. It
    /// waits first until the window admits its rank (see the module docs).
    /// Columns that are not the dropped ones fail the slot (their table is
    /// refused) and return [`RegenError::Mismatch`]; a closed window returns
    /// [`RegenError::Closed`]; a slot already settled refuses a second deposit.
    pub fn deposit(&self, packed: NarrowColumns) -> Result<(), RegenError> {
        let window = &self.window;
        let matches = if window.verifies() {
            Want::of(&packed) == self.want
        } else {
            self.want.same_shape(&packed)
        };
        let t = Instant::now();
        let mut shared = lock(&window.shared);
        if !matches!(shared.slots[self.id].state, SlotState::Waiting) {
            return Err(RegenError::Failed("deposited twice".to_string()));
        }
        if !matches {
            shared.settle(self.id, SlotState::Failed(RegenError::Mismatch));
            shared.mismatches += 1;
            drop(shared);
            window.changed.notify_all();
            return Err(RegenError::Mismatch);
        }
        let len = self.want.len as u64;
        let key = (self.rank, self.id);
        loop {
            if let Some(why) = &shared.closed {
                let why = why.clone();
                shared.deposit_wait_ns += t.elapsed().as_nanos() as u64;
                return Err(RegenError::Closed(why));
            }
            if !matches!(shared.slots[self.id].state, SlotState::Waiting) {
                return Err(RegenError::Failed("settled while it waited".to_string()));
            }
            if shared.admits(key, window.ahead) {
                break;
            }
            shared = window
                .changed
                .wait(shared)
                .unwrap_or_else(|e| e.into_inner());
        }
        shared.deposit_wait_ns += t.elapsed().as_nanos() as u64;
        shared.settle(self.id, SlotState::Ready(packed));
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
        if matches!(shared.slots[self.id].state, SlotState::Waiting) {
            shared.settle(
                self.id,
                SlotState::Failed(RegenError::Failed(why.to_string())),
            );
        }
        drop(shared);
        self.window.changed.notify_all();
    }

    /// Whether [`Self::wait`] would return at once: the columns are in, or
    /// they will never be. Lock-free.
    pub fn is_ready(&self) -> bool {
        self.settled.load(Ordering::Acquire)
            || self.window.closed.load(Ordering::Acquire)
            || self.window.gone.load(Ordering::Acquire)
    }

    /// Block until the columns are in, or they will never be.
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

    /// The regenerated columns, for their one reader, without blocking: the
    /// reader waited for the slot first ([`Self::wait`]). A slot not deposited
    /// yet is an error (and settles failed, so a late deposit is refused rather
    /// than parked for nobody); the bytes leave the window, which makes room
    /// for the next deposit.
    pub fn take(&self) -> Result<NarrowColumns, RegenError> {
        let mut shared = lock(&self.window.shared);
        let state = std::mem::replace(&mut shared.slots[self.id].state, SlotState::Taken);
        match state {
            SlotState::Ready(packed) => {
                shared.parked = shared.parked.saturating_sub(self.want.len as u64);
                shared.live.remove(&(self.rank, self.id));
                shared.taken += 1;
                drop(shared);
                self.window.changed.notify_all();
                Ok(packed)
            }
            SlotState::Failed(e) => {
                shared.slots[self.id].state = SlotState::Failed(e.clone());
                Err(e)
            }
            SlotState::Taken => Err(RegenError::Failed(
                "its columns were already taken".to_string(),
            )),
            SlotState::Waiting => {
                shared.slots[self.id].state = SlotState::Waiting;
                let e = match shared.settled(self.id) {
                    Some(Err(e)) => e,
                    _ => RegenError::Failed("not deposited when its group took it".to_string()),
                };
                shared.settle(self.id, SlotState::Failed(e.clone()));
                drop(shared);
                self.window.changed.notify_all();
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    /// Packed columns of `rows` × 3 whose words depend on `seed`.
    fn packed(rows: usize, seed: u64) -> NarrowColumns {
        let words: Vec<u64> = (0..rows as u64 * 3)
            .map(|i| (i ^ seed).wrapping_mul(0x9e37_79b9) % 70_000)
            .collect();
        NarrowColumns::pack_row_major(&words, 3).expect("packs")
    }

    /// `p` with the low bit of its first packed byte flipped: one wrong word,
    /// the same shape.
    fn bent(p: &NarrowColumns) -> NarrowColumns {
        let (rows, widths, mut data) = p.clone().into_parts();
        data[0] ^= 1;
        NarrowColumns::from_parts(rows, widths, data).expect("the same shape")
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

    /// ★ The digest is the spill store's: packed columns and the store's
    /// payload of the same parts digest alike, and one flipped bit, one row
    /// fewer or another width map changes it.
    #[test]
    fn the_digest_is_the_spill_stores() {
        let a = packed(100, 1);
        let (rows, widths, data) = a.clone().into_parts();
        let main = crate::narrow::NarrowMain::from_parts(rows, widths, data).expect("the store's");
        assert_eq!(digest_of(&a), main.digest());
        assert_ne!(digest_of(&a), digest_of(&bent(&a)));
        assert_ne!(digest_of(&a), digest_of(&packed(99, 1)));
        let (rows, mut widths, data) = a.clone().into_parts();
        widths.swap(0, 2);
        if widths != a.widths() {
            let other = NarrowColumns::from_parts(rows, widths, data).expect("same bytes");
            assert_ne!(digest_of(&a), digest_of(&other));
        }
    }

    /// The columns deposited are the columns taken, bit for bit, and their
    /// bytes leave the window.
    #[test]
    fn deposited_columns_are_taken_bit_for_bit() {
        let (window, _producer) = RegenWindow::new(1 << 30);
        let a = packed(100, 1);
        let slot = window.slot(&a, 0);
        assert!(!slot.is_ready());
        slot.deposit(a.clone()).unwrap();
        assert!(slot.is_ready());
        assert_eq!(slot.take().unwrap(), a);
        assert!(slot.take().is_err(), "one reader");
        let report = window.report();
        assert!(
            report.contains("1 deposited · 1 taken · 0 mismatches"),
            "{report}"
        );
    }

    /// ★ A deposit that is not the dropped columns — another word, another row
    /// count — fails its slot: the deposit and the taker both see `Mismatch`,
    /// and a second deposit is refused.
    #[test]
    fn other_columns_fail_their_slot() {
        let (window, _producer) = RegenWindow::new(1 << 30);
        let a = packed(100, 1);
        for other in [packed(100, 2), packed(99, 1)] {
            let slot = window.slot(&a, 0);
            assert_eq!(slot.deposit(other), Err(RegenError::Mismatch));
            assert!(slot.is_ready(), "a failed slot is settled");
            assert_eq!(slot.take(), Err(RegenError::Mismatch));
            assert!(matches!(
                slot.deposit(a.clone()),
                Err(RegenError::Failed(_))
            ));
        }
        let slot = window.slot(&a, 1);
        assert_eq!(slot.deposit(bent(&a)), Err(RegenError::Mismatch));
    }

    /// ★ A regenerator that stops — its producer dropped, as when its thread
    /// ends or panics — wakes every taker waiting on a slot it never
    /// deposited, which fails; what it deposited is still taken.
    #[test]
    fn a_dead_regenerator_fails_what_it_did_not_deposit() {
        let (window, producer) = RegenWindow::new(1 << 30);
        let a = packed(64, 1);
        let b = packed(64, 2);
        let slot_a = window.slot(&a, 0);
        let slot_b = window.slot(&b, 1);
        let waiter = {
            let slot_b = slot_b.clone();
            std::thread::spawn(move || {
                slot_b.wait();
                slot_b.take()
            })
        };
        std::thread::sleep(Duration::from_millis(50));
        let regenerator = std::thread::spawn(move || {
            let _producer = producer;
            slot_a.deposit(a).unwrap();
            panic!("the regenerator dies before its second table");
        });
        assert!(regenerator.join().is_err());
        let got = within(10, "the taker of a slot never deposited", move || {
            waiter.join().unwrap()
        });
        assert!(matches!(got, Err(RegenError::Failed(_))), "{got:?}");
        assert!(window.report().contains("1 deposited"));
    }

    /// ★ The window paces by rank from its frontier: with room for one table,
    /// ranks 1 and 2 wait while rank 0 is parked, rank 1 goes once rank 0 is
    /// taken, rank 2 once rank 1 is; a frontier larger than the window goes
    /// alone.
    #[test]
    fn a_deposit_waits_for_room_in_the_window() {
        let t: Vec<NarrowColumns> = (1..=3).map(|seed| packed(64, seed)).collect();
        let (window, _producer) = RegenWindow::new(t[0].data().len() as u64);
        let slots: Vec<RegenSlot> = (0..3).map(|r| window.slot(&t[r], r as u64)).collect();
        slots[0].deposit(t[0].clone()).unwrap();
        let (tx, rx) = mpsc::channel();
        let mut depositors = Vec::new();
        for r in [2, 1] {
            let (slot, columns, tx) = (slots[r].clone(), t[r].clone(), tx.clone());
            depositors.push(std::thread::spawn(move || {
                let out = slot.deposit(columns);
                tx.send(r).unwrap();
                out
            }));
            std::thread::sleep(Duration::from_millis(50));
        }
        let quiet = Duration::from_millis(200);
        assert!(rx.recv_timeout(quiet).is_err(), "deposited past the window");
        assert_eq!(slots[0].take().unwrap(), t[0]);
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(5)),
            Ok(1),
            "rank 1, the frontier"
        );
        assert!(rx.recv_timeout(quiet).is_err(), "rank 2 past the window");
        assert_eq!(slots[1].take().unwrap(), t[1]);
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)), Ok(2));
        for d in depositors {
            d.join().unwrap().unwrap();
        }
        assert_eq!(slots[2].take().unwrap(), t[2]);
        let (small, _p) = RegenWindow::new(1);
        let big = packed(1000, 3);
        small.slot(&big, 0).deposit(big).unwrap();
    }

    /// The window stays a bound for a regenerator that deposits in rank order
    /// with no taker: it deposits what the window reserves, then waits.
    #[test]
    fn an_in_order_regenerator_stays_inside_the_window() {
        let t: Vec<NarrowColumns> = (0..6).map(|seed| packed(64, seed)).collect();
        let len = t[0].data().len() as u64;
        let (window, _producer) = RegenWindow::new(2 * len);
        let slots: Vec<RegenSlot> = (0..6).map(|r| window.slot(&t[r], r as u64)).collect();
        let (tx, rx) = mpsc::channel();
        let regenerator = {
            let (slots, t) = (slots.clone(), t.clone());
            std::thread::spawn(move || {
                for (slot, columns) in slots.iter().zip(t) {
                    if slot.deposit(columns).is_err() {
                        return;
                    }
                    tx.send(()).unwrap();
                }
            })
        };
        for _ in 0..2 {
            rx.recv_timeout(Duration::from_secs(5))
                .expect("inside the window");
        }
        assert!(
            rx.recv_timeout(Duration::from_millis(200)).is_err(),
            "past the window"
        );
        window.close("done");
        regenerator.join().unwrap();
    }

    /// ★ Generators that take their jobs in rank order (a FIFO dispatch, as a
    /// regenerator hands them out) and finish them out of order never
    /// deadlock a taker that takes in rank order: with three generators, the
    /// later a rank the sooner its generation ends, inside a window of one
    /// table, every rank comes back.
    #[test]
    fn out_of_order_depositors_never_deadlock_an_in_order_taker() {
        let t: Vec<NarrowColumns> = (0..9).map(|seed| packed(64, seed)).collect();
        let (window, producer) = RegenWindow::new(t[0].data().len() as u64);
        let slots: Vec<RegenSlot> = (0..9).map(|r| window.slot(&t[r], r as u64)).collect();
        let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut generators = Vec::new();
        for _ in 0..3 {
            let token = window.producer().expect("alive");
            let (slots, t, next) = (slots.clone(), t.clone(), Arc::clone(&next));
            generators.push(std::thread::spawn(move || {
                let _token = token;
                loop {
                    let r = next.fetch_add(1, Ordering::SeqCst);
                    if r >= 9 {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(5 * (9 - r as u64)));
                    let _ = slots[r].deposit(t[r].clone());
                }
            }));
        }
        drop(producer);
        let taken = within(20, "an in-order taker", move || {
            (0..9)
                .map(|r| {
                    slots[r].wait();
                    slots[r].take()
                })
                .collect::<Vec<_>>()
        });
        for g in generators {
            g.join().unwrap();
        }
        for (r, got) in taken.into_iter().enumerate() {
            assert_eq!(got.as_ref(), Ok(&t[r]), "rank {r}");
        }
    }

    /// ★ Closing the window wakes a depositor waiting for room and a taker
    /// waiting for its columns: both return `Closed`, and nothing hangs.
    #[test]
    fn closing_the_window_wakes_every_waiter() {
        let a = packed(64, 1);
        let b = packed(64, 2);
        let c = packed(64, 3);
        let (window, _producer) = RegenWindow::new(a.data().len() as u64);
        let slot_a = window.slot(&a, 0);
        let slot_b = window.slot(&b, 1);
        let slot_c = window.slot(&c, 2);
        slot_a.deposit(a).unwrap();
        // Rank 2 waits for room (rank 1 is the lowest, never deposited); the
        // taker of rank 1 waits for it.
        let depositor = std::thread::spawn(move || slot_c.deposit(c));
        let taker = std::thread::spawn(move || {
            slot_b.wait();
            slot_b.take()
        });
        std::thread::sleep(Duration::from_millis(100));
        window.close("the prove ended");
        let (dep, take) = within(10, "a closed window's waiters", move || {
            (depositor.join().unwrap(), taker.join().unwrap())
        });
        assert!(matches!(dep, Err(RegenError::Closed(_))), "{dep:?}");
        assert!(matches!(take, Err(RegenError::Closed(_))), "{take:?}");
        assert!(window.is_closed());
        drop(b);
    }

    /// ★ A take never blocks: before its deposit it is an error, the slot
    /// settles failed, and the late deposit is refused instead of parked for
    /// a reader that is gone.
    #[test]
    fn a_take_before_the_deposit_is_an_error_not_a_wait() {
        let (window, _producer) = RegenWindow::new(1 << 30);
        let a = packed(16, 1);
        let slot = window.slot(&a, 0);
        let got = within(5, "a take before the deposit", {
            let slot = slot.clone();
            move || slot.take()
        });
        assert!(matches!(got, Err(RegenError::Failed(_))), "{got:?}");
        assert!(slot.is_ready());
        assert!(slot.deposit(a).is_err());
        assert!(window.report().contains("0 deposited"));
    }

    /// Readiness without the lock: settled by a deposit, by a failure, by the
    /// window closing, or by its last producer leaving.
    #[test]
    fn readiness_is_read_without_the_lock() {
        let a = packed(16, 1);
        let (window, producer) = RegenWindow::new(1 << 30);
        let (s0, s1, s2) = (window.slot(&a, 0), window.slot(&a, 1), window.slot(&a, 2));
        assert!(!s0.is_ready() && !s1.is_ready() && !s2.is_ready());
        s0.deposit(a.clone()).unwrap();
        s1.fail("no");
        assert!(s0.is_ready() && s1.is_ready() && !s2.is_ready());
        drop(producer);
        assert!(s2.is_ready(), "its last producer left");
        assert!(
            window.producer().is_none(),
            "no producer after the last one left"
        );
        let (closed, _p) = RegenWindow::new(1 << 30);
        let s = closed.slot(&a, 0);
        closed.close("done");
        assert!(s.is_ready() && closed.is_closed());
    }

    /// Test hook: with the digest off wrong columns of the right shape are
    /// deposited (the checks behind the digest then have to catch them); a
    /// wrong shape is refused all the same.
    #[test]
    fn the_shape_is_checked_with_the_digest_off() {
        let (window, _producer) = RegenWindow::new(1 << 30);
        window.set_verify(false);
        let a = packed(16, 1);
        window.slot(&a, 0).deposit(bent(&a)).unwrap();
        assert_eq!(
            window.slot(&a, 1).deposit(packed(15, 1)),
            Err(RegenError::Mismatch)
        );
    }

    /// A slot given up fails with the regenerator's reason.
    #[test]
    fn a_slot_given_up_fails() {
        let (window, _producer) = RegenWindow::new(1 << 30);
        let a = packed(10, 1);
        let slot = window.slot(&a, 0);
        slot.fail("the walk refused window 3");
        slot.wait();
        assert_eq!(
            slot.take(),
            Err(RegenError::Failed("the walk refused window 3".to_string()))
        );
    }
}

//! The tree programs' host budget (`LAMBDA_VM_TREE_PROGRAM_BUDGET`) and the
//! ordered streaming emitter that produces the programs it defers.
//!
//! A block tree's program — a leaf's or a node's [`super::compiler::LfmProgram`]
//! — is emitted on the host, waits until a prover takes it, and is dropped when
//! its proof returns. Emitted all ahead, a p90 block's 107 leaf programs (45.76
//! GiB) were held from phase B to level 0: they bound phase B and the
//! recursion's floor at every host size (BIG 659). Emitted just in time, the
//! same block's phase B fell 78.4 → 40.8 GiB and its recursion 93.7 → 63.3 GiB
//! (BIG 662, I-MEMFIT §1.7b). The budget decides, program by program, which
//! may be emitted ahead and which wait:
//!
//! - **Room.** `auto`, the default, admits a program while the process's
//!   resident bytes now (`VmRSS`) plus the program's bytes, and the bytes of the
//!   programs admitted but not yet emitted, stay under the host target the
//!   caller gives: the pipeline's spill target (`LAMBDA_VM_BLOCK_SPILL_TARGET_GIB`,
//!   else the cgroup's limit or `MemTotal` less 10 GiB). One knob sizes both,
//!   and the allocator's retained pages count because `VmRSS` counts them. A
//!   roomy host never reaches the target, so every program is emitted where it
//!   was before. `off` never refuses: the control. `<GiB>` caps the bytes of
//!   the admitted programs that are still alive, whatever the host.
//! - **The next [`AHEAD`] programs the provers will take are always
//!   admitted**, room or not: a program comes in once fewer than `AHEAD`
//!   programs below it in prove order are still untaken. The program the next
//!   prover needs never waits for room, so a tight budget cannot deadlock the
//!   tree; and with room gone the emission still runs `AHEAD` programs ahead
//!   of the provers (one at a time would make every emission wait for the
//!   take before it: ≈ 6 s a node). The programs alive stay under the room
//!   plus `AHEAD` more than the provers in flight hold.
//! - **No program is admitted while a lower one waits**, so programs emitted
//!   early (a node's, during level 0) never take the room a leaf needs.
//!
//! Prove order is the leaves 0..n, then level 1's nodes, level 2's, … the top:
//! the order the provers take them (`in_index_order` within a level, the levels
//! one after another). The budget only moves when a program is emitted; a
//! program's bytes are a function of the plan and its children's derived
//! artifacts, never of the time or the thread that emitted it, so the
//! programs, their ids and the top are the same under any setting.
//!
//! `VmRSS` is read from `/proc/self/status` (the kernel's RSS counters, no
//! page walk; one open and read, tens of µs): once per admission decision and
//! once per 200 ms while a program waits, never on a per-op path. Without the
//! file (not Linux) `auto` has no reading and admits by the other two rules
//! only, which is what the line reports.
//!
//! ⛔ [`ProgramBudget::acquire`] parks its thread on a condvar. Call it only
//! from a plain thread, never from a rayon worker: a worker that parked there
//! could hold the pool's jobs a prover waits for (the derive gate's hazard,
//! [`super::derive_gate`]). [`ProgramBudget::try_acquire`] never waits.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const GIB: u64 = 1 << 30;

/// The programs admitted past the room: the next this many to be taken. Four
/// is the streaming emitter's thread count, and the window BIG 662 measured
/// (a p90 recursion of 63.3 GiB, emitted by chunks of four).
pub const AHEAD: usize = 4;

/// The budget's knob.
pub const PROGRAM_BUDGET_ENV: &str = "LAMBDA_VM_TREE_PROGRAM_BUDGET";

/// What [`PROGRAM_BUDGET_ENV`] says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetSetting {
    /// Unset, empty or `auto` (the default): room is the host, `VmRSS`
    /// against the target the caller gives ([`Room::Host`]).
    Auto,
    /// `off`: never refuse ([`Room::Unbounded`]).
    Off,
    /// `<GiB>`: the admitted programs alive may hold this many bytes
    /// ([`Room::Bytes`]).
    Bytes(u64),
}

/// [`PROGRAM_BUDGET_ENV`], parsed.
pub fn setting_from_env() -> Result<BudgetSetting, String> {
    parse_setting(std::env::var(PROGRAM_BUDGET_ENV).ok().as_deref())
}

/// A value of [`PROGRAM_BUDGET_ENV`]; nonsense is an error, not a default.
pub fn parse_setting(value: Option<&str>) -> Result<BudgetSetting, String> {
    match value.map(str::trim) {
        None | Some("" | "auto") => Ok(BudgetSetting::Auto),
        Some("off") => Ok(BudgetSetting::Off),
        Some(v) => v
            .parse::<f64>()
            .ok()
            .filter(|g| g.is_finite() && *g >= 0.0)
            .map(|g| BudgetSetting::Bytes((g * GIB as f64) as u64))
            .ok_or_else(|| {
                format!("{PROGRAM_BUDGET_ENV} must be auto, off or a size in GiB, got `{v}`")
            }),
    }
}

/// What counts as room.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Room {
    /// Everything fits.
    Unbounded,
    /// The admitted programs alive hold at most this many bytes.
    Bytes(u64),
    /// The process's resident bytes, plus the programs admitted and not yet
    /// emitted, plus the new one, stay at or under `target`.
    Host { target: u64 },
}

impl Room {
    /// The room a setting gives against the host `target`.
    pub fn of(setting: BudgetSetting, target: u64) -> Self {
        match setting {
            BudgetSetting::Auto => Room::Host { target },
            BudgetSetting::Off => Room::Unbounded,
            BudgetSetting::Bytes(bytes) => Room::Bytes(bytes),
        }
    }
}

/// This process's resident bytes (`VmRSS`), on Linux.
pub fn resident_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib: u64 = status
        .lines()
        .find(|l| l.starts_with("VmRSS:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kib << 10)
}

/// How a program got in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Admission {
    Room,
    Ahead,
}

#[derive(Default)]
struct State {
    /// The bytes of the admitted programs still alive.
    held: u64,
    /// Of them, the estimates of the programs admitted and not yet emitted
    /// (the host does not hold them yet).
    unemitted: u64,
    /// The admitted programs alive: order → taken by a prover.
    alive: BTreeMap<usize, bool>,
    /// Every order ever admitted, and every order ever taken by a prover.
    admitted: BTreeSet<usize>,
    claimed: BTreeSet<usize>,
    /// The orders waiting in [`ProgramBudget::acquire`].
    waiting: BTreeMap<usize, usize>,
    failed: bool,
    // What the summary reports.
    high_water: u64,
    by_room: usize,
    by_ahead: usize,
    refused: usize,
    waited_secs: f64,
}

impl State {
    fn admit(&mut self, order: usize, bytes: u64, how: Admission) {
        self.held += bytes;
        self.unemitted += bytes;
        self.high_water = self.high_water.max(self.held);
        self.alive.insert(order, false);
        self.admitted.insert(order);
        match how {
            Admission::Room => self.by_room += 1,
            Admission::Ahead => self.by_ahead += 1,
        }
    }
}

/// The budget of one tree's programs. Every program is admitted once, by its
/// prove order, before it is emitted; its [`Permit`] travels with it and
/// gives its bytes back when it is dropped.
pub struct ProgramBudget {
    room: Room,
    ahead: usize,
    reading: Box<dyn Fn() -> Option<u64> + Send + Sync>,
    state: Mutex<State>,
    changed: Condvar,
}

/// A program's admission: claimed when a prover takes it, released when
/// dropped (with the program).
pub struct Permit {
    budget: Arc<ProgramBudget>,
    order: usize,
    bytes: u64,
    emitted: bool,
}

impl ProgramBudget {
    /// A budget with `room` and [`AHEAD`], reading the host with
    /// [`resident_bytes`].
    pub fn new(room: Room) -> Arc<Self> {
        Self::with_reading(room, AHEAD, Box::new(resident_bytes))
    }

    /// A budget with `room`, `ahead` programs always admitted ahead of the
    /// provers (at least one), and the host reading `reading` (the tests'
    /// fake hosts).
    pub fn with_reading(
        room: Room,
        ahead: usize,
        reading: Box<dyn Fn() -> Option<u64> + Send + Sync>,
    ) -> Arc<Self> {
        Arc::new(Self {
            room,
            ahead: ahead.max(1),
            reading,
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
        })
    }

    /// Whether this budget ever refuses.
    pub fn bounded(&self) -> bool {
        self.room != Room::Unbounded
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether `order` may come in now, and how.
    fn admissible(&self, st: &State, order: usize, bytes: u64) -> Option<Admission> {
        if st.waiting.range(..order).next().is_some() {
            return None;
        }
        let room = match self.room {
            Room::Unbounded => true,
            Room::Bytes(limit) => st.held + bytes <= limit,
            Room::Host { target } => {
                (self.reading)().is_none_or(|rss| rss + st.unemitted + bytes <= target)
            }
        };
        if room {
            return Some(Admission::Room);
        }
        let untaken_below = order - st.claimed.range(..order).count();
        (untaken_below < self.ahead).then_some(Admission::Ahead)
    }

    fn permit(self: &Arc<Self>, order: usize, bytes: u64) -> Permit {
        Permit {
            budget: self.clone(),
            order,
            bytes,
            emitted: false,
        }
    }

    /// Admits program `order` with `bytes` estimated, waiting until it may
    /// come in. An error once the budget has failed ([`Self::fail`]).
    pub fn acquire(self: &Arc<Self>, order: usize, bytes: u64) -> Result<Permit, String> {
        let t = Instant::now();
        let mut st = self.lock();
        assert!(
            !st.admitted.contains(&order),
            "program {order} admitted twice"
        );
        *st.waiting.entry(order).or_insert(0) += 1;
        let how = loop {
            if st.failed {
                break None;
            }
            if let Some(how) = self.admissible(&st, order, bytes) {
                break Some(how);
            }
            st = self
                .changed
                .wait_timeout(st, Duration::from_millis(200))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        };
        match st.waiting.get_mut(&order) {
            Some(n) if *n > 1 => *n -= 1,
            _ => {
                st.waiting.remove(&order);
            }
        }
        st.waited_secs += t.elapsed().as_secs_f64();
        let Some(how) = how else {
            drop(st);
            self.changed.notify_all();
            return Err(format!("program {order}: the tree's builder failed"));
        };
        st.admit(order, bytes, how);
        drop(st);
        self.changed.notify_all();
        Ok(self.permit(order, bytes))
    }

    /// Admits program `order` only if it may come in now.
    pub fn try_acquire(self: &Arc<Self>, order: usize, bytes: u64) -> Option<Permit> {
        let mut st = self.lock();
        if st.failed || st.admitted.contains(&order) {
            return None;
        }
        match self.admissible(&st, order, bytes) {
            Some(how) => {
                st.admit(order, bytes, how);
                Some(self.permit(order, bytes))
            }
            None => {
                st.refused += 1;
                None
            }
        }
    }

    /// Every waiter returns an error from now on (the builder failed).
    pub fn fail(&self) {
        self.lock().failed = true;
        self.changed.notify_all();
    }

    /// The summary: the most bytes alive at once, how the programs came in,
    /// and how long admissions waited.
    pub fn summary(&self) -> String {
        let st = self.lock();
        format!(
            "{} · held at most {:.2} GiB · {} by room, {} among the next {} to be taken · {} \
             refused beside the base · admissions waited Σ {:.2}s",
            self.describe(),
            st.high_water as f64 / GIB as f64,
            st.by_room,
            st.by_ahead,
            self.ahead,
            st.refused,
            st.waited_secs
        )
    }

    /// The room in words.
    pub fn describe(&self) -> String {
        let gib = |b: u64| b as f64 / GIB as f64;
        match self.room {
            Room::Unbounded => "off (every program where it was)".to_string(),
            Room::Bytes(b) => format!("{:.2} GiB of programs", gib(b)),
            Room::Host { target } => match (self.reading)() {
                Some(rss) => format!(
                    "auto: VmRSS ({:.2} GiB now) against the target {:.2} GiB",
                    gib(rss),
                    gib(target)
                ),
                None => format!(
                    "auto: no VmRSS reading (target {:.2} GiB): room is never short",
                    gib(target)
                ),
            },
        }
    }
}

impl Permit {
    /// The program is emitted: its estimate becomes its `actual` bytes, now
    /// on the host.
    pub fn emitted(&mut self, actual: u64) {
        let mut st = self.budget.lock();
        if !self.emitted {
            st.unemitted -= self.bytes;
            self.emitted = true;
        }
        st.held = st.held - self.bytes + actual;
        st.high_water = st.high_water.max(st.held);
        self.bytes = actual;
        drop(st);
        self.budget.changed.notify_all();
    }

    /// A prover took the program.
    pub fn claim(&self) {
        let mut st = self.budget.lock();
        if let Some(taken) = st.alive.get_mut(&self.order) {
            *taken = true;
        }
        st.claimed.insert(self.order);
        drop(st);
        self.budget.changed.notify_all();
    }

    /// The program's prove order.
    pub fn order(&self) -> usize {
        self.order
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        let mut st = self.budget.lock();
        st.held -= self.bytes;
        if !self.emitted {
            st.unemitted -= self.bytes;
        }
        st.alive.remove(&self.order);
        drop(st);
        self.budget.changed.notify_all();
    }
}

/// Emits `items` on `threads` plain threads, each taking the next item as it
/// is free, and hands each result to `deliver` strictly in item order, as soon
/// as it and every item before it are done (a reorder buffer: a slow item
/// holds back the delivery of the faster ones after it, not their emission).
/// Delivery stops after an error is delivered or once `deliver` returns
/// `false`; the workers then stop taking items. The bytes ahead are bounded
/// by what `emit` admits (a [`ProgramBudget`]), not here.
pub fn emit_ordered<T: Send>(
    items: Range<usize>,
    threads: usize,
    emit: &(dyn Fn(usize) -> Result<T, String> + Sync),
    deliver: &mut dyn FnMut(usize, Result<T, String>) -> bool,
) {
    let next = AtomicUsize::new(items.start);
    let stop = std::sync::atomic::AtomicBool::new(false);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<T, String>)>();
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            let (tx, next, stop) = (tx.clone(), &next, &stop);
            scope.spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    let k = next.fetch_add(1, Ordering::SeqCst);
                    if k >= items.end {
                        return;
                    }
                    if tx.send((k, emit(k))).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let mut pending: BTreeMap<usize, Result<T, String>> = BTreeMap::new();
        let mut want = items.start;
        'deliver: while want < items.end {
            let Ok((k, result)) = rx.recv() else {
                break;
            };
            pending.insert(k, result);
            while let Some(result) = pending.remove(&want) {
                let failed = result.is_err();
                if !deliver(want, result) || failed {
                    break 'deliver;
                }
                want += 1;
            }
        }
        stop.store(true, Ordering::SeqCst);
        // The workers still emitting finish their item; what they send is
        // dropped with the channel.
        drop(rx);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    const MIB: u64 = 1 << 20;

    #[test]
    fn the_knob_reads_auto_off_or_a_size_and_refuses_nonsense() {
        assert_eq!(parse_setting(None), Ok(BudgetSetting::Auto));
        assert_eq!(parse_setting(Some("")), Ok(BudgetSetting::Auto));
        assert_eq!(parse_setting(Some(" auto ")), Ok(BudgetSetting::Auto));
        assert_eq!(parse_setting(Some("off")), Ok(BudgetSetting::Off));
        assert_eq!(
            parse_setting(Some("1.5")),
            Ok(BudgetSetting::Bytes(3 * GIB / 2))
        );
        assert_eq!(parse_setting(Some("0")), Ok(BudgetSetting::Bytes(0)));
        assert!(parse_setting(Some("-1")).is_err());
        assert!(parse_setting(Some("lots")).is_err());
        assert_eq!(
            Room::of(BudgetSetting::Auto, 54 * GIB),
            Room::Host { target: 54 * GIB }
        );
        assert_eq!(Room::of(BudgetSetting::Off, 54 * GIB), Room::Unbounded);
    }

    /// A fake host: its resident bytes are the budget's held bytes plus a
    /// fixed rest, as if every emitted program were resident.
    fn host(rest: u64) -> (Arc<AtomicU64>, Box<dyn Fn() -> Option<u64> + Send + Sync>) {
        let rss = Arc::new(AtomicU64::new(rest));
        let r = rss.clone();
        (rss, Box::new(move || Some(r.load(Ordering::SeqCst))))
    }

    fn bytes_budget(limit: u64, ahead: usize) -> Arc<ProgramBudget> {
        ProgramBudget::with_reading(Room::Bytes(limit), ahead, Box::new(|| None))
    }

    #[test]
    fn room_admits_and_a_full_budget_refuses_all_but_the_next_to_be_taken() {
        let b = bytes_budget(100 * MIB, 1);
        let p0 = b.try_acquire(0, 60 * MIB).expect("room");
        let p1 = b.try_acquire(1, 30 * MIB).expect("room");
        assert!(b.try_acquire(3, 30 * MIB).is_none(), "no room, not next");
        assert!(
            b.try_acquire(2, 30 * MIB).is_none(),
            "next pending, but 0 and 1 are not taken"
        );
        p0.claim();
        p1.claim();
        let p2 = b.try_acquire(2, 30 * MIB).expect("the next to be taken");
        assert!(
            b.try_acquire(3, 30 * MIB).is_none(),
            "only one over the room while it is not taken (ahead 1)"
        );
        drop(p0);
        let p3 = b.try_acquire(3, 30 * MIB).expect("room again");
        assert!(
            b.summary()
                .contains("3 by room, 1 among the next 1 to be taken")
        );
        drop((p1, p2, p3));
        assert_eq!(b.lock().held, 0);
    }

    /// With room gone, the next `ahead` programs to be taken still come in
    /// (the emission keeps that many ahead of the provers), and no more.
    #[test]
    fn with_no_room_the_next_ahead_programs_still_come_in() {
        let b = bytes_budget(0, 3);
        let p: Vec<Permit> = (0..3)
            .map(|k| b.try_acquire(k, MIB).expect("one of the next three"))
            .collect();
        assert!(b.try_acquire(3, MIB).is_none(), "three untaken below 3");
        p[0].claim();
        let p3 = b.try_acquire(3, MIB).expect("two untaken below 3");
        assert!(b.try_acquire(4, MIB).is_none(), "three untaken below 4");
        drop(p);
        assert!(
            b.try_acquire(4, MIB).is_none(),
            "dropped without a take: 1 and 2 are still untaken"
        );
        drop(p3);
    }

    #[test]
    fn auto_counts_the_resident_set_and_the_programs_not_yet_emitted() {
        let (rss, reading) = host(40 * MIB);
        let b = ProgramBudget::with_reading(Room::Host { target: 100 * MIB }, 1, reading);
        let mut p0 = b.try_acquire(0, 30 * MIB).expect("40 + 30 ≤ 100");
        assert!(
            b.try_acquire(1, 31 * MIB).is_none(),
            "40 resident + 30 admitted, not emitted + 31 > 100"
        );
        // Emitted: the host now holds it, and the reading says so.
        rss.fetch_add(30 * MIB, Ordering::SeqCst);
        p0.emitted(30 * MIB);
        assert!(b.try_acquire(1, 31 * MIB).is_none(), "70 + 31 > 100");
        let p1 = b.try_acquire(1, 30 * MIB).expect("70 + 30 ≤ 100");
        drop((p0, p1));
        let none = ProgramBudget::with_reading(Room::Host { target: 0 }, 1, Box::new(|| None));
        assert!(
            none.try_acquire(0, GIB).is_some(),
            "no reading: room never short"
        );
    }

    #[test]
    fn a_program_never_jumps_a_lower_waiter() {
        let b = bytes_budget(50 * MIB, 1);
        let p0 = b.try_acquire(0, 50 * MIB).expect("room");
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| b.acquire(1, 10 * MIB).map(|p| p.order()));
            while b.lock().waiting.is_empty() {
                std::thread::yield_now();
            }
            drop(p0);
            // Room for 2 now, but 1 waits below it and comes first.
            let got1 = waiter.join().expect("the waiter").expect("admitted");
            assert_eq!(got1, 1);
        });
        let mut st = b.lock();
        st.waiting.insert(3, 1);
        drop(st);
        assert!(b.try_acquire(4, MIB).is_none(), "3 waits below 4");
        assert!(b.try_acquire(2, MIB).is_some(), "2 is below the waiter");
    }

    /// A one-byte budget and three levels of in-order consumers, as the tree
    /// takes its programs: every program is admitted, in order, no consumer
    /// waits forever, and at most `ahead` programs are alive beyond those in a
    /// prover's hands (ahead 1 and [`AHEAD`]). Without the lowest-pending rule the tree deadlocks: the
    /// timeout then fails the budget, which frees every thread, and the test
    /// fails by name instead of hanging.
    #[test]
    fn a_one_byte_budget_cannot_deadlock_the_tree() {
        for ahead in [1, AHEAD] {
            one_byte_tree(ahead);
        }
    }

    fn one_byte_tree(ahead: usize) {
        let b = bytes_budget(1, ahead);
        let levels: [Range<usize>; 3] = [0..12, 12..15, 15..16];
        let slots: Vec<Mutex<Option<Permit>>> = (0..16).map(|_| Mutex::new(None)).collect();
        let max_alive = AtomicUsize::new(0);
        let stop = std::sync::atomic::AtomicBool::new(false);
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let finished = std::thread::scope(|scope| {
            // Two emitters: the leaves in order, the nodes in order.
            for orders in [0..12usize, 12..16] {
                let (b, slots) = (&b, &slots);
                scope.spawn(move || {
                    for k in orders {
                        let Ok(p) = b.acquire(k, 10 * MIB) else {
                            return;
                        };
                        *slots[k].lock().unwrap() = Some(p);
                    }
                });
            }
            let (b, slots, max_alive, stop) = (&b, &slots, &max_alive, &stop);
            scope.spawn(move || {
                for level in levels {
                    // Three provers a level, each taking the next order.
                    let next = AtomicUsize::new(level.start);
                    std::thread::scope(|s| {
                        for _ in 0..3 {
                            s.spawn(|| {
                                loop {
                                    let k = next.fetch_add(1, Ordering::SeqCst);
                                    if k >= level.end {
                                        return;
                                    }
                                    let p = loop {
                                        if let Some(p) = slots[k].lock().unwrap().take() {
                                            break p;
                                        }
                                        if stop.load(Ordering::SeqCst) {
                                            return;
                                        }
                                        std::thread::sleep(Duration::from_millis(1));
                                    };
                                    p.claim();
                                    max_alive.fetch_max(b.lock().alive.len(), Ordering::SeqCst);
                                    std::thread::sleep(Duration::from_millis(2));
                                    drop(p);
                                }
                            });
                        }
                    });
                }
                let _ = done_tx.send(());
            });
            let finished = done_rx.recv_timeout(Duration::from_secs(20)).is_ok();
            if !finished {
                stop.store(true, Ordering::SeqCst);
                b.fail();
            }
            finished
        });
        assert!(
            finished,
            "the tree deadlocked under a one-byte budget (ahead {ahead})"
        );
        assert!(
            max_alive.load(Ordering::SeqCst) <= 3 + ahead,
            "{} alive: more than the provers in flight plus {ahead}",
            max_alive.load(Ordering::SeqCst)
        );
        assert_eq!(b.lock().held, 0);
    }

    #[test]
    fn a_failed_budget_wakes_every_waiter() {
        let b = bytes_budget(0, 1);
        let held = b.try_acquire(0, MIB).expect("the next to be taken");
        std::thread::scope(|scope| {
            let w = scope.spawn(|| b.acquire(1, MIB).err());
            while b.lock().waiting.is_empty() {
                std::thread::yield_now();
            }
            b.fail();
            assert!(w.join().unwrap().is_some(), "the waiter returns an error");
        });
        drop(held);
    }

    /// The streaming emitter delivers in item order even when later items
    /// finish first. Delivering in completion order instead fails here.
    #[test]
    fn the_streaming_emitter_delivers_in_order_under_adversarial_completion() {
        let items = 3..23;
        // Earlier items take longer: completion order is the reverse of item
        // order within each wave of four.
        let emit = |k: usize| -> Result<usize, String> {
            std::thread::sleep(Duration::from_millis(((23 - k) * 3) as u64));
            Ok(k * 10)
        };
        let mut got = Vec::new();
        emit_ordered(items.clone(), 4, &emit, &mut |k, r| {
            got.push((k, r.expect("emitted")));
            true
        });
        let want: Vec<(usize, usize)> = items.map(|k| (k, k * 10)).collect();
        assert_eq!(got, want);
    }

    /// An error is delivered after every item before it, and nothing after.
    #[test]
    fn the_streaming_emitter_stops_at_an_error() {
        let emit = |k: usize| -> Result<usize, String> {
            if k == 7 {
                Err(format!("item {k} does not emit"))
            } else {
                Ok(k)
            }
        };
        let mut got = Vec::new();
        emit_ordered(5..30, 3, &emit, &mut |k, r| {
            got.push((k, r));
            true
        });
        assert_eq!(
            got,
            vec![
                (5, Ok(5)),
                (6, Ok(6)),
                (7, Err("item 7 does not emit".to_string()))
            ]
        );
    }
}

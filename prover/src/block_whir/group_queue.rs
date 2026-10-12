//! The closed groups waiting for phase A (`LAMBDA_VM_BLOCK_GROUP_QUEUE`).
//!
//! The packer closes groups in arrival order and phase A commits them in that
//! order. By default one closed group waits, in a channel of one, and the
//! packer holds the next: a card that is behind holds back the layout, the
//! walk and the executor, and the finish behind its byte gate.
//!
//! With `room`, a block that proves with no disk lets closed groups wait
//! behind one another while the host has room for them ([`Rule::Room`]): the
//! host then builds ahead of the card. A group the rule refuses waits in the
//! packer's hands, as every group past the first does by default, until the
//! rule admits it or nothing waits. The groups, their order and the proof's
//! bytes are the same: the queue is first in, first out, and a group is closed
//! before it comes here.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::Error;
use crate::alloc_purge::NoDisk;

const GIB: f64 = (1u64 << 30) as f64;

/// `LAMBDA_VM_BLOCK_GROUP_QUEUE`: `none` (and unset or empty) | `room` |
/// `<GiB>`. Anything else refuses the prove.
pub(crate) const GROUP_QUEUE_ENV: &str = "LAMBDA_VM_BLOCK_GROUP_QUEUE";

/// What [`GROUP_QUEUE_ENV`] says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Setting {
    /// One closed group waits: the channel of one.
    None,
    /// Closed groups wait while the host has room ([`Rule::Room`]).
    Room,
    /// Closed groups wait up to this many bytes, the host not read
    /// ([`Rule::Bytes`]): a measurement arm.
    Bytes(u64),
}

/// [`Setting`] from [`GROUP_QUEUE_ENV`].
pub(crate) fn setting() -> Result<Setting, Error> {
    parse_setting(std::env::var(GROUP_QUEUE_ENV).ok().as_deref())
}

pub(crate) fn parse_setting(value: Option<&str>) -> Result<Setting, Error> {
    match value.map(str::trim) {
        None | Some("" | "none") => Ok(Setting::None),
        Some("room") => Ok(Setting::Room),
        Some(v) => v
            .parse::<f64>()
            .ok()
            .filter(|gib| gib.is_finite() && *gib > 0.0)
            .map(|gib| Setting::Bytes((gib * GIB) as u64))
            .ok_or_else(|| {
                Error::Prover(format!(
                    "{GROUP_QUEUE_ENV}={v}: none, room or a number of GiB"
                ))
            }),
    }
}

/// When a closed group may wait behind another. With nothing waiting a group
/// always goes, whatever the rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rule {
    /// While the host has room by the purge rule's own test
    /// ([`crate::alloc_purge::has_room`]): the resident set, the bytes
    /// waiting, the group's own and the block's headroom together stay under
    /// its pressure line; and never more than `cap` bytes waiting in all. The
    /// waiting bytes are in the resident set too: counted twice, on the
    /// careful side.
    Room { room: NoDisk, cap: u64 },
    /// While at most this many bytes wait.
    Bytes(u64),
}

impl Rule {
    /// The rule a block runs under `setting`. `no_disk` is what the purge
    /// rule weighs for a block that proves with no disk; any other block
    /// (`None`: one with a store, or with nothing to rebuild its tables) keeps
    /// the channel of one under every setting.
    pub(crate) fn of(setting: Setting, no_disk: Option<NoDisk>, cap: u64) -> Option<Self> {
        let room = no_disk?;
        match setting {
            Setting::None => None,
            Setting::Room => Some(Self::Room { room, cap }),
            Setting::Bytes(bytes) => Some(Self::Bytes(bytes)),
        }
    }

    /// Whether a group of `bytes` may wait behind `waiting` bytes. No reading
    /// of the resident set is no room.
    fn admits(self, waiting: u64, bytes: u64, resident: impl FnOnce() -> Option<u64>) -> bool {
        let after = waiting.saturating_add(bytes);
        match self {
            Self::Bytes(budget) => after <= budget,
            Self::Room { room, cap } => {
                after <= cap
                    && crate::alloc_purge::has_room(
                        Some(room),
                        resident().map(|resident| resident.saturating_add(after)),
                    )
            }
        }
    }
}

/// What a block's queue did, for its `BLOCK GROUP QUEUE` line.
#[derive(Debug, Default)]
pub(crate) struct Stats {
    /// Groups sent, and those that went in behind another.
    groups: AtomicUsize,
    behind: AtomicUsize,
    /// Groups the rule refused at least once.
    refused: AtomicUsize,
    /// The most groups and the most bytes waiting at once.
    most_groups: AtomicUsize,
    most_bytes: AtomicU64,
    /// When the last group went in behind another, in milliseconds since the
    /// queue was made.
    last_behind_ms: AtomicU64,
}

/// The `BLOCK GROUP QUEUE` line of a block: the setting, the rule it ran and
/// what waited.
pub(crate) fn line(setting: Setting, rule: Option<Rule>, stats: &Stats) -> String {
    let gib = |bytes: u64| bytes as f64 / GIB;
    let knob = match setting {
        Setting::None => "none".to_string(),
        Setting::Room => "room".to_string(),
        Setting::Bytes(bytes) => format!("{:.2} GiB", gib(bytes)),
    };
    let ran = match rule {
        None if setting == Setting::None => {
            return format!("BLOCK GROUP QUEUE: {knob} · one closed group waits (the default)");
        }
        None => {
            return format!(
                "BLOCK GROUP QUEUE: {knob} · the block does not prove with no disk: one closed \
                 group waits"
            );
        }
        Some(Rule::Room { room, cap }) => format!(
            "at most {:.2} GiB while VmRSS + waiting + {:.2} stays under the line {:.2} GiB",
            gib(cap),
            gib(room.headroom),
            gib(room.line)
        ),
        Some(Rule::Bytes(_)) => "fixed, the host not read".to_string(),
    };
    let (groups, behind) = (stats.groups.load(Relaxed), stats.behind.load(Relaxed));
    let waited = if behind == 0 {
        "none waited behind another".to_string()
    } else {
        format!(
            "{behind} of {groups} groups waited behind another, the last at {:.2}s",
            stats.last_behind_ms.load(Relaxed) as f64 / 1e3
        )
    };
    format!(
        "BLOCK GROUP QUEUE: {knob} · {ran} · most at once {} groups {:.2} GiB · {waited} · the rule \
         refused {} groups",
        stats.most_groups.load(Relaxed),
        gib(stats.most_bytes.load(Relaxed)),
        stats.refused.load(Relaxed),
    )
}

/// The process's resident bytes, as the rule reads them.
type Resident = Arc<dyn Fn() -> Option<u64> + Send + Sync>;

struct Queue<T> {
    rule: Rule,
    resident: Resident,
    stats: Arc<Stats>,
    made: Instant,
    state: Mutex<State<T>>,
    changed: Condvar,
}

struct State<T> {
    /// The groups waiting, oldest first, each with its bytes.
    waiting: VecDeque<(T, u64)>,
    bytes: u64,
    sender_gone: bool,
    receiver_gone: bool,
}

impl<T> Queue<T> {
    fn state(&self) -> std::sync::MutexGuard<'_, State<T>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A queue under `rule`: its two ends. The rule reads the process's resident
/// set ([`crate::lfm::program_budget::resident_bytes`]).
pub(crate) fn queue<T>(rule: Rule, stats: Arc<Stats>) -> (Sender<T>, Receiver<T>) {
    queue_reading(
        rule,
        stats,
        Arc::new(crate::lfm::program_budget::resident_bytes),
    )
}

fn queue_reading<T>(rule: Rule, stats: Arc<Stats>, resident: Resident) -> (Sender<T>, Receiver<T>) {
    let queue = Arc::new(Queue {
        rule,
        resident,
        stats,
        made: Instant::now(),
        state: Mutex::new(State {
            waiting: VecDeque::new(),
            bytes: 0,
            sender_gone: false,
            receiver_gone: false,
        }),
        changed: Condvar::new(),
    });
    (Sender(Arc::clone(&queue)), Receiver(queue))
}

/// The receiver is gone: nobody takes the groups.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Stopped;

/// The packer's end.
pub(crate) struct Sender<T>(Arc<Queue<T>>);

impl<T> Sender<T> {
    /// Hands a closed group of `bytes` on, behind those waiting. While
    /// another waits and the rule refuses this one, it waits here: each
    /// group taken is another look at the rule.
    fn send(&self, group: T, bytes: u64) -> Result<(), Stopped> {
        let queue = &*self.0;
        let mut state = queue.state();
        let mut refused = false;
        let behind = loop {
            if state.receiver_gone {
                return Err(Stopped);
            }
            let behind = !state.waiting.is_empty();
            if !behind || queue.rule.admits(state.bytes, bytes, || (queue.resident)()) {
                break behind;
            }
            if !refused {
                refused = true;
                queue.stats.refused.fetch_add(1, Relaxed);
            }
            state = queue.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        };
        state.waiting.push_back((group, bytes));
        state.bytes += bytes;
        let stats = &queue.stats;
        stats.groups.fetch_add(1, Relaxed);
        stats.most_groups.fetch_max(state.waiting.len(), Relaxed);
        stats.most_bytes.fetch_max(state.bytes, Relaxed);
        if behind {
            stats.behind.fetch_add(1, Relaxed);
            let ms = u64::try_from(queue.made.elapsed().as_millis()).unwrap_or(u64::MAX);
            stats.last_behind_ms.store(ms, Relaxed);
        }
        drop(state);
        queue.changed.notify_all();
        Ok(())
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        self.0.state().sender_gone = true;
        self.0.changed.notify_all();
    }
}

/// Phase A's end: the groups in the order they were sent, until the sender is
/// gone and nothing waits.
pub(crate) struct Receiver<T>(Arc<Queue<T>>);

impl<T> Iterator for Receiver<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let queue = &*self.0;
        let mut state = queue.state();
        loop {
            if let Some((group, bytes)) = state.waiting.pop_front() {
                state.bytes -= bytes;
                drop(state);
                queue.changed.notify_all();
                return Some(group);
            }
            if state.sender_gone {
                return None;
            }
            state = queue.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.0.state().receiver_gone = true;
        self.0.changed.notify_all();
    }
}

/// A closed group's way to phase A: the channel of one, or the queue.
pub(crate) enum Out<T> {
    Channel(std::sync::mpsc::SyncSender<T>),
    Queue(Sender<T>),
}

impl<T> Out<T> {
    /// Sends `group`, waiting as its way makes it wait; the queue weighs the
    /// group's `bytes`.
    pub(crate) fn send(&self, group: T, bytes: impl FnOnce(&T) -> u64) -> Result<(), Stopped> {
        match self {
            Self::Channel(out) => out.send(group).map_err(|_| Stopped),
            Self::Queue(out) => {
                let bytes = bytes(&group);
                out.send(group, bytes)
            }
        }
    }
}

/// The closed groups as phase A takes them, from either way.
pub(crate) enum In<'c, T> {
    Channel(std::sync::mpsc::Iter<'c, T>),
    Queue(Receiver<T>),
}

impl<T> Iterator for In<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self {
            Self::Channel(groups) => groups.next(),
            Self::Queue(groups) => groups.next(),
        }
    }
}

/// Runs `test` on a thread of its own and fails if it has not ended in a
/// minute: a queue that stops handing groups on hangs, it does not fail.
#[cfg(test)]
pub(crate) fn within_a_minute(what: &str, test: impl FnOnce() + Send + 'static) {
    use std::sync::mpsc::RecvTimeoutError;
    let (done, ended) = std::sync::mpsc::channel();
    let run = std::thread::spawn(move || {
        test();
        let _ = done.send(());
    });
    match ended.recv_timeout(std::time::Duration::from_secs(60)) {
        Ok(()) => run.join().expect("the test"),
        // The test panicked: its own message.
        Err(RecvTimeoutError::Disconnected) => {
            std::panic::resume_unwind(run.join().expect_err("a panic"))
        }
        Err(RecvTimeoutError::Timeout) => panic!("{what}: the queue hung"),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use super::*;

    const G: u64 = 1 << 30;

    fn fixed<T>(budget: u64) -> (Sender<T>, Receiver<T>, Arc<Stats>) {
        let stats = Arc::new(Stats::default());
        let (tx, rx) = queue_reading(Rule::Bytes(budget), Arc::clone(&stats), Arc::new(|| None));
        (tx, rx, stats)
    }

    /// Waits for `done`, at most ten seconds.
    fn until(what: &str, done: impl Fn() -> bool) {
        let t = Instant::now();
        while !done() {
            assert!(t.elapsed() < Duration::from_secs(10), "{what}: never");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// A small deterministic stream of numbers.
    fn stream(seed: u64) -> impl FnMut() -> u64 {
        let mut x = seed;
        move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            x >> 33
        }
    }

    /// Groups leave in the order they entered, whatever waits and however
    /// slowly they are taken.
    #[test]
    fn groups_leave_in_the_order_they_entered() {
        within_a_minute("the order", in_order);
    }

    fn in_order() {
        let (tx, rx, stats) = fixed::<usize>(64);
        let sender = std::thread::spawn(move || {
            let mut bytes = stream(1);
            for group in 0..2000 {
                tx.send(group, 1 + bytes() % 8)
                    .expect("the receiver is there");
            }
        });
        let mut pause = stream(2);
        let mut taken = Vec::new();
        for group in rx {
            taken.push(group);
            if pause().is_multiple_of(16) {
                std::thread::sleep(Duration::from_micros(200));
            }
        }
        sender.join().expect("the sender");
        assert_eq!(taken, (0..2000).collect::<Vec<_>>());
        assert!(
            stats.most_groups.load(Relaxed) >= 3,
            "at most {} groups waited at once: the order was never at stake",
            stats.most_groups.load(Relaxed)
        );
    }

    /// With no bytes to wait in, one group waits and the next stays in the
    /// sender's hands until that one is taken: the channel of one.
    #[test]
    fn no_budget_is_one_group_waiting() {
        let (tx, mut rx, stats) = fixed::<usize>(0);
        let sent = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&sent);
        let sender = std::thread::spawn(move || {
            for group in 0..4 {
                tx.send(group, 100).expect("the receiver is there");
                counted.fetch_add(1, Relaxed);
            }
        });
        for group in 0..4 {
            until("the next group is sent", || sent.load(Relaxed) == group + 1);
            if group < 3 {
                until("the one after it is refused", || {
                    stats.refused.load(Relaxed) == group + 1
                });
            }
            std::thread::sleep(Duration::from_millis(30));
            assert_eq!(sent.load(Relaxed), group + 1, "a second group went in");
            assert_eq!(rx.next(), Some(group));
        }
        sender.join().expect("the sender");
        assert_eq!(rx.next(), None);
        assert_eq!(stats.most_groups.load(Relaxed), 1);
        assert_eq!(stats.behind.load(Relaxed), 0);
        assert_eq!(stats.refused.load(Relaxed), 3);
        let text = line(Setting::Bytes(0), Some(Rule::Bytes(0)), &stats);
        assert!(text.contains("none waited behind another"), "{text}");
    }

    /// Never more than the budget waits, bar a group alone: one larger than
    /// the budget goes when nothing waits, so no budget can stop the block.
    #[test]
    fn at_most_the_budget_waits_and_a_larger_group_still_goes() {
        within_a_minute("the budget", under_a_budget);
    }

    fn under_a_budget() {
        let budget = 1000;
        let (tx, rx, stats) = fixed::<u64>(budget);
        let sizes: Vec<u64> = {
            let mut next = stream(3);
            (0..600)
                .map(|k| if k % 97 == 50 { 5000 } else { 1 + next() % 400 })
                .collect()
        };
        let sending = sizes.clone();
        let sender = std::thread::spawn(move || {
            for bytes in sending {
                tx.send(bytes, bytes).expect("the receiver is there");
            }
        });
        // Each group as it is taken, with what still waits behind it.
        let queue = Arc::clone(&rx.0);
        let mut taken = Vec::new();
        let mut pause = stream(4);
        for bytes in rx {
            let (waiting, groups) = {
                let state = queue.state();
                (state.bytes, state.waiting.len())
            };
            assert!(
                waiting <= budget || groups == 1,
                "{groups} groups of {waiting} bytes wait, the budget is {budget}"
            );
            taken.push(bytes);
            if pause().is_multiple_of(4) {
                std::thread::sleep(Duration::from_micros(300));
            }
        }
        sender.join().expect("the sender");
        assert_eq!(taken, sizes);
        let most = stats.most_bytes.load(Relaxed);
        assert_eq!(
            most, 5000,
            "the most bytes at once is the large group alone"
        );
        assert!(
            stats.behind.load(Relaxed) > 0,
            "nothing waited behind another"
        );
        assert!(stats.refused.load(Relaxed) > 0, "the budget never refused");
    }

    /// A receiver that stops wakes a sender waiting on the rule, with an
    /// error; a sender that stops ends the receiver once nothing waits.
    #[test]
    fn either_end_stopping_ends_the_other() {
        let (tx, rx, _) = fixed::<usize>(0);
        tx.send(0, 1).expect("the first goes");
        let stopped = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&stopped);
        let sender = std::thread::spawn(move || {
            let sent = tx.send(1, 1);
            seen.store(true, Relaxed);
            sent
        });
        std::thread::sleep(Duration::from_millis(30));
        assert!(
            !stopped.load(Relaxed),
            "the second went in behind the first"
        );
        drop(rx);
        until("the sender wakes", || stopped.load(Relaxed));
        assert_eq!(sender.join().expect("the sender"), Err(Stopped));

        let (tx, rx, _) = fixed::<usize>(u64::MAX);
        tx.send(7, 1).expect("the receiver is there");
        tx.send(8, 1).expect("the receiver is there");
        drop(tx);
        assert_eq!(rx.collect::<Vec<_>>(), [7, 8]);
    }

    fn room(line_gib: f64, cap_gib: u64) -> Rule {
        Rule::Room {
            room: NoDisk {
                line: (line_gib * GIB) as u64,
                headroom: 16 * G,
            },
            cap: cap_gib * G,
        }
    }

    /// The room rule on the readings of the large block (a group of 0.65
    /// GiB): a 118 GiB host, line 92.03 GiB, has room at its highest resident
    /// set of the commit phase until about nine GiB wait; a 64 GiB host, line
    /// 45.90 GiB, has none from 29.25 GiB of resident set on, which it passes
    /// nine seconds in. The waiting bytes count beside the resident set, and
    /// the cap holds whatever the host reads.
    #[test]
    fn the_room_rule_at_the_measured_readings() {
        let gib = |g: f64| (g * GIB) as u64;
        let group = gib(0.65);
        let admits = |rule: Rule, waiting: f64, resident: f64| {
            rule.admits(gib(waiting), group, || Some(gib(resident)))
        };
        let large = room(92.03, 16);
        assert!(admits(large, 0.65, 66.34));
        assert!(admits(large, 8.9, 66.34));
        assert!(!admits(large, 9.1, 66.34));
        assert!(admits(large, 5.3, 70.0));
        assert!(!admits(large, 5.5, 70.0));
        // The cap, on a host with all the room.
        assert!(admits(large, 15.3, 10.0));
        assert!(!admits(large, 15.4, 10.0));
        // Room is strictly under the line.
        let line = Rule::Room {
            room: NoDisk {
                line: 40 * G,
                headroom: 16 * G,
            },
            cap: 16 * G,
        };
        assert!(!line.admits(4 * G, G, || Some(19 * G)));
        assert!(line.admits(4 * G, G, || Some(19 * G - 1)));
        let small = room(45.90, 16);
        assert!(admits(small, 0.65, 25.62));
        assert!(!admits(small, 0.65, 29.34));
        assert!(!admits(small, 0.0, 36.08));
        // No reading is no room.
        assert!(!large.admits(0, group, || None));
        // A fixed budget does not read the host.
        assert!(Rule::Bytes(G).admits(0, G, || unreachable!("the host is not read")));
        assert!(!Rule::Bytes(G).admits(1, G, || unreachable!("the host is not read")));
    }

    /// The room rule in the queue: groups wait behind one another while the
    /// resident set leaves room, and one at a time once it does not.
    #[test]
    fn the_queue_follows_the_resident_set() {
        let resident = Arc::new(AtomicU64::new(10 * G));
        let reading = Arc::clone(&resident);
        let stats = Arc::new(Stats::default());
        let (tx, mut rx) = queue_reading::<usize>(
            room(45.90, 16),
            Arc::clone(&stats),
            Arc::new(move || Some(reading.load(Relaxed))),
        );
        let sent = Arc::new(AtomicUsize::new(0));
        let go = Arc::new(AtomicBool::new(false));
        let (counted, gone) = (Arc::clone(&sent), Arc::clone(&go));
        let sender = std::thread::spawn(move || {
            for group in 0..6 {
                if group == 3 {
                    while !gone.load(Relaxed) {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
                tx.send(group, G / 2).expect("the receiver is there");
                counted.fetch_add(1, Relaxed);
            }
        });
        // Room: three go in behind one another, none refused.
        until("three are sent", || sent.load(Relaxed) == 3);
        assert_eq!(stats.behind.load(Relaxed), 2);
        assert_eq!(stats.refused.load(Relaxed), 0);
        // No room from here: the fourth waits in the sender's hands until
        // nothing waits, as with the channel of one, and so does each after.
        resident.store(36 * G, Relaxed);
        go.store(true, Relaxed);
        until("the fourth is refused", || stats.refused.load(Relaxed) == 1);
        for group in 0..3 {
            std::thread::sleep(Duration::from_millis(10));
            assert_eq!(sent.load(Relaxed), 3, "a group went in with no room");
            assert_eq!(rx.next(), Some(group));
        }
        for group in 3..6 {
            until("the next is sent", || sent.load(Relaxed) == group + 1);
            if group < 5 {
                until("the one after it is refused", || {
                    stats.refused.load(Relaxed) == group - 1
                });
            }
            std::thread::sleep(Duration::from_millis(10));
            assert_eq!(
                sent.load(Relaxed),
                group + 1,
                "a second went in with no room"
            );
            assert_eq!(rx.next(), Some(group));
        }
        sender.join().expect("the sender");
        assert_eq!(rx.next(), None);
        assert_eq!(stats.behind.load(Relaxed), 2);
        assert_eq!(stats.refused.load(Relaxed), 3);
        assert_eq!(stats.most_groups.load(Relaxed), 3);
    }

    /// The knob: unset, empty and `none` are the channel of one; `room` and a
    /// number of GiB are read; anything else refuses the prove.
    #[test]
    fn the_knob_is_none_by_default_and_refuses_what_it_does_not_know() {
        assert_eq!(parse_setting(None).unwrap(), Setting::None);
        assert_eq!(parse_setting(Some("")).unwrap(), Setting::None);
        assert_eq!(parse_setting(Some("none")).unwrap(), Setting::None);
        assert_eq!(parse_setting(Some(" room ")).unwrap(), Setting::Room);
        assert_eq!(parse_setting(Some("16")).unwrap(), Setting::Bytes(16 * G));
        assert_eq!(parse_setting(Some("0.5")).unwrap(), Setting::Bytes(G / 2));
        for bad in ["on", "auto", "0", "-1", "inf", "nan", "room,16", "1e400"] {
            assert!(parse_setting(Some(bad)).is_err(), "{bad} was taken");
        }
    }

    /// The rule a block runs: none by default, none under any setting for a
    /// block that does not prove with no disk, the purge rule's line and
    /// headroom under `room`.
    #[test]
    fn only_a_no_disk_block_runs_a_rule() {
        let no_disk = NoDisk {
            line: 92 * G,
            headroom: 16 * G,
        };
        for setting in [Setting::None, Setting::Room, Setting::Bytes(G)] {
            assert_eq!(Rule::of(setting, None, 16 * G), None, "{setting:?}");
        }
        assert_eq!(Rule::of(Setting::None, Some(no_disk), 16 * G), None);
        assert_eq!(
            Rule::of(Setting::Room, Some(no_disk), 16 * G),
            Some(Rule::Room {
                room: no_disk,
                cap: 16 * G
            })
        );
        assert_eq!(
            Rule::of(Setting::Bytes(G), Some(no_disk), 16 * G),
            Some(Rule::Bytes(G))
        );
    }

    /// The line a block prints.
    #[test]
    fn the_line_says_what_ran_and_what_waited() {
        let stats = Stats::default();
        assert_eq!(
            line(Setting::None, None, &stats),
            "BLOCK GROUP QUEUE: none · one closed group waits (the default)"
        );
        assert_eq!(
            line(Setting::Room, None, &stats),
            "BLOCK GROUP QUEUE: room · the block does not prove with no disk: one closed group waits"
        );
        stats.groups.store(182, Relaxed);
        stats.most_groups.store(1, Relaxed);
        stats.most_bytes.store(G / 2, Relaxed);
        stats.refused.store(181, Relaxed);
        assert_eq!(
            line(Setting::Room, Some(room(45.90, 16)), &stats),
            "BLOCK GROUP QUEUE: room · at most 16.00 GiB while VmRSS + waiting + 16.00 stays under the \
             line 45.90 GiB · most at once 1 groups 0.50 GiB · none waited behind another · the rule \
             refused 181 groups"
        );
        stats.behind.store(96, Relaxed);
        stats.most_groups.store(13, Relaxed);
        stats.most_bytes.store(8 * G, Relaxed);
        stats.refused.store(41, Relaxed);
        stats.last_behind_ms.store(58_420, Relaxed);
        assert_eq!(
            line(Setting::Bytes(8 * G), Some(Rule::Bytes(8 * G)), &stats),
            "BLOCK GROUP QUEUE: 8.00 GiB · fixed, the host not read · most at once 13 groups 8.00 GiB · \
             96 of 182 groups waited behind another, the last at 58.42s · the rule refused 41 groups"
        );
    }

    /// The channel of one and the queue hand the same groups on in the same
    /// order through [`Out`] and [`In`]; only the queue weighs their bytes.
    #[test]
    fn both_ways_hand_the_same_groups_on() {
        within_a_minute("both ways", both_ways);
    }

    fn both_ways() {
        let groups: Vec<Vec<u8>> = (0..50u8).map(|k| vec![k; usize::from(k % 7)]).collect();
        let through = |out: Out<Vec<u8>>, weigh: bool| {
            let sending = groups.clone();
            std::thread::spawn(move || {
                for group in sending {
                    out.send(group, |group| {
                        assert!(weigh, "the channel weighed a group");
                        group.len() as u64
                    })
                    .expect("the receiver is there");
                }
            })
        };
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let sender = through(Out::Channel(tx), false);
        assert_eq!(In::Channel(rx.iter()).collect::<Vec<_>>(), groups);
        sender.join().expect("the sender");
        for budget in [0, 1, u64::MAX] {
            let (tx, rx, _) = fixed(budget);
            let sender = through(Out::Queue(tx), true);
            assert_eq!(In::Queue(rx).collect::<Vec<_>>(), groups, "budget {budget}");
            sender.join().expect("the sender");
        }
    }
}

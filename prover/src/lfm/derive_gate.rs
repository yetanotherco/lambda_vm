//! The block tree derivation's device running total.
//!
//! The block verifier derives every program of the tree with no proof read
//! ([`super::block_plan::BlockTreePlan::derive_top`]), and a level's artifact
//! builds run in parallel. Each build commits its column groups on the card
//! (`commit::commit_group_device_or_host_with`), and with the shared VRAM gate
//! unarmed — as it is after a prove — each commit was checked only alone,
//! against the whole card. A level of 32 builds, each with up to eleven groups
//! in flight, then asked the card for more than it has: BIG 123's p90 block
//! (110 leaves) took the card from 11.5 to 32 GiB in 2.5 s and aborted.
//!
//! So the derivation arms a [`ByteGate`] ([`arm`]) and each commit takes its
//! device set's bytes from it ([`admit`]) while the shared gate gives no
//! permit: a running total in bytes, not a count of builds.
//!
//! ⛔ **No permit is held across a rayon wait.** A commit holds its bytes only
//! around its device dispatch, which forks nothing. A holder that waited inside
//! rayon — a `join` whose other half was stolen, a parallel iterator — would
//! run the pool's queued jobs on its own thread meanwhile, and one of them that
//! asks this gate again parks the thread with the bytes it holds (the count
//! gate's hazard, and #1014's W3 hang on BIG 569). Every fork of a build
//! asserts it ([`assert_none_held`], at `registry::map_maybe_parallel`), and
//! so does [`ByteGate::admit`]. A waiter is a rayon worker parked on a condvar:
//! it steals nothing, and the holders it waits for need no rayon work to
//! finish.
//!
//! `LAMBDA_VM_BLOCK_DERIVE_GATE`: unset or `auto` (the default) calibrates the
//! budget at the arming, `off` keeps no total (each commit checked alone, as
//! before), `<GiB>` fixes the budget. Scheduling only: every artifact is a pure
//! function of its program and the options, so the derived tree and its top do
//! not depend on it.

// The gate's only dispatches are device commits: without `cuda`, only its tests
// and the fork's assert reach it.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

const GIB: f64 = (1u64 << 30) as f64;

/// Bytes kept under the card's free memory at the arming: the shared gate's
/// base margin (`device_permit::shared_margin_bytes` with no prove in flight),
/// for what the gate cannot see — device caches, compiled modules, each set's
/// allocations beyond its estimate.
const MARGIN_BYTES: u64 = 2 << 30;

/// [`LAMBDA_VM_BLOCK_DERIVE_GATE`](self)'s values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Setting {
    /// The budget is the card's free bytes at the arming, after a drain and a
    /// pool trim, less [`MARGIN_BYTES`], capped at the configured budget.
    Auto,
    /// No running total: each commit is checked alone, as before the gate.
    Off,
    /// A fixed budget in bytes.
    Fixed(u64),
}

/// [`Setting`] for a raw knob value.
fn setting_of(v: Option<&str>) -> Setting {
    match v.map(str::trim) {
        None | Some("") | Some("auto") => Setting::Auto,
        Some("off") => Setting::Off,
        Some(s) => match s.parse::<f64>() {
            Ok(g) if g.is_finite() && g > 0.0 => Setting::Fixed((g * GIB) as u64),
            _ => panic!(
                "LAMBDA_VM_BLOCK_DERIVE_GATE must be `auto`, `off` or a budget in GiB, got `{s}`"
            ),
        },
    }
}

fn setting() -> Setting {
    #[cfg(test)]
    if let Some(s) = *PIN.lock().unwrap_or_else(|e| e.into_inner()) {
        return s;
    }
    static SETTING: std::sync::OnceLock<Setting> = std::sync::OnceLock::new();
    *SETTING
        .get_or_init(|| setting_of(std::env::var("LAMBDA_VM_BLOCK_DERIVE_GATE").ok().as_deref()))
}

/// Test-only: a [`Setting`] for this process in place of the knob (`None`
/// reads the knob again).
#[cfg(test)]
static PIN: Mutex<Option<Setting>> = Mutex::new(None);

#[cfg(test)]
pub(crate) fn pin_setting(s: Option<Setting>) {
    *PIN.lock().unwrap_or_else(|e| e.into_inner()) = s;
}

thread_local! {
    /// [`BytePermit`]s held by this thread.
    static HELD: Cell<usize> = const { Cell::new(0) };
}

/// Panic if this thread holds a derive permit at `site`, a rayon fork: a job
/// the thread runs while it waits there could ask the gate again and park the
/// thread with the bytes it holds.
pub(crate) fn assert_none_held(site: &str) {
    assert_eq!(
        HELD.with(Cell::get),
        0,
        "a derive permit is held across a rayon fork ({site}): a job this thread runs while it \
         waits there can ask the gate again and park it; take the permit around the device \
         dispatch alone"
    );
}

/// Whether a set of `bytes` enters beside `used` held under `budget`: alone,
/// any set does, so a set over the budget runs by itself rather than never.
fn fits(used: u64, bytes: u64, budget: u64) -> bool {
    used == 0 || used.saturating_add(bytes) <= budget
}

/// A running total of device bytes under a budget.
pub(crate) struct ByteGate {
    used: Mutex<u64>,
    freed: Condvar,
    budget: u64,
    peak: AtomicU64,
    dispatches: AtomicUsize,
    waits: AtomicUsize,
    waited_nanos: AtomicU64,
}

/// Bytes of a [`ByteGate`], given back on drop.
pub(crate) struct BytePermit {
    gate: Arc<ByteGate>,
    bytes: u64,
}

impl ByteGate {
    pub(crate) fn new(budget: u64) -> Self {
        Self {
            used: Mutex::new(0),
            freed: Condvar::new(),
            budget,
            peak: AtomicU64::new(0),
            dispatches: AtomicUsize::new(0),
            waits: AtomicUsize::new(0),
            waited_nanos: AtomicU64::new(0),
        }
    }

    /// Take `bytes`, waiting until they [`fits`] beside what is held.
    pub(crate) fn admit(self: &Arc<Self>, bytes: u64) -> BytePermit {
        assert_eq!(
            HELD.with(Cell::get),
            0,
            "this thread already holds a derive permit; asking again could park it forever \
             with the bytes it holds"
        );
        let mut used = self.used.lock().unwrap_or_else(|e| e.into_inner());
        let mut since = None;
        while !fits(*used, bytes, self.budget) {
            since.get_or_insert_with(Instant::now);
            used = self.freed.wait(used).unwrap_or_else(|e| e.into_inner());
        }
        *used = used.saturating_add(bytes);
        self.peak.fetch_max(*used, Ordering::Relaxed);
        drop(used);
        self.dispatches.fetch_add(1, Ordering::Relaxed);
        if let Some(t) = since {
            self.waits.fetch_add(1, Ordering::Relaxed);
            self.waited_nanos
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        HELD.with(|h| h.set(h.get() + 1));
        BytePermit {
            gate: Arc::clone(self),
            bytes,
        }
    }

    /// The most held at once so far.
    pub(crate) fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    fn summary(&self) -> Summary {
        Summary {
            budget: self.budget,
            peak: self.peak(),
            dispatches: self.dispatches.load(Ordering::Relaxed),
            waits: self.waits.load(Ordering::Relaxed),
            waited_secs: self.waited_nanos.load(Ordering::Relaxed) as f64 * 1e-9,
        }
    }
}

impl Drop for BytePermit {
    fn drop(&mut self) {
        let mut used = self.gate.used.lock().unwrap_or_else(|e| e.into_inner());
        *used = used.saturating_sub(self.bytes);
        drop(used);
        // All: the waiters ask for different sizes, and the one woken alone
        // might not fit while another would.
        self.gate.freed.notify_all();
        HELD.with(|h| h.set(h.get().saturating_sub(1)));
    }
}

/// One armed stretch of the gate, as its last disarm reports it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Summary {
    pub budget: u64,
    pub peak: u64,
    pub dispatches: usize,
    pub waits: usize,
    pub waited_secs: f64,
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "peak {:.2} GiB of a {:.2} GiB budget · {} device sets · {} waited (Σ {:.2}s)",
            self.peak as f64 / GIB,
            self.budget as f64 / GIB,
            self.dispatches,
            self.waits,
            self.waited_secs
        )
    }
}

/// The gate while armed, and how many arming guards hold it.
static ARMED: Mutex<Option<(Arc<ByteGate>, usize)>> = Mutex::new(None);

/// The last armed stretch's [`Summary`], once its last guard dropped.
static LAST: Mutex<Option<Summary>> = Mutex::new(None);

/// The gate armed for a derivation, disarmed when the last guard drops.
pub(crate) struct DeriveGate(());

/// Arm the gate for a derivation, or join the derivation that armed it. `None`
/// with the knob off, with the shared VRAM gate armed (its own total then
/// covers each commit) or without a device. The first arming calibrates the
/// budget ([`stark::prover::arm_running_total`]): it drains the device, so
/// arm only where no other device work is in flight — the block verifier,
/// never a derivation beside a prove.
pub(crate) fn arm() -> Option<DeriveGate> {
    if !cfg!(feature = "cuda") || stark::prover::shared_vram_gate_on() {
        return None;
    }
    let fixed = match setting() {
        Setting::Off => return None,
        Setting::Auto => None,
        Setting::Fixed(b) => Some(b),
    };
    let mut armed = ARMED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, guards)) = armed.as_mut() {
        *guards += 1;
        return Some(DeriveGate(()));
    }
    let (calibrated, free, releasing) = stark::prover::arm_running_total(MARGIN_BYTES);
    let budget = fixed.unwrap_or(calibrated);
    eprintln!(
        "[prover] block derive gate armed: budget {:.2} GiB ({}; card free {} after a drain and a \
         pool trim, margin {:.2} GiB){}",
        budget as f64 / GIB,
        if fixed.is_some() {
            "fixed by LAMBDA_VM_BLOCK_DERIVE_GATE"
        } else {
            "calibrated"
        },
        free.map_or("unknown".to_string(), |f| format!(
            "{:.2} GiB",
            f as f64 / GIB
        )),
        MARGIN_BYTES as f64 / GIB,
        if releasing {
            "; the pool releases at each sync while armed"
        } else {
            ""
        },
    );
    *armed = Some((Arc::new(ByteGate::new(budget)), 1));
    Some(DeriveGate(()))
}

impl Drop for DeriveGate {
    fn drop(&mut self) {
        let mut armed = ARMED.lock().unwrap_or_else(|e| e.into_inner());
        let last = match armed.as_mut() {
            Some((_, guards)) if *guards > 1 => {
                *guards -= 1;
                false
            }
            _ => true,
        };
        if last && let Some((gate, _)) = armed.take() {
            drop(armed);
            stark::prover::disarm_running_total();
            let summary = gate.summary();
            eprintln!("[prover] block derive gate disarmed: {summary}");
            *LAST.lock().unwrap_or_else(|e| e.into_inner()) = Some(summary);
        }
    }
}

/// Take `bytes` of the armed gate for one device dispatch; `None` while it is
/// not armed. Hold the permit around the dispatch alone ([`assert_none_held`]).
pub(crate) fn admit(bytes: u64) -> Option<BytePermit> {
    let gate = ARMED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|(gate, _)| Arc::clone(gate))?;
    Some(gate.admit(bytes))
}

/// The last armed stretch's [`Summary`], taken: a second call reads `None`
/// until the gate is armed and disarmed again.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn take_summary() -> Option<Summary> {
    LAST.lock().unwrap_or_else(|e| e.into_inner()).take()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_knob_reads_auto_off_or_a_budget_in_gib() {
        assert_eq!(setting_of(None), Setting::Auto);
        assert_eq!(setting_of(Some("")), Setting::Auto);
        assert_eq!(setting_of(Some("auto")), Setting::Auto);
        assert_eq!(setting_of(Some("off")), Setting::Off);
        assert_eq!(setting_of(Some("8")), Setting::Fixed(8 << 30));
        assert_eq!(setting_of(Some(" 0.5 ")), Setting::Fixed(1 << 29));
        for bad in ["0", "-1", "on", "nan"] {
            assert!(
                std::panic::catch_unwind(|| setting_of(Some(bad))).is_err(),
                "`{bad}` is refused"
            );
        }
    }

    /// A set over the budget enters alone, and only alone.
    #[test]
    fn a_set_over_the_budget_runs_alone() {
        assert!(fits(0, 10, 4));
        assert!(!fits(1, 10, 4));
        assert!(fits(3, 1, 4));
        assert!(!fits(3, 2, 4));
        // The open gate (a budget of `u64::MAX`, the tests' control) admits all.
        assert!(fits(u64::MAX - 1, 10, u64::MAX));
    }

    #[test]
    fn a_thread_asking_twice_is_refused() {
        let gate = Arc::new(ByteGate::new(10));
        let held = gate.admit(1);
        let again = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gate.admit(1)));
        assert!(again.is_err(), "a second permit on one thread is refused");
        drop(held);
        drop(gate.admit(1));
        assert_eq!(HELD.with(Cell::get), 0);
    }

    /// One synthetic leaf's device sets, in `BuildPlan::walk`'s order: eleven
    /// row-pair groups, then the same eleven one-row.
    struct Leaf {
        row_pair: Vec<u64>,
        one_row: Vec<u64>,
    }

    /// Where a synthetic derivation takes its permits.
    #[derive(Clone, Copy, PartialEq)]
    enum Hold {
        /// Around each device set's dispatch (the gate's placement).
        Dispatch,
        /// ⛔ Around a leaf's whole walk (the count gate's placement), across
        /// the walk's forks: the mutation.
        Walk,
    }

    const UNIT: u64 = 1 << 20;

    /// 64 leaves, sets of 1 to 4 units from a fixed LCG.
    fn leaves() -> Vec<Leaf> {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut size = || {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (1 + (x >> 33) % 4) * UNIT
        };
        (0..64)
            .map(|_| {
                let row_pair: Vec<u64> = (0..11).map(|_| size()).collect();
                let one_row = row_pair.iter().map(|b| b / 2 + UNIT / 2).collect();
                Leaf { row_pair, one_row }
            })
            .collect()
    }

    /// A level's derivation shaped like the verifier's: the leaves in
    /// parallel, each walking its row-pair sets in windows of four and its
    /// one-row sets whole, through the build's own fork
    /// (`registry::map_maybe_parallel`); each dispatch holds its set for 2 ms.
    fn derive(gate: &Arc<ByteGate>, leaves: &[Leaf], hold: Hold) {
        use rayon::prelude::*;
        let dispatch = |&bytes: &u64| {
            let _p = (hold == Hold::Dispatch).then(|| gate.admit(bytes));
            std::thread::sleep(std::time::Duration::from_millis(2));
        };
        leaves.par_iter().for_each(|leaf| {
            let _whole = (hold == Hold::Walk).then(|| gate.admit(leaf.row_pair.iter().sum()));
            for window in leaf.row_pair.chunks(4) {
                super::super::registry::map_maybe_parallel(window, dispatch);
            }
            super::super::registry::map_maybe_parallel(&leaf.one_row, dispatch);
        });
    }

    fn pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("an 8-thread pool")
    }

    /// ★ A many-leaf derivation stays under its byte budget and keeps its
    /// parallelism: the gate's peak is at most the budget and above the
    /// largest set (more than one in flight), while the same derivation with
    /// no total (the control) peaks over the budget, so the bound binds.
    #[cfg(feature = "parallel")]
    #[test]
    fn a_many_leaf_derivation_stays_under_its_byte_budget() {
        let leaves = leaves();
        let largest = leaves
            .iter()
            .flat_map(|l| l.row_pair.iter().chain(&l.one_row))
            .copied()
            .max()
            .expect("sets");
        let budget = 10 * UNIT;
        let pool = pool();
        let open = Arc::new(ByteGate::new(u64::MAX));
        pool.install(|| derive(&open, &leaves, Hold::Dispatch));
        assert!(
            open.peak() > budget,
            "the control holds {} units at once, not over the {}-unit budget: the test would be \
             vacuous",
            open.peak() / UNIT,
            budget / UNIT
        );
        let gated = Arc::new(ByteGate::new(budget));
        pool.install(|| derive(&gated, &leaves, Hold::Dispatch));
        let s = gated.summary();
        assert!(s.peak <= budget, "the gate held {s}");
        assert!(s.peak > largest, "one set at a time: {s}");
        assert_eq!(s.dispatches, leaves.len() * 22);
        assert!(s.waits > 0, "nothing waited: {s}");
    }

    /// The build's fork refuses a thread that holds a permit, before it forks.
    #[cfg(feature = "parallel")]
    #[test]
    fn the_builds_fork_refuses_a_held_permit() {
        let gate = Arc::new(ByteGate::new(10));
        let fork = |held: bool| {
            let _p = held.then(|| gate.admit(1));
            std::panic::catch_unwind(|| {
                super::super::registry::map_maybe_parallel(&[1, 2, 3], |x| x + 1)
            })
        };
        assert_eq!(fork(false).ok(), Some(vec![2, 3, 4]));
        let refused = fork(true).expect_err("a fork under a held permit is refused");
        let msg = refused
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default();
        assert!(
            msg.contains("held across a rayon fork"),
            "refused for: {msg}"
        );
    }

    /// ⛔ The mutation: a permit held across the walk's forks (the count gate's
    /// placement) is refused, at the fork's assert or, when the holder ran a
    /// stolen job that asked again first, at the gate's.
    #[cfg(feature = "parallel")]
    #[test]
    fn a_permit_held_across_a_rayon_fork_is_refused() {
        let leaves = leaves();
        let gate = Arc::new(ByteGate::new(u64::MAX));
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool().install(|| derive(&gate, &leaves, Hold::Walk))
        }));
        let msg = match refused {
            Ok(()) => panic!("a permit held across the walk's forks went unnoticed"),
            Err(e) => e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default(),
        };
        assert!(
            msg.contains("held across a rayon fork")
                || msg.contains("already holds a derive permit"),
            "refused for: {msg}"
        );
        assert_eq!(*gate.used.lock().unwrap(), 0, "every permit was given back");
    }
}

//! Where the R2 composition window's host time goes (`LAMBDA_VM_R2_SPLIT=1`).
//!
//! The prover holds one process-wide lock over each table's R2 window (the
//! device composition and its decomposition, `stark::gpu_lde::r2_serialize_guard`).
//! This module accumulates, per thread, the host time spent inside that window
//! by kind, so a readout can tell host work from driver calls and from waits
//! on the device:
//!
//! - [`Cat::Prep`]: host computation before the device calls (zerofiers,
//!   lowering lookups, uniform tables, packing, twiddle tables);
//! - [`Cat::Upload`]: host→device copies of the per-call inputs (with their
//!   bytes: a pageable copy can block on its stream, and its size tells such a
//!   wait from a transfer);
//! - [`Cat::Alloc`]: device allocations, zero fills and device→device copies;
//! - [`Cat::Launch`]: kernel launches and the stream and event calls around them;
//! - [`Cat::Stage`]: taking a pinned staging slot and queueing a drain into it;
//! - [`Cat::DevWait`]: host waits on the device (a drain's event);
//! - [`Cat::Drain`]: host work on drained data (output buffers, unpacking,
//!   conversion to field elements).
//!
//! Regions nest: a region's time excludes the regions timed inside it, so the
//! categories add up to the time the regions cover, once each.
//!
//! Off by default: every hook is then one load of a cached flag, no clock is
//! read, and nothing the prover computes changes.

use std::cell::Cell;
use std::io::Write;
use std::sync::OnceLock;
use std::time::Instant;

/// A kind of host time inside the R2 window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cat {
    Prep = 0,
    Upload = 1,
    Alloc = 2,
    Launch = 3,
    Stage = 4,
    DevWait = 5,
    Drain = 6,
}

const CATS: usize = 7;

thread_local! {
    static NANOS: [Cell<u64>; CATS] = const { [const { Cell::new(0) }; CATS] };
    static UPLOAD_BYTES: Cell<u64> = const { Cell::new(0) };
}

/// Whether `LAMBDA_VM_R2_SPLIT=1` is set (read once).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LAMBDA_VM_R2_SPLIT").is_ok_and(|v| v.trim() == "1"))
}

fn charged_ns() -> u64 {
    NANOS.with(|n| n.iter().map(Cell::get).sum())
}

/// The start of a timed region (see [`start`]).
#[derive(Clone, Copy, Debug)]
pub struct Mark {
    at: Instant,
    /// What this thread had charged when the region opened, so the regions
    /// timed inside it can be taken out of its own time.
    charged: u64,
}

/// A start mark for [`charge`]: `None`, and no clock read, when the split is off.
pub fn start() -> Option<Mark> {
    enabled().then(|| Mark {
        at: Instant::now(),
        charged: charged_ns(),
    })
}

/// Charge the time since `mark` (from [`start`]) to `cat`, less what the
/// regions inside it were charged.
pub fn charge(cat: Cat, mark: Option<Mark>) {
    if let Some(m) = mark {
        let wall = m.at.elapsed().as_nanos() as u64;
        let inner = charged_ns().saturating_sub(m.charged);
        let own = wall.saturating_sub(inner);
        NANOS.with(|n| n[cat as usize].set(n[cat as usize].get() + own));
    }
}

/// Run `f`, charging its time to `cat` when the split is on.
pub fn timed<T>(cat: Cat, f: impl FnOnce() -> T) -> T {
    let mark = start();
    let out = f();
    charge(cat, mark);
    out
}

/// Count `bytes` uploaded (beside [`Cat::Upload`]'s time).
pub fn add_upload_bytes(bytes: usize) {
    if enabled() {
        UPLOAD_BYTES.with(|b| b.set(b.get() + bytes as u64));
    }
}

/// This thread's accumulated split since the last [`take`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Split {
    /// Nanoseconds per [`Cat`], in its order.
    pub nanos: [u64; CATS],
    pub upload_bytes: u64,
}

impl Split {
    /// Nanoseconds charged to `cat`.
    pub fn ns(&self, cat: Cat) -> u64 {
        self.nanos[cat as usize]
    }

    /// Every category's nanoseconds together.
    pub fn total_ns(&self) -> u64 {
        self.nanos.iter().sum()
    }
}

/// Return this thread's split and reset it.
pub fn take() -> Split {
    let mut s = Split::default();
    NANOS.with(|n| {
        for (i, c) in n.iter().enumerate() {
            s.nanos[i] = c.replace(0);
        }
    });
    s.upload_bytes = UPLOAD_BYTES.with(|b| b.replace(0));
    s
}

/// One R2 window's readout, opened when its lock is held.
pub struct Window {
    unix: f64,
    acquired: Instant,
    wait_ns: u64,
}

/// Open a window's readout right after its lock is taken; `asked` is the mark
/// from just before the lock was requested. Drops whatever this thread
/// accumulated outside a window.
pub fn open(asked: Option<Mark>) -> Option<Window> {
    let asked = asked?;
    let acquired = Instant::now();
    take();
    Some(Window {
        unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or_default(),
        acquired,
        wait_ns: acquired.duration_since(asked.at).as_nanos() as u64,
    })
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

impl Window {
    /// Close the window (before its lock is released) and print one
    /// `R2 WINDOW` line on stderr: the wait for the lock, the time it was held
    /// and the held time by category; `unattr` is the held time no category
    /// covers. One write, so a line from another thread cannot split it.
    pub fn close(self, table: &str, rows: usize, parts: usize, locked: bool) {
        let hold = self.acquired.elapsed().as_nanos() as u64;
        let s = take();
        let line = format!(
            "R2 WINDOW table={table} rows={rows} parts={parts} lock={} t={:.3} wait={:.3}ms \
             hold={:.3}ms prep={:.3}ms upload={:.3}ms upload_kib={} alloc={:.3}ms \
             launch={:.3}ms stage={:.3}ms devwait={:.3}ms drain={:.3}ms unattr={:.3}ms\n",
            u8::from(locked),
            self.unix,
            ms(self.wait_ns),
            ms(hold),
            ms(s.ns(Cat::Prep)),
            ms(s.ns(Cat::Upload)),
            s.upload_bytes / 1024,
            ms(s.ns(Cat::Alloc)),
            ms(s.ns(Cat::Launch)),
            ms(s.ns(Cat::Stage)),
            ms(s.ns(Cat::DevWait)),
            ms(s.ns(Cat::Drain)),
            ms(hold.saturating_sub(s.total_ns())),
        );
        let _ = std::io::stderr().write_all(line.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With the knob off (the test process sets none), nothing accumulates,
    /// no window opens, and `timed` still returns its closure's value.
    #[test]
    fn off_by_default_and_transparent() {
        assert!(!enabled());
        assert_eq!(timed(Cat::Prep, || 41 + 1), 42);
        let mark = start();
        assert!(mark.is_none());
        charge(Cat::Launch, mark);
        add_upload_bytes(1024);
        assert!(open(start()).is_none());
        assert_eq!(take(), Split::default());
    }

    /// A region's time excludes the regions timed inside it.
    #[test]
    fn nested_regions_are_charged_once() {
        // Marks built by hand: the knob is off in tests.
        take();
        let outer = Mark {
            at: Instant::now(),
            charged: charged_ns(),
        };
        let inner = Mark {
            at: Instant::now(),
            charged: charged_ns(),
        };
        std::thread::sleep(std::time::Duration::from_millis(20));
        charge(Cat::Upload, Some(inner));
        std::thread::sleep(std::time::Duration::from_millis(20));
        charge(Cat::Prep, Some(outer));
        let wall = outer.at.elapsed().as_nanos() as u64;
        let s = take();
        assert!(s.ns(Cat::Upload) >= 20_000_000);
        assert!(s.ns(Cat::Prep) >= 20_000_000);
        assert!(s.total_ns() <= wall);
    }
}

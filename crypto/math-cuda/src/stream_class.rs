//! Two classes of device work, so a proof run beside another can yield the card.
//!
//! Every caller is in the foreground unless its thread says otherwise
//! ([`enter`]). With `LAMBDA_VM_STREAM_CLASSES` unset (or `off`) the class is
//! read by nobody: one pool of streams serves every thread, as it always did.
//! With `priority`, the foreground pool is created at the device's greatest
//! stream priority and a background pool of [`BACKGROUND_STREAMS`] at its
//! least, so the card's block scheduler starts a foreground kernel's blocks
//! before a background kernel's pending ones; with `equal`, the background pool
//! exists but every stream has the default priority (the control for
//! `priority`).
//!
//! The class is a property of the THREAD that draws the stream. A thread a
//! background proof spawns must be marked again (`stark`'s per-table drivers
//! copy their spawner's class); a rayon worker of the global pool is never
//! marked, so a stream drawn inside a parallel iterator is a foreground one.
//! [`draws`] counts every draw by class, which is how a leak is seen.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Streams in the background pool when one exists.
pub const BACKGROUND_STREAMS: usize = 8;

/// The knob that creates the background pool.
pub const STREAM_CLASSES_ENV: &str = "LAMBDA_VM_STREAM_CLASSES";

/// Which class of work a thread's stream draws are for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamClass {
    /// Everything that does not say otherwise.
    Foreground,
    /// Work run beside the foreground's, which should take the card only where
    /// the foreground leaves it.
    Background,
}

/// What the backend builds its streams as (`LAMBDA_VM_STREAM_CLASSES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamClasses {
    /// One pool for every thread (unset, `off`): today's streams.
    Off,
    /// The foreground pool at the greatest priority, a background pool at the
    /// least (`priority`).
    Priority,
    /// A background pool, every stream at the default priority (`equal`).
    Equal,
}

impl StreamClasses {
    /// The knob's reading: unset or `off` is [`Self::Off`], `priority` and
    /// `equal` the two pools; anything else is an error naming the knob.
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.map(str::trim) {
            None | Some("off") => Ok(Self::Off),
            Some("priority") => Ok(Self::Priority),
            Some("equal") => Ok(Self::Equal),
            Some(other) => Err(format!(
                "{STREAM_CLASSES_ENV}={other}: off, priority or equal"
            )),
        }
    }

    /// Whether a background pool exists.
    pub fn split(self) -> bool {
        self != Self::Off
    }
}

/// The process's setting, read once. A value the knob does not take is read as
/// [`StreamClasses::Off`] and said once: the backend has no caller to refuse
/// to, and a measurement that needs the pools checks [`setting`] itself.
pub fn setting() -> StreamClasses {
    static SETTING: OnceLock<StreamClasses> = OnceLock::new();
    *SETTING.get_or_init(|| {
        StreamClasses::parse(std::env::var(STREAM_CLASSES_ENV).ok().as_deref()).unwrap_or_else(
            |e| {
                eprintln!("[gpu] stream classes: {e} — read as off");
                StreamClasses::Off
            },
        )
    })
}

thread_local! {
    static CLASS: Cell<StreamClass> = const { Cell::new(StreamClass::Foreground) };
}

/// This thread's class.
pub fn current() -> StreamClass {
    CLASS.with(Cell::get)
}

/// Mark this thread's stream draws as `class` until the guard drops, which
/// puts the previous class back (also as a panic unwinds).
pub fn enter(class: StreamClass) -> ClassGuard {
    let previous = CLASS.with(|c| c.replace(class));
    ClassGuard {
        previous,
        _this_thread: std::marker::PhantomData,
    }
}

/// Puts the thread's previous class back when it drops. Not `Send`: the class
/// belongs to the thread that entered it.
pub struct ClassGuard {
    previous: StreamClass,
    _this_thread: std::marker::PhantomData<*const ()>,
}

impl Drop for ClassGuard {
    fn drop(&mut self) {
        CLASS.with(|c| c.set(self.previous));
    }
}

/// Stream draws by class since the process started: (foreground pool,
/// background pool). A background thread's draw is counted in the pool it
/// came from, so with no background pool every draw is a foreground one.
static FOREGROUND_DRAWS: AtomicU64 = AtomicU64::new(0);
static BACKGROUND_DRAWS: AtomicU64 = AtomicU64::new(0);

/// (foreground-pool draws, background-pool draws) since the process started.
/// Monotone: a reader brackets the span it cares about and subtracts.
pub fn draws() -> (u64, u64) {
    (
        FOREGROUND_DRAWS.load(Ordering::Relaxed),
        BACKGROUND_DRAWS.load(Ordering::Relaxed),
    )
}

/// The two pools a backend draws its streams from, round-robin within each.
pub(crate) struct StreamPools<T> {
    foreground: Vec<T>,
    background: Vec<T>,
    next_foreground: AtomicUsize,
    next_background: AtomicUsize,
}

impl<T: Clone> StreamPools<T> {
    /// `background` empty is one pool for every thread.
    pub(crate) fn new(foreground: Vec<T>, background: Vec<T>) -> Self {
        assert!(!foreground.is_empty(), "a stream pool needs a stream");
        Self {
            foreground,
            background,
            next_foreground: AtomicUsize::new(0),
            next_background: AtomicUsize::new(0),
        }
    }

    /// The next stream for this thread's class: the background pool's for a
    /// background thread when there is one, the foreground's otherwise.
    pub(crate) fn next(&self) -> T {
        if current() == StreamClass::Background && !self.background.is_empty() {
            BACKGROUND_DRAWS.fetch_add(1, Ordering::Relaxed);
            let idx = self.next_background.fetch_add(1, Ordering::Relaxed) % self.background.len();
            return self.background[idx].clone();
        }
        FOREGROUND_DRAWS.fetch_add(1, Ordering::Relaxed);
        let idx = self.next_foreground.fetch_add(1, Ordering::Relaxed) % self.foreground.len();
        self.foreground[idx].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two pools of labelled stand-ins: foreground 0..4, background 100..102.
    fn pools(split: bool) -> StreamPools<usize> {
        StreamPools::new(
            (0..4).collect(),
            if split { (100..102).collect() } else { vec![] },
        )
    }

    /// A thread marked background draws only from the background pool, an
    /// unmarked one never does, and each pool is taken round-robin.
    #[test]
    fn a_background_thread_draws_only_background_streams() {
        let p = pools(true);
        let foreground: Vec<usize> = (0..8).map(|_| p.next()).collect();
        assert_eq!(foreground, vec![0, 1, 2, 3, 0, 1, 2, 3]);
        let background: Vec<usize> = std::thread::scope(|s| {
            s.spawn(|| {
                let _bg = enter(StreamClass::Background);
                (0..5).map(|_| p.next()).collect()
            })
            .join()
            .expect("the background thread")
        });
        assert_eq!(background, vec![100, 101, 100, 101, 100]);
        // Another thread, never marked, is in the foreground.
        let other = std::thread::scope(|s| s.spawn(|| p.next()).join().expect("a plain thread"));
        assert!(other < 100, "an unmarked thread drew {other}");
    }

    /// With no background pool (the knob off) a background thread draws the
    /// foreground's streams: today's behaviour whatever a thread is marked.
    #[test]
    fn without_a_background_pool_every_thread_draws_the_one_pool() {
        let p = pools(false);
        let _bg = enter(StreamClass::Background);
        let drawn: Vec<usize> = (0..6).map(|_| p.next()).collect();
        assert_eq!(drawn, vec![0, 1, 2, 3, 0, 1]);
    }

    /// The guard puts the previous class back, nested and on unwind.
    #[test]
    fn the_guard_restores_the_previous_class() {
        assert_eq!(current(), StreamClass::Foreground);
        {
            let _outer = enter(StreamClass::Background);
            {
                let _inner = enter(StreamClass::Foreground);
                assert_eq!(current(), StreamClass::Foreground);
            }
            assert_eq!(current(), StreamClass::Background);
        }
        assert_eq!(current(), StreamClass::Foreground);
        let unwound = std::panic::catch_unwind(|| {
            let _bg = enter(StreamClass::Background);
            panic!("unwinding through the guard");
        });
        assert!(unwound.is_err());
        assert_eq!(current(), StreamClass::Foreground, "an unwind restores it");
    }

    /// The knob's words, and a word it does not take refused by name.
    #[test]
    fn the_knob_reads_off_priority_and_equal() {
        assert_eq!(StreamClasses::parse(None), Ok(StreamClasses::Off));
        assert_eq!(StreamClasses::parse(Some("off")), Ok(StreamClasses::Off));
        assert_eq!(
            StreamClasses::parse(Some(" priority ")),
            Ok(StreamClasses::Priority)
        );
        assert_eq!(
            StreamClasses::parse(Some("equal")),
            Ok(StreamClasses::Equal)
        );
        let err = StreamClasses::parse(Some("high")).expect_err("not a setting");
        assert!(err.contains(STREAM_CLASSES_ENV), "{err}");
        assert!(!StreamClasses::Off.split());
        assert!(StreamClasses::Priority.split() && StreamClasses::Equal.split());
    }
}

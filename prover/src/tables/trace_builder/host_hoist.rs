//! `LAMBDA_VM_HOST_HOIST`: serial host work taken off the block build's
//! critical path, part by part. Every part builds the same tables, so the
//! proof's bytes are the same with any part on or off; unset, none is on.
//!
//! - `pages`: the finish computes the page layout once, each initial page
//!   copied whole from the image's pages instead of rebuilt a byte at a time
//!   through a map, and reuses it for BITWISE's page lookups and the plan.
//! - `image`: the builder writes the initial image a run of bytes at a time
//!   (a copy a page) instead of a byte at a time, and maps the walk's memory
//!   state from it on the pool, a page a job, so the walk starts sooner.
//! - `state`: in record mode, the finish's memory state, the windows'
//!   last step, is built on the builder's setup state instead of mapping the
//!   image again, its new pages allocated and filled on the pool, and the
//!   run's end read as the history keeps its timestamps.

use crate::Error;

/// The parts of `LAMBDA_VM_HOST_HOIST` that are on. The default is none: the
/// build as before the knob.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostHoist {
    /// The page layout computed once, from the image's pages.
    pub pages: bool,
    /// The builder's image written a run at a time, its memory state mapped
    /// on the pool.
    pub image: bool,
    /// Record mode's final memory state built on the setup state, its new
    /// pages on the pool.
    pub state: bool,
}

impl HostHoist {
    /// The knob's words.
    pub(crate) const WORDS: &'static str = "pages, image, state";

    /// Every part on.
    pub fn all() -> Self {
        Self {
            pages: true,
            image: true,
            state: true,
        }
    }

    /// Whether any part is on.
    pub fn any(self) -> bool {
        self.pages || self.image || self.state
    }

    /// The knob's reading: unset or empty, no part; else a comma-separated
    /// list of parts (`pages`, `image`, `state`). A word it does not take refuses the prove.
    pub(crate) fn parse(value: Option<&str>) -> Result<Self, Error> {
        let mut hoist = Self::default();
        let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(hoist);
        };
        for word in value.split(',').map(str::trim) {
            match word {
                "pages" => hoist.pages = true,
                "image" => hoist.image = true,
                "state" => hoist.state = true,
                _ => {
                    return Err(Error::Prover(format!(
                        "LAMBDA_VM_HOST_HOIST={value}: a comma-separated list of {} (unset: \
                         none, the default)",
                        Self::WORDS
                    )));
                }
            }
        }
        Ok(hoist)
    }

    /// The knob from the environment.
    pub(crate) fn from_env() -> Result<Self, Error> {
        Self::parse(std::env::var("LAMBDA_VM_HOST_HOIST").ok().as_deref())
    }

    /// The parts on, as the knob names them, or `none`.
    pub fn describe(self) -> String {
        let parts: Vec<&str> = [
            (self.pages, "pages"),
            (self.image, "image"),
            (self.state, "state"),
        ]
        .into_iter()
        .filter_map(|(on, word)| on.then_some(word))
        .collect();
        if parts.is_empty() {
            "none".to_string()
        } else {
            parts.join(",")
        }
    }
}

/// How often this thread computed the page layout each way, and wrote the
/// image a run at a time: a test's read of which path a build took.
#[cfg(test)]
pub(crate) mod calls {
    use std::cell::Cell;

    thread_local! {
        static BY_BYTE: Cell<usize> = const { Cell::new(0) };
        static HOISTED: Cell<usize> = const { Cell::new(0) };
        static RUNS: Cell<usize> = const { Cell::new(0) };
    }

    /// The image written a run at a time (`build_initial_image_by_runs`).
    pub(crate) fn note_runs() {
        RUNS.with(|n| n.set(n.get() + 1));
    }

    /// The images written a run at a time since the last take; reset.
    pub(crate) fn take_runs() -> usize {
        RUNS.with(|n| n.replace(0))
    }

    /// The layout computed a byte at a time (`page_configs_of`).
    pub(crate) fn note_by_byte() {
        BY_BYTE.with(|n| n.set(n.get() + 1));
    }

    /// The layout computed from whole pages (`page_configs_hoisted`).
    pub(crate) fn note_hoisted() {
        HOISTED.with(|n| n.set(n.get() + 1));
    }

    /// Final memory states built hoisted, process-wide (they are built on
    /// the walk's record thread): a test that reads it holds its own lock.
    static STATES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// A final memory state built hoisted (`finish_state_hoisted`).
    pub(crate) fn note_state() {
        STATES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// The final states built hoisted since the last take; reset.
    pub(crate) fn take_states() -> usize {
        STATES.swap(0, std::sync::atomic::Ordering::Relaxed)
    }

    /// Tests that read [`take_states`] take turns through this.
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// This test's turn at the final-state counter.
    pub(crate) fn turn() -> std::sync::MutexGuard<'static, ()> {
        TURN.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `(by byte, hoisted)` since the last take, and both reset.
    pub(crate) fn take() -> (usize, usize) {
        (
            BY_BYTE.with(|n| n.replace(0)),
            HOISTED.with(|n| n.replace(0)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::HostHoist;

    #[test]
    fn unset_or_empty_is_no_part() {
        for value in [None, Some(""), Some("  ")] {
            assert_eq!(HostHoist::parse(value).unwrap(), HostHoist::default());
        }
        assert!(!HostHoist::default().any());
        assert_eq!(HostHoist::default().describe(), "none");
    }

    #[test]
    fn each_word_sets_its_part_alone() {
        let pages = HostHoist::parse(Some("pages")).unwrap();
        assert_eq!(
            pages,
            HostHoist {
                pages: true,
                ..HostHoist::default()
            }
        );
        assert_eq!(pages.describe(), "pages");
        assert_eq!(HostHoist::parse(Some(" pages , pages ")).unwrap(), pages);
        let image = HostHoist::parse(Some("image")).unwrap();
        assert_eq!(
            image,
            HostHoist {
                image: true,
                ..HostHoist::default()
            }
        );
        assert_eq!(image.describe(), "image");
        let state = HostHoist::parse(Some("state")).unwrap();
        assert_eq!(
            state,
            HostHoist {
                state: true,
                ..HostHoist::default()
            }
        );
        assert_eq!(state.describe(), "state");
        let all = HostHoist::parse(Some("state,image,pages")).unwrap();
        assert_eq!(all, HostHoist::all());
        assert_eq!(all.describe(), "pages,image,state");
    }

    #[test]
    fn a_word_it_does_not_take_refuses() {
        for value in ["page", "pages,", "1", "all", "pages,bogus"] {
            assert!(HostHoist::parse(Some(value)).is_err(), "{value}");
        }
    }
}

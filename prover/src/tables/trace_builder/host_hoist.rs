//! `LAMBDA_VM_HOST_HOIST`: serial host work taken off the block build's
//! critical path, part by part. Every part builds the same tables, so the
//! proof's bytes are the same with any part on or off; unset, none is on.
//!
//! - `pages`: the finish computes the page layout once, each initial page
//!   copied whole from the image's pages instead of rebuilt a byte at a time
//!   through a map, and reuses it for BITWISE's page lookups and the plan.

use crate::Error;

/// The parts of `LAMBDA_VM_HOST_HOIST` that are on. The default is none: the
/// build as before the knob.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostHoist {
    /// The page layout computed once, from the image's pages.
    pub pages: bool,
}

impl HostHoist {
    /// The knob's words.
    pub(crate) const WORDS: &'static str = "pages";

    /// Every part on.
    pub fn all() -> Self {
        Self { pages: true }
    }

    /// Whether any part is on.
    pub fn any(self) -> bool {
        self.pages
    }

    /// The knob's reading: unset or empty, no part; else a comma-separated
    /// list of parts (`pages`). A word it does not take refuses the prove.
    pub(crate) fn parse(value: Option<&str>) -> Result<Self, Error> {
        let mut hoist = Self::default();
        let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(hoist);
        };
        for word in value.split(',').map(str::trim) {
            match word {
                "pages" => hoist.pages = true,
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
        let parts: Vec<&str> = [(self.pages, "pages")]
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

/// How often this thread computed the page layout each way: a test's read of
/// which path a build took.
#[cfg(test)]
pub(crate) mod calls {
    use std::cell::Cell;

    thread_local! {
        static BY_BYTE: Cell<usize> = const { Cell::new(0) };
        static HOISTED: Cell<usize> = const { Cell::new(0) };
    }

    /// The layout computed a byte at a time (`page_configs_of`).
    pub(crate) fn note_by_byte() {
        BY_BYTE.with(|n| n.set(n.get() + 1));
    }

    /// The layout computed from whole pages (`page_configs_hoisted`).
    pub(crate) fn note_hoisted() {
        HOISTED.with(|n| n.set(n.get() + 1));
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
        assert_eq!(pages, HostHoist { pages: true });
        assert_eq!(pages.describe(), "pages");
        assert_eq!(HostHoist::parse(Some(" pages , pages ")).unwrap(), pages);
        assert_eq!(HostHoist::all(), pages);
    }

    #[test]
    fn a_word_it_does_not_take_refuses() {
        for value in ["page", "pages,", "1", "all", "pages,bogus"] {
            assert!(HostHoist::parse(Some(value)).is_err(), "{value}");
        }
    }
}

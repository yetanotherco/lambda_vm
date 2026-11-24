//! `LAMBDA_VM_WALK_CUTS`: work taken out of the window walkers' loop, part by
//! part. Every part emits the same lists and counts the same lookups, so the
//! tables and the proof's bytes are the same with any part on or off; unset,
//! none is on.
//!
//! - `prescan`: a window's logs are no longer read once ahead of the walk for
//!   a halting cycle; the collect refuses such a cycle as it reads the log,
//!   with the same error.

use crate::Error;

/// The parts of `LAMBDA_VM_WALK_CUTS` that are on. The default is none: the
/// walk as before the knob.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalkCuts {
    /// The halt check folded into the collect, no pass of its own.
    pub prescan: bool,
}

impl WalkCuts {
    /// The knob's words.
    pub(crate) const WORDS: &'static str = "prescan";

    /// Every part on.
    pub fn all() -> Self {
        Self { prescan: true }
    }

    /// The knob's reading: unset or empty, no part; else a comma-separated
    /// list of parts, or `all`. A word it does not take refuses the prove.
    pub(crate) fn parse(value: Option<&str>) -> Result<Self, Error> {
        let mut cuts = Self::default();
        let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(cuts);
        };
        for word in value.split(',').map(str::trim) {
            match word {
                "all" => cuts = Self::all(),
                "prescan" => cuts.prescan = true,
                _ => {
                    return Err(Error::Prover(format!(
                        "LAMBDA_VM_WALK_CUTS={value}: a comma-separated list of {}, or all \
                         (unset: none, the default)",
                        Self::WORDS
                    )));
                }
            }
        }
        Ok(cuts)
    }

    /// The knob from the environment.
    pub(crate) fn from_env() -> Result<Self, Error> {
        Self::parse(std::env::var("LAMBDA_VM_WALK_CUTS").ok().as_deref())
    }

    /// The parts on, as the knob names them, or `none`.
    pub fn describe(self) -> String {
        let parts: Vec<&str> = [(self.prescan, "prescan")]
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

/// Each part alone, then every part: the arms a test compares with none.
#[cfg(test)]
pub(crate) fn each_part_and_all() -> Vec<(&'static str, WalkCuts)> {
    WalkCuts::WORDS
        .split(", ")
        .map(|word| (word, WalkCuts::parse(Some(word)).expect("its own word")))
        .chain([("all", WalkCuts::all())])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::WalkCuts;

    /// Unset, empty or blank is no part, the walk as before the knob.
    #[test]
    fn the_walk_cuts_knob_unset_or_empty_is_no_part() {
        for value in [None, Some(""), Some("  ")] {
            assert_eq!(WalkCuts::parse(value).ok(), Some(WalkCuts::default()));
        }
        assert_eq!(WalkCuts::default().describe(), "none");
    }

    /// Each word sets its part alone; `all` sets every part; anything else
    /// refuses the prove.
    #[test]
    fn the_walk_cuts_knob_takes_its_words_and_refuses_the_rest() {
        let prescan = WalkCuts::parse(Some(" prescan ")).expect("prescan");
        assert_eq!(prescan, WalkCuts { prescan: true });
        assert_eq!(WalkCuts::parse(Some("all")).ok(), Some(WalkCuts::all()));
        assert_eq!(WalkCuts::all().describe(), WalkCuts::WORDS.replace(' ', ""));
        for bad in ["pre-scan", "prescan,", "none", "1", "PRESCAN"] {
            assert!(WalkCuts::parse(Some(bad)).is_err(), "{bad}");
        }
    }
}

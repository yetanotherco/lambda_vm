//! `LAMBDA_VM_WALK_CUTS`: work taken out of the window walkers' loop, part by
//! part. Every part emits the same lists and counts the same lookups, so the
//! tables and the proof's bytes are the same with any part on or off. Unset
//! (or empty), every part is on, the default; `none` turns every part off,
//! and a list of parts turns on those alone.
//!
//! - `prescan`: a window's logs are no longer read once ahead of the walk for
//!   a halting cycle; the collect refuses such a cycle as it reads the log,
//!   with the same error.
//! - `const`: of a CPU row's seven lookups, the three its instruction alone
//!   gives are no longer counted a cycle at a time where the walker routes:
//!   each DECODE row's are added once, times the cycles the walker counted on
//!   that row, at the walker's end.
//! - `one-loop`: where the walker routes, its count's pass over a batch's
//!   CPU ops (DECODE rows, last ECALL) and its routing pass are one loop.
//! - `slots`: where the walker counts and lists none of the in-walk lookups,
//!   each in-row register slot's range lookup is counted where it is made
//!   instead of listed, counted after the batch and dropped.

use crate::Error;

/// The parts of `LAMBDA_VM_WALK_CUTS` that are on. The knob's default is
/// every part ([`WalkCuts::all`]); this type's `Default` is no part
/// ([`WalkCuts::none`]), the walk as before the knob, which a builder made
/// without the knob (`WindowedTraceBuilder::new`) takes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalkCuts {
    /// The halt check folded into the collect, no pass of its own.
    pub prescan: bool,
    /// A CPU row's three instruction lookups counted by DECODE row.
    pub constant: bool,
    /// The count's pass over the CPU ops folded into the routing's.
    pub one_loop: bool,
    /// The register slots' lookups counted where made.
    pub slots: bool,
}

impl WalkCuts {
    /// The knob's words.
    pub(crate) const WORDS: &'static str = "prescan, const, one-loop, slots";

    /// No part on: the walk as before the knob.
    pub fn none() -> Self {
        Self::default()
    }

    /// Every part on: the knob's default.
    pub fn all() -> Self {
        Self {
            prescan: true,
            constant: true,
            one_loop: true,
            slots: true,
        }
    }

    /// The knob's reading: unset or empty, every part (the default);
    /// `none`, no part; else a comma-separated list of the parts that are on,
    /// or `all`. A word it does not take refuses the prove.
    pub(crate) fn parse(value: Option<&str>) -> Result<Self, Error> {
        let mut cuts = Self::none();
        let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(Self::all());
        };
        if value == "none" {
            return Ok(cuts);
        }
        for word in value.split(',').map(str::trim) {
            match word {
                "all" => cuts = Self::all(),
                "prescan" => cuts.prescan = true,
                "const" => cuts.constant = true,
                "one-loop" => cuts.one_loop = true,
                "slots" => cuts.slots = true,
                _ => {
                    return Err(Error::Prover(format!(
                        "LAMBDA_VM_WALK_CUTS={value}: a comma-separated list of {}, all, or \
                         none (unset: every part, the default)",
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
        let parts: Vec<&str> = [
            (self.prescan, "prescan"),
            (self.constant, "const"),
            (self.one_loop, "one-loop"),
            (self.slots, "slots"),
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

    /// Unset, empty or blank is every part (the default); `none` is no
    /// part, the walk as before the knob; the type's `Default` is no part.
    #[test]
    fn the_walk_cuts_knob_unset_or_empty_is_every_part_and_none_is_no_part() {
        for value in [None, Some(""), Some("  ")] {
            assert_eq!(WalkCuts::parse(value).ok(), Some(WalkCuts::all()));
        }
        for value in ["none", " none "] {
            assert_eq!(WalkCuts::parse(Some(value)).ok(), Some(WalkCuts::none()));
        }
        assert_eq!(WalkCuts::none(), WalkCuts::default());
        assert_eq!(WalkCuts::none().describe(), "none");
    }

    /// Each word sets its part alone; `all` sets every part; anything else
    /// refuses the prove.
    #[test]
    fn the_walk_cuts_knob_takes_its_words_and_refuses_the_rest() {
        let one = |word: &str| WalkCuts::parse(Some(word)).expect(word);
        let none = WalkCuts::default();
        assert_eq!(
            one(" prescan "),
            WalkCuts {
                prescan: true,
                ..none
            }
        );
        assert_eq!(
            one("const"),
            WalkCuts {
                constant: true,
                ..none
            }
        );
        assert_eq!(
            one("one-loop"),
            WalkCuts {
                one_loop: true,
                ..none
            }
        );
        assert_eq!(
            one("slots"),
            WalkCuts {
                slots: true,
                ..none
            }
        );
        assert_eq!(one("prescan,const, one-loop ,slots"), WalkCuts::all());
        assert_eq!(WalkCuts::parse(Some("all")).ok(), Some(WalkCuts::all()));
        assert_eq!(WalkCuts::all().describe(), WalkCuts::WORDS.replace(' ', ""));
        for bad in [
            "pre-scan",
            "prescan,",
            "none,prescan",
            "prescan,none",
            "0",
            "1",
            "PRESCAN",
            "one_loop",
            "constant",
        ] {
            assert!(WalkCuts::parse(Some(bad)).is_err(), "{bad}");
        }
    }
}

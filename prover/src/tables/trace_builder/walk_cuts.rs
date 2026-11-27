//! `LAMBDA_VM_WALK_CUTS`: work taken out of the window walkers' loop, part by
//! part. Every part emits the same lists and counts the same lookups, so the
//! tables and the proof's bytes are the same with any part on or off. An unset
//! or blank value selects all six parts, as does `all`. `none` turns every
//! part off, and an explicit list selects its named parts. The previous
//! default is `prescan,const,one-loop,slots`.
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
//! - `rows-sized`: size the seven word tables' lists from the last window
//!   this walker walked.
//! - `count-direct`: count word and shift lookups without heap lists.

use crate::Error;

/// The parts of `LAMBDA_VM_WALK_CUTS` that are on. The knob defaults to
/// [`WalkCuts::all`]. The type's `Default` is [`WalkCuts::none`], which
/// `WindowedTraceBuilder::new` uses when constructed without the knob.
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
    /// Size the word tables' lists from the last window.
    pub rows_sized: bool,
    /// Count word and shift lookups without heap lists.
    pub count_direct: bool,
}

impl WalkCuts {
    /// The knob's words.
    pub(crate) const WORDS: &'static str =
        "prescan, const, one-loop, slots, rows-sized, count-direct";

    /// No part on: the walk as before the knob.
    pub fn none() -> Self {
        Self::default()
    }

    /// All six parts, as selected by the knob's `all` and default.
    pub fn all() -> Self {
        Self {
            prescan: true,
            constant: true,
            one_loop: true,
            slots: true,
            rows_sized: true,
            count_direct: true,
        }
    }

    /// Unset, blank or `all` selects all six parts. `none` selects no part.
    /// In a comma-separated list, `all` enables all six and each other word
    /// enables its own part.
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
                "rows-sized" => cuts.rows_sized = true,
                "count-direct" => cuts.count_direct = true,
                _ => {
                    return Err(Error::Prover(format!(
                        "LAMBDA_VM_WALK_CUTS={value}: a comma-separated list of {}, all, or \
                         none (unset: {default})",
                        Self::WORDS,
                        default = Self::all().describe(),
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

    /// The enabled parts as a comma-separated list, or `none`.
    pub fn describe(self) -> String {
        let parts: Vec<&str> = [
            (self.prescan, "prescan"),
            (self.constant, "const"),
            (self.one_loop, "one-loop"),
            (self.slots, "slots"),
            (self.rows_sized, "rows-sized"),
            (self.count_direct, "count-direct"),
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

/// Each part alone, then the environment default: the arms a test compares with none.
#[cfg(test)]
pub(crate) fn each_part_and_all() -> Vec<(&'static str, WalkCuts)> {
    WalkCuts::WORDS
        .split(", ")
        .map(|word| (word, WalkCuts::parse(Some(word)).expect("its own word")))
        .chain([("all", WalkCuts::parse(None).unwrap())])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::WalkCuts;

    #[test]
    fn the_walk_cuts_knob_defaults_to_all_six_parts() {
        for value in [None, Some(""), Some("  "), Some("all")] {
            let cuts = WalkCuts::parse(value).unwrap();
            assert_eq!(
                cuts.describe(),
                "prescan,const,one-loop,slots,rows-sized,count-direct"
            );
            assert!(cuts.rows_sized && cuts.count_direct);
        }
    }

    #[test]
    fn the_walk_cuts_knob_keeps_the_four_part_opt_out() {
        let cuts = WalkCuts::parse(Some("prescan,const, one-loop ,slots")).unwrap();
        assert_eq!(cuts.describe(), "prescan,const,one-loop,slots");
        assert!(!cuts.rows_sized && !cuts.count_direct);
        for value in ["none", " none "] {
            assert_eq!(WalkCuts::parse(Some(value)).ok(), Some(WalkCuts::none()));
        }
        assert_eq!(WalkCuts::none(), WalkCuts::default());
        assert_eq!(WalkCuts::none().describe(), "none");
    }

    /// Each part's word enables it alone. Unknown words refuse the prove.
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
        assert_eq!(
            one("rows-sized"),
            WalkCuts {
                rows_sized: true,
                ..none
            }
        );
        assert_eq!(
            one("count-direct"),
            WalkCuts {
                count_direct: true,
                ..none
            }
        );
        assert_eq!(one("rows-sized,count-direct,all"), WalkCuts::all());
        assert_eq!(one("all,rows-sized,count-direct"), WalkCuts::all());
        assert_eq!(
            one(WalkCuts::WORDS).describe(),
            WalkCuts::WORDS.replace(' ', "")
        );
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

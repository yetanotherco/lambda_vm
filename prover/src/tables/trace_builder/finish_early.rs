//! `LAMBDA_VM_FINISH_EARLY`: the finish's last jobs shortened, part by part.
//! Every part builds the same tables, so the proof's bytes are the same with
//! any part on or off; unset, none is on.
//!
//! - `bitwise-tail`: BITWISE's own job, the finish's last, adds the tables'
//!   lookups to the rest's in parallel chunks, writes its table straight into
//!   its packed form by row ranges at once, and digests its multiplicities
//!   for the log on a thread beside the fill, instead of each in turn on one
//!   thread with the table built at 8 bytes a cell first.

use crate::Error;

/// The parts of `LAMBDA_VM_FINISH_EARLY` that are on. The default is none:
/// the finish as before the knob.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FinishEarly {
    /// BITWISE's job split: added in chunks, written packed by row ranges,
    /// digested beside.
    pub bitwise_tail: bool,
}

impl FinishEarly {
    /// The knob's words.
    pub(crate) const WORDS: &'static str = "bitwise-tail";

    /// Every part on.
    pub fn all() -> Self {
        Self { bitwise_tail: true }
    }

    /// The knob's reading: unset or empty, no part; else a comma-separated
    /// list of parts, or `all`. A word it does not take refuses the prove.
    pub(crate) fn parse(value: Option<&str>) -> Result<Self, Error> {
        let mut early = Self::default();
        let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(early);
        };
        for word in value.split(',').map(str::trim) {
            match word {
                "all" => early = Self::all(),
                "bitwise-tail" => early.bitwise_tail = true,
                _ => {
                    return Err(Error::Prover(format!(
                        "LAMBDA_VM_FINISH_EARLY={value}: a comma-separated list of {}, or all \
                         (unset: none, the default)",
                        Self::WORDS
                    )));
                }
            }
        }
        Ok(early)
    }

    /// The knob from the environment.
    pub(crate) fn from_env() -> Result<Self, Error> {
        Self::parse(std::env::var("LAMBDA_VM_FINISH_EARLY").ok().as_deref())
    }

    /// The parts on, as the knob names them, or `none`.
    pub fn describe(self) -> String {
        let parts: Vec<&str> = [(self.bitwise_tail, "bitwise-tail")]
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

/// BITWISE jobs built split (`bitwise-tail`) since the process began: a
/// test's proof the part ran (any build that turns it on adds to it).
#[cfg(test)]
static BITWISE_TAILS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A BITWISE job built split.
#[cfg(test)]
pub(crate) fn note_bitwise_tail() {
    BITWISE_TAILS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// [`BITWISE_TAILS`].
#[cfg(test)]
pub(crate) fn bitwise_tails() -> usize {
    BITWISE_TAILS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::FinishEarly;

    /// Unset, empty or blank is no part, the finish as before the knob.
    #[test]
    fn the_finish_early_knob_unset_or_empty_is_no_part() {
        for value in [None, Some(""), Some("  ")] {
            assert_eq!(FinishEarly::parse(value).ok(), Some(FinishEarly::default()));
        }
        assert_eq!(FinishEarly::default().describe(), "none");
    }

    /// Each word sets its part alone; `all` sets every part; anything else
    /// refuses the prove.
    #[test]
    fn the_finish_early_knob_takes_its_words_and_refuses_the_rest() {
        for word in FinishEarly::WORDS.split(", ") {
            let early = FinishEarly::parse(Some(word)).expect(word);
            assert_eq!(early.describe(), word, "{word} alone");
        }
        assert_eq!(
            FinishEarly::parse(Some("all")).ok(),
            Some(FinishEarly::all())
        );
        assert_eq!(
            FinishEarly::all().describe(),
            FinishEarly::WORDS.replace(' ', "")
        );
        for bad in [
            "bitwise",
            "tail",
            "bitwise_tail",
            "bitwise-tail,",
            "none",
            "1",
        ] {
            assert!(FinishEarly::parse(Some(bad)).is_err(), "{bad}");
        }
    }
}

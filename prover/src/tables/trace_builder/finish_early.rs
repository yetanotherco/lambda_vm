//! `LAMBDA_VM_FINISH_EARLY`: the finish's last jobs shortened, part by part.
//! Every part builds the same tables, so the proof's bytes are the same with
//! any part on or off. Unset (or empty), every part is on, the default;
//! `none` turns every part off, and a list of parts turns on those alone.
//!
//! - `bitwise-tail`: BITWISE's own job, the finish's last, adds the tables'
//!   lookups to the rest's in parallel chunks, writes its table straight into
//!   its packed form by row ranges at once, and digests its multiplicities
//!   for the log on a thread beside the fill, instead of each in turn on one
//!   thread with the table built at 8 bytes a cell first.
//! - `ecdas-count`: the ECDAS rows' BITWISE lookups of the ECREC calls whose
//!   witnesses the walk hands off are counted where each witness is computed,
//!   and the finish adds those counts instead of counting the rows; when they
//!   are not every row's (a call computed in the walk, one not counted) it
//!   counts the rows, as without the part.

use crate::Error;

/// The parts of `LAMBDA_VM_FINISH_EARLY` that are on. The knob's default is
/// every part ([`FinishEarly::all`]); this type's `Default` is no part
/// ([`FinishEarly::none`]), the finish as before the knob, which a builder
/// made without the knob takes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FinishEarly {
    /// BITWISE's job split: added in chunks, written packed by row ranges,
    /// digested beside.
    pub bitwise_tail: bool,
    /// ECDAS's lookups counted beside the ECREC witnesses.
    pub ecdas_count: bool,
    /// Test only: a finish under this value counts the parts it ran
    /// ([`bitwise_tails`], [`ecdas_counts`], process-wide counts). Only the
    /// tests that read the counts set it, so another test's block prove
    /// beside them, which runs every part under the knob's default, does not
    /// move their counts.
    #[cfg(test)]
    pub(crate) counted: bool,
}

impl FinishEarly {
    /// The knob's words.
    pub(crate) const WORDS: &'static str = "bitwise-tail, ecdas-count";

    /// No part on: the finish as before the knob.
    pub fn none() -> Self {
        Self::default()
    }

    /// Every part on: the knob's default.
    pub fn all() -> Self {
        Self {
            bitwise_tail: true,
            ecdas_count: true,
            #[cfg(test)]
            counted: false,
        }
    }

    /// This value, the parts its finish runs counted ([`bitwise_tails`],
    /// [`ecdas_counts`]).
    #[cfg(test)]
    pub(crate) fn counted(self) -> Self {
        Self {
            counted: true,
            ..self
        }
    }

    /// The knob's reading: unset or empty, every part (the default);
    /// `none`, no part; else a comma-separated list of the parts that are on,
    /// or `all`. A word it does not take refuses the prove.
    pub(crate) fn parse(value: Option<&str>) -> Result<Self, Error> {
        let mut early = Self::none();
        let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(Self::all());
        };
        if value == "none" {
            return Ok(early);
        }
        for word in value.split(',').map(str::trim) {
            match word {
                "all" => early = Self::all(),
                "bitwise-tail" => early.bitwise_tail = true,
                "ecdas-count" => early.ecdas_count = true,
                _ => {
                    return Err(Error::Prover(format!(
                        "LAMBDA_VM_FINISH_EARLY={value}: a comma-separated list of {}, all, \
                         or none (unset: every part, the default)",
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
        let parts: Vec<&str> = [
            (self.bitwise_tail, "bitwise-tail"),
            (self.ecdas_count, "ecdas-count"),
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

/// BITWISE jobs built split (`bitwise-tail`) since the process began, under
/// a [`FinishEarly::counted`] value: a test's proof the part ran.
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

/// Finishes that added ECDAS's lookups counted beside the ECREC witnesses
/// (`ecdas-count`) since the process began, under a [`FinishEarly::counted`]
/// value: a test's proof the part ran.
#[cfg(test)]
static ECDAS_COUNTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A finish that added them.
#[cfg(test)]
pub(crate) fn note_ecdas_count() {
    ECDAS_COUNTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// [`ECDAS_COUNTS`].
#[cfg(test)]
pub(crate) fn ecdas_counts() -> usize {
    ECDAS_COUNTS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::FinishEarly;

    /// Unset, empty or blank is every part (the default); `none` is no
    /// part, the finish as before the knob; the type's `Default` is no part.
    #[test]
    fn the_finish_early_knob_unset_or_empty_is_every_part_and_none_is_no_part() {
        for value in [None, Some(""), Some("  ")] {
            assert_eq!(FinishEarly::parse(value).ok(), Some(FinishEarly::all()));
        }
        for value in ["none", " none "] {
            assert_eq!(
                FinishEarly::parse(Some(value)).ok(),
                Some(FinishEarly::none())
            );
        }
        assert_eq!(FinishEarly::none(), FinishEarly::default());
        assert_eq!(FinishEarly::none().describe(), "none");
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
            "ecdas",
            "ecdas_count",
            "none,bitwise-tail",
            "bitwise-tail,none",
            "0",
            "1",
        ] {
            assert!(FinishEarly::parse(Some(bad)).is_err(), "{bad}");
        }
    }
}

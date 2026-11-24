//! The LAST level's own format: the proof the host verifies, never a parent.
//!
//! Every level proves under the same levers ([`super::levels::level_levers`])
//! because every level but the last is verified in-guest by the level above
//! it, and the in-guest verifier needs the stacked instruction columns and
//! today's chains. The last level's proof is verified by the host alone, so it
//! may take the formats only the host verifier checks, each a lever of the
//! W-LFM format that exists and is tested on its own:
//!
//! - **evaluated instruction columns** ([`WlfmPrep::Evaluated`]): the verifier
//!   derives the program, so it evaluates each table's preprocessed prefix at
//!   the table's point itself; no prepared stack is committed or opened;
//! - **a first fold of its own** ([`WlfmLevers::first_fold`]): the first round's
//!   fold, a leaf of `2^k` values a polynomial, the query count sized over the
//!   schedule it gives;
//! - **a final polynomial** ([`FinalPoly::Vars`]): each chain stops committing
//!   once at most `v` variables remain and sends that polynomial.
//!
//! A fourth knob is an encoding, not a format: the last proof's file in the
//! compact wire encoding with every repeated Merkle sibling sent once
//! ([`super::wire::WireFormat::Dedup`]).
//!
//! # A verifier constant, read from the environment for measurement
//!
//! Each format knob is off by default, and off is today's format byte for
//! byte: [`LastLevel::DEFAULT`]'s levers are [`super::levels::level_levers`]'.
//! Both sides of one process read the same value ([`last_level`]): the
//! levels' prover gives it to the last level only, and the levels' verifier
//! re-derives the last level's artifacts under it. Nothing is read from a
//! proof: the levers are folded into the last level's config and identity,
//! which its statement absorbs, so a proof made under one setting is refused
//! under any other. The pinned public verifier refuses a process whose knobs
//! ask for a last level other than the default
//! (`levels::check_process_format`).

use multilinear::whir_chain::FinalPoly;

use super::wire::WireFormat;
use super::wlfm::{WlfmFamily, WlfmLevers, WlfmPrep};

/// `LAMBDA_VM_LAST_LEVEL_EVALUATED=1`: the last level's instruction columns
/// are evaluated by the verifier, not committed and opened. `0` or unset is
/// today's stacked binding.
pub const LAST_EVALUATED_ENV: &str = "LAMBDA_VM_LAST_LEVEL_EVALUATED";

/// `LAMBDA_VM_LAST_LEVEL_FIRST_FOLD=4`: the last level's chains fold 4
/// variables in their first round. Unset, `off` or `6` is today's first fold.
pub const LAST_FIRST_FOLD_ENV: &str = "LAMBDA_VM_LAST_LEVEL_FIRST_FOLD";

/// `LAMBDA_VM_LAST_LEVEL_FINAL=9`: the last level's chains stop at a final
/// polynomial of at most 9 variables. Unset, `0` or `off` is today's: every
/// round commits its successor.
pub const LAST_FINAL_ENV: &str = "LAMBDA_VM_LAST_LEVEL_FINAL";

/// `LAMBDA_VM_LAST_LEVEL_WIRE=rkyv|compact|dedup`: the encoding of the last
/// level's proof file (an encoding, not a format: the decoded proof is the
/// one the prover made). Unset is `rkyv`, today's archive.
pub const LAST_WIRE_ENV: &str = "LAMBDA_VM_LAST_LEVEL_WIRE";

/// The first fold [`LAST_FIRST_FOLD_ENV`] is measured at: a round-0 leaf of
/// 16 values a polynomial, where today's first fold of 6 opens 64.
pub const LAST_FIRST_FOLD_MEASURED: u8 = 4;

/// The final polynomial's width [`LAST_FINAL_ENV`] is measured at: the BLAKE3
/// base's own (`block_whir::B3_BASE_FINAL_VARS`).
pub const LAST_FINAL_MEASURED: u8 = 9;

/// The last level's format beyond every level's: a verifier constant, named
/// by both sides. [`LastLevel::DEFAULT`] is today's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LastLevel {
    /// The instruction columns evaluated by the verifier.
    pub evaluated: bool,
    /// The chains' first fold; `None` is today's.
    pub first_fold: Option<u8>,
    /// The chains' final polynomial.
    pub final_poly: FinalPoly,
}

impl LastLevel {
    /// Today's last level: the levers every level proves under.
    pub const DEFAULT: Self = Self {
        evaluated: false,
        first_fold: None,
        final_poly: FinalPoly::Off,
    };

    /// Whether this is today's last level.
    pub fn is_default(&self) -> bool {
        *self == Self::DEFAULT
    }

    /// The levers the last level proves under: every level's
    /// ([`super::levels::level_levers`]) with this format's own on top.
    pub fn levers(&self, family: WlfmFamily) -> WlfmLevers {
        let every = super::levels::level_levers(family);
        WlfmLevers {
            prep: if self.evaluated {
                WlfmPrep::Evaluated
            } else {
                every.prep
            },
            first_fold: self.first_fold.or(every.first_fold),
            final_poly: match self.final_poly {
                FinalPoly::Off => every.final_poly,
                on => on,
            },
            ..every
        }
    }
}

/// [`LAST_EVALUATED_ENV`]'s word: unset, empty or `0` is off, `1` is on;
/// anything else is refused.
pub fn evaluated_word(word: Option<&str>) -> Result<bool, String> {
    match word.map(str::trim) {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{LAST_EVALUATED_ENV}={other:?}: expected 0 or 1")),
    }
}

/// [`LAST_FIRST_FOLD_ENV`]'s word: unset, empty, `off` or `6` (today's first
/// fold, under its own name) is today's; `4` is [`LAST_FIRST_FOLD_MEASURED`];
/// anything else is refused until it is tested.
pub fn first_fold_word(word: Option<&str>) -> Result<Option<u8>, String> {
    match word.map(|w| w.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("off") | Some("6") => Ok(None),
        Some("4") => Ok(Some(LAST_FIRST_FOLD_MEASURED)),
        Some(other) => Err(format!(
            "{LAST_FIRST_FOLD_ENV}={other:?}: expected off or {LAST_FIRST_FOLD_MEASURED}"
        )),
    }
}

/// [`LAST_FINAL_ENV`]'s word: unset, empty, `0` or `off` is no final
/// polynomial; `9` is [`LAST_FINAL_MEASURED`]; anything else is refused until
/// it is tested.
pub fn final_word(word: Option<&str>) -> Result<FinalPoly, String> {
    match word.map(|w| w.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("0") | Some("off") => Ok(FinalPoly::Off),
        Some("9") => Ok(FinalPoly::Vars(LAST_FINAL_MEASURED)),
        Some(other) => Err(format!(
            "{LAST_FINAL_ENV}={other:?}: expected off or {LAST_FINAL_MEASURED}"
        )),
    }
}

/// [`LAST_WIRE_ENV`]'s word: unset, empty or `rkyv` is the archive;
/// `compact` and `dedup` the compact encodings; anything else is refused.
pub fn wire_word(word: Option<&str>) -> Result<WireFormat, String> {
    match word.map(|w| w.trim().to_ascii_lowercase()) {
        None => Ok(WireFormat::Rkyv),
        Some(w) => match w.as_str() {
            "" | "rkyv" => Ok(WireFormat::Rkyv),
            "compact" => Ok(WireFormat::Compact),
            "dedup" => Ok(WireFormat::Dedup),
            other => Err(format!(
                "{LAST_WIRE_ENV}={other:?}: expected rkyv, compact or dedup"
            )),
        },
    }
}

/// The last level's format from the three knobs' words.
pub fn last_level_setting(
    evaluated: Option<&str>,
    first_fold: Option<&str>,
    final_poly: Option<&str>,
) -> Result<LastLevel, String> {
    Ok(LastLevel {
        evaluated: evaluated_word(evaluated)?,
        first_fold: first_fold_word(first_fold)?,
        final_poly: final_word(final_poly)?,
    })
}

/// A knob's word in this process; a word that is not Unicode is refused.
fn env_word(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(w) => Ok(Some(w)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(w)) => Err(format!("{name}={w:?}: not a word")),
    }
}

/// The last level's format the environment names, parsed now.
pub fn last_level_from_env() -> Result<LastLevel, String> {
    last_level_setting(
        env_word(LAST_EVALUATED_ENV)?.as_deref(),
        env_word(LAST_FIRST_FOLD_ENV)?.as_deref(),
        env_word(LAST_FINAL_ENV)?.as_deref(),
    )
}

/// The last level's format in this process: a test's override on this
/// thread, else the environment's, read once.
pub fn last_level() -> Result<LastLevel, String> {
    #[cfg(test)]
    if let Some(last) = test_last::get() {
        return Ok(last);
    }
    static LAST: std::sync::OnceLock<Result<LastLevel, String>> = std::sync::OnceLock::new();
    LAST.get_or_init(last_level_from_env).clone()
}

/// The last level's file encoding in this process ([`LAST_WIRE_ENV`]).
pub fn last_wire() -> Result<WireFormat, String> {
    wire_word(env_word(LAST_WIRE_ENV)?.as_deref())
}

/// The last level's proof as a file in `wire`'s encoding: the rkyv archive,
/// or the compact bytes ([`super::wire`]).
pub fn last_level_file(
    proof: &super::wlfm::WlfmProof,
    wire: WireFormat,
) -> Result<Vec<u8>, String> {
    match super::wire::encode_as(proof, wire) {
        Some(bytes) => Ok(bytes),
        None => rkyv::to_bytes::<rkyv::rancor::Error>(proof)
            .map(|b| b.to_vec())
            .map_err(|e| format!("the last level's archive: {e}")),
    }
}

/// The proof a compact last-level file holds ([`last_level_file`] under
/// [`WireFormat::Compact`] or [`WireFormat::Dedup`]), refused unless the
/// bytes are exactly one proof's encoding. The archive is read where it is
/// written today, not here.
pub fn read_last_level_file(
    bytes: &[u8],
    wire: WireFormat,
) -> Result<super::wlfm::WlfmProof, String> {
    match wire {
        WireFormat::Compact => super::wire::decode(bytes),
        WireFormat::Dedup => super::wire::decode_dedup(bytes),
        WireFormat::Rkyv => Err("the archive is not read here".to_string()),
    }
}

/// A test's last-level format for this thread until the guard drops.
#[cfg(test)]
pub(crate) fn set_last_level_for_test(last: LastLevel) -> test_last::Guard {
    test_last::set(last)
}

#[cfg(test)]
pub(crate) mod test_last {
    use super::LastLevel;
    use std::cell::Cell;

    thread_local! {
        static LAST: Cell<Option<LastLevel>> = const { Cell::new(None) };
    }

    /// Restores the thread's earlier override when dropped.
    pub(crate) struct Guard(Option<LastLevel>);

    impl Drop for Guard {
        fn drop(&mut self) {
            LAST.with(|c| c.set(self.0));
        }
    }

    pub(super) fn set(last: LastLevel) -> Guard {
        Guard(LAST.with(|c| c.replace(Some(last))))
    }

    pub(super) fn get() -> Option<LastLevel> {
        LAST.with(Cell::get)
    }
}

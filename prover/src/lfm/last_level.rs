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
//! # A verifier constant, pinned a family
//!
//! The format is the verifier's constant ([`LastLevel::pinned`]): under the
//! BLAKE3 family every lever is on ([`LastLevel::ON`]); under Poseidon1 every
//! lever is off ([`LastLevel::OFF`], the levers every level proves under:
//! [`super::levels::level_levers`]), and a lever asked for there is refused.
//! Each knob can turn its own lever off (or, under BLAKE3, back on) in a
//! process, for measurement and as a fallback; an unset knob is the family's
//! pinned value. The levels' prover reads the process's value
//! ([`last_level`]) once and gives it to the last level only. The public
//! verifier never reads it: it names the family's pinned format, derives the
//! last level's artifacts under that, and refuses a process whose knobs
//! resolve to another before deriving anything
//! (`levels::check_process_format`). Nothing is read from a proof: the levers
//! are folded into the last level's config and identity, which its statement
//! absorbs, so a proof made under one format is refused under any other.
//! Only the formats proved end to end are taken ([`LastLevel::is_tested`]).
//!
//! # The file
//!
//! The last level's file ([`last_level_file`]) opens with a header: the
//! magic [`LAST_FILE_MAGIC`], the file version [`LAST_FILE_VERSION`], the
//! code of the format the proof was made in ([`LastLevel::code`]) and the
//! encoding's code. A reader is given the one format it checks and refuses a
//! file of another version or another format by name before decoding
//! anything ([`read_last_level_file`]). Version 2 carries a validated leaf map
//! as a second statement input to the opt-in cluster verifier. After the
//! common header come the map's little-endian u32 length, one byte per group,
//! and a 32-byte checksum to detect corruption before decoding the proof.
//! The join's program identity binds the validated map.

use multilinear::whir_chain::FinalPoly;

use super::wire::WireFormat;
use super::wlfm::{WlfmFamily, WlfmLevers, WlfmPrep};

/// `LAMBDA_VM_LAST_LEVEL_EVALUATED=0|1`: `1` (the BLAKE3 levels' pinned
/// value) evaluates the last level's instruction columns in the verifier,
/// `0` commits and opens them in a prepared stack, as every other level.
/// Unset or empty is the family's pinned value ([`LastLevel::pinned`]).
pub const LAST_EVALUATED_ENV: &str = "LAMBDA_VM_LAST_LEVEL_EVALUATED";

/// `LAMBDA_VM_LAST_LEVEL_FIRST_FOLD=4|off`: `4` (the BLAKE3 levels' pinned
/// value) folds 4 variables in the last level's chains' first round; `off`
/// (or `6`, the same first fold under its own name) is every level's. Unset
/// or empty is the family's pinned value.
pub const LAST_FIRST_FOLD_ENV: &str = "LAMBDA_VM_LAST_LEVEL_FIRST_FOLD";

/// `LAMBDA_VM_LAST_LEVEL_FINAL=9|off`: `9` (the BLAKE3 levels' pinned value)
/// stops the last level's chains at a final polynomial of at most 9
/// variables; `off` (or `0`) commits every round's successor, as every other
/// level. Unset or empty is the family's pinned value.
pub const LAST_FINAL_ENV: &str = "LAMBDA_VM_LAST_LEVEL_FINAL";

/// `LAMBDA_VM_LAST_LEVEL_WIRE=dedup|compact|rkyv`: the encoding of the last
/// level's proof file (an encoding, not a format: the decoded proof is the
/// one the prover made). Unset or empty is [`LAST_WIRE_DEFAULT`].
pub const LAST_WIRE_ENV: &str = "LAMBDA_VM_LAST_LEVEL_WIRE";

/// The last level's file encoding when [`LAST_WIRE_ENV`] is unset: the
/// compact encoding with every repeated Merkle sibling sent once.
pub const LAST_WIRE_DEFAULT: WireFormat = WireFormat::Dedup;

/// The last level's first fold under every lever: a round-0 leaf of 16
/// values a polynomial, where every level's first fold of 6 opens 64.
pub const LAST_FIRST_FOLD_MEASURED: u8 = 4;

/// The last level's final polynomial's width under every lever: the BLAKE3
/// base's own (`block_whir::B3_BASE_FINAL_VARS`).
pub const LAST_FINAL_MEASURED: u8 = 9;

/// The last level's format beyond every level's: a verifier constant, named
/// by both sides. [`LastLevel::pinned`] is a family's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LastLevel {
    /// The instruction columns evaluated by the verifier.
    pub evaluated: bool,
    /// The chains' first fold; `None` is every level's.
    pub first_fold: Option<u8>,
    /// The chains' final polynomial.
    pub final_poly: FinalPoly,
}

impl LastLevel {
    /// Every lever off: the levers every level proves under. The Poseidon1
    /// levels' pinned format, and the BLAKE3 levels' before the last level
    /// had a format of its own.
    pub const OFF: Self = Self {
        evaluated: false,
        first_fold: None,
        final_poly: FinalPoly::Off,
    };

    /// Every lever on, at its measured setting: the BLAKE3 levels' pinned
    /// format.
    pub const ON: Self = Self {
        evaluated: true,
        first_fold: Some(LAST_FIRST_FOLD_MEASURED),
        final_poly: FinalPoly::Vars(LAST_FINAL_MEASURED),
    };

    /// The format `family`'s public verifier names, and its prover makes
    /// with every knob unset: [`Self::ON`] under BLAKE3, [`Self::OFF`] under
    /// Poseidon1 (where a lever is refused,
    /// `levels::check_last_level_family`).
    pub const fn pinned(family: WlfmFamily) -> Self {
        match family {
            WlfmFamily::B3 => Self::ON,
            WlfmFamily::P1 => Self::OFF,
        }
    }

    /// Whether every lever is off.
    pub fn is_off(&self) -> bool {
        *self == Self::OFF
    }

    /// Whether this format has been proved and verified end to end: every
    /// lever off, every lever on, or one lever alone. The two-lever mixes
    /// the knobs can reach are refused until they are.
    pub fn is_tested(&self) -> bool {
        let on = [
            self.evaluated,
            self.first_fold.is_some(),
            self.final_poly != FinalPoly::Off,
        ]
        .iter()
        .filter(|&&b| b)
        .count();
        on != 2 && self.code().is_some()
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

    /// The format's code in the file header: bit 0 the evaluated columns,
    /// bit 1 the measured first fold, bit 2 the measured final polynomial.
    /// `None` for a setting no knob gives (a first fold or a final width
    /// other than the measured one), which no file is written in.
    pub fn code(&self) -> Option<u8> {
        let first = match self.first_fold {
            None => 0,
            Some(LAST_FIRST_FOLD_MEASURED) => 2,
            Some(_) => return None,
        };
        let last = match self.final_poly {
            FinalPoly::Off => 0,
            FinalPoly::Vars(LAST_FINAL_MEASURED) => 4,
            FinalPoly::Vars(_) => return None,
        };
        Some(u8::from(self.evaluated) | first | last)
    }

    /// The format a header's code names; `None` for a code no format has.
    pub fn from_code(code: u8) -> Option<Self> {
        (code < 8).then(|| Self {
            evaluated: code & 1 != 0,
            first_fold: (code & 2 != 0).then_some(LAST_FIRST_FOLD_MEASURED),
            final_poly: if code & 4 != 0 {
                FinalPoly::Vars(LAST_FINAL_MEASURED)
            } else {
                FinalPoly::Off
            },
        })
    }
}

impl std::fmt::Display for LastLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let columns = if self.evaluated {
            "evaluated"
        } else {
            "committed"
        };
        write!(f, "instruction columns {columns}, first fold ")?;
        match self.first_fold {
            Some(k) => write!(f, "{k}")?,
            None => write!(f, "every level's")?,
        }
        match self.final_poly {
            FinalPoly::Vars(v) => write!(f, ", final polynomial of {v} variables"),
            FinalPoly::Off => write!(f, ", no final polynomial"),
        }
    }
}

/// The last-level knobs' words, parsed: `None` where a knob is unset (the
/// family's pinned value, [`LastLevelKnobs::resolve`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LastLevelKnobs {
    /// [`LAST_EVALUATED_ENV`].
    pub evaluated: Option<bool>,
    /// [`LAST_FIRST_FOLD_ENV`]: `Some(None)` is every level's first fold.
    pub first_fold: Option<Option<u8>>,
    /// [`LAST_FINAL_ENV`].
    pub final_poly: Option<FinalPoly>,
}

impl LastLevelKnobs {
    /// The format under `family`: each knob's value where it is set, the
    /// family's pinned value ([`LastLevel::pinned`]) where it is not.
    pub fn resolve(self, family: WlfmFamily) -> LastLevel {
        let pinned = LastLevel::pinned(family);
        LastLevel {
            evaluated: self.evaluated.unwrap_or(pinned.evaluated),
            first_fold: self.first_fold.unwrap_or(pinned.first_fold),
            final_poly: self.final_poly.unwrap_or(pinned.final_poly),
        }
    }
}

/// A knob's word, trimmed and lower case; unset or empty is `None`.
fn set_word(word: Option<&str>) -> Option<String> {
    word.map(|w| w.trim().to_ascii_lowercase())
        .filter(|w| !w.is_empty())
}

/// [`LAST_EVALUATED_ENV`]'s word: unset or empty is the family's (`None`),
/// `0` or `off` is off, `1` or `on` is on; anything else is refused.
pub fn evaluated_word(word: Option<&str>) -> Result<Option<bool>, String> {
    match set_word(word).as_deref() {
        None => Ok(None),
        Some("0") | Some("off") => Ok(Some(false)),
        Some("1") | Some("on") => Ok(Some(true)),
        Some(other) => Err(format!(
            "{LAST_EVALUATED_ENV}={other:?}: expected 0, 1, off or on"
        )),
    }
}

/// [`LAST_FIRST_FOLD_ENV`]'s word: unset or empty is the family's (`None`);
/// `off` or `6` (every level's first fold, under its own name) is every
/// level's; `4` is [`LAST_FIRST_FOLD_MEASURED`]; anything else is refused
/// until it is tested.
pub fn first_fold_word(word: Option<&str>) -> Result<Option<Option<u8>>, String> {
    match set_word(word).as_deref() {
        None => Ok(None),
        Some("off") | Some("6") => Ok(Some(None)),
        Some("4") => Ok(Some(Some(LAST_FIRST_FOLD_MEASURED))),
        Some(other) => Err(format!(
            "{LAST_FIRST_FOLD_ENV}={other:?}: expected off or {LAST_FIRST_FOLD_MEASURED}"
        )),
    }
}

/// [`LAST_FINAL_ENV`]'s word: unset or empty is the family's (`None`); `0`
/// or `off` is no final polynomial; `9` is [`LAST_FINAL_MEASURED`];
/// anything else is refused until it is tested.
pub fn final_word(word: Option<&str>) -> Result<Option<FinalPoly>, String> {
    match set_word(word).as_deref() {
        None => Ok(None),
        Some("0") | Some("off") => Ok(Some(FinalPoly::Off)),
        Some("9") => Ok(Some(FinalPoly::Vars(LAST_FINAL_MEASURED))),
        Some(other) => Err(format!(
            "{LAST_FINAL_ENV}={other:?}: expected off or {LAST_FINAL_MEASURED}"
        )),
    }
}

/// [`LAST_WIRE_ENV`]'s word: unset or empty is [`LAST_WIRE_DEFAULT`];
/// `rkyv`, `compact` and `dedup` name their encodings; anything else is
/// refused.
pub fn wire_word(word: Option<&str>) -> Result<WireFormat, String> {
    match set_word(word).as_deref() {
        None => Ok(LAST_WIRE_DEFAULT),
        Some("rkyv") => Ok(WireFormat::Rkyv),
        Some("compact") => Ok(WireFormat::Compact),
        Some("dedup") => Ok(WireFormat::Dedup),
        Some(other) => Err(format!(
            "{LAST_WIRE_ENV}={other:?}: expected rkyv, compact or dedup"
        )),
    }
}

/// The three format knobs' words, parsed.
pub fn last_level_knobs(
    evaluated: Option<&str>,
    first_fold: Option<&str>,
    final_poly: Option<&str>,
) -> Result<LastLevelKnobs, String> {
    Ok(LastLevelKnobs {
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

/// The format knobs the environment sets, parsed now.
pub fn last_level_knobs_from_env() -> Result<LastLevelKnobs, String> {
    last_level_knobs(
        env_word(LAST_EVALUATED_ENV)?.as_deref(),
        env_word(LAST_FIRST_FOLD_ENV)?.as_deref(),
        env_word(LAST_FINAL_ENV)?.as_deref(),
    )
}

/// The last level's format in this process under `family`: a test's
/// override on this thread, else the environment's knobs (read once) over
/// the family's pinned format ([`LastLevelKnobs::resolve`]).
pub fn last_level(family: WlfmFamily) -> Result<LastLevel, String> {
    #[cfg(test)]
    if let Some(last) = test_last::get() {
        return Ok(last);
    }
    static KNOBS: std::sync::OnceLock<Result<LastLevelKnobs, String>> = std::sync::OnceLock::new();
    KNOBS
        .get_or_init(last_level_knobs_from_env)
        .clone()
        .map(|knobs| knobs.resolve(family))
}

/// The last level's file encoding in this process ([`LAST_WIRE_ENV`]).
pub fn last_wire() -> Result<WireFormat, String> {
    wire_word(env_word(LAST_WIRE_ENV)?.as_deref())
}

/// The first bytes of a last-level file.
pub const LAST_FILE_MAGIC: [u8; 4] = *b"LVLF";

/// The last-level file's version: the header's layout and what its codes
/// mean. A reader reads this version alone.
pub const LAST_FILE_VERSION: u8 = 1;

/// The opt-in file with the statement's cluster leaf map.
pub const CLUSTER_LAST_FILE_VERSION: u8 = 2;

/// The header's length: the magic, the version, the format's code and the
/// encoding's code.
pub const LAST_FILE_HEADER: usize = LAST_FILE_MAGIC.len() + 3;

/// An encoding's code in the file header.
fn wire_code(wire: WireFormat) -> u8 {
    match wire {
        WireFormat::Rkyv => 0,
        WireFormat::Compact => 1,
        WireFormat::Dedup => 2,
    }
}

/// The last level's proof, made in `format`, as a file in `wire`'s encoding:
/// the header ([`LAST_FILE_MAGIC`], [`LAST_FILE_VERSION`], `format`'s code,
/// the encoding's code), then the rkyv archive or the compact bytes
/// ([`super::wire`]).
pub fn last_level_file(
    proof: &super::wlfm::WlfmProof,
    format: LastLevel,
    wire: WireFormat,
) -> Result<Vec<u8>, String> {
    encode_file(proof, format, wire, None)
}

pub(crate) fn encode_file(
    proof: &super::wlfm::WlfmProof,
    format: LastLevel,
    wire: WireFormat,
    map: Option<&super::cluster_levels::LeafMap>,
) -> Result<Vec<u8>, String> {
    let code = format
        .code()
        .ok_or_else(|| format!("the last level's format ({format}) has no file code"))?;
    let body = match super::wire::encode_as(proof, wire) {
        Some(bytes) => bytes,
        None => rkyv::to_bytes::<rkyv::rancor::Error>(proof)
            .map(|b| b.to_vec())
            .map_err(|e| format!("the last level's archive: {e}"))?,
    };
    let mut file = Vec::with_capacity(LAST_FILE_HEADER + body.len());
    file.extend_from_slice(&LAST_FILE_MAGIC);
    let version = if map.is_some() {
        CLUSTER_LAST_FILE_VERSION
    } else {
        LAST_FILE_VERSION
    };
    file.extend_from_slice(&[version, code, wire_code(wire)]);
    if let Some(map) = map {
        let bytes = map.as_bytes();
        let len = u32::try_from(bytes.len()).map_err(|_| "leaf map length overflows u32")?;
        file.extend_from_slice(&len.to_le_bytes());
        file.extend_from_slice(bytes);
        file.extend_from_slice(&leaf_map_checksum(bytes));
    }
    file.extend_from_slice(&body);
    Ok(file)
}

/// The proof a last-level file holds ([`last_level_file`]), for a verifier
/// that checks the last level's format `format`. Refused before anything is
/// decoded unless the file opens with the magic, is of [`LAST_FILE_VERSION`]
/// and names `format`'s code: a file made in another format is refused with
/// both formats named. A compact body must be exactly one proof's encoding;
/// an archive body is validated by rkyv and is not held to one encoding. The
/// proof read is then verified in full by the caller; the header decides
/// nothing the verifier checks.
pub fn read_last_level_file(
    bytes: &[u8],
    format: LastLevel,
) -> Result<super::wlfm::WlfmProof, String> {
    let (_, wire, body) = file_header(bytes, format, false)?;
    decode_body(wire, body)
}

/// V2 checks the map before handing any bytes to a proof decoder.
pub(crate) fn cluster_file_parts(
    bytes: &[u8],
    format: LastLevel,
    groups: usize,
) -> Result<(Option<super::cluster_levels::LeafMap>, u8, &[u8]), String> {
    let (version, wire, body) = file_header(bytes, format, true)?;
    if version == LAST_FILE_VERSION {
        return Ok((None, wire, body));
    }
    let (len, remaining) = body
        .split_at_checked(4)
        .ok_or("truncated leaf map length")?;
    let len = u32::from_le_bytes(len.try_into().expect("four bytes")) as usize;
    if len != groups {
        return Err(format!("leaf map length {len}, expected {groups} groups"));
    }
    let (map, remaining) = remaining
        .split_at_checked(len)
        .ok_or("truncated leaf map")?;
    let (checksum, proof) = remaining
        .split_at_checked(32)
        .ok_or("truncated leaf map checksum")?;
    if checksum != leaf_map_checksum(map) {
        return Err("altered leaf map: checksum mismatch".into());
    }
    Ok((
        Some(super::cluster_levels::LeafMap::decode(map, groups)?),
        wire,
        proof,
    ))
}

fn leaf_map_checksum(bytes: &[u8]) -> [u8; 32] {
    use digest::Digest;
    crypto::hash::blake3::chain::Blake3Std::digest(bytes).into()
}

fn file_header(bytes: &[u8], format: LastLevel, cluster: bool) -> Result<(u8, u8, &[u8]), String> {
    let want = format
        .code()
        .ok_or_else(|| format!("the verifier's last-level format ({format}) has no file code"))?;
    let Some((header, body)) = bytes.split_at_checked(LAST_FILE_HEADER) else {
        return Err("not a last-level file: shorter than its header".into());
    };
    if header[..LAST_FILE_MAGIC.len()] != LAST_FILE_MAGIC {
        return Err("not a last-level file: no LVLF header".into());
    }
    let (version, code, wire) = (header[4], header[5], header[6]);
    if version != LAST_FILE_VERSION && !(cluster && version == CLUSTER_LAST_FILE_VERSION) {
        return Err(format!(
            "last-level file version {version}, this verifier reads {LAST_FILE_VERSION}"
        ));
    }
    if code != want {
        let made = LastLevel::from_code(code)
            .map_or_else(|| "a code no format has".to_string(), |f| f.to_string());
        return Err(format!(
            "the last-level file's proof is in format {code} ({made}); this verifier checks \
             format {want} ({format})"
        ));
    }
    Ok((version, wire, body))
}

pub(crate) fn decode_body(wire: u8, body: &[u8]) -> Result<super::wlfm::WlfmProof, String> {
    match wire {
        0 => {
            super::wire::check_wire_len(body)?;
            let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(body.len());
            aligned.extend_from_slice(body);
            rkyv::from_bytes::<super::wlfm::WlfmProof, rkyv::rancor::Error>(&aligned)
                .map_err(|e| format!("the last level's archive: {e}"))
        }
        1 => super::wire::decode(body),
        2 => super::wire::decode_dedup(body),
        other => Err(format!("the last-level file's encoding {other} is unknown")),
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

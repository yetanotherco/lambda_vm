//! The W-LFM prover's knobs read from the environment. They choose how the
//! prove holds its tables, never what the proof is; [`super::wlfm`] holds the
//! format and reads no environment.

/// `LAMBDA_VM_WLFM_NARROW_CHIPS=1`: the LFM chips' traces packed narrow
/// ([`multilinear::narrow::NarrowColumns`]) before the prove takes them, so
/// their columns go up at the bytes their words need and the card widens
/// them, as the lean tables' do. Off by default; a prover's choice, never the
/// proof's: the card widens the same words. Read once a process.
pub const NARROW_CHIPS_ENV: &str = "LAMBDA_VM_WLFM_NARROW_CHIPS";

/// Whether the prove packs the LFM chips narrow: [`NARROW_CHIPS_ENV`], or a
/// test's override.
pub fn narrow_chips() -> bool {
    #[cfg(test)]
    if let Some(on) = super::wlfm::test_format::narrow_chips() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(NARROW_CHIPS_ENV).is_ok_and(|v| v.trim() == "1"))
}

/// `LAMBDA_VM_WLFM_CHIP_RECORDS=1`: the LFM chips but `LFM_HASH` held as their
/// records ([`super::chip_records`]) instead of filled on the host: the fill
/// keeps the executor's value records, the build packs each chip table's
/// preprocessed prefix narrow once, and the card writes both into the table's
/// columns as its group uploads it. Off by default (and for any word but
/// `1`); a prover's choice, never the proof's: the card writes the words the
/// host's trace holds. Read once a process.
pub const CHIP_RECORDS_ENV: &str = "LAMBDA_VM_WLFM_CHIP_RECORDS";

/// Whether a knob's word turns it on: `1` (spaces around it ignored), and
/// nothing else.
pub fn knob_word_on(word: Option<&str>) -> bool {
    word.is_some_and(|v| v.trim() == "1")
}

/// Whether the W-LFM prover holds the chips as their records:
/// [`CHIP_RECORDS_ENV`], or a test's override.
pub fn chip_records() -> bool {
    #[cfg(test)]
    if let Some(on) = super::wlfm::test_format::chip_records() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| knob_word_on(std::env::var(CHIP_RECORDS_ENV).ok().as_deref()))
}

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

/// The environment's word for knob `name` (`None` when unset or not
/// unicode): the one place the W-LFM knobs read the environment, so the
/// format module ([`super::wlfm`]) reads none.
pub fn env_word(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Whether a knob's word turns it on: `1` (spaces around it ignored), and
/// nothing else.
pub fn knob_word_on(word: Option<&str>) -> bool {
    word.is_some_and(|v| v.trim() == "1")
}

/// Whether a default-on knob's word turns it off: `0` (spaces around it
/// ignored), and nothing else; unset, empty or any other word leaves it on.
pub fn knob_word_off(word: Option<&str>) -> bool {
    word.is_some_and(|v| v.trim() == "0")
}

/// A two-card chain knob's word: `Some(true)` for `1`, `Some(false)` for
/// `0` (spaces around either ignored); unset, empty or any other word `None`,
/// the default, on exactly where the process proves over two cards
/// ([`two_cards`]).
pub fn knob_word_two_card(word: Option<&str>) -> Option<bool> {
    match word.map(str::trim) {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    }
}

/// Whether this process proves over two cards: the base's two-card split
/// (`LAMBDA_VM_TWO_CARD_SPLIT`) or the levels' (`LAMBDA_VM_TWO_CARD_LEVELS`)
/// places a second driver. Both default to `auto`, so this is true on a box
/// with two cards or more; false with one card and without a device build.
/// The levels' chain knobs default to it: they pay where the base's phase B
/// and the levels' split over two cards, and one card proves as before.
pub fn two_cards() -> bool {
    multilinear::gpu::two_card_split_on() || multilinear::gpu::two_card_levels_on()
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

/// `LAMBDA_VM_WLFM_LEAN_BESIDE=1`: the level driver fills level 1's lean round
/// tables while level 1 executes, a wave at a time on a thread of its own
/// ([`super::b3l_tables::LeanFillBeside`]), so after the execution only the
/// head chunks and the byte table are left to fill. Taken only when the round
/// chunks are held as their records (`LAMBDA_VM_B3L_RECORDS=1`). Off by
/// default (and for any word but `1`); a prover's choice, never the proof's:
/// the tables are the records path's, column for column. Read once a process.
pub const LEAN_BESIDE_ENV: &str = "LAMBDA_VM_WLFM_LEAN_BESIDE";

/// `LAMBDA_VM_LEVELS_TAIL`: the levels' tail after the base taken off the
/// path in one switch: level 1's lean round tables filled beside its
/// execution ([`LEAN_BESIDE_ENV`]'s fill), and every card reading the level
/// driver makes inside the levels (the base's end, each level committed and
/// proved) taken without trimming the pool, which keeps its pages for the
/// levels' own allocations (each still trims on a failed allocation). The
/// reading at the clock's start is unchanged. On by default; `0` turns it off
/// (any other word leaves it on); the same bytes either way. Read once a
/// process.
pub const TAIL_ENV: &str = "LAMBDA_VM_LEVELS_TAIL";

/// [`TAIL_ENV`]'s word: on unless it is `0`.
pub fn tail_word(word: Option<&str>) -> bool {
    !knob_word_off(word)
}

/// A test's [`tail`], process wide (0 the knob, 1 off, 2 on).
static TAIL_FOR_TEST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Tests only: [`tail`] forced for the whole process (`None` restores the
/// knob). A test that sets it runs alone.
#[doc(hidden)]
pub fn set_tail_for_test(on: Option<bool>) {
    let v = match on {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    TAIL_FOR_TEST.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`TAIL_ENV`] is on (or a test's override).
pub fn tail() -> bool {
    match TAIL_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| tail_word(std::env::var(TAIL_ENV).ok().as_deref()))
}

/// `LAMBDA_VM_LEVELS_UNTRIMMED=1`: [`TAIL_ENV`]'s second half alone, every card
/// reading inside the levels taken without trimming the pool; the lean fill
/// as its own knob says. Off by default (and for any word but `1`); the same
/// bytes. Read once a process.
pub const UNTRIMMED_ENV: &str = "LAMBDA_VM_LEVELS_UNTRIMMED";

/// A test's [`untrimmed_only`], process wide (0 the knob, 1 off, 2 on).
static UNTRIMMED_FOR_TEST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Tests only: [`UNTRIMMED_ENV`] forced for the whole process (`None`
/// restores the knob). A test that sets it runs alone.
#[doc(hidden)]
pub fn set_untrimmed_for_test(on: Option<bool>) {
    let v = match on {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    UNTRIMMED_FOR_TEST.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`UNTRIMMED_ENV`] alone is on (or a test's override).
fn untrimmed_only() -> bool {
    match UNTRIMMED_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| knob_word_on(std::env::var(UNTRIMMED_ENV).ok().as_deref()))
}

/// Whether the level driver reads the card inside the levels without
/// trimming the pool: [`UNTRIMMED_ENV`] or [`TAIL_ENV`].
pub fn levels_untrimmed() -> bool {
    untrimmed_only() || tail()
}

/// Whether the level driver fills level 1's lean round tables beside its
/// execution: [`LEAN_BESIDE_ENV`], [`TAIL_ENV`], or a test's override.
pub fn lean_beside() -> bool {
    #[cfg(test)]
    if let Some(on) = super::wlfm::test_format::lean_beside() {
        return on;
    }
    if tail() {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| knob_word_on(std::env::var(LEAN_BESIDE_ENV).ok().as_deref()))
}

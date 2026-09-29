//! ★ `LAMBDA_VM_LFM_PROVER` — which prover the recursion's LFM proofs (wraps,
//! nodes, the root) are proved with.
//!
//! ```text
//! LAMBDA_VM_LFM_PROVER=stark   (default) one STARK per table, today's proofs
//! LAMBDA_VM_LFM_PROVER=whir    the base's stacked-WHIR prover
//!                              (`lfm::whir_proof`), verified in-guest by W-legs
//! ```
//!
//! Read ONCE per process and cached, so a tree cannot switch provers halfway
//! and produce a parent emitted for one kind of child over a proof of the other.
//!
//! # The three decisions of `whir_hash_knob`, taken the same way
//!
//! **An unknown value ABORTS.** `LAMBDA_VM_LFM_PROVER=wir` falling through to
//! `stark` would produce a perfectly valid tree under the prover the operator
//! was trying to move away from, and a measurement taken that way looks like
//! the WHIR arm and is not.
//!
//! **The banner prints on EVERY setting, including the default,** so its
//! absence in a log is a fact about the run (this code was never reached)
//! rather than an ambiguity about which arm ran.
//!
//! **It is read here and nowhere else.** Every consumer asks [`selected`].
//!
//! An EMPTY value reads as unset, as the ZF knobs do (`lfm::airs::env_switch`):
//! launchers export variables they leave blank, and blank is not a spelling of
//! either prover.

use std::sync::OnceLock;

/// Which prover this process's LFM proofs are proved with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Setting {
    /// One STARK per table (`lfm::proof::lfm_prove`), the default.
    Stark,
    /// The stacked-WHIR multilinear prover (`lfm::whir_proof::lfm_prove_whir`).
    Whir,
}

impl Setting {
    /// The name the banner prints.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Stark => "stark",
            Self::Whir => "whir",
        }
    }
}

/// The environment variable that selects it.
pub const ENV: &str = "LAMBDA_VM_LFM_PROVER";

/// The default when [`ENV`] is unset or empty: today's proofs.
pub const DEFAULT: Setting = Setting::Stark;

/// What [`ENV`] accepts, and what an error message lists.
const ACCEPTED: &[(&str, Setting)] = &[("stark", Setting::Stark), ("whir", Setting::Whir)];

/// ★ The setting for this process, read once and cached.
///
/// Prints the banner on the first call. Aborts on an unrecognised value — see
/// the module header.
pub fn selected() -> Setting {
    static SETTING: OnceLock<Setting> = OnceLock::new();
    *SETTING.get_or_init(|| {
        let raw = std::env::var(ENV).ok();
        let setting = setting_of(raw.as_deref()).unwrap_or_else(|| {
            let accepted: Vec<&str> = ACCEPTED.iter().map(|(name, _)| *name).collect();
            // eprintln then abort rather than a panic: this is a configuration
            // error at startup, and the operator needs the accepted values, not
            // a backtrace through the prover.
            eprintln!(
                "{ENV}={:?} is not a prover this path knows. Accepted: {}.",
                raw.unwrap_or_default(),
                accepted.join(", ")
            );
            std::process::abort()
        });
        // Always, including the default — see the module header.
        println!("★ LFM PROVER: {}", setting.name());
        setting
    })
}

/// [`ENV`]'s reading of a raw value: unset or empty is [`DEFAULT`], an accepted
/// spelling (case-insensitive) its setting, anything else `None`.
fn setting_of(raw: Option<&str>) -> Option<Setting> {
    match raw.map(str::trim) {
        None | Some("") => Some(DEFAULT),
        Some(value) => {
            let lowered = value.to_ascii_lowercase();
            ACCEPTED
                .iter()
                .find(|(name, _)| *name == lowered)
                .map(|(_, setting)| *setting)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unset and empty are today's prover; the two spellings map where they
    /// say, case-insensitively.
    ///
    /// Tested through [`setting_of`] rather than [`selected`], which caches in
    /// a process-global `OnceLock` and aborts on a bad value.
    #[test]
    fn the_default_is_stark_and_each_spelling_maps() {
        assert_eq!(setting_of(None), Some(Setting::Stark));
        assert_eq!(setting_of(Some("")), Some(Setting::Stark));
        assert_eq!(setting_of(Some("  ")), Some(Setting::Stark));
        assert_eq!(setting_of(Some("stark")), Some(Setting::Stark));
        assert_eq!(setting_of(Some("whir")), Some(Setting::Whir));
        assert_eq!(setting_of(Some("WHIR")), Some(Setting::Whir));
        assert_eq!(setting_of(Some(" whir ")), Some(Setting::Whir));
    }

    /// ★ The near-misses do NOT parse — the list that would otherwise prove
    /// under STARK and report itself as the WHIR arm.
    #[test]
    fn a_near_miss_is_not_silently_accepted() {
        for raw in ["wir", "whir2", "multilinear", "fri", "1", "0", "starks"] {
            assert_eq!(setting_of(Some(raw)), None, "{raw:?} must not parse");
        }
    }
}

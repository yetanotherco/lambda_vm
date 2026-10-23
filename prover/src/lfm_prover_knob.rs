//! ★ `LAMBDA_VM_LFM_PROVER` — which prover the recursion's LFM proofs (wraps,
//! nodes, the root) are proved with.
//!
//! ```text
//! LAMBDA_VM_LFM_PROVER=whir    (default) the base's stacked-WHIR prover
//!                              (`lfm::whir_proof`), verified in-guest by W-legs
//! LAMBDA_VM_LFM_PROVER=stark   one STARK per table: the opt-out, the recursion
//!                              as it was before pure WHIR
//! ```
//!
//! The default is pure WHIR (D-WHIR W4, an ABBA at 4150afab6: the block's whole
//! run 56.65 s against 60.70 s). Its two companions default with it: the prefix
//! in the prepared stack only ([`PREP_ENV`] = `prepared`) and the wide level 1
//! ([`WIDE_ENV`] on under `whir`).
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
    /// One STARK per table (`lfm::proof::lfm_prove`), the opt-out.
    Stark,
    /// The stacked-WHIR multilinear prover (`lfm::whir_proof::lfm_prove_whir`),
    /// the default.
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

/// The default when [`ENV`] is unset or empty: pure WHIR.
pub const DEFAULT: Setting = Setting::Whir;

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

/// `LAMBDA_VM_LFM_WHIR_PREP` — where a W-LFM proof commits each table's
/// preprocessed prefix (D-WHIR §2.4): `prepared` (policy B, the default) in the
/// prepared stack only, `both` (policy A) in the main stack AND the prepared
/// stack. A W-LFM FORMAT choice, stamped on the program's artifacts at build time
/// and folded into its `program_id_w`, so a verifier takes it from the
/// artifacts and never from a proof.
pub const PREP_ENV: &str = "LAMBDA_VM_LFM_WHIR_PREP";

/// The W-LFM prefix policy for this process, read once and bannered on every
/// setting; an unknown value aborts.
pub fn prep_policy() -> crate::lfm::whir_proof::PrepPolicy {
    use crate::lfm::whir_proof::PrepPolicy;
    static POLICY: OnceLock<PrepPolicy> = OnceLock::new();
    *POLICY.get_or_init(|| {
        let raw = std::env::var(PREP_ENV).ok();
        let policy = prep_policy_of(raw.as_deref()).unwrap_or_else(|| {
            eprintln!(
                "{PREP_ENV}={:?} is not a prefix policy. Accepted: both, prepared.",
                raw.unwrap_or_default()
            );
            std::process::abort()
        });
        println!("★ LFM WHIR PREP: {}", policy.name());
        policy
    })
}

/// `LAMBDA_VM_LFM_ROOT_PROVER=stark|whir` — which prover proves the tree's
/// BLOCK-ARTIFACT ROOT, when it is not the rest of the tree's. Unset or empty:
/// the root follows [`selected`]. It exists for the D-WHIR W2 stage, which runs
/// a W-LFM tree under a STARK root so the artifact's format does not move while
/// the interior's is measured; W3 unsets it.
pub const ROOT_ENV: &str = "LAMBDA_VM_LFM_ROOT_PROVER";

/// The root's prover for this process, read once and bannered on every
/// setting; an unknown value aborts.
///
/// ⛔ A W-LFM ROOT OVER A STARK TREE IS REFUSED: its legs would be STARK legs
/// and nothing of pure WHIR would be under measurement, so the combination is a
/// mislabeled arm rather than an experiment.
pub fn root_selected() -> Setting {
    static SETTING: OnceLock<Setting> = OnceLock::new();
    *SETTING.get_or_init(|| {
        let raw = std::env::var(ROOT_ENV).ok();
        let tree = selected();
        let setting = match raw.as_deref().map(str::trim) {
            None | Some("") => tree,
            Some(_) => setting_of(raw.as_deref()).unwrap_or_else(|| {
                eprintln!(
                    "{ROOT_ENV}={:?} is not a prover this path knows. Accepted: stark, whir.",
                    raw.clone().unwrap_or_default()
                );
                std::process::abort()
            }),
        };
        if tree == Setting::Stark && setting == Setting::Whir {
            eprintln!(
                "{ROOT_ENV}=whir under {ENV}=stark: a W-LFM root over STARK children has \
                 STARK legs and measures nothing of pure WHIR. Refused."
            );
            std::process::abort()
        }
        println!("★ LFM ROOT PROVER: {}", setting.name());
        setting
    })
}

/// `LAMBDA_VM_LFM_WIDE=off|on` — whether a W-LFM tree's level 1 verifies its
/// epochs DIRECTLY (wide nodes, `lfm::whir_wide`, D-WHIR §7 L1) instead of over
/// one wrap proof per epoch. `on` puts `k` = the tree's fan-in epochs in each
/// node, so level 1's grouping, the root's fold shape and every level above are
/// the wrap tree's. Unset or empty FOLLOWS THE TREE'S PROVER: on under `whir`
/// (the default), off under `stark`, whose tree has wraps.
pub const WIDE_ENV: &str = "LAMBDA_VM_LFM_WIDE";

/// Whether this process's WHIR tree builds wide level-1 nodes, read once and
/// bannered on every setting; an unknown value aborts.
///
/// ⛔ `on` UNDER `LAMBDA_VM_LFM_PROVER=stark` IS REFUSED: the wide node is a
/// pure-WHIR lever, measured against the W-LFM tree, and a STARK-proved wide
/// tree would be an arm nobody pre-registered.
pub fn wide_selected() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var(WIDE_ENV).ok();
        let on = wide_of(raw.as_deref())
            .unwrap_or_else(|| {
                eprintln!(
                    "{WIDE_ENV}={:?} is not a setting. Accepted: off, on.",
                    raw.clone().unwrap_or_default()
                );
                std::process::abort()
            })
            .unwrap_or(selected() == Setting::Whir);
        if on && selected() == Setting::Stark {
            eprintln!(
                "{WIDE_ENV}=on under {ENV}=stark: wide level-1 nodes are a W-LFM lever. \
                 Refused."
            );
            std::process::abort()
        }
        println!(
            "★ LFM WIDE: {}",
            if on {
                "on (level 1 verifies its epochs directly, k = the tree's fan-in)"
            } else {
                "off"
            }
        );
        on
    })
}

/// [`WIDE_ENV`]'s reading: `off` and `on` are themselves; unset and empty are
/// `Some(None)`, the tree prover's default; anything else does not parse.
fn wide_of(raw: Option<&str>) -> Option<Option<bool>> {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("") => Some(None),
        Some("off") => Some(Some(false)),
        Some("on") => Some(Some(true)),
        Some(_) => None,
    }
}

/// [`PREP_ENV`]'s reading: unset or empty is policy B (`prepared`).
fn prep_policy_of(raw: Option<&str>) -> Option<crate::lfm::whir_proof::PrepPolicy> {
    use crate::lfm::whir_proof::PrepPolicy;
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("") | Some("prepared") => Some(PrepPolicy::PreparedOnly),
        Some("both") => Some(PrepPolicy::Both),
        Some(_) => None,
    }
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

    /// Unset and empty are the default, pure WHIR; the two spellings map where
    /// they say, case-insensitively.
    ///
    /// Tested through [`setting_of`] rather than [`selected`], which caches in
    /// a process-global `OnceLock` and aborts on a bad value.
    #[test]
    fn the_default_is_whir_and_each_spelling_maps() {
        assert_eq!(setting_of(None), Some(Setting::Whir));
        assert_eq!(setting_of(Some("")), Some(Setting::Whir));
        assert_eq!(setting_of(Some("  ")), Some(Setting::Whir));
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

    /// The prefix policy: unset, empty and `prepared` are B, the default;
    /// `both` is A; anything else does not parse.
    #[test]
    fn the_prep_policy_parses_its_two_spellings_only() {
        use crate::lfm::whir_proof::PrepPolicy;
        assert_eq!(prep_policy_of(None), Some(PrepPolicy::PreparedOnly));
        assert_eq!(prep_policy_of(Some("")), Some(PrepPolicy::PreparedOnly));
        assert_eq!(prep_policy_of(Some("both")), Some(PrepPolicy::Both));
        assert_eq!(
            prep_policy_of(Some("Prepared")),
            Some(PrepPolicy::PreparedOnly)
        );
        for raw in ["a", "b", "B", "prep", "main", "1"] {
            assert_eq!(prep_policy_of(Some(raw)), None, "{raw:?} must not parse");
        }
    }

    /// The wide switch: unset and empty defer to the tree's prover; `off` and
    /// `on` are themselves, in any case; a number or a near miss does not parse
    /// (`k` is the tree's fan-in, never a value of its own).
    #[test]
    fn the_wide_switch_parses_off_and_on_only() {
        assert_eq!(wide_of(None), Some(None));
        assert_eq!(wide_of(Some("")), Some(None));
        assert_eq!(wide_of(Some("off")), Some(Some(false)));
        assert_eq!(wide_of(Some(" ON ")), Some(Some(true)));
        for raw in ["1", "0", "3", "yes", "true", "wide", "onn"] {
            assert_eq!(wide_of(Some(raw)), None, "{raw:?} must not parse");
        }
    }
}

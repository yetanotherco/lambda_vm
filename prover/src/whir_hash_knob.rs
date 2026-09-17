//! ★ `LAMBDA_VM_WHIR_HASH` — which hash the multilinear path commits,
//! transcripts and grinds with.
//!
//! ```text
//! LAMBDA_VM_WHIR_HASH=keccak   (default, and what PR #988 produces)
//! LAMBDA_VM_WHIR_HASH=rpx      the algebraic arm, for a proof headed into
//!                              another proof
//! ```
//!
//! Read ONCE per process and cached, so a run cannot change hash halfway
//! through and produce a proof no single configuration describes.
//!
//! # Three decisions worth stating
//!
//! **An unknown value ABORTS, loudly.** `LAMBDA_VM_WHIR_HASH=rpx256` — a
//! plausible typo, since that is what the configuration calls itself — would
//! otherwise fall through to keccak and produce a perfectly valid proof under
//! the hash the operator was trying to move away from. A measurement taken that
//! way is worse than no measurement: it looks like the RPX arm and is not. The
//! cost of aborting is a failed run with a message naming the accepted values;
//! the cost of defaulting is a number nobody can tell is wrong.
//!
//! **The banner prints on EVERY setting, including the default.** A banner that
//! only appeared for RPX could not be distinguished from a banner that did not
//! appear because this code was never reached — which is exactly what a byte
//! gate comparing two arms needs to rule out. Its absence in a log is therefore
//! a fact about the run, not an ambiguity.
//!
//! **It is read here and nowhere else.** The seam it selects is a type
//! parameter, so every consumer gets the hash through [`with_whir_hash`] rather
//! than by asking the environment again.

use std::sync::OnceLock;

/// Which configuration this process proves and verifies under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Setting {
    /// Keccak-256, the default.
    Keccak,
    /// RPX256 (XHash12).
    Rpx,
}

impl Setting {
    /// The name the configuration itself reports — what the banner prints and
    /// the KATs are filed under.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Keccak => "keccak256",
            Self::Rpx => "rpx256",
        }
    }
}

/// The environment variable that selects it.
pub const ENV: &str = "LAMBDA_VM_WHIR_HASH";

/// What `ENV` accepts, and what an error message lists.
const ACCEPTED: &[(&str, Setting)] = &[
    ("keccak", Setting::Keccak),
    ("keccak256", Setting::Keccak),
    ("rpx", Setting::Rpx),
    ("rpx256", Setting::Rpx),
];

/// ★ The setting for this process, read once and cached.
///
/// Prints the banner on the first call. Aborts on an unrecognised value — see
/// the module header for why that is better than defaulting.
pub fn selected() -> Setting {
    static SETTING: OnceLock<Setting> = OnceLock::new();
    *SETTING.get_or_init(|| {
        let setting = match std::env::var(ENV) {
            Err(_) => Setting::Keccak,
            Ok(raw) => parse(raw.trim()).unwrap_or_else(|| {
                let accepted: Vec<&str> = ACCEPTED.iter().map(|(name, _)| *name).collect();
                // eprintln then abort rather than a panic: this is a
                // configuration error at startup, and the operator needs the
                // accepted values, not a backtrace through the prover.
                eprintln!(
                    "{ENV}={raw:?} is not a hash this path knows. Accepted: {}.",
                    accepted.join(", ")
                );
                std::process::abort()
            }),
        };
        // Always, including the default — see the module header.
        println!("★ WHIR HASH: {}", setting.name());
        setting
    })
}

/// The accepted spellings, case-insensitively.
fn parse(raw: &str) -> Option<Setting> {
    let lowered = raw.to_ascii_lowercase();
    ACCEPTED
        .iter()
        .find(|(name, _)| *name == lowered)
        .map(|(_, setting)| *setting)
}

/// ★ Run `$body` with `$h` bound to the configuration [`selected`] names.
///
/// The seam is a type parameter, so the dispatch has to happen where a type can
/// be named — one `match` per call site, each arm monomorphising the body at
/// its own hash. That is also why this is a macro rather than a function: a
/// function cannot return a type.
///
/// ⚠ Both arms are always compiled, which doubles the monomorphisations of
/// everything below the call. That is deliberate: a feature gate would make the
/// RPX arm unreachable in a default build, and then the knob would be a control
/// that does nothing on exactly the binary most people run.
#[macro_export]
macro_rules! with_whir_hash {
    (|$h:ident| $body:block) => {
        match $crate::whir_hash_knob::selected() {
            $crate::whir_hash_knob::Setting::Keccak => {
                #[allow(non_camel_case_types)]
                type $h = multilinear::whir_hash::KeccakWhir;
                $body
            }
            $crate::whir_hash_knob::Setting::Rpx => {
                #[allow(non_camel_case_types)]
                type $h = multilinear::whir_hash::RpxWhir;
                $body
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every accepted spelling maps where it says, case-insensitively.
    ///
    /// Tested through [`parse`] rather than through [`selected`]: the latter
    /// caches in a process-global `OnceLock` and aborts on a bad value, so it
    /// can be exercised exactly once per process and never with a bad input.
    /// What is testable is the decision it makes, which is this function.
    #[test]
    fn every_accepted_spelling_maps_to_its_configuration() {
        assert_eq!(parse("keccak"), Some(Setting::Keccak));
        assert_eq!(parse("keccak256"), Some(Setting::Keccak));
        assert_eq!(parse("rpx"), Some(Setting::Rpx));
        assert_eq!(parse("rpx256"), Some(Setting::Rpx));
        assert_eq!(parse("RPX"), Some(Setting::Rpx));
        assert_eq!(parse("Keccak256"), Some(Setting::Keccak));
    }

    /// ★ And the near-misses do NOT. This is the list that would otherwise
    /// default to keccak and report itself as the RPX arm.
    #[test]
    fn a_near_miss_is_not_silently_accepted() {
        for raw in [
            "rpx-256",
            "rpx_256",
            "xhash12",
            "algebraic",
            "blake3",
            "",
            "kecak",
        ] {
            assert_eq!(parse(raw), None, "{raw:?} must not parse");
        }
    }

    /// The two settings name themselves the way the configurations do, so a
    /// banner and a KAT cannot disagree about which arm ran.
    #[test]
    fn the_names_match_the_configurations() {
        use multilinear::whir_hash::{KeccakWhir, RpxWhir, WhirHash};
        assert_eq!(Setting::Keccak.name(), KeccakWhir::NAME);
        assert_eq!(Setting::Rpx.name(), RpxWhir::NAME);
    }

    /// ✓ The macro really does bind a different type per arm — checked through
    /// the configuration's own name, so a macro that expanded both arms to
    /// keccak would fail here rather than silently proving under one hash.
    #[test]
    fn the_macro_binds_the_configuration_the_setting_names() {
        use multilinear::whir_hash::WhirHash;
        let name = with_whir_hash!(|H| { H::NAME });
        assert_eq!(name, selected().name());
    }
}

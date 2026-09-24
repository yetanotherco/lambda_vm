//! ★ The proof FORMAT this process proves under — the ZF campaign's levers.
//!
//! ```text
//! LAMBDA_VM_ZF_CAP         off | auto | 0..=16    Merkle cap, every univariate STARK tree (S1)
//! LAMBDA_VM_ZF_WHIR_CAP    off | auto | 0..=16    Merkle cap, every WHIR chain tree (W1)
//! LAMBDA_VM_ZF_FRI         pair | dp              FRI fold schedule (S3)
//! LAMBDA_VM_ZF_ONE_ROW     0 | 1 | auto           one-row trace openings (S2)
//! LAMBDA_VM_ZF_WHIR_FOLDS  uniform4 | first5 | first6   WHIR first-round fold (W2)
//! ```
//!
//! Every unset knob is today's format, so an unconfigured run proves exactly
//! what it proved before this module existed.
//!
//! # Where the format goes
//!
//! Parsed ONCE per process ([`ZfFormat::global`]) and read only where a
//! production format value is built — [`crate::lfm::proof::aggregation_wrap_options`]
//! (every LFM proof: wraps, nodes, the root), [`crate::multilinear_prove::chain_config`]
//! (the WHIR base proofs) and [`crate::lfm::proof::block_base_options`] (the
//! STARK block's base epochs). From there it travels inside the option types
//! the crypto crates already take — `stark::ProofOptions` and
//! `multilinear::ChainConfig` — which never read the environment themselves.
//! Host verification reads nothing global: it uses the options it is given.
//! Tests build those option values explicitly; none sets the environment.
//!
//! # Three rules, as in `whir_hash_knob`
//!
//! **An unknown value ABORTS.** A typo that fell back to the default would
//! produce a valid default-format proof labelled as the lever — a measurement
//! that looks like arm B and is arm A.
//!
//! **A lever this build does not implement ABORTS too.** The fields exist
//! before the levers do (so the option structs and this banner are stable
//! while the campaign lands them), and a knob set on a build that only parses
//! it would print a non-default format and prove the default one. Each lane
//! flips its `*_IMPLEMENTED` constant when its lever is real.
//!
//! **The banner prints on every setting, including the default**:
//! `ZF FORMAT: cap=off whir_cap=off fri=pair one_row=0 whir_folds=uniform4`.
//! Its absence in a log is then a fact about the run, not an ambiguity.

use std::sync::OnceLock;

use multilinear::whir_chain::{ChainConfig, ChainFormat, FirstFold, WhirFolds};
use stark::proof::options::{CapPolicy, FriMode, OneRowMode, ProofFormat, ProofOptions};

/// The knob names, in banner order.
pub const ENV_CAP: &str = "LAMBDA_VM_ZF_CAP";
pub const ENV_WHIR_CAP: &str = "LAMBDA_VM_ZF_WHIR_CAP";
pub const ENV_FRI: &str = "LAMBDA_VM_ZF_FRI";
pub const ENV_ONE_ROW: &str = "LAMBDA_VM_ZF_ONE_ROW";
pub const ENV_WHIR_FOLDS: &str = "LAMBDA_VM_ZF_WHIR_FOLDS";

/// The uniform WHIR schedule's fold, as production configures it
/// (`multilinear_prove::chain_config`); the banner spells the default
/// `uniform4` after it.
pub const PRODUCTION_WHIR_LOG_FOLDING: usize = 4;

/// One process's proof format. Every field's default is today's format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZfFormat {
    /// S1: the cap on every univariate STARK tree.
    pub cap: CapPolicy,
    /// W1: the cap on every WHIR chain tree.
    pub whir_cap: CapPolicy,
    /// S3: the FRI fold schedule.
    pub fri: FriMode,
    /// S2: one-row trace openings.
    pub one_row: OneRowMode,
    /// W2: the WHIR per-round fold schedule.
    pub whir_folds: WhirFolds,
}

impl ZfFormat {
    /// Today's format: every lever off.
    pub const DEFAULT: Self = Self {
        cap: CapPolicy::Off,
        whir_cap: CapPolicy::Off,
        fri: FriMode::Pair,
        one_row: OneRowMode::Off,
        whir_folds: WhirFolds::Uniform,
    };

    /// Parse the five knobs through `lookup` (the process environment in
    /// production, a map in tests). An unset knob is the default; a set one
    /// must be one of the accepted spellings (surrounding whitespace and case
    /// are ignored, as for `LAMBDA_VM_WHIR_HASH`).
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let mut format = Self::DEFAULT;
        let get = |name: &str| lookup(name).map(|raw| raw.trim().to_ascii_lowercase());
        if let Some(v) = get(ENV_CAP) {
            format.cap = parse_cap(ENV_CAP, &v)?;
        }
        if let Some(v) = get(ENV_WHIR_CAP) {
            format.whir_cap = parse_cap(ENV_WHIR_CAP, &v)?;
        }
        if let Some(v) = get(ENV_FRI) {
            format.fri = v
                .parse()
                .map_err(|()| format!("{ENV_FRI}={v:?}: expected `pair` or `dp`"))?;
        }
        if let Some(v) = get(ENV_ONE_ROW) {
            format.one_row = v
                .parse()
                .map_err(|()| format!("{ENV_ONE_ROW}={v:?}: expected `0`, `1` or `auto`"))?;
        }
        if let Some(v) = get(ENV_WHIR_FOLDS) {
            format.whir_folds = parse_whir_folds(&v)?;
        }
        Ok(format)
    }

    /// [`from_lookup`](Self::from_lookup) over the process environment. A
    /// variable that is set but not valid Unicode is an error, not "unset".
    pub fn from_env() -> Result<Self, String> {
        let non_unicode = std::cell::Cell::new(None);
        let format = Self::from_lookup(|name| match std::env::var(name) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                non_unicode.set(Some(name.to_string()));
                None
            }
        })?;
        match non_unicode.into_inner() {
            Some(name) => Err(format!("{name} is set but not valid Unicode")),
            None => Ok(format),
        }
    }

    /// The knobs set to a non-default value whose lever this build does not
    /// implement yet. Selecting one must fail: see the module header.
    pub fn unimplemented_levers(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if !self.cap.is_off() && !stark::proof::options::MERKLE_CAP_IMPLEMENTED {
            out.push(ENV_CAP);
        }
        if !self.whir_cap.is_off() && !multilinear::whir_chain::WHIR_CAP_IMPLEMENTED {
            out.push(ENV_WHIR_CAP);
        }
        if self.fri != FriMode::Pair && !stark::proof::options::FRI_MODE_IMPLEMENTED {
            out.push(ENV_FRI);
        }
        if self.one_row != OneRowMode::Off && !stark::proof::options::ONE_ROW_IMPLEMENTED {
            out.push(ENV_ONE_ROW);
        }
        if self.whir_folds != WhirFolds::Uniform && !multilinear::whir_chain::WHIR_FOLDS_IMPLEMENTED
        {
            out.push(ENV_WHIR_FOLDS);
        }
        out
    }

    /// ★ The format for this process, read once and cached.
    ///
    /// Prints the banner on the first call. Aborts on an unrecognised value
    /// or a lever this build does not implement — see the module header.
    pub fn global() -> &'static Self {
        static FORMAT: OnceLock<ZfFormat> = OnceLock::new();
        FORMAT.get_or_init(|| {
            let format = Self::from_env().unwrap_or_else(|e| {
                // eprintln then abort rather than a panic: a configuration
                // error at startup, and the operator needs the accepted
                // values, not a backtrace through the prover.
                eprintln!("ZF FORMAT: {e}");
                std::process::abort()
            });
            let missing = format.unimplemented_levers();
            if !missing.is_empty() {
                eprintln!(
                    "ZF FORMAT: {} set to a non-default value, but this build does not \
                     implement that lever yet ({})",
                    missing.join(", "),
                    format.banner()
                );
                std::process::abort()
            }
            // Always, including the default — see the module header.
            println!("{}", format.banner());
            println!("{}", format.whir_schedule_line());
            format
        })
    }

    /// `ZF FORMAT: cap=… whir_cap=… fri=… one_row=… whir_folds=…`, each value
    /// in the spelling its knob accepts.
    pub fn banner(&self) -> String {
        format!(
            "ZF FORMAT: cap={} whir_cap={} fri={} one_row={} whir_folds={}",
            self.cap,
            self.whir_cap,
            self.fri,
            self.one_row,
            whir_folds_name(&self.whir_folds)
        )
    }

    /// `ZF WHIR SCHEDULES: whir_folds=… n=20:[…] … n=25:[…]` — the fold
    /// schedule the WHIR base chains run at the production stack heights, so a
    /// log states the rounds it proved and not only the knob's name. Printed
    /// under the banner, on every setting.
    pub fn whir_schedule_line(&self) -> String {
        let config = crate::multilinear_prove::chain_config_under(self, &[(1, 25)]);
        let schedules = (20..=25)
            .map(|n| format!("n={n}:{:?}", config.schedule(n)).replace(' ', ""))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "ZF WHIR SCHEDULES: whir_folds={} q={} {schedules}",
            whir_folds_name(&self.whir_folds),
            config.num_queries
        )
    }

    /// The univariate part: what `stark::ProofOptions` carries.
    pub fn proof_format(&self) -> ProofFormat {
        ProofFormat {
            merkle_cap: self.cap,
            fri_mode: self.fri,
            one_row: self.one_row,
        }
    }

    /// The WHIR part: what `multilinear::ChainConfig` carries.
    pub fn chain_format(&self) -> ChainFormat {
        ChainFormat {
            cap: self.whir_cap,
            folds: self.whir_folds,
        }
    }

    /// Stamp this format's univariate fields onto `options`.
    pub fn apply_to_options(&self, options: &mut ProofOptions) {
        options.format = self.proof_format();
    }

    /// `options` with this format's univariate fields.
    pub fn options(&self, mut options: ProofOptions) -> ProofOptions {
        self.apply_to_options(&mut options);
        options
    }

    /// Stamp this format's WHIR fields onto `config`.
    pub fn apply_to_chain(&self, config: &mut ChainConfig) {
        config.format = self.chain_format();
    }

    /// `config` with this format's WHIR fields.
    pub fn chain(&self, mut config: ChainConfig) -> ChainConfig {
        self.apply_to_chain(&mut config);
        config
    }
}

fn parse_cap(name: &str, v: &str) -> Result<CapPolicy, String> {
    v.parse().map_err(|e| format!("{name}={v:?}: {e}"))
}

/// The first-round folds the knob accepts: the two arms RULINGS 15 builds.
///
/// ⚠ Not `first1..=first4`: a first fold narrower than the uniform one adds
/// rounds at some heights (Q would rise and the arms stop being comparable),
/// and `first4` IS `uniform4` under another statement word. Not `dp`: RULINGS
/// 15, no DP. Widening this list is a format decision, not a parser one.
pub const WHIR_FIRST_FOLDS: [usize; 2] = [5, 6];

/// `uniform4` | `first5` | `first6`.
fn parse_whir_folds(v: &str) -> Result<WhirFolds, String> {
    if v == format!("uniform{PRODUCTION_WHIR_LOG_FOLDING}") {
        return Ok(WhirFolds::Uniform);
    }
    WHIR_FIRST_FOLDS
        .iter()
        .find(|&&k| v == format!("first{k}"))
        .and_then(|&k| FirstFold::new(k))
        .map(WhirFolds::First)
        .ok_or_else(|| {
            format!(
                "{ENV_WHIR_FOLDS}={v:?}: expected `uniform{PRODUCTION_WHIR_LOG_FOLDING}`, {}",
                WHIR_FIRST_FOLDS
                    .iter()
                    .map(|k| format!("`first{k}`"))
                    .collect::<Vec<_>>()
                    .join(" or ")
            )
        })
}

fn whir_folds_name(folds: &WhirFolds) -> String {
    match folds {
        WhirFolds::Uniform => format!("uniform{PRODUCTION_WHIR_LOG_FOLDING}"),
        WhirFolds::First(k0) => format!("first{}", k0.get()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn parse(pairs: &[(&str, &str)]) -> Result<ZfFormat, String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        ZfFormat::from_lookup(|k| map.get(k).cloned())
    }

    #[test]
    fn nothing_set_is_todays_format() {
        let f = parse(&[]).unwrap();
        assert_eq!(f, ZfFormat::DEFAULT);
        assert_eq!(f, ZfFormat::default());
        assert_eq!(
            f.banner(),
            "ZF FORMAT: cap=off whir_cap=off fri=pair one_row=0 whir_folds=uniform4"
        );
        assert!(f.unimplemented_levers().is_empty());
    }

    #[test]
    fn the_default_spellings_parse_to_the_default() {
        let f = parse(&[
            (ENV_CAP, "off"),
            (ENV_WHIR_CAP, "0"),
            (ENV_FRI, "pair"),
            (ENV_ONE_ROW, "0"),
            (ENV_WHIR_FOLDS, "uniform4"),
        ])
        .unwrap();
        assert_eq!(f, ZfFormat::DEFAULT);
        assert!(f.unimplemented_levers().is_empty());
    }

    #[test]
    fn every_accepted_spelling_parses() {
        for (v, want) in [
            ("off", CapPolicy::Off),
            ("auto", CapPolicy::Auto),
            ("0", CapPolicy::Off),
            ("3", CapPolicy::Fixed(3)),
            ("16", CapPolicy::Fixed(16)),
            (" AUTO ", CapPolicy::Auto),
        ] {
            assert_eq!(parse(&[(ENV_CAP, v)]).unwrap().cap, want, "{v:?}");
            assert_eq!(parse(&[(ENV_WHIR_CAP, v)]).unwrap().whir_cap, want, "{v:?}");
        }
        assert_eq!(parse(&[(ENV_FRI, "dp")]).unwrap().fri, FriMode::Dp);
        assert_eq!(
            parse(&[(ENV_ONE_ROW, "1")]).unwrap().one_row,
            OneRowMode::On
        );
        assert_eq!(
            parse(&[(ENV_ONE_ROW, "auto")]).unwrap().one_row,
            OneRowMode::Auto
        );
        for k in [5, 6] {
            assert_eq!(
                parse(&[(ENV_WHIR_FOLDS, &format!("first{k}"))])
                    .unwrap()
                    .whir_folds,
                WhirFolds::First(FirstFold::new(k).unwrap())
            );
        }
        assert_eq!(
            parse(&[(ENV_WHIR_FOLDS, " FIRST6 ")]).unwrap().whir_folds,
            WhirFolds::First(FirstFold::new(6).unwrap())
        );
    }

    #[test]
    fn every_bad_spelling_is_refused_and_names_its_knob() {
        for (name, v) in [
            (ENV_CAP, ""),
            (ENV_CAP, "17"),
            (ENV_CAP, "on"),
            (ENV_CAP, "-1"),
            (ENV_CAP, "3.0"),
            (ENV_WHIR_CAP, "yes"),
            (ENV_FRI, "binary"),
            (ENV_FRI, ""),
            (ENV_ONE_ROW, "2"),
            (ENV_ONE_ROW, "on"),
            (ENV_WHIR_FOLDS, "uniform"),
            (ENV_WHIR_FOLDS, "uniform3"),
            (ENV_WHIR_FOLDS, ""),
            (ENV_WHIR_FOLDS, "4,4,4"),
            (ENV_WHIR_FOLDS, "dp"),
            (ENV_WHIR_FOLDS, "first"),
            (ENV_WHIR_FOLDS, "first4"),
            (ENV_WHIR_FOLDS, "first3"),
            (ENV_WHIR_FOLDS, "first7"),
            (ENV_WHIR_FOLDS, "first0"),
            (ENV_WHIR_FOLDS, "first 6"),
            (ENV_WHIR_FOLDS, "first06"),
            (ENV_WHIR_FOLDS, "list:6,4"),
        ] {
            let err = parse(&[(name, v)]).expect_err(&format!("{name}={v:?} must be refused"));
            assert!(err.contains(name), "{err}");
        }
    }

    #[test]
    fn the_banner_round_trips_through_the_knobs() {
        let f = ZfFormat {
            cap: CapPolicy::Auto,
            whir_cap: CapPolicy::Fixed(2),
            fri: FriMode::Dp,
            one_row: OneRowMode::Auto,
            whir_folds: WhirFolds::First(FirstFold::new(6).unwrap()),
        };
        assert_eq!(
            f.banner(),
            "ZF FORMAT: cap=auto whir_cap=2 fri=dp one_row=auto whir_folds=first6"
        );
        // Every banner value is a spelling its knob accepts, back to the same
        // format.
        let banner = f.banner();
        let fields: HashMap<&str, &str> = banner
            .trim_start_matches("ZF FORMAT: ")
            .split(' ')
            .map(|kv| kv.split_once('=').unwrap())
            .collect();
        let back = parse(&[
            (ENV_CAP, fields["cap"]),
            (ENV_WHIR_CAP, fields["whir_cap"]),
            (ENV_FRI, fields["fri"]),
            (ENV_ONE_ROW, fields["one_row"]),
            (ENV_WHIR_FOLDS, fields["whir_folds"]),
        ])
        .unwrap();
        assert_eq!(back, f);
    }

    #[test]
    fn a_lever_this_build_lacks_is_reported() {
        // Wave A implements none of the levers; the list names each knob set.
        let f = parse(&[(ENV_CAP, "auto"), (ENV_FRI, "dp")]).unwrap();
        let missing = f.unimplemented_levers();
        if !stark::proof::options::MERKLE_CAP_IMPLEMENTED {
            assert!(missing.contains(&ENV_CAP));
        }
        if !stark::proof::options::FRI_MODE_IMPLEMENTED {
            assert!(missing.contains(&ENV_FRI));
        }
        assert!(
            !missing.contains(&ENV_ONE_ROW),
            "an unset knob is never reported"
        );
    }

    #[test]
    fn apply_stamps_only_the_format_fields() {
        let base = crate::GoldilocksCubicProofOptions::with_blowup(4).unwrap();
        assert!(base.has_default_format());
        let f = ZfFormat {
            cap: CapPolicy::Auto,
            fri: FriMode::Dp,
            one_row: OneRowMode::On,
            ..ZfFormat::DEFAULT
        };
        let o = f.options(base.clone());
        assert_eq!(o.format.merkle_cap, CapPolicy::Auto);
        assert_eq!(o.format.fri_mode, FriMode::Dp);
        assert_eq!(o.format.one_row, OneRowMode::On);
        assert!(!o.has_default_format());
        assert_eq!(o.blowup_factor, base.blowup_factor);
        assert_eq!(o.fri_number_of_queries, base.fri_number_of_queries);
        assert_eq!(o.grinding_factor, base.grinding_factor);
        assert_eq!(o.coset_offset, base.coset_offset);
        assert_eq!(o.fri_final_poly_log_degree, base.fri_final_poly_log_degree);
        // The default format leaves options untouched.
        let d = ZfFormat::DEFAULT.options(base.clone());
        assert!(d.has_default_format());

        let chain = crate::multilinear_prove::chain_config(&[(8, 20)]);
        let c = ZfFormat {
            whir_cap: CapPolicy::Fixed(3),
            whir_folds: WhirFolds::First(FirstFold::new(5).unwrap()),
            ..ZfFormat::DEFAULT
        }
        .chain(chain);
        assert_eq!(c.format.cap, CapPolicy::Fixed(3));
        assert_eq!(c.format.folds, WhirFolds::First(FirstFold::new(5).unwrap()));
        assert_eq!(
            (c.log_blowup, c.log_folding, c.num_queries, c.grind),
            (
                chain.log_blowup,
                chain.log_folding,
                chain.num_queries,
                chain.grind
            )
        );
    }

    #[test]
    fn the_schedule_line_states_the_rounds() {
        assert_eq!(
            ZfFormat::DEFAULT.whir_schedule_line(),
            "ZF WHIR SCHEDULES: whir_folds=uniform4 q=112 n=20:[4,4,4,4,4] \
             n=21:[4,4,4,4,4,1] n=22:[4,4,4,4,4,2] n=23:[4,4,4,4,4,3] \
             n=24:[4,4,4,4,4,4] n=25:[4,4,4,4,4,4,1]"
        );
        let first6 = ZfFormat {
            whir_folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            ..ZfFormat::DEFAULT
        };
        assert_eq!(
            first6.whir_schedule_line(),
            "ZF WHIR SCHEDULES: whir_folds=first6 q=112 n=20:[6,4,4,4,2] \
             n=21:[6,4,4,4,3] n=22:[6,4,4,4,4] n=23:[6,4,4,4,4,1] \
             n=24:[6,4,4,4,4,2] n=25:[6,4,4,4,4,3]"
        );
    }

    /// The production WHIR config under each accepted knob value: the format
    /// is carried, Q is charged the schedule's rounds, and at the block's
    /// tallest stack (25) every arm keeps today's Q = 112.
    #[test]
    fn the_production_chain_config_under_each_arm() {
        use crate::multilinear_prove::chain_config_under;
        let today = chain_config_under(&ZfFormat::DEFAULT, &[(1, 25)]);
        assert_eq!(today, crate::multilinear_prove::chain_config(&[(1, 25)]));
        assert_eq!((today.rounds(25), today.num_queries), (7, 112));
        for (name, rounds25) in [("first5", 6), ("first6", 6)] {
            let f = parse(&[(ENV_WHIR_FOLDS, name)]).unwrap();
            let c = chain_config_under(&f, &[(1, 25)]);
            assert_eq!(c.format.folds, f.whir_folds);
            assert_eq!((c.rounds(25), c.num_queries), (rounds25, 112), "{name}");
            assert_eq!(
                (c.log_blowup, c.log_folding, c.grind),
                (today.log_blowup, today.log_folding, today.grind)
            );
            assert_ne!(c.fold_word(), today.fold_word());
        }
    }

    #[test]
    fn production_sites_build_the_default_format_when_nothing_is_set() {
        // No test sets a ZF knob, so the process format is the default and the
        // production constructors must produce today's values.
        assert_eq!(*ZfFormat::global(), ZfFormat::DEFAULT);
        assert!(crate::lfm::proof::aggregation_wrap_options().has_default_format());
        assert!(crate::lfm::proof::block_base_options().has_default_format());
        let chain = crate::multilinear_prove::chain_config(&[(8, 20)]);
        assert_eq!(chain.format, ChainFormat::DEFAULT);
        assert_eq!(chain.log_folding, PRODUCTION_WHIR_LOG_FOLDING);
    }

    #[test]
    fn the_recursion_guest_entries_refuse_a_non_default_format() {
        // RULINGS 11: the RV64 guest verifier stays default-only.
        let base = crate::recursion::Preset::Blowup4.options();
        for f in [
            ZfFormat {
                cap: CapPolicy::Auto,
                ..ZfFormat::DEFAULT
            },
            ZfFormat {
                fri: FriMode::Dp,
                ..ZfFormat::DEFAULT
            },
            ZfFormat {
                one_row: OneRowMode::On,
                ..ZfFormat::DEFAULT
            },
        ] {
            let opts = f.options(base.clone());
            for result in [
                crate::recursion::verify_and_attest_blob(&[], &opts),
                crate::recursion::verify_continuation_and_attest(&[], &opts),
            ] {
                let err = result.expect_err("a non-default format must be refused");
                assert!(format!("{err:?}").contains("default-format"), "{err:?}");
            }
        }
        // The default format gets past the guard (and fails on the empty blob).
        for result in [
            crate::recursion::verify_and_attest_blob(&[], &base),
            crate::recursion::verify_continuation_and_attest(&[], &base),
        ] {
            if let Err(err) = result {
                assert!(!format!("{err:?}").contains("default-format"), "{err:?}");
            }
        }
    }

    #[test]
    fn the_serialized_options_bytes_ignore_the_format_fields() {
        // RULINGS 10: the format fields are skipped by serde and rkyv, so a
        // serialized `ProofOptions` has the same bytes whatever the format,
        // and deserializes to the default format.
        let base = crate::GoldilocksCubicProofOptions::with_blowup(4).unwrap();
        let capped = ZfFormat {
            cap: CapPolicy::Auto,
            fri: FriMode::Dp,
            one_row: OneRowMode::Auto,
            ..ZfFormat::DEFAULT
        }
        .options(base.clone());
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(&base).unwrap();
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(&capped).unwrap();
        assert_eq!(a.as_slice(), b.as_slice());
        let back: ProofOptions = rkyv::from_bytes::<ProofOptions, rkyv::rancor::Error>(&b).unwrap();
        assert!(back.has_default_format());
        assert_eq!(back.fri_number_of_queries, base.fri_number_of_queries);

        let ja = serde_json::to_string(&base).unwrap();
        let jb = serde_json::to_string(&capped).unwrap();
        assert_eq!(ja, jb);
        assert!(!ja.contains("merkle_cap") && !ja.contains("fri_mode") && !ja.contains("one_row"));
        let back: ProofOptions = serde_json::from_str(&jb).unwrap();
        assert!(back.has_default_format());
    }
}

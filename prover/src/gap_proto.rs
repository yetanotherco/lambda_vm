//! ★ The PROTO lane's protocol variants — MEASUREMENT ONLY, every one OFF by
//! default.
//!
//! ```text
//! LAMBDA_VM_GAP_P1_STARK=1   the STARK block's base epochs at blowup 2: 219
//!                            queries at the 128-bit Johnson-bound target with
//!                            the 20-bit query grind (today: blowup 4, 110)
//! LAMBDA_VM_GAP_P1_WHIR=1    the WHIR base chains at rate 1/2 (`log_blowup`
//!                            1), the query count re-derived by the same rule:
//!                            225 at six rounds (today: rate 1/4, 112)
//! LAMBDA_VM_GAP_P2_WHIR=1    the WHIR base chains grind only before each
//!                            round's queries: folding 0, ood 0, query 20 bits
//!                            (today: 20 bits before all three)
//! ```
//!
//! Each variant changes a SECURITY PARAMETER. The knobs exist so one binary
//! can run a variant's ABBA arms against today's parameters; adopting one is
//! not a performance decision. The recursion (wraps, nodes, the root) keeps
//! its options under every knob.
//!
//! Unset or `0` is off and `1` is on; anything else ABORTS, as in
//! [`crate::zf_format`]: a typo that fell back to off would label a run of
//! today's parameters as the variant. The banner prints on every setting,
//! all-off included, so its absence in a log is a fact about the run.
//!
//! Read at the two production sites that build a base proof's parameters —
//! [`crate::lfm::proof::block_base_options`] (P1_STARK) and
//! [`crate::multilinear_prove::chain_config`] (P1_WHIR, P2_WHIR). The prover,
//! the host verifier and the in-guest emitter all derive their parameters from
//! those two values, so a knob moves all three together; the parameters stay
//! verifier-side constants, never read from a proof.

use std::sync::OnceLock;

use multilinear::whir_chain::GrindBits;

use crate::recursion::Preset;

/// The knob names, in banner order.
pub const ENV_P1_STARK: &str = "LAMBDA_VM_GAP_P1_STARK";
pub const ENV_P1_WHIR: &str = "LAMBDA_VM_GAP_P1_WHIR";
pub const ENV_P2_WHIR: &str = "LAMBDA_VM_GAP_P2_WHIR";

/// `log2` of the WHIR base code's inverse rate, as production configures it.
pub const PRODUCTION_WHIR_LOG_BLOWUP: usize = 2;

/// The WHIR base chains' proof-of-work bits, as production configures them.
pub const PRODUCTION_WHIR_GRIND_BITS: u8 = 20;

/// Which variants this process proves under. [`GapProto::OFF`] = today.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GapProto {
    /// P1 on the STARK pipeline: base epochs at blowup 2.
    pub p1_stark: bool,
    /// P1 on the WHIR pipeline: base chains at rate 1/2.
    pub p1_whir: bool,
    /// P2 on the WHIR pipeline: grinding only before the queries.
    pub p2_whir: bool,
}

impl GapProto {
    /// Every variant off: today's base parameters.
    pub const OFF: Self = Self {
        p1_stark: false,
        p1_whir: false,
        p2_whir: false,
    };

    /// Parse the three knobs through `lookup` (the process environment in
    /// production, a map in tests). Surrounding whitespace is ignored.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let flag = |name: &str| -> Result<bool, String> {
            match lookup(name).as_deref().map(str::trim) {
                None | Some("0") => Ok(false),
                Some("1") => Ok(true),
                Some(v) => Err(format!("{name}={v:?}: expected `0` or `1`")),
            }
        };
        Ok(Self {
            p1_stark: flag(ENV_P1_STARK)?,
            p1_whir: flag(ENV_P1_WHIR)?,
            p2_whir: flag(ENV_P2_WHIR)?,
        })
    }

    /// [`from_lookup`](Self::from_lookup) over the process environment. A
    /// variable that is set but not valid Unicode is an error, not "unset".
    pub fn from_env() -> Result<Self, String> {
        let non_unicode = std::cell::Cell::new(None);
        let proto = Self::from_lookup(|name| match std::env::var(name) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                non_unicode.set(Some(name.to_string()));
                None
            }
        })?;
        match non_unicode.into_inner() {
            Some(name) => Err(format!("{name} is set but not valid Unicode")),
            None => Ok(proto),
        }
    }

    /// ★ The variants for this process, read once and cached.
    ///
    /// Prints the banner on the first call; aborts on a value it does not
    /// accept.
    pub fn global() -> &'static Self {
        static PROTO: OnceLock<GapProto> = OnceLock::new();
        PROTO.get_or_init(|| {
            let proto = Self::from_env().unwrap_or_else(|e| {
                eprintln!("GAP PROTO: {e}");
                std::process::abort()
            });
            println!("{}", proto.banner(crate::zf_format::ZfFormat::global()));
            proto
        })
    }

    /// `GAP PROTO: p1_stark=… p1_whir=… p2_whir=…`, then the base parameters
    /// the setting resolves to under `format` — the query counts are derived,
    /// so the log states them rather than leaving them to the reader.
    pub fn banner(&self, format: &crate::zf_format::ZfFormat) -> String {
        let stark = self.stark_base_preset().options();
        let whir = crate::multilinear_prove::chain_config_with(format, self, &[(1, 25)]);
        format!(
            "GAP PROTO: p1_stark={} p1_whir={} p2_whir={} · STARK base blowup {} / {} q / \
             grind {} · WHIR base log_blowup {} / {} q at n=25 ({} rounds) / grind \
             folding {} ood {} query {}",
            u8::from(self.p1_stark),
            u8::from(self.p1_whir),
            u8::from(self.p2_whir),
            stark.blowup_factor,
            stark.fri_number_of_queries,
            stark.grinding_factor,
            whir.log_blowup,
            whir.num_queries,
            whir.rounds(25),
            whir.grind.folding,
            whir.grind.ood,
            whir.grind.query,
        )
    }

    /// The preset the STARK block's base epochs are proved under: blowup 2
    /// under P1, blowup 4 otherwise. Its format is legacy; the production site
    /// stamps the process's format on.
    pub fn stark_base_preset(&self) -> Preset {
        if self.p1_stark {
            Preset::Blowup2
        } else {
            Preset::Blowup4
        }
    }

    /// `log2` of the WHIR base code's inverse rate: 1 under P1, 2 otherwise.
    pub fn whir_log_blowup(&self) -> usize {
        if self.p1_whir {
            1
        } else {
            PRODUCTION_WHIR_LOG_BLOWUP
        }
    }

    /// The WHIR base chains' proof of work: the query grind alone under P2,
    /// the same bits before all three redrawable challenges otherwise. The
    /// query count reads only `query`, so P2 leaves it unchanged.
    pub fn whir_grind(&self) -> GrindBits {
        if self.p2_whir {
            GrindBits {
                folding: 0,
                ood: 0,
                query: PRODUCTION_WHIR_GRIND_BITS,
            }
        } else {
            GrindBits::uniform(PRODUCTION_WHIR_GRIND_BITS)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zf_format::ZfFormat;
    use multilinear::whir_chain::ChainConfig;
    use std::collections::HashMap;

    fn parse(pairs: &[(&str, &str)]) -> Result<GapProto, String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        GapProto::from_lookup(|k| map.get(k).cloned())
    }

    #[test]
    fn nothing_set_is_off() {
        assert_eq!(parse(&[]).unwrap(), GapProto::OFF);
        assert_eq!(GapProto::default(), GapProto::OFF);
        let all_zero = parse(&[
            (ENV_P1_STARK, "0"),
            (ENV_P1_WHIR, " 0 "),
            (ENV_P2_WHIR, "0"),
        ]);
        assert_eq!(all_zero.unwrap(), GapProto::OFF);
    }

    #[test]
    fn each_knob_turns_on_its_variant_only() {
        for (name, want) in [
            (
                ENV_P1_STARK,
                GapProto {
                    p1_stark: true,
                    ..GapProto::OFF
                },
            ),
            (
                ENV_P1_WHIR,
                GapProto {
                    p1_whir: true,
                    ..GapProto::OFF
                },
            ),
            (
                ENV_P2_WHIR,
                GapProto {
                    p2_whir: true,
                    ..GapProto::OFF
                },
            ),
        ] {
            assert_eq!(parse(&[(name, "1")]).unwrap(), want, "{name}");
        }
    }

    /// A value that is neither `0` nor `1` is refused, never read as off.
    #[test]
    fn an_unknown_value_is_an_error() {
        for v in ["2", "on", "yes", "true", "", "01"] {
            let err = parse(&[(ENV_P2_WHIR, v)]).expect_err(v);
            assert!(err.contains(ENV_P2_WHIR), "{err}");
        }
    }

    /// ★ Default-off: with no knob set, both production sites build exactly
    /// today's base parameters — each compared against a value built without
    /// this module, not against itself.
    #[test]
    fn off_is_todays_base_parameters() {
        let stark = GapProto::OFF.stark_base_preset().options();
        let today = Preset::Blowup4.options();
        assert_eq!(
            (
                stark.blowup_factor,
                stark.fri_number_of_queries,
                stark.grinding_factor,
                stark.coset_offset,
                stark.fri_final_poly_log_degree,
                stark.format,
            ),
            (
                today.blowup_factor,
                today.fri_number_of_queries,
                today.grinding_factor,
                today.coset_offset,
                today.fri_final_poly_log_degree,
                today.format,
            )
        );
        assert_eq!((stark.blowup_factor, stark.fri_number_of_queries), (4, 110));

        for format in [ZfFormat::DEFAULT, ZfFormat::LEGACY] {
            for shapes in [[(1usize, 25usize)], [(8, 20)], [(384, 21)]] {
                let tallest = multilinear::constraint_argument::one_stack(shapes[0].1, shapes[0].0);
                let want = format.chain(ChainConfig::with_security_folds(
                    2,
                    crate::zf_format::PRODUCTION_WHIR_LOG_FOLDING,
                    format.whir_folds,
                    tallest,
                    128,
                    GrindBits::uniform(20),
                ));
                let got =
                    crate::multilinear_prove::chain_config_with(&format, &GapProto::OFF, &shapes);
                assert_eq!(got, want, "{format:?} {shapes:?}");
                assert_eq!(
                    crate::multilinear_prove::chain_config_under(&format, &shapes),
                    want,
                    "{format:?} {shapes:?}"
                );
            }
        }
        let production = crate::multilinear_prove::chain_config_with(
            &ZfFormat::DEFAULT,
            &GapProto::OFF,
            &[(1, 25)],
        );
        assert_eq!(production.num_queries, 112);
        assert_eq!(production.grind, GrindBits::uniform(20));
    }

    /// P1 on the STARK pipeline: blowup 2 at the same target and grind, so
    /// `ceil((128 − 20) / 0.49321) = 219` queries; the terminal degree and
    /// the coset are unchanged.
    #[test]
    fn p1_stark_is_blowup_two_at_219_queries() {
        let p1 = GapProto {
            p1_stark: true,
            ..GapProto::OFF
        };
        let o = p1.stark_base_preset().options();
        assert_eq!(
            (
                o.blowup_factor,
                o.fri_number_of_queries,
                o.grinding_factor,
                o.coset_offset,
                o.fri_final_poly_log_degree
            ),
            (2, 219, 20, 3, 7)
        );
        // The WHIR site does not read it.
        assert_eq!(
            crate::multilinear_prove::chain_config_with(&ZfFormat::DEFAULT, &p1, &[(1, 25)]),
            crate::multilinear_prove::chain_config_under(&ZfFormat::DEFAULT, &[(1, 25)])
        );
    }

    /// P1 on the WHIR pipeline: rate 1/2 in every round, and the query count
    /// the production rule gives at that rate — `ceil((128 + log2 6 − 20) /
    /// 0.49321) = 225` at six rounds.
    #[test]
    fn p1_whir_is_rate_half_at_225_queries() {
        let p1 = GapProto {
            p1_whir: true,
            ..GapProto::OFF
        };
        let c = crate::multilinear_prove::chain_config_with(&ZfFormat::DEFAULT, &p1, &[(1, 25)]);
        assert_eq!(c.log_blowup, 1);
        assert_eq!(c.rounds(25), 6);
        assert_eq!(c.num_queries, 225);
        assert_eq!(c.grind, GrindBits::uniform(20));
        assert_eq!(
            c.num_queries,
            multilinear::query_count::num_queries(1, 6, 128, 20)
        );
        // The STARK site does not read it.
        assert_eq!(p1.stark_base_preset(), Preset::Blowup4);
    }

    /// P2: the folding and out-of-domain grinds go, the query grind stays, and
    /// so does the query count (it reads the query grind alone).
    #[test]
    fn p2_whir_grinds_only_before_the_queries() {
        let p2 = GapProto {
            p2_whir: true,
            ..GapProto::OFF
        };
        let c = crate::multilinear_prove::chain_config_with(&ZfFormat::DEFAULT, &p2, &[(1, 25)]);
        assert_eq!(
            c.grind,
            GrindBits {
                folding: 0,
                ood: 0,
                query: 20
            }
        );
        assert_eq!(c.num_queries, 112);
        assert_eq!(c.log_blowup, 2);

        let both = GapProto {
            p1_whir: true,
            p2_whir: true,
            ..GapProto::OFF
        };
        let c = crate::multilinear_prove::chain_config_with(&ZfFormat::DEFAULT, &both, &[(1, 25)]);
        assert_eq!((c.log_blowup, c.num_queries, c.grind.folding), (1, 225, 0));
    }

    #[test]
    fn the_banner_states_the_resolved_parameters() {
        assert_eq!(
            GapProto::OFF.banner(&ZfFormat::DEFAULT),
            "GAP PROTO: p1_stark=0 p1_whir=0 p2_whir=0 · STARK base blowup 4 / 110 q / grind \
             20 · WHIR base log_blowup 2 / 112 q at n=25 (6 rounds) / grind folding 20 ood 20 \
             query 20"
        );
        let all = GapProto {
            p1_stark: true,
            p1_whir: true,
            p2_whir: true,
        };
        assert_eq!(
            all.banner(&ZfFormat::DEFAULT),
            "GAP PROTO: p1_stark=1 p1_whir=1 p2_whir=1 · STARK base blowup 2 / 219 q / grind \
             20 · WHIR base log_blowup 1 / 225 q at n=25 (6 rounds) / grind folding 0 ood 0 \
             query 20"
        );
    }
}

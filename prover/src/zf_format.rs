//! ★ The proof FORMAT this process proves under — the ZF proof-format levers.
//!
//! ```text
//! LAMBDA_VM_ZF_CAP         off | auto | 0..=16    Merkle cap, every univariate STARK tree (S1)
//! LAMBDA_VM_ZF_WHIR_CAP    off | auto | 0..=16    Merkle cap, every WHIR chain tree (W1)
//! LAMBDA_VM_ZF_FRI         pair | dp              FRI fold schedule (S3)
//! LAMBDA_VM_ZF_ONE_ROW     0 | 1 | auto           one-row trace openings (S2)
//! LAMBDA_VM_ZF_WHIR_FOLDS  uniform4 | first5 | first6   WHIR first-round fold (W2)
//! LAMBDA_VM_ZF_WHIR_STACK  25 | 26 | 27                   WHIR stack cap, in variables (S2)
//! LAMBDA_VM_ZF_WHIR_GRIND  query | all                    WHIR proof of work: before the queries only, or all three (P2)
//! LAMBDA_VM_ZF_WHIR_GRIND_BITS  20 | 18               WHIR proof-of-work bits; Q buys back what the grind gives up
//! ```
//!
//! ★ Every unset knob is [`ZfFormat::DEFAULT`], the MEASURED configuration:
//! `cap=auto whir_cap=auto fri=dp one_row=0 whir_folds=first6 whir_stack=27 whir_grind=query
//! whir_grind_bits=18`. Each lever was measured net positive on block runs before it became the
//! default. Every knob keeps its OFF spelling (`cap=off`, `whir_cap=off`,
//! `fri=pair`, `one_row=0`, `whir_folds=uniform4`, `whir_stack=25`, `whir_grind=all`,
//! `whir_grind_bits=20`), so setting all eight to off reproduces [`ZfFormat::LEGACY`] — the format before any lever, byte for
//! byte — for rollback and for A/B arms. The crypto crates' own defaults
//! (`stark::proof::options::ProofFormat::DEFAULT`,
//! `multilinear::whir_chain::ChainFormat::DEFAULT`) stay the legacy format: a
//! library value built without a format is the legacy one, and the production
//! format reaches the proofs only through the three sites below.
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
//! before the levers land), and a knob set on a build that only parses
//! it would print a non-default format and prove the default one. Each
//! `*_IMPLEMENTED` constant is flipped when its lever is real.
//!
//! **The banner prints on every setting, including the default**:
//! `ZF FORMAT: cap=auto whir_cap=auto fri=dp one_row=0 whir_folds=first6 whir_stack=27 whir_grind=query
//! whir_grind_bits=18`. Its absence in a log is then a fact about the run, not an ambiguity.

use std::sync::OnceLock;

use multilinear::whir_chain::{
    ChainConfig, ChainFormat, FirstFold, GrindBits, NonceLayout, StackVars, WhirFolds,
};
use stark::proof::options::{CapPolicy, FriMode, OneRowMode, ProofFormat, ProofOptions};

/// The knob names, in banner order.
pub const ENV_CAP: &str = "LAMBDA_VM_ZF_CAP";
pub const ENV_WHIR_CAP: &str = "LAMBDA_VM_ZF_WHIR_CAP";
pub const ENV_FRI: &str = "LAMBDA_VM_ZF_FRI";
pub const ENV_ONE_ROW: &str = "LAMBDA_VM_ZF_ONE_ROW";
pub const ENV_WHIR_FOLDS: &str = "LAMBDA_VM_ZF_WHIR_FOLDS";
pub const ENV_WHIR_STACK: &str = "LAMBDA_VM_ZF_WHIR_STACK";
pub const ENV_WHIR_GRIND: &str = "LAMBDA_VM_ZF_WHIR_GRIND";
pub const ENV_WHIR_GRIND_BITS: &str = "LAMBDA_VM_ZF_WHIR_GRIND_BITS";

/// The uniform WHIR schedule's fold, as production configures it
/// (`multilinear_prove::chain_config`); the banner spells the default
/// `uniform4` after it.
pub const PRODUCTION_WHIR_LOG_FOLDING: usize = 4;

/// The WHIR chains' proof-of-work bits at the default, wherever [`WhirGrind`]
/// places them. The query count subtracts the query grind's bits from what the
/// queries must buy (`multilinear::query_count::num_queries`), so 18 bits run
/// Q 114 where 20 ran 112 (FAST 320: base −0.90 s at block 25368371).
pub const PRODUCTION_WHIR_GRIND_BITS: u8 = 18;

/// The bits the WHIR chains ground before the default moved to 18: the legacy
/// format's, and `LAMBDA_VM_ZF_WHIR_GRIND_BITS=20`'s (the opt-out).
pub const LEGACY_WHIR_GRIND_BITS: u8 = 20;

/// The grind bits `LAMBDA_VM_ZF_WHIR_GRIND_BITS` accepts. 18 trades two grind
/// bits for two queries (Q 112 → 114 at every production height from 15 to 27
/// variables), and every phase keeps at least the proven minimum it has at 20:
/// the binding phase is the unground fold at 27 variables (130.393), which
/// neither the query grind nor Q reaches; the query phases move 130.926 →
/// 130.907. Not below 18: each bit given up costs every chain the recursion
/// verifies a query (on the block, every block leaf's in-guest openings).
/// Widening this list is a format decision, not a parser one.
pub const WHIR_GRIND_BITS: [u8; 2] = [20, 18];

/// One process's proof format. [`ZfFormat::default`] is [`ZfFormat::DEFAULT`],
/// the measured configuration; [`ZfFormat::LEGACY`] is every lever off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// S2: how wide a WHIR stacked polynomial may get.
    pub whir_stack: StackVars,
    /// P2: where the WHIR base chains grind.
    pub whir_grind: WhirGrind,
    /// How many bits each of those grinds spends.
    pub whir_grind_bits: u8,
}

/// Where the WHIR base chains grind (P2), and so which nonces their rounds
/// carry.
///
/// Of a round's three grinds only the query one raises the proven minimum as
/// placed. The folding grind sits before the round's first sumcheck message,
/// so the first folding challenge is redrawn by varying that message without
/// grinding again. The out-of-domain grind comes after the out-of-domain
/// point, so the only challenge it guards is the batching one, which has far
/// more bits than the target without any grind (`multilinear::whir_chain`'s
/// module header). So `query` drops the other two and loses no proven bits,
/// and the query count, which reads the query grind alone, does not move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhirGrind {
    /// Before each round's query positions only ([`GrindBits::query_only`]),
    /// one nonce a round ([`NonceLayout::Spent`]). The default.
    Query,
    /// Before all three redrawable challenges ([`GrindBits::uniform`]), three
    /// nonces a round ([`NonceLayout::Three`]). The legacy format, byte for
    /// byte, and the rollback.
    All,
}

impl WhirGrind {
    /// The grind bits this placement puts in a chain config, `bits` a grind.
    pub const fn bits(self, bits: u8) -> GrindBits {
        match self {
            Self::Query => GrindBits::query_only(bits),
            Self::All => GrindBits::uniform(bits),
        }
    }

    /// The nonce layout that goes with it: a round carries the nonces it
    /// spends, except the legacy format, which keeps its three.
    pub const fn nonces(self) -> NonceLayout {
        match self {
            Self::Query => NonceLayout::Spent,
            Self::All => NonceLayout::Three,
        }
    }
}

impl std::fmt::Display for WhirGrind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Query => "query",
            Self::All => "all",
        })
    }
}

impl std::str::FromStr for WhirGrind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "query" => Ok(Self::Query),
            "all" => Ok(Self::All),
            _ => Err(()),
        }
    }
}

/// The first-round WHIR fold of the default format (`whir_folds=first6`).
const DEFAULT_WHIR_FIRST_FOLD: FirstFold = match FirstFold::new(6) {
    Some(k0) => k0,
    None => panic!("6 is a legal first fold"),
};

/// The WHIR stack cap of the default format (`whir_stack=27`).
const DEFAULT_WHIR_STACK: StackVars = match StackVars::new(27) {
    Some(n) => n,
    None => panic!("27 is a legal stack cap"),
};

impl Default for ZfFormat {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl ZfFormat {
    /// ★ The production format when no knob is set: the MEASURED
    /// configuration. S1 `cap=auto` (STARK block −15.35 s),
    /// S1+S3 `fri=dp` (−28.55 s), W1 `whir_cap=auto` and W2 `whir_folds=first6`
    /// (WHIR block −9.10 s together), S2 `whir_stack=27` (WHIR block −13.80 s
    /// against 25, 50 base chains instead of 145), each measured net positive
    /// in an ABBA block run. S2 changes no STARK proof: only the WHIR layouts
    /// read the stack. `one_row` stays off: in ABBA block runs it costs +3.2 s on
    /// the WHIR pipeline (the prover-side cost of one-row LFM proofs) and saves
    /// 8.0 s and 8 GiB of host memory on the STARK pipeline, so it is a knob
    /// (`LAMBDA_VM_ZF_ONE_ROW=auto`), recommended for the STARK pipeline.
    /// P2 `whir_grind=query` grinds the WHIR base chains before their queries
    /// only. It costs no proven bits: of a round's three grinds only the query
    /// one raises the proven minimum ([`WhirGrind`]), and the query counts, the
    /// query grinds and the blowups are the legacy ones. It changes no STARK
    /// proof. `LAMBDA_VM_ZF_WHIR_STACK=25` is the stack's rollback and
    /// `LAMBDA_VM_ZF_WHIR_GRIND=all` the grind's. `whir_grind_bits=18` grinds
    /// two bits fewer and buys them back with two queries (Q 112 → 114): the
    /// proven minimum stays 130.393 bits, at the unground fold (WHIR block base
    /// −0.90 s, FAST 320). `LAMBDA_VM_ZF_WHIR_GRIND_BITS=20` is its rollback.
    pub const DEFAULT: Self = Self {
        cap: CapPolicy::Auto,
        whir_cap: CapPolicy::Auto,
        fri: FriMode::Dp,
        one_row: OneRowMode::Off,
        whir_folds: WhirFolds::First(DEFAULT_WHIR_FIRST_FOLD),
        whir_stack: DEFAULT_WHIR_STACK,
        whir_grind: WhirGrind::Query,
        whir_grind_bits: PRODUCTION_WHIR_GRIND_BITS,
    };

    /// The legacy format: every lever off. What all eight knobs at their
    /// OFF spellings select, what the crypto crates' own defaults are, and the
    /// only format the RV64 recursion guest verifies.
    pub const LEGACY: Self = Self {
        cap: CapPolicy::Off,
        whir_cap: CapPolicy::Off,
        fri: FriMode::Pair,
        one_row: OneRowMode::Off,
        whir_folds: WhirFolds::Uniform,
        whir_stack: StackVars::LEGACY,
        whir_grind: WhirGrind::All,
        whir_grind_bits: LEGACY_WHIR_GRIND_BITS,
    };

    /// True when every lever is off: the format proves exactly what the
    /// prover proved before any lever existed.
    pub fn is_legacy(&self) -> bool {
        self.cap.is_off()
            && self.whir_cap.is_off()
            && self.fri == FriMode::Pair
            && self.one_row == OneRowMode::Off
            && self.whir_folds == WhirFolds::Uniform
            && self.whir_stack == StackVars::LEGACY
            && self.whir_grind == WhirGrind::All
            && self.whir_grind_bits == LEGACY_WHIR_GRIND_BITS
    }

    /// The grind bits the WHIR chains run at: [`WhirGrind`]'s placement with
    /// `whir_grind_bits` bits in each placed grind.
    pub const fn whir_grind_bits(&self) -> GrindBits {
        self.whir_grind.bits(self.whir_grind_bits)
    }

    /// Parse the eight knobs through `lookup` (the process environment in
    /// production, a map in tests). An unset knob is [`Self::DEFAULT`]'s value; a set one
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
        if let Some(v) = get(ENV_WHIR_STACK) {
            format.whir_stack = parse_whir_stack(&v)?;
        }
        if let Some(v) = get(ENV_WHIR_GRIND) {
            format.whir_grind = v
                .parse()
                .map_err(|()| format!("{ENV_WHIR_GRIND}={v:?}: expected `query` or `all`"))?;
        }
        if let Some(v) = get(ENV_WHIR_GRIND_BITS) {
            format.whir_grind_bits = parse_whir_grind_bits(&v)?;
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
        if self.whir_stack != StackVars::LEGACY && !multilinear::whir_chain::WHIR_STACK_IMPLEMENTED
        {
            out.push(ENV_WHIR_STACK);
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

    /// `ZF FORMAT: cap=… whir_cap=… fri=… one_row=… whir_folds=… whir_stack=…
    /// whir_grind=… whir_grind_bits=…`, each value in the spelling its knob accepts.
    pub fn banner(&self) -> String {
        format!(
            "ZF FORMAT: cap={} whir_cap={} fri={} one_row={} whir_folds={} whir_stack={} \
             whir_grind={} whir_grind_bits={}",
            self.cap,
            self.whir_cap,
            self.fri,
            self.one_row,
            whir_folds_name(&self.whir_folds),
            self.whir_stack.get(),
            self.whir_grind,
            self.whir_grind_bits
        )
    }

    /// `ZF WHIR SCHEDULES: whir_folds=… n=20:[…] … n=<stack>:[…]` — the fold
    /// schedule the WHIR base chains run at the production stack heights, up to
    /// the format's stack cap, so a log states the rounds it proved and not only
    /// the knob's name. Printed under the banner, on every setting.
    pub fn whir_schedule_line(&self) -> String {
        let stack = self.whir_stack.get();
        let config = crate::multilinear_prove::chain_config_under(self, &[(1, stack)]);
        let schedules = (20..=stack)
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
            // A test hook only; no knob sets it.
            fri_schedule_override: None,
        }
    }

    /// The WHIR part: what `multilinear::ChainConfig` carries. `whir_grind`'s
    /// other half, its bits ([`Self::whir_grind_bits`]), is a security
    /// parameter and goes in where the config is built
    /// ([`crate::multilinear_prove::chain_config_under`]).
    pub fn chain_format(&self) -> ChainFormat {
        ChainFormat {
            cap: self.whir_cap,
            folds: self.whir_folds,
            stack: self.whir_stack,
            nonces: self.whir_grind.nonces(),
            // No knob yet: the batched argue's proof type reaches no production
            // caller until its wiring lands (D-BATCH B-4b), so a knob here could
            // only make those callers refuse.
            argue: multilinear::whir_chain::ArgueFormat::PerTable,
            // No base-hash field yet (D-WHIR-P1 S3): every production tree is
            // binary, and this is read at arity 4 only.
            arity4_cap: multilinear::whir_chain::CapPolicy::Off,
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

/// The first-round folds the knob accepts: the two arms that are built.
///
/// ⚠ Not `first1..=first4`: a first fold narrower than the uniform one adds
/// rounds at some heights (Q would rise and the arms stop being comparable),
/// and `first4` IS `uniform4` under another statement word. Not `dp`: only
/// the first fold is a lever. Widening this list is a format decision, not a parser one.
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

/// `20` | `18`.
fn parse_whir_grind_bits(v: &str) -> Result<u8, String> {
    WHIR_GRIND_BITS
        .iter()
        .copied()
        .find(|&b| v == b.to_string())
        .ok_or_else(|| {
            format!(
                "{ENV_WHIR_GRIND_BITS}={v:?}: expected {}",
                WHIR_GRIND_BITS
                    .iter()
                    .map(|b| format!("`{b}`"))
                    .collect::<Vec<_>>()
                    .join(" or ")
            )
        })
}

/// The stack caps the knob accepts: the three proved, verified and measured
/// to fit on the 32 GiB card (block 25368371, E2c). Not 28: its argument's
/// reservations would not fit the device ledger. Not below 25: nothing asks
/// for a narrower stack than today's. Widening this list is a format decision,
/// not a parser one.
pub const WHIR_STACKS: [usize; 3] = [25, 26, 27];

/// `25` | `26` | `27`.
fn parse_whir_stack(v: &str) -> Result<StackVars, String> {
    WHIR_STACKS
        .iter()
        .find(|&&n| v == n.to_string())
        .and_then(|&n| StackVars::new(n))
        .ok_or_else(|| {
            format!(
                "{ENV_WHIR_STACK}={v:?}: expected {}",
                WHIR_STACKS
                    .iter()
                    .map(|n| format!("`{n}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
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

    /// ★ With no knob set the process proves the MEASURED
    /// configuration.
    #[test]
    fn nothing_set_is_the_measured_default() {
        let f = parse(&[]).unwrap();
        assert_eq!(f, ZfFormat::DEFAULT);
        assert_eq!(f, ZfFormat::default());
        assert_eq!(
            f,
            ZfFormat {
                cap: CapPolicy::Auto,
                whir_cap: CapPolicy::Auto,
                fri: FriMode::Dp,
                one_row: OneRowMode::Off,
                whir_folds: WhirFolds::First(FirstFold::new(6).unwrap()),
                whir_stack: StackVars::new(27).unwrap(),
                whir_grind: WhirGrind::Query,
                whir_grind_bits: 18,
            }
        );
        assert_eq!(
            f.banner(),
            "ZF FORMAT: cap=auto whir_cap=auto fri=dp one_row=0 whir_folds=first6 whir_stack=27 \
             whir_grind=query whir_grind_bits=18"
        );
        assert!(!f.is_legacy());
        assert!(f.unimplemented_levers().is_empty());
    }

    /// Every knob keeps its OFF spelling, and all seven at off are the legacy
    /// format (every lever off): the rollback and A/B arm.
    #[test]
    fn the_off_spellings_parse_to_the_legacy_format() {
        let f = parse(&[
            (ENV_CAP, "off"),
            (ENV_WHIR_CAP, "0"),
            (ENV_FRI, "pair"),
            (ENV_ONE_ROW, "0"),
            (ENV_WHIR_FOLDS, "uniform4"),
            (ENV_WHIR_STACK, "25"),
            (ENV_WHIR_GRIND, "all"),
            (ENV_WHIR_GRIND_BITS, "20"),
        ])
        .unwrap();
        assert_eq!(f, ZfFormat::LEGACY);
        assert!(f.is_legacy());
        assert_eq!(
            f.banner(),
            "ZF FORMAT: cap=off whir_cap=off fri=pair one_row=0 whir_folds=uniform4 whir_stack=25 \
             whir_grind=all whir_grind_bits=20"
        );
        assert!(f.unimplemented_levers().is_empty());
        assert!(f.proof_format().is_legacy());
        assert_eq!(f.proof_format(), ProofFormat::LEGACY);
        assert_eq!(f.chain_format(), ChainFormat::DEFAULT);
    }

    /// One knob at its off spelling turns off that lever ONLY; the others keep
    /// the default's value.
    #[test]
    fn one_off_knob_turns_off_one_lever() {
        for (name, v, want) in [
            (
                ENV_CAP,
                "off",
                ZfFormat {
                    cap: CapPolicy::Off,
                    ..ZfFormat::DEFAULT
                },
            ),
            (
                ENV_WHIR_CAP,
                "off",
                ZfFormat {
                    whir_cap: CapPolicy::Off,
                    ..ZfFormat::DEFAULT
                },
            ),
            (
                ENV_FRI,
                "pair",
                ZfFormat {
                    fri: FriMode::Pair,
                    ..ZfFormat::DEFAULT
                },
            ),
            (
                ENV_WHIR_FOLDS,
                "uniform4",
                ZfFormat {
                    whir_folds: WhirFolds::Uniform,
                    ..ZfFormat::DEFAULT
                },
            ),
            (
                ENV_WHIR_STACK,
                "25",
                ZfFormat {
                    whir_stack: StackVars::LEGACY,
                    ..ZfFormat::DEFAULT
                },
            ),
            (
                ENV_WHIR_GRIND,
                "all",
                ZfFormat {
                    whir_grind: WhirGrind::All,
                    ..ZfFormat::DEFAULT
                },
            ),
            (
                ENV_WHIR_GRIND_BITS,
                "20",
                ZfFormat {
                    whir_grind_bits: LEGACY_WHIR_GRIND_BITS,
                    ..ZfFormat::DEFAULT
                },
            ),
        ] {
            let f = parse(&[(name, v)]).unwrap();
            assert_eq!(f, want, "{name}={v}");
            assert!(!f.is_legacy(), "{name}={v}");
        }
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
        for n in WHIR_STACKS {
            assert_eq!(
                parse(&[(ENV_WHIR_STACK, &n.to_string())])
                    .unwrap()
                    .whir_stack,
                StackVars::new(n).unwrap()
            );
        }
        assert_eq!(
            parse(&[(ENV_WHIR_STACK, " 27 ")]).unwrap().whir_stack,
            StackVars::new(27).unwrap()
        );
        for (v, want) in [
            ("query", WhirGrind::Query),
            ("all", WhirGrind::All),
            (" ALL ", WhirGrind::All),
        ] {
            assert_eq!(
                parse(&[(ENV_WHIR_GRIND, v)]).unwrap().whir_grind,
                want,
                "{v:?}"
            );
        }
        for (v, want) in [("20", 20u8), ("18", 18), (" 18 ", 18)] {
            assert_eq!(
                parse(&[(ENV_WHIR_GRIND_BITS, v)]).unwrap().whir_grind_bits,
                want,
                "{v:?}"
            );
        }
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
            (ENV_WHIR_STACK, ""),
            (ENV_WHIR_STACK, "24"),
            (ENV_WHIR_STACK, "28"),
            (ENV_WHIR_STACK, "0"),
            (ENV_WHIR_STACK, "027"),
            (ENV_WHIR_STACK, "27.0"),
            (ENV_WHIR_STACK, "auto"),
            (ENV_WHIR_STACK, "off"),
            (ENV_WHIR_GRIND, ""),
            (ENV_WHIR_GRIND, "uniform"),
            (ENV_WHIR_GRIND, "0"),
            (ENV_WHIR_GRIND, "1"),
            (ENV_WHIR_GRIND, "off"),
            (ENV_WHIR_GRIND, "query-only"),
            (ENV_WHIR_GRIND, "20"),
            (ENV_WHIR_GRIND_BITS, ""),
            (ENV_WHIR_GRIND_BITS, "19"),
            (ENV_WHIR_GRIND_BITS, "16"),
            (ENV_WHIR_GRIND_BITS, "0"),
            (ENV_WHIR_GRIND_BITS, "018"),
            (ENV_WHIR_GRIND_BITS, "18.0"),
            (ENV_WHIR_GRIND_BITS, "off"),
            (ENV_WHIR_GRIND_BITS, "query"),
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
            whir_stack: StackVars::new(26).unwrap(),
            whir_grind: WhirGrind::All,
            whir_grind_bits: 18,
        };
        assert_eq!(
            f.banner(),
            "ZF FORMAT: cap=auto whir_cap=2 fri=dp one_row=auto whir_folds=first6 whir_stack=26 \
             whir_grind=all whir_grind_bits=18"
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
            (ENV_WHIR_STACK, fields["whir_stack"]),
            (ENV_WHIR_GRIND, fields["whir_grind"]),
            (ENV_WHIR_GRIND_BITS, fields["whir_grind_bits"]),
        ])
        .unwrap();
        assert_eq!(back, f);
    }

    #[test]
    fn a_lever_this_build_lacks_is_reported() {
        // Wave A implements none of the levers; the list names each knob set.
        let f = parse(&[(ENV_CAP, "auto"), (ENV_FRI, "dp")]).unwrap();
        let missing = f.unimplemented_levers();
        // S3 is implemented on the host (stark::proof::options::FRI_MODE_IMPLEMENTED).
        const { assert!(stark::proof::options::FRI_MODE_IMPLEMENTED) };
        assert!(!missing.contains(&ENV_FRI), "fri=dp is selectable");
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
    fn the_one_row_knob_is_selectable() {
        // S2 is implemented on the host CPU paths: `LAMBDA_VM_ZF_ONE_ROW` no
        // longer aborts, and every spelling reaches the options unchanged.
        const { assert!(stark::proof::options::ONE_ROW_IMPLEMENTED) };
        for (v, want) in [
            ("1", OneRowMode::On),
            ("auto", OneRowMode::Auto),
            ("0", OneRowMode::Off),
        ] {
            let f = parse(&[(ENV_ONE_ROW, v)]).unwrap();
            assert!(f.unimplemented_levers().is_empty(), "{v}");
            let base = crate::GoldilocksCubicProofOptions::with_blowup(4).unwrap();
            assert_eq!(f.options(base).format.one_row, want, "{v}");
        }
        assert_eq!(
            parse(&[(ENV_ONE_ROW, "auto"), (ENV_FRI, "dp")])
                .unwrap()
                .banner(),
            "ZF FORMAT: cap=auto whir_cap=auto fri=dp one_row=auto whir_folds=first6 whir_stack=27 \
             whir_grind=query whir_grind_bits=18"
        );
    }

    #[test]
    fn the_merkle_cap_knob_is_selectable() {
        // The STARK cap is real on host and device, so `LAMBDA_VM_ZF_CAP` does
        // not abort; every spelling reaches the options unchanged.
        const { assert!(stark::proof::options::MERKLE_CAP_IMPLEMENTED) };
        for (v, want) in [
            ("auto", CapPolicy::Auto),
            ("3", CapPolicy::Fixed(3)),
            ("off", CapPolicy::Off),
        ] {
            let f = parse(&[(ENV_CAP, v)]).unwrap();
            assert!(!f.unimplemented_levers().contains(&ENV_CAP), "{v}");
            let base = crate::GoldilocksCubicProofOptions::with_blowup(4).unwrap();
            assert_eq!(f.options(base).format.merkle_cap, want, "{v}");
        }
    }

    #[test]
    fn the_whir_cap_is_implemented_and_selectable() {
        const { assert!(multilinear::whir_chain::WHIR_CAP_IMPLEMENTED) };
        for v in ["auto", "3"] {
            let f = parse(&[(ENV_WHIR_CAP, v)]).unwrap();
            assert!(
                !f.unimplemented_levers().contains(&ENV_WHIR_CAP),
                "LAMBDA_VM_ZF_WHIR_CAP={v} must be selectable"
            );
        }
    }

    #[test]
    fn the_whir_fold_lever_is_selectable() {
        const { assert!(multilinear::whir_chain::WHIR_FOLDS_IMPLEMENTED) };
        for v in ["first5", "first6"] {
            let f = parse(&[(ENV_WHIR_FOLDS, v)]).unwrap();
            assert!(f.unimplemented_levers().is_empty(), "{v}");
        }
    }

    /// S2: every accepted stack is selectable, reaches the chain config the
    /// prover, the host verifier and the in-guest emitters all build their
    /// layouts from, moves no other field, and 25 is the legacy value wherever
    /// a default is spelled.
    #[test]
    fn the_whir_stack_lever_is_selectable_and_carried() {
        const { assert!(multilinear::whir_chain::WHIR_STACK_IMPLEMENTED) };
        assert_eq!(ZfFormat::LEGACY.whir_stack, StackVars::LEGACY);
        assert_eq!(ChainFormat::DEFAULT.stack, StackVars::LEGACY);
        for n in WHIR_STACKS {
            let f = parse(&[(ENV_WHIR_STACK, &n.to_string())]).unwrap();
            assert!(f.unimplemented_levers().is_empty(), "{n}");
            assert_eq!(f.chain_format().stack.get(), n);
            let c = crate::multilinear_prove::chain_config_under(&f, &[(1, 25)]);
            assert_eq!(c.format.stack.get(), n, "the config carries the stack");
            assert!(
                f.banner()
                    .split(' ')
                    .any(|kv| kv == format!("whir_stack={n}")),
                "{}",
                f.banner()
            );
            assert_eq!(
                ZfFormat {
                    whir_stack: ZfFormat::DEFAULT.whir_stack,
                    ..f
                },
                ZfFormat::DEFAULT,
                "the knob moves the stack and nothing else"
            );
        }
        assert_eq!(WHIR_STACKS.iter().max(), Some(&StackVars::WIDEST));
    }

    /// ★ Q is charged the tallest STACKED polynomial, not only the widest
    /// table: three narrow tables that each fit 14 variables stack to 16, a
    /// four-round chain under first6 where each table alone runs three.
    #[test]
    fn the_query_count_charges_the_tallest_stacked_polynomial() {
        use crate::multilinear_prove::chain_config_under;
        let narrow = [(1, 14), (1, 14), (1, 14)];
        let c = chain_config_under(&ZfFormat::DEFAULT, &narrow);
        assert_eq!(
            stark::multilinear_table::stack_height(&narrow, ZfFormat::DEFAULT.whir_stack),
            16
        );
        assert_eq!(
            (c.rounds(16), c.num_queries),
            (4, 114),
            "charged at 16 variables"
        );
        // What the widest table alone would charge: three rounds, one query fewer.
        let widest_alone = ChainConfig::with_security_folds(
            2,
            PRODUCTION_WHIR_LOG_FOLDING,
            ZfFormat::DEFAULT.whir_folds,
            14,
            128,
            ZfFormat::DEFAULT.whir_grind_bits(),
        );
        assert_eq!(widest_alone.num_queries, 113);
        // At the production shapes the charge moves nothing, at every accepted
        // stack: the widest table stands at or above the cap, and Q is 114 at
        // every production height.
        for n in WHIR_STACKS {
            let f = ZfFormat {
                whir_stack: StackVars::new(n).unwrap(),
                ..ZfFormat::DEFAULT
            };
            for shapes in [
                &[(1usize, 25usize)][..],
                &[(64, 21), (1480, 16), (8, 20)][..],
            ] {
                assert_eq!(
                    chain_config_under(&f, shapes).num_queries,
                    114,
                    "stack {n}, shapes {shapes:?}"
                );
            }
        }
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
        // The legacy format leaves options untouched.
        let d = ZfFormat::LEGACY.options(base.clone());
        assert!(d.has_default_format());
        assert!(d.has_legacy_format());
        // The production default stamps cap=auto and fri=dp, nothing else.
        let p = ZfFormat::DEFAULT.options(base.clone());
        assert_eq!(p.format.merkle_cap, CapPolicy::Auto);
        assert_eq!(p.format.fri_mode, FriMode::Dp);
        assert_eq!(p.format.one_row, ZfFormat::DEFAULT.one_row);
        assert_eq!(p.format.fri_schedule_override, None);
        assert!(!p.has_legacy_format());
        assert_eq!(
            (
                p.blowup_factor,
                p.fri_number_of_queries,
                p.grinding_factor,
                p.coset_offset,
                p.fri_final_poly_log_degree
            ),
            (
                base.blowup_factor,
                base.fri_number_of_queries,
                base.grinding_factor,
                base.coset_offset,
                base.fri_final_poly_log_degree
            ),
            "no security parameter moves with the format"
        );

        let chain = crate::multilinear_prove::chain_config_under(&ZfFormat::LEGACY, &[(8, 20)]);
        let c = ZfFormat {
            whir_cap: CapPolicy::Fixed(3),
            whir_folds: WhirFolds::First(FirstFold::new(5).unwrap()),
            ..ZfFormat::LEGACY
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
            ZfFormat::LEGACY.whir_schedule_line(),
            "ZF WHIR SCHEDULES: whir_folds=uniform4 q=112 n=20:[4,4,4,4,4] \
             n=21:[4,4,4,4,4,1] n=22:[4,4,4,4,4,2] n=23:[4,4,4,4,4,3] \
             n=24:[4,4,4,4,4,4] n=25:[4,4,4,4,4,4,1]"
        );
        let first6 = ZfFormat {
            whir_folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            ..ZfFormat::LEGACY
        };
        // The production default runs the first6 schedules, to its stack, at
        // its grind bits' Q.
        assert_eq!(
            ZfFormat::DEFAULT.whir_schedule_line(),
            ZfFormat {
                whir_stack: ZfFormat::DEFAULT.whir_stack,
                whir_grind_bits: ZfFormat::DEFAULT.whir_grind_bits,
                ..first6
            }
            .whir_schedule_line()
        );
        assert_eq!(
            first6.whir_schedule_line(),
            "ZF WHIR SCHEDULES: whir_folds=first6 q=112 n=20:[6,4,4,4,2] \
             n=21:[6,4,4,4,3] n=22:[6,4,4,4,4] n=23:[6,4,4,4,4,1] \
             n=24:[6,4,4,4,4,2] n=25:[6,4,4,4,4,3]"
        );
        // Under a wider stack the line runs to it, with Q charged there.
        let wide = ZfFormat {
            whir_stack: StackVars::new(27).unwrap(),
            ..first6
        };
        assert_eq!(
            wide.whir_schedule_line(),
            "ZF WHIR SCHEDULES: whir_folds=first6 q=112 n=20:[6,4,4,4,2] \
             n=21:[6,4,4,4,3] n=22:[6,4,4,4,4] n=23:[6,4,4,4,4,1] \
             n=24:[6,4,4,4,4,2] n=25:[6,4,4,4,4,3] n=26:[6,4,4,4,4,4] \
             n=27:[6,4,4,4,4,4,1]"
        );
    }

    /// The production WHIR config under each accepted knob value: the format
    /// is carried, Q is charged the schedule's rounds, and at the block's
    /// tallest stack (25) every fold arm keeps the legacy Q = 112 at 20 grind
    /// bits, and the default's Q = 114 at its 18.
    #[test]
    fn the_production_chain_config_under_each_arm() {
        use crate::multilinear_prove::chain_config_under;
        let today = chain_config_under(&ZfFormat::LEGACY, &[(1, 25)]);
        assert_eq!(today.format, ChainFormat::DEFAULT);
        assert_eq!((today.rounds(25), today.num_queries), (7, 112));
        // The production config (no knob set) is the measured default's:
        // cap auto, first6 — six rounds at 25; Q 114 at 18 grind bits.
        let production = crate::multilinear_prove::chain_config(&[(1, 25)]);
        assert_eq!(
            production,
            chain_config_under(&ZfFormat::DEFAULT, &[(1, 25)])
        );
        assert_eq!(production.format, ZfFormat::DEFAULT.chain_format());
        assert_eq!((production.rounds(25), production.num_queries), (6, 114));
        assert_eq!(production.schedule(25), vec![6, 4, 4, 4, 4, 3]);
        assert_eq!(
            (production.log_blowup, production.log_folding),
            (today.log_blowup, today.log_folding),
            "the blowup does not move with the format"
        );
        // P2: the default grinds before the queries alone, the legacy format
        // before all three challenges; the default at 18 bits, the legacy at 20.
        assert_eq!(production.grind, GrindBits::query_only(18));
        assert_eq!(today.grind, GrindBits::uniform(20));
        for (name, rounds25) in [("first5", 6), ("first6", 6)] {
            let f = parse(&[(ENV_WHIR_FOLDS, name)]).unwrap();
            let c = chain_config_under(&f, &[(1, 25)]);
            assert_eq!(c.format.folds, f.whir_folds);
            assert_eq!((c.rounds(25), c.num_queries), (rounds25, 114), "{name}");
            assert_eq!(
                (c.log_blowup, c.log_folding, c.grind),
                (today.log_blowup, today.log_folding, f.whir_grind_bits())
            );
            assert_ne!(c.fold_word(), today.fold_word());
        }
    }

    #[test]
    fn production_sites_build_the_default_format_when_nothing_is_set() {
        // No test sets a ZF knob, so the process format is the default and the
        // production constructors must stamp the MEASURED configuration.
        assert_eq!(*ZfFormat::global(), ZfFormat::DEFAULT);
        let want = ZfFormat::DEFAULT.proof_format();
        for (site, o) in [
            (
                "aggregation_wrap_options",
                crate::lfm::proof::aggregation_wrap_options(),
            ),
            (
                "block_base_options",
                crate::lfm::proof::block_base_options(),
            ),
        ] {
            assert_eq!(o.format, want, "{site}");
            assert!(!o.has_legacy_format(), "{site}");
        }
        // Security parameters are the legacy presets'.
        let base = crate::lfm::proof::block_base_options();
        let preset = crate::recursion::Preset::Blowup4.options();
        assert_eq!(
            (
                base.blowup_factor,
                base.fri_number_of_queries,
                base.grinding_factor,
                base.coset_offset,
                base.fri_final_poly_log_degree
            ),
            (
                preset.blowup_factor,
                preset.fri_number_of_queries,
                preset.grinding_factor,
                preset.coset_offset,
                preset.fri_final_poly_log_degree
            )
        );
        let chain = crate::multilinear_prove::chain_config(&[(8, 20)]);
        assert_eq!(chain.format, ZfFormat::DEFAULT.chain_format());
        assert_eq!(chain.log_folding, PRODUCTION_WHIR_LOG_FOLDING);
    }

    /// ★ The RV64 guest verifier stays on
    /// the LEGACY format after the default flip. Its presets NAME the legacy
    /// format (not the process default), and both guest entries refuse every
    /// other format — the production default included.
    #[test]
    fn the_recursion_guest_stays_on_the_legacy_format() {
        for preset in crate::recursion::Preset::ALL {
            let o = preset.options();
            assert_eq!(o.format, ProofFormat::LEGACY, "{}", preset.name());
            assert!(o.has_legacy_format(), "{}", preset.name());
        }
        assert_eq!(
            crate::recursion::MIN_PROOF_OPTIONS.format,
            ProofFormat::LEGACY
        );
        // The production default is NOT legacy, so the presets cannot have
        // inherited their format from it.
        assert!(!ZfFormat::DEFAULT.is_legacy());
        assert!(!ZfFormat::DEFAULT.proof_format().is_legacy());

        let base = crate::recursion::Preset::Blowup4.options();
        let refused = |opts: &ProofOptions| {
            for result in [
                crate::recursion::verify_and_attest_blob(&[], opts),
                crate::recursion::verify_continuation_and_attest(&[], opts),
            ] {
                let err = result.expect_err("a non-legacy format must be refused");
                assert!(format!("{err:?}").contains("legacy-format"), "{err:?}");
            }
        };
        for f in [
            ZfFormat::DEFAULT,
            ZfFormat {
                cap: CapPolicy::Auto,
                ..ZfFormat::LEGACY
            },
            ZfFormat {
                fri: FriMode::Dp,
                ..ZfFormat::LEGACY
            },
            ZfFormat {
                one_row: OneRowMode::On,
                ..ZfFormat::LEGACY
            },
            ZfFormat {
                one_row: OneRowMode::Auto,
                ..ZfFormat::LEGACY
            },
        ] {
            refused(&f.options(base.clone()));
        }
        // The production STARK base options are refused whenever the process
        // format is not legacy (always, with no knob set).
        if !ZfFormat::global().proof_format().is_legacy() {
            refused(&crate::lfm::proof::block_base_options());
        }
        // The legacy format gets past the guard (and fails on the empty blob).
        for opts in [base.clone(), ZfFormat::LEGACY.options(base.clone())] {
            for result in [
                crate::recursion::verify_and_attest_blob(&[], &opts),
                crate::recursion::verify_continuation_and_attest(&[], &opts),
            ] {
                if let Err(err) = result {
                    assert!(!format!("{err:?}").contains("legacy-format"), "{err:?}");
                }
            }
        }
    }

    #[test]
    fn the_serialized_options_bytes_ignore_the_format_fields() {
        // The format fields are skipped by serde and rkyv, so a
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

    /// P2: each spelling reaches the chain config the prover, the host verifier
    /// and the in-guest emitters build from, as its bits and its nonce layout
    /// together, at every accepted stack; Q does not move, and the knob moves
    /// nothing else.
    #[test]
    fn the_whir_grind_lever_is_selectable_and_carried() {
        for (v, bits, nonces) in [
            ("query", GrindBits::query_only(18), NonceLayout::Spent),
            ("all", GrindBits::uniform(18), NonceLayout::Three),
        ] {
            let f = parse(&[(ENV_WHIR_GRIND, v)]).unwrap();
            assert!(f.unimplemented_levers().is_empty(), "{v}");
            assert!(
                f.banner()
                    .ends_with(&format!(" whir_grind={v} whir_grind_bits=18")),
                "{}",
                f.banner()
            );
            for n in WHIR_STACKS {
                let at = ZfFormat {
                    whir_stack: StackVars::new(n).unwrap(),
                    ..f
                };
                let c = crate::multilinear_prove::chain_config_under(&at, &[(1, n)]);
                assert_eq!((c.grind, c.format.nonces), (bits, nonces), "{v} at {n}");
                assert_eq!(
                    c.num_queries, 114,
                    "{v} at {n}: Q reads the query grind alone"
                );
            }
            assert_eq!(
                ZfFormat {
                    whir_grind: ZfFormat::DEFAULT.whir_grind,
                    ..f
                },
                ZfFormat::DEFAULT,
                "the knob moves the grind and nothing else"
            );
        }
        assert_eq!(ZfFormat::LEGACY.whir_grind, WhirGrind::All);
        assert_eq!(ChainFormat::DEFAULT.nonces, NonceLayout::Three);
    }

    /// ★ THE OPT-OUT IS TODAY'S CONFIG. `LAMBDA_VM_ZF_WHIR_GRIND=all` with
    /// `LAMBDA_VM_ZF_WHIR_GRIND_BITS=20` builds the production WHIR config
    /// exactly as it was before P2, spelled out here as a literal rather than
    /// derived through the code under test: blowup 2^2, fold 4 under first6,
    /// Q 112, 20-bit grinds before all three challenges, the auto cap, stack 27,
    /// three nonces a round. The default differs from it in the grind (placed
    /// before the queries alone, at 18 bits), its Q (114) and the nonce layout,
    /// and in nothing else.
    #[test]
    fn the_grind_opt_out_is_the_production_config_before_p2() {
        let before_p2 = ChainConfig {
            log_blowup: 2,
            log_folding: 4,
            num_queries: 112,
            grind: GrindBits {
                folding: 20,
                ood: 20,
                query: 20,
            },
            format: ChainFormat {
                cap: CapPolicy::Auto,
                folds: WhirFolds::First(FirstFold::new(6).unwrap()),
                stack: StackVars::new(27).unwrap(),
                nonces: NonceLayout::Three,
                argue: multilinear::whir_chain::ArgueFormat::PerTable,
                arity4_cap: CapPolicy::Off,
            },
        };
        let opt_out = parse(&[(ENV_WHIR_GRIND, "all"), (ENV_WHIR_GRIND_BITS, "20")]).unwrap();
        for shapes in [
            &[(1usize, 25usize)][..],
            &[(1, 27)][..],
            &[(64, 21), (1480, 16), (8, 20)][..],
        ] {
            assert_eq!(
                crate::multilinear_prove::chain_config_under(&opt_out, shapes),
                before_p2,
                "{shapes:?}"
            );
            assert_eq!(
                crate::multilinear_prove::chain_config_under(&ZfFormat::DEFAULT, shapes),
                ChainConfig {
                    num_queries: 114,
                    grind: GrindBits {
                        folding: 0,
                        ood: 0,
                        query: 18,
                    },
                    format: ChainFormat {
                        nonces: NonceLayout::Spent,
                        ..before_p2.format
                    },
                    ..before_p2
                },
                "{shapes:?}"
            );
        }
    }

    /// The STARK pipeline has nothing to adopt: it grinds only before its
    /// queries already. The lever reaches no univariate option, so no STARK
    /// proof, wrap program or id moves with it.
    #[test]
    fn the_grind_lever_moves_no_stark_option() {
        let all = ZfFormat {
            whir_grind: WhirGrind::All,
            ..ZfFormat::DEFAULT
        };
        assert_eq!(all.proof_format(), ZfFormat::DEFAULT.proof_format());
        for preset in crate::recursion::Preset::ALL {
            let o = preset.options();
            assert_eq!(
                format!("{:?}", all.options(o.clone())),
                format!("{:?}", ZfFormat::DEFAULT.options(o)),
                "{}",
                preset.name()
            );
        }
    }

    /// The default's 18 grind bits trade two grind bits for two queries against
    /// `LAMBDA_VM_ZF_WHIR_GRIND_BITS=20` (the opt-out, the legacy bits) and
    /// move nothing else: Q 112 → 114 at every production height (15 to 27
    /// variables, every accepted stack, the block's shapes), 111 → 113 below a
    /// fourth round, under either placement.
    #[test]
    fn the_grind_bits_knob_trades_two_bits_for_two_queries() {
        use crate::multilinear_prove::chain_config_under;
        assert_eq!(PRODUCTION_WHIR_GRIND_BITS, 18);
        assert_eq!(LEGACY_WHIR_GRIND_BITS, 20);
        assert_eq!(
            ZfFormat::DEFAULT.whir_grind_bits,
            PRODUCTION_WHIR_GRIND_BITS
        );
        assert_eq!(ZfFormat::LEGACY.whir_grind_bits, LEGACY_WHIR_GRIND_BITS);
        assert_eq!(
            parse(&[(ENV_WHIR_GRIND_BITS, "18")]).unwrap(),
            ZfFormat::DEFAULT
        );
        assert!(ZfFormat::DEFAULT.whir_schedule_line().contains(" q=114 "));

        // The opt-out: 20 bits, Q 112, nothing else moved.
        let f = parse(&[(ENV_WHIR_GRIND_BITS, "20")]).unwrap();
        assert!(f.unimplemented_levers().is_empty());
        assert!(!f.is_legacy());
        assert!(
            f.banner().ends_with(" whir_grind=query whir_grind_bits=20"),
            "{}",
            f.banner()
        );
        assert_eq!(
            ZfFormat {
                whir_grind_bits: PRODUCTION_WHIR_GRIND_BITS,
                ..f
            },
            ZfFormat::DEFAULT,
            "the knob moves the bits and nothing else"
        );
        assert_eq!(f.chain_format(), ZfFormat::DEFAULT.chain_format());
        assert_eq!(f.proof_format(), ZfFormat::DEFAULT.proof_format());
        assert!(f.whir_schedule_line().contains(" q=112 "));

        for (placement, bits18, bits20) in [
            (
                WhirGrind::Query,
                GrindBits::query_only(18),
                GrindBits::query_only(20),
            ),
            (
                WhirGrind::All,
                GrindBits::uniform(18),
                GrindBits::uniform(20),
            ),
        ] {
            let at18 = ZfFormat {
                whir_grind: placement,
                ..ZfFormat::DEFAULT
            };
            let at20 = ZfFormat {
                whir_grind: placement,
                ..f
            };
            for n in WHIR_STACKS {
                let stack = StackVars::new(n).unwrap();
                for shapes in [
                    &[(1usize, n)][..],
                    &[(1, 27)][..],
                    &[(64, 21), (1480, 16), (8, 20)][..],
                ] {
                    let c18 = chain_config_under(
                        &ZfFormat {
                            whir_stack: stack,
                            ..at18
                        },
                        shapes,
                    );
                    let c20 = chain_config_under(
                        &ZfFormat {
                            whir_stack: stack,
                            ..at20
                        },
                        shapes,
                    );
                    assert_eq!(
                        (c18.num_queries, c18.grind, c20.num_queries, c20.grind),
                        (114, bits18, 112, bits20),
                        "{placement} at stack {n}, shapes {shapes:?}"
                    );
                    assert_eq!(
                        ChainConfig {
                            num_queries: c20.num_queries,
                            grind: c20.grind,
                            ..c18
                        },
                        c20,
                        "{placement} at stack {n}, shapes {shapes:?}: only Q and the grind move"
                    );
                }
            }
            // Below a fourth round (14 variables and narrower) both sides buy
            // one query less.
            assert_eq!(
                (
                    chain_config_under(&at18, &[(1, 14)]).num_queries,
                    chain_config_under(&at20, &[(1, 14)]).num_queries
                ),
                (113, 111),
                "{placement}"
            );
        }
        // Every height a production chain reaches, from the rule itself.
        for n in 15..=27 {
            let q = |bits| {
                ChainConfig::with_security_folds(
                    2,
                    PRODUCTION_WHIR_LOG_FOLDING,
                    ZfFormat::DEFAULT.whir_folds,
                    n,
                    128,
                    GrindBits::query_only(bits),
                )
                .num_queries
            };
            assert_eq!((q(18), q(20)), (114, 112), "{n} variables");
        }
    }
}

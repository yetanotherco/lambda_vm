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
//! LAMBDA_VM_ZF_LOGUP       pair | k3 | k4 | best          LogUp interactions per aux column, STARK base tables only
//! ```
//!
//! ★ Every unset knob is [`ZfFormat::DEFAULT`], the MEASURED configuration:
//! `cap=auto whir_cap=auto fri=dp one_row=auto whir_folds=first6 whir_stack=27 whir_grind=query
//! logup=k4`.
//! Each lever was measured net positive on block runs before it became the
//! default (`one_row=auto` on the STARK pipeline: see [`ZfFormat::DEFAULT`]).
//! Every knob keeps its OFF spelling (`cap=off`, `whir_cap=off`,
//! `fri=pair`, `one_row=0`, `whir_folds=uniform4`, `whir_stack=25`, `whir_grind=all`,
//! `logup=pair`), so setting all eight to off reproduces [`ZfFormat::LEGACY`] — the format
//! before any lever, byte for byte — for rollback and for A/B arms. The crypto crates' own defaults
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
//! STARK block's base epochs). `LAMBDA_VM_ZF_LOGUP` reaches the last site only:
//! [`ZfFormat::proof_format`] (the LFM chips' format) keeps the pair layout,
//! and [`ZfFormat::base_proof_format`] adds the base tables' arity. From there it travels inside the option types
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
//! `ZF FORMAT: cap=auto whir_cap=auto fri=dp one_row=auto whir_folds=first6 whir_stack=27 whir_grind=query logup=k4`.
//! Its absence in a log is then a fact about the run, not an ambiguity.

use std::sync::OnceLock;

use multilinear::whir_chain::{
    ChainConfig, ChainFormat, FirstFold, GrindBits, NonceLayout, StackVars, WhirFolds,
};
use stark::proof::options::{
    BaseFormat, CapPolicy, FriMode, LogUpPolicy, OneRowMode, ProofFormat, ProofOptions,
};

/// The knob names, in banner order.
pub const ENV_CAP: &str = "LAMBDA_VM_ZF_CAP";
pub const ENV_WHIR_CAP: &str = "LAMBDA_VM_ZF_WHIR_CAP";
pub const ENV_FRI: &str = "LAMBDA_VM_ZF_FRI";
pub const ENV_ONE_ROW: &str = "LAMBDA_VM_ZF_ONE_ROW";
pub const ENV_WHIR_FOLDS: &str = "LAMBDA_VM_ZF_WHIR_FOLDS";
pub const ENV_WHIR_STACK: &str = "LAMBDA_VM_ZF_WHIR_STACK";
pub const ENV_WHIR_GRIND: &str = "LAMBDA_VM_ZF_WHIR_GRIND";
pub const ENV_LOGUP: &str = "LAMBDA_VM_ZF_LOGUP";

/// The uniform WHIR schedule's fold, as production configures it
/// (`multilinear_prove::chain_config`); the banner spells the default
/// `uniform4` after it.
pub const PRODUCTION_WHIR_LOG_FOLDING: usize = 4;

/// The WHIR base chains' proof-of-work bits, wherever [`WhirGrind`] places
/// them. The query count subtracts the query grind's bits from what the
/// queries must buy (`multilinear::query_count::num_queries`).
pub const PRODUCTION_WHIR_GRIND_BITS: u8 = 20;

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
    /// LogUp interactions per aux term column, on the STARK base tables only
    /// ([`Self::base_proof_format`]); the LFM chips keep pairs.
    pub logup: LogUpPolicy,
    /// The base proof's commitment hash and its arity-4 cap
    /// ([`BaseFormat`]), on the STARK base tables only; the LFM proofs keep
    /// RPX. No knob sets it: a caller that proves or verifies Poseidon1 says
    /// so in the format it passes ([`Self::P1`]).
    pub base: BaseFormat,
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
    /// The grind bits this placement puts in a chain config.
    pub const fn bits(self) -> GrindBits {
        match self {
            Self::Query => GrindBits::query_only(PRODUCTION_WHIR_GRIND_BITS),
            Self::All => GrindBits::uniform(PRODUCTION_WHIR_GRIND_BITS),
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
    /// read the stack. S2 `one_row=auto` is on because it is the STARK pipeline's
    /// best setting: on top of the other levers it takes about 8 s and
    /// 7–8 GiB of host memory off the STARK block (ABBA block runs). It costs
    /// +3.2 s on the WHIR pipeline (the prover-side cost of one-row LFM
    /// proofs), where `LAMBDA_VM_ZF_ONE_ROW=0` is the faster setting. The
    /// one-row roots of the static preprocessed tables ship at blowup 4 only
    /// ([`crate::tables::STATIC_BLOWUP_FACTORS_ONE_ROW`]), the blowup both
    /// production sites prove at. At another blowup a static table that `auto`
    /// puts on one row (KECCAK_RC) has no root, and proving is an error, never
    /// a recompute.
    /// P2 `whir_grind=query` grinds the WHIR base chains before their queries
    /// only. It costs no proven bits: of a round's three grinds only the query
    /// one raises the proven minimum ([`WhirGrind`]), and the query counts, the
    /// query grinds and the blowups are the legacy ones. It changes no STARK
    /// proof. `LAMBDA_VM_ZF_WHIR_STACK=25` is the stack's rollback and
    /// `LAMBDA_VM_ZF_WHIR_GRIND=all` the grind's.
    /// `logup=k4` commits up to four LogUp interactions per aux column on the
    /// STARK base tables where the rule admits it (degree ≤ blowup + 1): the
    /// median block's phase B −9.56 s (BIG 600, t −27) and the 1× base −0.97 s
    /// (FAST 861), most of it from smaller per-table device sets packing under
    /// the VRAM gate. It reaches the base tables only ([`Self::base_proof_format`]);
    /// the LFM chips keep pairs. `LAMBDA_VM_ZF_LOGUP=pair` is its rollback.
    pub const DEFAULT: Self = Self {
        cap: CapPolicy::Auto,
        whir_cap: CapPolicy::Auto,
        fri: FriMode::Dp,
        one_row: OneRowMode::Auto,
        whir_folds: WhirFolds::First(DEFAULT_WHIR_FIRST_FOLD),
        whir_stack: DEFAULT_WHIR_STACK,
        whir_grind: WhirGrind::Query,
        logup: LogUpPolicy::K4,
        base: BaseFormat::RPX,
    };

    /// [`Self::DEFAULT`] with ZisK's Poseidon1 as the base hash, its trees
    /// capped at 4-ary height 4 ([`BaseFormat::P1`]). Not a default anywhere.
    pub const P1: Self = Self {
        base: BaseFormat::P1,
        ..Self::DEFAULT
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
        logup: LogUpPolicy::Pair,
        base: BaseFormat::RPX,
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
            && self.logup == LogUpPolicy::Pair
            && self.base == BaseFormat::RPX
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
        if let Some(v) = get(ENV_LOGUP) {
            format.logup = v.parse().map_err(|()| {
                format!("{ENV_LOGUP}={v:?}: expected `pair`, `k3`, `k4` or `best`")
            })?;
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
        if !self.logup.is_implemented() {
            out.push(ENV_LOGUP);
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
    /// whir_grind=… logup=…`, each value in the spelling its knob accepts.
    pub fn banner(&self) -> String {
        format!(
            "ZF FORMAT: cap={} whir_cap={} fri={} one_row={} whir_folds={} whir_stack={} \
             whir_grind={} logup={}",
            self.cap,
            self.whir_cap,
            self.fri,
            self.one_row,
            whir_folds_name(&self.whir_folds),
            self.whir_stack.get(),
            self.whir_grind,
            self.logup
        ) + &match self.base {
            BaseFormat::RPX => String::new(),
            base => format!(" base={:?}/c{}", base.hash, base.arity4_cap),
        }
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

    /// The univariate part every LFM proof carries (wraps, nodes, the root):
    /// what `stark::ProofOptions` carries, with the LogUp pair layout. The
    /// LFM chips keep pairs whatever `LAMBDA_VM_ZF_LOGUP` says.
    pub fn proof_format(&self) -> ProofFormat {
        ProofFormat {
            merkle_cap: self.cap,
            fri_mode: self.fri,
            one_row: self.one_row,
            // A test hook only; no knob sets it.
            fri_schedule_override: None,
            logup: LogUpPolicy::Pair,
            // The LFM proofs' own commitments stay RPX.
            base: BaseFormat::RPX,
        }
    }

    /// The univariate part the STARK block's base tables carry: the LFM
    /// format plus the base tables' LogUp arity policy.
    pub fn base_proof_format(&self) -> ProofFormat {
        ProofFormat {
            logup: self.logup,
            base: self.base,
            ..self.proof_format()
        }
    }

    /// The WHIR part: what `multilinear::ChainConfig` carries. `whir_grind`'s
    /// other half, its bits, is a security parameter and goes in where the
    /// config is built ([`crate::multilinear_prove::chain_config_under`]).
    pub fn chain_format(&self) -> ChainFormat {
        ChainFormat {
            cap: self.whir_cap,
            folds: self.whir_folds,
            stack: self.whir_stack,
            nonces: self.whir_grind.nonces(),
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

    /// `options` with this format's base-table univariate fields
    /// ([`Self::base_proof_format`]).
    pub fn base_options(&self, mut options: ProofOptions) -> ProofOptions {
        options.format = self.base_proof_format();
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
                one_row: OneRowMode::Auto,
                whir_folds: WhirFolds::First(FirstFold::new(6).unwrap()),
                whir_stack: StackVars::new(27).unwrap(),
                whir_grind: WhirGrind::Query,
                logup: LogUpPolicy::K4,
                base: BaseFormat::RPX,
            }
        );
        assert_eq!(
            f.banner(),
            "ZF FORMAT: cap=auto whir_cap=auto fri=dp one_row=auto whir_folds=first6 whir_stack=27 \
             whir_grind=query logup=k4"
        );
        assert!(!f.is_legacy());
        assert!(f.unimplemented_levers().is_empty());
    }

    /// Every knob keeps its OFF spelling, and all eight at off are the legacy
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
            (ENV_LOGUP, "pair"),
        ])
        .unwrap();
        assert_eq!(f, ZfFormat::LEGACY);
        assert!(f.is_legacy());
        assert_eq!(
            f.banner(),
            "ZF FORMAT: cap=off whir_cap=off fri=pair one_row=0 whir_folds=uniform4 whir_stack=25 \
             whir_grind=all logup=pair"
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
                ENV_ONE_ROW,
                "0",
                ZfFormat {
                    one_row: OneRowMode::Off,
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
            (ENV_LOGUP, ""),
            (ENV_LOGUP, "k2"),
            (ENV_LOGUP, "k5"),
            (ENV_LOGUP, "4"),
            (ENV_LOGUP, "off"),
            (ENV_LOGUP, "pairs"),
            (ENV_LOGUP, "auto"),
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
            logup: LogUpPolicy::K4,
            base: BaseFormat::RPX,
        };
        assert_eq!(
            f.banner(),
            "ZF FORMAT: cap=auto whir_cap=2 fri=dp one_row=auto whir_folds=first6 whir_stack=26 \
             whir_grind=all logup=k4"
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
            (ENV_LOGUP, fields["logup"]),
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
             whir_grind=query logup=k4"
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
            (4, 112),
            "charged at 16 variables"
        );
        // What the widest table alone would charge: three rounds, one query fewer.
        let widest_alone = ChainConfig::with_security_folds(
            2,
            PRODUCTION_WHIR_LOG_FOLDING,
            ZfFormat::DEFAULT.whir_folds,
            14,
            128,
            multilinear::whir_chain::GrindBits::uniform(20),
        );
        assert_eq!(widest_alone.num_queries, 111);
        // At the production shapes the charge moves nothing, at every accepted
        // stack: the widest table stands at or above the cap, and Q is 112
        // anywhere from 15 to 30 variables.
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
                    112,
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
        // The production default stamps cap=auto, fri=dp and one_row=auto,
        // nothing else.
        let p = ZfFormat::DEFAULT.options(base.clone());
        assert_eq!(p.format.merkle_cap, CapPolicy::Auto);
        assert_eq!(p.format.fri_mode, FriMode::Dp);
        assert_eq!(p.format.one_row, OneRowMode::Auto);
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
        // The production default runs the first6 schedules, to its stack.
        assert_eq!(
            ZfFormat::DEFAULT.whir_schedule_line(),
            ZfFormat {
                whir_stack: ZfFormat::DEFAULT.whir_stack,
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
    /// tallest stack (25) every arm keeps today's Q = 112.
    #[test]
    fn the_production_chain_config_under_each_arm() {
        use crate::multilinear_prove::chain_config_under;
        let today = chain_config_under(&ZfFormat::LEGACY, &[(1, 25)]);
        assert_eq!(today.format, ChainFormat::DEFAULT);
        assert_eq!((today.rounds(25), today.num_queries), (7, 112));
        // The production config (no knob set) is the measured default's:
        // cap auto, first6 — six rounds at 25, Q unchanged at 112.
        let production = crate::multilinear_prove::chain_config(&[(1, 25)]);
        assert_eq!(
            production,
            chain_config_under(&ZfFormat::DEFAULT, &[(1, 25)])
        );
        assert_eq!(production.format, ZfFormat::DEFAULT.chain_format());
        assert_eq!((production.rounds(25), production.num_queries), (6, 112));
        assert_eq!(production.schedule(25), vec![6, 4, 4, 4, 4, 3]);
        assert_eq!(
            (
                production.log_blowup,
                production.log_folding,
                production.grind.query
            ),
            (today.log_blowup, today.log_folding, today.grind.query),
            "the blowup and the query grind do not move with the format"
        );
        // P2: the default grinds before the queries alone, the legacy format
        // before all three challenges.
        assert_eq!(production.grind, GrindBits::query_only(20));
        assert_eq!(today.grind, GrindBits::uniform(20));
        for (name, rounds25) in [("first5", 6), ("first6", 6)] {
            let f = parse(&[(ENV_WHIR_FOLDS, name)]).unwrap();
            let c = chain_config_under(&f, &[(1, 25)]);
            assert_eq!(c.format.folds, f.whir_folds);
            assert_eq!((c.rounds(25), c.num_queries), (rounds25, 112), "{name}");
            assert_eq!(
                (c.log_blowup, c.log_folding, c.grind),
                (today.log_blowup, today.log_folding, f.whir_grind.bits())
            );
            assert_ne!(c.fold_word(), today.fold_word());
        }
    }

    #[test]
    fn production_sites_build_the_default_format_when_nothing_is_set() {
        // No test sets a ZF knob, so the process format is the default and the
        // production constructors must stamp the MEASURED configuration.
        assert_eq!(*ZfFormat::global(), ZfFormat::DEFAULT);
        // The LFM proofs carry the chips' pair layout; the base, the default arity.
        for (site, o, want) in [
            (
                "aggregation_wrap_options",
                crate::lfm::proof::aggregation_wrap_options(),
                ZfFormat::DEFAULT.proof_format(),
            ),
            (
                "block_base_options",
                crate::lfm::proof::block_base_options(),
                ZfFormat::DEFAULT.base_proof_format(),
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
            ("query", GrindBits::query_only(20), NonceLayout::Spent),
            ("all", GrindBits::uniform(20), NonceLayout::Three),
        ] {
            let f = parse(&[(ENV_WHIR_GRIND, v)]).unwrap();
            assert!(f.unimplemented_levers().is_empty(), "{v}");
            assert!(
                f.banner()
                    .split(' ')
                    .any(|kv| kv == format!("whir_grind={v}")),
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
                    c.num_queries, 112,
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

    /// ★ THE OPT-OUT IS TODAY'S CONFIG. `LAMBDA_VM_ZF_WHIR_GRIND=all`, alone,
    /// builds the production WHIR config exactly as it was before P2, spelled
    /// out here as a literal rather than derived through the code under test:
    /// blowup 2^2, fold 4 under first6, Q 112, 20-bit grinds before all three
    /// challenges, the auto cap, stack 27, three nonces a round. The default
    /// differs from it in the grind and the nonce layout, and in nothing else.
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
            },
        };
        let opt_out = parse(&[(ENV_WHIR_GRIND, "all")]).unwrap();
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
                    grind: GrindBits {
                        folding: 0,
                        ood: 0,
                        query: 20,
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

    /// `LAMBDA_VM_ZF_LOGUP`: every spelling parses to its policy and moves no
    /// other field; `k4` is the default, `pair` the legacy value and the rollback. A policy whose
    /// device path this build lacks is reported (and so aborts at
    /// [`ZfFormat::global`]), never proved as another format.
    #[test]
    fn the_logup_knob_parses_and_unbuilt_policies_are_refused() {
        assert_eq!(ZfFormat::DEFAULT.logup, LogUpPolicy::K4);
        assert_eq!(ZfFormat::LEGACY.logup, LogUpPolicy::Pair);
        for (v, want) in [
            ("pair", LogUpPolicy::Pair),
            ("k3", LogUpPolicy::K3),
            ("k4", LogUpPolicy::K4),
            ("best", LogUpPolicy::Best),
            (" K4 ", LogUpPolicy::K4),
        ] {
            let f = parse(&[(ENV_LOGUP, v)]).unwrap();
            assert_eq!(f.logup, want, "{v:?}");
            assert_eq!(
                ZfFormat {
                    logup: ZfFormat::DEFAULT.logup,
                    ..f
                },
                ZfFormat::DEFAULT,
                "the knob moves the LogUp policy and nothing else"
            );
            assert!(
                f.banner().ends_with(&format!(" logup={}", want.name())),
                "{}",
                f.banner()
            );
            assert_eq!(
                f.unimplemented_levers().contains(&ENV_LOGUP),
                !want.is_implemented(),
                "{v:?}"
            );
        }
        assert!(
            parse(&[(ENV_LOGUP, "pair")])
                .unwrap()
                .unimplemented_levers()
                .is_empty()
        );
        // The card splits four composition parts: `k4` is selectable. Three
        // parts have no device split yet: `k3` and `best` must abort.
        const { assert!(stark::proof::options::LOGUP_K4_IMPLEMENTED) };
        const { assert!(!stark::proof::options::LOGUP_K3_IMPLEMENTED) };
        assert!(
            parse(&[(ENV_LOGUP, "k4")])
                .unwrap()
                .unimplemented_levers()
                .is_empty()
        );
        for v in ["k3", "best"] {
            assert_eq!(
                parse(&[(ENV_LOGUP, v)]).unwrap().unimplemented_levers(),
                vec![ENV_LOGUP],
                "{v}"
            );
        }
    }

    /// The LogUp policy reaches the STARK block's base options and nothing
    /// else: every LFM proof (`proof_format`, `options`) keeps the pair
    /// layout under every policy, and the base format differs from the LFM
    /// one in the policy alone.
    #[test]
    fn the_logup_policy_reaches_the_base_options_only() {
        let base = crate::recursion::Preset::Blowup4.options();
        for policy in [
            LogUpPolicy::Pair,
            LogUpPolicy::K3,
            LogUpPolicy::K4,
            LogUpPolicy::Best,
        ] {
            let f = ZfFormat {
                logup: policy,
                ..ZfFormat::DEFAULT
            };
            assert_eq!(f.proof_format().logup, LogUpPolicy::Pair, "{policy}");
            assert_eq!(f.options(base.clone()).format.logup, LogUpPolicy::Pair);
            assert_eq!(f.base_proof_format().logup, policy);
            assert_eq!(f.base_options(base.clone()).format.logup, policy);
            assert_eq!(
                ProofFormat {
                    logup: LogUpPolicy::Pair,
                    ..f.base_proof_format()
                },
                f.proof_format()
            );
            assert_eq!(f.proof_format(), ZfFormat::DEFAULT.proof_format());
        }
        // No knob set: the production base options carry k4; the chips, pairs.
        assert_eq!(
            crate::lfm::proof::block_base_options().format.logup,
            LogUpPolicy::K4
        );
        assert_eq!(
            crate::lfm::proof::aggregation_wrap_options().format.logup,
            LogUpPolicy::Pair
        );
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
}

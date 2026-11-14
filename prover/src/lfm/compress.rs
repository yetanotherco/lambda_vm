//! ★ The block's compression step (C1): one LFM program that verifies the
//! tree's top proof and republishes the top's words, proved by the block's own
//! WHIR prover ([`super::wlfm`]) at a high rate, so the proof a consumer
//! receives is one stacked-WHIR proof instead of the top's per-table STARK.
//!
//! # What C1 is
//!
//! The program is a block node with ONE child, the top: the child's identity
//! (the top program's id, which the verifier derives from the ELF and the
//! statement) is a program constant, its proof is verified by the same leg
//! every node runs ([`super::per_table_aggregator::emit_leg`]), and its
//! published words are republished verbatim, in order, as full words. So C1's
//! words are the top's words, and the block's claim on them
//! ([`super::whir_block::top_claims`]) is checked on C1's words unchanged.
//! There is no bus to close and no sibling to bind: the top closed the bus.
//!
//! # On by default
//!
//! The block ships a compressed file: [`DEFAULT_PLAN`] is what a run proves
//! when no knob says otherwise ([`compress_plan`]): C1 after the top, C2 after
//! C1 when [`DEFAULT_COMPRESS2`] names a grind, the file in
//! [`DEFAULT_WIRE`]'s encoding. Each knob's off word is the rollback:
//! `LAMBDA_VM_BLOCK_COMPRESS=0` is the uncompressed pipeline (no program
//! emitted, no proof made, the top's file ships). The file is another
//! pipeline (`whir_block_tree::WhirBlockCompressedProof`), whose verifier takes
//! the rates and levers as its own constants ([`COMPRESS_RATE`],
//! [`COMPRESS2_RATE`]), never from the proof.
//!
//! # How C1 binds its own program
//!
//! C1's proof is the pipeline's last, verified on the host, which derives C1's
//! program anyway. `LAMBDA_VM_BLOCK_COMPRESS_EVAL` ([`compress_prep_setting`])
//! proves it under [`WlfmPrep::Evaluated`]: the verifier evaluates C1's
//! instruction columns itself, so the proof carries no prepared chain. C1's
//! levers default to [`DEFAULT_C1_LEVERS`] (stacked when C2 verifies C1
//! in-guest); any C1 lever word sets all three; each file names its verifier.
//!
//! # Where C1's chains stop committing
//!
//! `LAMBDA_VM_BLOCK_COMPRESS_FINAL` ([`compress_final_setting`]) proves C1's
//! chains under [`FinalPoly::Vars`]([`COMPRESS_FINAL_VARS`]): each
//! chain sends its folded polynomial once at most that many variables remain,
//! instead of committing and opening the rounds that would fold it to a
//! constant. Its files have verifiers of their own.
//!
//! # The second step (C2)
//!
//! `LAMBDA_VM_BLOCK_COMPRESS2` ([`compress2_setting`]; default
//! [`DEFAULT_COMPRESS2`], grind 26) adds a second W-LFM proof after C1: a program whose one child is C1's proof, which
//! it verifies in-guest ([`super::wlfm_leg`]) and whose words it republishes
//! in order, so its words are still the top's. C1 then stays in-guest
//! checkable (stacked binding, today's chains: any C1 lever knob is refused)
//! and C2 — the proof that ships, verified on the host — takes the format
//! levers: blowup 64 ([`COMPRESS2_RATE`]), the verifier's own columns, the
//! byte-minimising schedule, and a query grind of 22 or 26 bits
//! ([`compress2_levers`]). `LAMBDA_VM_BLOCK_COMPRESS2=0` ships C1's file.

use super::block_node::BlockLayout;
use super::builder::LfmBuilder;
use super::compiler::LfmProgram;
use super::per_table_aggregator::{ChildShape, DerivedChild, declare_leg_arenas, emit_leg};
use super::whir_block::{WhirBlockPlan, derived_of, finish_program, top_claims};
use super::wire::WireFormat;
use super::wlfm::{
    WlfmArtifacts, WlfmBuild, WlfmLevers, WlfmPrep, WlfmProof, WlfmRate, build_wlfm_under,
    prove_wlfm, verify_wlfm,
};
use super::wlfm_leg::{WlfmChild, wlfm_leg_arena, wlfm_leg_program};
use super::word::LfmWord;
use crate::hash_pin::{BlockFamily, ProofFamily};
use multilinear::whir_chain::FinalPoly;

/// The knob that adds the compression step to the block prover.
pub const COMPRESS_ENV: &str = "LAMBDA_VM_BLOCK_COMPRESS";

/// The rate the compressed block's verifier checks C1 at: a constant of the
/// verifier, never read from a proof or the environment.
pub const COMPRESS_RATE: WlfmRate = WlfmRate::Sixteen;

/// [`COMPRESS_ENV`]'s value: unset is [`DEFAULT_COMPRESS`]; `0` or `off` is
/// no compression (the rollback: the top's file ships); `1` or `16` is C1 at
/// blowup 16 ([`COMPRESS_RATE`]); `2` and `4` are C1 at those rates,
/// measurement arms whose files no pinned verifier takes. Anything else is
/// refused.
pub fn compress_setting(value: Option<&str>) -> Result<Option<WlfmRate>, String> {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        None => Ok(DEFAULT_COMPRESS),
        Some(v) => match v.as_str() {
            "" | "0" | "off" => Ok(None),
            "1" | "16" => Ok(Some(WlfmRate::Sixteen)),
            "4" => Ok(Some(WlfmRate::Four)),
            "2" => Ok(Some(WlfmRate::Two)),
            other => Err(format!(
                "{COMPRESS_ENV}={other:?}: expected off, 0, 1, 16, 4 or 2"
            )),
        },
    }
}

/// The knob that proves C1 with its instruction columns evaluated by the
/// verifier instead of a prepared chain.
pub const COMPRESS_EVAL_ENV: &str = "LAMBDA_VM_BLOCK_COMPRESS_EVAL";

/// How the pinned compressed block's verifier binds C1's instruction columns:
/// a constant of the verifier, never read from a proof or the environment.
pub const COMPRESS_PREP: WlfmPrep = WlfmPrep::Stacked;

/// [`COMPRESS_EVAL_ENV`]'s value: unset, `0` or `off` is
/// [`WlfmPrep::Stacked`] (the default); `1` or `on` is
/// [`WlfmPrep::Evaluated`]. Anything else is refused.
pub fn compress_prep_setting(value: Option<&str>) -> Result<WlfmPrep, String> {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        None => Ok(WlfmPrep::Stacked),
        Some(v) => match v.as_str() {
            "" | "0" | "off" => Ok(WlfmPrep::Stacked),
            "1" | "on" => Ok(WlfmPrep::Evaluated),
            other => Err(format!(
                "{COMPRESS_EVAL_ENV}={other:?}: expected off, 0, 1 or on"
            )),
        },
    }
}

/// The knob that proves C1's chains with a final polynomial.
pub const COMPRESS_FINAL_ENV: &str = "LAMBDA_VM_BLOCK_COMPRESS_FINAL";

/// The final polynomial's width C1's chains stop at under
/// [`COMPRESS_FINAL_ENV`]: at most `2^6` variables are sent in the clear.
pub const COMPRESS_FINAL_VARS: u8 = 6;

/// Where the pinned compressed block's verifier expects C1's chains to stop:
/// a constant of the verifier, today's chains.
pub const COMPRESS_FINAL: FinalPoly = FinalPoly::Off;

/// [`COMPRESS_FINAL_ENV`]'s value: unset, `0` or `off` is [`FinalPoly::Off`]
/// (the default); `1`, `on` or `6` is [`FinalPoly::Vars`] at
/// [`COMPRESS_FINAL_VARS`]. Anything else is refused.
pub fn compress_final_setting(value: Option<&str>) -> Result<FinalPoly, String> {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        None => Ok(FinalPoly::Off),
        Some(v) => match v.as_str() {
            "" | "0" | "off" => Ok(FinalPoly::Off),
            "1" | "on" | "6" => Ok(FinalPoly::Vars(COMPRESS_FINAL_VARS)),
            other => Err(format!(
                "{COMPRESS_FINAL_ENV}={other:?}: expected off, 0, 1, on or 6"
            )),
        },
    }
}

/// The knob that ships the compressed file in the compact wire encoding
/// ([`super::wire`]): C1's proof without archive headers.
pub const COMPRESS_WIRE_ENV: &str = "LAMBDA_VM_BLOCK_COMPRESS_WIRE";

/// [`COMPRESS_WIRE_ENV`]'s word: `0` or `off` is the rkyv file; `1`, `on` or
/// `compact` is the compact file; `2` or `dedup` is the compact file with
/// repeated Merkle siblings sent once. Anything else is refused. No word
/// (`None`) parses as the rkyv file; [`compress_plan`] gives an unset knob
/// [`DEFAULT_WIRE`] when the run compresses.
pub fn compress_wire_setting(value: Option<&str>) -> Result<WireFormat, String> {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        None => Ok(WireFormat::Rkyv),
        Some(v) => match v.as_str() {
            "" | "0" | "off" => Ok(WireFormat::Rkyv),
            "1" | "on" | "compact" => Ok(WireFormat::Compact),
            "2" | "dedup" => Ok(WireFormat::Dedup),
            other => Err(format!(
                "{COMPRESS_WIRE_ENV}={other:?}: expected off, 0, 1, on, compact, 2 or dedup"
            )),
        },
    }
}

/// The knob that proves C1's chains on the byte-minimising schedule: every
/// round after the first folds [`COMPRESS_FOLDS_FOLD`] variables and the chain
/// stops at a final polynomial of at most [`COMPRESS_FOLDS_FINAL_VARS`].
pub const COMPRESS_FOLDS_ENV: &str = "LAMBDA_VM_BLOCK_COMPRESS_FOLDS";

/// The fold width after round 0 under [`COMPRESS_FOLDS_ENV`].
pub const COMPRESS_FOLDS_FOLD: u8 = 5;

/// The final polynomial's width under [`COMPRESS_FOLDS_ENV`].
pub const COMPRESS_FOLDS_FINAL_VARS: u8 = 10;

/// The byte-minimising schedule's levers over `prep`.
pub const fn folds_levers(prep: WlfmPrep) -> WlfmLevers {
    WlfmLevers {
        prep,
        final_poly: FinalPoly::Vars(COMPRESS_FOLDS_FINAL_VARS),
        fold: Some(COMPRESS_FOLDS_FOLD),
        grind: None,
    }
}

/// [`COMPRESS_FOLDS_ENV`]'s value: unset, `0` or `off` is today's schedule;
/// `1` or `on` is the byte-minimising one. Anything else is refused.
pub fn compress_folds_setting(value: Option<&str>) -> Result<bool, String> {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        None => Ok(false),
        Some(v) => match v.as_str() {
            "" | "0" | "off" => Ok(false),
            "1" | "on" => Ok(true),
            other => Err(format!(
                "{COMPRESS_FOLDS_ENV}={other:?}: expected off, 0, 1 or on"
            )),
        },
    }
}

/// The knob that adds the second compression step C2 over C1.
pub const COMPRESS2_ENV: &str = "LAMBDA_VM_BLOCK_COMPRESS2";

/// The rate the compressed block's verifier checks C2 at: a constant of the
/// verifier, never read from a proof or the environment.
pub const COMPRESS2_RATE: WlfmRate = WlfmRate::SixtyFour;

/// The query grinds C2 is built at, each a verifier constant of its own files.
pub const COMPRESS2_GRINDS: [u8; 2] = [22, 26];

/// C2's format levers at `grind` query-grind bits: the instruction columns
/// evaluated by the verifier, the byte-minimising schedule (fold 5 after
/// round 0, a final polynomial of at most 2^10 values), and the grind.
pub const fn compress2_levers(grind: u8) -> WlfmLevers {
    WlfmLevers {
        prep: WlfmPrep::Evaluated,
        final_poly: FinalPoly::Vars(COMPRESS_FOLDS_FINAL_VARS),
        fold: Some(COMPRESS_FOLDS_FOLD),
        grind: Some(grind),
    }
}

/// [`COMPRESS2_ENV`]'s word: `0` or `off` is no second step (the rollback:
/// C1's file ships); `1` or `22` is C2 at a 22-bit query grind; `26` at 26
/// bits. Anything else is refused. No word (`None`) parses as no step;
/// [`compress_plan`] gives an unset knob [`DEFAULT_COMPRESS2`] when C1 runs
/// at its default.
pub fn compress2_setting(value: Option<&str>) -> Result<Option<u8>, String> {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        None => Ok(None),
        Some(v) => match v.as_str() {
            "" | "0" | "off" => Ok(None),
            "1" | "22" => Ok(Some(22)),
            "26" => Ok(Some(26)),
            other => Err(format!(
                "{COMPRESS2_ENV}={other:?}: expected off, 0, 1, 22 or 26"
            )),
        },
    }
}

/// The rule on a C2 word: refused unless C1 is on at [`COMPRESS_RATE`] with
/// today's levers (`c1_levers`), since C2 verifies C1 in-guest, which checks
/// a stacked C1 on today's chains only.
pub fn compress2_checked(
    grind: Option<u8>,
    compress: Option<WlfmRate>,
    c1_levers: WlfmLevers,
) -> Result<Option<u8>, String> {
    if grind.is_some() {
        if compress != Some(COMPRESS_RATE) {
            return Err(format!(
                "{COMPRESS2_ENV} adds a step over C1, which needs {COMPRESS_ENV} at blowup 16"
            ));
        }
        if c1_levers != WlfmLevers::DEFAULT {
            return Err(format!(
                "{COMPRESS2_ENV} verifies C1 in-guest: unset {COMPRESS_EVAL_ENV}, \
                 {COMPRESS_FINAL_ENV} and {COMPRESS_FOLDS_ENV}"
            ));
        }
    }
    Ok(grind)
}

/// The compression a run adds when no knob says otherwise: C1 at blowup 16.
pub const DEFAULT_COMPRESS: Option<WlfmRate> = Some(COMPRESS_RATE);

/// C2's query grind when no knob says otherwise; `None` ships C1's file.
/// Grind 26: C2's file is ≈ 296 KB on the median block (132.6 Johnson / 61.2
/// unique-decoding bits a round) for ≈ 0.6 s after C1, with the column
/// engine encoding its blowup-64 codeword. `LAMBDA_VM_BLOCK_COMPRESS2=0`
/// ships C1's file (C1 then takes [`DEFAULT_C1_LEVERS`]'s one-step levers
/// only when no C2 runs; with C2, C1 stays as C2 verifies it in-guest).
pub const DEFAULT_COMPRESS2: Option<u8> = Some(26);

/// C1's levers when no C1 lever knob is set: today's when C2 verifies C1
/// in-guest, else the one-step format (the evaluated columns on the
/// byte-minimising schedule).
pub const DEFAULT_C1_LEVERS: WlfmLevers = match DEFAULT_COMPRESS2 {
    Some(_) => WlfmLevers::DEFAULT,
    None => folds_levers(WlfmPrep::Evaluated),
};

/// The shipped file's encoding when no knob says otherwise and the run
/// compresses: compact, with repeated Merkle siblings sent once.
pub const DEFAULT_WIRE: WireFormat = WireFormat::Dedup;

/// The compression a block run proves and ships: C1's rate (none: the top's
/// file ships) and levers, C2's grind (none: C1's file ships), and the file's
/// encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressPlan {
    pub rate: Option<WlfmRate>,
    pub levers: WlfmLevers,
    pub wire: WireFormat,
    pub c2: Option<u8>,
}

/// What a run proves and ships with no compression knob set.
pub const DEFAULT_PLAN: CompressPlan = CompressPlan {
    rate: DEFAULT_COMPRESS,
    levers: DEFAULT_C1_LEVERS,
    wire: DEFAULT_WIRE,
    c2: DEFAULT_COMPRESS2,
};

/// The rollback plan (`LAMBDA_VM_BLOCK_COMPRESS=0`): no compression, the
/// top's file.
pub const UNCOMPRESSED_PLAN: CompressPlan = CompressPlan {
    rate: None,
    levers: WlfmLevers::DEFAULT,
    wire: WireFormat::Rkyv,
    c2: None,
};

/// ★ The run's compression from its knobs, read through `lookup` (the
/// process environment in production, a map in tests). Every knob unset is
/// [`DEFAULT_PLAN`]. A set knob wins over the defaults:
///
/// - `LAMBDA_VM_BLOCK_COMPRESS=0` is [`UNCOMPRESSED_PLAN`]; any other
///   compression word beside it is refused.
/// - Any C1 lever word (`_EVAL`, `_FINAL`, `_FOLDS`) sets all three (unset
///   ones off) and, with `LAMBDA_VM_BLOCK_COMPRESS2` unset, drops C2's default
///   (C2 verifies only a C1 with today's levers); `_FOLDS` and `_FINAL` refuse
///   each other.
/// - `LAMBDA_VM_BLOCK_COMPRESS2` set to a grind needs C1 at blowup 16 with
///   today's levers; `0` ships C1's file, whose levers are then
///   [`DEFAULT_C1_LEVERS`] unless a lever word says otherwise.
/// - `LAMBDA_VM_BLOCK_COMPRESS_WIRE` unset is [`DEFAULT_WIRE`] when the run
///   compresses, the rkyv file otherwise.
pub fn compress_plan(lookup: impl Fn(&str) -> Option<String>) -> Result<CompressPlan, String> {
    let rate = compress_setting(lookup(COMPRESS_ENV).as_deref())?;
    let words = [COMPRESS_EVAL_ENV, COMPRESS_FINAL_ENV, COMPRESS_FOLDS_ENV].map(&lookup);
    let lever_words = words.iter().any(Option::is_some);
    let prep = compress_prep_setting(words[0].as_deref())?;
    let final_poly = compress_final_setting(words[1].as_deref())?;
    let folds = compress_folds_setting(words[2].as_deref())?;
    if folds && final_poly != FinalPoly::Off {
        return Err(format!(
            "{COMPRESS_FOLDS_ENV} sets C1's final polynomial itself; unset {COMPRESS_FINAL_ENV}"
        ));
    }
    let explicit = if folds {
        folds_levers(prep)
    } else {
        WlfmLevers {
            prep,
            final_poly,
            fold: None,
            grind: None,
        }
    };
    if rate.is_none() && explicit != WlfmLevers::DEFAULT {
        return Err(format!(
            "{COMPRESS_EVAL_ENV}, {COMPRESS_FINAL_ENV} and {COMPRESS_FOLDS_ENV} shape C1, and \
             {COMPRESS_ENV} adds no C1"
        ));
    }
    let c2_word = lookup(COMPRESS2_ENV);
    let c2 = match &c2_word {
        Some(word) => {
            let c1_levers = if lever_words {
                explicit
            } else {
                WlfmLevers::DEFAULT
            };
            compress2_checked(compress2_setting(Some(word))?, rate, c1_levers)?
        }
        None if rate == Some(COMPRESS_RATE) && !lever_words => DEFAULT_COMPRESS2,
        None => None,
    };
    let levers = if lever_words {
        explicit
    } else if c2.is_none() && rate == Some(COMPRESS_RATE) {
        DEFAULT_C1_LEVERS
    } else {
        WlfmLevers::DEFAULT
    };
    let wire = match lookup(COMPRESS_WIRE_ENV) {
        Some(word) => {
            let wire = compress_wire_setting(Some(&word))?;
            if rate.is_none() && wire != WireFormat::Rkyv {
                return Err(format!(
                    "{COMPRESS_WIRE_ENV} encodes the compressed file, and {COMPRESS_ENV} adds no C1"
                ));
            }
            wire
        }
        None if rate.is_some() => DEFAULT_WIRE,
        None => WireFormat::Rkyv,
    };
    Ok(CompressPlan {
        rate,
        levers,
        wire,
        c2,
    })
}

/// [`compress_plan`] over the process's environment.
pub fn compress_plan_from_env() -> Result<CompressPlan, String> {
    compress_plan(|name| std::env::var(name).ok())
}

/// Emits C1 into `b`: the top's arenas, its leg, and every published word of
/// the top republished as a whole word, in order. `top` must publish `layout`
/// (the top layout: no bus sum).
pub fn emit_compress(b: &mut LfmBuilder, top: &ChildShape<'_>, layout: BlockLayout) {
    assert!(
        !layout.has_sum,
        "C1 compresses the top, which publishes no sum"
    );
    assert_eq!(
        top.num_public_words,
        layout.total(),
        "the top publishes the top layout"
    );
    let arenas = declare_leg_arenas(b, top);
    let leg = emit_leg(b, top, &arenas);
    for word in &leg.publics {
        let lanes = &word.lanes;
        let cell = b.pack_word([lanes[0], lanes[1], lanes[2], lanes[3]]);
        b.public(cell);
    }
}

/// C1 over `top` (the top program's shape, derived at the top layout), emitted
/// under the block family's wrap hash and validated.
pub fn compress_program(top: &DerivedChild, layout: BlockLayout) -> Result<LfmProgram, String> {
    let mut b = LfmBuilder::new().with_wrap_hash(BlockFamily::WRAP);
    emit_compress(&mut b, &top.shape(), layout);
    finish_program(b)
}

/// C1's program and its W-LFM build (artifacts and the prover's prepared
/// stacks).
pub struct CompressBuild {
    pub program: LfmProgram,
    pub build: WlfmBuild,
}

/// The top's shape at the top layout, from its artifacts under `wrap_opts`.
pub fn top_child(
    plan: &WhirBlockPlan,
    top_artifacts: &super::registry::LfmArtifacts,
    wrap_opts: &crate::ProofOptions,
) -> Result<DerivedChild, String> {
    derived_of(top_artifacts, wrap_opts, plan.top_layout().total())
}

/// ★ C1's program and build at `rate` under `levers`, for the top `top`: ONE
/// derivation for the prover and the verifier (whose prepared commits run on
/// the host, [`compressed_artifacts`]).
pub fn build_compress(
    top: &DerivedChild,
    layout: BlockLayout,
    rate: WlfmRate,
    levers: WlfmLevers,
) -> Result<CompressBuild, String> {
    build_compress_with(top, layout, rate, levers, false)
}

fn build_compress_with(
    top: &DerivedChild,
    layout: BlockLayout,
    rate: WlfmRate,
    levers: WlfmLevers,
    on_host: bool,
) -> Result<CompressBuild, String> {
    let program = compress_program(top, layout)?;
    let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
    let build = build_wlfm_under(&program, hasher, rate, levers, on_host)?;
    Ok(CompressBuild { program, build })
}

/// C1 executed over the top's arena (`top_arena`: its harvested words,
/// `harvest::try_child_arena_words`) and proved. A top that does not verify
/// under C1's leg does not execute: the error says where.
pub fn prove_compress(c: &CompressBuild, top_arena: &[Vec<LfmWord>]) -> Result<WlfmProof, String> {
    prove_wlfm(&c.program, &c.build, top_arena)
}

/// ★ A compressed block verified against `plan` (the verifier's own, derived
/// from the ELF and the statement): the top program derived under
/// `wrap_opts`, C1 built over it at `rate` under `levers`
/// ([`compressed_artifacts`]), and the proof checked against them
/// ([`verify_compressed_against`]).
pub fn verify_compressed(
    plan: &WhirBlockPlan,
    wrap_opts: &crate::ProofOptions,
    rate: WlfmRate,
    levers: WlfmLevers,
    proof: &WlfmProof,
) -> Result<(), String> {
    let artifacts = compressed_artifacts(plan, wrap_opts, rate, levers)?;
    verify_compressed_against(plan, &artifacts, proof)
}

/// C1's artifacts as a verifier derives them for `plan`: the top program's
/// artifacts under `wrap_opts`, its shape at the top layout, C1 over it at
/// `rate` under `levers`, its prepared stacks (if any) committed on the host.
pub fn compressed_artifacts(
    plan: &WhirBlockPlan,
    wrap_opts: &crate::ProofOptions,
    rate: WlfmRate,
    levers: WlfmLevers,
) -> Result<WlfmArtifacts, String> {
    let top = plan.derive_top(wrap_opts)?;
    let child = top_child(plan, &top, wrap_opts)?;
    Ok(
        build_compress_with(&child, plan.top_layout(), rate, levers, true)?
            .build
            .artifacts,
    )
}

/// `proof` verified against C1's `artifacts` over its own words, and those
/// words checked to claim `plan`'s block ([`top_claims`]).
pub fn verify_compressed_against(
    plan: &WhirBlockPlan,
    artifacts: &WlfmArtifacts,
    proof: &WlfmProof,
) -> Result<(), String> {
    verify_wlfm(artifacts, proof, &proof.public_words)
        .map_err(|e| format!("the compressed proof does not verify against C1: {e}"))?;
    if !top_claims(plan, &proof.public_words) {
        return Err("the compressed proof does not claim this block".to_string());
    }
    Ok(())
}

/// C1 as C2's child: its artifacts (every shape, root and the id are C2's
/// program constants) and its word count, the top layout's.
pub fn c1_child(c1: &WlfmArtifacts, layout: BlockLayout) -> WlfmChild {
    WlfmChild {
        artifacts: c1.clone(),
        num_public_words: layout.total(),
    }
}

/// ★ C2's program and build at [`COMPRESS2_RATE`] under
/// [`compress2_levers`]`(grind)`, over C1 (`c1`, C1's artifacts): ONE
/// derivation for the prover and the verifier. C2's program is the W-leg over
/// C1 with C1's words republished in order ([`wlfm_leg_program`]).
pub fn build_compress2(
    c1: &WlfmArtifacts,
    layout: BlockLayout,
    grind: u8,
) -> Result<CompressBuild, String> {
    build_compress2_with(c1, layout, grind, false)
}

fn build_compress2_with(
    c1: &WlfmArtifacts,
    layout: BlockLayout,
    grind: u8,
    on_host: bool,
) -> Result<CompressBuild, String> {
    if !COMPRESS2_GRINDS.contains(&grind) {
        return Err(format!(
            "C2 is built at a grind of {COMPRESS2_GRINDS:?} bits, not {grind}"
        ));
    }
    let program = wlfm_leg_program(&c1_child(c1, layout))?;
    let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
    let build = build_wlfm_under(
        &program,
        hasher,
        COMPRESS2_RATE,
        compress2_levers(grind),
        on_host,
    )?;
    Ok(CompressBuild { program, build })
}

/// C2 executed over C1's proof (`c1_proof`, as the W-leg's arena) and proved.
/// A C1 proof that does not verify under the leg does not execute.
pub fn prove_compress2(
    c2: &CompressBuild,
    c1: &WlfmArtifacts,
    layout: BlockLayout,
    c1_proof: &WlfmProof,
) -> Result<WlfmProof, String> {
    let arena = wlfm_leg_arena(&c1_child(c1, layout), c1_proof)?;
    prove_wlfm(&c2.program, &c2.build, &[arena])
}

/// C2's artifacts as a verifier derives them for `plan`: C1's
/// ([`compressed_artifacts`] at [`COMPRESS_RATE`], today's levers, its
/// prepared stacks committed on the host), then C2 over them at `grind`.
pub fn compressed2_artifacts(
    plan: &WhirBlockPlan,
    wrap_opts: &crate::ProofOptions,
    grind: u8,
) -> Result<WlfmArtifacts, String> {
    let c1 = compressed_artifacts(plan, wrap_opts, COMPRESS_RATE, WlfmLevers::DEFAULT)?;
    Ok(build_compress2_with(&c1, plan.top_layout(), grind, true)?
        .build
        .artifacts)
}

/// ★ A twice-compressed block verified against `plan`: C2's artifacts derived
/// ([`compressed2_artifacts`]) and the proof checked against them and to
/// claim the block ([`verify_compressed_against`]: C2's words are the top's).
pub fn verify_compressed2(
    plan: &WhirBlockPlan,
    wrap_opts: &crate::ProofOptions,
    grind: u8,
    proof: &WlfmProof,
) -> Result<(), String> {
    let artifacts = compressed2_artifacts(plan, wrap_opts, grind)?;
    verify_compressed_against(plan, &artifacts, proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The second step's word parses, refuses nonsense, needs C1 at blowup 16
    /// with today's levers, and its levers are the evaluated columns on the
    /// byte-minimising schedule at the grind.
    #[test]
    fn the_compress2_word_parses_and_refuses_nonsense() {
        assert_eq!(compress2_setting(None), Ok(None));
        for off in ["", "0", "off", " OFF "] {
            assert_eq!(compress2_setting(Some(off)), Ok(None), "{off:?}");
        }
        for on in ["1", "22", " 22 "] {
            assert_eq!(compress2_setting(Some(on)), Ok(Some(22)), "{on:?}");
        }
        assert_eq!(compress2_setting(Some("26")), Ok(Some(26)));
        for bad in ["2", "18", "20", "64", "on", "yes"] {
            assert!(compress2_setting(Some(bad)).is_err(), "{bad:?}");
        }
        let c1 = Some(COMPRESS_RATE);
        let today = WlfmLevers::DEFAULT;
        assert_eq!(compress2_checked(None, None, today), Ok(None));
        assert_eq!(compress2_checked(Some(22), c1, today), Ok(Some(22)));
        assert!(compress2_checked(Some(22), None, today).is_err());
        assert!(compress2_checked(Some(22), Some(WlfmRate::Four), today).is_err());
        for c1_levers in [
            WlfmLevers::with_prep(WlfmPrep::Evaluated),
            folds_levers(WlfmPrep::Stacked),
            WlfmLevers {
                final_poly: FinalPoly::Vars(COMPRESS_FINAL_VARS),
                ..today
            },
        ] {
            assert!(compress2_checked(Some(26), c1, c1_levers).is_err());
            assert_eq!(compress2_checked(None, c1, c1_levers), Ok(None));
        }
        let l = compress2_levers(26);
        assert_eq!(l.prep, WlfmPrep::Evaluated);
        assert_eq!(l.fold, Some(5));
        assert_eq!(l.final_poly, FinalPoly::Vars(10));
        assert_eq!(l.grind, Some(26));
        assert_eq!(COMPRESS2_RATE, WlfmRate::SixtyFour);
        assert_eq!(COMPRESS2_GRINDS, [22, 26]);
    }

    fn plan(words: &[(&str, &str)]) -> Result<CompressPlan, String> {
        compress_plan(|name| {
            words
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.to_string())
        })
    }

    /// ★ No knob is the default plan, and each knob's off word rolls its step
    /// back: no compression at all, or C1's file without C2.
    #[test]
    fn the_default_plan_and_each_knobs_rollback() {
        assert_eq!(plan(&[]), Ok(DEFAULT_PLAN));
        assert_eq!(DEFAULT_PLAN.rate, Some(COMPRESS_RATE));
        assert_eq!(DEFAULT_PLAN.wire, WireFormat::Dedup);
        if let Some(grind) = DEFAULT_PLAN.c2 {
            assert!(COMPRESS2_GRINDS.contains(&grind));
            assert_eq!(
                DEFAULT_PLAN.levers,
                WlfmLevers::DEFAULT,
                "C2 verifies C1 in-guest"
            );
        } else {
            assert_eq!(DEFAULT_PLAN.levers, folds_levers(WlfmPrep::Evaluated));
        }
        for off in ["0", "off", ""] {
            assert_eq!(
                plan(&[(COMPRESS_ENV, off)]),
                Ok(UNCOMPRESSED_PLAN),
                "{off:?}"
            );
        }
        // C2 rolled back: C1's file, C1 at its default levers for a file the
        // host verifies.
        let c1_only = plan(&[(COMPRESS2_ENV, "0")]).expect("C1 alone");
        assert_eq!(c1_only.c2, None);
        assert_eq!(c1_only.rate, Some(COMPRESS_RATE));
        assert_eq!(c1_only.levers, DEFAULT_C1_LEVERS);
        // The encoding rolled back: the rkyv file.
        assert_eq!(
            plan(&[(COMPRESS_WIRE_ENV, "0")]).map(|p| p.wire),
            Ok(WireFormat::Rkyv)
        );
    }

    /// A set knob wins over the defaults: a C1 lever word drops C2's default
    /// and sets all three levers; an explicit grind needs today's C1; a
    /// compression word beside `LAMBDA_VM_BLOCK_COMPRESS=0` is refused.
    #[test]
    fn a_set_knob_wins_over_the_default_plan() {
        let eval = plan(&[(COMPRESS_EVAL_ENV, "1")]).expect("C1 evaluated");
        assert_eq!(eval.c2, None);
        assert_eq!(eval.levers, WlfmLevers::with_prep(WlfmPrep::Evaluated));
        let folds = plan(&[(COMPRESS_FOLDS_ENV, "1")]).expect("C1 on the schedule");
        assert_eq!(folds.levers, folds_levers(WlfmPrep::Stacked));
        assert_eq!(folds.c2, None);
        for grind in ["22", "26"] {
            let p = plan(&[(COMPRESS2_ENV, grind)]).expect("C2 at a grind");
            assert_eq!(p.c2, Some(grind.parse().unwrap()));
            assert_eq!(p.levers, WlfmLevers::DEFAULT);
            assert!(plan(&[(COMPRESS2_ENV, grind), (COMPRESS_EVAL_ENV, "1")]).is_err());
            assert!(plan(&[(COMPRESS2_ENV, grind), (COMPRESS_ENV, "0")]).is_err());
            assert!(plan(&[(COMPRESS2_ENV, grind), (COMPRESS_ENV, "4")]).is_err());
        }
        assert!(plan(&[(COMPRESS_FOLDS_ENV, "1"), (COMPRESS_FINAL_ENV, "1")]).is_err());
        for word in [COMPRESS_EVAL_ENV, COMPRESS_FOLDS_ENV, COMPRESS_FINAL_ENV] {
            assert!(plan(&[(COMPRESS_ENV, "0"), (word, "1")]).is_err(), "{word}");
            assert!(plan(&[(COMPRESS_ENV, "0"), (word, "0")]).is_ok(), "{word}");
        }
        assert!(plan(&[(COMPRESS_ENV, "0"), (COMPRESS_WIRE_ENV, "dedup")]).is_err());
        assert_eq!(
            plan(&[(COMPRESS_ENV, "0"), (COMPRESS2_ENV, "0")]),
            Ok(UNCOMPRESSED_PLAN)
        );
        // A measurement rate keeps today's C1 and no C2.
        let four = plan(&[(COMPRESS_ENV, "4")]).expect("C1 at blowup 4");
        assert_eq!((four.levers, four.c2), (WlfmLevers::DEFAULT, None));
        assert!(plan(&[(COMPRESS2_ENV, "18")]).is_err());
    }

    #[test]
    fn the_compress_knob_is_on_unless_set_off_and_refuses_nonsense() {
        assert_eq!(compress_setting(None), Ok(DEFAULT_COMPRESS));
        assert_eq!(DEFAULT_COMPRESS, Some(WlfmRate::Sixteen));
        for off in ["", "0", "off", " OFF "] {
            assert_eq!(compress_setting(Some(off)), Ok(None), "{off:?}");
        }
        assert_eq!(compress_setting(Some("1")), Ok(Some(WlfmRate::Sixteen)));
        assert_eq!(compress_setting(Some("16")), Ok(Some(WlfmRate::Sixteen)));
        assert_eq!(compress_setting(Some("4")), Ok(Some(WlfmRate::Four)));
        assert_eq!(compress_setting(Some("2")), Ok(Some(WlfmRate::Two)));
        for bad in ["8", "32", "yes", "on", "-1"] {
            assert!(compress_setting(Some(bad)).is_err(), "{bad:?}");
        }
        assert_eq!(COMPRESS_RATE, WlfmRate::Sixteen);
    }

    /// The evaluated binding is off unless set, refuses nonsense, and the
    /// pinned verifier's constant is today's stacked binding.
    #[test]
    fn the_compress_eval_knob_is_off_unless_set_and_refuses_nonsense() {
        assert_eq!(compress_prep_setting(None), Ok(WlfmPrep::Stacked));
        for off in ["", "0", "off", " OFF "] {
            assert_eq!(compress_prep_setting(Some(off)), Ok(WlfmPrep::Stacked));
        }
        for on in ["1", "on", " ON "] {
            assert_eq!(compress_prep_setting(Some(on)), Ok(WlfmPrep::Evaluated));
        }
        for bad in ["2", "yes", "evaluated", "-1"] {
            assert!(compress_prep_setting(Some(bad)).is_err(), "{bad:?}");
        }
        assert_eq!(COMPRESS_PREP, WlfmPrep::Stacked);
    }

    /// The final-polynomial knob is off unless set, refuses nonsense, and the
    /// pinned verifier's constant is today's chains.
    #[test]
    fn the_compress_final_knob_is_off_unless_set_and_refuses_nonsense() {
        assert_eq!(compress_final_setting(None), Ok(FinalPoly::Off));
        for off in ["", "0", "off", " OFF "] {
            assert_eq!(compress_final_setting(Some(off)), Ok(FinalPoly::Off));
        }
        for on in ["1", "on", "6", " ON "] {
            assert_eq!(
                compress_final_setting(Some(on)),
                Ok(FinalPoly::Vars(COMPRESS_FINAL_VARS))
            );
        }
        for bad in ["2", "8", "yes", "-1"] {
            assert!(compress_final_setting(Some(bad)).is_err(), "{bad:?}");
        }
        assert_eq!(COMPRESS_FINAL, FinalPoly::Off);
        assert_eq!(
            FinalPoly::vars(COMPRESS_FINAL_VARS),
            Some(FinalPoly::Vars(6))
        );
    }

    /// The schedule knob is off unless set, refuses nonsense, and its levers
    /// are fold 5 with a final polynomial of 10 variables.
    #[test]
    fn the_compress_folds_knob_is_off_unless_set_and_refuses_nonsense() {
        assert_eq!(compress_folds_setting(None), Ok(false));
        for off in ["", "0", "off", " OFF "] {
            assert_eq!(compress_folds_setting(Some(off)), Ok(false));
        }
        for on in ["1", "on", " ON "] {
            assert_eq!(compress_folds_setting(Some(on)), Ok(true));
        }
        for bad in ["2", "5", "yes"] {
            assert!(compress_folds_setting(Some(bad)).is_err(), "{bad:?}");
        }
        let l = folds_levers(WlfmPrep::Stacked);
        assert_eq!(l.fold, Some(5));
        assert_eq!(l.final_poly, FinalPoly::Vars(10));
        assert_eq!(
            FinalPoly::vars(COMPRESS_FOLDS_FINAL_VARS),
            Some(l.final_poly)
        );
    }

    /// The file-encoding word parses and refuses nonsense.
    #[test]
    fn the_compress_wire_word_parses_and_refuses_nonsense() {
        assert_eq!(compress_wire_setting(None), Ok(WireFormat::Rkyv));
        for off in ["", "0", "off", " OFF "] {
            assert_eq!(compress_wire_setting(Some(off)), Ok(WireFormat::Rkyv));
        }
        for on in ["1", "on", "compact", " Compact "] {
            assert_eq!(compress_wire_setting(Some(on)), Ok(WireFormat::Compact));
        }
        for on in ["2", "dedup", " DEDUP "] {
            assert_eq!(compress_wire_setting(Some(on)), Ok(WireFormat::Dedup));
        }
        for bad in ["3", "rkyv", "yes"] {
            assert!(compress_wire_setting(Some(bad)).is_err(), "{bad:?}");
        }
    }
}

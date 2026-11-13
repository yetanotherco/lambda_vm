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
//! # Off by default
//!
//! `LAMBDA_VM_BLOCK_COMPRESS` ([`compress_setting`]) unset is today's pipeline
//! byte for byte: no program is emitted, no proof is made, the top ships. Set,
//! the prover adds C1 after the top; the file it ships is another pipeline
//! (`whir_block_tree::WhirBlockCompressedProof`), whose verifier takes the rate
//! as its own constant ([`COMPRESS_RATE`]), never from the proof.
//!
//! # How C1 binds its own program
//!
//! C1's proof is the pipeline's last, verified on the host, which derives C1's
//! program anyway. `LAMBDA_VM_BLOCK_COMPRESS_EVAL` ([`compress_prep_setting`];
//! default off) proves it under [`WlfmPrep::Evaluated`]: the verifier evaluates
//! C1's instruction columns itself, so the proof carries no prepared chain.
//! Unset is [`WlfmPrep::Stacked`], today's C1 byte for byte; the pinned
//! verifier checks [`COMPRESS_PREP`], the evaluated arm has its own verifier.
//!
//! # Where C1's chains stop committing
//!
//! `LAMBDA_VM_BLOCK_COMPRESS_FINAL` ([`compress_final_setting`]; default off)
//! proves C1's chains under [`FinalPoly::Vars`]([`COMPRESS_FINAL_VARS`]): each
//! chain sends its folded polynomial once at most that many variables remain,
//! instead of committing and opening the rounds that would fold it to a
//! constant. Unset is today's chains. Its files have verifiers of their own.

use super::block_node::BlockLayout;
use super::builder::LfmBuilder;
use super::compiler::LfmProgram;
use super::per_table_aggregator::{ChildShape, DerivedChild, declare_leg_arenas, emit_leg};
use super::whir_block::{WhirBlockPlan, derived_of, finish_program, top_claims};
use super::wlfm::{
    WlfmArtifacts, WlfmBuild, WlfmLevers, WlfmPrep, WlfmProof, WlfmRate, build_wlfm_under,
    prove_wlfm, verify_wlfm,
};
use super::word::LfmWord;
use crate::hash_pin::{BlockFamily, ProofFamily};
use multilinear::whir_chain::FinalPoly;

/// The knob that adds the compression step to the block prover.
pub const COMPRESS_ENV: &str = "LAMBDA_VM_BLOCK_COMPRESS";

/// The rate the compressed block's verifier checks C1 at: a constant of the
/// verifier, never read from a proof or the environment.
pub const COMPRESS_RATE: WlfmRate = WlfmRate::Sixteen;

/// [`COMPRESS_ENV`]'s value: unset, `0` or `off` is no compression (the
/// default); `1` or `16` is C1 at blowup 16 ([`COMPRESS_RATE`]); `2` and `4`
/// are C1 at those rates, measurement arms whose files no pinned verifier
/// takes. Anything else is refused.
pub fn compress_setting(value: Option<&str>) -> Result<Option<WlfmRate>, String> {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        None => Ok(None),
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

/// [`compress_setting`] of the process's environment.
pub fn compress_from_env() -> Result<Option<WlfmRate>, String> {
    compress_setting(std::env::var(COMPRESS_ENV).ok().as_deref())
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

/// C1's format levers from the process's environment
/// ([`compress_prep_setting`], [`compress_final_setting`]), refused when one
/// is set with no compression step to apply it to.
pub fn compress_levers_from_env(compress: Option<WlfmRate>) -> Result<WlfmLevers, String> {
    let levers = WlfmLevers {
        prep: compress_prep_setting(std::env::var(COMPRESS_EVAL_ENV).ok().as_deref())?,
        final_poly: compress_final_setting(std::env::var(COMPRESS_FINAL_ENV).ok().as_deref())?,
    };
    if compress.is_none() && levers != WlfmLevers::DEFAULT {
        return Err(format!(
            "{COMPRESS_EVAL_ENV} and {COMPRESS_FINAL_ENV} shape C1, and {COMPRESS_ENV} adds no C1"
        ));
    }
    Ok(levers)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_compress_knob_is_off_unless_set_and_refuses_nonsense() {
        assert_eq!(compress_setting(None), Ok(None));
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
}

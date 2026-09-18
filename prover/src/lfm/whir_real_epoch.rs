//! # ⚠ WHY THIS IS NOT IN `whir_epoch_tests`
//!
//! It was, and a production builder cannot take its input type from a
//! `#[cfg(test)]` module. [`WhirRealEpoch`] is the input to V1's level-0
//! program builder, so the driver that produces it compiles in every build and
//! the tests import it from here. Nothing else moved: the tests are unchanged
//! and are the control that says so.
//!
//! ⚠ NOT `whir_epoch.rs` — that name is V1's, for the emitter.
//!
//! # ⛔ THE `dead_code` ALLOW, AND WHEN IT COMES OUT
//!
//! Moving out of `#[cfg(test)]` put this module in the LIBRARY target, where
//! every item is currently unreachable: the only callers are the tests, and
//! V1's level-0 program builder — the production caller this move exists for —
//! does not exist yet. `make lint`'s first arm builds `--all-targets`, so the
//! lib target is compiled on its own and `-D warnings` turns that into seven
//! hard errors.
//!
//! The allow is therefore SCOPED TO THIS MODULE and temporary. It comes out
//! the moment `whir_epoch_program` calls `epoch_airs_for` and takes a
//! `WhirRealEpoch`, which is the whole point of the move; if it is still here
//! after that lands, something did not get wired.
//!
//! ⚠ What it costs while it stands: a genuinely unused item added here would
//! not be reported. That is why it is a module attribute with this note rather
//! than an `#[allow]` sprinkled per item, where it would quietly outlive its
//! reason.
#![allow(dead_code)]

//! The WHIR level-0 driver: one wrap input per epoch of a WHIR continuation.
//!
//! The STARK analogue is [`super::epoch_tests::real_epoch_from_continuation`],
//! and this sits beside it rather than inside it so the two lineages do not
//! collide in one file while both are being written.
//!
//! # What a driver owes, and what it must refuse to take
//!
//! Everything here is a value the VERIFIER computes for itself. The epoch's
//! starting registers are the ELF's for epoch 0 and the previous epoch's PROVED
//! `reg_fini` after that; `is_final` is a position, not a claim; the label is
//! the index. Nothing is read out of the bundle that the bundle is supposed to
//! be checked against — which is the same rule
//! [`crate::multilinear_continuation::verify_continuation`] follows, and the
//! reason it can verify a run from the ELF alone.
//!
//! # ⛔ The hash agreement, and why it is not a tag
//!
//! `L0-integration-design` §3 says the driver "reads the bundle's hash tag and
//! REFUSES a keccak bundle". There is no such tag and there cannot sensibly be
//! one: [`crate::multilinear_continuation::ContinuationProof`] carries
//! `epochs`, `global`, `num_private_input_pages` and `touched_page_bases`, and
//! a WHIR proof's BYTES are hash-agnostic by design — the byte gate exists to
//! assert exactly that, and both arms serialise to the same length.
//!
//! Nor can the process knob stand in for it. `whir_hash_knob::selected()` is
//! read once and cached, so it says what THIS PROCESS proves under, not what
//! some bundle was proven under.
//!
//! So the agreement is CRYPTOGRAPHIC here, not a label: every epoch this driver
//! harvests is VERIFIED first, and a bundle proven under a different hash fails
//! that verification because the transcript hash is part of the configuration
//! and every challenge diverges. A label can be wrong; this cannot.
//!
//! ★ AND IT IS MEASURED, not argued. [`real_epoch_from_whir_continuation_under`]
//! takes `H` and verifies under it;
//! `tests::an_epoch_proven_under_keccak_is_refused_when_harvested_under_rpx`
//! proves a real bundle's epochs with a literal `prove_epoch::<KeccakWhir>`,
//! harvests that bundle at `RpxWhir` and requires the refusal — beside the
//! same bundle harvested at `KeccakWhir`, which must be ACCEPTED. The control
//! is not decoration: `verify_epoch_bookend` collapses every failure to
//! `Ok(None)`, so without it a driver that refused everything would pass.
//!
//! # The process knob is REPORTED, never obeyed
//!
//! Level 0's wrap proofs commit under the RPX block hasher, so a process left
//! at keccak is usually an operator's mistake — but it is not this driver's to
//! decide, because a keccak bundle harvested under keccak is perfectly valid
//! and merely not the production posture. [`whir_process_posture_note`] names
//! it and refuses nothing.

use crate::multilinear_continuation::{ContinuationProof, EpochProof};
use crate::tables::local_to_global::epoch_label;
use crate::tables::register;
use executor::elf::Elf;
use multilinear::whir_chain::ChainConfig;
use multilinear::whir_hash::WhirHash;
use stark::config::Commitment;

/// Where an epoch sits in its run, as the VERIFIER derives it.
///
/// Not read from the bundle: `register_init` chains from the previous epoch's
/// proved `reg_fini`, `is_final` is the position, and the label is the index.
#[derive(Debug, Clone)]
pub(crate) struct WhirChainPosition {
    pub(crate) register_init: Vec<u32>,
    pub(crate) is_final: bool,
    pub(crate) label: u64,
}

/// One epoch's level-0 wrap input: the proof, everything its statement absorbs,
/// and the values the verifier derived rather than read.
///
/// The statement half is exactly
/// [`crate::multilinear_continuation::absorb_epoch`]'s argument list, because
/// the in-guest verifier's first job is to replay that absorb and any field it
/// cannot see is a challenge it cannot reproduce.
pub(crate) struct WhirRealEpoch {
    /// The epoch proof itself, cloned out of the bundle.
    pub(crate) proof: EpochProof,
    /// `statement::elf_digest(elf_bytes)` — the program this run was of.
    pub(crate) elf_digest: [u8; 32],
    /// The epoch's position in its run, derived.
    pub(crate) position: WhirChainPosition,
    /// The parameters the epoch's argument ran at, rebuilt from the shapes the
    /// proof states rather than carried: `chain_config` over
    /// `(width, num_vars)` per table.
    pub(crate) config: ChainConfig,
    /// DECODE's univariate preprocessed commitment for this (ELF, options)
    /// pair, taken once per bundle rather than once per epoch.
    pub(crate) decode_commitment: Commitment,
    /// The inner ELF's entry point — `program_id`'s `pc_start`.
    pub(crate) pc_start: u64,
    /// `(width, num_vars)` per table, in sub-proof order: the widths are the
    /// AIRs' and the heights are the proof's. Kept because the config is a
    /// function of them and a caller that wants to check one needs the other.
    pub(crate) shapes: Vec<(usize, usize)>,
}

impl WhirRealEpoch {
    /// The epoch's own published bytes, which the run's output concatenates.
    pub(crate) fn public_output(&self) -> &[u8] {
        &self.proof.public_output
    }

    /// How many tables this epoch's argument covers — the width of everything
    /// the guest walks.
    pub(crate) fn num_tables(&self) -> usize {
        self.proof.table_num_vars.len()
    }
}

/// The verifier's own chain position for epoch `index`.
///
/// `None` when the index is out of range. Epoch 0 starts from the ELF's entry
/// point; every later epoch starts from the PREVIOUS epoch's proved `reg_fini`,
/// which is what ties one epoch to the next and carries the commit index.
pub(crate) fn whir_epoch_chain_position(
    bundle: &ContinuationProof,
    elf: &Elf,
    index: usize,
) -> Option<WhirChainPosition> {
    let epochs = &bundle.epochs;
    if index >= epochs.len() {
        return None;
    }
    let register_init = if index == 0 {
        register::register_init_from_entry_point(elf.entry_point)
    } else {
        epochs[index - 1].reg_fini.clone()
    };
    Some(WhirChainPosition {
        register_init,
        is_final: index + 1 == epochs.len(),
        label: epoch_label(index as u64),
    })
}

/// [`WhirRealEpoch`] for epoch `epoch_index` of an existing WHIR continuation
/// bundle.
///
/// The signature mirrors [`super::epoch_tests::real_epoch_from_continuation`]
/// so the two level-0 drivers read the same way; the bundle type is the WHIR
/// one, which is a distinct rkyv type from the STARK bundle.
///
/// ★ THE EPOCH IS VERIFIED BEFORE IT IS HARVESTED. A wrap input built from a
/// proof nobody checked would push the failure into the guest, where it costs a
/// whole wrap prove to discover and reads as an emitter bug. It is also the
/// hash agreement (see this module's header): a bundle proven under another
/// hash fails here.
///
/// `decode_commitment`: `Some` reuses a root computed once per bundle — it is a
/// function of (ELF, options) only; `None` computes it here. The STARK driver's
/// note applies verbatim: with `None` a walk over every epoch rebuilds it once
/// per epoch for one distinct value.
pub(crate) fn real_epoch_from_whir_continuation(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    bundle: &ContinuationProof,
    epoch_index: usize,
    decode_commitment: Option<Commitment>,
) -> Result<WhirRealEpoch, String> {
    crate::with_whir_hash!(|H| {
        real_epoch_from_whir_continuation_under::<H>(
            opts,
            elf_bytes,
            bundle,
            epoch_index,
            decode_commitment,
            // ⚠ NOT AN OVERSIGHT. This form has no `H` to name, so
            // `&DecodePrepared<H>` cannot appear in its signature at all. The
            // reuse is available at the generic entry point, which is what a
            // level-0 walk calls once it knows its hash, and nowhere it would
            // be a lie.
            None,
        )
    })
}

/// [`real_epoch_from_whir_continuation`], told which hash to verify under.
///
/// ★ THIS IS WHERE THE HASH AGREEMENT LIVES, and it is why the function is
/// generic rather than reading the knob. `whir_hash_knob::selected()` is a
/// cached process setting: it says what THIS PROCESS proves under, never what
/// the bundle in front of it was proven under. The agreement is the
/// verification — a bundle proven under another hash fails here because the
/// transcript's sponge is part of the configuration and every challenge
/// diverges from the first squeeze.
///
/// The split mirrors [`crate::multilinear_continuation::verify_epoch`] and
/// `verify_epoch_bookend::<H>` in the module this drives: one entry point that
/// dispatches on the knob for production, one that takes `H` so the agreement
/// can be argued about — and tested — at all.
///
/// # `decode_commitment` and `prepared` are DIFFERENT OBJECTS, and both stay
///
/// They reach different places and collapsing them would read as a
/// simplification while quietly changing which root the AIR carries.
/// `decode_commitment` is the UNIVARIATE preprocessed root from
/// `commitment_from_elf`, and it feeds `build_epoch_airs`. `prepared` is the
/// MULTILINEAR prepared columns, their derived roots and the stacked
/// commitment, and it feeds `verify_epoch_bookend`.
///
/// `prepared` is `Some` so a walk over every epoch derives DECODE's prepared
/// commitment ONCE per bundle rather than once per epoch — fifteen derivations
/// to one on the block, the same saving `decode_commitment` exists for on the
/// STARK driver. `None` derives it here, exactly as before.
///
/// ⚠ AND THE SIZE OF IT, SO NOBODY REACHES FOR IT AS A LEVER. The derivation
/// costs ≈0.08 s per call on the card — rs4 on FAST at the 8f826601 fixture,
/// where DECODE is 5 x 2^20: `ONE-COMMIT 0.083 s`, one GPU commit, no host
/// fallback (rs2 read 0.068 s earlier; both are single untimed reads). Fifteen
/// harvests therefore spend about 1.2 s deriving, and handing `prepared` in
/// once saves about 1.1 s of it. On a 220-second block that is a tidy-up, not
/// a lever. What makes the parameter worth having is that a walk over every
/// epoch should not repeat a pure function of (ELF, options) fifteen times.
#[allow(clippy::too_many_arguments)]
pub(crate) fn real_epoch_from_whir_continuation_under<H>(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    bundle: &ContinuationProof,
    epoch_index: usize,
    decode_commitment: Option<Commitment>,
    prepared: Option<&crate::multilinear_continuation::DecodePrepared<H>>,
) -> Result<WhirRealEpoch, String>
where
    H: WhirHash,
{
    let elf = Elf::load(elf_bytes).map_err(|e| format!("the inner ELF must load: {e}"))?;
    let decode_commitment = match decode_commitment {
        Some(c) => c,
        None => crate::tables::decode::commitment_from_elf(&elf, opts)
            .map_err(|e| format!("DECODE commitment from ELF: {e}"))?,
    };

    let position = whir_epoch_chain_position(bundle, &elf, epoch_index).ok_or_else(|| {
        format!(
            "epoch {epoch_index} is out of range: the bundle carries {}",
            bundle.epochs.len()
        )
    })?;
    let proof = bundle.epochs[epoch_index].clone();

    // ★ The acceptance check, and the hash agreement with it. Both are the
    // same call: `H` configures the transcript's sponge, so verifying here IS
    // asking whether this bundle was proven under `H`.
    // ★ HANDED IN, OR DERIVED HERE. A caller walking every epoch derives once
    // and hands the same value to all of them; `None` keeps the old behaviour.
    // The binding below outlives the borrow, which is why it is declared first.
    let derived;
    let prepared = match prepared {
        Some(p) => p,
        None => {
            derived = crate::multilinear_continuation::decode_prepared_for::<H>(&elf, elf_bytes)
                .map_err(|e| {
                    format!(
                        "DECODE's prepared commitment under {}: {e:?}",
                        <H as WhirHash>::NAME
                    )
                })?;
            &derived
        }
    };
    let verified = crate::multilinear_continuation::verify_epoch_bookend::<H>(
        &elf,
        elf_bytes,
        &proof,
        &position.register_init,
        position.is_final,
        position.label,
        opts,
        prepared,
    )
    .map_err(|e| format!("epoch {epoch_index} could not be verified: {e:?}"))?;
    if verified.is_none() {
        // ⚠ THE NAME IS `H`'s, NOT THE KNOB'S. This message used to fill that
        // slot from `whir_hash_knob::selected()`, which is wrong in exactly the
        // case worth diagnosing: a keccak process harvesting under RPX would
        // have reported "keccak256" while the verifier ran RPX, pointing the
        // reader away from the defect. The refusal itself is a bare `None` —
        // `verify_epoch_bookend` collapses every failure — so this string is
        // the only reason anyone gets.
        return Err(format!(
            "epoch {epoch_index} of this bundle does not verify under {}. Either the \
             bundle is not the one this ELF and these options describe, or it was \
             proven under a different hash — see this module's header on why that \
             is checked cryptographically and not by a tag",
            <H as WhirHash>::NAME,
        ));
    }

    // ★★ THE CONFIG IS DERIVED THE WAY THE VERIFIER DERIVES IT, which means the
    // WIDTHS ARE THE AIRS' AND ONLY THE HEIGHTS ARE THE PROOF'S.
    //
    // `chain_config` takes the tallest STACK, and a stack's height is a
    // function of both — `one_stack(num_vars, width)`. A first draft of this
    // function passed width 1 because the proof does not state widths, which
    // silently produced a different query count from the one the epoch was
    // argued at. The AIR set is where widths live, and rebuilding it here is
    // the same rebuild `verify_epoch_bookend` performs, with the same
    // arguments, so the two cannot disagree.
    let airs = crate::continuation::build_epoch_airs(
        &elf,
        opts,
        &[],
        &proof.table_counts,
        &position.register_init,
        &proof.reg_fini,
        position.is_final,
        Some(decode_commitment),
    );
    let l2g_air = crate::continuation::l2g_memory_air(opts, position.label);
    let mut air_refs = airs.air_refs();
    air_refs.push(&l2g_air);
    if air_refs.len() != proof.table_num_vars.len() {
        return Err(format!(
            "epoch {epoch_index}'s layout has {} tables and the proof states {} heights",
            air_refs.len(),
            proof.table_num_vars.len(),
        ));
    }
    let shapes: Vec<(usize, usize)> = air_refs
        .iter()
        .zip(&proof.table_num_vars)
        .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
        .collect();
    let config = crate::multilinear_prove::chain_config(&shapes);

    Ok(WhirRealEpoch {
        elf_digest: crate::statement::elf_digest(elf_bytes),
        position,
        config,
        decode_commitment,
        pc_start: elf.entry_point,
        shapes,
        proof,
    })
}

/// REPORTS, and decides nothing: this process's WHIR hash is not the one level
/// 0's wraps are pinned to.
///
/// `None` when the process is set to RPX. Otherwise a line naming the setting.
///
/// ⚠ WHY THIS REFUSES NOTHING. The compile-time pin `hash_pin::BlockStarkHash`
/// is RPX, so a production tree built in a keccak process is almost certainly
/// an operator's mistake — but "almost certainly" is not a soundness property.
/// A keccak bundle harvested under keccak is a correctly verified epoch; it is
/// simply not the production posture, and a driver that refused it would break
/// every keccak test and every keccak A/B arm the campaign runs. The refusal
/// this driver DOES make is the cryptographic one in
/// [`real_epoch_from_whir_continuation_under`], which needs no knob at all.
///
/// So this exists to make a cheap mistake cheap to find: an operator who meant
/// to run the RPX arm learns it before a tree is built, not from a number that
/// looks like the RPX arm and is not — the same failure `whir_hash_knob`'s own
/// header refuses to allow for an unrecognised value.
pub(crate) fn whir_process_posture_note() -> Option<String> {
    let setting = crate::whir_hash_knob::selected();
    if setting == crate::whir_hash_knob::Setting::Rpx {
        return None;
    }
    Some(format!(
        "⚠ {}={} — level 0's wrap proofs commit under the RPX block hasher, so a \
         production tree wants the RPX arm. Nothing is refused on this: a bundle \
         proven under {} harvests fine under {}. Set {}=rpx if this run was meant \
         to be the production posture.",
        crate::whir_hash_knob::ENV,
        setting.name(),
        setting.name(),
        setting.name(),
        crate::whir_hash_knob::ENV,
    ))
}

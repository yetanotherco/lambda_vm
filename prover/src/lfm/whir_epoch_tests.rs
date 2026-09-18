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
pub(super) struct WhirChainPosition {
    pub(super) register_init: Vec<u32>,
    pub(super) is_final: bool,
    pub(super) label: u64,
}

/// One epoch's level-0 wrap input: the proof, everything its statement absorbs,
/// and the values the verifier derived rather than read.
///
/// The statement half is exactly
/// [`crate::multilinear_continuation::absorb_epoch`]'s argument list, because
/// the in-guest verifier's first job is to replay that absorb and any field it
/// cannot see is a challenge it cannot reproduce.
pub(super) struct WhirRealEpoch {
    /// The epoch proof itself, cloned out of the bundle.
    pub(super) proof: EpochProof,
    /// `statement::elf_digest(elf_bytes)` — the program this run was of.
    pub(super) elf_digest: [u8; 32],
    /// The epoch's position in its run, derived.
    pub(super) position: WhirChainPosition,
    /// The parameters the epoch's argument ran at, rebuilt from the shapes the
    /// proof states rather than carried: `chain_config` over
    /// `(width, num_vars)` per table.
    pub(super) config: ChainConfig,
    /// DECODE's univariate preprocessed commitment for this (ELF, options)
    /// pair, taken once per bundle rather than once per epoch.
    pub(super) decode_commitment: Commitment,
    /// The inner ELF's entry point — `program_id`'s `pc_start`.
    pub(super) pc_start: u64,
    /// `(width, num_vars)` per table, in sub-proof order: the widths are the
    /// AIRs' and the heights are the proof's. Kept because the config is a
    /// function of them and a caller that wants to check one needs the other.
    pub(super) shapes: Vec<(usize, usize)>,
}

impl WhirRealEpoch {
    /// The epoch's own published bytes, which the run's output concatenates.
    pub(super) fn public_output(&self) -> &[u8] {
        &self.proof.public_output
    }

    /// How many tables this epoch's argument covers — the width of everything
    /// the guest walks.
    pub(super) fn num_tables(&self) -> usize {
        self.proof.table_num_vars.len()
    }
}

/// The verifier's own chain position for epoch `index`.
///
/// `None` when the index is out of range. Epoch 0 starts from the ELF's entry
/// point; every later epoch starts from the PREVIOUS epoch's proved `reg_fini`,
/// which is what ties one epoch to the next and carries the commit index.
pub(super) fn whir_epoch_chain_position(
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
pub(super) fn real_epoch_from_whir_continuation(
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
#[allow(clippy::too_many_arguments)]
pub(super) fn real_epoch_from_whir_continuation_under<H>(
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
pub(super) fn whir_process_posture_note() -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multilinear_continuation;
    use crate::test_utils::asm_elf_bytes;
    use stark::proof::options::ProofOptions;

    /// A run whose epochs touch memory across the boundary, and which publishes
    /// output in its last epoch.
    fn a_run() -> (Vec<u8>, Vec<u8>) {
        let mut input: Vec<u8> = Vec::with_capacity(16);
        input.extend_from_slice(&16u32.to_le_bytes());
        input.extend_from_slice(&[0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
        input.extend_from_slice(&[0u8; 4]);
        (asm_elf_bytes("test_private_input_xpage"), input)
    }

    fn bundle() -> (
        Vec<u8>,
        ProofOptions,
        multilinear_continuation::ContinuationProof,
    ) {
        let (elf_bytes, input) = a_run();
        let opts = ProofOptions::default_test_options();
        let bundle = multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts)
            .expect("prove the continuation");
        (elf_bytes, opts, bundle)
    }

    /// ★ Every epoch of a real bundle harvests, including the LAST one — which
    /// the session path cannot build and which is the only epoch that publishes.
    #[test]
    fn every_epoch_of_a_bundle_harvests() {
        let (elf_bytes, opts, b) = bundle();
        assert!(b.epochs.len() >= 2, "a one-epoch run chains nothing");

        let mut published = 0usize;
        for index in 0..b.epochs.len() {
            let e = real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, index, None)
                .unwrap_or_else(|e| panic!("epoch {index}: {e}"));
            assert_eq!(e.position.is_final, index + 1 == b.epochs.len());
            assert_eq!(e.position.label, epoch_label(index as u64));
            assert!(e.num_tables() > 0);
            assert_eq!(e.pc_start, Elf::load(&elf_bytes).expect("load").entry_point);
            assert_eq!(e.elf_digest, crate::statement::elf_digest(&elf_bytes));
            // `chain_config`'s constants, which `agrees_with` on the other
            // lineage also rests on. The query count is not asserted here: it
            // is discharged by the epoch VERIFYING under this config above.
            assert_eq!(e.config.log_blowup, 2);
            assert_eq!(e.config.log_folding, 4);
            assert!(e.config.num_queries > 0);
            published += usize::from(!e.public_output().is_empty());
        }
        assert!(
            published > 0,
            "no epoch published output, so the harvest never exercised the one \
             shape that differs"
        );
    }

    /// ★★ THE CHAIN IS THE DRIVER'S, NOT THE BUNDLE'S.
    ///
    /// Epoch 0 starts from the ELF's entry point and every later epoch from the
    /// PREVIOUS epoch's proved `reg_fini`. A driver that took the starting
    /// registers from the epoch being harvested would accept a bundle whose
    /// epochs are about unrelated register files, and nothing else here would
    /// notice.
    #[test]
    fn the_starting_registers_come_from_the_previous_epoch() {
        let (elf_bytes, _, b) = bundle();
        let elf = Elf::load(&elf_bytes).expect("load");
        assert!(b.epochs.len() >= 2);

        let first = whir_epoch_chain_position(&b, &elf, 0).expect("epoch 0");
        assert_eq!(
            first.register_init,
            register::register_init_from_entry_point(elf.entry_point),
            "epoch 0 must start from the ELF"
        );
        for index in 1..b.epochs.len() {
            let p = whir_epoch_chain_position(&b, &elf, index).expect("in range");
            assert_eq!(
                p.register_init,
                b.epochs[index - 1].reg_fini,
                "epoch {index} must start where epoch {} was proved to end",
                index - 1
            );
        }
        assert!(whir_epoch_chain_position(&b, &elf, b.epochs.len()).is_none());
    }

    /// ★★ A PROOF THAT DOES NOT VERIFY IS NOT HARVESTED.
    ///
    /// The tamper is the epoch's PUBLIC OUTPUT, which `absorb_epoch` binds into
    /// the statement before any challenge — so restating it moves every
    /// challenge and the proof cannot verify. Without the acceptance check the
    /// driver would hand the guest a wrap input built from it, and the failure
    /// would surface a whole wrap prove later, reading as an emitter bug.
    ///
    /// ⚠ THE OBVIOUS TAMPER — THE REGISTER CARRY — WAS NOT USABLE HERE WHEN
    /// THIS TEST WAS WRITTEN, AND THE REASON WAS A DEFECT.
    /// `VmAirs::new` gave REGISTER a preprocessed COMMITMENT and no columns
    /// closure, so the multilinear verifier's `check_preprocessed` walked an
    /// empty list in zero iterations and a restated `reg_fini` verified. That
    /// is fixed on this base: 5e3df0c0 binds INIT and FINI as columns, and the
    /// carry is refused by `verify_epoch` itself under
    /// `a_restated_register_carry_is_refused_by_the_epoch_it_lands_in` and
    /// `a_restated_register_fini_is_refused_by_the_epoch_that_states_it`
    /// (`multilinear_continuation_tests`), each beside its accept control.
    ///
    /// The PUBLIC OUTPUT remains this test's tamper by choice rather than by
    /// necessity: it is the field `absorb_epoch` binds before any challenge, so
    /// it exercises the driver's acceptance check through the statement the
    /// guest's replay will read, which is what this driver is for.
    #[test]
    fn an_epoch_that_does_not_verify_is_refused() {
        let (elf_bytes, opts, mut b) = bundle();
        assert!(b.epochs.len() >= 2);

        // The control first: the untouched bundle harvests. A refusal test
        // without it passes on a driver that refuses everything.
        real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 1, None)
            .expect("the untouched bundle must harvest");

        b.epochs[1].public_output.push(0xFF);
        let refused = real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 1, None);
        assert!(
            refused.is_err(),
            "a restated public output was harvested into a wrap input"
        );
    }

    /// An index past the end is an error, not a panic.
    #[test]
    fn an_index_past_the_end_is_refused() {
        let (elf_bytes, opts, b) = bundle();
        assert!(
            real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, b.epochs.len(), None).is_err()
        );
    }

    /// ★★ THE WIDTHS ARE THE AIRS', AND THAT DISTINCTION IS NOT COSMETIC.
    ///
    /// `chain_config` takes the tallest STACK, and a stack's height is
    /// `one_stack(num_vars, width)` — so a config built from the heights alone,
    /// with widths stubbed at 1, is a DIFFERENT config with a different query
    /// count. The first draft of the driver did exactly that, because the proof
    /// states heights and not widths.
    ///
    /// This is what makes the distinction observable: the stubbed form must
    /// disagree with the harvested one. If it ever agrees, the assertion is
    /// vacuous at this shape and says so rather than passing quietly.
    #[test]
    fn a_config_built_without_the_widths_is_a_different_config() {
        let (elf_bytes, opts, b) = bundle();
        let harvested = real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 0, None)
            .expect("epoch 0 harvests");

        // ⚠ The assertion is on the TALLEST STACK and not on `num_queries`.
        // `num_queries` is a step function of the round count, so two different
        // stack heights can land on the same query count — measured: at this
        // fixture's shapes both forms give 112, and an inequality on the query
        // count would have been vacuous here while reading as a real check.
        // The stack height is where the widths actually enter.
        let tallest = |shapes: &[(usize, usize)]| {
            shapes
                .iter()
                .map(|&(width, num_vars)| {
                    multilinear::constraint_argument::one_stack(num_vars, width)
                })
                .max()
                .unwrap_or(1)
        };
        let stubbed: Vec<(usize, usize)> = harvested
            .proof
            .table_num_vars
            .iter()
            .map(|&n| (1usize, n as usize))
            .collect();
        let real: Vec<(usize, usize)> = harvested.shapes.to_vec();

        assert_ne!(
            tallest(&real),
            tallest(&stubbed),
            "the widths make no difference to the tallest stack at this shape, so \
             this test cannot see the defect it exists for"
        );
        assert_eq!(
            harvested.config,
            crate::multilinear_prove::chain_config(&real),
            "the harvested config is not the one its own shapes imply"
        );
    }

    /// ★ THE SUPPLIED COMMITMENT IS THE ONE CARRIED, not a rebuilt equal.
    ///
    /// `Some(c)` exists so a walk over every epoch takes DECODE's commitment
    /// once per bundle rather than once per epoch — lane P measured the STARK
    /// driver rebuilding it 38 times over 19 epochs. A driver that accepted the
    /// argument and rebuilt anyway would pass a value comparison against the
    /// real root, so the value handed in here is deliberately NOT the real one.
    /// It reaches only the AIR build's DECODE root, which the multilinear path
    /// does not compare, so the epoch still verifies.
    #[test]
    fn the_supplied_decode_commitment_is_the_one_carried() {
        let (elf_bytes, opts, b) = bundle();
        let supplied: Commitment = [0xABu8; 32];
        let e = real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 0, Some(supplied))
            .expect("epoch 0 harvests");
        assert_eq!(
            e.decode_commitment, supplied,
            "the driver ignored the commitment it was handed and built its own"
        );
    }

    /// A bundle whose epochs are proven under a hash NAMED HERE, not inherited
    /// from the process.
    ///
    /// ★★ AND THAT DISTINCTION IS THE WHOLE POINT OF THE HELPER. The cheap
    /// version of the refusal test proves with `prove_continuation` and harvests
    /// at `RpxWhir`; `prove_continuation` dispatches on the cached knob, which is
    /// keccak only because `LAMBDA_VM_WHIR_HASH` is usually unset. Run the suite
    /// with `LAMBDA_VM_WHIR_HASH=rpx` and that bundle is RPX, the two arms SILENTLY
    /// INVERT, and the failure reads as a broken hash agreement when it is a
    /// configuration mismatch. Re-proving the epochs under a literal `KeccakWhir`
    /// costs one more prove of the same run and makes the test mean the same thing
    /// in every process.
    ///
    /// The shell is a real bundle — a real cross-epoch proof, a real touched page
    /// set — because only the epochs are replaced. The driver reads `bundle.epochs`
    /// and nothing else, so the shell's own hash cannot reach this measurement.
    fn a_keccak_bundle() -> (
        Vec<u8>,
        ProofOptions,
        multilinear_continuation::ContinuationProof,
    ) {
        use multilinear::whir_hash::KeccakWhir;

        let (elf_bytes, input) = a_run();
        let opts = ProofOptions::default_test_options();
        let mut b = multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts)
            .expect("prove the continuation");

        let elf = Elf::load(&elf_bytes).expect("load");
        let artifacts = crate::tables::trace_builder::DecodeArtifacts::from_elf(&elf)
            .expect("decode artifacts");
        let prepared =
            multilinear_continuation::decode_prepared_for::<KeccakWhir>(&elf, &elf_bytes)
                .expect("DECODE's prepared commitment under keccak");

        let mut epochs = Vec::new();
        crate::continuation::for_each_epoch(&elf, &input, 2, &artifacts, |prepared_epoch, _| {
            let crate::continuation::PreparedEpoch {
                register_init,
                label,
                traces,
                boundary,
                is_final,
                ..
            } = prepared_epoch;
            epochs.push(multilinear_continuation::prove_epoch::<KeccakWhir>(
                &elf,
                &elf_bytes,
                &register_init,
                label,
                traces,
                is_final,
                &boundary,
                &opts,
                None,
                &prepared,
            )?);
            Ok(())
        })
        .expect("prove every epoch under a literal keccak");

        assert_eq!(
            epochs.len(),
            b.epochs.len(),
            "the explicit pass split the run differently from prove_continuation's"
        );
        assert!(epochs.len() >= 2, "a one-epoch run chains nothing");
        b.epochs = epochs;
        (elf_bytes, opts, b)
    }

    /// ★★ SEAM 1: THE HASH AGREEMENT, AT THE DRIVER'S OWN ENTRY POINT.
    ///
    /// The bundle carries no hash tag and cannot — a WHIR proof's bytes are
    /// hash-agnostic by design, which is the byte gate's own invariant — and the
    /// process knob describes the process, not the bundle. So the refusal is the
    /// verification: hand the driver an `H`, and a bundle proven under another
    /// hash fails because the transcript's sponge is part of the configuration.
    ///
    /// ⚠ BOTH ARMS, ON THE SAME BUNDLE. `verify_epoch_bookend` collapses every
    /// failure to `Ok(None)`, so the refusal carries no reason of its own and a
    /// driver that refused everything would pass the refusing half alone. The
    /// accept arm is what makes the refusal mean something.
    ///
    /// # What was mutated, and the ratio stated plainly
    ///
    /// ONE PROGRAM MUTATION: the verdict computed and discarded in
    /// [`real_epoch_from_whir_continuation_under`] — the mutated driver builds a
    /// `WhirRealEpoch` from a proof that failed verification, which the original
    /// never does. It turns THIS test red on the `HARVESTED under RPX` arm, and
    /// `an_epoch_that_does_not_verify_is_refused` and
    /// `a_restated_register_carry_is_refused_by_the_driver` red on theirs, while
    /// the six non-refusal tests stay green.
    ///
    /// ONE TEST-SIDE CONTROL, which is NOT a program mutation and is not counted
    /// as one: running the refusal arm at `KeccakWhir` too. That changes the
    /// TEST, not the driver — it is the accept arm read the other way — so what
    /// it demonstrates is this test's discriminating power, that the arm is
    /// measuring `H` and not something incidental about the bundle. It fails on
    /// the `Ok(_)` arm, as it must.
    #[test]
    fn an_epoch_proven_under_keccak_is_refused_when_harvested_under_rpx() {
        use multilinear::whir_hash::{KeccakWhir, RpxWhir};

        let (elf_bytes, opts, b) = a_keccak_bundle();

        // THE CONTROL FIRST, and on epoch 1 so the register carry is exercised
        // with it: the hash it was proven under must ACCEPT.
        let accepted = real_epoch_from_whir_continuation_under::<KeccakWhir>(
            &opts, &elf_bytes, &b, 1, None, None,
        )
        .expect("a keccak bundle must harvest under keccak");
        assert_eq!(
            accepted.position.label,
            epoch_label(1),
            "the control harvested some other epoch"
        );

        // A `match` rather than `expect_err`, which would want `WhirRealEpoch:
        // Debug` — a derive on a production type added for a test's
        // convenience, on a struct holding a whole proof.
        let refused = match real_epoch_from_whir_continuation_under::<RpxWhir>(
            &opts, &elf_bytes, &b, 1, None, None,
        ) {
            Err(reason) => reason,
            Ok(_) => panic!(
                "an epoch proven under keccak was HARVESTED under RPX; level 0's \
                 whole hash agreement rests on that being impossible"
            ),
        };

        // The reason is NAMED, and it names the hash the harvest RAN UNDER
        // rather than the process knob — which in this very test is whatever
        // the suite was started with, and is not RPX.
        assert!(
            refused.contains(<RpxWhir as WhirHash>::NAME),
            "the refusal must name the hash it verified under; it said: {refused}"
        );
        assert!(
            !refused.contains(<KeccakWhir as WhirHash>::NAME),
            "the refusal named the bundle's hash instead of the verifier's: {refused}"
        );
    }

    /// ★ THE POSTURE IS REPORTED AND DECIDES NOTHING.
    ///
    /// ⚠ THE ASSERTION IS A RELATION, NOT A VALUE. `whir_hash_knob::selected()`
    /// is a cached process-global, so `assert!(note.is_some())` would be an
    /// assertion about how the whole test binary was invoked — green under the
    /// default and red under `LAMBDA_VM_WHIR_HASH=rpx`, for no reason anybody
    /// reading the test would guess. What is true in every process is that the
    /// note fires exactly when the setting is not RPX.
    ///
    /// ★ AND IT FAILS IN BOTH DIRECTIONS, which a relation alone would not. The
    /// setting is read into a local ONCE and the note's text must contain THAT
    /// name, so a note naming a setting other than the one in force is red — not
    /// merely a note that fires at the wrong time. Measured: making the note
    /// report the opposite setting turns this test red on the naming assert
    /// while the `is_some()` relation still holds, so the two asserts catch
    /// different defects.
    #[test]
    fn the_process_hash_posture_is_reported_and_never_decides() {
        let setting = crate::whir_hash_knob::selected();
        let note = whir_process_posture_note();

        assert_eq!(
            note.is_some(),
            setting != crate::whir_hash_knob::Setting::Rpx,
            "the posture note disagreed with the process setting {setting:?}"
        );
        if let Some(text) = &note {
            assert!(
                text.contains(setting.name()),
                "the note must name the setting it reports: {text}"
            );
            assert!(
                text.contains(crate::whir_hash_knob::ENV),
                "the note must name the knob an operator would change: {text}"
            );
        }

        // AND IT DECIDES NOTHING. A bundle proven under this process's own hash
        // harvests whether the note fired or not — which is the half that would
        // break if anyone ever turned this report into a refusal.
        let (elf_bytes, opts, b) = bundle();
        real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 0, None)
            .expect("the posture note must not refuse anything");
    }

    /// ★ THE RESTATED REGISTER CARRY IS REFUSED — at the driver, on the base
    /// that fixed it.
    ///
    /// This file used to carry an `#[ignore]`d red flag recording the opposite:
    /// flipping one bit of `epochs[0].reg_fini` — the vector that IS epoch 1's
    /// `register_init` — and harvesting epoch 1 was ACCEPTED, because
    /// `VmAirs::new` gave REGISTER a preprocessed commitment and no columns
    /// closure and the multilinear verifier checks columns. 5e3df0c0 binds both
    /// ends as columns. The red flag is deleted; this is what replaces it, at
    /// the entry point level 0 actually calls.
    ///
    /// Both indices W1d established are flipped: index 1, and `X254_INDEX`, the
    /// synthetic commit index that rides in the same vector. The untouched
    /// bundle is harvested first, or a driver that refused everything would pass.
    #[test]
    fn a_restated_register_carry_is_refused_by_the_driver() {
        let (elf_bytes, opts, mut b) = bundle();
        assert!(b.epochs.len() >= 2, "a one-epoch run chains nothing");

        real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 1, None)
            .expect("the control: the untouched bundle must harvest epoch 1");

        let mut carry = b.epochs.clone();
        carry[0].reg_fini[1] ^= 1;
        let restated = std::mem::replace(&mut b.epochs, carry);
        assert!(
            real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 1, None).is_err(),
            "epoch 1 was harvested against a register file the chain never handed it"
        );

        let mut commit_index = restated;
        commit_index[0].reg_fini[crate::tables::register::X254_INDEX] ^= 1;
        b.epochs = commit_index;
        assert!(
            real_epoch_from_whir_continuation(&opts, &elf_bytes, &b, 1, None).is_err(),
            "epoch 1 was harvested against a restated commit index"
        );
    }
    /// ★ THE HANDED-IN PREPARED COMMITMENT IS THE ONE USED, not a rebuilt equal.
    ///
    /// `prepared` exists so a walk over every epoch derives DECODE's prepared
    /// commitment ONCE per bundle instead of once per epoch — fifteen
    /// derivations to one on the block. A driver that took the argument and
    /// derived its own anyway would be INDISTINGUISHABLE on the accept path, so
    /// the check is a prepared built from a DIFFERENT program: used, it must
    /// refuse; ignored, the epoch verifies exactly as it does today and this
    /// test fails. That asymmetry is the whole design — the refusal arm is what
    /// proves USE, and the accept arm is what stops a driver that refuses
    /// everything from passing it.
    ///
    /// ⚠ AND THE OTHER HALF IS UNREACHABLE RATHER THAN UNTESTED. A prepared
    /// built for the wrong HASH cannot be handed to this function at all:
    /// `DecodePrepared<H>` carries the hash in its type and the driver is
    /// `::<H>`, so the mismatch is a compile error. That is strictly better than
    /// a runtime refusal, and it is why no test for it exists — a test for a
    /// state the type system forbids is a check that cannot fail.
    ///
    /// ⚠ The refusal is NOT a shape guard. `DecodePrepared::agrees_with`
    /// compares only `log_blowup` and `log_folding`, which two programs at the
    /// same options share, so a wrong-program prepared sails past it and is
    /// caught by the derived roots block the transcript absorbs — the same
    /// cryptographic mechanism as the hash agreement. The reason is printed so
    /// a reader can see which path actually fired.
    #[test]
    fn the_prepared_commitment_handed_in_is_the_one_used() {
        use multilinear::whir_hash::KeccakWhir;

        let (elf_bytes, opts, b) = a_keccak_bundle();
        let elf = Elf::load(&elf_bytes).expect("load");

        // THE ACCEPT CONTROL: the bundle's own prepared commitment, handed in
        // rather than derived, must harvest.
        let own = multilinear_continuation::decode_prepared_for::<KeccakWhir>(&elf, &elf_bytes)
            .expect("the bundle's own prepared commitment");
        real_epoch_from_whir_continuation_under::<KeccakWhir>(
            &opts,
            &elf_bytes,
            &b,
            1,
            None,
            Some(&own),
        )
        .expect("the bundle's own prepared commitment must harvest");

        // A prepared built from a DIFFERENT program, at the same options.
        let other_bytes = asm_elf_bytes("sub");
        let other = Elf::load(&other_bytes).expect("load sub");
        let wrong =
            multilinear_continuation::decode_prepared_for::<KeccakWhir>(&other, &other_bytes)
                .expect("sub's prepared commitment");
        let refused = match real_epoch_from_whir_continuation_under::<KeccakWhir>(
            &opts,
            &elf_bytes,
            &b,
            1,
            None,
            Some(&wrong),
        ) {
            Err(reason) => reason,
            Ok(_) => panic!(
                "the driver harvested an epoch against a DECODE commitment prepared \
                 from a DIFFERENT program, so the `prepared` argument is accepted \
                 and ignored — a parameter that changes nothing is a display"
            ),
        };
        println!("PREPARED-REFUSAL  {refused}");
    }
}

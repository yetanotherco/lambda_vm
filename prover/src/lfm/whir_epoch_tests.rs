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
//! ⚠ AND THE REFUSAL IS NOT YET TESTABLE ON THIS BASE, which is stated rather
//! than papered over. On `whir/lfm` both `prove_epoch` and `verify_epoch`
//! dispatch on the cached process knob internally, so one process is one hash
//! and a test cannot hold a keccak bundle and an RPX verification at the same
//! time. The `whir/decode-group` lineage makes both generic over `H`, which is
//! exactly what the test needs: prove at `KeccakWhir`, harvest at `RpxWhir`,
//! and require the refusal — with the same bundle harvested at `KeccakWhir` as
//! the control that must be ACCEPTED, since a refusal test with only its
//! refusing half passes on a driver that refuses everything. The test is
//! written the moment that merge lands; see `the_hash_agreement_is_owed`.

use crate::multilinear_continuation::{ContinuationProof, EpochProof};
use crate::tables::local_to_global::epoch_label;
use crate::tables::register;
use executor::elf::Elf;
use multilinear::whir_chain::ChainConfig;
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

    // ★ The acceptance check, and the hash agreement with it.
    let verified = crate::multilinear_continuation::verify_epoch(
        &elf,
        elf_bytes,
        &proof,
        &position.register_init,
        position.is_final,
        position.label,
        opts,
    )
    .map_err(|e| format!("epoch {epoch_index} could not be verified: {e:?}"))?;
    if !verified {
        return Err(format!(
            "epoch {epoch_index} of this bundle does not verify under {}. Either the \
             bundle is not the one this ELF and these options describe, or it was \
             proven under a different hash — see this module's header on why that \
             is checked cryptographically and not by a tag",
            crate::whir_hash_knob::selected().name(),
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
    /// ⚠ The obvious tamper — the register carry — does NOT work here, and the
    /// reason is a finding rather than a quirk of this test. See
    /// `the_register_carry_is_not_bound_on_the_multilinear_path`.
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

    /// ⛔⛔ A FINDING, NOT A TEST OF THIS DRIVER: on the multilinear
    /// continuation path an epoch's INIT register file appears not to be bound,
    /// so a restated cross-epoch carry verifies.
    ///
    /// # What was measured
    ///
    /// Flipping one bit of `epochs[0].reg_fini` and harvesting epoch 1 — whose
    /// `register_init` is exactly that vector — was ACCEPTED: `verify_epoch`
    /// returned true for an epoch checked against a register file the chain
    /// never handed it. That was this test's first form, and it failed by not
    /// failing.
    ///
    /// # What reading says the mechanism is (✓ VERIFIED, not inferred)
    ///
    /// `continuation::build_epoch_airs` always passes
    /// `register_preprocessed = Some((commitment, NUM_PREPROCESSED_COLS_WITH_FINI))`,
    /// and in `VmAirs::new` that argument selects
    /// `.with_preprocessed(commitment, n)` — a COMMITMENT and NO columns
    /// closure. The other branch, taken only when the argument is `None`, uses
    /// `.with_preprocessed_columns(commitment, n, || register::preprocessed_columns(..))`
    /// and supplies both. The multilinear verifier checks preprocessed COLUMNS
    /// (`air.precomputed_columns()` feeding `check_preprocessed`), not the
    /// univariate root — so for a continuation epoch REGISTER hands it nothing
    /// to compare, and `register_init` and `reg_fini` are unconstrained on this
    /// path.
    ///
    /// # Why it is `#[ignore]`d and panics rather than asserting the behaviour
    ///
    /// Writing `assert!(verify_epoch(wrong_init))` and calling it green would
    /// record a suspected defect as intended behaviour. This is a visible
    /// obligation instead: it appears in a test listing, it names the mechanism,
    /// and it cannot be mistaken for a passing check. It is NOT within this
    /// lane's scope to fix — `continuation.rs` is a STARK-pipeline file this
    /// brief forbids touching, and the remedy (give the continuation REGISTER
    /// AIR its columns closure as well as its root, or make the multilinear
    /// path check the root) is a soundness change that needs its own review.
    ///
    /// ⚠ And note what this does NOT say: `a_broken_register_carry_is_rejected`
    /// in `multilinear_continuation_tests` passes, and it flips the same bit.
    /// It calls `verify_epochs`, which checks EVERY epoch, so its rejection may
    /// come from epoch 0 rather than from the carry into epoch 1 — which is a
    /// second thing to establish before anyone concludes how wide this is.
    #[test]
    #[ignore = "a reported finding, not a check: the multilinear path appears not to bind register_init"]
    fn the_register_carry_is_not_bound_on_the_multilinear_path() {
        panic!(
            "measured: flipping one bit of epochs[0].reg_fini and verifying epoch 1 \
             — whose register_init IS that vector — was ACCEPTED. Read: \
             `build_epoch_airs` gives REGISTER a preprocessed COMMITMENT and no \
             columns closure, and the multilinear verifier checks columns. Needs \
             its own review; do not close this by weakening a test"
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

    /// ⛔ THE HASH AGREEMENT IS OWED, AND THIS IS WHERE IT GOES.
    ///
    /// The test is: prove a bundle at `KeccakWhir`, harvest it at `RpxWhir`,
    /// require the refusal, and harvest the SAME bundle at `KeccakWhir` as the
    /// control that must be accepted. It cannot be written on this base,
    /// because `prove_epoch` and `verify_epoch` both dispatch on the cached
    /// process knob, so one process is one hash. The `whir/decode-group`
    /// lineage makes both generic over `H`, which is exactly what this needs.
    ///
    /// It is a failing-by-construction reminder rather than a comment: an
    /// `#[ignore]`d test with a name a grep finds, so the obligation is visible
    /// in a test listing and not only in prose. It is NOT a passing test, which
    /// would report the work as done.
    #[test]
    #[ignore = "blocked: needs the H-generic prove/verify entry points from whir/decode-group"]
    fn the_hash_agreement_is_owed() {
        panic!(
            "the cryptographic hash-agreement test is owed and cannot be written \
             on this base: `prove_epoch` and `verify_epoch` dispatch on the cached \
             process knob, so a keccak bundle and an RPX harvest cannot coexist in \
             one process. Merge the H-generic entry points from `whir/decode-group` \
             and write it here"
        );
    }
}

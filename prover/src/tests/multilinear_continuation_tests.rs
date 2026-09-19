//! Continuations through the multilinear path: the epochs, the cross-epoch
//! proof, and what ties them.
//!
//! The epoch split, the local-to-global bookend and the cross-epoch register
//! carry are [`crate::continuation`]'s and do not depend on the commitment
//! scheme; what these check is that an epoch's AIRs — which differ from the
//! monolithic ones, PAGE replaced by the bookend and REGISTER preprocessing
//! both ends — argue correctly under WHIR, and that everything the chain rests
//! on is rejected when it is restated: the register carry, the bookend root,
//! and the touched page set.
//!
//! # ⛔ WHICH OF THESE CAN SEE A BROKEN VERIFIER, AND WHICH CANNOT
//!
//! Measured, not assumed: breaking the cross-epoch AIR set reddens exactly the
//! tests that assert a bundle VERIFIES. Everything asserting a REFUSAL stays
//! green through any breakage, because a verifier that rejects everything
//! rejects the tampered bundle too — so a refusal arm alone is a check that
//! cannot fail the other way.
//!
//! ⇒ **Every refusal test here first asserts its UNTAMPERED bundle verifies
//! through the same call.** That honest-path control is what makes a refusal
//! arm able to see a broken verifier, and it is not optional: mutating
//! `global_airs_for` to build the set one page short reddened only
//! `a_continuation_proves_and_verifies` before the controls were added.
//!
//! ⚠ AND ONE REFUSAL HERE IS SATISFIED BY THE WRONG HALF.
//! `a_bookend_that_is_not_the_one_chained_is_rejected` moves a byte of a root
//! that lives inside the EPOCH's own proof, so the epoch half refuses and
//! `verify_continuation` returns before the cross-epoch half runs at all — its
//! doc's claim that "both halves still verify on their own" does not hold for
//! the tamper it performs. The binding it is named for,
//! `proved == chained`, is reached instead by
//! [`a_swapped_bookend_root_is_caught_by_the_binding`], which swaps two epochs'
//! roots inside the CROSS-EPOCH proof and leaves both halves valid.

use executor::elf::Elf;
use stark::proof::options::ProofOptions;

use crate::continuation::{self, PreparedEpoch};
use crate::multilinear_continuation;
use crate::tables::local_to_global;
use crate::tables::register;
use crate::tables::trace_builder::DecodeArtifacts;
use crate::test_utils::asm_elf_bytes;

/// Proves and verifies every epoch of `name` split at `epoch_size_log2`,
/// chaining the register file the way a driver would, and returns the epoch
/// count.
fn epochs_prove_and_verify(name: &str, epoch_size_log2: u32) -> usize {
    let elf_bytes = asm_elf_bytes(name);
    let elf = Elf::load(&elf_bytes).expect("load");
    let opts = ProofOptions::default_test_options();
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");

    // The verifier's own carry: the ELF's registers for epoch 0, the previous
    // epoch's proved `reg_fini` after that. Never the prover's.
    let mut carried = register::register_init_from_entry_point(elf.entry_point);
    let mut count = 0usize;

    let boundaries = crate::with_whir_hash!(|H| {
        let pinned = multilinear_continuation::decode_prepared_for::<H>(&elf, &elf_bytes)
            .expect("DECODE's out-of-band commitment");
        continuation::for_each_epoch(&elf, &[], epoch_size_log2, &artifacts, |prepared, _| {
            let PreparedEpoch {
                register_init,
                label,
                traces,
                boundary,
                is_final,
                ..
            } = prepared;
            assert_eq!(
                register_init, carried,
                "epoch {label} starts from registers the chain did not hand it"
            );

            let proof = multilinear_continuation::prove_epoch::<H>(
                &elf,
                &elf_bytes,
                &register_init,
                label,
                traces,
                is_final,
                &boundary,
                &opts,
                None,
                &pinned,
            )?;
            assert!(
                multilinear_continuation::verify_epoch(
                    &elf, &elf_bytes, &proof, &carried, is_final, label, &opts
                )?,
                "epoch {label} does not verify"
            );
            carried = proof.reg_fini;
            count += 1;
            Ok(())
        })
    })
    .expect("the epochs prepare");

    assert!(count > 0, "the program ran no epochs");
    // What the cross-epoch proof will be made of: one boundary per epoch.
    assert_eq!(boundaries.len(), count, "one boundary per epoch");
    count
}

#[test]
fn every_epoch_of_a_program_proves_and_verifies() {
    assert!(epochs_prove_and_verify("sub", 4) >= 1);
}

/// A smaller epoch means more of them, which is what exercises the carry: each
/// one starts where the last proof said it ended.
///
/// ⚠ THE SIZES ARE THE CHECK. At 2^6 and 2^4 `sub` is ONE epoch either way, so
/// `many >= few` read `1 >= 1` — a comparison whose two sides the fixture could
/// not separate. `sub` splits at 2^2, so the inequality is now strict and the
/// count that must exceed the other is the one the carry runs through.
#[test]
fn the_epochs_chain_through_their_registers() {
    let few = epochs_prove_and_verify("sub", 4);
    let many = epochs_prove_and_verify("sub", 2);
    assert!(
        many > few,
        "a smaller epoch must produce more of them, or nothing here chains: \
         {many} at 2^2 against {few} at 2^4"
    );
}

/// The driver: every epoch proved in order, and checked from the ELF alone with
/// the verifier deriving each epoch's starting registers itself.
#[test]
fn a_run_of_epochs_proves_and_verifies_from_the_elf() {
    let elf_bytes = asm_elf_bytes("sub");
    let opts = ProofOptions::default_test_options();
    let epochs = multilinear_continuation::prove_epochs(&elf_bytes, &[], 4, &opts).expect("prove");
    assert!(!epochs.is_empty());
    assert!(
        multilinear_continuation::verify_epochs(&elf_bytes, &epochs, &opts).expect("verify"),
        "the run does not verify"
    );
}

/// The registers are the chain: handing an epoch the wrong ones has to be
/// caught, or nothing links one epoch to the next.
#[test]
fn a_broken_register_carry_is_rejected() {
    let elf_bytes = asm_elf_bytes("sub");
    let opts = ProofOptions::default_test_options();
    let mut epochs =
        multilinear_continuation::prove_epochs(&elf_bytes, &[], 2, &opts).expect("prove");
    // ⚠ NOT `if epochs.len() < 2 { return; }`, which is how this test spent its
    // life: at 2^4 `sub` is a single epoch, so the early return fired and the
    // assertion below never ran. A run with nothing to chain is a fixture
    // failure, not a pass.
    assert!(
        epochs.len() >= 2,
        "the carry needs two epochs to cross; this fixture produced {}",
        epochs.len()
    );
    // Claim the first epoch ended somewhere it did not.
    epochs[0].reg_fini[1] ^= 1;
    assert!(
        !multilinear_continuation::verify_epochs(&elf_bytes, &epochs, &opts).expect("verify"),
        "a restated register carry was accepted"
    );
}

/// ★★ THE CARRY, CHECKED ONE EPOCH AT A TIME AND FROM BOTH ENDS.
///
/// `verify_epochs` checks every epoch of a run, so a `false` from it says the
/// RUN is bad and not which epoch objected, nor to what. This hands ONE epoch
/// its starting register file directly, which is the value the chain is made
/// of, and asserts the refusal beside the control that the same epoch under the
/// honest vector is ACCEPTED — without that half, a verifier that refused
/// everything would pass.
///
/// ⛔ WHAT THIS CAUGHT. Both arms were ACCEPTED before the fix in this commit,
/// on `verify_epoch` and through `verify_epochs`, measured on three epochs of
/// `test_private_input_xpage`: `continuation::build_epoch_airs` gave REGISTER a
/// preprocessed COMMITMENT and no columns closure, the multilinear verifier
/// checks preprocessed COLUMNS and never a root, and an empty column list is
/// walked in zero iterations. INIT and FINI were prover-chosen main trace, so
/// the epoch boundary bound nothing on this path. Two indices are flipped
/// because they are read by different consumers: a general-purpose word, and
/// [`register::X254_INDEX`], the synthetic commit index the public output rides
/// on.
#[test]
fn a_restated_register_carry_is_refused_by_the_epoch_it_lands_in() {
    let elf_bytes = asm_elf_bytes("sub");
    let elf = Elf::load(&elf_bytes).expect("load");
    let opts = ProofOptions::default_test_options();
    let epochs = multilinear_continuation::prove_epochs(&elf_bytes, &[], 2, &opts).expect("prove");
    assert!(
        epochs.len() >= 2,
        "a carry needs an epoch to land in; this fixture produced {}",
        epochs.len()
    );

    let verify_one = |index: usize, carried: &[u32]| {
        multilinear_continuation::verify_epoch(
            &elf,
            &elf_bytes,
            &epochs[index],
            carried,
            index + 1 == epochs.len(),
            local_to_global::epoch_label(index as u64),
            &opts,
        )
        .expect("verify_epoch")
    };

    let honest = epochs[0].reg_fini.clone();
    assert!(
        verify_one(1, &honest),
        "the control: epoch 1 under the registers epoch 0 proved it ended with"
    );

    for at in [1usize, register::X254_INDEX] {
        let mut restated = honest.clone();
        restated[at] ^= 1;
        assert!(
            !verify_one(1, &restated),
            "epoch 1 accepted a starting register file differing at index {at} from \
             the one the previous epoch proved"
        );
    }
}

/// The other end of the same binding: what an epoch CLAIMS it ended with.
///
/// The carry test restates the vector the next epoch starts from; this restates
/// the vector the epoch itself publishes, and verifies that epoch alone. The two
/// are the same column pair read by the two neighbours, and a fix that bound
/// only one of them would pass one of these tests.
#[test]
fn a_restated_register_fini_is_refused_by_the_epoch_that_states_it() {
    let elf_bytes = asm_elf_bytes("sub");
    let elf = Elf::load(&elf_bytes).expect("load");
    let opts = ProofOptions::default_test_options();
    let mut epochs =
        multilinear_continuation::prove_epochs(&elf_bytes, &[], 2, &opts).expect("prove");
    assert!(
        epochs.len() >= 2,
        "this fixture produced {} epochs",
        epochs.len()
    );
    let entry = register::register_init_from_entry_point(elf.entry_point);

    let verify_first = |epochs: &[multilinear_continuation::EpochProof]| {
        multilinear_continuation::verify_epoch(
            &elf,
            &elf_bytes,
            &epochs[0],
            &entry,
            epochs.len() == 1,
            local_to_global::epoch_label(0),
            &opts,
        )
        .expect("verify_epoch")
    };

    assert!(
        verify_first(&epochs),
        "the control: epoch 0 as it was proved"
    );
    epochs[0].reg_fini[register::X254_INDEX] ^= 1;
    assert!(
        !verify_first(&epochs),
        "epoch 0 was accepted while claiming a final register file it did not reach"
    );
}

/// ★★ THE HASH AGREEMENT IS CRYPTOGRAPHIC, AND HERE IS THE MEASUREMENT.
///
/// Level 0's driver cannot refuse a bundle by reading a label: a WHIR proof's
/// bytes are hash-agnostic by design — both arms serialise to the same 6904
/// bytes, which is the byte gate's own invariant — and `whir_hash_knob` is a
/// cached process setting that says what THIS process proves under, not what a
/// bundle was proven under. So the refusal the design settles on is the
/// verification itself: hand the epoch to a verifier configured with a hash,
/// and a bundle proven under another one fails because the transcript's sponge
/// is part of the configuration and every challenge diverges.
///
/// ⚠ THAT WAS A DESIGN CLAIM UNTIL SOMETHING RAN IT, and the driver that will
/// depend on it lives on another branch. The claim itself does not: `prove_epoch`
/// and `verify_epoch_bookend` both take `H` here, so it is measurable today,
/// and measuring it now means the seam arrives with its premise already checked
/// rather than assumed.
///
/// ⚠ BOTH HALVES, because a verifier that refused everything would pass the
/// refusing one alone. The same epoch is verified under the hash it was proven
/// with and must be ACCEPTED — and the refusal is a bare `None`, so the control
/// is what makes it mean anything (`verify_epoch_bookend` collapses every
/// failure to `Ok(None)`).
#[test]
fn an_epoch_proven_under_one_hash_is_refused_under_the_other() {
    use multilinear::whir_hash::{KeccakWhir, RpxWhir};

    let elf_bytes = asm_elf_bytes("sub");
    let elf = Elf::load(&elf_bytes).expect("load");
    let opts = ProofOptions::default_test_options();
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    let entry = register::register_init_from_entry_point(elf.entry_point);

    let keccak = multilinear_continuation::decode_prepared_for::<KeccakWhir>(&elf, &elf_bytes)
        .expect("DECODE's commitment under keccak");
    let rpx = multilinear_continuation::decode_prepared_for::<RpxWhir>(&elf, &elf_bytes)
        .expect("DECODE's commitment under rpx");

    // Epoch 0 only: the carry is not what this is about, and one epoch is one
    // prove. `is_final` and `label` travel with it, because a verifier that
    // disagreed about either would refuse for a reason that is not the hash.
    let mut first: Option<(multilinear_continuation::EpochProof, bool, u64)> = None;
    continuation::for_each_epoch(&elf, &[], 2, &artifacts, |prepared, _| {
        if first.is_none() {
            let PreparedEpoch {
                register_init,
                label,
                traces,
                boundary,
                is_final,
                ..
            } = prepared;
            let proof = multilinear_continuation::prove_epoch::<KeccakWhir>(
                &elf,
                &elf_bytes,
                &register_init,
                label,
                traces,
                is_final,
                &boundary,
                &opts,
                None,
                &keccak,
            )?;
            first = Some((proof, is_final, label));
        }
        Ok(())
    })
    .expect("prove epoch 0 under keccak");

    let (epoch, is_final, label) = first.expect("the program has at least one epoch");

    assert!(
        multilinear_continuation::verify_epoch_bookend::<KeccakWhir>(
            &elf, &elf_bytes, &epoch, &entry, is_final, label, &opts, &keccak,
        )
        .expect("verify")
        .is_some(),
        "the control: an epoch proven under keccak must verify under keccak, or \
         the refusal below says nothing"
    );

    assert!(
        multilinear_continuation::verify_epoch_bookend::<RpxWhir>(
            &elf, &elf_bytes, &epoch, &entry, is_final, label, &opts, &rpx,
        )
        .expect("verify")
        .is_none(),
        "an epoch proven under keccak was ACCEPTED by a verifier configured with \
         RPX; the level-0 driver's whole hash agreement rests on that being \
         impossible"
    );
}

/// ★★ THE CROSS-EPOCH HALF OF THE HASH AGREEMENT — the sibling of
/// [`an_epoch_proven_under_one_hash_is_refused_under_the_other`], and the test
/// that makes `real_global_from_whir_continuation_under::<H>`'s name true.
///
/// It could not be written at all until `verify_global_bookends` took `H`: a
/// function that reads `whir_hash_knob::selected()` for itself can only be
/// asked what the PROCESS proves under, never what the bundle in front of it
/// was proven under. The knob is a setting; the agreement is the verification,
/// because the transcript's sponge is part of the configuration and every
/// challenge diverges from the first squeeze.
///
/// ★ THE SHAPE IS KNOB-INDEPENDENT, deliberately. `prove_continuation`
/// dispatches on the knob, so this suite's bundle is proven under whichever
/// hash the process was started with — asserting "keccak accepts" would pass
/// for the wrong reason under `LAMBDA_VM_WHIR_HASH=rpx`. The property asserted
/// is that **exactly one of the two hashes accepts**, which is at once the
/// control and the refusal: a verifier accepting everything fails it, and so
/// does one accepting nothing.
///
/// ⚠ WHAT THE LAST ASSERTION CAN AND CANNOT CATCH. Comparing the DISPATCHING
/// entry point against the explicit arm is what would see `verify_global`'s
/// dispatch pinned to one hash — but only when the process is set to the OTHER
/// one. Under a default keccak suite a dispatch pinned to keccak is
/// indistinguishable from a correct one here; a dispatch pinned to RPX is not.
/// Stated so nobody reads this as covering both directions.
#[test]
fn a_cross_epoch_proof_proven_under_one_hash_is_refused_under_the_other() {
    use multilinear::whir_hash::{KeccakWhir, RpxWhir, WhirHash};

    let (elf_bytes, input) = a_run_that_touches_memory();
    let opts = ProofOptions::default_test_options();
    let bundle =
        multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts).expect("prove");
    let elf = Elf::load(&elf_bytes).expect("load");

    let verdict_under =
        |name: &str,
         verdict: Result<Option<multilinear_continuation::GlobalVerdict>, crate::Error>| {
            verdict
                .unwrap_or_else(|e| panic!("the cross-epoch proof errored under {name}: {e:?}"))
                .is_some()
        };
    let keccak = verdict_under(
        <KeccakWhir as WhirHash>::NAME,
        multilinear_continuation::verify_global_bookends::<KeccakWhir>(
            &elf,
            &elf_bytes,
            &bundle.global,
            bundle.num_epochs(),
            &bundle.touched_page_bases,
            bundle.num_private_input_pages,
            &opts,
        ),
    );
    let rpx = verdict_under(
        <RpxWhir as WhirHash>::NAME,
        multilinear_continuation::verify_global_bookends::<RpxWhir>(
            &elf,
            &elf_bytes,
            &bundle.global,
            bundle.num_epochs(),
            &bundle.touched_page_bases,
            bundle.num_private_input_pages,
            &opts,
        ),
    );

    assert!(
        keccak != rpx,
        "the cross-epoch proof was accepted under BOTH hashes or under NEITHER \
         (keccak {keccak}, rpx {rpx}); the driver's hash agreement rests on exactly \
         one of them accepting"
    );

    // Which one it is has to be the hash this process proved it under, and the
    // DISPATCHING entry point has to reach the same verdict.
    let setting = crate::whir_hash_knob::selected().name();
    // ★ PRINTED, because which direction this run exercises is not a property
    // of the code — it is a property of the process the suite was started in,
    // and the last assertion below can only see a dispatch pinned to the OTHER
    // hash. A reader of the log should not have to infer which arm was live.
    println!(
        "CROSS-EPOCH HASH AGREEMENT: process {setting}; accepted under keccak256 {keccak}, \
         under rpx256 {rpx}"
    );
    let expected = if setting == <KeccakWhir as WhirHash>::NAME {
        keccak
    } else {
        rpx
    };
    assert!(
        expected,
        "the bundle does not verify under {setting}, the very hash this process proved it with"
    );
    let dispatched = multilinear_continuation::verify_global(
        &elf,
        &elf_bytes,
        &bundle.global,
        bundle.num_epochs(),
        &bundle.touched_page_bases,
        bundle.num_private_input_pages,
        &opts,
    )
    .expect("the dispatching entry point");
    assert_eq!(
        dispatched, expected,
        "`verify_global` did not verify under {setting}, the hash its own knob names"
    );
}

/// **The binding.** An epoch commits its local-to-global bookend on its own and
/// the cross-epoch proof commits the same table: the two roots have to match,
/// or nothing says they are the same table.
///
/// This rebuilds that root from the boundary alone — which is all the
/// cross-epoch proof holds — and demands the epoch's proof carries it.
#[test]
fn the_bookend_commits_to_a_root_the_cross_epoch_proof_can_reproduce() {
    let elf_bytes = asm_elf_bytes("sub");
    let elf = Elf::load(&elf_bytes).expect("load");
    let opts = ProofOptions::default_test_options();
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");

    let mut roots: Vec<(
        Vec<stark::config::Commitment>,
        Vec<stark::config::Commitment>,
    )> = Vec::new();
    crate::with_whir_hash!(|H| {
        let pinned = multilinear_continuation::decode_prepared_for::<H>(&elf, &elf_bytes)
            .expect("DECODE's out-of-band commitment");
        continuation::for_each_epoch(&elf, &[], 4, &artifacts, |prepared, _| {
            let boundary = std::sync::Arc::clone(&prepared.boundary);
            let register_init = prepared.register_init.clone();
            let label = prepared.label;
            let is_final = prepared.is_final;
            let proof = multilinear_continuation::prove_epoch::<H>(
                &elf,
                &elf_bytes,
                &register_init,
                label,
                prepared.traces,
                is_final,
                &boundary,
                &opts,
                None,
                &pinned,
            )?;
            // The config the standalone commitment runs at is the epoch's: the root
            // depends on the blowup and the fold width, and those are fixed —
            // which is exactly what lets two proofs over different table sets agree
            // on it.
            let shapes: Vec<(usize, usize)> = proof
                .table_num_vars
                .iter()
                .map(|&n| (1usize, n as usize))
                .collect();
            let config = crate::multilinear_prove::chain_config(&shapes);
            let standalone = multilinear_continuation::l2g_commitment(&boundary, &config)
                .expect("the bookend commits");
            // How many polynomials the bookend's group stacks into, derived the way
            // the verifier derives it rather than read off the standalone roots.
            let widths: Vec<(usize, usize)> = proof
                .table_num_vars
                .iter()
                .map(|&n| (1usize, n as usize))
                .collect();
            let sizes = multilinear_continuation::epoch_groups(widths.len());
            let (layouts, _) = crate::multilinear_prove::stacks(&widths, &sizes, &config)
                .expect("the epoch's stacks");
            let num_polys = layouts.last().expect("a bookend group").num_polys();
            roots.push((
                proof
                    .l2g_roots(num_polys)
                    .expect("the epoch carries its bookend's roots")
                    .to_vec(),
                standalone,
            ));
            Ok(())
        })
    })
    .expect("the epochs prepare");

    assert!(!roots.is_empty());
    for (index, (carried, rebuilt)) in roots.iter().enumerate() {
        assert_eq!(
            carried, rebuilt,
            "epoch {index}'s bookend roots are not the ones its table commits to"
        );
    }
}

/// A run that reads its private input from another page, so it touches memory
/// across epochs — which is what the cross-epoch proof is about.
fn a_run_that_touches_memory() -> (Vec<u8>, Vec<u8>) {
    let mut input: Vec<u8> = Vec::with_capacity(16);
    input.extend_from_slice(&16u32.to_le_bytes());
    input.extend_from_slice(&[0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
    input.extend_from_slice(&[0u8; 4]);
    (asm_elf_bytes("test_private_input_xpage"), input)
}

/// A whole run: every epoch plus the one cross-epoch proof that chains their
/// memory, checked from the bundle and the ELF alone.
#[test]
fn a_continuation_proves_and_verifies() {
    let (elf_bytes, input) = a_run_that_touches_memory();
    let opts = ProofOptions::default_test_options();
    let bundle =
        multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts).expect("prove");
    // The committed output is the run's, not any one epoch's.
    assert_eq!(bundle.public_output(), input[4..12]);
    assert!(bundle.num_epochs() >= 1);
    assert!(
        multilinear_continuation::verify_continuation(&elf_bytes, &bundle, &opts).expect("verify"),
        "the continuation does not verify"
    );
}

/// ★ THE PROPERTY EVERY OTHER EPOCH TEST IN THIS FILE IS BLIND TO.
///
/// `verify_epoch_bookend` forks the transcript and replays the roots block to
/// draw the `z` and `alpha` that the COMMIT bus's counterparty is a function of,
/// then hands that counterparty to `multi_verify` as the balance the tables owe.
/// [`crate::compute_commit_bus_offset`] returns ZERO for an empty public output
/// **without reading either challenge**, so an epoch that publishes nothing
/// accepts whatever the replay drew: a replay absorbing a shorter roots block
/// than the verification is unobservable there. Every other epoch test in this
/// file proves `sub`, which publishes nothing in any epoch, and all of them were
/// green while the replay was drifting.
///
/// So this proves and verifies every epoch of a program that DOES publish, and
/// asserts that at least one epoch carried output. Without that second
/// assertion the test would pass vacuously the day the program stopped
/// publishing — the same way the rest of the file passed.
#[test]
fn an_epoch_that_publishes_output_proves_and_verifies() {
    let (elf_bytes, input) = a_run_that_touches_memory();
    let elf = Elf::load(&elf_bytes).expect("load");
    let opts = ProofOptions::default_test_options();
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");

    let mut carried = register::register_init_from_entry_point(elf.entry_point);
    let mut epochs = 0usize;
    let mut published = 0usize;

    crate::with_whir_hash!(|H| {
        let pinned = multilinear_continuation::decode_prepared_for::<H>(&elf, &elf_bytes)
            .expect("DECODE's out-of-band commitment");
        continuation::for_each_epoch(&elf, &input, 2, &artifacts, |prepared, _| {
            let PreparedEpoch {
                register_init,
                label,
                traces,
                boundary,
                is_final,
                ..
            } = prepared;
            let proof = multilinear_continuation::prove_epoch::<H>(
                &elf,
                &elf_bytes,
                &register_init,
                label,
                traces,
                is_final,
                &boundary,
                &opts,
                None,
                &pinned,
            )?;
            if !proof.public_output.is_empty() {
                published += 1;
            }
            assert!(
                multilinear_continuation::verify_epoch(
                    &elf, &elf_bytes, &proof, &carried, is_final, label, &opts
                )?,
                "epoch {label} does not verify"
            );
            carried = proof.reg_fini;
            epochs += 1;
            Ok(())
        })
    })
    .expect("the epochs prepare");

    assert!(epochs > 0, "the program ran no epochs");
    assert!(
        published > 0,
        "no epoch of this run published output, so the test cannot see the replay \
         it exists to check: pick a program that commits some"
    );
}

/// ⛔ THE REPLAY HAS TO SEE THE DERIVED ROOT.
///
/// Deliberately NOT a comparison against another spelling of the roots block —
/// that would compare the block with itself and could not fail. It asserts
/// instead that `owed`'s answer MOVES when the derived list does. A replay that
/// ignores its derived roots, which is what this branch shipped until the fix,
/// returns the same counterparty for both lists and fails here.
///
/// Sensitivity is not agreement: a replay could absorb the derived root in the
/// wrong POSITION and still be sensitive to it. Agreement with `multi_verify` is
/// what [`an_epoch_that_publishes_output_proves_and_verifies`] checks, and the
/// position is what `the_roots_block_binds_the_derived_root_to_the_first_challenge`
/// checks in the `stark` crate. Three different failures, three checks.
#[test]
fn the_owed_replay_is_sensitive_to_the_derived_roots() {
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;

    // Non-empty, or the counterparty is zero whatever the challenges are — which
    // is the very blindness this test exists beside.
    let public_output = [0xA5u8, 0x5A, 0x01];
    let register_init = register::register_init_from_entry_point(0x1000);
    let carried: Vec<stark::config::Commitment> = vec![[7u8; 32], [11u8; 32]];
    let derived: Vec<stark::config::Commitment> = vec![[13u8; 32]];
    let transcript = DefaultTranscript::<crate::test_utils::E>::new(b"w1c-owed-replay");

    let without =
        multilinear_continuation::owed(&public_output, &register_init, &carried, &[], &transcript)
            .expect("the commit fingerprints are invertible");
    let with = multilinear_continuation::owed(
        &public_output,
        &register_init,
        &carried,
        &derived,
        &transcript,
    )
    .expect("the commit fingerprints are invertible");

    assert_ne!(
        without, with,
        "the replay drew the same challenges with and without the derived root, \
         so it is absorbing a shorter roots block than `multi_verify` does"
    );
}

/// ⛔ THE REPLAY DRAWS WHAT THE BLOCK DRAWS, IN THE ORDER THE BLOCK DRAWS IT.
///
/// The replay shares the roots block's ABSORB half and spells its own two draws,
/// because a fork that is thrown away must not pay for a third squeeze nobody
/// reads. What that split leaves open is the draw ORDER, so it is checked here
/// against the block itself rather than trusted: the counterparty computed from
/// the block's first two challenges must equal the one `owed` returns.
///
/// Two implementations, not one — which is what makes this able to fail. Swap
/// `z` and `alpha` in either and it goes red.
#[test]
fn the_owed_replay_draws_what_the_roots_block_draws() {
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;

    let public_output = [0x11u8, 0x22, 0x33, 0x44];
    let register_init = register::register_init_from_entry_point(0x2000);
    let carried: Vec<stark::config::Commitment> = vec![[3u8; 32], [5u8; 32]];
    let derived: Vec<stark::config::Commitment> = vec![[17u8; 32]];
    let seed = DefaultTranscript::<crate::test_utils::E>::new(b"w1c-owed-agrees");

    let from_the_replay =
        multilinear_continuation::owed(&public_output, &register_init, &carried, &derived, &seed)
            .expect("the commit fingerprints are invertible");

    let mut block = seed.clone();
    let (z, alpha, _beta) = stark::multilinear_table::absorb_roots_and_challenge::<
        crate::test_utils::E,
        _,
    >(&mut block, &carried, &derived);
    let start_index = register_init[crate::tables::register::X254_INDEX] as u64;
    let from_the_block = crate::compute_commit_bus_offset(&public_output, start_index, &z, &alpha)
        .expect("the commit fingerprints are invertible");

    assert_eq!(
        from_the_replay, from_the_block,
        "the replay and the roots block disagree on the challenges, so the \
         counterparty is computed at challenges no table is checked at"
    );
}

/// ★ WHY THE REST OF THIS FILE COULD NOT SEE THE DRIFT, written where it can be
/// falsified instead of in a comment.
///
/// The COMMIT bus's counterparty is the only consumer of the challenges the
/// replay draws, and it does not read them at all when the epoch published
/// nothing. That is what made a wrong replay invisible on every `sub` epoch and
/// visible only on the one epoch of `test_private_input_xpage` that commits its
/// output. If either half of this stops holding — an empty output that starts
/// depending on the challenges, or a non-empty one that stops — the reasoning
/// above these tests needs rewriting, and this says so.
#[test]
fn an_epoch_with_no_public_output_cannot_see_a_replay_drift() {
    use crate::test_utils::E;
    use math::field::element::FieldElement;

    let z1 = FieldElement::<E>::from(3u64);
    let a1 = FieldElement::<E>::from(5u64);
    let z2 = FieldElement::<E>::from(7u64);
    let a2 = FieldElement::<E>::from(11u64);

    let empty_at_one = crate::compute_commit_bus_offset(&[], 4, &z1, &a1).expect("empty is zero");
    let empty_at_two = crate::compute_commit_bus_offset(&[], 4, &z2, &a2).expect("empty is zero");
    assert_eq!(
        empty_at_one,
        FieldElement::<E>::zero(),
        "an epoch that published nothing owes nothing"
    );
    assert_eq!(
        empty_at_one, empty_at_two,
        "the empty-output counterparty must not depend on the challenges, or the \
         blindness this reasoning rests on is not there"
    );

    let some_at_one =
        crate::compute_commit_bus_offset(&[1, 2, 3], 4, &z1, &a1).expect("invertible");
    let some_at_two =
        crate::compute_commit_bus_offset(&[1, 2, 3], 4, &z2, &a2).expect("invertible");
    assert_ne!(
        some_at_one, some_at_two,
        "a published output must bind the challenges, or no epoch could ever \
         detect a replay drift"
    );
}

/// The bookends are windows into one flat list of roots — one root per stacked
/// polynomial, not per table — so the binding rests on that arithmetic. An
/// epoch long enough that its bookend needs two polynomials widens the window;
/// it does not move the next one along.
#[test]
fn the_bookend_roots_are_consecutive_windows() {
    let (elf_bytes, input) = a_run_that_touches_memory();
    let opts = ProofOptions::default_test_options();
    let bundle =
        multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts).expect("prove");
    let epochs = bundle.num_epochs();
    let roots = &bundle.global.proof.roots;

    let windows = bundle
        .global
        .l2g_roots(&vec![1usize; epochs])
        .expect("the bookends are committed first");
    assert_eq!(windows.len(), epochs);
    for (index, window) in windows.iter().enumerate() {
        assert_eq!(*window, &roots[index..index + 1]);
    }

    // A window nobody can fill comes back empty-handed rather than reading
    // whatever root sits next to it.
    assert!(bundle.global.l2g_roots(&[roots.len() + 1]).is_none());
    assert!(bundle.global.l2g_roots(&[0]).is_none());
    assert!(bundle.global.l2g_roots(&[usize::MAX]).is_none());
    assert!(bundle.epochs[0].l2g_roots(0).is_none());
    assert!(
        bundle.epochs[0]
            .l2g_roots(bundle.epochs[0].proof.roots.len() + 1)
            .is_none()
    );
}

/// A moved bookend root is rejected.
///
/// ⚠ AND THE HALF THAT REJECTS IT IS THE EPOCH'S, NOT THE BINDING'S. The root
/// this moves is the last of epoch 0's OWN proof, so the epoch's `multi_verify`
/// refuses it and `verify_continuation` returns before the cross-epoch half is
/// reached — established by a mutation that broke the cross-epoch AIR set and
/// left this test green. It is kept for what it does cover, and its old claim
/// that "both halves still verify on their own" is withdrawn:
/// [`a_swapped_bookend_root_is_caught_by_the_binding`] is the one that reaches
/// `proved == chained`.
#[test]
fn a_bookend_that_is_not_the_one_chained_is_rejected() {
    let (elf_bytes, input) = a_run_that_touches_memory();
    let opts = ProofOptions::default_test_options();
    let mut bundle =
        multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts).expect("prove");
    // ★ THE HONEST-PATH CONTROL. Without it a verifier broken in any way keeps
    // this test green, because it asserts only that something was refused.
    assert!(
        multilinear_continuation::verify_continuation(&elf_bytes, &bundle, &opts).expect("verify"),
        "the untampered bundle does not verify, so the refusal below proves nothing"
    );
    let last = bundle.epochs[0].proof.roots.len() - 1;
    bundle.epochs[0].proof.roots[last][0] ^= 1;
    assert!(
        !multilinear_continuation::verify_continuation(&elf_bytes, &bundle, &opts).expect("verify"),
        "a bookend root the cross-epoch proof never chained was accepted"
    );
}

/// The touched page set drives which cross-epoch tables exist, so restating it
/// has to be rejected.
#[test]
fn a_restated_touched_page_set_is_rejected() {
    let (elf_bytes, input) = a_run_that_touches_memory();
    let opts = ProofOptions::default_test_options();
    let mut bundle =
        multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts).expect("prove");
    assert!(
        !bundle.touched_page_bases.is_empty(),
        "the run touched memory"
    );
    // ★ THE HONEST-PATH CONTROL, and here it is the one that carries the
    // weight: the refusal below accepts `Err` as well as `Ok(false)`, so
    // without this line a verifier that errored on every bundle would pass.
    assert!(
        multilinear_continuation::verify_continuation(&elf_bytes, &bundle, &opts).expect("verify"),
        "the untampered bundle does not verify, so the refusal below proves nothing"
    );
    bundle.touched_page_bases.pop();
    assert!(
        matches!(
            multilinear_continuation::verify_continuation(&elf_bytes, &bundle, &opts),
            Ok(false) | Err(_)
        ),
        "a restated touched page set was accepted"
    );
}

/// ★★ THE BINDING ITSELF, reached with BOTH HALVES VALID — the check
/// `verify_continuation` ends on (`proved == chained`) and the one thing that
/// makes the epochs and the cross-epoch proof one proof rather than two about
/// unrelated tables.
///
/// Nothing else in this file reaches it. The arm above moves a root inside an
/// epoch's own proof, so the epoch half refuses first; a restated page set
/// changes the table count, so the cross-epoch half errors first. The tamper
/// that reaches the comparison has to leave every root a VALID commitment and
/// only put them in the wrong ORDER: swapping two epochs' bookend windows
/// inside the CROSS-EPOCH proof does exactly that. Each half still argues about
/// tables it committed; what is false is which epoch's bookend the cross-epoch
/// proof says it chained.
///
/// ⚠ WHAT THIS FIXTURE CAN AND CANNOT SHOW. The run has three epochs and one
/// page, so there are two distinct bookends to swap and the tamper is
/// constructible — but each bookend's window is ONE root here, so the swap does
/// not exercise the multi-polynomial windows that a long epoch produces on a
/// block (where a bookend needing two polynomials widens its window). The
/// assertion that the two roots differ before the swap is what stops this from
/// silently becoming a no-op if a future fixture commits them identically.
#[test]
fn a_swapped_bookend_root_is_caught_by_the_binding() {
    let (elf_bytes, input) = a_run_that_touches_memory();
    let opts = ProofOptions::default_test_options();
    let mut bundle =
        multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts).expect("prove");
    let epochs = bundle.num_epochs();
    assert!(
        epochs >= 2,
        "a one-epoch run has no second bookend to swap with"
    );

    // ★ THE HONEST-PATH CONTROL.
    assert!(
        multilinear_continuation::verify_continuation(&elf_bytes, &bundle, &opts).expect("verify"),
        "the untampered bundle does not verify, so the refusal below proves nothing"
    );

    // The bookends are the first commitment groups, one root each at this
    // fixture, so epochs 0 and 1 are roots 0 and 1 of the CROSS-EPOCH proof.
    // Both stay valid commitments to tables the proof really committed; only
    // the order changes, which is the one thing the binding exists to catch.
    let windows = bundle
        .global
        .l2g_roots(&vec![1usize; epochs])
        .expect("the bookends are committed first");
    assert_eq!(windows.len(), epochs);
    assert_ne!(
        windows[0], windows[1],
        "the two bookends committed to the same root, so swapping them is a no-op"
    );
    bundle.global.proof.roots.swap(0, 1);

    assert!(
        !multilinear_continuation::verify_continuation(&elf_bytes, &bundle, &opts).expect("verify"),
        "the cross-epoch proof claimed to chain a bookend that is not epoch 0's, and the \
         binding accepted it"
    );
}

/// The cross-epoch AIR set is built ONCE, by `global_airs_for`, and this is
/// what that one set has to say about the proof it was asked for.
///
/// ★ WIDTH IS THE HALF A PROOF DOES NOT STATE. `GlobalProof` carries a height
/// per table and nothing else, so a set built in the wrong ORDER — the pages
/// before the bookends, or one family built twice — is invisible to every count
/// the proof itself can check. The two families' widths differ (9 against 4),
/// which is what makes the order observable at all.
///
/// The preprocessed route is the other half: a GLOBAL_MEMORY table preprocesses
/// OFFSET alone on a private-input page and OFFSET+INIT everywhere else. ⚠ What
/// that half can catch is a page routed against the wrong config — the configs
/// ordered differently from the page bases, or the branch taken the wrong way.
/// It does NOT independently check `is_private_input_page`, which both sides
/// reach.
///
/// ⛔ WHAT THIS FIXTURE DOES NOT COVER, measured and printed rather than
/// assumed. The run's cross-epoch shape is **4 tables = 3 bookends + 1 page,
/// and that page is the PRIVATE-INPUT one**, so:
///   * the OFFSET+INIT route — the one every ELF-backed and zero-init page on a
///     real block takes — has NO table here and therefore no gate here. Closing
///     it needs a fixture whose run leaves cross-epoch cells on a non-private
///     page, not another assertion on this one.
///   * the group split is pinned only as a COUNT: with a single page the page
///     group has one member and is shape-identical to a bookend singleton. At
///     block scale (fifteen singletons then one group of thirty-five) the same
///     assertion pins the split as well.
///
/// `touched_page_bases` lists the pages carrying cells that CROSS an epoch
/// boundary, which is why a run that reads its input from another page still
/// reaches only one of them.
#[test]
fn the_global_airs_describe_the_cross_epoch_proof_they_were_asked_for() {
    let (elf_bytes, input) = a_run_that_touches_memory();
    let opts = ProofOptions::default_test_options();
    let bundle =
        multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts).expect("prove");
    let elf = Elf::load(&elf_bytes).expect("load");

    let epochs = bundle.num_epochs();
    // The AIRs come back in the canonical page order the config build imposes,
    // not in the bundle's wire order.
    let mut bases = bundle.touched_page_bases.clone();
    bases.sort_unstable();
    bases.dedup();
    assert!(!bases.is_empty(), "the run touched no memory");

    let airs = multilinear_continuation::global_airs_for(
        &elf,
        &opts,
        epochs,
        &bundle.touched_page_bases,
        bundle.num_private_input_pages,
    );
    let refs = airs.refs();

    // The proof states one height per table and no width at all, so the AIR set
    // and the proof reach this count by different routes.
    assert_eq!(
        refs.len(),
        bundle.global.table_num_vars.len(),
        "the cross-epoch AIR set and the proof describe a different number of tables"
    );
    assert_eq!(
        refs.len(),
        epochs + bases.len(),
        "the set is not one bookend per epoch and one table per touched page"
    );

    for (index, air) in refs.iter().enumerate() {
        let expected = if index < epochs {
            local_to_global::cols::NUM_COLUMNS
        } else {
            crate::tables::global_memory::cols::NUM_COLUMNS
        };
        assert_eq!(
            air.trace_layout().0,
            expected,
            "table {index} is not the family the layout puts there (bookends 0..{epochs}, pages {epochs}..{})",
            refs.len()
        );
    }

    // Every bookend committed alone — which is what lets its root be compared
    // against the epoch that committed it — and then the pages together.
    let mut expected_groups = vec![1usize; epochs];
    expected_groups.push(bases.len());
    assert_eq!(
        airs.groups(),
        expected_groups,
        "the cross-epoch commitment groups are not fifteen-style singletons plus the pages"
    );

    let mut private = 0usize;
    let mut genesis = 0usize;
    for (page, base) in bases.iter().enumerate() {
        let air = refs[epochs + page];
        let is_private =
            crate::tables::page::is_private_input_page(*base, bundle.num_private_input_pages);
        let expected = if is_private {
            private += 1;
            crate::tables::page::NUM_PREPROCESSED_COLS_PRIVATE
        } else {
            genesis += 1;
            crate::tables::global_memory::NUM_PREPROCESSED_COLS
        };
        assert_eq!(
            air.num_precomputed_columns(),
            expected,
            "page {page} at base {base:#x} declares the wrong preprocessed route"
        );
        // The declared count and the columns the closure builds are two
        // different members of the AIR, and the statement is built from both.
        assert_eq!(
            air.precomputed_columns().len(),
            expected,
            "page {page} at base {base:#x} builds a different number of preprocessed columns than it declares"
        );
    }
    // Both arms bump, so the sum is a prediction and not a restatement of the
    // loop's own trip count.
    assert_eq!(private + genesis, bases.len());
    println!(
        "GLOBAL AIRS: {} tables = {epochs} bookends + {} pages ({private} private-input OFFSET-only, {genesis} OFFSET+INIT)",
        refs.len(),
        bases.len(),
    );
}

/// The cross-epoch inputs a caller of [`multilinear_continuation::prove_global`]
/// needs, WITHOUT proving a single epoch.
///
/// `for_each_epoch` is the same function `prove_continuation` walks the run
/// with, and its closure is what proves an epoch — so passing a closure that
/// does nothing hands back exactly the boundaries the real prover would have
/// chained, at the cost of the execution alone. That matters: re-deriving the
/// boundaries in a test would be a second spelling of the epoch split, which is
/// the thing the cross-epoch proof is about.
fn cross_epoch_inputs(
    elf_bytes: &[u8],
    input: &[u8],
    epoch_size_log2: u32,
) -> (
    Elf,
    Vec<std::sync::Arc<Vec<local_to_global::CellBoundary>>>,
    std::collections::HashMap<u64, Vec<u8>>,
    Vec<u64>,
    usize,
) {
    let elf = Elf::load(elf_bytes).expect("load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    let boundaries = continuation::for_each_epoch(
        &elf,
        input,
        epoch_size_log2,
        &artifacts,
        |_prepared, _so_far| Ok(()),
    )
    .expect("walk the run");
    let init_page_data = crate::tables::trace_builder::build_init_page_data(
        &crate::tables::trace_builder::build_initial_image_paged(&elf, input),
    );
    let page_bases = continuation::touched_page_bases(&boundaries);
    let num_private = crate::tables::page::private_input_page_count(input);
    (elf, boundaries, init_page_data, page_bases, num_private)
}

/// ★★★ THE CROSS-EPOCH HASH AGREEMENT, IN BOTH DIRECTIONS AND WITHOUT THE KNOB.
///
/// [`a_cross_epoch_proof_proven_under_one_hash_is_refused_under_the_other`] can
/// only assert that EXACTLY ONE hash accepts, because `prove_continuation`
/// dispatches on the process knob and the test cannot say which hash its bundle
/// was proven under. Its own doc says so: under a default keccak suite, a
/// verifier dispatch pinned to keccak is indistinguishable from a correct one.
///
/// With [`multilinear_continuation::prove_global`] generic, the prover can be
/// ASKED. Each proof is built under a named hash and checked under both, so the
/// diagonal must accept and the off-diagonal must refuse — four readings whose
/// pattern is fixed regardless of what `LAMBDA_VM_WHIR_HASH` is set to when the
/// suite runs.
///
/// ⚠ THE HONEST CONTROL IS THE DIAGONAL, and it is asserted first: a verifier
/// that refuses everything satisfies both refusals and fails both acceptances.
#[test]
fn a_cross_epoch_proof_verifies_under_the_hash_it_was_proven_under_and_no_other() {
    use multilinear::whir_hash::{KeccakWhir, RpxWhir, WhirHash};

    let (elf_bytes, input) = a_run_that_touches_memory();
    let opts = ProofOptions::default_test_options();
    let (elf, boundaries, init_page_data, page_bases, num_private) =
        cross_epoch_inputs(&elf_bytes, &input, 2);

    let keccak_proof = multilinear_continuation::prove_global::<KeccakWhir>(
        &boundaries,
        &elf_bytes,
        &init_page_data,
        &page_bases,
        num_private,
        &opts,
    )
    .expect("prove the cross-epoch chain under keccak256");
    let rpx_proof = multilinear_continuation::prove_global::<RpxWhir>(
        &boundaries,
        &elf_bytes,
        &init_page_data,
        &page_bases,
        num_private,
        &opts,
    )
    .expect("prove the cross-epoch chain under rpx256");

    let accepts = |proof: &multilinear_continuation::GlobalProof, rpx: bool| -> bool {
        let verdict = if rpx {
            multilinear_continuation::verify_global_bookends::<RpxWhir>(
                &elf,
                &elf_bytes,
                proof,
                boundaries.len(),
                &page_bases,
                num_private,
                &opts,
            )
        } else {
            multilinear_continuation::verify_global_bookends::<KeccakWhir>(
                &elf,
                &elf_bytes,
                proof,
                boundaries.len(),
                &page_bases,
                num_private,
                &opts,
            )
        };
        verdict.expect("the cross-epoch verifier errored").is_some()
    };

    // The control first: each proof verifies under the hash it names.
    assert!(
        accepts(&keccak_proof, false),
        "a proof built under {} was refused under {}",
        <KeccakWhir as WhirHash>::NAME,
        <KeccakWhir as WhirHash>::NAME
    );
    assert!(
        accepts(&rpx_proof, true),
        "a proof built under {} was refused under {}",
        <RpxWhir as WhirHash>::NAME,
        <RpxWhir as WhirHash>::NAME
    );
    // Then the refusals, which the control has now earned.
    assert!(
        !accepts(&keccak_proof, true),
        "a proof built under {} was ACCEPTED under {}: the transcript's sponge is \
         not part of the configuration after all",
        <KeccakWhir as WhirHash>::NAME,
        <RpxWhir as WhirHash>::NAME
    );
    assert!(
        !accepts(&rpx_proof, false),
        "a proof built under {} was ACCEPTED under {}",
        <RpxWhir as WhirHash>::NAME,
        <KeccakWhir as WhirHash>::NAME
    );

    // ⚠ And the two proofs are not the same object: if they were, the four
    // readings above would be about one proof and the pattern would mean
    // nothing.
    let a = rkyv::to_bytes::<rkyv::rancor::Error>(&keccak_proof.proof.tables).expect("serialize");
    let b = rkyv::to_bytes::<rkyv::rancor::Error>(&rpx_proof.proof.tables).expect("serialize");
    assert_ne!(
        a.as_ref(),
        b.as_ref(),
        "the two hashes produced byte-identical table arguments"
    );
}

/// ★★★★ THE PREPARED GENESIS OPENING, END TO END ON A DENSE PAGE.
///
/// `dense_data_page_touch` is `data_page_touch` with its touched cell surrounded
/// by non-zero bytes, so the page it lives on crosses
/// the genesis-stack threshold whatever offset the linker put `.data` at. It is
/// the only fixture in the tree whose cross-epoch proof carries a prepared
/// opening at all — every other one is entirely sparse and takes the `None`
/// path, which is why this guest had to be written rather than an assertion
/// added.
///
/// ⚠ WHAT THIS DOES AND DOES NOT COVER. It exercises the PROTOCOL: the stack is
/// committed, its root absorbed in the roots block, the opening produced, and
/// the three dense-page INIT checks skipped in favour of it. It does NOT
/// exercise the threshold's decision at BLOCK scale — one page is not the
/// block's three, and the census is the only reading of that. Stated so nobody
/// reads a green here as covering the block.
#[test]
fn a_dense_genesis_page_is_carried_by_a_prepared_opening() {
    let elf_bytes = asm_elf_bytes("dense_data_page_touch");
    let opts = ProofOptions::default_test_options();
    let (elf, boundaries, init_page_data, page_bases, num_private) =
        cross_epoch_inputs(&elf_bytes, &[], 3);

    // The plan, from the VERIFIER's own configs — the ELF's.
    let configs = continuation::global_memory_configs(&page_bases, &elf, num_private);
    let plan = crate::continuation::genesis_stack_plan(
        &configs,
        boundaries.len(),
        crate::continuation::PAGE_NUM_VARS,
    );
    for route in &plan.routes {
        println!(
            "DENSE FIXTURE PAGE {:#x}: nonzero {} has_init {} candidate {} dense {}",
            route.page_base, route.nonzero, route.has_init, route.candidate, route.dense
        );
    }
    // ⚠ THE PRECONDITION, ASSERTED. Without a dense page this test would take
    // the `None` path and pass while checking nothing about the opening.
    assert_eq!(
        plan.dense_pages().len(),
        1,
        "the fixture must put exactly one page over the threshold; it put {}",
        plan.dense_pages().len()
    );
    // ⚠ BOTH of that page's preprocessed columns, which is what keeps
    // `check_preprocessed`'s prefix contract expressible. See
    // `continuation::PAGE_PREPROCESSED_COLUMNS`.
    assert_eq!(
        plan.at,
        stark::multilinear_table::leading_columns(
            plan.at[0].table,
            crate::continuation::PAGE_PREPROCESSED_COLUMNS
        ),
        "a dense page must stack its whole preprocessed prefix"
    );

    let global = crate::with_whir_hash!(|H| {
        multilinear_continuation::prove_global::<H>(
            &boundaries,
            &elf_bytes,
            &init_page_data,
            &page_bases,
            num_private,
            &opts,
        )
    })
    .expect("the cross-epoch proof over a dense genesis page");

    // The opening exists, and its root is NOT among the carried ones — the
    // verifier derives it, and a copy in the proof would be a value a reader
    // assumes is checked.
    assert!(
        global.proof.preprocessed.is_some(),
        "a dense page was planned but the proof carries no prepared opening"
    );

    let verdict = crate::with_whir_hash!(|H| {
        multilinear_continuation::verify_global_bookends::<H>(
            &elf,
            &elf_bytes,
            &global,
            boundaries.len(),
            &page_bases,
            num_private,
            &opts,
        )
    })
    .expect("the cross-epoch verifier errored on an honest bundle");
    assert!(
        verdict.is_some(),
        "an honest cross-epoch proof with a prepared genesis opening was refused"
    );
}

/// ★★ THE SPARSE FIXTURES STILL CARRY NO OPENING, and their proofs are the ones
/// they were.
///
/// The honest control for the route as a whole: a run whose genesis is entirely
/// sparse must take the `None` path. If it did not, every fixture in this suite
/// would be paying for a commitment it has no use for, and the claim that this
/// change moves no proof anything in flight depends on would be false.
#[test]
fn a_sparse_genesis_page_set_carries_no_prepared_opening() {
    let (elf_bytes, input) = a_run_that_touches_memory();
    let opts = ProofOptions::default_test_options();
    let (elf, boundaries, init_page_data, page_bases, num_private) =
        cross_epoch_inputs(&elf_bytes, &input, 2);

    let configs = continuation::global_memory_configs(&page_bases, &elf, num_private);
    let plan = crate::continuation::genesis_stack_plan(
        &configs,
        boundaries.len(),
        crate::continuation::PAGE_NUM_VARS,
    );
    let worst = plan.routes.iter().map(|r| r.nonzero).max().unwrap_or(0);
    // ⚠ THE SAVINGS ARE PART OF THE READING. Under the two-part rule an empty
    // plan has two causes — no page cleared part 1, or the candidates could not
    // pay for the chain — and "no opening" alone does not say which. The line
    // prints the candidate count and the savings so the log does.
    println!(
        "SPARSE FIXTURE: {} pages, worst nonzero {worst}, sparse rows {}, candidates {}, \
         savings {} against a {}-row chain",
        plan.routes.len(),
        plan.sparse_rows,
        plan.routes.iter().filter(|route| route.candidate).count(),
        plan.savings,
        crate::continuation::PREPARED_LEG_ROWS,
    );
    assert!(
        !crate::continuation::chain_is_paid(plan.savings),
        "part 2 must be what refuses here"
    );
    assert!(
        plan.is_empty(),
        "this fixture's genesis crossed the threshold, so it is no longer the \
         sparse control this test is"
    );

    let global = crate::with_whir_hash!(|H| {
        multilinear_continuation::prove_global::<H>(
            &boundaries,
            &elf_bytes,
            &init_page_data,
            &page_bases,
            num_private,
            &opts,
        )
    })
    .expect("prove");
    assert!(
        global.proof.preprocessed.is_none(),
        "a sparse page set produced a prepared opening it has no use for"
    );
}

/// The ELF's genesis byte at `address`, derived from the `PT_LOAD` segments
/// DIRECTLY — never through `build_initial_image_paged` or
/// `page::preprocessed_columns`.
///
/// ⚠ THE INDEPENDENCE IS THE WHOLE VALUE. The provenance test below compares
/// the interned root against a stack built from this; built through the same
/// function the code under test uses, it would compare a value with itself and
/// pass for any pair of agreeing bugs. A byte outside every segment is zero,
/// which is the same rule a short `PageConfig::init_values` encodes.
fn elf_genesis_byte(elf: &Elf, address: u64) -> u8 {
    for segment in &elf.data {
        let end = segment
            .base_addr
            .saturating_add(segment.values.len() as u64 * 4);
        if address < segment.base_addr || address >= end {
            continue;
        }
        let offset = address - segment.base_addr;
        let word = segment.values[(offset / 4) as usize];
        // RISC-V is little-endian: byte `k` of a word is bits `8k..8k+8`.
        return (word >> (8 * (offset % 4))) as u8;
    }
    0
}

/// ⛔⛔ THE FOURTH OWED PER-ELF PIN, AND ITS PROVENANCE.
///
/// The in-guest verifier cannot recompute the genesis stack's commitment, so it
/// interns the ROOT as program text exactly as it interns DECODE's. The
/// statement owed out of band is **"this root is the commitment to the ELF's
/// genesis bytes at the dense page bases, under this blowup and folding"**, and
/// it is OWED rather than covered: nothing inside the program checks it.
///
/// This is the evidence for that statement. It rebuilds the stacked columns
/// from the ELF's `PT_LOAD` segments byte by byte — a second derivation on
/// purpose, see [`elf_genesis_byte`] — commits them through the same
/// `global_layout` and the same config the production path uses, and compares
/// the ROOTS.
///
/// ⛔ AND THESE ARE NOT THE 35 UNIVARIATE ROOTS. `recursion::precomputed_commitments`
/// builds one Merkle root per page config over that page's LDE codeword; those
/// are what the attestation's `program_id` folds, and no multilinear verifier
/// ever compares one. A stacked WHIR commitment over the same columns has no
/// per-page subtree to match against them. Two different objects over the same
/// bytes, and only one of them is this.
#[test]
fn the_interned_genesis_root_is_the_elfs_own_bytes_at_the_dense_pages() {
    use multilinear::whir_hash::KeccakWhir;

    let elf_bytes = asm_elf_bytes("dense_data_page_touch");
    let (elf, boundaries, _init_page_data, page_bases, num_private) =
        cross_epoch_inputs(&elf_bytes, &[], 3);

    let configs = continuation::global_memory_configs(&page_bases, &elf, num_private);
    let plan = crate::continuation::genesis_stack_plan(
        &configs,
        boundaries.len(),
        crate::continuation::PAGE_NUM_VARS,
    );
    assert!(
        !plan.is_empty(),
        "the fixture must carry a stack, or there is no root to have provenance"
    );
    // ⚠ AND WHY it carries one: part 1 made it a candidate and part 2 paid for
    // the chain. An absence here would otherwise be consistent with either.
    println!(
        "PROVENANCE PLAN: n_fixed {} savings {} chain {} candidates {}",
        plan.n_fixed,
        plan.savings,
        crate::continuation::PREPARED_LEG_ROWS,
        plan.routes.iter().filter(|route| route.candidate).count(),
    );
    assert!(crate::continuation::chain_is_paid(plan.savings));

    // Both halves commit under ONE config, so a difference in the roots is a
    // difference in the BYTES and not in the parameters.
    let config = crate::multilinear_prove::chain_config(&[(
        plan.at.len(),
        crate::continuation::PAGE_NUM_VARS,
    )]);
    let production = multilinear_continuation::genesis_prepared_for::<KeccakWhir>(
        &configs,
        plan.clone(),
        &config,
    )
    .expect("the production stack")
    .expect("a non-empty plan must produce a stack");

    // The independent half: each stacked column rebuilt from ITS OWN closed
    // form, selected by `entry.column`.
    //
    // ⛔⛔ THIS IS WHERE THIS TEST WAS WRONG, AND THE SHAPE IS WORTH MORE THAN
    // THE BUG. Written before the both-columns ruling, it read only
    // `entry.table` and rebuilt EVERY entry from the ELF's bytes — so once a
    // dense page began stacking `[OFFSET, INIT]` it committed `[INIT, INIT]`
    // against the production stack and reported a mismatch it had manufactured
    // itself. A second derivation that ignores part of what it is deriving is
    // not an independent check; it is a different object.
    //
    // ⚠ NEITHER ARM CALLS `page::preprocessed_columns` OR `page::offset_column`.
    // The ramp is written out here. Built through the function under test, the
    // two halves would agree for any pair of agreeing bugs — the same reason
    // [`elf_genesis_byte`] exists.
    type Fe = math::field::element::FieldElement<crate::test_utils::F>;
    let page_size = 1u64 << crate::continuation::PAGE_NUM_VARS;
    let rebuild = |entry: &stark::multilinear_table::PreparedColumn| -> Vec<Fe> {
        let base = configs[entry.table - boundaries.len()].page_base;
        match entry.column {
            // OFFSET: the row index. The same column for every page of this
            // size, independent of the program entirely.
            //
            // ⚠ It reads like `page::offset_column()` because `0..page_size`
            // has one spelling — but it is not a CALL to it, and that is the
            // whole of the independence here: a ramp that started at 1, or ran
            // to `page_size` inclusive, would redden this while agreeing with
            // itself everywhere else.
            0 => (0..page_size).map(Fe::from).collect(),
            // INIT: this page's genesis bytes, out of the ELF's own segments.
            1 => (0..page_size)
                .map(|offset| Fe::from(u64::from(elf_genesis_byte(&elf, base + offset))))
                .collect(),
            other => panic!(
                "the stack names preprocessed column {other} of table {}, and a genesis \
                 page presents {}. There is no independent derivation for it here, and \
                 rebuilding an unknown column as INIT is exactly the defect this test \
                 carried: it would compare a stack of the wrong columns and blame the \
                 root.",
                entry.table,
                crate::continuation::PAGE_PREPROCESSED_COLUMNS
            ),
        }
    };
    let rebuilt: Vec<multilinear::mle::Mle<crate::test_utils::F>> = plan
        .at
        .iter()
        .map(|entry| multilinear::mle::Mle::new(rebuild(entry)).expect("mle"))
        .collect();

    // ⚠ ANTI-VACUITY, PER COLUMN KIND — two all-zero stacks match and say
    // nothing. Each kind is asserted against the count only IT can have:
    //
    // - INIT must clear the fill this guest was BUILT with: `dense_data_page_
    //   touch` surrounds its touched cell with 32 KiB of non-zero bytes on each
    //   side, so the counter's own page holds at least one side's worth
    //   whatever offset the linker chose. ⛔ THE FLOOR IS THE GUEST'S, NOT THE
    //   ROUTING RULE'S: a page can be carried while holding far fewer entries
    //   than that (two pages of 5,000 share a chain), so a floor taken from the
    //   threshold would be true here only by accident of this fixture.
    // - OFFSET is the ramp `0..page_size`, so exactly one entry of it is zero
    //   and its count is `page_size - 1`: pinned EXACTLY, not by a floor.
    //   ⛔ This is the reading that names the old defect outright. An OFFSET
    //   entry rebuilt from the ELF's bytes prints the INIT count instead of
    //   262,143 — which is what the failing log showed, twice.
    let floor = 32_768usize;
    let ramp_nonzero = page_size as usize - 1;
    for (column, entry) in rebuilt.iter().zip(&plan.at) {
        let zero = Fe::from(0u64);
        let nonzero = column.evals().iter().filter(|v| **v != zero).count();
        println!(
            "PROVENANCE table {} column {} rebuilt nonzero {nonzero} (INIT floor {floor}, \
             OFFSET exactly {ramp_nonzero})",
            entry.table, entry.column
        );
        if entry.column == 0 {
            assert_eq!(
                nonzero, ramp_nonzero,
                "stack column {} of table {} is this page's OFFSET ramp, whose only zero \
                 is row 0 — a count of {nonzero} means it was rebuilt as something else",
                entry.column, entry.table
            );
        } else {
            assert!(
                nonzero > floor,
                "the rebuilt INIT column has {nonzero} nonzero entries, at or below the \
                 {floor} that put this page in the stack — the two halves are reading \
                 different pages"
            );
        }
    }

    let commit_independently = |columns: &[multilinear::mle::Mle<crate::test_utils::F>]| {
        multilinear::stacked_eval::StackedCommitment::<crate::test_utils::F, KeccakWhir>::commit(
            stark::multilinear_table::global_layout(&[(
                columns.len(),
                crate::continuation::PAGE_NUM_VARS,
            )])
            .expect("layout"),
            &multilinear::stacking::borrow(columns),
            None,
            &config,
        )
        .expect("the independent stack")
    };

    // THE HONEST CONTROL, FIRST: the two derivations agree.
    let independent = commit_independently(&rebuilt);
    assert_eq!(
        production.roots,
        independent.roots(),
        "the genesis stack's root is not the commitment to the ELF's own bytes at those \
         page bases — the pin this root owes could not be stated"
    );

    // ⛔ AND THE COMPARISON IS EXECUTED ON A STATE IT MUST REFUSE. One that has
    // only ever run on agreeing inputs is one nobody has seen work. A single
    // byte of ONE rebuilt INIT column is moved and the same commitment taken
    // again; the roots must then differ. INIT and not OFFSET on purpose: INIT
    // is the column this opening exists to settle.
    let init_at = plan
        .at
        .iter()
        .position(|entry| entry.column == 1)
        .expect("a dense page stacks an INIT column");
    let mut moved: Vec<Vec<Fe>> = rebuilt
        .iter()
        .map(|column| column.evals().to_vec())
        .collect();
    let zero = Fe::from(0u64);
    let byte = moved[init_at]
        .iter()
        .position(|value| *value != zero)
        .expect("a dense INIT column has a nonzero byte to move");
    moved[init_at][byte] += Fe::from(1u64);
    let moved: Vec<multilinear::mle::Mle<crate::test_utils::F>> = moved
        .into_iter()
        .map(|values| multilinear::mle::Mle::new(values).expect("mle"))
        .collect();
    assert_ne!(
        production.roots,
        commit_independently(&moved).roots(),
        "one genesis byte was moved in stack column {init_at} at offset {byte} and the \
         root did not change: this comparison cannot see a wrong stack, so its green says \
         nothing"
    );

    // ⛔⛔ THE FOURTH OWED PIN, NOW STATEABLE. The statement owed out of band is
    // the line below: this root, over this guest named by sha, this many
    // columns at this height, under this hash. Nothing inside the program
    // checks it. The BLOCK's own root is read by
    // `the_blocks_dense_pages_are_the_three_the_threshold_pre_registers` on the
    // box; this is the fixture-scale half of the same pin.
    println!(
        "GENESIS STACK ROOT {:02x?} guest dense_data_page_touch sha {} ({} bytes)  \
         columns {} at {} variables  hash {}",
        production.roots,
        sha256_hex(&elf_bytes),
        elf_bytes.len(),
        plan.at.len(),
        crate::continuation::PAGE_NUM_VARS,
        <KeccakWhir as multilinear::whir_hash::WhirHash>::NAME,
    );
}

/// The bench ELF by name, or `None` when it is simply not built here.
fn bench_elf_if_present(name: &str) -> Option<Vec<u8>> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("executor/program_artifacts");
    for dir in ["rust", "asm"] {
        if let Ok(bytes) = std::fs::read(root.join(dir).join(format!("{name}.elf"))) {
            return Some(bytes);
        }
    }
    None
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// ⛔⛔ THE PRE-REGISTRATION, READ OFF THE REAL ELF INSTEAD OF COPIED.
///
/// The split's unit tests in `continuation` encode the block's census — 116,692 nonzero
/// entries at `0x0`, 229,290 at `0x40000`, 223,380 at `0x280000`, zero at the
/// other 27 — as numbers I typed from a box log. That is enough to check the
/// closed form's ARITHMETIC and not enough to check that the form, run against
/// the actual program, selects those three pages. A routing rule whose
/// pre-registration rests on a transcription is a rule nobody has tested.
///
/// This runs the block guest once — no proving, no card, the same shape as
/// `whir_global_tests::the_block_genesis_census` — rebuilds the page configs
/// from the ELF, evaluates [`crate::continuation::genesis_stack_plan`] on them, and asserts
/// the dense set is EXACTLY those three bases.
///
/// ⛔ IT REFUSES RATHER THAN SKIPS when the shas are unstated, and SKIPS with
/// its own line when the ELF is simply absent. Two outcomes, two meanings: a
/// census quoted as a fact about one program and one input must not be
/// producible from an unnamed pair, and a missing build product is not a
/// failure of this check.
#[test]
#[ignore = "the box runs it: one execution of the block guest, no proving"]
fn the_blocks_dense_pages_are_the_three_the_threshold_pre_registers() {
    // What the ruling names, and what this arm exists to confirm against the ELF.
    const PRE_REGISTERED: [u64; 3] = [0x0, 0x40000, 0x280000];

    let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
    let input_name = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(21);

    let Some(elf_bytes) = bench_elf_if_present(&name) else {
        println!(
            "GENESIS-ROUTING SKIPPED - no ELF named {name} in \
             executor/program_artifacts/{{rust,asm}}; set LAMBDA_VM_BENCH_ELF"
        );
        return;
    };
    let input = crate::tests::multilinear_bench_tests::input_bytes(&input_name);

    let elf_sha = sha256_hex(&elf_bytes);
    let input_sha = sha256_hex(&input);
    let want_elf = std::env::var("LAMBDA_VM_CENSUS_ELF_SHA256").unwrap_or_else(|_| {
        panic!(
            "this routing is quoted as a fact about ONE program and ONE input, so it \
             refuses to run unnamed. Set LAMBDA_VM_CENSUS_ELF_SHA256={elf_sha} and \
             LAMBDA_VM_CENSUS_INPUT_SHA256={input_sha}"
        )
    });
    let want_input = std::env::var("LAMBDA_VM_CENSUS_INPUT_SHA256").unwrap_or_else(|_| {
        panic!("LAMBDA_VM_CENSUS_INPUT_SHA256 is unset; the input here is {input_sha}")
    });
    assert_eq!(
        elf_sha, want_elf,
        "the ELF is not the one this routing was asked for"
    );
    assert_eq!(
        input_sha, want_input,
        "the INPUT is not the one this routing was asked for, and the touched page \
         list is a function of it"
    );
    println!(
        "GENESIS-ROUTING elf {name} sha {elf_sha} ({} bytes)  input {} sha {input_sha} \
         ({} bytes)  epoch 2^{epoch_size_log2}",
        elf_bytes.len(),
        if input_name.is_empty() {
            "<none>"
        } else {
            &input_name
        },
        input.len(),
    );

    let started = std::time::Instant::now();
    let pages = continuation::block_page_census(&elf_bytes, &input, epoch_size_log2)
        .expect("the guest runs to completion");
    let elf = Elf::load(&elf_bytes).expect("load");
    let configs = continuation::global_memory_configs(
        &pages.touched_page_bases,
        &elf,
        pages.num_private_input_pages,
    );
    let plan = crate::continuation::genesis_stack_plan(
        &configs,
        pages.num_epochs,
        crate::continuation::PAGE_NUM_VARS,
    );
    println!(
        "EXECUTION: {} epochs, {} touched pages, {} private, in {:.1}s",
        pages.num_epochs,
        pages.touched_page_bases.len(),
        pages.num_private_input_pages,
        started.elapsed().as_secs_f64(),
    );

    let mut sparse_total = 0usize;
    let mut dense_total = 0usize;
    for route in &plan.routes {
        let rows =
            crate::continuation::sparse_leg_rows(crate::continuation::PAGE_NUM_VARS, route.nonzero);
        if route.dense {
            dense_total += rows;
        } else if route.has_init {
            sparse_total += rows;
        }
        println!(
            "ROUTE {:#x}: nonzero {} has_init {} candidate {} dense {} sparse_rows {rows}",
            route.page_base, route.nonzero, route.has_init, route.candidate, route.dense
        );
    }

    let dense_bases: Vec<u64> = plan
        .routes
        .iter()
        .filter(|r| r.dense)
        .map(|r| r.page_base)
        .collect();
    // ⚠ BOTH PARTS, AND BOTH OF THEIR INPUTS. A line quoting only the answer
    // would not say which rule produced it, and the two are separately wrong in
    // different ways.
    let num_vars = crate::continuation::PAGE_NUM_VARS;
    let marginal = crate::continuation::marginal_stacked_rows(num_vars, plan.n_fixed);
    println!(
        "GENESIS ROUTING: {} of {} pages stacked {dense_bases:02x?}; the sparse form \
         would have cost {dense_total} rows for them and costs {sparse_total} for the \
         rest. PART 1 at n_fixed {} ({} genesis pages): marginal {marginal} rows, so a \
         candidate needs {} nonzero entries; {} candidates. PART 2: savings {} against a \
         {}-row chain, {}",
        dense_bases.len(),
        plan.routes.len(),
        plan.n_fixed,
        plan.routes.iter().filter(|r| r.has_init).count(),
        crate::continuation::candidate_threshold_entries(num_vars, plan.n_fixed),
        plan.routes.iter().filter(|r| r.candidate).count(),
        plan.savings,
        crate::continuation::PREPARED_LEG_ROWS,
        if crate::continuation::chain_is_paid(plan.savings) {
            "PAID"
        } else {
            "REFUSED — every candidate stays sparse"
        },
    );

    // ⚠ THE ASSERTION IS THE SET AND ITS ORDER, not a count. Three pages of the
    // wrong three would pass a count, and the stack's column order IS page-base
    // order, so the order is part of what the opening means.
    assert_eq!(
        dense_bases,
        PRE_REGISTERED.to_vec(),
        "the threshold selected a different set of pages than the ruling names"
    );
    // The two parts' own pre-registrations, so a green here cannot come from
    // the right set reached by the wrong arithmetic.
    assert_eq!(
        plan.n_fixed, 24,
        "thirty genesis pages: 18 + ceil(log2(60))"
    );
    assert_eq!(
        marginal, 103,
        "one eq and its join, two indicators, one Sub"
    );
    assert_eq!(
        plan.routes.iter().filter(|r| r.candidate).count(),
        PRE_REGISTERED.len(),
        "the 27 all-zero pages must fail PART 1: 18 sparse rows against {marginal}"
    );
    assert_eq!(plan.savings, 10_248_261);
    assert!(crate::continuation::chain_is_paid(plan.savings));
    // And the pages left behind must be genuinely cheap, or the hybrid is not
    // the win the ruling claimed.
    assert!(
        sparse_total < dense_total / 100,
        "the sparse remainder is {sparse_total} rows against {dense_total} stacked: \
         the split is not the concentration the census read"
    );

    // ⛔⛔ THE FOURTH OWED PIN, AT THE BLOCK. The in-guest verifier interns this
    // root as program text and nothing inside the program checks it; the
    // statement owed out of band is this line. `a_dense_genesis_page_is_carried
    // _by_a_prepared_opening`'s sibling states the same thing at fixture scale,
    // and `the_interned_genesis_root_is_the_elfs_own_bytes_at_the_dense_pages`
    // is where the derivation is checked against the ELF's own bytes.
    let config = crate::multilinear_prove::chain_config(&[(plan.at.len(), num_vars)]);
    crate::with_whir_hash!(|H| {
        let prepared =
            multilinear_continuation::genesis_prepared_for::<H>(&configs, plan.clone(), &config)
                .expect("the block's genesis stack")
                .expect("three dense pages must produce a stack");
        println!(
            "GENESIS STACK ROOT {:02x?} elf {elf_sha} input {input_sha} columns {} at \
             {num_vars} variables hash {}",
            prepared.roots,
            plan.at.len(),
            <H as multilinear::whir_hash::WhirHash>::NAME,
        );
    });
}

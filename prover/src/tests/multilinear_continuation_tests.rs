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

/// The binding is what makes the two halves one proof: swapping an epoch's
/// bookend root has to be caught even though both halves still verify on their
/// own.
#[test]
fn a_bookend_that_is_not_the_one_chained_is_rejected() {
    let (elf_bytes, input) = a_run_that_touches_memory();
    let opts = ProofOptions::default_test_options();
    let mut bundle =
        multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts).expect("prove");
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
    bundle.touched_page_bases.pop();
    assert!(
        matches!(
            multilinear_continuation::verify_continuation(&elf_bytes, &bundle, &opts),
            Ok(false) | Err(_)
        ),
        "a restated touched page set was accepted"
    );
}

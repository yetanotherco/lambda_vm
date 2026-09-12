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

    let boundaries =
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

            let proof = multilinear_continuation::prove_epoch(
                &elf,
                &elf_bytes,
                &register_init,
                label,
                traces,
                is_final,
                &boundary,
                &opts,
                None,
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
#[test]
fn the_epochs_chain_through_their_registers() {
    let few = epochs_prove_and_verify("sub", 6);
    let many = epochs_prove_and_verify("sub", 4);
    assert!(
        many >= few,
        "a smaller epoch should not produce fewer of them: {many} against {few}"
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
        multilinear_continuation::prove_epochs(&elf_bytes, &[], 4, &opts).expect("prove");
    if epochs.len() < 2 {
        return; // nothing to chain
    }
    // Claim the first epoch ended somewhere it did not.
    epochs[0].reg_fini[1] ^= 1;
    assert!(
        !multilinear_continuation::verify_epochs(&elf_bytes, &epochs, &opts).expect("verify"),
        "a restated register carry was accepted"
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
    continuation::for_each_epoch(&elf, &[], 4, &artifacts, |prepared, _| {
        let boundary = std::sync::Arc::clone(&prepared.boundary);
        let register_init = prepared.register_init.clone();
        let label = prepared.label;
        let is_final = prepared.is_final;
        let proof = multilinear_continuation::prove_epoch(
            &elf,
            &elf_bytes,
            &register_init,
            label,
            prepared.traces,
            is_final,
            &boundary,
            &opts,
            None,
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
        let (layouts, _) =
            crate::multilinear_prove::stacks(&widths, &sizes, &config).expect("the epoch's stacks");
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

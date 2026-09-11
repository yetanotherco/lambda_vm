//! One continuation epoch, proved and verified through the multilinear path.
//!
//! The epoch split, the local-to-global bookend and the cross-epoch register
//! carry are [`crate::continuation`]'s and do not depend on the commitment
//! scheme; what these check is that an epoch's AIRs — which differ from the
//! monolithic ones, PAGE replaced by the bookend and REGISTER preprocessing
//! both ends — argue correctly under WHIR.
//!
//! The cross-epoch global-memory proof is not here yet, so these do not prove a
//! *continuation*: they prove every epoch of one, and check that the register
//! carry the epochs are chained by is the one each proof binds.

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

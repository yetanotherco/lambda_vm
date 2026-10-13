//! PoC: the HALT continue-past-exit forgery (see `halt_pc_forgery_poc`) against a
//! continuation proof.
//!
//! A continuation epoch preprocesses FINI against the bundle's `reg_fini`, which
//! the prover supplies. The verifier reuses it as the next epoch's INIT, but the
//! final epoch has no next epoch, so `verify_continuation` pins the final
//! `reg_fini` to the halted register file (`register::halted_fini`). Without that
//! check the final epoch could end its PC chain anywhere.
//!
//! Declared as a child of `continuation` so it can drive the private epoch/global
//! provers directly: a single (final) epoch, built sequentially from forged logs.

use std::collections::HashMap;
use std::sync::Arc;

use executor::elf::Elf;
use executor::vm::logs::Log;

use super::*;
use crate::test_utils::asm_elf_bytes;
use crate::tests::halt_pc_forgery_poc::{
    FULL_ATTACK, GUEST, Tamper, apply_tamper, forged_logs, opts, patched_elf, run_logs,
};

/// A one-epoch continuation bundle for `logs`, built like `prove_continuation`
/// but with `tamper` applied to the epoch's traces before proving. `reg_fini` is
/// read from the (tampered) REGISTER trace, exactly as `prove_epoch` does.
fn craft_bundle(
    elf_bytes: &[u8],
    logs: &[Log],
    tamper: Tamper,
) -> Result<ContinuationProof, Error> {
    let opts = opts();
    let elf = Elf::load(elf_bytes).expect("ELF load");
    let decode_commitment =
        crate::tables::decode::commitment_from_elf(&elf, &opts).expect("DECODE commitment");
    let decode_artifacts = DecodeArtifacts::from_elf(&elf)?;
    let image = build_initial_image_paged(&elf, &[]);
    let init_page_data = build_init_page_data(&image);
    let mut provenance =
        local_to_global::genesis_provenance(image.iter().map(|(a, v)| (a, v as u64)));

    let register_init = register::register_init_from_entry_point(elf.entry_point);
    let label = local_to_global::epoch_label(0);
    let collected = Traces::collect_epoch(&decode_artifacts, &image, &register_init, logs, true)?;
    let boundary = Arc::new(local_to_global::epoch_boundary(
        &mut provenance,
        label,
        &collected.touched_memory_cells(),
    ));
    let mut traces = Traces::build_from_collected(
        &decode_artifacts,
        collected,
        None::<&HashMap<u64, u8>>,
        &register_init,
        &MaxRowsConfig::default(),
        &[],
        true,
        true,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )?;
    apply_tamper(&mut traces, logs, tamper);

    let epoch = prove_epoch(
        &elf,
        elf_bytes,
        &EpochStart {
            register_init: &register_init,
            label,
        },
        traces,
        true,
        &boundary,
        &opts,
        decode_commitment,
    )?;

    let boundaries = vec![boundary];
    let touched = touched_page_bases(&boundaries);
    let num_private_input_pages = page::private_input_page_count(&[]);
    let global = prove_global(
        &boundaries,
        elf_bytes,
        &init_page_data,
        &touched,
        num_private_input_pages,
        &opts,
    )?;

    Ok(ContinuationProof {
        epochs: vec![epoch],
        global,
        num_private_input_pages,
        touched_page_bases: touched,
    })
}

/// A prover-side error counts as "not accepted".
fn accepted(crafted: Result<ContinuationProof, Error>, elf: &[u8]) -> bool {
    match crafted {
        Ok(bundle) => verify_continuation(elf, &bundle, &opts())
            .expect("verifier must not error")
            .is_some(),
        Err(_) => false,
    }
}

/// POSITIVE CONTROL: the hand-built one-epoch bundle of an honest run verifies.
#[test]
fn poc_control_patched_program_continuation_verifies() {
    let patched = patched_elf(&asm_elf_bytes(GUEST));
    let logs = run_logs(&patched);
    assert!(accepted(
        craft_bundle(&patched, &logs, Tamper::default()),
        &patched
    ));
}

/// NEGATIVE CONTROL: `exit(1)` honestly is rejected.
#[test]
fn poc_control_honest_exit_one_continuation_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let logs = run_logs(&elf);
    assert!(!accepted(
        craft_bundle(&elf, &logs, Tamper::default()),
        &elf
    ));
}

/// THE BUG, continuation edition: HALT `PC = (1, 0)`, run past `exit(1)`, zero
/// x10, and end the PC chain on the last post-exit `next_pc` through the final
/// epoch's prover-supplied `reg_fini`. Must be rejected.
#[test]
fn continuation_continue_past_exit_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let logs = forged_logs(&elf);
    assert!(
        !accepted(craft_bundle(&elf, &logs, FULL_ATTACK), &elf),
        "SOUNDNESS BUG: the continuation verifier accepted a final epoch that runs \
         past the exit ecall and turns exit(1) into exit(0)"
    );
}

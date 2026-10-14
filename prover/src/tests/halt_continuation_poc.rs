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
//! provers directly: epochs built sequentially from forged logs, as
//! `prove_continuation` builds them.

use std::collections::HashMap;
use std::sync::Arc;

use executor::elf::Elf;
use executor::vm::logs::Log;

use super::*;
use crate::test_utils::asm_elf_bytes;
use crate::tests::halt_pc_forgery_poc::{
    FORGED_ROWS, FULL_ATTACK, GUEST, Tamper, apply_tamper, forged_logs, opts, patched_elf, run_logs,
};

/// Rows of the guest before its exit ecall: a power of two, so they fill a
/// non-final epoch on their own and the exit lands in the next one.
const PRE_EXIT_ROWS: usize = 4;

/// A continuation bundle for `logs`, built like `prove_continuation`: one epoch per
/// entry of `epoch_lens` (non-final epochs a power of two), with `tamper` applied
/// to the final epoch's traces before proving. Each `reg_fini` is read from the
/// (tampered) REGISTER trace, exactly as `prove_epoch` does.
fn craft_bundle(
    elf_bytes: &[u8],
    logs: &[Log],
    epoch_lens: &[usize],
    tamper: Tamper,
) -> Result<ContinuationProof, Error> {
    assert_eq!(epoch_lens.iter().sum::<usize>(), logs.len());
    let opts = opts();
    let elf = Elf::load(elf_bytes).expect("ELF load");
    let decode_commitment =
        crate::tables::decode::commitment_from_elf(&elf, &opts).expect("DECODE commitment");
    let decode_artifacts = DecodeArtifacts::from_elf(&elf)?;
    let mut image = build_initial_image_paged(&elf, &[]);
    let init_page_data = build_init_page_data(&image);
    let mut provenance =
        local_to_global::genesis_provenance(image.iter().map(|(a, v)| (a, v as u64)));

    let mut register_init = register::register_init_from_entry_point(elf.entry_point);
    let mut epochs = Vec::with_capacity(epoch_lens.len());
    let mut boundaries = Vec::with_capacity(epoch_lens.len());
    let mut start = 0;
    for (index, &len) in epoch_lens.iter().enumerate() {
        let is_final = index == epoch_lens.len() - 1;
        let epoch_logs = &logs[start..start + len];
        start += len;

        let label = local_to_global::epoch_label(index as u64);
        let collected = Traces::collect_epoch(
            &decode_artifacts,
            &image,
            &register_init,
            epoch_logs,
            is_final,
        )?;
        let boundary = Arc::new(local_to_global::epoch_boundary(
            &mut provenance,
            label,
            &collected.touched_memory_cells(),
        ));
        let next_register_init = collected.register_fini(&register_init);
        for cell in boundary.iter() {
            image.set(cell.address, (cell.fini.value & 0xFF) as u8);
        }

        let mut traces = Traces::build_from_collected(
            &decode_artifacts,
            collected,
            None::<&HashMap<u64, u8>>,
            &register_init,
            &MaxRowsConfig::default(),
            &[],
            is_final,
            true,
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        )?;
        if is_final {
            apply_tamper(&mut traces, epoch_logs, tamper);
        }

        epochs.push(prove_epoch(
            &elf,
            elf_bytes,
            &EpochStart {
                register_init: &register_init,
                label,
            },
            traces,
            is_final,
            &boundary,
            &opts,
            decode_commitment,
        )?);
        boundaries.push(boundary);
        register_init = next_register_init;
    }

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
        epochs,
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

/// The forged bundle must build, so the rejection comes from the verifier and not
/// from the prover refusing the trace.
fn assert_verifier_rejects(crafted: Result<ContinuationProof, Error>, elf: &[u8], msg: &str) {
    assert!(crafted.is_ok(), "the forged bundle must build");
    assert!(!accepted(crafted, elf), "{msg}");
}

/// POSITIVE CONTROL: the hand-built one-epoch bundle of an honest run verifies.
#[test]
fn poc_control_patched_program_continuation_verifies() {
    let patched = patched_elf(&asm_elf_bytes(GUEST));
    let logs = run_logs(&patched);
    assert!(accepted(
        craft_bundle(&patched, &logs, &[logs.len()], Tamper::default()),
        &patched
    ));
}

/// POSITIVE CONTROL: the two-epoch split the forgery below uses verifies for an
/// honest run.
#[test]
fn poc_control_patched_program_two_epochs_verifies() {
    let patched = patched_elf(&asm_elf_bytes(GUEST));
    let logs = run_logs(&patched);
    assert!(accepted(
        craft_bundle(
            &patched,
            &logs,
            &[PRE_EXIT_ROWS, logs.len() - PRE_EXIT_ROWS],
            Tamper::default()
        ),
        &patched
    ));
}

/// NEGATIVE CONTROL: `exit(1)` honestly is rejected.
#[test]
fn poc_control_honest_exit_one_continuation_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let logs = run_logs(&elf);
    assert!(!accepted(
        craft_bundle(&elf, &logs, &[logs.len()], Tamper::default()),
        &elf
    ));
}

/// THE BUG, continuation edition: HALT `PC = (1, 0)`, run past `exit(1)`, zero
/// x10, and end the PC chain on the last post-exit `next_pc` through the final
/// epoch's prover-supplied `reg_fini`. Must be rejected.
#[test]
fn continuation_continue_past_exit_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let logs = forged_logs(&elf, FORGED_ROWS);
    assert_verifier_rejects(
        craft_bundle(&elf, &logs, &[FORGED_ROWS], FULL_ATTACK),
        &elf,
        "SOUNDNESS BUG: the continuation verifier accepted a final epoch that runs \
         past the exit ecall and turns exit(1) into exit(0)",
    );
}

/// THE BUG in the final epoch of a two-epoch bundle: an honest first epoch, then
/// the exit ecall and the post-exit rows in the final one, whose INIT is the
/// first epoch's `reg_fini`. Must be rejected like the one-epoch case.
#[test]
fn continuation_continue_past_exit_in_final_of_two_epochs_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let logs = forged_logs(&elf, PRE_EXIT_ROWS + FORGED_ROWS);
    assert_eq!(
        logs[PRE_EXIT_ROWS].src1_val, 93,
        "the final epoch starts on the exit ecall"
    );
    assert_verifier_rejects(
        craft_bundle(&elf, &logs, &[PRE_EXIT_ROWS, FORGED_ROWS], FULL_ATTACK),
        &elf,
        "SOUNDNESS BUG: the continuation verifier accepted a final epoch (after an \
         honest one) that runs past the exit ecall",
    );
}

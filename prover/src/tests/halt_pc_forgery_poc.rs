//! PoC: execution can continue past the exit ecall.
//!
//! HALT consumes the halting row's `next_pc` token and re-emits `pc = 1` for the
//! padding chain, both at `halt_ts + 1`. Its `PC_0/PC_1` columns are not tied to
//! anything, so a prover can set them to `(1, 0)`: HALT then consumes the `1` it
//! emits itself, and the real `next_pc` token is left for another CPU row to pick
//! up. The trace keeps executing after `exit` and HALT's x10 == 0 check reads x10
//! at `ts = 2^64 - 1`, not at the exit call.
//!
//! The guest calls `exit(1)`, which no honest prover can prove. The forgery runs
//! `li a0, 0` (+ nops) after the exit and must not verify against the same ELF.
//!
//! Production proof options, as in `page_offset_forgery_poc`.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};

use crate::statement::{StatementKind, absorb_statement};
use crate::tables::halt::cols as halt_cols;
use crate::tables::register::{self, cols as reg_cols};
use crate::tables::trace_builder::Traces;
use crate::tables::types::{FE, VmTable};
use crate::test_utils::{E, asm_elf_bytes};
use crate::{MaxRowsConfig, VmAirs, VmProof};

pub(crate) const GUEST: &str = "poc_halt_continue";

/// `li a7, 93; ecall` — the first occurrence is the `exit(1)` call.
const EXIT_SEQ: [u8; 8] = [0x93, 0x08, 0xd0, 0x05, 0x73, 0x00, 0x00, 0x00];
/// `addi x0, x0, 0`.
const NOP: [u8; 4] = [0x13, 0x00, 0x00, 0x00];

/// Rows the forged trace has: a power of two, so the CPU table gets no padding
/// rows and the last real row's `next_pc` goes straight to the REGISTER table.
const FORGED_ROWS: usize = 16;

pub(crate) fn opts() -> ProofOptions {
    crate::GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

/// The guest with its first `ecall` replaced by a nop, so the executor runs the
/// post-exit rows and produces their logs.
pub(crate) fn patched_elf(honest: &[u8]) -> Vec<u8> {
    let at = honest
        .windows(EXIT_SEQ.len())
        .position(|w| w == EXIT_SEQ)
        .expect("exit sequence not found");
    let mut patched = honest.to_vec();
    patched[at + 4..at + 8].copy_from_slice(&NOP);
    patched
}

pub(crate) fn run_logs(elf_bytes: &[u8]) -> Vec<Log> {
    let elf = Elf::load(elf_bytes).expect("ELF load");
    Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("run")
        .logs
}

/// The honest logs up to and including `exit(1)`, followed by the post-exit rows
/// taken from the patched run, cut at `FORGED_ROWS` (before the second ecall).
pub(crate) fn forged_logs(honest: &[u8]) -> Vec<Log> {
    let honest_logs = run_logs(honest);
    let exit_row = honest_logs.len() - 1;
    assert_eq!(
        honest_logs[exit_row].src1_val, 93,
        "honest run ends on exit"
    );

    let patched_logs = run_logs(&patched_elf(honest));
    let mut logs = honest_logs;
    logs.extend_from_slice(&patched_logs[exit_row + 1..FORGED_ROWS]);
    assert_eq!(logs.len(), FORGED_ROWS);
    assert_eq!(
        logs[exit_row + 1].current_pc,
        logs[exit_row].current_pc + 4,
        "first post-exit row runs the instruction after the ecall"
    );
    logs
}

/// How the malicious prover deviates from the honest traces.
#[derive(Clone, Copy, Default)]
pub(crate) struct Tamper {
    /// HALT `PC = (1, 0)`, so HALT's consume/emit cancel.
    pub(crate) halt_pc: bool,
    /// Move the REGISTER final x255 token to the last post-exit row's write
    /// timestamp. Main-trace columns.
    pub(crate) register_ts: bool,
    /// Set the REGISTER final x255 value to the last post-exit `next_pc`. A
    /// main-trace column on `main`; preprocessed (pinned to `pc = 1`) by the fix.
    pub(crate) register_fini: bool,
}

/// Every tamper of the attack: HALT PC plus the REGISTER final PC token.
pub(crate) const FULL_ATTACK: Tamper = Tamper {
    halt_pc: true,
    register_ts: true,
    register_fini: true,
};

/// Apply `tamper` to traces built from `logs` (the forged post-exit run).
pub(crate) fn apply_tamper(traces: &mut Traces, logs: &[Log], tamper: Tamper) {
    // Last real row writes next_pc at its timestamp + 1 (timestamps are 4i + 4).
    let last_ts = (logs.len() as u64) * 4;
    let last_next_pc = logs.last().unwrap().next_pc;
    let reg = &mut traces.register.main_table;
    let mut x255_rows = 0;
    for row in 0..reg.height {
        let word = match reg.get(row, reg_cols::OFFSET).to_raw() {
            510 => last_next_pc & 0xFFFF_FFFF,
            511 => last_next_pc >> 32,
            _ => continue,
        };
        if tamper.register_fini {
            reg.set(row, reg_cols::FINI, FE::from(word));
        }
        if tamper.register_ts {
            reg.set(row, reg_cols::TIMESTAMP_LO, FE::from(last_ts + 1));
            reg.set(row, reg_cols::TIMESTAMP_HI, FE::zero());
        }
        x255_rows += 1;
    }
    assert_eq!(x255_rows, 2, "x255 has two REGISTER rows");
    if tamper.halt_pc {
        traces.halt.main_table.set_dword_wl(0, halt_cols::PC_0, 1);
    }
}

fn craft_proof(
    elf_bytes: &[u8],
    logs: &[Log],
    tamper: Tamper,
) -> Result<VmProof, stark::prover::ProvingError> {
    let options = opts();
    let program = Elf::load(elf_bytes).expect("ELF load");
    let max_rows = MaxRowsConfig::default();
    let mut traces = Traces::from_elf_and_logs(
        &program,
        logs,
        &max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("trace build");

    apply_tamper(&mut traces, logs, tamper);

    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &program,
        &options,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        // As the production prover: FINI pinned to the halted register file.
        Some(register::halted_preprocessed(
            &options,
            program.entry_point,
            u32::try_from(traces.public_output_bytes.len()).unwrap(),
        )),
    );
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::Monolithic,
        elf_bytes,
        &traces.public_output_bytes,
        &table_counts,
        num_private_input_pages,
        &runtime_page_ranges,
        options.fri_final_poly_log_degree,
    );

    let proof = Prover::multi_prove(
        airs.air_trace_pairs(&mut traces),
        &mut transcript,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )?;

    Ok(VmProof {
        proof,
        runtime_page_ranges,
        table_counts,
        public_output: traces.public_output_bytes.clone(),
        num_private_input_pages,
    })
}

/// A prover-side error counts as "not accepted".
fn accepted(crafted: Result<VmProof, stark::prover::ProvingError>, elf: &[u8]) -> bool {
    match crafted {
        Ok(proof) => crate::verify_with_options(&proof, elf, &opts(), None, None)
            .expect("verifier must not error"),
        Err(_) => false,
    }
}

/// POSITIVE CONTROL: the harness itself produces valid proofs. The patched ELF
/// (no first ecall) exits 0 honestly at its second ecall.
#[test]
fn poc_control_patched_program_verifies() {
    let patched = patched_elf(&asm_elf_bytes(GUEST));
    let logs = run_logs(&patched);
    assert!(accepted(
        craft_proof(&patched, &logs, Tamper::default()),
        &patched
    ));
}

/// NEGATIVE CONTROL: `exit(1)` honestly is rejected (x10 != 0 at HALT).
#[test]
fn poc_control_honest_exit_one_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let logs = run_logs(&elf);
    assert!(!accepted(craft_proof(&elf, &logs, Tamper::default()), &elf));
}

/// NEGATIVE CONTROL: running past the exit while HALT's PC stays honest is
/// rejected — HALT consumes the real `next_pc` token, so the first post-exit
/// row has nothing to read. Isolates HALT's free PC column as the hole.
#[test]
fn poc_control_continue_without_halt_pc_tamper_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let logs = forged_logs(&elf);
    assert!(!accepted(
        craft_proof(
            &elf,
            &logs,
            Tamper {
                halt_pc: false,
                ..FULL_ATTACK
            }
        ),
        &elf
    ));
}

/// THE BUG: set HALT `PC = (1, 0)`, keep executing after `exit(1)`, zero x10,
/// stop. Must be rejected; if this test fails the verifier accepted a proof
/// that `exit(1)` exited with code 0.
#[test]
fn continue_past_exit_via_halt_pc_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let logs = forged_logs(&elf);
    assert!(
        !accepted(craft_proof(&elf, &logs, FULL_ATTACK), &elf),
        "SOUNDNESS BUG: the verifier accepted a trace that runs past the exit ecall \
         (HALT PC = (1, 0)) and turns exit(1) into exit(0)"
    );
}

/// THE BUG, respecting the fix's pinned FINI: HALT `PC = (1, 0)` and the final
/// x255 token moved to the end of the post-exit rows, but its value left at
/// `pc = 1`. Touches main-trace columns only, so the prover must build the
/// proof; the verifier must reject it because no real row writes `next_pc = 1`.
#[test]
fn continue_past_exit_with_pinned_fini_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let logs = forged_logs(&elf);
    let crafted = craft_proof(
        &elf,
        &logs,
        Tamper {
            register_fini: false,
            ..FULL_ATTACK
        },
    );
    assert!(crafted.is_ok(), "main-trace-only tamper must still prove");
    assert!(
        !accepted(crafted, &elf),
        "SOUNDNESS BUG: the PC chain ran past the exit ecall and still ended on pc = 1"
    );
}

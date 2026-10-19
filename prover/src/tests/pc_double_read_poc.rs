//! PoC: a free CPU `TIMESTAMP` lets `pc_double_read` reorder register accesses.
//!
//! Each CPU row reads the PC at `timestamp - 3 + 3 * pc_double_read` and writes
//! `next_pc` at `timestamp + 1`. `pc_double_read` is only bit-checked. With a
//! prover-chosen timestamp, row B can sit at `ts_B = ts_A + 1` with
//! `pc_double_read = 1`: its PC read lands on A's `next_pc` write directly, and
//! its `rs1` read at `ts_A + 1` sorts before A's `rd` write at `ts_A + 2`, so B
//! reads the register value from before A ran.
//!
//! The guest runs `li a0, 5; mv a1, a0; mv a0, a1; exit`, i.e. `exit(5)`, which
//! no honest prover can prove. The forgery has `mv a1, a0` read the stale
//! `a0 = 0` and proves `exit(0)`. The spec makes the timestamp a preprocessed
//! column (`4 * (i + 1)`), which leaves the prover no room for `ts_A + 1`.
//!
//! Production proof options, as in `halt_pc_forgery_poc`.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::elf::Elf;
use executor::vm::logs::Log;
use stark::prover::{IsStarkProver, Prover};

use super::halt_pc_forgery_poc::{opts, run_logs};
use crate::statement::{StatementKind, absorb_statement};
use crate::tables::cpu::{CpuOperation, cols as cpu_cols};
use crate::tables::decode;
use crate::tables::trace_builder::Traces;
use crate::tables::types::VmTable;
use crate::test_utils::{E, asm_elf_bytes};
use crate::{MaxRowsConfig, VmAirs, VmProof};

const GUEST: &str = "poc_pc_double_read";

/// `li a0, 5` and `li a0, 0` (`addi a0, zero, imm`).
const LI_A0_5: [u8; 4] = [0x13, 0x05, 0x50, 0x00];
const LI_A0_0: [u8; 4] = [0x13, 0x05, 0x00, 0x00];

/// Row A writes `a0`, row B (`mv a1, a0`) reads it, row C (`mv a0, a1`) copies it back.
const ROW_A: usize = 0;
const ROW_B: usize = 1;
const ROW_C: usize = 2;

/// The guest with `li a0, 5` replaced by `li a0, 0`: honestly provable.
fn patched_elf(honest: &[u8]) -> Vec<u8> {
    let at = honest
        .windows(LI_A0_5.len())
        .position(|w| w == LI_A0_5)
        .expect("li a0, 5 not found");
    let mut patched = honest.to_vec();
    patched[at..at + 4].copy_from_slice(&LI_A0_0);
    patched
}

/// CPU operations with the honest timestamps `4 * (i + 1)`, in program order.
fn honest_ops(program: &Elf, logs: &[Log]) -> Vec<CpuOperation> {
    let instructions = decode::instructions_from_elf(program).expect("decode");
    logs.iter()
        .enumerate()
        .map(|(i, log)| {
            CpuOperation::from_log_and_instruction(
                log,
                4 * (i as u64 + 1),
                instructions[&log.current_pc],
            )
        })
        .collect()
}

/// The forged run: B and C see the stale `a0 = 0`, B sits at `ts_A + 1` and every
/// later row moves down by 3 so the PC chain stays connected. B is listed before
/// A, so the trace builder's register walk serves B's read before A's write; row
/// order is otherwise irrelevant to the buses.
fn forged_ops(program: &Elf, logs: &[Log]) -> Vec<CpuOperation> {
    let mut logs = logs.to_vec();
    assert_eq!(logs[ROW_B].src1_val, 5, "B honestly reads a0 = 5");
    for row in [ROW_B, ROW_C] {
        logs[row].src1_val = 0;
        logs[row].dst_val = 0;
    }
    let instructions = decode::instructions_from_elf(program).expect("decode");
    let mut ops: Vec<CpuOperation> = logs
        .iter()
        .enumerate()
        .map(|(i, log)| {
            let honest_ts = 4 * (i as u64 + 1);
            let ts = if i == ROW_A { honest_ts } else { honest_ts - 3 };
            CpuOperation::from_log_and_instruction(log, ts, instructions[&log.current_pc])
        })
        .collect();
    ops.swap(ROW_A, ROW_B);
    ops
}

/// How the malicious prover deviates from the traces built from its ops.
#[derive(Clone, Copy, Default)]
struct Tamper {
    /// `pc_double_read = 1` on B (CPU row 0 after the swap).
    double_read: bool,
    /// Overwrite every CPU `TIMESTAMP` cell with the trace the builder wrote before
    /// the fix: the ops' own timestamps on the real rows, then padding continuing
    /// the +4 cadence from the last real row. With `double_read` this is exactly the
    /// CPU trace that verified on the base. TIMESTAMP is now preprocessed, so the
    /// column no longer matches the commitment the verifier expects.
    timestamp_column: bool,
}

const FULL_ATTACK: Tamper = Tamper {
    double_read: true,
    timestamp_column: true,
};

fn craft_proof(
    elf_bytes: &[u8],
    ops: Vec<CpuOperation>,
    tamper: Tamper,
) -> Result<VmProof, stark::prover::ProvingError> {
    let options = opts();
    let program = Elf::load(elf_bytes).expect("ELF load");
    let max_rows = MaxRowsConfig::default();
    let timestamps: Vec<u64> = ops.iter().map(|op| op.timestamp).collect();
    let mut traces = Traces::from_elf_and_cpu_ops(&program, ops, &max_rows).expect("trace build");
    let cpu = &mut traces.cpus[0].main_table;
    if tamper.double_read {
        cpu.set_bool(0, cpu_cols::PC_DOUBLE_READ, true);
    }
    if tamper.timestamp_column {
        let last_ts = *timestamps.last().expect("at least one row");
        let padding = (1..).map(|j| last_ts + 4 * j);
        for (row, ts) in timestamps
            .iter()
            .copied()
            .chain(padding)
            .take(cpu.height)
            .enumerate()
        {
            cpu.set_u64(row, cpu_cols::TIMESTAMP, ts);
        }
    }

    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &program,
        &options,
        false,
        &traces.page_configs,
        &table_counts,
        &crate::cpu_heights_from_traces(&traces),
        None,
        true,
        None,
        crate::test_utils::trace_register_commitment(
            &options,
            &crate::tables::register::register_init_from_entry_point(program.entry_point),
            &traces,
        ),
    );
    let runtime_page_ranges = traces.runtime_page_ranges();

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::Monolithic,
        elf_bytes,
        &traces.public_output_bytes,
        &table_counts,
        0,
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
        num_private_input_pages: 0,
    })
}

/// Did the verifier accept the crafted proof?
///
/// Only a tamper of the preprocessed TIMESTAMP column may be refused by the
/// prover, and only for its commitment. Which layer fires depends on the
/// precomputed-tree cache (`crypto/stark/src/prover.rs`): cold, the prover rebuilds
/// the forged column's tree, the root disagrees and it refuses with
/// `PrecomputedCommitmentMismatch`; warm (an honest proof of the same height ran
/// earlier in the process), it substitutes the honest tree and leaves the verifier
/// to reject. Every other crafted proof must build, so its rejection comes from
/// the verifier.
fn accepted(
    crafted: Result<VmProof, stark::prover::ProvingError>,
    elf: &[u8],
    tamper: Tamper,
) -> bool {
    match crafted {
        Ok(proof) => crate::verify_with_options(&proof, elf, &opts(), None, None)
            .expect("verifier must not error"),
        Err(e) => {
            assert!(
                tamper.timestamp_column
                    && matches!(
                        e,
                        stark::prover::ProvingError::PrecomputedCommitmentMismatch
                    ),
                "only the TIMESTAMP tamper may be refused, and only for its \
                 preprocessed commitment: {e:?}"
            );
            false
        }
    }
}

/// POSITIVE CONTROL: the harness builds valid proofs from its own CPU ops.
#[test]
fn poc_control_patched_program_verifies() {
    let patched = patched_elf(&asm_elf_bytes(GUEST));
    let program = Elf::load(&patched).unwrap();
    let ops = honest_ops(&program, &run_logs(&patched));
    let tamper = Tamper::default();
    assert!(accepted(
        craft_proof(&patched, ops, tamper),
        &patched,
        tamper
    ));
}

/// NEGATIVE CONTROL: `exit(5)` honestly is rejected (x10 != 0 at HALT).
#[test]
fn poc_control_honest_exit_five_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let program = Elf::load(&elf).unwrap();
    let ops = honest_ops(&program, &run_logs(&elf));
    let tamper = Tamper::default();
    assert!(!accepted(craft_proof(&elf, ops, tamper), &elf, tamper));
}

/// NEGATIVE CONTROL: the forged run without `pc_double_read` is rejected. On the
/// base, where the builder wrote the forged timestamps into the TIMESTAMP column,
/// B then read the PC at `ts_A - 2`, where no token was written: this isolated the
/// double-read bit as the lever. The TIMESTAMP column is left as the builder
/// writes it (the preprocessed `4·(i + 1)`), so the proof builds and the verifier
/// rejects it.
#[test]
fn poc_control_stale_read_without_double_read_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let program = Elf::load(&elf).unwrap();
    let ops = forged_ops(&program, &run_logs(&elf));
    let tamper = Tamper::default();
    assert!(!accepted(craft_proof(&elf, ops, tamper), &elf, tamper));
}

/// THE BUG, with the exact CPU trace that verified on the base: B at `ts_A + 1`
/// with `pc_double_read = 1` reads `a0` before A writes it, turning `exit(5)` into
/// `exit(0)`. Two independent pins now reject it: the TIMESTAMP column is not the
/// preprocessed `4·(i + 1)` (the prover refuses it, or the verifier rejects its
/// openings), and the forged PC chain, shifted by −3, ends at `4·N − 2` instead of
/// the verifier-known `4·N + 1` REGISTER consumes x255 at.
#[test]
fn stale_register_read_via_pc_double_read_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let program = Elf::load(&elf).unwrap();
    let ops = forged_ops(&program, &run_logs(&elf));
    assert!(
        !accepted(craft_proof(&elf, ops, FULL_ATTACK), &elf, FULL_ATTACK),
        "SOUNDNESS BUG: a row with pc_double_read = 1 one timestamp after the \
         previous row read a register before that row wrote it (exit(5) -> exit(0))"
    );
}

/// THE BUG with the CPU `TIMESTAMP` column left as the preprocessed `4·(i + 1)`:
/// the other tables still see the forged timestamps, so the buses do not balance
/// and the verifier rejects a proof the prover builds without complaint.
#[test]
fn stale_register_read_with_honest_timestamp_column_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let program = Elf::load(&elf).unwrap();
    let ops = forged_ops(&program, &run_logs(&elf));
    let tamper = Tamper {
        timestamp_column: false,
        ..FULL_ATTACK
    };
    assert!(!accepted(craft_proof(&elf, ops, tamper), &elf, tamper));
}

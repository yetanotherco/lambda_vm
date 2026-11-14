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
//! The forged run is shifted by −3 from B on, so two verifier pins reject it:
//! the TIMESTAMP column is not the preprocessed one, and the PC chain ends at
//! `4·N − 2` instead of the verifier-known `4·N + 1` REGISTER consumes x255 at.
//! The tests build the forgery against the pins the prover would choose (its own
//! TIMESTAMP root and final PC timestamp, as before the fix) and swap the honest
//! pins back one at a time, so each rejection has a single cause.
//!
//! Production proof options, as in `halt_pc_forgery_poc`.

use executor::elf::Elf;
use executor::vm::logs::Log;
use stark::config::Commitment;
use stark::leaf_layout::LeafLayout;
use stark::proof::view::MultiProofView;
use stark::prover::IsStarkProver;
use stark::verifier::IsStarkVerifier;

use super::halt_pc_forgery_poc::{opts, run_logs};
use crate::statement::{StatementKind, absorb_statement};
use crate::tables::cpu::{self, CpuOperation, cols as cpu_cols};
use crate::tables::trace_builder::Traces;
use crate::tables::types::VmTable;
use crate::tables::{decode, register};
use crate::test_utils::{asm_elf_bytes, create_cpu_air, create_register_air};
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

/// The CPU operation of `log` at timestamp `ts`, read against the program's
/// decode table (as the trace builder's own ops are).
fn op_at(table: &decode::DecodeTable, log: &Log, ts: u64) -> CpuOperation {
    let row = table
        .row(log.current_pc)
        .expect("an instruction of the program");
    CpuOperation::from_log(log, ts, table.entry(row), row)
}

/// The program's decode table, as the trace builder reads it.
fn decode_table(program: &Elf) -> decode::DecodeTable {
    let instructions = decode::instructions_from_elf(program).expect("decode");
    decode::DecodeTable::from_instructions(&instructions)
}

/// CPU operations with the honest timestamps `4 * (i + 1)`, in program order.
fn honest_ops(program: &Elf, logs: &[Log]) -> Vec<CpuOperation> {
    let table = decode_table(program);
    logs.iter()
        .enumerate()
        .map(|(i, log)| op_at(&table, log, 4 * (i as u64 + 1)))
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
    let table = decode_table(program);
    let mut ops: Vec<CpuOperation> = logs
        .iter()
        .enumerate()
        .map(|(i, log)| {
            let honest_ts = 4 * (i as u64 + 1);
            let ts = if i == ROW_A { honest_ts } else { honest_ts - 3 };
            op_at(&table, log, ts)
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
    /// CPU trace that verified on the base.
    timestamp_column: bool,
}

const FULL_ATTACK: Tamper = Tamper {
    double_read: true,
    timestamp_column: true,
};

/// What the CPU table's TIMESTAMP column and REGISTER's x255 final token are
/// pinned to. The verifier always uses [`Crafted::honest`]; the forgery is
/// committed against [`Crafted::forged`].
#[derive(Clone, Copy)]
struct Pins {
    timestamp_root: Commitment,
    final_pc_ts: u64,
}

/// A traced run, ready to be proven and verified against chosen pins.
struct Crafted {
    elf_bytes: Vec<u8>,
    program: Elf,
    traces: Traces,
    /// The verifier's pins: the preprocessed `4·(i + 1)` column and `4·N + 1`.
    honest: Pins,
    /// The pins the trace itself implies: the root of its TIMESTAMP column and the
    /// timestamp its last row writes `next_pc` at.
    forged: Pins,
}

fn craft(elf_bytes: &[u8], ops: Vec<CpuOperation>, tamper: Tamper) -> Crafted {
    let options = opts();
    let program = Elf::load(elf_bytes).expect("ELF load");
    let timestamps: Vec<u64> = ops.iter().map(|op| op.timestamp).collect();
    let mut traces = Traces::from_elf_and_cpu_ops(&program, ops, &MaxRowsConfig::default())
        .expect("trace build");
    assert_eq!(traces.cpus.len(), 1, "one CPU table");
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
    let height = cpu.height;
    let column: Vec<_> = (0..height)
        .map(|row| *cpu.get(row, cpu_cols::TIMESTAMP))
        .collect();
    let last_ts = column.last().expect("non-empty table").to_raw();
    let honest = Pins {
        timestamp_root: cpu::timestamp_commitment(&options, height),
        final_pc_ts: cpu::final_pc_timestamp(height as u64),
    };
    let forged = Pins {
        timestamp_root: cpu::commit_timestamp_column(&options, &column, LeafLayout::RowPair),
        final_pc_ts: last_ts + 1,
    };
    let crafted = Crafted {
        elf_bytes: elf_bytes.to_vec(),
        program,
        traces,
        honest,
        forged,
    };
    // HALT zeroizes x1-x31 on every run here, forged or not, so the trace's FINI is
    // the halted register file the production verifier pins.
    assert_eq!(
        crate::test_utils::trace_register_commitment(
            &opts(),
            &register::register_init_from_entry_point(crafted.program.entry_point),
            &crafted.traces,
        )
        .commitment(),
        crafted.register_commitment().commitment(),
        "the run must end in the halted register file"
    );
    crafted
}

impl Crafted {
    /// The production AIR set with the CPU TIMESTAMP root and the REGISTER final
    /// PC timestamp replaced by `pins`.
    fn airs(&self, pins: Pins) -> VmAirs {
        let options = opts();
        let table_counts = self.traces.table_counts();
        let mut airs = VmAirs::new(
            &self.program,
            &options,
            false,
            &self.traces.page_configs,
            &table_counts,
            &crate::cpu_heights_from_traces(&self.traces),
            None,
            true,
            None,
            self.register_commitment().preprocessed(),
        );
        airs.cpus[0] = Box::new(
            create_cpu_air(&options, 0)
                .with_name("CPU[0]")
                .with_preprocessed(pins.timestamp_root, cpu::NUM_PRECOMPUTED_COLS),
        );
        airs.register = Box::new(
            create_register_air(&options, pins.final_pc_ts).with_preprocessed(
                self.register_commitment().commitment(),
                register::NUM_PREPROCESSED_COLS,
            ),
        );
        airs
    }

    /// As the production verifier: REGISTER FINI pinned to the halted register file.
    fn register_commitment(&self) -> crate::RegisterFiles {
        crate::test_utils::halted_register_commitment(
            &opts(),
            &self.program,
            &self.traces.public_output_bytes,
        )
    }

    fn transcript(&self) -> crate::hash_pin::BlockTranscript {
        let mut transcript = crate::hash_pin::block_transcript(&[]);
        absorb_statement(
            &mut transcript,
            StatementKind::Monolithic,
            &self.elf_bytes,
            &self.traces.public_output_bytes,
            &self.traces.table_counts(),
            0,
            &self.traces.runtime_page_ranges(),
            opts().fri_final_poly_log_degree,
        );
        transcript
    }

    /// Prove against `pins`. The CPU TIMESTAMP root is whatever `pins` says, so
    /// with `forged` pins the prover commits its own column consistently.
    fn prove(mut self, pins: Pins) -> (Self, Result<VmProof, stark::prover::ProvingError>) {
        let airs = self.airs(pins);
        let mut transcript = self.transcript();
        let proof = crate::hash_pin::BlockProver::multi_prove(
            airs.air_trace_pairs(&mut self.traces),
            &mut transcript,
            stark::residency_mode::ResidencyMode::Retain,
        )
        .map(|proof| VmProof {
            proof,
            runtime_page_ranges: self.traces.runtime_page_ranges(),
            table_counts: self.traces.table_counts(),
            public_output: self.traces.public_output_bytes.clone(),
            num_private_input_pages: 0,
        });
        (self, proof)
    }

    /// Verify `proof` against `pins`, with every other check of the production
    /// verifier.
    fn verifies(&self, proof: &VmProof, pins: Pins) -> bool {
        let airs = self.airs(pins);
        let refs = airs.air_refs();
        let view = MultiProofView::Owned(&proof.proof);
        let Some(expected) = crate::compute_expected_commit_bus_balance_view(
            &refs,
            view,
            &proof.public_output,
            0,
            &mut self.transcript(),
        ) else {
            return false;
        };
        crate::hash_pin::BlockVerifier::multi_verify_views(
            &refs,
            view,
            &mut self.transcript(),
            &expected,
        )
    }
}

/// The forgery, proven against the pins it implies (what a free TIMESTAMP and a
/// prover-chosen final PC timestamp allowed before the fix).
fn forged_proof(tamper: Tamper) -> (Crafted, VmProof) {
    let elf = asm_elf_bytes(GUEST);
    let program = Elf::load(&elf).unwrap();
    let crafted = craft(&elf, forged_ops(&program, &run_logs(&elf)), tamper);
    let pins = crafted.forged;
    let (crafted, proof) = crafted.prove(pins);
    (
        crafted,
        proof.expect("the forgery is consistent with its own pins"),
    )
}

/// A verdict of the production verifier. A prover refusal counts as "not
/// accepted" only when it is the preprocessed TIMESTAMP root disagreeing with the
/// forged column (cold precomputed-tree cache; with a warm cache the honest tree is
/// substituted and the verifier rejects instead).
fn production_accepts(
    proof: Result<VmProof, stark::prover::ProvingError>,
    elf: &[u8],
    tamper: Tamper,
) -> bool {
    match proof {
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
    let crafted = craft(
        &patched,
        honest_ops(&program, &run_logs(&patched)),
        Tamper::default(),
    );
    let pins = crafted.honest;
    let (_, proof) = crafted.prove(pins);
    assert!(production_accepts(proof, &patched, Tamper::default()));
}

/// NEGATIVE CONTROL: `exit(5)` honestly is rejected (x10 != 0 at HALT).
#[test]
fn poc_control_honest_exit_five_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let program = Elf::load(&elf).unwrap();
    let crafted = craft(
        &elf,
        honest_ops(&program, &run_logs(&elf)),
        Tamper::default(),
    );
    let pins = crafted.honest;
    let (_, proof) = crafted.prove(pins);
    assert!(!production_accepts(proof, &elf, Tamper::default()));
}

/// THE BUG, as it was before the fix: against the pins the prover chooses, the
/// forged `exit(0)` is a valid proof. Everything but the two pins is the
/// production verifier.
#[test]
fn poc_forgery_verifies_against_prover_chosen_pins() {
    let (crafted, proof) = forged_proof(FULL_ATTACK);
    assert_ne!(crafted.forged.timestamp_root, crafted.honest.timestamp_root);
    assert_eq!(crafted.forged.final_pc_ts + 3, crafted.honest.final_pc_ts);
    assert!(crafted.verifies(&proof, crafted.forged));
}

/// NEGATIVE CONTROL: the same forgery without `pc_double_read` is rejected even
/// against the prover-chosen pins — B reads the PC at `ts_A - 2`, where no token
/// was written. The double-read bit is the lever.
#[test]
fn poc_control_forgery_without_double_read_is_rejected() {
    let (crafted, proof) = forged_proof(Tamper {
        double_read: false,
        ..FULL_ATTACK
    });
    assert!(!crafted.verifies(&proof, crafted.forged));
}

/// The preprocessed TIMESTAMP alone rejects the forgery: only the CPU root is
/// switched back to the verifier's `4·(i + 1)`.
#[test]
fn preprocessed_timestamp_alone_rejects_the_forgery() {
    let (crafted, proof) = forged_proof(FULL_ATTACK);
    let pins = Pins {
        timestamp_root: crafted.honest.timestamp_root,
        ..crafted.forged
    };
    assert!(!crafted.verifies(&proof, pins));
}

/// The final PC pin alone rejects the forgery: only REGISTER's x255 final token
/// is moved back to the verifier's `4·N + 1`.
#[test]
fn final_pc_pin_alone_rejects_the_forgery() {
    let (crafted, proof) = forged_proof(FULL_ATTACK);
    let pins = Pins {
        final_pc_ts: crafted.honest.final_pc_ts,
        ..crafted.forged
    };
    assert!(!crafted.verifies(&proof, pins));
}

/// THE BUG through the production prover and verifier: the exact CPU trace that
/// verified on the base is rejected (the prover refuses its TIMESTAMP column, or
/// the verifier rejects the proof).
#[test]
fn stale_register_read_via_pc_double_read_is_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let program = Elf::load(&elf).unwrap();
    let crafted = craft(&elf, forged_ops(&program, &run_logs(&elf)), FULL_ATTACK);
    let pins = crafted.honest;
    let (_, proof) = crafted.prove(pins);
    assert!(
        !production_accepts(proof, &elf, FULL_ATTACK),
        "SOUNDNESS BUG: a row with pc_double_read = 1 one timestamp after the \
         previous row read a register before that row wrote it (exit(5) -> exit(0))"
    );
}

//! End-to-end tests of the preprocessed CPU timestamp across several CPU tables.
//!
//! Every CPU table commits the same preprocessed column `4·(i + 1)` and adds
//! `4·S_k` (the total height of the tables before it) to the timestamps it
//! sends. These tests split one run over several CPU tables and check that an
//! honest proof verifies, and that a prover cannot move a CPU table to another
//! position: two full tables have the same preprocessed commitment, so only the
//! per-table base tells them apart. Without it the buses are permutation
//! invariant and the reordered tables would balance. They also check that the
//! verifier reads the CPU heights from the right sub-proofs and rejects hostile
//! ones cleanly.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::elf::Elf;
use stark::prover::{IsStarkProver, Prover};

use super::halt_pc_forgery_poc::{opts, run_logs};
use crate::statement::{StatementKind, absorb_statement};
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::test_utils::{E, asm_elf_bytes};
use crate::{VmAirs, VmProof};

const GUEST: &str = "all_instructions_64";

/// Prove `elf` split over 32-row CPU tables, after `tamper` edits the traces.
fn craft_proof(
    elf_bytes: &[u8],
    tamper: impl FnOnce(&mut Traces),
) -> Result<VmProof, stark::prover::ProvingError> {
    let options = opts();
    let program = Elf::load(elf_bytes).expect("ELF load");
    let mut traces = Traces::from_elf_and_logs(
        &program,
        &run_logs(elf_bytes),
        &MaxRowsConfig::small(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("trace build");
    tamper(&mut traces);

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
        crate::test_utils::halted_register_commitment(
            &options,
            &program,
            &traces.public_output_bytes,
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

fn verifies(crafted: Result<VmProof, stark::prover::ProvingError>, elf: &[u8]) -> bool {
    let proof = crafted.expect("the crafted proof must build");
    crate::verify_with_options(&proof, elf, &opts(), None, None).expect("verifier must not error")
}

/// POSITIVE CONTROL: a run split over several CPU tables verifies, the later
/// tables continuing the first one's timestamp cadence.
#[test]
fn split_cpu_tables_verify() {
    let elf = asm_elf_bytes(GUEST);
    let rows = run_logs(&elf).len();
    let mut full_tables = 0;
    let crafted = craft_proof(&elf, |traces| {
        full_tables = traces
            .cpus
            .iter()
            .filter(|t| t.num_rows() == MaxRowsConfig::small().cpu)
            .count();
        // The heights the prover checks right after execution are the ones the
        // trace builder produces.
        assert_eq!(
            crate::tables::cpu::table_heights(rows, MaxRowsConfig::small().cpu),
            crate::cpu_heights_from_traces(traces)
        );
    });
    assert!(
        full_tables >= 2,
        "need two full CPU tables, got {full_tables}"
    );
    assert!(verifies(crafted, &elf));
}

/// Two full CPU tables swapped: same preprocessed commitments, same heights,
/// same rows — only their timestamp bases differ.
#[test]
fn swapped_cpu_tables_are_rejected() {
    let elf = asm_elf_bytes(GUEST);
    let crafted = craft_proof(&elf, |traces| traces.cpus.swap(0, 1));
    assert!(
        !verifies(crafted, &elf),
        "CPU tables verified at each other's timestamp base"
    );
}

/// Hostile CPU heights in an honest proof: heights `validate_heights` refuses are
/// an error before any AIR is built; a valid but wrong height picks a different
/// preprocessed root and the proof is rejected. None of them panics.
#[test]
fn hostile_cpu_heights_are_rejected_by_the_verifier() {
    let elf = asm_elf_bytes(GUEST);
    let honest = craft_proof(&elf, |_| {}).expect("honest proof");
    assert!(crate::verify_with_options(&honest, &elf, &opts(), None, None).unwrap());
    let cpu = crate::cpu_proof_start(true, &honest.table_counts);
    assert_eq!(
        honest.proof.proofs[cpu].trace_length,
        MaxRowsConfig::small().cpu
    );

    let with_height = |height: usize| {
        let mut proof = honest.clone();
        proof.proof.proofs[cpu].trace_length = height;
        crate::verify_with_options(&proof, &elf, &opts(), None, None)
    };
    for height in [0, 2, 3, 6, crate::tables::max_rows::CPU * 2, usize::MAX] {
        assert!(
            matches!(
                with_height(height),
                Err(crate::Error::InvalidTableCounts(_))
            ),
            "height {height}"
        );
    }
    for height in [16, 64] {
        assert!(!with_height(height).unwrap(), "height {height}");
    }
}

/// `cpu_proof_start` (which the verifier reads CPU heights with) matches the
/// `air_refs` layout for every HALT / accelerator combination. Checked in
/// release builds too, unlike the `debug_assert` in `air_refs`.
#[test]
fn cpu_proof_start_matches_air_refs_layout() {
    let elf = asm_elf_bytes(GUEST);
    let program = Elf::load(&elf).unwrap();
    let options = opts();
    let honest = craft_proof(&elf, |_| {}).expect("honest proof");
    for include_halt in [false, true] {
        for mask in 0u8..32 {
            let mut counts = honest.table_counts.clone();
            let bit = |i: u8| usize::from(mask >> i & 1);
            counts.keccak = bit(0);
            counts.keccak_rnd = bit(1);
            counts.ecsm = bit(2);
            counts.ecdas = bit(3);
            counts.hint = bit(4);
            let airs = VmAirs::new(
                &program,
                &options,
                false,
                &[],
                &counts,
                &vec![4; counts.cpu],
                None,
                include_halt,
                None,
                [0u8; 32],
            );
            let refs = airs.air_refs();
            let start = crate::cpu_proof_start(include_halt, &counts);
            for k in 0..counts.cpu {
                assert_eq!(
                    refs[start + k].name(),
                    format!("CPU[{k}]"),
                    "halt={include_halt} accelerators={mask:05b}"
                );
            }
        }
    }
}

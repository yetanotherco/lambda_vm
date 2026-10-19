//! End-to-end tests of the preprocessed CPU timestamp across several CPU tables.
//!
//! Every CPU table commits the same preprocessed column `4·(i + 1)` and adds
//! `4·S_k` (the total height of the tables before it) to the timestamps it
//! sends. These tests split one run over several CPU tables and check that an
//! honest proof verifies, and that a prover cannot move a CPU table to another
//! position: two full tables have the same preprocessed commitment, so only the
//! per-table base tells them apart. Without it the buses are permutation
//! invariant and the reordered tables would balance.

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
    let mut full_tables = 0;
    let crafted = craft_proof(&elf, |traces| {
        full_tables = traces
            .cpus
            .iter()
            .filter(|t| t.num_rows() == MaxRowsConfig::small().cpu)
            .count();
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

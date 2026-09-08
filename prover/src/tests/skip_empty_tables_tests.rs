//! A chip the run never reaches is left out of the proof entirely.
//!
//! Every table used to cost a padded sub-proof — a full commitment, FRI chain
//! and OOD opening set — whether or not the program executed a single one of its
//! operations. These tests cover both directions of dropping them: an unused
//! chip is absent and the proof still verifies, and a chip whose operations
//! *did* run cannot be dropped, because the LogUp bus no longer balances.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use stark::proof::options::ProofOptions;
use stark::proof::view::StarkProofView;
use stark::verifier::{IsStarkVerifier, Verifier};

use executor::elf::Elf;

use crate::VmAirs;
use crate::tables::trace_builder::Traces;
use crate::test_utils::{E, F, multi_prove_ram, run_asm_elf};

/// Prove and verify `traces` with the AIR set that its own table counts
/// describe, so an absent chip is absent on both sides — exactly how the
/// production prover and verifier reconstruct the shape.
fn prove_and_verify(elf: &Elf, traces: &mut Traces) -> bool {
    let proof_options = ProofOptions::default_test_options();
    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        elf,
        &proof_options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );

    let multi_proof = match multi_prove_ram(
        airs.air_trace_pairs(traces),
        &mut DefaultTranscript::<E>::new(&[]),
    ) {
        Ok(proof) => proof,
        Err(_) => return false,
    };
    let views: Vec<StarkProofView<F, E, ()>> = multi_proof
        .proofs
        .iter()
        .map(StarkProofView::Owned)
        .collect();

    let expected_bus_balance = match crate::compute_expected_commit_bus_balance_view(
        &airs.air_refs(),
        &views,
        &traces.public_output_bytes,
        0,
        &mut DefaultTranscript::<E>::new(&[]),
    ) {
        Some(balance) => balance,
        None => return false,
    };

    Verifier::multi_verify_views(
        &airs.air_refs(),
        &views,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    )
}

/// The premise `TableCounts::validate` now leans on: a chip influences the run
/// only through its bus interactions, so an absent chip is caught by the
/// bus-balance check. A table with no interactions is skipped by that sum
/// (`Verifier::multi_verify` filters on `has_trace_interaction`), so adding one
/// would let a prover drop it unnoticed. Nothing in the AIR set may be in that
/// position.
#[test]
fn every_table_participates_in_the_bus() {
    let (elf, logs, _instructions) = run_asm_elf("test_mul_8");
    let traces = Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &elf,
        &ProofOptions::default_test_options(),
        true,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );

    let without_bus: Vec<&str> = airs
        .air_refs()
        .iter()
        .filter(|air| !air.has_trace_interaction())
        .map(|air| air.name())
        .collect();
    assert!(
        without_bus.is_empty(),
        "these tables carry no bus interactions, so dropping them would go undetected: {without_bus:?}"
    );
}

/// A program that never multiplies or divides gets no MUL and no DVRM table,
/// and still verifies.
#[test]
fn a_run_without_multiplication_omits_the_mul_table() {
    let (elf, logs, _instructions) = run_asm_elf("xori");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();

    let table_counts = traces.table_counts();
    assert_eq!(table_counts.mul, 0, "xori executes no multiplication");
    assert_eq!(table_counts.dvrm, 0, "xori executes no division");
    assert!(traces.muls.is_empty());
    assert!(traces.dvrms.is_empty());
    assert!(
        table_counts.validate().is_ok(),
        "zero counts on unused chips are legitimate"
    );

    assert!(
        prove_and_verify(&elf, &mut traces),
        "a proof without the unused chips must still verify"
    );
}

/// The forgery the relaxed `validate` has to survive: keep the CPU trace that
/// executed the multiplications, drop the MUL table, and declare `mul = 0`.
/// Both sides then agree on the reduced shape — the transcript replays, every
/// sub-proof is internally sound — and the only thing left to catch it is the
/// bus balance, whose sum over the *present* tables no longer matches once the
/// CPU's MUL sends have no receiver.
#[test]
fn omitting_a_table_whose_ops_ran_fails_the_bus_balance() {
    let (elf, logs, _instructions) = run_asm_elf("test_mul_8");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        !traces.muls.is_empty(),
        "test_mul_8 must exercise MUL for this to be a forgery"
    );

    assert!(
        prove_and_verify(&elf, &mut traces),
        "the honest proof must verify first, or the negative below proves nothing"
    );

    traces.muls.clear();
    assert_eq!(traces.table_counts().mul, 0);
    assert!(
        traces.table_counts().validate().is_ok(),
        "validate deliberately lets this through — the bus is what rejects it"
    );

    assert!(
        !prove_and_verify(&elf, &mut traces),
        "dropping MUL while the CPU still sends MUL requests must not verify"
    );
}

/// What `validate` still refuses: a proof with no CPU describes no execution,
/// and one with no register file has nothing to carry register state.
#[test]
fn validate_still_requires_cpu_and_the_register_file() {
    let (elf, logs, _instructions) = run_asm_elf("test_mul_8");
    let traces = Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    let honest = traces.table_counts();
    assert!(honest.validate().is_ok());

    let mut no_cpu = honest.clone();
    no_cpu.cpu = 0;
    assert!(
        no_cpu.validate().is_err(),
        "a proof with no CPU is rejected"
    );

    let mut no_registers = honest.clone();
    no_registers.memw_register = 0;
    assert!(
        no_registers.validate().is_err(),
        "a proof with no register file is rejected"
    );
}

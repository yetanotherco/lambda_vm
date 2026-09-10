//! A chip the run never reaches is left out of the proof entirely.
//!
//! Every table used to cost a padded sub-proof — a full commitment, FRI chain
//! and OOD opening set — whether or not the program executed a single one of its
//! operations. These tests cover both directions of dropping them: an unused
//! chip is absent and the proof still verifies, and a chip whose operations
//! *did* run cannot be dropped, because the LogUp bus no longer balances.

use stark::proof::options::ProofOptions;

use executor::elf::Elf;

use crate::VmAirs;
use crate::tables::trace_builder::Traces;
use crate::test_utils::run_asm_elf;
use crate::tests::prove_elfs_tests::prove_and_verify_vm_minimal;

/// The premise `TableCounts::validate` now leans on: a chip influences the run
/// only through its bus interactions, so an absent chip is caught by the
/// bus-balance check. A table with no interactions is skipped by that sum
/// (`Verifier::multi_verify` filters on `has_trace_interaction`), so adding one
/// would let a prover drop it unnoticed. Nothing in the AIR set may be in that
/// position.
///
/// Declaring interactions is necessary but not sufficient: a chip whose bus
/// footprint cancelled against itself would contribute zero however many rows it
/// carried, and dropping it would also go unnoticed. No chip is built that way —
/// each one receives its dispatch and sends its own lookups — but this test does
/// not prove that part, and there is no cheap structural check that would.
#[test]
fn every_table_participates_in_the_bus() {
    let (elf, logs, _instructions) = run_asm_elf("test_mul_8");
    let traces = Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    // One chunk of every table, not the counts this program happens to produce:
    // a zero count builds no AIR, so driving the set off a single program would
    // leave the very tables this change makes droppable out of the check.
    let table_counts = crate::TableCounts {
        cpu: 1,
        lt: 1,
        memw: 1,
        memw_aligned: 1,
        load: 1,
        mul: 1,
        dvrm: 1,
        shift: 1,
        branch: 1,
        memw_register: 1,
        eq: 1,
        bytewise: 1,
        store: 1,
        cpu32: 1,
        keccak: 1,
        keccak_rnd: 1,
        ecsm: 1,
        ecdas: 1,
        hint: 1,
        commit: 1,
    };
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
        prove_and_verify_vm_minimal(&elf, &mut traces),
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
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "the honest proof must verify first, or the negative below proves nothing"
    );

    traces.muls.clear();
    assert_eq!(traces.table_counts().mul, 0);
    assert!(
        traces.table_counts().validate().is_ok(),
        "validate deliberately lets this through — the bus is what rejects it"
    );

    assert!(
        !prove_and_verify_vm_minimal(&elf, &mut traces),
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

/// The counts ride in the proof, so they are the prover's to choose, and the
/// sub-proof cross-check compares only their sum. A plain `+` wraps silently in
/// release (the workspace sets no `overflow-checks`), so an attacker can park
/// one field near `usize::MAX`, pick a second to carry the sum around to
/// whatever `proofs.len()` is, and pass that check with the huge field intact —
/// straight into `VmAirs::new`, which sizes a `Vec` from it.
#[test]
fn counts_that_wrap_have_no_total() {
    let (elf, logs, _instructions) = run_asm_elf("test_mul_8");
    let traces = Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    let honest = traces.table_counts();
    let honest_total = honest.total().expect("an honest run has a total");

    let mut wrapped = honest.clone();
    wrapped.mul += usize::MAX - honest_total;
    assert!(
        wrapped.validate().is_ok(),
        "validate looks at two fields, not at the sum"
    );
    assert_eq!(
        wrapped.total(),
        Some(usize::MAX),
        "one below the wrap still totals"
    );

    // One more takes the *sum* past the end, not the field: `mul` itself stays
    // below `usize::MAX` because the other counts hold the difference.
    wrapped.mul += 1;
    assert_eq!(
        wrapped.total(),
        None,
        "a wrapped sum must not be reported as a small one"
    );
}

/// The accelerators are the ones a run most often never reaches, and each cost a
/// four-row sub-proof regardless. A program with no keccak, no EC and no hint
/// ecall now carries none of the six.
#[test]
fn a_run_without_accelerators_omits_all_six() {
    let (elf, logs, _instructions) = run_asm_elf("xori");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();

    let counts = traces.table_counts();
    for (name, count) in [
        ("keccak", counts.keccak),
        ("keccak_rnd", counts.keccak_rnd),
        ("ecsm", counts.ecsm),
        ("ecdas", counts.ecdas),
        ("hint", counts.hint),
        ("commit", counts.commit),
    ] {
        assert_eq!(
            count, 0,
            "{name} should be absent from a plain xori program"
        );
    }
    assert!(counts.validate().is_ok());

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "a proof without any accelerator table must still verify"
    );
}

/// The accelerator counterpart of the forgery above: HINT is the softest table
/// in the set — it constrains nothing about the hinted value — so it is the one
/// worth showing cannot simply be dropped. The CPU's ecall send has no receiver
/// without it, and the bus balance is what notices.
#[test]
fn omitting_a_used_accelerator_fails_the_bus_balance() {
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    // Hard failure, not a skip: this is the only negative test covering the
    // accelerator direction, so a missing artifact has to be loud.
    let elf_bytes =
        std::fs::read(workspace_root.join("executor/program_artifacts/rust/hint_min.elf"))
            .expect("need hint_min.elf — run `make compile-programs-rust`");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let executor = executor::vm::execution::Executor::new(&elf, Vec::new()).expect("executor");
    let result = executor.run().expect("execution");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();
    assert!(
        !traces.hints.is_empty(),
        "hint_min must make a hint ecall for this to be a forgery"
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "the honest proof must verify first, or the negative below proves nothing"
    );

    traces.hints.clear();
    assert_eq!(traces.table_counts().hint, 0);

    assert!(
        !prove_and_verify_vm_minimal(&elf, &mut traces),
        "dropping HINT while the CPU still sends its ecall must not verify"
    );
}

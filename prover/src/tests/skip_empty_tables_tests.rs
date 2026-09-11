//! A chip the run never reaches is left out of the proof entirely.
//!
//! Every table used to cost a padded sub-proof — a full commitment, FRI chain
//! and OOD opening set — whether or not the program executed a single one of its
//! operations. These tests cover both directions of dropping them: an unused
//! chip is absent and the proof still verifies, and a chip whose operations
//! *did* run cannot be dropped, because the LogUp bus no longer balances.

use math::field::element::FieldElement;
use stark::proof::options::ProofOptions;

use executor::elf::Elf;
use executor::vm::execution::ExecutionResult;

use crate::VmAirs;
use crate::tables::trace_builder::Traces;
use crate::tables::types::GoldilocksExtension;
use crate::test_utils::run_asm_elf;
use crate::tests::prove_elfs_tests::{BusOutcome, prove_and_verify_vm_minimal, weigh_the_bus};

/// Load and run one of the compiled Rust guest programs.
///
/// Hard failure, not a skip: a missing artifact would turn a negative test
/// green having asserted nothing.
fn run_rust_elf(name: &str) -> (Elf, ExecutionResult) {
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let path = workspace_root.join(format!("executor/program_artifacts/rust/{name}.elf"));
    let elf_bytes = std::fs::read(&path)
        .unwrap_or_else(|_| panic!("need {name}.elf — run `make compile-programs-rust`"));
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let executor = executor::vm::execution::Executor::new(&elf, Vec::new()).expect("executor");
    let result = executor.run().expect("execution");
    (elf, result)
}

/// The claim every drop-forgery below makes, in one place.
///
/// `multi_verify_views` returns `false` from about ten places, so
/// `assert!(!accepted)` on its own would pass just as happily if the forged
/// proof fell over in rounds 2-4 or tripped the bus_public_inputs presence
/// check — the opposite of what these tests say they demonstrate. Moving the
/// target onto the sum the forgery actually produced is what separates them: if
/// the same proof then verifies, the balance was the only objection.
fn assert_only_the_bus_rejected(outcome: &BusOutcome, what: &str) {
    assert!(!outcome.accepted, "dropping {what} must not verify");
    assert_ne!(
        outcome.contribution_sum, outcome.target,
        "dropping {what} left the bus balanced, so the rejection came from \
         elsewhere and the bus did not notice the forgery at all"
    );
    assert!(
        outcome.accepted_with_target_moved,
        "with the target moved onto the forged sum the proof still failed, so \
         something other than the bus balance rejected {what} and this test is \
         not showing what it claims"
    );
}

/// The premise `TableCounts::validate` now leans on: a chip influences the run
/// only through its bus interactions, so an absent chip is caught by the
/// bus-balance check. A table with no interactions is skipped by that sum
/// (`Verifier::multi_verify` filters on `has_trace_interaction`), so adding one
/// would let a prover drop it unnoticed. Nothing in the AIR set may be in that
/// position.
///
/// This is the *necessary* half, and it cannot fail for any chip that exists
/// today: `has_trace_interaction()` reads a list fixed at construction and
/// every `create_*_air` passes a non-empty one. It is a tripwire for a future
/// chip built like `test_utils::busless_air`, not evidence about this change.
/// The sufficient half — that a present table's contribution is actually
/// nonzero — is [`no_present_table_contributes_zero_to_the_bus`].
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

/// The half the test above cannot reach: declaring interactions is necessary,
/// but a chip whose bus footprint cancelled against itself would contribute
/// zero however many rows it carried, and dropping *that* would move the sum by
/// nothing. The chained chips are the candidates — ECDAS receives an
/// accumulator state and sends the updated one back on the same bus, COMMIT
/// telescopes on `CommitNextByte`, KECCAK_RND chains rounds — and each is
/// supposed to be anchored by a term that does not telescope away with it.
///
/// There is nothing structural to inspect, but there is a cheap dynamic check:
/// the contribution is a public input of every sub-proof, so proving a program
/// that reaches a chip says outright what that chip puts on the bus. A zero
/// here is a table that can be dropped undetected.
#[test]
fn no_present_table_contributes_zero_to_the_bus() {
    let asm = [
        "all_instructions_64",
        "all_loadstore_32",
        "test_keccak",
        "test_ecsm",
        "test_commit_4",
    ];

    // Only the tables a prover can declare away are at risk. The always-present
    // ones (BITWISE, DECODE, KECCAK_RC, REGISTER, HALT, the PAGEs, L2G) have no
    // `TableCounts` field, and they legitimately contribute zero when the run
    // never reaches them — KECCAK_RC does exactly that in a program with no
    // keccak, which is the padding this change is about and cannot be dropped.
    let droppable = [
        "CPU",
        "MEMW_R",
        "LT",
        "MEMW",
        "MEMW_A",
        "LOAD",
        "MUL",
        "DVRM",
        "SHIFT",
        "BRANCH",
        "EQ",
        "BYTEWISE",
        "STORE",
        "CPU32",
        "KECCAK",
        "KECCAK_RND",
        "ECSM",
        "ECDAS",
        "HINT",
        "COMMIT",
    ];

    let mut seen: Vec<String> = Vec::new();
    let mut fixed_seen: Vec<String> = Vec::new();
    let mut zero: Vec<(String, String)> = Vec::new();

    let mut weigh = |program: &str, elf: &Elf, traces: &mut Traces| {
        let outcome = weigh_the_bus(elf, traces, false);
        assert!(
            outcome.accepted,
            "{program} must prove and verify honestly first"
        );
        for (table, contribution) in &outcome.per_table {
            let base = table.split('[').next().unwrap_or(table).to_string();
            let list = if droppable.contains(&base.as_str()) {
                &mut seen
            } else {
                &mut fixed_seen
            };
            if !list.contains(&base) {
                list.push(base.clone());
            }
            if droppable.contains(&base.as_str())
                && *contribution == FieldElement::<GoldilocksExtension>::zero()
            {
                zero.push((program.to_string(), table.clone()));
            }
        }
    };

    for program in asm {
        let (elf, logs, _instructions) = run_asm_elf(program);
        let mut traces =
            Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
        weigh(program, &elf, &mut traces);
    }
    let (elf, result) = run_rust_elf("hint_min");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();
    weigh("hint_min", &elf, &mut traces);

    assert!(
        zero.is_empty(),
        "these tables contribute nothing to the bus, so dropping them would go \
         unnoticed however many rows they carry: {zero:?}"
    );

    // Coverage is the weak point of a dynamic check: a table no program here
    // reaches is simply unexamined. Pin the ones covered today so the check
    // cannot quietly stop covering them.
    for table in droppable {
        assert!(
            seen.iter().any(|s| s == table),
            "{table} is no longer reached by any program here, so its \
             contribution is unexamined; seen: {seen:?}"
        );
    }
    println!("droppable tables weighed on the bus: {seen:?}");
    println!("always-present tables, not at risk here: {fixed_seen:?}");
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
    assert!(
        traces.table_counts().validate().is_ok(),
        "validate deliberately lets this through — the bus is what rejects it"
    );

    assert_only_the_bus_rejected(&weigh_the_bus(&elf, &mut traces, true), "MUL");
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
    let (elf, result) = run_rust_elf("hint_min");
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
    assert_only_the_bus_rejected(&weigh_the_bus(&elf, &mut traces, true), "HINT");
}

/// The dispatch above is CPU-to-chip. KECCAK_RND is reached from KECCAK, not
/// from the CPU, so dropping it exercises a chip-to-chip bus: the anchor whose
/// terms go unmatched sits in another accelerator's contribution, not in the
/// CPU's.
#[test]
fn omitting_a_chip_dispatched_by_another_chip_fails_the_bus_balance() {
    let (elf, logs, _instructions) = run_asm_elf("test_keccak");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        !traces.keccaks.is_empty() && !traces.keccak_rnds.is_empty(),
        "test_keccak must exercise both keccak tables for this to be a forgery"
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "the honest proof must verify first, or the negative below proves nothing"
    );

    traces.keccak_rnds.clear();
    assert_only_the_bus_rejected(&weigh_the_bus(&elf, &mut traces, true), "KECCAK_RND");
}

/// COMMIT is the one newly-optional table whose bus target is not zero: the
/// verifier computes it from the public output the prover supplies
/// (`compute_expected_commit_bus_balance_view`), and nothing requires
/// `commit > 0` when that output is non-empty. Before this change the COMMIT
/// sub-proof was structurally mandatory; now its presence is a number the
/// prover picks, and the balance is the only thing standing behind it.
#[test]
fn omitting_the_commit_table_fails_the_bus_balance() {
    let (elf, logs, _instructions) = run_asm_elf("test_commit_4");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        !traces.commits.is_empty() && !traces.public_output_bytes.is_empty(),
        "test_commit_4 must commit output for this to be a forgery"
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "the honest proof must verify first, or the negative below proves nothing"
    );

    traces.commits.clear();
    assert_only_the_bus_rejected(&weigh_the_bus(&elf, &mut traces, true), "COMMIT");
}

/// The variant the test above does not cover: a prover who drops COMMIT can
/// also drop the output it committed, and that moves the verifier's *target*
/// rather than the sum — the balance is recomputed for an empty output. The
/// CPU's commit ecalls are still in the trace with no receiver, so the two
/// sides still have to disagree.
#[test]
fn omitting_the_commit_table_and_its_output_fails_the_bus_balance() {
    let (elf, logs, _instructions) = run_asm_elf("test_commit_4");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    let honest_target = weigh_the_bus(&elf, &mut traces, false).target;

    traces.commits.clear();
    traces.public_output_bytes.clear();
    let outcome = weigh_the_bus(&elf, &mut traces, true);
    assert_ne!(
        outcome.target, honest_target,
        "clearing the public output must move the target, or this test is the \
         previous one over again"
    );
    assert_only_the_bus_rejected(&outcome, "COMMIT together with its output");
}

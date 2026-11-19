//! The lean CPU row ([`CpuRowForm::Lean`]) end to end: blocks proved under it
//! verify under it and under no other row, and today's row is untouched. The
//! forgeries the row rests on are refused at the bus level in
//! `tables::trace_builder::cpu_row_lean_tests`.

use crate::block_whir::{
    BlockFormat, BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir,
};
use crate::tables::MaxRowsConfig;
use crate::tables::cpu::CpuRowForm;
use crate::tables::fused_memory::MemoryFormat;
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::ProofOptions;

/// The production format under the word memory format, its CPU row `form`.
fn format(form: CpuRowForm) -> BlockFormat {
    format_in(form, MemoryFormat::Words)
}

/// The production format under `memory`, its CPU row `form`.
fn format_in(form: CpuRowForm, memory: MemoryFormat) -> BlockFormat {
    BlockFormat::production()
        .with_memory(memory)
        .with_cpu_row(form)
}

fn prove(elf: &[u8], format: &BlockFormat) -> BlockWhirProof {
    let options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &options,
        &Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verifies(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

/// Every instruction class (word instructions among them), the loads and
/// stores, and a call into each syscall chip the word memory format proves.
const PROGRAMS: [&str; 7] = [
    "all_instructions_64",
    "fused_mem_cases",
    "test_wsuffix_64bit",
    "test_copy_ecalls",
    "test_commit_4",
    "test_keccak256",
    "test_ecrec",
];

/// A block proved under the lean row verifies under it, and a block proved
/// under either row is refused under the other: the two rows' AIRs, DECODE's
/// columns and statement tags differ, and a verifier takes its row from its
/// own format, never from the proof.
#[test]
fn a_lean_row_block_proves_and_verifies_under_its_row_alone() {
    let lean = format(CpuRowForm::Lean);
    let full = format(CpuRowForm::Full);
    // The recursion's plan takes today's row alone.
    let elf = asm_elf_bytes("test_commit_4");
    let proof = prove(&elf, &lean);
    let refused = crate::lfm::whir_block::WhirBlockPlan::derive(
        &elf,
        &ProofOptions::default_test_options(),
        &lean,
        proof.statement(),
        None,
    );
    assert!(
        matches!(&refused, Err(e) if e.contains("CPU row")),
        "the recursion's plan refuses the lean row: {:?}",
        refused.as_ref().err()
    );
    for name in PROGRAMS {
        let elf = asm_elf_bytes(name);
        let proof = prove(&elf, &lean);
        assert!(
            verifies(&proof, &elf, &lean),
            "{name}: the lean proof verifies"
        );
        assert!(
            !verifies(&proof, &elf, &full),
            "{name}: refused under today's row"
        );
        let today = prove(&elf, &full);
        assert!(
            verifies(&today, &elf, &full),
            "{name}: today's proof verifies"
        );
        assert!(
            !verifies(&today, &elf, &lean),
            "{name}: refused under the lean row"
        );
        println!(
            "LEAN ROW BLOCK {name}: verified; tables {} lean · {} today",
            proof.table_num_vars.len(),
            today.table_num_vars.len()
        );
    }
}

/// The lean row is a variant a caller names: no preset holds it, the
/// recursion's plan refuses it, and the harness takes it only in a base-only
/// run.
#[test]
fn the_lean_row_is_named_never_preset() {
    assert_eq!(BlockFormat::production().cpu_row, CpuRowForm::Full);
    assert_eq!(BlockFormat::pinned().cpu_row, CpuRowForm::Full);
    use crate::lfm::whir_block_tree::cpu_row_setting;
    assert_eq!(cpu_row_setting(None, false), Ok(CpuRowForm::Full));
    assert_eq!(cpu_row_setting(Some("full"), false), Ok(CpuRowForm::Full));
    assert_eq!(cpu_row_setting(Some("lean"), true), Ok(CpuRowForm::Lean));
    assert!(
        cpu_row_setting(Some("lean"), false).is_err(),
        "not with the recursion"
    );
    assert!(cpu_row_setting(Some("other"), true).is_err());
}

/// Every syscall chip under the formats that take its calls: under the fused
/// memory format (the production one, which a base-only run takes without a
/// memory word) and the split one, a block calling the Keccak permutation,
/// ECSM and BLAKE3 in both modes proves and verifies under the lean row.
#[test]
fn a_lean_row_block_calling_every_chip_proves_and_verifies_under_fused_and_split() {
    const CHIPS: [&str; 7] = [
        "test_keccak",
        "test_ecsm",
        "test_blake3",
        "test_blake3_absorb",
        "test_copy_ecalls",
        "test_ecrec",
        "fused_mem_cases",
    ];
    for memory in [MemoryFormat::Fused, MemoryFormat::Split] {
        let lean = format_in(CpuRowForm::Lean, memory);
        for name in CHIPS {
            let elf = asm_elf_bytes(name);
            let proof = prove(&elf, &lean);
            assert!(verifies(&proof, &elf, &lean), "{name} under {memory:?}");
            assert!(
                !verifies(&proof, &elf, &format_in(CpuRowForm::Full, memory)),
                "{name} under {memory:?}: refused under today's row"
            );
            println!("LEAN ROW BLOCK {name} {memory:?}: verified");
        }
    }
}

/// `elf`'s run built by the windowed builder under `format` (windows of 1000
/// cycles, the small chunks), REGISTER's files the verifier's.
fn built(elf: &[u8], format: &BlockFormat) -> crate::tables::trace_builder::Traces {
    use crate::tables::trace_builder::WindowedTraceBuilder;
    use crate::tables::trace_builder::in_row_forgery_tests::with_verifier_files;
    let program = executor::elf::Elf::load(elf).expect("the ELF loads");
    let logs = executor::vm::execution::Executor::new(&program, Vec::new())
        .expect("the executor")
        .run()
        .expect("the run")
        .logs;
    let mut builder = WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small())
        .and_then(|b| b.register_format(format.registers))
        .and_then(|b| b.memw_aligned_order(format.memw_aligned))
        .map(|b| b.keccak_rnd_form(format.keccak_rnd))
        .and_then(|b| b.memory_format(format.memory))
        .and_then(|b| b.cpu_row_form(format.cpu_row))
        .expect("the builder");
    let window = 1000;
    let body = logs.len() - 1;
    let cut = body - body % window;
    let mut chunks = Vec::new();
    for w in logs[..cut].chunks(window) {
        chunks.extend(builder.push(w).expect("a window"));
    }
    let mut traces = builder.finish(&logs[cut..]).expect("the last window");
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    with_verifier_files(&program, &mut traces);
    traces
}

/// The verifier's verdict on `proof` under `format` with `checks`.
fn verdict(
    proof: &BlockWhirProof,
    elf: &[u8],
    format: &BlockFormat,
    checks: stark::multilinear_table::batched::VerifierChecks,
) -> Result<Result<(), crate::block_whir::BlockRefusal>, String> {
    crate::block_whir::verify_block_whir_traced(
        proof,
        elf,
        &ProofOptions::default_test_options(),
        format,
        false,
        checks,
    )
    .map_err(|e| format!("{e:?}"))
}

/// The syscall number through the real verifier: a lean block whose ECALL row
/// carries another number than `a7` as its `res` and `rvd` (its dispatch then
/// names another chip's call), proved from the built traces. The verifier
/// refuses it twice over: with the bus balance off, by the constraint
/// sumcheck (the ADD flag's pair); with the constraints off, by the balance
/// (no chip takes that dispatch). With both off it verifies: nothing else
/// reads the forgery. The honest build proves and verifies.
#[test]
fn a_lean_block_whose_ecall_carries_another_number_is_refused_by_the_verifier() {
    use stark::multilinear_table::batched::VerifierChecks;
    let lean = format(CpuRowForm::Lean);
    let elf = asm_elf_bytes("test_commit_4");
    let options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let prove_built = |traces: &mut crate::tables::trace_builder::Traces| {
        let mut stamps = crate::block_whir::BlockStamps::default();
        crate::block_whir::prove_traces(
            &program,
            &elf,
            traces,
            &ProofOptions::default_test_options(),
            &lean,
            &options,
            &Deviations::default(),
            true,
            &|_, _| {},
            &mut stamps,
        )
        .expect("the build proves")
    };
    let mut honest = built(&elf, &lean);
    let proof = prove_built(&mut honest);
    assert!(verifies(&proof, &elf, &lean), "the honest build verifies");

    let mut forged = built(&elf, &lean);
    use crate::tables::cpu::cols;
    use crate::tables::types::FE;
    let (t, row) = forged
        .cpus
        .iter()
        .enumerate()
        .find_map(|(t, table)| {
            (0..table.num_rows())
                .find(|&r| table.get_main(r, cols::ECALL).canonical_u64() == 1)
                .map(|r| (t, r))
        })
        .expect("an ECALL row");
    let table = &mut forged.cpus[t];
    let a7 = table.get_main(row, cols::RV1_0).canonical_u64()
        + (table.get_main(row, cols::RV1_0 + 1).canonical_u64() << 32);
    let other = a7 ^ 1;
    for k in 0..4 {
        table
            .main_table
            .set(row, cols::RES_0 + k, FE::from((other >> (16 * k)) & 0xFFFF));
    }
    table
        .main_table
        .set(row, cols::RVD_0, FE::from(other & 0xFFFF_FFFF));
    table
        .main_table
        .set(row, cols::RVD_0 + 1, FE::from(other >> 32));
    let proof = prove_built(&mut forged);
    let all = verdict(&proof, &elf, &lean, VerifierChecks::ALL);
    let no_balance = verdict(
        &proof,
        &elf,
        &lean,
        VerifierChecks {
            balance: false,
            ..VerifierChecks::ALL
        },
    );
    let no_constraint = verdict(
        &proof,
        &elf,
        &lean,
        VerifierChecks {
            constraint: false,
            ..VerifierChecks::ALL
        },
    );
    let neither = verdict(
        &proof,
        &elf,
        &lean,
        VerifierChecks {
            balance: false,
            constraint: false,
            ..VerifierChecks::ALL
        },
    );
    println!(
        "LEAN ROW FORGED ECALL: every check {all:?} · without the balance {no_balance:?} · \
         without the constraints {no_constraint:?} · without both {neither:?}"
    );
    assert!(matches!(all, Ok(Err(_))), "every check: refused, {all:?}");
    assert!(
        matches!(no_balance, Ok(Err(_))),
        "the constraints refuse it: {no_balance:?}"
    );
    assert!(
        matches!(&no_constraint, Ok(Err(crate::block_whir::BlockRefusal::Proof(e))) if *e == multilinear::Error::BusImbalance),
        "the balance refuses it: {no_constraint:?}"
    );
    assert!(
        matches!(neither, Ok(Ok(()))),
        "the control: nothing else reads it, {neither:?}"
    );
}

/// The whole-run build refuses the lean row: it decodes by today's table.
#[test]
fn the_whole_run_build_refuses_the_lean_row() {
    // Every other form at the whole-run build's own, so the CPU row's refusal
    // is the one that answers.
    let format = BlockFormat {
        registers: crate::tables::register::RegisterFormat::Tokens32,
        keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm::Full,
        memw_aligned: crate::tables::memw_aligned::MemwAlignedOrder::Lt,
        ..format_in(CpuRowForm::Lean, MemoryFormat::Split)
    };
    let options = crate::block_whir::BlockOptions {
        window_log2: None,
        ..super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9)
    };
    let refused = prove_block_whir_with(
        &asm_elf_bytes("sub"),
        &[],
        &ProofOptions::default_test_options(),
        &format,
        &options,
        &Deviations::default(),
    );
    assert!(
        matches!(&refused, Err(crate::Error::Prover(e)) if e.contains("CPU row needs the windowed build")),
        "{:?}",
        refused.as_ref().err()
    );
}

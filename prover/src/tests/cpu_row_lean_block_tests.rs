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
    BlockFormat::production()
        .with_memory(MemoryFormat::Words)
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

//! Wall-clock profile of the prover on ethrex with a single transaction
//! as the guest program. Prints `print_report`'s full phase + per-table
//! breakdown to stderr.
//!
//! Run with:
//!
//! ```text
//! cargo test -p lambda-vm-prover --release --features instruments \
//!     --test profile_trace_expansion -- --ignored --nocapture
//! ```
//!
//! Output goes to stderr; capture with `2>&1 | tee profile.log` if you want it.

#![cfg(feature = "instruments")]

use lambda_vm_prover::prove_with_inputs;

/// Ethrex with one transfer transaction. Heavy enough to make per-phase /
/// per-table timing differences visible but light enough to prove in a
/// reasonable wall-clock on a development server (typically a few minutes).
#[test]
#[ignore = "long-running profile test; run with --ignored --nocapture"]
fn profile_ethrex_single_tx() {
    let _ = env_logger::builder().is_test(true).try_init();
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();

    let elf_bytes =
        std::fs::read(workspace_root.join("executor/program_artifacts/rust/ethrex.elf"))
            .expect("ethrex.elf — run `make ethrex-elf` first");
    let input = std::fs::read(workspace_root.join("executor/tests/ethrex_simple_tx.bin"))
        .expect("ethrex_simple_tx.bin");

    eprintln!("=== profile_ethrex_single_tx ===");
    eprintln!("ELF bytes:   {}", elf_bytes.len());
    eprintln!("Input bytes: {}", input.len());
    eprintln!();

    let proof = prove_with_inputs(&elf_bytes, &input).expect("prove");

    // We don't verify here — the timing report has already been printed by
    // prove_with_options_and_inputs' instruments block. Print one final line
    // so the test output ends with a clear "done" marker.
    eprintln!();
    eprintln!("=== profile_ethrex_single_tx complete ===");
    eprintln!("Public output bytes: {}", proof.public_output.len());
}

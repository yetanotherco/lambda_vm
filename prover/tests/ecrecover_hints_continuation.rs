//! Continuation-proof measurement for the `ecrecover_hints` guest on the hint
//! arena: proves the pass-2 run (arena hints) with continuations to bound
//! prover memory, and verifies the bundle.
//!
//! Fixtures are produced by the executor-side driver — run it first:
//!   cargo test -p executor --test hint_arena_ecrecover -- --ignored --nocapture
//! Then:
//!   cargo test -p lambda-vm-prover --test ecrecover_hints_continuation -- --ignored --nocapture

use lambda_vm_prover::continuation::{prove_continuation, verify_continuation};
use stark::proof::options::ProofOptions;
use std::time::Instant;

/// Epoch size: 2^16 cycles. ~866k cycles → ~14 epochs, bounding per-epoch
/// prover memory.
const EPOCH_SIZE_LOG2: u32 = 16;

#[test]
#[ignore = "measurement — run explicitly"]
fn ecrecover_arena_continuation_proof() {
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let dir = workspace_root.join("executor/program_artifacts/rust");
    let elf_bytes = std::fs::read(dir.join("ecrecover_hints.elf"))
        .expect("ecrecover_hints.elf missing — run `make executor/program_artifacts/rust/ecrecover_hints.elf`");
    let input = std::fs::read(dir.join("ecrecover_hints.input.bin")).expect(
        "ecrecover_hints.input.bin missing — run the executor driver: \
         cargo test -p executor --test hint_arena_ecrecover -- --ignored",
    );
    let hints_bin = std::fs::read(dir.join("ecrecover_hints.hints.bin"))
        .expect("ecrecover_hints.hints.bin missing — run the executor driver");
    assert_eq!(
        hints_bin.len() % 32,
        0,
        "hints fixture must be 32-byte slots"
    );
    let hints: Vec<[u8; 32]> = hints_bin
        .chunks_exact(32)
        .map(|c| c.try_into().unwrap())
        .collect();

    let opts = ProofOptions::default_test_options();

    let t0 = Instant::now();
    let bundle = prove_continuation(&elf_bytes, &input, &hints, EPOCH_SIZE_LOG2, &opts)
        .expect("continuation prove");
    let prove_time = t0.elapsed();

    let t0 = Instant::now();
    let public_output = verify_continuation(&elf_bytes, &bundle, &opts)
        .expect("continuation verify")
        .expect("bundle must verify");
    let verify_time = t0.elapsed();

    println!("[ecrecover-continuation] epochs = {}", bundle.num_epochs());
    println!("[ecrecover-continuation] hints = {} slots", hints.len());
    println!("[ecrecover-continuation] prove = {prove_time:?}, verify = {verify_time:?}");
    println!(
        "[ecrecover-continuation] public output = {} bytes, first 8: {:02x?}",
        public_output.len(),
        &public_output[..8.min(public_output.len())]
    );
}

//! Byte-identical proof test for streaming "retire-traces" mode (C.2b).
//!
//! The decisive correctness invariant: a proof produced with the full streaming
//! mode ON (`LAMBDA_STREAM_LDE=1`, which retires the LDE, Merkle leaves AND the
//! log-derived traces, rebuilding each on demand from a compact routed
//! intermediate) must be byte-identical to one produced with it OFF. C.2a made
//! the trace build deterministic, so the on-demand-rebuilt trace equals the
//! pre-built one and the proof matches exactly. If they differ, the on-demand
//! rebuild has diverged from the pre-built trace.
//!
//! Marked `#[ignore]` because the only execution fixture that runs end-to-end on
//! this branch's executor (`fib_iterative_1200k`) is heavy (~1.2M cycles → many
//! chunks per table). Run explicitly with:
//!   `cargo test -p lambda-vm-prover --lib -- --ignored streaming_retire`

use std::sync::Mutex;

use stark::proof::options::GoldilocksCubicProofOptions;

use crate::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;

/// `LAMBDA_STREAM_LDE` is process-global; serialize the two proving runs.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Prove `elf_bytes` once with streaming `on`/off, returning the serialized
/// `VmProof` bytes. Restores the prior env-var value. Also asserts the proof
/// verifies regardless of which mode produced it.
fn prove_serialized(elf_bytes: &[u8], max_rows: &MaxRowsConfig, on: bool) -> Vec<u8> {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let prev = std::env::var("LAMBDA_STREAM_LDE").ok();

    // SAFETY: single-threaded section guarded by ENV_LOCK; restored below.
    unsafe { std::env::set_var("LAMBDA_STREAM_LDE", if on { "1" } else { "0" }) };

    // Grinding is disabled (factor 0): the parallel grinding nonce search uses
    // `find_any`, which is non-deterministic, so any grinding>0 makes the proof
    // bytes vary run-to-run regardless of streaming mode. With grinding=0 the
    // proof is fully deterministic, isolating the trace-rebuild invariant.
    let options = GoldilocksCubicProofOptions::with_params(2, 128, 0)
        .expect("grinding=0 options are valid");
    let proof = crate::prove_with_options(elf_bytes, &options, max_rows)
        .expect("prove must succeed");

    match prev {
        Some(v) => unsafe { std::env::set_var("LAMBDA_STREAM_LDE", v) },
        None => unsafe { std::env::remove_var("LAMBDA_STREAM_LDE") },
    }

    assert!(
        crate::verify_with_options(&proof, elf_bytes, &options).expect("verify must run"),
        "proof failed to verify"
    );

    bincode::serialize(&proof).expect("serialize VmProof")
}

/// Assert the streaming-ON and streaming-OFF proofs are byte-identical for a
/// given ELF / max_rows configuration.
fn assert_byte_identical(name: &str, max_rows: MaxRowsConfig) {
    let elf_bytes = asm_elf_bytes(name);
    let off = prove_serialized(&elf_bytes, &max_rows, false);
    let on = prove_serialized(&elf_bytes, &max_rows, true);
    assert_eq!(
        off, on,
        "streaming retire-traces proof for `{name}` must be byte-identical to the resident proof"
    );
}

/// Multi-chunk case: many chunks per log-derived table (PAGE interleaved between
/// BRANCH and MEMW_R), exercising the air-order -> (TableKind, chunk) mapping.
#[test]
#[ignore = "heavy execution fixture (~1.2M cycles); run with --ignored"]
fn streaming_retire_proof_is_byte_identical_chunked() {
    assert_byte_identical("fib_iterative_1200k", MaxRowsConfig::default());
}

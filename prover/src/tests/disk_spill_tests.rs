//! End-to-end tests forcing `StorageMode::Disk` via a tiny `max_ram_bytes` cap.
//!
//! These exercise the same code paths the auto-detector would select for a
//! large program on a memory-constrained machine: trace spill, LDE spill, and
//! Merkle-tree spill. We pin the cap to 1 MB so even the smallest ELF crosses
//! the threshold deterministically.

use crate::VmProof;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::test_utils::asm_elf_bytes;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use stark::proof::options::GoldilocksCubicProofOptions;

const FORCE_DISK_CAP: u64 = 1_000_000;

fn options_forcing_disk() -> stark::proof::options::ProofOptions {
    let mut opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid");
    opts.max_ram_bytes = Some(FORCE_DISK_CAP);
    opts
}

/// Prove + verify a small program end-to-end with Disk storage forced.
/// This exercises the full pipeline: trace generation, main-trace spill,
/// LDE spill, Merkle-tree spill, and verification.
#[test]
fn test_disk_spill_prove_and_verify_small() {
    let elf_bytes = asm_elf_bytes("sub");
    let opts = options_forcing_disk();
    let vm_proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::default())
        .expect("prove failed");
    let ok = crate::verify_with_options(&vm_proof, &elf_bytes, &opts).expect("verify failed");
    assert!(ok, "verification returned false");
}

/// Prove + verify with `MaxRowsConfig::small()` (tiny chunks) to force many
/// chunks. Ensures disk-spill works across chunk boundaries where pool
/// buffers are reused and main traces are spilled per-chunk.
#[test]
fn test_disk_spill_prove_and_verify_with_chunks() {
    let elf_bytes = asm_elf_bytes("sub");
    let opts = options_forcing_disk();
    let vm_proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::small())
        .expect("prove failed");
    let ok = crate::verify_with_options(&vm_proof, &elf_bytes, &opts).expect("verify failed");
    assert!(ok, "verification returned false");
}

/// Prove, serialize with bincode, deserialize, then verify.
/// Reproduces the CLI path: prove → write → read → verify.
#[test]
fn test_disk_spill_serialization_roundtrip() {
    let elf_bytes = asm_elf_bytes("sub");
    let opts = options_forcing_disk();
    let proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::default())
        .expect("prove failed");

    let bytes = bincode::serialize(&proof).expect("serialize failed");
    let proof2: VmProof = bincode::deserialize(&bytes).expect("deserialize failed");
    let valid = crate::verify_with_options(&proof2, &elf_bytes, &opts).expect("verify failed");
    assert!(valid, "verification failed after serialization roundtrip");
}

/// Test prove+verify with a larger program (2M instructions).
/// Catches bugs that only manifest at scale (multiple chunks, larger tables).
#[test]
fn test_disk_spill_prove_and_verify_2m() {
    let _ = env_logger::builder().is_test(true).try_init();
    let elf_bytes = asm_elf_bytes("fib_iterative_2M");
    let opts = options_forcing_disk();
    let vm_proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::default())
        .expect("prove failed");
    let ok = crate::verify_with_options(&vm_proof, &elf_bytes, &opts).expect("verify failed");
    assert!(ok, "verification returned false for fib_iterative_2M");
}

/// `PreparedTraceInputs::estimate_main_elements` must match the post-build
/// `Traces::total_field_elements` so the pre-build Disk/Ram decision is based
/// on an honest number. Runs the same program both ways and compares.
#[test]
fn test_estimate_main_elements_matches_built_trace() {
    let elf_bytes = asm_elf_bytes("sub");
    let program = Elf::load(&elf_bytes).expect("elf load");
    let executor = Executor::new(&program, Vec::new()).expect("executor");
    let result = executor.run().expect("run");
    let max_rows = MaxRowsConfig::default();

    let prep =
        Traces::prepare_from_elf_and_logs(&program, &result.logs, &max_rows, &[]).expect("prepare");
    let estimated = prep.estimate_main_elements(&max_rows);

    let traces = prep
        .into_traces(
            Some(&program),
            &max_rows,
            stark::storage_mode::StorageMode::Ram,
        )
        .expect("into_traces");
    let actual = traces.total_field_elements();

    // Estimator omits REGISTER/PAGE contributions. On small programs fixed
    // tables dominate and the relative gap widens; for anything big enough
    // to warrant the disk/ram decision the gap collapses below 1%. Cap at
    // 10% here so we catch a chunked-table omission if one ever sneaks in.
    assert!(
        estimated <= actual,
        "estimator ({estimated}) > actual ({actual})"
    );
    let gap = actual - estimated;
    assert!(
        gap * 10 <= actual,
        "estimator gap {gap} is >10% of actual {actual}"
    );
}

/// Same as roundtrip test but with small chunks.
#[test]
fn test_disk_spill_serialization_roundtrip_chunked() {
    let elf_bytes = asm_elf_bytes("sub");
    let opts = options_forcing_disk();
    let proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::small())
        .expect("prove failed");

    let bytes = bincode::serialize(&proof).expect("serialize failed");
    let proof2: VmProof = bincode::deserialize(&bytes).expect("deserialize failed");
    let valid = crate::verify_with_options(&proof2, &elf_bytes, &opts).expect("verify failed");
    assert!(
        valid,
        "verification failed after serialization roundtrip (chunked)"
    );
}

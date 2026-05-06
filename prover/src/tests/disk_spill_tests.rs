//! End-to-end tests forcing `StorageMode::Disk` via a low `max_ram_bytes` cap.

use crate::VmProof;
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::GoldilocksCubicProofOptions;

const FORCE_DISK_CAP: u64 = 1_000_000;

fn options_forcing_disk() -> stark::proof::options::ProofOptions {
    let mut opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid");
    opts.max_ram_bytes = Some(FORCE_DISK_CAP);
    opts
}

/// Prove + verify a small program with Disk storage forced.
#[test]
fn test_disk_spill_prove_and_verify_small() {
    let elf_bytes = asm_elf_bytes("sub");
    let opts = options_forcing_disk();
    let vm_proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::default())
        .expect("prove failed");
    let ok = crate::verify_with_options(&vm_proof, &elf_bytes, &opts).expect("verify failed");
    assert!(ok, "verification returned false");
}

/// Prove + verify with small chunks to exercise spill across chunk boundaries.
#[test]
fn test_disk_spill_prove_and_verify_with_chunks() {
    let elf_bytes = asm_elf_bytes("sub");
    let opts = options_forcing_disk();
    let vm_proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::small())
        .expect("prove failed");
    let ok = crate::verify_with_options(&vm_proof, &elf_bytes, &opts).expect("verify failed");
    assert!(ok, "verification returned false");
}

/// Prove, serialize, deserialize, verify (CLI roundtrip).
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

/// Prove + verify a 2M-instruction program to catch scale-only bugs.
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

//! End-to-end tests forcing `StorageMode::Disk` via the `FORCE_DISK_SPILL` env var.

use crate::VmProof;
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::GoldilocksCubicProofOptions;

/// RAII guard that sets `FORCE_DISK_SPILL` for the test's scope and clears it
/// on drop. Tests must run with `--test-threads=1`.
struct ForceDiskGuard;

impl ForceDiskGuard {
    fn new() -> Self {
        // SAFETY: tests run with --test-threads=1, no concurrent env access.
        unsafe { std::env::set_var("FORCE_DISK_SPILL", "1") };
        Self
    }
}

impl Drop for ForceDiskGuard {
    fn drop(&mut self) {
        // SAFETY: same as new().
        unsafe { std::env::remove_var("FORCE_DISK_SPILL") };
    }
}

#[test]
fn test_disk_spill_prove_and_verify_small() {
    let _guard = ForceDiskGuard::new();
    let elf_bytes = asm_elf_bytes("sub");
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid");
    let vm_proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::default())
        .expect("prove failed");
    let ok = crate::verify_with_options(&vm_proof, &elf_bytes, &opts).expect("verify failed");
    assert!(ok, "verification returned false");
}

#[test]
fn test_disk_spill_prove_and_verify_with_chunks() {
    let _guard = ForceDiskGuard::new();
    let elf_bytes = asm_elf_bytes("all_instructions_64");
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid");
    let vm_proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::small())
        .expect("prove failed");
    let ok = crate::verify_with_options(&vm_proof, &elf_bytes, &opts).expect("verify failed");
    assert!(ok, "verification returned false");
}

#[test]
fn test_disk_spill_serialization_roundtrip() {
    let _guard = ForceDiskGuard::new();
    let elf_bytes = asm_elf_bytes("sub");
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid");
    let proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::default())
        .expect("prove failed");

    let bytes = bincode::serialize(&proof).expect("serialize failed");
    let proof2: VmProof = bincode::deserialize(&bytes).expect("deserialize failed");
    let valid = crate::verify_with_options(&proof2, &elf_bytes, &opts).expect("verify failed");
    assert!(valid, "verification failed after serialization roundtrip");
}

#[test]
fn test_disk_spill_prove_and_verify_372k() {
    let _guard = ForceDiskGuard::new();
    let elf_bytes = asm_elf_bytes("fib_iterative_372k");
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid");
    let vm_proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::default())
        .expect("prove failed");
    let ok = crate::verify_with_options(&vm_proof, &elf_bytes, &opts).expect("verify failed");
    assert!(ok, "verification returned false for fib_iterative_372k");
}

#[test]
fn test_disk_spill_serialization_roundtrip_chunked() {
    let _guard = ForceDiskGuard::new();
    let elf_bytes = asm_elf_bytes("all_instructions_64");
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid");
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

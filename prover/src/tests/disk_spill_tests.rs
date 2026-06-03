//! End-to-end tests forcing `StorageMode::Disk` via the `FORCE_DISK_SPILL` env var.
//!
//! Run with `FORCE_DISK_SPILL=1` set in the environment, e.g.
//! `FORCE_DISK_SPILL=1 cargo test --features disk-spill disk_spill`. Tests
//! fail fast if the var is unset to avoid silent loss of coverage.

use crate::VmProof;
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::GoldilocksCubicProofOptions;

fn require_force_disk_spill() {
    assert_eq!(
        std::env::var("FORCE_DISK_SPILL").as_deref(),
        Ok("1"),
        "set FORCE_DISK_SPILL=1 before running disk-spill tests",
    );
}

#[test]
fn test_disk_spill_prove_verify_and_roundtrip_small() {
    require_force_disk_spill();
    let elf_bytes = asm_elf_bytes("sub");
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid");
    let proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::default())
        .expect("prove failed");
    assert!(
        crate::verify_with_options(&proof, &elf_bytes, &opts, None).expect("verify failed"),
        "verification returned false"
    );

    let bytes = bincode::serialize(&proof).expect("serialize failed");
    let proof2: VmProof = bincode::deserialize(&bytes).expect("deserialize failed");
    assert!(
        crate::verify_with_options(&proof2, &elf_bytes, &opts, None).expect("verify failed"),
        "verification failed after serialization roundtrip"
    );
}

#[test]
fn test_disk_spill_prove_verify_and_roundtrip_chunked() {
    require_force_disk_spill();
    let elf_bytes = asm_elf_bytes("all_instructions_64");
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid");
    let proof = crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::small())
        .expect("prove failed");
    assert!(
        crate::verify_with_options(&proof, &elf_bytes, &opts, None).expect("verify failed"),
        "verification returned false"
    );

    let bytes = bincode::serialize(&proof).expect("serialize failed");
    let proof2: VmProof = bincode::deserialize(&bytes).expect("deserialize failed");
    assert!(
        crate::verify_with_options(&proof2, &elf_bytes, &opts, None).expect("verify failed"),
        "verification failed after serialization roundtrip (chunked)"
    );
}

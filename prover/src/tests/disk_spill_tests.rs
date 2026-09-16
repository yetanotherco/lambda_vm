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
        crate::verify_with_options(&proof, &elf_bytes, &opts, None, None).expect("verify failed"),
        "verification returned false"
    );

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serialize failed");
    let proof2: VmProof =
        rkyv::from_bytes::<VmProof, rkyv::rancor::Error>(&bytes).expect("deserialize failed");
    assert!(
        crate::verify_with_options(&proof2, &elf_bytes, &opts, None, None).expect("verify failed"),
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
        crate::verify_with_options(&proof, &elf_bytes, &opts, None, None).expect("verify failed"),
        "verification returned false"
    );

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serialize failed");
    let proof2: VmProof =
        rkyv::from_bytes::<VmProof, rkyv::rancor::Error>(&bytes).expect("deserialize failed");
    assert!(
        crate::verify_with_options(&proof2, &elf_bytes, &opts, None, None).expect("verify failed"),
        "verification failed after serialization roundtrip (chunked)"
    );
}

/// The five SHA-256 tables are fixed-size, so `chunk_and_generate` never sees
/// them and the fixed-table spill block covers only bitwise/decode/commit/
/// register/halt/pages. ROTXOR is 224 rows of 197 columns per compression call,
/// which makes it the largest thing a SHA-heavy run holds — left on the heap it
/// defeats the point of selecting disk mode. Builds traces directly rather than
/// through `prove`, so it needs no `FORCE_DISK_SPILL`.
#[test]
fn sha256_traces_spill_in_disk_mode() {
    use crate::tables::trace_builder::Traces;
    use crate::test_utils::run_asm_elf;
    use stark::storage_mode::StorageMode;

    let (elf, logs, _) = run_asm_elf("test_sha256_overlap");
    let traces = Traces::from_elf_and_logs(
        &elf,
        &logs,
        &MaxRowsConfig::default(),
        &[],
        StorageMode::Disk,
    )
    .expect("trace build failed");

    for (name, table) in [
        ("sha256", &traces.sha256),
        ("sha256_round", &traces.sha256_round),
        ("sha256_schedule", &traces.sha256_schedule),
        ("sha256_rotxor", &traces.sha256_rotxor),
        ("sha256_k", &traces.sha256_k),
    ] {
        assert!(
            table.main_table.is_spilled(),
            "{name} stayed on the heap in disk mode",
        );
    }
}

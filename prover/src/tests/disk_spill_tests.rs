//! Tests for the `disk-spill` feature.
//!
//! Verifies that proving and verification produce correct results when main
//! traces, LDE columns, and Merkle tree nodes are spilled to disk via mmap.

use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;
use crate::VmProof;

/// Prove + verify a small program end-to-end with disk-spill enabled.
/// This exercises the full pipeline: trace generation, main-trace spill,
/// LDE spill, Merkle-tree spill, and verification.
#[test]
fn test_disk_spill_prove_and_verify_small() {
    let elf_bytes = asm_elf_bytes("sub");
    let result = crate::prove_and_verify(&elf_bytes);
    assert!(
        result.is_ok(),
        "prove_and_verify failed: {:?}",
        result.err()
    );
    assert!(result.unwrap(), "verification returned false");
}

/// Prove + verify with `MaxRowsConfig::small()` (2^5 = 32 rows per chunk)
/// to force many chunks. This ensures disk-spill works across chunk boundaries
/// where pool buffers are reused and main traces are spilled per-chunk.
#[test]
fn test_disk_spill_prove_and_verify_with_chunks() {
    let elf_bytes = asm_elf_bytes("sub");
    let proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .expect("blowup=2 is always valid");
    let vm_proof = crate::prove_with_options(&elf_bytes, &proof_options, &MaxRowsConfig::small());
    assert!(
        vm_proof.is_ok(),
        "prove_with_options failed: {:?}",
        vm_proof.err()
    );
    let vm_proof = vm_proof.unwrap();

    let ok = crate::verify_with_options(&vm_proof, &elf_bytes, &proof_options);
    assert!(ok.is_ok(), "verify_with_options failed: {:?}", ok.err());
    assert!(ok.unwrap(), "verification returned false");
}

/// Prove, serialize with bincode, deserialize, then verify.
/// This reproduces the exact CLI path: prove → write → read → verify.
#[test]
fn test_disk_spill_serialization_roundtrip() {
    let elf_bytes = asm_elf_bytes("sub");
    let proof = crate::prove(&elf_bytes).expect("prove failed");

    let bytes = bincode::serialize(&proof).expect("serialize failed");
    eprintln!("Proof serialized: {} bytes", bytes.len());

    let proof2: VmProof = bincode::deserialize(&bytes).expect("deserialize failed");
    let valid = crate::verify(&proof2, &elf_bytes).expect("verify failed");
    assert!(valid, "verification failed after serialization roundtrip");
}

/// Print struct sizes to verify memory analysis
#[test]
fn test_print_struct_sizes() {
    use std::mem::size_of;
    eprintln!("CpuOperation:     {} bytes", size_of::<crate::tables::cpu::CpuOperation>());
    eprintln!("MemwOperation:    {} bytes", size_of::<crate::tables::memw::MemwOperation>());
    eprintln!("LtOperation:      {} bytes", size_of::<crate::tables::lt::LtOperation>());
    eprintln!("BranchOperation:  {} bytes", size_of::<crate::tables::branch::BranchOperation>());
    eprintln!("BitwiseOperation: {} bytes", size_of::<crate::tables::bitwise::BitwiseOperation>());
    eprintln!("ShiftOperation:   {} bytes", size_of::<crate::tables::shift::ShiftOperation>());
}

/// Test prove+verify with a larger program (2M instructions).
/// This catches bugs that only manifest at scale (multiple chunks, larger tables).
#[test]
fn test_disk_spill_prove_and_verify_2m() {
    let _ = env_logger::builder().is_test(true).try_init();
    let elf_bytes = asm_elf_bytes("fib_iterative_2M");
    let result = crate::prove_and_verify(&elf_bytes).expect("prove_and_verify failed");
    assert!(result, "verification returned false for fib_iterative_2M");
}

/// Same as above but with small chunks (MaxRowsConfig::small()).
#[test]
fn test_disk_spill_serialization_roundtrip_chunked() {
    let elf_bytes = asm_elf_bytes("sub");
    let opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .expect("blowup=2 is always valid");
    let proof =
        crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::small()).expect("prove failed");

    let bytes = bincode::serialize(&proof).expect("serialize failed");
    eprintln!("Chunked proof serialized: {} bytes", bytes.len());

    let proof2: VmProof = bincode::deserialize(&bytes).expect("deserialize failed");
    let valid = crate::verify_with_options(&proof2, &elf_bytes, &opts).expect("verify failed");
    assert!(valid, "verification failed after serialization roundtrip (chunked)");
}

//! Tests for the `disk-spill` feature.
//!
//! Verifies that proving and verification produce correct results when main
//! traces, LDE columns, and Merkle tree nodes are spilled to disk via mmap.

use crate::test_utils::asm_elf_bytes;
use crate::tables::MaxRowsConfig;

/// Prove + verify a small program end-to-end with disk-spill enabled.
/// This exercises the full pipeline: trace generation, main-trace spill,
/// LDE spill, Merkle-tree spill, and verification.
#[test]
fn test_disk_spill_prove_and_verify_small() {
    let elf_bytes = asm_elf_bytes("sub");
    let result = crate::prove_and_verify(&elf_bytes);
    assert!(result.is_ok(), "prove_and_verify failed: {:?}", result.err());
    assert!(result.unwrap(), "verification returned false");
}

/// Prove + verify with `MaxRowsConfig::small()` (2^5 = 32 rows per chunk)
/// to force many chunks. This ensures disk-spill works across chunk boundaries
/// where pool buffers are reused and main traces are spilled per-chunk.
#[test]
fn test_disk_spill_prove_and_verify_with_chunks() {
    let elf_bytes = asm_elf_bytes("sub");
    let proof_options =
        stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
            .expect("blowup=2 is always valid");
    let vm_proof =
        crate::prove_with_options(&elf_bytes, &proof_options, &MaxRowsConfig::small());
    assert!(vm_proof.is_ok(), "prove_with_options failed: {:?}", vm_proof.err());
    let vm_proof = vm_proof.unwrap();

    let ok = crate::verify_with_options(&vm_proof, &elf_bytes, &proof_options);
    assert!(ok.is_ok(), "verify_with_options failed: {:?}", ok.err());
    assert!(ok.unwrap(), "verification returned false");
}

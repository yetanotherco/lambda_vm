//! Tests for [`crate::VmVerifyingKey`] and the vkey-aware verify path.

use executor::elf::Elf;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

use crate::VmVerifyingKey;
use crate::test_utils::asm_elf_bytes;
use crate::vkey::VKEY_VERSION;

fn default_options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid")
}

#[test]
fn test_vkey_roundtrip() {
    let elf_bytes = asm_elf_bytes("sub");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();

    let vkey = VmVerifyingKey::from_elf_and_options(&elf, &options);
    assert_eq!(vkey.version, VKEY_VERSION, "version field must be set");
    let digest_before = vkey.compute_digest();

    // Two host derivations on the same inputs must produce the same vkey;
    // the BITWISE_COMMITMENT cache should not change between calls.
    let vkey_again = VmVerifyingKey::from_elf_and_options(&elf, &options);
    assert_eq!(vkey, vkey_again, "vkey derivation must be deterministic");

    // postcard round-trip preserves every field.
    let encoded = postcard::to_allocvec(&vkey).expect("postcard encode");
    let decoded: VmVerifyingKey = postcard::from_bytes(&encoded).expect("postcard decode");
    assert_eq!(vkey, decoded, "postcard round-trip must preserve the vkey");
    assert_eq!(
        decoded.compute_digest(),
        digest_before,
        "digest must be stable across serialization"
    );
}

#[test]
fn test_vkey_verify_equivalence() {
    // Prove a tiny program once with the full (non-minimal) bitwise table,
    // then verify it both ways: with and without a precomputed vkey.
    // Both paths must accept the proof. This is the core correctness
    // guarantee — the vkey shortcut produces identical results to the
    // recompute-from-scratch path.
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = crate::prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let vkey = VmVerifyingKey::from_elf_and_options(&elf, &options);

    let baseline = crate::verify_with_options(&vm_proof, &elf_bytes, &options)
        .expect("baseline verify errored");
    assert!(baseline, "baseline verify must accept the proof");

    let with_vkey =
        crate::verify_with_options_with_vkey(&vm_proof, &elf_bytes, &options, Some(&vkey))
            .expect("vkey verify errored");
    assert!(with_vkey, "vkey verify must accept the same proof");
}

#[test]
fn test_vkey_mismatch_rejects() {
    // Tamper with vkey.bitwise. Without an explicit `vk_digest` field on
    // VmProof (deferred to a later PR), rejection comes from Fiat-Shamir:
    // the verifier feeds the tampered commitment into the transcript,
    // derives different challenges from what the prover used, and the
    // proof's openings stop matching.
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = crate::prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let mut vkey = VmVerifyingKey::from_elf_and_options(&elf, &options);

    vkey.bitwise[0] ^= 0xFF;

    let result = crate::verify_with_options_with_vkey(&vm_proof, &elf_bytes, &options, Some(&vkey))
        .expect("verify must not return Err — Fiat-Shamir mismatch is Ok(false)");
    assert!(!result, "tampered bitwise commitment must cause rejection");
}

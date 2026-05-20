//! Tests for [`crate::VmVerifyingKey`] and the vkey-aware verify path.

use executor::elf::Elf;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

use crate::VmVerifyingKey;
use crate::tables::page::PageConfig;
use crate::tables::trace_builder::Traces;
use crate::test_utils::asm_elf_bytes;
use crate::vkey::VKEY_VERSION;
use crate::{VmProof, prove};

fn default_options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid")
}

/// Derive the same `page_configs` slice the verifier would reconstruct from
/// `vm_proof`. This is exactly what `verify_with_options_with_vkey` does
/// internally, lifted into the test so the test-side and verifier-side
/// `vkey.pages` indexing line up.
fn page_configs_from_proof(elf: &Elf, vm_proof: &VmProof) -> Vec<PageConfig> {
    Traces::page_configs_from_elf_and_runtime(
        elf,
        &vm_proof.runtime_page_ranges,
        vm_proof.num_private_input_pages,
    )
}

#[test]
fn test_vkey_roundtrip() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let page_configs = page_configs_from_proof(&elf, &vm_proof);

    let vkey = VmVerifyingKey::from_elf_and_options(&elf, &options, &page_configs);
    assert_eq!(vkey.version, VKEY_VERSION, "version field must be set");
    assert_eq!(
        vkey.pages.len(),
        page_configs.len(),
        "vkey.pages must have one entry per page config",
    );
    let digest_before = vkey.compute_digest();

    // Two host derivations on the same inputs must produce the same vkey;
    // the per-table commitment caches should not change between calls.
    let vkey_again = VmVerifyingKey::from_elf_and_options(&elf, &options, &page_configs);
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
    let vm_proof = prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let page_configs = page_configs_from_proof(&elf, &vm_proof);
    let vkey = VmVerifyingKey::from_elf_and_options(&elf, &options, &page_configs);

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
    let vm_proof = prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let page_configs = page_configs_from_proof(&elf, &vm_proof);
    let mut vkey = VmVerifyingKey::from_elf_and_options(&elf, &options, &page_configs);

    vkey.bitwise[0] ^= 0xFF;

    let result = crate::verify_with_options_with_vkey(&vm_proof, &elf_bytes, &options, Some(&vkey))
        .expect("verify must not return Err — Fiat-Shamir mismatch is Ok(false)");
    assert!(!result, "tampered bitwise commitment must cause rejection");
}

#[test]
fn test_vkey_page_mismatch_rejects() {
    // Same shape as `test_vkey_mismatch_rejects`, but tampers with the page
    // table that gets it first non-private-input slot. Fiat-Shamir rejects
    // the same way: the page commitment is in the verifier's transcript
    // exactly like the bitwise one.
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let page_configs = page_configs_from_proof(&elf, &vm_proof);
    let mut vkey = VmVerifyingKey::from_elf_and_options(&elf, &options, &page_configs);

    let target = page_configs
        .iter()
        .position(|c| !c.is_private_input)
        .expect("test ELF must produce at least one non-private-input page");
    vkey.pages[target][0] ^= 0xFF;

    let result = crate::verify_with_options_with_vkey(&vm_proof, &elf_bytes, &options, Some(&vkey))
        .expect("verify must not return Err — Fiat-Shamir mismatch is Ok(false)");
    assert!(!result, "tampered page commitment must cause rejection");
}

//! Tests for [`crate::VmVerifyingKey`] and the vkey-aware verify path.

use executor::elf::Elf;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

use crate::VmVerifyingKey;
use crate::tables::page::PageConfig;
use crate::tables::trace_builder::Traces;
use crate::test_utils::asm_elf_bytes;
use crate::vkey::VKEY_VERSION;
use crate::{Error, VmProof, prove};

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

/// Prove `program`, and derive the honest vkey the same way the verifier would.
fn proof_and_vkey_for(program: &str) -> (Vec<u8>, VmProof, ProofOptions, VmVerifyingKey) {
    let elf_bytes = asm_elf_bytes(program);
    let vm_proof = prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let page_configs = page_configs_from_proof(&elf, &vm_proof);
    let vkey = VmVerifyingKey::from_elf_and_options(&elf, &options, None, &page_configs);
    (elf_bytes, vm_proof, options, vkey)
}

/// Prove `sub`, and derive the honest vkey the same way the verifier would.
fn proof_and_vkey() -> (Vec<u8>, VmProof, ProofOptions, VmVerifyingKey) {
    proof_and_vkey_for("sub")
}

/// Build the `RecursionCommitment` the guest would commit for a proof.
fn commitment_for(
    elf_bytes: &[u8],
    vm_proof: &VmProof,
    options: &ProofOptions,
) -> crate::RecursionCommitment {
    crate::RecursionCommitment {
        elf_digest: crate::elf_digest(elf_bytes),
        vk_digest: vm_proof.vk_digest,
        options: options.clone(),
        table_counts: vm_proof.table_counts.clone(),
        num_private_input_pages: vm_proof.num_private_input_pages,
        runtime_page_ranges: vm_proof.runtime_page_ranges.clone(),
        public_output: vm_proof.public_output.clone(),
    }
}

/// A tampered or malformed vkey must be rejected with an explicit
/// `InvalidVerifyingKey` before any STARK work runs — either by the shape
/// checks or by the `vk_digest` comparison against the proof.
fn assert_rejects_vkey(
    elf_bytes: &[u8],
    vm_proof: &VmProof,
    options: &ProofOptions,
    vkey: &VmVerifyingKey,
    what: &str,
) {
    let result =
        crate::verify_with_options_with_vkey(vm_proof, elf_bytes, options, None, None, Some(vkey));
    assert!(
        matches!(result, Err(Error::InvalidVerifyingKey(_))),
        "{what} must be rejected with InvalidVerifyingKey, got {result:?}"
    );
}

#[test]
fn test_vkey_roundtrip() {
    let (_, vm_proof, options, vkey) = proof_and_vkey();
    let elf_bytes = asm_elf_bytes("sub");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let page_configs = page_configs_from_proof(&elf, &vm_proof);

    assert_eq!(vkey.version, VKEY_VERSION, "version field must be set");
    assert_eq!(vkey.options, options, "options must be embedded");
    assert_eq!(
        vkey.pages.len(),
        page_configs.len(),
        "vkey.pages must have one entry per page config",
    );
    let digest_before = vkey.compute_digest();
    assert_eq!(
        vm_proof.vk_digest, digest_before,
        "prover must stamp the same digest the verifier derives"
    );

    // Two host derivations on the same inputs must produce the same vkey;
    // the per-table commitment caches should not change between calls.
    let vkey_again = VmVerifyingKey::from_elf_and_options(&elf, &options, None, &page_configs);
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
    let (elf_bytes, vm_proof, options, vkey) = proof_and_vkey();

    let baseline = crate::verify_with_options(&vm_proof, &elf_bytes, &options, None, None)
        .expect("baseline verify errored");
    assert!(baseline, "baseline verify must accept the proof");

    let with_vkey = crate::verify_with_options_with_vkey(
        &vm_proof,
        &elf_bytes,
        &options,
        None,
        None,
        Some(&vkey),
    )
    .expect("vkey verify errored");
    assert!(with_vkey, "vkey verify must accept the same proof");
}

#[test]
fn test_vkey_mismatch_rejects() {
    let (elf_bytes, vm_proof, options, mut vkey) = proof_and_vkey();
    vkey.bitwise[0] ^= 0xFF;
    assert_rejects_vkey(&elf_bytes, &vm_proof, &options, &vkey, "tampered bitwise");
}

#[test]
fn test_vkey_page_mismatch_rejects() {
    let (elf_bytes, vm_proof, options, mut vkey) = proof_and_vkey();
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let target = page_configs_from_proof(&elf, &vm_proof)
        .iter()
        .position(|c| !c.is_private_input)
        .expect("test ELF must produce at least one non-private-input page");
    vkey.pages[target][0] ^= 0xFF;
    assert_rejects_vkey(&elf_bytes, &vm_proof, &options, &vkey, "tampered page");
}

#[test]
fn test_vkey_zero_init_slot_enforced() {
    // A zero-init page's vkey slot is ignored by `new_with_vkey` (the verifier
    // derives it locally), yet `compute_digest` still hashes it. Without an
    // explicit per-slot check, a slot could differ from the value the verifier
    // uses while the digest still matches — the gap that lets a program be
    // reclassified under an unchanged identity. Re-stamp `vk_digest` so the
    // digest check passes, and confirm the per-slot check rejects anyway.
    // `sub` touches no runtime pages; use a program that exercises the stack so
    // `page_configs` contains a zero-init page.
    let (elf_bytes, mut vm_proof, options, mut vkey) = proof_and_vkey_for("deep_stack");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let target = page_configs_from_proof(&elf, &vm_proof)
        .iter()
        .position(|c| !c.is_private_input && c.init_values.is_none())
        .expect("test ELF must produce at least one zero-init page (the stack)");
    vkey.pages[target] = [0xAB; 32];
    vm_proof.vk_digest = vkey.compute_digest();
    assert_rejects_vkey(&elf_bytes, &vm_proof, &options, &vkey, "tampered zero-init slot");
}

#[test]
fn test_vkey_decode_mismatch_rejects() {
    let (elf_bytes, vm_proof, options, mut vkey) = proof_and_vkey();
    vkey.decode[0] ^= 0xFF;
    assert_rejects_vkey(&elf_bytes, &vm_proof, &options, &vkey, "tampered decode");
}

#[test]
fn test_vkey_register_mismatch_rejects() {
    let (elf_bytes, vm_proof, options, mut vkey) = proof_and_vkey();
    vkey.register[0] ^= 0xFF;
    assert_rejects_vkey(&elf_bytes, &vm_proof, &options, &vkey, "tampered register");
}

#[test]
fn test_vkey_keccak_rc_mismatch_rejects() {
    let (elf_bytes, vm_proof, options, mut vkey) = proof_and_vkey();
    vkey.keccak_rc[0] ^= 0xFF;
    assert_rejects_vkey(&elf_bytes, &vm_proof, &options, &vkey, "tampered keccak_rc");
}

#[test]
fn test_vkey_short_pages_rejects() {
    // A short pages vec must be a clean error, not an out-of-bounds panic.
    let (elf_bytes, vm_proof, options, mut vkey) = proof_and_vkey();
    vkey.pages.clear();
    assert_rejects_vkey(&elf_bytes, &vm_proof, &options, &vkey, "short pages vec");
}

#[test]
fn test_vkey_options_mismatch_rejects() {
    // Query count and grinding factor affect soundness but no commitment,
    // so a weakened-options vkey must be caught by the explicit check.
    let (elf_bytes, vm_proof, options, mut vkey) = proof_and_vkey();
    vkey.options.fri_number_of_queries = 1;
    assert_rejects_vkey(&elf_bytes, &vm_proof, &options, &vkey, "weakened options");
}

#[test]
fn test_vkey_wrong_version_rejects() {
    let (elf_bytes, vm_proof, options, mut vkey) = proof_and_vkey();
    vkey.version = VKEY_VERSION - 1;
    assert_rejects_vkey(&elf_bytes, &vm_proof, &options, &vkey, "wrong version");
}

fn assert_rejects_commitment(
    commitment: &crate::RecursionCommitment,
    trusted_elf: &[u8],
    options: &ProofOptions,
    what: &str,
) {
    let result = crate::verify_recursion_commitment(commitment, trusted_elf, options);
    assert!(
        matches!(result, Err(Error::InvalidVerifyingKey(_))),
        "{what} must be rejected with InvalidVerifyingKey, got {result:?}"
    );
}

#[test]
fn test_recursion_commitment_accepts_honest() {
    let (elf_bytes, vm_proof, options, _vkey) = proof_and_vkey();
    let commitment = commitment_for(&elf_bytes, &vm_proof, &options);
    let out = crate::verify_recursion_commitment(&commitment, &elf_bytes, &options)
        .expect("honest commitment must be accepted");
    assert_eq!(out, vm_proof.public_output);
}

#[test]
fn test_recursion_commitment_forged_program_rejected() {
    // The core recursion soundness check: a prover commits `elf_digest` for the
    // trusted program `sub` but a `vk_digest` derived from a *different* program
    // (`deep_stack`). Re-deriving the canonical vkey from the trusted ELF yields
    // a different digest, so the substitution is caught.
    let (elf_bytes, vm_proof, options, _vkey) = proof_and_vkey();
    let (_other_elf, other_proof, _other_opts, _other_vkey) = proof_and_vkey_for("deep_stack");
    let mut commitment = commitment_for(&elf_bytes, &vm_proof, &options);
    commitment.vk_digest = other_proof.vk_digest;
    assert_rejects_commitment(&commitment, &elf_bytes, &options, "forged program vk_digest");
}

#[test]
fn test_recursion_commitment_wrong_trusted_elf_rejected() {
    let (elf_bytes, vm_proof, options, _vkey) = proof_and_vkey();
    let commitment = commitment_for(&elf_bytes, &vm_proof, &options);
    let wrong_elf = asm_elf_bytes("deep_stack");
    assert_rejects_commitment(&commitment, &wrong_elf, &options, "wrong trusted ELF");
}

#[test]
fn test_recursion_commitment_weak_options_rejected() {
    let (elf_bytes, vm_proof, options, _vkey) = proof_and_vkey();
    let mut commitment = commitment_for(&elf_bytes, &vm_proof, &options);
    commitment.options.fri_number_of_queries = 1;
    assert_rejects_commitment(&commitment, &elf_bytes, &options, "weakened options");
}

//! Tests for [`crate::VmVerifyingKey`] and the vkey-aware verify path.

use executor::elf::Elf;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::traits::AIR;

use crate::VmVerifyingKey;
use crate::tables::page::PageConfig;
use crate::tables::trace_builder::Traces;
use crate::test_utils::asm_elf_bytes;
use crate::vkey::VKEY_VERSION;
use crate::{VmAirs, VmProof, prove};

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
    // Same shape as `test_vkey_mismatch_rejects`, but tampers with the
    // first non-private-input slot of the page table. Fiat-Shamir rejects
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

#[test]
fn test_vkey_decode_mismatch_rejects() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let page_configs = page_configs_from_proof(&elf, &vm_proof);
    let mut vkey = VmVerifyingKey::from_elf_and_options(&elf, &options, &page_configs);

    vkey.decode[0] ^= 0xFF;

    let result = crate::verify_with_options_with_vkey(&vm_proof, &elf_bytes, &options, Some(&vkey))
        .expect("verify must not return Err — Fiat-Shamir mismatch is Ok(false)");
    assert!(!result, "tampered decode commitment must cause rejection");
}

#[test]
fn test_vkey_register_mismatch_rejects() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let page_configs = page_configs_from_proof(&elf, &vm_proof);
    let mut vkey = VmVerifyingKey::from_elf_and_options(&elf, &options, &page_configs);

    vkey.register[0] ^= 0xFF;

    let result = crate::verify_with_options_with_vkey(&vm_proof, &elf_bytes, &options, Some(&vkey))
        .expect("verify must not return Err — Fiat-Shamir mismatch is Ok(false)");
    assert!(!result, "tampered register commitment must cause rejection");
}

#[test]
fn test_vkey_keccak_rc_mismatch_rejects() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let page_configs = page_configs_from_proof(&elf, &vm_proof);
    let mut vkey = VmVerifyingKey::from_elf_and_options(&elf, &options, &page_configs);

    vkey.keccak_rc[0] ^= 0xFF;

    let result = crate::verify_with_options_with_vkey(&vm_proof, &elf_bytes, &options, Some(&vkey))
        .expect("verify must not return Err — Fiat-Shamir mismatch is Ok(false)");
    assert!(
        !result,
        "tampered keccak_rc commitment must cause rejection"
    );
}

#[test]
fn test_vkey_version_mismatch_today_accepts() {
    // Today `version` is advisory only — the verifier never reads it, so
    // mutating it should not affect verification. This test pins that
    // behavior so the deferred `vk_digest` PR (which will start digesting
    // `version` into the digest and enforcing it at verify time) has to
    // flip this assertion as a conscious choice.
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let page_configs = page_configs_from_proof(&elf, &vm_proof);
    let mut vkey = VmVerifyingKey::from_elf_and_options(&elf, &options, &page_configs);

    vkey.version = vkey.version.wrapping_add(7);

    let result = crate::verify_with_options_with_vkey(&vm_proof, &elf_bytes, &options, Some(&vkey))
        .expect("verify must not return Err");
    assert!(
        result,
        "today `version` is advisory — once vk_digest lands this assertion must flip"
    );
}

#[test]
fn test_vkey_empty_pages_falls_back_to_recompute() {
    // Pin the silent fallback in `VmAirs::new_with_vkey`'s page loop: when
    // `vkey.pages` is shorter than `page_configs`, the loop recomputes the
    // missing slots instead of panicking. A future tightening to return
    // `Err` would be a conscious break of this test.
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let page_configs = page_configs_from_proof(&elf, &vm_proof);
    let mut vkey = VmVerifyingKey::from_elf_and_options(&elf, &options, &page_configs);

    vkey.pages.clear();

    let result = crate::verify_with_options_with_vkey(&vm_proof, &elf_bytes, &options, Some(&vkey))
        .expect("verify must not return Err");
    assert!(
        result,
        "empty vkey.pages must fall back to recomputing page commitments and still accept the proof"
    );
}

#[test]
fn test_vkey_fields_match_air_commitments() {
    // Sharper version of `test_vkey_verify_equivalence`: directly assert
    // each cached commitment matches what `VmAirs::new` constructs from the
    // same elf + options. Catches "host helper diverges from AIR
    // construction" bugs explicitly rather than via Fiat-Shamir failure.
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("inner prove should succeed");
    let elf = Elf::load(&elf_bytes).expect("ELF load failed");
    let options = default_options();
    let page_configs = page_configs_from_proof(&elf, &vm_proof);
    let vkey = VmVerifyingKey::from_elf_and_options(&elf, &options, &page_configs);

    let airs = VmAirs::new(&elf, &options, false, &page_configs, &vm_proof.table_counts);

    assert!(
        airs.bitwise.is_preprocessed(),
        "bitwise AIR should be preprocessed"
    );
    assert_eq!(vkey.bitwise, airs.bitwise.precomputed_commitment());
    assert_eq!(vkey.decode, airs.decode.precomputed_commitment());
    assert_eq!(vkey.register, airs.register.precomputed_commitment());
    assert_eq!(vkey.keccak_rc, airs.keccak_rc.precomputed_commitment());

    assert_eq!(
        vkey.pages.len(),
        airs.pages.len(),
        "vkey.pages and airs.pages must have the same length",
    );
    for (i, (page_config, page_air)) in page_configs.iter().zip(airs.pages.iter()).enumerate() {
        if page_config.is_private_input {
            continue;
        }
        assert_eq!(
            vkey.pages[i],
            page_air.precomputed_commitment(),
            "vkey.pages[{i}] must match AIR commitment for non-private-input page",
        );
    }
}

//! Host-side tests for the Keccak-256 sponge (`keccak256_with_permute`),
//! driving it with the trusted `keccak` crate's f1600 permutation and
//! cross-checking against ethrex's reference `keccak_hash`.

use crate::*;

/// Cross-check our sponge body against the trusted `keccak` crate's f1600.
fn check_keccak(input: &[u8]) {
    let got = keccak256_with_permute(input, keccak::f1600);
    let want = keccak_hash(input);
    assert_eq!(got, want, "keccak256 mismatch for {}-byte input", input.len());
}

#[test]
fn keccak_sponge_matches_trusted_permutation() {
    // Empty input.
    check_keccak(&[]);
    // One byte.
    check_keccak(&[0xab]);
    // 135 bytes — RATE-1: padding lands on byte 135 (0x01) and byte 135 is
    // also the last byte (0x80), so both bits land on the same byte: 0x81.
    check_keccak(&[0x5a; 135]);
    // Exactly RATE (136): fills one full block, final block is all-padding.
    check_keccak(&[0x3c; 136]);
    // RATE+1: one full block + one-byte remainder.
    check_keccak(&[0x7e; 137]);
    // Multi-block: ~1.5 × RATE (200 bytes), deterministic pattern.
    let long: Vec<u8> = (0u8..200).collect();
    check_keccak(&long);
}

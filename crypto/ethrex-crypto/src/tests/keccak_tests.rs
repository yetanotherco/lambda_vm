//! Host-side tests for the Keccak-256 sponge (`keccak256_with_permute`),
//! driving it with the trusted `keccak` crate's f1600 permutation and
//! cross-checking against ethrex's reference `keccak_hash`.

use crate::*;

/// Cross-check our sponge body against the trusted `keccak` crate's f1600.
fn check_keccak(input: &[u8]) {
    let got = keccak256_with_permute(input, keccak::f1600);
    let want = keccak_hash(input);
    assert_eq!(
        got,
        want,
        "keccak256 mismatch for {}-byte input",
        input.len()
    );
}

/// Cross-check our sponge against a hardcoded vector from the Ethereum spec.
fn check_keccak_kat(input: &[u8], expected_hex: &str) {
    let expected: Vec<u8> = (0..expected_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&expected_hex[i..i + 2], 16).unwrap())
        .collect();
    let got = keccak256_with_permute(input, keccak::f1600);
    assert_eq!(
        got.as_ref(),
        expected.as_slice(),
        "KAT mismatch for {}-byte input",
        input.len()
    );
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
    // 2 × RATE (272 bytes): two full absorb blocks + all-padding final block.
    check_keccak(&[0xaa; 272]);
    // 2 × RATE - 1 (271 bytes): two full absorbs + one-byte remainder.
    check_keccak(&[0xbb; 271]);
}

/// Software mirror of the `ECALL -4` sponge-absorb accelerator, matching the
/// executor arm (`executor/src/vm/instruction/execution.rs`,
/// `SyscallNumbers::KeccakAbsorbBlocks`): per block, XOR 17 little-endian dword
/// lanes into the state, then permute. Lets the host test the *composition* the
/// guest uses — which blocks go to the chip, and where padding lands — without
/// a VM.
fn absorb_whole_blocks_mirror(state: &mut [u64; 25], blocks: &[u8]) -> usize {
    const RATE: usize = 136;
    assert_eq!(blocks.len() % RATE, 0);
    let n_blocks = blocks.len() / RATE;
    for k in 0..n_blocks {
        absorb_block(state, &blocks[k * RATE..(k + 1) * RATE]);
        keccak::f1600(state);
    }
    n_blocks
}

/// The accelerated composition must be digest-identical to the software sponge
/// and to ethrex's reference `keccak_hash`, at every length that crosses a rate
/// boundary. This is the host half of the `keccak-sponge-accel` correctness
/// argument: it pins the *split* (whole blocks to the chip, padded tail to
/// `keccak_permute`), while the chip's own semantics are the executor's and the
/// prover's business.
///
/// Also covers the accelerator DECLINING (returning 0, as it does on a
/// misaligned buffer): the software loop must then absorb everything and reach
/// the same digest.
#[test]
fn accelerated_absorb_matches_software_sponge() {
    let data: Vec<u8> = (0..4 * 136 + 8).map(|i| (i * 97 + 13) as u8).collect();

    for len in [
        0, 1, 8, 135, 136, 137, 271, 272, 273, 407, 408, 409, 500, 544, 552,
    ] {
        let msg = &data[..len];
        let software = keccak256_with_permute(msg, keccak::f1600);
        let accelerated = keccak256_with_backend(msg, keccak::f1600, absorb_whole_blocks_mirror);
        let declined = keccak256_with_backend(msg, keccak::f1600, |_, _| 0);

        assert_eq!(accelerated, keccak_hash(msg), "vs reference, len={len}");
        assert_eq!(accelerated, software, "accel vs software, len={len}");
        assert_eq!(declined, software, "declined vs software, len={len}");
    }
}

/// The seam lets the accelerator absorb only a PREFIX of the whole blocks it is
/// offered, with the software loop picking up the rest. Nothing in the guest
/// takes a partial bite today, but the contract allows it and an off-by-one in
/// the `offset` handoff would otherwise go unnoticed.
#[test]
fn partial_accelerated_take_matches_software_sponge() {
    const RATE: usize = 136;
    let data: Vec<u8> = (0..5 * RATE).map(|i| (i * 31 + 7) as u8).collect();

    for len in [3 * RATE, 3 * RATE + 40, 5 * RATE] {
        let msg = &data[..len];
        for take in 0..=len / RATE {
            let got = keccak256_with_backend(msg, keccak::f1600, |state, blocks| {
                absorb_whole_blocks_mirror(state, &blocks[..take * RATE])
            });
            assert_eq!(got, keccak_hash(msg), "len={len} take={take}");
        }
    }
}

#[test]
fn keccak_sponge_known_answer_vectors() {
    // Vectors from the Ethereum Yellow Paper / EIP-155. These use Keccak-256
    // (0x01 padding), NOT SHA3-256 (0x06 padding). Any sponge framing bug
    // (wrong rate, wrong padding byte, wrong lane endianness) breaks these
    // even if the differential test above passes.

    // keccak256("") = c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
    check_keccak_kat(
        b"",
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
    );
    // keccak256("abc")
    check_keccak_kat(
        b"abc",
        "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45",
    );
    // keccak256("The quick brown fox jumps over the lazy dog")
    check_keccak_kat(
        b"The quick brown fox jumps over the lazy dog",
        "4d741b6f1eb29cb2a9b9911c82f56fa8d73b04959d3d9d222895df6c0b28aa15",
    );
}

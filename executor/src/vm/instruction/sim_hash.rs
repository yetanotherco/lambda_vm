//! Host handlers for the field-native hash/transcript measurement ecalls
//! (EXPERIMENT 1 — see `others/accelerator_noop_sim_spec.md`).
//!
//! Each handler is a TRUSTED, execute-only stub: it reproduces host-side, in one
//! VM cycle, the EXACT byte semantics of a guest software path in the crypto
//! crate, then returns the correct value so the guest still accepts and attests.
//! A build using these ecalls is never proven (they drive no chip — the same
//! LogUp-bus caveat as the Print ecall), so this code only ever runs under
//! `cli execute`.
//!
//! Byte-identity is the whole point: any divergence from the software path flips
//! a Merkle root or Fiat-Shamir challenge and fails in-guest verification. The
//! four primitives reproduced here are:
//!   * the streaming Keccak-256 sponge of `lambda_vm_syscalls::keccak::Keccak256`
//!     (`[u64; 25]` state + byte `offset`), absorbing in place;
//!   * the fixed 64-byte Merkle-parent compression (`keccak256_pair`);
//!   * the canonical field-element serialization used by `AsBytes::stream_bytes`
//!     (each Goldilocks limb reduced to `[0, p)` then big-endian; an extension
//!     element is its limbs in coefficient order);
//!   * the transcript `sample()` (finalize-reset, reverse, re-absorb).
//!
//! The `#[cfg(test)]` module pins every one of these against `sha3::Keccak256`
//! and `math`'s `stream_bytes` — the same references the crypto software path is
//! itself tested against — so host tests guarantee guest byte-identity.

use crate::vm::instruction::execution::{ExecutionError, keccak_f1600};
use crate::vm::memory::Memory;

/// Keccak-256 sponge rate in bytes (1088 bits; capacity 512).
const RATE_BYTES: usize = 136;
/// Rate lanes (17 for r = 1088).
const RATE_LANES: usize = RATE_BYTES / 8;
/// Keccak (not SHA-3) domain-separator byte, per FIPS 202 / Ethereum.
const DELIMITER: u8 = 0x01;
/// Final `pad10*1` bit, pre-shifted into the high byte of the last rate lane.
const FINAL_PAD_LANE_BIT: u64 = 0x80u64 << 56;
/// Goldilocks prime p = 2^64 - 2^32 + 1.
const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;
/// Byte offset of the sponge's `offset` field within the guest struct. The
/// syscalls `Keccak256` is `#[repr(C)]` `{ state: [u64; 25], offset: usize }`,
/// so `state` occupies bytes `0..200` and `offset` bytes `200..208`.
const SPONGE_OFFSET_FIELD_DELTA: u64 = 25 * 8;
/// Total byte size of the guest sponge struct (`state` + `offset`).
const SPONGE_STRUCT_BYTES: u64 = SPONGE_OFFSET_FIELD_DELTA + 8;
/// Largest limb count of any serialized field element (Fp3 = 3 limbs / 24 bytes;
/// Goldilocks base = 1 limb / 8 bytes). Fp2 is never serialized.
const MAX_FELT_LIMBS: u64 = 3;

/// Streaming Keccak-256 sponge, byte-identical to
/// `lambda_vm_syscalls::keccak::Keccak256`. The state doubles as the absorption
/// buffer (input XORs directly into the rate lanes at a running `offset`); the
/// permutation is the executor's own `keccak_f1600`. Absorption is byte-wise —
/// the guest's whole-lane fast path is only an optimization and produces the
/// identical state, so there is nothing to mirror here.
#[derive(Clone)]
struct Sponge {
    state: [u64; 25],
    offset: usize,
}

impl Sponge {
    fn new() -> Self {
        Self {
            state: [0; 25],
            offset: 0,
        }
    }

    #[inline]
    fn absorb_byte(&mut self, b: u8) {
        self.state[self.offset / 8] ^= u64::from(b) << ((self.offset % 8) * 8);
        self.offset += 1;
        if self.offset == RATE_BYTES {
            keccak_f1600(&mut self.state);
            self.offset = 0;
        }
    }

    fn update(&mut self, input: &[u8]) {
        for &b in input {
            self.absorb_byte(b);
        }
    }

    /// Pad (`pad10*1`), permute once, squeeze the 32-byte digest. Consumes the
    /// sponge, matching `Keccak256::finalize`. `offset < RATE_BYTES` always holds
    /// here, so the padded block is exactly one permutation.
    fn finalize(mut self) -> [u8; 32] {
        self.state[self.offset / 8] ^= u64::from(DELIMITER) << ((self.offset % 8) * 8);
        self.state[RATE_LANES - 1] ^= FINAL_PAD_LANE_BIT;
        keccak_f1600(&mut self.state);
        squeeze32(&self.state)
    }
}

/// Squeeze rate lanes 0..4 (little-endian) into the 32-byte digest.
#[inline]
fn squeeze32(state: &[u64; 25]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, chunk) in out.chunks_exact_mut(8).enumerate() {
        chunk.copy_from_slice(&state[i].to_le_bytes());
    }
    out
}

/// Canonical `[0, p)` representative of a raw Goldilocks limb. A limb read from
/// guest memory is any `u64` (values are stored unreduced); a single conditional
/// subtraction canonicalizes it because every `u64 < 2p`.
#[inline]
fn goldilocks_canonical(v: u64) -> u64 {
    if v >= GOLDILOCKS_PRIME {
        v - GOLDILOCKS_PRIME
    } else {
        v
    }
}

/// Keccak-256 of exactly two concatenated 32-byte nodes (the fixed Merkle-parent
/// shape), byte-identical to `lambda_vm_syscalls::keccak::keccak256_pair`. 64
/// bytes fit one rate block: load eight data lanes, XOR the pad bits, one
/// permutation, squeeze four lanes.
fn keccak256_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut state = [0u64; 25];
    for i in 0..4 {
        state[i] = u64::from_le_bytes(left[i * 8..i * 8 + 8].try_into().unwrap());
        state[4 + i] = u64::from_le_bytes(right[i * 8..i * 8 + 8].try_into().unwrap());
    }
    // pad10*1 for a 64-byte message: delimiter at byte 64 (lane 8, low byte),
    // final bit at byte 135 (lane 16, high byte). Both lanes are still zero.
    state[8] ^= u64::from(DELIMITER);
    state[RATE_LANES - 1] ^= FINAL_PAD_LANE_BIT;
    keccak_f1600(&mut state);
    squeeze32(&state)
}

/// Require `addr` to be 8-aligned (doubleword operands must be, matching the
/// keccak-state ecall's contract) and report a clear error otherwise.
#[inline]
fn require_aligned8(addr: u64) -> Result<(), ExecutionError> {
    if addr.is_multiple_of(8) {
        Ok(())
    } else {
        Err(ExecutionError::SimHashUnalignedAddress(addr))
    }
}

/// Load the guest sponge struct at `state_ptr` (`[u64; 25]` state + `usize`
/// offset). Validates 8-alignment, non-overflow, and `offset < RATE_BYTES`.
fn load_sponge(memory: &Memory, state_ptr: u64) -> Result<Sponge, ExecutionError> {
    require_aligned8(state_ptr)?;
    state_ptr
        .checked_add(SPONGE_STRUCT_BYTES - 1)
        .ok_or(ExecutionError::SimHashAddressOverflow)?;
    let mut state = [0u64; 25];
    for (i, lane) in state.iter_mut().enumerate() {
        *lane = memory.load_doubleword(state_ptr + (i as u64) * 8)?;
    }
    let offset_raw = memory.load_doubleword(state_ptr + SPONGE_OFFSET_FIELD_DELTA)?;
    let offset = usize::try_from(offset_raw)
        .ok()
        .filter(|&o| o < RATE_BYTES)
        .ok_or(ExecutionError::SimHashInvalidState(offset_raw))?;
    Ok(Sponge { state, offset })
}

/// Store the sponge struct back to `state_ptr` (assumes `load_sponge` already
/// validated alignment/overflow for the same pointer).
fn store_sponge(
    memory: &mut Memory,
    state_ptr: u64,
    sponge: &Sponge,
) -> Result<(), ExecutionError> {
    for (i, &lane) in sponge.state.iter().enumerate() {
        memory.store_doubleword(state_ptr + (i as u64) * 8, lane)?;
    }
    memory.store_doubleword(state_ptr + SPONGE_OFFSET_FIELD_DELTA, sponge.offset as u64)?;
    Ok(())
}

/// Serialize `count` field elements of `kind` limbs each (read from `elems_ptr`
/// as raw doublewords) and absorb them into `sponge`, exactly as
/// `AsBytes::stream_bytes` would: each limb canonicalized to `[0, p)` then
/// emitted big-endian, in memory (coefficient) order.
fn absorb_felts_into(
    memory: &Memory,
    sponge: &mut Sponge,
    elems_ptr: u64,
    count: u64,
    kind: u64,
) -> Result<(), ExecutionError> {
    if kind == 0 || kind > MAX_FELT_LIMBS {
        return Err(ExecutionError::SimHashInvalidKind(kind));
    }
    require_aligned8(elems_ptr)?;
    let total_limbs = count
        .checked_mul(kind)
        .ok_or(ExecutionError::SimHashAddressOverflow)?;
    for j in 0..total_limbs {
        let limb_addr = elems_ptr
            .checked_add(j * 8)
            .ok_or(ExecutionError::SimHashAddressOverflow)?;
        let limb = memory.load_doubleword(limb_addr)?;
        sponge.update(&goldilocks_canonical(limb).to_be_bytes());
    }
    Ok(())
}

/// Write a 32-byte digest to `out_ptr` byte-wise (the guest output buffer is
/// only byte-aligned).
fn write_digest(
    memory: &mut Memory,
    out_ptr: u64,
    digest: &[u8; 32],
) -> Result<(), ExecutionError> {
    out_ptr
        .checked_add(31)
        .ok_or(ExecutionError::SimHashAddressOverflow)?;
    for (i, &b) in digest.iter().enumerate() {
        memory.store_byte(out_ptr + i as u64, b);
    }
    Ok(())
}

/// `ABSORB_FELTS(state_ptr, elems_ptr, count, kind)`: absorb serialized field
/// elements into the guest-memory sponge in place. Replaces the transcript's
/// per-element `stream_bytes` marshaling.
pub fn absorb_felts(
    memory: &mut Memory,
    state_ptr: u64,
    elems_ptr: u64,
    count: u64,
    kind: u64,
) -> Result<(), ExecutionError> {
    let mut sponge = load_sponge(memory, state_ptr)?;
    absorb_felts_into(memory, &mut sponge, elems_ptr, count, kind)?;
    store_sponge(memory, state_ptr, &sponge)
}

/// `ABSORB_BYTES(state_ptr, bytes_ptr, len)`: absorb raw bytes into the
/// guest-memory sponge in place.
pub fn absorb_bytes(
    memory: &mut Memory,
    state_ptr: u64,
    bytes_ptr: u64,
    len: u64,
) -> Result<(), ExecutionError> {
    let mut sponge = load_sponge(memory, state_ptr)?;
    let bytes = memory.load_bytes(bytes_ptr, len)?;
    sponge.update(&bytes);
    store_sponge(memory, state_ptr, &sponge)
}

/// `TRANSCRIPT_SAMPLE(state_ptr, out32_ptr)`: the whole transcript `sample()` —
/// finalize the current sponge to a digest, reset the sponge, reverse the
/// digest, re-absorb it, and return the reversed digest. Matches
/// `DefaultTranscript::sample` (finalize_reset + reverse + re-absorb).
pub fn transcript_sample(
    memory: &mut Memory,
    state_ptr: u64,
    out_ptr: u64,
) -> Result<(), ExecutionError> {
    let sponge = load_sponge(memory, state_ptr)?;
    let mut digest = sponge.finalize();
    digest.reverse();
    // finalize_reset leaves a default sponge, which then absorbs the reversed
    // digest (32 < 136 bytes, so no permutation, offset ends at 32).
    let mut fresh = Sponge::new();
    fresh.update(&digest);
    store_sponge(memory, state_ptr, &fresh)?;
    write_digest(memory, out_ptr, &digest)
}

/// `HASH_PAIR(l_ptr, r_ptr, out_ptr)`: fixed 64-byte Merkle-parent hash of two
/// 32-byte nodes. Replaces the guest `keccak256_pair` fast path.
pub fn hash_pair(
    memory: &mut Memory,
    l_ptr: u64,
    r_ptr: u64,
    out_ptr: u64,
) -> Result<(), ExecutionError> {
    let left = memory.load_bytes(l_ptr, 32)?;
    let right = memory.load_bytes(r_ptr, 32)?;
    // `load_bytes` returns exactly 32 bytes on success.
    let left: [u8; 32] = left.as_slice().try_into().unwrap();
    let right: [u8; 32] = right.as_slice().try_into().unwrap();
    let digest = keccak256_pair(&left, &right);
    write_digest(memory, out_ptr, &digest)
}

/// `HASH_FELTS(a_ptr, a_count, b_ptr, b_count, kind, out_ptr)`: one-shot leaf-row
/// hash of the concatenation `a ‖ b` of two field-element slices (each `kind`
/// limbs). Replaces the guest `hash_streamed` leaf fast path: a fresh sponge
/// absorbs the serialized elements of `a` then `b`, then finalizes.
///
/// The two-slice form matches the verifier's leaf shape (`evaluations ‖
/// evaluations_sym`, hashed without materializing the concatenation — see
/// `FieldElementVectorBackend::hash_data_from_slices`); a single-slice leaf
/// passes `b_count = 0`.
#[allow(clippy::too_many_arguments)]
pub fn hash_felts(
    memory: &mut Memory,
    a_ptr: u64,
    a_count: u64,
    b_ptr: u64,
    b_count: u64,
    kind: u64,
    out_ptr: u64,
) -> Result<(), ExecutionError> {
    let mut sponge = Sponge::new();
    for &(ptr, count) in &[(a_ptr, a_count), (b_ptr, b_count)] {
        if count > 0 {
            absorb_felts_into(memory, &mut sponge, ptr, count, kind)?;
        }
    }
    let digest = sponge.finalize();
    write_digest(memory, out_ptr, &digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksField;
    use math::traits::AsBytes;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use sha3::{Digest, Keccak256 as RefKeccak256};

    type Fp = FieldElement<GoldilocksField>;
    type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

    fn sha3(bytes: &[u8]) -> [u8; 32] {
        RefKeccak256::digest(bytes).into()
    }

    /// The bytes `stream_bytes` would emit for one field element.
    fn stream_bytes_of<T: AsBytes>(e: &T) -> Vec<u8> {
        let mut out = Vec::new();
        e.stream_bytes(&mut |b| out.extend_from_slice(b));
        out
    }

    // --- primitive-level byte identity (Sponge / pair / serialization) ---

    /// The ported sponge equals the `sha3` reference for every length across
    /// several rate-block boundaries — hence equals crypto's `Keccak256`, which
    /// is itself tested against `sha3`.
    #[test]
    fn sponge_matches_sha3_all_lengths() {
        let data: Vec<u8> = (0..3 * RATE_BYTES + 2)
            .map(|i| (i * 31 + 7) as u8)
            .collect();
        for len in 0..=data.len() {
            let mut sponge = Sponge::new();
            sponge.update(&data[..len]);
            assert_eq!(sponge.finalize(), sha3(&data[..len]), "len={len}");
        }
    }

    /// Splitting the input across arbitrary `update` calls must not change the
    /// digest (absorption chunking is offset-driven).
    #[test]
    fn sponge_chunked_updates_match_sha3() {
        let data: Vec<u8> = (0..1500).map(|i| (i * 131 + 17) as u8).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(0x51ED_10CC);
        for _ in 0..200 {
            let len = rng.random_range(0..data.len());
            let slice = &data[..len];
            let mut sponge = Sponge::new();
            let mut fed = 0;
            while fed < slice.len() {
                let n = 1 + rng.random_range(0..(slice.len() - fed).min(200));
                sponge.update(&slice[fed..fed + n]);
                fed += n;
            }
            assert_eq!(sponge.finalize(), sha3(slice), "len={len}");
        }
    }

    /// The fixed 64-byte parent path equals hashing the concatenation.
    #[test]
    fn keccak256_pair_matches_sha3() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xD1B5_4A32);
        for _ in 0..64 {
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            rng.fill(&mut left);
            rng.fill(&mut right);
            let mut concat = [0u8; 64];
            concat[..32].copy_from_slice(&left);
            concat[32..].copy_from_slice(&right);
            assert_eq!(keccak256_pair(&left, &right), sha3(&concat));
        }
    }

    /// The host limb serialization (canonical big-endian per limb) equals
    /// `AsBytes::stream_bytes` for both the base field and Fp3, including raw
    /// (unreduced) limbs `>= p`.
    #[test]
    fn felt_serialization_matches_stream_bytes() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xF17E_1D5E);
        for _ in 0..256 {
            // Base field: exercise raw limbs both below and above p.
            let raw: u64 = rng.random();
            let fp = Fp::from(raw);
            let host: Vec<u8> = goldilocks_canonical(raw).to_be_bytes().to_vec();
            assert_eq!(host, stream_bytes_of(&fp), "base raw={raw:#x}");

            // Fp3: three limbs, coefficient order.
            let coeffs = [
                rng.random::<u64>(),
                rng.random::<u64>(),
                rng.random::<u64>(),
            ];
            let fp3 = Fp3::new([
                Fp::from(coeffs[0]),
                Fp::from(coeffs[1]),
                Fp::from(coeffs[2]),
            ]);
            let mut host3 = Vec::new();
            for &c in &coeffs {
                host3.extend_from_slice(&goldilocks_canonical(c).to_be_bytes());
            }
            assert_eq!(host3, stream_bytes_of(&fp3), "fp3 coeffs={coeffs:?}");
        }
    }

    // --- handler-level round trips through Memory ---

    /// Write a fresh (zeroed) sponge struct at `ptr`.
    fn put_fresh_sponge(memory: &mut Memory, ptr: u64) {
        for i in 0..25u64 {
            memory.store_doubleword(ptr + i * 8, 0).unwrap();
        }
        memory
            .store_doubleword(ptr + SPONGE_OFFSET_FIELD_DELTA, 0)
            .unwrap();
    }

    /// Read the 32-byte digest at `ptr`.
    fn get_digest(memory: &Memory, ptr: u64) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, b) in out.iter_mut().enumerate() {
            *b = memory.load_byte(ptr + i as u64);
        }
        out
    }

    /// Write raw felt limbs contiguously at `ptr` and return the concatenated
    /// `stream_bytes` reference for the same elements.
    fn put_felts_and_reference(memory: &mut Memory, ptr: u64, limbs: &[u64], kind: u64) -> Vec<u8> {
        for (i, &l) in limbs.iter().enumerate() {
            memory.store_doubleword(ptr + (i as u64) * 8, l).unwrap();
        }
        let mut reference = Vec::new();
        for chunk in limbs.chunks(kind as usize) {
            for &l in chunk {
                reference.extend_from_slice(&goldilocks_canonical(l).to_be_bytes());
            }
        }
        reference
    }

    #[test]
    fn hash_pair_handler_matches_sha3() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x0BAD_F00D);
        for _ in 0..32 {
            let mut memory = Memory::default();
            let (l_ptr, r_ptr, out_ptr) = (0x1000, 0x1040, 0x2000);
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            rng.fill(&mut left);
            rng.fill(&mut right);
            for i in 0..32u64 {
                memory.store_byte(l_ptr + i, left[i as usize]);
                memory.store_byte(r_ptr + i, right[i as usize]);
            }
            hash_pair(&mut memory, l_ptr, r_ptr, out_ptr).unwrap();
            let mut concat = [0u8; 64];
            concat[..32].copy_from_slice(&left);
            concat[32..].copy_from_slice(&right);
            assert_eq!(get_digest(&memory, out_ptr), sha3(&concat));
        }
    }

    #[test]
    fn hash_felts_handler_matches_sha3() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xC0FF_EE00);
        for kind in [1u64, 3] {
            for count in [1u64, 2, 5, 17] {
                let mut memory = Memory::default();
                let (elems_ptr, out_ptr) = (0x8000, 0x9000);
                let limbs: Vec<u64> = (0..count * kind).map(|_| rng.random::<u64>()).collect();
                let reference = put_felts_and_reference(&mut memory, elems_ptr, &limbs, kind);
                // Single-slice leaf: b_count = 0.
                hash_felts(&mut memory, elems_ptr, count, 0, 0, kind, out_ptr).unwrap();
                assert_eq!(
                    get_digest(&memory, out_ptr),
                    sha3(&reference),
                    "kind={kind} count={count}"
                );
            }
        }
    }

    /// The two-slice leaf form (`a ‖ b`, the verifier's `evaluations ‖
    /// evaluations_sym` shape) hashes the concatenation.
    #[test]
    fn hash_felts_two_slice_matches_concat() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x7705_1119);
        for kind in [1u64, 3] {
            for (a_count, b_count) in [(1u64, 1u64), (3, 2), (7, 0), (4, 9)] {
                let mut memory = Memory::default();
                let (a_ptr, b_ptr, out_ptr) = (0x8000, 0xA000, 0xC000);
                let a_limbs: Vec<u64> = (0..a_count * kind).map(|_| rng.random::<u64>()).collect();
                let b_limbs: Vec<u64> = (0..b_count * kind).map(|_| rng.random::<u64>()).collect();
                let mut reference = put_felts_and_reference(&mut memory, a_ptr, &a_limbs, kind);
                reference.extend(put_felts_and_reference(&mut memory, b_ptr, &b_limbs, kind));
                hash_felts(&mut memory, a_ptr, a_count, b_ptr, b_count, kind, out_ptr).unwrap();
                assert_eq!(
                    get_digest(&memory, out_ptr),
                    sha3(&reference),
                    "kind={kind} a={a_count} b={b_count}"
                );
            }
        }
    }

    #[test]
    fn absorb_bytes_handler_matches_sha3() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xABBA_1234);
        for len in [0u64, 1, 7, 8, 135, 136, 137, 300] {
            let mut memory = Memory::default();
            let (state_ptr, bytes_ptr) = (0x400, 0x4000);
            put_fresh_sponge(&mut memory, state_ptr);
            let data: Vec<u8> = (0..len).map(|_| rng.random::<u8>()).collect();
            for (i, &b) in data.iter().enumerate() {
                memory.store_byte(bytes_ptr + i as u64, b);
            }
            absorb_bytes(&mut memory, state_ptr, bytes_ptr, len).unwrap();
            // Finalize the stored sponge and compare to the reference digest.
            let sponge = load_sponge(&memory, state_ptr).unwrap();
            assert_eq!(sponge.finalize(), sha3(&data), "len={len}");
        }
    }

    #[test]
    fn absorb_felts_handler_matches_stream_bytes_then_sha3() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x1357_9BDF);
        for kind in [1u64, 3] {
            for count in [1u64, 4] {
                let mut memory = Memory::default();
                let (state_ptr, elems_ptr) = (0x400, 0x5000);
                put_fresh_sponge(&mut memory, state_ptr);
                let limbs: Vec<u64> = (0..count * kind).map(|_| rng.random::<u64>()).collect();
                let reference = put_felts_and_reference(&mut memory, elems_ptr, &limbs, kind);
                absorb_felts(&mut memory, state_ptr, elems_ptr, count, kind).unwrap();
                let sponge = load_sponge(&memory, state_ptr).unwrap();
                assert_eq!(
                    sponge.finalize(),
                    sha3(&reference),
                    "kind={kind} count={count}"
                );
            }
        }
    }

    /// `TRANSCRIPT_SAMPLE` reproduces `DefaultTranscript::sample`: the returned
    /// digest is `reverse(keccak(state))`, and the sponge is replaced by a fresh
    /// one that has absorbed exactly those reversed 32 bytes.
    #[test]
    fn transcript_sample_handler_matches_reference() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5A3D_1E55);
        for prefix_len in [0u64, 1, 40, 136, 200] {
            let mut memory = Memory::default();
            let (state_ptr, out_ptr) = (0x400, 0x6000);
            put_fresh_sponge(&mut memory, state_ptr);
            // Prime the sponge with some absorbed bytes via ABSORB_BYTES.
            let prefix: Vec<u8> = (0..prefix_len).map(|_| rng.random::<u8>()).collect();
            let bytes_ptr = 0x7000;
            for (i, &b) in prefix.iter().enumerate() {
                memory.store_byte(bytes_ptr + i as u64, b);
            }
            absorb_bytes(&mut memory, state_ptr, bytes_ptr, prefix_len).unwrap();

            // Reference: finalize a clone, reverse, re-absorb into a fresh sponge.
            let primed = load_sponge(&memory, state_ptr).unwrap();
            let mut expected_digest = primed.finalize();
            expected_digest.reverse();
            let mut expected_sponge = Sponge::new();
            expected_sponge.update(&expected_digest);

            transcript_sample(&mut memory, state_ptr, out_ptr).unwrap();

            assert_eq!(
                get_digest(&memory, out_ptr),
                expected_digest,
                "prefix={prefix_len}"
            );
            let after = load_sponge(&memory, state_ptr).unwrap();
            assert_eq!(after.state, expected_sponge.state, "prefix={prefix_len}");
            assert_eq!(after.offset, expected_sponge.offset, "prefix={prefix_len}");
        }
    }

    #[test]
    fn invalid_kind_and_alignment_are_rejected() {
        let mut memory = Memory::default();
        put_fresh_sponge(&mut memory, 0x400);
        assert!(matches!(
            absorb_felts(&mut memory, 0x400, 0x5000, 1, 0),
            Err(ExecutionError::SimHashInvalidKind(0))
        ));
        assert!(matches!(
            absorb_felts(&mut memory, 0x400, 0x5000, 1, 4),
            Err(ExecutionError::SimHashInvalidKind(4))
        ));
        assert!(matches!(
            hash_felts(&mut memory, 0x5004, 1, 0, 0, 1, 0x9000),
            Err(ExecutionError::SimHashUnalignedAddress(0x5004))
        ));
    }
}

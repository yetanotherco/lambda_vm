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

/// One transcript `sample()` step on `sponge`: finalize it to a digest, reverse
/// the digest, and re-absorb it into a fresh sponge. Returns `(reversed_digest,
/// new_sponge)`. Matches `DefaultTranscript::sample` (finalize_reset + reverse +
/// re-absorb) — the digest is the sampled 32 bytes, the new sponge is the reset
/// state. Shared by `transcript_sample`, `sample_felt`, and `sample_u64` so the
/// three stubs mutate the sponge byte-identically.
fn sample_step(sponge: Sponge) -> ([u8; 32], Sponge) {
    let mut digest = sponge.finalize();
    digest.reverse();
    // finalize_reset leaves a default sponge, which then absorbs the reversed
    // digest (32 < 136 bytes, so no permutation, offset ends at 32).
    let mut fresh = Sponge::new();
    fresh.update(&digest);
    (digest, fresh)
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
    let (digest, fresh) = sample_step(sponge);
    store_sponge(memory, state_ptr, &fresh)?;
    write_digest(memory, out_ptr, &digest)
}

/// `SAMPLE_FELT(state_ptr, out_ptr)`: the whole `sample_field_element` for the
/// Fp3 transcript (ROUND-2 increment B). Reproduces #841's duplex sampling
/// host-side — pull 8 big-endian bytes at a time from the sponge's 32-byte
/// squeeze buffer (refilling with a `sample()` step when fewer than 8 remain,
/// buffer starting empty), driving the SAME `sample_field_element_from` the
/// guest runs — then writes the element's three raw limbs at `out_ptr`.
///
/// NOTE (grand composite): #841 replaced the ChaCha20 challenge expansion this
/// stub originally reproduced (`get_random_field_element_from_rng`) with the
/// duplex path above, and the guest call site was dropped in the sim/15 merge
/// (default_transcript.rs) because #841 already makes sampling cheap. This
/// handler is therefore never dispatched on this branch; it is kept faithful to
/// the duplex derivation (correct when the transcript's output buffer is empty
/// at entry, which is the Fiat-Shamir absorb-then-sample case) so that a build
/// re-enabling it can never silently accept a wrong challenge. MEASUREMENT-ONLY.
pub fn sample_felt(
    memory: &mut Memory,
    state_ptr: u64,
    out_ptr: u64,
) -> Result<(), ExecutionError> {
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::traits::HasDefaultTranscript;

    let mut sponge = load_sponge(memory, state_ptr)?;
    let mut out_buf = [0u8; 32];
    let mut out_pos = 32usize; // empty: the first draw forces a squeeze
    let next_u64 = || {
        if out_pos + 8 > out_buf.len() {
            let cur = core::mem::replace(&mut sponge, Sponge::new());
            let (digest, fresh) = sample_step(cur);
            sponge = fresh;
            out_buf = digest;
            out_pos = 0;
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&out_buf[out_pos..out_pos + 8]);
        out_pos += 8;
        u64::from_be_bytes(bytes)
    };
    let element = Degree3GoldilocksExtensionField::sample_field_element_from(next_u64);
    store_sponge(memory, state_ptr, &sponge)?;

    // In-memory layout of `FieldElement<Fp3>` is three consecutive Goldilocks
    // limbs (see `sim_reduced_opening::write_ext`); write the raw representatives.
    let limbs = element.value();
    for (i, limb) in limbs.iter().enumerate() {
        memory.store_doubleword(out_ptr + (i as u64) * 8, *limb.value())?;
    }
    Ok(())
}

/// `SAMPLE_U64(state_ptr, upper_bound, out_ptr)`: the whole `sample_u64` rejection
/// loop (ROUND-2 increment B). Each iteration runs one `sample()` step host-side
/// (mutating the sponge in place) and reads the first eight digest bytes
/// big-endian; the first candidate `>= wrapping_neg(upper_bound) % upper_bound`
/// is reduced mod `upper_bound` and written at `out_ptr`. Byte-identical to
/// `DefaultTranscript::sample_u64` (same threshold, same per-iteration sponge
/// mutation). Folds the loop's per-iteration TRANSCRIPT_SAMPLE ecalls into one
/// VM cycle. MEASUREMENT-ONLY, never proven.
pub fn sample_u64(
    memory: &mut Memory,
    state_ptr: u64,
    upper_bound: u64,
    out_ptr: u64,
) -> Result<(), ExecutionError> {
    if upper_bound == 0 {
        return Err(ExecutionError::SimSampleU64ZeroBound);
    }
    let threshold = upper_bound.wrapping_neg() % upper_bound;
    let mut sponge = load_sponge(memory, state_ptr)?;
    loop {
        let (digest, fresh) = sample_step(sponge);
        sponge = fresh;
        let candidate = u64::from_be_bytes(digest[..8].try_into().unwrap());
        if candidate >= threshold {
            store_sponge(memory, state_ptr, &sponge)?;
            memory.store_doubleword(out_ptr, candidate % upper_bound)?;
            return Ok(());
        }
    }
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

/// `VERIFY_PATH(leaf_hash_ptr, root_ptr, index, path_ptr, path_len, out_ptr)`:
/// verify a Merkle inclusion path in one call. Recomputes the root from the
/// already-hashed leaf at `leaf_hash_ptr` and the `path_len` sibling nodes at
/// `path_ptr` (contiguous 32-byte nodes), folding with the fixed 64-byte
/// Merkle-parent compression [`keccak256_pair`] and the same index-bit child
/// ordering as `verify_merkle_path_from_leaf_hash`, then writes `1` to `out_ptr`
/// if the recomputed root equals the 32-byte root at `root_ptr`, else `0`.
///
/// Byte-faithful and TRUSTED-but-real: it computes the ACTUAL accept/reject
/// answer (a tampered opening yields a mismatched root -> `0` -> the guest
/// rejects), so it subsumes the per-node `HASH_PAIR` ecalls on the verify paths
/// while keeping the tamper test live. Replaces the whole in-guest fold loop
/// (running-buffer glue + root compare + per-node parent-hash ecalls) with a
/// single VM cycle. MEASUREMENT-ONLY: drives no chip, never proven.
pub fn verify_path(
    memory: &mut Memory,
    leaf_hash_ptr: u64,
    root_ptr: u64,
    index: u64,
    path_ptr: u64,
    path_len: u64,
    out_ptr: u64,
) -> Result<(), ExecutionError> {
    // Guard the path buffer's byte span so the per-node offset arithmetic below
    // cannot wrap (`load_bytes` still bounds-checks each individual read).
    let span = path_len
        .checked_mul(32)
        .ok_or(ExecutionError::SimHashAddressOverflow)?;
    path_ptr
        .checked_add(span)
        .ok_or(ExecutionError::SimHashAddressOverflow)?;

    let leaf = memory.load_bytes(leaf_hash_ptr, 32)?;
    let root = memory.load_bytes(root_ptr, 32)?;
    // `load_bytes` returns exactly 32 bytes on success.
    let mut cur: [u8; 32] = leaf.as_slice().try_into().unwrap();
    let root: [u8; 32] = root.as_slice().try_into().unwrap();

    let mut idx = index;
    for i in 0..path_len {
        let sibling = memory.load_bytes(path_ptr + i * 32, 32)?;
        let sibling: [u8; 32] = sibling.as_slice().try_into().unwrap();
        // index-bit child ordering, matching `verify_merkle_path_from_leaf_hash`:
        // even index -> running hash is the left child, sibling the right.
        cur = if idx & 1 == 0 {
            keccak256_pair(&cur, &sibling)
        } else {
            keccak256_pair(&sibling, &cur)
        };
        idx >>= 1;
    }

    // Byte equality of the recomputed root against the committed root (matches
    // `IsMerkleTreeBackend::nodes_eq` for the 32-byte node).
    let accept = cur == root;
    memory.store_byte(out_ptr, u8::from(accept));
    Ok(())
}

/// `SIM_VERIFY_PATH_BATCH(&VerifyPathBatchInput)`: verify EVERY committed FRI
/// layer's Merkle opening for one query in a single call. For each layer it hashes
/// the ordered `(evaluation, evaluation_sym)` leaf pair (ordered by the layer
/// index's low bit) with the same field-leaf hash as `HASH_FELTS`, folds the auth
/// path to the committed root with the same index-bit ordering as `VERIFY_PATH`,
/// and ANDs the per-layer accept into a single byte at `out_ptr`. Subsumes the
/// per-layer `HASH_FELTS` + `VERIFY_PATH` ecalls into one call per query.
///
/// Byte-faithful and TRUSTED-but-real, exactly like [`verify_path`]: it computes
/// the ACTUAL accept (a tampered opening yields a mismatched root -> `0` -> the
/// guest rejects). MEASUREMENT-ONLY: drives no chip, never proven. Returns
/// `num_layers` for the CLI's per-call layer tally.
pub fn verify_path_batch(memory: &mut Memory, input_ptr: u64) -> Result<u64, ExecutionError> {
    use core::mem::offset_of;
    use math::sim_midlevel::VerifyPathBatchInput;

    let f = |off: usize| memory.load_doubleword(input_ptr.wrapping_add(off as u64));
    let num_layers = f(offset_of!(VerifyPathBatchInput, num_layers))?;
    let start_index = f(offset_of!(VerifyPathBatchInput, start_index))?;
    let roots_ptr = f(offset_of!(VerifyPathBatchInput, roots_ptr))?;
    let evals_ptr = f(offset_of!(VerifyPathBatchInput, evals_ptr))?;
    let evals_sym_ptr = f(offset_of!(VerifyPathBatchInput, evals_sym_ptr))?;
    let path_descs_ptr = f(offset_of!(VerifyPathBatchInput, path_descs_ptr))?;
    let out_ptr = f(offset_of!(VerifyPathBatchInput, out_ptr))?;

    // Extension element stride (`[FpE; 3]` = 3 `u64`); FRI layer leaves are Fp3.
    const EXT_STRIDE: u64 = 24;
    const FELT_KIND: u64 = 3;

    let mut all_ok = true;
    let mut index = start_index;
    for i in 0..num_layers {
        let eval_ptr = evals_ptr.wrapping_add(i.wrapping_mul(EXT_STRIDE));
        let sym_ptr = evals_sym_ptr.wrapping_add(i.wrapping_mul(EXT_STRIDE));
        // Leaf ordering by the index's low bit: odd -> (sym, eval), else
        // (eval, sym) — matching `verify_fri_layer_openings`.
        let (first_ptr, second_ptr) = if index & 1 == 1 {
            (sym_ptr, eval_ptr)
        } else {
            (eval_ptr, sym_ptr)
        };
        let mut sponge = Sponge::new();
        absorb_felts_into(memory, &mut sponge, first_ptr, 1, FELT_KIND)?;
        absorb_felts_into(memory, &mut sponge, second_ptr, 1, FELT_KIND)?;
        let leaf = sponge.finalize();

        // Committed root for this layer.
        let root = memory.load_bytes(roots_ptr.wrapping_add(i.wrapping_mul(32)), 32)?;
        let root: [u8; 32] = root.as_slice().try_into().unwrap();

        // Auth-path descriptor: (path_ptr, path_len) at path_descs_ptr + i*16.
        let desc = path_descs_ptr.wrapping_add(i.wrapping_mul(16));
        let path_ptr = memory.load_doubleword(desc)?;
        let path_len = memory.load_doubleword(desc.wrapping_add(8))?;
        // Guard the path buffer's byte span so the per-node offset can't wrap.
        let span = path_len
            .checked_mul(32)
            .ok_or(ExecutionError::SimHashAddressOverflow)?;
        path_ptr
            .checked_add(span)
            .ok_or(ExecutionError::SimHashAddressOverflow)?;

        // Fold the leaf up the path from `index >> 1` (matches
        // `verify_merkle_path_from_leaf_hash`).
        let mut cur = leaf;
        let mut idx = index >> 1;
        for k in 0..path_len {
            let sibling = memory.load_bytes(path_ptr + k * 32, 32)?;
            let sibling: [u8; 32] = sibling.as_slice().try_into().unwrap();
            cur = if idx & 1 == 0 {
                keccak256_pair(&cur, &sibling)
            } else {
                keccak256_pair(&sibling, &cur)
            };
            idx >>= 1;
        }
        all_ok &= cur == root;
        index >>= 1;
    }

    memory.store_byte(out_ptr, u8::from(all_ok));
    Ok(num_layers)
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

    /// Reference fold matching `verify_merkle_path_from_leaf_hash`: even index
    /// bit -> running hash is the left child, sibling the right; odd -> swapped.
    fn fold_reference(leaf: [u8; 32], path: &[[u8; 32]], mut index: u64) -> [u8; 32] {
        let mut cur = leaf;
        for sib in path {
            cur = if index & 1 == 0 {
                keccak256_pair(&cur, sib)
            } else {
                keccak256_pair(sib, &cur)
            };
            index >>= 1;
        }
        cur
    }

    /// `VERIFY_PATH` returns the REAL accept/reject answer: `1` for the honest
    /// root, `0` for any tampered leaf / sibling / root / index. This is the
    /// property that keeps the tamper test live while the ecall subsumes the
    /// per-node `HASH_PAIR` fold.
    #[test]
    fn verify_path_handler_computes_real_answer() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5EED_9A71);
        for path_len in [0u64, 1, 5, 20] {
            for _ in 0..8 {
                let mut memory = Memory::default();
                let (leaf_ptr, root_ptr, path_ptr, out_ptr) = (0x1000, 0x1040, 0x2000, 0x3000);
                let mut leaf = [0u8; 32];
                rng.fill(&mut leaf);
                let path: Vec<[u8; 32]> = (0..path_len)
                    .map(|_| {
                        let mut n = [0u8; 32];
                        rng.fill(&mut n);
                        n
                    })
                    .collect();
                let index: u64 = rng.random::<u64>() & ((1u64 << 40) - 1);
                let root = fold_reference(leaf, &path, index);

                let write = |m: &mut Memory, base: u64, bytes: &[u8; 32]| {
                    for (i, &b) in bytes.iter().enumerate() {
                        m.store_byte(base + i as u64, b);
                    }
                };
                write(&mut memory, leaf_ptr, &leaf);
                write(&mut memory, root_ptr, &root);
                for (i, node) in path.iter().enumerate() {
                    write(&mut memory, path_ptr + (i as u64) * 32, node);
                }

                // Honest path -> accept.
                verify_path(
                    &mut memory,
                    leaf_ptr,
                    root_ptr,
                    index,
                    path_ptr,
                    path_len,
                    out_ptr,
                )
                .unwrap();
                assert_eq!(memory.load_byte(out_ptr), 1, "honest path_len={path_len}");

                // Tampered root -> reject.
                let mut bad_root = root;
                bad_root[0] ^= 1;
                write(&mut memory, root_ptr, &bad_root);
                verify_path(
                    &mut memory,
                    leaf_ptr,
                    root_ptr,
                    index,
                    path_ptr,
                    path_len,
                    out_ptr,
                )
                .unwrap();
                assert_eq!(memory.load_byte(out_ptr), 0, "bad root path_len={path_len}");
                write(&mut memory, root_ptr, &root);

                // Tampered leaf -> reject.
                let mut bad_leaf = leaf;
                bad_leaf[7] ^= 0x80;
                write(&mut memory, leaf_ptr, &bad_leaf);
                verify_path(
                    &mut memory,
                    leaf_ptr,
                    root_ptr,
                    index,
                    path_ptr,
                    path_len,
                    out_ptr,
                )
                .unwrap();
                assert_eq!(memory.load_byte(out_ptr), 0, "bad leaf path_len={path_len}");
                write(&mut memory, leaf_ptr, &leaf);

                // Tampered sibling (when present) -> reject.
                if path_len > 0 {
                    let mut bad_sib = path[0];
                    bad_sib[3] ^= 0x40;
                    write(&mut memory, path_ptr, &bad_sib);
                    verify_path(
                        &mut memory,
                        leaf_ptr,
                        root_ptr,
                        index,
                        path_ptr,
                        path_len,
                        out_ptr,
                    )
                    .unwrap();
                    assert_eq!(memory.load_byte(out_ptr), 0, "bad sib path_len={path_len}");
                }
            }
        }
    }

    /// The field-leaf hash `absorb_felts_into` / `HASH_FELTS` produces for the
    /// ordered pair — one Fp3 element each (3 canonical big-endian limbs).
    fn ref_leaf(first: [u64; 3], second: [u64; 3]) -> [u8; 32] {
        let mut sponge = Sponge::new();
        for &limb in first.iter().chain(second.iter()) {
            sponge.update(&goldilocks_canonical(limb).to_be_bytes());
        }
        sponge.finalize()
    }

    /// `SIM_VERIFY_PATH_BATCH` verifies every layer's ordered-leaf hash + path
    /// fold in one call and reports the REAL AND of the per-layer accepts: `1`
    /// only when every committed layer matches, `0` if ANY layer's root / leaf /
    /// sibling is tampered. This is the property that keeps the tamper test live
    /// while the ecall subsumes the per-layer HASH_FELTS + VERIFY_PATH pairs.
    #[test]
    fn verify_path_batch_matches_per_layer_and_catches_tamper() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5EED_B47C);
        for num_layers in [1u64, 2, 5, 11] {
            for _ in 0..6 {
                let mut memory = Memory::default();
                let start_index: u64 = rng.random::<u64>() & ((1u64 << 40) - 1);
                // Layout: evals, evals_sym, roots, per-layer paths, path descs,
                // input struct, out byte — spaced well apart.
                let evals_ptr = 0x1_0000u64;
                let sym_ptr = 0x2_0000u64;
                let roots_ptr = 0x3_0000u64;
                let paths_base = 0x4_0000u64;
                let descs_ptr = 0x8_0000u64;
                let input_ptr = 0x9_0000u64;
                let out_ptr = 0x9_1000u64;

                let write32 = |m: &mut Memory, base: u64, b: &[u8; 32]| {
                    for (i, &x) in b.iter().enumerate() {
                        m.store_byte(base + i as u64, x);
                    }
                };

                let mut roots: Vec<[u8; 32]> = Vec::new();
                let mut index = start_index;
                for layer in 0..num_layers {
                    // Random Fp3 eval / eval_sym (raw limbs, little-endian u64s).
                    let eval: [u64; 3] = [rng.random(), rng.random(), rng.random()];
                    let sym: [u64; 3] = [rng.random(), rng.random(), rng.random()];
                    for k in 0..3 {
                        memory
                            .store_doubleword(evals_ptr + layer * 24 + k * 8, eval[k as usize])
                            .unwrap();
                        memory
                            .store_doubleword(sym_ptr + layer * 24 + k * 8, sym[k as usize])
                            .unwrap();
                    }
                    // Ordered leaf, matching verify_fri_layer_openings.
                    let (first, second) = if index & 1 == 1 {
                        (sym, eval)
                    } else {
                        (eval, sym)
                    };
                    let leaf = ref_leaf(first, second);
                    // Random path, folded from index>>1 to the honest root.
                    let path_len = (rng.random::<u64>() % 6) + 1;
                    let path: Vec<[u8; 32]> = (0..path_len)
                        .map(|_| {
                            let mut n = [0u8; 32];
                            rng.fill(&mut n);
                            n
                        })
                        .collect();
                    let path_ptr = paths_base + layer * 0x400;
                    for (i, node) in path.iter().enumerate() {
                        write32(&mut memory, path_ptr + (i as u64) * 32, node);
                    }
                    let root = fold_reference(leaf, &path, index >> 1);
                    write32(&mut memory, roots_ptr + layer * 32, &root);
                    roots.push(root);
                    // Path descriptor (path_ptr, path_len).
                    memory
                        .store_doubleword(descs_ptr + layer * 16, path_ptr)
                        .unwrap();
                    memory
                        .store_doubleword(descs_ptr + layer * 16 + 8, path_len)
                        .unwrap();
                    index >>= 1;
                }

                // Input struct: 7 contiguous u64 fields (repr(C) order).
                let fields = [
                    num_layers,
                    start_index,
                    roots_ptr,
                    evals_ptr,
                    sym_ptr,
                    descs_ptr,
                    out_ptr,
                ];
                for (i, &v) in fields.iter().enumerate() {
                    memory
                        .store_doubleword(input_ptr + (i as u64) * 8, v)
                        .unwrap();
                }

                // Honest -> accept, and the reported layer count is num_layers.
                let n = verify_path_batch(&mut memory, input_ptr).unwrap();
                assert_eq!(n, num_layers);
                assert_eq!(
                    memory.load_byte(out_ptr),
                    1,
                    "honest num_layers={num_layers}"
                );

                // Tamper the last layer's root -> reject (proves the AND is live).
                let mut bad = roots[(num_layers - 1) as usize];
                bad[0] ^= 1;
                write32(&mut memory, roots_ptr + (num_layers - 1) * 32, &bad);
                verify_path_batch(&mut memory, input_ptr).unwrap();
                assert_eq!(
                    memory.load_byte(out_ptr),
                    0,
                    "bad root num_layers={num_layers}"
                );
            }
        }
    }

    /// `sample_felt` advances the sponge by exactly one `sample()` step (matching
    /// the independently-tested `transcript_sample`, since Fp3's three coordinates
    /// consume one 32-byte squeeze) and writes three canonical Goldilocks limbs.
    /// It drives the guest's own `sample_field_element_from` (#841 duplex path)
    /// directly, so the value is identical by construction; the end-to-end
    /// honest-accept run is the derivation check.
    #[test]
    fn sample_felt_advances_sponge_and_writes_canonical_fp3() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5A3D_FE17);
        for prefix_len in [0u64, 1, 40, 200] {
            let (state_ptr, out_ptr) = (0x400, 0x8000);
            // Prime two identical sponges: one for sample_felt, one to derive the
            // expected post-sample() sponge via transcript_sample.
            let prefix: Vec<u8> = (0..prefix_len).map(|_| rng.random::<u8>()).collect();

            let mut m_felt = Memory::default();
            let mut m_ref = Memory::default();
            put_fresh_sponge(&mut m_felt, state_ptr);
            put_fresh_sponge(&mut m_ref, state_ptr);
            let bytes_ptr = 0x9000;
            for (i, &b) in prefix.iter().enumerate() {
                m_felt.store_byte(bytes_ptr + i as u64, b);
                m_ref.store_byte(bytes_ptr + i as u64, b);
            }
            absorb_bytes(&mut m_felt, state_ptr, bytes_ptr, prefix_len).unwrap();
            absorb_bytes(&mut m_ref, state_ptr, bytes_ptr, prefix_len).unwrap();

            // Expected sponge state after ONE sample() step.
            transcript_sample(&mut m_ref, state_ptr, 0xA000).unwrap();
            let expected = load_sponge(&m_ref, state_ptr).unwrap();

            sample_felt(&mut m_felt, state_ptr, out_ptr).unwrap();
            let after = load_sponge(&m_felt, state_ptr).unwrap();
            assert_eq!(after.state, expected.state, "prefix={prefix_len}");
            assert_eq!(after.offset, expected.offset, "prefix={prefix_len}");

            for i in 0..3u64 {
                let limb = m_felt.load_doubleword(out_ptr + i * 8).unwrap();
                assert!(
                    limb < GOLDILOCKS_PRIME,
                    "limb {i} not canonical: {limb:#x} (prefix={prefix_len})"
                );
            }
        }
    }

    /// `sample_u64` returns a value `< upper_bound` and advances the sponge; a
    /// power-of-two bound (threshold 0) takes exactly one `sample()` step, so the
    /// sponge matches `transcript_sample`. A zero bound is rejected.
    #[test]
    fn sample_u64_in_range_and_advances_sponge() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x0DDB_411D);
        for upper_bound in [1u64 << 4, 1 << 10, 1 << 20] {
            let (state_ptr, out_ptr) = (0x400, 0x8000);
            let prefix: Vec<u8> = (0..37u64).map(|_| rng.random::<u8>()).collect();

            let mut m_u64 = Memory::default();
            let mut m_ref = Memory::default();
            put_fresh_sponge(&mut m_u64, state_ptr);
            put_fresh_sponge(&mut m_ref, state_ptr);
            let bytes_ptr = 0x9000;
            for (i, &b) in prefix.iter().enumerate() {
                m_u64.store_byte(bytes_ptr + i as u64, b);
                m_ref.store_byte(bytes_ptr + i as u64, b);
            }
            absorb_bytes(&mut m_u64, state_ptr, bytes_ptr, 37).unwrap();
            absorb_bytes(&mut m_ref, state_ptr, bytes_ptr, 37).unwrap();

            // Power-of-two bound => threshold 0 => exactly one sample() step.
            transcript_sample(&mut m_ref, state_ptr, 0xA000).unwrap();
            let expected = load_sponge(&m_ref, state_ptr).unwrap();
            let expected_digest = get_digest(&m_ref, 0xA000);
            let expected_val =
                u64::from_be_bytes(expected_digest[..8].try_into().unwrap()) % upper_bound;

            sample_u64(&mut m_u64, state_ptr, upper_bound, out_ptr).unwrap();
            let got = m_u64.load_doubleword(out_ptr).unwrap();
            assert!(got < upper_bound, "out of range: {got} >= {upper_bound}");
            assert_eq!(got, expected_val, "value ub={upper_bound}");
            let after = load_sponge(&m_u64, state_ptr).unwrap();
            assert_eq!(after.state, expected.state, "ub={upper_bound}");
            assert_eq!(after.offset, expected.offset, "ub={upper_bound}");
        }
        // Zero bound is rejected.
        let mut memory = Memory::default();
        put_fresh_sponge(&mut memory, 0x400);
        assert!(matches!(
            sample_u64(&mut memory, 0x400, 0, 0x8000),
            Err(ExecutionError::SimSampleU64ZeroBound)
        ));
    }
}

//! Host side of the machine's keccak256: the byte-stream packing convention
//! and the padding the emitter bakes in as program constants.
//!
//! The machine has no bytes — its cells are felts — so a byte stream reaches
//! `edsl::keccak256` pre-packed as `u32` halves, four bytes each, little-endian.
//! That is the same convention the state itself uses (`keccak_adapter`), which
//! is what lets a rate block be assembled with plain `Pack` instructions and no
//! per-byte arithmetic.

use crate::tables::types::FE;

use super::layout::keccak::RATE_BYTES;

/// Bytes carried by one `u32`-half felt.
pub const BYTES_PER_HALF: usize = 4;

/// Packs a byte stream into `u32`-half felts, four bytes each, little-endian.
///
/// The final half is zero-padded when `bytes.len()` is not a multiple of four.
/// The emitter relies on exactly that: where a half straddles the end of the
/// message it adds the padding constant to the stream half, and addition equals
/// bitwise-or only because the stream half's high bytes are known zero.
/// [`assert_high_bytes_zero`] is the executable statement of that obligation.
pub fn pack_stream(bytes: &[u8]) -> Vec<FE> {
    bytes
        .chunks(BYTES_PER_HALF)
        .map(|chunk| {
            let mut half = [0u8; BYTES_PER_HALF];
            half[..chunk.len()].copy_from_slice(chunk);
            FE::from(u64::from(u32::from_le_bytes(half)))
        })
        .collect()
}

/// Number of halves [`pack_stream`] produces for `len_bytes`.
pub const fn num_stream_halves(len_bytes: usize) -> usize {
    len_bytes.div_ceil(BYTES_PER_HALF)
}

/// The keccak256 padded length: `pad10*1` always adds at least one byte, so the
/// message grows to the next multiple of the rate even when it already is one.
pub const fn padded_len(len_bytes: usize) -> usize {
    (len_bytes / RATE_BYTES + 1) * RATE_BYTES
}

/// Number of rate blocks the emitter absorbs for `len_bytes`.
pub const fn num_blocks(len_bytes: usize) -> usize {
    padded_len(len_bytes) / RATE_BYTES
}

/// The `pad10*1` byte at padded position `pos` for a message of `len_bytes`:
/// `0x01` at the first padding position, `0x80` at the last of the final block,
/// and `0x81` when they coincide.
pub fn pad_byte(len_bytes: usize, pos: usize) -> u8 {
    debug_assert!(pos >= len_bytes && pos < padded_len(len_bytes));
    let mut v = 0u8;
    if pos == len_bytes {
        v |= 0x01;
    }
    if pos == padded_len(len_bytes) - 1 {
        v |= 0x80;
    }
    v
}

/// The padding contribution to half `h` of the padded message — the value the
/// emitter adds to (or uses in place of) the stream half.
///
/// Returns `0` for halves that lie entirely inside the message.
pub fn pad_half(len_bytes: usize, h: usize) -> u64 {
    let mut acc = 0u64;
    for j in 0..BYTES_PER_HALF {
        let pos = h * BYTES_PER_HALF + j;
        if pos >= len_bytes && pos < padded_len(len_bytes) {
            acc |= u64::from(pad_byte(len_bytes, pos)) << (8 * j);
        }
    }
    acc
}

/// Checks the packing obligation for the half that straddles the end of the
/// message: its bytes at or beyond `len_bytes` must be zero, or the emitter's
/// `stream_half + pad_half` would carry instead of merging.
pub fn assert_high_bytes_zero(stream: &[FE], len_bytes: usize) {
    use math::field::traits::IsPrimeField;
    let tail = len_bytes % BYTES_PER_HALF;
    if tail == 0 {
        return;
    }
    let h = len_bytes / BYTES_PER_HALF;
    let v = crate::tables::types::GoldilocksField::canonical(stream[h].value());
    assert_eq!(
        v >> (8 * tail),
        0,
        "stream half {h} must be zero above byte {tail}: pack_stream guarantees it"
    );
}

/// `keccak256` over `bytes`, as the production hasher computes it. The machine
/// program's public output is compared against this.
pub fn keccak256(bytes: &[u8]) -> [u8; 32] {
    use crypto::hash::platform_keccak::PlatformKeccak256 as Keccak256;
    use digest::Digest;
    let mut h = Keccak256::new();
    h.update(bytes);
    h.finalize().into()
}

// ===================== Host model of DefaultTranscript =====================

/// Bytes in one squeeze; the duplex output buffer hands them out 8 at a time.
pub const SQUEEZE_LEN: usize = 32;

/// Host mirror of the production `DefaultTranscript` (post-#841), tracking the
/// same state the machine emitter tracks at emit time.
///
/// This exists so the emitter's static consumption schedule — which candidate
/// comes from which squeeze, and where absorbs invalidate the buffer — can be
/// derived and tested without a machine proof. It is checked against the real
/// `DefaultTranscript` in `machine_tests`.
#[derive(Clone)]
pub struct TranscriptModel {
    /// Bytes absorbed since the last finalize (the hasher's pending input).
    segment: Vec<u8>,
    buf: [u8; SQUEEZE_LEN],
    /// Bytes already handed out of `buf`; `SQUEEZE_LEN` means empty.
    pos: usize,
}

impl TranscriptModel {
    pub fn new(data: &[u8]) -> Self {
        Self {
            segment: data.to_vec(),
            buf: [0u8; SQUEEZE_LEN],
            pos: SQUEEZE_LEN,
        }
    }

    /// Absorbing invalidates the buffer: a later challenge must depend on this
    /// input, so bytes squeezed before it are dropped.
    pub fn append(&mut self, bytes: &[u8]) {
        self.pos = SQUEEZE_LEN;
        self.segment.extend_from_slice(bytes);
    }

    /// Finalize, reverse, re-absorb the reversed bytes, return them. Also
    /// invalidates the buffer.
    pub fn sample(&mut self) -> [u8; SQUEEZE_LEN] {
        let mut digest = keccak256(&self.segment);
        digest.reverse();
        self.segment = digest.to_vec();
        self.pos = SQUEEZE_LEN;
        digest
    }

    /// Next big-endian 64-bit candidate, refilling with one squeeze when fewer
    /// than 8 bytes remain.
    pub fn next_u64(&mut self) -> u64 {
        if self.pos + 8 > SQUEEZE_LEN {
            self.buf = self.sample();
            self.pos = 0;
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        u64::from_be_bytes(bytes)
    }

    /// Whether the next `next_u64` would refill (the emitter needs this to know
    /// where to place a keccak row).
    pub fn would_refill(&self) -> bool {
        self.pos + 8 > SQUEEZE_LEN
    }

    pub fn pos(&self) -> usize {
        self.pos
    }
}

/// THE IDENTITY THE MACHINE EMITTER RELIES ON.
///
/// The four big-endian candidates carved out of a reversed digest are exactly
/// the ORIGINAL digest's first four `u64` lanes, in reverse lane order — so the
/// machine never has to reverse anything to read candidates.
///
/// Candidate `i` is `Σ_{k<8} reversed[8i+k]·2^(8(7−k))`, and `reversed[j]` is
/// `digest[31−j]`, so substituting `m = 7−k` gives
/// `Σ_{m<8} digest[24−8i+m]·2^(8m)` — the LITTLE-endian `u64` at digest byte
/// offset `24−8i`, i.e. keccak state lane `3−i`. The big-endian read and the
/// byte reversal cancel exactly.
///
/// Consequence: candidates come straight off the plain digest words (state
/// lanes, already `u32` halves on the bus), and the reversed digest is needed
/// only for the RE-ABSORB. Verified by `be_candidates_are_plain_state_lanes`.
pub fn candidate_from_state(state: &[u64; 25], index: usize) -> u64 {
    debug_assert!(index < 4);
    state[3 - index]
}

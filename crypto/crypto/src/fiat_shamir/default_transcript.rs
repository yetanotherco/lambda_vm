use crate::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};
use crate::fiat_shamir::transcript_hash::{
    Blake3TranscriptHash, KeccakTranscriptHash, TranscriptHash,
};

use core::marker::PhantomData;
use digest::Digest;
use math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField, IsSubFieldOf},
    },
    traits::AsBytes,
};

/// Bytes produced by one Keccak squeeze; the duplex output buffer holds this
/// many bytes and hands them out `8` at a time (`SQUEEZE_LEN / 8` u64 candidates
/// per squeeze).
const SQUEEZE_LEN: usize = 32;

/// Sponge Fiat-Shamir transcript with a Plonky3-style duplex output buffer,
/// over the hash `T` names.
///
/// Challenges are derived by squeezing the sponge and rejection-sampling field
/// coordinates directly from those bytes — there is **no CSPRNG**. Earlier this
/// type seeded a `ChaCha20Rng` from every squeeze and pulled the field element
/// from the keystream; on the recursion guest that ChaCha block was pure
/// software (Keccak is a precompile, ChaCha is not), so it dominated the
/// challenge-sampling cost while producing bytes the sponge already gives for
/// free. The output buffer amortizes one squeeze across up to `SQUEEZE_LEN / 8`
/// 64-bit candidates, so a cubic-extension element (3 coordinates) usually costs
/// a single squeeze.
///
/// `T` defaults to [`KeccakTranscriptHash`], so `DefaultTranscript::<F>::new(..)`
/// still names exactly the transcript this system has always produced — every
/// method body below is hash-agnostic, and the keccak configuration selects the
/// unbounded rejection schedule, so its bytes do not move.
pub struct DefaultTranscript<F: HasDefaultTranscript, T: TranscriptHash = KeccakTranscriptHash> {
    hasher: T::Digest,
    /// Duplex output buffer: bytes squeezed from the sponge, consumed 8 at a
    /// time by field/`u64` sampling. Positions `[out_pos, SQUEEZE_LEN)` are the
    /// bytes not yet handed out; `out_pos == SQUEEZE_LEN` means "empty, squeeze
    /// to refill". Absorbing new data invalidates it (see `append_bytes`) so a
    /// squeeze can never reflect input appended after it was produced.
    out_buf: [u8; SQUEEZE_LEN],
    out_pos: usize,
    phantom: PhantomData<(F, T)>,
}

impl<F: HasDefaultTranscript, T: TranscriptHash> Clone for DefaultTranscript<F, T> {
    fn clone(&self) -> Self {
        Self {
            hasher: self.hasher.clone(),
            out_buf: self.out_buf,
            out_pos: self.out_pos,
            phantom: PhantomData,
        }
    }
}

impl<F, T> DefaultTranscript<F, T>
where
    F: HasDefaultTranscript,
    T: TranscriptHash,
    FieldElement<F>: AsBytes,
{
    pub fn new(data: &[u8]) -> Self {
        let mut res = Self {
            hasher: T::Digest::new(),
            out_buf: [0u8; SQUEEZE_LEN],
            // Empty: the first sample forces a squeeze.
            out_pos: SQUEEZE_LEN,
            phantom: PhantomData,
        };
        res.append_bytes(data);
        res
    }

    /// Raw squeeze: finalize the current sponge state, advance the hash chain by
    /// absorbing the (reversed) output, and return it. Also invalidates the
    /// duplex output buffer, so interleaving raw `sample()` calls with buffered
    /// field/`u64` sampling can never reuse stale squeeze bytes.
    pub fn sample(&mut self) -> [u8; 32] {
        let mut result_hash: [u8; 32] = self.hasher.finalize_reset().into();
        result_hash.reverse();
        self.hasher.update(result_hash);
        self.out_pos = SQUEEZE_LEN;
        result_hash
    }

    /// Next 64-bit candidate from the duplex output buffer, refilling with one
    /// squeeze when fewer than 8 bytes remain. Big-endian, matching the byte
    /// order `sample_u64` used when it read directly from `sample()`.
    fn next_sample_u64(&mut self) -> u64 {
        if self.out_pos + 8 > SQUEEZE_LEN {
            self.out_buf = self.sample();
            self.out_pos = 0;
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.out_buf[self.out_pos..self.out_pos + 8]);
        self.out_pos += 8;
        u64::from_be_bytes(bytes)
    }

    /// One base coordinate's worth of candidates under a FIXED schedule: draw
    /// exactly `n`, hand back the first that `F` would accept.
    ///
    /// All `n` are drawn whichever one lands in range — that is the entire
    /// point. Returning early on the first hit would restore the data-dependent
    /// schedule this exists to remove.
    ///
    /// The value handed back is one `F::sample_field_element_from` accepts, so
    /// its own rejection loop exits after a single call and consumption is
    /// exactly `n` per coordinate. When every candidate misses (≈ 2⁻³²ⁿ) the
    /// last one is returned, `F` rejects it, and the loop draws another `n` —
    /// the schedule stays a multiple of `n` and the distribution stays exactly
    /// uniform, because nothing is ever reduced into range.
    fn next_candidate_fixed(&mut self, n: usize) -> u64 {
        candidate_under_fixed_schedule::<F>(n, || self.next_sample_u64())
    }
}

/// One base coordinate's worth of candidates under a FIXED schedule: pull
/// exactly `n` from `next`, hand back the first that `F` would accept.
///
/// Free-standing rather than a method so the schedule can be driven by a
/// counting closure in a test — "consumes exactly `n`" is the whole property,
/// and it is not observable from the transcript's outputs.
pub(crate) fn candidate_under_fixed_schedule<F: HasDefaultTranscript>(
    n: usize,
    mut next: impl FnMut() -> u64,
) -> u64 {
    let mut chosen: Option<u64> = None;
    let mut last = 0u64;
    for _ in 0..n {
        let candidate = next();
        last = candidate;
        if chosen.is_none() && F::candidate_in_range(candidate) {
            chosen = Some(candidate);
        }
    }
    chosen.unwrap_or(last)
}

/// The BLAKE3 Fiat-Shamir transcript: `Blake3Chain` in the sponge, and rider
/// 1's constant-consumption sampling.
pub type Blake3Transcript<F> = DefaultTranscript<F, Blake3TranscriptHash>;

impl<F, T> Default for DefaultTranscript<F, T>
where
    F: HasDefaultTranscript,
    T: TranscriptHash,
    FieldElement<F>: AsBytes,
{
    fn default() -> Self {
        Self::new(&[])
    }
}

impl<F, T> IsTranscript<F> for DefaultTranscript<F, T>
where
    F: HasDefaultTranscript,
    T: TranscriptHash,
    FieldElement<F>: AsBytes,
{
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        // Absorbing new input invalidates any buffered squeeze output: a
        // subsequent challenge must depend on this input, so drop the bytes
        // squeezed before it.
        self.out_pos = SQUEEZE_LEN;
        self.hasher.update(new_bytes);
    }

    fn append_field_element(&mut self, element: &FieldElement<F>) {
        // Absorb, same invalidation as `append_bytes` (the field element's bytes
        // are streamed straight into the sponge with no intermediate `Vec`).
        self.out_pos = SQUEEZE_LEN;
        element.stream_bytes(&mut |b| self.hasher.update(b));
    }

    fn state(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

    fn sample_field_element(&mut self) -> FieldElement<F> {
        match T::CANDIDATES_PER_COORDINATE {
            None => F::sample_field_element_from(|| self.next_sample_u64()),
            Some(n) => F::sample_field_element_from(|| self.next_candidate_fixed(n.get())),
        }
    }

    /// Note this loop is already fixed-consumption where it matters. Its only
    /// production caller samples query indices against `domain_size >> 1`, a
    /// power of two, and for `upper_bound = 2^k` the threshold is
    /// `(-2^k) mod 2^k = 0` — so no candidate is ever rejected. The loop is here
    /// for non-power-of-two bounds, which the protocol does not use.
    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        assert!(upper_bound > 0, "upper_bound must be greater than 0");
        let threshold = upper_bound.wrapping_neg() % upper_bound;
        loop {
            let candidate = self.next_sample_u64();
            if candidate >= threshold {
                return candidate % upper_bound;
            }
        }
    }
}

impl<F, T, S> IsStarkTranscript<F, S> for DefaultTranscript<F, T>
where
    F: HasDefaultTranscript,
    T: TranscriptHash,
    FieldElement<F>: AsBytes,
    S: IsField + IsSubFieldOf<F>,
{
    // nothing to implement: sample_z_ood uses the default body
}

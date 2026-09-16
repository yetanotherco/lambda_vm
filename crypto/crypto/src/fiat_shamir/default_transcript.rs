use crate::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};
use crate::fiat_shamir::transcript_hash::{KeccakTranscriptHash, TranscriptHash};

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
/// still names exactly the transcript this system has always produced: every
/// method body below is hash-agnostic, and the parameter only decides which
/// `digest::Digest` the sponge is. Nothing about the keccak configuration's
/// bytes moves.
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
    /// absorbing the output, and return it. Also invalidates the duplex output
    /// buffer, so interleaving raw `sample()` calls with buffered field/`u64`
    /// sampling can never reuse stale squeeze bytes.
    ///
    /// ★ The byte order is the configuration's, via
    /// [`TranscriptHash::REVERSES_SQUEEZE`] — `true` for keccak, which is the
    /// convention every proof on this branch has been produced under, and
    /// `false` for an algebraic sponge, whose squeeze is already four canonical
    /// felts and whose consumer is a field-native verifier that would otherwise
    /// spend rows undoing the reversal.
    ///
    /// ⚠ The returned bytes and the chained bytes are the SAME value, and that
    /// is deliberate: a replaying verifier reproducing this chain would
    /// otherwise have two byte conventions to carry instead of none. Whichever
    /// order the constant selects applies to both.
    ///
    /// The constant is associated, so each configuration monomorphises to
    /// straight-line code — the keccak arm keeps the instruction sequence it
    /// had before this became a choice.
    pub fn sample(&mut self) -> [u8; 32] {
        // ★ Hash-agnostic, and deliberately here rather than inside a digest:
        // a counter that lives in keccak reads ZERO for an algebraic
        // transcript, which is indistinguishable from "no transcript ran".
        crate::hash_metrics::count_transcript_squeeze::<T::Digest>();
        let mut result_hash: [u8; 32] = self.hasher.finalize_reset().into();
        if T::REVERSES_SQUEEZE {
            result_hash.reverse();
        }
        self.hasher.update(result_hash);
        self.out_pos = SQUEEZE_LEN;
        result_hash
    }

    /// Next 64-bit candidate from the duplex output buffer, refilling with one
    /// squeeze when fewer than 8 bytes remain. Big-endian, matching the byte
    /// order `sample_u64` used when it read directly from `sample()`.
    ///
    /// ★ `SQUEEZE_LEN` is 32 and every read is 8, so `out_pos` only ever takes
    /// the values `0, 8, 16, 24, 32` and a candidate is always a whole 8-byte
    /// group — never two halves of adjacent ones. That is what lets a
    /// configuration whose squeeze is four canonical felts promise
    /// `CANDIDATES_PER_COORDINATE = Some(1)`: the felt boundaries and the read
    /// boundaries are the same boundaries. `append_bytes` invalidates the
    /// buffer wholesale rather than partially, so the alignment survives
    /// interleaved absorbs.
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
}

impl<F, T> crate::fiat_shamir::transcript_hash::HasTranscriptHash for DefaultTranscript<F, T>
where
    F: HasDefaultTranscript,
    T: TranscriptHash,
{
    type Hash = T;
}

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
        crate::hash_metrics::count_transcript_absorb::<T::Digest>();
        self.hasher.update(new_bytes);
    }

    fn append_field_element(&mut self, element: &FieldElement<F>) {
        // Absorb, same invalidation as `append_bytes` (the field element's bytes
        // are streamed straight into the sponge with no intermediate `Vec`).
        //
        // ⚠ Counted PER `update` rather than once per call, because that is the
        // unit `absorb_calls` has always used and the dimension a block-
        // absorption change moves. Today the degree-3 extension writes one
        // 24-byte buffer and calls the sink once, so the two happen to agree —
        // a field or a serialisation that streams in pieces would not, and the
        // counter should follow the sponge rather than the argument list.
        self.out_pos = SQUEEZE_LEN;
        let hasher = &mut self.hasher;
        element.stream_bytes(&mut |b| {
            crate::hash_metrics::count_transcript_absorb::<T::Digest>();
            hasher.update(b);
        });
    }

    fn state(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

    fn sample_field_element(&mut self) -> FieldElement<F> {
        F::sample_field_element_from(|| self.next_sample_u64())
    }

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

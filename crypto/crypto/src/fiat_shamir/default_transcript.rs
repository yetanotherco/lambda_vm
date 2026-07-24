use crate::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};

use crate::hash::platform_keccak::PlatformKeccak256 as Keccak256;
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

/// Keccak-sponge Fiat-Shamir transcript with a Plonky3-style duplex output
/// buffer.
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
pub struct DefaultTranscript<F: HasDefaultTranscript> {
    hasher: Keccak256,
    /// Duplex output buffer: bytes squeezed from the sponge, consumed 8 at a
    /// time by field/`u64` sampling. Positions `[out_pos, SQUEEZE_LEN)` are the
    /// bytes not yet handed out; `out_pos == SQUEEZE_LEN` means "empty, squeeze
    /// to refill". Absorbing new data invalidates it (see `append_bytes`) so a
    /// squeeze can never reflect input appended after it was produced.
    out_buf: [u8; SQUEEZE_LEN],
    out_pos: usize,
    phantom: PhantomData<F>,
}

impl<F: HasDefaultTranscript> Clone for DefaultTranscript<F> {
    fn clone(&self) -> Self {
        Self {
            hasher: self.hasher.clone(),
            out_buf: self.out_buf,
            out_pos: self.out_pos,
            phantom: PhantomData,
        }
    }
}

impl<F> DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: AsBytes,
{
    pub fn new(data: &[u8]) -> Self {
        let mut res = Self {
            hasher: Keccak256::new(),
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
        // A raw squeeze invalidates the duplex output buffer so buffered
        // field/`u64` sampling can never reuse stale bytes (matches #841).
        self.out_pos = SQUEEZE_LEN;
        // EXPERIMENT 1: one TRANSCRIPT_SAMPLE ecall does the whole finalize-reset
        // + reverse + re-absorb host-side (byte-identical result). #841 changed
        // only how the returned bytes are consumed downstream (see
        // `next_sample_u64`), not how the squeeze itself is produced, so the
        // ecall's semantics are unchanged.
        #[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
        {
            self.hasher.sim_transcript_sample()
        }
        #[cfg(not(all(target_arch = "riscv64", feature = "sim-hash-ecalls")))]
        {
            let mut result_hash: [u8; 32] = self.hasher.finalize_reset().into();
            result_hash.reverse();
            self.hasher.update(result_hash);
            result_hash
        }
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
}

impl<F> Default for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: AsBytes,
{
    fn default() -> Self {
        Self::new(&[])
    }
}

impl<F> IsTranscript<F> for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: AsBytes,
{
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        // Absorbing new input invalidates any buffered squeeze output: a
        // subsequent challenge must depend on this input, so drop the bytes
        // squeezed before it.
        self.out_pos = SQUEEZE_LEN;
        // EXPERIMENT 1: absorb raw bytes via the ABSORB_BYTES ecall.
        #[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
        {
            self.hasher.sim_absorb_bytes(new_bytes);
        }
        #[cfg(not(all(target_arch = "riscv64", feature = "sim-hash-ecalls")))]
        {
            self.hasher.update(new_bytes);
        }
    }

    fn append_field_element(&mut self, element: &FieldElement<F>) {
        // Same invalidation as `append_bytes`.
        self.out_pos = SQUEEZE_LEN;
        // EXPERIMENT 1: absorb the raw limbs via the ABSORB_FELTS ecall; the host
        // applies the canonical `stream_bytes` serialization. `FieldElement<F>`
        // is `#[repr(transparent)]` over its Goldilocks limbs ([u64; N]), so
        // `size_of / 8` is the limb count and the element pointer is the limb
        // pointer. Byte-identical to the `stream_bytes` path below for the
        // transcript field (Goldilocks base = 1 limb, Fp3 = 3 limbs).
        #[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
        {
            let kind = core::mem::size_of::<FieldElement<F>>() / 8;
            self.hasher
                .sim_absorb_felts(core::ptr::from_ref(element).cast::<u8>(), 1, kind);
        }
        #[cfg(not(all(target_arch = "riscv64", feature = "sim-hash-ecalls")))]
        {
            element.stream_bytes(&mut |b| self.hasher.update(b));
        }
    }

    fn state(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

    fn sample_field_element(&mut self) -> FieldElement<F> {
        // NOTE (grand composite): #841 replaced the ChaCha20 challenge expansion
        // with a duplex output buffer, deleting `rand`/`rand_chacha` and the
        // `get_random_field_element_from_rng` trait method. The ROUND-2
        // `sim-sample-ecall` challenger stub reproduced the *ChaCha* derivation
        // from a sponge-only pointer, so it is both non-compiling here (dead
        // symbols) and semantically wrong on #841 (the duplex state lives in the
        // transcript, not the sponge). #841 already collapses the sampling cost
        // the stub was meant to swallow, so the two levers fully overlap; we keep
        // #841's cheap duplex path and drop the stub. See report to team lead.
        F::sample_field_element_from(|| self.next_sample_u64())
    }

    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        assert!(upper_bound > 0, "upper_bound must be greater than 0");
        // See the note in `sample_field_element`: #841's duplex path supersedes
        // the ChaCha-era `sim-sample-ecall` stub, which is dropped here.
        let threshold = upper_bound.wrapping_neg() % upper_bound;
        loop {
            let candidate = self.next_sample_u64();
            if candidate >= threshold {
                return candidate % upper_bound;
            }
        }
    }
}

impl<F, S> IsStarkTranscript<F, S> for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: AsBytes,
    S: IsField + IsSubFieldOf<F>,
{
    // nothing to implement: sample_z_ood uses the default body
}

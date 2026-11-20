//! Keccak-256 implementation selected per target: the `keccak_permute`
//! precompile on the riscv64 guest, plain software `sha3::Keccak256` on host.
//! Wraps `lambda_vm_syscalls::keccak::Keccak256` with the `digest` crate
//! traits so it's a drop-in replacement anywhere a `D: Digest` is expected
//! (Merkle tree backends, Fiat-Shamir transcript).

#[cfg(target_arch = "riscv64")]
mod imp {
    use digest::{
        FixedOutput, FixedOutputReset, HashMarker, Output, OutputSizeUser, Reset, Update,
    };
    use lambda_vm_syscalls::keccak::Keccak256 as SyscallKeccak256;

    // INVARIANT (load-bearing): this adapter must remain a PURE PASSTHROUGH of
    // `SyscallKeccak256`. The TypeId specializations in
    // crypto/crypto/src/merkle_tree/backends/field_element_vector.rs bypass it
    // and drive the syscall sponge directly, on the assumption that both paths
    // hash identically. Adding ANY behavior here (a domain prefix, extra
    // absorption, a different reset policy) silently desyncs the specialized
    // branches from the generic path — and the failure surfaces as in-guest
    // proof rejection, not as a host test failure.

    #[derive(Clone, Default)]
    pub struct PlatformKeccak256(SyscallKeccak256);

    impl HashMarker for PlatformKeccak256 {}

    impl OutputSizeUser for PlatformKeccak256 {
        type OutputSize = digest::typenum::U32;
    }

    impl Update for PlatformKeccak256 {
        fn update(&mut self, data: &[u8]) {
            self.0.update(data);
        }
    }

    impl FixedOutput for PlatformKeccak256 {
        fn finalize_into(self, out: &mut Output<Self>) {
            let mut digest = [0u8; 32];
            self.0.finalize(&mut digest);
            out.copy_from_slice(&digest);
        }
    }

    impl Reset for PlatformKeccak256 {
        fn reset(&mut self) {
            *self = Self::default();
        }
    }

    impl FixedOutputReset for PlatformKeccak256 {
        fn finalize_into_reset(&mut self, out: &mut Output<Self>) {
            let mut digest = [0u8; 32];
            core::mem::take(&mut self.0).finalize(&mut digest);
            out.copy_from_slice(&digest);
        }
    }
}

// Host, `hash-metrics` feature ON: `sha3::Keccak256` plus a finalize counter for
// [`crate::hash_metrics`]. The counter is a PURE SIDE EFFECT — every method
// forwards to the inner hasher (byte-identical digest) and is `#[inline(always)]`,
// so no cross-crate call is added over the bare alias.
#[cfg(all(not(target_arch = "riscv64"), feature = "hash-metrics"))]
mod imp {
    use digest::{
        FixedOutput, FixedOutputReset, HashMarker, Output, OutputSizeUser, Reset, Update,
    };

    /// The `usize` accumulates bytes absorbed for the CURRENT hash, so
    /// `finalize` can report the permutation count (`bytes / 136 + 1`). It resets
    /// to zero on `reset` / `finalize_into_reset`, and `finalize_into` consumes
    /// `self`.
    #[derive(Clone, Default)]
    pub struct PlatformKeccak256(sha3::Keccak256, usize);

    impl HashMarker for PlatformKeccak256 {}

    impl OutputSizeUser for PlatformKeccak256 {
        type OutputSize = digest::typenum::U32;
    }

    impl Update for PlatformKeccak256 {
        #[inline(always)]
        fn update(&mut self, data: &[u8]) {
            // Absorption — the guest's dominant keccak cost (many small
            // `stream_bytes` absorbs), which no finalize counter would see. The
            // running byte total is what lets `finalize` report a permutation
            // count as well.
            self.1 += data.len();
            crate::hash_metrics::count_absorb(data.len());
            Update::update(&mut self.0, data);
        }
    }

    impl FixedOutput for PlatformKeccak256 {
        #[inline(always)]
        fn finalize_into(self, out: &mut Output<Self>) {
            crate::hash_metrics::count_finalize(self.1);
            FixedOutput::finalize_into(self.0, out);
        }
    }

    impl Reset for PlatformKeccak256 {
        #[inline(always)]
        fn reset(&mut self) {
            self.1 = 0;
            Reset::reset(&mut self.0);
        }
    }

    impl FixedOutputReset for PlatformKeccak256 {
        #[inline(always)]
        fn finalize_into_reset(&mut self, out: &mut Output<Self>) {
            crate::hash_metrics::count_finalize(self.1);
            self.1 = 0;
            FixedOutputReset::finalize_into_reset(&mut self.0, out);
        }
    }
}

// Default host build (no `hash-metrics` feature): the plain alias, provably
// unchanged from upstream.
#[cfg(all(not(target_arch = "riscv64"), not(feature = "hash-metrics")))]
mod imp {
    pub type PlatformKeccak256 = sha3::Keccak256;
}

pub use imp::PlatformKeccak256;

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

    // Field-native hash/transcript measurement ecalls (EXPERIMENT 1): thin
    // pass-throughs to the inner sponge's ecall helpers, so the transcript swap
    // sites can drive the host ecalls without reaching the private field. These
    // route to the same byte semantics as the software path (the INVARIANT above
    // still holds — the sponge produced is identical), just computed host-side.
    #[cfg(feature = "sim-hash-ecalls")]
    impl PlatformKeccak256 {
        pub fn sim_absorb_bytes(&mut self, bytes: &[u8]) {
            self.0.sim_absorb_bytes(bytes);
        }

        pub fn sim_absorb_felts(&mut self, elems_ptr: *const u8, count: usize, kind: usize) {
            self.0.sim_absorb_felts(elems_ptr, count, kind);
        }

        pub fn sim_transcript_sample(&mut self) -> [u8; 32] {
            self.0.sim_transcript_sample()
        }
    }

    // Transcript challenge-sampling ecalls (ROUND-2 increment B): thin
    // pass-throughs to the inner sponge, so `DefaultTranscript`'s
    // `sample_field_element` / `sample_u64` can drive the host ecalls without
    // reaching the private field. Byte-identical to the software path (same
    // sponge mutation + same ChaCha20/rejection sampling), just host-side.
    #[cfg(feature = "sim-sample-ecall")]
    impl PlatformKeccak256 {
        pub fn sim_sample_felt(&mut self, out_ptr: *mut u8) {
            self.0.sim_sample_felt(out_ptr);
        }

        pub fn sim_sample_u64(&mut self, upper_bound: u64) -> u64 {
            self.0.sim_sample_u64(upper_bound)
        }
    }
}

#[cfg(not(target_arch = "riscv64"))]
mod imp {
    pub type PlatformKeccak256 = sha3::Keccak256;
}

pub use imp::PlatformKeccak256;

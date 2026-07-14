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

#[cfg(not(target_arch = "riscv64"))]
mod imp {
    pub type PlatformKeccak256 = sha3::Keccak256;
}

pub use imp::PlatformKeccak256;

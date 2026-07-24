use crate::{Buffer, Permutation};

// The software permutation and its round constants are only used off the riscv64
// guest (where `keccakf` dispatches to the accelerator syscall); gate them so the
// guest build has no dead code.
#[cfg(not(target_arch = "riscv64"))]
const ROUNDS: usize = 24;

#[cfg(not(target_arch = "riscv64"))]
const RC: [u64; ROUNDS] = [
    1u64,
    0x8082u64,
    0x800000000000808au64,
    0x8000000080008000u64,
    0x808bu64,
    0x80000001u64,
    0x8000000080008081u64,
    0x8000000000008009u64,
    0x8au64,
    0x88u64,
    0x80008009u64,
    0x8000000au64,
    0x8000808bu64,
    0x800000000000008bu64,
    0x8000000000008089u64,
    0x8000000000008003u64,
    0x8000000000008002u64,
    0x8000000000000080u64,
    0x800au64,
    0x800000008000000au64,
    0x8000000080008081u64,
    0x8000000000008080u64,
    0x80000001u64,
    0x8000000080008008u64,
];

// Original software permutation, kept as `keccakf_software`. On host targets the
// public `keccakf` is exactly this; on the riscv64 guest it is replaced by the
// accelerator syscall below (so the software permutation is compiled out entirely).
#[cfg(not(target_arch = "riscv64"))]
keccak_function!("`keccak-f[1600, 24]`", keccakf_software, ROUNDS, RC);

/// keccak-f[1600, 24].
///
/// LAMBDA_VM FORK: on the riscv64 guest this dispatches to the `keccak_permute`
/// accelerator syscall (one ecall + 25 word reads/writes proven by the KECCAK
/// table, instead of ~600 ARX instructions on the CPU table); on every other
/// target it runs the unmodified software permutation. The state layout is the
/// canonical `[u64; 25]` lane order, identical on both sides — verified end to
/// end by the ethrex trie-root differential tests.
#[cfg(target_arch = "riscv64")]
pub fn keccakf(a: &mut [u64; crate::WORDS]) {
    lambda_vm_syscalls::syscalls::keccak_permute(a);
}

#[cfg(not(target_arch = "riscv64"))]
pub fn keccakf(a: &mut [u64; crate::WORDS]) {
    keccakf_software(a);
}

pub struct KeccakF;

impl Permutation for KeccakF {
    fn execute(buffer: &mut Buffer) {
        keccakf(buffer.words());
    }
}

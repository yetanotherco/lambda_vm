use crate::{Buffer, Permutation};

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

// LAMBDA VM PATCH (vendored tiny-keccak 2.0.2):
// On the riscv64 guest target the Keccak-f[1600] permutation is routed to the
// Lambda VM keccak chip via the `keccak_permute` ecall instead of running in
// software. Host builds keep the original software implementation.
/// Apply the Keccak-f[1600] permutation to `state` (Lambda VM: via the keccak chip).
#[cfg(target_arch = "riscv64")]
pub fn keccakf(state: &mut [u64; 25]) {
    lambda_vm_syscalls::syscalls::keccak_permute(state);
}

#[cfg(not(target_arch = "riscv64"))]
keccak_function!("`keccak-f[1600, 24]`", keccakf, ROUNDS, RC);

pub struct KeccakF;

impl Permutation for KeccakF {
    fn execute(buffer: &mut Buffer) {
        keccakf(buffer.words());
    }
}

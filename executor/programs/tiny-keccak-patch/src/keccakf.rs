use crate::{Buffer, Permutation};

#[cfg(not(target_os = "zkvm"))]
const ROUNDS: usize = 24;

#[cfg(not(target_os = "zkvm"))]
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

// On the lambda_vm zkVM target, delegate to the keccak precompile ecall.
// On other targets (host, tests), use the original software implementation.
/// `keccak-f[1600, 24]`
#[cfg(target_os = "zkvm")]
pub fn keccakf(a: &mut [u64; crate::WORDS]) {
    unsafe {
        core::arch::asm!(
            "ecall",
            in("a0") a.as_mut_ptr(),
            in("a7") (usize::MAX - 1), // KECCAK_SYSCALL_NUMBER
        );
    }
}

#[cfg(not(target_os = "zkvm"))]
keccak_function!("`keccak-f[1600, 24]`", keccakf, ROUNDS, RC);

pub struct KeccakF;

impl Permutation for KeccakF {
    fn execute(buffer: &mut Buffer) {
        keccakf(buffer.words());
    }
}

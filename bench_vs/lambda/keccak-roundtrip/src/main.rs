#![no_std]
#![no_main]

use core::arch::asm;
use core::panic::PanicInfo;

use sha3::{Digest, Keccak256};

const PRIVATE_INPUT_START: usize = 0xFF000000;
const SYSCALL_COMMIT: u64 = 64;
const SYSCALL_HALT: u64 = 93;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

/// Read the entire private-input region as a byte slice.
///
/// Layout (per `syscalls::get_private_input`): 4-byte LE length prefix at
/// `PRIVATE_INPUT_START`, payload at +4.
fn read_private_input() -> &'static [u8] {
    let len = unsafe { core::ptr::read_volatile(PRIVATE_INPUT_START as *const u32) } as usize;
    let data = (PRIVATE_INPUT_START + 4) as *const u8;
    unsafe { core::slice::from_raw_parts(data, len) }
}

fn commit(bytes: &[u8]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") 1u64,
            in("a1") bytes.as_ptr(),
            in("a2") bytes.len(),
            in("a7") SYSCALL_COMMIT,
        );
    }
}

fn halt() -> ! {
    unsafe {
        asm!(
            "ecall",
            in("a0") 0u64,
            in("a7") SYSCALL_HALT,
            options(noreturn),
        );
    }
}

/// Guest entry point.
///
/// Reads a message from the private input, computes its Keccak256 digest
/// (which on riscv64 routes `keccak::p1600` through the lambda-vm
/// `KeccakPermute` precompile syscall via the `keccak-patched` crate), and
/// commits the 32-byte digest as the public output.
///
/// If the precompile is mis-wired or computes the wrong permutation, the
/// committed digest will not match the FIPS-202 reference vector for the
/// supplied message, and the host-side test will fail.
#[unsafe(no_mangle)]
pub fn main() -> ! {
    let msg = read_private_input();

    let mut hasher = Keccak256::new();
    hasher.update(msg);
    let digest = hasher.finalize();

    commit(digest.as_slice());
    halt()
}

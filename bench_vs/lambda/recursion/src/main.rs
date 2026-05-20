#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use core::arch::asm;
use core::panic::PanicInfo;

use embedded_alloc::TlsfHeap as Heap;
use lambda_vm_prover::{ProofOptions, VmProof, VmVerifyingKey};
// Required to pull in the riscv crate's critical-section implementation.
use riscv as _;

const PRIVATE_INPUT_START: usize = 0xFF000000;
const SYSCALL_COMMIT: u64 = 64;
const SYSCALL_HALT: u64 = 93;
const MAX_MEMORY_SIZE: usize = 0xC000_0000;

#[global_allocator]
static HEAP: Heap = Heap::empty();

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

fn init_allocator() {
    unsafe extern "C" {
        static _end: u8;
    }
    let heap_pos = (&raw const _end) as usize;
    unsafe { HEAP.init(heap_pos, MAX_MEMORY_SIZE - heap_pos) }
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

/// Private input layout (postcard-encoded):
///   (VmProof, Vec<u8>, ProofOptions, VmVerifyingKey)
/// where the `Vec<u8>` holds the inner program's ELF bytes, the
/// `ProofOptions` specifies the parameters the inner prover used, and the
/// `VmVerifyingKey` carries the host-derived bitwise preprocessed commitment
/// so the guest can skip the ~87% of verifier cycles that would otherwise be
/// spent recomputing it from scratch.
#[unsafe(no_mangle)]
pub fn main() -> ! {
    init_allocator();

    let blob = read_private_input();
    let (vm_proof, inner_elf, options, vkey): (VmProof, Vec<u8>, ProofOptions, VmVerifyingKey) =
        postcard::from_bytes(blob).expect("failed to deserialize recursion input");

    let ok =
        lambda_vm_prover::verify_with_options_with_vkey(&vm_proof, &inner_elf, &options, Some(&vkey))
            .expect("verify errored");
    assert!(ok, "inner proof failed verification");

    commit(&[1u8]);
    halt()
}

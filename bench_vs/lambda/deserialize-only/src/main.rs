//! Deserialize-only counterpart to the recursion guest.
//!
//! Reads the same private-input blob as `recursion-bench`, postcard-decodes
//! `(VmProof, Vec<u8>, ProofOptions)`, then commits success and halts —
//! without ever calling `verify_with_options`. The cycle delta between this
//! guest and `recursion-bench` is the actual cost of the STARK verifier
//! inside the VM (everything else being equal).

#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use core::arch::asm;
use core::panic::PanicInfo;

use embedded_alloc::TlsfHeap as Heap;
use lambda_vm_prover::{ProofOptions, VmProof};
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

#[unsafe(no_mangle)]
pub fn main() -> ! {
    init_allocator();

    let blob = read_private_input();
    let decoded: (VmProof, Vec<u8>, ProofOptions) =
        postcard::from_bytes(blob).expect("failed to deserialize");

    // Force the commit byte to depend on the actually-decoded value. Without
    // this, LLVM at -O3 was eliding the postcard decode entirely — the only
    // sinks for `decoded` were `black_box(&decoded)` (which only forces the
    // *reference* to materialize, not the pointee) and `Drop`, neither of
    // which require the decoded bytes to be real. With the commit byte tied
    // to a deep field of the decoded value, the decode has to run.
    let proof_options_byte = decoded.2.blowup_factor;
    let inner_elf_byte = *decoded.1.first().unwrap_or(&0);
    let marker = proof_options_byte ^ inner_elf_byte;

    commit(&[marker]);
    halt()
}

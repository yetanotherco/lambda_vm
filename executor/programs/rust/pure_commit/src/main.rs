// Minimal Rust guest program: no_std, no_main, no allocator, no syscalls crate.
// Uses only raw inline `asm!("ecall")` for Commit (64) and Halt (93).
// Serves as a control case in the prover test suite (`test_pure_commit_rust`):
// verifies that Rust can compile to a provable ELF when the heap allocator is
// bypassed, independent of the Rust-std startup path.
#![no_std]
#![no_main]

use core::arch::asm;
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[unsafe(export_name = "main")]
pub extern "C" fn main() -> ! {
    // Commit 4 bytes [0xAA, 0xBB, 0xCC, 0xDD]
    let buf: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];
    unsafe {
        asm!(
            "ecall",
            in("a0") 1usize,          // fd = stdout
            in("a1") buf.as_ptr(),
            in("a2") 4usize,
            in("a7") 64usize,         // Commit
        );
    }
    // Halt
    unsafe {
        asm!(
            "ecall",
            in("a0") 0usize,          // exit_code = 0
            in("a7") 93usize,         // Halt
        );
    }
    loop {}
}

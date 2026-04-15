// Minimal Rust program that proves on main:
// - no_std, no_main (no Rust std runtime)
// - no allocator (no TLSF init)
// - no syscalls crate (no Print/Panic ecalls)
// - Only uses Commit (64) and Halt (93) ecalls, both of which have
//   receivers on the Ecall bus.
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

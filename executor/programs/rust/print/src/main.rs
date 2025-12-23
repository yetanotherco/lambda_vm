#![no_main]
#![no_std]

use core::arch::asm;

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

/// This is a template for printing in the vm
/// This test does not verify the output
pub fn print_string(s: &str) {
    unsafe {
        asm!(
            "mv a0, {ptr}",
            "mv a1, {len}",
            "ecall",
            ptr = in(reg) s.as_ptr(),
            len = in(reg) s.len(),
        );
    }
}

#[unsafe(export_name = "main")]
pub fn main() -> i32 {
    print_string("Hello, World!");
    1
}

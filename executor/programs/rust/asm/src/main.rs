#![no_main]
#![no_std]

use core::arch::asm;

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[unsafe(export_name = "main")]
pub fn main() {
    unsafe {
        asm!(
        "addi a0, zero, 42",
    )}
}

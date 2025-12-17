#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

fn use_byte(value: u32) -> [u8;4] {
    value.to_be_bytes()
}

#[unsafe(export_name = "main")]
pub fn main() -> u8 {
    let value: u32 = 0xDEADBEEF;
    let bytes = use_byte(value);
    bytes[0]
}

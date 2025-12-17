#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

fn use_byte(value: u32) -> [u16;2] {
    let bytes = value.to_be_bytes();
    [
        (bytes[0] as u16) << 8 | (bytes[1] as u16),
        (bytes[2] as u16) << 8 | (bytes[3] as u16),
    ]
}

#[unsafe(export_name = "main")]
pub fn main() -> u16 {
    let value: u32 = 0xDEADBEEF;
    let bytes = use_byte(value);
    bytes[0]
}

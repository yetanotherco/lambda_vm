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
pub fn main() -> i8 {
    let value: u32 = 0x01020304;
    let bytes = use_byte(value);
    // Return the sum of the bytes as i8
    (bytes[0] as i8) - (bytes[1] as i8) - (bytes[2] as i8) - (bytes[3] as i8)
}

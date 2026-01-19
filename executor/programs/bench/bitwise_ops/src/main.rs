#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

const ITERATIONS: usize = 55000;

#[inline(never)]
fn bitwise_mix(a: u32, b: u32) -> u32 {
    let x = a ^ b;
    let y = (a & b) | (!a & !b);
    let z = (a >> 16) | (b << 16);
    x.wrapping_add(y).wrapping_add(z)
}

#[unsafe(no_mangle)]
pub fn main() -> u32 {
    let mut result = 0x12345678u32;
    let mut i = 0;

    while i < ITERATIONS {
        result = bitwise_mix(result, i as u32);
        result = result.rotate_left(5);
        result ^= i as u32;
        i += 1;
    }

    result
}

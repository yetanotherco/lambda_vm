#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

/// Rotate right
fn rotr(x: u32, n: u32) -> u32 {
    (x >> n) | (x << (32 - n))
}

/// SHA-256 style operations (ch, maj, sigma0, sigma1)
fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (!x & z)
}

fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

fn sigma0(x: u32) -> u32 {
    rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
}

fn sigma1(x: u32) -> u32 {
    rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
}

fn gamma0(x: u32) -> u32 {
    rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
}

fn gamma1(x: u32) -> u32 {
    rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
}

#[unsafe(no_mangle)]
pub fn main() -> u32 {
    // Test values
    let a: u32 = 0x6a09e667;
    let b: u32 = 0xbb67ae85;
    let c: u32 = 0x3c6ef372;

    // Apply various bitwise operations
    let mut result = 0u32;

    // Test ch
    result = result.wrapping_add(ch(a, b, c));

    // Test maj
    result = result.wrapping_add(maj(a, b, c));

    // Test sigma functions
    result = result.wrapping_add(sigma0(a));
    result = result.wrapping_add(sigma1(b));

    // Test gamma functions
    result = result.wrapping_add(gamma0(c));
    result = result.wrapping_add(gamma1(a));

    // Mix with more operations
    let mut i = 0u32;
    while i < 100 {
        result = result ^ rotr(result.wrapping_add(i), (i & 31) as u32);
        result = result.wrapping_add(ch(result, i, result ^ i));
        i += 1;
    }

    result
}

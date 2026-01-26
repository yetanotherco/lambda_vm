#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

const SIZE: usize = 420000;

#[inline(never)]
fn sum_array(arr: &[u32; SIZE]) -> u32 {
    let mut sum = 0u32;
    let mut i = 0;
    while i < SIZE {
        sum = sum.wrapping_add(arr[i]);
        i += 1;
    }
    sum
}

#[inline(never)]
fn init_array(arr: &mut [u32; SIZE]) {
    let mut i = 0;
    while i < SIZE {
        arr[i] = (i + 1) as u32;
        i += 1;
    }
}

#[unsafe(no_mangle)]
pub fn main() -> u32 {
    let mut arr = [0u32; SIZE];

    // Initialize array with 1, 2, 3, ..., SIZE
    init_array(&mut arr);

    // Sum of 1 to 10000 = 10000 * 10001 / 2 = 50005000
    sum_array(&arr)
}

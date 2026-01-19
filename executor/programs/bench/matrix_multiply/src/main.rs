#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

const SIZE: usize = 64;

#[inline(never)]
fn matrix_multiply(a: &[[u32; SIZE]; SIZE], b: &[[u32; SIZE]; SIZE], result: &mut [[u32; SIZE]; SIZE]) {
    let mut i = 0;
    while i < SIZE {
        let mut j = 0;
        while j < SIZE {
            let mut sum = 0u32;
            let mut k = 0;
            while k < SIZE {
                sum = sum.wrapping_add(a[i][k].wrapping_mul(b[k][j]));
                k += 1;
            }
            result[i][j] = sum;
            j += 1;
        }
        i += 1;
    }
}

#[inline(never)]
fn init_matrix(m: &mut [[u32; SIZE]; SIZE], seed: u32) {
    let mut val = seed;
    let mut i = 0;
    while i < SIZE {
        let mut j = 0;
        while j < SIZE {
            m[i][j] = val % 100;
            val = val.wrapping_mul(1103515245).wrapping_add(12345);
            j += 1;
        }
        i += 1;
    }
}

#[unsafe(no_mangle)]
pub fn main() -> u32 {
    let mut a = [[0u32; SIZE]; SIZE];
    let mut b = [[0u32; SIZE]; SIZE];
    let mut result = [[0u32; SIZE]; SIZE];

    // Initialize matrices with pseudo-random values
    init_matrix(&mut a, 12345);
    init_matrix(&mut b, 67890);

    // Perform matrix multiplication
    matrix_multiply(&a, &b, &mut result);

    // Return checksum: sum of diagonal elements
    let mut checksum = 0u32;
    let mut i = 0;
    while i < SIZE {
        checksum = checksum.wrapping_add(result[i][i]);
        i += 1;
    }
    checksum
}

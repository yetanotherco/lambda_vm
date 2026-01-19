#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

const SIZE: usize = 12000;

#[inline(never)]
fn partition(arr: &mut [u32], len: usize) -> usize {
    let pivot = arr[len - 1];
    let mut i = 0;
    let mut j = 0;

    while j < len - 1 {
        if arr[j] <= pivot {
            let temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i += 1;
        }
        j += 1;
    }

    let temp = arr[i];
    arr[i] = arr[len - 1];
    arr[len - 1] = temp;

    i
}

#[inline(never)]
fn quicksort_range(arr: &mut [u32], start: usize, end: usize) {
    if end <= start + 1 {
        return;
    }
    let len = end - start;
    let pivot_idx = partition(&mut arr[start..end], len);
    let abs_pivot = start + pivot_idx;

    if pivot_idx > 0 {
        quicksort_range(arr, start, abs_pivot);
    }
    if abs_pivot + 1 < end {
        quicksort_range(arr, abs_pivot + 1, end);
    }
}

#[inline(never)]
fn init_array(arr: &mut [u32; SIZE], seed: u32) {
    let mut val = seed;
    let mut i = 0;
    while i < SIZE {
        arr[i] = val % 10000;
        val = val.wrapping_mul(1103515245).wrapping_add(12345);
        i += 1;
    }
}

#[inline(never)]
fn verify_sorted(arr: &[u32; SIZE]) -> bool {
    let mut i = 1;
    while i < SIZE {
        if arr[i] < arr[i - 1] {
            return false;
        }
        i += 1;
    }
    true
}

#[unsafe(no_mangle)]
pub fn main() -> u32 {
    let mut arr = [0u32; SIZE];

    // Initialize with pseudo-random values
    init_array(&mut arr, 42);

    // Sort
    quicksort_range(&mut arr, 0, SIZE);

    // Verify and return checksum
    if verify_sorted(&arr) {
        // Return first + last + middle elements as checksum
        arr[0].wrapping_add(arr[SIZE - 1]).wrapping_add(arr[SIZE / 2])
    } else {
        0 // Sort failed
    }
}

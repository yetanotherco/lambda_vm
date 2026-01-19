#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

const ARR_SIZE: usize = 1000;
const NUM_SEARCHES: usize = 30000;

#[inline(never)]
fn binary_search(arr: &[u32; ARR_SIZE], target: u32) -> i32 {
    let mut low: usize = 0;
    let mut high: usize = ARR_SIZE;

    while low < high {
        let mid = low + (high - low) / 2;
        if arr[mid] < target {
            low = mid + 1;
        } else if arr[mid] > target {
            high = mid;
        } else {
            return mid as i32;
        }
    }
    -1
}

#[inline(never)]
fn init_sorted_array(arr: &mut [u32; ARR_SIZE]) {
    let mut i = 0;
    while i < ARR_SIZE {
        arr[i] = (i * 3 + 1) as u32; // 1, 4, 7, 10, ...
        i += 1;
    }
}

#[unsafe(no_mangle)]
pub fn main() -> u32 {
    let mut arr = [0u32; ARR_SIZE];
    init_sorted_array(&mut arr);

    // Perform many searches and count successful ones
    let mut found_count = 0u32;
    let mut search_val = 1u32;
    let mut i = 0;

    while i < NUM_SEARCHES {
        let result = binary_search(&arr, search_val);
        if result >= 0 {
            found_count += 1;
        }
        // Generate next search value (some will hit, some will miss)
        search_val = search_val.wrapping_mul(1103515245).wrapping_add(12345);
        search_val = search_val % 3500; // Range that includes and exceeds array values
        i += 1;
    }

    found_count
}

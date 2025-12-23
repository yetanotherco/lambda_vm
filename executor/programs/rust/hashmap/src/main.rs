#![no_main]

use std::collections::HashMap;

use embedded_alloc::LlffHeap as Heap;
use riscv as _;

#[global_allocator]
static HEAP: Heap = Heap::empty();


#[unsafe(no_mangle)]
pub extern "C" fn sys_write(_: i32, _: *const u8, _: usize) -> isize {
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_panic(msg_ptr: *const u8, len: usize) {
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_load_8(ptr: *const u64, _order: i32) -> u64 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_fetch_add_4(
    ptr: *mut u32,
    val: u32,
    _order: i32,
) -> u32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_fetch_and_1(
    ptr: *mut u8,
    val: u8,
    _order: i32,
) -> u8 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_fetch_sub_4(
    ptr: *mut u32,
    val: u32,
    _order: i32,
) -> u32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_load_4(
    ptr: *const u32,
    _order: i32,
) -> u32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_load_1(
    ptr: *const u8,
    _order: i32,
) -> u8 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_store_1(
    ptr: *mut u8,
    val: u8,
    _order: i32,
) {
   
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_fetch_or_1(
    ptr: *mut u8,
    val: u8,
    _order: i32,
) -> u8 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_compare_exchange_8(
    ptr: *mut u64,
    expected: *mut u64,
    desired: u64,
    _weak: bool,
    _success_order: i32,
    _failure_order: i32,
) -> bool {
    false
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_rand(buf: *mut u8, len: usize) -> isize {
    0
}


#[unsafe(export_name = "main")]
pub fn main() -> i32 {
    {
        use core::mem::MaybeUninit;
        use core::ptr::addr_of_mut;
        const HEAP_SIZE: usize = 1024;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        let heap_ptr = addr_of_mut!(HEAP_MEM) as *mut u8;
        unsafe { HEAP.init(heap_ptr as usize, HEAP_SIZE) }
    }
    let mut map = HashMap::new();
    map.insert("one", 1);
    map.insert("two", 2);
    return map["two"] + map["one"];
}

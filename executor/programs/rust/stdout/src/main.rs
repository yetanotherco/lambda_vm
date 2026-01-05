#![no_main]

use core::arch::asm;

use embedded_alloc::LlffHeap as Heap;
use riscv as _;
use std::io::stdout;
use std::io::Write;
use core::ptr;

#[global_allocator]
static HEAP: Heap = Heap::empty();

/// This is a template for printing in the vm
pub fn print_string(s: &str) {
    unsafe {
        asm!(
            "mv a0, {ptr}",
            "mv a1, {len}",
            "ecall",
            ptr = in(reg) s.as_ptr(),
            len = in(reg) s.len(),
        );
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_write(fildes: i32, buf: *const u8, size: usize) -> isize {
    print_string("sys_write called\n");
    let content = unsafe { core::slice::from_raw_parts(buf, size) };
    print_string(&("SYS_WRITE: ".to_owned() + str::from_utf8(content).unwrap_or("<invalid utf8>"))); // Does the print of the sys write
    size.try_into().unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_panic(msg_ptr: *const u8, len: usize) {
    print_string("Sys panic called\n");
}

fn load_64(ptr: *const u8) -> u64 {
    unsafe {
        let mut v: u64 = 0;
        for i in 0..8 {
            v |= (*ptr.add(i) as u64) << (i * 8);
        }
        v
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_load_8(ptr: *const u64, _order: i32) -> u64 {
    print_string("__atomic_load_8 called\n");
    unsafe {
        if ptr.is_null() {
            return 0;
        }
        let p = ptr as *const u8;

        load_64(p)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_fetch_add_4(
    ptr: *mut u32,
    val: u32,
    _order: i32,
) -> u32 {
    print_string("__atomic_fetch_add_4 called\n");
    unsafe {
        let old = *ptr;
        *ptr = old.wrapping_add(val);
        old
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_fetch_and_1(
    ptr: *mut u8,
    val: u8,
    _order: i32,
) -> u8 {
    print_string("__atomic_fetch_and_1 called\n");
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_fetch_sub_4(
    ptr: *mut u32,
    val: u32,
    _order: i32,
) -> u32 {
    print_string("__atomic_fetch_sub_4 called\n");
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_load_4(
    ptr: *const u32,
    _order: i32,
) -> u32 {
    print_string("__atomic_load_4 called\n");
    unsafe {
        *ptr
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_load_1(
    ptr: *const u8,
    _order: i32,
) -> u8 {
    print_string("__atomic_load_1 called\n");
    unsafe {
        *ptr
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_store_1(
    ptr: *mut u8,
    val: u8,
    _order: i32,
) {
    print_string("__atomic_store_1 called\n");
   
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_fetch_or_1(
    ptr: *mut u8,
    val: u8,
    _order: i32,
) -> u8 {
    print_string("__atomic_fetch_or_1 called\n");
    unsafe {
        let old = *ptr;
        *ptr = old | val;
        old
    }
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
    print_string("__atomic_compare_exchange_8 called\n");
    unsafe {
        let current = load_64(ptr as *const u8);
        let expected_val = load_64(expected as *const u8);
        if current == expected_val {
            ptr::write_unaligned(ptr, desired);
            true
        } else {
            ptr::write_unaligned(expected, current);
            false
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_rand(buf: *mut u8, len: usize) -> isize {
    print_string("sys_rand called\n");
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn __atomic_store_8(ptr: *mut u8, val: u8, _order: i32) {
    print_string("__atomic_store_8 called\n");
}

/// This test does not verify the output, it only prints on stdout
#[unsafe(export_name = "main")]
pub fn main() -> i32 {
    {
        use core::mem::MaybeUninit;
        use core::ptr::addr_of_mut;
        const HEAP_SIZE: usize = 1024 * 64;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        let heap_ptr = addr_of_mut!(HEAP_MEM) as *mut u8;
        unsafe { HEAP.init(heap_ptr as usize, HEAP_SIZE) }
    }
    stdout().write(b"Hello from sys_write!\n").unwrap();
    return 1;
}

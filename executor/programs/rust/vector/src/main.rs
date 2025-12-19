#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn sys_alloc_aligned(nwords: usize, align: usize) -> *mut u8 {
    core::ptr::null_mut()
}

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


#[unsafe(export_name = "main")]
pub fn main() -> i32 {
    let vector = vec![1, 2, 3, 4, 5];
    return vector.iter().sum();
}

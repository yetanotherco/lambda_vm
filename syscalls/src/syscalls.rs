use core::arch::asm;
use core::ptr;

/// This is a template for printing in the vm
pub fn print_string(s: &str) {
    unsafe {
        asm!(
            "mv a0, {ptr}",
            "mv a1, {len}",
            "mv a7, {syscall_number}", // syscall number for print
            "ecall",
            ptr = in(reg) s.as_ptr(),
            len = in(reg) s.len(),
            syscall_number = in(reg) 1usize,
        );
    }
}

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_write(_fildes: i32, buf: *const u8, size: usize) -> isize {
    print_string("sys_write called\n");
    let content = unsafe { core::slice::from_raw_parts(buf, size) };
    print_string(&("SYS_WRITE: ".to_owned() + str::from_utf8(content).unwrap_or("<invalid utf8>"))); // Does the print of the sys write
    size.try_into().unwrap_or(-1)
}

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_panic(msg_ptr: *const u8, len: usize) {
    print_string("Sys panic called\n");
    unsafe {
        asm!(
            "mv a0, {ptr}",
            "mv a1, {len}",
            "mv a7, {syscall_number}", // syscall number for panic
            "ecall",
            ptr = in(reg) msg_ptr,
            len = in(reg) len,
            syscall_number = in(reg) 2usize,
        )
    }
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

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub extern "C" fn __atomic_load_8(ptr: *const u64, _order: i32) -> u64 {
    print_string("__atomic_load_8 called\n");
    if ptr.is_null() {
        return 0;
    }
    let p = ptr as *const u8;

    load_64(p)
}

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __atomic_fetch_add_4(ptr: *mut u32, val: u32, _order: i32) -> u32 {
    print_string("__atomic_fetch_add_4 called\n");
    unsafe {
        let old = *ptr;
        *ptr = old.wrapping_add(val);
        old
    }
}

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub extern "C" fn __atomic_fetch_and_1(_ptr: *mut u8, _val: u8, _order: i32) -> u8 {
    print_string("__atomic_fetch_and_1 called\n");
    0
}

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub extern "C" fn __atomic_fetch_sub_4(_ptr: *mut u32, _val: u32, _order: i32) -> u32 {
    print_string("__atomic_fetch_sub_4 called\n");
    0
}

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __atomic_load_4(ptr: *const u32, _order: i32) -> u32 {
    print_string("__atomic_load_4 called\n");
    unsafe { *ptr }
}

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __atomic_load_1(ptr: *const u8, _order: i32) -> u8 {
    print_string("__atomic_load_1 called\n");
    unsafe { *ptr }
}

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub extern "C" fn __atomic_store_1(_ptr: *mut u8, _val: u8, _order: i32) {
    print_string("__atomic_store_1 called\n");
}

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __atomic_fetch_or_1(ptr: *mut u8, val: u8, _order: i32) -> u8 {
    print_string("__atomic_fetch_or_1 called\n");
    unsafe {
        let old = *ptr;
        *ptr = old | val;
        old
    }
}

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __atomic_compare_exchange_8(
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

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub extern "C" fn sys_rand(_buf: *mut u8, _len: usize) -> isize {
    print_string("sys_rand called\n");
    0
}

/// # Safety
/// 
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub extern "C" fn __atomic_store_8(_ptr: *mut u8, _val: u8, _order: i32) {
    print_string("__atomic_store_8 called\n");
}

use lambda_vm_syscalls as syscalls;

unsafe extern "C" {
    fn memset(dst: *mut u8, fill: i32, count: usize) -> *mut u8;
}

pub fn main() {
    // 43 bytes = five eight-byte rows plus a three-byte tail, so one call yields
    // a first row, wide intermediate rows, tail rows and a terminal row.
    let mut buffer = [0u8; 43];
    let count = core::hint::black_box(buffer.len());

    unsafe {
        memset(buffer.as_mut_ptr(), 0x3C, count);
    }
    syscalls::syscalls::commit(&buffer);
}

use lambda_vm_syscalls as syscalls;

unsafe extern "C" {
    fn memset(dst: *mut u8, fill: i32, count: usize) -> *mut u8;
}

pub fn main() {
    // 43 bytes: the stub seeds the first 8 with stores and hands the chip the other
    // 35, which it moves as four eight-byte rows and three one-byte rows. The chain
    // ends on the row that moves the last byte; there is no terminal row.
    let mut buffer = [0u8; 43];
    let count = core::hint::black_box(buffer.len());

    unsafe {
        memset(buffer.as_mut_ptr(), 0x3C, count);
    }
    syscalls::syscalls::commit(&buffer);
}

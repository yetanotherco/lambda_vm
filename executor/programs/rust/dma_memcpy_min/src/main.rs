use lambda_vm_syscalls as syscalls;

unsafe extern "C" {
    fn memcpy(dst: *mut u8, src: *const u8, count: usize) -> *mut u8;
}

pub fn main() {
    let source = *b"DMA copies eight-byte rows and a short tail";
    let mut destination = [0u8; 43];
    let count = core::hint::black_box(destination.len());

    unsafe {
        memcpy(destination.as_mut_ptr(), source.as_ptr(), count);
    }
    syscalls::syscalls::commit(&destination);
}

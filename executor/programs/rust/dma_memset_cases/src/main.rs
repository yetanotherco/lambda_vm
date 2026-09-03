use lambda_vm_syscalls as syscalls;

unsafe extern "C" {
    fn memset(dst: *mut u8, fill: i32, count: usize) -> *mut u8;
}

/// `black_box` on the count keeps LLVM from turning these into inline stores,
/// so every call really does reach the strong `memset` symbol and the DMA ecall.
#[inline(never)]
fn dma_set(dst: *mut u8, fill: i32, count: usize) -> *mut u8 {
    let count = core::hint::black_box(count);
    unsafe { memset(dst, fill, count) }
}

pub fn main() {
    let mut buffer = [0u8; 777];

    // Every row-schedule boundary: empty, sub-tail, exact widths, the 256-byte
    // per-ecall cap, and one length that forces several chunked ecalls.
    for count in [0usize, 1, 7, 8, 9, 31, 32, 33, 127, 128, 255, 256] {
        buffer.fill(0xA5);
        let returned = dma_set(buffer.as_mut_ptr(), 0x3C, count);
        assert_eq!(returned, buffer.as_mut_ptr());
        assert!(buffer[..count].iter().all(|&byte| byte == 0x3C));
        assert!(buffer[count..].iter().all(|&byte| byte == 0xA5));
    }

    // More than one chunk: 777 bytes becomes four bounded DMA ecalls.
    buffer.fill(0);
    dma_set(buffer.as_mut_ptr(), 0x5A, buffer.len());
    assert!(buffer.iter().all(|&byte| byte == 0x5A));

    // Zero is the fill almost every real caller passes (`vec![0; n]` and the
    // allocator's `alloc_zeroed`), and it is the one value a dropped write is
    // indistinguishable from on a fresh buffer — so start from 0xA5.
    buffer.fill(0xA5);
    dma_set(buffer.as_mut_ptr(), 0, 100);
    assert!(buffer[..100].iter().all(|&byte| byte == 0));
    assert!(buffer[100..].iter().all(|&byte| byte == 0xA5));

    // The guest stub masks the fill to its low byte, matching C's
    // `memset(void*, int, size_t)` writing `(unsigned char)c`.
    buffer.fill(0);
    dma_set(buffer.as_mut_ptr(), 0x1FF, 64);
    assert!(buffer[..64].iter().all(|&byte| byte == 0xFF));

    // A negative int sign-extends to 0xFFFF_FFFF_FFFF_FFFF under lp64; the
    // `andi` is what keeps the executor from rejecting it as a wide fill.
    buffer.fill(0);
    dma_set(buffer.as_mut_ptr(), -1, 32);
    assert!(buffer[..32].iter().all(|&byte| byte == 0xFF));

    // Unaligned destination that also crosses a 4 KiB page boundary.
    let mut page_buffer = [0u8; 8192];
    let to_boundary = 4096 - (page_buffer.as_ptr() as usize & 4095);
    let offset = to_boundary.saturating_sub(5);
    dma_set(unsafe { page_buffer.as_mut_ptr().add(offset) }, 0x77, 256);
    assert!(page_buffer[offset..offset + 256]
        .iter()
        .all(|&byte| byte == 0x77));

    syscalls::syscalls::commit(b"dma-memset-ok");
}

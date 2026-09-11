use lambda_vm_syscalls as syscalls;

unsafe extern "C" {
    fn memcpy(dst: *mut u8, src: *const u8, count: usize) -> *mut u8;
}

#[inline(never)]
fn dma_copy(dst: *mut u8, src: *const u8, count: usize) -> *mut u8 {
    let count = core::hint::black_box(count);
    unsafe { memcpy(dst, src, count) }
}

fn fill_pattern(bytes: &mut [u8], seed: u8) {
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(37).wrapping_add(seed);
    }
}

pub fn main() {
    let mut source = [0u8; 777];
    let mut destination = [0xA5u8; 777];
    fill_pattern(&mut source, 11);

    for count in [0usize, 1, 7, 8, 9, 31, 32, 33, 127, 128, 255, 256] {
        destination.fill(0xA5);
        let returned = dma_copy(destination.as_mut_ptr(), source.as_ptr(), count);
        assert_eq!(returned, destination.as_mut_ptr());
        assert_eq!(&destination[..count], &source[..count]);
        assert!(destination[count..].iter().all(|&byte| byte == 0xA5));
    }

    // More than one chunk: 777 bytes becomes four bounded DMA ecalls.
    destination.fill(0);
    dma_copy(destination.as_mut_ptr(), source.as_ptr(), source.len());
    assert_eq!(destination, source);

    // Snapshot semantics in both overlap directions.
    let mut forward = [0u8; 320];
    fill_pattern(&mut forward, 23);
    let forward_before = forward;
    dma_copy(
        unsafe { forward.as_mut_ptr().add(17) },
        forward.as_ptr(),
        256,
    );
    assert_eq!(&forward[17..273], &forward_before[..256]);

    let mut backward = [0u8; 320];
    fill_pattern(&mut backward, 41);
    let backward_before = backward;
    dma_copy(
        backward.as_mut_ptr(),
        unsafe { backward.as_ptr().add(17) },
        256,
    );
    assert_eq!(&backward[..256], &backward_before[17..273]);

    // Force both operands to cross a 4 KiB page boundary.
    let mut page_source = [0u8; 8192];
    let mut page_destination = [0u8; 8192];
    fill_pattern(&mut page_source, 67);
    let src_to_boundary = 4096 - (page_source.as_ptr() as usize & 4095);
    let dst_to_boundary = 4096 - (page_destination.as_ptr() as usize & 4095);
    let src_offset = src_to_boundary.saturating_sub(3);
    let dst_offset = dst_to_boundary.saturating_sub(5);
    dma_copy(
        unsafe { page_destination.as_mut_ptr().add(dst_offset) },
        unsafe { page_source.as_ptr().add(src_offset) },
        256,
    );
    assert_eq!(
        &page_destination[dst_offset..dst_offset + 256],
        &page_source[src_offset..src_offset + 256]
    );

    syscalls::syscalls::commit(b"dma-cases-ok");
}

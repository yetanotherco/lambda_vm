use lambda_vm_syscalls as syscalls;

unsafe extern "C" {
    fn memmove(dst: *mut u8, src: *const u8, count: usize) -> *mut u8;
}

#[inline(never)]
fn dma_move(dst: *mut u8, src: *const u8, count: usize) -> *mut u8 {
    let count = core::hint::black_box(count);
    unsafe { memmove(dst, src, count) }
}

fn fill_pattern(bytes: &mut [u8], seed: u8) {
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(37).wrapping_add(seed);
    }
}

pub fn main() {
    // Disjoint regions behave like memcpy.
    let mut source = [0u8; 777];
    let mut destination = [0xA5u8; 777];
    fill_pattern(&mut source, 11);
    for count in [0usize, 1, 7, 8, 255, 256, 257, 777] {
        destination.fill(0xA5);
        let returned = dma_move(destination.as_mut_ptr(), source.as_ptr(), count);
        assert_eq!(returned, destination.as_mut_ptr());
        assert_eq!(&destination[..count], &source[..count]);
        assert!(destination[count..].iter().all(|&b| b == 0xA5));
    }

    // Forward overlap (dst inside [src, src+n)) is the case that needs BACKWARD
    // chunking; a forward-chunked copy corrupts it once n exceeds one chunk.
    // Offsets below and above 256 exercise both sides of the chunk boundary.
    for (offset, count) in [(1usize, 600usize), (17, 600), (255, 600), (256, 600), (300, 700), (4, 8)] {
        let mut buffer = [0u8; 1600];
        fill_pattern(&mut buffer, 23);
        let before = buffer;
        dma_move(
            unsafe { buffer.as_mut_ptr().add(offset) },
            buffer.as_ptr(),
            count,
        );
        assert_eq!(&buffer[offset..offset + count], &before[..count]);
        // Bytes below the destination must be untouched.
        assert_eq!(&buffer[..offset], &before[..offset]);
    }

    // Backward overlap (dst below src) stays forward-chunked.
    for (offset, count) in [(1usize, 600usize), (17, 600), (300, 700)] {
        let mut buffer = [0u8; 1600];
        fill_pattern(&mut buffer, 41);
        let before = buffer;
        dma_move(
            buffer.as_mut_ptr(),
            unsafe { buffer.as_ptr().add(offset) },
            count,
        );
        assert_eq!(&buffer[..count], &before[offset..offset + count]);
    }

    // Exact aliasing must be a no-op.
    let mut same = [0u8; 300];
    fill_pattern(&mut same, 7);
    let before = same;
    dma_move(same.as_mut_ptr(), same.as_ptr(), 300);
    assert_eq!(same, before);

    syscalls::syscalls::commit(b"dma-memmove-ok");
}

//! Every copy here is emitted by the compiler: nothing declares or names
//! `memcpy`. The guest computes the same output whether or not the strong
//! `memcpy` symbol won the guest's link, so its DMA ecall count — not its
//! output — is what pins the symbol resolution.

use lambda_vm_syscalls as syscalls;

#[inline(never)]
fn copy_slice(destination: &mut [u8], source: &[u8]) {
    destination.copy_from_slice(source);
}

fn fill_pattern(bytes: &mut [u8], seed: u8) {
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(31).wrapping_add(seed);
    }
}

pub fn main() {
    let mut source = [0u8; 512];
    fill_pattern(&mut source, 7);
    // A runtime-sized length keeps LLVM from lowering the copies inline.
    let length = core::hint::black_box(source.len());

    let mut destination = [0u8; 512];
    copy_slice(&mut destination[..length], &source[..length]);
    assert_eq!(destination, source);

    let mut grown = Vec::new();
    grown.extend_from_slice(&source[..length]);
    assert_eq!(grown.as_slice(), &source[..]);

    syscalls::syscalls::commit(b"dma-implicit-ok");
}

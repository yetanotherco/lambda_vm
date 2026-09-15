//! SHA-256 using the compression accelerator specified in spec/sha256.typ.
//! State and chunk are big-endian bytes; the ecall permits arbitrary alignment.
#[inline]
pub fn compress(state: &mut [u8; 32], chunk: &[u8; 64]) {
    #[cfg(target_arch = "riscv64")]
    unsafe {
        core::arch::asm!("ecall", in("a7") u64::MAX, in("a0") state.as_mut_ptr(), in("a1") chunk.as_ptr(), options(nostack));
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        let _ = (state, chunk);
        panic!("SHA256 accelerator requires LambdaVM");
    }
}
/// Standard SHA-256, including IV, length encoding, and one or two padding blocks.
pub fn sha256(input: &[u8]) -> [u8; 32] {
    let mut state = [
        0x6a, 0x09, 0xe6, 0x67, 0xbb, 0x67, 0xae, 0x85, 0x3c, 0x6e, 0xf3, 0x72, 0xa5, 0x4f, 0xf5,
        0x3a, 0x51, 0x0e, 0x52, 0x7f, 0x9b, 0x05, 0x68, 0x8c, 0x1f, 0x83, 0xd9, 0xab, 0x5b, 0xe0,
        0xcd, 0x19,
    ];
    let bit_len = (input.len() as u64)
        .checked_mul(8)
        .expect("SHA256 input too long");
    let mut chunks = input.chunks_exact(64);
    for chunk in &mut chunks {
        compress(&mut state, chunk.try_into().unwrap());
    }
    let rest = chunks.remainder();
    let mut last = [0u8; 64];
    last[..rest.len()].copy_from_slice(rest);
    last[rest.len()] = 0x80;
    if rest.len() >= 56 {
        compress(&mut state, &last);
        last = [0; 64];
    }
    last[56..].copy_from_slice(&bit_len.to_be_bytes());
    compress(&mut state, &last);
    state
}

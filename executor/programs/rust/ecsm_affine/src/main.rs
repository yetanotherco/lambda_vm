use lambda_vm_syscalls as syscalls;

/// Computes 5·G on secp256k1 via the **affine** ECSM precompile (`ecsm_mul_affine`) and
/// commits the 64-byte result point `xR‖yR` as public output. Exercises the affine ecall
/// end-to-end (IS_AFFINE=1: yG read from memory, yR written back).
pub fn main() {
    // secp256k1 generator (Gx, Gy), big-endian then reversed to little-endian.
    let mut gx: [u8; 32] = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    gx.reverse();
    let mut gy: [u8; 32] = [
        0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65, 0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08,
        0xA8, 0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19, 0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10,
        0xD4, 0xB8,
    ];
    gy.reverse();

    // Full input point as a contiguous 64-byte buffer: xG at [0..32], yG at [32..64].
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(&gx);
    input[32..].copy_from_slice(&gy);

    let mut k = [0u8; 32];
    k[0] = 5;

    let mut out = [0u8; 64];
    syscalls::syscalls::ecsm_mul_affine(&mut out, &input, &k);
    syscalls::syscalls::commit(&out);
}

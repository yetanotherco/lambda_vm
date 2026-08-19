use lambda_vm_syscalls as syscalls;

/// Computes 5·G on secp256k1 via the ECSM precompile (Rust-guest path) and commits the
/// 32-byte x-coordinate as public output.
pub fn main() {
    // secp256k1 Gx, given big-endian then reversed to little-endian for the precompile.
    let mut xg: [u8; 32] = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    xg.reverse();

    let mut k = [0u8; 32];
    k[0] = 5;

    // The precompile writes [xR ‖ yR ‖ yG]; only xR is committed. 8-byte aligned so the
    // twelve doubleword accesses take the aligned memory path (MEMW_A).
    #[repr(C, align(8))]
    struct Align8<const N: usize>([u8; N]);
    let mut out = Align8([0u8; 96]);
    syscalls::syscalls::ecsm_mul(&mut out.0, &xg, &k);
    syscalls::syscalls::commit(&out.0[..32]);
}

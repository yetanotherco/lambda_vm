use lambda_vm_syscalls as syscalls;

/// ECSM precompile benchmark: chains `ITERATIONS` full 256-bit scalar
/// multiplications (k = N-1 exercises the complete double-and-add ladder),
/// feeding each result back as the next base point.
const ITERATIONS: usize = 10;

pub fn main() {
    // secp256k1 Gx, big-endian then reversed to little-endian.
    // `Align8` keeps every operand on the aligned memory path (MEMW_A).
    let mut xg = syscalls::syscalls::Align8([
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ]);
    xg.0.reverse();

    // k = N - 1 (largest valid scalar), big-endian then reversed to little-endian.
    let mut k = syscalls::syscalls::Align8([
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFE, 0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B, 0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36,
        0x41, 0x40,
    ]);
    k.0.reverse();

    // The precompile writes [xR ‖ yR ‖ yG]; the chain feeds xR back as the next base point.
    let mut out = syscalls::syscalls::Align8::<96>::zeroed();
    for _ in 0..ITERATIONS {
        syscalls::syscalls::ecsm_mul(&mut out, &xg, &k);
        xg.0.copy_from_slice(&out.0[..32]);
    }
    syscalls::syscalls::commit(&out.0[..32]);
}

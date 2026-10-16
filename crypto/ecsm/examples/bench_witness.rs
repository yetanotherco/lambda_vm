//! Timing harness for `compute_witness` (one ECSM ecall's witness).
//! Run: cargo run --release --example bench_witness -p ecsm

use std::time::Instant;

// secp256k1 generator x-coordinate, big-endian.
const GX_BE: [u8; 32] = [
    0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b, 0x07,
    0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98,
];

fn le32(be: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = be[31 - i];
    }
    out
}

fn main() {
    // Worst-case-ish scalar: high popcount → ~380 double/add steps.
    let k_be: [u8; 32] = [
        0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd,
        0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x11, 0x22,
    ];
    let k_le = le32(&k_be);
    let xg_le = le32(&GX_BE);

    for _ in 0..2 {
        std::hint::black_box(ecsm::compute_witness(&k_le, &xg_le).unwrap());
    }

    const N: u32 = 20;
    let t = Instant::now();
    for _ in 0..N {
        std::hint::black_box(ecsm::compute_witness(&k_le, &xg_le).unwrap());
    }
    let d = t.elapsed() / N;
    println!("compute_witness: {d:?} per call ({N} runs)");
}

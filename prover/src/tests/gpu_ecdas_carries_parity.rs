//! Bit-exact parity: the device `ecdas_carries` kernel must reproduce, limb-for-limb, the per-step
//! carry witness the host `ecsm::witness::build_step` computes (`carries_lambda/xr/yr` + `limb_carries`)
//! — the ~190ms `conv` limb-convolution work moved to GPU (Step C). The EC scalar-mult
//! (`replay_double_and_add`, k256) and the tiny quotients stay on CPU; only the carries move.
//!
//! Uses the `ecsm` crate directly with a spread of scalars (covering both double `op=0` and add `op=1`
//! steps, small and near-256-bit) — the kernel is input-agnostic, so diverse witnesses fully exercise it.
//!
//! `cargo test -p lambda-vm-prover --release --features cuda --lib gpu_ecdas_carries -- --ignored --nocapture`

use math_cuda::precompile::{ECDAS_BSTRIDE, ECDAS_CSTRIDE};

/// secp256k1 generator x, little-endian (a valid on-curve x with recoverable y).
const GX_LE: [u8; 32] = [
    0x98, 0x17, 0xF8, 0x16, 0x5B, 0x81, 0xF2, 0x59, 0xD9, 0x28, 0xCE, 0x2D, 0xDB, 0xFC, 0x9B, 0x02,
    0x07, 0x0B, 0x87, 0xCE, 0x95, 0x62, 0xA0, 0x55, 0xAC, 0xBB, 0xDC, 0xF9, 0x7E, 0x66, 0xBE, 0x79,
];

/// A 32-byte little-endian scalar from a u64 low part plus an optional high byte (byte 31), so we can
/// reach near-256-bit `k` (many ECDAS steps) while staying < N (top byte ≤ 0x7F ⇒ k < N).
fn k_le(low: u64, hi_byte: u8) -> [u8; 32] {
    let mut k = [0u8; 32];
    k[..8].copy_from_slice(&low.to_le_bytes());
    k[31] = hi_byte;
    k
}

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_ecdas_carries_matches_build_step() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping gpu_ecdas_carries_matches_build_step: no CUDA backend: {e:?}");
        return;
    }

    // Scalars chosen to exercise diverse double/add schedules and limb values.
    let scalars: &[[u8; 32]] = &[
        k_le(5, 0),
        k_le(0xABCD, 0),
        k_le(0xFFFF_FFFF_FFFF_FFFF, 0),
        k_le(0x0123_4567_89AB_CDEF, 0x7F),
        k_le(0xDEAD_BEEF_CAFE_1234, 0x40),
        k_le(0x8000_0000_0000_0001, 0x01),
        k_le(2, 0x7E),
    ];

    // Collect every ECDAS step across all witnesses; pack the device input bytes (same layout as
    // `build_ecdas_resident_table`) and the expected CPU carries.
    let mut bytes: Vec<u8> = Vec::new();
    let mut expected: Vec<i64> = Vec::new();
    let mut n_steps = 0usize;
    for k in scalars {
        let w = ecsm::witness::compute_witness(k, &GX_LE).expect("ecsm witness");
        for s in &w.steps {
            bytes.extend_from_slice(&s.x_g);
            bytes.extend_from_slice(&s.y_g);
            bytes.extend_from_slice(&s.x_a);
            bytes.extend_from_slice(&s.y_a);
            bytes.push(s.round);
            bytes.push(s.op);
            bytes.extend_from_slice(&s.x_r);
            bytes.extend_from_slice(&s.y_r);
            bytes.extend_from_slice(&s.lambda);
            bytes.extend_from_slice(&s.q0);
            bytes.extend_from_slice(&s.q1);
            bytes.extend_from_slice(&s.q2);
            bytes.push(s.next_op);
            expected.extend_from_slice(&s.c0);
            expected.extend_from_slice(&s.c1);
            expected.extend_from_slice(&s.c2);
            n_steps += 1;
        }
    }
    assert!(n_steps > 0, "expected some ECDAS steps");
    assert_eq!(bytes.len(), n_steps * ECDAS_BSTRIDE, "packed bytes length");
    assert_eq!(expected.len(), n_steps * ECDAS_CSTRIDE, "expected carries length");

    let got = math_cuda::precompile::gpu_build_ecdas_carries(&bytes, n_steps)
        .expect("device ecdas_carries");
    assert_eq!(got.len(), expected.len(), "carries length");

    for step in 0..n_steps {
        for (rel, name) in ["c0", "c1", "c2"].iter().enumerate() {
            for limb in 0..64 {
                let idx = step * ECDAS_CSTRIDE + rel * 64 + limb;
                assert_eq!(
                    got[idx], expected[idx],
                    "{name}[{limb}] @ step {step}: device {} != cpu {}",
                    got[idx], expected[idx]
                );
            }
        }
    }
    println!("gpu_ecdas_carries parity OK over {n_steps} steps ({} scalars)", scalars.len());
}

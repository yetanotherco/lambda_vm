//! Unit tests for ECSM/ECDAS witness generation (relocated from `witness.rs`).

use crypto_bigint::{Encoding, U256};

use crate::witness::compute_witness;
use crate::{n, scalar_mul_x};

fn gx_le() -> [u8; 32] {
    U256::from_be_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798").to_le_bytes()
}

/// Drives `compute_witness` (whose internal asserts validate every carry/quotient)
/// across many scalars, and cross-checks the result against the reference scalar mul.
#[test]
fn witness_is_self_consistent_for_many_scalars() {
    let gx = gx_le();
    // small scalars plus bit patterns that exercise add/double scheduling
    let scalars: &[u64] = &[1, 2, 3, 4, 5, 7, 8, 0xFF, 0x101, 0xABCD, 0xFFFF, 123456789];
    for &kv in scalars {
        let k = U256::from(kv).to_le_bytes();
        let w = compute_witness(&k, &gx).expect("witness");
        // final point matches reference
        assert_eq!(
            w.x_r,
            scalar_mul_x(&k, &gx).expect("reference scalar mul"),
            "k = {kv}"
        );
        // len_k is the true MSB position
        assert_eq!(w.len_k as u32, 63 - (kv.leading_zeros()), "k = {kv}");
    }
}

#[test]
fn k_one_has_no_ecdas_steps() {
    let w = compute_witness(&U256::ONE.to_le_bytes(), &gx_le()).expect("witness");
    assert!(w.steps.is_empty());
    assert_eq!(w.x_r, w.x_g); // 1·G = G
    assert_eq!(w.len_k, 0);
}

#[test]
fn ecdas_step_schedule_matches_double_and_add() {
    // k = 5 = 0b101: double(G)->2G [bit1=0], double(2G)->4G [bit0=1], add(4G,G)->5G.
    let w = compute_witness(&U256::from(5u32).to_le_bytes(), &gx_le()).expect("witness");
    assert_eq!(w.len_k, 2);
    let ops: Vec<(u8, u8, u8)> = w.steps.iter().map(|s| (s.round, s.op, s.next_op)).collect();
    assert_eq!(ops, vec![(1, 0, 0), (0, 0, 1), (0, 1, 0)]);
}

#[test]
fn witness_works_near_curve_order() {
    let gx = gx_le();
    let w = compute_witness(&n().wrapping_sub(&U256::ONE).to_le_bytes(), &gx).expect("witness");
    assert_eq!(w.x_r, gx); // (N-1)·G = -G shares x with G
    assert_eq!(w.len_k, 255);
}

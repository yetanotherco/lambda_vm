//! Unit tests for ECSM/ECDAS witness generation (relocated from `witness.rs`).

use num_bigint::BigUint;

use crate::witness::compute_witness;
use crate::{n, scalar_mul_full, scalar_mul_x, to_le_32};

fn gx_le() -> [u8; 32] {
    let gx = BigUint::parse_bytes(
        b"79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
        16,
    )
    .expect("valid generator x hex");
    to_le_32(&gx)
}

/// Drives `compute_witness` (whose internal asserts validate every carry/quotient)
/// across many scalars, and cross-checks the result against the reference scalar mul.
#[test]
fn witness_is_self_consistent_for_many_scalars() {
    let gx = gx_le();
    // small scalars plus bit patterns that exercise add/double scheduling
    let scalars: &[u64] = &[1, 2, 3, 4, 5, 7, 8, 0xFF, 0x101, 0xABCD, 0xFFFF, 123456789];
    for &kv in scalars {
        let k = to_le_32(&BigUint::from(kv));
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
    let w = compute_witness(&to_le_32(&BigUint::from(1u8)), &gx_le()).expect("witness");
    assert!(w.steps.is_empty());
    assert_eq!(w.x_r, w.x_g); // 1·G = G
    assert_eq!(w.len_k, 0);
}

#[test]
fn ecdas_step_schedule_matches_double_and_add() {
    // k = 5 = 0b101: double(G)->2G [bit1=0], double(2G)->4G [bit0=1], add(4G,G)->5G.
    let w = compute_witness(&to_le_32(&BigUint::from(5u8)), &gx_le()).expect("witness");
    assert_eq!(w.len_k, 2);
    let ops: Vec<(u8, u8, u8)> = w.steps.iter().map(|s| (s.round, s.op, s.next_op)).collect();
    assert_eq!(ops, vec![(1, 0, 0), (0, 0, 1), (0, 1, 0)]);
}

#[test]
fn witness_works_near_curve_order() {
    let gx = gx_le();
    let w = compute_witness(&to_le_32(&(n() - BigUint::from(1u8))), &gx).expect("witness");
    assert_eq!(w.x_r, gx); // (N-1)·G = -G shares x with G
    assert_eq!(w.len_k, 255);
}

/// The executor writes `scalar_mul_full`'s bytes into guest memory while the prover writes the
/// witness columns, and the MEMW bus asserts the two images are the same claim. Publishing
/// `yR` and `yG` put two more values under that coupling — the executor takes them from k256's
/// affine scalar multiplication, the witness from its own double-and-add replay — so the two
/// disagreeing anywhere (a sign, a representative) would surface only as an unbalanced bus on
/// whichever scalar hit it. Pin the agreement directly instead.
#[test]
fn witness_matches_the_executor_output_image() {
    let gx = gx_le();
    let scalars = [
        BigUint::from(1u8),
        BigUint::from(5u8),
        BigUint::from(0xABCDEFu64),
        (n() - BigUint::from(1u8)) / BigUint::from(2u8),
        n() - BigUint::from(1u8),
    ];
    for k_big in scalars {
        let k = to_le_32(&k_big);
        let w = compute_witness(&k, &gx).expect("witness");
        let (x_r, y_r, y_g) = scalar_mul_full(&k, &gx).expect("executor output");
        assert_eq!(w.x_r, x_r, "xR disagrees for k = {k_big}");
        assert_eq!(w.y_r, y_r, "yR disagrees for k = {k_big}");
        assert_eq!(w.y_g, y_g, "yG disagrees for k = {k_big}");
        assert_eq!(y_g[0] & 1, 0, "xG is lifted to its even root on both sides");
    }
}

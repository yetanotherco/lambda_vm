//! Parity tests pinning the production k256 fast path to the BigUint reference
//! replay (relocated from `curve.rs::parity_tests`).

use num_bigint::BigUint;

use crate::curve::{AffinePoint, recover_y_canonical, replay_double_and_add, scalar_mul_affine_x};
use crate::n;
use crate::tests::reference::replay_double_and_add_reference;

/// secp256k1 generator (even y), via the canonical y recovery.
fn generator() -> AffinePoint {
    let gx = BigUint::parse_bytes(
        b"79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
        16,
    )
    .expect("valid generator x hex");
    let gy = recover_y_canonical(&gx).expect("G on curve");
    AffinePoint { x: gx, y: gy }
}

fn be(hex: &[u8]) -> BigUint {
    BigUint::parse_bytes(hex, 16).expect("valid hex literal")
}

/// The k256 fast path must produce byte-identical `StepPts` (points + λ) and the
/// same final point as the BigUint reference, across small, structured, large and
/// near-order scalars. This pins the audited fast path to the spec-faithful reference.
#[test]
fn k256_replay_matches_reference() {
    let g = generator();
    let mut scalars: Vec<BigUint> = (1u64..40).map(BigUint::from).collect();
    for &kv in &[
        0xFFu64,
        0x101,
        0xABCD,
        0xFFFF,
        0x1_0000,
        1 << 20,
        123_456_789,
        u64::MAX,
    ] {
        scalars.push(BigUint::from(kv));
    }
    // large 256-bit scalars (must stay < N) and the order boundary
    scalars.push(be(
        b"0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF",
    ));
    scalars.push(be(
        b"7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0",
    ));
    scalars.push(&n() / BigUint::from(2u8));
    scalars.push(&n() - BigUint::from(1u8));

    for k in scalars {
        let (steps, result) = replay_double_and_add(&k, &g);
        let (steps_ref, result_ref) = replay_double_and_add_reference(&k, &g);
        assert_eq!(result, result_ref, "final point mismatch for k = {k}");
        assert_eq!(steps, steps_ref, "step list mismatch for k = {k}");
    }
}

/// Same parity sweep with a non-generator base point: production feeds the
/// replay guest-supplied points (e.g. the recovered R in ecrecover), and every
/// other test uses G.
#[test]
fn k256_replay_matches_reference_non_generator_base() {
    let g = generator();
    let base_x = scalar_mul_affine_x(&BigUint::from(5u64), &g);
    let base = AffinePoint {
        y: recover_y_canonical(&base_x).expect("base on curve"),
        x: base_x,
    };
    let mut scalars: Vec<BigUint> = (1u64..40).map(BigUint::from).collect();
    for &kv in &[0xFFu64, 0xABCD, 1 << 20, 123_456_789, u64::MAX] {
        scalars.push(BigUint::from(kv));
    }
    scalars.push(&n() / BigUint::from(2u8));
    scalars.push(&n() - BigUint::from(1u8));

    for k in scalars {
        let (steps, result) = replay_double_and_add(&k, &base);
        let (steps_ref, result_ref) = replay_double_and_add_reference(&k, &base);
        assert_eq!(result, result_ref, "final point mismatch for k = {k} (non-G base)");
        assert_eq!(steps, steps_ref, "step list mismatch for k = {k} (non-G base)");
    }
}

/// The executor's fast path (`scalar_mul_affine_x`) and the prover's replay must agree
/// on `x(k·G)`: the executor writes it to guest memory and the prover proves it, so any
/// divergence would make a correct execution unprovable. They run through two distinct
/// k256 entry points (native scalar-mul vs projective double-and-add), so pin them here.
#[test]
fn executor_and_replay_agree_on_result_x() {
    let g = generator();
    let mut scalars: Vec<BigUint> = (1u64..40).map(BigUint::from).collect();
    for &kv in &[0xFFu64, 0xABCD, 1 << 20, 123_456_789, u64::MAX] {
        scalars.push(BigUint::from(kv));
    }
    scalars.push(&n() / BigUint::from(2u8));
    scalars.push(&n() - BigUint::from(1u8));

    for k in scalars {
        let (_steps, result) = replay_double_and_add(&k, &g);
        let exec_x = scalar_mul_affine_x(&k, &g);
        assert_eq!(result.x, exec_x, "executor/replay x mismatch for k = {k}");
    }
}

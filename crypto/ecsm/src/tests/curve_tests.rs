//! Parity tests pinning the production k256 fast path to the U256 reference
//! replay (relocated from `curve.rs::parity_tests`).

use crypto_bigint::{NonZero, U256};

use crate::curve::{AffinePoint, recover_y_canonical, replay_double_and_add, scalar_mul_affine_x};
use crate::n;
use crate::tests::reference::replay_double_and_add_reference;

/// secp256k1 generator (even y), via the canonical y recovery.
fn generator() -> AffinePoint {
    let gx = U256::from_be_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    let gy = recover_y_canonical(&gx).expect("G on curve");
    AffinePoint { x: gx, y: gy }
}

/// The k256 fast path must produce byte-identical `StepPts` (points + λ) and the
/// same final point as the U256 reference, across small, structured, large and
/// near-order scalars. This pins the audited fast path to the spec-faithful reference.
#[test]
fn k256_replay_matches_reference() {
    let g = generator();
    let mut scalars: Vec<U256> = (1u64..40).map(U256::from).collect();
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
        scalars.push(U256::from(kv));
    }
    // large 256-bit scalars (must stay < N) and the order boundary
    scalars.push(U256::from_be_hex(
        "0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF",
    ));
    scalars.push(U256::from_be_hex(
        "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0",
    ));
    let two = NonZero::new(U256::from(2u32)).expect("2 != 0");
    scalars.push(n().div_rem(&two).0);
    scalars.push(n().wrapping_sub(&U256::ONE));

    for k in scalars {
        let (steps, result) = replay_double_and_add(&k, &g);
        let (steps_ref, result_ref) = replay_double_and_add_reference(&k, &g);
        assert_eq!(result, result_ref, "final point mismatch for k = {k:?}");
        assert_eq!(steps, steps_ref, "step list mismatch for k = {k:?}");
    }
}

/// The executor's fast path (`scalar_mul_affine_x`) and the prover's replay must agree
/// on `x(k·G)`: the executor writes it to guest memory and the prover proves it, so any
/// divergence would make a correct execution unprovable.
#[test]
fn executor_and_replay_agree_on_result_x() {
    let g = generator();
    let mut scalars: Vec<U256> = (1u64..40).map(U256::from).collect();
    for &kv in &[0xFFu64, 0xABCD, 1 << 20, 123_456_789, u64::MAX] {
        scalars.push(U256::from(kv));
    }
    let two = NonZero::new(U256::from(2u32)).expect("2 != 0");
    scalars.push(n().div_rem(&two).0);
    scalars.push(n().wrapping_sub(&U256::ONE));

    for k in scalars {
        let (_steps, result) = replay_double_and_add(&k, &g);
        let exec_x = scalar_mul_affine_x(&k, &g);
        assert_eq!(result.x, exec_x, "executor/replay x mismatch for k = {k:?}");
    }
}

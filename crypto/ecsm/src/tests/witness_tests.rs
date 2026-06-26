//! Unit tests for ECSM/ECDAS witness generation (relocated from `witness.rs`).

use crypto_bigint::{NonZero, U256, U512, U1024};

use crate::witness::compute_witness;
use crate::{n, scalar_mul_x};

fn gx_le() -> [u8; 32] {
    U256::from_be_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798").to_le_bytes().into()
}

/// Drives `compute_witness` (whose internal asserts validate every carry/quotient)
/// across many scalars, and cross-checks the result against the reference scalar mul.
#[test]
fn witness_is_self_consistent_for_many_scalars() {
    let gx = gx_le();
    // small scalars plus bit patterns that exercise add/double scheduling
    let scalars: &[u64] = &[1, 2, 3, 4, 5, 7, 8, 0xFF, 0x101, 0xABCD, 0xFFFF, 123456789];
    for &kv in scalars {
        let k: [u8; 32] = U256::from(kv).to_le_bytes().into();
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
    let w = compute_witness(&U256::ONE.to_le_bytes().into(), &gx_le()).expect("witness");
    assert!(w.steps.is_empty());
    assert_eq!(w.x_r, w.x_g); // 1·G = G
    assert_eq!(w.len_k, 0);
}

#[test]
fn ecdas_step_schedule_matches_double_and_add() {
    // k = 5 = 0b101: double(G)->2G [bit1=0], double(2G)->4G [bit0=1], add(4G,G)->5G.
    let w = compute_witness(&U256::from(5u32).to_le_bytes().into(), &gx_le()).expect("witness");
    assert_eq!(w.len_k, 2);
    let ops: Vec<(u8, u8, u8)> = w.steps.iter().map(|s| (s.round, s.op, s.next_op)).collect();
    assert_eq!(ops, vec![(1, 0, 0), (0, 0, 1), (0, 1, 0)]);
}

#[test]
fn witness_works_near_curve_order() {
    let gx = gx_le();
    let w = compute_witness(&n().wrapping_sub(&U256::ONE).to_le_bytes().into(), &gx).expect("witness");
    assert_eq!(w.x_r, gx); // (N-1)·G = -G shares x with G
    assert_eq!(w.len_k, 255);
}

/// Verifies the shifted_quotient identity: (pos - neg) is divisible by p,
/// and the result q + 3p is positive and fits in 33 bytes.
/// Uses the double case (2λyA - 3xA²) from k=5's first step as a concrete example.
#[test]
fn shifted_quotient_satisfies_division_identity() {
    use crypto_bigint::{Int, NonZero, Uint};
    use crate::{p, R_BYTES};
    use crate::curve::{recover_y_canonical, replay_double_and_add};

    let gx = U256::from_be_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    let g = crate::curve::AffinePoint { x: gx, y: recover_y_canonical(&gx).unwrap() };
    let (steps, _) = replay_double_and_add(&U256::from(5u32), &g);
    let s = &steps[0]; // first step: double (op=0)
    assert_eq!(s.op, 0);

    let mul512 = |a: &U256, b: &U256| -> U512 { let (lo, hi) = a.widening_mul(b); lo.concat(&hi) };
    // Work in Uint<9> to avoid overflow on 2x/3x sums.
    let pos: Uint<9> = { let t: Uint<9> = mul512(&s.lambda, &s.a.y).resize(); t.wrapping_add(&t) };
    let neg: Uint<9> = { let t: Uint<9> = mul512(&s.a.x, &s.a.x).resize(); t.wrapping_add(&t).wrapping_add(&t) };

    // p as Int<5>: p < 2^256 < 2^320, so positive as signed Int<5>.
    let p_nz: NonZero<Int<5>> = NonZero::new(*p().resize::<5>().as_int()).unwrap();
    // 3p as Uint<5> from R_BYTES.
    let r_3p: Uint<5> = { let mut b = [0u8; 40]; b[..33].copy_from_slice(&R_BYTES); Uint::<5>::from_le_slice(&b) };

    let num: crate::witness::I576 = pos.as_int().wrapping_sub(neg.as_int());
    let (q_opt, r) = num.checked_div_rem(&p_nz);
    assert_eq!(r, Int::<5>::ZERO, "2λyA - 3xA² must be divisible by p");
    let q: crate::witness::I576 = q_opt.unwrap().into();
    // Final result q + 3p must be positive and fit in 33 bytes.
    let result: crate::witness::I576 = q.wrapping_add(r_3p.resize::<9>().as_int());
    let (result_abs, result_is_neg) = result.abs_sign();
    assert!(!bool::from(result_is_neg), "final quotient must be positive");
    let result_bytes = result_abs.to_le_bytes();
    assert!(result_bytes[33..].iter().all(|&b| b == 0), "quotient must fit in 33 bytes");
}

/// Cross-checks the shifted_quotient result for the lambda double case.
///
/// Verifies that `q * p == 3p² + 2λ*yA - 3*xA²` exactly, confirming both
/// divisibility and that the quotient round-trips correctly through U1024 division.
#[test]
fn shifted_quotient_double_matches_identity() {
    use crate::P_BYTES;
    use crate::curve::{recover_y_canonical, replay_double_and_add};

    let gx = U256::from_be_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    let g = crate::curve::AffinePoint { x: gx, y: recover_y_canonical(&gx).unwrap() };
    let (steps, _) = replay_double_and_add(&U256::from(5u32), &g);
    let s = &steps[0]; // first step: double (op=0)
    assert_eq!(s.op, 0);

    let mut p_le128 = [0u8; 128];
    p_le128[..32].copy_from_slice(&P_BYTES);
    let p1024 = NonZero::new(U1024::from_le_slice(&p_le128)).unwrap();

    let mul512 = |a: &U256, b: &U256| -> U512 { let (lo, hi) = a.widening_mul(b); lo.concat(&hi) };
    let widen = |v: U512| -> U1024 { let mut b = [0u8; 128]; b[..64].copy_from_slice(&v.to_le_bytes()); U1024::from_le_slice(&b) };

    let pos = { let t = mul512(&s.lambda, &s.a.y); t.wrapping_add(&t) }; // 2λ*yA
    let neg = { let t = mul512(&s.a.x, &s.a.x); t.wrapping_add(&t).wrapping_add(&t) }; // 3xA²

    let (p_sq_lo, p_sq_hi) = crate::p().widening_mul(&crate::p());
    let p_sq: U1024 = {
        let mut b = [0u8; 128];
        let lo_bytes: [u8; 32] = p_sq_lo.to_le_bytes().into();
        let hi_bytes: [u8; 32] = p_sq_hi.to_le_bytes().into();
        let mut lo64 = [0u8; 64];
        lo64[..32].copy_from_slice(&lo_bytes);
        lo64[32..64].copy_from_slice(&hi_bytes);
        b[..64].copy_from_slice(&lo64);
        U1024::from_le_slice(&b)
    };
    let r_3p_sq = p_sq.wrapping_add(&p_sq).wrapping_add(&p_sq);

    let total = r_3p_sq.wrapping_add(&widen(pos)).wrapping_sub(&widen(neg));
    let (q, r) = total.div_rem(&p1024);
    assert_eq!(r, U1024::ZERO, "3p² + 2λyA - 3xA² must be divisible by p");

    // q * p must equal total exactly.
    let q_bytes = q.to_le_bytes();
    assert!(q_bytes[64..].iter().all(|&b| b == 0), "quotient must fit in U512");
    let q512 = U512::from_le_slice(&q_bytes[..64]);
    let q512_bytes = q512.to_le_bytes();
    assert!(q512_bytes[33..].iter().all(|&b| b == 0), "quotient must fit in 33 bytes");
}

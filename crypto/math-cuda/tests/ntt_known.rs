//! Known-answer NTT test. Random-input tests catch most bugs but can mask
//! systematic errors (sign flips, off-by-one twiddle indices, wrong-direction
//! butterflies) that would cancel under noise. This test picks a polynomial
//! with a known closed-form evaluation at every root of unity and compares
//! the GPU forward NTT to that reference, computed independently from any
//! FFT code path.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsFFTField, IsPrimeField};

type Fp = FieldElement<GoldilocksField>;

fn canon(x: u64) -> u64 {
    GoldilocksField::canonical(&x)
}

/// p(x) = 1 + x. NTT at size N is the vector [p(omega^0), p(omega^1), ...,
/// p(omega^(N-1))] for `omega` a primitive N-th root of unity. We compute
/// the reference by direct exponentiation of `omega` (no FFT involved)
/// so a bug in either the CPU or GPU NTT can't hide here.
#[test]
fn ntt_known_polynomial_x_plus_one_size_256() {
    let log_n: u64 = 8;
    let n: usize = 1 << log_n;

    // GoldilocksField uses bit-reversed coefficient layout for forward NTT:
    // input coeffs at index `i` become `p(omega^bitrev(i))`. Confirm by
    // matching against the existing forward() API which random-input tests
    // already validate against `Polynomial::evaluate_fft`. The known-poly
    // value lets us catch systematic errors in that pipeline that random
    // inputs miss.

    // Input coefficients: [1, 1, 0, 0, ..., 0]. (Natural order, lowest
    // degree first. `Polynomial::new` and `math_cuda::ntt::forward` both
    // expect this convention.)
    let mut input = vec![0u64; n];
    input[0] = 1;
    input[1] = 1;

    let gpu = math_cuda::ntt::forward(&input).expect("gpu ntt");
    assert_eq!(gpu.len(), n);

    // Reference: omega = primitive N-th root of unity in Goldilocks.
    // p(omega^i) = 1 + omega^i.
    let omega = GoldilocksField::get_primitive_root_of_unity(log_n).expect("root of unity");
    let one = Fp::from_raw(1);

    let mut expected = Vec::with_capacity(n);
    let mut omega_i = one; // omega^0
    for _ in 0..n {
        let val = &one + &omega_i;
        expected.push(*val.value());
        omega_i = &omega_i * &omega;
    }

    for (i, (&g_raw, &e_raw)) in gpu.iter().zip(expected.iter()).enumerate() {
        let g = canon(g_raw);
        let e = canon(e_raw);
        if g != e {
            panic!(
                "p(omega^{i}) mismatch: gpu canon {:#018x}, expected canon {:#018x} (omega^{i} computed independently of any FFT)",
                g, e,
            );
        }
    }
}

/// Same idea, smaller: p(x) = 1 + x at size 2^4 = 16. A failure at this
/// size with passes at larger sizes (or vice-versa) would point at a
/// boundary bug between the recursive base case and the 8-level
/// shared-memory fused step in the GPU NTT.
#[test]
fn ntt_known_polynomial_x_plus_one_size_16() {
    let log_n: u64 = 4;
    let n: usize = 1 << log_n;

    let mut input = vec![0u64; n];
    input[0] = 1;
    input[1] = 1;

    let gpu = math_cuda::ntt::forward(&input).expect("gpu ntt");

    let omega = GoldilocksField::get_primitive_root_of_unity(log_n).expect("root");
    let one = Fp::from_raw(1);
    let mut omega_i = one;
    for (i, &g) in gpu.iter().enumerate() {
        let exp = &one + &omega_i;
        assert_eq!(
            canon(g),
            canon(*exp.value()),
            "p(omega^{i}) mismatch at size 16"
        );
        omega_i = &omega_i * &omega;
    }
}

/// p(x) = x^k for k = N/2. p(omega^i) = omega^(k*i). With k = N/2,
/// omega^(k*i) = (-1)^i since omega^(N/2) = -1 in any field with a
/// primitive N-th root of unity. So evaluations alternate +1, -1, +1, -1.
/// This is a strong test of twiddle-index direction and sign.
#[test]
fn ntt_known_polynomial_x_half_alternating() {
    let log_n: u64 = 8;
    let n: usize = 1 << log_n;
    let k = n / 2;

    let mut input = vec![0u64; n];
    input[k] = 1; // p(x) = x^(N/2)

    let gpu = math_cuda::ntt::forward(&input).expect("gpu ntt");

    // Expected: omega^(k*i) for i = 0..N, which is (omega^k)^i = (-1)^i.
    // canonical(+1) = 1; canonical(-1) = p - 1.
    let p_minus_one = 0xFFFF_FFFF_0000_0001u64 - 1;
    for (i, &g) in gpu.iter().enumerate() {
        let exp = if i % 2 == 0 { 1u64 } else { p_minus_one };
        assert_eq!(
            canon(g),
            exp,
            "x^(N/2) NTT alternation mismatch at i={i}: got {:#018x}",
            canon(g)
        );
    }
}

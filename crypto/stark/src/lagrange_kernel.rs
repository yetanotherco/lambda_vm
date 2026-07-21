use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Compute the Lagrange kernel (eq polynomial) for a random point `r`.
///
/// Given r = (r_0, r_1, ..., r_{n-1}), returns a vector s of length N = 2^n where:
///
///   s[i] = eq(bits(i), r) = prod_{l=0}^{n-1} [r_l * b_l(i) + (1 - r_l) * (1 - b_l(i))]
///
/// where b_l(i) = (i >> l) & 1 is the l-th bit of i.
///
/// The Lagrange kernel is the auxiliary column that bridges GKR multilinear extension
/// claims back to committed univariate traces. It satisfies the partition-of-unity
/// property: sum_{i=0}^{N-1} s[i] = 1 for any r.
///
/// Uses the butterfly algorithm in O(N) time and O(N) space.
pub fn compute_lagrange_kernel<E: IsField>(r: &[FieldElement<E>]) -> Vec<FieldElement<E>> {
    let n = r.len();
    let big_n = 1usize << n;
    let one = FieldElement::<E>::one();

    let mut s = vec![one.clone(); big_n];

    #[allow(clippy::needless_range_loop)]
    for j in 0..n {
        let rj = &r[j];
        let one_minus_rj = &one - rj;

        #[cfg(feature = "parallel")]
        {
            if big_n >= 1024 {
                s.par_iter_mut().enumerate().for_each(|(i, s_i)| {
                    if (i >> j) & 1 == 1 {
                        *s_i = &*s_i * rj;
                    } else {
                        *s_i = &*s_i * &one_minus_rj;
                    }
                });
            } else {
                for i in 0..big_n {
                    if (i >> j) & 1 == 1 {
                        s[i] = &s[i] * rj;
                    } else {
                        s[i] = &s[i] * &one_minus_rj;
                    }
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..big_n {
                if (i >> j) & 1 == 1 {
                    s[i] = &s[i] * rj;
                } else {
                    s[i] = &s[i] * &one_minus_rj;
                }
            }
        }
    }

    s
}

/// Evaluate the multilinear extension (MLE) of `values` at point `r`.
///
/// Given a function f defined on {0,1}^n by its truth table `values` (length N = 2^n),
/// its multilinear extension is:
///
///   MLE(r) = sum_{i=0}^{N-1} values[i] * eq(bits(i), r)
///
/// This is the inner product of `values` with the Lagrange kernel at `r`.
///
/// Panics if `values.len()` is not a power of 2 or if `r.len() != log2(values.len())`.
pub fn eval_mle<E: IsField>(values: &[FieldElement<E>], r: &[FieldElement<E>]) -> FieldElement<E> {
    let n = r.len();
    let big_n = 1usize << n;
    assert_eq!(
        values.len(),
        big_n,
        "values length ({}) must equal 2^r.len() = 2^{} = {}",
        values.len(),
        n,
        big_n
    );

    let kernel = compute_lagrange_kernel(r);

    let mut result = FieldElement::<E>::zero();
    for i in 0..big_n {
        result = &result + &(&values[i] * &kernel[i]);
    }

    result
}

/// Evaluate the multilinear extension (MLE) of base-field `values` at an extension-field point `r`.
///
/// Same as `eval_mle` but accepts base-field values and an extension-field evaluation point.
/// Uses `F * E -> E` multiplication directly (no `to_extension()` conversion).
///
/// Panics if `values.len()` is not a power of 2 or if `r.len() != log2(values.len())`.
pub fn eval_mle_base<F, E>(values: &[FieldElement<F>], r: &[FieldElement<E>]) -> FieldElement<E>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = r.len();
    let big_n = 1usize << n;
    assert_eq!(
        values.len(),
        big_n,
        "values length ({}) must equal 2^r.len() = 2^{} = {}",
        values.len(),
        n,
        big_n
    );

    let kernel = compute_lagrange_kernel(r);

    let mut result = FieldElement::<E>::zero();
    for i in 0..big_n {
        // F * E -> E multiplication (no to_extension)
        result = &result + &(&values[i] * &kernel[i]);
    }

    result
}

/// Evaluate the MLE of base-field `values` at an extension-field point, using a pre-computed
/// Lagrange kernel.
///
/// This avoids recomputing `compute_lagrange_kernel(r)` when evaluating multiple columns
/// at the same point `r`.
///
/// Panics if `values.len() != kernel.len()`.
pub fn eval_mle_base_with_kernel<F, E>(
    values: &[FieldElement<F>],
    kernel: &[FieldElement<E>],
) -> FieldElement<E>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
{
    let big_n = values.len();
    assert_eq!(
        big_n,
        kernel.len(),
        "values length ({}) must equal kernel length ({})",
        big_n,
        kernel.len()
    );

    #[cfg(feature = "parallel")]
    {
        if big_n >= 1024 {
            return (0..big_n)
                .into_par_iter()
                .fold(
                    || FieldElement::<E>::zero(),
                    |acc, i| acc + &values[i] * &kernel[i],
                )
                .reduce(|| FieldElement::<E>::zero(), |a, b| a + b);
        }
    }

    // Sequential fallback
    let mut result = FieldElement::<E>::zero();
    for i in 0..big_n {
        result = &result + &(&values[i] * &kernel[i]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField;

    type FE = FieldElement<GoldilocksField>;

    #[test]
    fn test_compute_lagrange_kernel_n2() {
        // n=2, r = (3, 7)
        // s[0] = (1-3)*(1-7) = (-2)*(-6) = 12
        // s[1] = 3*(1-7)    = 3*(-6)    = -18
        // s[2] = (1-3)*7    = (-2)*7    = -14
        // s[3] = 3*7        = 21
        let r = vec![FE::from(3u64), FE::from(7u64)];
        let s = compute_lagrange_kernel(&r);

        assert_eq!(s.len(), 4);

        // In Goldilocks (p = 2^64 - 2^32 + 1), negative values wrap around.
        // 12 mod p = 12
        assert_eq!(s[0], FE::from(12u64));
        // -18 mod p = p - 18
        let neg_18 = FE::zero() - FE::from(18u64);
        assert_eq!(s[1], neg_18);
        // -14 mod p = p - 14
        let neg_14 = FE::zero() - FE::from(14u64);
        assert_eq!(s[2], neg_14);
        // 21
        assert_eq!(s[3], FE::from(21u64));

        // Verify partition of unity: sum = 12 + (-18) + (-14) + 21 = 1
        let sum = ((s[0] + s[1]) + s[2]) + s[3];
        assert_eq!(sum, FE::one());
    }

    #[test]
    fn test_lagrange_kernel_partition_of_unity() {
        // For any random r, the sum of kernel values must equal 1.
        // Test with several different r vectors.
        for &(n, r_vals) in &[
            (1usize, &[42u64][..]),
            (2, &[100, 200][..]),
            (3, &[5, 17, 99][..]),
            (4, &[1, 2, 3, 4][..]),
        ] {
            let r: Vec<FE> = r_vals.iter().map(|&v| FE::from(v)).collect();
            let s = compute_lagrange_kernel(&r);
            assert_eq!(s.len(), 1 << n);

            let mut sum = FE::zero();
            for val in &s {
                sum = sum + val;
            }
            assert_eq!(
                sum,
                FE::one(),
                "Partition of unity failed for n={}, r={:?}",
                n,
                r_vals
            );
        }
    }

    #[test]
    fn test_eval_mle_linear() {
        // f(x1) = 3*x1 + 5 on {0, 1}
        // f(0) = 5, f(1) = 8
        // MLE at r: 5*(1-r) + 8*r = 5 + 3r
        let values = vec![FE::from(5u64), FE::from(8u64)];

        // Test at r = 0: should be 5
        let result = eval_mle(&values, &[FE::from(0u64)]);
        assert_eq!(result, FE::from(5u64));

        // Test at r = 1: should be 8
        let result = eval_mle(&values, &[FE::from(1u64)]);
        assert_eq!(result, FE::from(8u64));

        // Test at r = 10: should be 5 + 3*10 = 35
        let result = eval_mle(&values, &[FE::from(10u64)]);
        assert_eq!(result, FE::from(35u64));

        // Test at r = 100: should be 5 + 3*100 = 305
        let result = eval_mle(&values, &[FE::from(100u64)]);
        assert_eq!(result, FE::from(305u64));
    }

    #[test]
    fn test_eval_mle_2var() {
        // f(x1, x2) = 2*x1*x2 + 3*x1 + x2 + 7
        // Evaluate on {0,1}^2 (bit ordering: x1 = bit 0, x2 = bit 1):
        //   i=0 (x1=0, x2=0): 0 + 0 + 0 + 7 = 7
        //   i=1 (x1=1, x2=0): 0 + 3 + 0 + 7 = 10
        //   i=2 (x1=0, x2=1): 0 + 0 + 1 + 7 = 8
        //   i=3 (x1=1, x2=1): 2 + 3 + 1 + 7 = 13
        let values = vec![
            FE::from(7u64),
            FE::from(10u64),
            FE::from(8u64),
            FE::from(13u64),
        ];

        // Verify at Boolean points first (should recover original values)
        assert_eq!(
            eval_mle(&values, &[FE::from(0u64), FE::from(0u64)]),
            FE::from(7u64)
        );
        assert_eq!(
            eval_mle(&values, &[FE::from(1u64), FE::from(0u64)]),
            FE::from(10u64)
        );
        assert_eq!(
            eval_mle(&values, &[FE::from(0u64), FE::from(1u64)]),
            FE::from(8u64)
        );
        assert_eq!(
            eval_mle(&values, &[FE::from(1u64), FE::from(1u64)]),
            FE::from(13u64)
        );

        // Evaluate at (x1=5, x2=3):
        // f(5, 3) = 2*5*3 + 3*5 + 3 + 7 = 30 + 15 + 3 + 7 = 55
        let result = eval_mle(&values, &[FE::from(5u64), FE::from(3u64)]);
        assert_eq!(result, FE::from(55u64));

        // Evaluate at (x1=2, x2=4):
        // f(2, 4) = 2*2*4 + 3*2 + 4 + 7 = 16 + 6 + 4 + 7 = 33
        let result = eval_mle(&values, &[FE::from(2u64), FE::from(4u64)]);
        assert_eq!(result, FE::from(33u64));
    }
}

use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// In-Place Radix-2 NR DIT FFT algorithm over a slice of two-adic field elements.
/// It's required that the twiddle factors are in bit-reverse order. Else this function will not
/// return fourier transformed values.
/// Also the input size needs to be a power of two.
/// It's recommended to use the current safe abstractions instead of this function.
///
/// Performs a fast fourier transform with the next attributes:
/// - In-Place: an auxiliary vector of data isn't needed for the algorithm.
/// - Radix-2: the algorithm halves the problem size log(n) times.
/// - NR: natural to reverse order, meaning that the input is naturally ordered and the output will
///   be bit-reversed ordered.
/// - DIT: decimation in time
///
/// It supports values in a field E and domain in a subfield F.
pub fn in_place_nr_2radix_fft<F, E>(input: &mut [FieldElement<E>], twiddles: &[FieldElement<F>])
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    // divide input in groups, starting with 1, duplicating the number of groups in each stage.
    let mut group_count = 1;
    let mut group_size = input.len();

    // for each group, there'll be group_size / 2 butterflies.
    // a butterfly is the atomic operation of a FFT, e.g: (a, b) = (a + wb, a - wb).
    // The 0.5 factor is what gives FFT its performance, it recursively halves the problem size
    // (group size).

    while group_count < input.len() {
        #[allow(clippy::needless_range_loop)] // the suggestion would obfuscate a bit the algorithm
        for group in 0..group_count {
            let first_in_group = group * group_size;
            let first_in_next_group = first_in_group + group_size / 2;

            let w = &twiddles[group]; // a twiddle factor is used per group

            for i in first_in_group..first_in_next_group {
                let wi = w * &input[i + group_size / 2];

                let y0 = &input[i] + &wi;
                let y1 = &input[i] - &wi;

                input[i] = y0;
                input[i + group_size / 2] = y1;
            }
        }
        group_count *= 2;
        group_size /= 2;
    }
}

/// In-Place Radix-2 RN DIT FFT algorithm over a slice of two-adic field elements.
/// It's required that the twiddle factors are naturally ordered (so w[i] = w^i). Else this
/// function will not return fourier transformed values.
/// Also the input size needs to be a power of two.
/// It's recommended to use the current safe abstractions instead of this function.
///
/// Performs a fast fourier transform with the next attributes:
/// - In-Place: an auxiliary vector of data isn't needed for storing the results.
/// - Radix-2: the algorithm halves the problem size log(n) times.
/// - RN: reverse to natural order, meaning that the input is bit-reversed ordered and the output will
///   be naturally ordered.
/// - DIT: decimation in time
///
/// It supports values in a field E and domain in a subfield F.
#[allow(dead_code)]
pub fn in_place_rn_2radix_fft<F>(input: &mut [FieldElement<F>], twiddles: &[FieldElement<F>])
where
    F: IsFFTField,
{
    // divide input in groups, starting with 1, duplicating the number of groups in each stage.
    let mut group_count = 1;
    let mut group_size = input.len();

    // for each group, there'll be group_size / 2 butterflies.
    // a butterfly is the atomic operation of a FFT, e.g: (a, b) = (a + wb, a - wb).
    // The 0.5 factor is what gives FFT its performance, it recursively halves the problem size
    // (group size).

    while group_count < input.len() {
        let step_to_next = 2 * group_count; // next butterfly in the group
        let step_to_last = step_to_next * (group_size / 2 - 1);

        for group in 0..group_count {
            let w = &twiddles[group * group_size / 2];

            for i in (group..=group + step_to_last).step_by(step_to_next) {
                let wi = w * &input[i + group_count];

                let y0 = &input[i] + &wi;
                let y1 = &input[i] - &wi;

                input[i] = y0;
                input[i + group_count] = y1;
            }
        }
        group_count *= 2;
        group_size /= 2;
    }
}

/// In-Place Radix-4 NR DIT FFT algorithm over a slice of two-adic field elements.
/// It's required that the twiddle factors are in bit-reverse order. Else this function will not
/// return fourier transformed values.
/// Also the input size needs to be a power of two.
/// It's recommended to use the current safe abstractions instead of this function.
///
/// Performs a fast fourier transform with the next attributes:
/// - In-Place: an auxiliary vector of data isn't needed for the algorithm.
/// - Radix-4: the algorithm halves the problem size log(n) times.
/// - NR: natural to reverse order, meaning that the input is naturally ordered and the output will
///   be bit-reversed ordered.
/// - DIT: decimation in time
pub fn in_place_nr_4radix_fft<F, E>(input: &mut [FieldElement<E>], twiddles: &[FieldElement<F>])
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    debug_assert!(input.len().is_power_of_two());
    debug_assert!(input.len().ilog2().is_multiple_of(2)); // Even power of 2 => x is power of 4

    // divide input in groups, starting with 1, duplicating the number of groups in each stage.
    let mut group_count = 1;
    let mut group_size = input.len();

    // for each group, there'll be group_size / 4 butterflies.
    // a butterfly is the atomic operation of a FFT, e.g:
    // x' = x + yw2 + zw1 + tw1w2
    // y' = x - yw2 + zw1 - tw1w2
    // z' = x + yw3 - zw1 - tw1w3
    // t' = x - yw3 - zw1 + tw1w3
    // The 0.25 factor is what gives FFT its performance, it recursively divides the problem size
    // by 4 (group size).

    while group_count < input.len() {
        #[allow(clippy::needless_range_loop)] // the suggestion would obfuscate a bit the algorithm
        for group in 0..group_count {
            let first_in_group = group * group_size;
            let first_in_next_group = first_in_group + group_size / 4;

            let (w1, w2, w3) = (
                &twiddles[group],
                &twiddles[2 * group],
                &twiddles[2 * group + 1],
            );

            for i in first_in_group..first_in_next_group {
                let (j, k, l) = (
                    i + group_size / 4,
                    i + group_size / 2,
                    i + 3 * group_size / 4,
                );

                let zw1 = w1 * &input[k];
                let tw1 = w1 * &input[l];
                let a = w2 * (&input[j] + &tw1);
                let b = w3 * (&input[j] - &tw1);

                let x = &input[i] + &zw1 + &a;
                let y = &input[i] + &zw1 - &a;
                let z = &input[i] - &zw1 + &b;
                let t = &input[i] - &zw1 - &b;

                input[i] = x;
                input[j] = y;
                input[k] = z;
                input[l] = t;
            }
        }
        group_count *= 4;
        group_size /= 4;
    }
}

/// Minimum number of elements required for parallel FFT to be beneficial.
/// Below this threshold, sequential FFT is faster due to parallelization overhead.
/// Benchmarks show parallelization only provides speedup for sizes >= 2^18 (262144).
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 1 << 18; // 262144 elements

/// Parallel In-Place Radix-2 NR DIT FFT algorithm.
/// Uses rayon for parallel processing when input size exceeds threshold.
/// Falls back to sequential for smaller inputs.
#[cfg(feature = "parallel")]
pub fn in_place_nr_2radix_fft_parallel<F, E>(
    input: &mut [FieldElement<E>],
    twiddles: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Sync + Send,
    FieldElement<E>: Send + Sync,
    FieldElement<F>: Sync,
{
    let n = input.len();

    // For small inputs, use sequential algorithm
    if n < PARALLEL_THRESHOLD {
        in_place_nr_2radix_fft(input, twiddles);
        return;
    }

    let mut group_count = 1;
    let mut group_size = n;

    while group_count < n {
        let half_group = group_size / 2;

        // Only parallelize when we have enough groups to benefit from parallelization
        // Early stages: few groups, many butterflies -> parallelize butterflies within groups
        // Late stages: many groups, few butterflies -> parallelize across groups
        if group_count >= 8 {
            // Parallelize across groups
            input
                .par_chunks_mut(group_size)
                .enumerate()
                .for_each(|(group, chunk)| {
                    let w = &twiddles[group];
                    for i in 0..half_group {
                        let wi = w * &chunk[i + half_group];
                        let y0 = &chunk[i] + &wi;
                        let y1 = &chunk[i] - &wi;
                        chunk[i] = y0;
                        chunk[i + half_group] = y1;
                    }
                });
        } else {
            // For early stages with few groups, parallelize butterflies within each group
            for group in 0..group_count {
                let first_in_group = group * group_size;
                let w = &twiddles[group];

                // Process butterflies in parallel within this group
                let (left, right) =
                    input[first_in_group..first_in_group + group_size].split_at_mut(half_group);

                left.par_iter_mut()
                    .zip(right.par_iter_mut())
                    .for_each(|(a, b)| {
                        let wi = w * &*b;
                        let y0 = &*a + &wi;
                        let y1 = &*a - &wi;
                        *a = y0;
                        *b = y1;
                    });
            }
        }

        group_count *= 2;
        group_size /= 2;
    }
}

/// Parallel In-Place Radix-4 NR DIT FFT algorithm.
/// Uses rayon for parallel processing when input size exceeds threshold.
#[cfg(feature = "parallel")]
pub fn in_place_nr_4radix_fft_parallel<F, E>(
    input: &mut [FieldElement<E>],
    twiddles: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Sync + Send,
    FieldElement<E>: Send + Sync,
    FieldElement<F>: Sync,
{
    debug_assert!(input.len().is_power_of_two());
    debug_assert!(input.len().ilog2().is_multiple_of(2));

    let n = input.len();

    // For small inputs, use sequential algorithm
    if n < PARALLEL_THRESHOLD {
        in_place_nr_4radix_fft(input, twiddles);
        return;
    }

    let mut group_count = 1;
    let mut group_size = n;

    while group_count < n {
        let quarter_group = group_size / 4;

        if group_count >= 4 {
            // Parallelize across groups
            input
                .par_chunks_mut(group_size)
                .enumerate()
                .for_each(|(group, chunk)| {
                    let (w1, w2, w3) = (
                        &twiddles[group],
                        &twiddles[2 * group],
                        &twiddles[2 * group + 1],
                    );

                    for i in 0..quarter_group {
                        let (j, k, l) = (
                            i + quarter_group,
                            i + 2 * quarter_group,
                            i + 3 * quarter_group,
                        );

                        let zw1 = w1 * &chunk[k];
                        let tw1 = w1 * &chunk[l];
                        let a = w2 * (&chunk[j] + &tw1);
                        let b = w3 * (&chunk[j] - &tw1);

                        let x = &chunk[i] + &zw1 + &a;
                        let y = &chunk[i] + &zw1 - &a;
                        let z = &chunk[i] - &zw1 + &b;
                        let t = &chunk[i] - &zw1 - &b;

                        chunk[i] = x;
                        chunk[j] = y;
                        chunk[k] = z;
                        chunk[l] = t;
                    }
                });
        } else {
            // Sequential for early stages
            for group in 0..group_count {
                let first_in_group = group * group_size;
                let (w1, w2, w3) = (
                    &twiddles[group],
                    &twiddles[2 * group],
                    &twiddles[2 * group + 1],
                );

                for i in 0..quarter_group {
                    let idx = first_in_group + i;
                    let (j, k, l) = (
                        idx + quarter_group,
                        idx + 2 * quarter_group,
                        idx + 3 * quarter_group,
                    );

                    let zw1 = w1 * &input[k];
                    let tw1 = w1 * &input[l];
                    let a = w2 * (&input[j] + &tw1);
                    let b = w3 * (&input[j] - &tw1);

                    let x = &input[idx] + &zw1 + &a;
                    let y = &input[idx] + &zw1 - &a;
                    let z = &input[idx] - &zw1 + &b;
                    let t = &input[idx] - &zw1 - &b;

                    input[idx] = x;
                    input[j] = y;
                    input[k] = z;
                    input[l] = t;
                }
            }
        }

        group_count *= 4;
        group_size /= 4;
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use crate::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
    use crate::fft::cpu::roots_of_unity::get_twiddles;
    use crate::fft::test_helpers::naive_matrix_dft_test;
    use crate::field::{test_fields::u64_test_field::U64TestField, traits::RootsConfig};
    use proptest::{collection, prelude::*};

    use super::*;

    type F = U64TestField;
    type FE = FieldElement<F>;

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
        // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
        // also it can't exceed the test field's two-adicity.
    }
    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(vec in (1..max_exp).prop_flat_map(|i| collection::vec(field_element(), 1 << i))) -> alloc::vec::Vec<FE> {
            vec
        }
    }
    prop_compose! {
        fn field_vec_r4(max_exp: u8)(vec in (1..max_exp).prop_flat_map(|i| collection::vec(field_element(), 1 << (2 * i)))) -> alloc::vec::Vec<FE> {
            vec
        }
    }

    proptest! {
        // Property-based test that ensures NR Radix-2 FFT gives the same result as a naive DFT.
        #[test]
        fn test_nr_2radix_fft_matches_naive_eval(coeffs in field_vec(8)) {
            let expected = naive_matrix_dft_test(&coeffs);

            let order = coeffs.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).unwrap();

            let mut result = coeffs;
            in_place_nr_2radix_fft::<F, F>(&mut result, &twiddles);
            in_place_bit_reverse_permute(&mut result);

            prop_assert_eq!(expected, result);
        }

        // Property-based test that ensures RN Radix-2 FFT gives the same result as a naive DFT.
        #[test]
        fn test_rn_2radix_fft_matches_naive_eval(coeffs in field_vec(8)) {
            let expected = naive_matrix_dft_test(&coeffs);

            let order = coeffs.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::Natural).unwrap();

            let mut result = coeffs;
            in_place_bit_reverse_permute(&mut result);
            in_place_rn_2radix_fft(&mut result, &twiddles);

            prop_assert_eq!(result, expected);
        }

        // Property-based test that ensures NR Radix-2 FFT gives the same result as a naive DFT.
        #[test]
        fn test_nr_4radix_fft_matches_naive_eval(coeffs in field_vec_r4(5)) {
            let expected = naive_matrix_dft_test(&coeffs);

            let order = coeffs.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).unwrap();

            let mut result = coeffs;
            in_place_nr_4radix_fft::<F, F>(&mut result, &twiddles);
            in_place_bit_reverse_permute(&mut result);

            prop_assert_eq!(expected, result);
        }
    }

    #[cfg(feature = "parallel")]
    mod parallel_tests {
        use super::*;

        // Test parallel FFT matches sequential for small inputs
        #[test]
        fn test_parallel_2radix_fft_small_input() {
            let coeffs: Vec<FE> = (1..=16u64).map(FE::from).collect();
            let order = coeffs.len().trailing_zeros();
            let twiddles = get_twiddles::<F>(order.into(), RootsConfig::BitReverse).unwrap();

            let mut sequential = coeffs.clone();
            in_place_nr_2radix_fft::<F, F>(&mut sequential, &twiddles);
            in_place_bit_reverse_permute(&mut sequential);

            let mut parallel = coeffs;
            in_place_nr_2radix_fft_parallel::<F, F>(&mut parallel, &twiddles);
            in_place_bit_reverse_permute(&mut parallel);

            assert_eq!(sequential, parallel);
        }

        // Test parallel FFT matches sequential for large inputs (above threshold)
        #[test]
        fn test_parallel_2radix_fft_large_input() {
            let size = 1 << 13; // 8192 elements, above PARALLEL_THRESHOLD
            let coeffs: Vec<FE> = (1..=size as u64).map(FE::from).collect();
            let order = coeffs.len().trailing_zeros();
            let twiddles = get_twiddles::<F>(order.into(), RootsConfig::BitReverse).unwrap();

            let mut sequential = coeffs.clone();
            in_place_nr_2radix_fft::<F, F>(&mut sequential, &twiddles);
            in_place_bit_reverse_permute(&mut sequential);

            let mut parallel = coeffs;
            in_place_nr_2radix_fft_parallel::<F, F>(&mut parallel, &twiddles);
            in_place_bit_reverse_permute(&mut parallel);

            assert_eq!(sequential, parallel);
        }

        // Test parallel Radix-4 FFT matches sequential
        #[test]
        fn test_parallel_4radix_fft_large_input() {
            let size = 1 << 12; // 4096 elements (power of 4)
            let coeffs: Vec<FE> = (1..=size as u64).map(FE::from).collect();
            let order = coeffs.len().trailing_zeros();
            let twiddles = get_twiddles::<F>(order.into(), RootsConfig::BitReverse).unwrap();

            let mut sequential = coeffs.clone();
            in_place_nr_4radix_fft::<F, F>(&mut sequential, &twiddles);
            in_place_bit_reverse_permute(&mut sequential);

            let mut parallel = coeffs;
            in_place_nr_4radix_fft_parallel::<F, F>(&mut parallel, &twiddles);
            in_place_bit_reverse_permute(&mut parallel);

            assert_eq!(sequential, parallel);
        }
    }
}

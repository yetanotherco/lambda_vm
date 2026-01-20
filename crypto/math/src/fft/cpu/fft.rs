use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Minimum number of groups to trigger parallel execution.
/// Below this threshold, sequential execution is faster due to parallelization overhead.
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 8;

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
        #[cfg(feature = "parallel")]
        {
            if group_count >= PARALLEL_THRESHOLD {
                // Parallel execution: each group works on disjoint memory regions
                input
                    .par_chunks_mut(group_size)
                    .zip(twiddles.par_iter().take(group_count))
                    .for_each(|(chunk, w)| {
                        let half = group_size / 2;
                        for i in 0..half {
                            let wi = w * &chunk[i + half];
                            let y0 = &chunk[i] + &wi;
                            let y1 = &chunk[i] - &wi;
                            chunk[i] = y0;
                            chunk[i + half] = y1;
                        }
                    });
            } else {
                // Sequential execution for small group counts
                fft_stage_sequential(input, twiddles, group_count, group_size);
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            fft_stage_sequential(input, twiddles, group_count, group_size);
        }

        group_count *= 2;
        group_size /= 2;
    }
}

/// Sequential FFT stage helper function
#[inline]
fn fft_stage_sequential<F, E>(
    input: &mut [FieldElement<E>],
    twiddles: &[FieldElement<F>],
    group_count: usize,
    group_size: usize,
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    #[allow(clippy::needless_range_loop)]
    for group in 0..group_count {
        let first_in_group = group * group_size;
        let first_in_next_group = first_in_group + group_size / 2;

        let w = &twiddles[group];

        for i in first_in_group..first_in_next_group {
            let wi = w * &input[i + group_size / 2];

            let y0 = &input[i] + &wi;
            let y1 = &input[i] - &wi;

            input[i] = y0;
            input[i + group_size / 2] = y1;
        }
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
        #[cfg(feature = "parallel")]
        {
            if group_count >= PARALLEL_THRESHOLD {
                // Parallel execution: each group works on disjoint memory regions
                input
                    .par_chunks_mut(group_size)
                    .enumerate()
                    .for_each(|(group, chunk)| {
                        let (w1, w2, w3) = (
                            &twiddles[group],
                            &twiddles[2 * group],
                            &twiddles[2 * group + 1],
                        );
                        let quarter = group_size / 4;
                        for i in 0..quarter {
                            let (j, k, l) = (i + quarter, i + 2 * quarter, i + 3 * quarter);

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
                // Sequential execution for small group counts
                fft4_stage_sequential(input, twiddles, group_count, group_size);
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            fft4_stage_sequential(input, twiddles, group_count, group_size);
        }

        group_count *= 4;
        group_size /= 4;
    }
}

/// Sequential Radix-4 FFT stage helper function
#[inline]
fn fft4_stage_sequential<F, E>(
    input: &mut [FieldElement<E>],
    twiddles: &[FieldElement<F>],
    group_count: usize,
    group_size: usize,
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    #[allow(clippy::needless_range_loop)]
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

    fn in_place_nr_2radix_fft_sequential<F, E>(
        input: &mut [FieldElement<E>],
        twiddles: &[FieldElement<F>],
    ) where
        F: IsFFTField + IsSubFieldOf<E>,
        E: IsField,
    {
        let mut group_count = 1;
        let mut group_size = input.len();

        while group_count < input.len() {
            super::fft_stage_sequential(input, twiddles, group_count, group_size);
            group_count *= 2;
            group_size /= 2;
        }
    }

    fn in_place_nr_4radix_fft_sequential<F, E>(
        input: &mut [FieldElement<E>],
        twiddles: &[FieldElement<F>],
    ) where
        F: IsFFTField + IsSubFieldOf<E>,
        E: IsField,
    {
        let mut group_count = 1;
        let mut group_size = input.len();

        while group_count < input.len() {
            super::fft4_stage_sequential(input, twiddles, group_count, group_size);
            group_count *= 4;
            group_size /= 4;
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

        // Property-based test that ensures Parallel NR Radix-2 FFT gives the same result as Sequential.
        #[test]
        fn test_nr_2radix_parallel_matches_sequential(coeffs in field_vec(15)) {
            let order = coeffs.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).unwrap();

            let mut parallel_result = coeffs.clone();
            let mut sequential_result = coeffs;

            in_place_nr_2radix_fft::<F, F>(&mut parallel_result, &twiddles);
            in_place_nr_2radix_fft_sequential::<F, F>(&mut sequential_result, &twiddles);

            prop_assert_eq!(parallel_result, sequential_result);
        }

        // Property-based test that ensures Parallel NR Radix-4 FFT gives the same result as Sequential.
        #[test]
        fn test_nr_4radix_parallel_matches_sequential(coeffs in field_vec_r4(8)) {
            let order = coeffs.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).unwrap();

            let mut parallel_result = coeffs.clone();
            let mut sequential_result = coeffs;

            in_place_nr_4radix_fft::<F, F>(&mut parallel_result, &twiddles);
            in_place_nr_4radix_fft_sequential::<F, F>(&mut sequential_result, &twiddles);

            prop_assert_eq!(parallel_result, sequential_result);
        }
    }
}

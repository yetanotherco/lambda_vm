#![no_main]
use libfuzzer_sys::fuzz_target;

use math::fft::cpu::roots_of_unity::get_twiddles;
use math::fft::cpu::fft::{in_place_nr_2radix_fft, in_place_nr_4radix_fft};
use math::field::{
    element::FieldElement,
    test_fields::u64_test_field::U64TestField,
    traits::{IsFFTField, IsField, IsSubFieldOf, RootsConfig},
};

type F = U64TestField;
type FE = FieldElement<F>;

// Copied sequential helpers since they are private in the library
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

fn in_place_nr_2radix_fft_sequential<F, E>(input: &mut [FieldElement<E>], twiddles: &[FieldElement<F>])
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let mut group_count = 1;
    let mut group_size = input.len();

    while group_count < input.len() {
        fft_stage_sequential(input, twiddles, group_count, group_size);
        group_count *= 2;
        group_size /= 2;
    }
}

// Fuzzer Target
fuzz_target!(|data: Vec<u64>| {
    // 1. Ensure input size is a power of 2 and large enough to be interesting, but not too huge for fuzzing speed
    if data.is_empty() {
        return;
    }
    
    // Resize to next power of 2
    let len = data.len().next_power_of_two();
    // We limit max size to avoid timeouts
    if len > 8192 {
        return;
    }

    // Pad with zeros or truncate (though next_power_of_two >= len so we pad)
    let mut coeffs: Vec<FE> = data.iter().map(|&x| FE::from(x)).collect();
    while coeffs.len() < len {
        coeffs.push(FE::zero());
    }

    // 2. Setup Twiddles
    let order = coeffs.len().trailing_zeros();
    let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).unwrap();

    // 3. Run Parallel (Library Implementation)
    let mut parallel_result = coeffs.clone();
    in_place_nr_2radix_fft::<F, F>(&mut parallel_result, &twiddles);

    // 4. Run Sequential (Local Helper)
    let mut sequential_result = coeffs;
    in_place_nr_2radix_fft_sequential::<F, F>(&mut sequential_result, &twiddles);

    // 5. Assert Equality
    assert_eq!(parallel_result, sequential_result, "Parallel and Sequential FFT results differ!");
});
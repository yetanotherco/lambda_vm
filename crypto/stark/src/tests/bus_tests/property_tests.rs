//! Property-based tests for LogUp lookup implementation.
//!
//! These tests use proptest to verify correctness across random inputs.

use proptest::prelude::*;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};

use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::lookup::{Packing, SHIFT_8, SHIFT_16};
use crate::proof::options::ProofOptions;
use crate::prover::{IsStarkProver, Prover};
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;
type FE = FieldElement<F>;

/// Generate random field elements for Babybear
fn random_field_element() -> impl Strategy<Value = FE> {
    // Babybear prime is 2^31 - 2^27 + 1, so use u32 values
    (0u32..2147483647u32).prop_map(|v| FE::from(v as u64))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Packing::Direct should return the input unchanged
    #[test]
    fn prop_packing_direct_identity(value in random_field_element()) {
        let columns = vec![value.clone()];
        let result = Packing::Direct.combine(&columns);
        prop_assert_eq!(result.len(), 1);
        prop_assert_eq!(&result[0], &value);
    }

    /// Packing::Word2L should combine two halves correctly: h0 + 2^16 * h1
    #[test]
    fn prop_packing_word2l_formula(
        h0 in 0u32..65536u32,
        h1 in 0u32..65536u32,
    ) {
        let columns = vec![FE::from(h0 as u64), FE::from(h1 as u64)];
        let result = Packing::Word2L.combine(&columns);

        let expected = FE::from(h0 as u64) + FE::from(h1 as u64) * FE::from(SHIFT_16);
        prop_assert_eq!(result.len(), 1);
        prop_assert_eq!(&result[0], &expected);
    }

    /// Packing::Word4L should combine four bytes correctly: b0 + 2^8*b1 + 2^16*b2 + 2^24*b3
    #[test]
    fn prop_packing_word4l_formula(
        b0 in 0u32..256u32,
        b1 in 0u32..256u32,
        b2 in 0u32..256u32,
        b3 in 0u32..256u32,
    ) {
        let columns = vec![
            FE::from(b0 as u64),
            FE::from(b1 as u64),
            FE::from(b2 as u64),
            FE::from(b3 as u64),
        ];
        let result = Packing::Word4L.combine(&columns);

        let shift_8 = FE::from(SHIFT_8);
        let shift_16 = FE::from(SHIFT_16);
        let shift_24 = &shift_8 * &shift_16;

        let expected = FE::from(b0 as u64)
            + FE::from(b1 as u64) * &shift_8
            + FE::from(b2 as u64) * &shift_16
            + FE::from(b3 as u64) * &shift_24;

        prop_assert_eq!(result.len(), 1);
        prop_assert_eq!(&result[0], &expected);
    }

    /// DWordHL (2x Word2L) should equal combining Word2L twice
    #[test]
    fn prop_packing_dwordhl_equals_two_word2l(
        h0 in 0u32..65536u32,
        h1 in 0u32..65536u32,
        h2 in 0u32..65536u32,
        h3 in 0u32..65536u32,
    ) {
        let columns = vec![
            FE::from(h0 as u64),
            FE::from(h1 as u64),
            FE::from(h2 as u64),
            FE::from(h3 as u64),
        ];
        let result = Packing::DWordHL.combine(&columns);

        let word0 = Packing::Word2L.combine(&columns[0..2]);
        let word1 = Packing::Word2L.combine(&columns[2..4]);

        prop_assert_eq!(result.len(), 2);
        prop_assert_eq!(&result[0], &word0[0]);
        prop_assert_eq!(&result[1], &word1[0]);
    }

    /// DWordBL (2x Word4L) should equal combining Word4L twice
    #[test]
    fn prop_packing_dwordbl_equals_two_word4l(
        bytes in prop::collection::vec(0u32..256u32, 8),
    ) {
        let columns: Vec<FE> = bytes.iter().map(|&b| FE::from(b as u64)).collect();
        let result = Packing::DWordBL.combine(&columns);

        let word0 = Packing::Word4L.combine(&columns[0..4]);
        let word1 = Packing::Word4L.combine(&columns[4..8]);

        prop_assert_eq!(result.len(), 2);
        prop_assert_eq!(&result[0], &word0[0]);
        prop_assert_eq!(&result[1], &word1[0]);
    }

    /// QuadHL (4x Word2L) should equal combining Word2L four times
    #[test]
    fn prop_packing_quadhl_equals_four_word2l(
        halves in prop::collection::vec(0u32..65536u32, 8),
    ) {
        let columns: Vec<FE> = halves.iter().map(|&h| FE::from(h as u64)).collect();
        let result = Packing::QuadHL.combine(&columns);

        let word0 = Packing::Word2L.combine(&columns[0..2]);
        let word1 = Packing::Word2L.combine(&columns[2..4]);
        let word2 = Packing::Word2L.combine(&columns[4..6]);
        let word3 = Packing::Word2L.combine(&columns[6..8]);

        prop_assert_eq!(result.len(), 4);
        prop_assert_eq!(&result[0], &word0[0]);
        prop_assert_eq!(&result[1], &word1[0]);
        prop_assert_eq!(&result[2], &word2[0]);
        prop_assert_eq!(&result[3], &word3[0]);
    }

    /// num_columns and num_bus_elements should be consistent
    #[test]
    fn prop_packing_column_counts_consistent(packing_idx in 0usize..10) {
        let packings = [
            Packing::Direct,
            Packing::Word2L,
            Packing::Word4L,
            Packing::DWordWL,
            Packing::DWordHHW,
            Packing::DWordWHH,
            Packing::DWordHL,
            Packing::DWordBL,
            Packing::QuadHL,
        ];

        if packing_idx < packings.len() {
            let packing = &packings[packing_idx];
            let num_cols = packing.num_columns();
            let num_bus = packing.num_bus_elements();

            // Create dummy columns
            let columns: Vec<FE> = (0..num_cols).map(|i| FE::from(i as u64)).collect();
            let result = packing.combine(&columns);

            prop_assert_eq!(result.len(), num_bus, "Packing {:?} bus elements mismatch", packing);
        }
    }

    /// Batch inversion should equal individual inversions
    #[test]
    fn prop_batch_inverse_equals_individual(
        values in prop::collection::vec(1u64..1000000u64, 1..50),
    ) {
        let elements: Vec<FE> = values.iter().map(|&v| FE::from(v)).collect();

        // Individual inversions
        let individual: Vec<FE> = elements.iter().map(|e| e.inv().unwrap()).collect();

        // Batch inversion
        let mut batch = elements.clone();
        FieldElement::inplace_batch_inverse(&mut batch).unwrap();

        for (i, (ind, bat)) in individual.iter().zip(batch.iter()).enumerate() {
            prop_assert_eq!(ind, bat, "Mismatch at index {}", i);
        }
    }

    /// Verify that x * x^(-1) = 1 for batch inversions
    #[test]
    fn prop_batch_inverse_product_is_one(
        values in prop::collection::vec(1u64..1000000u64, 1..50),
    ) {
        let elements: Vec<FE> = values.iter().map(|&v| FE::from(v)).collect();

        let mut inverses = elements.clone();
        FieldElement::inplace_batch_inverse(&mut inverses).unwrap();

        for (orig, inv) in elements.iter().zip(inverses.iter()) {
            let product = orig * inv;
            prop_assert_eq!(product, FE::one(), "x * x^(-1) should equal 1");
        }
    }
}

/// Test that valid multi-table proofs with random operation values verify correctly
#[test_log::test]
fn test_random_valid_operations_verify() {
    use rand::Rng;

    let mut rng = rand::thread_rng();

    // Generate random operations
    let num_adds = 4;
    let num_muls = 4;

    let mut add_ops: Vec<(u32, u32)> = (0..num_adds)
        .map(|_| (rng.gen_range(1..1000), rng.gen_range(1..1000)))
        .collect();
    let mut mul_ops: Vec<(u32, u32)> = (0..num_muls)
        .map(|_| (rng.gen_range(1..100), rng.gen_range(1..100)))
        .collect();

    // Build CPU trace (interleaved adds and muls)
    let mut add_flags = Vec::new();
    let mut mul_flags = Vec::new();
    let mut a_vals = Vec::new();
    let mut b_vals = Vec::new();
    let mut c_vals = Vec::new();

    for i in 0..8 {
        if i % 2 == 0 && !add_ops.is_empty() {
            let (a, b) = add_ops.pop().unwrap();
            add_flags.push(FE::one());
            mul_flags.push(FE::zero());
            a_vals.push(FE::from(a as u64));
            b_vals.push(FE::from(b as u64));
            c_vals.push(FE::from((a + b) as u64));
        } else if !mul_ops.is_empty() {
            let (a, b) = mul_ops.pop().unwrap();
            add_flags.push(FE::zero());
            mul_flags.push(FE::one());
            a_vals.push(FE::from(a as u64));
            b_vals.push(FE::from(b as u64));
            c_vals.push(FE::from((a * b) as u64));
        } else {
            // Padding
            add_flags.push(FE::zero());
            mul_flags.push(FE::zero());
            a_vals.push(FE::zero());
            b_vals.push(FE::zero());
            c_vals.push(FE::zero());
        }
    }

    let mut cpu_trace =
        TraceTable::from_columns_main(vec![add_flags, mul_flags, a_vals, b_vals, c_vals], 1);

    // Build ADD trace from CPU adds
    let mut add_a = Vec::new();
    let mut add_b = Vec::new();
    let mut add_c = Vec::new();
    let mut add_mult = Vec::new();

    for i in 0..8 {
        if cpu_trace.get_main(i, 0) == &FE::one() {
            add_a.push(cpu_trace.get_main(i, 2).clone());
            add_b.push(cpu_trace.get_main(i, 3).clone());
            add_c.push(cpu_trace.get_main(i, 4).clone());
            add_mult.push(FE::one());
        }
    }
    // Pad to 4 rows
    while add_a.len() < 4 {
        add_a.push(FE::zero());
        add_b.push(FE::zero());
        add_c.push(FE::zero());
        add_mult.push(FE::zero());
    }

    let mut add_trace = TraceTable::from_columns_main(vec![add_a, add_b, add_c, add_mult], 1);

    // Build MUL trace from CPU muls
    let mut mul_a = Vec::new();
    let mut mul_b = Vec::new();
    let mut mul_c = Vec::new();
    let mut mul_mult = Vec::new();

    for i in 0..8 {
        if cpu_trace.get_main(i, 1) == &FE::one() {
            mul_a.push(cpu_trace.get_main(i, 2).clone());
            mul_b.push(cpu_trace.get_main(i, 3).clone());
            mul_c.push(cpu_trace.get_main(i, 4).clone());
            mul_mult.push(FE::one());
        }
    }
    // Pad to 4 rows
    while mul_a.len() < 4 {
        mul_a.push(FE::zero());
        mul_b.push(FE::zero());
        mul_c.push(FE::zero());
        mul_mult.push(FE::zero());
    }

    let mut mul_trace = TraceTable::from_columns_main(vec![mul_a, mul_b, mul_c, mul_mult], 1);

    // Prove and verify
    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Random valid operations should verify"
    );
}

// =============================================================================
// Native u64 Combining Operations Tests
// =============================================================================
// These tests verify that the native u64 combining operations produce correct
// results and match what the field operations would produce.

use crate::lookup::{combine_word2l_native, combine_word4l_native};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Native Word2L combining should produce h0 + h1 * 2^16
    #[test]
    fn prop_native_word2l_formula(
        h0 in 0u64..65536u64,
        h1 in 0u64..65536u64,
    ) {
        let result = combine_word2l_native(h0, h1);
        let expected = h0 + (h1 << 16);
        prop_assert_eq!(result, expected);
    }

    /// Native Word4L combining should produce b0 + b1*2^8 + b2*2^16 + b3*2^24
    #[test]
    fn prop_native_word4l_formula(
        b0 in 0u64..256u64,
        b1 in 0u64..256u64,
        b2 in 0u64..256u64,
        b3 in 0u64..256u64,
    ) {
        let result = combine_word4l_native(b0, b1, b2, b3);
        let expected = b0 + (b1 << 8) + (b2 << 16) + (b3 << 24);
        prop_assert_eq!(result, expected);
    }

    /// Native Direct packing should return input unchanged
    #[test]
    fn prop_native_direct_identity(value in 0u64..1000000u64) {
        let columns = vec![value];
        let result = Packing::Direct.combine_native(&columns);
        prop_assert_eq!(result.len(), 1);
        prop_assert_eq!(result[0], value);
    }

    /// Native Word2L packing should work correctly
    #[test]
    fn prop_native_packing_word2l(
        h0 in 0u64..65536u64,
        h1 in 0u64..65536u64,
    ) {
        let columns = vec![h0, h1];
        let result = Packing::Word2L.combine_native(&columns);
        prop_assert_eq!(result.len(), 1);
        prop_assert_eq!(result[0], h0 + (h1 << 16));
    }

    /// Native Word4L packing should work correctly
    #[test]
    fn prop_native_packing_word4l(
        b0 in 0u64..256u64,
        b1 in 0u64..256u64,
        b2 in 0u64..256u64,
        b3 in 0u64..256u64,
    ) {
        let columns = vec![b0, b1, b2, b3];
        let result = Packing::Word4L.combine_native(&columns);
        prop_assert_eq!(result.len(), 1);
        prop_assert_eq!(result[0], b0 + (b1 << 8) + (b2 << 16) + (b3 << 24));
    }

    /// Native DWordHL (2x Word2L) should produce two combined values
    #[test]
    fn prop_native_packing_dwordhl(
        h0 in 0u64..65536u64,
        h1 in 0u64..65536u64,
        h2 in 0u64..65536u64,
        h3 in 0u64..65536u64,
    ) {
        let columns = vec![h0, h1, h2, h3];
        let result = Packing::DWordHL.combine_native(&columns);
        prop_assert_eq!(result.len(), 2);
        prop_assert_eq!(result[0], h0 + (h1 << 16));
        prop_assert_eq!(result[1], h2 + (h3 << 16));
    }

    /// Native DWordBL (2x Word4L) should produce two combined values
    #[test]
    fn prop_native_packing_dwordbl(
        bytes in prop::collection::vec(0u64..256u64, 8),
    ) {
        let result = Packing::DWordBL.combine_native(&bytes);
        prop_assert_eq!(result.len(), 2);

        let expected0 = bytes[0] + (bytes[1] << 8) + (bytes[2] << 16) + (bytes[3] << 24);
        let expected1 = bytes[4] + (bytes[5] << 8) + (bytes[6] << 16) + (bytes[7] << 24);

        prop_assert_eq!(result[0], expected0);
        prop_assert_eq!(result[1], expected1);
    }

    /// Native QuadHL (4x Word2L) should produce four combined values
    #[test]
    fn prop_native_packing_quadhl(
        halves in prop::collection::vec(0u64..65536u64, 8),
    ) {
        let result = Packing::QuadHL.combine_native(&halves);
        prop_assert_eq!(result.len(), 4);
        prop_assert_eq!(result[0], halves[0] + (halves[1] << 16));
        prop_assert_eq!(result[1], halves[2] + (halves[3] << 16));
        prop_assert_eq!(result[2], halves[4] + (halves[5] << 16));
        prop_assert_eq!(result[3], halves[6] + (halves[7] << 16));
    }

    /// Native combining should produce values in valid Goldilocks range
    /// Goldilocks prime: p = 2^64 - 2^32 + 1 = 18446744069414584321
    #[test]
    fn prop_native_word4l_in_goldilocks_range(
        b0 in 0u64..256u64,
        b1 in 0u64..256u64,
        b2 in 0u64..256u64,
        b3 in 0u64..256u64,
    ) {
        let result = combine_word4l_native(b0, b1, b2, b3);
        // The max value is 255 + 255*256 + 255*65536 + 255*16777216 = 4294967295 (2^32 - 1)
        // which is well within the Goldilocks field
        let goldilocks_prime: u64 = 18446744069414584321;
        prop_assert!(result < goldilocks_prime);
    }
}

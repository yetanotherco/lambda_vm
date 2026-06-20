//! Tests for various AIR implementations (Fibonacci, periodic, RAP, memory, etc.).

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField,
    goldilocks::GoldilocksField,
};

use crate::traits::AIR;
use crate::{
    examples::{
        bit_flags::{self, BitFlagsAIR},
        dummy_air::{self, DummyAIR},
        fibonacci_2_cols_shifted::{self, Fibonacci2ColsShifted},
        fibonacci_2_columns::{self, Fibonacci2ColsAIR},
        fibonacci_multi_column::{self, FibonacciMultiColumnAIR},
        fibonacci_rap::{FibonacciRAP, FibonacciRAPPublicInputs, fibonacci_rap_trace},
        quadratic_air::{self, QuadraticAIR, QuadraticPublicInputs},
        read_only_memory::{ReadOnlyPublicInputs, ReadOnlyRAP, sort_rap_trace},
        simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs},
        simple_periodic_cols::{self, SimplePeriodicAIR, SimplePeriodicPublicInputs}, //         simple_periodic_cols::{self, SimplePeriodicAIR, SimplePeriodicPublicInputs},
    },
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    verifier::{IsStarkVerifier, Verifier},
};

type F = GoldilocksField;
type Felt = FieldElement<GoldilocksField>;

use crate::examples::read_only_memory_logup::{
    LogReadOnlyPublicInputs, LogReadOnlyRAP, read_only_logup_trace,
};
use crate::test_utils::multi_prove_ram;

#[test_log::test]
fn test_prove_fib() {
    let mut trace = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };

    let air = FibonacciAIR::<GoldilocksField>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .unwrap();
    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<F>::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_simple_periodic_8() {
    let mut trace = simple_periodic_cols::simple_periodic_trace::<GoldilocksField>(8);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = SimplePeriodicPublicInputs {
        a0: Felt::one(),
        a1: Felt::from(8),
    };

    let air = SimplePeriodicAIR::<GoldilocksField>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .unwrap();
    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<F>::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_simple_periodic_32() {
    let mut trace = simple_periodic_cols::simple_periodic_trace::<GoldilocksField>(32);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = SimplePeriodicPublicInputs {
        a0: Felt::one(),
        a1: Felt::from(32768),
    };

    let air = SimplePeriodicAIR::<GoldilocksField>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<F>::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_fib_2_cols() {
    let mut trace = fibonacci_2_columns::compute_trace([Felt::from(1), Felt::from(1)], 16);

    let proof_options = ProofOptions::default_test_options();
    let pub_inputs = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };

    let air = Fibonacci2ColsAIR::<GoldilocksField>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<F>::new(&[])
    ));
}

#[test_log::test]
fn test_prove_fib_2_cols_shifted() {
    let mut trace = fibonacci_2_cols_shifted::compute_trace(FieldElement::one(), 16);

    let claimed_index = 14;
    let claimed_value = trace.main_table.get_row(claimed_index)[0];
    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = fibonacci_2_cols_shifted::PublicInputs {
        claimed_value,
        claimed_index,
    };

    let air = Fibonacci2ColsShifted::<GoldilocksField>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<F>::new(&[])
    ));
}

#[test_log::test]
fn test_prove_quadratic() {
    let mut trace = quadratic_air::quadratic_trace(Felt::from(3), 32);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = QuadraticPublicInputs { a0: Felt::from(3) };

    let air = QuadraticAIR::<GoldilocksField>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<F>::new(&[])
    ));
}

#[test_log::test]
fn test_prove_rap_fib() {
    let steps = 16;
    let mut trace = fibonacci_rap_trace([Felt::from(1), Felt::from(1)], steps);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciRAPPublicInputs {
        steps,
        a0: Felt::one(),
        a1: Felt::one(),
    };

    let air = FibonacciRAP::<GoldilocksField>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<F>::new(&[])
    ));
}

#[test_log::test]
fn test_prove_dummy() {
    let trace_length = 16;
    let mut trace = dummy_air::dummy_trace(trace_length);

    let proof_options = ProofOptions::default_test_options();

    let air = DummyAIR::new(&proof_options);

    let proof =
        Prover::prove(&air, &mut trace, &(), &mut DefaultTranscript::<F>::new(&[])).unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<F>::new(&[])
    ));
}

#[test_log::test]
fn test_prove_bit_flags() {
    let mut trace = bit_flags::bit_prefix_flag_trace(32);
    let proof_options = ProofOptions::default_test_options();

    let air = BitFlagsAIR::new(&proof_options);

    let proof =
        Prover::prove(&air, &mut trace, &(), &mut DefaultTranscript::<F>::new(&[])).unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<F>::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_read_only_memory() {
    let address_col = vec![
        FieldElement::<GoldilocksField>::from(3), // a0
        FieldElement::<GoldilocksField>::from(2), // a1
        FieldElement::<GoldilocksField>::from(2), // a2
        FieldElement::<GoldilocksField>::from(3), // a3
        FieldElement::<GoldilocksField>::from(4), // a4
        FieldElement::<GoldilocksField>::from(5), // a5
        FieldElement::<GoldilocksField>::from(1), // a6
        FieldElement::<GoldilocksField>::from(3), // a7
    ];
    let value_col = vec![
        FieldElement::<GoldilocksField>::from(10), // v0
        FieldElement::<GoldilocksField>::from(5),  // v1
        FieldElement::<GoldilocksField>::from(5),  // v2
        FieldElement::<GoldilocksField>::from(10), // v3
        FieldElement::<GoldilocksField>::from(25), // v4
        FieldElement::<GoldilocksField>::from(25), // v5
        FieldElement::<GoldilocksField>::from(7),  // v6
        FieldElement::<GoldilocksField>::from(10), // v7
    ];

    let pub_inputs = ReadOnlyPublicInputs {
        a0: FieldElement::<GoldilocksField>::from(3),
        v0: FieldElement::<GoldilocksField>::from(10),
        a_sorted0: FieldElement::<GoldilocksField>::from(1), // a6
        v_sorted0: FieldElement::<GoldilocksField>::from(7), // v6
    };
    let mut trace = sort_rap_trace(address_col, value_col);
    let proof_options = ProofOptions::default_test_options();

    let air = ReadOnlyRAP::<GoldilocksField>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<F>::new(&[])
    ));
}

#[test_log::test]
fn test_prove_log_read_only_memory() {
    let address_col = vec![
        FieldElement::<GoldilocksField>::from(3), // a0
        FieldElement::<GoldilocksField>::from(2), // a1
        FieldElement::<GoldilocksField>::from(2), // a2
        FieldElement::<GoldilocksField>::from(3), // a3
        FieldElement::<GoldilocksField>::from(4), // a4
        FieldElement::<GoldilocksField>::from(5), // a5
        FieldElement::<GoldilocksField>::from(1), // a6
        FieldElement::<GoldilocksField>::from(3), // a7
    ];
    let value_col = vec![
        FieldElement::<GoldilocksField>::from(30), // v0
        FieldElement::<GoldilocksField>::from(20), // v1
        FieldElement::<GoldilocksField>::from(20), // v2
        FieldElement::<GoldilocksField>::from(30), // v3
        FieldElement::<GoldilocksField>::from(40), // v4
        FieldElement::<GoldilocksField>::from(50), // v5
        FieldElement::<GoldilocksField>::from(10), // v6
        FieldElement::<GoldilocksField>::from(30), // v7
    ];

    let pub_inputs = LogReadOnlyPublicInputs {
        a0: FieldElement::<GoldilocksField>::from(3),
        v0: FieldElement::<GoldilocksField>::from(30),
        a_sorted_0: FieldElement::<GoldilocksField>::from(1),
        v_sorted_0: FieldElement::<GoldilocksField>::from(10),
        m0: FieldElement::<GoldilocksField>::from(1),
    };
    let mut trace = read_only_logup_trace(address_col, value_col);
    let proof_options = ProofOptions::default_test_options();

    let air =
        LogReadOnlyRAP::<GoldilocksField, Degree3GoldilocksExtensionField>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[]),
    )
    .unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[]),
    ));
}

#[test_log::test]
fn test_multi_prove_fib_3_tables() {
    let mut trace_1 = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);
    let mut trace_2 = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 16);
    let mut trace_3 = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 32);
    let proof_options = ProofOptions::default_test_options();

    let pub_inputs_1 = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };
    let pub_inputs_2 = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };
    let pub_inputs_3 = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };

    let air_1 = FibonacciAIR::new(&proof_options);
    let air_2 = FibonacciAIR::new(&proof_options);
    let air_3 = FibonacciAIR::new(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<
            Field = GoldilocksField,
            FieldExtension = GoldilocksField,
            PublicInputs = FibonacciPublicInputs<GoldilocksField>,
        >,
        &mut _,
        &_,
    )> = vec![
        (&air_1, &mut trace_1, &pub_inputs_1),
        (&air_2, &mut trace_2, &pub_inputs_2),
        (&air_3, &mut trace_3, &pub_inputs_3),
    ];
    let multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<F>::new(&[])).unwrap();

    let airs: Vec<
        &dyn AIR<
            Field = GoldilocksField,
            FieldExtension = GoldilocksField,
            PublicInputs = FibonacciPublicInputs<GoldilocksField>,
        >,
    > = vec![&air_1, &air_2, &air_3];

    assert!(Verifier::multi_verify_owned(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<F>::new(&[]),
        &FieldElement::zero(),
    ));
}

#[test_log::test]
fn test_multi_prove_2_tables_small_field() {
    let address_col_1 = vec![
        FieldElement::<GoldilocksField>::from(3), // a0
        FieldElement::<GoldilocksField>::from(2), // a1
        FieldElement::<GoldilocksField>::from(2), // a2
        FieldElement::<GoldilocksField>::from(3), // a3
        FieldElement::<GoldilocksField>::from(4), // a4
        FieldElement::<GoldilocksField>::from(5), // a5
        FieldElement::<GoldilocksField>::from(1), // a6
        FieldElement::<GoldilocksField>::from(3), // a7
    ];
    let value_col_1 = vec![
        FieldElement::<GoldilocksField>::from(30), // v0
        FieldElement::<GoldilocksField>::from(20), // v1
        FieldElement::<GoldilocksField>::from(20), // v2
        FieldElement::<GoldilocksField>::from(30), // v3
        FieldElement::<GoldilocksField>::from(40), // v4
        FieldElement::<GoldilocksField>::from(50), // v5
        FieldElement::<GoldilocksField>::from(10), // v6
        FieldElement::<GoldilocksField>::from(30), // v7
    ];

    let address_col_2 = vec![
        FieldElement::<GoldilocksField>::from(15), // a0
        FieldElement::<GoldilocksField>::from(12), // a1
        FieldElement::<GoldilocksField>::from(17), // a2
        FieldElement::<GoldilocksField>::from(10), // a3
        FieldElement::<GoldilocksField>::from(14), // a4
        FieldElement::<GoldilocksField>::from(11), // a5
        FieldElement::<GoldilocksField>::from(16), // a6
        FieldElement::<GoldilocksField>::from(13), // a7
    ];
    let value_col_2 = vec![
        FieldElement::<GoldilocksField>::from(150), // v0
        FieldElement::<GoldilocksField>::from(120), // v1
        FieldElement::<GoldilocksField>::from(170), // v2
        FieldElement::<GoldilocksField>::from(100), // v3
        FieldElement::<GoldilocksField>::from(140), // v4
        FieldElement::<GoldilocksField>::from(110), // v5
        FieldElement::<GoldilocksField>::from(160), // v6
        FieldElement::<GoldilocksField>::from(130), // v7
    ];

    let pub_inputs_1 = LogReadOnlyPublicInputs {
        a0: FieldElement::<GoldilocksField>::from(3),
        v0: FieldElement::<GoldilocksField>::from(30),
        a_sorted_0: FieldElement::<GoldilocksField>::from(1),
        v_sorted_0: FieldElement::<GoldilocksField>::from(10),
        m0: FieldElement::<GoldilocksField>::from(1),
    };

    let pub_inputs_2 = LogReadOnlyPublicInputs {
        a0: FieldElement::<GoldilocksField>::from(15),
        v0: FieldElement::<GoldilocksField>::from(150),
        a_sorted_0: FieldElement::<GoldilocksField>::from(10),
        v_sorted_0: FieldElement::<GoldilocksField>::from(100),
        m0: FieldElement::<GoldilocksField>::from(1),
    };

    let mut trace_1 = read_only_logup_trace(address_col_1, value_col_1);
    let mut trace_2 = read_only_logup_trace(address_col_2, value_col_2);
    let proof_options = ProofOptions::default_test_options();

    let air_1 =
        LogReadOnlyRAP::<GoldilocksField, Degree3GoldilocksExtensionField>::new(&proof_options);
    let air_2 =
        LogReadOnlyRAP::<GoldilocksField, Degree3GoldilocksExtensionField>::new(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<
            Field = GoldilocksField,
            FieldExtension = Degree3GoldilocksExtensionField,
            PublicInputs = LogReadOnlyPublicInputs<GoldilocksField>,
        >,
        &mut _,
        &_,
    )> = vec![
        (&air_1, &mut trace_1, &pub_inputs_1),
        (&air_2, &mut trace_2, &pub_inputs_2),
    ];

    let multi_proof = multi_prove_ram(
        air_trace_pairs,
        &mut DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[]),
    )
    .unwrap();

    let airs: Vec<
        &dyn AIR<
            Field = GoldilocksField,
            FieldExtension = Degree3GoldilocksExtensionField,
            PublicInputs = LogReadOnlyPublicInputs<GoldilocksField>,
        >,
    > = vec![&air_1, &air_2];

    assert!(Verifier::multi_verify_owned(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[]),
        &FieldElement::zero(),
    ));
}

#[test_log::test]
fn test_multi_prove_different_airs() {
    let mut trace_1 = dummy_air::dummy_trace(16);
    let mut trace_2 = bit_flags::bit_prefix_flag_trace(32);
    let proof_options = ProofOptions::default_test_options();

    let air_1 = DummyAIR::new(&proof_options);
    let air_2 = BitFlagsAIR::new(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = GoldilocksField, FieldExtension = GoldilocksField, PublicInputs = ()>,
        &mut _,
        &_,
    )> = vec![(&air_1, &mut trace_1, &()), (&air_2, &mut trace_2, &())];

    let multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<F>::new(&[])).unwrap();

    let airs: Vec<
        &dyn AIR<Field = GoldilocksField, FieldExtension = GoldilocksField, PublicInputs = ()>,
    > = vec![&air_1, &air_2];

    assert!(Verifier::multi_verify_owned(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<F>::new(&[]),
        &FieldElement::zero(),
    ));
}

// Type aliases for multi-column Fibonacci tests
type GoldilocksExt = Degree3GoldilocksExtensionField;
type GoldilocksFE = FieldElement<GoldilocksField>;

#[test]
fn test_multi_column_fibonacci_2_cols() {
    let proof_options = ProofOptions::default_test_options();
    let num_columns = 2;
    let trace_length = 16;

    // Create initial values for each column
    let initial_values: Vec<(GoldilocksFE, GoldilocksFE)> = (0..num_columns)
        .map(|i| {
            (
                GoldilocksFE::from((i + 1) as u64),
                GoldilocksFE::from((i + 2) as u64),
            )
        })
        .collect();

    let mut trace = fibonacci_multi_column::compute_trace::<GoldilocksField, GoldilocksExt>(
        &initial_values,
        trace_length,
    );
    let pub_inputs = fibonacci_multi_column::create_public_inputs(initial_values);
    let air = FibonacciMultiColumnAIR::<GoldilocksField, GoldilocksExt>::with_num_columns(
        &proof_options,
        num_columns,
    );

    let proof = Prover::<GoldilocksField, GoldilocksExt, _>::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<GoldilocksExt>::new(&[]),
    )
    .unwrap();

    assert!(Verifier::<GoldilocksField, GoldilocksExt, _>::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<GoldilocksExt>::new(&[])
    ));
}

#[test]
fn test_multi_column_fibonacci_4_cols() {
    let proof_options = ProofOptions::default_test_options();
    let num_columns = 4;
    let trace_length = 16;

    let initial_values: Vec<(GoldilocksFE, GoldilocksFE)> = (0..num_columns)
        .map(|i| {
            (
                GoldilocksFE::from((i + 1) as u64),
                GoldilocksFE::from((i + 2) as u64),
            )
        })
        .collect();

    let mut trace = fibonacci_multi_column::compute_trace::<GoldilocksField, GoldilocksExt>(
        &initial_values,
        trace_length,
    );
    let pub_inputs = fibonacci_multi_column::create_public_inputs(initial_values);
    let air = FibonacciMultiColumnAIR::<GoldilocksField, GoldilocksExt>::with_num_columns(
        &proof_options,
        num_columns,
    );

    let proof = Prover::<GoldilocksField, GoldilocksExt, _>::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<GoldilocksExt>::new(&[]),
    )
    .unwrap();

    assert!(Verifier::<GoldilocksField, GoldilocksExt, _>::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<GoldilocksExt>::new(&[])
    ));
}

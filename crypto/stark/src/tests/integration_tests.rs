use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};

use crate::traits::AIR;
use crate::{
    Felt252,
    examples::{
        bit_flags::{self, BitFlagsAIR},
        dummy_air::{self, DummyAIR},
        fibonacci_2_cols_shifted::{self, Fibonacci2ColsShifted},
        fibonacci_2_columns::{self, Fibonacci2ColsAIR},
        fibonacci_rap::{FibonacciRAP, FibonacciRAPPublicInputs, fibonacci_rap_trace},
        quadratic_air::{self, QuadraticAIR, QuadraticPublicInputs},
        read_only_memory::{ReadOnlyPublicInputs, ReadOnlyRAP, sort_rap_trace},
        simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs},
        simple_periodic_cols::{self, SimplePeriodicAIR, SimplePeriodicPublicInputs}, //         simple_periodic_cols::{self, SimplePeriodicAIR, SimplePeriodicPublicInputs},
    },
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    transcript::StoneProverTranscript,
    verifier::{IsStarkVerifier, Verifier},
};

use crate::examples::read_only_memory_logup::{
    LogReadOnlyPublicInputs, LogReadOnlyRAP, read_only_logup_trace,
};

#[test_log::test]
fn test_prove_fib() {
    let mut trace = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 8);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let air = FibonacciAIR::<Stark252PrimeField>::new(&pub_inputs, &proof_options);

    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();
    assert!(Verifier::verify(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_simple_periodic_8() {
    let mut trace = simple_periodic_cols::simple_periodic_trace::<Stark252PrimeField>(8);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = SimplePeriodicPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::from(8),
    };

    let air = SimplePeriodicAIR::<Stark252PrimeField>::new(&pub_inputs, &proof_options);

    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();
    assert!(Verifier::verify(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_simple_periodic_32() {
    let mut trace = simple_periodic_cols::simple_periodic_trace::<Stark252PrimeField>(32);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = SimplePeriodicPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::from(32768),
    };

    let air = SimplePeriodicAIR::<Stark252PrimeField>::new(&pub_inputs, &proof_options);

    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_fib_2_cols() {
    let mut trace = fibonacci_2_columns::compute_trace([Felt252::from(1), Felt252::from(1)], 16);

    let proof_options = ProofOptions::default_test_options();
    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let air = Fibonacci2ColsAIR::<Stark252PrimeField>::new(&pub_inputs, &proof_options);

    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[])
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

    let air = Fibonacci2ColsShifted::<Stark252PrimeField>::new(&pub_inputs, &proof_options);

    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_quadratic() {
    let mut trace = quadratic_air::quadratic_trace(Felt252::from(3), 32);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = QuadraticPublicInputs {
        a0: Felt252::from(3),
    };

    let air = QuadraticAIR::<Stark252PrimeField>::new(&pub_inputs, &proof_options);

    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_rap_fib() {
    let steps = 16;
    let mut trace = fibonacci_rap_trace([Felt252::from(1), Felt252::from(1)], steps);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciRAPPublicInputs {
        steps,
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let air = FibonacciRAP::<Stark252PrimeField>::new(&pub_inputs, &proof_options);

    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_dummy() {
    let trace_length = 16;
    let mut trace = dummy_air::dummy_trace(trace_length);

    let proof_options = ProofOptions::default_test_options();

    let air = DummyAIR::new(&(), &proof_options);

    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_bit_flags() {
    let mut trace = bit_flags::bit_prefix_flag_trace(32);
    let proof_options = ProofOptions::default_test_options();

    let air = BitFlagsAIR::new(&(), &proof_options);

    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_read_only_memory() {
    let address_col = vec![
        FieldElement::<Stark252PrimeField>::from(3), // a0
        FieldElement::<Stark252PrimeField>::from(2), // a1
        FieldElement::<Stark252PrimeField>::from(2), // a2
        FieldElement::<Stark252PrimeField>::from(3), // a3
        FieldElement::<Stark252PrimeField>::from(4), // a4
        FieldElement::<Stark252PrimeField>::from(5), // a5
        FieldElement::<Stark252PrimeField>::from(1), // a6
        FieldElement::<Stark252PrimeField>::from(3), // a7
    ];
    let value_col = vec![
        FieldElement::<Stark252PrimeField>::from(10), // v0
        FieldElement::<Stark252PrimeField>::from(5),  // v1
        FieldElement::<Stark252PrimeField>::from(5),  // v2
        FieldElement::<Stark252PrimeField>::from(10), // v3
        FieldElement::<Stark252PrimeField>::from(25), // v4
        FieldElement::<Stark252PrimeField>::from(25), // v5
        FieldElement::<Stark252PrimeField>::from(7),  // v6
        FieldElement::<Stark252PrimeField>::from(10), // v7
    ];

    let pub_inputs = ReadOnlyPublicInputs {
        a0: FieldElement::<Stark252PrimeField>::from(3),
        v0: FieldElement::<Stark252PrimeField>::from(10),
        a_sorted0: FieldElement::<Stark252PrimeField>::from(1), // a6
        v_sorted0: FieldElement::<Stark252PrimeField>::from(7), // v6
    };
    let mut trace = sort_rap_trace(address_col, value_col);
    let proof_options = ProofOptions::default_test_options();

    let air = ReadOnlyRAP::<Stark252PrimeField>::new(&pub_inputs, &proof_options);

    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_log_read_only_memory() {
    let address_col = vec![
        FieldElement::<Babybear31PrimeField>::from(3), // a0
        FieldElement::<Babybear31PrimeField>::from(2), // a1
        FieldElement::<Babybear31PrimeField>::from(2), // a2
        FieldElement::<Babybear31PrimeField>::from(3), // a3
        FieldElement::<Babybear31PrimeField>::from(4), // a4
        FieldElement::<Babybear31PrimeField>::from(5), // a5
        FieldElement::<Babybear31PrimeField>::from(1), // a6
        FieldElement::<Babybear31PrimeField>::from(3), // a7
    ];
    let value_col = vec![
        FieldElement::<Babybear31PrimeField>::from(30), // v0
        FieldElement::<Babybear31PrimeField>::from(20), // v1
        FieldElement::<Babybear31PrimeField>::from(20), // v2
        FieldElement::<Babybear31PrimeField>::from(30), // v3
        FieldElement::<Babybear31PrimeField>::from(40), // v4
        FieldElement::<Babybear31PrimeField>::from(50), // v5
        FieldElement::<Babybear31PrimeField>::from(10), // v6
        FieldElement::<Babybear31PrimeField>::from(30), // v7
    ];

    let pub_inputs = LogReadOnlyPublicInputs {
        a0: FieldElement::<Babybear31PrimeField>::from(3),
        v0: FieldElement::<Babybear31PrimeField>::from(30),
        a_sorted_0: FieldElement::<Babybear31PrimeField>::from(1),
        v_sorted_0: FieldElement::<Babybear31PrimeField>::from(10),
        m0: FieldElement::<Babybear31PrimeField>::from(1),
    };
    let mut trace = read_only_logup_trace(address_col, value_col);
    let proof_options = ProofOptions::default_test_options();

    let air = LogReadOnlyRAP::<Babybear31PrimeField, Degree4BabyBearExtensionField>::new(
        &pub_inputs,
        &proof_options,
    );

    let proof = Prover::prove(
        &air,
        &mut trace,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    )
    .unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    ));
}

#[test_log::test]
fn test_multi_prove_fib_3_tables() {
    let mut trace_1 = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 8);
    let mut trace_2 = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 16);
    let mut trace_3 = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 32);
    let proof_options = ProofOptions::default_test_options();

    let pub_inputs_1 = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };
    let pub_inputs_2 = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };
    let pub_inputs_3 = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let air_1 = FibonacciAIR::new(&pub_inputs_1, &proof_options);
    let air_2 = FibonacciAIR::new(&pub_inputs_2, &proof_options);
    let air_3 = FibonacciAIR::new(&pub_inputs_3, &proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<
            Field = Stark252PrimeField,
            FieldExtension = Stark252PrimeField,
            PublicInputs = FibonacciPublicInputs<Stark252PrimeField>,
        >,
        &mut _,
    )> = vec![
        (&air_1, &mut trace_1),
        (&air_2, &mut trace_2),
        (&air_3, &mut trace_3),
    ];
    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut StoneProverTranscript::new(&[])).unwrap();

    let airs: Vec<
        &dyn AIR<
            Field = Stark252PrimeField,
            FieldExtension = Stark252PrimeField,
            PublicInputs = FibonacciPublicInputs<Stark252PrimeField>,
        >,
    > = vec![&air_1, &air_2, &air_3];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_multi_prove_2_tables_small_field() {
    let address_col_1 = vec![
        FieldElement::<Babybear31PrimeField>::from(3), // a0
        FieldElement::<Babybear31PrimeField>::from(2), // a1
        FieldElement::<Babybear31PrimeField>::from(2), // a2
        FieldElement::<Babybear31PrimeField>::from(3), // a3
        FieldElement::<Babybear31PrimeField>::from(4), // a4
        FieldElement::<Babybear31PrimeField>::from(5), // a5
        FieldElement::<Babybear31PrimeField>::from(1), // a6
        FieldElement::<Babybear31PrimeField>::from(3), // a7
    ];
    let value_col_1 = vec![
        FieldElement::<Babybear31PrimeField>::from(30), // v0
        FieldElement::<Babybear31PrimeField>::from(20), // v1
        FieldElement::<Babybear31PrimeField>::from(20), // v2
        FieldElement::<Babybear31PrimeField>::from(30), // v3
        FieldElement::<Babybear31PrimeField>::from(40), // v4
        FieldElement::<Babybear31PrimeField>::from(50), // v5
        FieldElement::<Babybear31PrimeField>::from(10), // v6
        FieldElement::<Babybear31PrimeField>::from(30), // v7
    ];

    let address_col_2 = vec![
        FieldElement::<Babybear31PrimeField>::from(15), // a0
        FieldElement::<Babybear31PrimeField>::from(12), // a1
        FieldElement::<Babybear31PrimeField>::from(17), // a2
        FieldElement::<Babybear31PrimeField>::from(10), // a3
        FieldElement::<Babybear31PrimeField>::from(14), // a4
        FieldElement::<Babybear31PrimeField>::from(11), // a5
        FieldElement::<Babybear31PrimeField>::from(16), // a6
        FieldElement::<Babybear31PrimeField>::from(13), // a7
    ];
    let value_col_2 = vec![
        FieldElement::<Babybear31PrimeField>::from(150), // v0
        FieldElement::<Babybear31PrimeField>::from(120), // v1
        FieldElement::<Babybear31PrimeField>::from(170), // v2
        FieldElement::<Babybear31PrimeField>::from(100), // v3
        FieldElement::<Babybear31PrimeField>::from(140), // v4
        FieldElement::<Babybear31PrimeField>::from(110), // v5
        FieldElement::<Babybear31PrimeField>::from(160), // v6
        FieldElement::<Babybear31PrimeField>::from(130), // v7
    ];

    let pub_inputs_1 = LogReadOnlyPublicInputs {
        a0: FieldElement::<Babybear31PrimeField>::from(3),
        v0: FieldElement::<Babybear31PrimeField>::from(30),
        a_sorted_0: FieldElement::<Babybear31PrimeField>::from(1),
        v_sorted_0: FieldElement::<Babybear31PrimeField>::from(10),
        m0: FieldElement::<Babybear31PrimeField>::from(1),
    };

    let pub_inputs_2 = LogReadOnlyPublicInputs {
        a0: FieldElement::<Babybear31PrimeField>::from(15),
        v0: FieldElement::<Babybear31PrimeField>::from(150),
        a_sorted_0: FieldElement::<Babybear31PrimeField>::from(10),
        v_sorted_0: FieldElement::<Babybear31PrimeField>::from(100),
        m0: FieldElement::<Babybear31PrimeField>::from(1),
    };

    let mut trace_1 = read_only_logup_trace(address_col_1, value_col_1);
    let mut trace_2 = read_only_logup_trace(address_col_2, value_col_2);
    let proof_options = ProofOptions::default_test_options();

    let air_1 = LogReadOnlyRAP::<Babybear31PrimeField, Degree4BabyBearExtensionField>::new(
        &pub_inputs_1,
        &proof_options,
    );
    let air_2 = LogReadOnlyRAP::<Babybear31PrimeField, Degree4BabyBearExtensionField>::new(
        &pub_inputs_2,
        &proof_options,
    );

    let air_trace_pairs: Vec<(
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearExtensionField,
            PublicInputs = LogReadOnlyPublicInputs<Babybear31PrimeField>,
        >,
        &mut _,
    )> = vec![(&air_1, &mut trace_1), (&air_2, &mut trace_2)];

    let multi_proof = Prover::multi_prove(
        air_trace_pairs,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    )
    .unwrap();

    let airs: Vec<
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearExtensionField,
            PublicInputs = LogReadOnlyPublicInputs<Babybear31PrimeField>,
        >,
    > = vec![&air_1, &air_2];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    ));
}

#[test_log::test]
fn test_multi_prove_different_airs() {
    let mut trace_1 = dummy_air::dummy_trace(16);
    let mut trace_2 = bit_flags::bit_prefix_flag_trace(32);
    let proof_options = ProofOptions::default_test_options();

    let air_1 = DummyAIR::new(&(), &proof_options);
    let air_2 = BitFlagsAIR::new(&(), &proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = Stark252PrimeField, FieldExtension = Stark252PrimeField, PublicInputs = ()>,
        &mut _,
    )> = vec![(&air_1, &mut trace_1), (&air_2, &mut trace_2)];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut StoneProverTranscript::new(&[])).unwrap();

    let airs: Vec<
        &dyn AIR<Field = Stark252PrimeField, FieldExtension = Stark252PrimeField, PublicInputs = ()>,
    > = vec![&air_1, &air_2];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut StoneProverTranscript::new(&[]),
    ));
}

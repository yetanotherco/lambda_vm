pub mod plonky3_config;
pub mod plonky3_fibonacci;

#[cfg(test)]
mod tests {
    use super::*;

    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksField;
    use p3_uni_stark::{prove, verify};
    use stark::examples::fibonacci_multi_column::{
        FibonacciMultiColumnAIR, compute_trace, create_public_inputs,
    };
    use stark::proof::options::ProofOptions;
    use stark::prover::{IsStarkProver, Prover};
    use stark::verifier::{IsStarkVerifier, Verifier};

    type F = GoldilocksField;
    type E = Degree3GoldilocksExtensionField;
    type FE = FieldElement<F>;

    fn benchmark_proof_options() -> ProofOptions {
        ProofOptions {
            blowup_factor: 4,
            fri_number_of_queries: 30,
            coset_offset: 3,
            grinding_factor: 0,
        }
    }

    #[test]
    fn lambda_fibonacci_prove_verify() {
        let num_columns = 2;
        let trace_length = 256; // 2^8
        let proof_options = benchmark_proof_options();

        let initial_values: Vec<(FE, FE)> = (0..num_columns)
            .map(|i| (FE::from((i + 1) as u64), FE::from((i + 2) as u64)))
            .collect();

        let mut trace = compute_trace::<F, E>(&initial_values, trace_length);
        let pub_inputs = create_public_inputs(initial_values);
        let air =
            FibonacciMultiColumnAIR::<F, E>::with_num_columns(&proof_options, num_columns);

        let proof = Prover::<F, E, _>::prove(
            &air,
            &mut trace,
            &pub_inputs,
            &mut DefaultTranscript::<E>::new(&[]),
        )
        .unwrap();

        assert!(Verifier::<F, E, _>::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<E>::new(&[]),
        ));
    }

    #[test]
    fn plonky3_fibonacci_prove_verify() {
        let num_sequences = 2;
        // Plonky3 uses 2 columns per sequence, half the rows
        let p3_rows = 128; // Lambda's 256 rows / 2

        let config = plonky3_config::matched_params_config();
        let air = plonky3_fibonacci::P3FibonacciAir { num_sequences };
        let trace = plonky3_fibonacci::generate_fibonacci_trace(num_sequences, p3_rows);

        let proof = prove(&config, &air, trace, &[]);
        verify(&config, &air, &proof, &[]).expect("Plonky3 verification failed");
    }
}

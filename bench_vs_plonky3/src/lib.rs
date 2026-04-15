pub mod lambda_fibonacci_pair;
pub mod plonky3_config;
pub mod plonky3_fibonacci;

#[cfg(test)]
mod tests {
    use super::*;

    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksField;
    use p3_field::PrimeField64;
    use p3_uni_stark::{prove, verify};
    use stark::proof::options::ProofOptions;
    use stark::prover::{IsStarkProver, Prover};
    use stark::verifier::{IsStarkVerifier, Verifier};

    type F = GoldilocksField;
    type E = Degree3GoldilocksExtensionField;
    type FE = FieldElement<F>;

    fn benchmark_proof_options() -> ProofOptions {
        ProofOptions {
            blowup_factor: 4,
            fri_number_of_queries: 100,
            coset_offset: 3,
            grinding_factor: 0,
        }
    }

    #[test]
    fn lambda_fibonacci_pair_prove_verify() {
        let num_sequences = 2;
        let trace_length = 128; // 2^7
        let proof_options = benchmark_proof_options();

        let initial_values: Vec<(FE, FE)> = (0..num_sequences)
            .map(|i| (FE::from((i + 1) as u64), FE::from((i + 2) as u64)))
            .collect();

        let mut trace =
            lambda_fibonacci_pair::compute_trace::<F, E>(&initial_values, trace_length);
        let pub_inputs = lambda_fibonacci_pair::create_public_inputs(initial_values);
        let air = lambda_fibonacci_pair::FibonacciPairMultiColAIR::<F, E>::with_num_sequences(
            &proof_options,
            num_sequences,
        );

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
        let rows = 128; // 2^7

        let config = plonky3_config::matched_params_config();
        let air = plonky3_fibonacci::P3FibonacciAir { num_sequences };
        let trace = plonky3_fibonacci::generate_fibonacci_trace(num_sequences, rows);
        let pis = plonky3_fibonacci::public_values(num_sequences);

        let proof = prove(&config, &air, trace, &pis);
        verify(&config, &air, &proof, &pis).expect("Plonky3 verification failed");
    }

    /// Verifies that the new Lambda pair AIR trace and the Plonky3 trace are
    /// cell-by-cell identical at the same (row, col) coordinates.
    #[test]
    fn lambda_pair_trace_matches_plonky3_trace() {
        let num_sequences = 3;
        let rows = 16;

        let initial_values: Vec<(FE, FE)> = (0..num_sequences)
            .map(|i| (FE::from((i + 1) as u64), FE::from((i + 2) as u64)))
            .collect();

        let lambda_trace =
            lambda_fibonacci_pair::compute_trace::<F, E>(&initial_values, rows);
        let p3_trace = plonky3_fibonacci::generate_fibonacci_trace(num_sequences, rows);

        assert_eq!(p3_trace.width, 2 * num_sequences);
        for row in 0..rows {
            for seq in 0..num_sequences {
                let p3_left = p3_trace.values[row * p3_trace.width + 2 * seq].as_canonical_u64();
                let p3_right =
                    p3_trace.values[row * p3_trace.width + 2 * seq + 1].as_canonical_u64();

                assert_eq!(
                    FE::from(p3_left),
                    lambda_trace.get_main(row, 2 * seq).clone(),
                    "left mismatch at row {row}, seq {seq}"
                );
                assert_eq!(
                    FE::from(p3_right),
                    lambda_trace.get_main(row, 2 * seq + 1).clone(),
                    "right mismatch at row {row}, seq {seq}"
                );
            }
        }
    }
}

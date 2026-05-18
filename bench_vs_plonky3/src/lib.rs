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
    use p3_uni_stark::{prove, verify};
    use stark::proof::options::ProofOptions;
    use stark::prover::{IsStarkProver, Prover};
    use stark::verifier::{IsStarkVerifier, Verifier};

    type F = GoldilocksField;
    type E = Degree3GoldilocksExtensionField;
    type FE = FieldElement<F>;

    fn proof_options() -> ProofOptions {
        ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 3,
            coset_offset: 3,
            grinding_factor: 0,
        }
    }

    #[test]
    fn lambda_fibonacci_pair_prove_verify() {
        let num_sequences = 2;
        let rows = 64;
        let options = proof_options();
        let initial_values: Vec<(FE, FE)> = (0..num_sequences)
            .map(|i| (FE::from((i + 1) as u64), FE::from((i + 2) as u64)))
            .collect();

        let mut trace = lambda_fibonacci_pair::compute_trace::<F, E>(&initial_values, rows);
        let pub_inputs = lambda_fibonacci_pair::create_public_inputs(initial_values);
        let air = lambda_fibonacci_pair::FibonacciPairMultiColAIR::<F, E>::with_num_sequences(
            &options,
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
        let rows = 64;
        let config = plonky3_config::params_config(2, 3, 0);
        let air = plonky3_fibonacci::P3FibonacciAir { num_sequences };
        let trace = plonky3_fibonacci::generate_fibonacci_trace(num_sequences, rows);
        let pis = plonky3_fibonacci::public_values(num_sequences);

        let proof = prove(&config, &air, trace, &pis);
        verify(&config, &air, &proof, &pis).expect("Plonky3 verification failed");
    }
}

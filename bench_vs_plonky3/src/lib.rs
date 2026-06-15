pub mod lambda_fibonacci_pair;
pub mod plonky3_config;
pub mod plonky3_fibonacci;
pub mod span_timing;

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
            blowup_factor: 2,
            fri_number_of_queries: 219,
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

        let mut trace = lambda_fibonacci_pair::compute_trace::<F, E>(&initial_values, trace_length);
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

    /// Lambda prove with instruments breakdown + P3 span-based breakdown.
    /// Run: cargo test -p bench-vs-plonky3 --features instruments --release -- instruments_breakdown --ignored --nocapture
    #[test]
    #[ignore = "heavy: run with --release -- instruments_breakdown --ignored --nocapture"]
    fn instruments_breakdown() {
        let num_sequences = 16;
        let rows = 1 << 19;
        let proof_options = benchmark_proof_options();

        let initial_values: Vec<(FE, FE)> = (0..num_sequences)
            .map(|i| (FE::from((i + 1) as u64), FE::from((i + 2) as u64)))
            .collect();

        let mut trace = lambda_fibonacci_pair::compute_trace::<F, E>(&initial_values, rows);
        let pub_inputs = lambda_fibonacci_pair::create_public_inputs(initial_values);
        let air = lambda_fibonacci_pair::FibonacciPairMultiColAIR::<F, E>::with_num_sequences(
            &proof_options,
            num_sequences,
        );

        let start = std::time::Instant::now();
        let _proof = Prover::<F, E, _>::prove(
            &air,
            &mut trace,
            &pub_inputs,
            &mut DefaultTranscript::<E>::new(&[]),
        )
        .unwrap();
        let total = start.elapsed();

        println!("\n============================================================");
        println!(
            "Lambda STARK Instruments (blowup={}, queries={})",
            proof_options.blowup_factor, proof_options.fri_number_of_queries
        );
        println!("Trace: {} rows x {} cols", rows, 2 * num_sequences);
        println!("Total prove: {:.3}s", total.as_secs_f64());

        #[cfg(feature = "instruments")]
        if let Some(timing) = stark::instruments::take() {
            println!("\n--- High-level phases ---");
            println!(
                "  Pre-pass:            {:>8.1}ms",
                timing.prepass.as_secs_f64() * 1000.0
            );
            println!(
                "  R1 Main commits:     {:>8.1}ms",
                timing.main_commits.as_secs_f64() * 1000.0
            );
            println!(
                "  R1 Aux build:        {:>8.1}ms",
                timing.aux_build.as_secs_f64() * 1000.0
            );
            println!(
                "  R1 Aux commit:       {:>8.1}ms",
                timing.aux_commit.as_secs_f64() * 1000.0
            );
            println!(
                "  Rounds 2-4:          {:>8.1}ms",
                timing.rounds_2_4.as_secs_f64() * 1000.0
            );

            let r1 = &timing.round1_sub;
            println!("\n--- Round 1 sub-ops ---");
            println!(
                "  Main LDE (FFT):      {:>8.1}ms",
                r1.main_lde.as_secs_f64() * 1000.0
            );
            println!(
                "  Main Merkle:         {:>8.1}ms",
                r1.main_merkle.as_secs_f64() * 1000.0
            );

            for (name, tbl_rows, dur, sub) in &timing.table_timings {
                println!(
                    "\n--- Rounds 2-4: {} ({} rows, {:.1}ms) ---",
                    name,
                    tbl_rows,
                    dur.as_secs_f64() * 1000.0
                );
                println!(
                    "  R2 constraint eval:{:>8.1}ms  ({:.0}%)",
                    sub.constraints.as_secs_f64() * 1000.0,
                    sub.constraints.as_secs_f64() / total.as_secs_f64() * 100.0
                );
                println!(
                    "  R2 decompose+ext:  {:>8.1}ms  ({:.0}%)",
                    sub.comp_decompose.as_secs_f64() * 1000.0,
                    sub.comp_decompose.as_secs_f64() / total.as_secs_f64() * 100.0
                );
                println!(
                    "  R2 comp Merkle:    {:>8.1}ms  ({:.0}%)",
                    sub.comp_commit.as_secs_f64() * 1000.0,
                    sub.comp_commit.as_secs_f64() / total.as_secs_f64() * 100.0
                );
                println!(
                    "  R3 OOD eval:       {:>8.1}ms  ({:.0}%)",
                    sub.ood.as_secs_f64() * 1000.0,
                    sub.ood.as_secs_f64() / total.as_secs_f64() * 100.0
                );
                println!(
                    "  R4 deep comp:      {:>8.1}ms  ({:.0}%)",
                    sub.deep_comp.as_secs_f64() * 1000.0,
                    sub.deep_comp.as_secs_f64() / total.as_secs_f64() * 100.0
                );
                println!(
                    "  R4 deep extend:    {:>8.1}ms  ({:.0}%)",
                    sub.deep_extend.as_secs_f64() * 1000.0,
                    sub.deep_extend.as_secs_f64() / total.as_secs_f64() * 100.0
                );
                println!(
                    "  R4 FRI commit:     {:>8.1}ms  ({:.0}%)",
                    sub.fri_commit.as_secs_f64() * 1000.0,
                    sub.fri_commit.as_secs_f64() / total.as_secs_f64() * 100.0
                );
                println!(
                    "  R4 queries+open:   {:>8.1}ms  ({:.0}%)",
                    sub.queries.as_secs_f64() * 1000.0,
                    sub.queries.as_secs_f64() / total.as_secs_f64() * 100.0
                );
            }
        }

        #[cfg(not(feature = "instruments"))]
        println!("(rebuild with --features instruments for breakdown)");

        // --- Plonky3 breakdown via tracing spans ---
        // Captures ALL spans (info + debug) so we see quotient_values, FRI commit, etc.
        println!("\n============================================================");
        println!("Plonky3 STARK Span Breakdown");

        use tracing_subscriber::layer::SubscriberExt;

        let (layer, results) = crate::span_timing::P3TimingLayer::new();
        let filter = tracing_subscriber::filter::LevelFilter::DEBUG;
        let subscriber = tracing_subscriber::registry().with(filter).with(layer);

        let config = plonky3_config::matched_params_config();
        let p3_air = plonky3_fibonacci::P3FibonacciAir { num_sequences };
        let p3_trace = plonky3_fibonacci::generate_fibonacci_trace(num_sequences, rows);
        let p3_pis = plonky3_fibonacci::public_values(num_sequences);

        let p3_prove_dur;
        {
            let _guard = tracing::subscriber::set_default(subscriber);
            let p3_start = std::time::Instant::now();
            let _p3_proof = p3_uni_stark::prove(&config, &p3_air, p3_trace, &p3_pis);
            p3_prove_dur = p3_start.elapsed();
        }

        let total_ms = p3_prove_dur.as_secs_f64() * 1000.0;
        println!("  Prove total:  {:.1}ms\n", total_ms);

        // Sort spans by duration descending and print
        let mut span_data = results.lock().unwrap().clone();
        span_data.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (name, ms) in &span_data {
            if *ms >= 0.1 {
                println!(
                    "  {:.<40} {:>8.1}ms  ({:.0}%)",
                    name,
                    ms,
                    ms / total_ms * 100.0
                );
            }
        }
        let accounted: f64 = span_data.iter().map(|(_, ms)| ms).sum();
        let unaccounted = total_ms - accounted;
        if unaccounted > 1.0 {
            println!(
                "  {:.<40} {:>8.1}ms  ({:.0}%)",
                "(unaccounted)",
                unaccounted,
                unaccounted / total_ms * 100.0
            );
        }
        println!("============================================================\n");
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

        let lambda_trace = lambda_fibonacci_pair::compute_trace::<F, E>(&initial_values, rows);
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

use crypto::fiat_shamir::default_transcript::DefaultTranscript;

use crate::{
    domain::{Domain, DomainConstants},
    examples::{
        quadratic_air::QuadraticAIR,
        simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs},
    },
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover, evaluate_polynomial_on_lde_domain},
    test_utils::multi_prove_ram,
    tests::domain_cache_stats,
    trace::{LDETraceTable, get_trace_evaluations, get_trace_evaluations_from_lde},
    traits::AIR,
    verifier::{IsStarkVerifier, Verifier},
};
use math::{
    field::{element::FieldElement, goldilocks::GoldilocksField, traits::IsFFTField},
    polynomial::Polynomial,
};

type Felt = FieldElement<GoldilocksField>;

#[test]
fn test_domain_constructor() {
    let trace = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);
    let trace_length = trace.num_rows();
    let coset_offset = 3;
    let blowup_factor: usize = 2;
    let grinding_factor = 20;

    let proof_options = ProofOptions {
        blowup_factor: blowup_factor as u8,
        fri_number_of_queries: 1,
        coset_offset,
        grinding_factor,
    };

    let domain = Domain::new(
        &simple_fibonacci::FibonacciAIR::new(&proof_options),
        trace_length,
    );
    assert_eq!(domain.blowup_factor, 2);
    assert_eq!(domain.interpolation_domain_size, trace_length);
    assert_eq!(domain.root_order, trace_length.trailing_zeros());
    assert_eq!(domain.coset_offset, FieldElement::from(coset_offset));

    let primitive_root = GoldilocksField::get_primitive_root_of_unity(
        (trace_length * blowup_factor).trailing_zeros() as u64,
    )
    .unwrap();

    assert_eq!(
        domain.trace_primitive_root,
        primitive_root.pow(blowup_factor)
    );
    for i in 0..(trace_length * blowup_factor) {
        assert_eq!(
            domain.lde_roots_of_unity_coset[i],
            primitive_root.pow(i) * FieldElement::from(coset_offset)
        );
    }
}

#[test]
fn test_evaluate_polynomial_on_lde_domain_on_trace_polys() {
    let trace = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);

    let trace_length = trace.num_rows();

    let trace_polys = trace.compute_trace_polys_main::<GoldilocksField>();
    let coset_offset = Felt::from(3);
    let blowup_factor: usize = 2;
    let domain_size = 8;

    let primitive_root = GoldilocksField::get_primitive_root_of_unity(
        (trace_length * blowup_factor).trailing_zeros() as u64,
    )
    .unwrap();

    for poly in trace_polys.iter() {
        let lde_evaluation =
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, domain_size, &coset_offset)
                .unwrap();
        assert_eq!(lde_evaluation.len(), trace_length * blowup_factor);
        for (i, evaluation) in lde_evaluation.iter().enumerate() {
            assert_eq!(
                *evaluation,
                poly.evaluate(&(coset_offset * primitive_root.pow(i)))
            );
        }
    }
}

#[test]
fn test_evaluate_polynomial_on_lde_domain_edge_case() {
    let poly = Polynomial::new_monomial(Felt::one(), 8);
    let blowup_factor: usize = 4;
    let domain_size: usize = 8;
    let offset = Felt::from(3);
    let evaluations =
        evaluate_polynomial_on_lde_domain(&poly, blowup_factor, domain_size, &offset).unwrap();
    assert_eq!(evaluations.len(), domain_size * blowup_factor);

    let primitive_root: Felt = GoldilocksField::get_primitive_root_of_unity(
        (domain_size * blowup_factor).trailing_zeros() as u64,
    )
    .unwrap();
    for (i, eval) in evaluations.iter().enumerate() {
        assert_eq!(*eval, poly.evaluate(&(offset * primitive_root.pow(i))));
    }
}

/// Tests that `get_trace_evaluations_from_lde` (barycentric) produces identical
/// results to `get_trace_evaluations` (Horner) for the Fibonacci trace.
#[test]
fn barycentric_trace_eval_matches_horner_trace_eval() {
    let trace = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);
    let trace_length = trace.num_rows();
    let blowup_factor: usize = 4;
    let coset_offset = 3u64;

    let proof_options = ProofOptions {
        blowup_factor: blowup_factor as u8,
        fri_number_of_queries: 1,
        coset_offset,
        grinding_factor: 0,
    };

    let air = simple_fibonacci::FibonacciAIR::<GoldilocksField>::new(&proof_options);
    let domain = Domain::new(&air, trace_length);

    // Compute trace polys (Horner path)
    let trace_polys = trace.compute_trace_polys_main::<GoldilocksField>();

    // Compute LDE evaluations for each column
    let lde_evaluations: Vec<Vec<Felt>> = trace_polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(
                poly,
                domain.blowup_factor,
                domain.interpolation_domain_size,
                &domain.coset_offset,
            )
            .expect("LDE evaluation failed")
        })
        .collect();

    // Build LDE trace table
    let lde_trace = LDETraceTable::from_columns(
        lde_evaluations,
        Vec::<Vec<Felt>>::new(),
        air.step_size(),
        domain.blowup_factor,
    );

    // Pick OOD point (just a deterministic value)
    let z = Felt::from(12345u64);

    let frame_offsets = air.context().transition_offsets.clone();
    let step_size = air.step_size();

    // Horner-based evaluation (ground truth)
    let expected = get_trace_evaluations::<GoldilocksField, GoldilocksField>(
        &trace_polys,
        &[],
        &z,
        &frame_offsets,
        &domain.trace_primitive_root,
        step_size,
    );

    let dc = DomainConstants::from_domain(&domain);

    // Barycentric evaluation (new path)
    let result =
        get_trace_evaluations_from_lde(&lde_trace, &domain, &z, &frame_offsets, step_size, &dc);

    assert_eq!(result.width, expected.width);
    assert_eq!(result.height, expected.height);
    assert_eq!(result.data, expected.data);
}

/// Test that direct quotient decomposition produces identical results to
/// the original iFFT + break_in_parts + FFT pipeline.
#[test]
fn test_decompose_and_extend_d2_matches_original() {
    // Build a known polynomial H of degree < 2N, evaluate on LDE coset.
    let n = 16usize; // trace length
    let blowup_factor = 2usize;
    let two_n = n * blowup_factor;

    let proof_options = ProofOptions {
        blowup_factor: blowup_factor as u8,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 0,
    };

    // We need an AIR with composition_poly_degree_bound = 2 * trace_length.
    // Use QuadraticAIR for this.
    let air = QuadraticAIR::<GoldilocksField>::new(&proof_options);
    let domain = Domain::new(&air, n);

    // Create a random-ish polynomial H(x) of degree < 2N (= 32 coefficients).
    let coeffs: Vec<Felt> = (0..two_n)
        .map(|i| FieldElement::from((i * 37 + 13) as u64))
        .collect();
    let h_poly = Polynomial::new(&coeffs);

    // Evaluate H on the LDE coset (2N points: g·ω^i for i=0..2N-1)
    let constraint_evaluations: Vec<Felt> = domain
        .lde_roots_of_unity_coset
        .iter()
        .map(|x| h_poly.evaluate(x))
        .collect();
    assert_eq!(constraint_evaluations.len(), two_n);

    // --- Original path: iFFT(2N) + break_in_parts(2) + FFT(2N) each ---
    let composition_poly =
        Polynomial::interpolate_offset_fft(&constraint_evaluations, &domain.coset_offset)
            .expect("interpolation failed");
    let parts = composition_poly.break_in_parts(2);
    let original: Vec<Vec<Felt>> = parts
        .iter()
        .map(|part| {
            evaluate_polynomial_on_lde_domain(part, blowup_factor, n, &domain.coset_offset)
                .expect("LDE evaluation failed")
        })
        .collect();

    // --- New path: algebraic decomposition ---
    let new_result = Prover::<GoldilocksField, GoldilocksField, ()>::decompose_and_extend_d2(
        &constraint_evaluations,
        &domain,
    );

    assert_eq!(new_result.len(), 2);
    assert_eq!(new_result[0].len(), original[0].len());
    assert_eq!(new_result[1].len(), original[1].len());
    for i in 0..new_result[0].len() {
        assert_eq!(new_result[0][i], original[0][i], "H₀ mismatch at index {i}");
        assert_eq!(new_result[1][i], original[1][i], "H₁ mismatch at index {i}");
    }
}

/// Test that the domain cache 3-tuple key `(trace_length, blowup, coset_offset)` correctly
/// distinguishes AIRs that share the same `(trace_length, blowup)` but differ in
/// `coset_offset`. Both AIRs must get their own `Domain` and the resulting proofs must
/// verify successfully.
#[test_log::test]
fn test_multi_prove_mixed_coset_offsets() {
    let proof_options_3 = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 3,
        coset_offset: 3,
        grinding_factor: 1,
    };
    let proof_options_7 = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 3,
        coset_offset: 7,
        grinding_factor: 1,
    };

    // Both AIRs have the same trace length and blowup, but different coset offsets.
    let mut trace_1 = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);
    let mut trace_2 = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };

    let air_1 = FibonacciAIR::<GoldilocksField>::new(&proof_options_3);
    let air_2 = FibonacciAIR::<GoldilocksField>::new(&proof_options_7);

    let air_trace_pairs: Vec<(
        &dyn AIR<
            Field = GoldilocksField,
            FieldExtension = GoldilocksField,
            PublicInputs = FibonacciPublicInputs<GoldilocksField>,
        >,
        &mut _,
        &_,
    )> = vec![
        (&air_1, &mut trace_1, &pub_inputs),
        (&air_2, &mut trace_2, &pub_inputs),
    ];

    let multi_proof = multi_prove_ram(
        air_trace_pairs,
        &mut DefaultTranscript::<GoldilocksField>::new(&[]),
    )
    .expect("proving should succeed");

    let airs: Vec<
        &dyn AIR<
            Field = GoldilocksField,
            FieldExtension = GoldilocksField,
            PublicInputs = FibonacciPublicInputs<GoldilocksField>,
        >,
    > = vec![&air_1, &air_2];

    assert!(
        Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<GoldilocksField>::new(&[]),
            &FieldElement::zero(),
        ),
        "verification should succeed when AIRs share (trace_length, blowup) but differ in coset_offset"
    );
}

/// Test that the domain cache deduplicates when multiple AIRs share all three key fields
/// `(trace_length, blowup, coset_offset)`. Asserts exactly one `Domain`/`LdeTwiddles`
/// construction for N identical AIRs and that the resulting proof still verifies.
#[test_log::test]
fn test_multi_prove_dedups_shared_domain_params() {
    domain_cache_stats::reset();

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 3,
        coset_offset: 3,
        grinding_factor: 1,
    };

    let mut trace_1 = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);
    let mut trace_2 = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);
    let mut trace_3 = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };

    let air_1 = FibonacciAIR::<GoldilocksField>::new(&proof_options);
    let air_2 = FibonacciAIR::<GoldilocksField>::new(&proof_options);
    let air_3 = FibonacciAIR::<GoldilocksField>::new(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<
            Field = GoldilocksField,
            FieldExtension = GoldilocksField,
            PublicInputs = FibonacciPublicInputs<GoldilocksField>,
        >,
        &mut _,
        &_,
    )> = vec![
        (&air_1, &mut trace_1, &pub_inputs),
        (&air_2, &mut trace_2, &pub_inputs),
        (&air_3, &mut trace_3, &pub_inputs),
    ];

    let multi_proof = multi_prove_ram(
        air_trace_pairs,
        &mut DefaultTranscript::<GoldilocksField>::new(&[]),
    )
    .expect("proving should succeed");

    let (hits, misses) = domain_cache_stats::get();
    assert_eq!(
        misses, 1,
        "only one Domain/LdeTwiddles must be constructed for 3 AIRs sharing domain params"
    );
    assert_eq!(
        hits, 2,
        "remaining 2 AIRs must hit the cache instead of reconstructing"
    );

    let airs: Vec<
        &dyn AIR<
            Field = GoldilocksField,
            FieldExtension = GoldilocksField,
            PublicInputs = FibonacciPublicInputs<GoldilocksField>,
        >,
    > = vec![&air_1, &air_2, &air_3];

    assert!(
        Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<GoldilocksField>::new(&[]),
            &FieldElement::zero(),
        ),
        "verification should succeed when AIRs share all domain parameters"
    );
}

/// Differential test for the DEEP composition polynomial "direct 2N" evaluation.
///
/// After this PR, `compute_deep_composition_poly_evaluations` evaluates the DEEP
/// polynomial directly at all 2N LDE points. The old path computed it at the N
/// trace-coset points and then extended via iFFT(N)+FFT(2N).
///
/// Both paths should produce the same values because `deep(X)` is a polynomial
/// of degree < N (the poles cancel by construction, since the numerators vanish
/// at the denominators' zeros). By uniqueness of polynomial interpolation, a
/// polynomial of degree < N is fully determined by its values on any N-point
/// subset — so extending from N matches evaluating directly at 2N.
///
/// This test constructs a synthetic scenario (known trace polys, composition
/// polys, OOD values, gammas), computes `deep(x)` at every LDE point two ways,
/// and asserts the results match exactly.
#[test]
fn test_deep_poly_direct_2n_matches_interpolate_fft_extend() {
    let n = 16usize;
    let blowup_factor = 2usize;
    let two_n = n * blowup_factor;

    let proof_options = ProofOptions {
        blowup_factor: blowup_factor as u8,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 0,
    };

    let air = QuadraticAIR::<GoldilocksField>::new(&proof_options);
    let domain = Domain::new(&air, n);

    // Trace polynomials (degree < N): two columns with deterministic coefficients.
    let num_trace_cols = 2usize;
    let trace_polys: Vec<Polynomial<Felt>> = (0..num_trace_cols)
        .map(|j| {
            let coeffs: Vec<Felt> = (0..n)
                .map(|i| Felt::from(((i + 1) * (j + 2) * 11 + 7) as u64))
                .collect();
            Polynomial::new(&coeffs)
        })
        .collect();

    // Composition poly parts (each of degree < N): two parts.
    let num_parts = 2usize;
    let h_polys: Vec<Polynomial<Felt>> = (0..num_parts)
        .map(|j| {
            let coeffs: Vec<Felt> = (0..n)
                .map(|i| Felt::from(((i + 3) * (j + 5) * 19 + 31) as u64))
                .collect();
            Polynomial::new(&coeffs)
        })
        .collect();

    // OOD evaluation point and the derived poles.
    let z = Felt::from(12345u64);
    let z_power = z.pow(num_parts);

    let num_eval_points = 2usize;
    let z_shifted: Vec<Felt> = (0..num_eval_points)
        .map(|k| domain.trace_primitive_root.pow(k) * &z)
        .collect();

    // OOD values: H_j(z^K) and t_j(z·w^k).
    let h_ood: Vec<Felt> = h_polys.iter().map(|h| h.evaluate(&z_power)).collect();
    let t_ood: Vec<Vec<Felt>> = trace_polys
        .iter()
        .map(|t| z_shifted.iter().map(|z_k| t.evaluate(z_k)).collect())
        .collect();

    // Random-ish gammas.
    let gamma_h: Vec<Felt> = (0..num_parts)
        .map(|j| Felt::from((j as u64 + 1) * 100))
        .collect();
    let gamma_t: Vec<Vec<Felt>> = (0..num_trace_cols)
        .map(|j| {
            (0..num_eval_points)
                .map(|k| Felt::from((((j + 1) * (k + 1)) as u64) * 200))
                .collect()
        })
        .collect();

    // Helper that computes deep(x) at a single point — same formula as the
    // production code, written here without the per-row optimizations.
    let compute_deep = |x: &Felt| -> Felt {
        let mut result = Felt::zero();
        // H terms
        for j in 0..num_parts {
            let numer = h_polys[j].evaluate(x) - &h_ood[j];
            let denom_inv = (x - &z_power).inv().expect("z^K not on coset");
            result += &gamma_h[j] * &numer * &denom_inv;
        }
        // Trace terms
        for (j, trace_poly) in trace_polys.iter().enumerate().take(num_trace_cols) {
            for k in 0..num_eval_points {
                let numer = trace_poly.evaluate(x) - &t_ood[j][k];
                let denom_inv = (x - &z_shifted[k]).inv().expect("z·w^k not on coset");
                result += &gamma_t[j][k] * &numer * &denom_inv;
            }
        }
        result
    };

    // Path A — direct evaluation at all 2N LDE points (the new path).
    let direct_2n: Vec<Felt> = domain
        .lde_roots_of_unity_coset
        .iter()
        .map(compute_deep)
        .collect();

    // Path B — evaluate at the N trace-coset points {g·ω^i} = lde_coset[i·bf],
    // interpolate via iFFT, then extend via FFT to all 2N LDE points.
    let trace_coset_evals: Vec<Felt> = (0..n)
        .map(|i| compute_deep(&domain.lde_roots_of_unity_coset[i * blowup_factor]))
        .collect();
    let deep_poly = Polynomial::interpolate_offset_fft(&trace_coset_evals, &domain.coset_offset)
        .expect("interpolation should succeed on trace-coset evaluations");
    let extended_2n =
        evaluate_polynomial_on_lde_domain(&deep_poly, blowup_factor, n, &domain.coset_offset)
            .expect("LDE extension should succeed");

    assert_eq!(direct_2n.len(), two_n);
    assert_eq!(extended_2n.len(), two_n);
    for i in 0..two_n {
        assert_eq!(
            direct_2n[i], extended_2n[i],
            "deep evaluation mismatch at LDE index {i}: direct-2N path diverges from \
             iFFT+FFT-extended path"
        );
    }
}

#[test]
fn commit_rows_bit_reversed_matches_commit_columns_bit_reversed() {
    type F = GoldilocksField;
    type FE = FieldElement<F>;

    for num_cols in [1usize, 3, 7] {
        for log_rows in [4usize, 6, 8] {
            let num_rows = 1usize << log_rows;

            let columns: Vec<Vec<FE>> = (0..num_cols)
                .map(|c| {
                    (0..num_rows)
                        .map(|r| FE::from((c * num_rows + r) as u64 * 6700417 + 1))
                        .collect()
                })
                .collect();

            // Row-major interleaving: data[row * num_cols + col] = columns[col][row].
            let mut row_major: Vec<FE> = Vec::with_capacity(num_rows * num_cols);
            for r in 0..num_rows {
                for col in &columns {
                    row_major.push(col[r]);
                }
            }

            let (_, root_col) = Prover::<F, F, ()>::commit_columns_bit_reversed(&columns)
                .expect("column-major commit must succeed");
            let (_, root_row) = Prover::<F, F, ()>::commit_rows_bit_reversed(&row_major, num_cols)
                .expect("row-major commit must succeed");

            assert_eq!(
                root_col, root_row,
                "commit root mismatch: num_cols={num_cols} log_rows={log_rows}"
            );
        }
    }
}

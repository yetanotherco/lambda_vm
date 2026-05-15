use crypto::fiat_shamir::default_transcript::DefaultTranscript;

use crate::{
    domain::{Domain, DomainConstants, QuotientDomain},
    examples::{
        quadratic_air::QuadraticAIR,
        simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs},
    },
    proof::options::ProofOptions,
    prover::{
        IsStarkProver, Prover, compute_chunks_deep_contribution, domain_cache_stats,
        evaluate_polynomial_on_lde_domain,
    },
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

    let multi_proof = Prover::multi_prove(
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

    let multi_proof = Prover::multi_prove(
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

/// Phase 1.2 sanity test for the chunks-based commitment migration.
///
/// Validates that `QuotientDomain::split_evals_interleaved` produces chunks
/// that correspond to the evaluations of a polynomial H(x) (of degree < d_max·N)
/// on the disjoint sub-cosets of the quotient domain.
///
/// Concretely: for a random polynomial H of degree < d_max·N, evaluated on a
/// quotient domain of size d_max·N, the interleaved split should yield
/// `num_chunks = next_pow2(d_max)` vectors of size N each, where each
/// `chunks[i][j]` equals `H(coset_offset · omega^(i + j·num_chunks))`.
///
/// This validates the split semantics before any prover code changes.
#[test]
fn quotient_domain_interleaved_split_matches_subcoset_evals() {
    // Setup a Domain with trace_length = 8 (so we have a small example).
    // We don't need a full AIR for this test; build the Domain manually.
    let trace_length: usize = 8;
    let blowup_factor: usize = 2;
    let coset_offset = Felt::from(3u64); // arbitrary non-trivial offset
    let root_order = trace_length.trailing_zeros();
    let trace_primitive_root =
        GoldilocksField::get_primitive_root_of_unity(root_order as u64).unwrap();
    let lde_root_order = (trace_length * blowup_factor).trailing_zeros();
    let lde_roots_of_unity_coset = math::fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset(
        lde_root_order as u64,
        trace_length * blowup_factor,
        &coset_offset,
    )
    .unwrap();
    let trace_roots_of_unity = math::fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset(
        root_order as u64,
        trace_length,
        &Felt::one(),
    )
    .unwrap();
    let domain = Domain::<GoldilocksField> {
        root_order,
        lde_roots_of_unity_coset,
        trace_primitive_root,
        trace_roots_of_unity,
        coset_offset: coset_offset.clone(),
        blowup_factor,
        interpolation_domain_size: trace_length,
    };

    // Try several d_max values to cover all branches:
    //   d_max=1 → num_chunks=1 (degenerate)
    //   d_max=2 → num_chunks=2
    //   d_max=3 → num_chunks=4 (next_power_of_two)
    for &d_max in &[1usize, 2, 3] {
        let qd = QuotientDomain::new(&domain, d_max);
        let num_chunks = qd.num_chunks;
        let size = qd.size;
        assert_eq!(num_chunks, d_max.next_power_of_two().max(1));
        assert_eq!(size, num_chunks * trace_length);

        // Build a random-looking polynomial H of degree size-1
        // (i.e., < num_chunks * trace_length).
        let h_coeffs: Vec<Felt> = (0..size).map(|i| Felt::from((7 * i + 13) as u64)).collect();
        let h_poly = Polynomial::new(&h_coeffs);

        // Evaluate H at every point of the quotient domain.
        let h_evals: Vec<Felt> = (0..size)
            .map(|i| h_poly.evaluate(qd.point_at(i)))
            .collect();

        // Split using the interleaved method.
        let chunks = qd.split_evals_interleaved(&h_evals);

        // Shape checks.
        assert_eq!(chunks.len(), num_chunks);
        for chunk in &chunks {
            assert_eq!(chunk.len(), trace_length);
        }

        // For each chunk_i, verify chunks[i][j] == H(coset_offset · omega^(i + j*num_chunks))
        // which is exactly point_at(i + j*num_chunks).
        for i in 0..num_chunks {
            for j in 0..trace_length {
                let global_idx = i + j * num_chunks;
                let expected = h_poly.evaluate(qd.point_at(global_idx));
                assert_eq!(
                    chunks[i][j], expected,
                    "chunk[{i}][{j}] does not match H evaluated at global index {global_idx} \
                     (d_max={d_max}, num_chunks={num_chunks})",
                );
            }
        }
    }
}

/// Phase 1.3 sanity test for the chunks-based commitment migration.
///
/// Validates the P3-style algebraic identity that the chunks verifier will use
/// to recover `H(z)` from the chunk openings `Q_i(z)`:
///
/// ```text
///   H(z) = sum_{i=0..K-1} zps[i] * Q_i(z)
/// ```
///
/// where `Q_i` is the polynomial of degree `< N` interpolating the i-th
/// interleaved chunk on its sub-coset, and `zps[i]` is the Lagrange-style
/// coefficient defined in `QuotientDomain::recompose_at`.
///
/// For each `d_max in {1, 2, 3}` we build a known `H` of degree `< K·N`,
/// evaluate it on the quotient domain, run the split + per-chunk
/// interpolation, then check `recompose_at` matches `H(z)` at several `z`.
/// `d_max = 1` is trivial (K = 1, zps is the empty product = 1), but is kept
/// to lock in the degenerate path.
#[test]
fn quotient_domain_recompose_at_matches_direct_h_eval() {
    let trace_length: usize = 8;
    let blowup_factor: usize = 2;
    let coset_offset = Felt::from(3u64);
    let root_order = trace_length.trailing_zeros();
    let trace_primitive_root =
        GoldilocksField::get_primitive_root_of_unity(root_order as u64).unwrap();
    let lde_root_order = (trace_length * blowup_factor).trailing_zeros();
    let lde_roots_of_unity_coset = math::fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset(
        lde_root_order as u64,
        trace_length * blowup_factor,
        &coset_offset,
    )
    .unwrap();
    let trace_roots_of_unity = math::fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset(
        root_order as u64,
        trace_length,
        &Felt::one(),
    )
    .unwrap();
    let domain = Domain::<GoldilocksField> {
        root_order,
        lde_roots_of_unity_coset,
        trace_primitive_root,
        trace_roots_of_unity,
        coset_offset: coset_offset.clone(),
        blowup_factor,
        interpolation_domain_size: trace_length,
    };

    for &d_max in &[1usize, 2, 3] {
        let qd = QuotientDomain::new(&domain, d_max);
        let num_chunks = qd.num_chunks;
        let size = qd.size;

        // Random-looking polynomial H of degree < K·N.
        let h_coeffs: Vec<Felt> = (0..size)
            .map(|i| Felt::from((11 * i + 17) as u64))
            .collect();
        let h_poly = Polynomial::new(&h_coeffs);

        // H on the quotient domain.
        let h_evals: Vec<Felt> = (0..size).map(|i| h_poly.evaluate(qd.point_at(i))).collect();

        // Interleaved split → per-chunk evaluations on sub-cosets.
        let chunks = qd.split_evals_interleaved(&h_evals);

        // Interpolate each chunk on its sub-coset to recover Q_i.
        let q_polys: Vec<Polynomial<Felt>> = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let (sub_offset, _) = qd.chunk_subdomain(i);
                Polynomial::interpolate_offset_fft::<GoldilocksField>(chunk, &sub_offset)
                    .expect("chunk interpolation should succeed on power-of-two-sized sub-coset")
            })
            .collect();

        // Probe several z values (different residues mod every sub-coset to avoid
        // accidental zero denominators inside the verifier identity).
        let zs = [
            Felt::from(12345u64),
            Felt::from(98765u64),
            Felt::from(42u64),
            Felt::from(0xdead_beefu64),
        ];
        for z in &zs {
            let q_at_z: Vec<Felt> = q_polys.iter().map(|q| q.evaluate(z)).collect();
            let h_z_direct = h_poly.evaluate(z);
            let h_z_recompose = qd.recompose_at(&q_at_z, z);
            assert_eq!(
                h_z_direct, h_z_recompose,
                "d_max={d_max} num_chunks={num_chunks}: recompose_at(chunk_evals, z) \
                 disagrees with direct H(z) at z={z:?}",
            );
        }
    }
}

/// Phase 1.4 round-trip test for the chunks-based commitment migration.
///
/// Validates `IsStarkProver::lde_and_commit_quotient_chunks` for `d_max ∈
/// {1, 2, 3}`. For each chunk index `i`, the kernel does
/// `iFFT(sub-coset) → Q_i → FFT(LDE)`. The test confirms two things:
///
/// 1. **LDE correctness**: the produced `chunk_lde[k]` equals
///    `Q_i(domain.lde_roots_of_unity_coset[k])`, where `Q_i` is the chunk
///    polynomial we interpolate independently inside the test.
/// 2. **End-to-end algebraic consistency**: applying `recompose_at` to the
///    per-chunk LDE values at every LDE point recovers `H(lde_point)`. This
///    ties together the Phase 1.2 split semantics, the Phase 1.3 Lagrange
///    identity, and the Phase 1.4 LDE+commit pipeline.
///
/// The kernel also produces a Merkle tree + root per chunk; we check the
/// returned vector has the expected shape but do not yet verify openings
/// (that's Phase 2 once we wire chunks through the prover's open path).
#[test]
fn lde_and_commit_quotient_chunks_round_trip() {
    let trace_length: usize = 8;
    let blowup_factor: usize = 2;
    let coset_offset = Felt::from(3u64);
    let root_order = trace_length.trailing_zeros();
    let trace_primitive_root =
        GoldilocksField::get_primitive_root_of_unity(root_order as u64).unwrap();
    let lde_root_order = (trace_length * blowup_factor).trailing_zeros();
    let lde_roots_of_unity_coset = math::fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset(
        lde_root_order as u64,
        trace_length * blowup_factor,
        &coset_offset,
    )
    .unwrap();
    let trace_roots_of_unity = math::fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset(
        root_order as u64,
        trace_length,
        &Felt::one(),
    )
    .unwrap();
    let domain = Domain::<GoldilocksField> {
        root_order,
        lde_roots_of_unity_coset,
        trace_primitive_root,
        trace_roots_of_unity,
        coset_offset: coset_offset.clone(),
        blowup_factor,
        interpolation_domain_size: trace_length,
    };

    for &d_max in &[1usize, 2, 3] {
        let qd = QuotientDomain::new(&domain, d_max);
        let num_chunks = qd.num_chunks;
        let size = qd.size;
        let lde_size = trace_length * blowup_factor;

        // Known H of degree < K·N, then evaluate on the quotient domain and split.
        let h_coeffs: Vec<Felt> = (0..size)
            .map(|i| Felt::from((11 * i + 17) as u64))
            .collect();
        let h_poly = Polynomial::new(&h_coeffs);
        let h_evals: Vec<Felt> = (0..size).map(|i| h_poly.evaluate(qd.point_at(i))).collect();
        let chunks = qd.split_evals_interleaved(&h_evals);

        // Run the kernel under test.
        let results = Prover::<GoldilocksField, GoldilocksField, ()>::lde_and_commit_quotient_chunks(
            &qd, &domain, &chunks,
        );

        assert_eq!(
            results.len(),
            num_chunks,
            "d_max={d_max}: kernel returned {} entries but num_chunks = {num_chunks}",
            results.len(),
        );

        // Reference Q_i polynomials (the test interpolates them independently of the kernel).
        let q_polys: Vec<Polynomial<Felt>> = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let (sub_offset, _) = qd.chunk_subdomain(i);
                Polynomial::interpolate_offset_fft::<GoldilocksField>(chunk, &sub_offset).unwrap()
            })
            .collect();

        // 1. LDE correctness per chunk.
        for (i, (chunk_lde, _tree, _root)) in results.iter().enumerate() {
            assert_eq!(
                chunk_lde.len(),
                lde_size,
                "d_max={d_max} chunk={i}: lde length {} != expected {lde_size}",
                chunk_lde.len(),
            );
            for k in 0..lde_size {
                let expected = q_polys[i].evaluate(&domain.lde_roots_of_unity_coset[k]);
                assert_eq!(
                    chunk_lde[k], expected,
                    "d_max={d_max} chunk={i}: chunk_lde[{k}] disagrees with Q_i evaluated \
                     at lde_roots_of_unity_coset[{k}]",
                );
            }
        }

        // 2. Algebraic consistency: recompose_at over the per-chunk LDE values
        //    recovers H at every LDE point.
        let chunk_ldes: Vec<&Vec<Felt>> = results.iter().map(|(lde, _, _)| lde).collect();
        for k in 0..lde_size {
            let lde_point = &domain.lde_roots_of_unity_coset[k];
            let chunk_evals_at_lde_k: Vec<Felt> =
                (0..num_chunks).map(|i| chunk_ldes[i][k].clone()).collect();
            let h_recomposed = qd.recompose_at(&chunk_evals_at_lde_k, lde_point);
            let h_direct = h_poly.evaluate(lde_point);
            assert_eq!(
                h_direct, h_recomposed,
                "d_max={d_max}: recompose_at over chunk_ldes mismatches H at lde index {k}",
            );
        }

        // 3. Per-chunk roots are well-formed (different chunks ⇒ generally different
        //    roots, since they commit to different polynomials). For num_chunks > 1
        //    assert that not all roots collapse to the same value.
        if num_chunks > 1 {
            let roots: Vec<_> = results.iter().map(|(_, _, root)| root.clone()).collect();
            let all_same = roots.iter().all(|r| *r == roots[0]);
            assert!(
                !all_same,
                "d_max={d_max}: all {num_chunks} chunk roots collapsed to the same value — \
                 chunks should commit to distinct polynomials",
            );
        }
    }
}

/// Phase 3.1 test for the chunks DEEP contribution.
///
/// For `d_max ∈ {1, 2, 3}` builds the chunk LDEs from a known `H`, runs
/// `compute_chunks_deep_contribution` over them at every LDE point, and
/// checks:
///
/// 1. **Pointwise correctness**: the result at LDE index `i` equals the naive
///    `sum_c gamma_c * (chunk_ldes[c][i] - chunk_ood_evals[c]) / (lde[i] - z)`.
/// 2. **Degree bound**: interpolating the result on the LDE coset yields a
///    polynomial of degree `< N`. The chunks-protocol DEEP contribution must
///    feed FRI a low-degree polynomial of the same shape as the single-H
///    path (`compute_deep_composition_poly_evaluations`).
#[test]
fn chunks_deep_contribution_is_degree_below_n() {
    let trace_length: usize = 8;
    let blowup_factor: usize = 2;
    let lde_size = trace_length * blowup_factor;
    let coset_offset = Felt::from(3u64);
    let root_order = trace_length.trailing_zeros();
    let trace_primitive_root =
        GoldilocksField::get_primitive_root_of_unity(root_order as u64).unwrap();
    let lde_root_order = (trace_length * blowup_factor).trailing_zeros();
    let lde_roots_of_unity_coset = math::fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset(
        lde_root_order as u64,
        trace_length * blowup_factor,
        &coset_offset,
    )
    .unwrap();
    let trace_roots_of_unity = math::fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset(
        root_order as u64,
        trace_length,
        &Felt::one(),
    )
    .unwrap();
    let domain = Domain::<GoldilocksField> {
        root_order,
        lde_roots_of_unity_coset,
        trace_primitive_root,
        trace_roots_of_unity,
        coset_offset: coset_offset.clone(),
        blowup_factor,
        interpolation_domain_size: trace_length,
    };

    // z chosen far enough from any LDE-coset element to avoid accidental coincidences.
    let z = Felt::from(0xdead_beefu64);

    for &d_max in &[1usize, 2, 3] {
        let qd = QuotientDomain::new(&domain, d_max);
        let num_chunks = qd.num_chunks;

        let h_coeffs: Vec<Felt> = (0..qd.size)
            .map(|i| Felt::from((11 * i + 17) as u64))
            .collect();
        let h_poly = Polynomial::new(&h_coeffs);
        let h_evals: Vec<Felt> = (0..qd.size).map(|i| h_poly.evaluate(qd.point_at(i))).collect();
        let chunks = qd.split_evals_interleaved(&h_evals);

        let results = Prover::<GoldilocksField, GoldilocksField, ()>::lde_and_commit_quotient_chunks(
            &qd, &domain, &chunks,
        );
        let chunk_ldes: Vec<Vec<Felt>> =
            results.iter().map(|(lde, _, _)| lde.clone()).collect();

        // Chunk OOD evaluations = Q_c(z).
        let chunk_ood: Vec<Felt> = chunks
            .iter()
            .enumerate()
            .map(|(c, chunk)| {
                let (sub_offset, _) = qd.chunk_subdomain(c);
                let q_c = Polynomial::interpolate_offset_fft::<GoldilocksField>(chunk, &sub_offset)
                    .unwrap();
                q_c.evaluate(&z)
            })
            .collect();

        // Random-looking gammas, one per chunk.
        let gammas: Vec<Felt> = (0..num_chunks)
            .map(|c| Felt::from((c as u64 + 1) * 31 + 7))
            .collect();

        let deep = compute_chunks_deep_contribution(
            &chunk_ldes,
            &chunk_ood,
            &z,
            &gammas,
            &domain.lde_roots_of_unity_coset,
        );
        assert_eq!(deep.len(), lde_size);

        // 1. Pointwise equality with naive reference.
        for i in 0..lde_size {
            let denom_inv = (&domain.lde_roots_of_unity_coset[i] - &z).inv().unwrap();
            let mut expected = Felt::zero();
            for c in 0..num_chunks {
                expected = expected + &gammas[c] * (&chunk_ldes[c][i] - &chunk_ood[c]) * &denom_inv;
            }
            assert_eq!(
                deep[i], expected,
                "d_max={d_max}: chunks DEEP contribution disagrees with naive formula at LDE \
                 index {i}",
            );
        }

        // 2. Degree bound: interpolating `deep` on the LDE coset yields a polynomial
        //    of degree < N. The chunks-protocol DEEP contribution must be of the
        //    same shape FRI consumes for the single-H path.
        let deep_poly =
            Polynomial::interpolate_offset_fft::<GoldilocksField>(&deep, &domain.coset_offset)
                .unwrap();
        let coeffs = deep_poly.coefficients();
        for (i, c) in coeffs.iter().enumerate().skip(trace_length) {
            assert_eq!(
                *c,
                Felt::zero(),
                "d_max={d_max}: DEEP contribution has non-zero coefficient at degree {i} \
                 (expected < N = {trace_length})",
            );
        }
    }
}

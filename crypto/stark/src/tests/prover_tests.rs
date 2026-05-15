use crypto::fiat_shamir::default_transcript::DefaultTranscript;

use crate::{
    domain::{Domain, DomainConstants, QuotientDomain},
    examples::{
        quadratic_air::QuadraticAIR,
        simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs},
    },
    proof::options::ProofOptions,
    config::BatchedMerkleTreeBackend,
    proof::quotient_chunks::QuotientChunksCommitments,
    prover::{
        IsStarkProver, Prover, ProvingError, compute_chunk_ood_evaluations,
        compute_chunks_deep_contribution, compute_trace_deep_contribution, domain_cache_stats,
        evaluate_polynomial_on_lde_domain, open_quotient_chunks_at_query,
    },
    table::Table,
    trace::{LDETraceTable, get_trace_evaluations, get_trace_evaluations_from_lde},
    traits::AIR,
    verifier::{IsStarkVerifier, Verifier},
};
use math::{
    fft::cpu::bit_reversing::reverse_index,
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

/// Phase 3.2 test for `open_quotient_chunks_at_query`.
///
/// For `d_max ∈ {1, 2, 3}` and a small set of query indices, this builds the
/// per-chunk Merkle commitments via `lde_and_commit_quotient_chunks`, opens
/// each chunk independently at the query position, and verifies:
///
/// 1. **Eval correctness**: the opening's `evaluations[0]` equals
///    `chunk_lde[reverse_index(2*index, 2N)]` and `evaluations_sym[0]` equals
///    `chunk_lde[reverse_index(2*index + 1, 2N)]`.
/// 2. **Path correctness**: the Merkle path verifies against the chunk root
///    using `[br_0, br_1]` as the leaf data (matching the leaf layout used
///    by `commit_composition_polynomial`).
/// 3. **Tamper detection**: flipping one of the leaf values makes Merkle
///    verification fail.
#[test]
fn open_quotient_chunks_at_query_paths_verify() {
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

    for &d_max in &[1usize, 2, 3] {
        let qd = QuotientDomain::new(&domain, d_max);

        let h_coeffs: Vec<Felt> = (0..qd.size)
            .map(|i| Felt::from((11 * i + 17) as u64))
            .collect();
        let h_poly = Polynomial::new(&h_coeffs);
        let h_evals: Vec<Felt> = (0..qd.size).map(|i| h_poly.evaluate(qd.point_at(i))).collect();
        let chunks = qd.split_evals_interleaved(&h_evals);
        let chunk_results = Prover::<GoldilocksField, GoldilocksField, ()>::lde_and_commit_quotient_chunks(
            &qd, &domain, &chunks,
        );

        // The Merkle tree has lde_size / 2 leaves (each pairs two LDE rows), so
        // FRI query positions range over 0..lde_size/2.
        let query_indices = [0usize, 1, 3, lde_size / 2 - 1];
        for &index in &query_indices {
            let openings = open_quotient_chunks_at_query(&chunk_results, index);
            assert_eq!(openings.len(), qd.num_chunks);

            for (c, opening) in openings.iter().enumerate() {
                let chunk_lde = &chunk_results[c].0;
                let root = &chunk_results[c].2;
                let br_0 = reverse_index(index * 2, lde_size as u64);
                let br_1 = reverse_index(index * 2 + 1, lde_size as u64);

                // 1. Eval correctness.
                assert_eq!(opening.evaluations, vec![chunk_lde[br_0].clone()]);
                assert_eq!(opening.evaluations_sym, vec![chunk_lde[br_1].clone()]);

                // 2. Path correctness — leaf layout is [br_0, br_1].
                let leaf_data = vec![chunk_lde[br_0].clone(), chunk_lde[br_1].clone()];
                assert!(
                    opening
                        .proof
                        .verify::<BatchedMerkleTreeBackend<GoldilocksField>>(
                            root, index, &leaf_data
                        ),
                    "d_max={d_max} chunk={c} index={index}: Merkle path must verify against \
                     the chunk root using [br_0, br_1] leaf data",
                );
                assert!(
                    opening
                        .proof_sym
                        .verify::<BatchedMerkleTreeBackend<GoldilocksField>>(
                            root, index, &leaf_data
                        ),
                    "d_max={d_max} chunk={c} index={index}: proof_sym must verify same leaf",
                );

                // 3. Tamper detection: flipping br_0 breaks the proof.
                let mut tampered = leaf_data.clone();
                tampered[0] = &tampered[0] + Felt::one();
                assert!(
                    !opening
                        .proof
                        .verify::<BatchedMerkleTreeBackend<GoldilocksField>>(
                            root, index, &tampered
                        ),
                    "d_max={d_max} chunk={c} index={index}: tampered leaf must not verify",
                );
            }
        }
    }
}

/// Phase 3.3 — end-to-end synthesis test for the chunks primitives.
///
/// Wires together every building block we've added so far:
///   * Phase 1.2 / 1.4: `lde_and_commit_quotient_chunks` produces
///     per-chunk LDE + Merkle commitments,
///   * Phase 1.3 / 2: `QuotientChunksCommitments::verify_at_ood` reconstructs
///     `H(z)` from chunk OOD evaluations and matches the directly-evaluated
///     `H(z)`,
///   * Phase 3.1: `compute_chunks_deep_contribution` produces the DEEP
///     polynomial evaluations on the LDE,
///   * Phase 3.2: `open_quotient_chunks_at_query` opens each chunk's tree
///     at FRI query positions.
///
/// The load-bearing consistency: at every queried LDE row, the prover-side
/// chunks-DEEP evaluation must equal what a verifier reconstructs from the
/// chunk openings alone, i.e.,
///
/// ```text
///   deep[br_x] == sum_c gamma_c * (Q_c(br_x) - Q_c(z)) / (lde[br_x] - z)
/// ```
///
/// for `br_x ∈ {br_0, br_1}` at every query index. This is the cross-check
/// FRI uses at query time to bind the committed DEEP polynomial to the
/// per-chunk Merkle commitments.
#[test]
fn chunks_protocol_primitives_end_to_end_synthesis() {
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

    // OOD point chosen far from any LDE-coset element.
    let z = Felt::from(0xdead_beefu64);

    for &d_max in &[1usize, 2, 3] {
        let qd = QuotientDomain::new(&domain, d_max);
        let num_chunks = qd.num_chunks;

        // Known H of degree < K·N.
        let h_coeffs: Vec<Felt> = (0..qd.size)
            .map(|i| Felt::from((11 * i + 17) as u64))
            .collect();
        let h_poly = Polynomial::new(&h_coeffs);
        let h_evals: Vec<Felt> = (0..qd.size).map(|i| h_poly.evaluate(qd.point_at(i))).collect();
        let chunks = qd.split_evals_interleaved(&h_evals);

        // Phase 1.4 — per-chunk LDE + Merkle commit.
        let chunk_results = Prover::<GoldilocksField, GoldilocksField, ()>::lde_and_commit_quotient_chunks(
            &qd, &domain, &chunks,
        );
        let chunk_ldes: Vec<Vec<Felt>> =
            chunk_results.iter().map(|(lde, _, _)| lde.clone()).collect();
        let chunk_roots: Vec<_> =
            chunk_results.iter().map(|(_, _, r)| r.clone()).collect();

        // Phase 1.3 / Phase 2 — chunk OOD evaluations and recompose check.
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
        let commitments = QuotientChunksCommitments {
            chunk_roots: chunk_roots.clone(),
            chunk_ood_evaluations: chunk_ood.clone(),
        };
        let h_at_z = h_poly.evaluate(&z);
        assert!(
            commitments.verify_at_ood(&qd, &z, &h_at_z),
            "d_max={d_max}: chunk OOD evaluations must verify against H(z)",
        );

        // Phase 3.1 — chunks DEEP contribution on the LDE.
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

        // Phase 3.2 + cross-check — at each FRI query position, opening the chunks
        // and applying the verifier's reconstruction formula must reproduce
        // `deep[br_x]` exactly.
        let query_indices = [0usize, 1, 3, lde_size / 2 - 1];
        for &index in &query_indices {
            let openings = open_quotient_chunks_at_query(&chunk_results, index);
            assert_eq!(openings.len(), num_chunks);

            let br_0 = reverse_index(index * 2, lde_size as u64);
            let br_1 = reverse_index(index * 2 + 1, lde_size as u64);
            let x_br_0 = &domain.lde_roots_of_unity_coset[br_0];
            let x_br_1 = &domain.lde_roots_of_unity_coset[br_1];
            let inv_denom_0 = (x_br_0 - &z).inv().unwrap();
            let inv_denom_1 = (x_br_1 - &z).inv().unwrap();

            let mut reconstructed_0 = Felt::zero();
            let mut reconstructed_1 = Felt::zero();
            for c in 0..num_chunks {
                reconstructed_0 = reconstructed_0
                    + &gammas[c] * (&openings[c].evaluations[0] - &chunk_ood[c]) * &inv_denom_0;
                reconstructed_1 = reconstructed_1
                    + &gammas[c] * (&openings[c].evaluations_sym[0] - &chunk_ood[c]) * &inv_denom_1;
            }
            assert_eq!(
                deep[br_0], reconstructed_0,
                "d_max={d_max} index={index}: prover-side DEEP at br_0 disagrees with \
                 verifier reconstruction from chunk openings",
            );
            assert_eq!(
                deep[br_1], reconstructed_1,
                "d_max={d_max} index={index}: prover-side DEEP at br_1 disagrees with \
                 verifier reconstruction from chunk openings",
            );
        }
    }
}

/// Phase 4.1 test for `round_2_chunks_kernel`.
///
/// The kernel takes constraint evaluations on the LDE coset (i.e., the
/// evaluator's output) and produces a `Round2Chunks`. The new logic, on top
/// of the already-tested `lde_and_commit_quotient_chunks`, is:
///
/// - **d_max == 1**: quotient domain has size N (half the LDE). The kernel
///   should reduce by stride-2 (every other LDE element).
/// - **d_max == 2**: quotient domain has size 2N (identical to LDE). The
///   kernel should pass evaluations through unchanged.
/// - **d_max >= 3**: the standard LDE doesn't have enough samples; the
///   kernel must return [`ProvingError::WrongParameter`] until the
///   evaluator is extended in a follow-up commit.
///
/// In each supported case the test reconstructs `H(z)` from the resulting
/// chunks via `QuotientDomain::recompose_at` and asserts equality with
/// `H(z)` evaluated directly from the known polynomial used to build the
/// LDE input.
#[test]
fn round_2_chunks_kernel_matches_direct_h_eval_and_rejects_d_max_3() {
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

    let z = Felt::from(0xdead_beefu64);

    for &d_max in &[1usize, 2] {
        let qd_size = d_max.next_power_of_two().max(1) * trace_length;
        // Known H of degree < d_max·N — well within LDE 2N for d_max in {1, 2}.
        let h_coeffs: Vec<Felt> = (0..qd_size)
            .map(|i| Felt::from((11 * i + 17) as u64))
            .collect();
        let h_poly = Polynomial::new(&h_coeffs);
        // Evaluate H on the standard LDE coset (this is what the evaluator would emit).
        let constraint_evals_lde: Vec<Felt> = (0..lde_size)
            .map(|i| h_poly.evaluate(&domain.lde_roots_of_unity_coset[i]))
            .collect();

        let r2 = Prover::<GoldilocksField, GoldilocksField, ()>::round_2_chunks_kernel(
            constraint_evals_lde,
            &domain,
            d_max,
        )
        .expect("round_2_chunks_kernel should succeed for d_max in {1, 2}");

        let qd = QuotientDomain::new(&domain, d_max);
        assert_eq!(r2.num_chunks, qd.num_chunks);
        assert_eq!(r2.chunk_lde_evaluations.len(), qd.num_chunks);
        assert_eq!(r2.chunk_merkle_trees.len(), qd.num_chunks);
        assert_eq!(r2.chunk_roots.len(), qd.num_chunks);
        for chunk_lde in &r2.chunk_lde_evaluations {
            assert_eq!(chunk_lde.len(), lde_size);
        }

        // H(z) reconstructed from chunk evals at z (Q_c interpolated from the
        // chunk LDE) must match direct H(z).
        let chunk_at_z: Vec<Felt> = r2
            .chunk_lde_evaluations
            .iter()
            .map(|chunk_lde| {
                Polynomial::interpolate_offset_fft::<GoldilocksField>(
                    chunk_lde,
                    &domain.coset_offset,
                )
                .unwrap()
                .evaluate(&z)
            })
            .collect();
        let h_z_recomposed = qd.recompose_at(&chunk_at_z, &z);
        let h_z_direct = h_poly.evaluate(&z);
        assert_eq!(
            h_z_recomposed, h_z_direct,
            "d_max={d_max}: recompose_at over Round2Chunks should reproduce H(z)",
        );
    }

    // d_max=3 must return an error — quotient_domain.size = 4N > lde_size = 2N.
    let dummy_lde: Vec<Felt> = vec![Felt::zero(); lde_size];
    let err = Prover::<GoldilocksField, GoldilocksField, ()>::round_2_chunks_kernel(
        dummy_lde, &domain, 3usize,
    );
    match err {
        Err(ProvingError::WrongParameter(_)) => {}
        Err(other) => panic!(
            "d_max=3 should currently return WrongParameter; got Err({other:?})",
        ),
        Ok(_) => panic!("d_max=3 should currently return WrongParameter; got Ok(_)"),
    }
}

/// Phase 4.2 test for `compute_chunk_ood_evaluations`.
///
/// The function is the chunks-protocol replacement for the
/// composition-parts-OOD block of `round_3_evaluate_polynomials_in_out_of_domain_element`:
/// barycentric-evaluate each chunk polynomial `Q_c` at `z` directly (no
/// `z^num_parts` power).
///
/// The test builds `Round2Chunks` via `round_2_chunks_kernel` from a known
/// `H`, runs `compute_chunk_ood_evaluations` at a fixed `z`, then folds the
/// resulting `Q_c(z)` values back to `H(z)` via `QuotientDomain::recompose_at`
/// and asserts the result equals `H(z)` evaluated directly. This is the
/// load-bearing identity for the chunks verifier's step-2 check.
#[test]
fn chunk_ood_evaluations_recompose_to_h() {
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

    let z = Felt::from(0xdead_beefu64);

    for &d_max in &[1usize, 2] {
        let qd_size = d_max.next_power_of_two().max(1) * trace_length;
        let h_coeffs: Vec<Felt> = (0..qd_size)
            .map(|i| Felt::from((11 * i + 17) as u64))
            .collect();
        let h_poly = Polynomial::new(&h_coeffs);
        let constraint_evals_lde: Vec<Felt> = (0..lde_size)
            .map(|i| h_poly.evaluate(&domain.lde_roots_of_unity_coset[i]))
            .collect();

        let r2 = Prover::<GoldilocksField, GoldilocksField, ()>::round_2_chunks_kernel(
            constraint_evals_lde,
            &domain,
            d_max,
        )
        .unwrap();

        let chunk_ood = compute_chunk_ood_evaluations(&r2.chunk_lde_evaluations, &domain, &z);
        assert_eq!(chunk_ood.len(), r2.num_chunks);

        let qd = QuotientDomain::new(&domain, d_max);
        let h_z_recomposed = qd.recompose_at(&chunk_ood, &z);
        let h_z_direct = h_poly.evaluate(&z);
        assert_eq!(
            h_z_recomposed, h_z_direct,
            "d_max={d_max}: recompose_at over compute_chunk_ood_evaluations output must \
             reproduce H(z)",
        );
    }
}

/// Phase 4.3a test for `compute_trace_deep_contribution`.
///
/// The function returns the trace-only piece of the DEEP composition
/// polynomial on the LDE. Each per-(column, eval-point) summand is
/// `(t_j(x) - t_j(z·w^k)) / (x - z·w^k)`, a polynomial of degree `< N - 1`
/// (the pole at `z·w^k` cancels since the numerator vanishes there). The
/// random linear combination must therefore have degree `< N`.
///
/// The test builds 3 trace polynomials of degree `< N`, evaluates them on
/// the standard LDE coset to construct an `LDETraceTable`, computes their
/// OOD values at `z` and `z·w` (one transition offset for `next_row`), runs
/// the helper, and asserts:
///
/// 1. **Pointwise correctness**: result at LDE row `i` equals the naive
///    `sum_{j,k} gamma_{j,k} * (t_j_lde[i] - t_j(z·w^k)) / (lde[i] - z·w^k)`.
/// 2. **Degree bound**: interpolating the result on the LDE coset yields a
///    polynomial of degree `< N` — FRI's expected input shape.
#[test]
fn trace_deep_contribution_is_degree_below_n() {
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
        trace_primitive_root: trace_primitive_root.clone(),
        trace_roots_of_unity,
        coset_offset: coset_offset.clone(),
        blowup_factor,
        interpolation_domain_size: trace_length,
    };

    let num_main_cols = 3usize;
    let trace_polys: Vec<Polynomial<Felt>> = (0..num_main_cols)
        .map(|j| {
            let coeffs: Vec<Felt> = (0..trace_length)
                .map(|i| Felt::from(((i + 1) * (j + 2) * 13 + 5) as u64))
                .collect();
            Polynomial::new(&coeffs)
        })
        .collect();

    // LDE each trace poly on the standard LDE coset.
    let lde_columns: Vec<Vec<Felt>> = trace_polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, trace_length, &coset_offset)
                .unwrap()
        })
        .collect();
    let lde_trace =
        LDETraceTable::from_columns(lde_columns.clone(), Vec::<Vec<Felt>>::new(), 1, blowup_factor);

    // Two transition offsets per column: at z (k=0) and z*w (k=1) — covers next_row.
    let num_eval_points = 2usize;
    let z = Felt::from(0xdead_beefu64);
    let z_shifted = [z.clone(), &trace_primitive_root * &z];

    // OOD trace evaluations: trace_ood[k][j] = trace_polys[j].evaluate(z_shifted[k]).
    let trace_ood_rows: Vec<Vec<Felt>> = (0..num_eval_points)
        .map(|k| {
            (0..num_main_cols)
                .map(|j| trace_polys[j].evaluate(&z_shifted[k]))
                .collect()
        })
        .collect();
    let trace_ood_evaluations = Table::from_columns(
        (0..num_main_cols)
            .map(|j| (0..num_eval_points).map(|k| trace_ood_rows[k][j].clone()).collect())
            .collect(),
    );

    // Random-looking gammas: trace_terms_gammas[j][k].
    let trace_terms_gammas: Vec<Vec<Felt>> = (0..num_main_cols)
        .map(|j| {
            (0..num_eval_points)
                .map(|k| Felt::from(((j + 1) * (k + 1) * 41 + 11) as u64))
                .collect()
        })
        .collect();

    let result = compute_trace_deep_contribution(
        &lde_trace,
        &trace_ood_evaluations,
        &z,
        &domain,
        &trace_primitive_root,
        &trace_terms_gammas,
    );
    assert_eq!(result.len(), lde_size);

    // 1. Pointwise naive reference.
    for i in 0..lde_size {
        let x_i = &domain.lde_roots_of_unity_coset[i];
        let mut expected = Felt::zero();
        for k in 0..num_eval_points {
            let denom_inv = (x_i - &z_shifted[k]).inv().unwrap();
            for j in 0..num_main_cols {
                expected = expected
                    + &trace_terms_gammas[j][k]
                        * (&lde_columns[j][i] - &trace_ood_rows[k][j])
                        * &denom_inv;
            }
        }
        assert_eq!(
            result[i], expected,
            "trace DEEP contribution disagrees with naive formula at LDE index {i}",
        );
    }

    // 2. Degree bound: result interpolated on the LDE coset has degree < N.
    let result_poly =
        Polynomial::interpolate_offset_fft::<GoldilocksField>(&result, &domain.coset_offset)
            .unwrap();
    let coeffs = result_poly.coefficients();
    for (i, c) in coeffs.iter().enumerate().skip(trace_length) {
        assert_eq!(
            *c,
            Felt::zero(),
            "trace DEEP contribution has non-zero coefficient at degree {i} (expected < N)",
        );
    }
}

/// Phase 4.3b test for `compute_deep_composition_poly_evaluations_chunks`.
///
/// The chunks-protocol DEEP composition polynomial = chunks contribution +
/// trace contribution. Each piece is degree `< N` (validated by the Phase 3.1
/// and 4.3a tests), so their sum must also be degree `< N` — that is the
/// FRI input shape.
///
/// Builds a synthetic AIR-like scenario: trace polynomials of degree `< N`
/// and a separate `H` polynomial of degree `< d_max·N` representing the
/// (otherwise opaque) constraint-evaluation result. Threads everything
/// through `round_2_chunks_kernel` + `compute_chunk_ood_evaluations` to
/// produce realistic `Round2Chunks` / `Round3Chunks` inputs, then asserts
/// the combiner's output has degree `< N` on the LDE coset.
#[test]
fn compute_deep_composition_poly_evaluations_chunks_is_degree_below_n() {
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
        trace_primitive_root: trace_primitive_root.clone(),
        trace_roots_of_unity,
        coset_offset: coset_offset.clone(),
        blowup_factor,
        interpolation_domain_size: trace_length,
    };

    let z = Felt::from(0xdead_beefu64);

    for &d_max in &[1usize, 2] {
        // === Trace side ===
        let num_main_cols = 2usize;
        let trace_polys: Vec<Polynomial<Felt>> = (0..num_main_cols)
            .map(|j| {
                let coeffs: Vec<Felt> = (0..trace_length)
                    .map(|i| Felt::from(((i + 1) * (j + 2) * 13 + 5) as u64))
                    .collect();
                Polynomial::new(&coeffs)
            })
            .collect();
        let lde_columns: Vec<Vec<Felt>> = trace_polys
            .iter()
            .map(|poly| {
                evaluate_polynomial_on_lde_domain(poly, blowup_factor, trace_length, &coset_offset)
                    .unwrap()
            })
            .collect();
        let lde_trace =
            LDETraceTable::from_columns(lde_columns, Vec::<Vec<Felt>>::new(), 1, blowup_factor);

        let num_eval_points = 2usize;
        let z_shifted = [z.clone(), &trace_primitive_root * &z];
        let trace_ood_rows: Vec<Vec<Felt>> = (0..num_eval_points)
            .map(|k| {
                (0..num_main_cols)
                    .map(|j| trace_polys[j].evaluate(&z_shifted[k]))
                    .collect()
            })
            .collect();
        let trace_ood_evaluations = Table::from_columns(
            (0..num_main_cols)
                .map(|j| {
                    (0..num_eval_points)
                        .map(|k| trace_ood_rows[k][j].clone())
                        .collect()
                })
                .collect(),
        );

        // === Chunks side ===
        let qd_size = d_max.next_power_of_two().max(1) * trace_length;
        let h_coeffs: Vec<Felt> = (0..qd_size)
            .map(|i| Felt::from((11 * i + 17) as u64))
            .collect();
        let h_poly = Polynomial::new(&h_coeffs);
        let constraint_evals_lde: Vec<Felt> = (0..lde_size)
            .map(|i| h_poly.evaluate(&domain.lde_roots_of_unity_coset[i]))
            .collect();
        let round_2_chunks =
            Prover::<GoldilocksField, GoldilocksField, ()>::round_2_chunks_kernel(
                constraint_evals_lde,
                &domain,
                d_max,
            )
            .unwrap();
        let chunk_ood = compute_chunk_ood_evaluations(
            &round_2_chunks.chunk_lde_evaluations,
            &domain,
            &z,
        );
        let round_3_chunks = crate::prover::Round3Chunks {
            trace_ood_evaluations,
            chunk_ood_evaluations: chunk_ood,
        };

        // === Random-looking challenge scalars ===
        let chunk_gammas: Vec<Felt> = (0..round_2_chunks.num_chunks)
            .map(|c| Felt::from((c as u64 + 1) * 31 + 7))
            .collect();
        let trace_terms_gammas: Vec<Vec<Felt>> = (0..num_main_cols)
            .map(|j| {
                (0..num_eval_points)
                    .map(|k| Felt::from(((j + 1) * (k + 1) * 41 + 11) as u64))
                    .collect()
            })
            .collect();

        let deep = Prover::<GoldilocksField, GoldilocksField, ()>::compute_deep_composition_poly_evaluations_chunks(
            &lde_trace,
            &round_2_chunks,
            &round_3_chunks,
            &z,
            &domain,
            &trace_primitive_root,
            &chunk_gammas,
            &trace_terms_gammas,
        );
        assert_eq!(deep.len(), lde_size);

        let deep_poly =
            Polynomial::interpolate_offset_fft::<GoldilocksField>(&deep, &domain.coset_offset)
                .unwrap();
        let coeffs = deep_poly.coefficients();
        for (i, c) in coeffs.iter().enumerate().skip(trace_length) {
            assert_eq!(
                *c,
                Felt::zero(),
                "d_max={d_max}: chunks DEEP polynomial has non-zero coefficient at degree {i} \
                 (expected < N = {trace_length})",
            );
        }
    }
}

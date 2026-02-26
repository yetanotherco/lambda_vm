use crate::{
    domain::Domain,
    examples::{
        quadratic_air::QuadraticAIR,
        simple_fibonacci,
    },
    proof::options::ProofOptions,
    prover::{evaluate_polynomial_on_lde_domain, IsStarkProver, Prover},
    trace::{get_trace_evaluations, get_trace_evaluations_from_lde, LDETraceTable},
    traits::AIR,
};
use math::{
    field::{
        element::FieldElement, fields::fft_friendly::u64_goldilocks::GoldilocksField,
        traits::IsFFTField,
    },
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

    // Barycentric evaluation (new path)
    let result =
        get_trace_evaluations_from_lde(&lde_trace, &domain, &z, &frame_offsets, step_size);

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
    let new_result = Prover::<GoldilocksField, GoldilocksField, ()>::decompose_and_extend_d2::<
        GoldilocksField,
    >(&constraint_evaluations, &domain);

    assert_eq!(new_result.len(), 2);
    assert_eq!(new_result[0].len(), original[0].len());
    assert_eq!(new_result[1].len(), original[1].len());
    for i in 0..new_result[0].len() {
        assert_eq!(new_result[0][i], original[0][i], "H₀ mismatch at index {i}");
        assert_eq!(new_result[1][i], original[1][i], "H₁ mismatch at index {i}");
    }
}

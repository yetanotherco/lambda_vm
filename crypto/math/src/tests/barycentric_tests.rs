use crate::fft::roots_of_unity::get_powers_of_primitive_root_coset;
use crate::field::element::FieldElement;
use crate::field::goldilocks::GoldilocksField;
use crate::polynomial::{
    Polynomial, barycentric_inv_denoms, interpolate_coset_eval_ext_with_g_n_inv,
};

type FE = FieldElement<GoldilocksField>;

/// Build a polynomial of degree < n with deterministic coefficients.
fn test_poly(n: usize) -> Polynomial<FE> {
    let coeffs: Vec<FE> = (0..n).map(|i| FE::from((i * 7 + 3) as u64)).collect();
    Polynomial::new(&coeffs)
}

#[test]
fn barycentric_matches_horner_simple() {
    let n = 8usize;
    let coset_offset = FE::from(3u64);
    let poly = test_poly(n);

    let root_order = n.trailing_zeros() as u64;
    let coset_points = get_powers_of_primitive_root_coset(root_order, n, &coset_offset).unwrap();
    let evaluations: Vec<FE> = coset_points.iter().map(|p| poly.evaluate(p)).collect();

    let z = FE::from(42u64);
    let expected = poly.evaluate(&z);

    let z_pow_n = z.pow(n);
    let g_pow_n = coset_offset.pow(n);
    let n_inv = FE::from(n as u64).inv().unwrap();
    let g_n_inv = g_pow_n.inv().unwrap();
    let inv_denoms = barycentric_inv_denoms(&z, &coset_points);
    let result = interpolate_coset_eval_ext_with_g_n_inv(
        &z_pow_n,
        &g_pow_n,
        &n_inv,
        &g_n_inv,
        &coset_points,
        &evaluations,
        &inv_denoms,
    );

    assert_eq!(result, expected);
}

#[test]
fn barycentric_matches_horner_various_sizes() {
    for log_n in 2..=8 {
        let n = 1 << log_n;
        let coset_offset = FE::from(7u64);
        let poly = test_poly(n);

        let root_order = n.trailing_zeros() as u64;
        let coset_points =
            get_powers_of_primitive_root_coset(root_order, n, &coset_offset).unwrap();
        let evaluations: Vec<FE> = coset_points.iter().map(|p| poly.evaluate(p)).collect();

        let z = FE::from(123u64);
        let expected = poly.evaluate(&z);

        let z_pow_n = z.pow(n);
        let g_pow_n = coset_offset.pow(n);
        let n_inv = FE::from(n as u64).inv().unwrap();
        let g_n_inv = g_pow_n.inv().unwrap();
        let inv_denoms = barycentric_inv_denoms(&z, &coset_points);
        let result = interpolate_coset_eval_ext_with_g_n_inv(
            &z_pow_n,
            &g_pow_n,
            &n_inv,
            &g_n_inv,
            &coset_points,
            &evaluations,
            &inv_denoms,
        );

        assert_eq!(result, expected, "failed for n={n}");
    }
}

#[test]
fn barycentric_ext_matches_horner() {
    let n = 16usize;
    let coset_offset = FE::from(5u64);
    let poly = test_poly(n);

    let root_order = n.trailing_zeros() as u64;
    let coset_points = get_powers_of_primitive_root_coset(root_order, n, &coset_offset).unwrap();
    let evaluations: Vec<FE> = coset_points.iter().map(|p| poly.evaluate(p)).collect();

    let z = FE::from(999u64);
    let expected = poly.evaluate(&z);

    let z_pow_n = z.pow(n);
    let g_pow_n = coset_offset.pow(n);
    let n_inv = FE::from(n as u64).inv().unwrap();
    let g_n_inv = g_pow_n.inv().unwrap();
    let inv_denoms = barycentric_inv_denoms(&z, &coset_points);
    let result = interpolate_coset_eval_ext_with_g_n_inv(
        &z_pow_n,
        &g_pow_n,
        &n_inv,
        &g_n_inv,
        &coset_points,
        &evaluations,
        &inv_denoms,
    );

    assert_eq!(result, expected);
}

#[test]
fn barycentric_from_lde_stride() {
    // Simulate what the prover does: given an LDE of size n*bf, extract stride-bf
    // evaluations and use barycentric to evaluate at an OOD point
    let n = 8usize;
    let bf = 4usize;
    let coset_offset = FE::from(3u64);
    let poly = test_poly(n);

    let lde_root_order = (n * bf).trailing_zeros() as u64;
    let lde_coset =
        get_powers_of_primitive_root_coset(lde_root_order, n * bf, &coset_offset).unwrap();
    let lde_evals: Vec<FE> = lde_coset.iter().map(|p| poly.evaluate(p)).collect();

    // Extract trace-size coset (stride = bf)
    let trace_coset: Vec<FE> = (0..n).map(|i| lde_coset[i * bf]).collect();
    let trace_evals: Vec<FE> = (0..n).map(|i| lde_evals[i * bf]).collect();

    let z = FE::from(42u64);
    let expected = poly.evaluate(&z);

    let z_pow_n = z.pow(n);
    let g_pow_n = coset_offset.pow(n);
    let n_inv = FE::from(n as u64).inv().unwrap();
    let g_n_inv = g_pow_n.inv().unwrap();
    let inv_denoms = barycentric_inv_denoms(&z, &trace_coset);
    let result = interpolate_coset_eval_ext_with_g_n_inv(
        &z_pow_n,
        &g_pow_n,
        &n_inv,
        &g_n_inv,
        &trace_coset,
        &trace_evals,
        &inv_denoms,
    );

    assert_eq!(result, expected);
}

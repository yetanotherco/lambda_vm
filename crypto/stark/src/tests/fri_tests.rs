use crate::fri::fri_functions::{compute_coset_twiddles_inv, fold_evaluations_in_place};
use math::fft::bit_reversing::in_place_bit_reverse_permute;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;
use math::polynomial::Polynomial;

type FE = FieldElement<GoldilocksField>;

/// FRI polynomial folding: computes P_even(x) + beta * P_odd(x)
/// where P(x) = P_even(x^2) + x * P_odd(x^2)
fn fold_polynomial<F>(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>>
where
    F: IsField,
{
    let coefficients = poly.coefficients();
    if coefficients.is_empty() {
        return Polynomial::new(&[]);
    }

    let mut result = Vec::with_capacity(coefficients.len().div_ceil(2));

    for chunk in coefficients.chunks(2) {
        let folded = if chunk.len() == 2 {
            &chunk[0] + &(&chunk[1] * beta)
        } else {
            chunk[0].clone()
        };
        result.push(folded);
    }

    Polynomial::new(&result)
}

#[test]
fn test_fold_power_of_2() {
    let p0 = Polynomial::new(&[
        FE::new(3),
        FE::new(1),
        FE::new(2),
        FE::new(7),
        FE::new(3),
        FE::new(5),
        FE::new(4),
        FE::new(2),
    ]);
    let beta = FE::new(4);
    let p1 = fold_polynomial(&p0, &beta);
    assert_eq!(
        p1,
        Polynomial::new(&[FE::new(7), FE::new(30), FE::new(23), FE::new(12)])
    );

    let gamma = FE::new(3);
    let p2 = fold_polynomial(&p1, &gamma);
    assert_eq!(p2, Polynomial::new(&[FE::new(97), FE::new(59)]));

    let delta = FE::new(2);
    let p3 = fold_polynomial(&p2, &delta);
    assert_eq!(p3, Polynomial::new(&[FE::new(215)]));
    assert_eq!(p3.degree(), 0);
}

#[test]
fn test_fold_size_2() {
    let p2 = Polynomial::new(&[FE::new(10), FE::new(20)]);
    let beta = FE::new(3);
    let result = fold_polynomial(&p2, &beta);
    assert_eq!(result, Polynomial::new(&[FE::new(70)]));
}

/// Reference coefficient-form FRI fold with doubling: 2 * (P_even(x) + beta * P_odd(x))
fn fold_polynomial_doubled_reference<F: IsField>(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>> {
    let coefficients = poly.coefficients();
    if coefficients.is_empty() {
        return Polynomial::new(&[]);
    }
    let mut result = Vec::with_capacity(coefficients.len().div_ceil(2));
    for chunk in coefficients.chunks(2) {
        let folded = if chunk.len() == 2 {
            (&chunk[0] + &(&chunk[1] * beta)).double()
        } else {
            chunk[0].double()
        };
        result.push(folded);
    }
    Polynomial::new(&result)
}

#[test]
fn test_eval_fold_matches_coeff_fold() {
    let coset_offset = FE::from(3u64);
    let beta = FE::from(7u64);

    // Use a degree-7 polynomial (8 coefficients)
    let poly = Polynomial::new(&[
        FE::from(1u64),
        FE::from(2u64),
        FE::from(3u64),
        FE::from(4u64),
        FE::from(5u64),
        FE::from(6u64),
        FE::from(7u64),
        FE::from(8u64),
    ]);
    let n = 8usize;

    // Evaluate polynomial on coset via FFT
    let evals_fft =
        Polynomial::evaluate_offset_fft::<GoldilocksField>(&poly, 1, None, &coset_offset).unwrap();

    // Path A: reference coeff fold -> FFT -> bit-reverse
    let folded_poly = fold_polynomial_doubled_reference(&poly, &beta);
    let squared_offset = coset_offset.square();
    let mut path_a_evals =
        Polynomial::evaluate_offset_fft::<GoldilocksField>(&folded_poly, 1, None, &squared_offset)
            .unwrap();
    in_place_bit_reverse_permute(&mut path_a_evals);

    // Path B: FFT -> bit-reverse -> eval fold (live fold_evaluations_in_place)
    let mut path_b_evals = evals_fft;
    in_place_bit_reverse_permute(&mut path_b_evals);
    let inv_twiddles = compute_coset_twiddles_inv::<GoldilocksField>(&coset_offset, n);
    fold_evaluations_in_place(&mut path_b_evals, &beta, &inv_twiddles);

    assert_eq!(path_a_evals, path_b_evals);
}

/// FRI commit-phase early-termination roundtrip.
///
/// Builds a known low-degree FRI codeword, runs `commit_phase_from_evaluations`
/// with `blowup_log = 1`, `final_poly_log_degree = 2`, and checks:
///   * the emitted final polynomial has exactly `2^final_poly_log_degree` coeffs,
///   * the number of committed FRI layers equals `total_folds - 1`,
///   * folding each queried evaluation through the committed layers reaches the
///     reconstructed terminal codeword at the query's terminal-layer position.
#[test]
fn test_commit_phase_early_termination_roundtrip() {
    use crate::config::KeccakStarkHash;
    use crate::fri::fri_functions::update_twiddles_in_place;
    use crate::fri::terminal::terminal_codeword_from_coeffs;
    use crate::fri::{commit_phase_from_evaluations, query_phase};
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use math::fft::bit_reversing::reverse_index;
    use math::field::traits::IsFFTField;

    type F = GoldilocksField;

    let blowup_log: u32 = 1;
    let final_poly_log_degree: u32 = 2;
    let initial_len = 64usize;
    let root_order = initial_len.trailing_zeros(); // 6
    let total_folds = (root_order - (blowup_log + final_poly_log_degree)) as usize; // 3
    let num_committed = total_folds - 1; // 2

    let offset = FE::from(3u64);

    // Degree-<32 polynomial; with blowup 2 its terminal poly has degree < 2^2 = 4,
    // so the emitted 2^2 coefficients capture it exactly.
    let coeffs_in: Vec<FE> = (1u64..=32).map(FE::new).collect();
    let poly = Polynomial::new(&coeffs_in);

    // Coset LDE (blowup 2) -> natural order -> bit-reverse -> FRI-order codeword.
    let mut codeword =
        Polynomial::evaluate_offset_fft::<F>(&poly, 2, Some(32), &offset).expect("LDE FFT");
    in_place_bit_reverse_permute(&mut codeword);
    assert_eq!(codeword.len(), initial_len);

    // ---- Commit phase with early termination ----
    let mut transcript = DefaultTranscript::<F>::new(&[]);
    let inv_twiddles =
        crate::fri::fri_functions::compute_coset_twiddles_inv::<F>(&offset, initial_len);
    let (final_poly_coeffs, fri_layers) = commit_phase_from_evaluations::<F, F, _, KeccakStarkHash>(
        codeword.clone(),
        &mut transcript,
        &offset,
        initial_len,
        blowup_log,
        final_poly_log_degree,
        &inv_twiddles,
    );

    assert_eq!(
        final_poly_coeffs.len(),
        1 << final_poly_log_degree,
        "final poly must have 2^k coefficients"
    );
    assert_eq!(
        fri_layers.len(),
        num_committed,
        "committed layers must equal total_folds - 1"
    );

    // query_phase must still work against the committed layers.
    let iotas = vec![0usize, 1, 5, 17, 30];
    let _decommitments = query_phase::<F, KeccakStarkHash>(&fri_layers, &iotas);

    // ---- Reconstruct terminal codeword from the emitted coefficients ----
    let terminal_len = (1usize << blowup_log) << final_poly_log_degree; // 8
    let terminal_offset = offset.pow(1u64 << total_folds); // offset^(2^3)
    let terminal_codeword =
        terminal_codeword_from_coeffs::<F, F>(&final_poly_coeffs, &terminal_offset, terminal_len);
    assert_eq!(terminal_codeword.len(), terminal_len);

    // Re-derive the prover's folding challenges by replaying the transcript.
    let mut replay = DefaultTranscript::<F>::new(&[]);
    let mut zetas: Vec<FE> = Vec::with_capacity(total_folds);
    for layer in &fri_layers {
        zetas.push(replay.sample_field_element());
        replay.append_bytes(&layer.merkle_tree.root);
    }
    zetas.push(replay.sample_field_element()); // final-fold challenge
    assert_eq!(zetas.len(), total_folds);

    // Strong check: folding the whole codeword with those challenges reproduces
    // the reconstructed terminal codeword.
    let mut refold = codeword.clone();
    let mut inv_tw = compute_coset_twiddles_inv::<F>(&offset, initial_len);
    for zeta in zetas.iter().take(total_folds) {
        fold_evaluations_in_place(&mut refold, zeta, &inv_tw);
        update_twiddles_in_place(&mut inv_tw);
    }
    assert_eq!(
        refold, terminal_codeword,
        "full re-fold must match reconstructed terminal codeword"
    );

    // Per-query check: replicate the verifier's fold path and land on
    // terminal_codeword[index] at the terminal-layer position.
    let omega = F::get_primitive_root_of_unity(root_order as u64).expect("root of unity");
    for &iota in &iotas {
        // p0(nu) and p0(-nu) live at FRI-order positions 2*iota and 2*iota+1.
        let p0 = codeword[2 * iota];
        let p0_sym = codeword[2 * iota + 1];
        // nu = offset * omega^reverse_index(2*iota, initial_len)
        let nu = &offset * omega.pow(reverse_index(2 * iota, initial_len as u64) as u64);
        let nu_inv = nu.inv().expect("evaluation point is non-zero");

        // Fold layer 0 -> 1 using the first challenge.
        let mut v = (&p0 + &p0_sym) + &nu_inv * &zetas[0] * (&p0 - &p0_sym);
        let mut index = iota;
        let mut ep_inv = nu_inv.square(); // nu^{-2} for the first committed layer
        for (i, layer) in fri_layers.iter().enumerate() {
            let sym = layer.evaluation[index ^ 1];
            v = (&v + &sym) + &ep_inv * &zetas[i + 1] * (&v - &sym);
            index >>= 1;
            ep_inv = ep_inv.square();
        }
        assert_eq!(
            v, terminal_codeword[index],
            "query {iota}: folded value must equal terminal_codeword[{index}]"
        );
    }
}

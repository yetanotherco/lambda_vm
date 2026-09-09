use math::fft::bit_reversing::in_place_bit_reverse_permute;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::polynomial::Polynomial;

use crate::fri::terminal::{coeffs_from_terminal_codeword, terminal_codeword_from_coeffs};

type F = GoldilocksField;
type FE = FieldElement<F>;

/// Roundtrip test: a degree-<8 polynomial survives
///   coeffs -> codeword (FRI bit-reversed) -> coeffs_from_terminal_codeword
///   and
///   recovered_coeffs -> terminal_codeword_from_coeffs -> original codeword.
#[test]
fn test_terminal_roundtrip() {
    // k=3: poly has 8 coefficients, degree < 8.
    // blowup=2: terminal codeword length = 8*2 = 16.
    let final_poly_log_degree: u32 = 3;
    let coeffs: Vec<FE> = (1u64..=8).map(FE::new).collect();
    let offset = FE::new(3);

    // Build the reference FRI-order codeword:
    //   evaluate_offset_fft returns natural order -> bit-reverse -> FRI order.
    let poly = Polynomial::new(&coeffs);
    let mut codeword = Polynomial::evaluate_offset_fft::<F>(&poly, 2, Some(8), &offset)
        .expect("evaluate_offset_fft failed");
    in_place_bit_reverse_permute(&mut codeword);
    assert_eq!(codeword.len(), 16);

    // --- prover direction ---
    let recovered_coeffs =
        coeffs_from_terminal_codeword::<F, F>(&codeword, &offset, final_poly_log_degree);
    assert_eq!(
        recovered_coeffs, coeffs,
        "coeffs_from_terminal_codeword did not recover the original coefficients"
    );

    // --- verifier direction ---
    let rebuilt_codeword = terminal_codeword_from_coeffs::<F, F>(&recovered_coeffs, &offset, 16);
    assert_eq!(
        rebuilt_codeword, codeword,
        "terminal_codeword_from_coeffs did not rebuild the original codeword"
    );
}

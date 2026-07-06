//! Conversion helpers between a FRI terminal codeword and the coefficients of
//! the low-degree polynomial it encodes.
//!
//! These are pure, self-contained helpers — no transcript, no FRI logic.
//! They are used by the prover (`commit_phase_from_evaluations`) and verifier FRI step.

use math::fft::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::polynomial::Polynomial;

/// Prover side: given a FRI terminal codeword in **bit-reversed** order,
/// recover the `2^final_poly_log_degree` coefficients of the underlying
/// low-degree polynomial.
///
/// The codeword is a coset evaluation of a polynomial of degree less than
/// `2^final_poly_log_degree` on the coset `terminal_offset·⟨ω⟩` of size
/// `blowup·2^k`.
///
/// Algorithm:
/// 1. Bit-reverse permute to convert from FRI order to natural (DFT) order.
/// 2. Decimate: extract the size-`2^k` sub-coset
///    `terminal_offset·⟨ω^blowup⟩` = every `blowup`-th natural-order point.
/// 3. Coset iFFT on the small (`2^k`-point) sub-domain — a `blowup×`-smaller
///    transform that recovers the `2^k` coefficients directly (no oversized
///    transform and no wasteful truncation).
pub(crate) fn coeffs_from_terminal_codeword<F, E>(
    codeword_bitrev: &[FieldElement<E>],
    terminal_offset: &FieldElement<F>,
    final_poly_log_degree: u32,
) -> Vec<FieldElement<E>>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
{
    // A degree-<2^k poly is determined by 2^k points: the size-2^k sub-coset
    // terminal_offset*<w^blowup> = every `blowup`-th natural-order evaluation,
    // i.e. natural-order index m*blowup for m in 0..2^k. The codeword is in
    // bit-reversed order, so gather those points straight from it via
    // reverse_index — no full-codeword clone or O(n) permute (only 2^k of the
    // blowup*2^k evaluations are ever read).
    let len = codeword_bitrev.len();
    let keep = 1usize << final_poly_log_degree;
    let blowup = len / keep;
    let sub_coset: Vec<FieldElement<E>> = (0..keep)
        .map(|m| codeword_bitrev[reverse_index(m * blowup, len as u64)].clone())
        .collect();

    // Coset iFFT on the small domain -> the 2^k coefficients directly (no oversized trim).
    let poly = Polynomial::interpolate_offset_fft::<F>(&sub_coset, terminal_offset)
        .expect("terminal sub-coset must have power-of-two length and non-zero offset");

    // Pad with zeros only if interpolation dropped trailing-zero coeffs, so the
    // proof always carries exactly 2^k coefficients (the verifier length-checks).
    let mut coeffs = poly.coefficients().to_vec();
    coeffs.resize(keep, FieldElement::<E>::zero());
    coeffs
}

/// Verifier side: given `2^k` coefficients of the low-degree polynomial,
/// reconstruct the full FRI terminal codeword in **bit-reversed** order.
///
/// Algorithm:
/// 1. FFT (coset): evaluate the polynomial on the full coset of size
///    `codeword_len` with shift `terminal_offset` to get natural order.
/// 2. Bit-reverse permute to convert natural order to FRI order.
///
/// # Panics
///
/// Panics if any of the following preconditions are violated:
/// - `coeffs` is non-empty,
/// - `coeffs.len()` is a power of two,
/// - `codeword_len` is a power of two,
/// - `coeffs.len() <= codeword_len`, and
/// - `codeword_len` is divisible by `coeffs.len()`.
///
/// In the normal verifier flow these conditions are guaranteed by the
/// final-polynomial length check that the verifier performs before calling
/// this helper, so the assert should never fire in production.
pub(crate) fn terminal_codeword_from_coeffs<F, E>(
    coeffs: &[FieldElement<E>],
    terminal_offset: &FieldElement<F>,
    codeword_len: usize,
) -> Vec<FieldElement<E>>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
{
    assert!(
        !coeffs.is_empty()
            && coeffs.len().is_power_of_two()
            && codeword_len.is_power_of_two()
            && coeffs.len() <= codeword_len
            && codeword_len.is_multiple_of(coeffs.len()),
        "terminal_codeword_from_coeffs: coeffs.len() ({}) must be a non-zero power of two dividing codeword_len ({}); the verifier must length-check coeffs before calling",
        coeffs.len(),
        codeword_len,
    );

    let poly = Polynomial::new(coeffs);
    let blowup = codeword_len / coeffs.len();

    // Step 1: coset FFT to get natural-order evaluations.
    let mut natural =
        Polynomial::evaluate_offset_fft::<F>(&poly, blowup, Some(coeffs.len()), terminal_offset)
            .expect("terminal coset size must be a power of two within the field's two-adicity");

    // Step 2: convert natural order to bit-reversed (FRI) order.
    in_place_bit_reverse_permute(&mut natural);
    natural
}

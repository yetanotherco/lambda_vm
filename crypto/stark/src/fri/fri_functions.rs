use math::fft::{
    bit_reversing::in_place_bit_reverse_permute, roots_of_unity::get_powers_of_primitive_root_coset,
};
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};
/// Evaluation-form FRI fold: given evaluations in bit-reversed order where
/// consecutive pairs (2j, 2j+1) are conjugates (p(x_j), p(-x_j)), compute
/// the folded evaluations: (lo + hi) + inv_twiddle[j] * zeta * (lo - hi)
/// = 2 * (p_even(x_j²) + zeta * p_odd(x_j²))
///
/// After folding, the N/2 results are evaluations on the squared coset
/// in bit-reversed order, preserving conjugate pairing for the next fold.
pub(crate) fn fold_evaluations_in_place<F: IsSubFieldOf<E>, E: IsField>(
    evals: &mut Vec<FieldElement<E>>,
    zeta: &FieldElement<E>,
    inv_twiddles: &[FieldElement<F>],
) {
    let half = evals.len() / 2;
    for j in 0..half {
        let lo = &evals[2 * j];
        let hi = &evals[2 * j + 1];
        let sum = lo + hi;
        let diff = lo - hi;
        evals[j] = &sum + &(&inv_twiddles[j] * &(zeta * &diff));
    }
    evals.truncate(half);
}

/// Compute inverse twiddle factors for evaluation-form FRI folding.
///
/// For a coset of size N with offset g, the twiddle factors are 1/x_j where
/// x_j are the coset points at even bit-reversed positions. Specifically:
/// generate g·w^i for i=0..N/2 (half the coset points), bit-reverse with
/// (logN-1) bits, then batch-invert.
pub(crate) fn compute_coset_twiddles_inv<F: IsFFTField>(
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> Vec<FieldElement<F>> {
    let half = domain_size / 2;
    let order = domain_size.trailing_zeros() as u64;
    let mut points = get_powers_of_primitive_root_coset(order, half, coset_offset).unwrap();
    in_place_bit_reverse_permute(&mut points);
    // Sequential: called from `Domain::fri_inv_twiddles`'s OnceLock init —
    // parallel inversion inside a lazy-init cell can deadlock the rayon pool.
    FieldElement::inplace_batch_inverse_sequential(&mut points).unwrap();
    points
}

/// Update inverse twiddle factors for the next FRI layer.
///
/// Between levels: new_tw[j'] = tw[2j']² (take even-indexed, square).
/// This corresponds to the squared coset offset and halved domain.
pub(crate) fn update_twiddles_in_place<F: IsField>(twiddles: &mut Vec<FieldElement<F>>) {
    let new_len = twiddles.len() / 2;
    for j in 0..new_len {
        twiddles[j] = twiddles[2 * j].square();
    }
    twiddles.truncate(new_len);
}

use math::fft::cpu::{
    bit_reversing::in_place_bit_reverse_permute, roots_of_unity::get_powers_of_primitive_root_coset,
};
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};
use math::polynomial::Polynomial;

/// In-place FRI polynomial folding with fused doubling: 2 * (P_even(x) + beta * P_odd(x))
///
/// This modifies the polynomial in place, avoiding memory allocation.
/// The polynomial degree is halved after this operation.
///
/// Note: This is the coefficient-form fold, retained for tests and reference.
/// Production FRI now uses `fold_evaluations_in_place` (evaluation-form).
#[allow(unused)]
pub fn fold_polynomial_doubled_inplace<F>(
    poly: &mut Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) where
    F: IsField,
{
    let coefficients = &mut poly.coefficients;
    if coefficients.is_empty() {
        return;
    }

    let new_len = coefficients.len().div_ceil(2);

    // Fold in place: process pairs and write results back to the beginning
    for i in 0..new_len {
        let idx = i * 2;
        let folded = if idx + 1 < coefficients.len() {
            (&coefficients[idx] + &(&coefficients[idx + 1] * beta)).double()
        } else {
            coefficients[idx].double()
        };
        coefficients[i] = folded;
    }

    // Truncate to the new length
    coefficients.truncate(new_len);
}

/// Evaluation-form FRI fold: given evaluations in bit-reversed order where
/// consecutive pairs (2j, 2j+1) are conjugates (p(x_j), p(-x_j)), compute
/// the folded evaluations: (lo + hi) + inv_twiddle[j] * zeta * (lo - hi)
/// = 2 * (p_even(x_j²) + zeta * p_odd(x_j²))
///
/// After folding, the N/2 results are evaluations on the squared coset
/// in bit-reversed order, preserving conjugate pairing for the next fold.
pub fn fold_evaluations_in_place<F: IsSubFieldOf<E>, E: IsField>(
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
pub fn compute_coset_twiddles_inv<F: IsFFTField>(
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> Vec<FieldElement<F>> {
    let half = domain_size / 2;
    let order = domain_size.trailing_zeros() as u64;
    let mut points = get_powers_of_primitive_root_coset(order, half, coset_offset).unwrap();
    in_place_bit_reverse_permute(&mut points);
    FieldElement::inplace_batch_inverse(&mut points).unwrap();
    points
}

/// Update inverse twiddle factors for the next FRI layer.
///
/// Between levels: new_tw[j'] = tw[2j']² (take even-indexed, square).
/// This corresponds to the squared coset offset and halved domain.
pub fn update_twiddles_in_place<F: IsField>(twiddles: &mut Vec<FieldElement<F>>) {
    let new_len = twiddles.len() / 2;
    for j in 0..new_len {
        twiddles[j] = twiddles[2 * j].square();
    }
    twiddles.truncate(new_len);
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::fft::cpu::bit_reversing::reverse_index;
    use math::field::element::FieldElement;
    use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
    use math::field::traits::IsFFTField;

    type GFE = FieldElement<GoldilocksField>;

    /// Verifies that the verifier's domain-based twiddle computation matches
    /// the prover's sequential fold for multi-fold (log_arity > 1).
    #[test]
    fn test_verifier_multifold_matches_prover() {
        // 16-element evaluation array, log_arity=2 (fold 2x per round)
        let domain_size = 16usize;
        let coset_offset = GFE::from(7u64);
        let zeta = GFE::from(13u64);

        // Create arbitrary evaluations
        let evals: Vec<GFE> = (0..domain_size)
            .map(|i| GFE::from((i * 3 + 5) as u64))
            .collect();

        // === Prover path: fold 2x using sequential fold_evaluations_in_place ===
        let mut prover_evals = evals.clone();
        let mut inv_twiddles = compute_coset_twiddles_inv(&coset_offset, domain_size);
        let mut challenge = zeta.clone();

        fold_evaluations_in_place(&mut prover_evals, &challenge, &inv_twiddles);
        update_twiddles_in_place(&mut inv_twiddles);
        challenge = challenge.square();
        fold_evaluations_in_place(&mut prover_evals, &challenge, &inv_twiddles);

        // prover_evals now has 4 elements (domain_size / 4)

        // === Verifier path: for each group of 4, compute twiddles from domain and fold ===
        let log_arity = 2usize;
        let group_size = 1usize << log_arity;
        let sub_offset_init = coset_offset.clone();
        let sub_domain_log_size_init = domain_size.trailing_zeros();

        for group_idx in 0..domain_size / group_size {
            let group_start = group_idx * group_size;
            let mut group_evals: Vec<GFE> = evals[group_start..group_start + group_size].to_vec();

            let mut sub_offset = sub_offset_init.clone();
            let mut sub_domain_log_size = sub_domain_log_size_init;
            let mut local_start = group_start;
            let mut ch = zeta.clone();

            for _ in 0..log_arity {
                let half = group_evals.len() / 2;
                let sub_root =
                    GoldilocksField::get_primitive_root_of_unity(sub_domain_log_size as u64)
                        .unwrap();
                let tw_bits = sub_domain_log_size - 1;
                let tw_domain_size = 1u64 << tw_bits;

                let mut coset_pts: Vec<GFE> = (0..half)
                    .map(|j| {
                        let gpi = local_start / 2 + j;
                        let natural_idx = reverse_index(gpi, tw_domain_size as u64);
                        &sub_offset * sub_root.pow(natural_idx)
                    })
                    .collect();
                GFE::inplace_batch_inverse(&mut coset_pts).unwrap();

                let mut new_evals = Vec::with_capacity(half);
                for j in 0..half {
                    let lo = &group_evals[2 * j];
                    let hi = &group_evals[2 * j + 1];
                    let sum = lo + hi;
                    let diff = lo - hi;
                    new_evals.push(&sum + &(&coset_pts[j] * &(&ch * &diff)));
                }

                group_evals = new_evals;
                sub_offset = sub_offset.square();
                sub_domain_log_size -= 1;
                ch = ch.square();
                local_start /= 2;
            }

            assert_eq!(
                group_evals.len(),
                1,
                "After log_arity folds, group should collapse to 1 element"
            );
            assert_eq!(
                group_evals[0], prover_evals[group_idx],
                "Group {group_idx}: verifier fold result doesn't match prover"
            );
        }
    }
}

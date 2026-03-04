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
pub fn fold_evaluations_in_place<F: IsSubFieldOf<E> + 'static, E: IsField + 'static>(
    evals: &mut Vec<FieldElement<E>>,
    zeta: &FieldElement<E>,
    inv_twiddles: &[FieldElement<F>],
) {
    // Packed fast path for Goldilocks + D3 extension
    #[cfg(feature = "parallel")]
    {
        use std::any::TypeId;
        use math::field::fields::fft_friendly::extensions_goldilocks::Degree3GoldilocksExtensionField as D3Ext;
        use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField as GF;

        if TypeId::of::<E>() == TypeId::of::<D3Ext>()
            && TypeId::of::<F>() == TypeId::of::<GF>()
        {
            // SAFETY: TypeId check guarantees type identity, FieldElement is repr(transparent)
            let evals_d3: &mut Vec<FieldElement<D3Ext>> = unsafe {
                &mut *(evals as *mut Vec<FieldElement<E>> as *mut Vec<FieldElement<D3Ext>>)
            };
            let zeta_d3: &FieldElement<D3Ext> = unsafe {
                &*(zeta as *const FieldElement<E> as *const FieldElement<D3Ext>)
            };
            let tw_gf: &[FieldElement<GF>] = unsafe {
                &*(inv_twiddles as *const [FieldElement<F>] as *const [FieldElement<GF>])
            };
            fold_evaluations_packed(evals_d3, zeta_d3, tw_gf);
            return;
        }
    }

    // Scalar fallback
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

/// Packed FRI fold using PackedFp3<PackedGoldilocks> to process WIDTH pairs simultaneously.
///
/// Each iteration loads WIDTH (lo, hi) pairs via gather, computes the fold formula
/// using packed Fp3 arithmetic, and scatters results back. The zeta challenge is
/// broadcast once and reused for all chunks.
#[cfg(feature = "parallel")]
fn fold_evaluations_packed(
    evals: &mut Vec<
        FieldElement<
            math::field::fields::fft_friendly::extensions_goldilocks::Degree3GoldilocksExtensionField,
        >,
    >,
    zeta: &FieldElement<
        math::field::fields::fft_friendly::extensions_goldilocks::Degree3GoldilocksExtensionField,
    >,
    inv_twiddles: &[FieldElement<
        math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField,
    >],
) {
    use math::field::fields::fft_friendly::u64_goldilocks_packed::{
        fp3::PackedFp3, PackedGoldilocks,
    };
    use math::field::packed::PackedField;

    let half = evals.len() / 2;
    let width = PackedGoldilocks::WIDTH;
    let zeta_packed = PackedFp3::<PackedGoldilocks>::broadcast(zeta);

    let packed_count = half / width;
    let remainder_start = packed_count * width;

    for chunk in 0..packed_count {
        let j = chunk * width;

        // Gather WIDTH lo values from stride-2 positions
        let lo = PackedFp3::from_fn(|i| {
            let v = evals[2 * (j + i)].value();
            [v[0].clone(), v[1].clone(), v[2].clone()]
        });

        // Gather WIDTH hi values from stride-2 positions
        let hi = PackedFp3::from_fn(|i| {
            let v = evals[2 * (j + i) + 1].value();
            [v[0].clone(), v[1].clone(), v[2].clone()]
        });

        // Load WIDTH contiguous base-field twiddles
        let tw = PackedGoldilocks::from_fn(|i| inv_twiddles[j + i].clone());

        // fold: sum + tw * (zeta * diff)
        let sum = lo + hi;
        let diff = lo - hi;
        let result = sum + (zeta_packed * diff).mul_scalar(tw);

        // Scatter WIDTH results back
        for i in 0..width {
            evals[j + i] = result.get_lane(i);
        }
    }

    // Scalar remainder
    for j in remainder_start..half {
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

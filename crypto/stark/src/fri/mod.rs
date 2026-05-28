pub mod fri_commitment;
pub mod fri_decommit;
pub(crate) mod fri_functions;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;

use crate::config::{FriLayerMerkleTree, FriLayerMerkleTreeBackend};

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::{
    compute_coset_twiddles_inv, fold_evaluations_in_place, update_twiddles_in_place,
};

/// Linearly combine N same-size evaluation vectors into one, using
/// successive powers of `delta_fri`:
///
/// `out[i] = sources[0][i] + delta_fri * sources[1][i] + delta_fri^2 *
/// sources[2][i] + ...`
///
/// This is the mathematical primitive behind Phase D batched FRI: a
/// bucket's chip-DEEP-LDEs are combined into one polynomial whose
/// low-degree-ness implies each summand's. FRI then commits + opens
/// just that combined polynomial.
///
/// Empty `sources` is a usage bug — caller must pre-filter.
/// All `sources[i]` must share the same length; debug-asserted.
pub fn linear_combine_evaluations<E: IsField>(
    sources: &[&[FieldElement<E>]],
    delta_fri: &FieldElement<E>,
) -> Vec<FieldElement<E>> {
    debug_assert!(
        !sources.is_empty(),
        "linear_combine_evaluations: caller must supply at least one source"
    );
    let n = sources[0].len();
    debug_assert!(
        sources.iter().all(|s| s.len() == n),
        "linear_combine_evaluations: all source vectors must share length"
    );

    if sources.len() == 1 {
        // Singleton bucket: combining one polynomial is the identity.
        return sources[0].to_vec();
    }

    let mut out = sources[0].to_vec();
    let mut coeff = delta_fri.clone();
    for src in &sources[1..] {
        for (o, s) in out.iter_mut().zip(src.iter()) {
            *o = &*o + &coeff * s;
        }
        coeff = &coeff * delta_fri;
    }
    out
}

/// FRI commit phase from pre-computed bit-reversed evaluations, skipping the
/// initial FFT. Use this when the caller already has the evaluation vector
/// (e.g. from a fused LDE pipeline).
pub fn commit_phase_from_evaluations<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    number_layers: usize,
    mut evals: Vec<FieldElement<E>>,
    transcript: &mut impl IsStarkTranscript<E, F>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> (
    FieldElement<E>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>>,
)
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    // Inverse twiddle factors for evaluation-form folding.
    let mut inv_twiddles = compute_coset_twiddles_inv(coset_offset, domain_size);

    // The loop commits `number_layers - 1` folded layers; the final fold below
    // produces the (uncommitted) last value.
    let num_committed_layers = number_layers.saturating_sub(1);
    let mut fri_layer_list = Vec::with_capacity(num_committed_layers);

    for _ in 0..num_committed_layers {
        // <<<< Receive challenge 𝜁ₖ₋₁
        let zeta = transcript.sample_field_element();

        // Fold evaluations in-place (no FFT needed).
        fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);

        // Build the Merkle tree from consecutive pairs.
        let leaves: Vec<[FieldElement<E>; 2]> = evals
            .chunks_exact(2)
            .map(|chunk| [chunk[0].clone(), chunk[1].clone()])
            .collect();
        let merkle_tree = FriLayerMerkleTree::build(&leaves)
            .expect("FRI commit: Merkle tree construction must succeed");
        let root = merkle_tree.root;
        fri_layer_list.push(FriLayer::new(&evals, merkle_tree));

        // >>>> Send commitment: [pₖ]
        transcript.append_bytes(&root);

        // Update twiddles for the next level.
        update_twiddles_in_place(&mut inv_twiddles);
    }

    // <<<< Receive challenge: 𝜁ₙ₋₁
    let zeta = transcript.sample_field_element();

    // Final fold.
    fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);

    let last_value = evals
        .first()
        .expect("FRI evals are non-empty after folding")
        .clone();

    // >>>> Send value: pₙ
    transcript.append_field_element(&last_value);

    (last_value, fri_layer_list)
}

pub fn query_phase<F: IsField>(
    fri_layers: &[FriLayer<F, FriLayerMerkleTreeBackend<F>>],
    iotas: &[usize],
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    if !fri_layers.is_empty() {
        let num_layers = fri_layers.len();
        iotas
            .iter()
            .map(|iota_s| {
                let mut layers_evaluations_sym = Vec::with_capacity(num_layers);
                let mut layers_auth_paths = Vec::with_capacity(num_layers);

                let mut index = *iota_s;
                for layer in fri_layers {
                    // symmetric element
                    let evaluation_sym = layer.evaluation[index ^ 1].clone();
                    let auth_path_sym = layer.merkle_tree.get_proof_by_pos(index >> 1).unwrap();
                    layers_evaluations_sym.push(evaluation_sym);
                    layers_auth_paths.push(auth_path_sym);

                    index >>= 1;
                }

                FriDecommitment {
                    layers_auth_paths,
                    layers_evaluations_sym,
                }
            })
            .collect()
    } else {
        // For 0 FRI layers (small traces), return empty decommitments for each query.
        // The verifier still needs one decommitment entry per query, even if the
        // FRI layer data is empty.
        iotas
            .iter()
            .map(|_| FriDecommitment {
                layers_auth_paths: vec![],
                layers_evaluations_sym: vec![],
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField;

    type FE = FieldElement<GoldilocksField>;

    #[test]
    fn linear_combine_singleton_is_identity() {
        let v = vec![FE::from(7u64), FE::from(11u64), FE::from(13u64), FE::from(17u64)];
        let delta = FE::from(99u64);
        let out = linear_combine_evaluations(&[&v[..]], &delta);
        assert_eq!(out, v);
    }

    #[test]
    fn linear_combine_two_sources_uses_horner_in_delta() {
        // out[i] = a[i] + delta * b[i]
        let a = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64)];
        let b = vec![FE::from(10u64), FE::from(20u64), FE::from(30u64)];
        let delta = FE::from(5u64);
        let out = linear_combine_evaluations(&[&a[..], &b[..]], &delta);
        let expected: Vec<FE> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| x + &delta * y)
            .collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn linear_combine_three_sources_powers_of_delta() {
        // out[i] = a[i] + delta * b[i] + delta^2 * c[i]
        let a = vec![FE::from(1u64), FE::from(0u64)];
        let b = vec![FE::from(0u64), FE::from(1u64)];
        let c = vec![FE::from(1u64), FE::from(1u64)];
        let delta = FE::from(3u64);
        let out = linear_combine_evaluations(&[&a[..], &b[..], &c[..]], &delta);
        let delta_sq = &delta * &delta;
        // out[0] = 1 + 3*0 + 9*1 = 10
        // out[1] = 0 + 3*1 + 9*1 = 12
        assert_eq!(out[0], FE::from(1u64) + &delta_sq);
        assert_eq!(out[1], FE::from(3u64) + &delta_sq);
    }

    #[test]
    fn linear_combine_zero_delta_keeps_only_first_source() {
        let a = vec![FE::from(7u64), FE::from(7u64)];
        let b = vec![FE::from(99u64), FE::from(99u64)];
        let zero = FE::from(0u64);
        let out = linear_combine_evaluations(&[&a[..], &b[..]], &zero);
        assert_eq!(out, a);
    }
}

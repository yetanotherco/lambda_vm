pub mod fri_commitment;
pub mod fri_decommit;
pub(crate) mod fri_functions;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
pub use math::field::element::FieldElement;
use math::field::traits::IsSubFieldOf;
use math::field::traits::{IsFFTField, IsField};
use math::traits::AsBytes;

use crate::config::{FriLayerMerkleTree, FriLayerMerkleTreeBackend, MERKLE_CAP_HEIGHT};

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::{
    compute_coset_twiddles_inv, fold_evaluations_in_place, update_twiddles_in_place,
};

/// FRI commit phase with arity-4 folding, from pre-computed bit-reversed
/// evaluations (skipping the initial FFT — use when the caller already has the
/// evaluation vector, e.g. from a fused LDE pipeline).
///
/// `number_layers` is `log2(trace_length)`. The first fold (p₀ → p₁) is
/// uncommitted — its inputs come from the DEEP/trace openings. The remaining
/// folds are grouped into `number_layers / 2` committed arity-4 layers; each
/// commits its input evaluations with quad leaves and folds twice. The folding
/// continues until the layer collapses to a constant, returned as the last
/// value (one extra binary fold happens for even `number_layers`).
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

    // Committed arity-4 layers. One uncommitted initial fold halves the
    // evaluations to trace_length, then `number_layers / 2` arity-4 layers fold
    // four-at-a-time. For odd `number_layers` (e.g. trace_length = 8, 32, …)
    // the final `evals` slice has 2 elements but they are equal — the
    // polynomial is degree 0 by then — so taking `evals[0]` as the last value
    // is still correct.
    let num_committed = number_layers / 2;
    let mut fri_layer_list = Vec::with_capacity(num_committed);

    // <<<< Receive challenge 𝜁₀ — uncommitted initial fold p₀ → p₁. The
    // verifier always replays this fold, so the prover always performs it.
    {
        let zeta = transcript.sample_field_element();
        fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);
        update_twiddles_in_place(&mut inv_twiddles);
    }

    for _ in 0..num_committed {
        // Commit the current layer with quad leaves: one leaf per fold orbit.
        let leaves: Vec<[FieldElement<E>; 4]> = evals
            .chunks_exact(4)
            .map(|chunk| {
                [
                    chunk[0].clone(),
                    chunk[1].clone(),
                    chunk[2].clone(),
                    chunk[3].clone(),
                ]
            })
            .collect();
        let merkle_tree = FriLayerMerkleTree::build(&leaves)
            .expect("FRI commit: Merkle tree construction must succeed");
        let cap = merkle_tree.cap(MERKLE_CAP_HEIGHT);

        // >>>> Send commitment cap: [pₖ] (before sampling this layer's challenges).
        for node in &cap {
            transcript.append_bytes(node);
        }
        fri_layer_list.push(FriLayer::new(&evals, merkle_tree));

        // Fold by 4 = two binary folds with independent challenges.
        let zeta_a = transcript.sample_field_element();
        fold_evaluations_in_place(&mut evals, &zeta_a, &inv_twiddles);
        update_twiddles_in_place(&mut inv_twiddles);

        let zeta_b = transcript.sample_field_element();
        fold_evaluations_in_place(&mut evals, &zeta_b, &inv_twiddles);
        update_twiddles_in_place(&mut inv_twiddles);
    }

    // >>>> Send value: pₙ — the constant value of the final layer.
    let last_value = evals.first().cloned().unwrap_or_else(FieldElement::zero);
    transcript.append_field_element(&last_value);

    (last_value, fri_layer_list)
}

/// FRI query phase. For each query index, reveal the 4-element fold orbit and
/// one authentication path per committed arity-4 layer.
pub fn query_phase<F: IsField>(
    fri_layers: &[FriLayer<F, FriLayerMerkleTreeBackend<F>>],
    iotas: &[usize],
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    if fri_layers.is_empty() {
        // No committed layers (tiny traces): the verifier still needs one
        // decommitment entry per query, even if empty.
        return iotas
            .iter()
            .map(|_| FriDecommitment {
                layers_auth_paths: vec![],
                layers_evaluations: vec![],
            })
            .collect();
    }

    let num_layers = fri_layers.len();
    iotas
        .iter()
        .map(|iota_s| {
            let mut layers_auth_paths = Vec::with_capacity(num_layers);
            let mut layers_evaluations = Vec::with_capacity(num_layers);

            let mut index = *iota_s;
            for layer in fri_layers {
                // The fold orbit: four consecutive (bit-reversed) evaluations
                // that fold together into one next-layer value.
                let base = index & !3;
                let orbit = [
                    layer.evaluation[base].clone(),
                    layer.evaluation[base + 1].clone(),
                    layer.evaluation[base + 2].clone(),
                    layer.evaluation[base + 3].clone(),
                ];
                let auth_path = layer
                    .merkle_tree
                    .get_proof_by_pos_capped(index >> 2, MERKLE_CAP_HEIGHT)
                    .expect("FRI query: layer orbit index in bounds");
                layers_evaluations.push(orbit);
                layers_auth_paths.push(auth_path);

                index >>= 2;
            }

            FriDecommitment {
                layers_auth_paths,
                layers_evaluations,
            }
        })
        .collect()
}

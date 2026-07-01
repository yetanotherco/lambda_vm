pub mod batched;
pub mod fri_commitment;
pub mod fri_decommit;
pub(crate) mod fri_functions;
pub mod mmcs;

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

/// FRI commit phase from pre-computed bit-reversed evaluations, skipping the
/// initial FFT. Use this when the caller already has the evaluation vector
/// (e.g. from a fused LDE pipeline).
///
/// The `T: Clone` and `F/E: 'static` bounds are required by the cuda GPU
/// fast path (`try_fri_commit_gpu` snapshots the transcript and TypeId-
/// checks the field types). They are present unconditionally (including
/// in builds without the `cuda` feature) to keep one stable signature.
pub fn commit_phase_from_evaluations<
    F: IsFFTField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
    T: IsStarkTranscript<E, F> + Clone,
>(
    number_layers: usize,
    mut evals: Vec<FieldElement<E>>,
    transcript: &mut T,
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
    // GPU fast path: drives the entire commit phase device-side (per-layer
    // fold + Keccak leaves + pair-hash tree, only D2H'ing each layer's root
    // + evals + nodes for FriLayer construction). Returns `None` on any
    // failure: precondition misses skip cleanly, and `try_fri_commit_gpu`
    // snapshots the transcript before mutating it so a mid-loop cudarc
    // error restores state and lets the CPU loop below run as if the GPU
    // had never been tried.
    #[cfg(feature = "cuda")]
    {
        if let Some(result) = crate::gpu_lde::try_fri_commit_gpu::<F, E, T>(
            number_layers,
            &evals,
            transcript,
            coset_offset,
            domain_size,
        ) {
            return result;
        }
    }

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

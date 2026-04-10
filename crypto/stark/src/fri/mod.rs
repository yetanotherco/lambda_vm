pub mod fri_commitment;
pub mod fri_decommit;
pub(crate) mod fri_functions;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
pub use math::field::element::FieldElement;
use math::field::traits::IsSubFieldOf;
use math::field::traits::{IsFFTField, IsField};
use math::traits::AsBytes;

use crate::config::{FriLayerMerkleTree, FriLayerMerkleTreeBackend};

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::{
    compute_coset_twiddles_inv, fold_evaluations_in_place, update_twiddles_in_place,
};

/// An insertion of new table DEEP evaluations into the FRI cascade at a specific layer.
/// Used by folding insertion FRI (S-two Protocol 3) to merge tables with different
/// domain sizes into a single shared FRI.
pub struct FriInsertion<E: IsField> {
    /// Which FRI layer to insert after folding (0-indexed from the first fold).
    /// Layer k means the insertion happens after the k-th fold, when the cascade
    /// domain has been halved k times from the initial size.
    pub layer: usize,
    /// Alpha-batched DEEP evaluations for tables at this domain size (bit-reversed).
    pub evals: Vec<FieldElement<E>>,
}

/// FRI commit phase from pre-computed bit-reversed evaluations.
/// skipping the initial FFT. Use this when the caller already has the evaluation
/// vector (e.g. from a fused LDE pipeline).
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
    // Inverse twiddle factors for evaluation-form folding
    let mut inv_twiddles = compute_coset_twiddles_inv(coset_offset, domain_size);

    let mut fri_layer_list = Vec::with_capacity(number_layers);
    let mut current_coset_offset = coset_offset.clone();
    let mut current_domain_size = domain_size;

    for _ in 1..number_layers {
        // <<<< Receive challenge 𝜁ₖ₋₁
        let zeta = transcript.sample_field_element();
        current_coset_offset = current_coset_offset.square();
        current_domain_size /= 2;

        // Fold evaluations in-place (no FFT needed)
        fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);

        // Build Merkle tree from consecutive pairs
        let leaves: Vec<[FieldElement<E>; 2]> = evals
            .chunks_exact(2)
            .map(|chunk| [chunk[0].clone(), chunk[1].clone()])
            .collect();
        let merkle_tree = FriLayerMerkleTree::build(&leaves)
            .expect("FRI commit: Merkle tree construction must succeed");
        let root = merkle_tree.root;
        fri_layer_list.push(FriLayer::new(
            &evals,
            merkle_tree,
            current_coset_offset.clone().to_extension(),
            current_domain_size,
        ));

        // >>>> Send commitment: [pₖ]
        transcript.append_bytes(&root);

        // Update twiddles for next level
        update_twiddles_in_place(&mut inv_twiddles);
    }

    // <<<< Receive challenge: 𝜁ₙ₋₁
    let zeta = transcript.sample_field_element();

    // Final fold
    fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);

    let last_value = evals.first().unwrap_or(&FieldElement::zero()).clone();

    // >>>> Send value: pₙ
    transcript.append_field_element(&last_value);

    (last_value, fri_layer_list)
}

/// FRI commit phase with folding insertion for multi-table proving.
///
/// Like `commit_phase_from_evaluations`, but supports inserting additional tables'
/// DEEP evaluations at specific folding rounds. This implements the S-two Protocol 3
/// (Section 4.2) adapted to standard FRI over Goldilocks.
///
/// Tables with the largest domain start the cascade. When folding reduces the domain
/// to match a smaller table's size, that table's alpha-batched DEEP evaluations are
/// added (with lambda^2 weighting) to the folded vector before continuing.
///
/// `insertions` must be sorted by layer (ascending). Each insertion's `evals` must
/// have length equal to the cascade domain size at that layer (after folding).
pub fn commit_phase_with_insertion<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    number_layers: usize,
    mut evals: Vec<FieldElement<E>>,
    insertions: &[FriInsertion<E>],
    lambda_sq: &FieldElement<E>,
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
    // Inverse twiddle factors for evaluation-form folding
    let mut inv_twiddles = compute_coset_twiddles_inv(coset_offset, domain_size);

    let mut fri_layer_list = Vec::with_capacity(number_layers);
    let mut current_coset_offset = coset_offset.clone();
    let mut current_domain_size = domain_size;
    let mut insertion_idx = 0;

    for layer_num in 1..number_layers {
        // <<<< Receive challenge zeta_k
        let zeta = transcript.sample_field_element();
        current_coset_offset = current_coset_offset.square();
        current_domain_size /= 2;

        // Fold evaluations in-place (no FFT needed)
        fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);

        // Check for insertions at this layer (layer_num corresponds to the fold number)
        // After fold number `layer_num`, the domain has been halved `layer_num` times.
        while insertion_idx < insertions.len() && insertions[insertion_idx].layer == layer_num {
            let insertion = &insertions[insertion_idx];
            debug_assert_eq!(
                insertion.evals.len(),
                evals.len(),
                "Insertion evals length ({}) must match folded cascade length ({}) at layer {}",
                insertion.evals.len(),
                evals.len(),
                layer_num,
            );

            // Add lambda^2 * h_new[i] to each folded evaluation
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                evals
                    .par_iter_mut()
                    .zip(insertion.evals.par_iter())
                    .for_each(|(e, h)| {
                        *e = &*e + &(lambda_sq * h);
                    });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for (e, h) in evals.iter_mut().zip(insertion.evals.iter()) {
                    *e = &*e + &(lambda_sq * h);
                }
            }

            insertion_idx += 1;
        }

        // Build Merkle tree from consecutive pairs
        let leaves: Vec<[FieldElement<E>; 2]> = evals
            .chunks_exact(2)
            .map(|chunk| [chunk[0].clone(), chunk[1].clone()])
            .collect();
        let merkle_tree = FriLayerMerkleTree::build(&leaves)
            .expect("FRI commit: Merkle tree construction must succeed");
        let root = merkle_tree.root;
        fri_layer_list.push(FriLayer::new(
            &evals,
            merkle_tree,
            current_coset_offset.clone().to_extension(),
            current_domain_size,
        ));

        // >>>> Send commitment: [p_k]
        transcript.append_bytes(&root);

        // Update twiddles for next level
        update_twiddles_in_place(&mut inv_twiddles);
    }

    // <<<< Receive challenge: zeta_{n-1}
    let zeta = transcript.sample_field_element();

    // Final fold
    fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);

    // Check for insertions at the final layer
    while insertion_idx < insertions.len() && insertions[insertion_idx].layer == number_layers {
        let insertion = &insertions[insertion_idx];
        debug_assert_eq!(
            insertion.evals.len(),
            evals.len(),
            "Insertion evals length must match folded cascade length at final layer"
        );
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            evals
                .par_iter_mut()
                .zip(insertion.evals.par_iter())
                .for_each(|(e, h)| {
                    *e = &*e + &(lambda_sq * h);
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for (e, h) in evals.iter_mut().zip(insertion.evals.iter()) {
                *e = &*e + &(lambda_sq * h);
            }
        }
        insertion_idx += 1;
    }

    let last_value = evals.first().unwrap_or(&FieldElement::zero()).clone();

    // >>>> Send value: p_n
    transcript.append_field_element(&last_value);

    (last_value, fri_layer_list)
}

pub fn query_phase<F: IsField>(
    fri_layers: &Vec<FriLayer<F, FriLayerMerkleTreeBackend<F>>>,
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
                let mut layers_auth_paths_sym = Vec::with_capacity(num_layers);

                let mut index = *iota_s;
                for layer in fri_layers {
                    // symmetric element
                    let evaluation_sym = layer.evaluation[index ^ 1].clone();
                    let auth_path_sym = layer.merkle_tree.get_proof_by_pos(index >> 1).unwrap();
                    layers_evaluations_sym.push(evaluation_sym);
                    layers_auth_paths_sym.push(auth_path_sym);

                    index >>= 1;
                }

                FriDecommitment {
                    layers_auth_paths: layers_auth_paths_sym,
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

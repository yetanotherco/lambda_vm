pub mod fri_commitment;
pub mod fri_decommit;
pub(crate) mod fri_functions;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
pub use math::field::element::FieldElement;
use math::field::traits::IsSubFieldOf;
use math::field::traits::{IsFFTField, IsField};
use math::traits::AsBytes;

use crate::config::{FriLayerQuadMerkleTree, FriLayerQuadMerkleTreeBackend};

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::{
    compute_coset_twiddles_inv, fold_evaluations_in_place, update_twiddles_in_place,
};

/// FRI commit phase using arity-4 folding (2 binary folds per committed layer).
///
/// For `number_layers` binary fold levels, this produces `(number_layers - 1) / 2`
/// committed layers (each covering 2 binary folds) plus a final single-fold to
/// produce the last value. For a 2^19 trace: 19 levels → 9 committed layers.
///
/// Each committed layer is a quad Merkle tree (4-element leaves), halving the
/// number of Merkle commits vs binary FRI.
pub fn commit_phase_from_evaluations<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    number_layers: usize,
    mut evals: Vec<FieldElement<E>>,
    transcript: &mut impl IsStarkTranscript<E, F>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> (
    FieldElement<E>,
    Vec<FriLayer<E, FriLayerQuadMerkleTreeBackend<E>>>,
)
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let mut inv_twiddles = compute_coset_twiddles_inv(coset_offset, domain_size);

    let mut fri_layer_list = Vec::new();
    let mut current_coset_offset = coset_offset.clone();
    let mut current_domain_size = domain_size;

    // Number of double-fold (arity-4) committed rounds from the (number_layers - 1) middle layers.
    // The final fold is handled separately below.
    let num_double_rounds = number_layers.saturating_sub(1) / 2;

    for _ in 0..num_double_rounds {
        // Sample both fold challenges before committing (Fiat-Shamir: both betas
        // depend only on transcript state before this round's commitment).
        let zeta1 = transcript.sample_field_element();
        let zeta2 = transcript.sample_field_element();

        // First binary fold: current_size → current_size / 2
        current_coset_offset = current_coset_offset.square();
        current_domain_size /= 2;
        fold_evaluations_in_place(&mut evals, &zeta1, &inv_twiddles);
        update_twiddles_in_place(&mut inv_twiddles);

        // Second binary fold: current_size / 2 → current_size / 4
        current_coset_offset = current_coset_offset.square();
        current_domain_size /= 2;
        fold_evaluations_in_place(&mut evals, &zeta2, &inv_twiddles);

        // Commit the doubly-folded evaluations as quad (4-element) Merkle leaves.
        let leaves: Vec<[FieldElement<E>; 4]> = evals
            .chunks_exact(4)
            .map(|c| [c[0].clone(), c[1].clone(), c[2].clone(), c[3].clone()])
            .collect();
        let merkle_tree = FriLayerQuadMerkleTree::build(&leaves)
            .expect("FRI commit: quad Merkle tree construction must succeed");
        let root = merkle_tree.root;
        fri_layer_list.push(FriLayer::new(
            &evals,
            merkle_tree,
            current_coset_offset.clone().to_extension(),
            current_domain_size,
        ));

        // Append commitment to transcript so subsequent samples depend on it.
        transcript.append_bytes(&root);

        update_twiddles_in_place(&mut inv_twiddles);
    }

    // Handle the leftover single binary round when (number_layers - 1) is odd.
    // For number_layers=19: (19-1)/2 = 9 double rounds, remainder 0 → skipped.
    // For number_layers=20: (20-1)/2 = 9 double rounds, remainder 1 → one extra.
    if number_layers.saturating_sub(1) % 2 == 1 {
        let zeta = transcript.sample_field_element();
        current_coset_offset = current_coset_offset.square();
        current_domain_size /= 2;
        fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);

        // Commit remaining as quad leaves (evals.len() must be >= 4 here).
        let leaves: Vec<[FieldElement<E>; 4]> = evals
            .chunks_exact(4)
            .map(|c| [c[0].clone(), c[1].clone(), c[2].clone(), c[3].clone()])
            .collect();
        let merkle_tree = FriLayerQuadMerkleTree::build(&leaves)
            .expect("FRI commit: quad Merkle tree construction must succeed");
        let root = merkle_tree.root;
        fri_layer_list.push(FriLayer::new(
            &evals,
            merkle_tree,
            current_coset_offset.clone().to_extension(),
            current_domain_size,
        ));
        transcript.append_bytes(&root);
        update_twiddles_in_place(&mut inv_twiddles);
    }

    // Final fold: one more binary fold to produce the last value (not committed).
    let zeta = transcript.sample_field_element();
    fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);

    let last_value = evals.first().unwrap_or(&FieldElement::zero()).clone();
    transcript.append_field_element(&last_value);

    (last_value, fri_layer_list)
}

pub fn query_phase<F: IsField>(
    fri_layers: &[FriLayer<F, FriLayerQuadMerkleTreeBackend<F>>],
    iotas: &[usize],
    num_double_rounds: usize,
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    if !fri_layers.is_empty() {
        let num_layers = fri_layers.len();
        iotas
            .iter()
            .map(|iota_s| {
                let mut layers_evaluations_siblings = Vec::with_capacity(num_layers);
                let mut layers_auth_paths = Vec::with_capacity(num_layers);

                // For double bootstrap (num_double_rounds >= 1): iota is already the
                // index in layer[0] (which has LDE/4 elements after 2 binary folds).
                // For single bootstrap (num_double_rounds == 0): layer[0] has LDE/2
                // elements, so the index is 2*iota.
                let mut index = if num_double_rounds >= 1 {
                    *iota_s
                } else {
                    iota_s * 2
                };
                for (i, layer) in fri_layers.iter().enumerate() {
                    // The 4-element orbit of `index` is {index&~3, ..., (index&~3)+3}.
                    // index^1, index^2, index^3 are the 3 siblings (XOR flips last 2 bits).
                    let s1 = layer.evaluation[index ^ 1].clone();
                    let s2 = layer.evaluation[index ^ 2].clone();
                    let s3 = layer.evaluation[index ^ 3].clone();

                    // Quad leaf position: each leaf holds 4 evaluations, leaf j covers
                    // indices {4j, 4j+1, 4j+2, 4j+3}, so the leaf index is index >> 2.
                    let auth_path = layer.merkle_tree.get_proof_by_pos(index >> 2).unwrap();

                    layers_evaluations_siblings.push([s1, s2, s3]);
                    layers_auth_paths.push(auth_path);

                    // Round (i+1) is a double fold iff (i+1) < num_double_rounds,
                    // meaning layer[i] → layer[i+1] involves 2 binary folds (index >>= 2).
                    // Otherwise it is a single fold (index >>= 1).
                    if (i + 1) < num_double_rounds {
                        index >>= 2;
                    } else {
                        index >>= 1;
                    }
                }

                FriDecommitment {
                    layers_auth_paths,
                    layers_evaluations_siblings,
                }
            })
            .collect()
    } else {
        iotas
            .iter()
            .map(|_| FriDecommitment {
                layers_auth_paths: vec![],
                layers_evaluations_siblings: vec![],
            })
            .collect()
    }
}

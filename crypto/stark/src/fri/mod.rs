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

/// FRI commit phase from pre-computed bit-reversed evaluations.
///
/// Supports configurable folding arity (`log_arity` binary folds per Merkle commit)
/// and early termination (`log_final_poly_len` remaining evaluations sent as coefficients).
///
/// Protocol structure (matches the verifier's expectations):
/// - Round 0: receive challenge ζ₀, fold **1×** (always arity-2 initial fold), commit p₁.
///   This corresponds to the verifier's "initial fold: always arity-2 from DEEP composition".
/// - Rounds 1+: receive challenge ζₖ, fold **log_arity×**, commit pₖ.
///   The verifier folds each committed layer's group by log_arity sub-folds using ζₖ.
/// - Last round: fold log_arity×, **no commit** — send final polynomial coefficients.
///
/// `total_folds` is adjusted so that `(total_folds - 1)` is divisible by `log_arity`,
/// ensuring every non-initial round does exactly `log_arity` folds.
/// The final polynomial may have > 1 element when `log_arity > 1`.
///
/// For `log_arity=1` this reproduces the standard arity-2 FRI exactly.
pub fn commit_phase_from_evaluations<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    number_layers: usize,
    mut evals: Vec<FieldElement<E>>,
    transcript: &mut impl IsStarkTranscript<E, F>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
    log_arity: usize,
    log_final_poly_len: usize,
) -> (
    Vec<FieldElement<E>>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>>,
)
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let mut inv_twiddles = compute_coset_twiddles_inv(coset_offset, domain_size);

    // Maximum binary folds available given the domain and desired final poly length.
    let max_folds = number_layers.max(1).saturating_sub(log_final_poly_len);

    // Adjust total_folds so that (total_folds - 1) is divisible by log_arity.
    // This ensures every non-initial round does exactly log_arity folds, matching
    // the verifier which always folds log_arity times per committed layer.
    // For log_arity=1 this is a no-op (any value satisfies (T-1) % 1 == 0).
    let total_folds = if max_folds == 0 {
        0
    } else {
        // Largest T ≤ max_folds with T = 1 + n*log_arity for integer n ≥ 0.
        let n = (max_folds - 1) / log_arity;
        1 + n * log_arity
    };

    // Number of committed FRI layers: all rounds except the last (final poly round).
    let num_committed_rounds = if total_folds > 0 {
        // Round 0 always commits (1 fold). Rounds 1..k commit every log_arity folds.
        // Total committed = 1 + (total_folds - 1) / log_arity - 1 = (total_folds - 1) / log_arity.
        (total_folds - 1) / log_arity
    } else {
        0
    };

    // Leaf grouping for Merkle trees: verifier always folds log_arity sub-folds per layer.
    let group_size = 1usize << log_arity;

    let mut fri_layer_list = Vec::with_capacity(num_committed_rounds);
    let mut current_coset_offset = coset_offset.clone();
    let mut current_domain_size = domain_size;
    let mut folds_done = 0;
    let mut is_first_round = true;

    while folds_done < total_folds {
        // Round 0 always folds exactly 1 time (the "initial arity-2 fold").
        // All subsequent rounds fold exactly log_arity times.
        let folds_this_round = if is_first_round { 1 } else { log_arity };
        is_first_round = false;
        let is_last_round = folds_done + folds_this_round >= total_folds;

        // <<<< Receive challenge: one per committed round
        let zeta = transcript.sample_field_element();

        // Apply folds_this_round sequential binary folds.
        // Challenges: zeta, zeta^2, zeta^4, ..., zeta^(2^(k-1))
        let mut challenge = zeta;
        for _ in 0..folds_this_round {
            fold_evaluations_in_place(&mut evals, &challenge, &inv_twiddles);
            current_coset_offset = current_coset_offset.square();
            current_domain_size /= 2;
            update_twiddles_in_place(&mut inv_twiddles);
            challenge = challenge.square();
        }
        folds_done += folds_this_round;

        // Commit post-fold evaluations (skip for last round — final poly sent separately).
        // Leaves always group `group_size = 2^log_arity` elements so the verifier
        // can open and fold an entire arity-k coset at each query.
        if !is_last_round {
            let leaves: Vec<Vec<FieldElement<E>>> = evals
                .chunks_exact(group_size)
                .map(|chunk| chunk.to_vec())
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
        }
    }

    // >>>> Send final polynomial coefficients
    let fri_final_poly = evals;
    for coeff in &fri_final_poly {
        transcript.append_field_element(coeff);
    }

    (fri_final_poly, fri_layer_list)
}

/// FRI query phase: for each query index, collect siblings and auth paths from each layer.
///
/// `log_arity` controls how many elements are in each leaf group.
/// For arity-2 (log_arity=1): each layer has 1 sibling.
/// For arity-4 (log_arity=2): each layer has 3 siblings.
pub fn query_phase<F: IsField>(
    fri_layers: &[FriLayer<F, FriLayerMerkleTreeBackend<F>>],
    iotas: &[usize],
    log_arity: usize,
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    if !fri_layers.is_empty() {
        let num_layers = fri_layers.len();
        let group_size = 1usize << log_arity;

        iotas
            .iter()
            .map(|iota_s| {
                let mut layers_evaluations_sym = Vec::with_capacity(num_layers);
                let mut layers_auth_paths_sym = Vec::with_capacity(num_layers);

                let mut index = *iota_s;
                for layer in fri_layers {
                    // Collect all siblings in the group (K-1 elements, excluding the queried one)
                    let group_start = (index / group_size) * group_size;
                    let siblings: Vec<FieldElement<F>> = (0..group_size)
                        .filter(|&k| group_start + k != index)
                        .map(|k| layer.evaluation[group_start + k].clone())
                        .collect();

                    let leaf_index = index / group_size;
                    let auth_path = layer.merkle_tree.get_proof_by_pos(leaf_index).unwrap();
                    layers_evaluations_sym.push(siblings);
                    layers_auth_paths_sym.push(auth_path);

                    index >>= log_arity;
                }

                FriDecommitment {
                    layers_auth_paths: layers_auth_paths_sym,
                    layers_evaluations_sym,
                }
            })
            .collect()
    } else {
        iotas
            .iter()
            .map(|_| FriDecommitment {
                layers_auth_paths: vec![],
                layers_evaluations_sym: vec![],
            })
            .collect()
    }
}

pub mod fri_commitment;
pub mod fri_decommit;
mod fri_functions;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::traits::{IsFFTField, IsField};
use math::traits::AsBytes;
use math::{fft::cpu::bit_reversing::in_place_bit_reverse_permute, field::traits::IsSubFieldOf};
pub use math::{field::element::FieldElement, polynomial::Polynomial};

use crate::config::{FriLayerMerkleTree, FriLayerMerkleTreeBackend};

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::fold_polynomial_doubled_inplace;

type CommitResult<E> = (
    Vec<FieldElement<E>>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>>,
);

/// FRI commit phase with configurable folding factor and early stopping.
///
/// - `number_layers`: total number of binary folds (= log2(trace_length))
/// - `folding_factor`: number of evaluations per Merkle leaf (power of 2, >= 2)
/// - `last_layer_degree_bound`: stop folding when degree <= this bound (0 = fold to constant)
///
/// Returns the coefficients of the last polynomial and the committed FRI layers.
pub fn commit_phase<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    number_layers: usize,
    p_0: Polynomial<FieldElement<E>>,
    transcript: &mut impl IsStarkTranscript<E, F>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
    folding_factor: usize,
    last_layer_degree_bound: usize,
) -> CommitResult<E>
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let log_f = folding_factor.trailing_zeros() as usize;

    // How many binary folds are absorbed by the last polynomial (not committed)
    let last_poly_log = if last_layer_degree_bound == 0 {
        0
    } else {
        (last_layer_degree_bound + 1).trailing_zeros() as usize
    };

    // Handle edge case: very small traces (number_layers <= 1)
    if number_layers == 0 {
        let last_value = p_0.coefficients().to_vec();
        for coeff in &last_value {
            transcript.append_field_element(coeff);
        }
        return (last_value, Vec::new());
    }

    // Total folds to perform = number_layers - last_poly_log.
    // Early stopping skips the last `last_poly_log` folds; the remaining polynomial
    // (degree <= last_layer_degree_bound) is sent directly.
    let folds_to_perform = number_layers.saturating_sub(last_poly_log);

    // The -1 accounts for the initial fold (zetas[0]) before the first committed layer.
    // Only folds that fit into complete committed layers are performed; any remainder
    // is absorbed by the last polynomial (slightly higher degree than degree_bound).
    let non_initial_folds = folds_to_perform.saturating_sub(1);
    let num_committed_layers = non_initial_folds / log_f;

    let mut domain_size = domain_size;
    let mut fri_layer_list = Vec::with_capacity(num_committed_layers);
    let mut current_poly = p_0;
    let mut coset_offset = coset_offset.clone();

    // Initial fold with zetas[0] (no commitment for this one)
    let zeta_0 = transcript.sample_field_element();
    coset_offset = coset_offset.square();
    domain_size /= 2;
    fold_polynomial_doubled_inplace(&mut current_poly, &zeta_0);

    // Committed layers: each does log_f folds and one Merkle commitment
    for _ in 0..num_committed_layers {
        // Build Merkle tree for current state (after previous folds)
        let current_layer =
            new_fri_layer(&current_poly, &coset_offset, domain_size, folding_factor);
        let new_data = current_layer.merkle_tree.root;
        fri_layer_list.push(current_layer);

        // >>>> Send commitment: [pₖ]
        transcript.append_bytes(&new_data);

        // Apply log_f binary folds (one challenge per fold)
        for _ in 0..log_f {
            let zeta = transcript.sample_field_element();
            coset_offset = coset_offset.square();
            domain_size /= 2;
            fold_polynomial_doubled_inplace(&mut current_poly, &zeta);
        }
    }

    // Extract last polynomial coefficients
    let last_value = current_poly.coefficients().to_vec();

    // >>>> Send last polynomial coefficients
    for coeff in &last_value {
        transcript.append_field_element(coeff);
    }

    (last_value, fri_layer_list)
}

pub fn query_phase<F: IsField>(
    fri_layers: &Vec<FriLayer<F, FriLayerMerkleTreeBackend<F>>>,
    iotas: &[usize],
    folding_factor: usize,
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    let log_f = folding_factor.trailing_zeros() as usize;

    if !fri_layers.is_empty() {
        let num_layers = fri_layers.len();
        iotas
            .iter()
            .map(|iota_s| {
                let mut layers_evaluations_sym = Vec::with_capacity(num_layers);
                let mut layers_auth_paths_sym = Vec::with_capacity(num_layers);

                let mut index = *iota_s;
                for layer in fri_layers {
                    let leaf_index = index >> log_f;
                    let known_pos = index % folding_factor;

                    // Collect all sibling evaluations except the one the verifier computes
                    let mut sym_evals = Vec::with_capacity(folding_factor - 1);
                    for pos in 0..folding_factor {
                        if pos != known_pos {
                            let eval_index = leaf_index * folding_factor + pos;
                            sym_evals.push(layer.evaluation[eval_index].clone());
                        }
                    }

                    let auth_path = layer.merkle_tree.get_proof_by_pos(leaf_index).unwrap();
                    layers_evaluations_sym.push(sym_evals);
                    layers_auth_paths_sym.push(auth_path);

                    index = leaf_index;
                }

                FriDecommitment {
                    layers_auth_paths: layers_auth_paths_sym,
                    layers_evaluations_sym,
                }
            })
            .collect()
    } else {
        // For 0 FRI layers (small traces), return empty decommitments for each query.
        iotas
            .iter()
            .map(|_| FriDecommitment {
                layers_auth_paths: vec![],
                layers_evaluations_sym: vec![],
            })
            .collect()
    }
}

pub fn new_fri_layer<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    poly: &Polynomial<FieldElement<E>>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
    folding_factor: usize,
) -> FriLayer<E, FriLayerMerkleTreeBackend<E>>
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let mut evaluation =
        Polynomial::evaluate_offset_fft(poly, 1, Some(domain_size), coset_offset).unwrap(); // TODO: return error

    in_place_bit_reverse_permute(&mut evaluation);

    debug_assert_eq!(
        evaluation.len() % folding_factor,
        0,
        "domain_size must be divisible by folding_factor"
    );
    let leaves: Vec<Vec<FieldElement<E>>> = evaluation
        .chunks_exact(folding_factor)
        .map(|chunk| chunk.to_vec())
        .collect();

    let merkle_tree = FriLayerMerkleTree::build(&leaves).unwrap();

    FriLayer::new(
        &evaluation,
        merkle_tree,
        coset_offset.clone().to_extension(),
        domain_size,
    )
}

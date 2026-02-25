pub mod fri_commitment;
pub mod fri_decommit;
mod fri_functions;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::field::traits::{IsFFTField, IsField};
use math::traits::AsBytes;
use math::field::traits::IsSubFieldOf;
pub use math::{
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
    polynomial::Polynomial,
};

use crate::config::{FriLayerMerkleTree, FriLayerMerkleTreeBackend};

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::{compute_coset_twiddles_inv, fold_evaluations_in_place, update_twiddles_in_place};

pub fn commit_phase<F: IsFFTField + IsSubFieldOf<E>, E: IsField + Send + Sync>(
    number_layers: usize,
    p_0: Polynomial<FieldElement<E>>,
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
    // FFT evaluation produces bit-reversed output; use bitrev variant to skip redundant BRP.
    let evals = Polynomial::evaluate_offset_fft_bitrev(&p_0, 1, Some(domain_size), coset_offset)
        .expect("FRI commit: FFT evaluation of p₀ on coset domain must succeed");
    drop(p_0);

    commit_phase_from_evaluations(
        number_layers, evals, transcript, coset_offset, domain_size,
        log_arity, log_final_poly_len,
    )
}

/// Like [`commit_phase`], but takes pre-computed bit-reversed evaluations directly,
/// skipping the initial FFT. Use this when the caller already has the evaluation
/// vector (e.g. from a fused LDE pipeline).
pub fn commit_phase_from_evaluations<F: IsFFTField + IsSubFieldOf<E>, E: IsField + Send + Sync>(
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
    let total_binary_folds = number_layers.saturating_sub(log_final_poly_len);
    let mut inv_twiddles = compute_coset_twiddles_inv(coset_offset, domain_size);
    let num_committed_layers = if total_binary_folds > log_arity {
        (total_binary_folds - 1) / log_arity - 1 + 1 // = ceil(total/arity) - 1
    } else {
        0
    };
    let mut fri_layer_list = Vec::with_capacity(num_committed_layers);
    let mut current_coset_offset = coset_offset.clone();
    let mut current_domain_size = domain_size;
    let mut binary_folds_done = 0;

    while binary_folds_done < total_binary_folds {
        let folds_this_round = log_arity.min(total_binary_folds - binary_folds_done);
        let is_last = binary_folds_done + folds_this_round >= total_binary_folds;

        // <<<< Receive challenge: one per round
        let zeta: FieldElement<E> = transcript.sample_field_element();

        // Apply folds_this_round sequential arity-2 folds
        // Challenges: zeta, zeta^2, zeta^4, ..., zeta^(2^(k-1))
        let mut challenge = zeta;
        for _ in 0..folds_this_round {
            fold_evaluations_in_place(&mut evals, &challenge, &inv_twiddles);
            current_coset_offset = current_coset_offset.square();
            current_domain_size /= 2;
            update_twiddles_in_place(&mut inv_twiddles);
            challenge = challenge.square();
        }
        binary_folds_done += folds_this_round;

        // Commit post-fold evaluations (except for last round → final poly)
        if !is_last {
            let group_size = 1usize << folds_this_round;
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

    // Extract final polynomial coefficients
    let final_poly = extract_final_poly::<F, E>(&evals, log_final_poly_len);

    // >>>> Send value: pₙ
    for coeff in &final_poly {
        transcript.append_field_element(coeff);
    }

    (final_poly, fri_layer_list)
}

/// Extract final polynomial from evaluation-form FRI residual.
///
/// When `log_final_poly_len == 0`, the evaluations have been folded to a single constant.
/// When `log_final_poly_len > 0`, there are `2^log_final_poly_len` evaluations remaining
/// on a coset; recover coefficients via bit-reverse + iFFT.
fn extract_final_poly<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    evals: &[FieldElement<E>],
    log_final_poly_len: usize,
) -> Vec<FieldElement<E>>
where
    E: Send + Sync,
{
    if log_final_poly_len == 0 {
        vec![evals.first().unwrap_or(&FieldElement::zero()).clone()]
    } else {
        let final_poly_len = 1usize << log_final_poly_len;
        let mut sub_evals: Vec<_> = evals[..final_poly_len].to_vec();
        in_place_bit_reverse_permute(&mut sub_evals);
        Polynomial::interpolate_fft::<F>(&sub_evals)
            .expect("iFFT for final poly must succeed")
            .coefficients()
            .to_vec()
    }
}

pub fn query_phase<F: IsField>(
    fri_layers: &[FriLayer<F, FriLayerMerkleTreeBackend<F>>],
    iotas: &[usize],
    log_arity: usize,
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    if !fri_layers.is_empty() {
        let group_size = 1usize << log_arity;
        let num_layers = fri_layers.len();
        iotas
            .iter()
            .map(|iota_s| {
                let mut layers_evaluations_sym = Vec::with_capacity(num_layers);
                let mut layers_auth_paths_sym = Vec::with_capacity(num_layers);

                let mut index = *iota_s;
                for layer in fri_layers {
                    // Sibling elements: all elements in the same leaf group except
                    // the queried one. For arity 2^k, there are 2^k - 1 siblings.
                    let group_index = index / group_size;
                    let pos_in_group = index % group_size;
                    let group_start = group_index * group_size;
                    let mut siblings = Vec::with_capacity(group_size - 1);
                    for i in 0..group_size {
                        if i != pos_in_group {
                            siblings.push(layer.evaluation[group_start + i].clone());
                        }
                    }
                    let auth_path = layer.merkle_tree.get_proof_by_pos(group_index).unwrap();
                    layers_evaluations_sym.push(siblings);
                    layers_auth_paths_sym.push(auth_path);

                    index = group_index;
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

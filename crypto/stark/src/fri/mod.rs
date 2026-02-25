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
    let evals = Polynomial::evaluate_offset_fft_bitrev(&p_0, 1, Some(domain_size), coset_offset)
        .expect("FRI commit: FFT evaluation of p₀ on coset domain must succeed");
    drop(p_0);

    commit_phase_from_evaluations(
        number_layers, evals, transcript, coset_offset, domain_size,
        log_arity, log_final_poly_len,
    )
}

/// FRI commit phase from pre-computed bit-reversed evaluations.
///
/// # Protocol structure
///
/// The protocol proceeds in rounds. Each round samples one challenge and applies
/// one or more sequential binary folds. The first round always does exactly 1 fold
/// (matching the verifier's initial fold from the DEEP polynomial pair). Subsequent
/// rounds fold `log_arity` times, with the challenge squaring between sub-folds:
/// `zeta, zeta^2, zeta^4, ...`.
///
/// After each round's fold(s), if more folding remains, the evaluations are committed
/// in a Merkle tree with leaves of `2^log_arity` consecutive elements. These leaves
/// provide the verifier with enough data to locally replay `log_arity` folds.
///
/// The Fiat-Shamir transcript order is preserved:
/// `sample challenge → fold → commit → sample challenge → fold → commit → ...`
pub fn commit_phase_from_evaluations<F: IsFFTField + IsSubFieldOf<E>, E: IsField + Send + Sync>(
    _number_layers: usize,
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
    // Compute total binary folds. For higher-arity FRI (log_arity > 1), ensure that
    // after the initial fold (1 binary fold matching the verifier's DEEP pair fold),
    // the remaining folds are a multiple of log_arity. This guarantees each committed
    // FRI layer has exactly log_arity sub-folds, matching the verifier's uniform
    // per-layer fold count.
    let base = _number_layers
        .saturating_sub(log_final_poly_len)
        .max(if evals.len() > 1 { 1 } else { 0 });
    let total_binary_folds = if base <= 1 || log_arity <= 1 {
        base
    } else {
        let after_initial = base - 1;
        let rounded_up = after_initial.div_ceil(log_arity) * log_arity;
        let candidate = 1 + rounded_up;
        // Ensure we don't exceed available evaluations
        let max_binary_folds = (domain_size.trailing_zeros() as usize)
            .saturating_sub(log_final_poly_len);
        if candidate <= max_binary_folds {
            candidate
        } else {
            // Round down instead
            1 + (after_initial / log_arity) * log_arity
        }
    };
    let mut inv_twiddles = compute_coset_twiddles_inv(coset_offset, domain_size);
    let mut fri_layer_list = Vec::new();
    let mut current_coset_offset = coset_offset.clone();
    let mut current_domain_size = domain_size;
    let mut binary_folds_done = 0;
    let group_size = 1usize << log_arity;

    while binary_folds_done < total_binary_folds {
        // First round: 1 fold (initial binary fold matching verifier's DEEP pair fold).
        // Subsequent rounds: log_arity folds (verifier replays from committed leaf group).
        let folds_this_round = if binary_folds_done == 0 {
            1.min(total_binary_folds)
        } else {
            log_arity.min(total_binary_folds - binary_folds_done)
        };
        let is_last = binary_folds_done + folds_this_round >= total_binary_folds;

        // <<<< Receive challenge: one per round
        let zeta: FieldElement<E> = transcript.sample_field_element();

        // Apply sequential binary folds with squaring challenges
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
    let final_poly = extract_final_poly::<F, E>(&evals, log_final_poly_len, &current_coset_offset);

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
/// on a coset with offset `coset_offset`; recover coefficients via bit-reverse + iFFT
/// + coset correction (divide coefficient j by offset^j).
fn extract_final_poly<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    evals: &[FieldElement<E>],
    log_final_poly_len: usize,
    coset_offset: &FieldElement<F>,
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
        // Standard iFFT treats evaluations as being at roots of unity.
        // Since they're actually at coset points (offset * w^i), the iFFT
        // gives d_j = c_j * offset^j. Divide by offset^j to recover c_j.
        let mut coeffs = Polynomial::interpolate_fft::<F>(&sub_evals)
            .expect("iFFT for final poly must succeed")
            .coefficients()
            .to_vec();
        let offset_inv = coset_offset.inv().expect("coset offset is nonzero");
        let mut offset_inv_power = FieldElement::<F>::one();
        for c in coeffs.iter_mut() {
            *c = &offset_inv_power * &*c;
            offset_inv_power = &offset_inv_power * &offset_inv;
        }
        coeffs
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
                    // The queried index falls within a leaf group. Extract all siblings
                    // (group elements except the queried one).
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

                    // After folding the group of 2^log_arity values down to 1 value,
                    // the next layer's index is the group_index.
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

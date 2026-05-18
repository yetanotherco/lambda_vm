pub mod fri_commitment;
pub mod fri_decommit;
pub(crate) mod fri_functions;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
pub use math::field::element::FieldElement;
use math::field::traits::IsSubFieldOf;
use math::field::traits::{IsFFTField, IsField};
use math::polynomial::Polynomial;
use math::traits::AsBytes;

use crate::config::{FriLayerMerkleTree, FriLayerMerkleTreeBackend};

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::{
    compute_coset_twiddles_inv, fold_evaluations_in_place, update_twiddles_in_place,
};

/// FRI commit phase from pre-computed bit-reversed evaluations.
/// skipping the initial FFT. Use this when the caller already has the evaluation
/// vector (e.g. from a fused LDE pipeline).
///
/// `folding_factor` and `last_layer_degree_bound` come from
/// [`crate::proof::options::ProofOptions`].
/// - `folding_factor` is required to be 2 here; higher folding factors need
///   different-sized Merkle leaves and are deferred to a follow-up commit.
/// - `last_layer_degree_bound = 0` (default) folds all the way to a constant
///   and ships a length-1 `fri_last_value`.
/// - `last_layer_degree_bound = K` (with `K + 1` a power of two) stops folding
///   when the remaining evaluation vector has size `K + 1` and ships those
///   evaluations directly — the verifier looks up its own value at the
///   reduced query index instead of comparing to a single constant.
pub fn commit_phase_from_evaluations<F: IsFFTField + IsSubFieldOf<E>, E: IsField + Send + Sync>(
    number_layers: usize,
    mut evals: Vec<FieldElement<E>>,
    transcript: &mut impl IsStarkTranscript<E, F>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
    folding_factor: usize,
    last_layer_degree_bound: usize,
) -> (
    Vec<FieldElement<E>>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>>,
)
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    assert_eq!(
        folding_factor, 2,
        "FRI folding_factor > 2 is not yet wired into commit_phase; only folding_factor=2 is supported"
    );

    let last_poly_log = if last_layer_degree_bound == 0 {
        0
    } else {
        (last_layer_degree_bound + 1).trailing_zeros() as usize
    };
    // We always do at least one fold to match the legacy convention
    // (`for _ in 1..number_layers` + 1 final fold = `max(1, number_layers)`).
    let folds_to_perform = number_layers.saturating_sub(last_poly_log).max(1);

    // Inverse twiddle factors for evaluation-form folding
    let mut inv_twiddles = compute_coset_twiddles_inv(coset_offset, domain_size);

    let mut fri_layer_list = Vec::with_capacity(folds_to_perform);
    let mut current_coset_offset = coset_offset.clone();
    let mut current_domain_size = domain_size;

    for _ in 1..folds_to_perform {
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
    fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);
    // After the final fold, evaluations live on the squared coset.
    current_coset_offset = current_coset_offset.square();

    // Compute the final polynomial (sent as coefficients, P3-style).
    //
    // - Fold-to-constant (last_poly_log = 0): `evals` is either empty
    //   (degenerate) or holds at least one value that IS the constant. Ship
    //   a length-1 Vec; matches the legacy wire format byte-for-byte and
    //   sidesteps the iFFT machinery.
    // - Early-stop (last_poly_log > 0): the residual polynomial has degree
    //   <= last_layer_degree_bound. Its evaluations on the squared coset of
    //   size `last_poly_size = 2^last_poly_log` are the first half of `evals`
    //   in bit-reversed order (the second half is the blowup-redundant
    //   duplicate). Truncate, bit-reverse to natural order, and iFFT via
    //   `Polynomial::interpolate_offset_fft` to recover the coefficients.
    //   The verifier consumes these by Horner-evaluating the polynomial at
    //   the layer-K point that its running `v` represents.
    let last_value: Vec<FieldElement<E>> = if last_poly_log == 0 {
        vec![evals.first().unwrap_or(&FieldElement::zero()).clone()]
    } else {
        let last_poly_size = 1usize << last_poly_log;
        evals.truncate(last_poly_size);
        in_place_bit_reverse_permute(&mut evals);
        let poly = Polynomial::<FieldElement<E>>::interpolate_offset_fft::<F>(
            &evals,
            &current_coset_offset,
        )
        .expect("FRI early-stop: iFFT on residual evals must succeed");
        let mut coefs = poly.coefficients;
        coefs.resize(last_poly_size, FieldElement::zero());
        coefs
    };

    // >>>> Send the final-polynomial coefficients.
    for coef in &last_value {
        transcript.append_field_element(coef);
    }

    (last_value, fri_layer_list)
}

/// FRI query phase.
///
/// `folding_factor` will determine how many sibling evaluations are exposed
/// per layer in `FriDecommitment.layers_evaluations_sym`. The current
/// implementation supports only `folding_factor == 2` (one sibling per
/// layer); other values panic here.
pub fn query_phase<F: IsField>(
    fri_layers: &Vec<FriLayer<F, FriLayerMerkleTreeBackend<F>>>,
    iotas: &[usize],
    folding_factor: usize,
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    assert_eq!(
        folding_factor, 2,
        "FRI folding_factor > 2 is not yet wired into query_phase; only folding_factor=2 is supported"
    );

    if !fri_layers.is_empty() {
        let num_layers = fri_layers.len();
        iotas
            .iter()
            .map(|iota_s| {
                let mut layers_evaluations_sym: Vec<Vec<FieldElement<F>>> =
                    Vec::with_capacity(num_layers);
                let mut layers_auth_paths_sym = Vec::with_capacity(num_layers);

                let mut index = *iota_s;
                for layer in fri_layers {
                    // Binary fold: one sibling evaluation per layer (the
                    // symmetric one). Wrapped in a Vec to share shape with
                    // the future folding=4 path that yields `folding_factor - 1`
                    // siblings per leaf.
                    let evaluation_sym = layer.evaluation[index ^ 1].clone();
                    let auth_path_sym = layer.merkle_tree.get_proof_by_pos(index >> 1).unwrap();
                    layers_evaluations_sym.push(vec![evaluation_sym]);
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

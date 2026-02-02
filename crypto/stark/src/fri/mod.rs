pub mod fri_commitment;
pub mod fri_decommit;
mod fri_functions;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::traits::{IsFFTField, IsField};
use math::traits::AsBytes;
use math::{fft::cpu::bit_reversing::in_place_bit_reverse_permute, field::traits::IsSubFieldOf};
pub use math::{
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
    polynomial::Polynomial,
};

use crate::config::{FriLayerMerkleTree, FriLayerMerkleTreeBackend};

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::fold_polynomial_doubled_inplace;

pub fn commit_phase<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    number_layers: usize,
    p_0: Polynomial<FieldElement<E>>,
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
    let mut domain_size = domain_size;

    let mut fri_layer_list = Vec::with_capacity(number_layers);
    let mut current_layer: FriLayer<E, FriLayerMerkleTreeBackend<E>>;
    let mut current_poly = p_0;

    let mut coset_offset = coset_offset.clone();

    for _ in 1..number_layers {
        // <<<< Receive challenge 𝜁ₖ₋₁
        let zeta = transcript.sample_field_element();
        coset_offset = coset_offset.square();
        domain_size /= 2;

        // In-place folding avoids memory allocation
        fold_polynomial_doubled_inplace(&mut current_poly, &zeta);
        current_layer = new_fri_layer(&current_poly, &coset_offset, domain_size);
        // Copy just the root (small, 32 bytes) before moving the layer
        let new_data = current_layer.merkle_tree.root;
        fri_layer_list.push(current_layer);

        // >>>> Send commitment: [pₖ]
        transcript.append_bytes(&new_data);
    }

    // <<<< Receive challenge: 𝜁ₙ₋₁
    let zeta = transcript.sample_field_element();

    // Final fold - still in-place
    fold_polynomial_doubled_inplace(&mut current_poly, &zeta);

    let last_value = current_poly
        .coefficients()
        .first()
        .unwrap_or(&FieldElement::zero())
        .clone();

    // >>>> Send value: pₙ
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

pub fn new_fri_layer<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    poly: &Polynomial<FieldElement<E>>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> FriLayer<E, FriLayerMerkleTreeBackend<E>>
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let mut evaluation =
        Polynomial::evaluate_offset_fft(poly, 1, Some(domain_size), coset_offset).unwrap(); // TODO: return error

    in_place_bit_reverse_permute(&mut evaluation);

    // Use fixed-size arrays instead of Vec for each pair (avoids allocation per pair)
    let leaves: Vec<[FieldElement<E>; 2]> = evaluation
        .chunks_exact(2)
        .map(|chunk| [chunk[0].clone(), chunk[1].clone()])
        .collect();

    let merkle_tree = FriLayerMerkleTree::build(&leaves).unwrap();

    FriLayer::new(
        &evaluation,
        merkle_tree,
        coset_offset.clone().to_extension(),
        domain_size,
    )
}

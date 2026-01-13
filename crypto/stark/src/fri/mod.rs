pub mod fri_commitment;
pub mod fri_decommit;
pub mod fri_functions;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::traits::{IsFFTField, IsField};
use math::traits::AsBytes;
use math::{fft::cpu::bit_reversing::in_place_bit_reverse_permute, field::traits::IsSubFieldOf};
pub use math::{
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
    polynomial::Polynomial,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::config::{FriLayerMerkleTree, FriLayerMerkleTreeBackend};

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::fold_polynomial_doubled_inplace;

pub fn commit_phase<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    num_layers: usize,
    initial_poly: Polynomial<FieldElement<E>>,
    transcript: &mut impl IsStarkTranscript<E, F>,
    initial_offset: &FieldElement<F>,
    initial_domain_size: usize,
) -> (
    FieldElement<E>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>>,
)
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let mut layers = Vec::with_capacity(num_layers);
    let mut current_poly = initial_poly;
    let mut coset_offset = initial_offset.clone();
    let mut domain_size = initial_domain_size;

    for _ in 1..num_layers {
        let zeta = transcript.sample_field_element();
        coset_offset = coset_offset.square();
        domain_size /= 2;

        // In-place folding avoids memory allocation
        fold_polynomial_doubled_inplace(&mut current_poly, &zeta);
        let layer = new_fri_layer(&current_poly, &coset_offset, domain_size);

        transcript.append_bytes(&layer.merkle_tree.root);
        layers.push(layer);
    }

    let zeta = transcript.sample_field_element();
    // Final fold - still in-place
    fold_polynomial_doubled_inplace(&mut current_poly, &zeta);
    let final_value = current_poly
        .coefficients()
        .first()
        .cloned()
        .unwrap_or_else(FieldElement::zero);

    transcript.append_field_element(&final_value);

    (final_value, layers)
}

pub fn query_phase<F: IsField>(
    layers: &[FriLayer<F, FriLayerMerkleTreeBackend<F>>],
    query_indices: &[usize],
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    if layers.is_empty() {
        return vec![];
    }

    // Parallelize query processing - each query is independent
    #[cfg(feature = "parallel")]
    let iter = query_indices.par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = query_indices.iter();

    iter.map(|&initial_index| {
        let mut evaluations = Vec::with_capacity(layers.len());
        let mut auth_paths = Vec::with_capacity(layers.len());
        let mut index = initial_index;

        for layer in layers {
            let symmetric_index = index ^ 1;
            evaluations.push(layer.evaluation[symmetric_index].clone());
            auth_paths.push(layer.merkle_tree.get_proof_by_pos(index >> 1).unwrap());
            index >>= 1;
        }

        FriDecommitment {
            layers_auth_paths: auth_paths,
            layers_evaluations_sym: evaluations,
        }
    })
    .collect()
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
        Polynomial::evaluate_offset_fft(poly, 1, Some(domain_size), coset_offset).unwrap();

    in_place_bit_reverse_permute(&mut evaluation);

    // Use fixed-size arrays instead of Vec for each pair (avoids allocation per pair)
    let leaves: Vec<[FieldElement<E>; 2]> = evaluation
        .chunks_exact(2)
        .map(|chunk| [chunk[0].clone(), chunk[1].clone()])
        .collect();

    let merkle_tree = FriLayerMerkleTree::build(&leaves).unwrap();

    FriLayer::new(
        evaluation,
        merkle_tree,
        coset_offset.clone().to_extension(),
        domain_size,
    )
}

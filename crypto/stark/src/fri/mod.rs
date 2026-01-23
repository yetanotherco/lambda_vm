pub mod fri_commitment;
pub mod fri_decommit;
mod fri_functions;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use crypto::merkle_tree::merkle::MerkleTree;
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;
pub use math::{
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
    polynomial::Polynomial,
};

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::fold_polynomial_doubled_inplace;

// ============================================================
// Shared helpers (Keccak + Poseidon2)
// ============================================================

fn commit_phase_impl<F, E, B, T, AppendCommitment, NewLayer>(
    number_layers: usize,
    p_0: Polynomial<FieldElement<E>>,
    transcript: &mut T,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
    mut append_commitment: AppendCommitment,
    new_layer: NewLayer,
) -> (FieldElement<E>, Vec<FriLayer<E, B>>)
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    T: IsStarkTranscript<E, F>,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend,
    AppendCommitment: FnMut(&B::Node, &mut T),
    NewLayer: Fn(&Polynomial<FieldElement<E>>, &FieldElement<F>, usize) -> FriLayer<E, B>,
{
    let mut domain_size = domain_size;

    let mut fri_layer_list = Vec::with_capacity(number_layers);
    let mut current_layer: FriLayer<E, B>;
    let mut current_poly = p_0;

    let mut coset_offset = coset_offset.clone();

    for _ in 1..number_layers {
        // <<<< Receive challenge 𝜁ₖ₋₁
        let zeta = transcript.sample_field_element();
        coset_offset = coset_offset.square();
        domain_size /= 2;

        // In-place folding avoids memory allocation
        fold_polynomial_doubled_inplace(&mut current_poly, &zeta);
        current_layer = new_layer(&current_poly, &coset_offset, domain_size);

        // >>>> Send commitment: [pₖ]
        append_commitment(&current_layer.merkle_tree.root, transcript);

        fri_layer_list.push(current_layer);
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

fn query_phase_impl<F, B>(
    fri_layers: &Vec<FriLayer<F, B>>,
    iotas: &[usize],
) -> Vec<FriDecommitment<F>>
where
    F: IsField,
    FieldElement<F>: AsBytes + Sync + Send,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend<Node = crate::config::Commitment>,
{
    if !fri_layers.is_empty() {
        iotas
            .iter()
            .map(|iota_s| {
                let mut layers_evaluations_sym = Vec::new();
                let mut layers_auth_paths_sym = Vec::new();

                let mut index = *iota_s;
                for layer in fri_layers {
                    // symmetric element
                    let evaluation_sym = layer.evaluation[index ^ 1].clone();
                    let auth_path_sym = layer
                        .merkle_tree
                        .get_proof_by_pos(index >> 1)
                        .expect("FRI query index out of bounds - invalid iota or layer size");
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
        vec![]
    }
}

fn new_fri_layer_impl<F, E, B, BuildLeaves>(
    poly: &Polynomial<FieldElement<E>>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
    build_leaves: BuildLeaves,
) -> FriLayer<E, B>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    B: crypto::merkle_tree::traits::IsMerkleTreeBackend,
    BuildLeaves: FnOnce(&[FieldElement<E>]) -> Vec<B::Data>,
{
    let mut evaluation = Polynomial::evaluate_offset_fft(poly, 1, Some(domain_size), coset_offset)
        .expect(
            "FFT evaluation failed - domain size must be a power of 2 and match polynomial degree",
        );

    in_place_bit_reverse_permute(&mut evaluation);

    let leaves = build_leaves(&evaluation);
    let merkle_tree = MerkleTree::<B>::build(&leaves)
        .expect("Merkle tree construction failed - leaves must be non-empty and power of 2");

    FriLayer::new(
        &evaluation,
        merkle_tree,
        coset_offset.clone().to_extension(),
        domain_size,
    )
}

// ============================================================
// Keccak256 Implementation (default)
// ============================================================

#[cfg(not(feature = "poseidon2"))]
use crate::config::FriLayerMerkleTreeBackend;

#[cfg(not(feature = "poseidon2"))]
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
    commit_phase_impl::<F, E, FriLayerMerkleTreeBackend<E>, _, _, _>(
        number_layers,
        p_0,
        transcript,
        coset_offset,
        domain_size,
        |root, t| t.append_bytes(root),
        |poly, offset, size| new_fri_layer(poly, offset, size),
    )
}

#[cfg(not(feature = "poseidon2"))]
pub fn query_phase<F: IsField>(
    fri_layers: &Vec<FriLayer<F, FriLayerMerkleTreeBackend<F>>>,
    iotas: &[usize],
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    query_phase_impl(fri_layers, iotas)
}

#[cfg(not(feature = "poseidon2"))]
pub fn new_fri_layer<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    poly: &Polynomial<FieldElement<E>>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> FriLayer<E, FriLayerMerkleTreeBackend<E>>
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    new_fri_layer_impl::<F, E, FriLayerMerkleTreeBackend<E>, _>(
        poly,
        coset_offset,
        domain_size,
        |evaluation| {
            evaluation
                .chunks_exact(2)
                .map(|chunk| [chunk[0].clone(), chunk[1].clone()])
                .collect()
        },
    )
}

// ============================================================
// Poseidon2 Implementation (feature = "poseidon2")
// ============================================================
// Note: These implementations use the Goldilocks field (Fp).
// The generic parameters are kept for API compatibility but internally
// use Poseidon2-specific types.

#[cfg(feature = "poseidon2")]
use crate::config::{FriLayerMerkleTreeBackendInner, field_element_to_fps};

#[cfg(feature = "poseidon2")]
pub fn commit_phase<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    number_layers: usize,
    p_0: Polynomial<FieldElement<E>>,
    transcript: &mut impl IsStarkTranscript<E, F>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> (
    FieldElement<E>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackendInner>>,
)
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    commit_phase_impl::<F, E, FriLayerMerkleTreeBackendInner, _, _, _>(
        number_layers,
        p_0,
        transcript,
        coset_offset,
        domain_size,
        |root, t| {
            let bytes = crate::config::commitment_to_bytes(root);
            t.append_bytes(&bytes);
        },
        |poly, offset, size| new_fri_layer(poly, offset, size),
    )
}

#[cfg(feature = "poseidon2")]
pub fn query_phase<F: IsField>(
    fri_layers: &Vec<FriLayer<F, FriLayerMerkleTreeBackendInner>>,
    iotas: &[usize],
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    query_phase_impl(fri_layers, iotas)
}

#[cfg(feature = "poseidon2")]
pub fn new_fri_layer<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    poly: &Polynomial<FieldElement<E>>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> FriLayer<E, FriLayerMerkleTreeBackendInner>
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    new_fri_layer_impl::<F, E, FriLayerMerkleTreeBackendInner, _>(
        poly,
        coset_offset,
        domain_size,
        |evaluation| {
            evaluation
                .chunks_exact(2)
                .map(|chunk| {
                    let mut leaf = field_element_to_fps(&chunk[0]);
                    leaf.extend(field_element_to_fps(&chunk[1]));
                    leaf
                })
                .collect()
        },
    )
}

use core::marker::PhantomData;

use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use alloc::vec::Vec;
use digest::{Digest, Output};
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

#[derive(Clone)]
pub struct FieldElementVectorBackend<F, D: Digest, const NUM_BYTES: usize, const ARITY: usize = 4> {
    phantom1: PhantomData<F>,
    phantom2: PhantomData<D>,
}

impl<F, D: Digest, const NUM_BYTES: usize, const ARITY: usize> Default
    for FieldElementVectorBackend<F, D, NUM_BYTES, ARITY>
{
    fn default() -> Self {
        Self {
            phantom1: PhantomData,
            phantom2: PhantomData,
        }
    }
}

impl<F, D: Digest, const NUM_BYTES: usize, const ARITY: usize> IsMerkleTreeBackend
    for FieldElementVectorBackend<F, D, NUM_BYTES, ARITY>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
    Vec<FieldElement<F>>: Sync + Send,
{
    type Node = [u8; NUM_BYTES];
    type Data = Vec<FieldElement<F>>;

    const ARITY: usize = ARITY;

    fn hash_data(input: &Vec<FieldElement<F>>) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        for element in input.iter() {
            hasher.update(element.as_bytes());
        }
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_new_parent(children: &[Self::Node]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        for child in children {
            hasher.update(child);
        }
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}

#[derive(Clone, Default)]
pub struct BatchPoseidonTree<P: Poseidon + Default> {
    _poseidon: PhantomData<P>,
}

impl<P> IsMerkleTreeBackend for BatchPoseidonTree<P>
where
    P: Poseidon + Default,
    Vec<FieldElement<P::F>>: Sync + Send,
    FieldElement<P::F>: Sync + Send,
{
    type Node = FieldElement<P::F>;
    type Data = Vec<FieldElement<P::F>>;

    const ARITY: usize = 4;

    fn hash_data(input: &Vec<FieldElement<P::F>>) -> FieldElement<P::F> {
        P::hash_many(input)
    }

    fn hash_new_parent(children: &[Self::Node]) -> FieldElement<P::F> {
        debug_assert!(
            children.len() >= 2,
            "hash_new_parent requires at least 2 children, got {}",
            children.len()
        );
        let mut acc = P::hash(&children[0], &children[1]);
        for child in &children[2..] {
            acc = P::hash(&acc, child);
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use math::field::{
        element::FieldElement, fields::fft_friendly::u64_goldilocks::GoldilocksField,
    };
    use sha2::Sha512;
    use sha3::{Keccak256, Keccak512, Sha3_256, Sha3_512};

    use crate::merkle_tree::{
        backends::field_element_vector::FieldElementVectorBackend, merkle::MerkleTree,
    };

    type F = GoldilocksField;
    type FE = FieldElement<F>;

    fn make_values(n: usize) -> Vec<Vec<FE>> {
        (2..2 + n as u64)
            .map(|i| vec![FE::from(i), FE::from(i * 3 + 1)])
            .collect()
    }

    #[test]
    fn hash_data_field_element_backend_works_with_sha3_256() {
        let values = make_values(16);
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Sha3_256, 32>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Sha3_256, 32>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_keccak256() {
        let values = make_values(16);
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Keccak256, 32>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Keccak256, 32>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_sha3_512() {
        let values = make_values(16);
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Sha3_512, 64>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Sha3_512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_keccak512() {
        let values = make_values(16);
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Keccak512, 64>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Keccak512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_sha2_512() {
        let values = make_values(16);
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Sha512, 64>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Sha512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }
}

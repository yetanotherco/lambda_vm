use alloc::vec::Vec;
use crypto::merkle_tree::{merkle::MerkleTree, traits::IsMerkleTreeBackend};
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

#[cfg_attr(not(feature = "disk-spill"), derive(Clone))]
pub struct FriLayer<F, B>
where
    F: IsField,
    FieldElement<F>: AsBytes + math::traits::ByteConversion,
    B: IsMerkleTreeBackend,
{
    pub evaluation: Vec<FieldElement<F>>,
    pub merkle_tree: MerkleTree<B>,
    pub coset_offset: FieldElement<F>,
    pub domain_size: usize,
}

impl<F, B> FriLayer<F, B>
where
    F: IsField,
    FieldElement<F>: AsBytes + math::traits::ByteConversion,
    B: IsMerkleTreeBackend,
{
    pub fn new(
        evaluation: &[FieldElement<F>],
        merkle_tree: MerkleTree<B>,
        coset_offset: FieldElement<F>,
        domain_size: usize,
    ) -> Self {
        Self {
            evaluation: evaluation.to_vec(),
            merkle_tree,
            coset_offset,
            domain_size,
        }
    }
}

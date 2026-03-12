use crypto::merkle_tree::{merkle::MerkleTree, traits::IsMerkleTreeBackend};
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

#[derive(Clone)]
pub struct FriLayer<F, B>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    B: IsMerkleTreeBackend,
{
    pub evaluation: Vec<FieldElement<F>>,
    pub merkle_tree: MerkleTree<B>,
    pub domain_size: usize,
}

impl<F, B> FriLayer<F, B>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    B: IsMerkleTreeBackend,
{
    pub fn new(evaluation: Vec<FieldElement<F>>, merkle_tree: MerkleTree<B>, domain_size: usize) -> Self {
        Self {
            evaluation,
            merkle_tree,
            domain_size,
        }
    }
}

use crypto::merkle_tree::{merkle::MerkleTree, traits::IsMerkleTreeBackend};
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

#[cfg_attr(not(feature = "disk-spill"), derive(Clone))]
pub struct FriLayer<F, B>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    B: IsMerkleTreeBackend,
{
    pub evaluation: Vec<FieldElement<F>>,
    pub merkle_tree: MerkleTree<B>,
    /// The layer's Merkle tree kept resident on device (GPU FRI commit path),
    /// so R4 query openings gather authentication paths on device. When set,
    /// `merkle_tree` is a root only placeholder. `None` on the CPU path.
    #[cfg(feature = "cuda")]
    pub gpu_tree: Option<math_cuda::lde::GpuMerkleTree>,
}

impl<F, B> FriLayer<F, B>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    B: IsMerkleTreeBackend,
{
    pub fn new(evaluation: &[FieldElement<F>], merkle_tree: MerkleTree<B>) -> Self {
        Self {
            evaluation: evaluation.to_vec(),
            merkle_tree,
            #[cfg(feature = "cuda")]
            gpu_tree: None,
        }
    }
}

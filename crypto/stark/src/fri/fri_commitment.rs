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
    /// The layer's folded evaluations kept resident on device (ext3 interleaved,
    /// `3 * len` u64), so the query phase can gather the opened
    /// `evaluation[value_pos]` values on device. `Some` on the GPU commit path
    /// (during F1 it coexists with the still-authoritative host `evaluation`;
    /// from F2 it becomes the sole source and `evaluation` is dropped); `None` on
    /// the CPU path.
    #[cfg(feature = "cuda")]
    pub gpu_evals: Option<math_cuda::fri::GpuFriEvals>,
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
            #[cfg(feature = "cuda")]
            gpu_evals: None,
        }
    }
}

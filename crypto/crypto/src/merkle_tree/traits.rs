use alloc::vec::Vec;
use math::field::element::FieldElement;
use math::field::traits::IsField;
use math::traits::AsBytes;
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// A backend for Merkle trees. This defines raw `Data` from which the Merkle
/// tree is built from. It also defines the `Node` type and the hash function
/// used to build parent nodes from children nodes.
pub trait IsMerkleTreeBackend {
    type Node: PartialEq + Eq + Clone + Sync + Send;
    type Data: Sync + Send;

    /// This function takes a single variable `Data` and converts it to a node.
    fn hash_data(leaf: &Self::Data) -> Self::Node;

    /// This function takes the list of data from which the Merkle
    /// tree will be built from and converts it to a list of leaf nodes.
    fn hash_leaves(unhashed_leaves: &[Self::Data]) -> Vec<Self::Node> {
        #[cfg(feature = "parallel")]
        let iter = unhashed_leaves.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = unhashed_leaves.iter();

        iter.map(|leaf| Self::hash_data(leaf)).collect()
    }

    /// This function takes to children nodes and builds a new parent node.
    /// It will be used in the construction of the Merkle tree.
    fn hash_new_parent(child_1: &Self::Node, child_2: &Self::Node) -> Self::Node;
}

/// A leaf backend that can hash a leaf without being handed one.
///
/// [`IsMerkleTreeBackend::hash_data`] takes a `&Self::Data`, which for the
/// batched backends is a `Vec<FieldElement<F>>`. Building one per leaf costs an
/// allocation per leaf — millions on a real trace — so the prover and verifier
/// never do: they serialize into a reused buffer, or hold two slices they want
/// hashed as if concatenated. These are the two shapes they use.
///
/// Both must agree with `hash_data` on the bytes they absorb, so a leaf hashed
/// through either route is the leaf the tree was built from. That is the whole
/// contract, and it is why these live on a trait rather than staying inherent
/// methods on one concrete backend: a commitment configuration that names its
/// leaf backend generically still has to reach them.
pub trait IsStreamingLeafBackend<F>: IsMerkleTreeBackend
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    /// Hash a pre-serialized leaf buffer. Equals `hash_data` applied to the
    /// elements `data` encodes, in that order.
    fn hash_bytes(data: &[u8]) -> Self::Node;

    /// Hash `a ‖ b` without materializing the concatenation. Equals
    /// `hash_data(&[a, b].concat())`.
    fn hash_data_from_slices(a: &[FieldElement<F>], b: &[FieldElement<F>]) -> Self::Node;
}

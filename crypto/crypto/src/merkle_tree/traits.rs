use alloc::vec::Vec;
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// A backend for Merkle trees. This defines raw `Data` from which the Merkle
/// tree is built from. It also defines the `Node` type and the hash function
/// used to build parent nodes from children nodes.
///
/// The `ARITY` const determines how many children each internal node has:
/// - `ARITY = 2`: Binary tree (1 sibling per proof level, `hash_new_parent`)
/// - `ARITY = 4`: Quaternary tree (3 siblings per proof level, `hash_new_parent_4`)
pub trait IsMerkleTreeBackend {
    type Node: PartialEq + Eq + Clone + Sync + Send;
    type Data: Sync + Send;

    /// Tree arity: 2 for binary, 4 for quaternary.
    const ARITY: usize;

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

    /// Hash child nodes into a parent. The slice length equals `ARITY`.
    fn hash_new_parent(children: &[Self::Node]) -> Self::Node;
}

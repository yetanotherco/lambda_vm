use alloc::vec::Vec;
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// A backend for Merkle trees. This defines raw `Data` from which the Merkle
/// tree is built from. It also defines the `Node` type and the hash function
/// used to build parent nodes from children nodes.
pub trait IsMerkleTreeBackend {
    type Node: PartialEq + Eq + Clone + Sync + Send;
    type Data: Sync + Send;

    /// Branching factor of the tree: each internal node has exactly `ARITY`
    /// children. The default is a binary tree (`ARITY == 2`). Backends can set a
    /// higher arity to make the tree shallower — e.g. `ARITY == 4` halves the
    /// number of levels, so each Merkle path is half as deep and a verifier hashes
    /// roughly half as many internal nodes per opening. The number of leaves is
    /// padded to a power of `ARITY` at build time.
    const ARITY: usize = 2;

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

    /// This function takes two children nodes and builds a new parent node.
    /// It will be used in the construction of binary (`ARITY == 2`) Merkle trees.
    fn hash_new_parent(child_1: &Self::Node, child_2: &Self::Node) -> Self::Node;

    /// Hash exactly `ARITY` children (in order) into their parent node. The
    /// default implementation handles the binary case by delegating to
    /// [`hash_new_parent`](Self::hash_new_parent); backends with `ARITY != 2` must
    /// override this. `children.len()` is always exactly `ARITY`.
    fn hash_children(children: &[Self::Node]) -> Self::Node {
        debug_assert_eq!(children.len(), 2, "default hash_children is binary-only");
        Self::hash_new_parent(&children[0], &children[1])
    }
}

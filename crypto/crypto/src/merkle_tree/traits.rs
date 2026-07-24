use alloc::vec::Vec;
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

    /// Like [`hash_new_parent`](Self::hash_new_parent) but writes the result into
    /// `out` instead of returning it, so a Merkle-path fold can accumulate in
    /// place (ping-ponging two buffers) without the per-step by-value node copy.
    /// `out` may not alias `child_1`/`child_2`. The default is the returning form
    /// plus a move; backends whose hash can target `out` directly override this.
    fn hash_new_parent_into(child_1: &Self::Node, child_2: &Self::Node, out: &mut Self::Node) {
        *out = Self::hash_new_parent(child_1, child_2);
    }

    /// Node equality for the root check at the end of a path fold. Defaults to
    /// `PartialEq`; fixed-size byte-array backends override it to compare wide
    /// words instead of falling into a generic `memcmp` call on the hot path.
    fn nodes_eq(a: &Self::Node, b: &Self::Node) -> bool {
        a == b
    }

    /// VERIFY_PATH ecall hook (ROUND-2 increment A, MEASUREMENT-ONLY). On the
    /// riscv64 guest under `sim-path-ecall`, a 32-byte keccak backend recomputes
    /// the Merkle root from `leaf_hash` and `merkle_path` in ONE trusted host
    /// ecall and returns `Some(accept)`; every other backend returns `None`, so
    /// [`verify_merkle_path_from_leaf_hash`](super::proof::verify_merkle_path_from_leaf_hash)
    /// runs the generic fold. The ecall computes the REAL accept/reject answer
    /// (a tampered opening yields `Some(false)`), so it only ever swallows the
    /// fold's cycles — it never changes acceptance. A build using it drives no
    /// chip and must never be proven.
    #[cfg(all(target_arch = "riscv64", feature = "sim-path-ecall"))]
    fn try_verify_path_ecall(
        _merkle_path: &[Self::Node],
        _root: &Self::Node,
        _index: usize,
        _leaf_hash: &Self::Node,
    ) -> Option<bool> {
        None
    }
}

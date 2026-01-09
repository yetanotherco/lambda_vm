use core::fmt::Display;

use alloc::vec::Vec;

use super::{proof::Proof, traits::IsMerkleTreeBackend, utils::*};

#[derive(Debug)]
pub enum Error {
    OutOfBounds,
}
impl Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Accessed node was out of bound")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// The struct for the Merkle tree, consisting of the root and the nodes.
/// A typical tree would look like this
///                 root
///              /        \
///          leaf 12     leaf 34
///        /         \    /      \
///    leaf 1     leaf 2 leaf 3  leaf 4
/// The bottom leafs correspond to the hashes of the elements, while each upper
/// layer contains the hash of the concatenation of the daughter nodes.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MerkleTree<B: IsMerkleTreeBackend> {
    pub root: B::Node,
    nodes: Vec<B::Node>,
}

const ROOT: usize = 0;

impl<B> MerkleTree<B>
where
    B: IsMerkleTreeBackend,
{
    /// Create a Merkle tree from a slice of data
    pub fn build(unhashed_leaves: &[B::Data]) -> Option<Self> {
        if unhashed_leaves.is_empty() {
            return None;
        }

        let hashed_leaves: Vec<B::Node> = B::hash_leaves(unhashed_leaves);

        //The leaf must be a power of 2 set
        let hashed_leaves = complete_until_power_of_two(hashed_leaves);
        let leaves_len = hashed_leaves.len();

        //The length of leaves minus one inner node in the merkle tree
        //The first elements are overwritten by build function, it doesn't matter what it's there
        let mut nodes = vec![hashed_leaves[0].clone(); leaves_len - 1];
        nodes.extend(hashed_leaves);

        //Build the inner nodes of the tree
        build::<B>(&mut nodes, leaves_len);

        Some(MerkleTree {
            root: nodes[ROOT].clone(),
            nodes,
        })
    }

    /// Returns a Merkle proof for the element/s at position pos
    /// For example, give me an inclusion proof for the 3rd element in the
    /// Merkle tree
    pub fn get_proof_by_pos(&self, pos: usize) -> Option<Proof<B::Node>> {
        let pos = pos + self.nodes.len() / 2;
        let Ok(merkle_path) = self.build_merkle_path(pos) else {
            return None;
        };

        self.create_proof(merkle_path)
    }

    /// Creates a proof from a Merkle pasth
    fn create_proof(&self, merkle_path: Vec<B::Node>) -> Option<Proof<B::Node>> {
        Some(Proof { merkle_path })
    }

    /// Returns the Merkle path for the element/s for the leaf at position pos
    fn build_merkle_path(&self, pos: usize) -> Result<Vec<B::Node>, Error> {
        let mut merkle_path = Vec::new();
        let mut pos = pos;

        while pos != ROOT {
            let Some(node) = self.nodes.get(sibling_index(pos)) else {
                // out of bounds, exit returning the current merkle_path
                return Err(Error::OutOfBounds);
            };
            merkle_path.push(node.clone());

            pos = parent_index(pos);
        }

        Ok(merkle_path)
    }
}

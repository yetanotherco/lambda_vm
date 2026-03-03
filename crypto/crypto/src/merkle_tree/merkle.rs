use core::fmt::Display;

use crate::merkle_tree::proof::BatchProof;

use super::{
    proof::Proof,
    traits::IsMerkleTreeBackend,
    utils::{
        build, complete_until_power_of_arity, compute_depth, internal_node_count,
        num_leaves_from_total, parent_index, sibling_indices,
    },
};
use alloc::{collections::BTreeSet, vec::Vec};

#[derive(Debug)]
pub enum Error {
    OutOfBounds,
    EmptyPositionList,
}
impl Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::OutOfBounds => write!(f, "Accessed node was out of bound"),
            Error::EmptyPositionList => write!(f, "Position list cannot be empty"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// A Merkle tree whose arity is determined by the backend's `ARITY` const.
///
/// For `ARITY = 2` (binary):
///           root
///          /    \
///        n1      n2
///       / \     / \
///     l1  l2  l3  l4
///
/// For `ARITY = 4` (quaternary):
///                root
///        /     |     |     \
///     n1      n2      n3      n4
///   / | | \ / | | \  ...
/// l1 l2 l3 l4 ...
///
/// Flat array layout: [root, level-1 nodes..., ..., leaves]
/// Internal nodes = (leaves - 1) / (arity - 1)
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
        Self::build_from_hashed_leaves(hashed_leaves)
    }

    /// Create a Merkle tree from pre-hashed leaf nodes.
    pub fn build_from_hashed_leaves(hashed_leaves: Vec<B::Node>) -> Option<Self> {
        if hashed_leaves.is_empty() {
            return None;
        }

        let arity = B::ARITY;
        let hashed_leaves = complete_until_power_of_arity(hashed_leaves, arity);
        let leaves_len = hashed_leaves.len();
        let internal_nodes = internal_node_count(leaves_len, arity);

        let mut nodes = vec![hashed_leaves[0].clone(); internal_nodes];
        nodes.extend(hashed_leaves);

        build::<B>(&mut nodes, leaves_len);

        Some(MerkleTree {
            root: nodes[ROOT].clone(),
            nodes,
        })
    }

    /// Returns the number of leaves in this tree.
    fn num_leaves(&self) -> usize {
        num_leaves_from_total(self.nodes.len(), B::ARITY)
    }

    /// Returns a Merkle proof for the element at position pos
    pub fn get_proof_by_pos(&self, pos: usize) -> Option<Proof<B::Node>> {
        let num_leaves = self.num_leaves();
        let internal_nodes = self.nodes.len() - num_leaves;
        let pos = pos + internal_nodes;
        let Ok(merkle_path) = self.build_merkle_path(pos) else {
            return None;
        };

        Some(Proof { merkle_path })
    }

    /// Returns the Merkle path for the leaf at position pos.
    /// Each element in the path is a Vec of `arity - 1` sibling nodes at that level.
    fn build_merkle_path(&self, pos: usize) -> Result<Vec<Vec<B::Node>>, Error> {
        let arity = B::ARITY;
        let num_leaves = self.num_leaves();
        let tree_depth = compute_depth(num_leaves, arity);
        let mut merkle_path = Vec::with_capacity(tree_depth);
        let mut pos = pos;

        while pos != ROOT {
            let siblings = sibling_indices(pos, arity);
            if siblings.iter().any(|&s| s >= self.nodes.len()) {
                return Err(Error::OutOfBounds);
            }
            merkle_path.push(siblings.iter().map(|&s| self.nodes[s].clone()).collect());
            pos = parent_index(pos, arity);
        }

        Ok(merkle_path)
    }

    /// Given a list of leaf indices, returns a batch proof.
    pub fn get_batch_proof(&self, pos_list: &[usize]) -> Result<BatchProof<B::Node>, Error> {
        if pos_list.is_empty() {
            return Err(Error::EmptyPositionList);
        }

        let num_leaves = self.num_leaves();
        for &pos in pos_list {
            if pos >= num_leaves {
                return Err(Error::OutOfBounds);
            }
        }

        let internal_nodes = self.nodes.len() - num_leaves;
        let leaf_positions: Vec<usize> = pos_list.iter().map(|pos| pos + internal_nodes).collect();
        let batch_auth_path_positions = self.get_batch_auth_path_positions(&leaf_positions);

        let batch_auth_path_nodes = batch_auth_path_positions
            .iter()
            .map(|pos| self.nodes[*pos].clone())
            .collect();

        Ok(BatchProof {
            path: batch_auth_path_nodes,
        })
    }

    /// Returns the internal tree indices of nodes needed in the batch proof.
    /// Ordered level by level from bottom to top, ascending within each level.
    fn get_batch_auth_path_positions(&self, leaf_positions: &[usize]) -> Vec<usize> {
        let arity = B::ARITY;
        let mut obtainable: BTreeSet<usize> = leaf_positions.iter().cloned().collect();
        let num_leaves = self.num_leaves();
        let num_levels = compute_depth(num_leaves, arity);

        let mut result = Vec::new();

        for _ in 0..num_levels {
            let mut level_auth = BTreeSet::new();
            let mut next_obtainable = BTreeSet::new();

            for &pos in &obtainable {
                let siblings = sibling_indices(pos, arity);
                for sibling_pos in siblings {
                    if !obtainable.contains(&sibling_pos) {
                        level_auth.insert(sibling_pos);
                    }
                }
                next_obtainable.insert(parent_index(pos, arity));
            }

            result.extend(level_auth.iter());

            obtainable = next_obtainable;
        }

        result
    }
}


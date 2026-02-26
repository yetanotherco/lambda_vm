use core::fmt::Display;

use crate::merkle_tree::proof::BatchProof;

use super::{proof::Proof, traits::IsMerkleTreeBackend, utils::*};
use alloc::{collections::BTreeSet, vec, vec::Vec};

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
        Self::build_from_hashed_leaves(hashed_leaves)
    }

    /// Create a Merkle tree from pre-hashed leaf nodes.
    ///
    /// This skips the `hash_leaves` step, useful when leaves have already been
    /// hashed externally (e.g., to avoid materializing large intermediate data).
    pub fn build_from_hashed_leaves(hashed_leaves: Vec<B::Node>) -> Option<Self> {
        if hashed_leaves.is_empty() {
            return None;
        }

        let arity = B::ARITY;

        // Pad leaves to a power of arity
        let hashed_leaves = complete_until_power_of_arity(hashed_leaves, arity);
        let leaves_len = hashed_leaves.len();

        // Number of internal nodes: (A * L - 1) / (A - 1) - L
        let total = total_nodes_arity(leaves_len, arity);
        let internal_nodes = total - leaves_len;

        // Allocate: internal nodes (placeholder) followed by leaves
        let mut nodes = vec![hashed_leaves[0].clone(); internal_nodes];
        nodes.extend(hashed_leaves);

        // Build the internal nodes
        if arity == 2 {
            build::<B>(&mut nodes, leaves_len);
        } else {
            build_arity::<B>(&mut nodes, leaves_len);
        }

        Some(MerkleTree {
            root: nodes[ROOT].clone(),
            nodes,
        })
    }

    /// Returns a Merkle proof for the element/s at position pos
    /// For example, give me an inclusion proof for the 3rd element in the
    /// Merkle tree
    pub fn get_proof_by_pos(&self, pos: usize) -> Option<Proof<B::Node>> {
        let num_leaves = self.num_leaves();
        let leaf_offset = self.nodes.len() - num_leaves;
        let pos = pos + leaf_offset;
        let Ok(merkle_path) = self.build_merkle_path(pos) else {
            return None;
        };

        Some(Proof { merkle_path })
    }

    /// Returns the number of leaves in the tree.
    fn num_leaves(&self) -> usize {
        let arity = B::ARITY;
        if arity == 2 {
            self.nodes.len().div_ceil(2)
        } else {
            // total = (A * L - 1) / (A - 1), solve for L:
            // L = (total * (A - 1) + 1) / A
            (self.nodes.len() * (arity - 1) + 1) / arity
        }
    }

    /// Returns the Merkle path for the leaf at the given tree-internal index.
    /// For arity A, each level pushes A-1 sibling nodes.
    fn build_merkle_path(&self, pos: usize) -> Result<Vec<B::Node>, Error> {
        let arity = B::ARITY;
        let mut merkle_path = Vec::new();
        let mut pos = pos;

        while pos != ROOT {
            if arity == 2 {
                let Some(node) = self.nodes.get(sibling_index(pos)) else {
                    return Err(Error::OutOfBounds);
                };
                merkle_path.push(node.clone());
                pos = parent_index(pos);
            } else {
                let siblings = sibling_indices_arity(pos, arity);
                for &sib in &siblings {
                    let Some(node) = self.nodes.get(sib) else {
                        return Err(Error::OutOfBounds);
                    };
                    merkle_path.push(node.clone());
                }
                pos = parent_index_arity(pos, arity);
            }
        }

        Ok(merkle_path)
    }

    /// Given a list of indices, returns a batch proof containing the nodes needed to verify that all the leaves
    /// in those indices belong to the tree.
    /// It optimizes the number of nodes in the proof since the verifier can create some of them using
    /// the leaves and the parent nodes known by hashing.
    ///
    /// # Proof Structure
    /// The proof contains the nodes in **descending order by tree index**, that is, from bottom to
    /// top and from right to left:
    /// - Higher indices (closer to leaves) come first.
    /// - Lower indices (closer to root) come last.
    /// - Within the same level, nodes are ordered from right to left (higher index first).
    ///
    /// This ordering matches the verification consumption order, which processes
    /// level-by-level from leaves to root.
    ///
    /// # Errors
    /// - `Error::EmptyPositionList` if `pos_list` is empty
    /// - `Error::OutOfBounds` if any position in `pos_list` is >= number of leaves
    pub fn get_batch_proof(&self, pos_list: &[usize]) -> Result<BatchProof<B::Node>, Error> {
        if pos_list.is_empty() {
            return Err(Error::EmptyPositionList);
        }

        let num_leaves = self.num_leaves();

        // Validate all positions are within bounds
        for &pos in pos_list {
            if pos >= num_leaves {
                return Err(Error::OutOfBounds);
            }
        }

        let leaf_offset = self.nodes.len() - num_leaves;

        // Since the nodes in the merkle tree are indexed from the root to the leaves, we redefine the indices
        // of the leaves.
        let leaf_positions = pos_list
            .iter()
            .map(|pos| pos + leaf_offset)
            .collect::<Vec<usize>>();
        // We get the positions of the nodes for the batch proof.
        let batch_auth_path_positions = self.get_batch_auth_path_positions(&leaf_positions);

        // We get the nodes for the batch proof.
        let batch_auth_path_nodes = batch_auth_path_positions
            .iter()
            .map(|pos| self.nodes[*pos].clone())
            .collect();

        Ok(BatchProof {
            path: batch_auth_path_nodes,
        })
    }

    /// Returns the internal tree indices of nodes needed in the batch proof of the given
    /// leaf positions.
    ///
    /// # Result Order:
    /// The resulting indices are in descending order, that is, from bottom to
    /// top and from right to left:
    /// - Higher indices (closer to leaves) come first.
    /// - Lower indices (closer to root) come last.
    /// - Within the same level, nodes are ordered from right to left (higher index first).
    ///
    /// This ordering is critical because the verifier consumes proof nodes level-by-level
    /// starting from leaves, so it needs leaf-level siblings first.
    fn get_batch_auth_path_positions(&self, leaf_positions: &[usize]) -> Vec<usize> {
        let arity = B::ARITY;
        // BTreeSet always maintains elements in ascending order (smaller indices first), regardless of insertion order.
        let mut auth_path_set = BTreeSet::<usize>::new();
        let mut obtainable: BTreeSet<usize> = leaf_positions.iter().cloned().collect();

        // Number of levels in tree
        let num_levels = if arity == 2 {
            (self.nodes.len() + 1).ilog2()
        } else {
            let mut levels = 0u32;
            let mut size = self.num_leaves();
            while size > 1 {
                size /= arity;
                levels += 1;
            }
            levels + 1
        };

        // Iterate level-by-level from leaves to root.
        for _ in 0..num_levels - 1 {
            let mut next_obtainable = BTreeSet::new();

            for &pos in &obtainable {
                if arity == 2 {
                    if let Some(sibling_pos) = get_sibling_pos(pos) {
                        let sibling_is_obtainable = obtainable.contains(&sibling_pos)
                            || auth_path_set.contains(&sibling_pos);
                        if !sibling_is_obtainable {
                            auth_path_set.insert(sibling_pos);
                        }
                    }
                    next_obtainable.insert(get_parent_pos(pos));
                } else {
                    let siblings = sibling_indices_arity(pos, arity);
                    for &sib in &siblings {
                        let sib_is_obtainable =
                            obtainable.contains(&sib) || auth_path_set.contains(&sib);
                        if !sib_is_obtainable {
                            auth_path_set.insert(sib);
                        }
                    }
                    next_obtainable.insert(parent_index_arity(pos, arity));
                }
            }

            obtainable = next_obtainable;
        }

        // Reverse to get descending order (larger indices first).
        // This makes the proof ordered from bottom (nodes closer to leaves) to top (nodes closer to root).
        auth_path_set.into_iter().rev().collect()
    }
}

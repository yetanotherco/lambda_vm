use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use math::traits::Serializable;
use math::{errors::DeserializationError, traits::Deserializable};
use std::collections::BTreeMap;

use super::{
    traits::IsMerkleTreeBackend,
    utils::{get_parent_pos, get_sibiling_pos},
};

/// Stores a merkle path to some leaf.
/// Internally, the necessary hashes are stored from root to leaf in the
/// `merkle_path` field, in such a way that, if the merkle tree is of height `n`, the
/// `i`-th element of `merkle_path` is the sibling node in the `n - 1 - i`-th check
/// when verifying.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Proof<T: PartialEq + Eq> {
    pub merkle_path: Vec<T>,
}

impl<T: PartialEq + Eq> Proof<T> {
    /// Verifies a Merkle inclusion proof for the value contained at leaf index.
    pub fn verify<B>(&self, root_hash: &B::Node, mut index: usize, value: &B::Data) -> bool
    where
        B: IsMerkleTreeBackend<Node = T>,
    {
        let mut hashed_value = B::hash_data(value);

        for sibling_node in self.merkle_path.iter() {
            if index.is_multiple_of(2) {
                hashed_value = B::hash_new_parent(&hashed_value, sibling_node);
            } else {
                hashed_value = B::hash_new_parent(sibling_node, &hashed_value);
            }

            index >>= 1;
        }

        root_hash == &hashed_value
    }
}

#[cfg(feature = "alloc")]
impl<T> Serializable for Proof<T>
where
    T: Serializable + PartialEq + Eq,
{
    fn serialize(&self) -> Vec<u8> {
        self.merkle_path
            .iter()
            .flat_map(|node| node.serialize())
            .collect()
    }
}

impl<T> Deserializable for Proof<T>
where
    T: Deserializable + PartialEq + Eq,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        let mut merkle_path = Vec::new();
        for elem in bytes[0..].chunks(8) {
            let node = T::deserialize(elem)?;
            merkle_path.push(node);
        }
        Ok(Self { merkle_path })
    }
}

/// Stores a all the nodes needed to prove the inclusion of multiple leaves.
/// Internally, the necessary hashes are stored from bottom (leaves) to top (root) and
/// from left to right.
#[derive(Debug, Clone)]
pub struct BatchProof<T: PartialEq + Eq> {
    pub path: Vec<T>,
}

impl<T: PartialEq + Eq + Clone> BatchProof<T> {
    /// Verifies a batch Merkle proof for multiple leaves.
    /// Mirrors the logic of `get_batch_auth_path_positions` exactly.
    ///
    /// # Arguments
    /// * `root_hash` - The expected Merkle root
    /// * `pos_list` - Leaf positions (0-indexed from left)
    /// * `values` - The leaf values at those positions (not hashed)
    /// * `num_leaves` - Total number of leaves in the tree (must be a power of 2)
    pub fn verify<B>(
        &self,
        root_hash: &B::Node,
        pos_list: &[usize],
        values: &[B::Data],
        num_leaves: usize,
    ) -> bool
    where
        B: IsMerkleTreeBackend<Node = T>,
    {
        if pos_list.len() != values.len() || pos_list.is_empty() {
            return false;
        }

        // Index of the first leaf as it is ordered in the tree struct (from top to bottom).
        let first_leaf_index = num_leaves - 1;
        // We zip the input positions with the input values.
        // We need to convert leaf positions to internal tree indices by adding the index of the first leaf.
        // We also need to hash all the values to obtain the leaf nodes.
        let mut current_level_known_nodes: BTreeMap<usize, T> = pos_list
            .iter()
            .zip(values.iter())
            .map(|(&pos, value)| (pos + first_leaf_index, B::hash_data(value)))
            .collect();

        let mut proof_path_iter = self.path.iter();

        let num_levels = (2 * num_leaves).ilog2();
        // Process level by level, same as `get_batch_auth_path_positions`.
        for _ in 0..num_levels - 1 {
            let mut next_level_known_nodes: BTreeMap<usize, T> = BTreeMap::new();

            // Process each known node
            for (pos, value) in &current_level_known_nodes {
                let sibling_pos = get_sibiling_pos(*pos);
                let parent_pos = get_parent_pos(*pos);

                // Skip if parent was already computed (i.e. sibling was processed first).
                if next_level_known_nodes.contains_key(&parent_pos) {
                    continue;
                }

                // Get sibling value: from known nodes or from proof path.
                let sibling_hash = if current_level_known_nodes.contains_key(&sibling_pos) {
                    current_level_known_nodes.get(&sibling_pos).unwrap().clone()
                } else {
                    match proof_path_iter.next() {
                        Some(h) => h.clone(),
                        None => return false,
                    }
                };

                // Compute parent hash.
                let parent_hash = if pos % 2 == 1 {
                    B::hash_new_parent(value, &sibling_hash)
                } else {
                    B::hash_new_parent(&sibling_hash, value)
                };

                next_level_known_nodes.insert(parent_pos, parent_hash);
            }
            current_level_known_nodes = next_level_known_nodes;
        }

        // Verify: root computed correctly and all proof nodes consumed.
        proof_path_iter.next().is_none()
            && current_level_known_nodes.len() == 1
            && (current_level_known_nodes.get(&0) == Some(root_hash))
    }
}

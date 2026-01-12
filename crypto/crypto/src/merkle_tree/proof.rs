use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use math::traits::Serializable;
use math::{errors::DeserializationError, traits::Deserializable};

use super::traits::IsMerkleTreeBackend;

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

#[derive(Debug, Clone)]
pub struct BatchProof<T: PartialEq + Eq> {
    pub path: Vec<T>,
}

// impl<T: PartialEq + Eq> BatchProof<T> {
//     pub fn verify_batch(&self,
//         root_hash:&B::Node,
//         pos_list:&[usize],
//         values:&[B::Data],
//     ) -> bool
//     where
//         B: IsMerkleTreeBackend<Node = T>,
//     {
//         let mut hashed_values = values.iter().map(|v| B::hash_data(v)).collect::<Vec<_>>();

//     }

impl<T: PartialEq + Eq + Clone> BatchProof<T> {
    /// Verifies a batch Merkle proof for multiple leaves.
    /// `pos_list` and `values` must have the same length.
    /// `leaves_count` is the total number of leaves in the tree (must be a power of 2).
    pub fn verify<B>(
        &self,
        root_hash: &B::Node,
        pos_list: &[usize],
        values: &[B::Data],
        leaves_count: usize,
    ) -> bool
    where
        B: IsMerkleTreeBackend<Node = T>,
    {
        use std::collections::HashMap;

        if pos_list.len() != values.len() {
            return false;
        }

        // Build initial level: hash all leaves and map pos -> hash
        let mut current_level: HashMap<usize, T> = pos_list
            .iter()
            .zip(values.iter())
            .map(|(&pos, value)| (pos, B::hash_data(value)))
            .collect();

        let mut path_iter = self.path.iter();
        let mut level_size = leaves_count;

        // Process level by level, same structure as get_batch_auth_path_positions
        while level_size > 1 {
            let mut next_level: HashMap<usize, T> = HashMap::new();

            // Sort positions for deterministic processing
            let mut positions: Vec<usize> = current_level.keys().cloned().collect();
            positions.sort_unstable();

            for pos in positions {
                let hash = current_level.get(&pos).unwrap();
                let sibling_pos = pos ^ 1;
                let parent_pos = pos / 2;

                // Skip if we already computed this parent
                if next_level.contains_key(&parent_pos) {
                    continue;
                }

                // Get sibling: either from current level or from proof path
                let sibling = if let Some(s) = current_level.get(&sibling_pos) {
                    s.clone()
                } else {
                    match path_iter.next() {
                        Some(s) => s.clone(),
                        None => return false,
                    }
                };

                // Hash parent with correct ordering (left child is even index)
                let parent_hash = if pos % 2 == 0 {
                    B::hash_new_parent(hash, &sibling)
                } else {
                    B::hash_new_parent(&sibling, hash)
                };

                next_level.insert(parent_pos, parent_hash);
            }

            current_level = next_level;
            level_size /= 2;
        }

        // Should have exactly the root
        current_level.len() == 1 && current_level.get(&0).map_or(false, |h| h == root_hash)
    }
}

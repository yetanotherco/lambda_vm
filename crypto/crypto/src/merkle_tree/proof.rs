use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use math::traits::Serializable;
use math::{errors::DeserializationError, traits::Deserializable};
use std::collections::{BTreeMap, HashSet};

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
    ///
    /// Uses the same internal indexing as the tree (root=0, leaves at end).
    /// This mirrors the logic of `get_batch_auth_path_positions` exactly.
    ///
    /// # Arguments
    /// * `root_hash` - The expected Merkle root
    /// * `pos_list` - Leaf positions (0-indexed from left)  
    /// * `values` - The leaf values at those positions
    /// * `leaves_count` - Total number of leaves in the tree (must be a power of 2)
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
        use super::utils::{get_parent_pos, get_sibiling_pos};

        if pos_list.len() != values.len() || pos_list.is_empty() {
            return false;
        }

        let first_leaf_index = leaves_count - 1;
        let tree_size = 2 * leaves_count - 1;
        let num_levels = (tree_size as f32).log2().ceil() as usize;

        // Convert leaf positions to internal tree indices and hash values
        // This mirrors: pos_list.map(|pos| pos + self.nodes.len() / 2) in generation
        let mut obtainable: HashSet<usize> = pos_list
            .iter()
            .map(|&pos| pos + first_leaf_index)
            .collect();

        let mut known_hashes: BTreeMap<usize, T> = pos_list
            .iter()
            .zip(values.iter())
            .map(|(&pos, value)| (pos + first_leaf_index, B::hash_data(value)))
            .collect();

        let mut proof_iter = self.path.iter();

        // Process level by level, same as get_batch_auth_path_positions
        for _ in 0..num_levels - 1 {
            let mut parent_obtainable: HashSet<usize> = HashSet::new();
            let mut new_hashes: BTreeMap<usize, T> = BTreeMap::new();

            // Process positions in descending order 
            let positions: Vec<usize> = obtainable.iter().cloned().collect();
            let mut positions_sorted: Vec<usize> = positions;
            positions_sorted.sort_unstable_by(|a, b| b.cmp(a)); 
            for pos in positions_sorted {
                let sibling_pos = get_sibiling_pos(pos);
                let parent_pos = get_parent_pos(pos);

                // Skip if parent already computed (sibling was processed first)
                if new_hashes.contains_key(&parent_pos) {
                    continue;
                }

                let my_hash = known_hashes.get(&pos).unwrap();

                // Get sibling hash: from known or from proof
                let sibling_hash = if obtainable.contains(&sibling_pos) {
                    known_hashes.get(&sibling_pos).unwrap().clone()
                } else {
                    match proof_iter.next() {
                        Some(h) => h.clone(),
                        None => return false,
                    }
                };

                let parent_hash = if pos % 2 == 1 {
                    B::hash_new_parent(my_hash, &sibling_hash)
                } else {
                    B::hash_new_parent(&sibling_hash, my_hash)
                };

                new_hashes.insert(parent_pos, parent_hash);
                parent_obtainable.insert(parent_pos);
            }

            obtainable = parent_obtainable;
            known_hashes = new_hashes;
        }

        // Verify: root computed correctly and all proof nodes consumed
        proof_iter.next().is_none()
            && known_hashes.len() == 1
            && known_hashes.get(&0).map_or(false, |h| h == root_hash)
    }
}

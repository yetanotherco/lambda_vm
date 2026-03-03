use alloc::{collections::BTreeMap, vec::Vec};
#[cfg(feature = "alloc")]
use math::traits::Serializable;
use math::{errors::DeserializationError, traits::Deserializable};

use super::{
    traits::IsMerkleTreeBackend,
    utils::{child_position, parent_index},
};

/// Stores a merkle path to some leaf.
/// Each level contains `arity - 1` sibling nodes.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Proof<T: PartialEq + Eq> {
    pub merkle_path: Vec<Vec<T>>,
}

impl<T: PartialEq + Eq + Clone> Proof<T> {
    /// Verifies a Merkle inclusion proof for the value contained at leaf index.
    pub fn verify<B>(&self, root_hash: &B::Node, index: usize, value: &B::Data) -> bool
    where
        B: IsMerkleTreeBackend<Node = T>,
    {
        let arity = B::ARITY;
        let mut hashed_value = B::hash_data(value);

        // Convert leaf index to internal tree index
        // For a tree with N leaves, internal nodes = (N-1)/(arity-1)
        // We need the tree index to compute child position correctly
        let num_levels = self.merkle_path.len();
        // Compute num_leaves from tree depth: arity^depth
        let num_leaves = arity.pow(num_levels as u32);
        let internal_nodes = if num_leaves > 1 {
            (num_leaves - 1) / (arity - 1)
        } else {
            0
        };
        let mut tree_index = index + internal_nodes;

        for siblings in self.merkle_path.iter() {
            let pos = child_position(tree_index, arity);
            let mut children = Vec::with_capacity(arity);
            let mut sibling_idx = 0;
            for i in 0..arity {
                if i == pos {
                    children.push(hashed_value.clone());
                } else {
                    children.push(siblings[sibling_idx].clone());
                    sibling_idx += 1;
                }
            }
            hashed_value = B::hash_new_parent(&children);
            tree_index = parent_index(tree_index, arity);
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
            .flat_map(|siblings| siblings.iter().flat_map(|node| node.serialize()))
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
            merkle_path.push(vec![node]);
        }
        Ok(Self { merkle_path })
    }
}

/// Stores all the nodes needed to prove the inclusion of multiple leaves.
///
/// # Proof Ordering
/// The `path` contains nodes ordered level-by-level from bottom to top,
/// ascending within each level.
#[derive(Debug, Clone)]
pub struct BatchProof<T: PartialEq + Eq> {
    pub path: Vec<T>,
}

impl<T: PartialEq + Eq + Clone> BatchProof<T> {
    /// Verifies a batch Merkle proof for multiple leaves.
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

        let arity = B::ARITY;
        let internal_nodes = if num_leaves > 1 {
            (num_leaves - 1) / (arity - 1)
        } else {
            0
        };

        // Build map of tree_index -> hashed value
        let mut known_nodes: BTreeMap<usize, T> = BTreeMap::new();
        for (&pos, value) in pos_list.iter().zip(values.iter()) {
            let tree_index = pos + internal_nodes;
            let hashed_value = B::hash_data(value);

            if let Some(existing) = known_nodes.get(&tree_index) {
                if existing != &hashed_value {
                    return false;
                }
            } else {
                known_nodes.insert(tree_index, hashed_value);
            }
        }

        let mut proof_iter = self.path.iter();
        let num_levels = compute_depth_for_verify(num_leaves, arity);

        for _ in 0..num_levels {
            let mut next_known: BTreeMap<usize, T> = BTreeMap::new();

            // Collect all parent groups we need to compute
            let parent_groups: BTreeMap<usize, Vec<usize>> = {
                let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                for &idx in known_nodes.keys() {
                    let p = parent_index(idx, arity);
                    groups.entry(p).or_default().push(idx);
                }
                groups
            };

            for (parent, _known_children) in &parent_groups {
                if next_known.contains_key(parent) {
                    continue;
                }

                let first_child = parent * arity + 1;
                let mut children = Vec::with_capacity(arity);
                for i in 0..arity {
                    let child_idx = first_child + i;
                    if let Some(val) = known_nodes.get(&child_idx) {
                        children.push(val.clone());
                    } else {
                        match proof_iter.next() {
                            Some(h) => children.push(h.clone()),
                            None => return false,
                        }
                    }
                }

                let parent_hash = B::hash_new_parent(&children);
                next_known.insert(*parent, parent_hash);
            }

            known_nodes = next_known;
        }

        proof_iter.next().is_none()
            && known_nodes.len() == 1
            && known_nodes.get(&0) == Some(root_hash)
    }
}

fn compute_depth_for_verify(num_leaves: usize, arity: usize) -> usize {
    if num_leaves <= 1 {
        return 0;
    }
    let mut depth = 0;
    let mut n = num_leaves;
    while n > 1 {
        n /= arity;
        depth += 1;
    }
    depth
}

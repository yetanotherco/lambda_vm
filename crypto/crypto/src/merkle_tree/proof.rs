use alloc::{collections::BTreeMap, vec, vec::Vec};
#[cfg(feature = "alloc")]
use math::traits::Serializable;
use math::{errors::DeserializationError, traits::Deserializable};

use super::traits::IsMerkleTreeBackend;

/// Stores a Merkle path to some leaf.
/// Each level stores the `arity - 1` sibling hashes needed to reconstruct the parent.
/// The path is ordered from leaf level to root.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Proof<T: PartialEq + Eq> {
    pub merkle_path: Vec<Vec<T>>,
}

impl<T: PartialEq + Eq + Clone> Proof<T> {
    /// Verifies a Merkle inclusion proof for the value contained at leaf index.
    pub fn verify<B>(&self, root_hash: &B::Node, mut index: usize, value: &B::Data) -> bool
    where
        B: IsMerkleTreeBackend<Node = T>,
    {
        let arity = B::ARITY;
        let mut hashed_value = B::hash_data(value);

        for siblings in self.merkle_path.iter() {
            if siblings.len() != arity - 1 {
                return false;
            }
            let pos = index % arity;
            let mut children: Vec<T> = Vec::with_capacity(arity);
            let mut sib_idx = 0;
            for j in 0..arity {
                if j == pos {
                    children.push(hashed_value.clone());
                } else {
                    children.push(siblings[sib_idx].clone());
                    sib_idx += 1;
                }
            }
            hashed_value = B::hash_new_parent(&children);
            index /= arity;
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
        // Deserialize assuming the default sibling count.
        // For backward-compatible deserialization, use `deserialize_with_arity`.
        // Default: binary tree (1 sibling per level, 8 bytes per node).
        let mut merkle_path = Vec::new();
        let mut chunks = bytes.chunks(8);
        while let Some(chunk) = chunks.next() {
            let node = T::deserialize(chunk)?;
            merkle_path.push(vec![node]);
        }
        Ok(Self { merkle_path })
    }
}

/// Stores all the nodes needed to prove the inclusion of multiple leaves.
///
/// # Proof Ordering
/// The `path` contains the nodes ordered **level by level from bottom to top**:
/// - Leaf-level siblings come first, root-level siblings come last
/// - Within each level, nodes are in **ascending tree index** (left to right)
///
/// This ordering is critical for verification, which processes levels bottom-to-top
/// and consumes children left-to-right within each group.
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
        let num_leaves = next_power_of(num_leaves, arity);
        let internal_nodes = (num_leaves - 1) / (arity - 1);

        let mut current_level_known_nodes: BTreeMap<usize, T> = BTreeMap::new();
        for (&pos, value) in pos_list.iter().zip(values.iter()) {
            let tree_index = pos + internal_nodes;
            let hashed_value = B::hash_data(value);

            if let Some(existing) = current_level_known_nodes.get(&tree_index) {
                if existing != &hashed_value {
                    return false;
                }
            } else {
                current_level_known_nodes.insert(tree_index, hashed_value);
            }
        }

        let mut proof_path_iter = self.path.iter();
        let num_levels = compute_depth(num_leaves, arity);

        for _ in 0..num_levels {
            let mut next_level_known_nodes: BTreeMap<usize, T> = BTreeMap::new();

            for (pos, _value) in current_level_known_nodes.iter() {
                let parent_pos = (*pos - 1) / arity;

                if next_level_known_nodes.contains_key(&parent_pos) {
                    continue;
                }

                let first_child = arity * parent_pos + 1;
                let mut children = Vec::with_capacity(arity);

                for i in 0..arity {
                    let child_idx = first_child + i;
                    if let Some(hash) = current_level_known_nodes.get(&child_idx) {
                        children.push(hash.clone());
                    } else {
                        match proof_path_iter.next() {
                            Some(h) => children.push(h.clone()),
                            None => return false,
                        }
                    }
                }

                let parent_hash = B::hash_new_parent(&children);
                next_level_known_nodes.insert(parent_pos, parent_hash);
            }
            current_level_known_nodes = next_level_known_nodes;
        }

        proof_path_iter.next().is_none()
            && current_level_known_nodes.len() == 1
            && (current_level_known_nodes.get(&0) == Some(root_hash))
    }
}

/// Compute tree depth = log_arity(num_leaves)
fn compute_depth(num_leaves: usize, arity: usize) -> usize {
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

/// Rounds `n` up to the next power of `base`.
fn next_power_of(n: usize, base: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p *= base;
    }
    p
}

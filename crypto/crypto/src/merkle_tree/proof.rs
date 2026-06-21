use alloc::{collections::BTreeMap, vec::Vec};
#[cfg(feature = "alloc")]
use math::traits::Serializable;
use math::{errors::DeserializationError, traits::Deserializable};

use super::{
    traits::IsMerkleTreeBackend,
    utils::{get_parent_pos, get_sibling_pos},
};

/// Stores a merkle path to some leaf.
/// Internally, the necessary hashes are stored from root to leaf in the
/// `merkle_path` field, in such a way that, if the merkle tree is of height `n`, the
/// `i`-th element of `merkle_path` is the sibling node in the `n - 1 - i`-th check
/// when verifying.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct Proof<T: PartialEq + Eq> {
    pub merkle_path: Vec<T>,
}

/// Verifies a Merkle inclusion proof given the authentication path as a borrowed
/// slice. Shared by [`Proof::verify`] (owned) and the zero-copy verifier (which
/// reads the path straight from an rkyv-archived proof buffer) so both compute
/// the identical root.
pub fn verify_merkle_path<B>(
    merkle_path: &[B::Node],
    root_hash: &B::Node,
    mut index: usize,
    value: &B::Data,
) -> bool
where
    B: IsMerkleTreeBackend,
{
    let mut hashed_value = B::hash_data(value);

    for sibling_node in merkle_path.iter() {
        if index.is_multiple_of(2) {
            hashed_value = B::hash_new_parent(&hashed_value, sibling_node);
        } else {
            hashed_value = B::hash_new_parent(sibling_node, &hashed_value);
        }

        index >>= 1;
    }

    root_hash == &hashed_value
}

/// Like [`verify_merkle_path`], but takes the leaf value as a borrowed slice of
/// field elements hashed via [`FieldElementVectorBackend::hash_data_slice`],
/// producing the identical root to the `Vec`-leaf path. Lets the verifier hash
/// openings straight from borrowed (e.g. zero-copy archived) slices without
/// materializing a `Vec` per opening.
pub fn verify_merkle_path_fe_slice<F, D, const NUM_BYTES: usize>(
    merkle_path: &[[u8; NUM_BYTES]],
    root_hash: &[u8; NUM_BYTES],
    mut index: usize,
    value: &[math::field::element::FieldElement<F>],
) -> bool
where
    F: math::field::traits::IsField,
    D: digest::Digest,
    math::field::element::FieldElement<F>: math::traits::ByteConversion,
    [u8; NUM_BYTES]: From<digest::Output<D>>,
{
    use super::backends::field_element_vector::FieldElementVectorBackend;
    let mut hashed_value = FieldElementVectorBackend::<F, D, NUM_BYTES>::hash_data_slice(value);

    for sibling_node in merkle_path.iter() {
        if index.is_multiple_of(2) {
            hashed_value = FieldElementVectorBackend::<F, D, NUM_BYTES>::hash_new_parent(
                &hashed_value,
                sibling_node,
            );
        } else {
            hashed_value = FieldElementVectorBackend::<F, D, NUM_BYTES>::hash_new_parent(
                sibling_node,
                &hashed_value,
            );
        }

        index >>= 1;
    }

    root_hash == &hashed_value
}

/// Keccak256-specialized form of [`verify_merkle_path_fe_slice`] that hashes via
/// the single-block [`keccak256_single_block`](crate::hash::keccak256::keccak256_single_block)
/// sponge instead of the generic `sha3` streaming wrapper. Produces the identical
/// Keccak256 root — a transparent implementation swap — but the leaf and each
/// parent hash skip `sha3`'s `block_buffer` and run the permutation as a single
/// `keccak::f1600` (the `KeccakPermute` precompile on the guest).
///
/// `value` is the leaf's field elements (serialized big-endian, matching the
/// backend's `hash_data_slice`); `merkle_path` are the 32-byte sibling nodes.
pub fn verify_merkle_path_keccak256<F>(
    merkle_path: &[[u8; 32]],
    root_hash: &[u8; 32],
    mut index: usize,
    value: &[math::field::element::FieldElement<F>],
) -> bool
where
    F: math::field::traits::IsField,
    math::field::element::FieldElement<F>: math::traits::ByteConversion,
{
    use crate::hash::keccak256::{keccak256, keccak256_single_block};
    use alloc::vec::Vec;
    use math::traits::ByteConversion;

    // Leaf: serialize the field elements big-endian (matching
    // `FieldElementVectorBackend::hash_data_slice`) and hash. The leaf can be wide
    // (e.g. a 1480-column trace row), so use the multi-block sponge here. This is
    // hashed once per path; the per-level parent hashing below dominates.
    let mut leaf_bytes: Vec<u8> = Vec::new();
    for element in value.iter() {
        leaf_bytes.extend_from_slice(element.to_bytes_be().as_ref());
    }
    let mut hashed_value = keccak256(&leaf_bytes);

    // Each internal node hashes the 64-byte concatenation of the two children —
    // always a single rate block, so the fast single-block path is exact.
    let mut pair = [0u8; 64];
    for sibling_node in merkle_path.iter() {
        if index.is_multiple_of(2) {
            pair[..32].copy_from_slice(&hashed_value);
            pair[32..].copy_from_slice(sibling_node);
        } else {
            pair[..32].copy_from_slice(sibling_node);
            pair[32..].copy_from_slice(&hashed_value);
        }
        hashed_value = keccak256_single_block(&pair);
        index >>= 1;
    }

    root_hash == &hashed_value
}

impl<T: PartialEq + Eq> Proof<T> {
    /// Verifies a Merkle inclusion proof for the value contained at leaf index.
    pub fn verify<B>(&self, root_hash: &B::Node, index: usize, value: &B::Data) -> bool
    where
        B: IsMerkleTreeBackend<Node = T>,
    {
        verify_merkle_path::<B>(&self.merkle_path, root_hash, index, value)
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

/// Stores all the nodes needed to prove the inclusion of multiple leaves.
///
/// # Proof Ordering
/// The `path` contains the nodes in **descending order by tree index**:
/// - Higher indices (closer to leaves) come first
/// - Lower indices (closer to root) come last
/// - Within the same level, nodes are ordered from right to left (higher index first)
///
/// This ordering is critical for verification, which consumes nodes in the same order
/// as they were generated by `get_batch_proof`.
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
    /// * `pos_list` - Leaf positions (0-indexed from left to right)
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

        // Build map of `position → hashed value`, validating that duplicate positions have the same value.
        // Since the nodes in the tree are indexed from the root to the leaves, we need to redefine the
        // given indices of the leaves.
        // We also need to hash all the given leaf values.
        // BTreeMap always maintains elements in ascending order, so here the leaves are ordered from
        // left (smaller index) to right (larger index).
        let mut current_level_known_nodes: BTreeMap<usize, T> = BTreeMap::new();
        for (&pos, value) in pos_list.iter().zip(values.iter()) {
            let tree_index = pos + first_leaf_index;
            let hashed_value = B::hash_data(value);

            if let Some(existing) = current_level_known_nodes.get(&tree_index) {
                // Duplicate position: values must be the same
                if existing != &hashed_value {
                    return false;
                }
                // Same value, skip (deduplicate)
            } else {
                current_level_known_nodes.insert(tree_index, hashed_value);
            }
        }

        let mut proof_path_iter = self.path.iter();

        let num_levels = (2 * num_leaves).ilog2();
        // Process level by level, from bottom to top, same as `get_batch_auth_path_positions`.
        for _ in 0..num_levels - 1 {
            let mut next_level_known_nodes: BTreeMap<usize, T> = BTreeMap::new();

            // Process each known node from right to left to match the order of the proof.
            // Since in `current_level_known_nodes` the nodes are ordered from left to right we take `.rev()`.
            for (pos, value) in current_level_known_nodes.iter().rev() {
                let parent_pos = get_parent_pos(*pos);

                // Skip if parent was already computed (i.e. sibling was processed first).
                if next_level_known_nodes.contains_key(&parent_pos) {
                    continue;
                }

                // Get sibling position (None only for root, which shouldn't appear here)
                let Some(sibling_pos) = get_sibling_pos(*pos) else {
                    continue;
                };

                // Get sibling value: from known nodes or from proof path.
                let sibling_hash = if let Some(hash) = current_level_known_nodes.get(&sibling_pos) {
                    hash
                } else {
                    match proof_path_iter.next() {
                        Some(h) => h,
                        None => return false,
                    }
                };

                // Compute parent hash.
                let parent_hash = if pos % 2 == 1 {
                    B::hash_new_parent(value, sibling_hash)
                } else {
                    B::hash_new_parent(sibling_hash, value)
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

use alloc::{collections::BTreeMap, vec::Vec};
#[cfg(feature = "alloc")]
use math::traits::Serializable;
use math::{errors::DeserializationError, traits::Deserializable};

use super::{
    traits::IsMerkleTreeBackend,
    utils::{get_parent_pos_arity, sibling_indices},
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
    let arity = B::ARITY;
    let mut hashed_value = B::hash_data(value);

    // The path stores `arity - 1` siblings per level, in ascending sibling-index
    // order (as produced by `build_merkle_path`). At each level the running hash
    // occupies slot `index % arity` among its `arity` siblings; rebuild that slot
    // group and hash all `arity` children into the parent.
    let mut group: Vec<B::Node> = Vec::with_capacity(arity);
    for level_siblings in merkle_path.chunks(arity - 1) {
        let slot = index % arity;
        group.clear();
        let mut sib = level_siblings.iter();
        for s in 0..arity {
            if s == slot {
                group.push(hashed_value.clone());
            } else {
                // `level_siblings` are in ascending index order, i.e. the children
                // other than `slot` taken left to right — exactly the fill order.
                group.push(sib.next().expect("path has arity-1 siblings").clone());
            }
        }
        hashed_value = B::hash_children(&group);
        index /= arity;
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
/// `ARITY` is the tree branching factor (matching the backend). Each internal
/// node concatenates its `ARITY` children's 32-byte hashes (running hash inserted
/// at its `index % ARITY` slot, the rest filled from `merkle_path` in order) and
/// hashes them; for `ARITY <= 4` that concatenation is `<= 128` bytes, a single
/// keccak block. The path stores `ARITY - 1` siblings per level in ascending slot
/// order, matching `build_merkle_path`.
///
/// `value` is the leaf's field elements (serialized big-endian, matching the
/// backend's `hash_data_slice`); `merkle_path` are the 32-byte sibling nodes.
pub fn verify_merkle_path_keccak256<F, const ARITY: usize>(
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

    // Each internal node hashes the concatenation of its `ARITY` children's
    // 32-byte hashes (`ARITY * 32 <= 128` bytes for ARITY <= 4 — a single keccak
    // block). The running hash sits at slot `index % ARITY`; the other slots are
    // filled left-to-right from this level's `ARITY - 1` path siblings.
    let mut concat = [0u8; 4 * 32];
    debug_assert!(ARITY <= 4, "single-block node hashing supports ARITY <= 4");
    let node_bytes = ARITY * 32;
    for level_siblings in merkle_path.chunks(ARITY - 1) {
        let slot = index % ARITY;
        let mut sib = level_siblings.iter();
        for s in 0..ARITY {
            let src = if s == slot {
                &hashed_value
            } else {
                sib.next().expect("path has ARITY-1 siblings per level")
            };
            concat[s * 32..(s + 1) * 32].copy_from_slice(src);
        }
        hashed_value = keccak256_single_block(&concat[..node_bytes]);
        index /= ARITY;
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
                // Batch verification is binary-only (mirrors `get_batch_proof`).
                let parent_pos = get_parent_pos_arity(*pos, 2);

                // Skip if parent was already computed (i.e. sibling was processed first).
                if next_level_known_nodes.contains_key(&parent_pos) {
                    continue;
                }

                // Get sibling position (None only for root, which shouldn't appear here)
                let Some(sibling_pos) = sibling_indices(*pos, 2).into_iter().next() else {
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

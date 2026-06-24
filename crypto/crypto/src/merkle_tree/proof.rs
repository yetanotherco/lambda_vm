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
    index: usize,
    value: &[math::field::element::FieldElement<F>],
) -> bool
where
    F: math::field::traits::IsField,
    math::field::element::FieldElement<F>: math::traits::ByteConversion,
{
    let mut scratch: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
    verify_merkle_path_keccak256_with_scratch::<F, ARITY>(
        merkle_path,
        root_hash,
        index,
        value,
        &mut scratch,
    )
}

/// Like [`verify_merkle_path_keccak256`] but takes a caller-owned `leaf_scratch`
/// buffer that is reused across calls to avoid per-invocation allocation.
/// The buffer is cleared and refilled on each call; the caller should keep it
/// alive across the query loop.
pub fn verify_merkle_path_keccak256_with_scratch<F, const ARITY: usize>(
    merkle_path: &[[u8; 32]],
    root_hash: &[u8; 32],
    mut index: usize,
    value: &[math::field::element::FieldElement<F>],
    leaf_scratch: &mut alloc::vec::Vec<u8>,
) -> bool
where
    F: math::field::traits::IsField,
    math::field::element::FieldElement<F>: math::traits::ByteConversion,
{
    use crate::hash::keccak256::{
        keccak256_field_elements_direct, keccak256_field_elements_streaming, keccak256_four_nodes,
        keccak256_two_nodes,
    };
    use math::traits::ByteConversion;
    // Keccak-256 rate in bytes.
    const RATE: usize = 136;

    // Leaf hash: for lane-aligned element sizes (BYTE_LEN % 8 == 0), absorb
    // directly into keccak state lanes without any intermediate byte buffer.
    // - Small leaves (< RATE bytes): single-block direct path.
    // - Wide leaves (≥ RATE bytes): streaming multi-block path — still no Vec.
    // The `leaf_scratch` Vec parameter is retained for callers that pass it but
    // the fast paths never write to it.
    let elem_bytes = <math::field::element::FieldElement<F>>::BYTE_LEN;
    let total_bytes = value.len() * elem_bytes;
    let mut hashed_value = if elem_bytes % 8 == 0 {
        if total_bytes < RATE {
            keccak256_field_elements_direct::<F>(value)
        } else {
            keccak256_field_elements_streaming::<F>(value)
        }
    } else {
        // Non-lane-aligned elements (rare): fall back to the scratch-buffer path.
        use crate::hash::keccak256::keccak256;
        leaf_scratch.clear();
        for element in value.iter() {
            leaf_scratch.extend_from_slice(element.to_bytes_be().as_ref());
        }
        keccak256(leaf_scratch)
    };

    // Each internal node hashes ARITY×32 bytes (≤ 128 for ARITY≤4 = one keccak
    // block). Collect children into a stack array, then dispatch to the
    // specialized no-buffer hash function that builds the keccak state directly
    // from lanes — no intermediate 136-byte copy.
    debug_assert!(ARITY <= 4, "single-block node hashing supports ARITY <= 4");
    let mut children = [[0u8; 32]; 4];

    for level_siblings in merkle_path.chunks(ARITY - 1) {
        let slot = index % ARITY;
        let mut sib = level_siblings.iter();
        for s in 0..ARITY {
            let src = if s == slot {
                &hashed_value
            } else {
                sib.next().expect("path has ARITY-1 siblings per level")
            };
            children[s] = *src;
        }
        hashed_value = if ARITY == 2 {
            keccak256_two_nodes(&children[0], &children[1])
        } else {
            keccak256_four_nodes(&children)
        };
        index /= ARITY;
    }

    root_hash == &hashed_value
}

/// Verify TWO Merkle openings at `(index, index+1)` that share the same ARITY-4
/// level-0 group — i.e. `index` is even. Because both leaves sit in the same
/// quaternary node at depth-0 of the tree, the level-0 parent hash and all
/// ancestor hashes are identical; this function:
///
/// 1. Hashes each leaf once (`value_a` at `index`, `value_b` at `index+1`).
/// 2. Assembles the level-0 group of 4 using 2 path siblings and the 2 leaf
///    hashes, then hashes once to get the shared ancestor.
/// 3. Walks the remaining `merkle_path[ARITY-1..]` ancestor path exactly once.
///
/// Compared to two independent `verify_merkle_path_keccak256` calls this saves:
/// - one full leaf serialization + keccak pass (both leaves still hashed once each)
/// - all duplicate ancestor-node hashes from depth-1 to the root
///
/// **Precondition**: `index` must be even and both leaves must be in the same
/// level-0 ARITY-4 group (`index / ARITY == (index+1) / ARITY`). This is always
/// true when called with `index = iota * 2` for any `iota`.
///
/// `merkle_path` must contain `(ARITY - 1)` siblings per level, same layout as
/// `verify_merkle_path_keccak256`.
///
/// `leaf_scratch` is a caller-owned byte buffer reused for leaf serialization.
pub fn verify_paired_keccak256_openings<F, const ARITY: usize>(
    merkle_path: &[[u8; 32]],
    root_hash: &[u8; 32],
    index: usize,
    value_a: &[math::field::element::FieldElement<F>],
    value_b: &[math::field::element::FieldElement<F>],
    leaf_scratch: &mut alloc::vec::Vec<u8>,
) -> bool
where
    F: math::field::traits::IsField,
    math::field::element::FieldElement<F>: math::traits::ByteConversion,
{
    use crate::hash::keccak256::{
        keccak256_field_elements_direct, keccak256_field_elements_streaming, keccak256_four_nodes,
        keccak256_two_nodes,
    };
    use math::traits::ByteConversion;

    debug_assert_eq!(index % 2, 0, "index must be even for paired opening");
    debug_assert!(ARITY <= 4, "single-block node hashing supports ARITY <= 4");
    // Both leaves must be in the same level-0 group.
    debug_assert_eq!(index / ARITY, (index + 1) / ARITY);

    // Keccak rate for 256-bit output.
    const RATE: usize = 136;

    let elem_bytes = <math::field::element::FieldElement<F>>::BYTE_LEN;
    let total_bytes = value_a.len() * elem_bytes;

    // Hash both leaves using the lane-direct path for aligned elements — no Vec.
    let (hash_a, hash_b) = if elem_bytes % 8 == 0 {
        if total_bytes < RATE {
            (
                keccak256_field_elements_direct::<F>(value_a),
                keccak256_field_elements_direct::<F>(value_b),
            )
        } else {
            (
                keccak256_field_elements_streaming::<F>(value_a),
                keccak256_field_elements_streaming::<F>(value_b),
            )
        }
    } else {
        use crate::hash::keccak256::keccak256;
        leaf_scratch.clear();
        for element in value_a.iter() {
            leaf_scratch.extend_from_slice(element.to_bytes_be().as_ref());
        }
        let ha = keccak256(leaf_scratch);
        leaf_scratch.clear();
        for element in value_b.iter() {
            leaf_scratch.extend_from_slice(element.to_bytes_be().as_ref());
        }
        let hb = keccak256(leaf_scratch);
        (ha, hb)
    };

    // Assemble the level-0 group of ARITY children.
    //
    // `merkle_path` is the authentication path for leaf `index`. At depth-0 it
    // stores the `ARITY-1` siblings of `index` in ascending slot order.  Among
    // those siblings is the leaf at `index+1` (slot_b) — we are computing that
    // hash ourselves as `hash_b`, so we must SKIP the corresponding entry in the
    // path.
    //
    // The path at depth-0 lists all slots `< slot_a` and `> slot_a` in ascending
    // order (excluding slot_a, which is the leaf itself).  The entry for `slot_b`
    // is therefore at path position `slot_b - 1` (because `slot_a` is skipped and
    // `slot_b = slot_a + 1` so there are exactly `slot_b - 1` entries before it).
    //
    // In practice slot_a is always 0 or 2 (for even `index` with ARITY=4):
    //   iota even  → slot_a=0, slot_b=1 → path[0..3] = [hash_1,hash_2,hash_3]
    //                                       skip path[0] (=hash_b), use path[1..2]
    //   iota odd   → slot_a=2, slot_b=3 → path[0..3] = [hash_0,hash_1,hash_3]
    //                                       skip path[2] (=hash_b), use path[0..1]
    let slot_a = index % ARITY;
    let slot_b = slot_a + 1;
    // Rank of slot_b in the path for slot_a (0-based, skipping slot_a): path
    // entries are all slots != slot_a in ascending order, so slot_b (> slot_a)
    // appears at rank `slot_b - 1`.
    let slot_b_path_rank = slot_b - 1; // slot_a positions before slot_b is just slot_a itself

    // Collect ARITY children into a fixed-size array for dispatch to specialized hash.
    let mut children = [[0u8; 32]; 4];

    // Level-0: assemble the group from hash_a, hash_b, and ARITY-2 path siblings.
    {
        let level0_path = &merkle_path[..ARITY - 1];
        let mut path_pos = 0usize;
        for s in 0..ARITY {
            let src: &[u8; 32] = if s == slot_a {
                &hash_a
            } else if s == slot_b {
                &hash_b
            } else {
                if path_pos == slot_b_path_rank {
                    path_pos += 1;
                }
                let entry = &level0_path[path_pos];
                path_pos += 1;
                entry
            };
            children[s] = *src;
        }
    }

    // Hash using no-buffer specialized function.
    let mut hashed_value = if ARITY == 2 {
        keccak256_two_nodes(&children[0], &children[1])
    } else {
        keccak256_four_nodes(&children)
    };
    let mut ancestor_index = index / ARITY;

    // Walk ancestor path (depth 1 and above).
    for level_siblings in merkle_path[ARITY - 1..].chunks(ARITY - 1) {
        let slot = ancestor_index % ARITY;
        let mut sib = level_siblings.iter();
        for s in 0..ARITY {
            let src = if s == slot {
                &hashed_value
            } else {
                sib.next().expect("path has ARITY-1 siblings per level")
            };
            children[s] = *src;
        }
        hashed_value = if ARITY == 2 {
            keccak256_two_nodes(&children[0], &children[1])
        } else {
            keccak256_four_nodes(&children)
        };
        ancestor_index /= ARITY;
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

#[cfg(test)]
mod tests {
    use alloc::vec;
    use math::field::{element::FieldElement, goldilocks::GoldilocksField};

    use crate::merkle_tree::{backends::types::BatchKeccak256Backend, merkle::MerkleTree};

    type F = GoldilocksField;
    type FE = FieldElement<F>;
    type Backend = BatchKeccak256Backend<F>;

    /// Build a quaternary (ARITY=4) Keccak256 tree with `n` leaves (each a single
    /// field element) and return (tree, leaves).
    fn build_tree(n: usize) -> (MerkleTree<Backend>, alloc::vec::Vec<alloc::vec::Vec<FE>>) {
        let leaves: alloc::vec::Vec<alloc::vec::Vec<FE>> =
            (0..n).map(|i| vec![FE::from(i as u64 + 1)]).collect();
        let tree = MerkleTree::<Backend>::build(&leaves).unwrap();
        (tree, leaves)
    }

    /// `verify_paired_keccak256_openings` must agree with two independent
    /// `verify_merkle_path_keccak256` calls for every (even, even+1) pair.
    #[test]
    fn paired_opening_matches_two_independent_openings() {
        // Build a tree with 16 leaves (quaternary depth-2).
        let (tree, leaves) = build_tree(16);

        for iota in 0..8usize {
            let index = iota * 2;
            let index_sym = index + 1;

            let proof_a = tree.get_proof_by_pos(index).unwrap();
            let proof_b = tree.get_proof_by_pos(index_sym).unwrap();

            let value_a = &leaves[index];
            let value_b = &leaves[index_sym];

            // Convert path to [[u8;32]] slices.
            let path_a: alloc::vec::Vec<[u8; 32]> = proof_a.merkle_path.clone();
            let path_b: alloc::vec::Vec<[u8; 32]> = proof_b.merkle_path.clone();
            let root = tree.root;

            // Independent verifications.
            let ok_a = super::verify_merkle_path_keccak256::<F, 4>(&path_a, &root, index, value_a);
            let ok_b =
                super::verify_merkle_path_keccak256::<F, 4>(&path_b, &root, index_sym, value_b);
            assert!(ok_a, "independent verify_a failed for iota={iota}");
            assert!(ok_b, "independent verify_b failed for iota={iota}");

            // Paired verification — uses path_a only.
            let mut scratch = alloc::vec::Vec::new();
            let ok_paired = super::verify_paired_keccak256_openings::<F, 4>(
                &path_a, &root, index, value_a, value_b, &mut scratch,
            );
            assert!(
                ok_paired,
                "paired opening failed for iota={iota} (index={index})"
            );
        }
    }

    /// Paired opening must fail when value_b is wrong.
    #[test]
    fn paired_opening_rejects_wrong_value_b() {
        let (tree, leaves) = build_tree(16);
        let proof_a = tree.get_proof_by_pos(0).unwrap();
        let path_a: alloc::vec::Vec<[u8; 32]> = proof_a.merkle_path.clone();
        let wrong_value_b = vec![FE::from(9999u64)];
        let mut scratch = alloc::vec::Vec::new();
        let ok = super::verify_paired_keccak256_openings::<F, 4>(
            &path_a,
            &tree.root,
            0,
            &leaves[0],
            &wrong_value_b,
            &mut scratch,
        );
        assert!(!ok, "paired opening should fail with wrong value_b");
    }

    /// Paired opening must fail when value_a is wrong.
    #[test]
    fn paired_opening_rejects_wrong_value_a() {
        let (tree, leaves) = build_tree(16);
        let proof_a = tree.get_proof_by_pos(0).unwrap();
        let path_a: alloc::vec::Vec<[u8; 32]> = proof_a.merkle_path.clone();
        let wrong_value_a = vec![FE::from(9999u64)];
        let mut scratch = alloc::vec::Vec::new();
        let ok = super::verify_paired_keccak256_openings::<F, 4>(
            &path_a,
            &tree.root,
            0,
            &wrong_value_a,
            &leaves[1],
            &mut scratch,
        );
        assert!(!ok, "paired opening should fail with wrong value_a");
    }

    /// Test with a tree that requires more depth (64 leaves = depth 3).
    #[test]
    fn paired_opening_works_at_depth_3() {
        let (tree, leaves) = build_tree(64);
        for iota in 0..32usize {
            let index = iota * 2;
            let proof_a = tree.get_proof_by_pos(index).unwrap();
            let path_a: alloc::vec::Vec<[u8; 32]> = proof_a.merkle_path.clone();
            let mut scratch = alloc::vec::Vec::new();
            let ok = super::verify_paired_keccak256_openings::<F, 4>(
                &path_a,
                &tree.root,
                index,
                &leaves[index],
                &leaves[index + 1],
                &mut scratch,
            );
            assert!(ok, "depth-3 paired opening failed for iota={iota}");
        }
    }

    /// Minimal tree: 4 leaves (depth-1, path has only one level-0 group = the root).
    #[test]
    fn paired_opening_works_at_depth_1() {
        let (tree, leaves) = build_tree(4);
        for iota in 0..2usize {
            let index = iota * 2;
            let proof_a = tree.get_proof_by_pos(index).unwrap();
            let path_a: alloc::vec::Vec<[u8; 32]> = proof_a.merkle_path.clone();
            let mut scratch = alloc::vec::Vec::new();
            let ok = super::verify_paired_keccak256_openings::<F, 4>(
                &path_a,
                &tree.root,
                index,
                &leaves[index],
                &leaves[index + 1],
                &mut scratch,
            );
            assert!(ok, "depth-1 paired opening failed for iota={iota}");
        }
    }
}

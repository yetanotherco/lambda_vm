use core::fmt::Display;

use crate::merkle_tree::proof::BatchProof;

use super::{proof::Proof, traits::IsMerkleTreeBackend, utils::*};
use alloc::{collections::BTreeSet, vec::Vec};
#[cfg(feature = "disk-spill")]
use math::spill_safe::SpillSafe;

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

/// File-backed mmap storage for Merkle tree nodes.
///
/// After `spill_nodes_to_disk()`, the in-memory node vector is freed and
/// node access goes through this mmap instead.
#[cfg(feature = "disk-spill")]
pub(crate) struct MmapNodeBacking {
    mmap: memmap2::Mmap,
    node_count: usize,
}

/// The struct for the Merkle tree, consisting of the root and the nodes.
/// A typical tree would look like this
///                 root
///              /        \
///          leaf 12     leaf 34
///        /         \    /      \
///    leaf 1     leaf 2 leaf 3  leaf 4
/// The bottom leafs correspond to the hashes of the elements, while each upper
/// layer contains the hash of the concatenation of the daughter nodes.
#[cfg_attr(not(feature = "disk-spill"), derive(Clone))]
#[cfg_attr(
    all(feature = "serde", not(feature = "disk-spill")),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    all(feature = "serde", feature = "disk-spill"),
    derive(serde::Deserialize)
)]
pub struct MerkleTree<B: IsMerkleTreeBackend> {
    pub root: B::Node,
    nodes: Vec<B::Node>,
    #[cfg(feature = "disk-spill")]
    #[cfg_attr(feature = "serde", serde(skip))]
    mmap_backing: Option<MmapNodeBacking>,
}

// `mmap_backing` is `#[serde(skip)]` and `spill_nodes_to_disk` empties `nodes`,
// so the default derive would emit `{root, nodes: []}` and lose the tree.
//
// Output matches the non-disk-spill derive byte-for-byte, so a proof from either
// storage mode deserializes with the same `Deserialize` impl.
#[cfg(all(feature = "serde", feature = "disk-spill"))]
impl<B: IsMerkleTreeBackend> serde::Serialize for MerkleTree<B>
where
    B::Node: serde::Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("MerkleTree", 2)?;
        s.serialize_field("root", &self.root)?;
        if self.mmap_backing.is_some() {
            s.serialize_field("nodes", &MmapNodesSeq(self))?;
        } else {
            s.serialize_field("nodes", &self.nodes)?;
        }
        s.end()
    }
}

/// Streams the spilled nodes through `serialize_seq` instead of buffering them
/// into a `Vec<B::Node>` the size of the tree.
#[cfg(all(feature = "serde", feature = "disk-spill"))]
struct MmapNodesSeq<'a, B: IsMerkleTreeBackend>(&'a MerkleTree<B>);

#[cfg(all(feature = "serde", feature = "disk-spill"))]
impl<B: IsMerkleTreeBackend> serde::Serialize for MmapNodesSeq<'_, B>
where
    B::Node: serde::Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let backing = self
            .0
            .mmap_backing
            .as_ref()
            .expect("MmapNodesSeq is only constructed when mmap_backing is Some");
        let n = backing.node_count;
        let mut seq = serializer.serialize_seq(Some(n))?;
        for i in 0..n {
            seq.serialize_element(self.0.node_get(i).expect("index in bounds"))?;
        }
        seq.end()
    }
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

        //The leaf must be a power of 2 set
        let hashed_leaves = complete_until_power_of_two(hashed_leaves);
        let leaves_len = hashed_leaves.len();

        //The length of leaves minus one inner node in the merkle tree
        //The first elements are overwritten by build function, it doesn't matter what it's there
        let mut nodes = vec![hashed_leaves[0].clone(); leaves_len - 1];
        nodes.extend(hashed_leaves);

        //Build the inner nodes of the tree
        build::<B>(&mut nodes, leaves_len);

        Some(MerkleTree {
            root: nodes[ROOT].clone(),
            nodes,
            #[cfg(feature = "disk-spill")]
            mmap_backing: None,
        })
    }

    /// Total number of nodes in the tree (inner + leaves).
    fn node_count(&self) -> usize {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            return backing.node_count;
        }
        self.nodes.len()
    }

    /// Access a node by index, returning a reference.
    ///
    /// Returns `None` if `idx` is out of bounds.
    fn node_get(&self, idx: usize) -> Option<&B::Node> {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            if idx < backing.node_count {
                // SAFETY: spill_nodes_to_disk is the only function that populates
                // mmap_backing, and its where-clause requires B::Node: SpillSafe.
                // Reaching this branch means that bound was checked at construction,
                // so B::Node carries no padding and every bit pattern is valid.
                //
                // Alignment: the mmap base is page-aligned (>= 4096), spill_nodes_to_disk
                // asserts align_of::<B::Node>() <= 4096, and Rust guarantees
                // size_of::<B::Node> is a multiple of align_of::<B::Node>, so every
                // offset idx * node_size lands on an aligned address.
                let ptr = unsafe { backing.mmap.as_ptr().add(idx * size_of::<B::Node>()) };
                return Some(unsafe { &*(ptr as *const B::Node) });
            }
            return None;
        }
        self.nodes.get(idx)
    }

    /// Returns a Merkle proof for the element/s at position pos
    /// For example, give me an inclusion proof for the 3rd element in the
    /// Merkle tree
    pub fn get_proof_by_pos(&self, pos: usize) -> Option<Proof<B::Node>> {
        let pos = pos + self.node_count() / 2;
        let Ok(merkle_path) = self.build_merkle_path(pos) else {
            return None;
        };

        self.create_proof(merkle_path)
    }

    /// Creates a proof from a Merkle pasth
    fn create_proof(&self, merkle_path: Vec<B::Node>) -> Option<Proof<B::Node>> {
        Some(Proof { merkle_path })
    }

    /// Returns the Merkle path for the element/s for the leaf at position pos
    fn build_merkle_path(&self, pos: usize) -> Result<Vec<B::Node>, Error> {
        // Pre-allocate based on tree depth (log2 of tree size)
        let tree_depth = (self.node_count() + 1).ilog2() as usize;
        let mut merkle_path = Vec::with_capacity(tree_depth);
        let mut pos = pos;

        while pos != ROOT {
            let Some(node) = self.node_get(sibling_index(pos)) else {
                // out of bounds, exit returning the current merkle_path
                return Err(Error::OutOfBounds);
            };
            merkle_path.push(node.clone());

            pos = parent_index(pos);
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

        let num_leaves = (self.node_count() + 1).div_ceil(2);

        // Validate all positions are within bounds
        for &pos in pos_list {
            if pos >= num_leaves {
                return Err(Error::OutOfBounds);
            }
        }

        // Since the nodes in the merkle tree are indexed from the root to the leaves, we redefine the indices
        // of the leaves.
        let leaf_positions = pos_list
            .iter()
            .map(|pos| pos + self.node_count() / 2)
            .collect::<Vec<usize>>();
        // We get the positions of the nodes for the batch proof.
        let batch_auth_path_positions = self.get_batch_auth_path_positions(&leaf_positions);

        // We get the nodes for the batch proof.
        let batch_auth_path_nodes = batch_auth_path_positions
            .iter()
            .map(|pos| {
                self.node_get(*pos)
                    .expect("batch auth path position in bounds")
                    .clone()
            })
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
        // BTreeSet always maintains elements in ascending order (smaller indices first), regardless of insertion order.
        let mut auth_path_set = BTreeSet::<usize>::new();
        let mut obtainable: BTreeSet<usize> = leaf_positions.iter().cloned().collect();

        // Number of levels in tree
        let num_levels = (self.node_count() + 1).ilog2();

        // Iter lefevel-by-level from leaves to root.
        for _ in 0..num_levels - 1 {
            let mut next_obtainable = BTreeSet::new();

            for &pos in &obtainable {
                // Check sibling (None only for root, which shouldn't appear here)
                if let Some(sibling_pos) = get_sibling_pos(pos) {
                    // If sibling not obtainable, include it in the proof
                    let sibling_is_obtainable =
                        obtainable.contains(&sibling_pos) || auth_path_set.contains(&sibling_pos);

                    if !sibling_is_obtainable {
                        auth_path_set.insert(sibling_pos);
                    }
                }

                // Parent becomes obtainable (computable from both children)
                next_obtainable.insert(get_parent_pos(pos));
            }

            obtainable = next_obtainable;
        }

        // Reverse to get descending order (larger indices first).
        // This makes the proof ordered from bottom (nodes closer to leaves) to top (nodes loser to root).
        auth_path_set.into_iter().rev().collect()
    }

    /// Mmap a temp file, copy the tree nodes into the mapping, and free the
    /// in-memory vector. Node access methods read from the mmap after this call.
    #[cfg(feature = "disk-spill")]
    pub fn spill_nodes_to_disk(&mut self) -> std::io::Result<()>
    where
        B::Node: SpillSafe,
    {
        const {
            assert!(
                align_of::<B::Node>() <= 4096,
                "B::Node alignment must fit within mmap page alignment"
            )
        }

        if self.nodes.is_empty() || self.mmap_backing.is_some() {
            return Ok(());
        }

        let node_count = self.nodes.len();
        let total_bytes = (node_count as u64)
            .checked_mul(size_of::<B::Node>() as u64)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "spill_nodes_to_disk: byte count overflows u64",
                )
            })?;

        let file = tempfile::tempfile()?;
        crate::mmap_util::reserve_file_blocks(&file, total_bytes)?;

        // Write directly through a writable mmap, then downgrade to read-only.
        // Avoids the write(2) → page-cache → mmap hand-off, which on Linux
        // under memory pressure could produce partially-zeroed reads from the
        // read-only mmap.
        //
        // SAFETY: tempfile() creates an anonymous file with no filesystem
        // path, so no other process can open or modify it.
        let mut mmap_mut = unsafe { memmap2::MmapOptions::new().map_mut(&file)? };
        // SAFETY: SpillSafe's safety contract requires no padding on B::Node, so
        // the contiguous Vec bytes are initialized and reading them as &[u8] is sound.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                self.nodes.as_ptr() as *const u8,
                node_count * size_of::<B::Node>(),
            )
        };
        mmap_mut.copy_from_slice(bytes);
        let mmap = mmap_mut.make_read_only()?;

        // Free the heap allocation
        self.nodes = Vec::new();

        self.mmap_backing = Some(MmapNodeBacking { mmap, node_count });

        Ok(())
    }
}

#[cfg(all(test, feature = "serde", feature = "disk-spill"))]
mod disk_spill_serde_tests {
    use super::*;
    use crate::merkle_tree::backends::field_element::FieldElementBackend;
    use math::field::{element::FieldElement, goldilocks::GoldilocksField};
    use sha3::Keccak256;

    type F = GoldilocksField;
    type FE = FieldElement<F>;
    type Backend = FieldElementBackend<F, Keccak256, 32>;

    /// Serializing a spilled MerkleTree must produce identical bytes to
    /// serializing the same tree before spilling, and round-trip back to an
    /// equal tree.
    #[test]
    fn test_serialize_spilled_merkle_tree_matches_unspilled() {
        let values: Vec<FE> = (1..17).map(FE::from).collect();
        let unspilled = MerkleTree::<Backend>::build(&values).expect("build merkle tree");
        let unspilled_bytes = bincode::serialize(&unspilled).expect("serialize unspilled");

        let mut spilled = MerkleTree::<Backend>::build(&values).expect("build merkle tree");
        spilled.spill_nodes_to_disk().expect("spill_nodes_to_disk");
        let spilled_bytes = bincode::serialize(&spilled).expect("serialize spilled");

        assert_eq!(
            spilled_bytes, unspilled_bytes,
            "spilled and unspilled trees must serialize to identical bytes"
        );

        let restored: MerkleTree<Backend> =
            bincode::deserialize(&spilled_bytes).expect("deserialize spilled bytes");
        assert!(restored.mmap_backing.is_none());
        assert_eq!(restored.root, unspilled.root);
    }
}

//! Multi-Matrix Commitment Scheme (MMCS): a single Merkle root that
//! commits to multiple matrices of (different or equal) heights, with
//! one authentication path per query covering all matrices.
//!
//! Plonky3-style layer injection: sort matrices by `padded_height` desc
//! (ties broken by `tag` asc); layer 0 starts with the first max-height
//! matrix's leaves and sequentially compresses in additional max-height
//! matrices; each upper layer compresses pairs of children then injects
//! every matrix whose `padded_height` matches that layer's length.
//!
//! Scope:
//! - Multiple matrices may share a `padded_height` (matches lambda-vm's
//!   chunked-table topology: 3 CPU chunks all at 2^20, BITWISE at 2^20,
//!   etc.). Combination order at a layer is deterministic (tag asc).
//! - No SIMD / parallel hashing yet.
//! - No streaming chunked absorption — caller materializes full leaf
//!   digest arrays per matrix.
//! - Single root (no caps).
//!
//! Security: see `docs/mmcs-streaming-design.md` for the 8-vector threat
//! model; each vector is tested below.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::traits::IsMerkleTreeBackend;

/// Per-matrix domain separator. Caller-defined; verifier reconstructs
/// from chip spec.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct MatrixTag(pub [u8; 8]);

impl MatrixTag {
    pub const fn new(tag: [u8; 8]) -> Self {
        Self(tag)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum MmcsError {
    DuplicateTag,
    EmptyMatrix,
    NotPowerOfTwo,
    Empty,
    IndexOutOfBounds,
}

struct MmcsMatrix<N> {
    tag: MatrixTag,
    leaf_digests: Vec<N>,
}

impl<N> MmcsMatrix<N> {
    fn padded_height(&self) -> usize {
        self.leaf_digests.len()
    }
}

pub struct MmcsBuilder<B: IsMerkleTreeBackend> {
    matrices: Vec<MmcsMatrix<B::Node>>,
}

impl<B: IsMerkleTreeBackend> Default for MmcsBuilder<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: IsMerkleTreeBackend> MmcsBuilder<B> {
    pub fn new() -> Self {
        Self {
            matrices: Vec::new(),
        }
    }

    /// Register a matrix. `leaf_digests` MUST be pre-hashed with the
    /// matrix tag embedded (e.g. `hash(tag || row_bytes)`). Length must
    /// be a power of two.
    pub fn add_matrix(
        &mut self,
        tag: MatrixTag,
        leaf_digests: Vec<B::Node>,
    ) -> Result<(), MmcsError> {
        if self.matrices.iter().any(|m| m.tag == tag) {
            return Err(MmcsError::DuplicateTag);
        }
        if leaf_digests.is_empty() {
            return Err(MmcsError::EmptyMatrix);
        }
        if !leaf_digests.len().is_power_of_two() {
            return Err(MmcsError::NotPowerOfTwo);
        }
        self.matrices.push(MmcsMatrix { tag, leaf_digests });
        Ok(())
    }

    pub fn finalize(mut self) -> Result<Mmcs<B>, MmcsError> {
        if self.matrices.is_empty() {
            return Err(MmcsError::Empty);
        }
        // Deterministic sort: height desc, then tag asc. The verifier
        // reproduces this exact ordering so prover/verifier agree on
        // which matrix contributes when.
        self.matrices.sort_by(|a, b| {
            b.padded_height()
                .cmp(&a.padded_height())
                .then(a.tag.cmp(&b.tag))
        });

        let max_height = self.matrices[0].padded_height();
        let depth = max_height.trailing_zeros() as usize;

        // Group matrix indices by padded_height (preserving tag-asc order
        // within each group because `matrices` is already sorted).
        let mut by_height: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (idx, m) in self.matrices.iter().enumerate() {
            by_height.entry(m.padded_height()).or_default().push(idx);
        }

        let mut layers: Vec<Vec<B::Node>> = Vec::with_capacity(depth + 1);

        // Layer 0: combine all max-height matrices in tag-asc order.
        let top_group = by_height
            .get(&max_height)
            .expect("max_height bucket exists");
        let mut layer0: Vec<B::Node> = self.matrices[top_group[0]].leaf_digests.clone();
        for &mi in &top_group[1..] {
            for (node, leaf) in layer0.iter_mut().zip(self.matrices[mi].leaf_digests.iter()) {
                *node = B::hash_new_parent(node, leaf);
            }
        }
        layers.push(layer0);

        // Walk upward; at each new layer, compress pairs then inject all
        // matrices at this layer's length in tag-asc order.
        for level in 0..depth {
            let cur = &layers[level];
            let new_len = cur.len() / 2;
            let mut next: Vec<B::Node> = (0..new_len)
                .map(|i| B::hash_new_parent(&cur[2 * i], &cur[2 * i + 1]))
                .collect();
            if let Some(group) = by_height.get(&new_len) {
                for &mi in group {
                    for (node, leaf) in next.iter_mut().zip(self.matrices[mi].leaf_digests.iter()) {
                        *node = B::hash_new_parent(node, leaf);
                    }
                }
            }
            layers.push(next);
        }

        Ok(Mmcs {
            layers,
            matrices: self.matrices,
        })
    }
}

pub struct Mmcs<B: IsMerkleTreeBackend> {
    layers: Vec<Vec<B::Node>>,
    matrices: Vec<MmcsMatrix<B::Node>>,
}

impl<B: IsMerkleTreeBackend> Mmcs<B> {
    pub fn root(&self) -> &B::Node {
        let top = self.layers.last().expect("layers always populated");
        &top[0]
    }

    /// `(tag, padded_height)` per matrix in deterministic sort order.
    pub fn spec(&self) -> Vec<(MatrixTag, usize)> {
        self.matrices
            .iter()
            .map(|m| (m.tag, m.padded_height()))
            .collect()
    }

    pub fn open(&self, global_index: usize) -> Result<MmcsOpening<B::Node>, MmcsError> {
        let max_height = self.matrices[0].padded_height();
        if global_index >= max_height {
            return Err(MmcsError::IndexOutOfBounds);
        }
        let depth = max_height.trailing_zeros() as usize;

        let mut matrix_leaves: Vec<(MatrixTag, B::Node)> = Vec::with_capacity(self.matrices.len());
        for matrix in &self.matrices {
            let shift = (max_height / matrix.padded_height()).trailing_zeros() as usize;
            let idx = global_index >> shift;
            matrix_leaves.push((matrix.tag, matrix.leaf_digests[idx].clone()));
        }

        let mut siblings: Vec<B::Node> = Vec::with_capacity(depth);
        let mut idx = global_index;
        for layer in &self.layers[..depth] {
            let sibling_idx = idx ^ 1;
            siblings.push(layer[sibling_idx].clone());
            idx >>= 1;
        }

        Ok(MmcsOpening {
            matrix_leaves,
            siblings,
            global_index,
        })
    }
}

#[derive(Debug, Clone)]
pub struct MmcsOpening<N> {
    /// `(tag, leaf_at_shifted_index)` per matrix, in the builder's sort
    /// order (height desc, tag asc).
    pub matrix_leaves: Vec<(MatrixTag, N)>,
    pub siblings: Vec<N>,
    pub global_index: usize,
}

impl<N: PartialEq + Eq + Clone> MmcsOpening<N> {
    pub fn verify<B>(&self, expected_root: &N, expected_specs: &[(MatrixTag, usize)]) -> bool
    where
        B: IsMerkleTreeBackend<Node = N>,
    {
        let mut specs = expected_specs.to_vec();
        specs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        if self.matrix_leaves.len() != specs.len() {
            return false;
        }
        for ((tag, _), (spec_tag, _)) in self.matrix_leaves.iter().zip(&specs) {
            if tag != spec_tag {
                return false;
            }
        }
        for (_, ph) in &specs {
            if !ph.is_power_of_two() || *ph == 0 {
                return false;
            }
        }
        let max_height = specs[0].1;
        if self.global_index >= max_height {
            return false;
        }
        let depth = max_height.trailing_zeros() as usize;
        if self.siblings.len() != depth {
            return false;
        }

        // Walk `matrix_leaves` left to right with a cursor; the leaves
        // are grouped by height (largest first) and within each group
        // are sorted by tag.
        let mut cursor = 0usize;

        // Reconstruct layer-0 at global_index: combine all max-height
        // matrices' leaves at global_index in tag-asc order.
        let mut current = self.matrix_leaves[cursor].1.clone();
        cursor += 1;
        while cursor < self.matrix_leaves.len() && specs[cursor].1 == max_height {
            current = B::hash_new_parent(&current, &self.matrix_leaves[cursor].1);
            cursor += 1;
        }

        let mut idx = self.global_index;
        for level in 0..depth {
            let sibling = &self.siblings[level];
            current = if idx & 1 == 0 {
                B::hash_new_parent(&current, sibling)
            } else {
                B::hash_new_parent(sibling, &current)
            };
            idx >>= 1;

            let new_len = max_height >> (level + 1);
            while cursor < self.matrix_leaves.len() && specs[cursor].1 == new_len {
                current = B::hash_new_parent(&current, &self.matrix_leaves[cursor].1);
                cursor += 1;
            }
        }

        if cursor != self.matrix_leaves.len() {
            // Unconsumed leaves => topology mismatch.
            return false;
        }
        &current == expected_root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha3::{Digest, Keccak256};

    struct TestBackend;
    type Node = [u8; 32];
    impl IsMerkleTreeBackend for TestBackend {
        type Node = Node;
        type Data = Vec<u8>;
        fn hash_data(leaf: &Vec<u8>) -> Node {
            let mut h = Keccak256::new();
            h.update(leaf);
            h.finalize().into()
        }
        fn hash_new_parent(a: &Node, b: &Node) -> Node {
            let mut h = Keccak256::new();
            h.update(a);
            h.update(b);
            h.finalize().into()
        }
    }

    fn hash_leaf_with_tag(tag: &MatrixTag, row: &[u8]) -> Node {
        let mut h = Keccak256::new();
        h.update(b"LEAF_V1");
        h.update(tag.0);
        h.update(row);
        h.finalize().into()
    }

    fn make_matrix(tag_byte: u8, height: usize) -> (MatrixTag, Vec<Node>) {
        let tag = MatrixTag::new([tag_byte; 8]);
        let leaves: Vec<Node> = (0..height)
            .map(|i| hash_leaf_with_tag(&tag, &(i as u64).to_le_bytes()))
            .collect();
        (tag, leaves)
    }

    fn build(matrices: Vec<(MatrixTag, Vec<Node>)>) -> Mmcs<TestBackend> {
        let mut b: MmcsBuilder<TestBackend> = MmcsBuilder::new();
        for (tag, leaves) in matrices {
            b.add_matrix(tag, leaves).expect("add_matrix");
        }
        b.finalize().expect("finalize")
    }

    // ---------- Basic ----------

    #[test]
    fn build_single_matrix_round_trips() {
        let (tag, leaves) = make_matrix(0xAA, 8);
        let tree = build(vec![(tag, leaves)]);
        for i in 0..8 {
            let opening = tree.open(i).expect("open");
            assert!(opening.verify::<TestBackend>(tree.root(), &tree.spec()));
        }
    }

    #[test]
    fn build_distinct_heights_round_trips() {
        let big = make_matrix(0x01, 8);
        let mid = make_matrix(0x02, 4);
        let small = make_matrix(0x03, 2);
        let tree = build(vec![big, mid, small]);
        for i in 0..8 {
            let opening = tree.open(i).expect("open");
            assert!(opening.verify::<TestBackend>(tree.root(), &tree.spec()));
        }
    }

    #[test]
    fn build_is_deterministic() {
        let m1 = make_matrix(0x01, 8);
        let m2 = make_matrix(0x02, 4);
        let r1 = *build(vec![m1.clone(), m2.clone()]).root();
        let r2 = *build(vec![m1.clone(), m2.clone()]).root();
        assert_eq!(r1, r2);
        let r3 = *build(vec![m2, m1]).root();
        assert_eq!(r1, r3);
    }

    // ---------- Same-height topology (lambda-vm style) ----------

    #[test]
    fn same_height_pair_round_trips() {
        // Two matrices both at max_height — combined into layer 0.
        let m1 = make_matrix(0x01, 4);
        let m2 = make_matrix(0x02, 4);
        let tree = build(vec![m1, m2]);
        for i in 0..4 {
            let opening = tree.open(i).expect("open");
            assert!(
                opening.verify::<TestBackend>(tree.root(), &tree.spec()),
                "round-trip at index {i}"
            );
        }
    }

    #[test]
    fn lambda_vm_style_multi_chunk_round_trips() {
        // 3 max-height chunks (CPU-like), 2 mid-height (MEMW-like at 1/2),
        // 1 small (REGISTER-like at 1/8). Heights: 8, 8, 8, 4, 4, 1.
        let cpus = vec![
            make_matrix(0x01, 8),
            make_matrix(0x02, 8),
            make_matrix(0x03, 8),
        ];
        let memws = vec![make_matrix(0x10, 4), make_matrix(0x11, 4)];
        let reg = make_matrix(0xF0, 1);
        let mut all = cpus;
        all.extend(memws);
        all.push(reg);
        let tree = build(all);
        for i in 0..8 {
            let opening = tree.open(i).expect("open");
            assert!(
                opening.verify::<TestBackend>(tree.root(), &tree.spec()),
                "round-trip at index {i}"
            );
        }
    }

    #[test]
    fn insertion_order_does_not_change_root() {
        // Multi-permutation determinism: any permutation of the same set
        // of matrices must produce the same root.
        let a = make_matrix(0x01, 8);
        let b = make_matrix(0x02, 8);
        let c = make_matrix(0x03, 4);
        let r1 = *build(vec![a.clone(), b.clone(), c.clone()]).root();
        let r2 = *build(vec![c.clone(), a.clone(), b.clone()]).root();
        let r3 = *build(vec![b, c, a]).root();
        assert_eq!(r1, r2);
        assert_eq!(r1, r3);
    }

    #[test]
    fn same_height_tampered_leaf_rejected() {
        let m1 = make_matrix(0x01, 4);
        let m2 = make_matrix(0x02, 4);
        let tree = build(vec![m1, m2]);
        let mut opening = tree.open(2).expect("open");
        // Flip one bit of the second max-height matrix's leaf.
        opening.matrix_leaves[1].1[0] ^= 1;
        assert!(!opening.verify::<TestBackend>(tree.root(), &tree.spec()));
    }

    // ---------- Threat model (vectors 1-8) ----------

    #[test]
    fn v1_cross_matrix_row_swap_is_rejected() {
        let big = make_matrix(0xAA, 4);
        let small = make_matrix(0xBB, 2);
        let tree = build(vec![big, small]);
        let mut opening = tree.open(0).expect("open");
        opening.matrix_leaves.swap(0, 1);
        assert!(!opening.verify::<TestBackend>(tree.root(), &tree.spec()));
    }

    #[test]
    fn v2_unpadded_matrix_is_rejected_at_build() {
        let tag = MatrixTag::new([0; 8]);
        let leaves: Vec<Node> = (0..3).map(|i| [i as u8; 32]).collect();
        let mut b: MmcsBuilder<TestBackend> = MmcsBuilder::new();
        assert_eq!(b.add_matrix(tag, leaves), Err(MmcsError::NotPowerOfTwo));
    }

    #[test]
    fn v3_layer_injection_order_deterministic_under_permutation() {
        // Two matrices at same height — combining is in tag-asc order
        // regardless of insertion. Already covered above; pin it here.
        let m1 = make_matrix(0x01, 4);
        let m2 = make_matrix(0x02, 4);
        assert_eq!(
            *build(vec![m1.clone(), m2.clone()]).root(),
            *build(vec![m2, m1]).root()
        );
    }

    #[test]
    fn v4_auth_path_forgery_via_relabeling_is_rejected() {
        let big = make_matrix(0xAA, 4);
        let small = make_matrix(0xBB, 2);
        let tree = build(vec![big, small]);
        let mut opening = tree.open(0).expect("open");
        opening.matrix_leaves[1].0 = MatrixTag::new([0xCC; 8]);
        assert!(!opening.verify::<TestBackend>(tree.root(), &tree.spec()));
    }

    #[test]
    fn v5_wrong_leaf_data_is_rejected() {
        let big = make_matrix(0xAA, 4);
        let small = make_matrix(0xBB, 2);
        let tree = build(vec![big, small]);
        let mut opening = tree.open(0).expect("open");
        opening.matrix_leaves[1].1[0] ^= 1;
        assert!(!opening.verify::<TestBackend>(tree.root(), &tree.spec()));
    }

    #[test]
    fn v6_index_tampering_rejected() {
        let big = make_matrix(0xAA, 4);
        let tree = build(vec![big]);
        let o0 = tree.open(0).expect("open 0");
        let o1 = tree.open(1).expect("open 1");
        assert_ne!(o0.matrix_leaves[0].1, o1.matrix_leaves[0].1);
        let mut faked = o0.clone();
        faked.global_index = 1;
        assert!(!faked.verify::<TestBackend>(tree.root(), &tree.spec()));
    }

    #[test]
    fn v7_truncated_path_is_rejected() {
        let big = make_matrix(0xAA, 8);
        let tree = build(vec![big]);
        let mut opening = tree.open(3).expect("open");
        opening.siblings.pop();
        assert!(!opening.verify::<TestBackend>(tree.root(), &tree.spec()));
    }

    #[test]
    fn v8_lying_about_spec_is_rejected() {
        let big = make_matrix(0xAA, 8);
        let tree = build(vec![big]);
        let opening = tree.open(0).expect("open");
        let bad_specs = vec![(MatrixTag::new([0xAA; 8]), 4)];
        assert!(!opening.verify::<TestBackend>(tree.root(), &bad_specs));
    }

    #[test]
    fn duplicate_tag_is_rejected() {
        let tag = MatrixTag::new([1; 8]);
        let leaves: Vec<Node> = vec![[0; 32]; 4];
        let mut b: MmcsBuilder<TestBackend> = MmcsBuilder::new();
        b.add_matrix(tag, leaves.clone()).expect("add first");
        assert_eq!(b.add_matrix(tag, leaves), Err(MmcsError::DuplicateTag));
    }

    #[test]
    fn open_out_of_bounds_is_rejected() {
        let big = make_matrix(0xAA, 4);
        let tree = build(vec![big]);
        assert_eq!(tree.open(4).err(), Some(MmcsError::IndexOutOfBounds));
    }
}

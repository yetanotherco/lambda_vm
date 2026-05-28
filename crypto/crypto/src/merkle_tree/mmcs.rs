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

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::traits::IsMerkleTreeBackend;

/// Per-matrix domain separator. Caller-defined; verifier reconstructs
/// from chip spec.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Returned by [`StreamingMmcsBuilder::add_matrix`] when the caller
    /// supplies a `(height, tag)` pair that violates the required
    /// (height desc, tag asc) insertion order.
    OutOfOrder,
}

struct MmcsMatrix<N> {
    tag: MatrixTag,
    /// Source row hashes. Populated by the one-shot [`MmcsBuilder`] and
    /// consulted by [`Mmcs::open`] to fill the per-matrix leaf in an
    /// opening. Empty when the Mmcs was produced by [`StreamingMmcsBuilder`]
    /// (which discards per-chip leaves as it folds them), in which case
    /// `Mmcs::open` is unavailable but `root()` / `spec()` still work.
    leaf_digests: Vec<N>,
    /// Padded height (= leaf_digests.len() for one-shot, or the height
    /// recorded at insertion time for streaming). Carried separately so
    /// `padded_height()` reports the right value when `leaf_digests` is
    /// empty.
    padded_height: usize,
}

impl<N> MmcsMatrix<N> {
    fn padded_height(&self) -> usize {
        self.padded_height
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
        let padded_height = leaf_digests.len();
        self.matrices.push(MmcsMatrix {
            tag,
            leaf_digests,
            padded_height,
        });
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

        // Layer 0: combine all max-height matrices' leaves at row i in
        // tag-asc order. Row-parallel: each row independently folds K
        // matrices (K is small — 1-5 typically), so the per-row sequential
        // chain is short while rows scale across cores. Mirrors Plonky3's
        // `first_digest_layer` parallelism, minus the SIMD vertical packing
        // (lambda-vm uses scalar Keccak).
        let top_group = by_height
            .get(&max_height)
            .expect("max_height bucket exists");
        let layer0: Vec<B::Node> = build_combined_layer::<B>(max_height, top_group, &self.matrices);
        layers.push(layer0);

        // Walk upward: compress pairs (pair-parallel), then inject any
        // matrices at this layer's length (row-parallel).
        for level in 0..depth {
            let cur = &layers[level];
            let new_len = cur.len() / 2;
            let mut next: Vec<B::Node> = compress_pairs::<B>(cur);
            if let Some(group) = by_height.get(&new_len) {
                inject_matrices::<B>(&mut next, group, &self.matrices);
            }
            layers.push(next);
            let _ = new_len;
        }

        Ok(Mmcs {
            layers,
            matrices: self.matrices,
        })
    }
}

/// Streaming MMCS builder. Equivalent to [`MmcsBuilder`] in output
/// (identical root + spec + opening *root* bytes for the same input set)
/// but folds per-chip leaves at the MAX height into a single shared
/// running layer-0 as they arrive, instead of holding every max-height
/// chip's leaf vector alive simultaneously.
///
/// # Why "max height only"?
///
/// MMCS layer-0 at the max height is built by left-folding every chip's
/// leaves at row `i`. With no left-anchor to compose with, the running
/// fold `acc = hash(acc, chip_k[i])` is mathematically equivalent to the
/// one-shot `hash(hash(hash(chip_0[i], chip_1[i]), chip_2[i]), ...)`.
///
/// For chips at heights BELOW max, the MMCS injection rule is
/// `next[i] = hash(hash(hash(next[i], chip_0[i]), chip_1[i]), ...)`,
/// which mixes the upward-compressed `next[i]` into the left-fold. Keccak
/// (and any non-associative hash) makes it impossible to pre-fold the
/// chips into a single summary and inject that summary later — the
/// resulting digest would differ from the one-shot builder, breaking
/// verifier compatibility. So we keep per-chip leaves for non-max heights
/// and inject them in left-fold order at `finalize`.
///
/// # Memory
///
/// Peak savings come from the max-height chips, which is where the
/// dominant per-row storage lives in lambda-vm (CPU chunks at 2^20).
/// Smaller-height chips contribute proportionally less per chip, so
/// keeping their per-chip leaves alive has modest impact.
///
/// # Add order
///
/// Callers MUST call [`StreamingMmcsBuilder::add_matrix`] in the same
/// order that [`MmcsBuilder::finalize`] would sort the matrices in:
/// height descending, then tag ascending within each height. The builder
/// returns [`MmcsError::OutOfOrder`] if a call would break this.
pub struct StreamingMmcsBuilder<B: IsMerkleTreeBackend> {
    /// Max-height layer-0 — incrementally folded as max-height chips
    /// arrive. `None` until the first chip is added (which fixes the
    /// max height).
    layer0: Option<Vec<B::Node>>,
    /// Per-chip leaves for chips at heights < max_height, grouped by
    /// height. Within each group, chips are in tag-asc order (enforced
    /// by `add_matrix`).
    by_height_below_max: BTreeMap<usize, Vec<Vec<B::Node>>>,
    /// `(tag, padded_height)` in caller-supplied order. Populates the
    /// final `Mmcs.matrices` (used by `spec()`).
    matrix_specs: Vec<(MatrixTag, usize)>,
    max_height: Option<usize>,
}

impl<B: IsMerkleTreeBackend> Default for StreamingMmcsBuilder<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: IsMerkleTreeBackend> StreamingMmcsBuilder<B> {
    pub fn new() -> Self {
        Self {
            layer0: None,
            by_height_below_max: BTreeMap::new(),
            matrix_specs: Vec::new(),
            max_height: None,
        }
    }

    /// Add a chip's leaves to the in-progress MMCS. The vector is
    /// consumed so the caller can drop the chip's source data
    /// immediately on return.
    ///
    /// At the MAX height the leaves are folded into the shared layer-0
    /// running and the vector is freed. At lower heights the vector is
    /// stored verbatim until `finalize`.
    pub fn add_matrix(
        &mut self,
        tag: MatrixTag,
        leaf_digests: Vec<B::Node>,
    ) -> Result<(), MmcsError> {
        if leaf_digests.is_empty() {
            return Err(MmcsError::EmptyMatrix);
        }
        if !leaf_digests.len().is_power_of_two() {
            return Err(MmcsError::NotPowerOfTwo);
        }
        // Order check first — protects all subsequent invariants.
        let h = leaf_digests.len();
        if let Some(&(prev_tag, prev_h)) = self.matrix_specs.last() {
            let ord = core::cmp::Ord::cmp(&prev_h, &h)
                .reverse()
                .then(prev_tag.cmp(&tag));
            if !matches!(ord, core::cmp::Ordering::Less) {
                return Err(MmcsError::OutOfOrder);
            }
        }
        if self.matrix_specs.iter().any(|(t, _)| *t == tag) {
            return Err(MmcsError::DuplicateTag);
        }

        match self.max_height {
            None => {
                // First chip — its height fixes max_height; its leaves
                // seed the running layer-0.
                self.max_height = Some(h);
                self.layer0 = Some(leaf_digests);
            }
            Some(max_h) if h == max_h => {
                // Subsequent max-height chip — fold into running layer-0.
                let running = self
                    .layer0
                    .as_mut()
                    .expect("layer0 populated once max_height is set");
                debug_assert_eq!(running.len(), leaf_digests.len());
                fold_into::<B>(running, &leaf_digests);
            }
            Some(_) => {
                // Below max — stash per-chip leaves, drop at finalize.
                self.by_height_below_max
                    .entry(h)
                    .or_default()
                    .push(leaf_digests);
            }
        }
        self.matrix_specs.push((tag, h));
        Ok(())
    }

    /// Compress the running layer-0 upward, injecting lower-height chips
    /// at the matching level using the same left-fold the one-shot
    /// [`MmcsBuilder::finalize`] uses.
    ///
    /// The returned [`Mmcs`] has empty `leaf_digests` for each matrix
    /// because the streaming builder consumed them. `root()` / `spec()`
    /// are fully functional; callers that also need [`Mmcs::open`] must
    /// regenerate the chip leaves or use [`MmcsBuilder`].
    pub fn finalize(self) -> Result<Mmcs<B>, MmcsError> {
        if self.matrix_specs.is_empty() {
            return Err(MmcsError::Empty);
        }
        let max_height = self.max_height.ok_or(MmcsError::Empty)?;
        let depth = max_height.trailing_zeros() as usize;

        let StreamingMmcsBuilder {
            layer0,
            mut by_height_below_max,
            matrix_specs,
            max_height: _,
        } = self;

        let mut layers: Vec<Vec<B::Node>> = Vec::with_capacity(depth + 1);
        layers.push(layer0.ok_or(MmcsError::Empty)?);

        for level in 0..depth {
            let mut next = compress_pairs::<B>(&layers[level]);
            let new_len = max_height >> (level + 1);
            if let Some(chips) = by_height_below_max.remove(&new_len) {
                inject_chips_left_fold::<B>(&mut next, &chips);
            }
            layers.push(next);
        }

        // Carry tag + height into the Mmcs so `spec()` reports the right
        // pairs. leaf_digests stays empty — opens are not supported on
        // streaming output (caller must use the one-shot builder when
        // openings are needed).
        let matrices = matrix_specs
            .into_iter()
            .map(|(tag, padded_height)| MmcsMatrix {
                tag,
                leaf_digests: Vec::new(),
                padded_height,
            })
            .collect();
        Ok(Mmcs { layers, matrices })
    }
}

/// Per-row fold: `acc[i] = hash_new_parent(acc[i], other[i])`.
fn fold_into<B: IsMerkleTreeBackend>(acc: &mut [B::Node], other: &[B::Node]) {
    debug_assert_eq!(acc.len(), other.len());
    let n = acc.len();
    let updated: Vec<B::Node> = {
        let inner = |i: usize| -> B::Node { B::hash_new_parent(&acc[i], &other[i]) };
        #[cfg(feature = "parallel")]
        {
            (0..n).into_par_iter().map(inner).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..n).map(inner).collect()
        }
    };
    acc.clone_from_slice(&updated);
}

/// Left-fold inject several chips' leaves into `layer` at every row in
/// tag-asc chip order:
/// `layer[i] = hash(hash(hash(layer[i], chips[0][i]), chips[1][i]), ...)`.
/// Mirrors `inject_matrices` in the one-shot path.
fn inject_chips_left_fold<B: IsMerkleTreeBackend>(
    layer: &mut [B::Node],
    chips: &[Vec<B::Node>],
) {
    let n = layer.len();
    let updated: Vec<B::Node> = {
        let inner = |i: usize| -> B::Node {
            let mut acc = layer[i].clone();
            for chip in chips {
                acc = B::hash_new_parent(&acc, &chip[i]);
            }
            acc
        };
        #[cfg(feature = "parallel")]
        {
            (0..n).into_par_iter().map(inner).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..n).map(inner).collect()
        }
    };
    layer.clone_from_slice(&updated);
}


/// Build layer 0 by folding all matrices at `max_height` at row `i`, in
/// tag-asc order (`group` already preserves this). Row-parallel.
fn build_combined_layer<B: IsMerkleTreeBackend>(
    max_height: usize,
    group: &[usize],
    matrices: &[MmcsMatrix<B::Node>],
) -> Vec<B::Node> {
    let inner = |i: usize| -> B::Node {
        let mut acc = matrices[group[0]].leaf_digests[i].clone();
        for &mi in &group[1..] {
            acc = B::hash_new_parent(&acc, &matrices[mi].leaf_digests[i]);
        }
        acc
    };
    #[cfg(feature = "parallel")]
    {
        (0..max_height).into_par_iter().map(inner).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..max_height).map(inner).collect()
    }
}

/// Compress pairs of children into the next layer up. Pair-parallel.
fn compress_pairs<B: IsMerkleTreeBackend>(prev: &[B::Node]) -> Vec<B::Node> {
    let new_len = prev.len() / 2;
    let inner = |i: usize| -> B::Node { B::hash_new_parent(&prev[2 * i], &prev[2 * i + 1]) };
    #[cfg(feature = "parallel")]
    {
        (0..new_len).into_par_iter().map(inner).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..new_len).map(inner).collect()
    }
}

/// Inject all matrices in `group` into `layer` (row-parallel).
fn inject_matrices<B: IsMerkleTreeBackend>(
    layer: &mut [B::Node],
    group: &[usize],
    matrices: &[MmcsMatrix<B::Node>],
) {
    let n = layer.len();
    let updated: Vec<B::Node> = {
        let inner = |i: usize| -> B::Node {
            let mut acc = layer[i].clone();
            for &mi in group {
                acc = B::hash_new_parent(&acc, &matrices[mi].leaf_digests[i]);
            }
            acc
        };
        #[cfg(feature = "parallel")]
        {
            (0..n).into_par_iter().map(inner).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..n).map(inner).collect()
        }
    };
    layer.clone_from_slice(&updated);
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "N: serde::Serialize + serde::de::DeserializeOwned")
)]
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

    // ---------- StreamingMmcsBuilder equivalence ----------

    fn build_streaming(
        matrices_in_spec_order: Vec<(MatrixTag, Vec<Node>)>,
    ) -> Mmcs<TestBackend> {
        let mut b: StreamingMmcsBuilder<TestBackend> = StreamingMmcsBuilder::new();
        for (tag, leaves) in matrices_in_spec_order {
            b.add_matrix(tag, leaves).expect("streaming add_matrix");
        }
        b.finalize().expect("streaming finalize")
    }

    /// Convert an arbitrary input set into the (height desc, tag asc)
    /// order required by `StreamingMmcsBuilder`. Matches the sort
    /// `MmcsBuilder::finalize` does internally.
    fn spec_sorted(mut v: Vec<(MatrixTag, Vec<Node>)>) -> Vec<(MatrixTag, Vec<Node>)> {
        v.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
        v
    }

    #[test]
    fn streaming_root_matches_oneshot_single_matrix() {
        let m = make_matrix(0xAA, 8);
        let r_oneshot = *build(vec![m.clone()]).root();
        let r_stream = *build_streaming(spec_sorted(vec![m])).root();
        assert_eq!(r_oneshot, r_stream);
    }

    #[test]
    fn streaming_root_matches_oneshot_lambdavm_topology() {
        let inputs = vec![
            make_matrix(0x01, 8),
            make_matrix(0x02, 8),
            make_matrix(0x03, 8),
            make_matrix(0x10, 4),
            make_matrix(0x11, 4),
            make_matrix(0xF0, 1),
        ];
        let r_oneshot = *build(inputs.clone()).root();
        let r_stream = *build_streaming(spec_sorted(inputs)).root();
        assert_eq!(r_oneshot, r_stream);
    }

    #[test]
    fn streaming_spec_matches_oneshot() {
        let inputs = vec![
            make_matrix(0x01, 8),
            make_matrix(0x02, 4),
            make_matrix(0x03, 8),
            make_matrix(0x04, 2),
        ];
        let oneshot = build(inputs.clone());
        let stream = build_streaming(spec_sorted(inputs));
        assert_eq!(oneshot.spec(), stream.spec());
    }

    #[test]
    fn streaming_rejects_height_ascending() {
        let mut b: StreamingMmcsBuilder<TestBackend> = StreamingMmcsBuilder::new();
        let (t0, l0) = make_matrix(0x01, 4);
        let (t1, l1) = make_matrix(0x02, 8);
        b.add_matrix(t0, l0).expect("first add");
        assert_eq!(b.add_matrix(t1, l1), Err(MmcsError::OutOfOrder));
    }

    #[test]
    fn streaming_rejects_same_height_tag_descending() {
        let mut b: StreamingMmcsBuilder<TestBackend> = StreamingMmcsBuilder::new();
        let (t0, l0) = make_matrix(0x02, 4);
        let (t1, l1) = make_matrix(0x01, 4);
        b.add_matrix(t0, l0).expect("first add");
        assert_eq!(b.add_matrix(t1, l1), Err(MmcsError::OutOfOrder));
    }

    #[test]
    fn streaming_rejects_duplicate_tag_same_height() {
        // Same tag and same height violates (height desc, tag asc); the
        // order check fires first.
        let mut b: StreamingMmcsBuilder<TestBackend> = StreamingMmcsBuilder::new();
        let (t, l) = make_matrix(0x01, 4);
        b.add_matrix(t, l.clone()).expect("first add");
        assert_eq!(b.add_matrix(t, l), Err(MmcsError::OutOfOrder));
    }

    #[test]
    fn streaming_rejects_duplicate_tag_smaller_height() {
        // Same tag at a strictly smaller height passes the order check,
        // so the dup-tag scan catches it instead.
        let mut b: StreamingMmcsBuilder<TestBackend> = StreamingMmcsBuilder::new();
        let (t, l) = make_matrix(0x01, 4);
        b.add_matrix(t, l).expect("first add");
        let l2: Vec<Node> = vec![[0; 32]; 2];
        assert_eq!(b.add_matrix(t, l2), Err(MmcsError::DuplicateTag));
    }

    #[test]
    fn streaming_rejects_empty_and_non_power_of_two() {
        let mut b: StreamingMmcsBuilder<TestBackend> = StreamingMmcsBuilder::new();
        let tag = MatrixTag::new([0; 8]);
        assert_eq!(b.add_matrix(tag, Vec::new()), Err(MmcsError::EmptyMatrix));
        let bad: Vec<Node> = vec![[0; 32]; 3];
        assert_eq!(b.add_matrix(tag, bad), Err(MmcsError::NotPowerOfTwo));
    }

    #[test]
    fn streaming_root_matches_oneshot_pure_same_height() {
        let inputs = vec![
            make_matrix(0x01, 8),
            make_matrix(0x02, 8),
            make_matrix(0x03, 8),
            make_matrix(0x04, 8),
            make_matrix(0x05, 8),
        ];
        let r_oneshot = *build(inputs.clone()).root();
        let r_stream = *build_streaming(spec_sorted(inputs)).root();
        assert_eq!(r_oneshot, r_stream);
    }
}

#[cfg(test)]
mod bench {
    //! Micro-benchmark comparing MMCS build against N independent
    //! `MerkleTree` builds for a lambda-vm-style topology. Marked
    //! `#[ignore]` so it doesn't run by default; trigger with
    //!     cargo test -p crypto --features parallel mmcs_bench -- --ignored --nocapture
    use super::*;
    use crate::merkle_tree::merkle::MerkleTree;
    use sha3::{Digest, Keccak256};
    use std::time::Instant;

    struct BenchBackend;
    type Node = [u8; 32];
    impl IsMerkleTreeBackend for BenchBackend {
        type Node = Node;
        type Data = Node;
        fn hash_data(leaf: &Node) -> Node {
            *leaf
        }
        fn hash_new_parent(a: &Node, b: &Node) -> Node {
            let mut h = Keccak256::new();
            h.update(a);
            h.update(b);
            h.finalize().into()
        }
    }

    fn synthetic_chip_leaves(seed: u8, height: usize) -> Vec<Node> {
        (0..height)
            .map(|i| {
                let mut h = Keccak256::new();
                h.update([seed]);
                h.update((i as u64).to_le_bytes());
                h.finalize().into()
            })
            .collect()
    }

    /// lambda-vm-style topology, scaled down so the bench finishes fast:
    /// - 3 chips at 2^14 (CPU-like chunked)
    /// - 2 chips at 2^12 (MEMW-like)
    /// - 2 chips at 2^10 (LT-like)
    /// - 1 chip at 2^8  (HALT/COMMIT-like)
    fn lambda_vm_topology() -> Vec<(MatrixTag, Vec<Node>)> {
        let mut out = Vec::new();
        let mut seed = 0u8;
        for height in [1 << 14, 1 << 14, 1 << 14] {
            out.push((
                MatrixTag::new([seed; 8]),
                synthetic_chip_leaves(seed, height),
            ));
            seed = seed.wrapping_add(1);
        }
        for height in [1 << 12, 1 << 12] {
            out.push((
                MatrixTag::new([seed; 8]),
                synthetic_chip_leaves(seed, height),
            ));
            seed = seed.wrapping_add(1);
        }
        for height in [1 << 10, 1 << 10] {
            out.push((
                MatrixTag::new([seed; 8]),
                synthetic_chip_leaves(seed, height),
            ));
            seed = seed.wrapping_add(1);
        }
        {
            let height = 1 << 8;
            out.push((
                MatrixTag::new([seed; 8]),
                synthetic_chip_leaves(seed, height),
            ));
        }
        out
    }

    #[test]
    #[ignore]
    fn mmcs_bench_lambda_vm_topology() {
        let chips = lambda_vm_topology();
        let total_leaves: usize = chips.iter().map(|(_, l)| l.len()).sum();
        let max_h = chips.iter().map(|(_, l)| l.len()).max().unwrap();

        // Warm caches.
        for _ in 0..2 {
            let mut b: MmcsBuilder<BenchBackend> = MmcsBuilder::new();
            for (t, l) in &chips {
                b.add_matrix(*t, l.clone()).unwrap();
            }
            let _ = b.finalize().unwrap();
        }

        // MMCS build.
        let t0 = Instant::now();
        let iters = 5;
        let mut mmcs_root = [0u8; 32];
        for _ in 0..iters {
            let mut b: MmcsBuilder<BenchBackend> = MmcsBuilder::new();
            for (t, l) in &chips {
                b.add_matrix(*t, l.clone()).unwrap();
            }
            let m = b.finalize().unwrap();
            mmcs_root = *m.root();
        }
        let mmcs_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // N independent trees build.
        let t0 = Instant::now();
        let mut n_roots = Vec::new();
        for _ in 0..iters {
            let roots: Vec<Node> = chips
                .iter()
                .map(|(_, leaves)| {
                    let tree = MerkleTree::<BenchBackend>::build_from_hashed_leaves(leaves.clone())
                        .unwrap();
                    tree.root
                })
                .collect();
            n_roots = roots;
        }
        let ntrees_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // Sanity: per-chip roots equal one of the layer-0 contributions for
        // MMCS *only* when the chip is the sole max-height matrix — we don't
        // assert equality, just print stats so reviewers can spot anomalies.
        let _ = (mmcs_root, n_roots);

        println!();
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│ MMCS micro-bench (lambda-vm-style topology)                 │");
        println!("├─────────────────────────────────────────────────────────────┤");
        println!(
            "│ Chips: {:<3}    Σh_i: {:<10}   max_h: {:<10}    │",
            chips.len(),
            total_leaves,
            max_h
        );
        println!(
            "│ Build N independent trees:  {:>8.0} µs                  │",
            ntrees_us
        );
        println!(
            "│ Build single MMCS tree:     {:>8.0} µs                  │",
            mmcs_us
        );
        println!(
            "│ MMCS / N-trees ratio:       {:>8.3}                     │",
            mmcs_us / ntrees_us
        );
        println!("└─────────────────────────────────────────────────────────────┘");
    }

    #[test]
    #[ignore]
    fn mmcs_opening_count_lambda_vm_topology() {
        let chips = lambda_vm_topology();
        let mut b: MmcsBuilder<BenchBackend> = MmcsBuilder::new();
        for (t, l) in &chips {
            b.add_matrix(*t, l.clone()).unwrap();
        }
        let tree = b.finalize().unwrap();
        let opening = tree.open(0).unwrap();

        // Path siblings + per-matrix leaves -> total opening hashes.
        let mmcs_hashes = opening.siblings.len() + opening.matrix_leaves.len() - 1;

        // Today (N independent trees): each chip's opening path is log2(h_i)
        // hashes; verifier must hash one extra per opening for the leaf
        // compute. Total per-query hashes = Σ (log2(h_i) + 1).
        let ntrees_hashes: usize = chips
            .iter()
            .map(|(_, l)| l.len().trailing_zeros() as usize + 1)
            .sum();

        println!();
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│ MMCS per-query opening hash count                           │");
        println!("├─────────────────────────────────────────────────────────────┤");
        println!(
            "│ N independent trees: {:>4} hashes per query             │",
            ntrees_hashes
        );
        println!(
            "│ Unified MMCS:        {:>4} hashes per query             │",
            mmcs_hashes
        );
        println!(
            "│ Reduction factor:    {:>4.2}x                              │",
            ntrees_hashes as f64 / mmcs_hashes as f64
        );
        println!("└─────────────────────────────────────────────────────────────┘");
    }
}

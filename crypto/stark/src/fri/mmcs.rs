//! Mixed-height, row-pair MMCS (Merkle Mixed Commitment Scheme).
//!
//! Commits ALL of an epoch's matrices (one per table, of possibly different
//! heights) into ONE mixed-height Merkle tree, so a single query opens ONE
//! authentication path that covers every table's row at that query — the
//! proof-size / opening-path win of the unified-shard design (SP1 / OpenVM /
//! Plonky3). Mirrors Plonky3's `MerkleTreeMmcs`, adapted to our Keccak backends
//! and to the row-pair `(x, -x)` leaf layout (#735).
//!
//! NOTE: this is a STANDALONE primitive (Task 1). It is not yet wired into the
//! prover/verifier — Tasks 2/7 consume the leaf/injection layout documented here
//! as the single source of truth.
//!
//! # Inputs
//!
//! `commit` takes matrices as `(row_major_bit_reversed_lde, log_height, width)`:
//! - `row_major_bit_reversed_lde`: the matrix's LDE evaluations, already
//!   bit-reversed and laid out ROW-MAJOR. Row `j` is the `width` contiguous
//!   elements `data[j*width .. (j+1)*width]`; row `j` corresponds to bit-reversed
//!   LDE position `j` (same layout the per-table trace commit produces internally).
//! - `log_height`: `log2` of the number of rows; the matrix has `2^log_height`
//!   rows and `data.len() == width << log_height`.
//! - `width`: number of columns.
//!
//! # Row-pair leaves
//!
//! Leaf `k` of a matrix groups LDE positions `2k` and `2k+1` (the FRI fold pair
//! `x` and `-x`), all `width` columns batched. A matrix of `log_height h` has
//! `2^(h-1)` leaves. In `open_batch`/`PolynomialOpenings`: `evaluations` = row
//! `2k`, `evaluations_sym` = row `2k+1`.
//!
//! # Tree layout (the soundness-relevant contract — verbatim for Task 7)
//!
//! Let `h_max = max(log_height)`. The base digest layer (layer 0) has
//! `N0 = 2^(h_max-1)` nodes. Layer `i` has `N0 >> i` nodes; the root is the sole
//! node of layer `h_max-1`. A matrix of `log_height h` is *injected* at layer
//! index `i = h_max - h` (so the tallest matrices, `h == h_max`, populate the
//! base layer; shorter matrices enter where the layer width matches their leaf
//! count `2^(h-1)`).
//!
//! Hashing (`H = BatchedMerkleTreeBackend::hash_data` over a `Vec` of field
//! elements; `C = BatchedMerkleTreeBackend::hash_new_parent`, the 2-input Keccak
//! compression — identical to the existing per-table tree):
//!
//! - **Base layer** node `k` (`k in [0, N0)`):
//!   `layer0[k] = H( CONCAT_{m : h_m == h_max} (row_m(2k) || row_m(2k+1)) )`
//!   where matrices of height `h_max` are concatenated in INPUT order.
//! - **Climb** from layer `i` to layer `i+1` (`j in [0, N_{i+1})`):
//!   `parent = C(layer_i[2j], layer_i[2j+1])`. Let `inject_h = h_max - 1 - i`. If
//!   any matrix has `h_m == inject_h`, then
//!   `layer_{i+1}[j] = C( parent, H( CONCAT_{m : h_m == inject_h} (row_m(2j) || row_m(2j+1)) ) )`
//!   (injecting matrices concatenated in INPUT order); otherwise
//!   `layer_{i+1}[j] = parent`.
//! - `root = layer_{h_max-1}[0]`.
//!
//! # Query opening
//!
//! For query `iota in [0, N0)`, matrix `m` is opened at leaf
//! `k_m = iota >> (h_max - h_m)` (`= iota >> i_m`). The shared authentication
//! path holds, for each level `level in [0, h_max-1)`, the sibling
//! `layer_level[(iota >> level) ^ 1]`. ONE path authenticates all matrices.
//! The per-matrix `PolynomialOpenings.proof` fields are empty; the single
//! `MixedOpening.proof` is the authenticator.
//!
//! # Width binding (soundness)
//!
//! `verify_batch` takes per-matrix `widths` alongside `heights`. Within a height
//! group the leaf hash is over the FLAT concatenation of every matrix's opened
//! row pair (`A.eval ‖ A.eval_sym ‖ B.eval ‖ B.eval_sym ‖ …`), which does NOT by
//! itself record where each matrix's columns end. Fixing `widths[m]` (matrix
//! `m`'s column count) makes those boundaries unambiguous: without it a prover
//! could shift a boundary — e.g. lengthen one matrix's `evaluations` by one
//! element and shorten its `evaluations_sym` by one — leaving the flat bytes (and
//! therefore the group hash) identical while feeding a corrupted row downstream.
//! Consumers (the Task 7 verifier) MUST pass the committed public per-table
//! column counts, in the same INPUT order as `heights`.
//!
//! NOTE: `widths` and `heights` should ALSO be bound into the Fiat-Shamir
//! transcript by the consumer. Scope A's `absorb_height_histogram` currently
//! binds heights only; extending it to `(height, width)` pairs is a Task 4 /
//! verifier concern (out of scope for this primitive) and is flagged here.
//!
//! # Determinism
//!
//! The tree is a pure function of `(matrices, input order)`. Grouping within a
//! height (base batching and injection) follows INPUT order; the prover and
//! verifier MUST pass matrices and `heights` in the same per-epoch order. The
//! single-matrix case is byte-identical to the existing per-table row-pair tree.

use core::marker::PhantomData;

use crypto::merkle_tree::proof::Proof;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math::fft::bit_reversing::reverse_index;
use math::field::element::FieldElement;
use math::field::traits::IsField;
use math::traits::AsBytes;

use crate::config::{BatchedMerkleTreeBackend, Commitment};
use crate::proof::stark::PolynomialOpenings;

/// On-demand supplier of committed matrix rows, so [`MixedMmcs`] builds its
/// digests and serves openings WITHOUT owning a copy of the (large) LDE buffers.
/// Both [`MixedMmcs::commit`] and [`MixedMmcs::open_batch`] read every leaf
/// through this trait, so the root and opened rows are byte-identical to those a
/// matrix-owning MMCS would produce — the prover keeps only the LDE buffers it
/// already retains for DEEP, and each MMCS stores just digests.
///
/// Rows are addressed in each matrix's committed row-pair layout: `append_row(m,
/// r, out)` appends matrix `m`'s row at **bit-reversed** LDE position `r` (its
/// `width(m)` committed columns, in column order). This is the same `r`-indexing
/// the module's "Tree layout" section uses; an implementor holding the
/// natural-order LDE maps `r` to `reverse_index(r, 2^log_height(m))`.
pub trait LeafSource<E: IsField> {
    /// Number of committed matrices, in canonical input order.
    fn num_matrices(&self) -> usize;
    /// `log2` of matrix `m`'s row count. Row-pair leaves require `>= 1`.
    fn log_height(&self, m: usize) -> usize;
    /// Matrix `m`'s committed column count.
    fn width(&self, m: usize) -> usize;
    /// Append matrix `m`'s bit-reversed LDE row `bitrev_row` (its `width(m)`
    /// committed columns) to `out`. `bitrev_row in [0, 2^log_height(m))`.
    fn append_row(&self, m: usize, bitrev_row: usize, out: &mut Vec<FieldElement<E>>);
}

/// One committed matrix borrowed from a retained LDE buffer. Resolves each
/// bit-reversed row on demand (mapping through `reverse_index`) so the MMCS owns
/// no copy of the evaluations. See [`LeafSource`].
pub enum BorrowedMatrix<'a, E: IsField> {
    /// A `stride`-wide, row-major, NATURAL-order LDE buffer (the main / aux LDE
    /// retained in `Round1::lde_trace`). This matrix occupies columns
    /// `[col_start, col_start + width)`; its bit-reversed row `r` lives at
    /// natural-order row `reverse_index(r, 2^log_height)`.
    RowMajorNatural {
        data: &'a [FieldElement<E>],
        stride: usize,
        col_start: usize,
        width: usize,
        log_height: usize,
    },
    /// Column-major NATURAL-order columns (the composition-poly LDE retained in
    /// `Round2::lde_composition_poly_evaluations`): `cols[c][nat]` is column `c`
    /// at natural-order row `nat`. Every committed column is used.
    ColMajorNatural {
        cols: &'a [Vec<FieldElement<E>>],
        log_height: usize,
    },
}

impl<E: IsField> BorrowedMatrix<'_, E> {
    fn log_height(&self) -> usize {
        match self {
            BorrowedMatrix::RowMajorNatural { log_height, .. }
            | BorrowedMatrix::ColMajorNatural { log_height, .. } => *log_height,
        }
    }

    fn width(&self) -> usize {
        match self {
            BorrowedMatrix::RowMajorNatural { width, .. } => *width,
            BorrowedMatrix::ColMajorNatural { cols, .. } => cols.len(),
        }
    }

    fn append_row(&self, bitrev_row: usize, out: &mut Vec<FieldElement<E>>) {
        match self {
            BorrowedMatrix::RowMajorNatural {
                data,
                stride,
                col_start,
                width,
                log_height,
            } => {
                let nat = reverse_index(bitrev_row, 1u64 << log_height);
                let base = nat * stride + col_start;
                out.extend_from_slice(&data[base..base + width]);
            }
            BorrowedMatrix::ColMajorNatural { cols, log_height } => {
                let nat = reverse_index(bitrev_row, 1u64 << log_height);
                for col in cols.iter() {
                    out.push(col[nat].clone());
                }
            }
        }
    }
}

impl<E: IsField> LeafSource<E> for Vec<BorrowedMatrix<'_, E>> {
    fn num_matrices(&self) -> usize {
        self.len()
    }
    fn log_height(&self, m: usize) -> usize {
        self[m].log_height()
    }
    fn width(&self, m: usize) -> usize {
        self[m].width()
    }
    fn append_row(&self, m: usize, bitrev_row: usize, out: &mut Vec<FieldElement<E>>) {
        self[m].append_row(bitrev_row, out);
    }
}

/// A committed mixed-height, row-pair MMCS. Stores ONLY the digest layers (to
/// serve the shared authentication path) plus each matrix's `(log_height, width)`
/// (to locate leaves). The row DATA is served on demand by the caller's
/// [`LeafSource`] — the MMCS never owns a copy of the LDE.
pub struct MixedMmcs<E: IsField> {
    root: Commitment,
    /// `layers[0]` is the base digest layer; `layers[h_max-1] == [root]`.
    layers: Vec<Vec<Commitment>>,
    /// Per committed matrix, in input order: `(log_height, width)`.
    dims: Vec<(usize, usize)>,
    h_max: usize,
    _marker: PhantomData<E>,
}

/// The opening of ALL matrices at one query index, authenticated by a single
/// shared Merkle path.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct MixedOpening<E: IsField> {
    /// The one authentication path covering every matrix's row at the query.
    pub proof: Proof<Commitment>,
    /// Per-matrix row pair (in the same INPUT order as `commit`). Each entry's
    /// own `proof` is empty — `MixedOpening::proof` is the authenticator.
    pub per_matrix: Vec<PolynomialOpenings<E>>,
}

/// Hash the row pair `(row(2*leaf), row(2*leaf+1))` of every matrix whose index
/// is in `group` (in the given order), all columns batched, into one digest.
/// Rows are pulled from `source` — the MMCS owns no copy.
fn hash_group_leaf<E, S>(source: &S, group: &[usize], leaf: usize) -> Commitment
where
    E: IsField,
    S: LeafSource<E>,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let mut buf: Vec<FieldElement<E>> = Vec::new();
    for &m in group {
        source.append_row(m, 2 * leaf, &mut buf);
        source.append_row(m, 2 * leaf + 1, &mut buf);
    }
    <BatchedMerkleTreeBackend<E> as IsMerkleTreeBackend>::hash_data(&buf)
}

/// Verifier-side analogue of [`hash_group_leaf`]: hash the opened row pairs of a
/// group of openings (in the given order) into one digest.
fn hash_group_openings<E>(group: &[&PolynomialOpenings<E>]) -> Commitment
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let mut buf: Vec<FieldElement<E>> = Vec::new();
    for o in group {
        buf.extend_from_slice(&o.evaluations);
        buf.extend_from_slice(&o.evaluations_sym);
    }
    <BatchedMerkleTreeBackend<E> as IsMerkleTreeBackend>::hash_data(&buf)
}

#[inline]
fn compress<E>(left: &Commitment, right: &Commitment) -> Commitment
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send,
{
    <BatchedMerkleTreeBackend<E> as IsMerkleTreeBackend>::hash_new_parent(left, right)
}

impl<E> MixedMmcs<E>
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send,
{
    /// Commit the matrices supplied by `source` into one mixed-height row-pair
    /// tree, storing only the digest layers. See the module docs for the exact
    /// leaf/injection layout. `source` provides each matrix's dimensions and its
    /// bit-reversed rows on demand; no copy of the evaluations is retained.
    ///
    /// Leaf hashing (the base layer and each injected climb layer) is parallel
    /// across leaves via [`crate::par::par_map_collect`]; the per-level output is
    /// index-ordered, so the root and layers are byte-identical to a sequential
    /// build. `S: Sync` lets leaf closures read `source` from worker threads.
    pub fn commit<S: LeafSource<E> + Sync>(source: &S) -> Self {
        let num_matrices = source.num_matrices();
        assert!(
            num_matrices > 0,
            "MixedMmcs::commit requires at least one matrix"
        );

        let dims: Vec<(usize, usize)> = (0..num_matrices)
            .map(|m| {
                let log_height = source.log_height(m);
                assert!(
                    log_height >= 1,
                    "log_height must be >= 1 (row-pair leaves need at least 2 rows)"
                );
                (log_height, source.width(m))
            })
            .collect();

        let h_max = dims
            .iter()
            .map(|(log_height, _)| *log_height)
            .max()
            .expect("dims is non-empty");
        let n0 = 1usize << (h_max - 1);

        // Base digest layer: batch all tallest matrices' row pairs (input order).
        let base_group: Vec<usize> = (0..num_matrices).filter(|&m| dims[m].0 == h_max).collect();

        let mut layers: Vec<Vec<Commitment>> = Vec::with_capacity(h_max);
        // Base layer: 2^(h_max-1) independent group-leaf hashes — the bulk of the
        // tree's hashing (half of all nodes). Parallel across leaves.
        let base: Vec<Commitment> =
            crate::par::par_map_collect(0..n0, |k| hash_group_leaf(source, &base_group, k));
        layers.push(base);

        // Climb, compressing pairs and injecting shorter matrices where the layer
        // width matches their leaf count. Each level's nodes are independent
        // (they read only the previous, already-materialized layer), so parallel
        // across nodes; levels stay sequential.
        let mut i = 0usize;
        while layers[i].len() > 1 {
            let next_len = layers[i].len() / 2;
            let inject_h = h_max - 1 - i;
            let inject_group: Vec<usize> = (0..num_matrices)
                .filter(|&m| dims[m].0 == inject_h)
                .collect();

            let cur = &layers[i];
            let next: Vec<Commitment> = crate::par::par_map_collect(0..next_len, |j| {
                let mut parent = compress::<E>(&cur[2 * j], &cur[2 * j + 1]);
                if !inject_group.is_empty() {
                    let inj = hash_group_leaf(source, &inject_group, j);
                    parent = compress::<E>(&parent, &inj);
                }
                parent
            });
            layers.push(next);
            i += 1;
        }

        let root = layers.last().expect("at least the base layer exists")[0];

        MixedMmcs {
            root,
            layers,
            dims,
            h_max,
            _marker: PhantomData,
        }
    }

    /// The committed root.
    pub fn root(&self) -> Commitment {
        self.root
    }

    /// Open all matrices at query `iota in [0, 2^(h_max-1))`, returning each
    /// matrix's row pair plus one shared authentication path. Row data is served
    /// by `source`, which MUST describe the same matrices (same order and
    /// dimensions) as the one passed to [`Self::commit`].
    pub fn open_batch<S: LeafSource<E>>(&self, iota: usize, source: &S) -> MixedOpening<E> {
        let n0 = 1usize << (self.h_max - 1);
        assert!(iota < n0, "iota {iota} out of range (n0 = {n0})");
        debug_assert_eq!(
            source.num_matrices(),
            self.dims.len(),
            "leaf source matrix count must match the committed tree"
        );

        let per_matrix: Vec<PolynomialOpenings<E>> = (0..self.dims.len())
            .map(|m| {
                let (log_height, width) = self.dims[m];
                debug_assert_eq!(source.log_height(m), log_height);
                debug_assert_eq!(source.width(m), width);
                let k = iota >> (self.h_max - log_height);
                let mut evaluations = Vec::with_capacity(width);
                source.append_row(m, 2 * k, &mut evaluations);
                let mut evaluations_sym = Vec::with_capacity(width);
                source.append_row(m, 2 * k + 1, &mut evaluations_sym);
                PolynomialOpenings {
                    proof: Proof {
                        merkle_path: Vec::new(),
                    },
                    evaluations,
                    evaluations_sym,
                }
            })
            .collect();

        let mut merkle_path = Vec::with_capacity(self.h_max - 1);
        for level in 0..(self.h_max - 1) {
            let sibling = (iota >> level) ^ 1;
            merkle_path.push(self.layers[level][sibling]);
        }

        MixedOpening {
            proof: Proof { merkle_path },
            per_matrix,
        }
    }

    /// Verify a batched opening at `iota` against `root`. `heights[m]` is the
    /// `log_height` of matrix `m` and `widths[m]` its column count, both in the
    /// SAME order as `opening.per_matrix`.
    ///
    /// `widths` binds each matrix's boundary inside the per-height-group leaf
    /// hash (see the module `# Width binding` section): the group leaf hashes the
    /// FLAT concatenation of every matrix's `evaluations ‖ evaluations_sym`, so
    /// without fixed widths a prover could shift a matrix boundary while keeping
    /// the flat bytes — and thus the hash — identical. Pinning `widths` makes the
    /// boundaries unambiguous and closes that forgery.
    pub fn verify_batch(
        root: &Commitment,
        iota: usize,
        opening: &MixedOpening<E>,
        heights: &[usize],
        widths: &[usize],
    ) -> bool {
        if opening.per_matrix.len() != heights.len()
            || heights.len() != widths.len()
            || heights.is_empty()
        {
            return false;
        }
        // Bind per-matrix boundaries: every opened matrix must present exactly
        // `widths[m]` columns in BOTH rows of its pair. A boundary shift keeps the
        // flat per-group concatenation identical but changes these lengths.
        for (o, w) in opening.per_matrix.iter().zip(widths.iter()) {
            if o.evaluations.len() != *w || o.evaluations_sym.len() != *w {
                return false;
            }
        }
        let h_max = *heights.iter().max().expect("heights is non-empty");
        // Defensive: honest heights are >= 1 (row-pair leaves need >= 2 rows), so
        // h_max >= 1; guard the `h_max - 1` underflow regardless.
        if h_max == 0 {
            return false;
        }
        // Only the low `h_max - 1` bits of `iota` are consumed (one per level).
        if opening.proof.merkle_path.len() != h_max - 1 {
            return false;
        }

        // Base node: batch all tallest matrices' opened row pairs (input order).
        let base_group: Vec<&PolynomialOpenings<E>> = opening
            .per_matrix
            .iter()
            .zip(heights.iter())
            .filter(|(_, h)| **h == h_max)
            .map(|(o, _)| o)
            .collect();
        let mut acc = hash_group_openings(&base_group);

        for level in 0..(h_max - 1) {
            let sibling = &opening.proof.merkle_path[level];
            let bit = (iota >> level) & 1;
            let mut parent = if bit == 0 {
                compress::<E>(&acc, sibling)
            } else {
                compress::<E>(sibling, &acc)
            };

            // Inject matrices whose leaf count matches this (halved) layer, in
            // INPUT order — mirroring `commit`'s climb exactly.
            let inject_h = h_max - 1 - level;
            let inject_group: Vec<&PolynomialOpenings<E>> = opening
                .per_matrix
                .iter()
                .zip(heights.iter())
                .filter(|(_, h)| **h == inject_h)
                .map(|(o, _)| o)
                .collect();
            if !inject_group.is_empty() {
                let inj = hash_group_openings(&inject_group);
                parent = compress::<E>(&parent, &inj);
            }
            acc = parent;
        }

        &acc == root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::commit_bit_reversed;
    use math::field::element::FieldElement;
    use math::field::goldilocks::GoldilocksField;

    type FE = FieldElement<GoldilocksField>;

    /// Reference [`LeafSource`] owning bit-reversed row-major matrices — the exact
    /// row layout the pre-refactor `MixedMmcs` stored internally. Every existing
    /// test commits/opens through this, so a byte-parity assertion against
    /// `commit_bit_reversed` still pins the tree contract; the equivalence test
    /// below cross-checks it against the borrowed (natural-order) sources the
    /// prover actually uses.
    struct OwnedMatrices<E: IsField> {
        /// Each entry: `(bit-reversed row-major data, log_height, width)`.
        mats: Vec<(Vec<FieldElement<E>>, usize, usize)>,
    }

    impl<E: IsField> LeafSource<E> for OwnedMatrices<E> {
        fn num_matrices(&self) -> usize {
            self.mats.len()
        }
        fn log_height(&self, m: usize) -> usize {
            self.mats[m].1
        }
        fn width(&self, m: usize) -> usize {
            self.mats[m].2
        }
        fn append_row(&self, m: usize, bitrev_row: usize, out: &mut Vec<FieldElement<E>>) {
            let (data, _log_height, width) = &self.mats[m];
            out.extend_from_slice(&data[bitrev_row * width..(bitrev_row + 1) * width]);
        }
    }

    fn owned(mats: Vec<(Vec<FE>, usize, usize)>) -> OwnedMatrices<GoldilocksField> {
        OwnedMatrices { mats }
    }

    /// Build a row-major, bit-reversed flat vec from column-major natural-order
    /// `columns`, matching the layout the existing trace commit consumes: row `j`
    /// of the output = `[col_0[br(j)], ..., col_{w-1}[br(j)]]` with
    /// `br = reverse_index(., num_rows)`.
    fn row_major_bit_reversed(columns: &[Vec<FE>], num_rows: usize) -> Vec<FE> {
        let width = columns.len();
        let mut out = vec![FE::from(0u64); num_rows * width];
        for (r, chunk) in out.chunks_exact_mut(width).enumerate() {
            let br = reverse_index(r, num_rows as u64);
            for (c, col) in columns.iter().enumerate() {
                chunk[c] = col[br];
            }
        }
        out
    }

    /// Build a row-major flat vec in NATURAL order (no bit reversal): row `r` =
    /// `[col_0[r], ..., col_{w-1}[r]]`. This is the layout the prover's
    /// `BorrowedMatrix::RowMajorNatural` reads (the retained main/aux LDE buffer).
    fn row_major_natural(columns: &[Vec<FE>], num_rows: usize) -> Vec<FE> {
        let width = columns.len();
        let mut out = vec![FE::from(0u64); num_rows * width];
        for (r, chunk) in out.chunks_exact_mut(width).enumerate() {
            for (c, col) in columns.iter().enumerate() {
                chunk[c] = col[r];
            }
        }
        out
    }

    fn make_columns(width: usize, num_rows: usize, seed: u64) -> Vec<Vec<FE>> {
        (0..width)
            .map(|c| {
                (0..num_rows)
                    .map(|r| {
                        FE::from(seed.wrapping_mul(31) + (c as u64) * 1009 + (r as u64) * 7 + 1)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn single_matrix_commit_open_verify_and_tamper() {
        let log_height = 2usize;
        let num_rows = 1usize << log_height;
        let width = 3usize;
        let columns = make_columns(width, num_rows, 5);
        let data = row_major_bit_reversed(&columns, num_rows);

        let src = owned(vec![(data.clone(), log_height, width)]);
        let mmcs = MixedMmcs::commit(&src);
        let heights = [log_height];
        let widths = [width];
        let n0 = 1usize << (log_height - 1);

        for iota in 0..n0 {
            let opening = mmcs.open_batch(iota, &src);
            assert_eq!(opening.per_matrix.len(), 1);
            let k = iota;
            let row_2k = data[(2 * k) * width..(2 * k + 1) * width].to_vec();
            let row_2k1 = data[(2 * k + 1) * width..(2 * k + 2) * width].to_vec();
            assert_eq!(opening.per_matrix[0].evaluations, row_2k);
            assert_eq!(opening.per_matrix[0].evaluations_sym, row_2k1);
            assert!(MixedMmcs::verify_batch(
                &mmcs.root(),
                iota,
                &opening,
                &heights,
                &widths
            ));
        }

        let mut opening = mmcs.open_batch(0, &src);
        opening.per_matrix[0].evaluations[0] =
            &opening.per_matrix[0].evaluations[0] + &FE::from(1u64);
        assert!(!MixedMmcs::verify_batch(
            &mmcs.root(),
            0,
            &opening,
            &heights,
            &widths
        ));
    }

    #[test]
    fn single_matrix_root_matches_existing_row_pair_tree() {
        let log_height = 3usize;
        let num_rows = 1usize << log_height;
        let width = 4usize;
        let columns = make_columns(width, num_rows, 9);

        let (_, existing_root) =
            commit_bit_reversed(&columns, 2).expect("non-empty columns build a tree");

        let data = row_major_bit_reversed(&columns, num_rows);
        let mmcs = MixedMmcs::commit(&owned(vec![(data, log_height, width)]));

        assert_eq!(mmcs.root(), existing_root);
    }

    #[test]
    fn mixed_height_open_positions_verify_and_tamper() {
        // Three matrices, log_heights {5, 5, 3}, widths {2, 1, 4}.
        let (ha, hb, hc) = (5usize, 5usize, 3usize);
        let (wa, wb, wc) = (2usize, 1usize, 4usize);
        let a = row_major_bit_reversed(&make_columns(wa, 1 << ha, 1), 1 << ha);
        let b = row_major_bit_reversed(&make_columns(wb, 1 << hb, 2), 1 << hb);
        let c = row_major_bit_reversed(&make_columns(wc, 1 << hc, 3), 1 << hc);

        let src = owned(vec![
            (a.clone(), ha, wa),
            (b.clone(), hb, wb),
            (c.clone(), hc, wc),
        ]);
        let mmcs = MixedMmcs::commit(&src);
        let heights = [ha, hb, hc];
        let widths = [wa, wb, wc];
        let h_max = 5usize;
        let n0 = 1usize << (h_max - 1); // 16

        let row = |data: &[FE], w: usize, r: usize| data[r * w..(r + 1) * w].to_vec();

        for iota in [0usize, 1, 2, 3, 7, 8, 13, n0 - 1] {
            let opening = mmcs.open_batch(iota, &src);
            assert_eq!(opening.per_matrix.len(), 3);

            // Tall matrices open at k = iota >> 0 = iota.
            assert_eq!(opening.per_matrix[0].evaluations, row(&a, wa, 2 * iota));
            assert_eq!(
                opening.per_matrix[0].evaluations_sym,
                row(&a, wa, 2 * iota + 1)
            );
            assert_eq!(opening.per_matrix[1].evaluations, row(&b, wb, 2 * iota));

            // Height-3 matrix opens at k = iota >> (5 - 3) = iota >> 2.
            let kc = iota >> (h_max - hc);
            assert_eq!(opening.per_matrix[2].evaluations, row(&c, wc, 2 * kc));
            assert_eq!(
                opening.per_matrix[2].evaluations_sym,
                row(&c, wc, 2 * kc + 1)
            );

            assert!(
                MixedMmcs::verify_batch(&mmcs.root(), iota, &opening, &heights, &widths),
                "honest opening at iota={iota} must verify"
            );
        }

        // Tamper the height-3 matrix's opened row -> rejection (proves the short
        // matrix is bound by the shared path via injection).
        let iota = 6usize;
        let mut opening = mmcs.open_batch(iota, &src);
        opening.per_matrix[2].evaluations[0] =
            &opening.per_matrix[2].evaluations[0] + &FE::from(1u64);
        assert!(
            !MixedMmcs::verify_batch(&mmcs.root(), iota, &opening, &heights, &widths),
            "tampered height-3 row must be rejected"
        );

        // Tamper a tall-matrix row too -> rejection.
        let mut opening2 = mmcs.open_batch(iota, &src);
        opening2.per_matrix[0].evaluations[0] =
            &opening2.per_matrix[0].evaluations[0] + &FE::from(1u64);
        assert!(
            !MixedMmcs::verify_batch(&mmcs.root(), iota, &opening2, &heights, &widths),
            "tampered tall-matrix row must be rejected"
        );
    }

    /// Vector test: hand-compute the root for `{log_height 2, log_height 1}`
    /// matrices per the documented layout and assert equality. Pins the
    /// leaf/injection contract consumed verbatim by Task 7, plus determinism.
    #[test]
    fn vector_root_layout_contract_and_determinism() {
        // A: log_height 2 (4 rows), width 2 ; B: log_height 1 (2 rows), width 3.
        let a_data = row_major_bit_reversed(&make_columns(2, 4, 3), 4);
        let b_data = row_major_bit_reversed(&make_columns(3, 2, 8), 2);

        let src = owned(vec![(a_data.clone(), 2, 2), (b_data.clone(), 1, 3)]);
        let mmcs = MixedMmcs::commit(&src);

        // Hand recomputation via the backend primitives, in the documented order.
        let arow = |r: usize| a_data[r * 2..(r + 1) * 2].to_vec();
        let brow = |r: usize| b_data[r * 3..(r + 1) * 3].to_vec();
        let h = |v: Vec<FE>| {
            <BatchedMerkleTreeBackend<GoldilocksField> as IsMerkleTreeBackend>::hash_data(&v)
        };

        // Base layer (matrix A only): leaf k = H(A.row(2k) || A.row(2k+1)).
        let mut leaf0 = arow(0);
        leaf0.extend(arow(1));
        let mut leaf1 = arow(2);
        leaf1.extend(arow(3));
        let l00 = h(leaf0);
        let l01 = h(leaf1);

        // Climb to layer 1 (root): compress the base pair, then inject B (h=1).
        let parent = compress::<GoldilocksField>(&l00, &l01);
        let mut binj = brow(0);
        binj.extend(brow(1));
        let inj = h(binj);
        let expected_root = compress::<GoldilocksField>(&parent, &inj);

        assert_eq!(
            mmcs.root(),
            expected_root,
            "root must match the hand-computed mixed-height layout"
        );

        // Determinism: a second commit over the same inputs yields the same root.
        let mmcs2 = MixedMmcs::commit(&owned(vec![(a_data, 2, 2), (b_data, 1, 3)]));
        assert_eq!(mmcs.root(), mmcs2.root(), "commit must be deterministic");

        for iota in 0..2usize {
            let opening = mmcs.open_batch(iota, &src);
            // heights {2, 1}, widths {2, 3}.
            assert!(MixedMmcs::verify_batch(
                &mmcs.root(),
                iota,
                &opening,
                &[2, 1],
                &[2, 3]
            ));
        }
    }

    /// Two SAME-HEIGHT matrices share one base-group leaf, whose hash is over the
    /// FLAT concatenation `A.eval ‖ A.eval_sym ‖ B.eval ‖ B.eval_sym`. A malicious
    /// prover can shift the A|A_sym boundary (move one element from A's
    /// `evaluations_sym` into A's `evaluations`) leaving that flat concatenation —
    /// and hence the leaf hash — byte-identical, so a width-blind `verify_batch`
    /// would accept it. The per-matrix width binding rejects the shift.
    #[test]
    fn boundary_shift_forgery_rejected() {
        let h = 2usize;
        let num_rows = 1usize << h;
        let (wa, wb) = (2usize, 1usize); // wA >= 2 so we can steal one column.
        let a = row_major_bit_reversed(&make_columns(wa, num_rows, 11), num_rows);
        let b = row_major_bit_reversed(&make_columns(wb, num_rows, 22), num_rows);

        let src = owned(vec![(a, h, wa), (b, h, wb)]);
        let mmcs = MixedMmcs::commit(&src);
        let heights = [h, h];
        let widths = [wa, wb];

        let iota = 0usize;
        let opening = mmcs.open_batch(iota, &src);
        assert!(
            MixedMmcs::verify_batch(&mmcs.root(), iota, &opening, &heights, &widths),
            "honest opening must verify"
        );

        // Forge: lengthen A.evaluations by one element taken from A.evaluations_sym.
        let mut forged = mmcs.open_batch(iota, &src);
        let moved = forged.per_matrix[0].evaluations_sym.remove(0);
        forged.per_matrix[0].evaluations.push(moved);

        // The FLAT per-group concatenation is byte-identical to the honest one, so
        // the group leaf hash is UNCHANGED — the rejection must come from the width
        // check, not from a differing hash.
        let flat = |o: &MixedOpening<GoldilocksField>| -> Vec<FE> {
            let mut v = Vec::new();
            for m in &o.per_matrix {
                v.extend_from_slice(&m.evaluations);
                v.extend_from_slice(&m.evaluations_sym);
            }
            v
        };
        assert_eq!(
            flat(&opening),
            flat(&forged),
            "the flat concatenation must be byte-identical (boundary-only shift)"
        );

        assert!(
            !MixedMmcs::verify_batch(&mmcs.root(), iota, &forged, &heights, &widths),
            "boundary-shift forgery must be rejected by the width binding"
        );
    }

    /// Extension-field (Fp3) coverage: the aux/composition matrices Tasks 2/3 feed
    /// the MMCS are cubic-extension. Byte-parity cross-check of a single Fp3 matrix
    /// against the existing per-table row-pair tree, plus an open/verify/tamper
    /// roundtrip over the extension path.
    #[test]
    fn single_matrix_fp3_root_matches_existing_row_pair_tree() {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Fp3;
        type F3 = FieldElement<Fp3>;

        let log_height = 3usize;
        let num_rows = 1usize << log_height;
        let width = 3usize;

        // Populate ALL three components so the 24-byte extension serialization is
        // exercised (not just the embedded-base subset).
        let columns: Vec<Vec<F3>> = (0..width)
            .map(|c| {
                (0..num_rows)
                    .map(|r| {
                        F3::new([
                            FE::from((c as u64) * 7 + r as u64 + 1),
                            FE::from((r as u64) * 13 + 2),
                            FE::from((c as u64) * 5 + (r as u64) * 3 + 4),
                        ])
                    })
                    .collect()
            })
            .collect();

        let (_, existing_root) =
            commit_bit_reversed(&columns, 2).expect("non-empty columns build a tree");

        // Row-major bit-reversed equivalent of the same column-major data.
        let mut data = vec![F3::zero(); num_rows * width];
        for (r, chunk) in data.chunks_exact_mut(width).enumerate() {
            let br = reverse_index(r, num_rows as u64);
            for (c, col) in columns.iter().enumerate() {
                chunk[c] = col[br];
            }
        }

        let src = OwnedMatrices {
            mats: vec![(data, log_height, width)],
        };
        let mmcs = MixedMmcs::commit(&src);
        assert_eq!(
            mmcs.root(),
            existing_root,
            "Fp3 single-matrix root must match the existing row-pair tree"
        );

        let heights = [log_height];
        let widths = [width];
        for iota in 0..(1usize << (log_height - 1)) {
            let opening = mmcs.open_batch(iota, &src);
            assert!(MixedMmcs::verify_batch(
                &mmcs.root(),
                iota,
                &opening,
                &heights,
                &widths
            ));
        }

        let mut opening = mmcs.open_batch(0, &src);
        opening.per_matrix[0].evaluations[0] = &opening.per_matrix[0].evaluations[0] + &F3::one();
        assert!(!MixedMmcs::verify_batch(
            &mmcs.root(),
            0,
            &opening,
            &heights,
            &widths
        ));
    }

    /// Equivalence (the soundness contract the batched prover relies on): the
    /// digest-only MMCS built from the prover's borrowed, NATURAL-order leaf
    /// sources yields the SAME root and the SAME opened rows as the reference
    /// owning source over the bit-reversed data — for the row-major (main / aux)
    /// layout, the column-major (composition) layout, AND a main-split column
    /// sub-range (`col_start > 0`). Only the leaf-byte source changes; nothing the
    /// verifier sees does.
    #[test]
    fn borrowed_sources_match_owned_reference() {
        // Mixed heights {5, 5, 3}; the height-3 matrix exercises injection.
        let specs = [(5usize, 3usize, 100u64), (5, 1, 200), (3, 4, 300)];

        // Column-major natural-order columns per matrix.
        let cols: Vec<Vec<Vec<FE>>> = specs
            .iter()
            .map(|&(lh, w, seed)| make_columns(w, 1 << lh, seed))
            .collect();

        // Reference: owned, bit-reversed row-major (the pre-refactor layout).
        let owned_src = owned(
            specs
                .iter()
                .zip(cols.iter())
                .map(|(&(lh, w, _), c)| (row_major_bit_reversed(c, 1 << lh), lh, w))
                .collect(),
        );

        // Borrowed row-major NATURAL (the retained main / aux LDE buffer).
        let rm_natural: Vec<Vec<FE>> = specs
            .iter()
            .zip(cols.iter())
            .map(|(&(lh, _, _), c)| row_major_natural(c, 1 << lh))
            .collect();
        let rm_src: Vec<BorrowedMatrix<GoldilocksField>> = specs
            .iter()
            .zip(rm_natural.iter())
            .map(|(&(lh, w, _), data)| BorrowedMatrix::RowMajorNatural {
                data: data.as_slice(),
                stride: w,
                col_start: 0,
                width: w,
                log_height: lh,
            })
            .collect();

        // Borrowed column-major NATURAL (the retained composition-poly LDE).
        let cm_src: Vec<BorrowedMatrix<GoldilocksField>> = specs
            .iter()
            .zip(cols.iter())
            .map(|(&(lh, _, _), c)| BorrowedMatrix::ColMajorNatural {
                cols: c.as_slice(),
                log_height: lh,
            })
            .collect();

        let owned_mmcs = MixedMmcs::commit(&owned_src);
        let rm_mmcs = MixedMmcs::commit(&rm_src);
        let cm_mmcs = MixedMmcs::commit(&cm_src);
        assert_eq!(
            owned_mmcs.root(),
            rm_mmcs.root(),
            "row-major natural root must match the owned reference"
        );
        assert_eq!(
            owned_mmcs.root(),
            cm_mmcs.root(),
            "column-major natural root must match the owned reference"
        );

        let n0 = 1usize << (5 - 1);
        for iota in 0..n0 {
            let o = owned_mmcs.open_batch(iota, &owned_src);
            let rm = rm_mmcs.open_batch(iota, &rm_src);
            let cm = cm_mmcs.open_batch(iota, &cm_src);
            assert_eq!(o.proof.merkle_path, rm.proof.merkle_path);
            assert_eq!(o.proof.merkle_path, cm.proof.merkle_path);
            for i in 0..specs.len() {
                assert_eq!(o.per_matrix[i].evaluations, rm.per_matrix[i].evaluations);
                assert_eq!(
                    o.per_matrix[i].evaluations_sym,
                    rm.per_matrix[i].evaluations_sym
                );
                assert_eq!(o.per_matrix[i].evaluations, cm.per_matrix[i].evaluations);
                assert_eq!(
                    o.per_matrix[i].evaluations_sym,
                    cm.per_matrix[i].evaluations_sym
                );
            }
        }

        // Main-split sub-range: a RowMajorNatural over a wider buffer with a
        // leading prefix (`col_start = prefix`) must match an owned matrix built
        // over ONLY the committed trailing columns.
        let (lh, prefix, w) = (4usize, 2usize, 3usize);
        let num_rows = 1usize << lh;
        let full = make_columns(prefix + w, num_rows, 42);
        let full_natural = row_major_natural(&full, num_rows);
        let sub_cols: Vec<Vec<FE>> = full[prefix..].to_vec();
        let sub_owned = owned(vec![(row_major_bit_reversed(&sub_cols, num_rows), lh, w)]);
        let split_src: Vec<BorrowedMatrix<GoldilocksField>> =
            vec![BorrowedMatrix::RowMajorNatural {
                data: full_natural.as_slice(),
                stride: prefix + w,
                col_start: prefix,
                width: w,
                log_height: lh,
            }];
        let sub_owned_mmcs = MixedMmcs::commit(&sub_owned);
        let split_mmcs = MixedMmcs::commit(&split_src);
        assert_eq!(
            sub_owned_mmcs.root(),
            split_mmcs.root(),
            "main-split (col_start>0) root must match the owned sub-range"
        );
        for iota in 0..(1usize << (lh - 1)) {
            let a = sub_owned_mmcs.open_batch(iota, &sub_owned);
            let b = split_mmcs.open_batch(iota, &split_src);
            assert_eq!(a.per_matrix[0].evaluations, b.per_matrix[0].evaluations);
            assert_eq!(
                a.per_matrix[0].evaluations_sym,
                b.per_matrix[0].evaluations_sym
            );
        }
    }
}

//! Mixed-height, row-pair MMCS (Merkle Mixed Commitment Scheme).
//!
//! Commits ALL of an epoch's matrices (one per table, of possibly different
//! heights) into ONE mixed-height Merkle tree, so a single query opens ONE
//! authentication path that covers every table's row at that query — the
//! proof-size / opening-path win of the unified-shard design (SP1 / OpenVM /
//! Plonky3). Mirrors Plonky3's `MerkleTreeMmcs`, adapted to the [`StarkHash`]
//! commitment configuration and to the row-pair `(x, -x)` leaf layout (#735).
//!
//! This is a standalone primitive: the prover and verifier do not build epoch
//! commitments with it yet. The leaf and injection layout documented below is
//! the single source of truth for whoever wires it in.
//!
//! # Inputs
//!
//! [`MixedMmcs::commit`] reads matrices through a [`LeafSource`], which reports
//! each matrix's `(log_height, width)` and serves its rows on demand:
//! - `log_height`: `log2` of the row count; the matrix has `2^log_height` rows.
//! - `width`: number of committed columns.
//! - rows are addressed by **bit-reversed** LDE position (the same layout the
//!   per-table trace commit produces internally).
//!
//! # Row-pair leaves
//!
//! Leaf `k` of a matrix groups LDE positions `2k` and `2k+1` (the FRI fold pair
//! `x` and `-x`), all `width` columns batched. A matrix of `log_height h` has
//! `2^(h-1)` leaves. In [`MixedMmcs::open_batch`] / [`PolynomialOpenings`]:
//! `evaluations` = row `2k`, `evaluations_sym` = row `2k+1`.
//!
//! # Tree layout (the soundness-relevant contract)
//!
//! Let `h_max = max(log_height)`. The base digest layer (layer 0) has
//! `N0 = 2^(h_max-1)` nodes. Layer `i` has `N0 >> i` nodes; the root is the sole
//! node of layer `h_max-1`. A matrix of `log_height h` is *injected* at layer
//! index `i = h_max - h` (so the tallest matrices, `h == h_max`, populate the
//! base layer; shorter matrices enter where the layer width matches their leaf
//! count `2^(h-1)`).
//!
//! Hashing (`H = <H::Batched<E>>::hash_data` over a `Vec` of field elements;
//! `C = <H::Batched<E>>::hash_new_parent`, the 2-input compression — the same
//! two functions, on the same backend, that the existing per-table tree uses):
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
//! Because the leaf and parent hashes come from `H::Batched<E>` — the backend the
//! per-table row-pair tree already commits with — a single-matrix `MixedMmcs` is
//! byte-identical to that tree by construction, not by coincidence. There is no
//! second encoding of a leaf to keep in step.
//!
//! # Query opening
//!
//! For query `iota in [0, N0)`, matrix `m` is opened at leaf
//! `k_m = iota >> (h_max - h_m)` (`= iota >> i_m`). The shared authentication
//! path holds, for each level `level in [0, h_max-1)`, the sibling
//! `layer_level[(iota >> level) ^ 1]`. ONE path authenticates all matrices.
//! The per-matrix [`PolynomialOpenings::proof`] fields are empty; the single
//! [`MixedOpening::proof`] is the authenticator.
//!
//! # ★ Index convention — a HARD PRECONDITION on the caller
//!
//! `iota` is a leaf index **in THIS tree**: it must be drawn from
//! `[0, 2^(h_max-1))` where `h_max` is *this MMCS's* tallest matrix.
//! [`MixedMmcs::verify_batch`] walks the path with `(iota >> level) & 1`, i.e. it
//! consumes the **low** `h_max - 1` bits, while a shorter matrix inside the tree
//! is located by `iota >> (h_max - h_m)`, i.e. by the **high** bits. Both are
//! consistent only when the two `h_max` agree.
//!
//! A caller that batches several rounds under one shared FRI query index must
//! therefore reduce a global index before calling in:
//!
//! ```text
//! iota_round = iota_fri >> (h_max_fri - h_max_round)
//! ```
//!
//! Passing the un-reduced `iota_fri` to a round whose `h_max` is below the FRI's
//! is not a loud error — prover and verifier share this routine, so a wrong
//! convention is self-consistent: honest proofs still verify and the failure is
//! that short matrices end up authenticated at positions the FRI join never
//! checks. [`MixedMmcs::verify_batch`] rejects an `iota` outside `[0, 2^(h_max-1))`
//! to turn most of that class of misuse into a rejection rather than a silent
//! mis-binding, but the reduction remains the caller's obligation: an index that
//! happens to land in range is accepted at the wrong leaf.
//! `short_round_low_bit_convention_is_exercised` is the control on this.
//!
//! # Width binding (soundness)
//!
//! [`MixedMmcs::verify_batch`] takes per-matrix `widths` alongside `heights`.
//! Within a height group the leaf hash is over the FLAT concatenation of every
//! matrix's opened row pair (`A.eval ‖ A.eval_sym ‖ B.eval ‖ B.eval_sym ‖ …`),
//! which does NOT by itself record where each matrix's columns end. Fixing
//! `widths[m]` (matrix `m`'s column count) makes those boundaries unambiguous:
//! without it a prover could shift a boundary — e.g. lengthen one matrix's
//! `evaluations` by one element and shorten its `evaluations_sym` by one —
//! leaving the flat bytes (and therefore the group hash) identical while feeding
//! a corrupted row downstream. Consumers MUST pass the committed public
//! per-table column counts, in the same INPUT order as `heights`, derived from
//! the AIR set rather than read out of the proof.
//!
//! `heights` and `widths` must ALSO be bound into the Fiat-Shamir transcript by
//! the consumer, before any challenge that depends on the epoch's shape — see
//! [`crate::fri::batched::absorb_shape_histogram`], which is the canonical
//! encoding of that binding.
//!
//! # Determinism
//!
//! The tree is a pure function of `(matrices, input order)`. Grouping within a
//! height (base batching and injection) follows INPUT order; the prover and
//! verifier MUST pass matrices and `heights` in the same per-epoch order.
//!
//! # Memory: what the caller may drop, and when
//!
//! The MMCS owns no evaluations. It stores the digest layers
//! (`O(2^(h_max-1))` nodes) plus each matrix's `(log_height, width)`; rows are
//! pulled through [`LeafSource`] both at commit and at open time. Two properties
//! follow, and `commit_reads_each_height_group_in_one_contiguous_phase` is the
//! control on the second:
//!
//! - `commit` reads matrix `m`'s rows **only while building level
//!   `h_max - h_m`**, and levels are built in descending height order. A caller
//!   may therefore produce a height group's LDEs, commit, and drop them before
//!   the next group is needed.
//! - Within one height group the leaf is a single `hash_data` over the group's
//!   concatenated rows, so every matrix of that height must be *readable*
//!   simultaneously. That does not require them all to be resident — a
//!   `LeafSource` may serve rows from disk, from device memory, or by
//!   recomputation — but a caller that serves them from full in-RAM LDE buffers
//!   holds the whole group at once. Streaming *within* a height group would need
//!   an incremental leaf hasher (absorb matrix by matrix into one sponge per
//!   leaf), which the backend trait does not currently expose.

use core::marker::PhantomData;

use crypto::merkle_tree::proof::Proof;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math::fft::bit_reversing::reverse_index;
use math::field::element::FieldElement;
use math::field::traits::IsField;
use math::traits::AsBytes;

use crate::config::{Commitment, StarkHash};
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

/// A committed mixed-height, row-pair MMCS under the commitment configuration
/// `H`. Stores ONLY the digest layers (to serve the shared authentication path)
/// plus each matrix's `(log_height, width)` (to locate leaves). The row DATA is
/// served on demand by the caller's [`LeafSource`] — the MMCS never owns a copy
/// of the LDE.
pub struct MixedMmcs<E: IsField, H: StarkHash> {
    root: Commitment,
    /// `layers[0]` is the base digest layer; `layers[h_max-1] == [root]`.
    layers: Vec<Vec<Commitment>>,
    /// Per committed matrix, in input order: `(log_height, width)`.
    dims: Vec<(usize, usize)>,
    h_max: usize,
    _marker: PhantomData<(E, H)>,
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
    /// own `proof` is empty — [`MixedOpening::proof`] is the authenticator.
    pub per_matrix: Vec<PolynomialOpenings<E>>,
}

/// Hash the row pair `(row(2*leaf), row(2*leaf+1))` of every matrix whose index
/// is in `group` (in the given order), all columns batched, into one digest.
/// Rows are pulled from `source` — the MMCS owns no copy.
fn hash_group_leaf<E, H, S>(source: &S, group: &[usize], leaf: usize) -> Commitment
where
    E: IsField + 'static,
    H: StarkHash,
    S: LeafSource<E>,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let mut buf: Vec<FieldElement<E>> = Vec::new();
    for &m in group {
        source.append_row(m, 2 * leaf, &mut buf);
        source.append_row(m, 2 * leaf + 1, &mut buf);
    }
    <H::Batched<E> as IsMerkleTreeBackend>::hash_data(&buf)
}

/// Verifier-side analogue of [`hash_group_leaf`]: hash the opened row pairs of a
/// group of openings (in the given order) into one digest.
fn hash_group_openings<E, H>(group: &[&PolynomialOpenings<E>]) -> Commitment
where
    E: IsField + 'static,
    H: StarkHash,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let mut buf: Vec<FieldElement<E>> = Vec::new();
    for o in group {
        buf.extend_from_slice(&o.evaluations);
        buf.extend_from_slice(&o.evaluations_sym);
    }
    <H::Batched<E> as IsMerkleTreeBackend>::hash_data(&buf)
}

#[inline]
fn compress<E, H>(left: &Commitment, right: &Commitment) -> Commitment
where
    E: IsField + 'static,
    H: StarkHash,
    FieldElement<E>: AsBytes + Sync + Send,
{
    <H::Batched<E> as IsMerkleTreeBackend>::hash_new_parent(left, right)
}

impl<E, H> MixedMmcs<E, H>
where
    E: IsField + 'static,
    H: StarkHash,
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
    ///
    /// Levels are built in descending height order and matrix `m` is read only
    /// while its own level is built, so the caller may release a height group's
    /// buffers once the next level starts — see the module's memory section.
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
        let base: Vec<Commitment> = crate::par::par_map_collect(0..n0, |k| {
            hash_group_leaf::<E, H, S>(source, &base_group, k)
        });
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
                let mut parent = compress::<E, H>(&cur[2 * j], &cur[2 * j + 1]);
                if !inject_group.is_empty() {
                    let inj = hash_group_leaf::<E, H, S>(source, &inject_group, j);
                    parent = compress::<E, H>(&parent, &inj);
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

    /// `log2` of the tallest committed matrix. The query index this MMCS accepts
    /// lives in `[0, 2^(h_max-1))` — see the module's index-convention section.
    pub fn h_max(&self) -> usize {
        self.h_max
    }

    /// Per committed matrix, in input order: `(log_height, width)`. The verifier
    /// is expected to rebuild these from the AIR set rather than read them here;
    /// this accessor exists so a prover can bind the shape it actually committed.
    pub fn dims(&self) -> &[(usize, usize)] {
        &self.dims
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
    /// SAME order as `opening.per_matrix`, and both supplied by the verifier from
    /// the AIR set rather than read out of the proof.
    ///
    /// `widths` binds each matrix's boundary inside the per-height-group leaf
    /// hash (see the module `# Width binding` section): the group leaf hashes the
    /// FLAT concatenation of every matrix's `evaluations ‖ evaluations_sym`, so
    /// without fixed widths a prover could shift a matrix boundary while keeping
    /// the flat bytes — and thus the hash — identical. Pinning `widths` makes the
    /// boundaries unambiguous and closes that forgery.
    ///
    /// `iota` must already be reduced to this tree's index space — see the
    /// module's index-convention section. Out-of-range indices are rejected here,
    /// but that check is a backstop, not a substitute for the reduction.
    ///
    /// Returns `false` on every malformed input; it never panics, so a verifier
    /// can call it on adversarial data.
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
        let Some(&h_max) = heights.iter().max() else {
            return false;
        };
        // Honest heights are >= 1 (row-pair leaves need >= 2 rows) and far below
        // the shift width; guard both ends rather than trust the proof's shape.
        if h_max == 0 || h_max >= usize::BITS as usize {
            return false;
        }
        // Only the low `h_max - 1` bits of `iota` are consumed (one per level), so
        // an index from a taller domain would authenticate the short matrices at a
        // position nothing else checks. Reject it instead.
        if iota >= 1usize << (h_max - 1) {
            return false;
        }
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
        let mut acc = hash_group_openings::<E, H>(&base_group);

        for level in 0..(h_max - 1) {
            let sibling = &opening.proof.merkle_path[level];
            let bit = (iota >> level) & 1;
            let mut parent = if bit == 0 {
                compress::<E, H>(&acc, sibling)
            } else {
                compress::<E, H>(sibling, &acc)
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
                let inj = hash_group_openings::<E, H>(&inject_group);
                parent = compress::<E, H>(&parent, &inj);
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
    use crate::config::DefaultStarkHash;
    use math::field::element::FieldElement;
    use math::field::goldilocks::GoldilocksField;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    type FE = FieldElement<GoldilocksField>;
    type Mmcs<E> = MixedMmcs<E, DefaultStarkHash>;

    /// Reference [`LeafSource`] owning bit-reversed row-major matrices. Every
    /// test commits/opens through this, so the byte-parity assertion against
    /// `commit_bit_reversed` pins the tree contract; `borrowed_sources_match_
    /// owned_reference` cross-checks it against the borrowed (natural-order)
    /// sources a prover would use.
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
        let mmcs = Mmcs::commit(&src);
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
            assert!(Mmcs::verify_batch(
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
        assert!(!Mmcs::verify_batch(
            &mmcs.root(),
            0,
            &opening,
            &heights,
            &widths
        ));
    }

    /// ★ The [`StarkHash`] backward-compatibility statement: a single-matrix MMCS
    /// IS the existing per-table row-pair tree. It holds by construction — both
    /// go through `H::Batched<E>`'s `hash_data` / `hash_new_parent` — and this
    /// pins that no second leaf encoding crept in.
    ///
    /// Both sides have to be the SAME `H` for that to mean anything, which is
    /// why this module commits under `DefaultStarkHash`: `commit_bit_reversed`
    /// is alias-pinned, so naming a fixed hash here compares two configurations
    /// and reports a hash difference as a layout difference. It did exactly that
    /// at the P-a flip, when the alias moved and this side did not.
    #[test]
    fn single_matrix_root_matches_existing_row_pair_tree() {
        let log_height = 3usize;
        let num_rows = 1usize << log_height;
        let width = 4usize;
        let columns = make_columns(width, num_rows, 9);

        let (_, existing_root) =
            commit_bit_reversed(&columns, 2).expect("non-empty columns build a tree");

        let data = row_major_bit_reversed(&columns, num_rows);
        let mmcs = Mmcs::commit(&owned(vec![(data, log_height, width)]));

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
        let mmcs = Mmcs::commit(&src);
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
                Mmcs::verify_batch(&mmcs.root(), iota, &opening, &heights, &widths),
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
            !Mmcs::verify_batch(&mmcs.root(), iota, &opening, &heights, &widths),
            "tampered height-3 row must be rejected"
        );

        // Tamper a tall-matrix row too -> rejection.
        let mut opening2 = mmcs.open_batch(iota, &src);
        opening2.per_matrix[0].evaluations[0] =
            &opening2.per_matrix[0].evaluations[0] + &FE::from(1u64);
        assert!(
            !Mmcs::verify_batch(&mmcs.root(), iota, &opening2, &heights, &widths),
            "tampered tall-matrix row must be rejected"
        );
    }

    /// Vector test: hand-compute the root for `{log_height 2, log_height 1}`
    /// matrices per the documented layout and assert equality. Pins the
    /// leaf/injection contract, plus determinism.
    #[test]
    fn vector_root_layout_contract_and_determinism() {
        // A: log_height 2 (4 rows), width 2 ; B: log_height 1 (2 rows), width 3.
        let a_data = row_major_bit_reversed(&make_columns(2, 4, 3), 4);
        let b_data = row_major_bit_reversed(&make_columns(3, 2, 8), 2);

        let src = owned(vec![(a_data.clone(), 2, 2), (b_data.clone(), 1, 3)]);
        let mmcs = Mmcs::commit(&src);

        // Hand recomputation via the backend primitives, in the documented order.
        let arow = |r: usize| a_data[r * 2..(r + 1) * 2].to_vec();
        let brow = |r: usize| b_data[r * 3..(r + 1) * 3].to_vec();
        let h = |v: Vec<FE>| {
            <<DefaultStarkHash as StarkHash>::Batched<GoldilocksField> as IsMerkleTreeBackend>::hash_data(&v)
        };

        // Base layer (matrix A only): leaf k = H(A.row(2k) || A.row(2k+1)).
        let mut leaf0 = arow(0);
        leaf0.extend(arow(1));
        let mut leaf1 = arow(2);
        leaf1.extend(arow(3));
        let l00 = h(leaf0);
        let l01 = h(leaf1);

        // Climb to layer 1 (root): compress the base pair, then inject B (h=1).
        let parent = compress::<GoldilocksField, DefaultStarkHash>(&l00, &l01);
        let mut binj = brow(0);
        binj.extend(brow(1));
        let inj = h(binj);
        let expected_root = compress::<GoldilocksField, DefaultStarkHash>(&parent, &inj);

        assert_eq!(
            mmcs.root(),
            expected_root,
            "root must match the hand-computed mixed-height layout"
        );

        // Determinism: a second commit over the same inputs yields the same root.
        let mmcs2 = Mmcs::commit(&owned(vec![(a_data, 2, 2), (b_data, 1, 3)]));
        assert_eq!(mmcs.root(), mmcs2.root(), "commit must be deterministic");

        for iota in 0..2usize {
            let opening = mmcs.open_batch(iota, &src);
            // heights {2, 1}, widths {2, 3}.
            assert!(Mmcs::verify_batch(
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
        let mmcs = Mmcs::commit(&src);
        let heights = [h, h];
        let widths = [wa, wb];

        let iota = 0usize;
        let opening = mmcs.open_batch(iota, &src);
        assert!(
            Mmcs::verify_batch(&mmcs.root(), iota, &opening, &heights, &widths),
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
            !Mmcs::verify_batch(&mmcs.root(), iota, &forged, &heights, &widths),
            "boundary-shift forgery must be rejected by the width binding"
        );
    }

    /// Extension-field (Fp3) coverage: the aux and composition matrices an epoch
    /// batches are cubic-extension. Byte-parity cross-check of a single Fp3 matrix
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
        let mmcs = MixedMmcs::<Fp3, DefaultStarkHash>::commit(&src);
        assert_eq!(
            mmcs.root(),
            existing_root,
            "Fp3 single-matrix root must match the existing row-pair tree"
        );

        let heights = [log_height];
        let widths = [width];
        for iota in 0..(1usize << (log_height - 1)) {
            let opening = mmcs.open_batch(iota, &src);
            assert!(MixedMmcs::<Fp3, DefaultStarkHash>::verify_batch(
                &mmcs.root(),
                iota,
                &opening,
                &heights,
                &widths
            ));
        }

        let mut opening = mmcs.open_batch(0, &src);
        opening.per_matrix[0].evaluations[0] = &opening.per_matrix[0].evaluations[0] + &F3::one();
        assert!(!MixedMmcs::<Fp3, DefaultStarkHash>::verify_batch(
            &mmcs.root(),
            0,
            &opening,
            &heights,
            &widths
        ));
    }

    /// Equivalence (the soundness contract a batched prover relies on): the
    /// digest-only MMCS built from borrowed, NATURAL-order leaf sources yields the
    /// SAME root and the SAME opened rows as the reference owning source over the
    /// bit-reversed data — for the row-major (main / aux) layout, the column-major
    /// (composition) layout, AND a main-split column sub-range (`col_start > 0`).
    /// Only the leaf-byte source changes; nothing the verifier sees does.
    #[test]
    fn borrowed_sources_match_owned_reference() {
        // Mixed heights {5, 5, 3}; the height-3 matrix exercises injection.
        let specs = [(5usize, 3usize, 100u64), (5, 1, 200), (3, 4, 300)];

        // Column-major natural-order columns per matrix.
        let cols: Vec<Vec<Vec<FE>>> = specs
            .iter()
            .map(|&(lh, w, seed)| make_columns(w, 1 << lh, seed))
            .collect();

        // Reference: owned, bit-reversed row-major.
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

        let owned_mmcs = Mmcs::commit(&owned_src);
        let rm_mmcs = Mmcs::commit(&rm_src);
        let cm_mmcs = Mmcs::commit(&cm_src);
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
        let sub_owned_mmcs = Mmcs::commit(&sub_owned);
        let split_mmcs = Mmcs::commit(&split_src);
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

    /// ★ The index-convention control (the module's "HARD PRECONDITION" section).
    ///
    /// A round whose tallest matrix is SHORTER than the FRI's tallest is the case
    /// where the two index conventions disagree: `verify_batch` consumes the LOW
    /// `h_max_round - 1` bits of whatever index it is handed, while a matrix
    /// inside the tree is located by the HIGH bits of the FRI index. This asserts
    /// three things about that case:
    ///
    /// 1. honest-path control — the correctly reduced index verifies;
    /// 2. a tampered row of a SHORT (injected) matrix is rejected, so the low-bits
    ///    walk really does authenticate the short matrices at the reduced index;
    /// 3. handing the un-reduced FRI index straight in is rejected — the misuse is
    ///    detectable, not silently accepted at some other leaf.
    ///
    /// A tamper control on the tallest matrix alone would pass under either
    /// convention and catch none of this.
    #[test]
    fn short_round_low_bit_convention_is_exercised() {
        // A hypothetical FRI over a 2^6 domain: iota_fri in [0, 2^5).
        let h_max_fri = 6usize;
        // This round's matrices are shorter: heights {4, 2}.
        let (h_tall, h_short) = (4usize, 2usize);
        let (w_tall, w_short) = (3usize, 2usize);
        let tall = row_major_bit_reversed(&make_columns(w_tall, 1 << h_tall, 77), 1 << h_tall);
        let short = row_major_bit_reversed(&make_columns(w_short, 1 << h_short, 88), 1 << h_short);

        let src = owned(vec![(tall, h_tall, w_tall), (short, h_short, w_short)]);
        let mmcs = Mmcs::commit(&src);
        let heights = [h_tall, h_short];
        let widths = [w_tall, w_short];
        assert_eq!(mmcs.h_max(), h_tall, "the round's h_max is below the FRI's");

        // The reduction the caller owes: iota_round = iota_fri >> (h_fri - h_round).
        let shift = h_max_fri - h_tall;
        // Pick a FRI index whose low bits differ from the reduced index's, so the
        // two conventions genuinely disagree here.
        let iota_fri = 0b10110usize;
        let iota_round = iota_fri >> shift;
        assert_ne!(
            iota_fri & ((1 << (h_tall - 1)) - 1),
            iota_round,
            "the test index must distinguish the low-bit and high-bit conventions"
        );

        // (1) Honest-path control at the reduced index.
        let opening = mmcs.open_batch(iota_round, &src);
        assert!(
            Mmcs::verify_batch(&mmcs.root(), iota_round, &opening, &heights, &widths),
            "the correctly reduced index must verify"
        );

        // (2) Tamper the SHORT (injected) matrix — the matrix a tall-only control
        // would never touch, and the one the disagreeing conventions move.
        let mut tampered = mmcs.open_batch(iota_round, &src);
        tampered.per_matrix[1].evaluations[0] =
            &tampered.per_matrix[1].evaluations[0] + &FE::from(1u64);
        assert!(
            !Mmcs::verify_batch(&mmcs.root(), iota_round, &tampered, &heights, &widths),
            "a tampered SHORT-matrix row must be rejected at the reduced index"
        );

        // (3) The misuse: hand the un-reduced FRI index in. It is out of this
        // tree's range, so the range guard rejects it rather than walking to some
        // unrelated leaf.
        assert!(
            iota_fri >= 1usize << (h_tall - 1),
            "the un-reduced index is outside this round's leaf range"
        );
        assert!(
            !Mmcs::verify_batch(&mmcs.root(), iota_fri, &opening, &heights, &widths),
            "an un-reduced FRI index must be rejected, not accepted at another leaf"
        );

        // And an in-range index that is simply the wrong leaf is rejected too, so
        // the guard is not the only thing standing between the two conventions.
        let wrong_but_in_range = iota_fri & ((1 << (h_tall - 1)) - 1);
        assert!(
            !Mmcs::verify_batch(
                &mmcs.root(),
                wrong_but_in_range,
                &opening,
                &heights,
                &widths
            ),
            "an opening replayed at the wrong in-range leaf must be rejected"
        );
    }

    /// The malformed-input surface of `verify_batch`: every shape error returns
    /// `false` rather than panicking, since a verifier calls this on proof data.
    #[test]
    fn verify_batch_rejects_malformed_shapes_without_panicking() {
        let h = 3usize;
        let w = 2usize;
        let data = row_major_bit_reversed(&make_columns(w, 1 << h, 4), 1 << h);
        let src = owned(vec![(data, h, w)]);
        let mmcs = Mmcs::commit(&src);
        let root = mmcs.root();
        let opening = mmcs.open_batch(1, &src);

        assert!(Mmcs::verify_batch(&root, 1, &opening, &[h], &[w]));
        // Mismatched metadata lengths.
        assert!(!Mmcs::verify_batch(&root, 1, &opening, &[h, h], &[w]));
        assert!(!Mmcs::verify_batch(&root, 1, &opening, &[h], &[w, w]));
        // Empty metadata.
        assert!(!Mmcs::verify_batch(&root, 1, &opening, &[], &[]));
        // A height that would overflow the level shift.
        assert!(!Mmcs::verify_batch(
            &root,
            1,
            &opening,
            &[usize::BITS as usize],
            &[w]
        ));
        // An index past this tree's leaf count.
        assert!(!Mmcs::verify_batch(
            &root,
            1usize << (h - 1),
            &opening,
            &[h],
            &[w]
        ));
        // A path of the wrong length.
        let mut short_path = opening.clone();
        short_path.proof.merkle_path.pop();
        assert!(!Mmcs::verify_batch(&root, 1, &short_path, &[h], &[w]));
    }

    /// The memory contract from the module's "what the caller may drop" section,
    /// made falsifiable: `commit` reads each height group's rows inside ONE
    /// contiguous window of the build, and the windows run in descending height
    /// order. A rewrite that materialized every matrix up front, or that revisited
    /// a group after moving on, would fail here.
    #[test]
    fn commit_reads_each_height_group_in_one_contiguous_phase() {
        /// Wraps a source and records, per matrix, the first and last global
        /// access sequence number. `Mutex` (not `Cell`) because `commit` reads the
        /// source from rayon workers.
        struct Tracing<'a, E: IsField> {
            inner: &'a OwnedMatrices<E>,
            clock: AtomicUsize,
            window: Mutex<Vec<(usize, usize)>>,
        }

        impl<E: IsField> LeafSource<E> for Tracing<'_, E> {
            fn num_matrices(&self) -> usize {
                self.inner.num_matrices()
            }
            fn log_height(&self, m: usize) -> usize {
                self.inner.log_height(m)
            }
            fn width(&self, m: usize) -> usize {
                self.inner.width(m)
            }
            fn append_row(&self, m: usize, bitrev_row: usize, out: &mut Vec<FieldElement<E>>) {
                let t = self.clock.fetch_add(1, Ordering::SeqCst);
                let mut w = self.window.lock().expect("no test thread panics here");
                w[m].0 = w[m].0.min(t);
                w[m].1 = w[m].1.max(t);
                drop(w);
                self.inner.append_row(m, bitrev_row, out);
            }
        }

        // Heights {5, 5, 3, 2}: two groups sharing the base layer, two injected.
        let specs = [(5usize, 2usize, 1u64), (5, 3, 2), (3, 1, 3), (2, 4, 4)];
        let inner = owned(
            specs
                .iter()
                .map(|&(lh, w, seed)| {
                    (
                        row_major_bit_reversed(&make_columns(w, 1 << lh, seed), 1 << lh),
                        lh,
                        w,
                    )
                })
                .collect(),
        );
        let tracing = Tracing {
            inner: &inner,
            clock: AtomicUsize::new(0),
            window: Mutex::new(vec![(usize::MAX, 0); specs.len()]),
        };

        let traced_root = Mmcs::commit(&tracing).root();
        assert_eq!(
            traced_root,
            Mmcs::commit(&inner).root(),
            "tracing must not change what is committed"
        );

        let windows = tracing
            .window
            .into_inner()
            .expect("uncontended after commit");
        for (m, (first, last)) in windows.iter().enumerate() {
            assert!(*first <= *last, "matrix {m} was never read");
        }

        // Same-height matrices share a window; different heights must not overlap,
        // and taller groups must come first.
        for (m, &(fm, lm)) in windows.iter().enumerate() {
            for (n, &(fn_, ln)) in windows.iter().enumerate() {
                if specs[m].0 <= specs[n].0 {
                    continue;
                }
                assert!(
                    lm < fn_ || ln < fm,
                    "matrices {m} (h={}) and {n} (h={}) were read in overlapping \
                     windows [{fm},{lm}] / [{fn_},{ln}] — a height group must be \
                     readable and then droppable",
                    specs[m].0,
                    specs[n].0
                );
                assert!(
                    lm < fn_,
                    "the taller matrix {m} (h={}) must be read before the shorter \
                     {n} (h={})",
                    specs[m].0,
                    specs[n].0
                );
            }
        }
    }
}

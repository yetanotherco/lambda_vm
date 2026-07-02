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

use crypto::merkle_tree::proof::Proof;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math::field::element::FieldElement;
use math::field::traits::IsField;
use math::traits::AsBytes;

use crate::config::{BatchedMerkleTreeBackend, Commitment};
use crate::proof::stark::PolynomialOpenings;

/// One committed matrix, kept so `open_batch` can serve its row pairs.
struct MatrixEntry<E: IsField> {
    /// Row-major, bit-reversed LDE evaluations. `data[r*width .. (r+1)*width]`
    /// is row `r`.
    data: Vec<FieldElement<E>>,
    log_height: usize,
    width: usize,
}

impl<E: IsField> MatrixEntry<E> {
    #[inline]
    fn row(&self, r: usize) -> &[FieldElement<E>] {
        &self.data[r * self.width..(r + 1) * self.width]
    }
}

/// A committed mixed-height, row-pair MMCS. Owns the matrices (to serve openings)
/// and every digest layer (to serve the shared authentication path).
pub struct MixedMmcs<E: IsField> {
    root: Commitment,
    /// `layers[0]` is the base digest layer; `layers[h_max-1] == [root]`.
    layers: Vec<Vec<Commitment>>,
    matrices: Vec<MatrixEntry<E>>,
    h_max: usize,
}

/// The opening of ALL matrices at one query index, authenticated by a single
/// shared Merkle path.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct MixedOpening<E: IsField> {
    /// The one authentication path covering every matrix's row at the query.
    pub proof: Proof<Commitment>,
    /// Per-matrix row pair (in the same INPUT order as `commit`). Each entry's
    /// own `proof` is empty — `MixedOpening::proof` is the authenticator.
    pub per_matrix: Vec<PolynomialOpenings<E>>,
}

/// Hash the row pair `(row(2*leaf), row(2*leaf+1))` of every matrix in `group`
/// (in the given order), all columns batched, into one digest.
fn hash_group_leaf<E>(group: &[&MatrixEntry<E>], leaf: usize) -> Commitment
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let mut buf: Vec<FieldElement<E>> = Vec::new();
    for m in group {
        buf.extend_from_slice(m.row(2 * leaf));
        buf.extend_from_slice(m.row(2 * leaf + 1));
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
    /// Commit the given matrices into one mixed-height row-pair tree. See the
    /// module docs for the exact leaf/injection layout.
    pub fn commit(matrices: &[(Vec<FieldElement<E>>, usize, usize)]) -> Self {
        assert!(
            !matrices.is_empty(),
            "MixedMmcs::commit requires at least one matrix"
        );

        let entries: Vec<MatrixEntry<E>> = matrices
            .iter()
            .map(|(data, log_height, width)| {
                assert!(
                    *log_height >= 1,
                    "log_height must be >= 1 (row-pair leaves need at least 2 rows)"
                );
                assert_eq!(
                    data.len(),
                    width << log_height,
                    "matrix data length must equal width * 2^log_height"
                );
                MatrixEntry {
                    data: data.clone(),
                    log_height: *log_height,
                    width: *width,
                }
            })
            .collect();

        let h_max = entries
            .iter()
            .map(|m| m.log_height)
            .max()
            .expect("entries is non-empty");
        let n0 = 1usize << (h_max - 1);

        // Base digest layer: batch all tallest matrices' row pairs (input order).
        let base_group: Vec<&MatrixEntry<E>> = entries
            .iter()
            .filter(|m| m.log_height == h_max)
            .collect();

        let mut layers: Vec<Vec<Commitment>> = Vec::with_capacity(h_max);
        let base: Vec<Commitment> = (0..n0).map(|k| hash_group_leaf(&base_group, k)).collect();
        layers.push(base);

        // Climb, compressing pairs and injecting shorter matrices where the layer
        // width matches their leaf count.
        let mut i = 0usize;
        while layers[i].len() > 1 {
            let next_len = layers[i].len() / 2;
            let inject_h = h_max - 1 - i;
            let inject_group: Vec<&MatrixEntry<E>> = entries
                .iter()
                .filter(|m| m.log_height == inject_h)
                .collect();

            let mut next: Vec<Commitment> = Vec::with_capacity(next_len);
            for j in 0..next_len {
                let mut parent = compress::<E>(&layers[i][2 * j], &layers[i][2 * j + 1]);
                if !inject_group.is_empty() {
                    let inj = hash_group_leaf(&inject_group, j);
                    parent = compress::<E>(&parent, &inj);
                }
                next.push(parent);
            }
            layers.push(next);
            i += 1;
        }

        let root = layers
            .last()
            .expect("at least the base layer exists")[0];

        MixedMmcs {
            root,
            layers,
            matrices: entries,
            h_max,
        }
    }

    /// The committed root.
    pub fn root(&self) -> Commitment {
        self.root
    }

    /// Open all matrices at query `iota in [0, 2^(h_max-1))`, returning each
    /// matrix's row pair plus one shared authentication path.
    pub fn open_batch(&self, iota: usize) -> MixedOpening<E> {
        let n0 = 1usize << (self.h_max - 1);
        assert!(iota < n0, "iota {iota} out of range (n0 = {n0})");

        let per_matrix: Vec<PolynomialOpenings<E>> = self
            .matrices
            .iter()
            .map(|m| {
                let k = iota >> (self.h_max - m.log_height);
                PolynomialOpenings {
                    proof: Proof {
                        merkle_path: Vec::new(),
                    },
                    evaluations: m.row(2 * k).to_vec(),
                    evaluations_sym: m.row(2 * k + 1).to_vec(),
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
    use math::fft::bit_reversing::reverse_index;
    use math::field::element::FieldElement;
    use math::field::goldilocks::GoldilocksField;

    type FE = FieldElement<GoldilocksField>;

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
                chunk[c] = col[br].clone();
            }
        }
        out
    }

    fn make_columns(width: usize, num_rows: usize, seed: u64) -> Vec<Vec<FE>> {
        (0..width)
            .map(|c| {
                (0..num_rows)
                    .map(|r| FE::from(seed.wrapping_mul(31) + (c as u64) * 1009 + (r as u64) * 7 + 1))
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

        let mmcs = MixedMmcs::commit(&[(data.clone(), log_height, width)]);
        let heights = [log_height];
        let widths = [width];
        let n0 = 1usize << (log_height - 1);

        for iota in 0..n0 {
            let opening = mmcs.open_batch(iota);
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

        let mut opening = mmcs.open_batch(0);
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
        let mmcs = MixedMmcs::commit(&[(data, log_height, width)]);

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

        let mmcs = MixedMmcs::commit(&[
            (a.clone(), ha, wa),
            (b.clone(), hb, wb),
            (c.clone(), hc, wc),
        ]);
        let heights = [ha, hb, hc];
        let widths = [wa, wb, wc];
        let h_max = 5usize;
        let n0 = 1usize << (h_max - 1); // 16

        let row = |data: &[FE], w: usize, r: usize| data[r * w..(r + 1) * w].to_vec();

        for iota in [0usize, 1, 2, 3, 7, 8, 13, n0 - 1] {
            let opening = mmcs.open_batch(iota);
            assert_eq!(opening.per_matrix.len(), 3);

            // Tall matrices open at k = iota >> 0 = iota.
            assert_eq!(opening.per_matrix[0].evaluations, row(&a, wa, 2 * iota));
            assert_eq!(opening.per_matrix[0].evaluations_sym, row(&a, wa, 2 * iota + 1));
            assert_eq!(opening.per_matrix[1].evaluations, row(&b, wb, 2 * iota));

            // Height-3 matrix opens at k = iota >> (5 - 3) = iota >> 2.
            let kc = iota >> (h_max - hc);
            assert_eq!(opening.per_matrix[2].evaluations, row(&c, wc, 2 * kc));
            assert_eq!(opening.per_matrix[2].evaluations_sym, row(&c, wc, 2 * kc + 1));

            assert!(
                MixedMmcs::verify_batch(&mmcs.root(), iota, &opening, &heights, &widths),
                "honest opening at iota={iota} must verify"
            );
        }

        // Tamper the height-3 matrix's opened row -> rejection (proves the short
        // matrix is bound by the shared path via injection).
        let iota = 6usize;
        let mut opening = mmcs.open_batch(iota);
        opening.per_matrix[2].evaluations[0] =
            &opening.per_matrix[2].evaluations[0] + &FE::from(1u64);
        assert!(
            !MixedMmcs::verify_batch(&mmcs.root(), iota, &opening, &heights, &widths),
            "tampered height-3 row must be rejected"
        );

        // Tamper a tall-matrix row too -> rejection.
        let mut opening2 = mmcs.open_batch(iota);
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

        let mmcs = MixedMmcs::commit(&[(a_data.clone(), 2, 2), (b_data.clone(), 1, 3)]);

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
        let mmcs2 = MixedMmcs::commit(&[(a_data, 2, 2), (b_data, 1, 3)]);
        assert_eq!(mmcs.root(), mmcs2.root(), "commit must be deterministic");

        for iota in 0..2usize {
            let opening = mmcs.open_batch(iota);
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

        let mmcs = MixedMmcs::commit(&[(a, h, wa), (b, h, wb)]);
        let heights = [h, h];
        let widths = [wa, wb];

        let iota = 0usize;
        let opening = mmcs.open_batch(iota);
        assert!(
            MixedMmcs::verify_batch(&mmcs.root(), iota, &opening, &heights, &widths),
            "honest opening must verify"
        );

        // Forge: lengthen A.evaluations by one element taken from A.evaluations_sym.
        let mut forged = mmcs.open_batch(iota);
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
                chunk[c] = col[br].clone();
            }
        }

        let mmcs = MixedMmcs::commit(&[(data, log_height, width)]);
        assert_eq!(
            mmcs.root(),
            existing_root,
            "Fp3 single-matrix root must match the existing row-pair tree"
        );

        let heights = [log_height];
        let widths = [width];
        for iota in 0..(1usize << (log_height - 1)) {
            let opening = mmcs.open_batch(iota);
            assert!(MixedMmcs::verify_batch(
                &mmcs.root(),
                iota,
                &opening,
                &heights,
                &widths
            ));
        }

        let mut opening = mmcs.open_batch(0);
        opening.per_matrix[0].evaluations[0] =
            &opening.per_matrix[0].evaluations[0] + &F3::one();
        assert!(!MixedMmcs::verify_batch(
            &mmcs.root(),
            0,
            &opening,
            &heights,
            &widths
        ));
    }
}

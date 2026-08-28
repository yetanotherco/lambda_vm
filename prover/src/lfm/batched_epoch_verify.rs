//! The batched epoch's verification legs — the mixed-height MMCS walk.
//!
//! The batched counterpart of [`super::sub_proof`]'s authentication half.
//! The order authority is `stark::fri::mmcs::MixedMmcs::verify_batch`
//! (fri/mmcs.rs' "Tree layout" section is the contract): ONE path
//! authenticates every matrix of a round — the tallest matrices batch into
//! the base leaf, and each shorter height group is INJECTED where the climb
//! reaches its layer, as one extra compression.
//!
//! Heights and widths are program shape, so the injection schedule UNROLLS at
//! emit time: the emitted walk is straight-line — per level one
//! compress-with-sibling (two `Select`s on the shared index bit) and, iff
//! some matrix sits at that level's injection height, one further compress
//! with that height group's leaf hash. No branch, no `Select` beyond the
//! sibling ordering, exactly as [`super::edsl::WrapHash::merkle_walk`]'s doc
//! anticipated ("a batched path that injects at mixed heights extends this
//! rather than replacing it").
//!
//! ## The index convention, in cells
//!
//! The machine's shared query index is a BIT VECTOR (low-to-high,
//! `h_max_fri − 1` bits, drawn once by the spine). `fri/mmcs.rs`' index
//! reduction — `iota_round = iota_fri >> (h_max_fri − h_max_round)` — is
//! [`reduce_iota_bits`]: DROP THE LOW BITS, keep the high ones. In LFM the
//! reduction is free (slicing a cell vector emits nothing), but the DIRECTION
//! is still the soundness-relevant choice: host-side a wrong shift is
//! self-consistent between prover and verifier and fails silently, which is
//! why `the_batched_openings_authenticate_against_the_spine_roots` ports the
//! `short_round_low_bit_convention_is_exercised` control to the machine.

use super::builder::{Bit, Cell, Felt, LfmBuilder};
use super::edsl::{self, WrapDigest};
use super::epoch::RootCells;
use super::sub_proof::GroupShape;

/// One matrix of a mixed round, as the walk consumes it — its shape (columns
/// and element kind), its height (the injection schedule's key), and its
/// opened row pair as the caller's CELLS. There is deliberately no
/// constructor that hints: the values are whatever the caller already holds,
/// which is what makes the authentication and the folds share them.
pub struct MixedMatrixOpening<'a> {
    pub shape: GroupShape,
    /// `log2` of the matrix's LDE height — where in the climb it enters.
    pub log_height: usize,
    /// `evaluations ‖ evaluations_sym` in leaf order — `2 · num_columns`
    /// cells.
    pub values: &'a [Cell],
}

/// The leaf hash of one HEIGHT GROUP's row pairs — `hash_group_openings`'
/// layout: every matrix's `evaluations ‖ evaluations_sym`, in round INPUT
/// order, flat, one hash. Each element renders exactly as the per-table leaf
/// does ([`super::sub_proof::emit_leaf_hash`]): a base element as its eight
/// big-endian bytes, an extension element as its three components, each eight
/// big-endian bytes. Lane 3 of an extension cell is NOT hashed — production
/// hashes three components — and the same caveat applies as there: every ext
/// value a query opens is also an ext operand of the DEEP crossing, which is
/// what pins lane 3 to zero.
pub fn emit_group_leaf_hash(b: &mut LfmBuilder, group: &[&MixedMatrixOpening<'_>]) -> WrapDigest {
    use super::keccak_host::BYTES_PER_HALF;
    use super::transcript_replay::felt_be_halves;

    assert!(!group.is_empty(), "a group leaf covers at least one matrix");

    // ★ The ALGEBRAIC path absorbs the felts — same reasoning as
    // `sub_proof::emit_leaf_hash`: the byte stream below is a serialisation of
    // field elements that exists only for a byte-oriented hash.
    let Some(byte_hash) = b.wrap_hash().byte_hash() else {
        let felts = group_leaf_felts(b, group);
        return edsl::wrap_leaf_hash(b, &felts);
    };

    let mut stream: Vec<Felt> = Vec::new();
    for m in group {
        assert_eq!(
            m.values.len(),
            m.shape.num_values(),
            "a matrix's opening covers its whole row pair"
        );
        for v in m.values {
            if m.shape.is_ext {
                let lanes = b.unpack(*v);
                for lane in lanes.iter().take(3) {
                    stream.extend(felt_be_halves(b, *lane));
                }
            } else {
                stream.extend(felt_be_halves(b, Felt(v.addr())));
            }
        }
    }
    let len_bytes = BYTES_PER_HALF * stream.len();
    edsl::wrap_hash_bytes(b, byte_hash, &stream, len_bytes)
}

/// ★ The FELT SEQUENCE the algebraic arm of [`emit_group_leaf_hash`] absorbs,
/// in absorption order — the machine's counterpart of
/// `stark::fri::mmcs::group_opening_felts`.
///
/// Split out of its only production caller so a differential can compare the
/// SEQUENCE the machine feeds against the sequence the host feeds, rather than
/// only the digests they disagree on. The disagreement this path fails with is
/// a `DivByZero` deep in a query walk, which names neither the site nor the
/// felt; a sequence differential names the index.
///
/// A base value is ONE felt (the cell's own lane 0); an extension value is its
/// three components, lanes 0, 1 and 2 of the unpacked word — lane 3 is not
/// absorbed, for the reason [`emit_group_leaf_hash`] states.
pub fn group_leaf_felts(b: &mut LfmBuilder, group: &[&MixedMatrixOpening<'_>]) -> Vec<Felt> {
    let mut felts: Vec<Felt> = Vec::new();
    for m in group {
        assert_eq!(
            m.values.len(),
            m.shape.num_values(),
            "a matrix's opening covers its whole row pair"
        );
        for v in m.values {
            if m.shape.is_ext {
                let lanes = b.unpack(*v);
                felts.extend_from_slice(&lanes[..3]);
            } else {
                felts.push(Felt(v.addr()));
            }
        }
    }
    felts
}

/// Authenticate one mixed round's openings against its committed root — the
/// injecting walk, `MixedMmcs::verify_batch` emitted.
///
/// `matrices` in round INPUT order; `siblings` leaf level first,
/// `h_max − 1` of them; `bits` the REDUCED shared index, low-to-high,
/// `h_max − 1` of them ([`reduce_iota_bits`]). The final assert against the
/// root's lanes is the binding: the root cells are the SAME cells the spine
/// absorbed, so there is no second copy for a prover to disagree with.
pub fn emit_mixed_verify_batch(
    b: &mut LfmBuilder,
    root: &RootCells,
    matrices: &[MixedMatrixOpening<'_>],
    siblings: &[WrapDigest],
    bits: &[Bit],
) {
    let h_max = matrices
        .iter()
        .map(|m| m.log_height)
        .max()
        .expect("a round has at least one matrix");
    assert!(h_max >= 1, "a row-pair tree needs at least two rows");
    assert_eq!(siblings.len(), h_max - 1, "one sibling per level");
    assert_eq!(
        bits.len(),
        h_max - 1,
        "the reduced index has h_max − 1 bits"
    );
    for m in matrices {
        assert!(
            (1..=h_max).contains(&m.log_height),
            "a matrix's height sits inside its round's climb"
        );
    }

    // Base node: every tallest matrix's row pair, one leaf hash.
    let base: Vec<&MixedMatrixOpening<'_>> =
        matrices.iter().filter(|m| m.log_height == h_max).collect();
    let mut acc = emit_group_leaf_hash(b, &base);

    for (level, (bit, sibling)) in bits.iter().zip(siblings).enumerate() {
        // Both halves of the digest must swap on the SAME bit; bit = 0 means
        // the current node is the LEFT child, as in every walk here.
        // ★ Every cell swaps on the SAME bit — a loop, so a one-cell algebraic
        // digest costs ONE select per level where a byte digest costs two.
        debug_assert_eq!(
            acc.len(),
            sibling.len(),
            "node and sibling widths must match"
        );
        let n = acc.len();
        let mut left = [acc[0]; edsl::MAX_DIGEST_CELLS];
        let mut right = [acc[0]; edsl::MAX_DIGEST_CELLS];
        for k in 0..n {
            let (l, r) = b.select(*bit, acc[k], sibling[k]);
            left[k] = l;
            right[k] = r;
        }
        let mut parent = edsl::wrap_hash_pair(
            b,
            edsl::WrapDigest::from_cells(&left[..n]),
            edsl::WrapDigest::from_cells(&right[..n]),
        );

        // The injection, unrolled: heights are shape, so whether a group
        // enters here is decided now, not by an emitted branch.
        let inject_h = h_max - 1 - level;
        let group: Vec<&MixedMatrixOpening<'_>> = matrices
            .iter()
            .filter(|m| m.log_height == inject_h)
            .collect();
        if !group.is_empty() {
            let inj = emit_group_leaf_hash(b, &group);
            parent = edsl::wrap_hash_pair(b, parent, inj);
        }
        acc = parent;
    }

    edsl::assert_digest_eq_lanes(b, acc, &root.lanes);
}

/// Reduce the SHARED query-index bits to a round (or per-table tree) whose
/// own tallest height is `h_max_round` — `reduce_iota_to_round`'s
/// `iota >> (h_max_fri − h_max_round)`, on a low-to-high bit vector: drop
/// the LOW `h_max_fri − h_max_round` bits, keep the high `h_max_round − 1`.
///
/// Free — slicing emits nothing — but direction-critical; see the module doc.
pub fn reduce_iota_bits(bits: &[Bit], h_max_fri: usize, h_max_round: usize) -> &[Bit] {
    assert!(
        h_max_round <= h_max_fri,
        "no round is taller than the FRI's domain"
    );
    assert_eq!(
        bits.len(),
        h_max_fri - 1,
        "the shared index has h_max_fri − 1 bits"
    );
    &bits[(h_max_fri - h_max_round)..]
}

// ================= the DEEP mix and the batched FRI leg =================

/// α-mix one query's per-table DEEP pairs into the tallest-domain pair `p0`
/// and the per-height injection buckets — `verify_epoch_fri`'s loop, emitted.
///
/// ★ Powers of α go by `plan_batched` POSITION, not table index and not
/// position within a height group — the three orders coincide on a same-height
/// epoch and diverge on a real one (`batched/verifier.rs`' warning). A short
/// table contributes ONE value, chosen from its pair by the injection
/// position's low bit — which is bit `h_max − h − 1` of the SHARED index, so
/// the choice is a `Select` on a bit the transcript drew, never a hint.
///
/// `deep_pairs` is indexed by TABLE; entries outside the batched class are
/// not read.
pub fn emit_query_mix(
    b: &mut LfmBuilder,
    plan_batched: &[usize],
    heights: &[usize],
    h_max: usize,
    alpha: super::builder::Ext,
    deep_pairs: &[(super::builder::Ext, super::builder::Ext)],
    bits: &[Bit],
) -> (
    super::builder::Ext,
    super::builder::Ext,
    Vec<Option<super::builder::Ext>>,
) {
    assert!(!plan_batched.is_empty(), "the batched class is never empty");
    assert_eq!(bits.len(), h_max - 1, "the shared index has h_max − 1 bits");

    let mut p0: Option<(super::builder::Ext, super::builder::Ext)> = None;
    let mut buckets: Vec<Option<super::builder::Ext>> = vec![None; h_max];
    let mut power: Option<super::builder::Ext> = None;
    for &table in plan_batched {
        let (d, d_sym) = deep_pairs[table];
        let h = heights[table];
        assert!(h <= h_max, "no batched table is taller than the instance");
        // α^pos — position in plan.batched. pos 0 multiplies by nothing.
        let scale = |b: &mut LfmBuilder, v: super::builder::Ext| match power {
            None => v,
            Some(p) => b.emul(p, v),
        };
        if h == h_max {
            let sd = scale(b, d);
            let sds = scale(b, d_sym);
            p0 = Some(match p0 {
                None => (sd, sds),
                Some((a, s)) => (b.eadd(a, sd), b.eadd(s, sds)),
            });
        } else {
            // `injected_value_at_query`: the injection position's low bit is
            // bit `h_max − h − 1` of the shared index; 0 picks the regular
            // value, 1 the symmetric one — `select` at 0 returns its first
            // argument first, so `.0` IS that conditional.
            let (chosen, _) = b.select(bits[h_max - h - 1], d.as_cell(), d_sym.as_cell());
            let sv = scale(b, chosen.as_ext());
            buckets[h] = Some(match buckets[h].take() {
                None => sv,
                Some(acc) => b.eadd(acc, sv),
            });
        }
        power = Some(match power {
            None => alpha,
            Some(p) => b.emul(p, alpha),
        });
    }
    let (p0, p0_sym) = p0.expect("the tallest table is always batched");
    (p0, p0_sym, buckets)
}

/// One query of the BATCHED FRI instance: the fold-with-injection recursion,
/// every committed layer's opening authenticated at the shared index, and the
/// terminal check — `verify_batched_fri_query`, emitted.
///
/// The per-table [`super::fri::emit_query_fri`]'s shape with two additions:
/// after EVERY fold (the uncommitted first one included) the height the
/// running codeword just reached may have a bucket, injected as
/// `v += ζ² · bucket` — the schedule is program shape and UNROLLS — and the
/// terminal Horner runs at `υ^(2^total_folds)` of the TALLEST domain, whose
/// coset offset the caller already folded into `point`.
#[allow(clippy::too_many_arguments)]
pub fn emit_batched_query_fri(
    b: &mut LfmBuilder,
    layout: &stark::fri::batched::BatchedFriLayout,
    h_max: usize,
    layers: &[super::fri::LayerCommitment],
    zetas: &[super::builder::Ext],
    coeffs: &[super::builder::Ext],
    bits: &[Bit],
    point: Felt,
    point_sym: Felt,
    p0: super::builder::Ext,
    p0_sym: super::builder::Ext,
    buckets: &[Option<super::builder::Ext>],
    openings: &[super::fri::LayerOpening],
) -> super::builder::Ext {
    use super::edsl::horner_ext;
    use super::fri::FRI_LEAF_GROUP;
    use crate::tables::types::FE;

    let c = layout.num_committed;
    assert_eq!(bits.len(), h_max - 1, "the shared index has h_max − 1 bits");
    assert_eq!(layers.len(), c, "one commitment per committed layer");
    assert_eq!(openings.len(), c, "one opening per committed layer");
    assert_eq!(
        coeffs.len(),
        1usize << layout.effective_k,
        "the terminal polynomial carries 2^effective_k coefficients"
    );
    assert_eq!(buckets.len(), h_max, "one bucket slot per height");

    if layout.total_folds == 0 {
        // The codeword never folds: the terminal IS the tallest codeword and
        // no bucket can exist (`h_min == h_max` is what makes folds zero).
        assert!(zetas.is_empty(), "a codeword that never folds draws no ζ");
        assert!(
            buckets.iter().all(Option::is_none),
            "no injection exists below a terminal-height instance"
        );
        let at = horner_ext(b, point.as_ext(), coeffs);
        b.assert_eq_ext(at, p0);
        let at_sym = horner_ext(b, point_sym.as_ext(), coeffs);
        b.assert_eq_ext(at_sym, p0_sym);
        return p0;
    }
    assert_eq!(zetas.len(), c + 1, "folds exceed committed layers by one");

    let inject = |b: &mut LfmBuilder,
                  v: super::builder::Ext,
                  zeta: super::builder::Ext,
                  height: usize|
     -> super::builder::Ext {
        match buckets.get(height).and_then(|o| o.as_ref()) {
            None => v,
            Some(bucket) => {
                let zeta_sq = b.emul(zeta, zeta);
                let term = b.emul(zeta_sq, *bucket);
                b.eadd(v, term)
            }
        }
    };

    let one = b.felt_const(FE::one());
    let inv = b.div(one, point);

    // Fold 0 consumes the mixed DEEP pair and authenticates nothing; the
    // height just below joins before the first committed layer, exactly as
    // `batched_commit_phase` injects before it commits.
    let mut v = super::edsl::fri_fold(b, p0, p0_sym, zetas[0], inv);
    v = inject(b, v, zetas[0], h_max - 1);

    let mut inv_pow = inv;
    for (i, opening) in openings.iter().enumerate() {
        let (first, second) = b.select(bits[i], v.as_cell(), opening.sym.as_cell());
        let leaf = super::sub_proof::emit_leaf_hash(b, FRI_LEAF_GROUP, &[first, second]);
        let root = super::edsl::wrap_merkle_walk(b, leaf, &bits[i + 1..], &opening.siblings);
        super::edsl::assert_digest_eq_lanes(b, root, &layers[i].root_lanes);

        inv_pow = b.mul(inv_pow, inv_pow);
        v = super::edsl::fri_fold(b, v, opening.sym, zetas[i + 1], inv_pow);
        if let Some(height) = (h_max - 1).checked_sub(i + 1) {
            v = inject(b, v, zetas[i + 1], height);
        }
    }

    // `υ^(2^total_folds)` — the terminal codeword's own point at the reduced
    // position, coset offset included by construction (the point already
    // carries it, so raising it raises the offset too:
    // `terminal_offset = coset_offset^(2^total_folds)`).
    let mut x = point;
    for _ in 0..layout.total_folds {
        x = b.mul(x, x);
    }
    let at = horner_ext(b, x.as_ext(), coeffs);
    b.assert_eq_ext(at, v);
    v
}

/// One query of a STANDALONE table's terminal-only instance: the sent
/// polynomial (the ARENA CELLS the spine absorbed — one cell, two consumers)
/// evaluated at the table's own reduced pair must equal its DEEP pair —
/// `verify_standalone_fri_query`, emitted. Nothing folds and nothing walks.
pub fn emit_standalone_terminal_check(
    b: &mut LfmBuilder,
    coeffs: &[super::builder::Ext],
    point: Felt,
    point_sym: Felt,
    deep: super::builder::Ext,
    deep_sym: super::builder::Ext,
) {
    let at = super::edsl::horner_ext(b, point.as_ext(), coeffs);
    b.assert_eq_ext(at, deep);
    let at_sym = super::edsl::horner_ext(b, point_sym.as_ext(), coeffs);
    b.assert_eq_ext(at_sym, deep_sym);
}

// ======================= the batched query census =======================

/// Wrap-hash permutations ONE query of the batched epoch costs, from shape
/// alone — the batched counterpart of
/// [`super::epoch_verify::query_permutations_for`], and the campaign's
/// wrap-side economy as one formula: authentication paths per ROUND (plus
/// each preprocessed table's own small tree), never per table per group.
///
/// Per query: each preprocessed table's leaf and path; per mixed round the
/// FUSED base leaf (every tallest matrix in one absorption), the ONE shared
/// path, and per injected height group one leaf plus ONE extra compression;
/// then the batched FRI instance's layer leaves and path steps. Standalone
/// tables cost NO hashing at all — their check is polynomial evaluation.
///
/// A closed form over the shapes (the layout and partition are production's
/// own), so comparing it against the emitted count is an absolute check.
pub fn batched_query_permutations_for(
    shape: &stark::batched::shape::EpochShape,
    params: &stark::batched::shape::EpochFriParams,
    hash: super::edsl::WrapHash,
) -> usize {
    use super::epoch_verify::{FRI_LEAF_FELTS, blocks_for};
    use stark::fri::batched::{BatchedFriLayout, FriInstancePlan};

    let mut per_query = 0usize;

    for &(h, w) in &shape.prep.dims {
        per_query += blocks_for(2 * w, hash);
        per_query += h - 1;
    }

    // The carved table's standalone main tree: exactly a preprocessed table's
    // cost shape — one row-pair leaf and its own path at the carved height —
    // with the root proof-carried instead of AIR-owned.
    if let Some(c) = &shape.carved_main {
        per_query += blocks_for(2 * c.width, hash);
        per_query += shape.heights[c.table] - 1;
    }

    for (round, ext) in [
        (&shape.main, false),
        (&shape.aux, true),
        (&shape.parts, true),
    ] {
        let Some(h_max) = round.h_max() else { continue };
        let per_value = if ext { 3 } else { 1 };
        let group_felts = |height: usize| -> usize {
            round
                .dims
                .iter()
                .filter(|&&(h, _)| h == height)
                .map(|&(_, w)| 2 * w * per_value)
                .sum()
        };
        per_query += blocks_for(group_felts(h_max), hash);
        per_query += h_max - 1;
        for h in 1..h_max {
            let felts = group_felts(h);
            if felts > 0 {
                per_query += blocks_for(felts, hash) + 1;
            }
        }
    }

    let plan = FriInstancePlan::new(
        &shape.heights,
        params.blowup_log,
        params.final_poly_log_degree,
    )
    .expect("a real epoch's heights partition");
    let layout = BatchedFriLayout::new(
        plan.h_max,
        plan.h_min,
        params.blowup_log,
        params.final_poly_log_degree,
    );
    per_query += layout.num_committed * blocks_for(FRI_LEAF_FELTS, hash);
    per_query += (0..layout.num_committed)
        .map(|i| plan.h_max - i - 2)
        .sum::<usize>();

    per_query
}

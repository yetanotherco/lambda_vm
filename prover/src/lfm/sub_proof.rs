//! One sub-proof's query verification: DEEP reconstruction over the SAME arena
//! cells the Merkle authentication authenticates.
//!
//! The [constraint](super::constraints) and [DEEP](super::deep) legs consume
//! opened values; the [Merkle walk](super::edsl::keccak_merkle_walk)
//! authenticates them. Built separately the two are each correct and neither
//! says anything about the other — a program could fold one set of values and
//! authenticate a different set, and every test that fed both halves the same
//! data would pass. This module is the join, and it is a join by CONSTRUCTION
//! rather than by convention: [`emit_group_authentication`] takes cells and
//! cannot hint, so the only values it can authenticate are the caller's, and
//! [`emit_query`] hands those same cells to the DEEP fold.
//!
//! # The two consumers disagree about layout, which is the whole difficulty
//!
//! A query opens four committed matrices — precomputed, main, aux, composition
//! — and each is a SEPARATE Merkle tree with its own root and its own path. The
//! leaf of one tree is that matrix's own row pair:
//!
//! ```text
//!   leaf(main) = keccak( main[υ] ‖ main[−υ] )
//! ```
//!
//! while DEEP walks one POINT across all matrices:
//!
//! ```text
//!   DEEP(υ) folds  precomputed[υ] ‖ main[υ] ‖ aux[υ]
//! ```
//!
//! So the authentication groups by matrix and the fold groups by point. The
//! two orders cross, which is exactly the situation that invites two parallel
//! copies of the same values in two arenas — sound only as long as the host
//! filling them agrees with itself, which no in-machine constraint requires.
//! Here `values` is one vector of cells per matrix, in LEAF order, and DEEP
//! indexes into it: column `c` at the regular point is `values[c]`, at the
//! symmetric point `values[num_columns + c]`.
//!
//! # The query point is derived from the index bits, not hinted
//!
//! `DEEP(υ)` is meaningless unless `υ` is the point the authenticated leaf
//! sits at. Production derives both from one challenge `iota`
//! (`query_challenge_to_evaluation_point`); a machine that hinted the point
//! separately would let a prover authenticate a leaf at one index and evaluate
//! DEEP at another. [`emit_query`] decomposes the hinted index ONCE and uses
//! the same bits for the walk and for the point, so the two cannot disagree.
//!
//! `υ = offset · g^{br(2·iota)}` where `br` is the bit reversal over the LDE
//! domain. Reversing `2·iota` maps index bit `i` to weight `2^{depth-1-i}`, so
//! the point is `offset · Π (g^{2^{depth-1-i}})^{b_i}` — one `Select` and one
//! `Mul` per bit against program constants, via [`super::edsl::pow_bits`]. The
//! symmetric point is `−υ`: `br(2·iota+1) = br(2·iota) + L/2` and `g^{L/2} =
//! −1`, so it costs one subtraction rather than a second derivation.

use math::field::traits::IsFFTField;

use crate::tables::types::{FE, GoldilocksField};

use super::builder::{Bit, Cell, Ext, Felt, LfmBuilder};
use super::deep::{DeepInvariants, DeepOpening, DeepShape, emit_deep_point};
use super::edsl::{self, KeccakDigest};

/// Rows a Merkle leaf covers — `crypto/stark`'s `ROWS_PER_LEAF`, mirrored here
/// because it fixes program shape: a leaf holds a row PAIR, which is why one
/// path authenticates both of a query's two points.
pub const ROWS_PER_LEAF: usize = 2;

/// The compile-time shape of one committed matrix of a sub-proof.
///
/// `is_ext` is the element kind, and it is not cosmetic: a base element is
/// rendered into the leaf as 8 big-endian bytes and an extension element as 24
/// (components 0, 1, 2, each big-endian — `write_bytes_be` for
/// `FieldElement<Degree3GoldilocksExtensionField>`). Getting it wrong changes
/// the byte string and therefore the leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupShape {
    /// Columns at ONE point. A leaf covers `ROWS_PER_LEAF · num_columns`.
    pub num_columns: usize,
    pub is_ext: bool,
}

impl GroupShape {
    /// Cells one query's opening of this group occupies — both points.
    pub fn num_values(&self) -> usize {
        ROWS_PER_LEAF * self.num_columns
    }

    /// Bytes the leaf hash covers.
    pub fn leaf_bytes(&self) -> usize {
        self.num_values() * if self.is_ext { 24 } else { 8 }
    }
}

/// One sub-proof's per-query verification shape.
///
/// Every field is program SHAPE. In particular the group list is: a proof that
/// carried an aux opening where the program expects none would not match the
/// arena schema, which is the straight-line discipline standing in for
/// production's `(Some(root), Some(opening)) | (None, None)` presence check.
#[derive(Clone, Debug)]
pub struct SubProofShape {
    /// The DEEP fold's shape — column count, OOD grid, part count.
    pub deep: DeepShape,
    /// The TRACE matrices in DEEP column order: precomputed, then main, then
    /// aux. Absent groups are omitted, exactly as the proof omits them. Their
    /// widths must sum to `deep.num_total_cols`.
    pub trace_groups: Vec<GroupShape>,
    /// Merkle depth — `log2(lde_length) − 1`, since a leaf is a row pair. All
    /// four trees commit over the same LDE domain, so one depth serves them
    /// all and one index addresses them all.
    pub merkle_depth: usize,
    /// `log2` of the LDE domain — `log2_trace_length + log2(blowup)`.
    pub log2_lde_length: u32,
    /// The LDE coset offset, `ProofOptions::coset_offset`.
    pub coset_offset: FE,
}

impl SubProofShape {
    /// The composition-parts group. Its width is the part count and its
    /// elements are extension, both of which are already DEEP shape.
    pub fn parts_group(&self) -> GroupShape {
        GroupShape {
            num_columns: self.deep.num_composition_parts,
            is_ext: true,
        }
    }

    /// Every group a query authenticates: the trace matrices then the parts.
    pub fn groups(&self) -> Vec<GroupShape> {
        let mut all = self.trace_groups.clone();
        all.push(self.parts_group());
        all
    }

    /// Arena words one query's openings occupy — every group's values, plus
    /// the index and the sibling digests (two words per level per group).
    pub fn query_words(&self) -> usize {
        1 + self.opening_words()
    }

    /// [`Self::query_words`] WITHOUT the index word.
    ///
    /// The assembled verifier's stride: its query index is not proof data at
    /// all but the transcript's own bits, so the arena carries only the opened
    /// values and the paths. An arena that still carried an index would be
    /// offering the prover a second one.
    pub fn opening_words(&self) -> usize {
        let values: usize = self.groups().iter().map(GroupShape::num_values).sum();
        let siblings = 2 * self.merkle_depth * self.groups().len();
        values + siblings
    }

    /// Checked invariants of a shape, so a caller cannot assemble one whose
    /// groups do not cover the fold.
    fn check(&self) {
        let width: usize = self.trace_groups.iter().map(|g| g.num_columns).sum();
        assert_eq!(
            width, self.deep.num_total_cols,
            "the trace groups must cover exactly the DEEP column set"
        );
        assert!(
            self.merkle_depth + 1 == self.log2_lde_length as usize,
            "a leaf is a row pair, so the tree is one level shallower than the \
             LDE domain: depth {} against log2(lde) {}",
            self.merkle_depth,
            self.log2_lde_length
        );
        assert!(
            self.merkle_depth >= 1,
            "a tree with no levels has no path to walk"
        );
    }
}

/// A committed matrix's root, unpacked once and shared by every query.
///
/// Hoisting the unpack is what `fri_toy_program` already does per query: the
/// root is a per-sub-proof value and a 219-query proof would otherwise pay
/// 219 redundant `Unpack`s per group.
pub struct GroupCommitment {
    /// The root's two words as lanes.
    pub root_lanes: [[Felt; 4]; 2],
    pub shape: GroupShape,
}

impl GroupCommitment {
    /// Reads a root out of the arena and hoists its unpack.
    pub fn hint(
        b: &mut LfmBuilder,
        arena: super::instr::ArenaId,
        base: u32,
        shape: GroupShape,
    ) -> Self {
        let w0 = b.hint_word(arena, base);
        let w1 = b.hint_word(arena, base + 1);
        GroupCommitment {
            root_lanes: [b.unpack(w0), b.unpack(w1)],
            shape,
        }
    }

    /// A commitment over lanes the caller already holds — the assembled
    /// verifier's route, where a root reaches this leg as the SAME cells the
    /// transcript absorbed rather than as a second hint.
    ///
    /// A root has two consumers (`epoch::RootCells`' doc comment names them):
    /// the Fiat-Shamir absorb and this comparison. Hinting it twice is the
    /// two-consumer hazard — a prover would absorb one root and authenticate
    /// against another, and no differential over honest data could see it,
    /// because the host packs the same bytes into both. This constructor is the
    /// join, and it takes lanes rather than words precisely so there is nothing
    /// left to hint.
    pub fn from_lanes(root_lanes: [[Felt; 4]; 2], shape: GroupShape) -> Self {
        GroupCommitment { root_lanes, shape }
    }
}

/// One query's opening of one committed matrix, as CELLS.
///
/// There is deliberately no constructor that hints: the values are whatever the
/// caller already holds, which is what makes the authentication and the fold
/// share them rather than agree about them.
pub struct GroupOpening {
    /// `evaluations ‖ evaluations_sym` in LEAF order — the row pair written
    /// column by column, the regular point first.
    pub values: Vec<Cell>,
    /// Sibling digests, LEAF LEVEL FIRST — the order
    /// `verify_merkle_path_from_leaf_hash` consumes them in.
    pub siblings: Vec<KeccakDigest>,
}

/// The leaf hash of one group's row pair, in the production commitment layout.
///
/// Base groups go through [`edsl::keccak_leaf_hash`] unchanged. Extension
/// groups render each element as its three components, each big-endian —
/// `write_bytes_be` writes components 0, 1, 2 in that order, so the machine
/// unpacks the word and byteswaps lanes 0, 1, 2.
///
/// Lane 3 is NOT hashed, which is correct (production hashes three components)
/// and worth stating: an extension cell whose lane 3 is nonzero would hash the
/// same as one whose lane 3 is zero. It cannot arise here because every
/// extension value a query opens is also consumed as an ext operand by the DEEP
/// fold, and an ext read of a word with a nonzero lane 3 is unprovable. A
/// caller that authenticated an extension group WITHOUT folding it would owe
/// that check itself.
pub fn emit_leaf_hash(b: &mut LfmBuilder, shape: GroupShape, values: &[Cell]) -> KeccakDigest {
    use super::keccak_host::BYTES_PER_HALF;
    use super::transcript_replay::felt_be_halves;

    assert_eq!(
        values.len(),
        shape.num_values(),
        "a leaf covers the whole row pair"
    );
    if !shape.is_ext {
        let felts: Vec<Felt> = values.iter().map(|c| Felt(c.addr())).collect();
        return edsl::keccak_leaf_hash(b, &felts);
    }

    let mut stream = Vec::with_capacity(6 * values.len());
    for v in values {
        let lanes = b.unpack(*v);
        for lane in lanes.iter().take(3) {
            stream.extend(felt_be_halves(b, *lane));
        }
    }
    let len_bytes = BYTES_PER_HALF * stream.len();
    debug_assert_eq!(len_bytes, shape.leaf_bytes());
    edsl::keccak256(b, &stream, len_bytes)
}

/// Authenticate one group's opened values against its committed root.
///
/// Takes the caller's cells and never hints a value, so what it authenticates
/// is what the caller folds. The assert is the binding; `bits` are shared with
/// every other group of the same query, which is what makes the four trees
/// agree about WHICH leaf they opened.
pub fn emit_group_authentication(
    b: &mut LfmBuilder,
    commitment: &GroupCommitment,
    opening: &GroupOpening,
    bits: &[Bit],
) {
    assert_eq!(
        opening.siblings.len(),
        bits.len(),
        "one sibling per level, and every group walks the same index"
    );
    let leaf = emit_leaf_hash(b, commitment.shape, &opening.values);
    let root = edsl::keccak_merkle_walk(b, leaf, bits, &opening.siblings);
    edsl::assert_word_eq_lanes(b, root[0], &commitment.root_lanes[0]);
    edsl::assert_word_eq_lanes(b, root[1], &commitment.root_lanes[1]);
}

/// The LDE-domain constants the point derivation multiplies together:
/// `factors[i] = g^{2^{depth-1-i}}`, matching index bit `i`'s weight after the
/// bit reversal.
fn point_factors(log2_lde_length: u32) -> Vec<FE> {
    let g = <GoldilocksField as IsFFTField>::get_primitive_root_of_unity(log2_lde_length as u64)
        .expect("a power-of-two LDE length has a root of unity");
    let depth = log2_lde_length as usize - 1;
    (0..depth).map(|i| g.pow(1u64 << (depth - 1 - i))).collect()
}

/// `(υ, −υ)` from the query index bits, for the LDE domain given by its size and
/// coset offset. Shape-only inputs: the factors are program constants.
///
/// Keyed on the domain rather than on a [`SubProofShape`] because the FRI leg
/// needs the same derivation and has no trace shape to hand — it holds a
/// [`super::fri::FriShape`], which carries both of these fields. One derivation
/// serves both, which is the point: `join_tests::the_join_premises_hold_on_a_real_proof`
/// checks THIS function against production's
/// `query_challenge_to_evaluation_point` at every index of a real proof, and a
/// second copy would not be covered by that check.
pub fn emit_points_from_bits(
    b: &mut LfmBuilder,
    log2_lde_length: u32,
    coset_offset: FE,
    bits: &[Bit],
) -> (Felt, Felt) {
    assert_eq!(
        bits.len(),
        log2_lde_length as usize - 1,
        "a leaf is a row pair, so the index is one bit narrower than the domain"
    );
    let point = edsl::pow_bits(b, bits, &point_factors(log2_lde_length), coset_offset);
    let zero = b.felt_const(FE::zero());
    (point, b.sub(zero, point))
}

/// `(υ, −υ)` from the query index bits.
pub fn emit_query_points(b: &mut LfmBuilder, shape: &SubProofShape, bits: &[Bit]) -> (Felt, Felt) {
    assert_eq!(bits.len(), shape.merkle_depth);
    emit_points_from_bits(b, shape.log2_lde_length, shape.coset_offset, bits)
}

/// Everything one query of one sub-proof contributes, emitted.
///
/// Order of business: decompose the index, authenticate every group against
/// its root, derive the two points from the same bits, then fold DEEP at both.
/// Returns `(DEEP(υ), DEEP(−υ))` for the FRI leg to consume.
///
/// `trace_openings` is parallel to [`SubProofShape::trace_groups`]; the parts
/// opening is separate because DEEP treats it separately.
pub fn emit_query(
    b: &mut LfmBuilder,
    shape: &SubProofShape,
    gamma: Ext,
    inv: &DeepInvariants,
    commitments: &[GroupCommitment],
    index: Felt,
    openings: &[GroupOpening],
) -> (Ext, Ext) {
    emit_query_with_bits(b, shape, gamma, inv, commitments, index, openings).deep
}

/// What one query contributes when the caller needs more than the DEEP pair.
pub struct QueryOutput {
    /// `(DEEP(υ), DEEP(−υ))`.
    pub deep: (Ext, Ext),
    /// The query index decomposed low-to-high — the SAME cells the Merkle walk
    /// consumed and the query points were derived from.
    ///
    /// Handing these out is what lets a later leg join to this one rather than
    /// run beside it. FRI reuses the index per layer (leaf position `index >> 1`,
    /// partner `index ^ 1`, halving each layer), and a leg that decomposed its
    /// own copy would authenticate one index while folding at another — the
    /// exact gap this module exists to close, reopened one level up. There is
    /// no way to return a DIFFERENT decomposition from here: `bit_dec` is
    /// called once and its result feeds the walk, the points and this field.
    pub bits: Vec<Bit>,
    /// `υ` — the cell the DEEP fold above evaluated at.
    ///
    /// Exposed for the same reason as [`Self::bits`], one step further along.
    /// FRI needs `υ⁻¹` for its first fold and `υ^(2^total_folds)` for its
    /// terminal check; both are functions of this cell, and a leg that
    /// re-derived the point from `bits` would pay `merkle_depth` `Select`s and
    /// `Mul`s per query for a value it was already holding. Handing the cell
    /// over is not just cheaper, it removes the question: there is exactly one
    /// `emit_query_points` call in this function and its outputs go to DEEP and
    /// to these fields, so no second point EXISTS to disagree.
    ///
    /// The structural guard is a count, not a comparison — see
    /// `fri_tests::the_fri_join_adds_no_second_point_derivation`.
    pub point: Felt,
    /// `−υ`, likewise. The zero-fold FRI shape checks the terminal polynomial
    /// at both points (production's `zetas.is_empty()` branch tests
    /// `terminal[2·iota]` AND `terminal[2·iota+1]`).
    pub point_sym: Felt,
}

/// [`emit_query`], additionally returning the index bits — see [`QueryOutput`].
///
/// The index arrives as a FELT here, which is the isolation drivers' route: the
/// differential supplies production's own `iota` and the emitter decomposes it.
/// The assembled verifier does not have a felt to supply — its index is
/// `TranscriptReplay::sample_u64_pow2`'s bits — and takes
/// [`emit_query_from_bits`] instead, which is the same emitter minus this one
/// `bit_dec`.
#[allow(clippy::too_many_arguments)]
pub fn emit_query_with_bits(
    b: &mut LfmBuilder,
    shape: &SubProofShape,
    gamma: Ext,
    inv: &DeepInvariants,
    commitments: &[GroupCommitment],
    index: Felt,
    openings: &[GroupOpening],
) -> QueryOutput {
    let bits = b.bit_dec(index, shape.merkle_depth);
    emit_query_from_bits(b, shape, gamma, inv, commitments, bits, openings)
}

/// [`emit_query_with_bits`] over an index the caller already holds as BITS.
///
/// This is the entry point the assembled epoch verifier uses. Production's query
/// index is `sample_u64(lde_length >> 1)`, whose output is `index_bits()` bits by
/// construction (`verifier.rs:138-141`), and the machine's
/// `TranscriptReplay::sample_u64_pow2` produces exactly those bits. Routing them
/// straight in — rather than recomposing a felt and decomposing it again — is
/// what makes the assembled machine's query index in-range by construction and
/// closes ledger entry 5: with no felt in the program, `ι` and `ι + 2^(n−1)`
/// cannot be the same query, because neither is ever a number.
#[allow(clippy::too_many_arguments)]
pub fn emit_query_from_bits(
    b: &mut LfmBuilder,
    shape: &SubProofShape,
    gamma: Ext,
    inv: &DeepInvariants,
    commitments: &[GroupCommitment],
    bits: Vec<Bit>,
    openings: &[GroupOpening],
) -> QueryOutput {
    shape.check();
    let groups = shape.groups();
    assert_eq!(commitments.len(), groups.len(), "one commitment per group");
    assert_eq!(openings.len(), groups.len(), "one opening per group");
    for (c, g) in commitments.iter().zip(&groups) {
        assert_eq!(c.shape, *g, "commitment shapes must match the sub-proof");
    }
    assert_eq!(
        bits.len(),
        shape.merkle_depth,
        "a query index is exactly the tree's depth in bits"
    );

    for (commitment, opening) in commitments.iter().zip(openings) {
        emit_group_authentication(b, commitment, opening, &bits);
    }

    let (point, point_sym) = emit_query_points(b, shape, &bits);

    // The crossing: the authenticated cells, re-read by POINT instead of by
    // matrix. Nothing is hinted here, so `trace` cannot hold anything the walk
    // above did not fold into a leaf.
    let mut trace = Vec::with_capacity(shape.deep.num_total_cols);
    let mut trace_sym = Vec::with_capacity(shape.deep.num_total_cols);
    for (opening, g) in openings.iter().zip(&groups).take(shape.trace_groups.len()) {
        for c in 0..g.num_columns {
            trace.push(opening.values[c].as_ext());
            trace_sym.push(opening.values[g.num_columns + c].as_ext());
        }
    }

    let parts_opening = openings.last().expect("the parts group is always present");
    let num_parts = shape.deep.num_composition_parts;
    let parts: Vec<Ext> = (0..num_parts)
        .map(|j| parts_opening.values[j].as_ext())
        .collect();
    let parts_sym: Vec<Ext> = (0..num_parts)
        .map(|j| parts_opening.values[num_parts + j].as_ext())
        .collect();

    let regular = DeepOpening {
        point,
        trace,
        parts,
    };
    let symmetric = DeepOpening {
        point: point_sym,
        trace: trace_sym,
        parts: parts_sym,
    };
    QueryOutput {
        deep: (
            emit_deep_point(b, &shape.deep, gamma, inv, &regular),
            emit_deep_point(b, &shape.deep, gamma, inv, &symmetric),
        ),
        bits,
        point,
        point_sym,
    }
}

// ===================== the whole sub-proof =====================

/// The arenas one sub-proof's verification reads, in declaration order.
///
/// Each field is packed into its OWN arena rather than one concatenated
/// stream — the packing rule [`super::proof_arena`] exists to enforce, applied
/// one level up: a query whose group widths shifted would otherwise silently
/// slide every query behind it.
pub struct SubProofArenas {
    /// `γ`, then `ζ`.
    pub uniforms: super::instr::ArenaId,
    /// The reconstructed OOD grid, row-major, `num_eval_points ×
    /// num_total_cols` — the same values the constraint leg folds.
    pub ood: super::instr::ArenaId,
    /// The composition parts claimed at `z^P`.
    pub parts: super::instr::ArenaId,
    /// Two words per group's committed root, in [`SubProofShape::groups`] order.
    pub roots: super::instr::ArenaId,
    /// Per query, in order: the index, then per group the row-pair values
    /// followed by the sibling digests (two words per level).
    pub queries: super::instr::ArenaId,
}

/// Emit a whole sub-proof's query verification: the invariants once, then every
/// query authenticated and folded.
///
/// Returns `(DEEP(υ), DEEP(−υ))` per query. The invariant hoist is the reason a
/// 219-query proof is affordable, and it is production's own hoist — the OOD
/// row sums and the block scalars do not depend on the query.
pub fn emit_sub_proof(
    b: &mut LfmBuilder,
    shape: &SubProofShape,
    num_queries: usize,
) -> (SubProofArenas, Vec<(Ext, Ext)>) {
    let (arenas, out) = emit_sub_proof_with_bits(b, shape, num_queries);
    (arenas, out.into_iter().map(|q| q.deep).collect())
}

/// [`emit_sub_proof`], additionally returning each query's index bits — see
/// [`QueryOutput`]. The FRI leg folds from these same cells.
pub fn emit_sub_proof_with_bits(
    b: &mut LfmBuilder,
    shape: &SubProofShape,
    num_queries: usize,
) -> (SubProofArenas, Vec<QueryOutput>) {
    use super::deep::emit_deep_invariants;

    shape.check();
    assert!(num_queries > 0, "a proof carries at least one query");
    let groups = shape.groups();

    let uniforms = b.declare_arena(2);
    let ood = b.declare_arena((shape.deep.num_eval_points * shape.deep.num_total_cols) as u32);
    let parts = b.declare_arena(shape.deep.num_composition_parts as u32);
    let roots = b.declare_arena(2 * groups.len() as u32);
    let queries = b.declare_arena((num_queries * shape.query_words()) as u32);
    let arenas = SubProofArenas {
        uniforms,
        ood,
        parts,
        roots,
        queries,
    };

    let gamma = b.hint_word(uniforms, 0).as_ext();
    let zeta = b.hint_word(uniforms, 1).as_ext();

    let mut next = 0u32;
    let ood_steps: Vec<Vec<Ext>> = (0..shape.deep.num_eval_points)
        .map(|_| {
            (0..shape.deep.num_total_cols)
                .map(|_| {
                    let c = b.hint_word(ood, next).as_ext();
                    next += 1;
                    c
                })
                .collect()
        })
        .collect();
    let claimed_parts: Vec<Ext> = (0..shape.deep.num_composition_parts as u32)
        .map(|j| b.hint_word(parts, j).as_ext())
        .collect();

    let commitments: Vec<GroupCommitment> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| GroupCommitment::hint(b, roots, 2 * i as u32, *g))
        .collect();

    let inv = emit_deep_invariants(b, &shape.deep, gamma, zeta, &ood_steps, &claimed_parts);

    let mut cursor = 0u32;
    let mut out = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let index = b.hint_felt(queries, cursor);
        cursor += 1;
        let openings: Vec<GroupOpening> = groups
            .iter()
            .map(|g| {
                let values: Vec<Cell> = (0..g.num_values())
                    .map(|_| {
                        let c = b.hint_word(queries, cursor);
                        cursor += 1;
                        c
                    })
                    .collect();
                let siblings: Vec<KeccakDigest> = (0..shape.merkle_depth)
                    .map(|_| {
                        let lo = b.hint_word(queries, cursor);
                        let hi = b.hint_word(queries, cursor + 1);
                        cursor += 2;
                        [lo, hi]
                    })
                    .collect();
                GroupOpening { values, siblings }
            })
            .collect();
        out.push(emit_query_with_bits(
            b,
            shape,
            gamma,
            &inv,
            &commitments,
            index,
            &openings,
        ));
    }
    assert_eq!(
        cursor as usize,
        num_queries * shape.query_words(),
        "the emitter's cursor must agree with the declared query stride"
    );

    (arenas, out)
}

//! One sub-proof VERIFIED — the legs hung off the Fiat-Shamir spine.
//!
//! [`super::epoch`] replays production's challenge derivation and hands back
//! [`TableChallenges`]; every leg built so far took those same challenges as
//! arena words instead. This module is where the two meet: it takes the cells
//! the spine absorbed and the challenges the spine derived, and emits the four
//! checks a real verifier performs on one sub-proof.
//!
//! ```text
//!   spine gives                     leg consumes
//!   -----------                     ------------
//!   ood_current / ood_next  ──────► the reconstructed grid: constraints AND DEEP
//!   parts                   ──────► the quotient's claimed value AND DEEP's h_sum
//!   z                       ──────► the zerofier's ζ AND DEEP's row points
//!   beta                    ──────► the β-power fold
//!   gamma                   ──────► the DEEP batching challenge
//!   zetas                   ──────► the FRI fold chain
//!   iota_bits               ──────► the Merkle walk, the query point, the FRI walk
//!   contribution (L)        ──────► the table offset AND the LogUp closure
//!   every ROOT              ──────► the authentication compare
//! ```
//!
//! Nothing in that table is hinted twice. The roots arrive as
//! [`super::epoch::RootCells`] and become [`GroupCommitment`]s through
//! `from_lanes`, the OOD blocks become one grid through
//! [`super::epoch::emit_reconstruct_ood`], and the query index never exists as a
//! felt. What the arenas still carry, per sub-proof, is exactly the data a real
//! proof carries and a verifier cannot derive: the opened row pairs, the Merkle
//! paths, and the FRI layers' symmetric evaluations.
//!
//! ## What this module cannot see
//!
//! It verifies ONE sub-proof. It says nothing about the epoch's statement, about
//! Phase A, or about the LogUp closure across tables — those are the spine's and
//! [`super::logup`]'s. It also does not check the preprocessed commitments
//! against anything: production takes them from the AIR, and where they come
//! from in the assembled machine is [`TableInputs`]' caller's problem (assembly
//! ledger entry 7).

use crate::tables::types::{FE, FEE};

use super::builder::{Ext, LfmBuilder};
use super::constraints::{
    Analysis, BoundaryTerm, OodOperands, QuotientShape, emit_alpha_powers, emit_analyzed,
    emit_quotient, emit_table_offset,
};
use super::deep::{DeepInvariants, emit_deep_invariants};
use super::edsl::WrapHash;
use super::epoch::{RootCells, TableAbsorbs, TableChallenges, emit_reconstruct_ood};
use super::fri::{
    FriCommitments, FriQuery, FriShape, LayerCommitment, emit_query_fri, hint_layer_openings_from,
};
use super::instr::ArenaId;
use super::sub_proof::{GroupCommitment, GroupOpening, SubProofShape, emit_query_from_bits};

/// The compile-time shape of one sub-proof's full verification.
///
/// Every field is program SHAPE, in the sense `others/lfm-target-shape.md` fixes:
/// a value the AIR set and the proof options determine, never a value the proof
/// carries. The one exception worth naming is [`Self::quotient`]'s boundary list,
/// which production computes from the public inputs — see [`boundary_terms`] for
/// the rule it is built from and the premise that rule rests on.
#[derive(Clone, Debug)]
pub struct TableVerifyShape {
    /// The trace/opening shape: DEEP columns, the committed groups, the tree
    /// depth and the LDE domain.
    pub sub: SubProofShape,
    /// The FRI shape, which must describe the same LDE domain.
    pub fri: FriShape,
    /// The zerofier, the part count and the boundary constraints.
    pub quotient: QuotientShape,
    /// Where the aux columns start in a full-width `[main | aux]` row.
    pub main_width: usize,
    /// `AIR::max_bus_elements()` — how long the α-power chain is. Zero for an
    /// AIR with no aux trace, which has no `Op::AlphaPow` to resolve.
    pub num_alpha_powers: usize,
    /// Queries the sub-proof carries.
    pub num_queries: usize,
}

impl TableVerifyShape {
    /// Frame STEPS the constraint program indexes — `Op::Var{offset}` runs over
    /// these, not over the OOD grid's rows.
    ///
    /// A frame step is `step_size` grid rows and production's own interpreter
    /// reads only row 0 of each (`constraint_ir/interp.rs:240-242` asserts
    /// `row == 0`), so the constraint leg's view of the grid is every
    /// `step_size`-th row while DEEP's is all of it. The two coincide at
    /// `step_size = 1`, which every production AIR has — carrying the stride
    /// anyway is the same discipline `DeepShape::block` applies to the
    /// coefficient run, and for the same reason.
    pub fn num_frame_steps(&self) -> usize {
        self.sub.deep.num_eval_points / self.sub.deep.step_size
    }

    fn check(&self) {
        assert_eq!(
            self.sub.log2_lde_length, self.fri.log2_lde_length,
            "both legs verify one sub-proof over one LDE domain"
        );
        assert_eq!(
            self.sub.merkle_depth,
            self.fri.index_bits(),
            "the FRI layers consume suffixes of the trace walk's decomposition"
        );
        assert_eq!(
            self.fri.num_queries, self.num_queries,
            "the query count is one shape, declared once"
        );
        assert_eq!(
            self.sub.deep.num_composition_parts, self.quotient.num_composition_parts,
            "the part count is one shape: DEEP folds the same parts the quotient \
             Horner claims"
        );
        assert_eq!(
            self.sub.deep.log2_trace_length, self.quotient.log2_trace_length,
            "the trace length is one shape"
        );
        assert_eq!(
            self.num_frame_steps() * self.sub.deep.step_size,
            self.sub.deep.num_eval_points,
            "the OOD grid is a whole number of frame steps"
        );
        assert!(
            self.main_width <= self.sub.deep.num_total_cols,
            "the aux columns start inside the row"
        );
    }

    /// Arena words this sub-proof's trace openings occupy.
    pub fn opening_words(&self) -> usize {
        self.num_queries * self.sub.opening_words()
    }

    /// Arena words this sub-proof's FRI openings occupy.
    pub fn fri_words(&self) -> usize {
        self.num_queries * self.fri.query_words()
    }
}

/// The two arenas one sub-proof's query verification reads, in declaration
/// order.
///
/// Deliberately only two. Everything else a leg used to hint — the roots, the
/// challenges, the OOD grid, the claimed parts, the FRI layer roots and terminal
/// coefficients — reaches the legs as cells the spine already bound.
#[derive(Clone, Copy, Debug)]
pub struct TableQueryArenas {
    /// Per query, per group: the row-pair values then the sibling digests (two
    /// words per level). NO index word — the index is the transcript's.
    pub openings: ArenaId,
    /// Per query, per committed FRI layer: the symmetric evaluation then the
    /// sibling digests.
    pub fri: ArenaId,
}

/// Declare the query arenas for one sub-proof.
pub fn declare_table_arenas(b: &mut LfmBuilder, shape: &TableVerifyShape) -> TableQueryArenas {
    TableQueryArenas {
        openings: b.declare_arena(shape.opening_words() as u32),
        fri: b.declare_arena(shape.fri_words() as u32),
    }
}

/// The cells one sub-proof's verification takes from OUTSIDE its own arenas.
pub struct TableInputs<'a> {
    /// The precomputed-columns root, when the AIR is preprocessed.
    ///
    /// Production never reads this from the proof: it takes
    /// `air.precomputed_commitment()`, absorbs THAT, and rejects a proof whose
    /// copy disagrees (`verifier.rs:1184-1209`). So the cells here are the ones
    /// Phase A absorbed, and the equality production checks explicitly is, in
    /// this machine, the absence of a second value.
    pub precomputed_root: Option<&'a RootCells>,
    /// The main trace root — the cells Phase A absorbed.
    pub main_root: &'a RootCells,
    /// The shared LogUp challenges, sampled once in Phase A and passed to every
    /// table (`verifier.rs:1216-1227`). Never per-table.
    pub rap_challenges: &'a [Ext],
}

/// What one sub-proof's verification produced, for the epoch to compose.
pub struct TableVerifyOutput {
    /// The recomputed composition at `z`, asserted equal to the claimed Horner
    /// inside the program.
    pub composition: Ext,
    /// Per query, the FRI terminal value the chain arrived at.
    pub fri_terminal: Vec<Ext>,
    /// The per-sub-proof DEEP invariants, exposed so a test can publish them.
    pub deep: DeepInvariants,
}

/// Emit one sub-proof's verification onto the spine's cells.
///
/// `challenges` must be the output of [`super::epoch::emit_table_challenges`] on
/// THIS table's fork, and `absorbs` the very struct that call was given. Passing
/// a different one would be the two-consumer hazard reintroduced by hand, which
/// is why both are borrowed rather than rebuilt.
pub fn emit_table_verification(
    b: &mut LfmBuilder,
    shape: &TableVerifyShape,
    analysis: &Analysis,
    challenges: &TableChallenges,
    absorbs: &TableAbsorbs<'_>,
    inputs: &TableInputs<'_>,
    arenas: &TableQueryArenas,
) -> TableVerifyOutput {
    shape.check();
    assert_eq!(
        challenges.iota_bits.len(),
        shape.num_queries,
        "one index per query"
    );

    // ---- the OOD grid, from the two blocks the transcript absorbed.
    let grid = emit_reconstruct_ood(b, &shape.sub.deep, absorbs.ood_current, absorbs.ood_next);

    // ---- the LogUp uniforms, DERIVED (never hinted): the α powers from the one
    // α Phase A sampled, and the per-row offset from the one `L` this table's
    // fork absorbed and the closure sums.
    let alpha_powers = if shape.num_alpha_powers > 0 {
        let alpha = inputs
            .rap_challenges
            .get(stark::lookup::LOGUP_CHALLENGE_ALPHA)
            .copied()
            .expect("an AIR with bus elements has the shared LogUp challenges");
        emit_alpha_powers(b, alpha, shape.num_alpha_powers)
    } else {
        Vec::new()
    };
    let table_offset = match absorbs.contribution {
        Some(l) => emit_table_offset(b, l, shape.quotient.log2_trace_length),
        // An AIR with no bus contribution has no `Op::TableOffset` to resolve;
        // the pooled zero is a placeholder the lowering never reads. It is a
        // program constant, so a prover cannot reach it either way.
        None => b.felt_const(FE::zero()).as_ext(),
    };

    // The constraint program indexes FRAME STEPS; DEEP folds every grid row.
    // Both views are of the one grid above, which is what makes the two legs
    // agree by construction rather than by the host filling two arenas alike.
    let steps = frame_step_view(&grid, shape.sub.deep.step_size);
    assert_eq!(
        steps.len(),
        shape.num_frame_steps(),
        "the strided view must have one entry per frame step"
    );
    let ood = OodOperands {
        steps,
        main_width: shape.main_width,
        rap_challenges: inputs.rap_challenges.to_vec(),
        alpha_powers,
        table_offset,
    };

    // ---- (1) the constraint evaluation and (2) the quotient check.
    let evals = emit_analyzed(b, analysis, &ood);
    let q = emit_quotient(
        b,
        &shape.quotient,
        &ood,
        challenges.z,
        challenges.beta,
        &evals,
        absorbs.parts,
    );
    b.assert_eq_ext(q.claimed, q.composition);

    // ---- (3) DEEP, over the same grid and the same parts.
    let inv = emit_deep_invariants(
        b,
        &shape.sub.deep,
        challenges.gamma,
        challenges.z,
        &grid,
        absorbs.parts,
    );

    // ---- the committed matrices, in DEEP column order then the parts.
    let groups = shape.sub.groups();
    let mut commitments: Vec<GroupCommitment> = Vec::with_capacity(groups.len());
    let push = |root: &RootCells, out: &mut Vec<GroupCommitment>| {
        let g = groups[out.len()];
        out.push(GroupCommitment::from_lanes(root.lanes.clone(), g));
    };
    if let Some(prep) = inputs.precomputed_root {
        push(prep, &mut commitments);
    }
    push(inputs.main_root, &mut commitments);
    if let Some(aux) = absorbs.aux_root {
        push(aux, &mut commitments);
    }
    push(absorbs.composition_root, &mut commitments);
    assert_eq!(
        commitments.len(),
        groups.len(),
        "one commitment per committed matrix: the sub-proof shape and the \
         supplied roots must describe the same proof"
    );

    // ---- the FRI commitments, likewise from the transcript's own cells.
    let fri = FriCommitments {
        layers: absorbs
            .fri_roots
            .iter()
            .map(|r| LayerCommitment::from_lanes(r.lanes.clone()))
            .collect(),
        zetas: challenges.zetas.clone(),
        coeffs: absorbs.fri_coeffs.to_vec(),
    };

    // ---- (4) per query: authenticate, fold DEEP, then fold FRI.
    let stride = shape.sub.opening_words();
    let mut fri_terminal = Vec::with_capacity(shape.num_queries);
    for (qi, bits) in challenges.iota_bits.iter().enumerate() {
        let mut cursor = (qi * stride) as u32;
        let openings: Vec<GroupOpening> = groups
            .iter()
            .map(|g| {
                let values = (0..g.num_values())
                    .map(|_| {
                        let c = b.hint_word(arenas.openings, cursor);
                        cursor += 1;
                        c
                    })
                    .collect();
                let siblings = (0..shape.sub.merkle_depth)
                    .map(|_| {
                        // The stride follows the DIGEST's width, not a literal.
                        let d = super::edsl::hint_digest(b, arenas.openings, cursor);
                        cursor += super::edsl::digest_words(b);
                        d
                    })
                    .collect();
                GroupOpening { values, siblings }
            })
            .collect();
        assert_eq!(
            cursor as usize,
            (qi + 1) * stride,
            "the emitter's cursor must agree with the declared query stride"
        );

        let out = emit_query_from_bits(
            b,
            &shape.sub,
            challenges.gamma,
            &inv,
            &commitments,
            bits.clone(),
            &openings,
        );
        let layers = hint_layer_openings_from(b, shape.fri, arenas.fri, qi);
        fri_terminal.push(emit_query_fri(
            b,
            shape.fri,
            &fri,
            &FriQuery {
                p0: out.deep.0,
                p0_sym: out.deep.1,
                point: out.point,
                point_sym: out.point_sym,
                bits: &out.bits,
            },
            &layers,
        ));
    }

    TableVerifyOutput {
        composition: q.composition,
        fri_terminal,
        deep: inv,
    }
}

/// The constraint frame's view of the reconstructed OOD grid: row 0 of each
/// evaluation STEP, which is every `step_size`-th grid row.
///
/// This is assembly ledger entry 9, extracted so it can be differentialled. The
/// rule is production's, not ours: the verifier builds its frame with
/// `StarkTableView::into_frame(main_cols, step_size)`
/// (`verifier.rs:320-321`), which groups the `num_eval_points`-row grid into
/// `step_size`-row steps, and `Op::Var{offset, row}` resolves to
/// `frame.get_evaluation_step(offset).get_main_evaluation_element(0, col)` with
/// `row == 0` asserted (`constraint_ir/interp.rs:240-242`). So step `o`'s value is
/// grid row `o · step_size`, and the rows between are read by DEEP alone.
///
/// A generic function rather than the two lines it replaces, because those two
/// lines had no witness: at `step_size = 1` the strided view and the whole grid
/// are the same vector, so an emitter that passed the whole grid to the
/// constraint fold — which is what the wave-5 sketch did — was indistinguishable.
/// `step_size_tests::the_frame_step_view_matches_productions_own_frame_assembly`
/// compares THIS function against `into_frame` at `step_size = 2`, where they
/// differ.
pub fn frame_step_view<T: Clone>(grid: &[Vec<T>], step_size: usize) -> Vec<Vec<T>> {
    assert!(step_size > 0, "a frame step is at least one row");
    assert!(
        grid.len().is_multiple_of(step_size),
        "the OOD grid is a whole number of frame steps: {} rows at step_size {step_size}",
        grid.len()
    );
    grid.iter().step_by(step_size).cloned().collect()
}

/// Keccak permutations one sub-proof's committed leaves cost, per query.
///
/// A leaf is NOT one permutation. It covers `ROWS_PER_LEAF · num_columns`
/// elements at 8 or 24 bytes each, and the sponge absorbs `⌊bytes/136⌋ + 1` rate
/// blocks — so the epoch's widest table (2,056 OOD columns) has a leaf worth
/// hundreds of permutations while a FRI layer's one-column leaf is worth one.
/// Predicting the leg's bill as "one leaf plus one per level" undercounts it by
/// the whole width of the trace, which is exactly the mistake this function
/// exists to not make.
pub fn leaf_permutations(shape: &SubProofShape) -> usize {
    shape
        .groups()
        .iter()
        .map(|g| super::keccak_host::num_blocks(g.leaf_bytes()))
        .sum()
}

/// Felts a keccak permutation absorbs — the 136-byte rate at 8 bytes per felt.
///
/// `⌊bytes/136⌋` and `⌊felts/17⌋` are the same function because every element the
/// leaf hasher streams is 8 bytes wide (a base felt) or 24 (an extension element,
/// three felts), so `bytes = 8 · felts` with no remainder either way.
pub const KECCAK_RATE_FELTS: usize = 17;

/// Felts an `LFM_HASH` invocation absorbs — **one digest cell**.
///
/// ⚠ **This was 8 and is now 4, because the construction it was derived from no
/// longer exists.** The old value was "2 of 3 state cells" — the rate of the
/// overwrite-duplex `edsl::SpongeVar` used to be. Option B1 replaced that with a
/// **compress chain** that absorbs exactly one cell per step, so the rate is
/// [`super::hash::HASH_DIGEST_FELTS`]. It is written as that constant rather
/// than as a literal so it cannot outlive its own derivation a second time.
///
/// **This is 4.25× worse than keccak's 17**, not the 2.125× the duplex gave,
/// and it is the one axis on which a field-native candidate loses: it pays more
/// permutations, each far cheaper.
///
/// **The lever moved with the construction, and it is now a worse one.** Under
/// the duplex the rate followed from `HASH_STATE_FELTS = 12`, so widening the
/// state bought throughput and nothing else. Under the chain it follows from
/// `HASH_DIGEST_FELTS = 4` — the absorbed operand IS a digest — so the only way
/// to raise it is to widen the digest, which is the same constant the socket's
/// 64-bit collision bound rests on. Throughput and collision resistance are no
/// longer independent knobs.
///
/// ✗ **What this models, and its remaining assumption.** A rate is the right
/// shape for a CHAIN — `state ← T(state, cell)`, one fresh cell per invocation —
/// which is what the machine now has. It is *not* the right shape for a 2-to-1
/// tree over leaf data, where both inputs are fresh and the count is a tree size
/// rather than `felts / rate`. Which of the two a candidate uses for LEAVES is
/// the still-open O1 leaf-convention question, so a future answer there could
/// move this model's FORMULA and not merely this constant.
pub const LFM_HASH_RATE_FELTS: usize = super::hash::HASH_DIGEST_FELTS;

/// Felts in one FRI-layer leaf: the symmetric evaluation PAIR, two extension
/// elements of three felts each.
///
/// Named because it is the number that decides whether the FRI leaf term is
/// rate-sensitive: six felts fit one keccak block (rate 17) and do NOT fit one
/// block at the candidate's rate 4.
pub const FRI_LEAF_FELTS: usize = 6;

/// Felts one query's opening of a group covers, the felt-side counterpart of
/// [`super::sub_proof::GroupShape::leaf_bytes`].
pub fn group_leaf_felts(g: &super::sub_proof::GroupShape) -> usize {
    g.num_values() * if g.is_ext { 3 } else { 1 }
}

/// Permutations a sponge of `rate_felts` spends absorbing `felts`, under keccak's
/// own padding convention — `⌊n/rate⌋ + 1`, i.e. always at least one block and
/// always a padding block even when the length divides the rate.
///
/// Carrying keccak's convention over to a candidate is deliberately
/// CONSERVATIVE: a candidate needs no trailing block, because it
/// domain-separates outside the absorbed data — in the capacity for a sponge,
/// and in the message tag for the B1 compress chain — so its true count lies
/// between `felts.div_ceil(rate)` and this. Using the same convention on both
/// sides is
/// what makes the rate-17 case reproduce [`leaf_permutations`] exactly, which is
/// the check that this felt-side reformulation is right at all.
pub fn blocks_at_rate(felts: usize, rate_felts: usize) -> usize {
    felts / rate_felts + 1
}

/// [`leaf_permutations`] at an arbitrary sponge rate.
///
/// Computed independently of [`leaf_permutations`] — through felts and a rate
/// rather than through bytes and `keccak_host::num_blocks` — precisely so that
/// asserting the two agree at [`KECCAK_RATE_FELTS`] is a real differential and
/// not two spellings of one function. Making either delegate to the other would
/// kill that test silently (standing-decisions rule 7).
pub fn leaf_permutations_at_rate(shape: &SubProofShape, rate_felts: usize) -> usize {
    shape
        .groups()
        .iter()
        .map(|g| blocks_at_rate(group_leaf_felts(g), rate_felts))
        .sum()
}

/// FRI-layer LEAF permutations one query costs, at an arbitrary rate.
///
/// Split out of [`super::fri::FriShape::permutations_per_query`] because it is
/// the half of that number which MOVES with the rate: a layer leaf is
/// [`FRI_LEAF_FELTS`] felts, and how many blocks that takes depends on the rate
/// like any other absorption. At keccak's 17 it is one block and this reduces to
/// `num_committed()`, which is what keeps the felt-side and byte-side closed
/// forms agreeing there.
///
/// ⚠ It used to be folded into the rate-INVARIANT remainder, on the premise
/// that "a FRI layer leaf fits any rate at or above six". That premise was true
/// while the candidate's rate was 8 and is false now that it is 4 — six felts
/// take two blocks. The premise is gone rather than re-asserted; this function
/// is what replaced it.
pub fn fri_leaf_permutations_at_rate(fri: &FriShape, rate_felts: usize) -> usize {
    fri.num_committed() * blocks_at_rate(FRI_LEAF_FELTS, rate_felts)
}

/// [`query_permutations`] at an arbitrary sponge rate.
///
/// **ABSORPTION is rate-sensitive; COMPRESSION is not.** That is the whole
/// decomposition, and it is a property of what each step does rather than of
/// which leg it belongs to:
///
/// - **Leaves absorb**, so both kinds move with the rate — the trace groups'
///   ([`leaf_permutations_at_rate`]) and the FRI layers'
///   ([`fri_leaf_permutations_at_rate`]).
/// - A **Merkle parent compresses**, so it is one permutation at any rate:
///   `LfmHasher::compress` is "a single permutation of `[a ‖ b ‖ IV]` truncated
///   to the first cell" (`hash.rs:23-26`), and keccak's 64-byte parent likewise
///   sits inside one 136-byte block (`edsl.rs:151-155`). Both the trace trees'
///   parents and the FRI layers' path steps are of this kind.
pub fn query_permutations_at_rate(shape: &TableVerifyShape, rate_felts: usize) -> usize {
    let groups = shape.sub.groups().len();
    let per_query = leaf_permutations_at_rate(&shape.sub, rate_felts)
        + fri_leaf_permutations_at_rate(&shape.fri, rate_felts)
        + groups * shape.sub.merkle_depth
        + shape.fri.path_steps_per_query();
    shape.num_queries * per_query
}

/// Felts a BLAKE3 block holds: 64 bytes at 8 bytes per felt.
pub const BLAKE3_BLOCK_FELTS: usize = 8;

/// Compressions absorbing `felts` costs under `hash`, with each hash's OWN
/// block rule.
///
/// The two rules genuinely differ and the difference is not a rate change.
/// Keccak's `⌊n/rate⌋ + 1` always spends a trailing block, because `pad10*1`
/// appends one even when the length divides the rate. `Blake3Chain` does not:
/// an exact multiple of 64 emits no spurious final block — that is KAT K5, and
/// the P2 parent identity depends on it. So BLAKE3 is `max(1, ⌈n/8⌉)`, with the
/// `max` carrying the other half of the same fact: the empty message is ONE
/// block, not zero.
pub fn blocks_for(felts: usize, hash: WrapHash) -> usize {
    match hash {
        WrapHash::Keccak => blocks_at_rate(felts, KECCAK_RATE_FELTS),
        WrapHash::Blake3 => felts.div_ceil(BLAKE3_BLOCK_FELTS).max(1),
        // ✓ PROVEN identical to the BLAKE3 arm for every leaf of one felt or
        // more (`rpo_chip_tests::the_rate_eight_census_is_hash_invariant`):
        // the rate is the same eight felts and neither spends a trailing block
        // on an exact multiple. Written as its own arm rather than merged, so
        // the equality stays a checked property instead of an assumption.
        WrapHash::Algebraic => felts.div_ceil(super::rpo::RATE_FELTS).max(1),
    }
}

/// [`query_permutations`] under an explicit wrap hash.
///
/// Same decomposition as [`query_permutations_at_rate`] — **absorption moves
/// with the block, compression does not** — but it takes the hash rather than a
/// rate, because the block COUNT rule is part of the hash and not a parameter
/// of it (see [`blocks_for`]). A Merkle parent is 64 bytes and is exactly one
/// compression under both, so every parent term is unchanged; only the leaf
/// absorptions move.
pub fn query_permutations_for(shape: &TableVerifyShape, hash: WrapHash) -> usize {
    let groups = shape.sub.groups().len();
    let leaves: usize = shape
        .sub
        .groups()
        .iter()
        .map(|g| blocks_for(group_leaf_felts(g), hash))
        .sum();
    let fri_leaves = shape.fri.num_committed() * blocks_for(FRI_LEAF_FELTS, hash);
    let per_query =
        leaves + fri_leaves + groups * shape.sub.merkle_depth + shape.fri.path_steps_per_query();
    shape.num_queries * per_query
}

/// Keccak permutations one sub-proof's whole query verification costs, from
/// shape alone.
///
/// Per query: every group's leaf ([`leaf_permutations`]), one permutation per
/// Merkle level per group (a parent hashes 64 bytes, one rate block), and the
/// FRI leg's own [`FriShape::permutations_per_query`]. A closed form over the
/// shapes, so comparing it against the emitted count is an absolute check and
/// not a difference of two of our own emitter passes.
pub fn query_permutations(shape: &TableVerifyShape) -> usize {
    let groups = shape.sub.groups().len();
    let per_query = leaf_permutations(&shape.sub)
        + groups * shape.sub.merkle_depth
        + shape.fri.permutations_per_query();
    shape.num_queries * per_query
}

/// The boundary constraints every production VM table carries, as program shape.
///
/// `AIR::boundary_constraints` is a function of the public inputs, so it is not a
/// static property of the AIR and cannot be captured into a
/// `ConstraintArtifact`. What IS static — and is checked against every AIR of a
/// real epoch by `epoch_verify_tests::the_boundary_terms_are_program_shape` — is
/// that a table with an aux trace carries exactly the framework's `acc[0] = 0` on
/// its last aux column and nothing else: a zero VALUE at the trace generator's
/// zeroth power, neither of which depends on a challenge or on the proof.
///
/// Building the list from that rule rather than from the call is what keeps the
/// emitted program independent of the proof it verifies; the test is what stops
/// the rule from silently ceasing to hold. Note which direction the risk runs: a
/// boundary term this rule MISSED would be a constraint the machine never
/// checks, so the test asserts set equality and not containment.
pub fn boundary_terms(has_aux_trace: bool, num_total_cols: usize) -> Vec<BoundaryTerm> {
    if has_aux_trace {
        vec![BoundaryTerm {
            col: num_total_cols - 1,
            point: FE::one(),
            value: FEE::zero(),
        }]
    } else {
        Vec::new()
    }
}

//! The aggregation NODE's building block: a per-table VERIFY LEG — the emitted
//! verifier of one child LFM proof — plus the cross-child binding legs.
//!
//! # What a leg is
//!
//! A leg is the machine twin of [`super::proof::verify_against_chunked`], which
//! is four steps and no more: absorb the LFM statement, fork the statement-bound
//! state and replay Phase A to recover the shared LogUp pair, recompute the
//! `LfmPublic` balance from the CLAIMED public words, and verify every
//! sub-proof against it. [`emit_leg`] emits exactly those four, so the leg and
//! the host verifier are two renderings of one contract rather than two
//! implementations of one idea.
//!
//! Nothing here is new cryptographic arithmetic. The spine is
//! [`super::epoch::emit_table_challenges`], the per-sub-proof verification is
//! [`super::epoch_verify::emit_table_verification`], Phase A is
//! [`super::statement_replay::replay_phase_a`] and the closure is
//! [`super::logup::emit_bus_closure`] — every one already gated by the epoch
//! wrap and by the global-memory leg. This module contributes the LFM-shaped
//! statement, the public-word hinting under a canonicity guard, the balance
//! target, and the binding legs.
//!
//! # Why a node is uniform
//!
//! A child is a plain per-table `MultiProof` whichever level produced it — a
//! wrap of an epoch, or another node. So one emitter serves every level, and a
//! node's statement is its children's published words and roots. What is NOT
//! uniform is the node's IDENTITY: a leg absorbs its child's `program_id` as an
//! emit-time CONSTANT, and `program_id` is derived from the compiled program, so
//! a program that verified its own shape would need its own id inside its own
//! instruction stream. That fixed point is why each level is a distinct program.
//!
//! # What the bindings are for
//!
//! Verifying two children says nothing about their relationship. The chain is a
//! CHECK on published words and never a trust: one shared attestation id across
//! every child, each child's register fini vector equal to the next child's init
//! vector, and each child's epoch label pinned to its chain position as an
//! emit-time constant. Those are [`emit_chain_bindings`], and they are the whole
//! reason a tree of these proves something a bag of them does not.

use stark::config::Commitment;

use crate::tables::types::{FE, FEE};

use super::builder::{Ext, Felt, LfmBuilder};
use super::constraints::Analysis;
use super::epoch::{RootCells, TableAbsorbs, TableChallengeShape, fork_table};
use super::epoch_verify::{TableQueryArenas, TableVerifyShape};
use super::instr::ArenaId;
use super::statement::{LFM_MACHINE_VERSION, LFM_STATEMENT_TAG};
use super::statement_replay::{PhaseAPreprocessed, PhaseATable, replay_phase_a};
use super::transcript_replay::{Candidate, TranscriptReplay, assert_canonical, candidate_to_felt};

/// `u32` halves one lane's canonical `u64` occupies — the unit
/// [`super::statement::absorb_lfm_statement`] appends a lane in.
const HALVES_PER_LANE: usize = 2;

/// Lanes one published word carries.
const LANES_PER_WORD: usize = super::word::WORD_LANES;

// ======================= the child's shape, at emit time ==================

/// One sub-proof of a child, as the leg needs it.
///
/// Every field is program SHAPE — derived from the AIR and the proof options,
/// never read off the proof at verify time. The one exception is the trace
/// length inside `verify`, which the prover chooses and which is program shape
/// here for the reason the arena schema makes it one: the node is emitted for a
/// specific child shape, and a child whose trace length disagreed would not fill
/// the arenas the node declares.
pub struct ChildTable<'a> {
    /// What the fork absorbs and what challenges come out of it.
    pub challenge: &'a TableChallengeShape,
    /// What the verification legs open.
    pub verify: &'a TableVerifyShape,
    /// The constraint program, captured from the AIR.
    pub analysis: &'a Analysis,
    /// The preprocessed-columns commitment, when the AIR is preprocessed.
    ///
    /// An AIR-SET constant at emit time, exactly as production takes it
    /// (`air.precomputed_commitment()`, never the proof's copy). Interning it
    /// here is what makes production's explicit proof-copy-equals-AIR-copy check
    /// the ABSENCE of a second value in this machine rather than a comparison.
    pub precomputed_root: Option<&'a Commitment>,
}

/// One child proof, as the node's emitter needs it.
pub struct ChildShape<'a> {
    /// The identity of the program that produced this child — a PROGRAM
    /// CONSTANT of the node, which is what pins WHICH program the node accepts a
    /// proof of.
    pub program_id: &'a Commitment,
    /// How many words the child publishes. Publish indices auto-increment from
    /// zero (`LfmBuilder::public`), so the count is the whole layout.
    pub num_public_words: usize,
    /// The child's `ProofOptions::fri_final_poly_log_degree` — the statement's
    /// last byte.
    pub fri_final_poly_log_degree: u8,
    /// The child's sub-proofs, in proof order.
    pub tables: Vec<ChildTable<'a>>,
}

// ==================== the emitted statement + publics ====================

/// One hinted public word of a child: the emit-time-constant index, the eight
/// hinted halves (absorbed by the statement), and the four lanes reassembled as
/// CANONICITY-GUARDED felts (consumed by the balance and the binding legs).
pub struct HintedPublicWord {
    pub index: u32,
    pub halves: Vec<Felt>,
    pub lanes: Vec<Felt>,
}

/// Hint a child's published words from `arena` (eight halves per word, the
/// serializer's layout) and reassemble each lane under the canonicity guard.
///
/// The guard is the same `(lo, hi)` predicate the transcript replay's own
/// digest-to-felt path enforces, so a hinted half pair CANNOT alias a second
/// representation of the same felt while absorbing different bytes. Without it a
/// prover could absorb one byte string into the statement and hand the balance
/// and the binding legs a different value for the same word.
///
/// ⚠ This is the node's per-word bill — eight hints, four guards, four
/// recombinations — and it is paid per word per child. What a child publishes is
/// therefore the size of the layer above it; see
/// `epoch_tests::Publishes`.
pub fn hint_public_words(
    b: &mut LfmBuilder,
    arena: ArenaId,
    count: usize,
) -> Vec<HintedPublicWord> {
    let mut cursor = 0u32;
    (0..count)
        .map(|index| {
            let mut halves = Vec::with_capacity(LANES_PER_WORD * HALVES_PER_LANE);
            let mut lanes = Vec::with_capacity(LANES_PER_WORD);
            for _ in 0..LANES_PER_WORD {
                let lo = b.hint_felt(arena, cursor);
                let hi = b.hint_felt(arena, cursor + 1);
                cursor += HALVES_PER_LANE as u32;
                let c = Candidate { lo, hi };
                assert_canonical(b, c);
                lanes.push(candidate_to_felt(b, c));
                halves.push(lo);
                halves.push(hi);
            }
            HintedPublicWord {
                index: index as u32,
                halves,
                lanes,
            }
        })
        .collect()
}

/// Emits [`super::statement::absorb_lfm_statement`] byte for byte: the tag, the
/// child's program id (a PROGRAM CONSTANT), the machine version, the word count,
/// each word's emit-time-constant index and hinted lane halves, and the FRI
/// terminal byte.
pub fn emit_lfm_statement(
    t: &mut TranscriptReplay,
    program_id: &Commitment,
    words: &[HintedPublicWord],
    fri_final_poly_log_degree: u8,
) {
    t.append_const_bytes(LFM_STATEMENT_TAG);
    t.append_const_bytes(program_id);
    t.append_const_bytes(&LFM_MACHINE_VERSION.to_le_bytes());
    t.append_const_bytes(&(words.len() as u64).to_le_bytes());
    for word in words {
        t.append_const_bytes(&word.index.to_le_bytes());
        // ⚠ ONE CALL PER LANE, not one for the word. `absorb_lfm_statement`
        // appends each lane's canonical `u64` separately, so a word is FIVE host
        // calls — the index and four lanes — not two. A byte transcript
        // concatenates and cannot tell the difference; an ALGEBRAIC one
        // length-prefixes every call, so absorbing the eight halves in one go is
        // a DIFFERENT transcript, and since the statement is absorbed first that
        // means every challenge downstream. See `transcript_replay::Append`.
        for lane in word.halves.chunks(HALVES_PER_LANE) {
            t.append_halves_misaligned(lane);
        }
    }
    t.append_const_bytes(&[fri_final_poly_log_degree]);
}

/// The `LfmPublic` balance the leg's LogUp closure must reach — the machine twin
/// of `proof::expected_public_balance`:
/// `Σ_i 1/(z − (LfmPublic + index_i·α + Σ_l lane_l·α^{2+l}))`, with each division
/// by an interned one so a fingerprint collision with `z` is unprovable rather
/// than `0/0 = 1`.
pub fn emit_public_balance(
    b: &mut LfmBuilder,
    words: &[HintedPublicWord],
    z: Ext,
    alpha: Ext,
) -> Ext {
    let bus = b.ext_const(&FEE::from(crate::tables::types::BusId::LfmPublic as u64));
    let one = b.ext_const(&FEE::one());
    // α¹..α⁵ — the index takes α, lane l takes α^{2+l}.
    let mut powers = Vec::with_capacity(1 + LANES_PER_WORD);
    powers.push(alpha);
    for i in 1..=LANES_PER_WORD {
        let next = b.emul(powers[i - 1], alpha);
        powers.push(next);
    }
    let mut acc: Option<Ext> = None;
    for word in words {
        let idx = b.felt_const(FE::from(word.index as u64));
        let idx_term = b.emul_base(powers[0], idx);
        let mut linear = b.eadd(bus, idx_term);
        for (l, lane) in word.lanes.iter().enumerate() {
            let term = b.emul_base(powers[1 + l], *lane);
            linear = b.eadd(linear, term);
        }
        let fingerprint = b.esub(z, linear);
        let term = b.ediv(one, fingerprint);
        acc = Some(match acc {
            None => term,
            Some(a) => b.eadd(a, term),
        });
    }
    acc.unwrap_or_else(|| b.ext_const(&FEE::zero()))
}

// ============================== the verify leg ============================

/// The arenas one child's leg reads, in DECLARATION ORDER — which is absorb
/// order. The caller declares one set per child, in child order, before emitting
/// any leg, so the node's declaration order is its absorb order end to end.
pub struct LegArenas {
    publics: ArenaId,
    main_roots: ArenaId,
    per_table: Vec<TableArenas>,
}

struct TableArenas {
    aux_root: Option<ArenaId>,
    contribution: Option<ArenaId>,
    composition_root: ArenaId,
    ood_current: ArenaId,
    ood_next: ArenaId,
    parts: ArenaId,
    fri_roots: ArenaId,
    fri_coeffs: ArenaId,
    nonce: Option<ArenaId>,
    legs: TableQueryArenas,
}

/// Declare one child's arenas.
pub fn declare_leg_arenas(b: &mut LfmBuilder, child: &ChildShape<'_>) -> LegArenas {
    let per_root = RootCells::words_per_root(b);
    let publics =
        b.declare_arena((LANES_PER_WORD * HALVES_PER_LANE * child.num_public_words) as u32);
    let main_roots = b.declare_arena(per_root * child.tables.len() as u32);
    let per_table = child
        .tables
        .iter()
        .map(|t| {
            let c = t.challenge;
            TableArenas {
                aux_root: c.has_aux_root.then(|| b.declare_arena(per_root)),
                contribution: c.has_contribution.then(|| b.declare_arena(1)),
                composition_root: b.declare_arena(per_root),
                ood_current: b.declare_arena((c.ood_current_dims.0 * c.ood_current_dims.1) as u32),
                ood_next: b.declare_arena((c.ood_next_dims.0 * c.ood_next_dims.1) as u32),
                parts: b.declare_arena(c.num_parts as u32),
                fri_roots: b.declare_arena(per_root * c.fri.num_committed() as u32),
                fri_coeffs: b.declare_arena(c.fri.num_terminal_coeffs() as u32),
                nonce: (c.grinding_factor > 0).then(|| b.declare_arena(1)),
                legs: super::epoch_verify::declare_table_arenas(b, t.verify),
            }
        })
        .collect();
    LegArenas {
        publics,
        main_roots,
        per_table,
    }
}

/// What a leg hands the node's binding layer.
pub struct LegCells {
    /// The child's published words — index plus canonicity-guarded lanes. This
    /// is the binding legs' entire input, and the only thing a node learns about
    /// a child beyond "its proof verifies".
    pub publics: Vec<HintedPublicWord>,
    /// The child's own shared LogUp pair, exposed for the differential gates.
    pub z_alpha: (Ext, Ext),
}

/// Emit ONE child's complete verification.
///
/// ⚠ ONE TRANSCRIPT PER CHILD. Each child proof was produced against a
/// transcript seeded by its OWN statement, so a node verifying two children runs
/// two independent replays: two Phase A's, two `(z, α)` pairs, two closures. A
/// shared spine that re-indexed every sub-proof over the union of both children
/// would be verifying one proof with `2n` tables — a different statement, and
/// not one either child ever made. (`per_table_census_tests::tenant_node_program`
/// is shaped that way ON PURPOSE, as a census instrument; it is not a template.)
pub fn emit_leg(b: &mut LfmBuilder, child: &ChildShape<'_>, a: &LegArenas) -> LegCells {
    let n = child.tables.len();
    let per_root = RootCells::words_per_root(b);

    // ---- the statement, over the child's claimed published words ----
    let publics = hint_public_words(b, a.publics, child.num_public_words);
    let mut t = TranscriptReplay::new(&[]);
    emit_lfm_statement(
        &mut t,
        child.program_id,
        &publics,
        child.fri_final_poly_log_degree,
    );

    // ---- Phase A: preprocessed roots as AIR-set constants, main roots hinted.
    let main_cells: Vec<RootCells> = (0..n)
        .map(|i| RootCells::hint(b, a.main_roots, per_root * i as u32))
        .collect();
    // ⚠ The DIGEST's felts, not the root's bytes. `replay_phase_a` absorbs
    // through `absorb_root_felts`, which declares the host's 32 bytes on both
    // arms and packs the algebraic arm's four felts into the one digest cell they
    // already are, so the root absorb CANCELS rather than paying a byte
    // regrouping. `byte_halves` is for `program_id`, which is deliberately
    // keccak-over-bytes.
    let main_halves: Vec<Vec<Felt>> = main_cells.iter().map(RootCells::lanes_flat).collect();
    let prep_cells: Vec<Option<RootCells>> = child
        .tables
        .iter()
        .map(|t| t.precomputed_root.map(|c| RootCells::constant(b, c)))
        .collect();
    let phase_a: Vec<PhaseATable> = child
        .tables
        .iter()
        .enumerate()
        .map(|(i, t)| PhaseATable {
            preprocessed_root: t.precomputed_root.map(PhaseAPreprocessed::Constant),
            main_root: &main_halves[i][..],
        })
        .collect();
    let (z, alpha) = replay_phase_a(&mut t, b, &phase_a);

    // ---- one fork per sub-proof, with the full verification legs ----
    let mut contributions: Vec<Ext> = Vec::new();
    for (i, table) in child.tables.iter().enumerate() {
        let c = table.challenge;
        let arenas = &a.per_table[i];
        let aux = arenas.aux_root.map(|id| RootCells::hint(b, id, 0));
        let contribution = arenas.contribution.map(|id| b.hint_word(id, 0).as_ext());
        let composition = RootCells::hint(b, arenas.composition_root, 0);
        let ood_current: Vec<Ext> = (0..(c.ood_current_dims.0 * c.ood_current_dims.1) as u32)
            .map(|k| b.hint_word(arenas.ood_current, k).as_ext())
            .collect();
        let ood_next: Vec<Ext> = (0..(c.ood_next_dims.0 * c.ood_next_dims.1) as u32)
            .map(|k| b.hint_word(arenas.ood_next, k).as_ext())
            .collect();
        let parts: Vec<Ext> = (0..c.num_parts as u32)
            .map(|k| b.hint_word(arenas.parts, k).as_ext())
            .collect();
        let fri_roots: Vec<RootCells> = (0..c.fri.num_committed())
            .map(|k| RootCells::hint(b, arenas.fri_roots, per_root * k as u32))
            .collect();
        let fri_coeffs: Vec<Ext> = (0..c.fri.num_terminal_coeffs() as u32)
            .map(|k| b.hint_word(arenas.fri_coeffs, k).as_ext())
            .collect();
        let nonce = arenas.nonce.map(|id| b.hint_felt(id, 0));
        if let Some(l) = contribution {
            contributions.push(l);
        }

        let mut fork = fork_table(&t, c.index, c.num_tables);
        let absorbs = TableAbsorbs {
            aux_root: aux.as_ref(),
            contribution,
            composition_root: &composition,
            ood_current: &ood_current,
            ood_next: &ood_next,
            parts: &parts,
            fri_roots: &fri_roots,
            fri_coeffs: &fri_coeffs,
            nonce,
        };
        let ch = super::epoch::emit_table_challenges(b, &mut fork, c, &absorbs);
        // ★ THE SEAM: `absorbs` is passed on by REFERENCE rather than rebuilt,
        // so there is no second reading of the proof for a leg to disagree with
        // the transcript about.
        super::epoch_verify::emit_table_verification(
            b,
            table.verify,
            table.analysis,
            &ch,
            &absorbs,
            &super::epoch_verify::TableInputs {
                precomputed_root: prep_cells[i].as_ref(),
                main_root: &main_cells[i],
                rap_challenges: &[z, alpha],
            },
            &arenas.legs,
        );
    }

    // ---- the closure, against the PUBLIC balance ----
    //
    // Every other LFM bus balances to zero internally; `LfmPublic` is the one
    // whose target is the claimed words, which is what makes this the binding
    // between "the proof verifies" and "it published THESE words".
    let target = emit_public_balance(b, &publics, z, alpha);
    let shape = super::logup::LogUpShape {
        num_contributing_tables: contributions.len(),
        num_output_bytes: 0,
    };
    super::logup::emit_bus_closure(b, &shape, &contributions, target);

    LegCells {
        publics,
        z_alpha: (z, alpha),
    }
}

// ============================ the binding legs ============================

/// Where each field of the block-binding schema sits in a child's published
/// words.
///
/// The order is the emitter's, and it is one order: `epoch_tests::epoch_program`
/// publishes the pair, the attestation id, then this schema, and
/// `epoch_tests::schema_words` is the same arithmetic as [`Self::schema_words`].
/// A node indexes by these accessors, so a field inserted on one side and not
/// the other re-binds every field below it — which is why
/// [`Self::assert_covers`] is called at assembly time rather than trusted.
pub struct SchemaLayout {
    pub num_reg: usize,
    pub out_halves: usize,
    /// Lanes the L2G root occupies — four on an algebraic hash, eight on a byte
    /// one. Read from `proof_arena::lanes_per_root` rather than spelled, because
    /// a reader that spells `8` walks an algebraic schema at twice the stride and
    /// lands on the wrong field instead of out of bounds.
    pub root_lanes: usize,
}

impl SchemaLayout {
    /// The words published before the schema: the shared pair and the two
    /// attestation-id words.
    pub const HEAD: usize = 4;

    pub fn new(out_halves: usize) -> Self {
        Self {
            num_reg: crate::tables::register::NUM_REGISTER_ADDRESSES,
            out_halves,
            root_lanes: super::proof_arena::lanes_per_root(),
        }
    }

    pub fn schema_words(&self) -> usize {
        2 * self.num_reg + 2 + self.out_halves + self.root_lanes
    }

    /// Total words a child publishes under `Publishes::Aggregation` — the head,
    /// the schema and the closure's bus total.
    pub fn total(&self) -> usize {
        Self::HEAD + self.schema_words() + 1
    }

    /// Pin the layout against what the child actually publishes.
    ///
    /// A level confusion — a layout built for one child applied to another — is
    /// a loud failure here rather than a silent mis-binding fifty asserts later.
    pub fn assert_covers(&self, num_public_words: usize) {
        assert_eq!(
            self.total(),
            num_public_words,
            "the layout must cover the child's published words exactly \
             (num_reg={}, out_halves={}, root_lanes={})",
            self.num_reg,
            self.out_halves,
            self.root_lanes,
        );
    }

    pub fn id(&self, half: usize) -> usize {
        2 + half
    }
    pub fn reg_init(&self, r: usize) -> usize {
        Self::HEAD + r
    }
    pub fn reg_fini(&self, r: usize) -> usize {
        Self::HEAD + self.num_reg + r
    }
    pub fn label(&self, half: usize) -> usize {
        Self::HEAD + 2 * self.num_reg + half
    }
    pub fn out_half(&self, i: usize) -> usize {
        Self::HEAD + 2 * self.num_reg + 2 + i
    }
    pub fn l2g_lane(&self, lane: usize) -> usize {
        Self::HEAD + 2 * self.num_reg + 2 + self.out_halves + lane
    }
}

/// Assert two hinted public words carry the same value, lane by lane.
pub fn assert_words_equal(b: &mut LfmBuilder, x: &HintedPublicWord, y: &HintedPublicWord) {
    for (xl, yl) in x.lanes.iter().zip(&y.lanes) {
        let xe = xl.as_ext();
        let ye = yl.as_ext();
        b.assert_eq_ext(xe, ye);
    }
}

/// Assert a hinted public word's base value equals a program constant — lanes
/// `1..4` must be zero, which is what a BASE publish looks like.
pub fn assert_word_is_const(b: &mut LfmBuilder, x: &HintedPublicWord, v: u64) {
    let c = b.ext_const(&FEE::from(v));
    let x0 = x.lanes[0].as_ext();
    b.assert_eq_ext(x0, c);
    let zero = b.ext_const(&FEE::zero());
    for lane in &x.lanes[1..] {
        let le = lane.as_ext();
        b.assert_eq_ext(le, zero);
    }
}

/// The cross-child binding legs: one shared attestation id across every child,
/// each child's register fini vector equal to the next child's init vector, and
/// each child's epoch label pinned to its chain position as an emit-time
/// constant.
///
/// These are CHECKS on published words, never trusts. Verifying two children
/// says nothing about their relationship; this is what makes a tree of proofs a
/// statement about one execution rather than about several.
pub fn emit_chain_bindings(
    b: &mut LfmBuilder,
    legs: &[LegCells],
    layouts: &[SchemaLayout],
    labels: &[u64],
) {
    assert_eq!(legs.len(), layouts.len(), "one layout per child");
    assert_eq!(legs.len(), labels.len(), "one chain position per child");

    // ---- one attestation id answers for every child.
    for k in 1..legs.len() {
        for half in 0..2 {
            assert_words_equal(
                b,
                &legs[0].publics[layouts[0].id(half)],
                &legs[k].publics[layouts[k].id(half)],
            );
        }
    }
    // ---- the register chain, across every seam.
    for k in 0..legs.len().saturating_sub(1) {
        for r in 0..layouts[k].num_reg {
            assert_words_equal(
                b,
                &legs[k].publics[layouts[k].reg_fini(r)],
                &legs[k + 1].publics[layouts[k + 1].reg_init(r)],
            );
        }
    }
    // ---- each label pinned to its position, as a constant of THIS program.
    for (k, &label) in labels.iter().enumerate() {
        assert_word_is_const(
            b,
            &legs[k].publics[layouts[k].label(0)],
            label & 0xFFFF_FFFF,
        );
        assert_word_is_const(b, &legs[k].publics[layouts[k].label(1)], label >> 32);
    }
}

/// Assert child `k`'s published L2G re-commit root equals the root the global
/// proof's verifier published for epoch `k`.
///
/// The in-VM half of the root-equality binding. In a TREE the two sides are not
/// generally in the same node — the global wrap rides at the ROOT — so this is
/// emitted where both are in scope and nowhere else.
pub fn assert_l2g_roots_equal(
    b: &mut LfmBuilder,
    epoch: &LegCells,
    epoch_layout: &SchemaLayout,
    global: &LegCells,
    global_first_root_word: usize,
    epoch_index: usize,
) {
    let lanes = epoch_layout.root_lanes;
    for lane in 0..lanes {
        assert_words_equal(
            b,
            &epoch.publics[epoch_layout.l2g_lane(lane)],
            &global.publics[global_first_root_word + lanes * epoch_index + lane],
        );
    }
}

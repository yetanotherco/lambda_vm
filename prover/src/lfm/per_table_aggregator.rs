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
/// words — for a WRAP child and for a NODE child alike.
///
/// The two differ in their head and in two fields, and the difference is real
/// rather than incidental, so it is a constructor apiece rather than a flag:
///
/// | | wrap child | node child |
/// |---|---|---|
/// | head | `z, α, id₀, id₁` | `id₀, id₁` — a node has FAN_IN Phase A's and no single pair |
/// | labels | one epoch label | the FIRST and LAST label of its subtree |
/// | L2G | the epoch's re-commit root, as lanes | the subtree's FOLDED digest, as cells |
/// | tail | the closure's bus total | nothing |
///
/// A node has no single `(z, α)` because it replays one transcript per child,
/// and no single epoch label because it covers a RANGE — publishing the range's
/// two ends is what lets a parent pin them to constants and get contiguity
/// across siblings for free.
///
/// ⚠ `l2g_words` is a WORD count, not a lane count, and the two constructors
/// disagree on purpose: a wrap publishes `RootCells::lanes_flat` (four lanes per
/// root word) while a node publishes the fold's digest CELLS. The codebase has
/// been bitten by lanes-versus-words twice; the field is named for the unit it
/// actually is.
pub struct SchemaLayout {
    /// Where the two attestation-id words start.
    pub id_at: usize,
    /// Words published before the schema run.
    pub head: usize,
    pub num_reg: usize,
    /// Label words: two for a wrap (one label, lo/hi), four for a node.
    pub label_words: usize,
    pub out_halves: usize,
    pub l2g_words: usize,
    /// Words published after the schema run.
    pub tail: usize,
}

impl SchemaLayout {
    /// The layout of an epoch WRAP published under `Publishes::Aggregation`.
    pub fn wrap(out_halves: usize) -> Self {
        Self {
            id_at: 2,
            head: 4,
            num_reg: crate::tables::register::NUM_REGISTER_ADDRESSES,
            label_words: 2,
            out_halves,
            l2g_words: super::proof_arena::lanes_per_root(),
            tail: 1,
        }
    }

    /// The layout of an aggregation NODE, at any level.
    pub fn node(out_halves: usize) -> Self {
        Self {
            id_at: 0,
            head: 2,
            num_reg: crate::tables::register::NUM_REGISTER_ADDRESSES,
            label_words: 4,
            out_halves,
            // ★ THE SAME SHAPE A WRAP PUBLISHES ITS ROOT IN — `lanes_per_root`
            // base words, one lane each — not the fold's digest CELLS.
            //
            // Publishing cells would have been one word instead of four, and it
            // would have made a node child and a wrap child structurally
            // different to read: `emit_node_publishes` takes `lanes[0]` of each
            // published l2g word, which is right for a wrap's lane-per-word
            // layout and silently wrong for a four-lane digest word — it would
            // hand ONE felt to `digest_from_lanes` where four are required.
            // Building the inner-node arm is what surfaced that; the asymmetry
            // is removed here rather than parameterised around.
            l2g_words: super::proof_arena::lanes_per_root(),
            tail: 0,
        }
    }

    pub fn schema_words(&self) -> usize {
        2 * self.num_reg + self.label_words + self.out_halves + self.l2g_words
    }

    /// Every word the child publishes.
    pub fn total(&self) -> usize {
        self.head + self.schema_words() + self.tail
    }

    /// Pin the layout against what the child actually publishes.
    ///
    /// A level confusion — a wrap layout applied to a node, or a layout built
    /// for one block applied to another — is a loud failure here rather than a
    /// silent mis-binding fifty asserts later.
    pub fn assert_covers(&self, num_public_words: usize) {
        assert_eq!(
            self.total(),
            num_public_words,
            "the layout must cover the child's published words exactly \
             (head={}, num_reg={}, label_words={}, out_halves={}, l2g_words={}, tail={})",
            self.head,
            self.num_reg,
            self.label_words,
            self.out_halves,
            self.l2g_words,
            self.tail,
        );
    }

    pub fn id(&self, half: usize) -> usize {
        self.id_at + half
    }
    pub fn reg_init(&self, r: usize) -> usize {
        self.head + r
    }
    pub fn reg_fini(&self, r: usize) -> usize {
        self.head + self.num_reg + r
    }
    pub fn label(&self, i: usize) -> usize {
        self.head + 2 * self.num_reg + i
    }
    pub fn out_half(&self, i: usize) -> usize {
        self.head + 2 * self.num_reg + self.label_words + i
    }
    pub fn l2g_word(&self, w: usize) -> usize {
        self.head + 2 * self.num_reg + self.label_words + self.out_halves + w
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
    labels: &[&[u64]],
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
    //
    // A wrap child carries ONE label; a node child carries the first and last of
    // its subtree. Pinning every one of them to a constant is what makes
    // contiguity across siblings free: the emitter knows the true label
    // sequence, so a child covering the wrong range cannot satisfy the pins.
    for (k, child_labels) in labels.iter().enumerate() {
        assert_eq!(
            2 * child_labels.len(),
            layouts[k].label_words,
            "child {k} publishes {} label words but {} labels were given",
            layouts[k].label_words,
            child_labels.len()
        );
        for (i, &label) in child_labels.iter().enumerate() {
            assert_word_is_const(
                b,
                &legs[k].publics[layouts[k].label(2 * i)],
                label & 0xFFFF_FFFF,
            );
            assert_word_is_const(
                b,
                &legs[k].publics[layouts[k].label(2 * i + 1)],
                label >> 32,
            );
        }
    }
}

// ============================ the aggregation node ========================

/// Children per node — the tree's arity.
///
/// A DEFAULT, not an assumption: every emitter below takes a slice, so the
/// arity is whatever the caller passes and nothing here depends on this value.
/// It exists so the tree builder has one place to change.
///
/// Two is the brief's working default and three is COORD's tie-break, on the
/// grounds that over ten epochs it is 5 distinct programs / 7 proofs / 3 levels
/// against two's 6 / 11 / 4. The measured host peak decides; until it has, the
/// conservative value stands.
pub const FAN_IN: usize = 2;

/// A digest rebuilt from the lanes a child PUBLISHED for it.
///
/// The inverse of `RootCells::lanes_flat`, and correct on both arms — four
/// lanes pack into one word, and a root is `words_per_root` of them.
///
/// ⚠ Deliberately not `RootCells::from_halves`, which asserts EIGHT halves and
/// is byte-arm-only by construction. Handing it an algebraic root's four felts
/// fails the assert; handing a byte root's eight halves to a caller expecting
/// felts would hash four values as if they were eight, silently.
pub fn digest_from_lanes(b: &mut LfmBuilder, lanes: &[Felt]) -> super::edsl::WrapDigest {
    let words = super::proof_arena::words_per_root();
    assert_eq!(
        lanes.len(),
        LANES_PER_WORD * words,
        "a published root is four lanes per root word"
    );
    let cells: Vec<super::builder::Cell> = lanes
        .chunks(LANES_PER_WORD)
        .map(|c| b.pack_word([c[0], c[1], c[2], c[3]]))
        .collect();
    match cells.len() {
        1 => super::edsl::WrapDigest::from_cell(cells[0]),
        2 => super::edsl::WrapDigest::from_pair(cells[0], cells[1]),
        n => unreachable!("a root is one or two words, got {n}"),
    }
}

/// Fold a subtree's L2G re-commit roots into ONE digest — a left fold of the
/// production hash's two-to-one compression, identity on a singleton.
///
/// # Why a fold and not a list
///
/// The batched aggregator compared each epoch's published L2G root against the
/// global proof's re-commit root for that epoch, and could do it as a local
/// assert because all six legs were in ONE program. A tree splits them: the
/// epoch wraps sit in leaf nodes and the global wrap rides at the ROOT, so the
/// compare must be deferred to their common ancestor.
///
/// If each node re-published its subtree's roots as a LIST, the node schema
/// would grow with the subtree and every level would have a different published
/// width — which is what makes the parent's leg shape depend on the block's
/// epoch count. Folding them keeps the node schema FIXED SIZE at every level,
/// and the root recomputes the same fold over the global wrap's published roots
/// and compares one digest.
///
/// ⛔ **THE FOLD COMPOUNDS: IT IS TREE-SHAPED, NOT FLAT.** An earlier version of
/// this paragraph said only *"left fold, in tree order"*, which is true of ONE
/// node and actively misleading at the ROOT — which is exactly where it is read.
///
/// A node folds its children's PUBLISHED digests, and a node child's published
/// digest is already a fold. So a level-1 node publishes `H(r0, r1)` and a
/// level-2 node publishes `H(H(r0,r1), H(r2,r3))` — **not** the flat left fold
/// `H(H(H(r0,r1),r2),r3)`. `hash_pair` is a two-to-one compression and is not
/// associative, so those are different digests.
///
/// ⇒ The global wrap publishes the FLAT list of per-epoch roots. A root that
/// folds them left to right computes a digest an honest prover never produced,
/// and the failure lands on **completeness, not soundness**: correct code,
/// correct inputs, honest prover, wrong answer — the worst kind to diagnose from
/// a failing prove. The root must group them exactly as the interior did, level
/// by level, which is what `block_root::FoldShape` exists to make unavoidable.
///
/// Within ONE node the rule is: left fold, `hash_pair(acc, next)`, in child
/// order; a single child folds to itself.
pub fn fold_l2g(
    b: &mut LfmBuilder,
    digests: &[super::edsl::WrapDigest],
) -> super::edsl::WrapDigest {
    assert!(!digests.is_empty(), "a subtree covers at least one epoch");
    let hash = b.wrap_hash();
    let mut acc = digests[0];
    for next in &digests[1..] {
        acc = hash.hash_pair(b, acc, *next);
    }
    acc
}

/// What a node publishes, so that its parent binds it exactly as it binds a
/// wrap. See [`SchemaLayout::node`] for the layout this fills.
pub struct NodePublishes<'a> {
    /// The children's legs, in chain order.
    pub legs: &'a [LegCells],
    /// One layout per child.
    pub layouts: &'a [SchemaLayout],
    /// The first and last epoch label of the subtree — emit-time constants.
    pub label_range: (u64, u64),
}

/// Emit a node's published words: the shared attestation id, the first child's
/// register INIT vector, the last child's register FINI vector, the subtree's
/// first and last epoch labels, the last child's output halves, and the folded
/// L2G digest.
///
/// Every value is republished in the SAME form the wrap published it, so a
/// parent's `assert_words_equal` reads a node and a wrap alike: the id as a
/// four-lane digest word, the register / label / output items as base words.
pub fn emit_node_publishes(b: &mut LfmBuilder, p: &NodePublishes<'_>) {
    let NodePublishes {
        legs,
        layouts,
        label_range,
    } = *p;
    assert_eq!(legs.len(), layouts.len(), "one layout per child");
    assert!(!legs.is_empty(), "a node has children");
    let first = &legs[0];
    let last = legs.last().expect("nonempty");
    let l_first = &layouts[0];
    let l_last = layouts.last().expect("nonempty");

    // ---- the attestation id, as the four-lane word the wrap published.
    for half in 0..2 {
        let lanes = &first.publics[l_first.id(half)].lanes;
        let word = b.pack_word([lanes[0], lanes[1], lanes[2], lanes[3]]);
        b.public(word);
    }
    // ---- the chain's two ends.
    for r in 0..l_first.num_reg {
        b.public(first.publics[l_first.reg_init(r)].lanes[0].as_cell());
    }
    for r in 0..l_last.num_reg {
        b.public(last.publics[l_last.reg_fini(r)].lanes[0].as_cell());
    }
    // ---- the label RANGE, as constants of this program. Publishing the ends
    // rather than the list is what keeps the schema fixed-size; the parent pins
    // both to constants and gets contiguity across siblings for free.
    for label in [label_range.0, label_range.1] {
        let lo = b.felt_const(FE::from(label & 0xFFFF_FFFF));
        b.public(lo.as_cell());
        let hi = b.felt_const(FE::from(label >> 32));
        b.public(hi.as_cell());
    }
    // ---- the last child's output halves — the block's output, carried up.
    for i in 0..l_last.out_halves {
        b.public(last.publics[l_last.out_half(i)].lanes[0].as_cell());
    }
    // ---- the folded L2G digest.
    let digests: Vec<super::edsl::WrapDigest> = legs
        .iter()
        .zip(layouts)
        .map(|(leg, layout)| {
            let lanes: Vec<Felt> = (0..layout.l2g_words)
                .map(|w| leg.publics[layout.l2g_word(w)].lanes[0])
                .collect();
            digest_from_lanes(b, &lanes)
        })
        .collect();
    let folded = fold_l2g(b, &digests);
    // Unpacked to lanes, so a node's L2G item reads exactly like a wrap's.
    for cell in folded.cells() {
        for lane in b.unpack(*cell) {
            b.public(lane.as_cell());
        }
    }
}

/// One aggregation node, end to end: verify every child, bind them, publish the
/// node's own schema.
///
/// The SAME function serves every level — a child is a plain per-table
/// `MultiProof` whether a wrap or another node produced it, and the only thing
/// that changes is the children's shapes and layouts. What does not carry across
/// levels is the node's IDENTITY: the legs absorb their children's `program_id`
/// as emit-time constants, so each level compiles to its own program.
pub struct NodeInputs<'a> {
    pub children: &'a [ChildShape<'a>],
    pub layouts: &'a [SchemaLayout],
    /// Per child, the epoch labels it must carry: one for a wrap child, the two
    /// ends of its subtree for a node child.
    pub labels: &'a [&'a [u64]],
    /// The first and last epoch label this node's subtree covers.
    pub label_range: (u64, u64),
    /// Which words this node publishes.
    pub publishes: NodePublishSet,
}

/// Which words a node publishes — the node-level counterpart of
/// `epoch_tests::Publishes`, and it exists for a different reason.
///
/// A node under [`NodePublishSet::Aggregation`] publishes no challenges at all,
/// which leaves its legs with no differential surface: the only evidence that a
/// leg derived its CHILD's challenges is that the node executes, since a leg on
/// different challenges cannot authenticate the child's walks. That is
/// implication rather than a value comparison, and it is weaker than what every
/// other emitted verifier in this crate is held to — the epoch wrap and the
/// global leg both publish their pair and differential it against production's
/// own replay.
///
/// [`NodePublishSet::Diagnostic`] restores that surface by publishing each
/// child's `(z, α)` AFTER the schema, so the schema's own indices do not move
/// and `SchemaLayout::node` reads both variants' heads identically. It is a gate
/// shape, never a child: nothing verifies a diagnostic node.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodePublishSet {
    /// The node schema and nothing else — what a node that will be VERIFIED
    /// publishes.
    Aggregation,
    /// The schema, then each child's `(z, α)` in child order — the differential
    /// surface.
    Diagnostic,
}

/// Declare every arena the node reads, in absorb order, then emit it.
pub fn emit_node(b: &mut LfmBuilder, inputs: &NodeInputs<'_>) {
    let NodeInputs {
        children,
        layouts,
        labels,
        label_range,
        publishes,
    } = *inputs;
    assert!(!children.is_empty(), "a node verifies at least one child");
    assert_eq!(children.len(), layouts.len(), "one layout per child");
    assert_eq!(children.len(), labels.len(), "one label list per child");
    for (child, layout) in children.iter().zip(layouts) {
        layout.assert_covers(child.num_public_words);
    }

    // Declaration order IS absorb order, end to end: every child's arenas are
    // declared before any leg is emitted.
    let arenas: Vec<LegArenas> = children
        .iter()
        .map(|child| declare_leg_arenas(b, child))
        .collect();
    let legs: Vec<LegCells> = children
        .iter()
        .zip(&arenas)
        .map(|(child, a)| emit_leg(b, child, a))
        .collect();
    emit_chain_bindings(b, &legs, layouts, labels);
    emit_node_publishes(
        b,
        &NodePublishes {
            legs: &legs,
            layouts,
            label_range,
        },
    );
    // The differential surface, AFTER the schema so no schema index moves.
    if publishes == NodePublishSet::Diagnostic {
        for leg in &legs {
            b.public(leg.z_alpha.0.as_cell());
            b.public(leg.z_alpha.1.as_cell());
        }
    }
}

// ============================ the tree's shape ============================

/// One level of the tree: how many proofs it consumes and how they group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Level {
    /// Children per node, in order. Every entry is `1..=fan_in`.
    pub arities: Vec<usize>,
}

impl Level {
    /// Proofs this level produces — one per node.
    pub fn nodes(&self) -> usize {
        self.arities.len()
    }
}

/// The whole tree's shape, derived from the epoch count and an arity.
///
/// # The leftover rule: WRAP, do not carry
///
/// A level with an odd count leaves one proof over. Two ways, and the choice
/// changes the PROGRAM SET rather than the code:
///
/// - **carry** it up unwrapped ⇒ its parent has children from two different
///   levels, so two different `program_id`s and two different shapes. A node
///   embeds each child's id as an emit-time constant, so every such parent is a
///   distinct program — and the VERIFIER must emit each one too.
/// - **wrap** it in an arity-1 node ⇒ one extra proof per odd level, and every
///   parent's children stay homogeneous: one program per `(level, arity)`.
///
/// ⇒ This wraps. Program-set growth lands on both sides of the protocol and is
/// combinatorial; the extra proofs are bounded and countable (three at 36
/// epochs, fan-in 2). ⚠ Revisit if a node prove is ever measured to dominate
/// program emission — the trade is real, and it is recorded rather than assumed.
///
/// ⓘ The ROOT is not described here. It takes `fan_in + 1` children — the global
/// wrap is the extra — performs the L2G compare and the attestation fold, and
/// publishes the block artifact's schema rather than the node schema. This
/// function describes the interior.
pub fn tree_shape(epochs: usize, fan_in: usize) -> Vec<Level> {
    assert!(epochs >= 1, "a block has at least one epoch");
    assert!(fan_in >= 2, "a tree needs an arity of at least two");
    let mut levels = Vec::new();
    let mut n = epochs;
    while n > 1 {
        let mut arities = Vec::with_capacity(n.div_ceil(fan_in));
        let mut left = n;
        while left > 0 {
            let take = left.min(fan_in);
            arities.push(take);
            left -= take;
        }
        n = arities.len();
        levels.push(Level { arities });
    }
    levels
}

/// Total aggregator proofs the interior costs — one per node, every level.
pub fn tree_node_count(shape: &[Level]) -> usize {
    shape.iter().map(Level::nodes).sum()
}

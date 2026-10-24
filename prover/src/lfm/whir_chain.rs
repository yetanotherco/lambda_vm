//! `whir_chain::verify_weighted` as a machine leg — the WHIR chain, assembled.
//!
//! This is the whole of `crypto/multilinear/src/whir_chain.rs:938-1105` and it
//! is where the wrap's cost lives: about ninety per cent of a chain's
//! permutations are the query phase's openings, and a chain is run once per
//! stacked polynomial.
//!
//! Every brick is already gated on its own against the host function it
//! replaces — the grind against `is_valid_nonce`
//! ([`super::whir_transcript::emit_grind_check`]), the degree-2 sumcheck
//! against `verify_rounds` ([`super::whir_poly`]), `eq` against `eq_eval`, the
//! opening against `verify_opening` ([`super::whir_open`]), the fold against
//! `fold_coset` ([`super::whir_fold`]), and the transcript against the K1–K6
//! vectors. What this module adds is the ORDER, and the order is exactly what
//! its own gate checks: the host's challenge stream, observed through a
//! recording transcript, against the stream the machine derives from the
//! sponge.
//!
//! # The round structure, and the three places a nonce is spent
//!
//! Per round: `check_grind(folding)`; `k_r` sumcheck rounds at degree two, each
//! absorbing its two evaluations and drawing one challenge; then the domain is
//! squared `k_r` times, which is emit-time and free. A round that has a
//! successor absorbs its root, draws `z0`, refuses an in-domain `z0`, absorbs
//! `y0`, spends the OOD nonce, draws `gamma`, folds `gamma·y0` into the claim,
//! spends the QUERY nonce, and opens `Q` queries against both trees. The last
//! round absorbs `final_value` instead, spends only the query nonce, and opens
//! `Q` queries against one tree — `verify_final` has no successor
//! (`whir_chain.rs:1125-1141`). That is why a uniformly ground chain has
//! `3R − 1` grinds.
//!
//! A grind of zero bits is not emitted at all. Which nonces the ARENA carries
//! is the config's [`NonceLayout`]: three a round under `Three` (today's
//! format, the unspent ones hinted and never read), only the spent ones under
//! `Spent`. P2 grinds before the queries alone (`GrindBits::query_only`) under
//! `Spent`, so a round carries ONE nonce word and an unspent nonce has no word
//! to be set in: [`ChainShape::carries`].
//!
//! # ★ The out-of-domain squarings are SHARED, and exactly
//!
//! `require_out_of_domain` squares `z0` once per bit of the successor domain
//! (`:93-96`) and `ood_point` keeps `z0^(2^j)` for `j < num_vars − bound`
//! (`:72-81`). Those are the same chain of squarings, and the second is a
//! PREFIX of the first: `bound_r` variables are bound after round `r`, the
//! successor domain is `2^(num_vars + 2 − bound_r)`, so the domain check needs
//! exactly two more squarings than the point does — always, at every round and
//! every stack height. One chain of `D_{r+1}` multiplies serves both. Computing
//! them apart would cost `2·D_{r+1} − 2`, which at `S = 25` is 156 multiplies a
//! chain against 78.
//!
//! # ★ The successor slot is a MUX, and it is not free
//!
//! `leaf_and_slot` splits the query into a leaf index and a slot
//! (`whir_commit.rs:119-121`); both bounds are powers of two, so the split is a
//! partition of the `BitDec`'s bits and costs nothing. But `nxt.values[slot]`
//! (`whir_round.rs:160`) then selects one of `2^k` extension values by the high
//! bits, which is `2^k − 1` `Select` rows a query a round. At `k = 4`, `Q =
//! 112` and six successor rounds that is 8,400 rows a chain. The slot is always
//! in range — `slot < 2^{k_{r+1}}` follows from `q < 2^{D_{r+1}}` — so there is
//! no out-of-range branch to emit, which is the host's `QueryOutOfRange` made
//! unreachable rather than checked.
//!
//! # ⚠ The tail divides, and must
//!
//! The host computes `required = claim · weight⁻¹` and rejects a zero weight
//! with `DegenerateEvaluationPoint` (`:1095-1102`). Asserting
//! `final_value · weight == claim` instead would be one row cheaper and
//! strictly WEAKER: it accepts every `final_value` when `weight` and `claim`
//! are both zero, where the host rejects. So the tail inverts —
//! `div(one, weight)` has no satisfying assignment at zero, which IS the
//! refusal — and then compares.
//!
//! # What is emit-time, and therefore has no runtime check
//!
//! The host's shape rejections (`proof.rounds.len() != schedule.len()`,
//! `round.sumcheck.len() != k`, the base-versus-extension opening variant at
//! `:983`, `openings.current.len() != num_queries`) are all properties of the
//! PROGRAM here: the emitter reads a fixed number of wires at fixed offsets, so
//! a proof of another shape has no way to be supplied. That is the
//! `epoch_verify.rs:171-179` idiom — the absence of a second value rather than
//! an assert somebody could forget.

use multilinear::stacked_eval::index_chains;
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    ChainConfig, ChainProof, ChainRound, GroupFold, LeafOrder, MAX_INDEX_POLYS, NonceLayout,
    RoundOpenings, index_vars,
};

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};
use stark::config::Commitment;

use super::algebraic_commit::commitment_to_digest;
use super::builder::{Bit, Cell, Ext, Felt, LfmBuilder};
use super::edsl::WrapDigest;
use super::whir_fold::{FoldEmission, RoundFold, fold_rows_for, lean_prepare_rows};
use super::whir_open::{
    BlockValues, CapCells, Siblings, TreeAuth, cap_check_perms, cap_check_rows,
    verify_opening_perms_capped, verify_opening_rows_capped,
};
use super::whir_poly::{
    emit_eq_eval, emit_sumcheck_round, eq_eval_rows_again, sumcheck_round_rows,
};
use super::whir_transcript::{
    COORDINATES_PER_EXT, SpongeEntry, SpongeHash, SpongeSchedule, WhirTranscript, emit_grind_check,
};
use super::word::{LfmWord, ext_word};

/// The degree the weight raises the plain `f` term to (`whir_chain.rs:991`).
pub(crate) const SUMCHECK_DEGREE: usize = 2;

/// One query's opening against one tree: the block, and its authentication
/// path.
pub struct QueryOpening<'a> {
    pub values: BlockValues<'a>,
    pub siblings: Siblings<'a>,
}

/// A round's nonce wires, one per slot the arena carries
/// ([`ChainShape::carries`]). A slot is read only when its grind spends bits,
/// and every such slot is carried, so `None` is a slot nothing reads. Under
/// [`NonceLayout::Three`] all three are hinted and the unspent ones (zero bits,
/// and the out-of-domain one on the last round) are not read.
#[derive(Clone, Copy)]
pub struct RoundNonces {
    pub folding: Option<Felt>,
    pub ood: Option<Felt>,
    pub query: Option<Felt>,
}

/// One round of the chain, as wires.
pub struct ChainRoundWires<'a> {
    /// `k_r` sumcheck rounds, each carrying `SUMCHECK_DEGREE` evaluations.
    pub sumcheck: &'a [Vec<Ext>],
    /// The successor's root as a word. `None` on the last round, where the
    /// folded message is a constant and `final_value` is sent instead.
    pub next_root: Option<Cell>,
    /// The successor's value at the out-of-domain point. `None` on the last
    /// round.
    pub ood_value: Option<Ext>,
    pub nonces: RoundNonces,
    /// Per query: the block of the current codeword that folds onto the query.
    /// A grouped round 0 holds one block a tree, query-major
    /// (`num_queries · trees`, [`ChainShape::trees_at`]).
    pub current: &'a [QueryOpening<'a>],
    /// Per query: the successor block holding the folded value. Empty on the
    /// last round.
    pub next: &'a [QueryOpening<'a>],
    /// The current tree's Merkle cap, when this round OWNS it: round 0 with
    /// `caps[0] > 0` — a grouped round 0's trees' caps, in tree order. Empty
    /// otherwise (W1; a later round's current tree was authenticated as the
    /// round before's successor).
    pub current_cap: &'a [WrapDigest],
    /// The successor tree's Merkle cap, when it has one (`caps[r + 1] > 0`).
    pub next_cap: &'a [WrapDigest],
}

/// The shape of one chain: everything the closed forms below are a function of.
///
/// Built from the same `ChainConfig::schedule` the host runs, so a change to
/// the schedule moves both sides together rather than only one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainShape {
    /// How the recursion leaf checks this chain's queries
    /// ([`LeafQuery`]): [`LeafQuery::V1`] for every chain but a block leaf's
    /// under its format's `leaf_query`.
    pub query: LeafQuery,
    /// Variables folded in each round.
    pub schedule: Vec<usize>,
    /// `log2` of the domain each round starts at: `D_0 = num_vars +
    /// log_blowup`, then `D_{r+1} = D_r − k_r`.
    pub domain_log: Vec<usize>,
    pub num_vars: usize,
    pub num_queries: usize,
    pub grind: (usize, usize, usize),
    /// Each tree's Merkle cap height (W1): tree `r` is round `r`'s current
    /// tree. From the same `ChainConfig::tree_caps` the host prover and
    /// verifier use; all zero at the default format.
    pub caps: Vec<usize>,
    /// Which nonces a round's arena words carry (P2): the config's own.
    pub nonces: NonceLayout,
    /// Children per Merkle node: 2, or 4 for a Poseidon1 base's trees, whose
    /// caps then count 4-ary levels (`ChainConfig::tree_caps_at`).
    pub arity: usize,
    /// How a stack's polynomials are cut into chains (D-ONECHAIN, W-14): the
    /// config's own, so the emitter, the arena and the plan split a stack as
    /// the host does ([`Self::chains`]).
    pub group: GroupFold,
    /// The polynomials this chain opens: one, or a run of up to
    /// `MAX_INDEX_POLYS` under [`GroupFold::Indexed`] ([`Self::run`]), whose
    /// round 0 opens one tree each at one position set and binds `m =
    /// ⌈log2 trees⌉` index variables after its `k0` data ones.
    pub trees: usize,
    /// Where the round-0 trees put their leaves: the config's own, so round
    /// 0's walks authenticate the position the host prover placed the block
    /// at ([`walk_bits`]).
    pub leaves: LeafOrder,
}

/// How a block's recursion leaf checks a chain's queries: a verifier
/// constant (`BlockFormat::leaf_query`) the prover's arena and the leaf
/// program both follow, never read from a proof.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LeafQuery {
    /// Every round's queries opened and folded one by one. The default.
    #[default]
    V1,
    /// As [`Self::V1`], except round 0's base values arrive four to a hinted
    /// word ([`ChainShape::round0_packed`]) and the chain's tail rounds
    /// ([`ChainShape::tail_round`]): a round whose current tree lies wholly in
    /// its cap, with at least [`TAIL_QUERIES_A_LEAF`] queries a leaf, is
    /// checked at EVERY leaf once, each leaf's block hashed against its cap
    /// node, folded at its constant position and compared with the successor
    /// block (or the final value). A superset of the sampled positions, so a
    /// leaf accepts only what the host verifier's query checks for those
    /// rounds accept. The rounds' query draws stay in the transcript, their
    /// bits unused.
    Lean,
}

/// Queries a tail tree's leaf ([`ChainShape::tail_round`]) needs at least.
///
/// The arena takes a tail tree's every block from the proof, which opens the
/// tree at `2Q` uniform positions (the round before's successor openings and
/// the round's own). A tree of `L` leaves misses one with probability at most
/// `L · (1 − 1/L)^(2Q) ≤ L · e^(−2Q/L)`; at `Q ≥ 24 L` that is `L · e^(−48)`,
/// below `2^−63` for every tree of at most 64 leaves. A miss leaves an honest
/// proof without an arena (completeness, never soundness).
pub const TAIL_QUERIES_A_LEAF: usize = 24;

impl ChainShape {
    pub fn new(config: &ChainConfig, num_vars: usize) -> Self {
        Self::new_at(config, num_vars, 2)
    }

    /// [`Self::new`] over trees of `arity` children per node: the hash's
    /// arity, as the host's prover and verifier take it.
    pub fn new_at(config: &ChainConfig, num_vars: usize, arity: usize) -> Self {
        let schedule = config.schedule(num_vars);
        let mut domain_log = Vec::with_capacity(schedule.len());
        let mut d = num_vars + config.log_blowup;
        for &k in &schedule {
            domain_log.push(d);
            d -= k;
        }
        let caps = config.tree_caps_at(num_vars, arity);
        debug_assert_eq!(caps.len(), schedule.len(), "one cap height per tree");
        Self {
            query: LeafQuery::V1,
            schedule,
            domain_log,
            num_vars,
            num_queries: config.num_queries,
            caps,
            grind: (
                config.grind.folding as usize,
                config.grind.ood as usize,
                config.grind.query as usize,
            ),
            nonces: config.format.nonces,
            arity,
            group: config.format.group,
            trees: 1,
            leaves: config.format.leaves,
        }
    }

    pub fn rounds(&self) -> usize {
        self.schedule.len()
    }

    /// ★ Whether round `r` is a tail round under [`LeafQuery::Lean`]: checked
    /// at every leaf of its current tree once instead of at each query. Round
    /// `r ≥ 1`, and it and every later round have a single current tree lying
    /// wholly in its 4-ary cap (no walked level, `mux_bits` = the depth) with
    /// at least [`TAIL_QUERIES_A_LEAF`] queries a leaf: a suffix of the chain,
    /// so a tail round's successor is a tail round too. Derived from the shape
    /// alone, so the arena, the leaf and the cost model agree.
    pub fn tail_round(&self, r: usize) -> bool {
        self.query == LeafQuery::Lean
            && r >= 1
            && r < self.rounds()
            && (r..self.rounds()).all(|t| self.tail_shaped(t))
    }

    /// Round `r`'s current tree on its own: one tree, at arity 4, capped, its
    /// every leaf a cap node, and [`TAIL_QUERIES_A_LEAF`] queries a leaf.
    fn tail_shaped(&self, r: usize) -> bool {
        let depth = self.current_depth(r);
        let c = self.caps[r];
        self.arity == 4
            && self.trees_at(r) == 1
            && c > 0
            && super::merkle_cap::mux_bits(depth, c, 4) == depth
            && TAIL_QUERIES_A_LEAF << depth <= self.num_queries
    }

    /// Whether round 0's base values arrive four to a hinted word
    /// ([`LeafQuery::Lean`] on a 4-ary tree): hashed as the leaf hash's rate
    /// words and unpacked into the felts the fold reads.
    pub fn round0_packed(&self) -> bool {
        self.query == LeafQuery::Lean && self.arity == 4
    }

    /// Whether the openings' paths stay in the arena until their walk, which
    /// hints each 4-ary level's siblings in hint order on its `b1` bit and
    /// places them with three `Select`s ([`LeafQuery::Lean`] on a 4-ary tree,
    /// [`Siblings::Rotated`]). The arena keeps the proof's child order.
    pub fn paths_rotated(&self) -> bool {
        self.query == LeafQuery::Lean && self.arity == 4
    }

    /// Hinted words one round-0 block of `block` values takes: one a value, or
    /// four values a word when packed.
    pub fn round0_block_words(&self, block: usize) -> usize {
        if self.round0_packed() {
            block.div_ceil(4)
        } else {
            block
        }
    }

    /// Leaves of round `r`'s current tree: the blocks a tail round checks.
    pub fn current_leaves(&self, r: usize) -> usize {
        1usize << self.current_depth(r)
    }

    /// The chains a stack of `num_polys` polynomials is opened by, each a range
    /// of its polynomials: the host's own split (`stacked_eval::index_chains`,
    /// REV-B2-JUDGE B2-1), which the emitter, the arena and the plan all cut by.
    pub fn chains(&self, num_polys: usize) -> Vec<std::ops::Range<usize>> {
        index_chains(num_polys, self.group)
    }

    /// This shape for a chain over `trees` polynomials, a run of
    /// [`Self::chains`]: the folds, trees, caps and `Q` stay the
    /// per-polynomial chain's, and round 0 carries the index (REV-B2-JUDGE J1).
    pub fn run(&self, trees: usize) -> Self {
        assert_eq!(self.trees, 1, "a run is cut from a per-polynomial shape");
        assert!(
            (1..=MAX_INDEX_POLYS).contains(&trees)
                && (trees == 1 || self.group == GroupFold::Indexed),
            "a run of {trees} polynomials under {:?}",
            self.group
        );
        // Lemma B's virtual domain `H_{n+m+lb}` must exist (REV-B2-JUDGE B1-1):
        // the host refuses such a config before reading, and no program for
        // one is emitted.
        let two_adicity =
            <GoldilocksField as math::field::traits::IsFFTField>::TWO_ADICITY as usize;
        assert!(
            self.domain_log[0] + index_vars(trees) <= two_adicity,
            "a run of {trees} over 2^{} has no virtual domain",
            self.domain_log[0]
        );
        Self {
            trees,
            ..self.clone()
        }
    }

    /// `m`: the index variables round 0 binds after its data ones (0 for one
    /// polynomial, today's chain).
    pub fn index_vars(&self) -> usize {
        index_vars(self.trees)
    }

    /// Sumcheck messages round `r` carries: its folds, and at round 0 the `m`
    /// index rounds after them (`k0 + m`, `ChainConfig::index_schedule`).
    pub fn messages(&self, r: usize) -> usize {
        self.schedule[r] + if r == 0 { self.index_vars() } else { 0 }
    }

    /// The trees round `r` opens at each query position: the run's at round 0,
    /// the one successor after it.
    pub fn trees_at(&self, r: usize) -> usize {
        if r == 0 { self.trees } else { 1 }
    }

    /// Which of round `r`'s nonce slots (folding, out-of-domain, query) have an
    /// arena word: all three under [`NonceLayout::Three`], the spent ones
    /// under [`NonceLayout::Spent`], a grind of nonzero bits and the
    /// out-of-domain one only where the round has a successor.
    ///
    /// ★ The one place the nonce layout is written: the arena's word count,
    /// the hints that read it and the words that fill it all take it from here.
    pub fn carries(&self, r: usize) -> [bool; 3] {
        let (folding, ood, query) = self.grind;
        match self.nonces {
            NonceLayout::Three => [true; 3],
            NonceLayout::Spent => [
                folding > 0,
                ood > 0 && self.next_depth(r).is_some(),
                query > 0,
            ],
        }
    }

    /// The current tree's depth at round `r`: its leaves are the round's domain
    /// folded by that round's `k`.
    pub fn current_depth(&self, r: usize) -> usize {
        self.domain_log[r] - self.schedule[r]
    }

    /// The successor tree's depth at round `r`, which is the NEXT round's
    /// current depth. Only rounds before the last have one.
    pub fn next_depth(&self, r: usize) -> Option<usize> {
        (r + 1 < self.rounds()).then(|| self.current_depth(r + 1))
    }

    /// Round `r`'s current tree's cap height. ⚠ [`current_depth`](Self::current_depth)
    /// stays the index-bit count; the sibling count is
    /// [`current_path`](Self::current_path).
    pub fn current_cap(&self, r: usize) -> usize {
        self.caps[r]
    }

    /// Siblings on a path to round `r`'s current tree's cap.
    pub fn current_path(&self, r: usize) -> usize {
        self.path_words(self.current_depth(r), self.caps[r])
    }

    /// Sibling words one opening's path to a height-`c` cap carries in a tree
    /// of `depth` index bits: `depth − c` at arity 2, three a walked 4-ary
    /// level and one at an uncapped odd top at arity 4
    /// ([`super::merkle_cap::path_hints`]).
    pub fn path_words(&self, depth: usize, c: usize) -> usize {
        if self.arity == 4 {
            super::merkle_cap::path_hints(depth, c, 4)
        } else {
            depth - c
        }
    }

    /// Words a height-`c` cap of a tree of `depth` index bits occupies: `2^c`
    /// at arity 2, its real nodes at arity 4
    /// ([`super::merkle_cap::cap_nodes`]); none uncapped.
    pub fn cap_words(&self, depth: usize, c: usize) -> usize {
        match (c, self.arity) {
            (0, _) => 0,
            (_, 4) => super::merkle_cap::cap_nodes(depth, c, 4),
            _ => 1usize << c,
        }
    }

    /// Permutations one opening of `felts` felts costs against a height-`c`
    /// cap of a tree of `depth` index bits: the leaf's blocks and a node a
    /// walked level (`Hash16` rows at arity 4: rate-12 blocks, 4-ary levels).
    pub fn opening_perms(&self, felts: usize, depth: usize, c: usize) -> usize {
        if self.arity == 4 {
            felts.div_ceil(super::p1w16_emit::RATE_FELTS)
                + super::merkle_cap::path_permutations(depth, c, 4)
        } else {
            verify_opening_perms_capped(felts, depth, c)
        }
    }

    /// Permutations a height-`c` cap of a tree of `depth` index bits costs up
    /// to its root, once a tree.
    pub fn cap_perms(&self, depth: usize, c: usize) -> usize {
        match (c, self.arity) {
            (0, _) => 0,
            (_, 4) => super::merkle_cap::cap_root_permutations(
                super::merkle_cap::cap_nodes(depth, c, 4),
                4,
            ),
            _ => cap_check_perms(c),
        }
    }

    /// The successor tree's cap height at round `r`.
    pub fn next_cap(&self, r: usize) -> Option<usize> {
        (r + 1 < self.rounds()).then(|| self.caps[r + 1])
    }

    /// Felts in round `r`'s current block: one per value in round 0, where the
    /// codeword is still base-field, and three after.
    pub fn current_felts(&self, r: usize) -> usize {
        let values = 1usize << self.schedule[r];
        if r == 0 { values } else { 3 * values }
    }

    /// Variables bound after round `r`, the index ones included (round 0 binds
    /// them).
    pub fn bound(&self, r: usize) -> usize {
        self.schedule[..=r].iter().sum::<usize>() + self.index_vars()
    }

    /// Variables the chain binds in all: the polynomial's and the index's.
    pub fn chain_vars(&self) -> usize {
        self.num_vars + self.index_vars()
    }
}

/// PERMUTATIONS one chain costs in its query phase and its openings — the term
/// the wrap's cost is dominated by, and a function of the tree depths and block
/// widths alone.
///
/// The transcript's own permutations are NOT in this: they depend on the
/// sponge's buffer at each squeeze, which is a schedule and not a shape, and
/// mixing the two would make a form that no longer says where its cost is.
/// They are [`chain_schedule_perms`], and the grinds' two-apiece are
/// [`chain_grind_perms`]; [`chain_perms`] is the sum of the three.
pub fn chain_opening_perms(shape: &ChainShape) -> usize {
    let mut per_query = 0;
    // A tail round ([`ChainShape::tail_round`]) hashes each leaf's block once
    // (no walked level) and opens no successor a query.
    let mut tail = 0;
    for r in 0..shape.rounds() {
        let current = shape.opening_perms(
            shape.current_felts(r),
            shape.current_depth(r),
            shape.caps[r],
        );
        if shape.tail_round(r) {
            tail += shape.current_leaves(r) * current;
            continue;
        }
        per_query += shape.trees_at(r) * current;
        if let Some(depth) = shape.next_depth(r) {
            per_query += shape.opening_perms(3 << shape.schedule[r + 1], depth, shape.caps[r + 1]);
        }
    }
    shape.num_queries * per_query + tail + chain_cap_perms(shape)
}

/// PERMUTATIONS the chain's cap checks cost: each capped tree's cap hashed up
/// to its root once, `2^c − 1` parents (a grouped round 0's trees each). Zero
/// at the default. Part of [`chain_opening_perms`], stated apart so the
/// per-tree term is visible.
pub fn chain_cap_perms(shape: &ChainShape) -> usize {
    (0..shape.rounds())
        .map(|r| shape.trees_at(r) * shape.cap_perms(shape.current_depth(r), shape.caps[r]))
        .sum()
}

/// INSTRUCTIONS one chain's query phase costs: per round, per query, the two
/// openings, the fold, the successor slot's mux and the assert that closes it.
pub fn chain_query_rows(shape: &ChainShape) -> usize {
    let mut per_round = 0;
    for r in 0..shape.rounds() {
        let felts = shape.current_felts(r);
        let unpacks = if r == 0 {
            0
        } else {
            1usize << shape.schedule[r]
        };
        let depth = shape.current_depth(r);
        if shape.tail_round(r) {
            // A tail round ([`emit_tail_phase`]): the draws, then per leaf its
            // block's leaf, the compare with its cap node (both cells
            // unpacked, four lowered asserts), the fold at its constant
            // position and `folded == claimed`.
            let leaf = super::whir_open::block_leaf_rows(felts, unpacks) + 2 + 2 * 4;
            per_round += shape.num_queries
                + shape.current_leaves(r)
                    * (leaf + fold_rows_for(1usize << shape.schedule[r], depth) + 2);
            if FoldEmission::current() == FoldEmission::Lean {
                per_round += lean_prepare_rows(shape.schedule[r]);
            }
            continue;
        }
        // The index draw, the current opening (one a tree), the trees' blocks
        // combined (a grouped round 0), and the fold.
        let mut q = 1
            + shape.trees_at(r) * verify_opening_rows_capped(felts, unpacks, depth, shape.caps[r])
            + combine_rows(shape, r)
            + fold_rows_for(1usize << shape.schedule[r], depth);
        match shape.next_depth(r) {
            Some(next_depth) => {
                let next_block = 1usize << shape.schedule[r + 1];
                // The successor opening, the slot mux, and `folded == claimed`.
                q += verify_opening_rows_capped(
                    3 * next_block,
                    next_block,
                    next_depth,
                    shape.caps[r + 1],
                ) + (next_block - 1)
                    + 2;
            }
            // `folded == final_value`.
            None => q += 2,
        }
        per_round += shape.num_queries * q;
        // The lean fold's once-a-round `½·α`.
        if FoldEmission::current() == FoldEmission::Lean {
            per_round += lean_prepare_rows(shape.schedule[r]);
        }
    }
    per_round
}

/// INSTRUCTIONS one chain costs OUTSIDE its query phase: the grinds' fixed
/// parts, the sumchecks, the out-of-domain chain, the claim updates, and the
/// tail.
///
/// ⚠ The grinds' `state_rows` term is NOT here. A `state()` hashes whatever the
/// sponge is holding at that moment, which is a property of the schedule rather
/// than of the shape; it is in [`chain_hash_schedule`], as a `State` event,
/// beside the transcript's own squeezes.
pub fn chain_fixed_rows(shape: &ChainShape) -> usize {
    let (folding, ood, query) = shape.grind;
    let mut rows = 0;
    for r in 0..shape.rounds() {
        let k = shape.schedule[r];
        rows += grind_fixed(folding);
        // Per sumcheck round: two absorbed evaluations (one `Unpack` each), one
        // drawn challenge (one `Pack`), and the round itself. Round 0 of a
        // grouped chain has `m` more, then its `eq(ρ, p)`s.
        rows += shape.messages(r) * (SUMCHECK_DEGREE + 1 + sumcheck_round_rows(SUMCHECK_DEGREE));
        if r == 0 {
            rows += eq_index_rows(shape.trees);
        }
        match shape.next_depth(r) {
            Some(_) => {
                let next_domain_log = shape.domain_log[r] - k;
                // The successor root: one `Unpack` into four absorbed felts.
                rows += 1;
                // `z0`: one `Pack`.
                rows += 1;
                // The squaring chain, shared between the domain check and the
                // out-of-domain point, plus the `!= 1` refusal.
                rows += next_domain_log + 2;
                // `y0` absorbed, the ood grind, `gamma` drawn, the claim update.
                rows += 1 + grind_fixed(ood) + 1 + 1;
                rows += grind_fixed(query);
            }
            None => {
                // `final_value` absorbed, then the query grind.
                rows += 1 + grind_fixed(query);
            }
        }
    }
    // The tail: one `eq` and one `MulAdd` per out-of-domain claim, then the
    // inversion, the multiply and the assert.
    for r in 0..shape.rounds() - 1 {
        rows += eq_eval_rows_again(shape.chain_vars() - shape.bound(r)) + 1;
    }
    rows += 1 + 1 + 2;
    // W1: each capped tree's cap check, once (its hinted words are arena
    // words, counted with the arena like every other hint).
    rows += shape
        .caps
        .iter()
        .enumerate()
        .map(|(r, &c)| shape.trees_at(r) * cap_check_rows(c))
        .sum::<usize>();
    rows
}

/// INSTRUCTIONS round `r`'s query spends combining its trees' blocks: one
/// extension row a value a tree ([`emit_combine`]), none with one tree.
pub fn combine_rows(shape: &ChainShape, r: usize) -> usize {
    match shape.trees_at(r) {
        1 => 0,
        trees => trees << shape.schedule[r],
    }
}

/// INSTRUCTIONS [`emit_eq_index`] spends on `trees` slots: one `1 − ρ_i` a
/// position some slot reads as a zero bit (shared), and `m − 1` products a
/// slot; none for one polynomial.
pub fn eq_index_rows(trees: usize) -> usize {
    let m = index_vars(trees);
    let complements = (0..m)
        .filter(|&i| (0..trees).any(|p| (p >> (m - 1 - i)) & 1 == 0))
        .count();
    if m == 0 {
        0
    } else {
        complements + trees * (m - 1)
    }
}

/// A grind's rows beyond the sponge state it reads — see
/// [`super::whir_transcript::grind_check_rows`], whose `state_rows` term
/// belongs to the schedule.
const fn grind_fixed(bits: usize) -> usize {
    super::whir_transcript::grind_check_rows(bits)
}

/// PERMUTATIONS the grinds cost: two a check — one for the inner hash of
/// `PREFIX ‖ seed ‖ factor` and one for the outer hash of `inner ‖ nonce`
/// (`epoch.rs:614-640`). A check at zero bits costs none, because it is not
/// emitted at all.
///
/// The count of checks is the round structure's `3R − 1`, written as the three
/// terms it comes from so a config that grinds in only one place still reads
/// correctly.
pub fn chain_grind_perms(shape: &ChainShape) -> usize {
    if shape.arity == 4 {
        // The width-8 grind is ALU instructions, not socket rows; the
        // `state()` it reads is in the schedule.
        return 0;
    }
    let (folding, ood, query) = shape.grind;
    let rounds = shape.rounds();
    let checks = rounds * usize::from(folding > 0)
        + (rounds - 1) * usize::from(ood > 0)
        + rounds * usize::from(query > 0);
    PERMS_PER_GRIND * checks
}

/// Permutations one grind check spends: its two hashes.
const PERMS_PER_GRIND: usize = 2;

/// ★ The chain's SCHEDULE: every hash its transcript performs, in order, and
/// how many felts each one hashes.
///
/// This is the half of the chain's cost that the shape alone does not give,
/// because a sponge's cost is what its BUFFER holds at each hash and the buffer
/// is a running quantity. It is derived here from the round structure — the
/// absorbs, the draws, the `state()` reads and the candidates a squeeze hands
/// out — and NOT read off [`WhirTranscript`], which is a different state
/// machine written for a different purpose. Its gate is the host's own call
/// stream (`whir_chain_tests::the_schedule_is_the_host_transcripts`).
///
/// Every absorb by the shape it comes from: `SUMCHECK_DEGREE ·
/// COORDINATES_PER_EXT` felts a sumcheck round, `DIGEST_FELTS` for a successor
/// root, `COORDINATES_PER_EXT` for `y0` or `final_value`, `NONCE_FELTS` a
/// spent grind. Every draw: one extension challenge a sumcheck round, `z0` and
/// `gamma` on a round with a successor, and `num_queries` bounded draws in the
/// query phase — which come FIRST and together, because `sample_queries` draws
/// them all before any opening is checked (`whir_round.rs:68-76`).
pub fn chain_hash_schedule(shape: &ChainShape, entry: SpongeEntry) -> Vec<SpongeHash> {
    let mut sponge = SpongeSchedule::new(entry);
    chain_sponge(shape, &mut sponge);
    sponge.hashes().to_vec()
}

/// ★ The same round structure, driven into a schedule the CALLER owns — so the
/// sponge a chain leaves behind is reachable, and the next leg of the same
/// program continues from it.
///
/// ⚠ **This is the one place the chain's round structure is written**, and
/// [`chain_hash_schedule`] is a thin wrapper over it rather than a second copy.
/// A caller that needs the successor entry — `stacked_eval`'s wrapper threading
/// one sponge through every polynomial's chain — could instead replay the
/// structure beside this one, and that replay would be a second thing to keep
/// in step: the schedule's gate
/// (`whir_chain_tests::the_schedule_is_the_host_transcripts`) compares THIS
/// walk against a real verifier's calls, and would say nothing about a copy.
/// Sharing the walk puts the caller inside that gate.
pub fn chain_sponge(shape: &ChainShape, sponge: &mut SpongeSchedule) {
    let (folding, ood, query) = shape.grind;

    for r in 0..shape.rounds() {
        sponge.grind(folding);
        for _ in 0..shape.messages(r) {
            sponge.absorb(SUMCHECK_DEGREE * COORDINATES_PER_EXT);
            sponge.draw_ext();
        }
        match shape.next_depth(r) {
            Some(_) => {
                sponge.absorb_root();
                sponge.draw_ext();
                sponge.absorb(COORDINATES_PER_EXT);
                sponge.grind(ood);
                sponge.draw_ext();
                sponge.grind(query);
            }
            None => {
                sponge.absorb(COORDINATES_PER_EXT);
                sponge.grind(query);
            }
        }
        for _ in 0..shape.num_queries {
            sponge.candidate();
        }
    }
}

/// INSTRUCTIONS the schedule costs: the `Pack`s that build each hash's words
/// and the `Unpack` a squeeze spends reading its digest.
pub fn chain_schedule_rows(shape: &ChainShape, entry: SpongeEntry) -> usize {
    let mut sponge = SpongeSchedule::new(entry);
    chain_sponge(shape, &mut sponge);
    sponge.rows()
}

/// PERMUTATIONS the schedule costs: one per rate-8 block of every hash (a
/// `Hash16` row per Poseidon1 permutation).
pub fn chain_schedule_perms(shape: &ChainShape, entry: SpongeEntry) -> usize {
    let mut sponge = SpongeSchedule::new(entry);
    chain_sponge(shape, &mut sponge);
    sponge.perms()
}

/// ★ INSTRUCTIONS one chain costs, whole: the shape half and the schedule half.
///
/// The two are pinned apart and not only as this sum, because they move
/// independently — a form that shifted work from the sponge to the arithmetic
/// at constant total would fail one of them rather than neither.
pub fn chain_rows(shape: &ChainShape, entry: SpongeEntry) -> usize {
    chain_shape_rows(shape) + chain_schedule_rows(shape, entry)
}

/// ★ PERMUTATIONS one chain costs, whole, in the THREE terms it has: the query
/// phase's openings, the grinds' two hashes each, and the transcript's own.
///
/// ⚠ Only the first was pinned before this; the sizing note's per-chain
/// permutation figure predates the other two and is not reproduced by them.
pub fn chain_perms(shape: &ChainShape, entry: SpongeEntry) -> usize {
    chain_opening_perms(shape) + chain_grind_perms(shape) + chain_schedule_perms(shape, entry)
}

/// ★★ THE FOLD CONSTANTS ONE WHOLE CHAIN INTERNS, by value — every round's,
/// unioned, over that round's own domain.
///
/// ⛔ A UNION AND NOT A MAXIMUM. Round `r` folds over the base domain squared
/// `Σ_{j<r} schedule[j]` times, so each round's generator differs and the same
/// exponent set yields different field elements. Nothing nests here the way the
/// sumcheck round's Newton pairs do, and a form that took a max of anything
/// would be quietly wrong.
///
/// The domain advance mirrors the emitter's own (`:484-489`): `k` squarings per
/// round, where `k` is that round's schedule entry.
pub fn chain_fold_constants(
    shape: &ChainShape,
    domain: &Domain<GoldilocksField>,
) -> Vec<super::word::LfmWord> {
    let mut words: Vec<super::word::LfmWord> = Vec::new();
    let mut current = domain.clone();
    for r in 0..shape.rounds() {
        for word in super::whir_fold::fold_constants_for(
            &current,
            shape.schedule[r],
            shape.current_depth(r),
        ) {
            if !words.contains(&word) {
                words.push(word);
            }
        }
        for _ in 0..shape.schedule[r] {
            current = current
                .squared()
                .expect("the schedule never folds past the domain");
        }
    }
    words
}

/// ★ `whir_chain::verify_weighted`, emitted.
///
/// `weight_at` is the caller's closure, mirroring the host's `W` parameter: it
/// is handed the concatenated round challenges and returns the weight at that
/// point. `whir_chain::verify` passes `eq_eval(z, ·)`, which is
/// [`emit_eq_eval`]; a stacked chain passes
/// [`super::whir_stacked::emit_weight_at`].
///
/// Every refusal is a division with no satisfying assignment rather than a
/// branch, so a proof the host rejects has no execution here.
#[allow(clippy::too_many_arguments)]
pub fn emit_verify_weighted(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    rounds: &[ChainRoundWires<'_>],
    root_lanes: &[Felt; 4],
    final_value: Ext,
    y: Ext,
    shape: &ChainShape,
    domain: &Domain<GoldilocksField>,
    weight_at: impl FnOnce(&mut LfmBuilder, &[Ext]) -> Ext,
) {
    assert_eq!(
        shape.trees, 1,
        "a chain over one polynomial; a run of several is emit_verify_grouped's"
    );
    emit_chain(
        b,
        transcript,
        rounds,
        std::slice::from_ref(root_lanes),
        final_value,
        y,
        shape,
        domain,
        |b, at, _| weight_at(b, at),
    );
}

/// ★ `whir_chain::verify_grouped`, emitted: ONE chain over a run of
/// `shape.trees` stacked polynomials ([`GroupFold::Indexed`], D-ONECHAIN
/// §2.7), each committed under its own root in `roots`, in slot order.
///
/// Round 0 runs `k0 + m` sumcheck rounds, the `m` index challenges `ρ` last;
/// at each query position it opens EVERY tree at the one drawn index, combines
/// the blocks as `Σ_p eq(ρ, p)·block_p` and folds that as today's `2^k0`
/// block. Every later round is [`emit_verify_weighted`]'s.
///
/// `weight_at` is handed the chain's `n + m` challenges in binding order and
/// the run's `eq(ρ, p)`s, `p < trees` ([`emit_eq_index`]): the one evaluation
/// the combine reads too, from the sampled `ρ` wires. An absent slot `p ≥
/// trees` has no `eq`, no block and no hint wire.
#[allow(clippy::too_many_arguments)]
pub fn emit_verify_grouped(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    rounds: &[ChainRoundWires<'_>],
    roots: &[[Felt; 4]],
    final_value: Ext,
    y: Ext,
    shape: &ChainShape,
    domain: &Domain<GoldilocksField>,
    weight_at: impl FnOnce(&mut LfmBuilder, &[Ext], &[Ext]) -> Ext,
) {
    assert!(
        shape.trees >= 2,
        "a run of one polynomial is emit_verify_weighted's"
    );
    emit_chain(
        b,
        transcript,
        rounds,
        roots,
        final_value,
        y,
        shape,
        domain,
        weight_at,
    );
}

/// [`emit_verify_weighted`] and [`emit_verify_grouped`]: one chain over
/// `roots.len() = shape.trees` polynomials. At one polynomial every row it
/// emits, and their order, are [`emit_verify_weighted`]'s as they were.
#[allow(clippy::too_many_arguments)]
fn emit_chain(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    rounds: &[ChainRoundWires<'_>],
    roots: &[[Felt; 4]],
    final_value: Ext,
    y: Ext,
    shape: &ChainShape,
    domain: &Domain<GoldilocksField>,
    weight_at: impl FnOnce(&mut LfmBuilder, &[Ext], &[Ext]) -> Ext,
) {
    assert_eq!(
        rounds.len(),
        shape.rounds(),
        "one set of wires per scheduled round"
    );
    assert_eq!(roots.len(), shape.trees, "one root per round-0 tree");
    let one = b.ext_const(&FEE::one());
    let (grind_folding, grind_ood, grind_query) = shape.grind;

    let mut claim = y;
    let mut alphas: Vec<Ext> = Vec::with_capacity(shape.chain_vars());
    // Round 0's trees: each one's cap (when it has one) is authenticated
    // against its root here, once; every later tree where its root is absorbed.
    // Tree `p` is checked against `roots[p]` alone, by its position.
    let depth0 = shape.current_depth(0);
    let cap_words = shape.cap_words(depth0, shape.caps[0]);
    assert_eq!(
        rounds[0].current_cap.len(),
        shape.trees * cap_words,
        "one cap a round-0 tree"
    );
    let mut current_trees: Vec<TreeAuth> = roots
        .iter()
        .enumerate()
        .map(|(p, lanes)| {
            tree_auth(
                b,
                shape,
                depth0,
                shape.caps[0],
                &rounds[0].current_cap[p * cap_words..(p + 1) * cap_words],
                lanes,
            )
        })
        .collect();
    let mut current_domain = domain.clone();
    // Each out-of-domain claim: its batching weight, its point, and how many
    // variables were bound when it entered.
    let mut ood: Vec<(Ext, Vec<Ext>, usize)> = Vec::new();
    // The run's `eq(ρ, p)`s, once round 0 has drawn `ρ`; none for one tree.
    let mut eqs: Vec<Ext> = Vec::new();

    for (r, round) in rounds.iter().enumerate() {
        let k = shape.schedule[r];
        let sent = shape.messages(r);
        assert_eq!(
            round.sumcheck.len(),
            sent,
            "round {r} carries {sent} sumcheck messages"
        );

        #[cfg(test)]
        onechain_steps::mark(b, onechain_steps::Step::Sumcheck { round: r });
        emit_spent_grind(b, transcript, grind_folding, round.nonces.folding);

        // The sumcheck, interleaved: a round's challenge is drawn only after
        // its evaluations are absorbed, which is the order `verify_rounds`
        // takes (`sumcheck.rs:386-390`) and the seam the round primitive was
        // left with.
        let mut point = Vec::with_capacity(sent);
        for evaluations in round.sumcheck {
            assert_eq!(evaluations.len(), SUMCHECK_DEGREE);
            for e in evaluations {
                transcript.absorb_ext(b, *e);
            }
            let challenge = transcript.sample_ext(b);
            claim = emit_sumcheck_round(b, claim, evaluations, challenge);
            point.push(challenge);
        }
        // A grouped round 0 bound its index LAST: the challenges past the
        // first `k` are `ρ`, and only the first `k` fold a block.
        let (data, rho) = point.split_at(k);
        if !rho.is_empty() {
            eqs = emit_eq_index(b, rho, shape.trees, one);
        }
        // The combine reads the same `eq(ρ, p)`s the weight does.
        #[cfg(test)]
        let faulted: Vec<Ext> =
            if r == 0 && onechain_faults::get() == onechain_faults::Fault::SwapRhoInCombine {
                let swapped: Vec<Ext> = rho.iter().rev().copied().collect();
                emit_eq_index(b, &swapped, shape.trees, one)
            } else {
                eqs.clone()
            };
        #[cfg(test)]
        let combine_eqs: &[Ext] = &faulted;
        #[cfg(not(test))]
        let combine_eqs: &[Ext] = &eqs;

        let mut next_domain = current_domain.clone();
        for _ in 0..k {
            next_domain = next_domain
                .squared()
                .expect("the schedule never folds past the domain");
        }
        let bound = alphas.len() + sent;

        #[cfg(test)]
        onechain_steps::mark(b, onechain_steps::Step::Ood { round: r });
        let next_tree = match (round.next_root, round.ood_value, shape.next_depth(r)) {
            (Some(next_root), Some(y0), Some(_)) => {
                let lanes = b.unpack(next_root);
                transcript.absorb_root_lanes(b, &lanes);

                let z0 = transcript.sample_ext(b);

                // ONE squaring chain for two consumers. `powers[j] = z0^(2^j)`:
                // the out-of-domain point takes the first `n + m − bound` of
                // them and the domain check takes the last.
                let depth = next_domain.log_size();
                let mut powers = Vec::with_capacity(depth + 1);
                powers.push(z0);
                for j in 0..depth {
                    let squared = b.emul(powers[j], powers[j]);
                    powers.push(squared);
                }
                emit_require_out_of_domain(b, powers[depth], one);
                let ood_point: Vec<Ext> = powers[..shape.chain_vars() - bound].to_vec();

                transcript.absorb_ext(b, y0);
                emit_spent_grind(b, transcript, grind_ood, round.nonces.ood);
                let gamma = transcript.sample_ext(b);
                claim = b.emul_add(gamma, y0, claim);
                ood.push((gamma, ood_point, bound));

                emit_spent_grind(b, transcript, grind_query, round.nonces.query);
                Some(tree_auth(
                    b,
                    shape,
                    shape.current_depth(r + 1),
                    shape.caps[r + 1],
                    round.next_cap,
                    &lanes,
                ))
            }
            (None, None, None) => {
                assert!(round.next_cap.is_empty(), "the last round has no successor");
                transcript.absorb_ext(b, final_value);
                emit_spent_grind(b, transcript, grind_query, round.nonces.query);
                None
            }
            _ => panic!(
                "round {r}: a successor root and an out-of-domain value exist together, and \
                 exactly on the rounds the schedule gives a successor"
            ),
        };

        if shape.tail_round(r) {
            let next_blocks = rounds.get(r + 1).map(|next| next.current);
            emit_tail_phase(
                b,
                transcript,
                round,
                shape,
                r,
                &current_domain,
                data,
                &current_trees,
                next_blocks,
                final_value,
            );
        } else {
            emit_query_phase(
                b,
                transcript,
                round,
                shape,
                r,
                &current_domain,
                data,
                &current_trees,
                if r == 0 { combine_eqs } else { &[] },
                next_tree.as_ref(),
                final_value,
            );
        }

        if let Some(tree) = next_tree {
            current_trees = vec![tree];
        }
        alphas.extend(point);
        current_domain = next_domain;
    }

    // The accumulated weight: the caller's own, plus each out-of-domain claim's
    // batched `eq` over the challenges that came after it.
    #[cfg(test)]
    onechain_steps::mark(b, onechain_steps::Step::Final);
    let mut weight = weight_at(b, &alphas, &eqs);
    for (gamma, point, bound) in &ood {
        let eq = emit_eq_eval(b, point, &alphas[*bound..]);
        weight = b.emul_add(*gamma, eq, weight);
    }

    emit_final_check(b, claim, weight, final_value, one);
}

/// Test only: emitter faults a grouped-chain test switches on, one at a time,
/// to show each in-guest reading of the index is load-bearing (REV-B2-JUDGE
/// B2-5, the host's B1-3 negatives): under any of them the honest arena has no
/// execution. Thread-local, so a test that sets one moves no other test's
/// program.
#[cfg(test)]
pub(crate) mod onechain_faults {
    /// Which piece of the emitted verifier reads the index wrongly.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) enum Fault {
        #[default]
        None,
        /// Round 0's combine reads `ρ` reversed.
        SwapRhoInCombine,
        /// The final weight reads `ρ` reversed.
        SwapRhoInWeight,
        /// The final weight drops `eq(ρ, p)`: `Σ_p w_p` at the data point.
        DropIndexFactor,
        /// The final weight gives each absent slot `p ≥ P'` a weight of 1.
        AbsentSlotWeightOne,
    }

    std::thread_local! {
        static FAULT: std::cell::Cell<Fault> = const { std::cell::Cell::new(Fault::None) };
    }

    pub(crate) fn get() -> Fault {
        FAULT.with(std::cell::Cell::get)
    }

    /// Runs `f` under `fault`, restoring the previous one after.
    pub(crate) fn with<R>(fault: Fault, f: impl FnOnce() -> R) -> R {
        struct Restore(Fault);
        impl Drop for Restore {
            fn drop(&mut self) {
                FAULT.with(|c| c.set(self.0));
            }
        }
        let _restore = Restore(FAULT.with(|c| c.replace(fault)));
        f()
    }
}

/// Test only: where each check of a chain's emission begins, as the builder's
/// instruction index, so a refusal is named by the check holding the assert
/// that failed — the executor reports a failed `assert_eq` as `DivByZero` at
/// the address of its difference (`executor::locate_addr`), and a negative
/// test names the step against its honest twin, which executes past it.
/// Recording only: nothing is emitted, and nothing is kept outside
/// [`onechain_steps::record`].
#[cfg(test)]
pub(crate) mod onechain_steps {
    use super::super::builder::LfmBuilder;
    use super::super::compiler::LfmProgram;
    use super::super::executor::LfmExecError;
    use super::super::instr::Instr;

    /// A check of the emitted chain, in emission order.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Step {
        /// A stacked opening's chain whose first polynomial is this begins
        /// (its roots, its claim).
        Chain(usize),
        /// A round's folding grind and sumcheck.
        Sumcheck { round: usize },
        /// A round's successor root, out-of-domain point, grinds (or the final
        /// value) and query draws.
        Ood { round: usize },
        /// Tree `tree`'s opening at a query (a grouped round 0 has several).
        Opening {
            round: usize,
            query: usize,
            tree: usize,
        },
        /// The successor's opening at a query.
        Successor { round: usize, query: usize },
        /// A query's combine (a grouped round 0), fold and compare.
        Fold { round: usize, query: usize },
        /// The weight and the final check.
        Final,
    }

    /// [`Step::Opening`].
    pub(crate) fn opening(round: usize, query: usize, tree: usize) -> Step {
        Step::Opening { round, query, tree }
    }

    std::thread_local! {
        static MARKS: std::cell::RefCell<Option<Vec<(Step, usize)>>> =
            const { std::cell::RefCell::new(None) };
    }

    pub(crate) fn mark(b: &LfmBuilder, step: Step) {
        MARKS.with(|m| {
            if let Some(marks) = m.borrow_mut().as_mut() {
                marks.push((step, b.instrs().len()));
            }
        });
    }

    /// Runs `f` and returns the marks its emission left.
    pub(crate) fn record<R>(f: impl FnOnce() -> R) -> (R, Vec<(Step, usize)>) {
        let previous = MARKS.with(|m| m.replace(Some(Vec::new())));
        let out = f();
        let marks = MARKS.with(|m| m.replace(previous)).unwrap_or_default();
        (out, marks)
    }

    /// The chain (by its first polynomial) and the step whose assert failed:
    /// the instruction that wrote a `DivByZero`'s address, under the last chain
    /// mark and the last step mark at or before it. `None` for any other
    /// refusal.
    pub(crate) fn failed_at(
        program: &LfmProgram,
        marks: &[(Step, usize)],
        error: &LfmExecError,
    ) -> Option<(Option<usize>, Step)> {
        let LfmExecError::DivByZero { addr } = error else {
            return None;
        };
        let index = program.instrs.iter().position(|i| match i {
            Instr::BaseAlu { out, .. } | Instr::ExtAlu { out, .. } => out.0 == *addr,
            _ => false,
        })?;
        let before = marks.iter().filter(|(_, at)| *at <= index);
        let chain = before.clone().rev().find_map(|(s, _)| match s {
            Step::Chain(c) => Some(*c),
            _ => None,
        });
        let step = before
            .rev()
            .find(|(s, _)| !matches!(s, Step::Chain(_)))
            .map(|(s, _)| *s)?;
        Some((chain, step))
    }
}

/// `eq(ρ, p)` for each slot `p < trees`, `ρ_0` against the MOST significant of
/// `p`'s bits: the host's `whir_chain::eq_index`, emitted from the sampled `ρ`
/// wires and never hinted (REV-B2-JUDGE B2-5). An absent slot `p ≥ trees` has
/// no `eq` at all. `1 − ρ_i` is emitted once a position, when a slot reads it.
pub fn emit_eq_index(b: &mut LfmBuilder, rho: &[Ext], trees: usize, one: Ext) -> Vec<Ext> {
    let m = rho.len();
    assert_eq!(
        m,
        index_vars(trees),
        "{trees} slots bind {m} index variables"
    );
    let mut complement: Vec<Option<Ext>> = vec![None; m];
    let mut eqs = Vec::with_capacity(trees);
    for p in 0..trees {
        let mut acc: Option<Ext> = None;
        for (i, &r) in rho.iter().enumerate() {
            let factor = if (p >> (m - 1 - i)) & 1 == 1 {
                r
            } else {
                match complement[i] {
                    Some(c) => c,
                    None => {
                        let c = b.esub(one, r);
                        complement[i] = Some(c);
                        c
                    }
                }
            };
            acc = Some(match acc {
                None => factor,
                Some(a) => b.emul(a, factor),
            });
        }
        eqs.push(acc.expect("a run of two or more binds an index variable"));
    }
    eqs
}

/// `Σ_p eqs[p]·block_p`, value by value: the one `2^k0` block a grouped round
/// 0 folds (the host's `whir_round::combine_blocks`).
///
/// ★ THE BASE-LIFT PIN, PER SLOT (REV-B2-JUDGE B2-2, REV-B2-B RF-2). Each block
/// is read as EXACTLY the felt slice tree `p`'s `verify_opening` hashed — the
/// caller hands both the same [`QueryOpening`] — so its upper lanes are pinned
/// by that leaf hash's `Pack`s before the lift reads them ([`super::whir_open`]'s
/// obligation). A block not hashed by its tree has no way in.
fn emit_combine(b: &mut LfmBuilder, eqs: &[Ext], blocks: &[QueryOpening<'_>]) -> Vec<Ext> {
    assert_eq!(eqs.len(), blocks.len(), "one eq(ρ, p) a tree's block");
    let felts: Vec<&[Felt]> = blocks
        .iter()
        .map(|block| match block.values {
            BlockValues::Base(felts) | BlockValues::BaseWords { felts, .. } => felts,
            BlockValues::Ext(_) => panic!("a grouped round 0's blocks are base-field"),
        })
        .collect();
    let width = felts[0].len();
    (0..width)
        .map(|t| {
            let mut acc = b.emul_base(eqs[0], felts[0][t]);
            for (eq, block) in eqs.iter().zip(&felts).skip(1) {
                acc = b.emul_add(*eq, block[t].as_ext(), acc);
            }
            acc
        })
        .collect()
}

/// One grind slot, emitted: [`emit_grind_check`] where the grind spends bits,
/// nothing where it does not (the host's `check_grind` reads nothing then).
///
/// A spent grind always has its nonce wire, because [`ChainShape::carries`]
/// carries every slot of nonzero bits under either layout; a missing one is a
/// wire walk that disagrees with the shape it was built from, refused here at
/// emit time rather than emitted as a grind that checks nothing.
fn emit_spent_grind(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    bits: usize,
    nonce: Option<Felt>,
) {
    if bits == 0 {
        return;
    }
    let nonce = nonce.expect("a grind of nonzero bits has its nonce wire under every layout");
    emit_grind_check(b, transcript, bits as u8, nonce);
}

/// How a tree's openings are checked: against its root lanes when its cap
/// height is 0 (today's emission, unchanged), else against its cap, hinted as
/// `cap` and authenticated against the root lanes HERE — the one place a
/// tree's [`CapCells`] are made.
fn tree_auth(
    b: &mut LfmBuilder,
    shape: &ChainShape,
    depth: usize,
    cap_height: usize,
    cap: &[WrapDigest],
    root_lanes: &[Felt; 4],
) -> TreeAuth {
    if cap_height == 0 {
        assert!(cap.is_empty(), "an uncapped tree carries no cap wires");
        TreeAuth::Root(*root_lanes)
    } else if shape.arity == 4 {
        assert_eq!(
            cap.len(),
            shape.cap_words(depth, cap_height),
            "a 4-ary cap is its level's real nodes"
        );
        TreeAuth::Cap(CapCells::authenticate_at(
            b,
            cap,
            cap_height,
            std::slice::from_ref(root_lanes),
        ))
    } else {
        assert_eq!(cap.len(), 1usize << cap_height, "a cap is 2^c digests");
        TreeAuth::Cap(CapCells::authenticate(
            b,
            cap,
            std::slice::from_ref(root_lanes),
        ))
    }
}

/// ★ `require_out_of_domain` (`whir_chain.rs:88-101`), emitted as a REFUSAL.
///
/// The host rejects `z0` whose `2^log_size`-th power is one, because such a
/// point lies inside the domain and constrains nothing beyond the in-domain
/// queries. Here `raised − 1` is inverted: `1/0` has no satisfying assignment,
/// so an in-domain `z0` has no execution.
///
/// ⚠ **Its own function because no honest fixture reaches it.** A transcript
/// draws an in-domain `z0` with negligible probability, so a chain-level
/// mutation that deletes this check passes every gate — measured, not assumed:
/// mutation CC removed it and all four chain tests stayed green. A refusal that
/// nothing can exercise is a check that cannot fail, and the cure is to make it
/// callable so a test can hand it the input the protocol never will.
pub fn emit_require_out_of_domain(b: &mut LfmBuilder, raised: Ext, one: Ext) {
    let gap = b.esub(raised, one);
    let _ = b.ediv(one, gap);
}

/// ★ The chain's last check: `final_value == claim · weight⁻¹`.
///
/// Inverting rather than cross-multiplying, and the difference is soundness
/// rather than taste. `final_value · weight == claim` is one row cheaper and
/// accepts EVERY `final_value` when `weight` and `claim` are both zero, where
/// the host returns `DegenerateEvaluationPoint` (`whir_chain.rs:1095-1102`).
///
/// ⚠ **Its own function for the same reason as above.** Weight and claim are
/// both zero only at a point a transcript does not produce, so mutation CD
/// replaced this with the cross-multiplied form and all four chain tests stayed
/// green. The gap is closed by calling this directly with the degenerate input.
pub fn emit_final_check(b: &mut LfmBuilder, claim: Ext, weight: Ext, final_value: Ext, one: Ext) {
    let inv = b.ediv(one, weight);
    let required = b.emul(claim, inv);
    b.assert_eq_ext(final_value, required);
}

/// `whir_round::verify` and `verify_final`: draw every query position, then
/// check every opening.
///
/// The draws come first and together, because that is the order
/// `sample_queries` takes (`whir_round.rs:68-76`) and the transcript is
/// sequential — interleaving them with the openings would change every
/// challenge after the first.
///
/// ★ A grouped round 0 (`current_trees.len() ≥ 2`, `whir_round::verify_grouped`)
/// reads `round.current` query-major, one opening a tree, and ONE draw a query
/// feeds every tree's `verify_opening`, the successor's leaf and slot, and the
/// fold point (REV-B2-JUDGE B2-3): a tree opened at another position has no
/// wire to arrive on. The trees' blocks are combined with `eqs` and folded as
/// one block.
#[allow(clippy::too_many_arguments)]
fn emit_query_phase(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    round: &ChainRoundWires<'_>,
    shape: &ChainShape,
    r: usize,
    current_domain: &Domain<GoldilocksField>,
    alphas: &[Ext],
    current_trees: &[TreeAuth],
    eqs: &[Ext],
    next_tree: Option<&TreeAuth>,
    final_value: Ext,
) {
    let depth = shape.current_depth(r);
    let trees = shape.trees_at(r);
    assert_eq!(current_trees.len(), trees, "round {r} opens {trees} trees");
    debug_assert!(
        current_trees
            .iter()
            .all(|t| t.cap_height() == shape.caps[r])
    );
    assert_eq!(round.current.len(), shape.num_queries * trees);
    assert_eq!(
        eqs.len(),
        if trees == 1 { 0 } else { trees },
        "one eq(ρ, p) a tree"
    );
    let queries: Vec<Vec<_>> = (0..shape.num_queries)
        .map(|_| transcript.sample_u64_pow2(b, depth))
        .collect();
    // The current trees are walked at the block's tree position; the fold, the
    // successor's leaf and its slot read the index itself.
    let walks: Vec<std::borrow::Cow<'_, [Bit]>> = queries
        .iter()
        .map(|bits| walk_bits(shape, r, bits))
        .collect();
    // The lean fold halves the round's challenges once, here, rather
    // than once per query; the classic fold prepares nothing.
    let fold = RoundFold::prepare(b, alphas);

    match (next_tree, shape.next_depth(r)) {
        (Some(next_tree), Some(next_depth)) => {
            let next_block = 1usize << shape.schedule[r + 1];
            assert_eq!(round.next.len(), shape.num_queries);
            for (q, bits) in queries.iter().enumerate() {
                // Each tree's block at the query's one position, against that
                // tree's own check.
                let blocks = &round.current[q * trees..(q + 1) * trees];
                for (p, current) in blocks.iter().enumerate() {
                    #[cfg(test)]
                    onechain_steps::mark(b, onechain_steps::opening(r, q, p));
                    current_trees[p].verify_opening(b, current.values, &walks[q], current.siblings);
                }
                let next = &round.next[q];
                // `leaf_and_slot`: the low `next_depth` bits index the successor
                // leaf and the high ones choose the slot inside it. Both bounds
                // are powers of two, so this is a partition of the bits.
                let (leaf_bits, slot_bits) = bits.split_at(next_depth);
                #[cfg(test)]
                onechain_steps::mark(b, onechain_steps::Step::Successor { round: r, query: q });
                next_tree.verify_opening(b, next.values, leaf_bits, next.siblings);

                #[cfg(test)]
                onechain_steps::mark(b, onechain_steps::Step::Fold { round: r, query: q });
                let block = folded_block(b, blocks, eqs);
                let folded = fold.fold(b, &block, current_domain, bits);
                let claimed = emit_slot_mux(b, &block_ext(next.values), slot_bits);
                b.assert_eq_ext(folded, claimed);
                debug_assert_eq!(1usize << slot_bits.len(), next_block);
            }
        }
        _ => {
            for (q, bits) in queries.iter().enumerate() {
                let blocks = &round.current[q * trees..(q + 1) * trees];
                for (p, current) in blocks.iter().enumerate() {
                    #[cfg(test)]
                    onechain_steps::mark(b, onechain_steps::opening(r, q, p));
                    current_trees[p].verify_opening(b, current.values, &walks[q], current.siblings);
                }
                #[cfg(test)]
                onechain_steps::mark(b, onechain_steps::Step::Fold { round: r, query: q });
                let block = folded_block(b, blocks, eqs);
                let folded = fold.fold(b, &block, current_domain, bits);
                b.assert_eq_ext(folded, final_value);
            }
        }
    }
}

/// ★ A tail round's check ([`ChainShape::tail_round`], [`LeafQuery::Lean`]):
/// the query draws, as [`emit_query_phase`] makes them, then EVERY leaf of the
/// current tree at once.
///
/// Each leaf's block is hashed and must equal its cap node: the tree lies
/// wholly in its cap, already authenticated against the root as the round
/// before's successor, so the positions are program constants and no node is
/// muxed. Each block folds at its constant position, and the fold must equal
/// the successor's value at that position — successor block `i mod 2^d'`,
/// slot `i >> d'`, read from the successor tail round's own hinted blocks
/// (`next_blocks`), which that round authenticates the same way — or the
/// final value at the last round.
///
/// Every sampled position is one of these, so the leaf accepts only what the
/// host's checks at the sampled positions accept; an honest proof passes at
/// every position (WHIR's completeness is perfect). The draws stay where the
/// host's transcript takes them, and their bits go unused.
#[allow(clippy::too_many_arguments)]
fn emit_tail_phase(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    round: &ChainRoundWires<'_>,
    shape: &ChainShape,
    r: usize,
    current_domain: &Domain<GoldilocksField>,
    alphas: &[Ext],
    current_trees: &[TreeAuth],
    next_blocks: Option<&[QueryOpening<'_>]>,
    final_value: Ext,
) {
    let depth = shape.current_depth(r);
    let leaves = shape.current_leaves(r);
    assert!(shape.tail_round(r), "round {r} is not a tail round");
    assert_eq!(
        round.current.len(),
        leaves,
        "a tail round hints every leaf once"
    );
    assert!(
        round.next.is_empty(),
        "a tail round opens no successor block a query"
    );
    for _ in 0..shape.num_queries {
        let _ = transcript.sample_u64_pow2(b, depth);
    }
    let fold = RoundFold::prepare(b, alphas);
    let [TreeAuth::Cap(cap)] = current_trees else {
        panic!("a tail round's one current tree is capped")
    };
    let digests: Vec<WrapDigest> = round
        .current
        .iter()
        .map(|block| super::whir_open::emit_block_leaf(b, block.values))
        .collect();
    cap.verify_every_leaf(b, &digests);
    let next = match (next_blocks, shape.next_depth(r)) {
        (Some(blocks), Some(next_depth)) => {
            debug_assert!(
                shape.tail_round(r + 1),
                "a tail round's successor is a tail round"
            );
            assert_eq!(
                blocks.len(),
                1usize << next_depth,
                "the successor's every leaf"
            );
            Some((blocks, next_depth))
        }
        (None, None) => None,
        _ => panic!("round {r}: successor blocks exactly where the schedule has a successor"),
    };
    for (i, block) in round.current.iter().enumerate() {
        let bits: Vec<Bit> = (0..depth).map(|t| b.bit_const((i >> t) & 1 == 1)).collect();
        let folded = fold.fold(b, &block_ext(block.values), current_domain, &bits);
        let claimed = match next {
            Some((blocks, next_depth)) => {
                let successor = &blocks[i % (1usize << next_depth)];
                let slot = i >> next_depth;
                #[cfg(test)]
                let slot = if tail_faults::slot_off() {
                    slot ^ 1
                } else {
                    slot
                };
                block_ext(successor.values)[slot]
            }
            None => final_value,
        };
        b.assert_eq_ext(folded, claimed);
    }
}

/// Test only: a tail round's compare reads the successor's NEIGHBOURING slot,
/// so an honest proof has no execution — the compare is live and reads the
/// slot the position names. Thread-local, as [`onechain_faults`] is.
#[cfg(test)]
pub(crate) mod tail_faults {
    std::thread_local! {
        static SLOT_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    pub(crate) fn slot_off() -> bool {
        SLOT_OFF.with(std::cell::Cell::get)
    }

    /// Runs `f` with the fault on, restoring it after.
    pub(crate) fn with_slot_off<R>(f: impl FnOnce() -> R) -> R {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                SLOT_OFF.with(|c| c.set(self.0));
            }
        }
        let _restore = Restore(SLOT_OFF.with(|c| c.replace(true)));
        f()
    }
}

/// The bits round `r`'s current trees are walked on for a query of index bits
/// `bits` (low first): the index itself, or at round 0 under
/// [`LeafOrder::Coset`] the position the order puts the block at, the index
/// bits reversed (`LeafOrder::position`) — the same wires in the other order,
/// no row.
fn walk_bits<'a>(shape: &ChainShape, r: usize, bits: &'a [Bit]) -> std::borrow::Cow<'a, [Bit]> {
    match (r, shape.leaves) {
        (0, LeafOrder::Coset) => std::borrow::Cow::Owned(bits.iter().rev().copied().collect()),
        _ => std::borrow::Cow::Borrowed(bits),
    }
}

/// The block a query folds: its one opening's values, or a grouped round 0's
/// trees' blocks combined with `eqs` ([`emit_combine`]).
fn folded_block(b: &mut LfmBuilder, blocks: &[QueryOpening<'_>], eqs: &[Ext]) -> Vec<Ext> {
    match blocks {
        [one] => block_ext(one.values),
        _ => emit_combine(b, eqs, blocks),
    }
}

/// A block as extension wires. A base block's values lift for free — a base
/// cell is `(v, 0, 0, 0)` — and the lift is sound because the leaf hash pinned
/// those lanes; [`super::whir_open`]'s module doc carries that obligation.
fn block_ext(values: BlockValues<'_>) -> Vec<Ext> {
    match values {
        BlockValues::Base(felts) | BlockValues::BaseWords { felts, .. } => {
            felts.iter().map(|f| f.as_ext()).collect()
        }
        BlockValues::Ext(cells) => cells.to_vec(),
    }
}

/// `values[slot]` where `slot` is given as bits, LOW first: a balanced mux,
/// `values.len() − 1` `Select` rows.
///
/// `Select` returns both arms for one row (`builder.rs:298-314`), so a level
/// that halves `2m` values into `m` costs `m` rows and the whole mux costs
/// `values.len() − 1`.
///
/// ⚠ The pairing is `(2t, 2t + 1)` and not `(t, t + half)`, because the bits
/// arrive low first. `t` against `t + half` differs in the HIGH bit, so pairing
/// that way while consuming the low bit first reads the index backwards — it
/// agrees at every palindromic slot and nowhere else, which is the shape of a
/// bug a small fixture hides.
fn emit_slot_mux(b: &mut LfmBuilder, values: &[Ext], bits: &[Bit]) -> Ext {
    assert_eq!(
        values.len(),
        1usize << bits.len(),
        "a slot mux covers the whole successor block"
    );
    let mut level: Vec<Ext> = values.to_vec();
    for bit in bits {
        level = level
            .chunks_exact(2)
            .map(|pair| {
                let (chosen, _) = b.select(*bit, pair[0].as_cell(), pair[1].as_cell());
                chosen.as_ext()
            })
            .collect();
    }
    level[0]
}

/// The half of a chain's cost that does not depend on the sponge's schedule.
///
/// The other half is [`chain_schedule_rows`] and the sum is [`chain_rows`].
pub fn chain_shape_rows(shape: &ChainShape) -> usize {
    chain_fixed_rows(shape) + chain_query_rows(shape)
}

// =============================================================================
// One chain's wires in an arena — MOVED OUT OF `whir_chain_tests` (V1g)
// =============================================================================
//
// These four were `pub(super)` inside the chain's own test module, which is
// `#[cfg(test)]`, so nothing in a production module could reach them. The
// assembled epoch verify (`whir_epoch`) is a production module that holds EIGHT
// chains plus the DECODE group's, and it has to hint their wires out of one
// arena; a second copy of this walk beside the one the chain suite gates would
// be exactly the drift the chain suite exists to prevent. So the walk moves
// here and the suite keeps gating it.
//
// Nothing about them changed but their visibility and `round_words` losing an
// `impl` block: the chain suite is the control that says so.

/// An opening's path as [`RoundStorage`] holds it.
enum PathStore {
    /// Hinted with the round's other wires, in the proof's child order.
    Hinted(Vec<WrapDigest>),
    /// Left in the arena from word `at`, for its walk to hint
    /// ([`ChainShape::paths_rotated`]).
    Arena {
        arena: super::instr::ArenaId,
        at: u32,
    },
}

impl PathStore {
    fn siblings(&self) -> Siblings<'_> {
        match self {
            PathStore::Hinted(path) => Siblings::Child(path),
            PathStore::Arena { arena, at } => Siblings::Rotated {
                arena: *arena,
                at: *at,
            },
        }
    }
}

/// A current block's wires, in the field its round holds them in.
pub enum CurrentBlock {
    Base(Vec<Felt>),
    /// Round 0's base values hinted four a word, and the felts their
    /// `Unpack`s write ([`ChainShape::round0_packed`]).
    BaseWords {
        words: Vec<Cell>,
        felts: Vec<Felt>,
    },
    Ext(Vec<Ext>),
}

impl CurrentBlock {
    fn as_block(&self) -> BlockValues<'_> {
        match self {
            CurrentBlock::Base(v) => BlockValues::Base(v),
            CurrentBlock::BaseWords { words, felts } => BlockValues::BaseWords { words, felts },
            CurrentBlock::Ext(v) => BlockValues::Ext(v),
        }
    }
}

/// ★ One chain's round wires, hinted and OWNED — because `ChainRoundWires`
/// borrows them.
///
/// Extracted from [`chain_program`] so a caller that emits SEVERAL chains in one
/// program — `stacked_eval`'s wrapper, one chain per stacked polynomial — builds
/// them from this walk rather than from a second copy of it. The chain suite is
/// the control that the extraction moved nothing.
pub struct RoundStorage {
    shape: ChainShape,
    sumchecks: Vec<Vec<Vec<Ext>>>,
    currents: Vec<Vec<(CurrentBlock, PathStore)>>,
    nexts: Vec<Vec<(Vec<Ext>, PathStore)>>,
    roots: Vec<Option<super::builder::Cell>>,
    oods: Vec<Option<Ext>>,
    nonces: Vec<RoundNonces>,
    /// Per round: the current tree's cap when the round owns it (round 0),
    /// and the successor's cap. Empty where the tree is uncapped.
    current_caps: Vec<Vec<WrapDigest>>,
    next_caps: Vec<Vec<WrapDigest>>,
}

impl RoundStorage {
    /// Words one chain's rounds occupy, which is what a caller placing several
    /// chains in one arena advances by.
    pub fn words(shape: &ChainShape) -> u32 {
        (0..shape.rounds()).map(|r| round_words(shape, r)).sum()
    }

    /// Hints every round's wires out of `arena`, starting at `base`, in the
    /// order [`push_round_words`] writes them.
    pub fn hint(
        b: &mut LfmBuilder,
        arena: super::instr::ArenaId,
        base: u32,
        shape: &ChainShape,
    ) -> Self {
        let mut sumchecks: Vec<Vec<Vec<Ext>>> = Vec::new();
        let mut currents: Vec<Vec<(CurrentBlock, PathStore)>> = Vec::new();
        let mut nexts: Vec<Vec<(Vec<Ext>, PathStore)>> = Vec::new();
        let mut roots: Vec<Option<super::builder::Cell>> = Vec::new();
        let mut oods: Vec<Option<Ext>> = Vec::new();
        let mut nonces: Vec<RoundNonces> = Vec::new();
        let mut current_caps: Vec<Vec<WrapDigest>> = Vec::new();
        let mut next_caps: Vec<Vec<WrapDigest>> = Vec::new();

        let mut at = base;
        for r in 0..shape.rounds() {
            let next_word = |b: &mut LfmBuilder, at: &mut u32| {
                let cell = b.hint_word(arena, *at);
                *at += 1;
                cell
            };
            // A path of `words` words: hinted now, or left for its walk.
            let next_path = |b: &mut LfmBuilder, at: &mut u32, words: usize| {
                if shape.paths_rotated() {
                    let path = PathStore::Arena { arena, at: *at };
                    *at += words as u32;
                    path
                } else {
                    PathStore::Hinted(
                        (0..words)
                            .map(|_| WrapDigest::from_cell(next_word(b, at)))
                            .collect(),
                    )
                }
            };
            let k = shape.schedule[r];
            let sumcheck: Vec<Vec<Ext>> = (0..shape.messages(r))
                .map(|_| (0..2).map(|_| next_word(b, &mut at).as_ext()).collect())
                .collect();
            // The nonces are FELTS: `append_bytes(&nonce.to_be_bytes())` is one
            // big-endian felt, which is how the grind absorbs them. One word a
            // slot the layout carries, in slot order.
            let [folding, ood_nonce, query] = shape.carries(r).map(|carried| {
                carried.then(|| {
                    let felt = b.hint_felt(arena, at);
                    at += 1;
                    felt
                })
            });

            // W1: tree 0's cap, right after round 0's nonces (the words the
            // owner path carried after its siblings) — a grouped round 0's
            // trees' caps, in tree order.
            let current_cap: Vec<WrapDigest> = if r == 0 {
                (0..shape.trees * shape.cap_words(shape.current_depth(0), shape.caps[0]))
                    .map(|_| WrapDigest::from_cell(next_word(b, &mut at)))
                    .collect()
            } else {
                Vec::new()
            };
            let depth = shape.current_path(r);
            let block = 1usize << k;
            // ★ ROUND 0's current codeword is BASE on the host
            // (`whir_chain.rs:983`), so its block hashes ONE felt a value and
            // not three. Hinting it as extension wires would hash forty-eight
            // felts where the committer hashed sixteen and the root would never
            // match — which is exactly how this test first failed.
            // A grouped round 0: one block and path a tree, query-major.
            // A tail round: every leaf's block once, in leaf order, no path.
            let openings = if shape.tail_round(r) {
                shape.current_leaves(r)
            } else {
                shape.num_queries * shape.trees_at(r)
            };
            let current: Vec<(CurrentBlock, PathStore)> = (0..openings)
                .map(|_| {
                    let values = if r == 0 && shape.round0_packed() {
                        // Four values a word: the leaf hash takes the words,
                        // the fold the felts their `Unpack`s write.
                        let words: Vec<Cell> = (0..shape.round0_block_words(block))
                            .map(|_| next_word(b, &mut at))
                            .collect();
                        let felts: Vec<Felt> = words
                            .iter()
                            .flat_map(|w| b.unpack(*w))
                            .take(block)
                            .collect();
                        CurrentBlock::BaseWords { words, felts }
                    } else if r == 0 {
                        let felts: Vec<Felt> = (0..block)
                            .map(|_| {
                                let f = b.hint_felt(arena, at);
                                at += 1;
                                f
                            })
                            .collect();
                        CurrentBlock::Base(felts)
                    } else {
                        CurrentBlock::Ext(
                            (0..block).map(|_| next_word(b, &mut at).as_ext()).collect(),
                        )
                    };
                    let path = if shape.tail_round(r) {
                        PathStore::Hinted(Vec::new())
                    } else {
                        next_path(b, &mut at, depth)
                    };
                    (values, path)
                })
                .collect();

            let (next_root, ood_value, next, next_cap) = match shape.next_depth(r) {
                Some(next_depth) => {
                    let nr = next_word(b, &mut at);
                    let ov = next_word(b, &mut at).as_ext();
                    // W1: the successor's cap, after its root and ood value.
                    let next_cap: Vec<WrapDigest> = (0..shape
                        .cap_words(next_depth, shape.caps[r + 1]))
                        .map(|_| WrapDigest::from_cell(next_word(b, &mut at)))
                        .collect();
                    let next_depth = shape.path_words(next_depth, shape.caps[r + 1]);
                    let next_block = 1usize << shape.schedule[r + 1];
                    // A tail round opens no successor block a query.
                    let next_openings = if shape.tail_round(r) {
                        0
                    } else {
                        shape.num_queries
                    };
                    let next: Vec<(Vec<Ext>, PathStore)> = (0..next_openings)
                        .map(|_| {
                            let values: Vec<Ext> = (0..next_block)
                                .map(|_| next_word(b, &mut at).as_ext())
                                .collect();
                            let path = next_path(b, &mut at, next_depth);
                            (values, path)
                        })
                        .collect();
                    (Some(nr), Some(ov), next, next_cap)
                }
                None => (None, None, Vec::new(), Vec::new()),
            };

            sumchecks.push(sumcheck);
            currents.push(current);
            nexts.push(next);
            roots.push(next_root);
            oods.push(ood_value);
            nonces.push(RoundNonces {
                folding,
                ood: ood_nonce,
                query,
            });
            current_caps.push(current_cap);
            next_caps.push(next_cap);
        }
        assert_eq!(
            at - base,
            Self::words(shape),
            "the round walk and the word count are one derivation"
        );

        Self {
            shape: shape.clone(),
            sumchecks,
            currents,
            nexts,
            roots,
            oods,
            nonces,
            current_caps,
            next_caps,
        }
    }

    /// The query openings, current and successor, borrowing this storage.
    #[allow(clippy::type_complexity)]
    pub fn openings(&self) -> (Vec<Vec<QueryOpening<'_>>>, Vec<Vec<QueryOpening<'_>>>) {
        let current: Vec<Vec<QueryOpening<'_>>> = self
            .currents
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|(values, path)| QueryOpening {
                        values: values.as_block(),
                        siblings: path.siblings(),
                    })
                    .collect()
            })
            .collect();
        let next: Vec<Vec<QueryOpening<'_>>> = self
            .nexts
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|(values, path)| QueryOpening {
                        values: BlockValues::Ext(values),
                        siblings: path.siblings(),
                    })
                    .collect()
            })
            .collect();
        (current, next)
    }

    /// The wires `emit_verify_weighted` takes, borrowing this storage and the
    /// openings built from it.
    pub fn wires<'a>(
        &'a self,
        current: &'a [Vec<QueryOpening<'a>>],
        next: &'a [Vec<QueryOpening<'a>>],
    ) -> Vec<ChainRoundWires<'a>> {
        (0..self.shape.rounds())
            .map(|r| ChainRoundWires {
                sumcheck: &self.sumchecks[r],
                next_root: self.roots[r],
                ood_value: self.oods[r],
                nonces: self.nonces[r],
                current: &current[r],
                next: &next[r],
                current_cap: &self.current_caps[r],
                next_cap: &self.next_caps[r],
            })
            .collect()
    }
}

pub fn round_words(shape: &ChainShape, r: usize) -> u32 {
    let k = shape.schedule[r];
    // The sumcheck's two evaluations a round, the nonces the layout carries,
    // and per query the current block plus its path (to the cap, when the tree
    // has one).
    let nonces = shape.carries(r).iter().filter(|&&carried| carried).count();
    let mut n = (2 * shape.messages(r) + nonces) as u32;
    let depth = shape.current_path(r);
    let block = 1usize << k;
    if shape.tail_round(r) {
        // A tail round: every leaf's block once, in leaf order, and no path.
        n += (shape.current_leaves(r) * block) as u32;
    } else {
        // A grouped round 0 opens each of its trees at every query; round 0's
        // base values four a word when packed.
        let words = if r == 0 {
            shape.round0_block_words(block)
        } else {
            block
        };
        n += (shape.num_queries * shape.trees_at(r) * (words + depth)) as u32;
    }
    if r == 0 {
        // W1: tree 0's cap words, a grouped round 0's trees' each.
        n += (shape.trees * shape.cap_words(shape.current_depth(0), shape.caps[0])) as u32;
    }
    if let Some(next_depth) = shape.next_depth(r) {
        // The successor root, its out-of-domain value, its cap, and per query
        // its block and path.
        let c = shape.caps[r + 1];
        n += 2 + shape.cap_words(next_depth, c) as u32;
        // A tail round reads its successor's blocks where that tail round
        // hints them, so it opens none a query.
        if !shape.tail_round(r) {
            let next_block = 1usize << shape.schedule[r + 1];
            n += (shape.num_queries * (next_block + shape.path_words(next_depth, c))) as u32;
        }
    }
    n
}

/// One chain's round wires, in the order [`RoundStorage::hint`] reads them.
///
/// Split out of [`chain_arena`] for the same reason [`RoundStorage`] was split
/// out of [`chain_program`]: a program holding several chains fills one arena
/// with several of these, and a second copy of the order would be a second thing
/// to keep in step with the walk that reads it.
pub fn push_round_words(
    words: &mut Vec<LfmWord>,
    shape: &ChainShape,
    proof: &ChainProof<GoldilocksField, GoldilocksExtension>,
) {
    // A proof [`try_push_round_words`] cannot lay out leaves the arena short,
    // and its length (a verifier constant) refuses it.
    let _ = try_push_round_words(words, shape, proof);
}

/// [`push_round_words`], refusing a proof whose tail round
/// ([`ChainShape::tail_round`]) misses a leaf's block: every leaf of a tail
/// tree is opened hundreds of times by an honest proof, so a missing one is a
/// completeness failure (probability ≈ 4·(3/4)^(2Q)), and the words written
/// before it stand.
pub fn try_push_round_words(
    words: &mut Vec<LfmWord>,
    shape: &ChainShape,
    proof: &ChainProof<GoldilocksField, GoldilocksExtension>,
) -> Result<(), String> {
    for (r, round) in proof.rounds.iter().enumerate() {
        for sc in &round.sumcheck {
            for e in &sc.evaluations {
                words.push(ext_word(e));
            }
        }
        // The slots the layout carries. A slot it does not carry has no word,
        // so its host field, which the host verifier requires to be 0 under
        // `Spent`, has no way into the machine.
        let nonces = [round.nonces.folding, round.nonces.ood, round.nonces.query];
        for (nonce, carried) in nonces.into_iter().zip(shape.carries(r)) {
            if carried {
                words.push([FE::from(nonce), FE::zero(), FE::zero(), FE::zero()]);
            }
        }
        // W1: round 0 owns tree 0, whose cap rides at the end of its first
        // current path; it goes to the arena here and the path goes without it.
        let current = PathSide {
            depth: shape.current_depth(r),
            cap: shape.caps[r],
            owner: r == 0,
        };
        if shape.tail_round(r) {
            push_tail_blocks(words, shape, proof, r)?;
        } else {
            push_openings(words, shape, round, true, current);
        }
        if let Some(next_depth) = shape.next_depth(r) {
            words.push(commitment_to_digest(
                round.next_root.as_ref().expect("a successor root"),
            ));
            words.push(ext_word(
                round.ood_value.as_ref().expect("an out-of-domain value"),
            ));
            let next = PathSide {
                depth: next_depth,
                cap: shape.caps[r + 1],
                owner: true,
            };
            if shape.tail_round(r) {
                // The successor's cap, for its tail round to authenticate; its
                // blocks are that round's own.
                for node in tail_cap(shape, round, r + 1)? {
                    words.push(commitment_to_digest(&node));
                }
            } else {
                push_openings(words, shape, round, false, next);
            }
        }
    }
    Ok(())
}

/// The successor openings round `r` carries, extension-valued at every round.
fn next_openings(
    round: &ChainRound<GoldilocksField, GoldilocksExtension>,
) -> &[multilinear::whir_commit::CosetOpening<GoldilocksExtension>] {
    match &round.openings {
        RoundOpenings::Base(p) => &p.next,
        RoundOpenings::Extension(p) => &p.next,
        RoundOpenings::BaseGrouped(p) => &p.next,
    }
}

/// Tail round `t`'s tree's cap: the end of the first successor path of the
/// round before (`round`), its owner (W1). Every leaf of a tail tree is a cap
/// node, so the cap is the leaves' digests in leaf order.
fn tail_cap(
    shape: &ChainShape,
    round: &ChainRound<GoldilocksField, GoldilocksExtension>,
    t: usize,
) -> Result<Vec<Commitment>, String> {
    let owner = next_openings(round)
        .first()
        .ok_or_else(|| format!("tail round {t}: no successor opening carries its cap"))?;
    let (siblings, cap) = super::merkle_cap::child_path_to_cap(
        &owner.proof.merkle_path,
        shape.current_depth(t),
        shape.caps[t],
        shape.arity,
        true,
    )?;
    match cap {
        Some(cap) if siblings.is_empty() && cap.len() == shape.current_leaves(t) => Ok(cap),
        _ => Err(format!(
            "tail round {t}: the owner path is not a cap over every leaf"
        )),
    }
}

/// ★ Tail round `r`'s blocks, one a leaf in leaf order ([`ChainShape::tail_round`]).
///
/// The proof opens the tree at sampled positions only, and the host keeps no
/// position. Each distinct opened block — round `r − 1`'s successor openings
/// and round `r`'s own — is placed by its leaf hash among the cap's nodes, which
/// are the leaves' digests: no index is needed, and the arena stays a function
/// of the proof. A leaf no opening hits is a completeness failure, refused.
/// The leaf checks every placed block against its cap node, so a misplaced one
/// has no execution.
fn push_tail_blocks(
    words: &mut Vec<LfmWord>,
    shape: &ChainShape,
    proof: &ChainProof<GoldilocksField, GoldilocksExtension>,
    r: usize,
) -> Result<(), String> {
    use crypto::merkle_tree::traits::IsMerkleTreeBackend;
    type Leaf = <crate::hash_pin::BlockWhirHash as multilinear::whir_hash::WhirHash>::Backend<
        GoldilocksExtension,
    >;
    let before = &proof.rounds[r - 1];
    let cap = tail_cap(shape, before, r)?;
    let current: &[multilinear::whir_commit::CosetOpening<GoldilocksExtension>] =
        match &proof.rounds[r].openings {
            RoundOpenings::Extension(p) => &p.current,
            _ => {
                return Err(format!(
                    "tail round {r}: its openings are not extension-valued"
                ));
            }
        };
    let block = 1usize << shape.schedule[r];
    let mut placed: Vec<Option<&[math::field::element::FieldElement<GoldilocksExtension>]>> =
        vec![None; cap.len()];
    for opening in next_openings(before).iter().chain(current) {
        if opening.values.len() != block {
            return Err(format!(
                "tail round {r}: a block of {} values",
                opening.values.len()
            ));
        }
        let digest = Leaf::hash_data(&opening.values);
        for (slot, node) in placed.iter_mut().zip(&cap) {
            if slot.is_none() && *node == digest {
                *slot = Some(&opening.values);
            }
        }
    }
    for (j, slot) in placed.iter().enumerate() {
        let values = slot.ok_or_else(|| format!("tail round {r}: no opening of leaf {j}"))?;
        for v in values {
            words.push(ext_word(v));
        }
    }
    Ok(())
}

/// One side's tree, as its openings' paths are laid out: its index bits, its
/// cap height, and whether its first opening is the tree's owner (carries the
/// cap after its siblings, W1).
#[derive(Clone, Copy)]
struct PathSide {
    depth: usize,
    cap: usize,
    owner: bool,
}

/// One round's query openings, current or successor, block then path.
///
/// The side's FIRST opening, when it is the owner, carries the cap's words at
/// the end of its path (the owner-path encoding, W1): they are written first,
/// ahead of every block, and the path is written without them. No cap at the
/// default. A malformed path is not repaired here: its words are written as
/// they are, and the arena's length (a verifier constant) refuses it.
///
/// At arity 4 each path keeps the proof's child order and an uncapped
/// odd-depth top keeps only its partner ([`super::merkle_cap::child_path_to_cap`]).
fn push_openings(
    words: &mut Vec<LfmWord>,
    shape: &ChainShape,
    round: &ChainRound<GoldilocksField, GoldilocksExtension>,
    current: bool,
    tree: PathSide,
) {
    let owner_cap = if tree.owner {
        shape.cap_words(tree.depth, tree.cap)
    } else {
        0
    };
    if shape.arity == 4 {
        push_openings4(words, round, current, tree, shape.round0_packed());
        return;
    }
    fn side<V>(
        words: &mut Vec<LfmWord>,
        openings: &[multilinear::whir_commit::CosetOpening<V>],
        owner_cap: usize,
        value_word: impl Fn(&math::field::element::FieldElement<V>) -> LfmWord,
    ) where
        V: math::field::traits::IsField,
    {
        let owner_split = openings
            .first()
            .map_or(0, |o| o.proof.merkle_path.len().saturating_sub(owner_cap));
        if let Some(owner) = openings.first() {
            for node in &owner.proof.merkle_path[owner_split..] {
                words.push(commitment_to_digest(node));
            }
        }
        for (i, opening) in openings.iter().enumerate() {
            for v in &opening.values {
                words.push(value_word(v));
            }
            let path = if i == 0 {
                &opening.proof.merkle_path[..owner_split]
            } else {
                &opening.proof.merkle_path[..]
            };
            for node in path {
                words.push(commitment_to_digest(node));
            }
        }
    }
    match &round.openings {
        RoundOpenings::Base(p) => {
            if current {
                // A base value arrives as `(v, 0, 0, 0)`.
                side(words, &p.current, owner_cap, |v| {
                    [*v, FE::zero(), FE::zero(), FE::zero()]
                });
            } else {
                side(words, &p.next, owner_cap, ext_word);
            }
        }
        RoundOpenings::Extension(p) => {
            let openings = if current { &p.current } else { &p.next };
            side(words, openings, owner_cap, ext_word);
        }
        // A grouped round 0 (D-ONECHAIN): every tree's cap off its owner
        // (query 0's opening of it), in tree order; then per query each
        // tree's block and path.
        RoundOpenings::BaseGrouped(p) => {
            if current {
                let owner_splits: Vec<usize> = p
                    .current
                    .first()
                    .map(|row| {
                        row.iter()
                            .map(|owner| {
                                let path = &owner.proof.merkle_path;
                                let split = path.len().saturating_sub(owner_cap);
                                for node in &path[split..] {
                                    words.push(commitment_to_digest(node));
                                }
                                split
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                for (i, row) in p.current.iter().enumerate() {
                    for (tree, opening) in row.iter().enumerate() {
                        for v in &opening.values {
                            words.push([*v, FE::zero(), FE::zero(), FE::zero()]);
                        }
                        let path = &opening.proof.merkle_path;
                        let path = match (i, owner_splits.get(tree)) {
                            (0, Some(&split)) => &path[..split],
                            _ => &path[..],
                        };
                        for node in path {
                            words.push(commitment_to_digest(node));
                        }
                    }
                }
            } else {
                side(words, &p.next, owner_cap, ext_word);
            }
        }
    }
}

/// Round 0's base block: a `(v, 0, 0, 0)` word a value, or four values a word,
/// the last zero-filled, when `packed` ([`ChainShape::round0_packed`]).
fn push_base_block(words: &mut Vec<LfmWord>, values: &[FE], packed: bool) {
    if packed {
        for lanes in values.chunks(4) {
            let mut word = [FE::zero(); 4];
            word[..lanes.len()].copy_from_slice(lanes);
            words.push(word);
        }
    } else {
        words.extend(
            values
                .iter()
                .map(|v| [*v, FE::zero(), FE::zero(), FE::zero()]),
        );
    }
}

/// [`push_openings`] at arity 4: the owner's cap, then per opening its block
/// and its path in child order, cut to the cap. Only a current base side is
/// ever `packed`.
fn push_openings4(
    words: &mut Vec<LfmWord>,
    round: &ChainRound<GoldilocksField, GoldilocksExtension>,
    current: bool,
    tree: PathSide,
    packed: bool,
) {
    fn side<V>(
        words: &mut Vec<LfmWord>,
        openings: &[multilinear::whir_commit::CosetOpening<V>],
        tree: PathSide,
        block: impl Fn(&mut Vec<LfmWord>, &[math::field::element::FieldElement<V>]),
    ) where
        V: math::field::traits::IsField,
    {
        let laid: Vec<(Vec<_>, Option<Vec<_>>)> = openings
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let path = &o.proof.merkle_path;
                super::merkle_cap::child_path_to_cap(
                    path,
                    tree.depth,
                    tree.cap,
                    4,
                    tree.owner && i == 0,
                )
                // A path of another shape goes in as it is: the arena's length
                // refuses it.
                .unwrap_or_else(|_| (path.clone(), None))
            })
            .collect();
        if let Some((_, Some(cap))) = laid.first() {
            for node in cap {
                words.push(commitment_to_digest(node));
            }
        }
        for (opening, (path, _)) in openings.iter().zip(&laid) {
            block(words, &opening.values);
            for node in path {
                words.push(commitment_to_digest(node));
            }
        }
    }
    let ext_block = |words: &mut Vec<LfmWord>, values: &[FEE]| {
        words.extend(values.iter().map(ext_word));
    };
    match &round.openings {
        RoundOpenings::Base(p) => {
            if current {
                side(words, &p.current, tree, |words, values| {
                    push_base_block(words, values, packed)
                });
            } else {
                side(words, &p.next, tree, ext_block);
            }
        }
        RoundOpenings::Extension(p) => {
            let openings = if current { &p.current } else { &p.next };
            side(words, openings, tree, ext_block);
        }
        // See `push_openings`: the trees' caps in tree order, then per query
        // each tree's block and path.
        RoundOpenings::BaseGrouped(p) => {
            if current {
                let laid: Vec<Vec<(Vec<_>, Option<Vec<_>>)>> = p
                    .current
                    .iter()
                    .enumerate()
                    .map(|(i, row)| {
                        row.iter()
                            .map(|o| {
                                let path = &o.proof.merkle_path;
                                super::merkle_cap::child_path_to_cap(
                                    path,
                                    tree.depth,
                                    tree.cap,
                                    4,
                                    tree.owner && i == 0,
                                )
                                .unwrap_or_else(|_| (path.clone(), None))
                            })
                            .collect()
                    })
                    .collect();
                for (_, cap) in laid.first().into_iter().flatten() {
                    for node in cap.iter().flatten() {
                        words.push(commitment_to_digest(node));
                    }
                }
                for (row, laid_row) in p.current.iter().zip(&laid) {
                    for (opening, (path, _)) in row.iter().zip(laid_row) {
                        push_base_block(words, &opening.values, packed);
                        for node in path {
                            words.push(commitment_to_digest(node));
                        }
                    }
                }
            } else {
                side(words, &p.next, tree, ext_block);
            }
        }
    }
}

// =============================================================================
// A program's rows by chip — what a Poseidon1 leaf's partition bounds
// =============================================================================

/// Rows a leaf part spends in each chip of the LFM machine, one a chip's
/// instruction (`exec_schedule`'s census holds a program's chips to its
/// instruction counts), so a leaf's padded heights follow from its sum: what a
/// Poseidon1 leaf's partition bounds ([`super::whir_block::leaf_partition_rows`]).
/// `LFM_RANGE` is the one chip no instruction opens — a fixed table, the same in
/// every leaf. `consts` counts distinct constants (the builder interns them), so
/// a leaf's is at most its parts' sum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChipRows {
    /// The socket (`Hash` or `Hash16`).
    pub hash: usize,
    pub select: usize,
    pub balu: usize,
    pub xalu: usize,
    /// `Pack` and `Unpack`.
    pub lanes: usize,
    /// The hinted words.
    pub hint: usize,
    pub bitdec: usize,
    pub consts: usize,
    pub public: usize,
    /// `KeccakF` and `Blake3`: none in a WHIR leaf.
    pub accel: usize,
}

impl ChipRows {
    /// The chips' names, in [`Self::to_array`]'s order.
    pub const NAMES: [&'static str; 10] = [
        "socket",
        "Select",
        "base-ALU",
        "ext-ALU",
        "lanes",
        "hint",
        "bit-dec",
        "const",
        "public",
        "accelerator",
    ];

    pub fn to_array(self) -> [usize; 10] {
        [
            self.hash,
            self.select,
            self.balu,
            self.xalu,
            self.lanes,
            self.hint,
            self.bitdec,
            self.consts,
            self.public,
            self.accel,
        ]
    }

    pub fn from_array(a: [usize; 10]) -> Self {
        let [
            hash,
            select,
            balu,
            xalu,
            lanes,
            hint,
            bitdec,
            consts,
            public,
            accel,
        ] = a;
        Self {
            hash,
            select,
            balu,
            xalu,
            lanes,
            hint,
            bitdec,
            consts,
            public,
            accel,
        }
    }

    pub fn plus(self, other: Self) -> Self {
        let (a, b) = (self.to_array(), other.to_array());
        Self::from_array(core::array::from_fn(|c| a[c] + b[c]))
    }

    /// Whether every chip's rows are at most `heights`'.
    pub fn under(self, heights: Self) -> bool {
        self.to_array()
            .iter()
            .zip(heights.to_array())
            .all(|(&r, h)| r <= h)
    }

    /// The rows by chip of some emitted instructions.
    pub fn of_instrs(instrs: &[super::instr::Instr]) -> Self {
        use super::instr::Instr;
        let mut rows = Self::default();
        for instr in instrs {
            let chip = match instr {
                Instr::Hash { .. } | Instr::Hash16(_) => &mut rows.hash,
                Instr::Select { .. } => &mut rows.select,
                Instr::BaseAlu { .. } => &mut rows.balu,
                Instr::ExtAlu { .. } => &mut rows.xalu,
                Instr::Pack { .. } | Instr::Unpack { .. } => &mut rows.lanes,
                Instr::Hint { .. } | Instr::HintOn { .. } => &mut rows.hint,
                Instr::BitDec { .. } => &mut rows.bitdec,
                Instr::Const { .. } => &mut rows.consts,
                Instr::Public { .. } => &mut rows.public,
                Instr::KeccakF(_) | Instr::Blake3(_) => &mut rows.accel,
            };
            *chip += 1;
        }
        rows
    }

    /// The rows by chip of a compiled program.
    pub fn of(program: &super::compiler::LfmProgram) -> Self {
        Self::of_instrs(&program.instrs)
    }
}

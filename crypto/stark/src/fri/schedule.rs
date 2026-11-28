//! The FRI fold schedule: which fold exponents the committed FRI layers use.
//!
//! Today every committed FRI layer folds by 2 (a pair leaf). A fold schedule
//! `[d_1, .., d_m]` generalises that: committed layer `j` folds by `2^{d_j}`
//! (a group leaf of `2^{d_j}` extension values), and `Σ d_j` covers the bits
//! between the first committed layer and the terminal codeword. The all-ones
//! schedule is today's protocol exactly.
//!
//! The schedule is a **format constant**: the prover, the verifier and the
//! in-guest verifier must derive the same one from public shape parameters
//! only, never from a proof. So the dynamic program below is integer-only,
//! with a fixed tie rule, and its inputs are all public:
//!
//! * `b0` — log2 length of the first committed layer (`lde_log − 1` when fold 0
//!   is the uncommitted binary fold of the trace pair, `lde_log` when the DEEP
//!   codeword itself is committed);
//! * `terminal_log` — log2 length of the terminal codeword;
//! * `num_queries` — FRI query count (every FRI tree is opened once per query);
//! * the active Merkle-cap policy ([`CapPolicy`]; `Off` caps nothing);
//! * `dmax` — the largest fold exponent the program may choose.
//!
//! # The objective: the cost law of every emitted row
//!
//! The DP minimises the in-guest verifier's price of the FRI leg under the
//! SAME cost-law weights the cap policy optimises ([`AUTO_WEIGHTS`], ns per
//! row from the node law 421 ns/instruction + 5.63 ns/cell and each chip's
//! committed width). Per query per committed layer it prices EVERY row the
//! in-guest group-layer emitter (`prover/src/lfm/fri.rs::emit_group_layer`)
//! and its opening's hints emit — [`fri_group_layer_rows`], at a tree of
//! `depth` levels (uncapped):
//!
//! ```text
//! leaf(d)·compress                        absorb the 2^d-value group leaf
//! + depth·(compress + select)             the authentication walk (a Select and a compression per level)
//! + (2^d − 1)·select                      the slot mux picking the query's value out of the group
//! + 2·XALU                                the slot check (assert_eq_ext: esub + ediv)
//! + (2^d − 1)·fold                        the group fold: 2^d − 1 binary folds (5 XALU each)
//! + d·twiddle                             the twiddle chain: one base mul per fold level
//! + d·(select + BALU)                     x_g⁻¹ = y⁻¹·ω^{br(slot)}: a constant Select and a base mul per slot bit
//! + max(0, d − 2)·XALU + [d ≥ 2]·BALU     fold-level scaling (emul_base per level of > 2 pairs; one base mul at 2 pairs)
//! + 8·BALU + 1·unpack                     the root compare (walked digest unpacked, four lowered asserts)
//! + 2^d·unpack + 2^d·hint                 the group's values: hinted, unpacked into the leaf
//! + packs(d)·unpack                       the leaf's 3·2^d felts packed four to a word (LFM_LANES rows)
//! + depth·hint                            the path's siblings
//! − cap_gain(Q, c(depth)) / Q − c·hint    what the tree's cap saves, per query (0 without a cap):
//!                                         the cap policy's own gain, plus the c sibling hints a
//!                                         capped path does not carry
//! ```
//!
//! `leaf(d) = max(1, ⌈3·2^d / 8⌉)` (an ext3 group at the RPX rate of 8 felts);
//! the digest is the production one-cell (algebraic) digest. Each row kind is
//! priced at one weight: `SELECT`, `LFM_HASH` (compress), `Unpack` and hint
//! at the cap policy's (a `Pack` is an `LFM_LANES` row, as an `Unpack` is,
//! and is priced like one), `XALU` at [`XALU_ROW_NS`], `BALU` at [`BALU_ROW_NS`].
//! The in-guest emitter's tests pin "emitted rows == [`fri_group_layer_rows`]" kind by
//! kind against its emitter (`lfm::fri_group_tests`), capped and uncapped.
//! Costs are kept in units of `1/Q` ns so every term is an integer.

use crypto::merkle_tree::cap::{AUTO_WEIGHTS, CapPolicy, CapWeights, cap_gain};

use crate::proof::options::{FriMode, FriScheduleOverride, ProofOptions};

/// Largest fold exponent the schedule may choose (a 64-value group leaf).
pub const FRI_SCHEDULE_DMAX: u32 = 6;

/// Leaf absorption rate of the cost model, in base-field elements per
/// permutation (the RPX sponge rate).
pub const FRI_LEAF_RATE_FELTS: u64 = 8;

/// Extension degree of the FRI codeword values.
pub const FRI_EXTENSION_DEGREE: u64 = 3;

/// `XALU` rows of one binary FRI fold in-guest: `edsl::fri_fold` emits
/// `eadd, esub, emul, emul_base, eadd`.
pub const FRI_FOLD_XALU_ROWS: u64 = 5;

/// `BALU` rows of one step of the twiddle chain in-guest (one base `mul`).
pub const FRI_TWIDDLE_BALU_ROWS: u64 = 1;

/// `SELECT` rows of one two-way select of the slot mux (an ext value is one
/// cell, so one `Select` instruction).
pub const FRI_SLOT_SELECT_ROWS: u64 = 1;

/// `XALU` rows of the slot check: `assert_eq_ext` lowers to an `esub` and an
/// `ediv` by zero.
pub const FRI_SLOT_ASSERT_XALU_ROWS: u64 = 2;

/// `SELECT` rows of one slot bit of the `x_g` derivation (`x_g⁻¹ =
/// y⁻¹·ω_{2^d}^{br(slot)}`): the bit picks `1` or a constant.
pub const FRI_XG_SELECT_ROWS: u64 = 1;

/// `BALU` rows of one slot bit of the `x_g` derivation (one base `mul`).
pub const FRI_XG_BALU_ROWS: u64 = 1;

/// `BALU` rows of one opening's root (or cap-node) compare at the production
/// one-cell digest: four lowered `assert_eq`s, a `sub` and a `div` each.
pub const FRI_ROOT_COMPARE_BALU_ROWS: u64 = 8;

/// `Unpack` rows of one opening's root compare: the walked digest's lanes (a
/// capped compare unpacks the muxed cap node too, which the cap's gain prices).
pub const FRI_ROOT_COMPARE_UNPACK_ROWS: u64 = 1;

/// Felts one `Pack` row assembles into a word for the algebraic leaf sponge.
pub const FRI_LEAF_PACK_FELTS: u64 = 4;

/// Cost-law price (ns) of one `XALU` row: 421 + 5.63 × 18 committed cells
/// (the `LFM_XALU` cliff in the census, `+18874368` cells per `2^20` rows).
pub const XALU_ROW_NS: u64 = 522;

/// Cost-law price (ns) of one `BALU` row: 421 + 5.63 × 10 committed cells
/// (the `LFM_BALU` cliff, `+5242880` cells per `2^19` rows).
pub const BALU_ROW_NS: u64 = 477;

/// The per-row prices the schedule DP weighs. `cap` is the cap policy's own
/// weights, so the two levers optimise one objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FriCostWeights {
    /// Compression, select, unpack, hint and compare prices (the cap policy's).
    pub cap: CapWeights,
    /// One binary fold in-guest ([`FRI_FOLD_XALU_ROWS`] `XALU` rows).
    pub fold: u64,
    /// One step of the twiddle chain in-guest ([`FRI_TWIDDLE_BALU_ROWS`] `BALU` rows).
    pub twiddle: u64,
    /// One `XALU` row.
    pub xalu: u64,
    /// One `BALU` row.
    pub balu: u64,
}

/// The weights the schedule DP optimises. ⚠ A FORMAT CONSTANT: changing any of
/// them can change the schedule, and so the proofs, of every table under
/// `LAMBDA_VM_ZF_FRI=dp`. Pinned by `fri_schedule_tests`.
pub const FRI_COST_WEIGHTS: FriCostWeights = FriCostWeights {
    cap: AUTO_WEIGHTS,
    fold: FRI_FOLD_XALU_ROWS * XALU_ROW_NS,
    twiddle: FRI_TWIDDLE_BALU_ROWS * BALU_ROW_NS,
    xalu: XALU_ROW_NS,
    balu: BALU_ROW_NS,
};

/// The rows one query's opening of one committed FRI layer emits in-guest, by
/// chip kind. `hashes` counts two-to-one compressions and leaf-absorption
/// permutations alike (both are `LFM_HASH` rows, priced `compress`); `packs`
/// and `unpacks` are both `LFM_LANES` rows, priced `unpack`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FriLayerRows {
    pub selects: u64,
    pub xalu: u64,
    pub balu: u64,
    pub hashes: u64,
    pub unpacks: u64,
    pub packs: u64,
    pub hints: u64,
}

impl FriLayerRows {
    /// The cost-law price (ns) of these rows under `weights`.
    pub fn price(&self, weights: &FriCostWeights) -> u64 {
        let w = &weights.cap;
        [
            (self.selects, w.select),
            (self.xalu, weights.xalu),
            (self.balu, weights.balu),
            (self.hashes, w.compress),
            (self.unpacks, w.unpack),
            (self.packs, w.unpack),
            (self.hints, w.hint),
        ]
        .iter()
        .fold(0u64, |acc, &(n, p)| acc.saturating_add(n.saturating_mul(p)))
    }
}

/// ★ The rows one query's opening of a GROUP layer (fold exponent `d`, tree
/// `depth` levels deep, cap height `cap_height`, clamped to the depth) emits
/// in-guest: `emit_group_layer` plus the opening's hints (see the module
/// docs; `lfm::fri_group_tests` pins these kind by kind against the emitter).
///
/// A cap of height `c` walks `depth − c` levels, hints `depth − c` siblings,
/// muxes the cap node (`2^c − 1` selects) and unpacks it for the compare.
pub fn fri_group_layer_rows(d: u32, depth: u32, cap_height: u32) -> FriLayerRows {
    let d = d.min(63);
    let n = 1u64 << d;
    let c = cap_height.min(depth).min(63);
    let walk = u64::from(depth - c);
    let cap_mux = (1u64 << c) - 1;
    let d64 = u64::from(d);
    FriLayerRows {
        selects: (n - 1) * FRI_SLOT_SELECT_ROWS + walk + cap_mux + d64 * FRI_XG_SELECT_ROWS,
        xalu: (n - 1) * FRI_FOLD_XALU_ROWS + FRI_SLOT_ASSERT_XALU_ROWS + d64.saturating_sub(2),
        balu: d64 * FRI_TWIDDLE_BALU_ROWS
            + d64 * FRI_XG_BALU_ROWS
            + u64::from(d >= 2)
            + FRI_ROOT_COMPARE_BALU_ROWS,
        hashes: fri_leaf_blocks(d) + walk,
        unpacks: n + FRI_ROOT_COMPARE_UNPACK_ROWS + u64::from(c > 0),
        packs: fri_leaf_packs(d),
        hints: n + walk,
    }
}

/// The rows one query's opening of a layer under TODAY's pair encoding
/// (`FriFormat::is_legacy`: one sibling value per layer, no slot check, no
/// `x_g`) emits in-guest (`prover/src/lfm/fri.rs::emit_pair_layer`): the
/// parity select, the pair leaf, the walk and compare, one squaring of the
/// point, one fold; hints: the sibling value and the path.
pub fn fri_pair_layer_rows(depth: u32, cap_height: u32) -> FriLayerRows {
    let c = cap_height.min(depth).min(63);
    let walk = u64::from(depth - c);
    FriLayerRows {
        selects: FRI_SLOT_SELECT_ROWS + walk + ((1u64 << c) - 1),
        xalu: FRI_FOLD_XALU_ROWS,
        balu: FRI_TWIDDLE_BALU_ROWS + FRI_ROOT_COMPARE_BALU_ROWS,
        hashes: fri_leaf_blocks(1) + walk,
        unpacks: 2 + FRI_ROOT_COMPARE_UNPACK_ROWS + u64::from(c > 0),
        packs: fri_leaf_packs(1),
        hints: 1 + walk,
    }
}

/// Log2 length of the first committed FRI layer for an LDE of `2^lde_log`.
///
/// Row-pair openings (`one_row == false`) consume the first fold uncommitted,
/// so the chain starts at `lde_log − 1`; one-row openings commit the DEEP
/// codeword itself, so the chain starts at `lde_log`.
pub fn fri_chain_start(lde_log: u32, one_row: bool) -> u32 {
    if one_row {
        lde_log
    } else {
        lde_log.saturating_sub(1)
    }
}

/// `Pack` rows that assemble one group leaf's `3·2^d` felts into words (four
/// per word, the tail zero-padded) before the sponge absorbs them.
pub fn fri_leaf_packs(d: u32) -> u64 {
    let felts = FRI_EXTENSION_DEGREE.saturating_mul(1u64.checked_shl(d).unwrap_or(u64::MAX));
    felts.div_ceil(FRI_LEAF_PACK_FELTS)
}

/// Permutations to absorb one group leaf of `2^d` extension values.
pub fn fri_leaf_blocks(d: u32) -> u64 {
    let felts = FRI_EXTENSION_DEGREE.saturating_mul(1u64.checked_shl(d).unwrap_or(u64::MAX));
    felts.div_ceil(FRI_LEAF_RATE_FELTS).max(1)
}

/// `Q ×` the per-query cost-law price (ns) of one committed GROUP layer of
/// fold exponent `d` whose tree has `depth` levels (the layer is
/// `2^{depth + d}` values long), under `weights` and the cap policy `cap`:
/// every emitted row ([`fri_group_layer_rows`]) of the uncapped opening, minus
/// the cap's gain and the sibling hints the cap removes. See the module docs.
pub fn fri_layer_cost_q(
    weights: &FriCostWeights,
    d: u32,
    depth: u32,
    num_queries: u64,
    cap: CapPolicy,
) -> u64 {
    layer_cost_q(
        weights,
        &fri_group_layer_rows(d, depth, 0),
        depth,
        num_queries,
        cap,
    )
}

/// `Q ×` the per-query price of one committed layer under TODAY's pair
/// encoding ([`fri_pair_layer_rows`]), under `weights` and `cap`.
pub fn fri_pair_layer_cost_q(
    weights: &FriCostWeights,
    depth: u32,
    num_queries: u64,
    cap: CapPolicy,
) -> u64 {
    layer_cost_q(
        weights,
        &fri_pair_layer_rows(depth, 0),
        depth,
        num_queries,
        cap,
    )
}

/// `Q × price(uncapped)` minus the cap's gain ([`cap_gain`], the cap policy's
/// own function) and the `c` sibling hints per query a capped path omits. In
/// i128, clamped into u64 (non-negative: a cap never saves more than the walk
/// it shortens, but the clamp keeps that a non-assumption).
fn layer_cost_q(
    weights: &FriCostWeights,
    uncapped: &FriLayerRows,
    depth: u32,
    num_queries: u64,
    cap: CapPolicy,
) -> u64 {
    let q = num_queries as i128;
    let per_query = uncapped.price(weights) as i128;
    let queries = usize::try_from(num_queries).unwrap_or(usize::MAX);
    let c = cap.height(queries, depth as usize);
    let hints_saved = q.saturating_mul(c as i128 * weights.cap.hint as i128);
    let total = q.saturating_mul(per_query) - cap_gain(&weights.cap, queries, c) - hints_saved;
    u64::try_from(total.max(0)).unwrap_or(u64::MAX)
}

/// `Q ×` the per-query cost of an arbitrary schedule starting at `b0` under a
/// per-layer cost function `layer_cost_q(d, depth)`, or `None` if a fold
/// exponent is zero or the schedule folds past zero bits.
pub fn fri_schedule_cost_by(
    b0: u32,
    schedule: &[u8],
    layer_cost_q: &dyn Fn(u32, u32) -> u64,
) -> Option<u64> {
    let mut b = b0;
    let mut cost = 0u64;
    for &d in schedule {
        let d = u32::from(d);
        if d == 0 {
            return None;
        }
        b = b.checked_sub(d)?;
        cost = cost.saturating_add(layer_cost_q(d, b));
    }
    Some(cost)
}

/// [`fri_schedule_cost_by`] under the production objective
/// ([`FRI_COST_WEIGHTS`], [`fri_layer_cost_q`]) — group layers.
pub fn fri_schedule_cost_q(
    b0: u32,
    schedule: &[u8],
    num_queries: u64,
    cap: CapPolicy,
) -> Option<u64> {
    fri_schedule_cost_by(b0, schedule, &|d, depth| {
        fri_layer_cost_q(&FRI_COST_WEIGHTS, d, depth, num_queries, cap)
    })
}

/// The optimum a schedule DP picks, with its cost.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FriScheduleChoice {
    /// `Q ×` the per-query cost (see the module docs).
    pub cost_q: u64,
    /// Number of committed trees (`schedule.len()`).
    pub trees: u32,
    /// Fold exponents, first committed layer first.
    pub schedule: Vec<u8>,
}

/// The schedule DP over an arbitrary per-layer cost `layer_cost_q(d, depth)`:
///
/// ```text
/// best(T)      = (0, 0, [])
/// best(b > T)  = min over d ∈ [1, min(dmax, b − T)] of
///                (layer_cost_q(d, b − d) + best(b − d).cost, best(b − d).trees + 1, [d] ++ best(b − d).sched)
/// ```
///
/// compared lexicographically on `(cost, trees)`; ties go to the smallest `d`
/// (the first reached). Equivalently, the result is the lexicographically
/// smallest schedule among the `(cost, trees)`-optimal ones. It lands exactly
/// on `terminal_log`: `Σ schedule == b0 − terminal_log`, and the schedule is
/// empty when `b0 ≤ terminal_log`. A `dmax` of 0 is treated as 1; `dmax` is
/// capped at 32.
pub fn fri_schedule_by(
    b0: u32,
    terminal_log: u32,
    dmax: u32,
    layer_cost_q: &dyn Fn(u32, u32) -> u64,
) -> FriScheduleChoice {
    if b0 <= terminal_log {
        return FriScheduleChoice {
            cost_q: 0,
            trees: 0,
            schedule: Vec::new(),
        };
    }
    let dmax = dmax.clamp(1, 32);
    let span = (b0 - terminal_log) as usize;
    // best[i] = optimum from b = terminal_log + i down to the terminal, stored
    // as (cost, trees, first fold exponent); the schedule is recovered by
    // following the first exponents.
    let mut best: Vec<(u64, u32, u32)> = Vec::with_capacity(span + 1);
    best.push((0, 0, 0));
    for i in 1..=span {
        let b = terminal_log + i as u32;
        let mut cand: Option<(u64, u32, u32)> = None;
        for d in 1..=dmax.min(i as u32) {
            let (rest_cost, rest_trees, _) = best[i - d as usize];
            let cost = layer_cost_q(d, b - d).saturating_add(rest_cost);
            let trees = rest_trees + 1;
            // Strictly better only: ties keep the smaller `d` reached first.
            if cand.is_none_or(|(c, t, _)| (cost, trees) < (c, t)) {
                cand = Some((cost, trees, d));
            }
        }
        // `d = 1` is always admissible (i ≥ 1, dmax ≥ 1), so `cand` is set.
        best.push(cand.unwrap_or((u64::MAX, u32::MAX, 1)));
    }
    let (cost_q, trees, _) = best[span];
    let mut schedule = Vec::with_capacity(trees as usize);
    let mut i = span;
    while i > 0 {
        let d = best[i].2;
        schedule.push(d as u8);
        i -= d as usize;
    }
    FriScheduleChoice {
        cost_q,
        trees,
        schedule,
    }
}

/// The production schedule DP: [`fri_schedule_by`] under the cost-law
/// objective ([`FRI_COST_WEIGHTS`]) with the cap policy `cap`.
pub fn fri_schedule_with_cost(
    b0: u32,
    terminal_log: u32,
    num_queries: u64,
    cap: CapPolicy,
    dmax: u32,
) -> FriScheduleChoice {
    fri_schedule_by(b0, terminal_log, dmax, &|d, depth| {
        fri_layer_cost_q(&FRI_COST_WEIGHTS, d, depth, num_queries, cap)
    })
}

/// The schedule of [`fri_schedule_with_cost`].
pub fn fri_schedule(
    b0: u32,
    terminal_log: u32,
    num_queries: u64,
    cap: CapPolicy,
    dmax: u32,
) -> Vec<u8> {
    fri_schedule_with_cost(b0, terminal_log, num_queries, cap, dmax).schedule
}

/// Today's schedule: every committed layer folds by 2.
pub fn legacy_fri_schedule(b0: u32, terminal_log: u32) -> Vec<u8> {
    vec![1; b0.saturating_sub(terminal_log) as usize]
}

/// Why a proof format cannot be laid out for a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FriFormatError {
    /// The schedule override does not cover this table's committed folds
    /// exactly, or has an exponent outside `1..=FRI_SCHEDULE_DMAX`.
    ScheduleOverrideMismatch,
}

impl core::fmt::Display for FriFormatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ScheduleOverrideMismatch => {
                f.write_str("the FRI schedule override does not cover this table's committed folds")
            }
        }
    }
}

/// Everything the fold layout needs to know about the proof format, for one
/// table. All fields are verifier-side constants; none is ever read from a
/// proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FriFormat {
    pub mode: FriMode,
    /// The RESOLVED one-row choice for this table (never `Auto`).
    pub one_row: bool,
    /// FRI query count (the opening count of every FRI tree).
    pub num_queries: u64,
    /// The active Merkle-cap policy (an input of the DP: the cap changes each layer's path cost).
    pub cap: CapPolicy,
    /// An explicit schedule that replaces the DP's under [`FriMode::Dp`].
    pub schedule_override: Option<FriScheduleOverride>,
}

impl FriFormat {
    /// Today's format: pair layers, row-pair openings. The query count and cap
    /// policy are unused by the all-ones schedule.
    pub const LEGACY: Self = Self {
        mode: FriMode::Pair,
        one_row: false,
        num_queries: 0,
        cap: CapPolicy::Off,
        schedule_override: None,
    };

    /// The format of a table proved under `options` whose trace trees use
    /// the RESOLVED leaf layout `one_row` (the table's
    /// [`crate::leaf_layout::table_leaf_layout`]; `options.format.one_row` may
    /// be `Auto`, which only the caller can resolve, from the AIR's widths).
    pub fn from_options(options: &ProofOptions, one_row: bool) -> Self {
        Self {
            mode: options.format.fri_mode,
            one_row,
            num_queries: options.fri_number_of_queries as u64,
            cap: options.format.merkle_cap,
            schedule_override: options.format.fri_schedule_override,
        }
    }

    /// Whether the proof uses today's FRI encoding: one sibling value per
    /// committed layer, pair leaves. True exactly for pair
    /// layers with row-pair openings; any other format carries every layer's
    /// full group, even where the schedule is all ones. Decided by the format,
    /// never by the schedule's values.
    pub fn is_legacy(&self) -> bool {
        self.mode == FriMode::Pair && !self.one_row
    }

    /// `Q ×` the per-query in-guest price of this table's whole FRI chain for
    /// an LDE of `2^lde_log` folding to a terminal of `2^terminal_log`: every
    /// committed layer of [`Self::schedule`] (group layers, or today's pair
    /// layers when [`Self::is_legacy`]) plus, for row-pair openings, the
    /// uncommitted fold 0 (one fold; the group encoding also squares the
    /// point once into the first layer's `y⁻¹`, where the pair encoding
    /// squares inside each layer).
    pub fn chain_cost_q(&self, lde_log: u32, terminal_log: u32) -> u64 {
        let w = &FRI_COST_WEIGHTS;
        let (q, cap) = (self.num_queries, self.cap);
        let b0 = fri_chain_start(lde_log, self.one_row);
        let schedule = self.schedule(lde_log, terminal_log);
        let layers = if self.is_legacy() {
            fri_schedule_cost_by(b0, &schedule, &|_, depth| {
                fri_pair_layer_cost_q(w, depth, q, cap)
            })
        } else {
            fri_schedule_cost_q(b0, &schedule, q, cap)
        }
        .unwrap_or(u64::MAX);
        let fold0 = if !self.one_row && lde_log > terminal_log {
            let per_query = if self.is_legacy() {
                w.fold
            } else {
                w.fold + w.twiddle
            };
            q.saturating_mul(per_query)
        } else {
            0
        };
        layers.saturating_add(fold0)
    }

    /// The committed-layer fold schedule for an LDE of `2^lde_log` folding to a
    /// terminal of `2^terminal_log` (the override's, verbatim, when one is
    /// set under `Dp`; the layout checks that it fits).
    pub fn schedule(&self, lde_log: u32, terminal_log: u32) -> Vec<u8> {
        let b0 = fri_chain_start(lde_log, self.one_row);
        match (self.mode, self.schedule_override) {
            (FriMode::Pair, _) => legacy_fri_schedule(b0, terminal_log),
            (FriMode::Dp, Some(o)) => o.as_slice().to_vec(),
            (FriMode::Dp, None) => fri_schedule(
                b0,
                terminal_log,
                self.num_queries,
                self.cap,
                FRI_SCHEDULE_DMAX,
            ),
        }
    }
}

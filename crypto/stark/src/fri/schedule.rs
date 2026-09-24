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
//! only, never from a proof. So the dynamic program below is integer-only
//! (`u64`), with a fixed tie rule, and its inputs are all public:
//!
//! * `b0` — log2 length of the first committed layer (`lde_log − 1` when fold 0
//!   is the uncommitted binary fold of the trace pair, `lde_log` when the DEEP
//!   codeword itself is committed);
//! * `terminal_log` — log2 length of the terminal codeword;
//! * `num_queries` — FRI query count;
//! * the cap-height function `depth ↦ c` of the active Merkle-cap policy (the
//!   caller closes over the opening count; `c ≡ 0` when caps are off);
//! * `dmax` — the largest fold exponent the program may choose.
//!
//! Cost model (per query, in units of `1/num_queries` of an in-guest hash
//! permutation, so every term is an integer): a layer of fold exponent `d`
//! whose tree has `depth` levels costs
//!
//! ```text
//! Q·leaf(d) + Q·(depth − c) + (2^c − 1),   leaf(d) = max(1, ⌈3·2^d / 8⌉),   c = cap(depth)
//! ```
//!
//! i.e. the leaf absorption of `2^d` cubic-extension values at an 8-felt rate,
//! the authentication walk down to the cap, and the cap-to-root reduction
//! amortised over the `Q` queries.

/// Largest fold exponent the schedule may choose (a 64-value group leaf).
pub const FRI_SCHEDULE_DMAX: u32 = 6;

/// Leaf absorption rate of the cost model, in base-field elements per
/// permutation (the RPX sponge rate).
pub const FRI_LEAF_RATE_FELTS: u64 = 8;

/// Extension degree of the FRI codeword values.
pub const FRI_EXTENSION_DEGREE: u64 = 3;

/// The FRI layer format (`LAMBDA_VM_ZF_FRI`). `Pair` is today's all-ones
/// schedule; `Dp` is the schedule [`fri_schedule`] picks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FriMode {
    #[default]
    Pair,
    Dp,
}

/// The trace-opening layout (`LAMBDA_VM_ZF_ONE_ROW`). `Off` is today's row-pair
/// leaves with an uncommitted binary fold 0; `On` opens one row and commits the
/// DEEP codeword as FRI layer 0; `Auto` decides per table. A layout is built
/// from the RESOLVED per-table choice (a `bool`), never from `Auto`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OneRowMode {
    #[default]
    Off,
    On,
    Auto,
}

/// A cap-height function that caps nothing (`c ≡ 0`).
pub fn no_cap(_depth: u32) -> u32 {
    0
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

/// Permutations to absorb one group leaf of `2^d` extension values.
pub fn fri_leaf_blocks(d: u32) -> u64 {
    let felts = FRI_EXTENSION_DEGREE.saturating_mul(1u64.checked_shl(d).unwrap_or(u64::MAX));
    felts.div_ceil(FRI_LEAF_RATE_FELTS).max(1)
}

/// `Q ×` the per-query authentication cost of a tree of `depth` levels:
/// `Q·(depth − c) + 2^c − 1`, `c = cap_height(depth)` clamped to `depth`.
pub fn fri_path_cost_q(depth: u32, num_queries: u64, cap_height: &dyn Fn(u32) -> u32) -> u64 {
    // Clamped so that a policy returning more than the tree has can never make
    // the walk negative (and `2^c` never overflows).
    let c = cap_height(depth).min(depth).min(63);
    num_queries
        .saturating_mul(u64::from(depth - c))
        .saturating_add((1u64 << c) - 1)
}

/// `Q ×` the per-query cost of one committed layer of fold exponent `d` whose
/// tree has `depth` levels (the layer is `2^{depth + d}` values long).
fn layer_cost_q(d: u32, depth: u32, num_queries: u64, cap_height: &dyn Fn(u32) -> u32) -> u64 {
    num_queries
        .saturating_mul(fri_leaf_blocks(d))
        .saturating_add(fri_path_cost_q(depth, num_queries, cap_height))
}

/// `Q ×` the per-query cost of an arbitrary schedule starting at `b0`, or
/// `None` if a fold exponent is zero or the schedule folds past zero bits.
/// (The model's own number for "today" is this at the all-ones schedule.)
pub fn fri_schedule_cost_q(
    b0: u32,
    schedule: &[u8],
    num_queries: u64,
    cap_height: &dyn Fn(u32) -> u32,
) -> Option<u64> {
    let mut b = b0;
    let mut cost = 0u64;
    for &d in schedule {
        let d = u32::from(d);
        if d == 0 {
            return None;
        }
        b = b.checked_sub(d)?;
        cost = cost.saturating_add(layer_cost_q(d, b, num_queries, cap_height));
    }
    Some(cost)
}

/// The optimum [`fri_schedule`] picks, with its cost.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FriScheduleChoice {
    /// `Q ×` the per-query cost (see the module docs).
    pub cost_q: u64,
    /// Number of committed trees (`schedule.len()`).
    pub trees: u32,
    /// Fold exponents, first committed layer first.
    pub schedule: Vec<u8>,
}

/// The fold schedule and its cost: the dynamic program of FRI.md §2.1.
///
/// ```text
/// best(T)      = (0, 0, [])
/// best(b > T)  = min over d ∈ [1, min(dmax, b − T)] of
///                (Q·leaf(d) + path_q(b − d) + best(b − d).cost, best(b − d).trees + 1, [d] ++ best(b − d).sched)
/// ```
///
/// compared lexicographically on `(cost, trees)`; ties go to the smallest `d`
/// (the first reached). Equivalently, the result is the lexicographically
/// smallest schedule among the `(cost, trees)`-optimal ones. It lands exactly
/// on `terminal_log`: `Σ schedule == b0 − terminal_log`, and the schedule is
/// empty when `b0 ≤ terminal_log`. A `dmax` of 0 is treated as 1.
pub fn fri_schedule_with_cost(
    b0: u32,
    terminal_log: u32,
    num_queries: u64,
    cap_height: &dyn Fn(u32) -> u32,
    dmax: u32,
) -> FriScheduleChoice {
    if b0 <= terminal_log {
        return FriScheduleChoice {
            cost_q: 0,
            trees: 0,
            schedule: Vec::new(),
        };
    }
    let dmax = dmax.max(1);
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
            let cost = layer_cost_q(d, b - d, num_queries, cap_height).saturating_add(rest_cost);
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

/// The fold schedule of FRI.md §2.1 (see [`fri_schedule_with_cost`]).
pub fn fri_schedule(
    b0: u32,
    terminal_log: u32,
    num_queries: u64,
    cap_height: &dyn Fn(u32) -> u32,
    dmax: u32,
) -> Vec<u8> {
    fri_schedule_with_cost(b0, terminal_log, num_queries, cap_height, dmax).schedule
}

/// Today's schedule: every committed layer folds by 2.
pub fn legacy_fri_schedule(b0: u32, terminal_log: u32) -> Vec<u8> {
    vec![1; b0.saturating_sub(terminal_log) as usize]
}

/// Everything the fold layout needs to know about the proof format.
///
/// All fields are verifier-side constants; none is ever read from a proof.
#[derive(Clone, Copy)]
pub struct FriFormat<'a> {
    pub mode: FriMode,
    /// The resolved one-row choice for this table.
    pub one_row: bool,
    /// FRI query count (the DP's opening count per tree).
    pub num_queries: u64,
    /// The active cap policy's height function for FRI-layer trees.
    pub cap_height: &'a dyn Fn(u32) -> u32,
}

impl FriFormat<'static> {
    /// Today's format: pair layers, row-pair openings. The query count and cap
    /// function are unused by the all-ones schedule.
    pub const LEGACY: Self = Self {
        mode: FriMode::Pair,
        one_row: false,
        num_queries: 0,
        cap_height: &no_cap,
    };
}

impl FriFormat<'_> {
    /// The committed-layer fold schedule for an LDE of `2^lde_log` folding to a
    /// terminal of `2^terminal_log`.
    pub fn schedule(&self, lde_log: u32, terminal_log: u32) -> Vec<u8> {
        let b0 = fri_chain_start(lde_log, self.one_row);
        match self.mode {
            FriMode::Pair => legacy_fri_schedule(b0, terminal_log),
            FriMode::Dp => fri_schedule(
                b0,
                terminal_log,
                self.num_queries,
                self.cap_height,
                FRI_SCHEDULE_DMAX,
            ),
        }
    }
}

impl std::fmt::Debug for FriFormat<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FriFormat")
            .field("mode", &self.mode)
            .field("one_row", &self.one_row)
            .field("num_queries", &self.num_queries)
            .finish_non_exhaustive()
    }
}

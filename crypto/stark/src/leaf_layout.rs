//! The trace-tree leaf layout of one table's proof (S2, design/FRI.md §7).
//!
//! Today every trace, precomputed, aux and composition tree commits one LDE
//! row PAIR per leaf (`commitment::ROWS_PER_LEAF = 2`): leaf `i` hashes the
//! bit-reversed rows `2i` and `2i + 1`, the points `x` and `−x`, and a query
//! opens that pair to rebuild the DEEP pair for the uncommitted FRI fold 0.
//!
//! Under one-row openings ([`LeafLayout::Row`]) every such tree commits ONE row
//! per leaf, the DEEP codeword itself is committed as FRI layer 0 (the "input
//! tree"), a query index ranges over the whole LDE (`r ∈ [0, N)`, bound `N`),
//! and the verifier computes DEEP at the one point `x_r` and checks it against
//! the input group's slot.
//!
//! The layout is decided PER TABLE by the proof format
//! ([`crate::proof::options::OneRowMode`]): `Off` = row pairs, `On` = one row,
//! `Auto` = whichever [`one_row_is_cheaper`] prices lower for this table's
//! committed widths and LDE size. Every input is AIR metadata or the trace
//! length the verifier already trusts for the FRI layout; none is read from
//! the proof's bytes. A proof may therefore mix layouts across tables, and
//! each table's layout is a verifier-side constant.
//!
//! [`LeafLayout::query_rows`] is the ONE place a query index becomes LDE rows
//! (REVIEW-FRI F7): every opening site, prover and verifier, goes through it.

use crypto::merkle_tree::cap::{CapPolicy, cap_gain};
use math::fft::bit_reversing::reverse_index;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};

use crate::fri::schedule::{FRI_COST_WEIGHTS, FriFormat};
use crate::proof::options::{OneRowMode, ProofOptions};
use crate::traits::AIR;

/// How many LDE rows one trace-tree leaf holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LeafLayout {
    /// Two bit-reversed rows per leaf, `(x, −x)`. Today's layout.
    #[default]
    RowPair,
    /// One bit-reversed row per leaf (S2).
    Row,
}

impl LeafLayout {
    /// `Row` iff `one_row`.
    pub const fn from_one_row(one_row: bool) -> Self {
        if one_row { Self::Row } else { Self::RowPair }
    }

    /// Whether this is the one-row layout.
    pub const fn is_one_row(self) -> bool {
        matches!(self, Self::Row)
    }

    /// Rows per leaf: 2 (today, [`crate::commitment::ROWS_PER_LEAF`]) or 1.
    pub const fn rows_per_leaf(self) -> usize {
        match self {
            Self::RowPair => crate::commitment::ROWS_PER_LEAF,
            Self::Row => 1,
        }
    }

    /// The exclusive bound of a query index over an LDE of `lde_len` points:
    /// a leaf index, so `lde / 2` for row pairs and `lde` for one row
    /// (FRI.md §7.7 (i): under one row `r` must be uniform over ALL of `D₀`).
    pub fn query_bound(self, lde_len: u64) -> u64 {
        match self {
            Self::RowPair => lde_len >> 1,
            #[cfg(test)]
            Self::Row
                if M3_PAIR_BOUND_AT_LDE.load(core::sync::atomic::Ordering::SeqCst) == lde_len =>
            {
                lde_len >> 1
            }
            Self::Row => lde_len,
        }
    }

    /// Depth of a trace tree over an LDE of `2^lde_log` rows: one level per
    /// bit of the leaf index (`log2(lde) − 1` for row pairs, `log2(lde)` for
    /// one row; 0 when the leaf hash is the root).
    pub const fn tree_depth(self, lde_log: usize) -> usize {
        match self {
            Self::RowPair => lde_log.saturating_sub(1),
            Self::Row => lde_log,
        }
    }

    /// The LDE storage rows (natural-order indices into the LDE columns) that
    /// query `q` opens: `(row, Some(sym_row))` for a row pair — the rows at
    /// bit-reversed positions `2q` and `2q + 1`, the points `x` and `−x` —
    /// and `(row, None)` for one row, the row at bit-reversed position `q`.
    ///
    /// The single site where a query index becomes rows (REVIEW-FRI F7).
    pub fn query_rows(self, q: usize, lde_len: usize) -> (usize, Option<usize>) {
        let n = lde_len as u64;
        match self {
            Self::RowPair => (reverse_index(q * 2, n), Some(reverse_index(q * 2 + 1, n))),
            Self::Row => (reverse_index(q, n), None),
        }
    }
}

/// Mutation M3 (FRI.md §10), test builds only: sample one-row query indexes
/// over the row-pair bound `N / 2` — for an LDE of exactly this many points
/// (0 = off). Prover and verifier both read it, so a mutated proof still
/// verifies; only `one_row_tests`' bound test sees the bias, which is what
/// makes that test load-bearing. Process-global (the prover samples on worker
/// threads) and keyed by the LDE size, so it touches only the M3 test's own
/// shape (an LDE no other one-row test uses), never a concurrent test's proof.
#[cfg(test)]
pub(crate) static M3_PAIR_BOUND_AT_LDE: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);

/// The committed widths of one table, in base-field elements per LDE row, per
/// tree. `0` = the tree does not exist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TableWidths {
    /// The precomputed tree (preprocessed tables only).
    pub precomputed: u64,
    /// The main tree (every main column, or the multiplicities of a
    /// preprocessed table).
    pub main: u64,
    /// The aux tree.
    pub aux: u64,
    /// The composition tree (every part).
    pub composition: u64,
    /// `XALU` rows of ONE in-guest DEEP point for this table
    /// ([`deep_point_xalu_rows`]); row pairs evaluate DEEP at two points, one
    /// row at one.
    pub deep_point_rows: u64,
}

/// `XALU` rows the in-guest verifier emits for DEEP at ONE query point
/// (`prover/src/lfm/deep.rs::emit_deep_point`), for a table whose DEEP
/// reconstruction folds `num_surviving` trace openings (the pruned OOD grid,
/// [`crate::ood::OodLayout::num_surviving`]) over `num_eval_points` OOD rows
/// and `num_parts` composition parts:
///
/// ```text
/// per OOD row r:  (|cols_r| − 1) Horner steps + [r ≥ 1] block scale
///                 + numerator esub + denominator esub + ediv + (emul | emul_add)
/// parts:          (P − 1) Horner steps + emul + esub + esub + ediv + emul_add
/// total:          num_surviving + 4·E + P + 3
/// ```
///
/// One `XALU` row per opened value plus a per-point constant; the prover
/// crate's `lfm::fri_group_tests` pins it against the emitter.
pub const fn deep_point_xalu_rows(num_surviving: u64, num_eval_points: u64, num_parts: u64) -> u64 {
    num_surviving + 4 * num_eval_points + num_parts + 3
}

impl TableWidths {
    /// The widths of `air`'s trees for a trace of `trace_length` rows. Main
    /// and precomputed columns are base-field elements; aux columns and
    /// composition parts are `FieldExtension` elements, each
    /// `ext_degree::<F, E>()` base elements wide.
    pub fn of<F, E, PI>(
        air: &dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>,
        trace_length: usize,
    ) -> Self
    where
        F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
        E: IsField + Send + Sync,
    {
        let precomputed = if air.is_preprocessed() {
            air.num_precomputed_columns()
        } else {
            0
        };
        let main = air.trace_layout().0.saturating_sub(precomputed);
        let aux = air.num_auxiliary_rap_columns();
        let parts = if trace_length == 0 {
            0
        } else {
            air.composition_poly_degree_bound(trace_length) / trace_length
        };
        let ext = ext_degree::<F, E>();
        let ctx = air.context();
        let num_eval_points = ctx.transition_offsets.len() * air.step_size();
        let ood = crate::ood::OodLayout::new(
            ctx.trace_columns,
            num_eval_points,
            air.step_size(),
            air.trace_ood_next_row_columns(),
        );
        Self {
            precomputed: precomputed as u64,
            main: main as u64,
            aux: (aux as u64).saturating_mul(ext),
            composition: (parts as u64).saturating_mul(ext),
            deep_point_rows: deep_point_xalu_rows(
                ood.num_surviving() as u64,
                num_eval_points as u64,
                parts as u64,
            ),
        }
    }
}

/// Base-field elements per `E` element (3 for the Goldilocks cubic
/// extension, 1 when `E = F`).
fn ext_degree<F: IsField, E: IsField>() -> u64 {
    let f = core::mem::size_of::<F::BaseType>().max(1);
    let e = core::mem::size_of::<E::BaseType>();
    (e / f).max(1) as u64
}

/// `Q ×` the per-query cost-law price of opening one trace tree whose leaf
/// holds `felts` base elements and whose tree is `depth` deep, under `cap`:
/// the leaf absorption, the walk (a compression and a select per level), and
/// minus what the tree's cap saves — the terms and weights of the FRI schedule
/// objective ([`crate::fri::schedule`]), applied to a trace tree.
pub fn trace_tree_cost_q(felts: u64, depth: u32, num_queries: u64, cap: CapPolicy) -> u64 {
    if felts == 0 {
        return 0;
    }
    let w = &FRI_COST_WEIGHTS.cap;
    let blocks = felts
        .div_ceil(crate::fri::schedule::FRI_LEAF_RATE_FELTS)
        .max(1) as i128;
    let per_query =
        blocks * w.compress as i128 + i128::from(depth) * (w.compress as i128 + w.select as i128);
    let queries = usize::try_from(num_queries).unwrap_or(usize::MAX);
    let c = cap.height(queries, depth as usize);
    let total = (num_queries as i128).saturating_mul(per_query) - cap_gain(w, queries, c);
    u64::try_from(total.max(0)).unwrap_or(u64::MAX)
}

/// `Q ×` the per-query price of one table's openings (every trace tree plus
/// the FRI chain) under `one_row`, for an LDE of `2^lde_log` rows with blowup
/// `2^blowup_log` and terminal log-degree `k`, under `options`' FRI mode, cap
/// policy and query count.
///
/// Row pairs: every tree's leaf holds two rows and is `lde_log − 1` deep, the
/// FRI chain starts at `lde_log − 1` with the uncommitted fold 0
/// ([`FriFormat::chain_cost_q`]), and DEEP is evaluated at TWO points (`υ`,
/// `−υ`). One row: every leaf holds one row and is `lde_log` deep, the FRI
/// chain (layer 0 = the committed DEEP codeword) starts at `lde_log`, and DEEP
/// is evaluated at ONE point (RULINGS 22). A DEEP point costs
/// [`TableWidths::deep_point_rows`] `XALU` rows.
pub fn table_openings_cost_q(
    widths: &TableWidths,
    options: &ProofOptions,
    lde_log: u32,
    blowup_log: u32,
    one_row: bool,
) -> u64 {
    let q = options.fri_number_of_queries as u64;
    let cap = options.format.merkle_cap;
    let layout = LeafLayout::from_one_row(one_row);
    let rows = layout.rows_per_leaf() as u64;
    let depth = layout.tree_depth(lde_log as usize) as u32;
    let trees = [
        widths.precomputed,
        widths.main,
        widths.aux,
        widths.composition,
    ]
    .iter()
    .map(|&w| trace_tree_cost_q(w.saturating_mul(rows), depth, q, cap))
    .fold(0u64, u64::saturating_add);

    let terminal_log = (blowup_log + u32::from(options.fri_final_poly_log_degree)).min(lde_log);
    let fmt = FriFormat {
        mode: options.format.fri_mode,
        one_row,
        num_queries: q,
        cap,
        schedule_override: options.format.fri_schedule_override,
    };
    let chain = fmt.chain_cost_q(lde_log, terminal_log);
    let deep_points: u64 = if one_row { 1 } else { 2 };
    let deep = q
        .saturating_mul(deep_points)
        .saturating_mul(widths.deep_point_rows)
        .saturating_mul(FRI_COST_WEIGHTS.xalu);
    trees.saturating_add(chain).saturating_add(deep)
}

/// RULINGS 6's `auto` rule: one row iff it is STRICTLY cheaper than row pairs
/// under [`table_openings_cost_q`] (a tie keeps today's layout).
pub fn one_row_is_cheaper(
    widths: &TableWidths,
    options: &ProofOptions,
    lde_log: u32,
    blowup_log: u32,
) -> bool {
    table_openings_cost_q(widths, options, lde_log, blowup_log, true)
        < table_openings_cost_q(widths, options, lde_log, blowup_log, false)
}

/// The leaf layout of a table with committed `widths` over an LDE of
/// `2^lde_log` rows (blowup `2^blowup_log`) under `options`' format.
pub fn resolve_leaf_layout(
    widths: &TableWidths,
    options: &ProofOptions,
    lde_log: u32,
    blowup_log: u32,
) -> LeafLayout {
    match options.format.one_row {
        OneRowMode::Off => LeafLayout::RowPair,
        OneRowMode::On => LeafLayout::Row,
        OneRowMode::Auto => {
            LeafLayout::from_one_row(one_row_is_cheaper(widths, options, lde_log, blowup_log))
        }
    }
}

/// ★ The leaf layout of `air`'s proof over a trace of `trace_length` rows —
/// what the prover and the host verifier both call. The format comes from
/// `air.options()` (a verifier-side constant), the widths from the AIR, and
/// the length from the trace (the verifier's `proof.trace_length()`, the same
/// value its FRI layout already trusts).
pub fn table_leaf_layout<F, E, PI>(
    air: &dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>,
    trace_length: usize,
) -> LeafLayout
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    let options = air.options();
    if options.format.one_row == OneRowMode::Off {
        return LeafLayout::RowPair;
    }
    let blowup = options.blowup_factor as usize;
    let lde_log = (trace_length.saturating_mul(blowup)).trailing_zeros();
    let blowup_log = blowup.trailing_zeros();
    resolve_leaf_layout(
        &TableWidths::of(air, trace_length),
        options,
        lde_log,
        blowup_log,
    )
}

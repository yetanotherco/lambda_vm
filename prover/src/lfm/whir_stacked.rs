//! `stacked_eval::verify` as a machine leg: the weight, and the wrapper over it.
//!
//! `stacked_eval::verify` hands `whir_chain::verify_weighted` a closure — the
//! weight of one stacked polynomial at the chain's final point — and that
//! closure is `weight_at` (`crypto/multilinear/src/stacked_eval.rs:287-317`):
//!
//! ```text
//!     Σ_c  w_c · eq(corner_c, high) · eq(point_c, low)
//! ```
//!
//! where `corner_c` is the column's prefix bits and `(high, low)` splits the
//! point at the prefix length. Written naively that is two `eq`s per column per
//! chain, which the sizing note priced as the term to collapse. Two structural
//! facts collapse it.
//!
//! [`emit_stacked_verify`] is the whole of `stacked_eval::verify`
//! (`stacked_eval.rs:409-459`) around it: the column values absorbed, one
//! batching draw, and one chain per stacked polynomial with this weight as its
//! `W`. It is the boundary an epoch's group is verified at, and the place the
//! transcript is THREADED — see [`stacked_verify_cost`].

use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;

use crate::tables::types::{FEE, GoldilocksField};

use super::algebraic_commit::leaf_capacity;
use super::builder::{Cell, Ext, LfmBuilder};
use super::instr::Addr;
use super::whir_chain::{
    ChainRoundWires, ChainShape, chain_grind_perms, chain_opening_perms, chain_shape_rows,
    chain_sponge, emit_verify_weighted,
};
use super::whir_poly::{
    challenge_powers_rows, emit_challenge_powers, emit_eq_eval, eq_eval_rows_again,
};
use super::whir_transcript::{
    COORDINATES_PER_EXT, SpongeEntry, SpongeSchedule, WhirTranscript, absorb_unpack_rows,
    sample_ext_rows,
};
use super::word::{LfmWord, ext_word};

/// One column's claim, as wires: the point it is claimed at and the batching
/// weight it carries (`gamma^column`).
///
/// ⚠ The weights are INPUTS here: whoever assembles the group emits the `n − 1`
/// multiplies. `multilinear::challenge_powers` is the host function that leg is
/// gated against — it was `pub(crate)` when this type was written, which is why
/// the weights arrive from outside rather than being built in place.
#[derive(Clone, Copy)]
pub struct ColumnClaim<'a> {
    pub point: &'a [Ext],
    pub weight: Ext,
}

/// INSTRUCTIONS [`emit_weight_at`] emits for one stacked polynomial, given
/// which columns share a claimed point.
///
/// `group_of[column]` names that point: one table's sumcheck leaves every one
/// of its columns at the same point (`stacked_eval.rs:185-188`), so in a real
/// epoch this is the table index and the group is a table's columns inside this
/// polynomial.
///
/// The three terms, each by its shape:
/// - one `Sub` per prefix position that ANY of the polynomial's columns reads as
///   a zero bit — `1 − at_j` depends on the position alone, so it is emitted
///   once and shared;
/// - per distinct claimed point, one `eq` over that group's columns' own
///   variables plus one `MulAdd` to join the group into the running total;
/// - per column, `p − 1` products for its prefix indicator plus one `MulAdd`
///   for its weighted fold, where `p = n_stack − num_vars`; a column that fills
///   the whole stack has no indicator and pays only the fold, hence `max(p, 1)`.
///
/// An empty polynomial is one interned zero.
pub fn weight_at_rows(layout: &StackedLayout, poly: usize, group_of: &[usize]) -> usize {
    let placements = layout.placements();
    let columns: Vec<usize> = (0..placements.len())
        .filter(|&column| placements[column].poly == poly)
        .collect();
    if columns.is_empty() {
        return 1;
    }

    let mut reads_a_zero = vec![false; layout.n_stack()];
    for &column in &columns {
        for (position, bit) in placements[column].prefix_bits().iter().enumerate() {
            if !bit {
                reads_a_zero[position] = true;
            }
        }
    }
    let shared_subs = reads_a_zero.iter().filter(|seen| **seen).count();

    let mut seen_groups: Vec<usize> = Vec::new();
    let mut group_rows = 0;
    for &column in &columns {
        if !seen_groups.contains(&group_of[column]) {
            seen_groups.push(group_of[column]);
            group_rows += eq_eval_rows_again(placements[column].num_vars) + 1;
        }
    }

    let column_rows: usize = columns
        .iter()
        .map(|&column| prefix_len(layout, column).max(1))
        .sum();

    shared_subs + group_rows + column_rows
}

/// `LFM_CONST` rows the leg interns, once per program: the `1`, which both
/// `eq` and a full-height column's empty indicator reach for.
pub const fn weight_at_consts() -> usize {
    1
}

/// ★ `weight_at` for one stacked polynomial, emitted.
///
/// ★ **Why this is not two `eq`s per column.** First, `corner_c` is a CONSTANT
/// zero/one vector, so its `eq` is just the prefix indicator
/// `Π_j (at_j or 1 − at_j)` — `p` factors, not a `5p`-row `eq` — and `1 − at_j`
/// depends only on the position, so every column that reads position `j` as
/// zero shares one `Sub`. Second, columns claimed at the same point share their
/// whole low-half `eq`; one table's sumcheck leaves all of its columns at one
/// point, so a real epoch pays one `eq` per (table, polynomial), not one per
/// column. Grouping is by the point's WIRES, which is exact at emit time and
/// needs no notion of a table.
///
/// The total is then `Σ_groups eq_low · Σ_{c in group} w_c · indicator_c`, which
/// also saves one multiply per column against multiplying each term out.
pub fn emit_weight_at(
    b: &mut LfmBuilder,
    layout: &StackedLayout,
    poly: usize,
    claims: &[ColumnClaim<'_>],
    at: &[Ext],
) -> Ext {
    assert_eq!(
        at.len(),
        layout.n_stack(),
        "the weight is evaluated on the stacked cube"
    );
    assert_eq!(
        claims.len(),
        layout.placements().len(),
        "one claim per column of the layout"
    );

    let placements = layout.placements();
    let columns: Vec<usize> = (0..placements.len())
        .filter(|&column| placements[column].poly == poly)
        .collect();
    if columns.is_empty() {
        return b.ext_const(&FEE::zero());
    }

    // Columns sharing a claimed point share its `eq`. The key is the point's
    // wires, in order.
    let mut groups: Vec<(Vec<Addr>, Vec<usize>)> = Vec::new();
    for &column in &columns {
        let key: Vec<Addr> = claims[column].point.iter().map(Ext::addr).collect();
        match groups.iter_mut().find(|(seen, _)| *seen == key) {
            Some((_, members)) => members.push(column),
            None => groups.push((key, vec![column])),
        }
    }

    let mut not_at: Vec<Option<Ext>> = vec![None; at.len()];
    let mut one: Option<Ext> = None;
    let mut total: Option<Ext> = None;

    for (_, members) in &groups {
        let leader = members[0];
        let prefix = prefix_len(layout, leader);
        assert_eq!(
            claims[leader].point.len(),
            placements[leader].num_vars,
            "a column is claimed on its own cube"
        );
        let eq_low = emit_eq_eval(b, claims[leader].point, &at[prefix..]);

        let mut inner: Option<Ext> = None;
        for &column in members {
            let mut indicator: Option<Ext> = None;
            for (position, bit) in placements[column].prefix_bits().iter().enumerate() {
                let factor = if *bit {
                    at[position]
                } else {
                    match not_at[position] {
                        Some(cached) => cached,
                        None => {
                            let unit = *one.get_or_insert_with(|| b.ext_const(&FEE::one()));
                            let complement = b.esub(unit, at[position]);
                            not_at[position] = Some(complement);
                            complement
                        }
                    }
                };
                indicator = Some(match indicator {
                    None => factor,
                    Some(acc) => b.emul(acc, factor),
                });
            }
            // A column filling the whole stack has an empty indicator, which is
            // the interned `1` — so the fold below is one row either way.
            let indicator =
                indicator.unwrap_or_else(|| *one.get_or_insert_with(|| b.ext_const(&FEE::one())));
            let weight = claims[column].weight;
            inner = Some(match inner {
                None => b.emul(indicator, weight),
                Some(acc) => b.emul_add(indicator, weight, acc),
            });
        }
        let inner = inner.expect("a group holds at least one column");
        total = Some(match total {
            None => b.emul(eq_low, inner),
            Some(acc) => b.emul_add(eq_low, inner, acc),
        });
    }

    total.expect("a non-empty polynomial has at least one group")
}

/// How many high variables of the stack a column's prefix selects.
fn prefix_len(layout: &StackedLayout, column: usize) -> usize {
    let place = &layout.placements()[column];
    place.n_stack - place.num_vars
}

/// The columns one stacked polynomial holds — `columns_in`
/// (`stacked_eval.rs:200-206`), which is private to the multilinear crate and
/// is exactly this filter over the public `placements()`.
fn columns_of(layout: &StackedLayout, poly: usize) -> Vec<usize> {
    let placements = layout.placements();
    (0..placements.len())
        .filter(|&column| placements[column].poly == poly)
        .collect()
}

/// One stacked polynomial's side of the proof, as wires.
pub struct StackedPolyWires<'a> {
    /// Its chain, one entry per scheduled round.
    pub rounds: &'a [ChainRoundWires<'a>],
    /// Its commitment root, as a word. Unpacked by the wrapper, because the
    /// wrapper is what holds it.
    pub root: Cell,
    /// The value its chain folds down to.
    pub final_value: Ext,
}

/// `claimed` (`stacked_eval.rs:320-331`): one stacked polynomial's columns,
/// batched by their `gamma` powers.
///
/// ⚠ The first term is a MULTIPLY and not a free wire, even though
/// `weights[0]` is the literal one: the weights arrive as WIRES out of
/// [`emit_challenge_powers`], and a polynomial past the first does not begin at
/// column zero anyway. This is where the analogy with `challenge_powers` — whose
/// first power IS free — misleads, so the form says `|columns|` and not
/// `|columns| − 1`.
fn emit_claimed(
    b: &mut LfmBuilder,
    layout: &StackedLayout,
    poly: usize,
    values: &[Ext],
    weights: &[Ext],
) -> Ext {
    let mut claimed: Option<Ext> = None;
    for column in columns_of(layout, poly) {
        claimed = Some(match claimed {
            None => b.emul(weights[column], values[column]),
            Some(acc) => b.emul_add(weights[column], values[column], acc),
        });
    }
    claimed.unwrap_or_else(|| b.ext_const(&FEE::zero()))
}

/// ★ `stacked_eval::verify` (`stacked_eval.rs:409-459`), emitted.
///
/// Every column value is absorbed BEFORE the batching challenge is drawn, which
/// is the host's order and the reason the claims cannot be picked after seeing
/// it; then one [`emit_verify_weighted`] per stacked polynomial, with
/// [`emit_weight_at`] as its `W` and [`emit_claimed`] as its claim.
///
/// ★ **One transcript, threaded.** The chains share `transcript`: the γ draw
/// precedes the first of them, so a chain here enters holding the digest that
/// draw re-absorbed rather than an empty sponge, and each chain enters where
/// the previous one left off. That is a cost as well as a value — see
/// [`stacked_verify_cost`].
///
/// `points` is the emitter's `Claimed`: one entry per COLUMN. Columns settled at
/// the same point pass the same WIRES, which is what [`emit_weight_at`] groups
/// on, so `Claimed::Shared` and `Claimed::PerColumn` are one object here.
///
/// The host's three shape refusals are emit-time assertions, the
/// `epoch_verify.rs:171-179` idiom: a proof of another shape has no way to be
/// supplied to a program that reads a fixed number of wires.
///
/// Returns the batching challenge. `stacked_eval::verify` returns only its
/// verdict, and every refusal here is already a division with no satisfying
/// assignment; γ comes back so a gate can compare it against the element the
/// HOST verifier sampled, which turns "the draw is in the right place" from an
/// absence of execution into a named assertion.
#[allow(clippy::too_many_arguments)]
pub fn emit_stacked_verify(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    layout: &StackedLayout,
    polys: &[StackedPolyWires<'_>],
    points: &[&[Ext]],
    values: &[Ext],
    shape: &ChainShape,
    domain: &Domain<GoldilocksField>,
) -> Ext {
    assert_eq!(
        values.len(),
        layout.placements().len(),
        "one claimed value per column of the layout"
    );
    assert_eq!(
        points.len(),
        layout.placements().len(),
        "one claimed point per column of the layout"
    );
    assert_eq!(
        polys.len(),
        layout.num_polys(),
        "one chain per stacked polynomial"
    );
    assert_eq!(
        shape.num_vars,
        layout.n_stack(),
        "every stacked polynomial has the stack's variables"
    );

    for value in values {
        transcript.absorb_ext(b, *value);
    }
    let gamma = transcript.sample_ext(b);
    let weights = emit_challenge_powers(b, gamma, values.len());

    let claims: Vec<ColumnClaim<'_>> = (0..values.len())
        .map(|column| ColumnClaim {
            point: points[column],
            weight: weights[column],
        })
        .collect();

    for (i, poly) in polys.iter().enumerate() {
        let root_lanes = b.unpack(poly.root);
        let claimed = emit_claimed(b, layout, i, values, &weights);
        emit_verify_weighted(
            b,
            transcript,
            poly.rounds,
            &root_lanes,
            poly.final_value,
            claimed,
            shape,
            domain,
            |b, at| emit_weight_at(b, layout, i, &claims, at),
        );
    }

    gamma
}

/// What [`emit_stacked_verify`] costs: its rows, its permutations, and the
/// sponge it hands on.
///
/// ⚠ **The row count is CONST-FREE**, in exactly the sense `chain_rows` is —
/// [`operations`](Self::operations) counts rows that are neither `LFM_CONST` nor
/// the arena's hints (`whir_chain_tests.rs:1062`). The per-table form
/// (`whir_table::table_verify_cost`) is the OTHER convention and includes its
/// constants, so a census that sums the two must say which it is in.
/// [`own_constants`](Self::own_constants) is the constants this leg is
/// responsible for, by VALUE, so a caller can union them into one pool.
pub struct StackedCost {
    ops: usize,
    /// The openings' and grinds' permutations, summed over the polynomials.
    chain_perms: usize,
    schedule: SpongeSchedule,
    chains: usize,
}

impl StackedCost {
    /// INSTRUCTIONS, excluding every `LFM_CONST`: the straight-line rows and
    /// the threaded sponge's `Pack`s and `Unpack`s.
    pub fn operations(&self) -> usize {
        self.ops + self.schedule.rows()
    }

    /// PERMUTATIONS, whole: each chain's openings and grinds, and the threaded
    /// transcript's own.
    pub fn perms(&self) -> usize {
        self.chain_perms + self.schedule.perms()
    }

    /// Where the sponge is left, so the next leg of the same program continues
    /// from it.
    pub fn entry(&self) -> SpongeEntry {
        self.schedule.entry()
    }

    /// Chains threaded — what a census multiplies a per-chain constant count by,
    /// and the number the pool assertion says it should NOT.
    pub fn chains(&self) -> usize {
        self.chains
    }

    pub fn schedule(&self) -> &SpongeSchedule {
        &self.schedule
    }

    /// ★ The `LFM_CONST` words this leg OWNS, deduplicated — the two kinds it
    /// can name:
    ///
    /// - `FEE::one()`, ONE row for the whole program. [`emit_challenge_powers`]
    ///   seeds its accumulator with it, [`emit_weight_at`] reaches for it for
    ///   `1 − at_j` and for a full-height column's empty indicator, and every
    ///   chain's `emit_verify_weighted` interns it as its `one`; the pool is
    ///   keyed on the canonical word (`builder.rs:170-181`), so they are one row
    ///   between them.
    /// - `leaf_capacity(felts)` per hash of the THREADED schedule.
    ///   `algebraic_leaf_hash` interns it for the leaf it is about to hash
    ///   (`edsl.rs:676-692`), so a program pays one per distinct leaf length —
    ///   and, because the word carries `felts % RATE_FELTS`, lengths sharing
    ///   that residue share the row. No per-hash form can see this; it is taken
    ///   off the finished schedule, the way `table_verify_cost` takes it.
    ///
    /// The chains' OTHER constants are not named here, for the same reason
    /// `chain_rows` does not name them.
    pub fn own_constants(&self) -> Vec<LfmWord> {
        let mut words: Vec<LfmWord> = vec![ext_word(&FEE::one())];
        for hash in self.schedule.hashes() {
            let word = leaf_capacity(hash.felts());
            if !words.contains(&word) {
                words.push(word);
            }
        }
        words
    }
}

/// ★ INSTRUCTIONS and permutations [`emit_stacked_verify`] costs, every term by
/// the shape it comes from.
///
/// With `C` the layout's columns and `P` its polynomials:
///
/// - `C` absorbs, one `Unpack` each, and `C` times three felts into the sponge;
/// - the batching draw: one `Pack`, and three candidates;
/// - `challenge_powers_rows(C)` for the weights — `C − 1`, the first power being
///   the interned one;
/// - per polynomial: one `Unpack` for its root, `|columns in poly|` rows for the
///   claimed fold (a `Mul` then `MulAdd`s — see [`emit_claimed`] for why the
///   first is not free), `weight_at_rows` for its weight closure, and
///   `chain_shape_rows` for its chain;
/// - the sponge, THREADED: the chains run `chain_sponge` into this one schedule
///   in order, so each enters what the last left. That is why this takes an
///   entry and hands one back, and why the chains' schedule halves are NOT
///   summed here as standalone numbers — a chain entered fresh reads a different
///   buffer at its first `state()`.
///
/// `group_of[column]` names which columns share a claimed point, as
/// [`weight_at_rows`] takes it.
pub fn stacked_verify_cost(
    layout: &StackedLayout,
    group_of: &[usize],
    shape: &ChainShape,
    entry: SpongeEntry,
) -> StackedCost {
    let columns = layout.placements().len();
    let polys = layout.num_polys();

    let mut schedule = SpongeSchedule::new(entry);
    let mut ops =
        columns * absorb_unpack_rows() + sample_ext_rows() + challenge_powers_rows(columns);
    for _ in 0..columns {
        schedule.absorb(COORDINATES_PER_EXT);
    }
    schedule.draw_ext();

    for poly in 0..polys {
        ops += 1;
        ops += columns_of(layout, poly).len();
        ops += weight_at_rows(layout, poly, group_of);
        ops += chain_shape_rows(shape);
        chain_sponge(shape, &mut schedule);
    }

    StackedCost {
        ops,
        chain_perms: polys * (chain_opening_perms(shape) + chain_grind_perms(shape)),
        schedule,
        chains: polys,
    }
}

//! The table and commitment-group walks of `multilinear_table::multi_verify`,
//! as machine legs: the shared emitters a block leaf assembles its verify from
//! ([`super::whir_block`]).
//!
//! Every leg is gated on its own against the host function it replaces — a
//! table against `multilinear_table::verify`, a commitment group against
//! `stacked_eval::verify`, a preprocessed column against `Mle::evaluate_in`.
//! What a caller adds is the ORDER, and the order is the only part of a
//! Fiat-Shamir verifier that no per-leg gate can see: each leg derives its own
//! challenges from the transcript it is handed, so a leg run at the wrong point
//! in the stream is internally consistent and globally wrong.
//!
//! The order, read from `multilinear_table::multi_verify`:
//!
//! 1. the CARRIED group roots, then the DERIVED roots ([`emit_absorb_roots`]),
//!    then `z`, `alpha`, `beta`;
//! 2. per table, in sub-proof order and threading the sponge table to table,
//!    `emit_table_verify` ([`emit_table_walk`]), with `settled_out_of_band`
//!    for the tables a prepared opening covers;
//! 3. the bus balance against the expected value ([`emit_expected`]);
//! 4. per commitment group, in group order, its opening
//!    ([`emit_group_walk_chains`]).

use multilinear::constraint_argument::FactorKind;
use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;

use crate::tables::bitwise::NUM_PRECOMPUTED_COLS;
use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::{Cell, Ext, LfmBuilder};
use super::preprocessed::{emit_bitwise_preprocessed, emit_const_mle_at};
use super::whir_chain::ChainShape;
use super::whir_stacked::{StackedChainWires, emit_stacked_verify_chains};
use super::whir_table::{TableProofWires, TableShape, TableVerdictWires, emit_table_verify};
use super::whir_transcript::WhirTranscript;
use super::word::LfmWord;

/// The shapes a group's wrapper needs, derived from what the epoch states.
///
/// `heights` is one entry per COLUMN — a table of width `w` at `v` variables
/// contributes `w` entries of `v` — and `group_of` names which columns share a
/// claimed point, which in an epoch is the table they belong to. Both are read
/// off `(width, num_vars)` and nothing else, which is why a census can run on a
/// laptop from a shape log while the proof itself is the box's.
pub fn group_columns(shapes: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
    let mut heights = Vec::new();
    let mut group_of = Vec::new();
    for (table, &(width, num_vars)) in shapes.iter().enumerate() {
        for _ in 0..width {
            heights.push(num_vars);
            group_of.push(table);
        }
    }
    (heights, group_of)
}

// =============================================================================
// The roots block
// =============================================================================

/// `multilinear_table::absorb_roots`, emitted: the carried roots, then the
/// derived ones. No draws: the block's front follows the roots with its
/// commit-time answers before it draws `z`, `alpha`, `beta`.
pub fn emit_absorb_roots(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    carried: &[Cell],
    derived: &[LfmWord],
) {
    for root in carried {
        transcript.absorb_digest(b, *root);
    }
    for word in derived {
        let root = b.digest_const(*word);
        transcript.absorb_digest(b, root.as_cell());
    }
}

/// `compute_commit_bus_offset`, emitted: `alpha^2` and `z - bus_id` once, then
/// per published byte two `MulAdd`s for `- index*alpha` and `- value*alpha^2`,
/// one `Div` for the inverse, and one `Add` into the running sum, which the
/// first byte does not need.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn emit_expected(
    b: &mut LfmBuilder,
    published: &[u8],
    start_index: u64,
    z: Ext,
    alpha: Ext,
) -> Ext {
    if published.is_empty() {
        // ⛔ The host's own early return, and it reads NEITHER challenge.
        return b.ext_const(&FEE::zero());
    }
    let alpha_sq = b.emul(alpha, alpha);
    let bus_id = b.ext_const(&FEE::from(crate::tables::types::BusId::Commit as u64));
    let base = b.esub(z, bus_id);
    let one = b.ext_const(&FEE::one());
    let mut sum: Option<Ext> = None;
    for (offset, value) in published.iter().enumerate() {
        // The two coefficients are NEGATED at emit time, so the fingerprint is
        // two `MulAdd`s rather than two multiplies and two subtractions.
        let index = b.ext_const(&-FEE::from(start_index + offset as u64));
        let byte = b.ext_const(&-FEE::from(u64::from(*value)));
        let term = b.emul_add(index, alpha, base);
        let term = b.emul_add(byte, alpha_sq, term);
        let inverse = b.ediv(one, term);
        sum = Some(match sum {
            None => inverse,
            Some(running) => b.eadd(running, inverse),
        });
    }
    sum.expect("a non-empty public output")
}

// =============================================================================
// The per-table walk
// =============================================================================

/// How one table's preprocessed columns are discharged.
///
/// The host has ONE route — rebuild each column's multilinear extension and
/// evaluate it at the reduced point (`multilinear_table.rs:959`) — and the
/// machine cannot take it: BITWISE alone is eleven columns of `2^20`. So the
/// machine has four, and which one a table takes is a property of the TABLE and
/// never of its position: they are selected by `air.name()`, the same rule
/// `block_whir::decode_table_index` applies.
pub enum PreprocessedRoute<'a> {
    /// Nothing left to check — either the table has no preprocessed columns, or
    /// a prepared opening settled all of them.
    None,
    /// BITWISE: the closed form, `O(NUM_VARS)` where the fold is `O(2^NUM_VARS)`.
    Bitwise,
    /// KECCAK_RC and REGISTER: one shared `eq(point, ·)` per table with each
    /// column folded against it, one row per NONZERO entry.
    ConstMle(&'a [&'a [FE]]),
    /// ★ A CROSS-EPOCH PAGE TABLE, and the only route an epoch never takes.
    ///
    /// `GLOBAL_MEMORY` carries OFFSET — the identity ramp, whose extension is
    /// `Σ_k 2^k · r_k` and costs `num_vars − 1` rows — and, on any page that is
    /// not a private-input page, INIT: that page's genesis bytes, discharged by
    /// [`super::preprocessed::emit_sparse_mle_at`] because they are a function
    /// of the program rather than of the row index and the cross-epoch proof
    /// has no prepared opening to settle them with.
    ///
    /// ⚠ BOTH COLUMNS ARE CARRIED, not just the one that needs deciding. The
    /// leg asserts each against the table's own claimed value through
    /// [`preprocessed_targets`], and the cost form reads `num_vars` off OFFSET's
    /// own length rather than taking it from a caller — a page whose columns
    /// were not `2^num_vars` tall would be a table nobody meant to build.
    ///
    /// `init` is `None` exactly on a private-input page, where INIT is a
    /// committed main column the verifier never recomputes
    /// (`continuation.rs:241-251`). That `None` is the ROUTE TABLE's decision,
    /// taken from the page's own config, and never inferred from how many
    /// columns the AIR happened to present.
    Page {
        offset: &'a [FE],
        init: Option<&'a [FE]>,
    },
    /// A CPU table's TIMESTAMP (main's preprocessed timestamp): the column
    /// `4·(i + 1)` of `2^num_vars` rows, whose extension is `4 + 4·Σ_k 2^k·r_k`
    /// ([`super::preprocessed::emit_timestamp_ramp`]) — no opening, no fold.
    Timestamp { num_vars: usize },
}

/// One table's preprocessed plan: what a prepared opening settled, and how
/// whatever is left is checked.
///
/// ⚠ `settled` must be the SAME value that drives the opening, never a knob a
/// caller sets on its own — the host's own warning at
/// `multilinear_table.rs:943`. A second, independent knob would be a way to
/// switch off a check with nothing put in its place.
pub struct PreprocessedPlan<'a> {
    /// Leading columns the prepared opening covers: `settled_out_of_band`.
    pub settled: usize,
    /// How the columns past those are checked.
    pub route: PreprocessedRoute<'a>,
}

/// Where each preprocessed column's claimed value sits among a table's reduced
/// column values.
///
/// ★ THIS IS THE HOST'S OWN INDIRECTION, AND IT IS NOT THE IDENTITY.
/// `check_preprocessed` does not compare preprocessed column `c` against
/// `column_values[c]`. It reads `slot(slot_of, c)` to get a factor, takes that
/// factor's SOURCE, requires the source unshifted, and indexes
/// `column_values[source.column]` (`multilinear_table.rs:1060-1080`). A leg that
/// indexed by `c` would agree with the host on every table whose layout happens
/// to be the identity and disagree silently on the rest, which is why the
/// mapping is derived here from the same two slices the host reads rather than
/// assumed.
///
/// Both of the host's refusals are EMIT-TIME assertions, the
/// `epoch_verify.rs:171-179` idiom: a column with no factor slot, or a factor
/// read at an offset, describes a layout nobody meant to build, and a program
/// emitted for it could not be supplied a proof of the right shape anyway.
pub fn preprocessed_targets(slot_of: &[usize], kinds: &[FactorKind], columns: usize) -> Vec<usize> {
    (0..columns)
        .map(|column| {
            let factor = *slot_of.get(column).unwrap_or_else(|| {
                panic!(
                    "preprocessed column {column} has no factor slot in a layout of {} columns",
                    slot_of.len()
                )
            });
            let source = kinds
                .get(factor)
                .and_then(FactorKind::source)
                .filter(|source| source.offset == 0)
                .unwrap_or_else(|| {
                    panic!(
                        "preprocessed column {column} rides factor {factor}, which is not an \
                         unshifted committed source"
                    )
                });
            source.column
        })
        .collect()
}

/// What the per-table walk leaves for the two legs below it.
///
/// The collections are the host's own under the host's names: `outputs` is what
/// the closure sums, `points` and `values` are what each commitment group
/// settles. ⚠ `points` is kept ONE PER TABLE and expanded to one per COLUMN by
/// [`Self::column_points`], because one per column is the shape
/// `emit_stacked_verify` takes and the expansion is where the host's
/// `for _ in 0..statement.slot_of.len()` (`multilinear_table.rs:1226`) lives.
pub struct TableWalk {
    /// `(p, q)` per table, in table order.
    pub outputs: Vec<(Ext, Ext)>,
    /// The reduced point, one per TABLE.
    pub points: Vec<Vec<Ext>>,
    /// Committed columns per table — what the group walk advances by.
    pub widths: Vec<usize>,
    /// Every table's reduced column values, concatenated in table order.
    pub values: Vec<Ext>,
}

impl TableWalk {
    /// The host's `points`: one entry per COLUMN, a table's columns sharing that
    /// table's point.
    pub fn column_points(&self) -> Vec<&[Ext]> {
        let mut out = Vec::with_capacity(self.values.len());
        for (point, &width) in self.points.iter().zip(&self.widths) {
            for _ in 0..width {
                out.push(point.as_slice());
            }
        }
        out
    }

    /// Where table `index`'s columns start in the global column order — the
    /// host's `prepared_at` when `index` is the prepared table.
    pub fn column_at(&self, index: usize) -> usize {
        self.widths[..index].iter().sum()
    }
}

/// ★ The per-table walk, emitted: `multi_verify`'s first loop
/// (`multilinear_table.rs:1218`).
///
/// Per table in index order, [`emit_table_verify`] with the sponge entry
/// threaded table to table, then that table's preprocessed leg IMMEDIATELY
/// after. The leg sits inside the same step rather than in a pass of its own
/// because `check_preprocessed` is the LAST statement of the host's `verify`
/// (`multilinear_table.rs:936`) and moves no transcript operation — a pass of
/// its own would compute the same values and say something different about
/// where they belong.
#[allow(clippy::too_many_arguments)]
pub fn emit_table_walk(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    tables: &[TableProofWires<'_>],
    shapes: &[TableShape<'_>],
    plans: &[PreprocessedPlan<'_>],
    slots: &[&[usize]],
    z: Ext,
    alpha_powers: &[Ext],
    beta: Ext,
) -> TableWalk {
    assert_eq!(tables.len(), shapes.len(), "one shape per table proof");
    assert_eq!(tables.len(), plans.len(), "one preprocessed plan per table");
    assert_eq!(tables.len(), slots.len(), "one slot map per table");

    let mut walk = TableWalk {
        outputs: Vec::with_capacity(tables.len()),
        points: Vec::with_capacity(tables.len()),
        widths: Vec::with_capacity(tables.len()),
        values: Vec::new(),
    };

    for (index, ((proof, shape), plan)) in tables.iter().zip(shapes).zip(plans).enumerate() {
        let verdict = emit_table_verify(b, transcript, proof, shape, z, alpha_powers, beta);
        emit_preprocessed_leg(b, plan, slots[index], shape, &verdict);
        walk.outputs.push(verdict.bus_output);
        walk.widths.push(verdict.column_values.len());
        walk.values.extend(verdict.column_values.iter().copied());
        walk.points.push(verdict.point);
    }
    walk
}

/// One table's preprocessed columns, checked against the values its own
/// argument settled on.
pub(crate) fn emit_preprocessed_leg(
    b: &mut LfmBuilder,
    plan: &PreprocessedPlan<'_>,
    slot_of: &[usize],
    shape: &TableShape<'_>,
    verdict: &TableVerdictWires,
) {
    let columns: &[&[FE]] = match &plan.route {
        PreprocessedRoute::None => return,
        PreprocessedRoute::Bitwise => {
            // The closed form covers all eleven at once, so a settled prefix
            // over them would be two checks on one column rather than none.
            assert_eq!(
                plan.settled, 0,
                "BITWISE's columns are covered by the closed form, not by an opening"
            );
            let targets = preprocessed_targets(slot_of, shape.kinds, NUM_PRECOMPUTED_COLS);
            let values = emit_bitwise_preprocessed(b, &verdict.point);
            for (value, &target) in values.iter().zip(&targets) {
                b.assert_eq_ext(*value, verdict.column_values[target]);
            }
            return;
        }
        PreprocessedRoute::Page { offset, init } => {
            assert_eq!(
                plan.settled, 0,
                "a cross-epoch page's columns are checked in the program, not settled \
                 by an opening the cross-epoch proof does not carry"
            );
            let columns = 1 + usize::from(init.is_some());
            let targets = preprocessed_targets(slot_of, shape.kinds, columns);
            assert_eq!(
                offset.len(),
                1usize << verdict.point.len(),
                "OFFSET must have one entry per row of its table"
            );
            let ramp = super::preprocessed::emit_offset_ramp(b, &verdict.point);
            b.assert_eq_ext(ramp, verdict.column_values[targets[0]]);
            if let Some(init) = init {
                let values = super::preprocessed::emit_sparse_mle_at(b, &[*init], &verdict.point);
                b.assert_eq_ext(values[0], verdict.column_values[targets[1]]);
            }
            return;
        }
        PreprocessedRoute::Timestamp { num_vars } => {
            assert_eq!(
                plan.settled, 0,
                "a CPU table's TIMESTAMP is checked by its closed form, not settled by an opening"
            );
            assert_eq!(
                *num_vars,
                verdict.point.len(),
                "the TIMESTAMP column is as tall as its table"
            );
            let targets = preprocessed_targets(
                slot_of,
                shape.kinds,
                crate::tables::cpu::NUM_PRECOMPUTED_COLS,
            );
            let timestamp = super::preprocessed::emit_timestamp_ramp(b, &verdict.point);
            b.assert_eq_ext(timestamp, verdict.column_values[targets[0]]);
            return;
        }
        PreprocessedRoute::ConstMle(columns) => columns,
    };

    // The host's own assert, at emit time: an opening may not cover more
    // columns than the table has (`multilinear_table.rs:1069`).
    assert!(
        plan.settled <= columns.len(),
        "a prepared opening settled {} of {} preprocessed columns",
        plan.settled,
        columns.len()
    );
    let remaining = &columns[plan.settled..];
    if remaining.is_empty() {
        return;
    }
    let targets = preprocessed_targets(slot_of, shape.kinds, columns.len());
    let values = emit_const_mle_at(b, remaining, &verdict.point);
    for (value, &target) in values.iter().zip(&targets[plan.settled..]) {
        b.assert_eq_ext(*value, verdict.column_values[target]);
    }
}

// =============================================================================
// The commitment-group walk
// =============================================================================

/// One commitment group's wires by CHAIN: its layout, the domain the chains
/// run over, and one chain per run of the shape's split
/// ([`ChainShape::chains`]), which is one per polynomial under
/// `GroupFold::PerPoly` and a run of up to four under `Indexed` (D-ONECHAIN).
pub struct GroupChainWires<'a> {
    pub layout: &'a StackedLayout,
    pub chains: &'a [StackedChainWires<'a>],
    pub shape: &'a ChainShape,
    pub domain: &'a Domain<GoldilocksField>,
    /// The group's commit-time answers, where its proof carries them (the
    /// block's groups; an epoch's carry none).
    pub ood: Option<super::whir_stacked::OodWires<'a>>,
}

/// ★ The commitment-group walk, emitted over each group's chains
/// ([`GroupChainWires`]): `multi_verify`'s second loop
/// (`multilinear_table.rs:1254`), each group's opening
/// [`emit_stacked_verify_chains`].
///
/// The three counters are the host's own: `statement_at` advances by the
/// group's table count, `column_at` by the sum of those tables' COLUMN counts,
/// and the roots by `layout.num_polys()`. A group handed the wrong column slice
/// settles one table's values against another table's commitment, which is
/// precisely what a counter that advanced by tables rather than columns would
/// do.
pub fn emit_group_walk_chains(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    groups: &[GroupChainWires<'_>],
    sizes: &[usize],
    walk: &TableWalk,
) -> Vec<Ext> {
    assert_eq!(groups.len(), sizes.len(), "one size per commitment group");
    assert_eq!(
        sizes.iter().sum::<usize>(),
        walk.widths.len(),
        "the groups' sizes must cover every table exactly once"
    );
    let points = walk.column_points();
    let mut statement_at = 0usize;
    let mut column_at = 0usize;
    let mut gammas = Vec::with_capacity(groups.len());
    for (group, &size) in groups.iter().zip(sizes) {
        let width: usize = walk.widths[statement_at..statement_at + size].iter().sum();
        gammas.push(emit_stacked_verify_chains(
            b,
            transcript,
            group.layout,
            group.chains,
            &points[column_at..column_at + width],
            &walk.values[column_at..column_at + width],
            group.ood,
            group.shape,
            group.domain,
        ));
        statement_at += size;
        column_at += width;
    }
    gammas
}

// =============================================================================
// The tables a verify settles by name
// =============================================================================

/// The four tables whose preprocessed columns a block carries, and the route
/// each one takes.
///
/// ⛔ A TABLE WITH PREPROCESSED COLUMNS AND NONE OF THESE NAMES IS A REFUSAL,
/// not a skip. Skipping it would drop a `check_preprocessed` with nothing put in
/// its place, which is exactly what `settled_out_of_band` is documented never to
/// become (`multilinear_table.rs:943`). A fifth preprocessed table added
/// upstream must fail this build rather than go quietly unchecked.
pub const BITWISE_NAME: &str = "BITWISE";
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) const DECODE_NAME: &str = "DECODE";
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) const KECCAK_RC_NAME: &str = "KECCAK_RC";
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) const REGISTER_NAME: &str = "REGISTER";
/// What every CPU table's name starts with (`CPU[k]`): its TIMESTAMP takes the
/// closed-form route.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) const CPU_NAME_PREFIX: &str = "CPU[";

/// One table's proof words, in the order a leaf hints them.
///
/// ⚠ The order is a CONTRACT with [`hint_table_wires_shaped`]: a second
/// spelling of it is exactly the drift that hands the machine somebody else's
/// field element with every value gate still green.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn push_table_words(
    words: &mut Vec<LfmWord>,
    table: &stark::multilinear_table::TableProof<GoldilocksExtension>,
) {
    let ext = super::word::ext_word;
    words.push(ext(&table.bus_output.0));
    words.push(ext(&table.bus_output.1));
    for layer in &table.gkr.layers {
        for round in &layer.sumcheck.rounds {
            words.extend(round.evaluations.iter().map(ext));
        }
        for value in [&layer.p_lo, &layer.p_hi, &layer.q_lo, &layer.q_hi] {
            words.push(ext(value));
        }
    }
    for round in &table.constraint.sumcheck.rounds {
        words.extend(round.evaluations.iter().map(ext));
    }
    words.extend(table.constraint.factor_values.iter().map(ext));
    for round in &table.constraint.reduce.sumcheck.rounds {
        words.extend(round.evaluations.iter().map(ext));
    }
    words.extend(table.constraint.reduce.column_values.iter().map(ext));
}

/// One table's hinted wires, OWNED, because `TableProofWires` borrows them.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) struct TableWires {
    bus_output: (Ext, Ext),
    gkr: Vec<super::whir_gkr::GkrLayerWires>,
    sumcheck: Vec<Vec<Ext>>,
    factor_values: Vec<Ext>,
    reduce_sumcheck: Vec<Vec<Ext>>,
    column_values: Vec<Ext>,
}

impl TableWires {
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(crate) fn borrow(&self) -> TableProofWires<'_> {
        TableProofWires {
            bus_output: self.bus_output,
            gkr: &self.gkr,
            sumcheck: &self.sumcheck,
            factor_values: &self.factor_values,
            reduce: super::whir_reduce::ReduceWires {
                sumcheck: &self.reduce_sumcheck,
                column_values: &self.column_values,
            },
        }
    }
}

/// How many words [`push_table_words`] writes for a table of `shape`: the
/// counts [`hint_table_wires_shaped`] hints, from the shape alone.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn table_words(shape: &TableShape<'_>) -> usize {
    use super::whir_gkr::GKR_SUMCHECK_DEGREE;
    use super::whir_reduce::REDUCE_DEGREE;
    let layers = shape.gkr_layers();
    let gkr: usize = (0..layers).map(|i| i * GKR_SUMCHECK_DEGREE + 4).sum();
    2 + gkr
        + shape.num_vars * shape.sumcheck_degree()
        + shape.sources().len()
        + shape.num_vars * REDUCE_DEGREE
        + shape.num_columns
}

/// A table's wires, hinted with every count taken from the table's SHAPE — the
/// AIR at the stated height — rather than from a proof: what an emitter that
/// derives its program from the statement alone hints. The counts are
/// `emit_table_verify`'s own (one GKR layer per input variable, layer `i` with
/// `i` rounds; one main round per row variable at the batch's degree; one factor
/// value per committed source; one reduce round per row variable; one value per
/// committed column), so a proof of another shape cannot be read into it.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn hint_table_wires_shaped(
    b: &mut LfmBuilder,
    arena: super::instr::ArenaId,
    at: &mut u32,
    shape: &TableShape<'_>,
) -> TableWires {
    use super::whir_gkr::GKR_SUMCHECK_DEGREE;
    use super::whir_reduce::REDUCE_DEGREE;
    let mut take = |b: &mut LfmBuilder, count: usize| -> Vec<Ext> {
        (0..count)
            .map(|_| {
                let wire = b.hint_word(arena, *at).as_ext();
                *at += 1;
                wire
            })
            .collect()
    };
    let output = take(b, 2);
    let gkr = (0..shape.gkr_layers())
        .map(|i| {
            let sumcheck: Vec<Vec<Ext>> = (0..i).map(|_| take(b, GKR_SUMCHECK_DEGREE)).collect();
            let halves = take(b, 4);
            super::whir_gkr::GkrLayerWires {
                sumcheck,
                p_lo: halves[0],
                p_hi: halves[1],
                q_lo: halves[2],
                q_hi: halves[3],
            }
        })
        .collect();
    let degree = shape.sumcheck_degree();
    let sumcheck: Vec<Vec<Ext>> = (0..shape.num_vars).map(|_| take(b, degree)).collect();
    let factor_values = take(b, shape.sources().len());
    let reduce_sumcheck: Vec<Vec<Ext>> = (0..shape.num_vars)
        .map(|_| take(b, REDUCE_DEGREE))
        .collect();
    let column_values = take(b, shape.num_columns);
    TableWires {
        bus_output: (output[0], output[1]),
        gkr,
        sumcheck,
        factor_values,
        reduce_sumcheck,
        column_values,
    }
}

/// A group's chains, hinted for a stack of `polys` polynomials under `shape`: one
/// chain per run of [`ChainShape::chains`], each at its run's shape — the count
/// and the shapes the verifier's own layout and format give rather than the
/// proof's.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn hint_group_chains_shaped(
    b: &mut LfmBuilder,
    arena: super::instr::ArenaId,
    at: &mut u32,
    polys: usize,
    shape: &ChainShape,
) -> GroupChains {
    let runs = shape.chains(polys);
    let mut finals = Vec::with_capacity(runs.len());
    let mut storage = Vec::with_capacity(runs.len());
    for run in &runs {
        let shape = shape.run(run.len());
        finals.push(b.hint_word(arena, *at).as_ext());
        *at += 1;
        storage.push(super::whir_chain::RoundStorage::hint(b, arena, *at, &shape));
        *at += super::whir_chain::RoundStorage::words(&shape);
    }
    GroupChains {
        finals,
        storage,
        runs,
    }
}

/// One commitment group's hinted chains, OWNED for the same reason.
///
/// ⛔ The ROOTS ARE NOT HERE. A group's roots are the cells the roots block
/// already hinted, and [`Self::polys`] takes them by reference: a second copy
/// would let a prover absorb one root into the transcript and open the chain
/// against another, with an honest arena looking identical to every value gate.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) struct GroupChains {
    pub(crate) finals: Vec<Ext>,
    pub(crate) storage: Vec<super::whir_chain::RoundStorage>,
    /// Each chain's polynomials, a range of the stack's ([`ChainShape::chains`]).
    pub(crate) runs: Vec<std::ops::Range<usize>>,
}

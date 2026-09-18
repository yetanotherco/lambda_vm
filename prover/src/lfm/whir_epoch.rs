//! The whole epoch verify, assembled: `multilinear_table::multi_verify` as one
//! machine program.
//!
//! Every leg this assembles is gated on its own against the host function it
//! replaces — the statement against `absorb_epoch`, a table against
//! `multilinear_table::verify`, a commitment group against
//! `stacked_eval::verify`, a preprocessed column against `Mle::evaluate_in`.
//! What is left to say here is the ORDER, and the order is the only part of a
//! Fiat-Shamir verifier that no per-leg gate can see: each leg derives its own
//! challenges from the transcript it is handed, so a leg run at the wrong point
//! in the stream is internally consistent and globally wrong.
//!
//! The order, read from `multilinear_continuation::verify_epoch_bookend` and
//! `multilinear_table::multi_verify`:
//!
//! 1. the epoch STATEMENT (`absorb_epoch`), which is all program text;
//! 2. the CARRIED group roots, then the DERIVED DECODE root — see the warning
//!    below — then `z`, `alpha`, `beta`, which is `absorb_roots_and_challenge`
//!    (`multilinear_table.rs:622`) and is THREE draws, not two;
//! 3. per table, in sub-proof order and threading the sponge table to table,
//!    `emit_table_verify`, with `settled_out_of_band` for the one table the
//!    prepared opening covers;
//! 4. the bus balance against the epoch's expected value, `owed`;
//! 5. per commitment group, in group order, `emit_stacked_verify`;
//! 6. the PREPARED opening last, the same wrapper on the DECODE group.
//!
//! # ⛔ The derived DECODE root is PROGRAM TEXT, and calling it "derived" would
//! overstate what the machine does
//!
//! The host recomputes DECODE's preprocessed commitment from the ELF it holds
//! and absorbs it; the machine cannot rebuild a `2^20` Merkle root, so this
//! interns the root as a program constant, exactly as the statement interns
//! `elf_digest`.
//!
//! What the constant IS bound to: the prepared opening (step 6) proves the
//! pinned commitment takes DECODE's own settled column values at DECODE's own
//! reduced point, so a program carrying a root nothing can be opened against
//! does not execute. What it is NOT bound to: the `elf_digest` the statement
//! names. Nothing inside ONE epoch program ties the root to the digest — that
//! pin is once per ELF and lives outside. This is the multilinear twin of the
//! univariate side's ledger entry 7 (`epoch_tests.rs:1560`), and it is recorded
//! as owed rather than described as covered.
//!
//! # The layouts are rebuilt here, and the execution gate is what says they are
//! the host's
//!
//! `verify_epoch_bookend` builds its `TableLayout`s through a private helper.
//! Rather than widen that file, this calls `TableLayout::new` with the same five
//! arguments — a second spelling, and second spellings drift. What catches a
//! drift is not a comment: every challenge downstream of a layout is derived
//! from it, so a layout that differed from the host's would give the machine a
//! different `z` and the honest proof would stop executing. The gate that runs
//! the host's own accepted proof through this program is therefore the check on
//! this paragraph.

use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;
use multilinear::whir_chain::ChainConfig;

use crate::tables::types::{GoldilocksExtension, GoldilocksField};

use super::whir_chain::ChainShape;
use super::whir_stacked::{StackedCost, stacked_verify_cost};
use super::whir_transcript::{SpongeEntry, SpongeSchedule};
use super::word::LfmWord;

/// The epoch's AIRs, as the builder takes them.
///
/// They cannot live inside the epoch struct: `multi_verify` takes
/// `TableStatement`s built from them, those borrow, and building them needs the
/// guest ELF the level-0 driver holds. So they travel alongside.
pub type EpochAirs<'a> = &'a [&'a dyn stark::traits::AIR<
    Field = GoldilocksField,
    FieldExtension = GoldilocksExtension,
    PublicInputs = (),
>];

/// How many polynomials the stacked DECODE commitment holds.
///
/// One: its five columns are all `2^20` and the stack is `2^23`, so they pack
/// into a single polynomial. Named rather than spelled `1` at the two places
/// that need it.
pub const DECODE_GROUP_POLYS: usize = 1;

/// What one epoch's verify costs, kept as the terms it is made of rather than
/// as a total.
///
/// ⚠ TWO CONVENTIONS MEET HERE, and the struct keeps them apart rather than
/// adding them. [`Self::operations`] is CONST-FREE in exactly the sense
/// `chain_rows` is: rows that are neither `LFM_CONST` nor a hint. The per-table
/// form (`whir_table::table_verify_cost().rows()`) is the OTHER convention and
/// counts its own constants, because in a per-table program they ARE that
/// program's pool. In ONE epoch program there is ONE pool, shared across the
/// tables, the chains, the wrappers, the preprocessed legs and the statement,
/// so the tables' constants COLLIDE and their per-table counts over-count.
/// [`Self::constants`] is that one pool, by value, and it is added ONCE.
#[derive(Debug, Clone)]
pub struct EpochCost {
    /// The 34 tables' legs, CONST-FREE.
    pub tables: usize,
    /// The commitment groups' wrappers and their chains, CONST-FREE — the
    /// chains' `chain_shape_rows` is inside this, not beside it.
    pub groups: usize,
    /// The prepared DECODE opening's wrapper and chain, CONST-FREE.
    pub prepared: usize,
    /// The preprocessed columns no opening settled: BITWISE's closed form and
    /// the constant-MLE folds.
    pub preprocessed: usize,
    /// The bus balance and the expected value.
    pub closure: usize,
    /// Permutations, whole.
    pub perms: usize,
    /// The ONE pool, by value.
    pub constants: Vec<LfmWord>,
    /// Where the sponge is left.
    pub entry: SpongeEntry,
}

impl Default for EpochCost {
    /// An epoch that has cost nothing yet, with the sponge entered fresh —
    /// `SpongeEntry` has no `Default` because "an empty buffer with no squeeze
    /// in hand" is a FACT about `DefaultTranscript::new` and not a zero value,
    /// so it is spelled `fresh()` and this names it.
    fn default() -> Self {
        Self {
            tables: 0,
            groups: 0,
            prepared: 0,
            preprocessed: 0,
            closure: 0,
            perms: 0,
            constants: Vec::new(),
            entry: SpongeEntry::fresh(),
        }
    }
}

impl EpochCost {
    /// INSTRUCTIONS, excluding every `LFM_CONST` — the convention `chain_rows`
    /// and `stacked_verify_cost().operations()` are in.
    pub fn operations(&self) -> usize {
        self.tables + self.groups + self.prepared + self.preprocessed + self.closure
    }

    /// INSTRUCTIONS including the ONE pool — the convention
    /// `table_verify_cost().rows()` is in, evaluated over a program that has a
    /// single pool instead of 34.
    pub fn rows(&self) -> usize {
        self.operations() + self.constants.len()
    }

    pub fn perms(&self) -> usize {
        self.perms
    }
}

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

/// ★ The two commitment groups' cost at an epoch's shape, threaded.
///
/// `sizes` is `epoch_groups(n)` — `[n − 1, 1]`, the bookend committed alone —
/// and `layouts` is `multilinear_prove::stacks`' own output, so this evaluates
/// the gated wrapper form at the layout the PROVER actually built rather than
/// at a reconstruction of it.
///
/// Returns the per-group costs in group order, each entered where the previous
/// one left the sponge.
pub fn epoch_group_costs(
    shapes: &[(usize, usize)],
    sizes: &[usize],
    layouts: &[StackedLayout],
    config: &ChainConfig,
    entry: SpongeEntry,
) -> Vec<StackedCost> {
    assert_eq!(
        layouts.len(),
        sizes.len(),
        "one layout per commitment group"
    );
    let mut costs = Vec::with_capacity(layouts.len());
    let mut at = 0usize;
    let mut entry = entry;
    for (group, &size) in sizes.iter().enumerate() {
        let (_, group_of) = group_columns(&shapes[at..at + size]);
        let shape = ChainShape::new(config, layouts[group].n_stack());
        let cost = stacked_verify_cost(&layouts[group], &group_of, &shape, entry);
        entry = cost.entry();
        costs.push(cost);
        at += size;
    }
    assert_eq!(at, shapes.len(), "the group sizes must cover every table");
    costs
}

/// The prepared DECODE opening's cost: the same wrapper on a one-polynomial
/// group of five columns, at the stack the pinned commitment was built at.
///
/// Its five columns are claimed at ONE point — `Claimed::Shared` on the host
/// (`multilinear_table.rs:1310`) — so `group_of` is all zeros and the wrapper
/// pays one `eq` for the five, not five.
pub fn prepared_cost(
    layout: &StackedLayout,
    config: &ChainConfig,
    entry: SpongeEntry,
) -> StackedCost {
    let group_of = vec![0usize; layout.placements().len()];
    let shape = ChainShape::new(config, layout.n_stack());
    stacked_verify_cost(layout, &group_of, &shape, entry)
}

/// The epoch's closure: a division per table for `p/q`, the running sum, and
/// the expected value the sum is compared against.
///
/// - `contribution` (`multilinear_table.rs:753`) is `p/q` per table — one
///   `Div`, and a `Div` is the refusal when `q` vanishes;
/// - the balance is a running `Add`, whose first term needs none;
/// - `compute_commit_bus_offset` (`lib.rs:1156`) is the expected value: zero
///   when the epoch publishes nothing, and otherwise `alpha²` once, then per
///   published byte two `MulAdd`s for `z − bus_id − index·alpha − value·alpha²`,
///   a `Div` for its inverse and an `Add` into the running sum, the first of
///   which needs no `Add`;
/// - one `assert_eq_ext`, which is a difference and a division by zero.
///
/// ⚠ THE PUBLISHED BYTES AND THE START INDEX ARE PROGRAM TEXT, not wires: the
/// statement already interns the public output, and `start_index` is the
/// carried `x254` of the register file this epoch's program is compiled for. So
/// the per-byte coefficients are constants and only `z` and `alpha` are wires.
pub fn closure_rows(num_tables: usize, published: usize) -> usize {
    let contributions = num_tables; // one Div each
    let balance = num_tables.saturating_sub(1); // the running Add
    let expected = if published == 0 {
        0
    } else {
        1 + published * 4 - 1
    };
    contributions + balance + expected + ASSERT_ROWS
}

/// Rows an `assert_eq_ext` lowers to: the difference and the division by zero
/// (`builder.rs:289-292`).
const ASSERT_ROWS: usize = 2;

/// The sponge the epoch's legs thread through, entered fresh.
///
/// Split out because both the program and the census need the SAME starting
/// entry and a second spelling of "fresh" is a place for them to differ.
pub fn fresh_schedule() -> SpongeSchedule {
    SpongeSchedule::new(SpongeEntry::fresh())
}

/// The domains a group's chains run over, as `stacks` built them — re-exported
/// through this module so a caller assembling an epoch does not reach into the
/// prover's own module for one type.
pub type EpochDomains = Vec<Domain<GoldilocksField>>;

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

use multilinear::constraint_argument::FactorKind;
use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;
use multilinear::whir_chain::ChainConfig;
use stark::multilinear_logup::InteractionShape;
use stark::multilinear_table::TableLayout;

use crate::tables::bitwise::NUM_PRECOMPUTED_COLS;
use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::leaf_capacity;
use super::builder::{Cell, Ext, LfmBuilder};
use super::compiler::LfmProgram;
use super::preprocessed::{
    bitwise_preprocessed_rows, const_mle_constants, const_mle_rows, emit_bitwise_preprocessed,
    emit_const_mle_at,
};
use super::whir_bus::Cost;
use super::whir_chain::ChainShape;
use super::whir_real_epoch::WhirRealEpoch;
use super::whir_stacked::{
    StackedCost, StackedPolyWires, emit_stacked_verify, stacked_verify_cost,
};
use super::whir_table::{
    TableProofWires, TableShape, TableVerdictWires, emit_table_verify, table_verify_cost,
};
use super::whir_transcript::{
    SpongeEntry, SpongeSchedule, WhirTranscript, absorb_unpack_rows, sample_ext_rows,
};
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
    contributions + balance + expected_rows(published) + ASSERT_ROWS
}

/// INSTRUCTIONS the epoch's EXPECTED value costs — `compute_commit_bus_offset`
/// (`lib.rs:1156`), emitted.
///
/// ⛔ ZERO when the epoch publishes nothing, and that is instance 51's seam
/// rather than an optimisation: the host returns the literal zero for an empty
/// public output WITHOUT reading `z` or `alpha`, so an epoch that publishes
/// nothing discharges this term without ever touching the challenges. A defect
/// in it is invisible on every silent epoch, which is why the fixture must
/// include one that publishes.
///
/// Otherwise, by the shape it comes from:
/// - `alpha^2`, once;
/// - `z - bus_id`, once, because the bus id is the same constant for every byte;
/// - per published byte, two `MulAdd`s for `- index*alpha` and `- value*alpha^2`
///   (both coefficients are program text: the bytes are the statement's and
///   `start_index` is the carried `x254` of the register file this program is
///   compiled for), one `Div` for the inverse, and one `Add` into the running
///   sum which the first byte does not need.
pub const fn expected_rows(published: usize) -> usize {
    if published == 0 {
        return 0;
    }
    2 + published * 4 - 1
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

// =============================================================================
// The roots block
// =============================================================================

/// ★ `absorb_roots_and_challenge` (`multilinear_table.rs:622`), emitted: every
/// CARRIED group root, then the DERIVED DECODE root, then `z`, `alpha`, `beta`.
///
/// This is the one step of the assembled epoch verify that no other gate can
/// see. Each leg below it derives its own challenges from the transcript it is
/// handed, so a leg run against the wrong `z` is internally consistent and
/// globally wrong; and the roots block is where `z` comes from. Its three
/// decisions, each of which a mutation moves:
///
/// 1. the carried roots first, in `proof.roots` order;
/// 2. the derived DECODE root AFTER them and BEFORE any draw;
/// 3. THREE draws, not two — `beta` is sampled here even though the tables read
///    it later, and a squeeze nobody reads still moves the sponge.
///
/// # ⛔ `derived` is PROGRAM TEXT, and the word "derived" would overstate it
///
/// The host recomputes DECODE's preprocessed commitment from the ELF it holds.
/// The machine cannot rebuild a `2^20` Merkle root, so the root is interned as a
/// program constant — `digest_const`, not `hint_word`, which is the whole
/// difference: a hinted root is a value the prover chooses and this one is not.
///
/// What the constant is bound to: the prepared opening proves the pinned
/// commitment takes DECODE's own settled column values at DECODE's own reduced
/// point, so a program carrying a root nothing can be opened against does not
/// execute. What it is NOT bound to: the `elf_digest` the statement names.
/// Nothing inside ONE epoch program ties the root to the digest; that pin is
/// once per ELF and lives outside. Recorded as owed, not described as covered.
///
/// Returns `(z, alpha, beta)` in the host's own order.
pub fn emit_roots_block(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    carried: &[Cell],
    derived: &[LfmWord],
) -> (Ext, Ext, Ext) {
    for root in carried {
        transcript.absorb_digest(b, *root);
    }
    for word in derived {
        let root = b.digest_const(*word);
        transcript.absorb_digest(b, root.as_cell());
    }
    let z = transcript.sample_ext(b);
    let alpha = transcript.sample_ext(b);
    let beta = transcript.sample_ext(b);
    (z, alpha, beta)
}

/// What [`emit_roots_block`] costs, by the shape it comes from.
///
/// - one `Unpack` per CARRIED root, which is what `absorb_digest` emits beyond
///   its sponge (`absorb_unpack_rows`);
/// - the same for each DERIVED root, plus the `LFM_CONST` that holds it —
///   counted in [`roots_block_constants`] rather than here, because this form is
///   const-free like every other in this module;
/// - three `Pack`s, one per draw (`sample_ext_rows`);
/// - the sponge: `COORDINATES_PER_DIGEST` felts absorbed per root and three
///   extension draws.
pub fn roots_block_cost(
    carried: usize,
    derived: usize,
    entry: SpongeEntry,
) -> (usize, SpongeSchedule) {
    let mut schedule = SpongeSchedule::new(entry);
    for _ in 0..carried + derived {
        schedule.absorb(FELTS_PER_DIGEST);
    }
    for _ in 0..DRAWS {
        schedule.draw_ext();
    }
    let ops = (carried + derived) * absorb_unpack_rows() + DRAWS * sample_ext_rows();
    (ops, schedule)
}

/// The `LFM_CONST` words [`emit_roots_block`] interns, deduplicated and BY
/// VALUE — TWO kinds, and the second is invisible to any per-step form.
///
/// - one per DERIVED root, which is the whole point of the leg: a carried root
///   is arena data and interns nothing, an interned one is program text;
/// - `leaf_capacity(felts)` per hash of the block's own SCHEDULE.
///   `algebraic_leaf_hash` interns it for the leaf it is about to hash
///   (`edsl.rs:676-692`), so a program pays one per distinct leaf length —
///   instance 63's constant, the same one `StackedCost::own_constants` and
///   `table_verify_cost` take off their finished schedules.
/// - the ZERO WORD, once, when the schedule hashes at all.
///   `algebraic_leaf_hash` opens with `felt_const(zero)` and
///   `digest_const([zero; 4])` UNCONDITIONALLY — the tail padding and the
///   initial rate — and both are the same pooled word. Item 4 found the same
///   zero from the statement's side and read it as conditional on the leaf
///   being partial; it is not, it is every leaf hash.
///
/// ⚠ This form came in TWO short at eight carried roots and one derived, twice:
/// first at 3 against 1, then at 3 against 2. Neither gap was closed by adding a
/// number — the test PRINTS the words the program interns that the form does not
/// name, and both times the printout said which term was missing.
pub fn roots_block_constants(derived: &[LfmWord], schedule: &SpongeSchedule) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = Vec::new();
    for word in derived {
        if !words.contains(word) {
            words.push(*word);
        }
    }
    for hash in schedule.hashes() {
        let word = leaf_capacity(hash.felts());
        if !words.contains(&word) {
            words.push(word);
        }
    }
    if !schedule.hashes().is_empty() {
        let zero = [FE::zero(); 4];
        if !words.contains(&zero) {
            words.push(zero);
        }
    }
    words
}

/// Challenges the roots block draws: `z`, `alpha` and `beta`.
///
/// ⚠ THREE, and the third is the one a reader drops. `beta` batches the
/// constraint roots and is not read until a table's own leg, but it is SAMPLED
/// here; a block that drew two would leave the sponge in a different state and
/// every table's first challenge would differ.
const DRAWS: usize = 3;

/// Felts a 32-byte commitment occupies in the sponge's stream — the transcript's
/// own constant, not a second spelling of four.
use super::whir_transcript::DIGEST_FELTS as FELTS_PER_DIGEST;

/// What the epoch's closure leaves for a caller to look at.
///
/// Both halves come back because a gate that could only see the refusal would
/// be checking "it did not execute", and a leg that refuses everything passes
/// that. The values let the expected half be compared against the host's own
/// `compute_commit_bus_offset` by value.
pub struct ClosureWires {
    /// `Σ contribution(table)`, the bus balance.
    pub balance: Ext,
    /// `owed`, the COMMIT bus's counterparty.
    pub expected: Ext,
}

/// ★ The epoch's closure, emitted: `balance != expected` is `BusImbalance`.
///
/// `contribution` (`multilinear_table.rs:753`) is `p/q` per table and is `None`
/// when the denominator vanished; here it is a `Div`, and a division by zero has
/// no satisfying assignment, so the host's `None` and the machine's refusal are
/// the same event rather than two behaviours that have to agree.
///
/// `published` and `start_index` are PROGRAM TEXT. The statement already interns
/// the public output (item 4: the whole statement is a run of program-constant
/// bytes), and `start_index` is `register_init[X254_INDEX]` of the register file
/// this program is compiled for. So every per-byte coefficient is emit-time and
/// only `z` and `alpha` are wires.
pub fn emit_epoch_closure(
    b: &mut LfmBuilder,
    outputs: &[(Ext, Ext)],
    published: &[u8],
    start_index: u64,
    z: Ext,
    alpha: Ext,
) -> ClosureWires {
    assert!(
        !outputs.is_empty(),
        "an epoch's balance is over at least one table"
    );
    let mut balance: Option<Ext> = None;
    for (p, q) in outputs {
        let share = b.ediv(*p, *q);
        balance = Some(match balance {
            None => share,
            Some(running) => b.eadd(running, share),
        });
    }
    let balance = balance.expect("at least one table");
    let expected = emit_expected(b, published, start_index, z, alpha);
    b.assert_eq_ext(balance, expected);
    ClosureWires { balance, expected }
}

/// `compute_commit_bus_offset`, emitted — see [`expected_rows`] for the terms.
fn emit_expected(
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
/// `decode_table_index` applies (`multilinear_continuation.rs:368`).
pub enum PreprocessedRoute<'a> {
    /// Nothing left to check — either the table has no preprocessed columns, or
    /// a prepared opening settled all of them.
    None,
    /// BITWISE: the closed form, `O(NUM_VARS)` where the fold is `O(2^NUM_VARS)`.
    Bitwise,
    /// KECCAK_RC and REGISTER: one shared `eq(point, ·)` per table with each
    /// column folded against it, one row per NONZERO entry.
    ConstMle(&'a [&'a [FE]]),
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
fn emit_preprocessed_leg(
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

/// What the per-table walk costs, threaded, under ONE constant pool.
///
/// ⚠ THE CONVENTION, stated because two meet in this file. `table_verify_cost`
/// reports a PER-TABLE program's own pool, and in an epoch program those pools
/// COLLIDE — the `one` every `eq` seeds, each leaf capacity, each bus id. So
/// this unions the tables' constant WORDS through [`Cost::merge`] instead of
/// adding their `rows()`, and the pool is charged once by the caller. Adding
/// `table_verify_cost(..).rows()` over the tables is the other convention and
/// over-counts; the recount's per-table term is quoted in that one deliberately,
/// which is why the two totals differ by the collisions.
pub struct TableWalkCost {
    /// Operations and the union of every table's interned words.
    pub leg: Cost,
    /// The sponge's own rows, summed over the tables.
    pub sponge_rows: usize,
    /// Permutations, summed over the tables.
    pub perms: usize,
    /// Where the sponge is left for the group walk.
    pub entry: SpongeEntry,
}

impl TableWalkCost {
    /// INSTRUCTIONS excluding the pool, so it composes with the other
    /// const-free forms in this module.
    pub fn operations(&self) -> usize {
        self.leg.operations() + self.sponge_rows
    }
}

/// [`emit_table_walk`]'s cost, table by table, threading the sponge.
pub fn table_walk_cost(
    shapes: &[TableShape<'_>],
    plans: &[PreprocessedPlan<'_>],
    entry: SpongeEntry,
) -> TableWalkCost {
    assert_eq!(shapes.len(), plans.len(), "one preprocessed plan per table");
    let mut leg = Cost::default();
    let mut sponge_rows = 0usize;
    let mut perms = 0usize;
    let mut entry = entry;
    for (shape, plan) in shapes.iter().zip(plans) {
        let table = table_verify_cost(shape, entry);
        // The per-table form's constants come back as VALUES so the pools
        // union rather than sum; its operations are the leg's.
        leg.ops(table.leg.operations());
        for word in table.leg.constant_values() {
            leg.constant_word(*word);
        }
        sponge_rows += table.schedule.rows();
        perms += table.schedule.perms();
        entry = table.entry();
        preprocessed_leg_cost(plan, &mut leg);
    }
    TableWalkCost {
        leg,
        sponge_rows,
        perms,
        entry,
    }
}

/// One preprocessed plan's rows and constants, added into the walk's pool.
///
/// The three routes' forms are the landed ones: `bitwise_preprocessed_rows`
/// counts INSTRUCTIONS including its interned `one`, so its constant is already
/// inside the number and is added as an operation here to keep this function's
/// pool the union of the ones the emitters actually intern.
fn preprocessed_leg_cost(plan: &PreprocessedPlan<'_>, leg: &mut Cost) {
    match &plan.route {
        PreprocessedRoute::None => {}
        PreprocessedRoute::Bitwise => {
            leg.ops(bitwise_preprocessed_rows());
            leg.ops(NUM_PRECOMPUTED_COLS * ASSERT_EQ_ROWS);
        }
        PreprocessedRoute::ConstMle(columns) => {
            let remaining = &columns[plan.settled..];
            if remaining.is_empty() {
                return;
            }
            let num_vars = remaining[0].len().trailing_zeros() as usize;
            leg.ops(const_mle_rows(remaining, num_vars));
            for word in const_mle_constants(remaining) {
                leg.constant_word(word);
            }
            leg.ops(remaining.len() * ASSERT_EQ_ROWS);
        }
    }
}

/// Rows an `assert_eq_ext` lowers to: the difference and the division by zero
/// (`builder.rs:289-292`). The same constant `whir_table` names; spelled here
/// rather than imported because the two forms are counted independently and a
/// shared name would hide a disagreement.
const ASSERT_EQ_ROWS: usize = 2;

// =============================================================================
// The commitment-group walk
// =============================================================================

/// One commitment group's wires: its layout, its chains, and the domain they
/// run over.
pub struct GroupWires<'a> {
    pub layout: &'a StackedLayout,
    pub polys: &'a [StackedPolyWires<'a>],
    pub shape: &'a ChainShape,
    pub domain: &'a Domain<GoldilocksField>,
}

/// ★ The commitment-group walk, emitted: `multi_verify`'s second loop
/// (`multilinear_table.rs:1254`).
///
/// The three counters ARE this function, and they are the host's own:
/// `statement_at` advances by the group's table count, `column_at` by the sum of
/// those tables' COLUMN counts, and the roots by `layout.num_polys()`. A group
/// handed the wrong column slice settles one table's values against another
/// table's commitment, which is precisely what a counter that advanced by
/// tables rather than columns would do.
///
/// Returns each group's batching challenge, so a gate can compare them against
/// the elements the HOST verifier sampled rather than only observing that the
/// program executed.
pub fn emit_group_walk(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    groups: &[GroupWires<'_>],
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
        gammas.push(emit_stacked_verify(
            b,
            transcript,
            group.layout,
            group.polys,
            &points[column_at..column_at + width],
            &walk.values[column_at..column_at + width],
            group.shape,
            group.domain,
        ));
        statement_at += size;
        column_at += width;
    }
    gammas
}

/// ★ The PREPARED opening, emitted: `multi_verify`'s last step
/// (`multilinear_table.rs:1295`) — the same wrapper on the DECODE group.
///
/// Three things differ from a group above, and all three come from the host:
///
/// 1. the claim is `Claimed::Shared`. DECODE's five columns are settled at ONE
///    point, the prepared table's own reduced point, so the wrapper pays one
///    `eq` for the five and not five;
/// 2. the LAYOUT and the DOMAIN are DECODE's, from `stacks` over its own shape,
///    while the CHAIN CONFIG is the EPOCH's — `multi_verify` passes its own
///    `config` to this call and only the first two belong to the group;
/// 3. there is no separate check that the opened values are the table's. The
///    values handed here ARE the ones DECODE's own argument settled on, so the
///    opening proves the pinned commitment takes exactly those at exactly that
///    point. An equality someone has to remember to write is replaced by a slice
///    nobody can omit.
///
/// ⚠ The root is the INTERNED CONSTANT the roots block absorbed, re-interned by
/// the caller. `LfmBuilder::word_const` (`builder.rs:169`) keys on the canonical
/// word and hands back the same address, so the second mention costs nothing and
/// the program still holds exactly one `Const` for that root — which is what the
/// arena schema asserts.
#[allow(clippy::too_many_arguments)]
pub fn emit_prepared_group(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    layout: &StackedLayout,
    polys: &[StackedPolyWires<'_>],
    shape: &ChainShape,
    domain: &Domain<GoldilocksField>,
    point: &[Ext],
    values: &[Ext],
) -> Ext {
    let points: Vec<&[Ext]> = (0..values.len()).map(|_| point).collect();
    emit_stacked_verify(b, transcript, layout, polys, &points, values, shape, domain)
}

// =============================================================================
// The assembled epoch program
// =============================================================================

/// Everything one epoch's program is emitted against, derived ONCE.
///
/// ★ ONE DERIVATION, TWO FUNCTIONS. [`whir_epoch_program`] and
/// [`whir_epoch_arena`] are only correct relative to each other: the arena is a
/// list of words in the order the program hints them, and two derivations of
/// that order is precisely the drift this campaign keeps finding. So both build
/// this and walk it.
///
/// ⚠ The `TableLayout`s are rebuilt here rather than borrowed from the
/// verifier, because `layout_of` is private to `multilinear_continuation`. What
/// catches a drift is not this comment: every challenge downstream of a layout
/// is derived from it, so a layout that differed from the host's would give the
/// machine a different `z` and the honest proof would stop executing.
struct EpochPlan<'a> {
    config: ChainConfig,
    /// `epoch_groups(n)`.
    sizes: Vec<usize>,
    layouts: Vec<TableLayout<'a, GoldilocksField, GoldilocksExtension>>,
    buses: Vec<Vec<InteractionShape<GoldilocksExtension>>>,
    /// Each table's preprocessed columns, owned, empty for a table with none.
    preprocessed: Vec<Vec<Vec<FE>>>,
    /// Each table's name, which is what selects its preprocessed route.
    names: Vec<String>,
    group_layouts: Vec<StackedLayout>,
    group_domains: Vec<Domain<GoldilocksField>>,
    /// Where DECODE sits, found by name.
    decode_at: usize,
    decode_layout: StackedLayout,
    decode_domain: Domain<GoldilocksField>,
}

/// The four tables whose preprocessed columns an epoch carries, and the route
/// each one takes.
///
/// ⛔ A TABLE WITH PREPROCESSED COLUMNS AND NONE OF THESE NAMES IS A REFUSAL,
/// not a skip. Skipping it would drop a `check_preprocessed` with nothing put in
/// its place, which is exactly what `settled_out_of_band` is documented never to
/// become (`multilinear_table.rs:943`). A fifth preprocessed table added
/// upstream must fail this build rather than go quietly unchecked.
const BITWISE_NAME: &str = "BITWISE";
const DECODE_NAME: &str = "DECODE";
const KECCAK_RC_NAME: &str = "KECCAK_RC";
const REGISTER_NAME: &str = "REGISTER";

impl<'a> EpochPlan<'a> {
    fn build(epoch: &WhirRealEpoch, airs: EpochAirs<'a>) -> Self {
        assert_eq!(
            airs.len(),
            epoch.proof.table_num_vars.len(),
            "one AIR per table the proof states a height for"
        );
        assert_eq!(
            airs.len(),
            epoch.proof.proof.tables.len(),
            "one AIR per table the proof argues"
        );

        let shapes: Vec<(usize, usize)> = airs
            .iter()
            .zip(&epoch.proof.table_num_vars)
            .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
            .collect();
        let config = epoch.config;
        let sizes = crate::multilinear_continuation::epoch_groups(shapes.len());

        let layouts: Vec<TableLayout<'a, GoldilocksField, GoldilocksExtension>> = airs
            .iter()
            .zip(&shapes)
            .map(|(air, &(width, num_vars))| {
                TableLayout::new(
                    air.constraint_program(),
                    air.constraints_meta(),
                    air.bus_interactions(),
                    width,
                    num_vars,
                    stark::multilinear_air::Uniforms::default(),
                )
                .expect("a table the verifier accepted must lay out")
            })
            .collect();
        let buses: Vec<Vec<InteractionShape<GoldilocksExtension>>> =
            airs.iter()
                .zip(&layouts)
                .map(|(air, layout)| {
                    let slots = layout.slot_of().to_vec();
                    stark::multilinear_logup::interaction_shapes(
                        air.bus_interactions(),
                        slots.len(),
                        |column| {
                            slots.get(column).copied().ok_or(
                                multilinear::Error::UnknownPolynomial {
                                    index: column,
                                    len: slots.len(),
                                },
                            )
                        },
                    )
                    .expect("the bus probes")
                })
                .collect();
        let preprocessed: Vec<Vec<Vec<FE>>> =
            airs.iter().map(|air| air.precomputed_columns()).collect();
        let names: Vec<String> = airs.iter().map(|air| air.name().to_string()).collect();

        // ⛔ The refusal, before anything is emitted.
        for (index, (name, columns)) in names.iter().zip(&preprocessed).enumerate() {
            assert!(
                columns.is_empty()
                    || matches!(
                        name.as_str(),
                        BITWISE_NAME | DECODE_NAME | KECCAK_RC_NAME | REGISTER_NAME
                    ),
                "table {index} is named {name} and carries {} preprocessed columns, which no \
                 route covers; a table whose columns nothing checks must fail the build, not be \
                 skipped",
                columns.len()
            );
        }

        let (group_layouts, group_domains) =
            crate::multilinear_prove::stacks(&shapes, &sizes, &config)
                .expect("the epoch's groups stack");

        let decode_at = crate::multilinear_continuation::decode_table_index(airs)
            .expect("an epoch's table set carries exactly one DECODE");
        let decode_columns = &preprocessed[decode_at];
        let decode_rows = decode_columns
            .first()
            .map(Vec::len)
            .expect("DECODE carries preprocessed columns");
        let decode_vars = decode_rows.trailing_zeros() as usize;
        let decode_config = crate::multilinear_continuation::decode_prepared_config(
            decode_columns.len(),
            decode_vars,
        );
        let (mut decode_layouts, mut decode_domains) = crate::multilinear_prove::stacks(
            &[(decode_columns.len(), decode_vars)],
            &[DECODE_GROUP_POLYS],
            &decode_config,
        )
        .expect("DECODE's prepared group stacks");
        assert_eq!(
            decode_layouts.len(),
            1,
            "the prepared group is ONE stacked polynomial"
        );

        Self {
            config,
            sizes,
            layouts,
            buses,
            preprocessed,
            names,
            group_layouts,
            group_domains,
            decode_at,
            decode_layout: decode_layouts.remove(0),
            decode_domain: decode_domains.remove(0),
        }
    }

    /// The emit-time shapes, borrowed from the layouts and the buses.
    fn table_shapes(&self) -> Vec<TableShape<'_>> {
        self.layouts
            .iter()
            .zip(&self.buses)
            .map(|(layout, bus)| TableShape {
                ir: layout.shape(),
                bus,
                kinds: layout.kinds(),
                num_columns: layout.num_columns(),
                num_vars: layout.num_vars(),
            })
            .collect()
    }

    /// Each table's slot map, which the preprocessed leg addresses through.
    fn slots(&self) -> Vec<&[usize]> {
        self.layouts.iter().map(|layout| layout.slot_of()).collect()
    }

    /// The preprocessed plan per table, over views the caller owns.
    fn routes<'v>(&self, views: &'v [Vec<&'v [FE]>]) -> Vec<PreprocessedPlan<'v>> {
        self.names
            .iter()
            .zip(views)
            .zip(&self.preprocessed)
            .map(|((name, view), columns)| {
                if columns.is_empty() {
                    return PreprocessedPlan {
                        settled: 0,
                        route: PreprocessedRoute::None,
                    };
                }
                match name.as_str() {
                    BITWISE_NAME => PreprocessedPlan {
                        settled: 0,
                        route: PreprocessedRoute::Bitwise,
                    },
                    // ★ Every one of DECODE's columns is settled by the prepared
                    // opening, so nothing is emitted for it here — and `settled`
                    // is the SAME number the opening covers, never a second knob.
                    DECODE_NAME => PreprocessedPlan {
                        settled: columns.len(),
                        route: PreprocessedRoute::None,
                    },
                    KECCAK_RC_NAME | REGISTER_NAME => PreprocessedPlan {
                        settled: 0,
                        route: PreprocessedRoute::ConstMle(view),
                    },
                    other => unreachable!("the build refused {other} already"),
                }
            })
            .collect()
    }
}

/// Every word one epoch's program hints, in the order it hints them.
///
/// ⚠ THE ORDER IS THE CONTRACT between this and [`whir_epoch_program`], and
/// nothing but EXECUTION catches a disagreement: a misaligned arena hands the
/// machine somebody else's field element, and the argument stops satisfying its
/// own refusals. Both walk the same [`EpochPlan`], and the gate on the pair is
/// that the program hints exactly as many words as this writes.
pub fn whir_epoch_arena(epoch: &WhirRealEpoch, airs: EpochAirs<'_>) -> Vec<Vec<LfmWord>> {
    let plan = EpochPlan::build(epoch, airs);
    let proof = &epoch.proof.proof;
    let mut words: Vec<LfmWord> = proof
        .roots
        .iter()
        .map(super::algebraic_commit::commitment_to_digest)
        .collect();
    for table in &proof.tables {
        push_table_words(&mut words, table);
    }
    for (group, opening) in proof.columns.iter().enumerate() {
        let shape = ChainShape::new(&plan.config, plan.group_layouts[group].n_stack());
        for chain in &opening.polys {
            words.push(super::word::ext_word(&chain.final_value));
            super::whir_chain::push_round_words(&mut words, &shape, chain);
        }
    }
    // The prepared opening's chain runs at the EPOCH's config, and only its
    // layout and domain are DECODE's (`multilinear_table.rs:1310`).
    let prepared = proof
        .preprocessed
        .as_ref()
        .expect("an epoch carries DECODE's prepared opening");
    let shape = ChainShape::new(&plan.config, plan.decode_layout.n_stack());
    for chain in &prepared.polys {
        words.push(super::word::ext_word(&chain.final_value));
        super::whir_chain::push_round_words(&mut words, &shape, chain);
    }
    vec![words]
}

/// One table's proof words, in the order [`whir_epoch_program`] hints them.
fn push_table_words(
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

/// ★★ ONE EPOCH'S VERIFY, ASSEMBLED — `verify_epoch_bookend`'s body as one
/// machine program.
///
/// The order is the host's, and the order is the only part of a Fiat-Shamir
/// verifier that no per-leg gate can see. Read from `multi_verify`, by line:
/// the STATEMENT (`absorb_epoch`), the ROOTS BLOCK (`:1205`), the per-table
/// walk (`:1218`), the CLOSURE (`:1245`), the group walk (`:1254`) and the
/// PREPARED opening last (`:1295`).
///
/// ⚠ ONE QUALIFICATION, so nobody reads a gate as covering more than it does:
/// the closure emits NO transcript operation — it is divisions, adds and one
/// `assert_eq_ext` over wires the roots block already produced — so its position
/// between the two walks is a readability choice and not a soundness one. Every
/// other step's position IS load-bearing.
pub fn whir_epoch_program(epoch: &WhirRealEpoch, airs: EpochAirs<'_>) -> LfmProgram {
    let plan = EpochPlan::build(epoch, airs);
    let proof = &epoch.proof.proof;
    let words = whir_epoch_arena(epoch, airs);
    let total = words[0].len() as u32;

    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(total);
    let mut at = 0u32;

    let carried: Vec<Cell> = proof
        .roots
        .iter()
        .map(|_| {
            let cell = b.hint_word(arena, at);
            at += 1;
            cell
        })
        .collect();

    // 1. The statement, which is entirely program text.
    let mut transcript = WhirTranscript::new();
    super::whir_statement::emit_epoch_statement(
        &mut transcript,
        &super::whir_statement::EpochStatement {
            elf_digest: &epoch.elf_digest,
            epoch_label: epoch.position.label,
            public_output: epoch.public_output(),
            table_counts: &epoch.proof.table_counts,
            table_num_vars: &epoch.proof.table_num_vars,
            config: &plan.config,
        },
    );

    // 2. The roots block. The DECODE root is INTERNED, not hinted — see this
    //    module's header for what that constant is and is not bound to.
    // ⛔ THE PREPARED ROOTS, NOT `decode_commitment`. The two are different
    // commitments with confusable names: `decode_commitment` is the UNIVARIATE
    // one that reaches `build_epoch_airs` and that the multilinear path never
    // compares, while what `absorb_roots_and_challenge` absorbs after the
    // carried roots is the prepared multilinear stack's. Absorbing the wrong
    // one derives a different `z` and every leg below it, with nothing naming
    // the cause — which is exactly how this was found.
    let derived: Vec<LfmWord> = epoch
        .decode_prepared_roots
        .iter()
        .map(super::algebraic_commit::commitment_to_digest)
        .collect();
    let (z, alpha, beta) = emit_roots_block(&mut b, &mut transcript, &carried, &derived);

    // The shared alpha ladder. ⚠ EPOCH-LEVEL: `emit_interaction` reads
    // `alpha_powers[i + 1]` and one ladder of the longest table's length serves
    // every table, so `table_verify_cost` does not charge it and the assembled
    // form names it here.
    let shapes = plan.table_shapes();
    let ladder_len = shapes
        .iter()
        .map(|shape| super::whir_bus::alpha_powers_read(shape.bus))
        .max()
        .unwrap_or(1);
    let ladder = super::whir_poly::emit_challenge_powers(&mut b, alpha, ladder_len);

    // 3. The per-table walk, with each table's proof wires hinted first.
    let mut store = Vec::with_capacity(proof.tables.len());
    for table in &proof.tables {
        store.push(hint_table_wires(&mut b, arena, &mut at, table));
    }
    let wires: Vec<TableProofWires<'_>> = store.iter().map(TableWires::borrow).collect();
    let views: Vec<Vec<&[FE]>> = plan
        .preprocessed
        .iter()
        .map(|columns| columns.iter().map(Vec::as_slice).collect())
        .collect();
    let routes = plan.routes(&views);
    let slots = plan.slots();
    let walk = emit_table_walk(
        &mut b,
        &mut transcript,
        &wires,
        &shapes,
        &routes,
        &slots,
        z,
        &ladder,
        beta,
    );

    // 4. The closure: the bus balance against the epoch's expected value.
    let start_index = u64::from(epoch.position.register_init[crate::tables::register::X254_INDEX]);
    emit_epoch_closure(
        &mut b,
        &walk.outputs,
        epoch.public_output(),
        start_index,
        z,
        alpha,
    );

    // 5. The commitment groups.
    let mut chains = Vec::with_capacity(proof.columns.len());
    for (group, opening) in proof.columns.iter().enumerate() {
        let shape = ChainShape::new(&plan.config, plan.group_layouts[group].n_stack());
        chains.push(hint_group_chains(&mut b, arena, &mut at, opening, &shape));
    }
    let group_shapes: Vec<ChainShape> = plan
        .group_layouts
        .iter()
        .map(|layout| ChainShape::new(&plan.config, layout.n_stack()))
        .collect();
    // ⚠ The openings and the round wires are LOCALS, because a chain's wires
    // borrow its openings and its openings borrow its storage. Returning them
    // from a helper would mean leaking them; keeping the whole ladder in one
    // scope is what the chain suite does and costs nothing.
    let group_openings: Vec<Vec<_>> = chains
        .iter()
        .map(|held| {
            held.storage
                .iter()
                .map(super::whir_chain::RoundStorage::openings)
                .collect::<Vec<_>>()
        })
        .collect();
    let group_rounds: Vec<Vec<Vec<super::whir_chain::ChainRoundWires<'_>>>> = chains
        .iter()
        .zip(&group_openings)
        .map(|(held, openings)| {
            held.storage
                .iter()
                .zip(openings)
                .map(|(chain, (current, next))| chain.wires(current, next))
                .collect()
        })
        .collect();
    let group_wires: Vec<Vec<StackedPolyWires<'_>>> = {
        let mut root_at = 0usize;
        let mut out = Vec::with_capacity(chains.len());
        for (group, held) in chains.iter().enumerate() {
            out.push(
                (0..held.finals.len())
                    .map(|poly| StackedPolyWires {
                        rounds: &group_rounds[group][poly],
                        root: carried[root_at + poly],
                        final_value: held.finals[poly],
                    })
                    .collect(),
            );
            root_at += plan.group_layouts[group].num_polys();
        }
        out
    };
    let groups: Vec<GroupWires<'_>> = (0..chains.len())
        .map(|group| GroupWires {
            layout: &plan.group_layouts[group],
            polys: &group_wires[group],
            shape: &group_shapes[group],
            domain: &plan.group_domains[group],
        })
        .collect();
    emit_group_walk(&mut b, &mut transcript, &groups, &plan.sizes, &walk);

    // 6. The prepared opening, last, on DECODE's own layout and domain at the
    //    EPOCH's config. Its root is the SAME interned constant the roots block
    //    absorbed: `word_const` keys on the canonical word, so this re-mention
    //    costs no row and the program still holds exactly one `Const` for it.
    let prepared = proof
        .preprocessed
        .as_ref()
        .expect("an epoch carries DECODE's prepared opening");
    let prepared_shape = ChainShape::new(&plan.config, plan.decode_layout.n_stack());
    let held = hint_group_chains(&mut b, arena, &mut at, prepared, &prepared_shape);
    let decode_roots: Vec<Cell> = derived
        .iter()
        .map(|word| b.digest_const(*word).as_cell())
        .collect();
    let prepared_openings: Vec<_> = held
        .storage
        .iter()
        .map(super::whir_chain::RoundStorage::openings)
        .collect();
    let prepared_rounds: Vec<Vec<super::whir_chain::ChainRoundWires<'_>>> = held
        .storage
        .iter()
        .zip(&prepared_openings)
        .map(|(chain, (current, next))| chain.wires(current, next))
        .collect();
    let prepared_polys: Vec<StackedPolyWires<'_>> = (0..held.finals.len())
        .map(|poly| StackedPolyWires {
            rounds: &prepared_rounds[poly],
            root: decode_roots[poly],
            final_value: held.finals[poly],
        })
        .collect();
    let column_at = walk.column_at(plan.decode_at);
    let settled = plan.preprocessed[plan.decode_at].len();
    emit_prepared_group(
        &mut b,
        &mut transcript,
        &plan.decode_layout,
        &prepared_polys,
        &prepared_shape,
        &plan.decode_domain,
        &walk.points[plan.decode_at],
        &walk.values[column_at..column_at + settled],
    );

    assert_eq!(
        at, total,
        "the program must hint exactly the words the arena writes"
    );
    b.public(z.as_cell());
    let program = super::compiler::compile(b.finish());
    super::validator::validate(&program).expect("an epoch program must be admissible");
    program
}

/// One table's hinted wires, OWNED, because `TableProofWires` borrows them.
struct TableWires {
    bus_output: (Ext, Ext),
    gkr: Vec<super::whir_gkr::GkrLayerWires>,
    sumcheck: Vec<Vec<Ext>>,
    factor_values: Vec<Ext>,
    reduce_sumcheck: Vec<Vec<Ext>>,
    column_values: Vec<Ext>,
}

impl TableWires {
    fn borrow(&self) -> TableProofWires<'_> {
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

/// Hints one table's proof, in [`push_table_words`]' order.
fn hint_table_wires(
    b: &mut LfmBuilder,
    arena: super::instr::ArenaId,
    at: &mut u32,
    table: &stark::multilinear_table::TableProof<GoldilocksExtension>,
) -> TableWires {
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
    let mut gkr = Vec::with_capacity(table.gkr.layers.len());
    for layer in &table.gkr.layers {
        let sumcheck: Vec<Vec<Ext>> = layer
            .sumcheck
            .rounds
            .iter()
            .map(|round| take(b, round.evaluations.len()))
            .collect();
        let halves = take(b, 4);
        gkr.push(super::whir_gkr::GkrLayerWires {
            sumcheck,
            p_lo: halves[0],
            p_hi: halves[1],
            q_lo: halves[2],
            q_hi: halves[3],
        });
    }
    let sumcheck: Vec<Vec<Ext>> = table
        .constraint
        .sumcheck
        .rounds
        .iter()
        .map(|round| take(b, round.evaluations.len()))
        .collect();
    let factor_values = take(b, table.constraint.factor_values.len());
    let reduce_sumcheck: Vec<Vec<Ext>> = table
        .constraint
        .reduce
        .sumcheck
        .rounds
        .iter()
        .map(|round| take(b, round.evaluations.len()))
        .collect();
    let column_values = take(b, table.constraint.reduce.column_values.len());
    TableWires {
        bus_output: (output[0], output[1]),
        gkr,
        sumcheck,
        factor_values,
        reduce_sumcheck,
        column_values,
    }
}

/// One commitment group's hinted chains, OWNED for the same reason.
///
/// ⛔ The ROOTS ARE NOT HERE. A group's roots are the cells the roots block
/// already hinted, and [`Self::polys`] takes them by reference: a second copy
/// would let a prover absorb one root into the transcript and open the chain
/// against another, with an honest arena looking identical to every value gate.
struct GroupChains {
    finals: Vec<Ext>,
    storage: Vec<super::whir_chain::RoundStorage>,
}

/// Hints one group's chains, in [`whir_epoch_arena`]'s order.
fn hint_group_chains(
    b: &mut LfmBuilder,
    arena: super::instr::ArenaId,
    at: &mut u32,
    opening: &multilinear::stacked_eval::StackedProof<GoldilocksField, GoldilocksExtension>,
    shape: &ChainShape,
) -> GroupChains {
    let mut finals = Vec::with_capacity(opening.polys.len());
    let mut storage = Vec::with_capacity(opening.polys.len());
    for _ in &opening.polys {
        finals.push(b.hint_word(arena, *at).as_ext());
        *at += 1;
        storage.push(super::whir_chain::RoundStorage::hint(b, arena, *at, shape));
        *at += super::whir_chain::RoundStorage::words(shape);
    }
    GroupChains { finals, storage }
}

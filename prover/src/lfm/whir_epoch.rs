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

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::leaf_capacity;
use super::builder::{Cell, Ext, LfmBuilder};
use super::whir_chain::ChainShape;
use super::whir_stacked::{StackedCost, stacked_verify_cost};
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

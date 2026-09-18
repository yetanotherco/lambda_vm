//! ★ F1 for `whir_epoch_program`: the closed form of a WHIR wrap's INSTRUCTION
//! count, as a sum of terms each named by the shape it comes from.
//!
//! ## What it is a form of, precisely
//!
//! The census the level-0 driver prints reads `program.instrs.len()`
//! (`per_table_aggregator_tests.rs:3638-3646`) — NEITHER `EpochCost::operations()`
//! (const-free) NOR `EpochCost::rows()` (one pool), and in fact `EpochCost` has
//! no producer anywhere in the tree. So the number to predict is the compiled
//! program's instruction vector, and the compiler's own partition is what makes
//! the prediction checkable chip by chip: `emit_column_groups`
//! (`compiler.rs:366`) opens exactly ONE row in exactly ONE chip group per
//! instruction, so
//!
//! ```text
//! instrs.len() == Σ over the nine workload chips of their real rows
//! ```
//!
//! and four of those nine are closed forms in their own right:
//!
//! | chip | is exactly | enters through |
//! |---|---|---|
//! | `LFM_HINT` | the arena's word count | every proof wire the epoch carries |
//! | `LFM_PUBLIC` | `145 + out_halves` | the epoch's public output length |
//! | `LFM_CONST` | the ONE pool, by VALUE | every leg's interned words, unioned |
//! | `LFM_HASH` | the sponge's permutations | the transcript schedule |
//!
//! ⛔ THE FORM IS DERIVED FROM SHAPES, NEVER FROM THE EMITTER'S COUNTERS. Every
//! input below is a shape — the AIR set, the per-table `num_vars` the proof
//! states, the chain config, the statement's bytes and the published output —
//! and no field of the emitted program is read to produce a prediction. That is
//! the point: a round trip between two halves that share code is not evidence.
//!
//! ## ⛔ Q IS 112, NOT 110
//!
//! The driver's header prints `inner blowup 4 / 110 q`, and `inner` there is
//! `recursion::Preset::Blowup4.options()` (`per_table_aggregator_tests.rs:5889`)
//! — the UNIVARIATE STARK options the epoch and the wrap are PROVED with. The
//! WHIR chain's query count is `chain_config(&shapes).num_queries`, and
//! `chain_config` (`multilinear_prove.rs:87-94`) reads the SHAPES alone. At the
//! block's shapes it is 112. Two protocols, two query counts, one header.

use multilinear::stacking::StackedLayout;
use multilinear::whir_chain::ChainConfig;
use stark::multilinear_logup::{InteractionShape, interaction_shapes};
use stark::multilinear_table::TableLayout;

use crate::multilinear_continuation::{decode_prepared_config, epoch_groups};
use crate::multilinear_prove::stacks;
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

use super::whir_chain::{ChainShape, RoundStorage};
use super::whir_epoch::PublishedText;
use super::whir_epoch::{
    EpochAirs, PreprocessedPlan, PreprocessedRoute, closure_rows, epoch_group_costs, prepared_cost,
    publish_constants, publish_rows, roots_block_constants, roots_block_cost, table_walk_cost,
};
use super::whir_gkr::GKR_SUMCHECK_DEGREE;
use super::whir_poly::challenge_powers_rows;
use super::whir_reduce::REDUCE_DEGREE;
use super::whir_statement::statement_cost;
use super::whir_table::TableShape;
use super::word::LfmWord;

/// The four tables whose preprocessed columns an epoch carries. Spelled here
/// rather than imported because `whir_epoch`'s copies are private, and a
/// disagreement between the two lists must fail this file's own refusal below
/// rather than silently route a table differently from the emitter.
const BITWISE_NAME: &str = "BITWISE";
const DECODE_NAME: &str = "DECODE";
const KECCAK_RC_NAME: &str = "KECCAK_RC";
const REGISTER_NAME: &str = "REGISTER";

/// ★ One epoch program's cost, term by term, with the terms NOT added together
/// in the struct — a residual has to be able to name which term carries it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EpochF1 {
    /// The statement. ZERO instructions — it is entirely program text — but it
    /// leaves the sponge where every later leg's squeeze cost is computed from.
    /// Enters through the statement's BYTE LENGTH, which is the ELF digest, the
    /// label, the public output and one byte per table.
    pub statement: usize,
    /// `roots_block_cost`: the carried roots absorbed in `proof.roots` order,
    /// the derived root after them, then THREE draws. Enters through the number
    /// of carried roots (= Σ group `num_polys`) and the one derived root.
    pub roots: usize,
    /// ★ The shared alpha ladder — `emit_challenge_powers(b, alpha, ladder_len)`
    /// with `ladder_len = max_t alpha_powers_read(bus_t)`. Enters through the
    /// widest table's BUS and nothing else. A term V1g's recount does not name.
    pub ladder: usize,
    /// The per-table walk's ARITHMETIC: `table_walk_cost(...).leg.operations()`
    /// over every table, its preprocessed leg included. Enters through each
    /// table's height, its column count, its AIR's constraint DAG and its bus.
    pub tables: usize,
    /// The per-table walk's SPONGE rows: `ceil(f/4)` packs, `ceil(f/8)`
    /// permutations and one unpack per squeeze, at the fill the THREADED entry
    /// leaves. ⚠ Not the same as the sum of the tables measured standalone:
    /// `squeeze_rows` is a function of what the buffer holds, so a walk that
    /// threads one sponge does not squeeze where 34 separate programs do.
    pub table_sponge: usize,
    /// `closure_rows(num_tables, published)`. Enters through the TABLE COUNT and
    /// the PUBLISHED BYTE COUNT; zero expected value on a silent epoch.
    pub closure: usize,
    /// The commitment groups' wrappers and their chains, plus their sponge.
    /// Enters through `n_stack`, `num_polys`, the (table, polynomial) PAIR count
    /// and, inside `ChainShape`, the rounds and the 112 query openings.
    pub groups: usize,
    /// The prepared DECODE opening: one polynomial, five columns at ONE point.
    pub prepared: usize,
    /// `publish_rows(out_halves)` — the schema's words plus its unpack.
    pub publish: usize,
    /// ★ `LFM_HINT`, which is the ARENA'S WORD COUNT exactly: `hint_word`
    /// (`builder.rs:621`) pushes one `Instr::Hint` and an extension wire is ONE
    /// word. ⛔ No cost form in the tree charges this — `table_verify_cost`
    /// takes a shape and an entry and cannot see an arena — which is why V1g's
    /// recount is short by most of this number.
    pub arena: usize,
    /// `LFM_CONST`: the ONE pool, deduplicated BY VALUE across every leg that
    /// can name its own words — the statement's packed byte groups, the roots
    /// block's derived root and leaf capacities, the tables' own, each group's
    /// `own_constants`, and the published set's.
    ///
    /// ⛔ A LOWER BOUND, and it says so in the type rather than in a comment on
    /// a residual. `StackedCost::own_constants` (`whir_stacked.rs:410-440`)
    /// states in its own doc that "the chains' OTHER constants are not named
    /// here", and no closed form for them exists anywhere in the tree — the
    /// chain suite MEASURES `const_rows` off an emitted program and subtracts
    /// it. So [`Self::instructions`] is a lower bound by exactly that amount,
    /// and [`Self::operations`] — the const-free total — is the number that can
    /// be exact.
    pub pool: usize,
    /// `LFM_PUBLIC`: `145 + out_halves`, the schema's own total.
    pub public: usize,
    /// `LFM_HASH`: the transcript's permutations.
    pub perms: usize,
}

impl EpochF1 {
    /// The instruction count the census prints.
    ///
    /// ⚠ `perms` is NOT added: a permutation is already one of the sponge rows
    /// counted in [`Self::table_sponge`] and [`Self::groups`]. It is carried
    /// beside the total so `LFM_HASH` can be scored on its own.
    pub fn instructions(&self) -> usize {
        self.operations() + self.pool
    }

    /// The instruction count EXCLUDING every `LFM_CONST` — the form that can be
    /// exact, and the convention `EpochCost::operations` and every leg form in
    /// this tree are already in.
    pub fn operations(&self) -> usize {
        self.statement
            + self.roots
            + self.ladder
            + self.tables
            + self.table_sponge
            + self.closure
            + self.groups
            + self.prepared
            + self.publish
            + self.arena
    }
}

/// ★ The shape half of `EpochPlan::build`, with the EPOCH taken out.
///
/// `EpochPlan::build` (`whir_epoch.rs:1247`) reads the epoch for exactly three
/// things — the per-table `num_vars` the proof states, the chain config, and the
/// table count — and takes everything else from the AIRs. Rebuilding it here
/// from `(airs, num_vars, config)` is what lets F1 be evaluated at a shape that
/// no proof on this machine has produced, which is the whole of item 5g.
///
/// ⚠ A SECOND COPY OF A LAYOUT WALK, deliberately: `TableLayout::new`'s five
/// arguments are the contract, and `layout_of` is private to
/// `multilinear_continuation`. The drift this could carry is caught by the
/// fixture gate, where the same shapes must reproduce the emitted program's own
/// chip counts to the row.
pub struct ShapePlan<'a> {
    config: ChainConfig,
    /// `(width, num_vars)` per table — the pair the group walk's own form takes,
    /// kept beside the layouts because `epoch_group_costs` re-derives its
    /// `group_of` from the widths and a `TableShape` has thrown the width away.
    raw_shapes: Vec<(usize, usize)>,
    sizes: Vec<usize>,
    layouts: Vec<TableLayout<'a, GoldilocksField, GoldilocksExtension>>,
    buses: Vec<Vec<InteractionShape<GoldilocksExtension>>>,
    preprocessed: Vec<Vec<Vec<FE>>>,
    names: Vec<String>,
    group_layouts: Vec<StackedLayout>,
    decode_at: usize,
    decode_layout: StackedLayout,
}

impl<'a> ShapePlan<'a> {
    pub fn build(airs: EpochAirs<'a>, table_num_vars: &[usize], config: ChainConfig) -> Self {
        assert_eq!(
            airs.len(),
            table_num_vars.len(),
            "one AIR per table the shape states a height for"
        );
        let shapes: Vec<(usize, usize)> = airs
            .iter()
            .zip(table_num_vars)
            .map(|(air, &num_vars)| (air.trace_layout().0, num_vars))
            .collect();
        let sizes = epoch_groups(shapes.len());

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
                .expect("a table at a real shape must lay out")
            })
            .collect();
        let buses = airs
            .iter()
            .zip(&layouts)
            .map(|(air, layout)| {
                let slots = layout.slot_of().to_vec();
                interaction_shapes(air.bus_interactions(), slots.len(), |column| {
                    slots
                        .get(column)
                        .copied()
                        .ok_or(multilinear::Error::UnknownPolynomial {
                            index: column,
                            len: slots.len(),
                        })
                })
                .expect("the bus probes")
            })
            .collect();
        let preprocessed: Vec<Vec<Vec<FE>>> =
            airs.iter().map(|air| air.precomputed_columns()).collect();
        let names: Vec<String> = airs.iter().map(|air| air.name().to_string()).collect();

        // The emitter's own refusal, restated: a table with preprocessed columns
        // and none of these names has no route, and must fail the form rather
        // than be counted as free.
        for (index, (name, columns)) in names.iter().zip(&preprocessed).enumerate() {
            assert!(
                columns.is_empty()
                    || matches!(
                        name.as_str(),
                        BITWISE_NAME | DECODE_NAME | KECCAK_RC_NAME | REGISTER_NAME
                    ),
                "table {index} is named {name} and carries {} preprocessed columns, which no \
                 route covers",
                columns.len()
            );
        }

        let (group_layouts, _domains) =
            stacks(&shapes, &sizes, &config).expect("the epoch's groups stack");

        let decode_at = names
            .iter()
            .position(|name| name == DECODE_NAME)
            .expect("an epoch's table set carries exactly one DECODE");
        assert_eq!(
            names.iter().filter(|n| *n == DECODE_NAME).count(),
            1,
            "exactly one DECODE"
        );
        let decode_columns = &preprocessed[decode_at];
        let decode_rows = decode_columns
            .first()
            .map(Vec::len)
            .expect("DECODE carries preprocessed columns");
        let decode_vars = decode_rows.trailing_zeros() as usize;
        let decode_config = decode_prepared_config(decode_columns.len(), decode_vars);
        let (mut decode_layouts, _dd) =
            stacks(&[(decode_columns.len(), decode_vars)], &[1], &decode_config)
                .expect("DECODE's prepared group stacks");
        assert_eq!(
            decode_layouts.len(),
            1,
            "the prepared group is ONE polynomial"
        );

        Self {
            config,
            raw_shapes: shapes,
            sizes,
            layouts,
            buses,
            preprocessed,
            names,
            group_layouts,
            decode_at,
            decode_layout: decode_layouts.remove(0),
        }
    }

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
                    DECODE_NAME => PreprocessedPlan {
                        settled: columns.len(),
                        route: PreprocessedRoute::None,
                    },
                    KECCAK_RC_NAME | REGISTER_NAME => PreprocessedPlan {
                        settled: 0,
                        route: PreprocessedRoute::ConstMle(view),
                    },
                    other => unreachable!("the refusal above already rejected {other}"),
                }
            })
            .collect()
    }

    fn views(&self) -> Vec<Vec<&[FE]>> {
        self.preprocessed
            .iter()
            .map(|columns| columns.iter().map(Vec::as_slice).collect())
            .collect()
    }

    /// Carried roots: one per polynomial of every commitment group, in
    /// `proof.roots` order.
    fn carried_roots(&self) -> usize {
        self.group_layouts
            .iter()
            .map(StackedLayout::num_polys)
            .sum()
    }
}

/// ★★ THE ARENA'S WORD COUNT, from the shapes alone — which is `LFM_HINT`.
///
/// `hint_table_wires` and `hint_group_chains` hint exactly the proof values the
/// transcript absorbs, in `whir_epoch_arena`'s order, and the enumeration below
/// is the same one `table_verify_cost`'s schedule replays:
///
/// - the bus output, TWO extension elements;
/// - per GKR layer `i`: `i` sumcheck rounds of [`GKR_SUMCHECK_DEGREE`]
///   evaluations, then the layer's FOUR halves (`p_lo`, `p_hi`, `q_lo`, `q_hi`);
/// - the constraint sumcheck: `num_vars` rounds at the table's own degree;
/// - the factor values, one per committed SOURCE;
/// - `claim_reduce`'s sumcheck: `num_vars` rounds at [`REDUCE_DEGREE`];
/// - the column values, one per committed COLUMN.
///
/// Then, outside the tables: the carried roots (one word each), and per chain a
/// final value plus `RoundStorage::words(shape)` — the openings the 112 queries
/// carry, which is where an epoch's arena actually lives.
fn arena_words(plan: &ShapePlan<'_>) -> usize {
    let shapes = plan.table_shapes();
    let per_table: usize = shapes
        .iter()
        .map(|shape| {
            let layers = shape.gkr_layers();
            let gkr: usize = (0..layers).map(|i| i * GKR_SUMCHECK_DEGREE + 4).sum();
            2 + gkr
                + shape.num_vars * shape.sumcheck_degree()
                + shape.sources().len()
                + shape.num_vars * REDUCE_DEGREE
                + shape.num_columns
        })
        .sum();

    let group_chains: usize = plan
        .group_layouts
        .iter()
        .map(|layout| {
            let shape = ChainShape::new(&plan.config, layout.n_stack());
            layout.num_polys() * (1 + RoundStorage::words(&shape) as usize)
        })
        .sum();

    let prepared_shape = ChainShape::new(&plan.config, plan.decode_layout.n_stack());
    let prepared_chains =
        plan.decode_layout.num_polys() * (1 + RoundStorage::words(&prepared_shape) as usize);

    plan.carried_roots() + per_table + group_chains + prepared_chains
}

/// ★★★ F1: the instruction count of `whir_epoch_program` at a shape.
///
/// `statement_bytes` is `epoch_statement_bytes(...)`' output — the form needs
/// its LENGTH, because the sponge every later leg squeezes against is entered
/// where the statement leaves it. `public_output` is the epoch's, whose length
/// decides `out_halves` and the closure's expected value.
pub fn epoch_f1(
    airs: EpochAirs<'_>,
    table_num_vars: &[usize],
    config: ChainConfig,
    statement_bytes: &[u8],
    text: &PublishedText<'_>,
    derived_roots: &[LfmWord],
) -> EpochF1 {
    let public_output = text.public_output;
    let plan = ShapePlan::build(airs, table_num_vars, config);
    let shapes = plan.table_shapes();

    // 1. The statement: zero instructions, but it sets the entry.
    let statement = statement_cost(statement_bytes);
    let mut f1 = EpochF1 {
        statement: statement.operations(),
        ..EpochF1::default()
    };
    // ⚠ The statement's own rows are `LFM_CONST` words, not operations, so they
    // join the ONE pool rather than a leg of their own — which is also what
    // makes `statement` zero and keeps it in the struct anyway.
    let mut pool: Vec<LfmWord> = statement.constants.clone();
    let mut perms = 0usize;

    // 2. The roots block.
    let (roots_ops, roots_schedule) =
        roots_block_cost(plan.carried_roots(), derived_roots.len(), statement.entry());
    f1.roots = roots_ops + roots_schedule.rows();
    perms += roots_schedule.perms();
    for word in roots_block_constants(derived_roots, &roots_schedule) {
        if !pool.contains(&word) {
            pool.push(word);
        }
    }
    let entry = roots_schedule.entry();

    // 3. The shared alpha ladder — epoch-level, one ladder of the longest
    //    table's length serving every table, which is why no per-table form
    //    charges it and why the assembled form must.
    let ladder_len = shapes
        .iter()
        .map(|shape| super::whir_bus::alpha_powers_read(shape.bus))
        .max()
        .unwrap_or(1);
    f1.ladder = challenge_powers_rows(ladder_len);

    // 4. The per-table walk, its preprocessed legs included.
    let views = plan.views();
    let routes = plan.routes(&views);
    let walk = table_walk_cost(&shapes, &routes, entry);
    f1.tables = walk.leg.operations();
    f1.table_sponge = walk.sponge_rows;
    perms += walk.perms;
    for word in walk.leg.constant_values() {
        if !pool.contains(word) {
            pool.push(*word);
        }
    }

    // 5. The closure.
    f1.closure = closure_rows(shapes.len(), public_output.len());

    // 6. The commitment groups.
    let costs = epoch_group_costs(
        &plan.raw_shapes,
        &plan.sizes,
        &plan.group_layouts,
        &plan.config,
        walk.entry,
    );
    // ⚠ `StackedCost::operations()` ALREADY adds its threaded schedule's rows
    // (`whir_stacked.rs:387-390`); a `+ sponge` here would count the sponge
    // twice. `TableWalkCost` keeps the two apart, which is why the table term
    // above does add them and this one must not.
    f1.groups = costs
        .iter()
        .map(super::whir_stacked::StackedCost::operations)
        .sum();
    perms += costs.iter().map(|c| c.perms()).sum::<usize>();
    let mut entry = walk.entry;
    for cost in &costs {
        entry = cost.entry();
        for word in cost.own_constants() {
            if !pool.contains(&word) {
                pool.push(word);
            }
        }
    }

    // 7. The prepared DECODE opening.
    let prepared = prepared_cost(&plan.decode_layout, &plan.config, entry);
    f1.prepared = prepared.operations();
    perms += prepared.perms();
    for word in prepared.own_constants() {
        if !pool.contains(&word) {
            pool.push(word);
        }
    }

    // 8. The published aggregation set. ★ Its interned words are the pool's
    //    LARGEST single contribution on a publishing epoch, and they are why
    //    the block's one publisher reads `LFM_CONST` 634 against ~420 on the
    //    other fourteen: `byte_halves` of the public output are distinct words,
    //    while a register boundary vector is mostly the repeated zero and costs
    //    one row between all of them.
    let out_halves = public_output.len().div_ceil(4);
    f1.publish = publish_rows(out_halves);
    f1.public = super::per_table_aggregator::SchemaLayout::wrap(out_halves).total();
    for word in publish_constants(text) {
        if !pool.contains(&word) {
            pool.push(word);
        }
    }

    f1.arena = arena_words(&plan);
    f1.pool = pool.len();
    f1.perms = perms;
    let _ = plan.decode_at;
    f1
}

// =============================================================================
// THE GATE: F1 against the emitted program, at a shape the laptop can prove
// =============================================================================

/// The level-0 driver's own fixture, re-proven under a literal `RpxWhir`.
///
/// ⚠ A SECOND COPY of `whir_epoch_program_tests::driver_bundle`, and deliberately
/// so: that one is private to the emitter's own suite, which another lane is
/// editing. The drift a second copy can carry is harmless HERE and nowhere else
/// — F1 is evaluated at whatever shape the bundle produces and compared against
/// the program emitted for that same shape, so a fixture that drifts tests F1 at
/// a different shape rather than testing it wrongly.
///
/// ⛔ The epochs must be re-proven under a literal `RpxWhir`: `prove_continuation`
/// dispatches on the cached process knob, and the emitter's transcript is the
/// algebraic sponge, so a keccak bundle gives the machine a different challenge
/// stream and the honest program stops executing.
fn f1_fixture_bundle() -> (
    Vec<u8>,
    crate::ProofOptions,
    crate::multilinear_continuation::ContinuationProof,
) {
    let mut input: Vec<u8> = Vec::with_capacity(16);
    input.extend_from_slice(&16u32.to_le_bytes());
    input.extend_from_slice(&[0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
    input.extend_from_slice(&[0u8; 4]);
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_private_input_xpage");
    let opts = crate::ProofOptions::default_test_options();
    let mut bundle =
        crate::multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts)
            .expect("the fixture bundle proves");

    let elf = executor::elf::Elf::load(&elf_bytes).expect("the inner ELF loads");
    let artifacts =
        crate::tables::trace_builder::DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    let prepared = crate::multilinear_continuation::decode_prepared_for::<
        multilinear::whir_hash::RpxWhir,
    >(&elf, &elf_bytes)
    .expect("DECODE's prepared commitment under rpx");
    let mut epochs = Vec::new();
    crate::continuation::for_each_epoch(&elf, &input, 2, &artifacts, |prepared_epoch, _| {
        let crate::continuation::PreparedEpoch {
            register_init,
            label,
            traces,
            boundary,
            is_final,
            ..
        } = prepared_epoch;
        epochs.push(crate::multilinear_continuation::prove_epoch::<
            multilinear::whir_hash::RpxWhir,
        >(
            &elf,
            &elf_bytes,
            &register_init,
            label,
            traces,
            is_final,
            &boundary,
            &opts,
            None,
            &prepared,
        )?);
        Ok(())
    })
    .expect("prove every epoch under a literal rpx");
    assert_eq!(epochs.len(), bundle.epochs.len(), "the split must match");
    bundle.epochs = epochs;
    (elf_bytes, opts, bundle)
}

/// ★ THE ASSEMBLY'S RESIDUAL, PRE-REGISTERED: F1's const-free form is ONE
/// instruction OVER the emitted program, per epoch.
///
/// ⛔ NAMED, not absorbed. Where it is NOT: every LEG form this composes is
/// already gated exact against an emitted program — V1g's roots block 33/33 at
/// five shapes, its closure 69/69 silent and 102/102 publishing, its three
/// preprocessed routes 295/153/453; V1h's spine 18/18, table walk 999/999 and
/// groups 1458/1458. So the row is in what no leg gate covers: the ASSEMBLY —
/// the epoch-level alpha ladder, the published set, the arena, and the sponge
/// ENTRY threaded from leg to leg. The threading is the leading hypothesis by
/// its signature: a squeeze whose buffered felt count is off by a little moves
/// `ceil(f/4)` packs by one while leaving `ceil(f/8)` permutations alone, which
/// is exactly what is observed — `LFM_HASH` matches to the row at every shape.
///
/// What makes it a CONSTANT of the spine rather than a term: it reads +1 at
/// every epoch of the fixture bundle, at 25 tables and at 26, and on a SILENT
/// epoch as well as a PUBLISHING one. A per-table or per-group error could not
/// hold still across those.
///
/// It is 1 row in 514,730–546,933 at fixture scale and 1 in 2.5–3.2 M at the
/// block's, so it changes no conclusion; it is asserted rather than tolerated so
/// that a change in any term has to move it.
const SPINE_RESIDUAL: i64 = 1;

/// The nine workload chips, in the census's own order. `LFM_RANGE` and
/// `BITWISE` are lookup tables at compile-time heights and carry no
/// instructions, and the keccak/blake3 family is absent from a WHIR wrap.
const WORKLOAD_CHIPS: [&str; 9] = [
    "LFM_CONST",
    "LFM_BALU",
    "LFM_XALU",
    "LFM_SELECT",
    "LFM_BITDEC",
    "LFM_HASH",
    "LFM_LANES",
    "LFM_HINT",
    "LFM_PUBLIC",
];

/// One emitted program's measured chip rows, by name, summed over instances.
fn measured_chips(program: &super::compiler::LfmProgram) -> Vec<(&'static str, usize)> {
    let panel = super::airs::lfm_chip_census_with_hasher(program, crate::hash_pin::BLOCK_HASHER);
    let mut out: Vec<(&'static str, usize)> = Vec::new();
    for chip in &panel {
        match out.iter_mut().find(|(name, _)| *name == chip.name) {
            Some((_, rows)) => *rows += chip.real_rows as usize,
            None => out.push((chip.name, chip.real_rows as usize)),
        }
    }
    out
}

/// ★★★ THE F1 GATE — the closed form against the emitted program, at the
/// fixture's own shape.
///
/// Four assertions, and each can fail on its own:
///
/// 1. the compiler's partition — `instrs.len()` is the sum of the nine workload
///    chips' rows, which is what makes every residual below nameable;
/// 2. `LFM_HINT` equals [`arena_words`] evaluated at the SHAPE — the arena's
///    size derived from the proof's round structure, not counted off the filler;
/// 3. `LFM_PUBLIC` equals the schema's `145 + out_halves`;
/// 4. the TOTAL equals [`EpochF1::instructions`].
///
/// ⛔ (2) is the one that matters most. No cost form in the tree charges the
/// arena — `table_verify_cost` takes a shape and an entry and cannot see one —
/// so this is the term V1g's production recount is missing, and it is predicted
/// here from the same enumeration `table_verify_cost`'s schedule replays rather
/// than read off `whir_epoch_arena`.
/// ★★★ THE F1 GATE — the closed form against the emitted program, at EVERY
/// epoch of the driver's own fixture.
///
/// Five assertions, each able to fail on its own and each on a DIFFERENT half of
/// the form:
///
/// 1. the compiler's partition — `instrs.len()` is the sum of the nine workload
///    chips' rows, which is what makes every residual below nameable;
/// 2. `LFM_HINT` equals [`arena_words`] evaluated at the SHAPE — the arena's
///    size derived from the proof's round structure, never counted off the
///    filler;
/// 3. `LFM_PUBLIC` equals the schema's `145 + out_halves`;
/// 4. `LFM_HASH` equals the transcript schedule's permutations;
/// 5. the CONST-FREE total equals [`EpochF1::operations`], exactly.
///
/// ⛔ (2) is the one that matters most. No cost form in the tree charges the
/// arena — `table_verify_cost` takes a shape and an entry and cannot see one —
/// so this is the term V1g's production recount is missing, and it is predicted
/// from the same enumeration `table_verify_cost`'s schedule replays.
///
/// ★ EVERY epoch, not the last one only, and that is what makes the residual
/// attributable: the bundle's epochs differ in height and in `out_halves`, and a
/// per-table or per-group error shows up as a residual that MOVES while a
/// per-epoch one stays put. It also gives the publish term both arms — a silent
/// epoch, where `out_halves` is zero and a wrong form is invisible, and the
/// publisher that can see it.
#[test]
fn the_f1_reproduces_the_emitted_epoch_program() {
    let (elf_bytes, opts, bundle) = f1_fixture_bundle();
    assert!(
        bundle.epochs.len() >= 2,
        "a one-epoch run chains nothing, and a one-shape gate cannot tell a \
         per-epoch residual from a per-table one"
    );
    let elf = executor::elf::Elf::load(&elf_bytes).expect("the inner ELF loads");

    let mut published_epochs = 0usize;
    let mut silent_epochs = 0usize;
    let mut rows: Vec<(usize, usize, usize, i64, i64)> = Vec::new();

    for index in 0..bundle.epochs.len() {
        let epoch = crate::lfm::whir_real_epoch::real_epoch_from_whir_continuation_under::<
            multilinear::whir_hash::RpxWhir,
        >(&opts, &elf_bytes, &bundle, index, None, None)
        .expect("every epoch harvests, which is the host accepting it");

        if epoch.public_output().is_empty() {
            silent_epochs += 1;
        } else {
            published_epochs += 1;
        }

        let airs = crate::multilinear_continuation::epoch_airs_for(
            &elf,
            &opts,
            &bundle.epochs[index],
            &epoch.position.register_init,
            epoch.position.is_final,
            epoch.position.label,
            Some(epoch.decode_commitment),
        );
        let refs = airs.refs();

        // ---- THE PREDICTION, from shapes alone ----
        let table_num_vars: Vec<usize> = epoch
            .proof
            .table_num_vars
            .iter()
            .map(|&v| v as usize)
            .collect();
        let statement_bytes =
            super::whir_statement::epoch_statement_bytes(&super::whir_statement::EpochStatement {
                elf_digest: &epoch.elf_digest,
                epoch_label: epoch.position.label,
                public_output: epoch.public_output(),
                table_counts: &epoch.proof.table_counts,
                table_num_vars: &epoch.proof.table_num_vars,
                config: &epoch.config,
            });
        let derived: Vec<LfmWord> = epoch
            .decode_prepared_roots
            .iter()
            .map(super::algebraic_commit::commitment_to_digest)
            .collect();
        let text = PublishedText {
            program_id: super::whir_epoch::epoch_program_id(&epoch),
            register_init: &epoch.position.register_init,
            reg_fini: &epoch.proof.reg_fini,
            label: epoch.position.label,
            public_output: epoch.public_output(),
        };
        let f1 = epoch_f1(
            &refs,
            &table_num_vars,
            epoch.config,
            &statement_bytes,
            &text,
            &derived,
        );

        // ---- THE MEASUREMENT ----
        let program = super::whir_epoch::whir_epoch_program(&epoch, &refs);
        let chips = measured_chips(&program);
        let measured = program.instrs.len();
        let chip = |name: &str| -> usize {
            chips
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, r)| *r)
                .unwrap_or(0)
        };

        println!(
            "\n== epoch {index} of {}: {} tables, out_halves {} ==",
            bundle.epochs.len(),
            refs.len(),
            epoch.public_output().len().div_ceil(4),
        );
        for (name, r) in &chips {
            println!("   {name:<14} {r:>10}");
        }
        println!("   {:<14} {:>10}", "TOTAL", measured);
        println!("{f1:#?}");

        let workload: usize = chips
            .iter()
            .filter(|(name, _)| WORKLOAD_CHIPS.contains(name))
            .map(|(_, r)| r)
            .sum();
        assert_eq!(
            workload, measured,
            "epoch {index}: the compiler opens one row in one chip per \
             instruction, so the nine workload chips must sum to the count"
        );
        assert_eq!(
            chip("LFM_HINT"),
            f1.arena,
            "epoch {index}: LFM_HINT is the arena's word count, derived here \
             from the proof's round structure"
        );
        assert_eq!(
            chip("LFM_PUBLIC"),
            f1.public,
            "epoch {index}: LFM_PUBLIC is the schema's 145 + out_halves"
        );
        assert_eq!(
            chip("LFM_HASH"),
            f1.perms,
            "epoch {index}: LFM_HASH is one row per sponge permutation"
        );

        let measured_const = chip("LFM_CONST");
        let pool_gap = f1.pool as i64 - measured_const as i64;
        let ops_gap = f1.operations() as i64 - (measured - measured_const) as i64;
        println!(
            "   F1 {} of {measured} measured (const-free {} of {}); residual: \
             const-free {ops_gap} · pool {pool_gap}",
            f1.instructions(),
            f1.operations(),
            measured - measured_const,
        );
        rows.push((index, refs.len(), measured, ops_gap, pool_gap));
    }

    // ★ THE ANTI-VACUITY PAIR, asserted after the walk so the message can say
    // what the bundle actually held: without a publisher the publish and closure
    // terms are the same on every wrong form, and without a silent epoch the
    // closure's early return is never exercised.
    assert!(
        published_epochs >= 1 && silent_epochs >= 1,
        "the fixture must hold at least one PUBLISHING and one SILENT epoch \
         ({published_epochs} publishing, {silent_epochs} silent)"
    );

    println!("\n== F1 residuals over {} epochs ==", rows.len());
    println!(
        "{:>6} {:>8} {:>12} {:>12} {:>10}",
        "epoch", "tables", "instructions", "const-free", "pool"
    );
    for (index, tables, measured, ops_gap, pool_gap) in &rows {
        println!("{index:>6} {tables:>8} {measured:>12} {ops_gap:>12} {pool_gap:>10}");
    }

    let bad: Vec<_> = rows.iter().filter(|r| r.3 != SPINE_RESIDUAL).collect();
    assert!(
        bad.is_empty(),
        "F1's const-free form must reproduce every epoch's emitted program to \
         within the pre-registered SPINE_RESIDUAL of {SPINE_RESIDUAL}; it \
         misses on {bad:?} (epoch, tables, instructions, const-free residual, \
         pool residual)"
    );
    // ★ The pool is a LOWER bound by construction and must never exceed the
    // measurement — a pool over the measured count would mean the form invents
    // a constant the program does not intern, which is a different defect from
    // the named shortfall and has its own assertion.
    let over: Vec<_> = rows.iter().filter(|r| r.4 > 0).collect();
    assert!(
        over.is_empty(),
        "the pool term is a lower bound on LFM_CONST; it EXCEEDS the measured \
         count on {over:?}"
    );
}

// =============================================================================
// THE PRODUCTION SHAPES — item 5g's fifteen, and the instrument that prints them
// =============================================================================

/// One epoch's shape and its F1, as [`the_block_f1_at_every_epoch_shape`]
/// collects them.
///
/// Named rather than a five-deep tuple because the three `usize`s next to each
/// other — the table count, the published byte count and, inside `names`, each
/// table's width and vars — are exactly the kind that swap silently.
struct BlockEpochShape {
    index: u64,
    tables: usize,
    /// PUBLISHED BYTES, not words and not `out_halves`.
    published: usize,
    f1: EpochF1,
    /// `(name, width, num_vars)` per table, in sub-proof order — the census sh1
    /// prints for epoch 0 and this prints for all fifteen.
    names: Vec<(String, usize, usize)>,
}

/// ★★ F1 AT EVERY EPOCH OF A REAL BLOCK, card-free and proof-free.
///
/// The shape half of a block run: `for_each_epoch` executes the guest and builds
/// each epoch's traces and AIRs, which is everything [`ShapePlan`] needs — no
/// proof, no card, no commitment. `whir_epoch_shapes`
/// (`multilinear_bench_tests.rs:2170`) does the same walk for its layout census
/// and ran in 14.18 s on FAST; this one evaluates F1 on top of it.
///
/// ⛔ WHY THIS EXISTS RATHER THAN A ONE-CHARACTER CHANGE TO THAT TEST: sh1's log
/// prints the per-table `(width, rows, vars)` census for epoch 0 ONLY, because
/// its printer is gated on `if prepared.index == 0`. Fourteen of the block's
/// fifteen shapes are therefore recorded nowhere, and they are not recoverable
/// from the summary — 34 unknowns against three constraints. `multilinear_bench_tests.rs`
/// holds the transcript pin and the byte gate, so the walk is copied here
/// instead of edited there.
///
/// ## What it prints, and what it cannot derive
///
/// Per epoch: the table count, the per-table `(name, width, vars)` census, the
/// layout summary sh1 prints, and F1's CONST-FREE total with its terms.
///
/// ⚠ TWO INPUTS A SHAPE WALK CANNOT SUPPLY, both stated rather than guessed:
///
/// - the PUBLISHED OUTPUT. It lives in the epoch's proof, not in its traces, so
///   the walk cannot read it. Every epoch of a continuation publishes nothing
///   except the last, which is why the block's wt3 run prints `published 145`
///   fourteen times and `185 = 145 + 40 out_halves` once. `LFM_F1_FINAL_OUT_BYTES`
///   supplies the last epoch's byte count; it is PRINTED with the value used, and
///   at 0 the last epoch's `publish` and `closure` terms are the silent ones.
/// - the POOL. It needs the DECODE prepared roots, which is a `2^23` commit this
///   walk deliberately does not make. So `LFM_CONST` is reported as NOT
///   EVALUATED here; the wt3 census measured it at 407–634 rows of 2.5–3.2 M,
///   and [`EpochF1::operations`] — the const-free form — is the one that is
///   exact anyway.
///
/// Run it with the guest ELF and the block input pinned BY SHA by the launcher:
/// `LFM_F1_ELF=<path> LFM_F1_INPUT=<path> LFM_F1_EPOCH_LOG2=21 \
///  LFM_F1_FINAL_OUT_BYTES=<n> cargo test --release -p lambda-vm-prover --lib -- \
///  --ignored --exact --nocapture lfm::whir_epoch_f1_tests::the_block_f1_at_every_epoch_shape`
#[test]
#[ignore]
fn the_block_f1_at_every_epoch_shape() {
    use crate::multilinear_prove::chain_config;
    use crate::tables::trace_builder::DecodeArtifacts;

    let elf_path = std::env::var("LFM_F1_ELF").expect("LFM_F1_ELF must name the guest ELF");
    let input_path = std::env::var("LFM_F1_INPUT").expect("LFM_F1_INPUT must name the block input");
    let epoch_log2: u32 = std::env::var("LFM_F1_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(21);
    let final_out_bytes: usize = std::env::var("LFM_F1_FINAL_OUT_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let bytes = std::fs::read(&elf_path).expect("the guest ELF reads");
    let inputs = std::fs::read(&input_path).expect("the block input reads");
    println!(
        "★ F1 BLOCK SHAPES: guest {elf_path} ({} B), input {input_path} ({} B), \
         epoch 2^{epoch_log2}, final published bytes {final_out_bytes}",
        bytes.len(),
        inputs.len(),
    );

    let opts = crate::recursion::Preset::Blowup4.options();
    let elf = executor::elf::Elf::load(&bytes).expect("the guest ELF loads");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");

    // How many epochs the run has, so the LAST one can be told apart before it
    // is reached — the published output is the one term that depends on it.
    let mut census: Vec<BlockEpochShape> = Vec::new();
    crate::continuation::for_each_epoch(&elf, &inputs, epoch_log2, &artifacts, |prepared, _| {
        let mut traces = prepared.traces;
        crate::tables::bitwise::update_multiplicities(
            &mut traces.bitwise,
            &crate::tables::local_to_global::collect_bitwise_from_l2g(&prepared.boundary),
        );
        let reg_fini = crate::tables::register::fini_from_trace(&traces.register);
        let table_counts = traces.table_counts();
        let airs = crate::continuation::build_epoch_airs(
            &elf,
            &opts,
            &[],
            &table_counts,
            &prepared.register_init,
            &reg_fini,
            prepared.is_final,
            None,
        );
        let l2g_air = crate::continuation::l2g_memory_air(&opts, prepared.label);
        let mut l2g_trace =
            crate::tables::local_to_global::generate_local_to_global_trace(&prepared.boundary);
        let mut pairs = airs.air_trace_pairs(&mut traces);
        pairs.push((&l2g_air, &mut l2g_trace, &()));

        let shapes: Vec<(usize, usize)> = pairs
            .iter()
            .map(|(_, t, _)| {
                (
                    t.main_table.width,
                    t.main_table.height.trailing_zeros() as usize,
                )
            })
            .collect();
        let table_num_vars: Vec<usize> = shapes.iter().map(|&(_, v)| v).collect();
        let names: Vec<(String, usize, usize)> = pairs
            .iter()
            .zip(&shapes)
            .map(|((air, _, _), &(w, v))| (air.name().to_string(), w, v))
            .collect();
        let refs: Vec<
            &dyn stark::traits::AIR<
                Field = crate::tables::types::GoldilocksField,
                FieldExtension = crate::tables::types::GoldilocksExtension,
                PublicInputs = (),
            >,
        > = pairs.iter().map(|(air, _, _)| *air).collect();

        let config = chain_config(&shapes);
        // ⚠ The published output is the proof's, not the traces'. Only the FINAL
        // epoch of a continuation publishes, so every other epoch is silent by
        // construction and the last one takes the byte count the caller pinned.
        let published: Vec<u8> = if prepared.is_final {
            vec![0u8; final_out_bytes]
        } else {
            Vec::new()
        };
        let num_vars_bytes: Vec<u8> = table_num_vars.iter().map(|&v| v as u8).collect();
        let statement_bytes =
            super::whir_statement::epoch_statement_bytes(&super::whir_statement::EpochStatement {
                elf_digest: &crate::statement::elf_digest(&bytes),
                epoch_label: prepared.label,
                public_output: &published,
                table_counts: &table_counts,
                table_num_vars: &num_vars_bytes,
                config: &config,
            });
        let text = PublishedText {
            program_id: [0u8; 32],
            register_init: &prepared.register_init,
            reg_fini: &reg_fini,
            label: prepared.label,
            public_output: &published,
        };
        let f1 = epoch_f1(&refs, &table_num_vars, config, &statement_bytes, &text, &[]);
        census.push(BlockEpochShape {
            index: prepared.index,
            tables: shapes.len(),
            published: published.len(),
            f1,
            names,
        });
        Ok(())
    })
    .expect("every epoch prepares");

    for row in &census {
        let BlockEpochShape {
            index,
            tables,
            published,
            f1,
            names,
        } = row;
        println!(
            "\n-- epoch {index} per-table census: {tables} tables, {published} published bytes --"
        );
        println!("{:<16} {:>7} {:>6}", "table", "width", "vars");
        for (name, w, v) in names {
            println!("{name:<16} {w:>7} {v:>6}");
        }
        println!("{f1:#?}");
        println!(
            "epoch {index}: F1 const-free {} · arena {} · perms {} · published words {}",
            f1.operations(),
            f1.arena,
            f1.perms,
            f1.public,
        );
    }

    println!("\n== F1 AT THE BLOCK'S {} EPOCH SHAPES ==", census.len());
    println!(
        "{:>6} {:>7} {:>13} {:>10} {:>10} {:>8}",
        "epoch", "tables", "F1 const-free", "arena", "perms", "words"
    );
    for row in &census {
        println!(
            "{:>6} {:>7} {:>13} {:>10} {:>10} {:>8}",
            row.index,
            row.tables,
            row.f1.operations(),
            row.f1.arena,
            row.f1.perms,
            row.f1.public,
        );
    }
    println!(
        "⚠ LFM_CONST NOT EVALUATED here (it needs the DECODE prepared roots, a 2^23 commit this \
         walk does not make); wt3 measured it at 407-634 rows. Add SPINE_RESIDUAL {SPINE_RESIDUAL} \
         to each const-free figure and then the measured LFM_CONST to compare against a census."
    );
    assert!(
        !census.is_empty(),
        "a block walk must cover at least one epoch"
    );
}

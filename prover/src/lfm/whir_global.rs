//! The whole CROSS-EPOCH verify, assembled: `verify_global_bookends`' body as
//! one machine program.
//!
//! [`super::whir_epoch`] is the level-0 analogue and this reads the same way on
//! purpose — the legs are the same legs, threaded in the same order, because
//! both programs are `multilinear_table::multi_verify` and the order is the one
//! part of a Fiat-Shamir verifier no per-leg gate can see. What is written here
//! is only what DIFFERS, and each difference is real work rather than a
//! parameter.
//!
//! # Four ways the cross-epoch proof is not an epoch
//!
//! 1. **Sixteen groups, not two.** `global_groups(num_epochs, num_pages)` is
//!    every bookend alone — so its root can be compared against the epoch that
//!    committed it, which IS the cross-epoch binding — then the pages together.
//!    The split is taken from the AIR SET's own two families
//!    (`WhirGlobalAirs::groups`) and never
//!    respelled from a table count.
//! 2. **No DECODE table and no prepared opening**, so the roots block carries
//!    ZERO derived roots and there is no sixth step. Nothing is interned here
//!    that a prover could have chosen; the cross-epoch program holds no pinned
//!    commitment at all, which is one owed pin FEWER than an epoch's, not one
//!    more.
//! 3. **The bus target is a LITERAL ZERO.** `multi_verify` is called with
//!    `FieldElement::<E>::zero()`: the cross-epoch bus has no counterparty in
//!    the statement, so there is no expected term, no published bytes and no
//!    commit index — see [`emit_global_closure`].
//! 4. **The statement is `absorb_global`**, and the published set is
//!    [`super::block_root::GlobalLayout`], not a `SchemaLayout`.
//!
//! # ⛔ THE GENESIS BINDING, AND WHERE IT ACTUALLY LIVES
//!
//! A page table's preprocessed columns are OFFSET and — on any page that is not
//! a private-input page — INIT, that page's genesis bytes. They are what ties
//! the cross-epoch memory argument to the ELF, and the tie is made by
//! `global_airs_for` REBUILDING the page configs from the ELF: there is no root
//! and no opening on this path, only the columns and `check_preprocessed`.
//!
//! ⚠ The thirty-five per-page commitments `recursion::precomputed_commitments`
//! builds are a DIFFERENT object with a confusable name. They are univariate
//! Merkle roots over each page's LDE codeword, they are what the attestation's
//! `program_id` folds, and the multilinear path never compares one of them.
//! Nothing in this program interns one, and nothing should: a root no verifier
//! on this path reads would attest to nothing.
//!
//! So the machine's obligation is the columns themselves, at each table's own
//! reduced point — OFFSET by [`super::preprocessed::emit_offset_ramp`]'s closed
//! form and INIT by [`super::preprocessed::emit_sparse_mle_at`]'s support-only
//! sum. That module's header carries the two reasons a prepared opening is not
//! available here; this one carries the consequence, which is that the leg's
//! cost is the genesis byte count and the cap is a refusal.

use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;
use multilinear::whir_chain::ChainConfig;
use stark::multilinear_logup::InteractionShape;
use stark::multilinear_table::TableLayout;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::{Cell, Ext, LfmBuilder};
use super::compiler::LfmProgram;
use super::whir_chain::ChainShape;
use super::whir_epoch::{
    GroupWires, PreprocessedPlan, PreprocessedRoute, TableWires, closure_rows, emit_group_walk,
    emit_roots_block, emit_table_walk, hint_group_chains, hint_table_wires, push_table_words,
};
use super::whir_real_global::WhirRealGlobal;
use super::whir_stacked::StackedPolyWires;
use super::whir_table::{TableProofWires, TableShape};
use super::whir_transcript::WhirTranscript;
use super::word::LfmWord;

/// How many polynomials ONE epoch's bookend group stacks into here.
///
/// One, and [`super::block_root::GlobalLayout`] is written for exactly one
/// root's lanes per epoch. A MEASURED property rather than a structural
/// certainty, so [`whir_global_program`] asserts it instead of assuming it.
///
/// ⚠ Deliberately NOT spelled `super::whir_epoch::L2G_GROUP_POLYS`. That one is
/// a fact about the bookend group inside an EPOCH's commitment; this is a fact
/// about the bookend's own singleton group in the CROSS-EPOCH commitment. They
/// are one today and they are different facts, and a single constant for the two
/// would make a change to either read as a change to both.
const BOOKEND_GROUP_POLYS: usize = 1;

/// Rows the published set costs beyond its publishes: ONE `Unpack` per epoch.
///
/// The schema wants one published WORD per lane of each bookend's root, and the
/// emitter holds every root as the single four-lane word
/// `algebraic_commit::commitment_to_digest` packs it into, so each
/// epoch's four lanes are one unpack away.
const UNPACK_ROWS_PER_EPOCH: usize = 1;

// =============================================================================
// The route table
// =============================================================================

/// ⛔ THE CROSS-EPOCH ROUTE TABLE, AND WHY IT CANNOT BE THE EPOCH'S.
///
/// `whir_epoch::EpochPlan` selects a preprocessed route by
/// `air.name()`, against four names. That rule is not merely wrong here, it is
/// **inexpressible**: `AirWithBuses::new` leaves its name `None`, neither
/// `l2g_global_air` nor `global_memory_air` calls `with_name`, and
/// `AIR::name()`'s default is `"unknown"` — so every table in a cross-epoch set
/// answers to one name and a name-keyed table would route all of them down one
/// arm.
///
/// So the key is the two things the set actually distinguishes:
///
/// 1. **the FAMILY**, from `WhirGlobalAirs`'
///    own split — the first `num_epochs` tables are bookends and carry no
///    preprocessed columns at all, the rest are pages;
/// 2. **whether a page is a PRIVATE-INPUT page**, from that page's own
///    `PageConfig::is_private_input` — the very flag
///    `global_memory_air` branched on when it decided which columns to carry.
///
/// ⛔ AND NOT FROM THE COLUMN COUNT, which is the tempting third option and is a
/// guard written on the answer. A page whose INIT column vanished upstream would
/// present one column, be routed as private, and have its genesis checked by
/// nothing — the check would pass because the quantity it read could not move
/// under the failure it exists to catch. The count is the CROSS-CHECK instead:
/// [`GlobalPlan::build`] refuses any table whose family and config do not agree
/// with the number of columns its AIR presents, and the refusal names both.
///
/// # ⚠ WHICH FIXTURE GATES WHICH ARM, said here so no reader infers coverage
///
/// The three arms are NOT gated by one run, and one of them was not gateable at
/// all until a second fixture was found:
///
/// | arm | what gates it | what that fixture cannot show |
/// |---|---|---|
/// | [`Self::Bookend`] | both fixtures; every cross-epoch proof has them | — |
/// | [`Self::PrivatePage`] | `test_private_input_xpage` | its ONLY page is private, so it never reaches the genesis arm |
/// | [`Self::GenesisPage`] | `data_page_touch` | its genesis column is a handful of `.data` bytes, not a block's |
///
/// `test_private_input_xpage`'s cross-epoch proof is three bookends and ONE
/// page, and that page is a private-input page — so a suite built on it alone
/// would gate two arms of three and read as if it gated all of them.
/// `data_page_touch` loads, increments and stores a static `.dword`, so its
/// touched page is genuinely ELF-backed and its INIT column is NONZERO; that is
/// what makes the genesis arm reachable at fixture scale at all.
///
/// ⛔ NEITHER FIXTURE SHOWS THE MIX. A block's cross-epoch proof carries both
/// kinds of page at once and a sixteen-group split; at fixture scale the page
/// group is a singleton and shape-identical to a bookend's. The route table's
/// behaviour ON A MIXED SET is therefore established by the block instrument,
/// not by these two, and saying so is the difference between a stated gap and a
/// false guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalRoute {
    /// A local-to-global bookend: no preprocessed columns.
    Bookend,
    /// A private-input page: OFFSET alone.
    PrivatePage,
    /// Any other page: OFFSET and INIT.
    GenesisPage,
}

impl GlobalRoute {
    /// How many preprocessed columns a table on this route must present.
    ///
    /// ★ THE CONSTANTS ARE THE PRODUCTION ONES, not literals: `global_memory`'s
    /// own column counts, so a column added to a page upstream moves this and
    /// the AIR together instead of leaving them to disagree.
    pub fn columns(self) -> usize {
        match self {
            Self::Bookend => 0,
            Self::PrivatePage => crate::tables::page::NUM_PREPROCESSED_COLS_PRIVATE,
            Self::GenesisPage => crate::tables::global_memory::NUM_PREPROCESSED_COLS,
        }
    }

    /// The route each table takes, in the AIR set's own order.
    ///
    /// `page_is_private` is one flag per PAGE table, in the page family's order
    /// — the canonical order `global_memory_configs` builds in, which is the
    /// order the AIRs are in.
    pub fn table_routes(num_epochs: usize, page_is_private: &[bool]) -> Vec<Self> {
        let mut routes = vec![Self::Bookend; num_epochs];
        routes.extend(page_is_private.iter().map(|&private| {
            if private {
                Self::PrivatePage
            } else {
                Self::GenesisPage
            }
        }));
        routes
    }
}

/// One page table's identity, from the config the AIR was built from.
///
/// ⚠ `base` is the CANONICAL list's, not the wire list's: `global_memory_configs`
/// canonicalises, and the AIRs are in that order. The wire list is what the
/// STATEMENT absorbs and they are different jobs — mixing them up is how a
/// bundle whose page list arrived out of order would derive one `z` and be
/// argued at another.
#[derive(Debug, Clone, Copy)]
pub struct GlobalPage {
    /// The page's base address, canonical order.
    pub base: u64,
    /// Whether it is a private-input page — the one bit that picks the route.
    pub is_private: bool,
    /// How many genesis bytes the ELF loads into it. Zero for a zero-init page
    /// (stack, heap, BSS) and for a private one, whose genesis the verifier
    /// never sees.
    pub init_len: usize,
}

// =============================================================================
// The plan
// =============================================================================

/// Everything the cross-epoch program is emitted against, derived ONCE.
///
/// ★ ONE DERIVATION, TWO FUNCTIONS, for `whir_epoch::EpochPlan`'s
/// reason: [`whir_global_program`] and [`whir_global_arena`] are only correct
/// relative to each other, so both build this and walk it.
///
/// ⚠ The `TableLayout`s are rebuilt here rather than borrowed, because
/// `layout_of` is private to `multilinear_continuation`. What catches a drift is
/// not this comment: every challenge downstream of a layout is derived from it,
/// so a layout that differed from the host's would give the machine a different
/// `z` and the honest proof would stop executing.
pub struct GlobalPlan<'a> {
    config: ChainConfig,
    /// The AIR set's own split: `num_epochs` singletons, then the pages.
    sizes: Vec<usize>,
    layouts: Vec<TableLayout<'a, GoldilocksField, GoldilocksExtension>>,
    buses: Vec<Vec<InteractionShape<GoldilocksExtension>>>,
    /// Each table's preprocessed columns, owned; empty for a bookend.
    preprocessed: Vec<Vec<Vec<FE>>>,
    /// Each table's route, decided before anything is emitted.
    route_of: Vec<GlobalRoute>,
    /// Per PAGE table, in the AIR set's own order: the page's base and whether
    /// it is a private-input page.
    ///
    /// ⛔ THE CONFIG'S OWN FLAG, NOT A COUNT OF COLUMNS. The emitter has to
    /// know which pages carry INIT, and the tempting source is how many
    /// preprocessed columns each AIR presents. That is a guard written on the
    /// answer: a page whose INIT column vanished upstream would present one
    /// column, be routed as private, and have its genesis checked by nothing —
    /// the reading could not move under the failure it exists to catch. So this
    /// is `PageConfig::is_private_input` of the very configs the AIR set was
    /// built from, and the column count becomes the CROSS-CHECK instead.
    pages: Vec<GlobalPage>,
    group_layouts: Vec<StackedLayout>,
    group_domains: Vec<Domain<GoldilocksField>>,
}

impl<'a> GlobalPlan<'a> {
    /// Builds the plan, and REFUSES before emitting anything.
    ///
    /// Three refusals, each an emit-time assertion in the
    /// `epoch_verify.rs:171-179` idiom — a set of another shape has no way to be
    /// supplied to a program that emits a fixed straight line:
    ///
    /// 1. one AIR per table the proof states a height for;
    /// 2. one route per table, and a bookend carries no preprocessed columns;
    /// 3. ★ **the column count against the route's own**, which is the check the
    ///    route table exists to make possible. A table whose family and config
    ///    say one thing and whose AIR presents another is a layout nobody meant
    ///    to build, and it fails HERE rather than being folded quietly or
    ///    skipped quietly.
    pub fn build(
        global: &WhirRealGlobal,
        airs: &'a [&'a dyn stark::traits::AIR<
            Field = GoldilocksField,
            FieldExtension = GoldilocksExtension,
            PublicInputs = (),
        >],
        elf_bytes: &[u8],
    ) -> Self {
        // ⛔ THE ELF IS NAMED, AND IT IS REFUSED IF IT IS NOT THE VERIFIED ONE.
        // The genesis binding IS the ELF, so the emitter takes it explicitly
        // rather than trusting a flag somebody computed elsewhere — and the
        // digest the harvest recorded is what says this is the same program the
        // cross-epoch proof was accepted against. An emitter handed another
        // ELF would intern another page's genesis bytes and the refusal would
        // land hundreds of thousands of rows later, at execution, with nothing
        // naming the cause.
        assert_eq!(
            crate::statement::elf_digest(elf_bytes),
            global.elf_digest,
            "the cross-epoch program must be emitted against the ELF its proof was \
             verified against"
        );
        let shapes = global.shapes.clone();
        assert_eq!(
            airs.len(),
            shapes.len(),
            "one AIR per table the cross-epoch proof states a height for"
        );
        assert_eq!(
            airs.len(),
            global.proof.proof.tables.len(),
            "one AIR per table the cross-epoch proof argues"
        );
        let config = global.config;
        let sizes = global.sizes.clone();

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

        // ⚠ ONE CALL OF `global_memory_configs`, with the arguments
        // `global_airs_for` itself passed — so the configs cannot describe a
        // different page set from the AIRs. A second SPELLING of the rule would
        // be the drift this avoids; a second CALL of one function on one set of
        // arguments cannot disagree with itself.
        let elf = executor::elf::Elf::load(elf_bytes).expect("the inner ELF must load");
        let pages: Vec<GlobalPage> = crate::continuation::global_memory_configs(
            &global.page_bases,
            &elf,
            global.num_private_input_pages,
        )
        .iter()
        .map(|config| GlobalPage {
            base: config.page_base,
            is_private: config.is_private_input,
            init_len: config.init_values.as_ref().map_or(0, Vec::len),
        })
        .collect();
        assert_eq!(
            pages.len() + global.num_epochs,
            airs.len(),
            "the cross-epoch layout has {} tables and {} bookends, which leaves {} pages, \
             and the ELF's page configs describe {}",
            airs.len(),
            global.num_epochs,
            airs.len().saturating_sub(global.num_epochs),
            pages.len(),
        );
        let private: Vec<bool> = pages.iter().map(|p| p.is_private).collect();
        let routes = GlobalRoute::table_routes(global.num_epochs, &private);
        assert_eq!(
            routes.len(),
            airs.len(),
            "the bookend count and the page flags must cover every table exactly once"
        );
        // ⛔ THE REFUSAL, before anything is emitted, and it names both numbers.
        for (index, (route, columns)) in routes.iter().zip(&preprocessed).enumerate() {
            assert_eq!(
                columns.len(),
                route.columns(),
                "cross-epoch table {index} is routed as {route:?}, which checks {} \
                 preprocessed columns, and its AIR presents {}; a table whose columns \
                 nothing checks must fail the build, not be skipped",
                route.columns(),
                columns.len(),
            );
        }

        let (group_layouts, group_domains) =
            crate::multilinear_prove::stacks(&shapes, &sizes, &config)
                .expect("the cross-epoch groups stack");

        Self {
            config,
            sizes,
            layouts,
            buses,
            preprocessed,
            route_of: routes,
            pages,
            group_layouts,
            group_domains,
        }
    }

    /// The emit-time shapes, borrowed from the layouts and the buses.
    pub fn table_shapes(&self) -> Vec<TableShape<'_>> {
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
    pub fn slots(&self) -> Vec<&[usize]> {
        self.layouts.iter().map(|layout| layout.slot_of()).collect()
    }

    /// The preprocessed plan per table, over views the caller owns.
    ///
    /// ⚠ `settled` is ZERO on every route: the cross-epoch proof carries no
    /// prepared opening, so there is nothing out of band to settle anything and
    /// a nonzero here would be switching a check off with nothing put in its
    /// place.
    pub fn routes<'v>(&self, views: &'v [Vec<&'v [FE]>]) -> Vec<PreprocessedPlan<'v>> {
        self.route_of
            .iter()
            .zip(views)
            .map(|(route, view)| {
                let plan = match route {
                    GlobalRoute::Bookend => PreprocessedRoute::None,
                    GlobalRoute::PrivatePage => PreprocessedRoute::Page {
                        offset: view[0],
                        init: None,
                    },
                    GlobalRoute::GenesisPage => PreprocessedRoute::Page {
                        offset: view[0],
                        init: Some(view[1]),
                    },
                };
                PreprocessedPlan {
                    settled: 0,
                    route: plan,
                }
            })
            .collect()
    }

    /// The routes, for a census or a gate to read without rebuilding them.
    pub fn table_routes(&self) -> &[GlobalRoute] {
        &self.route_of
    }

    /// The page family's bases and privacy, in the AIR set's own page order.
    pub fn pages(&self) -> &[GlobalPage] {
        &self.pages
    }

    /// Each commitment group's chain shape, in group order.
    pub fn group_shapes(&self) -> Vec<ChainShape> {
        self.group_layouts
            .iter()
            .map(|layout| ChainShape::new(&self.config, layout.n_stack()))
            .collect()
    }

    /// The group sizes — the AIR set's own split, carried through.
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// The stacked layouts the groups were committed under.
    pub fn group_layouts(&self) -> &[StackedLayout] {
        &self.group_layouts
    }

    /// The chain config the cross-epoch argument ran at.
    pub fn config(&self) -> &ChainConfig {
        &self.config
    }
}

// =============================================================================
// The closure
// =============================================================================

/// ★ THE CROSS-EPOCH CLOSURE: the bus balance against a LITERAL ZERO.
///
/// `verify_global_bookends` passes `&FieldElement::<E>::zero()` as `multi_verify`'s
/// expected value, and the reason is structural rather than incidental: the
/// cross-epoch bus's counterparty is the epochs' own bookends, all of which are
/// inside this one argument, so the sum must vanish. There is no public output
/// to fingerprint, no COMMIT bus start index, and therefore no
/// `super::whir_epoch::emit_expected` — `expected_rows(0)` is zero and this
/// emits nothing for it.
///
/// ⛔ AND THE ZERO IS AN INTERNED CONSTANT COMPARED AGAINST A WIRE, never a
/// value derived from the same sum. An `assert_eq_ext(total, total)` would be a
/// check whose two operands cannot differ at this call site — documentation, not
/// a check — which is the shape the campaign has already been bitten by.
///
/// Returns the balance, because the gate that could only see the refusal would
/// be checking "it did not execute", and a leg that refuses everything passes
/// that.
pub fn emit_global_closure(b: &mut LfmBuilder, outputs: &[(Ext, Ext)]) -> Ext {
    assert!(
        !outputs.is_empty(),
        "a cross-epoch balance is over at least one table"
    );
    let mut balance: Option<Ext> = None;
    for (p, q) in outputs {
        // `contribution` is `p/q` per table and is `None` when the denominator
        // vanished; here it is a `Div`, and a division by zero has no satisfying
        // assignment — the host's `None` and the machine's refusal are the same
        // event rather than two behaviours that have to agree.
        let share = b.ediv(*p, *q);
        balance = Some(match balance {
            None => share,
            Some(running) => b.eadd(running, share),
        });
    }
    let balance = balance.expect("at least one table");
    let zero = b.ext_const(&FEE::zero());
    b.assert_eq_ext(balance, zero);
    balance
}

/// What [`emit_global_closure`] costs — the epoch's own form at a published
/// length of zero.
///
/// ★ THE FORM IS SHARED AND THE EMITTER IS NOT, deliberately. The arithmetic
/// (one `Div` a table, the running `Add`, the assert) is the same arithmetic;
/// the expected term is what differs, and `expected_rows(0)` is where that
/// difference is already written down. A second form here would be a place for
/// the two to disagree about a thing they agree on.
pub fn global_closure_rows(num_tables: usize) -> usize {
    closure_rows(num_tables, 0)
}

// =============================================================================
// The published set
// =============================================================================

/// ★★ THE CROSS-EPOCH WRAP'S PUBLISHED SET, emitted —
/// [`super::block_root::GlobalLayout`], word for word.
///
/// `z`, `alpha`, then every epoch's bookend root as `lanes_per_root()` words, in
/// EPOCH order. No attestation id, no register run, no label pair, no bus tail:
/// the cross-epoch proof is about MEMORY, it carries no register run at all, and
/// the register chain and the labels are `per_table_aggregator::emit_chain_bindings`'
/// job BETWEEN SIBLINGS. Duplicating either here would be a second derivation of
/// one fact.
///
/// ⛔ THE ROOTS ARE THE CARRIED WIRES, not constants. Each bookend is committed
/// ALONE, so its root is its own singleton group's, and publishing the very cell
/// the roots block absorbed is what makes this the epoch's root: a prover who
/// published something else would have absorbed something else and derived a
/// different `z`. The root node's `emit_l2g_compare` then compares this flat
/// list against the fold the interior carried up.
///
/// ⚠ `num_epochs` IS in this schema and that is correct — it is an INTERIOR
/// interface, not the artifact. The campaign's rule governs what the ROOT
/// publishes onward, and seeing an epoch count here is not licence to publish
/// per-epoch at the root.
///
/// Returns the layout it emitted against, so a caller pins the count with the
/// layout's own accessor instead of a second count of its own.
pub fn emit_global_publishes(
    b: &mut LfmBuilder,
    layout: &super::block_root::GlobalLayout,
    z: Ext,
    alpha: Ext,
    bookend_roots: &[Cell],
) -> usize {
    assert_eq!(
        bookend_roots.len(),
        layout.num_epochs,
        "one carried bookend root per epoch"
    );
    // ⛔ THE LANES-VERSUS-WORDS REFUSAL, the epoch emitter's own.
    // `commitment_to_digest` packs a root into ONE four-lane word
    // unconditionally, so on a byte-hash build the layout would want EIGHT
    // published lanes out of a four-lane word and every root after the first
    // would shift. Here it is a build failure.
    assert_eq!(
        layout.lanes_per_root,
        super::word::WORD_LANES,
        "this emitter holds a root as one four-lane word, and the schema wants \
         {} published lanes for it",
        layout.lanes_per_root,
    );
    b.public(z.as_cell());
    b.public(alpha.as_cell());
    for root in bookend_roots {
        for lane in b.unpack(*root) {
            b.public(lane.as_cell());
        }
    }
    layout.total()
}

/// F1 for the published set, CONST-FREE: one `Public` per published word, plus
/// the one `Unpack` each epoch's lanes cost.
///
/// The published-word count is the layout's own accessor rather than
/// `2 + epochs × lanes`, because a literal here and a layout there is precisely
/// how a node comes to read the wrong field.
pub fn global_publish_rows(layout: &super::block_root::GlobalLayout) -> usize {
    layout.total() + layout.num_epochs * UNPACK_ROWS_PER_EPOCH
}

// =============================================================================
// The arena
// =============================================================================

/// Every word the cross-epoch program hints, in the order it hints them.
///
/// ⚠ THE ORDER IS THE CONTRACT between this and [`whir_global_program`], and
/// nothing but EXECUTION catches a disagreement: a misaligned arena hands the
/// machine somebody else's field element, and the argument stops satisfying its
/// own refusals. Both walk the same [`GlobalPlan`], and the gate on the pair is
/// that the program hints exactly as many words as this writes.
///
/// ★ It ends where an epoch's has one more step: there is no prepared opening,
/// so the last group's chains are the last words. The cross-epoch proof's
/// `preprocessed` field is `None` on every proof that exists, and a program that
/// hinted one would hint past the end of the arena.
pub fn whir_global_arena(
    global: &WhirRealGlobal,
    airs: &[&dyn stark::traits::AIR<
        Field = GoldilocksField,
        FieldExtension = GoldilocksExtension,
        PublicInputs = (),
    >],
    elf_bytes: &[u8],
) -> Vec<Vec<LfmWord>> {
    let plan = GlobalPlan::build(global, airs, elf_bytes);
    let proof = &global.proof.proof;
    assert!(
        proof.preprocessed.is_none(),
        "the cross-epoch proof carries no prepared opening; one here would be an \
         object `prove_global` never built and `verify_global_bookends` never checks"
    );
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
    vec![words]
}

// =============================================================================
// The assembled cross-epoch program
// =============================================================================

/// ★★ THE CROSS-EPOCH VERIFY, ASSEMBLED — `verify_global_bookends`' body as one
/// machine program.
///
/// The order is the host's, and it is the only part of a Fiat-Shamir verifier
/// that no per-leg gate can see. Read from `multi_verify`, by line: the
/// STATEMENT (`absorb_global`), the ROOTS BLOCK (`:1205`) with ZERO derived
/// roots, the per-table walk (`:1218`), the CLOSURE (`:1245`) against a literal
/// zero, and the group walk (`:1254`). There is no sixth step.
///
/// ⚠ ONE QUALIFICATION, so nobody reads a gate as covering more than it does:
/// the closure emits NO transcript operation — it is divisions, adds and one
/// `assert_eq_ext` over wires the roots block already produced — so its position
/// between the two walks is a readability choice and not a soundness one. Every
/// other step's position IS load-bearing.
pub fn whir_global_program(
    global: &WhirRealGlobal,
    airs: &[&dyn stark::traits::AIR<
        Field = GoldilocksField,
        FieldExtension = GoldilocksExtension,
        PublicInputs = (),
    >],
    elf_bytes: &[u8],
) -> LfmProgram {
    let plan = GlobalPlan::build(global, airs, elf_bytes);
    let proof = &global.proof.proof;
    let words = whir_global_arena(global, airs, elf_bytes);
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
    //
    // ⚠ `page_bases` is the WIRE list, not the canonicalised one. `absorb_global`
    // absorbs exactly what `verify_global`'s caller handed it, so a program that
    // absorbed the sorted form would derive a different `z` for any bundle whose
    // list arrived out of order — and the canonical list is what the AIRs are
    // built from, which is a different job done in a different place.
    let mut transcript = WhirTranscript::new();
    super::whir_statement::emit_global_statement(
        &mut transcript,
        &super::whir_statement::GlobalStatement {
            elf_digest: &global.elf_digest,
            num_epochs: global.num_epochs as u64,
            num_private_input_pages: global.num_private_input_pages as u64,
            page_bases: &global.page_bases,
            table_num_vars: &global.proof.table_num_vars,
            config: &plan.config,
        },
    );

    // 2. The roots block — the carried group roots, then NO derived root, then
    //    `z`, `alpha`, `beta`. The empty slice is the whole difference from an
    //    epoch's, and it is the reason this program interns no commitment and
    //    owes no pin.
    let (z, alpha, beta) = emit_roots_block(&mut b, &mut transcript, &carried, &[]);

    // The shared alpha ladder, epoch-level here too: `emit_interaction` reads
    // `alpha_powers[i + 1]` and one ladder of the longest table's length serves
    // every table.
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

    // 4. The closure, against the literal zero.
    emit_global_closure(&mut b, &walk.outputs);

    // 5. The commitment groups — sixteen of them on a block, every bookend
    //    alone and the pages together.
    let mut chains = Vec::with_capacity(proof.columns.len());
    for (group, opening) in proof.columns.iter().enumerate() {
        let shape = ChainShape::new(&plan.config, plan.group_layouts[group].n_stack());
        chains.push(hint_group_chains(&mut b, arena, &mut at, opening, &shape));
    }
    let group_shapes = plan.group_shapes();
    // ⚠ The openings and the round wires are LOCALS, because a chain's wires
    // borrow its openings and its openings borrow its storage. Keeping the whole
    // ladder in one scope is what the epoch program does and costs nothing.
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
    // ★ WHERE EACH GROUP'S ROOTS SIT, and it is the same walk the bookend
    // publishes are read off below: the roots are FLAT, one per stacked
    // polynomial, so group `g` owns the window starting at the sum of the
    // earlier groups' `num_polys()`.
    let mut root_at: Vec<usize> = Vec::with_capacity(chains.len());
    {
        let mut polys_at = 0usize;
        for layout in &plan.group_layouts {
            root_at.push(polys_at);
            polys_at += layout.num_polys();
        }
    }
    let group_wires: Vec<Vec<StackedPolyWires<'_>>> = (0..chains.len())
        .map(|group| {
            (0..chains[group].finals.len())
                .map(|poly| StackedPolyWires {
                    rounds: &group_rounds[group][poly],
                    root: carried[root_at[group] + poly],
                    final_value: chains[group].finals[poly],
                })
                .collect()
        })
        .collect();
    let groups: Vec<GroupWires<'_>> = (0..chains.len())
        .map(|group| GroupWires {
            layout: &plan.group_layouts[group],
            polys: &group_wires[group],
            shape: &group_shapes[group],
            domain: &plan.group_domains[group],
        })
        .collect();
    emit_group_walk(&mut b, &mut transcript, &groups, &plan.sizes, &walk);

    assert_eq!(
        at, total,
        "the program must hint exactly the words the arena writes"
    );

    // 6. The published set — `GlobalLayout`, and the roots are the CARRIED
    //    cells of the bookends' own singleton groups.
    for (epoch, layout) in plan.group_layouts[..global.num_epochs].iter().enumerate() {
        assert_eq!(
            layout.num_polys(),
            BOOKEND_GROUP_POLYS,
            "epoch {epoch}'s bookend fits in ONE stacked polynomial, so the schema's \
             per-epoch field covers one root's lanes; a bookend split into {} \
             publishes a set this layout does not describe",
            layout.num_polys(),
        );
    }
    let bookend_roots: Vec<Cell> = (0..global.num_epochs)
        .map(|epoch| carried[root_at[epoch]])
        .collect();
    let published = emit_global_publishes(&mut b, &global.published, z, alpha, &bookend_roots);

    let program = super::compiler::compile(b.finish());
    assert_eq!(
        program.public_len as usize, published,
        "the cross-epoch wrap publishes exactly the layout's words"
    );
    super::validator::validate(&program).expect("a cross-epoch program must be admissible");
    program
}

// =============================================================================
// The F1 — over the ASSEMBLED emission, which is the point of it
// =============================================================================

/// What the whole cross-epoch program costs, kept as the terms it is made of.
///
/// ⛔ WHY THIS EXISTS AT ALL, AND IT IS A CAMPAIGN-LEVEL LESSON RATHER THAN A
/// CONVENIENCE. The epoch program's F1s are evaluated against staged
/// sub-programs and per-leg forms, and none of them is compared to
/// `program.instrs.len()` of the ASSEMBLED program. That is why deleting a whole
/// preprocessed leg left seventeen tests green: every instrument measured a
/// quantity that could not move under the failure it was meant to catch. So this
/// form predicts the assembled instruction count, by KIND, and the gate compares
/// it against the compiled program — with a leg-deletion mutation required to
/// turn it red.
///
/// ⚠ TWO CONVENTIONS MEET HERE and this keeps them apart rather than adding
/// them, exactly as `whir_epoch::EpochCost` does. [`Self::operations`] is
/// CONST-FREE; the per-table forms report a PER-TABLE program's own pool, and in
/// one assembled program those pools COLLIDE, so the words are UNIONED by value
/// and the pool is charged once.
#[derive(Debug, Clone, Default)]
pub struct GlobalCost {
    /// The roots block and the shared alpha ladder, CONST-FREE.
    pub spine: usize,
    /// The tables' legs and their preprocessed routes, CONST-FREE — the sponge's
    /// own rows are inside this.
    pub tables: usize,
    /// The commitment groups' wrappers and their chains, CONST-FREE.
    pub groups: usize,
    /// The bus balance against the literal zero.
    pub closure: usize,
    /// The published set's `Unpack`s — one per epoch, and NOT its publishes.
    pub publish_ops: usize,
    /// `Instr::Public` rows: the layout's own word count.
    pub publics: usize,
    /// `LFM_HINT` rows: exactly the words the arena writes.
    pub hints: usize,
    /// Permutations, whole.
    pub perms: usize,
    /// The ONE pool, by value.
    pub constants: Vec<LfmWord>,
}

impl GlobalCost {
    /// INSTRUCTIONS excluding every `LFM_CONST`, every hint and every publish —
    /// the convention the per-leg forms are in.
    pub fn operations(&self) -> usize {
        self.spine + self.tables + self.groups + self.closure + self.publish_ops
    }

    /// Every instruction the compiled program should hold.
    ///
    /// ★ The four kinds are added ONCE each and named, so a gap lands on the
    /// kind it belongs to instead of on whichever term the arithmetic was
    /// written against.
    pub fn instructions(&self) -> usize {
        self.operations() + self.constants.len() + self.hints + self.publics
    }
}

/// [`whir_global_program`]'s cost, leg by leg, threading the sponge — the F1.
///
/// Every term is evaluated at the SHAPES the plan holds, never at a
/// reconstruction of them, and the sponge is threaded through the legs in the
/// program's own order because a leg entered at the wrong sponge state costs a
/// different number of rows.
pub fn global_cost(
    global: &WhirRealGlobal,
    airs: &[&dyn stark::traits::AIR<
        Field = GoldilocksField,
        FieldExtension = GoldilocksExtension,
        PublicInputs = (),
    >],
    elf_bytes: &[u8],
) -> GlobalCost {
    let plan = GlobalPlan::build(global, airs, elf_bytes);
    let shapes = plan.table_shapes();
    let views: Vec<Vec<&[FE]>> = plan
        .preprocessed
        .iter()
        .map(|columns| columns.iter().map(Vec::as_slice).collect())
        .collect();
    let routes = plan.routes(&views);

    let mut cost = GlobalCost::default();
    let mut pool = super::whir_bus::Cost::default();

    // The statement: no operation, only its interned run.
    let statement = super::whir_statement::statement_cost(
        &super::whir_statement::global_statement_bytes(&super::whir_statement::GlobalStatement {
            elf_digest: &global.elf_digest,
            num_epochs: global.num_epochs as u64,
            num_private_input_pages: global.num_private_input_pages as u64,
            page_bases: &global.page_bases,
            table_num_vars: &global.proof.table_num_vars,
            config: plan.config(),
        }),
    );
    for word in &statement.constants {
        pool.constant_word(*word);
    }

    // The roots block: every carried root, and ZERO derived.
    let carried = global.proof.proof.roots.len();
    let (roots_ops, schedule) = super::whir_epoch::roots_block_cost(carried, 0, statement.entry());
    cost.spine += roots_ops + schedule.rows();
    cost.perms += schedule.perms();
    for word in super::whir_epoch::roots_block_constants(&[], &schedule) {
        pool.constant_word(word);
    }

    // The shared alpha ladder.
    let ladder_len = shapes
        .iter()
        .map(|shape| super::whir_bus::alpha_powers_read(shape.bus))
        .max()
        .unwrap_or(1);
    cost.spine += super::whir_poly::challenge_powers_rows(ladder_len);
    pool.constant(FEE::one());

    // The per-table walk, with each table's preprocessed route inside it.
    let walk = super::whir_epoch::table_walk_cost(&shapes, &routes, schedule.entry());
    cost.tables += walk.operations();
    cost.perms += walk.perms;
    pool.merge(&walk.leg);

    // The closure, against the literal zero.
    cost.closure += global_closure_rows(shapes.len());
    pool.constant(FEE::zero());

    // The commitment groups, threaded from where the tables left the sponge.
    let pairs = global.shapes.clone();
    let groups = super::whir_epoch::epoch_group_costs(
        &pairs,
        plan.sizes(),
        plan.group_layouts(),
        plan.config(),
        walk.entry,
    );
    for group in &groups {
        cost.groups += group.operations();
        cost.perms += group.perms();
        for word in group.own_constants() {
            pool.constant_word(word);
        }
    }

    // The published set: its publishes and its unpacks, apart.
    cost.publics += global.published.total();
    cost.publish_ops += global.published.num_epochs * UNPACK_ROWS_PER_EPOCH;

    cost.hints += whir_global_arena(global, airs, elf_bytes)[0].len();
    cost.constants = pool.constant_values().to_vec();
    cost
}

// =============================================================================
// The genesis census — the measurement the cap is owed
// =============================================================================

/// One page's genesis cost, as the quantities the cap is set against.
#[derive(Debug, Clone, Copy)]
pub struct GenesisEntry {
    /// The page's base address, canonical order.
    pub page_base: u64,
    /// A private-input page contributes no genesis entries at all: its INIT is
    /// a committed main column the verifier never recomputes.
    pub is_private: bool,
    /// Genesis bytes the ELF loads into the page. Everything past this is read
    /// as zero, which is where the sparsity comes from.
    pub init_len: usize,
    /// Rows in the preprocessed columns — the page size.
    pub rows: usize,
    /// Surviving entries in INIT: the NONZERO genesis bytes. Not `init_len`,
    /// because a genesis byte may itself be zero.
    pub entries: usize,
    /// Rows the sparse leg emits for this page: the hoisted complements plus
    /// `num_vars` an entry.
    pub leg_rows: usize,
}

/// ★★ THE CENSUS THE CAP IS SIZED AGAINST — PROVE-FREE AND CARD-FREE.
///
/// [`super::preprocessed::MAX_SPARSE_INIT_ENTRIES`] is a placeholder, and the
/// quantity that decides whether a real block fits under it is a function of the
/// ELF and the touched page list alone. Neither needs a proof: the page list is
/// an EXECUTION fact — which cells cross an epoch boundary — and
/// `continuation::block_page_census` produces it by running the guest with every
/// prove and trace build omitted.
///
/// So this takes the two, rebuilds the page configs through the verifier's own
/// `global_memory_configs`, and counts. Nothing here touches an AIR, a proof or
/// a card.
///
/// ⚠ IT COUNTS, IT DOES NOT PRICE. The rows a page costs are `leg_rows`, and
/// they are computed here from [`super::preprocessed::sparse_mle_rows`] rather
/// than spelled again — two spellings of the row arithmetic would be two places
/// for it to drift.
pub fn genesis_census(
    elf_bytes: &[u8],
    page_bases: &[u64],
    num_private_input_pages: usize,
) -> Result<Vec<GenesisEntry>, String> {
    let elf = executor::elf::Elf::load(elf_bytes).map_err(|e| format!("the ELF must load: {e}"))?;
    let configs =
        crate::continuation::global_memory_configs(page_bases, &elf, num_private_input_pages);
    Ok(configs
        .iter()
        .map(|config| {
            let columns = crate::tables::page::preprocessed_columns(config);
            let rows = columns[0].len();
            let num_vars = rows.trailing_zeros() as usize;
            let (entries, leg_rows) = if config.is_private_input {
                // Its INIT is never recomputed and never interned, so it has no
                // genesis entries — zero here is a FACT about the route, not a
                // column that happened to be empty.
                (0usize, 0usize)
            } else {
                let init: &[FE] = &columns[1];
                (
                    super::preprocessed::sparse_entries(&[init]),
                    super::preprocessed::sparse_mle_rows(&[init], num_vars),
                )
            };
            GenesisEntry {
                page_base: config.page_base,
                is_private: config.is_private_input,
                init_len: config.init_values.as_ref().map_or(0, Vec::len),
                rows,
                entries,
                leg_rows,
            }
        })
        .collect())
}

/// The census as one line, for a box run's log.
///
/// ★ IT PRINTS THE SPLIT, THE TOTAL AND THE WORST PAGE, because the total alone
/// cannot tell "many pages with a handful of entries each" from "one dense
/// page", and those are different situations for a cap: the first fits under a
/// larger bound, the second does not fit under any bound worth having.
pub fn genesis_census_line(census: &[GenesisEntry]) -> String {
    let genesis = census.iter().filter(|e| !e.is_private).count();
    let private = census.len() - genesis;
    let total: usize = census.iter().map(|e| e.entries).sum();
    let worst = census.iter().map(|e| e.entries).max().unwrap_or(0);
    let rows: usize = census.iter().map(|e| e.leg_rows).sum();
    format!(
        "GENESIS CENSUS: {} pages = {genesis} genesis + {private} private; {total} nonzero \
         entries, worst page {worst}; {rows} leg rows; cap {} entries",
        census.len(),
        super::preprocessed::MAX_SPARSE_INIT_ENTRIES,
    )
}

/// One page's line, in the ruling's own column order.
///
/// ⚠ `init_len` AND `entries` BOTH, because they differ and the difference is
/// the point: a page can load a thousand genesis bytes of which most are zero,
/// and it is the nonzero count that the leg pays for.
pub fn genesis_entry_line(entry: &GenesisEntry) -> String {
    format!(
        "  page {:#018x}: private={} init_values={} rows={} nonzero={} leg_rows={}",
        entry.page_base,
        entry.is_private,
        entry.init_len,
        entry.rows,
        entry.entries,
        entry.leg_rows,
    )
}

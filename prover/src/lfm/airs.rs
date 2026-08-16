//! `LfmAirs` — the machine's fixed 15-chip AIR set, a sibling of `VmAirs`.
//!
//! The chip set never varies; only heights do (per program). Programs are
//! supplied preprocessed roots (resolved from `LFM_REGISTRY` at verify time),
//! so constructing the verify-side AIR set costs nothing — there is no keygen
//! in this framework. Proved and verified by the same generic
//! `multi_prove` / `multi_verify_views` machinery as the RV64 VM; **zero
//! `VmAirs` edits** — the sibling-AIR-set property, preserved deliberately.

use stark::config::Commitment;
use stark::constraints::builder::{ConstraintSet, EmptyConstraints};
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, NullBoundaryConstraintBuilder,
};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;
use stark::traits::AIR;

use crate::tables::types::{GoldilocksExtension, GoldilocksField};

use crate::tables::{bitwise, keccak_rc, keccak_rnd};

use super::blake3_chip;
use super::chips::{balu, bitdec, const_, hash, hint, keccak, lanes, public, range, select, xalu};
use super::hash::HasherKind;
use super::layout;
use super::trace::LfmTraces;

type F = GoldilocksField;
type E = GoldilocksExtension;

pub type LfmAir<CS> = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), CS>;
pub type DynLfmAir<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

/// The frozen chip order — everywhere: roots, digests, traces, proofs.
///
/// Slots 12–14 are the production keccak family, hosted unchanged. They belong
/// to the *fixed* machine, so **every** LFM proof carries them — including the
/// 2^20-row BITWISE table, which costs a few seconds of prove time even for a
/// program containing no keccak at all. That is the deliberate price of the
/// fixed-machine principle: the chip set never varies with the program, only
/// heights do, so a program stays nothing but a vector of preprocessed roots
/// plus a registry entry. Making the set program-dependent would move shape
/// negotiation onto the verify path, which this design refuses.
///
/// This is the count of chip *classes*, and the width of the roots and
/// log-heights arrays. `KECCAK_RND` (slot 12) may be instantiated more than
/// once — see [`num_lfm_airs`] — but its chunk count is program shape read
/// from the registry, not shape negotiated on the verify path, so the
/// principle above holds.
pub const NUM_LFM_CHIPS: usize = 15;
pub const LFM_CHIP_NAMES: [&str; NUM_LFM_CHIPS] = [
    "LFM_CONST",
    "LFM_BALU",
    "LFM_XALU",
    "LFM_SELECT",
    "LFM_BITDEC",
    "LFM_HASH",
    "LFM_KECCAK",
    "LFM_LANES",
    "LFM_HINT",
    "LFM_PUBLIC",
    "LFM_RANGE",
    "LFM_BLAKE3",
    "KECCAK_RND",
    "KECCAK_RC",
    "BITWISE",
];

/// Slot of `KECCAK_RND`, the one AIR in the set with **no** preprocessed
/// columns — it has no root to supply, pin, or bind into the program digest.
/// It is also the one slot that expands into several AIR instances; the
/// chunks sit contiguously at 12.., so `KECCAK_RC` and `BITWISE` follow them
/// in the AIR list while keeping chip-class indices 13 and 14 in the roots
/// and log-heights arrays.
///
/// `LFM_BLAKE3` was placed at 11 — last of the LFM chips, before the hosted
/// family — rather than appended at 14, so this constant stays the boundary
/// between "chips this machine owns" and "tables it hosts". Appending would
/// have left the count arithmetic below untouched at the cost of putting a
/// program-dependent group after two fixed tables, which is the shape every
/// index expression here is written against.
pub const KECCAK_RND_SLOT: usize = 12;

/// AIR instances (and sub-proofs) in a proof whose `KECCAK_RND` is split into
/// `keccak_rnd_chunks` instances.
pub const fn num_lfm_airs(keccak_rnd_chunks: usize) -> usize {
    NUM_LFM_CHIPS - 1 + keccak_rnd_chunks
}

/// Permutations in each `KECCAK_RND` chunk, in chunk order.
pub fn keccak_rnd_chunk_permutations(program: &super::compiler::LfmProgram) -> Vec<usize> {
    let total = program.groups.keccak.real_rows;
    let per = program.chunking.permutations_per_chunk();
    (0..program.chunking.chunk_count(total))
        .map(|i| total.saturating_sub(i * per).min(per))
        .collect()
}

/// Each `KECCAK_RND` chunk's trace height: 24 rows per permutation, padded —
/// the same `.next_power_of_two().max(4)` rule `generate_keccak_rnd_trace`
/// applies, now once per chunk.
pub fn keccak_rnd_chunk_rows(program: &super::compiler::LfmProgram) -> Vec<usize> {
    keccak_rnd_chunk_permutations(program)
        .into_iter()
        .map(|perms| {
            (perms * super::chunking::KECCAK_RND_ROWS_PER_PERMUTATION)
                .next_power_of_two()
                .max(4)
        })
        .collect()
}

/// One chip instance's trace geometry, as the cell instrument sees it.
///
/// The per-chip decomposition of [`lfm_cell_counts`] — that function sums
/// exactly these rows, so a census and a total can never disagree. `name` is not
/// unique: every `KECCAK_RND` chunk reports under the same chip name, which is
/// the point (they are the same AIR at different heights).
#[derive(Clone, Copy, Debug)]
pub struct LfmChipCells {
    pub name: &'static str,
    /// Padded trace rows — the height the prover commits.
    pub rows: u64,
    /// Value columns: the AIR's width less its preprocessed prefix.
    pub main_cols: usize,
    /// Aux (LogUp) columns, one per pair of bus interactions.
    pub aux_cols: usize,
}

impl LfmChipCells {
    pub fn main_cells(&self) -> u64 {
        self.rows * self.main_cols as u64
    }

    pub fn aux_cells(&self) -> u64 {
        self.rows * self.aux_cols as u64
    }
}

/// Per-chip trace geometry for a compiled program, in the frozen AIR order
/// (`KECCAK_RND`'s chunks expanded, so the vector has one entry per sub-proof).
///
/// Extracted from [`lfm_cell_counts`] rather than written beside it: a second
/// copy of this table is how a census would come to describe a different machine
/// than the one the totals describe.
pub fn lfm_chip_census(program: &super::compiler::LfmProgram) -> Vec<LfmChipCells> {
    lfm_chip_census_with_hasher(program, HasherKind::default())
}

/// [`lfm_chip_census`] for a program proved under `hasher`.
///
/// Only `LFM_HASH`'s width moves with the hasher; every other chip is
/// hash-independent, and the preprocessed prefix is the hasher-independent
/// instruction group, so the row counts and the roots do not move either.
pub fn lfm_chip_census_with_hasher(
    program: &super::compiler::LfmProgram,
    hasher: HasherKind,
) -> Vec<LfmChipCells> {
    let range_rows = layout::range::NUM_ROWS as u64;
    let g = &program.groups;
    // Every chip class except `KECCAK_RND`, which is counted per chunk below.
    let per_chip: [(u64, usize, usize, usize); NUM_LFM_CHIPS - 1] = [
        (
            g.const_.padded_rows as u64,
            const_::cols::NUM_COLUMNS,
            layout::const_::PREP_WIDTH,
            const_::bus_interactions().len(),
        ),
        (
            g.balu.padded_rows as u64,
            balu::cols::NUM_COLUMNS,
            layout::balu::PREP_WIDTH,
            balu::bus_interactions().len(),
        ),
        (
            g.xalu.padded_rows as u64,
            xalu::cols::NUM_COLUMNS,
            layout::xalu::PREP_WIDTH,
            xalu::bus_interactions().len(),
        ),
        (
            g.select.padded_rows as u64,
            select::cols::NUM_COLUMNS,
            layout::select::PREP_WIDTH,
            select::bus_interactions().len(),
        ),
        (
            g.bitdec.padded_rows as u64,
            bitdec::cols::NUM_COLUMNS,
            layout::bitdec::PREP_WIDTH,
            bitdec::bus_interactions().len(),
        ),
        (
            g.hash.padded_rows as u64,
            hash::num_columns(hasher),
            layout::hash::PREP_WIDTH,
            hash::bus_interactions(hasher).len(),
        ),
        (
            g.keccak.padded_rows as u64,
            keccak::cols::NUM_COLUMNS,
            layout::keccak::PREP_WIDTH,
            keccak::bus_interactions().len(),
        ),
        (
            g.lanes.padded_rows as u64,
            lanes::cols::NUM_COLUMNS,
            layout::lanes::PREP_WIDTH,
            lanes::bus_interactions().len(),
        ),
        (
            g.hint.padded_rows as u64,
            hint::cols::NUM_COLUMNS,
            layout::hint::PREP_WIDTH,
            hint::bus_interactions().len(),
        ),
        (
            g.public.padded_rows as u64,
            public::cols::NUM_COLUMNS,
            layout::public::PREP_WIDTH,
            public::bus_interactions().len(),
        ),
        (
            range_rows,
            range::cols::NUM_COLUMNS,
            layout::range::PREP_WIDTH,
            range::bus_interactions().len(),
        ),
        (
            g.blake3.padded_rows as u64,
            blake3_chip::cols::NUM_COLUMNS,
            layout::blake3::PREP_WIDTH,
            blake3_chip::bus_interactions().len(),
        ),
        // The keccak family's two fixed tables. `KECCAK_RND`'s chunks follow.
        (
            keccak_rc::NUM_ROWS as u64,
            keccak_rc::cols::NUM_COLUMNS,
            keccak_rc::NUM_PRECOMPUTED_COLS,
            keccak_rc::bus_interactions().len(),
        ),
        (
            bitwise::NUM_ROWS as u64,
            bitwise::cols::NUM_COLUMNS,
            bitwise::NUM_PRECOMPUTED_COLS,
            bitwise::bus_interactions().len(),
        ),
    ];
    // The frozen AIR order is `air_refs`': chip classes 0..=11 (`LFM_BLAKE3` is
    // the last of them), then every `KECCAK_RND` chunk, then `KECCAK_RC` and
    // `BITWISE`. `per_chip` above lists the classes with the last two at the
    // end, so the chunks are spliced in before them rather than appended.
    let rnd_interactions = keccak_rnd::bus_interactions().len();
    let mut census = Vec::with_capacity(per_chip.len() + 1);
    for (slot, (rows, num_cols, prep, interactions)) in per_chip.into_iter().enumerate() {
        if slot == KECCAK_RND_SLOT {
            for rows in keccak_rnd_chunk_rows(program) {
                census.push(LfmChipCells {
                    name: LFM_CHIP_NAMES[KECCAK_RND_SLOT],
                    rows: rows as u64,
                    main_cols: keccak_rnd::cols::NUM_COLUMNS,
                    aux_cols: rnd_interactions.div_ceil(2),
                });
            }
        }
        census.push(LfmChipCells {
            // `per_chip`'s last two entries are chip classes 13 and 14, which sit
            // at indices 12 and 13 of that array — hence the shift past the
            // `KECCAK_RND` slot rather than a plain index.
            name: LFM_CHIP_NAMES[if slot >= KECCAK_RND_SLOT {
                slot + 1
            } else {
                slot
            }],
            rows,
            main_cols: num_cols - prep,
            aux_cols: interactions.div_ceil(2),
        });
    }
    census
}

/// Trace-cell counts for a compiled program, the LFM analogue of the VM's
/// `total_field_elements` / `total_auxiliary_field_elements` (same
/// semantics: main counts base-field value cells excluding preprocessed
/// columns; aux counts extension-field elements, one per aux column per
/// row). This is the kill-risk-3 instrument: machine cells per verification
/// vs the verified proof's own cells.
pub fn lfm_cell_counts(program: &super::compiler::LfmProgram) -> (u64, u64) {
    lfm_cell_counts_with_hasher(program, HasherKind::default())
}

/// [`lfm_cell_counts`] for a program proved under `hasher` — the hash matrix's
/// instrument.
pub fn lfm_cell_counts_with_hasher(
    program: &super::compiler::LfmProgram,
    hasher: HasherKind,
) -> (u64, u64) {
    lfm_chip_census_with_hasher(program, hasher)
        .iter()
        .fold((0u64, 0u64), |(main, aux), c| {
            (main + c.main_cells(), aux + c.aux_cells())
        })
}

pub struct LfmAirs {
    const_: LfmAir<EmptyConstraints>,
    balu: LfmAir<balu::BaluConstraints>,
    xalu: LfmAir<xalu::XaluConstraints>,
    select: LfmAir<select::SelectConstraints>,
    bitdec: LfmAir<bitdec::BitDecConstraints>,
    hash: LfmAir<hash::HashConstraints>,
    keccak: LfmAir<keccak::KeccakAdapterConstraints>,
    lanes: LfmAir<EmptyConstraints>,
    hint: LfmAir<EmptyConstraints>,
    public: LfmAir<EmptyConstraints>,
    range: LfmAir<EmptyConstraints>,
    /// The BLAKE3 compression chip — a real constrained chip, unlike
    /// `LFM_KECCAK`, which is an adapter that delegates its permutation to the
    /// hosted family. Its AIR and its trace filler live in
    /// [`super::blake3_chip`] rather than in `chips.rs` for that reason: there
    /// is nothing here to adapt, only a chip to name.
    blake3: LfmAir<blake3_chip::Blake3LfmConstraints>,
    /// One instance per `KECCAK_RND` chunk. Every instance is the identical
    /// AIR — chunking changes only how many rows each one carries — so they
    /// are built in a loop rather than named individually.
    keccak_rnd: Vec<LfmAir<keccak_rnd::KeccakRndConstraints>>,
    keccak_rc: LfmAir<EmptyConstraints>,
    bitwise: LfmAir<EmptyConstraints>,
}

/// Builds an AIR with **no** preprocessed columns — `KECCAK_RND` only.
fn build_air_no_prep<CS: ConstraintSet<F, E> + 'static>(
    num_columns: usize,
    interactions: Vec<BusInteraction>,
    options: &ProofOptions,
    constraint_set: CS,
    name: &'static str,
) -> LfmAir<CS> {
    AirWithBuses::new(
        num_columns,
        AuxiliaryTraceBuildData { interactions },
        options,
        1,
        constraint_set,
    )
    .with_name(name)
}

#[allow(clippy::too_many_arguments)]
fn build_air<CS: ConstraintSet<F, E> + 'static>(
    num_columns: usize,
    interactions: Vec<BusInteraction>,
    options: &ProofOptions,
    constraint_set: CS,
    name: &'static str,
    root: Commitment,
    num_prep: usize,
) -> LfmAir<CS> {
    AirWithBuses::new(
        num_columns,
        AuxiliaryTraceBuildData { interactions },
        options,
        1,
        constraint_set,
    )
    .with_name(name)
    .with_preprocessed(root, num_prep)
}

impl LfmAirs {
    /// Builds the chip set against the supplied (registry-resolved or
    /// freshly built) instruction-column-group roots, in the frozen order,
    /// with `KECCAK_RND` instantiated `keccak_rnd_chunks` times.
    ///
    /// A zero chunk count builds no `KECCAK_RND` at all; callers on the verify
    /// path must reject that shape before getting here rather than relying on
    /// the resulting AIR-count mismatch (`verify_against` does).
    pub fn new(
        roots: &[Commitment; NUM_LFM_CHIPS],
        options: &ProofOptions,
        keccak_rnd_chunks: usize,
    ) -> Self {
        Self::new_with_hasher(roots, options, keccak_rnd_chunks, HasherKind::default())
    }

    /// [`LfmAirs::new`] with the `LFM_HASH` permutation chosen explicitly.
    ///
    /// The hasher is a construction-time property of the AIR set because the
    /// chip bakes its round constants into its constraints: the same `hasher`
    /// must reach execution and trace generation, which is what
    /// `proof::lfm_prove_with_hasher` guarantees. Nothing else in the set moves
    /// — the preprocessed prefix is the instruction column group, which no
    /// hasher changes, so the preprocessed roots and the program digest are
    /// hasher-independent.
    pub fn new_with_hasher(
        roots: &[Commitment; NUM_LFM_CHIPS],
        options: &ProofOptions,
        keccak_rnd_chunks: usize,
        hasher: HasherKind,
    ) -> Self {
        LfmAirs {
            const_: build_air(
                const_::cols::NUM_COLUMNS,
                const_::bus_interactions(),
                options,
                EmptyConstraints,
                LFM_CHIP_NAMES[0],
                roots[0],
                layout::const_::PREP_WIDTH,
            ),
            balu: build_air(
                balu::cols::NUM_COLUMNS,
                balu::bus_interactions(),
                options,
                balu::BaluConstraints,
                LFM_CHIP_NAMES[1],
                roots[1],
                layout::balu::PREP_WIDTH,
            ),
            xalu: build_air(
                xalu::cols::NUM_COLUMNS,
                xalu::bus_interactions(),
                options,
                xalu::XaluConstraints,
                LFM_CHIP_NAMES[2],
                roots[2],
                layout::xalu::PREP_WIDTH,
            ),
            select: build_air(
                select::cols::NUM_COLUMNS,
                select::bus_interactions(),
                options,
                select::SelectConstraints,
                LFM_CHIP_NAMES[3],
                roots[3],
                layout::select::PREP_WIDTH,
            ),
            bitdec: build_air(
                bitdec::cols::NUM_COLUMNS,
                bitdec::bus_interactions(),
                options,
                bitdec::BitDecConstraints,
                LFM_CHIP_NAMES[4],
                roots[4],
                layout::bitdec::PREP_WIDTH,
            ),
            hash: build_air(
                hash::num_columns(hasher),
                hash::bus_interactions(hasher),
                options,
                hash::HashConstraints { kind: hasher },
                LFM_CHIP_NAMES[5],
                roots[5],
                layout::hash::PREP_WIDTH,
            ),
            keccak: build_air(
                keccak::cols::NUM_COLUMNS,
                keccak::bus_interactions(),
                options,
                keccak::KeccakAdapterConstraints,
                LFM_CHIP_NAMES[6],
                roots[6],
                layout::keccak::PREP_WIDTH,
            ),
            lanes: build_air(
                lanes::cols::NUM_COLUMNS,
                lanes::bus_interactions(),
                options,
                EmptyConstraints,
                LFM_CHIP_NAMES[7],
                roots[7],
                layout::lanes::PREP_WIDTH,
            ),
            hint: build_air(
                hint::cols::NUM_COLUMNS,
                hint::bus_interactions(),
                options,
                EmptyConstraints,
                LFM_CHIP_NAMES[8],
                roots[8],
                layout::hint::PREP_WIDTH,
            ),
            public: build_air(
                public::cols::NUM_COLUMNS,
                public::bus_interactions(),
                options,
                EmptyConstraints,
                LFM_CHIP_NAMES[9],
                roots[9],
                layout::public::PREP_WIDTH,
            ),
            range: build_air(
                range::cols::NUM_COLUMNS,
                range::bus_interactions(),
                options,
                EmptyConstraints,
                LFM_CHIP_NAMES[10],
                roots[10],
                layout::range::PREP_WIDTH,
            ),
            blake3: build_air(
                blake3_chip::cols::NUM_COLUMNS,
                blake3_chip::bus_interactions(),
                options,
                blake3_chip::Blake3LfmConstraints,
                LFM_CHIP_NAMES[11],
                roots[11],
                layout::blake3::PREP_WIDTH,
            ),
            // KECCAK_RND has no preprocessed columns: `roots[KECCAK_RND_SLOT]`
            // is the all-zero sentinel and is never consulted. Its correctness
            // is entirely its own constraints plus bus balance, both
            // program-independent, so there is nothing for a root to pin —
            // and nothing that differs between chunks either, which is why
            // every instance is built from the same arguments.
            keccak_rnd: (0..keccak_rnd_chunks)
                .map(|_| {
                    build_air_no_prep(
                        keccak_rnd::cols::NUM_COLUMNS,
                        keccak_rnd::bus_interactions(),
                        options,
                        keccak_rnd::KeccakRndConstraints,
                        LFM_CHIP_NAMES[KECCAK_RND_SLOT],
                    )
                })
                .collect(),
            keccak_rc: build_air(
                keccak_rc::cols::NUM_COLUMNS,
                keccak_rc::bus_interactions(),
                options,
                EmptyConstraints,
                LFM_CHIP_NAMES[13],
                roots[13],
                keccak_rc::NUM_PRECOMPUTED_COLS,
            ),
            bitwise: build_air(
                bitwise::cols::NUM_COLUMNS,
                bitwise::bus_interactions(),
                options,
                EmptyConstraints,
                LFM_CHIP_NAMES[14],
                roots[14],
                bitwise::NUM_PRECOMPUTED_COLS,
            ),
        }
    }

    /// Number of `KECCAK_RND` instances this set was built with.
    pub fn keccak_rnd_chunks(&self) -> usize {
        self.keccak_rnd.len()
    }

    /// Verify-side projection, frozen order (must match `air_trace_pairs`).
    pub fn air_refs(&self) -> Vec<DynLfmAir<'_>> {
        let mut refs: Vec<DynLfmAir<'_>> = vec![
            &self.const_,
            &self.balu,
            &self.xalu,
            &self.select,
            &self.bitdec,
            &self.hash,
            &self.keccak,
            &self.lanes,
            &self.hint,
            &self.public,
            &self.range,
            &self.blake3,
        ];
        refs.extend(self.keccak_rnd.iter().map(|a| a as DynLfmAir<'_>));
        refs.push(&self.keccak_rc);
        refs.push(&self.bitwise);
        refs
    }

    /// Prove-side projection, frozen order (must match `air_refs`).
    ///
    /// `traces.keccak_rnd` must have exactly one trace per chunk; a mismatch
    /// would silently shorten the pair list under `zip`, so it is asserted.
    #[allow(clippy::type_complexity)]
    pub fn air_trace_pairs<'a>(
        &'a self,
        traces: &'a mut LfmTraces,
    ) -> Vec<(DynLfmAir<'a>, &'a mut TraceTable<F, E>, &'a ())> {
        debug_assert_eq!(
            self.keccak_rnd.len(),
            traces.keccak_rnd.len(),
            "KECCAK_RND chunk count differs between the AIR set and the traces \
             — artifacts and traces were built from different chunking policies"
        );
        let mut pairs: Vec<(DynLfmAir<'a>, &'a mut TraceTable<F, E>, &'a ())> = vec![
            (&self.const_, &mut traces.const_, &()),
            (&self.balu, &mut traces.balu, &()),
            (&self.xalu, &mut traces.xalu, &()),
            (&self.select, &mut traces.select, &()),
            (&self.bitdec, &mut traces.bitdec, &()),
            (&self.hash, &mut traces.hash, &()),
            (&self.keccak, &mut traces.keccak, &()),
            (&self.lanes, &mut traces.lanes, &()),
            (&self.hint, &mut traces.hint, &()),
            (&self.public, &mut traces.public, &()),
            (&self.range, &mut traces.range, &()),
            (&self.blake3, &mut traces.blake3, &()),
        ];
        pairs.extend(
            self.keccak_rnd
                .iter()
                .zip(traces.keccak_rnd.iter_mut())
                .map(|(air, trace)| (air as DynLfmAir<'a>, trace, &())),
        );
        pairs.push((&self.keccak_rc, &mut traces.keccak_rc, &()));
        pairs.push((&self.bitwise, &mut traces.bitwise, &()));
        pairs
    }
}

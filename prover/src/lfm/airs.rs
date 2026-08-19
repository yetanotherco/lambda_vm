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

/// The hash-family slots [`ChipSet`] gates. `BITWISE` (14) is deliberately not
/// among them: both families send to its `ByteAlu`/`AreBytes` buses, so it is
/// shared infrastructure rather than either family's chip.
pub const KECCAK_SLOT: usize = 6;
pub const BLAKE3_SLOT: usize = 11;
pub const KECCAK_RC_SLOT: usize = 13;

/// AIR instances (and sub-proofs) in a proof whose `KECCAK_RND` is split into
/// `keccak_rnd_chunks` instances.
pub const fn num_lfm_airs(keccak_rnd_chunks: usize) -> usize {
    ChipSet::FULL.num_airs(keccak_rnd_chunks)
}

/// ★ Which hash-family chip groups a program instantiates.
///
/// The machine hosts two hash families, and a program uses at most one of them.
/// Carrying the other is a whole AIR per chip — commitment, Merkle tree,
/// opening, FRI — for zero rows. On a real block that bill was measured: the
/// always-on BLAKE3 RV64 table cost the recursion wrap +31.2% of its trace
/// cells and +44% of its peak memory while carrying 0.035% of the epoch's own
/// cells (`thoughts/shared/block-compression/WRAP-GROWTH-BISECT.md`). Whichever
/// hash is not in use should not be paying for a chip group — in **both**
/// directions, which is why this is one mask and not two special cases.
///
/// ## Scope: the hash families only
///
/// Deliberately NOT "drop any chip with no rows". Most programs leave several
/// chips at the four-row floor (`TrivialV0` leaves eight), and dropping those
/// would dissolve the fixed-machine property for every program instead of
/// retiring two mutually-exclusive families. `BITWISE` is excluded for a
/// second reason: both families send to its `ByteAlu`/`AreBytes` buses, so it
/// is shared infrastructure, not part of either family.
///
/// ## Why this does not negotiate shape on the verify path
///
/// `airs.rs`' header refuses a program-dependent chip set *negotiated from the
/// proof*, and names the exception: shape "read from the registry, not
/// negotiated on the verify path" is legitimate, which is how `KECCAK_RND`'s
/// chunk count already works. This is that. The mask is computed from the
/// compiled program at bless time, stored in the registry entry, and folded
/// into `program_id` via [`Self::as_tag`] — so a verifier takes it from the
/// entry it resolved, never from the prover. There is no prover assertion here
/// to attack, which makes this a strictly stronger position than the RV64
/// side's conditional BLAKE3 table.
///
/// ## What decides it
///
/// A program's own compiled groups. `WrapHash::production()` is the upstream
/// cause — it is what makes an authenticating program's keccak group empty
/// once the commitment hash is BLAKE3 — but it is not the predicate: programs
/// that are ABOUT a hash (`KeccakChainV0`, `KeccakSpongeV0`) name keccak
/// directly and keep the keccak family no matter what the production hash is.
/// Keying off the global would have silently broken exactly those.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChipSet {
    /// `LFM_KECCAK`, every `KECCAK_RND` chunk, and `KECCAK_RC`.
    pub keccak: bool,
    /// `LFM_BLAKE3`.
    pub blake3: bool,
}

impl ChipSet {
    /// Both families — the shape every program had before this was conditional.
    pub const FULL: Self = Self {
        keccak: true,
        blake3: true,
    };

    /// The families a compiled program actually uses.
    pub fn for_program(program: &super::compiler::LfmProgram) -> Self {
        Self {
            keccak: program.groups.keccak.real_rows > 0,
            blake3: program.groups.blake3.real_rows > 0,
        }
    }

    /// Sub-proofs a proof under this mask carries.
    pub const fn num_airs(self, keccak_rnd_chunks: usize) -> usize {
        // The classes no family owns: all 15 less KECCAK_RND (counted per
        // chunk below), less LFM_KECCAK and KECCAK_RC (keccak's), less
        // LFM_BLAKE3 (blake3's).
        let mut n = NUM_LFM_CHIPS - 4;
        if self.keccak {
            n += 2 + keccak_rnd_chunks;
        }
        if self.blake3 {
            n += 1;
        }
        n
    }

    /// `KECCAK_RND` instances under this mask. Zero when the family is absent —
    /// the chunking policy's own floor of one exists to keep an unused chip
    /// present, which is the decision this reverses.
    pub fn keccak_rnd_chunks(self, policy_chunks: usize) -> usize {
        if self.keccak { policy_chunks } else { 0 }
    }

    /// One byte, folded into `program_id`. The mask is program shape, so a
    /// build under a different mask is a different program identity by name —
    /// not merely by a root that happens to differ.
    pub const fn as_tag(self) -> u8 {
        (self.keccak as u8) | ((self.blake3 as u8) << 1)
    }
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

/// What sets a chip's trace height, and therefore whether the padding below its
/// next power-of-two step is a margin the workload can consume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeightRule {
    /// `rows` is `real_rows.next_power_of_two()` and `real_rows` moves with the
    /// workload. The only rule under which headroom is a cliff warning: this
    /// chip's whole contribution doubles the moment the mix outgrows the gap.
    Workload,
    /// A lookup table at a compile-time constant height. Every row of it is
    /// real, so it reports zero headroom — because it is full, not because it
    /// is about to double. It cannot move with the workload at all.
    Fixed,
    /// One chunk of a split table. The chunking policy caps a chunk just under
    /// a power of two (`KECCAK_RND`: 21,845 permutations = 524,280 of 524,288
    /// rows), so a full chunk permanently reads ~0% headroom and yet can never
    /// cross one — the policy emits another chunk instead. Watching these would
    /// be watching a false alarm that is always on.
    Chunked,
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
    /// Rows the workload actually occupies, before padding to `rows`. Only the
    /// gap between the two is available to grow into; see
    /// [`LfmChipCells::headroom`].
    pub real_rows: u64,
    /// What sets this chip's height, and so whether its headroom is a margin
    /// worth watching. See [`HeightRule`].
    pub height_rule: HeightRule,
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

    /// Fraction of the committed height that is padding, i.e. how far this chip
    /// is below the next power-of-two step.
    ///
    /// A chip's cells are a STEP function of its workload: `rows` is
    /// `real_rows.next_power_of_two()`, so a chip sitting at 1% headroom doubles
    /// its whole contribution the moment the workload grows 1%, while one at 45%
    /// absorbs a near-doubling for free. That asymmetry is invisible in the row
    /// count alone, and it is what made #903's four near-empty BLAKE3 rows cost
    /// the wrap five simultaneous chip doublings — the campaign was standing
    /// 1.2% under a step on `LFM_LANES` and nobody could see it
    /// (`thoughts/shared/block-compression/WRAP-GROWTH-BISECT.md`).
    ///
    /// Only a cliff warning under [`HeightRule::Workload`]; see [`Self::at_risk`].
    pub fn headroom(&self) -> f64 {
        if self.rows == 0 {
            return 0.0;
        }
        (self.rows - self.real_rows) as f64 / self.rows as f64
    }

    /// Whether this chip's [`Self::headroom`] is a margin the workload can
    /// actually consume, rather than a constant of the machine.
    pub fn at_risk(&self) -> bool {
        self.height_rule == HeightRule::Workload
    }

    /// Base-field equivalents this chip would ADD by crossing its next step —
    /// its height doubles, so it adds exactly what it already contributes.
    /// An extension element is three base felts, matching the census total.
    pub fn cliff_cost(&self) -> u64 {
        self.main_cells() + 3 * self.aux_cells()
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

/// One entry of [`lfm_chip_census_with_hasher`]'s per-chip table: a chip class's
/// two heights plus its width decomposition.
///
/// Named rather than a positional tuple because `real_rows` and `padded_rows`
/// are interchangeable at a glance, and swapping them would invert every
/// headroom the panel prints while leaving the cell totals — the numbers under
/// test — correct.
struct ChipShape {
    real_rows: u64,
    padded_rows: u64,
    height_rule: HeightRule,
    num_cols: usize,
    prep: usize,
    interactions: usize,
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
    let g = &program.groups;
    // A workload-sized chip: the compiler already computed both heights, and
    // `padded_rows` is `real_rows.next_power_of_two()`, so the gap between them
    // is this chip's distance to its next doubling.
    let sized = |group: &super::compiler::ColumnGroup,
                 num_cols: usize,
                 prep: usize,
                 interactions: usize| ChipShape {
        real_rows: group.real_rows as u64,
        padded_rows: group.padded_rows as u64,
        height_rule: HeightRule::Workload,
        num_cols,
        prep,
        interactions,
    };
    // A lookup table: its height is a compile-time constant and every row of it
    // is real, so it reports no headroom and is flagged as unable to move.
    let lookup = |rows: u64, num_cols: usize, prep: usize, interactions: usize| ChipShape {
        real_rows: rows,
        padded_rows: rows,
        height_rule: HeightRule::Fixed,
        num_cols,
        prep,
        interactions,
    };
    // Every chip class except `KECCAK_RND`, which is counted per chunk below.
    let per_chip: [ChipShape; NUM_LFM_CHIPS - 1] = [
        sized(
            &g.const_,
            const_::cols::NUM_COLUMNS,
            layout::const_::PREP_WIDTH,
            const_::bus_interactions().len(),
        ),
        sized(
            &g.balu,
            balu::cols::NUM_COLUMNS,
            layout::balu::PREP_WIDTH,
            balu::bus_interactions().len(),
        ),
        sized(
            &g.xalu,
            xalu::cols::NUM_COLUMNS,
            layout::xalu::PREP_WIDTH,
            xalu::bus_interactions().len(),
        ),
        sized(
            &g.select,
            select::cols::NUM_COLUMNS,
            layout::select::PREP_WIDTH,
            select::bus_interactions().len(),
        ),
        sized(
            &g.bitdec,
            bitdec::cols::NUM_COLUMNS,
            layout::bitdec::PREP_WIDTH,
            bitdec::bus_interactions().len(),
        ),
        sized(
            &g.hash,
            hash::num_columns(hasher),
            layout::hash::PREP_WIDTH,
            hash::bus_interactions(hasher).len(),
        ),
        sized(
            &g.keccak,
            keccak::cols::NUM_COLUMNS,
            layout::keccak::PREP_WIDTH,
            keccak::bus_interactions().len(),
        ),
        sized(
            &g.lanes,
            lanes::cols::NUM_COLUMNS,
            layout::lanes::PREP_WIDTH,
            lanes::bus_interactions().len(),
        ),
        sized(
            &g.hint,
            hint::cols::NUM_COLUMNS,
            layout::hint::PREP_WIDTH,
            hint::bus_interactions().len(),
        ),
        sized(
            &g.public,
            public::cols::NUM_COLUMNS,
            layout::public::PREP_WIDTH,
            public::bus_interactions().len(),
        ),
        lookup(
            layout::range::NUM_ROWS as u64,
            range::cols::NUM_COLUMNS,
            layout::range::PREP_WIDTH,
            range::bus_interactions().len(),
        ),
        sized(
            &g.blake3,
            blake3_chip::cols::NUM_COLUMNS,
            layout::blake3::PREP_WIDTH,
            blake3_chip::bus_interactions().len(),
        ),
        // The keccak family's two fixed tables. `KECCAK_RND`'s chunks follow.
        lookup(
            keccak_rc::NUM_ROWS as u64,
            keccak_rc::cols::NUM_COLUMNS,
            keccak_rc::NUM_PRECOMPUTED_COLS,
            keccak_rc::bus_interactions().len(),
        ),
        lookup(
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
    // The census reports the chips the machine actually instantiates, so it
    // gates on the same mask `air_refs` does — a census that counted an absent
    // family would describe a different machine than the one being proved,
    // which is precisely what this function's doc promises it cannot.
    let chip_set = ChipSet::for_program(program);
    let mut census = Vec::with_capacity(per_chip.len() + 1);
    for (slot, shape) in per_chip.into_iter().enumerate() {
        if slot == KECCAK_RND_SLOT && chip_set.keccak {
            for (perms, rows) in keccak_rnd_chunk_permutations(program)
                .into_iter()
                .zip(keccak_rnd_chunk_rows(program))
            {
                census.push(LfmChipCells {
                    name: LFM_CHIP_NAMES[KECCAK_RND_SLOT],
                    rows: rows as u64,
                    // `keccak_rnd_chunk_rows` pads exactly this product, so the
                    // pair is the chunk's real-vs-committed height.
                    real_rows: (perms * super::chunking::KECCAK_RND_ROWS_PER_PERMUTATION) as u64,
                    height_rule: HeightRule::Chunked,
                    main_cols: keccak_rnd::cols::NUM_COLUMNS,
                    aux_cols: rnd_interactions.div_ceil(2),
                });
            }
        }
        // `per_chip`'s last two entries are chip classes 13 and 14, which sit
        // at indices 12 and 13 of that array — hence the shift past the
        // `KECCAK_RND` slot rather than a plain index.
        let class = if slot >= KECCAK_RND_SLOT {
            slot + 1
        } else {
            slot
        };
        let present = match class {
            KECCAK_SLOT | KECCAK_RC_SLOT => chip_set.keccak,
            BLAKE3_SLOT => chip_set.blake3,
            _ => true,
        };
        if !present {
            continue;
        }
        let ChipShape {
            real_rows,
            padded_rows,
            height_rule,
            num_cols,
            prep,
            interactions,
        } = shape;
        census.push(LfmChipCells {
            name: LFM_CHIP_NAMES[class],
            rows: padded_rows,
            real_rows,
            height_rule,
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
    /// Which hash families this set instantiates. The unused family's AIRs are
    /// still BUILT (construction is free — there is no keygen here) but are not
    /// offered to the prover or the verifier, so a proof never carries them.
    chip_set: ChipSet,
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
        chip_set: ChipSet,
    ) -> Self {
        Self::new_with_hasher(
            roots,
            options,
            keccak_rnd_chunks,
            HasherKind::default(),
            chip_set,
        )
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
        chip_set: ChipSet,
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
            chip_set,
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
        ];
        // The frozen order is unchanged; an absent family leaves a hole in it
        // rather than moving anything after it.
        if self.chip_set.keccak {
            refs.push(&self.keccak);
        }
        refs.push(&self.lanes);
        refs.push(&self.hint);
        refs.push(&self.public);
        refs.push(&self.range);
        if self.chip_set.blake3 {
            refs.push(&self.blake3);
        }
        if self.chip_set.keccak {
            refs.extend(self.keccak_rnd.iter().map(|a| a as DynLfmAir<'_>));
            refs.push(&self.keccak_rc);
        }
        refs.push(&self.bitwise);
        refs
    }

    /// Prove-side projection, frozen order (must match `air_refs`).
    ///
    /// When the keccak family is present, `traces.keccak_rnd` must have exactly
    /// one trace per chunk; a mismatch would silently shorten the pair list
    /// under `zip`, so it is asserted. When the family is ABSENT the counts
    /// legitimately differ: trace building is mask-blind (the chunk split
    /// always yields at least one, empty, trace), and the mask simply never
    /// pairs it — so the assert is gated, or every keccak-less program would
    /// panic any debug-profile prove while release proved fine.
    #[allow(clippy::type_complexity)]
    pub fn air_trace_pairs<'a>(
        &'a self,
        traces: &'a mut LfmTraces,
    ) -> Vec<(DynLfmAir<'a>, &'a mut TraceTable<F, E>, &'a ())> {
        debug_assert!(
            !self.chip_set.keccak || self.keccak_rnd.len() == traces.keccak_rnd.len(),
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
        ];
        // Same gating as `air_refs`, in the same order — these two ARE the
        // proof's layout and must move together.
        if self.chip_set.keccak {
            pairs.push((&self.keccak, &mut traces.keccak, &()));
        }
        pairs.push((&self.lanes, &mut traces.lanes, &()));
        pairs.push((&self.hint, &mut traces.hint, &()));
        pairs.push((&self.public, &mut traces.public, &()));
        pairs.push((&self.range, &mut traces.range, &()));
        if self.chip_set.blake3 {
            pairs.push((&self.blake3, &mut traces.blake3, &()));
        }
        if self.chip_set.keccak {
            pairs.extend(
                self.keccak_rnd
                    .iter()
                    .zip(traces.keccak_rnd.iter_mut())
                    .map(|(air, trace)| (air as DynLfmAir<'a>, trace, &())),
            );
            pairs.push((&self.keccak_rc, &mut traces.keccak_rc, &()));
        }
        pairs.push((&self.bitwise, &mut traces.bitwise, &()));
        pairs
    }
}

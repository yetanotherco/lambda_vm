//! `LfmAirs` — the machine's fixed 14-chip AIR set, a sibling of `VmAirs`.
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

use super::chips::{balu, bitdec, const_, hash, hint, keccak, lanes, public, range, select, xalu};
use super::layout;
use super::trace::LfmTraces;

type F = GoldilocksField;
type E = GoldilocksExtension;

pub type LfmAir<CS> = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), CS>;
pub type DynLfmAir<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

/// The frozen chip order — everywhere: roots, digests, traces, proofs.
///
/// Slots 11–13 are the production keccak family, hosted unchanged. They belong
/// to the *fixed* machine, so **every** LFM proof carries them — including the
/// 2^20-row BITWISE table, which costs a few seconds of prove time even for a
/// program containing no keccak at all. That is the deliberate price of the
/// fixed-machine principle: the chip set never varies with the program, only
/// heights do, so a program stays nothing but a vector of preprocessed roots
/// plus a registry entry. Making the set program-dependent would move shape
/// negotiation onto the verify path, which this design refuses.
pub const NUM_LFM_CHIPS: usize = 14;
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
    "KECCAK_RND",
    "KECCAK_RC",
    "BITWISE",
];

/// Slot of `KECCAK_RND`, the one AIR in the set with **no** preprocessed
/// columns — it has no root to supply, pin, or bind into the program digest.
pub const KECCAK_RND_SLOT: usize = 11;

/// `KECCAK_RND`'s trace height: 24 rows per permutation, padded — the same
/// `.next_power_of_two().max(4)` rule `generate_keccak_rnd_trace` applies.
fn keccak_rnd_rows(program: &super::compiler::LfmProgram) -> usize {
    (program.groups.keccak.real_rows * 24)
        .next_power_of_two()
        .max(4)
}

/// Trace-cell counts for a compiled program, the LFM analogue of the VM's
/// `total_field_elements` / `total_auxiliary_field_elements` (same
/// semantics: main counts base-field value cells excluding preprocessed
/// columns; aux counts extension-field elements, one per aux column per
/// row). This is the kill-risk-3 instrument: machine cells per verification
/// vs the verified proof's own cells.
pub fn lfm_cell_counts(program: &super::compiler::LfmProgram) -> (u64, u64) {
    let range_rows = layout::range::NUM_ROWS as u64;
    let g = &program.groups;
    let per_chip: [(u64, usize, usize, usize); NUM_LFM_CHIPS] = [
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
            hash::cols::NUM_COLUMNS,
            layout::hash::PREP_WIDTH,
            hash::bus_interactions().len(),
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
        // The keccak family's own heights: KECCAK_RND is 24 rows per
        // permutation, the other two are fixed tables.
        (
            keccak_rnd_rows(program) as u64,
            keccak_rnd::cols::NUM_COLUMNS,
            0,
            keccak_rnd::bus_interactions().len(),
        ),
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
    let mut main = 0u64;
    let mut aux = 0u64;
    for (rows, num_cols, prep, interactions) in per_chip {
        main += rows * (num_cols - prep) as u64;
        aux += rows * interactions.div_ceil(2) as u64;
    }
    (main, aux)
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
    keccak_rnd: LfmAir<keccak_rnd::KeccakRndConstraints>,
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
    /// freshly built) instruction-column-group roots, in the frozen order.
    pub fn new(roots: &[Commitment; NUM_LFM_CHIPS], options: &ProofOptions) -> Self {
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
                hash::cols::NUM_COLUMNS,
                hash::bus_interactions(),
                options,
                hash::HashConstraints,
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
            // KECCAK_RND has no preprocessed columns: `roots[KECCAK_RND_SLOT]`
            // is the all-zero sentinel and is never consulted. Its correctness
            // is entirely its own constraints plus bus balance, both
            // program-independent, so there is nothing for a root to pin.
            keccak_rnd: build_air_no_prep(
                keccak_rnd::cols::NUM_COLUMNS,
                keccak_rnd::bus_interactions(),
                options,
                keccak_rnd::KeccakRndConstraints,
                LFM_CHIP_NAMES[11],
            ),
            keccak_rc: build_air(
                keccak_rc::cols::NUM_COLUMNS,
                keccak_rc::bus_interactions(),
                options,
                EmptyConstraints,
                LFM_CHIP_NAMES[12],
                roots[12],
                keccak_rc::NUM_PRECOMPUTED_COLS,
            ),
            bitwise: build_air(
                bitwise::cols::NUM_COLUMNS,
                bitwise::bus_interactions(),
                options,
                EmptyConstraints,
                LFM_CHIP_NAMES[13],
                roots[13],
                bitwise::NUM_PRECOMPUTED_COLS,
            ),
        }
    }

    /// Verify-side projection, frozen order (must match `air_trace_pairs`).
    pub fn air_refs(&self) -> Vec<DynLfmAir<'_>> {
        vec![
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
            &self.keccak_rnd,
            &self.keccak_rc,
            &self.bitwise,
        ]
    }

    /// Prove-side projection, frozen order (must match `air_refs`).
    #[allow(clippy::type_complexity)]
    pub fn air_trace_pairs<'a>(
        &'a self,
        traces: &'a mut LfmTraces,
    ) -> Vec<(DynLfmAir<'a>, &'a mut TraceTable<F, E>, &'a ())> {
        vec![
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
            (&self.keccak_rnd, &mut traces.keccak_rnd, &()),
            (&self.keccak_rc, &mut traces.keccak_rc, &()),
            (&self.bitwise, &mut traces.bitwise, &()),
        ]
    }
}

//! Equal-height LFM chips proved as ONE AIR (`LAMBDA_VM_ZF_LFM_MERGE=1`).
//!
//! An LFM proof carries one sub-proof a chip: its own trees, quotient, FRI,
//! grind and queries, and a parent re-verifies each of them. Four chips —
//! `LFM_BALU`, `LFM_SELECT`, `LFM_LANES` and `LFM_HINT` — sit at the same padded
//! height in every leaf of today's trees, so under this format the chips of a
//! program that share a padded height are proved as one AIR: their columns side
//! by side over the same rows.
//!
//! # Why the merged AIR proves exactly what the parts did
//!
//! It is the direct product of the parts, nothing more:
//! - **columns**: every part's preprocessed prefix first, in part order, then
//!   every part's value columns, in part order (the STARK keeps preprocessed
//!   columns as a prefix: `AirWithBuses::with_preprocessed`). A part's columns
//!   keep their order inside each region; [`PartMap`] is the only translation.
//! - **constraints**: each part's own body, run through [`Remap`], which reads
//!   the part's columns where they now sit and shifts its constraint indices
//!   past the parts before it. No body reads another part's column.
//! - **buses**: the parts' interactions, concatenated in part order with every
//!   column index remapped. The bus sum over the merged rows is the sum of the
//!   parts' sums, so the LogUp closure is unchanged.
//! - **rows**: equal padded heights only — nothing is padded to merge — so a
//!   part's next-row window wraps at the same row it did alone, and the merged
//!   trace is the parts' traces side by side, row for row.
//!
//! # Where the layout comes from
//!
//! A function of the program's padded heights and the format flag
//! (`ProofFormat::merge_equal_height_tables`), computed where the artifacts are
//! built, stored in them ([`LfmMerge`]) and folded into `program_id`. A
//! verifier takes it from the artifacts it derived, never from a proof.

use stark::config::Commitment;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet, RowDomain};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity};

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

use super::airs::{LFM_CHIP_NAMES, NUM_LFM_CHIPS};
use super::chips::{balu, hint, lanes, select};
use super::compiler::{ColumnGroup, LfmProgram};
use super::layout;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// A chip a merge may take: the four the trees hold at one height. Not
/// `LFM_HASH` (the socket, with its own chunks and card paths), not
/// `LFM_PUBLIC` (it carries the public-word closure), not the fixed tables, and
/// not the small chips, which sit at heights of their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeChip {
    Balu,
    Select,
    Lanes,
    Hint,
}

/// The candidates in slot order.
pub const MERGE_CANDIDATES: [MergeChip; 4] = [
    MergeChip::Balu,
    MergeChip::Select,
    MergeChip::Lanes,
    MergeChip::Hint,
];

impl MergeChip {
    /// The chip's slot in the frozen chip order (`airs::LFM_CHIP_NAMES`).
    pub const fn slot(self) -> usize {
        match self {
            Self::Balu => 1,
            Self::Select => 3,
            Self::Lanes => 7,
            Self::Hint => 8,
        }
    }

    /// The candidate at chip slot `slot`, if it is one.
    pub fn at_slot(slot: usize) -> Option<Self> {
        MERGE_CANDIDATES.into_iter().find(|c| c.slot() == slot)
    }
}

/// One merged AIR of a program: the chips it holds, in slot order, their common
/// log-height and the root of their preprocessed columns side by side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergedTable {
    pub chips: Vec<MergeChip>,
    pub log_height: u8,
    pub root: Commitment,
}

/// A program's merged AIRs. Empty — the default, and every program built with
/// the flag off — is the machine of one AIR a chip, byte for byte.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LfmMerge {
    pub tables: Vec<MergedTable>,
}

impl LfmMerge {
    /// No merged AIR.
    pub const fn none() -> Self {
        Self { tables: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    /// The groups of chips, without the roots.
    pub fn plan(&self) -> MergePlan {
        self.tables.iter().map(|t| t.chips.clone()).collect()
    }

    /// How many AIRs the merge removes from a proof: a table of `k` chips is
    /// one AIR where there were `k`.
    pub fn airs_saved(&self) -> usize {
        self.tables
            .iter()
            .map(|t| t.chips.len().saturating_sub(1))
            .sum()
    }
}

/// The merged AIRs' chips, in AIR order; each group in slot order.
pub type MergePlan = Vec<Vec<MergeChip>>;

/// Which merged AIR, if any, chip slot `slot` belongs to under `plan`.
pub fn group_of(plan: &[Vec<MergeChip>], slot: usize) -> Option<usize> {
    plan.iter()
        .position(|g| g.iter().any(|c| c.slot() == slot))
}

/// What a chip slot is in the AIR order under `plan`: its own AIR, the first
/// chip of merged AIR `g` (where that AIR sits), or a later chip of one (no AIR
/// of its own).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotRole {
    Own,
    First(usize),
    Member,
}

pub fn role(plan: &[Vec<MergeChip>], slot: usize) -> SlotRole {
    match group_of(plan, slot) {
        None => SlotRole::Own,
        Some(g) if plan[g].first().map(|c| c.slot()) == Some(slot) => SlotRole::First(g),
        Some(_) => SlotRole::Member,
    }
}

/// The merge plan for chips at `log_heights` (the artifacts' per-slot
/// log-heights): the candidates grouped by log-height in slot order, every
/// class of two or more one merged AIR, ordered by its first chip. Empty when
/// `on` is false.
pub fn plan_for_heights(log_heights: &[u8; NUM_LFM_CHIPS], on: bool) -> MergePlan {
    if !on {
        return Vec::new();
    }
    let mut groups: MergePlan = Vec::new();
    for chip in MERGE_CANDIDATES {
        let h = log_heights[chip.slot()];
        match groups
            .iter_mut()
            .find(|g| g.first().is_some_and(|c| log_heights[c.slot()] == h))
        {
            Some(g) => g.push(chip),
            None => groups.push(vec![chip]),
        }
    }
    groups.retain(|g| g.len() > 1);
    groups
}

/// The merge plan for a compiled program, from its groups' padded heights —
/// what [`plan_for_heights`] reads off the artifacts the same program builds.
pub fn plan_for_program(program: &LfmProgram, on: bool) -> MergePlan {
    let mut heights = [0u8; NUM_LFM_CHIPS];
    for chip in MERGE_CANDIDATES {
        heights[chip.slot()] = group_of_program(program, chip).padded_rows.trailing_zeros() as u8;
    }
    plan_for_heights(&heights, on)
}

/// A candidate chip's instruction column group in `program`.
pub fn group_of_program(program: &LfmProgram, chip: MergeChip) -> &ColumnGroup {
    let g = &program.groups;
    match chip {
        MergeChip::Balu => &g.balu,
        MergeChip::Select => &g.select,
        MergeChip::Lanes => &g.lanes,
        MergeChip::Hint => &g.hint,
    }
}

/// A merge candidate's AIR shape: preprocessed prefix, total width, buses and
/// constraint body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartBody {
    Balu,
    Select,
    /// LogUp only (`LFM_LANES`, `LFM_HINT`).
    Empty,
}

#[derive(Clone, Copy, Debug)]
pub struct PartShape {
    pub chip: MergeChip,
    pub prep: usize,
    pub width: usize,
    pub body: PartBody,
}

impl PartShape {
    pub fn of(chip: MergeChip) -> Self {
        let (prep, width, body) = match chip {
            MergeChip::Balu => (
                layout::balu::PREP_WIDTH,
                balu::cols::NUM_COLUMNS,
                PartBody::Balu,
            ),
            MergeChip::Select => (
                layout::select::PREP_WIDTH,
                select::cols::NUM_COLUMNS,
                PartBody::Select,
            ),
            MergeChip::Lanes => (
                layout::lanes::PREP_WIDTH,
                lanes::cols::NUM_COLUMNS,
                PartBody::Empty,
            ),
            MergeChip::Hint => (
                layout::hint::PREP_WIDTH,
                hint::cols::NUM_COLUMNS,
                PartBody::Empty,
            ),
        };
        Self {
            chip,
            prep,
            width,
            body,
        }
    }

    pub fn interactions(&self) -> Vec<BusInteraction> {
        match self.chip {
            MergeChip::Balu => balu::bus_interactions(),
            MergeChip::Select => select::bus_interactions(),
            MergeChip::Lanes => lanes::bus_interactions(),
            MergeChip::Hint => hint::bus_interactions(),
        }
    }

    /// The base constraints the part's body emits (its whole meta: the bodies
    /// are base-only).
    pub fn num_base_constraints(&self) -> usize {
        match self.body {
            PartBody::Balu => ConstraintSet::<F, E>::meta(&balu::BaluConstraints).len(),
            PartBody::Select => ConstraintSet::<F, E>::meta(&select::SelectConstraints).len(),
            PartBody::Empty => 0,
        }
    }

    pub fn max_degree(&self) -> usize {
        match self.body {
            PartBody::Balu => ConstraintSet::<F, E>::max_degree(&balu::BaluConstraints),
            PartBody::Select => ConstraintSet::<F, E>::max_degree(&select::SelectConstraints),
            PartBody::Empty => {
                ConstraintSet::<F, E>::max_degree(&stark::constraints::builder::EmptyConstraints)
            }
        }
    }
}

/// Where one part's columns sit in the merged AIR: `map[c]` is part column
/// `c`'s merged column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartMap {
    pub chip: MergeChip,
    pub map: Vec<usize>,
    /// The merged column of the part's first preprocessed cell.
    pub prep_at: usize,
    /// The merged column of the part's first value cell.
    pub main_at: usize,
}

/// The merged AIR's whole layout: each part's map, the merged widths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergedLayout {
    pub parts: Vec<PartMap>,
    pub num_prep: usize,
    pub num_columns: usize,
}

impl MergedLayout {
    /// `slots` side by side: every part's prefix, then every part's values.
    pub fn of(chips: &[MergeChip]) -> Self {
        let shapes: Vec<PartShape> = chips.iter().map(|&c| PartShape::of(c)).collect();
        let num_prep: usize = shapes.iter().map(|s| s.prep).sum();
        let num_columns: usize = shapes.iter().map(|s| s.width).sum();
        let mut prep_at = 0;
        let mut main_at = num_prep;
        let parts = shapes
            .iter()
            .map(|s| {
                let map = (0..s.width)
                    .map(|c| {
                        if c < s.prep {
                            prep_at + c
                        } else {
                            main_at + (c - s.prep)
                        }
                    })
                    .collect();
                let part = PartMap {
                    chip: s.chip,
                    map,
                    prep_at,
                    main_at,
                };
                prep_at += s.prep;
                main_at += s.width - s.prep;
                part
            })
            .collect();
        Self {
            parts,
            num_prep,
            num_columns,
        }
    }
}

/// The merged AIR's name: the parts' chip names joined, so a log line names
/// the chips it holds.
pub fn merged_name(chips: &[MergeChip]) -> String {
    let parts: Vec<&str> = chips
        .iter()
        .map(|c| LFM_CHIP_NAMES[c.slot()].trim_start_matches("LFM_"))
        .collect();
    format!("LFM_MERGED[{}]", parts.join("+"))
}

// ============================ the constraint body ==========================

/// One part of a [`MergedConstraints`]: its body, its column map and the index
/// its first constraint takes.
#[derive(Clone, Debug)]
struct MergedPart {
    body: PartBody,
    map: Vec<usize>,
    idx_offset: usize,
}

/// The merged AIR's constraint set: each part's own body, in part order, its
/// columns remapped and its indices shifted ([`Remap`]).
#[derive(Clone, Debug)]
pub struct MergedConstraints {
    parts: Vec<MergedPart>,
    max_degree: usize,
}

impl MergedConstraints {
    pub fn new(layout: &MergedLayout) -> Self {
        let mut idx_offset = 0;
        let mut max_degree = 0;
        let parts = layout
            .parts
            .iter()
            .map(|p| {
                let shape = PartShape::of(p.chip);
                let part = MergedPart {
                    body: shape.body,
                    map: p.map.clone(),
                    idx_offset,
                };
                idx_offset += shape.num_base_constraints();
                max_degree = max_degree.max(shape.max_degree());
                part
            })
            .collect();
        Self { parts, max_degree }
    }
}

impl ConstraintSet<F, E> for MergedConstraints {
    /// The largest of the parts' declared degrees: the merged quotient is as
    /// wide as its widest part's.
    fn max_degree(&self) -> usize {
        self.max_degree
    }

    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        for part in &self.parts {
            #[cfg(test)]
            if part.body == PartBody::Select && test_hooks::select_body_dropped() {
                continue;
            }
            let mut r = Remap {
                inner: &mut *b,
                map: &part.map,
                idx_offset: part.idx_offset,
            };
            match part.body {
                PartBody::Balu => balu::BaluConstraints.eval(&mut r),
                PartBody::Select => select::SelectConstraints.eval(&mut r),
                PartBody::Empty => {}
            }
        }
    }
}

/// A part's view of the merged builder: its `main(offset, c)` reads merged
/// column `map[c]`, and its constraint `i` is merged constraint
/// `idx_offset + i`. Every other leaf passes through: the part bodies are
/// base-only and read no aux column, challenge or table offset of their own
/// (`a_part_body_reads_only_its_main_columns` pins that).
pub struct Remap<'a, B> {
    inner: &'a mut B,
    map: &'a [usize],
    idx_offset: usize,
}

impl<B: ConstraintBuilder<F, E>> ConstraintBuilder<F, E> for Remap<'_, B> {
    type Expr = B::Expr;
    type ExprE = B::ExprE;

    fn main(&self, offset: usize, col: usize) -> Self::Expr {
        self.inner.main(offset, self.map[col])
    }
    fn aux(&self, offset: usize, col: usize) -> Self::ExprE {
        self.inner.aux(offset, col)
    }
    fn challenge(&self, idx: usize) -> Self::ExprE {
        self.inner.challenge(idx)
    }
    fn alpha_pow(&self, idx: usize) -> Self::ExprE {
        self.inner.alpha_pow(idx)
    }
    fn table_offset(&self) -> Self::ExprE {
        self.inner.table_offset()
    }
    fn const_base(&self, v: u64) -> Self::Expr {
        self.inner.const_base(v)
    }
    fn const_signed(&self, v: i64) -> Self::Expr {
        self.inner.const_signed(v)
    }
    fn one(&self) -> Self::Expr {
        self.inner.one()
    }
    fn zero(&self) -> Self::Expr {
        self.inner.zero()
    }
    fn emit_base_rows(&mut self, constraint_idx: usize, rows: RowDomain, e: Self::Expr) {
        self.inner
            .emit_base_rows(self.idx_offset + constraint_idx, rows, e)
    }
    fn emit_ext_rows(&mut self, constraint_idx: usize, rows: RowDomain, e: Self::ExprE) {
        self.inner
            .emit_ext_rows(self.idx_offset + constraint_idx, rows, e)
    }
    fn fold_fingerprint_term(
        &self,
        fp: Self::ExprE,
        v: Self::Expr,
        alpha_idx: usize,
    ) -> Self::ExprE {
        self.inner.fold_fingerprint_term(fp, v, alpha_idx)
    }
}

// =============================== the buses =================================

/// `interaction` with every column it reads moved by `map`.
///
/// A multi-column packing must stay contiguous under `map`. A part's map moves
/// its prefix and its values as two blocks, so only a packing that crossed the
/// prefix's end could break — and no merge candidate has one: every packing
/// they use is one column (`every_merge_remaps_its_buses` pins it, against the
/// unmapped reading of the same interaction).
pub fn remap_interaction(interaction: &BusInteraction, map: &[usize]) -> BusInteraction {
    let term = |t: &LinearTerm| match *t {
        LinearTerm::Column {
            coefficient,
            column,
        } => LinearTerm::Column {
            coefficient,
            column: map[column],
        },
        LinearTerm::ColumnUnsigned {
            coefficient,
            column,
        } => LinearTerm::ColumnUnsigned {
            coefficient,
            column: map[column],
        },
        LinearTerm::Constant(c) => LinearTerm::Constant(c),
    };
    let multiplicity = match &interaction.multiplicity {
        Multiplicity::One => Multiplicity::One,
        Multiplicity::Column(c) => Multiplicity::Column(map[*c]),
        Multiplicity::Sum(a, b) => Multiplicity::Sum(map[*a], map[*b]),
        Multiplicity::Negated(c) => Multiplicity::Negated(map[*c]),
        Multiplicity::Diff(a, b) => Multiplicity::Diff(map[*a], map[*b]),
        Multiplicity::Sum3(a, b, c) => Multiplicity::Sum3(map[*a], map[*b], map[*c]),
        Multiplicity::Linear(terms) => Multiplicity::Linear(terms.iter().map(term).collect()),
    };
    let values = interaction
        .values
        .iter()
        .map(|v| match v {
            BusValue::Packed {
                start_column,
                packing,
            } => {
                let start = map[*start_column];
                debug_assert!(
                    (0..packing.num_columns()).all(|k| map[start_column + k] == start + k),
                    "a packing must stay contiguous in the merged layout"
                );
                BusValue::Packed {
                    start_column: start,
                    packing: *packing,
                }
            }
            BusValue::Linear(terms) => BusValue::Linear(terms.iter().map(term).collect()),
        })
        .collect();
    BusInteraction {
        bus_id: interaction.bus_id,
        multiplicity,
        values,
        is_sender: interaction.is_sender,
    }
}

/// The merged AIR's interactions: each part's, in part order, remapped.
pub fn merged_interactions(layout: &MergedLayout) -> Vec<BusInteraction> {
    layout
        .parts
        .iter()
        .flat_map(|p| {
            PartShape::of(p.chip)
                .interactions()
                .iter()
                .map(|i| remap_interaction(i, &p.map))
                .collect::<Vec<_>>()
        })
        .collect()
}

// ======================== the preprocessed columns =========================

/// The parts' instruction column groups side by side, row for row: the merged
/// AIR's preprocessed prefix, committed once (its root is [`MergedTable::root`]).
/// The groups must share their padded height.
pub fn merged_group(groups: &[&ColumnGroup]) -> ColumnGroup {
    let rows = groups.first().map_or(0, |g| g.padded_rows);
    debug_assert!(groups.iter().all(|g| g.padded_rows == rows));
    let width: usize = groups.iter().map(|g| g.width).sum();
    let mut data = crate::tables::types::zeroed_fe_vec(rows * width);
    let fill_row = |row: usize, out: &mut [FE]| {
        let mut at = 0;
        for g in groups {
            out[at..at + g.width].copy_from_slice(&g.data[row * g.width..(row + 1) * g.width]);
            at += g.width;
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, out)| fill_row(row, out));
    }
    #[cfg(not(feature = "parallel"))]
    for (row, out) in data.chunks_mut(width).enumerate() {
        fill_row(row, out);
    }
    ColumnGroup {
        width,
        real_rows: groups.iter().map(|g| g.real_rows).max().unwrap_or(0),
        padded_rows: rows,
        data,
    }
}

/// [`merged_group`] for each group of `plan`, from `program`'s groups.
pub fn merged_groups(program: &LfmProgram, plan: &[Vec<MergeChip>]) -> Vec<ColumnGroup> {
    plan.iter()
        .map(|chips| {
            let groups: Vec<&ColumnGroup> =
                chips.iter().map(|&c| group_of_program(program, c)).collect();
            merged_group(&groups)
        })
        .collect()
}

/// The mutation hook of `merge_tests`: a merged AIR whose `LFM_SELECT` body is
/// left out, to show that body is what refuses a bad `LFM_SELECT` row. Test
/// builds only; a process-wide switch, so every test that builds a merged AIR
/// holds `merge_tests`' lock.
#[cfg(test)]
pub(crate) mod test_hooks {
    use std::sync::atomic::{AtomicBool, Ordering};

    static DROP_SELECT_BODY: AtomicBool = AtomicBool::new(false);

    pub(crate) fn select_body_dropped() -> bool {
        DROP_SELECT_BODY.load(Ordering::SeqCst)
    }

    pub(crate) fn set_select_body_dropped(on: bool) {
        DROP_SELECT_BODY.store(on, Ordering::SeqCst);
    }
}

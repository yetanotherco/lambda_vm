//! On-demand trace source for the streaming prover.
//!
//! Approach 1 step C.2b: the chunked tables are built as empty placeholders and
//! their rows live only as the routed op lists they came from. The prover asks
//! this provider for a trace at the two points it needs one — the Round 1 main
//! commit, and the table's fused chain — and drops it again after each.

use std::collections::HashMap;
use std::sync::Mutex;

use stark::prover::TraceProvider;
use stark::trace::TraceTable;

use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::{CollectedOps, TableKind, Traces};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

/// The groups of chunked tables, in the order `VmAirs::air_trace_pairs` emits
/// them. `None` marks a group that stays resident (PAGE), which still consumes
/// AIR indices and so must be walked over.
pub(crate) const GROUP_ORDER: [Option<TableKind>; 15] = [
    Some(TableKind::Cpu),
    Some(TableKind::Lt),
    Some(TableKind::Shift),
    Some(TableKind::Memw),
    Some(TableKind::MemwAligned),
    Some(TableKind::Load),
    Some(TableKind::Mul),
    Some(TableKind::Dvrm),
    Some(TableKind::Branch),
    None, // PAGE — built from the ELF image, not from an op list
    Some(TableKind::MemwRegister),
    Some(TableKind::Eq),
    Some(TableKind::Bytewise),
    Some(TableKind::Store),
    Some(TableKind::Cpu32),
];

/// Number of tables that are in every proof, emitted before HALT: BITWISE,
/// DECODE, KECCAK_RC, REGISTER.
///
/// The accelerator chips used to be here too. They are now elided when the run
/// does not reach them, so they are variable-length groups sitting between HALT
/// and the chunked groups — see [`accel_lengths`].
pub(crate) const NUM_FIXED_AIRS: usize = 4;

/// One accelerator chip's group of AIRs.
///
/// A chip the run never reached has no table and no AIR, so these are
/// variable-length groups between HALT and the chunked groups rather than
/// fixed slots.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AccelGroup {
    Commit = 0,
    Keccak = 1,
    KeccakRnd = 2,
    Ecsm = 3,
    Ecdas = 4,
    Hint = 5,
}

/// The accelerator groups in the order `VmAirs::air_trace_pairs` emits them.
///
/// **This is the only place that order is written.** Every pass reads it from
/// here and indexes the tables by it; none re-types the list. The last reorder
/// desynchronised three hand-written copies of it at once, and the failure is
/// silent — a proof whose roots are absorbed in the wrong order simply does not
/// verify, with nothing pointing at the list that was missed.
pub(crate) const ACCEL_ORDER: [AccelGroup; 6] = [
    AccelGroup::Commit,
    AccelGroup::Keccak,
    AccelGroup::KeccakRnd,
    AccelGroup::Ecsm,
    AccelGroup::Ecdas,
    AccelGroup::Hint,
];

// `slot` is the enum's discriminant and `ACCEL_ORDER` is a list; two encodings
// of one order is how it drifts. This refuses to compile unless they agree, so
// reordering the list without reordering the discriminants is a build error
// rather than a proof that does not verify.
const _: () = {
    let mut i = 0;
    while i < ACCEL_ORDER.len() {
        assert!(
            ACCEL_ORDER[i].slot() == i,
            "ACCEL_ORDER must list the groups in discriminant order"
        );
        i += 1;
    }
};

impl AccelGroup {
    /// Position in [`ACCEL_ORDER`], which is also the index into
    /// `AccumulatedTables::accel`.
    pub(crate) const fn slot(self) -> usize {
        self as usize
    }

    /// The name the transcript's diagnostics use for this chip.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            AccelGroup::Commit => "COMMIT",
            AccelGroup::Keccak => "KECCAK",
            AccelGroup::KeccakRnd => "KECCAK_RND",
            AccelGroup::Ecsm => "ECSM",
            AccelGroup::Ecdas => "ECDAS",
            AccelGroup::Hint => "HINT",
        }
    }

    /// How many AIRs this group has, from the counts the statement binds.
    pub(crate) fn count(self, counts: &crate::TableCounts) -> usize {
        match self {
            AccelGroup::Commit => counts.commit,
            AccelGroup::Keccak => counts.keccak,
            AccelGroup::KeccakRnd => counts.keccak_rnd,
            AccelGroup::Ecsm => counts.ecsm,
            AccelGroup::Ecdas => counts.ecdas,
            AccelGroup::Hint => counts.hint,
        }
    }

    /// The same field, to be written by a pass that counts the tables it found.
    pub(crate) fn count_slot(self, counts: &mut crate::TableCounts) -> &mut usize {
        match self {
            AccelGroup::Commit => &mut counts.commit,
            AccelGroup::Keccak => &mut counts.keccak,
            AccelGroup::KeccakRnd => &mut counts.keccak_rnd,
            AccelGroup::Ecsm => &mut counts.ecsm,
            AccelGroup::Ecdas => &mut counts.ecdas,
            AccelGroup::Hint => &mut counts.hint,
        }
    }

    /// This group's AIRs. Read-only, so every group can be borrowed at once.
    pub(crate) fn airs(self, airs: &crate::VmAirs) -> &[crate::VmAir] {
        match self {
            AccelGroup::Commit => &airs.commits,
            AccelGroup::Keccak => &airs.keccaks,
            AccelGroup::KeccakRnd => &airs.keccak_rnds,
            AccelGroup::Ecsm => &airs.ecsms,
            AccelGroup::Ecdas => &airs.ecdases,
            AccelGroup::Hint => &airs.hints,
        }
    }

    /// This group's tables on the all-at-once path, where they are named fields
    /// rather than an ordered array.
    pub(crate) fn traces(
        self,
        traces: &Traces,
    ) -> &[TraceTable<GoldilocksField, GoldilocksExtension>] {
        match self {
            AccelGroup::Commit => &traces.commits,
            AccelGroup::Keccak => &traces.keccaks,
            AccelGroup::KeccakRnd => &traces.keccak_rnds,
            AccelGroup::Ecsm => &traces.ecsms,
            AccelGroup::Ecdas => &traces.ecdases,
            AccelGroup::Hint => &traces.hints,
        }
    }
}

/// How many AIRs each accelerator group has, in [`ACCEL_ORDER`].
///
/// An accelerator chip is proved as one table or not at all, so a count above
/// one is a malformed layout rather than a bigger one. [`AirOrder::new`]
/// refuses it instead of laying it out.
pub(crate) fn accel_lengths(counts: &crate::TableCounts) -> [usize; 6] {
    ACCEL_ORDER.map(|g| g.count(counts))
}

/// Where each table sits in `VmAirs::air_trace_pairs`.
///
/// The order is the protocol — the transcript absorbs roots in it, and each
/// table's own fork is domain-separated by its index — so every pass has to
/// agree on it. A pass that walks the execution produces chunks in the order
/// they close, which is not this order, so it needs to be able to ask.
///
/// Knowable before the second walk because the first one already counted the
/// chunks.
pub(crate) struct AirOrder {
    counts: crate::TableCounts,
    include_halt: bool,
    num_pages: usize,
}

impl AirOrder {
    /// Refuses an accelerator count above one rather than laying it out.
    ///
    /// Each accelerator chip is proved as a single table, so two is not a
    /// larger layout but a malformed one, and every chunked index below it
    /// would be shifted by the difference. The counts reach here from a proof
    /// on the verifying side, so this is the boundary where they stop being
    /// trusted.
    pub(crate) fn new(
        counts: crate::TableCounts,
        include_halt: bool,
        num_pages: usize,
    ) -> Result<Self, crate::Error> {
        for group in ACCEL_ORDER {
            let n = group.count(&counts);
            if n > 1 {
                return Err(crate::Error::Prover(format!(
                    "AIR order: {} has {n} tables; an accelerator chip has at most one",
                    group.name()
                )));
            }
        }
        Ok(Self {
            counts,
            include_halt,
            num_pages,
        })
    }

    /// The index of the first chunked table, after the fixed ones, HALT and the
    /// accelerator groups.
    fn first_chunked(&self) -> usize {
        NUM_FIXED_AIRS
            + usize::from(self.include_halt)
            + accel_lengths(&self.counts).iter().sum::<usize>()
    }

    fn group_len(&self, group: Option<TableKind>) -> usize {
        match group {
            None => self.num_pages,
            Some(kind) => crate::challenge_phase::count_for(&self.counts, kind),
        }
    }

    /// The AIR index of a chunk, or `None` when the layout has no such chunk.
    pub(crate) fn index_of(&self, kind: TableKind, chunk: usize) -> Option<usize> {
        let mut idx = self.first_chunked();
        for group in GROUP_ORDER {
            let len = self.group_len(group);
            if group == Some(kind) {
                return (chunk < len).then_some(idx + chunk);
            }
            idx += len;
        }
        None
    }

    /// The AIR index of the `i`th PAGE table.
    pub(crate) fn page_index(&self, i: usize) -> Option<usize> {
        let mut idx = self.first_chunked();
        for group in GROUP_ORDER {
            if group.is_none() {
                return (i < self.num_pages).then_some(idx + i);
            }
            idx += self.group_len(group);
        }
        None
    }

    pub fn counts(&self) -> &crate::TableCounts {
        &self.counts
    }

    /// How many tables the layout has in total.
    pub(crate) fn len(&self) -> usize {
        self.first_chunked()
            + GROUP_ORDER
                .iter()
                .map(|g| self.group_len(*g))
                .sum::<usize>()
    }
}

pub(crate) struct StreamingProvider {
    routed: CollectedOps,
    max_rows: MaxRowsConfig,
    /// AIR index -> the chunk that rebuilds it, or `None` when it is resident.
    slots: Vec<Option<(TableKind, usize)>>,
    /// Memoized `(rows, main_columns)` per retired AIR index. Only the shape is
    /// cached — caching the trace would give back the memory this mode exists
    /// to save.
    shapes: Mutex<HashMap<usize, (usize, usize)>>,
}

impl StreamingProvider {
    /// Walk the AIR order and record which index each retired chunk answers to.
    ///
    /// `include_halt` is not a detail: HALT sits between the fixed tables and
    /// the chunked groups and is emitted only for a final epoch, so getting it
    /// wrong shifts every slot by one and hands each table the trace of its
    /// neighbour. It is taken from the caller's `VmAirs` rather than assumed.
    ///
    /// The accelerator groups shift the slots the same way and for the same
    /// reason — a chip the run never reached has no table at all — so they are
    /// read off the traces here rather than counted as a constant.
    pub(crate) fn new(
        routed: CollectedOps,
        max_rows: MaxRowsConfig,
        traces: &Traces,
        include_halt: bool,
    ) -> Self {
        // From `ACCEL_ORDER`, not a second list; all resident, none retired.
        let accel = ACCEL_ORDER.map(|g| g.traces(traces).len());
        let group_lengths = [
            traces.cpus.len(),
            traces.lts.len(),
            traces.shifts.len(),
            traces.memws.len(),
            traces.memw_aligneds.len(),
            traces.loads.len(),
            traces.muls.len(),
            traces.dvrms.len(),
            traces.branches.len(),
            traces.pages.len(),
            traces.memw_registers.len(),
            traces.eqs.len(),
            traces.bytewises.len(),
            traces.stores.len(),
            traces.cpu32s.len(),
        ];

        let mut slots =
            vec![None; NUM_FIXED_AIRS + usize::from(include_halt) + accel.iter().sum::<usize>()];
        for (kind, len) in GROUP_ORDER.iter().zip(group_lengths.iter()) {
            for chunk in 0..*len {
                slots.push(kind.map(|k| (k, chunk)));
            }
        }

        Self {
            routed,
            max_rows,
            slots,
            shapes: Mutex::new(HashMap::new()),
        }
    }

    fn slot(&self, idx: usize) -> Option<(TableKind, usize)> {
        self.slots.get(idx).copied().flatten()
    }

    /// Rows and width of a retired chunk, without building it.
    ///
    /// `chunk_shape` derives both from the op counts and the table's constant
    /// width, so the pre-pass and the memory estimates no longer pay a full
    /// trace generation each just to learn a row count. Still memoized: for the
    /// deduplicating tables it is a counting pass, not free.
    fn shape(&self, idx: usize) -> (usize, usize) {
        if let Some(hit) = self.shapes.lock().unwrap().get(&idx) {
            return *hit;
        }
        let (kind, chunk) = self.slot(idx).expect("shape asked for a resident table");
        let shape = self.routed.chunk_shape(kind, chunk, &self.max_rows);
        #[cfg(feature = "instruments")]
        stark::instruments::count_retired_shape_query();
        self.shapes.lock().unwrap().insert(idx, shape);
        shape
    }
}

impl TraceProvider<GoldilocksField, GoldilocksExtension> for StreamingProvider {
    fn is_retired(&self, idx: usize) -> bool {
        self.slot(idx).is_some()
    }

    fn num_rows(&self, idx: usize) -> usize {
        self.shape(idx).0
    }

    fn num_main_columns(&self, idx: usize) -> usize {
        self.shape(idx).1
    }

    fn build_main(&self, idx: usize) -> TraceTable<GoldilocksField, GoldilocksExtension> {
        let (kind, chunk) = self.slot(idx).expect("build asked for a resident table");
        #[cfg(feature = "instruments")]
        stark::instruments::count_retired_trace_build();
        self.routed.build_chunk(kind, chunk, &self.max_rows)
    }
}

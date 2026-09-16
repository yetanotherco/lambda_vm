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

/// Number of singleton tables emitted before the chunked groups: BITWISE,
/// DECODE, COMMIT, KECCAK, KECCAK_RND, KECCAK_RC, ECSM, ECDAS, HINT, REGISTER.
pub(crate) const NUM_FIXED_AIRS: usize = 10;

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
    pub(crate) fn new(counts: crate::TableCounts, include_halt: bool, num_pages: usize) -> Self {
        Self {
            counts,
            include_halt,
            num_pages,
        }
    }

    /// The index of the first chunked table, after the fixed ones and HALT.
    fn first_chunked(&self) -> usize {
        NUM_FIXED_AIRS + usize::from(self.include_halt)
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

    pub(crate) fn counts(&self) -> &crate::TableCounts {
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
    pub(crate) fn new(
        routed: CollectedOps,
        max_rows: MaxRowsConfig,
        traces: &Traces,
        include_halt: bool,
    ) -> Self {
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

        let mut slots = vec![None; NUM_FIXED_AIRS + usize::from(include_halt)];
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

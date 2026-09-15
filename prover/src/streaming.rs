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
use crate::tables::trace_builder::{RoutedOps, TableKind, Traces};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

/// The groups of chunked tables, in the order `VmAirs::air_trace_pairs` emits
/// them. `None` marks a group that stays resident (PAGE), which still consumes
/// AIR indices and so must be walked over.
const GROUP_ORDER: [Option<TableKind>; 15] = [
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
const NUM_FIXED_AIRS: usize = 10;

pub(crate) struct StreamingProvider {
    routed: RoutedOps,
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
        routed: RoutedOps,
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

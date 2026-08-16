use crate::domain::{Domain, DomainConstants};
use crate::table::Table;
use math::field::traits::{IsField, IsSubFieldOf};
use math::field::{element::FieldElement, traits::IsFFTField};
use math::polynomial::barycentric_inv_denoms;
#[cfg(feature = "disk-spill")]
use math::spill_safe::SpillSafe;
#[cfg(feature = "parallel")]
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSliceMut,
};
#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

/// A two-dimensional representation of an execution trace of the STARK
/// protocol.
///
/// For the moment it is mostly a wrapper around the `Table` struct. It is a
/// layer above the raw two-dimensional table, with functionality relevant to the
/// STARK protocol, such as the step size (number of consecutive rows of the table)
/// of the computation being proven.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct TraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E> + IsField,
{
    pub main_table: Table<F>,
    pub aux_table: Table<E>,
    pub num_main_columns: usize,
    pub num_aux_columns: usize,
    pub step_size: usize,
    /// LogUp aux columns built resident on device (pre-LDE), threaded from the
    /// R1 aux build to the R1 aux commit so they feed the aux LDE without a host
    /// round-trip. None on the CPU / download path.
    #[cfg(feature = "cuda")]
    pub(crate) aux_resident: Option<math_cuda::logup::ResidentAux>,
    /// Whether the GPU-resident aux build is allowed (false under disk-spill,
    /// which needs the aux columns in the host trace to spill them).
    #[cfg(feature = "cuda")]
    pub(crate) resident_aux_ok: bool,
    /// Trace-domain main columns kept resident on device from the R1 main LDE
    /// (column-major `[col*rows + row]`), so the R1 LogUp aux fingerprint kernel
    /// reads them in place instead of re-uploading ~3 GB. None when the GPU main
    /// LDE did not run for this table.
    #[cfg(feature = "cuda")]
    pub(crate) main_trace_dev: Option<ResidentMainTrace>,
}

/// Device-resident trace-domain main columns (column-major `[col*rows + row]`),
/// retained from the R1 main LDE for the aux fingerprint kernel. GPU-only and
/// transient; the device buffer is excluded from logical trace equality (only
/// `rows` participates) and opaque in `Debug`, matching `ResidentAux`.
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub(crate) struct ResidentMainTrace {
    pub(crate) buf: std::sync::Arc<math_cuda::CudaSlice<u64>>,
    pub(crate) rows: usize,
}

#[cfg(feature = "cuda")]
impl core::fmt::Debug for ResidentMainTrace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ResidentMainTrace")
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
impl PartialEq for ResidentMainTrace {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
    }
}

#[cfg(feature = "cuda")]
impl Eq for ResidentMainTrace {}

impl<F, E> TraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E> + IsFFTField,
{
    /// Creates a new TraceTable from from a one-dimensional array in row major order and the intended width of the table.
    /// Step size is how many are needed to represent a state of the VM
    pub fn new_main(
        main_data: Vec<FieldElement<F>>,
        num_main_columns: usize,
        step_size: usize,
    ) -> Self {
        let num_aux_columns = 0;
        let main_table = Table::new(main_data, num_main_columns);
        let aux_table = Table::new(Vec::new(), num_aux_columns);

        Self {
            main_table,
            aux_table,
            num_main_columns,
            num_aux_columns,
            step_size,
            #[cfg(feature = "cuda")]
            aux_resident: None,
            #[cfg(feature = "cuda")]
            resident_aux_ok: true,
            #[cfg(feature = "cuda")]
            main_trace_dev: None,
        }
    }

    /// Creates a new TraceTable from its colummns
    /// Step size is how many are needed to represent a state of the VM
    pub fn from_columns(
        main_columns: Vec<Vec<FieldElement<F>>>,
        aux_columns: Vec<Vec<FieldElement<E>>>,
        step_size: usize,
    ) -> Self {
        let num_main_columns = main_columns.len();
        let num_aux_columns = aux_columns.len();

        let main_table = Table::from_columns(main_columns);
        let aux_table = Table::from_columns(aux_columns);

        Self {
            main_table,
            aux_table,
            num_main_columns,
            num_aux_columns,
            step_size,
            #[cfg(feature = "cuda")]
            aux_resident: None,
            #[cfg(feature = "cuda")]
            resident_aux_ok: true,
            #[cfg(feature = "cuda")]
            main_trace_dev: None,
        }
    }

    pub fn from_columns_main(columns: Vec<Vec<FieldElement<F>>>, step_size: usize) -> Self {
        let num_main_columns = columns.len();
        let num_aux_columns = 0;
        let main_table = Table::from_columns(columns);
        let aux_table = Table::from_columns(Vec::new());

        Self {
            main_table,
            aux_table,
            num_main_columns,
            num_aux_columns,
            step_size,
            #[cfg(feature = "cuda")]
            aux_resident: None,
            #[cfg(feature = "cuda")]
            resident_aux_ok: true,
            #[cfg(feature = "cuda")]
            main_trace_dev: None,
        }
    }

    pub fn num_rows(&self) -> usize {
        self.main_table.height
    }

    /// Store the resident (pre-LDE) LogUp aux columns, threaded to the aux commit.
    #[cfg(feature = "cuda")]
    pub fn set_aux_resident(&mut self, ra: math_cuda::logup::ResidentAux) {
        self.aux_resident = Some(ra);
    }

    /// Borrow the resident aux columns (read by the aux commit for the LDE).
    #[cfg(feature = "cuda")]
    pub fn aux_resident(&self) -> Option<&math_cuda::logup::ResidentAux> {
        self.aux_resident.as_ref()
    }

    /// Whether the GPU-resident aux build is allowed (false under disk-spill).
    #[cfg(feature = "cuda")]
    pub fn resident_aux_ok(&self) -> bool {
        self.resident_aux_ok
    }

    /// Disable the GPU-resident aux build (host trace needed, e.g. disk-spill).
    #[cfg(feature = "cuda")]
    pub fn set_resident_aux_ok(&mut self, ok: bool) {
        self.resident_aux_ok = ok;
    }

    /// Stash the device-resident trace-domain main columns from the R1 main LDE
    /// (column-major `[col*rows + row]`) so the aux fingerprint kernel reads them
    /// in place.
    #[cfg(feature = "cuda")]
    pub fn set_main_trace_dev(
        &mut self,
        buf: std::sync::Arc<math_cuda::CudaSlice<u64>>,
        rows: usize,
    ) {
        self.main_trace_dev = Some(ResidentMainTrace { buf, rows });
    }

    /// The device-resident main trace `(buffer, rows)`, if retained by R1.
    #[cfg(feature = "cuda")]
    pub fn main_trace_dev(&self) -> Option<(&math_cuda::CudaSlice<u64>, usize)> {
        self.main_trace_dev
            .as_ref()
            .map(|r| (r.buf.as_ref(), r.rows))
    }

    /// Drop the retained device-resident main trace. Its only consumer is the
    /// aux build, so the prover clears it right after that pass to reclaim the
    /// snapshot's VRAM before the aux-commit + DEEP/FRI peak.
    #[cfg(feature = "cuda")]
    pub fn clear_main_trace_dev(&mut self) {
        self.main_trace_dev = None;
    }

    pub fn num_steps(&self) -> usize {
        debug_assert!(self.main_table.height.is_multiple_of(self.step_size));
        self.main_table.height / self.step_size
    }

    /// Given a particular step of the computation represented on the trace,
    /// returns the row of the underlying table.
    pub fn step_to_row(&self, step: usize) -> usize {
        self.step_size * step
    }

    pub fn num_cols(&self) -> usize {
        self.main_table.width + self.aux_table.width
    }

    pub fn columns_main(&self) -> Vec<Vec<FieldElement<F>>> {
        self.main_table.columns()
    }

    pub fn columns_aux(&self) -> Vec<Vec<FieldElement<E>>> {
        self.aux_table.columns()
    }

    /// Given a row and a column index, gives stored value in that position
    pub fn get_main(&self, row: usize, col: usize) -> &FieldElement<F> {
        self.main_table.get(row, col)
    }

    /// Given a row and a column index, gives stored value in that position
    pub fn get_aux(&self, row: usize, col: usize) -> &FieldElement<E> {
        self.aux_table.get(row, col)
    }

    pub fn set_main(&mut self, row: usize, col: usize, value: FieldElement<F>) {
        self.main_table.set(row, col, value);
    }

    pub fn set_aux(&mut self, row: usize, col: usize, value: FieldElement<E>) {
        self.aux_table.set(row, col, value);
    }

    /// Allocates the auxiliary table with zeros.
    /// Single allocation - efficient for large tables.
    pub fn allocate_aux_table(&mut self, num_aux_columns: usize) {
        let num_rows = self.num_rows();
        let aux_data = vec![FieldElement::<E>::zero(); num_rows * num_aux_columns];
        self.aux_table = Table::new(aux_data, num_aux_columns);
        self.num_aux_columns = num_aux_columns;
    }

    /// Write main trace data to a temp file and free the in-memory vector.
    /// Accessors read from the mmap after this call.
    #[cfg(feature = "disk-spill")]
    pub fn spill_main_to_disk(&mut self) -> std::io::Result<()>
    where
        F: Copy + 'static,
        F::BaseType: SpillSafe,
    {
        self.main_table.spill_to_disk()
    }

    #[cfg(feature = "disk-spill")]
    pub fn spill_aux_to_disk(&mut self) -> std::io::Result<()>
    where
        E: Copy + 'static,
        E::BaseType: SpillSafe,
    {
        self.aux_table.spill_to_disk()
    }

    /// Extract main columns as owned vectors, each allocated at `capacity`.
    /// Pass the LDE size so downstream FFT expansion is in-place.
    pub fn extract_columns_main(&self, capacity: usize) -> Vec<Vec<FieldElement<F>>> {
        self.main_table.extract_columns(capacity)
    }

    /// Extract auxiliary columns as owned vectors, each allocated at `capacity`.
    pub fn extract_columns_aux(&self, capacity: usize) -> Vec<Vec<FieldElement<E>>> {
        self.aux_table.extract_columns(capacity)
    }

    /// Borrow the row-major main-trace buffer + its width. The trace `Table` is
    /// already stored row-major, so this is zero-copy — it feeds the batched
    /// row-major LDE without the col→row transpose `extract_columns_main` pays.
    pub fn main_data_row_major(&self) -> (&[FieldElement<F>], usize) {
        (self.main_table.row_major_data(), self.main_table.width)
    }

    /// Row-major aux-trace buffer + its width (empty / width 0 when no aux).
    pub fn aux_data_row_major(&self) -> (&[FieldElement<E>], usize) {
        (self.aux_table.row_major_data(), self.aux_table.width)
    }
}
/// Row-major LDE trace table.
///
/// Stores LDE evaluations in flat row-major buffers (`num_rows * num_cols`), so
/// each row is a contiguous slice. This is the layout the batched row-major FFT
/// (`coset_lde_full_expand_row_major`) produces directly and that the Merkle
/// commit consumes without gathering across columns — the win behind the
/// row-major LDE rework (batched twiddle reuse in the FFT + contiguous leaves).
pub struct LDETraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E> + IsField,
{
    /// Row-major main-trace buffer of length `num_rows * num_main_cols`.
    pub(crate) main_data: Vec<FieldElement<F>>,
    /// Row-major auxiliary-trace buffer of length `num_rows * num_aux_cols`.
    /// A `Table` rather than a bare `Vec` so it can carry a disk-spilled
    /// (mmap-backed) buffer — see the aux-LDE spill in the prover's aux stage.
    pub(crate) aux_data: Table<E>,
    pub(crate) num_main_cols: usize,
    pub(crate) num_aux_cols: usize,
    pub(crate) num_rows: usize,
    pub(crate) lde_step_size: usize,
    pub(crate) blowup_factor: usize,
    /// Full-residency (Stage 3): when true the round-1 D2H was intentionally
    /// skipped and at least one of `main_data`/`aux_data` is empty — those
    /// columns are read off the device instead. Set by `build_round1` when the
    /// device-only gate kept this table's round-1 LDE on the GPU, and cleared
    /// again by `set_host_data` once a downgrade has downloaded the resident
    /// LDEs back into the host buffers.
    ///
    /// The R4 and host-evaluator guards hard-abort on this flag rather than
    /// index an empty buffer, so a mis-gate or an unexpected GPU fallback
    /// fails loudly instead of producing a wrong proof. The R3 barycentric
    /// arms instead check the individual buffer they are about to read: mixed
    /// states (one side host-backed, the other device-only) are valid, and the
    /// populated side stays readable.
    #[cfg(feature = "cuda")]
    pub(crate) host_trace_empty: bool,
    /// Per table GPU residency session: owns this table's device LDE buffers
    /// and bound stream. Threaded R1 to R4. Empty on the CPU path.
    #[cfg(feature = "cuda")]
    pub(crate) gpu_session: GpuTableSession,
}

/// Per table GPU residency session.
///
/// Owns the device buffers for one trace table: the main and aux trace LDE
/// (resident R1 to R4), the composition parts LDE (R2 to R4), and a bound
/// stream. The R4 local inv_denoms and FRI state stay local to R4.
#[cfg(feature = "cuda")]
pub(crate) struct GpuTableSession {
    /// Main trace LDE, resident from the R1 fused pipeline through R4. None
    /// when the GPU LDE did not run (below threshold, preprocessed main, not
    /// Goldilocks, or a GPU error).
    main_lde: Option<math_cuda::lde::GpuLdeBase>,
    /// Aux trace LDE (ext3, deinterleaved on device), resident R1 to R4.
    aux_lde: Option<math_cuda::lde::GpuLdeExt3>,
    /// Composition parts LDE (ext3, deinterleaved on device), produced in R2
    /// and resident R2 to R4 so R4 DEEP reads them on device. None when the R2
    /// GPU path did not run.
    composition_parts: Option<math_cuda::lde::GpuLdeExt3>,
    /// Stream bound to this table's GPU work, acquired lazily from the backend
    /// pool and cached. None is cached when the backend is unavailable.
    stream: OnceLock<Option<Arc<math_cuda::CudaStream>>>,
}

#[cfg(feature = "cuda")]
impl GpuTableSession {
    fn new() -> Self {
        Self {
            main_lde: None,
            aux_lde: None,
            composition_parts: None,
            stream: OnceLock::new(),
        }
    }
}

impl<F, E> LDETraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E>,
{
    /// Build a row-major LDETraceTable by consuming column vectors and
    /// transposing them once into the flat buffers. The transpose is the only
    /// O(N · M) data shuffle the table sees — every subsequent row access is a
    /// contiguous slice. Used by the preprocessed / column-input path; the
    /// batched-LDE fast path uses [`Self::from_row_major`] (no transpose).
    pub fn from_columns(
        main_columns: Vec<Vec<FieldElement<F>>>,
        aux_columns: Vec<Vec<FieldElement<E>>>,
        trace_step_size: usize,
        blowup_factor: usize,
    ) -> Self
    where
        FieldElement<F>: Send + Sync,
        FieldElement<E>: Send + Sync,
        Vec<FieldElement<F>>: Sync,
        Vec<FieldElement<E>>: Sync,
    {
        let lde_step_size = trace_step_size * blowup_factor;
        let num_main_cols = main_columns.len();
        let num_aux_cols = aux_columns.len();
        let num_rows = if num_main_cols > 0 {
            main_columns[0].len()
        } else if num_aux_cols > 0 {
            aux_columns[0].len()
        } else {
            0
        };

        // Parallel col-major → row-major transpose: each row chunk gathers from
        // the source columns independently.
        let mut main_data: Vec<FieldElement<F>> =
            vec![FieldElement::<F>::zero(); num_rows * num_main_cols];
        if num_main_cols > 0 {
            #[cfg(feature = "parallel")]
            {
                main_data
                    .par_chunks_exact_mut(num_main_cols)
                    .enumerate()
                    .for_each(|(row, dst)| {
                        for (col, src_col) in main_columns.iter().enumerate() {
                            dst[col] = src_col[row].clone();
                        }
                    });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for (row, dst) in main_data.chunks_exact_mut(num_main_cols).enumerate() {
                    for (col, src_col) in main_columns.iter().enumerate() {
                        dst[col] = src_col[row].clone();
                    }
                }
            }
        }

        let mut aux_data: Vec<FieldElement<E>> =
            vec![FieldElement::<E>::zero(); num_rows * num_aux_cols];
        if num_aux_cols > 0 {
            #[cfg(feature = "parallel")]
            {
                aux_data
                    .par_chunks_exact_mut(num_aux_cols)
                    .enumerate()
                    .for_each(|(row, dst)| {
                        for (col, src_col) in aux_columns.iter().enumerate() {
                            dst[col] = src_col[row].clone();
                        }
                    });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for (row, dst) in aux_data.chunks_exact_mut(num_aux_cols).enumerate() {
                    for (col, src_col) in aux_columns.iter().enumerate() {
                        dst[col] = src_col[row].clone();
                    }
                }
            }
        }

        Self {
            main_data,
            aux_data: Table::from_row_major(aux_data, num_aux_cols),
            num_main_cols,
            num_aux_cols,
            num_rows,
            lde_step_size,
            blowup_factor,
            #[cfg(feature = "cuda")]
            host_trace_empty: false,
            #[cfg(feature = "cuda")]
            gpu_session: GpuTableSession::new(),
        }
    }

    /// Build an LDETraceTable directly from row-major flat buffers. Skips the
    /// O(N·M) col→row transpose that `from_columns` pays — the caller produces
    /// the buffers row-major already (e.g. via `coset_lde_full_expand_row_major`).
    ///
    /// `aux_data` arrives as a `Table` so an already disk-spilled aux LDE keeps
    /// its mmap backing instead of being pulled back onto the heap; its `width`
    /// is the aux column count.
    pub fn from_row_major(
        main_data: Vec<FieldElement<F>>,
        num_main_cols: usize,
        aux_data: Table<E>,
        trace_step_size: usize,
        blowup_factor: usize,
    ) -> Self {
        let lde_step_size = trace_step_size * blowup_factor;
        let num_aux_cols = aux_data.width;
        let num_rows = if num_main_cols > 0 {
            debug_assert_eq!(main_data.len() % num_main_cols, 0);
            main_data.len() / num_main_cols
        } else if num_aux_cols > 0 {
            aux_data.height
        } else {
            0
        };

        Self {
            main_data,
            aux_data,
            num_main_cols,
            num_aux_cols,
            num_rows,
            lde_step_size,
            blowup_factor,
            #[cfg(feature = "cuda")]
            host_trace_empty: false,
            #[cfg(feature = "cuda")]
            gpu_session: GpuTableSession::new(),
        }
    }

    /// Attach the device LDE handle for the main columns, produced by the GPU
    /// fused pipeline. Leave unset on the CPU path.
    #[cfg(feature = "cuda")]
    pub fn set_gpu_main(&mut self, h: math_cuda::lde::GpuLdeBase) {
        self.gpu_session.main_lde = Some(h);
    }

    /// Attach an already-populated device LDE handle for the aux columns.
    #[cfg(feature = "cuda")]
    pub fn set_gpu_aux(&mut self, h: math_cuda::lde::GpuLdeExt3) {
        self.gpu_session.aux_lde = Some(h);
    }

    /// Mark this table's host LDE trace as intentionally empty (Stage-3
    /// device-only path): the round-1 D2H was skipped, so the R4 and
    /// host-evaluator reads hard-abort on the flag instead of indexing the
    /// empty buffers, while the R3 arms consult the individual buffer. Cleared
    /// by [`Self::set_host_data`] once a downgrade has downloaded the resident
    /// LDEs back to the host.
    #[cfg(feature = "cuda")]
    pub fn set_host_trace_empty(&mut self, empty: bool) {
        self.host_trace_empty = empty;
    }

    /// Override the LDE row count. Needed on the device-only path: the host
    /// buffers are empty, so `from_row_major` cannot infer `num_rows` from
    /// `main_data.len()` — the caller supplies it from the device handle's
    /// `lde_size` instead.
    #[cfg(feature = "cuda")]
    pub fn set_num_rows(&mut self, num_rows: usize) {
        self.num_rows = num_rows;
    }

    /// Install downloaded host buffers on a device-only table and clear the
    /// flag: from here every host read is valid again. An empty Vec keeps
    /// that side's existing buffer (either the side has no columns or it
    /// already held a host copy in a mixed state). Only meaningful from
    /// [`crate::gpu_lde::materialize_lde_trace_host`], which guarantees the
    /// buffers match the device handles' layout.
    #[cfg(feature = "cuda")]
    pub(crate) fn set_host_data(
        &mut self,
        main_data: Vec<FieldElement<F>>,
        aux_data: Vec<FieldElement<E>>,
    ) {
        if !main_data.is_empty() {
            self.main_data = main_data;
        }
        if !aux_data.is_empty() {
            self.aux_data = Table::from_row_major(aux_data, self.num_aux_cols);
        }
        self.host_trace_empty = false;
    }

    /// Whether the host LDE trace was intentionally left empty (see
    /// [`Self::set_host_trace_empty`]). The R4 and host-evaluator fallbacks
    /// check this before touching `main_data`/`aux_data`; the R3 barycentric
    /// arms check the individual buffer instead, since a mixed state leaves
    /// one side readable. False again once a downgrade has repopulated the
    /// buffers through [`Self::set_host_data`].
    #[cfg(feature = "cuda")]
    pub fn host_trace_empty(&self) -> bool {
        self.host_trace_empty
    }

    #[cfg(feature = "cuda")]
    pub fn gpu_main(&self) -> Option<&math_cuda::lde::GpuLdeBase> {
        self.gpu_session.main_lde.as_ref()
    }

    #[cfg(feature = "cuda")]
    pub fn gpu_aux(&self) -> Option<&math_cuda::lde::GpuLdeExt3> {
        self.gpu_session.aux_lde.as_ref()
    }

    /// Attach the composition parts LDE produced in R2. Read by R4 DEEP so the
    /// parts are not re-uploaded.
    #[cfg(feature = "cuda")]
    pub fn set_gpu_composition_parts(&mut self, h: math_cuda::lde::GpuLdeExt3) {
        self.gpu_session.composition_parts = Some(h);
    }

    #[cfg(feature = "cuda")]
    pub fn gpu_composition_parts(&self) -> Option<&math_cuda::lde::GpuLdeExt3> {
        self.gpu_session.composition_parts.as_ref()
    }

    /// The stream bound to this table's GPU work. Acquired lazily from the
    /// backend pool on first call and cached, so all of a table's stream ops
    /// share one queue. Returns None (cached) when the backend is unavailable.
    #[cfg(feature = "cuda")]
    pub fn bound_stream(&self) -> Option<Arc<math_cuda::CudaStream>> {
        self.gpu_session
            .stream
            .get_or_init(|| math_cuda::device::backend().ok().map(|b| b.next_stream()))
            .clone()
    }

    pub fn num_main_cols(&self) -> usize {
        self.num_main_cols
    }

    pub fn num_aux_cols(&self) -> usize {
        self.num_aux_cols
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get a single main-trace element by (row, col).
    #[inline]
    pub fn get_main(&self, row: usize, col: usize) -> &FieldElement<F> {
        &self.main_data[row * self.num_main_cols + col]
    }

    /// Get a single aux-trace element by (row, col).
    #[inline]
    pub fn get_aux(&self, row: usize, col: usize) -> &FieldElement<E> {
        self.aux_data.get(row, col)
    }

    /// Borrow a full main-trace row as a contiguous slice (row-major buffer).
    #[inline]
    pub fn main_row(&self, row: usize) -> &[FieldElement<F>] {
        &self.main_data[row * self.num_main_cols..(row + 1) * self.num_main_cols]
    }

    /// Borrow a full aux-trace row as a contiguous slice (row-major buffer).
    #[inline]
    pub fn aux_row(&self, row: usize) -> &[FieldElement<E>] {
        self.aux_data.get_row(row)
    }

    /// Gather a full main-trace row into an owned Vec.
    /// Used by `open_trace_polys` (called ~30 times per table, allocation is negligible).
    pub fn gather_main_row(&self, row_idx: usize) -> Vec<FieldElement<F>> {
        (0..self.num_main_cols())
            .map(|col| self.get_main(row_idx, col).clone())
            .collect()
    }

    /// Gather a range of main-trace columns for a given row.
    /// Used by `open_trace_polys_with_columns` for preprocessed table openings.
    pub fn gather_main_row_range(
        &self,
        row_idx: usize,
        col_start: usize,
        col_end: usize,
    ) -> Vec<FieldElement<F>> {
        (col_start..col_end)
            .map(|col| self.get_main(row_idx, col).clone())
            .collect()
    }

    /// Gather a full aux-trace row into an owned Vec.
    pub fn gather_aux_row(&self, row_idx: usize) -> Vec<FieldElement<E>> {
        (0..self.num_aux_cols())
            .map(|col| self.get_aux(row_idx, col).clone())
            .collect()
    }

    pub fn num_steps(&self) -> usize {
        let height = self.num_rows();
        debug_assert!(height.is_multiple_of(self.lde_step_size));
        height / self.lde_step_size
    }

    pub fn step_to_row(&self, step: usize) -> usize {
        self.lde_step_size * step
    }
}

/// Evaluates trace polynomials at OOD points using barycentric interpolation
/// on the LDE evaluations, without needing coefficient-form polynomials.
///
/// This replaces `get_trace_evaluations` for use in Round 3 of the STARK prover.
/// Instead of evaluating coefficient-form polynomials via Horner's method, it uses
/// the LDE evaluations (already computed in Round 1) and performs barycentric
/// interpolation. This decouples Round 3 from `trace_polys`.
///
/// The key insight: the LDE contains evaluations on a coset of size N*blowup_factor.
/// Taking every blowup_factor-th point gives N evaluations on the trace-size coset
/// {g * w_trace^i}, which is sufficient to interpolate a degree < N polynomial.
///
/// Accepts a [`DomainConstants`] to avoid redundant computation when the caller
/// has already derived these values (e.g., round_3 shares them with composition
/// poly evaluation).
pub fn get_trace_evaluations_from_lde<F, E>(
    lde_trace: &LDETraceTable<F, E>,
    domain: &Domain<F>,
    z: &FieldElement<E>,
    frame_offsets: &[usize],
    step_size: usize,
    dc: &DomainConstants<F>,
) -> Table<E>
where
    F: IsSubFieldOf<E> + IsFFTField + 'static,
    E: IsField + 'static,
{
    let n = domain.interpolation_domain_size;
    let bf = domain.blowup_factor;
    let num_main_cols = lde_trace.num_main_cols();
    let num_aux_cols = lde_trace.num_aux_cols();
    let table_width = num_main_cols + num_aux_cols;

    debug_assert_eq!(
        dc.points.len(),
        n,
        "DomainConstants.points length must equal domain.interpolation_domain_size"
    );

    // Build evaluation points: for each frame offset and step within, z * w_trace^exponent
    let evaluation_points =
        compute_frame_evaluation_points(z, frame_offsets, &domain.trace_primitive_root, step_size);

    // Precompute size_inv * offset_pow_n_inv once — shared across all eval points and columns.
    let n_inv_g_n_inv: FieldElement<F> = &dc.size_inv * &dc.offset_pow_n_inv;

    let mut table_data = Vec::with_capacity(evaluation_points.len() * table_width);

    // GPU fast path for R3 OOD: bundle the inverted inv_denoms (all
    // eval points in one buffer) and the trace-size coset_points upload
    // into a single device context. The barycentric kernels below read
    // both via offset, with no per-eval-point or per-{main,aux} H2D.
    #[cfg(feature = "cuda")]
    let r3_ctx: Option<crate::gpu_lde::R3DevContext> =
        crate::gpu_lde::try_prep_r3_dev_context::<F, E>(
            &dc.points,
            &evaluation_points,
            lde_trace.bound_stream(),
        );
    #[allow(unused_variables)]
    #[cfg(not(feature = "cuda"))]
    let r3_ctx: Option<()> = None;

    #[cfg_attr(not(feature = "cuda"), allow(clippy::unused_enumerate_index))]
    for (eval_point_idx, eval_point) in evaluation_points.iter().enumerate() {
        // Silence unused warning under non-cuda where eval_point_idx is
        // only read inside the cuda-only block below.
        #[cfg(not(feature = "cuda"))]
        let _ = eval_point_idx;
        // z_pow_n for this evaluation point
        let z_pow_n = eval_point.pow(n);

        // vanishing_factor = (z^N - offset^N) * size_inv * offset_pow_n_inv
        let vanishing = z_pow_n.sub_subfield(&dc.offset_pow_n);
        let vanishing_factor = &n_inv_g_n_inv * &vanishing;

        // CPU inv_denoms = 1/(eval_point - coset_point_i). Materialised
        // eagerly only when the GPU dispatcher will need to H2D it (no
        // device-side inv_denoms buffer available). On the all-GPU happy
        // path it stays None and the `barycentric_inv_denoms` call is
        // skipped entirely (the GPU buffer covers every eval point).
        #[cfg(feature = "cuda")]
        let mut inv_denoms: Option<Vec<FieldElement<E>>> = if r3_ctx.is_some() {
            None
        } else {
            Some(barycentric_inv_denoms(eval_point, &dc.points))
        };
        #[cfg(not(feature = "cuda"))]
        let mut inv_denoms: Option<Vec<FieldElement<E>>> =
            Some(barycentric_inv_denoms(eval_point, &dc.points));

        // col_scale[i] = point[i] * inv_denom[i], shared across ALL CPU column
        // loops below. Computed lazily on first CPU-fallback use so the all-GPU
        // path pays nothing, while the all-CPU and mixed paths only pay once.
        let mut col_scale: Option<Vec<FieldElement<E>>> = None;

        // GPU fast path: batched strided barycentric over the main-trace LDE
        // already on device. Returns `None` when the GPU R1 path didn't run
        // for this table (handle absent), the size is below threshold, types
        // don't match, or the math-cuda call errored. Caller falls through
        // to the existing rayon CPU loop.
        // Per-eval-point block offset into the GPU inv_denoms buffer:
        // block k starts at u64 index k * 3 * n.
        #[cfg(feature = "cuda")]
        let r3_arg = r3_ctx.as_ref().map(|ctx| (ctx, eval_point_idx * 3 * n));
        #[cfg(feature = "cuda")]
        let main_gpu = crate::gpu_lde::try_barycentric_base_on_handle::<F, E>(
            lde_trace,
            bf,
            &dc.points,
            &dc.offset_pow_n,
            &dc.size_inv,
            &dc.offset_pow_n_inv,
            &z_pow_n,
            inv_denoms.as_deref().unwrap_or(&[]),
            r3_arg,
        );
        #[cfg(not(feature = "cuda"))]
        let main_gpu: Option<Vec<FieldElement<E>>> = None;

        let main_evals: Vec<FieldElement<E>> = if let Some(v) = main_gpu {
            v
        } else {
            // Device-only tables have no host trace; a GPU fall-through here would
            // read empty `main_data`. Hard-abort instead of a wrong OOD eval. The
            // check is on the buffer itself, not the table-wide flag: a mixed
            // state can leave a valid host copy on one side only.
            #[cfg(feature = "cuda")]
            assert!(
                lde_trace.num_main_cols() == 0 || !lde_trace.main_data.is_empty(),
                "R3 barycentric (main) fell back to the host trace, but it is device-only (empty)"
            );
            let inv_denoms_v =
                inv_denoms.get_or_insert_with(|| barycentric_inv_denoms(eval_point, &dc.points));
            let col_scale = col_scale.get_or_insert_with(|| {
                dc.points
                    .iter()
                    .zip(inv_denoms_v.iter())
                    .map(|(point, inv_d)| point * inv_d)
                    .collect()
            });
            // Evaluate all main columns directly from LDE (no extraction copy).
            // For main columns (base field F): sum = sum over i of col_scale[i] * lde_col[i*bf].
            // lde_col[i*bf] is F, col_scale[i] is E; use F*E -> E mixed arithmetic.
            #[cfg(feature = "parallel")]
            let main_iter = (0..num_main_cols).into_par_iter();
            #[cfg(not(feature = "parallel"))]
            let main_iter = 0..num_main_cols;
            main_iter
                .map(|col_idx| {
                    let sum = col_scale
                        .iter()
                        .enumerate()
                        .fold(FieldElement::<E>::zero(), |acc, (i, scale)| {
                            acc + lde_trace.get_main(i * bf, col_idx) * scale
                        });
                    &vanishing_factor * &sum
                })
                .collect()
        };
        table_data.extend(main_evals);

        // GPU fast path for aux columns reading the de-interleaved ext3 LDE handle.
        #[cfg(feature = "cuda")]
        let r3_arg_aux = r3_ctx.as_ref().map(|ctx| (ctx, eval_point_idx * 3 * n));
        #[cfg(feature = "cuda")]
        let aux_gpu = crate::gpu_lde::try_barycentric_ext3_on_handle::<F, E>(
            lde_trace,
            bf,
            &dc.points,
            &dc.offset_pow_n,
            &dc.size_inv,
            &dc.offset_pow_n_inv,
            &z_pow_n,
            inv_denoms.as_deref().unwrap_or(&[]),
            r3_arg_aux,
        );
        #[cfg(not(feature = "cuda"))]
        let aux_gpu: Option<Vec<FieldElement<E>>> = None;

        let aux_evals: Vec<FieldElement<E>> = if let Some(v) = aux_gpu {
            v
        } else {
            // Device-only tables have no host trace; a GPU fall-through here would
            // read empty `aux_data`. Hard-abort instead of a wrong OOD eval. Same
            // buffer-level check as the main arm: mixed states are valid here.
            #[cfg(feature = "cuda")]
            assert!(
                lde_trace.num_aux_cols() == 0 || !lde_trace.aux_data.row_major_data().is_empty(),
                "R3 barycentric (aux) fell back to the host trace, but it is device-only (empty)"
            );
            let inv_denoms_v =
                inv_denoms.get_or_insert_with(|| barycentric_inv_denoms(eval_point, &dc.points));
            let col_scale = col_scale.get_or_insert_with(|| {
                dc.points
                    .iter()
                    .zip(inv_denoms_v.iter())
                    .map(|(point, inv_d)| point * inv_d)
                    .collect()
            });
            // Evaluate all aux columns directly from LDE (no extraction copy).
            // For aux columns (extension field E): sum = sum over i of col_scale[i] * lde_col[i*bf].
            // Both col_scale and lde_col are in E, so each multiply is E*E -> E.
            #[cfg(feature = "parallel")]
            let aux_iter = (0..num_aux_cols).into_par_iter();
            #[cfg(not(feature = "parallel"))]
            let aux_iter = 0..num_aux_cols;
            aux_iter
                .map(|col_idx| {
                    let sum = col_scale
                        .iter()
                        .enumerate()
                        .fold(FieldElement::<E>::zero(), |acc, (i, scale)| {
                            acc + scale * lde_trace.get_aux(i * bf, col_idx)
                        });
                    &vanishing_factor * &sum
                })
                .collect()
        };
        table_data.extend(aux_evals);
    }

    Table::new(table_data, table_width)
}

pub(crate) fn compute_frame_evaluation_points<F, E>(
    x: &FieldElement<E>,
    frame_offsets: &[usize],
    primitive_root: &FieldElement<F>,
    step_size: usize,
) -> Vec<FieldElement<E>>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let mut evaluation_points = Vec::with_capacity(frame_offsets.len() * step_size);
    for &offset in frame_offsets {
        let start_exponent = offset * step_size;
        let mut current = primitive_root.pow(start_exponent) * x;
        for _ in 0..step_size {
            evaluation_points.push(current.clone());
            current = primitive_root * &current;
        }
    }
    evaluation_points
}

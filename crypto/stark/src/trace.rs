#[cfg(all(feature = "disk-spill", feature = "wasm"))]
compile_error!("disk-spill and wasm features are mutually exclusive");

use crate::domain::Domain;
use crate::table::Table;
use itertools::Itertools;
use math::fft::errors::FFTError;
use math::field::traits::{IsField, IsSubFieldOf};
use math::polynomial::{
    barycentric_inv_denoms, interpolate_coset_eval_ext_with_g_n_inv,
    interpolate_coset_eval_with_g_n_inv,
};
use math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

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
}

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
        }
    }

    pub fn num_rows(&self) -> usize {
        self.main_table.height
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
    pub fn spill_main_to_disk(&mut self) -> std::io::Result<()> {
        self.main_table.spill_to_disk()
    }

    #[cfg(feature = "disk-spill")]
    pub fn spill_aux_to_disk(&mut self) -> std::io::Result<()> {
        self.aux_table.spill_to_disk()
    }

    pub fn compute_trace_polys_main<S>(&self) -> Vec<Polynomial<FieldElement<F>>>
    where
        S: IsFFTField + IsSubFieldOf<F>,
        F: Send + Sync,
        FieldElement<F>: Send + Sync,
    {
        let columns = self.columns_main();
        #[cfg(feature = "parallel")]
        let iter = columns.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = columns.iter();

        iter.map(|col| Polynomial::interpolate_fft::<S>(col))
            .collect::<Result<Vec<Polynomial<FieldElement<F>>>, FFTError>>()
            .unwrap()
    }

    /// Extract main columns directly into pre-allocated output buffers.
    ///
    /// Eliminates the T1 transpose allocation that `columns_main()` performs.
    /// When `output` buffers have sufficient capacity, no heap allocation occurs.
    pub fn extract_columns_main_into(&self, output: &mut [Vec<FieldElement<F>>]) {
        self.main_table.extract_columns_into(output);
    }

    /// Extract auxiliary columns directly into pre-allocated output buffers.
    ///
    /// Eliminates the T1 transpose allocation that `columns_aux()` performs.
    /// When `output` buffers have sufficient capacity, no heap allocation occurs.
    pub fn extract_columns_aux_into(&self, output: &mut [Vec<FieldElement<E>>]) {
        self.aux_table.extract_columns_into(output);
    }
}
/// Column-major LDE trace table.
///
/// Stores LDE evaluations as separate column vectors rather than a row-major Table.
/// This eliminates the expensive T2 transpose (col→row) that `Table::from_columns`
/// performs, significantly reducing allocation and element clones.
///
/// Trade-off: row access requires gathering from columns (74 random reads per row),
/// but this is negligible vs constraint evaluation cost. Column access (used by
/// `get_main`/`get_aux`, barycentric eval, DEEP poly) is sequential and cache-friendly.
pub struct LDETraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E> + IsField,
{
    pub(crate) main_columns: Vec<Vec<FieldElement<F>>>,
    pub(crate) aux_columns: Vec<Vec<FieldElement<E>>>,
    pub(crate) lde_step_size: usize,
    pub(crate) blowup_factor: usize,
    /// When `disk-spill` is enabled and data has been spilled to disk,
    /// this holds the mmap backing. Access methods read from here instead
    /// of `main_columns`/`aux_columns` (which are empty after spill).
    #[cfg(feature = "disk-spill")]
    pub(crate) mmap_backing: Option<MmapBacking>,
    /// Pending async spill submissions. Populated between submit and
    /// `resolve_pending_spills()`; must be `None` before reads.
    #[cfg(feature = "disk-spill")]
    pub(crate) pending_spill: Option<PendingSpill>,
}

/// File-backed mmap storage for LDE column data (column-major layout).
/// Main and aux columns are in separate files since they are spilled
/// at different times (Phase A and Phase C).
#[cfg(feature = "disk-spill")]
pub(crate) struct MmapBacking {
    main_mmap: memmap2::Mmap,
    aux_mmap: Option<memmap2::Mmap>,
    num_rows: usize,
    num_main_cols: usize,
    num_aux_cols: usize,
    main_elem_size: usize,
    aux_elem_size: usize,
}

/// Handles for in-flight spill writes plus geometry needed to construct
/// the final `MmapBacking` once the writes complete.
#[cfg(feature = "disk-spill")]
pub(crate) struct PendingSpill {
    main_handle: Option<crate::spill_worker::SpillHandle>,
    aux_handle: Option<crate::spill_worker::SpillHandle>,
    num_rows: usize,
    num_main_cols: usize,
    num_aux_cols: usize,
    main_elem_size: usize,
    aux_elem_size: usize,
}

impl<F, E> LDETraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E>,
{
    /// Creates a column-major LDETraceTable by consuming column vectors directly.
    /// No transpose is performed — columns are stored as-is.
    pub fn from_columns(
        main_columns: Vec<Vec<FieldElement<F>>>,
        aux_columns: Vec<Vec<FieldElement<E>>>,
        trace_step_size: usize,
        blowup_factor: usize,
    ) -> Self {
        let lde_step_size = trace_step_size * blowup_factor;

        Self {
            main_columns,
            aux_columns,
            lde_step_size,
            blowup_factor,
            #[cfg(feature = "disk-spill")]
            mmap_backing: None,
            #[cfg(feature = "disk-spill")]
            pending_spill: None,
        }
    }

    /// Consume self and return the owned column vectors.
    /// When mmap-backed (disk-spill), returns empty Vecs since columns were freed.
    #[allow(clippy::type_complexity)]
    pub fn into_columns(self) -> (Vec<Vec<FieldElement<F>>>, Vec<Vec<FieldElement<E>>>) {
        (self.main_columns, self.aux_columns)
    }

    pub fn num_main_cols(&self) -> usize {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            return backing.num_main_cols;
        }
        #[cfg(feature = "disk-spill")]
        if let Some(ref pending) = self.pending_spill {
            return pending.num_main_cols;
        }
        self.main_columns.len()
    }

    pub fn num_aux_cols(&self) -> usize {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            return backing.num_aux_cols;
        }
        #[cfg(feature = "disk-spill")]
        if let Some(ref pending) = self.pending_spill {
            return pending.num_aux_cols;
        }
        self.aux_columns.len()
    }

    pub fn num_rows(&self) -> usize {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            return backing.num_rows;
        }
        #[cfg(feature = "disk-spill")]
        if let Some(ref pending) = self.pending_spill {
            return pending.num_rows;
        }
        if self.main_columns.is_empty() {
            0
        } else {
            self.main_columns[0].len()
        }
    }

    /// Get a single main-trace element by (row, col).
    #[inline]
    pub fn get_main(&self, row: usize, col: usize) -> &FieldElement<F> {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            // Guard the unsafe pointer math below; matches the non-spill
            // path's checked indexing so release builds don't drop the check.
            assert!(
                row < backing.num_rows && col < backing.num_main_cols,
                "get_main out of bounds: row={row}, col={col}, num_rows={}, num_main_cols={}",
                backing.num_rows,
                backing.num_main_cols
            );
            let offset = (col * backing.num_rows + row) * backing.main_elem_size;
            // SAFETY: spill_main_from_pool writes columns contiguously to this
            // mmap. FieldElement<F> is #[repr(transparent)] over F::BaseType.
            return unsafe { &*(backing.main_mmap.as_ptr().add(offset) as *const FieldElement<F>) };
        }
        &self.main_columns[col][row]
    }

    /// Get a single aux-trace element by (row, col).
    #[inline]
    pub fn get_aux(&self, row: usize, col: usize) -> &FieldElement<E> {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            // Guard the unsafe pointer math below; matches the non-spill
            // path's checked indexing so release builds don't drop the check.
            assert!(
                row < backing.num_rows && col < backing.num_aux_cols,
                "get_aux out of bounds: row={row}, col={col}, num_rows={}, num_aux_cols={}",
                backing.num_rows,
                backing.num_aux_cols
            );
            let aux_mmap = backing
                .aux_mmap
                .as_ref()
                .expect("aux mmap must exist when accessing aux columns");
            let offset = (col * backing.num_rows + row) * backing.aux_elem_size;
            // SAFETY: add_aux_from_pool writes columns contiguously to this
            // mmap. FieldElement<E> is #[repr(transparent)] over E::BaseType.
            return unsafe { &*(aux_mmap.as_ptr().add(offset) as *const FieldElement<E>) };
        }
        &self.aux_columns[col][row]
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

    /// Copy pool column data into a contiguous `Box<[u8]>` and submit the
    /// write to the async spill worker. Returns an `LDETraceTable` whose
    /// `mmap_backing` is filled in later by `resolve_pending_spills()`.
    /// Pool buffers keep their capacity for reuse.
    #[cfg(feature = "disk-spill")]
    pub fn spill_main_from_pool(
        main_pool: &[Vec<FieldElement<F>>],
        num_main_cols: usize,
        trace_step_size: usize,
        blowup_factor: usize,
    ) -> std::io::Result<Self> {
        let num_rows = if num_main_cols > 0 {
            main_pool[0].len()
        } else {
            0
        };

        let main_elem_size = std::mem::size_of::<FieldElement<F>>();
        let main_bytes =
            Self::copy_pool_columns_to_boxed(&main_pool[..num_main_cols], main_elem_size)?;
        let main_handle = crate::spill_worker::SpillWorker::global().submit(main_bytes);

        let lde_step_size = trace_step_size * blowup_factor;
        let aux_elem_size = std::mem::size_of::<FieldElement<E>>();

        Ok(Self {
            main_columns: Vec::new(),
            aux_columns: Vec::new(),
            lde_step_size,
            blowup_factor,
            mmap_backing: None,
            pending_spill: Some(PendingSpill {
                main_handle: Some(main_handle),
                aux_handle: None,
                num_rows,
                num_main_cols,
                num_aux_cols: 0,
                main_elem_size,
                aux_elem_size,
            }),
        })
    }

    /// Submit the aux LDE write to the async spill worker.
    ///
    /// Used during Phase C to attach aux data to a table whose main LDE was
    /// already submitted in Phase A. The aux handle is resolved alongside
    /// the main handle by `resolve_pending_spills()`.
    #[cfg(feature = "disk-spill")]
    pub fn add_aux_from_pool(
        &mut self,
        aux_pool: &[Vec<FieldElement<E>>],
        num_aux_cols: usize,
    ) -> std::io::Result<()> {
        if num_aux_cols == 0 {
            return Ok(());
        }

        let aux_elem_size = std::mem::size_of::<FieldElement<E>>();
        let aux_bytes = Self::copy_pool_columns_to_boxed(&aux_pool[..num_aux_cols], aux_elem_size)?;
        let aux_handle = crate::spill_worker::SpillWorker::global().submit(aux_bytes);

        let pending = self
            .pending_spill
            .as_mut()
            .expect("add_aux_from_pool requires pending main spill");
        pending.aux_handle = Some(aux_handle);
        pending.num_aux_cols = num_aux_cols;

        Ok(())
    }

    /// Wait on all pending spill writes and populate `mmap_backing`.
    /// After this call `pending_spill` is `None` and reads from
    /// `main_mmap`/`aux_mmap` are safe.
    #[cfg(feature = "disk-spill")]
    pub fn resolve_pending_spills(&mut self) -> std::io::Result<()> {
        let Some(pending) = self.pending_spill.take() else {
            return Ok(());
        };

        let main_mmap = pending
            .main_handle
            .expect("pending spill must have a main handle")
            .wait()?;
        let aux_mmap = pending.aux_handle.map(|h| h.wait()).transpose()?;

        self.mmap_backing = Some(MmapBacking {
            main_mmap,
            aux_mmap,
            num_rows: pending.num_rows,
            num_main_cols: pending.num_main_cols,
            num_aux_cols: pending.num_aux_cols,
            main_elem_size: pending.main_elem_size,
            aux_elem_size: pending.aux_elem_size,
        });

        Ok(())
    }

    /// Copy pool columns into a contiguous `Box<[u8]>` for async spill.
    /// Pool buffers keep their capacity for reuse.
    #[cfg(feature = "disk-spill")]
    fn copy_pool_columns_to_boxed<T>(
        columns: &[Vec<T>],
        elem_size: usize,
    ) -> std::io::Result<Box<[u8]>> {
        debug_assert_eq!(
            elem_size,
            std::mem::size_of::<T>(),
            "elem_size must match size_of::<T>(); the `col.len() * elem_size` byte count below assumes it"
        );

        let num_cols = columns.len();
        let num_rows = if num_cols > 0 { columns[0].len() } else { 0 };
        debug_assert!(
            columns.iter().all(|c| c.len() == num_rows),
            "all columns must have the same length"
        );
        let total_bytes_u64 = (num_cols as u64)
            .checked_mul(num_rows as u64)
            .and_then(|n| n.checked_mul(elem_size as u64))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "copy_pool_columns_to_boxed: byte count overflows u64",
                )
            })?;
        let total_bytes = usize::try_from(total_bytes_u64).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "copy_pool_columns_to_boxed: byte count overflows usize",
            )
        })?;

        let mut buf: Vec<u8> = Vec::with_capacity(total_bytes);
        for col in columns {
            // SAFETY: T is a FieldElement which is #[repr(transparent)],
            // so the Vec has the same byte layout as a contiguous array.
            // `col.len() * elem_size` fits in usize because Vec allocations
            // are bounded by isize::MAX bytes.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(col.as_ptr() as *const u8, col.len() * elem_size)
            };
            buf.extend_from_slice(bytes);
        }
        Ok(buf.into_boxed_slice())
    }
}

/// Given a slice of trace polynomials, an evaluation point `x`, the frame offsets
/// corresponding to the computation of the transitions, and a primitive root,
/// outputs the trace evaluations of each trace polynomial over the values used to
/// compute a transition.
/// Example: For a simple Fibonacci computation, if t(x) is the trace polynomial of
/// the computation, this will output evaluations t(x), t(g * x), t(g^2 * z).
pub fn get_trace_evaluations<F, E>(
    main_trace_polys: &[Polynomial<FieldElement<F>>],
    aux_trace_polys: &[Polynomial<FieldElement<E>>],
    x: &FieldElement<E>,
    frame_offsets: &[usize],
    primitive_root: &FieldElement<F>,
    step_size: usize,
) -> Table<E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let evaluation_points =
        compute_frame_evaluation_points(x, frame_offsets, primitive_root, step_size);

    let main_evaluations = evaluation_points
        .iter()
        .map(|eval_point| {
            main_trace_polys
                .iter()
                .map(|main_poly| main_poly.evaluate(eval_point))
                .collect_vec()
        })
        .collect_vec();

    let aux_evaluations = evaluation_points
        .iter()
        .map(|eval_point| {
            aux_trace_polys
                .iter()
                .map(|aux_poly| aux_poly.evaluate(eval_point))
                .collect_vec()
        })
        .collect_vec();

    debug_assert_eq!(main_evaluations.len(), aux_evaluations.len());
    let mut main_evaluations = main_evaluations;
    let mut table_data = Vec::new();
    for (main_row, aux_row) in main_evaluations.iter_mut().zip(aux_evaluations) {
        main_row.extend_from_slice(&aux_row);
        table_data.extend_from_slice(main_row);
    }

    let main_trace_width = main_trace_polys.len();
    let aux_trace_width = aux_trace_polys.len();
    let table_width = main_trace_width + aux_trace_width;

    Table::new(table_data, table_width)
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
pub fn get_trace_evaluations_from_lde<F, E>(
    lde_trace: &LDETraceTable<F, E>,
    domain: &Domain<F>,
    z: &FieldElement<E>,
    frame_offsets: &[usize],
    step_size: usize,
) -> Table<E>
where
    F: IsSubFieldOf<E> + IsFFTField,
    E: IsField,
{
    let n = domain.interpolation_domain_size;
    let bf = domain.blowup_factor;
    let num_main_cols = lde_trace.num_main_cols();
    let num_aux_cols = lde_trace.num_aux_cols();
    let table_width = num_main_cols + num_aux_cols;

    // Extract trace-size coset points: {g * w_trace^i} = lde_coset[i * blowup_factor]
    let coset_points: Vec<FieldElement<F>> = (0..n)
        .map(|i| domain.lde_roots_of_unity_coset[i * bf].clone())
        .collect();

    // Precompute constants for barycentric formula.
    // Keep coset_offset_pow_n and g_n_inv in base field F — the barycentric
    // functions use F×E→E mixed arithmetic, avoiding field conversions.
    let coset_offset_pow_n: FieldElement<F> = domain.coset_offset.pow(n);
    let n_inv: FieldElement<F> = FieldElement::<F>::from(n as u64)
        .inv()
        .expect("n is a power of two, hence non-zero in the field");
    // Precompute (g^N)^{-1} once in base field — shared across all columns and eval points.
    let g_n_inv: FieldElement<F> = coset_offset_pow_n
        .inv()
        .expect("coset_offset_pow_n is non-zero");

    // Build evaluation points: for each frame offset and step within, z * w_trace^exponent
    let evaluation_points =
        compute_frame_evaluation_points(z, frame_offsets, &domain.trace_primitive_root, step_size);

    // Coset points stay in base field — mixed F×E arithmetic is cheaper than E×E.

    // Extract trace-size evaluations from LDE for each column (stride = blowup_factor)
    #[cfg(feature = "parallel")]
    let main_iter = (0..num_main_cols).into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let main_iter = 0..num_main_cols;
    let main_col_evals: Vec<Vec<FieldElement<F>>> = main_iter
        .map(|col| {
            (0..n)
                .map(|i| lde_trace.get_main(i * bf, col).clone())
                .collect()
        })
        .collect();

    #[cfg(feature = "parallel")]
    let aux_iter = (0..num_aux_cols).into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let aux_iter = 0..num_aux_cols;
    let aux_col_evals: Vec<Vec<FieldElement<E>>> = aux_iter
        .map(|col| {
            (0..n)
                .map(|i| lde_trace.get_aux(i * bf, col).clone())
                .collect()
        })
        .collect();

    let mut table_data = Vec::with_capacity(evaluation_points.len() * table_width);

    for eval_point in &evaluation_points {
        // z_pow_n for this evaluation point
        let z_pow_n = eval_point.pow(n);

        // Precompute inv_denoms = 1/(eval_point - coset_point_i) — shared across all columns
        let inv_denoms = barycentric_inv_denoms(eval_point, &coset_points);

        // Evaluate all main columns (parallel when feature enabled)
        #[cfg(feature = "parallel")]
        let main_iter = main_col_evals.par_iter();
        #[cfg(not(feature = "parallel"))]
        let main_iter = main_col_evals.iter();
        let main_evals: Vec<FieldElement<E>> = main_iter
            .map(|col_evals| {
                interpolate_coset_eval_with_g_n_inv(
                    &z_pow_n,
                    &coset_offset_pow_n,
                    &n_inv,
                    &g_n_inv,
                    &coset_points,
                    col_evals,
                    &inv_denoms,
                )
            })
            .collect();
        table_data.extend(main_evals);

        // Evaluate all aux columns
        #[cfg(feature = "parallel")]
        let aux_iter = aux_col_evals.par_iter();
        #[cfg(not(feature = "parallel"))]
        let aux_iter = aux_col_evals.iter();
        let aux_evals: Vec<FieldElement<E>> = aux_iter
            .map(|col_evals| {
                interpolate_coset_eval_ext_with_g_n_inv(
                    &z_pow_n,
                    &coset_offset_pow_n,
                    &n_inv,
                    &g_n_inv,
                    &coset_points,
                    col_evals,
                    &inv_denoms,
                )
            })
            .collect();
        table_data.extend(aux_evals);
    }

    Table::new(table_data, table_width)
}

#[cfg(all(test, feature = "disk-spill"))]
mod disk_spill_tests {
    use super::*;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;
    type E = Degree3GoldilocksExtensionField;

    /// Spill main LDE columns from a simulated pool, then verify `get_main()`
    /// returns the correct values from the mmap backing.
    #[test]
    fn test_lde_spill_main_roundtrip() {
        let num_cols = 3;
        let num_rows = 16;

        // Simulate pool: column-major Vec<Vec<FE>>
        let pool: Vec<Vec<FieldElement<F>>> = (0..num_cols)
            .map(|c| {
                (0..num_rows)
                    .map(|r| FieldElement::<F>::from((c * num_rows + r) as u64))
                    .collect()
            })
            .collect();

        let mut lde = LDETraceTable::<F, E>::spill_main_from_pool(
            &pool, num_cols, /*trace_step_size=*/ 1, /*blowup_factor=*/ 1,
        )
        .expect("spill_main_from_pool failed");
        lde.resolve_pending_spills()
            .expect("resolve_pending_spills failed");

        assert_eq!(lde.num_main_cols(), num_cols);
        assert_eq!(lde.num_rows(), num_rows);
        assert!(
            lde.main_columns.is_empty(),
            "main_columns should be empty after spill"
        );

        // Verify every element
        for (c, pool_col) in pool.iter().enumerate() {
            for (r, pool_val) in pool_col.iter().enumerate() {
                assert_eq!(
                    lde.get_main(r, c),
                    pool_val,
                    "mismatch at (row={r}, col={c})"
                );
            }
        }
    }

    /// Spill main + aux LDE columns and verify both are accessible.
    #[test]
    fn test_lde_spill_main_and_aux_roundtrip() {
        let num_main = 2;
        let num_aux = 2;
        let num_rows = 8;

        let main_pool: Vec<Vec<FieldElement<F>>> = (0..num_main)
            .map(|c| {
                (0..num_rows)
                    .map(|r| FieldElement::<F>::from((c * num_rows + r) as u64))
                    .collect()
            })
            .collect();

        let aux_pool: Vec<Vec<FieldElement<E>>> = (0..num_aux)
            .map(|c| {
                (0..num_rows)
                    .map(|r| FieldElement::<E>::from((100 + c * num_rows + r) as u64))
                    .collect()
            })
            .collect();

        let mut lde = LDETraceTable::<F, E>::spill_main_from_pool(&main_pool, num_main, 1, 1)
            .expect("spill_main_from_pool failed");

        lde.add_aux_from_pool(&aux_pool, num_aux)
            .expect("add_aux_from_pool failed");
        lde.resolve_pending_spills()
            .expect("resolve_pending_spills failed");

        assert_eq!(lde.num_main_cols(), num_main);
        assert_eq!(lde.num_aux_cols(), num_aux);

        // Verify main
        for (c, main_col) in main_pool.iter().enumerate() {
            for (r, main_val) in main_col.iter().enumerate() {
                assert_eq!(lde.get_main(r, c), main_val);
            }
        }

        // Verify aux
        for (c, aux_col) in aux_pool.iter().enumerate() {
            for (r, aux_val) in aux_col.iter().enumerate() {
                assert_eq!(lde.get_aux(r, c), aux_val);
            }
        }
    }
}

pub fn columns2rows<F>(columns: Vec<Vec<F>>) -> Vec<Vec<F>>
where
    F: Clone,
{
    let num_rows = columns[0].len();
    let num_cols = columns.len();

    (0..num_rows)
        .map(|row_index| {
            (0..num_cols)
                .map(|col_index| columns[col_index][row_index].clone())
                .collect()
        })
        .collect()
}

fn compute_frame_evaluation_points<F, E>(
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

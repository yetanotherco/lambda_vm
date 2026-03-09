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
}

/// File-backed mmap storage for LDE column data.
///
/// Columns are stored in separate files for main and aux (since they may be
/// spilled at different times during Phase A and Phase B of proving).
/// Each file has column-major layout:
/// ```text
/// [col_0][col_1]...[col_N]
/// ```
/// Each column occupies `num_rows * elem_size` contiguous bytes.
/// Elements are stored as their native in-memory representation,
/// which is valid because `FieldElement<F>` is `#[repr(transparent)]`.
#[cfg(feature = "disk-spill")]
pub(crate) struct MmapBacking {
    main_mmap: memmap2::Mmap,
    _main_file: std::fs::File,
    aux_mmap: Option<memmap2::Mmap>,
    _aux_file: Option<std::fs::File>,
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
        self.main_columns.len()
    }

    pub fn num_aux_cols(&self) -> usize {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            return backing.num_aux_cols;
        }
        self.aux_columns.len()
    }

    pub fn num_rows(&self) -> usize {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            return backing.num_rows;
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
            debug_assert!(
                row < backing.num_rows && col < backing.num_main_cols,
                "get_main out of bounds: row={row}, col={col}, num_rows={}, num_main_cols={}",
                backing.num_rows,
                backing.num_main_cols
            );
            let offset = (col * backing.num_rows + row) * backing.main_elem_size;
            // SAFETY: FieldElement<F> is #[repr(transparent)] over F::BaseType.
            // The mmap is page-aligned and elements are contiguously packed at
            // multiples of main_elem_size, so alignment is satisfied.
            // The data was written from identical types on the same machine.
            return unsafe { &*(backing.main_mmap.as_ptr().add(offset) as *const FieldElement<F>) };
        }
        &self.main_columns[col][row]
    }

    /// Get a single aux-trace element by (row, col).
    #[inline]
    pub fn get_aux(&self, row: usize, col: usize) -> &FieldElement<E> {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            debug_assert!(
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
            // SAFETY: Same as get_main — repr(transparent) + page-aligned mmap.
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

    /// Write pool column data to a temp file, mmap it, and return an mmap-backed
    /// LDETraceTable. The pool buffers are NOT consumed — they keep their capacity
    /// for reuse by the next chunk.
    ///
    /// This is used during Phase A to snapshot the main LDE columns from the pool
    /// before the pool is overwritten by the next chunk.
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
        let (main_mmap, main_file) =
            Self::write_pool_columns_to_mmap(&main_pool[..num_main_cols], main_elem_size)?;

        let lde_step_size = trace_step_size * blowup_factor;
        let aux_elem_size = std::mem::size_of::<FieldElement<E>>();

        Ok(Self {
            main_columns: Vec::new(),
            aux_columns: Vec::new(),
            lde_step_size,
            blowup_factor,
            mmap_backing: Some(MmapBacking {
                main_mmap,
                _main_file: main_file,
                aux_mmap: None,
                _aux_file: None,
                num_rows,
                num_main_cols,
                num_aux_cols: 0,
                main_elem_size,
                aux_elem_size,
            }),
        })
    }

    /// Add aux LDE columns from the pool to an already-spilled LDETraceTable.
    ///
    /// Used during Phase B to attach aux data to a table whose main LDE was
    /// already spilled in Phase A.
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
        let (aux_mmap, aux_file) =
            Self::write_pool_columns_to_mmap(&aux_pool[..num_aux_cols], aux_elem_size)?;

        let backing = self
            .mmap_backing
            .as_mut()
            .expect("add_aux_from_pool requires main already spilled");
        backing.aux_mmap = Some(aux_mmap);
        backing._aux_file = Some(aux_file);
        backing.num_aux_cols = num_aux_cols;

        Ok(())
    }

    /// Write borrowed pool columns to a temp file and mmap them.
    /// Does NOT consume the pool — columns keep their capacity.
    ///
    /// Note: the concrete element types are `FieldElement<Goldilocks>` (8 bytes,
    /// `#[repr(transparent)]` over `u64`) and `FieldElement<Degree3Extension>`
    /// (24 bytes, `#[repr(transparent)]` over `[u64; 3]`). Neither has padding,
    /// so the raw byte round-trip is well-defined.
    #[cfg(feature = "disk-spill")]
    fn write_pool_columns_to_mmap<T>(
        columns: &[Vec<T>],
        elem_size: usize,
    ) -> std::io::Result<(memmap2::Mmap, std::fs::File)> {
        use std::io::Write;

        let num_cols = columns.len();
        let num_rows = if num_cols > 0 { columns[0].len() } else { 0 };
        debug_assert!(
            columns.iter().all(|c| c.len() == num_rows),
            "all columns must have the same length"
        );
        let total_bytes = (num_cols * num_rows * elem_size) as u64;

        let file = tempfile::tempfile()?;
        file.set_len(total_bytes)?;
        {
            let mut writer = std::io::BufWriter::new(&file);
            for col in columns {
                // SAFETY: FieldElement<F/E> is #[repr(transparent)] over BaseType,
                // so the Vec has the same byte layout as a contiguous array.
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(col.as_ptr() as *const u8, col.len() * elem_size)
                };
                writer.write_all(bytes)?;
            }
            writer.flush()?;
        }
        // SAFETY: We own the file exclusively.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        Ok((mmap, file))
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
    let evaluation_points = frame_offsets
        .iter()
        .flat_map(|offset| {
            let exponents_range_start = offset * step_size;
            let exponents_range_end = (offset + 1) * step_size;
            (exponents_range_start..exponents_range_end).collect_vec()
        })
        .map(|exponent| primitive_root.pow(exponent) * x)
        .collect_vec();

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
    let evaluation_points: Vec<FieldElement<E>> = frame_offsets
        .iter()
        .flat_map(|offset| {
            let start = offset * step_size;
            let end = (offset + 1) * step_size;
            (start..end).collect_vec()
        })
        .map(|exponent| &domain.trace_primitive_root.pow(exponent) * z)
        .collect_vec();

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

use crate::domain::{Domain, DomainConstants};
use crate::table::Table;
use alloc::vec;
use alloc::vec::Vec;
use itertools::Itertools;
use math::fft::errors::FFTError;
use math::field::traits::{IsField, IsSubFieldOf};
use math::polynomial::barycentric_inv_denoms;
#[cfg(feature = "disk-spill")]
use math::spill_safe::SpillSafe;
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

    /// Extract main columns as owned vectors, each allocated at `capacity`.
    /// Pass the LDE size so downstream FFT expansion is in-place.
    pub fn extract_columns_main(&self, capacity: usize) -> Vec<Vec<FieldElement<F>>> {
        self.main_table.extract_columns(capacity)
    }

    /// Extract auxiliary columns as owned vectors, each allocated at `capacity`.
    pub fn extract_columns_aux(&self, capacity: usize) -> Vec<Vec<FieldElement<E>>> {
        self.aux_table.extract_columns(capacity)
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
        }
    }

    /// Consume self and return the owned column vectors.
    #[allow(clippy::type_complexity)]
    pub fn into_columns(self) -> (Vec<Vec<FieldElement<F>>>, Vec<Vec<FieldElement<E>>>) {
        (self.main_columns, self.aux_columns)
    }

    pub fn num_main_cols(&self) -> usize {
        self.main_columns.len()
    }

    pub fn num_aux_cols(&self) -> usize {
        self.aux_columns.len()
    }

    pub fn num_rows(&self) -> usize {
        if self.main_columns.is_empty() {
            0
        } else {
            self.main_columns[0].len()
        }
    }

    /// Get a single main-trace element by (row, col).
    #[inline]
    pub fn get_main(&self, row: usize, col: usize) -> &FieldElement<F> {
        &self.main_columns[col][row]
    }

    /// Get a single aux-trace element by (row, col).
    #[inline]
    pub fn get_aux(&self, row: usize, col: usize) -> &FieldElement<E> {
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
    F: IsSubFieldOf<E> + IsFFTField,
    E: IsField,
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

    for eval_point in &evaluation_points {
        // z_pow_n for this evaluation point
        let z_pow_n = eval_point.pow(n);

        // vanishing_factor = (z^N - offset^N) * size_inv * offset_pow_n_inv
        let vanishing = z_pow_n.sub_subfield(&dc.offset_pow_n);
        let vanishing_factor = &n_inv_g_n_inv * &vanishing;

        // Precompute inv_denoms = 1/(eval_point - coset_point_i) — shared across all columns
        let inv_denoms = barycentric_inv_denoms(eval_point, &dc.points);

        // Precompute col_scale[i] = point[i] * inv_denom[i] — shared across ALL columns.
        // This eliminates N redundant F×E multiplies per column.
        let col_scale: Vec<FieldElement<E>> = dc
            .points
            .iter()
            .zip(inv_denoms.iter())
            .map(|(point, inv_d)| point * inv_d)
            .collect();

        // Evaluate all main columns directly from LDE (no extraction copy).
        // For main columns (base field F): sum = Σ col_scale[i] * lde_col[i*bf]
        // lde_col[i*bf] is F, col_scale[i] is E; use F×E → E mixed arithmetic.
        #[cfg(feature = "parallel")]
        let main_iter = (0..num_main_cols).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let main_iter = 0..num_main_cols;
        let main_evals: Vec<FieldElement<E>> = main_iter
            .map(|col_idx| {
                let lde_col = &lde_trace.main_columns[col_idx];
                let sum = col_scale
                    .iter()
                    .enumerate()
                    .fold(FieldElement::<E>::zero(), |acc, (i, scale)| {
                        acc + &lde_col[i * bf] * scale
                    });
                &vanishing_factor * &sum
            })
            .collect();
        table_data.extend(main_evals);

        // Evaluate all aux columns directly from LDE (no extraction copy).
        // For aux columns (extension field E): sum = Σ col_scale[i] * lde_col[i*bf]
        // Both col_scale and lde_col are in E, so each multiply is E×E → E.
        #[cfg(feature = "parallel")]
        let aux_iter = (0..num_aux_cols).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let aux_iter = 0..num_aux_cols;
        let aux_evals: Vec<FieldElement<E>> = aux_iter
            .map(|col_idx| {
                let lde_col = &lde_trace.aux_columns[col_idx];
                let sum = col_scale
                    .iter()
                    .enumerate()
                    .fold(FieldElement::<E>::zero(), |acc, (i, scale)| {
                        acc + scale * &lde_col[i * bf]
                    });
                &vanishing_factor * &sum
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

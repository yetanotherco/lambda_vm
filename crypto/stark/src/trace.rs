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
/// Flat row-major LDE trace for the STARK prover.
///
/// Main and auxiliary evaluations are stored in two flat buffers laid out as
/// `[row0_col0, row0_col1, ..., row0_colN, row1_col0, ...]`. This gives the
/// constraint evaluator a single contiguous slice read per frame row instead of
/// one random memory access per column.
pub struct LDETraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E> + IsField,
{
    main_flat: Vec<FieldElement<F>>,
    aux_flat: Vec<FieldElement<E>>,
    num_main: usize,
    num_aux: usize,
    nrows: usize,
    pub(crate) lde_step_size: usize,
    pub(crate) blowup_factor: usize,
}

impl<F, E> LDETraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E>,
{
    /// Build a row-major LDETraceTable by consuming the column vectors from the LDE
    /// step. Columns are scattered into the flat buffer one at a time so each source
    /// column's allocation is freed as soon as its elements are copied.
    pub fn from_columns(
        main_columns: Vec<Vec<FieldElement<F>>>,
        aux_columns: Vec<Vec<FieldElement<E>>>,
        trace_step_size: usize,
        blowup_factor: usize,
    ) -> Self {
        let nrows = main_columns.first().map(|c| c.len()).unwrap_or(0);
        let num_main = main_columns.len();
        let num_aux = aux_columns.len();
        let lde_step_size = trace_step_size * blowup_factor;

        let mut main_flat = vec![FieldElement::<F>::zero(); nrows * num_main];
        for (col_idx, col) in main_columns.into_iter().enumerate() {
            for (row, val) in col.into_iter().enumerate() {
                main_flat[row * num_main + col_idx] = val;
            }
        }

        let mut aux_flat = vec![FieldElement::<E>::zero(); nrows * num_aux];
        for (col_idx, col) in aux_columns.into_iter().enumerate() {
            for (row, val) in col.into_iter().enumerate() {
                aux_flat[row * num_aux + col_idx] = val;
            }
        }

        Self {
            main_flat,
            aux_flat,
            num_main,
            num_aux,
            nrows,
            lde_step_size,
            blowup_factor,
        }
    }

    /// Consume self and return owned column vectors (un-transpose).
    #[allow(clippy::type_complexity)]
    pub fn into_columns(self) -> (Vec<Vec<FieldElement<F>>>, Vec<Vec<FieldElement<E>>>) {
        let mut main_columns = vec![Vec::with_capacity(self.nrows); self.num_main];
        for row in 0..self.nrows {
            let row_slice = &self.main_flat[row * self.num_main..(row + 1) * self.num_main];
            for (col, val) in row_slice.iter().enumerate() {
                main_columns[col].push(val.clone());
            }
        }
        let mut aux_columns = vec![Vec::with_capacity(self.nrows); self.num_aux];
        for row in 0..self.nrows {
            let row_slice = &self.aux_flat[row * self.num_aux..(row + 1) * self.num_aux];
            for (col, val) in row_slice.iter().enumerate() {
                aux_columns[col].push(val.clone());
            }
        }
        (main_columns, aux_columns)
    }

    pub fn num_main_cols(&self) -> usize {
        self.num_main
    }

    pub fn num_aux_cols(&self) -> usize {
        self.num_aux
    }

    pub fn num_rows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn get_main(&self, row: usize, col: usize) -> &FieldElement<F> {
        &self.main_flat[row * self.num_main + col]
    }

    #[inline]
    pub fn get_aux(&self, row: usize, col: usize) -> &FieldElement<E> {
        &self.aux_flat[row * self.num_aux + col]
    }

    /// Borrow a full main-trace row as a contiguous slice.
    #[inline]
    pub(crate) fn main_row(&self, row: usize) -> &[FieldElement<F>] {
        let start = row * self.num_main;
        &self.main_flat[start..start + self.num_main]
    }

    /// Borrow a full aux-trace row as a contiguous slice.
    #[inline]
    pub(crate) fn aux_row(&self, row: usize) -> &[FieldElement<E>] {
        let start = row * self.num_aux;
        &self.aux_flat[start..start + self.num_aux]
    }

    pub fn gather_main_row(&self, row_idx: usize) -> Vec<FieldElement<F>> {
        self.main_row(row_idx).to_vec()
    }

    pub fn gather_main_row_range(
        &self,
        row_idx: usize,
        col_start: usize,
        col_end: usize,
    ) -> Vec<FieldElement<F>> {
        let start = row_idx * self.num_main + col_start;
        self.main_flat[start..start + (col_end - col_start)].to_vec()
    }

    pub fn gather_aux_row(&self, row_idx: usize) -> Vec<FieldElement<E>> {
        self.aux_row(row_idx).to_vec()
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

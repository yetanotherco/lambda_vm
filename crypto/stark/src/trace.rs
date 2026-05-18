use crate::domain::{Domain, DomainConstants};
use crate::table::Table;
use itertools::Itertools;
use math::fft::errors::FFTError;
use math::field::traits::{IsField, IsSubFieldOf};
use math::polynomial::barycentric_inv_denoms;
use math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
/// Row-major LDE trace table.
///
/// Stores LDE evaluations in two contiguous flat buffers, indexed as
/// `main_data[row * num_main_cols + col]` and `aux_data[row * num_aux_cols + col]`.
///
/// Replaces the prior column-major `Vec<Vec<FieldElement<_>>>` layout. The row-
/// major layout makes a single LDE row a contiguous slice, which collapses the
/// per-row scatter-gather in `Frame::fill_from_lde` to a memcpy and matches the
/// access pattern of the constraint hot loop (row-by-row reads). Column access
/// (barycentric / DEEP) still works via reordered loops that iterate rows and
/// accumulate per-column.
pub struct LDETraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E> + IsField,
{
    /// Row-major main-trace buffer of length `num_rows * num_main_cols`.
    pub(crate) main_data: Vec<FieldElement<F>>,
    /// Row-major auxiliary-trace buffer of length `num_rows * num_aux_cols`.
    pub(crate) aux_data: Vec<FieldElement<E>>,
    pub(crate) num_main_cols: usize,
    pub(crate) num_aux_cols: usize,
    pub(crate) num_rows: usize,
    pub(crate) lde_step_size: usize,
    pub(crate) blowup_factor: usize,
}

impl<F, E> LDETraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E>,
{
    /// Build a row-major LDETraceTable by consuming column vectors and
    /// transposing them once into the flat buffers.
    ///
    /// The transpose is the only O(N · M) data shuffle the table sees in its
    /// lifetime — every subsequent row access is a contiguous slice.
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

        // Parallel col-major → row-major transpose. Each row chunk is filled
        // independently by gathering from the source columns. Without
        // parallelism this transpose dominates at large sizes (~100ms at
        // log21 n=16); with `par_chunks_exact_mut` it drops to memory
        // bandwidth time (~5-15ms at the same size).
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
            aux_data,
            num_main_cols,
            num_aux_cols,
            num_rows,
            lde_step_size,
            blowup_factor,
        }
    }

    /// Consume self and re-materialize the column vectors. Inverse of
    /// `from_columns` — pays the same O(N · M) transpose cost.
    #[allow(clippy::type_complexity)]
    pub fn into_columns(self) -> (Vec<Vec<FieldElement<F>>>, Vec<Vec<FieldElement<E>>>) {
        let mut main_columns: Vec<Vec<FieldElement<F>>> = (0..self.num_main_cols)
            .map(|_| Vec::with_capacity(self.num_rows))
            .collect();
        for row in 0..self.num_rows {
            let row_off = row * self.num_main_cols;
            for col in 0..self.num_main_cols {
                main_columns[col].push(self.main_data[row_off + col].clone());
            }
        }

        let mut aux_columns: Vec<Vec<FieldElement<E>>> = (0..self.num_aux_cols)
            .map(|_| Vec::with_capacity(self.num_rows))
            .collect();
        for row in 0..self.num_rows {
            let row_off = row * self.num_aux_cols;
            for col in 0..self.num_aux_cols {
                aux_columns[col].push(self.aux_data[row_off + col].clone());
            }
        }

        (main_columns, aux_columns)
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
        &self.aux_data[row * self.num_aux_cols + col]
    }

    /// Borrow a full main-trace row as a contiguous slice (zero copy).
    #[inline]
    pub fn main_row_slice(&self, row_idx: usize) -> &[FieldElement<F>] {
        let off = row_idx * self.num_main_cols;
        &self.main_data[off..off + self.num_main_cols]
    }

    /// Borrow a full aux-trace row as a contiguous slice (zero copy).
    #[inline]
    pub fn aux_row_slice(&self, row_idx: usize) -> &[FieldElement<E>] {
        let off = row_idx * self.num_aux_cols;
        &self.aux_data[off..off + self.num_aux_cols]
    }

    /// Gather a full main-trace row into an owned Vec (allocates).
    /// Prefer [`Self::main_row_slice`] when zero-copy access is enough.
    pub fn gather_main_row(&self, row_idx: usize) -> Vec<FieldElement<F>> {
        self.main_row_slice(row_idx).to_vec()
    }

    /// Gather a range of main-trace columns for a given row.
    pub fn gather_main_row_range(
        &self,
        row_idx: usize,
        col_start: usize,
        col_end: usize,
    ) -> Vec<FieldElement<F>> {
        let row_off = row_idx * self.num_main_cols;
        self.main_data[row_off + col_start..row_off + col_end].to_vec()
    }

    /// Gather a full aux-trace row into an owned Vec (allocates).
    /// Prefer [`Self::aux_row_slice`] when zero-copy access is enough.
    pub fn gather_aux_row(&self, row_idx: usize) -> Vec<FieldElement<E>> {
        self.aux_row_slice(row_idx).to_vec()
    }

    pub fn num_steps(&self) -> usize {
        let height = self.num_rows;
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
        let col_scale: Vec<FieldElement<E>> = dc
            .points
            .iter()
            .zip(inv_denoms.iter())
            .map(|(point, inv_d)| point * inv_d)
            .collect();

        // Iterate by-row (cache-friendly on row-major storage) and accumulate
        // per-column partial sums. Use rayon fold+reduce so each thread
        // accumulates its own per-column sums over a row chunk and the
        // reduces sum them. Each i-th LDE row is read as one contiguous
        // slice of `num_main_cols` (or `num_aux_cols`) cells (one cache
        // line at typical widths).
        let zero_main = || vec![FieldElement::<E>::zero(); num_main_cols];
        let zero_aux = || vec![FieldElement::<E>::zero(); num_aux_cols];

        let accumulate = |(mut main_sums, mut aux_sums): (
            Vec<FieldElement<E>>,
            Vec<FieldElement<E>>,
        ),
                          i: usize|
         -> (Vec<FieldElement<E>>, Vec<FieldElement<E>>) {
            let lde_row = i * bf;
            let scale_i = &col_scale[i];
            if num_main_cols > 0 {
                let row_slice = lde_trace.main_row_slice(lde_row);
                for col_idx in 0..num_main_cols {
                    main_sums[col_idx] = &main_sums[col_idx] + &row_slice[col_idx] * scale_i;
                }
            }
            if num_aux_cols > 0 {
                let row_slice = lde_trace.aux_row_slice(lde_row);
                for col_idx in 0..num_aux_cols {
                    aux_sums[col_idx] = &aux_sums[col_idx] + scale_i * &row_slice[col_idx];
                }
            }
            (main_sums, aux_sums)
        };

        let combine = |(mut a_main, mut a_aux): (Vec<FieldElement<E>>, Vec<FieldElement<E>>),
                       (b_main, b_aux): (Vec<FieldElement<E>>, Vec<FieldElement<E>>)|
         -> (Vec<FieldElement<E>>, Vec<FieldElement<E>>) {
            for col_idx in 0..num_main_cols {
                a_main[col_idx] = &a_main[col_idx] + &b_main[col_idx];
            }
            for col_idx in 0..num_aux_cols {
                a_aux[col_idx] = &a_aux[col_idx] + &b_aux[col_idx];
            }
            (a_main, a_aux)
        };

        #[cfg(feature = "parallel")]
        let (main_sums, aux_sums) = (0..n)
            .into_par_iter()
            .fold(|| (zero_main(), zero_aux()), accumulate)
            .reduce(|| (zero_main(), zero_aux()), combine);

        #[cfg(not(feature = "parallel"))]
        let (main_sums, aux_sums) = (0..n).fold((zero_main(), zero_aux()), accumulate);

        for sum in &main_sums {
            table_data.push(&vanishing_factor * sum);
        }
        for sum in &aux_sums {
            table_data.push(&vanishing_factor * sum);
        }
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

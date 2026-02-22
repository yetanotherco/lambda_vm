use crate::domain::Domain;
use crate::table::Table;
use itertools::Itertools;
use math::fft::errors::FFTError;
use math::field::traits::{IsField, IsSubFieldOf};
use math::polynomial::{barycentric_inv_denoms, interpolate_coset_eval, interpolate_coset_eval_ext};
use math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

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
    pub fn new(
        main_data: Vec<FieldElement<F>>,
        aux_data: Vec<FieldElement<E>>,
        num_main_columns: usize,
        num_aux_columns: usize,
        step_size: usize,
    ) -> Self {
        let main_table = Table::new(main_data, num_main_columns);
        let aux_table = Table::new(aux_data, num_aux_columns);

        Self {
            main_table,
            aux_table,
            num_main_columns,
            num_aux_columns,
            step_size,
        }
    }

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

    pub fn empty() -> Self {
        Self::new(Vec::new(), Vec::new(), 0, 0, 0)
    }

    pub fn is_empty(&self) -> bool {
        self.main_table.width == 0 && self.aux_table.width == 0
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

    pub fn allocate_with_zeros(
        num_steps: usize,
        num_main_columns: usize,
        num_aux_columns: usize,
        step_size: usize,
    ) -> TraceTable<F, E> {
        let main_data = vec![FieldElement::<F>::zero(); step_size * num_steps * num_main_columns];
        let aux_data = vec![FieldElement::<E>::zero(); step_size * num_steps * num_aux_columns];
        TraceTable::new(
            main_data,
            aux_data,
            num_main_columns,
            num_aux_columns,
            step_size,
        )
    }

    pub fn compute_trace_polys_main<S>(&self) -> Vec<Polynomial<FieldElement<F>>>
    where
        S: IsFFTField + IsSubFieldOf<F>,
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

    pub fn compute_trace_polys_aux<S>(&self) -> Vec<Polynomial<FieldElement<E>>>
    where
        S: IsFFTField + IsSubFieldOf<F>,
        FieldElement<E>: Send + Sync,
    {
        let columns = self.columns_aux();
        #[cfg(feature = "parallel")]
        let iter = columns.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = columns.iter();

        iter.map(|col| Polynomial::interpolate_fft::<F>(col))
            .collect::<Result<Vec<Polynomial<FieldElement<E>>>, FFTError>>()
            .unwrap()
    }

    pub fn get_column_main(&self, col_idx: usize) -> Vec<FieldElement<F>> {
        self.main_table.get_column(col_idx)
    }

    pub fn get_column_aux(&self, col_idx: usize) -> Vec<FieldElement<E>> {
        self.aux_table.get_column(col_idx)
    }
}
pub struct LDETraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E> + IsField,
{
    pub(crate) main_table: Table<F>,
    pub(crate) aux_table: Table<E>,
    pub(crate) lde_step_size: usize,
    pub(crate) blowup_factor: usize,
}

impl<F, E> LDETraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E>,
{
    pub fn new(
        main_data: Vec<FieldElement<F>>,
        aux_data: Vec<FieldElement<E>>,
        n_columns: usize,
        trace_step_size: usize,
        blowup_factor: usize,
    ) -> Self {
        let main_table = Table::new(main_data, n_columns);
        let aux_table = Table::new(aux_data, n_columns);
        let lde_step_size = trace_step_size * blowup_factor;

        Self {
            main_table,
            aux_table,
            lde_step_size,
            blowup_factor,
        }
    }

    pub fn from_columns(
        main_columns: Vec<Vec<FieldElement<F>>>,
        aux_columns: Vec<Vec<FieldElement<E>>>,
        trace_step_size: usize,
        blowup_factor: usize,
    ) -> Self {
        let main_table = Table::from_columns(main_columns);
        let aux_table = Table::from_columns(aux_columns);
        let lde_step_size = trace_step_size * blowup_factor;

        Self {
            main_table,
            aux_table,
            lde_step_size,
            blowup_factor,
        }
    }

    /// Creates an LDETraceTable by borrowing column data without consuming the column Vecs.
    ///
    /// Used for LDE buffer reuse: the pool retains the column buffers for the next table.
    pub fn from_columns_borrowed(
        main_columns: &[Vec<FieldElement<F>>],
        aux_columns: &[Vec<FieldElement<E>>],
        trace_step_size: usize,
        blowup_factor: usize,
    ) -> Self {
        let main_table = Table::from_columns_borrowed(main_columns);
        let aux_table = Table::from_columns_borrowed(aux_columns);
        let lde_step_size = trace_step_size * blowup_factor;

        Self {
            main_table,
            aux_table,
            lde_step_size,
            blowup_factor,
        }
    }

    pub fn num_cols(&self) -> usize {
        self.main_table.width + self.aux_table.width
    }

    pub fn num_rows(&self) -> usize {
        self.main_table.height
    }

    pub fn get_main_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        self.main_table.get_row(row_idx)
    }

    pub fn get_aux_row(&self, row_idx: usize) -> &[FieldElement<E>] {
        self.aux_table.get_row(row_idx)
    }

    pub fn get_main(&self, row: usize, col: usize) -> &FieldElement<F> {
        self.main_table.get(row, col)
    }

    pub fn get_aux(&self, row: usize, col: usize) -> &FieldElement<E> {
        self.aux_table.get(row, col)
    }

    pub fn num_steps(&self) -> usize {
        debug_assert!(self.main_table.height.is_multiple_of(self.lde_step_size));
        self.main_table.height / self.lde_step_size
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
    let num_main_cols = lde_trace.main_table.width;
    let num_aux_cols = lde_trace.aux_table.width;
    let table_width = num_main_cols + num_aux_cols;

    // Extract trace-size coset points: {g * w_trace^i} = lde_coset[i * blowup_factor]
    let coset_points: Vec<FieldElement<F>> = (0..n)
        .map(|i| domain.lde_roots_of_unity_coset[i * bf].clone())
        .collect();

    // Precompute constants for barycentric formula
    let coset_offset_pow_n = domain.coset_offset.pow(n);
    // Lift coset_offset_pow_n to extension field
    let coset_offset_pow_n_ext: FieldElement<E> = coset_offset_pow_n.clone().to_extension();
    let n_inv: FieldElement<E> = FieldElement::<E>::from(n as u64)
        .inv()
        .expect("n is a power of two, hence non-zero in the field");

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
    // Main columns: Vec of N base-field evaluations per column
    let main_col_evals: Vec<Vec<FieldElement<F>>> = (0..num_main_cols)
        .map(|col| {
            (0..n)
                .map(|i| lde_trace.get_main(i * bf, col).clone())
                .collect()
        })
        .collect();

    // Aux columns: Vec of N extension-field evaluations per column
    let aux_col_evals: Vec<Vec<FieldElement<E>>> = (0..num_aux_cols)
        .map(|col| {
            (0..n)
                .map(|i| lde_trace.get_aux(i * bf, col).clone())
                .collect()
        })
        .collect();

    let mut table_data =
        Vec::with_capacity(evaluation_points.len() * table_width);

    for eval_point in &evaluation_points {
        // z_pow_n for this evaluation point
        let z_pow_n = eval_point.pow(n);

        // Precompute inv_denoms = 1/(eval_point - coset_point_i) — shared across all columns
        let inv_denoms = barycentric_inv_denoms(eval_point, &coset_points);

        // Evaluate each main column
        for col_evals in &main_col_evals {
            table_data.push(interpolate_coset_eval(
                &z_pow_n,
                &coset_offset_pow_n_ext,
                &n_inv,
                &coset_points,
                col_evals,
                &inv_denoms,
            ));
        }

        // Evaluate each aux column (coset_points in base field, evals in extension)
        for col_evals in &aux_col_evals {
            table_data.push(interpolate_coset_eval_ext(
                &z_pow_n,
                &coset_offset_pow_n_ext,
                &n_inv,
                &coset_points,
                col_evals,
                &inv_denoms,
            ));
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

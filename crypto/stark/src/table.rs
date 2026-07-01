use crate::frame::Frame;
use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A two-dimensional Table holding field elements, arranged in a row-major order.
/// This is the basic underlying data structure used for any two-dimensional component in the
/// the STARK protocol implementation, such as the `TraceTable` and the `EvaluationFrame`.
/// Since this struct is a representation of a two-dimensional table, all rows should have the same
/// length.
#[derive(Default, Debug, serde::Deserialize, serde::Serialize, Clone, PartialEq, Eq)]
#[serde(bound = "")]
pub struct Table<F: IsField> {
    pub(crate) data: Vec<FieldElement<F>>,
    pub width: usize,
    pub height: usize,
}

impl<F: IsField> Table<F> {
    /// Crates a new Table instance from a one-dimensional array in row major order
    /// and the intended width of the table.
    pub fn new(data: Vec<FieldElement<F>>, width: usize) -> Self {
        // Check if the intented width is 0, used for creating an empty table.
        if width == 0 {
            return Self {
                data: Vec::new(),
                width,
                height: 0,
            };
        }

        // Check that the one-dimensional data makes sense to be interpreted as a 2D one.
        debug_assert!(crate::debug::validate_2d_structure(&data, width));
        let height = data.len() / width;

        Self {
            data,
            width,
            height,
        }
    }

    /// Creates a Table instance from a vector of the intended columns.
    pub fn from_columns(columns: Vec<Vec<FieldElement<F>>>) -> Self {
        if columns.is_empty() {
            return Self::new(Vec::new(), 0);
        }
        let height = columns[0].len();

        // Check that all columns have the same length for integrity
        debug_assert!(columns.iter().all(|c| c.len() == height));

        let width = columns.len();
        let mut data = Vec::with_capacity(width * height);

        for row_idx in 0..height {
            for column in columns.iter() {
                data.push(column[row_idx].clone());
            }
        }

        Self::new(data, width)
    }

    /// Given a row index, returns a reference to that row as a slice of field elements.
    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        let row_offset = row_idx * self.width;
        &self.data[row_offset..row_offset + self.width]
    }

    /// Full row-major data as a contiguous slice.
    pub fn row_major_data(&self) -> &[FieldElement<F>] {
        &self.data
    }

    /// Returns a vector of vectors of field elements representing the table
    /// columns
    pub fn columns(&self) -> Vec<Vec<FieldElement<F>>> {
        (0..self.width)
            .map(|col_idx| {
                (0..self.height)
                    .map(|row_idx| self.get(row_idx, col_idx).clone())
                    .collect()
            })
            .collect()
    }

    /// Extract columns as owned vectors, with each allocated at `capacity`.
    ///
    /// `capacity` is a hint sized for downstream LDE expansion so the FFT grows
    /// in place without a second allocation. Avoids the T1 transpose `columns()`
    /// performs.
    pub fn extract_columns(&self, capacity: usize) -> Vec<Vec<FieldElement<F>>> {
        let capacity = capacity.max(self.height);
        #[cfg(feature = "parallel")]
        let iter = (0..self.width).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = 0..self.width;
        iter.map(|col_idx| {
            let mut buf = Vec::with_capacity(capacity);
            for row_idx in 0..self.height {
                buf.push(self.get(row_idx, col_idx).clone());
            }
            buf
        })
        .collect()
    }

    /// Given row and column indexes, returns the stored field element in that position of the table.
    pub fn get(&self, row: usize, col: usize) -> &FieldElement<F> {
        let idx = row * self.width + col;
        &self.data[idx]
    }

    pub fn set(&mut self, row: usize, col: usize, value: FieldElement<F>) {
        let idx = row * self.width + col;
        self.data[idx] = value;
    }

    /// Given a step size, converts the given table into a `Frame`.
    /// Clones row data into owned Vecs (only used by verifier on small OOD tables).
    pub fn into_frame(&self, main_trace_columns: usize, step_size: usize) -> Frame<F, F> {
        debug_assert!(self.height.is_multiple_of(step_size));
        let steps = (0..self.height)
            .step_by(step_size)
            .map(|initial_row_idx| {
                let end_row_idx = initial_row_idx + step_size;

                let mut step_main_data: Vec<Vec<FieldElement<F>>> = Vec::new();
                let mut step_aux_data: Vec<Vec<FieldElement<F>>> = Vec::new();

                (initial_row_idx..end_row_idx).for_each(|row_idx| {
                    let row = self.get_row(row_idx);
                    step_main_data.push(row[..main_trace_columns].to_vec());
                    step_aux_data.push(row[main_trace_columns..].to_vec());
                });

                TableView::new(step_main_data, step_aux_data)
            })
            .collect();

        Frame::new(steps)
    }
}

/// A view of a contiguous subset of rows of a table.
///
/// Owns its row data (Vec per row) so it can be built from either row-major Tables
/// (verifier path) or column-major LDE data (prover path) without lifetime issues.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableView<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E>,
{
    pub data: Vec<Vec<FieldElement<F>>>,
    pub aux_data: Vec<Vec<FieldElement<E>>>,
}

impl<F, E> TableView<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E>,
{
    pub fn new(data: Vec<Vec<FieldElement<F>>>, aux_data: Vec<Vec<FieldElement<E>>>) -> Self {
        Self { data, aux_data }
    }

    pub fn get_main_evaluation_element(&self, row: usize, col: usize) -> &FieldElement<F> {
        &self.data[row][col]
    }

    pub fn get_aux_evaluation_element(&self, row: usize, col: usize) -> &FieldElement<E> {
        &self.aux_data[row][col]
    }
}

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
#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct Table<F: IsField> {
    pub data: Vec<FieldElement<F>>,
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

    /// Creates a Table instance by borrowing column data without consuming it.
    ///
    /// Same transpose logic as [`from_columns`], but the column Vecs are NOT consumed —
    /// the caller retains them. This is used for LDE buffer reuse where the pool
    /// retains the column buffers for the next table.
    pub fn from_columns_borrowed(columns: &[Vec<FieldElement<F>>]) -> Self {
        if columns.is_empty() {
            return Self::new(Vec::new(), 0);
        }
        let height = columns[0].len();

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

    /// Returns a vector of vectors of field elements representing the table rows
    pub fn rows(&self) -> Vec<Vec<FieldElement<F>>> {
        self.data.chunks(self.width).map(|r| r.to_vec()).collect()
    }

    /// Given a row index, returns a reference to that row as a slice of field elements.
    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        let row_offset = row_idx * self.width;
        &self.data[row_offset..row_offset + self.width]
    }

    /// Given a slice of field elements representing a row, appends it to
    /// the end of the table.
    pub fn append_row(&mut self, row: &[FieldElement<F>]) {
        debug_assert_eq!(row.len(), self.width);
        self.data.extend_from_slice(row);
        self.height += 1
    }

    /// Returns a reference to the last row of the table
    pub fn last_row(&self) -> &[FieldElement<F>] {
        self.get_row(self.height - 1)
    }

    /// Returns a vector of vectors of field elements representing the table
    /// columns
    pub fn columns(&self) -> Vec<Vec<FieldElement<F>>> {
        (0..self.width)
            .map(|col_idx| {
                (0..self.height)
                    .map(|row_idx| self.data[row_idx * self.width + col_idx].clone())
                    .collect()
            })
            .collect()
    }

    pub fn get_column(&self, col_idx: usize) -> Vec<FieldElement<F>> {
        (0..self.height)
            .map(|row_idx| self.data[row_idx * self.width + col_idx].clone())
            .collect()
    }

    /// Extract columns directly into pre-allocated output buffers.
    ///
    /// Each `output[col_idx]` is cleared and filled with the column data.
    /// When `output[col_idx].capacity() >= height`, no heap allocation occurs.
    /// This eliminates the T1 transpose allocation that `columns()` performs.
    pub fn extract_columns_into(&self, output: &mut [Vec<FieldElement<F>>]) {
        debug_assert!(
            output.len() >= self.width,
            "output has {} buffers but table has {} columns",
            output.len(),
            self.width
        );
        #[cfg(feature = "parallel")]
        let iter = output[..self.width].par_iter_mut().enumerate();
        #[cfg(not(feature = "parallel"))]
        let iter = output[..self.width].iter_mut().enumerate();
        iter.for_each(|(col_idx, buf)| {
            buf.clear();
            buf.reserve(self.height.saturating_sub(buf.capacity()));
            for row_idx in 0..self.height {
                buf.push(self.data[row_idx * self.width + col_idx].clone());
            }
        });
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
    F: IsSubFieldOf<F>,
{
    pub data: Vec<Vec<FieldElement<F>>>,
    pub aux_data: Vec<Vec<FieldElement<E>>>,
}

impl<F, E> TableView<F, E>
where
    E: IsField,
    F: IsSubFieldOf<F>,
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

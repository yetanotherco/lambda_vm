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
#[derive(
    Default,
    Debug,
    serde::Deserialize,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct Table<F: IsField> {
    /// Row-major backing store. Crate-private: external callers go through the
    /// accessors (`get`/`get_row`/`set`).
    pub(crate) data: Vec<FieldElement<F>>,
    pub width: usize,
    pub height: usize,
}

/// Read API over an rkyv-archived [`Table`], used by the verifier to consume
/// the out-of-domain evaluations straight from the proof buffer. On
/// little-endian targets the element data is viewed in place with no copy.
#[cfg(target_endian = "little")]
impl<F: IsField> ArchivedTable<F>
where
    F::BaseType: math::field::element::NativeArchived,
{
    #[inline]
    pub fn width(&self) -> usize {
        self.width.to_native() as usize
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height.to_native() as usize
    }

    /// Full row-major element data, viewed in place.
    #[inline]
    pub fn row_major_data(&self) -> &[FieldElement<F>] {
        math::field::element::ArchivedFieldElement::slice_as_native(self.data.as_slice())
    }

    /// `true` iff the backing data holds exactly `width × height` elements —
    /// the invariant `get_row` indexing relies on. A malformed archive can
    /// advertise dimensions that disagree with the data length; callers must
    /// reject such tables before row access.
    #[inline]
    pub fn dimensions_consistent(&self) -> bool {
        self.width()
            .checked_mul(self.height())
            .is_some_and(|n| n == self.data.len())
    }

    /// Row `row_idx` as a native field-element slice (no copy).
    #[inline]
    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        let width = self.width();
        let start = row_idx * width;
        &self.row_major_data()[start..start + width]
    }
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

    /// `true` iff the backing data holds exactly `width × height` elements —
    /// the invariant `get_row` indexing relies on. Owned counterpart to
    /// `ArchivedTable::dimensions_consistent`.
    #[inline]
    pub fn dimensions_consistent(&self) -> bool {
        self.width
            .checked_mul(self.height)
            .is_some_and(|n| n == self.row_major_data().len())
    }

    /// Returns a vector of vectors of field elements representing the table
    /// columns
    pub fn columns(&self) -> Vec<Vec<FieldElement<F>>> {
        // One column per worker: the table is row-major, so this is a strided
        // read of the whole trace and the widest thing between the executor and
        // the commitment.
        let column = |col_idx: usize| -> Vec<FieldElement<F>> {
            (0..self.height)
                .map(|row_idx| self.get(row_idx, col_idx).clone())
                .collect()
        };
        #[cfg(feature = "parallel")]
        return (0..self.width).into_par_iter().map(column).collect();
        #[cfg(not(feature = "parallel"))]
        return (0..self.width).map(column).collect();
    }

    /// [`Self::columns`] by tiles: row blocks in parallel, and within a block
    /// a band of columns at a time, so each row's cache lines are read once
    /// rather than once per column. The same columns, in the same order.
    ///
    /// `columns` reads the row-major table at a stride of one row per element
    /// on every column's worker, which pulls a whole cache line for each value:
    /// on a block's traces that is most of the transposition's time, and its
    /// memory traffic is what the columns' upload then competes with.
    pub fn columns_blocked(&self) -> Vec<Vec<FieldElement<F>>> {
        let (width, height) = (self.width, self.height);
        if width == 0 || height == 0 || self.data.len() != width * height {
            return self.columns();
        }
        const ROWS: usize = 1024;
        const BAND: usize = 64;
        let mut columns: Vec<Vec<FieldElement<F>>> =
            (0..width).map(|_| Vec::with_capacity(height)).collect();
        // Addresses, not pointers, so the closure is `Sync`; each block writes
        // rows `r0..r1` of every column, disjoint from every other block.
        let starts: Vec<usize> = columns
            .iter_mut()
            .map(|column| column.as_mut_ptr() as usize)
            .collect();
        let data = &self.data;
        let block = |b: usize| {
            let r0 = b * ROWS;
            let r1 = (r0 + ROWS).min(height);
            let mut c0 = 0;
            while c0 < width {
                let c1 = (c0 + BAND).min(width);
                for row in r0..r1 {
                    let values = &data[row * width + c0..row * width + c1];
                    for (offset, value) in values.iter().enumerate() {
                        // SAFETY: `starts[c]` is column `c`'s buffer, `height`
                        // elements of capacity; `row < height`, and no other
                        // block writes this row.
                        unsafe {
                            (starts[c0 + offset] as *mut FieldElement<F>)
                                .add(row)
                                .write(value.clone());
                        }
                    }
                }
                c0 = c1;
            }
        };
        let blocks = height.div_ceil(ROWS);
        #[cfg(feature = "parallel")]
        (0..blocks).into_par_iter().for_each(block);
        #[cfg(not(feature = "parallel"))]
        (0..blocks).for_each(block);
        for column in &mut columns {
            // SAFETY: every row of every column was written above.
            unsafe { column.set_len(height) };
        }
        columns
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

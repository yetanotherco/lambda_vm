use crate::frame::Frame;
use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Mmap-backed storage for a spilled Table.
///
/// The table data is written row-major to a temp file and mmapped back.
/// Access goes through pointer arithmetic on the mmap, matching the
/// original `data[row * width + col]` layout.
#[cfg(feature = "disk-spill")]
pub(crate) struct TableMmapBacking {
    mmap: memmap2::Mmap,
    _file: std::fs::File,
    width: usize,
    height: usize,
    elem_size: usize,
}

// Manual trait impls so Table<F> can keep its derive macros.
// Spilled tables should not be cloned during proving.
#[cfg(feature = "disk-spill")]
impl Clone for TableMmapBacking {
    fn clone(&self) -> Self {
        panic!("TableMmapBacking cannot be cloned — spilled tables should not be cloned")
    }
}

#[cfg(feature = "disk-spill")]
impl Default for TableMmapBacking {
    fn default() -> Self {
        panic!("TableMmapBacking has no default — use None")
    }
}

#[cfg(feature = "disk-spill")]
impl std::fmt::Debug for TableMmapBacking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableMmapBacking")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("elem_size", &self.elem_size)
            .finish()
    }
}

#[cfg(feature = "disk-spill")]
impl PartialEq for TableMmapBacking {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.elem_size == other.elem_size
            && self.mmap[..] == other.mmap[..]
    }
}

#[cfg(feature = "disk-spill")]
impl Eq for TableMmapBacking {}

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
    #[cfg(feature = "disk-spill")]
    #[serde(skip)]
    pub(crate) mmap_backing: Option<TableMmapBacking>,
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
                #[cfg(feature = "disk-spill")]
                mmap_backing: None,
            };
        }

        // Check that the one-dimensional data makes sense to be interpreted as a 2D one.
        debug_assert!(crate::debug::validate_2d_structure(&data, width));
        let height = data.len() / width;

        Self {
            data,
            width,
            height,
            #[cfg(feature = "disk-spill")]
            mmap_backing: None,
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
        (0..self.height)
            .map(|row_idx| self.get_row(row_idx).to_vec())
            .collect()
    }

    /// Given a row index, returns a reference to that row as a slice of field elements.
    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            debug_assert!(
                row_idx < backing.height,
                "Table::get_row out of bounds: row={row_idx}, height={}",
                backing.height
            );
            let offset = row_idx * backing.width * backing.elem_size;
            // SAFETY: Row-major layout means width elements are contiguous.
            // Same repr(transparent) + page-aligned guarantees as get().
            return unsafe {
                std::slice::from_raw_parts(
                    backing.mmap.as_ptr().add(offset) as *const FieldElement<F>,
                    backing.width,
                )
            };
        }
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
                    .map(|row_idx| self.get(row_idx, col_idx).clone())
                    .collect()
            })
            .collect()
    }

    pub fn get_column(&self, col_idx: usize) -> Vec<FieldElement<F>> {
        (0..self.height)
            .map(|row_idx| self.get(row_idx, col_idx).clone())
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
        // Use get() which transparently reads from mmap or data Vec
        iter.for_each(|(col_idx, buf)| {
            buf.clear();
            buf.reserve(self.height.saturating_sub(buf.capacity()));
            for row_idx in 0..self.height {
                buf.push(self.get(row_idx, col_idx).clone());
            }
        });
    }

    /// Given row and column indexes, returns the stored field element in that position of the table.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> &FieldElement<F> {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            debug_assert!(
                row < backing.height && col < backing.width,
                "Table::get out of bounds: row={row}, col={col}, height={}, width={}",
                backing.height,
                backing.width
            );
            // Row-major layout: offset = (row * width + col) * elem_size
            let offset = (row * backing.width + col) * backing.elem_size;
            // SAFETY: FieldElement<F> is #[repr(transparent)] over F::BaseType.
            // The mmap is page-aligned and elements are contiguously packed.
            // The data was written from identical types on the same machine.
            return unsafe { &*(backing.mmap.as_ptr().add(offset) as *const FieldElement<F>) };
        }
        let idx = row * self.width + col;
        &self.data[idx]
    }

    pub fn set(&mut self, row: usize, col: usize, value: FieldElement<F>) {
        let idx = row * self.width + col;
        self.data[idx] = value;
    }

    /// Returns true if this table's data has been spilled to disk via mmap.
    pub fn is_spilled(&self) -> bool {
        #[cfg(feature = "disk-spill")]
        {
            self.mmap_backing.is_some()
        }
        #[cfg(not(feature = "disk-spill"))]
        {
            false
        }
    }

    /// Spill the table's row-major data to a temp file and mmap it back.
    /// Frees the heap `data` Vec while preserving access through `get()`,
    /// `get_row()`, `columns()`, and `extract_columns_into()`.
    ///
    /// No-op if the table is empty or already spilled.
    #[cfg(feature = "disk-spill")]
    pub fn spill_to_disk(&mut self) -> std::io::Result<()> {
        use std::io::Write;

        if self.data.is_empty() || self.mmap_backing.is_some() {
            return Ok(());
        }

        let elem_size = std::mem::size_of::<FieldElement<F>>();
        let total_bytes = self.data.len() * elem_size;

        let file = tempfile::tempfile()?;
        file.set_len(total_bytes as u64)?;
        {
            let mut writer = std::io::BufWriter::new(&file);
            // SAFETY: FieldElement<F> is #[repr(transparent)] over F::BaseType.
            // The Vec has the same byte layout as a contiguous array.
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u8, total_bytes) };
            writer.write_all(bytes)?;
            writer.flush()?;
        }

        // SAFETY: We own the file exclusively.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

        self.mmap_backing = Some(TableMmapBacking {
            mmap,
            _file: file,
            width: self.width,
            height: self.height,
            elem_size,
        });

        // Free heap allocation
        self.data = Vec::new();

        Ok(())
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

#[cfg(all(test, feature = "disk-spill"))]
mod disk_spill_tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;

    /// Create a Table, spill it to disk, and verify that `get()` and `get_row()`
    /// return the same values as before the spill.
    #[test]
    fn test_table_spill_roundtrip() {
        let width = 4;
        let height = 8;
        let data: Vec<FieldElement<F>> = (0..width * height)
            .map(|i| FieldElement::<F>::from(i as u64))
            .collect();

        let mut table = Table::new(data.clone(), width);
        assert!(!table.is_spilled());

        // Snapshot values before spill
        let pre_spill: Vec<Vec<FieldElement<F>>> = (0..height)
            .map(|r| (0..width).map(|c| *table.get(r, c)).collect())
            .collect();

        table.spill_to_disk().expect("spill_to_disk failed");
        assert!(table.is_spilled());
        assert!(
            table.data.is_empty(),
            "heap data should be freed after spill"
        );

        // Verify get() returns the same values
        for (r, pre_row) in pre_spill.iter().enumerate() {
            for (c, pre_val) in pre_row.iter().enumerate() {
                assert_eq!(table.get(r, c), pre_val, "mismatch at ({r}, {c})");
            }
        }

        // Verify get_row() returns the same values
        for (r, pre_row) in pre_spill.iter().enumerate() {
            let row = table.get_row(r);
            assert_eq!(row.len(), width);
            for (c, pre_val) in pre_row.iter().enumerate() {
                assert_eq!(&row[c], pre_val, "get_row mismatch at ({r}, {c})");
            }
        }
    }

    /// Spilling an empty table is a no-op.
    #[test]
    fn test_table_spill_empty_is_noop() {
        let mut table = Table::<F>::new(Vec::new(), 0);
        table
            .spill_to_disk()
            .expect("spill_to_disk on empty table failed");
        assert!(!table.is_spilled());
    }

    /// Spilling twice is idempotent (second call is a no-op).
    #[test]
    fn test_table_spill_idempotent() {
        let data: Vec<FieldElement<F>> =
            (0..16).map(|i| FieldElement::<F>::from(i as u64)).collect();
        let mut table = Table::new(data, 4);

        table.spill_to_disk().expect("first spill failed");
        assert!(table.is_spilled());

        table.spill_to_disk().expect("second spill should be no-op");
        assert!(table.is_spilled());

        // Still readable
        assert_eq!(table.get(0, 0), &FieldElement::<F>::from(0u64));
        assert_eq!(table.get(3, 3), &FieldElement::<F>::from(15u64));
    }
}

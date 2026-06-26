use crate::frame::Frame;
#[cfg(feature = "disk-spill")]
use crypto::mmap_util::spill_slice_to_mmap;
use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};
#[cfg(feature = "disk-spill")]
use math::spill_safe::SpillSafe;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Mmap-backed storage for a spilled Table.
///
/// Access goes through pointer arithmetic on the mmap, matching the
/// original `data[row * width + col]` layout.
#[cfg(feature = "disk-spill")]
struct TableMmapBacking {
    mmap: memmap2::Mmap,
    /// Number of columns per row.
    width: usize,
    /// Number of rows.
    height: usize,
    /// Size in bytes of a single element.
    elem_size: usize,
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

/// A two-dimensional Table holding field elements, arranged in a row-major order.
/// This is the basic underlying data structure used for any two-dimensional component in the
/// the STARK protocol implementation, such as the `TraceTable` and the `EvaluationFrame`.
/// Since this struct is a representation of a two-dimensional table, all rows should have the same
/// length.
#[derive(Default, Debug, serde::Deserialize)]
#[cfg_attr(
    not(feature = "disk-spill"),
    derive(serde::Serialize, Clone, PartialEq, Eq)
)]
#[serde(bound = "")]
pub struct Table<F: IsField> {
    /// Row-major backing store. Crate-private: external callers must go through
    /// the spill-safe accessors (`get`/`get_row`/`set`) rather than indexing the
    /// raw buffer, which bypasses the disk-spill mmap backing.
    pub(crate) data: Vec<FieldElement<F>>,
    pub width: usize,
    pub height: usize,
    #[cfg(feature = "disk-spill")]
    #[serde(skip)]
    mmap_backing: Option<TableMmapBacking>,
}

#[cfg(feature = "disk-spill")]
impl<F: IsField> serde::Serialize for Table<F>
where
    FieldElement<F>: serde::Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Table", 3)?;
        if self.mmap_backing.is_some() {
            s.serialize_field("data", &MmapDataSeq(self))?;
        } else {
            s.serialize_field("data", &self.data)?;
        }
        s.serialize_field("width", &self.width)?;
        s.serialize_field("height", &self.height)?;
        s.end()
    }
}

#[cfg(feature = "disk-spill")]
struct MmapDataSeq<'a, F: IsField>(&'a Table<F>);

#[cfg(feature = "disk-spill")]
impl<F: IsField> serde::Serialize for MmapDataSeq<'_, F>
where
    FieldElement<F>: serde::Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let table = self.0;
        let mut seq = serializer.serialize_seq(Some(table.width * table.height))?;
        for r in 0..table.height {
            for elem in table.get_row(r) {
                seq.serialize_element(elem)?;
            }
        }
        seq.end()
    }
}

/// Cloning a spilled table copies its mmap bytes into a fresh heap `Vec`
/// and returns an unspilled clone.
#[cfg(feature = "disk-spill")]
impl<F: IsField> Clone for Table<F> {
    fn clone(&self) -> Self {
        if self.mmap_backing.is_some() {
            let mut data = Vec::with_capacity(self.width * self.height);
            for row in 0..self.height {
                for col in 0..self.width {
                    data.push(self.get(row, col).clone());
                }
            }
            return Self {
                data,
                width: self.width,
                height: self.height,
                mmap_backing: None,
            };
        }
        Self {
            data: self.data.clone(),
            width: self.width,
            height: self.height,
            mmap_backing: None,
        }
    }
}

#[cfg(feature = "disk-spill")]
impl<F: IsField> PartialEq for Table<F> {
    fn eq(&self, other: &Self) -> bool {
        if self.width != other.width || self.height != other.height {
            return false;
        }
        for row in 0..self.height {
            for col in 0..self.width {
                if self.get(row, col) != other.get(row, col) {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(feature = "disk-spill")]
impl<F: IsField> Eq for Table<F> {}

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

    /// Given a row index, returns a reference to that row as a slice of field elements.
    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            // Ensures the unsafe block's read stays within the mmap.
            assert!(
                row_idx < backing.height,
                "Table::get_row out of bounds: row={row_idx}, height={}",
                backing.height
            );
            let offset = row_idx * backing.width * backing.elem_size;
            // SAFETY: spill_to_disk writes the table in row-major layout, so
            // width elements at this offset are contiguous. FieldElement<F>
            // is #[repr(transparent)] over F::BaseType, and spill_to_disk
            // requires F::BaseType: SpillSafe (no padding, all bit patterns
            // valid).
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

    /// Full row-major data as a contiguous slice, reading the mmap when spilled.
    pub fn row_major_data(&self) -> &[FieldElement<F>] {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            // SAFETY: same contract as get_row — spill_to_disk writes row-major and
            // FieldElement<F> is #[repr(transparent)] over F::BaseType: SpillSafe.
            return unsafe {
                std::slice::from_raw_parts(
                    backing.mmap.as_ptr() as *const FieldElement<F>,
                    backing.height * backing.width,
                )
            };
        }
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
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.mmap_backing {
            // Ensures the unsafe block's read stays within the mmap.
            assert!(
                row < backing.height && col < backing.width,
                "Table::get out of bounds: row={row}, col={col}, height={}, width={}",
                backing.height,
                backing.width
            );
            // Row-major layout: offset = (row * width + col) * elem_size
            let offset = (row * backing.width + col) * backing.elem_size;
            // SAFETY: FieldElement<F> is #[repr(transparent)] over F::BaseType.
            // The mmap is page-aligned and elements are contiguously packed.
            // The data was written from identical types on the same machine,
            // and spill_to_disk requires F::BaseType: SpillSafe (no padding,
            // all bit patterns valid).
            return unsafe { &*(backing.mmap.as_ptr().add(offset) as *const FieldElement<F>) };
        }
        let idx = row * self.width + col;
        &self.data[idx]
    }

    pub fn set(&mut self, row: usize, col: usize, value: FieldElement<F>) {
        #[cfg(feature = "disk-spill")]
        assert!(
            self.mmap_backing.is_none(),
            "Table::set on a spilled table — backing mmap is read-only"
        );
        let idx = row * self.width + col;
        self.data[idx] = value;
    }

    /// Spill the table's row-major data to a temp file and mmap it back.
    /// Frees the heap `data` Vec while preserving access through
    /// [`Self::get`], [`Self::get_row`], and [`Self::columns`].
    ///
    /// No-op if the table is empty or already spilled.
    #[cfg(feature = "disk-spill")]
    pub fn spill_to_disk(&mut self) -> std::io::Result<()>
    where
        F: Copy + 'static,
        F::BaseType: SpillSafe,
    {
        if self.data.is_empty() || self.mmap_backing.is_some() {
            return Ok(());
        }

        let mmap = spill_slice_to_mmap(&self.data)?;
        self.mmap_backing = Some(TableMmapBacking {
            mmap,
            width: self.width,
            height: self.height,
            elem_size: size_of::<FieldElement<F>>(),
        });
        self.data = Vec::new();

        Ok(())
    }

    /// Hint the kernel to drop mmap pages from the page cache.
    /// Call after reading spilled data into pool buffers so the same
    /// data doesn't occupy RAM in both places.
    ///
    /// Reliable on Linux for clean file-backed mappings; on other Unix
    /// (macOS/BSD) the hint may be a no-op. No-op on non-Unix targets.
    #[cfg(all(feature = "disk-spill", unix))]
    pub fn advise_drop_cache(&self) {
        if let Some(ref backing) = self.mmap_backing {
            // SAFETY: pointer and length are from a valid mmap.
            // MADV_DONTNEED is advisory and cannot cause UB.
            unsafe {
                libc::madvise(
                    backing.mmap.as_ptr() as *mut libc::c_void,
                    backing.mmap.len(),
                    libc::MADV_DONTNEED,
                );
            }
        }
    }

    #[cfg(all(feature = "disk-spill", not(unix)))]
    pub fn advise_drop_cache(&self) {}

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

#[cfg(all(test, feature = "disk-spill"))]
mod disk_spill_tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;

    #[test]
    fn test_table_spill_roundtrip() {
        let width = 4;
        let height = 8;
        let data: Vec<FieldElement<F>> = (0..width * height)
            .map(|i| FieldElement::<F>::from(i as u64))
            .collect();

        let mut table = Table::new(data.clone(), width);
        assert!(table.mmap_backing.is_none());

        // Snapshot values before spill
        let pre_spill: Vec<Vec<FieldElement<F>>> = (0..height)
            .map(|r| (0..width).map(|c| *table.get(r, c)).collect())
            .collect();

        table.spill_to_disk().expect("spill_to_disk failed");
        assert!(table.mmap_backing.is_some());
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

    /// `row_major_data()` is the accessor `Trace::main_data_row_major()` feeds
    /// to the row-major LDE. After a spill the heap `data` is freed, so it must
    /// read back through the mmap byte-for-byte. Regression guard for the
    /// EmptyCommitment bug (commit 08eaa8b), where a spilled main table handed
    /// the LDE an emptied buffer. `get`/`get_row` were already covered above;
    /// the full row-major read (the one the LDE actually uses) was not.
    #[test]
    fn test_table_spill_row_major_data_roundtrip() {
        let width = 5;
        let height = 8;
        let data: Vec<FieldElement<F>> = (0..width * height)
            .map(|i| FieldElement::<F>::from((i as u64).wrapping_mul(7) + 3))
            .collect();

        let mut table = Table::new(data.clone(), width);
        assert_eq!(
            table.row_major_data(),
            data.as_slice(),
            "row_major_data must match the source before spill"
        );

        table.spill_to_disk().expect("spill_to_disk failed");
        assert!(
            table.data.is_empty(),
            "heap data should be freed after spill"
        );

        assert_eq!(
            table.row_major_data(),
            data.as_slice(),
            "row_major_data must round-trip through the mmap after spill"
        );
    }

    #[test]
    fn test_table_spill_empty_is_noop() {
        let mut table = Table::<F>::new(Vec::new(), 0);
        table
            .spill_to_disk()
            .expect("spill_to_disk on empty table failed");
        assert!(table.mmap_backing.is_none());
    }

    #[test]
    fn test_table_spill_idempotent() {
        let data: Vec<FieldElement<F>> =
            (0..16).map(|i| FieldElement::<F>::from(i as u64)).collect();
        let mut table = Table::new(data, 4);

        table.spill_to_disk().expect("first spill failed");
        assert!(table.mmap_backing.is_some());

        table.spill_to_disk().expect("second spill should be no-op");
        assert!(table.mmap_backing.is_some());

        // Still readable
        assert_eq!(table.get(0, 0), &FieldElement::<F>::from(0u64));
        assert_eq!(table.get(3, 3), &FieldElement::<F>::from(15u64));
    }

    #[test]
    fn test_clone_spilled_table_materializes_to_heap() {
        let width = 4;
        let height = 8;
        let data: Vec<FieldElement<F>> = (0..width * height)
            .map(|i| FieldElement::<F>::from(i as u64))
            .collect();

        let mut table = Table::new(data, width);
        table.spill_to_disk().expect("spill_to_disk failed");
        assert!(table.mmap_backing.is_some());

        let cloned = table.clone();
        assert!(cloned.mmap_backing.is_none(), "clone should not be spilled");
        assert_eq!(cloned.width, width);
        assert_eq!(cloned.height, height);
        assert_eq!(cloned, table, "clone must equal source element-wise");
    }

    #[test]
    fn test_serialize_spilled_table_matches_unspilled() {
        let width = 4;
        let height = 8;
        let data: Vec<FieldElement<F>> = (0..width * height)
            .map(|i| FieldElement::<F>::from(i as u64))
            .collect();

        let unspilled = Table::new(data.clone(), width);
        let unspilled_bytes = bincode::serialize(&unspilled).expect("serialize unspilled");

        let mut spilled = Table::new(data, width);
        spilled.spill_to_disk().expect("spill_to_disk failed");
        let spilled_bytes = bincode::serialize(&spilled).expect("serialize spilled");

        assert_eq!(
            spilled_bytes, unspilled_bytes,
            "spilled and unspilled tables must serialize to identical bytes"
        );

        let restored: Table<F> =
            bincode::deserialize(&spilled_bytes).expect("deserialize spilled bytes");
        assert!(restored.mmap_backing.is_none());
        assert_eq!(restored, unspilled);
    }
}

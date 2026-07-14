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
pub(crate) struct TableMmapBacking {
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
    derive(
        Clone,
        PartialEq,
        Eq,
        serde::Serialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize
    )
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
    pub(crate) mmap_backing: Option<TableMmapBacking>,
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

// Manual rkyv impl under disk-spill: the derive can't handle `mmap_backing`,
// and serialization must read through `row_major_data()` so a spilled table
// archives its mmap contents (deserializing always yields an unspilled table).
// The archived layout matches what the derive generates without disk-spill, so
// both configurations produce byte-identical archives.
#[cfg(feature = "disk-spill")]
mod archived_table {
    use super::{FieldElement, IsField, Table};
    use math::field::element::ArchivedFieldElement;
    use rkyv::rancor::Fallible;
    use rkyv::ser::{Allocator, Writer};
    use rkyv::vec::{ArchivedVec, VecResolver};
    use rkyv::{Archive, Deserialize, Place, Portable, Serialize};

    #[derive(Portable, rkyv::bytecheck::CheckBytes)]
    #[bytecheck(crate = rkyv::bytecheck)]
    #[repr(C)]
    pub struct ArchivedTable<F: IsField>
    where
        F::BaseType: Archive,
    {
        pub data: ArchivedVec<ArchivedFieldElement<F>>,
        pub width: rkyv::primitive::ArchivedUsize,
        pub height: rkyv::primitive::ArchivedUsize,
    }

    pub struct TableResolver {
        data: VecResolver,
    }

    impl<F: IsField> Archive for Table<F>
    where
        F::BaseType: Archive,
    {
        type Archived = ArchivedTable<F>;
        type Resolver = TableResolver;

        fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
            rkyv::munge::munge!(let ArchivedTable { data, width, height } = out);
            ArchivedVec::resolve_from_len(self.width * self.height, resolver.data, data);
            self.width.resolve((), width);
            self.height.resolve((), height);
        }
    }

    impl<F: IsField, S> Serialize<S> for Table<F>
    where
        F::BaseType: Archive,
        FieldElement<F>: Serialize<S>,
        S: Fallible + Allocator + Writer + ?Sized,
    {
        fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
            Ok(TableResolver {
                data: ArchivedVec::serialize_from_slice(self.row_major_data(), serializer)?,
            })
        }
    }

    impl<F: IsField, D> Deserialize<Table<F>, D> for ArchivedTable<F>
    where
        F::BaseType: Archive,
        ArchivedFieldElement<F>: Deserialize<FieldElement<F>, D>,
        D: Fallible + ?Sized,
    {
        fn deserialize(&self, deserializer: &mut D) -> Result<Table<F>, D::Error> {
            // Element-by-element rather than `self.data.deserialize(...)`:
            // `ArchivedVec`'s blanket `Deserialize` impl needs a
            // `DeserializeUnsized` bound this crate doesn't otherwise use,
            // while the per-element bound below is already satisfied.
            let data = self
                .data
                .iter()
                .map(|elem| elem.deserialize(deserializer))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Table {
                data,
                width: self.width.to_native() as usize,
                height: self.height.to_native() as usize,
                mmap_backing: None,
            })
        }
    }
}

#[cfg(feature = "disk-spill")]
pub use archived_table::ArchivedTable;

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

    /// Build a [`Frame`] over this table, identical to [`Table::into_frame`].
    /// Only the small OOD frame is materialized (bounded by `step_size × width`),
    /// never the whole proof.
    pub fn into_frame(&self, main_trace_columns: usize, step_size: usize) -> Frame<F, F>
    where
        F: IsSubFieldOf<F>,
    {
        let height = self.height();
        debug_assert!(height.is_multiple_of(step_size));
        let steps = (0..height)
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

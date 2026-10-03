//! Narrow storage of a table's committed columns: each column at the bytes its
//! words need.
//!
//! A trace's 64-bit words are mostly small (flags, bytes, 16-bit limbs,
//! timestamps under 2^32), so a block's committed cells need about two bytes
//! each instead of eight. [`NarrowColumns`] keeps the raw words of a table's
//! columns, each little-endian at the width of its largest word (1, 2, 4 or 8
//! bytes) — the stark crate's `NarrowMain` layout, which `math_cuda::narrow`
//! packs and widens on the card — so widening gives back the same words bit
//! for bit and no proof byte can depend on it.
//!
//! The readers of a table's columns see them through [`HostColumns`] (a table)
//! and [`HostColumn`] (one column): the shape always, the field elements on
//! the host only when a reader there asks, which is what lets the columns stay
//! narrow on the host while the card reads them wide.

use std::sync::atomic::{AtomicU64, Ordering};

use math::field::{element::FieldElement, goldilocks::GoldilocksField, traits::IsField};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::mle::Mle;

/// Narrow tables widened on the host, and their cells, since the process
/// started: a device path widens on the card, so on a block these count only
/// the tables a host path read ([`host_widens`]).
static HOST_WIDENS: AtomicU64 = AtomicU64::new(0);
static HOST_WIDEN_CELLS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn note_host_widen(cells: usize) {
    HOST_WIDENS.fetch_add(1, Ordering::Relaxed);
    HOST_WIDEN_CELLS.fetch_add(cells as u64, Ordering::Relaxed);
}

/// `(tables, cells)` of narrow columns widened on the host so far.
pub fn host_widens() -> (u64, u64) {
    (
        HOST_WIDENS.load(Ordering::Relaxed),
        HOST_WIDEN_CELLS.load(Ordering::Relaxed),
    )
}

/// A table's columns at 1, 2, 4 or 8 bytes per cell (see the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NarrowColumns {
    rows: usize,
    /// Bytes per cell of each column.
    widths: Vec<u8>,
    /// Where each column starts in `data`.
    offsets: Vec<u64>,
    /// Column `c` is `rows × widths[c]` little-endian bytes from `offsets[c]`.
    data: Vec<u8>,
}

/// The bytes a word needs: 1, 2, 4 or 8.
fn width_of(max: u64) -> u8 {
    match max {
        0..=0xff => 1,
        0x100..=0xffff => 2,
        0x1_0000..=0xffff_ffff => 4,
        _ => 8,
    }
}

impl NarrowColumns {
    /// Columns packed from their parts (the card's pack): `rows` rows, the
    /// column widths, and the columns back to back. `None` when a width is not
    /// 1, 2, 4 or 8 or the bytes are not `rows × Σ widths`.
    pub fn from_parts(rows: usize, widths: Vec<u8>, data: Vec<u8>) -> Option<Self> {
        if widths.iter().any(|w| ![1, 2, 4, 8].contains(w)) {
            return None;
        }
        let mut offsets = Vec::with_capacity(widths.len());
        let mut total = 0usize;
        for &w in &widths {
            offsets.push(u64::try_from(total).ok()?);
            total = total.checked_add(rows.checked_mul(w as usize)?)?;
        }
        (total == data.len()).then_some(Self {
            rows,
            widths,
            offsets,
            data,
        })
    }

    /// Pack columns of raw words, every one `rows` long. `None` when they are
    /// not all one height.
    pub fn pack(columns: &[&[u64]]) -> Option<Self> {
        let rows = columns.first().map_or(0, |c| c.len());
        if columns.iter().any(|c| c.len() != rows) {
            return None;
        }
        let widths: Vec<u8> = columns
            .iter()
            .map(|c| width_of(c.iter().copied().max().unwrap_or(0)))
            .collect();
        let mut data = Vec::with_capacity(rows * widths.iter().map(|&w| w as usize).sum::<usize>());
        for (column, &w) in columns.iter().zip(&widths) {
            for word in column.iter() {
                data.extend_from_slice(&word.to_le_bytes()[..w as usize]);
            }
        }
        Self::from_parts(rows, widths, data)
    }

    /// [`Self::pack`] of field columns. `None` unless `F` is Goldilocks, whose
    /// elements are their raw words, or the columns are not one height.
    pub fn pack_columns<F: IsField + 'static>(columns: &[Mle<F>]) -> Option<Self> {
        let raw: Vec<&[u64]> = columns.iter().map(raw_words).collect::<Option<_>>()?;
        Self::pack(&raw)
    }

    /// Rows of each column.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// How many columns.
    pub fn cols(&self) -> usize {
        self.widths.len()
    }

    /// Bytes per cell of each column (1, 2, 4 or 8).
    pub fn widths(&self) -> &[u8] {
        &self.widths
    }

    /// Where each column starts in [`Self::data`].
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    /// The packed columns, back to back.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Column `c`'s words.
    pub fn column(&self, c: usize) -> Vec<u64> {
        let w = self.widths[c] as usize;
        let at = self.offsets[c] as usize;
        self.data[at..at + self.rows * w]
            .chunks_exact(w)
            .map(|cell| {
                let mut word = [0u8; 8];
                word[..w].copy_from_slice(cell);
                u64::from_le_bytes(word)
            })
            .collect()
    }

    /// The columns as field elements, their words bit for bit. `None` unless
    /// `F` is Goldilocks.
    pub fn widen<F: IsField + 'static>(&self) -> Option<Vec<Mle<F>>> {
        if std::any::TypeId::of::<F>() != std::any::TypeId::of::<GoldilocksField>() {
            return None;
        }
        let one = |c: usize| -> Option<Mle<F>> {
            let mut words = core::mem::ManuallyDrop::new(self.column(c));
            // SAFETY: `F == GoldilocksField`, whose elements are transparent
            // wrappers over one `u64`: the allocation describes the same words
            // either way.
            let evals = unsafe {
                Vec::from_raw_parts(
                    words.as_mut_ptr() as *mut FieldElement<F>,
                    words.len(),
                    words.capacity(),
                )
            };
            Mle::new(evals).ok()
        };
        #[cfg(feature = "parallel")]
        return (0..self.cols()).into_par_iter().map(one).collect();
        #[cfg(not(feature = "parallel"))]
        return (0..self.cols()).map(one).collect();
    }

    /// Narrow the first column wider than a byte by one width step in the map
    /// (its offsets unchanged): a wrong width map, which widens to other words,
    /// for the tests that check the prover refuses it. Returns whether a column
    /// was there to narrow.
    #[doc(hidden)]
    pub fn fault_width_map(&mut self) -> bool {
        match self.widths.iter_mut().find(|w| **w > 1) {
            Some(w) => {
                *w /= 2;
                true
            }
            None => false,
        }
    }
}

/// A column's raw words, when `F` is Goldilocks.
fn raw_words<F: IsField + 'static>(column: &Mle<F>) -> Option<&[u64]> {
    if std::any::TypeId::of::<F>() != std::any::TypeId::of::<GoldilocksField>() {
        return None;
    }
    // SAFETY: `F == GoldilocksField`, a transparent wrapper over `u64`.
    Some(unsafe {
        core::slice::from_raw_parts(column.evals().as_ptr() as *const u64, column.len())
    })
}

/// A table's base columns as a reader finds them: their shape always, their
/// field elements on the host on demand — widened then, if they are held
/// narrow ([`NarrowColumns`]). A device path reads the shape and the card's
/// copy; only a host path asks for [`host`](Self::host).
pub trait HostColumns<F: IsField>: Sync {
    /// How many columns.
    fn width(&self) -> usize;
    /// Rows of each column; `None` when there are none or they differ.
    fn rows(&self) -> Option<usize>;
    /// The columns on the host.
    fn host(&self) -> &[Mle<F>];
}

/// The height columns share, if they do.
fn common_rows<F: IsField + 'static>(columns: &[Mle<F>]) -> Option<usize> {
    let rows = columns.first()?.len();
    columns.iter().all(|c| c.len() == rows).then_some(rows)
}

impl<F: IsField + 'static> HostColumns<F> for [Mle<F>]
where
    FieldElement<F>: Sync,
{
    fn width(&self) -> usize {
        self.len()
    }

    fn rows(&self) -> Option<usize> {
        common_rows(self)
    }

    fn host(&self) -> &[Mle<F>] {
        self
    }
}

impl<F: IsField + 'static> HostColumns<F> for Vec<Mle<F>>
where
    FieldElement<F>: Sync,
{
    fn width(&self) -> usize {
        self.len()
    }

    fn rows(&self) -> Option<usize> {
        common_rows(self)
    }

    fn host(&self) -> &[Mle<F>] {
        self
    }
}

/// One column the same way: its height always, its field elements on demand.
/// What a stacked polynomial is made of (`whir_chain::Stacked`).
pub trait HostColumn<F: IsField>: Sync {
    /// Rows of the column.
    fn rows(&self) -> usize;
    /// The column on the host.
    fn host(&self) -> &Mle<F>;
}

impl<F: IsField + 'static> HostColumn<F> for Mle<F>
where
    FieldElement<F>: Sync,
{
    fn rows(&self) -> usize {
        self.len()
    }

    fn host(&self) -> &Mle<F> {
        self
    }
}

/// Column `index` of a table's [`HostColumns`]: widened with its table, if the
/// table is held narrow.
pub struct ColumnOf<'a, C: ?Sized> {
    pub columns: &'a C,
    pub index: usize,
}

impl<F: IsField, C: HostColumns<F> + ?Sized> HostColumn<F> for ColumnOf<'_, C> {
    fn rows(&self) -> usize {
        self.columns.rows().unwrap_or(0)
    }

    fn host(&self) -> &Mle<F> {
        &self.columns.host()[self.index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as Gl;

    /// Every width boundary round trips through the host pack and widen, in
    /// every position: 0, 2^8 − 1, 2^8, 2^16 − 1, 2^16, 2^32 − 1, 2^32, the
    /// Goldilocks p − 1, a non-canonical word above p, and u64::MAX.
    #[test]
    fn narrow_columns_round_trip_at_every_width_boundary() {
        let edges = [
            0u64,
            0xff,
            0x100,
            0xffff,
            0x1_0000,
            0xffff_ffff,
            0x1_0000_0000,
            0xffff_ffff_0000_0000,
            0xffff_ffff_0000_0001,
            u64::MAX,
        ];
        let rows = 1usize << 9;
        let columns: Vec<Vec<u64>> = edges
            .iter()
            .map(|&edge| {
                (0..rows)
                    .map(|r| {
                        if r == rows - 1 {
                            edge
                        } else {
                            (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) % (edge / 2 + 1)
                        }
                    })
                    .collect()
            })
            .collect();
        let raw: Vec<&[u64]> = columns.iter().map(Vec::as_slice).collect();
        let narrow = NarrowColumns::pack(&raw).expect("one height");
        assert_eq!(narrow.widths(), &[1, 1, 2, 2, 4, 4, 8, 8, 8, 8]);
        assert_eq!(
            narrow.data().len(),
            rows * narrow.widths().iter().map(|&w| w as usize).sum::<usize>()
        );
        for (c, column) in columns.iter().enumerate() {
            assert_eq!(&narrow.column(c), column, "column {c}");
        }
        let wide = narrow.widen::<Gl>().expect("Goldilocks");
        for (mle, column) in wide.iter().zip(&columns) {
            let words: Vec<u64> = mle.evals().iter().map(|v| *v.value()).collect();
            assert_eq!(&words, column);
        }
        let again =
            NarrowColumns::from_parts(rows, narrow.widths().to_vec(), narrow.data().to_vec());
        assert_eq!(again.as_ref(), Some(&narrow));
    }

    /// A width that is not 1, 2, 4 or 8, or bytes that are not the shape's,
    /// make no packed columns.
    #[test]
    fn packed_parts_of_the_wrong_shape_are_refused() {
        assert!(NarrowColumns::from_parts(4, vec![1, 3], vec![0; 16]).is_none());
        assert!(NarrowColumns::from_parts(4, vec![1, 2], vec![0; 11]).is_none());
        assert!(NarrowColumns::from_parts(4, vec![1, 2], vec![0; 12]).is_some());
        assert!(NarrowColumns::pack(&[&[1, 2], &[3]]).is_none());
    }

    /// A wrong width map widens to other words.
    #[test]
    fn a_wrong_width_map_widens_to_other_words() {
        let column: Vec<u64> = (0..64u64).map(|r| r * 100_000).collect();
        let mut narrow = NarrowColumns::pack(&[&column]).expect("one height");
        assert_eq!(narrow.widths(), &[4]);
        assert!(narrow.fault_width_map());
        assert_ne!(narrow.column(0), column);
    }

    /// Only Goldilocks columns widen to field elements: their elements are
    /// their raw words.
    #[test]
    fn only_goldilocks_columns_widen() {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
        let narrow = NarrowColumns::pack(&[&[1, 2, 3, 4]]).expect("one height");
        assert!(narrow.widen::<Ext3>().is_none());
        assert!(narrow.widen::<Gl>().is_some());
    }
}

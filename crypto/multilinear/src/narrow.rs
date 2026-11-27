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
use crate::pinned::PoolClass;

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
        Self::pack_in(columns, PoolClass::Shared)
    }

    fn pack_in(columns: &[&[u64]], class: PoolClass) -> Option<Self> {
        let rows = columns.first().map_or(0, |c| c.len());
        if columns.iter().any(|c| c.len() != rows) {
            return None;
        }
        let widths: Vec<u8> = columns
            .iter()
            .map(|c| width_of(c.iter().copied().max().unwrap_or(0)))
            .collect();
        let total = rows * widths.iter().map(|&w| w as usize).sum::<usize>();
        // The bytes a group's upload reads: page-locked when that is on.
        let mut data = class.with_pinned(total, || Vec::with_capacity(total));
        for (column, &w) in columns.iter().zip(&widths) {
            for word in column.iter() {
                data.extend_from_slice(&word.to_le_bytes()[..w as usize]);
            }
        }
        Self::from_parts(rows, widths, data)
    }

    /// [`Self::pack`] on the pool: each column's largest word, then its
    /// bytes, in blocks of rows at once. The same bytes as [`Self::pack`].
    #[cfg(feature = "parallel")]
    pub fn pack_par(columns: &[&[u64]], class: PoolClass) -> Option<Self> {
        const BLOCK: usize = 1 << 16;
        let rows = columns.first().map_or(0, |c| c.len());
        if columns.iter().any(|c| c.len() != rows) {
            return None;
        }
        let widths: Vec<u8> = columns
            .par_iter()
            .map(|c| {
                width_of(
                    c.par_chunks(BLOCK)
                        .map(|b| b.iter().copied().max().unwrap_or(0))
                        .max()
                        .unwrap_or(0),
                )
            })
            .collect();
        let total = rows * widths.iter().map(|&w| w as usize).sum::<usize>();
        // The bytes a store's upload reads: page-locked when that is on.
        let mut data = class.with_pinned(total, || vec![0u8; total]);
        let mut parts: Vec<(&[u64], usize, &mut [u8])> = Vec::with_capacity(columns.len());
        let mut rest = data.as_mut_slice();
        for (column, &w) in columns.iter().zip(&widths) {
            let (out, tail) = rest.split_at_mut(rows * w as usize);
            parts.push((column, w as usize, out));
            rest = tail;
        }
        parts.into_par_iter().for_each(|(column, w, out)| {
            out.par_chunks_mut(BLOCK * w)
                .zip(column.par_chunks(BLOCK))
                .for_each(|(out, words)| put_words(out, w, words));
        });
        Self::from_parts(rows, widths, data)
    }

    /// [`Self::pack`] (no pool without `parallel`).
    #[cfg(not(feature = "parallel"))]
    pub fn pack_par(columns: &[&[u64]], class: PoolClass) -> Option<Self> {
        Self::pack_in(columns, class)
    }

    /// [`Self::pack`] of a row-major trace: the `cols`-wide rows of words
    /// `row_major`, packed column by column (the stark crate's
    /// `NarrowMain::pack`, a block of rows at a time so the block stays in
    /// cache across the columns). On the caller's thread. `None` when `cols`
    /// is zero or does not divide the words.
    pub fn pack_row_major(row_major: &[u64], cols: usize) -> Option<Self> {
        const BLOCK_ROWS: usize = 1 << 10;
        if cols == 0 || !row_major.len().is_multiple_of(cols) {
            return None;
        }
        let rows = row_major.len() / cols;
        let mut max = vec![0u64; cols];
        for row in row_major.chunks_exact(cols) {
            for (m, &v) in max.iter_mut().zip(row) {
                *m = (*m).max(v);
            }
        }
        let widths: Vec<u8> = max.into_iter().map(width_of).collect();
        let total: usize = widths.iter().map(|&w| rows * w as usize).sum();
        // The bytes a group's upload reads: page-locked when that is on.
        let mut data = crate::pinned::with_pinned(total, || vec![0u8; total]);
        let mut columns: Vec<&mut [u8]> = Vec::with_capacity(cols);
        let mut rest = data.as_mut_slice();
        for &w in &widths {
            let (column, tail) = rest.split_at_mut(rows * w as usize);
            columns.push(column);
            rest = tail;
        }
        for first in (0..rows).step_by(BLOCK_ROWS) {
            let last = (first + BLOCK_ROWS).min(rows);
            let block = &row_major[first * cols..last * cols];
            for (c, (column, &w)) in columns.iter_mut().zip(&widths).enumerate() {
                let words = block.iter().skip(c).step_by(cols);
                let w = w as usize;
                let out = &mut column[first * w..last * w];
                match w {
                    1 => out.iter_mut().zip(words).for_each(|(o, &v)| *o = v as u8),
                    2 => out
                        .chunks_exact_mut(2)
                        .zip(words)
                        .for_each(|(o, &v)| o.copy_from_slice(&(v as u16).to_le_bytes())),
                    4 => out
                        .chunks_exact_mut(4)
                        .zip(words)
                        .for_each(|(o, &v)| o.copy_from_slice(&(v as u32).to_le_bytes())),
                    _ => out
                        .chunks_exact_mut(8)
                        .zip(words)
                        .for_each(|(o, &v)| o.copy_from_slice(&v.to_le_bytes())),
                }
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

    /// The parts, for a store to move elsewhere and back
    /// ([`Self::from_parts`]): rows, widths and the bytes. The offsets follow
    /// from the widths.
    pub fn into_parts(self) -> (usize, Vec<u8>, Vec<u8>) {
        (self.rows, self.widths, self.data)
    }

    /// The word at `row` of column `c`.
    pub fn get(&self, row: usize, c: usize) -> u64 {
        let w = self.widths[c] as usize;
        let at = self.offsets[c] as usize + row * w;
        let mut word = [0u8; 8];
        word[..w].copy_from_slice(&self.data[at..at + w]);
        u64::from_le_bytes(word)
    }

    /// Writes `value` at `row` of column `c` when it fits the column's width;
    /// `false` (nothing written) when it does not.
    pub fn set(&mut self, row: usize, c: usize, value: u64) -> bool {
        let w = self.widths[c] as usize;
        if width_of(value) as usize > w {
            return false;
        }
        let at = self.offsets[c] as usize + row * w;
        self.data[at..at + w].copy_from_slice(&value.to_le_bytes()[..w]);
        true
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

/// `words` little-endian at `w` bytes each into `out` (`words.len() × w`
/// bytes); each word fits `w`.
#[cfg(feature = "parallel")]
fn put_words(out: &mut [u8], w: usize, words: &[u64]) {
    match w {
        1 => out.iter_mut().zip(words).for_each(|(o, &v)| *o = v as u8),
        2 => out
            .chunks_exact_mut(2)
            .zip(words)
            .for_each(|(o, &v)| o.copy_from_slice(&(v as u16).to_le_bytes())),
        4 => out
            .chunks_exact_mut(4)
            .zip(words)
            .for_each(|(o, &v)| o.copy_from_slice(&(v as u32).to_le_bytes())),
        _ => out
            .chunks_exact_mut(8)
            .zip(words)
            .for_each(|(o, &v)| o.copy_from_slice(&v.to_le_bytes())),
    }
}

/// The cells a narrow store's run holds at most ([`store_runs`]): a quarter of
/// a GiB of words, so its packed bytes on the card while it widens stay at
/// most that, and well inside one widen launch.
pub const STORE_RUN_CELLS: usize = 1 << 25;

/// Columns of heights `lens`, in order, cut into the runs a narrow store packs
/// and widens one at a time: consecutive columns of one height, at most
/// `max_cells` cells a run unless one column alone is more. Every column is in
/// exactly one run, in order.
pub fn store_runs(lens: &[usize], max_cells: usize) -> Vec<std::ops::Range<usize>> {
    let mut runs = Vec::new();
    let mut start = 0usize;
    for k in 1..=lens.len() {
        let cells = (k - start) * lens[start];
        let ends = k == lens.len() || lens[k] != lens[start] || cells + lens[k] > max_cells;
        if ends {
            runs.push(start..k);
            start = k;
        }
    }
    runs
}

/// The most a narrow store of columns of heights `lens` holds on the card
/// beside its words while a run widens: the largest run's words at 8 bytes,
/// what its packed bytes come to at most.
pub fn store_staging_bound(lens: &[usize], max_cells: usize) -> u64 {
    store_runs(lens, max_cells)
        .into_iter()
        .map(|r| lens[r].iter().map(|&n| n as u64 * 8).sum::<u64>())
        .max()
        .unwrap_or(0)
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

    /// A row-major trace packs to what its columns pack to, at every width
    /// boundary and across the pack's row blocks; a width that does not divide
    /// the words packs nothing.
    #[test]
    fn a_row_major_trace_packs_as_its_columns() {
        let edges = [
            0u64,
            0xff,
            0x100,
            0xffff,
            0x1_0000,
            0xffff_ffff,
            0x1_0000_0000,
            u64::MAX,
        ];
        for rows in [1usize, 7, 1024, 1025, 3000] {
            let columns: Vec<Vec<u64>> = edges
                .iter()
                .map(|&edge| {
                    (0..rows)
                        .map(|r| {
                            if r == rows / 2 {
                                edge
                            } else {
                                (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) % (edge / 2 + 1)
                            }
                        })
                        .collect()
                })
                .collect();
            let row_major: Vec<u64> = (0..rows)
                .flat_map(|r| columns.iter().map(move |c| c[r]))
                .collect();
            let raw: Vec<&[u64]> = columns.iter().map(Vec::as_slice).collect();
            assert_eq!(
                NarrowColumns::pack_row_major(&row_major, edges.len()),
                NarrowColumns::pack(&raw),
                "{rows} rows"
            );
        }
        assert!(NarrowColumns::pack_row_major(&[1, 2, 3], 2).is_none());
        assert!(NarrowColumns::pack_row_major(&[1, 2], 0).is_none());
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

    /// The pool's pack writes the serial pack's bytes, at every width and
    /// across its row blocks (a height that is not a block multiple), and
    /// refuses columns of two heights.
    #[test]
    fn the_pools_pack_is_the_serial_pack() {
        let edges = [
            0u64,
            0xff,
            0x100,
            0xffff,
            0x1_0000,
            0xffff_ffff,
            0x1_0000_0000,
            u64::MAX,
        ];
        for rows in [1usize, 7, (1 << 16) + 3, 3 << 16] {
            let columns: Vec<Vec<u64>> = edges
                .iter()
                .map(|&edge| {
                    (0..rows)
                        .map(|r| {
                            if r == rows / 2 {
                                edge
                            } else {
                                (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) % (edge / 2 + 1)
                            }
                        })
                        .collect()
                })
                .collect();
            let raw: Vec<&[u64]> = columns.iter().map(Vec::as_slice).collect();
            let par = NarrowColumns::pack_par(&raw, PoolClass::Shared).expect("one height");
            assert_eq!(
                Some(&par),
                NarrowColumns::pack(&raw).as_ref(),
                "{rows} rows"
            );
            assert_eq!(par.widths(), &[1, 1, 2, 2, 4, 4, 8, 8], "{rows} rows");
        }
        assert!(NarrowColumns::pack_par(&[&[1, 2], &[3]], PoolClass::Shared).is_none());
    }

    /// A store's columns cut into runs: each run one height, in order, every
    /// column once, at most the cap's cells unless one column alone is more;
    /// the staging bound is the largest run's words at 8 bytes.
    #[test]
    fn a_stores_columns_cut_into_runs_of_one_height_under_the_cap() {
        assert!(store_runs(&[], 8).is_empty());
        assert_eq!(store_staging_bound(&[], 8), 0);
        // Heights change at 3 and 5; the cap (8 cells) cuts the first height's
        // columns of 2 into runs of 4.
        let lens = [2, 2, 2, 4, 4, 2, 2, 2, 2, 2, 16, 1];
        let runs = store_runs(&lens, 8);
        assert_eq!(runs, vec![0..3, 3..5, 5..9, 9..10, 10..11, 11..12]);
        for r in &runs {
            assert!(lens[r.clone()].iter().all(|&n| n == lens[r.start]));
            let cells: usize = lens[r.clone()].iter().sum();
            assert!(cells <= 8 || r.len() == 1, "{r:?}");
        }
        assert_eq!(store_staging_bound(&lens, 8), 16 * 8);
        assert_eq!(store_staging_bound(&[4, 4, 4], 8), 8 * 8);
        // Uncapped, a height is one run.
        assert_eq!(
            store_runs(&lens, usize::MAX),
            vec![0..3, 3..5, 5..10, 10..11, 11..12]
        );
    }
}

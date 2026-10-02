//! Narrow storage of a main trace: each column at the bytes its values need.
//!
//! A trace's 64-bit words are mostly small (flags, bytes, 16-bit limbs,
//! timestamps under 2^32): on the no-epoch record block the main cells need
//! 2.0 bytes on average instead of 8. [`NarrowMain`] keeps the raw words of a
//! row-major main trace column by column, each column little-endian at the
//! width of its largest word (1, 2, 4 or 8 bytes), so widening gives back the
//! same words bit for bit: no proof byte can depend on it.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Rows per block of the widen pass: each block reads and writes every column
/// once, so the row-major side is written sequentially.
const BLOCK_ROWS: usize = 1 << 12;

/// Rows per block of the pack pass: a block's words (a few hundred KiB) stay in
/// cache while each column takes its strided share.
const PACK_BLOCK_ROWS: usize = 1 << 10;

/// A main trace at 1, 2, 4 or 8 bytes per cell (see the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NarrowMain {
    rows: usize,
    /// Bytes per cell of each column.
    widths: Vec<u8>,
    /// Where each column starts in `data`.
    offsets: Vec<usize>,
    /// Column-major: column `c` is `rows * widths[c]` little-endian bytes from
    /// `offsets[c]`.
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

impl NarrowMain {
    /// Pack the `cols`-wide row-major words `row_major`. `cols` must divide its
    /// length (a trace of `rows * cols` cells).
    ///
    /// Runs on the caller's thread: the block packs each streamed instance on
    /// its committer, and a pack spread over the shared pool there competed
    /// with the executor and the walk (FAST 505).
    pub fn pack(row_major: &[u64], cols: usize) -> Self {
        let rows = row_major.len().checked_div(cols).unwrap_or(0);
        debug_assert_eq!(rows * cols, row_major.len());
        let row_major = &row_major[..rows * cols];

        let mut max = vec![0u64; cols];
        for row in row_major.chunks_exact(cols.max(1)) {
            for (m, &v) in max.iter_mut().zip(row) {
                *m = (*m).max(v);
            }
        }
        let widths: Vec<u8> = max.into_iter().map(width_of).collect();
        let mut offsets = Vec::with_capacity(cols);
        let mut total = 0usize;
        for &w in &widths {
            offsets.push(total);
            total += rows * w as usize;
        }
        let mut data = vec![0u8; total];

        // Each column's slice, written a block of rows at a time so the
        // block's rows stay in cache across the columns.
        let mut columns: Vec<&mut [u8]> = Vec::with_capacity(cols);
        let mut rest = data.as_mut_slice();
        for &w in &widths {
            let (column, tail) = rest.split_at_mut(rows * w as usize);
            columns.push(column);
            rest = tail;
        }
        for first in (0..rows).step_by(PACK_BLOCK_ROWS) {
            let last = (first + PACK_BLOCK_ROWS).min(rows);
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

        Self {
            rows,
            widths,
            offsets,
            data,
        }
    }

    /// Rows of the trace.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns of the trace.
    pub fn cols(&self) -> usize {
        self.widths.len()
    }

    /// Bytes per cell of each column (1, 2, 4 or 8).
    pub fn widths(&self) -> &[u8] {
        &self.widths
    }

    /// Where each column starts in [`Self::data`].
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// The packed columns (see [`NarrowMain`]'s fields).
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Column `c`'s words.
    pub fn column(&self, c: usize) -> Vec<u64> {
        let w = self.widths[c] as usize;
        let bytes = &self.data[self.offsets[c]..self.offsets[c] + self.rows * w];
        bytes
            .chunks_exact(w)
            .map(|cell| {
                let mut word = [0u8; 8];
                word[..w].copy_from_slice(cell);
                u64::from_le_bytes(word)
            })
            .collect()
    }

    /// Flip the low bit of the first packed byte: a wrong word, for the tests
    /// that check a bad widen is refused.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn flip_first_bit(&mut self) {
        if let Some(b) = self.data.first_mut() {
            *b ^= 1;
        }
    }

    /// Write the row-major words into `out` (`rows * cols` of them).
    pub fn widen_into(&self, out: &mut [u64]) {
        let cols = self.cols();
        assert_eq!(out.len(), self.rows * cols, "widen_into: output size");
        if cols == 0 {
            return;
        }
        let widen_block = |(b, block): (usize, &mut [u64])| {
            let first = b * BLOCK_ROWS;
            for (c, (&w, &off)) in self.widths.iter().zip(&self.offsets).enumerate() {
                let w = w as usize;
                let src = &self.data[off + first * w..];
                for (r, row) in block.chunks_exact_mut(cols).enumerate() {
                    let mut word = [0u8; 8];
                    word[..w].copy_from_slice(&src[r * w..(r + 1) * w]);
                    row[c] = u64::from_le_bytes(word);
                }
            }
        };
        #[cfg(feature = "parallel")]
        out.par_chunks_mut(BLOCK_ROWS * cols)
            .enumerate()
            .for_each(widen_block);
        #[cfg(not(feature = "parallel"))]
        out.chunks_mut(BLOCK_ROWS * cols)
            .enumerate()
            .for_each(widen_block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every width boundary, in every position, round trips: 0, 2^8 − 1,
    /// 2^8, 2^16 − 1, 2^16, 2^32 − 1, 2^32, the Goldilocks p − 1 and
    /// u64::MAX, over a row count that is not a multiple of the block.
    #[test]
    fn narrow_round_trips_at_every_width_boundary() {
        let edges = [
            0u64,
            0xff,
            0x100,
            0xffff,
            0x1_0000,
            0xffff_ffff,
            0x1_0000_0000,
            0xffff_ffff_0000_0000,
            u64::MAX,
        ];
        let cols = edges.len() + 1;
        let rows = BLOCK_ROWS + 3;
        let data: Vec<u64> = (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                // Column c < 9 tops out at edges[c] (in the last row); the
                // last column is all zeros.
                if c == edges.len() {
                    0
                } else if r == rows - 1 {
                    edges[c]
                } else {
                    (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) % (edges[c] / 2 + 1)
                }
            })
            .collect();
        let narrow = NarrowMain::pack(&data, cols);
        assert_eq!(narrow.widths(), &[1, 1, 2, 2, 4, 4, 8, 8, 8, 1]);
        let mut back = vec![0u64; data.len()];
        narrow.widen_into(&mut back);
        assert_eq!(back, data);
        for c in 0..cols {
            let column: Vec<u64> = data.iter().skip(c).step_by(cols).copied().collect();
            assert_eq!(narrow.column(c), column, "column {c}");
        }
        assert_eq!(
            narrow.data().len(),
            rows * narrow.widths().iter().map(|&w| w as usize).sum::<usize>()
        );
    }

    /// An empty trace packs to nothing and widens to nothing.
    #[test]
    fn an_empty_trace_packs_to_nothing() {
        let narrow = NarrowMain::pack(&[], 3);
        assert_eq!(
            (narrow.rows(), narrow.cols(), narrow.data().len()),
            (0, 3, 0)
        );
        narrow.widen_into(&mut []);
    }
}

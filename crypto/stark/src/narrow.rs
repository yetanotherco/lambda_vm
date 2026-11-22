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
        // The bytes a group's upload reads: page-locked when that is on.
        let mut data = multilinear::pinned::with_pinned(total, || vec![0u8; total]);

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

    /// A packed trace from its parts (the device pack's output): `rows` rows,
    /// the column widths, and the columns back to back. `None` when a width is
    /// not 1, 2, 4 or 8 or the bytes are not `rows × Σ widths`.
    pub fn from_parts(rows: usize, widths: Vec<u8>, data: Vec<u8>) -> Option<Self> {
        if widths.iter().any(|w| ![1, 2, 4, 8].contains(w)) {
            return None;
        }
        let mut offsets = Vec::with_capacity(widths.len());
        let mut total = 0usize;
        for &w in &widths {
            offsets.push(total);
            total = total.checked_add(rows.checked_mul(w as usize)?)?;
        }
        (total == data.len()).then_some(Self {
            rows,
            widths,
            offsets,
            data,
        })
    }

    /// The rows, the column widths and the packed bytes, for a holder of
    /// another type with this layout (`multilinear::narrow::NarrowColumns`);
    /// [`Self::from_parts`] puts them back together.
    pub fn into_parts(self) -> (usize, Vec<u8>, Vec<u8>) {
        (self.rows, self.widths, self.data)
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

    /// A 128-bit digest of the packed trace (its shape, widths and bytes): a
    /// fast, non-cryptographic check that a trace built again is this one. A
    /// prover bug is the threat, not an adversary.
    pub fn digest(&self) -> [u64; 2] {
        digest_parts(self.rows, &self.widths, &self.data)
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

/// [`NarrowMain::digest`] of a packed trace held as its parts (the spill
/// store digests the bytes on its writer thread).
pub(crate) fn digest_parts(rows: usize, widths: &[u8], data: &[u8]) -> [u64; 2] {
    const K: [u64; 4] = [
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
        0x1656_67b1_9e37_79f9,
        0x85eb_ca77_c2b2_ae63,
    ];
    let mix = |acc: u64, w: u64, k: u64| (acc ^ w).wrapping_mul(k).rotate_left(31);
    let mut lanes = [
        rows as u64,
        widths.len() as u64,
        data.len() as u64,
        0x243f_6a88_85a3_08d3,
    ];
    for (i, &w) in widths.iter().enumerate() {
        lanes[i % 4] = mix(lanes[i % 4], u64::from(w), K[i % 4]);
    }
    let mut blocks = data.chunks_exact(32);
    for block in &mut blocks {
        for (l, word) in block.chunks_exact(8).enumerate() {
            let w = u64::from_le_bytes(word.try_into().expect("8 bytes"));
            lanes[l] = mix(lanes[l], w, K[l]);
        }
    }
    let mut tail = [0u8; 32];
    tail[..blocks.remainder().len()].copy_from_slice(blocks.remainder());
    for (l, word) in tail.chunks_exact(8).enumerate() {
        let w = u64::from_le_bytes(word.try_into().expect("8 bytes"));
        lanes[l] = mix(lanes[l], w, K[l]);
    }
    // MurmurHash3's finalizer, so every input bit reaches every output bit.
    let fmix = |mut h: u64| {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    };
    [
        fmix(lanes[0] ^ lanes[2].rotate_left(23)),
        fmix(lanes[1] ^ lanes[3].rotate_left(41)),
    ]
}

/// A [`NarrowMain`] built a block of rows at a time, so a generator never holds
/// a whole 64-bit trace: each pushed block is packed into its columns at once,
/// and a column whose block needs more bytes than it had so far is re-encoded
/// at the wider width. Rows never pushed are zero. The result is the
/// [`NarrowMain::pack`] of the whole trace, byte for byte.
pub struct NarrowBuilder {
    rows: usize,
    cols: usize,
    pushed: usize,
    widths: Vec<u8>,
    /// Column-major, one buffer per column at its width so far.
    columns: Vec<Vec<u8>>,
}

impl NarrowBuilder {
    /// A builder for a `rows` × `cols` trace.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            pushed: 0,
            widths: vec![1; cols],
            columns: (0..cols).map(|_| Vec::with_capacity(rows)).collect(),
        }
    }

    /// Append whole rows, row-major (`cols` words each). Panics past `rows`.
    pub fn push_rows(&mut self, row_major: &[u64]) {
        let cols = self.cols.max(1);
        assert_eq!(row_major.len() % cols, 0, "push_rows: whole rows only");
        let n = row_major.len() / cols;
        assert!(
            self.pushed + n <= self.rows,
            "push_rows: past the trace's rows"
        );
        for c in 0..self.cols {
            let words = row_major.iter().skip(c).step_by(cols);
            let need = width_of(words.clone().copied().max().unwrap_or(0));
            if need > self.widths[c] {
                self.columns[c] = rewiden(&self.columns[c], self.widths[c], need, self.rows);
                self.widths[c] = need;
            }
            let column = &mut self.columns[c];
            match self.widths[c] {
                1 => column.extend(words.map(|&v| v as u8)),
                2 => words.for_each(|&v| column.extend_from_slice(&(v as u16).to_le_bytes())),
                4 => words.for_each(|&v| column.extend_from_slice(&(v as u32).to_le_bytes())),
                _ => words.for_each(|&v| column.extend_from_slice(&v.to_le_bytes())),
            }
        }
        self.pushed += n;
    }

    /// The packed trace: the rows pushed, then zero rows up to `rows`.
    pub fn finish(self) -> NarrowMain {
        let rows = self.rows;
        let mut offsets = Vec::with_capacity(self.cols);
        let total: usize = self.widths.iter().map(|&w| rows * w as usize).sum();
        let mut data = Vec::with_capacity(total);
        for (column, &w) in self.columns.into_iter().zip(&self.widths) {
            offsets.push(data.len());
            data.extend_from_slice(&column);
            data.resize(data.len() + (rows - self.pushed) * w as usize, 0);
        }
        NarrowMain {
            rows,
            widths: self.widths,
            offsets,
            data,
        }
    }
}

/// `bytes`, cells of `from` bytes each, re-encoded at `to` bytes each (little
/// endian, so each cell is zero-extended), with room for `rows` cells.
fn rewiden(bytes: &[u8], from: u8, to: u8, rows: usize) -> Vec<u8> {
    let (from, to) = (from as usize, to as usize);
    let mut out = Vec::with_capacity(rows * to);
    for cell in bytes.chunks_exact(from) {
        out.extend_from_slice(cell);
        out.resize(out.len() + (to - from), 0);
    }
    out
}

/// A [`NarrowMain`] written a cell at a time straight into its packed columns,
/// so a generator never holds a 64-bit trace, not even a block of one.
///
/// The caller picks each column's width up front: a guess, such as the widths
/// the table's earlier traces needed. The writer stores each word's low bytes
/// at its column's width and keeps the OR of every word written to each
/// column, which has the bit length of the column's largest word. When every
/// word fit, [`Self::finish`] gives the [`NarrowMain::pack`] of the same words,
/// byte for byte: a column given more bytes than it needed is narrowed in
/// place. When a word did not fit, the bytes written are not the trace, and
/// `finish` returns the widths the trace needs instead, for the caller to write
/// it again at those (which cannot miss: they hold every word written).
///
/// Cells never written are zero. A cell written twice keeps its last word, but
/// both words count toward its column's width: writers set each cell once.
pub struct NarrowWriter {
    rows: usize,
    columns: Vec<WriterColumn>,
    data: Vec<u8>,
}

/// Rows `[first_row, first_row + rows)` of a [`NarrowWriter`], writable on their
/// own: one part of [`NarrowWriter::split_rows`]. Rows are the writer's own
/// indices; a word is stored as [`NarrowWriter::set`] stores it, and its bits
/// are kept in this part's OR until [`NarrowWriter::absorb`] takes them.
pub struct NarrowRows<'a> {
    first_row: usize,
    rows: usize,
    columns: Vec<(&'a mut [u8], usize)>,
    seen: Vec<u64>,
}

impl NarrowRows<'_> {
    /// The writer's rows this part holds.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.first_row..self.first_row + self.rows
    }

    /// Write `word` into `row` (a row of this part, by the writer's index) of
    /// column `col`, as [`NarrowWriter::set`] does.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, word: u64) {
        debug_assert!(
            self.range().contains(&row),
            "NarrowRows: row {row} outside {:?}",
            self.range()
        );
        self.seen[col] |= word;
        let (data, width) = &mut self.columns[col];
        let at = (row - self.first_row) * *width;
        match *width {
            1 => data[at] = word as u8,
            2 => data[at..at + 2].copy_from_slice(&(word as u16).to_le_bytes()),
            4 => data[at..at + 4].copy_from_slice(&(word as u32).to_le_bytes()),
            _ => data[at..at + 8].copy_from_slice(&word.to_le_bytes()),
        }
    }

    /// This part's OR of the words it wrote, a column each.
    pub fn into_seen(self) -> Vec<u64> {
        self.seen
    }
}

/// A [`NarrowWriter`] column: where it starts in the data, its width, and the
/// OR of every word written to it.
#[derive(Clone, Copy)]
struct WriterColumn {
    offset: usize,
    width: usize,
    seen: u64,
}

/// The fewest bytes of 1, 2, 4 and 8 that hold `w` bytes.
fn valid_width(w: u8) -> usize {
    match w {
        0 | 1 => 1,
        2 => 2,
        3 | 4 => 4,
        _ => 8,
    }
}

/// `LAMBDA_VM_NARROW_STAGGER=<columns>`: a [`NarrowWriter`] of at least that
/// many columns leaves [`STAGGER_GAP`] bytes before each column after the
/// first while it is written, and [`NarrowWriter::finish`] closes the gaps.
/// The trace is the same, byte for byte. Without the gaps, at a power-of-two
/// row count every column starts at a multiple of the rows (times its width),
/// so a row's writes, one a column, all fall in the same cache sets, which
/// costs once a row writes more columns than the cache has ways; under huge
/// pages nothing spreads them. A narrower table keeps the columns back to
/// back: closing the gaps is a pass over the table it has no aliasing to win
/// back. Unset, it is [`STAGGER_DEFAULT_COLUMNS`] (the default); `0` turns it
/// off.
pub const NARROW_STAGGER_ENV: &str = "LAMBDA_VM_NARROW_STAGGER";

/// The fewest columns a staggered table has by default: the wide tables
/// (ECDAS 515, KECCAK 907, KECCAK_W 1031, KECCAK_RND 1480, ECREC 1543, ECSM
/// 667) and none of the narrow ones, whose rows write fewer columns than the
/// caches have ways (COPY's 41 is the widest of those).
pub const STAGGER_DEFAULT_COLUMNS: usize = 256;

/// The bytes left before each column after the first under
/// [`NARROW_STAGGER_ENV`]: a cache line, so the columns' starts step through
/// the cache sets (1024 columns before two share a set of a 64 KiB period).
pub const STAGGER_GAP: usize = 64;

static NARROW_STAGGER_FOR_TEST: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Force [`narrow_stagger_columns`] to `Some(columns)` (0 is off) or back to
/// the environment's word (`None`), process wide: tests only.
#[doc(hidden)]
pub fn set_narrow_stagger_for_test(columns: Option<usize>) {
    NARROW_STAGGER_FOR_TEST.store(
        columns.unwrap_or(usize::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// The fewest columns a new [`NarrowWriter`] needs to leave a gap before each
/// column ([`NARROW_STAGGER_ENV`], read once; [`STAGGER_DEFAULT_COLUMNS`]
/// unless the word is another number); 0 is off.
pub fn narrow_stagger_columns() -> usize {
    let forced = NARROW_STAGGER_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed);
    if forced != usize::MAX {
        return forced;
    }
    static COLUMNS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *COLUMNS.get_or_init(|| {
        std::env::var(NARROW_STAGGER_ENV)
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(STAGGER_DEFAULT_COLUMNS)
    })
}

/// Whether a new [`NarrowWriter`] of `columns` columns leaves the gaps.
pub fn narrow_stagger(columns: usize) -> bool {
    let at_least = narrow_stagger_columns();
    at_least > 0 && columns >= at_least
}

impl NarrowWriter {
    /// A writer for a trace of `rows` rows and `widths.len()` columns, column
    /// `c` at `widths[c]` bytes (rounded up to 1, 2, 4 or 8); with
    /// [`STAGGER_GAP`] bytes before each column after the first when it has at
    /// least [`NARROW_STAGGER_ENV`]'s columns.
    pub fn new(rows: usize, widths: &[u8]) -> Self {
        let gap = if narrow_stagger(widths.len()) {
            STAGGER_GAP
        } else {
            0
        };
        let mut columns = Vec::with_capacity(widths.len());
        let mut total = 0usize;
        for (c, &w) in widths.iter().enumerate() {
            let width = valid_width(w);
            if c > 0 {
                total += gap;
            }
            columns.push(WriterColumn {
                offset: total,
                width,
                seen: 0,
            });
            total += rows * width;
        }
        Self {
            rows,
            columns,
            // The bytes a group's upload reads: page-locked when that is on.
            data: multilinear::pinned::with_pinned(total, || vec![0u8; total]),
        }
    }

    /// Rows of the trace.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns of the trace.
    pub fn cols(&self) -> usize {
        self.columns.len()
    }

    /// Write `word` into row `row` of column `col`: its low bytes, at the
    /// column's width (see [`NarrowWriter`] for a word that does not fit).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, word: u64) {
        debug_assert!(row < self.rows, "NarrowWriter: row {row} of {}", self.rows);
        let column = &mut self.columns[col];
        column.seen |= word;
        let (width, at) = (column.width, column.offset + row * column.width);
        match width {
            1 => self.data[at] = word as u8,
            2 => self.data[at..at + 2].copy_from_slice(&(word as u16).to_le_bytes()),
            4 => self.data[at..at + 4].copy_from_slice(&(word as u32).to_le_bytes()),
            _ => self.data[at..at + 8].copy_from_slice(&word.to_le_bytes()),
        }
    }

    /// The writer cut into the row ranges `[bounds[i], bounds[i + 1])`, each
    /// writable on its own (a fill split by rows): each part holds its rows of
    /// every column and its own OR of the words it writes, which
    /// [`Self::absorb`] takes back once the parts are done. `bounds` rises from
    /// 0 to the rows; a bound past the rows is taken as the rows.
    pub fn split_rows(&mut self, bounds: &[usize]) -> Vec<NarrowRows<'_>> {
        let rows = self.rows;
        let starts: Vec<usize> = {
            let mut at = bounds.first().copied().unwrap_or(0).min(rows);
            let mut out = vec![at];
            for &end in bounds.iter().skip(1) {
                at = end.clamp(at, rows);
                out.push(at);
            }
            out
        };
        let mut parts: Vec<NarrowRows<'_>> = starts
            .windows(2)
            .map(|w| NarrowRows {
                first_row: w[0],
                rows: w[1] - w[0],
                columns: Vec::with_capacity(self.columns.len()),
                seen: vec![0; self.columns.len()],
            })
            .collect();
        // Columns lie in order from offset 0, a gap before each under the
        // stagger ([`Self::new`]): cut each off the front (its gap dropped),
        // then each column at the parts' rows.
        let mut rest: &mut [u8] = &mut self.data;
        let mut consumed = 0usize;
        let first = starts.first().copied().unwrap_or(0);
        for column in &self.columns {
            let (_, from) = std::mem::take(&mut rest).split_at_mut(column.offset - consumed);
            let (mut col, tail) = from.split_at_mut(rows * column.width);
            consumed = column.offset + rows * column.width;
            rest = tail;
            col = &mut col[first * column.width..];
            for part in &mut parts {
                let (piece, after) =
                    std::mem::take(&mut col).split_at_mut(part.rows * column.width);
                part.columns.push((piece, column.width));
                col = after;
            }
        }
        parts
    }

    /// Take a part's OR of its words ([`NarrowRows::into_seen`]) into the
    /// writer's, as if this writer had written them.
    pub fn absorb(&mut self, seen: &[u64]) {
        for (column, &s) in self.columns.iter_mut().zip(seen) {
            column.seen |= s;
        }
    }

    /// The widths the words written so far need: what [`Self::finish`] gives
    /// the trace when they all fit.
    pub fn needed_widths(&self) -> Vec<u8> {
        self.columns.iter().map(|c| width_of(c.seen)).collect()
    }

    /// The widths the columns are written at.
    pub fn widths(&self) -> Vec<u8> {
        self.columns.iter().map(|c| c.width as u8).collect()
    }

    /// The packed trace, every column narrowed to the bytes its words need; or,
    /// when a word did not fit its column, `Err` with the widths the trace
    /// needs (each at least the width given, for a column that missed).
    pub fn finish(self) -> Result<NarrowMain, Vec<u8>> {
        let need = self.needed_widths();
        if need
            .iter()
            .zip(&self.columns)
            .any(|(&n, c)| n as usize > c.width)
        {
            return Err(need);
        }
        let Self {
            rows,
            columns,
            mut data,
        } = self;
        // The stagger's gaps (0 for a table it leaves back to back).
        let gaps = data.len() - columns.iter().map(|c| rows * c.width).sum::<usize>();
        // Columns move only toward the front (closing the stagger's gaps) and
        // only narrow, so moving them in order never overwrites a byte not
        // yet moved.
        let mut offsets = Vec::with_capacity(columns.len());
        let mut at = 0usize;
        for (&n, c) in need.iter().zip(&columns) {
            let n = n as usize;
            offsets.push(at);
            if at != c.offset || n != c.width {
                move_column(&mut data, rows, c.offset, c.width, at, n);
            }
            at += rows * n;
        }
        // A tail freed by columns that narrowed goes back; the stagger's gaps
        // alone (64 bytes a column) are not worth a reallocation.
        let freed = data.len() - at;
        if freed > 0 {
            data.truncate(at);
            if freed > gaps {
                data.shrink_to_fit();
            }
        }
        let narrow = NarrowMain {
            rows,
            widths: need,
            offsets,
            data,
        };
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            narrow.widths,
            (0..narrow.cols())
                .map(|c| width_of(narrow.column(c).into_iter().fold(0, |m, v| m | v)))
                .collect::<Vec<_>>(),
            "NarrowWriter: a column's width is not its largest word's (a cell written twice?)"
        );
        #[cfg(any(test, feature = "test-utils"))]
        let narrow = mutation::apply(narrow);
        Ok(narrow)
    }
}

/// Move a column of `rows` cells at `from` (`w` bytes each) to `to` (`n` bytes
/// each, `n <= w`), keeping each cell's low `n` bytes. `to <= from`, so the
/// cells are moved front to back.
fn move_column(data: &mut [u8], rows: usize, from: usize, w: usize, to: usize, n: usize) {
    debug_assert!(to <= from && n <= w);
    if n == w {
        data.copy_within(from..from + rows * w, to);
        return;
    }
    fn narrow<const W: usize, const N: usize>(
        data: &mut [u8],
        rows: usize,
        from: usize,
        to: usize,
    ) {
        for r in 0..rows {
            let mut cell = [0u8; W];
            cell.copy_from_slice(&data[from + r * W..from + (r + 1) * W]);
            data[to + r * N..to + (r + 1) * N].copy_from_slice(&cell[..N]);
        }
    }
    match (w, n) {
        (2, 1) => narrow::<2, 1>(data, rows, from, to),
        (4, 1) => narrow::<4, 1>(data, rows, from, to),
        (4, 2) => narrow::<4, 2>(data, rows, from, to),
        (8, 1) => narrow::<8, 1>(data, rows, from, to),
        (8, 2) => narrow::<8, 2>(data, rows, from, to),
        _ => narrow::<8, 4>(data, rows, from, to),
    }
}

/// Test-only faults in [`NarrowWriter::finish`]'s output, per thread: one
/// column given a byte more than it needs, or one column's bytes shifted by a
/// byte. The gates that compare a written trace with the packed 64-bit one
/// must catch both.
#[cfg(any(test, feature = "test-utils"))]
pub mod mutation {
    use super::NarrowMain;
    use std::cell::Cell;

    /// A fault [`set`] arms.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Mutation {
        /// Column `c` at the next width up (a 1-byte column at 2 bytes): the
        /// words are the same, the bytes are not.
        WidenColumn(usize),
        /// Column `c`'s bytes rotated by one: a column read one byte off.
        ShiftColumn(usize),
    }

    thread_local! {
        static ARMED: Cell<Option<Mutation>> = const { Cell::new(None) };
    }

    /// Arm (or with `None`, disarm) a fault for every trace this thread's
    /// writers finish.
    pub fn set(m: Option<Mutation>) {
        ARMED.with(|a| a.set(m));
    }

    pub(super) fn apply(narrow: NarrowMain) -> NarrowMain {
        let Some(m) = ARMED.with(Cell::get) else {
            return narrow;
        };
        let cols = narrow.cols();
        match m {
            Mutation::WidenColumn(c) if c < cols && narrow.widths[c] < 8 => {
                let mut widths = narrow.widths.clone();
                widths[c] *= 2;
                let mut data = Vec::new();
                for (k, &w) in widths.iter().enumerate() {
                    for v in narrow.column(k) {
                        data.extend_from_slice(&v.to_le_bytes()[..w as usize]);
                    }
                }
                NarrowMain::from_parts(narrow.rows, widths, data).unwrap_or(narrow)
            }
            Mutation::ShiftColumn(c) if c < cols && narrow.rows > 1 => {
                let mut narrow = narrow;
                let (off, len) = (narrow.offsets[c], narrow.rows * narrow.widths[c] as usize);
                narrow.data[off..off + len].rotate_left(1);
                narrow
            }
            _ => narrow,
        }
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

    /// Built a block at a time — with columns that need more bytes only in a
    /// later block, and zero rows never pushed — the trace packs to the bytes
    /// [`NarrowMain::pack`] gives the whole trace.
    #[test]
    fn a_trace_built_a_block_at_a_time_packs_as_a_whole() {
        let cols = 6;
        let rows = 3 * 1000 + 37;
        let pushed = rows - 37;
        let data: Vec<u64> = (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                if r >= pushed {
                    return 0;
                }
                let x = (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (8 * c as u32);
                match c {
                    // Grows from 1 byte to 2 in the second block, to 8 in the last.
                    0 if r < 1000 => x & 0xff,
                    0 if r < 2000 => x & 0xffff,
                    0 => x,
                    1 => x & 0xff,
                    2 => 0,
                    3 if r == pushed - 1 => u64::MAX,
                    3 => x & 0xff,
                    _ => x & 0xffff_ffff,
                }
            })
            .collect();
        let whole = NarrowMain::pack(&data, cols);
        for block in [1, 7, 1000, 4096] {
            let mut builder = NarrowBuilder::new(rows, cols);
            for part in data[..pushed * cols].chunks(block * cols) {
                builder.push_rows(part);
            }
            assert_eq!(builder.finish(), whole, "blocks of {block} rows");
        }
        assert_eq!(whole.widths(), &[8, 1, 1, 8, 4, 4]);
    }

    /// The digest sees every packed byte and the shape: equal traces digest
    /// alike, and one flipped bit anywhere (head, middle, the tail past the
    /// last 32-byte block) or another row count changes it.
    #[test]
    fn the_digest_sees_every_byte() {
        let cols = 3;
        let words: Vec<u64> = (0..(1000 * cols) as u64).map(|i| i * 7 % 300).collect();
        let narrow = NarrowMain::pack(&words, cols);
        assert_eq!(narrow.digest(), NarrowMain::pack(&words, cols).digest());
        assert!(
            !narrow.data().len().is_multiple_of(32),
            "a tail past the last block"
        );
        for at in [0, narrow.data().len() / 2, narrow.data().len() - 1] {
            let mut bent = narrow.clone();
            bent.data[at] ^= 1;
            assert_ne!(bent.digest(), narrow.digest(), "byte {at}");
        }
        let fewer = NarrowMain::pack(&words[..999 * cols], cols);
        assert_ne!(fewer.digest(), narrow.digest());
    }

    /// Words with every column width, some cells never written (rows past
    /// `written`): column c tops out at a different byte boundary, in a row
    /// that is not the first.
    fn words_for_the_writer(rows: usize, written: usize) -> (Vec<u64>, usize) {
        let tops = [
            0u64,
            1,
            0xff,
            0x100,
            0xffff,
            0x1_0000,
            0xffff_ffff,
            0x1_0000_0000,
            u64::MAX,
        ];
        let cols = tops.len();
        let words = (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                if r >= written {
                    0
                } else if r == written / 2 {
                    tops[c]
                } else {
                    (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) % (tops[c] / 2 + 1)
                }
            })
            .collect();
        (words, cols)
    }

    /// Write `words` (row-major, `cols` wide) into a writer at `widths`, in
    /// row order or column by column from the last row.
    fn write(words: &[u64], cols: usize, widths: &[u8], scrambled: bool) -> NarrowWriter {
        let rows = words.len() / cols;
        let mut w = NarrowWriter::new(rows, widths);
        let mut cell = |r: usize, c: usize| {
            if words[r * cols + c] != 0 {
                w.set(r, c, words[r * cols + c]);
            }
        };
        if scrambled {
            for c in 0..cols {
                for r in (0..rows).rev() {
                    cell(r, c);
                }
            }
        } else {
            for r in 0..rows {
                for c in 0..cols {
                    cell(r, c);
                }
            }
        }
        w
    }

    /// ★ Written a cell at a time — at the widths the columns need, at more
    /// bytes than they need (each column narrowed in place, every width pair),
    /// at a mix, in any order, with cells never written — the trace finishes
    /// to the bytes [`NarrowMain::pack`] gives it. At fewer bytes than a column
    /// needs it does not finish: it names the widths, and written again at
    /// those it finishes to the same bytes.
    #[test]
    fn a_trace_written_cell_by_cell_packs_as_a_whole() {
        let rows = 3 * PACK_BLOCK_ROWS + 5;
        let (words, cols) = words_for_the_writer(rows, rows - 7);
        let whole = NarrowMain::pack(&words, cols);
        assert_eq!(whole.widths(), &[1, 1, 1, 2, 2, 4, 4, 8, 8]);
        let exact = whole.widths().to_vec();
        let mixed: Vec<u8> = exact
            .iter()
            .enumerate()
            .map(|(c, &w)| if c % 2 == 0 { 8 } else { w })
            .collect();
        let at_least = |floor: u8| exact.iter().map(|&w| w.max(floor)).collect::<Vec<u8>>();
        for widths in [at_least(1), at_least(2), at_least(4), at_least(8), mixed] {
            for scrambled in [false, true] {
                let written = write(&words, cols, &widths, scrambled);
                assert_eq!(written.needed_widths(), exact);
                assert_eq!(
                    written.finish().ok(),
                    Some(whole.clone()),
                    "{widths:?}, scrambled {scrambled}"
                );
            }
        }
        // Too narrow for every column past the first three: refused with the
        // widths, then written at those.
        let missed = write(&words, cols, &[1; 9], false).finish();
        assert_eq!(missed.as_ref().err(), Some(&exact));
        let again = write(&words, cols, &missed.unwrap_err(), true);
        assert_eq!(again.finish().ok(), Some(whole));
    }

    /// `words` written into a writer at `widths` by the row ranges `bounds`,
    /// each part on its own, then each part's OR absorbed (all of them, or all
    /// but `drop_part`'s).
    fn write_split(
        words: &[u64],
        cols: usize,
        widths: &[u8],
        bounds: &[usize],
        drop_part: Option<usize>,
    ) -> NarrowWriter {
        let rows = words.len() / cols;
        let mut w = NarrowWriter::new(rows, widths);
        let seen: Vec<Vec<u64>> = {
            let mut parts = w.split_rows(bounds);
            // Last part first: no part may lean on another's bytes.
            for part in parts.iter_mut().rev() {
                for r in part.range() {
                    for c in 0..cols {
                        if words[r * cols + c] != 0 {
                            part.set(r, c, words[r * cols + c]);
                        }
                    }
                }
            }
            parts.into_iter().map(NarrowRows::into_seen).collect()
        };
        for (p, s) in seen.iter().enumerate() {
            if Some(p) != drop_part {
                w.absorb(s);
            }
        }
        w
    }

    /// ★ Written by row ranges — parts of unequal rows, one part, a part of no
    /// rows, at the widths the columns need, at more bytes, and too narrow —
    /// the writer finishes to the bytes of the same words written in one pass:
    /// the parts' ORs, absorbed, are the one-pass OR. Leaving one part's OR
    /// out narrows a column it alone set the top of, and the bytes change.
    #[test]
    fn a_trace_written_by_row_ranges_is_the_trace_written_in_one_pass() {
        let rows = 3 * PACK_BLOCK_ROWS + 5;
        let (words, cols) = words_for_the_writer(rows, rows - 7);
        let whole = NarrowMain::pack(&words, cols);
        let exact = whole.widths().to_vec();
        let wide = vec![8u8; cols];
        let mid = rows / 2;
        let splits: [Vec<usize>; 5] = [
            vec![0, rows],
            vec![0, 1, rows],
            vec![0, mid, mid, rows],
            vec![0, 17, mid, mid + 3, rows - 1, rows],
            vec![0, rows / 3, 2 * rows / 3, rows],
        ];
        for bounds in &splits {
            for widths in [&exact, &wide] {
                let split = write_split(&words, cols, widths, bounds, None);
                assert_eq!(split.needed_widths(), exact, "{bounds:?}");
                assert_eq!(
                    split.finish().ok(),
                    Some(whole.clone()),
                    "{bounds:?} at {widths:?}"
                );
            }
            let missed = write_split(&words, cols, &[1; 9], bounds, None).finish();
            assert_eq!(missed.as_ref().err(), Some(&exact), "{bounds:?}");
        }
        // The row holding every column's top (`written / 2`) lies in the
        // middle part of a three-way split: without its OR the columns narrow.
        let bounds = [0, rows / 3, 2 * rows / 3, rows];
        let dropped = write_split(&words, cols, &exact, &bounds, Some(1));
        assert_ne!(dropped.needed_widths(), exact);
        assert_ne!(dropped.finish().ok(), Some(whole));
    }

    /// The tests that move the stagger knob, one at a time.
    static STAGGER_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Puts the stagger knob back however a test ends.
    struct StaggerKnob;
    impl Drop for StaggerKnob {
        fn drop(&mut self) {
            set_narrow_stagger_for_test(None);
        }
    }

    /// ★ Under the stagger (a gap of [`STAGGER_GAP`] bytes before each column
    /// after the first while written) the writer finishes to the same bytes:
    /// cell by cell in any order, by row ranges, narrowed, refused and written
    /// again, empty; at a power-of-two row count too. The gaps are there while
    /// it is written, and gone once it finishes. Off, or for a table of fewer
    /// columns than the word, the columns lie back to back.
    #[test]
    fn the_stagger_writes_the_same_bytes() {
        let _one = STAGGER_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let _knob = StaggerKnob;
        let rows = 1 << 12;
        let (words, cols) = words_for_the_writer(rows, rows - 3);
        let whole = NarrowMain::pack(&words, cols);
        let exact = whole.widths().to_vec();
        set_narrow_stagger_for_test(Some(0));
        let plain = NarrowWriter::new(rows, &exact);
        assert_eq!(plain.columns[1].offset, rows * exact[0] as usize);
        // The gate: a table of fewer columns than the word keeps them back to
        // back; one of as many or more leaves the gaps.
        for (at_least, gapped) in [(cols + 1, false), (cols, true), (1, true)] {
            set_narrow_stagger_for_test(Some(at_least));
            let w = NarrowWriter::new(rows, &exact);
            for c in 1..cols {
                let gap = if gapped { c * STAGGER_GAP } else { 0 };
                assert_eq!(
                    w.columns[c].offset,
                    plain.columns[c].offset + gap,
                    "at least {at_least}: column {c}"
                );
            }
        }
        for at_least in [1, cols, cols + 1] {
            set_narrow_stagger_for_test(Some(at_least));
            for scrambled in [false, true] {
                let written = write(&words, cols, &exact, scrambled);
                assert_eq!(
                    written.finish().ok(),
                    Some(whole.clone()),
                    "at least {at_least}, scrambled {scrambled}"
                );
            }
            let bounds = [0, 17, rows / 2, rows / 2, rows - 1, rows];
            let split = write_split(&words, cols, &[8; 9], &bounds, None);
            assert_eq!(
                split.finish().ok(),
                Some(whole.clone()),
                "at least {at_least}: by row ranges"
            );
        }
        // The finish of a staggered table that narrows no column keeps its
        // buffer, gaps' bytes as spare capacity (not worth a reallocation);
        // one that narrows gives the freed tail back. Unstaggered and exact,
        // the buffer is never touched (today's path).
        for (at_least, widths, spare) in [
            (1usize, exact.clone(), (cols - 1) * STAGGER_GAP),
            (1, vec![8u8; cols], 0),
            (0, exact.clone(), 0),
        ] {
            set_narrow_stagger_for_test(Some(at_least));
            let w = write(&words, cols, &widths, false);
            let before = w.data.as_ptr();
            let done = w.finish().expect("fits");
            assert_eq!(done, whole, "at least {at_least}, widths {widths:?}");
            assert_eq!(
                done.data.capacity() - done.data.len(),
                spare,
                "at least {at_least}, widths {widths:?}: spare capacity"
            );
            if spare > 0 || at_least == 0 {
                assert_eq!(
                    done.data.as_ptr(),
                    before,
                    "at least {at_least}: the buffer kept"
                );
            }
        }
        // The writer's own tests, every table staggered.
        set_narrow_stagger_for_test(Some(1));
        a_trace_written_cell_by_cell_packs_as_a_whole();
        a_trace_written_by_row_ranges_is_the_trace_written_in_one_pass();
        the_writer_rounds_widths_up_and_finishes_empty_traces();
        narrow_round_trips_at_every_width_boundary();
    }

    /// The stagger's word: the fewest columns a staggered table has; unset,
    /// 256 (on for the wide tables alone); 0 is off.
    #[test]
    fn the_stagger_knob_is_named_and_on_for_wide_tables_by_default() {
        assert_eq!(NARROW_STAGGER_ENV, "LAMBDA_VM_NARROW_STAGGER");
        assert_eq!(STAGGER_GAP, 64);
        assert_eq!(STAGGER_DEFAULT_COLUMNS, 256);
        if std::env::var(NARROW_STAGGER_ENV).is_err() {
            let _one = STAGGER_TESTS.lock().unwrap_or_else(|e| e.into_inner());
            let _knob = StaggerKnob;
            set_narrow_stagger_for_test(None);
            assert_eq!(
                narrow_stagger_columns(),
                256,
                "on for the wide tables by default"
            );
            assert!(
                narrow_stagger(515) && !narrow_stagger(41),
                "ECDAS staggered, COPY not"
            );
        }
        {
            let _one = STAGGER_TESTS.lock().unwrap_or_else(|e| e.into_inner());
            let _knob = StaggerKnob;
            set_narrow_stagger_for_test(Some(0));
            assert!(!narrow_stagger(10_000), "0 is off");
        }
        let _one = STAGGER_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let _knob = StaggerKnob;
        set_narrow_stagger_for_test(Some(256));
        assert!(!narrow_stagger(255) && narrow_stagger(256) && narrow_stagger(515));
    }

    /// A widths list that is not 1, 2, 4 or 8 is rounded up; a trace never
    /// written, of no rows or of no columns, finishes empty.
    #[test]
    fn the_writer_rounds_widths_up_and_finishes_empty_traces() {
        let w = NarrowWriter::new(10, &[0, 3, 5, 9]);
        assert_eq!(w.widths(), [1, 4, 8, 8]);
        assert_eq!(w.finish().ok(), Some(NarrowMain::pack(&[0; 40], 4)));
        assert_eq!(
            NarrowWriter::new(0, &[2, 4]).finish().ok(),
            Some(NarrowMain::pack(&[], 2))
        );
        assert_eq!(
            NarrowWriter::new(5, &[]).finish().ok(),
            NarrowMain::from_parts(5, Vec::new(), Vec::new())
        );
    }

    /// The test faults change the bytes: a column one width up widens to the
    /// same words, a shifted column does not.
    #[test]
    fn the_writer_faults_change_the_bytes() {
        let rows = 100;
        let (words, cols) = words_for_the_writer(rows, rows);
        let whole = NarrowMain::pack(&words, cols);
        for (m, same_words) in [
            (mutation::Mutation::WidenColumn(3), true),
            (mutation::Mutation::ShiftColumn(5), false),
        ] {
            mutation::set(Some(m));
            let bent = write(&words, cols, whole.widths(), false).finish();
            mutation::set(None);
            let bent = bent.expect("the words fit");
            assert_ne!(bent, whole, "{m:?}");
            assert_ne!(bent.digest(), whole.digest(), "{m:?}");
            let mut back = vec![0u64; words.len()];
            bent.widen_into(&mut back);
            assert_eq!(back == words, same_words, "{m:?}");
        }
        assert_eq!(
            write(&words, cols, whole.widths(), false).finish().ok(),
            Some(whole)
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

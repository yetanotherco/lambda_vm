//! Narrow storage of a main trace: each column at the bytes its values need.
//!
//! A trace's 64-bit words are mostly small (flags, bytes, 16-bit limbs,
//! timestamps under 2^32): on the no-epoch record block the main cells need
//! 2.0 bytes on average instead of 8. [`NarrowMain`] keeps the raw words of a
//! row-major main trace column by column, each column little-endian at the
//! width of its largest word (1, 2, 4 or 8 bytes), so widening gives back the
//! same words bit for bit: no proof byte can depend on it.

use math::page_bytes::PageBytes;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Rows per block of the widen pass: each block reads and writes every column
/// once, so the row-major side is written sequentially.
const BLOCK_ROWS: usize = 1 << 12;

/// Rows per block of the pack pass: a block's words (a few hundred KiB) stay in
/// cache while each column takes its strided share.
const PACK_BLOCK_ROWS: usize = 1 << 10;

/// Where a packed trace's bytes are allocated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backing {
    /// The global allocator's heap.
    #[default]
    Heap,
    /// Page-aligned memory of the trace's own ([`PageBytes`]). The spill store
    /// writes it with `O_DIRECT` as it is, with no aligned copy, and freeing
    /// it returns the pages to the system. The heap where the system refuses
    /// the mapping.
    Pages,
}

/// A packed trace's bytes, as [`Backing`] allocated them.
pub(crate) enum Bytes {
    Heap(Vec<u8>),
    Pages(PageBytes),
}

impl Bytes {
    /// `len` zero bytes.
    fn zeroed(len: usize, backing: Backing) -> Self {
        match backing {
            Backing::Heap => Self::Heap(vec![0u8; len]),
            Backing::Pages => PageBytes::zeroed(len)
                .map(Self::Pages)
                .unwrap_or_else(|_| Self::Heap(vec![0u8; len])),
        }
    }

    /// The bytes as an `O_DIRECT` write takes them whole: a block-aligned
    /// start and length, zeros past the bytes. `None` for heap bytes, which
    /// go through an aligned copy.
    pub(crate) fn direct(&self) -> Option<&[u8]> {
        match self {
            Self::Pages(pages) => {
                let padded = pages.padded();
                let block = PageBytes::BLOCK;
                ((padded.as_ptr() as usize).is_multiple_of(block)
                    && padded.len().is_multiple_of(block))
                .then_some(padded)
            }
            Self::Heap(_) => None,
        }
    }

    /// Whether the bytes are in page-aligned memory of their own.
    pub(crate) fn is_pages(&self) -> bool {
        matches!(self, Self::Pages(_))
    }
}

impl std::ops::Deref for Bytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Heap(bytes) => bytes,
            Self::Pages(pages) => pages,
        }
    }
}

impl std::ops::DerefMut for Bytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Heap(bytes) => bytes,
            Self::Pages(pages) => pages,
        }
    }
}

impl Clone for Bytes {
    /// A copy with the same backing (the heap where the system refuses a new
    /// mapping).
    fn clone(&self) -> Self {
        match self {
            Self::Heap(bytes) => Self::Heap(bytes.clone()),
            Self::Pages(pages) => PageBytes::copy_of(pages)
                .map(Self::Pages)
                .unwrap_or_else(|_| Self::Heap(pages.to_vec())),
        }
    }
}

/// Equal bytes are equal, whatever their backing.
impl PartialEq for Bytes {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for Bytes {}

impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = if self.is_pages() { "Pages" } else { "Heap" };
        write!(f, "{kind}({} bytes)", self.len())
    }
}

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
    data: Bytes,
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
        Self::pack_with(row_major, cols, Backing::Heap)
    }

    /// [`Self::pack`], into bytes allocated as `backing` says.
    pub fn pack_with(row_major: &[u64], cols: usize, backing: Backing) -> Self {
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
        let mut data = Bytes::zeroed(total, backing);

        // Each column's slice, written a block of rows at a time so the
        // block's rows stay in cache across the columns.
        let mut columns: Vec<&mut [u8]> = Vec::with_capacity(cols);
        let mut rest: &mut [u8] = &mut data;
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
        Self::from_bytes(rows, widths, Bytes::Heap(data))
    }

    /// [`Self::from_parts`] with bytes of either backing.
    pub(crate) fn from_bytes(rows: usize, widths: Vec<u8>, data: Bytes) -> Option<Self> {
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

    /// The rows, the column widths and the packed bytes, for a store that
    /// keeps the bytes elsewhere ([`crate::spill`]); [`Self::from_parts`] puts
    /// them back together.
    pub(crate) fn into_parts(self) -> (usize, Vec<u8>, Bytes) {
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

    /// Whether the bytes are in page-aligned memory of their own
    /// ([`Backing::Pages`]).
    pub fn is_page_backed(&self) -> bool {
        self.data.is_pages()
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
    let mut digest = Digester::new(rows, widths, data.len());
    digest.update(data);
    digest.finish()
}

/// [`digest_parts`] taken a piece at a time, in order: the same digest of the
/// same bytes however they are split. The spill store's writer digests each
/// chunk just before it writes it, while the chunk is in cache.
pub(crate) struct Digester {
    lanes: [u64; 4],
    /// Bytes of a 32-byte block not complete yet.
    pending: [u8; 32],
    filled: usize,
}

impl Digester {
    const K: [u64; 4] = [
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
        0x1656_67b1_9e37_79f9,
        0x85eb_ca77_c2b2_ae63,
    ];

    fn mix(acc: u64, w: u64, k: u64) -> u64 {
        (acc ^ w).wrapping_mul(k).rotate_left(31)
    }

    /// A digest of a packed trace of `rows` rows, column widths `widths` and
    /// `len` bytes, before any of the bytes.
    pub(crate) fn new(rows: usize, widths: &[u8], len: usize) -> Self {
        let mut lanes = [
            rows as u64,
            widths.len() as u64,
            len as u64,
            0x243f_6a88_85a3_08d3,
        ];
        for (i, &w) in widths.iter().enumerate() {
            lanes[i % 4] = Self::mix(lanes[i % 4], u64::from(w), Self::K[i % 4]);
        }
        Self {
            lanes,
            pending: [0; 32],
            filled: 0,
        }
    }

    fn block(&mut self, block: &[u8]) {
        for (l, word) in block.chunks_exact(8).enumerate() {
            let w = u64::from_le_bytes(word.try_into().expect("8 bytes"));
            self.lanes[l] = Self::mix(self.lanes[l], w, Self::K[l]);
        }
    }

    /// The next bytes, in order.
    pub(crate) fn update(&mut self, mut bytes: &[u8]) {
        if self.filled > 0 {
            let take = (32 - self.filled).min(bytes.len());
            self.pending[self.filled..self.filled + take].copy_from_slice(&bytes[..take]);
            self.filled += take;
            bytes = &bytes[take..];
            if self.filled < 32 {
                return;
            }
            let block = self.pending;
            self.block(&block);
            self.filled = 0;
        }
        let mut blocks = bytes.chunks_exact(32);
        for block in &mut blocks {
            self.block(block);
        }
        let rest = blocks.remainder();
        self.pending[..rest.len()].copy_from_slice(rest);
        self.filled = rest.len();
    }

    /// The digest: the last block zero-padded (a whole zero block when the
    /// bytes end on a block), then MurmurHash3's finalizer, so every input
    /// bit reaches every output bit.
    pub(crate) fn finish(mut self) -> [u64; 2] {
        let mut tail = [0u8; 32];
        tail[..self.filled].copy_from_slice(&self.pending[..self.filled]);
        self.block(&tail);
        let fmix = |mut h: u64| {
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
            h ^ (h >> 33)
        };
        let lanes = self.lanes;
        [
            fmix(lanes[0] ^ lanes[2].rotate_left(23)),
            fmix(lanes[1] ^ lanes[3].rotate_left(41)),
        ]
    }
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
    /// Where [`Self::finish`] puts the packed trace.
    backing: Backing,
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
            backing: Backing::Heap,
        }
    }

    /// This builder, its packed trace allocated as `backing` says.
    pub fn with_backing(self, backing: Backing) -> Self {
        Self { backing, ..self }
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
        let data = match self.backing {
            Backing::Heap => {
                let mut data = Vec::with_capacity(total);
                for (column, &w) in self.columns.into_iter().zip(&self.widths) {
                    offsets.push(data.len());
                    data.extend_from_slice(&column);
                    data.resize(data.len() + (rows - self.pushed) * w as usize, 0);
                }
                Bytes::Heap(data)
            }
            Backing::Pages => {
                // The rows never pushed stay zero.
                let mut data = Bytes::zeroed(total, Backing::Pages);
                let mut at = 0;
                for (column, &w) in self.columns.into_iter().zip(&self.widths) {
                    offsets.push(at);
                    data[at..at + column.len()].copy_from_slice(&column);
                    at += rows * w as usize;
                }
                data
            }
        };
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

    /// The digest's known answers, taken before it was written as a
    /// [`super::Digester`]: a spilled slot's digest and the read-back check
    /// stay the same function.
    #[test]
    fn the_digest_keeps_its_known_answers() {
        let data: Vec<u8> = (0..1000u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect();
        let kat = [
            (0, [0xd44a_1a81_71af_117b, 0x9757_bf2f_2672_4d91]),
            (31, [0xa3d8_a133_568d_3a18, 0x0e77_d881_b6a8_d234]),
            (32, [0x96b7_cdcb_65ae_daf4, 0x7f84_93e5_11a4_110c]),
            (1000, [0x194f_7331_f65a_1d82, 0xe27c_1f2d_2529_75b4]),
        ];
        for (len, want) in kat {
            assert_eq!(
                super::digest_parts(7, &[1, 2, 4, 8, 2], &data[..len]),
                want,
                "{len}"
            );
        }
    }

    /// The streaming digest is the whole-buffer one however the bytes are
    /// split: pieces of every size around the 32-byte block, empty pieces,
    /// and lengths on and off a block.
    #[test]
    fn the_streaming_digest_is_the_whole_one_however_split() {
        let widths = [1u8, 8, 2, 4];
        for len in [0usize, 1, 31, 32, 33, 64, 4095, 4096, 10_007] {
            let data: Vec<u8> = (0..len)
                .map(|i| {
                    (i as u64)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .rotate_left(17) as u8
                })
                .collect();
            let whole = super::digest_parts(123, &widths, &data);
            for piece in [1usize, 7, 31, 32, 33, 100, 4096] {
                let mut digest = super::Digester::new(123, &widths, len);
                for (i, part) in data.chunks(piece).enumerate() {
                    digest.update(part);
                    if i % 3 == 0 {
                        digest.update(&[]);
                    }
                }
                assert_eq!(digest.finish(), whole, "{len} bytes in pieces of {piece}");
            }
        }
    }

    /// A trace packed into page-aligned memory of its own is the heap-packed
    /// one, byte for byte, from the pack and from the block-at-a-time
    /// builder, and so is its copy.
    #[test]
    fn a_trace_packed_into_pages_is_the_heap_one() {
        let cols = 5;
        let rows = 3000;
        let words: Vec<u64> = (0..(rows * cols) as u64)
            .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (i % 64))
            .collect();
        let heap = NarrowMain::pack(&words, cols);
        let pages = NarrowMain::pack_with(&words, cols, Backing::Pages);
        assert!(!heap.is_page_backed() && pages.is_page_backed());
        assert_eq!(pages, heap);
        assert_eq!(pages.digest(), heap.digest());
        let copy = pages.clone();
        assert!(copy.is_page_backed());
        assert_eq!(copy, heap);
        for backing in [Backing::Heap, Backing::Pages] {
            let mut builder = NarrowBuilder::new(rows + 100, cols).with_backing(backing);
            for block in words.chunks(700 * cols) {
                builder.push_rows(block);
            }
            let built = builder.finish();
            assert_eq!(built.is_page_backed(), backing == Backing::Pages);
            let mut padded = words.clone();
            padded.resize((rows + 100) * cols, 0);
            assert_eq!(built, NarrowMain::pack(&padded, cols), "{backing:?}");
        }
    }

    /// Page-backed bytes offer an `O_DIRECT` write their whole padded span,
    /// block aligned at both ends and zero past the bytes; heap bytes offer
    /// none (alignment checked, not assumed).
    #[test]
    fn only_page_backed_bytes_are_written_directly() {
        for len in [1usize, 4095, 4096, 4097, 100_003] {
            let heap = super::Bytes::zeroed(len, Backing::Heap);
            assert!(heap.direct().is_none(), "{len}");
            let mut pages = super::Bytes::zeroed(len, Backing::Pages);
            pages.iter_mut().for_each(|b| *b = 0xa5);
            let padded = pages.direct().expect("page-backed bytes go direct");
            assert_eq!(padded.as_ptr() as usize % 4096, 0, "{len}: start");
            assert_eq!(padded.len() % 4096, 0, "{len}: length");
            assert!(padded.len() >= len && padded.len() < len + 4096);
            assert!(padded[..len].iter().all(|&b| b == 0xa5));
            assert!(padded[len..].iter().all(|&b| b == 0), "{len}: zero tail");
        }
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

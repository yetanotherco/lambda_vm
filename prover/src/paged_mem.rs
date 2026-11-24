//! Page-bucketed dense memory store: `page_base -> [T; PAGE_SIZE]`.
//!
//! The prover's per-cell memory bookkeeping (the local-to-global `provenance`,
//! and the carried memory `image`) is `O(footprint)` and held across the whole
//! run. A per-cell `HashMap` is wasteful for that: per cell it also stores the
//! 8-byte address key, hashing metadata, and ~30% empty load-factor slack.
//!
//! Measurement (ethrex 1-tx, `bench_continuation footprint`) showed the touched
//! footprint is ~98% two big *contiguous* blocks — i.e. dense. For dense data a
//! flat array indexed by offset is far cheaper: no keys, no hashing, no slack,
//! and cache-friendly. This stores one dense `[T; PAGE_SIZE]` array per touched
//! 256 KB page, in a small `Vec` sorted by page base (few entries — binary-search
//! lookup + sorted insert, no hashing at all; the bulk lives in the arrays).
//!
//! Unset cells read back as `fill` (the genesis/default value) — pages are
//! allocated filled — so callers that only `get`/`set` need no occupancy map.
//! An occupancy bitmap is tracked so [`PagedMem::iter`] can yield exactly the
//! cells that were explicitly `set`.

use std::collections::HashMap;

use crate::tables::page::DEFAULT_PAGE_SIZE;

const WORD_BITS: usize = 64;
const OCC_WORDS: usize = DEFAULT_PAGE_SIZE / WORD_BITS;

struct Page<T> {
    /// Dense values, length `DEFAULT_PAGE_SIZE`, initialized to `fill`.
    data: Box<[T]>,
    /// 1 bit per offset: set iff that offset was explicitly written via `set`.
    occupied: Box<[u64]>,
}

/// A dense, page-bucketed `addr -> T` store. Cheaper than a per-cell `HashMap`
/// when the touched addresses are contiguous. `get` on an unset cell returns
/// the `fill` value supplied at construction.
///
/// The pages themselves are kept in a `Vec` sorted by base address (page bases
/// are sparse across the 64-bit space, so a flat Vec-by-page-number is
/// infeasible, but there are only a handful of touched pages, so binary-search
/// lookup + sorted insert are cheap — and no hashing). The bulk (the cells)
/// lives in each page's dense array.
pub struct PagedMem<T> {
    pages: Vec<(u64, Page<T>)>,
    fill: T,
}

impl<T: Copy> PagedMem<T> {
    /// Create an empty store. Unset cells read back as `fill`.
    pub fn new(fill: T) -> Self {
        Self {
            pages: Vec::new(),
            fill,
        }
    }

    #[inline]
    fn split(addr: u64) -> (u64, usize) {
        // DEFAULT_PAGE_SIZE is a power of two, so the mask isolates the offset.
        let mask = DEFAULT_PAGE_SIZE as u64 - 1;
        (addr & !mask, (addr & mask) as usize)
    }

    /// What a cell never `set` reads.
    #[inline]
    pub fn fill(&self) -> T {
        self.fill
    }

    /// Keeps only the pages whose base `keep` accepts; the rest are dropped,
    /// their cells reading `fill` again. Moves page indices.
    pub fn retain_pages(&mut self, mut keep: impl FnMut(u64) -> bool) {
        self.pages.retain(|(base, _)| keep(*base));
    }

    /// Each allocated page, to be written: distinct pages can be written at
    /// once, from several threads.
    pub fn pages_mut(&mut self) -> impl Iterator<Item = PageMut<'_, T>> {
        self.pages
            .iter_mut()
            .map(|(base, page)| PageMut { base: *base, page })
    }

    /// Value at `addr`, or `fill` if never `set`.
    #[inline]
    pub fn get(&self, addr: u64) -> T {
        let (base, off) = Self::split(addr);
        match self.pages.binary_search_by_key(&base, |(b, _)| *b) {
            Ok(i) => self.pages[i].1.data[off],
            Err(_) => self.fill,
        }
    }

    /// Set `addr` to `val`, allocating its page (filled) on first touch.
    #[inline]
    pub fn set(&mut self, addr: u64, val: T) {
        let (base, off) = Self::split(addr);
        let i = self.page_index_or_insert(base);
        self.page_set(i, off, val);
    }

    /// `addr`'s page base and its offset in the page.
    #[inline]
    pub fn page_of(addr: u64) -> (u64, usize) {
        Self::split(addr)
    }

    /// The index of the page at `base` (page-aligned), if it is allocated. An
    /// index stays valid until a page is allocated.
    #[inline]
    pub fn page_index(&self, base: u64) -> Option<usize> {
        self.pages.binary_search_by_key(&base, |(b, _)| *b).ok()
    }

    /// The index of the page at `base` (page-aligned), allocating it (filled)
    /// on first touch, which moves the index of every page above it.
    #[inline]
    pub fn page_index_or_insert(&mut self, base: u64) -> usize {
        match self.pages.binary_search_by_key(&base, |(b, _)| *b) {
            Ok(i) => i,
            Err(i) => {
                self.pages.insert(
                    i,
                    (
                        base,
                        Page {
                            data: vec![self.fill; DEFAULT_PAGE_SIZE].into_boxed_slice(),
                            occupied: vec![0u64; OCC_WORDS].into_boxed_slice(),
                        },
                    ),
                );
                i
            }
        }
    }

    /// The value at offset `off` of page `index`.
    #[inline]
    pub fn page_get(&self, index: usize, off: usize) -> T {
        self.pages[index].1.data[off]
    }

    /// Sets offset `off` of page `index` to `val`, as [`Self::set`] does.
    #[inline]
    pub fn page_set(&mut self, index: usize, off: usize, val: T) {
        let page = &mut self.pages[index].1;
        page.data[off] = val;
        page.occupied[off / WORD_BITS] |= 1u64 << (off % WORD_BITS);
    }

    /// Sets `values.len()` cells from `addr` on to `values`, in order, as
    /// [`Self::set`] on each would: per page one copy and one run of marks.
    /// A run that wraps the address space is set cell by cell.
    pub fn set_slice(&mut self, addr: u64, values: &[T]) {
        let Some(last) = values.len().checked_sub(1) else {
            return;
        };
        if addr.checked_add(last as u64).is_none() {
            for (i, &value) in values.iter().enumerate() {
                self.set(addr.wrapping_add(i as u64), value);
            }
            return;
        }
        let mut done = 0usize;
        while done < values.len() {
            let (base, off) = Self::split(addr + done as u64);
            let take = (DEFAULT_PAGE_SIZE - off).min(values.len() - done);
            let i = self.page_index_or_insert(base);
            let page = &mut self.pages[i].1;
            page.data[off..off + take].copy_from_slice(&values[done..done + take]);
            mark_run(&mut page.occupied, off, off + take);
            done += take;
        }
    }

    /// The store with each cell's value mapped through `f`, page by page: the
    /// same pages and the same `set` cells, each offset holding `f` of what it
    /// held. `f(fill)` should be `fill` (the new store's), so that an offset
    /// never `set` reads alike in an allocated page and outside every page.
    pub fn map<U: Copy>(&self, fill: U, f: impl Fn(T) -> U) -> PagedMem<U> {
        PagedMem {
            pages: self
                .pages
                .iter()
                .map(|(base, page)| {
                    (
                        *base,
                        Page {
                            data: page.data.iter().map(|&value| f(value)).collect(),
                            occupied: page.occupied.clone(),
                        },
                    )
                })
                .collect(),
            fill,
        }
    }

    /// [`Self::map`] with the pages mapped on the pool, a page a job: the
    /// same store.
    #[cfg(feature = "parallel")]
    pub fn par_map<U: Copy + Send>(&self, fill: U, f: impl Fn(T) -> U + Sync) -> PagedMem<U>
    where
        T: Sync,
    {
        use rayon::prelude::*;
        PagedMem {
            pages: self
                .pages
                .par_iter()
                .map(|(base, page)| {
                    (
                        *base,
                        Page {
                            data: page.data.iter().map(|&value| f(value)).collect(),
                            occupied: page.occupied.clone(),
                        },
                    )
                })
                .collect(),
            fill,
        }
    }

    /// Base addresses of the pages that hold at least one `set` cell, ascending.
    /// (For a `DEFAULT_PAGE_SIZE`-aligned page, this equals `page_base_for_address`
    /// of every cell in it, so it replaces `cells.keys().map(page_base)`.)
    pub fn page_bases(&self) -> impl Iterator<Item = u64> + '_ {
        self.pages.iter().map(|(b, _)| *b)
    }

    /// The dense per-offset data slice for the page at `base` (page-aligned), or
    /// `None` if that page holds no `set` cell. One indexed read per offset (no
    /// hashing) — lets PAGE trace generation read each offset's final value
    /// straight from the store instead of a sparse `FinalStateMap` lookup.
    pub fn page_data(&self, base: u64) -> Option<&[T]> {
        // Unlike `get`/`set`, this searches the raw `base` (no `split`), so an
        // unaligned argument silently misses and returns `None` — which a caller
        // like `collect_bitwise_from_page` would read as "empty page" and skew its
        // ARE_BYTES multiplicities. Surface the misuse instead of hiding it.
        debug_assert!(
            base.is_multiple_of(DEFAULT_PAGE_SIZE as u64),
            "page_data: base must be page-aligned"
        );
        match self.pages.binary_search_by_key(&base, |(b, _)| *b) {
            Ok(i) => Some(&self.pages[i].1.data),
            Err(_) => None,
        }
    }

    /// The bytes its pages take on the heap.
    pub fn heap_bytes(&self) -> usize {
        self.pages.capacity() * std::mem::size_of::<(u64, Page<T>)>()
            + self.pages.len()
                * (DEFAULT_PAGE_SIZE * std::mem::size_of::<T>() + OCC_WORDS * size_of::<u64>())
    }

    /// Number of cells that were explicitly `set`.
    pub fn len(&self) -> usize {
        self.pages
            .iter()
            .map(|(_, p)| {
                p.occupied
                    .iter()
                    .map(|w| w.count_ones() as usize)
                    .sum::<usize>()
            })
            .sum()
    }

    /// True if no cell was ever `set`.
    pub fn is_empty(&self) -> bool {
        self.pages
            .iter()
            .all(|(_, p)| p.occupied.iter().all(|&w| w == 0))
    }

    /// Iterate `(addr, value)` over exactly the cells that were `set`.
    pub fn iter(&self) -> impl Iterator<Item = (u64, T)> + '_ {
        self.pages.iter().flat_map(|(base, page)| {
            let base = *base;
            page.occupied
                .iter()
                .enumerate()
                .flat_map(move |(w, &bits)| {
                    BitIter { bits }.map(move |b| {
                        let off = w * WORD_BITS + b;
                        (base + off as u64, page.data[off])
                    })
                })
        })
    }
}

/// One allocated page of a [`PagedMem`], to be written
/// ([`PagedMem::pages_mut`]).
pub struct PageMut<'a, T> {
    base: u64,
    page: &'a mut Page<T>,
}

impl<T: Copy> PageMut<'_, T> {
    /// The page's base address.
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Sets offset `off` to `val`, as [`PagedMem::page_set`] does.
    #[inline]
    pub fn set(&mut self, off: usize, val: T) {
        self.page.data[off] = val;
        self.page.occupied[off / WORD_BITS] |= 1u64 << (off % WORD_BITS);
    }
}

/// A read-only initial-memory image: `addr -> byte`, with an iterator over the
/// bytes it holds. Implemented for both `HashMap<u64, u8>` (the monolithic
/// prover's image) and [`PagedMem<u8>`] (the continuation's carried image), so
/// trace generation can consume either without changing the monolithic path.
pub trait ImageSource {
    /// Byte at `addr`, or 0 if absent.
    fn image_get(&self, addr: u64) -> u8;
    /// Iterate `(addr, byte)` over every byte present in the image.
    fn image_iter(&self) -> impl Iterator<Item = (u64, u8)> + '_;
    /// The image bucketed into whole pages: each page that holds a byte of
    /// the image, as `DEFAULT_PAGE_SIZE` bytes, each offset the image's byte
    /// or 0. [`init_page_data_by_byte`]'s map.
    fn init_page_data(&self) -> HashMap<u64, Vec<u8>> {
        init_page_data_by_byte(self)
    }
}

/// [`ImageSource::init_page_data`] a byte at a time: each byte the image
/// holds written into its page's array, the page allocated zeroed on its first
/// byte.
pub(crate) fn init_page_data_by_byte<I: ImageSource + ?Sized>(image: &I) -> HashMap<u64, Vec<u8>> {
    let mask = DEFAULT_PAGE_SIZE as u64 - 1;
    let mut init_page_data: HashMap<u64, Vec<u8>> = HashMap::new();
    for (addr, value) in image.image_iter() {
        let page_base = addr & !mask;
        let offset = (addr & mask) as usize;
        let page_data = init_page_data
            .entry(page_base)
            .or_insert_with(|| vec![0u8; DEFAULT_PAGE_SIZE]);
        page_data[offset] = value;
    }
    init_page_data
}

impl ImageSource for HashMap<u64, u8> {
    #[inline]
    fn image_get(&self, addr: u64) -> u8 {
        self.get(&addr).copied().unwrap_or(0)
    }
    fn image_iter(&self) -> impl Iterator<Item = (u64, u8)> + '_ {
        self.iter().map(|(&addr, &byte)| (addr, byte))
    }
}

impl ImageSource for PagedMem<u8> {
    #[inline]
    fn image_get(&self, addr: u64) -> u8 {
        self.get(addr)
    }
    fn image_iter(&self) -> impl Iterator<Item = (u64, u8)> + '_ {
        self.iter()
    }
    /// Each page that holds a `set` cell copied whole. A page is allocated
    /// filled with `fill` and only `set` writes it (marking the cell), so with
    /// `fill` 0 the copy holds each `set` cell's byte and 0 elsewhere: the map
    /// [`init_page_data_by_byte`] builds a byte at a time. Any other `fill`
    /// takes that path.
    fn init_page_data(&self) -> HashMap<u64, Vec<u8>> {
        if self.fill != 0 {
            return init_page_data_by_byte(self);
        }
        self.pages
            .iter()
            .filter(|(_, page)| page.occupied.iter().any(|&bits| bits != 0))
            .map(|(base, page)| (*base, page.data.to_vec()))
            .collect()
    }
}

/// Marks offsets `start..end` of a page's occupancy bitmap.
fn mark_run(occupied: &mut [u64], start: usize, end: usize) {
    let mut off = start;
    while off < end {
        let bit = off % WORD_BITS;
        let n = (WORD_BITS - bit).min(end - off);
        let mask = if n == WORD_BITS {
            u64::MAX
        } else {
            ((1u64 << n) - 1) << bit
        };
        occupied[off / WORD_BITS] |= mask;
        off += n;
    }
}

/// Yields the set-bit indices of a 64-bit word, low to high.
struct BitIter {
    bits: u64,
}

impl Iterator for BitIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.bits == 0 {
            None
        } else {
            let b = self.bits.trailing_zeros() as usize;
            self.bits &= self.bits - 1; // clear lowest set bit
            Some(b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `map` keeps every page and every `set` cell, maps each offset's value,
    /// and reads unset cells as the new fill inside and outside the pages: the
    /// store `set` cell by cell with the mapped values.
    #[test]
    fn a_mapped_store_is_the_store_set_with_the_mapped_values() {
        let page = DEFAULT_PAGE_SIZE as u64;
        let mut bytes = PagedMem::new(0u8);
        let cells = [
            (0x10, 7u8),
            (page - 1, 0),
            (page, 255),
            (5 * page + 3, 1),
            (u64::MAX, 9),
        ];
        for (addr, value) in cells {
            bytes.set(addr, value);
        }
        let mapped = bytes.map((0u8, 0u64), |value| (value, 0));
        let mut set = PagedMem::new((0u8, 0u64));
        for (addr, value) in cells {
            set.set(addr, (value, 0));
        }
        assert_eq!(
            mapped.page_bases().collect::<Vec<_>>(),
            set.page_bases().collect::<Vec<_>>()
        );
        for base in set.page_bases() {
            assert_eq!(
                mapped.page_data(base),
                set.page_data(base),
                "page {base:#x}"
            );
        }
        assert_eq!(
            mapped.iter().collect::<Vec<_>>(),
            set.iter().collect::<Vec<_>>()
        );
        assert_eq!(mapped.len(), cells.len());
        assert_eq!(mapped.get(page + 1), (0, 0));
        assert_eq!(mapped.get(3 * page), (0, 0));
    }

    /// A store's pages, each page's offsets and its set cells.
    #[allow(clippy::type_complexity)]
    fn parts<T: Copy>(m: &PagedMem<T>) -> (Vec<u64>, Vec<Vec<T>>, Vec<(u64, T)>) {
        let bases: Vec<u64> = m.page_bases().collect();
        let data = bases
            .iter()
            .map(|&b| m.page_data(b).expect("a page").to_vec())
            .collect();
        (bases, data, m.iter().collect())
    }

    /// A run set at once ([`PagedMem::set_slice`]) is the run set a cell at a
    /// time: within a page, across one and two page ends, a whole page,
    /// overwriting earlier cells (the later wins), wrapping the address space,
    /// empty, and on a store whose fill is not 0; the bitmap's word edges
    /// included.
    #[test]
    fn a_run_set_at_once_is_the_run_set_a_cell_at_a_time() {
        let page = DEFAULT_PAGE_SIZE as u64;
        let run = |n: usize, seed: u8| -> Vec<u8> {
            (0..n).map(|i| (i as u8).wrapping_mul(31) ^ seed).collect()
        };
        let runs: Vec<(u64, Vec<u8>)> = vec![
            (0x10, run(100, 1)),
            (63, run(2, 2)),
            (64, run(64, 3)),
            (0x40, run(5, 4)),
            (page - 3, run(10, 5)),
            (3 * page - 1, run(DEFAULT_PAGE_SIZE + 2, 6)),
            (8 * page, run(DEFAULT_PAGE_SIZE, 7)),
            (0x20, run(0, 8)),
            (u64::MAX - 3, run(9, 9)),
        ];
        for fill in [0u8, 7] {
            let mut at_once = PagedMem::new(fill);
            let mut by_cell = PagedMem::new(fill);
            for (addr, values) in &runs {
                at_once.set_slice(*addr, values);
                for (i, &v) in values.iter().enumerate() {
                    by_cell.set(addr.wrapping_add(i as u64), v);
                }
            }
            assert_eq!(parts(&at_once), parts(&by_cell), "fill {fill}");
            assert_eq!(at_once.len(), by_cell.len());
        }
    }

    /// The store mapped on the pool is the store mapped page by page.
    #[cfg(feature = "parallel")]
    #[test]
    fn a_store_mapped_on_the_pool_is_the_store_mapped_in_turn() {
        let page = DEFAULT_PAGE_SIZE as u64;
        let mut bytes = PagedMem::new(0u8);
        for (i, addr) in [0x10, page - 1, 2 * page + 5, 9 * page, u64::MAX]
            .into_iter()
            .enumerate()
        {
            bytes.set(addr, i as u8 + 1);
        }
        bytes.set_slice(4 * page, &[3u8; 1000]);
        let f = |value: u8| (value, u64::from(value) * 3);
        assert_eq!(
            parts(&bytes.par_map((0, 0), f)),
            parts(&bytes.map((0, 0), f))
        );
    }

    /// The whole-page copy ([`PagedMem::init_page_data`]) is the map built a
    /// byte at a time, and the map a `HashMap` image of the same bytes
    /// gives: on a sparse page, a dense page, a page whose only set bytes are
    /// zero, a page's last byte, the top of the address space, an allocated
    /// page with no set byte (in neither map) and a store whose fill is not 0
    /// (the byte path, so its unset offsets read 0 as there).
    #[test]
    fn whole_pages_are_the_pages_built_a_byte_at_a_time() {
        let page = DEFAULT_PAGE_SIZE as u64;
        let mut cells: Vec<(u64, u8)> = vec![(0x10, 7), (0x11, 0), (page - 1, 200)];
        cells.extend((0..page).map(|off| (2 * page + off, (off * 13 + 1) as u8)));
        cells.extend([(4 * page + 5, 0), (4 * page + 9, 0)]);
        cells.extend([(u64::MAX - 1, 3), (u64::MAX, 9)]);
        for fill in [0u8, 7] {
            let mut image = PagedMem::new(fill);
            let mut bytes = HashMap::new();
            for &(addr, value) in &cells {
                image.set(addr, value);
                bytes.insert(addr, value);
            }
            let empty = image.page_index_or_insert(6 * page);
            assert_eq!(image.pages[empty].0, 6 * page);
            let whole = image.init_page_data();
            assert_eq!(whole, init_page_data_by_byte(&image), "fill {fill}");
            assert_eq!(whole, init_page_data_by_byte(&bytes), "fill {fill}");
            assert_eq!(whole, bytes.init_page_data(), "fill {fill}");
            let mut bases: Vec<u64> = whole.keys().copied().collect();
            bases.sort_unstable();
            assert_eq!(
                bases,
                [0, 2 * page, 4 * page, u64::MAX - (page - 1)],
                "fill {fill}"
            );
            assert!(whole.values().all(|data| data.len() == DEFAULT_PAGE_SIZE));
            assert_eq!(whole[&0][0x10], 7);
            assert_eq!(whole[&0][0x12], 0, "an unset offset reads 0");
            assert_eq!(whole[&0][DEFAULT_PAGE_SIZE - 1], 200);
            assert!(whole[&(4 * page)].iter().all(|&b| b == 0));
            assert_eq!(whole[&(u64::MAX - (page - 1))][DEFAULT_PAGE_SIZE - 1], 9);
        }
    }
}

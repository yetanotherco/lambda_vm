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
//! 32 KB page, in a small `Vec` sorted by page base (few entries — binary-search
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
        let i = match self.pages.binary_search_by_key(&base, |(b, _)| *b) {
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
        };
        let page = &mut self.pages[i].1;
        page.data[off] = val;
        page.occupied[off / WORD_BITS] |= 1u64 << (off % WORD_BITS);
    }

    /// Base addresses of the pages that hold at least one `set` cell, ascending.
    /// (For a `DEFAULT_PAGE_SIZE`-aligned page, this equals `page_base_for_address`
    /// of every cell in it, so it replaces `cells.keys().map(page_base)`.)
    pub fn page_bases(&self) -> impl Iterator<Item = u64> + '_ {
        self.pages.iter().map(|(b, _)| *b)
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

/// A read-only initial-memory image: `addr -> byte`, with an iterator over the
/// bytes it holds. Implemented for both `HashMap<u64, u8>` (the monolithic
/// prover's image) and [`PagedMem<u8>`] (the continuation's carried image), so
/// trace generation can consume either without changing the monolithic path.
pub trait ImageSource {
    /// Byte at `addr`, or 0 if absent.
    fn image_get(&self, addr: u64) -> u8;
    /// Iterate `(addr, byte)` over every byte present in the image.
    fn image_iter(&self) -> impl Iterator<Item = (u64, u8)> + '_;
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

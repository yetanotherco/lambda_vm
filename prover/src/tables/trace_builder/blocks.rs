//! Lists appended to window by window without ever moving what they hold.
//!
//! A builder list grown with `Vec::extend` doubles its capacity when it fills:
//! a new buffer of twice the size, the old one copied over, then freed. At the
//! end of a large block's walk that is one allocation as large as the list and
//! a multi-GiB copy on the accumulator thread (RYZEN 015 at 195 Mgas: +24.2 GiB
//! live in one step, into the cgroup's limit). A [`BlockVec`] stores its
//! elements in blocks of a fixed length instead: the first grows as a `Vec`
//! does up to that length, every later one is allocated whole, and a full
//! block is never touched again. A block holds at most [`BLOCK_BYTES`] (or a
//! table's chunk, when that is smaller), so no single allocation, and no
//! copy, is larger. The elements and their order are the list's: readers see
//! it as the concatenation of its blocks ([`Segmented`]), and a chunk the
//! block length divides lies in one block.

use std::borrow::Cow;
use std::collections::VecDeque;

use super::{Part, Segmented};

/// The most bytes one block holds (rounded down to a power-of-two element
/// count): at or above the allocator's 8 MiB class, whose freed extents other
/// threads reuse.
pub(super) const BLOCK_BYTES: usize = 64 << 20;

/// Elements a block of `T` holds for a table chunked at `chunk` rows: the
/// largest power of two that fits [`BLOCK_BYTES`], and no more than `chunk`,
/// so that a power-of-two chunk is whole blocks (one block at the production
/// chunks for ops of ≤ 32 bytes). At least one.
pub(super) fn block_len<T>(chunk: usize) -> usize {
    let fits = (BLOCK_BYTES / std::mem::size_of::<T>().max(1)).max(1);
    let pow2 = 1usize << (usize::BITS - 1 - fits.leading_zeros());
    pow2.min(chunk.max(1))
}

/// A list stored in blocks of `block` elements: see the module docs.
pub(super) struct BlockVec<T> {
    blocks: VecDeque<Vec<T>>,
    /// Elements in a full block.
    block: usize,
    len: usize,
}

impl<T> BlockVec<T> {
    /// An empty list of `block`-element blocks (at least one).
    pub(super) fn new(block: usize) -> Self {
        Self {
            blocks: VecDeque::new(),
            block: block.max(1),
            len: 0,
        }
    }

    /// `v` as a list of one block, as it is (no copy): a list built whole.
    pub(super) fn from_vec(v: Vec<T>) -> Self {
        let len = v.len();
        let mut blocks = VecDeque::new();
        if len > 0 {
            blocks.push_back(v);
        }
        Self {
            blocks,
            block: len.max(1),
            len,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Appends `x`: into the last block while it has room (growing it, up to
    /// the block length, as a `Vec` grows), else into a new block allocated
    /// whole. Nothing already held moves once its block is full.
    pub(super) fn push(&mut self, x: T) {
        match self.blocks.back_mut() {
            Some(last) if last.len() < self.block => {
                if last.len() == last.capacity() {
                    let want = (last.capacity() * 2).max(16).min(self.block);
                    last.reserve_exact(want - last.len());
                }
                last.push(x);
            }
            _ => {
                // The first block starts small; once one block is full the
                // list is a long one, and each next block is allocated whole.
                let capacity = if self.blocks.is_empty() {
                    16.min(self.block)
                } else {
                    self.block
                };
                let mut block = Vec::with_capacity(capacity);
                block.push(x);
                self.blocks.push_back(block);
            }
        }
        self.len += 1;
    }

    /// Appends every element of `items`, in order.
    pub(super) fn extend(&mut self, items: impl IntoIterator<Item = T>) {
        for x in items {
            self.push(x);
        }
    }

    /// `other`'s elements after this list's, by moving its blocks (a list
    /// assembled from segments; no element is copied).
    pub(super) fn chain(mut self, other: Self) -> Self {
        self.len += other.len;
        self.blocks.extend(other.blocks);
        self
    }

    /// The blocks, in order: the list is their concatenation.
    pub(super) fn parts(&self) -> impl Iterator<Item = &[T]> {
        self.blocks.iter().map(Vec::as_slice)
    }

    /// The elements, in order.
    pub(super) fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.blocks.iter().flat_map(|block| block.iter())
    }

    /// The list as the segments it is the concatenation of.
    pub(super) fn segments(&self) -> Segmented<'_, T> {
        Segmented {
            parts: self.parts().map(Part::Ops).collect(),
        }
    }

    /// The first `n` elements (at most the list), taken out: the front block
    /// as it is when it holds exactly `n` (a chunk the block length is),
    /// copied out of the front blocks otherwise.
    pub(super) fn take_front(&mut self, n: usize) -> Vec<T> {
        let n = n.min(self.len);
        self.len -= n;
        if self.blocks.front().is_some_and(|front| front.len() == n) {
            return self.blocks.pop_front().unwrap_or_default();
        }
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let Some(front) = self.blocks.front_mut() else {
                break;
            };
            let want = n - out.len();
            if front.len() <= want {
                out.append(front);
                self.blocks.pop_front();
            } else {
                out.extend(front.drain(..want));
            }
        }
        out
    }

    /// The bytes its blocks take on the heap (capacities).
    pub(super) fn heap_bytes(&self) -> usize {
        self.blocks.iter().map(Vec::capacity).sum::<usize>() * std::mem::size_of::<T>()
            + self.blocks.capacity() * std::mem::size_of::<Vec<T>>()
    }

    /// The bytes its largest single allocation takes (a block's capacity).
    pub(super) fn largest_bytes(&self) -> usize {
        self.blocks.iter().map(Vec::capacity).max().unwrap_or(0) * std::mem::size_of::<T>()
    }
}

impl<T: Clone> BlockVec<T> {
    /// Elements `start..end` (clamped to the list): borrowed when one block
    /// holds them, copied otherwise.
    pub(super) fn range(&self, start: usize, end: usize) -> Cow<'_, [T]> {
        let end = end.min(self.len);
        if start >= end {
            return Cow::Borrowed(&[]);
        }
        self.segments().range(start, end)
    }

    /// The whole list as one slice: borrowed when it is one block, copied
    /// otherwise.
    pub(super) fn whole(&self) -> Cow<'_, [T]> {
        self.range(0, self.len)
    }

    /// The elements as one `Vec`.
    #[cfg(test)]
    pub(super) fn to_vec(&self) -> Vec<T> {
        self.whole().into_owned()
    }

    /// The elements as one `Vec`: the block itself when there is one, the
    /// blocks concatenated (each freed as it is copied) otherwise.
    pub(super) fn into_vec(mut self) -> Vec<T> {
        if self.blocks.len() <= 1 {
            return self.blocks.pop_front().unwrap_or_default();
        }
        let mut out = Vec::with_capacity(self.len);
        while let Some(mut block) = self.blocks.pop_front() {
            out.append(&mut block);
        }
        out
    }
}

impl<T> Default for BlockVec<T> {
    /// An empty list of single-element blocks: a list nothing is appended to.
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pushed one at a time, or appended a window at a time, a list holds the
    /// elements in order, every block but the last full at the block length,
    /// no block larger; and any range, and chunk k of any chunk size, is the
    /// `Vec`'s. Empty lists, exactly full blocks and one-element blocks too.
    #[test]
    fn a_block_list_is_the_vec_it_was_appended() {
        for block in [1usize, 2, 3, 7, 16, 64] {
            for n in [
                0usize,
                1,
                block - 1,
                block,
                block + 1,
                2 * block,
                5 * block + 3,
                1000,
            ] {
                let want: Vec<u64> = (0..n as u64).map(|i| (i * 0x9e37_79b9) ^ 5).collect();
                let mut pushed = BlockVec::new(block);
                for &x in &want {
                    pushed.push(x);
                }
                let mut windows = BlockVec::new(block);
                for window in want.chunks(13) {
                    windows.extend(window.iter().copied());
                }
                for list in [&pushed, &windows] {
                    assert_eq!(list.len(), n);
                    assert_eq!(list.to_vec(), want, "block {block} n {n}");
                    assert_eq!(list.iter().copied().collect::<Vec<_>>(), want);
                    let lens: Vec<usize> = list.parts().map(<[u64]>::len).collect();
                    let full = lens.len().saturating_sub(1);
                    assert!(
                        lens[..full].iter().all(|&l| l == block),
                        "block {block} n {n}: {lens:?}"
                    );
                    assert!(
                        lens.iter().all(|&l| 0 < l && l <= block),
                        "block {block} n {n}: {lens:?}"
                    );
                    assert!(list.largest_bytes() <= block.max(16) * 8);
                    for chunk in [1usize, 2, block, 2 * block, 3 * block + 1, 4096] {
                        for (k, expected) in want.chunks(chunk).enumerate() {
                            let got = list.range(k * chunk, (k + 1) * chunk);
                            assert_eq!(&*got, expected, "block {block} n {n} chunk {chunk} k {k}");
                            // A chunk the block length divides lies in one block.
                            if chunk % block == 0 && chunk == block {
                                assert!(matches!(got, Cow::Borrowed(_)), "chunk {k} copied");
                            }
                        }
                    }
                    assert!(list.range(n, n + 5).is_empty());
                }
            }
        }
    }

    /// Lists chained and taken from the front keep the order; a front block
    /// that is exactly the chunk taken moves out whole.
    #[test]
    fn chained_and_taken_from_the_front_in_order() {
        let a: Vec<u32> = (0..50).collect();
        let b: Vec<u32> = (50..77).collect();
        let mut first = BlockVec::new(8);
        first.extend(a.iter().copied());
        let mut second = BlockVec::new(8);
        second.extend(b.iter().copied());
        let mut all = first.chain(second).chain(BlockVec::from_vec(vec![77, 78]));
        assert_eq!(all.to_vec(), (0..79).collect::<Vec<u32>>());
        let front = all.take_front(8);
        assert_eq!(front, (0..8).collect::<Vec<u32>>());
        let mut taken = front;
        while !all.is_empty() {
            taken.extend(all.take_front(5));
        }
        assert_eq!(taken, (0..79).collect::<Vec<u32>>());
        assert!(all.take_front(3).is_empty());
        assert_eq!(all.len(), 0);
    }

    /// The block length: a power of two that fits 64 MiB, no more than the
    /// chunk; 2^21 for 32-byte ops at the production chunk.
    #[test]
    fn a_block_is_at_most_64_mib_and_at_most_a_chunk() {
        assert_eq!(block_len::<[u8; 32]>(1 << 21), 1 << 21);
        assert_eq!(block_len::<[u8; 24]>(1 << 21), 1 << 21);
        assert_eq!(block_len::<[u8; 40]>(1 << 21), 1 << 20);
        assert_eq!(block_len::<[u8; 1870]>(1 << 21), 1 << 15);
        assert_eq!(block_len::<u64>(24), 24);
        assert_eq!(block_len::<u64>(0), 1);
        for size_log2 in 0..12 {
            let bytes = 1usize << size_log2;
            let len = BLOCK_BYTES / bytes;
            assert!(len.is_power_of_two() && len * bytes <= BLOCK_BYTES);
        }
    }
}

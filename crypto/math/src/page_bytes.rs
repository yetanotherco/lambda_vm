//! Bytes in page-aligned memory of their own, outside the global allocator.
//!
//! A [`PageBytes`] is an anonymous mapping rounded up to whole
//! [`PageBytes::BLOCK`]s. Its start is page aligned and the bytes past its
//! length stay zero, so an `O_DIRECT` write can take [`PageBytes::padded`]
//! whole, with no aligned copy first. Dropping it unmaps the pages: they go
//! back to the system at once instead of into an allocator's free lists.

use core::fmt;
use core::ops::{Deref, DerefMut};

/// Zero-initialised bytes in an anonymous mapping of their own (see the
/// module docs). Not `Clone`: a copy is [`Self::copy_of`], which can fail.
pub struct PageBytes {
    map: memmap2::MmapMut,
    len: usize,
}

impl PageBytes {
    /// The alignment of the start and of [`Self::padded`]'s length: a disk
    /// block, and a divisor of every page size.
    pub const BLOCK: usize = 4096;

    /// `len` zero bytes. The pages are the system's zero pages until written.
    pub fn zeroed(len: usize) -> std::io::Result<Self> {
        let span = len.div_ceil(Self::BLOCK) * Self::BLOCK;
        let map = memmap2::MmapMut::map_anon(span)?;
        debug_assert!(span == 0 || (map.as_ptr() as usize).is_multiple_of(Self::BLOCK));
        Ok(Self { map, len })
    }

    /// The bytes as a copy in new pages.
    pub fn copy_of(bytes: &[u8]) -> std::io::Result<Self> {
        let mut out = Self::zeroed(bytes.len())?;
        out.copy_from_slice(bytes);
        Ok(out)
    }

    /// The number of bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bytes and the zeros after them up to the next [`Self::BLOCK`]:
    /// starts and ends block aligned.
    pub fn padded(&self) -> &[u8] {
        &self.map[..]
    }
}

impl Deref for PageBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.map[..self.len]
    }
}

impl DerefMut for PageBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.map[..self.len]
    }
}

impl PartialEq for PageBytes {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for PageBytes {}

impl fmt::Debug for PageBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PageBytes")
            .field("len", &self.len)
            .field("padded", &self.map.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::PageBytes;

    /// The start and the padded length are block aligned, the bytes start as
    /// zero, the tail past the length stays zero after every byte is written,
    /// and a copy is equal and in pages of its own.
    #[test]
    fn page_bytes_are_aligned_zeroed_and_padded_with_zeros() {
        for len in [1, 4095, 4096, 4097, 3 * 4096 + 17, 1 << 20] {
            let mut bytes = PageBytes::zeroed(len).unwrap();
            assert_eq!(bytes.len(), len);
            assert!(bytes.iter().all(|&b| b == 0), "{len}: zeroed");
            bytes
                .iter_mut()
                .enumerate()
                .for_each(|(i, b)| *b = (i % 251) as u8 | 1);
            let padded = bytes.padded();
            assert_eq!(
                padded.as_ptr() as usize % PageBytes::BLOCK,
                0,
                "{len}: start"
            );
            assert_eq!(padded.len() % PageBytes::BLOCK, 0, "{len}: padded length");
            assert_eq!(
                padded.len(),
                len.div_ceil(PageBytes::BLOCK) * PageBytes::BLOCK
            );
            assert!(padded[len..].iter().all(|&b| b == 0), "{len}: zero tail");
            let copy = PageBytes::copy_of(&bytes).unwrap();
            assert_eq!(copy, bytes);
            assert_ne!(copy.as_ptr(), bytes.as_ptr());
        }
        let empty = PageBytes::zeroed(0).unwrap();
        assert!(empty.is_empty() && empty.padded().is_empty());
    }
}

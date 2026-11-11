//! The fast engine's guest memory ([`FlatMemory`]): the address ranges a guest
//! lives in as two flat, lazily zeroed host regions addressed directly, and
//! today's paged store ([`Pages`]) for every other address.
//!
//! It answers every access the executor's [`Memory`](super::memory::Memory)
//! answers, with the same bytes and the same errors: the access checks live in
//! [`GuestMemory`]'s provided methods, which both memories share; this type only
//! moves bytes.
//!
//! Window-start states. The walkers read the memory as it was at each window's
//! start, as a [`MemorySnapshot`]. A flat region cannot be shared copy-on-write,
//! so the memory keeps a paged mirror: every 4 KiB chunk of a flat region
//! written since the last snapshot is marked, and [`FlatMemory::snapshot`]
//! copies the marked chunks into the mirror (a page a snapshot still holds is
//! replaced, not changed) and publishes a clone of it. The mirror is the same
//! [`Pages`] that holds the addresses outside the flat regions, so one clone is
//! the whole memory. No other thread ever reads the flat regions.
use super::memory::{
    GuestMemory, MAX_PRIVATE_INPUT_SIZE, MemoryError, MemorySnapshot, PAGE_BITS, PAGE_SIZE,
    PRIVATE_INPUT_START_INDEX, Pages,
};

/// Bytes a dirty mark covers.
const CHUNK_BITS: u32 = 12;
/// Chunks a mirror page holds (16 for 64 KiB pages and 4 KiB chunks).
const CHUNKS_PER_PAGE: usize = 1 << (PAGE_BITS - CHUNK_BITS);
const _: () = assert!(CHUNKS_PER_PAGE == 16, "a page's chunk mask is a u16");

/// Where the two flat regions lie: the low one is `[0, low_end)`, the high one
/// `[high_base, 2^64)`. Both bounds are multiples of a mirror page and each
/// region holds at least one page, so a flat page is never split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlatLayout {
    pub low_end: u64,
    pub high_base: u64,
}

impl FlatLayout {
    /// The guest's ranges: code, data, the heap (to `0xC000_0000`) and the
    /// private input (`0xFF00_0000` plus at most 512 MiB) in the low region
    /// (4.5 GiB), the stack (from `STACK_TOP` down) in the top 256 MiB.
    /// Lazily zeroed: only the pages a run touches take host memory.
    pub const GUEST: Self = Self {
        low_end: 0x1_2000_0000,
        high_base: 0u64.wrapping_sub(1 << 28),
    };

    fn valid(self) -> bool {
        let page = PAGE_SIZE as u64;
        self.low_end >= page
            && self.low_end.is_multiple_of(page)
            && self.high_base.is_multiple_of(page)
            && self.high_base != 0
            && self.low_end <= self.high_base
            && 0u64.wrapping_sub(self.high_base) <= (usize::MAX as u64)
            && self.low_end <= (usize::MAX as u64)
    }

    /// The pieces `address..address + len` falls into, region by region: for
    /// each, where it starts in the range, its length, and where it lies.
    fn split(self, address: u64, len: usize, mut piece: impl FnMut(usize, usize, Place)) {
        let mut done = 0;
        while done < len {
            let at = address + done as u64;
            let left = len - done;
            let (n, place) = if at < self.low_end {
                let n = left.min((self.low_end - at) as usize);
                (n, Place::Low(at as usize))
            } else if at >= self.high_base {
                (left, Place::High((at - self.high_base) as usize))
            } else {
                let n = left.min((self.high_base - at) as usize);
                (n, Place::Pages(at))
            };
            piece(done, n, place);
            done += n;
        }
    }

    fn low_len(self) -> usize {
        self.low_end as usize
    }

    fn high_len(self) -> usize {
        0u64.wrapping_sub(self.high_base) as usize
    }
}

const _: () = assert!(
    PRIVATE_INPUT_START_INDEX + 4 + MAX_PRIVATE_INPUT_SIZE <= FlatLayout::GUEST.low_end,
    "the private input sits in the low region"
);

/// A flat host region, zero until written.
struct Region {
    ptr: *mut u8,
    len: usize,
}

// The region is owned memory with no interior sharing: moving it to another
// thread moves the only handle to it.
unsafe impl Send for Region {}

impl Region {
    /// `len` bytes of zeroed address space. On Unix an anonymous private
    /// mapping, so untouched pages cost nothing and come back zero; a heap
    /// allocation elsewhere.
    fn new(len: usize) -> Result<Self, MemoryError> {
        #[cfg(unix)]
        {
            // SAFETY: an anonymous mapping of a fresh range, no file, no
            // fixed address; the result is checked.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON | map_noreserve(),
                    -1,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(MemoryError::AllocationFailed);
            }
            Ok(Self {
                ptr: ptr.cast(),
                len,
            })
        }
        #[cfg(not(unix))]
        {
            let layout = std::alloc::Layout::from_size_align(len, PAGE_SIZE)
                .map_err(|_| MemoryError::AllocationFailed)?;
            // SAFETY: a non-zero size (a region holds at least one page).
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                return Err(MemoryError::AllocationFailed);
            }
            Ok(Self { ptr, len })
        }
    }

    fn bytes(&self, offset: usize, len: usize) -> &[u8] {
        assert!(offset <= self.len && len <= self.len - offset);
        // SAFETY: in bounds (checked above) of a live allocation.
        unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len) }
    }

    fn bytes_mut(&mut self, offset: usize, len: usize) -> &mut [u8] {
        assert!(offset <= self.len && len <= self.len - offset);
        // SAFETY: in bounds (checked above) of a live allocation we own
        // exclusively (`&mut self`).
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), len) }
    }
}

#[cfg(target_os = "linux")]
fn map_noreserve() -> libc::c_int {
    libc::MAP_NORESERVE
}

#[cfg(all(unix, not(target_os = "linux")))]
fn map_noreserve() -> libc::c_int {
    0
}

impl Drop for Region {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: the mapping this region made, unmapped once.
        unsafe {
            libc::munmap(self.ptr.cast(), self.len);
        }
        #[cfg(not(unix))]
        // SAFETY: the allocation this region made, with its layout.
        unsafe {
            std::alloc::dealloc(
                self.ptr,
                std::alloc::Layout::from_size_align_unchecked(self.len, PAGE_SIZE),
            );
        }
    }
}

/// Guest memory as two flat regions and a paged store for the rest (see the
/// module docs).
pub struct FlatMemory {
    layout: FlatLayout,
    low: Region,
    high: Region,
    /// Chunks in the low region; the high region's chunks follow them.
    low_chunks: usize,
    /// The addresses outside the flat regions, and the flat regions as of the
    /// last snapshot.
    pages: Pages,
    /// One bit per flat chunk: written since the last snapshot.
    dirty_bits: Vec<u64>,
    /// The chunks whose bit is set, in the order first written.
    dirty: Vec<u32>,
    /// Flat pages the mirror holds (what the flat regions have in use, as
    /// [`FlatMemory::heap_bytes`] counts it).
    flat_pages: usize,
    public_output: Vec<u8>,
}

impl std::fmt::Debug for FlatMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlatMemory")
            .field("layout", &self.layout)
            .field("pages", &self.pages)
            .field("dirty", &self.dirty.len())
            .finish()
    }
}

impl FlatMemory {
    /// An empty memory over the given layout.
    pub fn new(layout: FlatLayout) -> Result<Self, MemoryError> {
        if !layout.valid() {
            return Err(MemoryError::AllocationFailed);
        }
        let low_chunks = layout.low_len() >> CHUNK_BITS;
        let chunks = low_chunks + (layout.high_len() >> CHUNK_BITS);
        if u32::try_from(chunks).is_err() {
            return Err(MemoryError::AllocationFailed);
        }
        Ok(Self {
            layout,
            low: Region::new(layout.low_len())?,
            high: Region::new(layout.high_len())?,
            low_chunks,
            pages: Pages::default(),
            dirty_bits: vec![0u64; chunks.div_ceil(64)],
            dirty: Vec::new(),
            flat_pages: 0,
            public_output: Vec::new(),
        })
    }

    /// The layout the memory was made with.
    pub fn layout(&self) -> FlatLayout {
        self.layout
    }

    /// The memory as it is now ([`MemorySnapshot`]): the flat chunks written
    /// since the last snapshot copied into the mirror, then the mirror shared.
    pub fn snapshot(&mut self) -> MemorySnapshot {
        self.sync();
        MemorySnapshot::from_pages(self.pages.clone())
    }

    /// The bytes the memory takes on the heap: the mirror (the paged store),
    /// the flat pages in use (each counted once more, for its flat copy) and
    /// the public output.
    pub fn heap_bytes(&self) -> usize {
        self.pages.heap_bytes()
            + self.flat_pages * PAGE_SIZE
            + self.dirty_bits.capacity() * 8
            + self.dirty.capacity() * 4
            + self.public_output.capacity()
    }

    /// The bytes committed so far.
    pub fn public_output_bytes(&self) -> &[u8] {
        &self.public_output
    }

    /// Copies every chunk written since the last snapshot into the mirror.
    fn sync(&mut self) {
        let mut dirty = std::mem::take(&mut self.dirty);
        dirty.sort_unstable();
        let mut i = 0;
        while i < dirty.len() {
            let page = dirty[i] as usize / CHUNKS_PER_PAGE;
            let mut mask = 0u16;
            while i < dirty.len() && dirty[i] as usize / CHUNKS_PER_PAGE == page {
                let chunk = dirty[i] as usize;
                mask |= 1 << (chunk % CHUNKS_PER_PAGE);
                self.dirty_bits[chunk / 64] &= !(1 << (chunk % 64));
                i += 1;
            }
            let first = page * CHUNKS_PER_PAGE;
            let (address, source) = if first < self.low_chunks {
                let offset = first << CHUNK_BITS;
                (offset as u64, self.low.bytes(offset, PAGE_SIZE))
            } else {
                let offset = (first - self.low_chunks) << CHUNK_BITS;
                (
                    self.layout.high_base + offset as u64,
                    self.high.bytes(offset, PAGE_SIZE),
                )
            };
            if self.pages.refresh(address >> PAGE_BITS, source, mask) {
                self.flat_pages += 1;
            }
        }
        dirty.clear();
        self.dirty = dirty;
    }

    /// Marks the chunk holding the flat address `address` written.
    #[inline(always)]
    fn mark(&mut self, chunk: usize) {
        let (word, bit) = (chunk / 64, chunk % 64);
        if self.dirty_bits[word] >> bit & 1 == 0 {
            self.dirty_bits[word] |= 1 << bit;
            // In range: `new` checked every chunk index fits a u32.
            self.dirty.push(chunk as u32);
        }
    }

    /// The low region's offset of `N` bytes at `address` when all of them lie
    /// in it.
    #[inline(always)]
    fn in_low<const N: usize>(&self, address: u64) -> Option<usize> {
        // `low_end >= PAGE_SIZE >= N`: no underflow.
        (address <= self.layout.low_end - N as u64).then_some(address as usize)
    }

    /// The high region's offset of `N` bytes at `address` when all of them
    /// lie in it (and so do not wrap).
    #[inline(always)]
    fn in_high<const N: usize>(&self, address: u64) -> Option<usize> {
        (address >= self.layout.high_base && address <= u64::MAX - (N as u64 - 1))
            .then(|| (address - self.layout.high_base) as usize)
    }

    /// Marks every chunk of `len` bytes at the low (or high) region's
    /// `offset` written.
    fn mark_range(&mut self, high: bool, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        let base = if high { self.low_chunks } else { 0 };
        for chunk in (offset >> CHUNK_BITS)..=((offset + len - 1) >> CHUNK_BITS) {
            self.mark(base + chunk);
        }
    }

    #[cold]
    #[inline(never)]
    fn read_split<const N: usize>(&self, address: u64) -> [u8; N] {
        let mut out = [0u8; N];
        self.read_range(address, &mut out);
        out
    }

    #[cold]
    #[inline(never)]
    fn write_split<const N: usize>(&mut self, address: u64, data: [u8; N]) {
        self.write_range(address, &data);
    }
}

/// Where a piece of a range lies.
#[derive(Clone, Copy)]
enum Place {
    /// At this offset of the low region.
    Low(usize),
    /// At this offset of the high region.
    High(usize),
    /// At this address of the paged store.
    Pages(u64),
}

impl GuestMemory for FlatMemory {
    #[inline(always)]
    fn read_n<const N: usize>(&self, address: u64) -> [u8; N] {
        if let Some(offset) = self.in_low::<N>(address) {
            // SAFETY: `in_low` put `offset..offset + N` inside the region.
            unsafe { self.low.ptr.add(offset).cast::<[u8; N]>().read_unaligned() }
        } else if let Some(offset) = self.in_high::<N>(address) {
            // SAFETY: `in_high` put `offset..offset + N` inside the region.
            unsafe { self.high.ptr.add(offset).cast::<[u8; N]>().read_unaligned() }
        } else {
            self.read_split(address)
        }
    }

    #[inline(always)]
    fn write_n<const N: usize>(&mut self, address: u64, data: [u8; N]) {
        if let Some(offset) = self.in_low::<N>(address) {
            self.mark(offset >> CHUNK_BITS);
            self.mark((offset + N - 1) >> CHUNK_BITS);
            // SAFETY: `in_low` put `offset..offset + N` inside the region.
            unsafe {
                self.low
                    .ptr
                    .add(offset)
                    .cast::<[u8; N]>()
                    .write_unaligned(data)
            }
        } else if let Some(offset) = self.in_high::<N>(address) {
            self.mark(self.low_chunks + (offset >> CHUNK_BITS));
            self.mark(self.low_chunks + ((offset + N - 1) >> CHUNK_BITS));
            // SAFETY: `in_high` put `offset..offset + N` inside the region.
            unsafe {
                self.high
                    .ptr
                    .add(offset)
                    .cast::<[u8; N]>()
                    .write_unaligned(data)
            }
        } else {
            self.write_split(address, data)
        }
    }

    fn read_range(&self, address: u64, out: &mut [u8]) {
        self.layout.split(address, out.len(), |done, n, place| {
            let out = &mut out[done..done + n];
            match place {
                Place::Low(offset) => out.copy_from_slice(self.low.bytes(offset, n)),
                Place::High(offset) => out.copy_from_slice(self.high.bytes(offset, n)),
                Place::Pages(at) => self.pages.read(at, out),
            }
        });
    }

    fn write_range(&mut self, address: u64, data: &[u8]) {
        let layout = self.layout;
        layout.split(address, data.len(), |done, n, place| {
            let data = &data[done..done + n];
            match place {
                Place::Low(offset) => {
                    self.mark_range(false, offset, n);
                    self.low.bytes_mut(offset, n).copy_from_slice(data);
                }
                Place::High(offset) => {
                    self.mark_range(true, offset, n);
                    self.high.bytes_mut(offset, n).copy_from_slice(data);
                }
                Place::Pages(at) => self.pages.write(at, data),
            }
        });
    }

    fn public_output(&self) -> &[u8] {
        &self.public_output
    }

    fn extend_public_output(&mut self, bytes: &[u8]) {
        self.public_output.extend_from_slice(bytes);
    }

    fn dma_bulk(&self) -> bool {
        true
    }
}

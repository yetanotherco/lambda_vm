use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

/// Fast hasher for u64 keys - uses the key directly as the hash value.
/// This avoids the overhead of SipHash for integer keys.
#[derive(Default)]
pub struct U64Hasher(u64);

impl Hasher for U64Hasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.wrapping_shl(8).wrapping_add(b as u64);
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Default, Clone)]
pub struct U64BuildHasher;

impl BuildHasher for U64BuildHasher {
    type Hasher = U64Hasher;
    #[inline]
    fn build_hasher(&self) -> U64Hasher {
        U64Hasher(0)
    }
}

pub type U64HashMap<V> = HashMap<u64, V, U64BuildHasher>;

/// Total cap on public output bytes across all `commit_public_output` calls.
/// The COMMIT AIR concatenates calls via the running `x254` index, so this
/// is enforced as a running-total budget rather than a per-call limit.
pub const MAX_PUBLIC_OUTPUT_TOTAL_SIZE: u64 = 1024 * 1024;
/// Maximum size of the private input memory region (in bytes). 512 MiB so a
/// real proof (e.g. a continuation bundle) fits as private input.
pub const MAX_PRIVATE_INPUT_SIZE: u64 = 512 * 1024 * 1024;
/// Fixed high address where private input is mapped. Guest programs can read
/// directly from this address (ZisK-style memory-mapped input).
/// Layout: 4-byte LE length prefix at `PRIVATE_INPUT_START_INDEX`, then data at +4.
/// Must match `PRIVATE_INPUT_START` in `syscalls/src/syscalls.rs`.
pub const PRIVATE_INPUT_START_INDEX: u64 = 0xFF000000;
/// Size in bytes of the private input's wire-format length prefix (the `u32` LE
/// written at `PRIVATE_INPUT_START_INDEX` by [`Memory::store_private_inputs`]; the
/// data follows at `+ PRIVATE_INPUT_LENGTH_PREFIX_BYTES`). Single source of truth
/// for every page-span computation over the private-input region.
pub const PRIVATE_INPUT_LENGTH_PREFIX_BYTES: usize = size_of::<u32>();

/// Page size for the private-input region's paged backing store. The private
/// input region ([`PRIVATE_INPUT_START_INDEX`], `+ MAX_PRIVATE_INPUT_SIZE`) is
/// backed by a `HashMap<page index, Box<[u8; PRIVATE_INPUT_PAGE_SIZE]>>`
/// instead of one hashmap entry per 4-byte word, so growing/rehashing that
/// table only ever moves 8-byte box pointers, never the page bytes
/// themselves. Chosen to match the prover's `DEFAULT_PAGE_SIZE` concept
/// (`prover/src/tables/page.rs`); redeclared locally since the executor must
/// not depend on the prover crate.
const PRIVATE_INPUT_PAGE_SIZE: usize = 256 * 1024;

#[inline]
fn is_private_input_addr(address: u64) -> bool {
    address >= PRIVATE_INPUT_START_INDEX
        && address < PRIVATE_INPUT_START_INDEX + MAX_PRIVATE_INPUT_SIZE
}

#[inline]
fn private_page_index(address: u64) -> u64 {
    (address - PRIVATE_INPUT_START_INDEX) / PRIVATE_INPUT_PAGE_SIZE as u64
}

#[inline]
fn private_page_offset(address: u64) -> usize {
    ((address - PRIVATE_INPUT_START_INDEX) % PRIVATE_INPUT_PAGE_SIZE as u64) as usize
}

/// Allocates a zero-filled private-input page using fallible allocation, so a
/// guest requesting a page near `MAX_PRIVATE_INPUT_SIZE` fails cleanly with
/// [`MemoryError::AllocationFailed`] instead of aborting the host process.
fn try_allocate_private_page() -> Result<Box<[u8; PRIVATE_INPUT_PAGE_SIZE]>, MemoryError> {
    let mut buf: Vec<u8> = Vec::new();
    buf.try_reserve_exact(PRIVATE_INPUT_PAGE_SIZE)
        .map_err(|_| MemoryError::AllocationFailed)?;
    buf.resize(PRIVATE_INPUT_PAGE_SIZE, 0);
    buf.into_boxed_slice()
        .try_into()
        .map_err(|_| MemoryError::AllocationFailed)
}

#[derive(Default, Debug, Clone)]
pub struct Memory {
    cells: U64HashMap<[u8; 4]>,
    /// Private-input region backing store, paged at [`PRIVATE_INPUT_PAGE_SIZE`]
    /// granularity: key is the page index (`(addr - PRIVATE_INPUT_START_INDEX)
    /// / PRIVATE_INPUT_PAGE_SIZE`), value is a heap-boxed page. Only the boxed
    /// pointer lives in the hashmap's table slot, so table growth/rehashing
    /// moves 8-byte pointers, never the 256 KiB page contents. Pages are
    /// allocated lazily, zero-filled, on first write.
    private_input_pages: U64HashMap<Box<[u8; PRIVATE_INPUT_PAGE_SIZE]>>,
    /// Bytes committed to public output via `commit_public_output`. The
    /// COMMIT AIR doesn't write to a fixed memory region (it streams bytes
    /// onto the Commit bus by `index`), so this buffer is purely the
    /// executor's view used by `read_return_value` and CLI display.
    public_output: Vec<u8>,
}

impl Memory {
    /// Fetches the boxed page for `page_idx`, allocating and inserting a
    /// zero-filled page on first access (fallible allocation).
    fn get_or_insert_private_page(
        &mut self,
        page_idx: u64,
    ) -> Result<&mut [u8; PRIVATE_INPUT_PAGE_SIZE], MemoryError> {
        use std::collections::hash_map::Entry;
        let page = match self.private_input_pages.entry(page_idx) {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => {
                let page = try_allocate_private_page()?;
                v.insert(page)
            }
        };
        Ok(&mut **page)
    }

    pub fn load_byte(&self, address: u64) -> u8 {
        if is_private_input_addr(address) {
            let page_idx = private_page_index(address);
            let offset = private_page_offset(address);
            return self
                .private_input_pages
                .get(&page_idx)
                .map(|p| p[offset])
                .unwrap_or(0);
        }
        let aligned_address = address - address % 4;
        let value = self
            .cells
            .get(&aligned_address)
            .cloned()
            .unwrap_or_default();
        value[(address % 4) as usize]
    }

    pub fn store_byte(&mut self, address: u64, value: u8) {
        if is_private_input_addr(address) {
            let page_idx = private_page_index(address);
            let offset = private_page_offset(address);
            let page = self
                .private_input_pages
                .entry(page_idx)
                .or_insert_with(|| Box::new([0u8; PRIVATE_INPUT_PAGE_SIZE]));
            page[offset] = value;
            return;
        }
        let aligned_address = address - address % 4;
        let entry = self
            .cells
            .entry(aligned_address)
            .or_insert_with(|| [0, 0, 0, 0]);
        entry[(address % 4) as usize] = value;
    }

    /// Iterate over all stored bytes as `(address, value)` pairs. Ordinary
    /// cells are stored as 4-byte words; each word expands into its four byte
    /// addresses. Private-input bytes are stored as
    /// [`PRIVATE_INPUT_PAGE_SIZE`] pages; each allocated page expands into its
    /// byte addresses (unallocated pages contribute nothing, matching bytes
    /// never having been written). Used to snapshot memory at an epoch
    /// boundary.
    pub fn iter_bytes(&self) -> impl Iterator<Item = (u64, u8)> + '_ {
        let cells_iter = self.cells.iter().flat_map(|(&addr, bytes)| {
            bytes
                .iter()
                .enumerate()
                .map(move |(i, &b)| (addr + i as u64, b))
        });
        let private_iter = self.private_input_pages.iter().flat_map(|(&page_idx, page)| {
            let base = PRIVATE_INPUT_START_INDEX + page_idx * PRIVATE_INPUT_PAGE_SIZE as u64;
            page.iter()
                .enumerate()
                .map(move |(i, &b)| (base + i as u64, b))
        });
        cells_iter.chain(private_iter)
    }

    pub fn load_word(&self, address: u64) -> Result<u32, MemoryError> {
        if address.is_multiple_of(4) {
            // `PRIVATE_INPUT_START_INDEX` and `MAX_PRIVATE_INPUT_SIZE` are both
            // multiples of `PRIVATE_INPUT_PAGE_SIZE` (itself a multiple of 4),
            // so a 4-aligned word address is either fully inside the private
            // region and fully inside one page, or fully outside it.
            if is_private_input_addr(address) {
                let page_idx = private_page_index(address);
                let offset = private_page_offset(address);
                let bytes = self
                    .private_input_pages
                    .get(&page_idx)
                    .map(|p| {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(&p[offset..offset + 4]);
                        b
                    })
                    .unwrap_or([0u8; 4]);
                return Ok(u32::from_le_bytes(bytes));
            }
            let bytes = self.cells.get(&address).cloned().unwrap_or_default();
            Ok(u32::from_le_bytes(bytes))
        } else {
            address.checked_add(3).ok_or(MemoryError::AddressOverflow)?;
            Ok(u32::from_le_bytes([
                self.load_byte(address),
                self.load_byte(address + 1),
                self.load_byte(address + 2),
                self.load_byte(address + 3),
            ]))
        }
    }

    pub fn store_word(&mut self, address: u64, value: u32) -> Result<(), MemoryError> {
        let bytes = value.to_le_bytes();
        if address.is_multiple_of(4) {
            if is_private_input_addr(address) {
                let page_idx = private_page_index(address);
                let offset = private_page_offset(address);
                let page = self.get_or_insert_private_page(page_idx)?;
                page[offset..offset + 4].copy_from_slice(&bytes);
            } else {
                self.cells.insert(address, bytes);
            }
        } else {
            address.checked_add(3).ok_or(MemoryError::AddressOverflow)?;
            for (i, b) in bytes.iter().enumerate() {
                self.store_byte(address + i as u64, *b);
            }
        }
        Ok(())
    }

    /// Load a doubleword (64-bit) from memory - for LD instruction
    pub fn load_doubleword(&self, address: u64) -> Result<u64, MemoryError> {
        if address.is_multiple_of(8) {
            // 8-alignment bounds `address` to `u64::MAX - 7`, so `address + 4` can't overflow.
            // As with `load_word`, both region boundaries and the page size are
            // multiples of 8, so an 8-aligned doubleword never straddles the
            // private-input region boundary or a page boundary.
            if is_private_input_addr(address) {
                let page_idx = private_page_index(address);
                let offset = private_page_offset(address);
                let (low, high) = self
                    .private_input_pages
                    .get(&page_idx)
                    .map(|p| {
                        let mut low_bytes = [0u8; 4];
                        let mut high_bytes = [0u8; 4];
                        low_bytes.copy_from_slice(&p[offset..offset + 4]);
                        high_bytes.copy_from_slice(&p[offset + 4..offset + 8]);
                        (
                            u32::from_le_bytes(low_bytes) as u64,
                            u32::from_le_bytes(high_bytes) as u64,
                        )
                    })
                    .unwrap_or((0, 0));
                return Ok(low | (high << 32));
            }
            let low_bytes = self.cells.get(&address).cloned().unwrap_or_default();
            let high_bytes = self.cells.get(&(address + 4)).cloned().unwrap_or_default();
            let low = u32::from_le_bytes(low_bytes) as u64;
            let high = u32::from_le_bytes(high_bytes) as u64;
            Ok(low | (high << 32))
        } else {
            address.checked_add(7).ok_or(MemoryError::AddressOverflow)?;
            let mut bytes = [0u8; 8];
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = self.load_byte(address + i as u64);
            }
            Ok(u64::from_le_bytes(bytes))
        }
    }

    /// Store a doubleword (64-bit) to memory - for SD instruction
    pub fn store_doubleword(&mut self, address: u64, value: u64) -> Result<(), MemoryError> {
        if address.is_multiple_of(8) {
            let low = (value & 0xFFFFFFFF) as u32;
            let high = (value >> 32) as u32;
            // 8-alignment bounds `address` to `u64::MAX - 7`, so `address + 4` can't overflow.
            if is_private_input_addr(address) {
                let page_idx = private_page_index(address);
                let offset = private_page_offset(address);
                let page = self.get_or_insert_private_page(page_idx)?;
                page[offset..offset + 4].copy_from_slice(&low.to_le_bytes());
                page[offset + 4..offset + 8].copy_from_slice(&high.to_le_bytes());
            } else {
                self.cells.insert(address, low.to_le_bytes());
                self.cells.insert(address + 4, high.to_le_bytes());
            }
        } else {
            address.checked_add(7).ok_or(MemoryError::AddressOverflow)?;
            let bytes = value.to_le_bytes();
            for (i, b) in bytes.iter().enumerate() {
                self.store_byte(address + i as u64, *b);
            }
        }
        Ok(())
    }

    pub fn load_half(&self, address: u64) -> Result<u16, MemoryError> {
        if address.is_multiple_of(2) {
            let aligned_address = address - address % 4;
            let offset = (address % 4) as usize;
            // `aligned_address` is word-aligned, so (as with `load_word`) it's
            // either fully inside the private region and one page, or fully
            // outside.
            if is_private_input_addr(aligned_address) {
                let page_idx = private_page_index(aligned_address);
                let page_offset = private_page_offset(aligned_address) + offset;
                let bytes = self
                    .private_input_pages
                    .get(&page_idx)
                    .map(|p| [p[page_offset], p[page_offset + 1]])
                    .unwrap_or([0, 0]);
                return Ok(u16::from_le_bytes(bytes));
            }
            let bytes = self
                .cells
                .get(&aligned_address)
                .cloned()
                .unwrap_or_default();
            Ok(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
        } else {
            address.checked_add(1).ok_or(MemoryError::AddressOverflow)?;
            Ok(u16::from_le_bytes([
                self.load_byte(address),
                self.load_byte(address + 1),
            ]))
        }
    }

    pub fn store_half(&mut self, address: u64, value: u16) -> Result<(), MemoryError> {
        let bytes = value.to_le_bytes();
        if address.is_multiple_of(2) {
            let aligned_address = address - address % 4;
            let offset = (address % 4) as usize;
            if is_private_input_addr(aligned_address) {
                let page_idx = private_page_index(aligned_address);
                let page_offset = private_page_offset(aligned_address) + offset;
                let page = self.get_or_insert_private_page(page_idx)?;
                page[page_offset] = bytes[0];
                page[page_offset + 1] = bytes[1];
                return Ok(());
            }
            let entry = self
                .cells
                .entry(aligned_address)
                .or_insert_with(|| [0, 0, 0, 0]);
            entry[offset] = bytes[0];
            entry[offset + 1] = bytes[1];
        } else {
            address.checked_add(1).ok_or(MemoryError::AddressOverflow)?;
            self.store_byte(address, bytes[0]);
            self.store_byte(address + 1, bytes[1]);
        }
        Ok(())
    }

    /// Append `length` bytes from guest memory starting at `address` to the
    /// public output. The COMMIT AIR concatenates calls via the running
    /// `x254` index, and the trace builder accumulates `commit_ops` into
    /// `VmProof.public_output`; this method maintains the executor's view
    /// of the same byte stream so `read_return_value` matches.
    pub fn commit_public_output(&mut self, address: u64, length: u64) -> Result<(), MemoryError> {
        let new_total = (self.public_output.len() as u64)
            .checked_add(length)
            .ok_or(MemoryError::CommitSizeExceeded)?;
        if new_total > MAX_PUBLIC_OUTPUT_TOTAL_SIZE {
            return Err(MemoryError::CommitSizeExceeded);
        }
        let bytes = self.load_bytes(address, length)?;
        self.public_output.extend_from_slice(&bytes);
        Ok(())
    }

    pub fn read_return_value(&self) -> Result<Vec<u8>, MemoryError> {
        Ok(self.public_output.clone())
    }

    /// Pre-loads private input bytes at `PRIVATE_INPUT_START_INDEX` as a
    /// 4-byte LE length prefix followed by the raw data. The guest reads these
    /// bytes directly via normal RISC-V loads (ZisK-style memory-mapped input).
    pub fn store_private_inputs(&mut self, inputs: Vec<u8>) -> Result<(), MemoryError> {
        if inputs.is_empty() {
            return Ok(());
        }
        if inputs.len() as u64 > MAX_PRIVATE_INPUT_SIZE {
            return Err(MemoryError::PrivateInputSizeExceeded);
        }
        let len_u32 =
            u32::try_from(inputs.len()).map_err(|_| MemoryError::PrivateInputSizeExceeded)?;
        self.store_word(PRIVATE_INPUT_START_INDEX, len_u32)?;
        self.set_bytes_aligned(
            PRIVATE_INPUT_START_INDEX + PRIVATE_INPUT_LENGTH_PREFIX_BYTES as u64,
            &inputs,
        )?;
        Ok(())
    }

    pub fn load_bytes(&self, mut addr: u64, len: u64) -> Result<Vec<u8>, MemoryError> {
        let end = addr.checked_add(len).ok_or(MemoryError::AddressOverflow)?;
        let len_usize = usize::try_from(len).map_err(|_| MemoryError::AllocationFailed)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len_usize)
            .map_err(|_| MemoryError::AllocationFailed)?;
        while addr < end {
            let aligned = addr - (addr % 4);
            let offset = (addr % 4) as usize;
            let take = std::cmp::min(4 - offset, (end - addr) as usize);
            // `aligned` is word-aligned, so it's either fully inside the
            // private region and one page, or fully outside (see `load_word`).
            let bytes = if is_private_input_addr(aligned) {
                let page_idx = private_page_index(aligned);
                let page_offset = private_page_offset(aligned);
                self.private_input_pages
                    .get(&page_idx)
                    .map(|p| {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(&p[page_offset..page_offset + 4]);
                        b
                    })
                    .unwrap_or([0u8; 4])
            } else {
                self.cells.get(&aligned).cloned().unwrap_or_default()
            };
            result.extend_from_slice(&bytes[offset..offset + take]);
            addr += take as u64;
        }
        Ok(result)
    }

    /// Helper method to store a given input at an aligned address. It may also overwrite existing bytes with zero if inputs is not divisible by 4
    /// Should only be used to write to public output and private input where these limitations are not a problem
    pub(crate) fn set_bytes_aligned(
        &mut self,
        addr: u64,
        inputs: &[u8],
    ) -> Result<(), MemoryError> {
        if !addr.is_multiple_of(4) {
            return Err(MemoryError::UnalignedAccess);
        }
        if is_private_input_addr(addr) {
            return self.set_private_bytes_aligned(addr, inputs);
        }
        let mut addr = addr;
        for chunk in inputs.chunks(4) {
            let mut bytes = [0u8; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            self.cells.insert(addr, bytes);
            addr += 4;
        }
        Ok(())
    }

    /// Writes `inputs` contiguously into the paged private-input store
    /// starting at word-aligned `addr`, touching only the pages the data
    /// actually spans (allocating each lazily, at most once). Unlike
    /// `cells`, pages are raw zero-initialized byte arrays, so no zero
    /// padding is needed for a partial trailing word: the page's
    /// zero-initialized bytes already stand in for it.
    fn set_private_bytes_aligned(&mut self, addr: u64, inputs: &[u8]) -> Result<(), MemoryError> {
        let mut remaining = inputs;
        let mut cur_addr = addr;
        while !remaining.is_empty() {
            let page_idx = private_page_index(cur_addr);
            let page_offset = private_page_offset(cur_addr);
            let space_in_page = PRIVATE_INPUT_PAGE_SIZE - page_offset;
            let take = remaining.len().min(space_in_page);
            let page = self.get_or_insert_private_page(page_idx)?;
            page[page_offset..page_offset + take].copy_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
            cur_addr += take as u64;
        }
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum MemoryError {
    #[error("Unaligned memory access")]
    UnalignedAccess,
    #[error("Public output commit size exceeded")]
    CommitSizeExceeded,
    #[error("Private input size exceeded")]
    PrivateInputSizeExceeded,
    #[error("Address range exceeds u64::MAX")]
    AddressOverflow,
    #[error("Failed to allocate memory for load_bytes")]
    AllocationFailed,
}

#[cfg(test)]
mod tests {
    use super::*;

    // The wire-format writer and every private-input page-span computation assume the
    // length prefix is exactly a 4-byte LE `u32`; pin that so a change to the constant
    // is caught rather than silently drifting from the page math.
    #[test]
    fn private_input_length_prefix_is_a_le_u32() {
        assert_eq!(PRIVATE_INPUT_LENGTH_PREFIX_BYTES, 4);
        assert_eq!(PRIVATE_INPUT_LENGTH_PREFIX_BYTES, size_of::<u32>());
    }

    // `store_private_inputs` must write a LE length prefix at the region base and the data
    // immediately after it, at `+ PRIVATE_INPUT_LENGTH_PREFIX_BYTES`.
    #[test]
    fn store_private_inputs_writes_le_length_prefix_then_data() {
        let mut memory = Memory::default();
        let inputs = vec![0xAAu8, 0xBB, 0xCC];
        memory.store_private_inputs(inputs.clone()).unwrap();

        assert_eq!(
            memory.load_word(PRIVATE_INPUT_START_INDEX).unwrap(),
            inputs.len() as u32
        );
        let data = memory
            .load_bytes(
                PRIVATE_INPUT_START_INDEX + PRIVATE_INPUT_LENGTH_PREFIX_BYTES as u64,
                inputs.len() as u64,
            )
            .unwrap();
        assert_eq!(data, inputs);
    }
}

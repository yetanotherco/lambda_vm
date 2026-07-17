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
/// Maximum size of the private input payload (in bytes). 512 MiB so a real
/// proof (e.g. a continuation bundle) fits as private input.
pub const MAX_PRIVATE_INPUT_SIZE: u64 = 512 * 1024 * 1024;
/// Fixed high address where private input is mapped. Guest programs can read
/// directly from this address (ZisK-style memory-mapped input).
/// Layout: 4-byte LE length prefix at `PRIVATE_INPUT_START_INDEX`, then data at +4.
/// Must match `PRIVATE_INPUT_START` in `syscalls/src/syscalls.rs`.
pub const PRIVATE_INPUT_START_INDEX: u64 = 0xFF000000;
/// Size in bytes of the private input's wire-format length prefix (the `u32` LE
/// written at `PRIVATE_INPUT_START_INDEX` by [`Memory::store_private_inputs`]; the
/// data follows at `+ PRIVATE_INPUT_LENGTH_PREFIX_BYTES`).
pub const PRIVATE_INPUT_LENGTH_PREFIX_BYTES: usize = size_of::<u32>();

/// Page size for the whole address space's paged backing store. Memory is
/// backed by a `HashMap<page index, Box<[u8; MEMORY_PAGE_SIZE]>>` instead of
/// one hashmap entry per 4-byte word, so growing/rehashing that table only
/// ever moves 8-byte box pointers, never page bytes themselves. Chosen to
/// match the prover's `DEFAULT_PAGE_SIZE` concept (`prover/src/tables/page.rs`);
/// redeclared locally since the executor must not depend on the prover crate.
const MEMORY_PAGE_SIZE: usize = 256 * 1024;

#[inline]
fn page_index(address: u64) -> u64 {
    address / MEMORY_PAGE_SIZE as u64
}

#[inline]
fn page_offset(address: u64) -> usize {
    (address % MEMORY_PAGE_SIZE as u64) as usize
}

/// Allocates a zero-filled page using fallible allocation, so a guest driving
/// memory usage up fails cleanly with [`MemoryError::AllocationFailed`]
/// instead of aborting the host process.
fn try_allocate_page() -> Result<Box<[u8; MEMORY_PAGE_SIZE]>, MemoryError> {
    let mut buf: Vec<u8> = Vec::new();
    buf.try_reserve_exact(MEMORY_PAGE_SIZE)
        .map_err(|_| MemoryError::AllocationFailed)?;
    buf.resize(MEMORY_PAGE_SIZE, 0);
    Ok(buf
        .into_boxed_slice()
        .try_into()
        .expect("length pinned by resize"))
}

#[derive(Default, Debug, Clone)]
pub struct Memory {
    /// Whole-address-space backing store, paged at [`MEMORY_PAGE_SIZE`]
    /// granularity: key is the page index (`address / MEMORY_PAGE_SIZE`),
    /// value is a heap-boxed page. Only the boxed pointer lives in the
    /// hashmap's table slot, so table growth/rehashing moves 8-byte
    /// pointers, never page contents. Pages are allocated lazily,
    /// zero-filled, on first write. Private input is just memory written at
    /// a fixed high address (see [`PRIVATE_INPUT_START_INDEX`]) — it isn't
    /// special-cased at this layer.
    pages: U64HashMap<Box<[u8; MEMORY_PAGE_SIZE]>>,
    /// Bytes committed to public output via `commit_public_output`. The
    /// COMMIT AIR doesn't write to a fixed memory region (it streams bytes
    /// onto the Commit bus by `index`), so this buffer is purely the
    /// executor's view used by `read_return_value` and CLI display.
    public_output: Vec<u8>,
}

impl Memory {
    /// Fetches the boxed page for `page_idx`, allocating and inserting a
    /// zero-filled page on first access (fallible allocation).
    fn get_or_insert_page(
        &mut self,
        page_idx: u64,
    ) -> Result<&mut [u8; MEMORY_PAGE_SIZE], MemoryError> {
        use std::collections::hash_map::Entry;
        let page = match self.pages.entry(page_idx) {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => {
                let page = try_allocate_page()?;
                v.insert(page)
            }
        };
        Ok(&mut **page)
    }

    pub fn load_byte(&self, address: u64) -> u8 {
        self.pages
            .get(&page_index(address))
            .map(|p| p[page_offset(address)])
            .unwrap_or(0)
    }

    pub fn store_byte(&mut self, address: u64, value: u8) -> Result<(), MemoryError> {
        let idx = page_index(address);
        let off = page_offset(address);
        let page = self.get_or_insert_page(idx)?;
        page[off] = value;
        Ok(())
    }

    /// Iterate over all stored bytes as `(address, value)` pairs. Each
    /// allocated page expands into its byte addresses (unallocated pages
    /// contribute nothing, matching bytes never having been written). Used
    /// to snapshot memory at an epoch boundary.
    pub fn iter_bytes(&self) -> impl Iterator<Item = (u64, u8)> + '_ {
        self.pages.iter().flat_map(|(&idx, page)| {
            let base = idx * MEMORY_PAGE_SIZE as u64;
            page.iter().enumerate().map(move |(i, &b)| (base + i as u64, b))
        })
    }

    pub fn load_word(&self, address: u64) -> Result<u32, MemoryError> {
        if address.is_multiple_of(4) {
            // `MEMORY_PAGE_SIZE` is a multiple of 4, so a 4-aligned word
            // address never straddles a page boundary.
            let idx = page_index(address);
            let off = page_offset(address);
            let bytes = self
                .pages
                .get(&idx)
                .map(|p| {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&p[off..off + 4]);
                    b
                })
                .unwrap_or([0u8; 4]);
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
            let idx = page_index(address);
            let off = page_offset(address);
            let page = self.get_or_insert_page(idx)?;
            page[off..off + 4].copy_from_slice(&bytes);
        } else {
            address.checked_add(3).ok_or(MemoryError::AddressOverflow)?;
            for (i, b) in bytes.iter().enumerate() {
                self.store_byte(address + i as u64, *b)?;
            }
        }
        Ok(())
    }

    /// Load a doubleword (64-bit) from memory - for LD instruction
    pub fn load_doubleword(&self, address: u64) -> Result<u64, MemoryError> {
        if address.is_multiple_of(8) {
            // 8-alignment bounds `address` to `u64::MAX - 7`, so `address + 4` can't
            // overflow. `MEMORY_PAGE_SIZE` is a multiple of 8, so an 8-aligned
            // doubleword never straddles a page boundary.
            let idx = page_index(address);
            let off = page_offset(address);
            let (low, high) = self
                .pages
                .get(&idx)
                .map(|p| {
                    let mut low_bytes = [0u8; 4];
                    let mut high_bytes = [0u8; 4];
                    low_bytes.copy_from_slice(&p[off..off + 4]);
                    high_bytes.copy_from_slice(&p[off + 4..off + 8]);
                    (
                        u32::from_le_bytes(low_bytes) as u64,
                        u32::from_le_bytes(high_bytes) as u64,
                    )
                })
                .unwrap_or((0, 0));
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
            let idx = page_index(address);
            let off = page_offset(address);
            let page = self.get_or_insert_page(idx)?;
            page[off..off + 4].copy_from_slice(&low.to_le_bytes());
            page[off + 4..off + 8].copy_from_slice(&high.to_le_bytes());
        } else {
            address.checked_add(7).ok_or(MemoryError::AddressOverflow)?;
            let bytes = value.to_le_bytes();
            for (i, b) in bytes.iter().enumerate() {
                self.store_byte(address + i as u64, *b)?;
            }
        }
        Ok(())
    }

    pub fn load_half(&self, address: u64) -> Result<u16, MemoryError> {
        if address.is_multiple_of(2) {
            // `MEMORY_PAGE_SIZE` is a multiple of 2, so a 2-aligned half
            // address never straddles a page boundary.
            let idx = page_index(address);
            let off = page_offset(address);
            let bytes = self
                .pages
                .get(&idx)
                .map(|p| [p[off], p[off + 1]])
                .unwrap_or([0, 0]);
            Ok(u16::from_le_bytes(bytes))
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
            let idx = page_index(address);
            let off = page_offset(address);
            let page = self.get_or_insert_page(idx)?;
            page[off] = bytes[0];
            page[off + 1] = bytes[1];
        } else {
            address.checked_add(1).ok_or(MemoryError::AddressOverflow)?;
            self.store_byte(address, bytes[0])?;
            self.store_byte(address + 1, bytes[1])?;
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
    /// The size cap leaves room for the length prefix so the write can't spill
    /// past the `MAX_PRIVATE_INPUT_SIZE` budget the prover's page accounting
    /// (`page::private_input_page_count`, computed independently from the raw
    /// input length) assumes.
    pub fn store_private_inputs(&mut self, inputs: Vec<u8>) -> Result<(), MemoryError> {
        if inputs.is_empty() {
            return Ok(());
        }
        if inputs.len() as u64 > MAX_PRIVATE_INPUT_SIZE - PRIVATE_INPUT_LENGTH_PREFIX_BYTES as u64 {
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
            let idx = page_index(addr);
            let off = page_offset(addr);
            let take = std::cmp::min(MEMORY_PAGE_SIZE - off, (end - addr) as usize);
            match self.pages.get(&idx) {
                Some(page) => result.extend_from_slice(&page[off..off + take]),
                None => result.extend(std::iter::repeat_n(0u8, take)),
            }
            addr += take as u64;
        }
        Ok(result)
    }

    /// Helper method to store a given input at an aligned address, spanning
    /// as many pages as the data needs (allocating each lazily, at most
    /// once). It may also overwrite existing bytes with zero if inputs is
    /// not divisible by 4. Should only be used to write to public output and
    /// private input where these limitations are not a problem.
    pub(crate) fn set_bytes_aligned(
        &mut self,
        addr: u64,
        inputs: &[u8],
    ) -> Result<(), MemoryError> {
        if !addr.is_multiple_of(4) {
            return Err(MemoryError::UnalignedAccess);
        }
        let mut cur_addr = addr;
        let mut remaining = inputs;
        while !remaining.is_empty() {
            let idx = page_index(cur_addr);
            let off = page_offset(cur_addr);
            let space_in_page = MEMORY_PAGE_SIZE - off;
            let take = remaining.len().min(space_in_page);
            let page = self.get_or_insert_page(idx)?;
            page[off..off + take].copy_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
            cur_addr += take as u64;
        }
        let trailing_zeros = (4 - inputs.len() % 4) % 4;
        for i in 0..trailing_zeros as u64 {
            self.store_byte(cur_addr + i, 0)?;
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
    #[error("Failed to allocate memory")]
    AllocationFailed,
}

#[cfg(test)]
mod tests {
    use super::*;

    // The wire-format writer assumes the length prefix is exactly a 4-byte LE
    // `u32`; pin that so a change to the constant is caught rather than
    // silently drifting.
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

    // A maximal-size private input must be fully readable back, including
    // the last 4 bytes that would spill past `MAX_PRIVATE_INPUT_SIZE` if the
    // length prefix weren't accounted for in the accepted payload size.
    #[test]
    fn store_private_inputs_max_size_roundtrips() {
        let mut memory = Memory::default();
        let max_payload =
            (MAX_PRIVATE_INPUT_SIZE - PRIVATE_INPUT_LENGTH_PREFIX_BYTES as u64) as usize;
        let inputs = vec![0x42u8; max_payload];
        memory.store_private_inputs(inputs.clone()).unwrap();

        let data = memory
            .load_bytes(
                PRIVATE_INPUT_START_INDEX + PRIVATE_INPUT_LENGTH_PREFIX_BYTES as u64,
                inputs.len() as u64,
            )
            .unwrap();
        assert_eq!(data, inputs);
    }

    #[test]
    fn store_private_inputs_rejects_oversized_payload() {
        let mut memory = Memory::default();
        let oversized =
            (MAX_PRIVATE_INPUT_SIZE - PRIVATE_INPUT_LENGTH_PREFIX_BYTES as u64 + 1) as usize;
        let err = memory
            .store_private_inputs(vec![0u8; oversized])
            .expect_err("payload that would spill past the region must error");
        assert!(matches!(err, MemoryError::PrivateInputSizeExceeded));
    }

    #[test]
    fn set_bytes_aligned_zero_pads_trailing_partial_word() {
        let mut memory = Memory::default();
        // Pre-fill so a stale nonzero byte would surface if the trailing
        // partial word weren't zeroed.
        memory.store_word(0x1000, 0xFFFF_FFFF).unwrap();
        memory.set_bytes_aligned(0x1000, &[0xAB]).unwrap();
        assert_eq!(memory.load_word(0x1000).unwrap(), 0x0000_00AB);
    }
}

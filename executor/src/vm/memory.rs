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

/// Guest memory: every byte of the 64-bit address space, zero until written.
///
/// It keeps the bytes in 64 KiB pages ([`Pages`]): an access finds its page by
/// indexing a directory, with no hashing, so its cost does not grow with the
/// memory a run touches. `LAMBDA_VM_EXEC_MEMORY=words` keeps them in the map of
/// 4-byte words the executor used before ([`Store::Words`]), the control the
/// pages are measured and tested against. Both answer every access alike.
#[derive(Debug, Clone)]
pub struct Memory {
    store: Store,
    /// Bytes committed to public output via `commit_public_output`. The
    /// COMMIT AIR doesn't write to a fixed memory region (it streams bytes
    /// onto the Commit bus by `index`), so this buffer is purely the
    /// executor's view used by `read_return_value` and CLI display.
    public_output: Vec<u8>,
}

impl Default for Memory {
    fn default() -> Self {
        Self::with_store(StoreKind::from_env())
    }
}

/// Which store a [`Memory`] keeps its bytes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreKind {
    /// 64 KiB pages found through a directory ([`Pages`]).
    Pages,
    /// A map of 4-byte words keyed by their address.
    Words,
}

impl StoreKind {
    /// `LAMBDA_VM_EXEC_MEMORY=words` picks [`StoreKind::Words`]; unset or
    /// anything else, [`StoreKind::Pages`]. Read once.
    pub fn from_env() -> Self {
        static KIND: std::sync::OnceLock<StoreKind> = std::sync::OnceLock::new();
        *KIND.get_or_init(|| match std::env::var("LAMBDA_VM_EXEC_MEMORY").as_deref() {
            Ok("words") => StoreKind::Words,
            _ => StoreKind::Pages,
        })
    }
}

#[derive(Debug, Clone)]
enum Store {
    Pages(Pages),
    Words(U64HashMap<[u8; 4]>),
}

const PAGE_BITS: u32 = 16;
const PAGE_SIZE: usize = 1 << PAGE_BITS;
const PAGE_MASK: u64 = PAGE_SIZE as u64 - 1;
/// The highest page number.
const LAST_PAGE: u64 = u64::MAX >> PAGE_BITS;
/// Pages each directory indexes: the low 8 GiB (code, data, heap, private
/// input) and the top 8 GiB (the stack, which starts at `STACK_TOP`).
const DIRECTORY_PAGES: u64 = 1 << (33 - PAGE_BITS);

/// One page's bytes and, per 4-byte word, whether it was written.
#[derive(Clone)]
struct Page {
    bytes: Box<[u8]>,
    written: Box<[u64]>,
}

impl Page {
    fn new() -> Self {
        Self {
            bytes: vec![0u8; PAGE_SIZE].into_boxed_slice(),
            written: vec![0u64; PAGE_SIZE / 4 / 64].into_boxed_slice(),
        }
    }

    /// Writes `data` at `offset` (inside the page) and marks its words.
    #[inline]
    fn write(&mut self, offset: usize, data: &[u8]) {
        self.bytes[offset..offset + data.len()].copy_from_slice(data);
        for word in offset / 4..=(offset + data.len() - 1) / 4 {
            self.written[word / 64] |= 1 << (word % 64);
        }
    }
}

/// The pages a run touched. Page `n` sits in `low[n]` when `n` is below
/// [`DIRECTORY_PAGES`], in `high[LAST_PAGE - n]` when it is among the top
/// [`DIRECTORY_PAGES`], and in `other` otherwise; the directories grow to the
/// furthest page touched.
#[derive(Clone, Default)]
struct Pages {
    low: Vec<Option<Box<Page>>>,
    high: Vec<Option<Box<Page>>>,
    other: std::collections::BTreeMap<u64, Box<Page>>,
}

impl std::fmt::Debug for Pages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = |d: &[Option<Box<Page>>]| d.iter().filter(|p| p.is_some()).count();
        f.debug_struct("Pages")
            .field("low", &count(&self.low))
            .field("high", &count(&self.high))
            .field("other", &self.other.len())
            .finish()
    }
}

impl Pages {
    #[inline]
    fn page(&self, number: u64) -> Option<&Page> {
        if number < DIRECTORY_PAGES {
            self.low.get(number as usize)?.as_deref()
        } else if LAST_PAGE - number < DIRECTORY_PAGES {
            self.high.get((LAST_PAGE - number) as usize)?.as_deref()
        } else {
            self.other.get(&number).map(|page| &**page)
        }
    }

    #[inline]
    fn page_mut(&mut self, number: u64) -> &mut Page {
        fn slot(directory: &mut Vec<Option<Box<Page>>>, index: usize) -> &mut Page {
            if directory.len() <= index {
                directory.resize_with(index + 1, || None);
            }
            directory[index].get_or_insert_with(|| Box::new(Page::new()))
        }
        if number < DIRECTORY_PAGES {
            slot(&mut self.low, number as usize)
        } else if LAST_PAGE - number < DIRECTORY_PAGES {
            slot(&mut self.high, (LAST_PAGE - number) as usize)
        } else {
            self.other
                .entry(number)
                .or_insert_with(|| Box::new(Page::new()))
        }
    }

    /// The bytes at `address..` into `out`; the range must not wrap.
    #[inline]
    fn read(&self, address: u64, out: &mut [u8]) {
        let mut done = 0;
        while done < out.len() {
            let at = address + done as u64;
            let offset = (at & PAGE_MASK) as usize;
            let n = (PAGE_SIZE - offset).min(out.len() - done);
            match self.page(at >> PAGE_BITS) {
                Some(page) => out[done..done + n].copy_from_slice(&page.bytes[offset..offset + n]),
                None => out[done..done + n].fill(0),
            }
            done += n;
        }
    }

    /// `data` at `address..`; the range must not wrap.
    #[inline]
    fn write(&mut self, address: u64, data: &[u8]) {
        let mut done = 0;
        while done < data.len() {
            let at = address + done as u64;
            let offset = (at & PAGE_MASK) as usize;
            let n = (PAGE_SIZE - offset).min(data.len() - done);
            self.page_mut(at >> PAGE_BITS)
                .write(offset, &data[done..done + n]);
            done += n;
        }
    }

    /// Every byte of every written word, page by page.
    fn iter_bytes(&self) -> impl Iterator<Item = (u64, u8)> + '_ {
        let low = self
            .low
            .iter()
            .enumerate()
            .filter_map(|(n, page)| Some((n as u64, page.as_deref()?)));
        let other = self.other.iter().map(|(&n, page)| (n, &**page));
        let high = self
            .high
            .iter()
            .enumerate()
            .filter_map(|(i, page)| Some((LAST_PAGE - i as u64, page.as_deref()?)));
        low.chain(other).chain(high).flat_map(|(number, page)| {
            let base = number << PAGE_BITS;
            (0..PAGE_SIZE / 4)
                .filter(move |&word| page.written[word / 64] >> (word % 64) & 1 == 1)
                .flat_map(move |word| {
                    (0..4).map(move |i| (base + (4 * word + i) as u64, page.bytes[4 * word + i]))
                })
        })
    }
}

impl Store {
    /// The bytes at `address..` into `out`; the range must not wrap.
    #[inline]
    fn read(&self, address: u64, out: &mut [u8]) {
        match self {
            Store::Pages(pages) => pages.read(address, out),
            Store::Words(cells) => {
                let mut done = 0;
                while done < out.len() {
                    let at = address + done as u64;
                    let offset = (at % 4) as usize;
                    let n = (4 - offset).min(out.len() - done);
                    let word = cells
                        .get(&(at - offset as u64))
                        .copied()
                        .unwrap_or_default();
                    out[done..done + n].copy_from_slice(&word[offset..offset + n]);
                    done += n;
                }
            }
        }
    }

    /// `data` at `address..`; the range must not wrap.
    #[inline]
    fn write(&mut self, address: u64, data: &[u8]) {
        match self {
            Store::Pages(pages) => pages.write(address, data),
            Store::Words(cells) => {
                let mut done = 0;
                while done < data.len() {
                    let at = address + done as u64;
                    let offset = (at % 4) as usize;
                    let n = (4 - offset).min(data.len() - done);
                    let word = cells.entry(at - offset as u64).or_insert([0; 4]);
                    word[offset..offset + n].copy_from_slice(&data[done..done + n]);
                    done += n;
                }
            }
        }
    }
}

impl Memory {
    /// An empty memory over the given store.
    pub fn with_store(kind: StoreKind) -> Self {
        Self {
            store: match kind {
                StoreKind::Pages => Store::Pages(Pages::default()),
                StoreKind::Words => Store::Words(U64HashMap::default()),
            },
            public_output: Vec::new(),
        }
    }

    pub fn load_byte(&self, address: u64) -> u8 {
        let mut byte = [0u8; 1];
        self.store.read(address, &mut byte);
        byte[0]
    }

    pub fn store_byte(&mut self, address: u64, value: u8) {
        self.store.write(address, &[value]);
    }

    /// Iterate over all stored bytes as `(address, value)` pairs. Bytes are
    /// stored in 4-byte words; each word ever written yields its four byte
    /// addresses. Used to snapshot memory at an epoch boundary.
    pub fn iter_bytes(&self) -> Box<dyn Iterator<Item = (u64, u8)> + '_> {
        match &self.store {
            Store::Pages(pages) => Box::new(pages.iter_bytes()),
            Store::Words(cells) => Box::new(cells.iter().flat_map(|(&addr, bytes)| {
                bytes
                    .iter()
                    .enumerate()
                    .map(move |(i, &b)| (addr + i as u64, b))
            })),
        }
    }

    pub fn load_word(&self, address: u64) -> Result<u32, MemoryError> {
        // A 4-aligned word cannot wrap.
        if !address.is_multiple_of(4) {
            address.checked_add(3).ok_or(MemoryError::AddressOverflow)?;
        }
        let mut bytes = [0u8; 4];
        self.store.read(address, &mut bytes);
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn store_word(&mut self, address: u64, value: u32) -> Result<(), MemoryError> {
        if !address.is_multiple_of(4) {
            address.checked_add(3).ok_or(MemoryError::AddressOverflow)?;
        }
        self.store.write(address, &value.to_le_bytes());
        Ok(())
    }

    /// Load a doubleword (64-bit) from memory - for LD instruction
    pub fn load_doubleword(&self, address: u64) -> Result<u64, MemoryError> {
        // An 8-aligned doubleword cannot wrap.
        if !address.is_multiple_of(8) {
            address.checked_add(7).ok_or(MemoryError::AddressOverflow)?;
        }
        let mut bytes = [0u8; 8];
        self.store.read(address, &mut bytes);
        Ok(u64::from_le_bytes(bytes))
    }

    /// Store a doubleword (64-bit) to memory - for SD instruction
    pub fn store_doubleword(&mut self, address: u64, value: u64) -> Result<(), MemoryError> {
        if !address.is_multiple_of(8) {
            address.checked_add(7).ok_or(MemoryError::AddressOverflow)?;
        }
        self.store.write(address, &value.to_le_bytes());
        Ok(())
    }

    pub fn load_half(&self, address: u64) -> Result<u16, MemoryError> {
        if !address.is_multiple_of(2) {
            address.checked_add(1).ok_or(MemoryError::AddressOverflow)?;
        }
        // A 2-aligned half never leaves its word, so it cannot wrap.
        let mut bytes = [0u8; 2];
        self.store.read(address, &mut bytes);
        Ok(u16::from_le_bytes(bytes))
    }

    pub fn store_half(&mut self, address: u64, value: u16) -> Result<(), MemoryError> {
        if !address.is_multiple_of(2) {
            address.checked_add(1).ok_or(MemoryError::AddressOverflow)?;
        }
        self.store.write(address, &value.to_le_bytes());
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

    pub fn load_bytes(&self, addr: u64, len: u64) -> Result<Vec<u8>, MemoryError> {
        addr.checked_add(len).ok_or(MemoryError::AddressOverflow)?;
        let len_usize = usize::try_from(len).map_err(|_| MemoryError::AllocationFailed)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len_usize)
            .map_err(|_| MemoryError::AllocationFailed)?;
        result.resize(len_usize, 0);
        self.store.read(addr, &mut result);
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
        let whole = inputs.len() - inputs.len() % 4;
        self.store.write(addr, &inputs[..whole]);
        if whole < inputs.len() {
            let mut last = [0u8; 4];
            last[..inputs.len() - whole].copy_from_slice(&inputs[whole..]);
            self.store.write(addr + whole as u64, &last);
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

//! The walk under the word memory format
//! ([`MemoryFormat::Words`](super::super::fused_memory::MemoryFormat::Words)):
//! each CPU load or store is one row of the CPU's word tables
//! ([`super::super::word_cpu`]), by the words it touches.
//!
//! The walk keeps its per-byte state ([`MemoryState`]) and stamps the bytes
//! an access touches, as under the other formats. A word's value is its eight
//! bytes and its timestamp the largest of their stamps: every access stamps
//! each byte it touches, so the latest access touching any of a word's bytes
//! is the word's last access, which is the token the word tables consume.
//! PAGE_W's final words are read the same way from the walk's final state
//! ([`page_table_trace_from_state`]).
//!
//! The chips' calls the format proves (the COPY chip's: memcpy and memmove,
//! memset, the commit syscall) are COPY_W rows and MEMW_W rows
//! ([`copy_call`]); any other ECALL but the halt is refused
//! ([`refuse_syscall`]), as the verifier refuses every other chip's table.

use super::super::bitwise::{BitwiseHistogram, BitwiseOperation};
use super::super::copy::Functionality;
use super::super::copy_w::{self, CallMemory, CopyWordRow};
use super::super::gpack::TraceForm;
use super::super::mem_load2w::{self, Load2WRow};
use super::super::mem_load8::{self, Load8Row};
use super::super::mem_loadw::{self, LoadWRow};
use super::super::mem_store2w::{self, Store2WRow};
use super::super::mem_store8::{self, Store8Row};
use super::super::mem_storew::{self, StoreWRow};
use super::super::memw_w::{self, MemwWordRow};
use super::super::page::PageConfig;
use super::super::page_w;
use super::super::types::{GoldilocksExtension, GoldilocksField};
use super::super::word_cpu::{WordClass, word_class};
use super::super::word_memory::WORD_BYTES;
use super::{ABSENT, HistoryOverlay, MemoryState, cpu_op_to_bytes_and_signed, vec_heap_bytes};
use crate::Error;
use crate::tables::cpu::CpuOp;
use stark::trace::TraceTable;
use std::sync::atomic::Ordering::Relaxed;

/// The word tables' rows a walk makes, family by family, in the ops' order
/// (`block_whir::WORD_TABLES`' order: MEM_LOAD8, MEM_STORE8, MEM_LOADW,
/// MEM_STOREW, MEM_LOAD2W, MEM_STORE2W, MEMW_W), then COPY_W's (COPY's
/// family under the word format). Empty under any other memory format.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct WordRows {
    pub(crate) load8: Vec<Load8Row>,
    pub(crate) store8: Vec<Store8Row>,
    pub(crate) loadw: Vec<LoadWRow>,
    pub(crate) storew: Vec<StoreWRow>,
    pub(crate) load2w: Vec<Load2WRow>,
    pub(crate) store2w: Vec<Store2WRow>,
    /// The chips' word accesses.
    pub(crate) memw: Vec<MemwWordRow>,
    /// COPY_W's rows.
    pub(crate) copy: Vec<CopyWordRow>,
}

/// [`WordRows`]' lists: the seven word families, then COPY_W.
pub(crate) const WORD_ROW_LISTS: usize = 8;

impl WordRows {
    /// Removes family `k`'s first row (`StreamTable::WORDS`' order): the
    /// tests' slicer bug (`WindowedTraceBuilder::drop_one_op`). Whether a row
    /// was there.
    #[cfg(test)]
    pub(super) fn drop_first(&mut self, k: usize) -> bool {
        fn first<T>(list: &mut Vec<T>) -> bool {
            (!list.is_empty()).then(|| list.remove(0)).is_some()
        }
        match k {
            0 => first(&mut self.load8),
            1 => first(&mut self.store8),
            2 => first(&mut self.loadw),
            3 => first(&mut self.storew),
            4 => first(&mut self.load2w),
            5 => first(&mut self.store2w),
            _ => first(&mut self.memw),
        }
    }

    /// Each list's row count ([`WORD_ROW_LISTS`]).
    pub(crate) fn lens(&self) -> [usize; WORD_ROW_LISTS] {
        [
            self.load8.len(),
            self.store8.len(),
            self.loadw.len(),
            self.storew.len(),
            self.load2w.len(),
            self.store2w.len(),
            self.memw.len(),
            self.copy.len(),
        ]
    }

    /// The bytes its lists take on the heap (capacities).
    pub(crate) fn heap_bytes(&self) -> usize {
        vec_heap_bytes(&self.load8)
            + vec_heap_bytes(&self.store8)
            + vec_heap_bytes(&self.loadw)
            + vec_heap_bytes(&self.storew)
            + vec_heap_bytes(&self.load2w)
            + vec_heap_bytes(&self.store2w)
            + vec_heap_bytes(&self.memw)
            + vec_heap_bytes(&self.copy)
    }

    /// `other`'s rows after this one's, family by family.
    pub(crate) fn append(&mut self, mut other: WordRows) {
        self.load8.append(&mut other.load8);
        self.store8.append(&mut other.store8);
        self.loadw.append(&mut other.loadw);
        self.storew.append(&mut other.storew);
        self.load2w.append(&mut other.load2w);
        self.store2w.append(&mut other.store2w);
        self.memw.append(&mut other.memw);
        self.copy.append(&mut other.copy);
    }

    /// `parts`' rows one after another, family by family.
    pub(crate) fn concat(parts: Vec<WordRows>) -> WordRows {
        let mut lens = [0usize; WORD_ROW_LISTS];
        for part in &parts {
            for (n, len) in lens.iter_mut().zip(part.lens()) {
                *n += len;
            }
        }
        let mut out = WordRows {
            load8: Vec::with_capacity(lens[0]),
            store8: Vec::with_capacity(lens[1]),
            loadw: Vec::with_capacity(lens[2]),
            storew: Vec::with_capacity(lens[3]),
            load2w: Vec::with_capacity(lens[4]),
            store2w: Vec::with_capacity(lens[5]),
            memw: Vec::with_capacity(lens[6]),
            copy: Vec::with_capacity(lens[7]),
        };
        for part in parts {
            out.append(part);
        }
        out
    }

    /// The BITWISE lookups of every row past `from` (each family's count of
    /// rows already counted), counted into `histogram`: the rows' lookups are
    /// in-walk lookups, as LOAD's are.
    pub(crate) fn count_bitwise_into(
        &self,
        from: [usize; WORD_ROW_LISTS],
        histogram: &mut BitwiseHistogram,
    ) {
        for row in &self.load8[from[0]..] {
            row.count_bitwise_into(histogram);
        }
        for row in &self.store8[from[1]..] {
            row.count_bitwise_into(histogram);
        }
        for row in &self.loadw[from[2]..] {
            row.count_bitwise_into(histogram);
        }
        for row in &self.storew[from[3]..] {
            row.count_bitwise_into(histogram);
        }
        for row in &self.load2w[from[4]..] {
            row.count_bitwise_into(histogram);
        }
        for row in &self.store2w[from[5]..] {
            row.count_bitwise_into(histogram);
        }
        for row in &self.memw[from[6]..] {
            row.count_bitwise_into(histogram);
        }
        for row in &self.copy[from[7]..] {
            row.count_bitwise_into(histogram);
        }
    }
}

/// The word at `base` (a multiple of 8) in `memory_state`: its eight bytes,
/// little-endian, and the largest of their stamps ([`ABSENT`] when the state
/// lacks any of them, which marks it missing).
#[inline]
fn read_word(memory_state: &mut MemoryState, base: u64) -> (u64, u64) {
    let (values, stamps) = memory_state.read_bytes(base, WORD_BYTES as usize);
    let word = values.iter().enumerate().fold(0u64, |word, (i, &byte)| {
        word | (u64::from(byte) & 0xFF) << (8 * i)
    });
    (word, stamps.iter().copied().max().unwrap_or(0))
}

/// The words a load or store at `address` touches, each its value and its
/// old timestamp: the word holding its first byte, and the next when it
/// crosses (`(0, 0)` for none). Under the light record walk (`light`) from
/// the cycle's word access record ([`HistoryOverlay::word_access`]), else
/// from the walk's memory ([`read_word`]). A missing record, or one that is
/// not this access's, marks the state missing and reads [`ABSENT`].
fn touched_words(
    memory_state: &mut MemoryState,
    address: u64,
    crossing: bool,
    light: bool,
) -> [(u64, u64); 2] {
    let base = address - address % WORD_BYTES;
    if !light {
        let first = read_word(memory_state, base);
        let second = if crossing {
            read_word(memory_state, base + WORD_BYTES)
        } else {
            (0, 0)
        };
        return [first, second];
    }
    let record = memory_state
        .overlay
        .as_deref()
        .and_then(HistoryOverlay::word_access)
        .filter(|&(at, _, _)| at == address);
    match record {
        Some((_, first, None)) if !crossing => [first, (0, 0)],
        Some((_, first, Some(second))) if crossing => [first, second],
        _ => {
            memory_state.missing.store(true, Relaxed);
            [(0, ABSENT), (0, ABSENT)]
        }
    }
}

/// Refuses an ECALL under the word memory format that is neither the halt,
/// nor a COPY chip's call (memcpy and memmove, memset, the commit syscall),
/// nor keccak256:
/// the other chips' memory messages have no receiver under it yet, and the
/// verifier refuses their tables (`block_whir::chip_tables_under`).
pub(super) fn refuse_syscall(op: CpuOp<'_>) -> Result<(), Error> {
    let halt = executor::vm::instruction::execution::SyscallNumbers::Halt as u64;
    let copy = op.ecall_commit() || op.ecall_dma_memcpy() || op.ecall_dma_memset();
    if op.decode.fields.ecall
        && op.rv1 != halt
        && !copy
        && !op.ecall_keccak256()
        && !op.ecall_ecrec()
    {
        return Err(Error::Prover(format!(
            "the word memory format proves no syscall but the halt, the COPY chip's, \
             keccak256 and ECREC yet: syscall {:#x} at timestamp {}",
            op.rv1, op.timestamp
        )));
    }
    Ok(())
}

/// A chip's memory accesses made as byte MEMW operations (each an aligned
/// doubleword, a read or a write, its old bytes and their stamps the walk's)
/// as MEMW_W rows: the word's old value its old bytes, its old timestamp the
/// largest of their stamps. `false` when the state lacked a word (marked
/// missing; nothing made). An access that is not an aligned doubleword is
/// the walk's error: the word format takes a chip's accesses by word (the
/// executor refuses such a call).
pub(super) fn chip_words(
    table: &str,
    ops: &[super::super::memw::MemwOperation],
    rows: &mut WordRows,
    lookups: Option<&mut Vec<BitwiseOperation>>,
) -> Result<bool, Error> {
    use super::super::memw_w::MessageKind;
    let word = |bytes: &[u32; 8]| {
        bytes
            .iter()
            .enumerate()
            .fold(0u64, |w, (i, &b)| w | (u64::from(b) & 0xFF) << (8 * i))
    };
    let mut made = Vec::with_capacity(ops.len());
    for op in ops {
        if op.is_register || op.width != 8 || !op.base_address.is_multiple_of(WORD_BYTES) {
            return Err(Error::Prover(format!(
                "{table}: an access at {:#x} of {} bytes is no aligned word: the word memory \
                 format takes a chip's memory by aligned words",
                op.base_address, op.width
            )));
        }
        let old_ts = op.old_timestamp.iter().copied().max().unwrap_or(0);
        if old_ts == ABSENT {
            return Ok(false);
        }
        let (old, new) = (word(&op.old), word(&op.value));
        let kind = if op.is_read {
            MessageKind::Access
        } else {
            MessageKind::Write
        };
        made.push(
            MemwWordRow::new(kind, op.base_address, op.timestamp, old_ts, old, new)
                .map_err(|why| Error::Prover(format!("{table}: {why}")))?,
        );
    }
    if let Some(out) = lookups {
        out.extend(made.iter().flat_map(MemwWordRow::bitwise_ops));
    }
    rows.memw.extend(made);
    Ok(true)
}

/// The walk's per-byte state as a call's word memory: a word is its eight
/// bytes and the largest of their stamps; a call's access stamps the bytes
/// of its range it touches (a read keeps their values).
struct StateWords<'a>(&'a mut MemoryState);

impl CallMemory for StateWords<'_> {
    fn word(&mut self, key: u64) -> Option<(u64, u64)> {
        let (word, ts) = read_word(self.0, key);
        (ts != ABSENT).then_some((word, ts))
    }

    fn set(&mut self, key: u64, timestamp: u64, value: u64, (first, len): (u64, u8)) {
        if len == 0 {
            return;
        }
        let shift = 8 * (first - key);
        self.0
            .write_bytes(first, value >> shift, usize::from(len), timestamp);
    }
}

/// A COPY chip's call under the word memory format (`functionality` at
/// `timestamp`, the operands as the walk read them: a commit's `dst` is its
/// commit index): its COPY_W rows (with `rest`) and its MEMW_W rows, the
/// words read from `memory_state` and its bytes moved there, the rows'
/// lookups appended to `lookups`. A word the state lacks makes no row: the
/// state is marked missing and the window is refused. A call COPY_W cannot
/// hold is the walk's error.
#[allow(clippy::too_many_arguments)]
pub(super) fn copy_call(
    functionality: Functionality,
    timestamp: u64,
    src: u64,
    dst: u64,
    count: u64,
    memory_state: &mut MemoryState,
    rows: &mut WordRows,
    rest: bool,
    lookups: Option<&mut Vec<BitwiseOperation>>,
) -> Result<(), Error> {
    let refused = |why: String| Error::Prover(format!("COPY_W: {why}"));
    let mut call = copy_w::call_rows(functionality, timestamp, src, dst, count).map_err(refused)?;
    let Some(memw) = copy_w::run_call(&mut call, src, dst, count, &mut StateWords(memory_state))
        .map_err(refused)?
    else {
        return Ok(());
    };
    if let Some(out) = lookups {
        out.extend(memw.iter().flat_map(MemwWordRow::bitwise_ops));
        if rest {
            out.extend(call.iter().flat_map(CopyWordRow::bitwise_ops));
        }
    }
    rows.memw.extend(memw);
    if rest {
        rows.copy.extend(call);
    }
    Ok(())
}

/// A word row's refusal as the walk's error.
fn refused(table: &str, why: String) -> Error {
    Error::Prover(format!("{table}: {why}"))
}

/// A CPU load under the word memory format: its row in the table of its class,
/// the words it touches read from `memory_state` (their old timestamps the
/// largest of their bytes'), its lookups appended to `lookups`, then the bytes
/// it read stamped. A word the state lacks makes no row: the state is marked
/// missing and the window is refused. A row whose result is not the CPU's
/// loaded value is the walk's error (the state is not the run's).
pub(super) fn word_load(
    op: CpuOp<'_>,
    memory_state: &mut MemoryState,
    light: bool,
    rows: &mut WordRows,
    lookups: Option<&mut Vec<BitwiseOperation>>,
) -> Result<(), Error> {
    let address = op.res;
    let (count, signed) = cpu_op_to_bytes_and_signed(op);
    let width = count as u8;
    let class = word_class(address, width).map_err(|why| refused("a word load", why))?;
    let ts = op.timestamp;
    let [(first, old_first), (second, old_second)] =
        touched_words(memory_state, address, class == WordClass::Crossing, light);
    let made = match class {
        WordClass::Whole => {
            let (word, old) = (first, old_first);
            (old != ABSENT)
                .then(|| {
                    Load8Row::new(address, ts, old, word).map(|row| {
                        let result = word;
                        if let Some(out) = lookups {
                            out.extend(row.bitwise_ops());
                        }
                        rows.load8.push(row);
                        result
                    })
                })
                .transpose()
                .map_err(|why| refused("MEM_LOAD8", why))?
        }
        WordClass::Within => {
            let (word, old) = (first, old_first);
            (old != ABSENT)
                .then(|| {
                    LoadWRow::new(address, ts, old, width, signed, word).map(|row| {
                        let result = u64::from_le_bytes(row.result().0);
                        if let Some(out) = lookups {
                            out.extend(row.bitwise_ops());
                        }
                        rows.loadw.push(row);
                        result
                    })
                })
                .transpose()
                .map_err(|why| refused("MEM_LOADW", why))?
        }
        WordClass::Crossing => (old_first != ABSENT && old_second != ABSENT)
            .then(|| {
                Load2WRow::new(
                    address,
                    ts,
                    [old_first, old_second],
                    width,
                    signed,
                    [first, second],
                )
                .map(|row| {
                    let result = u64::from_le_bytes(row.result().0);
                    if let Some(out) = lookups {
                        out.extend(row.bitwise_ops());
                    }
                    rows.load2w.push(row);
                    result
                })
            })
            .transpose()
            .map_err(|why| refused("MEM_LOAD2W", why))?,
    };
    if let Some(result) = made
        && result != op.rvd
    {
        return Err(Error::Prover(format!(
            "a word load at {address:#x} (timestamp {ts}) reads {result:#x} from the walk's \
             memory, the run loaded {:#x}",
            op.rvd
        )));
    }
    // The light walk keeps no cells: the next cycle's record holds the write.
    if !light {
        memory_state.write_bytes(address, op.rvd, count, ts);
    }
    Ok(())
}

/// A CPU store under the word memory format ([`word_load`]'s rules): its row,
/// the words it touches read before it, then the bytes it wrote stamped.
pub(super) fn word_store(
    op: CpuOp<'_>,
    memory_state: &mut MemoryState,
    light: bool,
    rows: &mut WordRows,
    lookups: Option<&mut Vec<BitwiseOperation>>,
) -> Result<(), Error> {
    let address = op.res;
    let (count, _) = cpu_op_to_bytes_and_signed(op);
    let width = count as u8;
    let class = word_class(address, width).map_err(|why| refused("a word store", why))?;
    let (ts, value) = (op.timestamp, op.rv2);
    let [(first, old_first), (second, old_second)] =
        touched_words(memory_state, address, class == WordClass::Crossing, light);
    match class {
        WordClass::Whole => {
            let (old_word, old) = (first, old_first);
            if old != ABSENT {
                let row = Store8Row::new(address, ts, old, old_word, value)
                    .map_err(|why| refused("MEM_STORE8", why))?;
                if let Some(out) = lookups {
                    out.extend(row.bitwise_ops());
                }
                rows.store8.push(row);
            }
        }
        WordClass::Within => {
            let (old_word, old) = (first, old_first);
            if old != ABSENT {
                let row = StoreWRow::new(address, ts, old, width, value, old_word)
                    .map_err(|why| refused("MEM_STOREW", why))?;
                if let Some(out) = lookups {
                    out.extend(row.bitwise_ops());
                }
                rows.storew.push(row);
            }
        }
        WordClass::Crossing => {
            if old_first != ABSENT && old_second != ABSENT {
                let row = Store2WRow::new(
                    address,
                    ts,
                    [old_first, old_second],
                    width,
                    value,
                    [first, second],
                )
                .map_err(|why| refused("MEM_STORE2W", why))?;
                if let Some(out) = lookups {
                    out.extend(row.bitwise_ops());
                }
                rows.store2w.push(row);
            }
        }
    }
    // The light walk keeps no cells: the next cycle's record holds the write.
    if !light {
        memory_state.write_bytes(address, value, count, ts);
    }
    Ok(())
}

/// A keccak256 call's memory under the word memory format: its rows
/// (`ops`, [`super::super::keccak::sponge_ops`]'s) given their words
/// ([`super::super::keccak_w::SpongeWords`]), and its MEMW_W rows: each block's
/// covering words read at `T` (a word two blocks share once, by the earlier
/// block), then the digest's words at `T + 1` (the edge words read and
/// written, the middle ones written), the bytes stamped as the call touches
/// them. `false` when the state lacks a word (marked missing; nothing made).
pub(super) fn keccak256_words(
    ops: &mut [crate::tables::keccak::KeccakOperation],
    ptr: u64,
    len: u64,
    out: u64,
    memory_state: &mut MemoryState,
    rows: &mut WordRows,
    lookups: Option<&mut Vec<BitwiseOperation>>,
) -> Result<bool, Error> {
    use super::super::keccak_w::{
        BLOCK_WORDS, DIGEST, DigestWords, SpongeWords, digest_of, digest_words, reads,
    };
    use super::super::memw_w::MessageKind;
    let refused = |why: String| Error::Prover(format!("KECCAK_W: {why}"));
    let Some(t) = ops.first().map(|op| op.timestamp & 0xFFFF_FFFF) else {
        return Ok(true);
    };
    let offset = (ptr % WORD_BYTES) as u8;
    let src_key = ptr - u64::from(offset);
    let q = (out % WORD_BYTES) as u8;
    let out_key = out - u64::from(q);
    let reads_of: Vec<[bool; BLOCK_WORDS]> = ops
        .iter()
        .map(|op| {
            let sponge = op.sponge.as_ref().expect("a sponge row");
            reads(&sponge.mask, offset, sponge.first)
        })
        .collect();
    let block_key = |j: usize| src_key + 136 * j as u64;
    let digest_keys: Vec<u64> = (0..if q == 0 { 4 } else { 5 })
        .map(|v| out_key + 8 * v)
        .collect();
    let mut words = StateWords(memory_state);
    let held = reads_of
        .iter()
        .enumerate()
        .flat_map(|(j, r)| {
            (0..BLOCK_WORDS)
                .filter(|&w| r[w])
                .map(move |w| block_key(j) + 8 * w as u64)
        })
        .chain(digest_keys.iter().copied())
        .all(|key| words.word(key).is_some());
    if !held {
        return Ok(false);
    }
    let input = (ptr, len);
    let mut made = Vec::new();
    let mut blocks: Vec<[u64; BLOCK_WORDS]> = vec![[0; BLOCK_WORDS]; ops.len()];
    for (j, r) in reads_of.iter().enumerate() {
        for w in (0..BLOCK_WORDS).filter(|&w| r[w]) {
            let key = block_key(j) + 8 * w as u64;
            let (value, old_ts) = words.word(key).expect("held");
            made.push(
                MemwWordRow::new(MessageKind::Access, key, t, old_ts, value, value)
                    .map_err(refused)?,
            );
            words.set(key, t, value, inside(key, input));
            blocks[j][w] = value;
        }
    }
    // The shared word: carried forward when the block is not aligned, the
    // next block's first word otherwise (its own, read or not).
    for j in 1..blocks.len() {
        if offset != 0 {
            blocks[j][0] = blocks[j - 1][17];
        } else {
            blocks[j - 1][17] = blocks[j][0];
        }
    }
    let last = ops.last().expect("a row");
    let digest = digest_of(&last.output);
    let mut olds = [0u64; 5];
    for (v, &key) in digest_keys.iter().enumerate() {
        let (old, old_ts) = words.word(key).expect("held");
        olds[v] = old;
        let new = digest_words(&digest, q, olds[0], old)[v];
        let kind = if v == 0 || v == 4 {
            MessageKind::Access
        } else {
            MessageKind::Write
        };
        made.push(MemwWordRow::new(kind, key, t + 1, old_ts, old, new).map_err(refused)?);
        words.set(key, t + 1, new, inside(key, (out, DIGEST as u64)));
    }
    let n = ops.len();
    for (j, op) in ops.iter_mut().enumerate() {
        let sponge = op.sponge.as_mut().expect("a sponge row");
        sponge.words = Some(Box::new(SpongeWords {
            offset,
            words: blocks[j],
            digest: (j + 1 == n).then_some(DigestWords {
                q,
                old0: olds[0],
                old4: olds[4],
            }),
        }));
    }
    if let Some(out) = lookups {
        out.extend(made.iter().flat_map(MemwWordRow::bitwise_ops));
    }
    rows.memw.extend(made);
    Ok(true)
}

/// The bytes a range (`len` bytes from `start`) holds in the word at `key`:
/// `(first address, length)`.
fn inside(key: u64, (start, len): (u64, u64)) -> (u64, u8) {
    let lo = u128::from(key).max(u128::from(start));
    let hi = (u128::from(key) + u128::from(WORD_BYTES)).min(u128::from(start) + u128::from(len));
    (lo as u64, hi.saturating_sub(lo) as u8)
}

/// The word tables of `rows`, family by family (MEM_LOAD8 … MEM_STORE2W,
/// MEMW_W), each cut into tables of at most `max_rows[k]` rows (a family
/// without rows makes none), in `form`.
pub(super) fn word_tables(
    rows: &WordRows,
    max_rows: [usize; 7],
    skip: [usize; 7],
    tails: bool,
    generated: impl Fn(
        TraceTable<GoldilocksField, GoldilocksExtension>,
    ) -> TraceTable<GoldilocksField, GoldilocksExtension>
    + Send
    + Sync,
    form: TraceForm,
) -> Result<[Vec<TraceTable<GoldilocksField, GoldilocksExtension>>; 7], Error> {
    use super::chunk_and_generate_skipping as cut;
    let g = &generated;
    Ok([
        cut(&rows.load8, max_rows[0], skip[0], tails, true, |r| {
            g(mem_load8::generate_mem_load8_trace_segments_as(&[r], form))
        })?,
        cut(&rows.store8, max_rows[1], skip[1], tails, true, |r| {
            g(mem_store8::generate_mem_store8_trace_segments_as(
                &[r],
                form,
            ))
        })?,
        cut(&rows.loadw, max_rows[2], skip[2], tails, true, |r| {
            g(mem_loadw::generate_mem_loadw_trace_segments_as(&[r], form))
        })?,
        cut(&rows.storew, max_rows[3], skip[3], tails, true, |r| {
            g(mem_storew::generate_mem_storew_trace_segments_as(
                &[r],
                form,
            ))
        })?,
        cut(&rows.load2w, max_rows[4], skip[4], tails, true, |r| {
            g(mem_load2w::generate_mem_load2w_trace_segments_as(
                &[r],
                form,
            ))
        })?,
        cut(&rows.store2w, max_rows[5], skip[5], tails, true, |r| {
            g(mem_store2w::generate_mem_store2w_trace_segments_as(
                &[r],
                form,
            ))
        })?,
        cut(&rows.memw, max_rows[6], skip[6], tails, true, |r| {
            g(memw_w::generate_memw_w_trace_segments_as(&[r], form))
        })?,
    ])
}

/// How many tables [`word_tables`] makes of each family, the first `skip`
/// of each streamed ([`super::StreamSkip::words`]; with `tails`, `rows` holds
/// only the rows past them).
pub(super) fn word_table_counts(
    rows: &WordRows,
    max_rows: [usize; 7],
    skip: [usize; 7],
    tails: bool,
) -> Result<[usize; 7], Error> {
    let lens = rows.lens();
    let mut counts = [0usize; 7];
    for k in 0..7 {
        counts[k] = super::finish::chunked(lens[k], max_rows[k], skip[k], tails, true)?;
    }
    Ok(counts)
}

pub(super) fn word_max_rows(max_rows: &crate::tables::MaxRowsConfig) -> [usize; 7] {
    let (load, store) = (max_rows.mem_load, max_rows.mem_store);
    [load, store, load, store, load, store, store]
}

/// The table of `table`, one of the page tables cut from `configs`
/// (`page_w::page_tables`), from the walk's final state: a word's final value
/// its eight bytes, its timestamp the largest of their stamps (0: untouched,
/// its initial word); a page's PAGE_W table, or a block's, its pages' rows in
/// order. Refuses a table its pages cannot make.
pub(super) fn page_table_trace_from_state(
    table: &page_w::PageTable,
    configs: &[PageConfig],
    memory_state: &MemoryState,
) -> Result<TraceTable<GoldilocksField, GoldilocksExtension>, Error> {
    page_w::generate_page_table_trace_from_bytes(table, configs, |base| {
        memory_state.cells.page_data(base)
    })
    .map_err(Error::Prover)
}

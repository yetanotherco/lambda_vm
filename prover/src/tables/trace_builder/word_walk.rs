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
//! ([`page_w_trace_from_state`]).
//!
//! The format takes no syscall but the halt yet: an ECALL that is another is
//! refused ([`refuse_syscall`]), as the verifier refuses every chip's table.

use super::super::bitwise::{BitwiseHistogram, BitwiseOperation};
use super::super::gpack::TraceForm;
use super::super::mem_load2w::{self, Load2WRow};
use super::super::mem_load8::{self, Load8Row};
use super::super::mem_loadw::{self, LoadWRow};
use super::super::mem_store2w::{self, Store2WRow};
use super::super::mem_store8::{self, Store8Row};
use super::super::mem_storew::{self, StoreWRow};
use super::super::page::PageConfig;
use super::super::page_w;
use super::super::types::{GoldilocksExtension, GoldilocksField};
use super::super::word_cpu::{WordClass, word_class};
use super::super::word_memory::WORD_BYTES;
use super::{ABSENT, MemoryState, cpu_op_to_bytes_and_signed, vec_heap_bytes};
use crate::Error;
use crate::tables::cpu::CpuOp;
use stark::trace::TraceTable;

/// The CPU's word-table rows a walk makes, family by family, in the ops'
/// order (`block_whir::WORD_TABLES`' first six: MEM_LOAD8, MEM_STORE8,
/// MEM_LOADW, MEM_STOREW, MEM_LOAD2W, MEM_STORE2W). Empty under any other
/// memory format.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct WordRows {
    pub(crate) load8: Vec<Load8Row>,
    pub(crate) store8: Vec<Store8Row>,
    pub(crate) loadw: Vec<LoadWRow>,
    pub(crate) storew: Vec<StoreWRow>,
    pub(crate) load2w: Vec<Load2WRow>,
    pub(crate) store2w: Vec<Store2WRow>,
}

impl WordRows {
    /// Each family's row count.
    pub(crate) fn lens(&self) -> [usize; 6] {
        [
            self.load8.len(),
            self.store8.len(),
            self.loadw.len(),
            self.storew.len(),
            self.load2w.len(),
            self.store2w.len(),
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
    }

    /// `other`'s rows after this one's, family by family.
    pub(crate) fn append(&mut self, mut other: WordRows) {
        self.load8.append(&mut other.load8);
        self.store8.append(&mut other.store8);
        self.loadw.append(&mut other.loadw);
        self.storew.append(&mut other.storew);
        self.load2w.append(&mut other.load2w);
        self.store2w.append(&mut other.store2w);
    }

    /// `parts`' rows one after another, family by family.
    pub(crate) fn concat(parts: Vec<WordRows>) -> WordRows {
        let mut lens = [0usize; 6];
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
        };
        for part in parts {
            out.append(part);
        }
        out
    }

    /// The BITWISE lookups of every row past `from` (each family's count of
    /// rows already counted), counted into `histogram`: the rows' lookups are
    /// in-walk lookups, as LOAD's are.
    pub(crate) fn count_bitwise_into(&self, from: [usize; 6], histogram: &mut BitwiseHistogram) {
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

/// Refuses an ECALL under the word memory format that is not the halt: the
/// chips' memory messages have no receiver under it, and the verifier refuses
/// every chip's table (`block_whir::chip_tables_under`).
pub(super) fn refuse_syscall(op: CpuOp<'_>) -> Result<(), Error> {
    let halt = executor::vm::instruction::execution::SyscallNumbers::Halt as u64;
    if op.decode.fields.ecall && op.rv1 != halt {
        return Err(Error::Prover(format!(
            "the word memory format proves no syscall but the halt yet: syscall {:#x} at timestamp {}",
            op.rv1, op.timestamp
        )));
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
    rows: &mut WordRows,
    lookups: Option<&mut Vec<BitwiseOperation>>,
) -> Result<(), Error> {
    let address = op.res;
    let (count, signed) = cpu_op_to_bytes_and_signed(op);
    let width = count as u8;
    let class = word_class(address, width).map_err(|why| refused("a word load", why))?;
    let base = address - address % WORD_BYTES;
    let ts = op.timestamp;
    let made = match class {
        WordClass::Whole => {
            let (word, old) = read_word(memory_state, base);
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
            let (word, old) = read_word(memory_state, base);
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
        WordClass::Crossing => {
            let (first, old_first) = read_word(memory_state, base);
            let (second, old_second) = read_word(memory_state, base + WORD_BYTES);
            (old_first != ABSENT && old_second != ABSENT)
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
                .map_err(|why| refused("MEM_LOAD2W", why))?
        }
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
    memory_state.write_bytes(address, op.rvd, count, ts);
    Ok(())
}

/// A CPU store under the word memory format ([`word_load`]'s rules): its row,
/// the words it touches read before it, then the bytes it wrote stamped.
pub(super) fn word_store(
    op: CpuOp<'_>,
    memory_state: &mut MemoryState,
    rows: &mut WordRows,
    lookups: Option<&mut Vec<BitwiseOperation>>,
) -> Result<(), Error> {
    let address = op.res;
    let (count, _) = cpu_op_to_bytes_and_signed(op);
    let width = count as u8;
    let class = word_class(address, width).map_err(|why| refused("a word store", why))?;
    let base = address - address % WORD_BYTES;
    let (ts, value) = (op.timestamp, op.rv2);
    match class {
        WordClass::Whole => {
            let (old_word, old) = read_word(memory_state, base);
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
            let (old_word, old) = read_word(memory_state, base);
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
            let (first, old_first) = read_word(memory_state, base);
            let (second, old_second) = read_word(memory_state, base + WORD_BYTES);
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
    memory_state.write_bytes(address, value, count, ts);
    Ok(())
}

/// The word tables of `rows`, family by family (MEM_LOAD8 … MEM_STORE2W),
/// each cut into tables of at most `max_rows[k]` rows (a family without rows
/// makes none), in `form`.
pub(super) fn word_tables(
    rows: &WordRows,
    max_rows: [usize; 6],
    form: TraceForm,
) -> [Vec<TraceTable<GoldilocksField, GoldilocksExtension>>; 6] {
    fn cut<T>(
        list: &[T],
        max: usize,
        make: impl Fn(&[T]) -> TraceTable<GoldilocksField, GoldilocksExtension>,
    ) -> Vec<TraceTable<GoldilocksField, GoldilocksExtension>> {
        list.chunks(max.max(1)).map(make).collect()
    }
    [
        cut(&rows.load8, max_rows[0], |r| {
            mem_load8::generate_mem_load8_trace_segments_as(&[r], form)
        }),
        cut(&rows.store8, max_rows[1], |r| {
            mem_store8::generate_mem_store8_trace_segments_as(&[r], form)
        }),
        cut(&rows.loadw, max_rows[2], |r| {
            mem_loadw::generate_mem_loadw_trace_segments_as(&[r], form)
        }),
        cut(&rows.storew, max_rows[3], |r| {
            mem_storew::generate_mem_storew_trace_segments_as(&[r], form)
        }),
        cut(&rows.load2w, max_rows[4], |r| {
            mem_load2w::generate_mem_load2w_trace_segments_as(&[r], form)
        }),
        cut(&rows.store2w, max_rows[5], |r| {
            mem_store2w::generate_mem_store2w_trace_segments_as(&[r], form)
        }),
    ]
}

/// How many tables [`word_tables`] makes of each family.
pub(super) fn word_table_counts(rows: &WordRows, max_rows: [usize; 6]) -> [usize; 7] {
    let lens = rows.lens();
    let mut counts = [0usize; 7];
    for k in 0..6 {
        counts[k] = lens[k].div_ceil(max_rows[k].max(1));
    }
    counts
}

/// The word tables' row caps from the block's: the fused tables' (MEM_LOAD's
/// for the loads, MEM_STORE's for the stores), the chunk a block takes of
/// each.
pub(super) fn word_max_rows(max_rows: &crate::tables::MaxRowsConfig) -> [usize; 6] {
    let (load, store) = (max_rows.mem_load, max_rows.mem_store);
    [load, store, load, store, load, store]
}

/// The PAGE_W table of `config`'s page from the walk's final state: a word's
/// final value its eight bytes, its timestamp the largest of their stamps (0:
/// untouched, its initial word). Refuses a page PAGE_W cannot hold.
pub(super) fn page_w_trace_from_state(
    config: &PageConfig,
    memory_state: &MemoryState,
) -> Result<TraceTable<GoldilocksField, GoldilocksExtension>, Error> {
    page_w::generate_page_w_trace_from_bytes(config, memory_state.cells.page_data(config.page_base))
        .map_err(Error::Prover)
}

//! Op lists the builder keeps until their tables are built, as a stream of
//! varints instead of fixed-size ops (S3c): a [`Codec`] writes each op from
//! the ops before it (deltas, zigzag LEB128) and reads it back exactly.
//!
//! Every [`DeltaStream::every`] ops a sparse index records where the stream
//! stands (the block, the byte, and the codec's state), so any range expands
//! on its own. The stream is stored in blocks of at most [`BLOCK_BYTES`] that
//! are never reallocated once full, and an op never straddles two blocks. The
//! ops expand exactly as pushed, in order, so the tables are the same.

use super::{Expand, Part, Segmented};

/// The most bytes one block of a stream holds: at or above the allocator's
/// 8 MiB class, whose freed extents other threads reuse.
const BLOCK_BYTES: usize = 64 << 20;

/// How a family's ops are written: each from the ops before it, whose part
/// the codec keeps as its state (a `Copy` value the index records).
pub(super) trait Codec: Copy + Default + Send + Sync {
    type Op: Clone + Send + Sync;
    /// The most bytes one op takes, whatever its values.
    const MAX_BYTES: usize;
    /// Write `op` to `out`, after the ops this state has seen.
    fn encode(&mut self, op: &Self::Op, out: &mut Vec<u8>);
    /// Read the next op from `bytes` at `at`, after the ops this state has seen.
    fn decode(&mut self, bytes: &[u8], at: &mut usize) -> Self::Op;
}

/// Where the stream stands at op `k * every`: its block, the byte in it, and
/// the codec's state there.
#[derive(Clone, Copy)]
struct Mark<C> {
    block: u32,
    at: u32,
    state: C,
}

pub(super) struct DeltaStream<C: Codec> {
    bytes: Vec<Vec<u8>>,
    index: Vec<Mark<C>>,
    len: usize,
    state: C,
    /// Bytes a block of the stream holds; 0 is [`BLOCK_BYTES`] (tests
    /// set a small one, to cross blocks).
    block_bytes: usize,
    /// Ops between two index marks.
    every: usize,
}

/// The index's spacing a new stream takes: 4096 ops, or a test's
/// ([`with_index_every`]).
fn default_every() -> usize {
    #[cfg(test)]
    {
        let every = EVERY.with(std::cell::Cell::get);
        if every > 0 {
            return every;
        }
    }
    4096
}

#[cfg(test)]
thread_local! {
    static EVERY: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// `f` with every stream it makes on this thread marked each `every` ops (a
/// test's spacing, so chunks and slices start past index marks).
#[cfg(test)]
pub(super) fn with_index_every<T>(every: usize, f: impl FnOnce() -> T) -> T {
    let before = EVERY.with(|cell| cell.replace(every.max(1)));
    let out = f();
    EVERY.with(|cell| cell.set(before));
    out
}

impl<C: Codec> Default for DeltaStream<C> {
    fn default() -> Self {
        Self {
            bytes: Vec::new(),
            index: Vec::new(),
            len: 0,
            state: C::default(),
            block_bytes: 0,
            every: default_every(),
        }
    }
}

pub(super) fn zigzag(d: u64) -> u64 {
    (d << 1) ^ ((d as i64 >> 63) as u64)
}

pub(super) fn unzigzag(z: u64) -> u64 {
    (z >> 1) ^ (z & 1).wrapping_neg()
}

/// The bytes LEB128 takes for `v`.
pub(super) fn varint_len(v: u64) -> usize {
    (64 - (v | 1).leading_zeros() as usize).div_ceil(7)
}

pub(super) fn put(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

pub(super) fn get(bytes: &[u8], at: &mut usize) -> u64 {
    let (mut v, mut shift) = (0u64, 0u32);
    loop {
        let b = bytes[*at];
        *at += 1;
        v |= u64::from(b & 0x7f) << shift;
        if b < 0x80 {
            return v;
        }
        shift += 7;
    }
}

/// The cheapest of `codings` (mode, zigzag value): the mode and the value.
pub(super) fn cheapest<const N: usize>(codings: [(u8, u64); N]) -> (u8, u64) {
    let mut best = codings[0];
    for coding in codings {
        if varint_len(coding.1) < varint_len(best.1) {
            best = coding;
        }
    }
    best
}

impl<C: Codec> DeltaStream<C> {
    /// A list whose stream is cut into blocks of `bytes` (at least an op's).
    #[cfg(test)]
    pub(super) fn with_block_bytes(bytes: usize) -> Self {
        Self {
            block_bytes: bytes.max(C::MAX_BYTES),
            ..Self::default()
        }
    }

    /// `ops`, in order.
    pub(super) fn from_ops(ops: &[C::Op]) -> Self {
        let mut list = Self::default();
        list.extend(ops);
        list
    }

    fn block_bytes(&self) -> usize {
        if self.block_bytes == 0 {
            BLOCK_BYTES
        } else {
            self.block_bytes
        }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Ops between two index marks.
    #[cfg(test)]
    pub(super) fn every(&self) -> usize {
        self.every
    }

    /// The stream's bytes (not its capacity).
    #[cfg(test)]
    pub(super) fn stream_bytes(&self) -> usize {
        self.bytes.iter().map(Vec::len).sum()
    }

    /// Append `op`.
    pub(super) fn push(&mut self, op: &C::Op) {
        self.extend(std::slice::from_ref(op));
    }

    /// Append `ops`, in order.
    pub(super) fn extend(&mut self, ops: &[C::Op]) {
        let block_bytes = self.block_bytes();
        for op in ops {
            // Room for the op in the last block: grown as a `Vec` grows up to
            // a block's bytes, else a new block, allocated whole.
            let room = self
                .bytes
                .last()
                .is_some_and(|last| last.len() + C::MAX_BYTES <= block_bytes);
            if !room {
                let capacity = if self.bytes.is_empty() {
                    4096.min(block_bytes)
                } else {
                    block_bytes
                };
                self.bytes.push(Vec::with_capacity(capacity));
            }
            let block = self.bytes.len() - 1;
            let Some(last) = self.bytes.last_mut() else {
                continue;
            };
            if last.capacity() - last.len() < C::MAX_BYTES {
                let want = (last.capacity() * 2)
                    .min(block_bytes)
                    .max(last.len() + C::MAX_BYTES);
                last.reserve_exact(want - last.len());
            }
            if self.len.is_multiple_of(self.every) {
                self.index.push(Mark {
                    block: block as u32,
                    at: last.len() as u32,
                    state: self.state,
                });
            }
            self.state.encode(op, last);
            self.len += 1;
        }
    }

    /// The bytes it takes on the heap (capacities).
    pub(super) fn heap_bytes(&self) -> usize {
        self.bytes.iter().map(Vec::capacity).sum::<usize>()
            + self.bytes.capacity() * std::mem::size_of::<Vec<u8>>()
            + self.index.capacity() * std::mem::size_of::<Mark<C>>()
    }

    /// The list as one compact segment.
    pub(super) fn segments(&self) -> Segmented<'_, C::Op> {
        Segmented {
            parts: vec![Part::Compact(self)],
        }
    }

    /// Ops `start..end`, expanded.
    pub(super) fn range(&self, start: usize, end: usize) -> Vec<C::Op> {
        let mut out = Vec::new();
        self.expand_into(start, end, &mut out);
        out
    }
}

/// The ops, as a list of them prints.
impl<C: Codec> std::fmt::Debug for DeltaStream<C>
where
    C::Op: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.range(0, self.len)).finish()
    }
}

impl<C: Codec> Expand<C::Op> for DeltaStream<C> {
    fn len(&self) -> usize {
        self.len
    }

    fn expand_into(&self, start: usize, end: usize, out: &mut Vec<C::Op>) {
        let end = end.min(self.len);
        if start >= end {
            return;
        }
        let first = start / self.every;
        let mark = self.index[first];
        let (mut block, mut at, mut state) = (mark.block as usize, mark.at as usize, mark.state);
        out.reserve(end - start);
        for k in first * self.every..end {
            // An op lies in one block: the next op starts the next block once
            // this one ends.
            if at >= self.bytes[block].len() {
                block += 1;
                at = 0;
            }
            let op = state.decode(&self.bytes[block], &mut at);
            if k >= start {
                out.push(op);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use executor::vm::logs::Log;

    use super::with_index_every;
    use crate::tables::MaxRowsConfig;
    use crate::tables::trace_builder::{Traces, WindowedTraceBuilder};
    use crate::test_utils::asm_elf_bytes;
    use crate::tests::windowed_builder_tests::same_traces;

    fn run(name: &str) -> (Elf, Vec<Log>) {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        let logs = Executor::new(&program, Vec::new())
            .expect("the executor starts")
            .run()
            .expect("the program runs")
            .logs;
        (program, logs)
    }

    /// The whole-run build, and the windowed build over windows of `window`
    /// cycles (dropping each streamed chunk's ops as it leaves, or not) with its
    /// streamed chunks put back.
    fn builds(
        program: &Elf,
        logs: &[Log],
        max_rows: &MaxRowsConfig,
        window: usize,
        drop: bool,
    ) -> (Traces, Traces) {
        let whole = Traces::from_elf_and_logs(
            program,
            logs,
            max_rows,
            &[],
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        )
        .expect("the whole-run build");
        let mut builder = WindowedTraceBuilder::new(program, &[], max_rows).expect("the builder");
        if drop {
            builder = builder.drop_streamed_ops().expect("before any window");
        }
        let body = logs.len() - 1;
        let cut = body - body % window;
        let mut chunks = Vec::new();
        for w in logs[..cut].chunks(window) {
            chunks.extend(builder.push(w).expect("a window"));
        }
        let mut windowed = builder.finish(&logs[cut..]).expect("the last window");
        windowed
            .insert_streamed(chunks)
            .expect("every chunk has a placeholder");
        (whole, windowed)
    }

    /// The delta-coded lists marked every three ops instead of every 4096, so
    /// nearly every chunk and every p4 slice starts past an index mark: the
    /// whole-run and the windowed builds make the same tables, row for row.
    #[test]
    fn the_tables_are_the_same_with_an_index_mark_every_three_ops() {
        for max_rows in [MaxRowsConfig::small(), MaxRowsConfig::uniform(4)] {
            for name in [
                "all_instructions_64",
                "misalign_sd",
                "test_keccak_multi",
                "test_ecsm_multi",
            ] {
                let (program, logs) = run(name);
                for window in [7, 1000] {
                    for drop in [false, true] {
                        let build = || builds(&program, &logs, &max_rows, window, drop);
                        let (whole, windowed) = build();
                        let (dense_whole, dense_windowed) = with_index_every(3, build);
                        same_traces(&whole, &dense_whole);
                        same_traces(&windowed, &dense_windowed);
                        same_traces(&whole, &dense_windowed);
                    }
                }
            }
        }
    }
}

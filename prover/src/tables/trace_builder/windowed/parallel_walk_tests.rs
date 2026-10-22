//! Can a run's windows be walked in parallel? A measurement: the laptop checks
//! below, and one box-only test.
//!
//! From one window to the next the walk carries each memory byte's value and
//! last-access timestamp and the 35 register cells (the 32 registers, the
//! commit index and the pc); every timestamp is a function of the window's
//! first cycle. So a window walked from its ENTRY STATE (the bytes it touches,
//! as they stood before it, and the registers) is the window the sequential
//! walk walks. A STATE PASS (here the walk's state updates with no op built)
//! records each window's entry state at each byte's first access in it.
//!
//! The box test, [`the_walk_in_parallel_from_window_states`] (`--ignored`),
//! runs a real block and prints `PARALLEL WALK` lines: the builder's setup as
//! built before and as built now; the executor's, the sequential walk's and the
//! state pass's seconds a cycle; the entry states' sizes; the pass checked
//! against the walk; and a sample of windows walked again on P threads, from a
//! copy of the walk's state at their start and from their entry state alone,
//! alone and beside threads that generate chunks, each checked against the
//! sequential walk.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::instruction::execution::{
    BLAKE3_ABSORB_CV_OUT_DWORD, BLAKE3_BLOCK_BYTES, blake3_compress_6round, compute_hint,
    keccak_f1600,
};
use executor::vm::logs::Log;

use super::super::{
    MemoryState, RegisterState, WalkSizes, build_initial_image, build_initial_image_paged,
    collect_cpu_ops_into, cpu_op_to_bytes_and_signed,
};
use super::{ChunkJob, WalkedWindow, Walker, WindowedTraceBuilder};
use crate::paged_mem::PagedMem;
use crate::tables::MaxRowsConfig;
use crate::tables::blake3::{Blake3AbsorbOperation, expand_absorb};
use crate::tables::cpu::CpuOperation;
use crate::tables::decode::{self, DecodeTable};
use crate::tables::gpack::TraceForm;
use crate::tables::page::DEFAULT_PAGE_SIZE;
use crate::test_utils::asm_elf_bytes;

// =============================================================================
// The state pass
// =============================================================================

/// A byte of the state pass's memory: the walk's value and timestamp, and the
/// window that last touched it (16 bytes, as the walk's `(u8, u64)` cell).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PassCell {
    value: u8,
    window: u32,
    ts: u64,
}

/// No window has touched the byte yet.
const UNTOUCHED: u32 = u32::MAX;
const PASS_FILL: PassCell = PassCell {
    value: 0,
    window: UNTOUCHED,
    ts: 0,
};

/// A byte as it stood before a window: address, value, last-access timestamp.
type Entry = (u64, u8, u64);

/// The walk's state updates with no op built. At each byte's first access in
/// the window, read or write, it records the byte's value and timestamp from
/// before the window: the window's entry state.
struct StatePass {
    cells: PagedMem<PassCell>,
    /// `(page base, page index)` of the page found last, as the lean walk's:
    /// every page allocation goes through [`Self::page`], which leaves the
    /// page it allocated here.
    hot: (u64, usize),
    window: u32,
    entries: Vec<Entry>,
    registers: RegisterState,
}

impl StatePass {
    /// The pass from the state the walk starts from.
    fn new(memory: &MemoryState, registers: &RegisterState) -> Self {
        Self {
            cells: memory.cells.map(PASS_FILL, |(value, ts)| PassCell {
                value,
                window: UNTOUCHED,
                ts,
            }),
            hot: (1, 0),
            window: UNTOUCHED,
            entries: Vec::new(),
            registers: registers.clone(),
        }
    }

    /// Window `window` starts: its entries are recorded afresh.
    fn begin(&mut self, window: u32) {
        assert_ne!(window, UNTOUCHED, "too many windows for the pass's marks");
        self.window = window;
        self.entries.clear();
    }

    #[inline]
    fn page(&mut self, base: u64) -> usize {
        if self.hot.0 == base {
            return self.hot.1;
        }
        let index = self.cells.page_index_or_insert(base);
        self.hot = (base, index);
        index
    }

    /// The cell at offset `off` of page `index` (address `addr`), recorded as
    /// an entry at the window's first access to it.
    #[inline]
    fn touch_at(&mut self, index: usize, off: usize, addr: u64) -> PassCell {
        let mut cell = self.cells.page_get(index, off);
        if cell.window != self.window {
            self.entries.push((addr, cell.value, cell.ts));
            cell.window = self.window;
            self.cells.page_set(index, off, cell);
        }
        cell
    }

    #[inline]
    fn read(&mut self, addr: u64) -> u8 {
        let (base, off) = PagedMem::<PassCell>::page_of(addr);
        let index = self.page(base);
        self.touch_at(index, off, addr).value
    }

    #[inline]
    fn write(&mut self, addr: u64, value: u8, ts: u64) {
        let (base, off) = PagedMem::<PassCell>::page_of(addr);
        let index = self.page(base);
        let mut cell = self.touch_at(index, off, addr);
        cell.value = value;
        cell.ts = ts;
        self.cells.page_set(index, off, cell);
    }

    /// `count` bytes from `base`, read (a memory access reads all eight).
    fn touch(&mut self, base: u64, count: usize) {
        let (page, off) = PagedMem::<PassCell>::page_of(base);
        if off + count <= DEFAULT_PAGE_SIZE {
            let index = self.page(page);
            for i in 0..count {
                self.touch_at(index, off + i, base + i as u64);
            }
        } else {
            for i in 0..count {
                self.read(base.wrapping_add(i as u64));
            }
        }
    }

    /// The little-endian dword at `base`, read byte by byte.
    fn read_dword(&mut self, base: u64) -> u64 {
        (0..8u64).fold(0, |dword, i| {
            dword | (self.read(base.wrapping_add(i)) as u64) << (8 * i)
        })
    }

    /// `count` bytes of `value` written from `base` at `ts`.
    fn write_bytes(&mut self, base: u64, value: u64, count: usize, ts: u64) {
        let (page, off) = PagedMem::<PassCell>::page_of(base);
        if off + count <= DEFAULT_PAGE_SIZE {
            let index = self.page(page);
            for i in 0..count {
                let mut cell = self.touch_at(index, off + i, base + i as u64);
                cell.value = (value >> (8 * i)) as u8;
                cell.ts = ts;
                self.cells.page_set(index, off + i, cell);
            }
        } else {
            for i in 0..count {
                self.write(base.wrapping_add(i as u64), (value >> (8 * i)) as u8, ts);
            }
        }
    }

    /// The walk's state updates over `cpu_ops`, in the walk's order
    /// (`walk_cpu_ops`): the load or store, M1/M3/M5 and the pc, then each
    /// ecall's accesses.
    fn pass(&mut self, cpu_ops: &[CpuOperation], decode: &DecodeTable) {
        for op in cpu_ops {
            let op = decode.op(op);
            let f = op.decode.fields;
            let ts = op.timestamp;
            if f.is_load() || f.is_store() {
                let (count, _) = cpu_op_to_bytes_and_signed(op);
                self.touch(op.res, 8);
                let value = if f.is_load() { op.rvd } else { op.rv2 };
                self.write_bytes(op.res, value, count, ts);
            }
            let regs = &mut self.registers;
            if f.read_register1 && f.rs1 != 0 {
                if f.rs1 == 255 {
                    regs.write_pc(op.rv1, ts);
                } else {
                    regs.write(f.rs1, op.rv1, ts);
                }
            }
            if f.read_register2 && f.rs2 != 0 {
                regs.write(f.rs2, op.rv2, ts + 1);
            }
            if f.write_register && f.rd != 0 {
                regs.write(f.rd, op.rvd, ts + 2);
            }
            regs.write_pc(op.next_pc, ts + 1);
            if op.ecall_commit() {
                let (buf, count) = (op.commit_buf_addr(), op.commit_count());
                // The COMMIT rows read the bytes first, then the MEMW reads.
                for i in 0..count {
                    self.read(buf.wrapping_add(i));
                }
                let regs = &mut self.registers;
                regs.write(10, count, ts);
                regs.write(11, buf, ts);
                regs.write(12, count, ts);
                let index = regs.read_index().0;
                regs.write_index(index.wrapping_add(count as u32), ts);
                for i in 0..count {
                    let addr = buf.wrapping_add(i);
                    let value = self.read(addr);
                    self.write(addr, value, ts);
                }
            }
            if op.ecall_keccak() {
                let state = op.keccak_state_addr();
                let mut lanes = [0u64; 25];
                for (i, lane) in lanes.iter_mut().enumerate() {
                    *lane = self.read_dword(state + 8 * i as u64);
                }
                keccak_f1600(&mut lanes);
                self.registers.write(10, state, ts);
                for (i, &lane) in lanes.iter().enumerate() {
                    for b in 0..8 {
                        self.write(state + 8 * i as u64 + b, (lane >> (8 * b)) as u8, ts);
                    }
                }
            }
            if op.ecall_blake3() {
                let state = op.blake3_state_addr();
                let mut words = [0u32; 28];
                for k in 0..14usize {
                    let dword = self.read_dword(state + 8 * k as u64);
                    words[2 * k] = dword as u32;
                    words[2 * k + 1] = (dword >> 32) as u32;
                }
                let h: [u32; 8] = words[0..8].try_into().expect("eight words");
                let m: [u32; 16] = words[8..24].try_into().expect("sixteen words");
                let t = (words[24] as u64) | ((words[25] as u64) << 32);
                let out = blake3_compress_6round(&h, &m, t, words[26], words[27]);
                // The BLAKE3 op holds the output region's old bytes.
                for b in 0..64 {
                    self.read(state + 112 + b);
                }
                self.registers.write(10, state, ts);
                for k in 0..22usize {
                    let dword = if k < 14 {
                        (words[2 * k] as u64) | ((words[2 * k + 1] as u64) << 32)
                    } else {
                        (out[2 * (k - 14)] as u64) | ((out[2 * (k - 14) + 1] as u64) << 32)
                    };
                    for b in 0..8 {
                        self.write(state + 8 * k as u64 + b, (dword >> (8 * b)) as u8, ts);
                    }
                }
            }
            if op.ecall_blake3_absorb() {
                let regs = &mut self.registers;
                let (ctrl, msg) = (regs.read(10).0, regs.read(11).0);
                let (blocks, first_flags) = (regs.read(12).0, regs.read(13).0);
                for (reg, value) in [(10u8, ctrl), (11, msg), (12, blocks), (13, first_flags)] {
                    regs.write(reg, value, ts);
                }
                // A pure 8-byte read, its bytes written back at `ts`.
                let read_dword = |pass: &mut Self, addr: u64| {
                    let dword = pass.read_dword(addr);
                    pass.write_bytes(addr, dword, 8, ts);
                    dword
                };
                let mut cv_in = [0u32; 8];
                for k in 0..4u64 {
                    let dword = read_dword(self, ctrl + 8 * k);
                    cv_in[2 * k as usize] = dword as u32;
                    cv_in[2 * k as usize + 1] = (dword >> 32) as u32;
                }
                let mut message = Vec::new();
                for i in 0..blocks {
                    let mut m = [0u32; 16];
                    for k in 0..8u64 {
                        let dword = read_dword(self, msg + i * BLAKE3_BLOCK_BYTES + 8 * k);
                        m[2 * k as usize] = dword as u32;
                        m[2 * k as usize + 1] = (dword >> 32) as u32;
                    }
                    message.push(m);
                }
                let absorb = Blake3AbsorbOperation {
                    timestamp: ts,
                    ctrl_addr: ctrl,
                    msg_addr: msg,
                    first_flags: first_flags as u32,
                    cv_in,
                    blocks: message,
                    old_cv_out: [0; 32],
                };
                let cv = expand_absorb(&absorb)
                    .last()
                    .expect("an absorb group has rows")
                    .h;
                for k in 0..4u64 {
                    let addr = ctrl + (BLAKE3_ABSORB_CV_OUT_DWORD + k) * 8;
                    let dword =
                        (cv[2 * k as usize] as u64) | ((cv[2 * k as usize + 1] as u64) << 32);
                    self.write_bytes(addr, dword, 8, ts);
                }
            }
            if op.ecall_ecsm() {
                let regs = &self.registers;
                let (xr, xg, k) = (regs.read(10).0, regs.read(11).0, regs.read(12).0);
                let (mut xg_bytes, mut k_bytes) = ([0u8; 32], [0u8; 32]);
                for i in 0..32 {
                    xg_bytes[i] = self.read(xg.wrapping_add(i as u64));
                    k_bytes[i] = self.read(k.wrapping_add(i as u64));
                }
                let witness = ::ecsm::compute_witness(&k_bytes, &xg_bytes)
                    .expect("ECSM witness: the executor validated the operands");
                for (reg, addr, bytes, at) in [
                    (11u8, xg, &witness.x_g, ts),
                    (12, k, &witness.k, ts + 1),
                    (10, xr, &witness.x_r, ts + 2),
                ] {
                    let value = self.registers.read(reg).0;
                    self.registers.write(reg, value, at);
                    for i in 0..4usize {
                        let dword =
                            (0..8).fold(0u64, |d, j| d | (bytes[8 * i + j] as u64) << (8 * j));
                        self.write_bytes(addr.wrapping_add(8 * i as u64), dword, 8, at);
                    }
                }
            }
            if op.ecall_hint() {
                let regs = &mut self.registers;
                let (id, input, out) = (regs.read(10).0, regs.read(11).0, regs.read(12).0);
                for (reg, value) in [(10u8, id), (11, input), (12, out)] {
                    regs.write(reg, value, ts);
                }
                let mut bytes = [0u8; 32];
                for (i, byte) in bytes.iter_mut().enumerate() {
                    *byte = self.read(input.wrapping_add(i as u64));
                }
                let result = compute_hint(id, &bytes);
                for i in 0..4usize {
                    let dword = (0..8).fold(0u64, |d, j| d | (result[8 * i + j] as u64) << (8 * j));
                    self.write_bytes(out.wrapping_add(8 * i as u64), dword, 8, ts);
                }
            }
        }
    }

    /// The pass's value and timestamp of `addr`.
    fn cell(&self, addr: u64) -> (u8, u64) {
        let cell = self.cells.get(addr);
        (cell.value, cell.ts)
    }
}

/// Memory cells (value and timestamp) and register cells where the pass's
/// state differs from the walk's.
fn pass_differences(
    pass: &StatePass,
    memory: &MemoryState,
    registers: &RegisterState,
) -> (usize, usize) {
    let mut bases: Vec<u64> = memory
        .cells
        .page_bases()
        .chain(pass.cells.page_bases())
        .collect();
    bases.sort_unstable();
    bases.dedup();
    let mut cells = 0;
    for base in bases {
        let (walk, ours) = (memory.cells.page_data(base), pass.cells.page_data(base));
        for off in 0..DEFAULT_PAGE_SIZE {
            let w = walk.map_or((0, 0), |page| page[off]);
            let o = ours.map_or((0, 0), |page| (page[off].value, page[off].ts));
            cells += usize::from(w != o);
        }
    }
    let r = &pass.registers;
    let regs = (0..32).filter(|&i| r.regs[i] != registers.regs[i]).count()
        + usize::from(r.index_register != registers.index_register)
        + usize::from(r.pc_register != registers.pc_register);
    (cells, regs)
}

/// The bytes a window's memory MEMW rows say it accessed first, within each
/// access's width (old timestamp before the window's first): addresses,
/// sorted, distinct. Every one is a byte the window touches, so an entry.
fn census(window: &WalkedWindow, first_ts: u64) -> Vec<u64> {
    let mut out = Vec::new();
    for op in window.walk.memw.general.iter().filter(|op| !op.is_register) {
        for i in 0..op.width as usize {
            if op.old_timestamp[i] < first_ts {
                out.push(op.base_address.wrapping_add(i as u64));
            }
        }
    }
    for row in window
        .walk
        .memw
        .aligned
        .iter()
        .filter(|row| !row.is_register())
    {
        if row.old_timestamp() < first_ts {
            out.extend((0..row.width() as u64).map(|i| row.base_address().wrapping_add(i)));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// A memory state holding only `entries` (lean, as the walk's).
fn state_from_entries(entries: &[Entry]) -> MemoryState {
    let mut state = MemoryState::new();
    state.lean = true;
    for &(addr, value, ts) in entries {
        state.cells.set(addr, (value, ts));
    }
    state
}

/// Entry bytes that `state` does not hold as the pass left them after the
/// window (`exits`).
fn exit_misses(state: &MemoryState, exits: &[Entry]) -> usize {
    exits
        .iter()
        .filter(|&&(addr, value, ts)| state.read_byte(addr) != (value, ts))
        .count()
}

// =============================================================================
// Walking a window again, and comparing walks
// =============================================================================

/// What a window is walked with: the builder's decode table and walk settings.
struct WalkSetup {
    decode: Arc<DecodeTable>,
    lookups: bool,
    batch: usize,
}

impl WalkSetup {
    /// One window walked from `memory` and `registers` by the production
    /// walker, its lists sized from `sizes` as the sequential walk sized them;
    /// and the walk's seconds.
    fn walk(
        &self,
        memory: &mut MemoryState,
        registers: &mut RegisterState,
        first: usize,
        sizes: WalkSizes,
        logs: &[Log],
    ) -> (WalkedWindow, f64) {
        let (mut sizes, mut cycles, mut windows, mut secs) = (sizes, first, 0, 0.0);
        let walked = Walker {
            decode: &self.decode,
            lookups: self.lookups,
            batch: self.batch,
            sizes: &mut sizes,
            memory_state: memory,
            register_state: registers,
            cycles: &mut cycles,
            windows: &mut windows,
            walk_secs: &mut secs,
        }
        .walk(logs)
        .expect("a window walks");
        (walked, secs)
    }
}

/// The lists two walks of one window differ in, by name: none for the same walk.
fn walk_differences(a: &WalkedWindow, b: &WalkedWindow) -> Vec<&'static str> {
    fn debug<T: std::fmt::Debug>(x: &[T], y: &[T]) -> bool {
        x.len() == y.len() && format!("{x:?}") == format!("{y:?}")
    }
    let (x, y) = (&a.walk, &b.walk);
    [
        ("cpu", a.cpu_ops == b.cpu_ops),
        ("memw_r", x.memw.register_rows == y.memw.register_rows),
        ("memw_a", x.memw.aligned == y.memw.aligned),
        ("memw", x.memw.general == y.memw.general),
        ("load", x.load_ops == y.load_ops),
        ("lt", x.lt_ops == y.lt_ops),
        ("shift", x.shift_ops == y.shift_ops),
        ("bitwise", x.bitwise_ops == y.bitwise_ops),
        ("cpu32", x.cpu32_ops == y.cpu32_ops),
        ("commit", debug(&x.commit_ops, &y.commit_ops)),
        ("keccak", debug(&x.keccak_ops, &y.keccak_ops)),
        ("blake3", debug(&x.blake3_ops, &y.blake3_ops)),
        (
            "blake3_absorb",
            debug(&x.blake3_absorb_ops, &y.blake3_absorb_ops),
        ),
        ("ecsm", debug(&x.ecsm_ops, &y.ecsm_ops)),
        ("ecdas", debug(&x.ecdas_ops, &y.ecdas_ops)),
        ("hint", debug(&x.hint_ops, &y.hint_ops)),
    ]
    .into_iter()
    .filter(|&(_, same)| !same)
    .map(|(name, _)| name)
    .collect()
}

// =============================================================================
// The sequential walk, with the state pass beside it
// =============================================================================

/// A window kept for walking again: its logs, the walk's state at its start,
/// the window the sequential walk walked, and its entry state.
struct Sample {
    window: usize,
    first: usize,
    logs: Vec<Log>,
    memory: MemoryState,
    registers: RegisterState,
    sizes: WalkSizes,
    reference: WalkedWindow,
    /// The sequential walk's seconds over it.
    secs: f64,
    entries: Vec<Entry>,
    /// The entry bytes as the pass left them after the window.
    exits: Vec<Entry>,
    /// [`census`]'s bytes, and how many of them are not entries.
    census: usize,
    census_missing: usize,
}

/// The sequential walk of a run's windows (all but the last, which is the
/// finish's), the state pass over each window just before it is walked.
struct Run {
    setup: WalkSetup,
    samples: Vec<Sample>,
    windows: usize,
    cycles: usize,
    walk_secs: f64,
    pass_secs: f64,
    entries: usize,
    /// The most entries a window has, and that window.
    most_entries: (usize, usize),
    /// Summed over the windows checked: cells and register cells where the
    /// pass's state differs from the walk's.
    pass_cells: usize,
    pass_registers: usize,
    /// Chunk jobs of the first windows not sampled (a generator load).
    jobs: Vec<ChunkJob>,
}

/// The run of `program` on `input`, `window` cycles a window, walked by a
/// builder as the block's prove makes it, with window `k` kept as a sample
/// when `sampled(k)`, the pass checked against the walk at each sample's start
/// (at every window's with `check_every_window`), and chunk jobs kept until
/// there are `jobs_wanted`.
fn walk_run(
    program: &Elf,
    input: &[u8],
    window: usize,
    max_rows: &MaxRowsConfig,
    sampled: impl Fn(usize) -> bool,
    check_every_window: bool,
    jobs_wanted: usize,
) -> Run {
    let mut builder = WindowedTraceBuilder::new(program, input, max_rows)
        .expect("the builder")
        .drop_streamed_ops()
        .expect("before any window");
    let setup = WalkSetup {
        decode: Arc::clone(&builder.decode),
        lookups: !builder.lean.lookups,
        batch: builder.walk_batch,
    };
    let mut pass = StatePass::new(&builder.memory_state, &builder.register_state);
    let mut executor = Executor::new(program, input.to_vec()).expect("the executor starts");
    let mut run = Run {
        setup,
        samples: Vec::new(),
        windows: 0,
        cycles: 0,
        walk_secs: 0.0,
        pass_secs: 0.0,
        entries: 0,
        most_entries: (0, 0),
        pass_cells: 0,
        pass_registers: 0,
        jobs: Vec::new(),
    };
    let mut cpu_ops = Vec::new();
    let mut held: Option<Vec<Log>> = None;
    while let Some(next) = executor
        .resume_with_limit(window)
        .expect("the run executes")
        .map(<[Log]>::to_vec)
    {
        // One window held back: the run's last is the finish's.
        let Some(logs) = held.replace(next) else {
            continue;
        };
        let (k, cycles) = (run.windows, logs.len());
        let first = builder.cycles;
        let sample = sampled(k);
        if sample || check_every_window {
            let (cells, regs) =
                pass_differences(&pass, &builder.memory_state, &builder.register_state);
            run.pass_cells += cells;
            run.pass_registers += regs;
        }
        let start = sample.then(|| {
            (
                builder.memory_state.clone(),
                builder.register_state.clone(),
                builder.walk_sizes,
            )
        });
        let t = Instant::now();
        pass.begin(u32::try_from(k).expect("the windows fit the pass's marks"));
        let batch = match run.setup.batch {
            0 => logs.len().max(1),
            n => n,
        };
        for (i, part) in logs.chunks(batch).enumerate() {
            cpu_ops.clear();
            collect_cpu_ops_into(part, &run.setup.decode, first + i * batch, &mut cpu_ops)
                .expect("the CPU ops");
            pass.pass(&cpu_ops, &run.setup.decode);
        }
        run.pass_secs += t.elapsed().as_secs_f64();
        run.entries += pass.entries.len();
        if pass.entries.len() > run.most_entries.0 {
            run.most_entries = (pass.entries.len(), k);
        }
        let (mut walker, mut accumulator) = builder.split();
        let t = Instant::now();
        let walked = walker.walk(&logs).expect("a window walks");
        let secs = t.elapsed().as_secs_f64();
        run.walk_secs += secs;
        if let Some((memory, registers, sizes)) = start {
            let entries = pass.entries.clone();
            let exits = entries
                .iter()
                .map(|&(addr, _, _)| {
                    let (value, ts) = pass.cell(addr);
                    (addr, value, ts)
                })
                .collect();
            let bytes = census(&walked, first as u64 * 4 + 4);
            let mut touched: Vec<u64> = entries.iter().map(|&(addr, _, _)| addr).collect();
            touched.sort_unstable();
            let census_missing = bytes
                .iter()
                .filter(|addr| touched.binary_search(addr).is_err())
                .count();
            run.samples.push(Sample {
                window: k,
                first,
                logs,
                memory,
                registers,
                sizes,
                reference: walked,
                secs,
                entries,
                exits,
                census: bytes.len(),
                census_missing,
            });
        } else if run.jobs.len() < jobs_wanted {
            run.jobs.extend(accumulator.absorb(walked));
        }
        run.windows += 1;
        run.cycles += cycles;
    }
    run
}

// =============================================================================
// Walking the samples again, in parallel
// =============================================================================

/// A sweep's reading: the wall of its walks, their seconds summed, the windows
/// whose walk differs from the sequential walk's, and (from entry states) the
/// entry bytes that did not end as the pass left them.
#[derive(Clone, Copy, Debug, Default)]
struct Sweep {
    wall: f64,
    window_secs: f64,
    mismatches: usize,
    exit_misses: usize,
}

/// Each sample walked again from a fresh copy of the walk's state at its
/// start, `threads` at a time (rounds of `threads` windows; a round's wall is
/// its slowest window's). The copies are made outside the timed walks.
fn sweep_copies(setup: &WalkSetup, samples: &[Sample], threads: usize) -> Sweep {
    let mut sweep = Sweep::default();
    for round in samples.chunks(threads) {
        let mut copies: Vec<(MemoryState, RegisterState)> = round
            .iter()
            .map(|s| (s.memory.clone(), s.registers.clone()))
            .collect();
        let t = Instant::now();
        let walked: Vec<(WalkedWindow, f64)> = std::thread::scope(|scope| {
            let walkers: Vec<_> = round
                .iter()
                .zip(copies.iter_mut())
                .map(|(s, (memory, registers))| {
                    scope.spawn(move || setup.walk(memory, registers, s.first, s.sizes, &s.logs))
                })
                .collect();
            walkers
                .into_iter()
                .map(|w| w.join().expect("a walker"))
                .collect()
        });
        sweep.wall += t.elapsed().as_secs_f64();
        for (s, (window, secs)) in round.iter().zip(&walked) {
            sweep.window_secs += secs;
            sweep.mismatches += usize::from(!walk_differences(&s.reference, window).is_empty());
        }
    }
    sweep
}

/// Each sample walked again from its entry state alone (the bytes the pass
/// recorded and the 35 register cells), on `threads` threads taking the next
/// window as they finish one. The states are built outside the timed walks.
fn sweep_entries(setup: &WalkSetup, samples: &[Sample], threads: usize) -> Sweep {
    type Slot = (MemoryState, RegisterState, Option<(WalkedWindow, f64)>);
    let slots: Vec<Mutex<Slot>> = samples
        .iter()
        .map(|s| Mutex::new((state_from_entries(&s.entries), s.registers.clone(), None)))
        .collect();
    let next = AtomicUsize::new(0);
    let t = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let (Some(slot), Some(s)) = (slots.get(i), samples.get(i)) else {
                        break;
                    };
                    let mut slot = slot.lock().expect("a slot");
                    let (memory, registers, walked) = &mut *slot;
                    *walked = Some(setup.walk(memory, registers, s.first, s.sizes, &s.logs));
                }
            });
        }
    });
    let mut sweep = Sweep {
        wall: t.elapsed().as_secs_f64(),
        ..Sweep::default()
    };
    for (s, slot) in samples.iter().zip(slots) {
        let (memory, _, walked) = slot.into_inner().expect("a slot");
        let (window, secs) = walked.expect("every sample was walked");
        sweep.window_secs += secs;
        sweep.mismatches += usize::from(!walk_differences(&s.reference, &window).is_empty());
        sweep.exit_misses += exit_misses(&memory, &s.exits);
    }
    sweep
}

/// [`sweep_copies`] at `threads` while `generators` threads generate `jobs`
/// (packed, as the layout does) round and round: the layout's memory traffic
/// beside the walkers. And the chunks they generated.
fn sweep_beside(
    setup: &WalkSetup,
    samples: &[Sample],
    threads: usize,
    generators: usize,
    jobs: &[ChunkJob],
) -> (Sweep, usize) {
    let generators = if jobs.is_empty() { 0 } else { generators };
    let stop = AtomicBool::new(false);
    let generated = AtomicUsize::new(0);
    let sweep = std::thread::scope(|scope| {
        for g in 0..generators {
            let (stop, generated) = (&stop, &generated);
            scope.spawn(move || {
                let mut i = g;
                while !stop.load(Ordering::Relaxed) {
                    drop(jobs[i % jobs.len()].clone().generate_as(TraceForm::Narrow));
                    generated.fetch_add(1, Ordering::Relaxed);
                    i += generators;
                }
            });
        }
        // The generators at their pace before the walks start.
        if generators > 0 {
            std::thread::sleep(Duration::from_millis(500));
        }
        let sweep = sweep_copies(setup, samples, threads);
        stop.store(true, Ordering::Relaxed);
        sweep
    });
    (sweep, generated.into_inner())
}

// =============================================================================
// The builder's setup, as built before and as built now
// =============================================================================

/// The median of `xs`.
fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

/// The builder's setup, each piece timed from nothing, three times in turn:
/// before (the image as a map of bytes, the memory state seeded byte by byte,
/// DECODE's artifacts, and the copy of DECODE's trace `drop_streamed_ops`
/// made) and now (the image as pages, the state seeded page by page, the
/// decode table; DECODE's trace on a thread of its own), and the builder as the
/// block's prove creates it now. The `PARALLEL WALK SETUP` line.
fn setup_line(program: &Elf, input: &[u8], max_rows: &MaxRowsConfig) -> String {
    let secs = |t: Instant| t.elapsed().as_secs_f64();
    let mut r: [Vec<f64>; 11] = Default::default();
    let mut image_bytes = 0;
    for _ in 0..3 {
        let t = Instant::now();
        let pages = build_initial_image_paged(program, input);
        r[0].push(secs(t));
        let t = Instant::now();
        let memory = MemoryState::from_pages(&pages);
        r[1].push(secs(t));
        let t = Instant::now();
        let instructions = decode::instructions_from_elf(program).expect("the instructions");
        let table = DecodeTable::from_instructions(&instructions);
        r[2].push(secs(t));
        image_bytes = pages.len();
        drop((pages, memory, instructions, table));
        let t = Instant::now();
        let map = build_initial_image(program, input);
        r[3].push(secs(t));
        let t = Instant::now();
        let memory = MemoryState::from_image(&map);
        r[4].push(secs(t));
        let t = Instant::now();
        let instructions = decode::instructions_from_elf(program).expect("the instructions");
        r[5].push(secs(t));
        let t = Instant::now();
        let (trace, _) = decode::generate_decode_trace(&instructions);
        r[6].push(secs(t));
        let t = Instant::now();
        let table = DecodeTable::from_instructions(&instructions);
        r[7].push(secs(t));
        let t = Instant::now();
        let copy = trace.clone();
        r[8].push(secs(t));
        drop((map, memory, instructions, trace, table, copy));
        let t = Instant::now();
        let mut builder = WindowedTraceBuilder::new(program, input, max_rows)
            .expect("the builder")
            .drop_streamed_ops()
            .expect("before any window");
        r[9].push(secs(t));
        r[10].push(builder.setup_stamps().image);
        // DECODE's trace thread done before the next piece is timed.
        if let Some(kept) = builder.kept.as_mut() {
            kept.decode.table();
        }
        drop(builder);
    }
    let first_before = r[3][0] + r[4][0] + r[5][0] + r[6][0] + r[7][0] + r[8][0];
    let first_now = r[0][0] + r[1][0] + r[2][0];
    let m: Vec<f64> = r.into_iter().map(median).collect();
    let before = m[3] + m[4] + m[5] + m[6] + m[7] + m[8];
    let now = m[0] + m[1] + m[2];
    format!(
        "PARALLEL WALK SETUP: before {before:.3} s (image map {:.3} · memory from the map {:.3} · decode artifacts {:.3} = instructions {:.3} + trace {:.3} + table {:.3} · trace copy {:.3}) · \
         image and memory before {:.3} s || now {now:.3} s (image pages {:.3} · memory from the pages {:.3} · decode table {:.3}) · builder {:.3} s (its image stamp {:.3}) || \
         saved {:.3} s · image bytes {image_bytes} · input bytes {} · medians of 3 (first run: before {first_before:.3} now {first_now:.3})",
        m[3],
        m[4],
        m[5] + m[6] + m[7],
        m[5],
        m[6],
        m[7],
        m[8],
        m[3] + m[4],
        m[0],
        m[1],
        m[2],
        m[9],
        m[10],
        before - now,
        input.len(),
    )
}

// =============================================================================
// The box test
// =============================================================================

/// The block's window: 2^20 cycles (`block_whir::BLOCK_WINDOW_LOG2`).
const WINDOW: usize = 1 << 20;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |v| {
        v.trim()
            .parse()
            .unwrap_or_else(|_| panic!("{name}={v}: a count"))
    })
}

/// The executor alone over the run, a window at a time, each window's logs
/// copied out as the block's executor copies them: windows, cycles, seconds.
fn execute_alone(program: &Elf, input: &[u8]) -> (usize, usize, f64) {
    let t = Instant::now();
    let mut executor = Executor::new(program, input.to_vec()).expect("the executor starts");
    let (mut windows, mut cycles) = (0, 0);
    while let Some(logs) = executor
        .resume_with_limit(WINDOW)
        .expect("the run executes")
    {
        let logs = logs.to_vec();
        windows += 1;
        cycles += logs.len();
        drop(logs);
    }
    (windows, cycles, t.elapsed().as_secs_f64())
}

/// `count` windows spread over `0..walked`, the first and the last included.
fn spread(walked: usize, count: usize) -> Vec<usize> {
    if walked == 0 || count == 0 {
        return Vec::new();
    }
    let count = count.min(walked);
    let mut out: Vec<usize> = (0..count)
        .map(|i| {
            if count == 1 {
                0
            } else {
                i * (walked - 1) / (count - 1)
            }
        })
        .collect();
    out.dedup();
    out
}

/// ★ Box only (`--ignored`): can a real block's windows be walked in parallel?
/// Reads `BLOCK_WHIR_ELF` and `BLOCK_WHIR_INPUT`, and `PARALLEL_WALK_SAMPLES`
/// (32), `PARALLEL_WALK_THREADS` (`1,2,4,8,12,16`), `PARALLEL_WALK_BESIDE` (8),
/// `PARALLEL_WALK_GENERATORS` (6); `PARALLEL_WALK_SETUP_ONLY=1` prints the
/// setup line alone. Prints `PARALLEL WALK` lines (the module docs), then fails
/// if any walk again differs, any entry byte ends otherwise, or the pass's
/// state ever differs from the walk's.
#[test]
#[ignore = "a real block: box only"]
fn the_walk_in_parallel_from_window_states() {
    let read = |name: &str| {
        let path = std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"));
        std::fs::read(&path).unwrap_or_else(|e| panic!("{name}={path}: {e}"))
    };
    let program = Elf::load(&read("BLOCK_WHIR_ELF")).expect("the ELF loads");
    let input = read("BLOCK_WHIR_INPUT");
    let max_rows = MaxRowsConfig::uniform(1 << 21);
    println!("{}", setup_line(&program, &input, &max_rows));
    if std::env::var("PARALLEL_WALK_SETUP_ONLY").is_ok_and(|v| v.trim() == "1") {
        return;
    }
    let samples = env_usize("PARALLEL_WALK_SAMPLES", 32);
    let threads: Vec<usize> = std::env::var("PARALLEL_WALK_THREADS")
        .unwrap_or_else(|_| "1,2,4,8,12,16".to_string())
        .split(',')
        .map(|p| p.trim().parse().expect("PARALLEL_WALK_THREADS: counts"))
        .collect();
    assert_eq!(threads.first(), Some(&1), "the thread counts start at 1");
    let beside = env_usize("PARALLEL_WALK_BESIDE", 8);
    let generators = env_usize("PARALLEL_WALK_GENERATORS", 6);

    let (windows, cycles, exec) = execute_alone(&program, &input);
    println!(
        "PARALLEL WALK EXECUTOR: windows {windows} · cycles {cycles} · {exec:.3} s · {:.2} ns/cycle (resume + copy, alone)",
        exec * 1e9 / cycles as f64
    );
    let picked = spread(windows.saturating_sub(1), samples);
    let run = walk_run(
        &program,
        &input,
        WINDOW,
        &max_rows,
        |k| picked.binary_search(&k).is_ok(),
        false,
        3 * generators,
    );
    let per_cycle = |s: f64| s * 1e9 / run.cycles as f64;
    println!(
        "PARALLEL WALK SEQUENTIAL: windows walked {} · cycles {} · walk {:.3} s ({:.2} ns/cycle) · state pass {:.3} s ({:.2} ns/cycle) · samples {} at windows {:?}",
        run.windows,
        run.cycles,
        run.walk_secs,
        per_cycle(run.walk_secs),
        run.pass_secs,
        per_cycle(run.pass_secs),
        run.samples.len(),
        run.samples.iter().map(|s| s.window).collect::<Vec<_>>()
    );
    let sample_entries: usize = run.samples.iter().map(|s| s.entries.len()).sum();
    let census: usize = run.samples.iter().map(|s| s.census).sum();
    let census_missing: usize = run.samples.iter().map(|s| s.census_missing).sum();
    println!(
        "PARALLEL WALK ENTRIES: total {} ({:.4} a cycle) · largest window {} (window {}) · registers 35 · at the samples: entries {sample_entries} · census within width {census} · census bytes not among the entries {census_missing}",
        run.entries,
        run.entries as f64 / run.cycles as f64,
        run.most_entries.0,
        run.most_entries.1,
    );
    println!(
        "PARALLEL WALK PASS: samples compared {} · cells differing {} · register cells differing {}",
        run.samples.len(),
        run.pass_cells,
        run.pass_registers
    );
    let sequential: f64 = run.samples.iter().map(|s| s.secs).sum();
    let mut mismatches = 0;
    let mut misses = 0;
    let mut alone = Sweep::default();
    for &p in &threads {
        let sweep = sweep_copies(&run.setup, &run.samples, p);
        if p == 1 {
            alone = sweep;
        }
        mismatches += sweep.mismatches;
        println!(
            "PARALLEL WALK COPIES P={p}: wall {:.3} s · window seconds {:.3} s · speedup {:.2} · window time {:.3} of P=1's · mismatches {}",
            sweep.wall,
            sweep.window_secs,
            alone.wall / sweep.wall,
            sweep.window_secs / alone.window_secs,
            sweep.mismatches
        );
    }
    println!(
        "PARALLEL WALK FROM A COPY: P=1 window time {:.3} of the sequential walk's over the same windows ({:.3} s against {sequential:.3} s)",
        alone.window_secs / sequential,
        alone.window_secs
    );
    let mut first_entry = Sweep::default();
    for &p in &threads {
        let sweep = sweep_entries(&run.setup, &run.samples, p);
        if p == 1 {
            first_entry = sweep;
        }
        mismatches += sweep.mismatches;
        misses += sweep.exit_misses;
        println!(
            "PARALLEL WALK ENTRY P={p}: wall {:.3} s · window seconds {:.3} s · speedup {:.2} · window time {:.3} of P=1's · mismatches {} · exit misses {}",
            sweep.wall,
            sweep.window_secs,
            first_entry.wall / sweep.wall,
            sweep.window_secs / first_entry.window_secs,
            sweep.mismatches,
            sweep.exit_misses
        );
    }
    let (sweep, generated) = sweep_beside(&run.setup, &run.samples, beside, generators, &run.jobs);
    mismatches += sweep.mismatches;
    println!(
        "PARALLEL WALK BESIDE P={beside} G={generators}: wall {:.3} s · window seconds {:.3} s · speedup {:.2} over COPIES P=1 · window time {:.3} of P=1's · mismatches {} · chunks generated {generated} from {} jobs",
        sweep.wall,
        sweep.window_secs,
        alone.wall / sweep.wall,
        sweep.window_secs / alone.window_secs,
        sweep.mismatches,
        run.jobs.len()
    );
    println!(
        "PARALLEL WALK DONE: mismatches {mismatches} · exit misses {misses} · pass cells differing {} · pass register cells differing {} · census bytes not among the entries {census_missing}",
        run.pass_cells, run.pass_registers
    );
    assert_eq!(
        (
            mismatches,
            misses,
            run.pass_cells,
            run.pass_registers,
            census_missing
        ),
        (0, 0, 0, 0, 0),
        "(mismatches, exit misses, pass cells, pass registers, census missing)"
    );
}

// =============================================================================
// The laptop checks
// =============================================================================

/// Programs over every walk path: loads and stores of every width and offset,
/// word instructions, COMMIT, KECCAK, BLAKE3 and its absorb, ECSM; and HINT.
const PROGRAMS: [&str; 8] = [
    "all_instructions_64",
    "lw_sw_offset_odd",
    "test_memw_split_ts",
    "test_keccak_multi",
    "test_blake3",
    "test_blake3_absorb",
    "test_commit_split",
    "test_ecsm",
];

/// Rust guests, built by `make compile-programs-rust`: longer runs over a heap
/// and a stack, KECCAK and ECSM from Rust, and HINT.
const RUST_PROGRAMS: [&str; 4] = ["hashmap", "keccak_precompile", "ecsm", "hint_min"];

fn programs() -> Vec<(String, Elf)> {
    let rust = |name: &str| {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the workspace root")
            .join(format!("executor/program_artifacts/rust/{name}.elf"));
        std::fs::read(&path).unwrap_or_else(|e| {
            panic!("{} — run `make compile-programs-rust`: {e}", path.display())
        })
    };
    PROGRAMS
        .iter()
        .map(|name| (name.to_string(), asm_elf_bytes(name)))
        .chain(
            RUST_PROGRAMS
                .iter()
                .map(|name| (name.to_string(), rust(name))),
        )
        .map(|(name, bytes)| {
            let elf = Elf::load(&bytes).expect("the ELF loads");
            (name, elf)
        })
        .collect()
}

/// The run's cycles.
fn cycles_of(elf: &Elf) -> usize {
    Executor::new(elf, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs
        .len()
}

/// The window lengths a program is walked at: 7 and 33 cycles for a short
/// run, about a 40th and a 7th of a long one (odd lengths).
fn windows_for(cycles: usize) -> [usize; 2] {
    if cycles < 5_000 {
        [7, 33]
    } else {
        [(cycles / 40) | 1, (cycles / 7) | 1]
    }
}

/// The pass checked at every window's start, and 16 windows spread over the
/// run kept as samples (the last window is the finish's).
fn laptop_run(elf: &Elf, window: usize) -> Run {
    let cycles = cycles_of(elf);
    let picked = spread(cycles.div_ceil(window).saturating_sub(1), 16);
    walk_run(
        elf,
        &[],
        window,
        &MaxRowsConfig::small(),
        |k| picked.binary_search(&k).is_ok(),
        true,
        0,
    )
}

/// ★ The state pass keeps the walk's state: at the start of every window, of
/// every program, at windows of 7 and 33 cycles, its memory cells and register
/// cells are the walk's. And at the samples its entries hold every byte the
/// window's MEMW rows say the window accessed first.
#[test]
fn the_state_pass_keeps_the_walks_state_and_each_windows_entries() {
    let mut ecalls = [0usize; 3];
    let (mut entries, mut windows) = (0, 0);
    for (name, elf) in programs() {
        for window in windows_for(cycles_of(&elf)) {
            let run = laptop_run(&elf, window);
            windows += run.windows;
            assert_eq!(run.samples.len(), run.windows.min(16), "{name}");
            assert_eq!(
                (run.pass_cells, run.pass_registers),
                (0, 0),
                "{name}, windows of {window}: the pass's state differs from the walk's"
            );
            for s in &run.samples {
                assert_eq!(s.census_missing, 0, "{name}/{window}, window {}", s.window);
                let walk = &s.reference.walk;
                ecalls[0] += walk.keccak_ops.len() + walk.blake3_ops.len();
                ecalls[1] += walk.ecsm_ops.len() + walk.blake3_absorb_ops.len();
                ecalls[2] += walk.hint_ops.len();
            }
            entries += run.entries;
        }
    }
    assert!(
        ecalls.iter().all(|&n| n > 0),
        "the programs reach KECCAK/BLAKE3, ECSM/absorb and HINT: {ecalls:?}"
    );
    assert!(entries > 1000, "{entries} entries recorded");
    assert!(windows > 200, "{windows} windows walked");
}

/// ★ Each sample walked again, from a copy of the walk's state at its start
/// and from its entry state alone, on one thread and on three, is the window
/// the sequential walk walked; walked from its entry state, its entry bytes end
/// as the pass left them.
#[test]
fn each_sample_walks_alike_from_its_entry_state() {
    for (name, elf) in programs() {
        for window in windows_for(cycles_of(&elf)) {
            let run = laptop_run(&elf, window);
            for threads in [1, 3] {
                let copies = sweep_copies(&run.setup, &run.samples, threads);
                assert_eq!(
                    copies.mismatches, 0,
                    "{name}/{window}: from copies, P={threads}"
                );
                let entries = sweep_entries(&run.setup, &run.samples, threads);
                assert_eq!(
                    (entries.mismatches, entries.exit_misses),
                    (0, 0),
                    "{name}/{window}: from entry states, P={threads}"
                );
            }
        }
    }
}

/// A byte some memory row of the window reads at position 0 first in the
/// window, last written before it: `(address, value, timestamp)` before the
/// window.
fn written_entry(s: &Sample) -> Option<Entry> {
    let first_ts = s.first as u64 * 4 + 4;
    let general = s
        .reference
        .walk
        .memw
        .general
        .iter()
        .filter(|op| !op.is_register);
    let aligned = s
        .reference
        .walk
        .memw
        .aligned
        .iter()
        .filter(|row| !row.is_register());
    general
        .map(|op| (op.base_address, op.old_timestamp[0]))
        .chain(aligned.map(|row| (row.base_address(), row.old_timestamp())))
        .find(|&(_, ts)| ts > 0 && ts < first_ts)
        .and_then(|(addr, _)| s.entries.iter().copied().find(|e| e.0 == addr))
}

/// Whether `s` walked from `memory` (its first cycle `first`) is caught: its
/// walk differs from the sequential walk's, or its entry bytes in `memory` end
/// otherwise (`exits`), or the walk stops (an ecall's operands read wrong, e.g.
/// a point off the curve).
fn caught(
    setup: &WalkSetup,
    s: &Sample,
    mut memory: MemoryState,
    first: usize,
    exits: bool,
) -> bool {
    let mut registers = s.registers.clone();
    let walked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        setup.walk(&mut memory, &mut registers, first, s.sizes, &s.logs)
    }));
    match walked {
        Err(_) => true,
        Ok((walked, _)) => {
            !walk_differences(&s.reference, &walked).is_empty()
                || (exits && exit_misses(&memory, &s.exits) > 0)
        }
    }
}

/// ★ The checks are not vacuous: walked from a copy with one such byte's
/// timestamp moved, from an entry state without that byte, or from a first
/// cycle off by one, a window is caught.
#[test]
fn a_wrong_entry_state_or_first_cycle_is_caught() {
    let mut count = [0usize; 3];
    for (name, elf) in programs() {
        let run = laptop_run(&elf, windows_for(cycles_of(&elf))[0]);
        for s in &run.samples {
            assert!(
                caught(&run.setup, s, s.memory.clone(), s.first + 1, false),
                "{name}, window {}: a first cycle off by one walked alike",
                s.window
            );
            count[2] += 1;
            let Some((addr, value, ts)) = written_entry(s) else {
                continue;
            };
            let mut moved = s.memory.clone();
            moved.cells.set(addr, (value, ts - 1));
            assert!(
                caught(&run.setup, s, moved, s.first, false),
                "{name}, window {}: a moved timestamp at {addr:#x} walked alike",
                s.window
            );
            count[0] += 1;
            let without: Vec<Entry> = s.entries.iter().copied().filter(|e| e.0 != addr).collect();
            assert!(
                caught(&run.setup, s, state_from_entries(&without), s.first, true),
                "{name}, window {}: the entry state without {addr:#x} walked alike",
                s.window
            );
            count[1] += 1;
        }
    }
    assert!(count.iter().all(|&n| n >= 8), "windows caught: {count:?}");
}

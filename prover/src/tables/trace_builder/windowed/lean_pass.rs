//! The walk's timestamps alone, a window at a time: the serial stage that lets
//! a run's windows be walked on several threads.
//!
//! From one window to the next the walk carries each memory byte's value and
//! last-access timestamp and the 35 register cells. A window walked from the
//! bytes it touches as they stood before it, and the registers, is the window
//! the sequential walk walks. The values can come from the executor's memory
//! at the window's start ([`executor::vm::memory::MemorySnapshot`]); this pass
//! supplies the rest. Straight from each window's logs it keeps every byte's
//! last-access timestamp and the register cells, and records each byte's
//! address and timestamp at the byte's first access in the window (its
//! ENTRIES), whatever the access: a load's or a store's eight bytes, an
//! accelerator's state, ECSM's and HINT's inputs.
//!
//! It builds each cycle's [`CpuOperation`] as the walk does, so the register
//! values and addresses are the walk's, and keeps no memory value: no list is
//! built, and no accelerator is computed again. Its memory is one `u64` a byte,
//! the timestamp above the window that last touched the byte
//! ([`WINDOW_BITS`]), in 64 KiB pages found through directories as the
//! executor's are.

use std::collections::BTreeMap;
use std::sync::Arc;

use executor::vm::instruction::execution::{BLAKE3_ABSORB_CV_OUT_DWORD, BLAKE3_BLOCK_BYTES};
use executor::vm::logs::Log;

use super::super::{RegisterState, clock};
use crate::Error;
use crate::tables::cpu::{CpuOp, CpuOperation};
use crate::tables::decode::DecodeTable;

/// The low bits of a cell: the window that last touched the byte.
const WINDOW_BITS: u32 = 24;
const WINDOW_MASK: u64 = (1 << WINDOW_BITS) - 1;
/// No window touched the byte (the most windows a pass takes is one less).
const UNTOUCHED: u64 = WINDOW_MASK;
/// A byte never touched: timestamp 0, no window.
const FILL: u64 = UNTOUCHED;
/// A timestamp must fit above the window.
const MAX_TIMESTAMP: u64 = (1 << (64 - WINDOW_BITS)) - 1;

/// 64 KiB pages of bytes, a `u64` cell each.
const PAGE_BITS: u32 = 16;
const PAGE: usize = 1 << PAGE_BITS;
const PAGE_MASK: u64 = PAGE as u64 - 1;
/// The highest page number.
const LAST_PAGE: u64 = u64::MAX >> PAGE_BITS;
/// Pages each directory indexes: the low 8 GiB and the top 8 GiB, as the
/// executor's memory.
const DIRECTORY_PAGES: u64 = 1 << (33 - PAGE_BITS);
/// A directory slot with no page.
const NO_PAGE: u32 = u32::MAX;

/// The most pcs a dense pc→row index spans (64 MiB of `u32`); a program
/// spread wider finds rows by [`DecodeTable::row`].
const DENSE_ROWS: u64 = 1 << 24;

/// The DECODE row of a pc: one indexed read over the program's pcs, or the
/// table's own search when the pcs do not fit one dense index.
enum Rows {
    /// Row of `first + 4·i` at `rows[i]`, [`NO_PAGE`] where no instruction is.
    Dense {
        first: u64,
        rows: Vec<u32>,
    },
    Search,
}

impl Rows {
    fn of(decode: &DecodeTable) -> Self {
        let n = decode.padding_row();
        if n == 0 {
            return Self::Search;
        }
        let first = decode.entry(0).pc;
        let last = decode.entry(n as u32 - 1).pc;
        let span = (last - first) / 4 + 1;
        let aligned = (0..n as u32).all(|r| (decode.entry(r).pc - first).is_multiple_of(4));
        if span > DENSE_ROWS || !aligned {
            return Self::Search;
        }
        let mut rows = vec![NO_PAGE; span as usize];
        for r in 0..n as u32 {
            rows[((decode.entry(r).pc - first) / 4) as usize] = r;
        }
        Self::Dense { first, rows }
    }

    /// [`DecodeTable::row`]'s answer for `pc`.
    #[inline]
    fn row(&self, pc: u64, decode: &DecodeTable) -> Option<u32> {
        match self {
            Self::Dense { first, rows } => {
                let offset = pc.wrapping_sub(*first);
                if !offset.is_multiple_of(4) {
                    return None;
                }
                let row = *rows.get(usize::try_from(offset / 4).ok()?)?;
                (row != NO_PAGE).then_some(row)
            }
            Self::Search => decode.row(pc),
        }
    }
}

/// Every byte's cell: [`FILL`] until a window touches it.
struct Cells {
    pages: Vec<Box<[u64; PAGE]>>,
    low: Vec<u32>,
    high: Vec<u32>,
    other: BTreeMap<u64, u32>,
    /// `(page number, index in pages)` of the last page found; page numbers
    /// never reach `u64::MAX`.
    hot: (u64, u32),
}

impl Cells {
    fn new() -> Self {
        Self {
            pages: Vec::new(),
            low: Vec::new(),
            high: Vec::new(),
            other: BTreeMap::new(),
            hot: (u64::MAX, NO_PAGE),
        }
    }

    /// The index of page `number`, allocated on first touch.
    #[inline]
    fn page(&mut self, number: u64) -> usize {
        if self.hot.0 == number {
            return self.hot.1 as usize;
        }
        fn slot(directory: &mut Vec<u32>, index: usize) -> &mut u32 {
            if directory.len() <= index {
                directory.resize(index + 1, NO_PAGE);
            }
            &mut directory[index]
        }
        let slot = if number < DIRECTORY_PAGES {
            slot(&mut self.low, number as usize)
        } else if LAST_PAGE - number < DIRECTORY_PAGES {
            slot(&mut self.high, (LAST_PAGE - number) as usize)
        } else {
            self.other.entry(number).or_insert(NO_PAGE)
        };
        if *slot == NO_PAGE {
            *slot = u32::try_from(self.pages.len()).expect("fewer than 2^32 pages");
            let page: Box<[u64]> = vec![FILL; PAGE].into_boxed_slice();
            self.pages
                .push(page.try_into().expect("a page holds PAGE cells"));
        }
        self.hot = (number, *slot);
        *slot as usize
    }

    /// The cell of `address`, [`FILL`] for a page never touched.
    fn get(&self, address: u64) -> u64 {
        let number = address >> PAGE_BITS;
        let slot = if number < DIRECTORY_PAGES {
            self.low.get(number as usize).copied()
        } else if LAST_PAGE - number < DIRECTORY_PAGES {
            self.high.get((LAST_PAGE - number) as usize).copied()
        } else {
            self.other.get(&number).copied()
        };
        match slot {
            Some(index) if index != NO_PAGE => {
                self.pages[index as usize][(address & PAGE_MASK) as usize]
            }
            _ => FILL,
        }
    }
}

/// One byte accessed in window `window`: recorded at its first access in the
/// window, then, with `write`, stamped.
#[inline(always)]
fn access_cell(
    cell: &mut u64,
    address: u64,
    window: u64,
    write: Option<u64>,
    entries: &mut Vec<(u64, u64)>,
) {
    let mut c = *cell;
    if c & WINDOW_MASK != window {
        entries.push((address, c >> WINDOW_BITS));
        c = (c & !WINDOW_MASK) | window;
    }
    if let Some(ts) = write {
        c = (ts << WINDOW_BITS) | window;
    }
    *cell = c;
}

/// The 35 register cells, as the walk's [`RegisterState`].
#[derive(Clone)]
struct Registers {
    regs: [(u64, u64); 32],
    index: (u32, u64),
    pc: (u64, u64),
}

impl Registers {
    fn of(state: &RegisterState) -> Self {
        Self {
            regs: state.regs,
            index: state.index_register,
            pc: state.pc_register,
        }
    }

    fn state(&self) -> RegisterState {
        RegisterState {
            regs: self.regs,
            index_register: self.index,
            pc_register: self.pc,
        }
    }

    /// A register write as [`RegisterState::write`]: x0 is never written. The
    /// decode rows are checked to name registers below 32 ([`LeanPass::new`]).
    #[inline(always)]
    fn write(&mut self, reg: u8, value: u64, timestamp: u64) {
        if reg != 0 {
            self.regs[usize::from(reg & 31)] = (value, timestamp);
        }
    }

    #[inline(always)]
    fn value(&self, reg: u8) -> u64 {
        self.regs[usize::from(reg & 31)].0
    }
}

/// Where a pass's memory accesses go: the cells ([`Bytes`]), or a list
/// another thread applies to them (the split pass, a measurement's arm).
trait Accesses {
    /// `count` bytes from `base` accessed; with `write`, each then stamped.
    fn access(&mut self, base: u64, count: usize, write: Option<u64>);

    /// A load or a store: the eight bytes at `base` read, the first `count`
    /// stamped `ts`, as the walk's load and store collectors access them.
    fn load_or_store(&mut self, base: u64, count: usize, ts: u64);
}

/// Every byte's cell, the window being passed and its entries.
struct Bytes {
    cells: Cells,
    /// The window being passed, or [`UNTOUCHED`] before the first.
    window: u64,
    entries: Vec<(u64, u64)>,
}

impl Bytes {
    fn new() -> Self {
        Self {
            cells: Cells::new(),
            window: UNTOUCHED,
            entries: Vec::new(),
        }
    }

    /// Starts window `window`, its entries cleared. Windows are numbered from
    /// 0, below 2^24 − 1.
    fn begin(&mut self, window: u32) -> Result<(), Error> {
        if u64::from(window) >= UNTOUCHED {
            return Err(Error::Prover(format!(
                "the lean pass: window {window} is past the {UNTOUCHED} it can mark"
            )));
        }
        self.window = u64::from(window);
        self.entries.clear();
        Ok(())
    }
}

impl Accesses for Bytes {
    #[inline]
    fn access(&mut self, base: u64, count: usize, write: Option<u64>) {
        let off = (base & PAGE_MASK) as usize;
        if off + count <= PAGE {
            let index = self.cells.page(base >> PAGE_BITS);
            let page = &mut self.cells.pages[index];
            for (i, cell) in page[off..off + count].iter_mut().enumerate() {
                access_cell(cell, base + i as u64, self.window, write, &mut self.entries);
            }
        } else {
            for i in 0..count {
                let address = base.wrapping_add(i as u64);
                let index = self.cells.page(address >> PAGE_BITS);
                let cell = &mut self.cells.pages[index][(address & PAGE_MASK) as usize];
                access_cell(cell, address, self.window, write, &mut self.entries);
            }
        }
    }

    #[inline]
    fn load_or_store(&mut self, base: u64, count: usize, ts: u64) {
        let off = (base & PAGE_MASK) as usize;
        if off + 8 > PAGE {
            self.access(base, 8, None);
            self.access(base, count, Some(ts));
            return;
        }
        let window = self.window;
        let index = self.cells.page(base >> PAGE_BITS);
        let cells: &mut [u64; 8] = (&mut self.cells.pages[index][off..off + 8])
            .try_into()
            .expect("eight cells");
        if cells.iter().any(|&c| c & WINDOW_MASK != window) {
            for (i, cell) in cells.iter_mut().enumerate() {
                if *cell & WINDOW_MASK != window {
                    self.entries.push((base + i as u64, *cell >> WINDOW_BITS));
                    *cell = (*cell & !WINDOW_MASK) | window;
                }
            }
        }
        let stamped = (ts << WINDOW_BITS) | window;
        for cell in &mut cells[..count] {
            *cell = stamped;
        }
    }
}

/// An ECALL's register writes and accesses, as the walk makes them.
fn ecall(memory: &mut impl Accesses, regs: &mut Registers, op: CpuOp<'_>, ts: u64) {
    if op.ecall_commit() {
        let (buf, count) = (op.commit_buf_addr(), op.commit_count());
        regs.write(10, count, ts);
        regs.write(11, buf, ts);
        regs.write(12, count, ts);
        regs.index = (regs.index.0.wrapping_add(count as u32), ts);
        for k in 0..count {
            memory.access(buf.wrapping_add(k), 1, Some(ts));
        }
    }
    if op.ecall_keccak() {
        let state = op.keccak_state_addr();
        regs.write(10, state, ts);
        memory.access(state, 200, Some(ts));
    }
    if op.ecall_blake3() {
        let state = op.blake3_state_addr();
        regs.write(10, state, ts);
        memory.access(state, 176, Some(ts));
    }
    if op.ecall_blake3_absorb() {
        let (ctrl, msg) = (regs.value(10), regs.value(11));
        let (blocks, first_flags) = (regs.value(12), regs.value(13));
        for (reg, value) in [(10u8, ctrl), (11, msg), (12, blocks), (13, first_flags)] {
            regs.write(reg, value, ts);
        }
        memory.access(ctrl, 32, Some(ts));
        for b in 0..blocks {
            memory.access(
                msg.wrapping_add(b.wrapping_mul(BLAKE3_BLOCK_BYTES)),
                64,
                Some(ts),
            );
        }
        memory.access(
            ctrl.wrapping_add(BLAKE3_ABSORB_CV_OUT_DWORD * 8),
            32,
            Some(ts),
        );
    }
    if op.ecall_ecsm() {
        let (xr, xg, k) = (regs.value(10), regs.value(11), regs.value(12));
        for i in 0..32u64 {
            memory.access(xg.wrapping_add(i), 1, None);
            memory.access(k.wrapping_add(i), 1, None);
        }
        for (reg, address, at) in [(11u8, xg, ts), (12, k, ts + 1), (10, xr, ts + 2)] {
            let value = regs.value(reg);
            regs.write(reg, value, at);
            for i in 0..4u64 {
                memory.access(address.wrapping_add(8 * i), 8, Some(at));
            }
        }
    }
    if op.ecall_hint() {
        let (id, input, out) = (regs.value(10), regs.value(11), regs.value(12));
        for (reg, value) in [(10u8, id), (11, input), (12, out)] {
            regs.write(reg, value, ts);
        }
        memory.access(input, 32, None);
        for i in 0..4u64 {
            memory.access(out.wrapping_add(8 * i), 8, Some(ts));
        }
    }
}

/// What the pass reads each cycle with: the decode table and its pc index.
struct Decode {
    table: Arc<DecodeTable>,
    rows: Rows,
}

impl Decode {
    /// Refuses a program whose decode names a register the walk could not
    /// hold (32 and up, but the pc's 255 read through rs1).
    fn new(table: Arc<DecodeTable>) -> Result<Self, Error> {
        for row in 0..table.padding_row() as u32 {
            let f = &table.entry(row).fields;
            if (f.read_register1 && f.rs1 >= 32 && f.rs1 != 255)
                || (f.read_register2 && f.rs2 >= 32)
                || (f.write_register && f.rd >= 32)
            {
                return Err(Error::Prover(format!(
                    "the lean pass: the instruction at {:#x} names a register past x31",
                    table.entry(row).pc
                )));
            }
        }
        Ok(Self {
            rows: Rows::of(&table),
            table,
        })
    }
}

/// One window's logs from cycle `first`: the register cells kept in `regs`,
/// and (with `MEMORY`) the memory's accesses sent to `memory`.
fn pass_window<M: Accesses, const MEMORY: bool>(
    decode: &Decode,
    memory: &mut M,
    regs: &mut Registers,
    logs: &[Log],
    first: usize,
) -> Result<(), Error> {
    let most = clock::window_max_timestamp(first, logs.len());
    clock::check_clock(most, "the lean pass")?;
    if most > MAX_TIMESTAMP {
        return Err(Error::Prover(format!(
            "the lean pass: timestamp {most} does not fit its cells"
        )));
    }
    let table = &*decode.table;
    for (i, log) in logs.iter().enumerate() {
        let ts = ((first + i) as u64) * 4 + 4;
        let row = decode
            .rows
            .row(log.current_pc, table)
            .ok_or(Error::MissingInstruction(log.current_pc))?;
        let entry = table.entry(row);
        let op = CpuOperation::from_log(log, ts, entry, row);
        let f = &entry.fields;
        if MEMORY && f.memory {
            memory.load_or_store(op.res, f.mem_bytes(), ts);
        }
        if f.read_register1 && f.rs1 != 0 {
            if f.rs1 == 255 {
                regs.pc = (op.rv1, ts);
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
        regs.pc = (op.next_pc, ts + 1);
        if MEMORY && f.ecall {
            ecall(memory, regs, CpuOp::new(entry, &op), ts);
        }
    }
    Ok(())
}

/// The walk's timestamps and registers, a window at a time, and each window's
/// entries (see the module's documentation).
pub(crate) struct LeanPass {
    decode: Decode,
    bytes: Bytes,
    registers: Registers,
}

impl LeanPass {
    /// A pass from a run's start: no byte touched, and `registers` (the walk's
    /// starting register cells). Refuses a program whose decode names a
    /// register the walk could not hold.
    pub(crate) fn new(decode: Arc<DecodeTable>, registers: &RegisterState) -> Result<Self, Error> {
        Ok(Self {
            decode: Decode::new(decode)?,
            bytes: Bytes::new(),
            registers: Registers::of(registers),
        })
    }

    /// Starts window `window` (its entries cleared). Windows are numbered
    /// from 0, below 2^24 − 1.
    pub(crate) fn begin(&mut self, window: u32) -> Result<(), Error> {
        self.bytes.begin(window)
    }

    /// One window's logs from cycle `first`, in the walk's order.
    pub(crate) fn pass(&mut self, logs: &[Log], first: usize) -> Result<(), Error> {
        pass_window::<_, true>(
            &self.decode,
            &mut self.bytes,
            &mut self.registers,
            logs,
            first,
        )
    }

    /// [`Self::pass`] with the memory's accesses left out (decode and registers
    /// only): a measurement of where the pass's time goes.
    #[cfg(test)]
    pub(crate) fn pass_registers_only(&mut self, logs: &[Log], first: usize) -> Result<(), Error> {
        pass_window::<_, false>(
            &self.decode,
            &mut self.bytes,
            &mut self.registers,
            logs,
            first,
        )
    }

    /// The window's entries: each byte's address and its timestamp before the
    /// window, in the order the window first accessed them.
    pub(crate) fn entries(&self) -> &[(u64, u64)] {
        &self.bytes.entries
    }

    /// The register cells now, as the walk's.
    pub(crate) fn registers(&self) -> RegisterState {
        self.registers.state()
    }

    /// The last-access timestamp of the byte at `address` (0 if never
    /// stamped).
    pub(crate) fn timestamp(&self, address: u64) -> u64 {
        self.bytes.cells.get(address) >> WINDOW_BITS
    }

    /// Whether rows are found through a dense pc index (else by the decode
    /// table's search).
    pub(crate) fn dense_rows(&self) -> bool {
        matches!(self.decode.rows, Rows::Dense { .. })
    }
}

#[cfg(test)]
pub(crate) mod split;
#[cfg(test)]
mod tests;

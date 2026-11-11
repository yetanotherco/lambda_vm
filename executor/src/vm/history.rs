//! The access history a run's walk needs, written by the fast engine as it
//! runs (`LAMBDA_VM_EXEC_HISTORY`, the prover's record mode).
//!
//! The prover's walk keeps, for every memory byte and register, the value and
//! the timestamp of its last access, and reads them at each access (the old
//! value and old timestamp of the MEMW rows). Today a lean pass and the
//! walkers rebuild that state window by window. In history mode the engine
//! keeps the timestamps itself — a `u32` per byte ([`TsShadow`]) and the 35
//! register cells ([`RegisterCells`]) — stamped by the walk's own rules (the
//! prover's `lean_pass.rs` states them; [`step_registers`] and
//! [`ecall_accesses`] restate them here), and writes, beside each cycle's
//! [`Log`], what the walk will read in that cycle:
//!
//! - every cycle: its register accesses' old timestamps and where its memory
//!   accesses are ([`HistExtra`], 16 bytes beside the 40-byte log);
//! - a load or store: the eight bytes at its address and their timestamp
//!   ([`MemAccess`], 24 bytes); eight timestamps that differ go to the
//!   window's [`WindowHistory::mixed`] list (32 bytes);
//! - an ECALL: every byte the walk touches, with its value and timestamp before
//!   the call, as side ranges, and its chip's register cells;
//! - each window: the register cells at its start.
//!
//! Timestamps are the walk's: cycle `c` is at `4·c + 4`, its accesses at that
//! and up to two later. They are held as `u32`, so a run in history mode stops
//! (cleanly, [`HISTORY_MAX_CYCLES`]) where the no-epoch clock already does.
use std::collections::BTreeMap;

use super::{
    flat_memory::{FlatLayout, Region},
    instruction::decoding::Instruction,
    instruction::execution::{
        BLAKE3_ABSORB_CV_OUT_DWORD, BLAKE3_ABSORB_MAX_BLOCKS, BLAKE3_ABSORB_SYSCALL_NUMBER,
        BLAKE3_BLOCK_BYTES, BLAKE3_SYSCALL_NUMBER, DMA_MEMCPY_MAX_BYTES, DMA_MEMCPY_SYSCALL_NUMBER,
        DMA_MEMSET_SYSCALL_NUMBER, ECREC_SYSCALL_NUMBER, ECSM_SYSCALL_NUMBER, HINT_SYSCALL_NUMBER,
        KECCAK_SYSCALL_NUMBER, KECCAK256_SYSCALL_NUMBER, copy_row_width,
    },
    logs::Log,
    memory::{
        GuestMemory, MAX_PUBLIC_OUTPUT_TOTAL_SIZE, MemoryError, MemorySnapshot, PAGE_BITS,
        PAGE_SIZE,
    },
    registers::STACK_TOP,
};

/// The most cycles a run in history mode takes: its last cycle's latest
/// access, `4·c + 6`, must fit a `u32`.
pub const HISTORY_MAX_CYCLES: u64 = (1 << 30) - 2;

/// The knob that turns history mode on (`1`).
pub const HISTORY_KNOB: &str = "LAMBDA_VM_EXEC_HISTORY";

/// The walk's timestamp of cycle `cycle`.
#[inline(always)]
pub fn cycle_timestamp(cycle: u64) -> u64 {
    cycle * 4 + 4
}

/// No memory access this cycle ([`HistExtra::side`]); a load or store whose
/// eight bytes share one old timestamp ([`MemAccess::mixed`]).
pub const NO_ACCESS: u32 = u32::MAX;
/// [`HistExtra::side`] of an ECALL: this bit and the ECALL's number in the
/// window.
pub const ECALL: u32 = 1 << 31;

/// What a cycle's walk reads besides its log: 16 bytes beside the 40-byte
/// [`Log`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct HistExtra {
    /// The old timestamps of the cycle's `rs1` (the pc's for JAL and AUIPC),
    /// `rs2` and `rd` accesses, in the walk's order (an `rd` that is also
    /// `rs1` finds `rs1`'s stamp); 0 where the cycle makes none.
    pub regs_ts: [u32; 3],
    /// [`NO_ACCESS`]; a load's or store's index in
    /// [`WindowHistory::accesses`]; or [`ECALL`] and the ECALL's number `k`
    /// in the window: its bytes are `groups[k]`, its chip's register cells
    /// (x10 to x13 and the commit index x254, before the call)
    /// `reg_groups[k]`.
    pub side: u32,
}

/// A load or store: the eight bytes from its address before the cycle and
/// their timestamp.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct MemAccess {
    /// The access's address (the eight bytes from it are the walk's).
    pub addr: u64,
    /// Those eight bytes before the cycle, little-endian.
    pub old: u64,
    /// Their timestamp before the cycle, when they share one.
    pub ts: u32,
    /// [`NO_ACCESS`] when they share one; else the index of their eight
    /// timestamps in [`WindowHistory::mixed`].
    pub mixed: u32,
}

const _: () = assert!(std::mem::size_of::<HistExtra>() == 16);
const _: () = assert!(std::mem::size_of::<MemAccess>() == 24);

/// A register's cell before an access: x0 to x31, or 254 for the commit index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegAccess {
    pub reg: u8,
    pub value: u64,
    pub ts: u32,
}

/// A cycle's side ranges: `ranges[first..first + count]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SideGroup {
    pub first: u32,
    pub count: u32,
}

/// `len` bytes from `addr` (wrapping past `2^64`), their values and
/// timestamps before the cycle at `values[start..]` and `ts[start..]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SideRange {
    pub addr: u64,
    pub start: u32,
    pub len: u32,
}

/// The 35 register cells, `(value, timestamp)`, as the walk holds them: x0 to
/// x31, the commit index x254 and the pc x255.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterCells {
    pub regs: [(u64, u64); 32],
    pub index: (u32, u64),
    pub pc: (u64, u64),
}

impl RegisterCells {
    /// The walk's cells at a run's start: every register 0 but the stack
    /// pointer, the pc at the entry point, all at timestamp 1.
    pub fn at_start(entry_point: u64) -> Self {
        let mut regs = [(0u64, 1u64); 32];
        regs[2] = (STACK_TOP, 1);
        Self {
            regs,
            index: (0, 1),
            pc: (entry_point, 1),
        }
    }

    #[inline(always)]
    fn write(&mut self, reg: u8, value: u64, ts: u64) {
        if reg != 0 {
            self.regs[usize::from(reg & 31)] = (value, ts);
        }
    }

    #[inline(always)]
    fn value(&self, reg: u8) -> u64 {
        self.regs[usize::from(reg & 31)].0
    }
}

/// One window's history: its first cycle, the register cells at its start,
/// one [`HistExtra`] a cycle and the side lists.
#[derive(Clone, Debug, Default)]
pub struct WindowHistory {
    pub first: u64,
    pub registers: Option<RegisterCells>,
    pub extras: Vec<HistExtra>,
    /// The loads and stores, in order.
    pub accesses: Vec<MemAccess>,
    /// The old timestamps of the loads' and stores' eight bytes where they
    /// differ ([`MemAccess::mixed`]), one entry such an access.
    pub mixed: Vec<[u32; 8]>,
    /// ECALLs' memory, one group an ECALL: `ranges[first..first + count]`.
    pub groups: Vec<SideGroup>,
    pub ranges: Vec<SideRange>,
    pub values: Vec<u8>,
    pub ts: Vec<u32>,
    /// ECALLs' register accesses, one group an ECALL: `reg_cells[first..
    /// first + count]`.
    pub reg_groups: Vec<SideGroup>,
    pub reg_cells: Vec<RegAccess>,
}

impl WindowHistory {
    /// The bytes it takes on the heap.
    pub fn heap_bytes(&self) -> usize {
        self.extras.capacity() * std::mem::size_of::<HistExtra>()
            + self.accesses.capacity() * std::mem::size_of::<MemAccess>()
            + self.mixed.capacity() * std::mem::size_of::<[u32; 8]>()
            + self.groups.capacity() * std::mem::size_of::<SideGroup>()
            + self.ranges.capacity() * std::mem::size_of::<SideRange>()
            + self.values.capacity()
            + self.ts.capacity() * 4
            + self.reg_groups.capacity() * std::mem::size_of::<SideGroup>()
            + self.reg_cells.capacity() * std::mem::size_of::<RegAccess>()
    }

    /// Cycle `i`'s ECALL register accesses (empty for any other cycle).
    pub fn cycle_registers(&self, i: usize) -> &[RegAccess] {
        match self.extras[i].side {
            NO_ACCESS => &[],
            side if side & ECALL != 0 => {
                let g = self.reg_groups[(side & !ECALL) as usize];
                &self.reg_cells[g.first as usize..(g.first + g.count) as usize]
            }
            _ => &[],
        }
    }

    /// Calls `cell(address, value, timestamp)` for every byte cycle `i` of the
    /// window reads before the cycle, in order (a byte two ranges hold, once
    /// a range).
    pub fn cycle_cells(&self, i: usize, mut cell: impl FnMut(u64, u8, u32)) {
        match self.extras[i].side {
            NO_ACCESS => {}
            side if side & ECALL == 0 => {
                let a = self.accesses[side as usize];
                let mixed = (a.mixed != NO_ACCESS).then(|| &self.mixed[a.mixed as usize]);
                for (k, byte) in a.old.to_le_bytes().into_iter().enumerate() {
                    cell(
                        a.addr.wrapping_add(k as u64),
                        byte,
                        mixed.map_or(a.ts, |ts| ts[k]),
                    );
                }
            }
            side => {
                let g = self.groups[(side & !ECALL) as usize];
                for r in &self.ranges[g.first as usize..(g.first + g.count) as usize] {
                    for k in 0..r.len as usize {
                        let at = r.start as usize + k;
                        cell(r.addr.wrapping_add(k as u64), self.values[at], self.ts[at]);
                    }
                }
            }
        }
    }

    /// Each byte the window touches, with its value and timestamp before the
    /// window, in the order the window first touches them: the cells a walk
    /// of the window from its start state is seeded with.
    pub fn first_touches(&self) -> Vec<(u64, u8, u64)> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for i in 0..self.extras.len() {
            self.cycle_cells(i, |address, value, ts| {
                if seen.insert(address) {
                    out.push((address, value, u64::from(ts)));
                }
            });
        }
        out
    }
}

/// The run's end, in history mode: the memory and the timestamps after its
/// last cycle.
pub struct HistoryFinal {
    pub memory: MemorySnapshot,
    pub shadow: TsShadow,
    /// The register cells after the last cycle: what HALT's register reads
    /// find.
    pub registers: RegisterCells,
}

/// A `u32` timestamp a byte: flat over the engine's flat regions (lazily
/// zeroed), 64 KiB pages elsewhere. Zero until stamped.
pub struct TsShadow {
    layout: FlatLayout,
    low: Region,
    high: Region,
    other: BTreeMap<u64, Box<[u32]>>,
    /// One bit per flat page (low then high): stamped at least once.
    stamped: Vec<u64>,
    /// The flat pages stamped, by number, in the order first stamped.
    pages: Vec<u64>,
}

// Owned memory, no interior sharing (see `Region`); through `&self` it is
// only read, so shared reads from several threads are sound.
unsafe impl Send for TsShadow {}
unsafe impl Sync for TsShadow {}

impl TsShadow {
    /// An empty shadow over `layout`'s regions.
    pub fn new(layout: FlatLayout) -> Result<Self, MemoryError> {
        let low_pages = layout.low_len() >> PAGE_BITS;
        let pages = low_pages + (layout.high_len() >> PAGE_BITS);
        Ok(Self {
            layout,
            low: Region::new(layout.low_len() * 4)?,
            high: Region::new(layout.high_len() * 4)?,
            other: BTreeMap::new(),
            stamped: vec![0u64; pages.div_ceil(64)],
            pages: Vec::new(),
        })
    }

    /// The flat cell of `address`, or `None` outside the flat regions.
    #[inline(always)]
    fn flat(&self, address: u64) -> Option<*mut u32> {
        // SAFETY: an index inside a region of `len / 4` cells.
        unsafe {
            if address < self.layout.low_end {
                Some(self.low.ptr.cast::<u32>().add(address as usize))
            } else if address >= self.layout.high_base {
                Some(
                    self.high
                        .ptr
                        .cast::<u32>()
                        .add((address - self.layout.high_base) as usize),
                )
            } else {
                None
            }
        }
    }

    /// The flat page number of `address` (low pages, then high ones).
    #[inline(always)]
    fn flat_page(&self, address: u64) -> usize {
        if address < self.layout.low_end {
            (address >> PAGE_BITS) as usize
        } else {
            (self.layout.low_len() >> PAGE_BITS)
                + ((address - self.layout.high_base) >> PAGE_BITS) as usize
        }
    }

    /// The timestamp of the byte at `address`.
    #[inline(always)]
    pub fn get(&self, address: u64) -> u32 {
        match self.flat(address) {
            // SAFETY: a cell of a live region.
            Some(cell) => unsafe { *cell },
            None => self
                .other
                .get(&(address >> PAGE_BITS))
                .map_or(0, |page| page[(address as usize) & (PAGE_SIZE - 1)]),
        }
    }

    /// Stamps the byte at `address` with `ts`.
    #[inline(always)]
    pub fn set(&mut self, address: u64, ts: u32) {
        match self.flat(address) {
            Some(cell) => {
                let page = self.flat_page(address);
                if self.stamped[page / 64] >> (page % 64) & 1 == 0 {
                    self.stamped[page / 64] |= 1 << (page % 64);
                    self.pages.push(page as u64);
                }
                // SAFETY: a cell of a live region, held exclusively.
                unsafe { *cell = ts }
            }
            None => {
                self.other
                    .entry(address >> PAGE_BITS)
                    .or_insert_with(|| vec![0u32; PAGE_SIZE].into_boxed_slice())
                    [(address as usize) & (PAGE_SIZE - 1)] = ts;
            }
        }
    }

    /// The cells of `len > 0` bytes from `address` when they lie in one flat
    /// region, with the flat page numbers of the first and the last.
    #[inline(always)]
    fn flat_run(&self, address: u64, len: u64) -> Option<(*mut u32, usize, usize)> {
        let last = address.checked_add(len - 1)?;
        let inside = last < self.layout.low_end || address >= self.layout.high_base;
        if !inside {
            return None;
        }
        let cell = self.flat(address)?;
        Some((cell, self.flat_page(address), self.flat_page(last)))
    }

    /// Stamps the `len` bytes from `address` (wrapping) with `ts`.
    #[inline(always)]
    pub fn fill(&mut self, address: u64, len: u64, ts: u32) {
        if len == 0 {
            return;
        }
        let Some((cell, first, last)) = self.flat_run(address, len) else {
            return self.fill_slow(address, len, ts);
        };
        for page in first..=last {
            if self.stamped[page / 64] >> (page % 64) & 1 == 0 {
                self.stamped[page / 64] |= 1 << (page % 64);
                self.pages.push(page as u64);
            }
        }
        // SAFETY: `len` cells inside one live region (`flat_run`), held
        // exclusively.
        unsafe { std::slice::from_raw_parts_mut(cell, len as usize) }.fill(ts);
    }

    #[cold]
    #[inline(never)]
    fn fill_slow(&mut self, address: u64, len: u64, ts: u32) {
        for k in 0..len {
            self.set(address.wrapping_add(k), ts);
        }
    }

    /// Appends the timestamps of the `len` bytes from `address` (wrapping)
    /// to `out`.
    #[inline]
    pub fn extend_into(&self, address: u64, len: u64, out: &mut Vec<u32>) {
        if len == 0 {
            return;
        }
        match self.flat_run(address, len) {
            // SAFETY: `len` cells inside one live region (`flat_run`).
            Some((cell, _, _)) => {
                out.extend_from_slice(unsafe { std::slice::from_raw_parts(cell, len as usize) })
            }
            None => out.extend((0..len).map(|k| self.get(address.wrapping_add(k)))),
        }
    }

    /// The timestamps of the eight bytes from `address` (wrapping).
    #[inline(always)]
    fn get8(&self, address: u64) -> [u32; 8] {
        if address < self.layout.low_end - 7
            || (address >= self.layout.high_base && address <= u64::MAX - 7)
        {
            let cell = self.flat(address).expect("inside a flat region");
            // SAFETY: eight cells inside one region (checked above).
            return unsafe { cell.cast::<[u32; 8]>().read_unaligned() };
        }
        std::array::from_fn(|k| self.get(address.wrapping_add(k as u64)))
    }

    /// Every page holding a stamped byte: its base address and its 65,536
    /// timestamps.
    pub fn pages(&self) -> Vec<(u64, &[u32])> {
        let low_pages = (self.layout.low_len() >> PAGE_BITS) as u64;
        let mut out: Vec<(u64, &[u32])> = self
            .pages
            .iter()
            .map(|&page| {
                let base = if page < low_pages {
                    page << PAGE_BITS
                } else {
                    self.layout.high_base + ((page - low_pages) << PAGE_BITS)
                };
                let cell = self.flat(base).expect("a flat page");
                // SAFETY: a whole page of cells inside one region.
                (base, unsafe { std::slice::from_raw_parts(cell, PAGE_SIZE) })
            })
            .collect();
        out.extend(
            self.other
                .iter()
                .map(|(&page, cells)| (page << PAGE_BITS, &cells[..])),
        );
        out
    }
}

/// Which registers an instruction's walk reads and writes: `r1` (255 for the
/// pc, as JAL and AUIPC read it), `r2` and `rd`, 0 for none. The prover's
/// decode fields (`types.rs`, `from_instruction`) restated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegUse {
    pub r1: u8,
    pub r2: u8,
    pub rd: u8,
    /// A load's or store's width in bytes, 0 for no memory access.
    pub mem: u8,
    pub ecall: bool,
}

impl RegUse {
    /// The registers and memory width `instruction`'s walk uses.
    pub fn of(instruction: &Instruction) -> Self {
        use super::instruction::decoding::LoadStoreWidth as W;
        let reg = |r: u32| r as u8;
        let bytes = |w: W| match w {
            W::Byte | W::ByteUnsigned => 1,
            W::Half | W::HalfUnsigned => 2,
            W::Word | W::WordUnsigned => 4,
            W::DoubleWord => 8,
        };
        match *instruction {
            Instruction::Arith {
                dst, src1, src2, ..
            }
            | Instruction::ArithW {
                dst, src1, src2, ..
            } => Self {
                r1: reg(src1),
                r2: reg(src2),
                rd: reg(dst),
                ..Self::default()
            },
            Instruction::ArithImm { dst, src, .. } | Instruction::ArithImmW { dst, src, .. } => {
                Self {
                    r1: reg(src),
                    rd: reg(dst),
                    ..Self::default()
                }
            }
            Instruction::JumpAndLink { dst, .. } | Instruction::AddUpperImmToPc { dst, .. } => {
                Self {
                    r1: 255,
                    rd: reg(dst),
                    ..Self::default()
                }
            }
            Instruction::JumpAndLinkRegister { base, dst, .. } => Self {
                r1: reg(base),
                rd: reg(dst),
                ..Self::default()
            },
            Instruction::Store {
                src, base, width, ..
            } => Self {
                r1: reg(base),
                r2: reg(src),
                mem: bytes(width),
                ..Self::default()
            },
            Instruction::Load {
                dst, base, width, ..
            } => Self {
                r1: reg(base),
                rd: reg(dst),
                mem: bytes(width),
                ..Self::default()
            },
            Instruction::Branch { src1, src2, .. } => Self {
                r1: reg(src1),
                r2: reg(src2),
                ..Self::default()
            },
            Instruction::LoadUpperImm { dst, .. } => Self {
                rd: reg(dst),
                ..Self::default()
            },
            Instruction::EcallEbreak => Self {
                r1: 17,
                ecall: true,
                ..Self::default()
            },
            Instruction::Fence | Instruction::CSR { .. } => Self::default(),
        }
    }
}

/// The register cells after a cycle, by the walk's rules: `r1` read at `ts`,
/// `r2` at `ts + 1`, `rd` written at `ts + 2`, the pc at `ts + 1` (an ECALL's
/// next pc is the one after it, a halt's included). An ECALL's own register
/// accesses come after these ([`History::ecall_registers`]). Returns the old
/// timestamps of the `r1`, `r2` and `rd` accesses (0 for none).
#[inline(always)]
pub fn step_registers(cells: &mut RegisterCells, uses: RegUse, log: &Log, ts: u64) -> [u32; 3] {
    let mut old = [0u32; 3];
    if uses.r1 == 255 {
        old[0] = cells.pc.1 as u32;
        cells.pc = (log.current_pc, ts);
    } else if uses.r1 != 0 {
        old[0] = cells.regs[usize::from(uses.r1 & 31)].1 as u32;
        cells.write(uses.r1, log.src1_val, ts);
    }
    if uses.r2 != 0 {
        old[1] = cells.regs[usize::from(uses.r2 & 31)].1 as u32;
        cells.write(uses.r2, log.src2_val, ts + 1);
    }
    if uses.rd != 0 {
        old[2] = cells.regs[usize::from(uses.rd & 31)].1 as u32;
        cells.write(uses.rd, log.dst_val, ts + 2);
    }
    let next = if uses.ecall {
        log.current_pc.wrapping_add(4)
    } else {
        log.next_pc
    };
    cells.pc = (next, ts + 1);
    old
}

/// An ECALL's register cells, as the walk writes them.
fn ecall_registers(cells: &mut RegisterCells, log: &Log, ts: u64, before: &mut Vec<RegAccess>) {
    let number = log.src1_val;
    let note = |before: &mut Vec<RegAccess>, cells: &RegisterCells, reg: u8| {
        let (value, at) = cells.regs[usize::from(reg & 31)];
        before.push(RegAccess {
            reg,
            value,
            ts: at as u32,
        });
    };
    let keep = |cells: &mut RegisterCells, before: &mut Vec<RegAccess>, reg: u8, at: u64| {
        note(before, cells, reg);
        let value = cells.value(reg);
        cells.write(reg, value, at);
    };
    match number {
        64 => {
            let (buf, count) = (log.src2_val, log.dst_val);
            for reg in [10, 11, 12] {
                note(before, cells, reg);
            }
            before.push(RegAccess {
                reg: 254,
                value: u64::from(cells.index.0),
                ts: cells.index.1 as u32,
            });
            cells.write(10, count, ts);
            cells.write(11, buf, ts);
            cells.write(12, count, ts);
            cells.index = (cells.index.0.wrapping_add(count as u32), ts);
        }
        n if n == DMA_MEMCPY_SYSCALL_NUMBER
            || n == DMA_MEMSET_SYSCALL_NUMBER
            || n == KECCAK256_SYSCALL_NUMBER
            || n == HINT_SYSCALL_NUMBER =>
        {
            for reg in [10, 11, 12] {
                keep(cells, before, reg, ts);
            }
        }
        n if n == KECCAK_SYSCALL_NUMBER || n == BLAKE3_SYSCALL_NUMBER => {
            keep(cells, before, 10, ts)
        }
        n if n == BLAKE3_ABSORB_SYSCALL_NUMBER => {
            for reg in [10, 11, 12, 13] {
                keep(cells, before, reg, ts);
            }
        }
        n if n == ECSM_SYSCALL_NUMBER => {
            keep(cells, before, 11, ts);
            keep(cells, before, 12, ts + 1);
            keep(cells, before, 10, ts + 2);
        }
        n if n == ECREC_SYSCALL_NUMBER => {
            keep(cells, before, 11, ts);
            keep(cells, before, 12, ts + 1);
            keep(cells, before, 10, ts + 2);
            keep(cells, before, 13, ts + 1);
        }
        _ => {}
    }
}

/// An ECALL's memory accesses, in the walk's order: `access(address, len,
/// stamp)` for `len` bytes from `address` (wrapping), each then stamped with
/// `stamp` when it is `Some`. `regs` are the register cells before the call
/// (the ECALL's own register writes keep their values but commit's x10).
pub fn ecall_accesses(
    log: &Log,
    regs: &RegisterCells,
    ts: u64,
    mut access: impl FnMut(u64, u64, Option<u64>),
) {
    let number = log.src1_val;
    let r = |reg: u8| regs.value(reg);
    // Operands the executor refuses (so no call that ran has them) are not
    // walked: they only reach here before a call that then fails.
    let too_long = |len: u64, most: u64| len > most;
    match number {
        64 if too_long(log.dst_val, MAX_PUBLIC_OUTPUT_TOTAL_SIZE) => {}
        64 => access(log.src2_val, log.dst_val, Some(ts)),
        n if (n == DMA_MEMCPY_SYSCALL_NUMBER || n == DMA_MEMSET_SYSCALL_NUMBER)
            && too_long(r(12), DMA_MEMCPY_MAX_BYTES) => {}
        n if n == KECCAK256_SYSCALL_NUMBER && too_long(r(11), u64::from(u32::MAX)) => {}
        n if n == BLAKE3_ABSORB_SYSCALL_NUMBER && too_long(r(12), BLAKE3_ABSORB_MAX_BLOCKS) => {}
        n if n == DMA_MEMCPY_SYSCALL_NUMBER => {
            let (dst, src, count) = (r(10), r(11), r(12));
            access(src, count, Some(ts));
            access(dst, count, Some(ts + 1));
        }
        n if n == DMA_MEMSET_SYSCALL_NUMBER => {
            let (dst, src, count) = (r(10), r(11), r(12));
            let mut offset = 0u64;
            while offset < count {
                let width = u64::from(copy_row_width(src, dst, offset, count - offset, false));
                access(dst.wrapping_add(offset), width, Some(ts));
                access(src.wrapping_add(offset), width, Some(ts + 1));
                offset += width;
            }
        }
        n if n == KECCAK_SYSCALL_NUMBER => access(log.src2_val, 200, Some(ts)),
        n if n == KECCAK256_SYSCALL_NUMBER => {
            let (ptr, len, out) = (r(10), r(11), r(12));
            access(ptr, len, Some(ts));
            for i in 0..4u64 {
                access(out.wrapping_add(8 * i), 8, Some(ts + 1));
            }
        }
        n if n == BLAKE3_SYSCALL_NUMBER => access(log.src2_val, 176, Some(ts)),
        n if n == BLAKE3_ABSORB_SYSCALL_NUMBER => {
            let (ctrl, msg, blocks) = (r(10), r(11), r(12));
            access(ctrl, 32, Some(ts));
            for b in 0..blocks {
                access(
                    msg.wrapping_add(b.wrapping_mul(BLAKE3_BLOCK_BYTES)),
                    64,
                    Some(ts),
                );
            }
            access(
                ctrl.wrapping_add(BLAKE3_ABSORB_CV_OUT_DWORD * 8),
                32,
                Some(ts),
            );
        }
        n if n == ECSM_SYSCALL_NUMBER => {
            let (xr, xg, k) = (r(10), r(11), r(12));
            for i in 0..32u64 {
                access(xg.wrapping_add(i), 1, None);
                access(k.wrapping_add(i), 1, None);
            }
            for (address, at) in [(xg, ts), (k, ts + 1), (xr, ts + 2)] {
                for i in 0..4u64 {
                    access(address.wrapping_add(8 * i), 8, Some(at));
                }
            }
        }
        n if n == ECREC_SYSCALL_NUMBER => {
            let (out, sig, msg) = (r(10), r(11), r(12));
            for i in 0..64u64 {
                access(sig.wrapping_add(i), 1, None);
            }
            for i in 0..32u64 {
                access(msg.wrapping_add(i), 1, None);
            }
            for (address, at, dwords) in [(sig, ts, 8u64), (msg, ts + 1, 4), (out, ts + 2, 9)] {
                for i in 0..dwords {
                    access(address.wrapping_add(8 * i), 8, Some(at));
                }
            }
        }
        n if n == HINT_SYSCALL_NUMBER => {
            let (input, out) = (r(11), r(12));
            access(input, 32, None);
            for i in 0..4u64 {
                access(out.wrapping_add(8 * i), 8, Some(ts));
            }
        }
        _ => {}
    }
}

/// The engine's history: the shadow, the register cells, and the window being
/// written.
pub(crate) struct History {
    pub(crate) shadow: TsShadow,
    pub(crate) cells: RegisterCells,
    pub(crate) window: WindowHistory,
    /// A buffer for a later window's extras, given back or prepared
    /// ([`crate::vm::fast::FastExecutor::recycle_extras`]).
    pub(crate) spare_extras: Option<Vec<HistExtra>>,
    /// The side lists' lengths in the last window handed out
    /// ([`History::take_window`]): `accesses`, `mixed`, `groups`, `ranges`,
    /// `values` (and `ts`), `reg_groups`, `reg_cells`.
    last_sides: [usize; 7],
}

impl History {
    pub(crate) fn new(layout: FlatLayout, entry_point: u64) -> Result<Self, MemoryError> {
        Ok(Self {
            shadow: TsShadow::new(layout)?,
            cells: RegisterCells::at_start(entry_point),
            window: WindowHistory::default(),
            spare_extras: None,
            last_sides: [0; 7],
        })
    }

    /// Starts a window at cycle `first`. Its side lists start with room for
    /// what the last window's took, and a quarter more, so they rarely grow
    /// (a grown list is copied whole).
    pub(crate) fn begin(&mut self, first: u64, capacity: usize) {
        let extras = match self.spare_extras.take() {
            Some(mut extras) if extras.capacity() == capacity => {
                extras.clear();
                extras
            }
            _ => Vec::with_capacity(capacity),
        };
        let [
            accesses,
            mixed,
            groups,
            ranges,
            values,
            reg_groups,
            reg_cells,
        ] = self.last_sides.map(|n| n + n / 4);
        self.window = WindowHistory {
            first,
            registers: Some(self.cells),
            extras,
            accesses: Vec::with_capacity(accesses),
            mixed: Vec::with_capacity(mixed),
            groups: Vec::with_capacity(groups),
            ranges: Vec::with_capacity(ranges),
            values: Vec::with_capacity(values),
            ts: Vec::with_capacity(values),
            reg_groups: Vec::with_capacity(reg_groups),
            reg_cells: Vec::with_capacity(reg_cells),
        };
    }

    /// The window written, handed out; the next one's side lists are sized
    /// from it.
    pub(crate) fn take_window(&mut self) -> WindowHistory {
        let w = std::mem::take(&mut self.window);
        self.last_sides = [
            w.accesses.len(),
            w.mixed.len(),
            w.groups.len(),
            w.ranges.len(),
            w.values.len(),
            w.reg_groups.len(),
            w.reg_cells.len(),
        ];
        w
    }

    /// The eight bytes from `address` before a load or store: one
    /// [`MemAccess`], and their eight timestamps in [`WindowHistory::mixed`]
    /// when they differ. Returns the cycle's [`HistExtra::side`].
    #[inline(always)]
    pub(crate) fn before_access<M: GuestMemory>(&mut self, memory: &M, address: u64) -> u32 {
        let old = if address <= u64::MAX - 7 {
            u64::from_le_bytes(memory.read_n::<8>(address))
        } else {
            u64::from_le_bytes(std::array::from_fn(|k| {
                memory.load_byte(address.wrapping_add(k as u64))
            }))
        };
        let ts = self.shadow.get8(address);
        let w = &mut self.window;
        let mixed = if ts.iter().all(|&t| t == ts[0]) {
            NO_ACCESS
        } else {
            w.mixed.push(ts);
            w.mixed.len() as u32 - 1
        };
        w.accesses.push(MemAccess {
            addr: address,
            old,
            ts: ts[0],
            mixed,
        });
        w.accesses.len() as u32 - 1
    }

    /// A load's or store's stamp: its first `count` bytes at `ts`.
    #[inline(always)]
    pub(crate) fn after_access(&mut self, address: u64, count: u8, ts: u64) {
        self.shadow.fill(address, u64::from(count), ts as u32);
    }

    /// Every byte an ECALL's walk touches, before the call, as the window's
    /// next group (empty for none). Returns the cycle's [`HistExtra::side`].
    pub(crate) fn before_ecall<M: GuestMemory>(
        &mut self,
        memory: &M,
        log_before: &Log,
        ts: u64,
    ) -> u32 {
        let (shadow, cells, w) = (&self.shadow, &self.cells, &mut self.window);
        let first = w.ranges.len() as u32;
        ecall_accesses(log_before, cells, ts, |address, len, _| {
            let start = w.values.len() as u32;
            let from = w.values.len();
            if len > 0 && address.checked_add(len - 1).is_some() {
                w.values.resize(from + len as usize, 0);
                memory.read_range(address, &mut w.values[from..]);
            } else {
                w.values
                    .extend((0..len).map(|k| memory.load_byte(address.wrapping_add(k))));
            }
            shadow.extend_into(address, len, &mut w.ts);
            w.ranges.push(SideRange {
                addr: address,
                start,
                len: len as u32,
            });
        });
        let count = w.ranges.len() as u32 - first;
        w.groups.push(SideGroup { first, count });
        ECALL | (w.groups.len() as u32 - 1)
    }

    /// An ECALL's register accesses after [`step_registers`]: its chip's
    /// cells before them go to the window's `reg_cells` as its group (empty
    /// for none), the same number as its memory group.
    #[cold]
    #[inline(never)]
    pub(crate) fn ecall_registers(&mut self, log: &Log, ts: u64) {
        let w = &mut self.window;
        let first = w.reg_cells.len() as u32;
        ecall_registers(&mut self.cells, log, ts, &mut w.reg_cells);
        let count = w.reg_cells.len() as u32 - first;
        w.reg_groups.push(SideGroup { first, count });
    }

    /// An ECALL's stamps, in the walk's order, from the register cells
    /// before it.
    pub(crate) fn after_ecall(&mut self, log: &Log, cells_before: &RegisterCells, ts: u64) {
        let shadow = &mut self.shadow;
        ecall_accesses(log, cells_before, ts, |address, len, stamp| {
            if let Some(at) = stamp {
                shadow.fill(address, len, at as u32);
            }
        });
    }
}

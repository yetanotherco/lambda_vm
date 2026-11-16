//! ★ COPY_W (`tables::copy_w`) on hand-built traces with no proof, on the
//! word format's bench (`tables::word_memory_tests`): PAGE_W, MEMW_W, COPY_W,
//! a stand-in CPU (its ECALLs the verifier's), a stand-in register file (its
//! values the verifier's) and the verifier's commit pairs.
//!
//! The honest calls are run by the production call runner
//! ([`copy_w::run_call`]) against a word model of memory, and checked byte for
//! byte against a byte model of what memcpy, memmove, memset and the commit
//! syscall do. The bench itself is built by replaying the rows' word messages
//! in timestamp order, so a forger's rows (any field changed) get the MEMW_W
//! rows and PAGE_W finals that make memory as consistent as it can be: a
//! forgery is then refused only by the check it targets, and each one is also
//! run with that one check taken out (a control) and accepted.

use std::collections::HashMap;

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity};

use super::copy::Functionality;
use super::copy_w::{self, CallMemory, CopyWordConstraints, CopyWordRow, cols as cw};
use super::gpack::TraceForm;
use super::memw_w::{self, MemwWordConstraints, MemwWordRow, MessageKind, cols as mw};
use super::page::{DEFAULT_PAGE_SIZE, PageConfig};
use super::page_w::{self, FinalWordMap, FinalWordState, cols as pw};
use super::types::{BusId, FE};
use super::word_memory::{WORD_BYTES, column};
use super::word_memory_tests::{
    Bench, ELF_BASE, Eval, INPUT_BASE, Piece, STACK_BASE, Table, census_table, census_table_with,
    chain_of, constraint_values, page_name, page_piece, pages, table, verdict,
};
use executor::vm::instruction::execution::{DMA_MEMCPY_SYSCALL_NUMBER, DMA_MEMSET_SYSCALL_NUMBER};

/// The commit syscall's number (ecall 64).
const COMMIT_CALL: u64 = 64;

fn copy_w_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&CopyWordConstraints, row)
}

fn memw_w_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&MemwWordConstraints, row)
}

/// COPY_W's constraints with constraint `I` reading zero: a control mutant.
fn copy_w_without<const I: usize>(row: &[FE]) -> Vec<FE> {
    let mut v = copy_w_constraints(row);
    v[I] = FE::zero();
    v
}

/// `bench` with COPY_W's constraints replaced by `eval`.
fn with_constraints(mut bench: Bench, eval: Eval) -> Bench {
    bench.piece_mut(copy_w::NAME).constraints = Some(eval);
    bench
}

/// Only COPY_W's constraint `idx` breaks, on `rows` (every one listed), and
/// nothing else refuses the bench.
fn refused_by_constraint(bench: &Bench, idx: usize, rows: &[usize]) -> Result<(), String> {
    let v = verdict(bench);
    let expected: Vec<(String, usize, usize)> = rows
        .iter()
        .map(|&r| (copy_w::NAME.to_string(), r, idx))
        .collect();
    if v.left.is_empty()
        && v.unanswered.is_empty()
        && v.preprocessed.is_empty()
        && v.broken == expected
    {
        Ok(())
    } else {
        Err(format!("{v:?}"))
    }
}

// ---- the stand-ins ----

/// The stand-in CPU: one `Ecall` send a call, `(TS, 0, number_lo,
/// number_hi)`, every column the verifier's.
mod cpu {
    pub const TS: usize = 0;
    pub const NUMBER_LO: usize = 1;
    pub const NUMBER_HI: usize = 2;
    pub const MU: usize = 3;
    pub const NUM_COLUMNS: usize = 4;
}

fn cpu_interactions() -> Vec<BusInteraction> {
    vec![BusInteraction::sender(
        BusId::Ecall,
        Multiplicity::Column(cpu::MU),
        vec![
            column(cpu::TS),
            BusValue::constant(0),
            column(cpu::NUMBER_LO),
            column(cpu::NUMBER_HI),
        ],
    )]
}

/// The stand-in register file: one access a row, `old → new` at `TS`, in
/// MEMW's 24-value register shape; every column but `MU` the verifier's (a
/// forger may leave an access unanswered, never answer it with other values).
mod regs {
    pub const REG2: usize = 0;
    pub const OLD_LO: usize = 1;
    pub const OLD_HI: usize = 2;
    pub const NEW_LO: usize = 3;
    pub const NEW_HI: usize = 4;
    pub const TS: usize = 5;
    pub const MU: usize = 6;
    pub const NUM_COLUMNS: usize = 7;
}

fn register_file_interactions() -> Vec<BusInteraction> {
    let zeros = || std::iter::repeat_n(BusValue::constant(0), 6);
    let mut tuple = vec![column(regs::OLD_LO), column(regs::OLD_HI)];
    tuple.extend(zeros());
    tuple.extend([
        BusValue::constant(1),
        column(regs::REG2),
        BusValue::constant(0),
        column(regs::NEW_LO),
        column(regs::NEW_HI),
    ]);
    tuple.extend(zeros());
    tuple.extend([
        column(regs::TS),
        BusValue::constant(0),
        BusValue::constant(1),
        BusValue::constant(0),
        BusValue::constant(0),
    ]);
    vec![BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(regs::MU),
        tuple,
    )]
}

/// The verifier's commit pairs `(index, byte)`, rebuilt from the public
/// output.
mod output {
    pub const INDEX: usize = 0;
    pub const BYTE: usize = 1;
    pub const MU: usize = 2;
    pub const NUM_COLUMNS: usize = 3;
}

fn output_interactions() -> Vec<BusInteraction> {
    vec![BusInteraction::receiver(
        BusId::Commit,
        Multiplicity::Column(output::MU),
        vec![column(output::INDEX), column(output::BYTE)],
    )]
}

/// A table whose first `kept` columns are the verifier's.
fn truth_piece(name: &str, interactions: Vec<BusInteraction>, trace: Table, kept: usize) -> Piece {
    let preprocessed = (0..kept)
        .map(|c| {
            (0..trace.num_rows())
                .map(|r| *trace.get_main(r, c))
                .collect()
        })
        .collect();
    Piece {
        name: name.into(),
        interactions,
        trace,
        preprocessed,
        constraints: None,
    }
}

#[derive(Clone, Copy, Debug)]
struct RegAccess {
    reg: u64,
    old: u64,
    new: u64,
    ts: u64,
}

// ---- memory, as words and as bytes ----

/// The run's memory: its pages, each word's value and last timestamp (what
/// the call runner reads), and a byte model (what the calls must do).
#[derive(Clone)]
struct Memory {
    pages: Vec<PageConfig>,
    words: FinalWordMap,
    bytes: HashMap<u64, u8>,
}

impl Memory {
    fn page_of(&self, address: u64) -> Option<&PageConfig> {
        self.pages.iter().find(|p| {
            address
                .checked_sub(p.page_base)
                .is_some_and(|d| d < DEFAULT_PAGE_SIZE as u64)
        })
    }

    fn init_word(&self, key: u64) -> Option<u64> {
        let page = self.page_of(key)?;
        Some(page_w::init_word(
            page,
            ((key - page.page_base) / WORD_BYTES) as usize,
        ))
    }

    fn byte(&self, address: u64) -> u8 {
        if let Some(&b) = self.bytes.get(&address) {
            return b;
        }
        let key = address - address % WORD_BYTES;
        let word = self
            .init_word(key)
            .unwrap_or_else(|| panic!("a page holds {address:#x}"));
        (word >> (8 * (address % WORD_BYTES))) as u8
    }

    /// Every word the runner touched, against the byte model.
    fn check(&self) {
        for (&key, state) in &self.words {
            let bytes: Vec<u8> = (0..WORD_BYTES).map(|i| self.byte(key + i)).collect();
            assert_eq!(
                state.value.to_le_bytes().to_vec(),
                bytes,
                "the word at {key:#x}"
            );
        }
    }
}

impl CallMemory for Memory {
    fn word(&mut self, key: u64) -> Option<(u64, u64)> {
        if let Some(state) = self.words.get(&key) {
            return Some((state.value, state.timestamp));
        }
        self.init_word(key).map(|w| (w, 0))
    }

    fn set(&mut self, key: u64, timestamp: u64, value: u64, (first, len): (u64, u8)) {
        // Empty only for a forged row's word outside the call's range.
        assert!(
            len == 0 || (first >= key && first - key + u64::from(len) <= WORD_BYTES),
            "the touched bytes lie in the word"
        );
        self.words.insert(key, FinalWordState { timestamp, value });
    }
}

/// A run of calls: the rows, and what the stand-ins answer.
#[derive(Clone)]
struct Run {
    memory: Memory,
    rows: Vec<CopyWordRow>,
    /// The production runner's MEMW_W rows.
    memw: Vec<MemwWordRow>,
    calls: Vec<(u64, u64)>,
    regs: Vec<RegAccess>,
    commits: Vec<(u64, u8)>,
    index: u64,
    ts: u64,
}

impl Run {
    fn new() -> Self {
        Self::with_pages(pages())
    }

    fn with_pages(pages: Vec<PageConfig>) -> Self {
        Self {
            memory: Memory {
                pages,
                words: FinalWordMap::new(),
                bytes: HashMap::new(),
            },
            rows: Vec::new(),
            memw: Vec::new(),
            calls: Vec::new(),
            regs: Vec::new(),
            commits: Vec::new(),
            index: 0,
            ts: 4,
        }
    }

    fn next_ts(&mut self) -> u64 {
        self.ts += 4;
        self.ts
    }

    /// The CPU's ECALL and the register accesses of a call `(functionality,
    /// src, dst, count)` at `ts`, as the guest made it; and the byte model
    /// moved as the call must move it.
    fn guest_call(
        &mut self,
        functionality: Functionality,
        src: u64,
        dst: u64,
        count: u64,
        ts: u64,
    ) {
        let reg = |run: &mut Self, reg: u64, old: u64, new: u64| {
            run.regs.push(RegAccess { reg, old, new, ts });
        };
        match functionality {
            Functionality::Copy => {
                let snapshot: Vec<u8> = (0..count).map(|i| self.memory.byte(src + i)).collect();
                for (i, b) in snapshot.into_iter().enumerate() {
                    self.memory.bytes.insert(dst + i as u64, b);
                }
                self.calls.push((ts, DMA_MEMCPY_SYSCALL_NUMBER));
            }
            Functionality::Set => {
                for i in 0..count {
                    let b = self.memory.byte(src + i);
                    self.memory.bytes.insert(dst + i, b);
                }
                self.calls.push((ts, DMA_MEMSET_SYSCALL_NUMBER));
            }
            Functionality::Write => {
                for i in 0..count {
                    let b = self.memory.byte(src + i);
                    self.commits.push((dst + i, b));
                }
                self.calls.push((ts, COMMIT_CALL));
            }
        }
        reg(self, 11, src, src);
        reg(self, 12, count, count);
        if functionality == Functionality::Write {
            reg(self, 254, dst, dst + count);
            reg(self, 10, 1, count);
        } else {
            reg(self, 10, dst, dst);
        }
    }

    /// One call through the production rows and runner, its rows first
    /// passed through `forge` (nothing, for an honest call). The rows' range
    /// in [`Self::rows`].
    fn call_with(
        &mut self,
        functionality: Functionality,
        src: u64,
        dst: u64,
        count: u64,
        forge: impl FnOnce(&mut Vec<CopyWordRow>),
    ) -> std::ops::Range<usize> {
        let ts = self.next_ts();
        self.guest_call(functionality, src, dst, count, ts);
        let mut rows = copy_w::call_rows(functionality, ts, src, dst, count)
            .unwrap_or_else(|why| panic!("rows of the call: {why}"));
        forge(&mut rows);
        let memw = copy_w::run_call(&mut rows, src, dst, count, &mut self.memory)
            .unwrap_or_else(|why| panic!("the call runs: {why}"))
            .expect("every word is held");
        let at = self.rows.len();
        self.rows.extend(rows);
        self.memw.extend(memw);
        at..self.rows.len()
    }

    fn copy(&mut self, src: u64, dst: u64, count: u64) -> std::ops::Range<usize> {
        let r = self.call_with(Functionality::Copy, src, dst, count, |_| {});
        self.memory.check();
        r
    }

    fn set(&mut self, src: u64, count: u64) -> std::ops::Range<usize> {
        let r = self.call_with(Functionality::Set, src, src + 8, count, |_| {});
        self.memory.check();
        r
    }

    fn commit(&mut self, src: u64, count: u64) -> std::ops::Range<usize> {
        let index = self.index;
        self.index += count;
        let r = self.call_with(Functionality::Write, src, index, count, |_| {});
        self.memory.check();
        r
    }

    /// The rows' word messages `(timestamp, key, old, new)`, in the rows'
    /// order.
    fn messages(&self) -> Vec<(u64, u64, u64, u64)> {
        let mut out = Vec::new();
        for r in &self.rows {
            if r.reads_a() {
                out.push((r.read_timestamp(), r.a_key(), r.a(), r.a()));
            }
            if r.reads_b() {
                out.push((r.read_timestamp(), r.b_key(), r.b(), r.b()));
            }
            if r.writes() {
                out.push((r.write_timestamp(), r.dst_key(), r.old(), r.new_word()));
            }
        }
        out
    }

    /// The MEMW_W rows of the rows' messages replayed in timestamp order on
    /// the pages' initial words (an unaligned key, which no page holds,
    /// starts from the message's own old word at timestamp 0), each with the
    /// key it really names; and the aligned words' finals.
    fn replay(&self) -> (Vec<(MemwWordRow, u64)>, FinalWordMap) {
        let mut messages = self.messages();
        messages.sort_by_key(|m| m.0);
        let mut state: HashMap<u64, (u64, u64)> = HashMap::new();
        let mut out = Vec::new();
        for (ts, key, old, new) in messages {
            let (_, cur_ts) = *state.entry(key).or_insert_with(|| {
                let init = (key % WORD_BYTES == 0)
                    .then(|| self.memory.init_word(key))
                    .flatten();
                (init.unwrap_or(old), 0)
            });
            // `MemwWordRow::new` refuses an unaligned key: its aligned twin,
            // the key put back in the trace.
            let row = MemwWordRow::new(
                MessageKind::Access,
                key - key % WORD_BYTES,
                ts,
                cur_ts,
                old,
                new,
            )
            .expect("a replayed access");
            out.push((row, key));
            state.insert(key, (new, ts));
        }
        let finals = state
            .into_iter()
            .filter(|(key, _)| key % WORD_BYTES == 0 && self.memory.page_of(*key).is_some())
            .map(|(key, (value, timestamp))| (key, FinalWordState { timestamp, value }))
            .collect();
        (out, finals)
    }

    fn bench(&self) -> Bench {
        let (memw, finals) = self.replay();
        let mut pieces: Vec<Piece> = self
            .memory
            .pages
            .iter()
            .map(|config| page_piece(config, &finals))
            .collect();
        let rows: Vec<MemwWordRow> = memw.iter().map(|(r, _)| *r).collect();
        let mut trace = memw_w::generate_memw_w_trace_segments_as(&[&rows], TraceForm::Wide);
        for (i, (_, key)) in memw.iter().enumerate() {
            trace
                .main_table
                .set(i, mw::ADDRESS_LO, FE::from(*key & 0xFFFF_FFFF));
        }
        pieces.push(Piece {
            name: "MEMW_W".into(),
            interactions: memw_w::bus_interactions(),
            trace,
            preprocessed: Vec::new(),
            constraints: Some(memw_w_constraints),
        });
        pieces.push(Piece {
            name: copy_w::NAME.into(),
            interactions: copy_w::bus_interactions(),
            trace: copy_w::generate_copy_w_trace_as(&self.rows, TraceForm::Wide),
            preprocessed: Vec::new(),
            constraints: Some(copy_w_constraints),
        });
        let calls = table(self.calls.len(), cpu::NUM_COLUMNS, |t, row| {
            let (ts, number) = self.calls[row];
            t.main_table.set(row, cpu::TS, FE::from(ts));
            t.main_table
                .set(row, cpu::NUMBER_LO, FE::from(number & 0xFFFF_FFFF));
            t.main_table
                .set(row, cpu::NUMBER_HI, FE::from(number >> 32));
            t.main_table.set(row, cpu::MU, FE::one());
        });
        pieces.push(truth_piece(
            "CPU",
            cpu_interactions(),
            calls,
            cpu::NUM_COLUMNS,
        ));
        let regs = table(self.regs.len(), regs::NUM_COLUMNS, |t, row| {
            let r = self.regs[row];
            for (c, v) in [
                (regs::REG2, 2 * r.reg),
                (regs::OLD_LO, r.old & 0xFFFF_FFFF),
                (regs::OLD_HI, r.old >> 32),
                (regs::NEW_LO, r.new & 0xFFFF_FFFF),
                (regs::NEW_HI, r.new >> 32),
                (regs::TS, r.ts),
                (regs::MU, 1),
            ] {
                t.main_table.set(row, c, FE::from(v));
            }
        });
        pieces.push(truth_piece(
            "REGS",
            register_file_interactions(),
            regs,
            regs::MU,
        ));
        let out = table(self.commits.len(), output::NUM_COLUMNS, |t, row| {
            let (index, byte) = self.commits[row];
            t.main_table.set(row, output::INDEX, FE::from(index));
            t.main_table
                .set(row, output::BYTE, FE::from(u64::from(byte)));
            t.main_table.set(row, output::MU, FE::one());
        });
        pieces.push(truth_piece(
            "OUTPUT",
            output_interactions(),
            out,
            output::NUM_COLUMNS,
        ));
        Bench { pieces }
    }
}

/// `MemwWordRow`s as comparable tuples, sorted.
fn sorted(rows: impl IntoIterator<Item = MemwWordRow>) -> Vec<(u64, u64, u64, u64, u64)> {
    let mut v: Vec<_> = rows
        .into_iter()
        .map(|r| {
            (
                r.address(),
                r.timestamp(),
                r.old_timestamp(),
                r.old(),
                r.new_word(),
            )
        })
        .collect();
    v.sort();
    v
}

// ---- the honest calls ----

/// Where the honest calls copy to and from: the zero page and the ELF page.
const LOW: u64 = 0x1000;
const ELF: u64 = ELF_BASE + 0x800;

/// ★ Every combination of the destination's offset, the shift and a length
/// up to four words, as memcpy (disjoint and overlapping both ways, from the
/// ELF page and the private-input page), memset and the commit syscall
/// (zero-length calls included): the production runner moves the bytes as
/// the calls must (the byte model, word by word), its MEMW_W rows are the
/// replay of its rows' messages, and the bench is accepted.
#[test]
fn honest_calls_of_every_offset_shift_and_length_are_accepted() {
    let mut run = Run::new();
    let lengths = [0u64, 1, 2, 3, 5, 7, 8, 9, 12, 15, 16, 17, 23, 24, 31, 32];
    for offset in 0..8u64 {
        for shift in 0..8u64 {
            for &n in &lengths {
                let dst = LOW + 0x400 + offset;
                let src = ELF + 8 * shift + (offset + shift) % 8;
                run.copy(src, dst, n);
            }
        }
    }
    // Overlapping both ways, every distance up to two words.
    for d in 1..=16u64 {
        for &n in &[5u64, 16, 29] {
            run.copy(LOW + 0x200, LOW + 0x200 + d, n);
            run.copy(LOW + 0x300 + d, LOW + 0x300, n);
        }
    }
    // From the private-input page.
    run.copy(INPUT_BASE + 0x13, LOW + 0x600, 27);
    for offset in 0..8u64 {
        for &n in &lengths {
            run.set(LOW + 0x700 + offset, n);
        }
    }
    for offset in 0..8u64 {
        for &n in &[0u64, 1, 4, 7, 8, 9, 20] {
            run.commit(ELF + 0x40 + offset, n);
        }
    }
    assert_eq!(
        sorted(run.memw.iter().copied()),
        sorted(run.replay().0.into_iter().map(|(r, _)| r)),
        "the runner's MEMW_W rows are its rows' messages"
    );
    let v = verdict(&run.bench());
    assert!(v.accepted(), "{v:?}");
}

/// ★ Calls on the top stack page (high word `0xFFFFFFFF`), up to its very
/// last word: the high words ride the chain, and nothing wraps.
#[test]
fn honest_calls_on_the_top_stack_page_are_accepted() {
    let mut run = Run::new();
    let top = STACK_BASE + (DEFAULT_PAGE_SIZE as u64 - 1);
    run.copy(top - 0x100 + 4, top - 36, 37);
    run.set(top - 0x80 + 5, 40);
    run.copy(top - 0x3F, top - 0x200 + 2, 0x3F);
    run.commit(top - 20, 21);
    let v = verdict(&run.bench());
    assert!(v.accepted(), "{v:?}");
}

// ---- across 2^32 ----

/// `2^32`: low word 0 of high word 1, the word after the last of the low range.
const LINE: u64 = 1 << 32;
/// The private-input pages on either side of the line.
const BELOW: u64 = LINE - DEFAULT_PAGE_SIZE as u64;

/// The bench's pages and the two around the line.
fn line_pages() -> Vec<PageConfig> {
    let below: Vec<u8> = (0..DEFAULT_PAGE_SIZE as u32)
        .map(|i| (i * 89 + 7) as u8)
        .collect();
    let above: Vec<u8> = (0..4096u32).map(|i| (i * 53 + 201) as u8).collect();
    let mut out = pages();
    out.push(PageConfig::with_private_input(BELOW, below));
    out.push(PageConfig::with_private_input(LINE, above));
    out
}

/// Calls across the line, every destination offset and shift: copies whose
/// source crosses it (from just below the line, and from just above it, where
/// a first source word the call does not read is the last below the line),
/// copies whose destination crosses it, a memset across it and a commit from
/// a source across it.
fn across(run: &mut Run) {
    for offset in 0..8u64 {
        for shift in 0..8u64 {
            let rot = (offset + shift) % 8;
            run.copy(LINE - 16 + rot, BELOW + 0x400 + offset, 29);
            run.copy(LINE + rot, BELOW + 0x500 + offset, 13);
            run.copy(BELOW + 0x100 + 8 * shift + rot, LINE - 16 + offset, 27);
        }
    }
    run.copy(LINE - 0x40 + 3, LINE - 13, 64);
    run.copy(LINE - 13, LINE - 0x40 + 3, 64);
    run.set(LINE - 13, 30);
    run.commit(LINE - 5, 12);
}

/// ★ Calls across `2^32` (every destination offset and shift, sources and
/// destinations across the line, a memset and a commit across it): the
/// production runner moves the bytes as the calls must, its MEMW_W rows are
/// its rows' messages, every carry is 1 on some row (`SRC_CARRY`, `DST_CARRY`,
/// `OP_CARRY`), and the bench is accepted.
#[test]
fn honest_calls_across_2_to_the_32_are_accepted() {
    let mut run = Run::with_pages(line_pages());
    across(&mut run);
    assert_eq!(
        sorted(run.memw.iter().copied()),
        sorted(run.replay().0.into_iter().map(|(r, _)| r)),
        "the runner's MEMW_W rows are its rows' messages"
    );
    for (k, name) in ["SRC_CARRY", "DST_CARRY", "OP_CARRY"]
        .into_iter()
        .enumerate()
    {
        assert!(
            run.rows.iter().any(|r| r.carries()[k]),
            "{name} is 1 on no row"
        );
    }
    let v = verdict(&run.bench());
    assert!(v.accepted(), "{v:?}");
}

/// For each of COPY_W's carries, an honest row of [`across`]'s calls with it
/// 1 and an honest row of a call with every carry 0 and no key at the last
/// word of its range: what the census's rule c19 reads each carry on
/// (`word_memory_tests::check_chip_carry`).
pub(super) fn carry_rows() -> Vec<super::word_memory_tests::CarryRows> {
    let mut run = Run::with_pages(line_pages());
    across(&mut run);
    let t = copy_w::generate_copy_w_trace_as(&run.rows, TraceForm::Wide);
    let row = |r: usize| -> Vec<FE> { (0..cw::NUM_COLUMNS).map(|c| *t.get_main(r, c)).collect() };
    let off = run
        .rows
        .iter()
        .position(|r| {
            r.first()
                && !r.zero()
                && r.carries() == [false; 3]
                && [r.a_key(), r.dst_key()]
                    .iter()
                    .all(|k| k & 0xFFFF_FFFF != LINE - 8)
        })
        .expect("a call away from the line");
    [cw::SRC_CARRY, cw::DST_CARRY, cw::OP_CARRY]
        .into_iter()
        .enumerate()
        .map(|(k, c)| {
            let on = run
                .rows
                .iter()
                .position(|r| r.carries()[k])
                .expect("a carried row");
            (c, row(on), row(off))
        })
        .collect()
}

/// The rows refuse what COPY_W cannot hold (each a completeness limit the
/// module documents) with its reason, and take calls across `2^32`.
#[test]
fn calls_copy_w_cannot_hold_are_refused_with_their_reason() {
    let refused = |f: Functionality, src: u64, dst: u64, n: u64, why: &str| {
        let e = copy_w::call_rows(f, 8, src, dst, n).expect_err("refused");
        assert!(e.contains(why), "{e}");
    };
    refused(Functionality::Copy, 0x1000, 0x2000, 1 << 32, "below 2^32");
    refused(Functionality::Set, 0x1000, 0x1009, 8, "not src");
    refused(
        Functionality::Write,
        0x1000,
        (1 << 32) - 4,
        8,
        "passes 2^32",
    );
    // A destination's words past the top of the address space.
    refused(Functionality::Copy, 0x1000, u64::MAX - 3, 8, "2^64");
    // A source whose first word would sit below address 0.
    refused(Functionality::Copy, 3, 7, 8, "below address 0");
    // Across 2^32 (the keys carry into the high word): a destination, a
    // source whose first word is the last below the line, a memset.
    for (f, src, dst) in [
        (Functionality::Copy, 0x1000, (1 << 32) - 4),
        (Functionality::Copy, 1 << 32, 7),
        (Functionality::Copy, 0x1000, (1 << 32) - 8),
        (Functionality::Set, (1 << 32) - 13, (1 << 32) - 5),
    ] {
        assert!(
            copy_w::call_rows(f, 8, src, dst, 8).is_ok(),
            "{src:#x} → {dst:#x}"
        );
    }
}

// ---- forgeries ----

/// A fresh run with one call `(functionality, src, dst, n)` as the guest
/// made it, its rows as `forge` leaves them and their words read by the
/// runner: the byte model is not consulted, a forged call need not do what
/// the call does.
fn forged(
    functionality: Functionality,
    src: u64,
    dst: u64,
    n: u64,
    forge: impl FnOnce(&mut Vec<CopyWordRow>),
) -> Run {
    let mut run = Run::new();
    run.call_with(functionality, src, dst, n, forge);
    run
}

/// COPY_W's interactions with the chain's values at `positions` (in both
/// its tuples) taken out: a control mutant.
fn chain_without(positions: std::ops::Range<usize>) -> Vec<BusInteraction> {
    let mut out = copy_w::bus_interactions();
    for bi in &mut out {
        if bi.bus_id == u64::from(BusId::CopyWordNext) {
            for p in positions.clone() {
                bi.values[p] = BusValue::constant(0);
            }
        }
    }
    out
}

/// The chain tuple's shift value, and the carried word's eight bytes.
const CHAIN_SHIFT: std::ops::Range<usize> = 8..9;
const CHAIN_CARRY: std::ops::Range<usize> = 9..17;

/// ★ A memmove that changes its shift mid-call (the middle row places its
/// bytes from another rotation of the same source words): refused by the
/// chain alone, which carries the shift. Control: the shift taken out of the
/// chain's tuples, the rotated bytes are accepted.
#[test]
fn a_shift_changed_mid_call_is_refused_by_the_chain() {
    let run = forged(Functionality::Copy, ELF + 3, LOW + 0x400, 40, |rows| {
        assert!(rows.len() >= 3 && rows[1].shift == 3);
        rows[1].shift = 5;
    });
    let bench = run.bench();
    let v = verdict(&bench);
    assert!(v.refused_only_on(BusId::CopyWordNext), "{v:?}");
    let control = bench.with_interactions(copy_w::NAME, chain_without(CHAIN_SHIFT));
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ A row whose first source word is not the previous row's second (the
/// carry): the forger places bytes the call never read. Refused by the chain
/// alone, which carries the word. Control: the word taken out of the chain's
/// tuples, accepted.
#[test]
fn a_carried_word_not_the_previous_rows_is_refused_by_the_chain() {
    let mut run = forged(Functionality::Copy, ELF + 5, LOW + 0x400, 40, |_| {});
    // Row 1 takes another A: its new word and the replayed memory follow.
    assert!(!run.rows[1].reads_a() && run.rows[1].shift == 5);
    run.rows[1].a ^= 0x0101_0101_0101_0101;
    let bench = run.bench();
    let v = verdict(&bench);
    assert!(v.refused_only_on(BusId::CopyWordNext), "{v:?}");
    let control = bench.with_interactions(copy_w::NAME, chain_without(CHAIN_CARRY));
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// `run`'s bench with COPY_W row `row`'s new word (`NEW_LO`, `NEW_HI`)
/// replaced by `new`, the word's MEMW_W write and PAGE_W final following it
/// (the forger's write).
fn with_new_word(run: &Run, row: usize, new: u64) -> Bench {
    let r = run.rows[row];
    let (key, ts) = (r.dst_key(), r.write_timestamp());
    let mut bench = run.bench();
    bench.set(copy_w::NAME, row, cw::NEW_LO, FE::from(new & 0xFFFF_FFFF));
    bench.set(copy_w::NAME, row, cw::NEW_HI, FE::from(new >> 32));
    let n = bench.piece("MEMW_W").trace.num_rows();
    let m = (0..n)
        .find(|&m| {
            bench.get("MEMW_W", m, mw::ADDRESS_LO) == FE::from(key & 0xFFFF_FFFF)
                && bench.get("MEMW_W", m, mw::TIMESTAMP) == FE::from(ts)
        })
        .expect("the row's write");
    bench.set("MEMW_W", m, mw::NEW_LO, FE::from(new & 0xFFFF_FFFF));
    bench.set("MEMW_W", m, mw::NEW_HI, FE::from(new >> 32));
    // The word's final: its page row, this write being its last.
    let page = run.memory.page_of(key).expect("a page").page_base;
    let name = page_name(page);
    let p = ((key - page) / WORD_BYTES) as usize;
    assert_eq!(bench.get(&name, p, pw::TIMESTAMP), FE::from(ts));
    bench.set(&name, p, pw::FINI_LO, FE::from(new & 0xFFFF_FFFF));
    bench.set(&name, p, pw::FINI_HI, FE::from(new >> 32));
    bench
}

/// ★ An edge word's write that changes a byte outside the window (the first
/// row's byte below `dst`, the last row's byte past the end): refused by the
/// new word's identity alone (its low half, its high half). Control: that
/// identity removed, accepted.
#[test]
fn an_edge_write_changing_an_untouched_byte_is_refused_by_its_identity() {
    let run = forged(Functionality::Copy, ELF + 1, LOW + 0x403, 18, |_| {});
    let last = run.rows.len() - 1;
    assert_eq!(run.rows[0].window(), Some((3, 7)));
    assert_eq!(run.rows[last].window(), Some((0, 4)));
    // Byte 1 of the first word (its low half), byte 6 of the last (high).
    let cases: [(usize, usize, usize, Eval); 2] = [
        (0, 1, 54, copy_w_without::<54>),
        (last, 6, 55, copy_w_without::<55>),
    ];
    for (row, byte, idx, without) in cases {
        let new = run.rows[row].new_word() ^ (0x5Au64 << (8 * byte));
        let bench = with_new_word(&run, row, new);
        refused_by_constraint(&bench, idx, &[row]).unwrap_or_else(|v| panic!("{idx}: {v}"));
        let v = verdict(&with_constraints(bench, without));
        assert!(v.accepted(), "control {idx}: {v:?}");
    }
}

/// ★ A source word placed without being read: the first row of a shifted
/// copy that clears `READ_A` though its window takes `A`'s bytes, and the
/// last row that clears `READ_B` though its window reaches `B`, each placing
/// invented bytes. Refused by the read rule alone (constraint 43, 44).
/// Control: that rule removed, accepted.
#[test]
fn a_source_word_placed_but_not_read_is_refused_by_its_read_rule() {
    // Shift 6: the first window [0, 7] takes A's last two bytes; the last
    // window [0, 3] reaches B (3 + 6 ≥ 8).
    let (src, dst, n) = (ELF + 6, LOW + 0x400, 20);
    let cases: [(usize, bool, Eval); 2] = [
        (43, true, copy_w_without::<43>),
        (44, false, copy_w_without::<44>),
    ];
    for (idx, first, without) in cases {
        let mut run = forged(Functionality::Copy, src, dst, n, |rows| {
            let r = if first { 0 } else { rows.len() - 1 };
            let flag = if first {
                copy_w::FLAG_READ_A
            } else {
                copy_w::FLAG_READ_B
            };
            assert!(rows[r].flags & flag != 0, "the honest row reads it");
            rows[r].flags &= !flag;
        });
        // The unread word is the forger's to invent.
        let r = if first { 0 } else { run.rows.len() - 1 };
        if first {
            run.rows[r].a = 0x0F0E_0D0C_0B0A_0908;
        } else {
            run.rows[r].b = 0x1716_1514_1312_1110;
        }
        let bench = run.bench();
        refused_by_constraint(&bench, idx, &[r]).unwrap_or_else(|v| panic!("{idx}: {v}"));
        let v = verdict(&with_constraints(bench, without));
        assert!(v.accepted(), "control {idx}: {v:?}");
    }
}

/// ★ The windows cover exactly the call's bytes. A first row that stops
/// short of its word's end (the skipped byte left as it was, one byte past
/// the call's end written instead): refused by constraint 41 alone. A row
/// after the first that starts past byte 0: refused by constraint 40 alone.
/// A last row that copies one byte too few and leaves it in the count:
/// refused by constraint 42 alone. Each control (that constraint removed) is
/// accepted.
#[test]
fn windows_that_skip_a_byte_or_leave_count_are_refused() {
    // dst ≡ 1: windows [1, 7], [0, 7], [0, 7], [0, 6] (30 bytes); shift 1.
    let (src, dst, n) = (ELF + 2, LOW + 0x401, 30);
    // A byte moved from `skip`'s window to one past the last row's.
    let moved = |rows: &mut Vec<CopyWordRow>, skip: usize, from_start: bool| {
        if from_start {
            rows[skip].start += 1;
        } else {
            rows[skip].last -= 1;
        }
        for r in rows.iter_mut().skip(skip) {
            r.count_decr += 1;
        }
        let last = rows.len() - 1;
        rows[last].count_decr -= 1;
        rows[last].last += 1;
    };
    let forge_41 = |rows: &mut Vec<CopyWordRow>| moved(rows, 0, false);
    let forge_40 = |rows: &mut Vec<CopyWordRow>| moved(rows, 1, true);
    let forge_42 = |rows: &mut Vec<CopyWordRow>| {
        let last = rows.len() - 1;
        rows[last].last -= 1;
        rows[last].count_decr += 1;
    };
    type Forge<'a> = &'a dyn Fn(&mut Vec<CopyWordRow>);
    let cases: [(usize, Option<usize>, Forge<'_>, Eval); 3] = [
        (41, Some(0), &forge_41, copy_w_without::<41>),
        (40, Some(1), &forge_40, copy_w_without::<40>),
        (42, None, &forge_42, copy_w_without::<42>),
    ];
    for (idx, row, forge, without) in cases {
        let run = forged(Functionality::Copy, src, dst, n, |rows| {
            forge(rows);
            // The last row's B read follows its window.
            let last = rows.len() - 1;
            let r = &mut rows[last];
            if u64::from(r.last) + u64::from(r.shift) >= WORD_BYTES {
                r.flags |= copy_w::FLAG_READ_B;
            } else {
                r.flags &= !copy_w::FLAG_READ_B;
            }
        });
        let row = row.unwrap_or(run.rows.len() - 1);
        let bench = run.bench();
        refused_by_constraint(&bench, idx, &[row]).unwrap_or_else(|v| panic!("{idx}: {v}"));
        let v = verdict(&with_constraints(bench, without));
        assert!(v.accepted(), "control {idx}: {v:?}");
    }
}

/// The rows of a call laid out by hand: `rows` with their words read from
/// the run's memory before the call (none of them written by it) and, if
/// `commit`, their pairs taken as the public output.
fn read_words(run: &mut Run, mut rows: Vec<CopyWordRow>, commit: bool) {
    let mut memory = run.memory.clone();
    for k in 0..rows.len() {
        let a = if rows[k].reads_a() {
            memory.word(rows[k].a_key()).expect("held").0
        } else if k > 0 {
            rows[k - 1].b()
        } else {
            0
        };
        let b = if rows[k].reads_b() {
            memory.word(rows[k].b_key()).expect("held").0
        } else {
            0
        };
        let old = if rows[k].writes() {
            memory.word(rows[k].dst_key()).expect("held").0
        } else {
            0
        };
        rows[k].set_words(a, b, old);
    }
    if commit {
        run.commits = rows.iter().flat_map(|r| r.commit_pairs()).collect();
    }
    run.rows = rows;
}

/// ★ A memset whose destination is 16 bytes past its source, not 8 (the
/// executor refuses it; the forger skips the executor): its rows place a
/// 16-byte pattern. Refused by the gap rule alone (constraint 57, on each
/// row). Control: the rule removed, accepted.
#[test]
fn a_memset_with_another_gap_is_refused_by_the_gap_rule() {
    let (src, n) = (LOW + 0x500, 16);
    let dst = src + 16;
    let mut run = Run::new();
    let ts = run.next_ts();
    run.calls.push((ts, DMA_MEMSET_SYSCALL_NUMBER));
    for (reg, value) in [(11, src), (12, n), (10, dst)] {
        run.regs.push(RegAccess {
            reg,
            old: value,
            new: value,
            ts,
        });
    }
    // A copy's rows (shift 0, every word aligned) run as a memset's.
    let mut rows = copy_w::call_rows(Functionality::Copy, ts, src, dst, n).expect("rows");
    for r in rows.iter_mut() {
        r.flags |= copy_w::FLAG_SET;
    }
    read_words(&mut run, rows, false);
    let bench = run.bench();
    refused_by_constraint(&bench, 57, &[0, 1]).unwrap_or_else(|v| panic!("{v}"));
    let v = verdict(&with_constraints(bench, copy_w_without::<57>));
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ A commit that shifts: its windows and keys claim the offset 3 and the
/// shift 5 on an aligned source (`3 + 5 ≡ 0`, so every key stays aligned),
/// and it commits `A`'s bytes at the indices the offset implies — bytes that
/// are not the source's. Refused by the no-shift rule alone (constraint 56,
/// on each row). Control: the rule removed, accepted, and the public output
/// it proves is not the source's bytes.
#[test]
fn a_shifted_commit_is_refused_by_the_no_shift_rule() {
    let (src, n) = (ELF + 0x80, 8);
    let mut run = Run::new();
    let ts = run.next_ts();
    run.calls.push((ts, COMMIT_CALL));
    for (reg, old, new) in [(11, src, src), (12, n, n), (254, 0, n), (10, 1, n)] {
        run.regs.push(RegAccess { reg, old, new, ts });
    }
    // A copy's rows to 3 (offset 3, shift 5), run as a commit at index 0:
    // its `DST_KEY` the index less the offset, plus 8.
    let mut rows = copy_w::call_rows(Functionality::Copy, ts, src, 3, n).expect("rows");
    for r in rows.iter_mut() {
        r.flags |= copy_w::FLAG_WRITE;
        r.dst_key += 5;
    }
    assert_eq!((rows[0].start, rows[0].shift), (3, 5));
    read_words(&mut run, rows, true);
    let bench = run.bench();
    refused_by_constraint(&bench, 56, &[0, 1]).unwrap_or_else(|v| panic!("{v}"));
    let v = verdict(&with_constraints(bench, copy_w_without::<56>));
    assert!(v.accepted(), "control: {v:?}");
    let honest: Vec<u8> = (0..n).map(|i| run.memory.byte(src + i)).collect();
    let mut proved = run.commits.clone();
    proved.sort();
    let proved: Vec<u8> = proved.into_iter().map(|(_, b)| b).collect();
    assert_ne!(honest, proved, "the forged output names other bytes");
}

/// ★ A copy whose shift is not `(src − dst) mod 8`: the forger claims 0 on a
/// source at `8m + 3` and reads its first word at the unaligned key `8m + 3`.
/// The read consumes a token nothing minted, refused on the word bus alone.
/// Control: PAGE_W's `OFFSET` the forger's (a main column, set to
/// `m + 3/8`), which mints at `8m + 3`, accepted.
#[test]
fn a_copy_with_a_wrong_shift_reads_a_word_nothing_minted() {
    let m = (LOW + 0x800) / WORD_BYTES;
    let (src, dst, n) = (8 * m + 3, LOW + 0x400, 8);
    let mut run = forged(Functionality::Copy, src, dst, n, |_| {});
    assert_eq!((run.rows.len(), run.rows[0].shift), (1, 3));
    // The forger's row: shift 0, its A the eight bytes at src.
    let a = u64::from_le_bytes(std::array::from_fn(|i| run.memory.byte(src + i as u64)));
    let r = &mut run.rows[0];
    r.shift = 0;
    r.src_key = src;
    r.flags = (r.flags | copy_w::FLAG_READ_A) & !copy_w::FLAG_READ_B;
    r.a = a;
    r.b = 0;
    let bench = run.bench();
    let v = verdict(&bench);
    assert!(v.refused_only_on(BusId::MemWord), "{v:?}");
    assert!(
        v.left_on(BusId::MemWord)
            .iter()
            .all(|(tuple, _)| tuple.first().is_some_and(|lo| lo % 8 == 3)),
        "{v:?}"
    );
    // Control: page 0's row m at OFFSET m + 3/8 mints 8m + 3 (a zero word,
    // the zero page's) and closes it at the read.
    let mut control = bench;
    let name = page_name(0);
    let row = m as usize;
    let eighth = FE::from(8u64).inv().expect("8 is invertible");
    control.set(
        &name,
        row,
        pw::OFFSET,
        FE::from(m) + FE::from(3u64) * eighth,
    );
    control.set(
        &name,
        row,
        pw::TIMESTAMP,
        FE::from(run.rows[0].read_timestamp()),
    );
    assert_eq!(a, 0, "the zero page");
    let t = &control.piece(&name).trace;
    let column = (0..t.num_rows())
        .map(|r| *t.get_main(r, pw::OFFSET))
        .collect();
    control.piece_mut(&name).preprocessed[pw::OFFSET] = column;
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// The chain's shift and carry positions are the ones the forgeries above
/// take out.
#[test]
fn the_chain_tuple_carries_the_shift_and_the_word() {
    let interactions = copy_w::bus_interactions();
    let chain: Vec<&BusInteraction> = interactions
        .iter()
        .filter(|bi| bi.bus_id == u64::from(BusId::CopyWordNext))
        .collect();
    assert_eq!(chain.len(), 2);
    for bi in chain {
        assert_eq!(bi.values.len(), 17);
        let BusValue::Linear(terms) = &bi.values[CHAIN_SHIFT.start] else {
            panic!("the shift is a linear form");
        };
        let cols: Vec<usize> = terms
            .iter()
            .filter_map(|t| match *t {
                LinearTerm::Column { column, .. } => Some(column),
                _ => None,
            })
            .collect();
        assert_eq!(cols, cw::SHIFT[1..].to_vec());
        let carried = if bi.is_sender { cw::B } else { cw::A };
        for (k, v) in bi.values[CHAIN_CARRY].iter().enumerate() {
            assert_eq!(v, &column(carried[k]));
        }
    }
}

// ---- the census ----

/// COPY_W's constraints without the new word's identities (54, 55).
fn copy_w_without_new(row: &[FE]) -> Vec<FE> {
    let mut v = copy_w_constraints(row);
    v[54] = FE::zero();
    v[55] = FE::zero();
    v
}

/// ★ The census holds COPY_W to its call (its sender rules): the honest table
/// passes; it has a chain (bus 42, carrying the timestamp and both keys' high
/// words, each with its carry); and each mutant is refused: a message's
/// timestamp on another column, the chain not carrying the timestamp, the
/// chain not carrying a key's high word, a timestamp past the call's plus 2,
/// the new word fixed by no constraint, a read word's bytes unchecked,
/// `READ_A` set on an idle row; each carry without its bit or its tie (rule
/// c19), `B`'s carry at `−2^31`, the chain's destination carry left out of
/// its high word.
#[test]
fn the_census_holds_copy_w_to_its_call() {
    let name = "COPY_W[0]";
    let honest = copy_w::bus_interactions();
    census_table(name, &honest, 0).unwrap_or_else(|e| panic!("{e}"));
    let chain = chain_of(&honest).expect("COPY_W's chain");
    let message = |k: usize| {
        honest
            .iter()
            .enumerate()
            .filter(|(_, bi)| bi.bus_id == u64::from(BusId::MemWordMsg))
            .nth(k)
            .map(|(i, _)| i)
            .expect("three messages")
    };
    let receive = honest
        .iter()
        .position(|bi| bi.bus_id == u64::from(BusId::CopyWordNext) && !bi.is_sender)
        .expect("the chain's receive");
    assert!(
        [cw::TIMESTAMP, cw::SRC_KEY_HI, cw::DST_KEY_HI]
            .iter()
            .all(|c| chain.carried.contains(c)),
        "the chain carries the timestamp and the high words"
    );
    let refused =
        |what: &str, interactions: Vec<BusInteraction>, eval: Option<(Eval, usize)>, why: &str| {
            let r = match eval {
                Some(eval) => census_table_with(name, &interactions, 0, Some(eval)),
                None => census_table(name, &interactions, 0),
            };
            assert!(matches!(&r, Err(e) if e.contains(why)), "{what}: {r:?}");
        };
    let mut free_ts = honest.clone();
    free_ts[message(0)].values[3] = column(cw::OLD[0]);
    refused("a free message timestamp", free_ts, None, "no bit at ±1");
    let mut uncarried_ts = honest.clone();
    uncarried_ts[receive].values[0] = BusValue::constant(0);
    refused(
        "the chain without the timestamp",
        uncarried_ts,
        None,
        "its chain does not carry",
    );
    let mut uncarried_hi = honest.clone();
    uncarried_hi[receive].values[4] = BusValue::constant(0);
    refused(
        "the chain without the source key's high word",
        uncarried_hi,
        None,
        "no canonical source",
    );
    let mut late = honest.clone();
    late[message(2)].values[3] = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cw::TIMESTAMP,
        },
        LinearTerm::Constant(3),
    ]);
    refused(
        "a timestamp the call's plus 3",
        late,
        None,
        "its call's plus 0..=2",
    );
    refused(
        "the new word fixed by nothing",
        honest.clone(),
        Some((copy_w_without_new as Eval, cw::NUM_COLUMNS)),
        "neither checked nor fixed",
    );
    let unchecked: Vec<BusInteraction> = honest
        .iter()
        .filter(|bi| {
            !(bi.bus_id == u64::from(BusId::AreBytes)
                && bi.values.first() == Some(&column(cw::A[0])))
        })
        .cloned()
        .collect();
    assert_eq!(unchecked.len() + 1, honest.len());
    refused(
        "A's first pair unchecked",
        unchecked,
        None,
        "not each byte in a checked pair",
    );
    refused(
        "READ_A's rule removed",
        honest.clone(),
        Some((copy_w_without::<43> as Eval, cw::NUM_COLUMNS)),
        "on an idle row breaks nothing",
    );
    // Rule c19: each carry held by its bit and its tie.
    for (what, eval, why) in [
        (
            "SRC_CARRY's bit removed",
            copy_w_without::<59> as Eval,
            "on the last word breaks nothing",
        ),
        (
            "SRC_CARRY's tie removed",
            copy_w_without::<60>,
            "off the last word breaks nothing",
        ),
        (
            "DST_CARRY's bit removed",
            copy_w_without::<61>,
            "on the last word breaks nothing",
        ),
        (
            "DST_CARRY's tie removed",
            copy_w_without::<62>,
            "off the last word breaks nothing",
        ),
        (
            "OP_CARRY's bit removed",
            copy_w_without::<63>,
            "on the last word breaks nothing",
        ),
        (
            "OP_CARRY's tie removed",
            copy_w_without::<64>,
            "off the last word breaks nothing",
        ),
    ] {
        refused(what, honest.clone(), Some((eval, cw::NUM_COLUMNS)), why);
    }
    // B's key carrying at −2^31: no carry, a column scaled past 2^16.
    let mut half = honest.clone();
    let b = message(1);
    if let BusValue::Linear(terms) = &mut half[b].values[1] {
        for t in terms.iter_mut() {
            if let LinearTerm::Column {
                coefficient,
                column,
            } = t
                && *column == cw::SRC_CARRY
            {
                *coefficient = -(1 << 31);
            }
        }
    }
    refused(
        "B's carry at −2^31",
        half,
        None,
        "scales a column past 2^16",
    );
    // The chain's next destination high word without its carry: the key is
    // still carried (its low word's carry alone is no carry pair).
    let mut no_hi = honest.clone();
    let send = no_hi
        .iter()
        .position(|bi| bi.bus_id == u64::from(BusId::CopyWordNext) && bi.is_sender)
        .expect("the chain's send");
    no_hi[send].values[2] = column(cw::DST_KEY_HI);
    refused(
        "the chain's destination carry in its low word only",
        no_hi,
        None,
        "whose next does not carry it",
    );
    // x11's operand carry in its low word only (old and new value).
    let mut x11_lo_only = honest.clone();
    let x11 = x11_lo_only
        .iter()
        .position(|bi| {
            bi.bus_id == u64::from(BusId::Memw) && bi.values.get(9) == Some(&BusValue::constant(22))
        })
        .expect("x11's read");
    x11_lo_only[x11].values[1] = column(cw::SRC_KEY_HI);
    x11_lo_only[x11].values[12] = column(cw::SRC_KEY_HI);
    refused(
        "x11's carry in its low word only",
        x11_lo_only,
        None,
        "whose next does not carry it",
    );
}

// ---- every constraint load-bearing ----

/// The distinct rows of a run's COPY_W trace (padding rows once).
fn distinct_rows(run: &Run) -> Vec<Vec<FE>> {
    let t = copy_w::generate_copy_w_trace_as(&run.rows, TraceForm::Wide);
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for r in 0..t.num_rows() {
        let row: Vec<FE> = (0..cw::NUM_COLUMNS).map(|c| *t.get_main(r, c)).collect();
        let key: Vec<u64> = row.iter().map(|v| v.canonical_u64()).collect();
        if seen.insert(key) {
            out.push(row);
        }
    }
    out
}

/// `row` with the columns COPY_W defines by an identity recomputed from the
/// rest (`FIRST_WRITE`, `READ_A`, `READ_B`, `MU_W`, `COMMIT`, `NEW_LO`,
/// `NEW_HI`): an edit to the other columns then breaks only the constraints
/// that read them otherwise.
fn completed(mut row: Vec<FE>) -> Vec<FE> {
    let g = |row: &[FE], c: usize| row[c];
    let sum = |row: &[FE], cols: &[usize]| cols.iter().fold(FE::zero(), |a, &c| a + row[c]);
    let one = FE::one();
    let mask: Vec<FE> = (0..8)
        .map(|j| sum(&row, &cw::START[..=j]) - sum(&row, &cw::LAST[..j]))
        .collect();
    let mut need_a = FE::zero();
    let mut need_b = FE::zero();
    for i in 0..8 {
        for d in 0..8 {
            if i + d < 8 {
                need_a += g(&row, cw::START[i]) * g(&row, cw::SHIFT[d]);
            } else {
                need_b += g(&row, cw::LAST[i]) * g(&row, cw::SHIFT[d]);
            }
        }
    }
    let (first, end, mu, zero) = (
        g(&row, cw::FIRST),
        g(&row, cw::END),
        g(&row, cw::MU),
        g(&row, cw::ZERO),
    );
    let write = g(&row, cw::IS_WRITE);
    row[cw::FIRST_WRITE] = first * write;
    row[cw::READ_A] = first * need_a;
    row[cw::READ_B] = (mu - end) + end * need_b;
    row[cw::MU_W] = (mu - zero) * (one - write);
    for j in 0..8 {
        row[cw::COMMIT[j]] = write * mask[j];
    }
    for (half, bytes) in [(cw::NEW_LO, 0..4usize), (cw::NEW_HI, 4..8)] {
        let mut acc = FE::zero();
        for j in bytes.clone() {
            let placed = (0..8).filter(|s| j + s < 16).fold(FE::zero(), |a, s| {
                a + g(&row, cw::SHIFT[s]) * g(&row, cw::AB[j + s])
            });
            let byte = mask[j] * placed + (one - mask[j]) * g(&row, cw::OLD[j]);
            acc += FE::from(1u64 << (8 * (j - bytes.start))) * byte;
        }
        row[half] = acc;
    }
    row
}

/// ★ Every one of COPY_W's 65 constraints is one a row can break alone: for
/// each, from an honest row (every functionality, offset, shift and window
/// shape, rows whose keys carry across `2^32`, and padding), an edit breaks
/// that constraint and no other — one
/// column set to a small value, a bound or its own plus or minus one; two of
/// the flag and one-hot columns set to 0 or 1; or one one-hot column up one
/// and another down one (its sum kept) — the columns an identity defines
/// recomputed from the rest, or edited alone; or `MU` at 2 or `ZERO` at −1
/// with each one-hot given a second bit. So each is load-bearing at the row:
/// taken out, a row the table refused meets every other. Three are implied by
/// the rest (listed with why); every other must be found.
#[test]
fn every_copy_w_constraint_breaks_alone_on_some_row() {
    let mut run = Run::with_pages(line_pages());
    // Each carry 1 on some row: a source's B, a destination, x11's operand,
    // a memset across the line.
    run.copy(LINE - 2, LOW + 0x404, 4);
    run.copy(BELOW + 0x200, LINE - 16, 24);
    run.copy(LINE + 3, LOW + 0x406, 5);
    run.set(LINE - 13, 30);
    for offset in 0..8u64 {
        for shift in 0..8u64 {
            for &n in &[1u64, 3, 9, 17] {
                run.copy(
                    ELF + 8 * shift + (offset + shift) % 8,
                    LOW + 0x400 + offset,
                    n,
                );
            }
        }
        run.set(LOW + 0x700 + offset, 13);
        run.commit(ELF + 0x40 + offset, 11);
    }
    run.copy(LOW, LOW + 0x100, 0);
    run.commit(ELF, 0);
    let rows = distinct_rows(&run);
    let flags = [
        cw::FIRST,
        cw::END,
        cw::MU,
        cw::ZERO,
        cw::IS_SET,
        cw::IS_WRITE,
    ];
    let selectors: Vec<usize> = flags
        .into_iter()
        .chain([cw::FIRST_WRITE, cw::READ_A, cw::READ_B, cw::MU_W])
        .chain(cw::START)
        .chain(cw::LAST)
        .chain(cw::SHIFT)
        .chain(cw::COMMIT)
        .collect();
    let mut found = [false; copy_w::NUM_CONSTRAINTS];
    let mut note = |row: &[FE]| {
        let v = copy_w_constraints(row);
        let broken: Vec<usize> = (0..v.len()).filter(|&k| v[k] != FE::zero()).collect();
        if let [k] = broken.as_slice() {
            found[*k] = true;
        }
    };
    let small = |x: FE| {
        [
            FE::zero(),
            FE::one(),
            FE::from(2u64),
            FE::zero() - FE::one(),
            x + FE::one(),
            x - FE::one(),
            FE::from(1u64 << 32),
        ]
    };
    for row in &rows {
        for c in 0..cw::NUM_COLUMNS {
            for v in small(row[c]) {
                let mut edited = row.clone();
                edited[c] = v;
                note(&edited);
                note(&completed(edited));
            }
        }
        for (i, &a) in selectors.iter().enumerate() {
            for &b in &selectors[i + 1..] {
                for (va, vb) in [(0u64, 0u64), (0, 1), (1, 0), (1, 1)] {
                    let mut edited = row.clone();
                    edited[a] = FE::from(va);
                    edited[b] = FE::from(vb);
                    note(&edited);
                    note(&completed(edited));
                }
            }
        }
        for one_hot in [cw::START, cw::LAST, cw::SHIFT] {
            for &a in &one_hot {
                for &b in &one_hot {
                    if a != b {
                        let mut edited = row.clone();
                        edited[a] += FE::one();
                        edited[b] = edited[b] - FE::one();
                        note(&completed(edited));
                    }
                }
            }
        }
        // The flags three at a time, each 0, 1 or 2.
        for (i, &a) in flags.iter().enumerate() {
            for (j, &b) in flags.iter().enumerate().skip(i + 1) {
                for &c in &flags[j + 1..] {
                    for code in 0..27u64 {
                        let mut edited = row.clone();
                        edited[a] = FE::from(code % 3);
                        edited[b] = FE::from(code / 3 % 3);
                        edited[c] = FE::from(code / 9);
                        note(&completed(edited));
                    }
                }
            }
        }
        // `MU` at 2 or `ZERO` at −1, each one-hot given a second bit (its
        // sum `MU − ZERO` then 2).
        for (col, value) in [(cw::MU, FE::from(2u64)), (cw::ZERO, FE::zero() - FE::one())] {
            if row[cw::MU] != FE::one() || row[cw::ZERO] != FE::zero() {
                continue;
            }
            for s in 0..8 {
                for l in 0..8 {
                    for d in 0..8 {
                        let mut edited = row.clone();
                        edited[col] = value;
                        edited[cw::START[s]] += FE::one();
                        edited[cw::LAST[l]] += FE::one();
                        edited[cw::SHIFT[d]] += FE::one();
                        note(&completed(edited));
                    }
                }
            }
        }
    }
    // Implied by the rest, so no row breaks them alone (kept as written):
    // - 6, `FIRST` or `END` only on an active row: `FIRST = 1` makes
    //   `(MU − FIRST)·(1 − START_0)` (40) ask `MU = 1` or `START_0 = 1`, and
    //   `START_0 = 1` with the one-hot's sum `MU − ZERO` (20) and both bits
    //   asks `MU = 1`; `END` likewise by 41 and 29;
    // - 10 and 11, a zero-length row is first and last: `ZERO = 1` leaves the
    //   one-hots empty (20, 29, so `MU = 1`), and then a row that is not first
    //   breaks 40, one that is not last breaks 41.
    let missing: Vec<usize> = (0..found.len()).filter(|&k| !found[k]).collect();
    assert_eq!(
        missing,
        vec![6, 10, 11],
        "no lone break found for {missing:?}"
    );
}

/// The chain tuple's destination key, low and high word (both tuples).
const CHAIN_DST: std::ops::Range<usize> = 1..3;

/// A control's mint: the spare (never touched) private-input row `spare` of
/// the page at `base` at `OFFSET = offset` (the verifier's `OFFSET` column
/// replaced by the trace's, so only the token range is taken out) mints `word`
/// at `base + 8·offset` and takes it back, unchanged, at `ts`.
fn mint_at(bench: &mut Bench, base: u64, spare: usize, offset: FE, word: u64, ts: u64) {
    let page = page_name(base);
    bench.set(&page, spare, pw::OFFSET, offset);
    let quarters = [word, word >> 16, word >> 32, word >> 48].map(|q| q & 0xFFFF);
    for (k, q) in quarters.into_iter().enumerate() {
        bench.set(&page, spare, pw::INIT[k], FE::from(q));
    }
    bench.set(&page, spare, pw::FINI_LO, FE::from(word & 0xFFFF_FFFF));
    bench.set(&page, spare, pw::FINI_HI, FE::from(word >> 32));
    bench.set(&page, spare, pw::TIMESTAMP, FE::from(ts));
    let t = &bench.piece(&page).trace;
    let column = (0..t.num_rows())
        .map(|r| *t.get_main(r, pw::OFFSET))
        .collect();
    bench.piece_mut(&page).preprocessed[pw::OFFSET] = column;
}

/// ★ A one-row copy whose `B` is low word 0 of high word 1 (4 bytes from
/// `2^32 − 2`: `SRC_KEY = 2^32 − 8`, `B` read), its `SRC_CARRY` cleared: `B`'s
/// message names `(2^32, 0)`, which MEMW_W does not take (its access is at the
/// word's own key). Refused on the message bus alone (the bit and the tie hold
/// at 0; a one-row call has no chain). Control: MEMW_W's access moved to
/// `(2^32, 0)`, that key minted (the page below the line at `OFFSET = 2^15`,
/// one past its last word) and `2^32`'s word left untouched: accepted, so the
/// token range is the check.
#[test]
fn a_source_carry_cleared_at_the_last_word_names_a_key_nothing_minted() {
    let mut run = Run::with_pages(line_pages());
    let r = run.copy(LINE - 2, LOW + 0x404, 4);
    let row = r.start;
    let held = run.rows[row];
    assert!(held.first() && held.end() && held.reads_b() && held.b_key() == LINE);
    assert_eq!(held.carries(), [true, false, false]);
    let mut forged = run.bench();
    forged.set(copy_w::NAME, row, cw::SRC_CARRY, FE::zero());
    let v = verdict(&forged);
    assert!(v.refused_only_on(BusId::MemWordMsg), "{v:?}");
    assert!(
        v.left_on(BusId::MemWordMsg)
            .iter()
            .any(|(tuple, _)| tuple.get(1) == Some(&LINE) && tuple.get(2) == Some(&0)),
        "the key (2^32, 0): {v:?}"
    );
    let mut control = forged;
    let t = &control.piece("MEMW_W").trace;
    let access = (0..t.num_rows())
        .find(|&r| {
            *t.get_main(r, mw::ADDRESS_LO) == FE::zero()
                && *t.get_main(r, mw::ADDRESS_HI) == FE::one()
        })
        .expect("the access to 2^32's word");
    control.set("MEMW_W", access, mw::ADDRESS_LO, FE::from(LINE));
    control.set("MEMW_W", access, mw::ADDRESS_HI, FE::zero());
    let page = page_name(LINE);
    control.set(&page, 0, pw::TIMESTAMP, FE::zero());
    mint_at(
        &mut control,
        BELOW,
        100,
        FE::from(DEFAULT_PAGE_SIZE as u64 / 8),
        held.b(),
        held.read_timestamp(),
    );
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ A copy of 24 bytes to `2^32 − 16` whose middle row (the last word below
/// the line) clears its `DST_CARRY`: it hands the next row the destination key
/// `(2^32, 0)`, the next row holds `(0, 1)`. Refused by the chain alone (the
/// bit and the tie hold at 0). Control: the destination key taken out of the
/// chain's tuples, accepted.
#[test]
fn a_destination_carry_cleared_at_the_last_word_is_refused_by_the_chain() {
    let mut run = Run::with_pages(line_pages());
    let r = run.copy(BELOW + 0x200, LINE - 16, 24);
    assert_eq!(r.len(), 3);
    assert_eq!(run.rows[r.start + 1].carries(), [false, true, false]);
    let mut forged = run.bench();
    forged.set(copy_w::NAME, r.start + 1, cw::DST_CARRY, FE::zero());
    let v = verdict(&forged);
    assert!(v.refused_only_on(BusId::CopyWordNext), "{v:?}");
    let control = forged.with_interactions(copy_w::NAME, chain_without(CHAIN_DST));
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ The same copy's first row (two words below the line) claiming a
/// `DST_CARRY`: it hands the next row `(2^32 − 8 − 2^32, 1)`. Refused by its
/// tie (constraint 62) and by the chain; with the tie taken out, by the chain
/// alone; with the destination key also out of the chain, accepted.
#[test]
fn a_destination_carry_off_the_last_word_is_refused_by_its_tie_and_the_chain() {
    let mut run = Run::with_pages(line_pages());
    let r = run.copy(BELOW + 0x200, LINE - 16, 24);
    let mut forged = run.bench();
    forged.set(copy_w::NAME, r.start, cw::DST_CARRY, FE::one());
    let v = verdict(&forged);
    assert_eq!(
        v.broken,
        vec![(copy_w::NAME.to_string(), r.start, 62)],
        "{v:?}"
    );
    assert!(
        v.left.keys().all(|&b| b == u64::from(BusId::CopyWordNext))
            && !v.left_on(BusId::CopyWordNext).is_empty(),
        "{v:?}"
    );
    let untied = with_constraints(forged, copy_w_without::<62>);
    let v = verdict(&untied);
    assert!(v.refused_only_on(BusId::CopyWordNext), "{v:?}");
    let control = untied.with_interactions(copy_w::NAME, chain_without(CHAIN_DST));
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ A copy from `2^32 + 3` to an offset-6 destination (`SHIFT = 5`, so
/// `SRC_KEY = 2^32 − 8`, unread, and x11's operand `SRC_KEY + 11` carries into
/// high word 1) with its `OP_CARRY` cleared: x11 then reads `(2^32 + 3, 0)`
/// against the register's `(3, 1)`. Refused on the register bus alone (the bit
/// and the tie hold at 0). Control: the x11 read out of COPY_W and the
/// register file's answer to it left out, accepted.
#[test]
fn an_operand_carry_cleared_is_refused_by_the_register_read() {
    let mut run = Run::with_pages(line_pages());
    let r = run.copy(LINE + 3, LOW + 0x406, 5);
    let held = run.rows[r.start];
    assert!(held.first() && !held.reads_a() && held.a_key() == LINE - 8);
    assert_eq!(held.carries(), [true, false, true]);
    let mut forged = run.bench();
    forged.set(copy_w::NAME, r.start, cw::OP_CARRY, FE::zero());
    let v = verdict(&forged);
    assert!(v.refused_only_on(BusId::Memw), "{v:?}");
    let x11 = |bi: &BusInteraction| {
        bi.bus_id == u64::from(BusId::Memw) && bi.values.get(9) == Some(&BusValue::constant(22))
    };
    let without: Vec<BusInteraction> = copy_w::bus_interactions()
        .into_iter()
        .filter(|bi| !x11(bi))
        .collect();
    assert_eq!(without.len() + 1, copy_w::bus_interactions().len());
    let mut control = forged.with_interactions(copy_w::NAME, without);
    let regs_row = run
        .regs
        .iter()
        .position(|a| a.reg == 11 && a.old == LINE + 3)
        .expect("x11's access");
    control.set("REGS", regs_row, regs::MU, FE::zero());
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ A copy whose count is `2^32` or more, which the walk refuses: COPY_W
/// reads the count register's high word as the constant 0, so a call whose
/// x12 holds `2^32 + 8` and whose rows copy 8 bytes leaves the count read on
/// the register bus alone. (The bound rests on that constant, an in-circuit
/// value, not on a constraint.) Control: the read's high word taken from a
/// column (`FIRST`, 1 on the call's row) instead of the constant, accepted.
#[test]
fn a_copy_of_2_to_the_32_bytes_or_more_is_refused_by_the_count_read() {
    let mut run = Run::new();
    let (src, dst) = (LOW + 0x200, LOW + 0x400);
    let count = (1u64 << 32) + 8;
    assert!(copy_w::call_rows(Functionality::Copy, 8, src, dst, count).is_err());
    let ts = run.next_ts();
    run.calls.push((ts, DMA_MEMCPY_SYSCALL_NUMBER));
    for (reg, value) in [(11, src), (12, count), (10, dst)] {
        run.regs.push(RegAccess {
            reg,
            old: value,
            new: value,
            ts,
        });
    }
    let mut rows = copy_w::call_rows(Functionality::Copy, ts, src, dst, 8).expect("8 bytes");
    let memw = copy_w::run_call(&mut rows, src, dst, 8, &mut run.memory)
        .unwrap_or_else(|why| panic!("the forged call runs: {why}"))
        .expect("every word is held");
    run.rows.extend(rows);
    run.memw.extend(memw);
    let bench = run.bench();
    let v = verdict(&bench);
    assert!(v.refused_only_on(BusId::Memw), "{v:?}");
    assert!(
        v.left_on(BusId::Memw)
            .iter()
            .any(|(tuple, _)| tuple.contains(&1)),
        "the count's high word left over: {v:?}"
    );
    // The x12 read's old and new high words (tuple positions 1 and 12).
    let mut control = copy_w::bus_interactions();
    let x12 = control
        .iter_mut()
        .find(|bi| {
            bi.bus_id == u64::from(BusId::Memw) && bi.values.get(9) == Some(&BusValue::constant(24))
        })
        .expect("the count register's read");
    assert_eq!(
        (&x12.values[1], &x12.values[12]),
        (&BusValue::constant(0), &BusValue::constant(0)),
        "the count's high words are the constant 0"
    );
    x12.values[1] = column(cw::FIRST);
    x12.values[12] = column(cw::FIRST);
    let v = verdict(&bench.with_interactions(copy_w::NAME, control));
    assert!(v.accepted(), "control: {v:?}");
}

// ---- the chain's other values and the count's limbs ----

/// The chain tuple's source key low word, count, memset flag, commit flag.
const CHAIN_SRC_LO: std::ops::Range<usize> = 3..4;
const CHAIN_COUNT: std::ops::Range<usize> = 5..6;
const CHAIN_SET: std::ops::Range<usize> = 6..7;
const CHAIN_WRITE: std::ops::Range<usize> = 7..8;

/// `bench` refused by the chain alone, and accepted with `positions` out of
/// the chain's tuples.
fn refused_by_the_chain_at(bench: Bench, positions: std::ops::Range<usize>) {
    let v = verdict(&bench);
    assert!(
        v.refused_only_on(BusId::CopyWordNext),
        "{positions:?}: {v:?}"
    );
    let v = verdict(&bench.with_interactions(copy_w::NAME, chain_without(positions.clone())));
    assert!(v.accepted(), "control, {positions:?} out: {v:?}");
}

/// ★ The source key rides the chain: a 40-byte copy whose rows after the
/// first read their source words 0x108 further on, so the bytes they place
/// are not the source's. Refused by the chain alone; accepted with the
/// source key's low word out of it.
#[test]
fn later_source_words_from_elsewhere_are_refused_by_the_chain() {
    let mut run = Run::new();
    let r = run.call_with(Functionality::Copy, ELF + 3, LOW + 0x400, 40, |rows| {
        assert!(rows.len() >= 3);
        for row in rows.iter_mut().skip(1) {
            row.src_key += 0x108;
        }
    });
    let row = run.rows[r.start + 1];
    let honest = run
        .memory
        .clone()
        .word(row.b_key() - 0x108)
        .expect("held")
        .0;
    assert_ne!(row.b(), honest, "the placed bytes are not the source's");
    refused_by_the_chain_at(run.bench(), CHAIN_SRC_LO);
}

/// ★ The count rides the chain: a 16-byte copy whose first row leaves 8 (as
/// x12's 16 asks) and whose later rows run as a 24-byte call's, writing 24
/// bytes. Refused by the chain alone; accepted with the count out of it.
#[test]
fn a_call_that_copies_past_its_count_is_refused_by_the_chain() {
    let (src, dst) = (ELF + 0x40, LOW + 0x400);
    let mut run = Run::new();
    let ts = run.next_ts();
    run.guest_call(Functionality::Copy, src, dst, 16, ts);
    let mut rows = copy_w::call_rows(Functionality::Copy, ts, src, dst, 24).expect("rows");
    assert_eq!(rows.len(), 3);
    rows[0].count_decr = 8;
    let memw = copy_w::run_call(&mut rows, src, dst, 24, &mut run.memory)
        .unwrap_or_else(|why| panic!("the forged call runs: {why}"))
        .expect("every word is held");
    run.rows.extend(rows);
    run.memw.extend(memw);
    refused_by_the_chain_at(run.bench(), CHAIN_COUNT);
}

/// ★ The memset flag rides the chain: an aligned 24-byte memcpy to `src + 8`
/// whose rows after the first run as a memset's (each writes its carried
/// word at `T`, the next reads it back at `T + 1`), so the last word gets the
/// second source word, not the third that memmove leaves. Refused by the
/// chain alone; accepted with the flag out of it.
#[test]
fn a_copy_that_turns_into_a_memset_is_refused_by_the_chain() {
    let src = ELF + 0x100;
    let dst = src + 8;
    let mut run = Run::new();
    let ts = run.next_ts();
    run.guest_call(Functionality::Copy, src, dst, 24, ts);
    let mut rows = copy_w::call_rows(Functionality::Copy, ts, src, dst, 24).expect("rows");
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|r| r.shift == 0));
    let s: Vec<u64> = (0..4u64)
        .map(|k| run.memory.word(src + 8 * k).expect("held").0)
        .collect();
    assert_ne!(s[1], s[2], "distinct source words");
    rows[0].set_words(s[0], s[1], s[1]);
    rows[1].flags |= copy_w::FLAG_SET;
    rows[1].set_words(s[1], 0, s[2]);
    let back = rows[1].new_word();
    assert_eq!(back, s[1]);
    rows[1].set_words(s[1], back, s[2]);
    rows[2].flags |= copy_w::FLAG_SET;
    rows[2].set_words(back, 0, s[3]);
    assert_eq!(
        rows[2].new_word(),
        s[1],
        "the last word: the second source word"
    );
    run.rows = rows;
    refused_by_the_chain_at(run.bench(), CHAIN_SET);
}

/// ★ The commit flag rides the chain: a commit `C` of 24 bytes from `s` at
/// index 0 and a later memcpy `M` of the same 24 bytes to address 8 (whose
/// words are the keys `C`'s commit index names) trade their rows after the
/// first: `C`'s write `s`'s words to 16 and 24 at `C`'s time, `M`'s send `C`'s
/// commit pairs. A commit `D` of the word at 16, between the two, then
/// commits `s`'s bytes where the program held zeros: another public output.
/// Refused by the chain alone; accepted with the commit flag out of it.
#[test]
fn a_commit_and_a_copy_that_trade_their_rows_are_refused_by_the_chain() {
    let s = ELF + 0x40;
    let mut run = Run::new();
    let words: Vec<u64> = (0..3u64)
        .map(|k| run.memory.word(s + 8 * k).expect("held").0)
        .collect();
    assert!(words.iter().all(|&w| w != 0));
    let (t1, t2, t3) = (run.next_ts(), run.next_ts(), run.next_ts());
    run.guest_call(Functionality::Write, s, 0, 24, t1);
    run.guest_call(Functionality::Write, 16, 24, 8, t2);
    run.guest_call(Functionality::Copy, s, 8, 24, t3);
    let honest_output = run.commits.clone();
    let mut c = copy_w::call_rows(Functionality::Write, t1, s, 0, 24).expect("C");
    let mut d = copy_w::call_rows(Functionality::Write, t2, 16, 24, 8).expect("D");
    let mut m = copy_w::call_rows(Functionality::Copy, t3, s, 8, 24).expect("M");
    assert_eq!((c.len(), d.len(), m.len()), (3, 1, 3));
    for k in 1..3 {
        assert_eq!(
            c[k].dst_key(),
            m[k].dst_key(),
            "the commit's keys are M's words"
        );
        c[k].flags &= !copy_w::FLAG_WRITE;
        m[k].flags |= copy_w::FLAG_WRITE;
    }
    // At t1, C reads s's words and (its rows 1, 2 now copy rows) writes
    // words 1 and 2 to 16 and 24, which held zeros.
    c[0].set_words(words[0], words[1], 0);
    c[1].set_words(words[1], words[2], 0);
    c[2].set_words(words[2], 0, 0);
    assert_eq!((c[1].new_word(), c[2].new_word()), (words[1], words[2]));
    // At t2, D reads 16: word 1, where the program holds zeros until t3.
    d[0].set_words(words[1], 0, 0);
    // At t3, M's first row writes word 0 to 8; its rows 1, 2 (now commit
    // rows) write nothing and send C's pairs.
    m[0].set_words(words[0], words[1], 0);
    m[1].set_words(words[1], words[2], 0);
    m[2].set_words(words[2], 0, 0);
    let rows: Vec<CopyWordRow> = c.into_iter().chain(d).chain(m).collect();
    run.commits = rows.iter().flat_map(|r| r.commit_pairs()).collect();
    run.commits.sort_unstable();
    let mut honest = honest_output;
    honest.sort_unstable();
    assert_eq!(run.commits.len(), honest.len());
    assert_ne!(run.commits, honest, "another public output");
    run.rows = rows;
    refused_by_the_chain_at(run.bench(), CHAIN_WRITE);
}

/// ★ The count's limbs are what keep it from running below zero: an 8-byte
/// memcpy laid out as a 16-byte call's two rows, the first leaving 0 (and not
/// the last), the second starting from 0 and leaving −8, held as limbs `(x,
/// −x)` with `x·(1 − 2^16) = −8`, which the last row's rule (its limbs' sum
/// zero) takes. Refused by the two `IS_HALF` on `COUNT_DECR` alone; accepted
/// with them out: 16 bytes written for an 8-byte call.
#[test]
fn a_count_that_runs_below_zero_is_refused_by_its_limb_checks() {
    let (src, dst) = (ELF + 0x40, LOW + 0x400);
    let mut run = Run::new();
    let ts = run.next_ts();
    run.guest_call(Functionality::Copy, src, dst, 8, ts);
    let mut rows = copy_w::call_rows(Functionality::Copy, ts, src, dst, 16).expect("rows");
    assert_eq!(rows.len(), 2);
    rows[0].count_decr = 0;
    let memw = copy_w::run_call(&mut rows, src, dst, 16, &mut run.memory)
        .unwrap_or_else(|why| panic!("the forged call runs: {why}"))
        .expect("every word is held");
    run.rows.extend(rows);
    run.memw.extend(memw);
    let mut bench = run.bench();
    let x = FE::from(8u64) * FE::from((1u64 << 16) - 1).inv().expect("invertible");
    bench.set(copy_w::NAME, 1, cw::COUNT_DECR[0], x);
    bench.set(copy_w::NAME, 1, cw::COUNT_DECR[1], FE::zero() - x);
    let v = verdict(&bench);
    let mut limbs = vec![
        (u64::from(BusId::IsHalfword), vec![x.canonical_u64()]),
        (
            u64::from(BusId::IsHalfword),
            vec![(FE::zero() - x).canonical_u64()],
        ),
    ];
    limbs.sort();
    assert!(
        v.left.is_empty()
            && v.broken.is_empty()
            && v.preprocessed.is_empty()
            && v.unanswered == limbs,
        "{v:?}"
    );
    let control = without_halfwords(&cw::COUNT_DECR);
    let v = verdict(&bench.with_interactions(copy_w::NAME, control));
    assert!(v.accepted(), "control: {v:?}");
}

/// COPY_W's interactions with the `IS_HALF` lookups on `limbs` at
/// multiplicity zero: a control mutant.
fn without_halfwords(limbs: &[usize]) -> Vec<BusInteraction> {
    let mut out = copy_w::bus_interactions();
    let mut off = 0;
    for bi in &mut out {
        if bi.bus_id == u64::from(BusId::IsHalfword)
            && limbs.iter().any(|&c| bi.values == vec![column(c)])
        {
            bi.multiplicity = Multiplicity::Linear(vec![LinearTerm::Constant(0)]);
            off += 1;
        }
    }
    assert_eq!(off, limbs.len(), "one IS_HALF a limb");
    out
}

/// ★ The keys' limbs are a second check: COPY_W reads a key's two limbs only
/// as their sum, and every key a row reads or writes is a PAGE_W key, which
/// holds a low word below 2^32. A 24-byte copy to `2^32 − 16` whose middle
/// row clears its `DST_CARRY` hands the last row `(2^32, 0)`, which the last
/// row holds as limbs `(0, 2^16)`: refused by its `IS_HALF` and by the word
/// bus (its write names a key no page has). With the destination limbs'
/// lookups out it is refused by the word bus alone; with that key also
/// minted (a spare row of the page below the line at `OFFSET = 2^15`, the
/// write moved to it), accepted: the token range is the check.
#[test]
fn a_destination_key_past_its_low_range_is_refused_by_the_word_bus() {
    let mut run = Run::with_pages(line_pages());
    let r = run.copy(BELOW + 0x200, LINE - 16, 24);
    let last = r.start + 2;
    assert_eq!(run.rows[r.start + 1].carries(), [false, true, false]);
    assert_eq!(run.rows[last].dst_key(), LINE);
    let mut forged = run.bench();
    forged.set(copy_w::NAME, r.start + 1, cw::DST_CARRY, FE::zero());
    forged.set(copy_w::NAME, last, cw::DST_KEY_LO[0], FE::zero());
    forged.set(copy_w::NAME, last, cw::DST_KEY_LO[1], FE::from(1u64 << 16));
    forged.set(copy_w::NAME, last, cw::DST_KEY_HI, FE::zero());
    let v = verdict(&forged);
    assert_eq!(
        v.unanswered,
        vec![(u64::from(BusId::IsHalfword), vec![1 << 16])],
        "{v:?}"
    );
    assert!(
        v.left.keys().all(|&b| b == u64::from(BusId::MemWordMsg)) && v.broken.is_empty(),
        "{v:?}"
    );
    let unchecked = forged.with_interactions(copy_w::NAME, without_halfwords(&cw::DST_KEY_LO));
    let v = verdict(&unchecked);
    assert!(v.refused_only_on(BusId::MemWordMsg), "{v:?}");
    let mut control = unchecked;
    let held = run.rows[last];
    let t = &control.piece("MEMW_W").trace;
    let access = (0..t.num_rows())
        .find(|&m| {
            *t.get_main(m, mw::ADDRESS_LO) == FE::zero()
                && *t.get_main(m, mw::ADDRESS_HI) == FE::one()
        })
        .expect("the write to 2^32's word");
    control.set("MEMW_W", access, mw::ADDRESS_LO, FE::from(LINE));
    control.set("MEMW_W", access, mw::ADDRESS_HI, FE::zero());
    let page = page_name(LINE);
    control.set(&page, 0, pw::TIMESTAMP, FE::zero());
    let init = run.memory.init_word(LINE).expect("held");
    control.set(&page, 0, pw::FINI_LO, FE::from(init & 0xFFFF_FFFF));
    control.set(&page, 0, pw::FINI_HI, FE::from(init >> 32));
    mint_at(
        &mut control,
        BELOW,
        100,
        FE::from(DEFAULT_PAGE_SIZE as u64 / 8),
        held.new_word(),
        held.write_timestamp(),
    );
    // The minted word starts as the write's old word.
    let quarters = [
        held.old(),
        held.old() >> 16,
        held.old() >> 32,
        held.old() >> 48,
    ];
    for (k, q) in quarters.into_iter().enumerate() {
        control.set(&page_name(BELOW), 100, pw::INIT[k], FE::from(q & 0xFFFF));
    }
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ The same for the source: an 11-byte copy from `2^32 − 3` (shift 5: its
/// first row reads `A` at `2^32 − 8` and `B` at `(0, 1)`, its last row places
/// the carried word's last three bytes and reads nothing) with the first row's `SRC_CARRY` cleared: `B` named `(2^32, 0)`
/// and the chain hands the last row that key, which it holds as limbs `(0,
/// 2^16)`. Refused by its `IS_HALF` and by the word bus; with the source
/// limbs' lookups out, by the word bus alone; with `(2^32, 0)` minted and the
/// read moved to it, accepted.
#[test]
fn a_source_key_past_its_low_range_is_refused_by_the_word_bus() {
    let mut run = Run::with_pages(line_pages());
    let r = run.copy(LINE - 3, LOW + 0x400, 11);
    assert_eq!(r.len(), 2);
    let (first, last) = (run.rows[r.start], run.rows[r.start + 1]);
    assert!(first.reads_a() && first.reads_b() && first.b_key() == LINE);
    assert!(
        !last.reads_a() && !last.reads_b(),
        "the last row reads nothing"
    );
    assert_eq!(first.carries(), [true, false, false]);
    let mut forged = run.bench();
    forged.set(copy_w::NAME, r.start, cw::SRC_CARRY, FE::zero());
    forged.set(copy_w::NAME, r.start + 1, cw::SRC_KEY_LO[0], FE::zero());
    forged.set(
        copy_w::NAME,
        r.start + 1,
        cw::SRC_KEY_LO[1],
        FE::from(1u64 << 16),
    );
    forged.set(copy_w::NAME, r.start + 1, cw::SRC_KEY_HI, FE::zero());
    let v = verdict(&forged);
    assert_eq!(
        v.unanswered,
        vec![(u64::from(BusId::IsHalfword), vec![1 << 16])],
        "{v:?}"
    );
    assert!(
        v.left.keys().all(|&b| b == u64::from(BusId::MemWordMsg)) && v.broken.is_empty(),
        "{v:?}"
    );
    let unchecked = forged.with_interactions(copy_w::NAME, without_halfwords(&cw::SRC_KEY_LO));
    let v = verdict(&unchecked);
    assert!(v.refused_only_on(BusId::MemWordMsg), "{v:?}");
    let mut control = unchecked;
    let t = &control.piece("MEMW_W").trace;
    let access = (0..t.num_rows())
        .find(|&m| {
            *t.get_main(m, mw::ADDRESS_LO) == FE::zero()
                && *t.get_main(m, mw::ADDRESS_HI) == FE::one()
        })
        .expect("the read of 2^32's word");
    control.set("MEMW_W", access, mw::ADDRESS_LO, FE::from(LINE));
    control.set("MEMW_W", access, mw::ADDRESS_HI, FE::zero());
    let page = page_name(LINE);
    control.set(&page, 0, pw::TIMESTAMP, FE::zero());
    mint_at(
        &mut control,
        BELOW,
        100,
        FE::from(DEFAULT_PAGE_SIZE as u64 / 8),
        first.b(),
        first.read_timestamp(),
    );
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// The chain tuple's timestamp, and the keys' high words.
const CHAIN_T: std::ops::Range<usize> = 0..1;
const CHAIN_DST_HI: std::ops::Range<usize> = 2..3;
const CHAIN_SRC_HI: std::ops::Range<usize> = 4..5;

/// The columns a chain value reads.
fn value_columns(value: &BusValue) -> Vec<usize> {
    match value {
        BusValue::Linear(terms) => terms
            .iter()
            .filter_map(|t| match *t {
                LinearTerm::Column { column, .. } => Some(column),
                _ => None,
            })
            .collect(),
        other => {
            let one = column(cw::TIMESTAMP);
            assert_eq!(std::mem::discriminant(other), std::mem::discriminant(&one));
            (0..cw::NUM_COLUMNS)
                .filter(|&c| *other == column(c))
                .collect()
        }
    }
}

/// The chain's positions the forgeries take out are the values they name:
/// the timestamp, the keys, the count, the two flags.
#[test]
fn the_chain_tuple_positions_are_the_forgeries_ones() {
    for bi in copy_w::bus_interactions()
        .iter()
        .filter(|bi| bi.bus_id == u64::from(BusId::CopyWordNext))
    {
        let at = |p: std::ops::Range<usize>| value_columns(&bi.values[p.start]);
        assert_eq!(at(CHAIN_T), vec![cw::TIMESTAMP]);
        assert!(at(CHAIN_DST).starts_with(&cw::DST_KEY_LO));
        assert_eq!(at(CHAIN_DST_HI)[0], cw::DST_KEY_HI);
        assert!(at(CHAIN_SRC_LO).starts_with(&cw::SRC_KEY_LO));
        assert_eq!(at(CHAIN_SRC_HI)[0], cw::SRC_KEY_HI);
        assert!(at(CHAIN_COUNT).starts_with(&cw::COUNT_DECR));
        assert_eq!(at(CHAIN_SET), vec![cw::IS_SET]);
        assert_eq!(at(CHAIN_WRITE), vec![cw::IS_WRITE]);
    }
}

/// ★ The timestamp rides the chain: a 40-byte copy whose rows after the
/// first sit 8 later (reads at `T + 8`, writes at `T + 9`), after whatever
/// ran between. Refused by the chain alone; accepted with it out.
#[test]
fn rows_after_the_first_at_another_time_are_refused_by_the_chain() {
    let run = forged(Functionality::Copy, ELF + 3, LOW + 0x400, 40, |rows| {
        assert!(rows.len() >= 3);
        for r in rows.iter_mut().skip(1) {
            r.timestamp += 8;
        }
    });
    refused_by_the_chain_at(run.bench(), CHAIN_T);
}

/// ★ The keys' high words ride the chain: a 40-byte copy whose rows after
/// the first write, or read, 4 GiB away. Refused by the chain alone;
/// accepted with that high word out of it.
#[test]
fn later_words_under_another_high_word_are_refused_by_the_chain() {
    for (destination, positions) in [(true, CHAIN_DST_HI), (false, CHAIN_SRC_HI)] {
        let mut run = Run::new();
        run.memory.pages.push(PageConfig::zero_init(1 << 32));
        run.memory
            .pages
            .push(PageConfig::zero_init((1 << 32) + ELF_BASE));
        run.call_with(Functionality::Copy, ELF + 3, LOW + 0x400, 40, |rows| {
            assert!(rows.len() >= 3);
            for r in rows.iter_mut().skip(1) {
                if destination {
                    r.dst_key += 1 << 32;
                } else {
                    r.src_key += 1 << 32;
                }
            }
        });
        refused_by_the_chain_at(run.bench(), positions);
    }
}

/// ★ A copy to `8m + 3` whose rows claim offset 0 and write the unaligned
/// words at `8m + 3` and `8m + 11`: no page mints a token there, so refused
/// on the word bus alone (every key left over is 3 mod 8). Control: PAGE_W's
/// `OFFSET` the forger's (`m + k + 3/8`), which mints the two keys, accepted.
#[test]
fn a_copy_with_a_wrong_destination_offset_writes_words_nothing_minted() {
    let m = (LOW + 0x400) / WORD_BYTES;
    let (src, dst, n) = (ELF + 3, 8 * m + 3, 10u64);
    let mut run = Run::new();
    let ts = run.next_ts();
    run.guest_call(Functionality::Copy, src, dst, n, ts);
    // The rows of the same call to the aligned word (offset 0, shift 3),
    // their destination keys the forger's.
    let mut rows = copy_w::call_rows(Functionality::Copy, ts, src, 8 * m, n).expect("rows");
    assert_eq!((rows.len(), rows[0].start, rows[0].shift), (2, 0, 3));
    for (k, r) in rows.iter_mut().enumerate() {
        r.dst_key = dst + 8 * k as u64;
    }
    read_words(&mut run, rows, false);
    let bench = run.bench();
    let v = verdict(&bench);
    assert!(v.refused_only_on(BusId::MemWord), "{v:?}");
    assert!(
        v.left_on(BusId::MemWord)
            .iter()
            .all(|(tuple, _)| tuple.first().is_some_and(|lo| lo % 8 == 3)),
        "{v:?}"
    );
    let mut control = bench;
    let name = page_name(0);
    let eighth = FE::from(8u64).inv().expect("8 is invertible");
    for (k, r) in run.rows.iter().enumerate() {
        let row = m as usize + k;
        control.set(
            &name,
            row,
            pw::OFFSET,
            FE::from(m + k as u64) + FE::from(3u64) * eighth,
        );
        control.set(&name, row, pw::TIMESTAMP, FE::from(r.write_timestamp()));
        let new = r.new_word();
        control.set(&name, row, pw::FINI_LO, FE::from(new & 0xFFFF_FFFF));
        control.set(&name, row, pw::FINI_HI, FE::from(new >> 32));
    }
    let t = &control.piece(&name).trace;
    let column = (0..t.num_rows())
        .map(|r| *t.get_main(r, pw::OFFSET))
        .collect();
    control.piece_mut(&name).preprocessed[pw::OFFSET] = column;
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

// ---- the byte pairs ----

/// `run`'s bench with COPY_W row `row`'s new word the halves `(lo, hi)`
/// (field values, so a half past 2^32 too), its MEMW_W write and its final
/// following them (the forger's write).
fn with_new_halves(run: &Run, row: usize, (lo, hi): (FE, FE)) -> Bench {
    let r = run.rows[row];
    let mut bench = with_new_word(run, row, r.new_word());
    let (key, ts) = (r.dst_key(), r.write_timestamp());
    bench.set(copy_w::NAME, row, cw::NEW_LO, lo);
    bench.set(copy_w::NAME, row, cw::NEW_HI, hi);
    let n = bench.piece("MEMW_W").trace.num_rows();
    let m = (0..n)
        .find(|&m| {
            bench.get("MEMW_W", m, mw::ADDRESS_LO) == FE::from(key & 0xFFFF_FFFF)
                && bench.get("MEMW_W", m, mw::TIMESTAMP) == FE::from(ts)
        })
        .expect("the row's write");
    bench.set("MEMW_W", m, mw::NEW_LO, lo);
    bench.set("MEMW_W", m, mw::NEW_HI, hi);
    let page = run.memory.page_of(key).expect("a page").page_base;
    let p = ((key - page) / WORD_BYTES) as usize;
    bench.set(&page_name(page), p, pw::FINI_LO, lo);
    bench.set(&page_name(page), p, pw::FINI_HI, hi);
    bench
}

/// A split pair's case: which word, its columns, src, dst, count, whether
/// the first row (else the last).
type PairCase = (&'static str, [usize; 8], u64, u64, u64, bool);

/// ★ Every byte pair is load-bearing where a window splits it: the pair's
/// low byte takes 256 more and its high byte one less, so the word read or
/// written is memory's, and the row's new word is another:
/// - `OLD`, the first row's kept bytes below an odd `START`: the kept byte
///   256 more, so the written byte above it comes out one more;
/// - `A`, the first row's unused bytes below an odd `START + SHIFT`: the
///   placed byte one less;
/// - `B`, the last row's placed byte at an even `LAST + SHIFT − 8` (unused
///   above it): 256 more, so the kept byte past the window comes out one
///   more, or, when the placed byte tops its half, the half passes 2^32 (a
///   word that is not canonical).
///
/// Each is refused by that pair's `ARE_BYTES` alone, and accepted with that
/// pair out (the new word, its MEMW_W write and its final the forger's).
#[test]
fn every_byte_pair_refuses_a_word_split_across_a_window_edge() {
    let byte = |word: u64, j: usize| (word >> (8 * j)) & 0xFF;
    let minus_one = |x: u64| (FE::from(x) - FE::one()).canonical_u64();
    // (which word, its columns, src, dst, count, first row or last).
    let mut cases: Vec<PairCase> = Vec::new();
    for odd in [1u64, 3, 5, 7] {
        // OLD: dst offset `odd`, shift 0; A: dst aligned, shift `odd`.
        cases.push((
            "OLD",
            cw::OLD,
            ELF + 0x40 + odd,
            LOW + 0x400 + odd,
            12,
            true,
        ));
        cases.push(("A", cw::A, ELF + 0x40 + odd, LOW + 0x400, 12, true));
    }
    for even in [0u64, 2, 4, 6] {
        // B: dst aligned, shift 7, the last row's window 0..=even + 1.
        cases.push(("B", cw::B, ELF + 0x47, LOW + 0x400, 10 + even, false));
    }
    let mut seen = Vec::new();
    for (what, cols, src, dst, n, first) in cases {
        let mut run = Run::new();
        let r = run.copy(src, dst, n);
        let row = if first { r.start } else { r.end - 1 };
        let held = run.rows[row];
        // The pair's first byte, and the destination byte whose half takes
        // the change with what it takes: the kept byte 256 more (OLD), the
        // placed byte one less (A) or 256 more (B), each inside its half.
        let (word, lo, at, change) = match what {
            "OLD" => {
                let kept = usize::from(held.start) - 1;
                (held.old(), kept, kept, FE::from(256u64))
            }
            "A" => {
                let used = usize::from(held.start + held.shift);
                assert!(
                    byte(held.a(), used) >= 1,
                    "A {used}: a byte to take one from"
                );
                (
                    held.a(),
                    used - 1,
                    usize::from(held.start),
                    FE::zero() - FE::one(),
                )
            }
            _ => {
                let used = usize::from(held.last + held.shift) - 8;
                (held.b(), used, usize::from(held.last), FE::from(256u64))
            }
        };
        assert_eq!(lo % 2, 0, "{what}: the window splits a pair");
        seen.push((what, lo));
        let (a, b) = (byte(word, lo) + 256, minus_one(byte(word, lo + 1)));
        let honest = held.new_word();
        let mut halves = [FE::from(honest & 0xFFFF_FFFF), FE::from(honest >> 32)];
        halves[at / 4] += change * FE::from(1u64 << (8 * (at % 4)));
        let mut bench = with_new_halves(&run, row, (halves[0], halves[1]));
        bench.set(copy_w::NAME, row, cols[lo], FE::from(a));
        bench.set(copy_w::NAME, row, cols[lo + 1], FE::from(b));
        let v = verdict(&bench);
        assert!(
            v.left.is_empty()
                && v.broken.is_empty()
                && v.preprocessed.is_empty()
                && v.unanswered == vec![(u64::from(BusId::AreBytes), vec![a, b])],
            "{what} pair at {lo}: {v:?}"
        );
        let control: Vec<BusInteraction> = copy_w::bus_interactions()
            .into_iter()
            .filter(|bi| {
                !(bi.bus_id == u64::from(BusId::AreBytes)
                    && bi.values.first() == Some(&column(cols[lo])))
            })
            .collect();
        assert_eq!(control.len() + 1, copy_w::bus_interactions().len());
        let v = verdict(&bench.with_interactions(copy_w::NAME, control));
        assert!(v.accepted(), "control, {what} pair at {lo}: {v:?}");
    }
    seen.sort_unstable();
    let all: Vec<(&str, usize)> = ["A", "B", "OLD"]
        .into_iter()
        .flat_map(|w| [0, 2, 4, 6].map(|lo| (w, lo)))
        .collect();
    assert_eq!(seen, all, "every one of the twelve pairs");
}

/// COPY_W's honest rows for the census (`word_memory_tests`): every
/// functionality, destination offset, shift and window shape, calls across
/// `2^32`, a memcpy to `src + 8` (a copy row a memset's gap rule would take)
/// and padding.
pub(super) fn census_rows() -> Vec<Vec<FE>> {
    let mut run = Run::with_pages(line_pages());
    for offset in 0..8u64 {
        for shift in 0..8u64 {
            for &n in &[1u64, 9, 17] {
                run.copy(
                    ELF + 8 * shift + (offset + shift) % 8,
                    LOW + 0x400 + offset,
                    n,
                );
            }
        }
        run.set(LOW + 0x700 + offset, 13);
        run.commit(ELF + 0x40 + offset, 11);
    }
    run.copy(ELF + 0x100, ELF + 0x108, 24);
    run.copy(LOW, LOW + 0x100, 0);
    across(&mut run);
    distinct_rows(&run)
}

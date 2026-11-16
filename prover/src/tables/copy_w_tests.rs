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
    Bench, ELF_BASE, Eval, INPUT_BASE, Piece, STACK_BASE, Table, constraint_values, page_name,
    page_piece, pages, table, verdict,
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
    if v.left.is_empty() && v.unanswered.is_empty() && v.preprocessed.is_empty() && v.broken == expected {
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
        .map(|c| (0..trace.num_rows()).map(|r| *trace.get_main(r, c)).collect())
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
        Self {
            memory: Memory {
                pages: pages(),
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
    fn guest_call(&mut self, functionality: Functionality, src: u64, dst: u64, count: u64, ts: u64) {
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
            t.main_table.set(row, cpu::NUMBER_HI, FE::from(number >> 32));
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

/// The rows refuse what COPY_W cannot hold (each a completeness limit the
/// module documents) with its reason.
#[test]
fn calls_copy_w_cannot_hold_are_refused_with_their_reason() {
    let refused = |f: Functionality, src: u64, dst: u64, n: u64, why: &str| {
        let e = copy_w::call_rows(f, 8, src, dst, n).expect_err("refused");
        assert!(e.contains(why), "{e}");
    };
    refused(Functionality::Copy, 0x1000, 0x2000, 1 << 32, "below 2^32");
    refused(Functionality::Set, 0x1000, 0x1009, 8, "not src");
    refused(Functionality::Write, 0x1000, (1 << 32) - 4, 8, "passes 2^32");
    // A destination's words past the top of a low word.
    refused(Functionality::Copy, 0x1000, (1 << 32) - 4, 8, "4 GiB");
    // A source whose first word would sit below its high word's range.
    refused(Functionality::Copy, 1 << 32, 7, 8, "below its high word");
    assert!(copy_w::call_rows(Functionality::Copy, 8, 0x1000, (1 << 32) - 8, 8).is_ok());
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
    control.set(&name, row, pw::OFFSET, FE::from(m) + FE::from(3u64) * eighth);
    control.set(&name, row, pw::TIMESTAMP, FE::from(run.rows[0].read_timestamp()));
    assert_eq!(a, 0, "the zero page");
    let t = &control.piece(&name).trace;
    let column = (0..t.num_rows()).map(|r| *t.get_main(r, pw::OFFSET)).collect();
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

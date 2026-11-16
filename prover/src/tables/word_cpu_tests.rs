//! ★ The CPU's word tables (`tables::word_cpu`) on hand-built traces with no
//! proof: MEM_LOAD8, MEM_STORE8, MEM_LOADW, MEM_STOREW, MEM_LOAD2W and
//! MEM_STORE2W fed by a stand-in CPU, with PAGE_W minting the words, every rule
//! the tables add shown load-bearing by a forgery and a control mutant.
//!
//! The bench is `word_memory_tests`': every bus's net as tuple multisets,
//! BITWISE as its preprocessed tuples (`IS_HALF`, `ARE_BYTES`, `MSB8`), every
//! table's constraints on every row, every preprocessed column against the
//! verifier's. The stand-in CPU sends each access's dispatch with LOAD's or
//! STORE's tuple; its timestamp, address, stored value and flags are the
//! verifier's (the clock, the program and the registers fix them), and a load's
//! result is the forger's to claim. A forgery edits an honest bench; the
//! control takes the one check it targets out, and the forgery is then
//! accepted.

use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};

use super::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
use super::gpack::TraceForm;
use super::mem_load2w::{self, Load2WRow, MemLoad2WConstraints, cols as l2};
use super::mem_load8::{self, Load8Row, MemLoad8Constraints, cols as l8};
use super::mem_loadw::{self, LoadWRow, MemLoadWConstraints, cols as lw};
use super::mem_store2w::{self, MemStore2WConstraints, Store2WRow, cols as s2};
use super::mem_store8::{self, MemStore8Constraints, Store8Row, cols as s8};
use super::mem_storew::{self, MemStoreWConstraints, StoreWRow, cols as sw};
use super::page::{DEFAULT_PAGE_SIZE, PageConfig};
use super::page_w::{self, FinalWordMap, FinalWordState, cols as pw};
use super::types::{BusId, FE};
use super::word_cpu::{FLAG_SIGNED, FLAG_STORE, FLAG_W2, FLAG_W4, FLAG_W8, WordClass, word_class};
use super::word_memory::{column, halves};
use super::word_memory_tests::{
    Bench, ELF_BASE, Eval, INPUT_BASE, Piece, STACK_BASE, Table, Verdict, constraint_values,
    multiplicity, page_name, page_piece, pages, table, verdict, widen,
};

// ---- the stand-in CPU ----

/// The stand-in CPU's columns: the verifier's prefix (timestamp, address,
/// stored value, flags, multiplicity), then the result a load claims.
mod cpu {
    pub const TS: usize = 0;
    pub const ADDRESS_LO: usize = 1;
    pub const ADDRESS_HI: usize = 2;
    pub const VALUE_LO: usize = 3;
    pub const VALUE_HI: usize = 4;
    pub const FLAGS: usize = 5;
    pub const MU: usize = 6;
    /// The verifier's columns end here.
    pub const FIXED: usize = 7;
    pub const RESULT_LO: usize = 7;
    pub const RESULT_HI: usize = 8;
    pub const NUM_COLUMNS: usize = 9;
}

/// The CPU's dispatch, sent at `MU`: `(TS, 0, ADDRESS_LO, ADDRESS_HI,
/// VALUE_LO, VALUE_HI, FLAGS, RESULT_LO, RESULT_HI)` (`cpu.rs`' MEMORY send).
fn cpu_interactions() -> Vec<BusInteraction> {
    vec![BusInteraction::sender(
        BusId::MemoryOp,
        Multiplicity::Column(cpu::MU),
        vec![
            column(cpu::TS),
            BusValue::constant(0),
            column(cpu::ADDRESS_LO),
            column(cpu::ADDRESS_HI),
            column(cpu::VALUE_LO),
            column(cpu::VALUE_HI),
            column(cpu::FLAGS),
            column(cpu::RESULT_LO),
            column(cpu::RESULT_HI),
        ],
    )]
}

/// One access as the CPU dispatches it.
#[derive(Clone, Copy, Debug)]
struct Dispatch {
    ts: u64,
    address: u64,
    /// A store's register value, 0 for a load.
    value: u64,
    flags: u64,
    /// A load's result, 0 for a store.
    result: u64,
}

/// The word tables' rows.
#[derive(Default)]
struct Rows {
    load8: Vec<Load8Row>,
    store8: Vec<Store8Row>,
    loadw: Vec<LoadWRow>,
    storew: Vec<StoreWRow>,
    load2w: Vec<Load2WRow>,
    store2w: Vec<Store2WRow>,
}

/// The memory as the honest walk keeps it, word by word, and the rows it
/// makes; one access every 4 timestamps.
struct Cpu {
    pages: Vec<PageConfig>,
    words: FinalWordMap,
    rows: Rows,
    dispatches: Vec<Dispatch>,
    ts: u64,
}

impl Cpu {
    fn new(pages: Vec<PageConfig>) -> Self {
        Self {
            pages,
            words: FinalWordMap::new(),
            rows: Rows::default(),
            dispatches: Vec::new(),
            ts: 4,
        }
    }

    /// The word at `word` (aligned) now.
    fn current(&self, word: u64) -> FinalWordState {
        if let Some(&state) = self.words.get(&word) {
            return state;
        }
        let page = self
            .pages
            .iter()
            .find(|p| {
                word.checked_sub(p.page_base)
                    .is_some_and(|d| d < DEFAULT_PAGE_SIZE as u64)
            })
            .unwrap_or_else(|| panic!("a page holds {word:#x}"));
        FinalWordState {
            timestamp: 0,
            value: page_w::init_word(page, ((word - page.page_base) / 8) as usize),
        }
    }

    fn touch(&mut self, word: u64, value: u64) {
        let timestamp = self.ts;
        self.words.insert(word, FinalWordState { timestamp, value });
    }

    /// A load of `width` bytes at `address`: its row, its dispatch, its result.
    fn load(&mut self, address: u64, width: u8, signed: bool) -> u64 {
        let ts = self.ts;
        let first = address & !7;
        let w0 = self.current(first);
        let result = match word_class(address, width).expect("a CPU access") {
            WordClass::Whole => {
                let row = Load8Row::new(address, ts, w0.timestamp, w0.value).expect("an LD");
                self.rows.load8.push(row);
                w0.value
            }
            WordClass::Within => {
                let row = LoadWRow::new(address, ts, w0.timestamp, width, signed, w0.value)
                    .expect("a narrow load");
                self.rows.loadw.push(row);
                u64::from_le_bytes(row.result().0)
            }
            WordClass::Crossing => {
                let w1 = self.current(first + 8);
                let row = Load2WRow::new(
                    address,
                    ts,
                    [w0.timestamp, w1.timestamp],
                    width,
                    signed,
                    [w0.value, w1.value],
                )
                .expect("a crossing load");
                self.rows.load2w.push(row);
                self.touch(first + 8, w1.value);
                u64::from_le_bytes(row.result().0)
            }
        };
        self.touch(first, w0.value);
        self.dispatches.push(Dispatch {
            ts,
            address,
            value: 0,
            flags: load_flags(width, signed),
            result,
        });
        self.ts += 4;
        result
    }

    /// A store of the `width` low bytes of `value` at `address`.
    fn store(&mut self, address: u64, width: u8, value: u64) {
        let ts = self.ts;
        let first = address & !7;
        let w0 = self.current(first);
        match word_class(address, width).expect("a CPU access") {
            WordClass::Whole => {
                let row =
                    Store8Row::new(address, ts, w0.timestamp, w0.value, value).expect("an SD");
                self.rows.store8.push(row);
                self.touch(first, value);
            }
            WordClass::Within => {
                let row = StoreWRow::new(address, ts, w0.timestamp, width, value, w0.value)
                    .expect("a narrow store");
                self.rows.storew.push(row);
                self.touch(first, row.new_word());
            }
            WordClass::Crossing => {
                let w1 = self.current(first + 8);
                let row = Store2WRow::new(
                    address,
                    ts,
                    [w0.timestamp, w1.timestamp],
                    width,
                    value,
                    [w0.value, w1.value],
                )
                .expect("a crossing store");
                self.rows.store2w.push(row);
                let [n0, n1] = row.new_words();
                self.touch(first, n0);
                self.touch(first + 8, n1);
            }
        }
        self.dispatches.push(Dispatch {
            ts,
            address,
            value,
            flags: store_flags(width),
            result: 0,
        });
        self.ts += 4;
    }

    /// Sets the word at `word` to `value` with no access: a forger's memory
    /// after a forged store.
    fn poke(&mut self, word: u64, value: u64) {
        let state = self.current(word);
        self.words.insert(word, FinalWordState { value, ..state });
    }

    fn bench(&self) -> Bench {
        let mut pieces: Vec<Piece> = self
            .pages
            .iter()
            .map(|config| page_piece(config, &self.words))
            .collect();
        let form = TraceForm::Wide;
        let r = &self.rows;
        pieces.push(piece(
            "MEM_LOAD8",
            mem_load8::bus_interactions(),
            mem_load8::generate_mem_load8_trace_segments_as(&[&r.load8], form),
            load8_constraints,
        ));
        pieces.push(piece(
            "MEM_STORE8",
            mem_store8::bus_interactions(),
            mem_store8::generate_mem_store8_trace_segments_as(&[&r.store8], form),
            store8_constraints,
        ));
        pieces.push(piece(
            "MEM_LOADW",
            mem_loadw::bus_interactions(),
            mem_loadw::generate_mem_loadw_trace_segments_as(&[&r.loadw], form),
            loadw_constraints,
        ));
        pieces.push(piece(
            "MEM_STOREW",
            mem_storew::bus_interactions(),
            mem_storew::generate_mem_storew_trace_segments_as(&[&r.storew], form),
            storew_constraints,
        ));
        pieces.push(piece(
            "MEM_LOAD2W",
            mem_load2w::bus_interactions(),
            mem_load2w::generate_mem_load2w_trace_segments_as(&[&r.load2w], form),
            load2w_constraints,
        ));
        pieces.push(piece(
            "MEM_STORE2W",
            mem_store2w::bus_interactions(),
            mem_store2w::generate_mem_store2w_trace_segments_as(&[&r.store2w], form),
            store2w_constraints,
        ));
        let d = &self.dispatches;
        let cpu = table(d.len(), cpu::NUM_COLUMNS, |t, row| {
            let x = d[row];
            let (a_lo, a_hi) = halves(x.address);
            let (v_lo, v_hi) = halves(x.value);
            let (r_lo, r_hi) = halves(x.result);
            for (c, v) in [
                (cpu::TS, x.ts),
                (cpu::ADDRESS_LO, u64::from(a_lo)),
                (cpu::ADDRESS_HI, u64::from(a_hi)),
                (cpu::VALUE_LO, u64::from(v_lo)),
                (cpu::VALUE_HI, u64::from(v_hi)),
                (cpu::FLAGS, x.flags),
                (cpu::MU, 1),
                (cpu::RESULT_LO, u64::from(r_lo)),
                (cpu::RESULT_HI, u64::from(r_hi)),
            ] {
                t.main_table.set(row, c, FE::from(v));
            }
        });
        let truth = (0..cpu::FIXED)
            .map(|c| (0..cpu.num_rows()).map(|r| *cpu.get_main(r, c)).collect())
            .collect();
        pieces.push(Piece {
            name: "CPU".into(),
            interactions: cpu_interactions(),
            trace: cpu,
            preprocessed: truth,
            constraints: None,
        });
        Bench { pieces }
    }

    /// The dispatch of the access at `ts`, by its row.
    fn dispatch_row(&self, ts: u64) -> usize {
        self.dispatches
            .iter()
            .position(|d| d.ts == ts)
            .unwrap_or_else(|| panic!("an access at {ts}"))
    }

    /// The timestamp of the first access at `address` with `flags`.
    fn ts_of(&self, address: u64, flags: u64) -> u64 {
        self.dispatches
            .iter()
            .find(|d| d.address == address && d.flags == flags)
            .unwrap_or_else(|| panic!("an access at {address:#x}, flags {flags}"))
            .ts
    }
}

fn piece(name: &str, interactions: Vec<BusInteraction>, trace: Table, constraints: Eval) -> Piece {
    Piece {
        name: name.into(),
        interactions,
        trace,
        preprocessed: Vec::new(),
        constraints: Some(constraints),
    }
}

fn load_flags(width: u8, signed: bool) -> u64 {
    let w = match width {
        1 => 0,
        2 => FLAG_W2,
        4 => FLAG_W4,
        _ => FLAG_W8,
    };
    (w + if signed { FLAG_SIGNED } else { 0 }) as u64
}

fn store_flags(width: u8) -> u64 {
    load_flags(width, false) + FLAG_STORE as u64
}

/// Replays `honest`'s accesses on `into`, `before` run before each.
fn replay_with(honest: &Cpu, into: &mut Cpu, mut before: impl FnMut(&mut Cpu, &Dispatch)) {
    for d in &honest.dispatches {
        before(into, d);
        let width = match d.flags & 0x1C {
            0 => 1,
            4 => 2,
            8 => 4,
            _ => 8,
        };
        if d.flags & 1 == 1 {
            into.store(d.address, width, d.value);
        } else {
            into.load(d.address, width, d.flags & 2 == 2);
        }
    }
}

fn load8_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&MemLoad8Constraints, row)
}
fn store8_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&MemStore8Constraints, row)
}
fn loadw_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&MemLoadWConstraints, row)
}
fn storew_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&MemStoreWConstraints, row)
}
fn load2w_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&MemLoad2WConstraints, row)
}
fn store2w_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&MemStore2WConstraints, row)
}

/// Where the honest program's stack accesses sit: a page's last 256 bytes,
/// high word `0xFFFFFFFF`.
const S: u64 = STACK_BASE + 0x3_FF00;
/// The private-input word a narrow load reads last.
const INPUT_WORD: u64 = INPUT_BASE + 0x18;

/// The honest program: every class of access, every width, signed and
/// unsigned with the sign bit set and clear, on a stack page (high word
/// `0xFFFFFFFF`), an ELF page, a private-input page and a zero page.
fn honest() -> Cpu {
    let mut c = Cpu::new(pages());
    // One whole word, then narrower reads inside it.
    c.store(S, 8, 0x8877_6655_4433_2211);
    c.load(S, 8, false);
    c.load(S + 7, 1, true); // 0x88: the sign bit set
    c.load(S + 7, 1, false);
    c.load(S + 6, 2, true); // 0x8877
    c.load(S + 3, 2, false);
    c.load(S + 4, 4, true); // 0x88776655
    c.load(S + 2, 4, false);
    c.load(S + 1, 1, true); // 0x22: the sign bit clear
    // Narrower stores inside one word.
    c.store(S + 1, 1, 0xFFFF_FFFF_FFFF_FFAB);
    c.store(S + 5, 2, 0xCDEF);
    c.store(S + 8, 4, 0x1234_5678);
    c.store(S + 0x20, 8, 0x0102_0304_0506_0708);
    c.store(S + 0x28, 8, 0x1112_1314_1516_1718);
    // Across two words.
    c.load(S + 7, 2, true);
    c.load(S + 6, 4, false);
    c.load(S + 3, 8, false);
    c.load(S + 0x26, 4, false); // the last access to S + 0x20 and S + 0x28
    c.store(S + 0xF, 2, 0x9988);
    c.store(S + 0xD, 4, 0x7766_5544);
    c.store(S + 0x9, 8, 0xA1A2_A3A4_A5A6_A7A8);
    c.load(S + 0x8, 8, false);
    // The ELF page: its words are the verifier's.
    c.load(ELF_BASE + 0x44, 4, true);
    c.load(ELF_BASE + 0x40, 8, false);
    c.store(ELF_BASE + 0x40, 8, 7);
    // The private input: a byte, then a crossing doubleword, and the last
    // access to its word at `INPUT_WORD`.
    c.load(INPUT_BASE + 0x13, 1, false);
    c.load(INPUT_BASE + 0x15, 8, false);
    c.load(INPUT_WORD + 3, 1, false);
    // A zero page: a byte stored and read back.
    c.store(0x101, 1, 0x5A);
    c.load(0x101, 1, false);
    c
}

/// The row of `table` holding the access at `ts` (its `TIMESTAMP` column).
fn row_at(bench: &Bench, table: &str, ts_col: usize, ts: u64) -> usize {
    let t = &bench.piece(table).trace;
    (0..t.num_rows())
        .find(|&r| *t.get_main(r, ts_col) == FE::from(ts))
        .unwrap_or_else(|| panic!("{table} has a row at {ts}"))
}

/// The first padding row of `table` (all zero).
fn spare_row(bench: &Bench, table: &str) -> usize {
    let t = &bench.piece(table).trace;
    (0..t.num_rows())
        .find(|&r| (0..t.main_table.width).all(|c| *t.get_main(r, c) == FE::zero()))
        .unwrap_or_else(|| panic!("{table} has a padding row"))
}

/// `row` of `table` set to zero: an access the forger takes out of it.
fn clear_row(bench: &mut Bench, table: &str, row: usize) {
    let width = bench.piece(table).trace.main_table.width;
    for c in 0..width {
        bench.set(table, row, c, FE::zero());
    }
}

/// `row` of `table` copied into its first padding row, returning that row.
fn copy_row(bench: &mut Bench, table: &str, row: usize) -> usize {
    let spare = spare_row(bench, table);
    let width = bench.piece(table).trace.main_table.width;
    for c in 0..width {
        let v = bench.get(table, row, c);
        bench.set(table, spare, c, v);
    }
    spare
}

/// The verdict with `table`'s constraints `idx` taken out: the control of a
/// forgery those constraints refuse.
fn without_constraints(bench: &Bench, table: &str, idx: &[usize]) -> Verdict {
    let mut v = verdict(bench);
    v.broken
        .retain(|(t, _, k)| !(t == table && idx.contains(k)));
    v
}

/// Only `table`'s constraints in `idx` are broken (on any rows), each of
/// them, and nothing else refuses the bench.
fn refused_only_by_constraints(v: &Verdict, table: &str, idx: &[usize]) -> bool {
    idx.iter()
        .all(|i| v.broken.iter().any(|(t, _, k)| t == table && k == i))
        && v.broken
            .iter()
            .all(|(t, _, k)| t == table && idx.contains(k))
        && v.left.is_empty()
        && v.unanswered.is_empty()
        && v.preprocessed.is_empty()
}

/// Only lookups on `bus` go unanswered, and nothing else refuses the bench.
fn refused_only_by_lookup(v: &Verdict, bus: BusId) -> bool {
    !v.unanswered.is_empty()
        && v.unanswered.iter().all(|(b, _)| *b == u64::from(bus))
        && v.left.is_empty()
        && v.broken.is_empty()
        && v.preprocessed.is_empty()
}

/// `interactions` without the `ARE_BYTES` pair on `(a, b)`.
fn without_pair(interactions: &[BusInteraction], a: usize, b: usize) -> Vec<BusInteraction> {
    let pair = vec![column(a), column(b)];
    let out: Vec<BusInteraction> = interactions
        .iter()
        .filter(|bi| !(bi.bus_id == u64::from(BusId::AreBytes) && bi.values == pair))
        .cloned()
        .collect();
    assert_eq!(out.len() + 1, interactions.len(), "one pair taken out");
    out
}

fn fe(v: u64) -> FE {
    FE::from(v)
}

/// `−k` as a field element.
fn neg(k: u64) -> FE {
    FE::zero() - FE::from(k)
}

/// Sets a row's two step limbs to `step`.
fn set_step(bench: &mut Bench, table: &str, row: usize, limbs: [usize; 2], step: u64) {
    bench.set_u64(table, row, limbs[0], step & 0xFFFF);
    bench.set_u64(table, row, limbs[1], step >> 16);
}

/// The PAGE_W row of the word at `word` on the page at `base`: its name and
/// row.
fn page_row(base: u64, word: u64) -> (String, usize) {
    (page_name(base), ((word - base) / 8) as usize)
}

// ---- the honest bench ----

/// ★ The honest program's bench is accepted: every bus balances, every lookup
/// is answered, every constraint reads zero, on every class of access.
#[test]
fn the_honest_bench_is_accepted() {
    let c = honest();
    let r = &c.rows;
    for (name, n) in [
        ("MEM_LOAD8", r.load8.len()),
        ("MEM_STORE8", r.store8.len()),
        ("MEM_LOADW", r.loadw.len()),
        ("MEM_STOREW", r.storew.len()),
        ("MEM_LOAD2W", r.load2w.len()),
        ("MEM_STORE2W", r.store2w.len()),
    ] {
        assert!(n >= 3, "{name}: {n} rows");
    }
    let v = verdict(&c.bench());
    assert!(v.accepted(), "{v:?}");
}

/// ★ The tables have the shapes their docs give (interactions, columns,
/// constraints), and each record refuses the accesses of the others' classes,
/// a crossing access whose next word would carry into the high word, and a
/// step the limbs cannot hold.
#[test]
fn the_tables_have_their_documented_shapes() {
    let zero = |n: usize| vec![FE::zero(); n];
    for (name, got, want) in [
        (
            "MEM_LOAD8",
            (
                mem_load8::bus_interactions().len(),
                l8::NUM_COLUMNS,
                load8_constraints(&zero(l8::NUM_COLUMNS)).len(),
            ),
            (5, 8, 1),
        ),
        (
            "MEM_STORE8",
            (
                mem_store8::bus_interactions().len(),
                s8::NUM_COLUMNS,
                store8_constraints(&zero(s8::NUM_COLUMNS)).len(),
            ),
            (5, 10, 1),
        ),
        (
            "MEM_LOADW",
            (
                mem_loadw::bus_interactions().len(),
                lw::NUM_COLUMNS,
                loadw_constraints(&zero(lw::NUM_COLUMNS)).len(),
            ),
            (12, 31, 25),
        ),
        (
            "MEM_STOREW",
            (
                mem_storew::bus_interactions().len(),
                sw::NUM_COLUMNS,
                storew_constraints(&zero(sw::NUM_COLUMNS)).len(),
            ),
            (11, 31, 18),
        ),
        (
            "MEM_LOAD2W",
            (
                mem_load2w::bus_interactions().len(),
                l2::NUM_COLUMNS,
                load2w_constraints(&zero(l2::NUM_COLUMNS)).len(),
            ),
            (19, 46, 33),
        ),
        (
            "MEM_STORE2W",
            (
                mem_store2w::bus_interactions().len(),
                s2::NUM_COLUMNS,
                store2w_constraints(&zero(s2::NUM_COLUMNS)).len(),
            ),
            (21, 47, 21),
        ),
    ] {
        assert_eq!(got, want, "{name}: (interactions, columns, constraints)");
    }
    assert!(Load8Row::new(S + 8, 8, 4, 0).is_ok());
    assert!(Load8Row::new(S + 4, 8, 4, 0).is_err());
    assert!(LoadWRow::new(S + 6, 8, 4, 2, false, 0).is_ok());
    assert!(LoadWRow::new(S + 7, 8, 4, 2, false, 0).is_err());
    assert!(LoadWRow::new(S, 8, 4, 8, false, 0).is_err());
    assert!(Load2WRow::new(S + 7, 8, [4, 4], 2, false, [0, 0]).is_ok());
    assert!(Load2WRow::new(S + 6, 8, [4, 4], 2, false, [0, 0]).is_err());
    assert!(Store2WRow::new(S + 1, 8, [4, 4], 8, 0, [0, 0]).is_ok());
    assert!(Store2WRow::new(S, 8, [4, 4], 8, 0, [0, 0]).is_err());
    assert!(
        StoreWRow::new(S + 3, 8, 4, 3, 0, 0).is_err(),
        "a width of 3"
    );
    // A crossing access at the top of a low word would key the next word at
    // a low word of 2^32.
    assert!(Load2WRow::new(0x1_FFFF_FFFC, 8, [4, 4], 8, false, [0, 0]).is_err());
    assert!(Load2WRow::new(0x1_FFFF_FFF4, 8, [4, 4], 8, false, [0, 0]).is_ok());
    // A token not earlier, or a step past 2^32.
    assert!(Load8Row::new(S, 8, 8, 0).is_err());
    assert!(StoreWRow::new(S + 1, 1 << 40, 0, 1, 0, 0).is_err());
}

/// The lookups `name`'s rows send, as BITWISE operations, each as many times
/// as its multiplicity.
fn sent_lookups(bench: &Bench, name: &str) -> Vec<BitwiseOperation> {
    let piece = bench.piece(name);
    let t = &piece.trace;
    let mut out = Vec::new();
    for row in 0..t.num_rows() {
        let g = |c: usize| *t.get_main(row, c);
        for bi in &piece.interactions {
            let m = multiplicity(&bi.multiplicity, &g).canonical_u64();
            let v: Vec<u64> = bi
                .values
                .iter()
                .flat_map(|v| v.combine_from(g))
                .map(|x: FE| x.canonical_u64())
                .collect();
            let op = if bi.bus_id == u64::from(BusId::IsHalfword) {
                BitwiseOperation::halfword(
                    BitwiseOperationType::IsHalf,
                    v[0] as u8,
                    (v[0] >> 8) as u8,
                )
            } else if bi.bus_id == u64::from(BusId::AreBytes) {
                BitwiseOperation::byte_op(BitwiseOperationType::AreBytes, v[0] as u8, v[1] as u8)
            } else if bi.bus_id == u64::from(BusId::Msb8) {
                BitwiseOperation::single_byte(BitwiseOperationType::Msb8, v[0] as u8)
            } else {
                continue;
            };
            out.extend(std::iter::repeat_n(op, m as usize));
        }
    }
    out
}

/// A list of operations as a sortable key.
fn op_keys(ops: &[BitwiseOperation]) -> Vec<(u8, u8, u8, u8)> {
    let mut keys: Vec<_> = ops
        .iter()
        .map(|op| (op.lookup_type as u8, op.x, op.y, op.z))
        .collect();
    keys.sort_unstable();
    keys
}

/// ★ Each record's BITWISE operations are exactly the lookups its row sends
/// (BITWISE's multiplicities are counted from the records), and its fields
/// read back as the access it holds.
#[test]
fn each_records_lookups_are_its_rows_and_its_fields_its_access() {
    let c = honest();
    let bench = c.bench();
    let r = &c.rows;
    let mut checked = 0;
    macro_rules! same {
        ($name:literal, $rows:expr) => {{
            let mut from_records = Vec::new();
            let mut histogram = BitwiseHistogram::new();
            for row in $rows.iter() {
                from_records.extend(row.bitwise_ops());
                row.count_bitwise_into(&mut histogram);
            }
            let sent = sent_lookups(&bench, $name);
            assert_eq!(op_keys(&from_records), op_keys(&sent), "{}", $name);
            let mut from_trace = BitwiseHistogram::new();
            from_trace.add_ops(&sent);
            assert_eq!(histogram.digest(), from_trace.digest(), "{}", $name);
            checked += sent.len();
        }};
    }
    same!("MEM_LOAD8", r.load8);
    same!("MEM_STORE8", r.store8);
    same!("MEM_LOADW", r.loadw);
    same!("MEM_STOREW", r.storew);
    same!("MEM_LOAD2W", r.load2w);
    same!("MEM_STORE2W", r.store2w);
    assert!(checked > 100, "{checked} lookups");

    // The fields read back: each record is its dispatch's access, its old
    // stamps earlier, a load's word its result's source.
    let dispatch = |address: u64, ts: u64| {
        *c.dispatches
            .iter()
            .find(|d| d.address == address && d.ts == ts)
            .expect("a dispatch")
    };
    for row in &r.load8 {
        let d = dispatch(row.address(), row.timestamp());
        assert_eq!(row.word(), d.result);
        assert!(row.old_timestamp() < row.timestamp());
    }
    for row in &r.store8 {
        let d = dispatch(row.address(), row.timestamp());
        assert_eq!(row.new_word(), d.value);
        assert!(row.old_timestamp() < row.timestamp());
    }
    for row in &r.loadw {
        let d = dispatch(row.address(), row.timestamp());
        assert_eq!(u64::from_le_bytes(row.result().0), d.result);
        let o = (row.address() % 8) as usize;
        assert_eq!(row.word().to_le_bytes()[o], d.result as u8);
        assert!(row.old_timestamp() < row.timestamp());
    }
    for row in &r.storew {
        let d = dispatch(row.address(), row.timestamp());
        let o = (row.address() % 8) as usize;
        assert_eq!(row.new_word().to_le_bytes()[o], d.value as u8);
        assert!(row.old_timestamp() < row.timestamp());
    }
    for row in &r.load2w {
        let d = dispatch(row.address(), row.timestamp());
        assert_eq!(u64::from_le_bytes(row.result().0), d.result);
        let o = (row.address() % 8) as usize;
        assert_eq!(row.words()[0].to_le_bytes()[o], d.result as u8);
        assert!(row.old_timestamps().iter().all(|&t| t < row.timestamp()));
    }
    for row in &r.store2w {
        let d = dispatch(row.address(), row.timestamp());
        let o = (row.address() % 8) as usize;
        let [first, second] = row.new_words();
        let mut bytes = first.to_le_bytes().to_vec();
        bytes.extend(second.to_le_bytes());
        let width = match d.flags & 0x1C {
            4 => 2,
            8 => 4,
            _ => 8,
        };
        assert_eq!(&bytes[o..o + width], &d.value.to_le_bytes()[..width]);
        assert!(row.old_timestamps().iter().all(|&t| t < row.timestamp()));
    }
}

// ---- the bytes, all eight checked ----

/// ★ A narrow load reads a byte one less than memory's, its lower neighbour
/// (outside the bytes read) carrying `+256`, the word unchanged: the LWU at
/// `S + 2` claims its low byte one less. Refused by the `ARE_BYTES` pair
/// holding the neighbour alone. Control: that pair removed.
#[test]
fn a_narrow_load_reading_a_split_byte_is_refused_by_its_pair() {
    let c = honest();
    let mut forged = c.bench();
    let ts = c.ts_of(S + 2, load_flags(4, false));
    let row = row_at(&forged, "MEM_LOADW", lw::TIMESTAMP, ts);
    let d = c.dispatch_row(ts);
    // Byte 1 + 256, byte 2 − 1: the word's halves unchanged.
    let b1 = forged.get("MEM_LOADW", row, lw::BYTES[1]);
    let b2 = forged.get("MEM_LOADW", row, lw::BYTES[2]);
    forged.set("MEM_LOADW", row, lw::BYTES[1], b1 + fe(256));
    forged.set("MEM_LOADW", row, lw::BYTES[2], b2 - FE::one());
    forged.set("MEM_LOADW", row, lw::RES[0], b2 - FE::one());
    let claimed = (c.dispatches[d].result & 0xFFFF_FFFF) - 1;
    forged.set_u64("CPU", d, cpu::RESULT_LO, claimed);
    let v = verdict(&forged);
    assert!(refused_only_by_lookup(&v, BusId::AreBytes), "{v:?}");
    assert_eq!(v.unanswered.len(), 1, "{v:?}");

    let control = without_pair(&mem_loadw::bus_interactions(), lw::BYTES[0], lw::BYTES[1]);
    let v = verdict(&forged.with_interactions("MEM_LOADW", control));
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ The same in a crossing load's first word: the LD at `S + 3` claims its
/// low byte one less, byte 2 (outside the bytes read) carrying `+256`. Refused
/// by that pair; control: the pair removed. And in its second word: the last
/// byte read `+256` with the next one `−1` puts the result's high word past
/// `2^32`; refused by that pair, control: removed.
#[test]
fn a_crossing_load_reading_a_split_byte_is_refused_by_its_pair() {
    let c = honest();
    let honest_bench = c.bench();
    let ts = c.ts_of(S + 3, load_flags(8, false));
    let row = row_at(&honest_bench, "MEM_LOAD2W", l2::TIMESTAMP, ts);
    let d = c.dispatch_row(ts);
    let result = c.dispatches[d].result;

    // The first word: byte 2 + 256, byte 3 − 1.
    let mut forged = honest_bench.clone();
    let b2 = forged.get("MEM_LOAD2W", row, l2::BYTES[2]);
    let b3 = forged.get("MEM_LOAD2W", row, l2::BYTES[3]);
    forged.set("MEM_LOAD2W", row, l2::BYTES[2], b2 + fe(256));
    forged.set("MEM_LOAD2W", row, l2::BYTES[3], b3 - FE::one());
    forged.set("MEM_LOAD2W", row, l2::RES[0], b3 - FE::one());
    forged.set_u64("CPU", d, cpu::RESULT_LO, (result & 0xFFFF_FFFF) - 1);
    let v = verdict(&forged);
    assert!(refused_only_by_lookup(&v, BusId::AreBytes), "{v:?}");
    let control = without_pair(&mem_load2w::bus_interactions(), l2::BYTES[2], l2::BYTES[3]);
    let v = verdict(&forged.with_interactions("MEM_LOAD2W", control));
    assert!(v.accepted(), "control: {v:?}");

    // The second word: byte 10 (the last read) + 256, byte 11 − 1.
    let mut forged = honest_bench;
    let b10 = forged.get("MEM_LOAD2W", row, l2::BYTES[10]);
    let b11 = forged.get("MEM_LOAD2W", row, l2::BYTES[11]);
    forged.set("MEM_LOAD2W", row, l2::BYTES[10], b10 + fe(256));
    forged.set("MEM_LOAD2W", row, l2::BYTES[11], b11 - FE::one());
    forged.set("MEM_LOAD2W", row, l2::RES[7], b10 + fe(256));
    forged.set_u64("CPU", d, cpu::RESULT_HI, (result >> 32) + (1 << 32));
    let v = verdict(&forged);
    assert!(refused_only_by_lookup(&v, BusId::AreBytes), "{v:?}");
    let control = without_pair(
        &mem_load2w::bus_interactions(),
        l2::BYTES[10],
        l2::BYTES[11],
    );
    let v = verdict(&forged.with_interactions("MEM_LOAD2W", control));
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ A byte store whose register bytes are split: `VALUE_0 = 0xAB + 256`,
/// `VALUE_1 = 0xFF − 1` keeps the dispatch's low word, and the stored word
/// then holds `0xAB` at the byte and the next byte one more: a byte store
/// that writes two bytes. The forger's memory carries that word to the end.
/// Refused by the `VALUE` pair; control: that pair removed.
#[test]
fn a_store_writing_a_split_register_byte_is_refused_by_its_pair() {
    let honest_cpu = honest();
    let ts = honest_cpu.ts_of(S + 1, store_flags(1));
    // The forger's memory: after the SB, byte 2 of S's word one more.
    let mut forged_cpu = Cpu::new(pages());
    let mut forged_word = None;
    replay_with(&honest_cpu, &mut forged_cpu, |cpu, d| {
        if d.ts == ts + 4 {
            let word = cpu.current(S).value + (1 << 16);
            cpu.poke(S, word);
            forged_word = Some(word);
        }
    });
    let forged_word = forged_word.expect("an access after the SB");
    let mut forged = forged_cpu.bench();
    let row = row_at(&forged, "MEM_STOREW", sw::TIMESTAMP, ts);
    let v0 = forged.get("MEM_STOREW", row, sw::VALUE[0]);
    let v1 = forged.get("MEM_STOREW", row, sw::VALUE[1]);
    forged.set("MEM_STOREW", row, sw::VALUE[0], v0 + fe(256));
    forged.set("MEM_STOREW", row, sw::VALUE[1], v1 - FE::one());
    let (lo, hi) = halves(forged_word);
    forged.set_u64("MEM_STOREW", row, sw::NEW_LO, u64::from(lo));
    forged.set_u64("MEM_STOREW", row, sw::NEW_HI, u64::from(hi));
    let v = verdict(&forged);
    assert!(refused_only_by_lookup(&v, BusId::AreBytes), "{v:?}");
    let control = without_pair(&mem_storew::bus_interactions(), sw::VALUE[0], sw::VALUE[1]);
    let v = verdict(&forged.with_interactions("MEM_STOREW", control));
    assert!(v.accepted(), "control: {v:?}");
    // The memory was changed: the word ends with its byte 2 one more.
    assert_eq!(
        forged_cpu.current(S).value,
        honest_cpu.current(S).value + (1 << 16),
        "the forged byte stays in memory"
    );
}

// ---- the window and the one-hot ----

/// ★ The LWU at `S + 0x26` crosses two words; taken as one word by a
/// MEM_LOADW row (`OFF_6`, `READ4`), it reads bytes 6 and 7 and the
/// placement then forces zeros for the next word's two. Refused by the
/// window (`READ4·(OFF_5 + OFF_6 + OFF_7) = 0`, constraint 15); control: the
/// window removed.
#[test]
fn a_crossing_load_taken_as_one_word_is_refused_by_the_window() {
    let c = honest();
    let mut forged = c.bench();
    // The last access to S + 0x20 and S + 0x28.
    let ts = c.ts_of(S + 0x26, load_flags(4, false));
    let row = row_at(&forged, "MEM_LOAD2W", l2::TIMESTAMP, ts);
    let record = *c
        .rows
        .load2w
        .iter()
        .find(|r| r.timestamp() == ts)
        .expect("its row");
    let old_ts = record.old_timestamps();
    let word = record.words()[0];
    let bytes = word.to_le_bytes();
    let spare = spare_row(&forged, "MEM_LOADW");
    let (a_lo, a_hi) = halves(S + 0x26);
    for (col, v) in [
        (lw::ADDRESS_LO, u64::from(a_lo)),
        (lw::ADDRESS_HI, u64::from(a_hi)),
        (lw::TIMESTAMP, ts),
        (lw::OFF[6], 1),
        (lw::READ4, 1),
        (lw::MU, 1),
        (lw::RES[0], u64::from(bytes[6])),
        (lw::RES[1], u64::from(bytes[7])),
    ] {
        forged.set_u64("MEM_LOADW", spare, col, v);
    }
    for (k, &b) in bytes.iter().enumerate() {
        forged.set_u64("MEM_LOADW", spare, lw::BYTES[k], u64::from(b));
    }
    set_step(
        &mut forged,
        "MEM_LOADW",
        spare,
        [lw::STEP_0, lw::STEP_1],
        ts - old_ts[0] - 1,
    );
    clear_row(&mut forged, "MEM_LOAD2W", row);
    // The second word is not touched now: its final is its state before.
    let (page, r28) = page_row(STACK_BASE, S + 0x28);
    forged.set_u64(&page, r28, pw::TIMESTAMP, old_ts[1]);
    let d = c.dispatch_row(ts);
    let claimed = u64::from(bytes[6]) | u64::from(bytes[7]) << 8;
    forged.set_u64("CPU", d, cpu::RESULT_LO, claimed);
    assert_ne!(claimed, c.dispatches[d].result, "the claim is a lie");
    let v = verdict(&forged);
    assert!(refused_only_by_constraints(&v, "MEM_LOADW", &[15]), "{v:?}");
    let v = without_constraints(&forged, "MEM_LOADW", &[15]);
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ A one-hot that is not a bit vector: at an LBU at offset 2 of a word,
/// `OFF_1 = OFF_3 = 1, OFF_2 = −1` keeps the sum 1 and the key (offset
/// `1 + 3 − 2 = 2`) and places `B_1 + B_3 − B_2`, a byte that is not
/// memory's. Refused by `OFF_2`'s bit (constraint 2); control: that bit
/// removed.
#[test]
fn a_one_hot_that_is_not_bits_is_refused() {
    let mut c = honest();
    c.store(S + 0x40, 8, 0x0000_0000_4433_5011);
    c.load(S + 0x42, 1, false);
    let ts = c.ts - 4;
    let mut forged = c.bench();
    let row = row_at(&forged, "MEM_LOADW", lw::TIMESTAMP, ts);
    let byte = |k: usize| forged.get("MEM_LOADW", row, lw::BYTES[k]).canonical_u64();
    let placed = byte(1) + byte(3) - byte(2);
    assert!(
        placed < 256 && placed != byte(2),
        "a placed byte not memory's"
    );
    forged.set_u64("MEM_LOADW", row, lw::OFF[1], 1);
    forged.set("MEM_LOADW", row, lw::OFF[2], neg(1));
    forged.set_u64("MEM_LOADW", row, lw::OFF[3], 1);
    forged.set_u64("MEM_LOADW", row, lw::RES[0], placed);
    forged.set_u64("MEM_LOADW", row, lw::SIGN_BIT, placed >> 7);
    forged.set_u64("CPU", c.dispatch_row(ts), cpu::RESULT_LO, placed);
    let v = verdict(&forged);
    assert!(refused_only_by_constraints(&v, "MEM_LOADW", &[2]), "{v:?}");
    let v = without_constraints(&forged, "MEM_LOADW", &[2]);
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ A narrow load at `8m + 3` claiming offset 2 names the key `8m + 1`,
/// which no PAGE_W row mints: refused on the word bus alone. Control: PAGE_W's
/// `OFFSET` the prover's (a spare private-input row at `3 + 1/8`) mints the
/// forger's word at `8m + 1`, and the load reads whatever byte it likes there.
#[test]
fn a_narrow_load_at_a_wrong_offset_names_a_key_nothing_minted() {
    let c = honest();
    let mut forged = c.bench();
    // The LBU at INPUT_WORD + 3 is the last access to INPUT_WORD.
    let ts = c.ts_of(INPUT_WORD + 3, load_flags(1, false));
    let row = row_at(&forged, "MEM_LOADW", lw::TIMESTAMP, ts);
    let old_ts = c
        .rows
        .loadw
        .iter()
        .find(|r| r.timestamp() == ts)
        .expect("its row")
        .old_timestamp();
    // The forger's word at `8m + 1`: byte 2 (address 8m + 3) is 0x77.
    let claimed: u64 = 0x77 << 16;
    forged.set_u64("MEM_LOADW", row, lw::OFF[3], 0);
    forged.set_u64("MEM_LOADW", row, lw::OFF[2], 1);
    for (k, b) in claimed.to_le_bytes().into_iter().enumerate() {
        forged.set_u64("MEM_LOADW", row, lw::BYTES[k], u64::from(b));
    }
    forged.set_u64("MEM_LOADW", row, lw::RES[0], 0x77);
    forged.set_u64("MEM_LOADW", row, lw::SIGN_BIT, 0);
    set_step(
        &mut forged,
        "MEM_LOADW",
        row,
        [lw::STEP_0, lw::STEP_1],
        ts - 1,
    );
    forged.set_u64("CPU", c.dispatch_row(ts), cpu::RESULT_LO, 0x77);
    // INPUT_WORD's final is its state before the load.
    let (page, r) = page_row(INPUT_BASE, INPUT_WORD);
    forged.set_u64(&page, r, pw::TIMESTAMP, old_ts);
    let v = verdict(&forged);
    assert!(v.refused_only_on(BusId::MemWord), "{v:?}");
    assert!(
        v.left_on(BusId::MemWord)
            .iter()
            .all(|(tuple, _)| tuple.first() == Some(&((INPUT_WORD + 1) & 0xFFFF_FFFF))),
        "{v:?}"
    );

    // Control: a spare private-input row (word 100, never touched) at
    // `OFFSET = 3 + 1/8` mints the forger's word at `8m + 1` and takes its
    // final.
    let mut control = forged.clone();
    let spare = 100usize;
    let eighth = FE::from(8u64).inv().expect("8 is invertible");
    control.set(&page, spare, pw::OFFSET, fe(r as u64) + eighth);
    let quarters = [claimed, claimed >> 16, claimed >> 32, claimed >> 48].map(|q| q & 0xFFFF);
    for (k, q) in quarters.into_iter().enumerate() {
        control.set_u64(&page, spare, pw::INIT[k], q);
    }
    control.set_u64(&page, spare, pw::FINI_LO, claimed & 0xFFFF_FFFF);
    control.set_u64(&page, spare, pw::FINI_HI, claimed >> 32);
    control.set_u64(&page, spare, pw::TIMESTAMP, ts);
    let v = verdict(&control);
    assert_eq!(v.preprocessed.len(), 1, "only OFFSET refuses it: {v:?}");
    let t = &control.piece(&page).trace;
    let column = (0..t.num_rows())
        .map(|r| *t.get_main(r, pw::OFFSET))
        .collect();
    control.piece_mut(&page).preprocessed[pw::OFFSET] = column;
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

// ---- the two words are adjacent ----

/// MEM_LOAD2W's interactions with its second word keyed at a column of its
/// own (appended after the table's), not `ADDRESS_LO − o + 8`.
fn load2w_second_key_free() -> Vec<BusInteraction> {
    let mut out = mem_load2w::bus_interactions();
    let mut moved = 0;
    for bi in &mut out[4..8] {
        if bi.bus_id == u64::from(BusId::MemWord) {
            bi.values[0] = BusValue::Packed {
                start_column: l2::NUM_COLUMNS,
                packing: Packing::Direct,
            };
            moved += 1;
        }
    }
    assert_eq!(moved, 2, "the second word's two tokens");
    out
}

/// ★ A crossing LWU whose second word is another aligned word: its bytes
/// 8..16 are that word's, keyed at the next word. Refused on the word bus
/// alone (no token holds that word at the next word's key). Control: the
/// second key a free column, set to the other word.
#[test]
fn a_crossing_load_whose_second_word_is_elsewhere_is_refused() {
    let c = honest();
    let honest_bench = c.bench();
    // The LWU at S + 0x26 is the last access to S + 0x20 and S + 0x28. The
    // forger reads S + 0x38's word (never touched) in S + 0x28's place.
    let ts = c.ts_of(S + 0x26, load_flags(4, false));
    let row = row_at(&honest_bench, "MEM_LOAD2W", l2::TIMESTAMP, ts);
    let other = S + 0x38;
    let other_word = c.current(other);
    assert_eq!(other_word.timestamp, 0, "never touched");
    let old_ts = c
        .rows
        .load2w
        .iter()
        .find(|r| r.timestamp() == ts)
        .expect("its row")
        .old_timestamps();
    let mut forged = honest_bench;
    for (k, b) in other_word.value.to_le_bytes().into_iter().enumerate() {
        forged.set_u64("MEM_LOAD2W", row, l2::BYTES[8 + k], u64::from(b));
    }
    // The bytes read: 6 and 7 of the first word, 0 and 1 of the other.
    let bytes: Vec<u64> = (6..10)
        .map(|k| forged.get("MEM_LOAD2W", row, l2::BYTES[k]).canonical_u64())
        .collect();
    for (j, &b) in bytes.iter().enumerate() {
        forged.set_u64("MEM_LOAD2W", row, l2::RES[j], b);
    }
    forged.set_u64("MEM_LOAD2W", row, l2::SIGN_BIT, bytes[3] >> 7);
    let claimed = bytes.iter().rev().fold(0u64, |acc, &b| acc << 8 | b);
    set_step(
        &mut forged,
        "MEM_LOAD2W",
        row,
        [l2::STEP2_0, l2::STEP2_1],
        ts - 1,
    );
    let d = c.dispatch_row(ts);
    forged.set_u64("CPU", d, cpu::RESULT_LO, claimed);
    assert_ne!(claimed, c.dispatches[d].result, "the claim is a lie");
    let (page, r28) = page_row(STACK_BASE, S + 0x28);
    let (_, r38) = page_row(STACK_BASE, other);
    forged.set_u64(&page, r28, pw::TIMESTAMP, old_ts[1]);
    forged.set_u64(&page, r38, pw::TIMESTAMP, ts);
    let v = verdict(&forged);
    assert!(v.refused_only_on(BusId::MemWord), "{v:?}");

    // Control: the second key a column: every other row's the word after its
    // first, the forged row's the other word.
    let mut control = forged;
    control.piece_mut("MEM_LOAD2W").trace = widen(&control.piece("MEM_LOAD2W").trace, 1);
    for r in 0..control.piece("MEM_LOAD2W").trace.num_rows() {
        let offset = (0..8).fold(FE::zero(), |acc, k| {
            acc + fe(k as u64) * control.get("MEM_LOAD2W", r, l2::OFF[k])
        });
        let mu = control.get("MEM_LOAD2W", r, l2::MU);
        let next = control.get("MEM_LOAD2W", r, l2::ADDRESS_LO) - offset + fe(8) * mu;
        control.set("MEM_LOAD2W", r, l2::NUM_COLUMNS, next);
    }
    control.set_u64(
        "MEM_LOAD2W",
        row,
        l2::NUM_COLUMNS,
        u64::from(halves(other).0),
    );
    let v = verdict(&control.with_interactions("MEM_LOAD2W", load2w_second_key_free()));
    assert!(v.accepted(), "control: {v:?}");
}

// ---- the flags are the table's widths ----

/// ★ The first LD (flags 16) taken by a MEM_LOADW row with `SIGNED = 4,
/// READ4 = 1` (`2·4 + 8 = 16`): it reads four bytes and returns a high word of
/// 0 for a word whose high half is not. Refused by `SIGNED`'s bit
/// (constraint 11); control: that bit removed.
#[test]
fn an_ld_taken_by_the_narrow_table_is_refused_by_the_signed_bit() {
    let c = honest();
    let mut forged = c.bench();
    let ts = c.ts_of(S, load_flags(8, false));
    let row8 = row_at(&forged, "MEM_LOAD8", l8::TIMESTAMP, ts);
    let lo = forged.get("MEM_LOAD8", row8, l8::LO).canonical_u64();
    let hi = forged.get("MEM_LOAD8", row8, l8::HI).canonical_u64();
    assert_ne!(hi, 0, "a word with a high half");
    assert_eq!(lo >> 31, 0, "its byte 3's top bit clear: no extension");
    let step = forged.get("MEM_LOAD8", row8, l8::STEP_0).canonical_u64()
        + (forged.get("MEM_LOAD8", row8, l8::STEP_1).canonical_u64() << 16);
    clear_row(&mut forged, "MEM_LOAD8", row8);
    let spare = spare_row(&forged, "MEM_LOADW");
    let (a_lo, a_hi) = halves(S);
    for (col, v) in [
        (lw::ADDRESS_LO, u64::from(a_lo)),
        (lw::ADDRESS_HI, u64::from(a_hi)),
        (lw::TIMESTAMP, ts),
        (lw::OFF[0], 1),
        (lw::READ4, 1),
        (lw::SIGNED, 4),
        (lw::MU, 1),
    ] {
        forged.set_u64("MEM_LOADW", spare, col, v);
    }
    set_step(
        &mut forged,
        "MEM_LOADW",
        spare,
        [lw::STEP_0, lw::STEP_1],
        step,
    );
    let word = lo | hi << 32;
    for (k, b) in word.to_le_bytes().into_iter().enumerate() {
        forged.set_u64("MEM_LOADW", spare, lw::BYTES[k], u64::from(b));
        if k < 4 {
            forged.set_u64("MEM_LOADW", spare, lw::RES[k], u64::from(b));
        }
    }
    forged.set_u64("CPU", c.dispatch_row(ts), cpu::RESULT_HI, 0);
    let v = verdict(&forged);
    assert!(refused_only_by_constraints(&v, "MEM_LOADW", &[11]), "{v:?}");
    let v = without_constraints(&forged, "MEM_LOADW", &[11]);
    assert!(v.accepted(), "control: {v:?}");
}

// ---- the multiplicity is a bit ----

/// ★ A `−1/+1` pair of MEM_LOADW rows for one byte, the `−1` twin with its
/// sign bit set: their dispatches, tokens and pairs cancel, and the twin's
/// `MSB8` of `(b, 1)` taken back pays for a forged LB of the same byte `b`
/// (top bit clear) that claims its sign bit set and returns `b` sign-extended.
/// The twin is refused twice: by `MU`'s bit (constraint 24) and by the
/// one-hot's sum `Σ OFF = MU` (constraint 8). Control: both removed; either
/// alone still refuses it.
#[test]
fn a_minus_one_twin_paying_for_a_forged_sign_bit_is_refused() {
    let c = honest();
    let mut forged = c.bench();
    // The LB at S + 1 reads 0x22 (top bit clear).
    let ts = c.ts_of(S + 1, load_flags(1, true));
    let row = row_at(&forged, "MEM_LOADW", lw::TIMESTAMP, ts);
    assert_eq!(forged.get("MEM_LOADW", row, lw::RES[0]), fe(0x22));
    // The twins: the LB's row as an LBU at +1, and at −1 with its sign bit
    // set; the same dispatch (an LBU extends nothing).
    let plus = copy_row(&mut forged, "MEM_LOADW", row);
    forged.set_u64("MEM_LOADW", plus, lw::SIGNED, 0);
    let minus = copy_row(&mut forged, "MEM_LOADW", plus);
    forged.set("MEM_LOADW", minus, lw::MU, neg(1));
    forged.set_u64("MEM_LOADW", minus, lw::SIGN_BIT, 1);
    // The forged LB: sign bit set, every byte past the first 0xFF.
    forged.set_u64("MEM_LOADW", row, lw::SIGN_BIT, 1);
    forged.set_u64("MEM_LOADW", row, lw::EXT, 1);
    for j in 1..4 {
        forged.set_u64("MEM_LOADW", row, lw::RES[j], 0xFF);
    }
    let d = c.dispatch_row(ts);
    forged.set_u64("CPU", d, cpu::RESULT_LO, 0xFFFF_FF22);
    forged.set_u64("CPU", d, cpu::RESULT_HI, 0xFFFF_FFFF);
    let v = verdict(&forged);
    assert!(
        refused_only_by_constraints(&v, "MEM_LOADW", &[8, 24]),
        "{v:?}"
    );
    assert!(v.broken.iter().all(|(_, r, _)| *r == minus), "{v:?}");
    for one in [8, 24] {
        let v = without_constraints(&forged, "MEM_LOADW", &[one]);
        assert!(!v.accepted(), "{one} alone taken out: {v:?}");
    }
    let v = without_constraints(&forged, "MEM_LOADW", &[8, 24]);
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ Each word table's `MU` is held to a bit: a row at `MU = 2` or `−1`, all
/// else zero, breaks a constraint in every table.
#[test]
fn every_word_tables_multiplicity_is_a_bit() {
    let tables: [(&str, usize, usize, Eval); 6] = [
        ("MEM_LOAD8", l8::NUM_COLUMNS, l8::MU, load8_constraints),
        ("MEM_STORE8", s8::NUM_COLUMNS, s8::MU, store8_constraints),
        ("MEM_LOADW", lw::NUM_COLUMNS, lw::MU, loadw_constraints),
        ("MEM_STOREW", sw::NUM_COLUMNS, sw::MU, storew_constraints),
        ("MEM_LOAD2W", l2::NUM_COLUMNS, l2::MU, load2w_constraints),
        ("MEM_STORE2W", s2::NUM_COLUMNS, s2::MU, store2w_constraints),
    ];
    for (name, width, mu, eval) in tables {
        for bad in [fe(2), neg(1)] {
            let mut row = vec![FE::zero(); width];
            row[mu] = bad;
            assert!(
                eval(&row).iter().any(|v| *v != FE::zero()),
                "{name}: MU at {bad:?} breaks nothing"
            );
        }
    }
}

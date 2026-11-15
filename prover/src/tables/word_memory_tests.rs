//! ★ Memory as 8-byte words (`tables::word_memory`), on hand-built traces with
//! no proof: PAGE_W and MEMW_W, two stand-in chips and a stand-in register
//! file, every rule of the word format shown load-bearing by a forgery and a
//! control mutant, and a census of the block's AIR set under the word format.
//!
//! The bench reads a set of tables the way a verifier would, without the
//! proof:
//! - every bus's net as a multiset of tuples, each tuple with its trailing
//!   zeros stripped, since the LogUp fingerprint has no length term and a short
//!   tuple equals a long one ending in zeros;
//! - BITWISE as its preprocessed tuples: a lookup it has no row for is never
//!   answered, and only its multiplicities are the prover's;
//! - every table's constraints on every row;
//! - every preprocessed column against the verifier's own.
//!
//! The stand-in chips read their pointer from register x12 as HINT does
//! (`hint.rs`' register read) and access two words at `pointer + 8i`; the
//! stand-in register file answers a register read only with the register's
//! true value (its value columns are the verifier's), though a forger may leave
//! a read unanswered. A forgery is a malicious prover's edit of an honest
//! bench; each one a check refuses also runs with that one check taken out (a
//! control mutant) and is then accepted, so the check is what refuses it.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use stark::constraints::builder::ConstraintSet;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;

use super::gpack::TraceForm;
use super::memw_aligned::{self, MemwAlignedOrder, cols as ma};
use super::memw_w::{self, MemwWordConstraints, MemwWordRow, MessageKind, cols as mw};
use super::page::{DEFAULT_PAGE_SIZE, PageConfig};
use super::page_w::{self, FinalWordMap, FinalWordState, cols as pw};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use super::word_memory::{
    access_message, byte_pair_checks, column, halves, step_less_one, step_limbs, term,
    word_of_bytes, write_message,
};
use crate::block_whir;
use crate::tables::fused_memory::MemoryFormat;

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

/// A table's constraints evaluated on one row.
type Eval = fn(&[FE]) -> Vec<FE>;

/// The field's order: `−k` reads as `P − k`.
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The register the stand-in chips read their pointer from (HINT's `out`).
const CHIP_REG: u64 = 12;

// ---- the bench ----

/// One table of a bench.
#[derive(Clone)]
struct Piece {
    name: String,
    interactions: Vec<BusInteraction>,
    trace: Table,
    /// The verifier's own values of the table's preprocessed prefix.
    preprocessed: Vec<Vec<FE>>,
    /// The table's constraints on one row.
    constraints: Option<Eval>,
}

#[derive(Clone, Default)]
struct Bench {
    pieces: Vec<Piece>,
}

impl Bench {
    fn piece(&self, name: &str) -> &Piece {
        self.pieces
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the bench holds {name}"))
    }

    fn piece_mut(&mut self, name: &str) -> &mut Piece {
        self.pieces
            .iter_mut()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the bench holds {name}"))
    }

    fn get(&self, name: &str, row: usize, col: usize) -> FE {
        *self.piece(name).trace.get_main(row, col)
    }

    fn set(&mut self, name: &str, row: usize, col: usize, value: FE) {
        self.piece_mut(name).trace.main_table.set(row, col, value);
    }

    fn set_u64(&mut self, name: &str, row: usize, col: usize, value: u64) {
        self.set(name, row, col, FE::from(value));
    }

    /// `name`'s interactions replaced by `interactions`: a control mutant.
    fn with_interactions(mut self, name: &str, interactions: Vec<BusInteraction>) -> Self {
        self.piece_mut(name).interactions = interactions;
        self
    }

    /// Every table's interactions on `from` moved to `to`.
    fn with_bus_moved(mut self, from: BusId, to: BusId) -> Self {
        for piece in &mut self.pieces {
            for bi in &mut piece.interactions {
                if bi.bus_id == u64::from(from) {
                    bi.bus_id = u64::from(to);
                }
            }
        }
        self
    }
}

/// What a verifier would refuse a bench for.
#[derive(Debug, Default)]
struct Verdict {
    /// Per bus, the tuples whose net is not zero (trailing zeros stripped).
    left: BTreeMap<u64, Vec<(Vec<u64>, i128)>>,
    /// Lookups no BITWISE row answers: `(bus, tuple)`.
    unanswered: Vec<(u64, Vec<u64>)>,
    /// `(table, row, constraint)` that does not read zero.
    broken: Vec<(String, usize, usize)>,
    /// `(table, row, column)` whose preprocessed value is not the verifier's.
    preprocessed: Vec<(String, usize, usize)>,
}

impl Verdict {
    fn accepted(&self) -> bool {
        self.left.is_empty()
            && self.unanswered.is_empty()
            && self.broken.is_empty()
            && self.preprocessed.is_empty()
    }

    fn left_on(&self, bus: BusId) -> &[(Vec<u64>, i128)] {
        self.left
            .get(&u64::from(bus))
            .map_or(&[][..], |v| v.as_slice())
    }

    /// Only `bus` is unbalanced, and nothing else is refused.
    fn refused_only_on(&self, bus: BusId) -> bool {
        self.left.keys().all(|&b| b == u64::from(bus))
            && !self.left_on(bus).is_empty()
            && self.unanswered.is_empty()
            && self.broken.is_empty()
            && self.preprocessed.is_empty()
    }
}

fn multiplicity(m: &Multiplicity, g: &dyn Fn(usize) -> FE) -> FE {
    let linear = |terms: &[LinearTerm]| {
        terms.iter().fold(FE::zero(), |acc, t| match *t {
            LinearTerm::Column {
                coefficient,
                column,
            } => acc + g(column) * FE::from(coefficient),
            LinearTerm::ColumnUnsigned {
                coefficient,
                column,
            } => acc + g(column) * FE::from(coefficient),
            LinearTerm::Constant(v) => acc + FE::from(v),
        })
    };
    match m {
        Multiplicity::One => FE::one(),
        Multiplicity::Column(c) => g(*c),
        Multiplicity::Sum(a, b) => g(*a) + g(*b),
        Multiplicity::Negated(c) => FE::one() - g(*c),
        Multiplicity::Diff(a, b) => g(*a) - g(*b),
        Multiplicity::Sum3(a, b, c) => g(*a) + g(*b) + g(*c),
        Multiplicity::Linear(ts) => linear(ts),
    }
}

/// Whether BITWISE has a row for a lookup tuple (trailing zeros stripped).
fn answered(bus: u64, tuple: &[u64]) -> bool {
    let at = |i: usize| tuple.get(i).copied().unwrap_or(0);
    if bus == u64::from(BusId::IsHalfword) {
        tuple.len() <= 1 && at(0) < 1 << 16
    } else if bus == u64::from(BusId::AreBytes) {
        tuple.len() <= 2 && at(0) < 256 && at(1) < 256
    } else {
        false
    }
}

const LOOKUPS: [BusId; 2] = [BusId::IsHalfword, BusId::AreBytes];

fn signed(v: FE) -> i128 {
    let v = v.canonical_u64();
    if v > u64::MAX / 2 {
        -((FE::zero() - FE::from(v)).canonical_u64() as i128)
    } else {
        v as i128
    }
}

fn verdict(bench: &Bench) -> Verdict {
    let mut net: HashMap<(u64, Vec<u64>), FE> = HashMap::new();
    let mut out = Verdict::default();
    for piece in &bench.pieces {
        let t = &piece.trace;
        for row in 0..t.num_rows() {
            let g = |c: usize| *t.get_main(row, c);
            for bi in &piece.interactions {
                let m = multiplicity(&bi.multiplicity, &g);
                if m == FE::zero() {
                    continue;
                }
                let mut tuple: Vec<u64> = bi
                    .values
                    .iter()
                    .flat_map(|v| v.combine_from(g))
                    .map(|x: FE| x.canonical_u64())
                    .collect();
                while tuple.last() == Some(&0) {
                    tuple.pop();
                }
                let e = net.entry((bi.bus_id, tuple)).or_insert(FE::zero());
                *e = if bi.is_sender { *e + m } else { *e - m };
            }
            if let Some(eval) = piece.constraints {
                let values = eval(&row_of(t, row));
                out.broken.extend(
                    (0..values.len())
                        .filter(|&k| values[k] != FE::zero())
                        .map(|k| (piece.name.clone(), row, k)),
                );
            }
            for (c, column) in piece.preprocessed.iter().enumerate() {
                if *t.get_main(row, c) != column[row] {
                    out.preprocessed.push((piece.name.clone(), row, c));
                }
            }
        }
    }
    let lookups: Vec<u64> = LOOKUPS.iter().map(|&b| u64::from(b)).collect();
    for ((bus, tuple), v) in net {
        if v == FE::zero() {
            continue;
        }
        if lookups.contains(&bus) {
            if !answered(bus, &tuple) {
                out.unanswered.push((bus, tuple));
            }
        } else {
            out.left.entry(bus).or_default().push((tuple, signed(v)));
        }
    }
    out.unanswered.sort();
    for tuples in out.left.values_mut() {
        tuples.sort();
    }
    out
}

fn row_of(t: &Table, r: usize) -> Vec<FE> {
    (0..t.main_table.width).map(|c| *t.get_main(r, c)).collect()
}

/// A constraint set's values on one row.
fn constraint_values<CS: ConstraintSet<GoldilocksField, GoldilocksExtension>>(
    set: &CS,
    row: &[FE],
) -> Vec<FE> {
    use math::field::element::FieldElement;
    use stark::constraints::builder::ProverEvalFolder;
    use stark::frame::Frame;
    use stark::table::TableView;
    use stark::traits::TransitionEvaluationContext;
    let n = set.meta().len();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![row.to_vec()],
        vec![vec![]],
    )]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    set.eval(&mut folder);
    base
}

fn memw_w_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&MemwWordConstraints, row)
}

/// A wide table of `rows` (padded with zero rows to a power of two, at least
/// four) and `width` columns, `fill` writing each row.
fn table(rows: usize, width: usize, fill: impl Fn(&mut Table, usize)) -> Table {
    let num_rows = rows.next_power_of_two().max(4);
    let mut t = TraceTable::new_main(vec![FE::zero(); num_rows * width], width, 1);
    for row in 0..rows {
        fill(&mut t, row);
    }
    t
}

/// `t` with `extra` zero columns appended.
fn widen(t: &Table, extra: usize) -> Table {
    let width = t.main_table.width + extra;
    let mut w = TraceTable::new_main(vec![FE::zero(); t.num_rows() * width], width, 1);
    for row in 0..t.num_rows() {
        for c in 0..t.main_table.width {
            w.main_table.set(row, c, *t.get_main(row, c));
        }
    }
    w
}

// ---- the stand-ins ----

/// The stand-in chips' columns.
mod chip {
    pub const PTR_LO: usize = 0;
    pub const PTR_HI: usize = 1;
    pub const TS: usize = 2;
    pub const MU: usize = 3;
    /// Word `i`'s halves at `W[2i]`, `W[2i + 1]`.
    pub const W: [usize; 4] = [4, 5, 6, 7];
    pub const NUM_COLUMNS: usize = 8;
}

/// A register read of `reg` → `(lo, hi)` at `(ts, 0)` on `Memw`, in HINT's
/// 24-value shape (`is_register` the constant 1, width 2).
fn register_read(reg: u64, lo: BusValue, hi: BusValue, ts: BusValue) -> Vec<BusValue> {
    let mut v = Vec::with_capacity(24);
    v.extend([lo.clone(), hi.clone()]);
    v.extend(std::iter::repeat_n(BusValue::constant(0), 6));
    v.extend([
        BusValue::constant(1),
        BusValue::constant(2 * reg),
        BusValue::constant(0),
    ]);
    v.extend([lo, hi]);
    v.extend(std::iter::repeat_n(BusValue::constant(0), 6));
    v.extend([
        ts,
        BusValue::constant(0),
        BusValue::constant(1),
        BusValue::constant(0),
        BusValue::constant(0),
    ]);
    v
}

/// A stand-in chip: reads its pointer from x12 and writes (or reads) two words
/// at `pointer + 8i`, one message a word.
fn chip_interactions(kind: MessageKind) -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(chip::MU);
    let mut out = vec![BusInteraction::sender(
        BusId::Memw,
        mu(),
        register_read(
            CHIP_REG,
            column(chip::PTR_LO),
            column(chip::PTR_HI),
            column(chip::TS),
        ),
    )];
    for i in 0..2 {
        let address = [
            BusValue::linear(vec![
                term(1, chip::PTR_LO),
                LinearTerm::Constant(8 * i as i64),
            ]),
            column(chip::PTR_HI),
        ];
        let word = || [column(chip::W[2 * i]), column(chip::W[2 * i + 1])];
        let message = match kind {
            MessageKind::Write => write_message(address, column(chip::TS), word()),
            MessageKind::Access => access_message(address, column(chip::TS), word(), word()),
        };
        out.push(BusInteraction::sender(BusId::MemWordMsg, mu(), message));
    }
    out
}

/// The stand-in register file's columns: the verifier's (the register, its
/// value at the timestamp) and the multiplicity.
mod regs {
    pub const REG2: usize = 0;
    pub const VALUE_LO: usize = 1;
    pub const VALUE_HI: usize = 2;
    pub const TS: usize = 3;
    pub const MU: usize = 4;
    pub const NUM_COLUMNS: usize = 5;
}

fn register_file_interactions() -> Vec<BusInteraction> {
    let mut tuple = register_read(
        0,
        column(regs::VALUE_LO),
        column(regs::VALUE_HI),
        column(regs::TS),
    );
    tuple[9] = column(regs::REG2);
    vec![BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(regs::MU),
        tuple,
    )]
}

#[derive(Clone, Copy)]
struct ChipRow {
    ptr: u64,
    ts: u64,
    words: [u64; 2],
}

#[derive(Clone, Copy)]
struct RegRow {
    reg: u64,
    value: u64,
    ts: u64,
}

// ---- the honest model ----

/// The pages of the honest benches: a zero page at 0, an ELF page, a
/// private-input page and the top stack page (high word `0xFFFFFFFF`).
fn pages() -> Vec<PageConfig> {
    let elf: Vec<u8> = (0..4096u32).map(|i| (i * 37 + 11) as u8).collect();
    let input: Vec<u8> = (0..4096u32).map(|i| (i * 101 + 3) as u8).collect();
    vec![
        PageConfig::zero_init(0),
        PageConfig::with_data(ELF_BASE, elf),
        PageConfig::with_private_input(executor::vm::memory::PRIVATE_INPUT_START_INDEX, input),
        PageConfig::zero_init(STACK_BASE),
    ]
}

const ELF_BASE: u64 = 0x40000;
const STACK_BASE: u64 = 0xFFFF_FFFF_FFFC_0000;
const INPUT_BASE: u64 = executor::vm::memory::PRIVATE_INPUT_START_INDEX;

/// Words in memory as the honest accesses leave them, and the rows they make.
struct Model {
    pages: Vec<PageConfig>,
    words: FinalWordMap,
    memw_w: Vec<MemwWordRow>,
    writers: Vec<ChipRow>,
    readers: Vec<ChipRow>,
    regs: Vec<RegRow>,
}

impl Model {
    fn new(pages: Vec<PageConfig>) -> Self {
        Self {
            pages,
            words: FinalWordMap::new(),
            memw_w: Vec::new(),
            writers: Vec::new(),
            readers: Vec::new(),
            regs: Vec::new(),
        }
    }

    fn current(&self, address: u64) -> FinalWordState {
        if let Some(&state) = self.words.get(&address) {
            return state;
        }
        let page = self
            .pages
            .iter()
            .find(|p| {
                address
                    .checked_sub(p.page_base)
                    .is_some_and(|d| d < DEFAULT_PAGE_SIZE as u64)
            })
            .unwrap_or_else(|| panic!("a page holds {address:#x}"));
        FinalWordState {
            timestamp: 0,
            value: page_w::init_word(page, ((address - page.page_base) / 8) as usize),
        }
    }

    fn touch(&mut self, kind: MessageKind, address: u64, ts: u64, new: u64) {
        let old = self.current(address);
        let row = MemwWordRow::new(kind, address, ts, old.timestamp, old.value, new)
            .expect("an honest word access");
        self.memw_w.push(row);
        self.words.insert(
            address,
            FinalWordState {
                timestamp: ts,
                value: new,
            },
        );
    }

    fn write(&mut self, ptr: u64, ts: u64, words: [u64; 2]) {
        self.regs.push(RegRow {
            reg: CHIP_REG,
            value: ptr,
            ts,
        });
        for (i, &w) in words.iter().enumerate() {
            self.touch(MessageKind::Write, ptr + 8 * i as u64, ts, w);
        }
        self.writers.push(ChipRow { ptr, ts, words });
    }

    fn read(&mut self, ptr: u64, ts: u64) -> [u64; 2] {
        self.regs.push(RegRow {
            reg: CHIP_REG,
            value: ptr,
            ts,
        });
        let words = [self.current(ptr).value, self.current(ptr + 8).value];
        for (i, &w) in words.iter().enumerate() {
            self.touch(MessageKind::Access, ptr + 8 * i as u64, ts, w);
        }
        self.readers.push(ChipRow { ptr, ts, words });
        words
    }

    fn bench(&self) -> Bench {
        let mut pieces = Vec::new();
        for config in &self.pages {
            pieces.push(page_piece(config, &self.words));
        }
        pieces.push(Piece {
            name: "MEMW_W".into(),
            interactions: memw_w::bus_interactions(),
            trace: memw_w::generate_memw_w_trace_segments_as(&[&self.memw_w], TraceForm::Wide),
            preprocessed: Vec::new(),
            constraints: Some(memw_w_constraints),
        });
        for (name, rows, kind) in [
            ("WRITER", &self.writers, MessageKind::Write),
            ("READER", &self.readers, MessageKind::Access),
        ] {
            pieces.push(Piece {
                name: name.into(),
                interactions: chip_interactions(kind),
                trace: table(rows.len(), chip::NUM_COLUMNS, |t, row| {
                    let r = rows[row];
                    let (lo, hi) = halves(r.ptr);
                    t.main_table.set(row, chip::PTR_LO, FE::from(u64::from(lo)));
                    t.main_table.set(row, chip::PTR_HI, FE::from(u64::from(hi)));
                    t.main_table.set(row, chip::TS, FE::from(r.ts));
                    t.main_table.set(row, chip::MU, FE::one());
                    for (i, &w) in r.words.iter().enumerate() {
                        let (lo, hi) = halves(w);
                        t.main_table
                            .set(row, chip::W[2 * i], FE::from(u64::from(lo)));
                        t.main_table
                            .set(row, chip::W[2 * i + 1], FE::from(u64::from(hi)));
                    }
                }),
                preprocessed: Vec::new(),
                constraints: None,
            });
        }
        let regs = table(self.regs.len(), regs::NUM_COLUMNS, |t, row| {
            let r = self.regs[row];
            let (lo, hi) = halves(r.value);
            t.main_table.set(row, regs::REG2, FE::from(2 * r.reg));
            t.main_table
                .set(row, regs::VALUE_LO, FE::from(u64::from(lo)));
            t.main_table
                .set(row, regs::VALUE_HI, FE::from(u64::from(hi)));
            t.main_table.set(row, regs::TS, FE::from(r.ts));
            t.main_table.set(row, regs::MU, FE::one());
        });
        // The register file's values are the verifier's: a forger may leave a
        // read unanswered (its multiplicity), never answer it with another value.
        let truth = (0..regs::MU)
            .map(|c| (0..regs.num_rows()).map(|r| *regs.get_main(r, c)).collect())
            .collect();
        pieces.push(Piece {
            name: "REGS".into(),
            interactions: register_file_interactions(),
            trace: regs,
            preprocessed: truth,
            constraints: None,
        });
        Bench { pieces }
    }
}

fn page_name(base: u64) -> String {
    format!("{}{base:#x}", page_w::NAME_PREFIX)
}

fn page_piece(config: &PageConfig, words: &FinalWordMap) -> Piece {
    Piece {
        name: page_name(config.page_base),
        interactions: page_w::bus_interactions(config.page_base, config.is_private_input)
            .expect("an aligned page"),
        trace: page_w::generate_page_w_trace(config, words).expect("an aligned page"),
        preprocessed: page_w::preprocessed_columns_of(config),
        constraints: None,
    }
}

/// The honest bench every forgery starts from: chips write and read words on
/// every page kind, a private input word is read, the stack is written.
fn honest() -> (Model, Bench) {
    let m = honest_model(|_| {});
    let bench = m.bench();
    (m, bench)
}

/// [`honest`]'s accesses, `between` run after the write at 8 and before the
/// read at 12.
fn honest_model(between: impl Fn(&mut Model)) -> Model {
    let mut m = Model::new(pages());
    m.write(0x100, 8, [0x1122_3344_5566_7788, 0xFFFF_FFFF_FFFF_FFFF]);
    between(&mut m);
    m.read(0x100, 12);
    m.read(ELF_BASE + 0x40, 16);
    m.write(ELF_BASE + 0x40, 20, [7, 9]);
    m.read(INPUT_BASE + 0x10, 24);
    m.write(STACK_BASE + 0x3_FF00, 28, [0xDEAD_BEEF, 0xCAFE]);
    m.read(STACK_BASE + 0x3_FF00, 32);
    m
}

/// The MEMW_W row of the access to `address` at `ts`.
fn memw_w_row(m: &Model, address: u64, ts: u64) -> usize {
    m.memw_w
        .iter()
        .position(|r| r.address() == address && r.timestamp() == ts)
        .unwrap_or_else(|| panic!("a MEMW_W row at {address:#x}, {ts}"))
}

// ---- the honest bench ----

/// ★ The honest bench is accepted: every bus balances, every lookup is
/// answered, MEMW_W's constraints read zero and every preprocessed column is
/// the verifier's, on a zero, an ELF, a private-input and a stack page.
#[test]
fn the_honest_bench_is_accepted() {
    let (m, bench) = honest();
    let v = verdict(&bench);
    assert!(v.accepted(), "{v:?}");
    // Each kind of page took part, and the stack page's addresses keep their
    // high word.
    assert_eq!(m.memw_w.len(), 14);
    let stack = bench.piece(&page_name(STACK_BASE));
    let row = (0x3_FF00 / 8) as usize;
    assert_eq!(stack.trace.get_main(row, pw::TIMESTAMP), &FE::from(32u64));
    assert_eq!(
        stack.trace.get_main(row, pw::FINI_LO),
        &FE::from(0xDEAD_BEEFu64)
    );
    assert_eq!(
        stack.interactions[0].values[1],
        BusValue::constant(0xFFFF_FFFF)
    );
}

/// ★ MEMW_W's record: 8-aligned, a strict step of at most `2^32`, laid out
/// as its row, and its BITWISE lookups (what a collector counts) the ones its
/// row sends.
#[test]
fn memw_w_records_lay_out_as_their_rows() {
    use super::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
    let write =
        |address, ts, old_ts| MemwWordRow::new(MessageKind::Write, address, ts, old_ts, 0, 1);
    assert!(write(0x104, 9, 0).is_err(), "unaligned");
    assert!(write(0x100, 9, 9).is_err(), "no later than its token");
    assert!(write(0x100, (1 << 32) + 2, 1).is_err(), "a step past 2^32");
    assert!(write(0x100, (1 << 32) + 1, 1).is_ok(), "a step of 2^32");

    let (m, bench) = honest();
    let t = &bench.piece("MEMW_W").trace;
    let at = |r: usize, c: usize| t.get_main(r, c).canonical_u64();
    let mut from_rows = BitwiseHistogram::new();
    let mut from_trace = BitwiseHistogram::new();
    for (r, row) in m.memw_w.iter().enumerate() {
        assert_eq!(
            at(r, mw::ADDRESS_LO) | at(r, mw::ADDRESS_HI) << 32,
            row.address()
        );
        assert_eq!(at(r, mw::TIMESTAMP), row.timestamp());
        assert_eq!(
            row.old_timestamp(),
            row.timestamp() - 1 - at(r, mw::STEP_0) - (at(r, mw::STEP_1) << 16)
        );
        assert_eq!(at(r, mw::OLD_LO) | at(r, mw::OLD_HI) << 32, row.old());
        assert_eq!(at(r, mw::NEW_LO) | at(r, mw::NEW_HI) << 32, row.new_word());
        row.count_bitwise_into(&mut from_rows);
        for c in [mw::STEP_0, mw::STEP_1] {
            let half = at(r, c);
            from_trace.bump(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                half as u8,
                (half >> 8) as u8,
            ));
        }
    }
    assert!(from_rows == from_trace);
}

/// ★ PAGE_W as PAGE's own tests have it, at word grain: the page grid, the
/// initial words little-endian from the page's bytes, the finals, the
/// interactions by page kind, the refusal of an unaligned base.
#[test]
fn page_w_lays_out_a_page_a_word_a_row() {
    assert_eq!(page_w::ROWS, 1 << 15);
    assert_eq!(page_w::NUM_VARS, 15);
    let data: Vec<u8> = (0..20u8).collect();
    let config = PageConfig::with_data(ELF_BASE, data);
    assert_eq!(page_w::init_word(&config, 0), 0x0706_0504_0302_0100);
    assert_eq!(page_w::init_word(&config, 2), 0x0000_0000_1312_1110);
    assert_eq!(page_w::init_word(&config, 3), 0);
    let mut finals = FinalWordMap::new();
    finals.insert(
        ELF_BASE + 8,
        FinalWordState {
            timestamp: 40,
            value: 0xAABB_CCDD_0011_2233,
        },
    );
    let t = page_w::generate_page_w_trace(&config, &finals).expect("aligned");
    assert_eq!(t.num_rows(), page_w::ROWS);
    let at = |r: usize, c: usize| t.get_main(r, c).canonical_u64();
    assert_eq!(
        (0..4).map(|k| at(0, pw::INIT[k])).collect::<Vec<_>>(),
        [0x0100, 0x0302, 0x0504, 0x0706]
    );
    // Untouched: its final word is its first, at timestamp 0.
    assert_eq!(
        (at(0, pw::FINI_LO), at(0, pw::FINI_HI)),
        (0x0302_0100, 0x0706_0504)
    );
    assert_eq!(at(0, pw::TIMESTAMP), 0);
    // Touched: the final state.
    assert_eq!(
        (at(1, pw::FINI_LO), at(1, pw::FINI_HI)),
        (0x0011_2233, 0xAABB_CCDD)
    );
    assert_eq!(at(1, pw::TIMESTAMP), 40);
    assert_eq!(at(page_w::ROWS - 1, pw::OFFSET), page_w::ROWS as u64 - 1);
    // A zero page: every word 0.
    let zero = page_w::generate_page_w_trace(&PageConfig::zero_init(0), &FinalWordMap::new())
        .expect("aligned");
    assert!(
        (0..page_w::ROWS).all(|r| (1..pw::NUM_COLUMNS).all(|c| *zero.get_main(r, c) == FE::zero()))
    );
    // Interactions: two, six on a private-input page; the high word constant.
    assert_eq!(
        page_w::bus_interactions(0, false).expect("aligned").len(),
        2
    );
    assert_eq!(page_w::bus_interactions(0, true).expect("aligned").len(), 6);
    let high = page_w::bus_interactions(0x0000_0007_0004_0000, false).expect("aligned");
    assert_eq!(high[0].values[1], BusValue::constant(7));
    assert_eq!(
        high[0].values[0],
        BusValue::linear(vec![LinearTerm::Constant(0x4_0000), term(8, pw::OFFSET)])
    );
    // An unaligned base: no table, no trace.
    assert!(page_w::bus_interactions(ELF_BASE + 8, false).is_err());
    assert!(
        page_w::generate_page_w_trace(&PageConfig::zero_init(8), &FinalWordMap::new()).is_err()
    );
    // Preprocessed prefixes by kind, and the private page's lookups by row.
    let private = PageConfig::with_private_input(INPUT_BASE, vec![1, 2, 3]);
    assert_eq!(page_w::num_preprocessed_cols(&private), 1);
    assert_eq!(page_w::preprocessed_columns_of(&private).len(), 1);
    assert_eq!(page_w::num_preprocessed_cols(&config), 5);
    assert_eq!(page_w::preprocessed_columns_of(&config).len(), 5);
    assert_eq!(page_w::bitwise_ops(&private).count(), 4 * page_w::ROWS);
    assert_eq!(page_w::bitwise_ops(&config).count(), 0);
}

/// ★ An ELF page's initial words are the verifier's: a word that differs from
/// the ELF's is refused by the preprocessed columns, and so is a moved
/// `OFFSET`; the page's root moves with one byte.
#[test]
fn an_elf_page_init_word_is_the_verifiers() {
    let (_, bench) = honest();
    let name = page_name(ELF_BASE);
    let mut forged = bench.clone();
    let v = forged.get(&name, 3, pw::INIT[0]) + FE::one();
    forged.set(&name, 3, pw::INIT[0], v);
    let v = verdict(&forged);
    assert_eq!(
        v.preprocessed,
        vec![(name.clone(), 3, pw::INIT[0])],
        "{v:?}"
    );
    let mut forged = bench.clone();
    forged.set_u64(&name, 5, pw::OFFSET, 6);
    let v = verdict(&forged);
    assert_eq!(v.preprocessed, vec![(name.clone(), 5, pw::OFFSET)], "{v:?}");

    let opts = ProofOptions::default_test_options();
    let config = PageConfig::with_data(ELF_BASE, vec![1, 2, 3]);
    let other = PageConfig::with_data(ELF_BASE, vec![1, 2, 4]);
    use stark::leaf_layout::LeafLayout;
    assert_ne!(
        page_w::commitment(&config, &opts, LeafLayout::RowPair),
        page_w::commitment(&other, &opts, LeafLayout::RowPair)
    );
}

// ---- canonical words at PAGE_W's mint ----

/// ★ A private-input page's initial word is canonical: `INIT_1 = 2^16 + y`
/// makes a low half of `2^32 + x`, carried through a chip's read to the page's
/// final word, refused by `IS_HALF` on `INIT_1` alone. Control: that lookup
/// removed, the chip reads a word that is not one.
#[test]
fn a_private_init_word_past_32_bits_is_refused_by_its_half_check() {
    let (m, bench) = honest();
    let name = page_name(INPUT_BASE);
    let word = 0x10 / 8;
    let row = memw_w_row(&m, INPUT_BASE + 0x10, 24);
    let reader = m.readers.iter().position(|r| r.ts == 24).expect("the read");
    let mut forged = bench.clone();
    let lo = forged.get("MEMW_W", row, mw::OLD_LO) + FE::from(1u64 << 32);
    let h1 = forged.get(&name, word, pw::INIT[1]) + FE::from(1u64 << 16);
    forged.set(&name, word, pw::INIT[1], h1);
    forged.set(&name, word, pw::FINI_LO, lo);
    forged.set("MEMW_W", row, mw::OLD_LO, lo);
    forged.set("MEMW_W", row, mw::NEW_LO, lo);
    forged.set("READER", reader, chip::W[0], lo);
    let v = verdict(&forged);
    assert_eq!(
        v.unanswered,
        vec![(u64::from(BusId::IsHalfword), vec![h1.canonical_u64()])],
        "{v:?}"
    );
    assert!(v.left.is_empty() && v.broken.is_empty() && v.preprocessed.is_empty());

    let mut control = page_w::bus_interactions(INPUT_BASE, true).expect("aligned");
    control.remove(2 + 1);
    let v = verdict(&forged.with_interactions(&name, control));
    assert!(v.accepted(), "control: {v:?}");
}

// ---- one access a word a timestamp ----

/// MEMW_W's interactions with the consumed token's timestamp missing its
/// `−1`: a row may consume the token it produces.
fn memw_w_without_the_minus_one() -> Vec<BusInteraction> {
    let mut out = memw_w::bus_interactions();
    out[2].values[2] = BusValue::linear(vec![
        term(1, mw::TIMESTAMP),
        term(-1, mw::STEP_0),
        term(-(1 << 16), mw::STEP_1),
    ]);
    out
}

/// MEMW_W's rows in use.
fn rows_used(bench: &Bench) -> usize {
    let t = &bench.piece("MEMW_W").trace;
    (0..t.num_rows())
        .filter(|&r| {
            *t.get_main(r, mw::MU_WRITE) != FE::zero()
                || *t.get_main(r, mw::MU_ACCESS) != FE::zero()
        })
        .count()
}

/// `bench` under [`memw_w_without_the_minus_one`], every MEMW_W row's
/// `STEP_0` one higher so each consumes the token it consumed before.
fn without_the_minus_one(bench: Bench) -> Bench {
    let mut b = bench.with_interactions("MEMW_W", memw_w_without_the_minus_one());
    for r in 0..rows_used(&b) {
        let s = b.get("MEMW_W", r, mw::STEP_0) + FE::one();
        b.set("MEMW_W", r, mw::STEP_0, s);
    }
    b
}

/// ★ A word written twice at one timestamp: the second MEMW_W row must
/// consume the first's token at a step of 0, and `step − 1 = −1` is no pair of
/// halves. Control: the `−1` removed, the second write balances.
#[test]
fn a_word_written_twice_at_one_timestamp_is_refused_by_the_step() {
    // Honest: a second call writes 0x100 at 9. The forger moves it to 8.
    let m = honest_model(|m| m.write(0x100, 9, [5, 6]));
    let mut forged = m.bench();
    let writer = m
        .writers
        .iter()
        .position(|r| r.ts == 9)
        .expect("the write at 9");
    forged.set_u64("WRITER", writer, chip::TS, 8);
    // x12 at 8 answers both calls.
    let reg_9 = m.regs.iter().position(|r| r.ts == 9).expect("x12 at 9");
    let reg_8 = m.regs.iter().position(|r| r.ts == 8).expect("x12 at 8");
    forged.set_u64("REGS", reg_9, regs::MU, 0);
    forged.set_u64("REGS", reg_8, regs::MU, 2);
    for address in [0x100u64, 0x108] {
        // `8 − 1 − step = 8` needs `step = −1`.
        let r = memw_w_row(&m, address, 9);
        forged.set_u64("MEMW_W", r, mw::TIMESTAMP, 8);
        forged.set("MEMW_W", r, mw::STEP_0, FE::zero() - FE::one());
        // The read at 12 now steps from 8.
        let r = memw_w_row(&m, address, 12);
        let s = forged.get("MEMW_W", r, mw::STEP_0) + FE::one();
        forged.set("MEMW_W", r, mw::STEP_0, s);
    }
    let v = verdict(&forged);
    assert_eq!(
        v.unanswered,
        vec![(u64::from(BusId::IsHalfword), vec![P - 1])],
        "{v:?}"
    );
    assert!(v.left.is_empty() && v.broken.is_empty(), "{v:?}");

    let v = verdict(&without_the_minus_one(forged));
    assert!(v.accepted(), "control: {v:?}");
}

// ---- tokens exist only at aligned words ----

/// The honest bench plus a chip call writing two words at `ptr` (x12's true
/// value) at `write_ts` and, if `read_ts`, one reading them back: MEMW_W rows
/// laid out by hand, the write's consuming words minted at timestamp 0 with
/// value 0. What a forger lays out for an unaligned pointer.
fn with_unaligned_calls(ptr: u64, write_ts: u64, words: [u64; 2], read_ts: Option<u64>) -> Bench {
    let mut m = honest_model(|_| {});
    let mut calls = vec![(write_ts, MessageKind::Write)];
    calls.extend(read_ts.map(|ts| (ts, MessageKind::Access)));
    let first = m.memw_w.len();
    for &(ts, kind) in &calls {
        m.regs.push(RegRow {
            reg: CHIP_REG,
            value: ptr,
            ts,
        });
        let row = ChipRow { ptr, ts, words };
        match kind {
            MessageKind::Write => m.writers.push(row),
            MessageKind::Access => m.readers.push(row),
        }
        for (i, &w) in words.iter().enumerate() {
            // `MemwWordRow::new` refuses an unaligned address: lay out its
            // aligned twin, then move the address.
            let (old_ts, old) = match kind {
                MessageKind::Write => (0, 0),
                MessageKind::Access => (write_ts, w),
            };
            let row = MemwWordRow::new(kind, ptr - ptr % 8 + 8 * i as u64, ts, old_ts, old, w)
                .expect("an aligned twin");
            m.memw_w.push(row);
        }
    }
    let mut bench = m.bench();
    for r in first..m.memw_w.len() {
        let (lo, _) = halves(ptr + 8 * ((r - first) % 2) as u64);
        bench.set_u64("MEMW_W", r, mw::ADDRESS_LO, u64::from(lo));
    }
    bench
}

/// ★ A chip writing at `8k + 4` (the pointer in x12 really is unaligned):
/// MEMW_W's rows consume tokens no PAGE_W row minted, and the word bus alone
/// is left unbalanced. Controls: (a) PAGE_W's `OFFSET` not the verifier's,
/// set to `k + 1/2`, mints the tokens at `8k + 4`; (b) the step without its
/// `−1`, each row consumes its own token.
#[test]
fn an_unaligned_chip_write_consumes_tokens_nothing_minted() {
    let k = 0x200u64;
    let ptr = 8 * k + 4;
    let words = [0xAB, 0xCD];
    let bench = with_unaligned_calls(ptr, 40, words, None);
    let v = verdict(&bench);
    assert!(v.refused_only_on(BusId::MemWord), "{v:?}");
    assert!(
        v.left_on(BusId::MemWord)
            .iter()
            .all(|(tuple, _)| tuple.first().is_some_and(|lo| lo % 8 == 4)),
        "{v:?}"
    );

    // (a) Rows k and k + 1 of page 0 at `OFFSET + 1/2`: they mint at 8k + 4
    // and 8k + 12 and take the writes' finals.
    let mut a = bench.clone();
    let name = page_name(0);
    let half = FE::from(2u64).inv().expect("2 is invertible");
    for (i, row) in [k, k + 1].into_iter().enumerate() {
        let row = row as usize;
        a.set(&name, row, pw::OFFSET, FE::from(row as u64) + half);
        a.set_u64(&name, row, pw::TIMESTAMP, 40);
        a.set_u64(&name, row, pw::FINI_LO, words[i]);
    }
    let v = verdict(&a);
    assert_eq!(v.preprocessed.len(), 2, "only OFFSET refuses it: {v:?}");
    let t = &a.piece(&name).trace;
    let column = (0..t.num_rows())
        .map(|r| *t.get_main(r, pw::OFFSET))
        .collect();
    a.piece_mut(&name).preprocessed[pw::OFFSET] = column;
    let v = verdict(&a);
    assert!(v.accepted(), "control (a): {v:?}");

    // (b) No `−1`: the two forged rows consume what they produce.
    let mut b = without_the_minus_one(bench);
    let n = rows_used(&b);
    for r in [n - 2, n - 1] {
        b.set("MEMW_W", r, mw::STEP_0, FE::zero());
        b.set("MEMW_W", r, mw::STEP_1, FE::zero());
        let new_lo = b.get("MEMW_W", r, mw::NEW_LO);
        b.set("MEMW_W", r, mw::OLD_LO, new_lo);
    }
    let v = verdict(&b);
    assert!(v.accepted(), "control (b): {v:?}");
}

// ---- a word's bytes, all eight checked ----

/// ★ The byte gadget: eight bytes summed into a word's halves, four pair
/// checks. A forger keeping the word moves a byte to `256 + x` and its
/// neighbour down one; the pair check refuses it. Control: that pair removed.
#[test]
fn a_byte_past_255_is_refused_by_its_pair() {
    const BYTES: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    const MU: usize = 8;
    // A table exposing a word's bytes, and one taking the word: on a bus of
    // the test's own.
    let bus = 1000u64;
    let split = {
        let mut i = byte_pair_checks(BYTES, Multiplicity::Column(MU));
        i.push(BusInteraction::sender(
            bus,
            Multiplicity::Column(MU),
            vec![
                word_of_bytes([0, 1, 2, 3]),
                word_of_bytes([4, 5, 6, 7]),
                column(BYTES[1]),
            ],
        ));
        i
    };
    let word: u64 = 0x8070_6050_4030_2010;
    let take = BusInteraction::receiver(
        bus,
        Multiplicity::One,
        vec![
            BusValue::constant(word & 0xFFFF_FFFF),
            BusValue::constant(word >> 32),
            column(0),
        ],
    );
    let bytes = word.to_le_bytes();
    let mut split_row: Vec<FE> = bytes.iter().map(|&b| FE::from(u64::from(b))).collect();
    split_row.push(FE::one());
    let honest = Bench {
        pieces: vec![
            Piece {
                name: "SPLIT".into(),
                interactions: split.clone(),
                trace: TraceTable::new_main(split_row, 9, 1),
                preprocessed: Vec::new(),
                constraints: None,
            },
            Piece {
                name: "TAKE".into(),
                interactions: vec![take],
                trace: TraceTable::new_main(vec![FE::from(u64::from(bytes[1]))], 1, 1),
                preprocessed: Vec::new(),
                constraints: None,
            },
        ],
    };
    assert!(verdict(&honest).accepted());

    // Byte 0 + 256 and byte 1 − 1: the word is unchanged, byte 1 is a lie.
    let mut forged = honest.clone();
    forged.set("SPLIT", 0, BYTES[0], FE::from(u64::from(bytes[0]) + 256));
    forged.set("SPLIT", 0, BYTES[1], FE::from(u64::from(bytes[1]) - 1));
    forged.set("TAKE", 0, 0, FE::from(u64::from(bytes[1]) - 1));
    let v = verdict(&forged);
    assert_eq!(
        v.unanswered,
        vec![(
            u64::from(BusId::AreBytes),
            vec![u64::from(bytes[0]) + 256, u64::from(bytes[1]) - 1]
        )],
        "{v:?}"
    );
    assert!(v.left.is_empty());
    let mut control = split;
    control.remove(0);
    let v = verdict(&forged.with_interactions("SPLIT", control));
    assert!(v.accepted(), "control: {v:?}");
}

// ---- PAGE_W's final words ----

/// ★ A PAGE_W final word, or final timestamp, that is not the last access's
/// is refused on the word bus alone.
#[test]
fn a_final_word_or_timestamp_not_the_last_tokens_is_refused() {
    let (_, bench) = honest();
    let name = page_name(0);
    let row = 0x100 / 8;
    for column in [pw::FINI_LO, pw::FINI_HI, pw::TIMESTAMP] {
        let mut forged = bench.clone();
        let v = forged.get(&name, row, column) + FE::one();
        forged.set(&name, row, column, v);
        let v = verdict(&forged);
        assert!(v.refused_only_on(BusId::MemWord), "column {column}: {v:?}");
    }
    // An untouched word's final not its first.
    let mut forged = bench.clone();
    forged.set_u64(&name, 7, pw::FINI_LO, 1);
    assert!(verdict(&forged).refused_only_on(BusId::MemWord));
}

// ---- the address, the bus, the message, the pair ----

/// A field element as the signed constant a [`LinearTerm::Constant`] holds.
fn field_i64(v: FE) -> i64 {
    let v = v.canonical_u64();
    if v > P / 2 {
        -((P - v) as i64)
    } else {
        v as i64
    }
}

/// Word-bus keys packed into one element, `lo + 2^32·hi`: the layout the word
/// format forbids, as a control mutant.
fn packed(interactions: Vec<BusInteraction>) -> Vec<BusInteraction> {
    interactions
        .into_iter()
        .map(|mut bi| {
            if bi.bus_id == u64::from(BusId::MemWord) {
                let hi = bi.values.remove(1);
                let mut terms = match bi.values.remove(0) {
                    BusValue::Linear(t) => t,
                    BusValue::Packed { start_column, .. } => vec![term(1, start_column)],
                };
                match hi {
                    BusValue::Linear(t) => terms.extend(t.into_iter().map(|t| match t {
                        LinearTerm::Constant(c) => LinearTerm::Constant(field_i64(
                            FE::from(c as u64) * FE::from(1u64 << 32),
                        )),
                        other => panic!("an unexpected high word {other:?}"),
                    })),
                    BusValue::Packed { start_column, .. } => {
                        terms.push(term(1 << 32, start_column));
                    }
                }
                bi.values.insert(0, BusValue::Linear(terms));
            }
            bi
        })
        .collect()
}

/// ★ The address is two elements. Packed into one,
/// `lo + 2^32·hi`, the stack's word at low word `8m` (high word `0xFFFFFFFF`)
/// is the same element as a low pointer `8m − 1`, and a chip reading at that
/// low pointer reads the stack. Two elements refuse it.
#[test]
fn a_packed_address_lets_a_low_pointer_read_the_stack() {
    let (m, bench) = honest();
    let stack_word = STACK_BASE + 0x3_FF00;
    let (stack_lo, _) = halves(stack_word);
    // The forged read: x12 = stack_lo − 1, high word 0, reading the two stack
    // words as if they were at that low pointer.
    let low = u64::from(stack_lo) - 1;
    let reader = m.readers.iter().position(|r| r.ts == 32).expect("the read");
    let mut forged = bench.clone();
    let r = m.regs.iter().position(|r| r.ts == 32).expect("x12 at 32");
    // The register file's truth for this read moves with it: the guest really
    // passed the low pointer.
    forged.piece_mut("REGS").preprocessed[regs::VALUE_LO][r] = FE::from(low);
    forged.piece_mut("REGS").preprocessed[regs::VALUE_HI][r] = FE::zero();
    forged.set_u64("REGS", r, regs::VALUE_LO, low);
    forged.set_u64("REGS", r, regs::VALUE_HI, 0);
    forged.set_u64("READER", reader, chip::PTR_LO, low);
    forged.set_u64("READER", reader, chip::PTR_HI, 0);
    for i in 0..2u64 {
        let row = memw_w_row(&m, stack_word + 8 * i, 32);
        forged.set_u64("MEMW_W", row, mw::ADDRESS_LO, low + 8 * i);
        forged.set_u64("MEMW_W", row, mw::ADDRESS_HI, 0);
    }
    let v = verdict(&forged);
    assert!(v.refused_only_on(BusId::MemWord), "{v:?}");

    // Control: every key packed. The stack page's keys and the low keys meet.
    let mut control = forged.clone();
    for piece in &mut control.pieces {
        if piece.name.starts_with(page_w::NAME_PREFIX) || piece.name == "MEMW_W" {
            piece.interactions = packed(std::mem::take(&mut piece.interactions));
        }
    }
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// The honest bench's tables moved onto the byte-token bus, a chip writing
/// eight zero bytes at the unaligned byte address `b` through a MEMW_A row,
/// and eight pages at bases `(b + j)·2^32` whose first rows answer it.
fn byte_table_beside_words(b: u64, ts: u64) -> Bench {
    let (_, mut bench) = honest();
    for j in 0..8u64 {
        // Row 0's final: `(0, b + j, 0, ts, 0)` read as the byte token
        // `(RAM, b + j, ts, 0)`.
        let config = PageConfig::zero_init((b + j) << 32);
        let mut piece = page_piece(&config, &FinalWordMap::new());
        piece.trace.main_table.set(0, pw::FINI_LO, FE::from(ts));
        bench.pieces.push(piece);
    }
    // A chip's 8-byte write of zeros at `b`, in today's MEMW message.
    const ADDR: usize = 0;
    const TS: usize = 1;
    const MU: usize = 2;
    let mut message = vec![BusValue::constant(0), column(ADDR), BusValue::constant(0)];
    message.extend(std::iter::repeat_n(BusValue::constant(0), 8));
    message.extend([
        column(TS),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(1),
    ]);
    bench.pieces.push(Piece {
        name: "BYTE_CHIP".into(),
        interactions: vec![BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(MU),
            message,
        )],
        trace: table(1, 3, |t, _| {
            t.main_table.set(0, ADDR, FE::from(b));
            t.main_table.set(0, TS, FE::from(ts));
            t.main_table.set(0, MU, FE::one());
        }),
        preprocessed: Vec::new(),
        constraints: None,
    });
    // The MEMW_A row taking it: eight byte tokens from timestamp 0.
    let (d0, d1) = step_limbs(step_less_one(0, ts).expect("a step"));
    bench.pieces.push(Piece {
        name: "MEMW_A".into(),
        interactions: memw_aligned::bus_interactions_for(MemwAlignedOrder::Limbs),
        trace: table(1, ma::NUM_COLUMNS, |t, _| {
            t.main_table
                .set(0, ma::BASE_ADDRESS[0], FE::from(b & 0xFFFF));
            t.main_table.set(0, ma::BASE_ADDRESS[1], FE::from(b >> 16));
            t.main_table.set(0, ma::TIMESTAMP_0, FE::from(ts));
            t.main_table.set(0, ma::WRITE8, FE::one());
            t.main_table.set(0, ma::STEP_0, FE::from(u64::from(d0)));
            t.main_table.set(0, ma::STEP_1, FE::from(u64::from(d1)));
            t.main_table.set(0, ma::MU_WRITE, FE::one());
        }),
        preprocessed: Vec::new(),
        constraints: None,
    });
    bench
}

/// ★ Word tokens on their own bus. On `BusId::Memory`
/// beside a byte table, a page at `(b + j)·2^32` mints the byte token at `b +
/// j`, and a chip's unaligned 8-byte write of zeros at `b` balances; on
/// `BusId::MemWord` the byte table's tokens meet nothing.
#[test]
fn word_tokens_on_the_byte_bus_let_a_byte_table_mint() {
    let b = 0x1003u64;
    let bench = byte_table_beside_words(b, 44);
    let v = verdict(&bench);
    assert!(!v.left_on(BusId::Memory).is_empty(), "{v:?}");
    assert!(!v.left_on(BusId::MemWord).is_empty(), "{v:?}");
    assert!(v.unanswered.is_empty() && v.broken.is_empty(), "{v:?}");

    let control = bench.with_bus_moved(BusId::MemWord, BusId::Memory);
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// The register-read mutant's extra columns: `is_register`, the width flags, and the
/// multiplicity of a row that takes a `Memw` message.
const RR_IS_REGISTER: usize = mw::NUM_COLUMNS;
const RR_W2: usize = mw::NUM_COLUMNS + 1;
const RR_MU: usize = mw::NUM_COLUMNS + 4;

/// MEMW_W also receiving `Memw` messages in the register read's shape, with
/// `is_register` and the width flags its own columns (a control mutant): its
/// word check fires on either kind of message.
fn memw_w_receiving_memw() -> Vec<BusInteraction> {
    let mu = Multiplicity::Sum3(mw::MU_WRITE, mw::MU_ACCESS, RR_MU);
    let mut out = memw_w::bus_interactions();
    for bi in &mut out[..4] {
        bi.multiplicity = mu.clone();
    }
    let mut tuple = Vec::with_capacity(24);
    tuple.extend([column(mw::OLD_LO), column(mw::OLD_HI)]);
    tuple.extend(std::iter::repeat_n(BusValue::constant(0), 6));
    tuple.extend([
        column(RR_IS_REGISTER),
        column(mw::ADDRESS_LO),
        column(mw::ADDRESS_HI),
        column(mw::NEW_LO),
        column(mw::NEW_HI),
    ]);
    tuple.extend(std::iter::repeat_n(BusValue::constant(0), 6));
    tuple.extend([
        column(mw::TIMESTAMP),
        BusValue::constant(0),
        column(RR_W2),
        column(RR_W2 + 1),
        column(RR_W2 + 2),
    ]);
    out.push(BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(RR_MU),
        tuple,
    ));
    out
}

/// ★ A register read never reaches MEMW_W. RAM word 24
/// (`2·12`) holds 0x300, and a chip claims x12 = 0x300 at 8 when x12 is 0x100:
/// its writes land at 0x300. Refused on `Memw` alone, where only the register
/// file answers a register read. Control: MEMW_W receiving `Memw` with
/// `is_register` a column answers it from RAM word 24, and the forgery
/// balances.
#[test]
fn a_register_read_is_never_answered_from_ram() {
    // The memory as the forgery has it: word 24 written with 0x300 at 2, the
    // call at 8 writing at 0x300, and the reads at 12 reading 0x300.
    let mut m = Model::new(pages());
    m.write(24, 2, [0x300, 0]);
    m.write(0x300, 8, [0x77, 0x88]);
    m.read(0x300, 12);
    let mut forged = m.bench();
    // The register file's truth: x12 = 0x100 at 8. The forger leaves that read
    // unanswered.
    let r = m.regs.iter().position(|r| r.ts == 8).expect("x12 at 8");
    forged.piece_mut("REGS").preprocessed[regs::VALUE_LO][r] = FE::from(0x100u64);
    forged.set_u64("REGS", r, regs::VALUE_LO, 0x100);
    forged.set_u64("REGS", r, regs::MU, 0);
    let v = verdict(&forged);
    assert!(v.refused_only_on(BusId::Memw), "{v:?}");

    // Control: one more MEMW_W row reads RAM word 24 at 8 for the chip's
    // register read; word 24's last token moves from 2 to 8.
    let mut control = forged.with_interactions("MEMW_W", memw_w_receiving_memw());
    let wide = widen(&control.piece("MEMW_W").trace, 5);
    control.piece_mut("MEMW_W").trace = wide;
    let row = rows_used(&control);
    for (c, value) in [
        (mw::ADDRESS_LO, 24),
        (mw::TIMESTAMP, 8),
        (mw::OLD_LO, 0x300),
        (mw::NEW_LO, 0x300),
        (mw::STEP_0, 8 - 2 - 1),
        (RR_IS_REGISTER, 1),
        (RR_W2, 1),
        (RR_MU, 1),
    ] {
        control.set_u64("MEMW_W", row, c, value);
    }
    control.set_u64(&page_name(0), 24 / 8, pw::TIMESTAMP, 8);
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// MEMW_W producing at `MU_WRITE` and consuming at `MU_ACCESS`: a pair whose
/// multiplicities differ, as a control mutant.
fn memw_w_unequal_pair() -> Vec<BusInteraction> {
    let mut out = memw_w::bus_interactions();
    out[2].multiplicity = Multiplicity::Column(mw::MU_ACCESS);
    out[3].multiplicity = Multiplicity::Column(mw::MU_WRITE);
    out
}

/// ★ A token pair with unequal multiplicities mints: under
/// the mutant a chip writes two words at an unaligned pointer and reads them
/// back, the write rows producing and the read rows consuming, with no page
/// behind them, and it balances; the real MEMW_W refuses it. The census
/// refuses the mutant table.
#[test]
fn an_unequal_token_pair_mints_and_the_census_refuses_it() {
    let bench = with_unaligned_calls(8 * 0x200 + 4, 40, [0xAB, 0xCD], Some(44));
    let v = verdict(&bench);
    assert!(v.refused_only_on(BusId::MemWord), "{v:?}");
    let mutant = bench.with_interactions("MEMW_W", memw_w_unequal_pair());
    // The honest rows break under the mutant (a write consumes nothing): read
    // only the unaligned words' tokens.
    let v = verdict(&mutant);
    assert!(
        v.left_on(BusId::MemWord)
            .iter()
            .all(|(tuple, _)| tuple.first().is_none_or(|lo| lo % 8 == 0)),
        "the unaligned words balance under the mutant: {v:?}"
    );
    assert!(census_table("MEMW_W", &memw_w_unequal_pair(), 0).is_err());
    assert!(census_table("MEMW_W", &memw_w::bus_interactions(), 0).is_ok());
}

/// A −1/+1 pair of MEMW_W rows taking one write message: the −1 row produces a
/// forged token where the +1 row consumes the real one, so a later read sees
/// the forged word. MEMW_W's bit constraint refuses it.
#[test]
fn a_minus_one_twin_is_refused_by_the_bit_constraint() {
    let (m, bench) = honest();
    // The read at 12 of 0x100 consumes (0x100, 8, real); forge it to read 0x55.
    let read = memw_w_row(&m, 0x100, 12);
    let real = m.memw_w[read].old();
    let mut forged = bench.clone();
    forged.set_u64("MEMW_W", read, mw::OLD_LO, 0x55);
    forged.set_u64("MEMW_W", read, mw::OLD_HI, 0);
    forged.set_u64("MEMW_W", read, mw::NEW_LO, 0x55);
    forged.set_u64("MEMW_W", read, mw::NEW_HI, 0);
    let reader = m.readers.iter().position(|r| r.ts == 12).expect("the read");
    forged.set_u64("READER", reader, chip::W[0], 0x55);
    forged.set_u64("READER", reader, chip::W[1], 0);
    let page = page_name(0);
    forged.set_u64(&page, 0x100 / 8, pw::FINI_LO, 0x55);
    forged.set_u64(&page, 0x100 / 8, pw::FINI_HI, 0);
    // Twins: one write message (0x100, 30, 0) received at +1 and at −1; the +1
    // row consumes (0x100, 8, real), the −1 row "consumes" (produces)
    // (0x100, 8, 0x55).
    let n = rows_used(&forged);
    let (real_lo, real_hi) = halves(real);
    for (k, (mu, old_lo, old_hi)) in [
        (FE::one(), u64::from(real_lo), u64::from(real_hi)),
        (FE::zero() - FE::one(), 0x55, 0),
    ]
    .into_iter()
    .enumerate()
    {
        let r = n + k;
        forged.set_u64("MEMW_W", r, mw::ADDRESS_LO, 0x100);
        forged.set_u64("MEMW_W", r, mw::TIMESTAMP, 30);
        forged.set_u64("MEMW_W", r, mw::OLD_LO, old_lo);
        forged.set_u64("MEMW_W", r, mw::OLD_HI, old_hi);
        forged.set_u64("MEMW_W", r, mw::STEP_0, 30 - 8 - 1);
        forged.set("MEMW_W", r, mw::MU_WRITE, mu);
    }
    let v = verdict(&forged);
    assert!(v.left.is_empty() && v.unanswered.is_empty(), "{v:?}");
    // The −1 row breaks `MU_WRITE`'s bit and the messages' sum bit, and
    // nothing else is refused.
    assert_eq!(
        v.broken,
        vec![
            ("MEMW_W".to_string(), n + 1, 0),
            ("MEMW_W".to_string(), n + 1, 2)
        ],
        "{v:?}"
    );
    let mut control = forged;
    control.piece_mut("MEMW_W").constraints = None;
    assert!(verdict(&control).accepted());
}

/// ★ MEMW_W's constraints: a row takes at most one message, each multiplicity
/// a bit.
#[test]
fn memw_w_constraints_hold_its_multiplicities_to_bits() {
    let mut row = vec![FE::zero(); mw::NUM_COLUMNS];
    assert!(memw_w_constraints(&row).iter().all(|v| *v == FE::zero()));
    row[mw::MU_WRITE] = FE::one();
    assert!(memw_w_constraints(&row).iter().all(|v| *v == FE::zero()));
    row[mw::MU_ACCESS] = FE::one();
    assert_ne!(memw_w_constraints(&row)[2], FE::zero());
    for bad in [FE::from(2u64), FE::zero() - FE::one()] {
        for c in [mw::MU_WRITE, mw::MU_ACCESS] {
            let mut row = vec![FE::zero(); mw::NUM_COLUMNS];
            row[c] = bad;
            assert!(memw_w_constraints(&row).iter().any(|v| *v != FE::zero()));
        }
    }
}

/// MEMW_W's two message receives without their kind constants, as a control
/// mutant: the fingerprint has no length term.
fn memw_w_untagged() -> Vec<BusInteraction> {
    let mut out = memw_w::bus_interactions();
    out[4].values.remove(0);
    out[5].values.remove(0);
    out
}

/// ★ The message's kind constant is load-bearing: a chip's write of `n` taken
/// as an access with old `n` and new `(0, 0)` leaves memory at 0 after the
/// write. Refused on the message bus; control: the kind constants removed
/// from both the chip's and MEMW_W's tuples, it balances.
#[test]
fn the_message_kind_keeps_a_write_from_reading_as_an_access() {
    let mut m = Model::new(pages());
    // Word ELF_BASE holds n already; the chip writes n again at 6 and a read
    // at 9 sees the result.
    let n = m.current(ELF_BASE).value;
    let n2 = m.current(ELF_BASE + 8).value;
    m.write(ELF_BASE, 6, [n, n2]);
    m.read(ELF_BASE, 9);
    let mut bench = m.bench();
    let w = memw_w_row(&m, ELF_BASE, 6);
    bench.set_u64("MEMW_W", w, mw::MU_WRITE, 0);
    bench.set_u64("MEMW_W", w, mw::MU_ACCESS, 1);
    bench.set_u64("MEMW_W", w, mw::NEW_LO, 0);
    bench.set_u64("MEMW_W", w, mw::NEW_HI, 0);
    let r = memw_w_row(&m, ELF_BASE, 9);
    for c in [mw::OLD_LO, mw::OLD_HI, mw::NEW_LO, mw::NEW_HI] {
        bench.set_u64("MEMW_W", r, c, 0);
    }
    let reader = m.readers.iter().position(|r| r.ts == 9).expect("the read");
    bench.set_u64("READER", reader, chip::W[0], 0);
    bench.set_u64("READER", reader, chip::W[1], 0);
    let page = page_name(ELF_BASE);
    bench.set_u64(&page, 0, pw::FINI_LO, 0);
    bench.set_u64(&page, 0, pw::FINI_HI, 0);
    let v = verdict(&bench);
    assert!(v.refused_only_on(BusId::MemWordMsg), "{v:?}");

    let mut control = bench.with_interactions("MEMW_W", memw_w_untagged());
    for name in ["WRITER", "READER"] {
        for bi in &mut control.piece_mut(name).interactions {
            if bi.bus_id == u64::from(BusId::MemWordMsg) {
                bi.values.remove(0);
            }
        }
    }
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

// ---- the census ----

/// A value as a linear form: column → coefficient (`None`, the constant).
/// `Err` on a packing the word buses never use.
fn form(v: &BusValue) -> Result<BTreeMap<Option<usize>, i128>, String> {
    let mut out = BTreeMap::new();
    match v {
        BusValue::Packed {
            start_column,
            packing: Packing::Direct,
        } => {
            out.insert(Some(*start_column), 1);
        }
        BusValue::Packed { packing, .. } => return Err(format!("a packed value {packing:?}")),
        BusValue::Linear(terms) => {
            for t in terms {
                let (k, c) = match *t {
                    LinearTerm::Column {
                        coefficient,
                        column,
                    } => (Some(column), i128::from(coefficient)),
                    LinearTerm::ColumnUnsigned {
                        coefficient,
                        column,
                    } => (Some(column), i128::from(coefficient)),
                    LinearTerm::Constant(c) => (None, i128::from(c)),
                };
                *out.entry(k).or_insert(0) += c;
            }
        }
    }
    out.retain(|_, c| *c != 0);
    Ok(out)
}

fn multiplicity_form(m: &Multiplicity) -> BTreeMap<Option<usize>, i128> {
    let mut out = BTreeMap::new();
    let mut add = |k: Option<usize>, c: i128| *out.entry(k).or_insert(0) += c;
    match m {
        Multiplicity::One => add(None, 1),
        Multiplicity::Column(c) => add(Some(*c), 1),
        Multiplicity::Sum(a, b) => {
            add(Some(*a), 1);
            add(Some(*b), 1);
        }
        Multiplicity::Negated(c) => {
            add(None, 1);
            add(Some(*c), -1);
        }
        Multiplicity::Diff(a, b) => {
            add(Some(*a), 1);
            add(Some(*b), -1);
        }
        Multiplicity::Sum3(a, b, c) => {
            add(Some(*a), 1);
            add(Some(*b), 1);
            add(Some(*c), 1);
        }
        Multiplicity::Linear(ts) => {
            for t in ts {
                match *t {
                    LinearTerm::Column {
                        coefficient,
                        column,
                    } => add(Some(column), i128::from(coefficient)),
                    LinearTerm::ColumnUnsigned {
                        coefficient,
                        column,
                    } => add(Some(column), i128::from(coefficient)),
                    LinearTerm::Constant(c) => add(None, i128::from(c)),
                }
            }
        }
    }
    out.retain(|_, c| *c != 0);
    out
}

/// The columns a form reads.
fn columns(f: &BTreeMap<Option<usize>, i128>) -> BTreeSet<usize> {
    f.keys().filter_map(|k| *k).collect()
}

/// A table's constraints on one row and its width, for the census's bit
/// check: by name.
fn evaluator(name: &str) -> Option<(Eval, usize)> {
    name.starts_with("MEMW_W")
        .then_some((memw_w_constraints as Eval, mw::NUM_COLUMNS))
}

/// Rule c2: a word-bus tuple's address is two elements, never packed.
fn check_address(values: &[BusValue], what: &str) -> Result<(), String> {
    let lo = form(&values[0])?;
    let hi = form(&values[1])?;
    if lo
        .iter()
        .any(|(k, c)| k.is_some() && ![1, 8, 1 << 16].contains(c))
    {
        return Err(format!(
            "{what}: a low word {lo:?} scales a column past 2^16"
        ));
    }
    let hi_columns = columns(&hi);
    if hi_columns.len() > 1
        || hi.iter().any(|(k, c)| k.is_some() && *c != 1)
        || !columns(&lo).is_disjoint(&hi_columns)
    {
        return Err(format!(
            "{what}: a high word {hi:?} is not one column or a constant"
        ));
    }
    Ok(())
}

/// The census of one table's word-bus interactions (rules c2–c6, c9),
/// `preprocessed` its preprocessed prefix's width. `Ok` if the table touches
/// neither word bus.
fn census_table(
    name: &str,
    interactions: &[BusInteraction],
    preprocessed: usize,
) -> Result<(), String> {
    let on = |bus: BusId| -> Vec<&BusInteraction> {
        interactions
            .iter()
            .filter(|bi| bi.bus_id == u64::from(bus))
            .collect()
    };
    let tokens = on(BusId::MemWord);
    let messages = on(BusId::MemWordMsg);
    for bi in &tokens {
        if bi.values.len() != 5 {
            return Err(format!(
                "{name}: a word token of {} values",
                bi.values.len()
            ));
        }
        check_address(&bi.values, name)?;
    }
    let is_page = name.starts_with(page_w::NAME_PREFIX);
    if is_page {
        // c3: one mint at timestamp 0 and one final, at 1, at one key
        // `base + 8·OFFSET` with OFFSET preprocessed.
        let [mint, last] = tokens.as_slice() else {
            return Err(format!("{name}: {} word tokens, not 2", tokens.len()));
        };
        let offset = Some(pw::OFFSET);
        let lo = form(&mint.values[0])?;
        if mint.is_sender
            || !last.is_sender
            || mint.multiplicity != Multiplicity::One
            || last.multiplicity != Multiplicity::One
            || form(&mint.values[2])? != BTreeMap::new()
            || mint.values[..2] != last.values[..2]
            || columns(&lo) != BTreeSet::from([pw::OFFSET])
            || lo.get(&offset) != Some(&8)
            || !(0..preprocessed).contains(&pw::OFFSET)
            || !columns(&form(&mint.values[1])?).is_empty()
        {
            return Err(format!(
                "{name}: not a mint at base + 8·OFFSET, OFFSET preprocessed"
            ));
        }
        return Ok(());
    }
    if tokens.is_empty() && messages.iter().all(|bi| bi.is_sender) {
        // A sender of word messages: c9, its address from a register read or
        // `IS_HALF` limbs, plus a constant below 2^8.
        for bi in &messages {
            check_message_source(name, interactions, bi)?;
        }
        return Ok(());
    }
    // c4: produced and consumed tokens pair up, one key, one multiplicity.
    let produced: Vec<_> = tokens.iter().filter(|bi| !bi.is_sender).collect();
    let consumed: Vec<_> = tokens.iter().filter(|bi| bi.is_sender).collect();
    if produced.len() != consumed.len() {
        return Err(format!(
            "{name}: {} produced, {} consumed",
            produced.len(),
            consumed.len()
        ));
    }
    // The clock-bound receives: the word messages (and the CPU's dispatch).
    let clocked: Vec<&BusInteraction> = interactions
        .iter()
        .filter(|bi| {
            !bi.is_sender
                && (bi.bus_id == u64::from(BusId::MemWordMsg)
                    || bi.bus_id == u64::from(BusId::MemoryOp))
        })
        .collect();
    for (p, c) in produced.iter().zip(&consumed) {
        if p.values[..2] != c.values[..2] {
            return Err(format!("{name}: a pair at two keys"));
        }
        let mu = multiplicity_form(&p.multiplicity);
        if mu != multiplicity_form(&c.multiplicity) {
            return Err(format!("{name}: a pair at two multiplicities"));
        }
        // The produced timestamp is a clock-bound receive's, and the pair
        // fires exactly when one of them does.
        let ts = &p.values[2];
        let clocks: Vec<_> = clocked
            .iter()
            .filter(|bi| bi.values.get(3) == Some(ts))
            .collect();
        let mut fired = BTreeMap::new();
        for bi in &clocks {
            for (k, v) in multiplicity_form(&bi.multiplicity) {
                *fired.entry(k).or_insert(0) += v;
            }
            if bi.values.get(1..3) != Some(&p.values[..2]) {
                return Err(format!("{name}: a key that is not its message's"));
            }
        }
        if clocks.is_empty() || fired != mu {
            return Err(format!(
                "{name}: a produced timestamp no clock-bound receive carries"
            ));
        }
        // The consumed timestamp: produced − 1 − limb₀ − 2^16·limb₁, each limb
        // an `IS_HALF` at the pair's multiplicity.
        let produced_ts = form(ts)?;
        let consumed_ts = form(&c.values[2])?;
        let steps: Vec<(usize, i128)> = consumed_ts
            .iter()
            .filter_map(|(k, v)| {
                k.filter(|k| !produced_ts.contains_key(&Some(*k)))
                    .map(|k| (k, *v))
            })
            .collect();
        let mut expected = produced_ts.clone();
        *expected.entry(None).or_insert(0) -= 1;
        expected.extend(steps.iter().map(|&(k, v)| (Some(k), v)));
        expected.retain(|_, c| *c != 0);
        let mut coefficients: Vec<i128> = steps.iter().map(|&(_, v)| v).collect();
        coefficients.sort();
        if expected != consumed_ts || coefficients != [-(1 << 16), -1] {
            return Err(format!(
                "{name}: a consumed timestamp that is not a strict step"
            ));
        }
        for (limb, _) in steps {
            let ranged = interactions.iter().any(|bi| {
                bi.is_sender
                    && bi.bus_id == u64::from(BusId::IsHalfword)
                    && bi.values.len() == 1
                    && form(&bi.values[0]).ok() == Some(BTreeMap::from([(Some(limb), 1)]))
                    && multiplicity_form(&bi.multiplicity) == mu
            });
            if !ranged {
                return Err(format!("{name}: a step limb no IS_HALF holds"));
            }
        }
        // c5: the multiplicity is held to a bit by a constraint.
        let Some((eval, width)) = evaluator(name) else {
            return Err(format!(
                "{name}: no constraint holds its multiplicity to a bit"
            ));
        };
        for c in columns(&mu) {
            for bad in [FE::from(2u64), FE::zero() - FE::one()] {
                let mut row = vec![FE::zero(); width];
                row[c] = bad;
                if eval(&row).iter().all(|v| *v == FE::zero()) {
                    return Err(format!(
                        "{name}: column {c} at {bad:?} breaks no constraint"
                    ));
                }
            }
        }
        let mut row = vec![FE::zero(); width];
        for c in columns(&mu) {
            row[c] = FE::one();
        }
        if columns(&mu).len() > 1 && eval(&row).iter().all(|v| *v == FE::zero()) {
            return Err(format!("{name}: two messages at once break no constraint"));
        }
    }
    // c6: only a table that passes tokens through receives word messages, in
    // the two tagged shapes.
    for bi in &messages {
        if bi.is_sender {
            return Err(format!("{name}: sends a word message and holds tokens"));
        }
        let tag = form(&bi.values[0])?;
        let shape = match tag.get(&None).copied().unwrap_or(0) {
            0 if columns(&tag).is_empty() => 8,
            1 if columns(&tag).is_empty() => 6,
            _ => return Err(format!("{name}: a word message with no kind constant")),
        };
        if bi.values.len() != shape {
            return Err(format!(
                "{name}: a word message of {} values",
                bi.values.len()
            ));
        }
        check_address(&bi.values[1..3], name)?;
    }
    Ok(())
}

/// Rule c9: a word message's address is a register value or `IS_HALF` limbs
/// read in the same row, plus a constant below 2^8.
fn check_message_source(
    name: &str,
    interactions: &[BusInteraction],
    message: &BusInteraction,
) -> Result<(), String> {
    check_address(&message.values[1..3], name)?;
    // Columns read as a register's value: a `Memw` tuple whose `is_register`
    // is the constant 1 (position 0 of a write, 8 of a read), or the `Reg` bus.
    let one = BTreeMap::from([(None, 1i128)]);
    let mut registered = BTreeSet::new();
    for bi in interactions {
        let register = (bi.bus_id == u64::from(BusId::Memw)
            && [0usize, 8]
                .iter()
                .any(|&i| bi.values.get(i).and_then(|v| form(v).ok()) == Some(one.clone())))
            || bi.bus_id == u64::from(BusId::Reg);
        if register {
            for v in &bi.values {
                if let Ok(f) = form(v)
                    && f.len() == 1
                    && let Some((Some(c), 1)) = f.iter().next().map(|(k, v)| (*k, *v))
                {
                    registered.insert(c);
                }
            }
        }
    }
    let halves: BTreeSet<usize> = interactions
        .iter()
        .filter(|bi| bi.bus_id == u64::from(BusId::IsHalfword) && bi.values.len() == 1)
        .filter_map(|bi| form(&bi.values[0]).ok())
        .filter(|f| f.len() == 1)
        .filter_map(|f| {
            f.into_iter()
                .next()
                .and_then(|(k, c)| (c == 1).then_some(k?))
        })
        .collect();
    for (i, v) in message.values[1..3].iter().enumerate() {
        let f = form(v)?;
        let constant = f.get(&None).copied().unwrap_or(0);
        let cols = columns(&f);
        let from_register = cols.len() == 1 && cols.iter().all(|c| registered.contains(c));
        let from_limbs = cols.len() == 2
            && cols.iter().all(|c| halves.contains(c))
            && f.values().filter(|&&c| c == 1 << 16).count() == 1;
        if !(0..256).contains(&constant) || !(from_register || from_limbs || cols.is_empty()) {
            return Err(format!(
                "{name}: a word message's address element {i} ({f:?}) has no canonical source"
            ));
        }
    }
    Ok(())
}

/// Every table count 0.
fn zero_counts() -> crate::TableCounts {
    crate::TableCounts {
        cpu: 0,
        lt: 0,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 0,
        dvrm: 0,
        shift: 0,
        branch: 0,
        memw_register: 0,
        eq: 0,
        bytewise: 0,
        store: 0,
        cpu32: 0,
        keccak: 0,
        keccak_rnd: 0,
        ecsm: 0,
        ecdas: 0,
        hint: 0,
        copy: 0,
        blake3: 0,
        is_wh: 0,
        ecrec: 0,
    }
}

/// The block's AIR set under the word memory format, as the verifier will
/// build it: the split set with no LOAD, STORE, MEMW or MEMW_A table, PAGE_W
/// in place of PAGE, and `memw_w` MEMW_W tables.
fn word_format_census(registers: crate::tables::register::RegisterFormat) -> Result<(), String> {
    let elf = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let opts = ProofOptions::default_test_options();
    let counts = crate::TableCounts {
        cpu: 1,
        lt: 1,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 1,
        dvrm: 1,
        shift: 1,
        branch: 1,
        memw_register: 1,
        eq: 1,
        bytewise: 1,
        store: 0,
        cpu32: 1,
        keccak: 1,
        keccak_rnd: 1,
        ecsm: 1,
        ecdas: 1,
        hint: 1,
        copy: 1,
        blake3: 1,
        is_wh: 1,
        ecrec: 1,
    };
    block_whir::word_tables_under(MemoryFormat::Words, &counts, [0, 0])
        .map_err(|e| format!("{e:?}"))?;
    let mut pages = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &program,
        &[],
        1,
        1 << 20,
    )
    .expect("pages");
    pages.push(PageConfig::zero_init(STACK_BASE));
    let inputs =
        crate::BlockAirInputs::new(&opts, &program, vec![1 << 10], &[]).expect("the AIR inputs");
    let pinned = block_whir::BlockFormat::pinned();
    let airs = crate::block_airs(
        &program,
        &opts,
        &pages,
        &counts,
        &inputs,
        pinned.keccak_rnd,
        pinned.memw_aligned,
        registers,
    );
    let words: Vec<crate::test_utils::VmAir> = pages
        .iter()
        .map(|config| crate::test_utils::create_page_w_air(&opts, config))
        .chain(std::iter::once(Ok(Box::new(
            crate::test_utils::create_memw_w_air(&opts).with_name("MEMW_W[0]"),
        ) as crate::test_utils::VmAir)))
        .collect::<Result<_, _>>()?;
    let mut set: Vec<
        &dyn stark::traits::AIR<
            Field = GoldilocksField,
            FieldExtension = GoldilocksExtension,
            PublicInputs = (),
        >,
    > = airs
        .air_refs()
        .into_iter()
        .filter(|air| !air.name().starts_with("PAGE:"))
        .collect();
    set.extend(words.iter().map(|air| air.as_ref()));
    let mut on_words = BTreeSet::new();
    let mut message_receivers = BTreeSet::new();
    for air in &set {
        let name = air.name();
        let interactions = air.bus_interactions();
        // c8: no cross-epoch table.
        if name.starts_with("LOCAL_TO_GLOBAL") || name.starts_with("GLOBAL_MEMORY") {
            return Err(format!("{name}: a cross-epoch table under the word format"));
        }
        // c7: no RAM token on the byte bus.
        let one = BTreeMap::from([(None, 1i128)]);
        if interactions.iter().any(|bi| {
            bi.bus_id == u64::from(BusId::Memory)
                && bi.values.first().and_then(|v| form(v).ok()) != Some(one.clone())
        }) {
            return Err(format!("{name}: a RAM token on the byte bus"));
        }
        let touches = |bus: BusId| interactions.iter().any(|bi| bi.bus_id == u64::from(bus));
        if touches(BusId::MemWord) {
            on_words.insert(name.split(['[', ':']).next().unwrap_or("").to_string());
        }
        if interactions
            .iter()
            .any(|bi| bi.bus_id == u64::from(BusId::MemWordMsg) && !bi.is_sender)
        {
            message_receivers.insert(name.split('[').next().unwrap_or("").to_string());
        }
        census_table(name, interactions, air.num_precomputed_columns())?;
    }
    // c1 and c6: the word bus is PAGE_W's and MEMW_W's alone, and MEMW_W
    // alone receives word messages.
    if on_words != BTreeSet::from(["PAGE_W".to_string(), "MEMW_W".to_string()]) {
        return Err(format!("the word bus is touched by {on_words:?}"));
    }
    if message_receivers != BTreeSet::from(["MEMW_W".to_string()]) {
        return Err(format!(
            "word messages are received by {message_receivers:?}"
        ));
    }
    Ok(())
}

/// ★ The census of the word format's AIR set, under both register formats:
/// only PAGE_W mints, at `base + 8·OFFSET` with `OFFSET` preprocessed; every
/// other token pair shares its key and multiplicity, the multiplicity a bit
/// and its produced timestamp a clock-bound receive's; every address is two
/// elements; MEMW_W alone takes word messages, tagged; no RAM token rides the
/// byte bus; no cross-epoch table.
#[test]
fn the_word_format_air_set_passes_the_census() {
    for registers in [
        crate::tables::register::RegisterFormat::InRow,
        crate::tables::register::RegisterFormat::Tokens32,
    ] {
        word_format_census(registers).unwrap_or_else(|e| panic!("{registers:?}: {e}"));
    }
}

/// ★ The census refuses each mutant it exists for: a packed address, an
/// unequal pair, a step without its `−1`, a PAGE_W with a main `OFFSET`, an
/// untagged message; and it reads the stand-in chips' addresses as register
/// values plus a small constant, refusing a chip whose pointer is free.
#[test]
fn the_census_refuses_each_mutant() {
    let page = page_w::bus_interactions(0, false).expect("aligned");
    assert!(census_table("PAGE_W:0x0", &page, page_w::NUM_PREPROCESSED_COLS).is_ok());
    assert!(
        census_table("PAGE_W:0x0", &page, 0).is_err(),
        "OFFSET not preprocessed"
    );
    assert!(
        census_table("PAGE_W:0x0", &packed(page.clone()), 1).is_err(),
        "packed"
    );
    let memw = memw_w::bus_interactions();
    assert!(census_table("MEMW_W", &memw, 0).is_ok());
    for (what, mutant) in [
        ("packed", packed(memw.clone())),
        ("unequal pair", memw_w_unequal_pair()),
        ("no −1", memw_w_without_the_minus_one()),
        ("untagged", memw_w_untagged()),
    ] {
        assert!(census_table("MEMW_W", &mutant, 0).is_err(), "{what}");
    }
    // The limbs' IS_HALF taken out.
    let mut unranged = memw.clone();
    unranged.remove(1);
    assert!(
        census_table("MEMW_W", &unranged, 0).is_err(),
        "a limb unranged"
    );

    // c9 on the stand-ins: their address is x12's value plus 0 or 8.
    for kind in [MessageKind::Write, MessageKind::Access] {
        let chip = chip_interactions(kind);
        assert!(census_table("CHIP", &chip, 0).is_ok(), "{kind:?}");
        // The register read dropped: the pointer is a free column.
        let free: Vec<_> = chip.into_iter().skip(1).collect();
        assert!(
            census_table("CHIP", &free, 0).is_err(),
            "{kind:?} free pointer"
        );
    }
}

// ---- the format ----

/// ★ The word memory format is refused on every block path until it is built:
/// the format check every prove and verify runs, the fused tables' check every
/// lay-out runs, and the windowed build's setters.
#[test]
fn the_word_format_is_refused_on_every_block_path() {
    let words = block_whir::BlockFormat::pinned().with_memory(MemoryFormat::Words);
    assert!(block_whir::check_memory_format(&words).is_err());
    assert!(crate::tables::fused_memory::refuse_unbuilt(MemoryFormat::Words).is_err());
    for stated in [[0, 0], [1, 0], [0, 1]] {
        assert!(block_whir::fused_tables_under(MemoryFormat::Words, stated).is_err());
    }
    for memory in [MemoryFormat::Split, MemoryFormat::Fused] {
        assert!(crate::tables::fused_memory::refuse_unbuilt(memory).is_ok());
    }
    // The windowed build and its regenerator refuse it at their setters.
    use crate::tables::trace_builder::{RegenBuilder, WindowedTraceBuilder};
    let elf = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let max_rows = crate::tables::MaxRowsConfig::small();
    let builder = WindowedTraceBuilder::new(&program, &[], &max_rows).expect("a builder");
    assert!(builder.memory_format(MemoryFormat::Words).is_err());
    let regen = RegenBuilder::new(&program, &[], &max_rows).expect("a regenerator");
    assert!(regen.memory_format(MemoryFormat::Words).is_err());
    // Its statement word is its own.
    assert_eq!(
        block_whir::fused_statement_words(MemoryFormat::Words, [0, 0]),
        vec![block_whir::WORD_MEMORY_WORD]
    );
    assert_ne!(block_whir::WORD_MEMORY_WORD, block_whir::FUSED_MEMORY_WORD);
}

/// ★ Under the word format every table it retires is refused by count:
/// LOAD, STORE, MEMW, MEMW_A, MEM_LOAD, MEM_STORE.
/// Under the split and fused formats the check reads nothing.
#[test]
fn the_word_format_refuses_every_retired_table() {
    let zero = zero_counts();
    assert!(block_whir::word_tables_under(MemoryFormat::Words, &zero, [0, 0]).is_ok());
    let each: [(&str, crate::TableCounts, [u64; 2]); 6] = [
        (
            "LOAD",
            crate::TableCounts {
                load: 1,
                ..zero.clone()
            },
            [0, 0],
        ),
        (
            "STORE",
            crate::TableCounts {
                store: 1,
                ..zero.clone()
            },
            [0, 0],
        ),
        (
            "MEMW",
            crate::TableCounts {
                memw: 1,
                ..zero.clone()
            },
            [0, 0],
        ),
        (
            "MEMW_A",
            crate::TableCounts {
                memw_aligned: 1,
                ..zero.clone()
            },
            [0, 0],
        ),
        ("MEM_LOAD", zero.clone(), [1, 0]),
        ("MEM_STORE", zero.clone(), [0, 1]),
    ];
    for (table, counts, fused) in each {
        let refused = block_whir::word_tables_under(MemoryFormat::Words, &counts, fused);
        assert!(
            matches!(&refused, Err(crate::Error::InvalidTableCounts(why)) if why.contains(&format!("no {table} table"))),
            "{table}: {refused:?}"
        );
        for memory in [MemoryFormat::Split, MemoryFormat::Fused] {
            assert!(block_whir::word_tables_under(memory, &counts, fused).is_ok());
        }
    }
}

/// ★ A PAGE_W table is held to its one height: a taller
/// one would mint the next page's words a second time.
#[test]
fn a_page_w_table_is_held_to_its_height() {
    let n = page_w::NUM_VARS as u8;
    assert!(block_whir::check_page_w_heights([("PAGE_W:0x0", n), ("CPU[0]", 20)]).is_ok());
    for bad in [n - 1, n + 1] {
        assert!(block_whir::check_page_w_heights([("PAGE_W:0x40000", bad)]).is_err());
    }
    // PAGE is not PAGE_W.
    assert!(block_whir::check_page_w_heights([("PAGE:0x0", 18)]).is_ok());
}

/// Whether `interactions` touch either word bus.
fn on_word_buses(interactions: &[BusInteraction]) -> bool {
    interactions.iter().any(|bi| {
        bi.bus_id == u64::from(BusId::MemWord) || bi.bus_id == u64::from(BusId::MemWordMsg)
    })
}

/// ★ Buses 40 and 41 are the word tables' alone: `BusId` is the one registry
/// of bus ids, the two ids decode to the word buses, and no table of the
/// continuation path (the cross-epoch tables) or of the recursion's programs
/// (every hasher's AIR set) touches them.
#[test]
fn no_other_table_or_program_uses_the_word_buses() {
    assert_eq!(BusId::try_from(40u64), Ok(BusId::MemWord));
    assert_eq!(BusId::try_from(41u64), Ok(BusId::MemWordMsg));
    assert_eq!(u64::from(BusId::MemWord), 40);
    assert_eq!(u64::from(BusId::MemWordMsg), 41);
    use crate::tables::{global_memory, local_to_global};
    for label in [
        local_to_global::epoch_label(0),
        local_to_global::epoch_label(7),
    ] {
        assert!(!on_word_buses(&local_to_global::bus_interactions(label)));
        assert!(!on_word_buses(&local_to_global::range_check_interactions(
            label
        )));
    }
    assert!(!on_word_buses(&local_to_global::memory_bus_interactions()));
    assert!(!on_word_buses(&global_memory::bus_interactions(0)));
    let opts = ProofOptions::default_test_options();
    use crate::lfm::airs::ChipSet;
    use crate::lfm::hash::HasherKind;
    for hasher in [
        HasherKind::Test,
        HasherKind::Blake3,
        HasherKind::Rpx,
        HasherKind::Poseidon1W16,
    ] {
        let airs = crate::lfm::wlfm::whir_lfm_airs(hasher, ChipSet::FULL, 2, &opts);
        for air in airs.air_refs() {
            assert!(
                !on_word_buses(air.bus_interactions()),
                "{hasher:?}: {} touches a word bus",
                air.name()
            );
        }
    }
}

/// ★ The split and fused AIR sets carry no word table and no word bus, so the
/// formats a block proves today are what they were.
#[test]
fn the_split_and_fused_sets_carry_no_word_table() {
    let elf = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let opts = ProofOptions::default_test_options();
    let counts = crate::TableCounts {
        cpu: 1,
        memw: 1,
        memw_aligned: 1,
        load: 1,
        store: 1,
        copy: 1,
        hint: 1,
        ecrec: 1,
        ..zero_counts()
    };
    let pages = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &program,
        &[],
        0,
        1 << 20,
    )
    .expect("pages");
    let inputs =
        crate::BlockAirInputs::new(&opts, &program, vec![1 << 10], &[]).expect("the AIR inputs");
    let pinned = block_whir::BlockFormat::pinned();
    for (fused, registers) in [[0, 0], [1, 1]].into_iter().flat_map(|f| {
        [
            crate::tables::register::RegisterFormat::InRow,
            crate::tables::register::RegisterFormat::Tokens32,
        ]
        .map(|r| (f, r))
    }) {
        let mut airs = crate::block_airs(
            &program,
            &opts,
            &pages,
            &counts,
            &inputs,
            pinned.keccak_rnd,
            pinned.memw_aligned,
            registers,
        );
        airs.with_fused_memory(&opts, fused);
        for air in airs.air_refs() {
            assert!(!air.name().starts_with(page_w::NAME_PREFIX));
            assert!(!air.name().starts_with("MEMW_W"));
            assert!(
                air.bus_interactions()
                    .iter()
                    .all(|bi| bi.bus_id != u64::from(BusId::MemWord)
                        && bi.bus_id != u64::from(BusId::MemWordMsg))
            );
        }
    }
}

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
//! The stand-in chips are held as a real chip is: each reads its pointer from
//! register x12 as HINT does (`hint.rs`' register read), takes its call on the
//! `Ecall` bus at the call's timestamp, accesses two words at `pointer + 8i`
//! with messages at that timestamp and multiplicity (a bit), and a writer
//! range-checks the 16-bit halves of the words it writes. The stand-in
//! register file answers a register read only with the register's true value
//! (its value columns are the verifier's), and the stand-in clock's calls are
//! the verifier's, though a forger may leave a read unanswered. A forgery is a malicious prover's edit of an honest
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
    access_message, checked_word, column, halves, step_less_one, step_limbs, term, write_message,
};
use crate::block_whir;
use crate::tables::fused_memory::MemoryFormat;

pub(super) type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

/// A table's constraints evaluated on one row.
pub(super) type Eval = fn(&[FE]) -> Vec<FE>;

/// The field's order: `−k` reads as `P − k`.
pub(super) const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The register the stand-in chips read their pointer from (HINT's `out`).
const CHIP_REG: u64 = 12;

// ---- the bench ----

/// One table of a bench.
#[derive(Clone)]
pub(super) struct Piece {
    pub(super) name: String,
    pub(super) interactions: Vec<BusInteraction>,
    pub(super) trace: Table,
    /// The verifier's own values of the table's preprocessed prefix.
    pub(super) preprocessed: Vec<Vec<FE>>,
    /// The table's constraints on one row.
    pub(super) constraints: Option<Eval>,
}

#[derive(Clone, Default)]
pub(super) struct Bench {
    pub(super) pieces: Vec<Piece>,
}

impl Bench {
    pub(super) fn piece(&self, name: &str) -> &Piece {
        self.pieces
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the bench holds {name}"))
    }

    pub(super) fn piece_mut(&mut self, name: &str) -> &mut Piece {
        self.pieces
            .iter_mut()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the bench holds {name}"))
    }

    pub(super) fn get(&self, name: &str, row: usize, col: usize) -> FE {
        *self.piece(name).trace.get_main(row, col)
    }

    pub(super) fn set(&mut self, name: &str, row: usize, col: usize, value: FE) {
        self.piece_mut(name).trace.main_table.set(row, col, value);
    }

    pub(super) fn set_u64(&mut self, name: &str, row: usize, col: usize, value: u64) {
        self.set(name, row, col, FE::from(value));
    }

    /// `name`'s interactions replaced by `interactions`: a control mutant.
    pub(super) fn with_interactions(
        mut self,
        name: &str,
        interactions: Vec<BusInteraction>,
    ) -> Self {
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
pub(super) struct Verdict {
    /// Per bus, the tuples whose net is not zero (trailing zeros stripped).
    pub(super) left: BTreeMap<u64, Vec<(Vec<u64>, i128)>>,
    /// Lookups no BITWISE row answers: `(bus, tuple)`.
    pub(super) unanswered: Vec<(u64, Vec<u64>)>,
    /// `(table, row, constraint)` that does not read zero.
    pub(super) broken: Vec<(String, usize, usize)>,
    /// `(table, row, column)` whose preprocessed value is not the verifier's.
    pub(super) preprocessed: Vec<(String, usize, usize)>,
}

impl Verdict {
    pub(super) fn accepted(&self) -> bool {
        self.left.is_empty()
            && self.unanswered.is_empty()
            && self.broken.is_empty()
            && self.preprocessed.is_empty()
    }

    pub(super) fn left_on(&self, bus: BusId) -> &[(Vec<u64>, i128)] {
        self.left
            .get(&u64::from(bus))
            .map_or(&[][..], |v| v.as_slice())
    }

    /// Only `bus` is unbalanced, and nothing else is refused.
    pub(super) fn refused_only_on(&self, bus: BusId) -> bool {
        self.left.keys().all(|&b| b == u64::from(bus))
            && !self.left_on(bus).is_empty()
            && self.unanswered.is_empty()
            && self.broken.is_empty()
            && self.preprocessed.is_empty()
    }
}

pub(super) fn multiplicity(m: &Multiplicity, g: &dyn Fn(usize) -> FE) -> FE {
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
    } else if bus == u64::from(BusId::Msb8) {
        // `(byte, its top bit)`.
        tuple.len() <= 2 && at(0) < 256 && at(1) == at(0) >> 7
    } else {
        false
    }
}

const LOOKUPS: [BusId; 3] = [BusId::IsHalfword, BusId::AreBytes, BusId::Msb8];

fn signed(v: FE) -> i128 {
    let v = v.canonical_u64();
    if v > u64::MAX / 2 {
        -((FE::zero() - FE::from(v)).canonical_u64() as i128)
    } else {
        v as i128
    }
}

pub(super) fn verdict(bench: &Bench) -> Verdict {
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
pub(super) fn constraint_values<CS: ConstraintSet<GoldilocksField, GoldilocksExtension>>(
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
pub(super) fn table(rows: usize, width: usize, fill: impl Fn(&mut Table, usize)) -> Table {
    let num_rows = rows.next_power_of_two().max(4);
    let mut t = TraceTable::new_main(vec![FE::zero(); num_rows * width], width, 1);
    for row in 0..rows {
        fill(&mut t, row);
    }
    t
}

/// `t` with `extra` zero columns appended.
pub(super) fn widen(t: &Table, extra: usize) -> Table {
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
    /// Word `i`'s four 16-bit halves, low first, at `H[4i..4i + 4]`.
    pub const H: [usize; 8] = [4, 5, 6, 7, 8, 9, 10, 11];
    pub const NUM_COLUMNS: usize = 12;
}

/// The call number the stand-in chips answer on the `Ecall` bus.
const CHIP_CALL: u64 = 0x5151;

/// The stand-in clock's columns, all the verifier's: a call's timestamp and
/// its multiplicity. The CPU sends each ECALL once, at its cycle's timestamp.
mod clock {
    pub const TS: usize = 0;
    pub const MU: usize = 1;
    pub const NUM_COLUMNS: usize = 2;
}

/// The stand-in clock's `Ecall` send `(TS, 0, CHIP_CALL, 0)`, the CPU's.
fn clock_interactions() -> Vec<BusInteraction> {
    vec![BusInteraction::sender(
        BusId::Ecall,
        Multiplicity::Column(clock::MU),
        vec![
            column(clock::TS),
            BusValue::constant(0),
            BusValue::constant(CHIP_CALL),
            BusValue::constant(0),
        ],
    )]
}

/// Word `i` of a stand-in chip as its two 32-bit halves, each two 16-bit
/// columns.
fn chip_word(i: usize) -> [BusValue; 2] {
    let h = |k: usize| chip::H[4 * i + k];
    [
        BusValue::linear(vec![term(1, h(0)), term(1 << 16, h(1))]),
        BusValue::linear(vec![term(1, h(2)), term(1 << 16, h(3))]),
    ]
}

/// The stand-in chips' constraint: `MU` a bit.
fn chip_constraints(row: &[FE]) -> Vec<FE> {
    vec![row[chip::MU] * (FE::one() - row[chip::MU])]
}

/// Sets word `i` of `piece`'s chip row `row` to `(lo, hi)` (each up to 48
/// bits: a forger's half past 32 bits sits in its high 16-bit column).
fn set_chip_word(bench: &mut Bench, piece: &str, row: usize, i: usize, lo: u64, hi: u64) {
    for (k, half) in [lo & 0xFFFF, lo >> 16, hi & 0xFFFF, hi >> 16]
        .into_iter()
        .enumerate()
    {
        bench.set_u64(piece, row, chip::H[4 * i + k], half);
    }
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

/// A stand-in chip, as a real one is held (the census's sender rules): it
/// reads its pointer from x12, takes its call on the `Ecall` bus at the
/// call's timestamp, and writes (or reads) two words at `pointer + 8i`, one
/// message a word at the call's timestamp and multiplicity; a writer checks
/// the 16-bit halves of each word it writes, a reader sends old = new.
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
        let message = match kind {
            MessageKind::Write => write_message(address, column(chip::TS), chip_word(i)),
            MessageKind::Access => {
                access_message(address, column(chip::TS), chip_word(i), chip_word(i))
            }
        };
        out.push(BusInteraction::sender(BusId::MemWordMsg, mu(), message));
    }
    out.push(BusInteraction::receiver(
        BusId::Ecall,
        mu(),
        vec![
            column(chip::TS),
            BusValue::constant(0),
            BusValue::constant(CHIP_CALL),
            BusValue::constant(0),
        ],
    ));
    if kind == MessageKind::Write {
        out.extend(
            chip::H.map(|h| BusInteraction::sender(BusId::IsHalfword, mu(), vec![column(h)])),
        );
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
pub(super) fn pages() -> Vec<PageConfig> {
    let elf: Vec<u8> = (0..4096u32).map(|i| (i * 37 + 11) as u8).collect();
    let input: Vec<u8> = (0..4096u32).map(|i| (i * 101 + 3) as u8).collect();
    vec![
        PageConfig::zero_init(0),
        PageConfig::with_data(ELF_BASE, elf),
        PageConfig::with_private_input(executor::vm::memory::PRIVATE_INPUT_START_INDEX, input),
        PageConfig::zero_init(STACK_BASE),
    ]
}

pub(super) const ELF_BASE: u64 = 0x40000;
pub(super) const STACK_BASE: u64 = 0xFFFF_FFFF_FFFC_0000;
pub(super) const INPUT_BASE: u64 = executor::vm::memory::PRIVATE_INPUT_START_INDEX;

/// Words in memory as the honest accesses leave them, and the rows they make.
struct Model {
    pages: Vec<PageConfig>,
    words: FinalWordMap,
    memw_w: Vec<MemwWordRow>,
    writers: Vec<ChipRow>,
    readers: Vec<ChipRow>,
    regs: Vec<RegRow>,
    /// The calls' timestamps, as the clock has them.
    calls: Vec<u64>,
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
            calls: Vec::new(),
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
        self.calls.push(ts);
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
        self.calls.push(ts);
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
                        for k in 0..4 {
                            let half = (w >> (16 * k)) & 0xFFFF;
                            t.main_table.set(row, chip::H[4 * i + k], FE::from(half));
                        }
                    }
                }),
                preprocessed: Vec::new(),
                constraints: Some(chip_constraints),
            });
        }
        // The clock is the verifier's: a forger may not move a call.
        let calls = table(self.calls.len(), clock::NUM_COLUMNS, |t, row| {
            t.main_table.set(row, clock::TS, FE::from(self.calls[row]));
            t.main_table.set(row, clock::MU, FE::one());
        });
        let truth = (0..clock::NUM_COLUMNS)
            .map(|c| {
                (0..calls.num_rows())
                    .map(|r| *calls.get_main(r, c))
                    .collect()
            })
            .collect();
        pieces.push(Piece {
            name: "CLOCK".into(),
            interactions: clock_interactions(),
            trace: calls,
            preprocessed: truth,
            constraints: None,
        });
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

pub(super) fn page_name(base: u64) -> String {
    format!("{}{base:#x}", page_w::NAME_PREFIX)
}

pub(super) fn page_piece(config: &PageConfig, words: &FinalWordMap) -> Piece {
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

/// ★ A private-input page's initial word is canonical, each of its four
/// 16-bit halves by its own `IS_HALF`: one half moved past 16 bits (`INIT_0`
/// or `INIT_2` by `2^32`, `INIT_1` or `INIT_3` by `2^16`) makes a 32-bit half of
/// `2^32 + x`, carried through a chip's read to the page's final word, and is
/// refused by that half's lookup alone. Control: that lookup removed, the chip
/// reads a word that is not one.
#[test]
fn a_private_init_word_past_32_bits_is_refused_by_its_half_check() {
    let (m, bench) = honest();
    let name = page_name(INPUT_BASE);
    let word = 0x10 / 8;
    let row = memw_w_row(&m, INPUT_BASE + 0x10, 24);
    let reader = m.readers.iter().position(|r| r.ts == 24).expect("the read");
    let read = m.readers[reader].words[0];
    for k in 0..4 {
        let mut forged = bench.clone();
        // The half `k` past 16 bits: its 32-bit half `2^32` more.
        let (delta, upper) = match k {
            0 | 2 => (FE::from(1u64 << 32), k == 2),
            _ => (FE::from(1u64 << 16), k == 3),
        };
        let h = forged.get(&name, word, pw::INIT[k]) + delta;
        forged.set(&name, word, pw::INIT[k], h);
        let (fini, old, new) = if upper {
            (pw::FINI_HI, mw::OLD_HI, mw::NEW_HI)
        } else {
            (pw::FINI_LO, mw::OLD_LO, mw::NEW_LO)
        };
        let half = forged.get("MEMW_W", row, old) + FE::from(1u64 << 32);
        forged.set(&name, word, fini, half);
        forged.set("MEMW_W", row, old, half);
        forged.set("MEMW_W", row, new, half);
        let (lo, hi) = (read & 0xFFFF_FFFF, read >> 32);
        let (lo, hi) = if upper {
            (lo, hi + (1 << 32))
        } else {
            (lo + (1 << 32), hi)
        };
        set_chip_word(&mut forged, "READER", reader, 0, lo, hi);
        let v = verdict(&forged);
        assert_eq!(
            v.unanswered,
            vec![(u64::from(BusId::IsHalfword), vec![h.canonical_u64()])],
            "INIT_{k}: {v:?}"
        );
        assert!(v.left.is_empty() && v.broken.is_empty() && v.preprocessed.is_empty());

        let mut control = page_w::bus_interactions(INPUT_BASE, true).expect("aligned");
        control.remove(2 + k);
        let v = verdict(&forged.with_interactions(&name, control));
        assert!(v.accepted(), "INIT_{k} control: {v:?}");
    }
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
    // The clock's truth moves with it: the test posits two calls at 8, as one
    // call that touches a word twice at its timestamp would.
    let call = m.calls.iter().position(|&t| t == 9).expect("the call at 9");
    forged.piece_mut("CLOCK").preprocessed[clock::TS][call] = FE::from(8u64);
    forged.set_u64("CLOCK", call, clock::TS, 8);
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
        m.calls.push(ts);
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

/// ★ The byte gadget ([`checked_word`]): eight bytes summed into a word's
/// halves, four pair checks on the same columns. For each pair, a forger
/// keeping the word moves its first byte to `256 + x` and its second down one
/// (a lie about the second byte); that pair's check refuses it, and nothing
/// else. Control: that pair removed.
#[test]
fn a_byte_past_255_is_refused_by_its_pair() {
    const BYTES: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    const MU: usize = 8;
    let word: u64 = 0x8070_6050_4030_2010;
    let bytes = word.to_le_bytes();
    let (halves_of, checks) = checked_word(&BYTES, Multiplicity::Column(MU));
    assert_eq!(checks.len(), 4, "every byte in a pair");
    for j in 0..4usize {
        // A table exposing the word's bytes and naming byte `2j + 1`, and one
        // taking the word and that byte: on a bus of the test's own.
        let bus = 1000u64;
        let mut split = checks.clone();
        split.push(BusInteraction::sender(
            bus,
            Multiplicity::Column(MU),
            vec![
                halves_of[0].clone(),
                halves_of[1].clone(),
                column(BYTES[2 * j + 1]),
            ],
        ));
        let take = BusInteraction::receiver(
            bus,
            Multiplicity::One,
            vec![
                BusValue::constant(word & 0xFFFF_FFFF),
                BusValue::constant(word >> 32),
                column(0),
            ],
        );
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
                    trace: TraceTable::new_main(vec![FE::from(u64::from(bytes[2 * j + 1]))], 1, 1),
                    preprocessed: Vec::new(),
                    constraints: None,
                },
            ],
        };
        assert!(verdict(&honest).accepted(), "pair {j}");

        // Byte 2j + 256 and byte 2j + 1 − 1: the word is unchanged, the second
        // byte is a lie.
        let (first, second) = (u64::from(bytes[2 * j]), u64::from(bytes[2 * j + 1]));
        let mut forged = honest.clone();
        forged.set_u64("SPLIT", 0, BYTES[2 * j], first + 256);
        forged.set_u64("SPLIT", 0, BYTES[2 * j + 1], second - 1);
        forged.set_u64("TAKE", 0, 0, second - 1);
        let v = verdict(&forged);
        assert_eq!(
            v.unanswered,
            vec![(u64::from(BusId::AreBytes), vec![first + 256, second - 1])],
            "pair {j}: {v:?}"
        );
        assert!(v.left.is_empty(), "pair {j}: {v:?}");
        let mut control = split;
        control.remove(j);
        let v = verdict(&forged.with_interactions("SPLIT", control));
        assert!(v.accepted(), "pair {j} control: {v:?}");
    }
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
    let base = forged.clone();
    // The attempt: a MEMW_W row reads RAM word 24 at 8. Its receive is on the
    // word-message bus, so the chip's register read stays open on `Memw`,
    // and the row's own message is sent by no one; the word bus balances.
    let row = rows_used(&forged);
    for (c, value) in [
        (mw::ADDRESS_LO, 24),
        (mw::TIMESTAMP, 8),
        (mw::OLD_LO, 0x300),
        (mw::NEW_LO, 0x300),
        (mw::STEP_0, 8 - 2 - 1),
        (mw::MU_ACCESS, 1),
    ] {
        forged.set_u64("MEMW_W", row, c, value);
    }
    forged.set_u64(&page_name(0), 24 / 8, pw::TIMESTAMP, 8);
    let v = verdict(&forged);
    assert_eq!(
        v.left.keys().copied().collect::<Vec<_>>(),
        vec![u64::from(BusId::Memw), u64::from(BusId::MemWordMsg)],
        "{v:?}"
    );
    assert!(
        v.left_on(BusId::MemWord).is_empty()
            && v.unanswered.is_empty()
            && v.broken.is_empty()
            && v.preprocessed.is_empty(),
        "{v:?}"
    );

    // Control: MEMW_W receiving `Memw`, one more row reads RAM word 24 at 8
    // for the chip's register read; word 24's last token moves from 2 to 8.
    let mut control = base.with_interactions("MEMW_W", memw_w_receiving_memw());
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
    set_chip_word(&mut forged, "READER", reader, 0, 0x55, 0);
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
    set_chip_word(&mut bench, "READER", reader, 0, 0, 0);
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

/// ★ A chip's word messages carry its call's timestamp. The honest bench's
/// read of `0x100` at 12 claims `(0, 0)`, the words before the write at 8,
/// by sending its messages at 5: MEMW_W's rows then read the page's first
/// tokens and the write steps from 5. Refused on the `Ecall` and `Memw` buses
/// alone (the call and its register read are at 12, the chip's row at 5).
/// Control: the messages' timestamp a column of their own, the call left at
/// 12, and the stale read balances; the census refuses that chip.
#[test]
fn a_chip_message_at_a_time_not_its_calls_is_refused() {
    let (m, honest_bench) = honest();
    let reader = m.readers.iter().position(|r| r.ts == 12).expect("the read");
    let mut forged = honest_bench.clone();
    for i in 0..2u64 {
        let address = 0x100 + 8 * i;
        let read = memw_w_row(&m, address, 12);
        let write = memw_w_row(&m, address, 8);
        forged.set_u64("MEMW_W", read, mw::TIMESTAMP, 5);
        for c in [mw::OLD_LO, mw::OLD_HI, mw::NEW_LO, mw::NEW_HI] {
            forged.set_u64("MEMW_W", read, c, 0);
        }
        forged.set_u64("MEMW_W", read, mw::STEP_0, 5 - 1);
        forged.set_u64("MEMW_W", write, mw::STEP_0, 8 - 5 - 1);
        forged.set_u64(&page_name(0), (address / 8) as usize, pw::TIMESTAMP, 8);
        set_chip_word(&mut forged, "READER", reader, i as usize, 0, 0);
    }
    let mut real = forged.clone();
    real.set_u64("READER", reader, chip::TS, 5);
    let v = verdict(&real);
    assert!(
        !v.left_on(BusId::Ecall).is_empty()
            && v.left
                .keys()
                .all(|&b| b == u64::from(BusId::Ecall) || b == u64::from(BusId::Memw))
            && v.unanswered.is_empty()
            && v.broken.is_empty()
            && v.preprocessed.is_empty(),
        "{v:?}"
    );

    // Control: the messages at a column of their own (5 on the forged row,
    // the call's timestamp on every other).
    let mut control =
        forged.with_interactions("READER", chip_with_free_message_ts(MessageKind::Access));
    control.piece_mut("READER").trace = widen(&control.piece("READER").trace, 1);
    for r in 0..m.readers.len() {
        let ts = control.get("READER", r, chip::TS);
        control.set("READER", r, chip::NUM_COLUMNS, ts);
    }
    control.set_u64("READER", reader, chip::NUM_COLUMNS, 5);
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
    assert!(census_table("READER", &chip_with_free_message_ts(MessageKind::Access), 0).is_err());
}

/// ★ A converter row: MEMW_W at `(MU_WRITE, MU_ACCESS) = (−1, 1)` moves no
/// token (its pair's multiplicity is 0), takes a chip's access message "read
/// `0x55`", and its write receive at −1 sends "write `0x55`", which a second
/// row takes, consuming the real word and producing `0x55`. The chip has read
/// a word memory never held. Refused by constraint 0 (`MU_WRITE` a bit) on the
/// converter row and nothing else. Control: constraint 0 reading zero.
#[test]
fn a_message_converter_row_is_refused_by_its_write_bit() {
    let (m, bench) = honest();
    let read = memw_w_row(&m, 0x100, 12);
    let real = m.memw_w[read].old();
    let mut forged = bench.clone();
    for c in 0..mw::NUM_COLUMNS {
        forged.set_u64("MEMW_W", read, c, 0);
    }
    let reader = m.readers.iter().position(|r| r.ts == 12).expect("the read");
    set_chip_word(&mut forged, "READER", reader, 0, 0x55, 0);
    let n = rows_used(&bench);
    let converter = n;
    let writer = n + 1;
    for (c, value) in [
        (mw::ADDRESS_LO, 0x100),
        (mw::TIMESTAMP, 12),
        (mw::OLD_LO, 0x55),
        (mw::NEW_LO, 0x55),
        (mw::MU_ACCESS, 1),
    ] {
        forged.set_u64("MEMW_W", converter, c, value);
    }
    forged.set("MEMW_W", converter, mw::MU_WRITE, FE::zero() - FE::one());
    let (real_lo, real_hi) = halves(real);
    for (c, value) in [
        (mw::ADDRESS_LO, 0x100),
        (mw::TIMESTAMP, 12),
        (mw::OLD_LO, u64::from(real_lo)),
        (mw::OLD_HI, u64::from(real_hi)),
        (mw::NEW_LO, 0x55),
        (mw::STEP_0, 12 - 8 - 1),
        (mw::MU_WRITE, 1),
    ] {
        forged.set_u64("MEMW_W", writer, c, value);
    }
    let page = page_name(0);
    forged.set_u64(&page, 0x100 / 8, pw::FINI_LO, 0x55);
    forged.set_u64(&page, 0x100 / 8, pw::FINI_HI, 0);
    let v = verdict(&forged);
    assert!(v.left.is_empty() && v.unanswered.is_empty(), "{v:?}");
    assert_eq!(
        v.broken,
        vec![("MEMW_W".to_string(), converter, 0)],
        "{v:?}"
    );
    let mut control = forged;
    control.piece_mut("MEMW_W").constraints = Some(memw_w_constraints_without_bit0);
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

pub(super) fn load8_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&super::mem_load8::MemLoad8Constraints, row)
}
pub(super) fn store8_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&super::mem_store8::MemStore8Constraints, row)
}
pub(super) fn loadw_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&super::mem_loadw::MemLoadWConstraints, row)
}
pub(super) fn storew_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&super::mem_storew::MemStoreWConstraints, row)
}
pub(super) fn load2w_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&super::mem_load2w::MemLoad2WConstraints, row)
}
pub(super) fn store2w_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&super::mem_store2w::MemStore2WConstraints, row)
}

/// A table's constraints on one row and its width, for the census's bit
/// checks: by name (a table's family, before its index).
fn evaluator(name: &str) -> Option<(Eval, usize)> {
    use super::{mem_load2w, mem_load8, mem_loadw, mem_store2w, mem_store8, mem_storew};
    let family = name.split('[').next().unwrap_or(name);
    match family {
        "MEMW_W" => Some((memw_w_constraints as Eval, mw::NUM_COLUMNS)),
        "CHIP" | "WRITER" | "READER" => Some((chip_constraints as Eval, chip::NUM_COLUMNS)),
        "MEM_LOAD8" => Some((load8_constraints as Eval, mem_load8::cols::NUM_COLUMNS)),
        "MEM_STORE8" => Some((store8_constraints as Eval, mem_store8::cols::NUM_COLUMNS)),
        "MEM_LOADW" => Some((loadw_constraints as Eval, mem_loadw::cols::NUM_COLUMNS)),
        "MEM_STOREW" => Some((storew_constraints as Eval, mem_storew::cols::NUM_COLUMNS)),
        "MEM_LOAD2W" => Some((load2w_constraints as Eval, mem_load2w::cols::NUM_COLUMNS)),
        "MEM_STORE2W" => Some((store2w_constraints as Eval, mem_store2w::cols::NUM_COLUMNS)),
        _ => None,
    }
}

/// Rule c11, a CPU word table's key: the dispatch's timestamp's high word the
/// constant 0, the key's high element the dispatch's high address word, and
/// its low element the dispatch's low address word less a one-hot offset
/// (`Σ k·OFF_k`, each `k` in `1..8` at most once) plus `8` or nothing; the
/// one-hot's columns held to bits, one at a time, by the table's constraints.
fn check_dispatch_key(
    name: &str,
    dispatch: &BusInteraction,
    produced: &BusInteraction,
    eval: Option<(Eval, usize)>,
) -> Result<(), String> {
    if dispatch.values.len() != 9 || form(&dispatch.values[1])? != BTreeMap::new() {
        return Err(format!(
            "{name}: a dispatch that is not the CPU's nine values with timestamp high word 0"
        ));
    }
    if produced.values[1] != dispatch.values[3] {
        return Err(format!(
            "{name}: a key high word that is not the dispatch's"
        ));
    }
    let mut diff = form(&produced.values[0])?;
    for (k, v) in form(&dispatch.values[2])? {
        *diff.entry(k).or_insert(0) -= v;
    }
    diff.retain(|_, c| *c != 0);
    let constant = diff.remove(&None).unwrap_or(0);
    let mut offsets: Vec<i128> = diff.values().map(|c| -c).collect();
    offsets.sort();
    let distinct = offsets.windows(2).all(|w| w[0] != w[1]);
    if ![0, 8].contains(&constant) || !distinct || offsets.iter().any(|k| !(1..8).contains(k)) {
        return Err(format!(
            "{name}: a key low word that is not the dispatch's address less a one-hot offset"
        ));
    }
    if diff.is_empty() {
        return Ok(());
    }
    let Some((eval, width)) = eval else {
        return Err(format!("{name}: no constraint holds its one-hot to bits"));
    };
    let cols: Vec<usize> = columns(&diff).into_iter().collect();
    for &c in &cols {
        for bad in [FE::from(2u64), FE::zero() - FE::one()] {
            let mut row = vec![FE::zero(); width];
            row[c] = bad;
            if eval(&row).iter().all(|v| *v == FE::zero()) {
                return Err(format!(
                    "{name}: one-hot column {c} at {bad:?} breaks nothing"
                ));
            }
        }
    }
    Ok(())
}

/// Rule c5: the multiplicity columns `mu` held to bits, and to one message a
/// row: every assignment of them in `{−1, 0, 1, 2}^k` other than all zero and
/// a unit vector (exactly one 1), all else zero, breaks a constraint.
fn check_bits(name: &str, mu: &BTreeSet<usize>, eval: Option<(Eval, usize)>) -> Result<(), String> {
    let Some((eval, width)) = eval else {
        return Err(format!(
            "{name}: no constraint holds its multiplicity to a bit"
        ));
    };
    let cols: Vec<usize> = mu.iter().copied().collect();
    let values = [-1i64, 0, 1, 2];
    for code in 0..values.len().pow(cols.len() as u32) {
        let assignment: Vec<i64> = (0..cols.len())
            .map(|i| values[(code / values.len().pow(i as u32)) % values.len()])
            .collect();
        let ones = assignment.iter().filter(|&&v| v == 1).count();
        let zeros = assignment.iter().filter(|&&v| v == 0).count();
        if zeros == cols.len() || (ones == 1 && zeros == cols.len() - 1) {
            continue;
        }
        let mut row = vec![FE::zero(); width];
        for (&c, &v) in cols.iter().zip(&assignment) {
            row[c] = if v < 0 {
                FE::zero() - FE::from(v.unsigned_abs())
            } else {
                FE::from(v as u64)
            };
        }
        if eval(&row).iter().all(|v| *v == FE::zero()) {
            return Err(format!(
                "{name}: multiplicities {assignment:?} on {cols:?} break no constraint"
            ));
        }
    }
    Ok(())
}

/// Rule c2: a word-bus tuple's address is two elements, never packed: its low
/// element's columns at `1`, `8` or `2^16` (an address, a word index, a limb),
/// or at `−1..−7` (a one-hot offset taken off a CPU word table's address).
fn check_address(values: &[BusValue], what: &str) -> Result<(), String> {
    let lo = form(&values[0])?;
    let hi = form(&values[1])?;
    if lo
        .iter()
        .any(|(k, c)| k.is_some() && ![1, 8, 1 << 16].contains(c) && !(-7..=-1).contains(c))
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

/// Whether `half` (a 32-bit half of a word a row sends) is range-checked in
/// the row at `mu`: two 16-bit columns `a + 2^16·b` each in an `IS_HALF`, or
/// four bytes `Σ b_k·2^(8k)` each in an `ARE_BYTES` pair.
fn is_checked_half(
    half: &BusValue,
    interactions: &[BusInteraction],
    mu: &BTreeMap<Option<usize>, i128>,
) -> bool {
    let Ok(f) = form(half) else {
        return false;
    };
    if f.contains_key(&None) {
        return false;
    }
    let at_mu = |bi: &&BusInteraction| bi.is_sender && multiplicity_form(&bi.multiplicity) == *mu;
    let halved: BTreeSet<usize> = interactions
        .iter()
        .filter(at_mu)
        .filter(|bi| bi.bus_id == u64::from(BusId::IsHalfword) && bi.values.len() == 1)
        .filter_map(|bi| form(&bi.values[0]).ok())
        .filter(|g| g.len() == 1)
        .filter_map(|g| {
            g.into_iter()
                .next()
                .and_then(|(k, c)| (c == 1).then_some(k?))
        })
        .collect();
    let bytes: BTreeSet<usize> = interactions
        .iter()
        .filter(at_mu)
        .filter(|bi| bi.bus_id == u64::from(BusId::AreBytes))
        .flat_map(|bi| bi.values.iter().filter_map(|v| form(v).ok()))
        .filter(|g| g.len() == 1)
        .filter_map(|g| {
            g.into_iter()
                .next()
                .and_then(|(k, c)| (c == 1).then_some(k?))
        })
        .collect();
    let mut coefficients: Vec<i128> = f.values().copied().collect();
    coefficients.sort();
    let cols = columns(&f);
    (coefficients == [1, 1 << 16] && cols.iter().all(|c| halved.contains(c)))
        || (coefficients == [1, 1 << 8, 1 << 16, 1 << 24] && cols.iter().all(|c| bytes.contains(c)))
}

/// The census of one table's word-bus interactions (rules c2–c6, c9–c10),
/// `preprocessed` its preprocessed prefix's width, its constraints by name
/// ([`evaluator`]). `Ok` if the table touches neither word bus.
fn census_table(
    name: &str,
    interactions: &[BusInteraction],
    preprocessed: usize,
) -> Result<(), String> {
    census_table_with(name, interactions, preprocessed, evaluator(name))
}

/// [`census_table`] with the table's constraints given (`eval`).
fn census_table_with(
    name: &str,
    interactions: &[BusInteraction],
    preprocessed: usize,
    eval: Option<(Eval, usize)>,
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
        // `base + 8·OFFSET` with OFFSET preprocessed and the base two 32-bit
        // constants; each initial-word column preprocessed or held to a half.
        let [mint, last] = tokens.as_slice() else {
            return Err(format!("{name}: {} word tokens, not 2", tokens.len()));
        };
        let offset = Some(pw::OFFSET);
        let lo = form(&mint.values[0])?;
        let hi = form(&mint.values[1])?;
        let is_word = |c: Option<&i128>| c.is_none_or(|&c| (0..1i128 << 32).contains(&c));
        let page_aligned = lo
            .get(&None)
            .is_none_or(|&c| c % DEFAULT_PAGE_SIZE as i128 == 0);
        if mint.is_sender
            || !last.is_sender
            || mint.multiplicity != Multiplicity::One
            || last.multiplicity != Multiplicity::One
            || form(&mint.values[2])? != BTreeMap::new()
            || mint.values[..2] != last.values[..2]
            || columns(&lo) != BTreeSet::from([pw::OFFSET])
            || lo.get(&offset) != Some(&8)
            || !(0..preprocessed).contains(&pw::OFFSET)
            || !columns(&hi).is_empty()
            || !is_word(lo.get(&None))
            || !page_aligned
            || !is_word(hi.get(&None))
        {
            return Err(format!(
                "{name}: not a mint at base + 8·OFFSET, OFFSET preprocessed, the base two \
                 page-aligned words"
            ));
        }
        let one = BTreeMap::from([(None, 1i128)]);
        for half in &mint.values[3..5] {
            for c in columns(&form(half)?) {
                let held = c < preprocessed
                    || interactions.iter().any(|bi| {
                        bi.is_sender
                            && bi.bus_id == u64::from(BusId::IsHalfword)
                            && multiplicity_form(&bi.multiplicity) == one
                            && bi.values.len() == 1
                            && form(&bi.values[0]).ok() == Some(BTreeMap::from([(Some(c), 1)]))
                    });
                if !held {
                    return Err(format!(
                        "{name}: initial-word column {c} neither preprocessed nor a half"
                    ));
                }
            }
        }
        return Ok(());
    }
    if tokens.is_empty() && messages.iter().all(|bi| bi.is_sender) {
        // A sender of word messages: c9, its address from a register read or
        // `IS_HALF` limbs, plus a constant below 2^8; c10, its timestamp and
        // multiplicity its call's, the multiplicity a bit, its written words
        // checked.
        for bi in &messages {
            check_message_source(name, interactions, bi)?;
            check_sender(name, interactions, bi, eval)?;
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
    let clocked: Vec<(usize, &BusInteraction)> = interactions
        .iter()
        .enumerate()
        .filter(|(_, bi)| {
            !bi.is_sender
                && (bi.bus_id == u64::from(BusId::MemWordMsg)
                    || bi.bus_id == u64::from(BusId::MemoryOp))
        })
        .collect();
    let mut claimed = BTreeSet::new();
    for (p, c) in produced.iter().zip(&consumed) {
        if p.values[..2] != c.values[..2] {
            return Err(format!("{name}: a pair at two keys"));
        }
        let mu = multiplicity_form(&p.multiplicity);
        if mu != multiplicity_form(&c.multiplicity) {
            return Err(format!("{name}: a pair at two multiplicities"));
        }
        // The produced timestamp is a clock-bound receive's (a word
        // message's at its fourth value, the CPU's dispatch's at its first),
        // and the pair fires exactly when one of them does.
        let ts = &p.values[2];
        let clocks: Vec<_> = clocked
            .iter()
            .filter(|(_, bi)| {
                (bi.bus_id == u64::from(BusId::MemWordMsg) && bi.values.get(3) == Some(ts))
                    || (bi.bus_id == u64::from(BusId::MemoryOp) && bi.values.first() == Some(ts))
            })
            .collect();
        let mut fired = BTreeMap::new();
        for (i, bi) in &clocks {
            claimed.insert(*i);
            for (k, v) in multiplicity_form(&bi.multiplicity) {
                *fired.entry(k).or_insert(0) += v;
            }
            if bi.bus_id == u64::from(BusId::MemoryOp) {
                check_dispatch_key(name, bi, p, eval)?;
                continue;
            }
            if bi.values.get(1..3) != Some(&p.values[..2]) {
                return Err(format!("{name}: a key that is not its message's"));
            }
            // c4: a word message's words are the pair's: an access's old
            // word the consumed token's and its new word the produced one's,
            // a write's word the produced one's.
            if bi.bus_id == u64::from(BusId::MemWordMsg) {
                let words_match = match form(&bi.values[0])?.get(&None).copied().unwrap_or(0) {
                    0 => bi.values[4..6] == c.values[3..5] && bi.values[6..8] == p.values[3..5],
                    1 => bi.values[4..6] == p.values[3..5],
                    _ => false,
                };
                if !words_match {
                    return Err(format!("{name}: a message's words are not its pair's"));
                }
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
        // c12: a word a pair reads or writes as bytes has every byte
        // checked, at the pair's multiplicity.
        for half in c.values[3..5].iter().chain(&p.values[3..5]) {
            let f = form(half)?;
            let bytes = f.len() == 4 && f.values().any(|&c| c == 1 << 8);
            if bytes && !is_checked_half(half, interactions, &mu) {
                return Err(format!(
                    "{name}: a word's bytes {f:?} not each in a checked pair"
                ));
            }
        }
        // c13: a pair the CPU's dispatch drives passes its word through (a
        // load: the produced word is the consumed one) or writes a word each
        // half of which is held: checked by lookups at the pair's
        // multiplicity, or one column a constraint fixes (that column at 1 on
        // an all-zero row breaks one). A register's words need not be
        // canonical, so a store that wrote them as they come would let such a
        // word into memory.
        let dispatched = clocks
            .iter()
            .any(|(_, bi)| bi.bus_id == u64::from(BusId::MemoryOp));
        if dispatched && p.values[3..5] != c.values[3..5] {
            for half in &p.values[3..5] {
                if is_checked_half(half, interactions, &mu) {
                    continue;
                }
                let f = form(half)?;
                let single = (f.len() == 1)
                    .then(|| f.iter().next())
                    .flatten()
                    .and_then(|(k, c)| (*c == 1).then_some(*k))
                    .flatten();
                let fixed = match (single, eval) {
                    (Some(col), Some((eval, width))) if col < width => {
                        let mut row = vec![FE::zero(); width];
                        row[col] = FE::one();
                        eval(&row).iter().any(|v| *v != FE::zero())
                    }
                    _ => false,
                };
                if !fixed {
                    return Err(format!(
                        "{name}: a store writes a half {f:?} that nothing holds"
                    ));
                }
            }
        }
        // c5: the multiplicities held to bits, one message a row.
        check_bits(name, &columns(&mu), eval)?;
    }
    // c4: every clock-bound receive belongs to a pair: a receive that moves
    // no token is a sink.
    if let Some((i, _)) = clocked.iter().find(|(i, _)| !claimed.contains(i)) {
        return Err(format!("{name}: receive {i} belongs to no token pair"));
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

/// Rule c10: a sender's word message is its call's: its timestamp is the
/// timestamp of an `Ecall` receive in the same row plus a constant in `0..=2`,
/// at that receive's multiplicity, which a constraint holds to a bit; and the
/// words it writes (a write's word, an access's new word unless it is the old
/// one) are range-checked in the row at that multiplicity.
fn check_sender(
    name: &str,
    interactions: &[BusInteraction],
    message: &BusInteraction,
    eval: Option<(Eval, usize)>,
) -> Result<(), String> {
    let mu = multiplicity_form(&message.multiplicity);
    let ts = form(&message.values[3])?;
    let called = interactions.iter().any(|bi| {
        !bi.is_sender
            && bi.bus_id == u64::from(BusId::Ecall)
            && multiplicity_form(&bi.multiplicity) == mu
            && form(&bi.values[0]).is_ok_and(|call| {
                let mut diff = ts.clone();
                for (k, v) in call {
                    *diff.entry(k).or_insert(0) -= v;
                }
                diff.retain(|_, c| *c != 0);
                columns(&diff).is_empty()
                    && (0..=2).contains(&diff.get(&None).copied().unwrap_or(0))
            })
    });
    if !called {
        return Err(format!(
            "{name}: a word message's timestamp or multiplicity is not its call's"
        ));
    }
    check_bits(name, &columns(&mu), eval)?;
    let written: &[BusValue] = match form(&message.values[0])?.get(&None).copied().unwrap_or(0) {
        1 => &message.values[4..6],
        0 if message.values[6..8] == message.values[4..6] => &[],
        0 => &message.values[6..8],
        _ => return Err(format!("{name}: a word message with no kind constant")),
    };
    if let Some(half) = written
        .iter()
        .find(|half| !is_checked_half(half, interactions, &mu))
    {
        return Err(format!(
            "{name}: a written word's half {half:?} is not range-checked"
        ));
    }
    Ok(())
}

/// Rule c9: a word message's address is a register's value read in the same
/// row at the message's multiplicity, or `IS_HALF` limbs, plus a constant below
/// 2^8.
fn check_message_source(
    name: &str,
    interactions: &[BusInteraction],
    message: &BusInteraction,
) -> Result<(), String> {
    check_address(&message.values[1..3], name)?;
    let mu = multiplicity_form(&message.multiplicity);
    // A register's value: the value positions of a register read on `Memw`
    // (`is_register` the constant 1 at position 8; old value at 0 and 1, new
    // at 11 and 12), or of a token on the `Reg` bus (2 and 3).
    let one = BTreeMap::from([(None, 1i128)]);
    let mut registered = BTreeSet::new();
    for bi in interactions {
        if multiplicity_form(&bi.multiplicity) != mu {
            continue;
        }
        let positions: &[usize] = if bi.bus_id == u64::from(BusId::Memw)
            && bi.values.get(8).and_then(|v| form(v).ok()) == Some(one.clone())
        {
            &[0, 1, 11, 12]
        } else if bi.bus_id == u64::from(BusId::Reg) {
            &[2, 3]
        } else {
            &[]
        };
        for &i in positions {
            if let Some(Ok(f)) = bi.values.get(i).map(form)
                && f.len() == 1
                && let Some((Some(c), 1)) = f.iter().next().map(|(k, v)| (*k, *v))
            {
                registered.insert(c);
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

/// Counts the word-format frames take: one table of every family that is
/// not retired (LOAD, STORE, MEMW, MEMW_A at 0).
fn words_counts() -> crate::TableCounts {
    crate::TableCounts {
        cpu: 1,
        lt: 1,
        mul: 1,
        dvrm: 1,
        shift: 1,
        branch: 1,
        memw_register: 1,
        eq: 1,
        bytewise: 1,
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
        ..zero_counts()
    }
}

/// [`words_counts`] with no chip's table: what a word-format frame takes
/// while the format takes no chip (`block_whir::chip_tables_under`).
fn chip_free_counts() -> crate::TableCounts {
    crate::TableCounts {
        keccak: 0,
        keccak_rnd: 0,
        ecsm: 0,
        ecdas: 0,
        hint: 0,
        copy: 0,
        blake3: 0,
        is_wh: 0,
        ecrec: 0,
        ..words_counts()
    }
}

/// One table of each CPU word family and no MEMW_W.
const CPU_WORD_TABLES: [u64; 7] = [1, 1, 1, 1, 1, 1, 0];

/// [`frame_of`] with the word format's chip refusal taken out: the census
/// reads every table a word-format set can hold, the chips' and MEMW_W's
/// included.
fn frame_with_chips(
    statement: &block_whir::OwnedBlockStatement,
    format: &block_whir::BlockFormat,
) -> Result<block_whir::BlockFrame, crate::Error> {
    block_whir::CHIP_COUNTS_UNCHECKED.with(|c| c.set(true));
    let frame = frame_of(statement, format);
    block_whir::CHIP_COUNTS_UNCHECKED.with(|c| c.set(false));
    frame
}

/// A block statement under `format` (its memory format the word one, or any
/// other to show what that one refuses): `counts`, the fused pair `[0, 0]`,
/// `words` word tables, one private-input page and the stack's zero page,
/// every table at 2^2 rows but BITWISE (its 2^20) and PAGE_W (its 2^15), one
/// group a table: what the verifier frames. The heights are read off the AIR
/// set the verifier builds for the word format.
pub(super) fn words_statement(
    format: &block_whir::BlockFormat,
    counts: &crate::TableCounts,
    words: [u64; 7],
) -> block_whir::OwnedBlockStatement {
    let elf = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let opts = ProofOptions::default_test_options();
    let runtime_page_ranges = vec![crate::RuntimePageRange {
        base: STACK_BASE,
        count: 1,
    }];
    let pages = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &program,
        &runtime_page_ranges,
        1,
        1 << 20,
    )
    .expect("pages");
    let inputs =
        crate::BlockAirInputs::new(&opts, &program, vec![4; counts.cpu], &[]).expect("inputs");
    let mut airs = crate::block_airs(
        &program,
        &opts,
        &pages,
        counts,
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        format.registers,
    );
    if format.memory == MemoryFormat::Words {
        airs.with_word_memory(&opts, words.map(|n| n as usize), &pages)
            .expect("the word tables");
    }
    let table_num_vars: Vec<u8> = airs
        .air_refs()
        .iter()
        .map(|air| {
            if air.name() == "BITWISE" {
                crate::tables::bitwise::NUM_VARS as u8
            } else if air.name().starts_with(page_w::NAME_PREFIX) {
                page_w::NUM_VARS as u8
            } else if air.name().starts_with("PAGE:") {
                crate::continuation::PAGE_NUM_VARS as u8
            } else {
                2
            }
        })
        .collect();
    block_whir::OwnedBlockStatement {
        groups: (0..table_num_vars.len() as u32).map(|t| vec![t]).collect(),
        table_num_vars,
        runtime_page_ranges,
        table_counts: counts.clone(),
        public_output: Vec::new(),
        num_private_input_pages: 1,
        fused_memory: [0, 0],
        word_memory: words,
    }
}

/// The verifier's frame of `statement` under `format`.
pub(super) fn frame_of(
    statement: &block_whir::OwnedBlockStatement,
    format: &block_whir::BlockFormat,
) -> Result<block_whir::BlockFrame, crate::Error> {
    let elf = crate::test_utils::asm_elf_bytes("all_instructions_64");
    block_whir::block_frame(
        statement.view(),
        &elf,
        &ProofOptions::default_test_options(),
        format,
    )
}

/// The block's AIR set under the word memory format, as the verifier builds it
/// (`block_frame` of a word-format statement with one table of every family,
/// the chips' included: [`frame_with_chips`]), under the census: no byte PAGE
/// table, no cross-epoch table, no RAM token on the byte bus; the word bus
/// PAGE_W's, MEMW_W's and the CPU word tables' alone; MEMW_W alone receiving
/// word messages; every table's rules (c2–c12).
fn word_format_census(registers: crate::tables::register::RegisterFormat) -> Result<(), String> {
    let format = block_whir::BlockFormat {
        registers,
        ..block_whir::BlockFormat::pinned().with_memory(MemoryFormat::Words)
    };
    let statement = words_statement(&format, &words_counts(), [1; 7]);
    let frame = frame_with_chips(&statement, &format).map_err(|e| format!("the frame: {e:?}"))?;
    let set = frame.airs.air_refs();
    let mut on_words = BTreeSet::new();
    let mut message_receivers = BTreeSet::new();
    for air in &set {
        let name = air.name();
        let interactions = air.bus_interactions();
        // c8: no cross-epoch table; no byte page.
        if name.starts_with("LOCAL_TO_GLOBAL") || name.starts_with("GLOBAL_MEMORY") {
            return Err(format!("{name}: a cross-epoch table under the word format"));
        }
        if name.starts_with("PAGE:") {
            return Err(format!("{name}: a byte page under the word format"));
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
        check_set_member(name, interactions)?;
    }
    // c1 and c6: the word bus is PAGE_W's, MEMW_W's and the CPU word tables'
    // alone, and MEMW_W alone receives word messages.
    let mut allowed: BTreeSet<String> = block_whir::WORD_TABLES
        .iter()
        .map(|t| t.to_string())
        .collect();
    allowed.insert("PAGE_W".to_string());
    if on_words != allowed {
        return Err(format!("the word bus is touched by {on_words:?}"));
    }
    if message_receivers != BTreeSet::from(["MEMW_W".to_string()]) {
        return Err(format!(
            "word messages are received by {message_receivers:?}"
        ));
    }
    Ok(())
}

/// Rules c14 and c15, on each table of a word-format set: every receive on
/// the byte-message bus `Memw` is a register's (its register flag the
/// constant 1: the first of a write's 16 values, the ninth of a read's 24,
/// whose first eight are the old bytes), so no table takes a chip's RAM
/// message once MEMW and MEMW_A are retired; and only a CPU table sends the
/// memory dispatch the word tables take their timestamp and address from.
fn check_set_member(name: &str, interactions: &[BusInteraction]) -> Result<(), String> {
    let one = BTreeMap::from([(None, 1i128)]);
    for bi in interactions {
        let flag = match bi.values.len() {
            16 => bi.values.first(),
            24 => bi.values.get(8),
            _ => None,
        };
        if bi.bus_id == u64::from(BusId::Memw)
            && !bi.is_sender
            && flag.and_then(|v| form(v).ok()) != Some(one.clone())
        {
            return Err(format!(
                "{name}: receives a byte message that is not a register's"
            ));
        }
        if bi.bus_id == u64::from(BusId::MemoryOp) && bi.is_sender && !name.starts_with("CPU[") {
            return Err(format!("{name}: sends the CPU's memory dispatch"));
        }
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

/// Word-bus addresses packed into one element and a zero, `(lo + 2^32·hi, 0)`:
/// still five values a token, refused by rule c2 alone.
fn packed_five(interactions: Vec<BusInteraction>) -> Vec<BusInteraction> {
    interactions
        .into_iter()
        .map(|mut bi| {
            let at = if bi.bus_id == u64::from(BusId::MemWord) {
                0
            } else if bi.bus_id == u64::from(BusId::MemWordMsg) {
                1
            } else {
                return bi;
            };
            let mut pair = bi.values[at..at + 2].to_vec();
            let one = packed(vec![BusInteraction::sender(
                BusId::MemWord,
                Multiplicity::One,
                std::mem::take(&mut pair),
            )]);
            bi.values[at] = one[0].values[0].clone();
            bi.values[at + 1] = BusValue::constant(0);
            bi
        })
        .collect()
}

/// MEMW_W with one more access receive on columns of its own, tied to no
/// token pair: a sink that takes a chip's message and moves nothing.
fn memw_w_with_a_sink() -> Vec<BusInteraction> {
    let own = |k: usize| column(mw::NUM_COLUMNS + k);
    let mut out = memw_w::bus_interactions();
    out.push(BusInteraction::receiver(
        BusId::MemWordMsg,
        Multiplicity::Column(mw::NUM_COLUMNS + 5),
        access_message([own(0), own(1)], own(2), [own(3), own(4)], [own(3), own(4)]),
    ));
    out
}

/// MEMW_W's constraints with constraint 0 (`MU_WRITE` a bit) reading zero.
fn memw_w_constraints_without_bit0(row: &[FE]) -> Vec<FE> {
    let mut v = memw_w_constraints(row);
    v[0] = FE::zero();
    v
}

/// A stand-in chip of `kind` with `edit` applied to each of its word messages.
fn chip_with(kind: MessageKind, edit: impl Fn(&mut BusInteraction)) -> Vec<BusInteraction> {
    let mut out = chip_interactions(kind);
    for bi in &mut out {
        if bi.bus_id == u64::from(BusId::MemWordMsg) {
            edit(bi);
        }
    }
    out
}

/// A stand-in chip whose word messages carry a timestamp column of their own
/// (after the chip's), not its call's: a control mutant.
fn chip_with_free_message_ts(kind: MessageKind) -> Vec<BusInteraction> {
    chip_with(kind, |bi| bi.values[3] = column(chip::NUM_COLUMNS))
}

/// ★ The verifier's frame under the word memory format holds each of its
/// rules, each by its own refusal: the honest statement frames, with PAGE_W
/// in every page's place and the word tables after STORE's (where the fused
/// tables sit) and before CPU32's; a statement stating any table the format
/// retires is refused by name (LOAD, STORE, MEMW, MEMW_A, MEM_LOAD,
/// MEM_STORE); a PAGE_W table stated at `2^16` rows is refused, and so is a
/// word table past the block's chunk. Each refusal is the rule's own, so a
/// frame that skipped the rule fails here. A chip-free statement frames as it
/// is; the rules are read on the whole set, every chip's table and MEMW_W
/// included, with the chip refusal taken out
/// ([`the_word_format_refuses_every_chip_table`] holds that one).
#[test]
fn the_word_format_frame_holds_its_rules() {
    let format = block_whir::BlockFormat::pinned().with_memory(MemoryFormat::Words);
    let counts = words_counts();
    // A chip-free statement frames as it is; the rest reads the whole set,
    // the chips' tables and MEMW_W included, with the chip refusal out.
    let chip_free = words_statement(&format, &chip_free_counts(), CPU_WORD_TABLES);
    frame_of(&chip_free, &format).expect("the chip-free statement frames");
    let honest = words_statement(&format, &counts, [1; 7]);
    let frame = frame_with_chips(&honest, &format).expect("the honest statement frames");
    let names: Vec<String> = frame
        .airs
        .air_refs()
        .iter()
        .map(|air| air.name().to_string())
        .collect();
    assert!(names.iter().all(|n| !n.starts_with("PAGE:")), "{names:?}");
    assert_eq!(
        names
            .iter()
            .filter(|n| n.starts_with(page_w::NAME_PREFIX))
            .count(),
        frame.page_configs.len()
    );
    let first_word = names
        .iter()
        .position(|n| n == "MEM_LOAD8[0]")
        .expect("the word tables");
    let words: Vec<&str> = names[first_word..first_word + 7]
        .iter()
        .map(|n| n.split('[').next().unwrap_or(""))
        .collect();
    assert_eq!(words, block_whir::WORD_TABLES, "in their order");
    assert!(names[first_word + 7].starts_with("CPU32["), "{names:?}");
    assert!(
        names[first_word - 1].starts_with("BYTEWISE["),
        "after STORE's place: {names:?}"
    );

    // Each retired table, stated: refused by name.
    for (table, mut retired, fused) in [
        ("LOAD", counts.clone(), [0, 0]),
        ("STORE", counts.clone(), [0, 0]),
        ("MEMW", counts.clone(), [0, 0]),
        ("MEMW_A", counts.clone(), [0, 0]),
        ("MEM_LOAD", counts.clone(), [1, 0]),
        ("MEM_STORE", counts.clone(), [0, 1]),
    ] {
        match table {
            "LOAD" => retired.load = 1,
            "STORE" => retired.store = 1,
            "MEMW" => retired.memw = 1,
            "MEMW_A" => retired.memw_aligned = 1,
            _ => {}
        }
        let mut statement = words_statement(&format, &retired, [1; 7]);
        statement.fused_memory = fused;
        let refused = frame_with_chips(&statement, &format);
        // The fused pair is refused first by the fused tables' own rule
        // under the word format, then by the retired tables' rule.
        let why = if fused == [0, 0] {
            format!("no {table} table")
        } else {
            "states no fused table".to_string()
        };
        assert!(
            matches!(&refused, Err(crate::Error::InvalidTableCounts(w)) if w.contains(&why)),
            "{table}: {:?}",
            refused.err()
        );
    }
    // A PAGE_W table at 2^16 rows, and a word table past the chunk.
    let page = names
        .iter()
        .position(|n| n.starts_with(page_w::NAME_PREFIX))
        .expect("a PAGE_W table");
    let mut tall = honest.clone();
    tall.table_num_vars[page] = page_w::NUM_VARS as u8 + 1;
    let refused = frame_with_chips(&tall, &format);
    assert!(
        matches!(&refused, Err(crate::Error::InvalidTableCounts(why)) if why.contains("a PAGE_W table is 2^15")),
        "{:?}",
        refused.err()
    );
    for k in 0..7 {
        let mut tall = honest.clone();
        tall.table_num_vars[first_word + k] = block_whir::BLOCK_WORD_MAX_VARS as u8 + 1;
        let refused = frame_with_chips(&tall, &format);
        assert!(
            matches!(&refused, Err(crate::Error::InvalidTableCounts(why)) if why.contains("a block takes at most 2^21")),
            "{}: {:?}",
            block_whir::WORD_TABLES[k],
            refused.err()
        );
    }
    // A count the heights do not carry.
    let mut short = honest.clone();
    short.word_memory[0] = 2;
    assert!(
        frame_with_chips(&short, &format).is_err(),
        "one more MEM_LOAD8 than stated heights"
    );
}

/// ★ Under the word memory format every chip's table is refused by count,
/// each by name: KECCAK, KECCAK_RND, IS_WH, BLAKE3, COPY, ECSM, ECDAS, ECREC,
/// HINT and MEMW_W. A chip-free statement frames; the same statement stating
/// one table of one chip (its heights carried, the set built with it) is
/// refused, and with the refusal taken out (a control mutant) it frames, so
/// the rule alone is what refuses it. Under the split and fused formats the
/// rule reads nothing.
#[test]
fn the_word_format_refuses_every_chip_table() {
    let format = block_whir::BlockFormat::pinned().with_memory(MemoryFormat::Words);
    let chip_free = chip_free_counts();
    frame_of(
        &words_statement(&format, &chip_free, CPU_WORD_TABLES),
        &format,
    )
    .expect("the chip-free statement frames");
    let names = [
        "KECCAK",
        "KECCAK_RND",
        "IS_WH",
        "BLAKE3",
        "COPY",
        "ECSM",
        "ECDAS",
        "ECREC",
        "HINT",
        "MEMW_W",
    ];
    for table in names {
        let mut counts = chip_free.clone();
        let mut words = CPU_WORD_TABLES;
        match table {
            "KECCAK" => counts.keccak = 1,
            "KECCAK_RND" => counts.keccak_rnd = 1,
            "IS_WH" => counts.is_wh = 1,
            "BLAKE3" => counts.blake3 = 1,
            "COPY" => counts.copy = 1,
            "ECSM" => counts.ecsm = 1,
            "ECDAS" => counts.ecdas = 1,
            "ECREC" => counts.ecrec = 1,
            "HINT" => counts.hint = 1,
            _ => words[6] = 1,
        }
        let statement = words_statement(&format, &counts, words);
        let refused = frame_of(&statement, &format);
        assert!(
            matches!(&refused, Err(crate::Error::InvalidTableCounts(why))
                if why.contains("takes no chip") && why.contains(&format!("no {table} table"))),
            "{table}: {:?}",
            refused.err()
        );
        assert!(
            frame_with_chips(&statement, &format).is_ok(),
            "{table}: without the refusal it frames"
        );
        assert!(
            block_whir::chip_tables_under(MemoryFormat::Words, &counts, words).is_err(),
            "{table}"
        );
        for memory in [MemoryFormat::Split, MemoryFormat::Fused] {
            assert!(
                block_whir::chip_tables_under(memory, &counts, words).is_ok(),
                "{memory:?} {table}"
            );
        }
    }
}

/// ★ Under the split and fused formats the word tables' counts are absorbed
/// nowhere, so any count but zero is refused: a statement that frames, its
/// counts then set to `1` in one family or `2^32 − 1`, is refused under each;
/// with the refusal taken out (a control mutant) each frames, the counts read
/// as none, so the field would otherwise be free.
#[test]
fn a_split_or_fused_statement_stating_word_tables_is_refused() {
    for memory in [MemoryFormat::Split, MemoryFormat::Fused] {
        let format = block_whir::BlockFormat::pinned().with_memory(memory);
        let honest = words_statement(&format, &words_counts(), [0; 7]);
        assert!(
            frame_of(&honest, &format).is_ok(),
            "{memory:?}: the honest statement"
        );
        for stated in [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, u64::from(u32::MAX)],
        ] {
            let mut forged = honest.clone();
            forged.word_memory = stated;
            let refused = frame_of(&forged, &format);
            assert!(
                matches!(&refused, Err(crate::Error::InvalidTableCounts(why)) if why.contains("states no word table")),
                "{memory:?} {stated:?}: {:?}",
                refused.err()
            );
            block_whir::WORD_COUNTS_UNCHECKED.with(|c| c.set(true));
            let control = frame_of(&forged, &format);
            block_whir::WORD_COUNTS_UNCHECKED.with(|c| c.set(false));
            assert!(
                control.is_ok(),
                "{memory:?} {stated:?}: without the refusal it frames"
            );
        }
    }
}

/// The CPU's memory dispatch (`cpu.rs`' MEMORY send) carries the address as
/// `res`'s two words, and every row sends `IS_HALF` on each of `res`'s four
/// halves at multiplicity one: the address is an integer below `2^64` by
/// halves, its low word below `2^32`.
fn check_cpu_address_halves(interactions: &[BusInteraction]) -> Result<(), String> {
    use crate::tables::cpu::cols;
    let dispatch: Vec<&BusInteraction> = interactions
        .iter()
        .filter(|bi| bi.bus_id == u64::from(BusId::MemoryOp))
        .collect();
    let [dispatch] = dispatch.as_slice() else {
        return Err(format!("{} memory dispatches", dispatch.len()));
    };
    if !dispatch.is_sender
        || dispatch.values.get(2)
            != Some(&BusValue::Packed {
                start_column: cols::RES_0,
                packing: Packing::DWordHL,
            })
    {
        return Err("the dispatch's address is not res's two words".to_string());
    }
    for res in cols::RES {
        let held = interactions.iter().any(|bi| {
            bi.is_sender
                && bi.bus_id == u64::from(BusId::IsHalfword)
                && bi.multiplicity == Multiplicity::One
                && bi.values == vec![column(res)]
        });
        if !held {
            return Err(format!("res column {res} has no IS_HALF at one"));
        }
    }
    Ok(())
}

/// ★ The CPU range-checks the address it dispatches, under both register
/// formats: `IS_HALF` at one on each half of `res`, the dispatch's address
/// `res`'s two words. The word tables take that address's low word as an
/// integer below `2^32` (their one-hot is forced to the address's offset by
/// PAGE_W's keys, which hold the address to a key range of their own as well).
/// Pinned on the AIRs the verifier builds; each of the four checks taken out
/// (a mutant) fails the pin.
#[test]
fn the_cpu_dispatches_a_range_checked_address() {
    use crate::tables::cpu;
    for interactions in [cpu::bus_interactions(0), cpu::bus_interactions_inrow(0)] {
        check_cpu_address_halves(&interactions).expect("the CPU's address halves");
        for res in cpu::cols::RES {
            let mutant: Vec<BusInteraction> = interactions
                .iter()
                .filter(|bi| {
                    !(bi.bus_id == u64::from(BusId::IsHalfword) && bi.values == vec![column(res)])
                })
                .cloned()
                .collect();
            assert!(
                check_cpu_address_halves(&mutant).is_err(),
                "res {res} unchecked"
            );
        }
    }
    // The CPU tables of a word-format frame are those.
    for registers in [
        crate::tables::register::RegisterFormat::InRow,
        crate::tables::register::RegisterFormat::Tokens32,
    ] {
        let format = block_whir::BlockFormat {
            registers,
            ..block_whir::BlockFormat::pinned().with_memory(MemoryFormat::Words)
        };
        let frame = frame_of(
            &words_statement(&format, &chip_free_counts(), CPU_WORD_TABLES),
            &format,
        )
        .expect("the frame");
        let cpus: Vec<_> = frame
            .airs
            .air_refs()
            .into_iter()
            .filter(|air| air.name().starts_with("CPU["))
            .collect();
        assert!(!cpus.is_empty());
        for air in cpus {
            check_cpu_address_halves(air.bus_interactions())
                .unwrap_or_else(|e| panic!("{registers:?} {}: {e}", air.name()));
        }
    }
}

/// ★ The census refuses each mutant it exists for, and passes the honest
/// tables: on PAGE_W, a main `OFFSET`, a packed key, a key whose base is not
/// two 32-bit words, a private initial half held by no `IS_HALF`; on MEMW_W, a
/// packed address (dropped or five values), an unequal pair, a step without
/// its `−1`, an untagged message, a limb unranged, a sink receive, a message
/// whose words are not its pair's (old or new), multiplicities that are not
/// bits one by one (MEMW_W without constraint 0), a `Memw` receive; on the
/// stand-in chips, a free pointer, an address that is not a register's value,
/// a message timestamp or multiplicity that is not its call's, a written word
/// not range-checked.
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
    let stack = page_w::bus_interactions(STACK_BASE, false).expect("aligned");
    assert!(census_table("PAGE_W:stack", &stack, page_w::NUM_PREPROCESSED_COLS).is_ok());
    assert!(
        census_table(
            "PAGE_W:stack",
            &packed_five(stack),
            page_w::NUM_PREPROCESSED_COLS
        )
        .is_err(),
        "a packed key of five values"
    );
    let private = page_w::bus_interactions(INPUT_BASE, true).expect("aligned");
    assert!(census_table("PAGE_W:input", &private, 1).is_ok());
    for k in 0..4 {
        // The half's IS_HALF moved to the next half: `INIT_k` held by none.
        let mut mutant = private.clone();
        mutant[2 + k].values = vec![column(pw::INIT[(k + 1) % 4])];
        assert!(
            census_table("PAGE_W:input", &mutant, 1).is_err(),
            "INIT_{k} unheld"
        );
    }

    let memw = memw_w::bus_interactions();
    assert!(census_table("MEMW_W", &memw, 0).is_ok());
    let mut old_elsewhere = memw.clone();
    old_elsewhere[5].values[4] = column(mw::NEW_LO);
    old_elsewhere[5].values[5] = column(mw::NEW_HI);
    let mut produces_old = memw.clone();
    produces_old[3].values[3] = column(mw::OLD_LO);
    produces_old[3].values[4] = column(mw::OLD_HI);
    for (what, mutant) in [
        ("packed", packed(memw.clone())),
        ("packed, five values", packed_five(memw.clone())),
        ("unequal pair", memw_w_unequal_pair()),
        ("no −1", memw_w_without_the_minus_one()),
        ("untagged", memw_w_untagged()),
        ("a sink", memw_w_with_a_sink()),
        ("an access's old word elsewhere", old_elsewhere),
        ("a produced token carrying the old word", produces_old),
        ("a Memw receive", memw_w_receiving_memw()),
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
    // Constraint 0 reading zero: `(MU_WRITE, MU_ACCESS) = (−1, 1)` breaks
    // nothing, the converter row of `a_message_converter_row_is_refused_by_its_write_bit`.
    assert!(
        census_table_with(
            "MEMW_W",
            &memw,
            0,
            Some((memw_w_constraints_without_bit0, mw::NUM_COLUMNS))
        )
        .is_err(),
        "a converter row"
    );

    // The stand-ins: their address is x12's value plus 0 or 8, their messages
    // their call's, a writer's words checked.
    for kind in [MessageKind::Write, MessageKind::Access] {
        let chip = chip_interactions(kind);
        assert!(census_table("CHIP", &chip, 0).is_ok(), "{kind:?}");
        // The register read dropped: the pointer is a free column.
        let free: Vec<_> = chip.iter().skip(1).cloned().collect();
        assert!(
            census_table("CHIP", &free, 0).is_err(),
            "{kind:?} free pointer"
        );
        let mutants = [
            (
                "the address its timestamp column",
                chip_with(kind, |bi| {
                    bi.values[1] = column(chip::TS);
                }),
            ),
            ("a free message timestamp", chip_with_free_message_ts(kind)),
            (
                "a message timestamp 3 after its call's",
                chip_with(kind, |bi| {
                    bi.values[3] =
                        BusValue::linear(vec![term(1, chip::TS), LinearTerm::Constant(3)]);
                }),
            ),
            (
                "a free message multiplicity",
                chip_with(kind, |bi| {
                    bi.multiplicity = Multiplicity::Column(chip::NUM_COLUMNS);
                }),
            ),
        ];
        for (what, mutant) in mutants {
            assert!(
                census_table("CHIP", &mutant, 0).is_err(),
                "{kind:?}: {what}"
            );
        }
        // No constraint holds the multiplicity to a bit.
        assert!(
            census_table_with("CHIP", &chip, 0, None).is_err(),
            "{kind:?}: an unconstrained multiplicity"
        );
    }
    // The CPU's word tables: honest, then a key off the dispatch (the offset
    // added, the second word 16 on, the high word another column), a dispatch
    // whose timestamp high word is a column, a byte left out of its pair, a
    // one-hot column no constraint holds to a bit.
    use super::{mem_load2w, mem_load8, mem_loadw};
    for (name, table) in [
        ("MEM_LOAD8[0]", mem_load8::bus_interactions()),
        ("MEM_LOADW[0]", mem_loadw::bus_interactions()),
        ("MEM_LOAD2W[0]", mem_load2w::bus_interactions()),
    ] {
        assert!(census_table(name, &table, 0).is_ok(), "{name}");
    }
    let flip_offset = |interactions: Vec<BusInteraction>| -> Vec<BusInteraction> {
        interactions
            .into_iter()
            .map(|mut bi| {
                if bi.bus_id == u64::from(BusId::MemWord)
                    && let BusValue::Linear(terms) = &mut bi.values[0]
                {
                    for t in terms.iter_mut() {
                        if let LinearTerm::Column { coefficient, .. } = t
                            && *coefficient < 0
                        {
                            *coefficient = -*coefficient;
                        }
                    }
                }
                bi
            })
            .collect()
    };
    assert!(
        census_table(
            "MEM_LOADW[0]",
            &flip_offset(mem_loadw::bus_interactions()),
            0
        )
        .is_err(),
        "the offset added"
    );
    let mut sixteen = mem_load2w::bus_interactions();
    for bi in &mut sixteen[4..8] {
        if bi.bus_id == u64::from(BusId::MemWord)
            && let BusValue::Linear(terms) = &mut bi.values[0]
        {
            for t in terms.iter_mut() {
                if let LinearTerm::Constant(c) = t {
                    *c = 16;
                }
            }
        }
    }
    assert!(
        census_table("MEM_LOAD2W[0]", &sixteen, 0).is_err(),
        "the second word 16 on"
    );
    let mut high = mem_load8::bus_interactions();
    for bi in &mut high {
        if bi.bus_id == u64::from(BusId::MemWord) {
            bi.values[1] = column(super::mem_load8::cols::LO);
        }
    }
    assert!(
        census_table("MEM_LOAD8[0]", &high, 0).is_err(),
        "a high word not the dispatch's"
    );
    let mut ts_hi = mem_load8::bus_interactions();
    let dispatch = ts_hi.len() - 1;
    ts_hi[dispatch].values[1] = column(super::mem_load8::cols::HI);
    assert!(
        census_table("MEM_LOAD8[0]", &ts_hi, 0).is_err(),
        "a timestamp high word"
    );
    let unpaired: Vec<BusInteraction> = mem_loadw::bus_interactions()
        .into_iter()
        .filter(|bi| {
            !(bi.bus_id == u64::from(BusId::AreBytes)
                && bi.values[0] == column(super::mem_loadw::cols::BYTES[2]))
        })
        .collect();
    assert!(
        census_table("MEM_LOADW[0]", &unpaired, 0).is_err(),
        "a byte unpaired"
    );
    fn loadw_without_off3_bit(row: &[FE]) -> Vec<FE> {
        let mut v = loadw_constraints(row);
        v[3] = FE::zero();
        // The sum `Σ OFF = MU` alone would still catch `OFF_3 = 2`: take it out
        // too, so only the bits hold the one-hot.
        v[8] = FE::zero();
        v
    }
    assert!(
        census_table_with(
            "MEM_LOADW[0]",
            &mem_loadw::bus_interactions(),
            0,
            Some((loadw_without_off3_bit, super::mem_loadw::cols::NUM_COLUMNS))
        )
        .is_err(),
        "a one-hot column no constraint holds"
    );

    // A writer whose written words are not range-checked.
    let unchecked: Vec<_> = chip_interactions(MessageKind::Write)
        .into_iter()
        .filter(|bi| bi.bus_id != u64::from(BusId::IsHalfword))
        .collect();
    assert!(
        census_table("CHIP", &unchecked, 0).is_err(),
        "an unchecked written word"
    );
    // A reader whose new word is not its old one: a write, unchecked.
    let reader_writes = chip_with(MessageKind::Access, |bi| {
        bi.values[6] = column(chip::PTR_LO);
    });
    assert!(
        census_table("CHIP", &reader_writes, 0).is_err(),
        "an access writing an unchecked word"
    );
}

// ---- the format ----

/// ★ The word memory format is the verifier's to read and the prover's to
/// make under the in-row register format: the format check takes it (the
/// verifier frames a statement under it), the windowed build's setters take
/// it, and the prove refuses it at its entry under the 32-bit register
/// format, whose register accesses MEMW_R cannot take go to tables the format
/// retires. Under it the fused pair is `[0, 0]`; its statement words are its
/// form word and each word family's count.
#[test]
fn the_word_format_is_proved_under_the_in_row_registers_alone() {
    let words = block_whir::BlockFormat {
        registers: crate::tables::register::RegisterFormat::Tokens32,
        ..block_whir::BlockFormat::pinned().with_memory(MemoryFormat::Words)
    };
    assert!(block_whir::check_memory_format(&words).is_ok());
    assert!(block_whir::refuse_unprovable(&words).is_err());
    let in_row = block_whir::BlockFormat::production().with_memory(MemoryFormat::Words);
    assert!(block_whir::check_memory_format(&in_row).is_ok());
    assert!(block_whir::refuse_unprovable(&in_row).is_ok());
    for memory in [MemoryFormat::Split, MemoryFormat::Fused] {
        assert!(block_whir::refuse_unprovable(&words.with_memory(memory)).is_ok());
    }
    assert_eq!(
        block_whir::fused_tables_under(MemoryFormat::Words, [0, 0]).ok(),
        Some([0, 0])
    );
    for stated in [[1, 0], [0, 1]] {
        assert!(block_whir::fused_tables_under(MemoryFormat::Words, stated).is_err());
    }
    // The prove refuses the 32-bit register format at its entry, before
    // anything runs.
    let elf = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let refused = block_whir::prove_block_whir_with(
        &elf,
        &[],
        &ProofOptions::default_test_options(),
        &words,
        &crate::tests::block_whir_tests::streamed(crate::tables::MaxRowsConfig::small(), 3, 9),
        &block_whir::Deviations::default(),
    );
    assert!(
        matches!(&refused, Err(crate::Error::Prover(why)) if why.contains("in-row register format")),
        "the prover refuses the word format's 32-bit registers: {:?}",
        refused.err()
    );
    // The windowed build and its regenerator take it at their setters.
    use crate::tables::trace_builder::{RegenBuilder, WindowedTraceBuilder};
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let max_rows = crate::tables::MaxRowsConfig::small();
    let builder = WindowedTraceBuilder::new(&program, &[], &max_rows).expect("a builder");
    assert!(builder.memory_format(MemoryFormat::Words).is_ok());
    let regen = RegenBuilder::new(&program, &[], &max_rows).expect("a regenerator");
    assert!(regen.memory_format(MemoryFormat::Words).is_ok());
    // Its statement words: its form word, then each family's count.
    assert_eq!(
        block_whir::fused_statement_words(MemoryFormat::Words, [0, 0], [1, 2, 3, 4, 5, 6, 7]),
        vec![block_whir::WORD_MEMORY_WORD, 1, 2, 3, 4, 5, 6, 7]
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

/// ★ The census refuses a CPU store whose new word nothing holds (rule c13):
/// MEM_STORE8 without the `IS_HALF` on its new word's halves, and MEM_STOREW
/// without the identities that fix its new halves; the honest tables pass. And
/// it refuses, in a word-format set, a table receiving a byte message that is
/// not a register's, and a table other than the CPU sending the memory
/// dispatch (rules c14, c15); the honest set's tables pass.
#[test]
fn the_census_holds_a_stores_new_word_and_the_sets_byte_messages() {
    use super::{mem_store8, mem_storew};
    let store8 = mem_store8::bus_interactions();
    assert!(census_table("MEM_STORE8[0]", &store8, 0).is_ok());
    let unchecked: Vec<BusInteraction> = store8
        .iter()
        .filter(|bi| {
            !(bi.bus_id == u64::from(BusId::IsHalfword)
                && mem_store8::cols::NEW
                    .iter()
                    .any(|&c| bi.values == vec![column(c)]))
        })
        .cloned()
        .collect();
    assert_eq!(unchecked.len(), store8.len() - 4);
    let refused = census_table("MEM_STORE8[0]", &unchecked, 0);
    assert!(
        matches!(&refused, Err(why) if why.contains("nothing holds")),
        "{refused:?}"
    );
    let storew = mem_storew::bus_interactions();
    assert!(census_table("MEM_STOREW[0]", &storew, 0).is_ok());
    fn storew_without_new_halves(row: &[FE]) -> Vec<FE> {
        let mut v = storew_constraints(row);
        let probe = |col: usize| {
            let mut r = vec![FE::zero(); super::mem_storew::cols::NUM_COLUMNS];
            r[col] = FE::one();
            storew_constraints(&r)
        };
        for col in [
            super::mem_storew::cols::NEW_LO,
            super::mem_storew::cols::NEW_HI,
        ] {
            for (k, x) in probe(col).iter().enumerate() {
                if *x != FE::zero() {
                    v[k] = FE::zero();
                }
            }
        }
        v
    }
    let refused = census_table_with(
        "MEM_STOREW[0]",
        &storew,
        0,
        Some((
            storew_without_new_halves as Eval,
            super::mem_storew::cols::NUM_COLUMNS,
        )),
    );
    assert!(
        matches!(&refused, Err(why) if why.contains("nothing holds")),
        "{refused:?}"
    );
    // c14, c15 on the honest set and on two stand-ins.
    word_format_census(crate::tables::register::RegisterFormat::InRow)
        .unwrap_or_else(|e| panic!("{e}"));
    // A write's 16 values and a read's 24, the flag a column (any message)
    // or the constant 1 (a register's).
    let message = |flag: BusValue, read: bool| {
        let mut values: Vec<BusValue> = (0..16).map(column).collect();
        values[0] = flag;
        if read {
            let mut old: Vec<BusValue> = (16..24).map(column).collect();
            old.extend(values);
            values = old;
        }
        vec![BusInteraction::receiver(
            BusId::Memw,
            Multiplicity::One,
            values,
        )]
    };
    for read in [false, true] {
        let refused = check_set_member("MEMW_LIKE[0]", &message(column(30), read));
        assert!(
            matches!(&refused, Err(why) if why.contains("not a register's")),
            "read {read}: {refused:?}"
        );
        let refused = check_set_member("MEMW_LIKE[0]", &message(BusValue::constant(0), read));
        assert!(refused.is_err(), "read {read}: a RAM message");
        assert!(
            check_set_member("MEMW_R[0]", &message(BusValue::constant(1), read)).is_ok(),
            "read {read}: a register's"
        );
    }
    let dispatch = vec![BusInteraction::sender(
        BusId::MemoryOp,
        Multiplicity::One,
        vec![column(0)],
    )];
    assert!(check_set_member("CPU[0]", &dispatch).is_ok());
    let refused = check_set_member("CPU32[0]", &dispatch);
    assert!(
        matches!(&refused, Err(why) if why.contains("memory dispatch")),
        "{refused:?}"
    );
}

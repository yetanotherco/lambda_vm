//! ★ KECCAK_W (`tables::keccak_w`) on hand-built traces with no proof, on the
//! word format's bench (`tables::word_memory_tests`): PAGE_W, MEMW_W,
//! KECCAK_W, a stand-in round chip (its permutations the verifier's: the real
//! round chip computes each one), a stand-in CPU and register file.
//!
//! The honest calls' words are read from a byte model of memory, the rows
//! are KECCAK's own (`keccak::sponge_ops`, the executor's permutations), and
//! the bench is built by replaying the rows' word messages in timestamp
//! order, so a forger's rows (any field changed, the permutations
//! recomputed) get the memory that makes them as consistent as they can be:
//! each forgery is refused by the one check it targets, and accepted with
//! that check taken out (its control).

use std::collections::HashMap;

use stark::lookup::{BusInteraction, BusValue, Multiplicity};

use super::gpack::TraceForm;
use super::keccak::{self, KeccakOperation, SpongeCheck};
use super::keccak_w::{
    self, BLOCK_WORDS, DigestWords, KeccakWordConstraints, SpongeWords, cols as kw, digest_of,
    digest_words, reads,
};
use super::memw_w::{self, MemwWordConstraints, MemwWordRow, MessageKind, cols as mw};
use super::page::{DEFAULT_PAGE_SIZE, PageConfig};
use super::page_w::{self, FinalWordMap, FinalWordState, cols as pw};
use super::types::{BusId, FE};
use super::word_memory::{WORD_BYTES, column};
use super::word_memory_tests::{
    Bench, ELF_BASE, Eval, Piece, Table, constraint_values, page_name, page_piece, pages, table,
    verdict,
};
use executor::vm::instruction::execution::KECCAK256_SYSCALL_NUMBER;

fn keccak_w_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&KeccakWordConstraints, row)
}

fn memw_w_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&MemwWordConstraints, row)
}

/// KECCAK_W's constraints with constraint `I` reading zero: a control mutant.
fn keccak_w_without<const I: usize>(row: &[FE]) -> Vec<FE> {
    let mut v = keccak_w_constraints(row);
    v[I] = FE::zero();
    v
}

fn with_constraints(mut bench: Bench, eval: Eval) -> Bench {
    bench.piece_mut(keccak_w::NAME).constraints = Some(eval);
    bench
}

/// Only KECCAK_W's constraint `idx` breaks, on `row`, and nothing else
/// refuses the bench.
fn refused_by_constraint(bench: &Bench, idx: usize, row: usize) -> Result<(), String> {
    let v = verdict(bench);
    if v.left.is_empty()
        && v.unanswered.is_empty()
        && v.preprocessed.is_empty()
        && v.broken == vec![(keccak_w::NAME.to_string(), row, idx)]
    {
        Ok(())
    } else {
        Err(format!("{v:?}"))
    }
}

// ---- the stand-ins ----

/// The stand-in round chip: a permutation a row, `(T, j, input, output)`,
/// every column the verifier's.
mod rounds {
    pub const TS: usize = 0;
    pub const J: usize = 1;
    pub const IN: usize = 2;
    pub const OUT: usize = 202;
    pub const MU: usize = 402;
    pub const NUM_COLUMNS: usize = 403;
}

/// A state's 200 byte columns from `start`, in KECCAK's bus order (x major,
/// then y, then the lane's bytes; the columns lane-major).
fn state(start: usize) -> impl Iterator<Item = BusValue> {
    (0..5).flat_map(move |x| {
        (0..5).flat_map(move |y| (0..8).map(move |b| column(start + 8 * (x + 5 * y) + b)))
    })
}

fn rounds_interactions() -> Vec<BusInteraction> {
    let head = |round: u64| {
        vec![
            column(rounds::TS),
            column(rounds::J),
            BusValue::constant(round),
        ]
    };
    let mut request = head(0);
    request.extend(state(rounds::IN));
    let mut reply = head(24);
    reply.extend(state(rounds::OUT));
    vec![
        BusInteraction::receiver(BusId::Keccak, Multiplicity::Column(rounds::MU), request),
        BusInteraction::sender(BusId::Keccak, Multiplicity::Column(rounds::MU), reply),
    ]
}

/// The stand-in CPU (its ECALLs) and register file (each read's value),
/// every column but the register file's `MU` the verifier's.
mod cpu {
    pub const TS: usize = 0;
    pub const MU: usize = 1;
    pub const NUM_COLUMNS: usize = 2;
}

mod regs {
    pub const REG2: usize = 0;
    pub const LO: usize = 1;
    pub const HI: usize = 2;
    pub const TS: usize = 3;
    pub const MU: usize = 4;
    pub const NUM_COLUMNS: usize = 5;
}

fn cpu_interactions() -> Vec<BusInteraction> {
    vec![BusInteraction::sender(
        BusId::Ecall,
        Multiplicity::Column(cpu::MU),
        vec![
            column(cpu::TS),
            BusValue::constant(0),
            BusValue::constant(KECCAK256_SYSCALL_NUMBER & 0xFFFF_FFFF),
            BusValue::constant(KECCAK256_SYSCALL_NUMBER >> 32),
        ],
    )]
}

fn register_file_interactions() -> Vec<BusInteraction> {
    let zeros = || std::iter::repeat_n(BusValue::constant(0), 6);
    let mut tuple = vec![column(regs::LO), column(regs::HI)];
    tuple.extend(zeros());
    tuple.extend([
        BusValue::constant(1),
        column(regs::REG2),
        BusValue::constant(0),
        column(regs::LO),
        column(regs::HI),
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

// ---- a run of calls ----

/// Where the inputs and the digests sit: the ELF page, the zero page.
const IN: u64 = ELF_BASE + 0x800;
const OUT: u64 = 0x2000;

/// One word message `(timestamp, key, kind, old, new)` (a write's old is
/// memory's).
type Message = (u64, u64, MessageKind, u64, u64);

#[derive(Clone)]
struct Run {
    pages: Vec<PageConfig>,
    bytes: HashMap<u64, u8>,
    ops: Vec<KeccakOperation>,
    /// Each call's rows in `ops`.
    calls: Vec<std::ops::Range<usize>>,
    ecalls: Vec<u64>,
    regs: Vec<(u64, u64, u64)>,
    /// Reads a forger leaves out: `(row in ops, w)`.
    unread: Vec<(usize, usize)>,
    ts: u64,
}

impl Run {
    fn new() -> Self {
        Self {
            pages: pages(),
            bytes: HashMap::new(),
            ops: Vec::new(),
            calls: Vec::new(),
            ecalls: Vec::new(),
            regs: Vec::new(),
            unread: Vec::new(),
            ts: 4,
        }
    }

    fn page_of(&self, address: u64) -> Option<&PageConfig> {
        self.pages.iter().find(|p| {
            address
                .checked_sub(p.page_base)
                .is_some_and(|d| d < DEFAULT_PAGE_SIZE as u64)
        })
    }

    fn init_word(&self, key: u64) -> u64 {
        let page = self
            .page_of(key)
            .unwrap_or_else(|| panic!("a page holds {key:#x}"));
        page_w::init_word(page, ((key - page.page_base) / WORD_BYTES) as usize)
    }

    fn byte(&self, address: u64) -> u8 {
        if let Some(&b) = self.bytes.get(&address) {
            return b;
        }
        let key = address - address % WORD_BYTES;
        (self.init_word(key) >> (8 * (address % WORD_BYTES))) as u8
    }

    fn word(&self, key: u64) -> u64 {
        u64::from_le_bytes(std::array::from_fn(|i| self.byte(key + i as u64)))
    }

    /// One keccak256 call of `len` bytes at `ptr`, its digest to `out`: its
    /// rows' range in `ops`.
    fn call(&mut self, ptr: u64, len: u64, out: u64) -> std::ops::Range<usize> {
        self.ts += 4;
        let ts = self.ts;
        let input: Vec<u8> = (0..len).map(|i| self.byte(ptr + i)).collect();
        let mut ops = keccak::sponge_ops(ts, ptr, out, &input);
        let offset = (ptr % WORD_BYTES) as u8;
        let q = (out % WORD_BYTES) as u8;
        let out_key = out - u64::from(q);
        let n = ops.len();
        for (j, op) in ops.iter_mut().enumerate() {
            let key = ptr - u64::from(offset) + 136 * j as u64;
            let words = std::array::from_fn(|w| self.word(key + 8 * w as u64));
            let digest = (j + 1 == n).then(|| DigestWords {
                q,
                old0: self.word(out_key),
                old4: if q == 0 { 0 } else { self.word(out_key + 32) },
            });
            op.sponge.as_mut().expect("a sponge row").words = Some(Box::new(SpongeWords {
                offset,
                words,
                digest,
            }));
        }
        let digest = digest_of(&ops[n - 1].output);
        for (i, &b) in digest.iter().enumerate() {
            self.bytes.insert(out + i as u64, b);
        }
        self.ecalls.push(ts);
        for (reg, value) in [(10, ptr), (11, len), (12, out)] {
            self.regs.push((reg, value, ts));
        }
        let at = self.ops.len();
        self.ops.extend(ops);
        self.calls.push(at..self.ops.len());
        at..self.ops.len()
    }

    fn words(&self, row: usize) -> &SpongeWords {
        self.ops[row]
            .sponge
            .as_ref()
            .and_then(|s| s.words.as_deref())
            .expect("a KECCAK_W row")
    }

    fn words_mut(&mut self, row: usize) -> &mut SpongeWords {
        self.ops[row]
            .sponge
            .as_mut()
            .and_then(|s| s.words.as_deref_mut())
            .expect("a KECCAK_W row")
    }

    fn msg_mut(&mut self, row: usize) -> &mut [u8; 136] {
        &mut self.ops[row].sponge.as_mut().expect("a sponge row").msg
    }

    /// Call `call`'s permutations recomputed from its blocks (a forger's
    /// changed `MSG`), from the zero state: the round chip computes them.
    fn rechain(&mut self, call: usize) {
        let range = self.calls[call].clone();
        keccak::rechain(&mut self.ops[range], [0; 25]);
    }

    /// The rows' word messages: each read word at `T`, the digest's at `T + 1`.
    fn messages(&self) -> Vec<Message> {
        let mut out = Vec::new();
        for (row, op) in self.ops.iter().enumerate() {
            let sponge = op.sponge.as_ref().expect("a sponge row");
            let words = self.words(row);
            let t = op.timestamp & 0xFFFF_FFFF;
            let key = sponge.src - u64::from(words.offset);
            for (w, read) in reads(&sponge.mask, words.offset, sponge.first)
                .into_iter()
                .enumerate()
            {
                if read && !self.unread.contains(&(row, w)) {
                    let word = words.words[w];
                    out.push((t, key + 8 * w as u64, MessageKind::Access, word, word));
                }
            }
            if let Some(d) = &words.digest {
                let out_key = sponge.out - u64::from(d.q);
                let new = digest_words(&digest_of(&op.output), d.q, d.old0, d.old4);
                out.push((t + 1, out_key, MessageKind::Access, d.old0, new[0]));
                for (v, &word) in new.iter().enumerate().take(4).skip(1) {
                    out.push((t + 1, out_key + 8 * v as u64, MessageKind::Write, 0, word));
                }
                if d.q != 0 {
                    out.push((t + 1, out_key + 32, MessageKind::Access, d.old4, new[4]));
                }
            }
        }
        out
    }

    /// The MEMW_W rows of the messages replayed in timestamp order (an
    /// unaligned key starts from the message's old word at 0), each with the
    /// key it names; and the aligned words' finals.
    fn replay(&self) -> (Vec<(MemwWordRow, u64)>, FinalWordMap) {
        let mut messages = self.messages();
        messages.sort_by_key(|m| m.0);
        let mut state: HashMap<u64, (u64, u64)> = HashMap::new();
        let mut out = Vec::new();
        for (ts, key, kind, old, new) in messages {
            let (cur, cur_ts) = *state.entry(key).or_insert_with(|| {
                if key % WORD_BYTES == 0 {
                    (self.init_word(key), 0)
                } else {
                    (old, 0)
                }
            });
            let old = if kind == MessageKind::Write { cur } else { old };
            let row = MemwWordRow::new(kind, key - key % WORD_BYTES, ts, cur_ts, old, new)
                .expect("a replayed access");
            out.push((row, key));
            state.insert(key, (new, ts));
        }
        let finals = state
            .into_iter()
            .filter(|(key, _)| key % WORD_BYTES == 0)
            .map(|(key, (value, timestamp))| (key, FinalWordState { timestamp, value }))
            .collect();
        (out, finals)
    }

    fn bench(&self) -> Bench {
        let (memw, finals) = self.replay();
        let mut pieces: Vec<Piece> = self
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
        let mut trace = keccak_w::generate_keccak_w_trace_as(&self.ops, TraceForm::Wide);
        for &(row, w) in &self.unread {
            trace.main_table.set(row, kw::read(w), FE::zero());
        }
        pieces.push(Piece {
            name: keccak_w::NAME.into(),
            interactions: keccak_w::bus_interactions(),
            trace,
            preprocessed: Vec::new(),
            constraints: Some(keccak_w_constraints),
        });
        let rounds = table(self.ops.len(), rounds::NUM_COLUMNS, |t, row| {
            let op = &self.ops[row];
            t.main_table
                .set(row, rounds::TS, FE::from(op.timestamp & 0xFFFF_FFFF));
            t.main_table
                .set(row, rounds::J, FE::from(op.timestamp >> 32));
            for lane in 0..25 {
                for b in 0..8 {
                    let at = 8 * lane + b;
                    let byte = |word: u64| FE::from((word >> (8 * b)) & 0xFF);
                    t.main_table.set(row, rounds::IN + at, byte(op.input[lane]));
                    t.main_table
                        .set(row, rounds::OUT + at, byte(op.output[lane]));
                }
            }
            t.main_table.set(row, rounds::MU, FE::one());
        });
        pieces.push(truth_piece(
            "ROUNDS",
            rounds_interactions(),
            rounds,
            rounds::NUM_COLUMNS,
        ));
        let calls = table(self.ecalls.len(), cpu::NUM_COLUMNS, |t, row| {
            t.main_table.set(row, cpu::TS, FE::from(self.ecalls[row]));
            t.main_table.set(row, cpu::MU, FE::one());
        });
        pieces.push(truth_piece(
            "CPU",
            cpu_interactions(),
            calls,
            cpu::NUM_COLUMNS,
        ));
        let regs = table(self.regs.len(), regs::NUM_COLUMNS, |t, row| {
            let (reg, value, ts) = self.regs[row];
            t.main_table.set(row, regs::REG2, FE::from(2 * reg));
            t.main_table
                .set(row, regs::LO, FE::from(value & 0xFFFF_FFFF));
            t.main_table.set(row, regs::HI, FE::from(value >> 32));
            t.main_table.set(row, regs::TS, FE::from(ts));
            t.main_table.set(row, regs::MU, FE::one());
        });
        pieces.push(truth_piece(
            "REGS",
            register_file_interactions(),
            regs,
            regs::MU,
        ));
        Bench { pieces }
    }
}

// ---- the honest calls ----

/// ★ keccak256 calls at every input offset, at lengths on each side of a
/// block's edges (empty, a byte, a lane, 135, 136, 137, two blocks and
/// more), their digests at every offset, and one digest written over its own
/// input: the bench is accepted.
#[test]
fn honest_keccak256_calls_are_accepted() {
    let mut run = Run::new();
    for offset in 0..8u64 {
        for (k, &len) in [0u64, 1, 7, 8, 135, 136, 137, 271, 300].iter().enumerate() {
            let ptr = IN + offset + 0x40 * k as u64;
            let out = OUT + 0x100 * offset + 0x30 * k as u64 + offset;
            run.call(ptr, len, out);
        }
    }
    run.call(IN + 4, 140, IN + 4 + 10);
    let v = verdict(&run.bench());
    assert!(v.accepted(), "{v:?}");
}

// ---- forgeries ----

/// ★ A byte past the input entering the hash: the one row of a 10-byte call
/// places byte 10 (the next byte of its word, not input) into its block.
/// Refused by that byte's placement rule alone (`MSG_10 = s_10 · …` with
/// `s_10 = 0`). Control: that rule removed, accepted — an 11-byte hash
/// proved for a 10-byte call.
#[test]
fn a_byte_past_the_input_is_refused_by_its_placement() {
    let mut run = Run::new();
    let row = run.call(IN + 3, 10, OUT).start;
    let placed = (run.words(row).words[(10 + 3) / 8] >> (8 * ((10 + 3) % 8))) as u8;
    assert!(
        placed != 0 && placed < 0xFF,
        "a byte past the input: {placed}"
    );
    run.msg_mut(row)[10] = placed;
    run.rechain(0);
    let bench = run.bench();
    refused_by_constraint(&bench, keccak_w::PLACEMENT + 10, row).unwrap_or_else(|v| panic!("{v}"));
    let without: Eval = keccak_w_without::<{ keccak_w::PLACEMENT + 10 }>;
    let v = verdict(&with_constraints(bench, without));
    assert!(v.accepted(), "control: {v:?}");
}

/// KECCAK_W's interactions with the link's carried word (its last eight
/// values, both sides) taken out: a control mutant.
fn link_without_the_word() -> Vec<BusInteraction> {
    let mut out = keccak_w::bus_interactions();
    for bi in &mut out {
        if bi.bus_id == u64::from(BusId::KeccakSponge) {
            let n = bi.values.len();
            for v in &mut bi.values[n - 8..] {
                *v = BusValue::constant(0);
            }
        }
    }
    out
}

/// ★ A block's first word not the previous block's last (the shared word):
/// row 1 of an unaligned two-block call places other bytes from its `W_0`,
/// which it does not read. Refused by the link alone, which carries the
/// word. Control: the word taken out of the link, accepted.
#[test]
fn a_shared_word_not_the_previous_blocks_is_refused_by_the_link() {
    let mut run = Run::new();
    let row = run.call(IN + 5, 150, OUT).start + 1;
    let mask = run.ops[row].sponge.as_ref().expect("sponge").mask;
    assert!(!reads(&mask, 5, false)[0], "row 1 carries its first word");
    let forged = run.words(row).words[0] ^ 0x0101_0101_0101_0101;
    run.words_mut(row).words[0] = forged;
    // The bytes of W_0 the block places: 0..3 (the word's bytes 5..8).
    for i in 0..3 {
        run.msg_mut(row)[i] = (forged >> (8 * (i + 5))) as u8;
    }
    run.rechain(0);
    let bench = run.bench();
    let v = verdict(&bench);
    assert!(v.refused_only_on(BusId::KeccakSponge), "{v:?}");
    let v = verdict(&bench.with_interactions(keccak_w::NAME, link_without_the_word()));
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ A word placed without being read: the one row of a call clears `READ_1`
/// (its block places that word's bytes) and invents them. Refused by that
/// read rule alone. Control: the rule removed, accepted.
#[test]
fn a_word_placed_but_not_read_is_refused_by_its_read_rule() {
    let mut run = Run::new();
    let row = run.call(IN + 2, 30, OUT).start;
    run.unread.push((row, 1));
    let forged = 0x1112_1314_1516_1718u64;
    run.words_mut(row).words[1] = forged;
    // Block bytes 6..14 are W_1's (offset 2).
    for i in 6..14 {
        run.msg_mut(row)[i] = (forged >> (8 * (i + 2 - 8))) as u8;
    }
    run.rechain(0);
    let bench = run.bench();
    refused_by_constraint(&bench, keccak_w::READ_RULES + 1, row).unwrap_or_else(|v| panic!("{v}"));
    let without: Eval = keccak_w_without::<{ keccak_w::READ_RULES + 1 }>;
    let v = verdict(&with_constraints(bench, without));
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ A digest write that changes a byte it does not write: at offset 3 the
/// first digest word keeps its bytes 0..3 and the fifth its bytes 3..8; the
/// forger changes byte 1 of the first (its low half), then byte 6 of the
/// fifth (its high half). Refused by that half's identity alone. Control:
/// the identity removed, accepted.
#[test]
fn a_digest_write_changing_an_untouched_byte_is_refused_by_its_identity() {
    let cases: [(usize, usize, usize, Eval); 2] = [
        (
            0,
            1,
            keccak_w::DIGEST_HALVES,
            keccak_w_without::<{ keccak_w::DIGEST_HALVES }>,
        ),
        (
            4,
            6,
            keccak_w::DIGEST_HALVES + 9,
            keccak_w_without::<{ keccak_w::DIGEST_HALVES + 9 }>,
        ),
    ];
    for (v, byte, idx, without) in cases {
        let mut run = Run::new();
        let row = run.call(IN, 20, OUT + 3).start;
        let op = run.ops[row].clone();
        let d = run.words(row).digest.expect("the last row");
        let new = digest_words(&digest_of(&op.output), d.q, d.old0, d.old4);
        let forged = new[v] ^ (0x5Au64 << (8 * byte));
        let key = OUT + 8 * v as u64;
        let ts = (op.timestamp & 0xFFFF_FFFF) + 1;
        let mut bench = run.bench();
        let (lo, hi) = (FE::from(forged & 0xFFFF_FFFF), FE::from(forged >> 32));
        bench.set(keccak_w::NAME, row, kw::new(v, 0), lo);
        bench.set(keccak_w::NAME, row, kw::new(v, 1), hi);
        let n = bench.piece("MEMW_W").trace.num_rows();
        let m = (0..n)
            .find(|&m| {
                bench.get("MEMW_W", m, mw::ADDRESS_LO) == FE::from(key)
                    && bench.get("MEMW_W", m, mw::TIMESTAMP) == FE::from(ts)
            })
            .expect("the digest word's write");
        bench.set("MEMW_W", m, mw::NEW_LO, lo);
        bench.set("MEMW_W", m, mw::NEW_HI, hi);
        let name = page_name(0);
        let p = (key / WORD_BYTES) as usize;
        bench.set(&name, p, pw::FINI_LO, lo);
        bench.set(&name, p, pw::FINI_HI, hi);
        refused_by_constraint(&bench, idx, row).unwrap_or_else(|e| panic!("word {v}: {e}"));
        let ok = verdict(&with_constraints(bench, without));
        assert!(ok.accepted(), "control, word {v}: {ok:?}");
    }
}

/// ★ A call whose offset is not `src mod 8`: the forger claims 0 on an input
/// at `8m + 3` and reads its words at the unaligned keys `8m + 3` and
/// `8m + 11` (the bytes the input's, the block the call's). The reads
/// consume tokens nothing minted, refused on the word bus alone. Control:
/// PAGE_W's `OFFSET` the forger's (a main column, set to `m + 3/8` and
/// `m + 1 + 3/8`), which mints the two keys, accepted.
#[test]
fn a_call_with_a_wrong_offset_reads_words_nothing_minted() {
    let m = 0x1800u64 / WORD_BYTES;
    let ptr = 8 * m + 3;
    let mut run = Run::new();
    let row = run.call(ptr, 10, OUT).start;
    let words: [u64; BLOCK_WORDS] = std::array::from_fn(|w| run.word(ptr + 8 * w as u64));
    assert!(words[..2].iter().all(|&w| w == 0), "the zero page");
    let sw = run.words_mut(row);
    sw.offset = 0;
    sw.words = words;
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
    let ts = run.ops[row].timestamp & 0xFFFF_FFFF;
    for k in [m, m + 1] {
        let r = k as usize;
        control.set(&name, r, pw::OFFSET, FE::from(k) + FE::from(3u64) * eighth);
        control.set(&name, r, pw::TIMESTAMP, FE::from(ts));
    }
    let t = &control.piece(&name).trace;
    let column = (0..t.num_rows())
        .map(|r| *t.get_main(r, pw::OFFSET))
        .collect();
    control.piece_mut(&name).preprocessed[pw::OFFSET] = column;
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

// ---- the constraint programs ----

/// The BLAKE3 digest of an AIR's captured constraint program (its
/// constraints and its LogUp emission), as the recursion reads it.
fn program_digest(
    air: &dyn stark::traits::AIR<
        Field = super::types::GoldilocksField,
        FieldExtension = super::types::GoldilocksExtension,
        PublicInputs = (),
    >,
) -> String {
    let artifact = stark::constraint_ir::ConstraintArtifact::capture(air);
    let bytes = artifact.to_bytes().expect("the artifact serializes");
    blake3::hash(&bytes).to_hex().to_string()
}

/// ★ KECCAK's and the word chips' constraint programs, byte for byte: KECCAK
/// and KECCAK_W share their sponge rules' code, and a change to that code that
/// moves either program, or any of the word chips', fails here.
#[test]
fn the_keccak_and_word_chip_constraint_programs_are_pinned() {
    let opts = stark::proof::options::ProofOptions::default_test_options();
    let got = [
        (
            "KECCAK",
            program_digest(&crate::test_utils::create_keccak_air(&opts)),
        ),
        (
            keccak_w::NAME,
            program_digest(&crate::test_utils::create_keccak_w_air(&opts)),
        ),
        (
            super::copy_w::NAME,
            program_digest(&crate::test_utils::create_copy_w_air(&opts)),
        ),
        (
            super::ecrec_w::NAME,
            program_digest(&crate::test_utils::create_ecrec_w_air(&opts)),
        ),
    ];
    let pinned = [
        (
            "KECCAK",
            "f244f497cf3d56efe5c617c34f3813cd32bb0e71b5b9624a9427fbe89c55ac26",
        ),
        (
            keccak_w::NAME,
            "d9c8fa00644f00b1990f936258ccc816062d2eaeaa0a54469b6dba8bca70b0aa",
        ),
        (
            super::copy_w::NAME,
            "c871a28670857f9953c4bd9d00b2890faa956dd0cc6053d13551d36e78dc0433",
        ),
        (
            super::ecrec_w::NAME,
            "09715886702cb4ca241b602604dd01848d4399ebe0e041b09bd7bbb75083af53",
        ),
    ];
    for (name, digest) in &got {
        println!("PROGRAM {name} {digest}");
    }
    for ((name, digest), (pinned_name, pinned_digest)) in got.iter().zip(pinned) {
        assert_eq!((*name, digest.as_str()), (pinned_name, pinned_digest));
    }
}

// ---- KECCAK's sponge rules on KECCAK_W ----
//
// KECCAK_W emits KECCAK's sponge rules from KECCAK's own code
// (`keccak::emit_sponge_rules`), so a check a test leaves out
// (`keccak::leave_out`, thread-local) is left out of both tables. Each
// forgery below is KECCAK's forgery of that check laid out on KECCAK_W's
// bench: it commits another digest, is refused by that check's constraints
// alone, and is accepted with the check left out.

/// KECCAK_W's constraints `check` covers: on a row of arbitrary values, the
/// ones that read zero only with the check left out.
fn indices_of(check: SpongeCheck) -> Vec<usize> {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let row: Vec<FE> = (0..kw::NUM_COLUMNS)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            FE::from(x)
        })
        .collect();
    keccak::leave_out(&[]);
    let with = keccak_w_constraints(&row);
    keccak::leave_out(&[check]);
    let without = keccak_w_constraints(&row);
    keccak::leave_out(&[]);
    let out: Vec<usize> = (0..with.len())
        .filter(|&k| with[k] != FE::zero() && without[k] == FE::zero())
        .collect();
    assert!(!out.is_empty(), "{check:?} covers a KECCAK_W constraint");
    out
}

/// Only KECCAK_W constraints of `check` break, and nothing else refuses the
/// bench.
fn refused_by_check(bench: &Bench, check: SpongeCheck) -> Result<(), String> {
    keccak::leave_out(&[]);
    let v = verdict(bench);
    let ours = indices_of(check);
    if v.left.is_empty()
        && v.unanswered.is_empty()
        && v.preprocessed.is_empty()
        && !v.broken.is_empty()
        && v.broken
            .iter()
            .all(|(table, _, idx)| table == keccak_w::NAME && ours.contains(idx))
    {
        Ok(())
    } else {
        Err(format!("{check:?} ({ours:?}): {v:?}"))
    }
}

/// The bench's verdict with `checks` left out of KECCAK_W (and KECCAK).
fn verdict_without(bench: &Bench, checks: &[SpongeCheck]) -> super::word_memory_tests::Verdict {
    keccak::leave_out(checks);
    let v = verdict(bench);
    keccak::leave_out(&[]);
    v
}

/// The digest call `call` proves: its last row's.
fn proved_digest(run: &Run, call: usize) -> [u8; 32] {
    digest_of(&run.ops[run.calls[call].end - 1].output)
}

/// `forge` on a run of one call (`ptr`, `len`, digest to `OUT`) proves
/// another digest, is refused by `check` alone, and accepted without it.
fn sponge_rule_forgery(check: SpongeCheck, ptr: u64, len: u64, forge: impl Fn(&mut Run) -> Bench) {
    let honest = {
        let mut run = Run::new();
        run.call(ptr, len, OUT);
        assert!(
            verdict(&run.bench()).accepted(),
            "{check:?}: the honest call"
        );
        proved_digest(&run, 0)
    };
    let mut run = Run::new();
    run.call(ptr, len, OUT);
    let bench = forge(&mut run);
    assert_ne!(proved_digest(&run, 0), honest, "{check:?}: another digest");
    refused_by_check(&bench, check).unwrap_or_else(|e| panic!("{e}"));
    let v = verdict_without(&bench, &[check]);
    assert!(v.accepted(), "control, {check:?} left out: {v:?}");
}

/// ★ The prefix rule (`(1 − s_i)·s_{i+1} = 0`): a 20-byte call at offset 3
/// read with a hole at block byte 17 and byte 20 (past the input, in a word
/// the row reads) read instead. The count holds, the `0x01` lands in the
/// hole and byte 18 drops by one: the hash of another message.
#[test]
fn a_hole_in_the_read_mask_is_refused_by_the_prefix_rule() {
    sponge_rule_forgery(SpongeCheck::Prefix, IN + 3, 20, |run| {
        let row = run.calls[0].start;
        let past = run.byte(IN + 3 + 20);
        let s = run.ops[row].sponge.as_mut().expect("a sponge row");
        assert!(s.msg[18] >= 1, "byte 18 can drop by one");
        s.mask[17] = false;
        s.msg[17] = 0;
        s.mask[20] = true;
        s.msg[20] = past;
        run.rechain(0);
        run.bench()
    });
}

/// ★ The count on the last row (`LAST·(COUNT − Σ s_i) = 0`): a 137-byte call
/// at offset 5 whose last block reads none of its one byte: the hash of the
/// first 136 bytes.
#[test]
fn a_last_block_reading_less_than_its_count_is_refused_by_the_count_rule() {
    sponge_rule_forgery(SpongeCheck::CountOnLast, IN + 5, 137, |run| {
        let row = run.calls[0].start + 1;
        let s = run.ops[row].sponge.as_mut().expect("a sponge row");
        s.mask = keccak::SpongeRow::prefix(0);
        s.msg = [0; 136];
        run.rechain(0);
        run.bench()
    });
}

/// ★ The last block holds at most 135 bytes (`LAST·FULL_16 = 0`): a 136-byte
/// aligned call as one full last block, its padding folded into it: one
/// permutation where Keccak takes two.
#[test]
fn a_full_last_block_is_refused_by_the_last_block_bound() {
    let ptr = (0..64u64)
        .map(|m| IN + 8 * m)
        .find(|&p| Run::new().byte(p + 135) < 0x80)
        .expect("a block whose last byte leaves room for 0x80");
    sponge_rule_forgery(SpongeCheck::LastNotFull, ptr, 136, |run| {
        let digest = run.words(1).digest;
        run.ops.truncate(1);
        run.calls[0] = 0..1;
        run.ops[0].sponge.as_mut().expect("a sponge row").last = true;
        run.words_mut(0).digest = digest;
        run.rechain(0);
        run.bench()
    });
}

/// ★ A row that is not last reads a whole block (`(MU − LAST)(1 − FULL_16)
/// = 0`): block 1 of an aligned 600-byte call reads 135 bytes, so a `0x01`
/// lands inside the input.
#[test]
fn a_short_middle_block_is_refused_by_the_whole_block_rule() {
    sponge_rule_forgery(SpongeCheck::FullUnlessLast, IN, 600, |run| {
        let row = run.calls[0].start + 1;
        let s = run.ops[row].sponge.as_mut().expect("a sponge row");
        s.mask = keccak::SpongeRow::prefix(135);
        s.msg[135] = 0;
        run.rechain(0);
        run.bench()
    });
}

/// ★ The zero first state (`FIRST·PREV_i = 0`, `FIRST·INPUT_i = 0` on the
/// capacity): a one-block call whose permutation starts from a state with
/// one rate bit set and one capacity byte set (`PREV_0 = 1`, `INPUT_150 =
/// 0x5a`): a digest that is not the input's.
#[test]
fn a_call_from_a_nonzero_state_is_refused_by_the_zero_first_state() {
    use executor::vm::instruction::execution::keccak_f1600;
    sponge_rule_forgery(SpongeCheck::FirstState, IN, 20, |run| {
        let row = run.calls[0].start;
        let mut state = run.ops[row].input;
        state[0] ^= 1;
        state[150 / 8] ^= 0x5a << (8 * (150 % 8));
        run.ops[row].input = state;
        keccak_f1600(&mut state);
        run.ops[row].output = state;
        let bench = run.bench();
        assert_eq!(bench.get(keccak_w::NAME, row, kw::prev(0)), FE::one());
        bench
    });
}

/// ★ The block end is the next block's address (`MU·(SRC_LO + 135 −
/// SRC_END) = 0`, the link sending `SRC_END + 1`): blocks 1 and 2 of an
/// aligned 600-byte call swapped, each row's `SRC_END` set to the next row's
/// address − 1, and each row's unread last word the next row's first, so
/// the link holds. The ends' halves are the honest ones permuted.
#[test]
fn swapped_blocks_are_refused_by_the_block_end() {
    sponge_rule_forgery(SpongeCheck::AddressHalves, IN, 600, |run| {
        let r = run.calls[0].start;
        let (a, b) = (r + 1, r + 2);
        let (sa, sb) = (
            run.ops[a].sponge.clone().expect("a sponge row"),
            run.ops[b].sponge.clone().expect("a sponge row"),
        );
        let (wa, wb) = (run.words(a).words, run.words(b).words);
        {
            let s = run.ops[a].sponge.as_mut().expect("a sponge row");
            s.src = sb.src;
            s.msg = sb.msg;
        }
        {
            let s = run.ops[b].sponge.as_mut().expect("a sponge row");
            s.src = sa.src;
            s.msg = sa.msg;
        }
        run.words_mut(a).words = wb;
        run.words_mut(b).words = wa;
        for j in r..r + 3 {
            assert!(!reads(&run.ops[j].sponge.as_ref().expect("sponge").mask, 0, j == r)[17]);
            let next = run.words(j + 1).words[0];
            run.words_mut(j).words[17] = next;
        }
        run.rechain(0);
        let mut bench = run.bench();
        for j in r..r + 3 {
            let end = run.ops[j + 1].sponge.as_ref().expect("sponge").src - 1;
            bench.set(keccak_w::NAME, j, kw::SRC_END, FE::from(end & 0xFFFF));
            bench.set(
                keccak_w::NAME,
                j,
                kw::SRC_END + 1,
                FE::from((end >> 16) & 0xFFFF),
            );
        }
        bench
    });
}

/// KECCAK_W's rows of a 141-byte aligned call (two blocks, the second
/// reading five bytes, all in lane 0) and two padding rows:
/// `(rows, [first, last, padding])`.
fn hand_built_rows() -> (Vec<Vec<FE>>, [usize; 3]) {
    let mut run = Run::new();
    run.call(IN, 141, OUT);
    let trace = keccak_w::generate_keccak_w_trace_as(&run.ops, TraceForm::Wide);
    let rows: Vec<Vec<FE>> = (0..trace.num_rows())
        .map(|r| {
            (0..kw::NUM_COLUMNS)
                .map(|c| *trace.get_main(r, c))
                .collect()
        })
        .collect();
    (rows, [0, 1, 2])
}

/// KECCAK_W's constraints that do not read zero on `row`, with this thread's
/// checks left out.
fn failing(row: &[FE]) -> Vec<usize> {
    let v = keccak_w_constraints(row);
    (0..v.len()).filter(|&k| v[k] != FE::zero()).collect()
}

/// `edit` on row `at` breaks `check`'s constraints alone; with `check` left
/// out the row meets every constraint.
fn row_refused_only_by(check: SpongeCheck, at: usize, edit: impl Fn(&mut Vec<FE>)) {
    let (rows, _) = hand_built_rows();
    keccak::leave_out(&[]);
    for (r, row) in rows.iter().enumerate() {
        assert_eq!(failing(row), Vec::<usize>::new(), "honest row {r}");
    }
    let mut row = rows[at].clone();
    edit(&mut row);
    let broken = failing(&row);
    let ours = indices_of(check);
    assert!(
        !broken.is_empty() && broken.iter().all(|k| ours.contains(k)),
        "{check:?}: refused by its constraints alone: {broken:?} (its: {ours:?})"
    );
    keccak::leave_out(&[check]);
    let without = failing(&row);
    keccak::leave_out(&[]);
    assert_eq!(without, Vec::<usize>::new(), "{check:?}: with it left out");
}

/// ★ `(FIRST + LAST)(1 − MU) = 0`: a padding row claiming to start a call.
#[test]
fn a_first_flag_off_a_row_is_refused_only_by_its_check() {
    let [_, _, padding] = hand_built_rows().1;
    row_refused_only_by(SpongeCheck::FlagsOnSponge, padding, |row| {
        row[kw::FIRST] = FE::one();
    });
}

/// ★ `(1 − MU)·s_0 = 0`: a padding row reading lane 0 (the prefix rule
/// holds; its offsets are all zero, so it places nothing and reads no word).
#[test]
fn a_read_off_a_row_is_refused_only_by_its_check() {
    let [_, _, padding] = hand_built_rows().1;
    row_refused_only_by(SpongeCheck::ReadsOnSponge, padding, |row| {
        row[kw::full(0)] = FE::one();
    });
}

/// ★ `FIRST·TIMESTAMP_1 = 0`: the first row claiming block 1.
#[test]
fn a_first_row_off_block_zero_is_refused_only_by_its_rule() {
    let [first, ..] = hand_built_rows().1;
    row_refused_only_by(SpongeCheck::FirstBlock, first, |row| {
        row[kw::TIMESTAMP_1] = FE::one();
    });
}

/// ★ The `PART` bits: the last block's last read byte counted twice (`PART =
/// 2`), `COUNT` one higher and that byte's `MSG` twice its word byte, so the
/// count, the prefix rule and the placement all hold.
#[test]
fn a_part_bit_of_two_is_refused_only_by_the_mask_bits() {
    let [_, last, _] = hand_built_rows().1;
    row_refused_only_by(SpongeCheck::MaskBits, last, |row| {
        assert_eq!(row[kw::part(0, 4)], FE::one(), "byte 4 is read");
        assert_eq!(row[kw::part(0, 5)], FE::zero(), "byte 5 is not");
        row[kw::part(0, 4)] = FE::from(2u64);
        row[kw::COUNT] += FE::one();
        row[kw::msg(4)] = FE::from(2u64) * row[kw::w(4)];
    });
}

// ---- the link's fields ----

/// The link's tuple: `(T, j, COUNT, SRC_LO, SRC_HI, OUT_LO, OUT_HI)`, the 200
/// state bytes, the offset, the carried word's eight bytes.
const LINK_COUNT: usize = 2;
const LINK_SRC_LO: usize = 3;
const LINK_OUT_LO: usize = 5;
const LINK_OFFSET: usize = 7 + 200;
const LINK_LEN: usize = LINK_OFFSET + 1 + 8;

/// KECCAK_W's interactions with value `p` of the link (both sides) the
/// constant 0: a control mutant.
fn link_without(p: usize) -> Vec<BusInteraction> {
    let mut out = keccak_w::bus_interactions();
    let mut sides = 0;
    for bi in &mut out {
        if bi.bus_id == u64::from(BusId::KeccakSponge) {
            assert_eq!(bi.values.len(), LINK_LEN, "the link's tuple");
            bi.values[p] = BusValue::constant(0);
            sides += 1;
        }
    }
    assert_eq!(sides, 2, "the link's two sides");
    out
}

/// `forge` on a run of one call proves another digest (unless
/// `same_digest`), is refused by the link alone, and accepted with link
/// value `p` out.
fn link_forgery(p: usize, ptr: u64, len: u64, same_digest: bool, forge: impl Fn(&mut Run)) {
    let honest = {
        let mut run = Run::new();
        run.call(ptr, len, OUT);
        proved_digest(&run, 0)
    };
    let mut run = Run::new();
    run.call(ptr, len, OUT);
    forge(&mut run);
    assert_eq!(
        proved_digest(&run, 0) == honest,
        same_digest,
        "value {p}: the digest"
    );
    let bench = run.bench();
    let v = verdict(&bench);
    assert!(v.refused_only_on(BusId::KeccakSponge), "value {p}: {v:?}");
    let v = verdict(&bench.with_interactions(keccak_w::NAME, link_without(p)));
    assert!(v.accepted(), "control, value {p} out: {v:?}");
}

/// ★ The offset rides the link: a 138-byte call at offset 5 whose second
/// block (two input bytes, both in the carried word, so the row reads no
/// word) takes offset 3, placing the carried word's bytes 3 and 4 (the
/// first block's last two) instead of 5 and 6: another message's digest.
#[test]
fn an_offset_changed_mid_call_is_refused_by_the_link() {
    link_forgery(LINK_OFFSET, IN + 5, 138, false, |run| {
        let row = run.calls[0].start + 1;
        let mask = run.ops[row].sponge.as_ref().expect("sponge").mask;
        assert!(reads(&mask, 3, false).iter().all(|r| !r), "nothing read");
        let w0 = run.words(row).words[0];
        run.words_mut(row).offset = 3;
        run.msg_mut(row)[0] = (w0 >> 24) as u8;
        run.msg_mut(row)[1] = (w0 >> 32) as u8;
        run.rechain(0);
    });
}

/// ★ The count rides the link: a 186-byte call whose second block takes 10
/// bytes, not 50: the digest of the first 146 bytes.
#[test]
fn a_count_changed_mid_call_is_refused_by_the_link() {
    link_forgery(LINK_COUNT, IN, 186, false, |run| {
        let row = run.calls[0].start + 1;
        let s = run.ops[row].sponge.as_mut().expect("sponge");
        s.count = 10;
        s.mask = keccak::SpongeRow::prefix(10);
        s.msg[10..].fill(0);
        run.rechain(0);
    });
}

/// ★ The block's address rides the link: an aligned 186-byte call whose
/// second block is read 0x108 further on (the first block's unread last
/// word the forger's): bytes that are not the input's hashed.
#[test]
fn a_block_read_from_elsewhere_mid_call_is_refused_by_the_link() {
    link_forgery(LINK_SRC_LO, IN, 186, false, |run| {
        let first = run.calls[0].start;
        let row = first + 1;
        let src = run.ops[row].sponge.as_ref().expect("sponge").src + 0x108;
        let words: [u64; BLOCK_WORDS] = std::array::from_fn(|w| run.word(src + 8 * w as u64));
        {
            let s = run.ops[row].sponge.as_mut().expect("sponge");
            s.src = src;
            for i in 0..50 {
                s.msg[i] = (words[i / 8] >> (8 * (i % 8))) as u8;
            }
        }
        run.words_mut(row).words = words;
        run.words_mut(first).words[17] = words[0];
        run.rechain(0);
    });
}

/// ★ The digest's address rides the link: a two-block call whose last row
/// writes the digest 0x100 past `out` (the same digest, written elsewhere,
/// `out` untouched).
#[test]
fn a_digest_written_elsewhere_mid_call_is_refused_by_the_link() {
    link_forgery(LINK_OUT_LO, IN, 186, true, |run| {
        let row = run.calls[0].start + 1;
        run.ops[row].sponge.as_mut().expect("sponge").out += 0x100;
    });
}

// ---- the absorb ----

/// ★ Every absorb lookup is load-bearing: for each rate byte `i`, a one-block
/// call whose permutation input differs from the padded block in one bit of
/// byte `i` (its `PREV_i` kept 0, so the zero first state holds) is refused
/// by byte `i`'s XOR lookup alone, and accepted with that lookup out: a
/// digest that is not the input's.
#[test]
fn every_absorb_lookup_refuses_a_permutation_input_not_the_block() {
    use executor::vm::instruction::execution::keccak_f1600;
    let honest = {
        let mut run = Run::new();
        run.call(IN, 20, OUT);
        proved_digest(&run, 0)
    };
    for i in 0..keccak_w::RATE {
        let mut run = Run::new();
        let row = run.call(IN, 20, OUT).start;
        let mut state = run.ops[row].input;
        state[i / 8] ^= 1 << (8 * (i % 8));
        run.ops[row].input = state;
        keccak_f1600(&mut state);
        run.ops[row].output = state;
        assert_ne!(proved_digest(&run, 0), honest, "byte {i}");
        let mut bench = run.bench();
        bench.set(keccak_w::NAME, row, kw::prev(i), FE::zero());
        let v = verdict(&bench);
        assert!(
            v.left.is_empty()
                && v.broken.is_empty()
                && v.preprocessed.is_empty()
                && v.unanswered.len() == 1
                && v.unanswered[0].0 == u64::from(BusId::ByteAlu),
            "byte {i}: {v:?}"
        );
        let mut control = keccak_w::bus_interactions();
        let mut off = 0;
        for bi in &mut control {
            if bi.bus_id == u64::from(BusId::ByteAlu)
                && bi.values.get(3) == Some(&column(kw::input_byte(i)))
            {
                bi.multiplicity =
                    Multiplicity::Linear(vec![stark::lookup::LinearTerm::Constant(0)]);
                off += 1;
            }
        }
        assert_eq!(off, 1, "byte {i}: one XOR lookup");
        let v = verdict(&bench.with_interactions(keccak_w::NAME, control));
        assert!(v.accepted(), "control, byte {i}: {v:?}");
    }
}

// ---- the words' byte pairs ----

/// ★ A word whose bytes are not all checked lets a byte move between an
/// unused and a used position: a 10-byte call at offset 3, whose first word
/// holds three bytes below the input and five input bytes. The forger moves
/// one unit of the word's value from input byte 0 (the word's byte 3, one
/// less) to the unused byte 2 (256 more): the word read is memory's, the
/// byte hashed is not. Refused by that pair's `ARE_BYTES` alone; accepted
/// with the pair out: the digest of another message.
#[test]
fn an_edge_word_split_between_an_unused_and_an_input_byte_is_refused_by_its_pair() {
    let honest = {
        let mut run = Run::new();
        run.call(IN + 3, 10, OUT);
        proved_digest(&run, 0)
    };
    let mut run = Run::new();
    let row = run.call(IN + 3, 10, OUT).start;
    let w0 = run.words(row).words[0];
    let (b2, b3) = ((w0 >> 16) & 0xFF, (w0 >> 24) & 0xFF);
    assert!(b3 >= 1, "an input byte to take one from: {b3}");
    assert_eq!(u64::from(run.msg_mut(row)[0]), b3);
    run.msg_mut(row)[0] = (b3 - 1) as u8;
    run.rechain(0);
    assert_ne!(proved_digest(&run, 0), honest);
    let mut bench = run.bench();
    bench.set(keccak_w::NAME, row, kw::w(2), FE::from(b2 + 256));
    bench.set(keccak_w::NAME, row, kw::w(3), FE::from(b3 - 1));
    let v = verdict(&bench);
    assert!(
        v.left.is_empty()
            && v.broken.is_empty()
            && v.preprocessed.is_empty()
            && v.unanswered == vec![(u64::from(BusId::AreBytes), vec![b2 + 256, b3 - 1])],
        "{v:?}"
    );
    let control: Vec<BusInteraction> = keccak_w::bus_interactions()
        .into_iter()
        .filter(|bi| {
            !(bi.bus_id == u64::from(BusId::AreBytes)
                && bi.values.first() == Some(&column(kw::w(2))))
        })
        .collect();
    assert_eq!(control.len() + 1, keccak_w::bus_interactions().len());
    let v = verdict(&bench.with_interactions(keccak_w::NAME, control));
    assert!(v.accepted(), "control: {v:?}");
}

// ---- the lone-break search ----

/// The distinct rows of `run`'s KECCAK_W table (padding included).
fn distinct_rows(run: &Run) -> Vec<Vec<FE>> {
    let t = keccak_w::generate_keccak_w_trace_as(&run.ops, TraceForm::Wide);
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for r in 0..t.num_rows() {
        let row: Vec<FE> = (0..kw::NUM_COLUMNS).map(|c| *t.get_main(r, c)).collect();
        let key: Vec<u64> = row.iter().map(|v| v.canonical_u64()).collect();
        if seen.insert(key) {
            out.push(row);
        }
    }
    out
}

/// The mask `s_i` on `row`.
fn mask_of(row: &[FE], i: usize) -> FE {
    let (lane, byte) = (i / 8, i % 8);
    let full = row[kw::full(lane)];
    if byte < 7 {
        full + row[kw::part(lane, byte)]
    } else {
        full
    }
}

/// `row` with the columns KECCAK_W defines by an identity recomputed from
/// the rest (`MSG`, `READ`, `W4`, `NEW`; the keys' and ends' low halves on a
/// row, `COUNT` on the last): an edit to the other columns then breaks only
/// the constraints that read them otherwise.
fn completed(mut row: Vec<FE>) -> Vec<FE> {
    let one = FE::one();
    let two16 = FE::from(1u64 << 16);
    let weighted = |row: &[FE], start: usize| {
        (1..8).fold(FE::zero(), |a, k| a + FE::from(k as u64) * row[start + k])
    };
    for i in 0..keccak_w::RATE {
        let placed = (0..8).fold(FE::zero(), |a, k| a + row[kw::off(k)] * row[kw::w(i + k)]);
        row[kw::msg(i)] = mask_of(&row, i) * placed;
    }
    let first = row[kw::FIRST];
    for w in 0..BLOCK_WORDS {
        row[kw::read(w)] = match w {
            0 => mask_of(&row, 0) * (first + (one - first) * row[kw::off(0)]),
            17 => (1..8).fold(FE::zero(), |a, k| {
                a + row[kw::off(k)] * mask_of(&row, keccak_w::RATE - k)
            }),
            _ => (0..8).fold(FE::zero(), |a, k| {
                a + row[kw::off(k)] * mask_of(&row, 8 * w - k)
            }),
        };
    }
    if row[kw::MU] != FE::zero() {
        row[kw::SRC_KEY] = row[kw::SRC_LO] - weighted(&row, kw::OFF) - two16 * row[kw::SRC_KEY + 1];
        row[kw::SRC_END] = row[kw::SRC_LO] + FE::from(135u64) - two16 * row[kw::SRC_END + 1];
    }
    let last = row[kw::LAST];
    if last != FE::zero() {
        row[kw::OUT_KEY] = row[kw::OUT_LO] - weighted(&row, kw::Q) - two16 * row[kw::OUT_KEY + 1];
        row[kw::OUT_END] = row[kw::OUT_LO] + FE::from(31u64) - two16 * row[kw::OUT_END + 1];
        row[kw::COUNT] = (0..keccak_w::RATE).fold(FE::zero(), |a, i| a + mask_of(&row, i));
    }
    row[kw::W4] = last * (one - row[kw::q(0)]);
    for v in 0..keccak_w::DIGEST_WORDS {
        for half in 0..2 {
            let mut acc = FE::zero();
            for jj in 0..4 {
                let j = 4 * half + jj;
                let at = 8 * v + j;
                let placed = (0..8)
                    .filter(|&q| at >= q && at - q < keccak_w::DIGEST)
                    .fold(FE::zero(), |a, q| {
                        a + row[kw::q(q)] * row[kw::output_byte(at - q)]
                    });
                let kept = |old: usize, qs: &mut dyn Iterator<Item = usize>| {
                    qs.fold(FE::zero(), |a, q| a + row[kw::q(q)]) * row[old + j]
                };
                let byte = match v {
                    0 => placed + kept(kw::OLD0, &mut (0..8).filter(|&q| q > j)),
                    4 => placed + kept(kw::OLD4, &mut (0..8).filter(|&q| q <= j)),
                    _ => placed,
                };
                acc += FE::from(1u64 << (8 * jj)) * byte;
            }
            row[kw::new(v, half)] = acc;
        }
    }
    row
}

/// The constraints no row breaks alone, because the rest imply them (kept as
/// written; each is still one of a pair or trio of which the rest need one):
/// - 0, `MU` a bit: `(1 − MU)·s_0 = 0` makes `MU = 1` when the row reads, and
///   a row that reads nothing has `FULL_16 = 0`, so `(MU − LAST)(1 − FULL_16)
///   = 0` makes `MU = LAST`, a bit;
/// - 2, `LAST` a bit: with `MU` a bit, `(MU − LAST)(1 − FULL_16) = 0` and
///   `LAST·FULL_16 = 0` leave `LAST` 0 or `MU`, and `MU = 0` with
///   `FULL_16 = 1` is a read off a row;
/// - each `FULL_k` a bit: `s_{8k+7} = FULL_k`, and `(1 − s_{8k+6})·s_{8k+7} = 0`
///   with `FULL_k·PART_{k,6} = 0` leave `FULL_k` 0 or 1;
/// - each `FULL_k·PART_{k,b} = 0`: both 1 puts `s_{8k+b} = 2` before a byte
///   of at least `FULL_k = 1`, which the prefix rule at `8k + b` refuses;
/// - the prefix rule at `8k + 6`: `s_{8k+7} = FULL_k`, and `FULL_k = 1` with
///   `FULL_k·PART_{k,6} = 0` puts `s_{8k+6} = 1`.
fn implied_keccak_w_constraints() -> Vec<usize> {
    let mask_bits = 4;
    let one_read = mask_bits + 8 * keccak_w::RATE_LANES;
    let prefix = one_read + 7 * keccak_w::RATE_LANES;
    let mut out = vec![0, 2];
    out.extend((0..keccak_w::RATE_LANES).map(|k| mask_bits + 8 * k));
    out.extend(one_read..prefix);
    out.extend((0..keccak_w::RATE_LANES).map(|k| prefix + 8 * k + 6));
    out.sort_unstable();
    out
}

/// ★ Every one of KECCAK_W's 786 constraints but the implied ones
/// ([`implied_keccak_w_constraints`]) is one a row can break alone: for each,
/// from an honest row (calls at every offset, the last block at every length
/// and the digest at every offset, middle rows, padding), an edit breaks that
/// constraint and no other — one column set to 0, 1, 2, −1 or its own plus
/// one; one offset's one-hot bit up one and another down one (its sum kept);
/// or a row that reads a whole block made the last (`LAST` and `Q_0` set) —
/// raw, or with the columns an identity defines recomputed from the rest. So
/// each is load-bearing at the row: taken out, a row the table refused meets
/// every other. KECCAK's sponge rules among them are KECCAK's own code.
#[test]
fn every_keccak_w_constraint_breaks_alone_on_some_row() {
    let mut run = Run::new();
    for offset in 0..8u64 {
        for (k, &len) in [0u64, 1, 7, 13, 135, 136, 137, 150, 271].iter().enumerate() {
            run.call(
                IN + offset + 0x200 * k as u64,
                len,
                OUT + 0x40 * k as u64 + offset,
            );
        }
    }
    // The last block's last read byte at every position.
    for len in 1..keccak_w::RATE as u64 {
        run.call(IN + 0x200, len, OUT);
    }
    let rows = distinct_rows(&run);
    keccak::leave_out(&[]);
    for (r, row) in rows.iter().enumerate() {
        assert_eq!(failing(row), Vec::<usize>::new(), "honest row {r}");
        assert_eq!(&completed(row.clone()), row, "an honest row is complete");
    }
    let mut found = vec![false; keccak_w::NUM_CONSTRAINTS];
    let mut note = |row: &[FE]| {
        if let [k] = failing(row).as_slice() {
            found[*k] = true;
        }
    };
    let (one, minus_one) = (FE::one(), FE::zero() - FE::one());
    for row in &rows {
        for c in 0..kw::NUM_COLUMNS {
            for v in [FE::zero(), one, FE::from(2u64), minus_one, row[c] + one] {
                let mut edited = row.clone();
                edited[c] = v;
                note(&edited);
                note(&completed(edited));
            }
        }
        for start in [kw::OFF, kw::Q] {
            for a in 0..8 {
                for b in (0..8).filter(|&b| b != a) {
                    let mut edited = row.clone();
                    edited[start + a] += one;
                    edited[start + b] += minus_one;
                    note(&edited);
                    note(&completed(edited));
                }
            }
        }
        if row[kw::MU] == one && row[kw::LAST] == FE::zero() {
            let mut edited = row.clone();
            edited[kw::LAST] = one;
            edited[kw::q(0)] = one;
            note(&completed(edited));
        }
    }
    let missing: Vec<usize> = (0..found.len()).filter(|&k| !found[k]).collect();
    assert_eq!(
        missing,
        implied_keccak_w_constraints(),
        "the constraints no lone break was found for"
    );
}

/// KECCAK_W's honest rows for the census (`word_memory_tests`): calls at
/// every input and digest offset, the last block at every length, middle
/// rows and padding.
pub(super) fn census_rows() -> Vec<Vec<FE>> {
    let mut run = Run::new();
    for offset in 0..8u64 {
        for (k, &len) in [0u64, 1, 13, 136, 137, 271].iter().enumerate() {
            run.call(
                IN + offset + 0x200 * k as u64,
                len,
                OUT + 0x40 * k as u64 + offset,
            );
        }
    }
    distinct_rows(&run)
}

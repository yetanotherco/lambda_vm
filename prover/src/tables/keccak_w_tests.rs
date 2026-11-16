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
use super::keccak::{self, KeccakOperation};
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

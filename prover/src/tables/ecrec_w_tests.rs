//! ★ ECREC_W (`tables::ecrec_w`) on hand-built traces with no proof, on the
//! word format's bench (`tables::word_memory_tests`): PAGE_W, MEMW_W, one
//! ECREC_W row (a bail: the row reads the signature and the hash and writes
//! the zero key with status 1, every other interaction gated off), a stand-in
//! CPU and register file whose values are the verifier's.
//!
//! The executor refuses an ECREC call with an unaligned operand: a
//! completeness rule on the guest. A forger skips the executor and lays out a
//! call whose signature sits at `8k + 4`: its word messages name keys no
//! PAGE_W row mints, and the word bus refuses it. The two controls take out,
//! each, one of the conditions that argument rests on: PAGE_W's `OFFSET` the
//! forger's (minting at `8k + 4 + 8i`), and the step without its `−1` (each
//! forged access consuming its own token).

use std::collections::HashMap;

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity};

use super::ecrec::{self, EcrecOperation};
use super::ecrec_w::{self, EcrecWordConstraints};
use super::gpack::TraceForm;
use super::memw_w::{self, MemwWordConstraints, MemwWordRow, MessageKind, cols as mw};
use super::page::{DEFAULT_PAGE_SIZE, PageConfig};
use super::page_w::{self, FinalWordMap, FinalWordState, cols as pw};
use super::types::{BusId, FE};
use super::word_memory::{WORD_BYTES, column, term};
use super::word_memory_tests::{
    Bench, Piece, Table, constraint_values, multiplicity, page_name, page_piece, pages, table,
    verdict,
};
use executor::vm::instruction::execution::ECREC_SYSCALL_NUMBER;

fn ecrec_w_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&EcrecWordConstraints, row)
}

fn memw_w_constraints(row: &[FE]) -> Vec<FE> {
    constraint_values(&MemwWordConstraints, row)
}

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

const TS: u64 = 40;
const OUT: u64 = 0x3000;
const MSG: u64 = 0x2000;

/// A bail row (r = 0) at `sig` (the zero page's bytes), as the walk makes
/// one: the key zero, status 1.
fn bail_op(sig: u64) -> EcrecOperation {
    EcrecOperation {
        timestamp: TS,
        addr_out: OUT,
        addr_sig: sig,
        addr_msg: MSG,
        witness: ecsm::compute_ecrec_witness(&[0u8; 64], &[0u8; 32], 0),
        pending: None,
        words: true,
    }
}

/// The word messages the row sends, read off its interactions: `(kind, key,
/// timestamp, old, new)` (a write's old is memory's).
fn messages(row: &[FE]) -> Vec<(MessageKind, u64, u64, u64, u64)> {
    let g = |c: usize| row[c];
    let mut out = Vec::new();
    for bi in ecrec_w::bus_interactions() {
        if bi.bus_id != u64::from(BusId::MemWordMsg)
            || multiplicity(&bi.multiplicity, &g) == FE::zero()
        {
            continue;
        }
        let v: Vec<u64> = bi
            .values
            .iter()
            .flat_map(|v| v.combine_from(g))
            .map(|x: FE| x.canonical_u64())
            .collect();
        let word = |lo: u64, hi: u64| lo | hi << 32;
        if v[0] == 1 {
            out.push((
                MessageKind::Write,
                word(v[1], v[2]),
                v[3],
                0,
                word(v[4], v[5]),
            ));
        } else {
            out.push((
                MessageKind::Access,
                word(v[1], v[2]),
                v[3],
                word(v[4], v[5]),
                word(v[6], v[7]),
            ));
        }
    }
    out
}

/// The bench of one bail call with its signature at `sig`: the MEMW_W rows
/// of its messages replayed on the pages (an unaligned key starts from the
/// message's own old word at 0), PAGE_W's finals, the CPU's call and the
/// register file's four reads.
fn bench(sig: u64) -> Bench {
    let pages: Vec<PageConfig> = pages();
    let ecrec = ecrec_w::generate_ecrec_w_trace(&[bail_op(sig)]);
    let row: Vec<FE> = (0..ecrec_w::cols::NUM_COLUMNS)
        .map(|c| *ecrec.get_main(0, c))
        .collect();
    let mut state: HashMap<u64, (u64, u64)> = HashMap::new();
    let mut memw = Vec::new();
    let mut keys = Vec::new();
    let mut msgs = messages(&row);
    msgs.sort_by_key(|m| m.2);
    for (kind, key, ts, old, new) in msgs {
        let (cur, cur_ts) = *state.entry(key).or_insert_with(|| {
            if key % WORD_BYTES == 0 {
                let page = pages
                    .iter()
                    .find(|p| {
                        key.checked_sub(p.page_base)
                            .is_some_and(|d| d < DEFAULT_PAGE_SIZE as u64)
                    })
                    .expect("a page");
                let w = page_w::init_word(page, ((key - page.page_base) / WORD_BYTES) as usize);
                (w, 0)
            } else {
                (old, 0)
            }
        });
        let old = if kind == MessageKind::Write { cur } else { old };
        memw.push(
            MemwWordRow::new(kind, key - key % WORD_BYTES, ts, cur_ts, old, new)
                .expect("a replayed access"),
        );
        keys.push(key);
        state.insert(key, (new, ts));
    }
    let finals: FinalWordMap = state
        .into_iter()
        .filter(|(key, _)| key % WORD_BYTES == 0)
        .map(|(key, (value, timestamp))| (key, FinalWordState { timestamp, value }))
        .collect();
    let mut pieces: Vec<Piece> = pages.iter().map(|c| page_piece(c, &finals)).collect();
    let mut trace = memw_w::generate_memw_w_trace_segments_as(&[&memw], TraceForm::Wide);
    for (i, key) in keys.iter().enumerate() {
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
        name: ecrec_w::NAME.into(),
        interactions: ecrec_w::bus_interactions(),
        trace: ecrec,
        preprocessed: Vec::new(),
        constraints: Some(ecrec_w_constraints),
    });
    let calls = table(1, cpu::NUM_COLUMNS, |t, row| {
        t.main_table.set(row, cpu::TS, FE::from(TS));
        t.main_table.set(row, cpu::MU, FE::one());
    });
    pieces.push(truth_piece(
        "CPU",
        vec![BusInteraction::sender(
            BusId::Ecall,
            Multiplicity::Column(cpu::MU),
            vec![
                column(cpu::TS),
                BusValue::constant(0),
                BusValue::constant(ECREC_SYSCALL_NUMBER & 0xFFFF_FFFF),
                BusValue::constant(ECREC_SYSCALL_NUMBER >> 32),
            ],
        )],
        calls,
        cpu::NUM_COLUMNS,
    ));
    let reads = [
        (11u64, sig, TS),
        (12, MSG, TS + 1),
        (13, 0, TS + 1),
        (10, OUT, TS + 2),
    ];
    let regs = table(reads.len(), regs::NUM_COLUMNS, |t, row| {
        let (reg, value, ts) = reads[row];
        t.main_table.set(row, regs::REG2, FE::from(2 * reg));
        t.main_table
            .set(row, regs::LO, FE::from(value & 0xFFFF_FFFF));
        t.main_table.set(row, regs::HI, FE::from(value >> 32));
        t.main_table.set(row, regs::TS, FE::from(ts));
        t.main_table.set(row, regs::MU, FE::one());
    });
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
    pieces.push(truth_piece(
        "REGS",
        vec![BusInteraction::receiver(
            BusId::Memw,
            Multiplicity::Column(regs::MU),
            tuple,
        )],
        regs,
        regs::MU,
    ));
    Bench { pieces }
}

/// ★ A bail call at aligned operands is accepted: its 21 word messages (the
/// signature's eight reads at `T`, the hash's four at `T + 1`, the key's and
/// status's nine writes at `T + 2`) balance against MEMW_W and PAGE_W, its
/// bytes are checked, its registers answered.
#[test]
fn an_aligned_ecrec_call_is_accepted() {
    let b = bench(0x1000);
    let v = verdict(&b);
    assert!(v.accepted(), "{v:?}");
    let messages = (0..b.piece("MEMW_W").trace.num_rows())
        .filter(|&r| {
            b.get("MEMW_W", r, mw::MU_WRITE) != FE::zero()
                || b.get("MEMW_W", r, mw::MU_ACCESS) != FE::zero()
        })
        .count();
    assert_eq!(messages, 21);
}

/// MEMW_W's interactions with the consumed token's timestamp missing its
/// `−1`: `TIMESTAMP − STEP_0 − 2^16·STEP_1`.
fn memw_w_without_the_minus_one() -> Vec<BusInteraction> {
    let mut out = memw_w::bus_interactions();
    out[2].values[2] = BusValue::linear(vec![
        term(1, mw::TIMESTAMP),
        term(-1, mw::STEP_0),
        LinearTerm::Column {
            coefficient: -(1 << 16),
            column: mw::STEP_1,
        },
    ]);
    out
}

/// ★ A forger's ECREC call with its signature at `8k + 4`, which the executor
/// refuses and the forger lays out by hand: its eight signature reads name
/// keys no PAGE_W row mints (each MEMW_W row consumes a token nothing
/// produced, its old word and timestamp invented), refused on the word bus
/// alone. Controls, each taking out one condition the refusal rests on:
/// (a) PAGE_W's `OFFSET` the forger's (a main column at `k + i + 1/2`), which
/// mints the eight keys; (b) the step without its `−1`, each forged row
/// consuming the token it produces. Both accepted.
#[test]
fn an_unaligned_ecrec_signature_reads_words_nothing_minted() {
    let k = 0x1000u64 / WORD_BYTES;
    let forged = bench(8 * k + 4);
    let v = verdict(&forged);
    assert!(v.refused_only_on(BusId::MemWord), "{v:?}");
    let left = v.left_on(BusId::MemWord);
    assert!(
        !left.is_empty() && left.iter().all(|(t, _)| t[0] % 8 == 4),
        "{v:?}"
    );

    // (a) Page 0's rows k..k+8 at OFFSET + 1/2: they mint at 8k + 4 + 8i (a
    // zero word, the zero page's) and close it at the read.
    let mut a = forged.clone();
    let name = page_name(0);
    let half = FE::from(2u64).inv().expect("2 is invertible");
    for i in 0..8u64 {
        let r = (k + i) as usize;
        a.set(&name, r, pw::OFFSET, FE::from(k + i) + half);
        a.set(&name, r, pw::TIMESTAMP, FE::from(TS));
    }
    let t = &a.piece(&name).trace;
    let offsets = (0..t.num_rows())
        .map(|r| *t.get_main(r, pw::OFFSET))
        .collect();
    a.piece_mut(&name).preprocessed[pw::OFFSET] = offsets;
    let v = verdict(&a);
    assert!(v.accepted(), "control (a): {v:?}");

    // (b) No `−1`: every row's step one higher (the honest ones consume the
    // token they did), the forged rows at step 0 consuming their own token.
    let mut b = forged.with_interactions("MEMW_W", memw_w_without_the_minus_one());
    let n = b.piece("MEMW_W").trace.num_rows();
    for r in 0..n {
        let active = b.get("MEMW_W", r, mw::MU_WRITE) != FE::zero()
            || b.get("MEMW_W", r, mw::MU_ACCESS) != FE::zero();
        if !active {
            continue;
        }
        let address = b.get("MEMW_W", r, mw::ADDRESS_LO).canonical_u64();
        if address % 8 == 4 {
            b.set("MEMW_W", r, mw::STEP_0, FE::zero());
            b.set("MEMW_W", r, mw::STEP_1, FE::zero());
        } else {
            let s = b.get("MEMW_W", r, mw::STEP_0) + FE::one();
            b.set("MEMW_W", r, mw::STEP_0, s);
        }
    }
    let v = verdict(&b);
    assert!(v.accepted(), "control (b): {v:?}");
}

/// ★ ECREC_W writes only canonical words and reads only checked bytes: each
/// of its 80 byte pairs, taken out alone, leaves a word a message carries
/// unchecked — the census of the word format's set refuses the table then
/// (rule c10 for the written words, c17 for the read ones).
#[test]
fn every_ecrec_w_byte_pair_is_held_by_the_census() {
    let honest = ecrec_w::bus_interactions();
    super::word_memory_tests::census_table("ECREC_W[0]", &honest, 0)
        .unwrap_or_else(|e| panic!("{e}"));
    let pairs: Vec<usize> = honest
        .iter()
        .enumerate()
        .filter(|(_, bi)| bi.bus_id == u64::from(BusId::AreBytes) && bi.values.len() == 2)
        .filter(|(_, bi)| {
            [
                ecrec::cols::R,
                ecrec::cols::S,
                ecrec::cols::Z,
                ecrec::cols::XQ,
                ecrec::cols::YQ,
            ]
            .iter()
            .any(|&start| {
                bi.values[0] == column(start)
                    || (start..start + 32).any(|c| bi.values[0] == column(c))
            })
        })
        .map(|(i, _)| i)
        .collect();
    assert_eq!(pairs.len(), 80);
    for i in pairs {
        let mut mutant = honest.clone();
        mutant.remove(i);
        assert!(
            super::word_memory_tests::census_table("ECREC_W[0]", &mutant, 0).is_err(),
            "pair {i} taken out: the census passed"
        );
    }
}

//! Laptop gates for the levels' host chain steps: the knobs' words, and each
//! reworked piece against the path it replaces. Step 1
//! (`LAMBDA_VM_LEVELS_FILL_PARTS`): a fill's three parts at once give the
//! fill in series (the same tables, the same level proof bytes, every one
//! verifying), honour the thread's lean holding, and fail as the series fill
//! fails.

use crate::tables::types::FE;

use super::b3l_tables::{B3lHold, LeanFillBeside, test_hold};
use super::builder::{B3Block, B3Cv, Cell, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapHash;
use super::executor::{ArenaPiece, LfmExecution, StreamedExecution, execute};
use super::proof::{LfmFilled, lfm_fill_executed_parts, take_fill_split, take_lean_parts};
use super::wlfm::{
    WlfmProving, WlfmRate, build_wlfm, prove_wlfm_traces_as, test_format, verify_wlfm,
};
use super::wlfm_knobs::knob_word_on;
use super::wlfm_tests::HASHER;
use super::word::LfmWord;

/// Hinted words a chain of compressions reads.
const CHAINS: usize = 6;
const STEPS: usize = 5;

fn digest(b: &mut LfmBuilder, v: u64) -> Cell {
    b.digest_const([FE::from(v); 4]).as_cell()
}

/// A lean program with every part of a fill: chips (bit decompositions,
/// selects with both outputs read, publics), lean compressions over hinted
/// arena words, one-output selects (muxes), and two cap tables with reads at
/// computed indices. `lean = false` builds the same instructions without the
/// lean BLAKE3 tables (no compression, no mux, no cap).
fn program(lean: bool) -> LfmProgram {
    let mut b = LfmBuilder::new();
    if lean {
        b = b.with_wrap_hash(WrapHash::Blake3).with_b3_lean(true);
    }
    let arena = b.declare_arena((4 * CHAINS) as u32);
    let mut publics = Vec::new();
    let x = b.felt_const(FE::from(0b1011_0110u64));
    let bits = b.bit_dec(x, 8);
    let (l, r) = (digest(&mut b, 10), digest(&mut b, 20));
    // One output read (a mux in a lean program), then both read.
    let (a, _) = b.select(bits[0], l, r);
    let (_, c) = b.select(bits[1], l, r);
    let (d, e) = b.select(bits[2], l, r);
    publics.extend([a, c, d, e]);
    let mut nodes = Vec::new();
    for ch in 0..CHAINS {
        let w: Vec<Cell> = (0..4)
            .map(|i| b.hint_word(arena, (4 * ch + i) as u32))
            .collect();
        if !lean {
            publics.push(w[0]);
            continue;
        }
        let mut cv = B3Cv::Iv;
        let mut last = None;
        for step in 0..STEPS {
            let block = if (ch + step) % 3 == 0 {
                B3Block::Felts([w[0], w[1]])
            } else {
                B3Block::Lanes([w[0], w[1], w[2], w[3]])
            };
            let out = b.b3_compress(cv, block, ((ch * 7 + step) % 65) as u8, (step % 16) as u8);
            cv = B3Cv::Words(out);
            last = Some(out);
        }
        let out = last.expect("steps");
        publics.push(out[1]);
        nodes.push(out[0]);
    }
    if lean {
        // Two cap tables over the chains' digests and constants, each read at
        // indices spelled by bits.
        let consts: Vec<Cell> = (0..4u64).map(|v| digest(&mut b, 300 + v)).collect();
        let ha = b.cap_table(&nodes[..4]);
        let hb = b.cap_table(&consts);
        for (handle, index) in [(ha, 3u64), (hb, 1), (ha, 0), (ha, 3), (hb, 2)] {
            let i = b.felt_const(FE::from(index));
            let ibits = b.bit_dec(i, 2);
            let out = b.cap_read(handle, &ibits);
            publics.push(out);
        }
    }
    for p in &publics {
        b.public(*p);
    }
    let program = compile(b.finish());
    super::validator::validate(&program).expect("admissible");
    program
}

fn arena() -> Vec<LfmWord> {
    (0..4 * CHAINS as u64)
        .map(|i| {
            core::array::from_fn(|k| FE::from((i * 0x9E37_79B9 + k as u64 * 77) & 0xFFFF_FFFF))
        })
        .collect()
}

/// The program executed over [`arena`] in one streamed piece with its
/// compressions tapped, and its lean tables' prefill made from the taps.
fn tapped_prefill(program: &LfmProgram) -> (LfmExecution, super::b3l_tables::LeanPrefilled) {
    let words = arena();
    let piece = ArenaPiece {
        arena: 0,
        start: 0,
        len: words.len() as u32,
        group: 0,
    };
    let mut ex = StreamedExecution::new_pieced_tapped(program, vec![piece], &HASHER, true)
        .expect("a tapped pieced execution starts");
    let mut fill = LeanFillBeside::new(program);
    fill.feed(&ex.take_lean_tap());
    ex.land_piece(0, &words).expect("the piece lands");
    fill.feed(&ex.take_lean_tap());
    let exec = ex.finish().expect("the execution finishes");
    (exec, fill.finish().expect("every compression fed"))
}

fn fill(
    program: &LfmProgram,
    exec: LfmExecution,
    hold_chips: bool,
    lean: Option<super::b3l_tables::LeanPrefilled>,
    at_once: bool,
) -> LfmFilled {
    lfm_fill_executed_parts(
        program,
        exec,
        HASHER,
        0.0,
        &Vec::new(),
        hold_chips,
        lean,
        at_once,
    )
}

/// The level proof of `filled`, verified; its bytes.
fn proof_bytes(program: &LfmProgram, filled: LfmFilled, proving: WlfmProving) -> Vec<u8> {
    let built = build_wlfm(program, HASHER, WlfmRate::Two).expect("the W-LFM build");
    let proof = prove_wlfm_traces_as(&built, filled, true, proving).expect("proves");
    verify_wlfm(&built.artifacts, &proof, &proof.public_words).expect("verifies");
    rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .expect("serializes")
        .to_vec()
}

/// The lean tables' shapes, table by table: rows, columns, held as records.
fn lean_shape(filled: &mut LfmFilled) -> Vec<(usize, usize, bool)> {
    let Some(t) = filled.b3l_mut() else {
        return Vec::new();
    };
    t.rounds
        .iter()
        .chain(&t.heads)
        .chain(core::iter::once(&t.table))
        .chain(&t.muxes)
        .chain(&t.caps)
        .chain(&t.cap_reads)
        .map(|h| (h.rows(), h.cols(), h.is_records()))
        .collect()
}

// =============================================================================
// The knobs
// =============================================================================

/// The knob is named, off by default, and on only for `1`; a test's
/// override is this thread's alone.
#[test]
fn the_fill_parts_knob_is_named_and_off_by_default() {
    use super::levels::{FILL_PARTS_ENV, fill_parts, set_fill_parts_for_test};
    assert_eq!(FILL_PARTS_ENV, "LAMBDA_VM_LEVELS_FILL_PARTS");
    if std::env::var_os(FILL_PARTS_ENV).is_none() {
        assert!(!fill_parts(), "off by default");
    }
    for (word, on) in [
        (None, false),
        (Some(""), false),
        (Some("0"), false),
        (Some("on"), false),
        (Some("true"), false),
        (Some("1"), true),
        (Some(" 1 "), true),
    ] {
        assert_eq!(knob_word_on(word), on, "{word:?}");
    }
    set_fill_parts_for_test(Some(true));
    assert!(fill_parts(), "the override");
    let other = std::thread::spawn(fill_parts).join().expect("joins");
    set_fill_parts_for_test(None);
    if std::env::var_os(FILL_PARTS_ENV).is_none() {
        assert!(!other, "another thread reads the knob");
    }
}

// =============================================================================
// Step 1: the fill's parts at once
// =============================================================================

/// ★ PIN: the fill's three parts at once are the fill in series: the same
/// lean tables (shape for shape, the selects' and caps' among them) and the
/// same level proof bytes, every proof verifying, with the chips filled and
/// held as their records, the lean tables filled here and (held as records)
/// prefilled beside the execution, held and streamed proving, under both
/// lean holdings.
#[test]
fn the_fill_parts_at_once_are_the_fill_in_series() {
    let _fast = test_format::set(Some(2));
    let program = program(true);
    assert!(program.groups.b3l.real_rows > 0, "compressions");
    assert!(program.groups.mux.real_rows > 0, "one-output selects");
    assert!(
        program.groups.cap.real_rows > 0 && program.groups.cap_read.real_rows > 0,
        "caps"
    );
    let words = arena();
    for hold in [B3lHold::Narrow, B3lHold::Records] {
        let _hold = test_hold::set(hold);
        // A prefill is made only where the lean tables are held as their
        // records (the driver's `lean_beside`).
        let prefills: &[bool] = if hold == B3lHold::Records {
            &[false, true]
        } else {
            &[false]
        };
        let cases = prefills.iter().flat_map(|&prefill| {
            [
                (false, prefill, WlfmProving::Held),
                (false, prefill, WlfmProving::Streamed),
                (true, prefill, WlfmProving::Streamed),
            ]
        });
        for (hold_chips, prefill, proving) in cases {
            let what = format!("{hold:?}, chips held {hold_chips}, prefill {prefill}, {proving:?}");
            let mut filled = [false, true].map(|at_once| {
                let (exec, lean) = if prefill {
                    let (exec, lean) = tapped_prefill(&program);
                    (exec, Some(lean))
                } else {
                    let exec =
                        execute(&program, std::slice::from_ref(&words), &HASHER).expect("executes");
                    (exec, None)
                };
                let filled = fill(&program, exec, hold_chips, lean, at_once);
                assert_eq!(
                    filled.lean_prefilled(),
                    prefill,
                    "{what}: the prefill taken"
                );
                let split = take_fill_split().expect("a fill split");
                let parts = take_lean_parts().expect("lean parts");
                assert!(split.0 >= 0.0 && split.1 >= 0.0 && parts.0 >= 0.0 && parts.1 >= 0.0);
                filled
            });
            let shapes = filled.each_mut().map(lean_shape);
            assert!(!shapes[0].is_empty(), "{what}: lean tables");
            assert_eq!(shapes[0], shapes[1], "{what}: the lean tables' shapes");
            // The thread's holding, honoured off this thread: under records
            // (and not prefilled, which holds records anyway) some round chunk
            // is a records table in both.
            if hold == B3lHold::Records {
                assert!(
                    shapes[1].iter().any(|s| s.2),
                    "{what}: rounds held as records"
                );
            } else {
                assert!(shapes[1].iter().all(|s| !s.2), "{what}: every table narrow");
            }
            let [series, at_once] = filled;
            assert_eq!(
                proof_bytes(&program, series, proving),
                proof_bytes(&program, at_once, proving),
                "{what}: the same level proof bytes"
            );
        }
    }
}

/// A program with no lean compression has no lean tables either way, and
/// the same proof bytes.
#[test]
fn the_fill_parts_at_once_without_lean_tables_are_the_fill_in_series() {
    let _fast = test_format::set(Some(2));
    let program = program(false);
    assert_eq!(program.groups.b3l.real_rows, 0);
    assert_eq!(
        program.groups.mux.real_rows + program.groups.cap.real_rows,
        0
    );
    let words = arena();
    let [series, at_once] = [false, true].map(|at_once| {
        let exec = execute(&program, std::slice::from_ref(&words), &HASHER).expect("executes");
        let mut filled = fill(&program, exec, true, None, at_once);
        assert!(filled.b3l_mut().is_none(), "no lean tables");
        filled
    });
    assert_eq!(
        proof_bytes(&program, series, WlfmProving::Streamed),
        proof_bytes(&program, at_once, WlfmProving::Streamed),
        "the same level proof bytes"
    );
}

/// A part that fails fails the fill at once as it fails it in series: the
/// same panic, raised on the caller's thread (here a mux record missing,
/// which the selects' part refuses).
#[test]
fn a_failing_part_fails_the_fill_at_once_as_in_series() {
    let program = program(true);
    let words = arena();
    let message = |at_once: bool| -> String {
        let mut exec = execute(&program, std::slice::from_ref(&words), &HASHER).expect("executes");
        exec.records.mux.pop().expect("a mux record");
        let caller = std::thread::current().id();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fill(&program, exec, false, None, at_once)
        }))
        .err()
        .expect("the fill refuses a short mux record list");
        assert_eq!(std::thread::current().id(), caller);
        panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    };
    let (series, at_once) = (message(false), message(true));
    assert!(series.contains("one LEAN_MUX record a mux row"), "{series}");
    assert_eq!(series, at_once, "the same refusal");
}

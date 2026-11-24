//! Laptop gates for the levels' host chain steps: the knobs' words, and each
//! reworked piece against the path it replaces. Step 1
//! (`LAMBDA_VM_LEVELS_FILL_PARTS`): a fill's three parts at once give the
//! fill in series (the same tables, the same level proof bytes, every one
//! verifying), honour the thread's lean holding, and fail as the series fill
//! fails. Step 2 (`LAMBDA_VM_LEVELS_COMMIT_NEXT_EARLY`) moves only when the
//! driver commits; its identity is a box-tier driver test
//! (`levels_tests::the_blake3_driver_committing_level_2_early_proves_the_same_level_bytes`).
//! Step 3 (`LAMBDA_VM_LEVELS_COLUMNS_TILED`): the tiled columns are today's
//! cell for cell, and a build's host part made by tiles, its chips' chunks
//! read in place, is today's stack for stack. Step 4
//! (`LAMBDA_VM_LEVELS_PACKS_BESIDE`): a level's chip prefixes and narrow packs
//! made at once are the ones made in turn.

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

/// Both knobs are named, off by default, and on only for `1`; a test's
/// override is this thread's alone.
#[test]
fn the_chain_knobs_are_named_and_off_by_default() {
    use super::levels::{
        COMMIT_NEXT_EARLY_ENV, FILL_PARTS_ENV, commit_next_early, fill_parts,
        set_commit_next_early_for_test, set_fill_parts_for_test,
    };
    assert_eq!(FILL_PARTS_ENV, "LAMBDA_VM_LEVELS_FILL_PARTS");
    assert_eq!(COMMIT_NEXT_EARLY_ENV, "LAMBDA_VM_LEVELS_COMMIT_NEXT_EARLY");
    if std::env::var_os(FILL_PARTS_ENV).is_none() {
        assert!(!fill_parts(), "off by default");
    }
    if std::env::var_os(COMMIT_NEXT_EARLY_ENV).is_none() {
        assert!(!commit_next_early(), "off by default");
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
    set_commit_next_early_for_test(Some(true));
    assert!(fill_parts() && commit_next_early(), "the overrides");
    let other = std::thread::spawn(|| (fill_parts(), commit_next_early()))
        .join()
        .expect("joins");
    set_fill_parts_for_test(None);
    set_commit_next_early_for_test(None);
    if std::env::var_os(FILL_PARTS_ENV).is_none()
        && std::env::var_os(COMMIT_NEXT_EARLY_ENV).is_none()
    {
        assert_eq!(other, (false, false), "another thread reads the knobs");
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

// =============================================================================
// Step 3: the program's tables into columns by tiles
// =============================================================================

/// A row-major table `width` words wide and `rows` rows tall, every cell its
/// own value.
fn table(width: usize, rows: usize) -> Vec<FE> {
    (0..width * rows)
        .map(|i| FE::from((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 7))
        .collect()
}

/// [`super::wlfm::group_columns`] of rows `rows` of `data` copied out and
/// padded with zeros to `padded`: today's path for a chip's chunk.
fn copied_columns(
    data: &[FE],
    width: usize,
    rows: std::ops::Range<usize>,
    padded: usize,
) -> Vec<Vec<FE>> {
    let mut cut = data[rows.start * width..rows.end * width].to_vec();
    cut.resize(padded * width, FE::zero());
    super::wlfm::group_columns(&super::compiler::ColumnGroup {
        width,
        real_rows: rows.len(),
        padded_rows: padded,
        data: cut,
    })
}

/// ★ PIN: the tiled columns are today's, cell for cell: widths 1 to 33 (under,
/// at and over a cache line of words), heights of 4 rows, a tile, a tile ± 1
/// and several tiles, the whole table and row ranges that start and end
/// inside a tile, padded or not.
#[test]
fn the_tiled_columns_are_the_columns() {
    use super::wlfm::{TILE_ROWS, group_columns_tiled};
    for width in [1usize, 2, 7, 8, 9, 33] {
        for height in [
            4usize,
            TILE_ROWS - 1,
            TILE_ROWS,
            TILE_ROWS + 1,
            3 * TILE_ROWS + 5,
        ] {
            let data = table(width, height);
            let whole = super::wlfm::group_columns(&super::compiler::ColumnGroup {
                width,
                real_rows: height,
                padded_rows: height,
                data: data.clone(),
            });
            assert_eq!(
                group_columns_tiled(&data, width, 0..height, height),
                whole,
                "width {width} height {height}: the whole table"
            );
            let inner = [
                (1, height),
                (0, height / 2),
                (height / 3, height - 1),
                (TILE_ROWS.min(height) - 1, height),
            ];
            for (a, b) in inner {
                for padded in [
                    b - a,
                    (b - a).next_power_of_two(),
                    2 * (b - a).next_power_of_two(),
                ] {
                    assert_eq!(
                        group_columns_tiled(&data, width, a..b, padded),
                        copied_columns(&data, width, a..b, padded),
                        "width {width} height {height}: rows {a}..{b} padded to {padded}"
                    );
                }
            }
        }
    }
    assert!(
        group_columns_tiled(&[], 0, 0..0, 4).is_empty(),
        "no columns"
    );
}

/// ★ PIN: a build's host part with its columns made by tiles, its chips'
/// chunks read in place, is today's: the same stacks column for column, the
/// same prepared cells and the same committed identity, with chips cut into
/// several chunks, under both families.
#[test]
fn a_tiled_prepare_is_the_prepare() {
    use super::wlfm::{StackPlace, WlfmFamily, WlfmLevers, WlfmRate, prepare_wlfm_tiled};
    let _chunks = test_format::set_chunk_log(2);
    for (program, family) in [
        (super::wlfm_tests::honest(), WlfmFamily::P1),
        (super::b3l_wlfm_tests::honest(), WlfmFamily::B3),
    ] {
        let levers = WlfmLevers::with_family(family);
        let [today, tiled] = [false, true].map(|tiled| {
            prepare_wlfm_tiled(&program, HASHER, WlfmRate::Two, levers, None, tiled)
                .unwrap_or_else(|e| panic!("{family:?} tiled {tiled}: prepares: {e}"))
        });
        let (a, b) = (
            today.stack_columns_for_test().expect("stacks"),
            tiled.stack_columns_for_test().expect("stacks"),
        );
        assert!(!a.is_empty(), "{family:?}: prepared stacks");
        assert!(a == b, "{family:?}: the same stacks, column for column");
        assert_eq!(today.prepared_cells(), tiled.prepared_cells());
        let [today, tiled] =
            [today, tiled].map(|pre| pre.commit(StackPlace::Host).expect("commits"));
        assert!(
            today.artifacts.lfm_chunks.iter().any(|c| c.len() > 1),
            "{family:?}: a chip cut into chunks"
        );
        assert_eq!(today.artifacts.lfm_chunks, tiled.artifacts.lfm_chunks);
        assert_eq!(
            today.artifacts.program_id, tiled.artifacts.program_id,
            "{family:?}: the same committed identity"
        );
    }
}

// =============================================================================
// Step 4: the two packs at once
// =============================================================================

/// ★ PIN: a level's two packs made at once are the packs made in turn: the
/// same chip prefixes table for table and the same narrow pack stack for
/// stack, under both families, and the knob's words.
#[test]
fn the_packs_at_once_are_the_packs_in_turn() {
    use super::levels::{PACKS_BESIDE_ENV, packs_beside, prepare_level_packs};
    use super::wlfm::{WlfmFamily, set_narrow_commit_for_test};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_narrow_commit_for_test(None);
        }
    }
    assert_eq!(PACKS_BESIDE_ENV, "LAMBDA_VM_LEVELS_PACKS_BESIDE");
    if std::env::var_os(PACKS_BESIDE_ENV).is_none() {
        assert!(!packs_beside(), "off by default");
    }
    let _restore = Restore;
    let _records = test_format::set_chip_records(true);
    set_narrow_commit_for_test(Some(true));
    for (program, family) in [
        (super::wlfm_tests::honest(), WlfmFamily::P1),
        (super::b3l_wlfm_tests::honest(), WlfmFamily::B3),
    ] {
        let [in_turn, at_once] = [false, true].map(|beside| {
            prepare_level_packs(&program, family, None, beside)
                .unwrap_or_else(|e| panic!("{family:?} beside {beside}: prepares: {e}"))
        });
        assert!(
            in_turn.chip_prefixes_for_test().iter().any(Option::is_some),
            "{family:?}: chip prefixes made"
        );
        assert!(
            in_turn.chip_prefixes_for_test() == at_once.chip_prefixes_for_test(),
            "{family:?}: the same prefixes"
        );
        let packs = in_turn.narrow_packs_for_test();
        assert!(
            !packs.is_empty() && packs.iter().all(Option::is_some),
            "{family:?}: every stack packed"
        );
        assert_eq!(
            packs,
            at_once.narrow_packs_for_test(),
            "{family:?}: the same packs"
        );
    }
}

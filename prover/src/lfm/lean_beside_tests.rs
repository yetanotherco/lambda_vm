//! Laptop gates for level 1's lean round tables filled beside its execution
//! ([`super::b3l_tables::LeanFillBeside`], `LAMBDA_VM_WLFM_LEAN_BESIDE`): the
//! knob's words, a streamed execution's waves reporting every compression
//! once whatever order its pieces land in, the tables made from those waves
//! equal to the records path's column for column, the level proof the same
//! bytes with the fill beside and without, and a wave fed twice or missed
//! refused, never handed on as a wrong count.

use crate::tables::types::FE;

use super::b3l::B3lValues;
use super::b3l_tables::{
    B3lHold, B3lTraces, LeanFillBeside, LeanPrefilled, build_b3l_traces_held,
    build_b3l_traces_prefilled, test_hold,
};
use super::builder::{B3Block, B3Cv, Cell, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::exec_identity_tests::assert_executions_identical;
use super::executor::{ArenaPiece, LfmExecution, StreamedExecution, execute};
use super::proof::lfm_fill_executed_prefilled;
use super::wlfm::{
    LEAN_BESIDE_ENV, WlfmBuild, WlfmProving, WlfmRate, build_wlfm, lean_beside,
    prove_wlfm_traces_as, test_format, verify_wlfm,
};
use super::wlfm_knobs::knob_word_on;
use super::wlfm_tests::HASHER;
use super::word::LfmWord;

/// Chains of compressions over hinted words, `per_piece` chains a piece of
/// the one arena (four words a chain), and one chain over constants that
/// reads no arena (it runs before any piece lands). The chains alternate
/// lane and felt blocks, IV and chained rows, plain and reversed digests;
/// every chain's last digest word is published.
const CHAINS: usize = 12;
const STEPS: usize = 7;
const PIECES: usize = 4;

fn program() -> LfmProgram {
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena((4 * CHAINS) as u32);
    let mut publics = Vec::new();
    let consts: Vec<Cell> = (0..4u64)
        .map(|k| b.digest_const([FE::from(900 + k); 4]).as_cell())
        .collect();
    let mut cv = B3Cv::Iv;
    for step in 0..5u64 {
        let d = b.b3_compress(
            cv,
            B3Block::Lanes([consts[0], consts[1], consts[2], consts[3]]),
            64,
            (step % 16) as u8,
        );
        cv = B3Cv::Words(d);
    }
    if let B3Cv::Words(d) = cv {
        publics.push(d[0]);
    }
    for c in 0..CHAINS {
        let w: Vec<Cell> = (0..4)
            .map(|i| b.hint_word(arena, (4 * c + i) as u32))
            .collect();
        let mut cv = B3Cv::Iv;
        let mut last = None;
        for step in 0..STEPS {
            let block = if (c + step) % 3 == 0 {
                B3Block::Felts([w[0], w[1]])
            } else {
                B3Block::Lanes([w[0], w[1], w[2], w[3]])
            };
            let d = if step % 4 == 3 {
                b.b3_compress_rev(cv, block, 64, (step % 16) as u8).0
            } else {
                b.b3_compress(cv, block, ((c * 7 + step) % 65) as u8, (step % 16) as u8)
            };
            cv = if step % 5 == 4 {
                B3Cv::Iv
            } else {
                B3Cv::Words(d)
            };
            last = Some(d);
        }
        publics.push(last.expect("steps")[1]);
    }
    for p in &publics {
        b.public(*p);
    }
    let program = compile(b.finish());
    super::validator::validate(&program).expect("admissible");
    program
}

/// The arena: word `i` holds four `u32` lanes.
fn arena() -> Vec<LfmWord> {
    (0..4 * CHAINS as u64)
        .map(|i| {
            core::array::from_fn(|k| FE::from((i * 0x9E37_79B9 + k as u64 * 77) & 0xFFFF_FFFF))
        })
        .collect()
}

/// The arena cut into [`PIECES`] pieces at chain boundaries.
fn pieces() -> Vec<ArenaPiece> {
    let per = (4 * CHAINS / PIECES) as u32;
    (0..PIECES)
        .map(|g| ArenaPiece {
            arena: 0,
            start: g as u32 * per,
            len: per,
            group: g,
        })
        .collect()
}

/// The program streamed with its compressions tapped, its pieces landing in
/// `order`: the execution, and every wave's tap (the first is the wave
/// before any piece lands).
fn tapped(
    program: &LfmProgram,
    words: &[LfmWord],
    order: &[usize],
) -> (LfmExecution, Vec<Vec<(u32, B3lValues)>>) {
    let pieces = pieces();
    let mut ex = StreamedExecution::new_pieced_tapped(program, pieces.clone(), &HASHER, true)
        .expect("a tapped pieced execution starts");
    let mut waves = vec![ex.take_lean_tap()];
    for &i in order {
        let p = pieces[i];
        ex.land_piece(
            p.group,
            &words[p.start as usize..(p.start + p.len) as usize],
        )
        .expect("a piece lands");
        waves.push(ex.take_lean_tap());
    }
    assert!(
        ex.take_lean_tap().is_empty(),
        "nothing left after the last wave"
    );
    (ex.finish().expect("the execution finishes"), waves)
}

/// The waves fed, in order, to a fill beside.
fn fed<'p>(program: &'p LfmProgram, waves: &[Vec<(u32, B3lValues)>]) -> LeanFillBeside<'p> {
    let mut fill = LeanFillBeside::new(program);
    for wave in waves {
        fill.feed(wave);
    }
    fill
}

/// Two lean table sets, table for table and column for column (a records
/// table through its host fill).
fn assert_tables_equal(what: &str, a: &B3lTraces, b: &B3lTraces) {
    let pairs = a
        .rounds
        .iter()
        .zip(&b.rounds)
        .chain(a.heads.iter().zip(&b.heads))
        .chain(core::iter::once((&a.table, &b.table)));
    assert_eq!(a.rounds.len(), b.rounds.len(), "{what}: round chunks");
    assert_eq!(a.heads.len(), b.heads.len(), "{what}: head chunks");
    for (t, (x, y)) in pairs.enumerate() {
        assert_eq!(
            (x.rows(), x.cols(), x.is_records()),
            (y.rows(), y.cols(), y.is_records()),
            "{what}: table {t} shape and holding"
        );
        for c in 0..x.cols() {
            assert_eq!(x.column(c), y.column(c), "{what}: table {t} column {c}");
        }
    }
}

// =============================================================================
// The knob
// =============================================================================

/// The lean round tables are filled beside the execution by default, as part
/// of the levels' tail (`LAMBDA_VM_LEVELS_TAIL`, on unless `0`); the knob of
/// its own turns on for `1` alone.
#[test]
fn the_lean_fill_beside_follows_the_tail_and_its_own_knob_is_on_for_one_alone() {
    assert_eq!(LEAN_BESIDE_ENV, "LAMBDA_VM_WLFM_LEAN_BESIDE");
    if std::env::var_os(LEAN_BESIDE_ENV).is_none() {
        assert_eq!(
            lean_beside(),
            super::wlfm_knobs::tail(),
            "the tail's by default"
        );
    }
    assert!(knob_word_on(Some("1")));
    for word in [
        None,
        Some(""),
        Some("0"),
        Some("true"),
        Some("on"),
        Some("11"),
    ] {
        assert!(!knob_word_on(word), "{word:?} turns the knob on");
    }
    let on = test_format::set_lean_beside(true);
    assert!(lean_beside(), "the override");
    let off = test_format::set_lean_beside(false);
    assert!(!lean_beside(), "the override off");
    drop(off);
    drop(on);
}

/// An execution not tapped reports nothing; a tapped one is the same
/// execution.
#[test]
fn an_untapped_execution_reports_no_compression() {
    let program = program();
    let words = arena();
    let mut ex =
        StreamedExecution::new_pieced(&program, pieces(), &HASHER).expect("a pieced execution");
    for p in pieces() {
        ex.land_piece(
            p.group,
            &words[p.start as usize..(p.start + p.len) as usize],
        )
        .expect("lands");
        assert!(ex.take_lean_tap().is_empty(), "untapped: nothing reported");
    }
    let plain = ex.finish().expect("finishes");
    let (tapped, _) = tapped(&program, &words, &[0, 1, 2, 3]);
    assert_executions_identical("tapped", program.num_addrs, &plain, &tapped);
}

// =============================================================================
// The tables
// =============================================================================

/// ★ PIN: in any landing order, the waves report every compression exactly
/// once, spread over several waves, with the record the execution kept; and
/// the tables made beside them are the records path's column for column (the
/// round chunks as their records through the host fill, the head chunks, the
/// byte table), over a program whose compressions straddle round chunks.
#[test]
fn the_lean_fill_beside_is_the_records_fill_column_for_column() {
    let program = program();
    let words = arena();
    let reference = execute(&program, std::slice::from_ref(&words), &HASHER).expect("executes");
    let n = program.groups.b3l.real_rows;
    assert_eq!(n, 5 + CHAINS * STEPS);
    let records = build_b3l_traces_held(&program, &reference.records.b3l, B3lHold::Records)
        .expect("the records path");
    assert!(records.rounds.len() > 2, "several round chunks");
    for order in [[0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]] {
        let (exec, waves) = tapped(&program, &words, &order);
        assert_executions_identical("tapped", program.num_addrs, &reference, &exec);
        let busy = waves.iter().filter(|w| !w.is_empty()).count();
        assert!(busy >= PIECES, "{order:?}: compressions in {busy} waves");
        let mut seen = vec![false; n];
        for (k, values) in waves.iter().flatten() {
            let k = *k as usize;
            assert!(!seen[k], "{order:?}: compression {k} reported twice");
            seen[k] = true;
            assert_eq!(
                format!("{values:?}"),
                format!("{:?}", reference.records.b3l[k]),
                "{order:?}: compression {k}'s record"
            );
        }
        assert!(seen.iter().all(|&s| s), "{order:?}: every compression");
        let fill = fed(&program, &waves);
        let stats = fill.stats();
        assert_eq!((stats.waves, stats.compressions), (PIECES + 1, n));
        let prefilled = fill.finish().expect("complete");
        let beside = build_b3l_traces_prefilled(&program, &exec.records.b3l, prefilled)
            .expect("the prefilled tables");
        assert!(beside.rounds.iter().all(|t| t.is_records()));
        assert_tables_equal(&format!("{order:?}"), &records, &beside);
    }
}

/// A prefill of another program's compressions is not taken: the caller
/// fills as usual.
#[test]
fn a_prefill_of_another_size_is_not_taken() {
    let program = program();
    let words = arena();
    let (exec, waves) = tapped(&program, &words, &[0, 1, 2, 3]);
    let prefilled = fed(&program, &waves).finish().expect("complete");
    assert!(
        build_b3l_traces_prefilled(&program, &exec.records.b3l[1..], prefilled).is_none(),
        "records of another length"
    );
}

// =============================================================================
// The proof
// =============================================================================

fn build(program: &LfmProgram) -> WlfmBuild {
    build_wlfm(program, HASHER, WlfmRate::Two).unwrap_or_else(|e| panic!("the W-LFM build: {e}"))
}

/// The level proof of `exec`, its lean tables prefilled by `lean` or not, its
/// chips held as records or not; refused by its prover or its verifier is
/// `None`.
fn proof_bytes(
    program: &LfmProgram,
    built: &WlfmBuild,
    exec: LfmExecution,
    lean: Option<LeanPrefilled>,
    hold_chips: bool,
    proving: WlfmProving,
) -> Option<Vec<u8>> {
    let expect = lean.is_some();
    let filled = lfm_fill_executed_prefilled(
        program,
        exec,
        built.artifacts.hasher,
        0.0,
        &Vec::new(),
        hold_chips,
        lean,
    );
    assert_eq!(filled.lean_prefilled(), expect, "the prefill taken");
    let proof = prove_wlfm_traces_as(built, filled, true, proving).ok()?;
    verify_wlfm(&built.artifacts, &proof, &proof.public_words).ok()?;
    Some(
        rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .expect("serializes")
            .to_vec(),
    )
}

/// ★ PIN: the level proof is the same bytes with its lean round tables filled
/// beside the execution and after it, held and streamed, with the chips
/// filled and held as their records; every one verifies.
#[test]
fn the_level_proof_is_the_same_bytes_with_the_lean_fill_beside() {
    let _fast = test_format::set(Some(2));
    let _records = test_hold::set(B3lHold::Records);
    let program = program();
    let words = arena();
    let built = build(&program);
    let arenas = std::slice::from_ref(&words);
    for (hold_chips, proving) in [
        (false, WlfmProving::Held),
        (false, WlfmProving::Streamed),
        (true, WlfmProving::Streamed),
    ] {
        let exec = execute(&program, arenas, &HASHER).expect("executes");
        let after = proof_bytes(&program, &built, exec, None, hold_chips, proving)
            .expect("the fill after the execution proves and verifies");
        let (exec, waves) = tapped(&program, &words, &[3, 1, 0, 2]);
        let prefilled = fed(&program, &waves).finish().expect("complete");
        let beside = proof_bytes(&program, &built, exec, Some(prefilled), hold_chips, proving)
            .expect("the fill beside proves and verifies");
        assert_eq!(
            after, beside,
            "chips held {hold_chips}, {proving:?}: the same bytes"
        );
    }
}

// =============================================================================
// Refusals
// =============================================================================

/// A wave fed twice is refused.
#[test]
fn a_wave_fed_twice_is_refused() {
    let program = program();
    let (_, mut waves) = tapped(&program, &arena(), &[0, 1, 2, 3]);
    let again = waves[2].clone();
    assert!(!again.is_empty());
    waves.push(again);
    let err = fed(&program, &waves).finish().err().expect("refused");
    assert!(err.contains("twice"), "{err}");
}

/// A wave never fed is refused.
#[test]
fn a_missed_wave_is_refused() {
    let program = program();
    let (_, mut waves) = tapped(&program, &arena(), &[0, 1, 2, 3]);
    assert!(!waves[3].is_empty());
    waves.remove(3);
    let err = fed(&program, &waves).finish().err().expect("refused");
    assert!(err.contains("of the program's"), "{err}");
}

/// A compression row past the program's is refused (and does not index
/// past its records).
#[test]
fn a_row_past_the_program_is_refused() {
    let program = program();
    let (_, mut waves) = tapped(&program, &arena(), &[0, 1, 2, 3]);
    let n = program.groups.b3l.real_rows as u32;
    let (_, values) = waves[1][0];
    waves[1][0] = (n, values);
    let err = fed(&program, &waves).finish().err().expect("refused");
    assert!(err.contains("past"), "{err}");
}

/// Why the refusal is owed: a wave counted twice, handed on unchecked, makes
/// a byte table whose counts the proof's lookups do not balance, and no
/// verifying proof comes of it. The check turns that into an error before the
/// prove.
#[test]
fn a_doubled_wave_handed_on_unchecked_proves_nothing_that_verifies() {
    let _fast = test_format::set(Some(2));
    let _records = test_hold::set(B3lHold::Records);
    let program = program();
    let words = arena();
    let built = build(&program);
    let (exec, mut waves) = tapped(&program, &words, &[0, 1, 2, 3]);
    let again = waves[2].clone();
    waves.push(again);
    let doubled = LeanPrefilled::unchecked(fed(&program, &waves));
    assert!(
        proof_bytes(
            &program,
            &built,
            exec,
            Some(doubled),
            false,
            WlfmProving::Held
        )
        .is_none(),
        "a doubled count must not verify"
    );
}

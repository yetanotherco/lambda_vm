//! ★ The card-built rows' upload on the card (`math_cuda::card_rows`,
//! `LAMBDA_VM_CARD_ROWS`); `words`' MEMW_R (its rows built from the walk's
//! words and the logs) at the end of the file: a group's upload in the knob's
//! `order` (every copy, then every kernel) and in `words` (CPU logs kept on
//! the card; LOAD and STORE derived from them)
//! writes the words today's upload writes, group by group, in the hand-out's
//! order and again in the reverse order (phase B's re-uploads, with the cache
//! holding other chunks than the ones asked for). Each test says that the
//! mechanism moved (tables derived, CPU tables served from the cache), so a
//! knob that did nothing cannot pass, and a mutant (each derived row's cycle
//! read one late) must change words or decline a group.
//!
//! Card only, and each test alone (`--test-threads=1`): the knob's override
//! and the cache are the process's.

use executor::elf::Elf;
use math::field::goldilocks::GoldilocksField as F;
use math_cuda::card_rows::{self, Mode};
use multilinear::gpu::{TableColumns, upload_tables};
use multilinear::records::Records;

use super::StreamTable;
use super::card_records_card_tests::{mismatches, records_chunks};
use crate::tables::MaxRowsConfig;
use crate::tests::windowed_builder_tests::run;

/// Logs kept across groups (as inside a block's prove) while it lives; the
/// knob back and the cache empty however the test ends.
struct Reset;

impl Reset {
    fn new() -> Self {
        card_rows::begin();
        Self
    }
}

impl Drop for Reset {
    fn drop(&mut self) {
        card_rows::FAULT_CYCLE_SHIFT.store(0, std::sync::atomic::Ordering::Relaxed);
        card_rows::FAULT_FAMILIES.store(0, std::sync::atomic::Ordering::Relaxed);
        card_rows::FAULT_REGISTERS.store(0, std::sync::atomic::Ordering::Relaxed);
        card_rows::set_mode_override(None);
        card_rows::end();
    }
}

/// The groups' stores in `mode`, uploaded in `order` (indices into
/// `groups`), each `None` when the card declined it.
fn stores(
    groups: &[Vec<&Records>],
    order: &[usize],
    mode: Mode,
) -> Vec<(usize, Option<Vec<Vec<u64>>>)> {
    card_rows::set_mode_override(Some(mode));
    order
        .iter()
        .map(|&g| {
            let tables: Vec<TableColumns<'_, F>> =
                groups[g].iter().map(|r| TableColumns::Records(r)).collect();
            (g, upload_tables(&tables).and_then(|store| store.download()))
        })
        .collect()
}

/// Consecutive groups of `size` of the chunks, in hand-out order.
fn grouped(chunks: &[(StreamTable, usize, Records)], size: usize) -> Vec<Vec<&Records>> {
    chunks
        .chunks(size)
        .map(|g| g.iter().map(|(_, _, r)| r).collect())
        .collect()
}

/// Every group in `order` and `words`, forwards and backwards, against
/// today's: the words differing in each (0 everywhere). Returns the counters
/// of the words mode's two passes and today's stores, group by group. The
/// chunks hold MEMW_R as rows, so `words` derives LOAD and STORE here and
/// sends MEMW_R as records.
fn compare(groups: &[Vec<&Records>], what: &str) -> (card_rows::Stats, Vec<Vec<Vec<u64>>>) {
    let forwards: Vec<usize> = (0..groups.len()).collect();
    let backwards: Vec<usize> = (0..groups.len()).rev().collect();
    card_rows::clear();
    let today = stores(groups, &forwards, Mode::Off);
    for (g, store) in &today {
        assert!(
            store.is_some(),
            "{what}: today's upload of group {g} declined"
        );
    }
    let want = |g: usize| today[g].1.as_ref().expect("today's store");
    let mut words_stats = card_rows::Stats::default();
    for (mode, order) in [
        (Mode::Order, &forwards),
        (Mode::Words, &forwards),
        (Mode::Words, &backwards),
    ] {
        if order == &forwards {
            card_rows::clear();
        }
        for (g, got) in stores(groups, order, mode) {
            let got = got.unwrap_or_else(|| panic!("{what}: {mode:?} declined group {g}"));
            let (count, first) = mismatches(&got, want(g));
            assert_eq!(
                count, 0,
                "{what}: {mode:?} group {g}: {count} words differ, the first at {first:?}"
            );
        }
        if mode == Mode::Words {
            let s = card_rows::stats();
            words_stats = s;
        }
    }
    let today = today
        .into_iter()
        .map(|(_, store)| store.expect("today's store"))
        .collect();
    (words_stats, today)
}

#[test]
fn card_rows_uploads_write_todays_words_forwards_and_backwards() {
    let _reset = Reset::new();
    let max_rows = MaxRowsConfig {
        load: 1 << 2,
        store: 1 << 2,
        ..MaxRowsConfig::small()
    };
    let (mut derived, mut hits, mut kept) = (0u64, 0u64, 0u64);
    for name in [
        "all_loadstore_32",
        "all_instructions_64",
        "misalign_ld",
        "misalign_sd",
        "test_keccak_multi",
    ] {
        let (program, _) = run(name);
        for window in [7usize, 33] {
            let chunks = records_chunks(&program, &[], &max_rows, window, (0, None), true);
            for size in [3usize, 8] {
                let groups = grouped(&chunks, size);
                let (s, _) = compare(
                    &groups,
                    &format!("{name}, window {window}, groups of {size}"),
                );
                derived += s.derived_load + s.derived_store;
                hits += s.cpu_hits;
                kept += s.cpu_kept;
                assert_eq!(s.declined, 0, "{name}: a group declined");
            }
        }
    }
    // Outside a block's prove nothing is kept past a group.
    card_rows::end();
    let (program, _) = run("all_loadstore_32");
    let chunks = records_chunks(&program, &[], &max_rows, 7, (0, None), true);
    let groups = grouped(&chunks, 8);
    let outside = stores(&groups, &[0], Mode::Words);
    assert!(outside[0].1.is_some(), "a group outside a block declined");
    assert_eq!(card_rows::held(), (0, 0), "logs kept outside a block");
    println!(
        "CARD ROWS SMALL: tables derived {derived} · CPU tables from the cache {hits} · kept {kept}"
    );
    assert!(
        derived > 0 && hits > 0 && kept > 0,
        "the mechanism did not move: derived {derived}, hits {hits}, kept {kept}"
    );
}

/// The real block (`BLOCK_WHIR_ELF`, `BLOCK_WHIR_INPUT`) at the production
/// heights: the hand-out's chunks up to its first LOAD and STORE chunks and
/// `CARD_ROWS_EXTRA_CHUNKS` (default 24) more, in groups of ten in hand-out
/// order and then backwards, in each mode against today's. Then the mutant
/// on the same groups and passes: in LOAD alone, then in STORE alone, each
/// derived row's cycle read one late. Each family must derive a table and
/// move a group (other words, or the group declined): the passes above
/// derive both families here, which small programs' chunks do not.
#[test]
#[ignore = "card: a box test (BLOCK_WHIR_ELF, BLOCK_WHIR_INPUT)"]
fn card_rows_on_a_real_block_write_todays_words() {
    let _reset = Reset::new();
    let elf =
        std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF")).expect("the ELF");
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).expect("input");
    let extra: usize = std::env::var("CARD_ROWS_EXTRA_CHUNKS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(24);
    let program = Elf::load(&elf).expect("the ELF loads");
    let options = crate::block_whir::BlockOptions::production();
    let window = 1usize << options.window_log2.expect("the block streams");
    let t = std::time::Instant::now();
    let all = records_chunks(
        &program,
        &input,
        &options.max_rows,
        window,
        (0, Some(60)),
        true,
    );
    let seen = |table: StreamTable| all.iter().position(|(t, _, _)| *t == table);
    let (Some(load), Some(store)) = (seen(StreamTable::Load), seen(StreamTable::Store)) else {
        panic!("no LOAD or STORE chunk in the first 60 windows");
    };
    let end = (load.max(store) + 1 + extra).min(all.len());
    let chunks = &all[..end];
    println!(
        "CARD ROWS REAL: {} chunks ({} CPU, {} LOAD, {} STORE) of the first {} in {:.1}s",
        chunks.len(),
        chunks.iter().filter(|c| c.0 == StreamTable::Cpu).count(),
        chunks.iter().filter(|c| c.0 == StreamTable::Load).count(),
        chunks.iter().filter(|c| c.0 == StreamTable::Store).count(),
        all.len(),
        t.elapsed().as_secs_f64()
    );
    let groups = grouped(chunks, 10);
    let (s, today) = compare(&groups, "the real block");
    println!(
        "CARD ROWS REAL VERDICT: 0 words differ in {} groups × 3 passes · derived LOAD {} STORE {} · as records LOAD {} STORE {} · CPU from the cache {} · kept {} · let go {}",
        groups.len(),
        s.derived_load,
        s.derived_store,
        s.records_load,
        s.records_store,
        s.cpu_hits,
        s.cpu_kept,
        s.evicted
    );
    assert!(
        s.derived_load > 0 && s.derived_store > 0,
        "no LOAD or STORE table derived on the real block"
    );

    // The mutant: the words passes again (forwards from an empty cache, then
    // backwards with it warm), each derived row's cycle read one late in one
    // family at a time.
    let forwards: Vec<usize> = (0..groups.len()).collect();
    let backwards: Vec<usize> = (0..groups.len()).rev().collect();
    for (family, bit) in [("LOAD", 1u8), ("STORE", 2u8)] {
        card_rows::clear();
        card_rows::FAULT_FAMILIES.store(bit, std::sync::atomic::Ordering::Relaxed);
        card_rows::FAULT_CYCLE_SHIFT.store(1, std::sync::atomic::Ordering::Relaxed);
        let mut moved = 0usize;
        let mut uploads = 0usize;
        for order in [&forwards, &backwards] {
            for (g, got) in stores(&groups, order, Mode::Words) {
                uploads += 1;
                moved += usize::from(match got {
                    None => true,
                    Some(got) => mismatches(&got, &today[g]).0 > 0,
                });
            }
        }
        let after = card_rows::stats();
        card_rows::FAULT_CYCLE_SHIFT.store(0, std::sync::atomic::Ordering::Relaxed);
        card_rows::FAULT_FAMILIES.store(0, std::sync::atomic::Ordering::Relaxed);
        let derived = if bit == 1 {
            after.derived_load
        } else {
            after.derived_store
        };
        println!(
            "CARD ROWS MUTANT {family} cycle read one late: {moved} of {uploads} groups moved · {family} tables derived {derived}"
        );
        assert!(
            derived > 0,
            "{family}: no table derived, the mutant cannot bite"
        );
        assert!(moved > 0, "{family}: the mutant moved no group");
    }
}

// ---------------------------------------------------------------------------
// `words`: MEMW_R's rows built on the card from the words the walk writes.
// ---------------------------------------------------------------------------

/// The absorbing side's word for "enough windows" ([`history_chunks_as`]).
const ENOUGH: &str = "card rows test: enough windows walked";

/// Every chunk a record-mode run of `program` hands out as records (every
/// row family's too): windows of `window` cycles walked from the fast
/// engine's history on two walkers, through `windows` windows absorbed (all
/// of them: `None`), in hand-out order; the walkers writing MEMW_R's rows as
/// words when `words` ([`WindowedTraceBuilder::card_writes_words`]: they mark
/// the rows the card cannot build), as rows otherwise.
fn history_chunks_as(
    program: &Elf,
    input: &[u8],
    max_rows: &MaxRowsConfig,
    window: usize,
    windows: Option<usize>,
    words: bool,
) -> Vec<(StreamTable, usize, Records)> {
    use super::parallel::{ExecWindow, walk_windows_from_history};
    use super::{RowFamilies, WindowedTraceBuilder};
    use executor::vm::fast::FastExecutor;
    let builder = WindowedTraceBuilder::new(program, input, max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(std::sync::Arc::new(|_| true), true, true))
        .and_then(|b| b.card_fill_streamed_rows(RowFamilies::ALL));
    let mut builder = if words {
        builder.and_then(WindowedTraceBuilder::card_writes_words)
    } else {
        builder
    }
    .expect("the builder");
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let mut held = Vec::new();
    let walked = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, input.to_vec()).expect("the fast engine");
            fast.enable_history().expect("history mode");
            while let Some((logs, history)) = fast.resume_history(window).expect("the run") {
                let last = (fast.pc() == 0)
                    .then(|| fast.take_history_final().map(Box::new))
                    .flatten();
                let sent = tx.send(ExecWindow {
                    logs: std::sync::Arc::new(logs),
                    start: None,
                    history: Some(Box::new(history)),
                    last,
                });
                if sent.is_err() {
                    break;
                }
            }
        });
        let mut absorbed = 0usize;
        walk_windows_from_history(&mut builder, rx, 2, 0, |accumulator, walked, _| {
            for job in accumulator.absorb(walked) {
                let (table, index) = (job.table, job.index);
                if !job.is_records() {
                    continue;
                }
                let mut chunk = job.generate();
                if let Some(records) = chunk.trace.take_records_main() {
                    held.push((table, index, records));
                }
            }
            absorbed += 1;
            if windows.is_some_and(|n| absorbed >= n) {
                return Err(crate::Error::Prover(ENOUGH.to_string()));
            }
            Ok(())
        })
    });
    match walked {
        Ok(_) => {}
        Err(crate::Error::Prover(e)) if e == ENOUGH => {}
        Err(e) => panic!("the record-mode walk: {e:?}"),
    }
    held
}

/// Each group's MEMW_R tables as the host fills them: (the table's first
/// column in the group's store, its columns' words).
fn memw_r_host_words(groups: &[Vec<&Records>]) -> Vec<Vec<(usize, Vec<Vec<u64>>)>> {
    use multilinear::records::RecordsFill;
    groups
        .iter()
        .map(|group| {
            let mut column = 0usize;
            let mut out = Vec::new();
            for records in group {
                if matches!(records.fill(), RecordsFill::MemwR { .. }) {
                    let packed = records.host_fill().expect("the host fills MEMW_R");
                    out.push((
                        column,
                        (0..packed.cols()).map(|c| packed.column(c)).collect(),
                    ));
                }
                column += records.cols();
            }
            out
        })
        .collect()
}

/// The same run's chunks walked as rows and as words, grouped alike: the
/// groups in [`Mode::Words`] (forwards from an empty cache, then backwards
/// with it warm) against the rows' groups in today's upload, word for word,
/// and each words-held MEMW_R table's columns against the host's fill of its
/// decoded rows. Returns the counters at the passes' end, today's stores, and
/// the MEMW_R tables compared with the host's fill.
fn compare_words(
    rows: &[Vec<&Records>],
    words: &[Vec<&Records>],
    what: &str,
) -> (card_rows::Stats, Vec<Vec<Vec<u64>>>, usize) {
    assert_eq!(rows.len(), words.len(), "{what}: the same groups");
    let forwards: Vec<usize> = (0..rows.len()).collect();
    let backwards: Vec<usize> = (0..rows.len()).rev().collect();
    card_rows::clear();
    let today: Vec<Vec<Vec<u64>>> = stores(rows, &forwards, Mode::Off)
        .into_iter()
        .map(|(g, store)| store.unwrap_or_else(|| panic!("{what}: today's group {g} declined")))
        .collect();
    let host = memw_r_host_words(words);
    let mut against_host = 0usize;
    card_rows::clear();
    for order in [&forwards, &backwards] {
        for (g, got) in stores(words, order, Mode::Words) {
            let got = got.unwrap_or_else(|| panic!("{what}: words declined group {g}"));
            let (count, first) = mismatches(&got, &today[g]);
            assert_eq!(
                count, 0,
                "{what}: words group {g}: {count} words differ, the first at {first:?}"
            );
            for (column, cols) in &host[g] {
                let (count, first) = mismatches(&got[*column..*column + cols.len()], cols);
                assert_eq!(
                    count, 0,
                    "{what}: words group {g}: the MEMW_R table at column {column}: \
                     {count} words differ from the host's fill, the first at {first:?}"
                );
                against_host += 1;
            }
        }
    }
    (card_rows::stats(), today, against_host)
}

/// ★ Record-mode runs walked as rows and as words, windows of 7 and 33
/// cycles, CPU and MEMW_R chunks of 32 rows, in groups of 3 and 8: under
/// `words` every group writes today's words both ways, MEMW_R's tables equal
/// the host's fill, and the card built MEMW_R tables from words.
#[test]
fn card_built_words_write_todays_words_forwards_and_backwards() {
    let _reset = Reset::new();
    let max_rows = MaxRowsConfig::small();
    let (mut derived, mut words_sent, mut against_host, mut not_held) = (0u64, 0u64, 0usize, 0u64);
    for name in [
        "all_instructions_64",
        "all_loadstore_32",
        "test_keccak_multi",
        "test_commit_split",
        "test_copy_ecalls",
        "test_ecsm",
        "test_blake3_absorb",
        "misalign_sd",
    ] {
        let (program, _) = run(name);
        for window in [7usize, 33] {
            let rows = history_chunks_as(&program, &[], &max_rows, window, None, false);
            let words = history_chunks_as(&program, &[], &max_rows, window, None, true);
            assert_eq!(
                rows.iter().map(|c| (c.0, c.1)).collect::<Vec<_>>(),
                words.iter().map(|c| (c.0, c.1)).collect::<Vec<_>>(),
                "{name}: the same chunks in both forms"
            );
            for size in [3usize, 8] {
                let (s, _, host) = compare_words(
                    &grouped(&rows, size),
                    &grouped(&words, size),
                    &format!("{name}, window {window}, groups of {size}"),
                );
                assert_eq!(s.declined, 0, "{name}: a group declined");
                derived += s.derived_memw_r;
                words_sent += s.register_words;
                not_held += s.whole.not_held;
                against_host += host;
            }
        }
    }
    println!(
        "CARD ROWS WORDS SMALL: MEMW_R built from words {derived} · rows as words {words_sent} · decoded on the host (not held) {not_held} · MEMW_R tables equal to the host's fill {against_host}"
    );
    assert!(
        derived > 0 && words_sent > 0 && against_host > 0,
        "the mechanism did not move: built {derived}, words {words_sent}, compared {against_host}"
    );
}

/// The real block (`BLOCK_WHIR_ELF`, `BLOCK_WHIR_INPUT`) at the production
/// heights in record mode, its first `CARD_ROWS_REGISTER_WINDOWS` windows
/// (default 20) walked as rows and as words, in groups of ten: under `words`
/// against today's, forwards and backwards, and MEMW_R's tables against the
/// host's fill. Then one mutant at a time, each required to build a MEMW_R
/// table from words and move a group (other words, or the group declined):
/// the kernel reading `rs2`'s row as `rs1`'s; the kernel reading a write's
/// old value whatever its slot; the walker sending an ECALL cycle's rows as
/// words; the walker sending a write whose old access lies before its window
/// as a word.
#[test]
#[ignore = "card: a box test (BLOCK_WHIR_ELF, BLOCK_WHIR_INPUT)"]
fn card_built_words_on_a_real_block_write_todays_words() {
    use super::super::register_words::{FAULT_WORDS, WORDS_FAULT_BEFORE, WORDS_FAULT_ECALL};
    use std::sync::atomic::Ordering;
    let _reset = Reset::new();
    let elf =
        std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF")).expect("the ELF");
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).expect("input");
    let windows: usize = std::env::var("CARD_ROWS_REGISTER_WINDOWS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(20);
    let program = Elf::load(&elf).expect("the ELF loads");
    let options = crate::block_whir::BlockOptions::production();
    let window = 1usize << options.window_log2.expect("the block streams");
    let walk = |words: bool| {
        history_chunks_as(
            &program,
            &input,
            &options.max_rows,
            window,
            Some(windows),
            words,
        )
    };
    let t = std::time::Instant::now();
    let rows = walk(false);
    let words = walk(true);
    println!(
        "CARD ROWS WORDS REAL: {} chunks ({} MEMW_R) of the first {windows} windows, both forms, in {:.1}s",
        words.len(),
        words
            .iter()
            .filter(|c| c.0 == StreamTable::MemwRegister)
            .count(),
        t.elapsed().as_secs_f64()
    );
    assert_eq!(rows.len(), words.len(), "the same chunks in both forms");
    let (row_groups, word_groups) = (grouped(&rows, 10), grouped(&words, 10));
    let (s, today, against_host) = compare_words(&row_groups, &word_groups, "the real block");
    let w = s.whole;
    println!(
        "CARD ROWS WORDS REAL VERDICT: 0 words differ in {} groups × 2 passes · MEMW_R tables equal to the host's fill {against_host} · MEMW_R built from words {} · rows as words {} · whole from the walk {} · decoded on the host (not held) {} · words {} B · whole rows {} B",
        word_groups.len(),
        s.derived_memw_r,
        s.register_words,
        w.other,
        w.not_held,
        s.register_word_bytes,
        s.register_whole_bytes
    );
    assert!(
        s.derived_memw_r > 0 && s.register_words > 0 && against_host > 0,
        "no MEMW_R table built from words on the real block: {s:?}"
    );

    let forwards: Vec<usize> = (0..word_groups.len()).collect();
    let backwards: Vec<usize> = (0..word_groups.len()).rev().collect();
    let moved_by = |groups: &[Vec<&Records>]| {
        let (mut moved, mut uploads) = (0usize, 0usize);
        for order in [&forwards, &backwards] {
            for (g, got) in stores(groups, order, Mode::Words) {
                uploads += 1;
                moved += usize::from(match got {
                    None => true,
                    Some(got) => mismatches(&got, &today[g]).0 > 0,
                });
            }
        }
        (moved, uploads)
    };
    for (name, bit) in [
        ("rs2 read as rs1", card_rows::REGISTER_FAULT_SLOT),
        (
            "old value read whatever its slot",
            card_rows::REGISTER_FAULT_VALUE_AT,
        ),
    ] {
        card_rows::clear();
        card_rows::FAULT_REGISTERS.store(bit, Ordering::Relaxed);
        let (moved, uploads) = moved_by(&word_groups);
        let after = card_rows::stats();
        card_rows::FAULT_REGISTERS.store(0, Ordering::Relaxed);
        println!(
            "CARD ROWS WORDS MUTANT {name}: {moved} of {uploads} groups moved · MEMW_R tables built from words {} · declined {}",
            after.derived_memw_r, after.declined
        );
        assert!(
            after.derived_memw_r > 0,
            "{name}: no MEMW_R table built from words"
        );
        assert!(moved > 0, "{name}: the mutant moved no group");
    }
    for (name, bit) in [
        ("an ECALL cycle's rows as words", WORDS_FAULT_ECALL),
        ("a write before its window as a word", WORDS_FAULT_BEFORE),
    ] {
        FAULT_WORDS.store(bit, Ordering::Relaxed);
        let faulted = walk(true);
        FAULT_WORDS.store(0, Ordering::Relaxed);
        let groups = grouped(&faulted, 10);
        card_rows::clear();
        let (moved, uploads) = moved_by(&groups);
        let after = card_rows::stats();
        println!(
            "CARD ROWS WORDS MUTANT {name}: {moved} of {uploads} groups moved · MEMW_R tables built from words {} · declined {}",
            after.derived_memw_r, after.declined
        );
        assert!(
            after.derived_memw_r > 0,
            "{name}: no MEMW_R table built from words"
        );
        assert!(moved > 0, "{name}: the mutant moved no group");
    }
}

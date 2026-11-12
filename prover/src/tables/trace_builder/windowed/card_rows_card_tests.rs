//! ★ The card-built rows' upload on the card (`math_cuda::card_rows`,
//! `LAMBDA_VM_CARD_ROWS`): a
//! group's upload in the knob's `order` (every copy, then every kernel) and
//! `derive` (CPU logs kept on the card; LOAD and STORE derived from them)
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

/// Every group in `order` and `derive`, forwards and backwards, against
/// today's: the words differing in each (0 everywhere). Returns the counters
/// of the derive mode's two passes and today's stores, group by group.
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
    let mut derive_stats = card_rows::Stats::default();
    for (mode, order) in [
        (Mode::Order, &forwards),
        (Mode::Derive, &forwards),
        (Mode::Derive, &backwards),
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
        if mode == Mode::Derive {
            let s = card_rows::stats();
            derive_stats = s;
        }
    }
    let today = today
        .into_iter()
        .map(|(_, store)| store.expect("today's store"))
        .collect();
    (derive_stats, today)
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
    let outside = stores(&groups, &[0], Mode::Derive);
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

    // The mutant: the derive passes again (forwards from an empty cache, then
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
            for (g, got) in stores(&groups, order, Mode::Derive) {
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

//! ★ The card derives MEMW_A's rows from the executor's history
//! (`derive_memw_a_from_cache`) and fills the table with the host fill's
//! words: every whole MEMW_A chunk of record-mode runs (asm programs, a Rust
//! guest, an ethrex block), uploaded beside the CPU chunks whose logs it
//! reads; and with `BLOCK_WHIR_ELF` and `BLOCK_WHIR_INPUT` every MEMW_A chunk
//! of a real block at the production heights. A store's old read as its
//! value makes other words; a chunk whose records break a rule declines the
//! group.
//!
//! Card only: without a device every upload is declined, and these tests say
//! so instead of passing.

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;

use executor::elf::Elf;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu::{TableColumns, upload_tables};
use multilinear::records::{ALIGNED_ECALL, Records};

use super::card_records::{AlignedRefRecords, CardDecode, CpuLogRecords};
use super::card_records_card_tests::mismatches;
use super::history_rows::RefWindow;
use super::history_rows_tests::{Walked, chunks, walked};
use super::history_sides_tests::rust_elf;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::tests::windowed_builder_tests::run;

fn host_words(records: &Records) -> Vec<Vec<u64>> {
    let packed = records.host_fill().expect("the host fills it");
    (0..packed.cols()).map(|c| packed.column(c)).collect()
}

/// A run's MEMW_A chunks held as refs, each with the whole CPU chunks of
/// `cpu_rows` cycles whose logs its regular rows read (`None` where the walked
/// cycles do not cover them in whole CPU chunks).
struct Run {
    name: String,
    memw_a: Vec<(Records, Option<Vec<Records>>)>,
}

fn held(
    name: &str,
    walked: &[Walked],
    decode: Arc<crate::tables::decode::DecodeTable>,
    rows: usize,
    cpu_rows: usize,
) -> Run {
    let order = MemwAlignedOrder::Limbs;
    let card = Arc::new(CardDecode::new(decode).expect("DECODE rows"));
    let logs: Vec<executor::vm::logs::Log> =
        walked.iter().flat_map(|w| w.logs.iter().cloned()).collect();
    let cpu: Vec<Records> = (0..logs.len() / cpu_rows)
        .map(|k| {
            let range = k * cpu_rows..(k + 1) * cpu_rows;
            let chunk = CpuLogRecords::new(
                super::Segments::one(logs[range].to_vec()),
                (k * cpu_rows) as u64,
                Arc::clone(&card),
            )
            .expect("a whole CPU chunk");
            Records(Arc::new(chunk))
        })
        .collect();
    let mut memw_a = Vec::new();
    for chunk in chunks(walked, rows) {
        let windows: Vec<Arc<RefWindow>> = chunk.windows.clone();
        let cycles: Vec<u64> = chunk
            .refs
            .iter()
            .filter(|r| r.cycle & ALIGNED_ECALL == 0)
            .map(|r| u64::from(r.cycle))
            .collect();
        let records = AlignedRefRecords::new(chunk.refs, windows, Arc::clone(&card), order)
            .unwrap_or_else(|_| panic!("{name}: a whole chunk"));
        let covering = match (cycles.iter().min(), cycles.iter().max()) {
            (Some(&lo), Some(&hi)) if (hi as usize) < cpu.len() * cpu_rows => Some(
                (lo as usize / cpu_rows..=hi as usize / cpu_rows)
                    .map(|k| cpu[k].clone())
                    .collect(),
            ),
            _ => None,
        };
        memw_a.push((Records(Arc::new(records)), covering));
    }
    Run {
        name: name.to_string(),
        memw_a,
    }
}

/// The small runs and a Rust guest: asm programs in windows of 7 (MEMW_A
/// chunks of 16 rows, CPU chunks of 64 cycles) and a guest in windows of 97
/// (chunks of 4096, CPU chunks of 4096).
fn small_runs() -> Vec<Run> {
    let order = MemwAlignedOrder::Limbs;
    let mut out = Vec::new();
    for name in [
        "all_loadstore_32",
        "test_keccak_multi",
        "test_copy_ecalls",
        "test_commit_split",
    ] {
        let (program, _) = run(name);
        let (w, decode) = walked(&program, &[], 7, order);
        out.push(held(name, &w, decode, 16, 64));
    }
    let program = rust_elf("dma_memmove_cases");
    let (w, decode) = walked(&program, &[], 97, order);
    out.push(held("dma_memmove_cases", &w, decode, 4096, 4096));
    out
}

/// The group `cpu` then `memw_a` filled on the card, MEMW_A's columns (the
/// last 29), and whether the card derived it.
fn card_memw_a(cpu: &[Records], memw_a: &Records) -> Option<(Vec<Vec<u64>>, bool)> {
    let mut tables: Vec<TableColumns<'_, F>> = cpu.iter().map(TableColumns::Records).collect();
    tables.push(TableColumns::Records(memw_a));
    let before = math_cuda::card_rows::stats().derived_memw_a;
    let mut columns = upload_tables(&tables)?.download()?;
    let derived = math_cuda::card_rows::stats().derived_memw_a > before;
    let at = columns.len() - memw_a.cols();
    Some((columns.split_off(at), derived))
}

/// ★ Every MEMW_A chunk of the small runs whose cycles whole CPU chunks
/// cover is derived on the card with the host fill's words, all 29 columns.
#[test]
fn the_card_derives_memw_a_rows_as_the_host_does() {
    let (mut derived, mut chunks) = (0usize, 0usize);
    for run in small_runs() {
        for (k, (records, cpu)) in run.memw_a.iter().enumerate() {
            let Some(cpu) = cpu else { continue };
            let what = format!("{}, MEMW_A[{k}]", run.name);
            let (card, on_card) = card_memw_a(cpu, records).expect("the card takes the group");
            assert!(on_card, "{what}: derived on the card");
            assert_eq!(mismatches(&card, &host_words(records)), (0, None), "{what}");
            derived += 1;
        }
        chunks += run.memw_a.len();
    }
    println!("CARD MEMW_A SMALL: {derived} of {chunks} chunks derived, 0 words differ");
    assert!(derived >= 8, "{derived} chunks derived");
}

/// A store's old read as its value makes other words on some chunk; a chunk
/// whose regular row's access is stamped at the row's own timestamp (no step
/// for the limbs) declines the group.
#[test]
fn the_card_catches_a_wrong_memw_a_row() {
    use math_cuda::card_rows::{FAULT_MEMW_A, MEMW_A_FAULT_STORE_OLD};
    let runs = small_runs();
    let mut caught = 0usize;
    FAULT_MEMW_A.store(MEMW_A_FAULT_STORE_OLD, Relaxed);
    for run in &runs {
        for (records, cpu) in &run.memw_a {
            let Some(cpu) = cpu else { continue };
            caught += match card_memw_a(cpu, records) {
                None => 1,
                Some((card, _)) => usize::from(mismatches(&card, &host_words(records)).0 > 0),
            };
        }
    }
    FAULT_MEMW_A.store(0, Relaxed);
    println!("CARD MEMW_A MUTANT store's old read as its value: caught on {caught} chunks");
    assert!(caught > 0);
    // The edit: a covered chunk's first regular row's access stamped at its
    // own timestamp, over a copy of that row's window.
    let order = MemwAlignedOrder::Limbs;
    let (program, _) = run("all_loadstore_32");
    let (w, decode) = walked(&program, &[], 7, order);
    let base = held("all_loadstore_32", &w, Arc::clone(&decode), 16, 64);
    let k = base
        .memw_a
        .iter()
        .position(|(_, cpu)| cpu.is_some())
        .expect("a covered chunk");
    let cpu = base.memw_a[k].1.clone().expect("covered");
    let mut chunk = chunks(&w, 16).swap_remove(k);
    let (part, r) = chunk
        .refs
        .slices()
        .iter()
        .enumerate()
        .find_map(|(i, refs)| {
            let accesses = &chunk.windows[i].history.accesses;
            refs.iter()
                .find(|r| {
                    r.cycle & ALIGNED_ECALL == 0
                        && accesses[r.at as usize].mixed == executor::vm::history::NO_ACCESS
                })
                .map(|r| (i, *r))
        })
        .expect("a regular row of one old timestamp");
    let window = &chunk.windows[part];
    let mut history = (*window.history).clone();
    history.accesses[r.at as usize].ts = 4 * r.cycle + 4;
    chunk.windows[part] = Arc::new(RefWindow {
        first: window.first,
        logs: Arc::clone(&window.logs),
        history: Arc::new(history),
        whole: window.whole.clone(),
    });
    let card = Arc::new(CardDecode::new(decode).expect("DECODE rows"));
    let edited = AlignedRefRecords::new(chunk.refs, chunk.windows, card, order)
        .unwrap_or_else(|_| panic!("a whole chunk"));
    let edited = Records(Arc::new(edited));
    assert!(edited.host_fill().is_none(), "the host refuses the row");
    assert!(
        card_memw_a(&cpu, &edited).is_none(),
        "a row with no step declines the group"
    );
}

/// ★ A real block (`BLOCK_WHIR_ELF`, `BLOCK_WHIR_INPUT`) walked at the
/// production window: every whole MEMW_A chunk of the production height,
/// held as refs, is derived on the card with the host fill's words, beside
/// the CPU chunks of the production height whose logs it reads. One `CARD
/// MEMW_A` line a chunk and a closing `CARD MEMW_A VERDICT`.
#[test]
#[ignore = "card: a box test (BLOCK_WHIR_ELF, BLOCK_WHIR_INPUT)"]
fn the_card_derives_every_memw_a_chunk_of_a_real_block() {
    let elf =
        std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF")).expect("the ELF");
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).expect("input");
    let program = Elf::load(&elf).expect("the ELF loads");
    let options = crate::block_whir::BlockOptions::production();
    let window = 1usize << options.window_log2.expect("the block streams");
    let t = std::time::Instant::now();
    let (w, decode) = walked(&program, &input, window, MemwAlignedOrder::Limbs);
    let run = held(
        "real",
        &w,
        decode,
        options.max_rows.memw_aligned,
        options.max_rows.cpu,
    );
    assert!(!run.memw_a.is_empty(), "no MEMW_A chunk");
    let (mut derived, mut differ, mut uncovered) = (0usize, 0usize, 0usize);
    for (k, (records, cpu)) in run.memw_a.iter().enumerate() {
        let Some(cpu) = cpu else {
            uncovered += 1;
            continue;
        };
        let c = std::time::Instant::now();
        let (card, on_card) = card_memw_a(cpu, records).expect("the card takes the group");
        let (count, first) = mismatches(&card, &host_words(records));
        derived += usize::from(on_card);
        differ += count;
        println!(
            "CARD MEMW_A[{k}] rows={} cpu_chunks={} derived={on_card} words_differ={count} first={first:?} secs={:.2}",
            records.rows(),
            cpu.len(),
            c.elapsed().as_secs_f64()
        );
    }
    let covered = run.memw_a.len() - uncovered;
    let verdict = if differ == 0 && derived == covered && derived > 0 {
        "PASS"
    } else {
        "FAIL"
    };
    println!(
        "CARD MEMW_A VERDICT {verdict} chunks={} derived={derived} uncovered={uncovered} words_differ={differ} secs={:.1}",
        run.memw_a.len(),
        t.elapsed().as_secs_f64()
    );
    assert_eq!(verdict, "PASS");
}

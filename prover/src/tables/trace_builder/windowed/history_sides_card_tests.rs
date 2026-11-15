//! ★ The card derives the in-row CPU rows' side records from the executor's
//! history (`derive_cpu_sides`) and fills the CPU table with the host fill's
//! words: every full CPU chunk of record-mode runs (asm programs with
//! keep-alives, a Rust guest), and with `BLOCK_WHIR_ELF` and
//! `BLOCK_WHIR_INPUT` every CPU chunk of a real block at the production
//! height; and each of the kernel's mutants makes other words or declines the
//! group.
//!
//! Card only: without a device every upload is declined, and these tests say
//! so instead of passing.

use std::sync::atomic::Ordering::Relaxed;

use executor::elf::Elf;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu::{TableColumns, upload_tables};
use multilinear::records::Records;

use super::card_records_card_tests::{mismatches, uploaded};
use super::history_sides_tests::{cpu_records_chunks, derives, rust_elf};
use crate::tables::MaxRowsConfig;
use crate::tests::windowed_builder_tests::run;

fn host_words(records: &Records) -> Vec<Vec<u64>> {
    let packed = records.host_fill().expect("the host fills it");
    (0..packed.cols()).map(|c| packed.column(c)).collect()
}

/// `records` filled on the card against its host fill: `(words that differ,
/// the first)`, the card having filled it from its records.
fn card_against_host(what: &str, records: &Records) -> (usize, Option<(usize, usize)>) {
    assert!(
        derives(records),
        "{what}: the card derives its side records"
    );
    let fills = multilinear::records::fills().0;
    let card = uploaded(&[TableColumns::Records(records)]);
    assert_eq!(
        multilinear::records::fills().0,
        fills + 1,
        "{what}: filled on the card"
    );
    mismatches(&card, &host_words(records))
}

/// The small runs' derived CPU chunks: `(what, records)`.
fn small_chunks() -> Vec<(String, Records)> {
    let max_rows = MaxRowsConfig::small();
    let mut out = Vec::new();
    let mut runs: Vec<(String, Elf, usize)> = [
        "all_instructions_64",
        "test_keccak_multi",
        "test_ecsm",
        "test_commit_split",
        "keep_alive_regs",
    ]
    .iter()
    .flat_map(|name| [1usize, 7, 33].map(|w| (name.to_string(), run(name).0, w)))
    .collect();
    runs.push((
        "dma_memmove_cases".to_string(),
        rust_elf("dma_memmove_cases"),
        97,
    ));
    for (name, program, window) in &runs {
        for (index, records) in cpu_records_chunks(program, &[], &max_rows, *window, true) {
            out.push((format!("{name}, window {window}, CPU[{index}]"), records));
        }
    }
    out
}

/// ★ Every derived CPU chunk of the small runs fills on the card with the
/// host fill's words, all 42 columns.
#[test]
fn the_card_derives_the_side_records_and_fills_as_the_host_does() {
    let chunks = small_chunks();
    assert!(chunks.len() > 8, "{} chunks", chunks.len());
    for (what, records) in &chunks {
        assert_eq!(card_against_host(what, records), (0, None), "{what}");
    }
    println!("CARD SIDES SMALL: {} chunks, 0 words differ", chunks.len());
}

/// Each of the kernel's mutants (rs2's stamp read as rs1's; no keep-alive;
/// rd's old value read whatever its slot; the exceptions ignored) makes
/// other words than the host's, or declines the group, on some chunk.
#[test]
fn each_mutant_of_the_derivation_is_caught() {
    use math_cuda::fill::{
        FAULT_SIDES, SIDES_FAULT_EXCEPTIONS, SIDES_FAULT_KEEP_ALIVE, SIDES_FAULT_SLOT,
        SIDES_FAULT_VALUE,
    };
    let chunks = small_chunks();
    for (fault, name) in [
        (SIDES_FAULT_SLOT, "rs2's stamp read as rs1's"),
        (SIDES_FAULT_KEEP_ALIVE, "no keep-alive"),
        (SIDES_FAULT_VALUE, "rd's old value whatever its slot"),
        (SIDES_FAULT_EXCEPTIONS, "the exceptions ignored"),
    ] {
        let mut caught = 0usize;
        FAULT_SIDES.store(fault, Relaxed);
        for (_, records) in &chunks {
            let card =
                upload_tables::<F>(&[TableColumns::Records(records)]).and_then(|s| s.download());
            caught += match card {
                None => 1,
                Some(card) => usize::from(mismatches(&card, &host_words(records)).0 > 0),
            };
        }
        FAULT_SIDES.store(0, Relaxed);
        println!(
            "CARD SIDES MUTANT {name}: caught on {caught} of {} chunks",
            chunks.len()
        );
        assert!(caught > 0, "{name}: caught on no chunk");
    }
}

/// ★ A real block (`BLOCK_WHIR_ELF`, `BLOCK_WHIR_INPUT`) at the production
/// height and window: every full CPU chunk carries its history records, and
/// the card's fill of it is the host fill's, every word. One `CARD SIDES CPU`
/// line a chunk and a closing `CARD SIDES VERDICT`.
#[test]
#[ignore = "card: a box test (BLOCK_WHIR_ELF, BLOCK_WHIR_INPUT)"]
fn the_card_derives_every_cpu_chunks_side_records_of_a_real_block() {
    let elf =
        std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF")).expect("the ELF");
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).expect("input");
    let program = Elf::load(&elf).expect("the ELF loads");
    let options = crate::block_whir::BlockOptions::production();
    let window = 1usize << options.window_log2.expect("the block streams");
    let t = std::time::Instant::now();
    let chunks = cpu_records_chunks(&program, &input, &options.max_rows, window, true);
    assert!(!chunks.is_empty(), "no CPU chunk held");
    let (mut listed, mut differ) = (0usize, 0usize);
    for (k, (index, records)) in chunks.iter().enumerate() {
        assert_eq!(*index, k, "every CPU chunk, in order");
        let c = std::time::Instant::now();
        let (count, first) = card_against_host(&format!("real CPU[{index}]"), records);
        let multilinear::records::RecordsFill::Cpu {
            history: Some(h), ..
        } = records.fill()
        else {
            panic!("real CPU[{index}]: derived");
        };
        let n: usize = h.exceptions.iter().map(|p| p.len()).sum();
        listed += n;
        differ += count;
        println!(
            "CARD SIDES CPU[{index}] rows={} exceptions={n} words_differ={count} first={first:?} secs={:.2}",
            records.rows(),
            c.elapsed().as_secs_f64()
        );
    }
    let verdict = if differ == 0 { "PASS" } else { "FAIL" };
    println!(
        "CARD SIDES VERDICT {verdict} chunks={} exceptions={listed} words_differ={differ} secs={:.1}",
        chunks.len(),
        t.elapsed().as_secs_f64()
    );
    assert_eq!(differ, 0);
}

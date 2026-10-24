//! ★ The card fills a streamed CPU or MEMW_R chunk from its records
//! ([`super::card_records`]) with the words the host fill writes, at the
//! places its group's store holds them: chunks whose logs or rows lie in
//! many window parts, alone and among a group's other tables, and, with
//! `BLOCK_WHIR_ELF` and `BLOCK_WHIR_INPUT`, a real block's chunks at the
//! production height. A CPU log at a pc with no DECODE row declines the group,
//! as the host fill refuses it; two CPU columns swapped are caught (with
//! `test-cuda-faults`).
//!
//! Card only: without a device every upload is declined, and these tests say
//! so instead of passing.

use std::sync::Arc;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu::{TableColumns, upload_tables};
use multilinear::mle::Mle;
use multilinear::narrow::NarrowColumns;
use multilinear::records::Records;

use super::card_records::{CardDecode, CpuLogRecords};
use super::{RecordsRoom, Segments, StreamTable, WindowedTraceBuilder};
use crate::tables::MaxRowsConfig;
use crate::tests::windowed_builder_tests::run;

fn always() -> RecordsRoom {
    Arc::new(|_| true)
}

/// Every full CPU and MEMW_R chunk the run of `program` hands out as records,
/// by windows of `window` cycles, from the hand-out of window `from` on and
/// through `windows` windows walked (all of them: `None`):
/// `(family, index, records)`.
fn records_chunks(
    program: &Elf,
    input: &[u8],
    max_rows: &MaxRowsConfig,
    window: usize,
    (from, windows): (usize, Option<usize>),
) -> Vec<(StreamTable, usize, Records)> {
    let mut builder = WindowedTraceBuilder::new(program, input, max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(always(), true, true))
        .expect("the builder");
    let mut executor = Executor::new(program, input.to_vec()).expect("the executor starts");
    let mut held = Vec::new();
    let (mut walker, mut accumulator) = builder.split();
    let mut walked = 0usize;
    let mut ahead: Option<Vec<Log>> = None;
    while let Some(logs) = executor.resume_with_limit(window).expect("the run") {
        // The run's last window is the finish's.
        if let Some(previous) = ahead.replace(logs.to_vec()) {
            let w = walker.walk_owned(previous).expect("a window");
            for job in accumulator.absorb(w) {
                let (table, index) = (job.table, job.index);
                if walked < from || !job.is_records() {
                    continue;
                }
                let mut chunk = job.generate();
                if let Some(records) = chunk.trace.take_records_main() {
                    held.push((table, index, records));
                }
            }
            walked += 1;
        }
        if windows.is_some_and(|n| walked >= n) {
            break;
        }
    }
    held
}

/// The group's store, column by column, as the card holds it.
fn uploaded(tables: &[TableColumns<'_, F>]) -> Vec<Vec<u64>> {
    upload_tables(tables)
        .expect("the card takes the group")
        .download()
        .expect("the store comes back")
}

/// Mismatched words between two stores, and the first.
fn mismatches(got: &[Vec<u64>], want: &[Vec<u64>]) -> (usize, Option<(usize, usize)>) {
    let mut count = 0;
    let mut first = None;
    assert_eq!(got.len(), want.len(), "the columns");
    for (c, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.len(), w.len(), "column {c}'s rows");
        for (r, (a, b)) in g.iter().zip(w).enumerate() {
            if a != b {
                count += 1;
                first.get_or_insert((c, r));
            }
        }
    }
    (count, first)
}

fn host_words(packed: &NarrowColumns) -> Vec<Vec<u64>> {
    (0..packed.cols()).map(|c| packed.column(c)).collect()
}

/// `records`' table filled on the card against its host fill, uploaded
/// packed and as the host's words.
fn check(what: &str, records: &Records) {
    let packed = records.host_fill().expect("the host fills it");
    let fills = multilinear::records::fills().0;
    let card = uploaded(&[TableColumns::Records(records)]);
    assert_eq!(
        multilinear::records::fills().0,
        fills + 1,
        "{what}: filled on the card"
    );
    assert_eq!(
        mismatches(&card, &uploaded(&[TableColumns::Narrow(&packed)])),
        (0, None),
        "{what}: against the widened host fill"
    );
    assert_eq!(
        mismatches(&card, &host_words(&packed)),
        (0, None),
        "{what}: against the host's words"
    );
}

#[test]
fn the_card_fills_cpu_and_memw_r_chunks_from_their_records_as_the_host_does() {
    let max_rows = MaxRowsConfig::small();
    let mut filled = [0usize; 2];
    for name in ["all_instructions_64", "test_keccak_multi", "test_ecsm"] {
        let (program, _) = run(name);
        for window in [1, 7, 33] {
            for (table, index, records) in
                records_chunks(&program, &[], &max_rows, window, (0, None))
            {
                check(
                    &format!("{name}, window {window}, {table:?}[{index}]"),
                    &records,
                );
                filled[usize::from(table == StreamTable::MemwRegister)] += 1;
            }
        }
    }
    assert!(filled[0] > 0 && filled[1] > 0, "{filled:?}");
}

/// Among a wide table, a packed one and each other, the filled tables'
/// columns sit where their widened columns would, and the others are as
/// uploaded.
#[test]
fn the_card_fills_cpu_and_memw_r_in_place_among_a_groups_tables() {
    let (program, _) = run("test_keccak_multi");
    let chunks = records_chunks(&program, &[], &MaxRowsConfig::small(), 7, (0, None));
    let cpu = &chunks
        .iter()
        .find(|(t, ..)| *t == StreamTable::Cpu)
        .expect("a CPU chunk")
        .2;
    let memw_r = &chunks
        .iter()
        .find(|(t, ..)| *t == StreamTable::MemwRegister)
        .expect("a MEMW_R chunk")
        .2;
    let wide: Vec<Mle<F>> = (0..4u64)
        .map(|c| {
            Mle::new(
                (0..16u64)
                    .map(|r| FieldElement::from(c * 1000 + r))
                    .collect(),
            )
            .expect("a column")
        })
        .collect();
    let (cpu_packed, memw_r_packed) = (
        cpu.host_fill().expect("the host fills it"),
        memw_r.host_fill().expect("the host fills it"),
    );
    let card = uploaded(&[
        TableColumns::Records(memw_r),
        TableColumns::Wide(&wide),
        TableColumns::Records(cpu),
        TableColumns::Narrow(&memw_r_packed),
        TableColumns::Records(memw_r),
    ]);
    let host = uploaded(&[
        TableColumns::Narrow(&memw_r_packed),
        TableColumns::Wide(&wide),
        TableColumns::Narrow(&cpu_packed),
        TableColumns::Narrow(&memw_r_packed),
        TableColumns::Narrow(&memw_r_packed),
    ]);
    assert_eq!(mismatches(&card, &host), (0, None));
}

/// A CPU chunk with one log at a pc the program has no instruction at: the
/// card declines its group (the kernel counts the log), as the host fill
/// refuses it; the same chunk untouched goes up.
#[test]
fn a_cpu_log_at_a_pc_with_no_decode_row_declines_the_group() {
    let (program, logs) = run("all_instructions_64");
    let max_rows = MaxRowsConfig::small();
    let builder = WindowedTraceBuilder::new(&program, &[], &max_rows).expect("the builder");
    let decode = Arc::new(CardDecode::new(Arc::clone(&builder.decode)).expect("a program"));
    let rows = max_rows.cpu;
    let honest = CpuLogRecords::new(Segments::one(logs[..rows].to_vec()), 0, Arc::clone(&decode))
        .expect("a whole table");
    let honest = Records(Arc::new(honest));
    assert!(upload_tables::<F>(&[TableColumns::Records(&honest)]).is_some());
    // A pc past every instruction, 4-aligned.
    let absent = decode
        .records
        .entries()
        .iter()
        .map(|e| e.pc)
        .max()
        .expect("a row")
        + 0x1000;
    assert_eq!(decode.records.row(absent), None);
    for at in [0, rows / 2, rows - 1] {
        let mut tampered = logs[..rows].to_vec();
        tampered[at].current_pc = absent;
        let records = CpuLogRecords::new(Segments::one(tampered), 0, Arc::clone(&decode))
            .expect("a whole table");
        let records = Records(Arc::new(records));
        assert!(
            records.host_fill().is_none(),
            "log {at}: the host refuses it"
        );
        assert!(
            upload_tables::<F>(&[TableColumns::Records(&records)]).is_none(),
            "log {at}: the card declines it"
        );
    }
}

/// Two CPU columns swapped by the fill are caught word by word: the
/// comparison above is not vacuous.
#[cfg(feature = "test-cuda-faults")]
#[test]
fn two_cpu_columns_swapped_are_caught() {
    use std::sync::atomic::Ordering::Relaxed;
    let (program, _) = run("test_keccak_multi");
    let chunks = records_chunks(&program, &[], &MaxRowsConfig::small(), 7, (0, None));
    let (_, index, cpu) = chunks
        .iter()
        .find(|(t, ..)| *t == StreamTable::Cpu)
        .expect("a CPU chunk");
    let packed = cpu.host_fill().expect("the host fills it");
    let columns = math_cuda::fill::CPU_COLUMNS;
    // TIMESTAMP and PC_0.
    math_cuda::fill::FAULT_CPU_SWAP.store(1 + columns, Relaxed);
    let card = upload_tables::<F>(&[TableColumns::Records(cpu)]).and_then(|s| s.download());
    math_cuda::fill::FAULT_CPU_SWAP.store(0, Relaxed);
    let card = card.expect("the card takes it");
    let (count, first) = mismatches(&card, &host_words(&packed));
    assert!(
        count > 0 && first.is_some_and(|(c, _)| c <= 1),
        "CPU[{index}]: columns 0 and 1 swapped, and every word the host's"
    );
}

/// The real block's CPU chunk 32 (windows 64 and 65, two pieces) and the
/// first MEMW_R chunk handed out from window 64, at the production height,
/// filled on the card as on the host; and chunk 32 with one log at an absent
/// pc declined.
#[test]
#[ignore = "card: a box test (BLOCK_WHIR_ELF, BLOCK_WHIR_INPUT)"]
fn the_card_fills_a_real_blocks_cpu_and_memw_r_chunks_as_the_host_does() {
    let elf =
        std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF")).expect("the ELF");
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).expect("input");
    let program = Elf::load(&elf).expect("the ELF loads");
    let options = crate::block_whir::BlockOptions::production();
    let window = 1usize << options.window_log2.expect("the block streams");
    let t = std::time::Instant::now();
    let chunks = records_chunks(&program, &input, &options.max_rows, window, (64, Some(66)));
    let cpu = chunks
        .iter()
        .find(|(t, i, _)| *t == StreamTable::Cpu && *i == 32)
        .expect("CPU chunk 32");
    let memw_r = chunks
        .iter()
        .find(|(t, _, r)| {
            *t == StreamTable::MemwRegister
                && matches!(r.fill(), multilinear::records::RecordsFill::MemwR { rows } if rows.len() > 1)
        })
        .or_else(|| chunks.iter().rev().find(|(t, ..)| *t == StreamTable::MemwRegister))
        .expect("a MEMW_R chunk");
    println!(
        "CARD FILL REAL capture_secs={:.2} cpu_rows={} memw_r_index={} memw_r_rows={} held_chunks={}",
        t.elapsed().as_secs_f64(),
        cpu.2.rows(),
        memw_r.1,
        memw_r.2.rows(),
        chunks.len()
    );
    for (table, index, records) in [cpu, memw_r] {
        let t = std::time::Instant::now();
        check(&format!("real {table:?}[{index}]"), records);
        println!(
            "CARD FILL REAL {table:?}[{index}] identity 0 words differ secs={:.2}",
            t.elapsed().as_secs_f64()
        );
    }
    let multilinear::records::RecordsFill::Cpu {
        logs, first_cycle, ..
    } = cpu.2.fill()
    else {
        panic!("CPU chunk 32 holds logs");
    };
    let builder =
        WindowedTraceBuilder::new(&program, &input, &options.max_rows).expect("the builder");
    let decode = Arc::new(CardDecode::new(Arc::clone(&builder.decode)).expect("a program"));
    let absent = decode
        .records
        .entries()
        .iter()
        .map(|e| e.pc)
        .max()
        .expect("a row")
        + 0x1000;
    let mut tampered: Vec<Log> = logs
        .iter()
        .flat_map(|piece| piece.iter())
        .map(|l| Log {
            current_pc: l.current_pc,
            next_pc: l.next_pc,
            src1_val: l.src1_val,
            src2_val: l.src2_val,
            dst_val: l.dst_val,
        })
        .collect();
    let at = tampered.len() / 3;
    tampered[at].current_pc = absent;
    let records = Records(Arc::new(
        CpuLogRecords::new(Segments::one(tampered), first_cycle, decode).expect("a whole table"),
    ));
    assert!(
        upload_tables::<F>(&[TableColumns::Records(&records)]).is_none(),
        "an absent pc declined"
    );
    println!("CARD FILL REAL absent pc at log {at}: declined by the card");
    println!("CARD FILL REAL VERDICT PASS");
}

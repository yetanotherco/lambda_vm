//! ★ The card fills a streamed CPU or MEMW_R chunk, or a row family's
//! (MEMW_A, LOAD, STORE, MEMW, SHIFT), from its records ([`super::card_records`]) with the words the host fill writes, at
//! the places its group's store holds them: chunks whose logs or rows lie in
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
use super::{RecordsRoom, RowFamilies, Segments, StreamTable, WindowedTraceBuilder};
use crate::tables::MaxRowsConfig;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::tables::register::RegisterFormat;
use crate::tests::windowed_builder_tests::run;

pub(super) fn always() -> RecordsRoom {
    Arc::new(|_| true)
}

/// Every full CPU and MEMW_R chunk (with `rows`, every row family's too)
/// the run of `program` hands out as records, by windows of `window` cycles,
/// from the hand-out of window `from` on and through `windows` windows walked
/// (all of them: `None`): `(family, index, records)`.
pub(super) fn records_chunks(
    program: &Elf,
    input: &[u8],
    max_rows: &MaxRowsConfig,
    window: usize,
    (from, windows): (usize, Option<usize>),
    rows: bool,
) -> Vec<(StreamTable, usize, Records)> {
    records_chunks_in(
        program,
        input,
        max_rows,
        window,
        (from, windows),
        rows,
        MemwAlignedOrder::Lt,
    )
}

/// [`records_chunks`] with MEMW_A's rows in `order`'s form.
pub(super) fn records_chunks_in(
    program: &Elf,
    input: &[u8],
    max_rows: &MaxRowsConfig,
    window: usize,
    (from, windows): (usize, Option<usize>),
    rows: bool,
    order: MemwAlignedOrder,
) -> Vec<(StreamTable, usize, Records)> {
    let mut held = Vec::new();
    for_each_records_chunk(
        program,
        input,
        max_rows,
        window,
        (from, windows),
        rows,
        (order, RegisterFormat::Tokens32),
        &mut |table, index, records| held.push((table, index, records)),
    );
    held
}

/// Each chunk [`records_chunks_in`] holds, handed to `each` as it is cut,
/// under the register format `registers` too: under
/// [`RegisterFormat::InRow`] the CPU chunks go up with their side records and
/// MEMW_R's slot (the residual register table) stays on the host, as the
/// block asks.
#[allow(clippy::too_many_arguments)]
pub(super) fn for_each_records_chunk(
    program: &Elf,
    input: &[u8],
    max_rows: &MaxRowsConfig,
    window: usize,
    (from, windows): (usize, Option<usize>),
    rows: bool,
    (order, registers): (MemwAlignedOrder, RegisterFormat),
    each: &mut dyn FnMut(StreamTable, usize, Records),
) {
    let memw_r = registers != RegisterFormat::InRow;
    let mut builder = WindowedTraceBuilder::new(program, input, max_rows)
        .and_then(|b| b.memw_aligned_order(order))
        .and_then(|b| b.register_format(registers))
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(always(), true, memw_r))
        .and_then(|b| {
            if rows {
                b.card_fill_streamed_rows(RowFamilies::ALL)
            } else {
                Ok(b)
            }
        })
        .expect("the builder");
    let mut executor = Executor::new(program, input.to_vec()).expect("the executor starts");
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
                    each(table, index, records);
                }
            }
            walked += 1;
        }
        if windows.is_some_and(|n| walked >= n) {
            break;
        }
    }
}

/// The group's store, column by column, as the card holds it.
pub(super) fn uploaded(tables: &[TableColumns<'_, F>]) -> Vec<Vec<u64>> {
    upload_tables(tables)
        .expect("the card takes the group")
        .download()
        .expect("the store comes back")
}

/// Mismatched words between two stores, and the first.
pub(super) fn mismatches(got: &[Vec<u64>], want: &[Vec<u64>]) -> (usize, Option<(usize, usize)>) {
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
                records_chunks(&program, &[], &max_rows, window, (0, None), false)
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

/// `records` (a CPU chunk under the in-row register format, read with
/// `decode`) with one side record's field moved: a table the run did not
/// make.
fn side_moved(records: &Records, decode: &Arc<CardDecode>, field: usize) -> Records {
    let multilinear::records::RecordsFill::Cpu {
        logs,
        sides,
        first_cycle,
        ..
    } = records.fill()
    else {
        panic!("a CPU chunk");
    };
    let logs: Vec<Log> = logs
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
    let mut sides: Vec<_> = sides
        .iter()
        .flat_map(|piece| piece.iter().copied())
        .collect();
    let at = sides.len() / 2;
    match field {
        0..3 => sides[at].prev_ts[field] ^= 1 << 3,
        _ => sides[at].old_rd[field - 3] ^= 1 << 3,
    }
    Records(Arc::new(
        CpuLogRecords::in_row(
            Segments::one(logs),
            Segments::one(sides),
            first_cycle,
            Arc::clone(decode),
        )
        .expect("a whole table"),
    ))
}

/// ★ The in-row register format: every full CPU chunk, held as its logs and
/// their side records (in the window parts they were cut from), fills on the
/// card with the host fill's words, its 42 columns; and a chunk with one side
/// record's field moved differs from the honest host fill in that row alone,
/// so the comparison reads the five columns.
#[test]
fn the_card_fills_in_row_cpu_chunks_from_their_records_as_the_host_does() {
    let max_rows = MaxRowsConfig::small();
    let mut filled = 0usize;
    for name in [
        "all_instructions_64",
        "test_keccak_multi",
        "test_ecsm",
        "keep_alive_regs",
    ] {
        let (program, _) = run(name);
        let builder = WindowedTraceBuilder::new(&program, &[], &max_rows).expect("the builder");
        let decode = Arc::new(CardDecode::new(Arc::clone(&builder.decode)).expect("a program"));
        for window in [1, 7, 33] {
            for_each_records_chunk(
                &program,
                &[],
                &max_rows,
                window,
                (0, None),
                false,
                (MemwAlignedOrder::Lt, RegisterFormat::InRow),
                &mut |table, index, records| {
                    assert_eq!(table, StreamTable::Cpu, "only CPU is held under in-row");
                    assert_eq!(
                        records.cols(),
                        crate::tables::cpu::inrow_cols::NUM_COLUMNS_INROW
                    );
                    let what = format!("{name}, window {window}, in-row CPU[{index}]");
                    check(&what, &records);
                    if filled == 0 {
                        let honest = host_words(&records.host_fill().expect("the host fills it"));
                        for field in 0..5 {
                            let moved = side_moved(&records, &decode, field);
                            let card = uploaded(&[TableColumns::Records(&moved)]);
                            let (count, first) = mismatches(&card, &honest);
                            assert!(
                                count == 1 && first.is_some_and(|(c, _)| c == 37 + field),
                                "{what}: side field {field} moved reads in its column alone: \
                                 {count} words, first {first:?}"
                            );
                        }
                    }
                    filled += 1;
                },
            );
        }
    }
    assert!(filled > 1, "{filled}");
}

/// A row family's records: its place in [`ROW_FAMILIES`] and its pieces.
fn row_family(records: &Records) -> Option<(usize, usize)> {
    use multilinear::records::RecordsFill as R;
    match records.fill() {
        R::MemwA { rows } => Some((0, rows.len())),
        R::Load { ops } => Some((1, ops.len())),
        R::Store { ops } => Some((2, ops.len())),
        R::Memw { ops } => Some((3, ops.len())),
        R::Shift { ops } => Some((4, ops.len())),
        _ => None,
    }
}

/// The row families, in [`row_family`]'s order.
const ROW_FAMILIES: [&str; 5] = ["MEMW_A", "LOAD", "STORE", "MEMW", "SHIFT"];

/// Every row family's chunks of four rows, so that the small programs fill
/// several, their rows in one window part or many: MEMW_A's (in both of its
/// forms: the step's limbs, or the old timestamp), LOAD's, STORE's and
/// SHIFT's (no small program makes four MEMW ops; the synthetic MEMW chunk
/// below is MEMW's).
#[test]
fn the_card_fills_row_family_chunks_from_their_rows_as_the_host_does() {
    let max_rows = MaxRowsConfig {
        memw_aligned: 1 << 2,
        load: 1 << 2,
        store: 1 << 2,
        memw: 1 << 2,
        shift: 1 << 2,
        ..MaxRowsConfig::small()
    };
    // Per family: chunks filled, and those in more than one piece.
    let mut filled = [[0usize; 2]; 5];
    // MEMW_A's limbs' chunks, and those in more than one piece.
    let mut limbs = [0usize; 2];
    for order in [MemwAlignedOrder::Lt, MemwAlignedOrder::Limbs] {
        for name in [
            "all_instructions_64",
            "test_keccak_multi",
            "all_loadstore_32",
            "test_blake3",
        ] {
            let (program, _) = run(name);
            for window in [1, 7, 33] {
                for (table, index, records) in
                    records_chunks_in(&program, &[], &max_rows, window, (0, None), true, order)
                {
                    let Some((family, pieces)) = row_family(&records) else {
                        continue;
                    };
                    check(
                        &format!("{name}, window {window}, {order:?}, {table:?}[{index}]"),
                        &records,
                    );
                    if order == MemwAlignedOrder::Limbs {
                        if table == StreamTable::MemwAligned {
                            limbs[0] += 1;
                            limbs[1] += usize::from(pieces > 1);
                        }
                        continue;
                    }
                    filled[family][0] += 1;
                    filled[family][1] += usize::from(pieces > 1);
                }
            }
        }
    }
    assert!(
        limbs[0] > 0 && limbs[1] > 0,
        "MEMW_A's limbs' chunks: {limbs:?}"
    );
    // No small program makes four MEMW ops (the synthetic MEMW chunk below
    // covers MEMW), and a STORE chunk is always one piece (its hand-out cuts
    // one list).
    let [memw_a, load, store, _memw, shift] = filled;
    assert!(
        [memw_a, load, store, shift].iter().all(|f| f[0] > 0)
            && [memw_a, load, shift].iter().all(|f| f[1] > 0),
        "{ROW_FAMILIES:?}: {filled:?}"
    );
}

/// A MEMW chunk the card fills from its ops as the host does (no small
/// program makes four MEMW ops): sixteen random ops, memory and register
/// accesses at every width, reads and writes, low words about to carry, old
/// timestamps up to 2^64 - 1, in three window parts.
#[test]
fn the_card_fills_a_memw_chunk_from_its_ops_as_the_host_does() {
    use super::card_records::RowRecords;
    use crate::tables::memw::MemwOperation;
    let mut rng = 0x4D45_4D57_u64;
    let mut next = move || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        rng ^ (rng >> 29)
    };
    let ops: Vec<MemwOperation> = (0..16u32)
        .map(|i| {
            let is_register = i % 3 == 1;
            let element = |w: u64| {
                if is_register {
                    w as u32
                } else {
                    (w & 0xFF) as u32
                }
            };
            let mut op = MemwOperation {
                base_address: (next() & !0xFFFF_FFFF) | (0xFFFF_FFF8 + u64::from(i % 8)),
                timestamp: next() >> 1,
                old_timestamp: std::array::from_fn(|_| next() >> 2),
                value: std::array::from_fn(|_| element(next())),
                old: std::array::from_fn(|_| element(next())),
                is_register,
                width: 1 << (i % 4),
                is_read: i % 2 == 0,
            };
            if i == 5 {
                op.old_timestamp[3] = u64::MAX;
            }
            op
        })
        .collect();
    let parts = [0..5, 5..12, 12..16]
        .into_iter()
        .map(|range| (Arc::new(ops[range.clone()].to_vec()), 0..range.len()))
        .collect();
    let Ok(records) = RowRecords::new(Segments { parts }) else {
        panic!("16 ops are a whole table");
    };
    check("MEMW, 16 ops in 3 parts", &Records(Arc::new(records)));
}

/// Among a wide table, a packed one and each other, the filled tables'
/// columns sit where their widened columns would, and the others are as
/// uploaded.
#[test]
fn the_card_fills_cpu_and_memw_r_in_place_among_a_groups_tables() {
    let (program, _) = run("test_keccak_multi");
    let chunks = records_chunks(&program, &[], &MaxRowsConfig::small(), 7, (0, None), false);
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
    let chunks = records_chunks(&program, &[], &MaxRowsConfig::small(), 7, (0, None), false);
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
    let chunks = records_chunks(
        &program,
        &input,
        &options.max_rows,
        window,
        (64, Some(66)),
        false,
    );
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

/// ★ The in-row register format on the real block (`BLOCK_WHIR_ELF`,
/// `BLOCK_WHIR_INPUT`): every full CPU chunk of the run, at the production
/// height and windows, held as its logs and their side records, fills on the
/// card with the host fill's words. One `CARD FILL IN-ROW` line a chunk and
/// a closing `CARD FILL IN-ROW VERDICT`.
#[test]
#[ignore = "card: a box test (BLOCK_WHIR_ELF, BLOCK_WHIR_INPUT)"]
fn the_card_fills_every_in_row_cpu_chunk_of_a_real_block_as_the_host_does() {
    let elf =
        std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF")).expect("the ELF");
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).expect("input");
    let program = Elf::load(&elf).expect("the ELF loads");
    let options = crate::block_whir::BlockOptions::production();
    let window = 1usize << options.window_log2.expect("the block streams");
    let t = std::time::Instant::now();
    let mut chunks = Vec::new();
    for_each_records_chunk(
        &program,
        &input,
        &options.max_rows,
        window,
        (0, None),
        false,
        (MemwAlignedOrder::Limbs, RegisterFormat::InRow),
        &mut |table, index, records| {
            assert_eq!(table, StreamTable::Cpu, "only CPU is held under in-row");
            let c = std::time::Instant::now();
            check(&format!("real in-row CPU[{index}]"), &records);
            println!(
                "CARD FILL IN-ROW CPU[{index}] rows={} cols={} 0 words differ secs={:.2}",
                records.rows(),
                records.cols(),
                c.elapsed().as_secs_f64()
            );
            chunks.push(index);
        },
    );
    assert!(!chunks.is_empty(), "no CPU chunk held");
    assert!(
        chunks.iter().enumerate().all(|(k, &i)| k == i),
        "every CPU chunk, in order: {chunks:?}"
    );
    println!(
        "CARD FILL IN-ROW VERDICT PASS chunks={} secs={:.1}",
        chunks.len(),
        t.elapsed().as_secs_f64()
    );
}

/// The real block's first chunk of each row family held as its rows, at the
/// production height (each in the window parts of several windows), filled on
/// the card as on the host.
#[test]
#[ignore = "card: a box test (BLOCK_WHIR_ELF, BLOCK_WHIR_INPUT)"]
fn the_card_fills_a_real_blocks_row_family_chunks_as_the_host_does() {
    let elf =
        std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF")).expect("the ELF");
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).expect("input");
    let program = Elf::load(&elf).expect("the ELF loads");
    let options = crate::block_whir::BlockOptions::production();
    let window = 1usize << options.window_log2.expect("the block streams");
    let t = std::time::Instant::now();
    // MEMW_A's rows in the production form (the step's limbs).
    let mut builder = WindowedTraceBuilder::new(&program, &input, &options.max_rows)
        .and_then(|b| b.memw_aligned_order(crate::block_whir::BlockFormat::pinned().memw_aligned))
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(always(), false, false))
        .and_then(|b| b.card_fill_streamed_rows(RowFamilies::ALL))
        .expect("the builder");
    let mut executor = Executor::new(&program, input.clone()).expect("the executor starts");
    let (mut walker, mut accumulator) = builder.split();
    // The first chunk of each family, in [`ROW_FAMILIES`]' order.
    let mut first: [Option<(StreamTable, usize, Records)>; 5] = Default::default();
    let mut walked = 0usize;
    let mut ahead: Option<Vec<Log>> = None;
    while let Some(logs) = executor.resume_with_limit(window).expect("the run") {
        if let Some(previous) = ahead.replace(logs.to_vec()) {
            let w = walker.walk_owned(previous).expect("a window");
            for job in accumulator.absorb(w) {
                let (table, index) = (job.table, job.index);
                let at = match table {
                    StreamTable::MemwAligned => 0,
                    StreamTable::Load => 1,
                    StreamTable::Store => 2,
                    StreamTable::Memw => 3,
                    StreamTable::Shift => 4,
                    _ => continue,
                };
                if first[at].is_some() || !job.is_records() {
                    continue;
                }
                let mut chunk = job.generate();
                first[at] = chunk
                    .trace
                    .take_records_main()
                    .map(|records| (table, index, records));
            }
            walked += 1;
        }
        if first.iter().all(Option::is_some) {
            break;
        }
    }
    let missing: Vec<&str> = ROW_FAMILIES
        .iter()
        .zip(&first)
        .filter_map(|(name, chunk)| chunk.is_none().then_some(*name))
        .collect();
    assert!(
        missing.is_empty(),
        "{walked} windows walked: no {missing:?} chunk held as rows"
    );
    for (table, index, records) in first.iter().flatten() {
        let pieces = row_family(records).map_or(0, |(_, pieces)| pieces);
        println!(
            "CARD FILL REAL ROWS capture_secs={:.2} windows={walked} {table:?}[{index}] rows={} pieces={pieces}",
            t.elapsed().as_secs_f64(),
            records.rows()
        );
        let t = std::time::Instant::now();
        check(&format!("real {table:?}[{index}]"), records);
        println!(
            "CARD FILL REAL ROWS {table:?}[{index}] identity 0 words differ secs={:.2}",
            t.elapsed().as_secs_f64()
        );
    }
    println!("CARD FILL REAL ROWS VERDICT PASS");
}

/// Chunks of four rows for the row families and the fused tables, so that the
/// small programs fill several, their rows in one window part or many.
fn fused_max() -> MaxRowsConfig {
    MaxRowsConfig {
        memw_aligned: 1 << 2,
        load: 1 << 2,
        store: 1 << 2,
        memw: 1 << 2,
        shift: 1 << 2,
        mem_load: 1 << 2,
        mem_store: 1 << 2,
        ..MaxRowsConfig::small()
    }
}

/// A builder under the block's formats with the fused memory format (in-row
/// registers, MEMW_A in the step's limbs), its streamed ops dropped; with
/// `held` (the test fault), its in-row CPU chunks held as their logs and every
/// row family's as its rows.
fn fused_builder<'a>(
    program: &Elf,
    input: &'a [u8],
    max_rows: &MaxRowsConfig,
    held: Option<u64>,
) -> WindowedTraceBuilder<'a> {
    use crate::tables::fused_memory::MemoryFormat;
    let builder = WindowedTraceBuilder::new(program, input, max_rows)
        .and_then(|b| b.register_format(RegisterFormat::InRow))
        .and_then(|b| b.memw_aligned_order(MemwAlignedOrder::Limbs))
        .and_then(|b| b.memory_format(MemoryFormat::Fused))
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .expect("the builder");
    match held {
        None => builder,
        Some(fault) => builder
            .card_fill_streamed(always(), true, false)
            .and_then(|b| b.card_fill_streamed_rows(RowFamilies::ALL))
            .expect("the card fill")
            .card_fill_streamed_fault(fault),
    }
}

/// Every chunk the fused build of `program` streams by windows of `window`
/// cycles (the finish's tables aside), in hand-out order: `(table, index,
/// chunk)`, built as [`fused_builder`] says.
fn fused_chunks(
    program: &Elf,
    max_rows: &MaxRowsConfig,
    window: usize,
    held: Option<u64>,
) -> Vec<(StreamTable, usize, super::StreamedChunk)> {
    let mut builder = fused_builder(program, &[], max_rows, held);
    let mut executor = Executor::new(program, Vec::new()).expect("the executor starts");
    let (mut walker, mut accumulator) = builder.split();
    let mut out = Vec::new();
    let mut ahead: Option<Vec<Log>> = None;
    while let Some(logs) = executor.resume_with_limit(window).expect("the run") {
        // The run's last window is the finish's.
        if let Some(previous) = ahead.replace(logs.to_vec()) {
            let w = walker.walk_owned(previous).expect("a window");
            for job in accumulator.absorb(w) {
                out.push((job.table, job.index, job.generate()));
            }
        }
    }
    out
}

/// ★ The fused memory format's whole streamed build on the card: every chunk
/// the build streams with every family held (the in-row CPU chunks as their
/// logs, MEMW_A, LOAD, STORE, MEMW, SHIFT, MEM_LOAD and MEM_STORE as their
/// rows) is, word for word, the chunk the same build makes without the card:
/// each records chunk as the card fills it from its rows (counted as a card
/// fill, and equal to its host fill), each other as built. The mutant: every
/// held row moved 4 cycles later (the test fault) makes the card fill MEM_LOAD
/// and MEM_STORE chunks other than the build's.
#[test]
fn the_card_fills_a_fused_build_as_the_host_builds_it() {
    use crate::tables::card_fill_inputs::col_major;
    use std::collections::HashMap;
    let max_rows = fused_max();
    // MEM_LOAD's and MEM_STORE's chunks filled on the card; and those in more
    // than one window part.
    let mut filled = [[0usize; 2]; 2];
    let fused_at = |table: StreamTable| match table {
        StreamTable::MemLoad => Some(0),
        StreamTable::MemStore => Some(1),
        _ => None,
    };
    for name in [
        "fused_mem_cases",
        "all_loadstore_32",
        "all_instructions_64",
        "test_blake3",
    ] {
        let (program, _) = run(name);
        for window in [1, 7, 33] {
            let plain: HashMap<(StreamTable, usize), Vec<u64>> =
                fused_chunks(&program, &max_rows, window, None)
                    .into_iter()
                    .map(|(table, index, chunk)| ((table, index), col_major(&chunk.trace)))
                    .collect();
            let held = fused_chunks(&program, &max_rows, window, Some(0));
            assert_eq!(held.len(), plain.len(), "{name}, window {window}: chunks");
            for (table, index, mut chunk) in held {
                let what = format!("{name}, window {window}, {table:?}[{index}]");
                let want = plain.get(&(table, index)).expect("the same chunk");
                match chunk.trace.take_records_main() {
                    Some(records) => {
                        check(&what, &records);
                        let card = uploaded(&[TableColumns::Records(&records)]).concat();
                        assert_eq!(
                            card.iter().zip(want).filter(|(a, b)| a != b).count(),
                            0,
                            "{what}: the card's fill against the build's chunk"
                        );
                        assert_eq!(card.len(), want.len(), "{what}: words");
                        if let Some(f) = fused_at(table) {
                            filled[f][0] += 1;
                            let pieces = row_family_pieces(&records);
                            filled[f][1] += usize::from(pieces > 1);
                        }
                    }
                    None => {
                        assert_eq!(&col_major(&chunk.trace), want, "{what}: as built");
                        assert!(fused_at(table).is_none(), "{what}: a fused chunk not held");
                    }
                }
            }
        }
    }
    assert!(
        filled.iter().all(|f| f[0] > 1 && f[1] > 0),
        "MEM_LOAD, MEM_STORE: {filled:?}"
    );
    // The mutant.
    let (program, _) = run("fused_mem_cases");
    let plain: HashMap<(StreamTable, usize), Vec<u64>> = fused_chunks(&program, &max_rows, 7, None)
        .into_iter()
        .map(|(table, index, chunk)| ((table, index), col_major(&chunk.trace)))
        .collect();
    let mut moved = [0usize; 2];
    for (table, index, mut chunk) in fused_chunks(&program, &max_rows, 7, Some(4)) {
        let Some(f) = fused_at(table) else {
            continue;
        };
        let records = chunk.trace.take_records_main().expect("held as its rows");
        let card = uploaded(&[TableColumns::Records(&records)]).concat();
        moved[f] += usize::from(&card != plain.get(&(table, index)).expect("the same chunk"));
    }
    assert!(
        moved.iter().all(|&n| n > 0),
        "the fault moved no fused chunk's words: {moved:?}"
    );
}

/// A fused family's records' pieces (window parts); 0 for another family.
fn row_family_pieces(records: &Records) -> usize {
    use multilinear::records::RecordsFill as R;
    match records.fill() {
        R::MemLoad { rows } => rows.len(),
        R::MemStore { rows } => rows.len(),
        _ => 0,
    }
}

/// ★ A real block under the fused memory format (`BLOCK_WHIR_ELF`,
/// `BLOCK_WHIR_INPUT`, the production heights): its first two MEM_LOAD and
/// MEM_STORE chunks, held as their rows (each in the window parts of several
/// windows), filled on the card with the words of the same chunks the build
/// makes without the card, and of their host fills.
#[test]
#[ignore = "card: a box test (BLOCK_WHIR_ELF, BLOCK_WHIR_INPUT)"]
fn the_card_fills_a_real_fused_blocks_chunks_as_the_host_builds_them() {
    use crate::tables::card_fill_inputs::col_major;
    let elf =
        std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF")).expect("the ELF");
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).expect("input");
    let program = Elf::load(&elf).expect("the ELF loads");
    let options = crate::block_whir::BlockOptions::production();
    let window = 1usize << options.window_log2.expect("the block streams");
    let t = std::time::Instant::now();
    let mut held = fused_builder(&program, &input, &options.max_rows, Some(0));
    let mut plain = fused_builder(&program, &input, &options.max_rows, None);
    let mut executor = Executor::new(&program, input.clone()).expect("the executor starts");
    let (mut held_walker, mut held_acc) = held.split();
    let (mut plain_walker, mut plain_acc) = plain.split();
    // Per family: the chunks compared.
    let mut compared = [0usize; 2];
    let mut walked = 0usize;
    let mut ahead: Option<Vec<Log>> = None;
    while let Some(logs) = executor.resume_with_limit(window).expect("the run") {
        if let Some(previous) = ahead.replace(logs.to_vec()) {
            let logs = Arc::new(previous);
            let built: Vec<_> = plain_acc
                .absorb(
                    plain_walker
                        .walk_owned(Arc::clone(&logs))
                        .expect("a window"),
                )
                .into_iter()
                .filter(|job| matches!(job.table, StreamTable::MemLoad | StreamTable::MemStore))
                .map(|job| (job.table, job.index, job.generate()))
                .collect();
            for job in held_acc.absorb(held_walker.walk_owned(logs).expect("a window")) {
                let f = match job.table {
                    StreamTable::MemLoad => 0,
                    StreamTable::MemStore => 1,
                    _ => continue,
                };
                let (table, index) = (job.table, job.index);
                if compared[f] >= 2 {
                    continue;
                }
                let what = format!("real fused {table:?}[{index}]");
                let mut chunk = job.generate();
                let records = chunk.trace.take_records_main().expect("held as its rows");
                let (_, _, want) = built
                    .iter()
                    .find(|(t, i, _)| (*t, *i) == (table, index))
                    .expect("the build without the card cut the same chunk");
                let c = std::time::Instant::now();
                check(&what, &records);
                let card = uploaded(&[TableColumns::Records(&records)]).concat();
                let want = col_major(&want.trace);
                let differ = card.iter().zip(&want).filter(|(a, b)| a != b).count();
                assert_eq!((differ, card.len()), (0, want.len()), "{what}");
                println!(
                    "CARD FILL REAL FUSED {table:?}[{index}] rows={} pieces={} windows={walked} \
                     0 words differ secs={:.2}",
                    records.rows(),
                    row_family_pieces(&records),
                    c.elapsed().as_secs_f64()
                );
                compared[f] += 1;
            }
            walked += 1;
        }
        if compared.iter().all(|&n| n >= 2) {
            break;
        }
    }
    assert!(
        compared.iter().all(|&n| n > 0),
        "{walked} windows walked: MEM_LOAD, MEM_STORE chunks compared {compared:?}"
    );
    println!(
        "CARD FILL REAL FUSED VERDICT PASS mem_load={} mem_store={} windows={walked} secs={:.1}",
        compared[0],
        compared[1],
        t.elapsed().as_secs_f64()
    );
}

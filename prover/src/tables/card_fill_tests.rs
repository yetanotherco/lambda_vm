//! ★ Box only (`--ignored`, a card): the card's table fills from records
//! (`math_cuda::fill`) against today's path, the host's packed fill widened on
//! the card, for MEMW_R, CPU and KECCAK_RND.
//!
//! Each case runs both paths on the same rows:
//! - **today:** the host fill in [`TraceForm::Narrow`] on this thread (timed),
//!   its packed bytes uploaded and widened (`fill::widen_col_major_timed`);
//! - **the card:** the case's records uploaded and filled by the family's
//!   kernel.
//!
//! Both slabs are downloaded and compared word for word, against each other
//! and against the host fill's [`TraceForm::Wide`] words. A mutant (one record
//! changed) must then differ from the host, which shows the comparison sees
//! the records. Timed cases repeat both paths 3 + 20 times and print the
//! median, min and max of the host fill, the uploads and the kernels. Each
//! case prints one `CARD FILL <family> <input> key=value …` line (KECCAK_RND's
//! with `minimal_bytes`, its records without the output the kernel never
//! reads).
//!
//! The inputs ([`super::card_fill_inputs`]):
//! - three seeded sets a family, CPU's over the decode table of the guest at
//!   `BLOCK_WHIR_ELF`;
//! - real chunks of the block at `BLOCK_WHIR_ELF` / `BLOCK_WHIR_INPUT`. The run
//!   is executed and walked with one walker (the split builder, as the
//!   streamed prove does) through window 65. The CPU chunk is chunk 32: the
//!   logs of windows 64 and 65 against the walk's ops. The MEMW_R chunk is the
//!   first full register chunk handed out after window 64;
//! - one padded table a family, identity only.
//!
//! KECCAK_RND runs first, then prints its own `CARD FILL VERDICT KECCAK_RND`,
//! so its rows stand even if the real block's capture fails. Then come CPU
//! and MEMW_R, their real chunks before their synthetic sets, so the host
//! fill's widths are learned from real rows as in a prove. A CPU fill with
//! one log at a pc no DECODE row holds must be refused (`CARD FILL REFUSAL`).
//! The loaded cases repeat the first KECCAK_RND set's and the real chunks'
//! uploads beside four threads copying 1 GiB buffers (a walker's memory
//! traffic), as `CARD FILL LOADED` lines. `CARD FILL VERDICT` closes the run.
//!
//! ```text
//! BLOCK_WHIR_ELF=…/ethrex_3e8f1ce1.elf BLOCK_WHIR_INPUT=…/block.bin \
//!   cargo test --release -p lambda-vm-prover --lib --features cuda \
//!   tables::card_fill_tests::the_card_fills_the_streamed_tables_as_the_host_does \
//!   -- --exact --ignored --nocapture
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;
use math_cuda::fill::{self, CardDecode, FillTimes, KeccakOpC, LogC, RegRowC};
use math_cuda::narrow::NarrowInput;
use math_cuda::{CudaSlice, CudaStream};

use super::card_fill_inputs::{self as inputs, CHUNK_ROWS, KECCAK_RND_ROWS, KECCAK_RND_SKIP};
use super::cpu::generate_cpu_trace_segments_as;
use super::gpack::{Trace, TraceForm};
use super::keccak_rnd::KeccakRoundOperation;
use super::memw_register::generate_memw_register_trace_from_rows_segments_as;
use super::trace_builder::{ChunkJob, StreamTable, WindowedTraceBuilder};

const WARMUPS: usize = 3;
const REPS: usize = 20;
/// The real CPU chunk, and the window its MEMW_R chunk is handed out at or
/// after: chunk 32 is windows 64 and 65.
const REAL_CPU_CHUNK: usize = 32;
const REAL_FIRST_WINDOW: usize = 64;
/// Rows a padded case leaves off a full table.
const PADDED_SHORT: usize = 12_345;
/// The padded KECCAK_RND case's ops: its last rows are past them.
const PADDED_KECCAK_OPS: usize = 2700;
/// The loaded cases' copy threads and their buffers.
const LOAD_THREADS: usize = 4;
const LOAD_BYTES: usize = 1 << 30;

/// What a case's card path fills from.
enum Records<'a> {
    MemwR(Vec<RegRowC>),
    Cpu {
        logs: Vec<LogC>,
        first_cycle: u64,
        decode: &'a CardDecode,
    },
    KeccakRnd {
        ops: Vec<KeccakOpC>,
        skip: usize,
    },
}

impl Records<'_> {
    /// The card's fill of the case's `rows` rows into `out`.
    fn fill(&self, stream: &Arc<CudaStream>, rows: usize, out: &mut CudaSlice<u64>) -> FillTimes {
        let dst = &mut out.slice_mut(..);
        match self {
            Records::MemwR(records) => fill::fill_memw_r_into(stream, records, rows, dst),
            Records::Cpu {
                logs,
                first_cycle,
                decode,
            } => fill::fill_cpu_from_logs_into(stream, logs, *first_cycle, decode, rows, dst),
            Records::KeccakRnd { ops, skip } => {
                fill::fill_keccak_rnd_into(stream, ops, *skip, rows, dst)
            }
        }
        .expect("the card's fill")
    }

    /// The bytes the records would take without what the kernel never reads
    /// (a keccak op's output): KECCAK_RND's only.
    fn minimal_bytes(&self) -> Option<usize> {
        match self {
            Records::KeccakRnd { ops, .. } => {
                Some(ops.len() * std::mem::offset_of!(KeccakOpC, output))
            }
            _ => None,
        }
    }

    /// The records with one changed where it moves a word: a register row's
    /// value, a log's pc (to another instruction's), a keccak op's input lane.
    fn mutant(&self) -> Self {
        match self {
            Records::MemwR(records) => {
                let mut records = records.clone();
                let at = records.len() / 2;
                records[at].val0 ^= 1;
                Records::MemwR(records)
            }
            Records::Cpu {
                logs,
                first_cycle,
                decode,
            } => {
                let mut logs = logs.clone();
                let at = (logs.len() / 2..logs.len() - 1)
                    .find(|&i| logs[i].current_pc != logs[i + 1].current_pc)
                    .expect("two logs at different pcs");
                logs[at].current_pc = logs[at + 1].current_pc;
                Records::Cpu {
                    logs,
                    first_cycle: *first_cycle,
                    decode,
                }
            }
            Records::KeccakRnd { ops, skip } => {
                let mut ops = ops.clone();
                let at = ops.len() / 2;
                ops[at].input[3] ^= 1 << 17;
                Records::KeccakRnd { ops, skip: *skip }
            }
        }
    }
}

/// One family's table on one input.
struct Case<'a> {
    family: &'static str,
    input: String,
    rows: usize,
    cols: usize,
    /// The host fill, in a form.
    host: Box<dyn Fn(TraceForm) -> Trace + 'a>,
    records: Records<'a>,
    /// Timed, or identity only.
    timed: bool,
}

/// Milliseconds since `t`.
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// `key=median key_min=… key_max=…` of `values`.
fn stat(key: &str, values: &[f64]) -> String {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let median = if v.is_empty() {
        f64::NAN
    } else if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    };
    format!(
        "{key}={median:.4} {key}_min={:.4} {key}_max={:.4}",
        v.first().copied().unwrap_or(f64::NAN),
        v.last().copied().unwrap_or(f64::NAN)
    )
}

/// One path's repetitions.
#[derive(Default)]
struct Reps {
    upload: Vec<f64>,
    kernel: Vec<f64>,
    wall: Vec<f64>,
    bytes: usize,
}

impl Reps {
    fn push(&mut self, start: Instant, times: FillTimes) {
        self.wall.push(ms(start));
        self.upload.push(times.upload_ms);
        self.kernel.push(times.kernel_ms);
        self.bytes = times.upload_bytes;
    }

    fn line(&self, path: &str) -> String {
        format!(
            "{} {} {} {path}_bytes={}",
            stat(&format!("{path}_upload_ms"), &self.upload),
            stat(&format!("{path}_kernel_ms"), &self.kernel),
            stat(&format!("{path}_wall_ms"), &self.wall),
            self.bytes
        )
    }
}

/// Four threads copying 1 GiB buffers until stopped: the host's memory
/// traffic beside the walkers.
struct HostLoad {
    stop: Arc<AtomicBool>,
    copied: Arc<AtomicU64>,
    threads: Vec<std::thread::JoinHandle<()>>,
    started: Instant,
    /// Bytes copied by then.
    base: u64,
}

impl HostLoad {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let copied = Arc::new(AtomicU64::new(0));
        let threads = (0..LOAD_THREADS)
            .map(|_| {
                let (stop, copied) = (stop.clone(), copied.clone());
                std::thread::spawn(move || {
                    let src = vec![0xA5u8; LOAD_BYTES];
                    let mut dst = vec![0u8; LOAD_BYTES];
                    while !stop.load(Relaxed) {
                        dst.copy_from_slice(&src);
                        std::hint::black_box(&mut dst);
                        copied.fetch_add(LOAD_BYTES as u64, Relaxed);
                    }
                })
            })
            .collect();
        // Every thread's buffers touched and copying.
        while copied.load(Relaxed) < (LOAD_THREADS * LOAD_BYTES) as u64 {
            std::thread::yield_now();
        }
        let base = copied.load(Relaxed);
        Self {
            stop,
            copied,
            threads,
            started: Instant::now(),
            base,
        }
    }

    /// Stops the threads; the GiB/s they copied since started.
    fn stop(self) -> f64 {
        let secs = self.started.elapsed().as_secs_f64();
        let copied = self.copied.load(Relaxed) - self.base;
        self.stop.store(true, Relaxed);
        for t in self.threads {
            t.join().expect("a copy thread");
        }
        copied as f64 / secs / (1u64 << 30) as f64
    }
}

/// What a case found: its words that differ (today's against the host, the
/// card's against the host, the card's against today's) and its mutant's.
struct Found {
    mismatches: usize,
    mutant: usize,
}

/// A card buffer of `cells` words, each `poison` (so a cell no kernel writes
/// is a mismatch).
fn poisoned(stream: &Arc<CudaStream>, poison: &[u64], cells: usize) -> CudaSlice<u64> {
    let mut out = stream.alloc_zeros::<u64>(cells).expect("card memory");
    repoison(stream, poison, &mut out);
    out
}

fn repoison(stream: &Arc<CudaStream>, poison: &[u64], out: &mut CudaSlice<u64>) {
    let cells = out.len();
    stream
        .memcpy_htod(&poison[..cells], out)
        .expect("poison the slab");
}

fn download(stream: &Arc<CudaStream>, out: &CudaSlice<u64>) -> Vec<u64> {
    let words = stream.clone_dtoh(out).expect("download");
    stream.synchronize().expect("the stream");
    words
}

fn first(m: &inputs::Mismatches) -> String {
    match m.first {
        Some((col, row, want, got)) => format!("{col}:{row}:{want}:{got}"),
        None => "none".into(),
    }
}

/// Runs `case` (the module docs); with `load`, only its timed repetitions,
/// beside the copy threads.
fn run_case(case: &Case, stream: &Arc<CudaStream>, poison: &[u64], load: bool) -> Found {
    let cells = case.rows * case.cols;
    assert!(poison.len() >= cells, "the poison covers the table");

    // Today: the host fill, packed.
    let mut host_ms = Vec::new();
    let mut packed = None;
    let fills = if case.timed && !load {
        WARMUPS + REPS
    } else {
        1
    };
    for rep in 0..fills {
        let t = Instant::now();
        let trace = (case.host)(TraceForm::Narrow);
        if rep >= fills.saturating_sub(REPS) && case.timed {
            host_ms.push(ms(t));
        }
        packed = Some(trace);
    }
    let packed = packed.expect("a host fill");
    let columns = packed
        .narrow_main()
        .expect("the host fill packs (a build without debug-checks)");
    assert_eq!(
        (columns.rows(), columns.cols()),
        (case.rows, case.cols),
        "{} {}: the host table's shape",
        case.family,
        case.input
    );
    let narrow = NarrowInput::new(
        columns.data(),
        columns.offsets(),
        columns.widths(),
        columns.rows(),
        columns.cols(),
    );
    let mut today = poisoned(stream, poison, cells);
    let mut card = poisoned(stream, poison, cells);

    if load {
        let host_load = HostLoad::start();
        let (mut a, mut b) = (Reps::default(), Reps::default());
        for rep in 0..WARMUPS + REPS {
            let t = Instant::now();
            let times = fill::widen_col_major_timed(stream, narrow, &mut today.slice_mut(..))
                .expect("the widen");
            if rep >= WARMUPS {
                a.push(t, times);
            }
            let t = Instant::now();
            let times = case.records.fill(stream, case.rows, &mut card);
            if rep >= WARMUPS {
                b.push(t, times);
            }
        }
        let gib_s = host_load.stop();
        println!(
            "CARD FILL LOADED {} {} threads={LOAD_THREADS} copy_gib_s={gib_s:.2} {} {}",
            case.family,
            case.input,
            a.line("widen"),
            b.line("fill")
        );
        return Found {
            mismatches: 0,
            mutant: 1,
        };
    }

    // Identity: both slabs against the host's words and each other.
    let want = inputs::col_major(&(case.host)(TraceForm::Wide));
    assert_eq!(want.len(), cells);
    let widen =
        fill::widen_col_major_timed(stream, narrow, &mut today.slice_mut(..)).expect("the widen");
    let filled = case.records.fill(stream, case.rows, &mut card);
    let (a, b) = (download(stream, &today), download(stream, &card));
    let a_host = inputs::mismatches(&want, &a, case.rows);
    let b_host = inputs::mismatches(&want, &b, case.rows);
    let b_a = inputs::mismatches(&a, &b, case.rows);
    drop((a, b));

    // The mutant: one record changed must show.
    repoison(stream, poison, &mut card);
    case.records.mutant().fill(stream, case.rows, &mut card);
    let mutant = inputs::mismatches(&want, &download(stream, &card), case.rows);
    drop(want);

    let mut line = format!(
        "CARD FILL {} {} rows={} cols={} packed_bytes={} record_bytes={} \
         mismatches_widen_host={} mismatches_fill_host={} mismatches_fill_widen={} \
         first_fill_host={} mutant_mismatches={} timed={}",
        case.family,
        case.input,
        case.rows,
        case.cols,
        widen.upload_bytes,
        filled.upload_bytes,
        a_host.count,
        b_host.count,
        b_a.count,
        first(&b_host),
        mutant.count,
        case.timed,
    );
    if let Some(bytes) = case.records.minimal_bytes() {
        line += &format!(" minimal_bytes={bytes}");
    }
    if case.timed {
        let (mut a, mut b) = (Reps::default(), Reps::default());
        for rep in 0..WARMUPS + REPS {
            let t = Instant::now();
            let times = fill::widen_col_major_timed(stream, narrow, &mut today.slice_mut(..))
                .expect("the widen");
            if rep >= WARMUPS {
                a.push(t, times);
            }
            let t = Instant::now();
            let times = case.records.fill(stream, case.rows, &mut card);
            if rep >= WARMUPS {
                b.push(t, times);
            }
        }
        line += &format!(
            " {} {} {}",
            stat("host_fill_ms", &host_ms),
            a.line("widen"),
            b.line("fill")
        );
    }
    println!("{line}");
    Found {
        mismatches: a_host.count + b_host.count + b_a.count,
        mutant: mutant.count,
    }
}

/// The real block's chunks (the module docs).
struct RealChunks {
    cpu: ChunkJob,
    /// Windows 64 and 65's logs.
    logs: Vec<Log>,
    first_cycle: u64,
    memw_r: ChunkJob,
}

fn real_chunks(elf: &[u8]) -> RealChunks {
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).unwrap();
    let program = Elf::load(elf).expect("the ELF loads");
    let options = crate::block_whir::BlockOptions::production();
    let window = 1usize << options.window_log2.expect("the block streams");
    assert_eq!(2 * window, CHUNK_ROWS, "a CPU chunk is two windows");
    let t = Instant::now();
    let mut builder = WindowedTraceBuilder::new(&program, &input, &options.max_rows)
        .expect("the builder")
        .drop_streamed_ops()
        .expect("before any window");
    let mut executor = Executor::new(&program, input.clone()).expect("the executor starts");
    let (mut walker, mut accumulator) = builder.split();
    let (mut cpu, mut memw_r) = (None, None);
    let mut logs = Vec::with_capacity(2 * window);
    for w in 0..REAL_FIRST_WINDOW + 2 {
        let window_logs = executor
            .resume_with_limit(window)
            .expect("the block runs")
            .expect("the block runs past window 65");
        assert_eq!(window_logs.len(), window, "window {w} is full");
        if w >= REAL_FIRST_WINDOW {
            logs.extend_from_slice(window_logs);
        }
        let walked = walker.walk(window_logs).expect("the walk");
        for job in accumulator.absorb(walked) {
            match job.table {
                StreamTable::Cpu if job.index == REAL_CPU_CHUNK => cpu = Some(job),
                StreamTable::MemwRegister if w >= REAL_FIRST_WINDOW && memw_r.is_none() => {
                    memw_r = Some(job)
                }
                _ => {}
            }
        }
    }
    let (cpu, memw_r) = (
        cpu.expect("CPU chunk 32 handed out with window 65"),
        memw_r.expect("a MEMW_R chunk handed out from window 64"),
    );
    println!(
        "CARD FILL CAPTURE windows={} secs={:.2} memw_r_chunk={} memw_r_rows={}",
        REAL_FIRST_WINDOW + 2,
        t.elapsed().as_secs_f64(),
        memw_r.index,
        memw_r.op_count()
    );
    RealChunks {
        cpu,
        logs,
        first_cycle: (REAL_FIRST_WINDOW * window) as u64,
        memw_r,
    }
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// KECCAK_RND's table of `ops` (the synthetic sets' rows and cut).
fn keccak_case(input: String, ops: &[KeccakRoundOperation], timed: bool) -> Case<'_> {
    Case {
        family: "KECCAK_RND",
        input,
        rows: KECCAK_RND_ROWS,
        cols: fill::KECCAK_RND_COLUMNS,
        host: Box::new(move |form| {
            inputs::keccak_rnd_host(ops, KECCAK_RND_SKIP, KECCAK_RND_ROWS, form)
        }),
        records: Records::KeccakRnd {
            ops: ops.iter().map(inputs::keccak_record).collect(),
            skip: KECCAK_RND_SKIP,
        },
        timed,
    }
}

/// What the cases in `found` add up to: the words that differ, and the cases
/// whose mutant showed nothing.
fn tally(found: &[(String, Found)]) -> (usize, Vec<&String>) {
    let mismatches = found.iter().map(|(_, f)| f.mismatches).sum();
    let blind = found
        .iter()
        .filter(|(_, f)| f.mutant == 0)
        .map(|(name, _)| name)
        .collect();
    (mismatches, blind)
}

/// Every family's card fill is the host fill's words, on synthetic and real
/// rows, and both paths are timed (the module docs).
#[test]
#[ignore = "card: a box test (BLOCK_WHIR_ELF, BLOCK_WHIR_INPUT)"]
fn the_card_fills_the_streamed_tables_as_the_host_does() {
    let elf_path = std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF names the guest");
    let elf = std::fs::read(&elf_path).unwrap_or_else(|e| panic!("{elf_path}: {e}"));
    // The synthetic CPU sets are drawn over this guest's decode table.
    println!(
        "CARD FILL ENV elf_sha256={} decode_source=BLOCK_WHIR_ELF reps={REPS} warmups={WARMUPS}",
        sha256(&elf)
    );
    let be = math_cuda::device::backend().expect("a card");
    let stream = be.next_stream();
    let poison = vec![0x5A5A_5A5A_5A5A_5A5Au64; KECCAK_RND_ROWS * fill::KECCAK_RND_COLUMNS];
    // Each case's findings (a loaded case finds nothing).
    let found = std::cell::RefCell::new(Vec::new());
    let run = |case: &Case, load: bool| {
        let f = run_case(case, &stream, &poison, load);
        if !load {
            found
                .borrow_mut()
                .push((format!("{} {}", case.family, case.input), f));
        }
    };
    let seed = inputs::SEEDS[0];

    // KECCAK_RND first: its rows stand even if the real block's capture fails.
    let sets: Vec<_> = inputs::SEEDS
        .iter()
        .map(|&s| inputs::keccak_ops(s, inputs::KECCAK_RND_OPS))
        .collect();
    for (s, ops) in inputs::SEEDS.iter().zip(&sets) {
        run(&keccak_case(format!("seed{s}"), ops, true), false);
    }
    let short = inputs::keccak_ops(seed, PADDED_KECCAK_OPS);
    run(&keccak_case("padded".into(), &short, false), false);
    run(&keccak_case(format!("seed{seed}"), &sets[0], true), true);
    drop((sets, short));
    {
        let found = found.borrow();
        let (mismatches, blind) = tally(&found);
        println!(
            "CARD FILL VERDICT KECCAK_RND: identity={} mismatches={mismatches} mutants={} cases={}",
            if mismatches == 0 { "ok" } else { "FAIL" },
            if blind.is_empty() { "ok" } else { "FAIL" },
            found.len()
        );
    }

    // The real chunks, before the synthetic sets, so the host fill's widths
    // are learned from real rows as in a prove.
    let real = real_chunks(&elf);
    let (cpu_parts, real_decode) = real.cpu.cpu_parts().expect("a CPU chunk");
    let real_ops: usize = cpu_parts.iter().map(|s| s.len()).sum();
    assert_eq!(real_ops, CHUNK_ROWS);
    assert_eq!(real.logs.len(), CHUNK_ROWS);
    assert_eq!(
        cpu_parts[0][0].timestamp,
        4 * real.first_cycle + 4,
        "chunk 32 starts at window 64"
    );
    let (entries, runs) = inputs::decode_records(real_decode);
    let real_card_decode = CardDecode::upload(&stream, &entries, &runs).expect("the decode");
    let real_cpu = Case {
        family: "CPU",
        input: "real".into(),
        rows: CHUNK_ROWS,
        cols: fill::CPU_COLUMNS,
        host: Box::new(|form| generate_cpu_trace_segments_as(&cpu_parts, real_decode, form)),
        records: Records::Cpu {
            logs: real.logs.iter().map(inputs::log_record).collect(),
            first_cycle: real.first_cycle,
            decode: &real_card_decode,
        },
        timed: true,
    };
    run(&real_cpu, false);
    let reg_parts = real.memw_r.register_rows().expect("a MEMW_R chunk");
    assert_eq!(
        reg_parts.iter().map(|s| s.len()).sum::<usize>(),
        CHUNK_ROWS,
        "a full register chunk"
    );
    let real_memw_r = Case {
        family: "MEMW_R",
        input: "real".into(),
        rows: CHUNK_ROWS,
        cols: fill::MEMW_R_COLUMNS,
        host: Box::new(|form| generate_memw_register_trace_from_rows_segments_as(&reg_parts, form)),
        records: Records::MemwR(
            reg_parts
                .iter()
                .flat_map(|s| s.iter())
                .map(|r| r.card_record())
                .collect(),
        ),
        timed: true,
    };
    run(&real_memw_r, false);

    // The synthetic sets, over the guest's decode table.
    let decode = inputs::program_decode();
    let (entries, runs) = inputs::decode_records(&decode);
    let card_decode = CardDecode::upload(&stream, &entries, &runs).expect("the decode");
    for s in inputs::SEEDS {
        let rows = inputs::memw_r_rows(s, CHUNK_ROWS);
        run(
            &Case {
                family: "MEMW_R",
                input: format!("seed{s}"),
                rows: CHUNK_ROWS,
                cols: fill::MEMW_R_COLUMNS,
                host: Box::new(|form| inputs::memw_r_host(&rows, form)),
                records: Records::MemwR(rows.iter().map(|r| r.card_record()).collect()),
                timed: true,
            },
            false,
        );
        let logs = inputs::cpu_logs(s, CHUNK_ROWS, &decode);
        let ops = inputs::cpu_ops(&logs, inputs::CPU_FIRST_CYCLE, &decode);
        run(
            &Case {
                family: "CPU",
                input: format!("seed{s}"),
                rows: CHUNK_ROWS,
                cols: fill::CPU_COLUMNS,
                host: Box::new(|form| inputs::cpu_host(&ops, &decode, form)),
                records: Records::Cpu {
                    logs: logs.iter().map(inputs::log_record).collect(),
                    first_cycle: inputs::CPU_FIRST_CYCLE,
                    decode: &card_decode,
                },
                timed: true,
            },
            false,
        );
    }

    // A log whose pc no DECODE row holds: the fill is refused, as the host
    // fill's `MissingInstruction`.
    // The first instruction's pc + 2, unless the host finds a row there (a
    // 2-byte instruction); then the pc past the last run's end.
    let (last_pc, _, last_len) = *decode.runs().last().expect("a program");
    let absent = [
        decode.entries()[0].pc + 2,
        last_pc + 4 * u64::from(last_len),
    ]
    .into_iter()
    .find(|&pc| decode.row(pc).is_none())
    .expect("a pc no DECODE row holds");
    assert_eq!(decode.row(absent), None);
    let refusal_rows = 1 << 12;
    let mut logs: Vec<LogC> = inputs::cpu_logs(seed, refusal_rows, &decode)
        .iter()
        .map(inputs::log_record)
        .collect();
    logs[1234].current_pc = absent;
    let mut out = poisoned(&stream, &poison, refusal_rows * fill::CPU_COLUMNS);
    let refused = fill::fill_cpu_from_logs_into(
        &stream,
        &logs,
        inputs::CPU_FIRST_CYCLE,
        &card_decode,
        refusal_rows,
        &mut out.slice_mut(..),
    )
    .is_err();
    println!("CARD FILL REFUSAL CPU absent_pc={absent} refused={refused}");

    // Padded tables, identity only.
    let short = CHUNK_ROWS - PADDED_SHORT;
    let rows = inputs::memw_r_rows(seed, short);
    run(
        &Case {
            family: "MEMW_R",
            input: "padded".into(),
            rows: CHUNK_ROWS,
            cols: fill::MEMW_R_COLUMNS,
            host: Box::new(|form| inputs::memw_r_host(&rows, form)),
            records: Records::MemwR(rows.iter().map(|r| r.card_record()).collect()),
            timed: false,
        },
        false,
    );
    let logs = inputs::cpu_logs(seed, short, &decode);
    let ops = inputs::cpu_ops(&logs, inputs::CPU_FIRST_CYCLE, &decode);
    run(
        &Case {
            family: "CPU",
            input: "padded".into(),
            rows: CHUNK_ROWS,
            cols: fill::CPU_COLUMNS,
            host: Box::new(|form| inputs::cpu_host(&ops, &decode, form)),
            records: Records::Cpu {
                logs: logs.iter().map(inputs::log_record).collect(),
                first_cycle: inputs::CPU_FIRST_CYCLE,
                decode: &card_decode,
            },
            timed: false,
        },
        false,
    );

    // The real chunks' uploads beside the host's memory traffic.
    run(&real_cpu, true);
    run(&real_memw_r, true);

    let found = found.into_inner();
    let (mismatches, blind) = tally(&found);
    println!(
        "CARD FILL VERDICT: identity={} mismatches={mismatches} mutants={} refusal={} cases={}",
        if mismatches == 0 { "ok" } else { "FAIL" },
        if blind.is_empty() { "ok" } else { "FAIL" },
        if refused { "ok" } else { "FAIL" },
        found.len()
    );
    assert_eq!(mismatches, 0, "a card fill differs from the host's");
    assert!(blind.is_empty(), "mutants the comparison missed: {blind:?}");
    assert!(refused, "a log with no DECODE row was filled");
}

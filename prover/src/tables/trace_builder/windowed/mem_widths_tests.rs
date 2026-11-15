//! A probe, test only: the memory accesses of a real block counted by width
//! and alignment (`BLOCK MEM WIDTHS:` lines). A walk of the block in record
//! mode at the production window, each window counted and then let go:
//! - every CPU load and store, from its cycle's log, its DECODE row and the
//!   access the executor's history records for it: load or store, width,
//!   aligned to its width, inside one 8-byte word, one old timestamp over its
//!   width, one over the eight bytes from its address, and the route the
//!   walk's own rule gives it (`classify_memw`: MEMW_A or MEMW);
//! - every memory message the ECALL chips make, from the walk's MEMW_A rows and
//!   MEMW ops of the ECALL cycles, by syscall: width, alignment, and the
//!   accesses that touch an 8-byte word another access of the same cycle
//!   touched at the same timestamp.
//!
//! The totals of the median block are checked against the census taken of it
//! before (loads 31,324,197; stores 20,527,030; aligned with one old timestamp
//! over the width 46,389,351; unaligned 4,495,470; aligned with mixed
//! timestamps 966,406): the closing `PW0 VERDICT` line says whether they match.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use executor::elf::Elf;
use executor::vm::fast::FastExecutor;
use executor::vm::history::{ECALL, NO_ACCESS, WindowHistory};
use executor::vm::logs::Log;

use super::WindowedTraceBuilder;
use super::parallel::{ExecWindow, walk_windows_from_history};
use crate::tables::cpu::CpuOperation;
use crate::tables::decode::DecodeTable;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::tables::register::RegisterFormat;

/// The CPU accesses of one (load or store, width) class.
#[derive(Clone, Copy, Debug, Default)]
struct Class {
    count: u64,
    aligned: u64,
    in_word: u64,
    one_ts_width: u64,
    one_ts_word: u64,
    /// Aligned, one old timestamp over the width; and of those, the eight
    /// bytes' timestamps differing.
    aligned_one: u64,
    aligned_one_word_differs: u64,
    memw_a: u64,
}

/// An ECALL chip's memory messages of one width.
#[derive(Clone, Copy, Debug, Default)]
struct Chip {
    count: u64,
    aligned: u64,
    collide: u64,
}

#[derive(Default)]
struct Census {
    /// By (store, width).
    cpu: BTreeMap<(bool, u8), Class>,
    /// By (syscall, width).
    chips: BTreeMap<(u64, u8), Chip>,
    /// A CPU load or store whose history holds no access (refused: counted).
    no_access: u64,
    windows: u64,
    /// Windows counted from their logs and history only (the run's last).
    unwalked: u64,
}

impl Census {
    fn cpu_window(
        &mut self,
        first: u64,
        logs: &[Log],
        history: &WindowHistory,
        decode: &DecodeTable,
    ) {
        for (i, log) in logs.iter().enumerate() {
            let Some(row) = decode.row(log.current_pc) else {
                continue;
            };
            let entry = decode.entry(row);
            let f = &entry.fields;
            if f.ecall || !(f.is_load() || f.is_store()) {
                continue;
            }
            let side = history.extras[i].side;
            if side == NO_ACCESS || side & ECALL != 0 {
                self.no_access += 1;
                continue;
            }
            let a = history.accesses[side as usize];
            let cycle = first + i as u64;
            let ts = 4 * cycle + 4;
            let op = CpuOperation::from_log(log, ts, entry, row);
            let addr = decode.op(&op).res;
            let width = f.mem_bytes();
            let stamps: [u32; 8] = if a.mixed == NO_ACCESS {
                [a.ts; 8]
            } else {
                history.mixed[a.mixed as usize]
            };
            let low = addr as u32;
            let aligned = width == 1 || low & (width as u32 - 1) == 0;
            let in_word = (addr % 8) as usize + width <= 8;
            let one_width = stamps[..width].iter().all(|&t| t == stamps[0]);
            let one_word = stamps.iter().all(|&t| t == stamps[0]);
            let old = u64::from(stamps[0]);
            // `classify_memw` under the limbs' form: aligned, one old timestamp
            // over the width, a step on one high word.
            let memw_a = aligned && one_width && old >> 32 == ts >> 32 && old < ts;
            let c = self.cpu.entry((f.is_store(), width as u8)).or_default();
            c.count += 1;
            c.aligned += u64::from(aligned);
            c.in_word += u64::from(in_word);
            c.one_ts_width += u64::from(one_width);
            c.one_ts_word += u64::from(one_word);
            c.aligned_one += u64::from(aligned && one_width);
            c.aligned_one_word_differs += u64::from(aligned && one_width && !one_word);
            c.memw_a += u64::from(memw_a);
        }
    }

    /// The ECALL cycles' memory messages: `(cycle, address, width, timestamp)`
    /// in walk order.
    fn chip_messages(
        &mut self,
        first: u64,
        logs: &[Log],
        decode: &DecodeTable,
        messages: impl Iterator<Item = (u64, u8, u64)>,
    ) {
        let mut seen: HashSet<(u64, u64)> = HashSet::new();
        let mut cycle_of_seen = u64::MAX;
        for (addr, width, ts) in messages {
            let cycle = (ts.saturating_sub(4)) / 4;
            let Some(log) = cycle.checked_sub(first).and_then(|i| logs.get(i as usize)) else {
                continue;
            };
            let is_ecall = decode
                .row(log.current_pc)
                .is_some_and(|r| decode.entry(r).fields.ecall);
            if !is_ecall {
                continue;
            }
            if cycle != cycle_of_seen {
                seen.clear();
                cycle_of_seen = cycle;
            }
            let width = width.max(1);
            let words = (addr / 8)..=(addr.wrapping_add(u64::from(width) - 1) / 8);
            let mut collide = false;
            for word in words {
                collide |= !seen.insert((word, ts));
            }
            let aligned = width == 1 || (addr as u32) & (u32::from(width) - 1) == 0;
            let c = self.chips.entry((log.src1_val, width)).or_default();
            c.count += 1;
            c.aligned += u64::from(aligned);
            c.collide += u64::from(collide);
        }
    }

    /// Prints the lines; the totals `(loads, stores, aligned with one old
    /// timestamp, of those the word's differing, unaligned, aligned mixed)`.
    fn print(&self, block: &str) -> [u64; 6] {
        let mut t = [0u64; 6];
        for (&(store, width), c) in &self.cpu {
            println!(
                "BLOCK MEM WIDTHS: {block} CPU {} width {width}: {} · aligned {} · inside one word {} · one old ts over the width {} · over the word {} · aligned with one over the width {} (the word's differing {}) · routed MEMW_A {} MEMW {}",
                if store { "store" } else { "load" },
                c.count,
                c.aligned,
                c.in_word,
                c.one_ts_width,
                c.one_ts_word,
                c.aligned_one,
                c.aligned_one_word_differs,
                c.memw_a,
                c.count - c.memw_a
            );
            t[usize::from(store)] += c.count;
            t[2] += c.aligned_one;
            t[3] += c.aligned_one_word_differs;
            t[4] += c.count - c.aligned;
            t[5] += c.aligned - c.aligned_one;
        }
        for (&(syscall, width), c) in &self.chips {
            println!(
                "BLOCK MEM WIDTHS: {block} ECALL {syscall:#x} width {width}: {} · aligned {} · touch a word touched at the same timestamp {}",
                c.count, c.aligned, c.collide
            );
        }
        let routed: u64 = self.cpu.values().map(|c| c.memw_a).sum();
        println!(
            "BLOCK MEM WIDTHS: {block} totals: loads {} · stores {} · aligned with one old ts over the width {} (the word's differing {}) · unaligned {} · aligned mixed {} · routed MEMW_A {routed} · no access {} · windows {} walked, {} not (their ECALL messages not counted)",
            t[0], t[1], t[2], t[3], t[4], t[5], self.no_access, self.windows, self.unwalked
        );
        t
    }
}

/// The windows the executor sent, by first cycle: their logs and history.
type Sent = Mutex<BTreeMap<u64, (Arc<Vec<Log>>, Arc<WindowHistory>)>>;

/// The census of `program` run on `input` in record mode at windows of
/// `window` cycles, walked on `walkers` threads under the in-row register
/// format in the limbs' form, each window counted and let go.
fn census(program: &Elf, input: &[u8], window: usize, walkers: usize) -> Census {
    let options = crate::block_whir::BlockOptions::production();
    let mut builder = WindowedTraceBuilder::new(program, input, &options.max_rows)
        .and_then(|b| b.register_format(RegisterFormat::InRow))
        .and_then(|b| b.memw_aligned_order(MemwAlignedOrder::Limbs))
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .expect("the builder");
    let decode = Arc::clone(&builder.decode);
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let sent: Sent = Mutex::new(BTreeMap::new());
    let mut out = Census::default();
    std::thread::scope(|scope| {
        let sent = &sent;
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, input.to_vec()).expect("the fast engine");
            fast.enable_history().expect("history mode");
            while let Some((logs, history)) = fast.resume_history(window).expect("the run") {
                let logs = Arc::new(logs);
                sent.lock().expect("the windows").insert(
                    history.first,
                    (Arc::clone(&logs), Arc::new(history.clone())),
                );
                let last = (fast.pc() == 0)
                    .then(|| fast.take_history_final().map(Box::new))
                    .flatten();
                let window = ExecWindow {
                    logs,
                    start: None,
                    history: Some(Box::new(history)),
                    last,
                };
                if tx.send(window).is_err() {
                    break;
                }
            }
        });
        walk_windows_from_history(&mut builder, rx, walkers, 0, |accumulator, walked, _| {
            let first = walked.first as u64;
            let (logs, history) = sent
                .lock()
                .expect("the windows")
                .remove(&first)
                .expect("the window was sent");
            out.cpu_window(first, &logs, &history, &decode);
            let aligned = walked
                .walk
                .memw
                .aligned
                .iter()
                .map(|r| (r.base_address(), r.width(), r.timestamp()));
            let general = walked
                .walk
                .memw
                .general
                .iter()
                .filter(|op| !op.is_register)
                .map(|op| (op.base_address, op.width, op.timestamp));
            out.chip_messages(first, &logs, &decode, aligned.chain(general));
            out.windows += 1;
            // The window's chunks are let go unbuilt.
            drop(accumulator.absorb(walked));
            Ok(())
        })
        .expect("the walk");
    });
    // The run's last window is the finish's, not the walkers': its CPU
    // accesses are counted from its logs and history all the same (its ECALL
    // chips' messages, which need its walk, are not).
    for (first, (logs, history)) in sent.into_inner().expect("the windows") {
        out.cpu_window(first, &logs, &history, &decode);
        out.unwalked += 1;
    }
    out
}

/// ★ The probe: `BLOCK_WHIR_ELF` and `BLOCK_WHIR_INPUT` (and `PW0_BLOCK`, the
/// block's name: `median` checks the census) counted at the production
/// window on four walkers. The run's last window (the finish's) is not
/// walked: its CPU accesses are counted from its logs and history, its ECALL
/// chips' messages are not.
#[test]
#[ignore = "a box probe (BLOCK_WHIR_ELF, BLOCK_WHIR_INPUT, PW0_BLOCK)"]
fn count_the_memory_widths_of_a_real_block() {
    let elf =
        std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF")).expect("the ELF");
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).expect("input");
    let block = std::env::var("PW0_BLOCK").unwrap_or_else(|_| "block".to_string());
    let program = Elf::load(&elf).expect("the ELF loads");
    let window = 1usize
        << crate::block_whir::BlockOptions::production()
            .window_log2
            .expect("the block streams");
    let t = std::time::Instant::now();
    let c = census(&program, &input, window, 4);
    let totals = c.print(&block);
    // The median block's census, taken before (loads, stores, aligned with
    // one old timestamp over the width, of those the word's differing,
    // unaligned, aligned mixed).
    const MEDIAN: [u64; 6] = [
        31_324_197, 20_527_030, 46_389_351, 5_002_762, 4_495_470, 966_406,
    ];
    let verdict = match block.as_str() {
        "median" if totals == MEDIAN => "GUARD PASS (the census's counts)",
        "median" => "GUARD FAIL (not the census's counts: VOID)",
        _ => "INFO (no census to check)",
    };
    println!(
        "PW0 VERDICT {block}: {verdict} · totals {totals:?} · no access {} · {:.1} s",
        c.no_access,
        t.elapsed().as_secs_f64()
    );
}

/// A Rust guest's ELF from `program_artifacts/rust`.
fn rust_elf(name: &str) -> Elf {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../executor/program_artifacts/rust")
        .join(format!("{name}.elf"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Elf::load(&bytes).expect("the ELF loads")
}

/// The counter on a small record-mode run: every load and store counted
/// once, each class's subsets no larger than it, and the ECALL chips seen.
#[test]
fn the_memory_width_census_counts_every_access_once() {
    let runs = [
        (
            "test_keccak_multi",
            crate::tests::windowed_builder_tests::run("test_keccak_multi").0,
            7usize,
        ),
        ("dma_memmove_cases", rust_elf("dma_memmove_cases"), 97),
    ];
    let mut widths = HashSet::new();
    for (name, program, window) in &runs {
        let c = census(program, &[], *window, 2);
        let totals = c.print(name);
        assert!(totals[0] + totals[1] > 0, "{name}: {totals:?}");
        assert_eq!(c.no_access, 0, "{name}");
        for (&(_, width), k) in &c.cpu {
            widths.insert(width);
            assert!(k.aligned <= k.count && k.in_word <= k.count && k.memw_a <= k.aligned_one);
            assert!(k.aligned_one <= k.aligned && k.aligned_one_word_differs <= k.aligned_one);
        }
        assert!(
            !c.chips.is_empty(),
            "{name}: no ECALL memory message counted"
        );
    }
    assert!(widths.len() >= 3, "{widths:?}");
}

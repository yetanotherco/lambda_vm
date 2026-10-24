//! The lean pass against the walk: at every window's start its timestamps and
//! registers are the walk's, and each window walked from its entries (values
//! from the executor's snapshot at the window's start) is the window the
//! sequential walk walks.

use std::sync::Arc;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;
use executor::vm::memory::MemorySnapshot;

use super::super::super::{MemoryState, RegisterState, WalkSizes};
use super::super::{WalkedWindow, Walker, WindowedTraceBuilder};
use super::split::{Back, Front};
use super::{LeanPass, Rows, UNTOUCHED};
use crate::tables::MaxRowsConfig;
use crate::tables::decode::DecodeTable;
use crate::test_utils::asm_elf_bytes;

const PROGRAMS: [&str; 8] = [
    "all_instructions_64",
    "lw_sw_offset_odd",
    "test_memw_split_ts",
    "test_keccak_multi",
    "test_blake3",
    "test_blake3_absorb",
    "test_commit_split",
    "test_ecsm",
];

/// Rust guests, built by `make compile-programs-rust`: longer runs over a heap
/// and a stack, KECCAK and ECSM from Rust, and HINT.
const RUST_PROGRAMS: [&str; 4] = ["hashmap", "keccak_precompile", "ecsm", "hint_min"];

fn programs() -> Vec<(String, Elf)> {
    let rust = |name: &str| {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the workspace root")
            .join(format!("executor/program_artifacts/rust/{name}.elf"));
        std::fs::read(&path).unwrap_or_else(|e| {
            panic!("{} — run `make compile-programs-rust`: {e}", path.display())
        })
    };
    PROGRAMS
        .iter()
        .map(|name| (name.to_string(), asm_elf_bytes(name)))
        .chain(
            RUST_PROGRAMS
                .iter()
                .map(|name| (name.to_string(), rust(name))),
        )
        .map(|(name, bytes)| (name, Elf::load(&bytes).expect("the ELF loads")))
        .collect()
}

/// The run's windows of `window` cycles, each with the executor's memory at
/// its start.
fn windows(elf: &Elf, window: usize) -> Vec<(Vec<Log>, MemorySnapshot)> {
    let mut executor = Executor::new(elf, Vec::new()).expect("the executor starts");
    let mut at_start = executor.memory().snapshot();
    let mut out = Vec::new();
    while let Some(logs) = executor
        .resume_with_limit(window)
        .expect("the run executes")
        .map(<[Log]>::to_vec)
    {
        let next = executor.memory().snapshot();
        out.push((logs, std::mem::replace(&mut at_start, next)));
    }
    out
}

/// 7 and 33 cycles for a short run, about a 40th and a 7th of a long one.
fn window_lengths(cycles: usize) -> [usize; 2] {
    if cycles < 5_000 {
        [7, 33]
    } else {
        [(cycles / 40) | 1, (cycles / 7) | 1]
    }
}

/// The lists two walks of one window differ in, by name.
fn differences(a: &WalkedWindow, b: &WalkedWindow) -> Vec<&'static str> {
    fn same<T: std::fmt::Debug>(x: &[T], y: &[T]) -> bool {
        x.len() == y.len() && format!("{x:?}") == format!("{y:?}")
    }
    let (x, y) = (&a.walk, &b.walk);
    [
        ("cpu", same(&a.cpu_ops, &b.cpu_ops)),
        ("memw_r", same(&x.memw.register_rows, &y.memw.register_rows)),
        ("memw_a", same(&x.memw.aligned, &y.memw.aligned)),
        ("memw", same(&x.memw.general, &y.memw.general)),
        ("load", same(&x.load_ops, &y.load_ops)),
        ("lt", same(&x.lt_ops, &y.lt_ops)),
        ("shift", same(&x.shift_ops, &y.shift_ops)),
        ("bitwise", same(&x.bitwise_ops, &y.bitwise_ops)),
        ("cpu32", same(&x.cpu32_ops, &y.cpu32_ops)),
        ("copy", same(&x.copy_ops, &y.copy_ops)),
        ("keccak", same(&x.keccak_ops, &y.keccak_ops)),
        ("blake3", same(&x.blake3_ops, &y.blake3_ops)),
        (
            "blake3_absorb",
            same(&x.blake3_absorb_ops, &y.blake3_absorb_ops),
        ),
        ("ecsm", same(&x.ecsm_ops, &y.ecsm_ops)),
        ("ecdas", same(&x.ecdas_ops, &y.ecdas_ops)),
        ("hint", same(&x.hint_ops, &y.hint_ops)),
    ]
    .into_iter()
    .filter(|&(_, same)| !same)
    .map(|(name, _)| name)
    .collect()
}

/// The register cells that differ.
fn register_differences(a: &RegisterState, b: &RegisterState) -> usize {
    (0..32).filter(|&r| a.regs[r] != b.regs[r]).count()
        + usize::from(a.index_register != b.index_register)
        + usize::from(a.pc_register != b.pc_register)
}

/// A copy of `r` (the walk's register state is not `Clone`).
fn copy(r: &RegisterState) -> RegisterState {
    RegisterState {
        regs: r.regs,
        index_register: r.index_register,
        pc_register: r.pc_register,
    }
}

/// A memory state holding only `entries`, each with its value in `snapshot`.
fn state_from_entries(entries: &[(u64, u64)], snapshot: &MemorySnapshot) -> MemoryState {
    let mut state = MemoryState::new();
    state.lean = true;
    for &(address, ts) in entries {
        state.cells.set(address, (snapshot.load_byte(address), ts));
    }
    state
}

/// What a window is walked with.
struct Setup {
    decode: Arc<DecodeTable>,
    lookups: bool,
    batch: usize,
}

impl Setup {
    /// One window walked by the production walker from `memory` and
    /// `registers`, from cycle `first`.
    fn walk(
        &self,
        memory: &mut MemoryState,
        registers: &mut RegisterState,
        first: usize,
        logs: &[Log],
    ) -> WalkedWindow {
        let (mut sizes, mut cycles, mut windows, mut secs) = (WalkSizes::default(), first, 0, 0.0);
        Walker {
            decode: &self.decode,
            lookups: self.lookups,
            batch: self.batch,
            sizes: &mut sizes,
            memory_state: memory,
            register_state: registers,
            cycles: &mut cycles,
            windows: &mut windows,
            walk_secs: &mut secs,
        }
        .walk(logs)
        .expect("a window walks")
    }
}

/// The memory bytes whose old timestamp the window's MEMW and MEMW_A rows keep
/// (within each access's width), last accessed before the window (timestamps
/// below `first_ts`), sorted.
fn first_accessed(window: &WalkedWindow, first_ts: u64) -> Vec<u64> {
    let mut out = Vec::new();
    for op in window.walk.memw.general.iter().filter(|op| !op.is_register) {
        for i in 0..op.width as usize {
            if op.old_timestamp[i] < first_ts {
                out.push(op.base_address.wrapping_add(i as u64));
            }
        }
    }
    for row in window
        .walk
        .memw
        .aligned
        .iter()
        .filter(|row| !row.is_register())
    {
        if row.old_timestamp() < first_ts {
            out.extend((0..row.width() as u64).map(|i| row.base_address().wrapping_add(i)));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// What one program run at one window length showed.
#[derive(Default)]
struct Seen {
    windows: usize,
    entries: usize,
    /// KECCAK + BLAKE3, ECSM + absorb, HINT ops walked.
    ecalls: [usize; 3],
    /// Entries a memory row keeps the old timestamp of, stamped before the
    /// window, each dropped once and its window walked again; and of them, the
    /// walks that differed.
    dropped: usize,
    caught: usize,
}

/// The pass and the walk side by side over `elf`'s run at `window` cycles a
/// window. At each window's start the pass's timestamps (of every byte the
/// walk holds and every entry) and registers must be the walk's, the entries
/// distinct, each entry's snapshot value the walk's; the window walked from
/// its entries must be the sequential walk's. With `drop`, up to three entries
/// a window that a memory row needs (see [`first_accessed`]) are each left out
/// once and the window walked again.
fn side_by_side(name: &str, elf: &Elf, window: usize, drop: bool) -> Seen {
    let mut builder = WindowedTraceBuilder::new(elf, &[], &MaxRowsConfig::small())
        .expect("the builder")
        .drop_streamed_ops()
        .expect("before any window");
    let setup = Setup {
        decode: Arc::clone(&builder.decode),
        lookups: !builder.lean.lookups,
        batch: builder.walk_batch,
    };
    let mut pass = LeanPass::new(Arc::clone(&builder.decode), &builder.register_state)
        .expect("the pass starts");
    let mut runs = windows(elf, window);
    // The run's last window is the finish's.
    runs.pop();
    let (mut walker, _accumulator) = builder.split();
    let mut seen = Seen::default();
    for (k, (logs, snapshot)) in runs.iter().enumerate() {
        let first = *walker.cycles;
        let regs = pass.registers();
        assert_eq!(
            register_differences(&regs, walker.register_state),
            0,
            "{name}/{window}, window {k}: registers at its start"
        );
        for (address, (_, ts)) in walker.memory_state.cells.iter() {
            assert_eq!(
                pass.timestamp(address),
                ts,
                "{name}/{window}, window {k}: the timestamp at {address:#x}"
            );
        }
        pass.begin(u32::try_from(k).expect("few windows"))
            .expect("a window to mark");
        pass.pass(logs, first).expect("the pass");
        let entries = pass.entries().to_vec();
        let mut distinct: Vec<u64> = entries.iter().map(|&(a, _)| a).collect();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            entries.len(),
            "{name}/{window}, window {k}: entries repeat"
        );
        for &(address, ts) in &entries {
            assert_eq!(
                walker.memory_state.read_byte(address),
                (snapshot.load_byte(address), ts),
                "{name}/{window}, window {k}: the entry at {address:#x}"
            );
        }
        let mut memory = state_from_entries(&entries, snapshot);
        let mut registers = copy(&regs);
        let again = setup.walk(&mut memory, &mut registers, first, logs);
        let walked = walker.walk(logs).expect("the sequential walk");
        assert_eq!(
            differences(&again, &walked),
            Vec::<&str>::new(),
            "{name}/{window}, window {k}: walked from its entries"
        );
        if drop {
            // Bytes whose old timestamp a memory row of the window keeps: an
            // entry for one stamped before (timestamp > 0) must be needed.
            let kept = first_accessed(&walked, first as u64 * 4 + 4);
            let full: Vec<usize> = (0..entries.len())
                .filter(|&i| entries[i].1 > 0 && kept.binary_search(&entries[i].0).is_ok())
                .collect();
            let picks = [full.first(), full.get(full.len() / 2), full.last()];
            let mut picked: Vec<usize> = picks.into_iter().flatten().copied().collect();
            picked.dedup();
            for i in picked {
                let mut fewer = entries.clone();
                fewer.remove(i);
                let mut memory = state_from_entries(&fewer, snapshot);
                let mut registers = copy(&regs);
                // A wrong ECSM operand stops the walk (its witness refuses the
                // point): seen as well.
                let short = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    setup.walk(&mut memory, &mut registers, first, logs)
                }));
                seen.dropped += 1;
                seen.caught += usize::from(
                    short.map_or(true, |short| !differences(&short, &again).is_empty()),
                );
            }
        }
        seen.windows += 1;
        seen.entries += entries.len();
        let w = &walked.walk;
        seen.ecalls[0] += w.keccak_ops.len() + w.blake3_ops.len();
        seen.ecalls[1] += w.ecsm_ops.len() + w.blake3_absorb_ops.len();
        seen.ecalls[2] += w.hint_ops.len();
    }
    seen
}

fn cycles_of(elf: &Elf) -> usize {
    Executor::new(elf, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs
        .len()
}

/// ★ For twelve programs at two window lengths each, at every window's start
/// the pass holds the walk's timestamps and registers, its entries hold the
/// walk's timestamps and the snapshot holds the walk's values there, and the
/// window walked from them alone is the sequential walk's.
#[test]
fn the_pass_keeps_the_walks_timestamps_registers_and_entries() {
    let mut total = Seen::default();
    for (name, elf) in programs() {
        for window in window_lengths(cycles_of(&elf)) {
            let seen = side_by_side(&name, &elf, window, false);
            total.windows += seen.windows;
            total.entries += seen.entries;
            for (t, s) in total.ecalls.iter_mut().zip(seen.ecalls) {
                *t += s;
            }
        }
    }
    assert!(
        total.ecalls.iter().all(|&n| n > 0),
        "the programs reach KECCAK/BLAKE3, ECSM/absorb and HINT: {:?}",
        total.ecalls
    );
    assert!(total.entries > 1000, "{} entries recorded", total.entries);
    assert!(total.windows > 200, "{} windows passed", total.windows);
}

/// ★ The entries are needed: an entry for a byte stamped before the window
/// whose old timestamp a memory row keeps, left out, walks the window
/// differently (or stops it), every time. (Not every entry is: a load's value
/// comes from its log, and a row keeps the old timestamps of its width only.)
#[test]
fn a_window_walked_without_one_of_its_entries_is_another_walk() {
    let mut dropped = 0;
    for (name, elf) in programs() {
        let window = window_lengths(cycles_of(&elf))[0];
        let seen = side_by_side(&name, &elf, window, true);
        assert_eq!(
            seen.caught, seen.dropped,
            "{name}: an entry left out went unseen"
        );
        dropped += seen.dropped;
    }
    assert!(dropped > 50, "{dropped} entries left out");
}

/// The dense pc index finds every program's rows as the decode table does,
/// and nothing where it finds nothing.
#[test]
fn the_dense_pc_index_finds_the_decode_tables_rows() {
    for (name, elf) in programs() {
        let builder =
            WindowedTraceBuilder::new(&elf, &[], &MaxRowsConfig::small()).expect("the builder");
        let decode = &builder.decode;
        let rows = Rows::of(decode);
        assert!(matches!(rows, Rows::Dense { .. }), "{name}: a dense index");
        let n = decode.padding_row() as u32;
        let last = decode.entry(n - 1).pc;
        for r in 0..n {
            let pc = decode.entry(r).pc;
            for probe in [pc, pc + 1, pc + 2, pc + 3, pc.wrapping_sub(4), last + 4, 0] {
                assert_eq!(
                    rows.row(probe, decode),
                    decode.row(probe),
                    "{name}: pc {probe:#x}"
                );
            }
        }
    }
}

/// The pass with its memory left out (a measurement's arm) keeps the same
/// registers and records no entry; both find rows through the dense index.
#[test]
fn the_pass_without_its_memory_keeps_the_same_registers() {
    for (name, elf) in programs() {
        let window = window_lengths(cycles_of(&elf))[1];
        let builder =
            WindowedTraceBuilder::new(&elf, &[], &MaxRowsConfig::small()).expect("the builder");
        let decode = Arc::clone(&builder.decode);
        let mut full = LeanPass::new(Arc::clone(&decode), &builder.register_state).expect("a pass");
        let mut bare = LeanPass::new(decode, &builder.register_state).expect("a pass");
        assert!(full.dense_rows() && bare.dense_rows(), "{name}");
        let mut first = 0;
        for (k, (logs, _)) in windows(&elf, window).iter().enumerate() {
            let k = u32::try_from(k).expect("few windows");
            full.begin(k).expect("a window");
            bare.begin(k).expect("a window");
            full.pass(logs, first).expect("the pass");
            bare.pass_registers_only(logs, first).expect("the pass");
            first += logs.len();
            assert!(bare.entries().is_empty(), "{name}, window {k}");
        }
        // ECALLs also write x10..x13 (and COMMIT the index); the bare pass
        // leaves them out.
        let (a, b) = (full.registers(), bare.registers());
        let general = (1..32).filter(|&r| !(10..=13).contains(&r));
        for r in general {
            assert_eq!(a.regs[r], b.regs[r], "{name}: x{r}");
        }
        assert_eq!(a.pc_register, b.pc_register, "{name}: pc");
    }
}

/// The pass split over two threads (a measurement's arm) records the pass's
/// entries, in order, and keeps its registers, at every window.
#[test]
fn the_split_pass_is_the_pass() {
    for (name, elf) in programs() {
        for window in window_lengths(cycles_of(&elf)) {
            let builder =
                WindowedTraceBuilder::new(&elf, &[], &MaxRowsConfig::small()).expect("the builder");
            let decode = Arc::clone(&builder.decode);
            let mut pass =
                LeanPass::new(Arc::clone(&decode), &builder.register_state).expect("a pass");
            let mut front = Front::new(decode, &builder.register_state).expect("a front");
            let mut back = Back::new();
            let (mut first, mut accesses) = (0, Vec::new());
            for (k, (logs, _)) in windows(&elf, window).iter().enumerate() {
                let k = u32::try_from(k).expect("few windows");
                pass.begin(k).expect("a window");
                back.begin(k).expect("a window");
                pass.pass(logs, first).expect("the pass");
                accesses.clear();
                front.pass(logs, first, &mut accesses).expect("the front");
                back.apply(&accesses);
                first += logs.len();
                assert_eq!(
                    back.entries(),
                    pass.entries(),
                    "{name}/{window}, window {k}"
                );
                assert_eq!(
                    register_differences(&front.registers(), &pass.registers()),
                    0,
                    "{name}/{window}, window {k}"
                );
            }
        }
    }
}

/// A window the pass cannot mark is refused; the last it can is taken.
#[test]
fn the_pass_refuses_a_window_past_its_marks() {
    let elf = Elf::load(&asm_elf_bytes(PROGRAMS[0])).expect("the ELF loads");
    let builder =
        WindowedTraceBuilder::new(&elf, &[], &MaxRowsConfig::small()).expect("the builder");
    let mut pass = LeanPass::new(Arc::clone(&builder.decode), &builder.register_state)
        .expect("the pass starts");
    assert!(pass.begin(UNTOUCHED as u32 - 1).is_ok());
    assert!(pass.begin(UNTOUCHED as u32).is_err());
}

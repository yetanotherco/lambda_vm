//! Regeneration's pieces of the windowed builder (D-REGEN): the first-touch
//! census of a walked window, the regeneration walk and the regenerator.

use executor::elf::Elf;
use executor::vm::execution::Executor;

use super::{RegenBuilder, StreamTable, WindowedTraceBuilder};
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;

/// [`super::WalkedWindow::first_touch_bytes`] counts each byte a window touches
/// once: on the keccak programs it is the number of distinct memory bytes the
/// window's accesses cover, window by window, at windows of 7 and 33 cycles.
#[test]
fn first_touch_bytes_are_the_distinct_bytes_a_window_touches() {
    for name in ["test_keccak", "test_keccak_multi", "lw_sw_offset_odd"] {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        let logs = Executor::new(&program, Vec::new())
            .expect("the executor starts")
            .run()
            .expect("the program runs")
            .logs;
        for window in [7, 33] {
            let mut builder = WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small())
                .expect("the builder");
            let (mut walker, _) = builder.split();
            let mut touched_any = false;
            for (i, w) in logs[..logs.len() - 1].chunks(window).enumerate() {
                let walked = walker.walk(w).expect("a window");
                assert_eq!(walked.cycles(), w.len());
                let memw = &walked.walk.memw;
                let distinct: std::collections::BTreeSet<u64> = memw
                    .aligned
                    .iter()
                    .filter(|row| !row.is_register())
                    .map(|row| (row.base_address(), row.width()))
                    .chain(
                        memw.general
                            .iter()
                            .filter(|op| !op.is_register)
                            .map(|op| (op.base_address, op.width)),
                    )
                    .flat_map(|(base, width)| {
                        (0..u64::from(width)).map(move |k| base.wrapping_add(k))
                    })
                    .collect();
                assert_eq!(
                    walked.first_touch_bytes(),
                    distinct.len(),
                    "{name}, windows of {window}, window {i}"
                );
                touched_any |= !distinct.is_empty();
            }
            assert!(touched_any, "{name}: no window touched memory");
        }
    }
}

/// The asm programs the regeneration tests run: every walk path (loads and
/// stores of every width, word instructions, COMMIT, KECCAK, BLAKE3 and its
/// absorb, ECSM).
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

fn run_logs(program: &Elf) -> Vec<executor::vm::logs::Log> {
    Executor::new(program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs
}

/// The streamed lists of a walk, printed field by field.
fn streamed_lists(cpu_ops: &[super::super::CpuOperation], walk: &super::WalkOutputs) -> String {
    let memw = &walk.memw;
    format!(
        "cpu {cpu_ops:?}\nmemw_r {:?}\nmemw_a {:?}\nmemw {:?}\nload {:?}\nlt {:?}\nshift {:?}",
        memw.register_rows, memw.aligned, memw.general, walk.load_ops, walk.lt_ops, walk.shift_ops
    )
}

/// ★ The regeneration walk is the walk on the streamed tables' lists: over
/// windows of 1, 7 and 33 cycles it emits the CPU, MEMW_R, MEMW_A, MEMW, LOAD,
/// LT and SHIFT ops the full walk emits (field by field), leaves every other
/// list empty, and ends in the same memory and register state.
#[test]
fn the_regeneration_walk_is_the_walk_on_its_streamed_lists() {
    use super::super::{
        DecodeArtifacts, MemoryState, RegisterState, WalkOutputs, build_initial_image,
        collect_cpu_ops_into, collect_ops_from_cpu_into, collect_streamed_ops_from_cpu_into,
        register,
    };
    let mut rest_seen = 0;
    for name in PROGRAMS {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        let logs = run_logs(&program);
        let artifacts = DecodeArtifacts::from_elf(&program).expect("the decode artifacts");
        let decode = &artifacts.decode;
        let image = build_initial_image(&program, &[]);
        let register_init = register::register_init_from_entry_point(program.entry_point);
        for window in [1, 7, 33] {
            let mut states = [
                (
                    MemoryState::from_image(&image),
                    RegisterState::from_init(&register_init),
                ),
                (
                    MemoryState::from_image(&image),
                    RegisterState::from_init(&register_init),
                ),
            ];
            for (memory, _) in states.iter_mut() {
                memory.lean = true;
            }
            let mut cycles = 0;
            for (k, part) in logs.chunks(window).enumerate() {
                let mut cpu_ops = Vec::new();
                collect_cpu_ops_into(part, decode, cycles, &mut cpu_ops).expect("the CPU ops");
                cycles += part.len();
                let [(full_mem, full_reg), (lite_mem, lite_reg)] = &mut states;
                let mut full = WalkOutputs::for_walk(cpu_ops.len(), false);
                collect_ops_from_cpu_into(&cpu_ops, decode, full_mem, full_reg, &mut full, false);
                let mut lite = WalkOutputs::for_walk(cpu_ops.len(), false);
                collect_streamed_ops_from_cpu_into(&cpu_ops, decode, lite_mem, lite_reg, &mut lite);
                assert_eq!(
                    streamed_lists(&cpu_ops, &full),
                    streamed_lists(&cpu_ops, &lite),
                    "{name}, windows of {window}, window {k}"
                );
                let rest = |w: &WalkOutputs| {
                    w.bitwise_ops.len()
                        + w.commit_ops.len()
                        + w.keccak_ops.len()
                        + w.blake3_ops.len()
                        + w.blake3_absorb_ops.len()
                        + w.cpu32_ops.len()
                        + w.ecsm_ops.len()
                        + w.ecdas_ops.len()
                        + w.hint_ops.len()
                };
                assert_eq!(
                    rest(&lite),
                    0,
                    "{name}: the regeneration walk listed a finish's op"
                );
                rest_seen += rest(&full);
            }
            let [(full_mem, full_reg), (lite_mem, lite_reg)] = &states;
            let cells = |m: &MemoryState| m.cells.iter().collect::<Vec<_>>();
            assert!(
                cells(full_mem) == cells(lite_mem),
                "{name}, windows of {window}: memory"
            );
            let registers = |r: &RegisterState| (r.regs, r.index_register, r.pc_register);
            assert_eq!(
                registers(full_reg),
                registers(lite_reg),
                "{name}, windows of {window}: registers"
            );
        }
    }
    assert!(rest_seen > 0, "no program made an op only the finish reads");
}

/// Each phase-A chunk's packed digest, by (table, index): a windowed build that
/// drops its streamed ops, walked `window` cycles at a time, every handed-out
/// chunk generated and packed.
fn handed_out_digests(
    program: &Elf,
    logs: &[executor::vm::logs::Log],
    max_rows: &MaxRowsConfig,
    window: usize,
) -> std::collections::BTreeMap<(usize, usize), [u64; 2]> {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .expect("the builder")
        .drop_streamed_ops()
        .expect("before the first window");
    let (body, _last) = logs.split_at(logs.len() - logs.len() % window.max(1));
    let body = if body.len() == logs.len() {
        &body[..body.len().saturating_sub(window)]
    } else {
        body
    };
    let mut digests = std::collections::BTreeMap::new();
    for w in body.chunks(window) {
        for job in builder.push_jobs(w).expect("a window") {
            let key = (slot(job.table), job.index);
            assert!(
                digests.insert(key, packed_digest(job)).is_none(),
                "{key:?} twice"
            );
        }
    }
    digests
}

fn slot(table: StreamTable) -> usize {
    StreamTable::ALL
        .iter()
        .position(|&t| t == table)
        .expect("a streamed table")
}

fn packed_digest(job: super::ChunkJob) -> [u64; 2] {
    let mut trace = job.generate().trace;
    assert!(trace.pack_main_narrow(), "the chunk packs");
    trace.narrow_main().expect("packed").digest()
}

/// Each chunk a regenerator cuts, walked `window` cycles at a time over the
/// whole run, by (table, index).
fn regenerated_digests(
    builder: RegenBuilder,
    logs: &[executor::vm::logs::Log],
    window: usize,
) -> std::collections::BTreeMap<(usize, usize), [u64; 2]> {
    let mut builder = builder;
    let mut digests = std::collections::BTreeMap::new();
    for w in logs.chunks(window) {
        for job in builder.push(w).expect("a window") {
            let key = (slot(job.table), job.index);
            assert!(
                digests.insert(key, packed_digest(job)).is_none(),
                "{key:?} twice"
            );
        }
    }
    assert_eq!(builder.cycles(), logs.len());
    digests
}

/// ★ The regenerator rebuilds every chunk phase A handed out, byte for byte:
/// on every program, phase A walked 5 cycles at a time and the regenerator 1,
/// 7 and 33 at a time, each phase-A chunk's packed digest is the regenerated
/// one; the regenerator's other chunks are the finish's (past phase A's).
#[test]
fn the_regenerator_rebuilds_the_handed_out_chunks() {
    let max_rows = MaxRowsConfig {
        keccak_rnd: 48,
        ..MaxRowsConfig::small()
    };
    let mut handed = 0;
    for name in PROGRAMS {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        let logs = run_logs(&program);
        let phase_a = handed_out_digests(&program, &logs, &max_rows, 5);
        handed += phase_a.len();
        for window in [1, 7, 33] {
            let regen = RegenBuilder::new(&program, &[], &max_rows).expect("the regenerator");
            let again = regenerated_digests(regen, &logs, window);
            for (key, digest) in &phase_a {
                assert_eq!(
                    again.get(key),
                    Some(digest),
                    "{name}, windows of {window}: chunk {key:?}"
                );
            }
            let mut last = [None; 8];
            for &(t, i) in phase_a.keys() {
                last[t] = Some(last[t].map_or(i, |l: usize| l.max(i)));
            }
            for &(t, i) in again.keys() {
                assert!(
                    phase_a.contains_key(&(t, i)) || last[t].is_none_or(|l| i > l),
                    "{name}, windows of {window}: chunk ({t}, {i}) is not phase A's and not past it"
                );
            }
        }
    }
    assert!(handed > 0, "no program handed out a chunk");
}

/// ★ A slicer that drops one op is caught: for each streamed table phase A
/// handed out a chunk of, the regenerator dropping that table's first op
/// rebuilds a different chunk at the first index, and leaves the other tables'
/// chunks as they were.
#[test]
fn a_slicer_that_drops_an_op_is_caught() {
    let max_rows = MaxRowsConfig::small();
    let mut caught = 0;
    for name in ["all_instructions_64", "test_keccak_multi"] {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        let logs = run_logs(&program);
        let phase_a = handed_out_digests(&program, &logs, &max_rows, 7);
        for table in StreamTable::ALL {
            let t = slot(table);
            if !phase_a.contains_key(&(t, 0)) {
                continue;
            }
            let regen = RegenBuilder::new(&program, &[], &max_rows)
                .expect("the regenerator")
                .drop_one_op(table);
            let again = regenerated_digests(regen, &logs, 7);
            assert_ne!(
                again.get(&(t, 0)),
                phase_a.get(&(t, 0)),
                "{name}: {table:?} chunk 0 with an op dropped"
            );
            for (key, digest) in phase_a.iter().filter(|(k, _)| k.0 != t) {
                assert_eq!(again.get(key), Some(digest), "{name}: {key:?}");
            }
            caught += 1;
        }
    }
    assert!(caught >= 4, "only {caught} tables checked");
}

//! G-pack (`tables::gpack`): a table written packed as it is generated is the
//! 64-bit table packed, byte for byte — on every path the writer takes, for
//! every table of a windowed build in the block's configuration, and in the
//! block's phase-A stream — and the comparison that says so catches a column
//! written at the wrong width or offset.

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;
use stark::narrow::NarrowMain;
use stark::narrow::mutation::{self, Mutation};
use stark::trace::TraceTable;

use crate::tables::MaxRowsConfig;
use crate::tables::gpack::{TraceForm, WidthHint, generate_main, thread_counts};
use crate::tables::trace_builder::{Traces, WindowedTraceBuilder};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, VmTable, fe_words};
use crate::test_utils::asm_elf_bytes;

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

fn run(name: &str) -> (Elf, Vec<Log>) {
    let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    (program, logs)
}

/// The whole-run build, every table at 8 bytes a cell: today's words.
fn whole(program: &Elf, logs: &[Log], max_rows: &MaxRowsConfig) -> Traces {
    Traces::from_elf_and_logs(
        program,
        logs,
        max_rows,
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("the whole-run build")
}

/// A windowed build in the block's configuration under G-pack: the streamed
/// ops dropped, each streamed chunk written packed on this thread
/// (`ChunkJob::generate_as`), the finish's tables written packed as they are
/// generated (`generate_packed`).
fn gpack_build(program: &Elf, logs: &[Log], max_rows: &MaxRowsConfig, window: usize) -> Traces {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .expect("the builder")
        .drop_streamed_ops()
        .expect("before any window")
        .pack_finished_tables()
        .generate_packed();
    let body = logs.len() - 1;
    let cut = body - body % window;
    let mut chunks = Vec::new();
    for w in logs[..cut].chunks(window) {
        for job in builder.push_jobs(w).expect("a window") {
            chunks.push(job.generate_as(TraceForm::Narrow));
        }
    }
    let mut traces = builder.finish(&logs[cut..]).expect("the last window");
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    traces
}

/// Every table of `t`, named by kind and instance.
fn tables(t: &Traces) -> Vec<(String, &Table)> {
    let lists: [(&str, &Vec<Table>); 21] = [
        ("CPU", &t.cpus),
        ("MEMW_R", &t.memw_registers),
        ("MEMW_A", &t.memw_aligneds),
        ("MEMW", &t.memws),
        ("LOAD", &t.loads),
        ("STORE", &t.stores),
        ("SHIFT", &t.shifts),
        ("CPU32", &t.cpu32s),
        ("COMMIT", &t.commits),
        ("KECCAK", &t.keccaks),
        ("KECCAK_RND", &t.keccak_rnds),
        ("ECSM", &t.ecsms),
        ("ECDAS", &t.ecdases),
        ("HINT", &t.hints),
        ("PAGE", &t.pages),
        ("LT", &t.lts),
        ("MUL", &t.muls),
        ("DVRM", &t.dvrms),
        ("BRANCH", &t.branches),
        ("EQ", &t.eqs),
        ("BYTEWISE", &t.bytewises),
    ];
    let mut out = Vec::new();
    for (name, list) in lists {
        for (i, table) in list.iter().enumerate() {
            out.push((format!("{name}[{i}]"), table));
        }
    }
    for (name, table) in [
        ("BITWISE", &t.bitwise),
        ("DECODE", &t.decode),
        ("REGISTER", &t.register),
        ("HALT", &t.halt),
        ("KECCAK_RC", &t.keccak_rc),
        ("BLAKE3", &t.blake3),
    ] {
        out.push((name.to_string(), table));
    }
    out
}

/// A table's raw words, row-major: widened when it is packed.
fn words(t: &Table) -> Vec<u64> {
    match t.narrow_main() {
        Some(narrow) => {
            let mut words = vec![0u64; narrow.rows() * narrow.cols()];
            narrow.widen_into(&mut words);
            words
        }
        None => fe_words(t.main_table.row_major_data()).to_vec(),
    }
}

/// How a table differs from the reference's.
#[derive(Debug, PartialEq, Eq)]
enum Miss {
    /// Another table count, width or height.
    Shape,
    /// Packed, at other bytes than the reference's 64-bit table packs to.
    Bytes,
    /// Other words.
    Words,
}

/// ★ The gate's comparison: every table of `got` has the shape and the words
/// of `reference`'s (a whole-run build at 8 bytes a cell), and a packed one
/// has the bytes (`NarrowMain::digest`: rows, widths and every byte)
/// `NarrowMain::pack` gives the reference's words. Also how many of `got`'s
/// tables are packed.
fn misses(reference: &Traces, got: &Traces) -> (Vec<(String, Miss)>, usize) {
    let (want, have) = (tables(reference), tables(got));
    let mut out = Vec::new();
    if want.len() != have.len() {
        out.push(("tables".to_string(), Miss::Shape));
        return (out, 0);
    }
    let mut packed = 0;
    for ((name, a), (_, b)) in want.iter().zip(&have) {
        let shape = |t: &Table| (t.main_table.width, t.main_table.height);
        if shape(a) != shape(b) {
            out.push((name.clone(), Miss::Shape));
            continue;
        }
        let want_words = words(a);
        if let Some(narrow) = b.narrow_main() {
            packed += 1;
            let want_packed = NarrowMain::pack(&want_words, a.main_table.width);
            if narrow.digest() != want_packed.digest() {
                out.push((name.clone(), Miss::Bytes));
            }
        }
        if words(b) != want_words {
            out.push((name.clone(), Miss::Words));
        }
    }
    (out, packed)
}

/// Columns topping out at `tops[c]` (in the row before the last), one written
/// as a field element above the modulus; column 3 a flag.
fn synthetic(t: &mut impl VmTable, rows: usize, tops: [u64; 2]) {
    for r in 0..rows {
        let top = r == rows - 2;
        t.set_u64(r, 0, r as u64 % 200);
        for (c, &m) in tops.iter().enumerate() {
            let v = if top {
                m
            } else {
                (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) % (m / 2 + 1)
            };
            if v != 0 {
                t.set_u64(r, 1 + c, v);
            }
        }
        t.set_bool(r, 3, r % 3 == 0);
        if r % 5 == 0 {
            // Reduced mod p as `FE::from` reduces it: 2^32 − 2.
            t.set_fe(r, 4, FE::from(u64::MAX));
        }
    }
}

/// `synthetic` at 8 bytes a cell, as a generator writes it today.
fn synthetic_wide(rows: usize, tops: [u64; 2]) -> Table {
    let mut trace = TraceTable::new_main(crate::tables::types::zeroed_fe_vec(rows * 5), 5, 1);
    synthetic(&mut trace.main_table, rows, tops);
    trace
}

/// ★ Every path `generate_main!` takes gives the 64-bit table packed: no
/// widths yet (built wide, then packed, the widths learned), the widths
/// right (written packed as it is), a word wider than its column (written
/// again at the widths the trace needs), columns narrower than learned
/// (narrowed in place). `TraceForm::Wide` gives the 64-bit table.
#[test]
fn every_path_of_the_writer_gives_the_wide_table_packed() {
    static HINT: WidthHint = WidthHint::new();
    let rows = 1000;
    let build = |form, tops| generate_main!(form, &HINT, rows, 5, |t| synthetic(t, rows, tops));
    // (tops, the path: wide then packed, as written, written again, narrowed).
    let steps = [
        ([0x1234, 0xff], [0, 0, 0, 1]),
        ([0x1234, 0xff], [1, 0, 0, 0]),
        ([0x1_0000_0000, 0xff], [0, 0, 1, 0]),
        ([0x12, 0x1_0000], [0, 0, 1, 0]),
        ([0x12, 0x12], [0, 1, 0, 0]),
    ];
    for (i, (tops, path)) in steps.into_iter().enumerate() {
        let wide = synthetic_wide(rows, tops);
        let before = thread_counts();
        let got = build(TraceForm::Narrow, tops);
        let after = thread_counts();
        let taken: Vec<usize> = (0..4).map(|k| after[k] - before[k]).collect();
        assert_eq!(taken, path, "step {i}: the path");
        let want = NarrowMain::pack(&words(&wide), 5);
        assert_eq!(got.narrow_main(), Some(&want), "step {i}: the packed bytes");
        assert_eq!(words(&got), words(&wide), "step {i}: the words");
        // The 64-bit form is today's table and takes no G-pack path.
        let plain = build(TraceForm::Wide, tops);
        assert!(!plain.is_main_narrow());
        assert_eq!(
            plain.main_table, wide.main_table,
            "step {i}: the 64-bit form"
        );
        assert_eq!(thread_counts(), after, "step {i}: the 64-bit form");
    }
}

/// ★ The laptop gate: every table of a windowed build in the block's
/// configuration under G-pack — the streamed chunks written packed by their
/// jobs, the finish's tables as it generates them — has the bytes the
/// whole-run build's 64-bit table packs to, and its words; on four programs,
/// at chunk sizes that stream every table and split KECCAK_RND, twice (the
/// second build writes every kind at the widths the first learned).
#[test]
fn every_table_written_packed_is_the_wide_table_packed() {
    let chunked_keccak = MaxRowsConfig {
        keccak_rnd: 48,
        ..MaxRowsConfig::small()
    };
    for (name, max_rows) in [
        ("sub", MaxRowsConfig::small()),
        ("all_instructions_64", MaxRowsConfig::small()),
        ("all_instructions_64", MaxRowsConfig::uniform(4)),
        ("test_keccak", chunked_keccak.clone()),
        ("test_keccak_multi", chunked_keccak),
    ] {
        let (program, logs) = run(name);
        let reference = whole(&program, &logs, &max_rows);
        for (pass, window) in [(0, 7), (1, 33)] {
            let before = thread_counts();
            let got = gpack_build(&program, &logs, &max_rows, window);
            let after = thread_counts();
            let (bad, packed) = misses(&reference, &got);
            assert!(bad.is_empty(), "{name}/{window}: {bad:?}");
            assert!(packed > 0, "{name}/{window}: nothing packed");
            // Streamed chunks written packed on this thread, after the first
            // build taught their kinds' widths.
            if pass == 1 && logs.len() > 2 * max_rows.cpu {
                assert!(
                    after[0] + after[1] + after[2] > before[0] + before[1] + before[2],
                    "{name}/{window}: no chunk was written packed"
                );
            }
        }
    }
}

/// ★ The gate is load-bearing: the streamed chunks written with one column a
/// width too wide (the same words, other bytes) or one column's bytes shifted
/// fail its comparison — the first on the bytes alone, the second on the
/// words.
#[test]
fn a_column_at_the_wrong_width_or_offset_fails_the_gate() {
    let max_rows = MaxRowsConfig::small();
    let (program, logs) = run("all_instructions_64");
    let reference = whole(&program, &logs, &max_rows);
    // Every streamed kind's widths learned, so its chunks are written packed.
    let clean = gpack_build(&program, &logs, &max_rows, 33);
    assert!(misses(&reference, &clean).0.is_empty());
    for m in [Mutation::WidenColumn(0), Mutation::ShiftColumn(0)] {
        mutation::set(Some(m));
        let bent = gpack_build(&program, &logs, &max_rows, 33);
        mutation::set(None);
        let (bad, _) = misses(&reference, &bent);
        let (bytes, words) = (
            bad.iter().any(|(_, k)| *k == Miss::Bytes),
            bad.iter().any(|(_, k)| *k == Miss::Words),
        );
        match m {
            Mutation::WidenColumn(_) => assert!(bytes && !words, "{m:?}: {bad:?}"),
            Mutation::ShiftColumn(_) => assert!(words, "{m:?}: {bad:?}"),
        }
    }
}

/// ★ The block's phase-A stream builds the same packed traces, and precommits
/// the same instances, with G-pack on (the generators write each chunk packed,
/// the finish each table) or off (each built at 8 bytes a cell and packed),
/// with generators ahead of the committers or the committers generating.
#[test]
fn the_stream_builds_the_same_packed_traces_with_gpack_on_or_off() {
    use crate::block::{stream_config, stream_spill_for_test};
    let opts = crate::lfm::proof::block_base_options();
    let chunked_keccak = MaxRowsConfig {
        keccak_rnd: 48,
        ..MaxRowsConfig::small()
    };
    for (name, max_rows) in [
        ("all_instructions_64", MaxRowsConfig::small()),
        ("test_keccak_multi", chunked_keccak),
    ] {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        for (committers, generators, finish_in_a) in [(2, 3, true), (3, 0, false)] {
            let config = stream_config(committers, generators, finish_in_a);
            let stream = |gpack: bool| {
                let (traces, mut names, _) = stream_spill_for_test(
                    &program,
                    &opts,
                    &max_rows,
                    config.with_gpack(gpack),
                    None,
                )
                .expect("the stream");
                names.sort();
                (traces, names)
            };
            let (off, off_names) = stream(false);
            let (on, on_names) = stream(true);
            let what = format!("{name}/{committers}/{generators}/{finish_in_a}");
            assert_eq!(off_names, on_names, "{what}: the instances precommitted");
            let (bad, packed) = misses(&off, &on);
            assert!(bad.is_empty(), "{what}: {bad:?}");
            assert!(packed > 0, "{what}: nothing packed");
            for ((n, a), (_, b)) in tables(&off).into_iter().zip(tables(&on)) {
                assert_eq!(
                    a.narrow_main().map(NarrowMain::digest),
                    b.narrow_main().map(NarrowMain::digest),
                    "{what}: {n} packed alike"
                );
            }
        }
    }
}

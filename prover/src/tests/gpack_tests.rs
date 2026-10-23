//! G-pack (`tables::gpack`): a table written packed as it is generated is the
//! 64-bit table packed, byte for byte — on every path the writer takes, and for
//! every table of a windowed build in the WHIR block's configuration against
//! the same build without G-pack — and the comparison that says so catches a
//! column written at the wrong width or offset.

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;
use multilinear::narrow::NarrowColumns;
use stark::narrow::mutation::{self, Mutation};
use stark::trace::TraceTable;

use crate::tables::MaxRowsConfig;
use crate::tables::gpack::{TraceForm, WidthHint, generate_main, thread_counts};
use crate::tables::trace_builder::{Traces, WindowedTraceBuilder};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
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

/// A windowed build of `logs` in the WHIR block's configuration — KECCAK_RND
/// built as its tables of `keccak_rnd_rows` rows at the finish (0: one table),
/// the streamed ops dropped, the finish's tables packed as they are generated —
/// over windows of `window` cycles, each streamed chunk packed. With G-pack,
/// each chunk is written packed by its job (on this thread, or on the pool
/// with `parallel`) and the finish writes its tables packed; without, each is
/// built at 8 bytes a cell and packed, as before.
#[allow(clippy::too_many_arguments)]
fn block_build(
    program: &Elf,
    input: &[u8],
    logs: &[Log],
    max_rows: &MaxRowsConfig,
    window: usize,
    keccak_rnd_rows: usize,
    gpack: bool,
    parallel: bool,
) -> Traces {
    use rayon::prelude::*;
    let mut builder = WindowedTraceBuilder::new(program, input, max_rows).expect("the builder");
    if keccak_rnd_rows > 0 {
        builder = builder
            .keccak_rnd_chunks_at_finish(keccak_rnd_rows)
            .expect("before any window");
    }
    builder = builder
        .drop_streamed_ops()
        .expect("before any window")
        .pack_finished_tables();
    if gpack {
        builder = builder.generate_packed();
    }
    let form = if gpack {
        TraceForm::Narrow
    } else {
        TraceForm::Wide
    };
    let generate = |job: crate::tables::trace_builder::ChunkJob| {
        let mut chunk = job.generate_as(form);
        chunk.trace.pack_main_narrow();
        chunk
    };
    let body = logs.len() - 1;
    let cut = body - body % window;
    let mut chunks = Vec::new();
    for w in logs[..cut].chunks(window) {
        let jobs = builder.push_jobs(w).expect("a window");
        if parallel {
            chunks.par_extend(jobs.into_par_iter().map(generate));
        } else {
            chunks.extend(jobs.into_iter().map(generate));
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

/// A 64-bit table's raw words, row-major.
fn raw_words(t: &Table) -> Vec<u64> {
    t.main_table
        .row_major_data()
        .iter()
        .map(|v| *v.value())
        .collect()
}

/// A table's raw words, row-major: widened when it is packed.
fn words(t: &Table) -> Vec<u64> {
    match t.narrow_main() {
        Some(packed) => {
            let (rows, cols) = (packed.rows(), packed.cols());
            let mut words = vec![0u64; rows * cols];
            for c in 0..cols {
                for (r, w) in packed.column(c).into_iter().enumerate() {
                    words[r * cols + c] = w;
                }
            }
            words
        }
        None => raw_words(t),
    }
}

/// Each table's name and digest: its packed form's (rows, widths, bytes), or,
/// for a table not packed, its words'.
fn table_digests(t: &Traces) -> Vec<(String, String)> {
    use rayon::prelude::*;
    tables(t)
        .into_par_iter()
        .map(|(name, table)| {
            let mut h = blake3::Hasher::new();
            let digest = match table.narrow_main() {
                Some(packed) => {
                    h.update(&(packed.rows() as u64).to_le_bytes());
                    h.update(packed.widths());
                    h.update(packed.data());
                    format!("packed {}", &h.finalize().to_hex()[..32])
                }
                None => {
                    h.update(&(table.main_table.width as u64).to_le_bytes());
                    for w in raw_words(table) {
                        h.update(&w.to_le_bytes());
                    }
                    format!("words {}", &h.finalize().to_hex()[..32])
                }
            };
            (name, digest)
        })
        .collect()
}

/// How a table differs from the reference's.
#[derive(Debug, PartialEq, Eq)]
enum Miss {
    /// Another table count, width or height.
    Shape,
    /// Other packed bytes (rows, widths and every byte), or packed in one
    /// build and not the other.
    Bytes,
    /// Other words.
    Words,
}

/// ★ The gate's comparison: every table of `got` has the shape, the words and
/// the packed bytes (or the unpacked form) of `reference`'s; and how many of
/// `got`'s tables are packed.
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
        packed += usize::from(b.narrow_main().is_some());
        if a.narrow_main() != b.narrow_main() {
            out.push((name.clone(), Miss::Bytes));
        }
        if words(a) != words(b) {
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

/// ★ Every path `generate_main!` takes gives the 64-bit table packed
/// (`NarrowColumns::pack_row_major`): no widths yet (built wide, then packed,
/// the widths learned), the widths right (written packed as it is), a word
/// wider than its column (written again at the widths the trace needs),
/// columns narrower than learned (narrowed in place). `TraceForm::Wide` gives
/// the 64-bit table.
#[test]
fn every_path_of_the_writer_gives_the_wide_table_packed() {
    static HINT: WidthHint = WidthHint::new();
    let rows = 1000;
    let build = |form, tops| generate_main!(form, &HINT, rows, 5, |t| synthetic(t, rows, tops));
    // (tops, the path: as written, narrowed, written again, wide then packed).
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
        let want = NarrowColumns::pack_row_major(&words(&wide), 5);
        assert_eq!(
            got.narrow_main(),
            want.as_ref(),
            "step {i}: the packed bytes"
        );
        assert_eq!(words(&got), words(&wide), "step {i}: the words");
        let plain = build(TraceForm::Wide, tops);
        assert!(plain.narrow_main().is_none());
        assert_eq!(
            plain.main_table, wide.main_table,
            "step {i}: the 64-bit form"
        );
        assert_eq!(thread_counts(), after, "step {i}: the 64-bit form");
    }
}

/// ★ The laptop gate: every table of a windowed build in the WHIR block's
/// configuration under G-pack — the streamed chunks written packed by their
/// jobs, the finish's tables as it generates them — has the packed bytes and
/// the words of the same build without G-pack; on four programs, at chunk
/// sizes that stream every table and cut KECCAK_RND, twice (the second build
/// writes every kind at the widths the first learned).
#[test]
fn every_table_written_packed_is_the_wide_table_packed() {
    for (name, max_rows, kr_rows) in [
        ("sub", MaxRowsConfig::small(), 0),
        ("all_instructions_64", MaxRowsConfig::small(), 0),
        ("all_instructions_64", MaxRowsConfig::uniform(4), 0),
        ("test_keccak", MaxRowsConfig::small(), 64),
        ("test_keccak_multi", MaxRowsConfig::small(), 64),
    ] {
        let (program, logs) = run(name);
        let reference = block_build(&program, &[], &logs, &max_rows, 7, kr_rows, false, false);
        for pass in 0..2 {
            let before = thread_counts();
            let got = block_build(&program, &[], &logs, &max_rows, 33, kr_rows, true, false);
            let after = thread_counts();
            let (bad, packed) = misses(&reference, &got);
            assert!(bad.is_empty(), "{name}/{pass}: {bad:?}");
            assert!(packed > 0, "{name}/{pass}: nothing packed");
            if pass == 1 && logs.len() > 2 * max_rows.cpu {
                assert!(
                    after[0] + after[1] + after[2] > before[0] + before[1] + before[2],
                    "{name}: no chunk was written packed"
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
    let reference = block_build(&program, &[], &logs, &max_rows, 33, 0, false, false);
    // Every streamed kind's widths learned, so its chunks are written packed.
    let clean = block_build(&program, &[], &logs, &max_rows, 33, 0, true, false);
    assert!(misses(&reference, &clean).0.is_empty());
    for m in [Mutation::WidenColumn(0), Mutation::ShiftColumn(0)] {
        mutation::set(Some(m));
        let bent = block_build(&program, &[], &logs, &max_rows, 33, 0, true, false);
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

/// The box test's builds and digests on small programs: the build with G-pack
/// and the build without give every table the same digest.
#[test]
fn the_real_block_comparison_holds_on_small_programs() {
    for (name, max_rows, kr_rows) in [
        ("all_instructions_64", MaxRowsConfig::small(), 0),
        ("test_keccak_multi", MaxRowsConfig::small(), 64),
    ] {
        let (program, logs) = run(name);
        let want = table_digests(&block_build(
            &program,
            &[],
            &logs,
            &max_rows,
            33,
            kr_rows,
            false,
            true,
        ));
        assert!(want.iter().any(|(_, d)| d.starts_with("packed")));
        for _ in 0..2 {
            let got = table_digests(&block_build(
                &program,
                &[],
                &logs,
                &max_rows,
                33,
                kr_rows,
                true,
                true,
            ));
            assert_eq!(got, want, "{name}");
        }
    }
}

/// ★ Box only (`--ignored`): on a real block (`BLOCK_WHIR_ELF`,
/// `BLOCK_WHIR_INPUT`) at the WHIR block's caps and windows, every table of the
/// windowed build under G-pack has the packed bytes of the build without it —
/// twice in one process, the second build writing every kind at the widths
/// the first learned. Prints `GPACK` lines.
#[test]
#[ignore = "a real block: box only (BLOCK_WHIR_ELF, BLOCK_WHIR_INPUT)"]
fn every_table_of_a_real_block_written_packed_is_the_wide_table_packed() {
    use crate::tables::gpack::{counts, reset_counts};
    let elf = std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF")).unwrap();
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).unwrap();
    let program = Elf::load(&elf).expect("the ELF loads");
    let options = crate::block_whir::BlockOptions::production();
    let window = 1usize << options.window_log2.expect("the block streams");
    let kr_rows = 1usize << options.keccak_rnd_rows_log2;
    let logs = Executor::new(&program, input.clone())
        .expect("the executor starts")
        .run()
        .expect("the block runs")
        .logs;
    let max_rows = &options.max_rows;
    let t = std::time::Instant::now();
    let want = table_digests(&block_build(
        &program, &input, &logs, max_rows, window, kr_rows, false, true,
    ));
    let packed = want.iter().filter(|(_, d)| d.starts_with("packed")).count();
    println!(
        "GPACK off: {} tables ({packed} packed) in {:.2} s",
        want.len(),
        t.elapsed().as_secs_f64()
    );
    let mut equal = 0;
    for pass in 0..2 {
        reset_counts();
        let t = std::time::Instant::now();
        let got = table_digests(&block_build(
            &program, &input, &logs, max_rows, window, kr_rows, true, true,
        ));
        let secs = t.elapsed().as_secs_f64();
        let [direct, narrowed, again, wide] = counts();
        let differ: Vec<&String> = want
            .iter()
            .zip(&got)
            .filter(|(a, b)| a != b)
            .map(|(a, _)| &a.0)
            .collect();
        println!(
            "GPACK on, build {pass}: {} tables in {secs:.2} s · {direct} written packed, {narrowed} \
             narrowed after, {again} written again, {wide} built wide then packed · {} differ {differ:?}",
            got.len(),
            differ.len()
        );
        if got.len() == want.len() && differ.is_empty() {
            equal += 1;
        }
    }
    println!("GPACK RESULT: {equal}/2 G-pack builds have every table's packed bytes");
    assert_eq!(equal, 2);
}

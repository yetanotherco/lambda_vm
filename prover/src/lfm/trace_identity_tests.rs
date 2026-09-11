//! The gate on `trace`'s row walk: a chip's trace is a pure function of the
//! records, so **which** walk fills it may not change one cell of it.
//!
//! The two sides are [`Walk::Parallel`] — the rows chunked across rayon's pool,
//! what production runs — and [`Walk::SerialReference`], the two plain row loops
//! `trace.rs` had before. They are driven through `build_traces_walked`, so both
//! sides go through the *production* fill closures and no others: a test that
//! re-spelled the closures would gate its own copy instead of the machine.
//!
//! ## What is checked
//!
//! 1. **Byte identity**, cell for cell, over every table in `LfmTraces` — the
//!    chip traces the walk fills, and also the keccak-family ones it does not,
//!    since a regression there would be just as fatal and costs nothing to see.
//! 2. **Coverage**: the cases must, between them, give every chip with a
//!    non-empty fill at least one real row. Without this the identity assertion
//!    passes vacuously on a chip nothing exercised.
//! 3. **The census** — total cells and instruction count — computed on each side
//!    and asserted equal, then printed.
//!
//! ## What is NOT checked
//!
//! Proof bytes. Grinding picks a nonce that is not reproducible run to run
//! (`grinding-nonce-nondeterminism`), so comparing proofs would gate the wrong
//! thing; the trace is the object the walk produces and the trace is what this
//! compares.
//!
//! Under `--no-default-features` both walks are serial (`fill_rows` has a
//! non-rayon twin, since the crate still has to build there), so the gate is
//! true by construction on that arm and says nothing. It is the default build it
//! is written for.
//!
//! The scale probe at the bottom is a measurement, not a gate: it drives the
//! `LFM_HASH` filler at a production-shaped height so the walk's speedup has a
//! number attached. It asserts identity too, because that is free.

use std::time::Instant;

use stark::trace::TraceTable;

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

use super::chips::hash;
use super::compiler::{ColumnGroup, LfmProgram};
use super::executor::{HashRow, execute};
use super::hash::HasherKind;
use super::instr::{HashMode, Instr};
use super::keccak_host::pack_stream;
use super::trace::{LfmTraces, Walk, build_traces_walked, chip_trace, fill_hash_row};
use super::word::{LfmWord, base_word};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// One thing to fill traces for: a compiled program, the arenas it reads, and
/// the hasher both the executor and the trace builder are handed.
struct Case {
    name: &'static str,
    program: LfmProgram,
    arenas: Vec<Vec<LfmWord>>,
    hasher: HasherKind,
}

/// The message the sponge cases run over: byte `i` is `37i + 11`, the generator
/// `crypto`'s own chain KATs use.
fn message(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
        .collect()
}

fn sponge_arenas(msg: &[u8]) -> Vec<Vec<LfmWord>> {
    vec![pack_stream(msg).into_iter().map(base_word).collect()]
}

fn fri_arenas(inner: &super::fixture::FriToyProof) -> Vec<Vec<LfmWord>> {
    vec![inner.commitments.clone(), inner.openings.clone()]
}

/// The cases, chosen for coverage rather than for size.
///
/// `FriToyV0` is a real verification program over real FRI data and is what
/// gives the value chips their rows; it appears twice because the `LFM_HASH`
/// fill DISPATCHES on the hasher, so RPX (the production pin, and the most
/// expensive filler — twelve lanes of `x^{1/7}` per round) and BLAKE3 (the
/// socket filler, the one that reads its domain back off the row) are two
/// different bodies under test. The two sponge programs are the smallest things
/// that give `LFM_BLAKE3` and `LFM_KECCAK` rows.
fn cases() -> Vec<Case> {
    let msg = message(202);
    vec![
        Case {
            name: "FriToyV0 @ RPX",
            program: super::programs::fri_toy_program(),
            arenas: fri_arenas(&super::fixture::fixture_prove_with_hasher(HasherKind::Rpx)),
            hasher: HasherKind::Rpx,
        },
        Case {
            name: "FriToyV0 @ BLAKE3",
            program: super::programs::fri_toy_program(),
            arenas: fri_arenas(&super::fixture::fixture_prove_with_hasher(
                HasherKind::Blake3,
            )),
            hasher: HasherKind::Blake3,
        },
        Case {
            name: "Blake3Chain(202)",
            program: super::programs::blake3_sponge_program(202),
            arenas: sponge_arenas(&msg),
            hasher: HasherKind::Test,
        },
        Case {
            name: "Keccak256(202)",
            program: super::programs::keccak_sponge_program(202),
            arenas: sponge_arenas(&msg),
            hasher: HasherKind::Test,
        },
    ]
}

/// Every table in an `LfmTraces`, paired with the chip's name, in one flat list
/// so the comparison walks both sides in lockstep and can say which chip moved.
fn tables(t: &LfmTraces) -> Vec<(String, &TraceTable<F, E>)> {
    let mut out: Vec<(String, &TraceTable<F, E>)> = vec![
        ("LFM_CONST".into(), &t.const_),
        ("LFM_BALU".into(), &t.balu),
        ("LFM_XALU".into(), &t.xalu),
        ("LFM_SELECT".into(), &t.select),
        ("LFM_BITDEC".into(), &t.bitdec),
        ("LFM_HASH".into(), &t.hash),
        ("LFM_KECCAK".into(), &t.keccak),
        ("LFM_LANES".into(), &t.lanes),
        ("LFM_HINT".into(), &t.hint),
        ("LFM_PUBLIC".into(), &t.public),
        ("LFM_RANGE".into(), &t.range),
        ("KECCAK_RC".into(), &t.keccak_rc),
        ("BITWISE".into(), &t.bitwise),
    ];
    for (c, tr) in t.blake3.iter().enumerate() {
        out.push((format!("LFM_BLAKE3[{c}]"), tr));
    }
    for (c, tr) in t.keccak_rnd.iter().enumerate() {
        out.push((format!("KECCAK_RND[{c}]"), tr));
    }
    out
}

/// Total main-trace cells across every table — the census figure.
///
/// Counted off the backing store rather than `Table`'s `width`/`height`, which
/// are fields on one `disk-spill` arm and methods on the other.
fn cells(t: &LfmTraces) -> u64 {
    tables(t)
        .iter()
        .map(|(_, tr)| tr.main_table.row_major_data().len() as u64)
        .sum()
}

/// The share of [`cells`] the row walk actually fills.
///
/// The three keccak-family tables do not go through `chip_trace` — `BITWISE`
/// alone is a fixed 2^20 rows whatever the program is — so at fixture scale they
/// are most of the census and all of the wall. Printing the split is what keeps
/// the timing column below from being read as a fill measurement; the fill
/// measurement is the scale probe.
fn walked_cells(t: &LfmTraces) -> u64 {
    tables(t)
        .iter()
        .filter(|(name, _)| name.starts_with("LFM_"))
        .map(|(_, tr)| tr.main_table.row_major_data().len() as u64)
        .sum()
}

/// Asserts two trace sets are the same object, and says where they first differ
/// if they are not.
fn assert_identical(case: &str, a: &LfmTraces, b: &LfmTraces) {
    let (ta, tb) = (tables(a), tables(b));
    assert_eq!(
        ta.len(),
        tb.len(),
        "{case}: the two walks produced different table counts"
    );
    for ((name, x), (other, y)) in ta.iter().zip(&tb) {
        assert_eq!(name, other, "{case}: table order diverged");
        let width = x.num_main_columns;
        assert_eq!(
            width, y.num_main_columns,
            "{case}/{name}: the column count moved"
        );
        let (dx, dy) = (x.main_table.row_major_data(), y.main_table.row_major_data());
        assert_eq!(dx.len(), dy.len(), "{case}/{name}: the row count moved");
        if dx == dy {
            continue;
        }
        let at = dx
            .iter()
            .zip(dy)
            .position(|(p, q)| p != q)
            .expect("the slices differ, so some index differs");
        panic!(
            "{case}/{name}: the parallel walk and the serial reference disagree \
             at row {}, column {} — {:?} vs {:?}",
            at / width,
            at % width,
            dx[at],
            dy[at]
        );
    }
}

/// ★ The gate. Both walks, every case, cell for cell.
#[test]
fn the_parallel_row_walk_is_byte_identical_to_the_serial_reference() {
    // Chips whose fill closure writes something. `LFM_CONST` and `LFM_RANGE`
    // take `|_, _| {}` — their rows are preprocessed data only — so there is
    // nothing for a case to "cover" on them.
    let mut covered: Vec<&str> = Vec::new();

    for case in cases() {
        let exec = execute(&case.program, &case.arenas, &case.hasher)
            .unwrap_or_else(|e| panic!("{}: the case must execute: {e:?}", case.name));

        let t0 = Instant::now();
        let par = build_traces_walked(&case.program, &exec.records, case.hasher, Walk::Parallel);
        let t_par = t0.elapsed();

        let t1 = Instant::now();
        let seq = build_traces_walked(
            &case.program,
            &exec.records,
            case.hasher,
            Walk::SerialReference,
        );
        let t_seq = t1.elapsed();

        assert_identical(case.name, &par, &seq);

        // The census, both ways. Byte identity already implies this; it is
        // asserted and printed separately because the census is the number the
        // campaign quotes, and a silent change to it is the failure mode that
        // would otherwise be found on a box rather than here.
        let instrs = case.program.instrs.len();
        assert_eq!(
            cells(&par),
            cells(&seq),
            "{}: the census moved with the walk",
            case.name
        );
        assert_eq!(
            walked_cells(&par),
            walked_cells(&seq),
            "{}: the walked share of the census moved",
            case.name
        );
        println!(
            "{:<18} {:>3}t  cells {:>10} (walked {:>9})  instrs {:>5}  \
             build: parallel {:>7.1} ms  serial {:>7.1} ms",
            case.name,
            threads(),
            cells(&par),
            walked_cells(&par),
            instrs,
            t_par.as_secs_f64() * 1e3,
            t_seq.as_secs_f64() * 1e3,
        );

        for (name, group) in named_groups(&case.program) {
            if group.real_rows > 0 && !covered.contains(&name) {
                covered.push(name);
            }
        }
    }

    // Vacuity guard: without this the identity assertion above is satisfied by
    // two empty traces.
    for chip in FILLED_CHIPS {
        assert!(
            covered.contains(chip),
            "no case gave {chip} a real row, so its fill closure was never run \
             on either walk — the gate is vacuous on it. Add a case that \
             exercises it rather than dropping it from this list."
        );
    }
}

/// The chips whose fill closure writes value columns. `LFM_CONST` and
/// `LFM_RANGE` are absent on purpose: their fills are `|_, _| {}`.
const FILLED_CHIPS: &[&str] = &[
    "LFM_BALU",
    "LFM_XALU",
    "LFM_SELECT",
    "LFM_BITDEC",
    "LFM_HASH",
    "LFM_KECCAK",
    "LFM_BLAKE3",
    "LFM_LANES",
    "LFM_HINT",
    "LFM_PUBLIC",
];

fn named_groups(p: &LfmProgram) -> Vec<(&'static str, &ColumnGroup)> {
    let g = &p.groups;
    vec![
        ("LFM_CONST", &g.const_),
        ("LFM_BALU", &g.balu),
        ("LFM_XALU", &g.xalu),
        ("LFM_SELECT", &g.select),
        ("LFM_BITDEC", &g.bitdec),
        ("LFM_HASH", &g.hash),
        ("LFM_KECCAK", &g.keccak),
        ("LFM_BLAKE3", &g.blake3),
        ("LFM_LANES", &g.lanes),
        ("LFM_HINT", &g.hint),
        ("LFM_PUBLIC", &g.public),
    ]
}

#[cfg(feature = "parallel")]
fn threads() -> usize {
    rayon::current_num_threads()
}

#[cfg(not(feature = "parallel"))]
fn threads() -> usize {
    1
}

// =========================================================================
// The scale probe
// =========================================================================

/// log2 of the row count the probe fills. 2^18 rows of RPX witness is 0.64 GiB
/// of trace, which a laptop can hold twice; the box arm can raise it to the
/// production 2^20 through the environment without a rebuild.
fn probe_log_rows() -> u32 {
    std::env::var("LFM_TRACE_PROBE_LOG_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(18)
}

/// `group` with its real rows repeated, cyclically, up to `rows`.
///
/// Repetition is sound for a timing probe because the `LFM_HASH` fill reads the
/// row's own record and, under BLAKE3, the row's own mode columns — never a
/// neighbour, never a row ordinal. So a repeated row costs what the row it
/// repeats costs. The resulting group is not a provable trace and is never
/// proved; it exists to be filled.
fn repeated(group: &ColumnGroup, rows: usize) -> ColumnGroup {
    let src = group.real_rows;
    assert!(src > 0, "nothing to repeat");
    let mut data = Vec::with_capacity(rows * group.width);
    for row in 0..rows {
        let at = (row % src) * group.width;
        data.extend_from_slice(&group.data[at..at + group.width]);
    }
    ColumnGroup {
        width: group.width,
        real_rows: rows,
        padded_rows: rows,
        data,
    }
}

fn hash_modes(p: &LfmProgram) -> Vec<HashMode> {
    p.instrs
        .iter()
        .filter_map(|i| match i {
            Instr::Hash { mode, .. } => Some(*mode),
            _ => None,
        })
        .collect()
}

/// The measurement the handback quotes: `LFM_HASH` at a production-shaped
/// height, filled both ways, on this machine's pool.
///
/// It drives `chip_trace` directly rather than a program, because no cheap
/// program has a million hash rows — the one that does is the recursion leaf,
/// and building one is a box job. The filler is the production `fill_hash_row`,
/// so the work per row is the real work.
#[test]
fn the_row_walk_scales_on_the_hash_chip() {
    let hasher = HasherKind::Rpx;
    let program = super::programs::fri_toy_program();
    let inner = super::fixture::fixture_prove_with_hasher(hasher);
    let exec = execute(&program, &fri_arenas(&inner), &hasher).expect("the fixture must execute");

    let rows = 1usize << probe_log_rows();
    let group = repeated(&program.groups.hash, rows);
    let modes = hash_modes(&program);
    let src = program.groups.hash.real_rows;
    assert_eq!(
        modes.len(),
        src,
        "one mode per hash row, or the repetition would read the wrong one"
    );
    let records: &Vec<HashRow> = &exec.records.hash;
    let num_columns = hash::num_columns(hasher);

    let fill = |row: usize, out: &mut [FE]| {
        let at = row % src;
        fill_hash_row(hasher, &records[at], modes[at], out)
    };

    let t0 = Instant::now();
    let par = chip_trace(Walk::Parallel, &group, num_columns, fill);
    let t_par = t0.elapsed();

    let t1 = Instant::now();
    let seq = chip_trace(Walk::SerialReference, &group, num_columns, fill);
    let t_seq = t1.elapsed();

    assert_eq!(
        par.main_table.row_major_data(),
        seq.main_table.row_major_data(),
        "the probe's two walks must agree too"
    );

    println!(
        "LFM_HASH @ RPX  {rows} rows x {num_columns} cols ({:.2} GiB)  {} threads  \
         parallel {:.3} s  serial {:.3} s  speedup {:.2}x",
        (rows * num_columns * 8) as f64 / (1u64 << 30) as f64,
        threads(),
        t_par.as_secs_f64(),
        t_seq.as_secs_f64(),
        t_seq.as_secs_f64() / t_par.as_secs_f64(),
    );
}

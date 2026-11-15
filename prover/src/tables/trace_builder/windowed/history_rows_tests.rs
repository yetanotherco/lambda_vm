//! MEMW_A's rows derived from the executor's history
//! ([`super::history_rows`]) against the walk's own, row for row, on
//! record-mode runs of every ECALL and misalignment program, the Rust guests
//! and an ethrex block, in both MEMW_A forms; and the rules the derivation
//! stands on, each broken once.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use executor::elf::Elf;
use executor::vm::fast::FastExecutor;
use executor::vm::history::{NO_ACCESS, WindowHistory};
use executor::vm::logs::Log;
use multilinear::records::{ALIGNED_ECALL, AlignedRef};

use super::WindowedTraceBuilder;
use super::history_rows::{RefWindow, aligned_refs_of_rows, aligned_rows_from_refs};
use super::history_sides_tests::rust_elf;
use super::parallel::{ExecWindow, walk_windows_from_history};
use crate::tables::MaxRowsConfig;
use crate::tables::decode::DecodeTable;
use crate::tables::memw_aligned::{AlignedRow, MemwAlignedOrder};
use crate::tables::register::RegisterFormat;
use crate::tests::windowed_builder_tests::run;

/// The asm programs: every ECALL the walk handles but HINT, misaligned and
/// page-crossing accesses (the record-mode tests' list).
const PROGRAMS: [&str; 14] = [
    "all_instructions_64",
    "all_loadstore_32",
    "test_keccak_multi",
    "test_keccak256",
    "test_blake3",
    "test_blake3_absorb",
    "test_ecsm",
    "test_ecrec",
    "test_commit_split",
    "test_copy_ecalls",
    "test_copy_overlaps",
    "misalign_ld",
    "misalign_sd",
    "misalign_sh",
];

/// A walked window: its logs and history, and the MEMW_A rows its walk made.
pub(super) struct Walked {
    pub(super) first: u64,
    pub(super) logs: Arc<Vec<Log>>,
    pub(super) history: Arc<WindowHistory>,
    pub(super) rows: Vec<AlignedRow>,
}

/// The windows the executor sent, by first cycle.
type Sent = Mutex<BTreeMap<u64, (Arc<Vec<Log>>, Arc<WindowHistory>)>>;

/// `program` with `input` run in history mode in windows of `window` cycles
/// and walked from the history under the in-row register format in `order`
/// on two walkers: every window the walkers walked (the run's last, the
/// finish's, is not), in order, and the program's DECODE rows.
pub(super) fn walked(
    program: &Elf,
    input: &[u8],
    window: usize,
    order: MemwAlignedOrder,
) -> (Vec<Walked>, Arc<DecodeTable>) {
    let mut builder = WindowedTraceBuilder::new(program, input, &MaxRowsConfig::small())
        .and_then(|b| b.register_format(RegisterFormat::InRow))
        .and_then(|b| b.memw_aligned_order(order))
        .expect("the builder");
    let decode = Arc::clone(&builder.decode);
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let sent: Sent = Mutex::new(BTreeMap::new());
    let mut windows = Vec::new();
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
        walk_windows_from_history(&mut builder, rx, 2, 0, |accumulator, walked, _| {
            let first = walked.first as u64;
            let rows = walked.walk.memw.aligned.clone();
            let (logs, history) = sent
                .lock()
                .expect("the windows")
                .remove(&first)
                .expect("the window was sent");
            windows.push(Walked {
                first,
                logs,
                history,
                rows,
            });
            let _ = accumulator.absorb(walked);
            Ok(())
        })
        .expect("the walk");
    });
    windows.sort_by_key(|w| w.first);
    (windows, decode)
}

/// What the derivation reads of a window, and its refs: the walk's rows as
/// refs and whole rows.
pub(super) fn ref_window(w: &Walked) -> (RefWindow, Vec<AlignedRef>) {
    let (refs, whole) =
        aligned_refs_of_rows(&w.rows, w.first, &w.history).expect("the window's refs");
    let window = RefWindow {
        first: w.first,
        logs: Arc::clone(&w.logs),
        history: Arc::clone(&w.history),
        whole,
    };
    (window, refs)
}

/// What a run's rows hold: rows, ECALL rows, loads, stores, loads that fill
/// past their width with ones, rows whose access's timestamps differ.
#[derive(Default, Debug)]
struct Census {
    rows: usize,
    ecall: usize,
    loads: usize,
    stores: usize,
    sign_filled: usize,
    mixed: usize,
}

/// Every window's rows derived from its refs, against the walk's; the
/// census of what they hold.
fn check(windows: &[Walked], decode: &DecodeTable, order: MemwAlignedOrder, what: &str) -> Census {
    let mut census = Census::default();
    for w in windows {
        let (window, refs) = ref_window(w);
        assert_eq!(refs.len(), w.rows.len(), "{what}: one ref a row");
        let got = aligned_rows_from_refs(&[(&refs, &window)], decode, order)
            .unwrap_or_else(|e| panic!("{what}, window from {}: {e}", w.first));
        if let Some(at) = (0..got.len()).find(|&i| got[i] != w.rows[i]) {
            panic!(
                "{what}, window from {}: row {at} ({:?}): derived {:?}, walked {:?}",
                w.first, refs[at], got[at], w.rows[at]
            );
        }
        for (r, row) in refs.iter().zip(&w.rows) {
            census.rows += 1;
            if r.cycle & ALIGNED_ECALL != 0 {
                census.ecall += 1;
                continue;
            }
            let a = w.history.accesses[r.at as usize];
            census.mixed += usize::from(a.mixed != NO_ACCESS);
            if row.is_read() {
                census.loads += 1;
                let value = row.value();
                census.sign_filled +=
                    usize::from(row.width() < 8 && value[row.width() as usize] == 0xFF);
            } else {
                census.stores += 1;
            }
        }
    }
    census
}

/// Every walked MEMW_A row derived from the history is the walk's: 14 asm
/// programs × windows 1, 7, 4096 × both MEMW_A forms. ECALL rows, loads,
/// stores, sign fills and mixed timestamps all occur.
#[test]
fn the_derived_memw_a_rows_are_the_walks() {
    let mut total = Census::default();
    for name in PROGRAMS {
        let (program, _) = run(name);
        for window in [1usize, 7, 4096] {
            for order in [MemwAlignedOrder::Limbs, MemwAlignedOrder::Lt] {
                let (windows, decode) = walked(&program, &[], window, order);
                let what = format!("{name}, window {window}, {order:?}");
                let c = check(&windows, &decode, order, &what);
                total.rows += c.rows;
                total.ecall += c.ecall;
                total.loads += c.loads;
                total.stores += c.stores;
                total.sign_filled += c.sign_filled;
                total.mixed += c.mixed;
            }
        }
    }
    println!("{total:?}");
    assert!(total.ecall > 0, "{total:?}");
    assert!(total.loads > 0 && total.stores > 0, "{total:?}");
    assert!(total.sign_filled > 0, "{total:?}");
    assert!(total.mixed > 0, "{total:?}");
}

/// The derivation against the walk on the Rust guests (HINT, DMA, KECCAK)
/// and an ethrex block in the block's form: real loads and stores, ECALLs
/// between them.
#[test]
fn rust_guests_and_ethrex_derive_the_walks_memw_a_rows() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../executor/tests");
    let mut cases: Vec<(String, Elf, Vec<u8>, usize)> = [
        "hint_min",
        "hint_multi",
        "dma_memset_cases",
        "dma_memmove_cases",
        "keccak_precompile",
    ]
    .iter()
    .map(|name| (name.to_string(), rust_elf(name), Vec::new(), 97))
    .collect();
    let input = std::fs::read(fixtures.join("ethrex_empty_block.bin")).expect("the fixture");
    cases.push((
        "ethrex_empty_block".to_string(),
        rust_elf("ethrex"),
        input,
        1 << 16,
    ));
    let mut total = 0usize;
    for (name, program, input, window) in &cases {
        let order = MemwAlignedOrder::Limbs;
        let (windows, decode) = walked(program, input, *window, order);
        let c = check(&windows, &decode, order, name);
        println!("{name}: {c:?}");
        total += c.rows;
    }
    assert!(total > 100_000, "{total} rows");
}

/// Each rule is load-bearing, broken once on a real run: a mixed access read
/// as uniform, a store's old bytes moved, a ref pointed at the next access, a
/// step that does not fit the limbs, an ECALL ref past its window's whole
/// rows, and a cycle outside its window; each is refused or changes the row.
/// And a narrow load's log moved past its width keeps its row (the fill).
#[test]
fn each_rule_of_the_memw_a_derivation_is_load_bearing() {
    let order = MemwAlignedOrder::Limbs;
    let mut seen = [0usize; 7];
    for name in ["all_loadstore_32", "test_keccak_multi", "misalign_sd"] {
        let (program, _) = run(name);
        let (windows, decode) = walked(&program, &[], 7, order);
        for w in &windows {
            let (window, refs) = ref_window(w);
            let derive = |window: &RefWindow, refs: &[AlignedRef]| {
                aligned_rows_from_refs(&[(refs, window)], &decode, order)
            };
            let differs = |got: Result<Vec<AlignedRow>, String>| got.map_or(true, |g| g != w.rows);
            let regular = |i: usize| refs[i].cycle & ALIGNED_ECALL == 0;
            let with_history = |edit: &dyn Fn(&mut WindowHistory)| {
                let mut h = (*w.history).clone();
                edit(&mut h);
                RefWindow {
                    history: Arc::new(h),
                    first: window.first,
                    logs: Arc::clone(&window.logs),
                    whole: window.whole.clone(),
                }
            };
            // The mixed list read: an access of two bytes or more given an
            // entry whose second byte's timestamp differs is no MEMW_A row
            // (a derivation that reads the shared timestamp alone keeps it).
            if let Some(i) = (0..refs.len()).find(|&i| {
                regular(i)
                    && w.rows[i].width() >= 2
                    && w.history.accesses[refs[i].at as usize].mixed == NO_ACCESS
            }) {
                let at = refs[i].at as usize;
                let moved = with_history(&|h| {
                    let ts = h.accesses[at].ts;
                    let mut entry = [ts; 8];
                    entry[1] = ts.wrapping_sub(1);
                    h.mixed.push(entry);
                    h.accesses[at].mixed = h.mixed.len() as u32 - 1;
                });
                assert!(
                    derive(&moved, &refs).is_err(),
                    "{name}: the mixed list ignored"
                );
                seen[0] += 1;
            }
            // A store's old bytes moved.
            if let Some(i) = (0..refs.len()).find(|&i| regular(i) && !w.rows[i].is_read()) {
                let at = refs[i].at as usize;
                let moved = with_history(&|h| h.accesses[at].old ^= 1);
                assert!(differs(derive(&moved, &refs)), "{name}: a store's old");
                seen[1] += 1;
            }
            // A ref pointed at the next access.
            if let Some(i) = (0..refs.len())
                .find(|&i| regular(i) && (refs[i].at as usize) + 1 < w.history.accesses.len())
            {
                let mut moved = refs.clone();
                moved[i].at += 1;
                assert!(differs(derive(&window, &moved)), "{name}: the next access");
                seen[2] += 1;
            }
            // A step that does not fit the limbs: the old timestamp at the
            // row's own.
            if let Some(i) = (0..refs.len())
                .find(|&i| regular(i) && w.history.accesses[refs[i].at as usize].mixed == NO_ACCESS)
            {
                let at = refs[i].at as usize;
                let ts = 4 * u64::from(refs[i].cycle) + 4;
                let moved = with_history(&|h| h.accesses[at].ts = ts as u32);
                assert!(derive(&moved, &refs).is_err(), "{name}: no step");
                seen[3] += 1;
            }
            // An ECALL ref past the whole rows.
            if let Some(i) = (0..refs.len()).find(|&i| !regular(i)) {
                let mut moved = refs.clone();
                moved[i].at = window.whole.len() as u32;
                assert!(
                    derive(&window, &moved).is_err(),
                    "{name}: past the whole rows"
                );
                seen[4] += 1;
            }
            // A narrow load's log with its written value's top bit flipped
            // (past its width): the fill makes the same row.
            if let Some(i) = (0..refs.len())
                .find(|&i| regular(i) && w.rows[i].is_read() && w.rows[i].width() < 8)
            {
                let mut logs = (*window.logs).clone();
                logs[(u64::from(refs[i].cycle) - window.first) as usize].dst_val ^= 1 << 63;
                let edited = RefWindow {
                    logs: Arc::new(logs),
                    first: window.first,
                    history: Arc::clone(&window.history),
                    whole: window.whole.clone(),
                };
                assert_eq!(
                    derive(&edited, &refs).as_deref(),
                    Ok(&w.rows[..]),
                    "{name}: the fill"
                );
                seen[6] += 1;
            }
            // A cycle outside the window.
            if !refs.is_empty() {
                let mut moved = refs.clone();
                moved[0].cycle = (window.first + window.logs.len() as u64) as u32;
                assert!(
                    derive(&window, &moved).is_err(),
                    "{name}: outside its window"
                );
                seen[5] += 1;
            }
        }
    }
    assert!(seen.iter().all(|&n| n > 0), "{seen:?}");
}

/// A whole MEMW_A chunk held as refs, and the walk's rows it stands for.
pub(super) struct RefChunk {
    pub(super) refs: super::Segments<AlignedRef>,
    pub(super) windows: Vec<Arc<RefWindow>>,
    pub(super) want: Vec<AlignedRow>,
}

/// A run's refs cut into whole chunks of `rows` rows across its windows, as
/// the hand-out cuts them: each chunk's parts its windows' ranges.
pub(super) fn chunks(walked: &[Walked], rows: usize) -> Vec<RefChunk> {
    let lists: Vec<(Arc<RefWindow>, Arc<Vec<AlignedRef>>, &[AlignedRow])> = walked
        .iter()
        .map(|w| {
            let (window, refs) = ref_window(w);
            (Arc::new(window), Arc::new(refs), &w.rows[..])
        })
        .collect();
    let mut out = Vec::new();
    let (mut k, mut at) = (0usize, 0usize);
    let total: usize = lists.iter().map(|(_, r, _)| r.len()).sum();
    for _ in 0..total / rows {
        let mut chunk = RefChunk {
            refs: super::Segments { parts: Vec::new() },
            windows: Vec::new(),
            want: Vec::new(),
        };
        let mut want = rows;
        while want > 0 {
            let (window, refs, walked_rows) = &lists[k];
            let take = (refs.len() - at).min(want);
            if take > 0 {
                chunk.refs.parts.push((Arc::clone(refs), at..at + take));
                chunk.windows.push(Arc::clone(window));
                chunk.want.extend_from_slice(&walked_rows[at..at + take]);
            }
            at += take;
            want -= take;
            if at == refs.len() {
                (k, at) = (k + 1, 0);
            }
        }
        out.push(chunk);
    }
    out
}

/// A chunk held as refs digests and fills as the same rows held as rows, and
/// hands the card each part's records: every regular row's access inside
/// its part's accesses, every ECALL row inside its whole rows.
#[test]
fn a_memw_a_chunk_held_as_refs_reads_as_its_rows() {
    use super::card_records::{AlignedRefRecords, CardDecode, RowRecords};
    use multilinear::records::{TableRecords, digest_of};
    let order = MemwAlignedOrder::Limbs;
    let mut checked = 0usize;
    for (name, window, rows) in [
        ("test_keccak_multi", 7usize, 16usize),
        ("test_copy_ecalls", 33, 32),
        ("all_loadstore_32", 7, 64),
    ] {
        let (program, _) = run(name);
        let (walked, decode) = walked(&program, &[], window, order);
        let card = Arc::new(CardDecode::new(Arc::clone(&decode)).expect("DECODE rows"));
        for chunk in chunks(&walked, rows) {
            let held = AlignedRefRecords::new(
                chunk.refs.clone(),
                chunk.windows.clone(),
                Arc::clone(&card),
                order,
            )
            .unwrap_or_else(|_| panic!("{name}: a whole chunk"));
            let as_rows = RowRecords::new(super::Segments::one(chunk.want.clone()))
                .unwrap_or_else(|_| panic!("{name}: rows"));
            assert_eq!(digest_of(&held), digest_of(&as_rows), "{name}");
            assert_eq!(held.host_fill(), as_rows.host_fill(), "{name}");
            let a = held.aligned_refs().expect("refs");
            assert_eq!(a.len(), rows);
            assert!(a.limbs);
            for p in &a.parts {
                for r in p.refs {
                    let (base, n) = if r.cycle & ALIGNED_ECALL != 0 {
                        (p.whole_base, p.whole.len())
                    } else {
                        (p.access_base, p.accesses.len())
                    };
                    assert!(
                        r.at >= base && ((r.at - base) as usize) < n,
                        "{name}: {r:?}"
                    );
                }
            }
            checked += 1;
        }
    }
    assert!(checked >= 3, "{checked} chunks");
}

fn put(out: &mut Vec<u8>, words: &[u64]) {
    words
        .iter()
        .for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
}

/// `v`'s bytes as they lie (plain `#[repr(C)]` records without padding, or
/// whose padding the replay never reads).
fn raw<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain records; the bytes are read, never interpreted here.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// Writes `derive_memw_a_from_cache`'s known-answer vectors to
/// `MEMW_A_VECTORS` for `crypto/math-cuda/tests/host_kat/memw_a_host_kat.cpp`,
/// which replays the kernel on the host: record walks of asm programs (chunks
/// of 16 or 64 rows), Rust guests and an ethrex block (chunks of 4096), each
/// whole chunk's parts as the card gets them, the windows' logs as the card's
/// cache holds them, the program's DECODE rows, and the words the host fills
/// of the chunk's MEMW_A table with the WALK's rows. Nothing is checked here
/// but that the vectors hold the rows the replay's mutants move: ECALL rows,
/// loads that sign-fill, stores whose old bytes are not their value, and a
/// row with a mixed access of two bytes or more to break.
#[test]
#[ignore = "writes the kernel's vectors to MEMW_A_VECTORS"]
fn write_the_memw_a_kernel_vectors() {
    use super::card_records::{AlignedRefRecords, CardDecode};
    use crate::tables::card_fill_inputs::{col_major, decode_records, log_record};
    use crate::tables::gpack::TraceForm;
    use multilinear::records::TableRecords;
    let path = std::env::var("MEMW_A_VECTORS").expect("MEMW_A_VECTORS");
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../executor/tests");
    let mut cases: Vec<(String, Elf, Vec<u8>, usize, usize, usize)> = [
        ("all_loadstore_32", 8usize),
        ("test_keccak_multi", 16),
        ("test_copy_ecalls", 16),
        ("misalign_sd", 4),
        ("misalign_ld", 4),
    ]
    .iter()
    .map(|&(name, rows)| {
        (
            name.to_string(),
            run(name).0,
            Vec::new(),
            7,
            rows,
            usize::MAX,
        )
    })
    .collect();
    for (name, rows) in [
        ("hint_multi", 64usize),
        ("dma_memmove_cases", 4096),
        ("keccak_precompile", 256),
    ] {
        cases.push((
            name.to_string(),
            rust_elf(name),
            Vec::new(),
            97,
            rows,
            usize::MAX,
        ));
    }
    let input = std::fs::read(fixtures.join("ethrex_empty_block.bin")).expect("the fixture");
    cases.push((
        "ethrex_empty_block".to_string(),
        rust_elf("ethrex"),
        input,
        1 << 16,
        4096,
        8,
    ));
    let order = MemwAlignedOrder::Limbs;
    let mut out: Vec<u8> = b"CTMEMWA1".to_vec();
    put(&mut out, &[cases.len() as u64]);
    let (mut n_chunks, mut ecall, mut sign, mut store_old, mut mixed2) = (0, 0, 0, 0, 0);
    for (name, program, input, window, rows, most) in &cases {
        let (walked, decode) = walked(program, input, *window, order);
        let card = Arc::new(CardDecode::new(Arc::clone(&decode)).expect("DECODE rows"));
        let (entries, runs) = decode_records(&decode);
        put(&mut out, &[entries.len() as u64]);
        out.extend_from_slice(raw(&entries));
        put(&mut out, &[runs.len() as u64]);
        out.extend_from_slice(raw(&runs));
        // The card's cache: every walked window's logs.
        put(&mut out, &[walked.len() as u64]);
        for w in &walked {
            let logs: Vec<_> = w.logs.iter().map(log_record).collect();
            put(&mut out, &[w.first, logs.len() as u64]);
            out.extend_from_slice(raw(&logs));
        }
        let chunks: Vec<RefChunk> = chunks(&walked, *rows).into_iter().take(*most).collect();
        put(&mut out, &[chunks.len() as u64]);
        for chunk in &chunks {
            let held = AlignedRefRecords::new(
                chunk.refs.clone(),
                chunk.windows.clone(),
                Arc::clone(&card),
                order,
            )
            .unwrap_or_else(|_| panic!("{name}: a whole chunk"));
            let a = held.aligned_refs().expect("refs");
            put(&mut out, &[u64::from(a.limbs), a.parts.len() as u64]);
            for p in &a.parts {
                put(&mut out, &[p.refs.len() as u64]);
                out.extend_from_slice(raw(p.refs));
                put(
                    &mut out,
                    &[p.accesses.len() as u64, u64::from(p.access_base)],
                );
                out.extend_from_slice(raw(p.accesses));
                // Eight timestamps an entry, as `u32`s.
                put(&mut out, &[8 * p.mixed.len() as u64]);
                out.extend_from_slice(raw(p.mixed));
                put(&mut out, &[p.whole.len() as u64, u64::from(p.whole_base)]);
                out.extend_from_slice(raw(p.whole));
                for r in p.refs {
                    if r.cycle & ALIGNED_ECALL != 0 {
                        ecall += 1;
                        continue;
                    }
                    let acc = p.accesses[(r.at - p.access_base) as usize];
                    mixed2 += usize::from(acc.mixed != NO_ACCESS);
                }
            }
            for row in &chunk.want {
                let v = row.value();
                sign += usize::from(
                    row.is_read() && row.width() < 8 && v[row.width() as usize] == 0xFF,
                );
                store_old += usize::from(!row.is_read() && row.old() != row.value());
            }
            let want = col_major(
                &crate::tables::memw_aligned::generate_memw_aligned_trace_segments_as(
                    &[&chunk.want],
                    TraceForm::Wide,
                ),
            );
            put(&mut out, &[want.len() as u64]);
            put(&mut out, &want);
            n_chunks += 1;
        }
        println!("{name}: {} chunks of {rows}", chunks.len());
    }
    std::fs::write(&path, &out).expect("the vectors written");
    println!(
        "MEMW_A VECTORS: {} cases, {n_chunks} chunks ({ecall} ECALL rows, {sign} sign fills, \
         {store_old} stores with old ≠ value, {mixed2} mixed accesses), {} bytes → {path}",
        cases.len(),
        out.len()
    );
    assert!(ecall > 0 && sign > 0 && store_old > 0 && mixed2 > 0);
}

/// The whole in-row build of a record-mode run in the limbs' form, walked
/// from the history on two walkers, CPU and MEMW_A chunks held as records,
/// the side records derived on the card; with `refs` MEMW_A's rows too
/// ([`WindowedTraceBuilder::card_derives_memw_a`]), the windows from `until`
/// on coming without their history; its streamed chunks put back. And the
/// MEMW_A chunks held as refs.
fn build(
    program: &Elf,
    input: &[u8],
    window: usize,
    refs: bool,
    until: u32,
) -> (crate::tables::trace_builder::Traces, usize) {
    use super::RowFamilies;
    let room: super::RecordsRoom = Arc::new(|_| true);
    let rows = RowFamilies {
        memw_a: true,
        ..RowFamilies::NONE
    };
    let mut builder = WindowedTraceBuilder::new(program, input, &MaxRowsConfig::small())
        .and_then(|b| b.register_format(RegisterFormat::InRow))
        .and_then(|b| b.memw_aligned_order(MemwAlignedOrder::Limbs))
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(room, true, false))
        .and_then(|b| b.card_fill_streamed_rows(rows))
        .and_then(WindowedTraceBuilder::card_derives_sides)
        .and_then(|b| if refs { b.card_derives_memw_a() } else { Ok(b) })
        .map(|b| b.card_derives_sides_until(until))
        .expect("the builder");
    assert_eq!(builder.derives_memw_a(), refs);
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let mut chunks = Vec::new();
    let mut held = 0usize;
    let report = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, input.to_vec()).expect("the fast engine");
            fast.enable_history().expect("history mode");
            while let Some((logs, history)) = fast.resume_history(window).expect("the run") {
                let last = (fast.pc() == 0)
                    .then(|| fast.take_history_final().map(Box::new))
                    .flatten();
                let window = ExecWindow {
                    logs: Arc::new(logs),
                    start: None,
                    history: Some(Box::new(history)),
                    last,
                };
                if tx.send(window).is_err() {
                    break;
                }
            }
        });
        walk_windows_from_history(&mut builder, rx, 2, 0, |accumulator, walked, _| {
            for job in accumulator.absorb(walked) {
                let chunk = job.generate();
                if let Some(records) = chunk.trace.records_main() {
                    held += usize::from(records.aligned_refs().is_some());
                }
                // Held as records: the host fill's table, compared below.
                chunks.push(super::card_records_tests::unheld(chunk).0);
            }
            Ok(())
        })
    })
    .expect("the walk");
    let mut traces = builder.finish(&report.last).expect("the finish");
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    (traces, held)
}

/// ★ The whole build with MEMW_A's rows derived from the history (the walk's
/// rows handed out as refs) is the build with the walk's rows, every table,
/// in windows that cut chunks inside windows and over the MEMW_A tail the
/// finish derives on the host; and full MEMW_A chunks are held as refs.
#[test]
fn the_build_with_memw_a_rows_derived_is_todays() {
    let mut held = 0usize;
    let mut cases: Vec<(String, Elf, usize)> = ["test_keccak_multi", "all_loadstore_32"]
        .iter()
        .flat_map(|name| [7usize, 33].map(|w| (name.to_string(), run(name).0, w)))
        .collect();
    cases.push((
        "dma_memmove_cases".to_string(),
        rust_elf("dma_memmove_cases"),
        97,
    ));
    for (name, program, window) in &cases {
        let (today, none) = build(program, &[], *window, false, u32::MAX);
        let (derived, n) = build(program, &[], *window, true, u32::MAX);
        assert_eq!(none, 0, "{name}");
        crate::tests::windowed_builder_tests::same_traces(&today, &derived);
        println!("{name}, window {window}: {n} MEMW_A chunks held as refs");
        held += n;
    }
    assert!(held > 4, "{held} chunks held as refs");
}

/// ★ A window that comes without its history mid-run (as a rebuilt one
/// would): the pending refs' rows are derived on the host, every later chunk
/// is held as rows, no chunk mixes the two, and the whole build is today's.
#[test]
fn a_window_without_its_history_ends_the_memw_a_refs_and_the_build_is_todays() {
    let program = rust_elf("dma_memmove_cases");
    let window = 97usize;
    let (today, _) = build(&program, &[], window, false, u32::MAX);
    let (all, n_all) = build(&program, &[], window, true, u32::MAX);
    crate::tests::windowed_builder_tests::same_traces(&today, &all);
    assert!(n_all > 1, "{n_all} chunks held as refs");
    let mut between = false;
    for until in [0u32, 1, 7, 100, 333] {
        let (part, n) = build(&program, &[], window, true, until);
        crate::tests::windowed_builder_tests::same_traces(&today, &part);
        assert!(n <= n_all, "until {until}: {n} of {n_all}");
        if until == 0 {
            assert_eq!(n, 0, "no window hooked, no chunk held as refs");
        }
        between |= n > 0 && n < n_all;
    }
    assert!(between, "no run ended the mode between its chunks");
}

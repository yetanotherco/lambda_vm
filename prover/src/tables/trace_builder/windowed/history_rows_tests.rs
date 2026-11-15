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
#[test]
fn each_rule_of_the_memw_a_derivation_is_load_bearing() {
    let order = MemwAlignedOrder::Limbs;
    let mut seen = [0usize; 6];
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

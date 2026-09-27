//! Temporary switches for the gap campaign's kernel fixes (lane KERN), each OFF
//! unless its environment variable is set, so one binary runs both arms of an
//! A/B. At integration a switch is removed and its fix becomes the only path.
//!
//! - `LAMBDA_VM_GAP_K2=1`: the constraint composition kernel fills the card
//!   instead of stopping at 256 blocks ([`crate::constraint_interp`]).
//! - `LAMBDA_VM_GAP_K6=1`: the DEEP and OOD denominators are inverted row by
//!   row in registers instead of by the six-kernel global scan
//!   ([`crate::inverse`]); the R4 DEEP kernel inverts its own row's
//!   denominators ([`crate::deep`]); the single-point OOD sums are chunked over
//!   rows like the multi-point ones ([`crate::barycentric`]). A comma list of
//!   `inv`, `deep`, `bary` turns on only those parts (a diagnostic; `1` is all
//!   three).
//! - `LAMBDA_VM_GAP_K7=1`: a sumcheck whose slot file caps its launch below
//!   the card's width gets a larger slot budget, reserved against the device
//!   budget ([`crate::sumcheck`]).
//!
//! None of them changes a value: every fix computes the same field elements —
//! K2 bit for bit (a row's walk does not depend on the grid); K6 and K7 up to
//! the representation of a value (an inverse computed another way, a sum added
//! in another order), which nothing downstream observes — see
//! `kernels/ext3_inv.cuh`.
//!
//! Each switch prints one `GAP KNOB` line the first time it is read on, so a
//! log shows that the arm it belongs to really ran with it.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

fn read(var: &str) -> Option<String> {
    std::env::var(var)
        .ok()
        .filter(|v| !v.is_empty() && v != "0")
}

/// K2: the composition kernel's grid fills the card.
pub fn k2() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = read("LAMBDA_VM_GAP_K2").is_some();
        if on {
            eprintln!(
                "GAP KNOB K2=1: constraint_composition_kernel grid = min(rows, card fill, \
                 scratch cap {} MiB), never below the 256-block legacy grid",
                k2_scratch_cap_bytes() >> 20
            );
        }
        on
    })
}

/// Cap on the composition kernel's per-launch value scratch under K2, in bytes.
/// `LAMBDA_VM_GAP_K2_SCRATCH_MB` overrides the 512 MiB default. The legacy
/// grid's scratch is allowed whatever this says: K2 never launches fewer
/// threads than the code it replaces.
pub fn k2_scratch_cap_bytes() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| {
        read("LAMBDA_VM_GAP_K2_SCRATCH_MB")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(512)
            .saturating_mul(1 << 20)
    })
}

/// The three parts of K6, each on under `LAMBDA_VM_GAP_K6=1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct K6 {
    /// Row-wise in-register inversion of the denominators into the same buffer
    /// the scan path fills.
    pub inv: bool,
    /// The R4 DEEP kernel inverts its own row's denominators; no buffer.
    pub deep: bool,
    /// Single-point OOD sums chunked over rows (and the multi-point chunk
    /// count raised to fill the card).
    pub bary: bool,
}

pub fn k6() -> K6 {
    static PARTS: OnceLock<K6> = OnceLock::new();
    *PARTS.get_or_init(|| {
        let parts = match read("LAMBDA_VM_GAP_K6") {
            None => K6::default(),
            Some(v) if v == "1" => K6 {
                inv: true,
                deep: true,
                bary: true,
            },
            Some(v) => {
                // A typo must not read as "every part off": the arm would
                // then measure the A configuration under the B label.
                if let Some(bad) = v
                    .split(',')
                    .map(str::trim)
                    .find(|s| !["inv", "deep", "bary"].contains(s))
                {
                    panic!(
                        "LAMBDA_VM_GAP_K6={v}: `{bad}` is not a part; expected 1 or a comma \
                         list of inv, deep, bary"
                    );
                }
                let has = |p: &str| v.split(',').any(|s| s.trim() == p);
                K6 {
                    inv: has("inv"),
                    deep: has("deep"),
                    bary: has("bary"),
                }
            }
        };
        if parts != K6::default() {
            eprintln!(
                "GAP KNOB K6: inv={} deep={} bary={} (row-wise denominator inversion, \
                 fused DEEP inversion, chunked single-point OOD)",
                parts.inv as u8, parts.deep as u8, parts.bary as u8
            );
        }
        parts
    })
}

/// K7: a sumcheck whose slot file starves its launch gets a larger budget.
pub fn k7() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = read("LAMBDA_VM_GAP_K7").is_some();
        if on {
            eprintln!(
                "GAP KNOB K7=1: sumcheck slot budget up to {} MiB (reserved against the device \
                 budget) for sessions below {} threads",
                k7_slot_budget_bytes() >> 20,
                K7_TARGET_THREADS
            );
        }
        on
    })
}

/// Threads a K7 session aims for before its slot file stops growing.
pub const K7_TARGET_THREADS: u64 = 1 << 16;

/// The slot budget a K7 session may grow to, in bytes.
/// `LAMBDA_VM_GAP_K7_SLOT_MB` overrides the 2 GiB default.
pub fn k7_slot_budget_bytes() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| {
        read("LAMBDA_VM_GAP_K7_SLOT_MB")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(2048)
            .saturating_mul(1 << 20)
    })
}

/// What the fixes did in this process: the mechanism half of an A/B, read
/// beside the kernel traces. Every counter stays zero with its switch off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Composition launches whose grid K2 widened past 256 blocks.
    pub k2_widened: u64,
    /// Composition launches K2 left at the legacy grid (rows or scratch cap).
    pub k2_legacy: u64,
    /// Widened launches whose scratch allocation failed and ran at the legacy grid.
    pub k2_alloc_retries: u64,
    /// Row-wise denominator inversions (one launch each).
    pub k6_rowwise: u64,
    /// Inversions K6 left on the scan path (more points than the row kernels take).
    pub k6_scan: u64,
    /// Fused DEEP launches.
    pub k6_fused_deep: u64,
    /// Single-point OOD sums run through the chunked kernel.
    pub k6_chunked_bary: u64,
    /// Sumcheck sessions K7 widened.
    pub k7_widened: u64,
    /// Sessions K7 wanted to widen and could not (reservation refused or
    /// allocation failed); they ran at the default budget.
    pub k7_declined: u64,
}

pub(crate) static K2_WIDENED: AtomicU64 = AtomicU64::new(0);
pub(crate) static K2_LEGACY: AtomicU64 = AtomicU64::new(0);
pub(crate) static K2_ALLOC_RETRIES: AtomicU64 = AtomicU64::new(0);
pub(crate) static K6_ROWWISE: AtomicU64 = AtomicU64::new(0);
pub(crate) static K6_SCAN: AtomicU64 = AtomicU64::new(0);
pub(crate) static K6_FUSED_DEEP: AtomicU64 = AtomicU64::new(0);
pub(crate) static K6_CHUNKED_BARY: AtomicU64 = AtomicU64::new(0);
pub(crate) static K7_WIDENED: AtomicU64 = AtomicU64::new(0);
pub(crate) static K7_DECLINED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

/// Under K7, one line per distinct sumcheck shape it tried to widen: the
/// program's live values, the first round's indices, the threads before and
/// after, and whether the growth was granted.
pub fn note_k7_shape(
    num_slots: usize,
    first_half: u64,
    base_threads: u64,
    threads: u64,
    granted: bool,
) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(usize, u64, bool)>>> = Mutex::new(None);
    let fresh = SEEN
        .lock()
        .map(|mut seen| {
            seen.get_or_insert_with(HashSet::new)
                .insert((num_slots, first_half, granted))
        })
        .unwrap_or(false);
    if fresh {
        let mib = |t: u64| (t * num_slots as u64 * 24) as f64 / (1u64 << 20) as f64;
        eprintln!(
            "GAP K7 shape: slots {num_slots} first half {first_half}: threads {base_threads} \
             ({:.0} MiB) -> {threads} ({:.0} MiB) {}",
            mib(base_threads),
            mib(threads),
            if granted {
                "granted"
            } else {
                "DECLINED (budget or card)"
            }
        );
    }
}

/// Under K2, one line per distinct composition shape — the program's scratch
/// per thread and the grid K2 launched it at — so an arm's log carries the
/// sizes its mechanism reading depends on. Printed once per shape, not per
/// launch.
pub fn note_k2_shape(
    num_nodes: usize,
    base_slots: usize,
    ext_slots: usize,
    rows: usize,
    legacy_grid: u32,
    grid: u32,
) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    /// (nodes, base slots, ext slots, rows)
    type Shape = (usize, usize, usize, usize);
    static SEEN: Mutex<Option<HashSet<Shape>>> = Mutex::new(None);
    let key = (num_nodes, base_slots, ext_slots, rows);
    let fresh = SEEN
        .lock()
        .map(|mut seen| seen.get_or_insert_with(HashSet::new).insert(key))
        .unwrap_or(false);
    if fresh {
        let per_thread = (base_slots + 3 * ext_slots) * 8;
        let mib = |grid: u32| (grid as usize * 256 * per_thread) as f64 / (1u64 << 20) as f64;
        eprintln!(
            "GAP K2 shape: nodes {num_nodes} slots b/e {base_slots}/{ext_slots} \
             ({per_thread} B/thread) rows {rows}: grid {legacy_grid} ({:.0} MiB) -> {grid} ({:.0} MiB)",
            mib(legacy_grid),
            mib(grid)
        );
    }
}

pub fn counts() -> Counts {
    let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
    Counts {
        k2_widened: get(&K2_WIDENED),
        k2_legacy: get(&K2_LEGACY),
        k2_alloc_retries: get(&K2_ALLOC_RETRIES),
        k6_rowwise: get(&K6_ROWWISE),
        k6_scan: get(&K6_SCAN),
        k6_fused_deep: get(&K6_FUSED_DEEP),
        k6_chunked_bary: get(&K6_CHUNKED_BARY),
        k7_widened: get(&K7_WIDENED),
        k7_declined: get(&K7_DECLINED),
    }
}

/// One line for a run's summary: the switches as read, and what each did.
pub fn summary_line() -> String {
    format_summary(k2(), k6(), k7(), counts())
}

/// [`summary_line`]'s text. The lane's box script parses it (its integration
/// steps compare the switches-off line verbatim and read the counts of the
/// switches-on one), so its format is pinned by a test.
fn format_summary(k2: bool, parts: K6, k7: bool, c: Counts) -> String {
    format!(
        "GAP KERN: K2={} widened {} legacy {} alloc-retries {} · K6 inv={} deep={} bary={} \
         rowwise {} scan {} fused-deep {} chunked-bary {} · K7={} widened {} declined {}",
        k2 as u8,
        c.k2_widened,
        c.k2_legacy,
        c.k2_alloc_retries,
        parts.inv as u8,
        parts.deep as u8,
        parts.bary as u8,
        c.k6_rowwise,
        c.k6_scan,
        c.k6_fused_deep,
        c.k6_chunked_bary,
        k7 as u8,
        c.k7_widened,
        c.k7_declined,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two lines `kern-tests.sh` is written against: every switch off
    /// (its knobs-off integration step requires this line verbatim), and every
    /// field distinct, so a parser that reads one count from another's place
    /// fails its selftest.
    #[test]
    fn summary_line_format_is_the_one_the_box_script_parses() {
        assert_eq!(
            format_summary(false, K6::default(), false, Counts::default()),
            "GAP KERN: K2=0 widened 0 legacy 0 alloc-retries 0 · K6 inv=0 deep=0 bary=0 \
             rowwise 0 scan 0 fused-deep 0 chunked-bary 0 · K7=0 widened 0 declined 0"
        );
        let all = K6 {
            inv: true,
            deep: true,
            bary: true,
        };
        let c = Counts {
            k2_widened: 11,
            k2_legacy: 12,
            k2_alloc_retries: 13,
            k6_rowwise: 14,
            k6_scan: 15,
            k6_fused_deep: 16,
            k6_chunked_bary: 17,
            k7_widened: 18,
            k7_declined: 19,
        };
        assert_eq!(
            format_summary(true, all, true, c),
            "GAP KERN: K2=1 widened 11 legacy 12 alloc-retries 13 · K6 inv=1 deep=1 bary=1 \
             rowwise 14 scan 15 fused-deep 16 chunked-bary 17 · K7=1 widened 18 declined 19"
        );
    }
}

//! The recursion proofs' domains and twiddles built ahead of their card holds.
//!
//! A recursion proof builds each table length's domain and twiddles on first
//! use, inside `multi_prove`, under the exclusive card permit: the first proof
//! of a process holds the card for about 0.3 s of host work while the next
//! proof waits for it. Instead, the early leaf's thread warms its tables' lengths
//! beside phase B, and every proof warms its own lengths before it takes the
//! card; `multi_prove` then finds them in the cache. The cached values are the
//! ones a proof builds for itself, so no proof byte moves. Measured against
//! building them on first use: the median and the large block's recursion were
//! 0.17 s and 0.62 s shorter, with ≈ 0.2 GiB more host memory at the base's
//! peak.

use std::sync::atomic::{AtomicU64, Ordering};

use stark::proof::options::ProofOptions;

use super::trace::LfmTraces;
use crate::tables::types::GoldilocksField;

static BUILT: AtomicU64 = AtomicU64::new(0);
static BUILD_NS: AtomicU64 = AtomicU64::new(0);
/// The bytes the built entries hold, and how many of them were built by the
/// time the early leaf's lengths were warm (the ones a base peak can carry).
static BYTES: AtomicU64 = AtomicU64::new(0);
static EARLY_BYTES: AtomicU64 = AtomicU64::new(0);
/// When the early leaf's lengths were warmed, in seconds on the tree's clock
/// (the bits of an `f64`; `u64::MAX` for never).
static EARLY_AT: AtomicU64 = AtomicU64::new(u64::MAX);

/// The shortest table length warmed: the LFM's padded heights start at 4 rows,
/// and a shorter table (a one-row chip) costs nothing to build at its prove.
const MIN_ROWS: usize = 4;

/// Every table's row count of at least [`MIN_ROWS`], chunks included, each
/// length once, in order.
fn lengths(traces: &LfmTraces) -> Vec<usize> {
    let mut rows: Vec<usize> = [
        &traces.const_,
        &traces.balu,
        &traces.xalu,
        &traces.select,
        &traces.bitdec,
        &traces.hash,
        &traces.keccak,
        &traces.lanes,
        &traces.hint,
        &traces.public,
        &traces.range,
        &traces.keccak_rc,
        &traces.bitwise,
    ]
    .into_iter()
    .chain(&traces.hash_tail)
    .chain(&traces.merged)
    .chain(&traces.blake3)
    .chain(&traces.keccak_rnd)
    .map(|t| t.num_rows())
    .filter(|&n| n >= MIN_ROWS)
    .collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Builds the domain and twiddles of every table length of `traces` under
/// `options` that the cache does not hold yet; returns how many it built.
/// Call it from a plain thread, never a rayon worker.
pub(crate) fn warm(traces: &LfmTraces, options: &ProofOptions) -> usize {
    let t = std::time::Instant::now();
    let built: Vec<u64> = lengths(traces)
        .into_iter()
        .filter_map(|n| {
            stark::prover::warm_domain_twiddles::<GoldilocksField>(
                options.blowup_factor as usize,
                options.coset_offset,
                n,
            )
        })
        .collect();
    BYTES.fetch_add(built.iter().sum::<u64>(), Ordering::Relaxed);
    let built = built.len();
    BUILT.fetch_add(built as u64, Ordering::Relaxed);
    BUILD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    built
}

/// `W3_NODE_DOMAINS_EARLY` (default on), set by the tree at its start: a node
/// program's lengths are warmed as soon as it is emitted ([`warm_lengths`] on
/// a thread of its own), instead of at its prove's start.
static NODES_EARLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The lengths the nodes' early warm-ups built this tree.
static NODES_EARLY_BUILT: AtomicU64 = AtomicU64::new(0);

/// Sets `W3_NODE_DOMAINS_EARLY` for the tree that starts now.
pub(crate) fn set_nodes_early(on: bool) {
    NODES_EARLY.store(on, Ordering::Relaxed);
    NODES_EARLY_BUILT.store(0, Ordering::Relaxed);
}

/// Counts what a node's early warm-up built.
pub(crate) fn note_nodes_early_built(built: usize) {
    NODES_EARLY_BUILT.fetch_add(built as u64, Ordering::Relaxed);
}

/// The tree's `W3 NODE DOMAINS` line: where the nodes' lengths were warmed.
pub(crate) fn nodes_line() -> String {
    if nodes_early() {
        format!(
            "warmed as each node program is emitted · {} lengths built early",
            NODES_EARLY_BUILT.load(Ordering::Relaxed)
        )
    } else {
        "at each prove's start (W3_NODE_DOMAINS_EARLY=0)".to_string()
    }
}

/// Whether node programs warm their lengths when emitted.
pub(crate) fn nodes_early() -> bool {
    NODES_EARLY.load(Ordering::Relaxed)
}

/// The lengths a program's traces take that depend on the program, from the
/// program alone: its groups' padded rows (a merged table keeps its tables'
/// height), its BLAKE3 chunks' and its split `LFM_HASH` chunks', each at least
/// [`MIN_ROWS`], once each. The fixed tables (range, KECCAK_RC, BITWISE, the
/// KECCAK_RND chunks) take the same lengths in every prove, so the first leaf's
/// prove has built them. A length missed here is built at the prove's start as
/// before; an extra one costs only its build.
pub(crate) fn program_lengths(program: &super::compiler::LfmProgram) -> Vec<usize> {
    let mut rows: Vec<usize> = super::registry::program_groups(program)
        .iter()
        .map(|g| g.padded_rows)
        .chain(super::airs::blake3_chunk_rows(program))
        .chain(
            program
                .hash_chunk_real_rows()
                .into_iter()
                .map(super::layout::padded_rows),
        )
        .filter(|&n| n >= MIN_ROWS && n.is_power_of_two())
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Builds the domains and twiddles of `lengths` under `options` that are not
/// built yet ([`warm`]'s work, for lengths known before the traces); returns
/// how many it built.
pub(crate) fn warm_lengths(lengths: &[usize], options: &ProofOptions) -> usize {
    let t = std::time::Instant::now();
    let built: Vec<u64> = lengths
        .iter()
        .filter_map(|&n| {
            stark::prover::warm_domain_twiddles::<GoldilocksField>(
                options.blowup_factor as usize,
                options.coset_offset,
                n,
            )
        })
        .collect();
    BYTES.fetch_add(built.iter().sum::<u64>(), Ordering::Relaxed);
    let built = built.len();
    BUILT.fetch_add(built as u64, Ordering::Relaxed);
    BUILD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    built
}

/// The early leaf's lengths are warm, `at` seconds into the tree's clock.
pub(crate) fn note_early(at: f64) {
    EARLY_BYTES.store(BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
    EARLY_AT.store(at.to_bits(), Ordering::Relaxed);
}

/// What the warm-up did this process, for a harness's line.
pub fn line() -> String {
    let early = EARLY_AT.load(Ordering::Relaxed);
    let gib = |b: &AtomicU64| b.load(Ordering::Relaxed) as f64 / (1u64 << 30) as f64;
    format!(
        "on · {} lengths built in {:.3} s · the early leaf's by {} · {:.3} GiB built, {:.3} GiB by the early leaf's",
        BUILT.load(Ordering::Relaxed),
        BUILD_NS.load(Ordering::Relaxed) as f64 / 1e9,
        if early == u64::MAX {
            "never".to_string()
        } else {
            format!("{:.2}s", f64::from_bits(early))
        },
        gib(&BYTES),
        gib(&EARLY_BYTES),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters are process-wide: one of these tests at a time.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The lengths a program gives before it runs (`W3_NODE_DOMAINS_EARLY`)
    /// cover the lengths its filled traces take, so an early warm leaves its
    /// prove nothing to build; and a warm of them finds them after.
    #[test]
    fn a_programs_lengths_cover_its_filled_traces() {
        let _g = lock();
        use crate::tables::types::FE;
        let program = super::super::programs::trivial_program();
        let mut opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(16)
            .expect("blowup 16 is valid");
        opts.coset_offset = 13;
        let arenas = vec![
            (0..4u64)
                .map(|i| core::array::from_fn(|j| FE::from(2_000 * (i + 1) + j as u64)))
                .collect(),
        ];
        let hasher = super::super::registry::build_artifacts(&program, &opts).hasher;
        let filled =
            super::super::proof::lfm_execute_and_fill(&program, &arenas, hasher).expect("filled");
        let early = program_lengths(&program);
        let t = filled.traces();
        let fixed: Vec<usize> = [&t.range, &t.keccak_rc, &t.bitwise]
            .into_iter()
            .chain(&t.keccak_rnd)
            .map(|table| table.num_rows())
            .collect();
        for l in lengths(t).into_iter().filter(|l| l.is_power_of_two()) {
            assert!(
                early.contains(&l) || fixed.contains(&l),
                "length {l} of the traces is neither the program's {early:?} nor a fixed table's {fixed:?}"
            );
        }
        assert_eq!(warm_lengths(&early, &opts), early.len(), "each built once");
        let fixed_built = warm_lengths(&fixed, &opts);
        assert!(fixed_built <= fixed.len());
        assert_eq!(
            warm(filled.traces(), &opts),
            0,
            "the prove then builds nothing"
        );
    }

    /// ★ A proof's table lengths are built once and found after: the second
    /// warm builds nothing (the proof's own `multi_prove` lookups then hit,
    /// which the stark crate's test shows value for value).
    #[test]
    fn a_proofs_lengths_are_built_once_then_found() {
        let _g = lock();
        use crate::tables::types::FE;
        let program = super::super::programs::trivial_program();
        // A blowup and coset offset no other test proves under: the
        // process-wide cache holds none of these keys yet.
        let mut opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(8)
            .expect("blowup 8 is valid");
        opts.coset_offset = 11;
        let arenas = vec![
            (0..4u64)
                .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
                .collect(),
        ];
        let hasher = super::super::registry::build_artifacts(&program, &opts).hasher;
        let filled =
            super::super::proof::lfm_execute_and_fill(&program, &arenas, hasher).expect("filled");
        let warmed: Vec<usize> = lengths(filled.traces())
            .into_iter()
            .filter(|l| l.is_power_of_two())
            .collect();
        let n = warmed.len();
        assert!(n > 0, "a proof has tables");
        // Each entry holds n·(2·blowup + 3) field elements (the LDE coset and
        // trace roots, the two-half sets, the coset weights), 8 bytes each.
        let want: u64 = warmed.iter().map(|&l| (l * (2 * 8 + 3) * 8) as u64).sum();
        let before = BYTES.load(Ordering::Relaxed);
        assert_eq!(warm(filled.traces(), &opts), n, "every length built once");
        assert_eq!(
            BYTES.load(Ordering::Relaxed) - before,
            want,
            "their bytes counted"
        );
        assert_eq!(warm(filled.traces(), &opts), 0, "then found");
        assert_eq!(
            BYTES.load(Ordering::Relaxed) - before,
            want,
            "found ones count nothing"
        );
    }
}

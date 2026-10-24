//! The recursion proofs' domains and twiddles built ahead of their card holds
//! (`LFM_WARM_DOMAINS=1`; off by default).
//!
//! A recursion proof builds each table length's domain and twiddles on first
//! use, inside `multi_prove`, under the exclusive card permit: the first proof
//! of a process holds the card for about 0.3 s of host work while the next
//! proof waits for it. On, the early leaf's thread warms its tables' lengths
//! beside phase B, and every proof warms its own lengths before it takes the
//! card; `multi_prove` then finds them in the cache. The cached values are the
//! ones a proof builds for itself, so no proof byte moves.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use stark::proof::options::ProofOptions;

use super::trace::LfmTraces;
use crate::tables::types::GoldilocksField;

static BUILT: AtomicU64 = AtomicU64::new(0);
static BUILD_NS: AtomicU64 = AtomicU64::new(0);
/// When the early leaf's lengths were warmed, in seconds on the tree's clock
/// (the bits of an `f64`; `u64::MAX` for never).
static EARLY_AT: AtomicU64 = AtomicU64::new(u64::MAX);

/// `LFM_WARM_DOMAINS` as given: `1` on, anything else off.
fn setting(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Whether the warm-up is on, read once.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| setting(std::env::var("LFM_WARM_DOMAINS").ok().as_deref()))
}

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
    let built = lengths(traces)
        .into_iter()
        .filter(|&n| {
            stark::prover::warm_domain_twiddles::<GoldilocksField>(
                options.blowup_factor as usize,
                options.coset_offset,
                n,
            )
        })
        .count();
    BUILT.fetch_add(built as u64, Ordering::Relaxed);
    BUILD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    built
}

/// The early leaf's lengths are warm, `at` seconds into the tree's clock.
pub(crate) fn note_early(at: f64) {
    EARLY_AT.store(at.to_bits(), Ordering::Relaxed);
}

/// What the warm-up did this process, for a harness's line.
pub fn line() -> String {
    if !on() {
        return "off (LFM_WARM_DOMAINS=1 builds the recursion proofs' domains and twiddles before their card holds)"
            .to_string();
    }
    let early = EARLY_AT.load(Ordering::Relaxed);
    format!(
        "on · {} lengths built in {:.3} s · the early leaf's by {}",
        BUILT.load(Ordering::Relaxed),
        BUILD_NS.load(Ordering::Relaxed) as f64 / 1e9,
        if early == u64::MAX {
            "never".to_string()
        } else {
            format!("{:.2}s", f64::from_bits(early))
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_knob_is_on_only_at_one() {
        assert!(setting(Some("1")));
        for off in [None, Some(""), Some("0"), Some("on"), Some("true")] {
            assert!(!setting(off), "{off:?}");
        }
    }

    /// ★ A proof's table lengths are built once and found after: the second
    /// warm builds nothing (the proof's own `multi_prove` lookups then hit,
    /// which the stark crate's test shows value for value).
    #[test]
    fn a_proofs_lengths_are_built_once_then_found() {
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
        let n = lengths(filled.traces())
            .into_iter()
            .filter(|l| l.is_power_of_two())
            .count();
        assert!(n > 0, "a proof has tables");
        assert_eq!(warm(filled.traces(), &opts), n, "every length built once");
        assert_eq!(warm(filled.traces(), &opts), 0, "then found");
    }
}

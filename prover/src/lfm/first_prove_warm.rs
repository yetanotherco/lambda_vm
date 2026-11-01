//! The recursion tree's first prove warmed up beside the base
//! (`LFM_WARM_FIRST_PROVE=1`; off by default).
//!
//! The tree's first prove pays three first-use costs on the tree's critical
//! path, with the card idle or waiting:
//!
//! - its per-domain prove constants (`stark::prover::warm_prove_constants`:
//!   LDE-size boundary-zerofier inversions and their card copies), built in its
//!   host preparation after the leaves' artifacts are published;
//! - the DEEP's whole LDE coset, uploaded from pageable memory inside its card
//!   window (`stark::prover::warm_deep_coset_points`; 192 MiB at a doubled leaf);
//! - the staging pairs' pinned allocations, made inside the first artifact
//!   build's and the first prove's card windows.
//!
//! On, the early leaf's thread makes the staging pairs when it starts and,
//! after its fill, warms the first two for its own tables from an AIR set built
//! from shape-only artifacts (`registry::census_artifacts_merged`: every height,
//! chunk and merge the real artifacts have, every root zero). Every prove also
//! warms the DEEP cosets before it takes the card, as it warms its constants.
//! A warmed value is the one the prove caches for itself, keyed by the domain,
//! never by a root, so no proof byte moves.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use stark::proof::options::{OneRowMode, ProofOptions};

use super::compiler::LfmProgram;
use super::hash::HasherKind;
use super::trace::LfmTraces;

/// The staging pairs made ahead: what the tree made on first use at both block
/// sizes, as measured (`W3 STAGING AT END: staging pairs 4 of 8`).
pub const STAGING_PAIRS: usize = 4;

/// `LFM_WARM_FIRST_PROVE` as given: on for `1` or `on`; off when unset, empty,
/// `0` or `off`; any other value is an error.
fn setting(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        None | Some("" | "0" | "off") => false,
        Some("1" | "on") => true,
        Some(other) => {
            panic!("LFM_WARM_FIRST_PROVE must be `0`, `1`, `off` or `on`, got `{other}`")
        }
    }
}

/// Runtime override of [`on`]: 0 = the environment's choice, 1 = off, 2 = on.
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Force the warm-up on, off, or back to the environment (tests).
pub fn set(on: Option<bool>) {
    let v = match on {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    OVERRIDE.store(v, Ordering::Relaxed);
}

/// Whether the warm-up is on (the environment read once).
pub fn on() -> bool {
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| setting(std::env::var("LFM_WARM_FIRST_PROVE").ok().as_deref()))
}

static PAIRS_MADE: AtomicU64 = AtomicU64::new(0);
static PAIRS_NS: AtomicU64 = AtomicU64::new(0);
static EARLY_NS: AtomicU64 = AtomicU64::new(0);
static EARLY_TABLES: AtomicU64 = AtomicU64::new(0);
static EARLY_COSETS: AtomicU64 = AtomicU64::new(0);
/// When the early leaf's warm-up ended, in seconds on the tree's clock (the
/// bits of an `f64`; `u64::MAX` for never).
static EARLY_AT: AtomicU64 = AtomicU64::new(u64::MAX);
/// The host preparation of the first proves (AIR set and pre-hold warm-up),
/// seconds, in prove order; both arms record it.
static PREPS: Mutex<Vec<f64>> = Mutex::new(Vec::new());
const PREPS_KEPT: usize = 16;

/// Makes the staging pairs the tree's transfers would make on first use
/// ([`STAGING_PAIRS`]), and counts them. A no-op without a card.
// The tree's early leaf (`whir_block_tree`, a `parallel` build) is the caller.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn prewarm_staging() {
    let t = std::time::Instant::now();
    let made = stark::prover::prewarm_staging_pairs(STAGING_PAIRS);
    PAIRS_MADE.fetch_add(made as u64, Ordering::Relaxed);
    PAIRS_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Warms what `program`'s first prove would build on first use, for its
/// filled `traces` under `options`: the AIR set from shape-only artifacts,
/// then the per-domain prove constants and the DEEP's LDE cosets for its
/// tables (what [`super::proof::warm_before_hold`] warms, whichever of its
/// knobs are on). Returns how many tables it paired.
/// Nothing under a one-row format (its AIRs need the real one-row roots).
/// Call it from a plain thread, not a rayon worker.
// The tree's early leaf (`whir_block_tree`, a `parallel` build) is the caller.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn warm(
    program: &LfmProgram,
    traces: &mut LfmTraces,
    options: &ProofOptions,
    hasher: HasherKind,
) -> Result<usize, String> {
    if options.format.one_row != OneRowMode::Off {
        return Ok(0);
    }
    let t = std::time::Instant::now();
    let cosets_before = stark::prover::warmed_deep_cosets();
    let shape = super::registry::census_artifacts_merged(
        program,
        hasher,
        options.format.merge_equal_height_tables,
    );
    traces.merge_to(&shape.merge.plan())?;
    let airs = super::proof::lfm_airs(&shape, options, hasher).map_err(|e| format!("{e:?}"))?;
    let pairs = airs.air_trace_pairs(traces);
    let tables: Vec<_> = pairs
        .iter()
        .map(|(air, trace, pi)| (*air, trace.num_rows(), *pi))
        .collect();
    stark::prover::warm_prove_constants(&tables);
    stark::prover::warm_deep_coset_points(&tables);
    let tables = tables.len();
    EARLY_TABLES.fetch_add(tables as u64, Ordering::Relaxed);
    EARLY_COSETS.fetch_add(
        stark::prover::warmed_deep_cosets() - cosets_before,
        Ordering::Relaxed,
    );
    EARLY_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(tables)
}

/// The early leaf's warm-up ended `at` seconds into the tree's clock.
// The tree's early leaf (`whir_block_tree`, a `parallel` build) is the caller.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn note_early(at: f64) {
    EARLY_AT.store(at.to_bits(), Ordering::Relaxed);
}

/// A prove's host preparation before it takes the card, in seconds.
pub(crate) fn note_prep(secs: f64) {
    let mut preps = PREPS.lock().unwrap_or_else(|e| e.into_inner());
    if preps.len() < PREPS_KEPT {
        preps.push(secs);
    }
}

/// What the warm-up did this process, for a harness's line. The proves'
/// preparation times are printed in both arms.
pub fn line() -> String {
    let preps = PREPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let preps = preps
        .iter()
        .map(|s| format!("{s:.3}"))
        .collect::<Vec<_>>()
        .join(" ");
    if !on() {
        return format!("off · prove prep s [{preps}]");
    }
    let early = EARLY_AT.load(Ordering::Relaxed);
    format!(
        "on · staging pairs made {} in {:.3} s · early leaf warmed {} tables, {} DEEP cosets in {:.3} s, by {} · prove prep s [{preps}]",
        PAIRS_MADE.load(Ordering::Relaxed),
        PAIRS_NS.load(Ordering::Relaxed) as f64 / 1e9,
        EARLY_TABLES.load(Ordering::Relaxed),
        EARLY_COSETS.load(Ordering::Relaxed),
        EARLY_NS.load(Ordering::Relaxed) as f64 / 1e9,
        if early == u64::MAX {
            "never".to_string()
        } else {
            format!("{:.2}s", f64::from_bits(early))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off by default: unset, empty, `0` or `off`; on for `1` or `on`.
    #[test]
    fn the_knob_is_off_unless_one_or_on() {
        for off in [None, Some(""), Some("0"), Some("off"), Some(" 0 ")] {
            assert!(!setting(off), "{off:?}");
        }
        for on in [Some("1"), Some("on"), Some(" 1 ")] {
            assert!(setting(on), "{on:?}");
        }
    }

    #[test]
    #[should_panic(expected = "LFM_WARM_FIRST_PROVE must be")]
    fn an_unknown_value_is_refused() {
        setting(Some("yes"));
    }

    /// The shape-only artifacts the warm-up builds its AIR set from carry every
    /// height, chunk list, chip set and merge a real build has; only roots and
    /// the id differ. So the warmed tables are the prove's tables, row for row.
    #[test]
    fn the_census_artifacts_have_the_built_shape() {
        let opts = super::super::proof::aggregation_wrap_options();
        for program in [
            super::super::programs::trivial_program(),
            super::super::programs::fri_toy_program(),
        ] {
            let built = super::super::registry::build_artifacts(&program, &opts);
            let shape = super::super::registry::census_artifacts_merged(
                &program,
                built.hasher,
                opts.format.merge_equal_height_tables,
            );
            assert_eq!(shape.log_heights, built.log_heights);
            assert_eq!(shape.keccak_rnd_chunks, built.keccak_rnd_chunks);
            assert_eq!(
                shape.blake3_chunk_log_heights,
                built.blake3_chunk_log_heights
            );
            assert_eq!(shape.hash_chunk_log_heights, built.hash_chunk_log_heights);
            assert_eq!(shape.chip_set, built.chip_set);
            assert_eq!(shape.hasher, built.hasher);
            assert_eq!(shape.merge.plan(), built.merge.plan());
            let heights = |m: &super::super::merge::LfmMerge| {
                m.tables.iter().map(|t| t.log_height).collect::<Vec<_>>()
            };
            assert_eq!(heights(&shape.merge), heights(&built.merge));
        }
    }

    /// ★ The warm-up leaves nothing for the prove's own pre-hold warm-up to
    /// build: before it the real AIR set's tables (built from the real
    /// artifacts) are cold, after it they find every boundary-zerofier
    /// inverse in the cache.
    #[test]
    fn the_early_warm_leaves_the_prove_nothing_to_build() {
        use crate::tables::types::FE;
        let program = super::super::programs::trivial_program();
        // The recursion's options (its merge, which the fill follows) under a
        // coset offset no other test proves under: the process-wide caches
        // hold none of these domains yet.
        let mut opts = super::super::proof::aggregation_wrap_options();
        opts.coset_offset = 19;
        let arenas = vec![
            (0..4u64)
                .map(|i| core::array::from_fn(|j| FE::from(2_000 * (i + 1) + j as u64)))
                .collect(),
        ];
        let artifacts = super::super::registry::build_artifacts(&program, &opts);
        let mut filled =
            super::super::proof::lfm_execute_and_fill(&program, &arenas, artifacts.hasher)
                .expect("filled");
        let real = super::super::proof::lfm_airs(&artifacts, &opts, artifacts.hasher)
            .expect("the AIR set");
        let real_warm = |filled: &mut super::super::proof::LfmFilled| {
            let pairs = real.air_trace_pairs(filled.traces_mut());
            let tables: Vec<_> = pairs
                .iter()
                .map(|(air, trace, pi)| (*air, trace.num_rows(), *pi))
                .collect();
            stark::prover::prove_constants_warm(&tables)
        };
        assert!(
            !real_warm(&mut filled),
            "cold before the warm-up: these domains were in the cache already"
        );
        let tables = warm(&program, filled.traces_mut(), &opts, artifacts.hasher).expect("warmed");
        assert!(tables > 0, "a program has tables");
        assert!(
            real_warm(&mut filled),
            "the real AIR set's tables still have an inverse to build after the warm-up"
        );
    }
}

//! Per level: how many proofs were emitted, and how many DISTINCT PROGRAMS were
//! behind them.
//!
//! # This line exists to document why there is no artifact cache
//!
//! `build_artifacts_with_hasher` is a pure function of `(program, options,
//! hasher)`, and lane P measured it as the largest phase of the recursion
//! pipeline — 22.0 s of an L1 interior node's 34.0 s wall, all host CPU. That
//! shape invites a memo: if a level's proofs shared a program, a level would
//! build its artifacts once instead of once per proof.
//!
//! ⛔ **They do not share one, and it is load-bearing that they do not.** Every
//! child's epoch label and every node's label range are pinned as CONSTANTS of
//! the emitted program — that pinning is exactly what makes contiguity across
//! sibling subtrees a consequence of the pins rather than a check of its own. So
//! sibling programs differ by construction: different constants, different
//! instruction column groups, different preprocessed roots, different
//! `program_id`. A memo keyed on anything sound records zero hits, and an LDE is
//! global, so no incremental Merkle trick survives a changed constant either.
//!
//! ⇒ What is left is the COUNT, and the count is worth printing: it is the
//! evidence for the paragraph above. A level reporting `N proofs, N distinct
//! programs` is the measurement that says a cache was not left on the table. A
//! level that ever reported fewer programs than proofs would mean the premise
//! had changed and the memo was worth revisiting.
//!
//! # The key is `program_id`, and here it can be
//!
//! A memo cannot be keyed on `program_id`: it is the build's OUTPUT, the digest
//! of the roots the build produces, so a lookup meant to AVOID the build cannot
//! have it in hand. A census has no such problem — it counts after the build —
//! so it keys on the identity the parents absorb and the soundness argument
//! already trusts, rather than on a second hash of the build's inputs.

use std::collections::HashSet;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use stark::config::Commitment;
use stark::proof::options::ProofOptions;

use super::compiler::LfmProgram;
use super::hash::HasherKind;
use super::registry::{LfmArtifacts, build_artifacts_with_hasher};

/// What one level's worth of artifact builds did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LevelStats {
    /// Proofs that built artifacts inside the window.
    pub proofs: usize,
    /// Distinct `program_id`s among them. Expected to equal `proofs`; see the
    /// module note for why, and for what it would mean if it did not.
    pub distinct: usize,
    /// Time spent inside `build_artifacts_with_hasher`.
    pub build_nanos: u128,
}

impl LevelStats {
    /// The line the driver prints per level.
    pub fn describe(&self, label: &str) -> String {
        format!(
            "{label}: {} proofs, {} distinct programs, artifacts {:.1}s{}",
            self.proofs,
            self.distinct,
            Duration::from_nanos(self.build_nanos as u64).as_secs_f64(),
            if self.distinct == self.proofs {
                " (N/N — sibling programs differ by construction; nothing to cache)"
            } else {
                " ⚠ FEWER PROGRAMS THAN PROOFS — the label pinning changed, a memo is back on the table"
            },
        )
    }
}

struct Window {
    seen: HashSet<Commitment>,
    stats: LevelStats,
}

#[derive(Default)]
struct Census {
    window: Option<Window>,
}

fn census() -> &'static Mutex<Census> {
    static CENSUS: OnceLock<Mutex<Census>> = OnceLock::new();
    CENSUS.get_or_init(|| Mutex::new(Census::default()))
}

fn lock() -> MutexGuard<'static, Census> {
    census()
        .lock()
        .expect("the program census mutex is never held across a panic")
}

/// [`build_artifacts_with_hasher`], counted.
///
/// ⛔ NOT a cache and never becomes one: no map, no key, no reuse. It runs the
/// build every time, exactly as the raw call does, and records which program it
/// built for so a level can report how many distinct ones it saw.
///
/// ⚠ The lock is taken ONCE, after the build, and the counter is updated inside
/// it. The memo this replaced took it twice — once to look up, once to count —
/// and held the first guard across the second call. `Mutex` is not reentrant, so
/// that was a self-deadlock, and a quiet one: the thread parked, the test binary
/// sat at 0% CPU and 22 MB of RSS, and nothing printed.
pub fn build_artifacts_counted(
    program: &LfmProgram,
    options: &ProofOptions,
    hasher: HasherKind,
) -> LfmArtifacts {
    let t = Instant::now();
    let artifacts = build_artifacts_with_hasher(program, options, hasher);
    let build_nanos = t.elapsed().as_nanos();
    lock().record(artifacts.program_id, build_nanos);
    artifacts
}

impl Census {
    fn record(&mut self, program_id: Commitment, build_nanos: u128) {
        if let Some(w) = self.window.as_mut() {
            w.seen.insert(program_id);
            w.stats.proofs += 1;
            w.stats.distinct = w.seen.len();
            w.stats.build_nanos += build_nanos;
        }
    }
}

/// Start counting a level. Replaces any window already open — the driver's
/// levels do not nest.
pub fn begin_level() {
    lock().window = Some(Window {
        seen: HashSet::new(),
        stats: LevelStats::default(),
    });
}

/// Close the window and take what it counted. `None` when no window was open.
pub fn end_level() -> Option<LevelStats> {
    lock().window.take().map(|w| w.stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lfm::programs::{fri_toy_program, trivial_program};
    use crate::tables::types::FE;
    use stark::proof::options::GoldilocksCubicProofOptions;

    /// ⛔ EVERY TEST HERE THAT BUILDS TAKES THIS, not just the one that counts.
    ///
    /// The window is process-global, so a sibling test calling
    /// `build_artifacts_counted` on another harness thread lands INSIDE an open
    /// window and inflates it. That is not hypothetical: at `--test-threads=2`
    /// the counting test read 5 proofs where it had made 3, and a first run that
    /// happened to serialize them passed. Serializing the builders is the fix;
    /// counting only under the guard would leave the same race for the next test
    /// somebody adds.
    ///
    /// Outside this module nothing touches the census in a default `cargo test`
    /// run — the driver tests that do are all `#[ignore]`.
    static WINDOW: Mutex<()> = Mutex::new(());

    fn opts() -> ProofOptions {
        GoldilocksCubicProofOptions::with_blowup(2).expect("options")
    }

    /// ★★★ THE GATE. Counting must not change what is built — field by field,
    /// roots included, not `program_id` alone.
    #[test]
    fn counted_artifacts_are_byte_identical_to_a_raw_build() {
        let _guard = WINDOW.lock().expect("the window guard is never poisoned");
        let program = trivial_program();
        let options = opts();
        let raw = build_artifacts_with_hasher(&program, &options, HasherKind::Test);
        let counted = build_artifacts_counted(&program, &options, HasherKind::Test);
        assert_eq!(raw, counted, "the counted build drifted from the raw one");
        assert_eq!(raw.roots, counted.roots, "group roots");
        assert_eq!(raw.log_heights, counted.log_heights, "group heights");
        assert_eq!(
            raw.blake3_chunk_roots, counted.blake3_chunk_roots,
            "LFM_BLAKE3 chunk roots"
        );
        assert_eq!(raw.program_id, counted.program_id, "program_id");
    }

    /// ★ THE PREMISE THE CENSUS EXISTS TO CHECK, at the smallest scale that has
    /// it: two programs that differ only in an emitted CONSTANT are two
    /// programs with two identities. That is why sibling wraps and sibling
    /// nodes never share artifacts, and it is asserted here rather than in
    /// prose.
    #[test]
    fn one_changed_felt_is_a_different_program() {
        let _guard = WINDOW.lock().expect("the window guard is never poisoned");
        let options = opts();
        let base = trivial_program();
        let mut tweaked = trivial_program();
        let g = &mut tweaked.groups.hint;
        assert!(!g.data.is_empty(), "the HINT group must carry columns");
        let last = g.data.len() - 1;
        g.data[last] += FE::one();
        assert_eq!(g.width, base.groups.hint.width, "the shapes must agree");
        assert_eq!(
            g.padded_rows, base.groups.hint.padded_rows,
            "the shapes must agree"
        );

        let a = build_artifacts_counted(&base, &options, HasherKind::Test);
        let b = build_artifacts_counted(&tweaked, &options, HasherKind::Test);
        assert_ne!(
            a.program_id, b.program_id,
            "one changed field element must move the program identity"
        );
        assert_ne!(a.roots, b.roots, "and must move the preprocessed roots");
    }

    /// Both readings the driver can print, including the one that would put a
    /// memo back on the table — so the counter is known to be able to produce
    /// it rather than only ever printing N/N.
    #[test]
    fn a_level_window_counts_proofs_and_programs() {
        let _guard = WINDOW.lock().expect("the window guard is never poisoned");
        let options = opts();

        begin_level();
        for _ in 0..3 {
            let _ = build_artifacts_counted(&fri_toy_program(), &options, HasherKind::Test);
        }
        let repeated = end_level().expect("a window was open");
        assert_eq!(repeated.proofs, 3);
        assert_eq!(repeated.distinct, 1, "three proofs of ONE program");
        assert!(
            !repeated.describe("level x").contains("N/N"),
            "1 of 3 is not N/N: {}",
            repeated.describe("level x")
        );

        begin_level();
        let _ = build_artifacts_counted(&fri_toy_program(), &options, HasherKind::Test);
        let _ = build_artifacts_counted(&trivial_program(), &options, HasherKind::Test);
        let distinct = end_level().expect("a window was open");
        assert_eq!(distinct.proofs, 2);
        assert_eq!(distinct.distinct, 2, "two proofs of TWO programs");
        assert!(
            distinct.describe("level y").contains("N/N"),
            "an N/N level must say so: {}",
            distinct.describe("level y")
        );

        assert!(
            end_level().is_none(),
            "the window closes once, not once per read"
        );
    }
}

/// The artifact build's own cost, on this host, at this concurrency.
///
/// ⛔ NOT a benchmark of the production node — the biggest registry program is
/// orders of magnitude smaller than an L1 node's, and the campaign's numbers
/// come from the box A/B. It exists so the branch is not pushed with an
/// unmeasured claim about the parallel build: run it twice, once with
/// `LFM_ARTIFACT_PARALLEL=0`, and the two walls bracket what the change does.
///
/// ```text
/// cargo test -p lambda-vm-prover --lib -- the_artifact_build_measures \
///     --ignored --nocapture --exact lfm::program_census::measure::the_artifact_build_measures
/// ```
#[cfg(test)]
mod measure {
    use super::*;
    use stark::proof::options::GoldilocksCubicProofOptions;

    #[test]
    #[ignore = "measurement: prints the artifact build's wall at this host's rayon width"]
    fn the_artifact_build_measures() {
        let options = GoldilocksCubicProofOptions::with_blowup(4).expect("options");
        // ⚠ The registry fixtures are half a megabyte of committed felts and a
        // production node's groups are hundreds, so the SCALED sponge is the
        // only row here whose program dominates its own build. The others are
        // mostly the two FIXED tables (`bitwise` and `keccak_rc` at 2^20 rows),
        // which this change does not touch — read them as a floor.
        //
        // ⚠⚠ AND THE COLUMN COUNT IS WHAT THIS PASS SPREADS, which is why the
        // widest-group width is printed beside each wall. These fixtures
        // concentrate their felts in ONE narrow group, so they are close to the
        // WORST case for a `par_iter` over columns and must not be read as the
        // production figure. Lane P measures `emit_lde` at 445% CPU on the real
        // node — 4.5 of 30.7 cores — which is where the headroom is.
        let sponge_len = crate::lfm::programs::KECCAK_SPONGE_LEN;
        let programs: [(&str, LfmProgram); 4] = [
            (
                "statement_replay",
                crate::lfm::programs::statement_replay_program(),
            ),
            (
                "transcript_replay",
                crate::lfm::programs::transcript_replay_program(),
            ),
            (
                "keccak_sponge",
                crate::lfm::programs::keccak_sponge_program(sponge_len),
            ),
            (
                "keccak_sponge x8192",
                crate::lfm::programs::keccak_sponge_program(sponge_len * 8192),
            ),
        ];
        #[cfg(feature = "parallel")]
        let width = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let width = 1usize;
        println!(
            "\n=== ARTIFACT BUILD — rayon width {width}, parallel {}, groups in flight {} ===",
            crate::lfm::commit::parallel_build(),
            crate::lfm::registry::groups_in_flight()
        );
        for (name, program) in &programs {
            let groups = crate::lfm::registry::program_groups(program);
            let felts: usize = groups.iter().map(|g| g.data.len()).sum::<usize>()
                + program.groups.blake3.data.len();
            let widest = groups.iter().map(|g| g.width).max().unwrap_or(0);
            // Three reps, reported as the MIN: this laptop shares eleven cores
            // with the other lanes, so the mean prices their load and the min
            // prices the build.
            let mut build = f64::MAX;
            let mut id = None;
            for _ in 0..3 {
                let t = Instant::now();
                let built = build_artifacts_with_hasher(program, &options, HasherKind::Test);
                build = build.min(t.elapsed().as_secs_f64());
                id = Some(built.program_id);
            }
            let id = id.expect("three reps");
            println!(
                "  {name:<20} {felts:>12} committed felts ({:>8.1} MiB) · widest group \
                 {widest:>4} cols · build {build:.3}s · id {:02x}{:02x}",
                (felts * 8) as f64 / (1024.0 * 1024.0),
                id[0],
                id[1],
            );
        }
    }
}

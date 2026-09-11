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
use std::thread::ThreadId;
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
    /// Time spent inside `build_artifacts_with_hasher`, SUMMED over the
    /// proofs above.
    ///
    /// ⚠ A SUM, not a wall, and the distinction only appears when a level
    /// proves its siblings concurrently: two overlapping 9 s builds sum to 18 s
    /// inside a 9 s window. Serial, the two are the same number, which is
    /// exactly why this needs saying before anything runs concurrently — a
    /// figure that has been a wall for the whole campaign would silently start
    /// being something else and read as a regression. [`Self::wall_nanos`] is
    /// the window's own elapsed time, and [`Self::describe`] says so on the
    /// line whenever the two disagree.
    pub build_nanos: u128,
    /// How long the window was open: `begin_level` to `end_level`.
    pub wall_nanos: u128,
    /// The largest device working set any artifact commit has ASKED FOR in this
    /// process, in bytes — `stark::device_set`'s own term-by-term accounting,
    /// which is the number admission decides on. Zero when nothing went to the
    /// card, which is every host build, so the line stays quiet there.
    ///
    /// ⓘ A running process maximum, not this level's own: the commits are
    /// sequential and the card is released between them, so the largest single
    /// set is what the build ever needed. It is what the artifact build ASKED
    /// for; an external sampler is what says what the process held.
    pub device_peak_bytes: u64,
    /// Committed groups that took the device path, and those that stayed on the
    /// host because `padded_rows · blowup` was under `gpu_lde`'s 2^14 floor.
    /// ★ THIS LEVEL's own, not the process total — see [`Window::groups_at_open`].
    pub device_groups: u64,
    pub host_groups: u64,
}

impl LevelStats {
    /// The line the driver prints per level.
    pub fn describe(&self, label: &str) -> String {
        format!(
            "{label}: {} proofs, {} distinct programs, artifacts {:.1}s{}{}",
            self.proofs,
            self.distinct,
            Duration::from_nanos(self.build_nanos as u64).as_secs_f64(),
            match self.device_peak_bytes {
                0 => String::new(),
                // ⚠ The two figures have DIFFERENT scopes and the line says so.
                // The groups are this level's; the set is the largest any commit
                // has asked for since the process started, because it is an
                // atomic maximum and a window cannot un-see one.
                b => format!(
                    " · groups {}/{} on device · device set <= {:.2} GiB (process max)",
                    self.device_groups,
                    self.device_groups + self.host_groups,
                    b as f64 / (1024.0 * 1024.0 * 1024.0),
                ),
            },
            if self.distinct == self.proofs {
                " (N/N — sibling programs differ by construction; nothing to cache)"
            } else {
                " ⚠ FEWER PROGRAMS THAN PROOFS — the label pinning changed, a memo is back on the table"
            },
        ) + &self.concurrency_note()
    }

    /// Empty while the level is serial, so the line is byte-identical to every
    /// one already in the campaign's logs — and present the moment the sum
    /// stops being a wall, so nobody reads a concurrent level's `artifacts`
    /// figure as the wall it used to be.
    ///
    /// The 5% margin is a clock margin, not a tolerance for overlap: the two
    /// spans are taken by different timers and a serial level lands a hair
    /// under, never over.
    fn concurrency_note(&self) -> String {
        let sum = Duration::from_nanos(self.build_nanos as u64).as_secs_f64();
        let wall = Duration::from_nanos(self.wall_nanos as u64).as_secs_f64();
        if wall > 0.0 && sum > wall * 1.05 {
            format!(
                " · ⓘ artifacts is a SUM over {} proofs ({sum:.1}s) inside a {wall:.1}s \
                 window — this level proved concurrently",
                self.proofs
            )
        } else {
            String::new()
        }
    }
}

struct Window {
    seen: HashSet<Commitment>,
    stats: LevelStats,
    /// When the window opened, for [`LevelStats::wall_nanos`].
    opened_at: Instant,
    /// ★★★ THE THREADS THIS LEVEL IS MADE OF. A build on any other thread is
    /// not this level's work and is not counted.
    ///
    /// ⛔ THIS IS WHY THE WINDOW CAN SURVIVE CONCURRENCY. It is process-global
    /// and it already miscounted once: at `--test-threads=2` a sibling test
    /// building on another thread landed inside an open window and the counting
    /// test read 5 proofs where it had made 3 — and a first run that happened
    /// to serialize them passed, which is the worst way for a counter to be
    /// wrong. Serializing the builders was the fix available then. It is not
    /// available to a level that proves its siblings concurrently, because
    /// concurrent builders are the point.
    ///
    /// ⇒ So membership is EXPLICIT rather than ambient. A thread that did not
    /// enrol cannot inflate the window no matter what it builds or when, which
    /// makes the bad state unreachable instead of merely unlikely — and the
    /// same guarantee covers the serial path, where it is the case that
    /// actually bit.
    enrolled: HashSet<ThreadId>,
    /// The device/host group counters as they stood when the window OPENED.
    ///
    /// ⛔ The counters in `commit` are process-cumulative, because the device
    /// floor is a process constant and a running total is the useful thing to
    /// keep. A per-LEVEL line reporting them raw reads as "all groups so far",
    /// which is exactly how a reader ends up subtracting two levels by hand to
    /// recover the one they wanted. The window keeps its own baseline and
    /// reports the DELTA.
    groups_at_open: (u64, u64),
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
    // ⛔ THE CARD, FOR THE FIRST OF A PROOF'S TWO DEVICE PHASES.
    // `commit_group_device_or_host` dispatches to `gpu_lde` from inside this
    // build, outside `multi_prove` and so outside every `VramGate`, and its
    // admission is a per-dispatch bound with no running total. Inert unless a
    // driver has armed it.
    //
    // ⚠ Inside the timer on purpose: what a build COST a concurrent level
    // includes what it waited for the card, and a figure that excluded the wait
    // would make a device-bound level look host-bound.
    let t = Instant::now();
    let artifacts = {
        let _card = super::device_permit::hold();
        build_artifacts_with_hasher(program, options, hasher)
    };
    let build_nanos = t.elapsed().as_nanos();
    lock().record(artifacts.program_id, build_nanos);
    artifacts
}

impl Census {
    fn record(&mut self, program_id: Commitment, build_nanos: u128) {
        let Some(w) = self.window.as_mut() else {
            return;
        };
        if !w.enrolled.contains(&std::thread::current().id()) {
            return;
        }
        w.seen.insert(program_id);
        w.stats.proofs += 1;
        w.stats.distinct = w.seen.len();
        w.stats.build_nanos += build_nanos;
    }
}

/// A worker thread's membership of the open level window, for as long as it is
/// held.
///
/// Taken by each thread that proves a sibling of the level being counted, and
/// dropped when that thread is done. The driver's own thread is enrolled by
/// [`begin_level`], so a serial level needs none of this.
#[must_use = "an enrolment that is dropped immediately counts nothing"]
pub struct Enrolment {
    thread: ThreadId,
}

impl Drop for Enrolment {
    fn drop(&mut self) {
        // ⚠ NOT `lock()`. This runs during unwinding when a worker panics, and
        // `lock()` panics on a poisoned mutex — a panic inside a `Drop` that is
        // already unwinding aborts the process. A counter is not worth that, so
        // a poisoned census here simply leaves the enrolment in place: the
        // window is about to be discarded by a failing run anyway.
        if let Ok(mut census) = census().lock()
            && let Some(w) = census.window.as_mut()
        {
            w.enrolled.remove(&self.thread);
        }
    }
}

/// Enrol the CALLING thread in the open level window.
///
/// ⚠ Call it on the worker itself, not on the thread that spawned it: the
/// enrolment names `std::thread::current()`, and enrolling from the spawner
/// would enrol the spawner twice and the worker never — a mistake that counts
/// zero rather than counting wrong, which is the direction this should fail in.
///
/// A no-op when no window is open, so a worker need not know whether its level
/// is being counted.
pub fn enrol() -> Enrolment {
    let thread = std::thread::current().id();
    if let Some(w) = lock().window.as_mut() {
        w.enrolled.insert(thread);
    }
    Enrolment { thread }
}

/// Start counting a level. Replaces any window already open — the driver's
/// levels do not nest.
pub fn begin_level() {
    let groups_at_open = super::commit::device_host_group_counts();
    lock().window = Some(Window {
        seen: HashSet::new(),
        stats: LevelStats::default(),
        opened_at: Instant::now(),
        // The caller's own thread, so a level that never spawns anything counts
        // exactly what it did before this existed.
        enrolled: HashSet::from([std::thread::current().id()]),
        groups_at_open,
    });
}

/// Close the window and take what it counted. `None` when no window was open.
pub fn end_level() -> Option<LevelStats> {
    let w = lock().window.take()?;
    let mut stats = w.stats;
    stats.wall_nanos = w.opened_at.elapsed().as_nanos();
    stats.device_peak_bytes = super::commit::device_artifact_peak_bytes();
    let (dev, host) = super::commit::device_host_group_counts();
    stats.device_groups = dev.saturating_sub(w.groups_at_open.0);
    stats.host_groups = host.saturating_sub(w.groups_at_open.1);
    Some(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lfm::programs::{fri_toy_program, trivial_program};
    use crate::tables::types::FE;
    use stark::proof::options::GoldilocksCubicProofOptions;

    /// ⛔ EVERY TEST HERE THAT BUILDS TAKES THIS, not just the one that counts.
    ///
    /// It no longer guards the COUNT. A sibling test building on another
    /// harness thread used to land inside an open window and inflate it — at
    /// `--test-threads=2` the counting test read 5 proofs where it had made 3,
    /// and a first run that happened to serialize them passed. Enrolment closed
    /// that by construction: an un-enrolled thread's build is not counted, and
    /// `a_build_on_an_unenrolled_thread_is_not_counted` is the assertion that it
    /// stays closed.
    ///
    /// What is left for the guard is the WINDOW ITSELF: `begin_level` replaces
    /// whatever is open, so two tests opening one concurrently would each see
    /// the other's. One window at a time is the invariant, and this is it.
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

    /// ★★★ THE CONTAMINATION, MADE UNREACHABLE. A thread that did not enrol
    /// builds inside an open window and the window does not move.
    ///
    /// This is the defect that already shipped once and passed on scheduling
    /// luck. It is asserted rather than avoided because avoiding it — keeping
    /// the builders serial — is precisely what a level proving its siblings
    /// concurrently cannot do.
    #[test]
    fn a_build_on_an_unenrolled_thread_is_not_counted() {
        let _guard = WINDOW.lock().expect("the window guard is never poisoned");
        let options = opts();

        begin_level();
        std::thread::scope(|s| {
            s.spawn(|| {
                // No `enrol()`. This thread is not part of the level.
                let _ = build_artifacts_counted(&fri_toy_program(), &options, HasherKind::Test);
            });
        });
        let stats = end_level().expect("a window was open");
        assert_eq!(
            stats.proofs, 0,
            "an un-enrolled thread's build inflated the window: {stats:?}"
        );
        assert_eq!(
            stats.build_nanos, 0,
            "and it must contribute no time either"
        );
    }

    /// ★★ AND THE OTHER DIRECTION, which is what makes the test above a gate
    /// rather than a way of counting nothing: an ENROLLED worker IS counted.
    ///
    /// Without this, `record` could return early on every call and the
    /// unreachability test would still pass.
    #[test]
    fn an_enrolled_worker_thread_is_counted() {
        let _guard = WINDOW.lock().expect("the window guard is never poisoned");
        let options = opts();

        begin_level();
        std::thread::scope(|s| {
            for p in [fri_toy_program(), trivial_program()] {
                let options = &options;
                s.spawn(move || {
                    let _enrolled = enrol();
                    let _ = build_artifacts_counted(&p, options, HasherKind::Test);
                });
            }
        });
        let stats = end_level().expect("a window was open");
        assert_eq!(
            stats.proofs, 2,
            "two enrolled workers, two proofs: {stats:?}"
        );
        assert_eq!(stats.distinct, 2, "two different programs");
        assert!(
            stats.build_nanos > 0,
            "and their build time reached the window"
        );
    }

    /// ★ THE LINE SAYS WHEN ITS `artifacts` FIGURE STOPPED BEING A WALL — and
    /// says nothing when it has not, so every line already in the campaign's
    /// logs still reads the same.
    #[test]
    fn the_level_line_announces_a_concurrent_sum_and_only_then() {
        let serial = LevelStats {
            proofs: 2,
            distinct: 2,
            build_nanos: 9_000_000_000,
            wall_nanos: 9_100_000_000,
            ..LevelStats::default()
        };
        assert!(
            !serial.describe("level x").contains("SUM"),
            "a serial level must not have acquired a new clause: {}",
            serial.describe("level x")
        );

        let concurrent = LevelStats {
            build_nanos: 18_000_000_000,
            ..serial
        };
        assert!(
            concurrent.describe("level x").contains("SUM"),
            "a sum twice its window must say so: {}",
            concurrent.describe("level x")
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

    /// ★ WHICH SIDE OF THE DEVICE FLOOR EACH FIXTURE GROUP FALLS ON.
    ///
    /// `gpu_lde` admits on `lde_size = padded_rows · blowup >= 2^14` — a ROW
    /// count, not bytes and not columns (`DEFAULT_GPU_LDE_THRESHOLD`, and the
    /// note beside it says why a cells floor would degenerate). Below it the
    /// commit falls back to the host, so a gate whose groups are all below the
    /// floor compares a host root against a host root and says nothing about the
    /// device path.
    #[test]
    #[ignore = "diagnostic: prints each registry fixture group against the 2^14 device floor"]
    fn the_fixture_groups_against_the_device_floor() {
        use crate::lfm::registry::program_groups;
        const FLOOR: usize = 1 << 14;
        let progs: [(&str, LfmProgram); 4] = [
            ("trivial", crate::lfm::programs::trivial_program()),
            ("fri_toy", crate::lfm::programs::fri_toy_program()),
            (
                "statement_replay",
                crate::lfm::programs::statement_replay_program(),
            ),
            (
                "keccak_sponge",
                crate::lfm::programs::keccak_sponge_program(
                    crate::lfm::programs::KECCAK_SPONGE_LEN,
                ),
            ),
        ];
        let range = crate::lfm::trace::range_group();
        for blowup in [2usize, 4] {
            println!("\n=== device floor {FLOOR} rows (lde = padded_rows x blowup {blowup}) ===");
            for (name, p) in &progs {
                let mut admitted = 0;
                let mut total = 0;
                // ⛔ TWELVE GROUPS, NOT ELEVEN. `build_artifacts_with_hasher`
                // commits `program_groups` (10) plus `LFM_RANGE`, and THEN one
                // group per `LFM_BLAKE3` chunk in a second loop. Slot 11 is
                // committed whether or not the family is used — the digest binds
                // it either way — so a walk that stops at `range` undercounts by
                // the chunk count, which is how this lane predicted 8 of 11 for
                // a production wrap when the answer is 8 of 12.
                let chunks: Vec<_> = (0..p.blake3_chunk_count())
                    .map(|c| p.blake3_chunk_group(c))
                    .collect();
                for (i, g) in program_groups(p)
                    .iter()
                    .copied()
                    .chain(std::iter::once(&range))
                    .chain(chunks.iter())
                    .enumerate()
                {
                    let lde = g.padded_rows * blowup;
                    total += 1;
                    if lde >= FLOOR {
                        admitted += 1;
                    }
                    println!(
                        "  {name:<17} slot {i:>2}  {:>9} rows x {:>4} cols -> lde {:>9}  {}",
                        g.padded_rows,
                        g.width,
                        lde,
                        if lde >= FLOOR { "DEVICE" } else { "host" }
                    );
                }
                println!("  {name:<17} => {admitted} of {total} groups reach the device");
            }
        }
    }

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

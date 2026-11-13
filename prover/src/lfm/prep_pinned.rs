//! `LAMBDA_VM_LFM_PREP_PINNED`: the block tree's leaves' preprocessed groups
//! made ready for the card beside the base, before the tree builds their
//! artifacts.
//!
//! A leaf's artifact build uploads each preprocessed group it commits
//! ([`super::commit::commit_group_device_or_host_under`]). A large group goes
//! through the staging buffers: the host copies it, 32 MiB at a time, into
//! page-locked memory, and the card reads that. The builds run at the tree's
//! start, on the critical path, while the other leaves execute and fill on
//! every core, so that copy competes with them for host memory bandwidth.
//!
//! Each leaf's program exists seconds before the base ends. With this knob a
//! helper thread takes each leaf's program as it is emitted and:
//! - `programs`: page-locks the program's own large preprocessed groups in
//!   place (`multilinear::pinned::register_in_place`), so their uploads go to
//!   the card as direct DMAs, with no staging copy;
//! - `merged`: lays the merged AIRs' groups side by side
//!   ([`super::merge::merged_groups`], the build's own call) and page-locks
//!   them;
//! - `merged-pageable`: lays the merged groups side by side and leaves them
//!   pageable (the control for `merged`'s locking: the layout moved alone);
//! - `all`: `programs` and `merged`.
//!
//! The leaf's build then takes what was prepared ([`take`]). The merged
//! groups are used only when the build's own merge plan equals the one they
//! were laid out for. The bytes the card receives are the same in every mode,
//! so the artifacts and the proofs do not change. Off (unset, `0`, `off`) is
//! today's path: no helper, and [`take`] returns at once. Registration is
//! undone on a release thread after each build, off the builder's path.
//!
//! The helper holds each program until its build has taken the preparation
//! (or the tree ends), so no registered buffer is freed while registered.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use super::compiler::{ColumnGroup, LfmProgram};
use super::merge::MergePlan;
use crate::ProofOptions;

/// The knob's name.
pub const KNOB: &str = "LAMBDA_VM_LFM_PREP_PINNED";

/// The smallest group the helper locks: the size from which an upload takes
/// the staging buffers (`math_cuda::lde`'s `PINNED_H2D_MIN_U64`, 2^20 words).
pub const MIN_LOCK_BYTES: usize = 8 << 20;

/// What the helper prepares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrepPinned {
    Off,
    Programs,
    Merged,
    MergedPageable,
    All,
}

impl PrepPinned {
    /// Reads the knob's value: unset, `0` or `off` is [`PrepPinned::Off`].
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.map(str::trim) {
            None | Some("") | Some("0") | Some("off") => Ok(Self::Off),
            Some("programs") => Ok(Self::Programs),
            Some("merged") => Ok(Self::Merged),
            Some("merged-pageable") => Ok(Self::MergedPageable),
            Some("all") => Ok(Self::All),
            Some(other) => Err(format!(
                "{KNOB}={other}: expected off, programs, merged, merged-pageable or all"
            )),
        }
    }

    /// The process's setting, read once. An unreadable value is an error here
    /// and is read as off by the builds ([`take`]); the tree refuses it at its
    /// start ([`Session::start`]).
    pub fn get() -> Result<Self, String> {
        static MODE: OnceLock<Result<PrepPinned, String>> = OnceLock::new();
        MODE.get_or_init(|| Self::parse(std::env::var(KNOB).ok().as_deref()))
            .clone()
    }

    pub fn word(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Programs => "programs",
            Self::Merged => "merged",
            Self::MergedPageable => "merged-pageable",
            Self::All => "all",
        }
    }

    fn locks_programs(self) -> bool {
        matches!(self, Self::Programs | Self::All)
    }

    fn lays_merged(self) -> bool {
        matches!(self, Self::Merged | Self::MergedPageable | Self::All)
    }

    fn locks_merged(self) -> bool {
        matches!(self, Self::Merged | Self::All)
    }
}

/// A leaf's program as the tree holds it: the helper keeps one alive until the
/// leaf's build has taken its preparation.
pub(crate) trait HoldsProgram: Send + Sync + 'static {
    fn program(&self) -> &LfmProgram;
}

/// One leaf's preparation. Dropping it undoes its registrations first, then
/// frees the merged groups and lets go of the program.
pub(crate) struct Prepared {
    key: usize,
    locked: Vec<multilinear::pinned::InPlace>,
    merged: Option<(MergePlan, Vec<ColumnGroup>)>,
    _program: Arc<dyn HoldsProgram>,
}

impl Prepared {
    /// Whether merged groups were laid out at all.
    pub(crate) fn has_merged(&self) -> bool {
        self.merged.is_some()
    }

    /// The merged groups, when they were laid out under `plan`.
    pub(crate) fn merged_for(&self, plan: &MergePlan) -> Option<&[ColumnGroup]> {
        match &self.merged {
            Some((p, groups)) if p == plan => Some(groups),
            _ => None,
        }
    }
}

impl Drop for Prepared {
    fn drop(&mut self) {
        for r in self.locked.drain(..) {
            multilinear::pinned::unregister_in_place(r);
        }
    }
}

/// What the helper did, for the tree's summary line.
#[derive(Clone, Copy, Debug, Default)]
struct Tally {
    leaves: usize,
    late: usize,
    locked_groups: usize,
    locked_bytes: u64,
    refused_pieces: usize,
    lock_seconds: f64,
    merged_bytes: u64,
    merged_seconds: f64,
    taken: usize,
    merged_used: usize,
    plan_mismatch: usize,
}

struct State {
    ready: Vec<Prepared>,
    /// Programs whose build has started: a preparation that finishes after it
    /// is dropped at once.
    built: Vec<usize>,
    tally: Tally,
}

static STATE: Mutex<State> = Mutex::new(State {
    ready: Vec::new(),
    built: Vec::new(),
    tally: Tally {
        leaves: 0,
        late: 0,
        locked_groups: 0,
        locked_bytes: 0,
        refused_pieces: 0,
        lock_seconds: 0.0,
        merged_bytes: 0,
        merged_seconds: 0.0,
        taken: 0,
        merged_used: 0,
        plan_mismatch: 0,
    },
});

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn key_of(program: &LfmProgram) -> usize {
    program as *const LfmProgram as usize
}

fn group_bytes(g: &ColumnGroup) -> usize {
    std::mem::size_of_val(g.data.as_slice())
}

/// The program's own groups a build uploads through the staging buffers: the
/// slots no merged AIR takes, at least [`MIN_LOCK_BYTES`]. A split `LFM_HASH`
/// is built from chunks at build time, so its slot is not one of them.
fn program_groups_to_lock<'p>(program: &'p LfmProgram, plan: &MergePlan) -> Vec<&'p ColumnGroup> {
    super::registry::program_groups(program)
        .into_iter()
        .enumerate()
        .filter(|(i, _)| super::merge::group_of(plan, *i).is_none())
        .filter(|(i, _)| !(*i == super::airs::HASH_SLOT && program.hash_chunk_count() > 1))
        .map(|(_, g)| g)
        .filter(|g| g.padded_rows > 0 && g.width > 0 && group_bytes(g) >= MIN_LOCK_BYTES)
        .collect()
}

/// Prepares one leaf's program under `mode` (on the helper thread).
fn prepare(holder: Arc<dyn HoldsProgram>, options: &ProofOptions, mode: PrepPinned) {
    let program = holder.program();
    let key = key_of(program);
    if state().built.contains(&key) {
        state().tally.late += 1;
        return;
    }
    let plan = super::registry::merge_plan_for(program, options);
    let mut locked = Vec::new();
    let (mut locked_groups, mut locked_bytes, mut refused, mut lock_seconds) = (0, 0u64, 0, 0.0);
    let mut lock = |g: &ColumnGroup, locked: &mut Vec<multilinear::pinned::InPlace>| {
        let r = multilinear::pinned::register_in_place(g.data.as_ptr().cast(), group_bytes(g));
        locked_groups += 1;
        locked_bytes += r.bytes;
        refused += r.refused;
        lock_seconds += r.seconds;
        locked.push(r);
    };
    if mode.locks_programs() {
        for g in program_groups_to_lock(program, &plan) {
            lock(g, &mut locked);
        }
    }
    let (mut merged_bytes, mut merged_seconds) = (0u64, 0.0);
    let merged = mode.lays_merged().then(|| {
        let t = Instant::now();
        let groups = super::merge::merged_groups(program, &plan);
        merged_seconds = t.elapsed().as_secs_f64();
        merged_bytes = groups.iter().map(|g| group_bytes(g) as u64).sum();
        if mode.locks_merged() {
            for g in groups.iter().filter(|g| group_bytes(g) >= MIN_LOCK_BYTES) {
                lock(g, &mut locked);
            }
        }
        (plan, groups)
    });
    let prepared = Prepared {
        key,
        locked,
        merged,
        _program: holder,
    };
    let mut s = state();
    s.tally.leaves += 1;
    s.tally.locked_groups += locked_groups;
    s.tally.locked_bytes += locked_bytes;
    s.tally.refused_pieces += refused;
    s.tally.lock_seconds += lock_seconds;
    s.tally.merged_bytes += merged_bytes;
    s.tally.merged_seconds += merged_seconds;
    if s.built.contains(&key) {
        s.tally.late += 1;
        drop(s);
        drop(prepared);
    } else {
        s.ready.push(prepared);
    }
}

/// What a build takes before it lays out its merged groups: this program's
/// preparation, if the helper made one, and nothing ever again for it. `None`
/// at once when the knob is off.
pub(crate) fn take(program: &LfmProgram) -> Option<Prepared> {
    take_in(program, PrepPinned::get().unwrap_or(PrepPinned::Off))
}

fn take_in(program: &LfmProgram, mode: PrepPinned) -> Option<Prepared> {
    if mode == PrepPinned::Off {
        return None;
    }
    let key = key_of(program);
    let mut s = state();
    s.built.push(key);
    let at = s.ready.iter().position(|p| p.key == key)?;
    s.tally.taken += 1;
    Some(s.ready.swap_remove(at))
}

/// Counts how a build used what it took.
pub(crate) fn note_use(used_merged: bool, had_merged: bool) {
    let mut s = state();
    if used_merged {
        s.tally.merged_used += 1;
    } else if had_merged {
        s.tally.plan_mismatch += 1;
    }
}

/// Hands a build's preparation to the release thread, which undoes its
/// registrations off the builder's path. Inline when there is no session.
pub(crate) fn release(prepared: Option<Prepared>) {
    let Some(p) = prepared else {
        return;
    };
    let tx = RELEASE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    match tx {
        Some(tx) => {
            if let Err(e) = tx.send(p) {
                drop(e.0);
            }
        }
        None => drop(p),
    }
}

static RELEASE: Mutex<Option<Sender<Prepared>>> = Mutex::new(None);

/// One tree's helper and release threads, from the tree's start to its end.
/// Dropping it joins both and undoes every registration still held.
pub(crate) struct Session {
    mode: PrepPinned,
    tx: Option<Sender<Arc<dyn HoldsProgram>>>,
    helper: Option<std::thread::JoinHandle<()>>,
    releaser: Option<std::thread::JoinHandle<()>>,
    started: Instant,
}

impl Session {
    /// The session for this process's setting: inert when off; an error when
    /// the knob holds a value it does not read.
    pub(crate) fn start() -> Result<Self, String> {
        let mode = PrepPinned::get()?;
        let mut session = Self {
            mode,
            tx: None,
            helper: None,
            releaser: None,
            started: Instant::now(),
        };
        if mode == PrepPinned::Off {
            return Ok(session);
        }
        {
            let mut s = state();
            s.ready.clear();
            s.built.clear();
            s.tally = Tally::default();
        }
        multilinear::pinned::reset_in_place_stats();
        let (tx, rx): (Sender<Arc<dyn HoldsProgram>>, Receiver<_>) = channel();
        let helper = std::thread::Builder::new()
            .name("lfm-prep-pinned".into())
            .spawn(move || {
                let options = super::proof::aggregation_wrap_options();
                while let Ok(holder) = rx.recv() {
                    prepare(holder, &options, mode);
                }
            })
            .map_err(|e| format!("{KNOB}: the helper thread did not start: {e}"))?;
        let (rtx, rrx): (Sender<Prepared>, Receiver<Prepared>) = channel();
        let releaser = std::thread::Builder::new()
            .name("lfm-prep-release".into())
            .spawn(move || {
                while let Ok(p) = rrx.recv() {
                    drop(p);
                }
            })
            .map_err(|e| format!("{KNOB}: the release thread did not start: {e}"))?;
        *RELEASE.lock().unwrap_or_else(|e| e.into_inner()) = Some(rtx);
        session.tx = Some(tx);
        session.helper = Some(helper);
        session.releaser = Some(releaser);
        Ok(session)
    }

    /// Where the tree sends each leaf's program as soon as it is emitted;
    /// `None` when the knob is off.
    pub(crate) fn sender(&self) -> Option<Sender<Arc<dyn HoldsProgram>>> {
        self.tx.clone()
    }

    /// Ends the session (as its drop does) and says what it did: the line the
    /// tree prints, with the page-locked pool's state beside it.
    pub(crate) fn finish(mut self) -> String {
        self.close();
        let t = state().tally;
        let r = multilinear::pinned::in_place_stats();
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let pool = match multilinear::pinned::stats() {
            Some(p) => format!(
                "pinned pool on: {:.2} GiB, {:.2} locked in {:.3}s · uploads {:.2} GiB, {:.2} from the pool",
                gib(p.pool_bytes),
                gib(p.locked_bytes),
                p.lock_seconds,
                gib(p.upload_bytes),
                gib(p.upload_pinned),
            ),
            None => "pinned pool off".to_string(),
        };
        format!(
            "W3 PREP PINNED: {} · leaves prepared {} ({} late) · locked {} groups {:.3} GiB in {:.3}s ({:.3} s/GiB), {} pieces refused · merged laid out {:.3} GiB in {:.3}s · builds took {} (merged used {}, plan mismatch {}) · direct uploads {:.3} GiB in {} calls · unregistered in {:.3}s · {} pieces left · {pool} · session {:.2}s",
            self.mode.word(),
            t.leaves,
            t.late,
            t.locked_groups,
            gib(t.locked_bytes),
            t.lock_seconds,
            t.lock_seconds / gib(t.locked_bytes).max(1e-9),
            t.refused_pieces,
            gib(t.merged_bytes),
            t.merged_seconds,
            t.taken,
            t.merged_used,
            t.plan_mismatch,
            gib(r.direct_bytes),
            r.direct_calls,
            r.unregister_seconds,
            r.pieces_now,
            self.started.elapsed().as_secs_f64(),
        )
    }

    fn close(&mut self) {
        self.tx = None;
        if let Some(h) = self.helper.take() {
            let _ = h.join();
        }
        // Whatever no build took: released now, before the programs go.
        let left: Vec<Prepared> = std::mem::take(&mut state().ready);
        drop(left);
        let rtx = RELEASE.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(rtx);
        if let Some(h) = self.releaser.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_knob_reads_its_five_words_and_refuses_others() {
        assert_eq!(PrepPinned::parse(None), Ok(PrepPinned::Off));
        assert_eq!(PrepPinned::parse(Some("0")), Ok(PrepPinned::Off));
        assert_eq!(PrepPinned::parse(Some("off")), Ok(PrepPinned::Off));
        assert_eq!(
            PrepPinned::parse(Some(" programs ")),
            Ok(PrepPinned::Programs)
        );
        assert_eq!(PrepPinned::parse(Some("merged")), Ok(PrepPinned::Merged));
        assert_eq!(
            PrepPinned::parse(Some("merged-pageable")),
            Ok(PrepPinned::MergedPageable)
        );
        assert_eq!(PrepPinned::parse(Some("all")), Ok(PrepPinned::All));
        assert!(PrepPinned::parse(Some("1")).is_err(), "no bare on");
        assert!(PrepPinned::parse(Some("yes")).is_err());
        for m in [
            PrepPinned::Off,
            PrepPinned::Programs,
            PrepPinned::Merged,
            PrepPinned::MergedPageable,
            PrepPinned::All,
        ] {
            assert_eq!(
                PrepPinned::parse(Some(m.word())),
                Ok(m),
                "the word reads back"
            );
        }
    }

    use super::super::builder::LfmBuilder;
    use super::super::compiler::compile;
    use crate::tables::types::FE;
    use stark::proof::options::GoldilocksCubicProofOptions;

    /// The module's state is process-wide: one test at a time touches it.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn reset() {
        let mut s = state();
        s.ready.clear();
        s.built.clear();
        s.tally = Tally::default();
    }

    struct Held(LfmProgram);
    impl HoldsProgram for Held {
        fn program(&self) -> &LfmProgram {
            &self.0
        }
    }

    fn options(merge: bool) -> ProofOptions {
        let mut o = GoldilocksCubicProofOptions::with_blowup(2).expect("options");
        o.format.merge_equal_height_tables = merge;
        o
    }

    /// Base ALU, a select, an unpack and a pack, one hint: the four merge
    /// candidates at one height, so they merge (merge_tests' program).
    fn program() -> LfmProgram {
        let mut b = LfmBuilder::new();
        let arena = b.declare_arena(1);
        let h0 = b.hint_felt(arena, 0);
        let x = b.felt_const(FE::from(7u64));
        let y = b.felt_const(FE::from(5u64));
        let s = b.add(x, y);
        let d = b.sub(s, h0);
        let bit = b.bit_const(true);
        let (l, r) = b.select(bit, x.as_cell(), y.as_cell());
        let da = b.digest_const([1u64, 2, 3, 4].map(FE::from));
        let db = b.digest_const([5u64, 6, 7, 8].map(FE::from));
        let dc = b.compress(da, db);
        let lanes = b.unpack(dc.as_cell());
        let packed = b.pack_word(lanes);
        b.public(packed);
        b.public(l);
        b.public(r);
        b.public(d.as_cell());
        compile(b.finish())
    }

    /// The groups a build lays out beside the base are the build's own: the
    /// same merged groups, and the same artifacts (roots, program id) as a
    /// build that lays them out itself.
    #[test]
    fn a_prepared_leaf_builds_the_same_artifacts_as_a_fresh_one() {
        let _g = lock();
        reset();
        let opts = options(true);
        let held = Arc::new(Held(program()));
        let p = held.program();
        let plan = super::super::registry::merge_plan_for(p, &opts);
        assert!(!plan.is_empty(), "the fixture merges");
        prepare(held.clone(), &opts, PrepPinned::MergedPageable);
        let prepared = take_in(p, PrepPinned::MergedPageable).expect("prepared before its build");
        let fresh = super::super::merge::merged_groups(p, &plan);
        let ready = prepared
            .merged_for(&plan)
            .expect("laid out under this plan");
        assert_eq!(ready.len(), fresh.len());
        for (a, b) in ready.iter().zip(&fresh) {
            assert_eq!(
                (a.width, a.padded_rows, &a.data),
                (b.width, b.padded_rows, &b.data)
            );
        }
        let hasher = super::super::registry::build_artifacts(p, &opts).hasher;
        let with = super::super::registry::build_artifacts_merged_under::<
            crate::hash_pin::BlockStarkHash,
        >(p, &opts, hasher, &plan, ready);
        let without = super::super::registry::build_artifacts_merged_under::<
            crate::hash_pin::BlockStarkHash,
        >(p, &opts, hasher, &plan, &fresh);
        assert_eq!(with.program_id, without.program_id);
        assert_eq!(with.roots, without.roots);
        assert!(
            take_in(p, PrepPinned::MergedPageable).is_none(),
            "taken once"
        );
        drop(prepared);
        assert_eq!(
            Arc::strong_count(&held),
            1,
            "the preparation let go of the program"
        );
    }

    /// Merged groups laid out under another plan are never used.
    #[test]
    fn merged_groups_under_another_plan_are_not_used() {
        let _g = lock();
        reset();
        let held = Arc::new(Held(program()));
        let p = held.program();
        prepare(held.clone(), &options(true), PrepPinned::MergedPageable);
        let prepared = take_in(p, PrepPinned::MergedPageable).expect("prepared");
        let other = super::super::registry::merge_plan_for(p, &options(false));
        assert!(other.is_empty());
        assert!(prepared.has_merged());
        assert!(prepared.merged_for(&other).is_none());
    }

    /// A preparation that finishes after its build started is dropped, never
    /// handed to anyone; and with the knob off nothing is taken or recorded.
    #[test]
    fn a_late_preparation_is_dropped_and_off_takes_nothing() {
        let _g = lock();
        reset();
        let held = Arc::new(Held(program()));
        let p = held.program();
        assert!(take_in(p, PrepPinned::Off).is_none());
        assert!(state().built.is_empty(), "off records nothing");
        assert!(
            take_in(p, PrepPinned::MergedPageable).is_none(),
            "nothing yet"
        );
        prepare(held.clone(), &options(true), PrepPinned::MergedPageable);
        assert!(state().ready.is_empty());
        assert_eq!(state().tally.late, 1);
        assert_eq!(Arc::strong_count(&held), 1);
    }

    #[test]
    fn each_mode_prepares_only_its_own_part() {
        use PrepPinned::*;
        let row = |m: PrepPinned| (m.locks_programs(), m.lays_merged(), m.locks_merged());
        assert_eq!(row(Off), (false, false, false));
        assert_eq!(row(Programs), (true, false, false));
        assert_eq!(row(Merged), (false, true, true));
        assert_eq!(row(MergedPageable), (false, true, false));
        assert_eq!(row(All), (true, true, true));
    }
}

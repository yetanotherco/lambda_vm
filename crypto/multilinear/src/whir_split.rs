//! The WHIR base's stage breakdown, behind `LAMBDA_VM_BASE_SPLIT=1`.
//!
//! # Why this exists, and why the knob is not a new name
//!
//! The WHIR base proves the block's epochs and prints **one number** for all of
//! them — `base (WHIR): 15 epochs in 84.5s`, 57% of the block's wall with
//! nothing under it. There is not one timer, span or print between
//! `multilinear_continuation::prove_continuation` and the bottom of the chain,
//! so every optimisation round so far has moved that number without anyone
//! being able to say which part of it moved.
//!
//! The STARK base has had a breakdown for months, under `LAMBDA_VM_BASE_SPLIT`.
//! The LFM tree launcher exports that knob on the **WHIR** arm too — "byte
//! identical to the D-S exports" — and it reached nothing at all, because the
//! WHIR base does not go through `continuation::prove_continuation`. An inert
//! knob printed as if it mattered is worse than a missing one: the export is
//! the evidence a reader uses to believe the breakdown was taken.
//!
//! ⇒ this module reuses **the same knob name and the same line format**, so one
//! name means one thing on both pipelines.
//!
//! # Why it lives in `multilinear` and not beside the STARK instrument
//!
//! Not taste — reachability. The stages span four places: the producer
//! (`prover::continuation`), the epoch prover and the global stage
//! (`prover::multilinear_continuation`), the argument and the openings
//! (`stark::multilinear_table`), and the harness that reads them back. `prover`
//! depends on `stark`, `stark` depends on `multilinear`, and `multilinear`
//! depends on neither. This crate is the only one all four can reach, so the
//! helpers the STARK instrument keeps private to `prover::continuation` are not
//! an option for the two lower layers, whatever their visibility.
//!
//! # Phase walls and per-epoch records
//!
//! Two kinds of number, and mixing them is the way to misread the table:
//!
//! - **Stage walls** are measured on the thread that runs the stage, at stage
//!   boundaries. Within one thread they do not overlap and they **partition**
//!   that thread's epoch wall.
//! - **The producer and the prover run concurrently**, so their two sums do
//!   NOT add up to the base wall and must never be added. What the pair says is
//!   which of the two set the wall — see `handoff` in
//!   `prover::continuation::for_each_epoch`.
//!
//! # Cost when disabled
//!
//! [`enabled`] is a `OnceLock<bool>` load and a predictable branch; [`mark`]
//! returns `None` and **no clock is read**, so a disabled run pays no
//! `Instant::now` at all. Every call site is at stage granularity — about a
//! dozen per epoch against a ~5.6 s epoch — so even enabled it is far below the
//! noise of what it measures.
//!
//! # ⚠ One prove at a time
//!
//! The inner slots are process-global, because they are written inside
//! `multi_prove` and read one layer up. Two `multi_prove` calls in flight would
//! mix their numbers. In the base that cannot happen — there is a single prover
//! thread — but the **same process** later proves the LFM tree with several
//! workers at once, so rather than assume the base is the only writer,
//! [`begin_prove`] counts concurrent proves and a record that saw one carries
//! `OVERLAPPED`. A mixed reading says so instead of looking clean.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// `LAMBDA_VM_BASE_SPLIT=1` (any non-empty value other than `0`) turns the
/// lines and the records on.
///
/// The same spelling as `prover::continuation`'s own gate, deliberately: the
/// two are read in different crates and must not be able to disagree about what
/// the knob means.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("LAMBDA_VM_BASE_SPLIT") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    })
}

/// Unix epoch seconds, for aligning a stage with an external GPU sampler.
///
/// A duplicate of `stark::prove_split::epoch_secs` by necessity, not by
/// oversight: `multilinear` cannot reach `stark`. Kept byte-identical in
/// behaviour so a stamp from either instrument lands on the same timeline.
pub fn epoch_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

/// Start a timed stage — `None`, and no clock read, when the knob is off.
#[inline]
pub fn mark() -> Option<(Instant, f64)> {
    enabled().then(|| (Instant::now(), epoch_secs()))
}

/// Close a stage opened by [`mark`], print its line, and return its seconds.
///
/// The line is the STARK base's format, unchanged:
/// `BASE EPOCH {index}: {stage} {secs:.2}s t=[{t0:.3},{t1:.3}]`.
///
/// The two wall-clock stamps are what let an external GPU sampler be sliced by
/// stage; the duration alone cannot place the stage on the sampler's timeline.
#[inline]
pub fn stage_done(index: u64, stage: &str, open: Option<(Instant, f64)>) -> f64 {
    match open {
        Some((start, t0)) => {
            let secs = start.elapsed().as_secs_f64();
            // ⛔ The cross-epoch stage's index is `u64::MAX`, and printing it
            // raw put `BASE EPOCH 18446744073709551615` in the log — read back
            // from a real run, not imagined. It is unreadable and it defeats a
            // parse that expects a small integer, so the sentinel is rendered
            // by NAME. The record keeps the sentinel; only the line differs.
            let who = if index == GLOBAL_INDEX {
                "global".to_string()
            } else {
                index.to_string()
            };
            println!(
                "BASE EPOCH {who}: {stage} {secs:.2}s t=[{t0:.3},{:.3}]",
                epoch_secs()
            );
            secs
        }
        None => 0.0,
    }
}

// ── the inner slots: written inside `multi_prove`, read one layer up ────────

/// A nanosecond accumulator. Public so call sites name their slot as a constant
/// rather than passing an index.
#[derive(Debug)]
pub struct Slot(AtomicU64);

impl Slot {
    const fn new() -> Self {
        Self(AtomicU64::new(0))
    }
    /// Read and CLEAR. Reading a slot clears it, so a caller that drops the
    /// value silently would hand this epoch's time to the next one.
    fn take(&self) -> f64 {
        self.0.swap(0, Ordering::Relaxed) as f64 / 1e9
    }
}

/// `absorb_roots_and_challenge`: the roots into the transcript and the
/// challenge out. Host.
pub static CHALLENGE: Slot = Slot::new();
/// The per-table argument: LogUp, the GKR input layer, the GKR prove, the
/// zerocheck and the constraint core. **Serial over the epoch's tables** — the
/// loop has no rayon and no `k` — so this is both a sum and a wall.
pub static ARGUE: Slot = Slot::new();
/// The per-GROUP WHIR openings. Two per epoch: `epoch_groups(n)` is
/// `[n - 1, 1]`, so every epoch pays two full chains.
pub static OPEN_GROUPS: Slot = Slot::new();
/// The out-of-band DECODE opening, one more chain on top of the two.
pub static OPEN_PREPARED: Slot = Slot::new();

// ── inside an opening: the WHIR chain's round loop ──────────────────────────
//
// `open_groups` is 43.8% of the base and 24.9% of the block's whole wall, and
// it was one number. These six PARTITION the round (`whir_chain.rs:715-795`),
// so `open_groups - Σ(six)` is loop overhead and nothing else.
//
// ⛔ They are taken at the END OF THE GROUP LOOP, before the prepared opening
// runs. The prepared chain writes the same slots, so a take placed after it
// would fold DECODE's chain into `open_groups` and arm E would close on a
// number that is not what it claims.

/// All THREE 20-bit grinds a round pays: folding, out-of-domain, query.
pub static GRIND: Slot = Slot::new();
/// `factors.rounds` — the opening sumcheck.
pub static SUMCHECK: Slot = Slot::new();
/// `fold_held` — the codeword folded in place where it lies.
pub static FOLD: Slot = Slot::new();
/// The FRESH Merkle commit of the successor, once per round, plus its root
/// absorb; or the final-value path on the last round.
pub static COMMIT_FOLDED: Slot = Slot::new();
/// Out-of-domain: the sampled point, `evaluate_message`, `add_scaled_eq`.
/// Its grind is in [`GRIND`], not here.
pub static OOD: Slot = Slot::new();
/// The query openings — `whir_round::prove` / `final_openings`, which REBUILD
/// the Merkle tree on device per query batch.
pub static QUERIES: Slot = Slot::new();

/// The six, as taken at the group-loop boundary, awaiting the record.
static CHAIN_AT_GROUPS: Mutex<[f64; 6]> = Mutex::new([0.0; 6]);

/// Park the six for the record being built one layer up.
pub fn note_chain(chain: [f64; 6]) {
    if !enabled() {
        return;
    }
    if let Ok(mut held) = CHAIN_AT_GROUPS.lock() {
        *held = chain;
    }
}

/// Read and clear the six chain slots, in record order.
pub fn take_chain() -> [f64; 6] {
    [
        GRIND.take(),
        SUMCHECK.take(),
        FOLD.take(),
        COMMIT_FOLDED.take(),
        OOD.take(),
        QUERIES.take(),
    ]
}

/// Close a region opened by [`mark`] into `slot`, returning its seconds.
///
/// It returns the value rather than making the caller read the clock again:
/// a second `elapsed()` for the same region measures a longer one, and the two
/// numbers would then disagree by the cost of the instrument itself.
#[inline]
pub fn add(slot: &Slot, start: Option<(Instant, f64)>) -> f64 {
    match start {
        Some((t, _)) => {
            let nanos = t.elapsed().as_nanos() as u64;
            slot.0.fetch_add(nanos, Ordering::Relaxed);
            nanos as f64 / 1e9
        }
        None => 0.0,
    }
}

/// How many groups the last argument opened, so the record carries the count
/// rather than a reader assuming `epoch_groups`' shape held.
static GROUPS: AtomicUsize = AtomicUsize::new(0);
/// The slowest single table of the current argument, by INDEX into the epoch's
/// table order.
static MAX_TABLE: Mutex<Option<(usize, f64)>> = Mutex::new(None);
/// The epoch's table names, in that same order.
///
/// ⓘ Set one layer up, because the committed table does not carry a name — it
/// is a layout and a trace. The index is what the argument can observe; the
/// name is what a reader can act on, and only the caller holding the AIRs has
/// it.
static TABLE_NAMES: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Note one table's argument time, keeping the maximum by index.
///
/// ⛔ The SUM alone cannot choose a lever here. `argue` being large is
/// consistent with fifty even tables (where parallelism is the answer) and with
/// one dominating table (where it is not), and those want opposite fixes.
#[inline]
pub fn note_table(index: usize, secs: f64) {
    if !enabled() {
        return;
    }
    if let Ok(mut held) = MAX_TABLE.lock()
        && held.as_ref().is_none_or(|(_, best)| secs > *best)
    {
        *held = Some((index, secs));
    }
}

/// Name the epoch's tables, in the order the argument walks them.
pub fn set_table_names(names: Vec<String>) {
    if !enabled() {
        return;
    }
    if let Ok(mut held) = TABLE_NAMES.lock() {
        *held = names;
    }
}

/// Note how many groups this argument opened.
#[inline]
pub fn note_groups(n: usize) {
    if enabled() {
        GROUPS.store(n, Ordering::Relaxed);
    }
}

// ── the overlap falsifier ───────────────────────────────────────────────────

/// Proves inside `multi_prove` right now.
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
/// Set once if two proves were ever concurrent. Never cleared: one mixed
/// reading taints every later record, because a slot it polluted is only zeroed
/// by the take that reports it.
static OVERLAPPED: AtomicUsize = AtomicUsize::new(0);

/// What [`begin_prove`] hands back.
///
/// ⛔ It releases the count on DROP, not on the success path. `multi_prove` has
/// `?` early-returns, and a release that only ran when the prove succeeded
/// would leave the count stuck at one forever — every later record would then
/// be stamped `OVERLAPPED` by a prove that FAILED rather than by two that
/// overlapped. A false alarm on a falsifier is worse than no falsifier, because
/// it reads as evidence.
#[derive(Debug)]
pub struct ProveGuard(());

impl Drop for ProveGuard {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Open a prove. Cheap and inert when the knob is off.
pub fn begin_prove() -> Option<ProveGuard> {
    if !enabled() {
        return None;
    }
    if IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1 > 1 {
        OVERLAPPED.store(1, Ordering::Relaxed);
    }
    Some(ProveGuard(()))
}

// ── the per-epoch records ───────────────────────────────────────────────────

/// One epoch's producer-side stages. The four **partition** `wall`, so the only
/// thing that can break `execute + collect + build + handoff == wall` is a
/// stage whose timer is missing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProducerSplit {
    pub index: u64,
    pub execute: f64,
    pub collect: f64,
    pub build: f64,
    /// The blocking hand-off to the prover over an unbuffered channel. **This
    /// is the backpressure**: large means the prover is the bottleneck, ~0
    /// means the producer is.
    pub handoff: f64,
    pub wall: f64,
}

impl ProducerSplit {
    /// What the four stages leave over. Named rather than left implicit — an
    /// unattributed remainder is how a phase hides.
    pub fn other(&self) -> f64 {
        self.wall - self.execute - self.collect - self.build - self.handoff
    }
}

/// One epoch's prover-side stages, and the argument's own split inside `prove`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProverSplit {
    /// The epoch index, or [`GLOBAL_INDEX`] for the cross-epoch stage.
    pub index: u64,
    pub prep: f64,
    pub absorb: f64,
    pub commit: f64,
    pub prove: f64,
    pub wall: f64,
    pub challenge: f64,
    pub argue: f64,
    pub open_groups: f64,
    pub open_prepared: f64,
    pub groups: usize,
    pub max_table: Option<(String, f64)>,
    pub airs: usize,
    pub overlapped: bool,
    /// The chain's six, for the GROUP openings only, in the order
    /// [`take_chain`] returns them: grind, sumcheck, fold, commit_folded, ood,
    /// queries. The prepared opening keeps its wall and no breakdown — it is
    /// 2.5% of the base, and six more fields would not move a ranking.
    pub chain: [f64; 6],
}

/// The six chain slots' names, in record order — so a message can name the one
/// that went missing instead of printing an index.
pub const CHAIN_NAMES: [&str; 6] = [
    "grind",
    "sumcheck",
    "fold",
    "commit_folded",
    "ood",
    "queries",
];

/// The index the cross-epoch global stage records under. It is the last thing
/// the base does and it is INSIDE the base's wall, so it belongs in the table —
/// but it is not an epoch and must not be averaged with them.
pub const GLOBAL_INDEX: u64 = u64::MAX;

impl ProverSplit {
    /// What the four stages leave over.
    pub fn other(&self) -> f64 {
        self.wall - self.prep - self.absorb - self.commit - self.prove
    }
    /// What the four inner slots leave over inside `prove`.
    pub fn prove_other(&self) -> f64 {
        self.prove - self.challenge - self.argue - self.open_groups - self.open_prepared
    }
    /// What the six chain slots leave over inside `open_groups` — the round
    /// loop's own overhead, and nothing else.
    pub fn chain_other(&self) -> f64 {
        self.open_groups - self.chain.iter().sum::<f64>()
    }
    pub fn is_global(&self) -> bool {
        self.index == GLOBAL_INDEX
    }
}

static PRODUCER: Mutex<Vec<ProducerSplit>> = Mutex::new(Vec::new());
static PROVER: Mutex<Vec<ProverSplit>> = Mutex::new(Vec::new());

/// Record one epoch's producer stages.
pub fn push_producer(rec: ProducerSplit) {
    if !enabled() {
        return;
    }
    if let Ok(mut held) = PRODUCER.lock() {
        held.push(rec);
    }
}

/// Close the prover's stages into a record: take the inner slots, print the
/// line, and store it.
///
/// The line and the record come from the SAME values, so the shell's parse and
/// the harness's table cannot disagree about a number they both report.
pub fn push_prover(mut rec: ProverSplit) {
    if !enabled() {
        return;
    }
    rec.challenge = CHALLENGE.take();
    rec.argue = ARGUE.take();
    rec.open_groups = OPEN_GROUPS.take();
    rec.open_prepared = OPEN_PREPARED.take();
    rec.groups = GROUPS.swap(0, Ordering::Relaxed);
    // ⓘ NOT taken here. The six are read at the end of the GROUP loop, where
    // the prepared opening has not run yet; by this point they would carry
    // DECODE's chain too. `rec.chain` is already filled by the caller.
    // Anything still in them is the prepared opening's, and clearing it keeps
    // it out of the next epoch.
    let _ = take_chain();
    rec.chain = CHAIN_AT_GROUPS
        .lock()
        .map(|mut h| std::mem::replace(&mut *h, [0.0; 6]))
        .unwrap_or([0.0; 6]);
    let names = TABLE_NAMES
        .lock()
        .map(|mut h| std::mem::take(&mut *h))
        .unwrap_or_default();
    rec.max_table = MAX_TABLE
        .lock()
        .ok()
        .and_then(|mut h| h.take())
        .map(|(at, secs)| {
            let name = names
                .get(at)
                .cloned()
                .unwrap_or_else(|| format!("table#{at}"));
            (name, secs)
        });
    rec.overlapped = OVERLAPPED.load(Ordering::Relaxed) != 0;

    let who = if rec.is_global() {
        "GLOBAL (in base)".to_string()
    } else {
        format!("#{}", rec.index)
    };
    let (max_name, max_secs) = rec
        .max_table
        .clone()
        .unwrap_or_else(|| ("-".to_string(), 0.0));
    println!(
        "WHIR PROVE SPLIT {who}{tainted}: airs {airs} · wall {wall:.2}s · \
         prep {prep:.2} · absorb {absorb:.3} · commit {commit:.2} · \
         prove {prove:.2} · other {other:.2} || inside[Σ] challenge {challenge:.3} · \
         argue {argue:.2} (max {max_name} {max_secs:.2}) · \
         open_groups {open_groups:.2} ({groups} groups) · \
         open_prepared {open_prepared:.2} · other {prove_other:.2} || \
         chain[Σ groups] grind {c0:.2} · sumcheck {c1:.2} · fold {c2:.2} · \
         commit_folded {c3:.2} · ood {c4:.2} · queries {c5:.2} · other {c6:.2}",
        tainted = if rec.overlapped { " ⛔OVERLAPPED" } else { "" },
        airs = rec.airs,
        wall = rec.wall,
        prep = rec.prep,
        absorb = rec.absorb,
        commit = rec.commit,
        prove = rec.prove,
        other = rec.other(),
        challenge = rec.challenge,
        argue = rec.argue,
        open_groups = rec.open_groups,
        groups = rec.groups,
        open_prepared = rec.open_prepared,
        prove_other = rec.prove_other(),
        c0 = rec.chain[0],
        c1 = rec.chain[1],
        c2 = rec.chain[2],
        c3 = rec.chain[3],
        c4 = rec.chain[4],
        c5 = rec.chain[5],
        c6 = rec.chain_other(),
    );

    if let Ok(mut held) = PROVER.lock() {
        held.push(rec);
    }
}

/// Take every record collected so far, clearing the stores.
///
/// Clearing is what keeps a later phase — the LFM tree proves in the same
/// process — from being read as part of the base.
pub fn drain() -> (Vec<ProducerSplit>, Vec<ProverSplit>) {
    let producer = PRODUCER
        .lock()
        .map(|mut h| std::mem::take(&mut *h))
        .unwrap_or_default();
    let prover = PROVER
        .lock()
        .map(|mut h| std::mem::take(&mut *h))
        .unwrap_or_default();
    (producer, prover)
}

/// Does the breakdown close? `Ok(())`, or the first failure spelled out.
///
/// ★ A PURE FUNCTION OVER THE RECORDS, deliberately. The arms it runs are the
/// only reason the table can be quoted, so they have to be testable against
/// MANUFACTURED states — a state they must accept and states they must refuse —
/// rather than only against whatever a real run happens to produce. A check
/// that has never been seen to fail is not evidence.
///
/// ⛔ IT DOES NOT CHECK `Σ producer + Σ prover == base`. The two run on
/// different threads at once, so that identity is false on a CORRECT
/// instrument: the base wall is epoch 0's preparation plus the proofs plus the
/// pipeline's waiting, while the sum double-counts every overlap. Asserting it
/// would redden the honest path, and a check that reddens honestly gets its
/// tolerance widened until it cannot fail at all. Arm D asserts the two
/// INEQUALITIES that are actually true of a two-thread pipeline.
pub fn check_closure(
    producer: &[ProducerSplit],
    prover: &[ProverSplit],
    base_secs: f64,
    tol: f64,
) -> Result<(), String> {
    for r in prover {
        let who = if r.is_global() {
            "global".to_string()
        } else {
            format!("epoch {}", r.index)
        };
        // Arm A: prep + absorb + commit + prove partition the prover's wall.
        if r.other().abs() > tol * r.wall.max(1e-9) {
            return Err(format!(
                "arm A: {who}'s prover stages do not close — wall {:.3}s but \
                 prep+absorb+commit+prove = {:.3}s, leaving {:.3}s ({:.1}%) \
                 unattributed. A stage's timer is missing.",
                r.wall,
                r.wall - r.other(),
                r.other(),
                100.0 * r.other() / r.wall.max(1e-9),
            ));
        }
        // Arm B: the four inner slots partition `prove`.
        if r.prove_other().abs() > tol * r.prove.max(1e-9) {
            return Err(format!(
                "arm B: {who}'s argument does not close — prove {:.3}s but \
                 challenge+argue+open_groups+open_prepared = {:.3}s, leaving \
                 {:.3}s ({:.1}%) unattributed inside `prove`.",
                r.prove,
                r.prove - r.prove_other(),
                r.prove_other(),
                100.0 * r.prove_other() / r.prove.max(1e-9),
            ));
        }
        // Arm E: the six chain slots partition `open_groups`, one level below
        // arm B. `open_groups` is the largest single term in the base, and
        // until 2b it was one number.
        if r.chain_other().abs() > tol * r.open_groups.max(1e-9) {
            let named: Vec<String> = CHAIN_NAMES
                .iter()
                .zip(r.chain.iter())
                .map(|(n, v)| format!("{n} {v:.3}"))
                .collect();
            return Err(format!(
                "arm E: {who}'s chain does not close — open_groups {:.3}s but \
                 the six sum to {:.3}s, leaving {:.3}s ({:.1}%) unattributed \
                 inside the round loop{}. Slots: {}.",
                r.open_groups,
                r.open_groups - r.chain_other(),
                r.chain_other(),
                100.0 * r.chain_other() / r.open_groups.max(1e-9),
                // ⛔ A NEGATIVE remainder is not "a bit of drift": it means the
                // slots hold time from outside the window, so the sum is of
                // parts that are not parts. It gets its own words, because
                // reading it as a small overshoot is how it survives.
                if r.chain_other() < 0.0 {
                    " — NEGATIVE, so the slots carry time from OUTSIDE the \
                     group loop and the window is not what it claims"
                } else {
                    ""
                },
                named.join(" · "),
            ));
        }
    }
    // Arm C: execute + collect + build + handoff partition the producer's wall.
    for r in producer {
        if r.other().abs() > tol * r.wall.max(1e-9) {
            return Err(format!(
                "arm C: epoch {}'s producer stages do not close — wall {:.3}s \
                 but execute+collect+build+handoff = {:.3}s, leaving {:.3}s \
                 ({:.1}%) unattributed. A stage's timer is missing.",
                r.index,
                r.wall,
                r.wall - r.other(),
                r.other(),
                100.0 * r.other() / r.wall.max(1e-9),
            ));
        }
    }
    // Arm D: the two inequalities a two-thread pipeline really satisfies.
    let p_wall: f64 = producer.iter().map(|r| r.wall).sum();
    let v_wall: f64 = prover.iter().map(|r| r.wall).sum();
    let busiest = p_wall.max(v_wall);
    if busiest > base_secs * (1.0 + tol) {
        return Err(format!(
            "arm D: the busiest thread ({busiest:.2}s) exceeds the base wall \
             ({base_secs:.2}s) — a thread cannot take longer than the pipeline \
             that contains it.",
        ));
    }
    if base_secs > (p_wall + v_wall) * (1.0 + tol) {
        return Err(format!(
            "arm D: the base wall ({base_secs:.2}s) exceeds the serial bound \
             ({:.2}s) — the pipeline took longer than running every stage one \
             after another, so time is being spent outside every stage.",
            p_wall + v_wall,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⛔ The disabled path must read no clock. Asserted through the only
    /// observable it has: [`mark`] returns `None`, so [`add`] cannot move a
    /// slot and [`stage_done`] cannot print.
    ///
    /// The knob is a process-wide `OnceLock` and the test binary does not set
    /// it, so this is the state every other test in the crate runs under.
    #[test]
    fn disabled_is_inert() {
        assert!(
            !enabled(),
            "the test binary must not set LAMBDA_VM_BASE_SPLIT"
        );
        assert!(mark().is_none(), "a disabled mark must read no clock");
        assert_eq!(
            add(&ARGUE, mark()),
            0.0,
            "a disabled add must not move a slot"
        );
        assert_eq!(ARGUE.take(), 0.0, "a disabled add must not move a slot");
        assert_eq!(stage_done(0, "execute", mark()), 0.0);
        note_table(3, 9.0);
        set_table_names(vec!["KECCAK".to_string()]);
        note_groups(7);
        push_producer(ProducerSplit {
            index: 0,
            wall: 1.0,
            ..Default::default()
        });
        push_prover(ProverSplit {
            index: 0,
            wall: 1.0,
            ..Default::default()
        });
        let (producer, prover) = drain();
        assert!(
            producer.is_empty() && prover.is_empty(),
            "disabled records nothing"
        );
    }

    /// The remainders are what catch a missing timer, so they must be the
    /// arithmetic they claim and not a restatement of it.
    #[test]
    fn a_missing_stage_shows_up_in_the_remainder() {
        let whole = ProducerSplit {
            index: 0,
            execute: 1.0,
            collect: 2.0,
            build: 3.0,
            handoff: 4.0,
            wall: 10.0,
        };
        assert!(
            whole.other().abs() < 1e-9,
            "a complete partition leaves nothing over"
        );

        // The mutation the gate runs: one stage's timer omitted. The stage
        // reads 0 and its time lands in the remainder, where the check sees it.
        let missing = ProducerSplit {
            build: 0.0,
            ..whole.clone()
        };
        assert!(
            (missing.other() - 3.0).abs() < 1e-9,
            "an omitted `build` must surface as 3.0s of remainder, got {}",
            missing.other()
        );

        let p = ProverSplit {
            index: 0,
            prep: 1.0,
            absorb: 0.5,
            commit: 2.0,
            prove: 6.5,
            wall: 10.0,
            challenge: 0.5,
            argue: 4.0,
            open_groups: 1.5,
            open_prepared: 0.5,
            ..Default::default()
        };
        assert!(p.other().abs() < 1e-9);
        assert!(p.prove_other().abs() < 1e-9);
        assert!(!p.is_global());
        assert!(
            ProverSplit {
                index: GLOBAL_INDEX,
                ..Default::default()
            }
            .is_global()
        );
    }

    /// ⛔ The cross-epoch stage's index is `u64::MAX`. A real run printed
    /// `BASE EPOCH 18446744073709551615` before this was fixed, so the
    /// rendering is pinned rather than left to a reader to notice again.
    #[test]
    fn the_global_sentinel_is_rendered_by_name() {
        assert_eq!(GLOBAL_INDEX, u64::MAX);
        assert!(
            !format!("{GLOBAL_INDEX}").contains("global"),
            "the raw sentinel is what the line must NOT carry",
        );
        assert!(
            ProverSplit {
                index: GLOBAL_INDEX,
                ..Default::default()
            }
            .is_global()
        );
        assert!(
            !ProverSplit {
                index: 0,
                ..Default::default()
            }
            .is_global()
        );
    }

    /// A realistic pair of records: the producer and the prover each close, and
    /// the two threads overlap, so `Σ producer + Σ prover` is well above the
    /// base wall and arm D's inequalities are the only true statements about it.
    fn honest() -> (Vec<ProducerSplit>, Vec<ProverSplit>, f64) {
        let producer: Vec<_> = (0..3)
            .map(|i| ProducerSplit {
                index: i,
                execute: 1.0,
                collect: 0.5,
                build: 1.5,
                handoff: 2.0,
                wall: 5.0,
            })
            .collect();
        let prover: Vec<_> = (0..3)
            .map(|i| ProverSplit {
                index: i,
                prep: 0.5,
                absorb: 0.1,
                commit: 1.4,
                prove: 3.0,
                wall: 5.0,
                challenge: 0.1,
                argue: 1.9,
                open_groups: 0.8,
                open_prepared: 0.2,
                // The six partition `open_groups`: 0.3 + 0.2 + 0.1 + 0.1 +
                // 0.05 + 0.05 = 0.8.
                chain: [0.3, 0.2, 0.1, 0.1, 0.05, 0.05],
                ..Default::default()
            })
            .collect();
        // Two threads over three epochs: ~one epoch of preparation, then the
        // proofs. Not 30s.
        (producer, prover, 17.0)
    }

    /// ⛔ THE STATE THE CHECK EXISTS TO ACCEPT. Run first, because an arm that
    /// cannot pass is the failure mode that costs a box launch.
    #[test]
    fn closure_accepts_an_honest_run() {
        let (producer, prover, base) = honest();
        assert_eq!(check_closure(&producer, &prover, base, 0.03), Ok(()));
    }

    /// ★ AND THE STATES IT MUST REFUSE, one per arm, each a MANUFACTURED
    /// omission of exactly the kind the box mutation will make. Each assertion
    /// reads the arm's NAME out of the message: an arm that reddened for some
    /// other reason would not be evidence that this arm works.
    #[test]
    fn closure_refuses_every_manufactured_omission() {
        // Arm A: one prover stage's timer omitted. Its time becomes remainder.
        let (producer, mut prover, base) = honest();
        prover[1].commit = 0.0;
        let err = check_closure(&producer, &prover, base, 0.03).unwrap_err();
        assert!(err.starts_with("arm A:"), "expected arm A, got: {err}");
        assert!(err.contains("epoch 1"), "arm A must name the epoch: {err}");

        // Arm A must name the GLOBAL stage as `global`, not as an epoch index.
        let (producer, mut prover, base) = honest();
        prover.push(ProverSplit {
            index: GLOBAL_INDEX,
            prep: 0.1,
            prove: 0.4,
            wall: 2.0,
            ..Default::default()
        });
        let err = check_closure(&producer, &prover, base, 0.03).unwrap_err();
        assert!(err.starts_with("arm A:"), "expected arm A, got: {err}");
        assert!(
            err.contains("global"),
            "arm A must name the global stage: {err}"
        );

        // Arm B: one INNER slot omitted. The four stages still close, so only
        // arm B can catch it — which is why arm B exists.
        let (producer, mut prover, base) = honest();
        prover[2].argue = 0.0;
        let err = check_closure(&producer, &prover, base, 0.03).unwrap_err();
        assert!(err.starts_with("arm B:"), "expected arm B, got: {err}");
        assert!(err.contains("epoch 2"), "arm B must name the epoch: {err}");

        // Arm C: one producer stage's timer omitted.
        let (mut producer, prover, base) = honest();
        producer[0].handoff = 0.0;
        let err = check_closure(&producer, &prover, base, 0.03).unwrap_err();
        assert!(err.starts_with("arm C:"), "expected arm C, got: {err}");
        assert!(err.contains("epoch 0"), "arm C must name the epoch: {err}");

        // Arm E: one of the SIX chain slots omitted. Arms A and B still close
        // — `open_groups` is unchanged and so is `prove` — so arm E is the
        // only arm that can see it, which is the whole reason it exists.
        let (producer, mut prover, base) = honest();
        prover[1].chain[5] = 0.0;
        let err = check_closure(&producer, &prover, base, 0.03).unwrap_err();
        assert!(err.starts_with("arm E:"), "expected arm E, got: {err}");
        assert!(err.contains("epoch 1"), "arm E must name the epoch: {err}");
        assert!(
            err.contains("queries 0.000"),
            "arm E must print the six by NAME so the missing one is visible: {err}",
        );

        // Arm D, first inequality: a thread claiming more than the pipeline.
        let (producer, prover, _) = honest();
        let err = check_closure(&producer, &prover, 5.0, 0.03).unwrap_err();
        assert!(err.starts_with("arm D:"), "expected arm D, got: {err}");
        assert!(err.contains("busiest thread"), "{err}");

        // Arm D, second: a base wall beyond the serial bound — time spent
        // outside every stage.
        let (producer, prover, _) = honest();
        let err = check_closure(&producer, &prover, 100.0, 0.03).unwrap_err();
        assert!(err.starts_with("arm D:"), "expected arm D, got: {err}");
        assert!(err.contains("serial bound"), "{err}");
    }

    /// ⛔ THE SECOND MUTATION THE GATE RUNS: the tolerance zeroed must redden a
    /// run that is merely REALISTIC rather than exact. A real stage sum is
    /// strictly below its wall — the wall's own clock reads bracket the stage
    /// reads — so a check that still passed at tolerance zero would be reading
    /// numbers that cannot have come from a measurement.
    #[test]
    fn a_zero_tolerance_refuses_a_realistic_run() {
        let (mut producer, prover, base) = honest();
        // The instrument's own cost: a real wall is a hair longer than the sum
        // of its parts, because the wall's clock reads bracket the stages'.
        producer[0].wall += 0.002;
        assert_eq!(check_closure(&producer, &prover, base, 0.03), Ok(()));

        // Arm C alone, with no prover records to reach first: the 2 ms of
        // instrument cost is what a zeroed tolerance refuses.
        let err = check_closure(&producer, &[], base, 0.0).unwrap_err();
        assert!(
            err.starts_with("arm C:"),
            "expected arm C at tol 0, got: {err}"
        );
        assert!(err.contains("epoch 0"), "{err}");

        // And on the WHOLE record set a zeroed tolerance still reddens — the
        // arm that fires first is whichever record is walked first, which is
        // why the isolated check above is the one that names arm C.
        assert!(
            check_closure(&producer, &prover, base, 0.0).is_err(),
            "a zeroed tolerance must refuse a run carrying real timer cost",
        );
    }
}

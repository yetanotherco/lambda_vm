//! What running a recursion tree needs besides its programs: proving a level's
//! proofs a bounded number at once and handing them back in index order
//! ([`in_index_order`]), the sibling counts ([`tree_siblings`],
//! [`tree_siblings_l0`]), a program's census panel ([`census_panel`]), a
//! prove's split ([`prove_split_text`]), and the host's memory as a tree run
//! reports it ([`HostSampler`], [`rss_marks`], [`cgroup_limit_gib`]).
//!
//! The whole-block driver ([`super::block_tree`]) and the tree suites
//! (`per_table_aggregator_tests`, `block_tree_tests`) share them, so a harness
//! and the CLI read one knob and print one line.

use super::compiler::LfmProgram;

/// Run `task` over `0..n` on `workers` threads, and return the results **in
/// index order** whatever order they finished in.
///
/// ★★★ THE ORDER IS THE SOUNDNESS PROPERTY, not a convenience. A level's
/// children, layouts and label runs are three parallel vectors, and a node at
/// the level above takes a contiguous SUBSLICE of each — which is exactly what
/// makes contiguity across sibling subtrees a consequence of the label pins
/// rather than a check of its own. Drain them in completion order and the pins
/// still verify, one subtree at a time, while the tree they describe is not the
/// tree that was built. ⇒ results land in per-index slots and are drained by
/// index, so nothing downstream can observe that a scheduler ran at all.
///
/// `workers <= 1` runs `task` inline, on this thread, in order: the control arm
/// is the original path and not this function with one worker.
///
/// # Panics
///
/// Re-raises the FIRST worker panic on the caller's thread, payload intact.
/// ⚠ `std::thread::scope` otherwise propagates with the fixed string "a scoped
/// thread panicked", which names neither the cause nor its location, and
/// libtest's global hook files a spawned thread's own message against no test
/// and drops it on the floor. The prover's `run_admitted` learned that the
/// expensive way — eleven anonymous failures in one suite run.
pub(crate) fn in_index_order<T: Send>(
    n: usize,
    workers: usize,
    task: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let slots: Vec<std::sync::Mutex<Option<T>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    if workers <= 1 {
        for (j, slot) in slots.iter().enumerate() {
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(task(j));
        }
    } else {
        let cursor = std::sync::atomic::AtomicUsize::new(0);
        let first_panic: std::sync::Mutex<Option<Box<dyn std::any::Any + Send>>> =
            std::sync::Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                let (cursor, slots, first_panic, task) = (&cursor, &slots, &first_panic, &task);
                scope.spawn(move || {
                    // ⛔ THIS THREAD IS PART OF THIS LEVEL. Without the enrolment
                    // its artifact builds are not counted and the level reports
                    // fewer proofs than it made.
                    let _enrolled = super::program_census::enrol();
                    loop {
                        let j = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if j >= slots.len() {
                            break;
                        }
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| task(j))) {
                            Ok(out) => {
                                *slots[j].lock().unwrap_or_else(|e| e.into_inner()) = Some(out);
                            }
                            Err(payload) => {
                                let mut first =
                                    first_panic.lock().unwrap_or_else(|e| e.into_inner());
                                if first.is_none() {
                                    *first = Some(payload);
                                }
                                break;
                            }
                        }
                    }
                });
            }
        });
        if let Some(payload) = first_panic.into_inner().unwrap_or_else(|e| e.into_inner()) {
            std::panic::resume_unwind(payload);
        }
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(j, slot)| {
            // Every index ran (a panicking task was re-raised above).
            slot.into_inner()
                .unwrap_or_else(|e| e.into_inner())
                .unwrap_or_else(|| panic!("index {j} produced no result"))
        })
        .collect()
}

/// How many proofs of one INTERIOR level run at once: `LFM_TREE_SIBLINGS`, or
/// `LFM_TREE_K` for the same thing ([`resolve_siblings`]; unset is 1, the
/// serial control).
pub(crate) fn tree_siblings() -> Result<usize, String> {
    let k = std::env::var("LFM_TREE_K").ok();
    let s = std::env::var("LFM_TREE_SIBLINGS").ok();
    resolve_siblings(k.as_deref(), s.as_deref())
}

/// How many proofs of level 0 run at once: `LFM_TREE_SIBLINGS_L0`, or
/// `LFM_TREE_K_L0` ([`resolve_siblings`]; unset is 1). A knob of its own: level
/// 0's falsifier is the host peak, the interior's the card.
pub(crate) fn tree_siblings_l0() -> Result<usize, String> {
    let k = std::env::var("LFM_TREE_K_L0").ok();
    let s = std::env::var("LFM_TREE_SIBLINGS_L0").ok();
    resolve_siblings(k.as_deref(), s.as_deref())
}

/// [`tree_siblings`] with the environment supplied, so the resolution is
/// testable without mutating process state.
///
/// An empty value reads as unset — `FOO= cmd` is the shell clearing a variable,
/// and failing a run for that spelling of "default" helps nobody. Two values
/// for the one knob refuse: the launch line then names no one experiment.
pub(crate) fn resolve_siblings(k: Option<&str>, siblings: Option<&str>) -> Result<usize, String> {
    fn clean(v: Option<&str>) -> Option<&str> {
        v.filter(|v| !v.is_empty())
    }
    let (k, siblings) = (clean(k), clean(siblings));
    let named = match (k, siblings) {
        (None, None) => return Ok(1),
        (Some(a), Some(b)) => {
            if a != b {
                return Err(format!(
                    "LFM_TREE_K=`{a}` and LFM_TREE_SIBLINGS=`{b}` are the same knob \
                     set to two values, so the launch line does not name one \
                     experiment. Set one of them"
                ));
            }
            a
        }
        (Some(v), None) | (None, Some(v)) => v,
    };
    let n: usize = named
        .parse()
        .map_err(|_| format!("the sibling count must be a positive integer, got `{named}`"))?;
    if n < 1 {
        return Err(format!("the sibling count must be at least 1, got {n}"));
    }
    Ok(n)
}

/// A program's census and chip padding panel, as the text its caller prints
/// in one write ([`census_panel_text`]), with the cells and instructions.
///
/// The panel is a FORWARD instrument, not a diagnostic: printed from the working
/// fan-in-2 configuration it named `LFM_HASH`, and `LFM_HASH` is the table that
/// stepped 2^20 → 2^21 and put fan-in 3 over the card at 25.95 GiB of ~26.2
/// usable. Print it at EVERY level.
pub(crate) fn census_panel(
    program: &LfmProgram,
    label: &str,
    fan_in: usize,
) -> (u64, usize, String) {
    let (main, aux) = super::airs::lfm_cell_counts_with_hasher(
        program,
        program.hasher(crate::hash_pin::BLOCK_HASHER),
    );
    let cells = main + 3 * aux;
    let panel = super::airs::lfm_chip_census_with_hasher(
        program,
        program.hasher(crate::hash_pin::BLOCK_HASHER),
    );
    let text = census_panel_text(label, cells, program.instrs.len(), &panel, fan_in);
    (cells, program.instrs.len(), text)
}

/// A program's census line, its chip panel and its step line, as the ONE string
/// [`census_panel`] returns.
///
/// One `print!` holds the stdout lock for the whole panel. Printed a line at a
/// time, the panels of proofs census'd at once (level 0's wraps in flight,
/// sibling nodes, the level pool) interleaved LINE BY LINE with each other and
/// with any other thread's output, and a chip line carries no label: a reader
/// of the log could only tell whose line it was from the numbers on it. The
/// bytes are exactly what the per-line `println!`s wrote
/// ([`the_census_panel_is_the_per_line_bytes`]), so every reader of the panel
/// parses it unchanged.
pub(crate) fn census_panel_text(
    label: &str,
    cells: u64,
    instrs: usize,
    panel: &[super::airs::LfmChipCells],
    fan_in: usize,
) -> String {
    use std::fmt::Write as _;
    const EMPTY_MACHINE_CELLS: u64 = 26_482_828;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "   ★ CENSUS {label}: {cells} cells ({} instructions), floor {:.1}%",
        instrs,
        100.0 * EMPTY_MACHINE_CELLS as f64 / cells as f64,
    );
    let step = (fan_in + 1) as f64 / fan_in as f64;
    // A split `LFM_HASH` is two census entries, printed as ONE panel line —
    // real and committed rows summed, the chunk heights in a trailing
    // `[split a+b]` — so a panel never repeats a chip name (the box reader treats a
    // repeat as two interleaved panels) and `LFM_HASH` real rows stay the
    // permutation count. Unsplit programs print exactly as before.
    let hash = super::airs::LFM_CHIP_NAMES[super::airs::HASH_SLOT];
    let hash_chunks: Vec<&super::airs::LfmChipCells> =
        panel.iter().filter(|c| c.name == hash).collect();
    for c in panel {
        if c.name == hash && hash_chunks.len() > 1 {
            if !std::ptr::eq(c, hash_chunks[0]) {
                continue;
            }
            let real: u64 = hash_chunks.iter().map(|h| h.real_rows).sum();
            let rows: u64 = hash_chunks.iter().map(|h| h.rows).sum();
            let heights: Vec<String> = hash_chunks.iter().map(|h| h.rows.to_string()).collect();
            let _ = writeln!(
                out,
                "     {:<14} {:>10}/{:>10}  headroom {:>5.1}%  AT RISK  cliff +{} cells  [split {}]",
                c.name,
                real,
                rows,
                100.0 * (rows - real) as f64 / rows as f64,
                hash_chunks.iter().map(|h| h.cliff_cost()).sum::<u64>(),
                heights.join("+"),
            );
            continue;
        }
        let _ = writeln!(
            out,
            "     {:<14} {:>10}/{:>10}  headroom {:>5.1}%  {}  cliff +{} cells",
            c.name,
            c.real_rows,
            c.rows,
            100.0 * c.headroom(),
            if c.at_risk() { "AT RISK" } else { "fixed  " },
            c.cliff_cost(),
        );
    }
    let stepping: Vec<&str> = panel
        .iter()
        .filter(|c| c.at_risk() && c.real_rows as f64 * step > c.rows as f64)
        .map(|c| c.name)
        .collect();
    let _ = writeln!(
        out,
        "     ⇒ at {step:.3}× the workload these would STEP: {stepping:?}"
    );
    out
}

/// ★ THE `prove` FIELD, SPLIT — `execute · fill · multi_prove`.
///
/// The per-node TIMING line prints the whole of `lfm_prove` as one number, and
/// the three phases inside it are three different machines: a single-threaded
/// interpreter, a parallel trace fill, and the only one that reaches the card.
/// A lever aimed at the wrong one reads zero.
///
/// ⓘ Printed by the caller beside the stage's label — a bare line is
/// unattributable the moment two proofs are in flight. `None` on a LOADED
/// stage, because nothing proved: `take_prove_split` returns `None` and there
/// is nothing to attribute. The text is the lines, each ending in `\n`.
pub(crate) fn prove_split_text(label: &str) -> Option<String> {
    use std::fmt::Write as _;
    let split = super::proof::take_prove_split()?;
    let mut out = String::new();
    {
        let _ = writeln!(
            out,
            "   {label} LFM PROVE: execute {:.2}s · fill {:.2}s · multi_prove {:.2}s{}",
            split.execute,
            split.fill,
            split.multi_prove,
            // ⓘ Absent when there was no wait, so a serial line is byte-identical
            // to every one already in the campaign's logs and the two arms diff
            // on their numbers rather than on their shape.
            if split.permit_wait > 0.0 {
                format!(" · permit wait {:.2}s", split.permit_wait)
            } else {
                String::new()
            },
        );
        // ⛔ A SECOND LINE, never extra fields on the first. Every log the
        // campaign has compared greps `LFM PROVE:` and reads its three numbers
        // positionally; widening that line would re-baseline every one of them.
        //
        // ⓘ `parallel 0 levels` beside a large `levels` is the reading this line
        // exists to make visible: it means every level was coalesced onto the
        // calling thread and the parallel path never ran, which a wall alone
        // would show only as a lever that did nothing.
        let e = split.exec;
        let _ = writeln!(
            out,
            "   {label} LFM EXEC: levels {} · parallel {} levels / {} hashes · \
             depth pass {:.2}s · setup {:.2}s · hash phase {:.2}s · apply {:.2}s · \
             residue {:.2}s · sum {:.2}s",
            e.levels,
            e.parallel_levels,
            e.parallel_hashes,
            e.depth_pass,
            e.setup,
            e.hash_phase,
            e.apply,
            e.residue,
            e.depth_pass + e.setup + e.hash_phase + e.apply + e.residue,
        );
        // ⓘ A THIRD line, for the two questions a wall cannot answer: how big
        // the thing `setup` touches is, and what width the pool actually gave.
        // `ns/perm` is measured on the coalesced levels of THIS proof — same
        // box, same load, no rayon — so `width` is a reading, not an argument.
        if e.serial_hashes > 0 {
            let _ = writeln!(
                out,
                "   {label} LFM EXEC WIDTH: records {:.0} MiB · {} serial hashes at \
                 {:.0} ns/perm · effective width {:.1}",
                e.record_bytes as f64 / (1u64 << 20) as f64,
                e.serial_hashes,
                e.secs_per_perm() * 1e9,
                e.effective_width(),
            );
        }
    }
    Some(out)
}

/// Both memory numbers at once — `(VmRSS, VmHWM)` in GiB, live and high-water.
///
/// ⚠ `VmHWM` alone cannot see a TROUGH, and the trough is half the model. A
/// high-water mark only rises, so a mark taken after a phase reports the largest
/// the process has EVER been, not what that phase left resident. That difference
/// decides whether the next phase's peak is `live + its own working set` or is
/// hidden under an earlier phase's mark entirely — and on this workload the two
/// diverge hard: box A measured `VmHWM` static at 72.50 GiB while `VmRSS`
/// oscillated between 19.98 and 58.13.
///
/// A `ps` sample of the LIVE figure once reached this lane as if it were a
/// high-water mark, and every derivation built on it was wrong by the gap
/// between them. Both are printed so that cannot recur.
pub(crate) fn rss_marks() -> (Option<f64>, Option<f64>) {
    let read = |key: &str| -> Option<f64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|l| l.starts_with(key))?;
        let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb / (1024.0 * 1024.0))
    };
    (read("VmRSS:"), read("VmHWM:"))
}

/// A 100 Hz `VmRSS` sampler with an ARGMAX TIMESTAMP.
///
/// ⛔ WHY NOT `VmHWM`. A high-water only rises, so it cannot see a peak BELOW
/// itself and cannot say WHEN its own peak happened. That is what struck the
/// 37.169 GiB "the tree does not grow per level": three reads of one monotone
/// counter, in a process that had already proved a base and four wraps, with no
/// mark before the first level — a whole-process bound that priced no level in
/// it, level 1 included, and being a within-run comparison did not rescue it.
///
/// ⇒ A sampled max carries an argmax `t`, which buys two things a high-water
/// cannot: a level's peak is the max inside ITS OWN window rather than the
/// process's, and the host peak can be checked for SIMULTANEITY against an
/// external device trace. Non-simultaneous maxima must not be summed.
pub(crate) struct HostSampler {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<(f64, f64)>>,
}

impl HostSampler {
    pub(crate) fn start() -> Self {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            let (mut peak, mut at) = (0.0f64, unix_now());
            while !flag.load(Ordering::Relaxed) {
                if let (Some(rss), _) = rss_marks()
                    && rss > peak
                {
                    peak = rss;
                    at = unix_now();
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            (peak, at)
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }

    /// `(peak GiB, argmax UNIX seconds)` over this sampler's window.
    pub(crate) fn stop(mut self) -> (f64, f64) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        // A sampler that could not run reports no peak, at the time it stopped.
        self.handle
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or_else(|| (0.0, unix_now()))
    }
}

pub(crate) fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// The cgroup memory ceiling, trying **v2 then v1**, or `None` with the reason.
///
/// ⚠ A sampler hard-coded to one layout reads NOTHING on the other box and
/// reports no error, so a percentage-of-ceiling silently becomes a percentage of
/// zero — or of a default nobody chose. Both paths are tried and a miss is
/// LOUD; the caller must not print a percentage without a ceiling.
pub(crate) fn cgroup_limit_gib() -> Result<f64, String> {
    const PATHS: [&str; 2] = [
        "/sys/fs/cgroup/memory.max",                   // v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes", // v1
    ];
    let mut tried = Vec::new();
    for p in PATHS {
        match std::fs::read_to_string(p) {
            Ok(s) if s.trim() == "max" => tried.push(format!("{p}=max (unlimited)")),
            Ok(s) => match s.trim().parse::<u64>() {
                Ok(b) => return Ok(b as f64 / (1024.0 * 1024.0 * 1024.0)),
                Err(e) => tried.push(format!("{p} unparsable: {e}")),
            },
            Err(e) => tried.push(format!("{p}: {e}")),
        }
    }
    Err(tried.join("; "))
}

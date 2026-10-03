//! `LAMBDA_VM_BLOCK_MEMLOG=1` ([`super::BlockOptions::memlog`]): where a
//! streamed block's host memory is, by term, as the data moves.
//!
//! A sampler prints a `BLOCK MEM` line every half second, and one at each
//! mark (the executor's end, the windows walked, each phase of the finish, the
//! rest laid out, each group committed and installed, phase A's end, each
//! phase-B group's end, the proof). A line carries the process's resident set
//! and minor faults, jemalloc's allocated / active / resident / mapped /
//! retained bytes (in the lib's tests, which run jemalloc, as the shipped
//! binary does), `rss − resident`, the pinned staging buffers, every term,
//! their sum (`named`) and `unnamed` = allocated − named. `rss − resident` is
//! often below zero: jemalloc counts a huge extent's pages as resident before
//! they are touched, so neither it nor `active` is a lower bound on the
//! resident set; the retention is `rss most − peak active`. Every 0.1 s the sampler reads the
//! allocator; the line at the highest `active` it saw is printed again at the
//! end with each arena's active and dirty bytes and the threads bound to it:
//! under a never-purge posture the resident set only grows, so its peak names
//! no instant, and `active`'s does.
//!
//! It reads sizes only: no trace, no transcript, no proof byte depends on it.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

use stark::multilinear_block::BlockMem;

const GIB: f64 = (1u64 << 30) as f64;

fn gib(bytes: usize) -> f64 {
    bytes as f64 / GIB
}

/// What `LAMBDA_VM_BLOCK_MEMLOG=1` turns on. Read where the options are made.
pub(crate) fn from_env() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_MEMLOG").is_ok_and(|v| v.trim() == "1")
}

/// The block's host bytes by where they are; every list at its capacity.
pub(crate) struct Ledger {
    start: Instant,
    /// The committer's terms: the group in its commit, the groups being
    /// packed, the committed tables as held, the kept tree tops.
    pub(crate) block: Arc<BlockMem>,
    /// The executor's memory and its own window of logs.
    pub(crate) exec: AtomicUsize,
    /// Windows of logs on their way to the walker or held by it.
    pub(crate) logs: AtomicUsize,
    /// The walk's carried memory state.
    pub(crate) walk: AtomicUsize,
    /// Walked windows on their way to the accumulator.
    pub(crate) walked: AtomicUsize,
    /// The builder's run so far (kept tails or windows, routed segments,
    /// what was counted ahead), as of the last window absorbed.
    pub(crate) builder: AtomicUsize,
    /// The initial memory image the builder keeps.
    pub(crate) image: AtomicUsize,
    /// Chunk jobs' ops queued for the layout thread.
    pub(crate) jobs: AtomicUsize,
    /// The chunk the layout thread is generating and laying out: its ops,
    /// then its trace.
    pub(crate) laying: AtomicUsize,
    /// The packer's open group, laid out (column-major), eight bytes a cell.
    pub(crate) open: AtomicUsize,
    /// Closed groups not yet taken by phase A's committer.
    pub(crate) sent: AtomicUsize,
    /// The finish's tables, row-major, until each is laid out.
    pub(crate) rest: AtomicUsize,
    /// The rest's tables laid out and not yet placed in a group.
    pub(crate) rest_laid: AtomicUsize,
    /// The prepared tables' columns (DECODE's, the dense genesis pages').
    pub(crate) prepared: AtomicUsize,
    peak: Mutex<Option<Peak>>,
    /// The highest resident set a sample saw, and when.
    rss_most: Mutex<(usize, f64)>,
    threads: Mutex<Vec<(String, u32)>>,
}

/// The sample at the highest `active`: its line and every arena then.
struct Peak {
    active: usize,
    at: f64,
    line: String,
    terms: Vec<(&'static str, usize)>,
    unnamed: i64,
    arenas: Vec<ArenaBytes>,
}

/// One jemalloc arena's active, dirty and muzzy bytes.
#[derive(Clone, Copy)]
#[cfg_attr(not(test), allow(dead_code))]
struct ArenaBytes {
    arena: u32,
    active: usize,
    dirty: usize,
    muzzy: usize,
}

impl Ledger {
    /// A ledger on the prove's clock `start`; the committer's terms mark
    /// their events with a line of this ledger.
    pub(crate) fn new(start: Instant) -> Arc<Self> {
        Arc::new_cyclic(|this: &Weak<Ledger>| {
            let this = this.clone();
            Ledger {
                start,
                block: Arc::new(BlockMem::new(move |label| {
                    if let Some(ledger) = this.upgrade() {
                        ledger.line(label);
                    }
                })),
                exec: AtomicUsize::new(0),
                logs: AtomicUsize::new(0),
                walk: AtomicUsize::new(0),
                walked: AtomicUsize::new(0),
                builder: AtomicUsize::new(0),
                image: AtomicUsize::new(0),
                jobs: AtomicUsize::new(0),
                laying: AtomicUsize::new(0),
                open: AtomicUsize::new(0),
                sent: AtomicUsize::new(0),
                rest: AtomicUsize::new(0),
                rest_laid: AtomicUsize::new(0),
                prepared: AtomicUsize::new(0),
                peak: Mutex::new(None),
                rss_most: Mutex::new((0, 0.0)),
                threads: Mutex::new(Vec::new()),
            }
        })
    }

    /// Every term, in a fixed order.
    pub(crate) fn terms(&self) -> Vec<(&'static str, usize)> {
        let b = &self.block;
        [
            ("exec", &self.exec),
            ("logs", &self.logs),
            ("walk", &self.walk),
            ("walked", &self.walked),
            ("builder", &self.builder),
            ("image", &self.image),
            ("jobs", &self.jobs),
            ("laying", &self.laying),
            ("open", &self.open),
            ("sent", &self.sent),
            ("rest", &self.rest),
            ("rest_laid", &self.rest_laid),
            ("prepared", &self.prepared),
            ("committing", &b.committing),
            ("ahead", &b.ahead),
            ("pack_wide", &b.packing_wide),
            ("pack_ready", &b.packed_ready),
            ("held_narrow", &b.held_narrow),
            ("held_wide", &b.held_wide),
            ("tops", &b.tree_tops),
        ]
        .into_iter()
        .map(|(name, term)| (name, term.load(Relaxed)))
        .collect()
    }

    /// The line for `label` now, and the allocator's active bytes (0 when
    /// they are not read).
    fn sample(&self, label: &str) -> (String, usize, Vec<(&'static str, usize)>, i64) {
        let t = self.start.elapsed().as_secs_f64();
        let rss = proc_rss_bytes();
        if let Some(rss) = rss
            && let Ok(mut most) = self.rss_most.lock()
            && rss > most.0
        {
            *most = (rss, t);
        }
        let terms = self.terms();
        let named: usize = terms.iter().map(|&(_, b)| b).sum();
        let heap = heap_stats();
        let unnamed = heap.map_or(0, |[allocated, ..]| allocated as i64 - named as i64);
        let signed = |b: i64| b as f64 / GIB;
        let mut line = format!(
            "BLOCK MEM {label} t={t:.2} · rss {} · minflt {} M",
            rss.map_or("n/a".to_string(), |b| format!("{:.2}", gib(b))),
            proc_minor_faults().map_or("n/a".to_string(), |f| format!("{:.2}", f as f64 / 1e6)),
        );
        match heap {
            Some([allocated, active, resident, mapped, retained]) => {
                line.push_str(&format!(
                    " · alloc {:.2} active {:.2} resident {:.2} mapped {:.2} retained {:.2} · \
                     rss − resident {:+.2} (pinned {:.2}) · named {:.2} · unnamed {:+.2}",
                    gib(allocated),
                    gib(active),
                    gib(resident),
                    gib(mapped),
                    gib(retained),
                    rss.map_or(0.0, |r| signed(r as i64 - resident as i64)),
                    gib(multilinear::gpu::pinned_host_bytes()),
                    gib(named),
                    signed(unnamed),
                ));
            }
            None => line.push_str(&format!(
                " · heap n/a · pinned {:.2} · named {:.2}",
                gib(multilinear::gpu::pinned_host_bytes()),
                gib(named),
            )),
        }
        let listed: Vec<String> = terms
            .iter()
            .map(|(name, b)| format!("{name} {:.2}", gib(*b)))
            .collect();
        line.push_str(&format!(" || {} (GiB)", listed.join(" · ")));
        let active = heap.map_or(0, |[_, active, ..]| active);
        (line, active, terms, unnamed)
    }

    /// Keeps the sample if its `active` is the highest so far.
    fn consider(
        &self,
        line: String,
        active: usize,
        terms: Vec<(&'static str, usize)>,
        unnamed: i64,
    ) {
        if active == 0 {
            return;
        }
        let Ok(mut peak) = self.peak.lock() else {
            return;
        };
        if peak.as_ref().is_some_and(|p| p.active >= active) {
            return;
        }
        *peak = Some(Peak {
            active,
            at: self.start.elapsed().as_secs_f64(),
            line,
            terms,
            unnamed,
            arenas: arena_bytes(),
        });
    }

    /// A line now, labelled.
    pub(crate) fn line(&self, label: &str) {
        let (line, active, terms, unnamed) = self.sample(label);
        eprintln!("{line}");
        self.consider(line, active, terms, unnamed);
    }

    /// A sampler's tick: the allocator read, the line printed when `print`.
    fn tick(&self, print: bool) {
        let (line, active, terms, unnamed) = self.sample("tick");
        if print {
            eprintln!("{line}");
        }
        self.consider(line, active, terms, unnamed);
    }

    /// One line of `what`'s parts, largest first, those of 0.01 GiB or more.
    pub(crate) fn parts(&self, what: &str, mut parts: Vec<(String, usize)>) {
        let total: usize = parts.iter().map(|(_, b)| b).sum();
        parts.sort_by(|a, b| b.1.cmp(&a.1));
        let listed: Vec<String> = parts
            .iter()
            .filter(|(_, b)| gib(*b) >= 0.01)
            .map(|(name, b)| format!("{name} {:.2}", gib(*b)))
            .collect();
        eprintln!(
            "BLOCK MEM {what} parts: {:.2} GiB · {}",
            gib(total),
            listed.join(" · ")
        );
    }

    /// Records the arena the calling thread allocates from, as `name`.
    pub(crate) fn thread(&self, name: &str) {
        if let (Some(arena), Ok(mut threads)) = (thread_arena(), self.threads.lock()) {
            threads.push((name.to_string(), arena));
        }
    }

    /// Records the arenas the global rayon pool's workers allocate from.
    pub(crate) fn pool(&self) {
        let arenas = rayon::broadcast(|_| thread_arena());
        if let Ok(mut threads) = self.threads.lock() {
            threads.extend(
                arenas
                    .into_iter()
                    .flatten()
                    .map(|a| ("rayon".to_string(), a)),
            );
        }
    }

    /// The peak sample again, its terms largest first, each arena's bytes at
    /// the peak and now, the threads bound to each, and the highest resident
    /// set a sample saw.
    pub(crate) fn report(&self) {
        let threads = self.threads.lock().map(|t| t.clone()).unwrap_or_default();
        let who = |arena: u32| -> String {
            let mut names: Vec<(String, usize)> = Vec::new();
            for (name, a) in &threads {
                if *a != arena {
                    continue;
                }
                match names.iter_mut().find(|(n, _)| n == name) {
                    Some((_, count)) => *count += 1,
                    None => names.push((name.clone(), 1)),
                }
            }
            let names: Vec<String> = names
                .into_iter()
                .map(|(n, c)| if c > 1 { format!("{n}×{c}") } else { n })
                .collect();
            names.join(",")
        };
        let arenas_line = |arenas: &[ArenaBytes]| -> String {
            let mut arenas = arenas.to_vec();
            arenas.sort_by(|a, b| (b.active + b.dirty).cmp(&(a.active + a.dirty)));
            let total = |f: fn(&ArenaBytes) -> usize| gib(arenas.iter().map(f).sum());
            let top: Vec<String> = arenas
                .iter()
                .take(12)
                .map(|a| {
                    format!(
                        "a{} [{}] {:.2} · {:.2} · {:.2}",
                        a.arena,
                        who(a.arena),
                        gib(a.active),
                        gib(a.dirty),
                        gib(a.muzzy)
                    )
                })
                .collect();
            format!(
                "{} arenas, Σ active {:.2} · dirty {:.2} · muzzy {:.2} | {}",
                arenas.len(),
                total(|a| a.active),
                total(|a| a.dirty),
                total(|a| a.muzzy),
                top.join(" | ")
            )
        };
        if let Ok(peak) = self.peak.lock()
            && let Some(peak) = peak.as_ref()
        {
            eprintln!(
                "BLOCK MEM PEAK active {:.2} GiB at t={:.2}: {}",
                gib(peak.active),
                peak.at,
                peak.line
            );
            let mut terms = peak.terms.clone();
            terms.sort_by(|a, b| b.1.cmp(&a.1));
            let listed: Vec<String> = terms
                .iter()
                .filter(|(_, b)| gib(*b) >= 0.01)
                .map(|(name, b)| format!("{name} {:.2}", gib(*b)))
                .collect();
            eprintln!(
                "BLOCK MEM PEAK TERMS (GiB): {} · unnamed {:+.2}",
                listed.join(" · "),
                peak.unnamed as f64 / GIB
            );
            eprintln!(
                "BLOCK MEM PEAK ARENAS (active · dirty · muzzy GiB): {}",
                arenas_line(&peak.arenas)
            );
        }
        let _ = heap_stats();
        eprintln!(
            "BLOCK MEM END ARENAS (active · dirty · muzzy GiB): {}",
            arenas_line(&arena_bytes())
        );
        let mut bound: Vec<String> = Vec::new();
        for (name, arena) in &threads {
            bound.push(format!("{name} a{arena}"));
        }
        eprintln!("BLOCK MEM THREADS: {}", bound.join(" · "));
        if let Ok(most) = self.rss_most.lock() {
            eprintln!(
                "BLOCK MEM RSS MOST: {:.2} GiB, first at t={:.2}",
                gib(most.0),
                most.1
            );
            if let Ok(peak) = self.peak.lock()
                && let Some(peak) = peak.as_ref()
                && most.0 > 0
            {
                eprintln!(
                    "BLOCK MEM RETENTION: rss most − peak active {:+.2} GiB (the allocator's freed \
                     pages kept beyond the live peak)",
                    (most.0 as f64 - peak.active as f64) / GIB
                );
            }
        }
    }
}

/// The bytes a window of logs takes, at its capacity.
pub(crate) fn logs_bytes(logs: &Vec<executor::vm::logs::Log>) -> usize {
    crate::tables::trace_builder::vec_heap_bytes(logs)
}

/// A committed table's columns as held on the host: packed, or eight bytes a
/// cell.
pub(crate) fn table_bytes(
    table: &stark::multilinear_table::CommittedTable<'_, super::F, super::E>,
) -> usize {
    match table.narrow() {
        Some(packed) => packed.data().len(),
        None => (table.num_committed_columns() << table.num_vars()) * std::mem::size_of::<u64>(),
    }
}

/// A group's tables as held.
pub(crate) fn group_bytes(
    group: &[stark::multilinear_table::CommittedTable<'_, super::F, super::E>],
) -> usize {
    group.iter().map(table_bytes).sum()
}

/// A trace table's main columns, eight bytes a cell.
pub(crate) fn rows_bytes(trace: &stark::trace::TraceTable<super::F, super::E>) -> usize {
    trace.main_table.width * trace.main_table.height * std::mem::size_of::<u64>()
}

/// The ledger's tick every 0.1 s, and its line every half second, on a
/// thread of its own until dropped.
pub(crate) struct Sampler {
    stop: Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Sampler {
    pub(crate) fn start(ledger: Arc<Ledger>) -> Self {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = stop.clone();
        let handle = std::thread::Builder::new()
            .name("block-memlog".to_string())
            .spawn(move || {
                let mut n = 0usize;
                while !flag.load(Relaxed) {
                    std::thread::park_timeout(std::time::Duration::from_millis(100));
                    if !flag.load(Relaxed) {
                        n += 1;
                        ledger.tick(n.is_multiple_of(5));
                    }
                }
            })
            .ok();
        Self { stop, handle }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop.store(true, Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

/// This process's minor page faults so far (`/proc/self/stat`, field 10), on
/// Linux: each a first touch of a page the allocator had not kept.
fn proc_minor_faults() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The command name may hold spaces; the fields after it do not.
    let after = stat.rsplit_once(')')?.1;
    after.split_whitespace().nth(7)?.parse().ok()
}

/// This process's resident set (`VmRSS`), on Linux.
fn proc_rss_bytes() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib: usize = status
        .lines()
        .find(|l| l.starts_with("VmRSS:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kib * 1024)
}

/// jemalloc's allocated, active, resident, mapped and retained bytes, with
/// the statistics refreshed: in the lib's tests, which install jemalloc as
/// the allocator (`lib.rs`).
#[cfg(test)]
fn heap_stats() -> Option<[usize; 5]> {
    use tikv_jemalloc_ctl::{epoch, stats};
    // The statistics are cached until the epoch turns.
    epoch::advance().ok()?;
    Some([
        stats::allocated::read().ok()?,
        stats::active::read().ok()?,
        stats::resident::read().ok()?,
        stats::mapped::read().ok()?,
        stats::retained::read().ok()?,
    ])
}

/// Outside the lib's tests the allocator's statistics are not read.
#[cfg(not(test))]
fn heap_stats() -> Option<[usize; 5]> {
    None
}

/// Each initialised arena's active, dirty and muzzy bytes, as of the last
/// epoch ([`heap_stats`] turns it).
#[cfg(test)]
fn arena_bytes() -> Vec<ArenaBytes> {
    use tikv_jemalloc_ctl::raw;
    // SAFETY: each key's type is jemalloc's documented one (`unsigned` for
    // `arenas.narenas`, `size_t` for the page size and the page counts).
    let (narenas, page) = unsafe {
        (
            raw::read::<u32>(b"arenas.narenas\0").unwrap_or(0),
            raw::read::<usize>(b"arenas.page\0").unwrap_or(4096),
        )
    };
    (0..narenas)
        .filter_map(|arena| {
            let pages = |what: &str| -> Option<usize> {
                let key = format!("stats.arenas.{arena}.{what}\0");
                // SAFETY: the page counts are `size_t`.
                unsafe { raw::read::<usize>(key.as_bytes()).ok() }
            };
            Some(ArenaBytes {
                arena,
                active: pages("pactive")? * page,
                dirty: pages("pdirty")? * page,
                muzzy: pages("pmuzzy")? * page,
            })
        })
        .filter(|a| a.active + a.dirty + a.muzzy > 0)
        .collect()
}

#[cfg(not(test))]
fn arena_bytes() -> Vec<ArenaBytes> {
    Vec::new()
}

/// The arena the calling thread allocates from.
#[cfg(test)]
fn thread_arena() -> Option<u32> {
    // SAFETY: `thread.arena` is an `unsigned`.
    unsafe { tikv_jemalloc_ctl::raw::read::<u32>(b"thread.arena\0").ok() }
}

#[cfg(not(test))]
fn thread_arena() -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A line carries the allocator's statistics and every term, and the
    /// peak keeps the highest `active` seen.
    #[test]
    fn a_line_names_every_term_and_the_peak_is_kept() {
        let ledger = Ledger::new(Instant::now());
        ledger.thread("test");
        ledger.open.store(3 << 30, Relaxed);
        let (line, active, terms, unnamed) = ledger.sample("test");
        assert!(
            active > 0,
            "jemalloc's statistics are read in the lib's tests"
        );
        assert!(
            line.contains("alloc ") && line.contains("unnamed "),
            "{line}"
        );
        assert!(line.contains("open 3.00"), "{line}");
        assert_eq!(terms.len(), 20);
        assert!(
            unnamed < 0,
            "a term larger than the heap leaves unnamed below zero"
        );
        ledger.consider(line, active, terms.clone(), unnamed);
        ledger.consider("lower".to_string(), 1, terms, unnamed);
        {
            let peak = ledger.peak.lock().expect("the peak");
            let peak = peak.as_ref().expect("a peak");
            assert_eq!(peak.active, active);
            assert!(peak.line.contains("open 3.00"));
            assert!(!peak.arenas.is_empty(), "the arenas are read");
        }
        assert!(!ledger.threads.lock().expect("threads").is_empty());
        ledger.report();
    }
}

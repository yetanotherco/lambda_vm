//! The head of a streamed phase A, stamped (`LAMBDA_VM_BLOCK_HEAD_LOG=1`):
//! for the run's first windows, when the pass took each from the executor,
//! passed it and handed it out, when a walker walked it (and how much of that
//! went on seeding its entries) and when it was absorbed; for the first streamed chunks, when each was handed out, laid
//! out and placed; for every streamed chunk, the layout's seconds by table;
//! and when the card's early setup ended (`LAMBDA_VM_BLOCK_CARD_EARLY`).
//! Seconds since the prove started, printed as phase A ends. A measurement:
//! the proof is the same, and with the knob off nothing is recorded.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::Instant;

use crate::Error;

pub(crate) const HEAD_LOG_ENV: &str = "LAMBDA_VM_BLOCK_HEAD_LOG";

/// The windows and chunks stamped one by one; past them a chunk counts only
/// in its table's sums.
const WINDOWS: usize = 48;
const CHUNKS: usize = 128;

static ON: AtomicBool = AtomicBool::new(false);
static LOG: Mutex<Option<Log>> = Mutex::new(None);

/// [`HEAD_LOG_ENV`]: unset, empty or `0` is off, `1` on; anything else
/// refuses the prove.
pub(crate) fn knob() -> Result<bool, Error> {
    parse(std::env::var(HEAD_LOG_ENV).ok().as_deref())
}

fn parse(value: Option<&str>) -> Result<bool, Error> {
    match value.map(str::trim) {
        None | Some("" | "0") => Ok(false),
        Some("1") => Ok(true),
        Some(v) => Err(Error::Prover(format!(
            "{HEAD_LOG_ENV}={v}: 1 (phase A's head stamped) or 0"
        ))),
    }
}

/// A window of the run, as the head saw it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Window {
    entries: usize,
    received: f64,
    passed: f64,
    handed: f64,
    walk_from: f64,
    seed: f64,
    walked: f64,
    absorbed: f64,
}

/// A streamed chunk, as the head saw it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Chunk {
    table: &'static str,
    index: usize,
    rows: usize,
    handed: f64,
    laid_from: f64,
    laid: f64,
    placed: f64,
}

/// What a prove's head recorded.
#[derive(Debug)]
struct Log {
    start: Instant,
    windows: Vec<Window>,
    chunks: Vec<Chunk>,
    /// Every chunk's layout, by table: its chunks, their seconds and the
    /// longest.
    tables: Vec<(&'static str, usize, f64, f64)>,
    card_ready: Option<f64>,
}

impl Log {
    fn new(start: Instant) -> Self {
        Self {
            start,
            windows: Vec::new(),
            chunks: Vec::new(),
            tables: Vec::new(),
            card_ready: None,
        }
    }

    fn at(&self, t: Instant) -> f64 {
        t.saturating_duration_since(self.start).as_secs_f64()
    }

    fn window(&mut self, k: usize) -> Option<&mut Window> {
        if k >= WINDOWS {
            return None;
        }
        if self.windows.len() <= k {
            self.windows.resize(k + 1, Window::default());
        }
        self.windows.get_mut(k)
    }

    fn chunk(&mut self, seq: usize) -> Option<&mut Chunk> {
        if seq >= CHUNKS {
            return None;
        }
        if self.chunks.len() <= seq {
            self.chunks.resize(seq + 1, Chunk::default());
        }
        self.chunks.get_mut(seq)
    }

    fn window_handed(&mut self, k: usize, entries: usize, received: f64, passed: f64, now: f64) {
        if let Some(w) = self.window(k) {
            w.entries = entries;
            w.received = received;
            w.passed = passed;
            w.handed = now;
        }
    }

    fn window_walked(&mut self, k: usize, from: f64, seed: f64, now: f64) {
        if let Some(w) = self.window(k) {
            w.walk_from = from;
            w.seed = seed;
            w.walked = now;
        }
    }

    fn window_absorbed(&mut self, k: usize, now: f64) {
        if let Some(w) = self.window(k) {
            w.absorbed = now;
        }
    }

    fn chunk_handed(
        &mut self,
        seq: usize,
        table: &'static str,
        index: usize,
        rows: usize,
        now: f64,
    ) {
        if let Some(c) = self.chunk(seq) {
            c.table = table;
            c.index = index;
            c.rows = rows;
            c.handed = now;
        }
    }

    fn chunk_laid(&mut self, seq: usize, table: &'static str, from: f64, now: f64) {
        let secs = (now - from).max(0.0);
        match self.tables.iter_mut().find(|(name, ..)| *name == table) {
            Some((_, n, sum, most)) => {
                *n += 1;
                *sum += secs;
                *most = most.max(secs);
            }
            None => self.tables.push((table, 1, secs, secs)),
        }
        if let Some(c) = self.chunk(seq) {
            c.laid_from = from;
            c.laid = now;
        }
    }

    fn chunk_placed(&mut self, seq: usize, now: f64) {
        if let Some(c) = self.chunk(seq) {
            c.placed = now;
        }
    }

    /// The `BLOCK HEAD` lines: the windows, the chunks, the layout by table
    /// and the card's setup.
    fn lines(&self) -> Vec<String> {
        let windows: Vec<String> = self
            .windows
            .iter()
            .enumerate()
            .map(|(k, w)| {
                format!(
                    "{k} {} r{:.3} p{:.3} h{:.3} w{:.3}-{:.3} s{:.3} a{:.3}",
                    w.entries,
                    w.received,
                    w.passed,
                    w.handed,
                    w.walk_from,
                    w.walked,
                    w.seed,
                    w.absorbed
                )
            })
            .collect();
        let chunks: Vec<String> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(seq, c)| {
                format!(
                    "{seq} {}[{}] {} h{:.3} l{:.3}-{:.3} p{:.3}",
                    c.table, c.index, c.rows, c.handed, c.laid_from, c.laid, c.placed
                )
            })
            .collect();
        let tables: Vec<String> = self
            .tables
            .iter()
            .map(|(name, n, sum, most)| {
                format!("{name} {n} in {sum:.2} s, the longest {most:.3} s")
            })
            .collect();
        vec![
            format!(
                "BLOCK HEAD WINDOWS: k entries r<from the executor> p<passed> h<handed out> w<walked from>-<to> s<seeding> a<absorbed> (s since the prove started) · {}",
                windows.join(" · ")
            ),
            format!(
                "BLOCK HEAD CHUNKS: seq table[index] rows h<handed out> l<laid out from>-<to> p<placed> · {}",
                chunks.join(" · ")
            ),
            format!(
                "BLOCK HEAD TABLES: every streamed chunk's layout by table · {}",
                tables.join(" · ")
            ),
            match self.card_ready {
                Some(at) => format!("BLOCK HEAD CARD: set up at {at:.3} s on a thread of its own"),
                None => "BLOCK HEAD CARD: set up at its first use".to_string(),
            },
        ]
    }
}

/// Starts a prove's head: recording when `on`, nothing (and what an earlier
/// prove left cleared) when not.
pub(crate) fn begin(start: Instant, on: bool) {
    if let Ok(mut log) = LOG.lock() {
        *log = on.then(|| Log::new(start));
    }
    ON.store(on, Relaxed);
}

fn with(f: impl FnOnce(&mut Log, f64)) {
    if !ON.load(Relaxed) {
        return;
    }
    if let Ok(mut log) = LOG.lock()
        && let Some(log) = log.as_mut()
    {
        let now = log.at(Instant::now());
        f(log, now);
    }
}

/// The pass took window `k` from the executor at `received`, passed it by
/// `passed` and handed it out now, with its entries.
pub(crate) fn window_handed(k: usize, entries: usize, received: Instant, passed: Instant) {
    with(|log, now| {
        let (received, passed) = (log.at(received), log.at(passed));
        log.window_handed(k, entries, received, passed, now)
    });
}

/// A walker walked window `k` from `from`, `seed` seconds of it seeding.
pub(crate) fn window_walked(k: usize, from: Instant, seed: f64) {
    with(|log, now| {
        let from = log.at(from);
        log.window_walked(k, from, seed, now)
    });
}

pub(crate) fn window_absorbed(k: usize) {
    with(|log, now| log.window_absorbed(k, now));
}

/// Streamed chunk `seq` (in hand-out order), chunk `index` of `table`, with
/// `rows` rows before padding, handed to the layout.
pub(crate) fn chunk_handed(seq: usize, table: &'static str, index: usize, rows: usize) {
    with(|log, now| log.chunk_handed(seq, table, index, rows, now));
}

/// Streamed chunk `seq`, of `table`, laid out from `from` until now.
pub(crate) fn chunk_laid(seq: usize, table: &'static str, from: Instant) {
    with(|log, now| {
        let from = log.at(from);
        log.chunk_laid(seq, table, from, now)
    });
}

pub(crate) fn chunk_placed(seq: usize) {
    with(|log, now| log.chunk_placed(seq, now));
}

/// The card's early setup ended.
pub(crate) fn card_ready() {
    with(|log, now| log.card_ready = Some(now));
}

/// Stops recording and hands back the `BLOCK HEAD` lines; none when the head
/// was not recorded.
pub(crate) fn take_lines() -> Vec<String> {
    ON.store(false, Relaxed);
    LOG.lock()
        .ok()
        .and_then(|mut log| log.take())
        .map(|log| log.lines())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_knob_is_off_unless_its_variable_is_one() {
        assert!(!parse(None).unwrap());
        assert!(!parse(Some("")).unwrap());
        assert!(!parse(Some(" 0 ")).unwrap());
        assert!(parse(Some("1")).unwrap());
        assert!(parse(Some(" 1")).unwrap());
        for v in ["2", "on", "true", "01"] {
            assert!(parse(Some(v)).is_err(), "{v} must refuse the prove");
        }
    }

    #[test]
    fn the_head_lines_name_each_window_and_chunk_and_the_layout_by_table() {
        let mut log = Log::new(Instant::now());
        log.window_handed(0, 1_234, 0.12, 0.14, 0.15);
        log.window_walked(0, 0.16, 0.05, 0.40);
        log.window_absorbed(0, 0.52);
        log.window_handed(2, 99, 0.29, 0.30, 0.31);
        log.chunk_handed(0, "CPU", 0, 2_097_152, 0.60);
        log.chunk_laid(0, "CPU", 0.61, 0.62);
        log.chunk_placed(0, 0.63);
        log.chunk_handed(1, "MEMW", 0, 1_000, 0.70);
        log.chunk_laid(1, "MEMW", 0.71, 1.21);
        log.chunk_laid(1, "MEMW", 1.30, 1.50);
        log.chunk_placed(1, 1.22);
        log.card_ready = Some(0.33);
        let lines = log.lines();
        assert_eq!(lines.len(), 4);
        assert!(
            lines[0].ends_with(
                "· 0 1234 r0.120 p0.140 h0.150 w0.160-0.400 s0.050 a0.520 · 1 0 r0.000 p0.000 \
                 h0.000 w0.000-0.000 s0.000 a0.000 · 2 99 r0.290 p0.300 h0.310 w0.000-0.000 \
                 s0.000 a0.000"
            ),
            "{}",
            lines[0]
        );
        assert!(
            lines[1].ends_with(
                "· 0 CPU[0] 2097152 h0.600 l0.610-0.620 p0.630 · 1 MEMW[0] 1000 h0.700 \
                 l1.300-1.500 p1.220"
            ),
            "{}",
            lines[1]
        );
        assert!(
            lines[2].ends_with(
                "· CPU 1 in 0.01 s, the longest 0.010 s · MEMW 2 in 0.70 s, the longest 0.500 s"
            ),
            "{}",
            lines[2]
        );
        assert_eq!(
            lines[3],
            "BLOCK HEAD CARD: set up at 0.330 s on a thread of its own"
        );
    }

    #[test]
    fn past_the_stamped_windows_and_chunks_only_the_tables_count() {
        let mut log = Log::new(Instant::now());
        log.window_handed(WINDOWS, 5, 0.9, 0.95, 1.0);
        log.window_walked(WINDOWS + 3, 1.0, 0.1, 2.0);
        log.chunk_handed(CHUNKS, "LOAD", 7, 9, 1.0);
        log.chunk_laid(CHUNKS + 1, "LOAD", 1.0, 1.25);
        assert!(log.windows.is_empty());
        assert!(log.chunks.is_empty());
        assert_eq!(log.tables, vec![("LOAD", 1, 0.25, 0.25)]);
        assert_eq!(log.lines()[3], "BLOCK HEAD CARD: set up at its first use");
    }
}

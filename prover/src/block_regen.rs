//! Regeneration on the block path (D-REGEN): the measurements it is planned
//! from (R0).
//!
//! `LAMBDA_VM_BLOCK_REGEN_PROBE=1` prints two `BLOCK REGEN PROBE` lines and
//! changes nothing else:
//! - **first-touch**: per walked window, the memory bytes whose carried state
//!   the window reads ([`WalkedWindow::first_touch_bytes`]), summed over the
//!   windows the walker walks (the run's last is the finish's): the size of a
//!   per-window entry state, D-REGEN §2.4;
//! - **traces by class**: the main traces phase B starts from, split by how
//!   they could be rebuilt (D-REGEN §2.2): the streamed chunks (rebuilt from a
//!   window replay), class 2 (KECCAK_RND, ECDAS, CPU32, EQ, BYTEWISE, BRANCH:
//!   ordered lists, once streamed) and whole-run (the rest).
//!
//! [`WalkedWindow::first_touch_bytes`]: crate::tables::trace_builder::WalkedWindow::first_touch_bytes

use crate::tables::trace_builder::{StreamSkip, Traces};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

type Table = stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>;

const GIB: f64 = (1u64 << 30) as f64;

/// `LAMBDA_VM_BLOCK_REGEN_PROBE=1`: the R0 probe lines (module docs). Off by
/// default; read once.
pub(crate) fn probe() -> bool {
    static PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PROBE
        .get_or_init(|| std::env::var("LAMBDA_VM_BLOCK_REGEN_PROBE").is_ok_and(|v| v.trim() == "1"))
}

/// Per walked window: its cycles and its first-touch bytes.
#[derive(Debug, Default)]
pub(crate) struct TouchCensus {
    windows: Vec<(usize, usize)>,
}

impl TouchCensus {
    /// The next window's cycles and first-touch bytes.
    pub(crate) fn push(&mut self, cycles: usize, bytes: usize) {
        self.windows.push((cycles, bytes));
    }

    /// The `BLOCK REGEN PROBE first-touch` line: the windows, the cycles they
    /// hold, the bytes summed, per cycle, and the windows' spread (window 0,
    /// which first-touches the program's image, also apart).
    pub(crate) fn line(&self) -> String {
        let n = self.windows.len();
        let cycles: usize = self.windows.iter().map(|w| w.0).sum();
        let total: usize = self.windows.iter().map(|w| w.1).sum();
        let mut sorted: Vec<usize> = self.windows.iter().map(|w| w.1).collect();
        sorted.sort_unstable();
        let at = |q: f64| {
            sorted
                .get(((n as f64 - 1.0) * q).round() as usize)
                .copied()
                .unwrap_or(0)
        };
        let (max_at, max) = self
            .windows
            .iter()
            .enumerate()
            .max_by_key(|(_, w)| w.1)
            .map_or((0, 0), |(i, w)| (i, w.1));
        let first = self.windows.first().map_or(0, |w| w.1);
        let rest_cycles = cycles - self.windows.first().map_or(0, |w| w.0);
        let per = |bytes: usize, cycles: usize| {
            if cycles == 0 {
                0.0
            } else {
                bytes as f64 / cycles as f64
            }
        };
        let m = |b: usize| b as f64 / 1e6;
        format!(
            "BLOCK REGEN PROBE first-touch: {n} windows · {:.2} M cycles · {:.2} M bytes · \
             {:.4} per cycle · without window 0 {:.4} per cycle · per window max {:.3} M \
             (window {max_at}) · p50 {:.3} M · p90 {:.3} M · window 0 {:.3} M",
            m(cycles),
            m(total),
            per(total, cycles),
            per(total - first, rest_cycles),
            m(max),
            m(at(0.5)),
            m(at(0.9)),
            m(first),
        )
    }
}

/// How a main trace could be rebuilt for phase B (D-REGEN §2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TraceClass {
    /// A chunk phase A streamed: rebuilt by replaying the windows it spans.
    Streamed,
    /// KECCAK_RND, ECDAS, CPU32, EQ, BYTEWISE, BRANCH: lists in execution
    /// order, rebuildable once streamed.
    Ordered,
    /// The rest: built from whole-run data.
    WholeRun,
}

/// One class's traces: instances, main cells, and their bytes as phase B
/// starts (packed, spilled, at 8 bytes a cell).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ClassBytes {
    pub(crate) instances: usize,
    pub(crate) cells: u64,
    pub(crate) packed: u64,
    pub(crate) spilled: u64,
    pub(crate) wide: u64,
}

impl ClassBytes {
    fn add(&mut self, trace: &Table) {
        let cells = (trace.num_rows() * trace.num_main_columns) as u64;
        self.instances += 1;
        self.cells += cells;
        if let Some(narrow) = trace.narrow_main() {
            self.packed += narrow.data().len() as u64;
        } else if let Some(slot) = trace.spilled_main() {
            self.spilled += slot.len() as u64;
        } else {
            self.wide += cells * std::mem::size_of::<u64>() as u64;
        }
    }

    /// Its bytes, wherever they are.
    pub(crate) fn bytes(&self) -> u64 {
        self.packed + self.spilled + self.wide
    }
}

/// The main traces phase B starts from, by [`TraceClass`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TraceClasses {
    pub(crate) streamed: ClassBytes,
    pub(crate) ordered: ClassBytes,
    pub(crate) whole_run: ClassBytes,
}

impl TraceClasses {
    /// Every trace in `traces` with rows, by class: the first `streamed.<t>`
    /// chunks of each streamed table are [`TraceClass::Streamed`].
    pub(crate) fn of(traces: &Traces, streamed: &StreamSkip) -> Self {
        let mut classes = Self::default();
        for (class, trace) in classify(traces, streamed) {
            if trace.num_rows() == 0 || trace.num_main_columns == 0 {
                continue;
            }
            match class {
                TraceClass::Streamed => classes.streamed.add(trace),
                TraceClass::Ordered => classes.ordered.add(trace),
                TraceClass::WholeRun => classes.whole_run.add(trace),
            }
        }
        classes
    }

    /// All classes together.
    pub(crate) fn total(&self) -> ClassBytes {
        let [a, b, c] = [self.streamed, self.ordered, self.whole_run];
        ClassBytes {
            instances: a.instances + b.instances + c.instances,
            cells: a.cells + b.cells + c.cells,
            packed: a.packed + b.packed + c.packed,
            spilled: a.spilled + b.spilled + c.spilled,
            wide: a.wide + b.wide + c.wide,
        }
    }

    /// The `BLOCK REGEN PROBE traces by class` line: per class its instances,
    /// cells and bytes (packed + spilled + 8 B/cell), and the streamed share
    /// of the bytes and of the cells.
    pub(crate) fn line(&self) -> String {
        let total = self.total();
        let share = |part: u64, whole: u64| {
            if whole == 0 {
                0.0
            } else {
                part as f64 / whole as f64
            }
        };
        let class = |name: &str, c: &ClassBytes| {
            format!(
                "{name} {} instances {:.3} G cells {:.2} GiB ({:.2} packed + {:.2} spilled + {:.2} at 8 B/cell)",
                c.instances,
                c.cells as f64 / 1e9,
                c.bytes() as f64 / GIB,
                c.packed as f64 / GIB,
                c.spilled as f64 / GIB,
                c.wide as f64 / GIB,
            )
        };
        format!(
            "BLOCK REGEN PROBE traces by class: {} · {} · {} · total {} instances {:.2} GiB · \
             streamed share {:.4} of bytes, {:.4} of cells · with class 2 {:.4} of bytes",
            class("streamed", &self.streamed),
            class("class-2", &self.ordered),
            class("whole-run", &self.whole_run),
            total.instances,
            total.bytes() as f64 / GIB,
            share(self.streamed.bytes(), total.bytes()),
            share(self.streamed.cells, total.cells),
            share(self.streamed.bytes() + self.ordered.bytes(), total.bytes()),
        )
    }
}

/// Every trace in `traces`, each with its [`TraceClass`]. Every field of
/// [`Traces`] that holds a trace is listed here.
fn classify<'a>(
    traces: &'a Traces,
    streamed: &StreamSkip,
) -> impl Iterator<Item = (TraceClass, &'a Table)> {
    use TraceClass::{Ordered, Streamed, WholeRun};
    let split = |list: &'a [Table], n: usize| {
        list.iter().enumerate().map(
            move |(i, t)| {
                if i < n { (Streamed, t) } else { (WholeRun, t) }
            },
        )
    };
    let all = |class: TraceClass, list: &'a [Table]| list.iter().map(move |t| (class, t));
    let one = |class: TraceClass, t: &'a Table| std::iter::once((class, t));
    let Traces {
        cpus,
        bitwise,
        lts,
        shifts,
        memws,
        memw_aligneds,
        loads,
        decode,
        muls,
        dvrms,
        pages,
        page_configs: _,
        register,
        public_output_bytes: _,
        branches,
        halt,
        commits,
        keccaks,
        keccak_rnds,
        keccak_rc,
        blake3,
        num_blake3_ops,
        ecsms,
        ecdases,
        hints,
        memw_registers,
        local_to_global,
        touched_memory_cells: _,
        eqs,
        bytewises,
        stores,
        cpu32s,
    } = traces;
    split(cpus, streamed.cpu)
        .chain(split(memw_registers, streamed.memw_register))
        .chain(split(memw_aligneds, streamed.memw_aligned))
        .chain(split(memws, streamed.memw))
        .chain(split(loads, streamed.load))
        .chain(split(lts, streamed.lt))
        .chain(split(shifts, streamed.shift))
        .chain(split(stores, streamed.store))
        .chain(all(Ordered, keccak_rnds))
        .chain(all(Ordered, ecdases))
        .chain(all(Ordered, cpu32s))
        .chain(all(Ordered, eqs))
        .chain(all(Ordered, bytewises))
        .chain(all(Ordered, branches))
        .chain(all(WholeRun, muls))
        .chain(all(WholeRun, dvrms))
        .chain(all(WholeRun, pages))
        .chain(all(WholeRun, commits))
        .chain(all(WholeRun, keccaks))
        .chain(all(WholeRun, ecsms))
        .chain(all(WholeRun, hints))
        .chain(one(WholeRun, bitwise))
        .chain(one(WholeRun, decode))
        .chain(one(WholeRun, register))
        .chain(one(WholeRun, halt))
        .chain(one(WholeRun, keccak_rc))
        // An unused BLAKE3 table still pads to rows; the proof leaves it out.
        .chain((*num_blake3_ops > 0).then_some((WholeRun, blake3)))
        .chain(one(WholeRun, local_to_global))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::MaxRowsConfig;

    /// The classes of a streamed build: the streamed chunks are exactly the
    /// ones the stream handed out, class 2 is the six ordered tables, and the
    /// three cover every trace with rows.
    #[test]
    fn the_traces_split_into_their_classes() {
        let opts = crate::lfm::proof::block_base_options();
        let max_rows = MaxRowsConfig {
            keccak_rnd: 48,
            ..MaxRowsConfig::small()
        };
        for name in ["all_instructions_64", "test_keccak_multi"] {
            let program = executor::elf::Elf::load(&crate::test_utils::asm_elf_bytes(name))
                .expect("the ELF loads");
            let (traces, streamed) =
                crate::block::stream_skip_for_test(&program, &opts, &max_rows).expect("streamed");
            let classes = TraceClasses::of(&traces, &streamed);
            let handed_out = streamed.cpu
                + streamed.memw_register
                + streamed.memw_aligned
                + streamed.memw
                + streamed.load
                + streamed.lt
                + streamed.shift
                + streamed.store;
            assert!(handed_out > 0, "{name}: nothing was streamed");
            assert_eq!(classes.streamed.instances, handed_out, "{name}");
            let rows = |list: &[Table]| list.iter().filter(|t| t.num_rows() > 0).count();
            assert_eq!(
                classes.ordered.instances,
                rows(&traces.keccak_rnds)
                    + rows(&traces.ecdases)
                    + rows(&traces.cpu32s)
                    + rows(&traces.eqs)
                    + rows(&traces.bytewises)
                    + rows(&traces.branches),
                "{name}"
            );
            let total = classes.total();
            assert!(
                classes.whole_run.instances > 0 && total.bytes() > 0,
                "{name}"
            );
            assert_eq!(
                total.instances,
                classes.streamed.instances
                    + classes.ordered.instances
                    + classes.whole_run.instances
            );
            let line = classes.line();
            assert!(
                line.starts_with("BLOCK REGEN PROBE traces by class: streamed "),
                "{line}"
            );
            eprintln!("{line}");
        }
    }

    /// The census line: the sum, the rate per cycle with and without window
    /// 0, and the largest window.
    #[test]
    fn the_touch_census_sums_its_windows() {
        let mut census = TouchCensus::default();
        for bytes in [500_000, 100_000, 300_000, 200_000] {
            census.push(1_000_000, bytes);
        }
        let line = census.line();
        assert!(line.contains("4 windows · 4.00 M cycles"), "{line}");
        assert!(line.contains("· 1.10 M bytes"), "{line}");
        assert!(line.contains("0.2750 per cycle"), "{line}");
        assert!(line.contains("without window 0 0.2000 per cycle"), "{line}");
        assert!(line.contains("max 0.500 M (window 0)"), "{line}");
        assert!(line.contains("p50 0.300 M"), "{line}");
        assert!(TouchCensus::default().line().contains("0 windows"));
    }
}

//! Lambda VM CLI - execute, prove, and verify RISC-V programs.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueHint};

#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
use executor::{
    elf::{Elf, SymbolTable},
    flamegraph::FlamegraphGenerator,
    profile::InstrHistogram,
    vm::execution::Executor,
};
use prover::VmProof;
use prover::tables::trace_builder::TableReport;
use stark::proof::options::GoldilocksCubicProofOptions;

/// Polls jemalloc `stats.allocated` every 10ms from a background thread,
/// tracking the high-water mark. Near-zero overhead because jemalloc uses
/// thread-local caches — `epoch::advance()` just merges cached counters.
#[cfg(feature = "jemalloc-stats")]
mod heap_tracker {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    use tikv_jemalloc_ctl::{epoch, stats};

    pub struct HeapTracker {
        stop: Arc<AtomicBool>,
        peak: Arc<AtomicUsize>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl HeapTracker {
        pub fn start() -> Self {
            let stop = Arc::new(AtomicBool::new(false));
            let peak = Arc::new(AtomicUsize::new(0));
            let stop_clone = stop.clone();
            let peak_clone = peak.clone();

            let handle = thread::spawn(move || {
                while !stop_clone.load(Ordering::Relaxed) {
                    // Refresh jemalloc's cached stats
                    epoch::advance().ok();
                    if let Ok(allocated) = stats::allocated::read() {
                        peak_clone.fetch_max(allocated, Ordering::Relaxed);
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                // One final sample after stop signal
                epoch::advance().ok();
                if let Ok(allocated) = stats::allocated::read() {
                    peak_clone.fetch_max(allocated, Ordering::Relaxed);
                }
            });

            Self {
                stop,
                peak,
                handle: Some(handle),
            }
        }

        pub fn stop(mut self) -> usize {
            self.shutdown();
            self.peak.load(Ordering::Relaxed)
        }

        fn shutdown(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(h) = self.handle.take() {
                h.join().ok();
            }
        }
    }

    impl Drop for HeapTracker {
        fn drop(&mut self) {
            self.shutdown();
        }
    }
}

#[derive(Parser)]
#[command(author, version, about = "Lambda VM - RISC-V zkVM", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute an ELF program without generating a proof
    Execute {
        /// Path to the ELF file
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,

        /// Path to the private input file
        #[arg(long, value_hint = ValueHint::FilePath)]
        private_input: Option<PathBuf>,

        /// Generate flamegraph folded stacks to file
        #[arg(long, value_hint = ValueHint::FilePath)]
        flamegraph: Option<PathBuf>,

        /// Print the dynamic instruction (cycle) count
        #[arg(long)]
        cycles: bool,
    },

    /// Generate a proof for an ELF program
    Prove {
        /// Path to the ELF file
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,

        /// Output path for the proof bundle
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        output: PathBuf,

        /// Path to the private input file
        #[arg(long, value_hint = ValueHint::FilePath)]
        private_input: Option<PathBuf>,

        /// Blowup factor (power of 2). Higher = fewer queries, smaller proof, slower proving.
        #[arg(long, default_value = "2")]
        blowup: Option<u8>,

        /// Print proving time
        #[arg(long)]
        time: bool,

        /// Execute one pre-pass outside the timer and print dynamic instruction count
        #[arg(long)]
        cycles: bool,

        /// Generate flamegraph folded stacks for the proven run, written to this
        /// file during the pre-pass (outside the proving timer).
        #[arg(long, value_hint = ValueHint::FilePath)]
        flamegraph: Option<PathBuf>,

        /// Build traces and print total main-trace field elements (rows × columns summed across
        /// all tables) and aux-trace field elements (committed EF columns × rows)
        #[arg(long)]
        elements: bool,

        /// With --elements, also print a per-table breakdown (rows, columns,
        /// field elements, % of total) sorted by cost.
        #[arg(long)]
        tables: bool,
    },

    /// Verify a proof bundle
    Verify {
        /// Path to the proof bundle file
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        proof: PathBuf,

        /// Path to the ELF file (required for DECODE table verification)
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,

        /// Blowup factor used during proving (must match)
        #[arg(long, default_value = "2")]
        blowup: Option<u8>,

        /// Print verification time
        #[arg(long)]
        time: bool,
    },

    /// Count main-trace and aux-trace field elements without proving
    CountElements {
        /// Path to the ELF file
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,

        /// Path to the private input file
        #[arg(long, value_hint = ValueHint::FilePath)]
        private_input: Option<PathBuf>,

        /// Print a per-table breakdown (rows, columns, field elements, % of
        /// total) sorted by cost, in addition to the totals.
        #[arg(long)]
        tables: bool,
    },
}

fn main() -> ExitCode {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Execute {
            elf,
            private_input,
            flamegraph,
            cycles,
        } => cmd_execute(elf, private_input, flamegraph, cycles),
        Commands::Prove {
            elf,
            output,
            private_input,
            blowup,
            time,
            cycles,
            flamegraph,
            elements,
            tables,
        } => cmd_prove(
            elf,
            output,
            private_input,
            blowup,
            time,
            cycles,
            flamegraph,
            elements,
            tables,
        ),
        Commands::Verify {
            proof,
            elf,
            blowup,
            time,
        } => cmd_verify(proof, elf, blowup, time),
        Commands::CountElements {
            elf,
            private_input,
            tables,
        } => cmd_count_elements(elf, private_input, tables),
    }
}

fn read_private_input(path: Option<&PathBuf>) -> Result<Vec<u8>, String> {
    match path {
        Some(path) => {
            eprintln!("Reading private input file...");
            std::fs::read(path).map_err(|e| format!("Failed to read private input file: {e}"))
        }
        None => Ok(vec![]),
    }
}

/// What a profiling run should accumulate, in addition to the cycle count.
#[derive(Default, Clone, Copy)]
struct ProfileOpts {
    flamegraph: bool,
    histogram: bool,
}

/// Result of a profiling run (besides the cycle count).
#[derive(Default)]
struct ProfileResult {
    flamegraph: Option<FlamegraphGenerator>,
    histogram: Option<InstrHistogram>,
}

/// Run an ELF to completion in chunks, returning the dynamic instruction
/// (cycle) count and whichever profiling artifacts `opts` requested
/// (flamegraph symbols are resolved from `elf_data`).
///
/// Used by both `execute` and the `prove` cycle pre-pass so a single run can
/// produce the cycle count plus any requested profiles without re-executing.
fn run_and_profile(
    program: &Elf,
    elf_data: &[u8],
    private_inputs: Vec<u8>,
    opts: ProfileOpts,
) -> Result<(u64, ProfileResult), String> {
    let mut executor = Executor::new(program, private_inputs).map_err(|e| format!("{e:?}"))?;

    let mut generator = opts.flamegraph.then(|| {
        let symbols = SymbolTable::parse(elf_data);
        FlamegraphGenerator::new(symbols, program.entry_point)
    });
    let mut histogram = opts.histogram.then(InstrHistogram::new);

    let mut cycle_count: u64 = 0;
    while let Some(logs) = executor.resume().map_err(|e| format!("{e:?}"))? {
        cycle_count += logs.len() as u64;
        if generator.is_some() || histogram.is_some() {
            let logs: Vec<_> = logs.to_vec();
            if let Some(ref mut fg) = generator {
                fg.process_logs(&logs, &executor.instructions)
                    .map_err(|e| format!("Failed to process logs for flamegraph: {e:?}"))?;
            }
            if let Some(ref mut h) = histogram {
                h.process_logs(&logs, &executor.instructions)
                    .map_err(|e| format!("Failed to process logs for histogram: {e:?}"))?;
            }
        }
    }
    executor.finish().map_err(|e| format!("{e:?}"))?;

    Ok((
        cycle_count,
        ProfileResult {
            flamegraph: generator,
            histogram,
        },
    ))
}

/// Write a flamegraph generator's folded stacks to `output_path`.
fn write_flamegraph(generator: &FlamegraphGenerator, output_path: &PathBuf) -> Result<(), String> {
    let file = File::create(output_path).map_err(|e| format!("{e}"))?;
    let mut writer = BufWriter::new(file);
    generator
        .write_folded(&mut writer)
        .map_err(|e| format!("{e:?}"))?;
    eprintln!(
        "Flamegraph written to {:?} ({} instructions)",
        output_path,
        generator.total_instructions()
    );
    Ok(())
}

fn cmd_execute(
    elf_path: PathBuf,
    private_input_path: Option<PathBuf>,
    flamegraph_path: Option<PathBuf>,
    cycles: bool,
) -> ExitCode {
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let program = match Elf::load(&elf_data) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to load ELF program: {:?}", e);
            return ExitCode::FAILURE;
        }
    };

    let private_inputs = match read_private_input(private_input_path.as_ref()) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let mut executor = match Executor::new(&program, private_inputs) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create executor: {:?}", e);
            return ExitCode::FAILURE;
        }
    };

    // Set up flamegraph generator if requested
    let mut generator = flamegraph_path.as_ref().map(|_| {
        let symbols = SymbolTable::parse(&elf_data);
        FlamegraphGenerator::new(symbols, program.entry_point)
    });

    // Execute in chunks, counting cycles and (if requested) feeding the flamegraph.
    let mut cycle_count: u64 = 0;
    loop {
        let logs = match executor.resume() {
            Ok(logs) => logs,
            Err(e) => {
                eprintln!("Execution failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        match logs {
            Some(logs) => {
                cycle_count += logs.len() as u64;
                if let Some(ref mut fg) = generator {
                    let logs: Vec<_> = logs.to_vec();
                    if let Err(e) = fg.process_logs(&logs, &executor.instructions) {
                        eprintln!("Failed to process logs for flamegraph: {:?}", e);
                        return ExitCode::FAILURE;
                    }
                }
            }
            None => break,
        }
    }

    if let Err(e) = executor.finish() {
        eprintln!("Failed to finish execution: {:?}", e);
        return ExitCode::FAILURE;
    }

    // Write flamegraph output if requested
    if let (Some(output_path), Some(generator)) = (flamegraph_path, profile.flamegraph)
        && let Err(e) = write_flamegraph(&generator, &output_path)
    {
        eprintln!("Failed to write flamegraph output: {e}");
        return ExitCode::FAILURE;
    }

    if cycles {
        println!("Cycles: {}", cycle_count);
    }

    ExitCode::SUCCESS
}

#[allow(clippy::too_many_arguments)]
fn cmd_prove(
    elf_path: PathBuf,
    output_path: PathBuf,
    private_input_path: Option<PathBuf>,
    blowup: Option<u8>,
    time: bool,
    cycles: bool,
    flamegraph_path: Option<PathBuf>,
    elements: bool,
    tables: bool,
) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let private_inputs = match read_private_input(private_input_path.as_ref()) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    // Pre-pass: execute once outside the timer to count dynamic instructions
    // and (if requested) build a flamegraph of the proven run. Mirrors SP1's
    // cycle-count pass so both provers report the same kind of number without
    // inflating the measured proving time. The flamegraph is folded-stack text
    // only — rendering to SVG (e.g. with inferno) is a separate manual step and
    // is not linked into the prover. This pre-pass is read-only and has no
    // effect on the trace the proof is generated from.
    let want_prepass = cycles || flamegraph_path.is_some();
    let cycle_count = if want_prepass {
        let program = match Elf::load(&elf_data) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to load ELF for cycle count: {:?}", e);
                return ExitCode::FAILURE;
            }
        };
        let opts = ProfileOpts {
            flamegraph: flamegraph_path.is_some(),
            histogram: false,
        };
        let (count, profile) =
            match run_and_profile(&program, &elf_data, private_inputs.clone(), opts) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("Execution failed during cycle count pre-pass: {e}");
                    return ExitCode::FAILURE;
                }
            };
        if let (Some(output_path), Some(generator)) = (&flamegraph_path, profile.flamegraph)
            && let Err(e) = write_flamegraph(&generator, output_path)
        {
            eprintln!("Failed to write flamegraph output: {e}");
            return ExitCode::FAILURE;
        }
        cycles.then_some(count)
    } else {
        None
    };

    // Pre-pass: build traces and count field elements without running the proof.
    // When --tables is set, build the per-table report (and derive the totals
    // from it, so the trace is built only once).
    let element_count = if elements {
        if tables {
            match prover::table_report(&elf_data, &private_inputs) {
                Ok(reports) => {
                    let totals = print_table_report(&reports);
                    Some(totals)
                }
                Err(e) => {
                    eprintln!("Failed to build table report: {:?}", e);
                    return ExitCode::FAILURE;
                }
            }
        } else {
            match prover::count_elements(&elf_data, &private_inputs) {
                Ok(counts) => Some(counts),
                Err(e) => {
                    eprintln!("Failed to count elements: {:?}", e);
                    return ExitCode::FAILURE;
                }
            }
        }
    } else {
        None
    };

    #[cfg(feature = "jemalloc-stats")]
    let tracker = heap_tracker::HeapTracker::start();

    #[cfg(all(feature = "jemalloc-stats", feature = "instruments"))]
    stark::instruments::set_heap_reader(|| {
        tikv_jemalloc_ctl::epoch::advance().ok();
        tikv_jemalloc_ctl::stats::allocated::read().ok()
    });

    let start = Instant::now();
    let proof = match blowup {
        Some(b) => {
            let opts = match GoldilocksCubicProofOptions::with_blowup(b) {
                Ok(opts) => opts,
                Err(e) => {
                    eprintln!("Invalid proof options: {e}");
                    return ExitCode::FAILURE;
                }
            };
            eprintln!(
                "Generating proof (blowup={b}, queries={})...",
                opts.fri_number_of_queries
            );
            prover::prove_with_options_and_inputs(
                &elf_data,
                &private_inputs,
                &opts,
                &Default::default(),
            )
        }
        None => {
            eprintln!("Generating proof...");
            prover::prove_with_inputs(&elf_data, &private_inputs)
        }
    };
    let prove_elapsed = start.elapsed();
    let proof = match proof {
        Ok(proof) => proof,
        Err(e) => {
            eprintln!("Proof generation failed: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Writing proof...");
    let file = match File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create output file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let mut writer = BufWriter::new(file);

    let bytes = match bincode::serialize(&proof) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to serialize proof: {}", e);
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = writer.write_all(&bytes) {
        eprintln!("Failed to write proof: {}", e);
        return ExitCode::FAILURE;
    }

    eprintln!("Proof written to {:?}", output_path);
    if let Some(c) = cycle_count {
        println!("Cycles: {}", c);
    }
    if let Some((main, aux)) = element_count {
        println!("Elements: {}", main);
        println!("Aux elements (EF-cols): {}", aux);
    }
    if time {
        println!("Proving time: {:.3}s", prove_elapsed.as_secs_f64());
    }
    #[cfg(feature = "jemalloc-stats")]
    {
        let peak_bytes = tracker.stop();
        println!("Peak heap: {} MB", peak_bytes / (1024 * 1024));
    }
    ExitCode::SUCCESS
}

fn cmd_verify(proof_path: PathBuf, elf_path: PathBuf, blowup: Option<u8>, time: bool) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Reading proof...");
    let proof_bytes = match std::fs::read(&proof_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read proof file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let proof: VmProof = match bincode::deserialize(&proof_bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to deserialize proof: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Verifying proof...");
    let start = Instant::now();
    let result = match blowup {
        Some(b) => {
            let opts = match GoldilocksCubicProofOptions::with_blowup(b) {
                Ok(opts) => opts,
                Err(e) => {
                    eprintln!("Invalid proof options: {e}");
                    return ExitCode::FAILURE;
                }
            };
            prover::verify_with_options(&proof, &elf_data, &opts, None, None)
        }
        None => prover::verify(&proof, &elf_data),
    };
    let verify_elapsed = start.elapsed();
    let result = match result {
        Ok(valid) => valid,
        Err(e) => {
            eprintln!("Verification error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    if result {
        eprintln!("Verification succeeded!");
        if time {
            println!("Verification time: {:.3}s", verify_elapsed.as_secs_f64());
        }
        ExitCode::SUCCESS
    } else {
        eprintln!("Verification failed!");
        ExitCode::FAILURE
    }
}

fn cmd_count_elements(
    elf_path: PathBuf,
    private_input_path: Option<PathBuf>,
    tables: bool,
) -> ExitCode {
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let private_inputs = match read_private_input(private_input_path.as_ref()) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    // With --tables, build the per-table report once and derive the totals
    // from it; otherwise just count totals.
    let (main, aux) = if tables {
        match prover::table_report(&elf_data, &private_inputs) {
            Ok(reports) => print_table_report(&reports),
            Err(e) => {
                eprintln!("Failed to build table report: {:?}", e);
                return ExitCode::FAILURE;
            }
        }
    } else {
        match prover::count_elements(&elf_data, &private_inputs) {
            Ok(counts) => counts,
            Err(e) => {
                eprintln!("Failed to count elements: {:?}", e);
                return ExitCode::FAILURE;
            }
        }
    };

    println!("Elements: {}", main);
    println!("Aux elements (EF-cols): {}", aux);
    ExitCode::SUCCESS
}

/// Print a per-table breakdown to stderr (rows, columns, main/aux field
/// elements, % of total main elements), sorted by descending main elements.
/// Returns the `(total_main_elements, total_aux_elements)` so callers can also
/// print the existing totals.
fn print_table_report(reports: &[TableReport]) -> (u64, u64) {
    let total_main: u64 = reports.iter().map(|r| r.main_elements()).sum();
    let total_aux: u64 = reports.iter().map(|r| r.aux_elements()).sum();

    // Sort by descending main elements; drop empty (zero-row) tables.
    let mut sorted: Vec<&TableReport> = reports.iter().filter(|r| r.rows > 0).collect();
    sorted.sort_by_key(|r| std::cmp::Reverse(r.main_elements()));

    eprintln!();
    eprintln!("=== TABLE BREAKDOWN (by main-trace field elements) ===");
    eprintln!(
        "  {:<16} {:>12} {:>5} {:>5} {:>14} {:>14} {:>7}",
        "Table", "Rows", "MCol", "ACol", "MainElems", "AuxElems", "%",
    );
    eprintln!("  {}", "-".repeat(78));
    for r in &sorted {
        let main_e = r.main_elements();
        let pct = if total_main > 0 {
            main_e as f64 / total_main as f64 * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  {:<16} {:>12} {:>5} {:>5} {:>14} {:>14} {:>6.2}%",
            r.name,
            r.rows,
            r.main_cols,
            r.aux_cols,
            main_e,
            r.aux_elements(),
            pct,
        );
    }
    eprintln!("  {}", "-".repeat(78));
    eprintln!(
        "  {:<16} {:>12} {:>5} {:>5} {:>14} {:>14}",
        "TOTAL", "", "", "", total_main, total_aux,
    );
    eprintln!();

    (total_main, total_aux)
}

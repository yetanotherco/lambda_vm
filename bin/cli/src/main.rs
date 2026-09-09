//! Lambda VM CLI - execute, prove, and verify RISC-V programs.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueHint};

#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
use executor::vm::instruction::decoding::Instruction;
use executor::vm::instruction::execution::{Accelerator, SyscallNumbers};
use executor::{elf::Elf, flamegraph::FlamegraphGenerator, vm::execution::Executor};
use prover::VmProof;
use stark::proof::options::GoldilocksCubicProofOptions;

const DEFAULT_CONTINUATION_EPOCH_SIZE_LOG2: u32 = 20;
const MIN_CONTINUATION_EPOCH_SIZE_LOG2: u32 = 18;

/// Read a file into a buffer aligned for `rkyv::from_bytes`. A plain
/// `Vec<u8>` from `std::fs::read` is align-1 by the type system even though
/// the allocator happens to return well-aligned memory in practice — read
/// straight into an `AlignedVec` instead of relying on that.
fn read_aligned_file(path: &Path) -> std::io::Result<rkyv::util::AlignedVec<16>> {
    use std::os::unix::fs::FileExt;

    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len() as usize;
    let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(len);
    aligned.resize(len, 0);
    file.read_exact_at(&mut aligned, 0)?;
    Ok(aligned)
}

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

        /// Key the folded stacks by raw hex address instead of resolving
        /// through the ELF symtab (pairs with scripts/enrich_flamegraph.py).
        /// Only meaningful with --flamegraph.
        #[arg(long, requires = "flamegraph")]
        flamegraph_raw: bool,

        /// Checkpoint the flamegraph's folded output to --flamegraph every N
        /// cycles, so a killed run still leaves usable (partial) output on
        /// disk. Only meaningful with --flamegraph.
        #[arg(long, requires = "flamegraph")]
        flamegraph_checkpoint_cycles: Option<u64>,

        /// Stop execution early once at least this many cycles have run.
        #[arg(long)]
        cycle_budget: Option<u64>,

        /// Print the dynamic instruction (cycle) count, plus `Keccak calls` /
        /// `Ecsm calls` (accelerator syscall invocations). The accelerator lines
        /// are omitted when combined with --flamegraph (that path has no per-log
        /// data).
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
        blowup: u8,

        /// Print proving time
        #[arg(long)]
        time: bool,

        /// Execute once outside the timer and print dynamic instruction count
        #[arg(long)]
        cycles: bool,

        /// Build traces and print total main-trace field elements (rows × columns summed across
        /// all tables) and aux-trace field elements (committed EF columns × rows)
        #[arg(long, conflicts_with = "continuations")]
        elements: bool,

        /// Prove with continuations (split execution into epochs; flat peak memory)
        #[arg(long)]
        continuations: bool,

        /// Continuation epoch size as log2(cycles); e.g. 20 means 1,048,576 cycles.
        #[arg(
            long,
            value_name = "N",
            requires = "continuations",
            value_parser = parse_epoch_size_log2,
            long_help = "Continuation epoch size as log2(cycles); e.g. 20 means 1,048,576 cycles.\n\nDefault when omitted: 20. Values below 18 are rejected for the CLI because tiny epochs are dominated by fixed overhead. Indicative ethrex 10-transfer distinct-account peak heap from a local sweep: 19 ~= 6.9 GB, 20 ~= 9.5 GB, 21 ~= 15.8 GB, 22 ~= 26.8 GB. Higher values reduce epoch count, continuation bundle size, and fixed per-epoch overhead, but increase peak memory. For a new workload, try the highest value your machine can run without swapping."
        )]
        epoch_size_log2: Option<u32>,
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
        blowup: u8,

        /// Print verification time
        #[arg(long)]
        time: bool,

        /// Verify a continuation proof bundle (produced by `prove --continuations`)
        #[arg(long)]
        continuations: bool,
    },

    /// Count main-trace and aux-trace field elements without proving
    CountElements {
        /// Path to the ELF file
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,

        /// Path to the private input file
        #[arg(long, value_hint = ValueHint::FilePath)]
        private_input: Option<PathBuf>,
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
            flamegraph_raw,
            flamegraph_checkpoint_cycles,
            cycle_budget,
            cycles,
        } => cmd_execute(
            elf,
            private_input,
            FlamegraphCliOptions {
                path: flamegraph,
                raw: flamegraph_raw,
                checkpoint_cycles: flamegraph_checkpoint_cycles,
            },
            cycle_budget,
            cycles,
        ),
        Commands::Prove {
            elf,
            output,
            private_input,
            blowup,
            time,
            cycles,
            elements,
            continuations,
            epoch_size_log2,
        } => {
            if continuations {
                cmd_prove_continuation(
                    elf,
                    output,
                    private_input,
                    epoch_size_log2,
                    blowup,
                    time,
                    cycles,
                )
            } else {
                cmd_prove(elf, output, private_input, blowup, time, cycles, elements)
            }
        }
        Commands::Verify {
            proof,
            elf,
            blowup,
            time,
            continuations,
        } => {
            if continuations {
                cmd_verify_continuation(proof, elf, blowup, time)
            } else {
                cmd_verify(proof, elf, blowup, time)
            }
        }
        Commands::CountElements { elf, private_input } => cmd_count_elements(elf, private_input),
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

fn count_cycles(elf_data: &[u8], private_inputs: &[u8]) -> Result<u64, String> {
    let program =
        Elf::load(elf_data).map_err(|e| format!("Failed to load ELF for cycle count: {e:?}"))?;
    let executor = Executor::new(&program, private_inputs.to_vec())
        .map_err(|e| format!("Failed to create executor for cycle count: {e:?}"))?;
    executor
        .run()
        .map(|result| result.logs.len() as u64)
        .map_err(|e| format!("Execution failed during cycle count: {e:?}"))
}

/// Write the flamegraph's current (possibly partial) folded output to
/// `output_path`, replacing any previous contents. Used both for the final
/// write and for periodic checkpoints during a long run.
///
/// Writes to a `tempfile` in the same directory, flushes it, then persists
/// (renames) it over `output_path` — the whole file is replaced atomically,
/// so a kill mid-write can never leave `output_path` empty or torn (the
/// previous good checkpoint stays put until the new one is fully on disk).
fn write_flamegraph_checkpoint(
    output_path: &PathBuf,
    generator: &FlamegraphGenerator,
    raw: bool,
) -> Result<(), String> {
    let dir = output_path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = tempfile::NamedTempFile::new_in(dir)
        .map_err(|e| format!("Failed to create temp output file: {e}"))?;

    let mut writer = BufWriter::new(tmp.as_file());
    let result = if raw {
        generator.write_folded_raw(&mut writer)
    } else {
        generator.write_folded(&mut writer)
    };
    result.map_err(|e| format!("Failed to write flamegraph output: {e:?}"))?;
    writer
        .flush()
        .map_err(|e| format!("Failed to flush flamegraph output: {e}"))?;
    drop(writer);

    tmp.persist(output_path)
        .map_err(|e| format!("Failed to replace {output_path:?} with temp output: {e}"))?;
    Ok(())
}

/// Flamegraph-related flags grouped so `cmd_execute` doesn't need a flat
/// 8-argument signature.
struct FlamegraphCliOptions {
    path: Option<PathBuf>,
    raw: bool,
    checkpoint_cycles: Option<u64>,
}

/// Classifies one executed instruction as an accelerator syscall invocation.
///
/// Delegates to the executor's canonical `SyscallNumbers::accelerator()` so the
/// CLI's counts equal the prover's chip-trigger counts by construction: the
/// prover sets `ecall_keccak`/`ecall_ecsm` from `f.ecall && log.src1_val ==
/// <SYSCALL_NUMBER>`. Here `f.ecall` is the instruction at the log's
/// `current_pc` being `EcallEbreak`, and `src1_val` carries a7 (the syscall
/// number) on ECALL logs. (`get_private_input` is a memory-mapped read, not a
/// syscall, so it never reaches this path.)
fn accelerator_of(instruction: Option<&Instruction>, src1_val: u64) -> Option<Accelerator> {
    if !matches!(instruction, Some(Instruction::EcallEbreak)) {
        return None;
    }
    SyscallNumbers::try_from(src1_val)
        .ok()
        .and_then(|s| s.accelerator())
}

fn cmd_execute(
    elf_path: PathBuf,
    private_input_path: Option<PathBuf>,
    flamegraph: FlamegraphCliOptions,
    cycle_budget: Option<u64>,
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

    // Accelerator invocation counts, tallied only in the plain streaming path
    // below (the flamegraph path drives execution inside the executor and does
    // not expose per-log data). `None` means "not counted", so the accel lines
    // are omitted rather than printed as misleading zeros.
    let mut accel_counts: Option<(u64, u64)> = None;

    let cycle_count = if let Some(ref output_path) = flamegraph.path {
        // Shared execute+flamegraph path (executor::flamegraph) instead of
        // hand-rolling the SymbolTable/Executor/drive-loop wiring here.
        let mut next_checkpoint = flamegraph.checkpoint_cycles;
        let result = executor::flamegraph::run_with_flamegraph(
            &elf_data,
            &program,
            private_inputs,
            cycle_budget,
            |total_cycles, generator| {
                let Some(threshold) = next_checkpoint else {
                    return;
                };
                if total_cycles < threshold {
                    return;
                }
                if let Err(e) = write_flamegraph_checkpoint(output_path, generator, flamegraph.raw)
                {
                    eprintln!("Warning: flamegraph checkpoint failed: {e}");
                }
                next_checkpoint = flamegraph.checkpoint_cycles.map(|step| threshold + step);
            },
        );

        let (generator, result) = result;
        let total_cycles = match result {
            Ok(total_cycles) => total_cycles,
            Err(e) => {
                eprintln!("Execution failed: {:?}", e);
                // Best-effort: persist whatever the generator accumulated
                // before the fault instead of discarding it outright.
                match write_flamegraph_checkpoint(output_path, &generator, flamegraph.raw) {
                    Ok(()) => eprintln!(
                        "Partial flamegraph written to {:?} ({} instructions)",
                        output_path,
                        generator.total_instructions()
                    ),
                    Err(e) => eprintln!("Warning: failed to write partial flamegraph: {e}"),
                }
                return ExitCode::FAILURE;
            }
        };

        if let Err(e) = write_flamegraph_checkpoint(output_path, &generator, flamegraph.raw) {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
        eprintln!(
            "Flamegraph written to {:?} ({} instructions)",
            output_path,
            generator.total_instructions()
        );

        total_cycles
    } else {
        let mut executor = match Executor::new(&program, private_inputs) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to create executor: {:?}", e);
                return ExitCode::FAILURE;
            }
        };

        let mut cycle_count: u64 = 0;
        let mut keccak_calls: u64 = 0;
        let mut ecsm_calls: u64 = 0;
        // Reused per chunk: `(current_pc, a7)` for logs whose a7 matches an
        // accelerator syscall number. This is a cheap superset — a non-ECALL
        // instruction can hold the same value in src1 — that `accelerator_of`
        // confirms below, once the chunk's `&Log` borrow (tied to the executor's
        // `&mut`) is released so the instruction cache can be read again.
        let mut accel_candidates: Vec<(u64, u64)> = Vec::new();
        loop {
            let logs = match executor.resume_budgeted(cycle_count, cycle_budget) {
                Ok(logs) => logs,
                Err(e) => {
                    eprintln!("Execution failed: {:?}", e);
                    return ExitCode::FAILURE;
                }
            };
            let Some(logs) = logs else { break };
            cycle_count += logs.len() as u64;
            if cycles {
                for log in logs {
                    if SyscallNumbers::try_from(log.src1_val)
                        .map(|s| s.accelerator().is_some())
                        .unwrap_or(false)
                    {
                        accel_candidates.push((log.current_pc, log.src1_val));
                    }
                }
            }
            // `logs` is no longer used, so the executor's `&mut` borrow is free
            // and the instruction cache can be read to confirm each candidate.
            for (pc, a7) in accel_candidates.drain(..) {
                match accelerator_of(executor.instructions.get(pc), a7) {
                    Some(Accelerator::Keccak) => keccak_calls += 1,
                    Some(Accelerator::Ecsm) => ecsm_calls += 1,
                    None => {}
                }
            }
            if cycle_budget.is_some_and(|budget| cycle_count >= budget) {
                break;
            }
        }

        if let Err(e) = executor.finish() {
            eprintln!("Failed to finish execution: {:?}", e);
            return ExitCode::FAILURE;
        }

        if cycles {
            accel_counts = Some((keccak_calls, ecsm_calls));
        }
        cycle_count
    };

    if cycles {
        println!("Cycles: {}", cycle_count);
        if let Some((keccak_calls, ecsm_calls)) = accel_counts {
            println!("Keccak calls: {}", keccak_calls);
            println!("Ecsm calls: {}", ecsm_calls);
        }
    }

    ExitCode::SUCCESS
}

fn cmd_prove(
    elf_path: PathBuf,
    output_path: PathBuf,
    private_input_path: Option<PathBuf>,
    blowup: u8,
    time: bool,
    cycles: bool,
    elements: bool,
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

    // Pre-pass: execute once outside the timer to count dynamic instructions.
    // Mirrors SP1's cycle-count pass so both provers report the same kind of
    // number without inflating the measured proving time.
    let cycle_count = if cycles {
        match count_cycles(&elf_data, &private_inputs) {
            Ok(count) => Some(count),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    // Pre-pass: build traces and count field elements without running the proof.
    let element_count = if elements {
        match prover::count_elements(&elf_data, &private_inputs) {
            Ok(counts) => Some(counts),
            Err(e) => {
                eprintln!("Failed to count elements: {:?}", e);
                return ExitCode::FAILURE;
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
    let opts = match GoldilocksCubicProofOptions::with_blowup(blowup) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("Invalid proof options: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "Generating proof (blowup={blowup}, queries={})...",
        opts.fri_number_of_queries
    );
    let proof = prover::prove_with_options_and_inputs(
        &elf_data,
        &private_inputs,
        &opts,
        &Default::default(),
    );
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

    let bytes = match rkyv::to_bytes::<rkyv::rancor::Error>(&proof) {
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

fn cmd_verify(proof_path: PathBuf, elf_path: PathBuf, blowup: u8, time: bool) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Reading proof...");
    let proof_bytes = match read_aligned_file(&proof_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read proof file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let proof: VmProof = match rkyv::from_bytes::<VmProof, rkyv::rancor::Error>(&proof_bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to deserialize proof: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Verifying proof...");
    let start = Instant::now();
    let opts = match GoldilocksCubicProofOptions::with_blowup(blowup) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("Invalid proof options: {e}");
            return ExitCode::FAILURE;
        }
    };
    let result = prover::verify_with_options(&proof, &elf_data, &opts, None, None);
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
        eprintln!("Verification failed! Ensure --blowup matches the value used for proving.");
        ExitCode::FAILURE
    }
}

fn cmd_prove_continuation(
    elf_path: PathBuf,
    output_path: PathBuf,
    private_input_path: Option<PathBuf>,
    epoch_size_log2: Option<u32>,
    blowup: u8,
    time: bool,
    cycles: bool,
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

    let cycle_count = if cycles {
        match count_cycles(&elf_data, &private_inputs) {
            Ok(count) => Some(count),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    let epoch_size_log2 = epoch_size_log2.unwrap_or(DEFAULT_CONTINUATION_EPOCH_SIZE_LOG2);
    let epoch_size = match continuation_epoch_size(epoch_size_log2) {
        Ok(size) => size,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let opts = match GoldilocksCubicProofOptions::with_blowup(blowup) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("Invalid proof options: {e}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!(
        "Generating continuation proof (blowup={blowup}, epoch_size_log2={epoch_size_log2}, epoch_size={epoch_size})...",
    );
    // Same tracker as the monolithic path: the peak here is the flat per-epoch
    // working set rather than a whole-trace high-water mark, and it is the metric
    // that decides which epoch size a machine can run, so the benchmarks need it
    // reported identically on both paths.
    #[cfg(feature = "jemalloc-stats")]
    let tracker = heap_tracker::HeapTracker::start();
    let start = Instant::now();
    let bundle = match prover::continuation::prove_continuation(
        &elf_data,
        &private_inputs,
        epoch_size_log2,
        &opts,
    ) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Continuation proof generation failed: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let prove_elapsed = start.elapsed();

    eprintln!("Writing proof...");
    let file = match File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create output file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let mut writer = BufWriter::new(file);
    let bytes = match rkyv::to_bytes::<rkyv::rancor::Error>(&bundle) {
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
    println!("Epochs: {}", bundle.num_epochs());
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

fn cmd_verify_continuation(
    proof_path: PathBuf,
    elf_path: PathBuf,
    blowup: u8,
    time: bool,
) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Reading proof...");
    let proof_bytes = match read_aligned_file(&proof_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read proof file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let bundle: prover::continuation::ContinuationProof =
        match rkyv::from_bytes::<prover::continuation::ContinuationProof, rkyv::rancor::Error>(
            &proof_bytes,
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to deserialize proof: {}", e);
                return ExitCode::FAILURE;
            }
        };

    let opts = match GoldilocksCubicProofOptions::with_blowup(blowup) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("Invalid proof options: {e}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Verifying continuation proof...");
    let start = Instant::now();
    let result = prover::continuation::verify_continuation(&elf_data, &bundle, &opts);
    let verify_elapsed = start.elapsed();

    match result {
        Ok(Some(output)) => {
            eprintln!("Verification succeeded!");
            let hex: String = output.iter().map(|b| format!("{:02x}", b)).collect();
            println!("Output: {}", hex);
            if time {
                println!("Verification time: {:.3}s", verify_elapsed.as_secs_f64());
            }
            ExitCode::SUCCESS
        }
        Ok(None) => {
            eprintln!("Verification failed! Ensure --blowup matches the value used for proving.");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("Verification error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn cmd_count_elements(elf_path: PathBuf, private_input_path: Option<PathBuf>) -> ExitCode {
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

    match prover::count_elements(&elf_data, &private_inputs) {
        Ok((main, aux)) => {
            println!("Elements: {}", main);
            println!("Aux elements (EF-cols): {}", aux);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Failed to count elements: {:?}", e);
            ExitCode::FAILURE
        }
    }
}

fn continuation_epoch_size(epoch_size_log2: u32) -> Result<usize, String> {
    if epoch_size_log2 < MIN_CONTINUATION_EPOCH_SIZE_LOG2 {
        return Err(format!(
            "--epoch-size-log2 must be at least {MIN_CONTINUATION_EPOCH_SIZE_LOG2} for CLI proving"
        ));
    }
    1usize.checked_shl(epoch_size_log2).ok_or_else(|| {
        format!("--epoch-size-log2 {epoch_size_log2} is too large for this platform")
    })
}

fn parse_epoch_size_log2(value: &str) -> Result<u32, String> {
    let epoch_size_log2 = value
        .parse::<u32>()
        .map_err(|_| format!("--epoch-size-log2 must be an integer, got `{value}`"))?;
    continuation_epoch_size(epoch_size_log2)?;
    Ok(epoch_size_log2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    // The arg graph is well-formed (e.g. `requires`/`conflicts_with` reference real args).
    #[test]
    fn cli_command_is_valid() {
        Cli::command().debug_assert();
    }

    // The continuation epoch flag requires --continuations.
    #[test]
    fn epoch_size_log2_requires_continuations() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--epoch-size-log2",
            "20",
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn epoch_size_log2_accepts_continuations() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--epoch-size-log2",
            "20",
        ]);
        assert!(r.is_ok());
    }

    #[test]
    fn cycles_accepts_continuations() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--cycles",
        ]);
        assert!(r.is_ok());
    }

    #[test]
    fn elements_conflicts_with_continuations() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--elements",
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn epoch_size_log2_rejects_tiny_cli_values() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--epoch-size-log2",
            "17",
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn old_epoch_size_flag_is_rejected() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--epoch-size",
            "1048576",
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn old_num_epochs_flag_is_rejected() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--num-epochs",
            "4",
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn prove_help_omits_removed_epoch_flags() {
        let mut cmd = Cli::command();
        let prove = cmd.find_subcommand_mut("prove").unwrap();
        let mut help = Vec::new();
        prove.write_long_help(&mut help).unwrap();
        let help = String::from_utf8(help).unwrap();

        assert!(help.contains("--epoch-size-log2 <N>"));
        assert!(!help.contains("--num-epochs"));
        assert!(!help.contains("--epoch-size <"));
    }

    #[test]
    fn continuation_epoch_size_rejects_tiny_cli_values() {
        assert!(continuation_epoch_size(17).is_err());
    }

    #[test]
    fn continuation_epoch_size_uses_exact_power_of_two() {
        assert_eq!(continuation_epoch_size(20).unwrap(), 1 << 20);
    }

    // `accelerator_of` must match the prover's `CpuOperation::from_log`: count an
    // invocation only when the instruction is an ECALL AND a7 is the accelerator
    // syscall number. Covers both accelerators, the non-accelerator syscalls, a
    // non-ECALL whose src1 collides with an accelerator number, and a cache miss.
    #[test]
    fn accelerator_of_mirrors_prover_classification() {
        use executor::vm::instruction::execution::{ECSM_SYSCALL_NUMBER, KECCAK_SYSCALL_NUMBER};

        let ecall = Instruction::EcallEbreak;

        assert_eq!(
            accelerator_of(Some(&ecall), KECCAK_SYSCALL_NUMBER),
            Some(Accelerator::Keccak)
        );
        assert_eq!(
            accelerator_of(Some(&ecall), ECSM_SYSCALL_NUMBER),
            Some(Accelerator::Ecsm)
        );

        // Non-accelerator syscalls (Commit=64, Halt=93) count as neither.
        assert_eq!(
            accelerator_of(Some(&ecall), SyscallNumbers::Commit as u64),
            None
        );
        assert_eq!(
            accelerator_of(Some(&ecall), SyscallNumbers::Halt as u64),
            None
        );

        // A non-ECALL instruction whose src1 happens to equal an accelerator a7
        // must not count — this is the `f.ecall &&` guard the prover applies.
        assert_eq!(
            accelerator_of(Some(&Instruction::Fence), KECCAK_SYSCALL_NUMBER),
            None
        );

        // No decoded instruction at the pc (cache miss) counts as neither.
        assert_eq!(accelerator_of(None, KECCAK_SYSCALL_NUMBER), None);
    }
}

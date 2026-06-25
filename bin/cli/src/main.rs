//! Lambda VM CLI - execute, prove, and verify RISC-V programs.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueHint};

#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
use executor::{
    elf::{Elf, SymbolTable},
    flamegraph::FlamegraphGenerator,
    vm::execution::Executor,
};
use prover::VmProof;
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

        /// Build traces and print total main-trace field elements (rows × columns summed across
        /// all tables) and aux-trace field elements (committed EF columns × rows)
        #[arg(long)]
        elements: bool,

        /// Prove with continuations (split execution into epochs; flat peak memory)
        #[arg(long)]
        continuations: bool,

        /// Epoch length in cycles (continuations only). Rounded up to a power of two (>=4).
        #[arg(long, requires = "continuations", conflicts_with = "num_epochs")]
        epoch_size: Option<usize>,

        /// Target number of epochs (continuations only); sets epoch_size = ceil(cycles / N).
        /// Default when neither flag is given: 4.
        #[arg(long, requires = "continuations", conflicts_with = "epoch_size")]
        num_epochs: Option<usize>,
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
            cycles,
        } => cmd_execute(elf, private_input, flamegraph, cycles),
        Commands::Prove {
            elf,
            output,
            private_input,
            blowup,
            time,
            cycles,
            elements,
            continuations,
            epoch_size,
            num_epochs,
        } => {
            if continuations {
                cmd_prove_continuation(
                    elf,
                    output,
                    private_input,
                    epoch_size,
                    num_epochs,
                    blowup,
                    time,
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
                eprintln!("Execution failed: {:?}", e);
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
    if let (Some(output_path), Some(generator)) = (flamegraph_path, generator) {
        let file = match File::create(&output_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to create flamegraph output file: {}", e);
                return ExitCode::FAILURE;
            }
        };
        let mut writer = BufWriter::new(file);
        if let Err(e) = generator.write_folded(&mut writer) {
            eprintln!("Failed to write flamegraph output: {:?}", e);
            return ExitCode::FAILURE;
        }

        eprintln!(
            "Flamegraph written to {:?} ({} instructions)",
            output_path,
            generator.total_instructions()
        );
    }

    if cycles {
        println!("Cycles: {}", cycle_count);
    }

    ExitCode::SUCCESS
}

fn cmd_prove(
    elf_path: PathBuf,
    output_path: PathBuf,
    private_input_path: Option<PathBuf>,
    blowup: Option<u8>,
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
        let program = match Elf::load(&elf_data) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to load ELF for cycle count: {:?}", e);
                return ExitCode::FAILURE;
            }
        };
        let executor = match Executor::new(&program, private_inputs.clone()) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to create executor for cycle count: {:?}", e);
                return ExitCode::FAILURE;
            }
        };
        match executor.run() {
            Ok(result) => Some(result.logs.len() as u64),
            Err(e) => {
                eprintln!("Execution failed during cycle count: {:?}", e);
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

fn cmd_prove_continuation(
    elf_path: PathBuf,
    output_path: PathBuf,
    private_input_path: Option<PathBuf>,
    epoch_size: Option<usize>,
    num_epochs: Option<usize>,
    blowup: Option<u8>,
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

    let private_inputs = match read_private_input(private_input_path.as_ref()) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    // Resolve the epoch size. An explicit --epoch-size wins; otherwise split the
    // run into N epochs (--num-epochs, default 4) via a cycle pre-pass.
    let epoch_size = match epoch_size {
        Some(n) => n,
        None => {
            let n = num_epochs.unwrap_or(4).max(1);
            let program = match Elf::load(&elf_data) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Failed to load ELF for cycle count: {:?}", e);
                    return ExitCode::FAILURE;
                }
            };
            let executor = match Executor::new(&program, private_inputs.clone()) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("Failed to create executor for cycle count: {:?}", e);
                    return ExitCode::FAILURE;
                }
            };
            let total_cycles = match executor.run() {
                Ok(result) => result.logs.len(),
                Err(e) => {
                    eprintln!("Execution failed during cycle count: {:?}", e);
                    return ExitCode::FAILURE;
                }
            };
            total_cycles.div_ceil(n).max(1)
        }
    };

    let blowup = blowup.unwrap_or(2);
    let opts = match GoldilocksCubicProofOptions::with_blowup(blowup) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("Invalid proof options: {e}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!(
        "Generating continuation proof (blowup={blowup}, epoch_size={epoch_size}, rounded to {})...",
        epoch_size.next_power_of_two().max(4)
    );
    let start = Instant::now();
    let bundle = match prover::continuation::prove_continuation(
        &elf_data,
        &private_inputs,
        epoch_size,
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
    let bytes = match bincode::serialize(&bundle) {
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
    println!("Epochs: {}", bundle.num_epochs());
    if time {
        println!("Proving time: {:.3}s", prove_elapsed.as_secs_f64());
    }
    ExitCode::SUCCESS
}

fn cmd_verify_continuation(
    proof_path: PathBuf,
    elf_path: PathBuf,
    blowup: Option<u8>,
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
    let proof_bytes = match std::fs::read(&proof_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read proof file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let bundle: prover::continuation::ContinuationProof = match bincode::deserialize(&proof_bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to deserialize proof: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let blowup = blowup.unwrap_or(2);
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
            eprintln!("Verification failed!");
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    // The arg graph is well-formed (e.g. `requires`/`conflicts_with` reference real args).
    #[test]
    fn cli_command_is_valid() {
        Cli::command().debug_assert();
    }

    // --epoch-size and --num-epochs are mutually exclusive.
    #[test]
    fn epoch_size_and_num_epochs_conflict() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--epoch-size",
            "8",
            "--num-epochs",
            "4",
        ]);
        assert!(r.is_err());
    }

    // The continuation epoch flags require --continuations.
    #[test]
    fn epoch_size_requires_continuations() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--epoch-size",
            "8",
        ]);
        assert!(r.is_err());
    }
}

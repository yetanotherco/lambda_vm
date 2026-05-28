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
    },

    /// Generate a proof and report its serialized byte size, broken down
    /// by component (trace openings, FRI, OOD evals, MMCS metadata, ...).
    /// Intended for CI to track proof-size regressions / improvements
    /// (e.g. the streaming MMCS migration).
    ProofSize {
        /// Path to the ELF file
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,

        /// Optional path to a pre-generated proof bundle. When supplied,
        /// the ELF is not re-proven; the file is decoded and its sizes
        /// reported directly. The ELF is still needed to bind the proof
        /// to the program statement.
        #[arg(long, value_hint = ValueHint::FilePath)]
        proof: Option<PathBuf>,

        /// Path to the private input file
        #[arg(long, value_hint = ValueHint::FilePath)]
        private_input: Option<PathBuf>,

        /// Emit machine-readable JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
    },

    /// Diff two `proof-size --json` reports and emit a comparison suitable
    /// for posting to a PR / Slack channel. Pure post-processing — does not
    /// run the prover. Designed to mirror the `tooling/loc` workflow:
    ///   cli proof-size base.elf --json > base.json
    ///   cli proof-size pr.elf   --json > pr.json
    ///   cli proof-size-diff base.json pr.json --format github > comment.md
    ProofSizeDiff {
        /// JSON report from the baseline (e.g. main) build.
        #[arg(value_hint = ValueHint::FilePath)]
        previous: PathBuf,
        /// JSON report from the candidate (e.g. PR) build.
        #[arg(value_hint = ValueHint::FilePath)]
        current: PathBuf,
        /// Output format: `github` (markdown table for PR comments),
        /// `slack` (Slack-flavoured markdown), or `text` (plain table).
        #[arg(long, default_value = "text")]
        format: String,
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
        } => cmd_execute(elf, private_input, flamegraph),
        Commands::Prove {
            elf,
            output,
            private_input,
            blowup,
            time,
            cycles,
            elements,
        } => cmd_prove(elf, output, private_input, blowup, time, cycles, elements),
        Commands::Verify {
            proof,
            elf,
            blowup,
            time,
        } => cmd_verify(proof, elf, blowup, time),
        Commands::CountElements { elf, private_input } => cmd_count_elements(elf, private_input),
        Commands::ProofSize {
            elf,
            proof,
            private_input,
            json,
        } => cmd_proof_size(elf, proof, private_input, json),
        Commands::ProofSizeDiff {
            previous,
            current,
            format,
        } => cmd_proof_size_diff(previous, current, &format),
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

    // Execute in chunks, processing logs only if generating flamegraph
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
            prover::verify_with_options(&proof, &elf_data, &opts)
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

// =============================================================================
// proof-size: serialize a VmProof and report a per-section byte breakdown.
// =============================================================================

/// One row of the proof-size report. `bytes` are the serialized length of
/// the corresponding piece of the proof under the same encoder used for the
/// full bundle (bincode v1).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ProofSizeEntry {
    section: String,
    bytes: usize,
}

/// Top-level JSON shape emitted by `cli proof-size --json` and consumed by
/// `cli proof-size-diff`. Stable enough for CI to depend on.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ProofSizeReport {
    elf: String,
    total_vm_proof_bytes: usize,
    multi_proof_bytes: usize,
    sub_proof_count: usize,
    main_mmcs_spec_entries: usize,
    sections: Vec<ProofSizeEntry>,
}

fn ser_len<T: serde::Serialize>(value: &T) -> usize {
    // bincode v1 mirrors the encoding used by VmProof callers (bin/cli prove
    // and prover tests), so per-section sums add up to the total bundle.
    bincode::serialize(value).map(|v| v.len()).unwrap_or(0)
}

fn cmd_proof_size(
    elf_path: PathBuf,
    proof_path: Option<PathBuf>,
    private_input_path: Option<PathBuf>,
    json: bool,
) -> ExitCode {
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let vm_proof: VmProof = if let Some(path) = proof_path {
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Failed to read proof file {}: {}", path.display(), e);
                return ExitCode::FAILURE;
            }
        };
        match bincode::deserialize(&bytes) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to decode proof bundle: {}", e);
                return ExitCode::FAILURE;
            }
        }
    } else {
        let private_inputs = match read_private_input(private_input_path.as_ref()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        };
        eprintln!("Generating proof to measure...");
        match prover::prove_with_inputs(&elf_data, &private_inputs) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Proving failed: {:?}", e);
                return ExitCode::FAILURE;
            }
        }
    };

    let total = ser_len(&vm_proof);
    let multi_proof_bytes = ser_len(&vm_proof.proof);
    let main_mmcs_roots_bytes = ser_len(&vm_proof.proof.main_mmcs_roots);
    let main_mmcs_specs_bytes = ser_len(&vm_proof.proof.main_mmcs_specs);
    let aux_mmcs_roots_bytes = ser_len(&vm_proof.proof.aux_mmcs_roots);
    let aux_mmcs_specs_bytes = ser_len(&vm_proof.proof.aux_mmcs_specs);
    let comp_mmcs_roots_bytes = ser_len(&vm_proof.proof.comp_mmcs_roots);
    let comp_mmcs_specs_bytes = ser_len(&vm_proof.proof.comp_mmcs_specs);
    let chunk_size_bytes = ser_len(&vm_proof.proof.chunk_size);
    // Phase D: per-(chunk, bucket) batched FRI.
    let fri_chunk_buckets_bytes = ser_len(&vm_proof.proof.fri_chunk_buckets);

    // Sum per-section across every sub-proof so a single number captures the
    // contribution of, e.g., "all FRI query lists across all tables".
    let mut s_main_trace_openings = 0usize;
    let mut s_precomputed_trace_openings = 0usize;
    let mut s_aux_trace_openings = 0usize;
    let mut s_composition_openings = 0usize;
    let mut s_trace_ood = 0usize;
    let mut s_composition_ood = 0usize;
    let mut s_per_table_main_root = 0usize;
    let mut s_precomputed_root = 0usize;
    let mut s_bus_public_inputs = 0usize;
    let s_other;

    for proof in &vm_proof.proof.proofs {
        s_per_table_main_root += ser_len(&proof.lde_trace_main_merkle_root);
        s_precomputed_root += ser_len(&proof.lde_trace_precomputed_merkle_root);
        s_trace_ood += ser_len(&proof.trace_ood_evaluations);
        s_composition_ood += ser_len(&proof.composition_poly_parts_ood_evaluation);
        s_bus_public_inputs += ser_len(&proof.bus_public_inputs);

        for opening in &proof.deep_poly_openings {
            s_main_trace_openings += ser_len(&opening.main_trace_polys);
            s_precomputed_trace_openings += ser_len(&opening.precomputed_trace_polys);
            s_aux_trace_openings += ser_len(&opening.aux_trace_polys);
            s_composition_openings += ser_len(&opening.composition_poly);
        }
    }

    // Anything not captured above (public_inputs, trace_length, headers...).
    // Calculate as the bundle delta so the breakdown still sums to ~total.
    let accounted = main_mmcs_roots_bytes
        + main_mmcs_specs_bytes
        + aux_mmcs_roots_bytes
        + aux_mmcs_specs_bytes
        + comp_mmcs_roots_bytes
        + comp_mmcs_specs_bytes
        + chunk_size_bytes
        + fri_chunk_buckets_bytes
        + s_main_trace_openings
        + s_precomputed_trace_openings
        + s_aux_trace_openings
        + s_composition_openings
        + s_trace_ood
        + s_composition_ood
        + s_per_table_main_root
        + s_precomputed_root
        + s_bus_public_inputs;
    s_other = multi_proof_bytes.saturating_sub(accounted);

    let entries: Vec<ProofSizeEntry> = vec![
        ProofSizeEntry { section: "main_mmcs_roots (per-chunk)".into(), bytes: main_mmcs_roots_bytes },
        ProofSizeEntry { section: "main_mmcs_specs (per-chunk)".into(), bytes: main_mmcs_specs_bytes },
        ProofSizeEntry { section: "aux_mmcs_roots (per-chunk)".into(), bytes: aux_mmcs_roots_bytes },
        ProofSizeEntry { section: "aux_mmcs_specs (per-chunk)".into(), bytes: aux_mmcs_specs_bytes },
        ProofSizeEntry { section: "comp_mmcs_roots (per-chunk)".into(), bytes: comp_mmcs_roots_bytes },
        ProofSizeEntry { section: "comp_mmcs_specs (per-chunk)".into(), bytes: comp_mmcs_specs_bytes },
        ProofSizeEntry { section: "chunk_size".into(), bytes: chunk_size_bytes },
        ProofSizeEntry { section: "per_table_main_merkle_root (preprocessed)".into(), bytes: s_per_table_main_root },
        ProofSizeEntry { section: "per_table_precomputed_merkle_root".into(), bytes: s_precomputed_root },
        ProofSizeEntry { section: "deep_poly_openings.main_trace_polys".into(), bytes: s_main_trace_openings },
        ProofSizeEntry { section: "deep_poly_openings.precomputed_trace_polys".into(), bytes: s_precomputed_trace_openings },
        ProofSizeEntry { section: "deep_poly_openings.aux_trace_polys".into(), bytes: s_aux_trace_openings },
        ProofSizeEntry { section: "deep_poly_openings.composition_poly".into(), bytes: s_composition_openings },
        ProofSizeEntry { section: "fri_chunk_buckets (per-chunk batched FRI)".into(), bytes: fri_chunk_buckets_bytes },
        ProofSizeEntry { section: "trace_ood_evaluations".into(), bytes: s_trace_ood },
        ProofSizeEntry { section: "composition_poly_parts_ood_evaluation".into(), bytes: s_composition_ood },
        ProofSizeEntry { section: "bus_public_inputs".into(), bytes: s_bus_public_inputs },
        ProofSizeEntry { section: "other (headers / public_inputs / ...)".into(), bytes: s_other },
    ];

    if json {
        let report = ProofSizeReport {
            elf: elf_path.display().to_string(),
            total_vm_proof_bytes: total,
            multi_proof_bytes,
            sub_proof_count: vm_proof.proof.proofs.len(),
            main_mmcs_spec_entries: vm_proof.proof.main_mmcs_specs.iter().map(|s| s.len()).sum::<usize>(),
            sections: entries.clone(),
        };
        match serde_json::to_string_pretty(&report) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("Failed to encode JSON: {}", e);
                return ExitCode::FAILURE;
            }
        }
    } else {
        println!();
        println!("== VmProof size report ==");
        println!("ELF:               {}", elf_path.display());
        println!("Total VmProof:     {:>10}  bytes", total);
        println!("MultiProof only:   {:>10}  bytes", multi_proof_bytes);
        println!("Sub-proofs:        {:>10}", vm_proof.proof.proofs.len());
        println!("MMCS spec entries: {:>10}", vm_proof.proof.main_mmcs_specs.iter().map(|s| s.len()).sum::<usize>());
        println!();
        println!("{:<48}{:>14}{:>10}", "section", "bytes", "% of total");
        println!("{}", "-".repeat(72));
        let denom = total.max(1) as f64;
        for e in &entries {
            println!(
                "{:<48}{:>14}{:>9.2}%",
                e.section,
                e.bytes,
                (e.bytes as f64) * 100.0 / denom
            );
        }
    }

    ExitCode::SUCCESS
}

// =============================================================================
// proof-size-diff: read two ProofSizeReport JSONs and emit a comparison.
// =============================================================================

fn cmd_proof_size_diff(previous: PathBuf, current: PathBuf, format: &str) -> ExitCode {
    let prev: ProofSizeReport = match load_report(&previous) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to load previous report ({}): {}", previous.display(), e);
            return ExitCode::FAILURE;
        }
    };
    let curr: ProofSizeReport = match load_report(&current) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to load current report ({}): {}", current.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let rendered = match format {
        "github" => render_github(&prev, &curr),
        "slack" => render_slack(&prev, &curr),
        "text" | "txt" => render_text(&prev, &curr),
        other => {
            eprintln!("Unknown --format value: {other:?}. Try github | slack | text.");
            return ExitCode::FAILURE;
        }
    };
    println!("{rendered}");
    ExitCode::SUCCESS
}

fn load_report(path: &PathBuf) -> Result<ProofSizeReport, String> {
    let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&s).map_err(|e| e.to_string())
}

/// Pair sections from two reports by name. The order returned mirrors the
/// section order of `curr`; any section present in `prev` but missing in
/// `curr` is appended at the end so the diff is lossless.
fn paired_sections<'a>(
    prev: &'a ProofSizeReport,
    curr: &'a ProofSizeReport,
) -> Vec<(String, Option<usize>, Option<usize>)> {
    let mut out: Vec<(String, Option<usize>, Option<usize>)> = Vec::new();
    for c in &curr.sections {
        let p = prev.sections.iter().find(|p| p.section == c.section);
        out.push((c.section.clone(), p.map(|p| p.bytes), Some(c.bytes)));
    }
    for p in &prev.sections {
        if curr.sections.iter().all(|c| c.section != p.section) {
            out.push((p.section.clone(), Some(p.bytes), None));
        }
    }
    out
}

fn fmt_delta(prev: Option<usize>, curr: Option<usize>) -> String {
    match (prev, curr) {
        (Some(p), Some(c)) => {
            let d = c as i64 - p as i64;
            let pct = if p == 0 { 0.0 } else { d as f64 * 100.0 / p as f64 };
            format!("{:+} ({:+.2}%)", d, pct)
        }
        (None, Some(c)) => format!("+{} (new)", c),
        (Some(p), None) => format!("-{} (gone)", p),
        (None, None) => "—".to_string(),
    }
}

fn fmt_total_delta(prev: usize, curr: usize) -> String {
    let d = curr as i64 - prev as i64;
    let pct = if prev == 0 { 0.0 } else { d as f64 * 100.0 / prev as f64 };
    format!("{:+} ({:+.2}%)", d, pct)
}

fn render_text(prev: &ProofSizeReport, curr: &ProofSizeReport) -> String {
    let mut s = String::new();
    s.push_str("== Proof size diff ==\n");
    s.push_str(&format!("previous: {}  ({} bytes)\n", prev.elf, prev.total_vm_proof_bytes));
    s.push_str(&format!("current:  {}  ({} bytes)\n", curr.elf, curr.total_vm_proof_bytes));
    s.push_str(&format!(
        "total delta: {}\n\n",
        fmt_total_delta(prev.total_vm_proof_bytes, curr.total_vm_proof_bytes)
    ));
    s.push_str(&format!("{:<48}{:>12}{:>12}{:>22}\n", "section", "previous", "current", "delta"));
    s.push_str(&format!("{}\n", "-".repeat(94)));
    for (section, p, c) in paired_sections(prev, curr) {
        let p_str = p.map(|v| v.to_string()).unwrap_or_else(|| "—".into());
        let c_str = c.map(|v| v.to_string()).unwrap_or_else(|| "—".into());
        s.push_str(&format!("{:<48}{:>12}{:>12}{:>22}\n", section, p_str, c_str, fmt_delta(p, c)));
    }
    s
}

fn render_github(prev: &ProofSizeReport, curr: &ProofSizeReport) -> String {
    let mut s = String::new();
    s.push_str("### 📦 Proof size diff\n\n");
    s.push_str(&format!(
        "| | bytes |\n|---|---:|\n| previous (`{}`) | {} |\n| current (`{}`) | {} |\n| **total delta** | **{}** |\n\n",
        prev.elf,
        prev.total_vm_proof_bytes,
        curr.elf,
        curr.total_vm_proof_bytes,
        fmt_total_delta(prev.total_vm_proof_bytes, curr.total_vm_proof_bytes),
    ));
    s.push_str("<details><summary>Per-section breakdown</summary>\n\n");
    s.push_str("| section | previous | current | delta |\n|---|---:|---:|---:|\n");
    for (section, p, c) in paired_sections(prev, curr) {
        let p_str = p.map(|v| v.to_string()).unwrap_or_else(|| "—".into());
        let c_str = c.map(|v| v.to_string()).unwrap_or_else(|| "—".into());
        s.push_str(&format!("| `{}` | {} | {} | {} |\n", section, p_str, c_str, fmt_delta(p, c)));
    }
    s.push_str("\n</details>\n");
    s
}

fn render_slack(prev: &ProofSizeReport, curr: &ProofSizeReport) -> String {
    let mut s = String::new();
    s.push_str("*Proof size diff*\n");
    s.push_str(&format!(
        "previous (`{}`): {} bytes\n",
        prev.elf, prev.total_vm_proof_bytes
    ));
    s.push_str(&format!(
        "current  (`{}`): {} bytes\n",
        curr.elf, curr.total_vm_proof_bytes
    ));
    s.push_str(&format!(
        "*total delta*: {}\n\n```\n",
        fmt_total_delta(prev.total_vm_proof_bytes, curr.total_vm_proof_bytes)
    ));
    s.push_str(&format!("{:<48}{:>12}{:>12}{:>22}\n", "section", "previous", "current", "delta"));
    for (section, p, c) in paired_sections(prev, curr) {
        let p_str = p.map(|v| v.to_string()).unwrap_or_else(|| "—".into());
        let c_str = c.map(|v| v.to_string()).unwrap_or_else(|| "—".into());
        s.push_str(&format!("{:<48}{:>12}{:>12}{:>22}\n", section, p_str, c_str, fmt_delta(p, c)));
    }
    s.push_str("```\n");
    s
}

#[cfg(test)]
mod proof_size_diff_tests {
    use super::*;

    fn r(elf: &str, total: usize, sections: &[(&str, usize)]) -> ProofSizeReport {
        ProofSizeReport {
            elf: elf.into(),
            total_vm_proof_bytes: total,
            multi_proof_bytes: total,
            sub_proof_count: 1,
            main_mmcs_spec_entries: 0,
            sections: sections
                .iter()
                .map(|(s, b)| ProofSizeEntry { section: (*s).into(), bytes: *b })
                .collect(),
        }
    }

    #[test]
    fn text_diff_shows_total_and_per_section_delta() {
        let prev = r("base.elf", 100, &[("a", 60), ("b", 40)]);
        let curr = r("pr.elf", 110, &[("a", 50), ("b", 60)]);
        let out = render_text(&prev, &curr);
        assert!(out.contains("total delta: +10"));
        assert!(out.contains("-10"));
        assert!(out.contains("+20"));
    }

    #[test]
    fn diff_handles_new_and_removed_sections() {
        let prev = r("base.elf", 50, &[("a", 30), ("gone", 20)]);
        let curr = r("pr.elf", 60, &[("a", 30), ("new", 30)]);
        let pairs = paired_sections(&prev, &curr);
        // Order: current sections first, then prev-only.
        assert_eq!(pairs[0].0, "a");
        assert_eq!(pairs[1].0, "new");
        assert_eq!(pairs[2].0, "gone");
        let text = render_text(&prev, &curr);
        assert!(text.contains("(new)"));
        assert!(text.contains("(gone)"));
    }

    #[test]
    fn github_format_has_collapsible_section() {
        let prev = r("base.elf", 100, &[("a", 100)]);
        let curr = r("pr.elf", 90, &[("a", 90)]);
        let out = render_github(&prev, &curr);
        assert!(out.contains("### 📦 Proof size diff"));
        assert!(out.contains("<details>"));
        assert!(out.contains("-10 (-10.00%)"));
    }
}

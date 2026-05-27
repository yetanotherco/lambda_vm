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
#[derive(Debug, Clone, serde::Serialize)]
struct ProofSizeEntry {
    section: &'static str,
    bytes: usize,
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
    let main_mmcs_root_bytes = ser_len(&vm_proof.proof.main_mmcs_root);
    let main_mmcs_spec_bytes = ser_len(&vm_proof.proof.main_mmcs_spec);

    // Sum per-section across every sub-proof so a single number captures the
    // contribution of, e.g., "all FRI query lists across all tables".
    let mut s_main_trace_openings = 0usize;
    let mut s_precomputed_trace_openings = 0usize;
    let mut s_aux_trace_openings = 0usize;
    let mut s_composition_openings = 0usize;
    let mut s_fri_query_list = 0usize;
    let mut s_fri_layers_roots = 0usize;
    let mut s_trace_ood = 0usize;
    let mut s_composition_ood = 0usize;
    let mut s_per_table_main_root = 0usize;
    let mut s_aux_root = 0usize;
    let mut s_precomputed_root = 0usize;
    let mut s_bus_public_inputs = 0usize;
    let s_other;

    for proof in &vm_proof.proof.proofs {
        s_per_table_main_root += ser_len(&proof.lde_trace_main_merkle_root);
        s_aux_root += ser_len(&proof.lde_trace_aux_merkle_root);
        s_precomputed_root += ser_len(&proof.lde_trace_precomputed_merkle_root);
        s_trace_ood += ser_len(&proof.trace_ood_evaluations);
        s_composition_ood += ser_len(&proof.composition_poly_parts_ood_evaluation);
        s_fri_query_list += ser_len(&proof.query_list);
        s_fri_layers_roots += ser_len(&proof.fri_layers_merkle_roots);
        s_bus_public_inputs += ser_len(&proof.bus_public_inputs);

        for opening in &proof.deep_poly_openings {
            s_main_trace_openings += ser_len(&opening.main_trace_polys);
            s_precomputed_trace_openings += ser_len(&opening.precomputed_trace_polys);
            s_aux_trace_openings += ser_len(&opening.aux_trace_polys);
            s_composition_openings += ser_len(&opening.composition_poly);
        }
    }

    // Anything not captured above (composition_poly_root, fri_last_value,
    // nonce, public_inputs, trace_length, headers...). Calculate as the
    // bundle delta so the breakdown still sums to ~total.
    let accounted = main_mmcs_root_bytes
        + main_mmcs_spec_bytes
        + s_main_trace_openings
        + s_precomputed_trace_openings
        + s_aux_trace_openings
        + s_composition_openings
        + s_fri_query_list
        + s_fri_layers_roots
        + s_trace_ood
        + s_composition_ood
        + s_per_table_main_root
        + s_aux_root
        + s_precomputed_root
        + s_bus_public_inputs;
    s_other = multi_proof_bytes.saturating_sub(accounted);

    let entries: Vec<ProofSizeEntry> = vec![
        ProofSizeEntry { section: "main_mmcs_root", bytes: main_mmcs_root_bytes },
        ProofSizeEntry { section: "main_mmcs_spec", bytes: main_mmcs_spec_bytes },
        ProofSizeEntry { section: "per_table_main_merkle_root (preprocessed)", bytes: s_per_table_main_root },
        ProofSizeEntry { section: "per_table_precomputed_merkle_root", bytes: s_precomputed_root },
        ProofSizeEntry { section: "per_table_aux_merkle_root", bytes: s_aux_root },
        ProofSizeEntry { section: "deep_poly_openings.main_trace_polys", bytes: s_main_trace_openings },
        ProofSizeEntry { section: "deep_poly_openings.precomputed_trace_polys", bytes: s_precomputed_trace_openings },
        ProofSizeEntry { section: "deep_poly_openings.aux_trace_polys", bytes: s_aux_trace_openings },
        ProofSizeEntry { section: "deep_poly_openings.composition_poly", bytes: s_composition_openings },
        ProofSizeEntry { section: "fri_layers_merkle_roots", bytes: s_fri_layers_roots },
        ProofSizeEntry { section: "fri_query_list", bytes: s_fri_query_list },
        ProofSizeEntry { section: "trace_ood_evaluations", bytes: s_trace_ood },
        ProofSizeEntry { section: "composition_poly_parts_ood_evaluation", bytes: s_composition_ood },
        ProofSizeEntry { section: "bus_public_inputs", bytes: s_bus_public_inputs },
        ProofSizeEntry { section: "other (headers / public_inputs / nonce / ...)", bytes: s_other },
    ];

    if json {
        #[derive(serde::Serialize)]
        struct Report<'a> {
            elf: String,
            total_vm_proof_bytes: usize,
            multi_proof_bytes: usize,
            sub_proof_count: usize,
            main_mmcs_spec_entries: usize,
            sections: &'a [ProofSizeEntry],
        }
        let report = Report {
            elf: elf_path.display().to_string(),
            total_vm_proof_bytes: total,
            multi_proof_bytes,
            sub_proof_count: vm_proof.proof.proofs.len(),
            main_mmcs_spec_entries: vm_proof.proof.main_mmcs_spec.len(),
            sections: &entries,
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
        println!("MMCS spec entries: {:>10}", vm_proof.proof.main_mmcs_spec.len());
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

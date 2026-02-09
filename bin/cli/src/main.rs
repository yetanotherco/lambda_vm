//! Lambda VM CLI - execute, prove, and verify RISC-V programs.

mod proof_bundle;

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueHint};
use executor::{
    elf::{Elf, SymbolTable},
    flamegraph::FlamegraphGenerator,
    vm::execution::Executor,
};
use sha3::{Digest, Sha3_256};
use stark::proof::options::ProofOptions;

use proof_bundle::{PROOF_BUNDLE_VERSION, ProofBundle};

fn truncated_hex(bytes: &[u8]) -> String {
    format!("{}...", &hex::encode(bytes)[..16])
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
    },

    /// Verify a proof bundle
    Verify {
        /// Path to the proof bundle file
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        proof: PathBuf,

        /// Path to the ELF file (required for DECODE table verification)
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Commands::Execute { elf, flamegraph } => cmd_execute(elf, flamegraph),
        Commands::Prove { elf, output } => cmd_prove(elf, output),
        Commands::Verify { proof, elf } => cmd_verify(proof, elf),
    }
}

fn cmd_execute(elf_path: PathBuf, flamegraph_path: Option<PathBuf>) -> ExitCode {
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

    let mut executor = match Executor::new(&program, vec![]) {
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

fn cmd_prove(elf_path: PathBuf, output_path: PathBuf) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let elf_hash: [u8; 32] = Sha3_256::digest(&elf_data).into();
    let proof_options = ProofOptions::default_test_options();

    eprintln!("Generating proof (this may take a while)...");
    let multi_proof = match prover::prove_with_options(&elf_data, &proof_options) {
        Ok(proof) => proof,
        Err(e) => {
            eprintln!("Proof generation failed: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let bundle = ProofBundle::new(multi_proof, proof_options, elf_hash);

    eprintln!("Writing proof bundle...");
    let file = match File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create output file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let mut writer = BufWriter::new(file);

    if let Err(e) = ciborium::into_writer(&bundle, &mut writer) {
        eprintln!("Failed to serialize proof bundle: {}", e);
        return ExitCode::FAILURE;
    }

    if let Err(e) = writer.flush() {
        eprintln!("Failed to flush proof bundle: {}", e);
        return ExitCode::FAILURE;
    }

    eprintln!("Proof written to {:?}", output_path);
    eprintln!("  ELF hash: {}", truncated_hex(&elf_hash));

    ExitCode::SUCCESS
}

fn cmd_verify(proof_path: PathBuf, elf_path: PathBuf) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Reading proof bundle...");
    let file = match File::open(&proof_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open proof file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let reader = BufReader::new(file);

    let bundle: ProofBundle = match ciborium::from_reader(reader) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to deserialize proof bundle: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Validate proof bundle version
    if bundle.metadata.version != PROOF_BUNDLE_VERSION {
        eprintln!(
            "Unsupported proof bundle version: {} (expected {})",
            bundle.metadata.version, PROOF_BUNDLE_VERSION
        );
        return ExitCode::FAILURE;
    }

    // Detect wrong ELF early with a clear error message.
    // The cryptographic binding happens inside verify_with_options via the DECODE table.
    let elf_hash: [u8; 32] = Sha3_256::digest(&elf_data).into();
    if elf_hash != bundle.metadata.elf_hash {
        eprintln!("ELF hash mismatch: the proof was generated for a different program");
        return ExitCode::FAILURE;
    }

    eprintln!("Proof metadata:");
    eprintln!("  Version: {}", bundle.metadata.version);
    eprintln!("  ELF hash: {}", truncated_hex(&bundle.metadata.elf_hash));
    eprintln!("Verifying proof...");
    let result =
        match prover::verify_with_options(&bundle.multi_proof, &elf_data, &bundle.proof_options) {
            Ok(valid) => valid,
            Err(e) => {
                eprintln!("Verification error: {}", e);
                return ExitCode::FAILURE;
            }
        };

    if result {
        eprintln!("Verification succeeded!");
        ExitCode::SUCCESS
    } else {
        eprintln!("Verification failed!");
        ExitCode::FAILURE
    }
}

//! Lambda VM CLI - execute, prove, and verify RISC-V programs.

mod proof_bundle;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum, ValueHint};
use executor::{
    elf::{Elf, SymbolTable},
    flamegraph::FlamegraphGenerator,
    vm::execution::Executor,
};
use sha3::{Digest, Sha3_256};
use stark::proof::options::{ProofOptions, SecurityLevel};

use proof_bundle::{PROOF_BUNDLE_VERSION, ProofBundle};

/// Maximum ELF file size: 256 MB
const MAX_ELF_FILE_SIZE: u64 = 256 * 1024 * 1024;

/// Maximum proof bundle file size: 1 GB
const MAX_PROOF_FILE_SIZE: u64 = 1024 * 1024 * 1024;

/// Minimum acceptable blowup factor for proof verification
const MIN_BLOWUP_FACTOR: u8 = 4;

/// Maximum acceptable blowup factor (32 is very high, prevents memory exhaustion)
const MAX_BLOWUP_FACTOR: u8 = 32;

/// Minimum acceptable FRI queries for proof verification.
/// 41 = [`SecurityLevel::Conjecturable100Bits`] minimum from [`ProofOptions::new_secure`].
const MIN_FRI_QUERIES: usize = 41;

/// Maximum acceptable FRI queries (1000 is far above any reasonable security level)
const MAX_FRI_QUERIES: usize = 1000;

/// Minimum acceptable grinding factor for proof verification
const MIN_GRINDING_FACTOR: u8 = 1;

/// Maximum acceptable grinding factor (32 bits is very high)
const MAX_GRINDING_FACTOR: u8 = 32;

/// Maximum acceptable coset offset (reasonable upper bound)
const MAX_COSET_OFFSET: u64 = 1000;

fn truncated_hex(bytes: &[u8]) -> String {
    format!("{}...", &hex::encode(bytes)[..16])
}

fn read_file_with_limit(path: &Path, max_size: u64, file_type: &str) -> Result<Vec<u8>, String> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("Failed to get {} file metadata: {}", file_type, e))?;

    if metadata.len() > max_size {
        return Err(format!(
            "{} file too large: {} bytes (max: {} bytes)",
            file_type,
            metadata.len(),
            max_size
        ));
    }

    std::fs::read(path).map_err(|e| format!("Failed to read {} file: {}", file_type, e))
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

        /// Security level preset
        #[arg(long, value_enum, default_value = "standard")]
        security: SecurityPreset,
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

#[derive(Clone, Copy, ValueEnum)]
enum SecurityPreset {
    /// Conjecturable 100-bit security (development)
    Fast,
    /// Provable 100-bit security (default)
    Standard,
    /// Provable 128-bit security (production)
    Maximum,
}

impl SecurityPreset {
    fn to_proof_options(self) -> ProofOptions {
        const COSET_OFFSET: u64 = 3;
        match self {
            SecurityPreset::Fast => {
                ProofOptions::new_secure(SecurityLevel::Conjecturable100Bits, COSET_OFFSET)
            }
            SecurityPreset::Standard => {
                ProofOptions::new_secure(SecurityLevel::Provable100Bits, COSET_OFFSET)
            }
            SecurityPreset::Maximum => {
                ProofOptions::new_secure(SecurityLevel::Provable128Bits, COSET_OFFSET)
            }
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Commands::Execute { elf, flamegraph } => cmd_execute(elf, flamegraph),
        Commands::Prove {
            elf,
            output,
            security,
        } => cmd_prove(elf, output, security),
        Commands::Verify { proof, elf } => cmd_verify(proof, elf),
    }
}

fn cmd_execute(elf_path: PathBuf, flamegraph_path: Option<PathBuf>) -> ExitCode {
    let elf_data = match read_file_with_limit(&elf_path, MAX_ELF_FILE_SIZE, "ELF") {
        Ok(data) => data,
        Err(e) => {
            eprintln!("{}", e);
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

fn cmd_prove(elf_path: PathBuf, output_path: PathBuf, security: SecurityPreset) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match read_file_with_limit(&elf_path, MAX_ELF_FILE_SIZE, "ELF") {
        Ok(data) => data,
        Err(e) => {
            eprintln!("{}", e);
            return ExitCode::FAILURE;
        }
    };

    let elf_hash: [u8; 32] = Sha3_256::digest(&elf_data).into();
    let proof_options = security.to_proof_options();

    eprintln!("Generating proof (this may take a while)...");
    let multi_proof = match prover::prove_with_options(&elf_data, &proof_options) {
        Ok(proof) => proof,
        Err(e) => {
            eprintln!("Proof generation failed: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Get step count by re-executing (prove_with_options doesn't return it)
    // TODO: Consider returning step count from prove_with_options
    let program = Elf::load(&elf_data).unwrap();
    let executor = Executor::new(&program, vec![]).unwrap();
    let result = executor.run().unwrap();
    let num_steps = result.logs.len();

    let bundle = ProofBundle::new(multi_proof, proof_options, elf_hash, num_steps);

    eprintln!("Writing proof bundle...");
    let file = match File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create output file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let writer = BufWriter::new(file);

    if let Err(e) = ciborium::into_writer(&bundle, writer) {
        eprintln!("Failed to serialize proof bundle: {}", e);
        return ExitCode::FAILURE;
    }

    eprintln!("Proof written to {:?}", output_path);
    eprintln!("  ELF hash: {}", truncated_hex(&elf_hash));
    eprintln!("  Steps: {}", num_steps);

    ExitCode::SUCCESS
}

fn cmd_verify(proof_path: PathBuf, elf_path: PathBuf) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match read_file_with_limit(&elf_path, MAX_ELF_FILE_SIZE, "ELF") {
        Ok(data) => data,
        Err(e) => {
            eprintln!("{}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Reading proof bundle...");
    let proof_metadata = match std::fs::metadata(&proof_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to get proof file metadata: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if proof_metadata.len() > MAX_PROOF_FILE_SIZE {
        eprintln!(
            "Proof file too large: {} bytes (max: {} bytes)",
            proof_metadata.len(),
            MAX_PROOF_FILE_SIZE
        );
        return ExitCode::FAILURE;
    }

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

    // Validate proof options to prevent malicious bundles with weak or DoS parameters
    if bundle.proof_options.blowup_factor < MIN_BLOWUP_FACTOR
        || bundle.proof_options.blowup_factor > MAX_BLOWUP_FACTOR
    {
        eprintln!(
            "Invalid proof options: blowup_factor {} is out of valid range ({}-{})",
            bundle.proof_options.blowup_factor, MIN_BLOWUP_FACTOR, MAX_BLOWUP_FACTOR
        );
        return ExitCode::FAILURE;
    }
    if !bundle.proof_options.blowup_factor.is_power_of_two() {
        eprintln!("Invalid proof options: blowup_factor must be a power of two");
        return ExitCode::FAILURE;
    }
    if bundle.proof_options.fri_number_of_queries < MIN_FRI_QUERIES
        || bundle.proof_options.fri_number_of_queries > MAX_FRI_QUERIES
    {
        eprintln!(
            "Invalid proof options: fri_number_of_queries {} is out of valid range ({}-{})",
            bundle.proof_options.fri_number_of_queries, MIN_FRI_QUERIES, MAX_FRI_QUERIES
        );
        return ExitCode::FAILURE;
    }
    if bundle.proof_options.coset_offset == 0
        || bundle.proof_options.coset_offset > MAX_COSET_OFFSET
    {
        eprintln!(
            "Invalid proof options: coset_offset {} is out of valid range (1-{})",
            bundle.proof_options.coset_offset, MAX_COSET_OFFSET
        );
        return ExitCode::FAILURE;
    }
    if bundle.proof_options.grinding_factor < MIN_GRINDING_FACTOR
        || bundle.proof_options.grinding_factor > MAX_GRINDING_FACTOR
    {
        eprintln!(
            "Invalid proof options: grinding_factor {} is out of valid range ({}-{})",
            bundle.proof_options.grinding_factor, MIN_GRINDING_FACTOR, MAX_GRINDING_FACTOR
        );
        return ExitCode::FAILURE;
    }

    // Verify ELF hash matches proof metadata
    let elf_hash: [u8; 32] = Sha3_256::digest(&elf_data).into();
    if elf_hash != bundle.metadata.elf_hash {
        eprintln!("ELF hash mismatch: the proof was generated for a different program");
        return ExitCode::FAILURE;
    }

    eprintln!("Proof metadata:");
    eprintln!("  Version: {}", bundle.metadata.version);
    eprintln!("  ELF hash: {}", truncated_hex(&bundle.metadata.elf_hash));
    eprintln!("  Steps: {}", bundle.metadata.num_steps);

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

//! Lambda VM CLI
//!
//! A command-line interface for executing, proving, and verifying RISC-V ELF programs
//! using the Lambda VM zkVM (zero-knowledge virtual machine).
//!
//! # Commands
//!
//! The CLI provides three main commands:
//!
//! - **execute**: Run a program without generating a proof (fast, for testing)
//! - **prove**: Execute a program and generate a STARK proof of correct execution
//! - **verify**: Verify a previously generated proof (requires the original ELF file)
//!
//! # Architecture
//!
//! The proving system uses a multi-table STARK architecture with the following tables:
//!
//! - **CPU Table**: Main execution trace (registers, memory, control flow)
//! - **Bitwise Table**: Precomputed lookup table for AND, OR, XOR operations
//! - **LT Table**: Less-than comparison results
//! - **MEMW Table**: Memory write operations with timestamp ordering
//! - **LOAD Table**: Memory load operations
//! - **DECODE Table**: Instruction decoding verification (precomputed from ELF)
//!
//! Tables are linked via a LogUp bus protocol for cross-table lookups.
//!
//! # Security Levels
//!
//! The prover supports three security levels:
//!
//! | Level | Security | Blowup | Queries | Use Case |
//! |-------|----------|--------|---------|----------|
//! | fast | Conjecturable 100-bit | 4 | 41 | Development |
//! | standard | Provable 100-bit | 4 | 104 | Default |
//! | maximum | Provable 128-bit | 4 | 140 | Production |
//!
//! # Example Usage
//!
//! ```bash
//! # Execute without proving
//! lambda-vm execute program.elf
//!
//! # Generate a proof
//! lambda-vm prove program.elf -o proof.cbor --security fast
//!
//! # Verify the proof (requires the original ELF file)
//! lambda-vm verify proof.cbor program.elf
//! ```

mod proof_bundle;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use subtle::ConstantTimeEq;

use clap::{Parser, Subcommand, ValueEnum, ValueHint};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::{
    elf::{Elf, SymbolTable},
    flamegraph::FlamegraphGenerator,
    vm::execution::Executor,
};
use prover::tables::bitwise;
use prover::tables::decode;
use prover::tables::trace_builder::Traces;
use prover::tables::types::{GoldilocksExtension, GoldilocksField};
use prover::test_utils::{
    create_bitwise_air, create_cpu_air, create_decode_air, create_load_air, create_lt_air,
    create_memw_air,
};
use sha3::{Digest, Sha3_256};
use stark::proof::options::{ProofOptions, SecurityLevel};
use stark::prover::{IsStarkProver, Prover};
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use proof_bundle::{PROOF_BUNDLE_VERSION, ProofBundle};

type F = GoldilocksField;

/// Maximum ELF file size: 256 MB
const MAX_ELF_FILE_SIZE: u64 = 256 * 1024 * 1024;

/// Maximum proof bundle file size: 1 GB
const MAX_PROOF_FILE_SIZE: u64 = 1024 * 1024 * 1024;

/// Minimum acceptable blowup factor for proof verification
const MIN_BLOWUP_FACTOR: u8 = 4;

/// Minimum acceptable FRI queries for proof verification (31 = Conjecturable80Bits minimum)
const MIN_FRI_QUERIES: usize = 31;

/// Minimum acceptable grinding factor for proof verification
const MIN_GRINDING_FACTOR: u8 = 1;

/// Maximum acceptable coset offset (reasonable upper bound)
const MAX_COSET_OFFSET: u64 = 1000;

/// Read a file with size validation to prevent memory exhaustion.
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
type E = GoldilocksExtension;

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

/// Security level presets for proof generation.
///
/// These presets configure the STARK proof parameters to achieve different
/// security/performance tradeoffs. Higher security levels require more
/// computation and produce larger proofs.
///
/// # Security Guarantees
///
/// - **Conjecturable**: Security relies on commonly accepted cryptographic assumptions
/// - **Provable**: Security can be formally proven under standard assumptions
///
/// # Performance Impact
///
/// Higher security levels increase:
/// - Proof generation time (more FRI queries, larger blowup)
/// - Proof size (more query responses)
/// - Verification time (more queries to check)
#[derive(Clone, Copy, ValueEnum)]
enum SecurityPreset {
    /// Conjecturable 100-bit security - fast, suitable for development/testing.
    ///
    /// Uses minimal blowup factor and fewer FRI queries for faster proving.
    /// Suitable for testing and development where proof soundness is less critical.
    Fast,

    /// Provable 100-bit security - default, suitable for most use cases.
    ///
    /// Balanced security and performance. Recommended for most production uses
    /// where 100-bit security is sufficient.
    Standard,

    /// Provable 128-bit security - maximum security for production.
    ///
    /// Highest security level with larger blowup factor. Use for high-value
    /// applications where maximum security is required.
    Maximum,
}

impl SecurityPreset {
    /// Converts the preset to concrete proof options.
    ///
    /// The coset offset (3) is used to shift the evaluation domain away from
    /// roots of unity, which is required for proper FRI operation.
    fn to_proof_options(self) -> ProofOptions {
        // Coset offset shifts the LDE domain away from the trace domain.
        // Value 3 is a standard choice that works well with the Goldilocks field.
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

/// Execute command: run an ELF program without generating a proof.
///
/// This command is useful for:
/// - Testing that a program executes correctly before proving
/// - Debugging program behavior with register dumps
/// - Generating flamegraphs to profile execution
///
/// # Arguments
///
/// * `elf_path` - Path to the RISC-V ELF binary to execute
/// * `flamegraph_path` - Optional path to write flamegraph folded stacks
///
/// # Exit Codes
///
/// * `0` - Execution completed successfully
/// * `1` - Execution failed (file not found, invalid ELF, runtime error)
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

/// Prove command: execute a program and generate a STARK proof.
///
/// This command performs the full proving pipeline:
/// 1. Load and parse the ELF file
/// 2. Execute the program to generate execution logs
/// 3. Build execution traces for all VM tables
/// 4. Generate AIRs (Algebraic Intermediate Representations)
/// 5. Run the STARK prover to create a multi-proof
/// 6. Bundle the proof with metadata and serialize to CBOR
///
/// The output proof bundle can be verified with the `verify` command.
///
/// # Arguments
///
/// * `elf_path` - Path to the RISC-V ELF binary to prove
/// * `output_path` - Path to write the proof bundle (.cbor file)
/// * `security` - Security level preset (fast, standard, maximum)
///
/// # Exit Codes
///
/// * `0` - Proof generated successfully
/// * `1` - Proof generation failed
///
/// # Performance Notes
///
/// Proof generation time depends on:
/// - Number of instructions executed (trace size)
/// - Security level (affects blowup factor and query count)
/// - Available CPU cores (prover uses parallel FFT)
fn cmd_prove(elf_path: PathBuf, output_path: PathBuf, security: SecurityPreset) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match read_file_with_limit(&elf_path, MAX_ELF_FILE_SIZE, "ELF") {
        Ok(data) => data,
        Err(e) => {
            eprintln!("{}", e);
            return ExitCode::FAILURE;
        }
    };

    // Hash the ELF file
    let elf_hash: [u8; 32] = Sha3_256::digest(&elf_data).into();

    let program = match Elf::load(&elf_data) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to load ELF program: {:?}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Executing program...");
    let executor = match Executor::new(&program, vec![]) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create executor: {:?}", e);
            return ExitCode::FAILURE;
        }
    };

    let result = match executor.run() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Execution failed: {:?}", e);
            return ExitCode::FAILURE;
        }
    };

    let num_steps = result.logs.len();
    eprintln!("Executed {} instructions", num_steps);

    eprintln!("Generating traces...");
    let mut traces = match Traces::from_logs(&result.logs, result.instructions) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to generate traces: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let proof_options = security.to_proof_options();

    eprintln!("Creating AIRs...");
    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options).with_preprocessed(
        bitwise::preprocessed_commitment(),
        bitwise::NUM_PRECOMPUTED_COLS,
    );
    let lt_air = create_lt_air(&proof_options);
    let memw_air = create_memw_air(&proof_options);
    let load_air = create_load_air(&proof_options);
    let decode_commitment = match decode::commitment_from_elf(&program, &proof_options) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to compute decode commitment: {:?}", e);
            return ExitCode::FAILURE;
        }
    };
    let decode_air = create_decode_air(&proof_options)
        .with_preprocessed(decode_commitment, decode::NUM_PRECOMPUTED_COLS);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut traces.cpu, &()),
        (&bitwise_air, &mut traces.bitwise, &()),
        (&lt_air, &mut traces.lt, &()),
        (&memw_air, &mut traces.memw, &()),
        (&load_air, &mut traces.load, &()),
        (&decode_air, &mut traces.decode, &()),
    ];

    eprintln!("Generating proof (this may take a while)...");
    let multi_proof =
        match Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])) {
            Ok(proof) => proof,
            Err(e) => {
                eprintln!("Proof generation failed: {:?}", e);
                return ExitCode::FAILURE;
            }
        };

    // Create the proof bundle
    let bundle = ProofBundle::new(multi_proof, proof_options, elf_hash, num_steps);

    // Serialize and write to file
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
    eprintln!(
        "  ELF hash: {}",
        hex::encode(elf_hash).chars().take(16).collect::<String>() + "..."
    );
    eprintln!("  Steps: {}", num_steps);

    ExitCode::SUCCESS
}

/// Verify command: verify a STARK proof bundle.
///
/// This command verifies that a proof bundle represents valid execution
/// of a RISC-V program. Verification requires the original ELF file to:
/// - Compute the DECODE table commitment for verification
/// - Verify the ELF hash matches the proof metadata
///
/// The verification process:
/// 1. Load the ELF file and verify its hash matches the proof
/// 2. Deserialize the proof bundle from CBOR
/// 3. Reconstruct AIRs using the embedded proof options
/// 4. Compute the DECODE commitment from the ELF
/// 5. Run the STARK verifier on all table proofs
/// 6. Verify LogUp bus consistency across tables
///
/// # Arguments
///
/// * `proof_path` - Path to the proof bundle (.cbor file)
/// * `elf_path` - Path to the RISC-V ELF binary (must match the program that was proven)
///
/// # Exit Codes
///
/// * `0` - Proof is valid
/// * `1` - Proof is invalid or verification failed
///
/// # Security Notes
///
/// A valid proof guarantees that:
/// - The specific RISC-V program (identified by ELF hash) was executed correctly
/// - The execution followed RISC-V semantics
/// - Memory operations were consistent
/// - All table lookups were valid
/// - Instruction decoding was correct (via DECODE table)
fn cmd_verify(proof_path: PathBuf, elf_path: PathBuf) -> ExitCode {
    eprintln!("Reading ELF file...");
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

    eprintln!("Reading proof bundle...");
    // Check proof file size before reading
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

    // Validate proof options to prevent malicious bundles with weak parameters
    if bundle.proof_options.blowup_factor < MIN_BLOWUP_FACTOR {
        eprintln!(
            "Invalid proof options: blowup_factor {} is below minimum {}",
            bundle.proof_options.blowup_factor, MIN_BLOWUP_FACTOR
        );
        return ExitCode::FAILURE;
    }
    if !bundle.proof_options.blowup_factor.is_power_of_two() {
        eprintln!("Invalid proof options: blowup_factor must be a power of two");
        return ExitCode::FAILURE;
    }
    if bundle.proof_options.fri_number_of_queries < MIN_FRI_QUERIES {
        eprintln!(
            "Invalid proof options: fri_number_of_queries {} is below minimum {}",
            bundle.proof_options.fri_number_of_queries, MIN_FRI_QUERIES
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
    if bundle.proof_options.grinding_factor < MIN_GRINDING_FACTOR {
        eprintln!(
            "Invalid proof options: grinding_factor {} is below minimum {}",
            bundle.proof_options.grinding_factor, MIN_GRINDING_FACTOR
        );
        return ExitCode::FAILURE;
    }

    // Verify ELF hash matches proof metadata (constant-time comparison)
    let elf_hash: [u8; 32] = Sha3_256::digest(&elf_data).into();
    if elf_hash.ct_eq(&bundle.metadata.elf_hash).unwrap_u8() != 1 {
        eprintln!("ELF hash mismatch!");
        eprintln!(
            "  Expected: {}...",
            hex::encode(bundle.metadata.elf_hash)
                .chars()
                .take(16)
                .collect::<String>()
        );
        eprintln!(
            "  Got:      {}...",
            hex::encode(elf_hash).chars().take(16).collect::<String>()
        );
        eprintln!("The proof was generated for a different program.");
        return ExitCode::FAILURE;
    }

    eprintln!("Proof metadata:");
    eprintln!("  Version: {}", bundle.metadata.version);
    eprintln!(
        "  ELF hash: {}",
        hex::encode(bundle.metadata.elf_hash)
            .chars()
            .take(16)
            .collect::<String>()
            + "..."
    );
    eprintln!("  Steps: {}", bundle.metadata.num_steps);

    // Reconstruct AIRs with the same proof options
    let proof_options = &bundle.proof_options;
    let cpu_air = create_cpu_air(proof_options);
    let bitwise_air = create_bitwise_air(proof_options).with_preprocessed(
        bitwise::preprocessed_commitment(),
        bitwise::NUM_PRECOMPUTED_COLS,
    );
    let lt_air = create_lt_air(proof_options);
    let memw_air = create_memw_air(proof_options);
    let load_air = create_load_air(proof_options);
    let decode_commitment = match decode::commitment_from_elf(&program, proof_options) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to compute decode commitment: {:?}", e);
            return ExitCode::FAILURE;
        }
    };
    let decode_air = create_decode_air(proof_options)
        .with_preprocessed(decode_commitment, decode::NUM_PRECOMPUTED_COLS);

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![
        &cpu_air,
        &bitwise_air,
        &lt_air,
        &memw_air,
        &load_air,
        &decode_air,
    ];

    eprintln!("Verifying proof...");
    let result = Verifier::multi_verify(
        &airs,
        &bundle.multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
    );

    if result {
        eprintln!("Verification succeeded!");
        ExitCode::SUCCESS
    } else {
        eprintln!("Verification failed!");
        ExitCode::FAILURE
    }
}

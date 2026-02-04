//! STARK prover for the Lambda VM.
//!
//! Proves correct execution of RISC-V ELF binaries by generating
//! multi-table STARK proofs (CPU, bitwise, LT, memory, load).
//!
//! # Example
//! ```ignore
//! let elf_bytes = std::fs::read("program.elf").unwrap();
//! let proof = lambda_vm_prover::prove(&elf_bytes).unwrap();
//! assert!(lambda_vm_prover::verify(&proof));
//! ```

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

pub mod constraints;
pub mod tables;
pub mod test_utils;
pub mod tests;
pub mod utils;

use std::fmt;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use stark::prover::{IsStarkProver, Prover};
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::bitwise;
use crate::tables::trace_builder::Traces;
use crate::test_utils::{
    E, F, create_bitwise_air, create_branch_air, create_cpu_air, create_decode_air,
    create_halt_air, create_load_air, create_lt_air, create_memw_air, create_page_air,
    create_register_air,
};

use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;

/// Error type for the prover crate.
#[derive(Debug)]
pub enum Error {
    /// Failed to load ELF binary
    ElfLoad(String),
    /// Instruction not found for a given PC address
    MissingInstruction(u64),
    /// Program does not contain an ECALL (halt) instruction
    MissingHaltEcall,
    /// Executor failed (setup or runtime error)
    Execution(String),
    /// STARK proving failed
    Prover(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ElfLoad(msg) => write!(f, "ELF load error: {msg}"),
            Error::MissingInstruction(pc) => write!(f, "instruction not found for PC {pc:#x}"),
            Error::MissingHaltEcall => {
                write!(f, "program does not contain an ECALL (halt) instruction")
            }
            Error::Execution(msg) => write!(f, "execution error: {msg}"),
            Error::Prover(msg) => write!(f, "proving error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Prove an ELF binary execution. Returns a serializable proof.
pub fn prove(elf_bytes: &[u8]) -> Result<MultiProof<F, E, ()>, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let executor = Executor::new(&program, vec![]).map_err(|e| Error::Execution(format!("{e}")))?;
    let result = executor
        .run()
        .map_err(|e| Error::Execution(format!("{e}")))?;

    // Generate all traces from ELF and execution logs
    // This uses the combined ELF processing to generate DECODE and PAGE tables
    let mut traces = Traces::from_elf_and_logs(&program, &result.logs)?;

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options).with_preprocessed(
        bitwise::preprocessed_commitment(),
        bitwise::NUM_PRECOMPUTED_COLS,
    );
    let lt_air = create_lt_air(&proof_options);
    let memw_air = create_memw_air(&proof_options);
    let load_air = create_load_air(&proof_options);
    let decode_air = create_decode_air(&proof_options);
    let branch_air = create_branch_air(&proof_options);
    let halt_air = create_halt_air(&proof_options);

    // Create PAGE AIRs (one per page, each with its own page_base constant)
    let page_airs: Vec<_> = traces
        .page_configs
        .iter()
        .map(|config| create_page_air(&proof_options, config.page_base))
        .collect();

    // Create REGISTER AIR
    let register_air = create_register_air(&proof_options);

    // Build air_trace_pairs for core tables
    let mut air_trace_pairs: Vec<(
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
        (&branch_air, &mut traces.branch, &()),
        (&halt_air, &mut traces.halt, &()),
        (&register_air, &mut traces.register, &()),
    ];

    // Add PAGE table pairs
    for (i, page_trace) in traces.pages.iter_mut().enumerate() {
        air_trace_pairs.push((&page_airs[i], page_trace, &()));
    }

    Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
        .map_err(|e| Error::Prover(format!("{e:?}")))
}

/// Verify a proof produced by [`prove`].
pub fn verify(proof: &MultiProof<F, E, ()>) -> bool {
    let proof_options = ProofOptions::default_test_options();
    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options).with_preprocessed(
        bitwise::preprocessed_commitment(),
        bitwise::NUM_PRECOMPUTED_COLS,
    );
    let lt_air = create_lt_air(&proof_options);
    let memw_air = create_memw_air(&proof_options);
    let load_air = create_load_air(&proof_options);
    let decode_air = create_decode_air(&proof_options);
    let branch_air = create_branch_air(&proof_options);
    let halt_air = create_halt_air(&proof_options);

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![
        &cpu_air,
        &bitwise_air,
        &lt_air,
        &memw_air,
        &load_air,
        &decode_air,
        &branch_air,
        &halt_air,
    ];

    Verifier::multi_verify(&airs, proof, &mut DefaultTranscript::<E>::new(&[]))
}

/// Prove and verify in one call (convenience).
pub fn prove_and_verify(elf_bytes: &[u8]) -> Result<bool, Error> {
    let proof = prove(elf_bytes)?;
    Ok(verify(&proof))
}

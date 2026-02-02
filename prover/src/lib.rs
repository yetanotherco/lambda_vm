//! STARK prover for the Lambda VM.
//!
//! Proves correct execution of RISC-V ELF binaries by generating
//! multi-table STARK proofs (CPU, bitwise, LT, memory, load).
//!
//! # Example
//! ```no_run
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
    E, F, create_bitwise_air, create_cpu_air, create_load_air, create_lt_air, create_memw_air,
};

use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;

/// Error type for the prover crate.
#[derive(Debug)]
pub enum Error {
    /// Failed to load ELF binary
    ElfLoad(String),
    /// Trace generation failed (includes missing instructions, execution errors)
    TraceGeneration(String),
    /// STARK proving failed
    Prover(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ElfLoad(msg) => write!(f, "ELF load error: {msg}"),
            Error::TraceGeneration(msg) => write!(f, "trace generation error: {msg}"),
            Error::Prover(msg) => write!(f, "proving error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Prove an ELF binary execution. Returns a serializable proof.
pub fn prove(elf_bytes: &[u8]) -> Result<MultiProof<F, E, ()>, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let executor =
        Executor::new(&program, vec![]).map_err(|e| Error::TraceGeneration(format!("{e}")))?;
    let result = executor
        .run()
        .map_err(|e| Error::TraceGeneration(format!("{e}")))?;

    let mut traces = Traces::from_logs(&result.logs, result.instructions)?;

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options).with_preprocessed(
        bitwise::preprocessed_commitment(),
        bitwise::NUM_PRECOMPUTED_COLS,
    );
    let lt_air = create_lt_air(&proof_options);
    let memw_air = create_memw_air(&proof_options);
    let load_air = create_load_air(&proof_options);

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
    ];

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

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &bitwise_air, &lt_air, &memw_air, &load_air];

    Verifier::multi_verify(&airs, proof, &mut DefaultTranscript::<E>::new(&[]))
}

/// Prove and verify in one call (convenience).
pub fn prove_and_verify(elf_bytes: &[u8]) -> Result<bool, Error> {
    let proof = prove(elf_bytes)?;
    Ok(verify(&proof))
}

/// Prove with minimal (unsound) bitwise table — fast, for testing only.
///
/// The minimal table only contains rows needed to balance the bus,
/// rather than the full 2^20 row preprocessed table.
pub fn prove_minimal(elf_bytes: &[u8]) -> Result<MultiProof<F, E, ()>, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let executor =
        Executor::new(&program, vec![]).map_err(|e| Error::TraceGeneration(format!("{e}")))?;
    let result = executor
        .run()
        .map_err(|e| Error::TraceGeneration(format!("{e}")))?;

    let mut traces = Traces::from_logs(&result.logs, result.instructions)?;
    traces.bitwise = bitwise::trim_zero_rows(traces.bitwise);

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options);
    let lt_air = create_lt_air(&proof_options);
    let memw_air = create_memw_air(&proof_options);
    let load_air = create_load_air(&proof_options);

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
    ];

    Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
        .map_err(|e| Error::Prover(format!("{e:?}")))
}

/// Verify a proof produced by [`prove_minimal`].
///
/// Unlike [`verify`], this does not check the preprocessed bitwise commitment.
pub fn verify_minimal(proof: &MultiProof<F, E, ()>) -> bool {
    let proof_options = ProofOptions::default_test_options();
    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options);
    let lt_air = create_lt_air(&proof_options);
    let memw_air = create_memw_air(&proof_options);
    let load_air = create_load_air(&proof_options);

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &bitwise_air, &lt_air, &memw_air, &load_air];

    Verifier::multi_verify(&airs, proof, &mut DefaultTranscript::<E>::new(&[]))
}

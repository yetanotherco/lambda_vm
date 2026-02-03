//! STARK prover for the Lambda VM.
//!
//! Proves correct execution of RISC-V ELF binaries by generating
//! multi-table STARK proofs (CPU, bitwise, LT, memory, load).
//!
//! # Example
//! ```ignore
//! let elf_bytes = std::fs::read("program.elf").unwrap();
//! let proof = lambda_vm_prover::prove(&elf_bytes).unwrap();
//! assert!(lambda_vm_prover::verify(&proof, &elf_bytes).unwrap());
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
use crate::tables::decode;
use crate::tables::trace_builder::Traces;
use crate::test_utils::{
    E, F, VmAir, create_bitwise_air, create_branch_air, create_cpu_air, create_decode_air,
    create_halt_air, create_load_air, create_lt_air, create_memw_air, create_mul_air,
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

/// Type alias for AIR-trace-public-inputs triples used in multi-table proving.
type AirTracePair<'a> = (
    &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
    &'a mut stark::trace::TraceTable<F, E>,
    &'a (),
);

/// All VM AIR instances, grouped by table.
pub(crate) struct VmAirs {
    pub cpu: VmAir,
    pub bitwise: VmAir,
    pub lt: VmAir,
    pub memw: VmAir,
    pub load: VmAir,
    pub decode: VmAir,
    pub mul: VmAir,
    pub branch: VmAir,
    pub halt: VmAir,
}

impl VmAirs {
    /// Build `(air, trace, public_inputs)` triples for [`Prover::multi_prove`].
    pub fn air_trace_pairs<'a>(&'a self, traces: &'a mut Traces) -> Vec<AirTracePair<'a>> {
        vec![
            (&self.cpu, &mut traces.cpu, &()),
            (&self.bitwise, &mut traces.bitwise, &()),
            (&self.lt, &mut traces.lt, &()),
            (&self.memw, &mut traces.memw, &()),
            (&self.load, &mut traces.load, &()),
            (&self.decode, &mut traces.decode, &()),
            (&self.mul, &mut traces.mul, &()),
            (&self.branch, &mut traces.branch, &()),
            (&self.halt, &mut traces.halt, &()),
        ]
    }

    /// Collect AIR references for [`Verifier::multi_verify`].
    pub fn air_refs(&self) -> Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> {
        vec![
            &self.cpu,
            &self.bitwise,
            &self.lt,
            &self.memw,
            &self.load,
            &self.decode,
            &self.mul,
            &self.branch,
            &self.halt,
        ]
    }

    /// Create all VM AIR instances. `minimal_bitwise` controls whether the full
    /// 2^20 bitwise preprocessed table is included (false = full, true = minimal).
    /// DECODE is always preprocessed.
    pub fn new(elf: &Elf, proof_options: &ProofOptions, minimal_bitwise: bool) -> Self {
        let cpu = create_cpu_air(proof_options);
        let bitwise = if minimal_bitwise {
            create_bitwise_air(proof_options)
        } else {
            create_bitwise_air(proof_options)
                .with_preprocessed(
                    bitwise::preprocessed_commitment(),
                    bitwise::NUM_PRECOMPUTED_COLS,
                )
                .with_precomputed_cache(
                    bitwise::precomputed_polynomials(),
                    bitwise::precomputed_lde_columns(),
                )
        };
        let lt = create_lt_air(proof_options);
        let memw = create_memw_air(proof_options);
        let load = create_load_air(proof_options);
        let decode = create_decode_air(proof_options).with_preprocessed(
            decode::commitment_from_elf(elf, proof_options)
                .expect("Failed to compute decode commitment"),
            decode::NUM_PRECOMPUTED_COLS,
        );
        let mul = create_mul_air(proof_options);
        let branch = create_branch_air(proof_options);
        let halt = create_halt_air(proof_options);
        Self {
            cpu,
            bitwise,
            lt,
            memw,
            load,
            decode,
            mul,
            branch,
            halt,
        }
    }
}

/// Prove an ELF binary execution. Returns a serializable proof.
pub fn prove(elf_bytes: &[u8]) -> Result<MultiProof<F, E, ()>, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let executor = Executor::new(&program, vec![]).map_err(|e| Error::Execution(format!("{e}")))?;
    let result = executor
        .run()
        .map_err(|e| Error::Execution(format!("{e}")))?;

    let mut traces = Traces::from_logs(&result.logs, result.instructions)?;

    let proof_options = ProofOptions::default_test_options();
    let airs = VmAirs::new(&program, &proof_options, false);

    Prover::multi_prove(
        airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))
}

/// Verify a proof produced by [`prove`].
pub fn verify(proof: &MultiProof<F, E, ()>, elf_bytes: &[u8]) -> Result<bool, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let proof_options = ProofOptions::default_test_options();
    let airs = VmAirs::new(&program, &proof_options, false);

    Ok(Verifier::multi_verify(
        &airs.air_refs(),
        proof,
        &mut DefaultTranscript::<E>::new(&[]),
    ))
}

/// Prove and verify in one call (convenience).
pub fn prove_and_verify(elf_bytes: &[u8]) -> Result<bool, Error> {
    let proof = prove(elf_bytes)?;
    verify(&proof, elf_bytes)
}

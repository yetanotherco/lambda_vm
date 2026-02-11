//! STARK prover for the Lambda VM.
//!
//! Proves correct execution of RISC-V ELF binaries by generating
//! multi-table STARK proofs (CPU, bitwise, LT, memory, load).
//!
//! # Example
//! ```ignore
//! let elf_bytes = std::fs::read("program.elf").unwrap();
//! let vm_proof = lambda_vm_prover::prove(&elf_bytes).unwrap();
//! assert!(lambda_vm_prover::verify(&vm_proof, &elf_bytes).unwrap());
//! ```

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

pub mod constraints;
#[cfg(feature = "debug-checks")]
mod debug_report;
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
use crate::tables::page;
use crate::tables::register;
use crate::tables::trace_builder::Traces;
use crate::test_utils::{
    E, F, VmAir, create_bitwise_air, create_branch_air, create_cpu_air, create_decode_air,
    create_dvrm_air, create_halt_air, create_load_air, create_lt_air, create_memw_air,
    create_mul_air, create_page_air, create_register_air,
};

use crate::tables::page::DEFAULT_STACK_SIZE;
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;

/// A complete VM proof bundle containing the STARK proof and metadata
/// needed by the verifier to reconstruct the AIR configuration.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct VmProof {
    /// The multi-table STARK proof.
    pub proof: MultiProof<F, E, ()>,
    /// Stack size used during proving (bytes). The verifier uses this to
    /// reconstruct the PAGE table layout.
    pub stack_size: u64,
}

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
    pub dvrm: VmAir,
    pub branch: VmAir,
    pub halt: VmAir,
    pub register: VmAir,
    pub pages: Vec<VmAir>,
}

impl VmAirs {
    /// Build `(air, trace, public_inputs)` triples for [`Prover::multi_prove`].
    pub fn air_trace_pairs<'a>(&'a self, traces: &'a mut Traces) -> Vec<AirTracePair<'a>> {
        let mut pairs: Vec<AirTracePair<'a>> = vec![
            (&self.cpu, &mut traces.cpu, &()),
            (&self.bitwise, &mut traces.bitwise, &()),
            (&self.lt, &mut traces.lt, &()),
            (&self.memw, &mut traces.memw, &()),
            (&self.load, &mut traces.load, &()),
            (&self.decode, &mut traces.decode, &()),
            (&self.mul, &mut traces.mul, &()),
            (&self.dvrm, &mut traces.dvrm, &()),
            (&self.branch, &mut traces.branch, &()),
            (&self.halt, &mut traces.halt, &()),
            (&self.register, &mut traces.register, &()),
        ];
        for (i, page_trace) in traces.pages.iter_mut().enumerate() {
            pairs.push((&self.pages[i], page_trace, &()));
        }
        pairs
    }

    /// Collect AIR references for [`Verifier::multi_verify`].
    pub fn air_refs(&self) -> Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> {
        let mut refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![
            &self.cpu,
            &self.bitwise,
            &self.lt,
            &self.memw,
            &self.load,
            &self.decode,
            &self.mul,
            &self.dvrm,
            &self.branch,
            &self.halt,
            &self.register,
        ];
        for page in &self.pages {
            refs.push(page);
        }
        refs
    }

    /// Create all VM AIR instances. `minimal_bitwise` controls whether the full
    /// 2^20 bitwise preprocessed table is included (false = full, true = minimal).
    /// DECODE is always preprocessed.
    ///
    /// `page_configs` provides the page base addresses for creating PAGE AIRs.
    pub fn new(
        elf: &Elf,
        proof_options: &ProofOptions,
        minimal_bitwise: bool,
        page_configs: &[crate::tables::page::PageConfig],
    ) -> Self {
        let cpu = create_cpu_air(proof_options);
        let bitwise = if minimal_bitwise {
            create_bitwise_air(proof_options)
        } else {
            create_bitwise_air(proof_options).with_preprocessed(
                bitwise::preprocessed_commitment(),
                bitwise::NUM_PRECOMPUTED_COLS,
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
        let dvrm = create_dvrm_air(proof_options);
        let branch = create_branch_air(proof_options);
        let halt = create_halt_air(proof_options);
        let register = create_register_air(proof_options).with_preprocessed(
            register::preprocessed_commitment(proof_options),
            register::NUM_PREPROCESSED_COLS,
        );
        let pages: Vec<_> = page_configs
            .iter()
            .map(|config| {
                create_page_air(proof_options, config.page_base).with_preprocessed(
                    page::precomputed_commitment_cached(config, proof_options),
                    page::NUM_PREPROCESSED_COLS,
                )
            })
            .collect();

        #[cfg(feature = "debug-checks")]
        debug_report::print_bus_legend();

        Self {
            cpu,
            bitwise,
            lt,
            memw,
            load,
            decode,
            mul,
            dvrm,
            branch,
            halt,
            register,
            pages,
        }
    }
}

/// Prove an ELF binary execution. Returns a serializable proof bundle.
pub fn prove(elf_bytes: &[u8]) -> Result<VmProof, Error> {
    prove_with_options(elf_bytes, &ProofOptions::default_proving_options())
}

/// Prove an ELF binary execution with custom proof options and stack size.
pub fn prove_with_options(
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
) -> Result<VmProof, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let executor = Executor::new(&program, vec![]).map_err(|e| Error::Execution(format!("{e}")))?;
    let result = executor
        .run()
        .map_err(|e| Error::Execution(format!("{e}")))?;

    // Generate all traces from ELF and execution logs
    // This uses the combined ELF processing to generate DECODE and PAGE tables
    let mut traces = Traces::from_elf_and_logs(&program, &result.logs, DEFAULT_STACK_SIZE)?;
    let airs = VmAirs::new(&program, proof_options, false, &traces.page_configs);

    let proof = Prover::multi_prove(
        airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))?;

    Ok(VmProof {
        proof,
        stack_size: DEFAULT_STACK_SIZE,
    })
}

/// Verify a proof produced by [`prove`] using default proof options.
///
/// Uses [`ProofOptions::default_proving_options`] for verification — the
/// `proof_options` stored in [`VmProof`] are metadata only and NOT trusted.
/// `stack_size` is extracted from the proof; it is safe to trust because
/// preprocessed commitments bind the verifier to the correct page layout.
pub fn verify(vm_proof: &VmProof, elf_bytes: &[u8]) -> Result<bool, Error> {
    verify_with_options(
        vm_proof,
        elf_bytes,
        &ProofOptions::default_proving_options(),
    )
}

/// Verify a proof with caller-specified proof options.
///
/// The verifier enforces its own `proof_options` (security parameters),
/// ignoring the options embedded in the proof bundle. This prevents a
/// malicious prover from weakening the security level.
pub fn verify_with_options(
    vm_proof: &VmProof,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
) -> Result<bool, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let page_configs = Traces::page_configs_from_elf(&program, vm_proof.stack_size);
    let airs = VmAirs::new(&program, proof_options, false, &page_configs);

    Ok(Verifier::multi_verify(
        &airs.air_refs(),
        &vm_proof.proof,
        &mut DefaultTranscript::<E>::new(&[]),
    ))
}

/// Prove and verify in one call (convenience).
pub fn prove_and_verify(elf_bytes: &[u8]) -> Result<bool, Error> {
    let vm_proof = prove(elf_bytes)?;
    verify(&vm_proof, elf_bytes)
}

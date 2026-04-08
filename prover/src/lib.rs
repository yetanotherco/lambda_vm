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

pub mod constraints;
#[cfg(feature = "debug-checks")]
mod debug_report;
#[cfg(feature = "instruments")]
pub mod instruments;
pub mod tables;
pub mod test_utils;
#[cfg(test)]
pub mod tests;

use std::fmt;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use math::field::element::FieldElement;
use stark::prover::{IsStarkProver, Prover};
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

pub use crate::tables::MaxRowsConfig;
use crate::tables::bitwise;
use crate::tables::decode;
use crate::tables::page;
use crate::tables::register;
use crate::tables::trace_builder::Traces;
use crate::tables::types::BusId;
use crate::test_utils::{
    E, F, VmAir, create_bitwise_air, create_branch_air, create_commit_air, create_cpu_air,
    create_decode_air, create_dvrm_air, create_halt_air, create_load_air, create_lt_air,
    create_memw_air, create_memw_aligned_air, create_memw_register_air, create_mul_air,
    create_page_air, create_register_air, create_shift_air,
};

use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::stark::MultiProof;

/// A run-length encoded range of contiguous zero-initialized 4KB pages.
///
/// Represents `count` contiguous pages starting at `base`, used for
/// runtime-allocated memory (stack, heap) not covered by ELF segments.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RuntimePageRange {
    /// Base address of the first page (4KB-aligned).
    pub base: u64,
    /// Number of contiguous 4KB pages starting at `base`.
    pub count: u64,
}

/// Number of chunks for each split table.
/// The verifier needs this to reconstruct matching AIRs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TableCounts {
    pub cpu: usize,
    pub lt: usize,
    pub memw: usize,
    pub memw_aligned: usize,
    pub load: usize,
    pub mul: usize,
    pub dvrm: usize,
    pub shift: usize,
    pub branch: usize,
    pub memw_register: usize,
}

impl TableCounts {
    /// Validate that all required tables have at least one chunk.
    ///
    /// A zero count for any table would remove its constraints from verification,
    /// allowing a malicious prover to bypass soundness checks.
    /// Sum of all chunk counts across split tables.
    pub fn total(&self) -> usize {
        self.cpu
            + self.lt
            + self.memw
            + self.memw_aligned
            + self.load
            + self.mul
            + self.dvrm
            + self.shift
            + self.branch
            + self.memw_register
    }

    /// Validate that all required tables have at least one chunk.
    ///
    /// A zero count for any table would remove its constraints from verification,
    /// allowing a malicious prover to bypass soundness checks.
    pub fn validate(&self) -> Result<(), Error> {
        let checks = [
            ("cpu", self.cpu),
            ("lt", self.lt),
            ("memw", self.memw),
            ("memw_aligned", self.memw_aligned),
            ("load", self.load),
            ("mul", self.mul),
            ("dvrm", self.dvrm),
            ("shift", self.shift),
            ("branch", self.branch),
            ("memw_register", self.memw_register),
        ];
        for (name, count) in checks {
            if count == 0 {
                return Err(Error::InvalidTableCounts(format!(
                    "{name} count is 0 — every table must have at least 1 chunk"
                )));
            }
        }
        Ok(())
    }
}

/// A complete VM proof bundle containing the STARK proof and metadata
/// needed by the verifier to reconstruct the AIR configuration.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct VmProof {
    /// The multi-table STARK proof.
    pub proof: MultiProof<F, E, ()>,
    /// Run-length encoded runtime page ranges.
    /// These are zero-initialized pages accessed during execution but not
    /// covered by ELF segments (stack, heap, etc.).
    pub runtime_page_ranges: Vec<RuntimePageRange>,
    /// Number of chunks for each split table.
    /// The verifier needs this to reconstruct matching AIRs.
    pub table_counts: TableCounts,
    /// Committed public output bytes.
    pub public_output: Vec<u8>,
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
    /// Proof contains invalid table_counts (e.g. zero for a required table)
    InvalidTableCounts(String),
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
            Error::InvalidTableCounts(msg) => write!(f, "invalid table_counts: {msg}"),
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
    pub cpus: Vec<VmAir>,
    pub bitwise: VmAir,
    pub lts: Vec<VmAir>,
    pub shifts: Vec<VmAir>,
    pub memws: Vec<VmAir>,
    pub memw_aligneds: Vec<VmAir>,
    pub loads: Vec<VmAir>,
    pub decode: VmAir,
    pub muls: Vec<VmAir>,
    pub dvrms: Vec<VmAir>,
    pub branches: Vec<VmAir>,
    pub halt: Option<VmAir>,
    pub commit: Option<VmAir>,
    pub register: VmAir,
    pub pages: Vec<VmAir>,
    pub memw_registers: Vec<VmAir>,
}

impl VmAirs {
    /// Build `(air, trace, public_inputs)` triples for [`Prover::multi_prove`].
    pub fn air_trace_pairs<'a>(&'a self, traces: &'a mut Traces) -> Vec<AirTracePair<'a>> {
        let mut pairs: Vec<AirTracePair<'a>> = vec![
            (&self.bitwise, &mut traces.bitwise, &()),
            (&self.decode, &mut traces.decode, &()),
            (&self.register, &mut traces.register, &()),
        ];
        if let Some(ref halt) = self.halt {
            pairs.push((halt, &mut traces.halt, &()));
        }
        if let Some(ref commit) = self.commit {
            pairs.push((commit, &mut traces.commit, &()));
        }

        for (air, trace) in self.cpus.iter().zip(traces.cpus.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.lts.iter().zip(traces.lts.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.shifts.iter().zip(traces.shifts.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.memws.iter().zip(traces.memws.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self
            .memw_aligneds
            .iter()
            .zip(traces.memw_aligneds.iter_mut())
        {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.loads.iter().zip(traces.loads.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.muls.iter().zip(traces.muls.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.dvrms.iter().zip(traces.dvrms.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.branches.iter().zip(traces.branches.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.pages.iter().zip(traces.pages.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self
            .memw_registers
            .iter()
            .zip(traces.memw_registers.iter_mut())
        {
            pairs.push((air, trace, &()));
        }

        pairs
    }

    /// Collect AIR references for [`Verifier::multi_verify`].
    pub fn air_refs(&self) -> Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> {
        let mut refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![
            &self.bitwise,
            &self.decode,
            &self.register,
        ];
        if let Some(ref halt) = self.halt {
            refs.push(halt);
        }
        if let Some(ref commit) = self.commit {
            refs.push(commit);
        }

        for air in &self.cpus {
            refs.push(air);
        }
        for air in &self.lts {
            refs.push(air);
        }
        for air in &self.shifts {
            refs.push(air);
        }
        for air in &self.memws {
            refs.push(air);
        }
        for air in &self.memw_aligneds {
            refs.push(air);
        }
        for air in &self.loads {
            refs.push(air);
        }
        for air in &self.muls {
            refs.push(air);
        }
        for air in &self.dvrms {
            refs.push(air);
        }
        for air in &self.branches {
            refs.push(air);
        }
        for air in &self.pages {
            refs.push(air);
        }
        for air in &self.memw_registers {
            refs.push(air);
        }

        refs
    }

    /// Create all VM AIR instances. `minimal_bitwise` controls whether the full
    /// 2^20 bitwise preprocessed table is included (false = full, true = minimal).
    /// DECODE is always preprocessed.
    ///
    /// `page_configs` provides the page base addresses and init values for PAGE AIRs.
    /// `table_counts` specifies how many chunks for each split table.
    ///
    /// For sharded proving: `register_commitment` overrides the REGISTER preprocessed
    /// commitment (computed from the segment's initial register state instead of ELF defaults).
    /// If `None`, uses the default ELF-based commitment.
    pub fn new(
        elf: &Elf,
        proof_options: &ProofOptions,
        minimal_bitwise: bool,
        page_configs: &[crate::tables::page::PageConfig],
        table_counts: &TableCounts,
    ) -> Self {
        Self::new_with_overrides(elf, proof_options, minimal_bitwise, page_configs, table_counts, None, true, true)
    }

    /// Create all VM AIR instances with optional overrides for sharded proving.
    ///
    /// - `register_commitment`: override REGISTER preprocessed commitment (for non-first segments)
    /// - `include_halt`: include HALT table (only for last segment)
    /// - `include_commit`: include COMMIT table (only if segment has commit ops)
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_overrides(
        elf: &Elf,
        proof_options: &ProofOptions,
        minimal_bitwise: bool,
        page_configs: &[crate::tables::page::PageConfig],
        table_counts: &TableCounts,
        register_commitment: Option<stark::config::Commitment>,
        include_halt: bool,
        include_commit: bool,
    ) -> Self {
        let cpus: Vec<_> = (0..table_counts.cpu)
            .map(|i| create_cpu_air(proof_options).with_name(&format!("CPU[{}]", i)))
            .collect();
        let bitwise = if minimal_bitwise {
            create_bitwise_air(proof_options)
        } else {
            create_bitwise_air(proof_options).with_preprocessed(
                bitwise::preprocessed_commitment(proof_options),
                bitwise::NUM_PRECOMPUTED_COLS,
            )
        };
        let lts: Vec<_> = (0..table_counts.lt)
            .map(|i| create_lt_air(proof_options).with_name(&format!("LT[{}]", i)))
            .collect();
        let shifts: Vec<_> = (0..table_counts.shift)
            .map(|i| create_shift_air(proof_options).with_name(&format!("SHIFT[{}]", i)))
            .collect();
        let memws: Vec<_> = (0..table_counts.memw)
            .map(|i| create_memw_air(proof_options).with_name(&format!("MEMW[{}]", i)))
            .collect();
        let memw_aligneds: Vec<_> = (0..table_counts.memw_aligned)
            .map(|i| create_memw_aligned_air(proof_options).with_name(&format!("MEMW_A[{}]", i)))
            .collect();
        let loads: Vec<_> = (0..table_counts.load)
            .map(|i| create_load_air(proof_options).with_name(&format!("LOAD[{}]", i)))
            .collect();
        let decode = create_decode_air(proof_options).with_preprocessed(
            decode::commitment_from_elf(elf, proof_options)
                .expect("Failed to compute decode commitment"),
            decode::NUM_PRECOMPUTED_COLS,
        );
        let muls: Vec<_> = (0..table_counts.mul)
            .map(|i| create_mul_air(proof_options).with_name(&format!("MUL[{}]", i)))
            .collect();
        let dvrms: Vec<_> = (0..table_counts.dvrm)
            .map(|i| create_dvrm_air(proof_options).with_name(&format!("DVRM[{}]", i)))
            .collect();
        let branches: Vec<_> = (0..table_counts.branch)
            .map(|i| create_branch_air(proof_options).with_name(&format!("BRANCH[{}]", i)))
            .collect();
        let halt = if include_halt {
            Some(create_halt_air(proof_options))
        } else {
            None
        };
        let commit = if include_commit {
            Some(create_commit_air(proof_options))
        } else {
            None
        };
        let reg_commitment = register_commitment
            .unwrap_or_else(|| register::preprocessed_commitment(proof_options, elf.entry_point));
        let register = create_register_air(proof_options).with_preprocessed(
            reg_commitment,
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
        let memw_registers: Vec<_> = (0..table_counts.memw_register)
            .map(|i| create_memw_register_air(proof_options).with_name(&format!("MEMW_R[{}]", i)))
            .collect();

        #[cfg(feature = "debug-checks")]
        debug_report::print_bus_legend();

        Self {
            cpus,
            bitwise,
            lts,
            shifts,
            memws,
            memw_aligneds,
            loads,
            decode,
            muls,
            dvrms,
            branches,
            halt,
            commit,
            register,
            pages,
            memw_registers,
        }
    }
}

// =============================================================================
// Bus Balance Target: Verifier-Computed COMMIT Output Bus
// =============================================================================

/// Replay the prover's Phase A (main trace commitments) to recover the shared
/// LogUp challenges (z, alpha). Creates a fresh transcript, appends all main
/// trace commitments in the same order as the prover, then samples two
/// challenge elements.
pub(crate) fn replay_transcript_phase_a(
    airs: &[&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>],
    multi_proof: &MultiProof<F, E, ()>,
) -> (FieldElement<E>, FieldElement<E>) {
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    for (air, proof) in airs.iter().zip(&multi_proof.proofs) {
        if air.is_preprocessed() {
            transcript.append_bytes(&air.precomputed_commitment());
            transcript.append_bytes(&proof.lde_trace_main_merkle_root);
        } else {
            transcript.append_bytes(&proof.lde_trace_main_merkle_root);
        }
    }
    let z: FieldElement<E> = transcript.sample_field_element();
    let alpha: FieldElement<E> = transcript.sample_field_element();
    (z, alpha)
}

/// Compute the bus balance offset for the COMMIT[index, value] bus.
///
/// For each public output byte at index `i` with value `v`:
///   `fingerprint = z - (BusId::Commit * α^0 + i * α^1 + v * α^2)`
///   `term = +1 / fingerprint`
///
/// Returns `Some(Σ term)` — the positive receiver contribution that is no
/// longer present as an in-trace table. For empty public output, returns
/// `Some(zero)`. Returns `None` on a fingerprint collision (zero divisor),
/// which the caller should treat as verification failure.
pub(crate) fn compute_commit_bus_offset(
    public_output: &[u8],
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
) -> Option<FieldElement<E>> {
    if public_output.is_empty() {
        return Some(FieldElement::zero());
    }

    let bus_id = FieldElement::<E>::from(BusId::Commit as u64);
    let alpha_sq = alpha * alpha;

    let mut total = FieldElement::<E>::zero();
    for (i, &value) in public_output.iter().enumerate() {
        let linear_combination = bus_id
            + (FieldElement::<E>::from(i as u64) * alpha)
            + (FieldElement::<E>::from(value as u64) * alpha_sq);
        let fingerprint = z - linear_combination;
        total += fingerprint.inv().ok()?;
    }
    Some(total)
}

/// Compute the expected COMMIT bus balance for a `MultiProof`.
///
/// Replays Phase A of the transcript to recover (z, alpha), then computes
/// the offset from the given public output bytes. Call this after `multi_prove`
/// and before `multi_verify`.
pub(crate) fn compute_expected_commit_bus_balance(
    airs: &[&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>],
    proof: &MultiProof<F, E, ()>,
    public_output_bytes: &[u8],
) -> Option<FieldElement<E>> {
    let (z, alpha) = replay_transcript_phase_a(airs, proof);
    compute_commit_bus_offset(public_output_bytes, &z, &alpha)
}

// =============================================================================
// Public API: Prove / Verify
// =============================================================================

/// Prove an ELF binary execution. Returns a serializable proof bundle.
pub fn prove(elf_bytes: &[u8]) -> Result<VmProof, Error> {
    prove_with_inputs(elf_bytes, &[])
}

/// Prove an ELF binary execution with private inputs. Returns a serializable proof bundle.
pub fn prove_with_inputs(elf_bytes: &[u8], private_inputs: &[u8]) -> Result<VmProof, Error> {
    prove_with_options_and_inputs(
        elf_bytes,
        private_inputs,
        &GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid"),
        &MaxRowsConfig::default(),
    )
}

/// Prove an ELF binary execution with custom proof options and max rows config.
pub fn prove_with_options(
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    max_rows: &MaxRowsConfig,
) -> Result<VmProof, Error> {
    prove_with_options_and_inputs(elf_bytes, &[], proof_options, max_rows)
}

/// Prove an ELF binary execution with custom proof options, max rows config,
/// and explicit private inputs.
pub fn prove_with_options_and_inputs(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    max_rows: &MaxRowsConfig,
) -> Result<VmProof, Error> {
    #[cfg(feature = "instruments")]
    let total_start = std::time::Instant::now();

    // Phase 1: Execute (ELF load + run)
    #[cfg(feature = "instruments")]
    let phase_start = std::time::Instant::now();

    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let executor = Executor::new(&program, private_inputs.to_vec())
        .map_err(|e| Error::Execution(format!("{e}")))?;
    let result = executor
        .run()
        .map_err(|e| Error::Execution(format!("{e}")))?;

    #[cfg(feature = "instruments")]
    let execute_elapsed = phase_start.elapsed();

    // Phase 2: Trace build
    #[cfg(feature = "instruments")]
    let phase_start = std::time::Instant::now();

    // Generate all traces from ELF and execution logs.
    // Page tables are derived from the prover's MemoryState (all accessed pages).
    let mut traces = Traces::from_elf_and_logs(&program, &result.logs, max_rows)?;

    #[cfg(feature = "instruments")]
    let trace_build_elapsed = phase_start.elapsed();

    // Phase 3: AIR construction
    #[cfg(feature = "instruments")]
    let phase_start = std::time::Instant::now();

    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &program,
        proof_options,
        false,
        &traces.page_configs,
        &table_counts,
    );

    #[cfg(feature = "instruments")]
    let air_elapsed = phase_start.elapsed();

    let runtime_page_ranges = traces.runtime_page_ranges();

    // Phase 4: Prove (multi_prove)
    let proof = Prover::multi_prove(
        airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))?;

    #[cfg(feature = "instruments")]
    {
        instruments::print_report(
            execute_elapsed,
            trace_build_elapsed,
            air_elapsed,
            total_start.elapsed(),
        );
    }

    Ok(VmProof {
        proof,
        runtime_page_ranges,
        table_counts,
        public_output: traces.public_output_bytes.clone(),
    })
}

/// Verify a proof produced by [`prove`] using default proof options.
///
/// Uses [`GoldilocksCubicProofOptions::with_blowup(2)`] for verification.
/// `runtime_page_ranges` from the proof are hints — preprocessed commitments
/// bind the verifier to the correct page layout.
pub fn verify(vm_proof: &VmProof, elf_bytes: &[u8]) -> Result<bool, Error> {
    verify_with_options(
        vm_proof,
        elf_bytes,
        &GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid"),
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
    // Validate table_counts before constructing AIRs.
    // A malicious prover could set counts to 0, removing entire constraint sets.
    vm_proof.table_counts.validate()?;

    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let page_configs =
        Traces::page_configs_from_elf_and_runtime(&program, &vm_proof.runtime_page_ranges);

    // Cross-check: table_counts must match the number of sub-proofs.
    // Fixed tables (bitwise, decode, halt, commit, register) = 5, plus page tables.
    let expected_proof_count = vm_proof.table_counts.total() + 5 + page_configs.len();
    if expected_proof_count != vm_proof.proof.proofs.len() {
        return Err(Error::InvalidTableCounts(format!(
            "table_counts total ({}) + 5 fixed + {} pages = {}, but proof contains {} sub-proofs",
            vm_proof.table_counts.total(),
            page_configs.len(),
            expected_proof_count,
            vm_proof.proof.proofs.len(),
        )));
    }

    let airs = VmAirs::new(
        &program,
        proof_options,
        false,
        &page_configs,
        &vm_proof.table_counts,
    );

    // Recompute the COMMIT output bus offset from VmProof.public_output.
    // If public_output was tampered, the recomputed offset won't match the
    // actual bus total in the proof, and multi_verify will reject.
    let air_refs = airs.air_refs();
    let expected_bus_balance = match compute_expected_commit_bus_balance(
        &air_refs,
        &vm_proof.proof,
        &vm_proof.public_output,
    ) {
        Some(balance) => balance,
        None => return Ok(false),
    };

    Ok(Verifier::multi_verify(
        &air_refs,
        &vm_proof.proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    ))
}

/// Prove and verify in one call (convenience).
pub fn prove_and_verify(elf_bytes: &[u8]) -> Result<bool, Error> {
    let vm_proof = prove(elf_bytes)?;
    verify(&vm_proof, elf_bytes)
}

// =============================================================================
// Sharded proving (execution segmentation)
// =============================================================================

/// Default segment size in instructions for sharded proving.
pub const DEFAULT_SEGMENT_SIZE: usize = 1 << 22; // ~4M instructions

/// Proof for a single execution segment.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SegmentProof {
    /// The multi-table STARK proof for this segment.
    pub proof: MultiProof<F, E, ()>,
    /// Runtime page ranges for this segment.
    pub runtime_page_ranges: Vec<RuntimePageRange>,
    /// Table chunk counts for this segment.
    pub table_counts: TableCounts,
    /// Public output bytes committed in this segment.
    pub public_output: Vec<u8>,
    /// Page configurations for this segment (init values may differ from ELF for non-first segments).
    /// Phase 1 (trusted executor): verifier uses these directly.
    pub page_configs: Vec<page::PageConfig>,
    /// REGISTER preprocessed commitment for this segment.
    /// None = use ELF default. Some = use this override (non-first segments).
    pub register_commitment: Option<stark::config::Commitment>,
    /// Whether this segment includes the HALT table (only the last segment does).
    pub is_last_segment: bool,
}

/// A sharded VM proof: one independent STARK proof per execution segment.
///
/// Phase 1: trusted executor — each segment is verified independently.
/// Continuity between segments is not yet proved cryptographically.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ShardedVmProof {
    /// One proof per execution segment, in order.
    pub segments: Vec<SegmentProof>,
    /// Combined public output (concatenation of all segments).
    pub public_output: Vec<u8>,
    /// Total number of instructions across all segments.
    pub total_instructions: usize,
}

/// Prove an ELF binary execution using sharded proving.
///
/// Splits execution into segments of `segment_size` instructions. Each segment
/// generates its own tables and proof. Trace building is sequential (each segment
/// needs the previous segment's final state), but proving is parallel.
pub fn prove_sharded(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    segment_size: usize,
) -> Result<ShardedVmProof, Error> {
    prove_sharded_with_options(
        elf_bytes,
        private_inputs,
        segment_size,
        &GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid"),
        &MaxRowsConfig::default(),
    )
}

/// Prove an ELF binary using sharded proving with custom options.
pub fn prove_sharded_with_options(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    segment_size: usize,
    proof_options: &ProofOptions,
    max_rows: &MaxRowsConfig,
) -> Result<ShardedVmProof, Error> {
    use crate::tables::trace_builder::{MemoryState, RegisterState};

    // Phase 1: Execute fully
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let executor = Executor::new(&program, private_inputs.to_vec())
        .map_err(|e| Error::Execution(format!("{e}")))?;
    let result = executor
        .run()
        .map_err(|e| Error::Execution(format!("{e}")))?;
    let all_logs = &result.logs;
    let total_instructions = all_logs.len();

    // Phase 2: Split logs into segments, build traces sequentially
    let num_segments = if total_instructions == 0 {
        1
    } else {
        (total_instructions + segment_size - 1) / segment_size
    };

    let mut memory_state = MemoryState::from_elf(&program);
    let mut register_state = RegisterState::new(program.entry_point);
    let mut segment_traces: Vec<Traces> = Vec::with_capacity(num_segments);
    let mut segment_public_outputs: Vec<Vec<u8>> = Vec::with_capacity(num_segments);
    // Capture initial register state per segment for preprocessed commitment
    let mut segment_initial_reg_states: Vec<Option<register::FinalRegisterStateMap>> =
        Vec::with_capacity(num_segments);

    for seg_idx in 0..num_segments {
        let start = seg_idx * segment_size;
        let end = (start + segment_size).min(total_instructions);
        let segment_logs = &all_logs[start..end];
        let is_last = seg_idx == num_segments - 1;
        let global_log_base = start as u64;

        // For non-first segments, capture register state for preprocessed commitment
        let initial_reg_state = if seg_idx > 0 {
            Some(register_state.to_final_state_map())
        } else {
            None // Segment 0 uses ELF defaults
        };

        let (traces, new_mem, new_reg) = Traces::from_segment(
            &program,
            segment_logs,
            max_rows,
            memory_state,
            register_state,
            global_log_base,
            is_last,
        )?;

        segment_public_outputs.push(traces.public_output_bytes.clone());
        segment_initial_reg_states.push(initial_reg_state);
        segment_traces.push(traces);
        memory_state = new_mem;
        register_state = new_reg;
    }

    // Phase 3: Prove each segment (sequentially for now, parallel later)
    let mut segment_proofs: Vec<SegmentProof> = Vec::with_capacity(num_segments);

    for (seg_idx, mut traces) in segment_traces.into_iter().enumerate() {
        let table_counts = traces.table_counts();

        // For non-first segments, compute REGISTER preprocessed commitment from
        // the segment's initial state (not ELF defaults).
        let reg_commitment = segment_initial_reg_states[seg_idx].as_ref().map(|state| {
            register::preprocessed_commitment_from_state(proof_options, state, program.entry_point)
        });

        let is_last = seg_idx == num_segments - 1;
        let airs = VmAirs::new_with_overrides(
            &program,
            proof_options,
            false,
            &traces.page_configs,
            &table_counts,
            reg_commitment,
            is_last, // include_halt: only for last segment
            true,    // include_commit: always (handles empty ops with zero mults)
        );

        let runtime_page_ranges = traces.runtime_page_ranges();

        let proof = Prover::multi_prove(
            airs.air_trace_pairs(&mut traces),
            &mut DefaultTranscript::<E>::new(&[]),
        )
        .map_err(|e| Error::Prover(format!("Segment {seg_idx}: {e:?}")))?;

        segment_proofs.push(SegmentProof {
            proof,
            runtime_page_ranges,
            table_counts,
            public_output: segment_public_outputs[seg_idx].clone(),
            page_configs: traces.page_configs,
            register_commitment: segment_initial_reg_states[seg_idx].as_ref().map(|state| {
                register::preprocessed_commitment_from_state(
                    proof_options,
                    state,
                    program.entry_point,
                )
            }),
            is_last_segment: is_last,
        });
    }

    let combined_output: Vec<u8> = segment_public_outputs.into_iter().flatten().collect();

    Ok(ShardedVmProof {
        segments: segment_proofs,
        public_output: combined_output,
        total_instructions,
    })
}

/// Verify a sharded VM proof.
///
/// Phase 1: verifies each segment independently. Does NOT verify
/// cross-segment continuity (trusted executor).
pub fn verify_sharded(proof: &ShardedVmProof, elf_bytes: &[u8]) -> Result<bool, Error> {
    verify_sharded_with_options(
        proof,
        elf_bytes,
        &GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid"),
    )
}

/// Verify a sharded VM proof with custom options.
pub fn verify_sharded_with_options(
    proof: &ShardedVmProof,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
) -> Result<bool, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;

    for (seg_idx, seg) in proof.segments.iter().enumerate() {
        seg.table_counts.validate().map_err(|e| {
            Error::Prover(format!("Segment {seg_idx} invalid table_counts: {e}"))
        })?;

        // Phase 1 (trusted executor): use page_configs, register_commitment,
        // and is_last_segment from the proof.
        let airs = VmAirs::new_with_overrides(
            &program,
            proof_options,
            false,
            &seg.page_configs,
            &seg.table_counts,
            seg.register_commitment,
            seg.is_last_segment, // include_halt
            true,                // include_commit
        );

        let air_refs = airs.air_refs();

        // Compute expected bus balance from this segment's public output
        let expected_bus_balance = match compute_expected_commit_bus_balance(
            &air_refs,
            &seg.proof,
            &seg.public_output,
        ) {
            Some(balance) => balance,
            None => return Ok(false),
        };

        let ok = Verifier::multi_verify(
            &air_refs,
            &seg.proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &expected_bus_balance,
        );

        if !ok {
            return Ok(false);
        }
    }

    // Verify combined public output matches
    let combined: Vec<u8> = proof
        .segments
        .iter()
        .flat_map(|s| s.public_output.iter().copied())
        .collect();
    if combined != proof.public_output {
        return Ok(false);
    }

    Ok(true)
}

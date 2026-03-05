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

pub use crate::tables::MaxRowsConfig;
use crate::tables::bitwise;
use crate::tables::decode;
use crate::tables::page;
use crate::tables::register;
use crate::tables::trace_builder::Traces;
use crate::test_utils::{
    E, F, VmAir, create_bitwise_air, create_branch_air, create_cpu_air, create_decode_air,
    create_dvrm_air, create_halt_air, create_load_air, create_lt_air, create_memw_air,
    create_memory_bridge_air, create_mul_air, create_page_air, create_register_air,
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
    pub load: usize,
    pub mul: usize,
    pub dvrm: usize,
    pub branch: usize,
}

impl TableCounts {
    /// Validate that all required tables have at least one chunk.
    ///
    /// A zero count for any table would remove its constraints from verification,
    /// allowing a malicious prover to bypass soundness checks.
    /// Sum of all chunk counts across split tables.
    pub fn total(&self) -> usize {
        self.cpu + self.lt + self.memw + self.load + self.mul + self.dvrm + self.branch
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
            ("load", self.load),
            ("mul", self.mul),
            ("dvrm", self.dvrm),
            ("branch", self.branch),
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
    pub memws: Vec<VmAir>,
    pub loads: Vec<VmAir>,
    pub decode: VmAir,
    pub muls: Vec<VmAir>,
    pub dvrms: Vec<VmAir>,
    pub branches: Vec<VmAir>,
    /// HALT AIR for program termination.
    /// None for non-final segments in continuation proofs (no ecall in intermediate segments).
    pub halt: Option<VmAir>,
    pub register: VmAir,
    pub pages: Vec<VmAir>,
    /// MemoryBridge AIR for continuation proof segments > 0.
    pub memory_bridge: Option<VmAir>,
}

impl VmAirs {
    /// Build `(air, trace, public_inputs)` triples for [`Prover::multi_prove`].
    pub fn air_trace_pairs<'a>(&'a self, traces: &'a mut Traces) -> Vec<AirTracePair<'a>> {
        let mut pairs: Vec<AirTracePair<'a>> = vec![
            (&self.bitwise, &mut traces.bitwise, &()),
            (&self.decode, &mut traces.decode, &()),
            (&self.register, &mut traces.register, &()),
        ];
        if let Some(ref halt_air) = self.halt {
            if let Some(ref mut halt_trace) = traces.halt {
                pairs.push((halt_air, halt_trace, &()));
            }
        }

        for (air, trace) in self.cpus.iter().zip(traces.cpus.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.lts.iter().zip(traces.lts.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.memws.iter().zip(traces.memws.iter_mut()) {
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
        if let Some(ref bridge_air) = self.memory_bridge {
            if let Some(ref mut bridge_trace) = traces.memory_bridge {
                pairs.push((bridge_air, bridge_trace, &()));
            }
        }

        pairs
    }

    /// Collect AIR references for [`Verifier::multi_verify`].
    pub fn air_refs(&self) -> Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> {
        let mut refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
            vec![&self.bitwise, &self.decode, &self.register];
        if let Some(ref halt_air) = self.halt {
            refs.push(halt_air);
        }

        for air in &self.cpus {
            refs.push(air);
        }
        for air in &self.lts {
            refs.push(air);
        }
        for air in &self.memws {
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
        if let Some(ref bridge_air) = self.memory_bridge {
            refs.push(bridge_air);
        }

        refs
    }

    /// Create all VM AIR instances. `minimal_bitwise` controls whether the full
    /// 2^20 bitwise preprocessed table is included (false = full, true = minimal).
    /// DECODE is always preprocessed.
    ///
    /// `page_configs` provides the page base addresses for creating PAGE AIRs.
    /// `table_counts` specifies how many chunks for each split table.
    pub fn new(
        elf: &Elf,
        proof_options: &ProofOptions,
        minimal_bitwise: bool,
        page_configs: &[crate::tables::page::PageConfig],
        table_counts: &TableCounts,
        has_halt: bool,
        has_memory_bridge: bool,
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
        let memws: Vec<_> = (0..table_counts.memw)
            .map(|i| create_memw_air(proof_options).with_name(&format!("MEMW[{}]", i)))
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
        let halt = if has_halt {
            Some(create_halt_air(proof_options))
        } else {
            None
        };
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

        let memory_bridge = if has_memory_bridge {
            Some(create_memory_bridge_air(proof_options))
        } else {
            None
        };

        #[cfg(feature = "debug-checks")]
        debug_report::print_bus_legend();

        Self {
            cpus,
            bitwise,
            lts,
            memws,
            loads,
            decode,
            muls,
            dvrms,
            branches,
            halt,
            register,
            pages,
            memory_bridge,
        }
    }
}

/// Prove an ELF binary execution. Returns a serializable proof bundle.
pub fn prove(elf_bytes: &[u8]) -> Result<VmProof, Error> {
    prove_with_options(
        elf_bytes,
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
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let executor = Executor::new(&program, vec![]).map_err(|e| Error::Execution(format!("{e}")))?;
    let result = executor
        .run()
        .map_err(|e| Error::Execution(format!("{e}")))?;

    // Generate all traces from ELF and execution logs.
    // Page tables are derived from the prover's MemoryState (all accessed pages).
    let mut traces = Traces::from_elf_and_logs(&program, &result.logs, max_rows)?;
    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &program,
        proof_options,
        false,
        &traces.page_configs,
        &table_counts,
        true,  // always has halt for single-segment proofs
        false, // no memory bridge for single-segment proofs
    );

    let runtime_page_ranges = traces.runtime_page_ranges();

    let proof = Prover::multi_prove(
        airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))?;

    Ok(VmProof {
        proof,
        runtime_page_ranges,
        table_counts,
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
    // Fixed tables: bitwise + decode + register = 3, plus halt + page tables.
    let expected_proof_count = vm_proof.table_counts.total() + 4 + page_configs.len();
    if expected_proof_count != vm_proof.proof.proofs.len() {
        return Err(Error::InvalidTableCounts(format!(
            "table_counts total ({}) + 4 fixed + {} pages = {}, but proof contains {} sub-proofs",
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
        true,  // always has halt for single-segment proofs
        false, // no memory bridge for single-segment proofs
    );

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

// =============================================================================
// Continuation Proofs (Segment-based)
// =============================================================================

use crate::tables::trace_builder::SegmentCheckpoint;
use math::field::element::FieldElement;

/// A single segment's proof within a continuation.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SegmentProof {
    /// The multi-table STARK proof for this segment.
    pub proof: MultiProof<F, E, ()>,
    /// Run-length encoded runtime page ranges for this segment.
    pub runtime_page_ranges: Vec<RuntimePageRange>,
    /// Table chunk counts for this segment.
    pub table_counts: TableCounts,
    /// Bus residual for this segment (sum of table_contribution values).
    /// For a standalone proof this is zero; for continuation segments it may be non-zero.
    pub bus_residual: FieldElement<E>,
    /// Index of this segment in the continuation.
    pub segment_index: usize,
    /// Whether this segment includes a HALT table.
    /// True for the final segment, false for intermediate segments.
    pub has_halt: bool,
    /// Whether this segment includes a MemoryBridge table.
    /// True for segments > 0 in a continuation proof.
    pub has_memory_bridge: bool,
}

/// A continuation proof: multiple segment proofs that together prove full program execution.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ContinuationProof {
    /// Per-segment proofs, ordered by segment index.
    pub segment_proofs: Vec<SegmentProof>,
}

/// Default segment size (number of instruction logs per segment).
/// `usize::MAX` effectively disables segmentation (single segment = full execution).
pub const DEFAULT_SEGMENT_SIZE: usize = usize::MAX;

/// Prove an ELF binary execution using continuation (segmented) proving.
///
/// Splits execution into segments of `segment_size` logs each.
/// Each segment generates its own traces and multi-table STARK proof.
/// Segments are proved sequentially (trace generation requires state from
/// the previous segment), but the proving step for each segment is independent.
///
/// When `segment_size == usize::MAX`, this produces a single-segment proof
/// equivalent to [`prove`].
pub fn prove_continuation(
    elf_bytes: &[u8],
    segment_size: usize,
) -> Result<ContinuationProof, Error> {
    prove_continuation_with_options(
        elf_bytes,
        segment_size,
        &GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid"),
        &MaxRowsConfig::default(),
    )
}

/// Prove with continuation using custom options.
pub fn prove_continuation_with_options(
    elf_bytes: &[u8],
    segment_size: usize,
    proof_options: &ProofOptions,
    max_rows: &MaxRowsConfig,
) -> Result<ContinuationProof, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let executor = Executor::new(&program, vec![]).map_err(|e| Error::Execution(format!("{e}")))?;
    let result = executor
        .run()
        .map_err(|e| Error::Execution(format!("{e}")))?;

    let logs = &result.logs;

    // Split logs into segments.
    let segments: Vec<&[executor::vm::logs::Log]> = if segment_size >= logs.len() {
        vec![logs]
    } else {
        logs.chunks(segment_size).collect()
    };

    let mut segment_proofs = Vec::with_capacity(segments.len());
    let mut checkpoint = SegmentCheckpoint::from_elf(&program);

    for (seg_idx, segment_logs) in segments.iter().enumerate() {
        let log_offset = seg_idx * segment_size;

        // Generate traces for this segment.
        let (mut traces, next_checkpoint) = Traces::from_segment(
            &program,
            segment_logs,
            &checkpoint,
            log_offset as u64,
            max_rows,
        )?;

        let table_counts = traces.table_counts();
        let has_halt = traces.halt.is_some();
        let has_memory_bridge = traces.memory_bridge.is_some();
        let airs = VmAirs::new(
            &program,
            proof_options,
            false,
            &traces.page_configs,
            &table_counts,
            has_halt,
            has_memory_bridge,
        );

        let runtime_page_ranges = traces.runtime_page_ranges();

        let proof = Prover::multi_prove(
            airs.air_trace_pairs(&mut traces),
            &mut DefaultTranscript::<E>::new(&[]),
        )
        .map_err(|e| Error::Prover(format!("{e:?}")))?;

        let bus_residual = proof.bus_residual();

        segment_proofs.push(SegmentProof {
            proof,
            runtime_page_ranges,
            table_counts,
            bus_residual,
            segment_index: seg_idx,
            has_halt,
            has_memory_bridge,
        });

        checkpoint = next_checkpoint;
    }

    Ok(ContinuationProof { segment_proofs })
}

/// Verify a continuation proof.
///
/// Checks:
/// 1. Each segment's multi-table STARK proof verifies independently
///    (constraint satisfaction, commitment integrity, FRI soundness).
///    Bus balance is NOT checked per-segment.
/// 2. Global bus balance: Σ segment_residual = 0 across all segments.
pub fn verify_continuation(
    continuation_proof: &ContinuationProof,
    elf_bytes: &[u8],
) -> Result<bool, Error> {
    verify_continuation_with_options(
        continuation_proof,
        elf_bytes,
        &GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid"),
    )
}

/// Verify a continuation proof with custom options.
pub fn verify_continuation_with_options(
    continuation_proof: &ContinuationProof,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
) -> Result<bool, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;

    // 1. Verify each segment's STARK proof (without bus balance check).
    for segment in &continuation_proof.segment_proofs {
        segment.table_counts.validate()?;

        let page_configs =
            Traces::page_configs_from_elf_and_runtime(&program, &segment.runtime_page_ranges);

        let halt_count = if segment.has_halt { 1 } else { 0 };
        let bridge_count = if segment.has_memory_bridge { 1 } else { 0 };
        // Fixed tables: bitwise + decode + register = 3
        let expected_proof_count =
            segment.table_counts.total() + 3 + halt_count + page_configs.len() + bridge_count;
        if expected_proof_count != segment.proof.proofs.len() {
            return Err(Error::InvalidTableCounts(format!(
                "segment {}: table_counts total ({}) + 3 fixed + {} halt + {} pages + {} bridge = {}, but proof has {} sub-proofs",
                segment.segment_index,
                segment.table_counts.total(),
                halt_count,
                page_configs.len(),
                bridge_count,
                expected_proof_count,
                segment.proof.proofs.len(),
            )));
        }

        let airs = VmAirs::new(
            &program,
            proof_options,
            false,
            &page_configs,
            &segment.table_counts,
            segment.has_halt,
            segment.has_memory_bridge,
        );

        // Use multi_verify_segment (skips bus balance check).
        let valid = Verifier::multi_verify_segment(
            &airs.air_refs(),
            &segment.proof,
            &mut DefaultTranscript::<E>::new(&[]),
        );

        if !valid {
            return Ok(false);
        }

        // Cross-check: bus_residual in proof metadata must match computed residual.
        let computed_residual = segment.proof.bus_residual();
        if computed_residual != segment.bus_residual {
            return Ok(false);
        }
    }

    // 2. Global bus balance: Σ bus_residual = 0.
    let total_residual: FieldElement<E> = continuation_proof
        .segment_proofs
        .iter()
        .fold(FieldElement::<E>::zero(), |acc, seg| {
            acc + &seg.bus_residual
        });

    if total_residual != FieldElement::<E>::zero() {
        return Ok(false);
    }

    Ok(true)
}

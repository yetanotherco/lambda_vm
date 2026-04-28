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

pub mod auto_storage;
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
use stark::storage_mode::StorageMode;
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
    /// Number of PAGE tables that hold private input data.
    /// These pages are NOT preprocessed — the verifier reconstructs them
    /// as non-preprocessed tables starting at `PRIVATE_INPUT_START_INDEX`.
    pub num_private_input_pages: usize,
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
    pub halt: VmAir,
    pub commit: VmAir,
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
            (&self.halt, &mut traces.halt, &()),
            (&self.commit, &mut traces.commit, &()),
            (&self.register, &mut traces.register, &()),
        ];

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
            &self.halt,
            &self.commit,
            &self.register,
        ];

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
    /// `page_configs` provides the page base addresses for creating PAGE AIRs.
    /// `table_counts` specifies how many chunks for each split table.
    pub fn new(
        elf: &Elf,
        proof_options: &ProofOptions,
        minimal_bitwise: bool,
        page_configs: &[crate::tables::page::PageConfig],
        table_counts: &TableCounts,
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
        let halt = create_halt_air(proof_options);
        let commit = create_commit_air(proof_options);
        let register = create_register_air(proof_options).with_preprocessed(
            register::preprocessed_commitment(proof_options, elf.entry_point),
            register::NUM_PREPROCESSED_COLS,
        );
        let pages: Vec<_> = page_configs
            .iter()
            .map(|config| {
                if config.is_private_input {
                    // Private-input pages: all columns are main trace (not preprocessed).
                    // The verifier doesn't see the init values; correctness is enforced
                    // by the memory bus constraints.
                    create_page_air(proof_options, config.page_base)
                } else {
                    // ELF and zero-init pages: OFFSET + INIT are preprocessed.
                    // The verifier independently recomputes the commitment from public data.
                    create_page_air(proof_options, config.page_base).with_preprocessed(
                        page::precomputed_commitment_cached(config, proof_options),
                        page::NUM_PREPROCESSED_COLS,
                    )
                }
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

/// Count the total number of main-trace and auxiliary-trace field elements without
/// running the STARK proof step.
///
/// Returns `(main_elements, aux_elements)` where `main_elements` is the sum of
/// `rows × columns` over all main (base-field) trace columns, and `aux_elements`
/// is the sum of `rows × ⌈bus_interactions/2⌉` over all tables — i.e. the number
/// of committed extension-field columns times rows (LogUp batching packs two
/// interactions per column).
pub fn count_elements(elf_bytes: &[u8], private_inputs: &[u8]) -> Result<(u64, u64), Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let executor = Executor::new(&program, private_inputs.to_vec())
        .map_err(|e| Error::Execution(format!("{e}")))?;
    let result = executor
        .run()
        .map_err(|e| Error::Execution(format!("{e}")))?;
    let lengths = crate::tables::trace_builder::count_table_lengths(
        &program,
        &result.logs,
        &MaxRowsConfig::default(),
        private_inputs,
    )?;
    Ok((lengths.total_main_elements(), lengths.total_aux_elements()))
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
    #[cfg(feature = "instruments")]
    let heap_before = stark::instruments::heap_bytes();

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
    #[cfg(feature = "instruments")]
    let heap_after_execute = stark::instruments::heap_bytes();

    // Phase 2: Trace build
    #[cfg(feature = "instruments")]
    let phase_start = std::time::Instant::now();

    // Stream over logs once to compute exact per-table row counts without
    // allocating any op vectors. Use the resulting `TableLengths` to estimate
    // peak heap analytically and pick a storage mode.
    let lengths = crate::tables::trace_builder::count_table_lengths(
        &program,
        &result.logs,
        max_rows,
        private_inputs,
    )?;

    let available = auto_storage::available_ram_bytes();
    let estimated_peak = auto_storage::peak_bytes(
        &lengths,
        proof_options.blowup_factor,
        stark::prover::table_parallelism(),
    );
    let storage_mode =
        auto_storage::select_storage_mode(estimated_peak, available, proof_options.max_ram_bytes);

    eprintln!("predicted_peak_bytes: {estimated_peak}");

    if available.is_none() && proof_options.max_ram_bytes.is_none() {
        log::warn!(
            "Auto disk-spill: OS did not report available memory — staying in Ram mode. \
             Set ProofOptions::max_ram_bytes to force Disk in memory-constrained environments."
        );
    }
    if storage_mode == StorageMode::Disk {
        let budget =
            auto_storage::effective_budget(available, proof_options.max_ram_bytes).unwrap_or(0);
        log::info!(
            "Auto disk-spill: estimated peak {} MB exceeds 80% of {} MB budget",
            estimated_peak / 1_000_000,
            budget / 1_000_000,
        );
    }

    // Phase 5: build the full traces with the chosen mode. `Disk` spills each
    // chunk as it's built, so the trace never fully materializes in RAM.
    let mut traces = Traces::from_elf_and_logs(
        &program,
        &result.logs,
        max_rows,
        private_inputs,
        storage_mode,
    )?;
    drop(result);

    #[cfg(feature = "instruments")]
    let trace_build_elapsed = phase_start.elapsed();
    #[cfg(feature = "instruments")]
    let heap_after_trace = stark::instruments::heap_bytes();

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
    #[cfg(feature = "instruments")]
    let heap_after_air = stark::instruments::heap_bytes();

    let runtime_page_ranges = traces.runtime_page_ranges();

    // Phase 4: Prove (multi_prove)
    let proof = Prover::multi_prove_with_mode(
        airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
        storage_mode,
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))?;

    #[cfg(feature = "instruments")]
    {
        instruments::print_report(
            execute_elapsed,
            trace_build_elapsed,
            air_elapsed,
            total_start.elapsed(),
            &stark::instruments::ProveHeapProfile {
                before: heap_before,
                after_execute: heap_after_execute,
                after_trace_build: heap_after_trace,
                after_air: heap_after_air,
            },
        );
    }

    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();

    Ok(VmProof {
        proof,
        runtime_page_ranges,
        table_counts,
        public_output: traces.public_output_bytes.clone(),
        num_private_input_pages,
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

    // Bound num_private_input_pages before allocating PageConfigs.
    // MAX_PRIVATE_INPUT_SIZE fits in ~26 pages of DEFAULT_PAGE_SIZE.
    {
        use crate::tables::page::DEFAULT_PAGE_SIZE;
        use executor::vm::memory::MAX_PRIVATE_INPUT_SIZE;
        let max_pages = (MAX_PRIVATE_INPUT_SIZE as usize + 4).div_ceil(DEFAULT_PAGE_SIZE) + 1;
        if vm_proof.num_private_input_pages > max_pages {
            return Err(Error::InvalidTableCounts(format!(
                "num_private_input_pages ({}) exceeds max ({max_pages})",
                vm_proof.num_private_input_pages,
            )));
        }
    }

    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let page_configs = Traces::page_configs_from_elf_and_runtime(
        &program,
        &vm_proof.runtime_page_ranges,
        vm_proof.num_private_input_pages,
    );

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

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

#[cfg(feature = "disk-spill")]
pub mod auto_storage;
pub mod constraints;
pub mod continuation;
#[cfg(feature = "debug-checks")]
mod debug_report;
#[cfg(feature = "instruments")]
pub mod instruments;
mod paged_mem;
mod statement;
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
use stark::config::Commitment;
use stark::prover::{IsStarkProver, Prover};
#[cfg(feature = "disk-spill")]
use stark::storage_mode::StorageMode;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::statement::{StatementKind, absorb_statement};
pub use crate::tables::MaxRowsConfig;
use crate::tables::bitwise;
use crate::tables::decode;
use crate::tables::page;
use crate::tables::register;
use crate::tables::trace_builder::Traces;
#[cfg(feature = "disk-spill")]
use crate::tables::trace_builder::count_table_lengths;
use crate::tables::types::BusId;
use crate::test_utils::{
    E, F, VmAir, create_bitwise_air, create_branch_air, create_bytewise_air, create_commit_air,
    create_cpu_air, create_cpu32_air, create_decode_air, create_dvrm_air, create_ec_scalar_air,
    create_ecdas_air, create_ecsm_air, create_eq_air, create_halt_air, create_keccak_air,
    create_keccak_rc_air, create_keccak_rnd_air, create_load_air, create_lt_air, create_memw_air,
    create_memw_aligned_air, create_memw_register_air, create_mul_air, create_page_air,
    create_register_air, create_shift_air, create_store_air,
};

// Re-exported so downstream verifier guests (e.g. the in-VM recursion guest) can
// name the proof-options type carried in their private input alongside `VmProof`.
pub use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::stark::{BatchedMultiProof, MultiProof};

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

/// Number of tables that always contribute exactly one sub-proof, regardless
/// of `TableCounts`: bitwise, decode, halt, commit, keccak, keccak_rnd,
/// keccak_rc, register, ecsm, ec_scalar, ecdas.
pub const FIXED_TABLE_COUNT: usize = 11;

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
    // Auxiliary ALU / memory / CPU32 dispatch chips
    pub eq: usize,
    pub bytewise: usize,
    pub store: usize,
    pub cpu32: usize,
}

impl TableCounts {
    /// Sum of all chunk counts across the split tables.
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
            + self.eq
            + self.bytewise
            + self.store
            + self.cpu32
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
            ("eq", self.eq),
            ("bytewise", self.bytewise),
            ("store", self.store),
            ("cpu32", self.cpu32),
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
    /// The multi-table STARK proof (unified-shard / batched MMCS).
    pub proof: BatchedMultiProof<F, E, ()>,
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
    /// Continuation epoch size exponent is invalid.
    InvalidContinuationEpochSize(String),
    /// Continuation proof construction hit an internal invariant failure.
    ContinuationInvariant(String),
    /// A non-final continuation epoch contains the program-terminating
    /// instruction. The terminating instruction must be in the final epoch.
    HaltInNonFinalEpoch,
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
            Error::InvalidContinuationEpochSize(msg) => {
                write!(f, "invalid continuation epoch size: {msg}")
            }
            Error::ContinuationInvariant(msg) => {
                write!(f, "continuation invariant failed: {msg}")
            }
            Error::HaltInNonFinalEpoch => {
                write!(
                    f,
                    "the program-terminating instruction must be in the final epoch"
                )
            }
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
    pub keccak: VmAir,
    pub keccak_rnd: VmAir,
    pub keccak_rc: VmAir,
    pub ecsm: VmAir,
    pub ec_scalar: VmAir,
    pub ecdas: VmAir,
    pub register: VmAir,
    pub pages: Vec<VmAir>,
    pub memw_registers: Vec<VmAir>,
    /// Whether the HALT table participates in this proof. False for intermediate
    /// continuation epochs, which do not terminate the program.
    pub include_halt: bool,
    // Auxiliary ALU / memory / CPU32 dispatch chips
    pub eqs: Vec<VmAir>,
    pub bytewises: Vec<VmAir>,
    pub stores: Vec<VmAir>,
    pub cpu32s: Vec<VmAir>,
}

impl VmAirs {
    /// Build `(air, trace, public_inputs)` triples for [`Prover::multi_prove`].
    pub fn air_trace_pairs<'a>(&'a self, traces: &'a mut Traces) -> Vec<AirTracePair<'a>> {
        let mut pairs: Vec<AirTracePair<'a>> = vec![
            (&self.bitwise, &mut traces.bitwise, &()),
            (&self.decode, &mut traces.decode, &()),
            (&self.commit, &mut traces.commit, &()),
            (&self.keccak, &mut traces.keccak, &()),
            (&self.keccak_rnd, &mut traces.keccak_rnd, &()),
            (&self.keccak_rc, &mut traces.keccak_rc, &()),
            (&self.ecsm, &mut traces.ecsm, &()),
            (&self.ec_scalar, &mut traces.ec_scalar, &()),
            (&self.ecdas, &mut traces.ecdas, &()),
            (&self.register, &mut traces.register, &()),
        ];
        if self.include_halt {
            pairs.push((&self.halt, &mut traces.halt, &()));
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
        for (air, trace) in self.eqs.iter().zip(traces.eqs.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.bytewises.iter().zip(traces.bytewises.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.stores.iter().zip(traces.stores.iter_mut()) {
            pairs.push((air, trace, &()));
        }
        for (air, trace) in self.cpu32s.iter().zip(traces.cpu32s.iter_mut()) {
            pairs.push((air, trace, &()));
        }

        pairs
    }

    /// Collect AIR references for [`Verifier::multi_verify`].
    pub fn air_refs(&self) -> Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> {
        let mut refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![
            &self.bitwise,
            &self.decode,
            &self.commit,
            &self.keccak,
            &self.keccak_rnd,
            &self.keccak_rc,
            &self.ecsm,
            &self.ec_scalar,
            &self.ecdas,
            &self.register,
        ];
        if self.include_halt {
            refs.push(&self.halt);
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
        for air in &self.eqs {
            refs.push(air);
        }
        for air in &self.bytewises {
            refs.push(air);
        }
        for air in &self.stores {
            refs.push(air);
        }
        for air in &self.cpu32s {
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
    ///
    /// `decode_commitment` is an optional precomputed DECODE preprocessed
    /// commitment. When `Some`, the supplied value is used directly and the
    /// FFT + Merkle build is skipped — useful for callers who have already
    /// computed the commitment offline and embedded it as a compile-time
    /// constant (e.g. the recursion guest, where the in-VM recompute is too
    /// expensive). When `None`, the commitment is computed from the ELF.
    ///
    /// `page_commitments` is an optional list of precomputed ELF-data-page
    /// preprocessed commitments, keyed by `page_base`. For each ELF data page
    /// the verifier constructs, if a matching `(page_base, commitment)` pair
    /// is supplied, it is used directly and that page's FFT + Merkle build is
    /// skipped. Pages not in the list — including all zero-init pages and
    /// pages without a match — take the normal compute path (zero-init pages
    /// hit a compile-time constant via
    /// `page::zero_init_preprocessed_commitment`; ELF data pages recompute
    /// from the ELF). When `None`, every ELF data page recomputes from
    /// scratch.
    ///
    /// The trust anchor for both `decode_commitment` and `page_commitments`
    /// is the caller's compiled binary — never accept prover-supplied bytes
    /// here. A wrong value is rejected, never silently accepted: it either
    /// mismatches the prover's committed precomputed root (an explicit
    /// verifier check) or yields diverging Fiat-Shamir challenges.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        elf: &Elf,
        proof_options: &ProofOptions,
        minimal_bitwise: bool,
        page_configs: &[crate::tables::page::PageConfig],
        table_counts: &TableCounts,
        decode_commitment: Option<Commitment>,
        include_halt: bool,
        register_init: Option<&[u32]>,
        page_commitments: Option<&[(u64, Commitment)]>,
        register_preprocessed: Option<(Commitment, usize)>,
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
        let decode_root = decode_commitment.unwrap_or_else(|| {
            decode::commitment_from_elf(elf, proof_options)
                .expect("Failed to compute decode commitment")
        });
        let decode = create_decode_air(proof_options)
            .with_preprocessed(decode_root, decode::NUM_PRECOMPUTED_COLS);
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
        let keccak = create_keccak_air(proof_options);
        let keccak_rnd = create_keccak_rnd_air(proof_options);
        let keccak_rc = create_keccak_rc_air(proof_options).with_preprocessed(
            tables::keccak_rc::preprocessed_commitment(proof_options),
            tables::keccak_rc::NUM_PRECOMPUTED_COLS,
        );
        let ecsm = create_ecsm_air(proof_options);
        let ec_scalar = create_ec_scalar_air(proof_options);
        let ecdas = create_ecdas_air(proof_options);
        let register = if let Some((commitment, num_preprocessed_cols)) = register_preprocessed {
            create_register_air(proof_options).with_preprocessed(commitment, num_preprocessed_cols)
        } else {
            let register_init = register_init
                .map(<[u32]>::to_vec)
                .unwrap_or_else(|| register::register_init_from_entry_point(elf.entry_point));
            create_register_air(proof_options).with_preprocessed(
                register::preprocessed_commitment(proof_options, &register_init),
                register::NUM_PREPROCESSED_COLS,
            )
        };
        // Every zero-init page shares one preprocessed commitment: OFFSET is
        // page-relative and INIT is all-zero, so it depends only on
        // (blowup, coset) — all fixed here. Compute it once (static const
        // when shipped, else a single recompute) rather than per page. Every
        // program has at least one zero-init page (the stack is zero-
        // initialized), so this commitment is always used.
        let zero_init_commitment = page::zero_init_preprocessed_commitment(proof_options);

        let pages: Vec<_> = page_configs
            .iter()
            .map(|config| {
                let air = create_page_air(proof_options, config.page_base);
                if config.is_private_input {
                    // Private-input pages: all columns are main trace (not preprocessed).
                    // The verifier doesn't see the init values; correctness is enforced
                    // by the memory bus constraints.
                    air
                } else if config.init_values.is_none() {
                    // Zero-init pages: the shared commitment computed once above.
                    air.with_preprocessed(zero_init_commitment, page::NUM_PREPROCESSED_COLS)
                } else {
                    // ELF data pages: INIT is program-specific, so the commitment is
                    // per-page. Prefer a caller-supplied `(page_base, commitment)`
                    // (recursion guest); otherwise recompute from the ELF.
                    let commitment = page_commitments
                        .unwrap_or(&[])
                        .iter()
                        .find(|(pb, _)| *pb == config.page_base)
                        .map(|(_, c)| *c)
                        .unwrap_or_else(|| {
                            page::compute_precomputed_commitment(config, proof_options)
                        });
                    air.with_preprocessed(commitment, page::NUM_PREPROCESSED_COLS)
                }
            })
            .collect();
        let memw_registers: Vec<_> = (0..table_counts.memw_register)
            .map(|i| create_memw_register_air(proof_options).with_name(&format!("MEMW_R[{}]", i)))
            .collect();
        let eqs: Vec<_> = (0..table_counts.eq)
            .map(|i| create_eq_air(proof_options).with_name(&format!("EQ[{}]", i)))
            .collect();
        let bytewises: Vec<_> = (0..table_counts.bytewise)
            .map(|i| create_bytewise_air(proof_options).with_name(&format!("BYTEWISE[{}]", i)))
            .collect();
        let stores: Vec<_> = (0..table_counts.store)
            .map(|i| create_store_air(proof_options).with_name(&format!("STORE[{}]", i)))
            .collect();
        let cpu32s: Vec<_> = (0..table_counts.cpu32)
            .map(|i| create_cpu32_air(proof_options).with_name(&format!("CPU32[{}]", i)))
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
            keccak,
            keccak_rnd,
            keccak_rc,
            ecsm,
            ec_scalar,
            ecdas,
            register,
            pages,
            memw_registers,
            include_halt,
            eqs,
            bytewises,
            stores,
            cpu32s,
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
    transcript: &mut DefaultTranscript<E>,
) -> (FieldElement<E>, FieldElement<E>) {
    for (air, proof) in airs.iter().zip(&multi_proof.proofs) {
        if air.is_preprocessed() {
            transcript.append_bytes(&air.precomputed_commitment());
        }
        transcript.append_bytes(&proof.lde_trace_main_merkle_root);
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
    start_index: u64,
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
) -> Option<FieldElement<E>> {
    if public_output.is_empty() {
        return Some(FieldElement::zero());
    }

    let bus_id = FieldElement::<E>::from(BusId::Commit as u64);
    let alpha_sq = alpha * alpha;

    // fingerprint_i = z - (BusId::Commit + (start_index + i)·α + value_i·α²).
    // `start_index` is the carried x254: 0 for a monolithic proof or the first
    // epoch, nonzero for a continuation epoch whose commits continue a prior one.
    let mut fingerprints: Vec<FieldElement<E>> = public_output
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let global_index = start_index + i as u64;
            let linear_combination = bus_id
                + (FieldElement::<E>::from(global_index) * alpha)
                + (FieldElement::<E>::from(value as u64) * alpha_sq);
            z - linear_combination
        })
        .collect();

    // Batch inversion: 1 inversion + O(3N) muls instead of N field inversions.
    // `Err` iff some fingerprint is zero (a collision) — treat as failure.
    FieldElement::inplace_batch_inverse(&mut fingerprints).ok()?;

    Some(
        fingerprints
            .iter()
            .fold(FieldElement::<E>::zero(), |acc, term| acc + term),
    )
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
    start_index: u64,
    transcript: &mut DefaultTranscript<E>,
) -> Option<FieldElement<E>> {
    let (z, alpha) = replay_transcript_phase_a(airs, proof, transcript);
    compute_commit_bus_offset(public_output_bytes, start_index, &z, &alpha)
}

/// Batched (unified-shard) analogue of [`replay_transcript_phase_a`]: appends
/// each preprocessed table's hardcoded precomputed root and the SINGLE batched
/// main MMCS root (Phase A of the linear transcript), then samples the shared
/// LogUp `(z, alpha)`. Mirrors `Prover::prove_rounds_1_to_3` Phase A + B.
pub(crate) fn replay_transcript_phase_a_batched(
    airs: &[&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>],
    proof: &BatchedMultiProof<F, E, ()>,
    transcript: &mut DefaultTranscript<E>,
) -> (FieldElement<E>, FieldElement<E>) {
    for air in airs.iter() {
        if air.is_preprocessed() {
            transcript.append_bytes(&air.precomputed_commitment());
        }
    }
    transcript.append_bytes(&proof.main_root);
    let z: FieldElement<E> = transcript.sample_field_element();
    let alpha: FieldElement<E> = transcript.sample_field_element();
    (z, alpha)
}

/// Batched analogue of [`compute_expected_commit_bus_balance`] for a
/// [`BatchedMultiProof`].
pub(crate) fn compute_expected_commit_bus_balance_batched(
    airs: &[&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>],
    proof: &BatchedMultiProof<F, E, ()>,
    public_output_bytes: &[u8],
    start_index: u64,
    transcript: &mut DefaultTranscript<E>,
) -> Option<FieldElement<E>> {
    let (z, alpha) = replay_transcript_phase_a_batched(airs, proof, transcript);
    compute_commit_bus_offset(public_output_bytes, start_index, &z, &alpha)
}

/// Bind the final cross-epoch GlobalMemory proof to the per-epoch proofs.
///
/// The final proof commits one local-to-global sub-table per epoch as its first
/// `N` tables, so `final_proof.proofs[i].lde_trace_main_merkle_root` is epoch
/// `i`'s L2G commitment. `epoch_l2g_roots[i]` is the same root as committed in
/// epoch `i`'s own proof. Equal roots prove the cross-epoch matching ran over
/// the very same L2G tables the epochs committed (shared commitments).
///
/// Called by `continuation::verify_continuation`; also exercised by the
/// local-to-global bus tests.
pub(crate) fn verify_l2g_commitment_binding(
    epoch_l2g_roots: &[Commitment],
    final_proof: &MultiProof<F, E, ()>,
) -> bool {
    final_proof.proofs.len() >= epoch_l2g_roots.len()
        && epoch_l2g_roots
            .iter()
            .enumerate()
            .all(|(i, root)| final_proof.proofs[i].lde_trace_main_merkle_root == *root)
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
    let traces = Traces::from_elf_and_logs(
        &program,
        &result.logs,
        &MaxRowsConfig::default(),
        private_inputs,
        #[cfg(feature = "disk-spill")]
        StorageMode::Ram,
    )?;
    Ok((
        traces.total_field_elements(),
        traces.total_auxiliary_field_elements(),
    ))
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

    #[cfg(feature = "disk-spill")]
    let storage_mode = {
        let lengths = count_table_lengths(&program, &result.logs, max_rows, private_inputs)?;
        auto_storage::decide(&lengths, proof_options.blowup_factor)
    };

    let mut traces = Traces::from_elf_and_logs(
        &program,
        &result.logs,
        max_rows,
        private_inputs,
        #[cfg(feature = "disk-spill")]
        storage_mode,
    )?;
    debug_assert_eq!(
        traces.public_output_bytes, result.return_values.memory_values,
        "public output diverged between executor view and trace reconstruction"
    );
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
        None,
        true,
        None,
        None,
        None,
    );

    #[cfg(feature = "instruments")]
    let air_elapsed = phase_start.elapsed();
    #[cfg(feature = "instruments")]
    let heap_after_air = stark::instruments::heap_bytes();

    let runtime_page_ranges = traces.runtime_page_ranges();

    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();

    // Bind the full statement (program, public output, table layout) into the
    // Fiat-Shamir transcript so every challenge depends on it.
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::Monolithic,
        elf_bytes,
        &traces.public_output_bytes,
        &table_counts,
        num_private_input_pages,
        &runtime_page_ranges,
    );

    // Phase 4: Prove (unified-shard batched MMCS + single FRI)
    let proof = Prover::multi_prove_batched(
        airs.air_trace_pairs(&mut traces),
        &mut transcript,
        #[cfg(feature = "disk-spill")]
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
        None,
        None,
    )
}

/// Verify a proof with caller-specified proof options.
///
/// The verifier enforces its own `proof_options` (security parameters),
/// ignoring the options embedded in the proof bundle. This prevents a
/// malicious prover from weakening the security level.
///
/// `decode_commitment` is an optional precomputed DECODE preprocessed
/// commitment. When `Some`, the supplied value is used directly and the
/// in-verifier FFT + Merkle build for the DECODE preprocessed columns is
/// skipped — useful for callers (e.g. the recursion guest) that embed the
/// commitment as a compile-time constant to avoid the in-VM recompute
/// cost. When `None`, the verifier computes the commitment from the ELF.
///
/// `page_commitments` is an optional list of precomputed ELF-data-page
/// preprocessed commitments, keyed by `page_base`. For each ELF data page
/// the verifier constructs, if a matching `(page_base, commitment)` pair is
/// supplied, the FFT + Merkle build for that page is skipped. Pages without
/// a match — including all zero-init pages — take the normal compute path
/// (zero-init pages hit a compile-time constant via
/// `page::zero_init_preprocessed_commitment`; ELF data pages recompute
/// from the ELF). When `None`, every ELF data page recomputes from scratch.
///
/// Trust model: both `decode_commitment` and `page_commitments`, when
/// supplied, must come from the caller's compiled binary (e.g. a
/// `const [u8; 32]` and a `const [(u64, [u8; 32])]`), never from prover-
/// supplied bytes. A wrong value is rejected, never silently accepted: it
/// either mismatches the prover's committed precomputed root (an explicit
/// verifier check) or yields diverging Fiat-Shamir challenges.
pub fn verify_with_options(
    vm_proof: &VmProof,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    decode_commitment: Option<Commitment>,
    page_commitments: Option<&[(u64, Commitment)]>,
) -> Result<bool, Error> {
    // Validate table_counts before constructing AIRs.
    // A malicious prover could set counts to 0, removing entire constraint sets.
    vm_proof.table_counts.validate()?;

    // Bound num_private_input_pages before allocating PageConfigs.
    // MAX_PRIVATE_INPUT_SIZE fits in ~257 pages of DEFAULT_PAGE_SIZE.
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
    // FIXED_TABLE_COUNT always-present tables, plus page tables.
    let expected_proof_count =
        vm_proof.table_counts.total() + FIXED_TABLE_COUNT + page_configs.len();
    if expected_proof_count != vm_proof.proof.per_table.len() {
        return Err(Error::InvalidTableCounts(format!(
            "table_counts total ({}) + {FIXED_TABLE_COUNT} fixed + {} pages = {}, but proof contains {} sub-proofs",
            vm_proof.table_counts.total(),
            page_configs.len(),
            expected_proof_count,
            vm_proof.proof.per_table.len(),
        )));
    }

    let airs = VmAirs::new(
        &program,
        proof_options,
        false,
        &page_configs,
        &vm_proof.table_counts,
        decode_commitment,
        true,
        None,
        page_commitments,
        None,
    );

    // Recompute the COMMIT output bus offset from VmProof.public_output.
    // If public_output was tampered, the recomputed offset won't match the
    // actual bus total in the proof, and multi_verify will reject.
    let air_refs = airs.air_refs();

    // Bind the statement into the verifier's transcript. A tampered statement
    // field makes this diverge from the prover's transcript state, so every
    // derived challenge differs and verification rejects.
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::Monolithic,
        elf_bytes,
        &vm_proof.public_output,
        &vm_proof.table_counts,
        vm_proof.num_private_input_pages,
        &vm_proof.runtime_page_ranges,
    );

    // Fork the post-absorb state: the replay helper advances through Phase A
    // independently of the multi_verify transcript, but both must start from
    // the same statement-bound state.
    let mut transcript_for_replay = transcript.clone();
    let expected_bus_balance = match compute_expected_commit_bus_balance_batched(
        &air_refs,
        &vm_proof.proof,
        &vm_proof.public_output,
        // Monolithic proof: commits are indexed from 0.
        0,
        &mut transcript_for_replay,
    ) {
        Some(balance) => balance,
        None => return Ok(false),
    };

    Ok(Verifier::batched_multi_verify(
        &air_refs,
        &vm_proof.proof,
        &mut transcript,
        &expected_bus_balance,
    ))
}

/// Prove and verify in one call (convenience).
pub fn prove_and_verify(elf_bytes: &[u8]) -> Result<bool, Error> {
    let vm_proof = prove(elf_bytes)?;
    verify(&vm_proof, elf_bytes)
}

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

#![cfg_attr(not(feature = "std"), no_std)]
// In guest builds (`prove` feature off) the prove-side helpers — trace generators,
// executor-typed imports, internal Operation structs, etc. — are unreferenced.
// They're real code, used by the host build, and there's nothing to fix there.
// Silence the resulting dead_code / unused_imports noise in the guest build only.
#![cfg_attr(not(feature = "prove"), allow(dead_code, unused_imports))]

extern crate alloc;

#[cfg(feature = "disk-spill")]
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
pub mod vkey;

pub use vkey::VmVerifyingKey;

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use executor::elf::Elf;
#[cfg(feature = "prove")]
use executor::vm::execution::Executor;
use math::field::element::FieldElement;
#[cfg(feature = "prove")]
use stark::prover::{IsStarkProver, Prover};
#[cfg(feature = "disk-spill")]
use stark::storage_mode::StorageMode;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

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
    E, F, VmAir, create_bitwise_air, create_branch_air, create_commit_air, create_cpu_air,
    create_decode_air, create_dvrm_air, create_fp3_mul_air, create_halt_air, create_keccak_air,
    create_keccak_rc_air,
    create_keccak_rnd_air, create_load_air, create_lt_air, create_memw_air,
    create_memw_aligned_air, create_memw_register_air, create_mul_air, create_page_air,
    create_register_air, create_shift_air,
};

pub use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::stark::MultiProof;

/// A run-length encoded range of contiguous zero-initialized 4KB pages.
///
/// Represents `count` contiguous pages starting at `base`, used for
/// runtime-allocated memory (stack, heap) not covered by ELF segments.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct RuntimePageRange {
    /// Base address of the first page (4KB-aligned).
    pub base: u64,
    /// Number of contiguous 4KB pages starting at `base`.
    pub count: u64,
}

/// Number of chunks for each split table.
/// The verifier needs this to reconstruct matching AIRs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
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
/// The private-input bundle the recursion verifier guest consumes: an inner
/// proof plus everything needed to verify it (inner ELF, the inner prover's
/// options, and the host-derived verifying key).
///
/// Grouping these in one rkyv-archivable struct lets the guest `rkyv::access`
/// the whole blob and read each field straight from the input buffer with no
/// deserialization pass — the previous `postcard::from_bytes` of the same tuple
/// was ~23% of the verifier's RISC-V cycles.
#[cfg(feature = "rkyv")]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct RecursionInput {
    pub vm_proof: VmProof,
    pub inner_elf: alloc::vec::Vec<u8>,
    pub options: ProofOptions,
    pub vkey: VmVerifyingKey,
}

// ============================================================================
// Recursion-input wire format: aligning magic prefix + rkyv archive
// ============================================================================
//
// rkyv reads the archive in place and issues naturally-aligned loads (the
// archived field element is `rend::u64_le`, alignment 8; we play it safe and
// require 16). The lambda-vm executor *traps* unaligned doubleword loads, so the
// archive's first byte must sit at a 16-aligned guest address.
//
// The executor maps the private input as `[u32 len][payload...]` with the
// payload starting at `PRIVATE_INPUT_START_INDEX + 4`. That base is 4-aligned,
// not 16. We prepend a fixed prefix to the payload so the archive that follows
// lands on a 16-aligned address, and make the prefix a magic + version so the
// guest can reject a wrong-format/version blob *before* the unsafe access.
//
// Prefix length is chosen so `(PRIVATE_INPUT_START_INDEX + 4) + PREFIX_LEN` is a
// multiple of 16:
//   (16 - ((0xFF000004) mod 16)) mod 16 = (16 - 4) mod 16 = 12.

/// 4-byte magic identifying a lambda-vm recursion input blob ("LVMR").
#[cfg(feature = "rkyv")]
pub const RECURSION_INPUT_MAGIC: [u8; 4] = *b"LVMR";

/// Wire-format version of the recursion input blob.
#[cfg(feature = "rkyv")]
pub const RECURSION_INPUT_VERSION: u32 = 1;

/// Required alignment (bytes) of the archive's first byte in guest memory.
#[cfg(feature = "rkyv")]
pub const RECURSION_INPUT_ALIGN: usize = 16;

/// Aligning prefix length: `magic(4) + version(4) + reserved(4) = 12` bytes,
/// chosen so the archive starts 16-aligned given the executor's
/// `PRIVATE_INPUT_START_INDEX + 4` payload base. Asserted below.
#[cfg(feature = "rkyv")]
pub const RECURSION_INPUT_PREFIX_LEN: usize = 12;

#[cfg(feature = "rkyv")]
const _: () = {
    let payload_base = (executor::constants::PRIVATE_INPUT_START_INDEX as usize) + 4;
    let pad =
        (RECURSION_INPUT_ALIGN - (payload_base % RECURSION_INPUT_ALIGN)) % RECURSION_INPUT_ALIGN;
    assert!(
        RECURSION_INPUT_PREFIX_LEN == pad,
        "prefix length must align the archive to RECURSION_INPUT_ALIGN given the private-input payload base",
    );
    assert!(
        (payload_base + RECURSION_INPUT_PREFIX_LEN) % RECURSION_INPUT_ALIGN == 0,
        "archive must start at a RECURSION_INPUT_ALIGN-aligned guest address",
    );
};

/// Encode a [`RecursionInput`] into the on-wire blob: a 12-byte
/// `magic + version + reserved` prefix followed by the rkyv archive. The prefix
/// both aligns the archive (so the guest's in-place reads don't trap) and tags
/// the format/version so the guest can validate before the unsafe access.
#[cfg(all(feature = "rkyv", feature = "prove"))]
pub fn encode_recursion_input(input: &RecursionInput) -> Result<alloc::vec::Vec<u8>, Error> {
    use rkyv::rancor::Error as RkyvError;
    let archive = rkyv::to_bytes::<RkyvError>(input)
        .map_err(|e| Error::Execution(format!("rkyv encode failed: {e}")))?;
    let mut blob = alloc::vec::Vec::with_capacity(RECURSION_INPUT_PREFIX_LEN + archive.len());
    blob.extend_from_slice(&RECURSION_INPUT_MAGIC);
    blob.extend_from_slice(&RECURSION_INPUT_VERSION.to_le_bytes());
    blob.extend_from_slice(&[0u8; 4]); // reserved
    debug_assert_eq!(blob.len(), RECURSION_INPUT_PREFIX_LEN);
    blob.extend_from_slice(&archive);
    Ok(blob)
}

/// Validate the wire prefix and return the archive bytes (zero-copy slice).
/// Returns `None` if the magic or version doesn't match — the caller should
/// halt cleanly rather than proceed into an `access_unchecked`.
#[cfg(feature = "rkyv")]
pub fn recursion_archive_bytes(blob: &[u8]) -> Option<&[u8]> {
    if blob.len() < RECURSION_INPUT_PREFIX_LEN {
        return None;
    }
    if blob[0..4] != RECURSION_INPUT_MAGIC {
        return None;
    }
    let version = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]);
    if version != RECURSION_INPUT_VERSION {
        return None;
    }
    Some(&blob[RECURSION_INPUT_PREFIX_LEN..])
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
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

impl core::error::Error for Error {}

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
    pub fp3_mul: VmAir,
    pub register: VmAir,
    pub pages: Vec<VmAir>,
    pub memw_registers: Vec<VmAir>,
}

impl VmAirs {
    /// Build `(air, trace, public_inputs)` triples for [`Prover::multi_prove`].
    #[cfg(feature = "prove")]
    pub fn air_trace_pairs<'a>(&'a self, traces: &'a mut Traces) -> Vec<AirTracePair<'a>> {
        let mut pairs: Vec<AirTracePair<'a>> = vec![
            (&self.bitwise, &mut traces.bitwise, &()),
            (&self.decode, &mut traces.decode, &()),
            (&self.halt, &mut traces.halt, &()),
            (&self.commit, &mut traces.commit, &()),
            (&self.keccak, &mut traces.keccak, &()),
            (&self.keccak_rnd, &mut traces.keccak_rnd, &()),
            (&self.keccak_rc, &mut traces.keccak_rc, &()),
            (&self.fp3_mul, &mut traces.fp3_mul, &()),
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
            &self.keccak,
            &self.keccak_rnd,
            &self.keccak_rc,
            &self.fp3_mul,
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
        Self::new_with_vkey(
            elf,
            proof_options,
            minimal_bitwise,
            page_configs,
            table_counts,
            None,
        )
    }

    /// Same as [`Self::new`] but accepts a precomputed [`VmVerifyingKey`].
    /// When `vkey` is `Some`, the bitwise preprocessed commitment is taken
    /// from it instead of being recomputed from `proof_options` — that
    /// recomputation is ~87% of verifier cycles inside the recursion guest.
    pub fn new_with_vkey(
        elf: &Elf,
        proof_options: &ProofOptions,
        minimal_bitwise: bool,
        page_configs: &[crate::tables::page::PageConfig],
        table_counts: &TableCounts,
        vkey: Option<&VmVerifyingKey>,
    ) -> Self {
        let cpus: Vec<_> = (0..table_counts.cpu)
            .map(|i| create_cpu_air(proof_options).with_name(&format!("CPU[{}]", i)))
            .collect();
        let bitwise = if minimal_bitwise {
            create_bitwise_air(proof_options)
        } else {
            let commitment = match vkey {
                Some(vk) => vk.bitwise,
                None => bitwise::preprocessed_commitment(proof_options),
            };
            create_bitwise_air(proof_options)
                .with_preprocessed(commitment, bitwise::NUM_PRECOMPUTED_COLS)
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
        let decode_commitment = vkey.map(|vk| vk.decode).unwrap_or_else(|| {
            decode::commitment_from_elf(elf, proof_options)
                .expect("Failed to compute decode commitment")
        });
        let decode = create_decode_air(proof_options)
            .with_preprocessed(decode_commitment, decode::NUM_PRECOMPUTED_COLS);
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
        let keccak_rc_commitment = vkey
            .map(|vk| vk.keccak_rc)
            .unwrap_or_else(|| tables::keccak_rc::preprocessed_commitment(proof_options));
        let fp3_mul = create_fp3_mul_air(proof_options);
        let keccak_rc = create_keccak_rc_air(proof_options).with_preprocessed(
            keccak_rc_commitment,
            tables::keccak_rc::NUM_PRECOMPUTED_COLS,
        );
        let register_commitment = vkey
            .map(|vk| vk.register)
            .unwrap_or_else(|| register::preprocessed_commitment(proof_options, elf.entry_point));
        let register = create_register_air(proof_options)
            .with_preprocessed(register_commitment, register::NUM_PREPROCESSED_COLS);
        let pages: Vec<_> = page_configs
            .iter()
            .enumerate()
            .map(|(i, config)| {
                if config.is_private_input {
                    // Private-input pages: all columns are main trace (not preprocessed).
                    // The verifier doesn't see the init values; correctness is enforced
                    // by the memory bus constraints.
                    create_page_air(proof_options, config.page_base)
                } else {
                    // ELF and zero-init pages: OFFSET + INIT are preprocessed.
                    // Prefer the vkey-supplied commitment when present (cached on host,
                    // saves the FFT + Merkle pipeline inside the verifier). If the vkey
                    // is absent or shorter than expected, fall back to recomputing — the
                    // length mismatch path is defensive only; Fiat-Shamir would catch a
                    // genuine mismatch downstream anyway.
                    let commitment =
                        vkey.and_then(|vk| vk.pages.get(i))
                            .copied()
                            .unwrap_or_else(|| {
                                page::precomputed_commitment_cached(config, proof_options)
                            });
                    create_page_air(proof_options, config.page_base)
                        .with_preprocessed(commitment, page::NUM_PREPROCESSED_COLS)
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
            keccak,
            keccak_rnd,
            keccak_rc,
            fp3_mul,
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
pub(crate) fn replay_transcript_phase_a<'p, P>(
    airs: &[&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>],
    num_proofs: usize,
    get_proof: impl Fn(usize) -> P,
) -> (FieldElement<E>, FieldElement<E>)
where
    P: stark::proof::zerocopy::StarkProofRef<'p, F, E, ()>,
{
    debug_assert_eq!(airs.len(), num_proofs);
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    for (idx, air) in airs.iter().enumerate() {
        let proof = get_proof(idx);
        if air.is_preprocessed() {
            transcript.append_bytes(&air.precomputed_commitment());
            transcript.append_bytes(proof.lde_trace_main_merkle_root());
        } else {
            transcript.append_bytes(proof.lde_trace_main_merkle_root());
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
pub(crate) fn compute_expected_commit_bus_balance<'p, P>(
    airs: &[&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>],
    num_proofs: usize,
    get_proof: impl Fn(usize) -> P,
    public_output_bytes: &[u8],
) -> Option<FieldElement<E>>
where
    P: stark::proof::zerocopy::StarkProofRef<'p, F, E, ()>,
{
    let (z, alpha) = replay_transcript_phase_a(airs, num_proofs, get_proof);
    compute_commit_bus_offset(public_output_bytes, &z, &alpha)
}

/// Owned-proof convenience wrapper over [`compute_expected_commit_bus_balance`].
pub(crate) fn compute_expected_commit_bus_balance_owned(
    airs: &[&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>],
    proof: &MultiProof<F, E, ()>,
    public_output_bytes: &[u8],
) -> Option<FieldElement<E>> {
    compute_expected_commit_bus_balance(
        airs,
        proof.proofs.len(),
        |i| &proof.proofs[i],
        public_output_bytes,
    )
}

// =============================================================================
// Public API: Prove / Verify
// =============================================================================

/// Prove an ELF binary execution. Returns a serializable proof bundle.
#[cfg(feature = "prove")]
pub fn prove(elf_bytes: &[u8]) -> Result<VmProof, Error> {
    prove_with_inputs(elf_bytes, &[])
}

/// Prove an ELF binary execution with private inputs. Returns a serializable proof bundle.
#[cfg(feature = "prove")]
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
#[cfg(feature = "prove")]
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

/// Build the trace tables for an ELF + input and return a per-table size
/// breakdown (rows, main columns, aux columns) without running the STARK proof.
///
/// Summing `main_elements()` / `aux_elements()` over the result reproduces the
/// totals from [`count_elements`] exactly. Intended for profiling: it shows
/// which tables dominate the trace, and therefore proving cost, for a given
/// program + input.
///
/// Gated on `prove` like [`count_elements`]: it builds traces via the
/// executor + `Traces::from_elf_and_logs`, which are only compiled with that
/// feature (so the no_std guest build of the prover stays lean).
#[cfg(feature = "prove")]
pub fn table_report(
    elf_bytes: &[u8],
    private_inputs: &[u8],
) -> Result<Vec<crate::tables::trace_builder::TableReport>, Error> {
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
    Ok(traces.table_reports())
}

/// Prove an ELF binary execution with custom proof options and max rows config.
#[cfg(feature = "prove")]
pub fn prove_with_options(
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    max_rows: &MaxRowsConfig,
) -> Result<VmProof, Error> {
    prove_with_options_and_inputs(elf_bytes, &[], proof_options, max_rows)
}

/// Prove an ELF binary execution with custom proof options, max rows config,
/// and explicit private inputs.
#[cfg(feature = "prove")]
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
    );

    #[cfg(feature = "instruments")]
    let air_elapsed = phase_start.elapsed();
    #[cfg(feature = "instruments")]
    let heap_after_air = stark::instruments::heap_bytes();

    let runtime_page_ranges = traces.runtime_page_ranges();

    // Phase 4: Prove (multi_prove)
    let proof = Prover::multi_prove(
        airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
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
    verify_with_options_with_vkey(vm_proof, elf_bytes, proof_options, None)
}

/// Verify a recursion-input blob produced by `rkyv::to_bytes::<RecursionInput>`.
///
/// `rkyv::access` validates and views the blob in place (no deserialization),
/// then we materialize the proof/options/vkey via rkyv's structural
/// deserialize — a pointer-following + memcpy traversal with no format parsing,
/// which replaces the postcard varint parse that dominated verifier cycles.
///
/// The `elf` is read directly from the archived bytes (`&[u8]`, zero-copy).
#[cfg(feature = "rkyv")]
pub fn verify_recursion_blob(blob: &[u8]) -> Result<bool, Error> {
    use rkyv::rancor::Error as RkyvError;

    // Validate + strip the aligning magic/version prefix. The returned slice
    // starts at the 16-aligned archive base (the prefix exists precisely so the
    // archive lands aligned at `PRIVATE_INPUT_START + 4 + PREFIX_LEN`), so the
    // in-place doubleword loads below do not trap.
    let archive_bytes = recursion_archive_bytes(blob).ok_or_else(|| {
        Error::Execution(alloc::string::String::from(
            "recursion blob: bad magic or version",
        ))
    })?;

    // SAFETY: `archive_bytes` is produced by our own `encode_recursion_input`
    // in the trusted host path and is 16-aligned (prefix-aligned). A corrupted
    // blob can only cause verification to fail (the proof is checked
    // cryptographically), not unsoundness here.
    let archived = unsafe { rkyv::access_unchecked::<ArchivedRecursionInput>(archive_bytes) };

    // The big STARK proof (the nested FieldElement Vecs) is read IN PLACE from
    // the archived buffer — never deserialized to owned, which would trigger a
    // catastrophic allocation storm in the guest's bump allocator. Only the
    // small metadata is materialized: deserializing these is a handful of tiny
    // allocations, not the per-field-element storm.
    let options: ProofOptions = rkyv::deserialize::<ProofOptions, RkyvError>(&archived.options)
        .map_err(|e| Error::Execution(format!("rkyv deserialize options failed: {e}")))?;
    let vkey: VmVerifyingKey = rkyv::deserialize::<VmVerifyingKey, RkyvError>(&archived.vkey)
        .map_err(|e| Error::Execution(format!("rkyv deserialize vkey failed: {e}")))?;
    let table_counts: TableCounts =
        rkyv::deserialize::<TableCounts, RkyvError>(&archived.vm_proof.table_counts)
            .map_err(|e| Error::Execution(format!("rkyv deserialize table_counts failed: {e}")))?;
    let runtime_page_ranges: alloc::vec::Vec<RuntimePageRange> =
        rkyv::deserialize::<alloc::vec::Vec<RuntimePageRange>, RkyvError>(
            &archived.vm_proof.runtime_page_ranges,
        )
        .map_err(|e| Error::Execution(format!("rkyv deserialize page ranges failed: {e}")))?;
    // Bytes read straight from the archived buffer (zero-copy).
    let inner_elf: &[u8] = archived.inner_elf.as_ref();
    let public_output: &[u8] = archived.vm_proof.public_output.as_ref();
    let num_private_input_pages = archived.vm_proof.num_private_input_pages.to_native() as usize;

    // The archived MultiProof, read in place.
    let archived_proofs = archived.vm_proof.proof.proofs.as_slice();

    verify_archived_with_vkey(
        archived_proofs,
        &table_counts,
        &runtime_page_ranges,
        num_private_input_pages,
        public_output,
        inner_elf,
        &options,
        &vkey,
    )
}

/// Verify a STARK proof whose sub-proofs are read in place from an rkyv-archived
/// buffer (zero-copy: no per-field-element deserialization or allocation).
/// Mirrors [`verify_with_options_with_vkey`] but takes the already-extracted
/// metadata plus a slice of archived sub-proofs.
#[cfg(feature = "rkyv")]
#[allow(clippy::too_many_arguments)]
fn verify_archived_with_vkey(
    archived_proofs: &[<stark::proof::stark::StarkProof<F, E, ()> as rkyv::Archive>::Archived],
    table_counts: &TableCounts,
    runtime_page_ranges: &[RuntimePageRange],
    num_private_input_pages: usize,
    public_output: &[u8],
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    vkey: &VmVerifyingKey,
) -> Result<bool, Error> {
    table_counts.validate()?;

    {
        use crate::tables::page::DEFAULT_PAGE_SIZE;
        use executor::constants::MAX_PRIVATE_INPUT_SIZE;
        let max_pages = (MAX_PRIVATE_INPUT_SIZE as usize + 4).div_ceil(DEFAULT_PAGE_SIZE) + 1;
        if num_private_input_pages > max_pages {
            return Err(Error::InvalidTableCounts(format!(
                "num_private_input_pages ({num_private_input_pages}) exceeds max ({max_pages})",
            )));
        }
    }

    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let page_configs = Traces::page_configs_from_elf_and_runtime(
        &program,
        runtime_page_ranges,
        num_private_input_pages,
    );

    let expected_proof_count = table_counts.total() + 9 + page_configs.len();
    if expected_proof_count != archived_proofs.len() {
        return Err(Error::InvalidTableCounts(format!(
            "table_counts total ({}) + 9 fixed + {} pages = {expected_proof_count}, but proof contains {} sub-proofs",
            table_counts.total(),
            page_configs.len(),
            archived_proofs.len(),
        )));
    }

    let airs = VmAirs::new_with_vkey(
        &program,
        proof_options,
        false,
        &page_configs,
        table_counts,
        Some(vkey),
    );

    // In the recursion guest the process verifies a single proof and then halts,
    // so the heap is reclaimed wholesale on exit — running `drop(VmAirs)` walks
    // ~9.3k tiny Vec/Box deallocations (the per-interaction `Vec<BusValue>`,
    // `Vec<LinearTerm>`, and boxed constraints) for nothing (~7% of guest verify
    // cycles in the profile). Suppress teardown so those deallocations never run.
    // `ManuallyDrop` adds no allocation (unlike `Box::leak`); the AIRs simply live
    // for the rest of the (single-shot) process. Guarded to the guest target only;
    // the host (long-lived prover process) keeps normal drop semantics so
    // verifying in a loop does not leak.
    #[cfg(target_arch = "riscv64")]
    let airs = core::mem::ManuallyDrop::new(airs);

    let air_refs = airs.air_refs();
    let get_proof = |i: usize| &archived_proofs[i];
    let expected_bus_balance = match compute_expected_commit_bus_balance(
        &air_refs,
        archived_proofs.len(),
        get_proof,
        public_output,
    ) {
        Some(balance) => balance,
        None => return Ok(false),
    };

    Ok(Verifier::multi_verify(
        &air_refs,
        archived_proofs.len(),
        get_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    ))
}

/// Same as [`verify_with_options`] but accepts a precomputed
/// [`VmVerifyingKey`]. When `vkey` is `Some`, the bitwise preprocessed
/// commitment is taken from it instead of being recomputed inside
/// `VmAirs::new`. A tampered vkey is caught by Fiat-Shamir: the verifier
/// feeds the supplied commitment into the transcript, derives different
/// challenges from what the prover used, and the openings stop matching.
pub fn verify_with_options_with_vkey(
    vm_proof: &VmProof,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    vkey: Option<&VmVerifyingKey>,
) -> Result<bool, Error> {
    // Validate table_counts before constructing AIRs.
    // A malicious prover could set counts to 0, removing entire constraint sets.
    vm_proof.table_counts.validate()?;

    // Bound num_private_input_pages before allocating PageConfigs.
    // MAX_PRIVATE_INPUT_SIZE fits in ~26 pages of DEFAULT_PAGE_SIZE.
    {
        use crate::tables::page::DEFAULT_PAGE_SIZE;
        use executor::constants::MAX_PRIVATE_INPUT_SIZE;
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
    // Fixed tables (bitwise, decode, halt, commit, keccak, keccak_rnd, keccak_rc, fp3_mul, register) = 9, plus page tables.
    let expected_proof_count = vm_proof.table_counts.total() + 9 + page_configs.len();
    if expected_proof_count != vm_proof.proof.proofs.len() {
        return Err(Error::InvalidTableCounts(format!(
            "table_counts total ({}) + 9 fixed + {} pages = {}, but proof contains {} sub-proofs",
            vm_proof.table_counts.total(),
            page_configs.len(),
            expected_proof_count,
            vm_proof.proof.proofs.len(),
        )));
    }

    let airs = VmAirs::new_with_vkey(
        &program,
        proof_options,
        false,
        &page_configs,
        &vm_proof.table_counts,
        vkey,
    );

    // Recompute the COMMIT output bus offset from VmProof.public_output.
    // If public_output was tampered, the recomputed offset won't match the
    // actual bus total in the proof, and multi_verify will reject.
    let air_refs = airs.air_refs();
    let expected_bus_balance = match compute_expected_commit_bus_balance(
        &air_refs,
        vm_proof.proof.proofs.len(),
        |i| &vm_proof.proof.proofs[i],
        &vm_proof.public_output,
    ) {
        Some(balance) => balance,
        None => return Ok(false),
    };

    Ok(Verifier::multi_verify(
        &air_refs,
        vm_proof.proof.proofs.len(),
        |i| &vm_proof.proof.proofs[i],
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    ))
}

/// Prove and verify in one call (convenience).
#[cfg(feature = "prove")]
pub fn prove_and_verify(elf_bytes: &[u8]) -> Result<bool, Error> {
    let vm_proof = prove(elf_bytes)?;
    verify(&vm_proof, elf_bytes)
}

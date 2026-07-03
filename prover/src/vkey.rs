//! Verifying key for the lambda-vm STARK verifier.
//!
//! Caches preprocessed-table Merkle commitments that the verifier would
//! otherwise recompute on every call. Mirrors the SP1 `MachineVerifyingKey`
//! pattern (preprocessed commitments derived once at setup, never recomputed
//! per-proof) and the prover-side companion in
//! <https://github.com/yetanotherco/lambda_vm/pull/282> (which caches the
//! same data on the prover side).
//!
//! ## Current scope
//!
//! All five preprocessed tables — BITWISE, DECODE, REGISTER, KECCAK_RC, and
//! every non-private-input PAGE — are cached here, together with the
//! [`ProofOptions`] the commitments were derived under. `VmAirs::new_with_vkey`
//! prefers the vkey-supplied commitment over recomputing when a vkey is
//! provided. The `version` field exists so a vkey serialized against an
//! older layout produces a different `compute_digest()` and stops
//! validating.
//!
//! ## Security
//!
//! The vkey is **trusted input**. Fiat-Shamir only detects a vkey that is
//! inconsistent with the proof (post-hoc tampering); a coordinated prover
//! can commit to a forged preprocessed table from the start and supply a
//! matching vkey, and the transcript stays self-consistent. The binding
//! that makes recursion sound is `compute_digest()`:
//!
//! - The prover stamps it into `VmProof::vk_digest` and binds it into the
//!   Fiat-Shamir statement; the verifier recomputes it from its own vkey
//!   and rejects on mismatch before any STARK work.
//! - The recursion guest commits `vk_digest ‖ inner public output`, so the
//!   *outer* verifier can check which vkey was used in-guest against a
//!   digest derived from the trusted inner ELF. Without that outer check
//!   the guest's result says nothing — every guest input is prover-supplied.
//!
//! The digest covers the embedded [`ProofOptions`]: query count and
//! grinding factor affect soundness but no commitment, so nothing else
//! would pin them.

use executor::elf::Elf;
use sha3::{Digest, Keccak256};
use stark::config::Commitment;
use stark::proof::options::ProofOptions;

use crate::tables::bitwise;
use crate::tables::decode;
use crate::tables::keccak_rc;
use crate::tables::page::{self, PageConfig};
use crate::tables::register;

/// Current `VmVerifyingKey` layout version. Bump whenever fields are added,
/// removed, or reordered so that vkeys serialized against an older layout
/// produce a different `compute_digest()` and stop validating.
pub const VKEY_VERSION: u32 = 4;

/// Placeholder commitment stored in [`VmVerifyingKey::pages`] for
/// private-input page slots, where there is no preprocessed commitment to
/// cache. The verifier never reads these slots (private-input pages have no
/// `with_preprocessed(...)` call in `VmAirs::new`).
const PRIVATE_INPUT_PAGE_PLACEHOLDER: Commitment = [0u8; 32];

/// Cached preprocessed-table commitments the verifier would otherwise
/// recompute on every call.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct VmVerifyingKey {
    /// Layout version. See [`VKEY_VERSION`].
    pub version: u32,
    /// The options every commitment below was derived under. In the digest
    /// because query count and grinding factor affect soundness but no
    /// commitment.
    pub options: ProofOptions,
    /// Merkle root over the LDE of the bitwise preprocessed columns.
    /// Program-independent; depends only on `ProofOptions`.
    pub bitwise: Commitment,
    /// Merkle root over the LDE of the decode preprocessed columns.
    /// Program-dependent: derived from the inner ELF's instruction stream.
    pub decode: Commitment,
    /// Merkle root over the LDE of the register preprocessed columns.
    /// Program-dependent via the ELF's entry point.
    pub register: Commitment,
    /// Merkle root over the LDE of the keccak round-constants preprocessed
    /// columns. Program-independent; depends only on `ProofOptions`.
    pub keccak_rc: Commitment,
    /// Per-page preprocessed Merkle roots, indexed parallel to the
    /// `page_configs` slice the verifier reconstructs from the proof via
    /// [`crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime`].
    /// Private-input slots hold a zero placeholder and are never read by the
    /// verifier — they exist only to keep the index aligned with
    /// `page_configs`, which interleaves preprocessed and private-input pages.
    /// Prover (`traces.page_configs`) and verifier
    /// (`page_configs_from_elf_and_runtime`) must derive the same page order
    /// or the digests diverge.
    pub pages: Vec<Commitment>,
}

impl VmVerifyingKey {
    /// Derive the verifying key on the host.
    ///
    /// `elf` is read to derive the program-dependent commitments (DECODE
    /// from the instruction stream, REGISTER from `elf.entry_point`).
    ///
    /// `page_configs` must match exactly what the verifier will reconstruct
    /// from the proof — i.e. the output of
    /// `Traces::page_configs_from_elf_and_runtime(elf, runtime_page_ranges,
    /// num_private_input_pages)`. The host can call that helper with the
    /// values it already has after producing the inner proof.
    pub fn from_elf_and_options(
        elf: &Elf,
        options: &ProofOptions,
        register_init: Option<&[u32]>,
        page_configs: &[PageConfig],
    ) -> Self {
        let pages = page_configs
            .iter()
            .map(|config| {
                if config.is_private_input {
                    PRIVATE_INPUT_PAGE_PLACEHOLDER
                } else {
                    page::precomputed_commitment_cached(config, options)
                }
            })
            .collect();
        let register_init = register_init
            .map(<[u32]>::to_vec)
            .unwrap_or_else(|| register::register_init_from_entry_point(elf.entry_point));
        Self {
            version: VKEY_VERSION,
            options: options.clone(),
            bitwise: bitwise::preprocessed_commitment(options),
            decode: decode::commitment_from_elf(elf, options)
                .expect("decode commitment must compute"),
            register: register::preprocessed_commitment(options, &register_init),
            keccak_rc: keccak_rc::preprocessed_commitment(options),
            pages,
        }
    }

    /// Keccak256 fingerprint of a canonical, framework-free encoding of the
    /// vkey: every field is absorbed fixed-width (integers as little-endian
    /// u64/u8, commitments raw, `pages` length-prefixed), so the encoding is
    /// injective and stable as long as the field layout (and [`VKEY_VERSION`])
    /// does not change. The exhaustive destructure makes any field added to
    /// `VmVerifyingKey` a compile error here — the signal to extend the
    /// absorption below and bump [`VKEY_VERSION`].
    pub fn compute_digest(&self) -> [u8; 32] {
        let Self {
            version,
            options:
                ProofOptions {
                    blowup_factor,
                    fri_number_of_queries,
                    coset_offset,
                    grinding_factor,
                },
            bitwise,
            decode,
            register,
            keccak_rc,
            pages,
        } = self;
        let mut hasher = Keccak256::new();
        hasher.update(version.to_le_bytes());
        hasher.update([*blowup_factor]);
        hasher.update((*fri_number_of_queries as u64).to_le_bytes());
        hasher.update(coset_offset.to_le_bytes());
        hasher.update([*grinding_factor]);
        hasher.update(bitwise);
        hasher.update(decode);
        hasher.update(register);
        hasher.update(keccak_rc);
        hasher.update((pages.len() as u64).to_le_bytes());
        for page in pages {
            hasher.update(page);
        }
        hasher.finalize().into()
    }
}
